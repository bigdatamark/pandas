# [MODULE DOC]
# Low-level building blocks of the pandas BlockManager. Provides BlockPlacement
# (the set of manager column positions a Block occupies, stored as a bounded slice
# and/or an intp array, converted lazily between the two), slice utilities that
# canonicalize, measure and compose bounded slices, routines that group block
# numbers into runs (used for take/concat), the Cython base classes Block and
# BlockManager (fast construction, pickling, row slicing), and BlockValuesRefs, a
# weak-reference list used by Copy-on-Write to detect shared array data. Under
# free-threaded Python, several accessors use critical sections / per-object locks.
from collections import defaultdict

cimport cython
from cpython.object cimport PyObject
from cpython.pyport cimport PY_SSIZE_T_MAX
from cpython.slice cimport PySlice_GetIndicesEx
from cpython.weakref cimport PyWeakref_NewRef
from cython cimport Py_ssize_t

import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_INTP,
    int64_t,
    intp_t,
    ndarray,
)

cnp.import_array()

from pandas._libs.algos import ensure_int64

from pandas._libs.util cimport (
    is_array,
    is_integer_object,
)

include "free_threading_config.pxi"

IF CYTHON_COMPATIBLE_WITH_FREE_THREADING:
    from cpython.ref cimport Py_DECREF
    from cpython.weakref cimport PyWeakref_GetRef
ELSE:
    from cpython.weakref cimport PyWeakref_GetObject


cdef extern from "Python.h":
    PyObject* Py_None


@cython.final
@cython.freelist(32)
cdef class BlockPlacement:
    # [CLASS DOC]
    # Immutable set of non-negative integer positions (the manager locations of a
    # Block). Stored as a canonical bounded slice (_as_slice) and/or a 1-D intp array
    # (_as_array); _has_slice/_has_array flag which representations are computed.
    # A missing representation is derived lazily: array from slice via arange; slice
    # from array if the values form an arithmetic progression of non-negative values
    # (otherwise the slice is None, i.e. not slice-like). Uses a freelist of 32.
    cdef:
        slice _as_slice
        ndarray _as_array  # Note: this still allows `None`; will be intp_t
        bint _has_slice, _has_array, _is_known_slice_like

    def __cinit__(self, val):
        # [FUNCTION DOC]
        # Construct from an integer, slice or array-like.
        # Inputs: val - integer i (becomes slice(i, i+1, 1)); a slice (canonicalized; an
        # empty canonical slice is stored as an empty intp array instead); or any
        # array-like (converted to a writeable intp array, copying unless it already is
        #   a
        # writeable intp ndarray).
        # Outputs: none.
        # Exceptions: ValueError for zero-step or unbounded slices (see slice_canonize);
        # conversion errors from numpy for other inputs.
        # Side effects: none on val beyond possibly aliasing it (no copy when already a
        # writeable intp array; caller must ensure 1-D).
        cdef:
            slice slc

        self._as_slice = None
        self._as_array = None
        self._has_slice = False
        self._has_array = False

        if is_integer_object(val):
            slc = slice(val, val + 1, 1)
            self._as_slice = slc
            self._has_slice = True
        elif isinstance(val, slice):
            slc = slice_canonize(val)

            if slc.start != slc.stop:
                self._as_slice = slc
                self._has_slice = True
            else:
                arr = np.empty(0, dtype=np.intp)
                self._as_array = arr
                self._has_array = True
        else:
            # Cython memoryview interface requires ndarray to be writeable.
            if (
                not is_array(val)
                or not cnp.PyArray_ISWRITEABLE(val)
                or (<ndarray>val).descr.type_num != cnp.NPY_INTP
            ):
                arr = np.require(val, dtype=np.intp, requirements="W")
            else:
                arr = val
            # Caller is responsible for ensuring arr.ndim == 1
            self._as_array = arr
            self._has_array = True

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # String form "BlockPlacement(<slice or array>)", preferring the slice form when
        # the placement is slice-like.
        # Side effects: may compute and cache the slice representation.
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            v = self._as_slice
        else:
            v = self._as_array

        return f"{type(self).__name__}({v})"

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Same as __str__.
        return str(self)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Number of positions.
        # Outputs: slice length (computed from the bounded slice) when slice-like, else
        # the array length.
        # Side effects: may cache the slice representation.
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            return slice_len(s)
        else:
            return len(self._as_array)

    def __iter__(self):
        # [FUNCTION DOC]
        # Iterate over positions.
        # Outputs: a range iterator when slice-like, else an iterator over the array.
        cdef:
            slice s = self._ensure_has_slice()
            Py_ssize_t start, stop, step, _

        if s is not None:
            start, stop, step, _ = slice_get_indices_ex(s)
            return iter(range(start, stop, step))
        else:
            return iter(self._as_array)

    @property
    def as_slice(self) -> slice:
        # [FUNCTION DOC]
        # Property: the slice representation.
        # Outputs: the canonical slice.
        # Exceptions: TypeError("Not slice-like") if positions do not form a
        #   progression.
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            return s
        else:
            raise TypeError("Not slice-like")

    @property
    def indexer(self):
        # [FUNCTION DOC]
        # Property: the preferred indexer for numpy indexing.
        # Outputs: the slice if slice-like, else the intp array.
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            return s
        else:
            return self._as_array

    @property
    @cython.critical_section
    def as_array(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Property: positions as a 1-D intp array.
        # Outputs: cached array; if absent, created by arange over the slice bounds and
        # cached (double-checked under a critical section for free-threading).
        cdef:
            Py_ssize_t start, stop, _

        if not self._has_array:
            start, stop, step, _ = slice_get_indices_ex(self._as_slice)
            # NOTE: this is the C-optimized equivalent of
            #  `np.arange(start, stop, step, dtype=np.intp)`
            as_array = cnp.PyArray_Arange(start, stop, step, NPY_INTP)
            if not self._has_array:
                self._as_array = as_array
                self._has_array = True

        return self._as_array

    @property
    def is_slice_like(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether the placement can be expressed as a slice.
        # Outputs: bool. Side effects: may cache the slice representation.
        cdef:
            slice s = self._ensure_has_slice()

        return s is not None

    def __getitem__(self, loc):
        # [FUNCTION DOC]
        # Index into the placement.
        # Inputs: loc - integer, slice, or array indexer.
        # Outputs: a 0-d scalar position if the result is scalar, otherwise a new
        # BlockPlacement. Slice-like placements compose the slices (slice_getitem).
        # Exceptions: IndexError for out-of-bounds integer locations.
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            val = slice_getitem(s, loc)
        else:
            val = self._as_array[loc]

        if not isinstance(val, slice) and val.ndim == 0:
            return val

        return BlockPlacement(val)

    def delete(self, loc) -> BlockPlacement:
        # [FUNCTION DOC]
        # Remove positions by location.
        # Inputs: loc - integer/slice/array of locations into the placement.
        # Outputs: new BlockPlacement built from the array with those entries deleted.
        return BlockPlacement(np.delete(self.as_array, loc, axis=0))

    def append(self, others) -> BlockPlacement:
        # [FUNCTION DOC]
        # Concatenate placements.
        # Inputs: others - sequence of BlockPlacement.
        # Outputs: self if others is empty, else a new BlockPlacement from the
        # concatenated arrays (self first).
        if not len(others):
            return self

        return BlockPlacement(
            np.concatenate([self.as_array] + [o.as_array for o in others])
        )

    cdef BlockPlacement iadd(self, other):
        # [FUNCTION DOC]
        # HELPER: Add an offset to every position (the implementation of add).
        # Inputs: other - integer or array broadcastable against the positions.
        # Outputs: self if other is integer 0 and slice-like; for integer offsets on a
        # slice-like placement, a new slice-based placement with shifted bounds (stop
        #   None
        # if it becomes negative); otherwise a new array-based placement of
        #   as_array+other.
        # Exceptions: ValueError("iadd causes length change") if any resulting position
        #   is
        # negative (for slices: positive step with negative start, or negative step with
        # stop less than step).
        # Side effects: none (despite the name, self is not mutated).
        cdef:
            slice s = self._ensure_has_slice()
            Py_ssize_t other_int, start, stop, step

        if is_integer_object(other) and s is not None:
            other_int = <Py_ssize_t>other

            if other_int == 0:
                # BlockPlacement is treated as immutable
                return self

            start, stop, step, _ = slice_get_indices_ex(s)
            start += other_int
            stop += other_int

            if (step > 0 and start < 0) or (step < 0 and stop < step):
                raise ValueError("iadd causes length change")

            if stop < 0:
                val = slice(start, None, step)
            else:
                val = slice(start, stop, step)

            return BlockPlacement(val)
        else:
            newarr = self.as_array + other
            if (newarr < 0).any():
                raise ValueError("iadd causes length change")

            val = newarr
            return BlockPlacement(val)

    def add(self, other) -> BlockPlacement:
        # [FUNCTION DOC]
        # Public wrapper returning the result of iadd(other).
        # Inputs: other - integer or ndarray offset.
        # Outputs: new (or same) BlockPlacement. Exceptions: ValueError as in iadd.
        # We can get here with int or ndarray
        return self.iadd(other)

    @cython.critical_section
    cdef slice _ensure_has_slice(self):
        # [FUNCTION DOC]
        # HELPER: Return the slice representation, computing it on first use.
        # Outputs: the canonical slice, or None if positions are not a progression.
        # Side effects: caches the result (under a critical section).
        if not self._has_slice:
            as_slice = indexer_as_slice(self._as_array)
            # check again after indexer_as_slice call
            if not self._has_slice:
                self._as_slice = as_slice
                self._has_slice = True
        return self._as_slice

    cpdef BlockPlacement increment_above(self, Py_ssize_t loc):
        # [FUNCTION DOC]
        # Shift positions >= loc up by one (used when inserting a column at loc).
        # Inputs: loc - insertion location.
        # Outputs: self if slice-like and entirely below loc; a shifted slice placement
        #   if
        # slice-like and entirely at/above loc; otherwise a new array placement with
        # positions >= loc incremented (all incremented when loc is 0).
        # Side effects: allocates a new array when not on a fast path.
        """
        Increment any entries of 'loc' or above by one.
        """
        cdef:
            slice nv, s = self._ensure_has_slice()
            Py_ssize_t start, stop, step
            ndarray[intp_t, ndim=1] newarr

        if s is not None:
            # see if we are either all-above or all-below, each of which
            #  have fastpaths available.

            start, stop, step, _ = slice_get_indices_ex(s)

            if start < loc and stop <= loc:
                # We are entirely below, nothing to increment
                return self

            if start >= loc and stop >= loc:
                # We are entirely above, we can efficiently increment out slice
                nv = slice(start + 1, stop + 1, step)
                return BlockPlacement(nv)

        if loc == 0:
            # fastpath where we know everything is >= 0
            newarr = self.as_array + 1
            return BlockPlacement(newarr)

        newarr = self.as_array.copy()
        newarr[newarr >= loc] += 1
        return BlockPlacement(newarr)

    def tile_for_unstack(self, factor: int) -> np.ndarray:
        # [FUNCTION DOC]
        # Compute the manager locations of a block after unstacking.
        # Inputs: factor - number of new columns each original column expands into.
        # Outputs: intp array: for each position x (in iteration order) the run
        # x*factor .. (x+1)*factor-1; computed directly as one arange when the placement
        #   is
        # a step-1 slice.
        """
        Find the new mgr_locs for the un-stacked version of a Block.
        """
        cdef:
            slice slc = self._ensure_has_slice()
            ndarray[intp_t, ndim=1] new_placement

        if slc is not None and slc.step == 1:
            new_slc = slice(slc.start * factor, slc.stop * factor, 1)
            # equiv: np.arange(new_slc.start, new_slc.stop, dtype=np.intp)
            new_placement = cnp.PyArray_Arange(new_slc.start, new_slc.stop, 1, NPY_INTP)
        else:
            # Note: test_pivot_table_empty_aggfunc gets here with `slc is not None`
            mapped = [
                # equiv: np.arange(x * factor, (x + 1) * factor, dtype=np.intp)
                cnp.PyArray_Arange(x * factor, (x + 1) * factor, 1, NPY_INTP)
                for x in self
            ]
            new_placement = np.concatenate(mapped)
        return new_placement


cdef slice slice_canonize(slice s):
    # [FUNCTION DOC]
    # HELPER: Convert a slice to canonical bounded form.
    # Inputs: s - slice with integer or None fields.
    # Outputs: slice with explicit step; for positive step: start defaults to 0 and is
    # clipped to stop; for negative step: stop defaults to None (represented by -1)
    # and is clipped to start. A negative final stop is returned as None.
    # Exceptions: ValueError if step is 0, if the slice is unbounded (missing stop for
    # positive step, missing start for negative step), or if start (or, for positive
    # step with explicit stop, stop) is negative.
    """
    Convert slice to canonical bounded form.
    """
    cdef:
        Py_ssize_t start = 0, stop = 0, step = 1

    if s.step is None:
        step = 1
    else:
        step = <Py_ssize_t>s.step
        if step == 0:
            raise ValueError("slice step cannot be zero")

    if step > 0:
        if s.stop is None:
            raise ValueError("unbounded slice")

        stop = <Py_ssize_t>s.stop
        if s.start is None:
            start = 0
        else:
            start = <Py_ssize_t>s.start
            if start > stop:
                start = stop
    elif step < 0:
        if s.start is None:
            raise ValueError("unbounded slice")

        start = <Py_ssize_t>s.start
        if s.stop is None:
            stop = -1
        else:
            stop = <Py_ssize_t>s.stop
            if stop > start:
                stop = start

    if start < 0 or (stop < 0 and s.stop is not None and step > 0):
        raise ValueError("unbounded slice")

    if stop < 0:
        return slice(start, None, step)
    else:
        return slice(start, stop, step)


cpdef Py_ssize_t slice_len(slice slc, Py_ssize_t objlen=PY_SSIZE_T_MAX) except -1:
    # [FUNCTION DOC]
    # Length of a bounded slice.
    # Inputs: slc - slice; objlen - container length (default the maximum ssize_t,
    # which treats the slice as bounded).
    # Outputs: number of elements selected (ssize_t).
    # Exceptions: TypeError if slc is None; errors from Python slice resolution.
    """
    Get length of a bounded slice.

    The slice must not have any "open" bounds that would create dependency on
    container size, i.e.:
    - if ``s.step is None or s.step > 0``, ``s.stop`` is not ``None``
    - if ``s.step < 0``, ``s.start`` is not ``None``

    Otherwise, the result is unreliable.
    """
    cdef:
        Py_ssize_t start, stop, step, length

    if slc is None:
        raise TypeError("slc must be slice")  # pragma: no cover

    PySlice_GetIndicesEx(slc, objlen, &start, &stop, &step, &length)

    return length


cdef (Py_ssize_t, Py_ssize_t, Py_ssize_t, Py_ssize_t) slice_get_indices_ex(
    slice slc, Py_ssize_t objlen=PY_SSIZE_T_MAX
):
    # [FUNCTION DOC]
    # HELPER: Resolve a slice against a length.
    # Inputs: slc - slice; objlen - container length (default maximum ssize_t).
    # Outputs: C tuple (start, stop, step, length) as computed by Python's slice
    # index resolution.
    # Exceptions: TypeError if slc is None.
    """
    Get (start, stop, step, length) tuple for a slice.

    If `objlen` is not specified, slice must be bounded, otherwise the result
    will be wrong.
    """
    cdef:
        Py_ssize_t start, stop, step, length

    if slc is None:
        raise TypeError("slc should be a slice")  # pragma: no cover

    PySlice_GetIndicesEx(slc, objlen, &start, &stop, &step, &length)

    return start, stop, step, length


cdef slice_getitem(slice slc, ind):
    # [FUNCTION DOC]
    # HELPER: Index a bounded slice with another indexer, without materializing.
    # Inputs: slc - bounded slice; ind - slice or any numpy-style indexer.
    # Outputs: if ind is a slice, a composed slice (slc itself for a full positive
    # no-op slice); otherwise the result of indexing arange(start, stop, step) (intp)
    # with ind (array or scalar).
    # KNOWN ISSUE: composition with a negative-step ind is incorrect: start is moved to
    # the last element but the composed step stays positive and the start/stop use
    # ind's reversed indices, e.g. slice(0,5,1) indexed by [::-1] yields slice(8,3,1)
    # (empty) instead of positions 4..0.
    cdef:
        Py_ssize_t s_start, s_stop, s_step, s_len
        Py_ssize_t ind_start, ind_stop, ind_step, ind_len

    s_start, s_stop, s_step, s_len = slice_get_indices_ex(slc)

    if isinstance(ind, slice):
        ind_start, ind_stop, ind_step, ind_len = slice_get_indices_ex(ind, s_len)

        if ind_step > 0 and ind_len == s_len:
            # short-cut for no-op slice
            if ind_len == s_len:
                return slc

        if ind_step < 0:
            s_start = s_stop - s_step
            ind_step = -ind_step

        s_step *= ind_step
        s_stop = s_start + ind_stop * s_step
        s_start = s_start + ind_start * s_step

        if s_step < 0 and s_stop < 0:
            return slice(s_start, None, s_step)
        else:
            return slice(s_start, s_stop, s_step)

    else:
        # NOTE:
        # this is the C-optimized equivalent of
        # `np.arange(s_start, s_stop, s_step, dtype=np.intp)[ind]`
        return cnp.PyArray_Arange(s_start, s_stop, s_step, NPY_INTP)[ind]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef slice indexer_as_slice(intp_t[:] vals):
    # [FUNCTION DOC]
    # HELPER: Try to express an array of positions as a slice.
    # Inputs: vals - 1-D intp memoryview.
    # Outputs: slice(start, stop, d) if all values are non-negative and form an
    # arithmetic progression with non-zero difference d (stop None if it would be
    # negative); a length-1 array gives slice(v, v+1, 1); otherwise None (including
    # empty input).
    # Exceptions: TypeError if vals is None.
    cdef:
        Py_ssize_t i, n, start, stop
        int64_t d

    if vals is None:
        raise TypeError("vals must be ndarray")  # pragma: no cover

    n = vals.shape[0]

    if n == 0 or vals[0] < 0:
        return None

    if n == 1:
        return slice(vals[0], vals[0] + 1, 1)

    if vals[1] < 0:
        return None

    # n > 2
    d = vals[1] - vals[0]

    if d == 0:
        return None

    for i in range(2, n):
        if vals[i] < 0 or vals[i] - vals[i - 1] != d:
            return None

    start = vals[0]
    stop = start + n * d
    if stop < 0 and d < 0:
        return slice(start, None, d)
    else:
        return slice(start, stop, d)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_concat_blkno_indexers(list blknos_list not None):
    # [FUNCTION DOC]
    # Split columns into runs where the block number is constant in every manager.
    # Inputs: blknos_list - list of 1-D intp arrays (one per manager), equal lengths.
    # Outputs: list of tuples (blknos_for_run, BlockPlacement(slice(start, stop))),
    # where blknos_for_run is an intp array holding each manager's block number for
    # the run. Empty list if there are no columns.
    # Exceptions: AssertionError if the arrays have different lengths.
    # Side effects: allocates arrays; bounds/wraparound checks disabled.
    """
    Given the mgr.blknos for a list of mgrs, break range(len(mgrs[0])) into
    slices such that within each slice blknos_list[i] is constant for each i.

    This is a multi-Manager analogue to get_blkno_indexers with group=False.
    """
    # we have the blknos for each of several BlockManagers
    # list[np.ndarray[int64_t]]
    cdef:
        Py_ssize_t i, j, k, start, ncols
        cnp.npy_intp n_mgrs
        ndarray[intp_t] blknos, cur_blknos, run_blknos
        BlockPlacement bp
        list result = []

    n_mgrs = len(blknos_list)
    cur_blknos = cnp.PyArray_EMPTY(1, &n_mgrs, cnp.NPY_INTP, 0)

    blknos = blknos_list[0]
    ncols = len(blknos)
    if ncols == 0:
        return []

    start = 0
    for i in range(n_mgrs):
        blknos = blknos_list[i]
        cur_blknos[i] = blknos[0]
        assert len(blknos) == ncols

    for i in range(1, ncols):
        # For each column, we check if the Block it is part of (i.e. blknos[i])
        #  is the same the previous column (i.e. blknos[i-1]) *and* if this is
        #  the case for each blknos in blknos_list.  If not, we start a new "run".
        for k in range(n_mgrs):
            blknos = blknos_list[k]
            # assert cur_blknos[k] == blknos[i - 1]

            if blknos[i] != blknos[i - 1]:
                bp = BlockPlacement(slice(start, i))
                run_blknos = cnp.PyArray_Copy(cur_blknos)
                result.append((run_blknos, bp))

                start = i
                for j in range(n_mgrs):
                    blknos = blknos_list[j]
                    cur_blknos[j] = blknos[i]
                break  # break out of `for k in range(n_mgrs)` loop

    if start != ncols:
        bp = BlockPlacement(slice(start, ncols))
        run_blknos = cnp.PyArray_Copy(cur_blknos)
        result.append((run_blknos, bp))
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def get_blkno_indexers(
    const int64_t[:] blknos, bint group=True
) -> list[tuple[int, slice | np.ndarray]]:
    # [FUNCTION DOC]
    # Enumerate contiguous runs of equal block numbers.
    # Inputs: blknos - 1-D int64 memoryview; group - whether to merge runs per value.
    # Outputs: list of (blkno, indexer). With group False: one (blkno, slice) per run
    # in order. With group True: one entry per distinct blkno in order of first
    # appearance; the indexer is a slice if that value has a single run, else an int64
    # array of all its positions.
    """
    Enumerate contiguous runs of integers in ndarray.

    Iterate over elements of `blknos` yielding ``(blkno, slice(start, stop))``
    pairs for each contiguous run found.

    If `group` is True and there is more than one run for a certain blkno,
    ``(blkno, array)`` with an array containing positions of all elements equal
    to blkno.

    Returns
    -------
    list[tuple[int, slice | np.ndarray]]
    """
    # There's blkno in this function's name because it's used in block &
    # blockno handling.
    cdef:
        int64_t cur_blkno
        Py_ssize_t i, start, stop, n, diff
        cnp.npy_intp tot_len
        int64_t blkno
        object group_dict = defaultdict(list)
        ndarray[int64_t, ndim=1] arr

    n = blknos.shape[0]
    result = list()

    if n == 0:
        return result

    start = 0
    cur_blkno = blknos[start]

    if group is False:
        for i in range(1, n):
            if blknos[i] != cur_blkno:
                result.append((cur_blkno, slice(start, i)))

                start = i
                cur_blkno = blknos[i]

        result.append((cur_blkno, slice(start, n)))
    else:
        for i in range(1, n):
            if blknos[i] != cur_blkno:
                group_dict[cur_blkno].append((start, i))

                start = i
                cur_blkno = blknos[i]

        group_dict[cur_blkno].append((start, n))

        for blkno, slices in group_dict.items():
            if len(slices) == 1:
                result.append((blkno, slice(slices[0][0], slices[0][1])))
            else:
                tot_len = sum(stop - start for start, stop in slices)
                # equiv np.empty(tot_len, dtype=np.int64)
                arr = cnp.PyArray_EMPTY(1, &tot_len, cnp.NPY_INT64, 0)

                i = 0
                for start, stop in slices:
                    for diff in range(start, stop):
                        arr[i] = diff
                        i += 1

                result.append((blkno, arr))

    return result


def get_blkno_placements(blknos, group: bool = True):
    # [FUNCTION DOC]
    # Generator wrapping get_blkno_indexers.
    # Inputs: blknos - array-like of block numbers (converted to int64); group - bool.
    # Outputs: yields (blkno, BlockPlacement(indexer)) for each run/group.
    """
    Parameters
    ----------
    blknos : np.ndarray[int64]
    group : bool, default True

    Returns
    -------
    iterator
        yield (blkno, BlockPlacement)
    """
    blknos = ensure_int64(blknos)

    for blkno, indexer in get_blkno_indexers(blknos, group):
        yield blkno, BlockPlacement(indexer)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_blklocs_and_blknos(
    const intp_t[:] blklocs,
    const intp_t[:] blknos,
    Py_ssize_t loc,
    intp_t nblocks,
):
    # [FUNCTION DOC]
    # Update blklocs/blknos for insertion of a new single-column block at loc.
    # Inputs: blklocs, blknos - 1-D intp arrays (length n); loc - insertion position
    # (0..n); nblocks - number assigned to the new block.
    # Outputs: tuple (new_blklocs, new_blknos) of length n+1 with (0, nblocks) inserted
    # at loc and the rest shifted.
    # Side effects: allocates arrays; bounds/wraparound checks disabled.
    """
    Update blklocs and blknos when a new column is inserted at 'loc'.
    """
    cdef:
        Py_ssize_t i
        cnp.npy_intp length = blklocs.shape[0] + 1
        ndarray[intp_t, ndim=1] new_blklocs, new_blknos

    # equiv: new_blklocs = np.empty(length, dtype=np.intp)
    new_blklocs = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)
    new_blknos = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)

    for i in range(loc):
        new_blklocs[i] = blklocs[i]
        new_blknos[i] = blknos[i]

    new_blklocs[loc] = 0
    new_blknos[loc] = nblocks

    for i in range(loc, length - 1):
        new_blklocs[i + 1] = blklocs[i]
        new_blknos[i + 1] = blknos[i]

    return new_blklocs, new_blknos


def _unpickle_block(values, placement, ndim):
    # [FUNCTION DOC]
    # HELPER: Pickle reconstructor for Block.
    # Inputs: values - array data; placement - BlockPlacement or indexer; ndim.
    # Outputs: new Block made by pandas.core.internals.blocks.new_block after coercing
    # values with maybe_coerce_values.
    # We have to do some gymnastics b/c "ndim" is keyword-only

    from pandas.core.internals.blocks import (
        maybe_coerce_values,
        new_block,
    )
    values = maybe_coerce_values(values)

    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return new_block(values, placement, ndim=ndim)


@cython.freelist(64)
cdef class Block:
    # [CLASS DOC]
    # Base class for pandas Blocks: a homogeneous array (values) plus the manager
    # locations it occupies (_mgr_locs), dimensionality (ndim, 1 or 2) and a
    # BlockValuesRefs object shared by all blocks viewing the same data (refs).
    """
    Defining __init__ in a cython class significantly improves performance.
    """
    cdef:
        public BlockPlacement _mgr_locs
        public BlockValuesRefs refs
        readonly int ndim
        # 2023-08-15 no apparent performance improvement from declaring values
        #  as ndarray in a type-special subclass (similar for NDArrayBacked).
        #  This might change if slice_block_rows can be optimized with something
        #  like https://github.com/numpy/numpy/issues/23934
        public object values

    def __cinit__(
        self,
        values,
        placement: BlockPlacement,
        ndim: int,
        refs: BlockValuesRefs | None = None,
    ):
        # [FUNCTION DOC]
        # Allocate/initialize a Block.
        # Inputs: values - ndarray or ExtensionArray (already coerced); placement -
        # BlockPlacement; ndim - 1 or 2; refs - existing BlockValuesRefs or None.
        # Outputs: none.
        # Side effects: when refs is None, creates a new BlockValuesRefs referencing
        #   only
        # this block; otherwise registers a weak reference to this block in refs.
        """
        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            We assume maybe_coerce_values has already been called.
        placement : BlockPlacement
        ndim : int
            1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        refs: BlockValuesRefs, optional
            Ref tracking object or None if block does not have any refs.
        """
        self.values = values

        self._mgr_locs = placement
        self.ndim = ndim
        if refs is None:
            # if no refs are passed, that means we are creating a Block from
            # new values that it uniquely owns -> start a new BlockValuesRefs
            # object that only references this block
            self.refs = BlockValuesRefs(self)
        else:
            # if refs are passed, this is the BlockValuesRefs object that is shared
            # with the parent blocks which share the values, and a reference to this
            # new block is added
            refs.add_reference(self)
            self.refs = refs

    cpdef __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support.
        # Outputs: (_unpickle_block, (values, mgr_locs indexer, ndim)).
        args = (self.values, self.mgr_locs.indexer, self.ndim)
        return _unpickle_block, args

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Restore state from a pickle.
        # Inputs: state - tuple (placement indexer, values[, ndim]).
        # Outputs: none.
        # Side effects: sets mgr_locs and values (unwrapping pandas containers to
        #   arrays);
        # infers ndim for older pickles that lack it.
        from pandas.core.construction import extract_array

        self.mgr_locs = BlockPlacement(state[0])
        self.values = extract_array(state[1], extract_numpy=True)
        if len(state) > 2:
            # we stored ndim
            self.ndim = state[2]
        else:
            # older pickle
            from pandas.core.internals.api import _maybe_infer_ndim

            ndim = _maybe_infer_ndim(self.values, self.mgr_locs)
            self.ndim = ndim

    cpdef Block slice_block_rows(self, slice slicer):
        # [FUNCTION DOC]
        # Slice the block along its last (row) axis.
        # Inputs: slicer - slice. Assumes ndim == 2 for a frame block.
        # Outputs: new block of the same type viewing values[..., slicer], same
        #   placement,
        # ndim and shared refs.
        # Side effects: registers the new block in refs.
        """
        Perform __getitem__-like specialized to slicing along index.

        Assumes self.ndim == 2
        """
        new_values = self.values[..., slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)


@cython.freelist(64)
cdef class BlockManager:
    # [CLASS DOC]
    # Base class for BlockManager / SingleBlockManager. Fields: blocks - tuple of
    # Blocks; axes - list of Index objects (axes[0] are the columns/items for 2-D);
    # consolidation flags; private blknos (block number per item) and blklocs
    # (position within its block per item), built lazily. Uses a freelist of 64.
    cdef:
        public tuple blocks
        public list axes
        public bint _known_consolidated, _is_consolidated
        ndarray __blknos, __blklocs

    def __cinit__(
        self,
        blocks=None,
        axes=None,
        verify_integrity=True,
    ):
        # [FUNCTION DOC]
        # Initialize the manager.
        # Inputs: blocks - tuple or list of Blocks, or None (used for unpickling, then
        # nothing is set); axes - list of Index; verify_integrity - ignored here.
        # Outputs: none.
        # Side effects: stores a tuple of blocks and a shallow copy of axes; resets
        # consolidation flags and the blknos/blklocs caches to None.
        # None as defaults for unpickling GH#42345
        if blocks is None:
            # This adds 1-2 microseconds to DataFrame(np.array([]))
            return

        if isinstance(blocks, list):
            # Backward compat for e.g. pyarrow
            blocks = tuple(blocks)

        self.blocks = blocks
        self.axes = axes.copy()  # copy to make sure we are not remotely-mutable

        # Populate known_consolidate, blknos, and blklocs lazily
        self._known_consolidated = False
        self._is_consolidated = False
        self._blknos = None
        self._blklocs = None

    # -------------------------------------------------------------------
    # Block Placement
    @property
    @cython.critical_section
    def _blknos(self):
        # [FUNCTION DOC]
        # Property getter for the cached blknos array (or None), read under a critical
        # section.
        return self.__blknos

    @_blknos.setter
    @cython.critical_section
    def _blknos(self, ndarray val):
        # [FUNCTION DOC]
        # Property setter for the cached blknos array.
        # Inputs: val - intp ndarray or None. Side effects: replaces the cache.
        self.__blknos = val

    @property
    @cython.critical_section
    def _blklocs(self):
        # [FUNCTION DOC]
        # Property getter for the cached blklocs array (or None).
        return self.__blklocs

    @_blklocs.setter
    @cython.critical_section
    def _blklocs(self, ndarray val):
        # [FUNCTION DOC]
        # Property setter for the cached blklocs array.
        # Inputs: val - intp ndarray or None. Side effects: replaces the cache.
        self.__blklocs = val

    @cython.critical_section
    cpdef _rebuild_blknos_and_blklocs(self):
        # [FUNCTION DOC]
        # Recompute blknos and blklocs from the blocks' placements.
        # Inputs: none (uses self.shape[0] as the number of items).
        # Outputs: none.
        # Exceptions: AssertionError("Gaps in blk ref_locs") if some item is not covered
        # by any block.
        # Side effects: allocates and stores new arrays (initialized to -1 then filled:
        # item j of block b at position i gets blkno b, blkloc i); critical section
        #   held.
        """
        Update mgr._blknos / mgr._blklocs.
        """
        cdef:
            intp_t blkno, i, j
            cnp.npy_intp length = self.shape[0]
            Block blk
            BlockPlacement bp
            ndarray[intp_t, ndim=1] new_blknos, new_blklocs

        # equiv: np.empty(length, dtype=np.intp)
        new_blknos = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)
        new_blklocs = cnp.PyArray_EMPTY(1, &length, cnp.NPY_INTP, 0)
        # equiv: new_blknos.fill(-1)
        cnp.PyArray_FILLWBYTE(new_blknos, -1)
        cnp.PyArray_FILLWBYTE(new_blklocs, -1)

        for blkno, blk in enumerate(self.blocks):
            bp = blk._mgr_locs
            # Iterating over `bp` is a faster equivalent to
            #  new_blknos[bp.indexer] = blkno
            #  new_blklocs[bp.indexer] = np.arange(len(bp))
            for i, j in enumerate(bp):
                new_blknos[j] = blkno
                new_blklocs[j] = i

        for i in range(length):
            # faster than `for blkno in new_blknos`
            #  https://github.com/cython/cython/issues/4393
            blkno = new_blknos[i]

            # If there are any -1s remaining, this indicates that our mgr_locs
            #  are invalid.
            if blkno == -1:
                raise AssertionError("Gaps in blk ref_locs")

        self.__blknos = new_blknos
        self.__blklocs = new_blklocs

    # -------------------------------------------------------------------
    # Pickle

    cpdef __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support.
        # Outputs: (type(self), args) where args is (block, axis) for a single-axis
        # manager or (blocks, axes) otherwise.
        if len(self.axes) == 1:
            # SingleBlockManager, __init__ expects Block, axis
            args = (self.blocks[0], self.axes[0])
        else:
            args = (self.blocks, self.axes)
        return type(self), args

    cpdef __setstate__(self, state):
        # [FUNCTION DOC]
        # Restore from the legacy pickle format.
        # Inputs: state - tuple whose 4th element is a dict with key "0.14.1" holding
        # "axes" and "blocks" (each with "values" and "mgr_locs").
        # Outputs: none.
        # Exceptions: NotImplementedError for older formats.
        # Side effects: rebuilds axes and blocks (coercing values/shapes), then resets
        # consolidation flags and rebuilds blknos/blklocs; mutates the state dicts.
        from pandas.core.construction import extract_array
        from pandas.core.internals.blocks import (
            ensure_block_shape,
            maybe_coerce_values,
            new_block,
        )
        from pandas.core.internals.managers import ensure_index

        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            state = state[3]["0.14.1"]
            axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(axes)

            for blk in state["blocks"]:
                vals = blk["values"]
                # older versions may hold e.g. DatetimeIndex instead of DTA
                vals = extract_array(vals, extract_numpy=True)
                blk["values"] = maybe_coerce_values(ensure_block_shape(vals, ndim=ndim))

                if not isinstance(blk["mgr_locs"], BlockPlacement):
                    blk["mgr_locs"] = BlockPlacement(blk["mgr_locs"])

            nbs = [
                new_block(blk["values"], blk["mgr_locs"], ndim=ndim)
                for blk in state["blocks"]
            ]
            blocks = tuple(nbs)
            self.blocks = blocks
            self.axes = axes

        else:  # pragma: no cover
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        self._post_setstate()

    def _post_setstate(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Finalize unpickling: clear consolidation flags and rebuild
        # blknos/blklocs.
        self._is_consolidated = False
        self._known_consolidated = False
        self._rebuild_blknos_and_blklocs()

    # -------------------------------------------------------------------
    # Indexing

    cdef BlockManager _slice_mgr_rows(self, slice slobj):
        # [FUNCTION DOC]
        # HELPER: Slice all blocks along rows (axis 1 of the manager).
        # Inputs: slobj - slice.
        # Outputs: new manager of the same type with sliced blocks, a view of axes[0]
        #   and
        # axes[1] sliced; copies cached blknos/blklocs when available.
        cdef:
            Block blk, nb
            BlockManager mgr
            ndarray blknos, blklocs

        nbs = []
        for blk in self.blocks:
            nb = blk.slice_block_rows(slobj)
            nbs.append(nb)

        new_axes = [self.axes[0].view(), self.axes[1]._getitem_slice(slobj)]
        mgr = type(self)(tuple(nbs), new_axes, verify_integrity=False)

        # We can avoid having to rebuild blklocs/blknos
        with cython.critical_section(self):
            blklocs = self.__blklocs
            blknos = self.__blknos
        if blknos is not None:
            mgr._blknos = blknos.copy()
            mgr._blklocs = blklocs.copy()
        return mgr

    def get_slice(self, slobj: slice, axis: int = 0) -> BlockManager:
        # [FUNCTION DOC]
        # Slice the manager along an axis.
        # Inputs: slobj - slice; axis - 0 (items, via the subclass method
        # _slice_take_blocks_ax0) or 1 (rows).
        # Outputs: new manager with sliced axis and a view of the other axis.
        # Exceptions: IndexError for any other axis.

        if axis == 0:
            new_blocks = self._slice_take_blocks_ax0(slobj)
        elif axis == 1:
            return self._slice_mgr_rows(slobj)
        else:
            raise IndexError("Requested axis not found in manager")

        new_axes = list(self.axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)
        new_axes[1 - axis] = self.axes[1 - axis].view()

        return type(self)(tuple(new_blocks), new_axes, verify_integrity=False)


cdef class BlockValuesRefs:
    # [CLASS DOC]
    # Tracks all objects (Blocks and Indexes) that share one underlying array, via a
    # list of weak references, for Copy-on-Write. clear_counter is the list size at
    # which dead references are purged (adaptive, minimum 500).
    """Tracks all references to a given array.

    Keeps track of all blocks (through weak references) that reference the same
    data.
    """
    cdef:
        public list referenced_blocks
        public int clear_counter

    def __cinit__(self, blk: Block | None = None) -> None:
        # [FUNCTION DOC]
        # Initialize the tracker.
        # Inputs: blk - optional Block to reference first.
        # Outputs: none.
        # Side effects: creates the weakref list (with blk if given); clear_counter =
        #   500.
        if blk is not None:
            self.referenced_blocks = [PyWeakref_NewRef(blk, None)]
        else:
            self.referenced_blocks = []
        self.clear_counter = 500  # set reasonably high

    cdef _clear_dead_references(self, bint force=False):
        # [FUNCTION DOC]
        # HELPER: Purge weak references whose targets are dead.
        # Inputs: force - purge regardless of size.
        # Outputs: none.
        # Side effects: when forced or the list exceeds clear_counter, filters the list;
        # then halves clear_counter (min 500) if fewer than half remain, or grows it to
        # max(double, current count) if still above it. Under free-threading, returns
        # early without changing anything if a weakref lookup errors.
        # Use exponential backoff to decide when we want to clear references
        # if force=False. Clearing for every insertion causes slowdowns if
        # all these objects stay alive, e.g. df.items() for wide DataFrames
        # see GH#55245 and GH#55008
        IF CYTHON_COMPATIBLE_WITH_FREE_THREADING:
            cdef PyObject* pobj
            cdef bint status

        if force or len(self.referenced_blocks) > self.clear_counter:
            IF CYTHON_COMPATIBLE_WITH_FREE_THREADING:
                new_referenced_blocks = []
                for ref in self.referenced_blocks:
                    status = PyWeakref_GetRef(ref, &pobj)
                    if status == -1:
                        return
                    elif status == 1:
                        new_referenced_blocks.append(ref)
                        Py_DECREF(<object>pobj)
                self.referenced_blocks = new_referenced_blocks
            ELSE:
                self.referenced_blocks = [
                    ref for ref in self.referenced_blocks
                    if PyWeakref_GetObject(ref) != Py_None
                ]

            nr_of_refs = len(self.referenced_blocks)
            if nr_of_refs < self.clear_counter // 2:
                self.clear_counter = max(self.clear_counter // 2, 500)
            elif nr_of_refs > self.clear_counter:
                self.clear_counter = max(self.clear_counter * 2, nr_of_refs)

    cpdef _add_reference_maybe_locked(self, Block blk):
        # [FUNCTION DOC]
        # HELPER: Add a weak reference to a block without locking.
        # Inputs: blk - Block. Side effects: may purge dead refs; appends to the list.
        self._clear_dead_references()
        self.referenced_blocks.append(PyWeakref_NewRef(blk, None))

    cpdef add_reference(self, Block blk):
        # [FUNCTION DOC]
        # Add a weak reference to a block (locked with a critical section when
        # free-threading). Inputs: blk - Block. Side effects: mutates the list.
        """Adds a new reference to our reference collection.

        Parameters
        ----------
        blk : Block
            The block that the new references should point to.
        """
        IF CYTHON_COMPATIBLE_WITH_FREE_THREADING:
            with cython.critical_section(self):
                self._add_reference_maybe_locked(blk)
        ELSE:
            self._add_reference_maybe_locked(blk)

    def _add_index_reference_maybe_locked(self, index: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Add a weak reference to an Index without locking.
        # Inputs: index - Index object. Side effects: may purge; appends.
        self._clear_dead_references()
        self.referenced_blocks.append(PyWeakref_NewRef(index, None))

    def add_index_reference(self, index: object) -> None:
        # [FUNCTION DOC]
        # Add a weak reference to an Index (locked when free-threading).
        # Inputs: index - Index object. Side effects: mutates the list.
        """Adds a new reference to our reference collection when creating an index.

        Parameters
        ----------
        index : Index
            The index that the new reference should point to.
        """
        IF CYTHON_COMPATIBLE_WITH_FREE_THREADING:
            with cython.critical_section(self):
                self._add_index_reference_maybe_locked(index)
        ELSE:
            self._add_index_reference_maybe_locked(index)

    def _has_reference_maybe_locked(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Unlocked implementation of has_reference.
        # Outputs: True if more than one live reference remains after a forced purge.
        # Side effects: purges dead references.
        self._clear_dead_references(force=True)
        # Checking for more references than block pointing to itself
        return len(self.referenced_blocks) > 1

    def has_reference(self) -> bool:
        # [FUNCTION DOC]
        # Whether the data is referenced by more than one live object (the owner's own
        # reference does not count).
        # Outputs: bool. Side effects: purges dead references; locks when free-
        #   threading.
        """Checks if block has foreign references.

        A reference is only relevant if it is still alive. The reference to
        ourselves does not count.

        Returns
        -------
        bool
        """
        IF CYTHON_COMPATIBLE_WITH_FREE_THREADING:
            with cython.critical_section(self):
                return self._has_reference_maybe_locked()
        ELSE:
            return self._has_reference_maybe_locked()
