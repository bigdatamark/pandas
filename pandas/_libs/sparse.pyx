# [MODULE DOC]
# Sparse index structures and kernels backing pandas SparseArray. A sparse vector of
# logical length "length" stores only "npoints" explicit values (sp_values) plus a
# fill_value used everywhere else; the SparseIndex describes which logical positions are
# stored. Two representations: IntIndex (sorted int32 array of stored positions) and
# BlockIndex (sorted runs given by int32 block start locations and block lengths).
# Provides conversion, intersection, union, position lookup, get_blocks (positions to
# runs), the templated binary op kernels from sparse_op_helper.pxi
# (sparse_<op>_<dtype>), and make_mask_object_ndarray.
cimport cython

import numpy as np

cimport numpy as cnp
from libc.math cimport (
    INFINITY as INF,
    NAN as NaN,
)
from numpy cimport (
    float64_t,
    int8_t,
    int32_t,
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()


# -----------------------------------------------------------------------------


cdef class SparseIndex:
    # [CLASS DOC]
    # Abstract base of IntIndex and BlockIndex; exists for typing and isinstance
    # dispatch. Cannot be instantiated directly.
    """
    Abstract superclass for sparse index types.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Always raises NotImplementedError (abstract class).
        raise NotImplementedError


cdef class IntIndex(SparseIndex):
    # [CLASS DOC]
    # Sparse index storing the stored logical positions explicitly. Attributes
    # (read-only): length - logical length; npoints - number of stored positions;
    # indices - C-contiguous int32 ndarray of strictly increasing positions in
    # [0, length).
    """
    Object for holding exact integer sparse indexing information

    Parameters
    ----------
    length : integer
    indices : array-like
        Contains integers corresponding to the indices.
    check_integrity : bool, default=True
        Check integrity of the input.
    """

    cdef readonly:
        Py_ssize_t length, npoints
        ndarray indices

    def __init__(self, Py_ssize_t length, indices, bint check_integrity=True):
        # [FUNCTION DOC]
        # Store length, convert indices to a contiguous int32 array, set npoints, and
        # optionally validate. Inputs: length - int; indices - array-like of ints;
        # check_integrity - bool. Outputs: None.
        # Exceptions: ValueError from check_integrity; numpy errors if indices cannot be
        # cast (values outside int32 range wrap or raise depending on numpy).
        # Side effects: mutates self; may copy indices.
        self.length = length
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.npoints = len(self.indices)

        if check_integrity:
            self.check_integrity()

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns (IntIndex, (length, indices)). Side effects: none.
        args = (self.length, self.indices)
        return IntIndex, args

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return "IntIndex\nIndices: <repr of indices>\n". Side effects: none.
        output = "IntIndex\n"
        output += f"Indices: {repr(self.indices)}\n"
        return output

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: memory usage in bytes of the indices array.
        return self.indices.nbytes

    cdef check_integrity(self):
        # [FUNCTION DOC]
        # HELPER: Validate: npoints <= length; if non-empty, min index >= 0, max index <
        # length and indices strictly increasing.
        # Outputs: None. Exceptions: ValueError describing the first violated condition.
        # Side effects: none.
        """
        Checks the following:

        - Indices are strictly ascending
        - Number of indices is at most self.length
        - Indices are at least 0 and at most the total length less one

        A ValueError is raised if any of these conditions is violated.
        """

        if self.npoints > self.length:
            raise ValueError(
                f"Too many indices. Expected {self.length} but found {self.npoints}"
            )

        # Indices are vacuously ordered and non-negative
        # if the sequence of indices is empty.
        if self.npoints == 0:
            return

        if self.indices.min() < 0:
            raise ValueError("No index can be less than zero")

        if self.indices.max() >= self.length:
            raise ValueError("All indices must be less than the length")

        monotonic = np.all(self.indices[:-1] < self.indices[1:])
        if not monotonic:
            raise ValueError("Indices must be strictly increasing")

    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # True if other is an IntIndex with the same length and identical indices (False
        # for any other type, including an equivalent BlockIndex).
        # Inputs: other - any object. Outputs: bool. Side effects: none.
        if not isinstance(other, IntIndex):
            return False

        if self is other:
            return True

        same_length = self.length == other.length
        same_indices = np.array_equal(self.indices, other.indices)
        return same_length and same_indices

    @property
    def ngaps(self) -> int:
        # [FUNCTION DOC]
        # Property: number of non-stored positions (length - npoints).
        return self.length - self.npoints

    cpdef to_int_index(self):
        # [FUNCTION DOC]
        # Return self (already an IntIndex). Side effects: none.
        return self

    def to_block_index(self):
        # [FUNCTION DOC]
        # Convert to an equivalent BlockIndex by grouping consecutive indices into runs
        # via get_blocks. Outputs: new BlockIndex. Side effects: none.
        locs, lens = get_blocks(self.indices)
        return BlockIndex(self.length, locs, lens)

    cpdef IntIndex intersect(self, SparseIndex y_):
        # [FUNCTION DOC]
        # Return an IntIndex of positions stored in both self and y_ (y_ is first
        # converted to IntIndex). Two-pointer merge over the sorted index arrays into a
        # buffer of size min(npoints), truncated to the number of matches.
        # Inputs: y_ - SparseIndex. Outputs: new IntIndex with the same length.
        # Exceptions: Exception (generic) if lengths differ. Side effects: none.
        # KNOWN ISSUE: raises a bare Exception on length mismatch while make_union
        # raises ValueError for the same condition.
        cdef:
            Py_ssize_t xi, yi = 0, result_indexer = 0
            int32_t xind
            ndarray[int32_t, ndim=1] xindices, yindices, new_indices
            IntIndex y

        # if is one already, returns self
        y = y_.to_int_index()

        if self.length != y.length:
            raise Exception("Indices must reference same underlying length")

        xindices = self.indices
        yindices = y.indices
        new_indices = np.empty(min(
            len(xindices), len(yindices)), dtype=np.int32)

        for xi in range(self.npoints):
            xind = xindices[xi]

            while yi < y.npoints and yindices[yi] < xind:
                yi += 1

            if yi >= y.npoints:
                break

            # TODO: would a two-pass algorithm be faster?
            if yindices[yi] == xind:
                new_indices[result_indexer] = xind
                result_indexer += 1

        new_indices = new_indices[:result_indexer]
        return IntIndex(self.length, new_indices)

    cpdef IntIndex make_union(self, SparseIndex y_):

        # [FUNCTION DOC]
        # Return an IntIndex of positions stored in either self or y_ (y_ converted to
        # IntIndex), computed with a sorted set union.
        # Inputs: y_ - SparseIndex. Outputs: new IntIndex.
        # Exceptions: ValueError if lengths differ. Side effects: none.
        cdef:
            ndarray[int32_t, ndim=1] new_indices
            IntIndex y

        # if is one already, returns self
        y = y_.to_int_index()

        if self.length != y.length:
            raise ValueError("Indices must reference same underlying length")

        new_indices = np.union1d(self.indices, y.indices)
        return IntIndex(self.length, new_indices)

    @cython.wraparound(False)
    cpdef int32_t lookup(self, Py_ssize_t index):
        # [FUNCTION DOC]
        # Map a logical position to its position within the stored values via binary
        # search. Inputs: index - logical position. Outputs: int32 offset into
        # sp_values, or -1 if index is not stored or out of [0, length). Side effects:
        # none.
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
        cdef:
            int32_t res
            ndarray[int32_t, ndim=1] inds

        inds = self.indices
        if self.npoints == 0:
            return -1
        elif index < 0 or self.length <= index:
            return -1

        res = inds.searchsorted(index)
        if res == self.npoints:
            return -1
        elif inds[res] == index:
            return res
        else:
            return -1

    @cython.wraparound(False)
    cpdef ndarray[int32_t] lookup_array(self, ndarray[int32_t, ndim=1] indexer):
        # [FUNCTION DOC]
        # Vectorized lookup: for each logical position in indexer return its offset into
        # the stored values or -1. Positions outside [first, last] stored index are -1;
        # others are binary searched and kept only on exact match.
        # Inputs: indexer - 1-D int32 ndarray. Outputs: int32 ndarray of same length.
        # Side effects: none.
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
        cdef:
            Py_ssize_t n
            ndarray[int32_t, ndim=1] inds
            ndarray[uint8_t, ndim=1, cast=True] mask
            ndarray[int32_t, ndim=1] masked
            ndarray[int32_t, ndim=1] res
            ndarray[int32_t, ndim=1] results

        n = len(indexer)
        results = np.empty(n, dtype=np.int32)
        results[:] = -1

        if self.npoints == 0:
            return results

        inds = self.indices
        mask = (inds[0] <= indexer) & (indexer <= inds[len(inds) - 1])

        masked = indexer[mask]
        res = inds.searchsorted(masked).astype(np.int32)

        res[inds[res] != masked] = -1
        results[mask] = res
        return results


cpdef get_blocks(ndarray[int32_t, ndim=1] indices):
    # [FUNCTION DOC]
    # Group sorted positions into runs of consecutive integers.
    # Inputs: indices - 1-D int32 ndarray, assumed strictly increasing.
    # Outputs: tuple (locs, lens) int32 ndarrays: start position and length of each run;
    # both empty for empty input. Side effects: none.
    cdef:
        Py_ssize_t i, npoints, result_indexer = 0
        int32_t block, length = 1, cur, prev
        ndarray[int32_t, ndim=1] locs, lens

    npoints = len(indices)

    # just handle the special empty case separately
    if npoints == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # block size can't be longer than npoints
    locs = np.empty(npoints, dtype=np.int32)
    lens = np.empty(npoints, dtype=np.int32)

    # TODO: two-pass algorithm faster?
    prev = block = indices[0]
    for i in range(1, npoints):
        cur = indices[i]
        if cur - prev > 1:
            # new block
            locs[result_indexer] = block
            lens[result_indexer] = length
            block = cur
            length = 1
            result_indexer += 1
        else:
            # same block, increment length
            length += 1

        prev = cur

    locs[result_indexer] = block
    lens[result_indexer] = length
    result_indexer += 1
    locs = locs[:result_indexer]
    lens = lens[:result_indexer]
    return locs, lens


# -----------------------------------------------------------------------------
# BlockIndex

cdef class BlockIndex(SparseIndex):
    # [CLASS DOC]
    # Sparse index storing runs of stored positions. Attributes (read-only): length -
    # logical length; nblocks - number of blocks; npoints - total stored positions (sum
    # of block lengths); blocs - int32 block start positions (strictly increasing);
    # blengths - int32 block lengths (non-zero). Private raw pointers locbuf / lenbuf
    # alias the data of blocs / blengths for fast C access; __weakref__ slot enables
    # weak references.
    """
    Object for holding block-based sparse indexing information

    Parameters
    ----------
    """
    cdef readonly:
        int32_t nblocks, npoints, length
        ndarray blocs, blengths

    cdef:
        object __weakref__  # need to be picklable
        int32_t *locbuf
        int32_t *lenbuf

    def __init__(self, length, blocs, blengths):

        # [FUNCTION DOC]
        # Store blocs and blengths as contiguous int32 arrays, cache raw data pointers,
        # set length, nblocks, npoints and validate (always).
        # Inputs: length - int; blocs, blengths - array-likes of ints. Outputs: None.
        # Exceptions: ValueError from check_integrity. Side effects: mutates self.
        self.blocs = np.ascontiguousarray(blocs, dtype=np.int32)
        self.blengths = np.ascontiguousarray(blengths, dtype=np.int32)

        # in case we need
        self.locbuf = <int32_t*>self.blocs.data
        self.lenbuf = <int32_t*>self.blengths.data

        self.length = length
        self.nblocks = np.int32(len(self.blocs))
        self.npoints = self.blengths.sum()

        self.check_integrity()

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns (BlockIndex, (length, blocs, blengths)).
        args = (self.length, self.blocs, self.blengths)
        return BlockIndex, args

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return a multi-line repr listing block locations and lengths.
        output = "BlockIndex\n"
        output += f"Block locations: {repr(self.blocs)}\n"
        output += f"Block lengths: {repr(self.blengths)}"

        return output

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: bytes used by blocs plus blengths.
        return self.blocs.nbytes + self.blengths.nbytes

    @property
    def ngaps(self) -> int:
        # [FUNCTION DOC]
        # Property: number of non-stored positions (length - npoints).
        return self.length - self.npoints

    cdef check_integrity(self):
        # [FUNCTION DOC]
        # HELPER: Validate that blocs and blengths have equal length, block starts are
        # strictly increasing, no block overlaps the next, the last block does not
        # extend past length, and no block has zero length.
        # Outputs: None. Exceptions: ValueError on the first violation. Side effects:
        # none.
        # KNOWN ISSUE: negative block locations and negative block lengths are not
        # rejected (e.g. BlockIndex(10, [-5], [2]) and BlockIndex(10, [2], [-1]) are
        # accepted).
        """
        Check:
        - Locations are in ascending order
        - No overlapping blocks
        - Blocks to not start after end of index, nor extend beyond end
        """
        cdef:
            Py_ssize_t i
            ndarray[int32_t, ndim=1] blocs, blengths

        blocs = self.blocs
        blengths = self.blengths

        if len(blocs) != len(blengths):
            raise ValueError("block bound arrays must be same length")

        for i in range(self.nblocks):
            if i > 0:
                if blocs[i] <= blocs[i - 1]:
                    raise ValueError("Locations not in ascending order")

            if i < self.nblocks - 1:
                if blocs[i] + blengths[i] > blocs[i + 1]:
                    raise ValueError(f"Block {i} overlaps")
            else:
                if blocs[i] + blengths[i] > self.length:
                    raise ValueError(f"Block {i} extends beyond end")

            # no zero-length blocks
            if blengths[i] == 0:
                raise ValueError(f"Zero-length block {i}")

    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # True if other is a BlockIndex with the same length, blocs and blengths.
        # Inputs: other - any. Outputs: bool. Side effects: none.
        if not isinstance(other, BlockIndex):
            return False

        if self is other:
            return True

        same_length = self.length == other.length
        same_blocks = (np.array_equal(self.blocs, other.blocs) and
                       np.array_equal(self.blengths, other.blengths))
        return same_length and same_blocks

    def to_block_index(self):
        # [FUNCTION DOC]
        # Return self (already a BlockIndex).
        return self

    cpdef to_int_index(self):
        # [FUNCTION DOC]
        # Expand blocks into an IntIndex listing every stored position.
        # Outputs: new IntIndex (validated). Side effects: none.
        cdef:
            int32_t i = 0, j, b
            int32_t offset
            ndarray[int32_t, ndim=1] indices

        indices = np.empty(self.npoints, dtype=np.int32)

        for b in range(self.nblocks):
            offset = self.locbuf[b]

            for j in range(self.lenbuf[b]):
                indices[i] = offset + j
                i += 1

        return IntIndex(self.length, indices)

    @property
    def indices(self):
        # [FUNCTION DOC]
        # Property: int32 ndarray of all stored positions (via to_int_index).
        return self.to_int_index().indices

    cpdef BlockIndex intersect(self, SparseIndex other):
        # [FUNCTION DOC]
        # Return the BlockIndex of positions stored in both self and other (other
        # converted to BlockIndex). Two-pointer sweep over the blocks: whichever block
        # starts later defines the overlap start; if the earlier block ends before it,
        # skip the earlier block; otherwise emit the overlap up to the earlier of the
        # two block ends and advance the block that ends first (x on ties).
        # Inputs: other - SparseIndex. Outputs: new BlockIndex.
        # Exceptions: Exception (generic) if lengths differ; IndexError if output
        # exceeds the preallocated buffer (see KNOWN ISSUE).
        # Side effects: none.
        # KNOWN ISSUE: the output buffer is sized length // 2 + 1, assuming blocks are
        # separated by gaps, but adjacent blocks are allowed, so e.g. ten adjacent
        # length-1 blocks intersected with themselves (length 10) raises IndexError.
        """
        Intersect two BlockIndex objects

        Returns
        -------
        BlockIndex
        """
        cdef:
            BlockIndex y
            ndarray[int32_t, ndim=1] xloc, xlen, yloc, ylen, out_bloc, out_blen
            Py_ssize_t xi = 0, yi = 0, max_len, result_indexer = 0
            int32_t cur_loc, cur_length, diff

        y = other.to_block_index()

        if self.length != y.length:
            raise Exception("Indices must reference same underlying length")

        xloc = self.blocs
        xlen = self.blengths
        yloc = y.blocs
        ylen = y.blengths

        # block may be split, but can't exceed original len / 2 + 1
        max_len = min(self.length, y.length) // 2 + 1
        out_bloc = np.empty(max_len, dtype=np.int32)
        out_blen = np.empty(max_len, dtype=np.int32)

        while True:
            # we are done (or possibly never began)
            if xi >= self.nblocks or yi >= y.nblocks:
                break

            # completely symmetric...would like to avoid code dup but oh well
            if xloc[xi] >= yloc[yi]:
                cur_loc = xloc[xi]
                diff = xloc[xi] - yloc[yi]

                if ylen[yi] <= diff:
                    # have to skip this block
                    yi += 1
                    continue

                if ylen[yi] - diff < xlen[xi]:
                    # take end of y block, move onward
                    cur_length = ylen[yi] - diff
                    yi += 1
                else:
                    # take end of x block
                    cur_length = xlen[xi]
                    xi += 1

            else:  # xloc[xi] < yloc[yi]
                cur_loc = yloc[yi]
                diff = yloc[yi] - xloc[xi]

                if xlen[xi] <= diff:
                    # have to skip this block
                    xi += 1
                    continue

                if xlen[xi] - diff < ylen[yi]:
                    # take end of x block, move onward
                    cur_length = xlen[xi] - diff
                    xi += 1
                else:
                    # take end of y block
                    cur_length = ylen[yi]
                    yi += 1

            out_bloc[result_indexer] = cur_loc
            out_blen[result_indexer] = cur_length
            result_indexer += 1

        out_bloc = out_bloc[:result_indexer]
        out_blen = out_blen[:result_indexer]

        return BlockIndex(self.length, out_bloc, out_blen)

    cpdef BlockIndex make_union(self, SparseIndex y):
        # [FUNCTION DOC]
        # Return the BlockIndex of positions stored in either index, computed by
        # BlockUnion (other converted to BlockIndex; overlapping or touching blocks are
        # merged). Inputs: y - SparseIndex. Outputs: new BlockIndex.
        # Exceptions: Exception if lengths differ; IndexError on buffer overflow (see
        # BlockUnion). Side effects: none.
        """
        Combine together two BlockIndex objects, accepting indices if contained
        in one or the other

        Parameters
        ----------
        other : SparseIndex

        Notes
        -----
        union is a protected keyword in Cython, hence make_union

        Returns
        -------
        BlockIndex
        """
        return BlockUnion(self, y.to_block_index()).result

    cpdef Py_ssize_t lookup(self, Py_ssize_t index):
        # [FUNCTION DOC]
        # Map a logical position to its offset among stored values by linear scan over
        # blocks, accumulating block lengths. Inputs: index - logical position.
        # Outputs: offset or -1 if not stored. Side effects: none.
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
        cdef:
            Py_ssize_t i, cum_len
            ndarray[int32_t, ndim=1] locs, lens

        locs = self.blocs
        lens = self.blengths

        if self.nblocks == 0:
            return -1
        elif index < locs[0]:
            return -1

        cum_len = 0
        for i in range(self.nblocks):
            if index >= locs[i] and index < locs[i] + lens[i]:
                return cum_len + index - locs[i]
            cum_len += lens[i]

        return -1

    @cython.wraparound(False)
    cpdef ndarray[int32_t] lookup_array(self, ndarray[int32_t, ndim=1] indexer):
        # [FUNCTION DOC]
        # Vectorized lookup: for each position in indexer within [0, length), linearly
        # scans all blocks to find the containing block and computes its offset; -1 if
        # not stored or out of range. O(n * nblocks).
        # Inputs: indexer - 1-D int32 ndarray. Outputs: int32 ndarray. Side effects:
        # none.
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
        cdef:
            Py_ssize_t n, i, j, ind_val
            ndarray[int32_t, ndim=1] locs, lens
            ndarray[int32_t, ndim=1] results

        locs = self.blocs
        lens = self.blengths

        n = len(indexer)
        results = np.empty(n, dtype=np.int32)
        results[:] = -1

        if self.npoints == 0:
            return results

        for i in range(n):
            ind_val = indexer[i]
            if not (ind_val < 0 or self.length <= ind_val):
                cum_len = 0
                for j in range(self.nblocks):
                    if ind_val >= locs[j] and ind_val < locs[j] + lens[j]:
                        results[i] = cum_len + ind_val - locs[j]
                    cum_len += lens[j]
        return results


@cython.internal
cdef class BlockMerge:
    # [CLASS DOC]
    # Internal base class for merging two BlockIndex objects of equal length. Holds x,
    # y, start arrays (blocs) and exclusive end arrays (blocs + blengths) for each side,
    # current block cursors xi/yi, and result, computed in __init__ by the subclass's
    # _make_merged_blocks. xlen/ylen are declared but unused.
    """
    Object-oriented approach makes sharing state between recursive functions a
    lot easier and reduces code duplication
    """
    cdef:
        BlockIndex x, y, result
        ndarray xstart, xlen, xend, ystart, ylen, yend
        int32_t xi, yi  # block indices

    def __init__(self, BlockIndex x, BlockIndex y):
        # [FUNCTION DOC]
        # Store both indices, precompute block starts and ends, reset cursors, and
        # compute self.result. Inputs: x, y - BlockIndex. Outputs: None.
        # Exceptions: Exception if lengths differ; NotImplementedError when used on the
        # base class. Side effects: mutates self.
        self.x = x
        self.y = y

        if x.length != y.length:
            raise Exception("Indices must reference same underlying length")

        self.xstart = self.x.blocs
        self.ystart = self.y.blocs

        self.xend = self.x.blocs + self.x.blengths
        self.yend = self.y.blocs + self.y.blengths

        # self.xlen = self.x.blengths
        # self.ylen = self.y.blengths

        self.xi = 0
        self.yi = 0

        self.result = self._make_merged_blocks()

    cdef _make_merged_blocks(self):
        # [FUNCTION DOC]
        # HELPER: abstract; subclasses return the merged BlockIndex. Always raises
        # NotImplementedError here.
        raise NotImplementedError

    cdef _set_current_indices(self, int32_t xi, int32_t yi, bint mode):
        # [FUNCTION DOC]
        # HELPER: Set cursors; mode 0 assigns (xi, yi) directly, mode 1 means the
        # caller's roles were swapped so it assigns xi to self.yi and yi to self.xi.
        # Inputs: xi, yi - block cursors; mode - 0 or 1. Outputs: None.
        # Side effects: mutates self.xi / self.yi.
        if mode == 0:
            self.xi = xi
            self.yi = yi
        else:
            self.xi = yi
            self.yi = xi


@cython.internal
cdef class BlockUnion(BlockMerge):
    # [CLASS DOC]
    # BlockMerge subclass computing the union of two BlockIndex objects: the result
    # contains every position stored in either, with overlapping or touching (end ==
    # next start) blocks merged into one.
    """
    Object-oriented approach makes sharing state between recursive functions a
    lot easier and reduces code duplication
    """

    cdef _make_merged_blocks(self):
        # [FUNCTION DOC]
        # HELPER: Produce the union blocks. Repeatedly: if both sides are exhausted
        # stop; if one is exhausted, copy the other's next block; otherwise start a new
        # block at the earlier start and find its end with _find_next_block_end.
        # Outputs: new BlockIndex. Exceptions: IndexError if more blocks are produced
        # than the buffer of size min(lengths) // 2 + 1.
        # Side effects: advances self.xi / self.yi.
        # KNOWN ISSUE: the buffer size assumes gaps between blocks; adjacent blocks are
        # valid input (e.g. ten adjacent length-1 blocks unioned with an empty index of
        # length 10) and overflow it, raising IndexError.
        cdef:
            ndarray[int32_t, ndim=1] xstart, xend, ystart
            ndarray[int32_t, ndim=1] yend, out_bloc, out_blen
            int32_t nstart, nend
            Py_ssize_t max_len, result_indexer = 0

        xstart = self.xstart
        xend = self.xend
        ystart = self.ystart
        yend = self.yend

        max_len = min(self.x.length, self.y.length) // 2 + 1
        out_bloc = np.empty(max_len, dtype=np.int32)
        out_blen = np.empty(max_len, dtype=np.int32)

        while True:
            # we are done (or possibly never began)
            if self.xi >= self.x.nblocks and self.yi >= self.y.nblocks:
                break
            elif self.yi >= self.y.nblocks:
                # through with y, just pass through x blocks
                nstart = xstart[self.xi]
                nend = xend[self.xi]
                self.xi += 1
            elif self.xi >= self.x.nblocks:
                # through with x, just pass through y blocks
                nstart = ystart[self.yi]
                nend = yend[self.yi]
                self.yi += 1
            else:
                # find end of new block
                if xstart[self.xi] < ystart[self.yi]:
                    nstart = xstart[self.xi]
                    nend = self._find_next_block_end(0)
                else:
                    nstart = ystart[self.yi]
                    nend = self._find_next_block_end(1)

            out_bloc[result_indexer] = nstart
            out_blen[result_indexer] = nend - nstart
            result_indexer += 1

        out_bloc = out_bloc[:result_indexer]
        out_blen = out_blen[:result_indexer]

        return BlockIndex(self.x.length, out_bloc, out_blen)

    cdef int32_t _find_next_block_end(self, bint mode) except -1:
        # [FUNCTION DOC]
        # HELPER: Starting from the current block of side "x" (mode 0: self.x, mode 1:
        # self.y, roles swapped otherwise), compute the end of the merged block: take
        # that block's end; skip blocks of the other side that end before it; if the
        # next other-side block starts at or before that end, the merged block continues
        # and the search recurses with roles swapped; otherwise return the end. Cursor
        # positions are updated via _set_current_indices as blocks are consumed.
        # Inputs: mode - 0 or 1. Outputs: exclusive end position of the merged block.
        # Exceptions: Exception if mode is not 0/1. Side effects: mutates self.xi /
        # self.yi; recursion depth grows with the number of chained overlapping blocks.
        """
        Wow, this got complicated in a hurry

        mode 0: block started in index x
        mode 1: block started in index y
        """
        cdef:
            ndarray[int32_t, ndim=1] xstart, xend, ystart, yend
            int32_t xi, yi, ynblocks, nend

        if mode != 0 and mode != 1:
            raise Exception("Mode must be 0 or 1")

        # so symmetric code will work
        if mode == 0:
            xstart = self.xstart
            xend = self.xend
            xi = self.xi

            ystart = self.ystart
            yend = self.yend
            yi = self.yi
            ynblocks = self.y.nblocks
        else:
            xstart = self.ystart
            xend = self.yend
            xi = self.yi

            ystart = self.xstart
            yend = self.xend
            yi = self.xi
            ynblocks = self.x.nblocks

        nend = xend[xi]

        # done with y?
        if yi == ynblocks:
            self._set_current_indices(xi + 1, yi, mode)
            return nend
        elif nend < ystart[yi]:
            # block ends before y block
            self._set_current_indices(xi + 1, yi, mode)
            return nend
        else:
            while yi < ynblocks and nend > yend[yi]:
                yi += 1

            self._set_current_indices(xi + 1, yi, mode)

            if yi == ynblocks:
                return nend

            if nend < ystart[yi]:
                # we're done, return the block end
                return nend
            else:
                # merge blocks, continue searching
                # this also catches the case where blocks
                return self._find_next_block_end(1 - mode)


# -----------------------------------------------------------------------------
# Sparse arithmetic

include "sparse_op_helper.pxi"


# -----------------------------------------------------------------------------
# SparseArray mask create operations

def make_mask_object_ndarray(ndarray[object, ndim=1] arr, object fill_value):
    # [FUNCTION DOC]
    # Build a boolean mask that is False where arr[i] equals fill_value and has exactly
    # the same type, True otherwise (i.e. which values must be stored sparsely).
    # Inputs: arr - 1-D object ndarray; fill_value - object.
    # Outputs: bool ndarray (view of int8) of len(arr).
    # Exceptions: propagates errors from __eq__ (e.g. ambiguous truth value of an array
    # element). Side effects: none.
    # Note: NaN fill values never compare equal, so NaN elements would always be kept;
    # the caller (make_sparse) handles NA fill values with notna instead.
    cdef:
        object value
        Py_ssize_t i
        Py_ssize_t new_length = len(arr)
        ndarray[int8_t, ndim=1] mask

    mask = np.ones(new_length, dtype=np.int8)

    for i in range(new_length):
        value = arr[i]
        if value == fill_value and type(value) is type(fill_value):
            mask[i] = 0

    return mask.view(dtype=bool)
