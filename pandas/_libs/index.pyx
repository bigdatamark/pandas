# [MODULE DOC]
# Index engines: the lookup machinery behind pandas Index objects. An engine wraps
# the index's backing array and answers get_loc (label -> integer position, slice
# or boolean mask when labels repeat), get_indexer (array of labels -> positions,
# -1 when missing) and get_indexer_non_unique, plus cached properties is_unique,
# is_monotonic_increasing/decreasing. IndexEngine (numpy-backed) lazily builds a
# dtype-specific hash table mapping value -> position; for very large monotonic
# indexes (>= _SIZE_CUTOFF = 1,000,000 elements) it uses binary search instead of
# populating the hash table. Typed subclasses for numeric dtypes come from the
# included template index_class_helper.pxi; this module also defines object,
# string, datetime/timedelta/period, bool, masked (nullable) and extension-array
# engines, and the MultiIndex codes engine base which packs each row's level
# codes into one integer (codes shifted by multiindex_nulls_shift = 2 so that
# missing codes -1 become 1 and 0 can mean "not found").
cimport cython
from cpython.sequence cimport PySequence_GetItem

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
)

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.hashtable cimport HashTable
from pandas._libs.tslibs.nattype cimport c_NaT as NaT
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    get_unit_from_dtype,
    import_pandas_datetime,
)

import_pandas_datetime()


from pandas._libs.tslibs.period cimport is_period_object
from pandas._libs.tslibs.timedeltas cimport _Timedelta
from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs import (
    algos,
    hashtable as _hash,
)

from pandas._libs.lib cimport eq_NA_compat
from pandas._libs.missing cimport (
    C_NA,
    checknull,
    is_matching_na,
)

from decimal import InvalidOperation

# Defines shift of MultiIndex codes to avoid negative codes (missing values)
multiindex_nulls_shift = 2


cdef bint is_definitely_invalid_key(object val):
    # [FUNCTION DOC]
    # HELPER: Decide whether a key can never be a valid label.
    # Inputs: val - any object.
    # Outputs: true if hashing val raises TypeError (unhashable), else false.
    try:
        hash(val)
    except TypeError:
        return True
    return False


cdef ndarray _get_bool_indexer(ndarray values, object val, ndarray mask = None):
    # [FUNCTION DOC]
    # HELPER: Boolean mask of positions whose value matches val.
    # Inputs: values - 1-D ndarray; val - scalar already validated by _check_type;
    # mask - optional boolean NA mask (only for non-object values).
    # Outputs: boolean ndarray. Object dtype: elementwise NA-compatible equality if val
    # is not null, else positions holding a "matching" NA (same kind of missing value).
    # Otherwise: with mask, the mask itself when val is the NA singleton, else
    # equality with val excluding masked entries; without mask, isnan(values) for a
    # NaN val, else elementwise equality.
    # Exceptions: AssertionError if a mask is given for object values.
    """
    Return an ndarray[bool] of locations where val matches self.values.

    If val is not NA, this is equivalent to `self.values == val`
    """
    # Caller is responsible for ensuring _check_type has already been called
    cdef:
        ndarray[uint8_t, ndim=1, cast=True] indexer
        Py_ssize_t i
        object item

    if values.descr.type_num == cnp.NPY_OBJECT:
        assert mask is None  # no mask for object dtype
        # i.e. values.dtype == object
        if not checknull(val):
            indexer = eq_NA_compat(values, val)

        else:
            # We need to check for _matching_ NA values
            indexer = np.empty(len(values), dtype=np.uint8)

            for i in range(len(values)):
                item = PySequence_GetItem(values, i)
                indexer[i] = is_matching_na(item, val)

    else:
        if mask is not None:
            if val is C_NA:
                indexer = mask == 1
            else:
                indexer = (values == val) & ~mask
        else:
            if util.is_nan(val):
                indexer = np.isnan(values)
            else:
                indexer = values == val

    return indexer.view(bool)


cdef _maybe_resize_array(ndarray values, Py_ssize_t loc, Py_ssize_t max_length):
    # [FUNCTION DOC]
    # HELPER: Grow an array so that index loc is valid.
    # Inputs: values - ndarray; loc - required index; max_length - upper bound.
    # Outputs: values unchanged if loc < len, else a resized copy whose length is the
    # current length doubled until it exceeds loc, capped at max_length.
    # KNOWN ISSUE: if values is empty (length 0) and loc >= 0 the doubling loop never
    # terminates (0 * 2 stays 0); this is reachable from MaskedIndexEngine
    # .get_indexer_non_unique on an empty index with non-empty targets.
    """
    Resize array if loc is out of bounds.
    """
    cdef:
        Py_ssize_t n = len(values)

    if loc >= n:
        while loc >= n:
            n *= 2
        values = np.resize(values, min(n, max_length))
    return values


# Don't populate hash tables in monotonic indexes larger than this
_SIZE_CUTOFF = 1_000_000


cdef _unpack_bool_indexer(ndarray[uint8_t, ndim=1, cast=True] indexer, object val):
    # [FUNCTION DOC]
    # HELPER: Reduce a boolean match mask to the most compact location.
    # Inputs: indexer - boolean (uint8) 1-D array; val - key, used in the error.
    # Outputs: the mask itself if more than one match, the integer position if
    # exactly one.
    # Exceptions: KeyError(val) if there are no matches.
    """
    Possibly unpack a boolean mask to a single indexer.
    """
    # Returns ndarray[bool] or int
    cdef:
        ndarray[intp_t, ndim=1] found
        int count

    found = np.where(indexer)[0]
    count = len(found)

    if count > 1:
        return indexer
    if count == 1:
        return int(found[0])

    raise KeyError(val)


@cython.freelist(32)
cdef class IndexEngine:
    # [CLASS DOC]
    # Base engine for numpy-backed indexes. Read-only fields: values - backing array;
    # mask - NA mask or None; mapping - hash table (None until populated);
    # over_size_threshold - len(values) >= _SIZE_CUTOFF. Private state: unique,
    # monotonic_inc, monotonic_dec flags and need_unique_check/need_monotonic_check
    # flags marking whether those are stale; _np_type - numpy scalar type of values.
    # Subclasses provide _make_hash_table and _check_type. Uses a freelist of 32.

    cdef readonly:
        ndarray values
        ndarray mask
        HashTable mapping
        bint over_size_threshold

    cdef:
        bint unique, monotonic_inc, monotonic_dec
        bint need_monotonic_check, need_unique_check
        object _np_type

    def __init__(self, ndarray values):
        # [FUNCTION DOC]
        # Initialize the engine.
        # Inputs: values - 1-D ndarray of labels.
        # Outputs: none.
        # Side effects: stores values (no copy), mask = None, computes the size
        #   threshold
        # flag, resets all caches via clear_mapping, records the scalar type.
        self.values = values
        self.mask = None

        self.over_size_threshold = len(values) >= _SIZE_CUTOFF
        self.clear_mapping()
        self._np_type = values.dtype.type

    def __contains__(self, val: object) -> bool:
        # [FUNCTION DOC]
        # Membership test.
        # Inputs: val - label.
        # Outputs: True if get_loc succeeds, False on KeyError.
        # Exceptions: TypeError if val is unhashable (from hash) or invalid for get_loc.
        hash(val)
        try:
            self.get_loc(val)
        except KeyError:
            return False
        return True

    cpdef get_loc(self, object val):
        # [FUNCTION DOC]
        # Locate a label.
        # Inputs: val - label.
        # Outputs: integer position if unique; for duplicates, an integer (single
        #   match),
        # a slice (monotonic increasing index) or a boolean mask.
        # Algorithm: reject unhashable keys; validate/coerce with _check_type. If large
        #   and
        # monotonic increasing: duplicates go to _get_loc_duplicates; otherwise binary
        # search and verify equality. Else populate the hash table; duplicates go to
        # _get_loc_duplicates; NA for masked engines uses the table's NA slot; otherwise
        # hash table lookup.
        # Exceptions: TypeError for unhashable keys; KeyError if not found, if the key
        #   is
        # incompatible with the dtype, or on integer overflow during lookup.
        # Side effects: may compute and cache monotonic/unique flags and the hash table.
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        val = self._check_type(val)

        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(val)
            values = self.values

            loc = self._searchsorted_left(val)
            if loc >= len(values):
                raise KeyError(val)
            if values[loc] != val:
                raise KeyError(val)
            return loc

        self._ensure_mapping_populated()
        if not self.unique:
            return self._get_loc_duplicates(val)
        if self.mask is not None and val is C_NA:
            return self.mapping.get_na()

        try:
            return self.mapping.get_item(val)
        except OverflowError as err:
            # GH#41775 OverflowError e.g. if we are uint64 and val is -1
            #  or if we are int64 and value is np.iinfo(np.int64).max+1
            #  (the uint64 with -1 case should actually be excluded by _check_type)
            raise KeyError(val) from err

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        # [FUNCTION DOC]
        # HELPER: Leftmost insertion position of val in sorted values.
        # Inputs: val - key already validated by _check_type (converted to the values'
        # scalar type).
        # Outputs: position (0..len).
        """
        See ObjectEngine._searchsorted_left.__doc__.
        """
        # Caller is responsible for ensuring _check_type has already been called
        loc = self.values.searchsorted(self._np_type(val), side="left")
        return loc

    cdef _get_loc_duplicates(self, object val):
        # [FUNCTION DOC]
        # HELPER: Locate a label in an index known to contain duplicates.
        # Inputs: val - validated key.
        # Outputs: if monotonic increasing: the integer position for one match or a
        #   slice
        # for a run of matches (found by left/right binary search); otherwise the result
        # of _maybe_get_bool_indexer (int or boolean mask).
        # Exceptions: KeyError if not found or if values cannot be compared with val.
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t diff, left, right

        if self.is_monotonic_increasing:
            values = self.values
            try:
                left = values.searchsorted(val, side="left")
                right = values.searchsorted(val, side="right")
            except TypeError:
                # e.g. GH#29189 get_loc(None) with a Float64Index
                #  2021-09-29 Now only reached for object-dtype
                raise KeyError(val)

            diff = right - left
            if diff == 0:
                raise KeyError(val)
            elif diff == 1:
                return left
            else:
                return slice(left, right)

        return self._maybe_get_bool_indexer(val)

    cdef _maybe_get_bool_indexer(self, object val):
        # [FUNCTION DOC]
        # HELPER: Compute the boolean match mask and reduce it (int if exactly one
        #   match).
        # Inputs: val - validated key.
        # Outputs: int or boolean ndarray. Exceptions: KeyError if no match.
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer

        indexer = _get_bool_indexer(self.values, val, self.mask)
        return _unpack_bool_indexer(indexer, val)

    def sizeof(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # Memory used by the hash table.
        # Inputs: deep - passed to the table's sizeof.
        # Outputs: bytes as int; 0 if the table has not been built.
        """ return the sizeof our mapping """
        if not self.is_mapping_populated:
            return 0
        return self.mapping.sizeof(deep=deep)

    def __sizeof__(self) -> int:
        # [FUNCTION DOC]
        # sys.getsizeof hook returning sizeof() (shallow table size).
        return self.sizeof()

    cpdef _update_from_sliced(self, IndexEngine other, reverse: bool):
        # [FUNCTION DOC]
        # Copy cached uniqueness/monotonicity from the engine of a parent index after
        #   the
        # index was sliced (a subset of a unique index is unique; a slice of a monotonic
        # index is monotonic).
        # Inputs: other - parent IndexEngine; reverse - True if the slice reversed
        #   order.
        # Outputs: none.
        # Side effects: if other is unique, marks self unique. If other's monotonic
        #   state
        # is known and monotonic in either direction, marks self's as known: when the
        # first and last values differ, copies the directions (swapped if reversed);
        # otherwise (empty or all-equal endpoints) marks both directions true.
        if other.unique:
            self.unique = other.unique
            self.need_unique_check = other.need_unique_check

        if not other.need_monotonic_check and (
                other.is_monotonic_increasing or other.is_monotonic_decreasing):
            self.need_monotonic_check = 0
            if len(self.values) > 0 and self.values[0] != self.values[-1]:
                # reverse=True means the index has been reversed
                if reverse:
                    self.monotonic_inc = other.monotonic_dec
                    self.monotonic_dec = other.monotonic_inc
                else:
                    self.monotonic_inc = other.monotonic_inc
                    self.monotonic_dec = other.monotonic_dec
            else:
                self.monotonic_inc = 1
                self.monotonic_dec = 1

    @property
    @cython.critical_section
    def is_unique(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether all labels are distinct.
        # Outputs: bool. Triggers the monotonic check first if stale (it may determine
        # uniqueness cheaply), then populates the hash table if still unknown.
        # Side effects: caches flags and possibly the hash table (critical section).
        # for why we check is_monotonic_increasing here, see
        # https://github.com/pandas-dev/pandas/pull/55342#discussion_r1361405781
        if self.need_monotonic_check:
            self.is_monotonic_increasing
        if self.need_unique_check:
            self._do_unique_check()

        return self.unique == 1

    cdef _do_unique_check(self):
        # [FUNCTION DOC]
        # HELPER: Determine uniqueness by building the hash table (which records whether
        # its size equals the number of values).
        # Side effects: populates mapping and the unique flag.
        self._ensure_mapping_populated()

    @property
    @cython.critical_section
    def is_monotonic_increasing(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether values are monotonic non-decreasing.
        # Outputs: bool. Side effects: runs and caches the monotonic check if stale.
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    @cython.critical_section
    def is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether values are monotonic non-increasing.
        # Outputs: bool. Side effects: runs and caches the monotonic check if stale.
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_dec == 1

    cdef _do_monotonic_check(self):
        # [FUNCTION DOC]
        # HELPER: Compute monotonicity flags.
        # Inputs: none.
        # Outputs: none.
        # Side effects: if any value is masked as NA, sets both flags false (and, note,
        # leaves need_monotonic_check set so it is recomputed on each access). Otherwise
        # calls _call_monotonic (comparison errors -> not monotonic), clears the stale
        # flag, and if strictly monotonic marks unique; if non-strictly monotonic marks
        # non-unique; both make the uniqueness check unnecessary.
        cdef:
            bint is_strict_monotonic
        if self.mask is not None and np.any(self.mask):
            self.monotonic_inc = 0
            self.monotonic_dec = 0
        else:
            try:
                values = self.values
                self.monotonic_inc, self.monotonic_dec, is_strict_monotonic = \
                    self._call_monotonic(values)
            except (TypeError, InvalidOperation, ValueError):
                self.monotonic_inc = 0
                self.monotonic_dec = 0
                is_strict_monotonic = 0

            self.need_monotonic_check = 0

            # we can only be sure of uniqueness if is_strict_monotonic=1
            if is_strict_monotonic:
                self.unique = 1
                self.need_unique_check = 0
            elif self.monotonic_inc == 1 or self.monotonic_dec == 1:
                self.unique = 0
                self.need_unique_check = 0

    cdef _call_monotonic(self, values):
        # [FUNCTION DOC]
        # HELPER: Compute (is_increasing, is_decreasing, is_strict) via
        #   algos.is_monotonic
        # with timelike=False (NaT not special).
        # Inputs: values - ndarray.
        return algos.is_monotonic(values, timelike=False)

    cdef _make_hash_table(self, Py_ssize_t n):
        # [FUNCTION DOC]
        # HELPER: Abstract factory for the hash table; subclasses return a dtype-
        #   specific
        # table sized for n elements.
        # Exceptions: NotImplementedError in the base class.
        raise NotImplementedError  # pragma: no cover

    cdef _check_type(self, object val):
        # [FUNCTION DOC]
        # HELPER: Validate a lookup key; the base version only checks hashability.
        # Inputs: val - key. Outputs: val unchanged.
        # Exceptions: TypeError if unhashable.
        hash(val)
        return val

    @property
    @cython.critical_section
    def is_mapping_populated(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether the hash table has been built.
        # Outputs: bool.
        return self.mapping is not None

    @cython.critical_section
    cdef _ensure_mapping_populated(self):
        # [FUNCTION DOC]
        # HELPER: Build the hash table if missing.
        # Inputs: none.
        # Outputs: none.
        # Side effects: creates a table sized for len(values), maps each value to its
        # position (honouring mask), sets mapping, unique (table size equals length) and
        # clears need_unique_check. Runs under a critical section, re-checking after the
        # build because the section may be released.
        # this populates the mapping
        # if its not already populated
        # also satisfies the need_unique_check
        if not self.is_mapping_populated:
            values = self.values
            mapping = self._make_hash_table(len(values))
            mapping.map_locations(values, self.mask)
            unique = len(mapping) == len(values)
        # check again after creating the mapping
        # which could have released the critical_section
        if not self.is_mapping_populated:
            self.mapping = mapping
            self.unique = unique
            self.need_unique_check = 0

    def clear_mapping(self):
        # [FUNCTION DOC]
        # Reset all caches: drop the hash table, mark uniqueness and monotonicity stale
        #   and
        # set the flags to false.
        self.mapping = None
        self.need_monotonic_check = 1
        self.need_unique_check = 1

        self.unique = 0
        self.monotonic_inc = 0
        self.monotonic_dec = 0

    def get_indexer(self, ndarray values) -> np.ndarray:
        # [FUNCTION DOC]
        # Positions of target labels (assumes the index is unique).
        # Inputs: values - ndarray of targets with compatible dtype.
        # Outputs: intp array, -1 where not found.
        # Side effects: builds the hash table if needed.
        self._ensure_mapping_populated()
        return self.mapping.lookup(values)

    def get_indexer_non_unique(self, ndarray targets):
        # [FUNCTION DOC]
        # Positions of target labels allowing duplicates in the index.
        # Inputs: targets - 1-D ndarray (generally object or same dtype as values).
        # Outputs: tuple (indexer, missing). indexer lists, for each target in order,
        #   all
        # matching index positions in increasing order, or -1 if none; missing lists the
        # target positions with no match. Both intp arrays.
        # Algorithm: builds a dict from distinct target value to list of positions. If
        # there are few distinct targets (fewer than n / (2 * bit_length(n))), none are
        # null and the index is monotonic increasing, uses binary search per target
        # (targets that raise TypeError fall back to scanning). Otherwise scans all
        # values once; null values are matched to null targets with is_matching_na, with
        # one representative per kind of NA. Output array grows by doubling, capped at
        # len(values) * len(targets).
        # Side effects: may compute monotonic flags; allocates arrays.
        # KNOWN ISSUE: with an empty index (n = 0) and non-empty targets, the fast-path
        # condition divides 0 by 2 * bit_length(0) = 0 and raises ZeroDivisionError.
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results

        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        cdef:
            ndarray values
            ndarray[intp_t] result, missing
            set stargets, remaining_stargets, found_nas
            dict d = {}
            object val
            Py_ssize_t count = 0, count_missing = 0
            Py_ssize_t i, j, n, n_t, n_alloc, max_alloc, start, end
            bint check_na_values = False

        values = self.values
        stargets = set(targets)

        na_in_stargets = any(checknull(t) for t in stargets)

        n = len(values)
        n_t = len(targets)
        max_alloc = n * n_t
        if n > 10_000:
            n_alloc = 10_000
        else:
            n_alloc = n

        result = np.empty(n_alloc, dtype=np.intp)
        missing = np.empty(n_t, dtype=np.intp)

        # map each starget to its position in the index
        if (
                stargets and
                len(stargets) < (n / (2 * n.bit_length())) and
                not na_in_stargets and
                self.is_monotonic_increasing
        ):
            # if there are few enough stargets and the index is monotonically
            # increasing, then use binary search for each starget
            remaining_stargets = set()
            for starget in stargets:
                try:
                    start = values.searchsorted(starget, side="left")
                    end = values.searchsorted(starget, side="right")
                except TypeError:  # e.g. if we tried to search for string in int array
                    remaining_stargets.add(starget)
                else:
                    if start != end:
                        d[starget] = list(range(start, end))

            stargets = remaining_stargets

        if stargets:
            # otherwise, map by iterating through all items in the index

            # short-circuit na check
            if na_in_stargets:
                check_na_values = True
                # keep track of nas in values
                found_nas = set()

            for i in range(n):
                val = PySequence_GetItem(values, i)

                # GH#43870
                # handle lookup for nas
                # (ie. np.nan, float("NaN"), Decimal("NaN"), dt64nat, td64nat)
                if check_na_values and checknull(val):
                    match = [na for na in found_nas if is_matching_na(val, na)]

                    # matching na not found
                    if not len(match):
                        found_nas.add(val)

                        # add na to stargets to utilize `in` for stargets/d lookup
                        match_stargets = [
                            x for x in stargets if is_matching_na(val, x)
                        ]

                        if len(match_stargets):
                            # add our 'standardized' na
                            stargets.add(val)

                    # matching na found
                    else:
                        assert len(match) == 1
                        val = match[0]

                if val in stargets:
                    if val not in d:
                        d[val] = []
                    d[val].append(i)

        for i in range(n_t):
            val = PySequence_GetItem(targets, i)

            # ensure there are nas in values before looking for a matching na
            if check_na_values and checknull(val):
                match = [na for na in found_nas if is_matching_na(val, na)]
                if len(match):
                    assert len(match) == 1
                    val = match[0]

            # found
            if val in d:
                key = val
                result = _maybe_resize_array(
                    result,
                    count + len(d[key]) - 1,
                    max_alloc
                )
                for j in d[key]:
                    result[count] = j
                    count += 1

            # value not found
            else:
                result = _maybe_resize_array(result, count, max_alloc)
                result[count] = -1
                count += 1
                missing[count_missing] = i
                count_missing += 1

        return result[0:count], missing[0:count_missing]


cdef Py_ssize_t _bin_search(ndarray values, object val) except -1:
    # [FUNCTION DOC]
    # HELPER: Leftmost insertion position (like bisect_left) for an object array,
    # treating a tuple key as a single value.
    # Inputs: values - sorted 1-D object ndarray; val - key.
    # Outputs: position in 0..len(values).
    # Exceptions: TypeError if comparisons fail; IndexError if values is empty.
    # KNOWN ISSUE: for a 1-element array (hi == 0) it always returns 1, even when val
    # is <= the only element (bisect_left would give 0). Only reached for indexes
    # above the size cutoff, so it is not hit in practice.
    # GH#1757 ndarray.searchsorted is not safe to use with array of tuples
    #  (treats a tuple `val` as a sequence of keys instead of a single key),
    #  so we implement something similar.
    # This is equivalent to the stdlib's bisect.bisect_left

    cdef:
        Py_ssize_t mid = 0, lo = 0, hi = len(values) - 1
        object pval

    if hi == 0 or (hi > 0 and val > PySequence_GetItem(values, hi)):
        return len(values)

    while lo < hi:
        mid = (lo + hi) // 2
        pval = PySequence_GetItem(values, mid)
        if val < pval:
            hi = mid
        elif val > pval:
            lo = mid + 1
        else:
            while mid > 0 and val == PySequence_GetItem(values, mid - 1):
                mid -= 1
            return mid

    if val <= PySequence_GetItem(values, mid):
        return mid
    else:
        return mid + 1


cdef class ObjectEngine(IndexEngine):
    # [CLASS DOC]
    # Engine for object-dtype indexes, using a PyObject hash table and a tuple-safe
    # binary search.
    """
    Index Engine for use with object-dtype Index, namely the base class Index.
    """
    cdef _make_hash_table(self, Py_ssize_t n):
        # [FUNCTION DOC]
        # HELPER: Create a PyObject hash table sized for n.
        return _hash.PyObjectHashTable(n)

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        # [FUNCTION DOC]
        # HELPER: Leftmost insertion position of val using the tuple-safe _bin_search.
        # Exceptions: KeyError if values cannot be compared with val.
        # using values.searchsorted here would treat a tuple `val` as a sequence
        #  instead of a single key, so we use a different implementation
        try:
            loc = _bin_search(self.values, val)
        except TypeError as err:
            raise KeyError(val) from err
        return loc


cdef class StringEngine(IndexEngine):
    # [CLASS DOC]
    # Engine for string-dtype indexes backed by an array of Python strings, using a
    # string hash table.

    cdef _make_hash_table(self, Py_ssize_t n):
        # [FUNCTION DOC]
        # HELPER: Create a string hash table sized for n.
        return _hash.StringHashTable(n)

    cdef _check_type(self, object val):
        # [FUNCTION DOC]
        # HELPER: Validate a key: must be a str.
        # Outputs: str(val). Exceptions: KeyError for non-strings.
        if not isinstance(val, str):
            raise KeyError(val)
        return str(val)

cdef class StringObjectEngine(ObjectEngine):
    # [CLASS DOC]
    # Object engine for string arrays whose missing value is a specific NA marker
    # (na_value); null keys are normalized to that marker.

    cdef:
        object na_value

    def __init__(self, ndarray values, na_value):
        # [FUNCTION DOC]
        # Initialize with the object values and the NA marker used in them.
        # Inputs: values - object ndarray; na_value - missing value marker.
        super().__init__(values)
        self.na_value = na_value

    cdef _check_type(self, object val):
        # [FUNCTION DOC]
        # HELPER: Validate a key: strings pass through; any null becomes na_value.
        # Exceptions: KeyError for other types.
        if isinstance(val, str):
            return val
        elif checknull(val):
            return self.na_value
        else:
            raise KeyError(val)


cdef class DatetimeEngine(Int64Engine):
    # [CLASS DOC]
    # Engine for datetime64 indexes: works on the int64 view of the values; keys are
    # Timestamps/NaT converted to int64 at the index's resolution (_creso).

    cdef:
        NPY_DATETIMEUNIT _creso

    def __init__(self, ndarray values):
        # [FUNCTION DOC]
        # Initialize from a datetime64 (or timedelta64) array.
        # Inputs: values - ndarray with datetime-like dtype.
        # Side effects: stores the int64 view; records its time unit.
        super().__init__(values.view("i8"))
        self._creso = get_unit_from_dtype(values.dtype)

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # [FUNCTION DOC]
        # HELPER: Convert a key to its int64 representation.
        # Inputs: scalar - NaT or Timestamp (tz compatibility checked by the caller).
        # Outputs: NaT's sentinel value, or the Timestamp value converted to the index's
        # resolution (no rounding allowed).
        # Exceptions: TypeError for other types; ValueError if conversion would lose
        # precision or overflow.
        # NB: caller is responsible for ensuring tzawareness compat
        #  before we get here
        if scalar is NaT:
            return NaT._value
        elif isinstance(scalar, _Timestamp):
            if scalar._creso == self._creso:
                return scalar._value
            else:
                # Note: caller is responsible for catching potential ValueError
                #  from _as_creso
                return (
                    (<_Timestamp>scalar)._as_creso(self._creso, round_ok=False)._value
                )
        raise TypeError(scalar)

    def __contains__(self, val: object) -> bool:
        # [FUNCTION DOC]
        # Membership test; key must be hashable.
        # Outputs: False if resolution conversion raises ValueError or get_loc raises
        # KeyError, else True.
        # Exceptions: TypeError from _unbox_scalar for unsupported key types propagates.
        # We assume before we get here:
        #  - val is hashable
        try:
            self._unbox_scalar(val)
        except ValueError:
            return False

        try:
            self.get_loc(val)
            return True
        except KeyError:
            return False

    cdef _call_monotonic(self, values):
        # [FUNCTION DOC]
        # HELPER: algos.is_monotonic with timelike=True (NaT treated as missing).
        return algos.is_monotonic(values, timelike=True)

    cpdef get_loc(self, object val):
        # [FUNCTION DOC]
        # Locate a datetime key.
        # Inputs: val - Timestamp or NaT.
        # Outputs: int, slice or boolean mask, same scheme as IndexEngine.get_loc but
        # using the int64 converted key.
        # Exceptions: TypeError if unhashable; KeyError if not found or not convertible.
        # Side effects: may cache flags/hash table.
        # NB: the caller is responsible for ensuring that we are called
        #  with either a Timestamp or NaT (Timedelta or NaT for TimedeltaEngine)

        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        try:
            conv = self._unbox_scalar(val)
        except (TypeError, ValueError) as err:
            raise KeyError(val) from err

        # Welcome to the spaghetti factory
        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(conv)
            values = self.values

            loc = values.searchsorted(conv, side="left")

            if loc == len(values) or PySequence_GetItem(values, loc) != conv:
                raise KeyError(val)
            return loc

        self._ensure_mapping_populated()
        if not self.unique:
            return self._get_loc_duplicates(conv)

        try:
            return self.mapping.get_item(conv)
        except KeyError:
            raise KeyError(val)


cdef class TimedeltaEngine(DatetimeEngine):
    # [CLASS DOC]
    # Engine for timedelta64 indexes; identical to DatetimeEngine but keys are
    # Timedeltas/NaT.

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # [FUNCTION DOC]
        # HELPER: Convert a NaT or Timedelta key to int64 at the index's resolution.
        # Exceptions: TypeError for other types; ValueError if lossy.
        if scalar is NaT:
            return NaT._value
        elif isinstance(scalar, _Timedelta):
            if scalar._creso == self._creso:
                return scalar._value
            else:
                # Note: caller is responsible for catching potential ValueError
                #  from _as_creso
                return (
                    (<_Timedelta>scalar)._as_creso(self._creso, round_ok=False)._value
                )
        raise TypeError(scalar)


cdef class PeriodEngine(Int64Engine):
    # [CLASS DOC]
    # Engine for Period indexes, operating on int64 ordinals.

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # [FUNCTION DOC]
        # HELPER: Convert a NaT or Period key to its int64 ordinal (frequency assumed to
        # match). Exceptions: TypeError for other types.
        if scalar is NaT:
            return scalar._value
        if is_period_object(scalar):
            # NB: we assume that we have the correct freq here.
            return scalar.ordinal
        raise TypeError(scalar)

    cpdef get_loc(self, object val):
        # [FUNCTION DOC]
        # Locate a Period key.
        # Inputs: val - Period or NaT.
        # Outputs: result of Int64Engine.get_loc on the ordinal.
        # Exceptions: KeyError if not a Period/NaT or not found.
        # NB: the caller is responsible for ensuring that we are called
        #  with either a Period or NaT
        cdef:
            int64_t conv

        try:
            conv = self._unbox_scalar(val)
        except TypeError:
            raise KeyError(val)

        return Int64Engine.get_loc(self, conv)

    cdef _call_monotonic(self, values):
        # [FUNCTION DOC]
        # HELPER: algos.is_monotonic with timelike=True.
        return algos.is_monotonic(values, timelike=True)


cdef class BaseMultiIndexCodesEngine:
    # [CLASS DOC]
    # Base for MultiIndex engines. Each row's level codes (+2 shift) are bit-shifted by
    # per-level offsets and OR-ed into one integer (unsigned 64-bit or Python int, set
    # by the subclass's _codes_dtype); lookups then delegate to an integer engine
    # given by the subclass attribute _base. Attributes: levels, offsets,
    # level_has_nans (per level, whether any code is -1).
    """
    Base class for MultiIndexUIntEngine and MultiIndexPyIntEngine, which
    represent each label in a MultiIndex as an integer, by juxtaposing the bits
    encoding each level, with appropriate offsets.

    For instance: if 3 levels have respectively 3, 6 and 1 possible values,
    then their labels can be represented using respectively 2, 3 and 1 bits,
    as follows:
     _ _ _ _____ _ __ __ __
    |0|0|0| ... |0| 0|a1|a0| -> offset 0 (first level)
     — — — ————— — —— —— ——
    |0|0|0| ... |0|b2|b1|b0| -> offset 2 (bits required for first level)
     — — — ————— — —— —— ——
    |0|0|0| ... |0| 0| 0|c0| -> offset 5 (bits required for first two levels)
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾
    and the resulting unsigned integer representation will be:
     _ _ _ _____ _ __ __ __ __ __ __
    |0|0|0| ... |0|c0|b2|b1|b0|a1|a0|
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾

    Offsets are calculated at initialization, labels are transformed by method
    _codes_to_ints.

    Keys are located by first locating each component against the respective
    level, then locating (the integer representation of) codes.
    """
    def __init__(self, object levels, object labels, ndarray offsets):
        # [FUNCTION DOC]
        # Initialize from MultiIndex levels and codes.
        # Inputs: levels - list of Index; labels - list of integer code arrays (one per
        # level, -1 for missing); offsets - integer array of bit offsets per level.
        # Outputs: none.
        # Side effects: stores levels/offsets, computes level_has_nans, packs codes into
        # integers and initializes the underlying integer engine with them.
        """
        Parameters
        ----------
        levels : list-like of numpy arrays
            Levels of the MultiIndex.
        labels : list-like of numpy arrays of integer dtype
            Labels of the MultiIndex.
        offsets : numpy array of int dtype
            Pre-calculated offsets, one for each level of the index.
        """
        self.levels = levels
        self.offsets = offsets

        # Transform labels in a single array, and add 2 so that we are working
        # with positive integers (-1 for NaN becomes 1). This enables us to
        # differentiate between values that are missing in other and matching
        # NaNs. We will set values that are not found to 0 later:
        codes = np.array(labels).T
        codes += multiindex_nulls_shift  # inplace sum optimisation

        self.level_has_nans = [-1 in lab for lab in labels]

        # Map each codes combination in the index to an integer unambiguously
        # (no collisions possible), based on the "offsets", which describe the
        # number of bits to switch labels for each level:
        lab_ints = self._codes_to_ints(codes)

        # Initialize underlying index (e.g. libindex.UInt64Engine) with
        # integers representing labels: we will use its get_loc and get_indexer
        self._base.__init__(self, lab_ints)

    def _codes_to_ints(self, ndarray codes) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: Pack code combinations into integers.
        # Inputs: codes - 1-D (one key) or 2-D (rows = keys, columns = levels) array of
        # non-negative codes.
        # Outputs: scalar or 1-D array of _codes_dtype: bitwise OR across levels of each
        # code shifted left by its level's offset (order-preserving).
        # Side effects: may modify codes in place if already of _codes_dtype.
        """
        Transform combination(s) of uint in one uint or Python integer (each), in a
        strictly monotonic way (i.e. respecting the lexicographic order of integer
        combinations).

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint
            Combinations of integers (one per row)

        Returns
        -------
        scalar or 1-dimensional array, of dtype _codes_dtype
            Integer(s) representing one combination (each).
        """
        # To avoid overflows, first make sure we are working with the right dtype:
        codes = codes.astype(self._codes_dtype, copy=False)

        # Shift the representation of each level by the pre-calculated number of bits:
        codes <<= self.offsets  # inplace shift optimisation

        # Now sum and OR are in fact interchangeable. This is a simple
        # composition of the (disjunct) significant bits of each level (i.e.
        # each column in "codes") in a single positive integer (per row):
        if codes.ndim == 1:
            # Single key
            return np.bitwise_or.reduce(codes)

        # Multiple keys
        return np.bitwise_or.reduce(codes, axis=1)

    def _extract_level_codes(self, target) -> np.ndarray:
        # [FUNCTION DOC]
        # Convert a target MultiIndex into packed integers comparable with this index.
        # Inputs: target - MultiIndex.
        # Outputs: 1-D array of packed integers.
        # Algorithm: recode target to this index's levels (unknown labels -> -1). For
        # levels containing NaN, target's missing codes map to the NaN position. Then
        # codes are shifted so unknown -> 0 (never matches) and known -> code + 2; where
        # this level has missing codes, target missing codes become 1 so they match.
        """
        Map the requested list of (tuple) keys to their integer representations
        for searching in the underlying integer index.

        Parameters
        ----------
        target : MultiIndex

        Returns
        ------
        int_keys : 1-dimensional array of dtype uint64 or object
            Integers representing one combination each
        """
        level_codes = list(target._recode_for_new_levels(self.levels, copy=True))
        for i, codes in enumerate(level_codes):
            if self.levels[i].hasnans:
                na_index = self.levels[i].isna().nonzero()[0][0]
                codes[target.codes[i] == -1] = na_index
            codes += 1
            codes[codes > 0] += 1
            if self.level_has_nans[i]:
                codes[target.codes[i] == -1] += 1
        return self._codes_to_ints(np.array(level_codes, dtype=self._codes_dtype).T)

    def get_indexer(self, target: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Positions of packed target integers in the index.
        # Inputs: target - array of packed integers.
        # Outputs: intp array, -1 when missing (delegates to the base engine).
        """
        Returns an array giving the positions of each value of `target` in
        `self.values`, where -1 represents a value in `target` which does not
        appear in `self.values`

        Parameters
        ----------
        target : np.ndarray

        Returns
        -------
        np.ndarray[intp_t, ndim=1] of the indexer of `target` into
        `self.values`
        """
        return self._base.get_indexer(self, target)

    def get_loc(self, object key):
        # [FUNCTION DOC]
        # Locate a tuple key.
        # Inputs: key - tuple with one label per level.
        # Outputs: int, slice or boolean mask from the base engine lookup of the packed
        # key (nulls map to code 1, others to level position + 2).
        # Exceptions: TypeError if unhashable; KeyError if not a tuple, a label is not
        #   in
        # its level or the combination is absent; ValueError if the tuple length differs
        # from the number of levels (strict zip).
        if is_definitely_invalid_key(key):
            raise TypeError(f"'{key}' is an invalid key")
        if not isinstance(key, tuple):
            raise KeyError(key)
        try:
            indices = [1 if checknull(v) else lev.get_loc(v) + multiindex_nulls_shift
                       for lev, v in zip(self.levels, key, strict=True)]
        except KeyError:
            raise KeyError(key)

        # Transform indices into single integer:
        lab_int = self._codes_to_ints(np.array(indices, dtype=self._codes_dtype))

        return self._base.get_loc(self, lab_int)

    def get_indexer_non_unique(self, target: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # Non-unique lookup of packed target integers; returns (indexer, missing) from
        #   the
        # base engine.
        indexer = self._base.get_indexer_non_unique(self, target)

        return indexer

    def __contains__(self, val: object) -> bool:
        # [FUNCTION DOC]
        # Membership test for tuple keys.
        # Outputs: True if get_loc succeeds; False on KeyError, TypeError or ValueError.
        # We assume before we get here:
        #  - val is hashable
        # Default __contains__ looks in the underlying mapping, which in this
        # case only contains integer representations.
        try:
            self.get_loc(val)
            return True
        except (KeyError, TypeError, ValueError):
            return False


# Generated from template.
include "index_class_helper.pxi"


cdef class BoolEngine(UInt8Engine):
    # [CLASS DOC]
    # Engine for boolean indexes, stored as uint8.
    cdef _check_type(self, object val):
        # [FUNCTION DOC]
        # HELPER: Validate a key: must be a bool (Python or numpy).
        # Outputs: the key as 0/1 integer. Exceptions: KeyError otherwise.
        if not util.is_bool_object(val):
            raise KeyError(val)
        return <uint8_t>val


cdef class MaskedBoolEngine(MaskedUInt8Engine):
    # [CLASS DOC]
    # Engine for nullable boolean indexes (mask-based), stored as uint8.
    cdef _check_type(self, object val):
        # [FUNCTION DOC]
        # HELPER: Validate a key: NA passes through; otherwise must be a bool.
        # Outputs: NA or 0/1 integer. Exceptions: KeyError otherwise.
        if val is C_NA:
            return val
        if not util.is_bool_object(val):
            raise KeyError(val)
        return <uint8_t>val


@cython.internal
@cython.freelist(32)
cdef class SharedEngine:
    # [CLASS DOC]
    # Internal base for engines over ExtensionArrays (no hash table). Fields: values,
    # over_size_threshold, and the same unique/monotonic flags as IndexEngine.
    # Lookups are done with comparisons / searchsorted on the array itself.
    cdef readonly:
        object values  # ExtensionArray
        bint over_size_threshold

    cdef:
        bint unique, monotonic_inc, monotonic_dec
        bint need_monotonic_check, need_unique_check

    def __contains__(self, val: object) -> bool:
        # [FUNCTION DOC]
        # Membership test (key assumed hashable): True if get_loc succeeds.
        # We assume before we get here:
        #  - val is hashable
        try:
            self.get_loc(val)
            return True
        except KeyError:
            return False

    def clear_mapping(self):
        # [FUNCTION DOC]
        # No-op for compatibility with IndexEngine.
        # for compat with IndexEngine
        pass

    cpdef _update_from_sliced(self, ExtensionEngine other, reverse: bool):
        # [FUNCTION DOC]
        # Copy cached uniqueness/monotonicity from a parent engine after slicing; same
        # logic as IndexEngine._update_from_sliced.
        # Inputs: other - parent ExtensionEngine; reverse - whether order was reversed.
        if other.unique:
            self.unique = other.unique
            self.need_unique_check = other.need_unique_check

        if not other.need_monotonic_check and (
                other.is_monotonic_increasing or other.is_monotonic_decreasing):
            self.need_monotonic_check = 0
            if len(self.values) > 0 and self.values[0] != self.values[-1]:
                # reverse=True means the index has been reversed
                if reverse:
                    self.monotonic_inc = other.monotonic_dec
                    self.monotonic_dec = other.monotonic_inc
                else:
                    self.monotonic_inc = other.monotonic_inc
                    self.monotonic_dec = other.monotonic_dec
            else:
                self.monotonic_inc = 1
                self.monotonic_dec = 1

    @property
    def is_unique(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether all values are distinct.
        # Outputs: bool. Runs the monotonic check first if stale, then if still unknown
        # compares len(values.unique()) with len(values) and caches the result.
        # for why we check is_monotonic_increasing here, see
        # https://github.com/pandas-dev/pandas/pull/55342#discussion_r1361405781
        if self.need_monotonic_check:
            self.is_monotonic_increasing
        if self.need_unique_check:
            arr = self.values.unique()
            self.unique = len(arr) == len(self.values)

            self.need_unique_check = False
        return self.unique

    cdef _do_monotonic_check(self):
        # [FUNCTION DOC]
        # HELPER: Abstract monotonic check. Exceptions: NotImplementedError.
        raise NotImplementedError

    @property
    def is_monotonic_increasing(self) -> bool:
        # [FUNCTION DOC]
        # Property: monotonic non-decreasing (computed lazily and cached).
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    def is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # Property: monotonic non-increasing (computed lazily and cached).
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_dec == 1

    cdef _call_monotonic(self, values):
        # [FUNCTION DOC]
        # HELPER: algos.is_monotonic with timelike=False.
        return algos.is_monotonic(values, timelike=False)

    def sizeof(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # Memory used by the (non-existent) mapping: always 0.
        """ return the sizeof our mapping """
        return 0

    def __sizeof__(self) -> int:
        # [FUNCTION DOC]
        # sys.getsizeof hook returning sizeof().
        return self.sizeof()

    cdef _check_type(self, object obj):
        # [FUNCTION DOC]
        # HELPER: Abstract key validation. Exceptions: NotImplementedError.
        raise NotImplementedError

    cpdef get_loc(self, object val):
        # [FUNCTION DOC]
        # Locate a key in the extension array.
        # Inputs: val - label.
        # Outputs: int, slice or boolean mask. Large monotonic unique arrays use
        # searchsorted plus an equality check; everything else goes through
        # _get_loc_duplicates (both branches of the final check are identical).
        # Exceptions: TypeError for unhashable keys; KeyError if not found.
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        self._check_type(val)

        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(val)

            values = self.values

            loc = self._searchsorted_left(val)
            if loc >= len(values):
                raise KeyError(val)
            if values[loc] != val:
                raise KeyError(val)
            return loc

        if not self.unique:
            return self._get_loc_duplicates(val)

        return self._get_loc_duplicates(val)

    cdef _get_loc_duplicates(self, object val):
        # [FUNCTION DOC]
        # HELPER: Locate a key allowing duplicates.
        # Outputs: if monotonic increasing, int (one match) or slice (several) from
        # left/right searchsorted; otherwise int or boolean mask from comparisons.
        # Exceptions: KeyError if not found or not comparable.
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t diff

        if self.is_monotonic_increasing:
            values = self.values
            try:
                left = values.searchsorted(val, side="left")
                right = values.searchsorted(val, side="right")
            except TypeError:
                # e.g. GH#29189 get_loc(None) with a Float64Index
                raise KeyError(val)

            diff = right - left
            if diff == 0:
                raise KeyError(val)
            elif diff == 1:
                return left
            else:
                return slice(left, right)

        return self._maybe_get_bool_indexer(val)

    cdef Py_ssize_t _searchsorted_left(self, val) except? -1:
        # [FUNCTION DOC]
        # HELPER: Leftmost insertion position via the array's searchsorted.
        # Exceptions: KeyError if the comparison raises TypeError.
        """
        See ObjectEngine._searchsorted_left.__doc__.
        """
        try:
            loc = self.values.searchsorted(val, side="left")
        except TypeError as err:
            # GH#35788 e.g. val=None with float64 values
            raise KeyError(val)
        return loc

    cdef ndarray _get_bool_indexer(self, val):
        # [FUNCTION DOC]
        # HELPER: Abstract boolean match computation. Exceptions: NotImplementedError.
        raise NotImplementedError

    cdef _maybe_get_bool_indexer(self, object val):
        # [FUNCTION DOC]
        # HELPER: Compute the boolean match mask and reduce it (int if exactly one).
        # Exceptions: KeyError if no matches.
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer

        indexer = self._get_bool_indexer(val)
        return _unpack_bool_indexer(indexer, val)

    def get_indexer(self, values) -> np.ndarray:
        # [FUNCTION DOC]
        # Positions of target values (index assumed unique).
        # Inputs: values - array-like of targets.
        # Outputs: intp array, -1 for missing, computed by calling get_loc per element.
        # Exceptions: AssertionError if a lookup returns a non-integer (non-unique
        #   index).
        # values : type(self.values)
        # Note: we only get here with self.is_unique
        cdef:
            Py_ssize_t i, N = len(values)

        res = np.empty(N, dtype=np.intp)

        for i in range(N):
            val = PySequence_GetItem(values, i)
            try:
                loc = self.get_loc(val)
                # Because we are unique, loc should always be an integer
            except KeyError:
                loc = -1
            else:
                assert util.is_integer_object(loc), (loc, val)
            res[i] = loc

        return res

    def get_indexer_non_unique(self, targets):
        # [FUNCTION DOC]
        # Positions of targets allowing duplicates.
        # Inputs: targets - array-like.
        # Outputs: (indexer, missing): indexer concatenates, per target, all matching
        # positions (from int, slice or mask results) or -1; missing lists target
        # positions with no match. Both intp arrays.
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results
        Parameters
        ----------
        targets : type(self.values)
        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        cdef:
            Py_ssize_t i, N = len(targets)

        indexer = []
        missing = []

        # See also IntervalIndex.get_indexer_pointwise
        for i in range(N):
            val = PySequence_GetItem(targets, i)

            try:
                locs = self.get_loc(val)
            except KeyError:
                locs = np.array([-1], dtype=np.intp)
                missing.append(i)
            else:
                if isinstance(locs, slice):
                    # Only needed for get_indexer_non_unique
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype=np.intp)
                elif util.is_integer_object(locs):
                    locs = np.array([locs], dtype=np.intp)
                else:
                    assert locs.dtype.kind == "b"
                    locs = locs.nonzero()[0]

            indexer.append(locs)

        try:
            indexer = np.concatenate(indexer, dtype=np.intp)
        except TypeError:
            # numpy<1.20 doesn't accept dtype keyword
            indexer = np.concatenate(indexer).astype(np.intp, copy=False)
        missing = np.array(missing, dtype=np.intp)

        return indexer, missing


cdef class ExtensionEngine(SharedEngine):
    # [CLASS DOC]
    # Concrete engine for generic ExtensionArrays; uses ranks for monotonicity and
    # array equality for matching.
    def __init__(self, values: "ExtensionArray"):
        # [FUNCTION DOC]
        # Initialize with an ExtensionArray.
        # Side effects: stores values (no copy), size threshold flag, marks unique and
        # monotonic state as unknown.
        self.values = values

        self.over_size_threshold = len(values) >= _SIZE_CUTOFF
        self.need_unique_check = True
        self.need_monotonic_check = True
        self.need_unique_check = True

    cdef _do_monotonic_check(self):
        # [FUNCTION DOC]
        # HELPER: Compute monotonic flags.
        # Side effects: if the array has NA values, both flags are false and uniqueness
        # is computed directly from unique() (need_monotonic_check stays set, so this
        # repeats on each access). Otherwise ranks the values (TypeError -> not
        # monotonic) and checks monotonicity of the ranks; strict monotonicity marks the
        # array unique.
        cdef:
            bint is_unique

        values = self.values
        if values._hasna:
            self.monotonic_inc = 0
            self.monotonic_dec = 0

            nunique = len(values.unique())
            self.unique = nunique == len(values)
            self.need_unique_check = 0
            return

        try:
            ranks = values._rank()

        except TypeError:
            self.monotonic_inc = 0
            self.monotonic_dec = 0
            is_unique = 0
        else:
            self.monotonic_inc, self.monotonic_dec, is_unique = \
                self._call_monotonic(ranks)

        self.need_monotonic_check = 0

        # we can only be sure of uniqueness if is_unique=1
        if is_unique:
            self.unique = 1
            self.need_unique_check = 0

    cdef ndarray _get_bool_indexer(self, val):
        # [FUNCTION DOC]
        # HELPER: Boolean match mask for val.
        # Outputs: isna() for null keys; otherwise values == val, converting a masked
        # boolean result to a numpy bool array with NA as False.
        # Exceptions: KeyError if equality cannot produce an array.
        if checknull(val):
            return self.values.isna()

        try:
            return self.values == val
        except TypeError:
            # e.g. if __eq__ returns a BooleanArray instead of ndarray[bool]
            try:
                return (self.values == val).to_numpy(dtype=bool, na_value=False)
            except (TypeError, AttributeError) as err:
                # e.g. (self.values == val) returned a bool
                #  see test_get_loc_generator[string[pyarrow]]
                # e.g. self.value == val raises TypeError bc generator has no len
                #  see test_get_loc_generator[string[python]]
                raise KeyError from err

    cdef _check_type(self, object val):
        # [FUNCTION DOC]
        # HELPER: Validate a key by hashing it only. Outputs: None.
        # Exceptions: TypeError if unhashable.
        hash(val)


cdef class MaskedIndexEngine(IndexEngine):
    # [CLASS DOC]
    # Base for engines on masked (nullable) arrays: values hold the raw data, mask holds
    # the NA flags; hash tables are created with mask support (see the template).
    def __init__(self, object values):
        # [FUNCTION DOC]
        # Initialize from a masked or Arrow-backed extension array.
        # Side effects: stores raw data and NA mask (both possibly shared, no copy).
        super().__init__(self._get_data(values))
        self.mask = self._get_mask(values)

    def _get_data(self, object values) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: Raw numpy data of a masked array; for Arrow arrays, converts to numpy
        # with NA replaced by 1.
        if hasattr(values, "_mask"):
            return values._data
        # We are an ArrowExtensionArray
        # Set 1 as na_value to avoid ending up with NA and an object array
        # TODO: Remove when arrow engine is implemented
        return values.to_numpy(na_value=1, dtype=values.dtype.numpy_dtype)

    def _get_mask(self, object values) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: Boolean NA mask: the array's mask or isna() for Arrow arrays.
        if hasattr(values, "_mask"):
            return values._mask
        # We are an ArrowExtensionArray
        return values.isna()

    def get_indexer(self, object values) -> np.ndarray:
        # [FUNCTION DOC]
        # Positions of masked target values (index assumed unique).
        # Inputs: values - masked/Arrow array.
        # Outputs: intp array, -1 for missing; NA targets match the index's NA.
        # Side effects: builds the hash table if needed.
        self._ensure_mapping_populated()
        return self.mapping.lookup(self._get_data(values), self._get_mask(values))

    def get_indexer_non_unique(self, object targets):
        # [FUNCTION DOC]
        # Positions of masked targets allowing duplicates.
        # Inputs: targets - masked/Arrow array.
        # Outputs: (indexer, missing) as in IndexEngine.get_indexer_non_unique; NA
        #   targets
        # match every NA position of the index.
        # Algorithm: if fewer than 5 distinct targets, no NA targets and the index is
        # monotonic increasing, binary search per target; otherwise one scan building a
        # value -> positions dict and a list of NA positions.
        # Exceptions: AssertionError if values are object dtype.
        # KNOWN ISSUE: when all targets are NA, the set of non-NA targets is empty so
        #   the
        # scan is skipped and the NA position list is never built; NA targets are then
        # reported as missing even if the index contains NA.
        # KNOWN ISSUE: empty index with non-empty targets loops forever in
        # _maybe_resize_array (zero-length result buffer).
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results

        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        # TODO: Unify with parent class
        cdef:
            ndarray values, mask, target_vals, target_mask
            ndarray[intp_t] result, missing
            set stargets
            list na_pos
            dict d = {}
            object val
            Py_ssize_t count = 0, count_missing = 0
            Py_ssize_t i, j, n, n_t, n_alloc, max_alloc, start, end, na_idx

        target_vals = self._get_data(targets)
        target_mask = self._get_mask(targets)

        values = self.values
        assert not values.dtype == object  # go through object path instead

        mask = self.mask
        stargets = set(target_vals[~target_mask])

        n = len(values)
        n_t = len(target_vals)
        max_alloc = n * n_t
        if n > 10_000:
            n_alloc = 10_000
        else:
            n_alloc = n

        result = np.empty(n_alloc, dtype=np.intp)
        missing = np.empty(n_t, dtype=np.intp)

        # map each starget to its position in the index
        if (
                stargets and
                len(stargets) < 5 and
                not np.any(target_mask) and
                self.is_monotonic_increasing
        ):
            # if there are few enough stargets and the index is monotonically
            # increasing, then use binary search for each starget
            for starget in stargets:
                start = values.searchsorted(starget, side="left")
                end = values.searchsorted(starget, side="right")
                if start != end:
                    d[starget] = list(range(start, end))

            stargets = set()

        if stargets:
            # otherwise, map by iterating through all items in the index

            na_pos = []

            for i in range(n):
                val = PySequence_GetItem(values, i)

                if PySequence_GetItem(mask, i):
                    na_pos.append(i)

                else:
                    if val in stargets:
                        if val not in d:
                            d[val] = []
                        d[val].append(i)

        for i in range(n_t):
            val = PySequence_GetItem(target_vals, i)

            if PySequence_GetItem(target_mask, i):
                if na_pos:
                    result = _maybe_resize_array(
                        result,
                        count + len(na_pos) - 1,
                        max_alloc,
                    )
                    for na_idx in na_pos:
                        result[count] = na_idx
                        count += 1
                    continue

            elif val in d:
                # found
                key = val
                result = _maybe_resize_array(
                    result,
                    count + len(d[key]) - 1,
                    max_alloc,
                )
                for j in d[key]:
                    result[count] = j
                    count += 1
                continue

            # value not found
            result = _maybe_resize_array(result, count, max_alloc)
            result[count] = -1
            count += 1
            missing[count_missing] = i
            count_missing += 1

        return result[0:count], missing[0:count_missing]
