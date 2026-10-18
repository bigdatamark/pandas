# [MODULE DOC]
# Join kernels used by pandas.core.reshape.merge and Index.join. Three families: (1)
# factorized hash-free joins (inner_join, left_outer_join, full_outer_join) operating on
# integer group codes in [-1, max_groups) where -1 is NA, producing pairs of row
# indexers via counting sort; (2) merge-style joins of two sorted (monotonic increasing)
# arrays (left_join_indexer_unique, left_join_indexer, inner_join_indexer,
# outer_join_indexer) returning the joined values plus indexers; (3) asof joins
# (backward, forward, nearest) optionally grouped by int64 "by" codes via an
# Int64HashTable. Throughout, -1 in an output indexer means "no match".
cimport cython
from cython cimport Py_ssize_t
import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
)

cnp.import_array()

from pandas._libs.algos import groupsort_indexer

from pandas._libs.dtypes cimport (
    numeric_object_t,
    numeric_t,
)


@cython.wraparound(False)
@cython.boundscheck(False)
def inner_join(const intp_t[:] left, const intp_t[:] right,
               Py_ssize_t max_groups, bint sort=True):
    # [FUNCTION DOC]
    # Inner join of two arrays of group codes; returns all (left row, right row) pairs
    # with equal non-NA codes. Algorithm: groupsort_indexer (counting sort) on both
    # sides gives, per side, a sorter (positions ordered by group code) and counts per
    # group, where slot 0 counts code -1 (NA) and slot g+1 counts code g. A first nogil
    # pass computes the output size; a second nogil pass walks groups 1..max_groups
    # emitting, in sorted space, the cartesian product of left and right members of each
    # group (left-major), skipping the NA group; _get_result_indexer then maps sorted
    # positions back to original positions. If sort is False the pairs are reordered to
    # follow the original left order: when every left row appears exactly once, by
    # inverting left_sorter; otherwise by a stable counting sort of left_indexer.
    # Inputs: left, right - 1-D intp arrays of codes in [-1, max_groups); max_groups -
    # number of distinct codes; sort - whether output is ordered by group code.
    # Outputs: tuple (left_indexer, right_indexer), intp ndarrays of equal length.
    # Exceptions: none checked (out of range codes are undefined behavior).
    # Side effects: releases the GIL during the counting and filling loops.
    cdef:
        Py_ssize_t i, j, k, count = 0
        intp_t[::1] left_sorter, right_sorter
        intp_t[::1] left_count, right_count
        intp_t[::1] left_indexer, right_indexer
        intp_t lc, rc
        Py_ssize_t left_pos = 0, right_pos = 0, position = 0
        Py_ssize_t offset

    left_sorter, left_count = groupsort_indexer(left, max_groups)
    right_sorter, right_count = groupsort_indexer(right, max_groups)

    with nogil:
        # First pass, determine size of result set, do not use the NA group
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc > 0 and lc > 0:
                count += lc * rc

    left_indexer = np.empty(count, dtype=np.intp)
    right_indexer = np.empty(count, dtype=np.intp)

    with nogil:
        # exclude the NA group
        left_pos = left_count[0]
        right_pos = right_count[0]
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc > 0 and lc > 0:
                for j in range(lc):
                    offset = position + j * rc
                    for k in range(rc):
                        left_indexer[offset + k] = left_pos + j
                        right_indexer[offset + k] = right_pos + k
                position += lc * rc
            left_pos += lc
            right_pos += rc

        # Will overwrite left/right indexer with the result
        _get_result_indexer(left_sorter, left_indexer)
        _get_result_indexer(right_sorter, right_indexer)

    if not sort:
        # if not asked to sort, revert to original order
        if len(left) == len(left_indexer):
            # no multiple matches for any row on the left
            # this is a short-cut to avoid groupsort_indexer
            # otherwise, the `else` path also works in this case
            rev = np.empty(len(left), dtype=np.intp)
            rev.put(np.asarray(left_sorter), np.arange(len(left)))
        else:
            rev, _ = groupsort_indexer(left_indexer, len(left))

        return np.asarray(left_indexer).take(rev), np.asarray(right_indexer).take(rev)
    else:
        return np.asarray(left_indexer), np.asarray(right_indexer)


@cython.wraparound(False)
@cython.boundscheck(False)
def left_outer_join(const intp_t[:] left, const intp_t[:] right,
                    Py_ssize_t max_groups, bint sort=True):
    # [FUNCTION DOC]
    # Left outer join of group codes: like inner_join, but every left row with a non-NA
    # code that has no right match yields one pair (left row, -1). Left rows with NA
    # code (-1) are dropped. Algorithm: groupsort_indexer (counting sort) on both sides
    # gives, per side, a sorter (positions ordered by group code) and counts per group,
    # where slot 0 counts code -1 (NA) and slot g+1 counts code g. A first nogil pass
    # computes the output size; a second nogil pass walks groups 1..max_groups emitting,
    # in sorted space, the cartesian product of left and right members of each group
    # (left-major), skipping the NA group; _get_result_indexer then maps sorted
    # positions back to original positions. If sort is False the pairs are reordered to
    # follow the original left order: when every left row appears exactly once, by
    # inverting left_sorter; otherwise by a stable counting sort of left_indexer.
    # Inputs: left, right - 1-D intp codes in [-1, max_groups); max_groups - int; sort -
    # bool. Outputs: tuple (left_indexer, right_indexer) intp ndarrays.
    # Exceptions: none checked. Side effects: releases the GIL in the main loops.
    cdef:
        Py_ssize_t i, j, k, count = 0
        ndarray[intp_t] rev
        intp_t[::1] left_count, right_count
        intp_t[::1] left_sorter, right_sorter
        intp_t[::1] left_indexer, right_indexer
        intp_t lc, rc
        Py_ssize_t left_pos = 0, right_pos = 0, position = 0
        Py_ssize_t offset

    left_sorter, left_count = groupsort_indexer(left, max_groups)
    right_sorter, right_count = groupsort_indexer(right, max_groups)

    with nogil:
        # First pass, determine size of result set, do not use the NA group
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc > 0:
                count += lc * rc
            else:
                count += lc

    left_indexer = np.empty(count, dtype=np.intp)
    right_indexer = np.empty(count, dtype=np.intp)

    with nogil:
        # exclude the NA group
        left_pos = left_count[0]
        right_pos = right_count[0]
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc == 0:
                for j in range(lc):
                    left_indexer[position + j] = left_pos + j
                    right_indexer[position + j] = -1
                position += lc
            else:
                for j in range(lc):
                    offset = position + j * rc
                    for k in range(rc):
                        left_indexer[offset + k] = left_pos + j
                        right_indexer[offset + k] = right_pos + k
                position += lc * rc
            left_pos += lc
            right_pos += rc

        # Will overwrite left/right indexer with the result
        _get_result_indexer(left_sorter, left_indexer)
        _get_result_indexer(right_sorter, right_indexer)

    if not sort:  # if not asked to sort, revert to original order
        if len(left) == len(left_indexer):
            # no multiple matches for any row on the left
            # this is a short-cut to avoid groupsort_indexer
            # otherwise, the `else` path also works in this case
            rev = np.empty(len(left), dtype=np.intp)
            rev.put(np.asarray(left_sorter), np.arange(len(left)))
        else:
            rev, _ = groupsort_indexer(left_indexer, len(left))

        return np.asarray(left_indexer).take(rev), np.asarray(right_indexer).take(rev)
    else:
        return np.asarray(left_indexer), np.asarray(right_indexer)


@cython.wraparound(False)
@cython.boundscheck(False)
def full_outer_join(const intp_t[:] left, const intp_t[:] right,
                    Py_ssize_t max_groups):
    # [FUNCTION DOC]
    # Full outer join of group codes: per group, the cartesian product when both sides
    # have members, else (left row, -1) for left-only groups or (-1, right row) for
    # right-only groups; NA codes are dropped. Output is ordered by group code (no sort
    # option). Algorithm: groupsort_indexer (counting sort) on both sides gives, per
    # side, a sorter (positions ordered by group code) and counts per group, where slot
    # 0 counts code -1 (NA) and slot g+1 counts code g. A first nogil pass computes the
    # output size; a second nogil pass walks groups 1..max_groups emitting, in sorted
    # space, the cartesian product of left and right members of each group (left-major),
    # skipping the NA group; _get_result_indexer then maps sorted positions back to
    # original positions.
    # Inputs: left, right - 1-D intp codes; max_groups - int.
    # Outputs: tuple (left_indexer, right_indexer) intp ndarrays.
    # Exceptions: none checked. Side effects: releases the GIL in the main loops.
    cdef:
        Py_ssize_t i, j, k, count = 0
        intp_t[::1] left_sorter, right_sorter
        intp_t[::1] left_count, right_count
        intp_t[::1] left_indexer, right_indexer
        intp_t lc, rc
        intp_t left_pos = 0, right_pos = 0
        Py_ssize_t offset, position = 0

    left_sorter, left_count = groupsort_indexer(left, max_groups)
    right_sorter, right_count = groupsort_indexer(right, max_groups)

    with nogil:
        # First pass, determine size of result set, do not use the NA group
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc > 0 and lc > 0:
                count += lc * rc
            else:
                count += lc + rc

    left_indexer = np.empty(count, dtype=np.intp)
    right_indexer = np.empty(count, dtype=np.intp)

    with nogil:
        # exclude the NA group
        left_pos = left_count[0]
        right_pos = right_count[0]
        for i in range(1, max_groups + 1):
            lc = left_count[i]
            rc = right_count[i]

            if rc == 0:
                for j in range(lc):
                    left_indexer[position + j] = left_pos + j
                    right_indexer[position + j] = -1
                position += lc
            elif lc == 0:
                for j in range(rc):
                    left_indexer[position + j] = -1
                    right_indexer[position + j] = right_pos + j
                position += rc
            else:
                for j in range(lc):
                    offset = position + j * rc
                    for k in range(rc):
                        left_indexer[offset + k] = left_pos + j
                        right_indexer[offset + k] = right_pos + k
                position += lc * rc
            left_pos += lc
            right_pos += rc

        # Will overwrite left/right indexer with the result
        _get_result_indexer(left_sorter, left_indexer)
        _get_result_indexer(right_sorter, right_indexer)

    return np.asarray(left_indexer), np.asarray(right_indexer)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _get_result_indexer(
    const intp_t[::1] sorter,
    intp_t[::1] indexer,
) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: In place, replace each sorted-space position in indexer by
    # sorter[position], keeping -1 as -1. If sorter is empty every entry becomes -1.
    # Inputs: sorter - intp array mapping sorted position to original row; indexer -
    # intp array to rewrite. Outputs: none. Exceptions: none (noexcept, nogil).
    # Side effects: mutates indexer.
    """NOTE: overwrites indexer with the result to avoid allocating another array"""
    cdef:
        Py_ssize_t i, n, idx

    if len(sorter) > 0:
        # cython-only equivalent to
        #  `res = algos.take_nd(sorter, indexer, fill_value=-1)`
        n = indexer.shape[0]
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                indexer[i] = -1
            else:
                indexer[i] = sorter[idx]
    else:
        # length-0 case
        indexer[:] = -1


@cython.wraparound(False)
@cython.boundscheck(False)
def ffill_indexer(const intp_t[:] indexer) -> np.ndarray:
    # [FUNCTION DOC]
    # Forward-fill -1 entries of an indexer with the last non -1 value seen (leading -1s
    # stay -1).
    # Inputs: indexer - 1-D intp array. Outputs: new intp ndarray of same length.
    # Side effects: none.
    cdef:
        Py_ssize_t i, n = len(indexer)
        ndarray[intp_t] result
        intp_t val, last_obs

    result = np.empty(n, dtype=np.intp)
    last_obs = -1

    for i in range(n):
        val = indexer[i]
        if val == -1:
            result[i] = last_obs
        else:
            result[i] = val
            last_obs = val

    return result


# ----------------------------------------------------------------------
# left_join_indexer, inner_join_indexer, outer_join_indexer
# ----------------------------------------------------------------------

# Joins on ordered, unique indices

# right might contain non-unique values

@cython.wraparound(False)
@cython.boundscheck(False)
def left_join_indexer_unique(
    ndarray[numeric_object_t] left,
    ndarray[numeric_object_t] right
):
    # [FUNCTION DOC]
    # For each element of sorted unique left, find the position of an equal element in
    # sorted right (the first one when right has duplicates), or -1. Two-pointer merge:
    # equal values record j (and consecutive equal left values reuse it, then j
    # advances); left greater than right advances j; left smaller writes -1 and advances
    # i; once right is exhausted remaining left get -1.
    # Inputs: left, right - 1-D ndarrays (fused numeric or object) sorted ascending;
    # left must be unique, right may contain duplicates (per the module comment; the
    # docstring says both strictly increasing).
    # Outputs: intp ndarray of length len(left).
    # Exceptions: comparison errors for incomparable objects. Side effects: none.
    """
    Both left and right are strictly monotonic increasing.
    """
    cdef:
        Py_ssize_t i, j, nleft, nright
        ndarray[intp_t] indexer
        numeric_object_t rval

    i = 0
    j = 0
    nleft = len(left)
    nright = len(right)

    indexer = np.empty(nleft, dtype=np.intp)
    while True:
        if i == nleft:
            break

        if j == nright:
            indexer[i] = -1
            i += 1
            continue

        rval = right[j]

        while i < nleft - 1 and left[i] == rval:
            indexer[i] = j
            i += 1

        if left[i] == rval:
            indexer[i] = j
            i += 1
            while i < nleft - 1 and left[i] == rval:
                indexer[i] = j
                i += 1
            j += 1
        elif left[i] > rval:
            indexer[i] = -1
            j += 1
        else:
            indexer[i] = -1
            i += 1
    return indexer


@cython.wraparound(False)
@cython.boundscheck(False)
def left_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right):
    # [FUNCTION DOC]
    # Left join of two sorted arrays (many-to-one): every left element appears; matches
    # pair with right positions, unmatched left elements get -1. The shared merge step
    # on a match lval == rval: if not at the last left element, advance j when
    # right[j+1] is the same value, otherwise advance i and also advance j if the new
    # left[i] differs; if at the last left element advance j (and i if right[j] now
    # differs); if both at their ends stop. Two passes are made: the first only counts
    # the output size, the second fills result, lindexer, rindexer.
    # Inputs: left, right - 1-D sorted ndarrays of the same fused numeric or object
    # type.
    # Outputs: tuple (result values ndarray with left's dtype, lindexer, rindexer intp).
    # Exceptions: comparison errors for incomparable objects. Side effects: none.
    # KNOWN ISSUE: many-to-many duplicates yield an incomplete product (see
    # inner_join_indexer).
    """
    Two-pass algorithm for monotonic indexes. Handles many-to-one merges.

    Both left and right are monotonic increasing, but at least one of them
    is non-unique (if both were unique we'd use left_join_indexer_unique).
    """
    cdef:
        Py_ssize_t i, j, nright, nleft, count
        numeric_object_t lval, rval
        ndarray[intp_t] lindexer, rindexer
        ndarray[numeric_object_t] result

    nleft = len(left)
    nright = len(right)

    # First pass is to find the size 'count' of our output indexers.
    i = 0
    j = 0
    count = 0
    if nleft > 0:
        while i < nleft:
            if j == nright:
                count += nleft - i
                break

            lval = left[i]
            rval = right[j]

            if lval == rval:
                # This block is identical across
                #  left_join_indexer, inner_join_indexer, outer_join_indexer
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                count += 1
                i += 1
            else:
                j += 1

    # do it again now that result size is known

    lindexer = np.empty(count, dtype=np.intp)
    rindexer = np.empty(count, dtype=np.intp)
    result = np.empty(count, dtype=left.dtype)

    i = 0
    j = 0
    count = 0
    if nleft > 0:
        while i < nleft:
            if j == nright:
                while i < nleft:
                    lindexer[count] = i
                    rindexer[count] = -1
                    result[count] = left[i]
                    i += 1
                    count += 1
                break

            lval = left[i]
            rval = right[j]

            if lval == rval:
                lindexer[count] = i
                rindexer[count] = j
                result[count] = lval
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                # i.e. lval not in right; we keep for left_join_indexer
                lindexer[count] = i
                rindexer[count] = -1
                result[count] = lval
                count += 1
                i += 1
            else:
                # i.e. rval not in left; we discard for left_join_indexer
                j += 1

    return result, lindexer, rindexer


@cython.wraparound(False)
@cython.boundscheck(False)
def inner_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right):
    # [FUNCTION DOC]
    # Inner join of two sorted arrays: only values present in both appear, with their
    # left and right positions; non-matching values on either side are skipped. The
    # shared merge step on a match lval == rval: if not at the last left element,
    # advance j when right[j+1] is the same value, otherwise advance i and also advance
    # j if the new left[i] differs; if at the last left element advance j (and i if
    # right[j] now differs); if both at their ends stop. Two passes are made: the first
    # only counts the output size, the second fills result, lindexer, rindexer.
    # Inputs: left, right - 1-D sorted ndarrays of the same fused numeric or object
    # type.
    # Outputs: tuple (result values, lindexer, rindexer intp ndarrays).
    # Exceptions: comparison errors for incomparable objects. Side effects: none.
    # KNOWN ISSUE: with duplicates on both sides (many-to-many) the merge step does not
    # produce the full cartesian product (e.g. [1, 1] with [1, 1] yields 3 pairs, not
    # 4), although the docstring says inputs need not be unique; callers must ensure at
    # most one side has duplicates of a given value.
    """
    Two-pass algorithm for monotonic indexes. Handles many-to-one merges.

    Both left and right are monotonic increasing but not necessarily unique.
    """
    cdef:
        Py_ssize_t i, j, nright, nleft, count
        numeric_object_t lval, rval
        ndarray[intp_t] lindexer, rindexer
        ndarray[numeric_object_t] result

    nleft = len(left)
    nright = len(right)

    # First pass is to find the size 'count' of our output indexers.
    i = 0
    j = 0
    count = 0
    if nleft > 0 and nright > 0:
        while True:
            if i == nleft:
                break
            if j == nright:
                break

            lval = left[i]
            rval = right[j]
            if lval == rval:
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                # i.e. lval not in right; we discard for inner_indexer
                i += 1
            else:
                # i.e. rval not in left; we discard for inner_indexer
                j += 1

    # do it again now that result size is known

    lindexer = np.empty(count, dtype=np.intp)
    rindexer = np.empty(count, dtype=np.intp)
    result = np.empty(count, dtype=left.dtype)

    i = 0
    j = 0
    count = 0
    if nleft > 0 and nright > 0:
        while True:
            if i == nleft:
                break
            if j == nright:
                break

            lval = left[i]
            rval = right[j]
            if lval == rval:
                lindexer[count] = i
                rindexer[count] = j
                result[count] = lval
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                # i.e. lval not in right; we discard for inner_indexer
                i += 1
            else:
                # i.e. rval not in left; we discard for inner_indexer
                j += 1

    return result, lindexer, rindexer


@cython.wraparound(False)
@cython.boundscheck(False)
def outer_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right):
    # [FUNCTION DOC]
    # Outer join of two sorted arrays: union of values in sorted order; matched values
    # carry both positions, left-only values have rindexer -1, right-only values have
    # lindexer -1. Empty-side shortcuts copy the other side. The shared merge step on a
    # match lval == rval: if not at the last left element, advance j when right[j+1] is
    # the same value, otherwise advance i and also advance j if the new left[i] differs;
    # if at the last left element advance j (and i if right[j] now differs); if both at
    # their ends stop. Two passes are made: the first only counts the output size, the
    # second fills result, lindexer, rindexer.
    # Inputs: left, right - 1-D sorted ndarrays of the same fused numeric or object
    # type.
    # Outputs: tuple (result values with left's dtype, lindexer, rindexer intp
    # ndarrays).
    # Exceptions: comparison errors for incomparable objects. Side effects: none.
    # KNOWN ISSUE: with duplicates on both sides (many-to-many) the merge step does not
    # produce the full cartesian product (e.g. [1, 1] with [1, 1] yields 3 pairs, not
    # 4), although the docstring says inputs need not be unique; callers must ensure at
    # most one side has duplicates of a given value.
    """
    Both left and right are monotonic increasing but not necessarily unique.
    """
    cdef:
        Py_ssize_t i, j, nright, nleft, count
        numeric_object_t lval, rval
        ndarray[intp_t] lindexer, rindexer
        ndarray[numeric_object_t] result

    nleft = len(left)
    nright = len(right)

    # First pass is to find the size 'count' of our output indexers.
    # count will be length of left plus the number of elements of right not in
    # left (counting duplicates)
    i = 0
    j = 0
    count = 0
    if nleft == 0:
        count = nright
    elif nright == 0:
        count = nleft
    else:
        while True:
            if i == nleft:
                count += nright - j
                break
            if j == nright:
                count += nleft - i
                break

            lval = left[i]
            rval = right[j]
            if lval == rval:
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                count += 1
                i += 1
            else:
                count += 1
                j += 1

    lindexer = np.empty(count, dtype=np.intp)
    rindexer = np.empty(count, dtype=np.intp)
    result = np.empty(count, dtype=left.dtype)

    # do it again, but populate the indexers / result

    i = 0
    j = 0
    count = 0
    if nleft == 0:
        for j in range(nright):
            lindexer[j] = -1
            rindexer[j] = j
            result[j] = right[j]
    elif nright == 0:
        for i in range(nleft):
            lindexer[i] = i
            rindexer[i] = -1
            result[i] = left[i]
    else:
        while True:
            if i == nleft:
                while j < nright:
                    lindexer[count] = -1
                    rindexer[count] = j
                    result[count] = right[j]
                    count += 1
                    j += 1
                break
            if j == nright:
                while i < nleft:
                    lindexer[count] = i
                    rindexer[count] = -1
                    result[count] = left[i]
                    count += 1
                    i += 1
                break

            lval = left[i]
            rval = right[j]

            if lval == rval:
                lindexer[count] = i
                rindexer[count] = j
                result[count] = lval
                count += 1
                if i < nleft - 1:
                    if j < nright - 1 and right[j + 1] == rval:
                        j += 1
                    else:
                        i += 1
                        if left[i] != rval:
                            j += 1
                elif j < nright - 1:
                    j += 1
                    if lval != right[j]:
                        i += 1
                else:
                    # end of the road
                    break
            elif lval < rval:
                # i.e. lval not in right; we keep for outer_join_indexer
                lindexer[count] = i
                rindexer[count] = -1
                result[count] = lval
                count += 1
                i += 1
            else:
                # i.e. rval not in left; we keep for outer_join_indexer
                lindexer[count] = -1
                rindexer[count] = j
                result[count] = rval
                count += 1
                j += 1

    return result, lindexer, rindexer


# ----------------------------------------------------------------------
# asof_join_by
# ----------------------------------------------------------------------

from pandas._libs.hashtable cimport Int64HashTable


def asof_join_backward_on_X_by_Y(const numeric_t[:] left_values,
                                 const numeric_t[:] right_values,
                                 const int64_t[:] left_by_values,
                                 const int64_t[:] right_by_values,
                                 bint allow_exact_matches=True,
                                 tolerance=None,
                                 bint use_hashtable=True):

    # [FUNCTION DOC]
    # Backward asof join: for each left row, the last right row whose key is <= (or < if
    # not allow_exact_matches) the left key, restricted to rows with the same by-value
    # when use_hashtable. Single forward sweep: right_pos advances while the right key
    # qualifies, inserting by-value -> right_pos into a hash table (so the table holds
    # the latest qualifying row per by-value); the match is the table lookup (or
    # right_pos - 1 without hashtable, -1 if none). If tolerance is given and the left
    # minus right key exceeds it, the match becomes -1.
    # Inputs: left_values, right_values - 1-D sorted (ascending) numeric memoryviews
    # (fused numeric_t, same type) of the "on" keys; left_by_values, right_by_values -
    # int64 codes of the "by" keys; allow_exact_matches - whether equal keys count as
    # matches; tolerance - optional scalar maximum distance; use_hashtable - if False,
    # ignore the by-values (plain asof join).
    # Outputs: tuple (left_indexer = arange(len(left)), right_indexer with -1 for no
    # match), intp ndarrays.
    # Exceptions: TypeError/OverflowError if tolerance cannot convert to numeric_t.
    # Side effects: allocates a hash table.
    cdef:
        Py_ssize_t left_pos, right_pos, left_size, right_size, found_right_pos
        ndarray[intp_t] left_indexer, right_indexer
        bint has_tolerance = False
        numeric_t tolerance_ = 0
        numeric_t diff = 0
        Int64HashTable hash_table

    # if we are using tolerance, set our objects
    if tolerance is not None:
        has_tolerance = True
        tolerance_ = tolerance

    left_size = len(left_values)
    right_size = len(right_values)

    left_indexer = np.empty(left_size, dtype=np.intp)
    right_indexer = np.empty(left_size, dtype=np.intp)

    if use_hashtable:
        hash_table = Int64HashTable(right_size)

    right_pos = 0
    for left_pos in range(left_size):
        # restart right_pos if it went negative in a previous iteration
        if right_pos < 0:
            right_pos = 0

        # find last position in right whose value is less than left's
        if allow_exact_matches:
            while (right_pos < right_size and
                   right_values[right_pos] <= left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos += 1
        else:
            while (right_pos < right_size and
                   right_values[right_pos] < left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos += 1
        right_pos -= 1

        # save positions as the desired index
        if use_hashtable:
            by_value = left_by_values[left_pos]
            found_right_pos = (hash_table.get_item(by_value)
                               if by_value in hash_table else -1)
        else:
            found_right_pos = right_pos

        left_indexer[left_pos] = left_pos
        right_indexer[left_pos] = found_right_pos

        # if needed, verify that tolerance is met
        if has_tolerance and found_right_pos != -1:
            diff = left_values[left_pos] - right_values[found_right_pos]
            if diff > tolerance_:
                right_indexer[left_pos] = -1

    return left_indexer, right_indexer


def asof_join_forward_on_X_by_Y(const numeric_t[:] left_values,
                                const numeric_t[:] right_values,
                                const int64_t[:] left_by_values,
                                const int64_t[:] right_by_values,
                                bint allow_exact_matches=1,
                                tolerance=None,
                                bint use_hashtable=True):

    # [FUNCTION DOC]
    # Forward asof join: mirror of the backward version, sweeping left rows from last to
    # first and right_pos downward; the match is the first right row whose key is >= (or
    # > if not allow_exact_matches) the left key, per by-value when use_hashtable.
    # Tolerance rejects matches where right minus left exceeds it.
    # Inputs: left_values, right_values - 1-D sorted (ascending) numeric memoryviews
    # (fused numeric_t, same type) of the "on" keys; left_by_values, right_by_values -
    # int64 codes of the "by" keys; allow_exact_matches - whether equal keys count as
    # matches; tolerance - optional scalar maximum distance; use_hashtable - if False,
    # ignore the by-values (plain asof join).
    # Outputs: tuple (left_indexer, right_indexer) intp ndarrays, -1 for no match.
    # Exceptions: as for the backward version. Side effects: allocates a hash table.
    cdef:
        Py_ssize_t left_pos, right_pos, left_size, right_size, found_right_pos
        ndarray[intp_t] left_indexer, right_indexer
        bint has_tolerance = False
        numeric_t tolerance_ = 0
        numeric_t diff = 0
        Int64HashTable hash_table

    # if we are using tolerance, set our objects
    if tolerance is not None:
        has_tolerance = True
        tolerance_ = tolerance

    left_size = len(left_values)
    right_size = len(right_values)

    left_indexer = np.empty(left_size, dtype=np.intp)
    right_indexer = np.empty(left_size, dtype=np.intp)

    if use_hashtable:
        hash_table = Int64HashTable(right_size)

    right_pos = right_size - 1
    for left_pos in range(left_size - 1, -1, -1):
        # restart right_pos if it went over in a previous iteration
        if right_pos == right_size:
            right_pos = right_size - 1

        # find first position in right whose value is greater than left's
        if allow_exact_matches:
            while (right_pos >= 0 and
                   right_values[right_pos] >= left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos -= 1
        else:
            while (right_pos >= 0 and
                   right_values[right_pos] > left_values[left_pos]):
                if use_hashtable:
                    hash_table.set_item(right_by_values[right_pos], right_pos)
                right_pos -= 1
        right_pos += 1

        # save positions as the desired index
        if use_hashtable:
            by_value = left_by_values[left_pos]
            found_right_pos = (hash_table.get_item(by_value)
                               if by_value in hash_table else -1)
        else:
            found_right_pos = (right_pos
                               if right_pos != right_size else -1)

        left_indexer[left_pos] = left_pos
        right_indexer[left_pos] = found_right_pos

        # if needed, verify that tolerance is met
        if has_tolerance and found_right_pos != -1:
            diff = right_values[found_right_pos] - left_values[left_pos]
            if diff > tolerance_:
                right_indexer[left_pos] = -1

    return left_indexer, right_indexer


def asof_join_nearest_on_X_by_Y(const numeric_t[:] left_values,
                                const numeric_t[:] right_values,
                                const int64_t[:] left_by_values,
                                const int64_t[:] right_by_values,
                                bint allow_exact_matches=True,
                                tolerance=None,
                                bint use_hashtable=True):

    # [FUNCTION DOC]
    # Nearest asof join: runs the backward and forward joins (with the same tolerance),
    # then for each left row picks the candidate with the smaller key distance,
    # preferring backward on ties; if only one exists it is used, else -1.
    # Inputs: left_values, right_values - 1-D sorted (ascending) numeric memoryviews
    # (fused numeric_t, same type) of the "on" keys; left_by_values, right_by_values -
    # int64 codes of the "by" keys; allow_exact_matches - whether equal keys count as
    # matches; tolerance - optional scalar maximum distance; use_hashtable - if False,
    # ignore the by-values (plain asof join).
    # Outputs: tuple (left_indexer, right_indexer) intp ndarrays.
    # Exceptions: as for the backward version. Side effects: none beyond allocation.
    cdef:
        ndarray[intp_t] bli, bri, fli, fri

        ndarray[intp_t] left_indexer, right_indexer
        Py_ssize_t left_size, i
        numeric_t bdiff, fdiff

    # search both forward and backward
    bli, bri = asof_join_backward_on_X_by_Y(
        left_values,
        right_values,
        left_by_values,
        right_by_values,
        allow_exact_matches,
        tolerance,
        use_hashtable
    )
    fli, fri = asof_join_forward_on_X_by_Y(
        left_values,
        right_values,
        left_by_values,
        right_by_values,
        allow_exact_matches,
        tolerance,
        use_hashtable
    )

    # choose the smaller timestamp
    left_size = len(left_values)
    left_indexer = np.empty(left_size, dtype=np.intp)
    right_indexer = np.empty(left_size, dtype=np.intp)

    for i in range(len(bri)):
        # choose timestamp from right with smaller difference
        if bri[i] != -1 and fri[i] != -1:
            bdiff = left_values[bli[i]] - right_values[bri[i]]
            fdiff = right_values[fri[i]] - left_values[fli[i]]
            right_indexer[i] = bri[i] if bdiff <= fdiff else fri[i]
        else:
            right_indexer[i] = bri[i] if bri[i] != -1 else fri[i]
        left_indexer[i] = bli[i]

    return left_indexer, right_indexer
