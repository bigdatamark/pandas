# [MODULE DOC]
# Cython kernel that computes per-row [start, end) slice bounds for variable
#   (time-offset based) rolling windows over a monotonic int64 index. Used by
#   pandas.core.indexers.objects.VariableWindowIndexer; the resulting start/end arrays
#   are consumed by the aggregation kernels in aggregations.pyx.
# Compiled with boundscheck/wraparound off and C division semantics, so callers must
#   pass a valid, monotonic index of the stated length.
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from numpy cimport (
    int64_t,
    ndarray,
)

# Cython routines for window indexers


def calculate_variable_window_bounds(
    int64_t num_values,
    int64_t window_size,
    object min_periods,  # unused but here to match get_window_bounds signature
    bint center,
    str closed,
    const int64_t[:] index
):
    # [FUNCTION DOC]
    # Compute, for each row i, the half-open slice [start[i], end[i]) of positions whose
    #   index values fall in the window ending (or centered) at index[i].
    # Inputs: num_values - number of rows (length of index); window_size - window length
    #   in index units (e.g. nanoseconds); min_periods - unused, kept for signature
    #   compatibility; center - if true the window spans window_size/2 on each side of
    #   index[i]; closed - 'right' (default when None), 'left', 'both' or 'neither',
    #   selecting which window endpoints are inclusive; index - 1-D int64 array,
    #   monotonically increasing or decreasing (direction detected by comparing first
    #   and last element).
    # Algorithm: closed flags are derived from 'closed'; if center and window_size is
    #   odd both ends are forced closed (GH 43997). A growth sign of +1/-1 is used so
    #   comparisons work for decreasing indexes. Row 0 starts at 0 and ends at 1 if
    #   right closed (else 0); when centered end[0] is found by a linear scan. For each
    #   later row, start_bound = index[i] - sign*window (or - sign*window/2 when
    #   centered), shifted by one unit when left closed; start[i] is the first j in
    #   [start[i-1], i) with (index[j]-start_bound)*sign > 0, else i. end[i] is found by
    #   scanning forward from end[i-1] when centered, otherwise it is i+1 (or end[i-1]+1
    #   / end[i-1] for duplicate index values); when not right closed and not centered,
    #   end[i] is decremented by one.
    # Outputs: tuple (start, end) of int64 ndarrays of length num_values; two empty
    #   int64 arrays when num_values <= 0.
    # Exceptions: none raised explicitly; out-of-bounds access is unchecked.
    # Side effects: allocates two arrays; releases the GIL for the main loop.
    # Note: window_size / 2 is computed in floating point and truncated back to
    #   int64 on assignment, so odd window sizes give a half-width rounded toward
    #   zero; this is why both ends are forced closed in that case.
    """
    Calculate window boundaries for rolling windows from a time offset.

    Parameters
    ----------
    num_values : int64
        total number of values

    window_size : int64
        window size calculated from the offset

    min_periods : object
        ignored, exists for compatibility

    center : bint
        center the rolling window on the current observation

    closed : str
        string of side of the window that should be closed

    index : ndarray[int64]
        time series index to roll over

    Returns
    -------
    (ndarray[int64], ndarray[int64])
    """
    cdef:
        bint left_closed = False
        bint right_closed = False
        ndarray[int64_t, ndim=1] start, end
        int64_t start_bound, end_bound, index_growth_sign = 1
        Py_ssize_t i, j

    if num_values <= 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

    # default is 'right'
    if closed is None:
        closed = "right"

    if closed in ["right", "both"]:
        right_closed = True

    if closed in ["left", "both"]:
        left_closed = True

    # GH 43997:
    # If the forward and the backward facing windows
    # would result in a fraction of 1/2 a nanosecond
    # we need to make both interval ends inclusive.
    if center and window_size % 2 == 1:
        right_closed = True
        left_closed = True

    if index[num_values - 1] < index[0]:
        index_growth_sign = -1

    start = np.empty(num_values, dtype="int64")
    start.fill(-1)
    end = np.empty(num_values, dtype="int64")
    end.fill(-1)

    start[0] = 0

    # right endpoint is closed
    if right_closed:
        end[0] = 1
    # right endpoint is open
    else:
        end[0] = 0
    if center:
        end_bound = index[0] + index_growth_sign * window_size / 2
        for j in range(0, num_values):
            if (index[j] - end_bound) * index_growth_sign < 0:
                end[0] = j + 1
            elif (index[j] - end_bound) * index_growth_sign == 0 and right_closed:
                end[0] = j + 1
            elif (index[j] - end_bound) * index_growth_sign >= 0:
                end[0] = j
                break

    with nogil:

        # start is start of slice interval (including)
        # end is end of slice interval (not including)
        for i in range(1, num_values):
            if center:
                end_bound = index[i] + index_growth_sign * window_size / 2
                start_bound = index[i] - index_growth_sign * window_size / 2
            else:
                end_bound = index[i]
                start_bound = index[i] - index_growth_sign * window_size

            # left endpoint is closed
            if left_closed:
                start_bound -= 1 * index_growth_sign

            # advance the start bound until we are
            # within the constraint
            start[i] = i
            for j in range(start[i - 1], i):
                if (index[j] - start_bound) * index_growth_sign > 0:
                    start[i] = j
                    break

            # for centered window advance the end bound until we are
            # outside the constraint
            if center:
                for j in range(end[i - 1], num_values + 1):
                    if j == num_values:
                        end[i] = j
                    elif ((index[j] - end_bound) * index_growth_sign == 0 and
                          right_closed):
                        end[i] = j + 1
                    elif (index[j] - end_bound) * index_growth_sign >= 0:
                        end[i] = j
                        break
            # end bound is previous end
            # or current index
            elif index[end[i - 1]] == end_bound and not right_closed:
                end[i] = end[i - 1] + 1
            elif (index[end[i - 1]] - end_bound) * index_growth_sign <= 0:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # right endpoint is open
            if not right_closed and not center:
                end[i] -= 1
    return start, end
