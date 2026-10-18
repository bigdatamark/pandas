"""
Numba 1D min/max kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

# [MODULE DOC]
# HELPER: internal numba kernels for minimum and maximum, used by the numba engine of
# rolling/expanding windows and groupby.
# Public contents: bisect_left (binary search helper that numba lacks),
# sliding_min_max (monotonic queue algorithm for windows, driven by start/end
# bound arrays) and grouped_min_max (groupby kernel driven by group labels).
# Both kernels take a flag choosing maximum versus minimum. Results are consumed
# by the looper in the executor module. Importing requires numba; functions are
# compiled lazily on first call (no other import-time side effects).
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numba
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt


@numba.njit(nogil=True, parallel=False)
def bisect_left(a: list[Any], x: Any, lo: int = 0, hi: int = -1) -> int:
    # [FUNCTION DOC]
    # HELPER: binary search for the leftmost insertion point of x in the sorted
    # sequence a, within the half open range from lo (default 0) to hi. A hi of
    # minus one (the default) means the length of a. Returns the integer position;
    # does not modify a. No exceptions are raised.
    """Same as https://docs.python.org/3/library/bisect.html; not in numba yet!"""
    if hi == -1:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    # [FUNCTION DOC]
    # Window minimum or maximum kernel for one 1D array.
    # Inputs: values (1D array, floats or integers), result_dtype (output dtype),
    # start and end (integer arrays with one half open window each), min_periods
    # (minimum non-missing observations; values below one are treated as one) and
    # is_max (true for maximum, false for minimum).
    # Behavior: uses a monotonic queue of candidate positions plus a stack of
    # "dominator" windows (windows that start earlier than their predecessors and
    # so cover them) to remember values that earlier windows may still need.
    # Windows are processed in order and must have non-decreasing ends (and
    # non-decreasing starts when ends are equal); otherwise a value error is
    # raised naming the offending index. A window without enough valid values
    # yields NaN for float input, or, for integer input, its position is recorded
    # to be masked by the caller.
    # Returns a pair: the output array and the list of positions to mask.
    # Basic idea of the algorithm: https://stackoverflow.com/a/12239580
    # It was generalized to work with an arbitrary list of any window size and position
    # by adding the Dominators stack.

    N = len(start)
    na_pos = []
    output = np.empty(N, dtype=result_dtype)

    def cmp(a: Any, b: Any, is_max: bool) -> bool:
        # [FUNCTION DOC]
        # HELPER: comparison used by the monotonic queue. Returns whether a is at
        # least b when computing a maximum, or at most b when computing a minimum.
        if is_max:
            return a >= b
        else:
            return a <= b

    # Indices of bounded extrema in `values`. `candidates[i]` is always increasing.
    # `values[candidates[i]]` is decreasing for max and increasing for min.
    candidates: list[int] = []  # this is a queue
    # Indices of largest windows that "cover" preceding windows.
    dominators: list[int] = []  # this is a stack

    if min_periods < 1:
        min_periods = 1

    if N > 2:
        i_next = N - 1  # equivalent to i_next = i+1 inside the loop
        for i in range(N - 2, -1, -1):
            next_dominates = start[i_next] < start[i]
            if next_dominates and (
                not dominators or start[dominators[-1]] > start[i_next]
            ):
                dominators.append(i_next)
            i_next = i

    # NaN tracking to guarantee min_periods
    valid_start = -min_periods

    last_end = 0
    last_start = -1

    for i in range(N):
        this_start = start[i].item()
        this_end = end[i].item()

        if dominators and dominators[-1] == i:
            dominators.pop()

        if not (
            this_end > last_end or (this_end == last_end and this_start >= last_start)
        ):
            raise ValueError(
                "Start/End ordering requirement is violated at index " + str(i)
            )

        stash_start = (
            this_start if not dominators else min(this_start, start[dominators[-1]])
        )
        while candidates and candidates[0] < stash_start:
            candidates.pop(0)

        for k in range(last_end, this_end):
            if not np.isnan(values[k]):
                valid_start += 1
                while valid_start >= 0 and np.isnan(values[valid_start]):
                    valid_start += 1
                while candidates and cmp(values[k], values[candidates[-1]], is_max):
                    candidates.pop()  # Q.pop_back()
                candidates.append(k)  # Q.push_back(k)

        if not candidates or (this_start > valid_start):
            if values.dtype.kind != "i":
                output[i] = np.nan
            else:
                na_pos.append(i)
        elif candidates[0] >= this_start:
            # ^^ This is here to avoid costly bisection for fixed window sizes.
            output[i] = values[candidates[0]]
        else:
            q_idx = bisect_left(candidates, this_start, lo=1)
            output[i] = values[candidates[q_idx]]
        last_end = this_end
        last_start = this_start

    return output, na_pos


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
    is_max: bool,
    skipna: bool = True,
) -> tuple[np.ndarray, list[int]]:
    # [FUNCTION DOC]
    # Groupby minimum or maximum kernel for one 1D array.
    # Inputs: values (1D array), result_dtype (output dtype), labels (group code per
    # element; negative codes are ignored), ngroups (number of groups),
    # min_periods (minimum observations per group), is_max (true for maximum) and
    # skipna (default true; if false, a group that contains a missing value
    # produces NaN and later elements of that group are ignored).
    # Behavior: counts observations per group (integers are always valid), takes
    # the first value of each group as the starting extremum and then updates it.
    # Returns a pair: the output array (uninitialized for groups with no
    # observations) and the list of group positions whose count is below
    # min_periods, which the caller must mask.
    N = len(labels)
    nobs = np.zeros(ngroups, dtype=np.int64)
    na_pos = []
    output = np.empty(ngroups, dtype=result_dtype)

    for i in range(N):
        lab = labels[i]
        val = values[i]
        if lab < 0 or (not skipna and nobs[lab] >= 1 and np.isnan(output[lab])):
            continue

        if values.dtype.kind == "i" or not np.isnan(val):
            nobs[lab] += 1
        else:
            if not skipna:
                # If skipna is False and we encounter a NaN,
                # both min and max of the group will be NaN
                output[lab] = np.nan
            continue

        if nobs[lab] == 1:
            # First element in group, set output equal to this
            output[lab] = val
            continue

        if is_max:
            if val > output[lab]:
                output[lab] = val
        elif val < output[lab]:
            output[lab] = val

    # Set labels that don't satisfy min_periods as np.nan
    for lab, count in enumerate(nobs):
        if count < min_periods:
            na_pos.append(lab)

    return output, na_pos
