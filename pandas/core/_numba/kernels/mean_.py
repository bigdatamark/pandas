"""
Numba 1D mean kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

# [MODULE DOC]
# HELPER: internal numba kernels for the arithmetic mean, used by the numba engine of
# rolling/expanding windows and groupby.
# Public contents: add_mean and remove_mean (single-step accumulator updates using
# Kahan compensated summation), sliding_mean (window kernel driven by start/end
# bound arrays) and grouped_mean (groupby kernel driven by integer group labels).
# Relationships: uses the monotonic bounds check from the shared kernels module
# and the compensated grouped sum from the sum kernels module; results are
# consumed by the looper in the executor module. Importing requires numba, and all
# functions are compiled lazily on first call (no other import-time side effects).
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from pandas.core._numba.kernels.shared import is_monotonic_increasing
from pandas.core._numba.kernels.sum_ import grouped_kahan_sum

if TYPE_CHECKING:
    from pandas._typing import npt


@numba.jit(nopython=True, nogil=True, parallel=False)
def add_mean(
    val: float,
    nobs: int,
    sum_x: float,
    neg_ct: int,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[int, float, int, float, int, float]:
    # [FUNCTION DOC]
    # HELPER: adds one observation to the running mean state of a window.
    # Inputs: val (the new value), nobs (count of non-missing values), sum_x
    # (running sum), neg_ct (count of negative values), compensation (Kahan
    # compensation term), num_consecutive_same_value (length of the current run of
    # equal values) and prev_value (previous non-missing value).
    # Missing (NaN) values leave all state unchanged. Otherwise the count is
    # incremented, the value is added with compensated summation, the negative
    # count is incremented for negative values and the run length is extended (if
    # equal to the previous value) or reset to one; the previous value is updated.
    # Returns the updated tuple (nobs, sum_x, neg_ct, compensation,
    # num_consecutive_same_value, prev_value).
    if not np.isnan(val):
        nobs += 1
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct += 1

        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

    return nobs, sum_x, neg_ct, compensation, num_consecutive_same_value, prev_value


@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_mean(
    val: float, nobs: int, sum_x: float, neg_ct: int, compensation: float
) -> tuple[int, float, int, float]:
    # [FUNCTION DOC]
    # HELPER: removes one observation from the running mean state when it leaves a
    # window. Inputs: val, nobs, sum_x, neg_ct and compensation, as in the add
    # step. A missing value changes nothing; otherwise the count is decremented,
    # the value is subtracted with compensated summation and the negative count is
    # decremented for negative values. Returns the updated tuple (nobs, sum_x,
    # neg_ct, compensation).
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct -= 1
    return nobs, sum_x, neg_ct, compensation


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_mean(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    # [FUNCTION DOC]
    # Window mean kernel for one 1D array of floats.
    # Inputs: values (1D array), result_dtype (dtype of the output array), start and
    # end (integer arrays with one window each, half open ranges), min_periods
    # (minimum non-missing observations needed for a result).
    # Behavior: when the bounds are monotonically increasing, the window state is
    # updated incrementally (remove values that left, add values that entered);
    # otherwise the state is recomputed from scratch for every window. The mean is
    # the compensated sum divided by the count, but it is replaced by the repeated
    # value when all observations are equal, and clamped to zero when its sign
    # contradicts the signs of the inputs (all non-negative inputs but negative
    # result, or all negative inputs but positive result). Windows with fewer
    # than min_periods observations, or no observations, give NaN.
    # Returns a pair: the output array (one entry per window) and an empty list of
    # integer positions (no positions need masking as floats hold NaN).
    N = len(start)
    nobs = 0
    sum_x = 0.0
    neg_ct = 0
    compensation_add = 0.0
    compensation_remove = 0.0

    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    output = np.empty(N, dtype=result_dtype)

    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not is_monotonic_increasing_bounds:
            prev_value = values[s]
            num_consecutive_same_value = 0

            for j in range(s, e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_mean(
                    val,
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,  # pyright: ignore[reportArgumentType]
                )
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, neg_ct, compensation_remove = remove_mean(
                    val, nobs, sum_x, neg_ct, compensation_remove
                )

            for j in range(end[i - 1], e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_mean(
                    val,
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,  # pyright: ignore[reportArgumentType]
                )

        if nobs >= min_periods and nobs > 0:
            result = sum_x / nobs
            if num_consecutive_same_value >= nobs:
                result = prev_value
            elif neg_ct == 0 and result < 0:
                result = 0
            elif neg_ct == nobs and result > 0:
                result = 0
        else:
            result = np.nan

        output[i] = result

        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0.0
            neg_ct = 0
            compensation_remove = 0.0

    # na_position is empty list since float64 can already hold nans
    # Do list comprehension, since numba cannot figure out that na_pos is
    # empty list of ints on its own
    na_pos = [0 for i in range(0)]
    return output, na_pos


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_mean(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
    skipna: bool,
) -> tuple[np.ndarray, list[int]]:
    # [FUNCTION DOC]
    # Groupby mean kernel for one 1D array.
    # Inputs: values (1D array), result_dtype (output dtype), labels (group code per
    # element; negative codes are ignored), ngroups (number of groups),
    # min_periods (minimum observations per group) and skipna (whether missing
    # values are skipped; if false a group containing a missing value gives NaN).
    # Behavior: computes per-group compensated sums, counts, and runs of equal
    # values via the shared grouped sum helper, then for each group divides by the
    # count; groups with fewer than min_periods observations become NaN, and a
    # group whose values are all equal uses the repeated value times the count to
    # avoid rounding error.
    # Returns the per-group output array and an empty list of positions to mask.
    # KNOWN ISSUE: the division by the count is applied to every group, including
    # groups already set to NaN and groups with zero observations (when
    # min_periods is zero this is zero divided by zero), instead of only to groups
    # that satisfied min_periods.
    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(
        values, result_dtype, labels, ngroups, skipna
    )

    # Post-processing, replace sums that don't satisfy min_periods
    for lab in range(ngroups):
        nobs = nobs_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]
        sum_x = output[lab]
        if nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = np.nan
        result /= nobs
        output[lab] = result

    # na_position is empty list since float64 can already hold nans
    # Do list comprehension, since numba cannot figure out that na_pos is
    # empty list of ints on its own
    na_pos = [0 for i in range(0)]
    return output, na_pos
