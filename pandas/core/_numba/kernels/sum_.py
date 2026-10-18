"""
Numba 1D sum kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

# [MODULE DOC]
# HELPER: internal numba kernels for the sum, used by the numba engine of
# rolling/expanding windows and groupby, with Kahan compensated summation.
# Public contents: add_sum and remove_sum (single-step accumulator updates),
# sliding_sum (window kernel driven by start/end bounds), grouped_kahan_sum
# (reusable jittable helper returning the per-group sums and bookkeeping arrays,
# also used by the mean kernel) and grouped_sum (groupby kernel driven by labels).
# Importing requires numba; functions are compiled lazily on first call.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numba
from numba.extending import register_jitable
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt

from pandas.core._numba.kernels.shared import is_monotonic_increasing


@numba.jit(nopython=True, nogil=True, parallel=False)
def add_sum(
    val: Any,
    nobs: int,
    sum_x: Any,
    compensation: Any,
    num_consecutive_same_value: int,
    prev_value: Any,
) -> tuple[int, Any, Any, int, Any]:
    # [FUNCTION DOC]
    # HELPER: adds one observation to the running sum state of a window or group.
    # Inputs: val, nobs (count), sum_x (running sum), compensation (Kahan term),
    # num_consecutive_same_value (length of the current run of equal values) and
    # prev_value. A missing value changes nothing. Otherwise the count is
    # incremented, the value is added with compensated summation and the run
    # length is extended or reset to one; prev_value is updated.
    # Returns the updated tuple (nobs, sum_x, compensation,
    # num_consecutive_same_value, prev_value).
    if not np.isnan(val):
        nobs += 1
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t

        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

    return nobs, sum_x, compensation, num_consecutive_same_value, prev_value


@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_sum(
    val: Any, nobs: int, sum_x: Any, compensation: Any
) -> tuple[int, Any, Any]:
    # [FUNCTION DOC]
    # HELPER: removes one observation from the running sum state when it leaves a
    # window. Inputs: val, nobs, sum_x and compensation. A missing value changes
    # nothing; otherwise the count is decremented and the value subtracted with
    # compensated summation. Returns the updated tuple (nobs, sum_x, compensation).
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
    return nobs, sum_x, compensation


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    # [FUNCTION DOC]
    # Window sum kernel for one 1D array.
    # Inputs: values (1D array of floats or integers), result_dtype (output dtype),
    # start and end (integer arrays, one half open window each), min_periods
    # (minimum non-missing observations).
    # Behavior: with monotonically increasing bounds the state is updated
    # incrementally (values leaving are removed, values entering are added),
    # otherwise recomputed per window. A window with zero observations and zero
    # min_periods gives zero; a window with enough observations gives the
    # compensated sum, or the repeated value times the count if all observations
    # are equal (to avoid rounding drift); otherwise the result is NaN for floats,
    # and for integer input a placeholder zero is stored and the position is
    # recorded for masking.
    # Returns a pair: the output array and the list of positions to mask.
    dtype = values.dtype

    na_val: object = np.nan
    if dtype.kind == "i":
        na_val = 0

    N = len(start)
    nobs = 0
    sum_x = 0
    compensation_add = 0
    compensation_remove = 0
    na_pos = []

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
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_sum(
                    val,
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, compensation_remove = remove_sum(
                    val, nobs, sum_x, compensation_remove
                )

            for j in range(end[i - 1], e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_sum(
                    val,
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        if nobs == 0 == min_periods:
            result: object = 0
        elif nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = na_val
            if dtype.kind == "i":
                na_pos.append(i)

        output[i] = result

        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0
            compensation_remove = 0

    return output, na_pos


@register_jitable
def grouped_kahan_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    skipna: bool,
) -> tuple[
    np.ndarray, npt.NDArray[np.int64], np.ndarray, npt.NDArray[np.int64], np.ndarray
]:
    # [FUNCTION DOC]
    # HELPER: shared jittable routine computing per-group compensated sums, used by
    # the grouped sum and grouped mean kernels.
    # Inputs: values (1D array), result_dtype (dtype of the sums), labels (group
    # code per element; negative codes are ignored), ngroups and skipna (if false,
    # a missing value poisons its group: the sum becomes NaN and later elements of
    # that group are skipped).
    # Returns five arrays of length ngroups: sums, observation counts, Kahan
    # compensation terms, lengths of the trailing run of equal values, and the last
    # value seen per group. Groups already NaN are skipped.
    N = len(labels)

    nobs_arr = np.zeros(ngroups, dtype=np.int64)
    comp_arr = np.zeros(ngroups, dtype=values.dtype)
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)
    prev_vals = np.zeros(ngroups, dtype=values.dtype)
    output = np.zeros(ngroups, dtype=result_dtype)

    for i in range(N):
        lab = labels[i]
        val = values[i]

        if lab < 0 or np.isnan(output[lab]):
            continue

        if not skipna and np.isnan(val):
            output[lab] = np.nan
            nobs_arr[lab] += 1
            comp_arr[lab] = np.nan
            consecutive_counts[lab] = 1
            prev_vals[lab] = np.nan
            continue

        sum_x = output[lab]
        nobs = nobs_arr[lab]
        compensation_add = comp_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]

        (
            nobs,
            sum_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        ) = add_sum(
            val,
            nobs,
            sum_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        )

        output[lab] = sum_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs
    return output, nobs_arr, comp_arr, consecutive_counts, prev_vals


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
    skipna: bool,
) -> tuple[np.ndarray, list[int]]:
    # [FUNCTION DOC]
    # Groupby sum kernel for one 1D array.
    # Inputs: values, result_dtype, labels (negative codes ignored), ngroups,
    # min_periods (minimum observations per group) and skipna (see the shared
    # grouped helper).
    # Behavior: computes per-group compensated sums; a group whose values are all
    # equal uses the repeated value times the count. Groups with fewer than
    # min_periods observations keep their sum but their position is recorded.
    # Returns a pair: the output array and the list of group positions that the
    # caller must mask as missing.
    na_pos = []

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
            result = sum_x  # Don't change val, will be replaced by nan later
            na_pos.append(lab)
        output[lab] = result

    return output, na_pos
