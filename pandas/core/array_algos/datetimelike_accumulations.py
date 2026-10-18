"""
datetimelke_accumulations.py is for accumulations of datetimelike extension arrays
"""

# [MODULE DOC]
# Cumulative operations (sum, minimum, maximum) for one dimensional numpy arrays of
# datetime-like data (datetime64 / timedelta64 values handled as 64-bit integers
# with the integer "not a time" sentinel marking missing entries).
# Public contents: cumsum, cummin and cummax thin wrappers around the private
# shared implementation. Used by the datetime-like array classes. Import-time side
# effects: none.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import iNaT

from pandas.core.dtypes.missing import isna

if TYPE_CHECKING:
    from collections.abc import Callable


def _cum_func(
    func: Callable,
    values: np.ndarray,
    *,
    skipna: bool = True,
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: shared implementation of the cumulative operations.
    # Inputs: func, the numpy accumulation function (cumulative sum, or the
    # accumulating minimum or maximum ufunc method), values (1D datetime-like numpy
    # array) and keyword-only skipna (default true).
    # Behavior: the func is looked up in a table of neutral fill values (zero for
    # sum, the integer extremes for max and min); an unknown func raises a value
    # error. Missing positions are found, then the values are viewed as 64-bit
    # integers and the missing positions are overwritten with the neutral fill
    # value. SIDE EFFECT: this write goes through a view, so the input array is
    # modified in place. If skipna is false, the missing mask is propagated forward
    # (everything after the first missing entry is missing). The accumulation is
    # applied, the masked positions are set to the integer missing sentinel, and the
    # result is viewed back as the original datetime64 or timedelta64 dtype (other
    # dtypes are returned as integers).
    # KNOWN ISSUE: the error message names BaseMaskedArray, copied from the masked
    # accumulation module; it is misleading for datetime-like arrays.
    """
    Accumulations for 1D datetimelike arrays.

    Parameters
    ----------
    func : np.cumsum, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation). Values is changed is modified inplace.
    skipna : bool, default True
        Whether to skip NA.
    """
    try:
        fill_value = {
            np.maximum.accumulate: np.iinfo(np.int64).min,
            np.cumsum: 0,
            np.minimum.accumulate: np.iinfo(np.int64).max,
        }[func]
    except KeyError as err:
        raise ValueError(
            f"No accumulation for {func} implemented on BaseMaskedArray"
        ) from err

    mask = isna(values)
    y = values.view("i8")
    y[mask] = fill_value

    if not skipna:
        mask = np.maximum.accumulate(mask)

    # GH 57956
    result = func(y, axis=0)
    result[mask] = iNaT

    if values.dtype.kind in "mM":
        return result.view(values.dtype.base)
    return result


def cumsum(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    # [FUNCTION DOC]
    # Cumulative sum of a 1D datetime-like array (typically timedeltas). Inputs:
    # values, skipna (default true: missing entries are skipped and stay missing
    # in the output; if false, every entry after the first missing is missing).
    # Returns a new array; the input is modified in place (see the shared helper).
    return _cum_func(np.cumsum, values, skipna=skipna)


def cummin(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    # [FUNCTION DOC]
    # Running minimum of a 1D datetime-like array. Parameters and in-place side
    # effect as for the cumulative sum; returns a new array of the same dtype.
    return _cum_func(np.minimum.accumulate, values, skipna=skipna)


def cummax(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    # [FUNCTION DOC]
    # Running maximum of a 1D datetime-like array. Parameters and in-place side
    # effect as for the cumulative sum; returns a new array of the same dtype.
    return _cum_func(np.maximum.accumulate, values, skipna=skipna)
