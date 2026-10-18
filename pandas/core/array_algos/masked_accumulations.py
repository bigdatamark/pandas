"""
masked_accumulations.py is for accumulation algorithms using a mask-based approach
for missing values.
"""

# [MODULE DOC]
# Cumulative operations (sum, product, minimum, maximum) for one dimensional numpy
# data accompanied by a separate boolean mask marking missing entries, as used by
# the nullable masked extension arrays.
# Public contents: cumsum, cumprod, cummin, cummax wrappers around a private shared
# implementation. Import-time side effects: none.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import npt


def _cum_func(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # HELPER: shared implementation of the masked cumulative operations.
    # Inputs: func (numpy cumulative sum, cumulative product, or the accumulating
    # minimum or maximum ufunc method), values (1D numpy array of float, integer or
    # boolean kind), mask (boolean array, true means missing) and keyword-only
    # skipna (default true).
    # Behavior: the dtype limits are determined (booleans use the unsigned 8-bit
    # limits); any other dtype kind raises a not-implemented error. The neutral
    # fill value is chosen per func (one for product, zero for sum, dtype minimum
    # for max, dtype maximum for min); an unknown func raises a not-implemented
    # error. SIDE EFFECT: missing positions of the input values are overwritten
    # in place with the fill value. If skipna is false the mask is propagated
    # forward. Then the accumulation is applied.
    # Returns a pair: the accumulated values and the (possibly propagated) mask.
    # KNOWN ISSUE: the error message names BaseMaskedArray, although this is a
    # generic helper operating on plain arrays.
    """
    Accumulations for 1D masked array.

    We will modify values in place to replace NAs with the appropriate fill value.

    Parameters
    ----------
    func : np.cumsum, np.cumprod, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    """
    dtype_info: np.iinfo | np.finfo
    if values.dtype.kind == "f":
        dtype_info = np.finfo(values.dtype.type)
    elif values.dtype.kind in "iu":
        dtype_info = np.iinfo(values.dtype.type)
    elif values.dtype.kind == "b":
        # Max value of bool is 1, but since we are setting into a boolean
        # array, 255 is fine as well. Min value has to be 0 when setting
        # into the boolean array.
        dtype_info = np.iinfo(np.uint8)
    else:
        raise NotImplementedError(
            f"No masked accumulation defined for dtype {values.dtype.type}"
        )
    try:
        fill_value = {
            np.cumprod: 1,
            np.maximum.accumulate: dtype_info.min,
            np.cumsum: 0,
            np.minimum.accumulate: dtype_info.max,
        }[func]
    except KeyError as err:
        raise NotImplementedError(
            f"No accumulation for {func} implemented on BaseMaskedArray"
        ) from err

    values[mask] = fill_value

    if not skipna:
        mask = np.maximum.accumulate(mask)

    values = func(values)
    return values, mask


def cumsum(
    values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Cumulative sum with a missing-value mask. Inputs: values, mask (true is
    # missing) and keyword-only skipna (default true). Returns the pair of
    # accumulated values and resulting mask; modifies the input values in place at
    # missing positions (see the shared helper).
    return _cum_func(np.cumsum, values, mask, skipna=skipna)


def cumprod(
    values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Cumulative product with a missing-value mask; same parameters, return value
    # and in-place side effect as the cumulative sum.
    return _cum_func(np.cumprod, values, mask, skipna=skipna)


def cummin(
    values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Running minimum with a missing-value mask; same parameters, return value and
    # in-place side effect as the cumulative sum.
    return _cum_func(np.minimum.accumulate, values, mask, skipna=skipna)


def cummax(
    values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Running maximum with a missing-value mask; same parameters, return value and
    # in-place side effect as the cumulative sum.
    return _cum_func(np.maximum.accumulate, values, mask, skipna=skipna)
