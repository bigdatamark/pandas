"""
masked_reductions.py is for reduction algorithms using a mask-based approach
for missing values.
"""

# [MODULE DOC]
# Reductions (sum, product, minimum, maximum, mean, variance, standard deviation)
# for numpy data accompanied by a boolean mask marking missing entries, as used by
# the nullable masked extension arrays. Results that cannot be computed are
# returned as the pandas missing-value singleton.
# Public contents: sum, prod, min, max, mean, var, std (these deliberately shadow
# the builtin names inside this module) plus private shared helpers for the
# sum/product/mean family and for min/max. Depends on the nanops helper that checks
# the minimum count. Import-time side effects: none.
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)

from pandas.core.nanops import check_below_min_count

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        AxisInt,
        npt,
    )


def _reductions(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
    initial: object | lib.NoDefault = lib.no_default,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: shared implementation for sum, product, mean, variance and standard
    # deviation. Inputs: func (numpy reduction function), values, mask (true is
    # missing), keyword-only skipna (default true), min_count (default 0, minimum
    # number of valid values), axis (default none, reduce everything), initial
    # (optional starting value, only forwarded when given) and extra keyword
    # arguments forwarded to func (such as degrees of freedom).
    # Behavior: when skipna is false, returns the missing singleton if any value is
    # masked or the size is below min_count, else reduces the raw values. When
    # skipna is true, returns the missing singleton if the number of valid values
    # is below min_count (only checked for full or 1D reductions), else reduces
    # with the masked positions excluded. Returns a scalar or array result.
    """
    Sum, mean or product for 1D masked array.

    Parameters
    ----------
    func : np.sum or np.prod
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    min_count : int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.
    axis : int, optional, default None
    initial : scalar, optional
        Starting value for the reduction. NumPy has a default value for most
        data types, but for object-dtype arrays we need to specify it explicitly
    """
    if initial is not lib.no_default:
        kwargs["initial"] = initial

    if not skipna:
        if mask.any() or check_below_min_count(values.shape, None, min_count):
            return libmissing.NA
        else:
            return func(values, axis=axis, **kwargs)
    else:
        if check_below_min_count(values.shape, mask, min_count) and (
            axis is None or values.ndim == 1
        ):
            return libmissing.NA

        return func(values, where=~mask, axis=axis, **kwargs)


def sum(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
    initial: object | lib.NoDefault = lib.no_default,
):
    # [FUNCTION DOC]
    # Masked sum. Inputs: values, mask, keyword-only skipna, min_count, axis and
    # initial (see the shared helper). Returns the sum or the missing singleton.
    return _reductions(
        np.sum,
        values=values,
        mask=mask,
        skipna=skipna,
        min_count=min_count,
        axis=axis,
        initial=initial,
    )


def prod(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = None,
):
    # [FUNCTION DOC]
    # Masked product. Inputs: values, mask, keyword-only skipna, min_count and axis
    # (see the shared helper; no initial value). Returns the product or the missing
    # singleton.
    return _reductions(
        np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis
    )


def _minmax(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    # [FUNCTION DOC]
    # HELPER: shared implementation of minimum and maximum. Inputs: func (numpy
    # min or max), values, mask (true is missing), keyword-only skipna (default
    # true) and axis (default none).
    # Behavior: if skipna is false, returns the missing singleton when any value
    # is masked or the array is empty, otherwise applies func to all values. If
    # skipna is true, applies func to the unmasked values, or returns the missing
    # singleton if none remain (numpy would raise on empty input).
    """
    Reduction for 1D masked array.

    Parameters
    ----------
    func : np.min or np.max
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    axis : int, optional, default None
    """
    if not skipna:
        if mask.any() or not values.size:
            # min/max with empty array raise in numpy, pandas returns NA
            return libmissing.NA
        else:
            return func(values, axis=axis)
    else:
        subset = values[~mask]
        if subset.size:
            return func(subset, axis=axis)
        else:
            # min/max with empty array raise in numpy, pandas returns NA
            return libmissing.NA


def min(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    # [FUNCTION DOC]
    # Masked minimum. Inputs: values, mask, keyword-only skipna (default true) and
    # axis. Returns the minimum of the valid values or the missing singleton.
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)


def max(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    # [FUNCTION DOC]
    # Masked maximum. Inputs: values, mask, keyword-only skipna (default true) and
    # axis. Returns the maximum of the valid values or the missing singleton.
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)


def mean(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
):
    # [FUNCTION DOC]
    # Masked mean. Inputs: values, mask, keyword-only skipna (default true) and
    # axis. Returns the missing singleton if the array is empty or fully masked;
    # otherwise the mean via the shared helper.
    if not values.size or mask.all():
        return libmissing.NA
    return _reductions(np.mean, values=values, mask=mask, skipna=skipna, axis=axis)


def var(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
    ddof: int = 1,
):
    # [FUNCTION DOC]
    # Masked variance. Inputs: values, mask, keyword-only skipna (default true),
    # axis and ddof (delta degrees of freedom, default 1). Returns the missing
    # singleton if the array is empty or fully masked; otherwise the variance via
    # the shared helper, with numpy runtime warnings (such as too few degrees of
    # freedom) suppressed during the computation.
    if not values.size or mask.all():
        return libmissing.NA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _reductions(
            np.var, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof
        )


def std(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: AxisInt | None = None,
    ddof: int = 1,
):
    # [FUNCTION DOC]
    # Masked standard deviation. Same inputs, empty/fully-masked behavior and
    # warning suppression as the masked variance, but returns the standard
    # deviation.
    if not values.size or mask.all():
        return libmissing.NA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _reductions(
            np.std, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof
        )
