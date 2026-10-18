# [MODULE DOC]
# Quantile computation for numpy arrays and extension arrays, skipping missing
# values.
# Public contents: quantile_compat (dispatcher: numpy arrays are handled here,
# extension arrays use their own quantile method) and quantile_with_mask (core
# routine for two dimensional data with an explicit missing-value mask and fill
# value). Private helpers wrap numpy quantile for a row and for a 2D block, with
# special handling for datetime-like data (computed on the integer
# representation) and for restoring integer dtypes. Import-time side effects:
# none.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Scalar,
        npt,
    )


def quantile_compat(
    values: ArrayLike, qs: npt.NDArray[np.float64], interpolation: str
) -> ArrayLike:
    # [FUNCTION DOC]
    # Computes quantiles of an array for each requested quantile level. Inputs:
    # values (numpy array or extension array), qs (1D float array of levels
    # between zero and one) and interpolation (method name passed on to numpy).
    # For a numpy array it builds the missing mask and the missing fill value for
    # its dtype and delegates to the mask-based routine; for an extension array it
    # calls the array's own quantile method. Returns an array of quantiles (the
    # array type depends on the input kind).
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    qs : np.ndarray[float64]
    interpolation : str

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    if isinstance(values, np.ndarray):
        fill_value = na_value_for_dtype(values.dtype, compat=False)
        mask = isna(values)
        return quantile_with_mask(values, mask, fill_value, qs, interpolation)
    else:
        return values._quantile(qs, interpolation)


def quantile_with_mask(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    fill_value,
    qs: npt.NDArray[np.float64],
    interpolation: str,
) -> np.ndarray:
    # [FUNCTION DOC]
    # Core quantile routine with an explicit missing mask. Inputs: values (numpy
    # array, one or two dimensional), mask (boolean array of the same shape, true
    # means missing; an assertion error is raised if shapes differ), fill_value
    # (value used for rows with no valid data), qs (levels) and interpolation.
    # Behavior: a 1D input is promoted to 2D, processed, and the single result row
    # returned. For 2D input quantiles are taken along the second axis: if there are
    # no columns, the result is filled with fill_value; otherwise it is computed by
    # the private 2D helper and transposed.
    # Returns an array whose first dimension is the number of rows (or, in 1D, just
    # the quantile levels) and second the number of levels.
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray
        For ExtensionArray, this is _values_for_factorize()[0]
    mask : np.ndarray[bool]
        mask = isna(values)
        For ExtensionArray, this is computed before calling _value_for_factorize
    fill_value : Scalar
        The value to interpret fill NA entries with
        For ExtensionArray, this is _values_for_factorize()[1]
    qs : np.ndarray[float64]
    interpolation : str
        Type of interpolation

    Returns
    -------
    np.ndarray

    Notes
    -----
    Assumes values is already 2D.  For ExtensionArray this means np.atleast_2d
    has been called on _values_for_factorize()[0]

    Quantile is computed along axis=1.
    """
    assert values.shape == mask.shape
    if values.ndim == 1:
        # unsqueeze, operate, re-squeeze
        values = np.atleast_2d(values)
        mask = np.atleast_2d(mask)
        res_values = quantile_with_mask(values, mask, fill_value, qs, interpolation)
        return res_values[0]

    assert values.ndim == 2

    is_empty = values.shape[1] == 0

    if is_empty:
        # create the array of na_values
        # 2d len(values) * len(qs)
        flat = np.full(len(qs), fill_value)
        result = np.repeat(flat, len(values)).reshape(len(values), len(qs))
    else:
        result = _nanquantile(
            values,
            qs,
            na_value=fill_value,
            mask=mask,
            interpolation=interpolation,
        )

        result = np.asarray(result)
        result = result.T

    return result


def _nanquantile_1d(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    qs: npt.NDArray[np.float64],
    na_value: Scalar,
    interpolation: str,
) -> Scalar | np.ndarray:
    # [FUNCTION DOC]
    # HELPER: quantiles of a single 1D row ignoring masked entries, used by the 2D
    # helper. Inputs: values, mask (true is missing), qs, na_value (returned for
    # every level when no valid values remain) and interpolation method. Returns an
    # array of one result per level (filled with na_value if the row has no valid
    # values).
    """
    Wrapper for np.quantile that skips missing values, specialized to
    1-dimensional case.

    Parameters
    ----------
    values : array over which to find quantiles
    mask : ndarray[bool]
        locations in values that should be considered missing
    qs : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
    # mask is Union[ExtensionArray, ndarray]
    values = values[~mask]

    if len(values) == 0:
        # Can't pass dtype=values.dtype here bc we might have na_value=np.nan
        #  with values.dtype=int64 see test_quantile_empty
        # equiv: 'np.array([na_value] * len(qs))' but much faster
        return np.full(len(qs), na_value)

    return np.quantile(
        values,
        qs,
        # error: No overload variant of "percentile" matches argument
        # types "ndarray[Any, Any]", "ndarray[Any, dtype[floating[_64Bit]]]"
        # , "Dict[str, str]"  [call-overload]
        method=interpolation,  # type: ignore[call-overload]
    )


def _nanquantile(
    values: np.ndarray,
    qs: npt.NDArray[np.float64],
    *,
    na_value,
    mask: npt.NDArray[np.bool_],
    interpolation: str,
):
    # [FUNCTION DOC]
    # HELPER: quantiles along axis one of a 2D block while skipping masked entries.
    # Inputs: values (2D array), qs, keyword-only na_value, mask and interpolation.
    # Behavior: datetime64 / timedelta64 data is processed as 64-bit integers
    # (recursive call) and the result converted back to the original dtype. If any
    # entry is masked (shapes must match, assertion error otherwise), each row is
    # processed separately, floats keep their item size, and for other dtypes the
    # result is cast back to the input dtype only when not everything is masked and
    # all result values are exactly representable. If nothing is masked, a single
    # vectorized numpy quantile call is used.
    # Returns an array with the quantile levels along the first axis and the
    # input rows along the second (both the masked and the vectorized paths give
    # this orientation); the caller transposes it.
    """
    Wrapper for np.quantile that skips missing values.

    Parameters
    ----------
    values : np.ndarray[ndim=2]  over which to find quantiles
    qs : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    mask : np.ndarray[bool]
        locations in values that should be considered missing
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """

    if values.dtype.kind in "mM":
        # need to cast to integer to avoid rounding errors in numpy
        result = _nanquantile(
            values.view("i8"),
            qs=qs,
            na_value=na_value.view("i8"),
            mask=mask,
            interpolation=interpolation,
        )

        # Note: we have to do `astype` and not view because in general we
        #  have float result at this point, not i8
        return result.astype(values.dtype)

    if mask.any():
        # Caller is responsible for ensuring mask shape match
        assert mask.shape == values.shape
        result = [
            _nanquantile_1d(val, m, qs, na_value, interpolation=interpolation)
            for (val, m) in zip(list(values), list(mask), strict=True)
        ]
        if values.dtype.kind == "f":
            # preserve itemsize
            result = np.asarray(result, dtype=values.dtype).T
        else:
            result = np.asarray(result).T
            if (
                result.dtype != values.dtype
                and not mask.all()
                and (result == result.astype(values.dtype, copy=False)).all()
            ):
                # mask.all() will never get cast back to int
                # e.g. values id integer dtype and result is floating dtype,
                #  only cast back to integer dtype if result values are all-integer.
                result = result.astype(values.dtype, copy=False)
        return result
    else:
        return np.quantile(
            values,
            qs,
            axis=1,
            # error: No overload variant of "percentile" matches argument types
            # "ndarray[Any, Any]", "ndarray[Any, dtype[floating[_64Bit]]]",
            # "int", "Dict[str, str]"  [call-overload]
            method=interpolation,  # type: ignore[call-overload]
        )
