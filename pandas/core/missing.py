"""
Routines for filling missing data.
"""

# [MODULE DOC]
# Routines for filling and interpolating missing data on numpy arrays: forward and backward fill with limits and area restrictions, interpolation along an axis (numpy or scipy based), mask construction for replace, and validation helpers for the related keyword arguments.
# Public contents: fill-size and mask helpers, fill-method and interpolate-method cleaners, argument validators, the in-place two-dimensional interpolation entry point, in-place pad/backfill entry point, the fill-function selector, and the module lists of numpy-backed and scipy-backed interpolation method names. Private helpers implement the one-dimensional interpolation, scipy dispatch and the limit bookkeeping.
# Used by Block-level and array-level fill and interpolate operations. Interpolation methods other than the numpy ones require the optional scientific library, imported lazily. Most fill functions mutate their inputs in place. Import-time side effects: builds a module-level table mapping fill method names to the one-dimensional fill functions.

from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._config import is_nan_na

from pandas._libs import (
    NaT,
    algos,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    F,
    ReindexMethod,
    npt,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    BaseMaskedDtype,
    DatetimeTZDtype,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from pandas import Index

    _CubicBC: TypeAlias = Literal["not-a-knot", "clamped", "natural", "periodic"]


def check_value_size(value, mask: npt.NDArray[np.bool_], length: int):
    # [FUNCTION DOC]
    # Validate the value passed to an extension array fill.
    # Parameters: value (scalar or array-like), mask (boolean array of positions to fill), length (expected length of an array-like value).
    # If value is array-like and its length differs from length, raises a value error. An array-like value is reduced to the entries at masked positions; scalars are returned unchanged.
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
    if is_array_like(value):
        if len(value) != length:
            raise ValueError(
                f"Length of 'value' does not match. Got ({len(value)}) "
                f" expected {length}"
            )
        value = value[mask]

    return value


def mask_missing(arr: ArrayLike, value) -> npt.NDArray[np.bool_]:
    # [FUNCTION DOC]
    # Build a boolean mask of the entries of arr that equal value (used by replace).
    # Parameters: arr (array or extension array); value (a scalar already checked by the caller to be non-list-like and holdable by arr).
    # Behavior: for masked or Arrow floating dtypes with a NaN value (when NaN is not treated as missing), the mask is the positions holding actual NaN data that are not missing; for such integer dtypes the mask is all false. A missing value matches missing entries. Otherwise starts with all false and leaves it all false for incompatible pairs (numeric non-boolean array with a boolean value, boolean array with a non-boolean numeric value, numeric array with a string); object arrays compare only the non-missing entries; other arrays compare equal and, if the result is an extension boolean array, convert it to a numpy mask with missing treated as false.
    # Returns a boolean ndarray with the shape of arr.
    """
    Return a masking array of same size/shape as arr
    with entries equaling value set to True.

    Parameters
    ----------
    arr : ArrayLike
    value : scalar-like
        Caller has ensured `not is_list_like(value)` and that it can be held
        by `arr`.

    Returns
    -------
    np.ndarray[bool]
    """
    dtype, value = infer_dtype_from(value)

    if (
        isinstance(arr.dtype, (BaseMaskedDtype, ArrowDtype))
        and lib.is_float(value)
        and np.isnan(value)
        and not is_nan_na()
    ):
        # TODO: this should be done in an EA method?
        if arr.dtype.kind == "f":
            # GH#55127
            if isinstance(arr.dtype, BaseMaskedDtype):
                # error: "ExtensionArray" has no attribute "_data"  [attr-defined]
                mask = np.isnan(arr._data) & ~arr.isna()  # type: ignore[attr-defined,operator]
                return mask
            else:
                # error: "ExtensionArray" has no attribute "_pa_array"  [attr-defined]
                import pyarrow.compute as pc

                mask = pc.is_nan(arr._pa_array).fill_null(False).to_numpy()  # type: ignore[attr-defined]
                return mask

        elif arr.dtype.kind in "iu":
            # GH#51237
            mask = np.zeros(arr.shape, dtype=bool)
            return mask

    if isna(value):
        return isna(arr)

    # GH 21977
    mask = np.zeros(arr.shape, dtype=bool)
    if (
        is_numeric_dtype(arr.dtype)
        and not is_bool_dtype(arr.dtype)
        and lib.is_bool(value)
    ):
        # e.g. test_replace_ea_float_with_bool, see GH#62048
        pass
    elif (
        is_bool_dtype(arr.dtype) and is_numeric_dtype(dtype) and not lib.is_bool(value)
    ):
        # e.g. test_replace_ea_float_with_bool, see GH#62048
        pass
    elif is_numeric_dtype(arr.dtype) and isinstance(value, str):
        # GH#29553 prevent numpy deprecation warnings
        pass
    elif is_object_dtype(arr.dtype):
        # pre-compute mask to avoid comparison to NA
        # e.g. test_replace_na_in_obj_column
        arr_mask = ~isna(arr)
        mask[arr_mask] = arr[arr_mask] == value
    else:
        new_mask = arr == value

        if not isinstance(new_mask, np.ndarray):
            # usually BooleanArray
            new_mask = new_mask.to_numpy(dtype=bool, na_value=False)
        mask = new_mask

    return mask


@overload
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill"],
    *,
    allow_nearest: Literal[False] = ...,
) -> Literal["pad", "backfill"]:
    # [FUNCTION DOC]
    # Typed overload stub (no nearest allowed): accepts forward/backward fill names and returns the normalized pad or backfill name. Behavior is in the implementation below.
    ...


@overload
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: Literal[True],
) -> Literal["pad", "backfill", "nearest"]:
    # [FUNCTION DOC]
    # Typed overload stub (nearest allowed): accepts the fill names plus nearest and returns the normalized name. Behavior is in the implementation below.
    ...


def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: bool = False,
) -> Literal["pad", "backfill", "nearest"]:
    # [FUNCTION DOC]
    # Implementation: normalize a fill method name.
    # Parameters: method (string, case-insensitive; the aliases ffill and bfill map to pad and backfill), allow_nearest (keyword, default False; permits the nearest method).
    # Raises a value error stating the expected names if the method is not valid. Returns the normalized name. Non-string inputs are not normalized and fail the validity check.
    if isinstance(method, str):
        # error: Incompatible types in assignment (expression has type "str", variable
        # has type "Literal['ffill', 'pad', 'bfill', 'backfill', 'nearest']")
        method = method.lower()  # type: ignore[assignment]
        if method == "ffill":
            method = "pad"
        elif method == "bfill":
            method = "backfill"

    valid_methods = ["pad", "backfill"]
    expecting = "pad (ffill) or backfill (bfill)"
    if allow_nearest:
        valid_methods.append("nearest")
        expecting = "pad (ffill), backfill (bfill) or nearest"
    if method not in valid_methods:
        raise ValueError(f"Invalid fill method. Expecting {expecting}. Got {method}")
    return method


# interpolation methods that dispatch to np.interp

NP_METHODS = ["linear", "time", "index", "values"]

# interpolation methods that dispatch to _interpolate_scipy_wrapper

SP_METHODS = [
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "krogh",
    "spline",
    "polynomial",
    "from_derivatives",
    "piecewise_polynomial",
    "pchip",
    "akima",
    "cubicspline",
]


def clean_interp_method(method: str, index: Index, **kwargs) -> str:
    # [FUNCTION DOC]
    # Validate an interpolation method name.
    # Parameters: method (string); index (the index whose monotonicity is checked for some methods); keyword options, of which only the order is read.
    # Raises a value error if spline or polynomial lacks an order, if the method is not among the numpy and scipy methods, or if krogh, piecewise polynomial or pchip is requested with a non-monotonic-increasing index. Returns the method unchanged.
    order = kwargs.get("order")

    if method in ("spline", "polynomial") and order is None:
        raise ValueError("You must specify the order of the spline or polynomial.")

    valid = NP_METHODS + SP_METHODS
    if method not in valid:
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")

    if method in ("krogh", "piecewise_polynomial", "pchip"):
        if not index.is_monotonic_increasing:
            raise ValueError(
                f"{method} interpolation requires that the index be monotonic."
            )

    return method


def find_valid_index(how: str, is_valid: npt.NDArray[np.bool_]) -> int | None:
    # [FUNCTION DOC]
    # Position of the first or last valid entry.
    # Parameters: how ("first" or "last", asserted), is_valid (boolean array; two-dimensional input is first reduced with any across columns). Returns none for an empty array or if there is no valid entry, else the integer position.
    """
    Retrieves the positional index of the first valid value.

    Parameters
    ----------
    how : {'first', 'last'}
        Use this parameter to change between the first or last valid index.
    is_valid: np.ndarray
        Mask to find na_values.

    Returns
    -------
    int or None
    """
    assert how in ["first", "last"]

    if len(is_valid) == 0:  # early stop
        return None

    if is_valid.ndim == 2:
        # reduce axis 1
        is_valid = is_valid.any(axis=1)  # type: ignore[assignment]

    if how == "first":
        idxpos = is_valid[::].argmax()

    elif how == "last":
        idxpos = len(is_valid) - 1 - is_valid[::-1].argmax()

    chk_notna = is_valid[idxpos]

    if not chk_notna:
        return None
    # Incompatible return value type (got "signedinteger[Any]",
    # expected "Optional[int]")
    return idxpos  # type: ignore[return-value]


def validate_limit_direction(
    limit_direction: str,
) -> Literal["forward", "backward", "both"]:
    # [FUNCTION DOC]
    # Validate a limit-direction string (case-insensitive). Returns it lower-cased; raises a value error naming the accepted values (forward, backward, both) otherwise.
    valid_limit_directions = ["forward", "backward", "both"]
    limit_direction = limit_direction.lower()
    if limit_direction not in valid_limit_directions:
        raise ValueError(
            "Invalid limit_direction: expecting one of "
            f"{valid_limit_directions}, got '{limit_direction}'."
        )
    # error: Incompatible return value type (got "str", expected
    # "Literal['forward', 'backward', 'both']")
    return limit_direction  # type: ignore[return-value]


def validate_limit_area(limit_area: str | None) -> Literal["inside", "outside"] | None:
    # [FUNCTION DOC]
    # Validate a limit-area value. None is returned unchanged; a string is lower-cased and must be inside or outside, otherwise a value error is raised. Returns the lower-cased string or none.
    if limit_area is not None:
        valid_limit_areas = ["inside", "outside"]
        limit_area = limit_area.lower()
        if limit_area not in valid_limit_areas:
            raise ValueError(
                f"Invalid limit_area: expecting one of {valid_limit_areas}, got "
                f"{limit_area}."
            )
    # error: Incompatible return value type (got "Optional[str]", expected
    # "Optional[Literal['inside', 'outside']]")
    return limit_area  # type: ignore[return-value]


def infer_limit_direction(
    limit_direction: Literal["backward", "forward", "both"] | None, method: str
) -> Literal["backward", "forward", "both"]:
    # [FUNCTION DOC]
    # Resolve the limit direction for an interpolation method.
    # Parameters: limit_direction (forward, backward, both or none), method (fill or interpolation method name). When none: backward for the backward-fill names, forward otherwise. When given: raises a value error if the forward-fill names are used with a non-forward direction or the backward-fill names with a non-backward direction. Returns the direction.
    # Set `limit_direction` depending on `method`
    if limit_direction is None:
        if method in ("backfill", "bfill"):
            limit_direction = "backward"
        else:
            limit_direction = "forward"
    else:
        if method in ("pad", "ffill") and limit_direction != "forward":
            raise ValueError(
                f"`limit_direction` must be 'forward' for method `{method}`"
            )
        if method in ("backfill", "bfill") and limit_direction != "backward":
            raise ValueError(
                f"`limit_direction` must be 'backward' for method `{method}`"
            )
    return limit_direction


def get_interp_index(method, index: Index) -> Index:
    # [FUNCTION DOC]
    # Compute the index used as the x-coordinates for interpolation.
    # Parameters: method (name), index. For linear, a default integer range of the same length replaces the index. Otherwise a known method is required (value error for unknown names, "Can not interpolate"), and methods other than index, values, nearest and time need a numeric or datetime index (value error otherwise). In all cases a missing value in the resulting index raises a not-implemented error. Returns the index.
    # create/use the index
    if method == "linear":
        # prior default
        from pandas import RangeIndex

        index = RangeIndex(len(index))
    else:
        methods = {"index", "values", "nearest", "time"}
        is_numeric_or_datetime = (
            is_numeric_dtype(index.dtype)
            or isinstance(index.dtype, DatetimeTZDtype)
            or lib.is_np_dtype(index.dtype, "mM")
        )
        valid = NP_METHODS + SP_METHODS
        if method in valid:
            if method not in methods and not is_numeric_or_datetime:
                raise ValueError(
                    "Index column must be numeric or datetime type when "
                    f"using {method} method other than linear. "
                    "Try setting a numeric or datetime index column before "
                    "interpolating."
                )
        else:
            raise ValueError(f"Can not interpolate with method={method}.")

    if isna(index).any():
        raise NotImplementedError(
            "Interpolation with NaNs in the index "
            "has not been implemented. Try filling "
            "those NaNs before interpolating."
        )
    return index


def interpolate_2d_inplace(
    data: np.ndarray,  # floating dtype
    index: Index,
    axis: AxisInt,
    method: str = "linear",
    limit: int | None = None,
    limit_direction: str = "forward",
    limit_area: str | None = None,
    fill_value: Any | None = None,
    mask=None,
    **kwargs,
) -> None:
    # [FUNCTION DOC]
    # Interpolate each one-dimensional slice of a floating two-dimensional array along an axis, modifying the data in place.
    # Parameters: data (floating ndarray, mutated); index (the axis labels used as x-coordinates); axis; method (default linear); limit (maximum consecutive fills, none for unlimited); limit_direction (default forward); limit_area (inside, outside or none); fill_value (value for out-of-bounds in scipy methods; a missing-like value becomes the dtype's missing value); mask (optional boolean array updated in place by the slice routine); extra keywords forwarded to the scipy routines.
    # Validates the method, converts time-weighted interpolation to the values method (raising a value error if the index is not datetime-like), validates direction and area, validates the limit, converts the index to numeric coordinates once, and applies the one-dimensional routine along the axis. Returns nothing. The signature carries only what block-level interpolation needs.
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
    # validate the interp method
    clean_interp_method(method, index, **kwargs)

    if is_valid_na_for_dtype(fill_value, data.dtype):
        fill_value = na_value_for_dtype(data.dtype, compat=False)

    if method == "time":
        if not needs_i8_conversion(index.dtype):
            raise ValueError(
                "time-weighted interpolation only works "
                "on Series or DataFrames with a "
                "DatetimeIndex"
            )
        method = "values"

    limit_direction = validate_limit_direction(limit_direction)
    limit_area_validated = validate_limit_area(limit_area)

    # default limit is unlimited GH #16282
    limit = algos.validate_limit(nobs=None, limit=limit)

    indices = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        # [FUNCTION DOC]
        # HELPER: nested function applied to each one-dimensional slice; calls the one-dimensional interpolation with the settings captured from the enclosing function (bounds error disabled), mutating the slice. Returns nothing.
        # process 1-d slices in the axis direction

        _interpolate_1d(
            indices=indices,
            yvalues=yvalues,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area_validated,
            fill_value=fill_value,
            bounds_error=False,
            mask=mask,
            **kwargs,
        )

    np.apply_along_axis(func, axis, data)


def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: convert an index to a numeric array of x-coordinates for numpy or scipy. Datetime-like data are viewed as 64-bit integers; for the linear method the values are used as is; otherwise converted to a plain array, and for the values and index methods an object array is converted to a numeric array when possible.
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    if needs_i8_conversion(xarr.dtype):
        # GH#1646 for dt64tz
        xarr = xarr.view("i8")

    if method == "linear":
        inds = xarr
        inds = cast(np.ndarray, inds)
    else:
        inds = np.asarray(xarr)

        if method in ("values", "index"):
            if inds.dtype == np.object_:
                inds = lib.maybe_convert_objects(inds)

    return inds


def _interpolate_1d(
    indices: np.ndarray,
    yvalues: np.ndarray,
    method: str = "linear",
    limit: int | None = None,
    limit_direction: str = "forward",
    limit_area: Literal["inside", "outside"] | None = None,
    fill_value: Any | None = None,
    bounds_error: bool = False,
    order: int | None = None,
    mask=None,
    **kwargs,
) -> None:
    # [FUNCTION DOC]
    # HELPER: the one-dimensional interpolation, filling yvalues in place.
    # Parameters: indices (x-coordinates) and yvalues (same length); method; limit; limit_direction; limit_area; fill_value; bounds_error (only passed to the scipy routines); order; mask (optional; if given, it is used instead of detecting missing values and is overwritten at the end); extra keywords.
    # Does nothing if there are no valid values or no invalid values. Determines the leading and trailing missing runs and the set of positions to keep missing according to the limit, direction and area. Numpy methods use linear interpolation on sorted valid points; others call the scipy wrapper. Datetime-like data are viewed as integers. Finally positions to preserve are restored: with a mask, the mask is cleared and set true at those positions; otherwise they are set to NaT or NaN.
    # KNOWN ISSUE: the checks for a none first or last valid index can never trigger, because the function has already returned when no valid values exist.
    # KNOWN ISSUE: the docstring says bounds_error is hardcoded to false, but it is an ordinary parameter (the only caller passes false).
    """
    Logic for the 1-d interpolation.  The input
    indices and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.

    Notes
    -----
    Fills 'yvalues' in-place.
    """
    if mask is not None:
        invalid = mask
    else:
        invalid = isna(yvalues)
    valid = ~invalid

    if not valid.any():
        return

    if valid.all():
        return

    # These index pointers to invalid values... i.e. {0, 1, etc...
    all_nans = np.flatnonzero(invalid)

    first_valid_index = find_valid_index(how="first", is_valid=valid)
    if first_valid_index is None:  # no nan found in start
        first_valid_index = 0
    start_nans = np.arange(first_valid_index)

    last_valid_index = find_valid_index(how="last", is_valid=valid)
    if last_valid_index is None:  # no nan found in end
        last_valid_index = len(yvalues)
    end_nans = np.arange(1 + last_valid_index, len(valid))

    # preserve_nans contains indices of invalid values,
    # but in this case, it is the final set of indices that need to be
    # preserved as NaN after the interpolation.

    # For example if limit_direction='forward' then preserve_nans will
    # contain indices of NaNs at the beginning of the series, and NaNs that
    # are more than 'limit' away from the prior non-NaN.

    # set preserve_nans based on direction using _interp_limit
    if limit_direction == "forward":
        preserve_nans = np.union1d(start_nans, _interp_limit(invalid, limit, 0))
    elif limit_direction == "backward":
        preserve_nans = np.union1d(end_nans, _interp_limit(invalid, 0, limit))
    else:
        # both directions... just use _interp_limit
        preserve_nans = np.unique(_interp_limit(invalid, limit, limit))

    # if limit_area is set, add either mid or outside indices
    # to preserve_nans GH #16284
    if limit_area == "inside":
        # preserve NaNs on the outside
        preserve_nans = np.union1d(preserve_nans, start_nans)
        preserve_nans = np.union1d(preserve_nans, end_nans)
    elif limit_area == "outside":
        # preserve NaNs on the inside
        mid_nans = np.setdiff1d(all_nans, start_nans, assume_unique=True)
        mid_nans = np.setdiff1d(mid_nans, end_nans, assume_unique=True)
        preserve_nans = np.union1d(preserve_nans, mid_nans)

    is_datetimelike = yvalues.dtype.kind in "mM"

    if is_datetimelike:
        yvalues = yvalues.view("i8")

    if method in NP_METHODS:
        # np.interp requires sorted X values, #21037

        indexer = np.argsort(indices[valid])
        yvalues[invalid] = np.interp(
            indices[invalid], indices[valid][indexer], yvalues[valid][indexer]
        )
    else:
        yvalues[invalid] = _interpolate_scipy_wrapper(
            indices[valid],
            yvalues[valid],
            indices[invalid],
            method=method,
            fill_value=fill_value,
            bounds_error=bounds_error,
            order=order,
            **kwargs,
        )

    if mask is not None:
        mask[:] = False
        mask[preserve_nans] = True
    elif is_datetimelike:
        yvalues[preserve_nans] = NaT.value
    else:
        yvalues[preserve_nans] = np.nan
    return


def _interpolate_scipy_wrapper(
    x: np.ndarray,
    y: np.ndarray,
    new_x: np.ndarray,
    method: str,
    fill_value=None,
    bounds_error: bool = False,
    order=None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: dispatch to scipy interpolation.
    # Parameters: x, y (known points), new_x (points to evaluate), method, fill_value, bounds_error, order, extra keywords. Requires the optional scientific library (error message names the method). Methods nearest, zero, slinear, quadratic, cubic and polynomial (using order as degree) go through the standard one-dimensional interpolator; spline uses a univariate spline (value error unless order is a positive number); the remaining names use a table of functions (barycentric, krogh, derivatives, piecewise polynomial, cubic spline, akima, pchip), after making read-only inputs writable copies and dropping a downcast keyword; unknown names raise a value error. Returns the interpolated array.
    # KNOWN ISSUE: the docstring refers to a function named for cleaning interpolation methods with a leading underscore, but the real function has no leading underscore.
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.  Add any new methods to
    the list in _clean_interp_method.
    """
    extra = f"{method} interpolation requires SciPy."
    import_optional_dependency("scipy", extra=extra)
    from scipy import interpolate

    new_x = np.asarray(new_x)

    # ignores some kwargs that could be passed along.
    alt_methods: dict[str, Callable[..., np.ndarray]] = {
        "barycentric": interpolate.barycentric_interpolate,
        "krogh": interpolate.krogh_interpolate,
        "from_derivatives": _from_derivatives,
        "piecewise_polynomial": _from_derivatives,
        "cubicspline": _cubicspline_interpolate,
        "akima": _akima_interpolate,
        "pchip": interpolate.pchip_interpolate,
    }

    interp1d_methods = [
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "polynomial",
    ]
    terp: Callable[..., np.ndarray] | None
    if method in interp1d_methods:
        if method == "polynomial":
            kind = order
        else:
            kind = method
        terp = interpolate.interp1d(
            x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error
        )
        new_y = terp(new_x)
    elif method == "spline":
        # GH #10633, #24014
        if isna(order) or (order <= 0):
            raise ValueError(
                f"order needs to be specified and greater than 0; got order: {order}"
            )
        terp = interpolate.UnivariateSpline(x, y, k=order, **kwargs)
        new_y = terp(new_x)
    else:
        # GH 7295: need to be able to write for some reason
        # in some circumstances: check all three
        if not x.flags.writeable:
            x = x.copy()
        if not y.flags.writeable:
            y = y.copy()
        if not new_x.flags.writeable:
            new_x = new_x.copy()
        terp = alt_methods.get(method, None)
        if terp is None:
            raise ValueError(f"Can not interpolate with method={method}.")

        # Make sure downcast is not in kwargs for alt methods
        kwargs.pop("downcast", None)
        new_y = terp(x, y, new_x, **kwargs)
    return new_y


def _from_derivatives(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    order=None,
    der: int | list[int] | None = 0,
    extrapolate: bool = False,
):
    # [FUNCTION DOC]
    # HELPER: convenience wrapper for the piecewise polynomial constructed from known values and derivatives (Bernstein basis).
    # Parameters: xi (sorted x-coordinates), yi (values, reshaped to a column), x (evaluation points), order (local polynomial degree, default none), der (unused), extrapolate (default False). Returns the polynomial evaluated at x.
    # KNOWN ISSUE: the docstring states extrapolate defaults to True, but the signature defaults it to False; the der argument is accepted but never used.
    """
    Convenience function for interpolate.BPoly.from_derivatives.

    Construct a piecewise polynomial in the Bernstein basis, compatible
    with the specified values and derivatives at breakpoints.

    Parameters
    ----------
    xi : array-like
        sorted 1D array of x-coordinates
    yi : array-like or list of array-likes
        yi[i][j] is the j-th derivative known at xi[i]
    order: None or int or array-like of ints. Default: None.
        Specifies the degree of local polynomials. If not None, some
        derivatives are ignored.
    der : int or list
        How many derivatives to extract; None for all potentially nonzero
        derivatives (that is a number equal to the number of points), or a
        list of derivatives to extract. This number includes the function
        value as 0th derivative.
     extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points based on first and last
        intervals, or to return NaNs. Default: True.

    See Also
    --------
    scipy.interpolate.BPoly.from_derivatives

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R.
    """
    from scipy import interpolate

    # return the method for compat with scipy version & backwards compat
    method = interpolate.BPoly.from_derivatives
    m = method(xi, yi.reshape(-1, 1), orders=order, extrapolate=extrapolate)

    return m(x)


def _akima_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    der: int = 0,
    axis: AxisInt = 0,
):
    # [FUNCTION DOC]
    # HELPER: convenience wrapper for Akima interpolation. Parameters: xi (sorted x-coordinates), yi (values), x (evaluation points), der (derivative order, default 0), axis (default 0). Builds the interpolator and evaluates it at x with the derivative order. Returns the values.
    """
    Convenience function for akima interpolation.
    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``.

    See `Akima1DInterpolator` for details.

    Parameters
    ----------
    xi : np.ndarray
        A sorted list of x-coordinates, of length N.
    yi : np.ndarray
        A 1-D array of real values.  `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.
    x : np.ndarray
        Of length M.
    der : int, optional
        How many derivatives to extract. This number includes the function
        value as 0th derivative.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    See Also
    --------
    scipy.interpolate.Akima1DInterpolator

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R,

    """
    from scipy import interpolate

    P = interpolate.Akima1DInterpolator(xi, yi, axis=axis)

    return P(x, nu=der)


def _cubicspline_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    axis: AxisInt = 0,
    bc_type: _CubicBC | tuple[Any, Any] = "not-a-knot",
    extrapolate: Literal["periodic"] | bool | None = None,
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: convenience wrapper for cubic spline interpolation. Parameters: xi (strictly increasing x-coordinates), yi (values), x (evaluation points), axis (default 0), bc_type (boundary condition: not-a-knot default, clamped, natural, periodic, or a pair), extrapolate (boolean, periodic or none for the library default). Builds the spline and returns it evaluated at x.
    """
    Convenience function for cubic spline data interpolator.

    See `scipy.interpolate.CubicSpline` for details.

    Parameters
    ----------
    xi : np.ndarray, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    yi : np.ndarray
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    x : np.ndarray, shape (m,)
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.
        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:
        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.
        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:
        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array-like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1D, then `deriv_value` must be a scalar. If `y` is 3D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    See Also
    --------
    scipy.interpolate.CubicHermiteSpline

    Returns
    -------
    y : scalar or array-like
        The result, of shape (m,)

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    from scipy import interpolate

    P = interpolate.CubicSpline(
        xi, yi, axis=axis, bc_type=bc_type, extrapolate=extrapolate
    )

    return P(x)


def pad_or_backfill_inplace(
    values: np.ndarray,
    method: Literal["pad", "backfill"] = "pad",
    axis: AxisInt = 0,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
) -> None:
    # [FUNCTION DOC]
    # Forward or backward fill a numpy array in place.
    # Parameters: values (ndarray, one or two dimensional, mutated); method (pad default, or backfill; aliases accepted); axis (0 or 1; must be 0 for one-dimensional data, else an assertion error is raised); limit (maximum consecutive fills); limit_area (inside, outside or none). One-dimensional input is treated as a single row; axis 1 works on a transposed view. Uses the two-dimensional fill kernel. Returns nothing.
    # KNOWN ISSUE: the docstring says the result is returned, but nothing is returned; the function only modifies the array in place.
    # The two small anonymous transposition helpers inside this function are lambdas and carry no separate documentation: one returns its argument and the other returns the transpose.
    """
    Perform an actual interpolation of values, values will be make 2-d if
    needed fills inplace, returns the result.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str, default "pad"
        Interpolation method. Could be "bfill" or "pad"
    axis: 0 or 1
        Interpolation axis
    limit: int, optional
        Index limit on interpolation.
    limit_area: str, optional
        Limit area for interpolation. Can be "inside" or "outside"

    Notes
    -----
    Modifies values in-place.
    """
    transf = (lambda x: x) if axis == 0 else (lambda x: x.T)

    # reshape a 1 dim if needed
    if values.ndim == 1:
        if axis != 0:  # pragma: no cover
            raise AssertionError("cannot interpolate on an ndim == 1 with axis != 0")
        values = values.reshape((1, *values.shape))

    method = clean_fill_method(method)
    tvalues = transf(values)

    func = get_fill_func(method, ndim=2)
    # _pad_2d and _backfill_2d both modify tvalues inplace
    func(tvalues, limit=limit, limit_area=limit_area)


def _fillna_prep(
    values, mask: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.bool_]:
    # [FUNCTION DOC]
    # HELPER: shared preamble of the fill kernels. Returns the given mask, or computes the missing-value mask of values if none was given.
    # boilerplate for _pad_1d, _backfill_1d, _pad_2d, _backfill_2d

    if mask is None:
        mask = isna(values)

    return mask


def _datetimelike_compat(func: F) -> F:
    # [FUNCTION DOC]
    # HELPER: decorator for the fill kernels that handles datetime64 and timedelta64 data. For such data it computes the mask first, runs the wrapped kernel on a 64-bit integer view, and returns the result viewed in the original dtype along with the mask; other data pass straight through. Returns the wrapper.
    """
    Wrapper to handle datetime64 and timedelta64 dtypes.
    """

    @wraps(func)
    def new_func(
        values,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        mask=None,
    ):
        # [FUNCTION DOC]
        # HELPER: wrapper created by the datetime-like decorator. Parameters: values, limit (default none), limit_area (default none), mask (default none). Returns the tuple of filled values and mask.
        if needs_i8_conversion(values.dtype):
            if mask is None:
                # This needs to occur before casting to int64
                mask = isna(values)

            result, mask = func(
                values.view("i8"), limit=limit, limit_area=limit_area, mask=mask
            )
            return result.view(values.dtype), mask

        return func(values, limit=limit, limit_area=limit_area, mask=mask)

    return cast(F, new_func)


@_datetimelike_compat
def _pad_1d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # HELPER: forward fill a one-dimensional array in place (decorated for datetime-like data). Parameters: values, limit, limit_area, mask. Computes the mask if absent; if limit_area is given and not everything is missing, modifies the mask so only the allowed area is filled; runs the compiled forward fill. Returns the tuple of values and mask (the mask may no longer reflect missingness).
    mask = _fillna_prep(values, mask)
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    algos.pad_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _backfill_1d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # HELPER: backward fill a one-dimensional array in place; same parameters, preparation and return as the forward fill version, using the compiled backward fill.
    mask = _fillna_prep(values, mask)
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    algos.backfill_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _pad_2d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # HELPER: forward fill a two-dimensional array in place (rows are series). Same parameters and return as the one-dimensional version; applies the two-dimensional area restriction when limit_area is given and only runs the kernel when the array is nonempty.
    mask = _fillna_prep(values, mask)
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)

    if values.size:
        algos.pad_2d_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _backfill_2d(
    values,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
):
    # [FUNCTION DOC]
    # HELPER: backward fill a two-dimensional array in place; same behavior as the two-dimensional forward fill with the backward kernel (nothing is done for empty arrays). Returns the tuple of values and mask.
    mask = _fillna_prep(values, mask)
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)

    if values.size:
        algos.backfill_2d_inplace(values, mask, limit=limit)
    else:
        # for test coverage
        pass
    return values, mask


def _fill_limit_area_1d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    # [FUNCTION DOC]
    # HELPER: adjust a one-dimensional missing-value mask so filling honors limit_area, modifying the mask in place.
    # Parameters: mask (true where missing), limit_area (inside: only fill between the first and last valid values; outside: only fill outside them). The caller must ensure at least one entry is not missing. Afterwards the mask no longer represents missingness.
    """Prepare 1d mask for ffill/bfill with limit_area.

    Caller is responsible for checking at least one value of mask is False.
    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    neg_mask = ~mask
    first = neg_mask.argmax()
    last = len(neg_mask) - neg_mask[::-1].argmax() - 1
    if limit_area == "inside":
        mask[:first] = False
        mask[last + 1 :] = False
    elif limit_area == "outside":
        mask[first + 1 : last] = False


def _fill_limit_area_2d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    # [FUNCTION DOC]
    # HELPER: two-dimensional version of the limit-area mask adjustment, working per column of the transposed mask using running maxima from both ends to find the region between the first and last valid values; modifies the mask in place. For outside, the inside region is cleared; for inside, the outside region is cleared.
    # KNOWN ISSUE: the docstring declares the mask as one-dimensional, but this routine operates on a two-dimensional mask.
    """Prepare 2d mask for ffill/bfill with limit_area.

    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    neg_mask = ~mask.T
    if limit_area == "outside":
        # Identify inside
        la_mask = (
            np.maximum.accumulate(neg_mask, axis=0)
            & np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    else:
        # Identify outside
        la_mask = (
            ~np.maximum.accumulate(neg_mask, axis=0)
            | ~np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    mask[la_mask.T] = False


_fill_methods = {"pad": _pad_1d, "backfill": _backfill_1d}


def get_fill_func(method, ndim: int = 1):
    # [FUNCTION DOC]
    # Select the fill kernel. Parameters: method (normalized with the fill method cleaner; value error if invalid), ndim (default 1). Returns the one-dimensional pad or backfill kernel for one dimension, otherwise the two-dimensional kernel.
    method = clean_fill_method(method)
    if ndim == 1:
        return _fill_methods[method]
    return {"pad": _pad_2d, "backfill": _backfill_2d}[method]


def clean_reindex_fill_method(method) -> ReindexMethod | None:
    # [FUNCTION DOC]
    # Normalize a method for reindexing: none passes through; otherwise the fill method cleaner is applied with nearest allowed. Returns the normalized name or none.
    if method is None:
        return None
    return clean_fill_method(method, allow_nearest=True)


def _interp_limit(
    invalid: npt.NDArray[np.bool_], fw_limit: int | None, bw_limit: int | None
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: positions that must stay missing because they lie beyond the fill limits.
    # Parameters: invalid (boolean array, true where missing); fw_limit and bw_limit (forward and backward limits, none meaning no limit in that direction, zero meaning no fill in that direction). Computes, for the forward direction, positions not within the limit of a preceding valid value (and leading positions before any valid value), similarly backward on the reversed array, and combines them by intersection. A zero backward limit returns the forward result; a zero forward limit with a backward limit returns the backward result.
    # Returns an integer array of positions (sorted for the combined case).
    # KNOWN ISSUE: the docstring says it returns a set of indexers, but it returns an integer array.
    """
    Get indexers of values that won't be filled
    because they exceed the limits.

    Parameters
    ----------
    invalid : np.ndarray[bool]
    fw_limit : int or None
        forward limit to index
    bw_limit : int or None
        backward limit to index

    Returns
    -------
    set of indexers

    Notes
    -----
    This is equivalent to the more readable, but slower

    .. code-block:: python

        def _interp_limit(invalid, fw_limit, bw_limit):
            for x in np.where(invalid)[0]:
                if invalid[max(0, x - fw_limit) : x + bw_limit + 1].all():
                    yield x
    """
    # handle forward first; the backward direction is the same except
    # 1. operate on the reversed array
    # 2. subtract the returned indices from N - 1
    N = len(invalid)
    f_idx = np.array([], dtype=np.int64)
    b_idx = np.array([], dtype=np.int64)
    assume_unique = True

    def inner(invalid, limit: int):
        # [FUNCTION DOC]
        # HELPER: nested in the limit computation. Parameters: invalid (boolean array, possibly reversed), limit (int, capped to length minus one). Finds runs of missing values longer than the limit using a sliding window and returns the sorted positions that exceed it, together with the leading positions before the first valid value. Returns an integer array.
        limit = min(limit, N - 1)
        windowed = np.lib.stride_tricks.sliding_window_view(invalid, limit + 1).all(1)
        idx = np.union1d(
            np.where(windowed)[0] + limit,
            np.where((~invalid[: limit + 1]).cumsum() == 0)[0],
        )
        return idx

    if fw_limit is not None:
        if fw_limit == 0:
            f_idx = np.where(invalid)[0]
            assume_unique = False
        else:
            f_idx = inner(invalid, fw_limit)

    if bw_limit is not None:
        if bw_limit == 0:
            # then we don't even need to care about backwards
            # just use forwards
            return f_idx
        else:
            b_idx = N - 1 - inner(invalid[::-1], bw_limit)
            if fw_limit == 0:
                return b_idx

    return np.intersect1d(f_idx, b_idx, assume_unique=assume_unique)
