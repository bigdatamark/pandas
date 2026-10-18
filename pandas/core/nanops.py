# [MODULE DOC]
# NaN-aware reduction kernels operating on plain numpy arrays: any, all, sum, mean, median, std, var, sem, min, max, argmin, argmax, skew, kurtosis, product, correlation, covariance, and a cumulative-function helper. These are the numpy-level engines behind Series, DataFrame and array reductions.
# Conventions shared by the kernels: values is an ndarray; axis is an integer or none (reduce everything); skipna says whether missing values are ignored; mask is an optional precomputed boolean array marking missing entries (required for datetime-like and masked inputs to be interpreted correctly); keyword-only arguments throughout. Results are scalars or arrays; datetime-like inputs are converted to integer views, processed, and converted back.
# Machinery: a decorator that rejects given dtypes, a decorator that dispatches to the optional bottleneck library when allowed, a datetime-like compatibility decorator, and a decorator that processes very wide two-dimensional arrays row by row. The helper functions compute masks, fill values, counts, null-out handling for minimum-count rules, and numeric coercion.
# Import-time side effects: tries to import the optional bottleneck library (warning if it is present but unusable), records whether it is installed, and sets the module-level switch for using it from the compute.use_bottleneck configuration option. Built-in reductions nanmin and nanmax are created at import time by a factory.

from __future__ import annotations

import functools
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    NaTType,
    iNaT,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    CorrelationMethod,
    Dtype,
    DtypeObj,
    F,
    Scalar,
    Shape,
    npt,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.common import (
    is_complex,
    is_float,
    is_float_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

if TYPE_CHECKING:
    from collections.abc import Callable

bn = import_optional_dependency("bottleneck", errors="warn")
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    # [FUNCTION DOC]
    # Turn the use of the optional bottleneck library on or off by setting a module-level flag.
    # Parameter: v (default True) is the new setting. The flag is only changed if bottleneck is installed; otherwise nothing happens. Called once at import time with the configuration option value and again by the configuration callback when the option changes. Returns nothing; mutates global state.
    # set/unset to use bottleneck
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    # [CLASS DOC]
    # Decorator class that forbids the wrapped reduction for certain dtypes.
    # State: dtypes, a tuple of numpy scalar types derived from the dtype specifications given at construction. Calling an instance on a function returns a wrapper function.
    def __init__(self, *dtypes: Dtype) -> None:
        # [FUNCTION DOC]
        # Takes any number of dtype specifications, converts each to a pandas dtype and stores its scalar type in the attribute dtypes.
        super().__init__()
        self.dtypes = tuple(pandas_dtype(dtype).type for dtype in dtypes)

    def check(self, obj) -> bool:
        # [FUNCTION DOC]
        # Returns True if obj has a dtype attribute whose scalar type is a subclass of any of the disallowed types, otherwise False.
        return hasattr(obj, "dtype") and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        # [FUNCTION DOC]
        # Wrap the function f (the decorated reduction) and return the wrapper, preserving its metadata.
        @functools.wraps(f)
        def _f(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: the wrapper created by the decorator. Takes any positional and keyword arguments. If any argument (positional or keyword value) has a disallowed dtype, raises a type error saying the reduction (name of f with the nan prefix removed) is not allowed for this dtype. Otherwise calls f; if f raises a value error and the first positional argument has object dtype, the value error is re-raised as a type error; other value errors propagate. Returns f's result.
            obj_iter = itertools.chain(args, kwargs.values())
            if any(self.check(obj) for obj in obj_iter):
                f_name = f.__name__.replace("nan", "")
                raise TypeError(
                    f"reduction operation '{f_name}' not allowed for this dtype"
                )
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                # we want to transform an object array
                # ValueError message to the more typical TypeError
                # e.g. this is normally a disallowed function on
                # object arrays that contain strings
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise

        return cast(F, _f)


class bottleneck_switch:
    # [CLASS DOC]
    # Decorator class that routes a reduction to the bottleneck library when conditions allow, else to the wrapped pure-numpy implementation.
    # State: name (bottleneck function name, default is the wrapped function name) and kwargs (default keyword arguments, such as a degrees-of-freedom value, merged into each call unless the caller gave them).
    def __init__(self, name=None, **kwargs) -> None:
        # [FUNCTION DOC]
        # Store the optional bottleneck function name and any default keyword arguments.
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        # [FUNCTION DOC]
        # Wrap the alternative implementation alt and return the wrapper. Looks up the bottleneck function by name (absent if bottleneck is missing or lacks it, in which case it is None and never used because the flag is off or the dtype check fails).
        bn_name = self.name or alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(
            values: np.ndarray,
            *,
            axis: AxisInt | None = None,
            skipna: bool = True,
            **kwds,
        ):
            # [FUNCTION DOC]
            # HELPER: the wrapper produced by the decorator.
            # Parameters: values (ndarray), axis (default none), skipna (default True), extra keyword arguments passed to the underlying function. Defaults stored on the decorator are added for keywords not supplied.
            # If the array is empty and no minimum count was specified, returns the missing value for the dtype (shaped per the axis). If bottleneck usage is enabled, skipna is true and the dtype is suitable, and no mask was supplied (any mask keyword is dropped since bottleneck does not understand it), calls bottleneck; if the result contains infinities, recomputes with the alternative so infinities are treated as missing. In all other cases calls the alternative. Returns the result.
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v

            if values.size == 0 and kwds.get("min_count") is None:
                # We are empty, returning NA for our type
                # Only applies for the default `min_count` of None
                # since that affects how empty arrays are handled.
                # TODO(GH-18976) update all the nanops methods to
                # correctly handle empty inputs and remove this check.
                # It *may* just be `var`
                return _na_for_min_count(values, axis)

            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get("mask", None) is None:
                    # `mask` is not recognised by bottleneck, would raise
                    #  TypeError if called
                    kwds.pop("mask", None)
                    result = bn_func(values, axis=axis, **kwds)

                    # prefer to treat inf/-inf as NA, but must compute the func
                    # twice :(
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)

            return result

        return cast(F, f)


def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: decides whether bottleneck may be used for a dtype and function name. Returns False for object dtype and for datetime-like (needing integer conversion) dtypes. Otherwise returns True unless the name is the sum, product or mean reduction, which bottleneck handles with insufficient precision, overflow risk, or wrong all-missing results.
    # Bottleneck chokes on datetime64, PeriodDtype (or and EA)
    if dtype != object and not needs_i8_conversion(dtype):
        # GH 42878
        # Bottleneck uses naive summation leading to O(n) loss of precision
        # unlike numpy which implements pairwise summation, which has O(log(n)) loss
        # crossref: https://github.com/pydata/bottleneck/issues/379

        # GH 15507
        # bottleneck does not properly upcast during the sum
        # so can overflow

        # GH 9422
        # further we also want to preserve NaN when all elements
        # are NaN, unlike bottleneck/numpy which consider this
        # to be 0
        return name not in ["nansum", "nanprod", "nanmean"]
    return False


def _has_infs(result) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns whether a result contains positive or negative infinity. Float32 and float64 arrays use a fast compiled check on a flattened view; anything else uses the generic infinity test, and types that cannot hold infinities (type or not-implemented errors) return False.
    if isinstance(result, np.ndarray):
        if result.dtype in ("f8", "f4"):
            # Note: outside of a nanops-specific test, we always have
            #  result.ndim == 1, so there is no risk of this ravel making a copy.
            return lib.has_infs(result.ravel("K"))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        # if it doesn't support infs, then it can't have infs
        return False


def _get_fill_value(
    dtype: DtypeObj, fill_value: Scalar | None = None, fill_value_typ=None
):
    # [FUNCTION DOC]
    # HELPER: choose the value used to replace missing entries before a reduction.
    # Parameters: dtype of the data; fill_value, returned unchanged if not None; fill_value_typ, one of none, a positive-infinity marker or other (treated as negative infinity).
    # For dtypes that can hold NaN (not integer-like or datetime-like): NaN, positive infinity or negative infinity according to the marker. For others: the largest 64-bit integer if the marker is positive infinity, else the integer sentinel for NaT (as 64-bit integers so that array selection promotes dtype instead of overflowing).
    """return the correct fill value for the dtype of the values"""
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        elif fill_value_typ == "+inf":
            return np.inf
        else:
            return -np.inf
    elif fill_value_typ == "+inf":
        # need the max int here
        # Return as np.int64 so that np.where promotes the dtype
        # instead of raising OverflowError (numpy 2.5+) when the
        # value doesn't fit in the array's dtype (e.g. int8).
        return np.int64(lib.i8max)
    else:
        return np.int64(iNaT)


def _maybe_get_mask(
    values: np.ndarray, skipna: bool, mask: npt.NDArray[np.bool_] | None
) -> npt.NDArray[np.bool_] | None:
    # [FUNCTION DOC]
    # HELPER: compute a missing-value mask only when needed.
    # Parameters: values (ndarray), skipna (bool), mask (optional precomputed mask).
    # If a mask is given it is returned. Otherwise returns none for boolean/integer dtypes (cannot hold missing values); computes a missing-value mask when skipna is true or the dtype is datetime-like; otherwise returns none.
    # KNOWN ISSUE: the docstring mentions an invert option selecting the not-missing test, but the function has no such parameter.
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    """
    if mask is None:
        if values.dtype.kind in "biu":
            # Boolean data cannot contain nulls, so signal via mask being None
            return None

        if skipna or values.dtype.kind in "mM":
            mask = isna(values)

    return mask


def _get_values(
    values: np.ndarray,
    skipna: bool,
    fill_value: Any = None,
    fill_value_typ: str | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_] | None]:
    # [FUNCTION DOC]
    # HELPER: prepare values and mask for a reduction.
    # Parameters: values (ndarray), skipna, fill_value (scalar used to replace missing entries), fill_value_typ (infinity marker used to derive the fill value), mask (optional precomputed).
    # Computes the mask via the mask helper; datetime-like values are viewed as 64-bit integers after the mask is found. If skipna is true and a mask exists, determines the fill value and, if any entries are masked, replaces them: in place on a copy for datetime-like or NaN-capable dtypes, otherwise through a conditional select that promotes dtype. Returns a tuple of (values, mask). Does not mutate the input; copies only when replacement is needed.
    # KNOWN ISSUE: the docstring summary says it returns dtype, maximum dtype and fill value too, but only values and mask are returned.
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray[bool]]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    """
    # In _get_values is only called from within nanops, and in all cases
    #  with scalar fill_value.  This guarantee is important for the
    #  np.where call below

    mask = _maybe_get_mask(values, skipna, mask)

    dtype = values.dtype

    datetimelike = False
    if values.dtype.kind in "mM":
        # changing timedelta64/datetime64 to int64 needs to happen after
        #  finding `mask` above
        values = np.asarray(values.view("i8"))
        datetimelike = True

    if skipna and (mask is not None):
        # get our fill value (in case we need to provide an alternative
        # dtype for it)
        fill_value = _get_fill_value(
            dtype, fill_value=fill_value, fill_value_typ=fill_value_typ
        )

        if fill_value is not None:
            if mask.any():
                if datetimelike or _na_ok_dtype(dtype):
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    # np.where will promote if needed
                    values = np.where(~mask, values, fill_value)

    return values, mask


def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    # [FUNCTION DOC]
    # HELPER: platform-independent accumulation dtype. Boolean and signed integer dtypes map to 64-bit signed; unsigned to 64-bit unsigned; floating to 64-bit float; others are returned unchanged.
    # return a platform independent precision dtype
    dtype_max = dtype
    if dtype.kind in "bi":
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == "u":
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == "f":
        dtype_max = np.dtype(np.float64)
    return dtype_max


def _na_ok_dtype(dtype: DtypeObj) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if the dtype can store NaN: False for datetime-like dtypes needing integer conversion and for integer scalar types, True otherwise.
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)


def _wrap_results(result, dtype: np.dtype, fill_value=None):
    # [FUNCTION DOC]
    # HELPER: convert a reduction result back to the original datetime-like dtype.
    # Parameters: result (scalar or array), dtype (original dtype), fill_value (sentinel indicating missing, default none meaning the integer NaT sentinel for datetimes).
    # NaT passes through. Datetime64 scalars: sentinel or NaN becomes NaT, else reinterpreted as integer nanosecond-count then cast to the dtype retaining unit (asserts the fill value is not missing); arrays are cast to the dtype. Timedelta64 scalars: sentinel or NaN becomes NaT in the dtype's unit; magnitude above the 64-bit integer maximum raises a value error about overflow; else cast to the dtype; arrays are cast through nanosecond timedeltas then viewed as the dtype. Other dtypes return the result unchanged.
    """wrap our results if needed"""
    if result is NaT:
        pass

    elif dtype.kind == "M":
        if fill_value is None:
            # GH#24293
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), "Expected non-null fill_value"
            if result == fill_value:
                result = np.nan

            if isna(result):
                result = np.datetime64("NaT", "ns").astype(dtype)
            else:
                result = np.int64(result).view(dtype)
            # retain original unit
            result = result.astype(dtype, copy=False)
        else:
            # If we have float dtype, taking a view will give the wrong result
            result = result.astype(dtype)
    elif dtype.kind == "m":
        if not isinstance(result, np.ndarray):
            if result == fill_value or np.isnan(result):
                unit = np.datetime_data(dtype)[0]
                result = np.timedelta64("NaT", unit)  # type: ignore[call-overload]

            elif np.fabs(result) > lib.i8max:
                # raise if we have a timedelta64[ns] which is too large
                raise ValueError("overflow in timedelta operation")
            else:
                # return a timedelta64 with the original unit
                result = np.int64(result).astype(dtype, copy=False)

        else:
            result = result.astype("m8[ns]").view(dtype)

    return result


def _datetimelike_compat(func: F) -> F:
    # [FUNCTION DOC]
    # HELPER: decorator that makes a reduction work for datetime64 and timedelta64 inputs.
    # For datetime-like input, a mask is computed if absent, the wrapped function is called with the mask, the result is converted back to the original dtype, and, if skipna is false, positions where any masked values occurred are set to NaT. Other dtypes pass straight through. Returns the wrapper.
    """
    If we have datetime64 or timedelta64 values, ensure we have a correct
    mask before calling the wrapped function, then cast back afterwards.
    """

    @functools.wraps(func)
    def new_func(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: the wrapper built by the compat decorator. Parameters: values, axis (default none), skipna (default True), mask (default none), extra keywords forwarded to the wrapped function. Returns the converted result.
        orig_values = values

        datetimelike = values.dtype.kind in "mM"
        if datetimelike and mask is None:
            mask = isna(values)

        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if not skipna:
                assert mask is not None  # checked above
                result = _mask_datetimelike_result(result, axis, mask, orig_values)

        return result

    return cast(F, new_func)


def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
    # [FUNCTION DOC]
    # HELPER: the missing value that a reduction of values should return.
    # Parameters: values (ndarray); axis (int or none; needed to compute the output shape for multi-dimensional input, though none is accepted and gives a scalar).
    # Numeric and boolean data are treated as float64. For one-dimensional data or axis none, returns the scalar missing value for the dtype (NaN or NaT). Otherwise returns an array of that missing value with the shape of the input without the reduced axis.
    # KNOWN ISSUE: the docstring says axis is required for multi-dimensional input, but the code accepts none and returns a scalar.
    """
    Return the missing value for `values`.

    Parameters
    ----------
    values : ndarray
    axis : int or None
        axis for the reduction, required if values.ndim > 1.

    Returns
    -------
    result : scalar or ndarray
        For 1-D values, returns a scalar of the correct missing type.
        For 2-D values, returns a 1-D array where each element is missing.
    """
    # we either return np.nan or pd.NaT
    if values.dtype.kind in "iufcb":
        values = values.astype("float64")
    fill_value = na_value_for_dtype(values.dtype)

    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1 :]

        return np.full(result_shape, fill_value, dtype=values.dtype)


def maybe_operate_rowwise(func: F) -> F:
    # [FUNCTION DOC]
    # Decorator that processes very wide C-contiguous two-dimensional arrays row by row when reducing along axis 1 (numpy is slow in that case).
    # The row-wise path is used only when axis is 1, the array is two-dimensional C-contiguous, the number of columns divided by one thousand exceeds the number of rows, and the dtype is neither object nor boolean. It calls the function on each row (slicing the mask per row if given) and returns an array of results. Otherwise it calls the function unchanged. Returns the wrapper.
    """
    NumPy operations on C-contiguous ndarrays with axis=1 can be
    very slow if axis 1 >> axis 0.
    Operate row-by-row and concatenate the results.
    """

    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: AxisInt | None = None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the wrapper created by the row-wise decorator. Parameters: values, axis (default none), extra keywords (including an optional mask). Returns the array of per-row results or the plain function result.
        if (
            axis == 1
            and values.ndim == 2
            and values.flags["C_CONTIGUOUS"]
            # only takes this path for wide arrays (long dataframes), for threshold see
            # https://github.com/pandas-dev/pandas/pull/43311#issuecomment-974891737
            and (values.shape[1] / 1000) > values.shape[0]
            and values.dtype not in (object, bool)
        ):
            arrs = list(values)
            if kwargs.get("mask") is not None:
                mask = kwargs.pop("mask")
                results = [
                    func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))
                ]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)

        return func(values, axis=axis, **kwargs)

    return cast(F, newfunc)


def nanany(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Whether any element along the axis is true, ignoring missing values if skipna.
    # Parameters: values (ndarray), axis (default none), skipna (default True), mask (optional).
    # Integer and boolean data without a mask use the plain numpy any. Datetime64 raises a type error. Otherwise missing entries are replaced with false, object data are cast to bool, and numpy any is applied. Returns a boolean scalar or array.
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    np.True_

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    np.False_
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.any(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        raise TypeError("datetime64 type does not support operation 'any'")

    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)

    # For object type, any won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.any(axis)  # type: ignore[return-value]


def nanall(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Whether all elements along the axis are true, ignoring missing values if skipna.
    # Same parameters and structure as the any kernel, but missing entries are replaced with true, and datetime64 raises a type error. Returns a boolean scalar or array.
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s.values)
    np.True_

    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s.values)
    np.False_
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.all(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        raise TypeError("datetime64 type does not support operation 'all'")

    values, _ = _get_values(values, skipna, fill_value=True, mask=mask)

    # For object type, all won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.all(axis)  # type: ignore[return-value]


@disallow("M8")
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.floating] | float | NaTType:
    # [FUNCTION DOC]
    # Sum along an axis, skipping missing values if skipna.
    # Parameters: values (ndarray), axis, skipna, min_count (default 0; if fewer non-missing values than this, the result is missing), mask. Decorated to reject datetime64, to handle timedelta, and to go row-wise on wide arrays.
    # Missing entries are replaced with zero; the accumulator dtype is the platform 64-bit type for the kind (the same float dtype for floats, float64 for timedeltas). Applies the minimum-count rule to the sum. Returns a scalar or array.
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s.values)
    np.float64(3.0)
    """
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    if dtype.kind == "f":
        dtype_sum = dtype
    elif dtype.kind == "m":
        dtype_sum = np.dtype(np.float64)

    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)

    return the_sum


def _mask_datetimelike_result(
    result: np.ndarray | np.datetime64 | np.timedelta64,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_],
    orig_values: np.ndarray,
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    # [FUNCTION DOC]
    # HELPER: after a reduction on datetime-like data with skipna false, sets the result to NaT where any input along the reduced axis was missing.
    # Parameters: result (array or scalar), axis, mask, orig_values (original array supplying the dtype). For array results: converts to integer, views as the original dtype and sets NaT in positions where the mask has any true along the axis (mutates that new array). For scalar results: returns NaT of the original dtype if any mask entry is true. Otherwise returns the result unchanged.
    if isinstance(result, np.ndarray):
        # we need to apply the mask
        result = result.astype("i8").view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT
    elif mask.any():
        return np.int64(iNaT).view(orig_values.dtype)
    return result


@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    # [FUNCTION DOC]
    # Mean along an axis, skipping missing values if skipna. Decorated for bottleneck dispatch and datetime-like conversion.
    # Parameters: values, axis, skipna, mask. For large object arrays without a mask, first runs the reduction on a leading slice of one thousand elements and discards the result, purely to fail fast on non-numeric data.
    # Fills missing with zero, sums with a float64 accumulator (the input dtype for floats; counts use the same dtype), divides by the count of non-missing entries; zero counts give NaN. Returns a float scalar or array (float input precision preserved).
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s.values)
    np.float64(1.5)
    """
    if values.dtype == object and len(values) > 1_000 and mask is None:
        # GH#54754 if we are going to fail, try to fail-fast
        nanmean(values[:1000], axis=axis, skipna=skipna)

    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)

    # not using needs_i8_conversion because that includes period
    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype

    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _ensure_numeric(the_sum)

    if axis is not None and getattr(the_sum, "ndim", False):
        count = cast(np.ndarray, count)
        with np.errstate(all="ignore"):
            # suppress division by zero warnings
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan

    return the_mean


@bottleneck_switch()
def nanmedian(
    values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, mask=None
) -> float | np.ndarray:
    # [FUNCTION DOC]
    # Median along an axis (ignoring missing values if skipna). Decorated for bottleneck dispatch.
    # Parameters: values, axis, skipna, mask.
    # Raises a type error for object arrays inferred as strings or mixed, and for values that cannot be converted to float. Non-float data are converted to float64 and masked entries set to NaN (copying if read-only). Multi-dimensional input with an axis: empty arrays give an all-NaN array of the reduced shape; skipna false applies the scalar median along the axis; skipna true uses numpy nanmedian, with a faster path for single-row/column inputs. Otherwise a scalar median (NaN if empty). The result is converted back to the original dtype for datetime-like input. Suppresses numpy all-NaN and empty-slice warnings.
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float | ndarray
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s.values)
    2.0

    >>> s = pd.Series([np.nan, np.nan, np.nan])
    >>> nanops.nanmedian(s.values)
    nan
    """
    # for floats without mask, the data already uses NaN as missing value
    # indicator, and `mask` will be calculated from that below -> in those
    # cases we never need to set NaN to the masked values
    using_nan_sentinel = values.dtype.kind == "f" and mask is None

    def get_median(x: np.ndarray, _mask=None):
        # [FUNCTION DOC]
        # HELPER: nested scalar median for a one-dimensional array. Parameters: x, and an optional mask of missing entries (otherwise computed from the array). If skipna is false and anything is missing, returns NaN; otherwise returns the numpy nan-median of the non-missing entries with warnings suppressed. Uses skipna from the enclosing function.
        if _mask is None:
            _mask = notna(x)
        else:
            _mask = ~_mask
        if not skipna and not _mask.all():
            return np.nan
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about All-NaN slice
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            res = np.nanmedian(x[_mask])
        return res

    dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask, fill_value=None)
    if values.dtype.kind != "f":
        if values.dtype == object:
            # GH#34671 avoid casting strings to numeric
            inferred = lib.infer_dtype(values)
            if inferred in ["string", "mixed"]:
                raise TypeError(f"Cannot convert {values} to numeric")
        try:
            values = values.astype("f8")
        except ValueError as err:
            # e.g. "could not convert string to float: 'a'"
            raise TypeError(str(err)) from err
    if not using_nan_sentinel and mask is not None:
        if not values.flags.writeable:
            values = values.copy()
        values[mask] = np.nan

    notempty = values.size

    res: float | np.ndarray

    # an array from a frame
    if values.ndim > 1 and axis is not None:
        # there's a non-empty array to apply over otherwise numpy raises
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)

            else:
                # fastpath for the skipna case
                with warnings.catch_warnings():
                    # Suppress RuntimeWarning about All-NaN slice
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered", RuntimeWarning
                    )
                    if (values.shape[1] == 1 and axis == 0) or (
                        values.shape[0] == 1 and axis == 1
                    ):
                        # GH52788: fastpath when squeezable, nanmedian for 2D array slow
                        res = np.nanmedian(np.squeeze(values), keepdims=True)
                    else:
                        res = np.nanmedian(values, axis=axis)

        else:
            # must return the correct shape, but median is not defined for the
            # empty set so return nans of shape "everything but the passed axis"
            # since "axis" is where the reduction would occur if we had a nonempty
            # array
            res = _get_empty_reduction_result(values.shape, axis)

    else:
        # otherwise return a scalar value
        res = get_median(values, mask) if notempty else np.nan
    return _wrap_results(res, dtype)


def _get_empty_reduction_result(
    shape: Shape,
    axis: AxisInt,
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: the result of a reduction over an empty array. Parameters: shape (tuple of ints) and axis. Returns a float64 array of NaN with the shape of the input with the given axis removed.
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int, ...]
    axis : int

    Returns
    -------
    np.ndarray
    """
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[dims != axis], dtype=np.float64)
    ret.fill(np.nan)
    return ret


def _get_counts_nanvar(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    ddof: int,
    dtype: np.dtype = np.dtype(np.float64),
) -> tuple[float | np.ndarray, float | np.ndarray]:
    # [FUNCTION DOC]
    # HELPER: count of non-missing values along an axis and the divisor (count minus the degrees of freedom) for variance.
    # Parameters: values_shape, mask (optional), axis, ddof, dtype (count dtype, default float64). Where the count is not above ddof, both count and divisor are NaN (never infinity). Returns a tuple of (count, divisor), as scalars or arrays; modifies freshly computed arrays in place.
    """
    Get the count of non-null values along an axis, accounting
    for degrees of freedom.

    Parameters
    ----------
    values_shape : Tuple[int, ...]
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    ddof : int
        degrees of freedom
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : int, np.nan or np.ndarray
    d : int, np.nan or np.ndarray
    """
    count = _get_counts(values_shape, mask, axis, dtype=dtype)
    d = count - dtype.type(ddof)

    # always return NaN, never inf
    if is_float(count):
        if count <= ddof:
            # error: Incompatible types in assignment (expression has type
            # "float", variable has type "Union[floating[Any], ndarray[Any,
            # dtype[floating[Any]]]]")
            count = np.nan  # type: ignore[assignment]
            d = np.nan
    else:
        # count is not narrowed by is_float check
        count = cast(np.ndarray, count)
        mask = count <= ddof
        if mask.any():
            np.putmask(d, mask, np.nan)
            np.putmask(count, mask, np.nan)
    return count, d


@bottleneck_switch(ddof=1)
def nanstd(
    values,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    # [FUNCTION DOC]
    # Standard deviation along an axis, skipping missing values if skipna. Decorated for bottleneck dispatch with ddof default one.
    # Parameters: values, axis, skipna, ddof (default 1; divisor is count minus ddof), mask. Datetime64 is viewed as timedelta; the mask and fill are computed, then the square root of the variance kernel is returned, converted back to the original datetime-like dtype if needed.
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s.values)
    1.0
    """
    if values.dtype.kind == "M":
        unit = np.datetime_data(values.dtype)[0]
        values = values.view(f"m8[{unit}]")

    orig_dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask)

    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)


@disallow("M8", "m8")
@bottleneck_switch(ddof=1)
def nanvar(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    # [FUNCTION DOC]
    # Variance along an axis, skipping missing values if skipna. Decorated to reject datetime64 and timedelta64 and for bottleneck dispatch (ddof default one).
    # Parameters: values, axis, skipna, ddof (default 1), mask. Integers become float64 with masked entries NaN. Complex data return the sum of the variances of the real and imaginary parts. Uses a two-pass algorithm: mean first, then summed squared deviations (masked entries zero) divided by the count minus ddof. Float results keep the input precision; others are float64.
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s.values)
    1.0
    """
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if dtype.kind in "iu":
        values = values.astype("f8")
        if mask is not None:
            values[mask] = np.nan
    elif dtype.kind == "c":
        # https://en.wikipedia.org/wiki/Complex_random_variable#Variance_and_pseudo-variance
        # The variance is equal to the sum of
        # the variances of the real and imaginary part of the complex random variable.
        return nanvar(
            values.real, axis=axis, skipna=skipna, ddof=ddof, mask=mask
        ) + nanvar(values.imag, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    if values.dtype.kind == "f":
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    # xref GH10242
    # Compute variance via two-pass algorithm, which is stable against
    # cancellation errors and relatively accurate for small numbers of
    # observations.
    #
    # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)

    sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d

    # Return variance as np.float64 (the datatype used in the accumulator),
    # unless we were dealing with a float array, in which case use the same
    # precision as the original values array.
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    return result


@disallow("M8", "m8")
def nansem(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    # [FUNCTION DOC]
    # Standard error of the mean along an axis. Rejects datetime64 and timedelta64.
    # Parameters: values, axis, skipna, ddof (default 1), mask. First calls the variance kernel for validation of the dtype (discarding the result). Converts non-float, non-complex data to float64; with skipna false and any missing value, returns NaN. Returns the square root of the variance divided by the square root of the count (count already NaN when not above ddof).
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     np.float64(0.5773502691896258)
    """
    # This checks if non-numeric-like data is passed with numeric_only=False
    # and raises a TypeError otherwise
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    mask = _maybe_get_mask(values, skipna, mask)
    # Convert to bottleneck return a float
    if values.dtype.kind not in "fc":
        values = values.astype("f8")

    if not skipna and mask is not None and mask.any():
        return np.nan

    dtype_count = np.dtype(np.float64)
    if values.dtype.kind == "f":
        dtype_count = values.dtype
    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, dtype_count)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    return np.sqrt(var) / np.sqrt(count)


def _nanminmax(meth, fill_value_typ):
    # [FUNCTION DOC]
    # Factory building the minimum or maximum kernel. Parameters: meth is the numpy method name ("min" or "max") and fill_value_typ the infinity marker used to fill missing values (positive for min, negative for max). Returns the decorated reduction function.
    @bottleneck_switch(name=f"nan{meth}")
    @_datetimelike_compat
    def reduction(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: the reduction produced by the factory, decorated for bottleneck dispatch (under the matching bottleneck name) and datetime-like conversion.
        # Parameters: values, axis (default none), skipna (default True), mask (default none). Empty input returns the missing value for the dtype. Otherwise fills missing entries with the infinity marker so they do not win, applies the numpy method along the axis, and nulls out positions that were entirely missing. Returns scalar or array.
        if values.size == 0:
            return _na_for_min_count(values, axis)

        dtype = values.dtype
        values, mask = _get_values(
            values, skipna, fill_value_typ=fill_value_typ, mask=mask
        )
        result = getattr(values, meth)(axis)
        result = _maybe_null_out(
            result, axis, mask, values.shape, datetimelike=dtype.kind in "mM"
        )
        return result

    return reduction


nanmin = _nanminmax("min", fill_value_typ="+inf")
nanmax = _nanminmax("max", fill_value_typ="-inf")


def nanargmax(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    # [FUNCTION DOC]
    # Position of the maximum along an axis (or of the flattened array when axis is none).
    # Parameters: values, axis, skipna (default True), mask. Missing entries are always filled with negative infinity so they never win, then argmax is taken. Then, if a mask exists, raises a value error when every element along the axis is missing (skipna true) or any element is missing (skipna false).
    # KNOWN ISSUE: the docstring says -1 is returned in the missing case, but the code raises a value error instead.
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    np.int64(4)

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
    values, mask = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    result = values.argmax(axis)
    # error: Argument 1 to "_maybe_arg_null_out" has incompatible type "Any |
    # signedinteger[Any]"; expected "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


def nanargmin(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    # [FUNCTION DOC]
    # Position of the minimum along an axis (or of the flattened array when axis is none).
    # Same parameters and behavior as the argmax kernel, with positive infinity as the fill so missing entries never win. Raises a value error if all values (skipna true) or any value (skipna false) along the axis are missing.
    # KNOWN ISSUE: the docstring says -1 is returned in the missing case, but the code raises a value error instead.
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    np.int64(0)

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
    values, mask = _get_values(values, True, fill_value_typ="+inf", mask=mask)
    result = values.argmin(axis)
    # error: Argument 1 to "_maybe_arg_null_out" has incompatible type "Any |
    # signedinteger[Any]"; expected "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanskew(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    # [FUNCTION DOC]
    # Sample skewness (adjusted Fisher-Pearson G1) along an axis. Rejects datetime64 and timedelta64; row-wise for wide arrays.
    # Parameters: values, axis, skipna, mask. Non-float data become float64. With skipna false and any missing value returns NaN. Computes mean, then second and third central moment sums with masked entries zeroed; near-constant data are detected using a tolerance based on machine epsilon, the largest absolute value and count, and moments below it are zeroed. Result: the scaled ratio of the third moment to the second moment to the three-halves power; zero when the second moment is zero; NaN when fewer than three observations. Float input keeps its precision. Returns scalar or array.
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s.values)
    np.float64(1.7320508075688787)
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted3 = adjusted2 * adjusted
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)

    # floating point error. See comment in [nankurt]
    max_abs = np.abs(values).max(axis, initial=0.0)
    eps = np.finfo(m2.dtype).eps
    constant_tolerance2 = ((eps * max_abs) ** 2) * count
    constant_tolerance3 = ((eps * max_abs) ** 3) * count
    m2 = _zero_out_fperr(m2, constant_tolerance2)
    m3 = _zero_out_fperr(m3, constant_tolerance3)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)
        result[count < 3] = np.nan
    else:
        result = dtype.type(0) if m2 == 0 else result
        if count < 3:
            return np.nan

    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nankurt(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    # [FUNCTION DOC]
    # Sample excess kurtosis (adjusted Fisher-Pearson G2) along an axis. Rejects datetime64 and timedelta64; row-wise for wide arrays.
    # Parameters and setup as for skewness, using second and fourth central moments with the same near-constant tolerance logic (described in the comment inside). Scalar case: NaN with fewer than four observations; zero when the denominator is zero. Array case: zero where the denominator is zero and NaN where the count is below four. Float input keeps its precision.
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s.values)
    np.float64(-1.2892561983471076)
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted4 = adjusted2**2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)

    # Several floating point errors may occur during the summation due to rounding.
    # This computation is similar to the one in Scipy
    # https://github.com/scipy/scipy/blob/04d6d9c460b1fed83f2919ecec3d743cfa2e8317/scipy/stats/_stats_py.py#L1429
    # With a few modifications, like using the maximum value instead of the averages
    # and some adaptations because they use the average and we use the sum for `m2`.
    # We need to estimate an upper bound to the error to consider the data constant.
    # Let's call:
    # x: true value in data
    # y: floating point representation
    # e: relative approximation error
    # n: number of observations in array
    #
    # We have that:
    # |x - y|/|x| <= e (See https://en.wikipedia.org/wiki/Machine_epsilon)
    # (|x - y|/|x|)² <= e²
    # Σ (|x - y|/|x|)² <= ne²
    #
    # Let's say that the fperr upper bound for m2 is constrained by the summation.
    # |m2 - y|/|m2| <= ne²
    # |m2 - y| <= n|m2|e²
    #
    # We will use max (x²) to estimate |m2|
    max_abs = np.abs(values).max(axis, initial=0.0)
    eps = np.finfo(m2.dtype).eps
    constant_tolerance2 = ((eps * max_abs) ** 2) * count
    constant_tolerance4 = ((eps * max_abs) ** 4) * count
    m2 = _zero_out_fperr(m2, constant_tolerance2)
    m4 = _zero_out_fperr(m4, constant_tolerance4)

    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2**2

    if not isinstance(denominator, np.ndarray):
        # if ``denom`` is a scalar, check these corner cases first before
        # doing division
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan

    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanprod(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    # [FUNCTION DOC]
    # Product along an axis, with missing values treated as one when skipna is true. Rejects datetime64 and timedelta64; row-wise for wide arrays.
    # Parameters: values, axis, skipna, min_count (default 0), mask. Copies and replaces masked entries with one, takes the numpy product, then applies the minimum-count null-out rule. Returns scalar or array.
    """
    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s.values)
    np.float64(6.0)
    """
    mask = _maybe_get_mask(values, skipna, mask)

    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    # error: Incompatible return value type (got "Union[ndarray, float]", expected
    # "float")
    return _maybe_null_out(  # type: ignore[return-value]
        result, axis, mask, values.shape, min_count=min_count
    )


def _maybe_arg_null_out(
    result: np.ndarray,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    skipna: bool,
) -> np.ndarray | int:
    # [FUNCTION DOC]
    # HELPER: post-check for the argument-position kernels. Parameters: result (positions), axis, mask, skipna. Returns the result unchanged if no mask; otherwise raises a value error stating all values are missing (when skipna is true and everything along the axis is missing) or that a missing value was found (when skipna is false and anything is missing).
    # helper function for nanargmin/nanargmax
    if mask is None:
        return result

    if axis is None or not getattr(result, "ndim", False):
        if skipna and mask.all():
            raise ValueError("Encountered all NA values")
        elif not skipna and mask.any():
            raise ValueError("Encountered an NA value with skipna=False")
    elif skipna and mask.all(axis).any():
        raise ValueError("Encountered all NA values")
    elif not skipna and mask.any(axis).any():
        raise ValueError("Encountered an NA value with skipna=False")
    return result


def _get_counts(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    dtype: np.dtype[np.floating] = np.dtype(np.float64),
) -> np.floating | npt.NDArray[np.floating]:
    # [FUNCTION DOC]
    # HELPER: count of non-missing values.
    # Parameters: values_shape, mask (optional), axis (none counts all), dtype (floating type for the count, default float64). Without a mask the count is based on the shape. Returns a scalar of the given type, or an array of counts converted to that type when axis is given and a mask is present.
    """
    Get the count of non-null values along an axis

    Parameters
    ----------
    values_shape : tuple of int
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : scalar or array
    """
    if axis is None:
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            n = np.prod(values_shape)
        return dtype.type(n)

    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]

    if is_integer(count):
        return dtype.type(count)
    return count.astype(dtype, copy=False)


def _maybe_null_out(
    result: np.ndarray | float | NaTType,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    shape: tuple[int, ...],
    min_count: int = 1,
    datetimelike: bool = False,
) -> np.ndarray | float | NaTType:
    # [FUNCTION DOC]
    # HELPER: replace results with missing where fewer than the required number of non-missing values were present.
    # Parameters: result (array, scalar or NaT), axis, mask (optional), shape (of the original values), min_count (default 1), datetimelike (default False, use the integer NaT sentinel instead of casting to float).
    # Short-circuits when there is no mask and min_count is zero. For array results with an axis: finds positions with too few valid values; numeric results are cast to float (or complex) if needed and set to NaN, datetime-like set to the sentinel, other kinds set to none. For scalar results: replaced by NaN (typed like the result if floating) when the total valid count is below min_count; NaT stays. Array results are modified in place.
    # KNOWN ISSUE: the docstring is copied from the product kernel (it describes a product with NaNs as one) and does not describe this function.
    """
    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)
    """
    if mask is None and min_count == 0:
        # nothing to check; short-circuit
        return result

    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        else:
            # we have no nulls, kept mask=None in _maybe_get_mask
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1 :]
            null_mask = np.broadcast_to(below_count, new_shape)

        if np.any(null_mask):
            if datetimelike:
                # GH#60646 For datetimelike, no need to cast to float
                result[null_mask] = iNaT
            elif is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype("c16")
                elif not is_float_dtype(result):
                    result = result.astype("f8", copy=False)
                result[null_mask] = np.nan
            else:
                # GH12941, use None to auto cast null
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result_dtype = getattr(result, "dtype", None)
            if is_float_dtype(result_dtype):
                # error: Item "None" of "Optional[Any]" has no attribute "type"
                result = result_dtype.type("nan")  # type: ignore[union-attr]
            else:
                result = np.nan

    return result


def check_below_min_count(
    shape: tuple[int, ...], mask: npt.NDArray[np.bool_] | None, min_count: int
) -> bool:
    # [FUNCTION DOC]
    # HELPER: whether the number of valid values is below the minimum count.
    # Parameters: shape (of the values), mask (optional), min_count. Returns False when min_count is not positive. Otherwise counts non-missing as the shape product (no mask) or the mask size minus its true count, and returns True if that count is below min_count.
    """
    Check for the `min_count` keyword. Returns True if below `min_count` (when
    missing value should be returned from the reduction).

    Parameters
    ----------
    shape : tuple
        The shape of the values (`values.shape`).
    mask : ndarray[bool] or None
        Boolean numpy array (typically of same shape as `shape`) or None.
    min_count : int
        Keyword passed through from sum/prod call.

    Returns
    -------
    bool
    """
    if min_count > 0:
        if mask is None:
            # no missing values, only check size
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False


def _zero_out_fperr(arg, tol: float | np.ndarray):
    # [FUNCTION DOC]
    # HELPER: zero out values within floating point error of zero. Parameters: arg (array or scalar), tol (scalar or array tolerance). Returns arg with entries whose absolute value is below tol replaced by zero (new array for arrays; zero of the same type for scalars). Used by skewness and kurtosis for near-constant data.
    # #18044 reference this behavior to fix rolling skew/kurt issue
    if isinstance(arg, np.ndarray):
        return np.where(np.abs(arg) < tol, 0, arg)
    else:
        return arg.dtype.type(0) if np.abs(arg) < tol else arg


@disallow("M8", "m8")
def nancorr(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int | None = None,
) -> float:
    # [FUNCTION DOC]
    # Correlation of two one-dimensional arrays, ignoring pairs where either is missing. Rejects datetime64 and timedelta64.
    # Parameters: a and b (ndarrays), method (pearson default, kendall, spearman, or a callable of two arrays), min_periods (default none meaning one).
    # Raises an assertion error if the lengths differ. Drops pairs with a missing value; returns NaN if fewer than min_periods pairs remain; coerces both to numeric; applies the chosen correlation function. Returns a float.
    """
    a, b: ndarrays
    """
    if len(a) != len(b):
        raise AssertionError("Operands to nancorr must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    f = get_corr_func(method)
    return f(a, b)


def get_corr_func(
    method: CorrelationMethod,
) -> Callable[[np.ndarray, np.ndarray], float]:
    # [FUNCTION DOC]
    # Return the correlation function for a method name.
    # Parameter: method is kendall, spearman, pearson, or a callable. Kendall and spearman import the optional scientific library lazily and return the first element of its result; pearson returns the off-diagonal entry of numpy's correlation matrix. A callable is returned unchanged. Raises a value error for unknown names.
    if method == "kendall":
        from scipy.stats import kendalltau

        def func(a, b):
            # [FUNCTION DOC]
            # HELPER: nested correlation function: Kendall tau of two arrays, returning the statistic only.
            return kendalltau(a, b)[0]

        return func
    elif method == "spearman":
        from scipy.stats import spearmanr

        def func(a, b):
            # [FUNCTION DOC]
            # HELPER: nested correlation function: Spearman rank correlation of two arrays, returning the coefficient only.
            return spearmanr(a, b)[0]

        return func
    elif method == "pearson":

        def func(a, b):
            # [FUNCTION DOC]
            # HELPER: nested correlation function: Pearson correlation coefficient of two arrays from numpy's correlation matrix.
            return np.corrcoef(a, b)[0, 1]

        return func
    elif callable(method):
        return method

    raise ValueError(
        f"Unknown method '{method}', expected one of "
        "'kendall', 'spearman', 'pearson', or callable"
    )


@disallow("M8", "m8")
def nancov(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_periods: int | None = None,
    ddof: int | None = 1,
) -> float:
    # [FUNCTION DOC]
    # Covariance of two one-dimensional arrays, ignoring pairs where either is missing. Rejects datetime64 and timedelta64.
    # Parameters: a and b (ndarrays), min_periods (default none meaning one), ddof (default 1). Raises an assertion error if the lengths differ. Drops incomplete pairs; returns NaN if fewer than min_periods remain; coerces to numeric; returns the off-diagonal entry of numpy's covariance matrix with the given ddof.
    if len(a) != len(b):
        raise AssertionError("Operands to nancov must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    return np.cov(a, b, ddof=ddof)[0, 1]


def _ensure_numeric(x):
    # [FUNCTION DOC]
    # HELPER: coerce a value to a numeric type.
    # Arrays: boolean and integers become float64; object arrays inferred as string or mixed raise a type error; other object arrays are cast to complex (and reduced to real if the imaginary part is entirely zero) or, failing that, float64, raising a type error if neither works. Scalars that are not already float, integer or complex: strings raise a type error; others are converted to float, then complex, raising a type error if neither works. Returns the coerced value.
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biu":
            x = x.astype(np.float64)
        elif x.dtype == object:
            inferred = lib.infer_dtype(x)
            if inferred in ["string", "mixed"]:
                # GH#44008, GH#36703 avoid casting e.g. strings to numeric
                raise TypeError(f"Could not convert {x} to numeric")
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    # GH#29941 we get here with object arrays containing strs
                    raise TypeError(f"Could not convert {x} to numeric") from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        if isinstance(x, str):
            # GH#44008, GH#36703 avoid casting e.g. strings to numeric
            raise TypeError(f"Could not convert string '{x}' to numeric")
        try:
            x = float(x)
        except (TypeError, ValueError):
            # e.g. "1+1j" or "foo"
            try:
                x = complex(x)
            except ValueError as err:
                # e.g. "foo"
                raise TypeError(f"Could not convert {x} to numeric") from err
    return x


def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    # [FUNCTION DOC]
    # Cumulative function with missing-value support.
    # Parameters: values (ndarray or extension array; datetime-like asserted absent), accum_func (one of cumulative product, cumulative sum, running maximum or running minimum numpy functions, looked up in a table that gives the neutral fill value; an unlisted function raises a key error), skipna (keyword only).
    # If skipna is true and the dtype can hold missing values, copies the data, replaces missing entries with the neutral element, accumulates along axis zero, and restores NaN at the originally missing positions. Otherwise accumulates directly. Returns the accumulated array.
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    mask_a, mask_b = {
        np.cumprod: (1.0, np.nan),
        np.maximum.accumulate: (-np.inf, np.nan),
        np.cumsum: (0.0, np.nan),
        np.minimum.accumulate: (np.inf, np.nan),
    }[accum_func]

    # This should go through ea interface
    assert values.dtype.kind not in "mM"

    # We will be applying this function to block values
    if skipna and not issubclass(values.dtype.type, (np.integer, np.bool_)):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)

    return result
