# [MODULE DOC]
# Positional "take" with fill-value support for numpy and extension arrays: gather
# elements along an axis by an integer indexer in which minus one marks "missing",
# producing the missing fill value at those positions in one pass.
# Public contents: take_nd (general entry point, dispatching extension arrays to
# their own take) and take_2d_multi (take along both axes of a 2D array).
# Private contents: the numpy implementation, lookup tables (1D, 2D axis 0, 2D
# axis 1 and 2D multi) mapping a pair of (input dtype name, output dtype name) to
# a compiled routine from the algorithms extension module (some wrapped so that
# boolean and datetime-like data is processed as unsigned/signed integers), wrapper
# factories, an object-dtype fallback, and promotion/mask preprocessing.
# KNOWN ISSUE: in the 1D lookup table, the entries for unsigned 16, 32 and 64 bit
# integers use the signed 64-bit integer as the output dtype in the key while
# pointing to same-dtype routines, unlike the 2D tables, which key them by the
# unsigned dtype on both sides; a 1D lookup with unsigned input and output misses
# and falls back to the generic converting path.
# Import-time side effects: builds the lookup tables.
from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)

from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
)
from pandas.core.dtypes.missing import na_value_for_dtype

from pandas.core.construction import ensure_wrapped_if_datetimelike

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        npt,
    )

    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
    from pandas.core.arrays.base import ExtensionArray


@overload
def take_nd(
    arr: np.ndarray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> np.ndarray:
    # [FUNCTION DOC]
    # Typing overload stub for numpy array input (returns a numpy array); see the
    # implementation below for behavior.
    ...


@overload
def take_nd(
    arr: ExtensionArray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> ArrayLike:
    # [FUNCTION DOC]
    # Typing overload stub for extension array input (returns an array-like); see
    # the implementation below for behavior.
    ...


def take_nd(
    arr: ArrayLike,
    indexer,
    axis: AxisInt = 0,
    fill_value=lib.no_default,
    allow_fill: bool = True,
) -> ArrayLike:
    # [FUNCTION DOC]
    # General take along an axis with fill. Inputs: arr (numpy or extension array),
    # indexer (already validated 1D integer positions, no out-of-bounds values; minus
    # one means missing; for numpy input it may be none meaning the identity),
    # axis (default 0), fill_value (default: the missing value appropriate for
    # the array dtype) and allow_fill (default true; if false, the indexer must
    # contain no minus-one entries, and the result is undefined otherwise).
    # Behavior: when a fill value is given for datetime64/timedelta64 numpy data,
    # the dtype is promoted for it and the array cast first if needed. Extension
    # arrays dispatch to their own take (passing the axis only for those that
    # support more than one dimension, i.e. datetime-like arrays). Numpy data goes
    # through the private numpy implementation.
    # Returns an array of the same kind as the input (extension input can give an
    # extension output; numpy may be promoted to hold the fill value).
    # KNOWN ISSUE: the docstring says the fill_value default is NaN, but the
    # actual default is a sentinel that selects the dtype-appropriate missing value.
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        May be the same type as the input, or cast to an ndarray.
    """
    if fill_value is lib.no_default:
        fill_value = na_value_for_dtype(arr.dtype, compat=False)
    elif lib.is_np_dtype(arr.dtype, "mM"):
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if arr.dtype != dtype:
            # EA.take is strict about returning a new object of the same type
            # so for that case cast upfront
            arr = arr.astype(dtype)

    if not isinstance(arr, np.ndarray):
        # i.e. ExtensionArray,
        # includes for EA to catch DatetimeArray, TimedeltaArray
        if not is_1d_only_ea_dtype(arr.dtype):
            # i.e. DatetimeArray, TimedeltaArray
            arr = cast("NDArrayBackedExtensionArray", arr)
            return arr.take(
                indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis
            )

        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    arr = np.asarray(arr)
    return _take_nd_ndarray(arr, indexer, axis, fill_value, allow_fill)


def _take_nd_ndarray(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp] | None,
    axis: AxisInt,
    fill_value,
    allow_fill: bool,
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: numpy implementation behind the general take. Inputs: arr (numpy
    # array), indexer (integer positions or none for identity), axis, fill_value
    # and allow_fill.
    # Behavior: a missing indexer becomes the identity positions. Dtype promotion
    # and the missing-position mask are computed by the preprocessing helper. A
    # Fortran-ordered 2D input is transposed (and the axis mirrored) so the
    # compiled routine sees C order, and the output is allocated Fortran-ordered
    # when the take is along the last axis of a Fortran-contiguous input. The
    # compiled routine for the dtype combination is looked up (object fallback if
    # none) and fills a freshly allocated output. Returns the output array
    # (transposed back if needed).
    # KNOWN ISSUE: when the indexer is none, the dtype and fill value assigned in
    # this branch are always overwritten by the preprocessing call below, so those
    # assignments have no effect.
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        dtype, fill_value = arr.dtype, arr.dtype.type()
    else:
        indexer = ensure_platform_int(indexer)

    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, allow_fill
    )

    flip_order = False
    if arr.ndim == 2 and arr.flags.f_contiguous:
        flip_order = True

    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    out_shape_ = list(arr.shape)
    out_shape_[axis] = len(indexer)
    out_shape = tuple(out_shape_)
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        # minor tweak that can make an order-of-magnitude difference
        # for dataframes initialized directly from 2-d ndarrays
        # (s.t. df.values is c-contiguous and df._mgr.blocks[0] is its
        # f-contiguous transpose)
        out = np.empty(out_shape, dtype=dtype, order="F")
    else:
        out = np.empty(out_shape, dtype=dtype)

    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info
    )
    func(arr, indexer, out, fill_value)

    if flip_order:
        out = out.T
    return out


def take_2d_multi(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    fill_value=np.nan,
) -> np.ndarray:
    # [FUNCTION DOC]
    # Take along both axes of a 2D numpy array at once. Inputs: arr (2D array),
    # indexer (pair of row positions and column positions; neither may be none, an
    # assertion error otherwise; minus one marks missing) and fill_value (default
    # NaN).
    # Behavior: indexers are converted to platform integers; the dtype is promoted
    # to hold the fill value only if some position is actually missing; a compiled
    # routine for the dtype pair is looked up (with a conversion wrapper if only the
    # output dtype has one), falling back to a slow python loop. Returns a new
    # array of shape (number of rows taken, number of columns taken). Only used by
    # frame multi-axis reindexing, so the indexer is assumed well-behaved.
    """
    Specialized Cython take which sets NaN values in one pass.
    """
    # This is only called from one place in DataFrame._reindex_multi,
    #  so we know indexer is well-behaved.
    assert indexer is not None
    assert indexer[0] is not None
    assert indexer[1] is not None

    row_idx, col_idx = indexer

    row_idx = ensure_platform_int(row_idx)
    col_idx = ensure_platform_int(col_idx)
    indexer = row_idx, col_idx
    mask_info = None

    # check for promotion based on types only (do this first because
    # it's faster than computing a mask)
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)
    if dtype != arr.dtype:
        # check if promotion is actually required based on indexer
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
        mask_info = (row_mask, col_mask), (row_needs, col_needs)

        if not (row_needs or col_needs):
            # if not, then depromote, set fill_value to dummy
            # (it won't be used but we don't want the cython code
            # to crash when trying to cast it to dtype)
            dtype, fill_value = arr.dtype, arr.dtype.type()

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    out_shape = len(row_idx), len(col_idx)
    out = np.empty(out_shape, dtype=dtype)

    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if func is not None:
            func = _convert_wrapper(func, out.dtype)

    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        # test_reindex_multi
        _take_2d_multi_object(
            arr, indexer, out, fill_value=fill_value, mask_info=mask_info
        )

    return out


@functools.lru_cache
def _get_take_nd_function_cached(
    ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt
):
    # [FUNCTION DOC]
    # HELPER: cached lookup of the compiled take routine (the cache is keyed on all
    # four arguments, so no unhashable mask info may be passed). Inputs: ndim (1 or
    # 2), arr_dtype, out_dtype and axis (for 2D, zero selects the axis-0 table,
    # anything else the axis-1 table).
    # Behavior: first looks up the exact dtype pair; if absent, looks up the
    # (output, output) pair and wraps it to convert the input array to the output
    # dtype first. Returns the routine, or none if neither exists.
    # KNOWN ISSUE: for ndim values other than 1 or 2 the local routine variable is
    # never assigned and the function would fail; callers only pass 1 or 2.
    """
    Part of _get_take_nd_function below that doesn't need `mask_info` and thus
    can be cached (mask_info potentially contains a numpy ndarray which is not
    hashable and thus cannot be used as argument for cached function).
    """
    tup = (arr_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        return func

    # We get here with string, uint, float16, and complex dtypes that could
    #  potentially be handled in algos_take_helper.
    #  Also a couple with (M8[ns], object) and (m8[ns], object)
    tup = (out_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        func = _convert_wrapper(func, out_dtype)
        return func

    return None


def _get_take_nd_function(
    ndim: int,
    arr_dtype: np.dtype,
    out_dtype: np.dtype,
    axis: AxisInt = 0,
    mask_info=None,
):
    # [FUNCTION DOC]
    # HELPER: returns the take routine for the dimension, axis and dtypes. Inputs:
    # ndim, arr_dtype, out_dtype, axis (default 0) and mask_info (optional
    # precomputed mask and need-flag, only used by the object fallback). For up to
    # two dimensions the cached lookup is used; if no routine exists (or ndim is
    # larger), a fallback is returned that converts the indexer to platform
    # integers and calls the generic object-based routine. The returned callable
    # takes the array, indexer, output array and fill value, and fills the output.
    """
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    """
    func = None
    if ndim <= 2:
        # for this part we don't need `mask_info` -> use the cached algo lookup
        func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis)

    if func is None:

        def func(arr, indexer, out, fill_value=np.nan) -> None:
            # [FUNCTION DOC]
            # HELPER: fallback routine using the generic object-based take; fills
            # out in place using the surrounding axis and mask info.
            indexer = ensure_platform_int(indexer)
            _take_nd_object(
                arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info
            )

    return func


def _view_wrapper(f, arr_dtype=None, out_dtype=None, fill_wrap=None):
    # [FUNCTION DOC]
    # HELPER: wraps a compiled take routine f so it can run on reinterpreted data
    # (for example booleans as unsigned bytes, datetime-like as 64-bit integers).
    # Inputs: f, arr_dtype and out_dtype (dtypes to view the input and output as;
    # none means no view) and fill_wrap (callable applied to the fill value after
    # converting a timedelta/datetime fill to nanosecond resolution; none means the
    # fill value is passed unchanged). Returns the wrapper function.
    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: the wrapper; views arr and out as the requested dtypes, converts
        # the fill value if a wrapping callable was given, then calls the wrapped
        # routine, which fills out in place. Returns nothing.
        # KNOWN ISSUE: a FIXME notes the datetime-like fill conversion assumes
        # nanosecond resolution and does not verify that resolutions match.
        if arr_dtype is not None:
            arr = arr.view(arr_dtype)
        if out_dtype is not None:
            out = out.view(out_dtype)
        if fill_wrap is not None:
            # FIXME: if we get here with dt64/td64 we need to be sure we have
            #  matching resos
            if fill_value.dtype.kind == "m":
                fill_value = fill_value.astype("m8[ns]")
            else:
                fill_value = fill_value.astype("M8[ns]")
            fill_value = fill_wrap(fill_value)

        f(arr, indexer, out, fill_value=fill_value)

    return wrapper


def _convert_wrapper(f, conv_dtype):
    # [FUNCTION DOC]
    # HELPER: wraps a compiled take routine f so that the input array is first
    # converted to conv_dtype. Returns the wrapper function.
    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: the wrapper; for an object target dtype, datetime-like input is
        # first wrapped so it is not cast to integers; the array is converted to
        # the target dtype (a copy) and the routine fills out in place.
        if conv_dtype == object:
            # GH#39755 avoid casting dt64/td64 to integers
            arr = ensure_wrapped_if_datetimelike(arr)
        arr = arr.astype(conv_dtype)
        f(arr, indexer, out, fill_value=fill_value)

    return wrapper


_take_1d_dict = {
    ("int8", "int8"): libalgos.take_1d_int8_int8,
    ("int8", "int32"): libalgos.take_1d_int8_int32,
    ("int8", "int64"): libalgos.take_1d_int8_int64,
    ("int8", "float64"): libalgos.take_1d_int8_float64,
    ("int16", "int16"): libalgos.take_1d_int16_int16,
    ("int16", "int32"): libalgos.take_1d_int16_int32,
    ("int16", "int64"): libalgos.take_1d_int16_int64,
    ("int16", "float64"): libalgos.take_1d_int16_float64,
    ("int32", "int32"): libalgos.take_1d_int32_int32,
    ("int32", "int64"): libalgos.take_1d_int32_int64,
    ("int32", "float64"): libalgos.take_1d_int32_float64,
    ("int64", "int64"): libalgos.take_1d_int64_int64,
    ("uint8", "uint8"): libalgos.take_1d_bool_bool,
    ("uint16", "int64"): libalgos.take_1d_uint16_uint16,
    ("uint32", "int64"): libalgos.take_1d_uint32_uint32,
    ("uint64", "int64"): libalgos.take_1d_uint64_uint64,
    ("int64", "float64"): libalgos.take_1d_int64_float64,
    ("float32", "float32"): libalgos.take_1d_float32_float32,
    ("float32", "float64"): libalgos.take_1d_float32_float64,
    ("float64", "float64"): libalgos.take_1d_float64_float64,
    ("object", "object"): libalgos.take_1d_object_object,
    ("bool", "bool"): _view_wrapper(libalgos.take_1d_bool_bool, np.uint8, np.uint8),
    ("bool", "object"): _view_wrapper(libalgos.take_1d_bool_object, np.uint8, None),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64
    ),
}

_take_2d_axis0_dict = {
    ("int8", "int8"): libalgos.take_2d_axis0_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis0_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis0_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis0_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis0_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis0_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis0_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis0_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis0_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis0_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis0_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis0_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis0_int64_float64,
    ("uint8", "uint8"): libalgos.take_2d_axis0_bool_bool,
    ("uint16", "uint16"): libalgos.take_2d_axis0_uint16_uint16,
    ("uint32", "uint32"): libalgos.take_2d_axis0_uint32_uint32,
    ("uint64", "uint64"): libalgos.take_2d_axis0_uint64_uint64,
    ("float32", "float32"): libalgos.take_2d_axis0_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis0_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis0_float64_float64,
    ("object", "object"): libalgos.take_2d_axis0_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis0_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis0_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

_take_2d_axis1_dict = {
    ("int8", "int8"): libalgos.take_2d_axis1_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis1_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis1_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis1_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis1_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis1_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis1_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis1_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis1_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis1_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis1_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis1_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis1_int64_float64,
    ("uint8", "uint8"): libalgos.take_2d_axis1_bool_bool,
    ("uint16", "uint16"): libalgos.take_2d_axis1_uint16_uint16,
    ("uint32", "uint32"): libalgos.take_2d_axis1_uint32_uint32,
    ("uint64", "uint64"): libalgos.take_2d_axis1_uint64_uint64,
    ("float32", "float32"): libalgos.take_2d_axis1_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis1_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis1_float64_float64,
    ("object", "object"): libalgos.take_2d_axis1_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis1_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis1_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

_take_2d_multi_dict = {
    ("int8", "int8"): libalgos.take_2d_multi_int8_int8,
    ("int8", "int32"): libalgos.take_2d_multi_int8_int32,
    ("int8", "int64"): libalgos.take_2d_multi_int8_int64,
    ("int8", "float64"): libalgos.take_2d_multi_int8_float64,
    ("int16", "int16"): libalgos.take_2d_multi_int16_int16,
    ("int16", "int32"): libalgos.take_2d_multi_int16_int32,
    ("int16", "int64"): libalgos.take_2d_multi_int16_int64,
    ("int16", "float64"): libalgos.take_2d_multi_int16_float64,
    ("int32", "int32"): libalgos.take_2d_multi_int32_int32,
    ("int32", "int64"): libalgos.take_2d_multi_int32_int64,
    ("int32", "float64"): libalgos.take_2d_multi_int32_float64,
    ("int64", "int64"): libalgos.take_2d_multi_int64_int64,
    ("int64", "float64"): libalgos.take_2d_multi_int64_float64,
    ("float32", "float32"): libalgos.take_2d_multi_float32_float32,
    ("float32", "float64"): libalgos.take_2d_multi_float32_float64,
    ("float64", "float64"): libalgos.take_2d_multi_float64_float64,
    ("object", "object"): libalgos.take_2d_multi_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_multi_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_multi_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}


def _take_nd_object(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    out: np.ndarray,
    axis: AxisInt,
    fill_value,
    mask_info,
) -> None:
    # [FUNCTION DOC]
    # HELPER: slow generic take for dtype combinations without a compiled routine.
    # Inputs: arr, indexer (platform integers, minus one = missing), out (output
    # array, filled in place), axis, fill_value and mask_info (optional pair of
    # mask and need-flag; computed from the indexer when none).
    # Behavior: converts arr to the output dtype if they differ, takes along the
    # axis into out when the axis is non-empty, then assigns fill_value at the
    # masked positions if needed. Returns nothing.
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value


def _take_2d_multi_object(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value,
    mask_info,
) -> None:
    # [FUNCTION DOC]
    # HELPER: slow python-loop fallback for the two-axis take. Inputs: arr, indexer
    # (pair of row and column positions, minus one = missing), out (output array,
    # filled in place), fill_value and mask_info (optional precomputed pair of masks
    # and pair of flags; recomputed when none). Behavior: fills the missing rows and
    # columns with fill_value (unless it is none), then copies each element for
    # positions where both indexers are valid. Returns nothing.
    # this is not ideal, performance-wise, but it's better than raising
    # an exception (best to optimize in Cython to avoid getting here)
    row_idx, col_idx = indexer  # both np.intp
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    for i, u_ in enumerate(row_idx):
        if u_ != -1:
            for j, v in enumerate(col_idx):
                if v != -1:
                    out[i, j] = arr[u_, v]


def _take_preprocess_indexer_and_fill_value(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    fill_value,
    allow_fill: bool,
    mask: npt.NDArray[np.bool_] | None = None,
):
    # [FUNCTION DOC]
    # HELPER: determines the output dtype, fill value and mask info for a numpy
    # take. Inputs: arr, indexer (platform integers), fill_value, allow_fill and an
    # optional precomputed mask of missing positions.
    # Behavior: if filling is disallowed, uses the input dtype, a dummy fill value
    # and mask info of (none, false). Otherwise the dtype is promoted to hold the
    # fill value; if promotion occurred, the mask (computed from minus-one entries
    # when not given) is evaluated and, if no position is actually missing, the
    # promotion is undone with a dummy fill value.
    # Returns a triple: dtype, fill_value and mask info (a pair of mask and a flag
    # saying masking is needed, or none when no promotion occurred).
    mask_info: tuple[np.ndarray | None, bool] | None = None

    if not allow_fill:
        dtype, fill_value = arr.dtype, arr.dtype.type()
        mask_info = None, False
    else:
        # check for promotion based on types only (do this first because
        # it's faster than computing a mask)
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if dtype != arr.dtype:
            # check if promotion is actually required based on indexer
            if mask is not None:
                needs_masking = True
            else:
                mask = indexer == -1
                needs_masking = bool(mask.any())
            mask_info = mask, needs_masking
            if not needs_masking:
                # if not, then depromote, set fill_value to dummy
                # (it won't be used but we don't want the cython code
                # to crash when trying to cast it to dtype)
                dtype, fill_value = arr.dtype, arr.dtype.type()

    return dtype, fill_value, mask_info
