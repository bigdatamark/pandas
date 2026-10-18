"""
Functions for implementing 'astype' methods according to pandas conventions,
particularly ones that differ from numpy.
"""

# [MODULE DOC]
# Array type-conversion routines behind the astype methods of the data
# structures. Public contents: astype_array (convert an array to a dtype),
# astype_array_safe (validated entry point with error-handling mode) and
# astype_is_view (predict whether a conversion can avoid copying). Private
# helpers handle numpy conversion with missing-value safety. Differences from
# plain numpy: non-finite floats cannot become integers, fixed-width string
# results are stored as object arrays, datetime-like dtypes need units, and
# extension dtypes dispatch to their own array construction.
# Used by the array, block and generic container modules. No import-time side
# effects beyond imports.

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError

from pandas.core.dtypes.common import (
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    NumpyEADtype,
    PeriodDtype,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        IgnoreRaise,
    )

    from pandas.core.arrays import ExtensionArray


@overload
def _astype_nansafe(
    arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...
) -> np.ndarray:
    # [FUNCTION DOC]
    # Typing overload stub (numpy dtype target gives an ndarray); see the
    # implementation below.
    ...


@overload
def _astype_nansafe(
    arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...
) -> ExtensionArray:
    # [FUNCTION DOC]
    # Typing overload stub (extension dtype target gives an extension array); see
    # the implementation below.
    ...


def _astype_nansafe(
    arr: np.ndarray, dtype: DtypeObj, copy: bool = True, skipna: bool = False
) -> ArrayLike:
    # [FUNCTION DOC]
    # HELPER: numpy-array conversion used by the public conversion routine.
    # Inputs: arr, a numpy array; dtype, numpy or extension dtype target; copy
    # (default true), whether to force a copy; skipna (default false), whether
    # missing values are skipped when converting to a string type.
    # Behaviour in order: an extension dtype builds its array type from the
    # sequence (honouring copy). A target that is neither dtype kind raises a
    # value error. Datetime-like sources are wrapped in their array type, cast
    # there, and returned as a plain numpy array. String targets flatten
    # multi-dimensional input, convert elements to strings (keeping missing
    # values unconverted) and restore the shape. Floating sources with integer
    # targets go through the missing-value-safe float-to-integer routine. Object
    # sources with a datetime target are parsed by the datetime array type;
    # with a timedelta target are parsed to nanosecond timedeltas then cast.
    # Unit-less datetime or timedelta targets raise a value error telling the
    # user to add a unit. Finally the array is cast with numpy, always copying
    # when copy is true or when either side is object (numpy cannot view
    # to/from object).
    # Returns the converted array.
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if
        e.g. the item sizes don't align.
    skipna: bool, default False
        Whether or not we should skip NaN when casting as a string-type.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """

    # dispatch on extension dtype if needed
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)

    elif not isinstance(dtype, np.dtype):  # pragma: no cover
        raise ValueError("dtype must be np.dtype or ExtensionDtype")

    if arr.dtype.kind in "mM":
        from pandas.core.construction import ensure_wrapped_if_datetimelike

        arr = ensure_wrapped_if_datetimelike(arr)
        res = arr.astype(dtype, copy=copy)
        return np.asarray(res)

    if issubclass(dtype.type, str):
        shape = arr.shape
        if arr.ndim > 1:
            arr = arr.ravel()
        return lib.ensure_string_array(
            arr, skipna=skipna, convert_na_value=False
        ).reshape(shape)

    elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
        return _astype_float_to_int_nansafe(arr, dtype, copy)

    elif arr.dtype == object:
        # if we have a datetime/timedelta array of objects
        # then coerce to datetime64[ns] and use DatetimeArray.astype

        if lib.is_np_dtype(dtype, "M"):
            from pandas.core.arrays import DatetimeArray

            dta = DatetimeArray._from_sequence(arr, dtype=dtype)
            return dta._ndarray

        elif lib.is_np_dtype(dtype, "m"):
            from pandas.core.construction import ensure_wrapped_if_datetimelike

            # bc we know arr.dtype == object, this is equivalent to
            #  `np.asarray(to_timedelta(arr))`, but using a lower-level API that
            #  does not require a circular import.
            tdvals = array_to_timedelta64(arr)

            tda = ensure_wrapped_if_datetimelike(tdvals)
            return tda.astype(dtype, copy=False)._ndarray

    if dtype.name in ("datetime64", "timedelta64"):
        msg = (
            f"The '{dtype.name}' dtype has no unit. Please pass in "
            f"'{dtype.name}[ns]' instead."
        )
        raise ValueError(msg)

    if copy or object in (arr.dtype, dtype):
        # Explicit copy, or required since NumPy can't view from / to object.
        return arr.astype(dtype, copy=True)

    return arr.astype(dtype, copy=copy)


def _astype_float_to_int_nansafe(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: float-to-integer conversion guard. Inputs: values, float array;
    # dtype, integer numpy dtype; copy, whether to copy. Raises the integer
    # casting error for NaN or infinite values. For unsigned targets raises a
    # value error if any value is negative. Then casts with numpy, suppressing
    # numpy runtime warnings. Returns the integer array.
    # KNOWN ISSUE: the first error message is built from string pieces that lack
    # separating spaces ("integer." runs into "Replace", and "type" runs into
    # "that").
    """
    astype with a check preventing converting NaN to a meaningless integer value.
    """
    if not np.isfinite(values).all():
        raise IntCastingNaNError(
            "Cannot convert non-finite values (NA or inf) to integer."
            "Replace or remove non-finite values or cast to an integer type"
            "that supports these values (e.g. 'Int64')"
        )
    if dtype.kind == "u":
        # GH#45151
        if not (values >= 0).all():
            raise ValueError(f"Cannot losslessly cast from {values.dtype} to {dtype}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return values.astype(dtype, copy=copy)


def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = False) -> ArrayLike:
    # [FUNCTION DOC]
    # Converts an array to a dtype. Inputs: values, numpy or extension array;
    # dtype, an already-resolved dtype object; copy (default false). If the
    # dtype already matches, returns the input itself or a copy. Extension-array
    # inputs use their own conversion method (honouring copy); numpy inputs go
    # through the missing-value-safe helper. If the target is a numpy dtype and
    # the result has a fixed-width string dtype, it is converted to an object
    # array (pandas does not store numpy string dtypes). Returns the result.
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
    if values.dtype == dtype:
        if copy:
            return values.copy()
        return values

    if not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        values = values.astype(dtype, copy=copy)

    else:
        values = _astype_nansafe(values, dtype, copy=copy)

    # in pandas we don't store numpy str dtypes, so convert to object
    if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
        values = np.array(values, dtype=object)

    return values


def astype_array_safe(
    values: ArrayLike, dtype, copy: bool = False, errors: IgnoreRaise = "raise"
) -> ArrayLike:
    # [FUNCTION DOC]
    # Validated conversion used by the astype methods of Series and DataFrame.
    # Inputs: values, numpy or extension array; dtype, anything convertible to a
    # dtype (a string, type or dtype object); copy (default false); errors,
    # "raise" (default) or "ignore". Raises a value error for any other errors
    # setting, and a type error if dtype is an extension dtype class rather than
    # an instance. Resolves dtype (a wrapped-numpy extension dtype is unwrapped
    # to the numpy dtype), then converts. With errors set to ignore, value or
    # type errors during conversion are swallowed and the original values are
    # returned; otherwise they propagate. Returns the converted (or original)
    # array.
    # KNOWN ISSUE: the docstring's description sentence is truncated ("not
    # allowing )").
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
    errors_legal_values = ("raise", "ignore")

    if errors not in errors_legal_values:
        invalid_arg = (
            "Expected value of kwarg 'errors' to be one of "
            f"{list(errors_legal_values)}. Supplied value is '{errors}'"
        )
        raise ValueError(invalid_arg)

    if inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype):
        msg = (
            f"Expected an instance of {dtype.__name__}, "
            "but got the class instead. Try instantiating 'dtype'."
        )
        raise TypeError(msg)

    dtype = pandas_dtype(dtype)
    if isinstance(dtype, NumpyEADtype):
        # Ensure we don't end up with a NumpyExtensionArray
        dtype = dtype.numpy_dtype

    try:
        new_values = astype_array(values, dtype, copy=copy)
    except (ValueError, TypeError):
        # e.g. _astype_nansafe can fail on object-dtype of strings
        #  trying to convert to float
        if errors == "ignore":
            new_values = values
        else:
            raise

    return new_values


def astype_is_view(dtype: DtypeObj, new_dtype: DtypeObj) -> bool:
    # [FUNCTION DOC]
    # Predicts whether converting from dtype to new_dtype can return a view of
    # the original data (True) or must be a copy (False). True also means
    # "not guaranteed to be a copy".
    # Rules in order: same-kind numeric or boolean numpy-like dtypes are a view
    # when item sizes match. (If one dtype is numpy and the other extension, the
    # two are swapped so the extension one is examined first.) Equal dtypes: view.
    # Two numpy dtypes that are unequal: copy. Two string dtypes: copy only for
    # arrow-backed string to object, else treated as a possible view. Two object
    # dtypes: view. Two datetime/timedelta dtypes: view when units match.
    # Timezone, period or interval dtype to object: copy. Categorical to a
    # non-categorical dtype: copy. Otherwise, if both have an underlying numpy
    # dtype, view only when those are equal. In all remaining cases returns True.
    """Checks if astype avoided copying the data.

    Parameters
    ----------
    dtype : Original dtype
    new_dtype : target dtype

    Returns
    -------
    True if new data is a view or not guaranteed to be a copy, False otherwise
    """
    if dtype.kind in "iufb" and dtype.kind == new_dtype.kind:
        # fastpath for numeric dtypes
        if hasattr(dtype, "itemsize") and hasattr(new_dtype, "itemsize"):
            return dtype.itemsize == new_dtype.itemsize  # pyright: ignore[reportAttributeAccessIssue]

    if isinstance(dtype, np.dtype) and not isinstance(new_dtype, np.dtype):
        new_dtype, dtype = dtype, new_dtype

    if dtype == new_dtype:
        return True

    elif isinstance(dtype, np.dtype) and isinstance(new_dtype, np.dtype):
        # Only equal numpy dtypes avoid a copy
        return False

    elif is_string_dtype(dtype) and is_string_dtype(new_dtype):
        from pandas.core.arrays.string_ import StringDtype

        if (
            isinstance(dtype, StringDtype)
            and dtype.storage == "pyarrow"
            and new_dtype == "object"
        ):
            # for conversion of pyarrow array to numpy object array -> always a copy
            return False
        # Potentially! a view when converting from object to string
        return True

    elif is_object_dtype(dtype) and new_dtype.kind == "O":
        # When the underlying array has dtype object, we don't have to make a copy
        return True

    elif dtype.kind in "mM" and new_dtype.kind in "mM":
        dtype = getattr(dtype, "numpy_dtype", dtype)
        new_dtype = getattr(new_dtype, "numpy_dtype", new_dtype)
        return getattr(dtype, "unit", None) == getattr(new_dtype, "unit", None)

    elif new_dtype == object and isinstance(
        dtype, (DatetimeTZDtype, PeriodDtype, IntervalDtype)
    ):
        return False

    elif isinstance(dtype, CategoricalDtype) and not isinstance(
        new_dtype, CategoricalDtype
    ):
        return False

    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    new_numpy_dtype = getattr(new_dtype, "numpy_dtype", None)

    if numpy_dtype is None and isinstance(dtype, np.dtype):
        numpy_dtype = dtype

    if new_numpy_dtype is None and isinstance(new_dtype, np.dtype):
        new_numpy_dtype = new_dtype

    if numpy_dtype is not None and new_numpy_dtype is not None:
        # if both have NumPy dtype or one of them is a numpy dtype
        # they are only a view when the numpy dtypes are equal, e.g.
        # int64 -> Int64 or int64[pyarrow]
        # int64 -> Int32 copies
        return numpy_dtype == new_numpy_dtype

    # Assume this is a view since we don't know for sure if a copy was made
    return True
