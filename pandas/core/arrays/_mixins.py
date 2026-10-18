# [MODULE DOC]
# Internal module defining the base class for extension arrays that are backed by
# a single numpy array, plus a small decorator for operating on flattened data.
# Public contents: the ravel-compat decorator and the numpy-backed extension
# array base class. The latter combines the compiled backing-data class with the
# generic extension array base and is subclassed by the datetime-like arrays,
# categorical, numpy-backed array and others. Depends on the generic array
# algorithms, missing-value helpers, indexer validation and sorting helpers.
# No import-time side effects.
from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import is_supported_dtype
from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    F,
    FillnaOptions,
    PositionalIndexer2D,
    PositionalIndexerTuple,
    ScalarIndexer,
    SequenceIndexer,
    Shape,
    TakeIndexer,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import array_equivalent

from pandas.core import missing
from pandas.core.algorithms import (
    take,
    unique,
    value_counts_internal as value_counts,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
)
from pandas.core.sorting import nargminmax

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        NumpySorter,
        NumpyValueArrayLike,
    )

    from pandas import Series


def ravel_compat(meth: F) -> F:
    # [FUNCTION DOC]
    # Decorator factory step: wraps a method of a (possibly 2-dimensional) array-backed
    # extension array so that it can be applied to flat data. Parameter meth is the
    # method to wrap; it is expected to operate on one-dimensional arrays (for
    # example because it calls into compiled code). Returns the wrapper, typed as the
    # original callable type.
    """
    Decorator to ravel a 2D array before passing it to a cython operation,
    then reshape the result to our own shape.
    """

    @wraps(meth)
    def method(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the wrapper created by the ravel_compat decorator. For one-dimensional
        # arrays it simply calls the wrapped method. Otherwise it remembers whether the
        # backing ndarray was Fortran-contiguous, flattens the array in memory order
        # (keeping layout), calls the wrapped method on the flattened array, and
        # reshapes
        # the result back to the original shape using Fortran order if the original was
        # Fortran-contiguous and C order otherwise. Arguments are passed through.
        # Returns
        # the reshaped result.
        if self.ndim == 1:
            return meth(self, *args, **kwargs)

        flags = self._ndarray.flags
        flat = self.ravel("K")
        result = meth(flat, *args, **kwargs)
        order = "F" if flags.f_contiguous else "C"
        return result.reshape(self.shape, order=order)

    return cast(F, method)


class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    # [CLASS DOC]
    # Base class for extension arrays whose data lives in a single numpy array
    # (datetime-like arrays, categorical, numpy-backed arrays, etc.). Inherits the
    # compiled backing-data base class (which provides the stored ndarray, dtype,
    # shape and the construct-from-backing-data hook) and the generic extension array
    # interface. State: the backing ndarray attribute and an internal fill value that
    # marks missing entries inside it (such as -1 for categoricals or the integer
    # missing sentinel for periods); outside of object dtype, the missing-value mask
    # should be exactly the positions equal to this sentinel. Also uses a read-only
    # flag for view protection. Subclasses must define the scalar boxing and
    # validation hooks. Methods here implement take, equality, unique, concat,
    # searchsorted, shift, item get and set, fill/pad, insert, value counts,
    # quantile, reduction wrapping and empty construction on top of the ndarray.
    """
    ExtensionArray that is backed by a single NumPy ndarray.
    """

    _ndarray: np.ndarray

    # scalar used to denote NA value inside our self._ndarray, e.g. -1
    #  for Categorical, iNaT for Period. Outside of object dtype,
    #  self.isna() should be exactly locations in self._ndarray with
    #  _internal_fill_value.
    _internal_fill_value: Any

    def _box_func(self, x):
        # [FUNCTION DOC]
        # HELPER: hook that converts a raw element of the backing array into the scalar
        # type exposed to users. Default returns the input unchanged; subclasses
        # override
        # (for instance to build timestamps).
        """
        Wrap numpy type in our dtype.type if necessary.
        """
        return x

    def _validate_scalar(self, value):
        # [FUNCTION DOC]
        # HELPER: abstract hook, used by take with fill, shift, insert and others, to
        # validate a scalar and convert it to its storage representation. Subclasses
        # must
        # override; this default raises an abstract-method error.
        # used by NDArrayBackedExtensionIndex.insert
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------------

    @overload
    def view(self) -> Self:
        # [FUNCTION DOC]
        # Typing-only overload stub: no-argument form, returns the same type. See
        # the implementation below.
        ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike:
        # [FUNCTION DOC]
        # Typing-only overload stub: form with a dtype argument, returns an
        # array-like. See the implementation below.
        ...

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # [FUNCTION DOC]
        # Return a view of the data, optionally reinterpreted as another dtype.
        # Parameter dtype is None (default) or a dtype-like. If None or identical to the
        # current dtype, returns a new array of the same type over the same backing
        # data.
        # If dtype is a plain class (not a dtype instance, e.g. an ndarray type), the
        # call is passed to the backing ndarray. Otherwise the dtype is normalized and:
        # a period dtype gives a period array built from the data viewed as 64-bit
        # integers; a timezone-aware datetime dtype gives a datetime array over the data
        # viewed with that unit; a supported numpy datetime64 or timedelta64 dtype gives
        # a datetime or timedelta array respectively; any other dtype returns a plain
        # numpy view of the backing data. Returns an array-like. No copy of data.
        # We handle datetime64, datetime64tz, timedelta64, and period
        #  dtypes here. Everything else we pass through to the underlying
        #  ndarray.
        if dtype is None or dtype is self.dtype:
            return self._from_backing_data(self._ndarray)

        if isinstance(dtype, type):
            # we sometimes pass non-dtype objects, e.g np.ndarray;
            #  pass those through to the underlying ndarray
            return self._ndarray.view(dtype)

        dtype = pandas_dtype(dtype)
        arr = self._ndarray

        if isinstance(dtype, PeriodDtype):
            cls = dtype.construct_array_type()
            return cls(arr.view("i8"), dtype=dtype)
        elif isinstance(dtype, DatetimeTZDtype):
            dt_cls = dtype.construct_array_type()
            dt64_values = arr.view(f"M8[{dtype.unit}]")
            return dt_cls._simple_new(dt64_values, dtype=dtype)
        elif lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
            from pandas.core.arrays import DatetimeArray

            dt64_values = arr.view(dtype)
            return DatetimeArray._simple_new(dt64_values, dtype=dtype)
        elif lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
            from pandas.core.arrays import TimedeltaArray

            td64_values = arr.view(dtype)
            return TimedeltaArray._simple_new(td64_values, dtype=dtype)
        # error: Argument "dtype" to "view" of "ndarray" has incompatible type
        # "ExtensionDtype | dtype[Any]"; expected "dtype[Any] | _HasDType[dtype[Any]]"
        return arr.view(dtype=dtype)  # type: ignore[arg-type]

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
        axis: AxisInt = 0,
    ) -> Self:
        # [FUNCTION DOC]
        # Take elements by position along an axis. Parameters: indices are the integer
        # positions; allow_fill (keyword, default false) enables treating -1 as missing
        # instead of the last element; fill_value (default None) is validated and
        # converted to storage form (only when allow_fill is true) and used for filled
        # positions; axis (default 0) is the axis to take along. Returns a new array of
        # the same type built from the taken backing data.
        if allow_fill:
            fill_value = self._validate_scalar(fill_value)

        new_data = take(
            self._ndarray,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value,
            axis=axis,
        )
        return self._from_backing_data(new_data)

    # ------------------------------------------------------------------------

    def equals(self, other) -> bool:
        # [FUNCTION DOC]
        # Whether another object is an equal array. Returns false unless the other
        # object
        # has exactly the same type and an equal dtype; otherwise compares the backing
        # ndarrays treating missing values in the same location as equal. Returns a
        # bool.
        if type(self) is not type(other):
            return False
        if self.dtype != other.dtype:
            return False
        return bool(array_equivalent(self._ndarray, other._ndarray, dtype_equal=True))

    @classmethod
    def _from_factorized(cls, values, original):
        # [FUNCTION DOC]
        # HELPER: class-level hook used by factorization. Parameters: values is the
        # unique-values array as produced by the factorize protocol (must have the same
        # dtype as the original's backing array; an assertion fails otherwise) and
        # original is the array that was factorized. Returns a new array of the
        # original's
        # type wrapping values.
        assert values.dtype == original._ndarray.dtype
        return original._from_backing_data(values)

    def _values_for_argsort(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: returns the backing ndarray itself (no copy) as the values used for
        # sorting.
        return self._ndarray

    def _values_for_factorize(self):
        # [FUNCTION DOC]
        # HELPER: returns a pair of the backing ndarray and the internal fill value, as
        # required by the generic factorize machinery.
        return self._ndarray, self._internal_fill_value

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        # [FUNCTION DOC]
        # HELPER: hash support for hashing of pandas objects. Keyword parameters:
        # encoding (string encoding for object values), hash_key (string key for the
        # hash) and categorize (whether to factorize object values first). Returns an
        # unsigned 64-bit integer array of hashes computed over the backing ndarray with
        # the generic array hashing function (imported lazily).
        from pandas.core.util.hashing import hash_array

        values = self._ndarray
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    def _cast_pointwise_result(self, values: ArrayLike) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: converts the result of an element-by-element mapping (a sequence of
        # values) into the best array type. The values are first turned into an object
        # ndarray and then inferred, including non-numeric types such as datetimes,
        # using
        # the library's object conversion. Returns the converted array.
        values = np.asarray(values, dtype=object)
        return lib.maybe_convert_objects(values, convert_non_numeric=True)

    # Signature of "argmin" incompatible with supertype "ExtensionArray"
    def argmin(self, axis: AxisInt = 0, skipna: bool = True):  # type: ignore[override]
        # [FUNCTION DOC]
        # Position of the smallest element. Parameters: axis (default 0) and skipna
        # (default true, validated as a boolean). Overrides the base signature by adding
        # the axis keyword. Raises a value error when skipna is false and the array has
        # missing values. Returns the position (or positions along the axis) computed by
        # the shared sorting helper.
        # override base class by adding axis keyword
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return nargminmax(self, "argmin", axis=axis)

    # Signature of "argmax" incompatible with supertype "ExtensionArray"
    def argmax(self, axis: AxisInt = 0, skipna: bool = True):  # type: ignore[override]
        # [FUNCTION DOC]
        # Position of the largest element. Parameters: axis (default 0) and skipna
        # (default true, validated as a boolean). Overrides the base signature by adding
        # the axis keyword. Raises a value error when skipna is false and the array has
        # missing values. Returns the position (or positions along the axis) computed by
        # the shared sorting helper.
        # override base class by adding axis keyword
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return nargminmax(self, "argmax", axis=axis)

    def unique(self) -> Self:
        # [FUNCTION DOC]
        # Unique values of the backing data in order of appearance. Returns a new array
        # of
        # the same type.
        new_data = unique(self._ndarray)
        return self._from_backing_data(new_data)

    @classmethod
    @doc(ExtensionArray._concat_same_type)
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level concatenation of arrays of this type. Parameters:
        # to_concat
        # is a sequence of arrays; axis (default 0) is the concatenation axis. Raises a
        # value error (including the set of dtype names) if the dtypes are not all
        # equal;
        # otherwise delegates to the base class implementation. Returns the combined
        # array. Docstring is inherited from the base class through a decorator.
        if not lib.dtypes_all_equal([x.dtype for x in to_concat]):
            dtypes = {str(x.dtype) for x in to_concat}
            raise ValueError("to_concat must have the same dtype", dtypes)

        return super()._concat_same_type(to_concat, axis=axis)

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Find insertion positions for values in this sorted array. Parameters: value is
        # a
        # scalar or array-like; side is left (default) or right; sorter is an optional
        # index array that sorts this array. The value is validated and converted to
        # storage form (raising if not castable), then the backing ndarray's search is
        # used. Returns an integer position or array of positions. Docstring is
        # inherited
        # from the base class through a decorator.
        npvalue = self._validate_setitem_value(value)
        return self._ndarray.searchsorted(npvalue, side=side, sorter=sorter)

    @doc(ExtensionArray.shift)
    def shift(self, periods: int = 1, fill_value=None) -> Self:
        # [FUNCTION DOC]
        # Shift values along the first axis by a number of periods. Parameters: periods
        # (default 1; negative shifts backward) and fill_value (default None, validated
        # and converted to storage form) used for vacated positions. Always shifts along
        # axis 0. Returns a new array of the same type. Docstring is inherited from the
        # base class through a decorator.
        # NB: shift is always along axis=0
        axis = 0
        fill_value = self._validate_scalar(fill_value)
        new_values = shift(self._ndarray, periods, axis, fill_value)

        return self._from_backing_data(new_values)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Set values in place. Parameters: key is any supported indexer (validated and
        # normalized as an array indexer) and value is the value or values to assign
        # (validated and converted to storage form). Raises a value error if the array
        # is
        # read-only. Mutates the backing ndarray; returns nothing.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        key = check_array_indexer(self, key)
        value = self._validate_setitem_value(value)
        self._ndarray[key] = value

    def _validate_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: hook that validates and converts a value (scalar or array-like) before
        # assignment into the backing array. The default accepts anything and returns it
        # unchanged; subclasses override to raise on incompatible values.
        return value

    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any:
        # [FUNCTION DOC]
        # Typing-only overload stub: scalar key returns a single element. See the
        # implementation below.
        ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | PositionalIndexerTuple,
    ) -> Self:
        # [FUNCTION DOC]
        # Typing-only overload stub: sequence or tuple key returns a new array.
        # See the implementation below.
        ...

    def __getitem__(
        self,
        key: PositionalIndexer2D,
    ) -> Self | Any:
        # [FUNCTION DOC]
        # Get an element or subset. Parameter key is an integer, slice, boolean or
        # integer array, or (for 2-dimensional arrays) a tuple. Integer keys are a fast
        # path: on one-dimensional data the raw element is boxed to a scalar; on
        # higher-dimensional data the selected row is wrapped as an array. Other keys
        # are
        # unwrapped from extension arrays, validated as array indexers, and applied to
        # the backing ndarray; a scalar result is boxed, otherwise a new array is
        # returned. When the indexing produces a view of the data, the result inherits
        # this array's read-only flag.
        if lib.is_integer(key):
            # fast-path
            result = self._ndarray[key]
            if self.ndim == 1:
                return self._box_func(result)
            result = self._from_backing_data(result)
            if getitem_returns_view(self, key):
                result._readonly = self._readonly
            return result

        # error: Incompatible types in assignment (expression has type "ExtensionArray",
        # variable has type "Union[int, slice, ndarray]")
        key = extract_array(key, extract_numpy=True)  # type: ignore[assignment]
        key = check_array_indexer(self, key)
        result = self._ndarray[key]
        if lib.is_scalar(result):
            return self._box_func(result)

        result = self._from_backing_data(result)
        if getitem_returns_view(self, key):
            result._readonly = self._readonly
        return result

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: implements forward or backward filling of missing values. Keyword
        # parameters: method names the fill direction; limit is the maximum number of
        # consecutive fills (default no limit); limit_area restricts filling to inside
        # or
        # outside of valid values (default unrestricted); copy (default true) controls
        # whether the result is a new array or the filling is done in place on this
        # array's data. Fills along axis 0 for 2-dimensional data. If there are no
        # missing values it returns a copy or self depending on copy. Returns an array
        # of the same type.
        mask = self.isna()
        if mask.any():
            # (for now) when self.ndim == 2, we assume axis=0
            func = missing.get_fill_func(method, ndim=self.ndim)

            npvalues = self._ndarray.T
            if copy:
                npvalues = npvalues.copy()
            func(npvalues, limit=limit, limit_area=limit_area, mask=mask.T)
            npvalues = npvalues.T

            if copy:
                new_values = self._from_backing_data(npvalues)
            else:
                new_values = self

        elif copy:
            new_values = self.copy()
        else:
            new_values = self
        return new_values

    @doc(ExtensionArray.fillna)
    def fillna(self, value, limit: int | None = None, copy: bool = True) -> Self:
        # [FUNCTION DOC]
        # Fill missing values. Parameters: value is a scalar or array-like of the same
        # length; limit (default None) is the maximum number of missing entries to fill,
        # counted from the start; copy (default true) says whether to return a new
        # array (otherwise a shallow slice of this array is filled, which shares data).
        # Value size is checked against the mask. If there are missing entries after
        # applying the limit, they are assigned the value; otherwise the value is still
        # validated. Returns an array of the same type. Docstring is inherited through a
        # decorator.
        mask = self.isna()
        if limit is not None and limit < len(self):
            # mypy doesn't like that mask can be an EA which need not have `cumsum`
            modify = mask.cumsum() > limit  # type: ignore[union-attr]
            if modify.any():
                # Only copy mask if necessary
                mask = mask.copy()
                mask[modify] = False
        # error: Argument 2 to "check_value_size" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        value = missing.check_value_size(
            value,
            mask,  # type: ignore[arg-type]
            len(self),
        )

        if mask.any():
            # fill with value
            if copy:
                new_values = self.copy()
            else:
                new_values = self[:]
            new_values[mask] = value
        else:
            # We validate the fill_value even if there is nothing to fill
            self._validate_setitem_value(value)

            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values

    # ------------------------------------------------------------------------
    # Reductions

    def _wrap_reduction_result(self, axis: AxisInt | None, result) -> Any:
        # [FUNCTION DOC]
        # HELPER: wraps the raw result of a reduction. Parameters: axis (None or an int)
        # and result. If the reduction is over all axes or the array is one-dimensional,
        # the result is boxed as a scalar; otherwise it is wrapped in a new array of the
        # same type.
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    # ------------------------------------------------------------------------
    # __array_function__ methods

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # [FUNCTION DOC]
        # HELPER: in-place analogue of the numpy masked assignment function. Parameters:
        # mask (boolean ndarray) and value (scalar or list-like), which is validated and
        # converted to storage form (raising a type error if it cannot be cast). Mutates
        # the backing ndarray; returns nothing.
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        value = self._validate_setitem_value(value)

        np.putmask(self._ndarray, mask, value)

    def _where(self: Self, mask: npt.NDArray[np.bool_], value) -> Self:
        # [FUNCTION DOC]
        # HELPER: analogue of the numpy conditional-select function. Parameters: mask
        # (boolean ndarray, true where this array's values are kept) and value (scalar
        # or
        # list-like used elsewhere, validated and converted). Returns a new array of the
        # same type. Raises a type error if the value cannot be cast and an assertion
        # error if the combined result dtype differs from the backing dtype.
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        value = self._validate_setitem_value(value)

        res_values = np.where(mask, self._ndarray, value)
        if res_values.dtype != self._ndarray.dtype:
            raise AssertionError(
                # GH#56410
                "Something has gone wrong, please report a bug at "
                "github.com/pandas-dev/pandas/"
            )
        return self._from_backing_data(res_values)

    # ------------------------------------------------------------------------
    # Index compat methods

    def insert(self, loc: int, item) -> Self:
        # [FUNCTION DOC]
        # Return a new array with an item inserted. Parameters: loc is the integer
        # position (validated against the length, negative values count from the end);
        # item is the value, converted via the scalar validation hook. Returns a new
        # array
        # of the same type with the backing data concatenated around the new element.
        """
        Make new ExtensionArray inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        type(self)
        """
        loc = validate_insert_loc(loc, len(self))

        code = self._validate_scalar(item)

        new_vals = np.concatenate(
            (
                self._ndarray[:loc],
                np.asarray([code], dtype=self._ndarray.dtype),
                self._ndarray[loc:],
            )
        )
        return self._from_backing_data(new_vals)

    # ------------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts of unique values, as a Series. Parameter dropna (default true) excludes
        # missing values. Not supported for arrays with more than one dimension (raises
        # a
        # not-implemented error). Counts are computed on the backing ndarray without
        # sorting; the result's index is rewrapped as this array's type. Returns a
        # Series
        # indexed by the unique values.
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NA values.

        Returns
        -------
        Series
        """
        if self.ndim != 1:
            raise NotImplementedError

        from pandas import (
            Index,
            Series,
        )

        if dropna:
            # error: Unsupported operand type for ~ ("ExtensionArray")
            values = self[~self.isna()]._ndarray  # type: ignore[operator]
        else:
            values = self._ndarray

        result = value_counts(values, sort=False, dropna=dropna)

        index_arr = self._from_backing_data(np.asarray(result.index._data))
        index = Index(index_arr, name=result.index.name, copy=False)
        return Series(result._values, index=index, name=result.name, copy=False)

    def _quantile(
        self,
        qs: npt.NDArray[np.float64],
        interpolation: str,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: quantile calculation. Parameters: qs is a float array of quantiles in
        # the unit interval; interpolation is the interpolation method name. The missing
        # mask and internal fill value are passed to the masked quantile helper. If the
        # result keeps the backing dtype it is wrapped through the backing-data
        # constructor; otherwise (for instance an empty integer array yielding floats) a
        # new instance is built by calling the class constructor directly.
        # KNOWN ISSUE: the constructor is not defined on this base class, as noted by
        # the
        # author's TODO and a type-ignore; this path only works for subclasses whose
        # constructor accepts a values array.
        # TODO: disable for Categorical if not ordered?

        mask = np.asarray(self.isna())
        arr = self._ndarray
        fill_value = self._internal_fill_value

        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        if res_values.dtype == self._ndarray.dtype:
            return self._from_backing_data(res_values)
        else:
            # e.g. test_quantile_empty we are empty integer dtype and res_values
            #  has floating dtype
            # TODO: technically __init__ isn't defined here.
            #  Should we raise NotImplementedError and handle this on NumpyEA?
            return type(self)(res_values)  # type: ignore[call-arg]

    # ------------------------------------------------------------------------
    # numpy-like methods

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level analogue of the numpy uninitialized-array constructor.
        # Parameters: shape (tuple of ints) and dtype (an extension dtype). Builds an
        # empty array of the dtype to discover the backing ndarray dtype, then returns a
        # new array wrapping an uninitialized ndarray of the requested shape.
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : ExtensionDtype
        """
        # The base implementation uses a naive approach to find the dtype
        #  for the backing ndarray
        arr = cls._from_sequence([], dtype=dtype)
        backing = np.empty(shape, dtype=arr._ndarray.dtype)
        return arr._from_backing_data(backing)
