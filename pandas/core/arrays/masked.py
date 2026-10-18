# [MODULE DOC]
# Base class for nullable ("masked") extension arrays: arrays that store the
# values in one numpy array and the missingness in a parallel boolean numpy
# array (true means missing). The concrete types (nullable integer, floating
# and boolean arrays, defined in sibling modules, and the numeric base in the
# numeric module) subclass the single class defined here, which implements the
# generic behavior: construction, indexing and assignment, null handling
# (fill, pad/backfill), conversion to numpy and to other dtypes, unary and
# binary operators with missing-value propagation (logical, arithmetic and
# comparison), reductions, accumulations, sorting helpers, unique/factorize,
# value counts, quantiles, mode, hashing, take, copying, concatenation and
# equality. Subclasses supply the dtype, the coercion of inputs to
# (data, mask) pairs, and the result-masking rules for operations.
# Relationships: relies on the operator mixin, the extension array base, the
# masked reduction/accumulation helper modules, the nan-aware reduction module
# and the general ops machinery; arrow-backed arrays and other masked arrays
# are handled for interoperability. No import-time side effects.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import (
    is_nan_na,
    using_python_scalars,
)

from pandas._libs import (
    algos as libalgos,
    lib,
    missing as libmissing,
)
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.errors import AbstractMethodError

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
    maybe_downcast_to_dtype,
)
from pandas.core.dtypes.common import (
    is_bool,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    BaseMaskedDtype,
)
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core import (
    algorithms as algos,
    arraylike,
    missing,
    nanops,
    ops,
)
from pandas.core.algorithms import (
    factorize_array,
    isin,
    map_array,
    mode,
    take,
)
from pandas.core.array_algos import (
    masked_accumulations,
    masked_reductions,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.array_algos.transforms import shift
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
)
from pandas.core.ops import invalid_comparison
from pandas.core.util.hashing import hash_array

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import (
        Iterator,
        Sequence,
    )
    from pandas import Series
    from pandas.core.arrays import BooleanArray
    from pandas._typing import (
        NumpySorter,
        NumpyValueArrayLike,
        ArrayLike,
        AstypeArg,
        AxisInt,
        DtypeObj,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        PositionalIndexer,
        Scalar,
        ScalarIndexer,
        SequenceIndexer,
        Shape,
        npt,
    )
    from pandas._libs.missing import NAType
    from pandas.core.arrays import FloatingArray

from pandas.compat.numpy import function as nv


class BaseMaskedArray(OpsMixin, ExtensionArray):
    # [CLASS DOC]
    # Abstract base for numpy-backed nullable arrays. Inherits the operator
    # mixin (binary operator dunder plumbing) and the extension array base.
    # State: the data array and the mask array, both numpy arrays of equal shape
    # (the mask has boolean dtype and is true where the value is missing; data
    # values at masked positions are arbitrary placeholders and must not be
    # relied upon). Subclasses must provide the dtype property, the coercion
    # classmethod turning input into (data, mask), and operator result wrapping.
    # Invariant: data and mask shapes are identical. The class mostly works on
    # one dimension but supports reshaping/transposing for internal use.
    # Missing value representation comes from the dtype (the NA singleton,
    # or NaN for float types when configured so).
    # our underlying data and mask are each ndarrays
    """
    Base class for masked arrays (which use _data and _mask to store the data).

    numpy based
    """

    _data: np.ndarray
    _mask: npt.NDArray[np.bool_]

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        # [FUNCTION DOC]
        # HELPER: fast internal constructor that stores the given data and mask
        # arrays directly with no validation or copying. Returns the new
        # instance of the class.
        result = BaseMaskedArray.__new__(cls)
        result._data = values
        result._mask = mask
        return result

    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Constructor. values is the data numpy array (assumed already validated
        # by the subclass constructor); mask is a boolean numpy array of the same
        # shape (true means missing); copy (default false) copies both.
        # Raises a type error if mask is not a boolean numpy array, and a value
        # error if the shapes differ. Stores data and mask.
        # values is supposed to already be validated in the subclass
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_):
            raise TypeError(
                "mask should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.shape != mask.shape:
            raise ValueError("values.shape must match mask.shape")

        if copy:
            values = values.copy()
            mask = mask.copy()

        self._data = values
        self._mask = mask

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method constructor used by the generic array factory.
        # scalars is any sequence; keyword-only dtype (default none) and copy
        # (default false) go to the subclass coercion routine, which returns the
        # (data, mask) pair used to build the new array. Errors from coercion
        # propagate.
        values, mask = cls._coerce_to_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    def _cast_pointwise_result(self, values) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: post-processes the sequence of results of an element-wise
        # mapping. If all results are missing, builds an array of this dtype.
        # Otherwise infers a nullable type from the object values; when both the
        # original and the inferred kinds are integer-like, or both float, the
        # data is losslessly downcast to this array's numpy dtype and re-wrapped
        # with the inferred mask. Returns the resulting array.
        if isna(values).all():
            return type(self)._from_sequence(values, dtype=self.dtype)
        values = np.asarray(values, dtype=object)
        result = lib.maybe_convert_objects(values, convert_to_nullable_dtype=True)
        lkind = self.dtype.kind
        rkind = result.dtype.kind
        if (lkind in "iu" and rkind in "iu") or (lkind == rkind == "f"):
            result = cast(BaseMaskedArray, result)
            new_data = maybe_downcast_to_dtype(
                result._data, dtype=self.dtype.numpy_dtype
            )
            result = type(result)(new_data, result._mask)
        return result

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method creating an all-missing array. shape is the
        # desired shape; dtype is the masked dtype. The data is filled with the
        # dtype's internal fill value and the mask is all true. Raises a
        # not-implemented error if the constructed array's dtype does not equal
        # the requested one. Returns the array.
        """
        Create an ExtensionArray with the given shape and dtype.

        See also
        --------
        ExtensionDtype.empty
            ExtensionDtype.empty is the 'official' public version of this API.
        """
        dtype = cast(BaseMaskedDtype, dtype)
        values: np.ndarray = np.empty(shape, dtype=dtype.type)
        values.fill(dtype._internal_fill_value)
        mask = np.ones(shape, dtype=bool)
        result = cls(values, mask)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        return result

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # [FUNCTION DOC]
        # HELPER: returns the per-element display function: the plain text
        # conversion (to avoid the new numpy scalar repr). boxed is ignored.
        # NEP 51: https://github.com/numpy/numpy/pull/22449
        return str

    @property
    def dtype(self) -> BaseMaskedDtype:
        # [FUNCTION DOC]
        # Abstract property; subclasses return their masked dtype. The base
        # implementation always raises an abstract-method error.
        raise AbstractMethodError(self)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        # [FUNCTION DOC]
        # Typing stub only: scalar position gives a scalar. See implementation.
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self:
        # [FUNCTION DOC]
        # Typing stub only: sequence/slice/mask key gives an array. See
        # implementation.
        ...

    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        # [FUNCTION DOC]
        # Positional indexing. item is an integer, slice, boolean mask or integer
        # array (validated). The mask is indexed first: if the result is a scalar
        # boolean, the dtype's missing value is returned for a missing entry,
        # otherwise the data element. For array results, a new array of the
        # indexed data and mask is returned and, if the indexing returned views,
        # the read-only flag is inherited.
        item = check_array_indexer(self, item)

        newmask = self._mask[item]
        if is_bool(newmask):
            # This is a scalar indexing
            if newmask:
                return self.dtype.na_value
            return self._data[item]

        result = self._simple_new(self._data[item], newmask)
        if getitem_returns_view(self, item):
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
        # HELPER: forward/backward fill hook. Keyword-only: method (fill
        # direction name), limit (default none), limit_area (inside, outside or
        # none; restricts which runs of missing values may be filled), copy
        # (default true).
        # If nothing is missing returns a copy or self. Otherwise a fill routine
        # for the method operates on the transposed data and mask (on copies when
        # copy is true, in place otherwise), propagating values and updating the
        # mask. When limit_area is given and not everything is missing, the
        # leading and trailing missing runs (inside) or the interior missing runs
        # (outside) are restored as missing. Returns a new array if copy is true
        # else self (mutated).
        mask = self._mask

        if mask.any():
            func = missing.get_fill_func(method, ndim=self.ndim)

            npvalues = self._data.T
            new_mask = mask.T
            if copy:
                npvalues = npvalues.copy()
                new_mask = new_mask.copy()
            elif limit_area is not None:
                mask = mask.copy()
            func(npvalues, limit=limit, mask=new_mask)

            if limit_area is not None and not mask.all():
                mask = mask.T
                neg_mask = ~mask
                first = neg_mask.argmax()
                last = len(neg_mask) - neg_mask[::-1].argmax() - 1
                if limit_area == "inside":
                    new_mask[:first] |= mask[:first]
                    new_mask[last + 1 :] |= mask[last + 1 :]
                elif limit_area == "outside":
                    new_mask[first + 1 : last] |= mask[first + 1 : last]

            if copy:
                return self._simple_new(npvalues.T, new_mask.T)
            else:
                return self
        elif copy:
            new_values = self.copy()
        else:
            new_values = self
        return new_values

    def fillna(self, value, limit: int | None = None, copy: bool = True) -> Self:
        # [FUNCTION DOC]
        # Fills missing entries. value is a scalar or an array-like of the same
        # length as the array (size-validated); limit (default none) caps the
        # number of entries filled (counting from the start); copy (default true)
        # selects a copy versus a view that is modified in place.
        # Only the first limit missing positions are filled. Assignment uses the
        # array's own element assignment with the mask, so value type errors from
        # there apply. Returns the filled array (a copy, or a full-slice view of
        # this array when copy is false and something was filled).
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It's expected
            that the array-like have the same length as 'self'.
        limit : int, default None
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        See Also
        --------
        api.extensions.ExtensionArray.dropna : Return ExtensionArray without
            NA values.
        api.extensions.ExtensionArray.isna : A 1-D array indicating if
            each value is missing.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        """
        mask = self._mask
        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit
            if modify.any():
                # Only copy mask if necessary
                mask = mask.copy()
                mask[modify] = False

        value = missing.check_value_size(value, mask, len(self))

        if mask.any():
            # fill with value
            if copy:
                new_values = self.copy()
            else:
                new_values = self[:]
            new_values[mask] = value
        elif copy:
            new_values = self.copy()
        else:
            new_values = self[:]
        return new_values

    @classmethod
    def _coerce_to_array(
        cls, values, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: abstract class method. Subclasses convert values (any input)
        # with a keyword-only dtype and copy flag (default false) to a (data,
        # mask) pair of numpy arrays. The base implementation raises an
        # abstract-method error.
        raise AbstractMethodError(cls)

    def _validate_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: checks that a non-missing scalar can be stored losslessly.
        # Boolean arrays accept boolean values; float arrays accept integers and
        # floats; other (integer) arrays accept integers and integral floats.
        # Anything else raises a type error naming the value and dtype. Returns
        # the value unchanged.
        # KNOWN ISSUE: a source comment notes that range checks for unsigned
        # integer types are missing, so values out of range for the concrete
        # integer width are only rejected by the later numpy assignment.
        """
        Check if we have a scalar that we can cast losslessly.

        Raises
        ------
        TypeError
        """
        kind = self.dtype.kind
        # TODO: get this all from np_can_hold_element?
        if kind == "b":
            if lib.is_bool(value):
                return value

        elif kind == "f":
            if lib.is_integer(value) or lib.is_float(value):
                return value

        elif lib.is_integer(value) or (lib.is_float(value) and value.is_integer()):
            return value
            # TODO: unsigned checks

        # Note: without the "str" here, the f-string rendering raises in
        #  py38 builds.
        raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # In-place assignment. key is an integer, slice, boolean mask or
        # position array; value is a scalar or array-like.
        # Raises a value error if read-only; type errors from scalar validation.
        # A scalar that is a valid missing marker for the dtype marks the mask
        # true (a float NaN is only treated as missing when the NaN-is-NA option
        # is on); other scalars are validated, written to the data and the mask
        # cleared. Non-scalars are coerced to (data, mask) with this dtype and
        # both are written. Mutates this array.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        key = check_array_indexer(self, key)

        if is_scalar(value):
            if is_valid_na_for_dtype(value, self.dtype) and not (
                lib.is_float(value) and not is_nan_na()
            ):
                self._mask[key] = True
            else:
                value = self._validate_setitem_value(value)
                self._data[key] = value
                self._mask[key] = False
            return

        value, mask = self._coerce_to_array(value, dtype=self.dtype)

        self._data[key] = value
        self._mask[key] = mask

    def __contains__(self, key) -> bool:
        # [FUNCTION DOC]
        # Membership test for a single key. A missing-like key that is not the
        # dtype's missing object is normalized: a float NaN becomes the dtype
        # missing value when NaN-is-NA is on, and for float data without that
        # option a float NaN key is found iff the data holds a genuine (unmasked)
        # NaN. Everything else uses the parent membership logic (iteration with
        # equality). Returns a boolean.
        if isna(key) and key is not self.dtype.na_value:
            # GH#52840
            if lib.is_float(key) and is_nan_na():
                key = self.dtype.na_value
            elif self._data.dtype.kind == "f" and lib.is_float(key):
                return bool((np.isnan(self._data) & ~self._mask).any())

        return bool(super().__contains__(key))

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Generator over elements. One-dimensional: yields each data element, with
        # the dtype's missing value in place of masked entries (a fast path
        # yields data directly if nothing is missing). Higher dimensions: yields
        # the sub-array at each leading position.
        if self.ndim == 1:
            if not self._hasna:
                for val in self._data:
                    yield val
            else:
                na_value = self.dtype.na_value
                for isna_, val in zip(self._mask, self._data, strict=True):
                    if isna_:
                        yield na_value
                    else:
                        yield val
        else:
            for i in range(len(self)):
                yield self[i]

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Length of the first axis of the data array.
        return len(self._data)

    @property
    def shape(self) -> Shape:
        # [FUNCTION DOC]
        # Property: the shape of the data array (shared with the mask).
        return self._data.shape

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of dimensions of the data array.
        return self._data.ndim

    def swapaxes(self, axis1, axis2) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with the two given axes interchanged in both data
        # and mask (views, no copy).
        data = self._data.swapaxes(axis1, axis2)
        mask = self._mask.swapaxes(axis1, axis2)
        return self._simple_new(data, mask)

    def delete(self, loc, axis: AxisInt = 0) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with the entries at loc (integer, sequence of
        # integers or slice) removed along axis (default zero) from both data and
        # mask. This array is unchanged.
        data = np.delete(self._data, loc, axis=axis)
        mask = np.delete(self._mask, loc, axis=axis)
        return self._simple_new(data, mask)

    def reshape(self, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Returns a new array whose data and mask are reshaped with the given
        # numpy reshape arguments (views when possible).
        data = self._data.reshape(*args, **kwargs)
        mask = self._mask.reshape(*args, **kwargs)
        return self._simple_new(data, mask)

    def ravel(self, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Returns a flattened array of the same class; data and mask are
        # flattened with the same numpy arguments (e.g. memory order). The source
        # notes the order of data and mask should be verified to agree.
        # TODO: need to make sure we have the same order for data/mask
        data = self._data.ravel(*args, **kwargs)
        mask = self._mask.ravel(*args, **kwargs)
        return type(self)(data, mask)

    def shift(self, periods: int = 1, fill_value=None) -> Self:
        # [FUNCTION DOC]
        # Shifts entries along the first axis by periods (default one; positive
        # moves toward the end). fill_value (default none) fills vacated slots; a
        # none fill value fills with missing entries (data zero, mask true), any
        # other value is written to data with mask false. The fill value is not
        # type-validated here. Returns a new array of the same class.
        # NB: shift is always along axis=0
        axis = 0
        if fill_value is None:
            new_data = shift(self._data, periods, axis, 0)
            new_mask = shift(self._mask, periods, axis, True)
        else:
            new_data = shift(self._data, periods, axis, fill_value)
            new_mask = shift(self._mask, periods, axis, False)
        return type(self)(new_data, new_mask)

    @property
    def T(self) -> Self:
        # [FUNCTION DOC]
        # Property: transposed view of data and mask as a new array.
        return self._simple_new(self._data.T, self._mask.T)

    def round(self, decimals: int = 0, *args, **kwargs):
        # [FUNCTION DOC]
        # Rounds values. decimals (default zero) may be negative; extra
        # positional and keyword arguments are validated for numpy compatibility
        # (invalid ones raise) and extra keywords are forwarded to numpy rounding.
        # Boolean arrays are returned unchanged (self). Otherwise the data is
        # rounded and wrapped with a copy of the mask using the subclass hook,
        # which may change the array type (e.g. integer to integer or boolean to
        # float). Returns the new array.
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        if self.dtype.kind == "b":
            return self
        nv.validate_round(args, kwargs)
        values = np.round(self._data, decimals=decimals, **kwargs)

        # Usually we'll get same type as self, but ndarray[bool] casts to float
        return self._maybe_mask_result(values, self._mask.copy())

    # ------------------------------------------------------------------
    # Unary Methods

    def __invert__(self) -> Self:
        # [FUNCTION DOC]
        # Unary inversion: bitwise/logical not of the data with a copy of the
        # mask (missing stays missing). Returns a new array.
        return self._simple_new(~self._data, self._mask.copy())

    def __neg__(self) -> Self:
        # [FUNCTION DOC]
        # Unary negation of the data with a copy of the mask. Returns a new array.
        return self._simple_new(-self._data, self._mask.copy())

    def __pos__(self) -> Self:
        # [FUNCTION DOC]
        # Unary plus: returns a copy of the array.
        return self.copy()

    def __abs__(self) -> Self:
        # [FUNCTION DOC]
        # Absolute value of the data with a copy of the mask. Returns a new array.
        return self._simple_new(abs(self._data), self._mask.copy())

    # ------------------------------------------------------------------

    def _values_for_json(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: values handed to the JSON writer. Returns an object ndarray of
        # the elements with missing entries as the dtype's missing value.
        return np.asarray(self, dtype=object)

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Converts to a numpy array. dtype (default none: inferred; object when
        # not inferable) is the target numpy dtype; copy (default false) forces a
        # copy when no missing values are present; na_value (default: not
        # specified) is the value that replaces missing entries.
        # The dtype and fill value are decided by the shared inference helper. If
        # there are missing entries, the data is cast to a new array and the
        # missing positions are set to the fill value; a non-object, non-string
        # target with the NA singleton as fill raises a value error. Otherwise
        # the data is cast (copying if requested); a read-only array without copy
        # whose cast is a view is returned as a non-writeable view. Runtime
        # warnings from the cast are silenced. Returns the numpy array.
        """
        Convert to a NumPy Array.

        By default converts to an object-dtype NumPy array. Specify the `dtype` and
        `na_value` keywords to customize the conversion.

        Parameters
        ----------
        dtype : dtype, default object
            The numpy dtype to convert to.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array (pd.NA).

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        An object-dtype is the default result

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a.to_numpy()
        array([True, False, <NA>], dtype=object)

        When no missing values are present, an equivalent dtype can be used.

        >>> pd.array([True, False], dtype="boolean").to_numpy(dtype="bool")
        array([ True, False])
        >>> pd.array([1, 2], dtype="Int64").to_numpy("int64")
        array([1, 2])

        However, requesting such dtype will raise a ValueError if
        missing values are present and the default missing value :attr:`NA`
        is used.

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a
        <BooleanArray>
        [True, False, <NA>]
        Length: 3, dtype: boolean

        >>> a.to_numpy(dtype="bool")
        Traceback (most recent call last):
        ...
        ValueError: cannot convert to bool numpy array in presence of missing values

        Specify a valid `na_value` instead

        >>> a.to_numpy(dtype="bool", na_value=False)
        array([ True, False, False])
        """
        hasna = self._hasna
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, hasna)
        if dtype is None:
            dtype = np.dtype(object)

        if hasna:
            if (
                dtype != np.dtype(object)
                and not is_string_dtype(dtype)
                and na_value is libmissing.NA
            ):
                raise ValueError(
                    f"cannot convert to '{dtype}'-dtype NumPy array "
                    "with missing values. Specify an appropriate 'na_value' "
                    "for this dtype."
                )
            # don't pass copy to astype -> always need a copy since we are mutating
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                data = self._data.astype(dtype)
            data[self._mask] = na_value
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                data = self._data.astype(dtype, copy=copy)
            if self._readonly and not copy and astype_is_view(self.dtype, dtype):
                data = data.view()
                data.flags.writeable = False
        return data

    def tolist(self) -> list:
        # [FUNCTION DOC]
        # Returns a plain list of Python scalars (nested for more than one
        # dimension). Missing entries appear as the NA singleton: when any
        # missing entries exist the array is converted to object first, otherwise
        # to its own numpy dtype, then each element is converted to a Python
        # scalar.
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list
            Python list of values in array.

        See Also
        --------
        Index.to_list: Return a list of the values in the Index.
        Series.to_list: Return a list of the values in the Series.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]
        dtype = None if self._hasna else self._data.dtype
        return self.to_numpy(dtype=dtype, na_value=libmissing.NA).tolist()

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        # [FUNCTION DOC]
        # Typing stub only: numpy dtype target gives an ndarray. See
        # implementation.
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        # [FUNCTION DOC]
        # Typing stub only: extension dtype target gives an extension array. See
        # implementation.
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        # [FUNCTION DOC]
        # Typing stub only: generic dtype argument gives an array-like. See
        # implementation.
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        # [FUNCTION DOC]
        # Casts to another dtype. dtype is anything resolvable to a dtype; copy
        # (default true) forces copying even when not required.
        # Same dtype: returns a copy or self. Another masked dtype: the data is
        # cast to the target numpy dtype (rounding or truncating as numpy does;
        # runtime warnings silenced) and wrapped with the mask, which is copied
        # only if the data was copied. Another extension dtype: delegated to its
        # sequence constructor with this array as input. Otherwise the target is
        # a numpy dtype: float targets use NaN as the fill value, datetime
        # targets use not-a-time, and other targets use no explicit fill value.
        # Integer or boolean targets with any missing entries raise a value error
        # (cannot convert NA to integer / NaN to bool). The conversion itself is
        # done by the numpy conversion method with that fill value and the copy
        # flag; the result is a numpy array.
        # KNOWN ISSUE: a source TODO notes NaN handling for floating targets is
        # not dealt with in the masked-to-masked fast path.
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        # if we are astyping to another nullable masked dtype, we can fastpath
        if isinstance(dtype, BaseMaskedDtype):
            # TODO deal with NaNs for FloatingArray case
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # TODO: Is rounding what we want long term?
                data = self._data.astype(dtype.numpy_dtype, copy=copy)
            # mask is copied depending on whether the data was copied, and
            # not directly depending on the `copy` keyword
            mask = self._mask if data is self._data else self._mask.copy()
            cls = dtype.construct_array_type()
            return cls(data, mask, copy=False)

        if isinstance(dtype, ExtensionDtype):
            eacls = dtype.construct_array_type()
            return eacls._from_sequence(self, dtype=dtype, copy=copy)

        na_value: float | np.datetime64 | lib.NoDefault

        # coerce
        if dtype.kind == "f":
            # In astype, we consider dtype=float to also mean na_value=np.nan
            na_value = np.nan
        elif dtype.kind == "M":
            unit = np.datetime_data(dtype)[0]
            na_value = np.datetime64("NaT", unit)  # type: ignore[call-overload]
        else:
            na_value = lib.no_default

        # to_numpy will also raise, but we get somewhat nicer exception messages here
        if dtype.kind in "iu" and self._hasna:
            raise ValueError("cannot convert NA to integer")
        if dtype.kind == "b" and self._hasna:
            # careful: astype_nansafe converts np.nan to True
            raise ValueError("cannot convert float NaN to bool")

        data = self.to_numpy(dtype=dtype, na_value=na_value, copy=copy)
        return data

    __array_priority__ = 1000  # higher than ndarray so ops dispatch to us

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Numpy array protocol. dtype is the requested dtype (none lets the
        # numpy-conversion method infer it); copy is none, true or false.
        # With copy false: if no entries are missing returns the data array
        # directly (as a non-writeable view for read-only arrays), otherwise
        # raises a value error since a copy is unavoidable. Otherwise (copy true
        # or none, none being treated as false) converts through the numpy
        # conversion method, so missing entries force an object array unless
        # the dtype inference says otherwise.
        # KNOWN ISSUE: the docstring says an object array is always returned to
        # preserve scalars, but the dtype is inferred and can be a numeric dtype
        # when there are no missing values.
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
        if copy is False:
            if not self._hasna:
                # special case, here we can simply return the underlying data
                result = np.array(self._data, dtype=dtype, copy=copy)
                # If the ExtensionArray is readonly, make the numpy array readonly too
                if self._readonly:
                    result = result.view()
                    result.flags.writeable = False
                return result
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )

        if copy is None:
            copy = False  # The NumPy copy=False meaning is different here.
        return self.to_numpy(dtype=dtype, copy=copy)

    _HANDLED_TYPES: tuple[type, ...]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy universal-function protocol. ufunc is the function, method the
        # call style (call, reduce, at, ...), inputs the operands, kwargs the
        # options (possibly output arrays).
        # Returns the not-implemented marker if any operand (or output) is not a
        # masked array nor one of the subclass's handled scalar/array types.
        # Binary operators are first dispatched to the equivalent dunder methods;
        # calls with outputs and reductions have dedicated dispatchers. Otherwise
        # the union of the operands' masks is built, the ufunc runs on the raw
        # data, and results are rebuilt: multiple outputs give a tuple; a reduce
        # returns the missing value if any element is missing, else the plain
        # result; other results are re-wrapped by dtype kind (boolean, integer,
        # float arrays; any other kind is returned as numpy with masked slots set
        # to NaN). Float results that are 16-bit are widened to 32-bit, and NaN
        # results are added to the mask when NaN counts as NA.
        # For MaskedArray inputs, we apply the ufunc to ._data
        # and mask the result.

        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, (*self._HANDLED_TYPES, BaseMaskedArray)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_ufunc_with_out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, BaseMaskedArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x: np.ndarray):
            # [FUNCTION DOC]
            # HELPER: nested function in the ufunc protocol. x is a raw numpy
            # result of the ufunc. Wraps it by dtype kind into a boolean, integer
            # or float masked array using a copy of the combined mask (for floats
            # widening 16-bit data and masking NaN when NaN counts as NA); for
            # other kinds, sets masked slots to NaN in place and returns the plain
            # numpy array.
            # we don't worry about scalar `x` here, since we
            # raise for reduce up above.
            from pandas.core.arrays import (
                BooleanArray,
                FloatingArray,
                IntegerArray,
            )

            if x.dtype.kind == "b":
                m = mask.copy()
                return BooleanArray(x, m)
            elif x.dtype.kind in "iu":
                m = mask.copy()
                return IntegerArray(x, m)
            elif x.dtype.kind == "f":
                m = mask.copy()
                if x.dtype == np.float16:
                    # reached in e.g. np.sqrt on BooleanArray
                    # we don't support float16
                    x = x.astype(np.float32)
                if is_nan_na():
                    m[np.isnan(x)] = True
                return FloatingArray(x, m)
            else:
                x[mask] = np.nan
            return x

        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if ufunc.nout > 1:
            # e.g. np.divmod
            return tuple(reconstruct(x) for x in result)
        elif method == "reduce":
            # e.g. np.add.reduce; test_ufunc_reduce_raises
            if self._mask.any():
                return self._na_value
            return result
        else:
            return reconstruct(result)

    def __arrow_array__(self, type=None):
        # [FUNCTION DOC]
        # Arrow export protocol (pyarrow imported lazily). type (default none)
        # is the requested arrow type. Builds an arrow array from the data with
        # the mask marking nulls and returns it.
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        return pa.array(self._data, mask=self._mask, type=type)

    @property
    def _hasna(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property telling whether any entry is missing. Scans the whole
        # mask each time (not cached), so it is linear in the length.
        # Note: this is expensive right now! The hope is that we can
        # make this faster by having an optional mask, but not have to change
        # source code using it..

        return bool(self._mask.any())

    def _propagate_mask(
        self, mask: npt.NDArray[np.bool_] | None, other
    ) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # HELPER: computes the result mask of a binary operation. mask is the
        # other operand's mask or none; other is the other operand itself. If mask
        # is given, returns the element-wise union with this array's mask. If none,
        # starts from a copy of this mask; when other is the NA singleton, every
        # position becomes masked (a new array, not in place); when other is a
        # list-like of equal length, positions where other is missing are added.
        # Returns a new boolean array.
        if mask is None:
            mask = self._mask.copy()  # TODO: need test for BooleanArray needing a copy
            if other is libmissing.NA:
                # GH#45421 don't alter inplace
                mask = mask | True
            elif is_list_like(other) and len(other) == len(mask):
                mask = mask | isna(other)
        else:
            mask = self._mask | mask
        return mask

    def _arith_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: implementation of arithmetic operators, also bound as the logical
        # operator implementation. other is a scalar, array-like or array (masked
        # arrays donate their own mask; one-dimensional list-likes of equal
        # length are first converted with array inference); op is the operator
        # function.
        # Raises a not-implemented error for multi-dimensional list-likes, and for
        # certain operators (floor division, power, true division) with boolean
        # data and the NA singleton. Returns the not-implemented marker when
        # other is an arrow-backed array. Details: the computation reuses the
        # non-masked array-op machinery (with errors suppressed); integer floor
        # division and modulo use the raw operator to keep historical behavior;
        # NA as other gives an all-masked result of a suitable dtype (float for
        # true division on non-float data, integer 8-bit for boolean modulo, and a
        # pair for divmod). For power, positions where the base is one or the
        # exponent is zero are unmasked since the result is defined (one) even
        # for missing partners. The result is wrapped with the result-masking
        # hook, which chooses the array class from the result dtype.
        # KNOWN ISSUE: a source TODO notes integer floor-division by zero does not
        # match the non-masked behavior.
        op_name = op.__name__
        omask = None

        if (
            not hasattr(other, "dtype")
            and is_list_like(other)
            and len(other) == len(self)
        ):
            # Try inferring masked dtype instead of casting to object
            other = pd_array(other)
            other = extract_array(other, extract_numpy=True)

        if isinstance(other, BaseMaskedArray):
            other, omask = other._data, other._mask

        elif is_list_like(other):
            if not isinstance(other, ExtensionArray):
                other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")

        # We wrap the non-masked arithmetic logic used for numpy dtypes
        #  in Series/Index arithmetic ops.
        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)

        if isinstance(other, ExtensionArray) and isinstance(other.dtype, ArrowDtype):
            # GH#58602
            return NotImplemented

        if op_name in {"pow", "rpow"} and isinstance(other, np.bool_):
            # Avoid DeprecationWarning: In future, it will be an error
            #  for 'np.bool_' scalars to be interpreted as an index
            #  e.g. test_array_scalar_like_equivalence
            other = bool(other)

        mask = self._propagate_mask(omask, other)

        if other is libmissing.NA:
            result = np.ones_like(self._data)
            if self.dtype.kind == "b":
                if op_name in {
                    "floordiv",
                    "rfloordiv",
                    "pow",
                    "rpow",
                    "truediv",
                    "rtruediv",
                }:
                    # GH#41165 Try to match non-masked Series behavior
                    #  This is still imperfect GH#46043
                    raise NotImplementedError(
                        f"operator '{op_name}' not implemented for bool dtypes"
                    )
                if op_name in {"mod", "rmod"}:
                    dtype = "int8"
                else:
                    dtype = "bool"
                result = result.astype(dtype)
            elif "truediv" in op_name and self.dtype.kind != "f":
                # The actual data here doesn't matter since the mask
                #  will be all-True, but since this is division, we want
                #  to end up with floating dtype.
                result = result.astype(np.float64)
            elif op_name in {"divmod", "rdivmod"}:
                # GH#62196
                res = self._maybe_mask_result(result, mask)
                return res, res.copy()
        else:
            # Make sure we do this before the "pow" mask checks
            #  to get an expected exception message on shape mismatch.
            if self.dtype.kind in "iu" and op_name in ["floordiv", "mod"]:
                # TODO(GH#30188) ATM we don't match the behavior of non-masked
                #  types with respect to floordiv-by-zero
                pd_op = op

            with np.errstate(all="ignore"):
                result = pd_op(self._data, other)

        if op_name == "pow":
            # 1 ** x is 1.
            mask = np.where((self._data == 1) & ~self._mask, False, mask)
            # x ** 0 is 1.
            if omask is not None:
                mask = np.where((other == 0) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 0, False, mask)

        elif op_name == "rpow":
            # 1 ** x is 1.
            if omask is not None:
                mask = np.where((other == 1) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 1, False, mask)
            # x ** 0 is 1.
            mask = np.where((self._data == 0) & ~self._mask, False, mask)

        return self._maybe_mask_result(result, mask)

    _logical_method = _arith_method

    def _cmp_method(self, other, op) -> BooleanArray:
        # [FUNCTION DOC]
        # HELPER: implementation of the six comparison operators. other is a
        # scalar, array-like or masked array (donating its mask); op is the
        # operator function. Returns the not-implemented marker for arrow-backed
        # arrays; raises a not-implemented error for multi-dimensional list-likes
        # and a value error for length mismatch. The NA singleton gives an
        # all-false data with an all-true mask. Otherwise the comparison runs on
        # the raw data using the data array's own method (numpy warnings about
        # elementwise comparison are ignored), falling back to the generic
        # invalid-comparison handler if the method returns the not-implemented
        # marker. The result mask is the union of the masks. Returns a nullable
        # boolean array (no copy of the result).
        from pandas.core.arrays import BooleanArray

        mask = None

        if isinstance(other, ExtensionArray) and isinstance(other.dtype, ArrowDtype):
            # GH#58602
            return NotImplemented

        elif isinstance(other, BaseMaskedArray):
            other, mask = other._data, other._mask

        elif is_list_like(other):
            other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        if other is libmissing.NA:
            # numpy does not handle pd.NA well as "other" scalar (it returns
            # a scalar False instead of an array)
            # This may be fixed by NA.__array_ufunc__. Revisit this check
            # once that's implemented.
            result = np.zeros(self._data.shape, dtype="bool")
            mask = np.ones(self._data.shape, dtype="bool")
        else:
            with warnings.catch_warnings():
                # numpy may show a FutureWarning or DeprecationWarning:
                #     elementwise comparison failed; returning scalar instead,
                #     but in the future will perform elementwise comparison
                # before returning NotImplemented. We fall back to the correct
                # behavior today, so that should be fine to ignore.
                warnings.filterwarnings("ignore", "elementwise", FutureWarning)
                warnings.filterwarnings("ignore", "elementwise", DeprecationWarning)
                method = getattr(self._data, f"__{op.__name__}__")
                result = method(other)

                if result is NotImplemented:
                    result = invalid_comparison(self._data, other, op)

        mask = self._propagate_mask(mask, other)
        return BooleanArray(result, mask, copy=False)

    def _maybe_mask_result(
        self, result: np.ndarray | tuple[np.ndarray, np.ndarray], mask: np.ndarray
    ):
        # [FUNCTION DOC]
        # HELPER: wraps a raw operation result with a mask into the right array
        # type. result is a numpy array or a pair of arrays (divmod, wrapped
        # individually); mask is the boolean missing mask.
        # By result dtype kind: float gives a floating masked array (and when NaN
        # counts as NA, NaN positions are set in the passed mask in place, a side
        # effect on the argument); boolean gives a boolean masked array; supported
        # timedelta gives a timedelta array with masked slots set to not-a-time
        # (mutating result); integer gives an integer masked array; object arrays
        # get the dtype's missing value at masked slots; anything else gets NaN at
        # masked slots. The last two return plain numpy arrays.
        """
        Parameters
        ----------
        result : array-like or tuple[array-like]
        mask : array-like bool
        """
        if isinstance(result, tuple):
            # i.e. divmod
            div, mod = result
            return (
                self._maybe_mask_result(div, mask),
                self._maybe_mask_result(mod, mask),
            )

        if result.dtype.kind == "f":
            from pandas.core.arrays import FloatingArray

            if is_nan_na():
                mask[np.isnan(result)] = True

            return FloatingArray(result, mask, copy=False)

        elif result.dtype.kind == "b":
            from pandas.core.arrays import BooleanArray

            return BooleanArray(result, mask, copy=False)

        elif lib.is_np_dtype(result.dtype, "m") and is_supported_dtype(result.dtype):
            # e.g. test_numeric_arr_mul_tdscalar_numexpr_path
            from pandas.core.arrays import TimedeltaArray

            unit = np.datetime_data(result.dtype)[0]
            result[mask] = np.timedelta64("NaT", unit)  # type: ignore[call-overload]

            if not isinstance(result, TimedeltaArray):
                return TimedeltaArray._simple_new(result, dtype=result.dtype)

            return result

        elif result.dtype.kind in "iu":
            from pandas.core.arrays import IntegerArray

            return IntegerArray(result, mask, copy=False)

        elif result.dtype == object:
            result[mask] = self.dtype.na_value
            return result
        else:
            result[mask] = np.nan
            return result

    def isna(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns a copy of the mask as a boolean array (true where missing).
        return self._mask.copy()

    @property
    def _na_value(self):
        # [FUNCTION DOC]
        # HELPER: property returning the dtype's missing value.
        return self.dtype.na_value

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: total memory in bytes, data plus mask.
        return self._data.nbytes + self._mask.nbytes

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-method used by concatenation. to_concat is a sequence of
        # arrays of the same class; axis (default zero). Data arrays and mask
        # arrays are concatenated separately along the axis. Returns the new array
        # (via the regular constructor, so shape checks apply).
        data = np.concatenate([x._data for x in to_concat], axis=axis)
        mask = np.concatenate([x._mask for x in to_concat], axis=axis)
        return cls(data, mask)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        # [FUNCTION DOC]
        # HELPER: per-element 64-bit hashes used for hashing of rows/objects.
        # Keyword-only: encoding (text encoding for strings), hash_key (the key
        # for the keyed hash), categorize (whether to factorize before hashing).
        # Hashes the data array and overwrites the hashes at masked positions with
        # the hash of the dtype's missing value. Returns an unsigned 64-bit array.
        hashed_array = hash_array(
            self._data, encoding=encoding, hash_key=hash_key, categorize=categorize
        )
        hashed_array[self.isna()] = hash(self.dtype.na_value)
        return hashed_array

    def take(
        self,
        indexer,
        *,
        allow_fill: bool = False,
        fill_value: Scalar | None = None,
        axis: AxisInt = 0,
    ) -> Self:
        # [FUNCTION DOC]
        # Positional take. indexer is an integer sequence; keyword-only
        # allow_fill (default false: negatives count from the end; true: minus one
        # marks positions to fill), fill_value (default none meaning missing) and
        # axis (default zero).
        # Data is taken with an internal placeholder fill (to avoid upcasting) and
        # the mask is taken with true as fill. If filling with a non-missing
        # value, the filled positions get that value and are unmasked (their mask
        # is toggled relative to the taken mask). Errors for out-of-bounds or
        # invalid negative positions come from the generic take. Returns a new
        # array.
        # KNOWN ISSUE: the in-code comment says the fill is always the integer
        # one, but the placeholder is the dtype's internal fill value.
        # we always fill with 1 internally
        # to avoid upcasting
        data_fill_value = (
            self.dtype._internal_fill_value if isna(fill_value) else fill_value
        )
        result = take(
            self._data,
            indexer,
            fill_value=data_fill_value,
            allow_fill=allow_fill,
            axis=axis,
        )

        mask = take(
            self._mask, indexer, fill_value=True, allow_fill=allow_fill, axis=axis
        )

        # if we are filling
        # we only fill where the indexer is null
        # not existing missing values
        # TODO(jreback) what if we have a non-na float as a fill value?
        if allow_fill and notna(fill_value):
            fill_mask = np.asarray(indexer) == -1
            result[fill_mask] = fill_value
            mask = mask ^ fill_mask

        return self._simple_new(result, mask)

    # error: Return type "BooleanArray" of "isin" incompatible with return type
    # "ndarray" in supertype "ExtensionArray"
    def isin(self, values: ArrayLike) -> BooleanArray:  # type: ignore[override]
        # [FUNCTION DOC]
        # Membership test. values is an array-like of candidates. Returns a
        # nullable boolean array of this array's shape with no missing entries:
        # non-missing elements are checked against the candidates; missing
        # elements are true exactly when the candidates contain the dtype's
        # missing object (missing does not propagate).
        from pandas.core.arrays import BooleanArray

        # algorithms.isin will eventually convert values to an ndarray, so no extra
        # cost to doing it here first
        values_arr = np.asarray(values)
        result = isin(self._data, values_arr)

        if self._hasna:
            values_have_NA = values_arr.dtype == object and any(
                val is self.dtype.na_value for val in values_arr
            )

            # For now, NA does not propagate so set result according to presence of NA,
            # see https://github.com/pandas-dev/pandas/pull/38379 for some discussion
            result[self._mask] = values_have_NA

        mask = np.zeros(self._data.shape, dtype=bool)
        return BooleanArray(result, mask, copy=False)

    def copy(self) -> Self:
        # [FUNCTION DOC]
        # Returns a new array with copies of both the data and the mask.
        data = self._data.copy()
        mask = self._mask.copy()
        return self._simple_new(data, mask)

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        # [FUNCTION DOC]
        # HELPER: ranking hook. Keyword-only: axis (default zero), method (tie
        # handling, default average), na_option (default keep; or top/bottom),
        # ascending (default true), pct (default false; return percentile ranks).
        # Raises a not-implemented error unless axis is zero and the array is one-
        # dimensional. Boolean data is viewed as small integers. The compiled
        # ranking routine is run with the missing mask. The result mask is the
        # original mask when missing values are kept, otherwise all false. Returns
        # a nullable unsigned 64-bit integer array when the method is not average
        # and pct is false (masked slots zeroed first), otherwise a nullable
        # floating array.
        # GH#62043 Avoid going through copy-making ensure_data in algorithms.rank
        if axis != 0 or self.ndim != 1:
            raise NotImplementedError

        from pandas.core.arrays import FloatingArray

        data = self._data
        if data.dtype.kind == "b":
            data = data.view("uint8")

        result = libalgos.rank_1d(
            data,
            is_datetimelike=False,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
            mask=self.isna(),
        )
        if na_option in ["top", "bottom"]:
            mask = np.zeros(self.shape, dtype=bool)
        else:
            mask = self._mask.copy()

        if method != "average" and not pct:
            if na_option not in ["top", "bottom"]:
                result[self._mask] = 0  # avoid warning on casting
            result = result.astype("uint64", copy=False)
            from pandas.core.arrays import IntegerArray

            return IntegerArray(result, mask=mask)

        return FloatingArray(result, mask=mask)

    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Marks repeated values. keep (default first) says which occurrence is
        # not marked: first, last, or false meaning all repeats are marked.
        # Missing values are treated as equal to each other. Returns a boolean
        # numpy array computed by the generic duplicated routine on data and mask.
        """
        Return boolean ndarray denoting duplicate values.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        ndarray[bool]
            With true in indices where elements are duplicated and false otherwise.

        See Also
        --------
        DataFrame.duplicated : Return boolean Series denoting
            duplicate rows.
        Series.duplicated : Indicate duplicate Series values.
        api.extensions.ExtensionArray.unique : Compute the ExtensionArray
            of unique values.

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        """
        values = self._data
        mask = self._mask
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        # [FUNCTION DOC]
        # Returns a new array of the distinct values in order of first appearance,
        # with at most one missing entry (the generic unique-with-mask routine
        # returns both the values and the mask).
        """
        Compute the BaseMaskedArray of unique values.

        Returns
        -------
        uniques : BaseMaskedArray
        """
        uniques, mask = algos.unique_with_mask(self._data, self._mask)
        return self._simple_new(uniques, mask)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Binary search for insertion points. value is a scalar or array-like (an
        # extension array is converted to object first); side is left or right
        # (default left); sorter (default none) is an integer array that sorts
        # the data. Raises a value error if this array has any missing entries.
        # The search is done directly on the data array (avoiding object
        # conversion). Returns an integer array, or a single integer for a scalar.
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
        if self._hasna:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        # Base class searchsorted would cast to object, which is *much* slower.
        return self._data.searchsorted(value, side=side, sorter=sorter)

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        # [FUNCTION DOC]
        # Encodes the values as integer codes plus the array of distinct values.
        # use_na_sentinel (default true): missing values get code minus one and are
        # not in the uniques; when false, missing values are given their own code
        # (placed in order of first appearance among the codes, shifting later
        # codes up) and the uniques gain a masked entry at that code.
        # The generic array factorizer runs on data and mask (always with the
        # sentinel); an assertion checks the uniques dtype equals this dtype's
        # numpy dtype. Returns a pair (integer code array, masked array of
        # uniques).
        # KNOWN ISSUE: the docstring example factorizes a period index, not a
        # masked array.
        """
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

        Returns
        -------
        codes : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(
        ...     ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
        ...     freq="M",
        ... )
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex(['2014-01', '2014-02', '2014-03'], dtype='period[M]')
        """
        arr = self._data
        mask = self._mask

        # Use a sentinel for na; recode and add NA to uniques if necessary below
        codes, uniques = factorize_array(arr, use_na_sentinel=True, mask=mask)

        # check that factorize_array correctly preserves dtype.
        assert uniques.dtype == self.dtype.numpy_dtype, (uniques.dtype, self.dtype)

        has_na = mask.any()
        if use_na_sentinel or not has_na:
            size = len(uniques)
        else:
            # Make room for an NA value
            size = len(uniques) + 1
        uniques_mask = np.zeros(size, dtype=bool)
        if not use_na_sentinel and has_na:
            na_index = mask.argmax()
            # Insert na with the proper code
            if na_index == 0:
                na_code = np.intp(0)
            else:
                na_code = codes[:na_index].max() + 1
            codes[codes >= na_code] += 1
            codes[codes == -1] = na_code
            # dummy value for uniques; not used since uniques_mask will be True
            uniques = np.insert(uniques, na_code, 0)
            uniques_mask[na_code] = True
        uniques_ea = self._simple_new(uniques, uniques_mask)

        return codes, uniques_ea

    def _values_for_argsort(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: returns the raw data array (a view, not a copy) as the sort keys;
        # missing slots hold arbitrary placeholders that callers must ignore using
        # the missing mask.
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.

        Notes
        -----
        The caller is responsible for *not* modifying these values in-place, so
        it is safe for implementers to give views on ``self``.

        Functions that use this (e.g. ``ExtensionArray.argsort``) should ignore
        entries with missing values in the original array (according to
        ``self.isna()``). This means that the corresponding entries in the returned
        array don't need to be modified to sort correctly.

        Examples
        --------
        In most cases, this is the underlying Numpy array of the ``ExtensionArray``:

        >>> arr = pd.array([1, 2, 3])
        >>> arr._values_for_argsort()
        array([1, 2, 3])
        """
        return self._data

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts occurrences of each distinct value. dropna (default true) leaves
        # out missing values. The generic routine returns keys, counts and the
        # number of missing entries; when missing entries are counted they are the
        # last key (masked in the index). Returns a series named "count" holding
        # nullable 64-bit integer counts, indexed by an index of this array type
        # built from the keys and mask.
        """
        Returns a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        from pandas import (
            Index,
            Series,
        )
        from pandas.arrays import IntegerArray

        keys, value_counts, na_counter = algos.value_counts_arraylike(
            self._data, dropna=dropna, mask=self._mask
        )
        mask_index = np.zeros((len(value_counts),), dtype=np.bool_)
        mask = mask_index.copy()

        if na_counter > 0:
            mask_index[-1] = True

        arr = IntegerArray(value_counts, mask)
        index = Index(
            self.dtype.construct_array_type()(
                keys,  # type: ignore[arg-type]
                mask_index,
            ),
            copy=False,
        )
        return Series(arr, index=index, name="count", copy=False)

    def _mode(self, dropna: bool = True) -> Self:
        # [FUNCTION DOC]
        # HELPER: most-frequent-values hook. dropna (default true) ignores missing
        # entries when counting (otherwise missing can be a mode). The generic
        # mode routine returns the modal values and mask; they are wrapped in a
        # new array of this class and returned sorted ascending (via the argsort
        # positions).
        result, res_mask = mode(self._data, dropna=dropna, mask=self._mask)
        result = type(self)(result, res_mask)
        return result[result.argsort()]

    def equals(self, other) -> bool:
        # [FUNCTION DOC]
        # Whether other is an equivalent array. Returns false unless other has the
        # same class and dtype and an identical mask (so a missing value of one
        # kind never equals a differently-represented one), then compares the
        # non-missing data with NaN-aware equality. Returns a boolean.
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        See Also
        --------
        numpy.array_equal : Equivalent method for numpy array.
        Series.equals : Equivalent method for Series.
        DataFrame.equals : Equivalent method for DataFrame.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True

        >>> arr1 = pd.array([1, 3, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        False
        """
        if type(self) != type(other):
            return False
        if other.dtype != self.dtype:
            return False

        # GH#44382 if e.g. self[1] is np.nan and other[1] is pd.NA, we are NOT
        #  equal.
        if not np.array_equal(self._mask, other._mask):
            return False

        left = self._data[~self._mask]
        right = other._data[~other._mask]
        return array_equivalent(left, right, strict_nan=True, dtype_equal=True)

    def _quantile(
        self, qs: npt.NDArray[np.float64], interpolation: str
    ) -> BaseMaskedArray:
        # [FUNCTION DOC]
        # HELPER: quantile hook. qs is a float array of quantile positions in the
        # unit interval; interpolation names the interpolation method. The
        # masked-quantile routine computes values with NaN as fill. If the array
        # has missing entries: two-dimensional input raises a not-implemented
        # error; an all-missing array yields an all-masked result (integer dtypes
        # keep an integer zero placeholder); otherwise no result is masked. Without
        # missing entries nothing is masked. The result is wrapped using the
        # result-masking hook (so a floating array for float output). Returns the
        # wrapped quantiles.
        """
        Dispatch to quantile_with_mask, needed because we do not have
        _from_factorized.

        Notes
        -----
        We assume that all impacted cases are 1D-only.
        """
        res = quantile_with_mask(
            self._data,
            mask=self._mask,
            # TODO(GH#40932): na_value_for_dtype(self.dtype.numpy_dtype)
            #  instead of np.nan
            fill_value=np.nan,
            qs=qs,
            interpolation=interpolation,
        )

        if self._hasna:
            # Our result mask is all-False unless we are all-NA, in which
            #  case it is all-True.
            if self.ndim == 2:
                # I think this should be out_mask=self.isna().all(axis=1)
                #  but am holding off until we have tests
                raise NotImplementedError
            if self.isna().all():
                out_mask = np.ones(res.shape, dtype=bool)

                if is_integer_dtype(self.dtype):
                    # We try to maintain int dtype if possible for not all-na case
                    # as well
                    res = np.zeros(res.shape, dtype=self.dtype.numpy_dtype)
            else:
                out_mask = np.zeros(res.shape, dtype=bool)
        else:
            out_mask = np.zeros(res.shape, dtype=bool)
        return self._maybe_mask_result(res, mask=out_mask)

    # ------------------------------------------------------------------
    # Reductions

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # Generic reduction entry point. name is the reduction name; skipna
        # (keyword-only, default true); keepdims (default false) returns a
        # one-element array; extra keywords (such as axis or ddof) go to the
        # method.
        # For the common reductions (any, all, min, max, sum, prod, mean, var, std)
        # the same-named method is called. For the others (median, skew, kurt,
        # sem and similar) the nan-aware reduction of that name runs on the data
        # with the mask (axis popped from the keywords, default none); an unknown
        # name fails with an attribute error. With keepdims, a missing result
        # gives a one-element all-masked array of a suitable dtype and a valid
        # result is reshaped to one element and wrapped. Without keepdims, a
        # missing result is returned as the NA singleton, else the plain result.
        if name in {"any", "all", "min", "max", "sum", "prod", "mean", "var", "std"}:
            result = getattr(self, name)(skipna=skipna, **kwargs)
        else:
            # median, skew, kurt, sem
            data = self._data
            mask = self._mask
            op = getattr(nanops, f"nan{name}")
            axis = kwargs.pop("axis", None)
            result = op(data, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if keepdims:
            if isna(result):
                return self._wrap_na_result(name=name, axis=0, mask_size=(1,))
            else:
                if using_python_scalars():
                    result = np.array([result])
                else:
                    result = result.reshape(1)
                mask = np.zeros(1, dtype=bool)
                return self._maybe_mask_result(result, mask)

        if isna(result):
            return libmissing.NA
        else:
            return result

    def _wrap_reduction_result(self, name: str, result, *, skipna, axis):
        # [FUNCTION DOC]
        # HELPER: finalizes a reduction result. name is the reduction name
        # (unused here); result is a scalar or numpy array; skipna and axis are
        # keyword-only. Scalars are returned unchanged. Array results (reductions
        # along an axis) are wrapped with a mask: with skipna, only rows or columns
        # that are entirely missing are masked; otherwise any missing entry masks
        # its row or column.
        if isinstance(result, np.ndarray):
            if skipna:
                # we only retain mask for all-NA rows/columns
                mask = self._mask.all(axis=axis)
            else:
                mask = self._mask.any(axis=axis)

            return self._maybe_mask_result(result, mask)
        return result

    def _wrap_na_result(self, *, name, axis, mask_size):
        # [FUNCTION DOC]
        # HELPER: builds an all-missing result array for a reduction. Keyword-only:
        # name (reduction name), axis (unused), mask_size (shape of the result).
        # The result dtype is chosen to match what a real reduction would produce:
        # float (32-bit for the 32-bit floating dtype, else 64-bit) for mean-like
        # reductions; the array's own numpy dtype for min and max or for 8-byte
        # types; otherwise the platform-width signed integer for boolean and
        # signed kinds, unsigned for unsigned, and float for float kinds (32-bit
        # integers on Windows or 32-bit platforms). Returns a masked array with
        # all entries missing.
        mask = np.ones(mask_size, dtype=bool)

        float_dtyp = "float32" if self.dtype == "Float32" else "float64"
        if name in ["mean", "median", "var", "std", "skew", "kurt", "sem"]:
            np_dtype = float_dtyp
        elif name in ["min", "max"] or self.dtype.itemsize == 8:
            np_dtype = self.dtype.numpy_dtype.name
        else:
            is_windows_or_32bit = is_platform_windows() or not IS64
            int_dtyp = "int32" if is_windows_or_32bit else "int64"
            uint_dtyp = "uint32" if is_windows_or_32bit else "uint64"
            np_dtype = {"b": int_dtyp, "i": int_dtyp, "u": uint_dtyp, "f": float_dtyp}[
                self.dtype.kind
            ]

        value = np.array([1], dtype=np_dtype)
        return self._maybe_mask_result(value, mask=mask)

    def _wrap_min_count_reduction_result(
        self, name: str, result, *, skipna, min_count, axis
    ):
        # [FUNCTION DOC]
        # HELPER: finalizes sum/prod style results. name is the reduction name;
        # result the raw result; skipna, min_count and axis are keyword-only. With
        # no minimum count and an array result, nothing is masked (the identity
        # value is valid for empty groups). Otherwise defers to the general
        # reduction-result wrapper.
        if min_count == 0 and isinstance(result, np.ndarray):
            return self._maybe_mask_result(result, np.zeros(result.shape, dtype=bool))
        return self._wrap_reduction_result(name, result, skipna=skipna, axis=axis)

    def sum(
        self,
        *,
        skipna: bool = True,
        min_count: int = 0,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Sum of the values. Keyword-only: skipna (default true), min_count
        # (default zero; minimum number of valid values needed, else the result is
        # missing), axis (default zero); extra keywords only validated for numpy
        # compatibility. Uses the masked sum routine, then the min-count result
        # wrapper. Returns a scalar (possibly the NA singleton) or, for axis
        # reductions of higher-dimensional data, an array.
        nv.validate_sum((), kwargs)

        result = masked_reductions.sum(
            self._data,
            self._mask,
            skipna=skipna,
            min_count=min_count,
            axis=axis,
        )
        return self._wrap_min_count_reduction_result(
            "sum", result, skipna=skipna, min_count=min_count, axis=axis
        )

    def prod(
        self,
        *,
        skipna: bool = True,
        min_count: int = 0,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Product of the values; same parameters and behavior as the sum (skipna,
        # min_count, axis, validated extra keywords) using the masked product
        # routine.
        nv.validate_prod((), kwargs)

        result = masked_reductions.prod(
            self._data,
            self._mask,
            skipna=skipna,
            min_count=min_count,
            axis=axis,
        )
        return self._wrap_min_count_reduction_result(
            "prod", result, skipna=skipna, min_count=min_count, axis=axis
        )

    def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        # [FUNCTION DOC]
        # Arithmetic mean. Keyword-only skipna (default true) and axis (default
        # zero); extra keywords only validated. Uses the masked mean routine and
        # the reduction result wrapper (missing result for empty or all-missing
        # input).
        nv.validate_mean((), kwargs)
        result = masked_reductions.mean(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        return self._wrap_reduction_result("mean", result, skipna=skipna, axis=axis)

    def var(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs
    ):
        # [FUNCTION DOC]
        # Variance. Keyword-only skipna (default true), axis (default zero), ddof
        # (delta degrees of freedom, default one); extra keywords only validated.
        # Uses the masked variance routine and the reduction result wrapper.
        nv.validate_stat_ddof_func((), kwargs, fname="var")
        result = masked_reductions.var(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
            ddof=ddof,
        )
        return self._wrap_reduction_result("var", result, skipna=skipna, axis=axis)

    def std(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs
    ):
        # [FUNCTION DOC]
        # Standard deviation; same parameters and behavior as the variance
        # (skipna, axis, ddof default one).
        nv.validate_stat_ddof_func((), kwargs, fname="std")
        result = masked_reductions.std(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
            ddof=ddof,
        )
        return self._wrap_reduction_result("std", result, skipna=skipna, axis=axis)

    def min(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        # [FUNCTION DOC]
        # Minimum. Keyword-only skipna (default true) and axis (default zero);
        # extra keywords only validated. Uses the masked minimum routine and the
        # reduction result wrapper.
        nv.validate_min((), kwargs)
        result = masked_reductions.min(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        return self._wrap_reduction_result("min", result, skipna=skipna, axis=axis)

    def max(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        # [FUNCTION DOC]
        # Maximum; same parameters and behavior as the minimum.
        nv.validate_max((), kwargs)
        result = masked_reductions.max(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        return self._wrap_reduction_result("max", result, skipna=skipna, axis=axis)

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # [FUNCTION DOC]
        # Applies a mapping to every element. mapper is a function or mapping
        # (dict-like); na_action (default none) may be "ignore" to leave missing
        # entries unmapped. The array is converted to a numpy array first (object
        # with missing values) and the generic mapping routine returns its result
        # (a numpy or extension array).
        return map_array(self.to_numpy(), mapper, na_action=na_action)

    @overload
    def any(
        self, *, skipna: Literal[True] = ..., axis: AxisInt | None = ..., **kwargs
    ) -> np.bool_:
        # [FUNCTION DOC]
        # Typing stub only: with skipna true the result is a plain boolean. See
        # implementation.
        ...

    @overload
    def any(
        self, *, skipna: bool, axis: AxisInt | None = ..., **kwargs
    ) -> np.bool_ | NAType:
        # [FUNCTION DOC]
        # Typing stub only: with an arbitrary skipna the result may be the NA
        # singleton. See implementation.
        ...

    def any(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs
    ) -> np.bool_ | NAType:
        # [FUNCTION DOC]
        # True if any element is truthy. Keyword-only skipna (default true) and
        # axis (default zero); extra keywords only validated.
        # Missing entries are replaced by the dtype's falsey value in a copy and
        # the data reduced. With skipna, that result is returned. Without skipna,
        # if the result is true, or the array is empty or has no missing entries,
        # the result is returned; otherwise the answer is unknown and the NA
        # singleton is returned (three-valued logic).
        # KNOWN ISSUE: the axis argument is accepted but not used in the
        # reduction (the data is reduced over all elements), unlike the all
        # reduction which passes it on.
        """
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA's present.
        axis : int, optional, default 0
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        numpy.any : Numpy version of this method.
        BaseMaskedArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True]).any()
        np.True_
        >>> pd.array([True, False, pd.NA]).any()
        np.True_
        >>> pd.array([False, False, pd.NA]).any()
        np.False_
        >>> pd.array([], dtype="boolean").any()
        np.False_
        >>> pd.array([pd.NA], dtype="boolean").any()
        np.False_
        >>> pd.array([pd.NA], dtype="Float64").any()
        np.False_

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA]).any(skipna=False)
        np.True_
        >>> pd.array([1, 0, pd.NA]).any(skipna=False)
        np.True_
        >>> pd.array([False, False, pd.NA]).any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA]).any(skipna=False)
        <NA>
        """
        nv.validate_any((), kwargs)

        values = self._data.copy()
        np.putmask(values, self._mask, self.dtype._falsey_value)
        result = values.any()
        if skipna:
            return result
        elif result or len(self) == 0 or not self._mask.any():
            return result
        else:
            return self.dtype.na_value

    @overload
    def all(
        self, *, skipna: Literal[True] = ..., axis: AxisInt | None = ..., **kwargs
    ) -> np.bool_:
        # [FUNCTION DOC]
        # Typing stub only: with skipna true the result is a plain boolean. See
        # implementation.
        ...

    @overload
    def all(
        self, *, skipna: bool, axis: AxisInt | None = ..., **kwargs
    ) -> np.bool_ | NAType:
        # [FUNCTION DOC]
        # Typing stub only: with an arbitrary skipna the result may be the NA
        # singleton. See implementation.
        ...

    def all(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs
    ) -> np.bool_ | NAType:
        # [FUNCTION DOC]
        # True if all elements are truthy. Keyword-only skipna (default true) and
        # axis (default zero, passed to the reduction); extra keywords only
        # validated. Missing entries are replaced by the dtype's truthy value in a
        # copy and the data reduced. With skipna the result is returned. Without
        # skipna, a false result, an empty array, or no missing entries return the
        # result; otherwise the NA singleton (three-valued logic).
        """
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA's present.
        axis : int, optional, default 0
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        numpy.all : Numpy version of this method.
        BooleanArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA]).all()
        np.True_
        >>> pd.array([1, 1, pd.NA]).all()
        np.True_
        >>> pd.array([True, False, pd.NA]).all()
        np.False_
        >>> pd.array([], dtype="boolean").all()
        np.True_
        >>> pd.array([pd.NA], dtype="boolean").all()
        np.True_
        >>> pd.array([pd.NA], dtype="Float64").all()
        np.True_

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA]).all(skipna=False)
        np.False_
        >>> pd.array([1, 0, pd.NA]).all(skipna=False)
        np.False_
        """
        nv.validate_all((), kwargs)

        values = self._data.copy()
        np.putmask(values, self._mask, self.dtype._truthy_value)
        result = values.all(axis=axis)

        if skipna:
            return result  # type: ignore[return-value]
        elif not result or len(self) == 0 or not self._mask.any():
            return result  # type: ignore[return-value]
        else:
            return self.dtype.na_value

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> FloatingArray:
        # [FUNCTION DOC]
        # Fills missing values by interpolation. Keyword-only: method (algorithm
        # name), axis (accepted but zero is always used for the work), index (index
        # labels for index-based methods), limit, limit_direction, limit_area
        # (fill restrictions), copy (work on copies else in place), plus extra
        # keywords forwarded to the algorithm.
        # Float arrays work on copies of data and mask (or in place when copy is
        # false). Integer arrays always work on a float copy of the data (copy
        # forced true) and the result is a floating array. Other kinds (boolean)
        # raise a not-implemented error. The interpolation updates data and mask in
        # place. Returns self if not copying, a same-class array for floats, or a
        # floating array otherwise.
        # KNOWN ISSUE: the axis parameter is ignored; the in-place interpolation
        # is always called with axis zero.
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if self.dtype.kind == "f":
            if copy:
                data = self._data.copy()
                mask = self._mask.copy()
            else:
                data = self._data
                mask = self._mask
        elif self.dtype.kind in "iu":
            copy = True
            data = self._data.astype("f8")
            mask = self._mask.copy()
        else:
            raise NotImplementedError(
                f"interpolate is not implemented for dtype={self.dtype}"
            )

        missing.interpolate_2d_inplace(
            data,
            method=method,
            axis=0,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            mask=mask,
            **kwargs,
        )
        if not copy:
            return self  # type: ignore[return-value]
        if self.dtype.kind == "f":
            return type(self)._simple_new(data, mask)  # type: ignore[return-value]
        else:
            from pandas.core.arrays import FloatingArray

            return FloatingArray._simple_new(data, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> BaseMaskedArray:
        # [FUNCTION DOC]
        # Cumulative operation hook. name is the accumulation name (cumsum, cumprod,
        # cummin or cummax); skipna (keyword-only, default true); extra keywords
        # are forwarded. Calls the same-named routine in the masked-accumulation
        # module with data, mask and skipna, which returns new data and mask arrays.
        # Returns a new array of the same class. An unknown name fails with an
        # attribute error.
        data = self._data
        mask = self._mask

        op = getattr(masked_accumulations, name)
        data, mask = op(data, mask, skipna=skipna, **kwargs)

        return self._simple_new(data, mask)

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: group-by hook. Keyword-only: how (operation name such as sum,
        # mean, rank, ohlc, idxmin), has_dropped_na (whether rows with missing
        # keys were dropped), min_count (minimum valid values per group), ngroups
        # (number of groups), ids (integer array mapping each row to its group
        # number), plus extra keywords for the operation (e.g. rank options).
        # The operation kind (aggregate versus transform) is derived from how and
        # the compiled grouped routine is run on the data with the mask. The
        # compiled code must not change the input mask. Result mask: for transforms
        # a copy of the input mask; for aggregations all false of length ngroups;
        # for rank with top or bottom placement, all false. For the open-high-low-
        # close operation the mask is tiled across its multiple output columns.
        # Index-returning operations (idxmin, idxmax) return the plain integer
        # array; everything else is wrapped with the result-masking hook.
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        # libgroupby functions are responsible for NOT altering mask
        mask = self._mask
        if op.kind != "aggregate":
            result_mask = mask.copy()
        else:
            result_mask = np.zeros(ngroups, dtype=bool)

        if how == "rank" and kwargs.get("na_option") in ["top", "bottom"]:
            result_mask[:] = False

        res_values = op._cython_op_ndim_compat(
            self._data,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

        if op.how == "ohlc":
            arity = op._cython_arity.get(op.how, 1)
            result_mask = np.tile(result_mask, (arity, 1)).T

        if op.how in ["idxmin", "idxmax"]:
            # Result values are indexes to take, keep as ndarray
            return res_values
        else:
            # res_values should already have the correct dtype, we just need to
            #  wrap in a MaskedArray
            return self._maybe_mask_result(res_values, result_mask)


def transpose_homogeneous_masked_arrays(
    masked_arrays: Sequence[BaseMaskedArray],
) -> list[BaseMaskedArray]:
    # [FUNCTION DOC]
    # Transposes a collection of masked arrays viewed as the columns of a matrix.
    # masked_arrays is a sequence of equal-length one-dimensional masked arrays
    # sharing one dtype (not validated; an empty sequence fails with an index
    # error). All data are stacked into one column-major matrix and all masks
    # into one boolean matrix of the same layout; then each matrix column
    # (originally a row across the inputs) becomes a new masked array of the
    # same class. Returns the list of new arrays, one per original position; they
    # are views of the shared matrices.
    """Transpose masked arrays in a list, but faster.

    Input should be a list of 1-dim masked arrays of equal length and all have the
    same dtype. The caller is responsible for ensuring validity of input data.
    """
    masked_arrays = list(masked_arrays)
    dtype = masked_arrays[0].dtype

    values = [arr._data.reshape(1, -1) for arr in masked_arrays]
    transposed_values = np.concatenate(
        values,
        axis=0,
        out=np.empty(
            (len(masked_arrays), len(masked_arrays[0])),
            order="F",
            dtype=dtype.numpy_dtype,
        ),
    )

    masks = [arr._mask.reshape(1, -1) for arr in masked_arrays]
    transposed_masks = np.concatenate(
        masks, axis=0, out=np.empty_like(transposed_values, dtype=bool)
    )

    arr_type = dtype.construct_array_type()
    transposed_arrays: list[BaseMaskedArray] = []
    for i in range(transposed_values.shape[1]):
        transposed_arr = arr_type(transposed_values[:, i], mask=transposed_masks[:, i])
        transposed_arrays.append(transposed_arr)

    return transposed_arrays
