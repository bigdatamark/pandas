# [MODULE DOC]
# Implementation of the nullable boolean extension type: the dtype class, the
# array class and a coercion helper. Public contents: the boolean dtype (exposed
# in the pandas namespace) and the boolean array (exposed in the arrays
# namespace); the coercion function is used by other masked-array modules and
# the array factory. Builds on the masked-array base module and on the Kleene
# logic routines in the ops module. Import-time side effect: the dtype class is
# registered in the extension dtype registry.
from __future__ import annotations

import numbers
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Self,
    cast,
)

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.util._decorators import set_module

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna

from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    import pyarrow

    from pandas._typing import (
        DtypeObj,
        npt,
        type_t,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype


@register_extension_dtype
@set_module("pandas")
class BooleanDtype(BaseMaskedDtype):
    # [CLASS DOC]
    # Extension dtype for nullable boolean data, registered under the name boolean and
    # exposed publicly from the pandas namespace. Inherits the masked-dtype base class.
    # Class state: the name, and the internal fill value false used to fill the data
    # array under missing positions so that no upcasting occurs. Reports its scalar type
    # as the numpy boolean scalar, kind code b, underlying numpy dtype bool, and is
    # treated as both boolean and numeric. Its array type is the boolean array class.
    # Can build arrays from arrow data.
    """
    Extension dtype for boolean data.

    This is a pandas Extension dtype for boolean data with support for
    missing values. BooleanDtype is the dtype companion to :class:`.BooleanArray`,
    which implements Kleene logic (sometimes called three-value logic) for
    logical operations. See :ref:`boolean.kleene` for more.

    .. warning::

        BooleanDtype is considered experimental. The implementation and
        parts of the API may change without warning.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    arrays.BooleanArray : Array of boolean (True/False) data with missing values.
    Int64Dtype : Extension dtype for int64 integer data.
    StringDtype : Extension dtype for string data.

    Examples
    --------
    >>> pd.BooleanDtype()
    BooleanDtype

    >>> pd.array([True, False, None], dtype=pd.BooleanDtype())
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    """

    name: ClassVar[str] = "boolean"

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = False

    # https://github.com/python/mypy/issues/4125
    # error: Signature of "type" incompatible with supertype "BaseMaskedDtype"
    @property
    def type(self) -> type:  # type: ignore[override]
        # [FUNCTION DOC]
        # Property: the scalar type of elements, the numpy boolean scalar class. (A
        # type-ignore is used because the signature differs from the base class.)
        return np.bool_

    @property
    def kind(self) -> str:
        # [FUNCTION DOC]
        # Property: the numpy kind character, always b for boolean.
        return "b"

    @property
    def numpy_dtype(self) -> np.dtype:
        # [FUNCTION DOC]
        # Property: the numpy dtype of the underlying data array, plain bool.
        return np.dtype("bool")

    def construct_array_type(self) -> type_t[BooleanArray]:
        # [FUNCTION DOC]
        # Returns the array class associated with this dtype: the boolean array class.
        # Takes no arguments.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return BooleanArray

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the fixed string name of the dtype, without parentheses.
        return "BooleanDtype"

    @property
    def _is_boolean(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property used by the dtype-inspection utilities; always true.
        return True

    @property
    def _is_numeric(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property used by the dtype-inspection utilities; always true, so this
        # dtype is treated as numeric.
        return True

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BooleanArray:
        # [FUNCTION DOC]
        # Build a boolean array from a pyarrow array or chunked array (hook used by
        # arrow conversion). Parameter array is the arrow data. Raises a type error when
        # its type is neither boolean nor null. A null-typed input gives an all-missing
        # array of the same length. Otherwise each chunk is converted separately: the
        # data bits come from the data buffer and, when the chunk has nulls, the mask is
        # the negation of the validity bits (no nulls gives an all-false mask), each
        # chunk becoming a boolean array. The chunk arrays are concatenated; with no
        # chunks an empty array is returned. Returns the boolean array. Imports pyarrow
        # lazily.
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        if array.type != pyarrow.bool_() and not pyarrow.types.is_null(array.type):
            raise TypeError(f"Expected array of boolean type, got {array.type} instead")

        if isinstance(array, pyarrow.Array):
            chunks = [array]
            length = len(array)
        else:
            # pyarrow.ChunkedArray
            chunks = array.chunks
            length = array.length()

        if pyarrow.types.is_null(array.type):
            mask = np.ones(length, dtype=bool)
            # No need to init data, since all null
            data = np.empty(length, dtype=bool)
            return BooleanArray(data, mask)

        results = []
        for arr in chunks:
            buflist = arr.buffers()
            data = pyarrow.BooleanArray.from_buffers(
                arr.type, len(arr), [None, buflist[1]], offset=arr.offset
            ).to_numpy(zero_copy_only=False)
            if arr.null_count != 0:
                mask = pyarrow.BooleanArray.from_buffers(
                    arr.type, len(arr), [None, buflist[0]], offset=arr.offset
                ).to_numpy(zero_copy_only=False)
                mask = ~mask
            else:
                mask = np.zeros(len(arr), dtype=bool)

            bool_arr = BooleanArray(data, mask)
            results.append(bool_arr)

        if not results:
            return BooleanArray(
                np.array([], dtype=np.bool_), np.array([], dtype=np.bool_)
            )
        else:
            return BooleanArray._concat_same_type(results)


def coerce_to_array(
    values, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC]
    # Coerce input values to a pair of numpy arrays: boolean data and boolean mask (true
    # where missing). Parameters: values is a one-dimensional list-like (boolean array,
    # numeric array, object array or sequence); mask is an optional boolean mask to
    # combine with whatever missingness is found in values (default None); copy (default
    # false) whether to copy inputs when they are already suitable. Returns a tuple of
    # (data, mask). Cases: a boolean array input is reused (raising a value error if a
    # mask is also given, since the mask comes from it; copied when requested). A numpy
    # boolean array is used directly. A numeric numpy array has its missing entries
    # masked and other entries converted to boolean, raising a type error if the
    # conversion is lossy, i.e. values are not 0 or 1. Anything else is converted to an
    # object array and its inferred type must be boolean, empty, or numeric with
    # integer-like values (otherwise a type error is raised); missing entries are
    # masked, the rest converted to boolean, and numeric values must be exactly 0 or 1
    # (else a type error). The final mask: all false if there is none; the detected
    # missingness if no mask was given; otherwise the given mask (converted to boolean
    # array if needed) combined by logical or with the detected missingness. A value
    # error is raised if the data and mask shapes differ.
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    if isinstance(values, BooleanArray):
        if mask is not None:
            raise ValueError("cannot pass mask for BooleanArray input")
        values, mask = values._data, values._mask
        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    mask_values = None
    if isinstance(values, np.ndarray) and values.dtype == np.bool_:
        if copy:
            values = values.copy()
    elif isinstance(values, np.ndarray) and values.dtype.kind in "iufcb":
        mask_values = isna(values)

        values_bool = np.zeros(len(values), dtype=bool)
        values_bool[~mask_values] = values[~mask_values].astype(bool)

        if not np.all(
            values_bool[~mask_values].astype(values.dtype) == values[~mask_values]
        ):
            raise TypeError("Need to pass bool-like values")

        values = values_bool
    else:
        values_object = np.asarray(values, dtype=object)

        inferred_dtype = lib.infer_dtype(values_object, skipna=True)
        integer_like = ("floating", "integer", "mixed-integer-float")
        if inferred_dtype not in ("boolean", "empty", *integer_like):
            raise TypeError("Need to pass bool-like values")

        # mypy does not narrow the type of mask_values to npt.NDArray[np.bool_]
        # within this branch, it assumes it can also be None
        mask_values = cast("npt.NDArray[np.bool_]", isna(values_object))
        values = np.zeros(len(values), dtype=bool)
        values[~mask_values] = values_object[~mask_values].astype(bool)

        # if the values were integer-like, validate it were actually 0/1's
        if (inferred_dtype in integer_like) and not (
            np.all(
                values[~mask_values].astype(float)
                == values_object[~mask_values].astype(float)
            )
        ):
            raise TypeError("Need to pass bool-like values")

    if mask is None and mask_values is None:
        mask = np.zeros(values.shape, dtype=bool)
    elif mask is None:
        mask = mask_values
    elif isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
        if mask_values is not None:
            mask = mask | mask_values
        elif copy:
            mask = mask.copy()
    else:
        mask = np.array(mask, dtype=bool)
        if mask_values is not None:
            mask = mask | mask_values

    if values.shape != mask.shape:
        raise ValueError("values.shape and mask.shape must match")

    return values, mask


@set_module("pandas.arrays")
class BooleanArray(BaseMaskedArray):
    # [CLASS DOC]
    # Extension array of nullable boolean values, public as an array type in the arrays
    # namespace. Inherits the masked-array base class, which stores the data (numpy
    # boolean array) and mask (numpy boolean array, true meaning missing) as private
    # attributes. The dtype is always the boolean dtype. Logical operators follow Kleene
    # three-valued logic. Class constants: accepted string spellings of true and false
    # for string parsing, and the tuple of scalar types handled directly in arithmetic
    # dispatch. Cumulative min and max are done natively; other accumulations go through
    # integer arrays.
    """
    Array of boolean (True/False) data with missing values.

    This is a pandas Extension array for boolean data, under the hood
    represented by 2 numpy arrays: a boolean array with the data and
    a boolean array with the mask (True indicating missing).

    BooleanArray implements Kleene logic (sometimes called three-value
    logic) for logical operations. See :ref:`boolean.kleene` for more.

    To construct a BooleanArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples
    below).

    .. warning::

       BooleanArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d boolean-dtype array with the data.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values (True
        indicates missing).
    copy : bool, default False
        Whether to copy the `values` and `mask` arrays.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    BooleanArray

    See Also
    --------
    array : Create an array from data with the appropriate dtype.
    BooleanDtype : Extension dtype for boolean data.
    Series : One-dimensional ndarray with axis labels (including time series).
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Examples
    --------
    Create a BooleanArray with :func:`pandas.array`:

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    """

    _TRUE_VALUES = {"True", "TRUE", "true", "1", "1.0"}
    _FALSE_VALUES = {"False", "FALSE", "false", "0", "0.0"}

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level fast constructor used internally. Parameters: values
        # (numpy array) and mask (boolean numpy array), not validated here. Calls the
        # parent constructor and then sets the dtype attribute to the boolean dtype.
        # Returns the new array.
        result = super()._simple_new(values, mask)
        result._dtype = BooleanDtype()
        return result

    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Constructor. Parameters: values is a one-dimensional numpy array of bool
        # dtype; mask is a boolean numpy array of the same shape marking missing values;
        # copy (default false) whether to copy both. Raises a type error, suggesting the
        # array factory function, if values is not a numpy boolean array. Sets the dtype
        # and delegates validation of the mask and storage to the parent constructor.
        if not (isinstance(values, np.ndarray) and values.dtype == np.bool_):
            raise TypeError(
                "values should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        self._dtype = BooleanDtype()
        super().__init__(values, mask, copy=copy)

    @property
    def dtype(self) -> BooleanDtype:
        # [FUNCTION DOC]
        # Property: the boolean dtype instance of this array.
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: list[str],
        *,
        dtype: ExtensionDtype,
        copy: bool = False,
        true_values: list[str] | None = None,
        false_values: list[str] | None = None,
        none_values: list[str] | None = None,
    ) -> BooleanArray:
        # [FUNCTION DOC]
        # HELPER: class-level parser used by file readers. Parameters: strings is a list
        # of strings (missing values allowed); dtype is the target dtype (keyword,
        # required); copy (default false); true_values, false_values and none_values are
        # optional extra strings to treat as true, false and missing (added to the
        # built-in spellings for true and false; none values default to empty). Raises a
        # value error (from the nested mapper) for a non-missing string not in any of
        # the sets. Returns a boolean array built from the mapped values.
        true_values_union = cls._TRUE_VALUES.union(true_values or [])
        false_values_union = cls._FALSE_VALUES.union(false_values or [])

        if none_values is None:
            none_values = []

        def map_string(s) -> bool | None:
            # [FUNCTION DOC]
            # HELPER: nested mapper from one string to true, false or None. Returns true
            # if the string is in the true set, false if in the false set, None if in
            # the none set, otherwise raises a value error saying the string cannot be
            # cast to bool.
            if s in true_values_union:
                return True
            elif s in false_values_union:
                return False
            elif s in none_values:
                return None
            else:
                raise ValueError(f"{s} cannot be cast to bool")

        scalars = np.array(strings, dtype=object)
        mask = isna(scalars)
        scalars[~mask] = list(map(map_string, scalars[~mask]))
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number, bool, np.bool_)

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: class-level hook converting arbitrary input to data and mask arrays.
        # Parameters: value is the input; dtype (keyword) must be falsy or equal to the
        # boolean dtype name (asserted); copy (default false). Returns the pair from the
        # module's coerce function.
        if dtype:
            assert dtype == "boolean"
        return coerce_to_array(value, copy=copy)

    def _logical_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: implements the logical operators with Kleene logic. Parameters: other
        # is a scalar (boolean, missing marker, numpy boolean), a boolean array or a
        # one-dimensional list-like; op is an operator function whose name must be one
        # of or, and, xor or their reflected forms (asserted). List-likes are converted
        # to boolean data and mask; arrays with more than one dimension give the not-
        # implemented sentinel. Raises a type error if a scalar is neither missing nor
        # boolean, and a value error if the lengths differ. Returns a new array of the
        # result computed by the Kleene or, and, or xor routine with data and masks of
        # both operands (so a missing operand does not propagate when the other operand
        # determines the answer).
        assert op.__name__ in {"or_", "ror_", "and_", "rand_", "xor", "rxor"}
        other_is_scalar = lib.is_scalar(other)
        mask = None

        if isinstance(other, BooleanArray):
            other, mask = other._data, other._mask
        elif is_list_like(other):
            other = np.asarray(other, dtype="bool")
            if other.ndim > 1:
                return NotImplemented
            other, mask = coerce_to_array(other, copy=False)
        elif isinstance(other, np.bool_):
            other = other.item()

        if other_is_scalar and other is not libmissing.NA and not lib.is_bool(other):
            raise TypeError(
                "'other' should be pandas.NA or a bool. "
                f"Got {type(other).__name__} instead."
            )

        if not other_is_scalar and len(self) != len(other):
            raise ValueError("Lengths must match")

        if op.__name__ in {"or_", "ror_"}:
            result, mask = ops.kleene_or(self._data, other, self._mask, mask)
        elif op.__name__ in {"and_", "rand_"}:
            result, mask = ops.kleene_and(self._data, other, self._mask, mask)
        else:
            # i.e. xor, rxor
            result, mask = ops.kleene_xor(self._data, other, self._mask, mask)

        # i.e. BooleanArray
        return self._maybe_mask_result(result, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> BaseMaskedArray:
        # [FUNCTION DOC]
        # HELPER: cumulative operations. Parameters: name is the operation name; skipna
        # (keyword, default true); extra keyword arguments passed through. For
        # cumulative minimum and maximum the masked accumulation routine runs on data
        # and mask and a new boolean array is returned. For other operations the data is
        # converted to an integer array with the same mask and the integer array's
        # accumulate is used, returning its result type.
        data = self._data
        mask = self._mask
        if name in ("cummin", "cummax"):
            op = getattr(masked_accumulations, name)
            data, mask = op(data, mask, skipna=skipna, **kwargs)
            return self._simple_new(data, mask)
        else:
            from pandas.core.arrays import IntegerArray

            return IntegerArray(data.astype(int), mask)._accumulate(
                name, skipna=skipna, **kwargs
            )
