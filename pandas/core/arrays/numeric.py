# [MODULE DOC]
# Shared base classes for the nullable numeric extension arrays (integer and
# floating). Public-ish contents: the numeric dtype base class (a masked dtype
# with signedness properties, arrow conversion and dtype standardization), the
# numeric array base class (a masked array that validates its numpy payload and
# resolves its dtype through the dtype class mapping), and a private coercion
# function that turns arbitrary input into a (data, mask) pair.
# Relationships: subclasses live in the integer and floating modules, which
# provide the dtype mapping, the checker, default numpy dtype, fill value and
# the safe-cast implementation; everything here builds on the masked module.
# No import-time side effects.
from __future__ import annotations

import numbers
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
)

import numpy as np

from pandas._config import is_nan_na

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_string_dtype,
    pandas_dtype,
)

from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

    import pyarrow

    from pandas._typing import (
        DtypeObj,
        npt,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype


class NumericDtype(BaseMaskedDtype):
    # [CLASS DOC]
    # Abstract base for the nullable integer and floating dtypes. Inherits the
    # masked dtype behavior. Subclasses must define two class-level attributes:
    # the default numpy dtype (used for inference when no dtype is given) and a
    # checker callable that tells whether an arbitrary dtype belongs to the
    # family. Subclasses must also implement the dtype mapping (numpy dtype to
    # extension dtype) and the safe cast hook, which raise an abstract-method
    # error here. Key methods: the representation, signedness properties, the
    # arrow conversion, and dtype standardization.
    _default_np_dtype: np.dtype
    _checker: Callable[[Any], bool]  # is_foo_dtype

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the text made of the dtype name followed by the word Dtype and
        # an empty pair of parentheses, as in the name Int64 giving Int64Dtype
        # with the parentheses.
        return f"{self.name}Dtype()"

    @cache_readonly
    def is_signed_integer(self) -> bool:
        # [FUNCTION DOC]
        # Cached property. Returns true when the numpy kind character of this
        # dtype is the signed integer kind, otherwise false.
        return self.kind == "i"

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        # [FUNCTION DOC]
        # Cached property. Returns true when the numpy kind character of this
        # dtype is the unsigned integer kind, otherwise false.
        return self.kind == "u"

    @property
    def _is_numeric(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: internal flag read by generic dtype machinery. Always true
        # for numeric dtypes.
        return True

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray:
        # [FUNCTION DOC]
        # Builds the matching nullable array (via the construct-array-type
        # hook) from a pyarrow array or chunked array. pyarrow is imported
        # lazily.
        # Behavior: if the arrow type differs from the arrow equivalent of this
        # dtype and is not the null type, the equivalent pandas dtype of the
        # arrow type must have kind integer, unsigned or float, else a type
        # error is raised; the array is then cast to the target arrow type
        # (so signed/unsigned and width changes are allowed). Chunked input is
        # combined into one chunk. The data and validity mask are extracted as
        # numpy; the extracted mask marks valid entries, so for float data when
        # NaN is treated as NA (global option) the valid flag is cleared at NaN
        # positions, making NaN count as missing. The result wraps a copy of
        # the data and the inverted mask (True meaning missing) without
        # copying again.
        # Returns the new masked array; raises a type error for incompatible
        # arrow types.
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        array_class = self.construct_array_type()

        pyarrow_type = pyarrow.from_numpy_dtype(self.type)
        if not array.type.equals(pyarrow_type) and not pyarrow.types.is_null(
            array.type
        ):
            # test_from_arrow_type_error raise for string, but allow
            #  through itemsize conversion GH#31896
            rt_dtype = pandas_dtype(array.type.to_pandas_dtype())
            if rt_dtype.kind not in "iuf":
                # Could allow "c" or potentially disallow float<->int conversion,
                #  but at the moment we specifically test that uint<->int works
                raise TypeError(
                    f"Expected array of {self} type, got {array.type} instead"
                )

            array = array.cast(pyarrow_type)

        if isinstance(array, pyarrow.ChunkedArray):
            array = array.combine_chunks()

        data, mask = pyarrow_array_to_numpy_and_mask(array, dtype=self.numpy_dtype)
        if data.dtype.kind == "f" and is_nan_na():
            mask[np.isnan(data)] = False
        return array_class(data.copy(), ~mask, copy=False)

    @classmethod
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:
        # [FUNCTION DOC]
        # HELPER: abstract class method; subclasses return the mapping from
        # numpy dtype to the corresponding extension dtype instance. The base
        # implementation always raises an abstract-method error.
        raise AbstractMethodError(cls)

    @classmethod
    def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
        # [FUNCTION DOC]
        # HELPER: normalizes a dtype specification to an extension dtype
        # instance. Input is an extension dtype (returned unchanged), a string
        # or a numpy dtype. Strings starting with the capitalized family names
        # (signed, unsigned, float) are lower-cased first to avoid a numpy
        # deprecation warning; then the numpy dtype built from it is looked up
        # in the subclass mapping. Raises a value error when it is not found
        # (the lookup key error is converted).
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
        if isinstance(dtype, str) and (dtype.startswith(("Int", "UInt", "Float"))):
            # Avoid DeprecationWarning from NumPy about np.dtype("Int64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not isinstance(dtype, NumericDtype):
            mapping = cls._get_dtype_mapping()
            try:
                dtype = mapping[np.dtype(dtype)]
            except KeyError as err:
                raise ValueError(f"invalid dtype specified {dtype}") from err
        return dtype

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: abstract class method; subclasses cast the values array to
        # the given numpy dtype, honoring the copy flag, and must raise if the
        # cast would lose information. The base implementation always raises an
        # abstract-method error.
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        raise AbstractMethodError(cls)


def _coerce_to_data_and_mask(values, dtype, copy: bool, dtype_cls: type[NumericDtype]):
    # [FUNCTION DOC]
    # HELPER: converts arbitrary input into the (numpy data, boolean mask)
    # pair backing a nullable numeric array; used by the numeric array
    # coercion hook.
    # Inputs: values is any array-like; dtype is None or a dtype spec (when
    # None and the input has a dtype accepted by the family checker, that dtype
    # is used); copy asks for a copy; dtype_cls is the concrete dtype class
    # providing the checker, default numpy dtype, fill value, standardizer and
    # safe cast.
    # Output: tuple of data array and mask (true means missing).
    # Behavior: an input that is already the matching masked array is unpacked
    # (data cast if a dtype is given, both copied if requested) and returned.
    # Otherwise the input becomes a numpy array. Object or string arrays have
    # their inferred type checked: a boolean object array without an explicit
    # dtype is rejected. Boolean arrays with a family target dtype are cast to
    # the default dtype with an all-false mask. Other non-numeric kinds are
    # rejected. The array must be 1D. The mask is computed when absent: none for
    # ints; for floats NaN positions when NaN counts as NA, otherwise none (and
    # with an integer target a NaN raises); for other kinds, the numeric-NA
    # detection helper. Float input with an integer target may lose precision,
    # in which case the original values are re-cast (or kept as object) so the
    # safe cast can detect it. Masked positions are filled with the family's
    # internal fill value in a copy. Strings are cast directly; everything
    # else goes through the safe cast hook.
    # Raises a type error for unsupported input kinds, boolean object input,
    # non-1D data or mask; a value error when NaN is cast to an integer dtype
    # while NaN is not treated as NA; errors from string parsing and from the
    # safe cast propagate.
    # KNOWN ISSUE: the inferred type variable is initialized to none twice (the
    # second assignment is redundant), harmless dead code.
    checker = dtype_cls._checker
    default_dtype = dtype_cls._default_np_dtype

    mask = None
    inferred_type = None

    if dtype is None and hasattr(values, "dtype"):
        if checker(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        dtype = dtype_cls._standardize_dtype(dtype)

    cls = dtype_cls().construct_array_type()
    if isinstance(values, cls):
        values, mask = values._data, values._mask
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    original = values
    if not copy:
        values = np.asarray(values)
    else:
        values = np.array(values, copy=copy)
    inferred_type = None
    if values.dtype == object or is_string_dtype(values.dtype):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == "boolean" and dtype is None:
            # object dtype array of bools
            name = dtype_cls.__name__.strip("_")
            raise TypeError(f"{values.dtype} cannot be converted to {name}")

    elif values.dtype.kind == "b" and checker(dtype):
        # fastpath
        mask = np.zeros(len(values), dtype=np.bool_)
        if not copy:
            values = np.asarray(values, dtype=default_dtype)
        else:
            values = np.array(values, dtype=default_dtype, copy=copy)

    elif values.dtype.kind not in "iuf":
        name = dtype_cls.__name__.strip("_")
        raise TypeError(f"{values.dtype} cannot be converted to {name}")

    if values.ndim != 1:
        raise TypeError("values must be a 1D list-like")

    if mask is None:
        if values.dtype.kind in "iu":
            # fastpath
            mask = np.zeros(len(values), dtype=np.bool_)
        elif values.dtype.kind == "f":
            # np.isnan is faster than is_numeric_na() for floats
            # github issue: #60066
            if is_nan_na():
                mask = np.isnan(values)
            else:
                mask = np.zeros(len(values), dtype=np.bool_)
                if dtype_cls.__name__.strip("_").startswith(("I", "U")):
                    wrong = np.isnan(values)
                    if wrong.any():
                        raise ValueError("Cannot cast NaN value to Integer dtype.")
        elif is_nan_na():
            mask = libmissing.is_numeric_na(values)
        else:
            # is_numeric_na will raise on non-numeric NAs
            libmissing.is_numeric_na(values)
            mask = libmissing.is_pdna_or_none(values)
    else:
        assert len(mask) == len(values)

    if mask.ndim != 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = default_dtype
    else:
        dtype = dtype.numpy_dtype

    if is_integer_dtype(dtype) and values.dtype.kind == "f" and len(values) > 0:
        if mask.all():
            values = np.ones(values.shape, dtype=dtype)
        else:
            idx = np.nanargmax(values)
            if int(values[idx]) != original[idx]:
                # We have ints that lost precision during the cast.
                inferred_type = lib.infer_dtype(original, skipna=True)
                if (
                    inferred_type not in ["floating", "mixed-integer-float"]
                    and not mask.any()
                ):
                    values = np.asarray(original, dtype=dtype)
                else:
                    values = np.asarray(original, dtype="object")

    # we copy as need to coerce here
    if mask.any():
        values = values.copy()
        values[mask] = dtype_cls._internal_fill_value
    if inferred_type in ("string", "unicode"):
        # casts from str are always safe since they raise
        # a ValueError if the str cannot be parsed into a float
        values = values.astype(dtype, copy=copy)
    else:
        values = dtype_cls._safe_cast(values, dtype, copy=False)
    return values, mask


class NumericArray(BaseMaskedArray):
    # [CLASS DOC]
    # Abstract base for the nullable integer and floating arrays; inherits all
    # masked-array behavior (data array plus boolean mask where true means
    # missing). Subclasses set the class attribute holding their dtype class,
    # which supplies the checker and the dtype mapping. Adds constructor
    # validation, the dtype property, coercion hooks, and a tuple of scalar
    # types handled by arithmetic (arrays and numbers).
    """
    Base class for IntegerArray and FloatingArray.
    """

    _dtype_cls: type[NumericDtype]

    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Validates and stores the data and mask.
        # Inputs: values must be a numpy array whose dtype passes the family
        # checker; mask is a boolean array of the same length (true means
        # missing); copy asks the parent constructor to copy both.
        # Raises a type error when values is not a matching numpy array (the
        # message mentions floating or integer by the family kind and suggests
        # the general array factory) and when the dtype is 16-bit float.
        # The caller is responsible for the mask covering NaNs when NaN counts
        # as NA. Delegates to the masked-array constructor.
        # KNOWN ISSUE: the 16-bit float error message always says that the
        # floating array does not support it, even though it is raised from the
        # shared base class; harmless for floats since only floats can reach it.
        checker = self._dtype_cls._checker
        if not (isinstance(values, np.ndarray) and checker(values.dtype)):
            descr = (
                "floating"
                if self._dtype_cls.kind == "f"  # type: ignore[comparison-overlap]
                else "integer"
            )
            raise TypeError(
                f"values should be {descr} numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.dtype == np.float16:
            # If we don't raise here, then accessing self.dtype would raise
            raise TypeError("FloatingArray does not support np.float16 dtype.")

        # NB: if is_nan_na() is True
        #  then caller is responsible for ensuring
        #  assert mask[np.isnan(values)].all()

        super().__init__(values, mask, copy=copy)

    @cache_readonly
    def dtype(self) -> NumericDtype:
        # [FUNCTION DOC]
        # Cached property. Returns the extension dtype instance matching the
        # numpy dtype of the stored data, looked up in the dtype-class mapping.
        mapping = self._dtype_cls._get_dtype_mapping()
        return mapping[self._data.dtype]

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: class method used by the masked-array construction paths.
        # Takes any array-like value, a keyword-only dtype (may be none despite
        # the annotation) and a keyword-only copy flag (default false) and
        # returns the (data, mask) tuple produced by the module's coercion
        # function using this class's dtype class.
        dtype_cls = cls._dtype_cls
        values, mask = _coerce_to_data_and_mask(value, dtype, copy, dtype_cls)
        return values, mask

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class method building an array from a sequence of strings
        # (used by readers/parsers). Strings are parsed with the numeric
        # conversion function in raising mode, producing nullable numpy-backed
        # values, which are then passed to the standard sequence constructor
        # with the given keyword-only dtype and copy flag (default false).
        # Parse failures propagate as errors.
        from pandas.core.tools.numeric import to_numeric

        scalars = to_numeric(strings, errors="raise", dtype_backend="numpy_nullable")
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)
