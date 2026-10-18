# [MODULE DOC]
# Arrow-backed string extension array. Public contents: ArrowStringArray
# (published in the arrays namespace), which stores string data in a pyarrow
# large-string array (possibly chunked) and implements the string-method
# interface, comparison, reductions and conversions on top of the generic
# Arrow extension array, the Arrow string-method mixin, the object-based string
# fallback mixin and the base string array. Private helpers check that pyarrow
# is installed and detect string-view types.
# Two missing-value semantics are supported, selected by the dtype's NA value:
# the NA scalar (nullable results of boolean/integer extension types) or NaN
# (plain numpy results, False/NaN propagation).
# pyarrow is optional: this module imports it only when available, and the
# array constructor raises an import error when it is missing. No other
# import-time side effects.
from __future__ import annotations

import operator
import re
from typing import (
    TYPE_CHECKING,
    Self,
)

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.compat import (
    HAS_PYARROW,
    PYARROW_MIN_VERSION,
    pa_version_under16p0,
)
from pandas.util._decorators import set_module
from pandas.util._validators import validate_na_arg

from pandas.core.dtypes.common import (
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import isna

from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringDtype,
)
from pandas.core.strings.object_array import ObjectStringArrayMixin

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.compute as pc


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        Dtype,
        NpDtype,
        Scalar,
        npt,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import Series


def _check_pyarrow_available() -> None:
    # [FUNCTION DOC]
    # HELPER: guard for operations that need pyarrow. Raises an import error
    # stating the minimum required pyarrow version when pyarrow is not
    # installed; returns nothing otherwise.
    if not HAS_PYARROW:
        msg = (
            f"pyarrow>={PYARROW_MIN_VERSION} is required for PyArrow "
            "backed ArrowExtensionArray."
        )
        raise ImportError(msg)


def _is_string_view(typ):
    # [FUNCTION DOC]
    # HELPER: True when typ is pyarrow's string-view type; always False when the
    # installed pyarrow is older than version 16, which lacks that type.
    return not pa_version_under16p0 and pa.types.is_string_view(typ)


# TODO: Inherit directly from BaseStringArrayMethods. Currently we inherit from
# ObjectStringArrayMixin because we want to have the object-dtype based methods as
# fallback for the ones that pyarrow doesn't yet support


@set_module("pandas.arrays")
class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    # [CLASS DOC]
    # Extension array for string data held in a pyarrow large-string array
    # (subclass of the object-based string method mixin, the Arrow extension
    # array and the base string array; the object-based mixin is a fallback for
    # methods pyarrow does not support). State: the pyarrow data (inherited) and
    # _dtype, the string dtype instance, whose NA value (NA scalar or NaN)
    # drives result types. Invariant: the stored pyarrow type is always large
    # string. Methods wrap the Arrow-mixin implementations for most string
    # operations and fall back to the object implementation for flags,
    # compiled-pattern flags, callables and unsupported regex features.
    """
    Extension array for string data in a ``pyarrow.ChunkedArray``.

    .. warning::

       ArrowStringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array of data.
    dtype : StringDtype
        The dtype for the array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`array`
        The recommended function for creating an ArrowStringArray.
    Series.str
        The string methods are available on Series backed by
        an ArrowStringArray.

    Notes
    -----
    ArrowStringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(["This is", "some text", None, "data."], dtype="string[pyarrow]")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string
    """

    # error: Incompatible types in assignment (expression has type "StringDtype",
    # base class "ArrowExtensionArray" defined the type as "ArrowDtype")
    _dtype: StringDtype  # type: ignore[assignment]

    def __init__(self, values, *, dtype: StringDtype | None = None) -> None:
        # [FUNCTION DOC]
        # Inputs: values, a pyarrow array or chunked array; dtype (keyword-only,
        # default none), a string dtype instance, defaulting to the pyarrow-storage
        # dtype with the NA scalar as missing value. Raises an import error if
        # pyarrow is missing. String, string-view and dictionary-of-string pyarrow
        # data are first cast to large string. After the parent initialiser runs,
        # a value error is raised unless the stored type is large string.
        _check_pyarrow_available()
        if isinstance(values, (pa.Array, pa.ChunkedArray)) and (
            pa.types.is_string(values.type)
            or _is_string_view(values.type)
            or (
                pa.types.is_dictionary(values.type)
                and (
                    pa.types.is_string(values.type.value_type)
                    or pa.types.is_large_string(values.type.value_type)
                    or _is_string_view(values.type.value_type)
                )
            )
        ):
            values = pc.cast(values, pa.large_string())

        super().__init__(values)

        if dtype is None:
            dtype = StringDtype(storage="pyarrow", na_value=libmissing.NA)
        self._dtype = dtype

        if not pa.types.is_large_string(self._pa_array.type):
            raise ValueError(
                "ArrowStringArray requires a PyArrow (chunked) array of "
                "large_string type"
            )

    def _from_pyarrow_array(self, pa_array):
        # [FUNCTION DOC]
        # HELPER: wraps an operation's pyarrow result in a new array of the same
        # class, keeping this array's dtype (and so its missing-value semantics).
        """
        Construct from the pyarrow array result of an operation, retaining
        self.dtype.na_value.
        """
        return type(self)(pa_array, dtype=self.dtype)

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        # [FUNCTION DOC]
        # HELPER (class method): boxes a Python value as a pyarrow scalar using the
        # parent logic, then, when no explicit pyarrow type was requested and the
        # result is a regular string, casts it to large string. Returns the scalar.
        pa_scalar = super()._box_pa_scalar(value, pa_type)
        if pa.types.is_string(pa_scalar.type) and pa_type is None:
            pa_scalar = pc.cast(pa_scalar, pa.large_string())
        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        # [FUNCTION DOC]
        # HELPER (class method): boxes values as a pyarrow array using the parent
        # logic, then, when no explicit pyarrow type was requested and the result is
        # a regular string, casts it to large string. Returns the array.
        # KNOWN ISSUE: the copy argument is accepted but never forwarded to the
        # parent method, so it has no effect.
        pa_array = super()._box_pa_array(value, pa_type)
        if pa.types.is_string(pa_array.type) and pa_type is None:
            pa_array = pc.cast(pa_array, pa.large_string())
        return pa_array

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of elements: the length of the pyarrow data.
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._pa_array)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # Class method building an array from a sequence. Inputs: scalars,
        # anything array-like; dtype (keyword-only, default none), a string dtype,
        # or the alias "string" or none; copy (keyword-only, default false).
        # Raises an import error without pyarrow. A given dtype other than the
        # plain alias is resolved and asserted to be a pyarrow-storage string dtype.
        # Conversion by input kind: masked arrays use their data and mask directly
        # (strings made without converting missing values); Arrow extension arrays
        # use pyarrow's cast for integer, string and boolean types (booleans
        # capitalised to match Python's text), and the generic string conversion for
        # other types such as floats because their text differs; raw pyarrow arrays
        # are cast to large string; everything else is converted to strings with
        # missing values preserved. Builds the array with the dtype given.
        # KNOWN ISSUE: when dtype is the alias string "string" it is not resolved to
        # a dtype object and is passed on as-is to the constructor, which then
        # stores the raw string as the dtype instead of a dtype instance.
        from pandas.core.arrays.masked import BaseMaskedArray

        _check_pyarrow_available()

        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "pyarrow"

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype in ensure_string_array and
            # numerical issues with Float32Dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(
                result, copy=copy, convert_na_value=False, skipna=False
            )
            pa_arr = pa.array(result, mask=na_values, type=pa.large_string())
        elif isinstance(scalars, ArrowExtensionArray):
            pa_type = scalars._pa_array.type
            # Use PyArrow's native cast for integer, string, and boolean types.
            # Float has different representation in PyArrow: 1.0 -> "1" instead
            # of "1.0", and uses different scientific notation (1e+10 vs 1e10).
            # Boolean needs capitalize (true -> True, false -> False).
            if (
                pa.types.is_integer(pa_type)
                or pa.types.is_large_string(pa_type)
                or pa.types.is_string(pa_type)
                or pa.types.is_boolean(pa_type)
            ):
                pa_arr = pc.cast(scalars._pa_array, pa.large_string())
                if pa.types.is_boolean(pa_type):
                    pa_arr = pc.utf8_capitalize(pa_arr)
            else:
                # Fall back for types where PyArrow's string representation
                # differs from Python's str()
                result = lib.ensure_string_array(scalars, copy=copy)
                pa_arr = pa.array(result, type=pa.large_string(), from_pandas=True)
        elif isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            pa_arr = pc.cast(scalars, pa.large_string())
        else:
            # convert non-na-likes to str
            result = lib.ensure_string_array(scalars, copy=copy)
            pa_arr = pa.array(result, type=pa.large_string(), from_pandas=True)
        # error: Argument "dtype" to "ArrowStringArray" has incompatible type
        return cls(pa_arr, dtype=dtype)  # type: ignore[arg-type]

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # Class method: builds an array from a sequence of strings by simply
        # delegating to the general sequence constructor with the same dtype and
        # copy setting.
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self) -> StringDtype:  # type: ignore[override]
        # [FUNCTION DOC]
        # Property: the string dtype instance stored on the array.
        """
        An instance of 'string[pyarrow]'.
        """
        return self._dtype

    def insert(self, loc: int, item) -> ArrowStringArray:
        # [FUNCTION DOC]
        # Inserts one item at position loc, returning a new array. Item must be a
        # string or a missing value, else a type error naming the offending type is
        # raised. For NaN-semantics dtypes a NaN item is first replaced by the NA
        # scalar. Insertion itself is done by the parent method.
        if self.dtype.na_value is np.nan and item is np.nan:
            item = libmissing.NA
        if not isinstance(item, str) and item is not libmissing.NA:
            raise TypeError(
                f"Invalid value '{item}' for dtype 'str'. Value should be a "
                f"string or missing value, got '{type(item).__name__}' instead."
            )
        return super().insert(loc, item)

    def _convert_bool_result(self, values, na=lib.no_default, method_name=None):
        # [FUNCTION DOC]
        # HELPER: converts a pyarrow boolean result of a string method into the
        # right output type. Inputs: values, pyarrow boolean data; na (default: not
        # given), the value to use for missing results; method_name (unused). The
        # na argument is validated. NaN-semantics dtypes: missing results become
        # false (when na is not given or is itself missing) or na, and a numpy
        # boolean array is returned. NA-semantics dtypes: missing results are
        # replaced by na only if a non-missing na was given, and a nullable boolean
        # array is returned.
        validate_na_arg(na, name="na")
        if self.dtype.na_value is np.nan:
            if na is lib.no_default or isna(na):
                # NaN propagates as False
                values = values.fill_null(False)
            else:
                values = values.fill_null(na)
            return values.to_numpy()
        elif na is not lib.no_default and not isna(na):  # pyright: ignore [reportGeneralTypeIssues]
            values = values.fill_null(na)
        return BooleanDtype().__from_arrow__(values)

    def _maybe_convert_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: validates and normalises a value being assigned. Scalars: missing
        # values become none; non-strings raise a type error. Another array of this
        # class passes through. Other inputs are converted to a numpy array (object
        # dtype if not array-like) and must be one-dimensional and entirely strings
        # or missing, else a type error is raised (empty input is accepted). The
        # result is then given to the parent conversion, whose result is returned.
        """Maybe convert value to be pyarrow compatible."""
        if is_scalar(value):
            if isna(value):
                value = None
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype 'str'. Value should be a "
                    f"string or missing value, got '{type(value).__name__}' instead."
                )
        elif isinstance(value, type(self)):
            pass
        else:
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            else:
                value = np.asarray(value)
            if len(value) and not (
                value.ndim == 1 and lib.is_string_array(value, skipna=True)
            ):
                raise TypeError(
                    "Invalid value for dtype 'str'. Value should be a "
                    "string or missing value (or array of those)."
                )
        return super()._maybe_convert_setitem_value(value)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Membership test. Input values: array-like of candidates. Only candidates
        # that convert to string, large string or null pyarrow scalars are kept
        # (others cannot match). If none remain, returns an all-false numpy
        # boolean array; otherwise tests membership in pyarrow and converts to a
        # numpy boolean array, turning null results into false. Returns that array.
        value_set = [
            pa_scalar.as_py()
            for pa_scalar in [pa.scalar(value, from_pandas=True) for value in values]
            if pa_scalar.type in (pa.string(), pa.null(), pa.large_string())
        ]

        # short-circuit to return all False array.
        if not value_set:
            return np.zeros(len(self), dtype=bool)

        result = pc.is_in(
            self._pa_array, value_set=pa.array(value_set, type=self._pa_array.type)
        )
        # pyarrow 2.0.0 returned nulls, so we explicitly specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Converts to another dtype. Inputs: dtype, anything convertible to a
        # dtype; copy (default true). The same dtype returns a copy (or self when
        # copy is false). Numeric extension dtypes are produced by casting the
        # pyarrow data to the matching pyarrow type and converting from it. Numpy
        # floating dtypes use numpy conversion with NaN for missing values. All
        # other dtypes go to the parent implementation.
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, NumericDtype):
            data = self._pa_array.cast(pa.from_numpy_dtype(dtype.numpy_dtype))
            return dtype.__from_arrow__(data)
        elif isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating):
            return self.to_numpy(dtype=dtype, na_value=np.nan)

        return super().astype(dtype, copy=copy)

    # ------------------------------------------------------------------------
    # String methods interface

    _str_isalnum = ArrowStringArrayMixin._str_isalnum
    _str_isalpha = ArrowStringArrayMixin._str_isalpha
    _str_isdecimal = ArrowStringArrayMixin._str_isdecimal
    _str_isdigit = ArrowStringArrayMixin._str_isdigit
    _str_islower = ArrowStringArrayMixin._str_islower
    _str_isnumeric = ArrowStringArrayMixin._str_isnumeric
    _str_isspace = ArrowStringArrayMixin._str_isspace
    _str_istitle = ArrowStringArrayMixin._str_istitle
    _str_isupper = ArrowStringArrayMixin._str_isupper

    _str_map = BaseStringArray._str_map
    _str_startswith = ArrowStringArrayMixin._str_startswith
    _str_endswith = ArrowStringArrayMixin._str_endswith
    _str_pad = ArrowStringArrayMixin._str_pad
    _str_lower = ArrowStringArrayMixin._str_lower
    _str_upper = ArrowStringArrayMixin._str_upper
    _str_strip = ArrowStringArrayMixin._str_strip
    _str_lstrip = ArrowStringArrayMixin._str_lstrip
    _str_rstrip = ArrowStringArrayMixin._str_rstrip
    _str_removesuffix = ArrowStringArrayMixin._str_removesuffix
    _str_removeprefix = ArrowStringArrayMixin._str_removeprefix
    _str_find = ArrowStringArrayMixin._str_find
    _str_get = ArrowStringArrayMixin._str_get
    _str_getitem = ArrowStringArrayMixin._str_getitem
    _str_capitalize = ArrowStringArrayMixin._str_capitalize
    _str_title = ArrowStringArrayMixin._str_title
    _str_swapcase = ArrowStringArrayMixin._str_swapcase
    _str_slice_replace = ArrowStringArrayMixin._str_slice_replace
    _str_len = ArrowStringArrayMixin._str_len
    _str_slice = ArrowStringArrayMixin._str_slice

    @staticmethod
    def _is_re_pattern_with_flags(pat: str | re.Pattern) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when pat is a compiled regular expression carrying flags
        # other than ignore-case and unicode (the only ones pyarrow can honour).
        # check if `pat` is a compiled regex pattern with flags that are not
        # supported by pyarrow
        return (
            isinstance(pat, re.Pattern)
            and (pat.flags & ~(re.IGNORECASE | re.UNICODE)) != 0
        )

    @staticmethod
    def _preprocess_re_pattern(
        pat: str | re.Pattern, case: bool, flags: int
    ) -> tuple[str, bool, int]:
        # [FUNCTION DOC]
        # HELPER (static): prepares a regular expression for pyarrow. Inputs: pat,
        # a pattern string or compiled pattern; case, whether matching is case
        # sensitive; flags, regex flag bits. For a compiled pattern its text is used,
        # its flag bits replace the flags argument, ignore-case is turned into
        # case=false, and the unicode flag is dropped (pyarrow assumes unicode). A
        # pattern ending in an unescaped end-of-string anchor in Python's spelling
        # is rewritten to pyarrow's spelling. Returns (pattern text, case, flags).
        # KNOWN ISSUE: for compiled patterns, flags passed separately by the caller
        # are discarded (a source comment marks this as a to-do).
        if isinstance(pat, re.Pattern):
            pattern = pat.pattern
            # TODO flags passed separately by user are ignored
            flags = pat.flags
            # flags is not supported by pyarrow, but `case` is -> extract and remove
            if flags & re.IGNORECASE:
                case = False
                flags = flags & ~re.IGNORECASE
            # when creating a pattern with re.compile and a string, it automatically
            # gets a UNICODE flag, while pyarrow assumes unicode for strings anyway
            flags = flags & ~re.UNICODE
        else:
            pattern = pat

        if (
            pattern.endswith("\\Z")
            # Second condition counts the number of `\` that patterns ends with
            # prior to Z -> needs to be odd to end with an unescaped \Z
            and (len(pattern) - len(pattern[:-1].rstrip("\\")) + 1) % 2 == 1
        ):
            pattern = pattern[:-2] + "\\z"

        return pattern, case, flags

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        # [FUNCTION DOC]
        # Substring/regex containment test. Inputs: pat, pattern; case (default
        # true); flags (default 0); na (default: not given), result for missing
        # values; regex (default true). Falls back to the object-based
        # implementation when flags are given, the pattern is a compiled pattern
        # with unsupported flags, or (in regex mode) the pattern uses features
        # pyarrow lacks. Otherwise preprocesses the pattern and uses the Arrow
        # implementation.
        if (
            flags
            or self._is_re_pattern_with_flags(pat)
            or (regex and self._has_unsupported_regex(pat))
        ):
            return super()._str_contains(pat, case, flags, na, regex)

        pat, case, flags = self._preprocess_re_pattern(pat, case, flags)
        return ArrowStringArrayMixin._str_contains(self, pat, case, flags, na, regex)

    def _str_match(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        # [FUNCTION DOC]
        # Match from the start of each string. Inputs: pat (string or compiled
        # pattern), case (default true), flags (default 0), na (default: not
        # given). Uses the object-based fallback when flags exist, the compiled
        # pattern has unsupported flags or the regex is unsupported by pyarrow;
        # otherwise the Arrow implementation after pattern preprocessing.
        if (
            flags
            or self._is_re_pattern_with_flags(pat)
            or self._has_unsupported_regex(pat)
        ):
            return super()._str_match(pat, case, flags, na)

        pat, case, flags = self._preprocess_re_pattern(pat, case, flags)
        return ArrowStringArrayMixin._str_match(self, pat, case, flags, na)

    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        # [FUNCTION DOC]
        # Match the whole string. Same inputs and fallback rules as the start-match
        # method, but uses the Arrow whole-string match after preprocessing.
        if (
            flags
            or self._is_re_pattern_with_flags(pat)
            or self._has_unsupported_regex(pat)
        ):
            return super()._str_fullmatch(pat, case, flags, na)

        pat, case, flags = self._preprocess_re_pattern(pat, case, flags)
        return ArrowStringArrayMixin._str_fullmatch(self, pat, case, flags, na)

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        # [FUNCTION DOC]
        # Replace occurrences of a pattern. Inputs: pat, repl (text or callable), n
        # (maximum replacements, -1 for all), case, flags, regex. Falls back to the
        # object-based implementation for compiled patterns, callable replacements,
        # case-insensitive matching, flags, replacement text with a named-group
        # reference, or unsupported regex features in regex mode. Otherwise the
        # pattern is preprocessed (regex mode only) and the Arrow implementation is
        # used.
        if (
            isinstance(pat, re.Pattern)
            or callable(repl)
            or not case
            or flags
            or (  # substitution contains a named group pattern
                # https://docs.python.org/3/library/re.html
                isinstance(repl, str) and r"\g<" in repl
            )
            or (regex and self._has_unsupported_regex(pat))
        ):
            return super()._str_replace(pat, repl, n, case, flags, regex)

        if regex:
            pat, case, flags = self._preprocess_re_pattern(pat, case, flags)

        return ArrowStringArrayMixin._str_replace(
            self, pat, repl, n, case, flags, regex
        )

    def _str_repeat(self, repeats: int | Sequence[int]):
        # [FUNCTION DOC]
        # Repeats each string. Input repeats: one integer, or a sequence of
        # per-element counts. Sequences use the object-based fallback; a single
        # integer uses the Arrow extension array's implementation.
        if not isinstance(repeats, int):
            return super()._str_repeat(repeats)
        else:
            return ArrowExtensionArray._str_repeat(self, repeats=repeats)

    def _str_count(self, pat: str, flags: int = 0):
        # [FUNCTION DOC]
        # Counts pattern matches per element. Inputs: pat, regular expression text;
        # flags (default 0). With flags or an unsupported regex, uses the
        # object-based fallback. Otherwise counts with pyarrow and converts the
        # count array to the dtype-appropriate integer result.
        if flags or self._has_unsupported_regex(pat):
            return super()._str_count(pat, flags)

        pat, _, _ = self._preprocess_re_pattern(pat, True, 0)
        result = pc.count_substring_regex(self._pa_array, pat)
        return self._convert_int_result(result)

    def _str_get_dummies(self, sep: str = "|", dtype: NpDtype | None = None):
        # [FUNCTION DOC]
        # Builds indicator columns from delimited strings. Inputs: sep (default
        # pipe), dtype (default 64-bit integer). Delegates to the plain Arrow
        # extension array version, stacks the per-row indicator arrays into a
        # two-dimensional numpy array, and casts it to the requested numpy dtype
        # (boolean if the dtype is not a numpy dtype). Returns (indicator matrix,
        # labels); with no labels returns an empty zero-by-zero matrix.
        if dtype is None:
            dtype = np.int64
        dummies_pa, labels = ArrowExtensionArray(self._pa_array)._str_get_dummies(
            sep, dtype
        )
        if len(labels) == 0:
            return np.empty(shape=(0, 0), dtype=dtype), labels
        dummies = np.vstack(dummies_pa.to_numpy())
        _dtype = pandas_dtype(dtype)
        dummies_dtype: NpDtype
        if isinstance(_dtype, np.dtype):
            dummies_dtype = _dtype
        else:
            dummies_dtype = np.bool_
        return dummies.astype(dummies_dtype, copy=False), labels

    def _convert_int_result(self, result):
        # [FUNCTION DOC]
        # HELPER: converts a pyarrow integer result to the output type. For
        # NaN-semantics dtypes: casts to 64-bit integers and returns a numpy
        # array (via the appropriate pyarrow conversion for arrays versus chunked
        # arrays). For NA-semantics dtypes: returns a nullable 64-bit integer array.
        if self.dtype.na_value is np.nan:
            result = result.cast(pa.int64())
            if isinstance(result, pa.Array):
                result = result.to_numpy(zero_copy_only=False)
            else:
                result = result.to_numpy()
            return result

        return Int64Dtype().__from_arrow__(result)

    def _convert_rank_result(self, result):
        # [FUNCTION DOC]
        # HELPER: converts a pyarrow rank result. NaN-semantics dtypes: returns a
        # numpy array cast to 64-bit floats. NA-semantics dtypes: returns a nullable
        # 64-bit float array.
        if self.dtype.na_value is np.nan:
            if isinstance(result, pa.Array):
                result = result.to_numpy(zero_copy_only=False)
            else:
                result = result.to_numpy()
            return result.astype("float64", copy=False)

        return Float64Dtype().__from_arrow__(result)

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # Reduction. Inputs: name, reduction name; skipna (keyword-only, default
        # true); keepdims (default false); kwargs passed through. For NaN-semantics
        # dtypes, any and all treat a string as true when non-empty (missing values
        # count as true when not skipping missing) and run on a derived boolean
        # array; with keepdims the result is converted to numpy boolean. Supported
        # for all dtypes: minimum, maximum, sum, argmin, argmax, computed by the
        # parent calculation; other names raise a type error. Position results
        # become dtype-appropriate integer arrays; array results are rewrapped in
        # this class; scalars are returned unchanged.
        if self.dtype.na_value is np.nan and name in ["any", "all"]:
            if not skipna:
                nas = pc.is_null(self._pa_array)
                arr = pc.or_kleene(nas, pc.not_equal(self._pa_array, ""))
            else:
                arr = pc.not_equal(self._pa_array, "")
            result = ArrowExtensionArray(arr)._reduce(
                name, skipna=skipna, keepdims=keepdims, **kwargs
            )
            if keepdims:
                # ArrowExtensionArray will return a length-1 bool[pyarrow] array
                return result.astype(np.bool_)
            return result

        if name in ("min", "max", "sum", "argmin", "argmax"):
            result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
        else:
            raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

        if name in ("argmin", "argmax") and isinstance(result, pa.Array):
            return self._convert_int_result(result)
        elif isinstance(result, pa.Array):
            return type(self)(result, dtype=self.dtype)
        else:
            return result

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts unique values. Input dropna (default true). Uses the parent method;
        # for NaN-semantics dtypes, rebuilds the result Series with plain numpy
        # counts (same index and name, no copy). Returns a Series.
        result = super().value_counts(dropna=dropna)
        if self.dtype.na_value is np.nan:
            res_values = result._values.to_numpy()
            return result._constructor(
                res_values, index=result.index, name=result.name, copy=False
            )
        return result

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # Element-wise comparison. Inputs: other, comparand; op, the comparison
        # operator function. If this array uses NaN semantics and other is a string
        # or Arrow array using NA semantics, returns the not-implemented marker so
        # that the NA-semantics side handles it (NA has priority). Otherwise uses
        # the parent comparison; for NaN-semantics dtypes the result is converted
        # to a numpy boolean array with missing positions being true for
        # inequality and false for every other operator.
        if (
            isinstance(other, (BaseStringArray, ArrowExtensionArray))
            and self.dtype.na_value is not libmissing.NA
            and other.dtype.na_value is libmissing.NA
        ):
            # NA has priority of NaN semantics
            return NotImplemented

        result = super()._cmp_method(other, op)
        if self.dtype.na_value is np.nan:
            if op == operator.ne:
                return result.to_numpy(np.bool_, na_value=True)
            else:
                return result.to_numpy(np.bool_, na_value=False)
        return result

    def __pos__(self) -> Self:
        # [FUNCTION DOC]
        # Unary plus is unsupported: always raises a type error naming the dtype.
        raise TypeError(f"bad operand type for unary +: '{self.dtype}'")
