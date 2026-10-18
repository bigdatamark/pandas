# [MODULE DOC]
# Implements the pyarrow-backed extension array and its supporting helpers.
# Public contents: the arrow extension array class (exposed in the arrays namespace),
# a function converting dtype-likes to pyarrow types, and a fast transpose helper for
# lists of arrow arrays. Private helpers handle buffer ownership and pyarrow-bug
# workarounds. When pyarrow is installed, the module also builds tables mapping operator
# names to pyarrow compute kernels (comparison, Kleene logical, bitwise and checked
# arithmetic, including compatibility helpers for true and floor division).
# Relationships: the class combines the generic extension array interface, the operator
# mixin and the arrow string mixin; it is the base for the arrow-backed string array
# class elsewhere in the package, uses the arrow dtype from the core dtypes package, and
# converts to numpy-backed datetime, timedelta and masked arrays for some operations
# (groupby).
# Import-time side effects: imports pyarrow and its compute module if available (the
# module can be imported without pyarrow, but the class cannot be instantiated); no other
# global state is changed.
from __future__ import annotations

from datetime import (
    date,
    datetime,
)
import functools
import operator
import re
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)
import unicodedata
import warnings

import numpy as np

from pandas._config import is_nan_na

from pandas._libs import lib
from pandas._libs.missing import is_pdna_or_none
from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
    timezones,
)
from pandas.compat import (
    HAS_PYARROW,
    PYARROW_MIN_VERSION,
    pa_version_under16p0,
    pa_version_under21p0,
)
from pandas.errors import Pandas4Warning
from pandas.util._decorators import (
    doc,
    set_module,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    can_hold_element,
    construct_1d_object_array_from_listlike,
    infer_dtype_from_scalar,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    algorithms as algos,
    missing,
    ops,
    roperator,
)
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionArraySupportsAnyAll,
)
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
    unpack_tuple_and_ellipses,
    validate_indices,
)
from pandas.core.nanops import check_below_min_count

from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.compat.pyarrow import _safe_fill_null

    from pandas.core.dtypes.dtypes import ArrowDtype

    ARROW_CMP_FUNCS = {
        "eq": pc.equal,
        "ne": pc.not_equal,
        "lt": pc.less,
        "gt": pc.greater,
        "le": pc.less_equal,
        "ge": pc.greater_equal,
    }

    ARROW_LOGICAL_FUNCS = {
        "and_": pc.and_kleene,
        "rand_": lambda x, y: pc.and_kleene(y, x),
        "or_": pc.or_kleene,
        "ror_": lambda x, y: pc.or_kleene(y, x),
        "xor": pc.xor,
        "rxor": lambda x, y: pc.xor(y, x),
    }

    ARROW_BIT_WISE_FUNCS = {
        "and_": pc.bit_wise_and,
        "rand_": lambda x, y: pc.bit_wise_and(y, x),
        "or_": pc.bit_wise_or,
        "ror_": lambda x, y: pc.bit_wise_or(y, x),
        "xor": pc.bit_wise_xor,
        "rxor": lambda x, y: pc.bit_wise_xor(y, x),
    }

    def cast_for_truediv(
        arrow_array: pa.ChunkedArray, pa_object: pa.Array | pa.Scalar
    ) -> tuple[pa.ChunkedArray, pa.Array | pa.Scalar]:
        # [FUNCTION DOC]
        # HELPER: defined only when pyarrow is available; used by the true-division
        # entries of the arithmetic function table. Parameters: arrow_array is a chunked
        # array and pa_object an array or scalar. If both have integer types, both are
        # cast to 64-bit float without safety checks so that dividing integers yields
        # floats like Python and numpy do (pyarrow's checked integer division would stay
        # integer); otherwise they are returned unchanged. Returns the pair (left,
        # right).
        # Ensure int / int -> float mirroring Python/Numpy behavior
        # as pc.divide_checked(int, int) -> int
        if pa.types.is_integer(arrow_array.type) and pa.types.is_integer(
            pa_object.type
        ):
            # GH: 56645.
            # https://github.com/apache/arrow/issues/35563
            return pc.cast(arrow_array, pa.float64(), safe=False), pc.cast(
                pa_object, pa.float64(), safe=False
            )

        return arrow_array, pa_object

    def floordiv_compat(
        left: pa.ChunkedArray | pa.Array | pa.Scalar,
        right: pa.ChunkedArray | pa.Array | pa.Scalar,
    ) -> pa.ChunkedArray:
        # [FUNCTION DOC]
        # HELPER: defined only when pyarrow is available; floor division used by the
        # arithmetic function table until pyarrow provides a kernel. Parameters: left
        # and right are arrow arrays or scalars. For two integer types it uses checked
        # integer division and, for signed results, subtracts one where there is a non-
        # zero remainder and the operands have opposite signs (to round toward negative
        # infinity instead of zero); the result is cast back to the left type. For other
        # types it divides and takes the floor. Returns the arrow result. Division by
        # zero raises the arrow error from the checked kernel.
        # TODO: Replace with pyarrow floordiv kernel.
        # https://github.com/apache/arrow/issues/39386
        if pa.types.is_integer(left.type) and pa.types.is_integer(right.type):
            divided = pc.divide_checked(left, right)
            if pa.types.is_signed_integer(divided.type):
                # GH 56676
                has_remainder = pc.not_equal(pc.multiply(divided, right), left)
                has_one_negative_operand = pc.less(
                    pc.bit_wise_xor(left, right),
                    pa.scalar(0, type=divided.type),
                )
                result = pc.if_else(
                    pc.and_(
                        has_remainder,
                        has_one_negative_operand,
                    ),
                    # GH: 55561
                    pc.subtract(divided, pa.scalar(1, type=divided.type)),
                    divided,
                )
            else:
                result = divided
            result = result.cast(left.type)
        else:
            divided = pc.divide(left, right)
            result = pc.floor(divided)
        return result

    ARROW_ARITHMETIC_FUNCS = {
        "add": pc.add_checked,
        "radd": lambda x, y: pc.add_checked(y, x),
        "sub": pc.subtract_checked,
        "rsub": lambda x, y: pc.subtract_checked(y, x),
        "mul": pc.multiply_checked,
        "rmul": lambda x, y: pc.multiply_checked(y, x),
        "truediv": lambda x, y: pc.divide(*cast_for_truediv(x, y)),
        "rtruediv": lambda x, y: pc.divide(*cast_for_truediv(y, x)),
        "floordiv": lambda x, y: floordiv_compat(x, y),
        "rfloordiv": lambda x, y: floordiv_compat(y, x),
        "mod": NotImplemented,
        "rmod": NotImplemented,
        "divmod": NotImplemented,
        "rdivmod": NotImplemented,
        "pow": pc.power_checked,
        "rpow": lambda x, y: pc.power_checked(y, x),
    }

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from pandas._libs.missing import NAType
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        Iterator,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        Scalar,
        SortKind,
        TakeIndexer,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import Series
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray


def to_pyarrow_type(
    dtype: ArrowDtype | pa.DataType | Dtype | None,
) -> pa.DataType | None:
    # [FUNCTION DOC]
    # Convert a dtype-like to a pyarrow data type. Parameter dtype may be an arrow dtype
    # (its pyarrow type is returned), a pyarrow type (returned as is), a timezone-aware
    # datetime dtype (becomes a pyarrow timestamp with the same unit and zone), or any
    # other truthy value, which is passed to pyarrow's numpy-dtype conversion (python
    # types accepted, not all numpy types handled). Returns None for falsy input or when
    # pyarrow has no conversion (the not-implemented error is swallowed).
    """
    Convert dtype to a pyarrow type instance.
    """
    if isinstance(dtype, ArrowDtype):
        return dtype.pyarrow_dtype
    elif isinstance(dtype, pa.DataType):
        return dtype
    elif isinstance(dtype, DatetimeTZDtype):
        return pa.timestamp(dtype.unit, dtype.tz)
    elif dtype:
        try:
            # Accepts python types too
            # Doesn't handle all numpy types
            return pa.from_numpy_dtype(dtype)
        except pa.ArrowNotImplementedError:
            pass
    return None


def _is_varbinary_type(pa_type: pa.DataType) -> bool:
    # [FUNCTION DOC]
    # HELPER: predicate for the conditional-select workaround. Returns true if the
    # pyarrow type is string, large string, binary or large binary; these are the four
    # types for which pyarrow's conditional select mishandles non-zero offsets
    # (GH#64320).
    """
    Whether this is one of string, large_string, binary and large_binary.

    pc.if_else misreads a non-zero offset for exactly these four, silently
    truncating values (GH#64320, https://github.com/apache/arrow/issues/49410).
    Other offset-carrying layouts such as list and map are unaffected.
    """
    return (
        pa.types.is_string(pa_type)
        or pa.types.is_large_string(pa_type)
        or pa.types.is_binary(pa_type)
        or pa.types.is_large_binary(pa_type)
    )


def _is_string_or_binary_view(typ):
    # [FUNCTION DOC]
    # HELPER: returns true if the pyarrow type is a string-view or binary-view type;
    # always false when pyarrow is older than version 16, which lacks those types.
    return not pa_version_under16p0 and (
        pa.types.is_string_view(typ) or pa.types.is_binary_view(typ)
    )


def _boxing_may_borrow_memory(pa_type: pa.DataType) -> bool:
    # [FUNCTION DOC]
    # HELPER: says whether converting data of this pyarrow type to an arrow array can
    # return a view of memory owned by the caller. False for the character layouts
    # (which are always repacked) and string/binary views, true otherwise (fixed-width
    # and nested types). Used to decide whether a defensive buffer copy is needed before
    # adopting an array.
    """
    Whether ``pa.array`` on this type can return a view on caller-owned memory.

    Zero-copy over numpy/masked arrays for fixed-width layouts; character
    layouts always repack, so copying those would cost a full copy of the
    character data for no safety gain. Nested types may have a zero-copy child.
    """
    return not (_is_varbinary_type(pa_type) or _is_string_or_binary_view(pa_type))


def _copy_pyarrow_buffers(
    pa_array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    # [FUNCTION DOC]
    # HELPER: returns an equal array or chunked array that owns its own buffers
    # (GH#67990). Chunked arrays are rebuilt chunk by chunk with the same type.
    # Dictionary arrays are rebuilt from copies of their indices and dictionary,
    # preserving orderedness, because concatenation would reuse the dictionary. Other
    # arrays are copied by concatenating the single array with nothing else.
    """
    Return an equal array that owns its buffers (GH#67990).

    ``pa.concat_arrays`` reuses the ``dictionary`` child rather than copying it,
    so dictionary types are rebuilt from copies of both halves.
    """
    if isinstance(pa_array, pa.ChunkedArray):
        return pa.chunked_array(
            [_copy_pyarrow_buffers(chunk) for chunk in pa_array.chunks],
            type=pa_array.type,
        )
    if pa.types.is_dictionary(pa_array.type):
        return pa.DictionaryArray.from_arrays(
            _copy_pyarrow_buffers(pa_array.indices),
            _copy_pyarrow_buffers(pa_array.dictionary),
            ordered=pa_array.type.ordered,
        )
    return pa.concat_arrays([pa_array])


@set_module("pandas.arrays")
class ArrowExtensionArray(
    OpsMixin,
    ExtensionArraySupportsAnyAll,
    ArrowStringArrayMixin,
):
    # [CLASS DOC]
    # Public extension array (exposed in the arrays namespace) holding data in a pyarrow
    # chunked array. Inherits the operator mixin (providing dunder operators from the
    # private comparison, arithmetic and logical methods), the any/all-supporting
    # extension array base, and the string-method mixin, whose abstract hooks are
    # implemented here. State: the chunked array attribute (always chunked; a plain
    # array is wrapped in a single chunk) and the arrow dtype attribute derived from its
    # type. Most behavior is delegated to pyarrow compute kernels, with numpy or
    # pointwise fallbacks where kernels are missing. Instances are mutable by
    # reassigning the backing array in item assignment; pickling combines chunks. Also
    # hosts the private datetime-accessor (names starting with a date-part prefix) and
    # string-accessor hooks used by the series accessors. Groups of methods:
    # construction and boxing, item access and iteration, operators, missing-value
    # handling, sorting, reductions and accumulations, set-like operations, conversion,
    # groupby support, string and datetime hooks.
    """
    Pandas ExtensionArray backed by a PyArrow ChunkedArray.

    .. warning::

       ArrowExtensionArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The input data to initialize the ArrowExtensionArray.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    ArrowExtensionArray

    See Also
    --------
    array : Create a Pandas array with a specified dtype.
    DataFrame.to_feather : Write a DataFrame to the binary Feather format.
    read_feather : Load a feather-format object from the file path.

    Notes
    -----
    Most methods are implemented using `pyarrow compute functions. <https://arrow.apache.org/docs/python/api/compute.html>`__
    Some methods may either raise an exception or raise a ``PerformanceWarning`` if an
    associated compute function is not available based on the installed version of PyArrow.

    Please install the latest version of PyArrow to enable the best functionality and avoid
    potential bugs in prior versions of PyArrow.

    Examples
    --------
    Create an ArrowExtensionArray with :func:`pandas.array`:

    >>> pd.array([1, 1, None], dtype="int64[pyarrow]")
    <ArrowExtensionArray>
    [1, 1, <NA>]
    Length: 3, dtype: int64[pyarrow]
    """  # noqa: E501 (http link too long)

    _pa_array: pa.ChunkedArray
    _dtype: ArrowDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        # [FUNCTION DOC]
        # Constructor. Parameter values is a pyarrow array or chunked array; an array is
        # wrapped as a one-chunk chunked array. Raises an import error stating the
        # minimum pyarrow version when pyarrow is not installed, and a value error for
        # any other input type. Sets the dtype attribute to an arrow dtype of the data
        # type.
        if not HAS_PYARROW:
            msg = (
                f"pyarrow>={PYARROW_MIN_VERSION} is required for PyArrow "
                "backed ArrowExtensionArray."
            )
            raise ImportError(msg)
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
        self._dtype = ArrowDtype(self._pa_array.type)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level constructor from scalars. Parameters: scalars (sequence or
        # array); dtype (keyword, default None; converted to a pyarrow type); copy
        # (keyword, default false). Boxes the input into a pyarrow array of that type
        # through the array-boxing helper and wraps it. Returns a new array.
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        pa_type = to_pyarrow_type(dtype)
        pa_array = cls._box_pa_array(scalars, pa_type=pa_type, copy=copy)
        arr = cls(pa_array)
        return arr

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level constructor from strings. Parameters: strings (sequence or
        # array); dtype (keyword, required; converted to a pyarrow type); copy (keyword,
        # default false). Missing positions are recorded first. The strings are
        # converted according to the target type: no type, string, large string or
        # binary pass through; timestamps use the datetime parser (raising on errors);
        # dates use the datetime parser then take the date part; durations use the
        # timedelta parser, and for units other than nanoseconds the strings are instead
        # cast to 64-bit integers if possible (matching pyarrow's duration/string
        # behavior); times use the time parser with coercion; booleans are cast through
        # pyarrow after mapping the strings for 1.0 and 0.0 to 1 and 0; integer,
        # floating and decimal types use the numeric parser, or an arrow cast when the
        # input is already arrow data. Any other type raises a not-implemented error.
        # The parsed values are passed to the scalar constructor. Returns a new array.
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
        mask = isna(strings)

        if isinstance(strings, cls):
            strings = strings._pa_array

        pa_type = to_pyarrow_type(dtype)
        if (
            pa_type is None
            or pa.types.is_binary(pa_type)
            or pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
        ):
            # pa_type is None: Let pa.array infer
            # pa_type is string/binary: scalars already correct type
            scalars = strings
        elif pa.types.is_timestamp(pa_type):
            from pandas.core.tools.datetimes import to_datetime

            scalars = to_datetime(strings, errors="raise")
        elif pa.types.is_date(pa_type):
            from pandas.core.tools.datetimes import to_datetime

            scalars = to_datetime(strings, errors="raise").date
            scalars = pa.array(scalars, type=pa_type, mask=mask)
        elif pa.types.is_duration(pa_type):
            from pandas.core.tools.timedeltas import to_timedelta

            scalars = to_timedelta(strings, errors="raise")

            if pa_type.unit != "ns":
                # GH51175: test_from_sequence_of_strings_pa_array
                # attempt to parse as int64 reflecting pyarrow's
                # duration to string casting behavior
                mask = isna(scalars)
                if not isinstance(strings, (pa.Array, pa.ChunkedArray)):
                    strings = pa.array(strings, type=pa.string(), mask=mask)
                strings = pc.if_else(mask, None, strings)
                try:
                    scalars = strings.cast(pa.int64())
                except pa.ArrowInvalid:
                    pass
        elif pa.types.is_time(pa_type):
            from pandas.core.tools.times import to_time

            # "coerce" to allow "null times" (None) to not raise
            scalars = to_time(strings, errors="coerce")
        elif pa.types.is_boolean(pa_type):
            # pyarrow string->bool casting is case-insensitive:
            #   "true" or "1" -> True
            #   "false" or "0" -> False
            # Note: BooleanArray was previously used to parse these strings
            #   and allows "1.0" and "0.0". Pyarrow casting does not support
            #   this, but we allow it here.
            if isinstance(strings, (pa.Array, pa.ChunkedArray)):
                scalars = strings
            else:
                scalars = pa.array(strings, type=pa.string(), mask=mask)
            scalars = pc.if_else(pc.equal(scalars, "1.0"), "1", scalars)
            scalars = pc.if_else(pc.equal(scalars, "0.0"), "0", scalars)
            scalars = scalars.cast(pa.bool_())
        elif (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            from pandas.core.tools.numeric import to_numeric

            scalars = to_numeric(strings, errors="raise")
            if isinstance(strings, (pa.Array, pa.ChunkedArray)):
                scalars = strings.cast(pa_type)
            elif mask is not None:
                scalars = pa.array(scalars, mask=mask, type=pa_type)

        else:
            raise NotImplementedError(
                f"Converting strings to {pa_type} is not implemented."
            )
        return cls._from_sequence(scalars, dtype=pa_type, copy=copy)

    def _from_pyarrow_array(self, pa_array):
        # [FUNCTION DOC]
        # HELPER: construction hook for the string mixin and for operation results.
        # Parameter pa_array is a pyarrow array or chunked array. Returns a new instance
        # of the same concrete class wrapping it.
        """
        Construct from the pyarrow array result of an operation, for
        compatibility with ArrowStringArray.
        """
        return type(self)(pa_array)

    def _cast_pointwise_result(self, values) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: converts the result of an element-wise operation (a sequence) into an
        # array, preferring this array's type. An empty input returns an empty copy
        # retaining the dtype. Otherwise it builds an arrow array: for numeric dtypes
        # when NaN is not treated as missing, the values go through an object array with
        # a missing mask (missing marker or None); in other cases pyarrow converts with
        # pandas-style missing handling. If pyarrow raises a value or type error, falls
        # back to generic inference over an object array. Post-processing: a null-typed
        # result of decimal inferred values becomes a small decimal; durations are
        # rebuilt through the sequence constructor and cast to a matching duration unit
        # (same duration type, or the unit of a timestamp, or milliseconds for 64-bit
        # dates, or seconds for 32-bit dates when possible); dates, times, decimals,
        # integers and floats are cast to this array's type where possible (integer and
        # float cast failures are ignored). Finally, for a string dtype: string results
        # are returned as the same class; when the string dtype uses NaN as missing the
        # result falls back to a numpy-based inferred array; otherwise a generic arrow
        # array is returned. Other dtypes return this class wrapping the result.
        if len(values) == 0:
            # Retain our dtype
            return self[:0].copy()

        try:
            if self.dtype.kind in "iufc" and not is_nan_na():
                values = np.asarray(values, dtype=object)
                mask = is_pdna_or_none(values)
                arr = pa.array(values, mask=mask)
            else:
                arr = pa.array(values, from_pandas=True)
        except (ValueError, TypeError):
            # e.g. test_by_column_values_with_same_starting_value with nested
            #  values, one entry of which is an ArrowStringArray
            #  or test_agg_lambda_complex128_dtype_conversion for complex values
            values = np.asarray(values, dtype=object)
            return lib.maybe_convert_objects(values, convert_non_numeric=True)

        if pa.types.is_null(arr.type):
            if lib.infer_dtype(values) == "decimal":
                # GH#62522; the specific decimal precision here is arbitrary
                arr = arr.cast(pa.decimal128(1))
        if pa.types.is_duration(arr.type):
            # workaround for https://github.com/apache/arrow/issues/40620
            result = ArrowExtensionArray._from_sequence(values)
            if pa.types.is_duration(self._pa_array.type):
                result = result.astype(self.dtype)  # type: ignore[assignment]
            elif pa.types.is_timestamp(self._pa_array.type):
                # Try to retain original unit
                new_dtype = ArrowDtype(pa.duration(self._pa_array.type.unit))
                try:
                    result = result.astype(new_dtype)  # type: ignore[assignment]
                except ValueError:
                    pass
            elif pa.types.is_date64(self._pa_array.type):
                # Try to match unit we get on non-pointwise op
                dtype = ArrowDtype(pa.duration("ms"))
                result = result.astype(dtype)  # type: ignore[assignment]
            elif pa.types.is_date(self._pa_array.type):
                # Try to match unit we get on non-pointwise op
                dtype = ArrowDtype(pa.duration("s"))
                result = result.astype(dtype)  # type: ignore[assignment]
            return result

        elif pa.types.is_date(arr.type) and pa.types.is_date(self._pa_array.type):
            arr = arr.cast(self._pa_array.type)
        elif pa.types.is_time(arr.type) and pa.types.is_time(self._pa_array.type):
            arr = arr.cast(self._pa_array.type)
        elif pa.types.is_decimal(arr.type) and pa.types.is_decimal(self._pa_array.type):
            arr = arr.cast(self._pa_array.type)
        elif pa.types.is_integer(arr.type) and pa.types.is_integer(self._pa_array.type):
            try:
                arr = arr.cast(self._pa_array.type)
            except pa.lib.ArrowInvalid:
                # e.g. test_combine_add if we can't cast
                pass
        elif pa.types.is_floating(arr.type) and pa.types.is_floating(
            self._pa_array.type
        ):
            try:
                arr = arr.cast(self._pa_array.type)
            except pa.lib.ArrowInvalid:
                # e.g. test_combine_add if we can't cast
                pass

        if isinstance(self.dtype, StringDtype):
            if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
                # ArrowStringArray preserves dtype.na_value
                return self._from_pyarrow_array(arr)
            if self.dtype.na_value is np.nan:
                # ArrowEA has different semantics, so we return numpy-based
                #  result instead
                values = np.asarray(values, dtype=object)
                return lib.maybe_convert_objects(values, convert_non_numeric=True)
            return ArrowExtensionArray(arr)
        return self._from_pyarrow_array(arr)

    @classmethod
    def _box_pa(
        cls, value, pa_type: pa.DataType | None = None
    ) -> pa.Array | pa.ChunkedArray | pa.Scalar:
        # [FUNCTION DOC]
        # HELPER: class-level boxing dispatcher. Parameters: value (anything) and
        # pa_type (optional target pyarrow type). Returns a pyarrow scalar for scalars
        # (including arrow scalars and non-list-likes) and a pyarrow array or chunked
        # array for list-likes.
        """
        Box value into a pyarrow Array, ChunkedArray or Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray or pa.Scalar
        """
        if isinstance(value, pa.Scalar) or not is_list_like(value):
            return cls._box_pa_scalar(value, pa_type)
        return cls._box_pa_array(value, pa_type)

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        # [FUNCTION DOC]
        # HELPER: class-level conversion of one value to a pyarrow scalar. Parameters:
        # value and optional pa_type. Arrow scalars are used as is. Missing values
        # (except a float NaN when NaN is not treated as missing) become a null scalar
        # of pa_type. Otherwise, as a workaround for a pyarrow bug, timedeltas and
        # timestamps are converted to their integer value in the target unit (inferring
        # a duration or timestamp type, with zone, when no type is given) before
        # creating the scalar. If the resulting scalar's type differs from pa_type it is
        # cast. Returns the scalar.
        """
        Box value into a pyarrow Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Scalar
        """
        if isinstance(value, pa.Scalar):
            pa_scalar = value
        elif isna(value) and not (lib.is_float(value) and not is_nan_na()):
            pa_scalar = pa.scalar(None, type=pa_type)
        else:
            # Workaround https://github.com/apache/arrow/issues/37291
            if isinstance(value, Timedelta):
                if pa_type is None:
                    pa_type = pa.duration(value.unit)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value
            elif isinstance(value, Timestamp):
                if pa_type is None:
                    pa_type = pa.timestamp(value.unit, tz=value.tz)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value

            pa_scalar = pa.scalar(value, type=pa_type)

        if pa_type is not None and pa_scalar.type != pa_type:
            pa_scalar = pa_scalar.cast(pa_type)

        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        # [FUNCTION DOC]
        # HELPER: class-level conversion of a sequence to a pyarrow array or chunked
        # array. Parameters: value (sequence or array), pa_type (optional target type),
        # copy (default false; copy numpy-like inputs, or masked/arrow-convertible
        # inputs, so the result does not alias them). Behavior: numpy-wrapped inputs are
        # unwrapped; instances of this class and arrow arrays are used directly; masked
        # arrays and objects with an arrow conversion method use that conversion.
        # Otherwise numpy-like or list input is handled: large string and binary targets
        # force an object array; duration targets (unless the input is already timedelta
        # or integer numpy data) are converted through the timedelta parser; timestamp
        # targets go through a datetime array (so naive and timezone handling matches
        # non-arrow dtypes), with a masked, writable integer view and are returned
        # immediately. Missing values are detected: when NaN is treated as missing, via
        # numpy missingness (a ragged input falls back to an object array, and multi-
        # dimensional input means no scalars are missing); otherwise, for inputs whose
        # dtype is not integer, unsigned, datetime, timedelta or float, via a mask that
        # excludes NaN and NaT. The pyarrow array is built with that mask; on arrow
        # invalid or type errors it is rebuilt letting pyarrow infer the type (to be
        # cast later). Duration results without a requested type are rebuilt through the
        # timedelta parser, and duration arrays with nulls get their nulls refilled with
        # the dtype's missing value (upstream bug workaround). Finally, if pa_type
        # differs from the built type: dictionary targets are dictionary-encoded then
        # cast if still different; other targets are cast, and a failed cast of string
        # data falls back to the string-parsing constructor; any other failure is re-
        # raised. Returns the arrow array.
        # KNOWN ISSUE: a TODO in the body notes that the string-parsing logic should be
        # moved into this method instead of being delegated to the string-parsing
        # constructor.
        """
        Box value into a pyarrow Array or ChunkedArray.

        Parameters
        ----------
        value : Sequence
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
        value = extract_array(value, extract_numpy=True)
        if isinstance(value, cls):
            pa_array = value._pa_array
        elif isinstance(value, (pa.Array, pa.ChunkedArray)):
            pa_array = value
        elif isinstance(value, BaseMaskedArray):
            # GH 52625
            if copy:
                value = value.copy()
            pa_array = value.__arrow_array__()

        elif hasattr(value, "__arrow_array__"):
            # e.g. StringArray
            if copy:
                value = value.copy()
            pa_array = value.__arrow_array__()

        else:
            if (
                isinstance(value, np.ndarray)
                and pa_type is not None
                and (
                    pa.types.is_large_binary(pa_type)
                    or pa.types.is_large_string(pa_type)
                )
            ):
                # See https://github.com/apache/arrow/issues/35289
                value = np.asarray(value, dtype=object)
            elif copy and is_array_like(value):
                # pa array should not get updated when numpy array is updated
                value = value.copy()

            if (
                pa_type is not None
                and pa.types.is_duration(pa_type)
                and (not isinstance(value, np.ndarray) or value.dtype.kind not in "mi")
            ):
                # Workaround https://github.com/apache/arrow/issues/37291
                from pandas.core.tools.timedeltas import to_timedelta

                value = to_timedelta(value, unit=pa_type.unit).as_unit(pa_type.unit)
                value = value.to_numpy()

            if pa_type is not None and pa.types.is_timestamp(pa_type):
                # Use DatetimeArray to exclude Decimal(NaN) (GH#61774) and
                #  ensure constructor treats tznaive the same as non-pyarrow
                #  dtypes (GH#61775)
                from pandas.core.arrays.datetimes import (
                    DatetimeArray,
                    tz_to_dtype,
                )

                pass_dtype = tz_to_dtype(tz=pa_type.tz, unit=pa_type.unit)
                value = extract_array(value, extract_numpy=True)
                if isinstance(value, DatetimeArray):
                    dta = value
                else:
                    dta = DatetimeArray._from_sequence(
                        value, copy=copy, dtype=pass_dtype
                    )
                dta_mask = dta.isna()
                value_i8 = cast("npt.NDArray", dta.view("i8"))
                if not value_i8.flags["WRITEABLE"]:
                    # e.g. test_setitem_frame_2d_values
                    value_i8 = value_i8.copy()
                    dta = DatetimeArray._from_sequence(value_i8, dtype=dta.dtype)
                value_i8[dta_mask] = 0  # GH#61776 avoid __sub__ overflow
                pa_array = pa.array(dta._ndarray, type=pa_type, mask=dta_mask)
                return pa_array

            mask = None
            if is_nan_na():
                try:
                    arr_value = np.asarray(value)
                    if arr_value.ndim > 1:
                        # e.g. test_fixed_size_list we have list data.  ndim > 1
                        #  means there were no scalar (NA) entries.
                        mask = np.zeros(len(value), dtype=np.bool_)
                    else:
                        mask = isna(arr_value)
                except ValueError:
                    # Ragged data that numpy raises on
                    arr_value = construct_1d_object_array_from_listlike(value)
                    mask = isna(arr_value)
            elif (
                getattr(value, "dtype", None) is None or value.dtype.kind not in "iumMf"
            ):
                arr_value = np.asarray(value, dtype=object)
                # similar to isna(value) but exclude NaN, NaT, nat-like, nan-like
                mask = is_pdna_or_none(arr_value)

            try:
                pa_array = pa.array(value, type=pa_type, mask=mask)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                # GH50430: let pyarrow infer type, then cast
                pa_array = pa.array(value, mask=mask)

            if pa_type is None and pa.types.is_duration(pa_array.type):
                # Workaround https://github.com/apache/arrow/issues/37291
                from pandas.core.tools.timedeltas import to_timedelta

                value = to_timedelta(value)
                value = value.to_numpy()
                pa_array = pa.array(value, type=pa_type)

            if pa.types.is_duration(pa_array.type) and pa_array.null_count > 0:
                # GH52843: upstream bug for duration types when originally
                # constructed with data containing numpy NaT.
                # https://github.com/apache/arrow/issues/35088
                arr = cls(pa_array)
                arr = arr.fillna(arr.dtype.na_value)
                pa_array = arr._pa_array

        if pa_type is not None and pa_array.type != pa_type:
            if pa.types.is_dictionary(pa_type):
                pa_array = pa_array.dictionary_encode()
                if pa_array.type != pa_type:
                    pa_array = pa_array.cast(pa_type)
            else:
                try:
                    pa_array = pa_array.cast(pa_type)
                except (pa.ArrowNotImplementedError, pa.ArrowTypeError):
                    if pa.types.is_string(pa_array.type) or pa.types.is_large_string(
                        pa_array.type
                    ):
                        # TODO: Move logic in _from_sequence_of_strings into
                        # _box_pa_array
                        dtype = ArrowDtype(pa_type)
                        return cls._from_sequence_of_strings(
                            value, dtype=dtype
                        )._pa_array
                    else:
                        raise

        return pa_array

    def __getitem__(self, item: PositionalIndexer):
        # [FUNCTION DOC]
        # Item access. Parameter item is an integer, slice, Ellipsis, tuple (single-
        # element tuples and ellipses are unpacked), integer array/list or boolean
        # array. The indexer is validated first. An empty array indexer returns an empty
        # array (of the proper string storage type for the pyarrow-storage string
        # dtype). Integer arrays go through take, boolean arrays filter the chunked
        # array, other array kinds raise an index error. A non-integer scalar raises an
        # index error with the numpy-style message. For slices, a workaround avoids a
        # pyarrow bug with a stop before the start for negative steps. The backing array
        # is indexed; chunked results are wrapped in a new array that inherits the read-
        # only flag when the indexing is a view. For a scalar, missing returns the
        # dtype's missing value; timestamps and durations with non-nanosecond units are
        # returned as pandas scalars of that unit; anything else is returned as a Python
        # object.
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if not len(item):
                # Removable once we migrate StringDtype[pyarrow] to ArrowDtype[string]
                if (
                    isinstance(self._dtype, StringDtype)
                    and self._dtype.storage == "pyarrow"
                ):
                    # TODO(infer_string) should this be large_string?
                    pa_dtype = pa.string()
                else:
                    pa_dtype = self._dtype.pyarrow_dtype
                result = pa.chunked_array([], type=pa_dtype)
                return self._from_pyarrow_array(result)

            elif item.dtype.kind in "iu":
                return self.take(item)
            elif item.dtype.kind == "b":
                return self._from_pyarrow_array(self._pa_array.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if item is Ellipsis:
            # TODO: should be handled by pyarrow?
            item = slice(None)

        if is_scalar(item) and not is_integer(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        # We are not an array indexer, so maybe e.g. a slice or integer
        # indexer. We dispatch to pyarrow.
        if isinstance(item, slice):
            # Arrow bug https://github.com/apache/arrow/issues/38768
            if item.start == item.stop:
                pass
            elif (
                item.stop is not None
                and item.stop < -len(self)
                and item.step is not None
                and item.step < 0
            ):
                item = slice(item.start, None, item.step)

        value = self._pa_array[item]
        if isinstance(value, pa.ChunkedArray):
            result = self._from_pyarrow_array(value)
            if getitem_returns_view(self, item):
                result._readonly = self._readonly
            return result
        else:
            pa_type = self._pa_array.type
            scalar = value.as_py()
            if scalar is None:
                return self._dtype.na_value
            elif pa.types.is_timestamp(pa_type) and pa_type.unit != "ns":
                # GH 53326
                return Timestamp(scalar).as_unit(pa_type.unit)
            elif pa.types.is_duration(pa_type) and pa_type.unit != "ns":
                # GH 53326
                return Timedelta(scalar).as_unit(pa_type.unit)
            else:
                return scalar

    def __iter__(self) -> Iterator[Any]:
        # [FUNCTION DOC]
        # Generator over the elements. Yields the dtype's missing value for nulls,
        # pandas timestamp or timedelta scalars with the proper unit for non-nanosecond
        # timestamp or duration types, and Python scalars for everything else.
        """
        Iterate over elements of the array.
        """
        na_value = self._dtype.na_value
        # GH 53326
        pa_type = self._pa_array.type
        box_timestamp = pa.types.is_timestamp(pa_type) and pa_type.unit != "ns"
        box_timedelta = pa.types.is_duration(pa_type) and pa_type.unit != "ns"
        for value in self._pa_array:
            val = value.as_py()
            if val is None:
                yield na_value
            elif box_timestamp:
                yield Timestamp(val).as_unit(pa_type.unit)
            elif box_timedelta:
                yield Timedelta(val).as_unit(pa_type.unit)
            else:
                yield val

    def __arrow_array__(self, type=None):
        # [FUNCTION DOC]
        # Pyarrow conversion protocol. Parameter type is ignored. Returns the backing
        # chunked array itself (no copy).
        """Convert myself to a pyarrow ChunkedArray."""
        return self._pa_array

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy ufunc hook. Parameters as in the base class. Calls the base
        # implementation and, when the instance is exactly this class (not the string
        # subclass), wraps the result into a new array through the sequence constructor
        # so numpy outputs become arrow-backed (GH#62800); otherwise returns the base
        # result.
        # Need to wrap np.array results GH#62800
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if type(self) is ArrowExtensionArray:
            # Exclude ArrowStringArray
            return type(self)._from_sequence(result)
        return result

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Numpy conversion protocol. Parameters: dtype (optional) and copy (None or
        # bool). Raises a value error if copy is false (a no-copy conversion cannot be
        # guaranteed); copy of None is treated as false for the conversion method.
        # Returns the result of converting to numpy with that dtype and copy flag.
        # KNOWN ISSUE: a TODO in the body notes that the zero-copy-only option could be
        # used to honor a request to avoid copies.
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
        if copy is False:
            # TODO: By using `zero_copy_only` it may be possible to implement this
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        elif copy is None:
            # `to_numpy(copy=False)` has the meaning of NumPy `copy=None`.
            copy = False

        return self.to_numpy(dtype=dtype, copy=copy)

    def __invert__(self) -> Self:
        # [FUNCTION DOC]
        # Bitwise or logical negation. Integer types use bitwise not; string and large
        # string types raise a type error; other types (booleans) use logical invert.
        # Returns a new array.
        # This is a bit wise op for integer types
        if pa.types.is_integer(self._pa_array.type):
            return self._from_pyarrow_array(pc.bit_wise_not(self._pa_array))
        elif pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            # Raise TypeError instead of pa.ArrowNotImplementedError
            raise TypeError("__invert__ is not supported for string dtypes")
        else:
            return self._from_pyarrow_array(pc.invert(self._pa_array))

    def __neg__(self) -> Self:
        # [FUNCTION DOC]
        # Unary minus using the overflow-checked kernel. Raises a type error naming the
        # dtype if the kernel does not support it. Returns a new array.
        try:
            return self._from_pyarrow_array(pc.negate_checked(self._pa_array))
        except pa.ArrowNotImplementedError as err:
            raise TypeError(
                f"unary '-' not supported for dtype '{self.dtype}'"
            ) from err

    def __pos__(self) -> Self:
        # [FUNCTION DOC]
        # Unary plus: returns a new array wrapping the same backing data.
        return self._from_pyarrow_array(self._pa_array)

    def __abs__(self) -> Self:
        # [FUNCTION DOC]
        # Absolute value using the overflow-checked kernel. Returns a new array.
        return self._from_pyarrow_array(pc.abs_checked(self._pa_array))

    # GH 42600: __getstate__/__setstate__ not necessary once
    # https://issues.apache.org/jira/browse/ARROW-10739 is addressed
    def __getstate__(self):
        # [FUNCTION DOC]
        # Pickle support. Returns a copy of the instance dictionary in which the backing
        # chunked array is replaced by its chunks combined into a single array (works
        # around GH 42600, which a pyarrow fix would make unnecessary).
        state = self.__dict__.copy()
        state["_pa_array"] = self._pa_array.combine_chunks()
        return state

    def __setstate__(self, state) -> None:
        # [FUNCTION DOC]
        # Pickle support. Parameter state is the dictionary from the pickle. Takes the
        # data from the legacy data key if present (removing it), otherwise from the
        # backing array key, wraps it as a chunked array, stores it in the state and
        # updates the instance dictionary. Mutates the instance.
        if "_data" in state:
            data = state.pop("_data")
        else:
            data = state["_pa_array"]
        state["_pa_array"] = pa.chunked_array(data)
        self.__dict__.update(state)

    def _cmp_method(self, other, op) -> ArrowExtensionArray:
        # [FUNCTION DOC]
        # HELPER: element-wise comparison (called by the operator mixin). Parameters:
        # other and op, a comparison operator function whose name selects the arrow
        # kernel. For extension arrays, numpy arrays, lists and ranges: the other side
        # is boxed; if boxing fails with an arrow invalid-data error the comparison is
        # done pointwise in Python. Timestamp versus date comparisons, and any range,
        # use the generic invalid-comparison fallback to match non-arrow behavior; if a
        # kernel is not implemented the same fallback is used. For scalars: date versus
        # datetime (and the reverse) use the invalid-comparison fallback; otherwise the
        # kernel is tried with the boxed scalar and, if arrow cannot handle it, the
        # comparison is done on valid positions with numpy (falling back to the invalid-
        # comparison result on type error), then missing positions are nulled. Any other
        # operand type raises a not-implemented error. Returns a boolean arrow-backed
        # array.
        pc_func = ARROW_CMP_FUNCS[op.__name__]
        ltype = self._pa_array.type

        if isinstance(other, (ExtensionArray, np.ndarray, list, range)):
            try:
                boxed = self._box_pa(other)
            except pa.lib.ArrowInvalid:
                # e.g. GH#60228 [1, "b"] we have to operate pointwise
                res_values = [op(x, y) for x, y in zip(self, other, strict=True)]
                result = pa.array(res_values, type=pa.bool_(), from_pandas=True)
            else:
                rtype = boxed.type
                if (
                    (pa.types.is_timestamp(ltype) and pa.types.is_date(rtype))
                    or (pa.types.is_timestamp(rtype) and pa.types.is_date(ltype))
                    or isinstance(other, range)
                ):
                    # GH#62157 match non-pyarrow behavior
                    result = ops.invalid_comparison(self, other, op)
                    result = pa.array(result, type=pa.bool_())
                else:
                    try:
                        result = pc_func(self._pa_array, boxed)
                    except pa.ArrowNotImplementedError:
                        result = ops.invalid_comparison(self, other, op)
                        result = pa.array(result, type=pa.bool_())

        elif is_scalar(other):
            if (isinstance(other, datetime) and pa.types.is_date(ltype)) or (
                type(other) is date and pa.types.is_timestamp(ltype)
            ):
                # GH#62157 match non-pyarrow behavior
                result = ops.invalid_comparison(self, other, op)
                result = pa.array(result, type=pa.bool_())
            else:
                try:
                    result = pc_func(self._pa_array, self._box_pa(other))
                except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
                    mask = isna(self) | isna(other)
                    valid = ~mask
                    result = np.zeros(len(self), dtype="bool")
                    np_array = np.array(self)
                    try:
                        result[valid] = op(np_array[valid], other)
                    except TypeError:
                        result = ops.invalid_comparison(self, other, op)
                    result = pa.array(result, type=pa.bool_())
                    result = pc.if_else(valid, result, None)
        else:
            raise NotImplementedError(
                f"{op.__name__} not implemented for {type(other)}"
            )
        return ArrowExtensionArray(result)

    def _op_method_error_message(self, other, op) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the error text for unsupported operations: names the operator,
        # this array's dtype and the other operand, described by its dtype if it has one
        # and by its type otherwise. Returns the string.
        if hasattr(other, "dtype"):
            other_type = f"dtype '{other.dtype}'"
        else:
            other_type = f"object of type {type(other)}"
        return (
            f"operation '{op.__name__}' not supported for "
            f"dtype '{self.dtype}' with {other_type}"
        )

    def _evaluate_op_method(self, other, op, arrow_funcs) -> Self:
        # [FUNCTION DOC]
        # HELPER: shared engine for arithmetic and bitwise/logical operators.
        # Parameters: other (operand, boxed to arrow), op (operator function) and
        # arrow_funcs (table mapping operator names to kernels). For string/binary
        # arrays: addition and reflected addition concatenate through the element-wise
        # join kernel (mixing string and large string by casting to large string, typing
        # null scalars; unsupported types raise a type error); multiplication and its
        # reflected form repeat the string by a non-negative integer (negative counts
        # are treated as zero; a non-integer count raises a type error). The same
        # repetition applies when the other operand is a string and this array holds
        # integers. For logical operators with a null scalar, the null is cast to this
        # type because kleene kernels need typed nulls. The table entry must exist; a
        # placeholder entry raises a type error for strings and a not-implemented error
        # otherwise. Kernel not-implemented errors become type errors with the formatted
        # message. Returns a new array.
        pa_type = self._pa_array.type
        other_original = other
        other = self._box_pa(other)

        if (
            pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
            or pa.types.is_binary(pa_type)
        ):
            if op in [operator.add, roperator.radd]:
                # binary_join_element_wise does not support mixed types, but we
                # want to allow addition between string and large_string types
                self_array = self._pa_array
                if pa.types.is_string(pa_type) and pa.types.is_large_string(other.type):
                    self_array = self._pa_array.cast(pa.large_string())
                elif pa.types.is_large_string(pa_type) and pa.types.is_string(
                    other.type
                ):
                    other = other.cast(pa.large_string())

                sep = pa.scalar("", type=self_array.type)
                if isinstance(other, pa.Scalar) and pc.is_null(other).as_py():
                    other = other.cast(self_array.type)
                try:
                    if op is operator.add:
                        result = pc.binary_join_element_wise(self_array, other, sep)
                    elif op is roperator.radd:
                        result = pc.binary_join_element_wise(other, self_array, sep)
                except pa.ArrowNotImplementedError as err:
                    raise TypeError(
                        self._op_method_error_message(other_original, op)
                    ) from err
                return self._from_pyarrow_array(result)
            elif op in [operator.mul, roperator.rmul]:
                binary = self._pa_array
                integral = other
                if not pa.types.is_integer(integral.type):
                    raise TypeError("Can only string multiply by an integer.")
                pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
                result = pc.binary_repeat(binary, pa_integral)
                return self._from_pyarrow_array(result)
        elif (
            pa.types.is_string(other.type)
            or pa.types.is_binary(other.type)
            or pa.types.is_large_string(other.type)
        ) and op in [operator.mul, roperator.rmul]:
            binary = other
            integral = self._pa_array
            if not pa.types.is_integer(integral.type):
                raise TypeError("Can only string multiply by an integer.")
            pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
            result = pc.binary_repeat(binary, pa_integral)
            return self._from_pyarrow_array(result)
        if (
            isinstance(other, pa.Scalar)
            and pc.is_null(other).as_py()
            and op.__name__ in ARROW_LOGICAL_FUNCS
        ):
            # pyarrow kleene ops require null to be typed
            other = other.cast(pa_type)

        pc_func = arrow_funcs[op.__name__]
        if pc_func is NotImplemented:
            if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
                raise TypeError(self._op_method_error_message(other_original, op))
            raise NotImplementedError(f"{op.__name__} not implemented.")

        try:
            result = pc_func(self._pa_array, other)
        except pa.ArrowNotImplementedError as err:
            raise TypeError(self._op_method_error_message(other_original, op)) from err
        return self._from_pyarrow_array(result)

    def _logical_method(self, other, op) -> Self:
        # [FUNCTION DOC]
        # HELPER: logical operators (and, or, xor and their reflected forms). Integer
        # types use the bitwise table. String arrays combined, in reflected form, with a
        # numpy boolean array keep the legacy behavior: emit a deprecation warning
        # (Pandas4Warning) that boolean/string operations will raise, convert self to
        # boolean and apply the operator. All other cases use the Kleene logical table.
        # Returns a new array.
        # For integer types `^`, `|`, `&` are bitwise operators and return
        # integer types. Otherwise these are boolean ops.
        if pa.types.is_integer(self._pa_array.type):
            return self._evaluate_op_method(other, op, ARROW_BIT_WISE_FUNCS)
        elif (
            (
                pa.types.is_string(self._pa_array.type)
                or pa.types.is_large_string(self._pa_array.type)
            )
            and op in (roperator.ror_, roperator.rand_, roperator.rxor)
            and isinstance(other, np.ndarray)
            and other.dtype == bool
        ):
            # GH#60234 backward compatibility for the move to StringDtype in 3.0
            op_name = op.__name__[1:].strip("_")
            warnings.warn(
                f"'{op_name}' operations between boolean dtype and {self.dtype} are "
                "deprecated and will raise in a future version. Explicitly "
                "cast the strings to a boolean dtype before operating instead.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )
            return op(other, self.astype(bool))
        else:
            return self._evaluate_op_method(other, op, ARROW_LOGICAL_FUNCS)

    def _str_arith_method_object_fallback(
        self, other, op
    ) -> Self | npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER: object-dtype fallback for string arithmetic that arrow cannot do.
        # Parameters: other (scalar or list-like) and op. Raises a value error when a
        # list-like other has a different length. Missing positions (in either operand)
        # are filled with the dtype's missing value; the operator is applied to the
        # valid positions on object data. If the result is entirely strings (ignoring
        # missing) it is converted back to this array type with this dtype; otherwise
        # the object numpy array is returned.
        mask = isna(self) | isna(other)
        valid = ~mask

        if is_list_like(other):
            if len(other) != len(self):
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )
            if not is_array_like(other):
                other = np.asarray(other)
            other = other[valid]

        result = np.empty(len(self), dtype=object)
        result[mask] = self.dtype.na_value
        result[valid] = op(np.asarray(self, dtype=object)[valid], other)

        if not lib.is_string_array(result, skipna=True):
            return result
        return type(self)._from_sequence(result, dtype=self.dtype)

    def _arith_method(self, other, op) -> Self | npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER: arithmetic operators (called by the operator mixin). Parameters: other
        # and op. For string arrays, tries the arrow path and, on arrow invalid-data or
        # type errors, falls back to the object path; other types use the arrow path
        # directly. A numpy result is returned as is. When NaN is treated as missing and
        # the result is floating, NaN entries are replaced with nulls. Returns an array
        # or a numpy object array.
        result: Self | npt.NDArray[np.object_]
        if pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            try:
                result = self._evaluate_op_method(other, op, ARROW_ARITHMETIC_FUNCS)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                result = self._str_arith_method_object_fallback(other, op)
        else:
            result = self._evaluate_op_method(other, op, ARROW_ARITHMETIC_FUNCS)
        if isinstance(result, np.ndarray):
            return result
        if is_nan_na() and result.dtype.kind == "f":
            parr = result._pa_array
            mask = pc.is_nan(parr).fill_null(False).to_numpy()
            arr = pc.replace_with_mask(parr, mask, pa.scalar(None, type=parr.type))
            result = type(self)(arr)
        return result

    def equals(self, other) -> bool:
        # [FUNCTION DOC]
        # Equality of two arrays as a whole. Returns false if other is not an instance
        # of this class; otherwise compares the two backing chunked arrays with arrow's
        # equality check.
        # KNOWN ISSUE: the comment in the body expresses uncertainty ("I'm told") that
        # the arrow comparison mirrors the pandas equals semantics; this is unverified.
        if not isinstance(other, ArrowExtensionArray):
            return False
        # I'm told that pyarrow makes __eq__ behave like pandas' equals;
        #  TODO: is this documented somewhere?
        return self._pa_array == other._pa_array

    @property
    def dtype(self) -> ArrowDtype:
        # [FUNCTION DOC]
        # Property: the arrow dtype of this array (the stored attribute).
        """
        An instance of 'ExtensionDtype'.
        """
        return self._dtype

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: bytes used by the backing arrow data, as reported by pyarrow.
        """
        The number of bytes needed to store this object in memory.
        """
        return self._pa_array.nbytes

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Number of elements in the backing array.
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._pa_array)

    def __contains__(self, key) -> bool:
        # [FUNCTION DOC]
        # Membership test. A missing-value key that is not the dtype's own missing
        # marker is handled specially: a float NaN when NaN is treated as missing checks
        # for the dtype's missing marker instead; for float dtypes, a float key checks
        # for any NaN in the data; all other such keys return false (None and NaN-like
        # values do not match pd.NA). Everything else uses the base class membership
        # test, converted to a bool.
        # https://github.com/pandas-dev/pandas/pull/51307#issuecomment-1426372604
        if isna(key) and key is not self.dtype.na_value:
            if lib.is_float(key) and is_nan_na():
                return self.dtype.na_value in self
            elif self.dtype.kind == "f" and lib.is_float(key):
                # Check specifically for NaN
                return pc.any(pc.is_nan(self._pa_array)).as_py()

            # e.g. date or timestamp types we do not allow None here to match pd.NA
            return False
            # TODO: maybe complex? object?

        return bool(super().__contains__(key))

    @property
    def _hasna(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property, true when the backing array has any nulls (uses the cached
        # null count).
        return self._pa_array.null_count > 0

    def isna(self) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Boolean numpy array marking nulls. Fast paths: no nulls gives all false; all
        # nulls gives all true. Otherwise converts arrow's null mask to numpy.
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        # GH51630: fast paths
        null_count = self._pa_array.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=np.bool_)
        elif null_count == len(self):
            return np.ones(len(self), dtype=np.bool_)

        return self._pa_array.is_null().to_numpy()

    @overload
    def any(self, *, skipna: Literal[True] = ..., **kwargs) -> bool:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def any(self, *, skipna: bool, **kwargs) -> bool | NAType:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def any(self, *, skipna: bool = True, **kwargs) -> bool | NAType:
        # [FUNCTION DOC]
        # Whether any element is truthy. Keyword parameters: skipna (default true) and
        # extra keywords passed through. Delegates to the reduction dispatcher with the
        # any operation; with skipna false, uses Kleene logic so the result may be
        # missing. Returns a bool or the missing marker.
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

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        """
        return self._reduce("any", skipna=skipna, **kwargs)

    @overload
    def all(self, *, skipna: Literal[True] = ..., **kwargs) -> bool:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def all(self, *, skipna: bool, **kwargs) -> bool | NAType:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def all(self, *, skipna: bool = True, **kwargs) -> bool | NAType:
        # [FUNCTION DOC]
        # Whether all elements are truthy. Keyword parameters: skipna (default true) and
        # extra keywords passed through. Delegates to the reduction dispatcher with the
        # all operation; with skipna false, uses Kleene logic so the result may be
        # missing. Returns a bool or the missing marker.
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

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        """
        return self._reduce("all", skipna=skipna, **kwargs)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Indices that would sort the array, via the arrow sort-indices kernel. Keyword
        # parameters: ascending (default true); kind and extra keywords are accepted but
        # ignored; na_position (last, default, or first). Raises a value error for any
        # other na_position. Returns an integer index array of the platform index type.
        order = "ascending" if ascending else "descending"
        null_placement = {"last": "at_end", "first": "at_start"}.get(na_position, None)
        if null_placement is None:
            raise ValueError(f"invalid na_position: {na_position}")

        result = pc.array_sort_indices(
            self._pa_array, order=order, null_placement=null_placement
        )
        np_result = result.to_numpy()
        return np_result.astype(np.intp, copy=False)

    def _argmin_max(self, skipna: bool, method: str) -> int:
        # [FUNCTION DOC]
        # HELPER: shared implementation of argmin and argmax. Parameters: skipna (bool)
        # and method (min or max). For an empty array, an all-missing array, or missing
        # values with skipna false it delegates to the base class method (so the base
        # raises the expected error). Otherwise finds the extreme value with the arrow
        # kernel (durations are cast to 64-bit integers first) and returns the position
        # of its first occurrence.
        # KNOWN ISSUE: the comment says that for skipna false with missing data pandas
        # expects a not-implemented error, but the base class method that is called
        # raises a value error.
        if self._pa_array.length() in (0, self._pa_array.null_count) or (
            self._hasna and not skipna
        ):
            # For empty or all null, pyarrow returns -1 but pandas expects TypeError
            # For skipna=False and data w/ null, pandas expects NotImplementedError
            # let ExtensionArray.arg{max|min} raise
            return getattr(super(), f"arg{method}")(skipna=skipna)

        data = self._pa_array
        if pa.types.is_duration(data.type):
            data = data.cast(pa.int64())

        value = getattr(pc, method)(data, skip_nulls=skipna)
        return pc.index(data, value).as_py()

    def argmin(self, skipna: bool = True) -> int:
        # [FUNCTION DOC]
        # Position of the minimum. Parameter skipna (default true). Delegates to the
        # shared implementation with the min method.
        return self._argmin_max(skipna, "min")

    def argmax(self, skipna: bool = True) -> int:
        # [FUNCTION DOC]
        # Position of the maximum. Parameter skipna (default true). Delegates to the
        # shared implementation with the max method.
        return self._argmin_max(skipna, "max")

    def copy(self) -> Self:
        # [FUNCTION DOC]
        # Shallow copy: a new array wrapping the same (immutable) backing chunked array,
        # so no data is duplicated.
        """
        Return a shallow copy of the array.

        Underlying ChunkedArray is immutable, so a deep copy is unnecessary.

        Returns
        -------
        type(self)
        """
        return self._from_pyarrow_array(self._pa_array)

    def dropna(self) -> Self:
        # [FUNCTION DOC]
        # New array without the null entries, using the arrow drop-null kernel.
        """
        Return ArrowExtensionArray without NA values.

        Returns
        -------
        ArrowExtensionArray
        """
        return self._from_pyarrow_array(pc.drop_null(self._pa_array))

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: forward or backward fill. Keyword parameters as in the base class:
        # method, limit, limit_area, copy. Returns self when nothing is missing. When
        # both limit and limit_area are unset, tries the arrow forward/backward fill
        # kernel (some types such as durations lack a kernel and fall through).
        # Otherwise, or on failure, delegates to the base class implementation.
        # KNOWN ISSUE: two TODO comments in the body (why the fast cases no longer
        # suffice, and removing the method after a deprecation is enforced) show the
        # method's necessity is unclear.
        if not self._hasna:
            return self

        if limit is None and limit_area is None:
            method = missing.clean_fill_method(method)
            try:
                if method == "pad":
                    return self._from_pyarrow_array(
                        pc.fill_null_forward(self._pa_array)
                    )
                elif method == "backfill":
                    return self._from_pyarrow_array(
                        pc.fill_null_backward(self._pa_array)
                    )
            except pa.ArrowNotImplementedError:
                # ArrowNotImplementedError: Function 'coalesce' has no kernel
                #   matching input types (duration[ns], duration[ns])
                # TODO: remove try/except wrapper if/when pyarrow implements
                #   a kernel for duration types.
                pass

        # TODO: Why do we no longer need the above cases?
        # TODO(3.0): after EA.fillna 'method' deprecation is enforced, we can remove
        #  this method entirely.
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    @doc(ExtensionArray.fillna)
    def fillna(
        self,
        value: object | ArrayLike,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill missing values. Parameters: value (scalar, array-like), limit (default
        # None) and copy (default true). Returns a copy when nothing is missing, and
        # delegates to the base implementation when a limit is given. Array-like values
        # must have the same length as this array (value error otherwise). The value is
        # boxed to this array's arrow type (type errors from arrow become a type error
        # naming the value and dtype), and the data is filled with the arrow fill-null
        # helper; if the kernel does not support the type (for example durations) the
        # base implementation is used. Returns the filled array. Docstring is inherited
        # from the base class through a decorator.
        if not self._hasna:
            return self.copy()

        if limit is not None:
            return super().fillna(value=value, limit=limit, copy=copy)

        if isinstance(value, (np.ndarray, ExtensionArray)):
            # Similar to check_value_size, but we do not mask here since we may
            #  end up passing it to the super() method.
            if len(value) != len(self):
                raise ValueError(
                    f"Length of 'value' does not match. Got ({len(value)}) "
                    f" expected {len(self)}"
                )

        try:
            fill_value = self._box_pa(value, pa_type=self._pa_array.type)
        except pa.ArrowTypeError as err:
            msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
            raise TypeError(msg) from err

        try:
            return self._from_pyarrow_array(
                _safe_fill_null(self._pa_array, fill_value=fill_value)
            )
        except pa.ArrowNotImplementedError:
            # ArrowNotImplementedError: Function 'coalesce' has no kernel
            #   matching input types (duration[ns], duration[ns])
            # TODO: remove try/except wrapper if/when pyarrow implements
            #   a kernel for duration types.
            pass

        return super().fillna(value=value, limit=limit, copy=copy)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Membership test. Parameter values is a list-like of values. An empty values
        # input returns an all-false array. Otherwise the values are boxed to arrow and
        # the arrow membership kernel is used; the result is converted to a numpy bool
        # array (nulls from old pyarrow versions become false).
        # short-circuit to return all False array.
        if not len(values):
            return np.zeros(len(self), dtype=bool)

        value_set = self._box_pa(values)
        result = pc.is_in(self._pa_array, value_set=value_set)
        # pyarrow 2.0.0 returned nulls, so we explicitly specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        # [FUNCTION DOC]
        # HELPER: returns the pair (numpy array of the data, dtype missing value), used
        # by factorization and hashing. The conversion uses arrow's numpy conversion of
        # the chunked array.
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
        na_value : pd.NA

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`.
        """
        values = self._pa_array.to_numpy()
        return values, self.dtype.na_value

    @doc(ExtensionArray.factorize)
    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        # [FUNCTION DOC]
        # Encode as integer codes and unique values. Parameter use_na_sentinel (default
        # true): nulls are masked and coded -1 (true) or encoded as a regular value
        # (false). Uses arrow dictionary encoding with the corresponding null handling;
        # already dictionary-typed input is cast back to its value type before encoding
        # when nulls must be encoded. Empty input returns an empty intp code array and
        # empty uniques. Otherwise the chunks are combined, codes (with any null code
        # filled by -1) are converted to a writable numpy array of the platform index
        # type, and the dictionary becomes the uniques array. Returns (codes, uniques).
        # Docstring is inherited from the base class through a decorator.
        null_encoding = "mask" if use_na_sentinel else "encode"

        data = self._pa_array

        if pa.types.is_dictionary(data.type):
            if null_encoding == "encode":
                # dictionary encode does nothing if an already encoded array is given
                data = data.cast(data.type.value_type)
                encoded = data.dictionary_encode(null_encoding=null_encoding)
            else:
                encoded = data
        else:
            encoded = data.dictionary_encode(null_encoding=null_encoding)
        if encoded.length() == 0:
            indices = np.array([], dtype=np.intp)
            uniques = self._from_pyarrow_array(
                pa.chunked_array([], type=encoded.type.value_type)
            )
        else:
            # GH 54844
            combined = encoded.combine_chunks()
            pa_indices = combined.indices
            if pa_indices.null_count > 0:
                pa_indices = _safe_fill_null(pa_indices, -1)
            indices = pa_indices.to_numpy(zero_copy_only=False, writable=True).astype(
                np.intp, copy=False
            )
            uniques = self._from_pyarrow_array(combined.dictionary)

        return indices, uniques

    def reshape(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Reshaping is not supported. Accepts any arguments and always raises a not-
        # implemented error saying the array is backed by a one-dimensional chunked
        # array.
        raise NotImplementedError(
            f"{type(self)} does not support reshape "
            f"as backed by a 1D pyarrow.ChunkedArray."
        )

    def round(self, decimals: int = 0, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Round values. Parameters: decimals (default 0; negative rounds left of the
        # decimal point); extra positional and keyword arguments are accepted and
        # ignored. Uses the arrow rounding kernel. Returns a new array.
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect.

        Returns
        -------
        ArrowExtensionArray
            Rounded values of the ArrowExtensionArray.

        See Also
        --------
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        return self._from_pyarrow_array(pc.round(self._pa_array, ndigits=decimals))

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Insertion positions that would keep the sorted array ordered. Parameters:
        # value, side (left default or right), sorter. Raises a value error if the array
        # has nulls. Extension-array values are cast to object. The search is run on the
        # numpy conversion of this array; for nanosecond timestamps and durations the
        # conversion uses object dtype to work around a numpy failure. Returns a
        # position or an array of positions. Docstring is inherited from the base class
        # through a decorator.
        if self._hasna:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        # Base class searchsorted would cast to object, which is *much* slower.
        dtype = None
        if isinstance(self.dtype, ArrowDtype):
            pa_dtype = self.dtype.pyarrow_dtype
            if (
                pa.types.is_timestamp(pa_dtype) or pa.types.is_duration(pa_dtype)
            ) and pa_dtype.unit == "ns":
                # np.array[datetime/timedelta].searchsorted(datetime/timedelta)
                # erroneously fails when numpy type resolution is nanoseconds
                dtype = object
        return self.to_numpy(dtype=dtype).searchsorted(value, side=side, sorter=sorter)

    def take(
        self,
        indices: TakeIndexer,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> ArrowExtensionArray:
        # [FUNCTION DOC]
        # Take by position. Parameters: indices (sequence or integer array), allow_fill
        # (default false), fill_value (default None). Raises an index error for a non-
        # empty take from an empty array, or an index at or beyond the length. With
        # allow_fill: negative indices mark positions to fill; they are validated (only
        # -1 is allowed) and taken as nulls; if fill_value is missing the nulls remain,
        # otherwise those positions are assigned the fill value afterwards through item
        # assignment. With no negative indices it takes directly. Without allow_fill
        # negative indices count from the end (adjusted on a copy). Returns a new array.
        # KNOWN ISSUE: unlike the base class, allow_fill is not keyword-only here, and
        # the fast path for nothing to fill takes with the original indices rather than
        # the converted array.
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        api.extensions.take

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.
        """
        indices_array = np.asanyarray(indices)

        if len(self._pa_array) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._pa_array):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                validate_indices(indices_array, len(self._pa_array))
                # TODO(ARROW-9433): Treat negative indices as NULL
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._pa_array.take(indices_array)
                if isna(fill_value):
                    return self._from_pyarrow_array(result)
                # TODO: ArrowNotImplementedError: Function fill_null has no
                # kernel matching input types (array[string], scalar[string])
                result = self._from_pyarrow_array(result)
                result[fill_mask] = fill_value
                return result
                # return type(self)(pc.fill_null(result, pa.scalar(fill_value)))
            else:
                # Nothing to fill
                return self._from_pyarrow_array(self._pa_array.take(indices))
        else:  # allow_fill=False
            # TODO(ARROW-9432): Treat negative indices as indices from the right.
            if (indices_array < 0).any():
                # Don't modify in-place
                indices_array = np.copy(indices_array)
                indices_array[indices_array < 0] += len(self._pa_array)
            return self._from_pyarrow_array(self._pa_array.take(indices_array))

    def _maybe_convert_datelike_array(self):
        # [FUNCTION DOC]
        # HELPER: converts a timestamp-typed array to a datetime array and a duration-
        # typed array to a timedelta array; any other type returns self.
        """Maybe convert to a datelike array."""
        pa_type = self._pa_array.type
        if pa.types.is_timestamp(pa_type):
            return self._to_datetimearray()
        elif pa.types.is_duration(pa_type):
            return self._to_timedeltaarray()
        return self

    def _to_datetimearray(self) -> DatetimeArray:
        # [FUNCTION DOC]
        # HELPER: converts a timestamp-typed array to a numpy-backed datetime array with
        # the same unit and time zone. Asserts the type is timestamp. Missing entries
        # become the numpy missing-time marker. Returns the datetime array.
        """Convert a pyarrow timestamp typed array to a DatetimeArray."""
        from pandas.core.arrays.datetimes import (
            DatetimeArray,
            tz_to_dtype,
        )

        pa_type = self._pa_array.type
        assert pa.types.is_timestamp(pa_type)
        np_dtype = np.dtype(f"M8[{pa_type.unit}]")
        dtype = tz_to_dtype(pa_type.tz, pa_type.unit)
        np_array = self._pa_array.to_numpy()
        np_array = np_array.astype(np_dtype, copy=False)
        return DatetimeArray._simple_new(np_array, dtype=dtype)

    def _to_timedeltaarray(self) -> TimedeltaArray:
        # [FUNCTION DOC]
        # HELPER: converts a duration-typed array to a numpy-backed timedelta array with
        # the same unit. Asserts the type is duration. Returns the timedelta array.
        """Convert a pyarrow duration typed array to a TimedeltaArray."""
        from pandas.core.arrays.timedeltas import TimedeltaArray

        pa_type = self._pa_array.type
        assert pa.types.is_duration(pa_type)
        np_dtype = np.dtype(f"m8[{pa_type.unit}]")
        np_array = self._pa_array.to_numpy()
        np_array = np_array.astype(np_dtype, copy=False)
        return TimedeltaArray._simple_new(np_array, dtype=np_dtype)

    def _values_for_json(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: values used for JSON output: numeric dtypes yield an object numpy
        # array (so missing values are None); other dtypes use the base class
        # conversion.
        if is_numeric_dtype(self.dtype):
            return np.asarray(self, dtype=object)
        return super()._values_for_json()

    @doc(ExtensionArray.to_numpy)
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Convert to numpy. Parameters: dtype (default None), copy (default false) and
        # na_value (default the library "no default" marker). The dtype and missing
        # sentinel are inferred through the shared helper. If there are missing values
        # and a non-missing sentinel is requested (and the type is not null), a filled
        # copy is used and the copy flag is cleared. Timestamps and durations are
        # converted via numpy datetime/timedelta arrays (the default sentinel is dropped
        # when the dtype is not object); times and dates are converted via a list of
        # Python objects with missing positions set to the sentinel; null-typed data
        # become a full array of the sentinel (None when a dtype is given and the
        # sentinel is missing). Without missing data, or for floats where NaN is the
        # sentinel, the arrow numpy conversion is cast to dtype and copied if requested.
        # The remaining case builds an empty array of inferred dtype (the arrow default
        # if it can hold the sentinel, else object), writes the sentinel at missing
        # positions and fills the rest from the valid data. Returns the numpy array.
        # Docstring is inherited from the base class through a decorator.
        original_na_value = na_value
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, self._hasna)
        pa_type = self._pa_array.type
        if not self._hasna or isna(na_value) or pa.types.is_null(pa_type):
            data = self
        else:
            data = self.fillna(na_value)
            copy = False

        if pa.types.is_timestamp(pa_type) or pa.types.is_duration(pa_type):
            # GH 55997
            if dtype != object and na_value is self.dtype.na_value:
                na_value = lib.no_default
            result = data._maybe_convert_datelike_array().to_numpy(
                dtype=dtype, na_value=na_value
            )
        elif pa.types.is_time(pa_type) or pa.types.is_date(pa_type):
            # convert to list of python datetime.time objects before
            # wrapping in ndarray
            result = np.array(list(data), dtype=dtype)
            if data._hasna:
                result[data.isna()] = na_value
        elif pa.types.is_null(pa_type):
            if dtype is not None and isna(na_value):
                na_value = None
            result = np.full(len(data), fill_value=na_value, dtype=dtype)
        elif not data._hasna or (
            pa.types.is_floating(pa_type)
            and (
                na_value is np.nan
                or (
                    original_na_value is lib.no_default
                    and is_float_dtype(dtype)
                    and is_nan_na()
                )
            )
        ):
            result = data._pa_array.to_numpy()
            if dtype is not None:
                result = result.astype(dtype, copy=False)
            if copy:
                result = result.copy()
        else:
            if dtype is None:
                empty = pa.array([], type=pa_type).to_numpy(zero_copy_only=False)
                if can_hold_element(empty, na_value):
                    dtype = empty.dtype
                else:
                    dtype = np.object_
            result = np.empty(len(data), dtype=dtype)
            mask = data.isna()
            result[mask] = na_value
            result[~mask] = data[~mask]._pa_array.to_numpy()
        return result

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # [FUNCTION DOC]
        # Map values through a mapper. Parameters: mapper (function, dict or series) and
        # na_action (None or ignore). Numeric dtypes map over the numpy conversion;
        # other dtypes use the base class implementation (which keeps missing values
        # instead of turning them into None).
        if is_numeric_dtype(self.dtype):
            return map_array(self.to_numpy(), mapper, na_action=na_action)
        else:
            # For "mM" cases, the super() method passes `self` without the
            #  to_numpy call, which inside map_array casts to ndarray[object].
            #  Without the to_numpy() call, NA is preserved instead of changed
            #  to None.
            return super().map(mapper, na_action)

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Boolean numpy array marking duplicates. Parameter keep (first default, last,
        # or false). Chooses the values to compare: numeric as numpy with missing
        # replaced by 0, booleans with missing as false, temporal types cast to 32- or
        # 64-bit integers with missing as 0, and anything else replaced by its
        # factorized codes. Passes the missing mask (only if there are nulls) to the
        # shared duplicated routine. Docstring is inherited from the base class through
        # a decorator.
        pa_type = self._pa_array.type
        if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type):
            values = self.to_numpy(na_value=0)
        elif pa.types.is_boolean(pa_type):
            values = self.to_numpy(na_value=False)
        elif pa.types.is_temporal(pa_type):
            if pa_type.bit_width == 32:
                pa_type = pa.int32()
            else:
                pa_type = pa.int64()
            arr = self.astype(ArrowDtype(pa_type))
            values = arr.to_numpy(na_value=0)
        else:
            # factorize the values to avoid the performance penalty of
            # converting to object dtype
            values = self.factorize()[0]

        mask = self.isna() if self._hasna else None
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        # [FUNCTION DOC]
        # Unique values in order of appearance, via the arrow unique kernel. Returns a
        # new array.
        """
        Compute the ArrowExtensionArray of unique values.

        Returns
        -------
        ArrowExtensionArray
        """
        pa_result = pc.unique(self._pa_array)
        return self._from_pyarrow_array(pa_result)

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts of unique values as a series. Parameter dropna (default true) removes
        # counts of missing values. Uses arrow's value counting; the series has an index
        # of the values (as this array type), the counts as an arrow-backed array and
        # the name count.
        """
        Return a Series containing counts of each unique value.

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

        data = self._pa_array
        vc = data.value_counts()

        values = vc.field(0)
        counts = vc.field(1)
        if dropna and data.null_count > 0:
            mask = values.is_valid()
            values = values.filter(mask)
            counts = counts.filter(mask)

        counts = ArrowExtensionArray(counts)

        index = Index(self._from_pyarrow_array(values), copy=False)

        return Series(counts, index=index, name="count", copy=False)

    @classmethod
    def _concat_same_type(cls, to_concat) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level concatenation. Parameter to_concat is a non-empty sequence
        # of arrays of this class with the same dtype. Gathers all chunks; for the
        # generic string dtype (which lacks an arrow type attribute) the type is large
        # string, otherwise the arrow type of the first array. Returns a new array of
        # the first element's class built from the combined chunked array.
        """
        Concatenate multiple ArrowExtensionArrays.

        Parameters
        ----------
        to_concat : sequence of ArrowExtensionArrays

        Returns
        -------
        ArrowExtensionArray
        """
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        if to_concat[0].dtype == "string":
            # StringDtype has no attribute pyarrow_dtype
            pa_dtype = pa.large_string()
        else:
            pa_dtype = to_concat[0].dtype.pyarrow_dtype
        arr = pa.chunked_array(chunks, type=pa_dtype)
        return to_concat[0]._from_pyarrow_array(arr)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ArrowExtensionArray | ExtensionArray:
        # [FUNCTION DOC]
        # HELPER: cumulative operations. Parameters: name (cumulative min, max, sum or
        # product), skipna (keyword, default true), extra keywords passed to the kernel.
        # String data use the string implementation. The name is mapped to the arrow
        # cumulative kernel; if arrow has none, the base class implementation (which
        # raises a not-implemented error) is used. Temporal min/max and duration sums
        # are computed on integer casts and cast back. A kernel not-implemented error is
        # raised as a type error naming the operation and dtype. Returns a new array.
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations
        """
        if is_string_dtype(self):
            return self._str_accumulate(name=name, skipna=skipna, **kwargs)

        pyarrow_name = {
            "cummax": "cumulative_max",
            "cummin": "cumulative_min",
            "cumprod": "cumulative_prod_checked",
            "cumsum": "cumulative_sum_checked",
        }.get(name, name)
        pyarrow_meth = getattr(pc, pyarrow_name, None)
        if pyarrow_meth is None:
            return super()._accumulate(name, skipna=skipna, **kwargs)

        data_to_accum = self._pa_array

        pa_dtype = data_to_accum.type

        convert_to_int = (
            pa.types.is_temporal(pa_dtype) and name in ["cummax", "cummin"]
        ) or (pa.types.is_duration(pa_dtype) and name == "cumsum")

        if convert_to_int:
            if pa_dtype.bit_width == 32:
                data_to_accum = data_to_accum.cast(pa.int32())
            else:
                data_to_accum = data_to_accum.cast(pa.int64())

        try:
            result = pyarrow_meth(data_to_accum, skip_nulls=skipna, **kwargs)
        except pa.ArrowNotImplementedError as err:
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg) from err

        if convert_to_int:
            result = result.cast(pa_dtype)

        return self._from_pyarrow_array(result)

    def _str_accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ArrowExtensionArray | ExtensionArray:
        # [FUNCTION DOC]
        # HELPER: cumulative operations for strings, which arrow does not implement.
        # Parameters as in the accumulate method. Cumulative product raises a type
        # error. All-missing input is returned unchanged. With missing values: when
        # skipping, sums fill missing with the empty string and min/max forward then
        # backward fill (preserving the running extreme); otherwise the data are cut at
        # the first missing value and the tail is all nulls. The numpy accumulate
        # function (sum, minimum, maximum) is applied to the data, and then missing
        # positions are restored (skipna) or the null tail appended. Returns a new
        # array.
        """
        Accumulate implementation for strings, see `_accumulate` docstring for details.

        pyarrow.compute does not implement these methods for strings.
        """
        if name == "cumprod":
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg)

        # We may need to strip out trailing NA values
        tail: pa.array | None = None
        na_mask: pa.array | None = None
        pa_array = self._pa_array
        np_func = {
            "cumsum": np.cumsum,
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
        }[name]

        if self._hasna:
            na_mask = pc.is_null(pa_array)
            if pc.all(na_mask) == pa.scalar(True):
                return self._from_pyarrow_array(pa_array)
            if skipna:
                if name == "cumsum":
                    pa_array = _safe_fill_null(pa_array, "")
                else:
                    # We can retain the running min/max by forward/backward filling.
                    pa_array = pc.fill_null_forward(pa_array)
                    pa_array = pc.fill_null_backward(pa_array)
            else:
                # When not skipping NA values, the result should be null from
                # the first NA value onward.
                idx = pc.index(na_mask, True).as_py()
                tail = pa.nulls(len(pa_array) - idx, type=pa_array.type)
                pa_array = pa_array[:idx]

        # error: Cannot call function of unknown type
        pa_result = pa.array(np_func(pa_array), type=pa_array.type)  # type: ignore[operator]

        if tail is not None:
            pa_result = pa.concat_arrays([pa_result, tail])
        elif na_mask is not None:
            pa_result = pc.if_else(na_mask, None, pa_result)

        result = self._from_pyarrow_array(pa_result)
        return result

    def _reduce_pyarrow(self, name: str, *, skipna: bool = True, **kwargs) -> pa.Scalar:
        # [FUNCTION DOC]
        # HELPER: runs the reduction in arrow and returns an arrow scalar. Parameters:
        # name (any, all, min, max, sum, mean, median, prod, std, var, sem, kurt, skew),
        # skipna (keyword, default true), extra keywords passed through (such as the
        # delta degrees of freedom or a minimum count). Preparation: any/all on integer,
        # floating, duration and decimal data compare to zero first (arrow supports them
        # only for booleans); min, max and sum on durations and the mean, median, std
        # and sem of temporal types operate on integer casts. Kernel choice: sem is
        # defined as the standard deviation divided by the square root of the count of
        # valid values; string sums join the strings (honoring skipna and a minimum
        # count, returning a null scalar when too few values, and converting large
        # strings to strings first); other names are mapped to the arrow kernel (median
        # uses quantile 0.5, prod is product, std is stddev, var is variance); if there
        # is no such kernel the base class reduction raises a type error. any/all get a
        # default minimum count of zero (arrow defaults to one). Errors from the kernel
        # (attribute, not-implemented or type errors) become a type error mentioning the
        # pyarrow version. Results are converted back for temporal types (median takes
        # the first element of the quantile result; min/max/sum on durations are cast
        # back; mean/median on temporal types round-trip via integers; std/sem become
        # durations in the appropriate unit, using seconds for dates). Returns the arrow
        # scalar.
        """
        Return a pyarrow scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        pyarrow scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        pa_type = self._pa_array.type

        data_to_reduce = self._pa_array

        if name in ["any", "all"] and (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_duration(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            # pyarrow only supports any/all for boolean dtype, we allow
            #  for other dtypes, matching our non-pyarrow behavior

            if pa.types.is_duration(pa_type):
                data_to_cmp = self._pa_array.cast(pa.int64())
            else:
                data_to_cmp = self._pa_array

            not_eq = pc.not_equal(data_to_cmp, 0)
            data_to_reduce = not_eq

        elif name in ["min", "max", "sum"] and pa.types.is_duration(pa_type):
            data_to_reduce = self._pa_array.cast(pa.int64())

        elif name in ["median", "mean", "std", "sem"] and pa.types.is_temporal(pa_type):
            nbits = pa_type.bit_width
            if nbits == 32:
                data_to_reduce = self._pa_array.cast(pa.int32())
            else:
                data_to_reduce = self._pa_array.cast(pa.int64())

        if name == "sem":

            def pyarrow_meth(data, skip_nulls, **kwargs):
                # [FUNCTION DOC]
                # HELPER: nested replacement kernel for the standard error of the mean.
                # Parameters: data, skip_nulls and extra keywords for the standard-
                # deviation kernel. Returns the standard deviation divided by the square
                # root of the count of valid values in the whole array.
                numerator = pc.stddev(data, skip_nulls=skip_nulls, **kwargs)
                denominator = pc.sqrt_checked(pc.count(self._pa_array))
                return pc.divide_checked(numerator, denominator)

        elif name == "sum" and (
            pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type)
        ):

            def pyarrow_meth(data, skip_nulls, min_count=0):  # type: ignore[misc]
                # [FUNCTION DOC]
                # HELPER: nested replacement kernel for summing strings (concatenation).
                # Parameters: data, skip_nulls and min_count (default 0). With
                # skip_nulls and a minimum count above zero, returns a null scalar if
                # the number of valid values is below the minimum; nulls are filtered
                # out before joining. Without skip_nulls, any null, or too few values,
                # gives a null scalar. Large strings are cast to strings (the join
                # kernel needs them). Returns the joined string as an arrow scalar.
                mask = pc.is_null(data) if data.null_count > 0 else None
                if skip_nulls:
                    if min_count > 0 and check_below_min_count(
                        (len(data),),
                        None if mask is None else mask.to_numpy(),
                        min_count,
                    ):
                        return pa.scalar(None, type=data.type)
                    if data.null_count > 0:
                        # binary_join returns null if there is any null ->
                        # have to filter out any nulls
                        data = data.filter(pc.invert(mask))
                elif mask is not None or check_below_min_count(
                    (len(data),), None, min_count
                ):
                    return pa.scalar(None, type=data.type)

                if pa.types.is_large_string(data.type):
                    # binary_join only supports string, not large_string
                    data = data.cast(pa.string())
                data_list = pa.ListArray.from_arrays(
                    [0, len(data)], data.combine_chunks()
                )[0]
                return pc.binary_join(data_list, "")

        else:
            pyarrow_name = {
                "median": "quantile",
                "prod": "product",
                "std": "stddev",
                "var": "variance",
            }.get(name, name)
            # error: Incompatible types in assignment
            # (expression has type "Optional[Any]", variable has type
            # "Callable[[Any, Any, KwArg(Any)], Any]")
            pyarrow_meth = getattr(pc, pyarrow_name, None)  # type: ignore[assignment]
            if pyarrow_meth is None:
                # Let ExtensionArray._reduce raise the TypeError
                return super()._reduce(name, skipna=skipna, **kwargs)

        # GH51624: pyarrow defaults to min_count=1, pandas behavior is min_count=0
        if name in ["any", "all"] and "min_count" not in kwargs:
            kwargs["min_count"] = 0
        elif name == "median":
            # GH 52679: Use quantile instead of approximate_median
            kwargs["q"] = 0.5

        try:
            result = pyarrow_meth(data_to_reduce, skip_nulls=skipna, **kwargs)
        except (AttributeError, NotImplementedError, TypeError) as err:
            msg = (
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support operation '{name}' with pyarrow "
                f"version {pa.__version__}. '{name}' may be supported by "
                f"upgrading pyarrow."
            )
            raise TypeError(msg) from err
        if name == "median":
            # GH 52679: Use quantile instead of approximate_median; returns array
            result = result[0]

        if name in ["min", "max", "sum"] and pa.types.is_duration(pa_type):
            result = result.cast(pa_type)
        if name in ["median", "mean"] and pa.types.is_temporal(pa_type):
            nbits = pa_type.bit_width
            if nbits == 32:
                result = result.cast(pa.int32(), safe=False)
            else:
                result = result.cast(pa.int64(), safe=False)
            result = result.cast(pa_type)
        if name in ["std", "sem"] and pa.types.is_temporal(pa_type):
            result = result.cast(pa.int64(), safe=False)
            if pa.types.is_duration(pa_type):
                result = result.cast(pa_type)
            elif pa.types.is_time(pa_type):
                result = result.cast(pa.duration(pa_type.unit))
            elif pa.types.is_date(pa_type):
                # go with closest available unit, i.e. "s"
                result = result.cast(pa.duration("s"))
            else:
                # i.e. timestamp
                result = result.cast(pa.duration(pa_type.unit))

        return result

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: reduction entry point. Keyword parameters: skipna (default true),
        # keepdims (default false) and extra keywords. Computes through the calculation
        # helper and, if the result is an arrow array (keepdims), wraps it in an array
        # of this class; otherwise returns the scalar.
        # KNOWN ISSUE: the docstring does not mention the keepdims parameter.
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if isinstance(result, pa.Array):
            return self._from_pyarrow_array(result)
        else:
            return result

    def _reduce_calc(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: reduction computation returning Python-level results. Parameters as in
        # the reduction method. Runs the arrow reduction. With keepdims it returns a
        # one-element arrow array of the same type (non-scalar results get a type
        # inferred from the value). Otherwise a null result becomes the dtype's missing
        # value; scalar results become Python objects, with timedeltas and timestamps of
        # non-nanosecond units returned as pandas scalars in that unit; anything else is
        # returned unchanged.
        pa_result = self._reduce_pyarrow(name, skipna=skipna, **kwargs)

        if keepdims:
            if isinstance(pa_result, pa.Scalar):
                result = pa.array([pa_result.as_py()], type=pa_result.type)
            else:
                result = pa.array(
                    [pa_result],
                    type=to_pyarrow_type(infer_dtype_from_scalar(pa_result)[0]),
                )
            return result

        if pc.is_null(pa_result).as_py():
            return self.dtype.na_value
        elif isinstance(pa_result, pa.Scalar):
            result = pa_result.as_py()
            pa_type = pa_result.type
            if pa.types.is_duration(pa_type) and pa_type.unit != "ns":
                return Timedelta(result).as_unit(pa_type.unit)
            elif pa.types.is_timestamp(pa_type) and pa_type.unit != "ns":
                return Timestamp(result).as_unit(pa_type.unit)
            return result
        else:
            return pa_result

    def _explode(self):
        # [FUNCTION DOC]
        # HELPER: explode support. Non-list (and non-large-list) arrow types use the
        # base class implementation. Otherwise: counts the length of each list (nulls
        # count as one), replaces empty lists with a one-element list holding null so
        # that empty and missing rows survive, fills nulls similarly, and flattens.
        # Returns (values array, counts numpy array giving output rows per input
        # element).
        """
        See Series.explode.__doc__.
        """
        # child class explode method supports only list types; return
        # default implementation for non list types.
        if not hasattr(self.dtype, "pyarrow_dtype") or (
            not pa.types.is_list(self.dtype.pyarrow_dtype)
            and not pa.types.is_large_list(self.dtype.pyarrow_dtype)
        ):
            return super()._explode()
        values = self
        counts = pa.compute.list_value_length(values._pa_array)
        counts = counts.fill_null(1).to_numpy()
        fill_value = pa.scalar([None], type=self._pa_array.type)
        mask = counts == 0
        if mask.any():
            # pc.if_else here is similar to `values[mask] = fill_value`
            #  but this avoids an object-dtype round-trip.
            pa_values = pc.if_else(~mask, values._pa_array, fill_value)
            values = self._from_pyarrow_array(pa_values)
            counts = counts.copy()
            counts[mask] = 1
        values = values.fillna(fill_value)
        values = self._from_pyarrow_array(pa.compute.list_flatten(values._pa_array))
        return values, counts

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Set values in place by replacing the backing array. Parameters: key (integer,
        # slice, integer or boolean array, tuple of one element) and value (scalar or
        # array-like). Raises a value error if read-only. The key is validated and the
        # value boxed to arrow. Cases: a null slice with an arrow value of the same type
        # and length adopts the value (copying its buffers first if the caller could
        # still mutate them); a null slice with anything else uses the conditional-
        # select helper. An integer key (negative allowed, out of range raises an index
        # error) rebuilds the chunks around the new single value (a list-like value
        # raises a value error). A boolean key replaces under the mask. A scalar value
        # with other keys builds a mask and uses the conditional-select helper.
        # Otherwise the value must have as many elements as the key selects (value error
        # otherwise); an empty selection returns; repeated indices are resolved by
        # keeping the first occurrence of each position; the replacement is done under a
        # mask. The result is stored as a chunked array.
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        # GH50085: unwrap 1D indexers
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        key = check_array_indexer(self, key)
        value = self._maybe_convert_setitem_value(value)

        if com.is_null_slice(key):
            # fast path (GH50248)
            if (
                isinstance(value, (pa.Array, pa.ChunkedArray))
                and value.type == self._pa_array.type
                and len(value) == len(self)
            ):
                # GH#67990 this adopts ``value`` as our backing array, so copy
                #  first if the caller may still own and mutate its buffers.
                if _boxing_may_borrow_memory(value.type):
                    value = _copy_pyarrow_buffers(value)
                data = value
            else:
                data = self._if_else(True, value, self._pa_array)

        elif is_integer(key):
            # fast path
            key = cast(int, key)
            n = len(self)
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {n}"
                )
            if isinstance(value, pa.Scalar):
                value = value.as_py()
            elif is_list_like(value):
                raise ValueError("Length of indexer and values mismatch")
            chunks = [
                *self._pa_array[:key].chunks,
                pa.array([value], type=self._pa_array.type, from_pandas=is_nan_na()),
                *self._pa_array[key + 1 :].chunks,
            ]
            data = pa.chunked_array(chunks).combine_chunks()

        elif is_bool_dtype(key):
            key = np.asarray(key, dtype=np.bool_)
            data = self._replace_with_mask(self._pa_array, key, value)

        elif is_scalar(value) or isinstance(value, pa.Scalar):
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[key] = True
            data = self._if_else(mask, value, self._pa_array)

        else:
            indices = np.arange(len(self))[key]
            if len(indices) != len(value):
                raise ValueError("Length of indexer and values mismatch")
            if len(indices) == 0:
                return
            # GH#58530 wrong item assignment by repeated key
            _, argsort = np.unique(indices, return_index=True)
            indices = indices[argsort]
            value = value.take(argsort)
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[indices] = True
            data = self._replace_with_mask(self._pa_array, mask, value)

        if isinstance(data, pa.Array):
            data = pa.chunked_array([data])
        self._pa_array = data

    def _rank_calc(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        # [FUNCTION DOC]
        # HELPER: ranking computation, returning an arrow array. Keyword parameters:
        # axis (0 default), method (average default, min, max, first, dense), na_option
        # (keep default, top, bottom), ascending (default true), pct (default false).
        # For an axis other than zero it uses the base class implementation and converts
        # to 64-bit float (average or percent) or unsigned 64-bit rank type. For axis
        # zero it uses the arrow rank kernel on the combined chunks, with nulls placed
        # at the start for top and at the end otherwise; average uses the minimum
        # tiebreaker plus the maximum tiebreaker averaged; with keep, nulls in the input
        # give null ranks; percent divides by the maximum rank for dense and by the
        # count of valid ranks otherwise. Returns the arrow ranks.
        if axis != 0:
            ranked = super()._rank(
                axis=axis,
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
            # keep dtypes consistent with the implementation below
            if method == "average" or pct:
                pa_type = pa.float64()
            else:
                pa_type = pa.uint64()
            result = pa.array(ranked, type=pa_type, from_pandas=is_nan_na())
            return result

        data = self._pa_array.combine_chunks()
        sort_keys = "ascending" if ascending else "descending"
        null_placement = "at_start" if na_option == "top" else "at_end"
        tiebreaker = "min" if method == "average" else method

        result = pc.rank(
            data,
            sort_keys=sort_keys,
            null_placement=null_placement,
            tiebreaker=tiebreaker,
        )

        if na_option == "keep":
            mask = pc.is_null(self._pa_array)
            null = pa.scalar(None, type=result.type)
            result = pc.if_else(mask, null, result)

        if method == "average":
            result_max = pc.rank(
                data,
                sort_keys=sort_keys,
                null_placement=null_placement,
                tiebreaker="max",
            )
            result_max = result_max.cast(pa.float64())
            result_min = result.cast(pa.float64())
            result = pc.divide(pc.add(result_min, result_max), 2)

        if pct:
            if not pa.types.is_floating(result.type):
                result = result.cast(pa.float64())
            if method == "dense":
                divisor = pc.max(result)
            else:
                divisor = pc.count(result)
            result = pc.divide(result, divisor)

        return result

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: ranking entry point (see series rank for semantics). Keyword
        # parameters as in the calculation helper; the arrow ranks are converted to the
        # result type by the conversion hook.
        """
        See Series.rank.__doc__.
        """
        return self._convert_rank_result(
            self._rank_calc(
                axis=axis,
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
        )

    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        # [FUNCTION DOC]
        # HELPER: quantiles. Parameters: qs (float array of quantiles) and interpolation
        # (method). Temporal types are cast to 32- or 64-bit integers for the kernel
        # (working around an arrow limitation); the result is floored if it came back
        # floating and cast back to the original type. Returns a new array.
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
        interpolation: str

        Returns
        -------
        same type as self
        """
        pa_dtype = self._pa_array.type

        data = self._pa_array
        if pa.types.is_temporal(pa_dtype):
            # https://github.com/apache/arrow/issues/33769 in these cases
            #  we can cast to ints and back
            nbits = pa_dtype.bit_width
            if nbits == 32:
                data = data.cast(pa.int32())
            else:
                data = data.cast(pa.int64())

        result = pc.quantile(data, q=qs, interpolation=interpolation)

        if pa.types.is_temporal(pa_dtype):
            if pa.types.is_floating(result.type):
                result = pc.floor(result)
            nbits = pa_dtype.bit_width
            if nbits == 32:
                result = result.cast(pa.int32())
            else:
                result = result.cast(pa.int64())
            result = result.cast(pa_dtype)

        return self._from_pyarrow_array(result)

    def _mode(self, dropna: bool = True) -> Self:
        # [FUNCTION DOC]
        # HELPER: modes. Parameter dropna (default true). Temporal types are cast to 32-
        # or 64-bit integers (other widths raise a not-implemented error); nulls are
        # dropped when requested; the values whose count equals the maximum count are
        # selected, cast back for temporal types, and sorted. Returns a new array.
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA values.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
        pa_type = self._pa_array.type
        if pa.types.is_temporal(pa_type):
            nbits = pa_type.bit_width
            if nbits == 32:
                data = self._pa_array.cast(pa.int32())
            elif nbits == 64:
                data = self._pa_array.cast(pa.int64())
            else:
                raise NotImplementedError(pa_type)
        else:
            data = self._pa_array

        if dropna:
            data = data.drop_null()

        res = pc.value_counts(data)
        most_common = res.field("values").filter(
            pc.equal(res.field("counts"), pc.max(res.field("counts")))
        )

        if pa.types.is_temporal(pa_type):
            most_common = most_common.cast(pa_type)

        most_common = most_common.take(pc.array_sort_indices(most_common))
        return self._from_pyarrow_array(most_common)

    def _maybe_convert_setitem_value(self, value):
        # [FUNCTION DOC]
        # HELPER: boxes a value for assignment into this array's arrow type. Parameter
        # value. Arrow type errors become a type error naming the value and dtype.
        # Returns an arrow scalar or array.
        """Maybe convert value to be pyarrow compatible."""
        try:
            value = self._box_pa(value, self._pa_array.type)
        except pa.ArrowTypeError as err:
            msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
            raise TypeError(msg) from err
        return value

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
    ) -> Self:
        # [FUNCTION DOC]
        # Interpolate missing values; same parameters as the base class (keyword-only:
        # method, axis, index, limit, limit_direction, limit_area, copy and extra
        # keywords). Raises a type error for non-numeric dtypes and a not-implemented
        # error for numeric dtypes other than float or integer kinds (for example
        # decimal). Floats use their numpy conversion; integers are converted to 64-bit
        # float with zeros at missing positions. The shared in-place two-dimensional
        # interpolation routine fills the data along axis zero using the missing mask,
        # then the data are re-boxed with the original mask as nulls. A faster native
        # path was removed because it mishandled consecutive and trailing missing values
        # and integer dtypes. Always returns a new array (the copy flag is not
        # consulted), of the same type; integer inputs come back as floats.
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if not self.dtype._is_numeric:
            raise TypeError(f"Cannot interpolate with {self.dtype} dtype")

        # GH#65345: a pyarrow-native fast path for
        # method="linear"/limit_direction="forward" was removed here because
        # it only handled isolated NAs (leaving consecutive and trailing NAs
        # unfilled), truncated interpolated values for integer dtypes, and
        # did not upcast to float64 like the general path below.

        mask = self.isna()
        if self.dtype.kind == "f":
            data = self._pa_array.to_numpy()
        elif self.dtype.kind in "iu":
            data = self.to_numpy(dtype="f8", na_value=0.0)
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
        return self._from_pyarrow_array(self._box_pa_array(pa.array(data, mask=mask)))

    @classmethod
    def _if_else(
        cls,
        cond: npt.NDArray[np.bool_] | bool,
        left: ArrayLike | Scalar,
        right: ArrayLike | Scalar,
    ) -> pa.Array:
        # [FUNCTION DOC]
        # HELPER: class-level conditional select with fallbacks. Parameters: cond
        # (boolean numpy array or bool), left and right (arrays or scalars chosen where
        # cond is true and false). First, string chunked arrays with non-zero chunk
        # offsets are combined into single arrays (a workaround for GH#64320; combining
        # failure leaves them as is). If both are usable, the arrow conditional-select
        # kernel is tried. If it is not implemented for the types, falls back to numpy:
        # both sides become object arrays (scalars converted to Python objects), the
        # numpy conditional select is applied and the result is converted to an arrow
        # array with the arrow type taken from the left side, else the right. Returns an
        # arrow array.
        """
        Choose values based on a condition.

        Analogous to pyarrow.compute.if_else, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        cond : npt.NDArray[np.bool_] or bool
        left : ArrayLike | Scalar
        right : ArrayLike | Scalar

        Returns
        -------
        pa.Array
        """

        # TODO: Remove this part when pa.if_else is fixed (GH#64320)
        def _maybe_combine(arr):
            # [FUNCTION DOC]
            # HELPER: nested function for the conditional-select workaround. Parameter
            # arr. Returns it unchanged unless it is a chunked string or large string
            # array with at least one chunk whose offset is non-zero, in which case it
            # returns the chunks combined into one array, or None if combining fails
            # (invalid data, capacity or memory error).
            # KNOWN ISSUE: a TODO in the source says this workaround should be removed
            # once pyarrow fixes the conditional-select bug.
            if not isinstance(arr, pa.ChunkedArray) or not (
                pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)
            ):
                return arr
            if not any(c.offset != 0 for c in arr.chunks):
                return arr
            try:
                return arr.combine_chunks()
            except (pa.ArrowInvalid, pa.ArrowCapacityError, MemoryError):
                return None

        left_c, right_c = _maybe_combine(left), _maybe_combine(right)
        if left_c is not None and right_c is not None:
            try:
                return pc.if_else(cond, left_c, right_c)
            except pa.ArrowNotImplementedError:
                pass
        if left_c is not None:
            left = left_c
        if right_c is not None:
            right = right_c

        def _to_numpy_and_type(value) -> tuple[np.ndarray, pa.DataType | None]:
            # [FUNCTION DOC]
            # HELPER: nested function for the numpy fallback of the conditional select.
            # Parameter value (arrow array, chunked array, arrow scalar or other).
            # Returns a pair: an object numpy array of the value (arrow scalars first
            # converted to Python objects) and the arrow type if the value was arrow
            # data, otherwise None.
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                pa_type = value.type
            elif isinstance(value, pa.Scalar):
                pa_type = value.type
                value = value.as_py()
            else:
                pa_type = None
            return np.array(value, dtype=object), pa_type

        left, left_type = _to_numpy_and_type(left)
        right, right_type = _to_numpy_and_type(right)
        pa_type = left_type or right_type
        result = np.where(cond, left, right)
        return pa.array(result, type=pa_type, from_pandas=is_nan_na())

    @classmethod
    def _replace_with_mask(
        cls,
        values: pa.Array | pa.ChunkedArray,
        mask: npt.NDArray[np.bool_] | bool,
        replacements: ArrayLike | Scalar,
    ) -> pa.Array | pa.ChunkedArray:
        # [FUNCTION DOC]
        # HELPER: class-level analogue of the arrow replace-with-mask kernel with a
        # numpy fallback. Parameters: values (arrow array or chunked array), mask
        # (boolean array or bool) and replacements (array-like or scalar). Chunked
        # replacements are combined; chunked boolean values are combined too (works
        # around a crash, GH#52059). If the kernel is not implemented, the values and
        # replacements are converted to Python-level data and assigned under the mask in
        # an object numpy array, then converted back to an arrow array of the original
        # type. Returns an arrow array or chunked array.
        """
        Replace items selected with a mask.

        Analogous to pyarrow.compute.replace_with_mask, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        values : pa.Array or pa.ChunkedArray
        mask : npt.NDArray[np.bool_] or bool
        replacements : ArrayLike or Scalar
            Replacement value(s)

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
        if isinstance(replacements, pa.ChunkedArray):
            # replacements must be array or scalar, not ChunkedArray
            replacements = replacements.combine_chunks()
        if isinstance(values, pa.ChunkedArray) and pa.types.is_boolean(values.type):
            # GH#52059 replace_with_mask segfaults for chunked array
            # https://github.com/apache/arrow/issues/34634
            values = values.combine_chunks()
        try:
            return pc.replace_with_mask(values, mask, replacements)
        except pa.ArrowNotImplementedError:
            pass
        if isinstance(replacements, pa.Array):
            replacements = np.array(replacements, dtype=object)
        elif isinstance(replacements, pa.Scalar):
            replacements = replacements.as_py()

        result = np.array(values, dtype=object)
        result[mask] = replacements
        return pa.array(result, type=values.type, from_pandas=is_nan_na())

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _to_masked(self):
        # [FUNCTION DOC]
        # HELPER: converts to the matching masked array for groupby operations.
        # Supported only for floating, integer and boolean arrow types (others raise a
        # not-implemented error). The masked dtype comes from the arrow-to-masked dtype
        # mapping; the data is the numpy conversion with a harmless placeholder at
        # missing positions and the mask is the missing mask. Returns the masked array.
        pa_dtype = self._pa_array.type

        if pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype):
            na_value = 1
        elif pa.types.is_boolean(pa_dtype):
            na_value = True
        else:
            raise NotImplementedError

        dtype = _arrow_dtype_mapping()[pa_dtype]
        mask = self.isna()
        arr = self.to_numpy(dtype=dtype.numpy_dtype, na_value=na_value)
        return dtype.construct_array_type()(arr, mask)

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
        # HELPER: groupby hook. Keyword parameters as in the base class (how,
        # has_dropped_na, min_count, ngroups, ids, extra keywords). For string dtypes,
        # product, mean, median, cumulative sum/product, std, sem, var and skew raise a
        # type error and everything else is delegated to the base class. For other types
        # the data is converted to a datetime array (timestamps), timedelta array
        # (durations) or masked array, the operation is run there, and the result is
        # converted back: numpy results are returned as is, masked results are converted
        # through arrow into this class, and datetime-like results are converted to
        # arrow and wrapped.
        if isinstance(self.dtype, StringDtype):
            if how in [
                "prod",
                "mean",
                "median",
                "cumsum",
                "cumprod",
                "std",
                "sem",
                "var",
                "skew",
            ]:
                raise TypeError(
                    f"dtype '{self.dtype}' does not support operation '{how}'"
                )
            return super()._groupby_op(
                how=how,
                has_dropped_na=has_dropped_na,
                min_count=min_count,
                ngroups=ngroups,
                ids=ids,
                **kwargs,
            )

        # maybe convert to a compatible dtype optimized for groupby
        values: ExtensionArray
        pa_type = self._pa_array.type
        if pa.types.is_timestamp(pa_type):
            values = self._to_datetimearray()
        elif pa.types.is_duration(pa_type):
            values = self._to_timedeltaarray()
        else:
            values = self._to_masked()

        result = values._groupby_op(
            how=how,
            has_dropped_na=has_dropped_na,
            min_count=min_count,
            ngroups=ngroups,
            ids=ids,
            **kwargs,
        )
        if isinstance(result, np.ndarray):
            return result
        elif isinstance(result, BaseMaskedArray):
            pa_result = result.__arrow_array__()
            return self._from_pyarrow_array(pa_result)
        else:
            # DatetimeArray, TimedeltaArray
            pa_result = pa.array(result)
            return self._from_pyarrow_array(pa_result)

    def _apply_elementwise(self, func: Callable) -> list[list[Any]]:
        # [FUNCTION DOC]
        # HELPER: implementation of the string mixin's hook. Parameter func is a
        # callable applied to every non-null element; nulls stay None. Returns a list of
        # lists mirroring the chunk structure of the backing array.
        """Apply a callable to each element while maintaining the chunking structure."""
        return [
            [
                None if val is None else func(val)
                for val in chunk.to_numpy(zero_copy_only=False)
            ]
            for chunk in self._pa_array.iterchunks()
        ]

    def _convert_bool_result(self, result, na=lib.no_default, method_name=None):
        # [FUNCTION DOC]
        # HELPER: implementation of the string mixin's hook. Parameters: result (boolean
        # arrow data), na (default "no default" marker; if given and not missing, nulls
        # are filled with it) and method_name (unused). Returns this class wrapping the
        # result.
        if na is not lib.no_default and not isna(na):  # pyright: ignore [reportGeneralTypeIssues]
            result = result.fill_null(na)
        return self._from_pyarrow_array(result)

    def _convert_int_result(self, result):
        # [FUNCTION DOC]
        # HELPER: implementation of the string mixin's hook; wraps the integer arrow
        # result in this class.
        return self._from_pyarrow_array(result)

    def _convert_rank_result(self, result):
        # [FUNCTION DOC]
        # HELPER: hook that wraps the rank result in this class (overridden elsewhere to
        # produce other result types).
        return self._from_pyarrow_array(result)

    def _str_count(self, pat: str, flags: int = 0) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook counting regex pattern matches per string.
        # Parameters: pat and flags (default 0). Raises a not-implemented error if flags
        # is non-zero. Returns a new array of counts.
        if flags:
            raise NotImplementedError(f"count not implemented with {flags=}")
        return self._from_pyarrow_array(pc.count_substring_regex(self._pa_array, pat))

    def _str_repeat(self, repeats: int | Sequence[int]) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook repeating each string. Parameter repeats must be
        # a single integer (otherwise a not-implemented error is raised). Returns a new
        # array.
        if not isinstance(repeats, int):
            raise NotImplementedError(
                f"repeat is not implemented when repeats is {type(repeats).__name__}"
            )
        return self._from_pyarrow_array(pc.binary_repeat(self._pa_array, repeats))

    def _str_join(self, sep: str) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook joining list elements. Parameter sep is the
        # separator. String and large string data are first split into lists of
        # characters element by element (so each string is joined by its characters);
        # list data are joined directly. Returns a new array.
        if pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            result = self._apply_elementwise(list)
            result = pa.chunked_array(result, type=pa.list_(pa.string()))
        else:
            result = self._pa_array
        return self._from_pyarrow_array(pc.binary_join(result, sep))

    def _str_partition(self, sep: str, expand: bool) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook splitting each string around the first separator
        # into a three-part list (before, separator, after), computed element by
        # element. Parameters: sep and expand.
        # KNOWN ISSUE: the expand parameter is accepted but never used.
        predicate = lambda val: val.partition(sep)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_rpartition(self, sep: str, expand: bool) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook splitting each string around the last separator
        # into a three-part list, computed element by element. Parameters: sep and
        # expand.
        # KNOWN ISSUE: the expand parameter is accepted but never used.
        predicate = lambda val: val.rpartition(sep)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_casefold(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook that case-folds each string element by element.
        # Returns a new array.
        predicate = lambda val: val.casefold()
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_encode(self, encoding: str, errors: str = "strict") -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook encoding each string to bytes element by element.
        # Parameters: encoding and errors (default strict). Returns a new array of
        # binary data.
        predicate = lambda val: val.encode(encoding, errors)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        # [FUNCTION DOC]
        # HELPER: string accessor hook extracting regex capture groups. Parameters: pat
        # (a pattern which must contain named groups), flags (only zero supported;
        # otherwise a not-implemented error) and expand (default true). Raises a value
        # error if the pattern has no named group. Uses the arrow regex extraction. With
        # expand, returns a mapping from group name to a new array of that group's
        # values; otherwise returns a new array of the first field.
        if flags:
            raise NotImplementedError("Only flags=0 is implemented.")
        groups = re.compile(pat).groupindex.keys()
        if len(groups) == 0:
            raise ValueError(f"{pat=} must contain a symbolic group name.")
        result = pc.extract_regex(self._pa_array, pat)
        if expand:
            return {
                col: self._from_pyarrow_array(pc.struct_field(result, [i]))
                for col, i in zip(groups, range(result.type.num_fields), strict=True)
            }
        else:
            return type(self)(pc.struct_field(result, [0]))

    def _str_findall(self, pat: str, flags: int = 0) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook finding all regex matches in each string, element
        # by element with Python's regex engine. Parameters: pat and flags (default 0).
        # Returns a new array of lists.
        regex = re.compile(pat, flags=flags)
        predicate = lambda val: regex.findall(val)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_get_dummies(self, sep: str = "|", dtype: NpDtype | None = None):
        # [FUNCTION DOC]
        # HELPER: string accessor hook producing indicator columns. Parameters: sep
        # (default a vertical bar) and dtype (default boolean; non-numpy dtypes fall
        # back to boolean). Splits each string on sep, finds the sorted unique tokens,
        # and builds a rows-by-tokens matrix with ones where a token occurs. Returns a
        # pair: an array whose elements are the matrix rows, and the list of sorted
        # tokens.
        if dtype is None:
            dtype = np.bool_
        split = pc.split_pattern(self._pa_array, sep)
        flattened_values = pc.list_flatten(split)
        uniques = flattened_values.unique()
        uniques_sorted = uniques.take(pa.compute.array_sort_indices(uniques))
        lengths = pc.list_value_length(split).fill_null(0).to_numpy()
        n_rows = len(self)
        n_cols = len(uniques)
        indices = pc.index_in(flattened_values, uniques_sorted).to_numpy()
        indices = indices + np.arange(n_rows).repeat(lengths) * n_cols
        _dtype = pandas_dtype(dtype)
        dummies_dtype: NpDtype
        if isinstance(_dtype, np.dtype):
            dummies_dtype = _dtype
        else:
            dummies_dtype = np.bool_
        dummies = np.zeros(n_rows * n_cols, dtype=dummies_dtype)
        dummies[indices] = True
        dummies = dummies.reshape((n_rows, n_cols))
        result = self._from_pyarrow_array(pa.array(list(dummies)))
        return result, uniques_sorted.to_pylist()

    def _str_index(self, sub: str, start: int = 0, end: int | None = None) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook finding a substring's position, element by
        # element with Python semantics (so a missing substring raises a value error).
        # Parameters: sub, start (default 0), end (default None). Returns a new array.
        predicate = lambda val: val.index(sub, start, end)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook finding the last position of a substring, element
        # by element (a missing substring raises a value error). Parameters: sub, start
        # (default 0), end (default None). Returns a new array.
        predicate = lambda val: val.rindex(sub, start, end)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_normalize(self, form: Literal["NFC", "NFD", "NFKC", "NFKD"]) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook applying Unicode normalization element by
        # element. Parameter form is one of four normal forms. Returns a new array.
        predicate = lambda val: unicodedata.normalize(form, val)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_rfind(self, sub: str, start: int = 0, end=None) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook finding the last position of a substring (-1 if
        # absent), element by element. Parameters: sub, start (default 0), end (default
        # None). Returns a new array.
        predicate = lambda val: val.rfind(sub, start, end)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_split(
        self,
        pat: str | None = None,
        n: int | None = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook splitting strings into lists. Parameters: pat
        # (None means whitespace), n (maximum splits; -1 or 0 mean unlimited, default
        # -1), expand (unused), regex (whether pat is a regex; None or false means
        # literal). Uses the arrow split kernels. Returns a new array of lists.
        # KNOWN ISSUE: the expand parameter is accepted but never used.
        if n in {-1, 0}:
            n = None
        if pat is None:
            split_func = pc.utf8_split_whitespace
        elif regex:
            split_func = functools.partial(pc.split_pattern_regex, pattern=pat)
        else:
            split_func = functools.partial(pc.split_pattern, pattern=pat)
        return self._from_pyarrow_array(split_func(self._pa_array, max_splits=n))

    def _str_rsplit(self, pat: str | None = None, n: int | None = -1) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook splitting from the right. Parameters: pat (None
        # means whitespace) and n (maximum splits; -1 or 0 mean unlimited). Uses the
        # arrow split kernels in reverse mode; pat is always treated as a literal.
        # Returns a new array of lists.
        if n in {-1, 0}:
            n = None
        if pat is None:
            return self._from_pyarrow_array(
                pc.utf8_split_whitespace(self._pa_array, max_splits=n, reverse=True)
            )
        return self._from_pyarrow_array(
            pc.split_pattern(self._pa_array, pat, max_splits=n, reverse=True)
        )

    def _str_translate(self, table: dict[int, str]) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook applying a character translation table element by
        # element. Parameter table maps code points to replacement strings. Returns a
        # new array.
        predicate = lambda val: val.translate(table)
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_wrap(self, width: int, **kwargs) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook wrapping text element by element with a text
        # wrapper configured from the width and extra keyword arguments; lines are
        # joined with newlines. Returns a new array.
        kwargs["width"] = width
        tw = textwrap.TextWrapper(**kwargs)
        predicate = lambda val: "\n".join(tw.wrap(val))
        result = self._apply_elementwise(predicate)
        return self._from_pyarrow_array(pa.chunked_array(result))

    def _str_zfill(self, width: int) -> Self:
        # [FUNCTION DOC]
        # HELPER: string accessor hook zero-padding strings on the left to a width.
        # Parameter width. With pyarrow older than version 21, done element by element;
        # otherwise uses the arrow kernel. Returns a new array of this class.
        if pa_version_under21p0:
            predicate = lambda val: val.zfill(width)
            result = self._apply_elementwise(predicate)
            return type(self)(pa.chunked_array(result))
        return type(self)(pc.utf8_zfill(self._pa_array, width))

    @property
    def _dt_days(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; days component of timedeltas, as 32-bit
        # integers (nulls preserved). Computed from the timedelta components.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.days,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_hours(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; hours component of timedeltas, as 32-bit
        # integers.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.hours,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_minutes(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; minutes component of timedeltas, as 32-bit
        # integers.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.minutes,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_seconds(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; seconds component of timedeltas, as 32-bit
        # integers.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.seconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_milliseconds(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; milliseconds component of timedeltas, as
        # 32-bit integers.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.milliseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_microseconds(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; microseconds component of timedeltas, as
        # 32-bit integers.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.microseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_nanoseconds(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; nanoseconds component of timedeltas, as 32-bit
        # integers.
        return self._from_pyarrow_array(
            pa.array(
                self._to_timedeltaarray().components.nanoseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    def _dt_to_pytimedelta(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook converting durations to Python timedelta
        # objects. Takes the data as Python objects (pandas timedeltas for nanosecond
        # units, converted to Python timedeltas); nulls become None. Returns an object
        # numpy array.
        data = self._pa_array.to_pylist()
        if self._dtype.pyarrow_dtype.unit == "ns":
            data = [None if ts is None else ts.to_pytimedelta() for ts in data]
        return np.array(data, dtype=object)

    def _dt_total_seconds(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook giving the duration in seconds as floating
        # point. Casts to 64-bit integer and divides by the number of units per second
        # of the array's time unit. Returns a new array.
        unit = self._pa_array.type.unit
        unit_per_second = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        result = pc.divide(pc.cast(self._pa_array, pa.int64()), unit_per_second[unit])
        return self._from_pyarrow_array(result)

    def _dt_as_unit(self, unit: str) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook changing the time unit. Parameter unit (s, ms,
        # us, ns). Timestamps keep their zone; durations keep being durations; other
        # types raise a not-implemented error. The cast is unsafe, so values are
        # truncated rather than raising. Returns a new array.
        pa_type = self._pa_array.type
        if pa.types.is_timestamp(pa_type):
            target_type = pa.timestamp(unit, tz=pa_type.tz)
        elif pa.types.is_duration(pa_type):
            target_type = pa.duration(unit)
        else:
            raise NotImplementedError(f"as_unit not implemented for {pa_type}")
        # Use safe=False to allow truncation, matching pandas as_unit behavior
        result = pc.cast(self._pa_array, target_type, safe=False)
        return self._from_pyarrow_array(result)

    @property
    def _dt_year(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; year of each timestamp.
        result = pc.year(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_day(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; day of the month of each timestamp.
        result = pc.day(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_day_of_week(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; day of the week of each timestamp (Monday is
        # zero). Also bound to the alternative names for day of week and weekday.
        result = pc.day_of_week(self._pa_array)
        return self._from_pyarrow_array(result)

    _dt_dayofweek = _dt_day_of_week
    _dt_weekday = _dt_day_of_week

    @property
    def _dt_day_of_year(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; day of the year of each timestamp. Also bound
        # to the alternative name without underscores.
        result = pc.day_of_year(self._pa_array)
        return self._from_pyarrow_array(result)

    _dt_dayofyear = _dt_day_of_year

    @property
    def _dt_hour(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; hour of each timestamp.
        result = pc.hour(self._pa_array)
        return self._from_pyarrow_array(result)

    def _dt_isocalendar(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; ISO year, week and day as a struct per
        # timestamp, from the arrow kernel. Returns a new array of structs.
        result = pc.iso_calendar(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_leap_year(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the year of each timestamp is a leap
        # year.
        result = pc.is_leap_year(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_month_start(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the day of month is the first.
        result = pc.equal(pc.day(self._pa_array), 1)
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_month_end(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the timestamp falls on the last day of
        # its month, computed as the day count between the start of the day and the
        # ceiling to month being one.
        result = pc.equal(
            pc.days_between(
                pc.floor_temporal(self._pa_array, unit="day"),
                pc.ceil_temporal(self._pa_array, unit="month"),
            ),
            1,
        )
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_year_start(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the month is January and the day is
        # the first.
        result = pc.and_(
            pc.equal(pc.month(self._pa_array), 1),
            pc.equal(pc.day(self._pa_array), 1),
        )
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_year_end(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the month is December and the day is
        # the 31st.
        result = pc.and_(
            pc.equal(pc.month(self._pa_array), 12),
            pc.equal(pc.day(self._pa_array), 31),
        )
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_quarter_start(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the quarter floor equals the day
        # floor.
        result = pc.equal(
            pc.floor_temporal(self._pa_array, unit="quarter"),
            pc.floor_temporal(self._pa_array, unit="day"),
        )
        return self._from_pyarrow_array(result)

    @property
    def _dt_is_quarter_end(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; whether the timestamp falls on the last day of
        # its quarter, computed from the day count to the quarter ceiling.
        result = pc.equal(
            pc.days_between(
                pc.floor_temporal(self._pa_array, unit="day"),
                pc.ceil_temporal(self._pa_array, unit="quarter"),
            ),
            1,
        )
        return self._from_pyarrow_array(result)

    @property
    def _dt_days_in_month(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; number of days in the month, as the day
        # difference between the month floor and the month ceiling. Also bound to the
        # alternative name without underscores.
        result = pc.days_between(
            pc.floor_temporal(self._pa_array, unit="month"),
            pc.ceil_temporal(self._pa_array, unit="month"),
        )
        return self._from_pyarrow_array(result)

    _dt_daysinmonth = _dt_days_in_month

    @property
    def _dt_microsecond(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; microsecond of each timestamp, computed as the
        # arrow microsecond component plus the millisecond component times one thousand
        # (GH 59154).
        # GH 59154
        us = pc.microsecond(self._pa_array)
        ms_to_us = pc.multiply(pc.millisecond(self._pa_array), 1000)
        result = pc.add(us, ms_to_us)
        return self._from_pyarrow_array(result)

    @property
    def _dt_minute(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; minute of each timestamp.
        result = pc.minute(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_month(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; month of each timestamp.
        result = pc.month(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_nanosecond(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; nanosecond of each timestamp.
        result = pc.nanosecond(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_quarter(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; quarter of each timestamp.
        result = pc.quarter(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_second(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; second of each timestamp.
        result = pc.second(self._pa_array)
        return self._from_pyarrow_array(result)

    @property
    def _dt_date(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; casts timestamps to 32-bit dates.
        result = self._pa_array.cast(pa.date32())
        return self._from_pyarrow_array(result)

    @property
    def _dt_time(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; casts timestamps to 64-bit times, in
        # microseconds if the timestamp unit is microseconds or nanoseconds if
        # nanoseconds, and nanoseconds for coarser units.
        unit = (
            self.dtype.pyarrow_dtype.unit
            if self.dtype.pyarrow_dtype.unit in {"us", "ns"}
            else "ns"
        )
        result = self._pa_array.cast(pa.time64(unit))
        return self._from_pyarrow_array(result)

    @property
    def _dt_tz(self):
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; property returning the time zone object (or
        # None) for the arrow zone name.
        return timezones.maybe_get_tz(self.dtype.pyarrow_dtype.tz)

    @property
    def _dt_unit(self):
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; property returning the time unit string of the
        # timestamp type.
        return self.dtype.pyarrow_dtype.unit

    def _dt_normalize(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; floors timestamps to midnight (one-day
        # multiple).
        result = pc.floor_temporal(self._pa_array, 1, "day")
        return self._from_pyarrow_array(result)

    def _dt_strftime(self, format: str) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook formatting timestamps as strings. Parameter
        # format is a strftime format string. Returns a new array of strings.
        result = pc.strftime(self._pa_array, format=format)
        return self._from_pyarrow_array(result)

    def _round_temporally(
        self,
        method: Literal["ceil", "floor", "round"],
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: shared implementation of ceil, floor and round for timestamps.
        # Parameters: method (ceil, floor or round); freq (frequency offset or alias);
        # ambiguous and nonexistent (default raise; any other value raises a not-
        # implemented error). The frequency is converted to an offset (a value error if
        # invalid). Offset prefixes are mapped to arrow time units (year, quarter,
        # month, week, day, hour, minute, second, millisecond, microsecond, nanosecond);
        # an unsupported prefix raises a value error. The arrow temporal rounding kernel
        # is applied with the offset multiple. Returns a new array.
        if ambiguous != "raise":
            raise NotImplementedError("ambiguous is not supported.")
        if nonexistent != "raise":
            raise NotImplementedError("nonexistent is not supported.")
        offset = to_offset(freq)
        if offset is None:
            raise ValueError(f"Must specify a valid frequency: {freq}")
        pa_supported_unit = {
            "Y": "year",
            "YS": "year",
            "Q": "quarter",
            "QS": "quarter",
            "M": "month",
            "MS": "month",
            "W": "week",
            "D": "day",
            "h": "hour",
            "min": "minute",
            "s": "second",
            "ms": "millisecond",
            "us": "microsecond",
            "ns": "nanosecond",
        }
        unit = pa_supported_unit.get(offset._prefix, None)
        if unit is None:
            raise ValueError(f"{freq=} is not supported")
        multiple = offset.n
        rounding_method = getattr(pc, f"{method}_temporal")
        result = rounding_method(self._pa_array, multiple=multiple, unit=unit)
        return self._from_pyarrow_array(result)

    def _dt_ceil(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; rounds timestamps up to a frequency.
        # Parameters freq, ambiguous, nonexistent (see the shared rounding helper).
        return self._round_temporally("ceil", freq, ambiguous, nonexistent)

    def _dt_floor(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; rounds timestamps down to a frequency.
        # Parameters freq, ambiguous, nonexistent (see the shared rounding helper).
        return self._round_temporally("floor", freq, ambiguous, nonexistent)

    def _dt_round(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; rounds timestamps to the nearest frequency.
        # Parameters freq, ambiguous, nonexistent (see the shared rounding helper).
        return self._round_temporally("round", freq, ambiguous, nonexistent)

    def _dt_day_name(self, locale: str | None = None) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; weekday names via the strftime kernel.
        # Parameter locale (default None, meaning the C locale). Returns a new array of
        # strings.
        if locale is None:
            locale = "C"
        result = pc.strftime(self._pa_array, format="%A", locale=locale)
        return self._from_pyarrow_array(result)

    def _dt_month_name(self, locale: str | None = None) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook; month names via the strftime kernel. Parameter
        # locale (default None, meaning the C locale). Returns a new array of strings.
        if locale is None:
            locale = "C"
        result = pc.strftime(self._pa_array, format="%B", locale=locale)
        return self._from_pyarrow_array(result)

    def _dt_to_pydatetime(self) -> Series:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook converting timestamps to Python datetime
        # objects. Raises a value error for date types. Takes the data as Python objects
        # (pandas timestamps for nanosecond units, converted without warning). Returns a
        # series of object dtype.
        from pandas import Series

        if pa.types.is_date(self.dtype.pyarrow_dtype):
            raise ValueError(
                f"to_pydatetime cannot be called with {self.dtype.pyarrow_dtype} type. "
                "Convert to pyarrow timestamp type."
            )
        data = self._pa_array.to_pylist()
        if self._dtype.pyarrow_dtype.unit == "ns":
            data = [None if ts is None else ts.to_pydatetime(warn=False) for ts in data]
        return Series(data, dtype=object)

    def _dt_tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook attaching or removing a time zone. Parameters:
        # tz (zone or None); ambiguous (only raise supported, else not-implemented
        # error); nonexistent (raise, shift backward or shift forward, mapped to the
        # arrow earliest/latest options; any other value raises a not-implemented
        # error). With tz None, converts to local wall-clock timestamps; otherwise
        # assumes the wall times are in the given zone (by name). Returns a new array.
        if ambiguous != "raise":
            raise NotImplementedError(f"{ambiguous=} is not supported")
        nonexistent_pa = {
            "raise": "raise",
            "shift_backward": "earliest",
            "shift_forward": "latest",
        }.get(
            nonexistent,  # type: ignore[arg-type]
            None,
        )
        if nonexistent_pa is None:
            raise NotImplementedError(f"{nonexistent=} is not supported")
        if tz is None:
            result = pc.local_timestamp(self._pa_array)
        else:
            result = pc.assume_timezone(
                self._pa_array, str(tz), ambiguous=ambiguous, nonexistent=nonexistent_pa
            )
        return self._from_pyarrow_array(result)

    def _dt_tz_convert(self, tz) -> Self:
        # [FUNCTION DOC]
        # HELPER: datetime accessor hook converting to another time zone, keeping the
        # unit. Raises a type error for timezone-naive data. Parameter tz. Returns a new
        # array.
        if self.dtype.pyarrow_dtype.tz is None:
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )
        current_unit = self.dtype.pyarrow_dtype.unit
        result = self._pa_array.cast(pa.timestamp(current_unit, tz))
        return self._from_pyarrow_array(result)


def transpose_homogeneous_pyarrow(
    arrays: Sequence[ArrowExtensionArray],
) -> list[ArrowExtensionArray]:
    # [FUNCTION DOC]
    # Faster transpose of a list of arrow extension arrays that have equal length and
    # the same dtype (the caller must ensure this; nothing is validated). Parameter
    # arrays is the input sequence. Combines all chunks into one chunked array, reorders
    # it with a computed index so that row-major order becomes column-major, and slices
    # out one array per row. Returns a list with one array per original position, each
    # with as many elements as there were input arrays.
    """Transpose arrow extension arrays in a list, but faster.

    Input should be a list of arrays of equal length and all have the same
    dtype. The caller is responsible for ensuring validity of input data.
    """
    arrays = list(arrays)
    nrows, ncols = len(arrays[0]), len(arrays)
    indices = np.arange(nrows * ncols).reshape(ncols, nrows).T.reshape(-1)
    arr = pa.chunked_array([chunk for arr in arrays for chunk in arr._pa_array.chunks])
    arr = arr.take(indices)
    return [ArrowExtensionArray(arr.slice(i * ncols, ncols)) for i in range(nrows)]
