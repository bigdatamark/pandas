"""
Utility functions and objects for implementing the interchange API.
"""
# [MODULE DOC]
# HELPER: internal support module for the interchange implementation.
# Contents: a mapping from pyarrow type names to Arrow C data interface format
# strings; an enum-like class of Arrow C format strings; an enum-like class of
# byte-order characters; a function converting a pandas dtype to an Arrow C
# format string; and a function that merges multi-chunk pyarrow-backed series.
# Used by the column and from_dataframe modules. No import-time side effects
# beyond importing pandas itself.
# KNOWN ISSUE: in the pyarrow type-name mapping the entry for the signed 16-bit integer uses the uppercase letter that denotes unsigned 16-bit in the Arrow C format (the same value as the unsigned entry); the signed format letter is lowercase, as defined in the format class below.

from __future__ import annotations

import typing

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
)

import pandas as pd

if typing.TYPE_CHECKING:
    from pandas._typing import DtypeObj


# Maps str(pyarrow.DataType) = C type format string
# Currently, no pyarrow API for this
PYARROW_CTYPES = {
    "null": "n",
    "bool": "b",
    "uint8": "C",
    "uint16": "S",
    "uint32": "I",
    "uint64": "L",
    "int8": "c",
    "int16": "S",
    "int32": "i",
    "int64": "l",
    "halffloat": "e",  # float16
    "float": "f",  # float32
    "double": "g",  # float64
    "string": "u",
    "large_string": "U",
    "binary": "z",
    "time32[s]": "tts",
    "time32[ms]": "ttm",
    "time64[us]": "ttu",
    "time64[ns]": "ttn",
    "date32[day]": "tdD",
    "date64[ms]": "tdm",
    "timestamp[s]": "tss:",
    "timestamp[ms]": "tsm:",
    "timestamp[us]": "tsu:",
    "timestamp[ns]": "tsn:",
    "duration[s]": "tDs",
    "duration[ms]": "tDm",
    "duration[us]": "tDu",
    "duration[ns]": "tDn",
}


class ArrowCTypes:
    # [CLASS DOC]
    # HELPER: namespace of string constants (no instances intended) holding Apache
    # Arrow C data interface format strings for null, boolean, signed and unsigned
    # integers, floats, strings, dates, plus template strings for timestamp (taking
    # a resolution letter and a timezone) and time (taking a resolution letter).
    """
    Enum for Apache Arrow C type format strings.

    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """

    NULL = "n"
    BOOL = "b"
    INT8 = "c"
    UINT8 = "C"
    INT16 = "s"
    UINT16 = "S"
    INT32 = "i"
    UINT32 = "I"
    INT64 = "l"
    UINT64 = "L"
    FLOAT16 = "e"
    FLOAT32 = "f"
    FLOAT64 = "g"
    STRING = "u"  # utf-8
    LARGE_STRING = "U"  # utf-8
    DATE32 = "tdD"
    DATE64 = "tdm"
    # Resoulution:
    #   - seconds -> 's'
    #   - milliseconds -> 'm'
    #   - microseconds -> 'u'
    #   - nanoseconds -> 'n'
    TIMESTAMP = "ts{resolution}:{tz}"
    TIME = "tt{resolution}"


class Endianness:
    # [CLASS DOC]
    # HELPER: namespace of string constants (no instances intended) naming the
    # byte-order characters little, big, native and not-applicable.
    """Enum indicating the byte-order of a data-type."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"


def dtype_to_arrow_c_fmt(dtype: DtypeObj) -> str:
    # [FUNCTION DOC]
    # Convert a pandas or numpy dtype object into its Apache Arrow C format string.
    # Parameter dtype: the dtype to describe. Returns a string.
    # Rules in order: categorical dtypes map to the signed 64-bit integer code (the
    # codes format); object dtype maps to the utf-8 string code; Arrow-backed dtypes
    # map decimals to a decimal string with precision and scale, tz-aware
    # timestamps to a timestamp string with the first letter of the unit and the
    # timezone, and otherwise use the pyarrow type name lookup table when it has an
    # entry; then any dtype whose name, uppercased, is a constant of the format
    # class is mapped to it; then the string extension dtype maps to the utf-8 code
    # (a pending note says large string should be used for pyarrow storage);
    # numpy datetimes map to a timestamp string with the first letter of the
    # resolution and empty timezone; tz-aware datetime dtypes map to a timestamp
    # string with unit letter and timezone; nullable boolean maps to the boolean
    # code. Raises a not-implemented error for anything else.
    """
    Represent pandas `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : np.dtype
        Datatype of pandas DataFrame to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
    if isinstance(dtype, CategoricalDtype):
        return ArrowCTypes.INT64
    elif dtype == np.dtype("O"):
        return ArrowCTypes.STRING
    elif isinstance(dtype, ArrowDtype):
        import pyarrow as pa

        pa_type = dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_type):
            return f"d:{pa_type.precision},{pa_type.scale}"
        elif pa.types.is_timestamp(pa_type) and pa_type.tz is not None:
            return f"ts{pa_type.unit[0]}:{pa_type.tz}"
        format_str = PYARROW_CTYPES.get(str(pa_type), None)
        if format_str is not None:
            return format_str

    format_str = getattr(ArrowCTypes, dtype.name.upper(), None)
    if format_str is not None:
        return format_str

    if isinstance(dtype, pd.StringDtype):
        # TODO(infer_string) this should be LARGE_STRING for pyarrow storage,
        # but current tests don't cover this distinction
        return ArrowCTypes.STRING

    elif lib.is_np_dtype(dtype, "M"):
        # Selecting the first char of resolution string:
        # dtype.str -> '<M8[ns]' -> 'n'
        resolution = np.datetime_data(dtype)[0][0]
        return ArrowCTypes.TIMESTAMP.format(resolution=resolution, tz="")

    elif isinstance(dtype, DatetimeTZDtype):
        return ArrowCTypes.TIMESTAMP.format(resolution=dtype.unit[0], tz=dtype.tz)

    elif isinstance(dtype, pd.BooleanDtype):
        return ArrowCTypes.BOOL

    raise NotImplementedError(
        f"Conversion of {dtype} to Arrow C format string is not implemented."
    )


def maybe_rechunk(series: pd.Series, *, allow_copy: bool) -> pd.Series | None:
    # [FUNCTION DOC]
    # Rechunk a pyarrow-backed series into a single chunk if needed.
    # Parameters: series (a Series); allow_copy (keyword-only, required boolean).
    # Returns None when the dtype is not Arrow-backed or the underlying chunked
    # array already has exactly one chunk. Otherwise, if allow_copy is false, raises
    # a runtime error asking the caller to rechunk first. If allowed, combines the
    # chunks and returns a new Series with the same dtype, name and index.
    """
    Rechunk a multi-chunk pyarrow array into a single-chunk array, if necessary.

    - Returns `None` if the input series is not backed by a multi-chunk pyarrow array
      (and so doesn't need rechunking)
    - Returns a single-chunk-backed-Series if the input is backed by a multi-chunk
      pyarrow array and `allow_copy` is `True`.
    - Raises a `RuntimeError` if `allow_copy` is `False` and input is a
      based by a multi-chunk pyarrow array.
    """
    if not isinstance(series.dtype, pd.ArrowDtype):
        return None
    chunked_array = series.array._pa_array  # type: ignore[attr-defined]
    if len(chunked_array.chunks) == 1:
        return None
    if not allow_copy:
        raise RuntimeError(
            "Found multi-chunk pyarrow array, but `allow_copy` is False. "
            "Please rechunk the array before calling this function, or set "
            "`allow_copy=True`."
        )
    arr = chunked_array.combine_chunks()
    return pd.Series(arr, dtype=series.dtype, name=series.name, index=series.index)
