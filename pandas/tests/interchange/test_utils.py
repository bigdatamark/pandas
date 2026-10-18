# [MODULE DOC]
# Tests for the utility that maps a dtype to its Arrow C data interface format string, as used by the dataframe interchange implementation.
# Covers numpy dtypes, pandas dtypes and pyarrow-backed extension dtypes. The reference strings are hard-coded because there is no pyarrow API to obtain the format string from Python (see the note in the module about exporting the schema to a C pointer).

import numpy as np
import pytest

import pandas as pd
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt

# TODO: use ArrowSchema to get reference C-string.
# At the time, there is no way to access ArrowSchema holding a type format string
# from python. The only way to access it is to export the structure to a C-pointer,
# see DataType._export_to_c() method defined in
# https://github.com/apache/arrow/blob/master/python/pyarrow/types.pxi


@pytest.mark.parametrize(
    "pandas_dtype, c_string",
    [
        (np.dtype("bool"), "b"),
        (np.dtype("int8"), "c"),
        (np.dtype("uint8"), "C"),
        (np.dtype("int16"), "s"),
        (np.dtype("uint16"), "S"),
        (np.dtype("int32"), "i"),
        (np.dtype("uint32"), "I"),
        (np.dtype("int64"), "l"),
        (np.dtype("uint64"), "L"),
        (np.dtype("float16"), "e"),
        (np.dtype("float32"), "f"),
        (np.dtype("float64"), "g"),
        (pd.Series(["a"]).dtype, "u"),
        (
            pd.Series([0]).astype("datetime64[ns]").dtype,
            "tsn:",
        ),
        (pd.CategoricalDtype(["a"]), "l"),
        (np.dtype("O"), "u"),
    ],
)
def test_dtype_to_arrow_c_fmt(pandas_dtype, c_string):  # PR01
    # [FUNCTION DOC]
    # Verifies that the format-string utility returns the expected Arrow C format string for numpy and pandas dtypes.
    # Parametrized over (dtype, expected string) pairs: boolean, signed and unsigned integers of all widths, three float widths, the default string dtype, nanosecond datetime, categorical (formatted as the integer code type) and object dtype (formatted as a string). Asserts exact equality of the returned string.
    """Test ``dtype_to_arrow_c_fmt`` utility function."""
    assert dtype_to_arrow_c_fmt(pandas_dtype) == c_string


@pytest.mark.parametrize(
    "pa_dtype, args_kwargs, c_string",
    [
        ["null", {}, "n"],
        ["bool_", {}, "b"],
        ["uint8", {}, "C"],
        ["uint16", {}, "S"],
        ["uint32", {}, "I"],
        ["uint64", {}, "L"],
        ["int8", {}, "c"],
        ["int16", {}, "S"],
        ["int32", {}, "i"],
        ["int64", {}, "l"],
        ["float16", {}, "e"],
        ["float32", {}, "f"],
        ["float64", {}, "g"],
        ["string", {}, "u"],
        ["binary", {}, "z"],
        ["time32", ("s",), "tts"],
        ["time32", ("ms",), "ttm"],
        ["time64", ("us",), "ttu"],
        ["time64", ("ns",), "ttn"],
        ["date32", {}, "tdD"],
        ["date64", {}, "tdm"],
        ["timestamp", {"unit": "s"}, "tss:"],
        ["timestamp", {"unit": "ms"}, "tsm:"],
        ["timestamp", {"unit": "us"}, "tsu:"],
        ["timestamp", {"unit": "ns"}, "tsn:"],
        ["timestamp", {"unit": "ns", "tz": "UTC"}, "tsn:UTC"],
        ["duration", ("s",), "tDs"],
        ["duration", ("ms",), "tDm"],
        ["duration", ("us",), "tDu"],
        ["duration", ("ns",), "tDn"],
        ["decimal128", {"precision": 4, "scale": 2}, "d:4,2"],
    ],
)
def test_dtype_to_arrow_c_fmt_arrowdtype(pa_dtype, args_kwargs, c_string):
    # [FUNCTION DOC]
    # Verifies the format string for pyarrow-backed extension dtypes (issue 52323). Skipped when pyarrow is not installed.
    # Parametrized over (pyarrow factory name, arguments, expected string); arguments are empty, a positional tuple or a keyword mapping and are used to construct the pyarrow type by name, which is then wrapped in the pandas Arrow extension dtype. Covers null, boolean, integers, floats, string, binary, time, date, timestamp (with and without time zone), duration and decimal types. Asserts exact equality with the expected string.
    # KNOWN ISSUE: the case for the signed 16-bit integer expects the unsigned 16-bit format letter, but the Arrow specification uses the lowercase letter for signed 16-bit; the expectation mirrors the same mistake in the lookup table of the implementation rather than the specification.
    # GH 52323
    pa = pytest.importorskip("pyarrow")
    if not args_kwargs:
        pa_type = getattr(pa, pa_dtype)()
    elif isinstance(args_kwargs, tuple):
        pa_type = getattr(pa, pa_dtype)(*args_kwargs)
    else:
        pa_type = getattr(pa, pa_dtype)(**args_kwargs)
    arrow_type = pd.ArrowDtype(pa_type)
    assert dtype_to_arrow_c_fmt(arrow_type) == c_string
