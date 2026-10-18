# [MODULE DOC]
# Tests for the Arrow PyCapsule interface of Series: exporting a Series as an Arrow C stream and importing a Series from Arrow-compatible objects.
# Requires pyarrow of at least version 16 at import time: the module is skipped entirely (import-or-skip) when it is missing or older, and binds the imported library at module level. Defines two small wrapper classes exposing only the C array or C stream protocol.

import ctypes

import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

pa = pytest.importorskip("pyarrow", minversion="16.0")


def test_series_arrow_interface():
    # [FUNCTION DOC]
    # Verifies that exporting an integer Series to the Arrow C stream protocol gives a valid capsule named for an Arrow array stream, and that building a chunked array from the Series gives the expected int64 data, also when a target type of 32-bit integer is requested.
    s = pd.Series([1, 4, 2])

    capsule = s.__arrow_c_stream__()
    assert (
        ctypes.pythonapi.PyCapsule_IsValid(
            ctypes.py_object(capsule), b"arrow_array_stream"
        )
        == 1
    )

    ca = pa.chunked_array(s)
    expected = pa.chunked_array([[1, 4, 2]])
    assert ca.equals(expected)
    ca = pa.chunked_array(s, type=pa.int32())
    expected = pa.chunked_array([[1, 4, 2]], type=pa.int32())
    assert ca.equals(expected)


def test_series_arrow_interface_arrow_dtypes():
    # [FUNCTION DOC]
    # Same as the plain export test but for a pyarrow-backed Int64 Series: valid capsule, equal chunked array, and the requested 32-bit target type honored.
    s = pd.Series([1, 4, 2], dtype="Int64[pyarrow]")

    capsule = s.__arrow_c_stream__()
    assert (
        ctypes.pythonapi.PyCapsule_IsValid(
            ctypes.py_object(capsule), b"arrow_array_stream"
        )
        == 1
    )

    ca = pa.chunked_array(s)
    expected = pa.chunked_array([[1, 4, 2]])
    assert ca.equals(expected)
    ca = pa.chunked_array(s, type=pa.int32())
    expected = pa.chunked_array([[1, 4, 2]], type=pa.int32())
    assert ca.equals(expected)


def test_series_arrow_interface_stringdtype():
    # [FUNCTION DOC]
    # Verifies the export of a pyarrow-backed string Series: the capsule is valid and the chunked array built from it holds the strings with the large string Arrow type.
    s = pd.Series(["foo", "bar"], dtype="string[pyarrow]")

    capsule = s.__arrow_c_stream__()
    assert (
        ctypes.pythonapi.PyCapsule_IsValid(
            ctypes.py_object(capsule), b"arrow_array_stream"
        )
        == 1
    )

    ca = pa.chunked_array(s)
    expected = pa.chunked_array([["foo", "bar"]], type=pa.large_string())
    assert ca.equals(expected)


class ArrowArrayWrapper:
    # [CLASS DOC]
    # HELPER: test wrapper holding an Arrow array in an attribute and exposing only the Arrow C array protocol, to check that import does not require a native pyarrow object.
    def __init__(self, array):
        # [FUNCTION DOC]
        # HELPER: stores the wrapped Arrow array.
        self.array = array

    def __arrow_c_array__(self, requested_schema=None):
        # [FUNCTION DOC]
        # HELPER: delegates the C array protocol to the wrapped array, passing along the optional requested schema (default none), and returns its result.
        return self.array.__arrow_c_array__(requested_schema)


class ArrowStreamWrapper:
    # [CLASS DOC]
    # HELPER: test wrapper holding an Arrow chunked array in an attribute and exposing only the Arrow C stream protocol.
    def __init__(self, chunked_array):
        # [FUNCTION DOC]
        # HELPER: stores the wrapped chunked array.
        self.stream = chunked_array

    def __arrow_c_stream__(self, requested_schema=None):
        # [FUNCTION DOC]
        # HELPER: delegates the C stream protocol to the wrapped chunked array, passing along the optional requested schema (default none), and returns its result.
        return self.stream.__arrow_c_stream__(requested_schema)


@td.skip_if_no("pyarrow", min_version="14.0")
def test_dataframe_from_arrow():
    # [FUNCTION DOC]
    # Verifies the Series constructor from Arrow-compatible objects (requires pyarrow 14 or newer): a chunked array and a wrapper with only the stream protocol give an equal integer Series; a table is interpreted as a struct-valued Series of dictionaries; an array and a wrapper with only the array protocol give an equal integer Series; and a plain list raises a type error saying an Arrow-compatible array-like object was expected.
    # KNOWN ISSUE: the test name refers to a data frame but the code under test is the Series import method.
    # objects with __arrow_c_stream__
    arr = pa.chunked_array([[1, 2, 3], [4, 5]])

    result = pd.Series.from_arrow(arr)
    expected = pd.Series([1, 2, 3, 4, 5])
    tm.assert_series_equal(result, expected)

    # not only pyarrow object are supported
    result = pd.Series.from_arrow(ArrowStreamWrapper(arr))
    tm.assert_series_equal(result, expected)

    # table works as well, but will be seen as a StructArray
    table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    result = pd.Series.from_arrow(table)
    expected = pd.Series([{"a": 1, "b": "a"}, {"a": 2, "b": "b"}, {"a": 3, "b": "c"}])
    tm.assert_series_equal(result, expected)

    # objects with __arrow_c_array__
    arr = pa.array([1, 2, 3])

    expected = pd.Series([1, 2, 3])
    result = pd.Series.from_arrow(arr)
    tm.assert_series_equal(result, expected)

    result = pd.Series.from_arrow(ArrowArrayWrapper(arr))
    tm.assert_series_equal(result, expected)

    # only accept actual Arrow objects
    with pytest.raises(
        TypeError, match="Expected an Arrow-compatible array-like object"
    ):
        pd.Series.from_arrow([1, 2, 3])
