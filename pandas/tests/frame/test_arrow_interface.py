# [MODULE DOC]
# Tests for the Arrow PyCapsule interface of DataFrame: exporting as an Arrow stream and constructing a frame
# from Arrow-compatible objects. Skipped entirely if pyarrow is not installed (import-time skip). Defines two small
# wrapper classes that expose only the Arrow capsule protocols.

import ctypes

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

pa = pytest.importorskip("pyarrow")


@td.skip_if_no("pyarrow", min_version="14.0")
def test_dataframe_arrow_interface(using_infer_string):
    # [FUNCTION DOC]
    # Needs pyarrow 14 or newer. The stream capsule export returns a valid capsule named arrow_array_stream. Converting the frame
    # with pyarrow's table constructor gives a table equal to the expected one (large string type when string inference is on, standard string otherwise), and when given a target schema the table is cast to it.
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    capsule = df.__arrow_c_stream__()
    assert (
        ctypes.pythonapi.PyCapsule_IsValid(
            ctypes.py_object(capsule), b"arrow_array_stream"
        )
        == 1
    )

    table = pa.table(df)
    string_type = pa.large_string() if using_infer_string else pa.string()
    expected = pa.table({"a": [1, 2, 3], "b": pa.array(["a", "b", "c"], string_type)})
    assert table.equals(expected)

    schema = pa.schema([("a", pa.int8()), ("b", pa.string())])
    table = pa.table(df, schema=schema)
    expected = expected.cast(schema)
    assert table.equals(expected)


@td.skip_if_no("pyarrow", min_version="15.0")
def test_dataframe_to_arrow(using_infer_string):
    # [FUNCTION DOC]
    # Needs pyarrow 15 or newer. Reading a record-batch reader created from the frame's stream gives the same table as expected, also with a requested schema (cast to 8-bit integer and string).
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    table = pa.RecordBatchReader.from_stream(df).read_all()
    string_type = pa.large_string() if using_infer_string else pa.string()
    expected = pa.table({"a": [1, 2, 3], "b": pa.array(["a", "b", "c"], string_type)})
    assert table.equals(expected)

    schema = pa.schema([("a", pa.int8()), ("b", pa.string())])
    table = pa.RecordBatchReader.from_stream(df, schema=schema).read_all()
    expected = expected.cast(schema)
    assert table.equals(expected)


class ArrowArrayWrapper:
    # [CLASS DOC]
    # HELPER: test wrapper holding a record batch; it exposes only the Arrow array capsule protocol, delegating to the held batch, so tests can show non-pyarrow objects are accepted.
    def __init__(self, batch):
        # [FUNCTION DOC]
        # HELPER: stores the given record batch in the array attribute.
        self.array = batch

    def __arrow_c_array__(self, requested_schema=None):
        # [FUNCTION DOC]
        # HELPER: Arrow array capsule protocol method. Takes an optional requested schema (default none) and returns whatever the wrapped batch returns for the same request.
        return self.array.__arrow_c_array__(requested_schema)


class ArrowStreamWrapper:
    # [CLASS DOC]
    # HELPER: test wrapper holding a table; it exposes only the Arrow stream capsule protocol, delegating to the held table.
    def __init__(self, table):
        # [FUNCTION DOC]
        # HELPER: stores the given table in the stream attribute.
        self.stream = table

    def __arrow_c_stream__(self, requested_schema=None):
        # [FUNCTION DOC]
        # HELPER: Arrow stream capsule protocol method. Takes an optional requested schema (default none) and returns whatever the wrapped table returns for the same request.
        return self.stream.__arrow_c_stream__(requested_schema)


@td.skip_if_no("pyarrow", min_version="14.0")
def test_dataframe_from_arrow(using_infer_string):
    # [FUNCTION DOC]
    # Needs pyarrow 14 or newer. Frame construction from Arrow data: from a table (strings become the string dtype backed by NaN-missing semantics unless string inference is on), from a wrapper that only implements the stream protocol, and from a wrapper that only implements the array protocol. A plain dictionary raises a type error saying an Arrow-compatible tabular object was expected.
    # KNOWN ISSUE: in the part labeled as covering objects with the array protocol, the first check again passes the table rather than the batch, so it repeats the earlier table check.
    # objects with __arrow_c_stream__
    table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    result = pd.DataFrame.from_arrow(table)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    if not using_infer_string:
        expected["b"] = expected["b"].astype(pd.StringDtype(na_value=np.nan))
    tm.assert_frame_equal(result, expected)

    # not only pyarrow object are supported
    result = pd.DataFrame.from_arrow(ArrowStreamWrapper(table))
    tm.assert_frame_equal(result, expected)

    # objects with __arrow_c_array__
    batch = pa.record_batch([[1, 2, 3], ["a", "b", "c"]], names=["a", "b"])

    result = pd.DataFrame.from_arrow(table)
    tm.assert_frame_equal(result, expected)

    result = pd.DataFrame.from_arrow(ArrowArrayWrapper(batch))
    tm.assert_frame_equal(result, expected)

    # only accept actual Arrow objects
    with pytest.raises(TypeError, match="Expected an Arrow-compatible tabular object"):
        pd.DataFrame.from_arrow({"a": [1, 2, 3], "b": ["a", "b", "c"]})
