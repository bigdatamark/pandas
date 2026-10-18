"""
A verbatim copy (vendored) of the spec tests.
Taken from https://github.com/data-apis/dataframe-api
"""

# [MODULE DOC]
# Vendored copy of the dataframe-api specification conformance tests, applied to the pandas interchange implementation.
# Every test builds a frame, obtains the interchange object (which emits an interchange warning that is asserted) and checks the spec-mandated properties of columns, chunks, categorical description and buffers. The tests intentionally use raw integer dtype kind codes instead of importing the spec, because the file is meant to be vendored anywhere.

import ctypes
import math

import pytest

import pandas as pd
import pandas._testing as tm


@pytest.fixture
def df_from_dict():
    # [FUNCTION DOC]
    # Fixture returning a factory that builds a frame from a column mapping.
    # The factory takes the mapping of column names to values and an optional flag (default false) to convert all columns to categorical dtype; it returns the constructed frame.
    def maker(dct, is_categorical=False):
        # [FUNCTION DOC]
        # HELPER: factory returned by the frame-building fixture; builds a frame from the mapping and converts every column to categorical dtype when the flag is true, otherwise returns it unchanged.
        df = pd.DataFrame(dct)
        return df.astype("category") if is_categorical else df

    return maker


@pytest.mark.parametrize(
    "test_data",
    [
        {"a": ["foo", "bar"], "b": ["baz", "qux"]},
        {"a": [1.5, 2.5, 3.5], "b": [9.2, 10.5, 11.8]},
        {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]},
    ],
    ids=["str_data", "float_data", "int_data"],
)
def test_only_one_dtype(test_data, df_from_dict):
    # [FUNCTION DOC]
    # Verifies a frame whose columns share one dtype (strings, floats or integers) through the interchange object.
    # Parametrized by three data sets with ids. For each column checks the null count is zero and is a Python integer, the size equals the row count and the offset is zero. Constructing the interchange object must emit the interchange warning.
    columns = list(test_data.keys())
    df = df_from_dict(test_data)
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()

    column_size = len(test_data[columns[0]])
    for column in columns:
        null_count = dfX.get_column_by_name(column).null_count
        assert null_count == 0
        assert isinstance(null_count, int)
        assert dfX.get_column_by_name(column).size() == column_size
        assert dfX.get_column_by_name(column).offset == 0


def test_mixed_dtypes(df_from_dict):
    # [FUNCTION DOC]
    # Verifies a frame with integer, float, boolean and string columns: each column has zero nulls, size three, offset zero and the spec dtype kind code (integer 0, float 2, boolean 20, string 21). The float column bit width is 64.
    df = df_from_dict(
        {
            "a": [1, 2, 3],  # dtype kind INT = 0
            "b": [3, 4, 5],  # dtype kind INT = 0
            "c": [1.5, 2.5, 3.5],  # dtype kind FLOAT = 2
            "d": [9, 10, 11],  # dtype kind INT = 0
            "e": [True, False, True],  # dtype kind BOOLEAN = 20
            "f": ["a", "", "c"],  # dtype kind STRING = 21
        }
    )
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    # for meanings of dtype[0] see the spec; we cannot import the spec here as this
    # file is expected to be vendored *anywhere*;
    # values for dtype[0] are explained above
    columns = {"a": 0, "b": 0, "c": 2, "d": 0, "e": 20, "f": 21}

    for column, kind in columns.items():
        colX = dfX.get_column_by_name(column)
        assert colX.null_count == 0
        assert isinstance(colX.null_count, int)
        assert colX.size() == 3
        assert colX.offset == 0

        assert colX.dtype[0] == kind

    assert dfX.get_column_by_name("c").dtype[1] == 64


def test_na_float(df_from_dict):
    # [FUNCTION DOC]
    # Verifies that the null count of a float column containing a missing value is one and is a Python integer.
    df = df_from_dict({"a": [1.0, math.nan, 2.0]})
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    colX = dfX.get_column_by_name("a")
    assert colX.null_count == 1
    assert isinstance(colX.null_count, int)


def test_noncategorical(df_from_dict):
    # [FUNCTION DOC]
    # Verifies that asking a non-categorical column for its categorical description raises a type error whose message mentions categorical.
    df = df_from_dict({"a": [1, 2, 3]})
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    colX = dfX.get_column_by_name("a")
    with pytest.raises(TypeError, match=".*categorical.*"):
        colX.describe_categorical


def test_categorical(df_from_dict):
    # [FUNCTION DOC]
    # Verifies a categorical column of weekday names: the categorical description reports an ordered flag and a dictionary flag, both Python booleans.
    df = df_from_dict(
        {"weekday": ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]},
        is_categorical=True,
    )

    with tm.assert_produces_warning(match="Interchange"):
        colX = df.__dataframe__().get_column_by_name("weekday")
    categorical = colX.describe_categorical
    assert isinstance(categorical["is_ordered"], bool)
    assert isinstance(categorical["is_dictionary"], bool)


def test_dataframe(df_from_dict):
    # [FUNCTION DOC]
    # Verifies frame-level properties of the interchange object for a three-column, three-row frame: three columns, three rows, one chunk, the column names in order, and that selecting columns by position yields the same names as selecting by name.
    df = df_from_dict(
        {"x": [True, True, False], "y": [1, 2, 0], "z": [9.2, 10.5, 11.8]}
    )
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()

    assert dfX.num_columns() == 3
    assert dfX.num_rows() == 3
    assert dfX.num_chunks() == 1
    assert list(dfX.column_names()) == ["x", "y", "z"]
    assert list(dfX.select_columns((0, 2)).column_names()) == list(
        dfX.select_columns_by_name(("x", "z")).column_names()
    )


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
def test_df_get_chunks(size, n_chunks, df_from_dict):
    # [FUNCTION DOC]
    # Verifies splitting the interchange frame into the requested number of chunks.
    # Parametrized over (row count, requested chunks) of (10, 3), (12, 3) and (12, 5). Expected: exactly the requested number of chunks is returned and their row counts sum to the original row count.
    df = df_from_dict({"x": list(range(size))})
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    chunks = list(dfX.get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum(chunk.num_rows() for chunk in chunks) == size


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
def test_column_get_chunks(size, n_chunks, df_from_dict):
    # [FUNCTION DOC]
    # Same as the frame-chunks test but for a single column: splitting the first column into the requested chunk count gives that many chunks whose sizes sum to the row count. Same parametrization.
    df = df_from_dict({"x": list(range(size))})
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    chunks = list(dfX.get_column(0).get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum(chunk.size() for chunk in chunks) == size


def test_get_columns(df_from_dict):
    # [FUNCTION DOC]
    # Verifies column enumeration: each of the two columns reports size two and one chunk, and the kind codes of the first and second columns are integer (0) and float (2).
    df = df_from_dict({"a": [0, 1], "b": [2.5, 3.5]})
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    for colX in dfX.get_columns():
        assert colX.size() == 2
        assert colX.num_chunks() == 1
    # for meanings of dtype[0] see the spec; we cannot import the spec here as this
    # file is expected to be vendored *anywhere*
    assert dfX.get_column(0).dtype[0] == 0  # INT
    assert dfX.get_column(1).dtype[0] == 2  # FLOAT


def test_buffer(df_from_dict):
    # [FUNCTION DOC]
    # Verifies the data buffer of an integer column holding negative and positive values.
    # Checks the buffer size is positive and the pointer is non-null, the dtype kind is integer, and, only when the buffer device code indicates CPU, reads each element straight from memory using the declared bit width and compares with the source values, reporting the index of any mismatch.
    arr = [0, 1, -1]
    df = df_from_dict({"a": arr})
    with tm.assert_produces_warning(match="Interchange"):
        dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    bufX = colX.get_buffers()

    dataBuf, dataDtype = bufX["data"]

    assert dataBuf.bufsize > 0
    assert dataBuf.ptr != 0
    device, _ = dataBuf.__dlpack_device__()

    # for meanings of dtype[0] see the spec; we cannot import the spec here as this
    # file is expected to be vendored *anywhere*
    assert dataDtype[0] == 0  # INT

    if device == 1:  # CPU-only as we're going to directly read memory here
        bitwidth = dataDtype[1]
        ctype = {
            8: ctypes.c_int8,
            16: ctypes.c_int16,
            32: ctypes.c_int32,
            64: ctypes.c_int64,
        }[bitwidth]

        for idx, truth in enumerate(arr):
            val = ctype.from_address(dataBuf.ptr + idx * (bitwidth // 8)).value
            assert val == truth, f"Buffer at index {idx} mismatch"
