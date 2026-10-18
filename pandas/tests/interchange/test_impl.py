# [MODULE DOC]
# Tests for the dataframe interchange protocol implementation: producing the interchange object from a pandas frame (columns, null descriptions, buffers, dtypes, chunks) and consuming interchange objects (pandas or pyarrow) back into frames with the conversion function.
# Public targets under test come from the core interchange package: the column class, the null-type and dtype-kind enumerations, the from-dataframe converter and the Arrow C format constants. Most tests assert that creating the interchange object emits the interchange deprecation-style warning. Many tests are skipped when a minimum pyarrow version is missing. No import-time side effects.

from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas.compat import (
    is_ci_environment,
    is_platform_windows,
)
from pandas.compat.pyarrow import pa_version_under22p0

import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
    ColumnNullType,
    DtypeKind,
)
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes


@pytest.mark.parametrize("data", [("ordered", True), ("unordered", False)])
def test_categorical_dtype(data):
    # [FUNCTION DOC]
    # Verifies the interchange column for ordered and unordered categorical data and a round trip.
    # Parametrized by (kind label, expected ordered flag). The column has a repeated letter pattern producing five categories. Expected: dtype kind is categorical, null count zero, null description is a sentinel with value minus one, one chunk, categorical description reports the ordered flag as given, a dictionary flag true and a categories column wrapping a series of the sorted unique letters; converting back reproduces the original frame.
    data_categorical = {
        "ordered": pd.Categorical(list("testdata") * 30, ordered=True),
        "unordered": pd.Categorical(list("testdata") * 30, ordered=False),
    }
    df = pd.DataFrame({"A": (data_categorical[data[0]])})

    with tm.assert_produces_warning(match="Interchange"):
        col = df.__dataframe__().get_column_by_name("A")
    assert col.dtype[0] == DtypeKind.CATEGORICAL
    assert col.null_count == 0
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, -1)
    assert col.num_chunks() == 1
    desc_cat = col.describe_categorical
    assert desc_cat["is_ordered"] == data[1]
    assert desc_cat["is_dictionary"] is True
    assert isinstance(desc_cat["categories"], PandasColumn)
    tm.assert_series_equal(
        desc_cat["categories"]._col, pd.Series(["a", "d", "e", "s", "t"])
    )

    with tm.assert_produces_warning(match="Interchange"):
        tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_categorical_pyarrow():
    # [FUNCTION DOC]
    # Verifies that a pyarrow dictionary-encoded string column converts to a pandas categorical whose categories are in order of first appearance (issue 49889). Requires pyarrow 11 or newer.
    # GH 49889
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]
    table = pa.table({"weekday": pa.array(arr).dictionary_encode()})
    exchange_df = table.__dataframe__()
    with tm.assert_produces_warning(match="Interchange"):
        result = from_dataframe(exchange_df)
    weekday = pd.Categorical(
        arr, categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    expected = pd.DataFrame({"weekday": weekday})
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Constructing a Categorical with a dtype and values containing"
)
def test_empty_categorical_pyarrow():
    # [FUNCTION DOC]
    # Verifies converting a pyarrow dictionary-encoded column that contains only a null (float values) to a categorical with a missing value (issue 53077). Filters the warning about constructing a categorical with a dtype and values containing missing data. Requires pyarrow 11 or newer.
    # https://github.com/pandas-dev/pandas/issues/53077
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = [None]
    table = pa.table({"arr": pa.array(arr, "float64").dictionary_encode()})
    exchange_df = table.__dataframe__()
    with tm.assert_produces_warning(match="Interchange"):
        result = pd.api.interchange.from_dataframe(exchange_df)
    expected = pd.DataFrame({"arr": pd.Categorical([np.nan])})
    tm.assert_frame_equal(result, expected)


def test_large_string_pyarrow():
    # [FUNCTION DOC]
    # Verifies a pyarrow large string column converts to a string column (issue 52795) and that a round trip back through the pyarrow interchange reader gives an equal table; stack level checking is disabled in the round trip because pyarrow itself calls the deprecated method. Requires pyarrow 11.
    # GH 52795
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = ["Mon", "Tue"]
    table = pa.table({"weekday": pa.array(arr, "large_string")})
    exchange_df = table.__dataframe__()
    with tm.assert_produces_warning(match="Interchange"):
        result = from_dataframe(exchange_df)
    expected = pd.DataFrame({"weekday": ["Mon", "Tue"]})
    tm.assert_frame_equal(result, expected)

    # check round-trip
    # Don't check stacklevel as PyArrow calls the deprecated `__dataframe__` method.
    with tm.assert_produces_warning(match="Interchange", check_stacklevel=False):
        assert pa.Table.equals(pa.interchange.from_dataframe(result), table)


@pytest.mark.parametrize(
    ("offset", "length", "expected_values"),
    [
        (0, None, [3.3, float("nan"), 2.1]),
        (1, None, [float("nan"), 2.1]),
        (2, None, [2.1]),
        (0, 2, [3.3, float("nan")]),
        (0, 1, [3.3]),
        (1, 1, [float("nan")]),
    ],
)
def test_bitmasks_pyarrow(offset, length, expected_values):
    # [FUNCTION DOC]
    # Verifies validity bitmask handling for sliced pyarrow tables (issue 52795), including a round trip to pyarrow.
    # Parametrized by (offset, length, expected float values) covering slices from the start, from offsets one and two and bounded lengths over a float column with one missing value. Expected: the converted frame has the expected values (missing as nan) and converting back to pyarrow equals the sliced table. Requires pyarrow 11.
    # GH 52795
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = [3.3, None, 2.1]
    table = pa.table({"arr": arr}).slice(offset, length)
    exchange_df = table.__dataframe__()
    with tm.assert_produces_warning(match="Interchange"):
        result = from_dataframe(exchange_df)
    expected = pd.DataFrame({"arr": expected_values})
    tm.assert_frame_equal(result, expected)

    # check round-trip
    # Don't check stacklevel as PyArrow calls the deprecated `__dataframe__` method.
    with tm.assert_produces_warning(match="Interchange", check_stacklevel=False):
        assert pa.Table.equals(pa.interchange.from_dataframe(result), table)


@pytest.mark.parametrize(
    "data",
    [
        lambda: np.random.default_rng(2).integers(-100, 100),
        lambda: np.random.default_rng(2).integers(1, 100),
        lambda: np.random.default_rng(2).random(),
        lambda: np.random.default_rng(2).choice([True, False]),
        lambda: datetime(
            year=np.random.default_rng(2).integers(1900, 2100),
            month=np.random.default_rng(2).integers(1, 12),
            day=np.random.default_rng(2).integers(1, 20),
        ),
    ],
)
def test_dataframe(data):
    # [FUNCTION DOC]
    # Verifies frame-level interchange behavior for random single-type data.
    # Parametrized by a generator of cell values (random integer, random positive integer, random float, random boolean and random datetime). Builds a ten by twenty frame with column names generated from a rotated numbering. Expected: column count, row count and names match; selecting columns by position and by the corresponding names convert to equal frames; and the converted frames keep the interchange buffers list in their attributes.
    NCOLS, NROWS = 10, 20
    data = {
        f"col{int((i - NCOLS / 2) % NCOLS + 1)}": [data() for _ in range(NROWS)]
        for i in range(NCOLS)
    }
    df = pd.DataFrame(data)

    with tm.assert_produces_warning(match="Interchange"):
        df2 = df.__dataframe__()

    assert df2.num_columns() == NCOLS
    assert df2.num_rows() == NROWS

    assert list(df2.column_names()) == list(data.keys())

    indices = (0, 2)
    names = tuple(list(data.keys())[idx] for idx in indices)

    with tm.assert_produces_warning(match="Interchange"):
        result = from_dataframe(df2.select_columns(indices))
        expected = from_dataframe(df2.select_columns_by_name(names))
    tm.assert_frame_equal(result, expected)

    assert isinstance(result.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"], list)
    assert isinstance(expected.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"], list)


def test_missing_from_masked():
    # [FUNCTION DOC]
    # Verifies null counts for float columns that had random entries set to missing: the interchange column null count equals the number of values nulled for each of the three columns. The random generator is seeded.
    df = pd.DataFrame(
        {
            "x": np.array([1.0, 2.0, 3.0, 4.0, 0.0]),
            "y": np.array([1.5, 2.5, 3.5, 4.5, 0]),
            "z": np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
        }
    )

    rng = np.random.default_rng(2)
    dict_null = {col: rng.integers(low=0, high=len(df)) for col in df.columns}
    for col, num_nulls in dict_null.items():
        null_idx = df.index[
            rng.choice(np.arange(len(df)), size=num_nulls, replace=False)
        ]
        df.loc[null_idx, col] = None

    with tm.assert_produces_warning(match="Interchange"):
        df2 = df.__dataframe__()

    assert df2.get_column_by_name("x").null_count == dict_null["x"]
    assert df2.get_column_by_name("y").null_count == dict_null["y"]
    assert df2.get_column_by_name("z").null_count == dict_null["z"]


@pytest.mark.parametrize(
    "data",
    [
        {"x": [1.5, 2.5, 3.5], "y": [9.2, 10.5, 11.8]},
        {"x": [1, 2, 0], "y": [9.2, 10.5, 11.8]},
        {
            "x": np.array([True, True, False]),
            "y": np.array([1, 2, 0]),
            "z": np.array([9.2, 10.5, 11.8]),
        },
    ],
)
def test_mixed_data(data):
    # [FUNCTION DOC]
    # Verifies that columns without missing values (floats, mixed int and float, bool, int, float arrays) report a null count of zero.
    df = pd.DataFrame(data)
    with tm.assert_produces_warning(match="Interchange"):
        df2 = df.__dataframe__()

    for col_name in df.columns:
        assert df2.get_column_by_name(col_name).null_count == 0


def test_mixed_missing():
    # [FUNCTION DOC]
    # Verifies that boolean-like, integer-like and float arrays containing none values (object or float dtype) report a null count of two in each column.
    df = pd.DataFrame(
        {
            "x": np.array([True, None, False, None, True]),
            "y": np.array([None, 2, None, 1, 2]),
            "z": np.array([9.2, 10.5, None, 11.8, None]),
        }
    )

    with tm.assert_produces_warning(match="Interchange"):
        df2 = df.__dataframe__()

    for col_name in df.columns:
        assert df2.get_column_by_name(col_name).null_count == 2


def test_string():
    # [FUNCTION DOC]
    # Verifies the string column description: size, null count of one (missing value), string dtype kind and a byte-mask null description with value zero. Repeats the checks on a slice that skips the first row (size reduces by one, the null is still counted).
    string_data = {
        "separator data": [
            "abC|DeF,Hik",
            "234,3245.67",
            "gSaf,qWer|Gre",
            "asd3,4sad|",
            np.nan,
        ]
    }
    test_str_data = string_data["separator data"] + [""]
    df = pd.DataFrame({"A": test_str_data})
    with tm.assert_produces_warning(match="Interchange"):
        col = df.__dataframe__().get_column_by_name("A")

    assert col.size() == 6
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.STRING
    assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)

    df_sliced = df[1:]
    with tm.assert_produces_warning(match="Interchange"):
        col = df_sliced.__dataframe__().get_column_by_name("A")
    assert col.size() == 5
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.STRING
    assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)


def test_nonstring_object():
    # [FUNCTION DOC]
    # Verifies that an object column with mixed non-string values raises a not-implemented error saying the type is not supported yet when its dtype is requested.
    df = pd.DataFrame({"A": ["a", 10, 1.0, ()]})
    with tm.assert_produces_warning(match="Interchange"):
        col = df.__dataframe__().get_column_by_name("A")
    with pytest.raises(NotImplementedError, match="not supported yet"):
        col.dtype


def test_datetime():
    # [FUNCTION DOC]
    # Verifies a datetime column with a missing value: size two, one null, datetime kind, a sentinel null description using the integer minimum used for not-a-time, and a lossless round trip.
    df = pd.DataFrame({"A": [pd.Timestamp("2022-01-01"), pd.NaT]})
    with tm.assert_produces_warning(match="Interchange"):
        col = df.__dataframe__().get_column_by_name("A")

    assert col.size() == 2
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.DATETIME
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, iNaT)

    with tm.assert_produces_warning(match="Interchange"):
        tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_categorical_to_numpy_dlpack():
    # [FUNCTION DOC]
    # Verifies the categorical codes buffer can be consumed through the dlpack protocol into a numpy array of small signed integers (issue 48393).
    # https://github.com/pandas-dev/pandas/issues/48393
    df = pd.DataFrame({"A": pd.Categorical(["a", "b", "a"])})
    with tm.assert_produces_warning(match="Interchange"):
        col = df.__dataframe__().get_column_by_name("A")
    result = np.from_dlpack(col.get_buffers()["data"][0])
    expected = np.array([0, 1, 0], dtype="int8")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("data", [{}, {"a": []}])
def test_empty_pyarrow(data):
    # [FUNCTION DOC]
    # Verifies converting an empty pyarrow-produced frame (with no columns, or with one empty column) into pandas (issue 53155). The column type is not compared. Requires pyarrow 14.
    # GH 53155
    pytest.importorskip("pyarrow", "14.0.0")
    from pyarrow.interchange import from_dataframe as pa_from_dataframe

    expected = pd.DataFrame(data)
    # Don't check stacklevel as PyArrow calls the deprecated `__dataframe__` method.
    with tm.assert_produces_warning(match="Interchange", check_stacklevel=False):
        arrow_df = pa_from_dataframe(expected)
    result = from_dataframe(arrow_df)
    tm.assert_frame_equal(result, expected, check_column_type=False)


def test_multi_chunk_pyarrow() -> None:
    # [FUNCTION DOC]
    # Verifies that converting a multi-chunk pyarrow table with copying disallowed raises a runtime error saying a zero copy conversion into a multi-column block is not possible. Requires pyarrow 14.
    pa = pytest.importorskip("pyarrow", "14.0.0")
    n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    names = ["n_legs"]
    table = pa.table([n_legs], names=names)
    with pytest.raises(
        RuntimeError,
        match="Cannot do zero copy conversion into multi-column DataFrame block",
    ):
        pd.api.interchange.from_dataframe(table, allow_copy=False)


def test_multi_chunk_column() -> None:
    # [FUNCTION DOC]
    # Verifies handling of a pandas Arrow-backed integer column with more than one chunk.
    # With copying disallowed the conversion raises a runtime error saying a multi-chunk array was found; with copying allowed it returns float64 values with missing as nan (the interchange default is numpy-backed); the original frame is unchanged and its column still has two chunks. Requires pyarrow 11.
    pytest.importorskip("pyarrow", "11.0.0")
    ser = pd.Series([1, 2, None], dtype="Int64[pyarrow]")
    df = pd.concat([ser, ser], ignore_index=True).to_frame("a")
    df_orig = df.copy()

    with tm.assert_produces_warning(match="Interchange"):
        with pytest.raises(
            RuntimeError,
            match="Found multi-chunk pyarrow array, but `allow_copy` is False",
        ):
            pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=False))
    with tm.assert_produces_warning(match="Interchange"):
        result = pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=True))
    # Interchange protocol defaults to creating numpy-backed columns, so currently this
    # is 'float64'.
    expected = pd.DataFrame({"a": [1.0, 2.0, None, 1.0, 2.0, None]}, dtype="float64")
    tm.assert_frame_equal(result, expected)

    # Check that the rechunking we did didn't modify the original DataFrame.
    tm.assert_frame_equal(df, df_orig)
    assert len(df["a"].array._pa_array.chunks) == 2
    assert len(df_orig["a"].array._pa_array.chunks) == 2


def test_timestamp_ns_pyarrow():
    # [FUNCTION DOC]
    # Verifies a nanosecond timestamp Arrow-backed column round trips through the interchange to an equal timestamp value (issue 56712). Requires pyarrow 11.
    # GH 56712
    pytest.importorskip("pyarrow", "11.0.0")
    timestamp_args = {
        "year": 2000,
        "month": 1,
        "day": 1,
        "hour": 1,
        "minute": 1,
        "second": 1,
    }
    df = pd.Series(
        [datetime(**timestamp_args)],
        dtype="timestamp[ns][pyarrow]",
        name="col0",
    ).to_frame()

    with tm.assert_produces_warning(match="Interchange"):
        dfi = df.__dataframe__()
        result = pd.api.interchange.from_dataframe(dfi)["col0"].item()

    expected = pd.Timestamp(**timestamp_args)
    assert result == expected


@pytest.mark.parametrize("tz", ["UTC", "US/Pacific"])
def test_datetimetzdtype(tz, unit):
    # [FUNCTION DOC]
    # Verifies a timezone-aware datetime column, with the unit fixture and the two time zones UTC and a Pacific zone, survives an interchange round trip unchanged (issue 54239).
    # GH 54239
    tz_data = (
        pd.date_range("2018-01-01", periods=5, freq="D").tz_localize(tz).as_unit(unit)
    )
    df = pd.DataFrame({"ts_tz": tz_data})
    with tm.assert_produces_warning(match="Interchange"):
        tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_interchange_from_non_pandas_tz_aware(request):
    # [FUNCTION DOC]
    # Verifies converting a timezone-aware pyarrow column (Kathmandu) with a null into a pandas frame with a datetime column of microsecond unit and that zone (issues 54239 and 54287). On Windows CI with an old pyarrow the test is marked as an expected failure because of a missing tz database.
    # GH 54239, 54287
    pa = pytest.importorskip("pyarrow", "11.0.0")
    import pyarrow.compute as pc

    if is_platform_windows() and is_ci_environment() and pa_version_under22p0:
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason=(
                "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                "on CI to path to the tzdata for pyarrow."
            ),
        )
        request.applymarker(mark)

    arr = pa.array([datetime(2020, 1, 1), None, datetime(2020, 1, 2)])
    arr = pc.assume_timezone(arr, "Asia/Kathmandu")
    table = pa.table({"arr": arr})
    exchange_df = table.__dataframe__()
    with tm.assert_produces_warning(match="Interchange"):
        result = from_dataframe(exchange_df)

    expected = pd.DataFrame(
        ["2020-01-01 00:00:00+05:45", "NaT", "2020-01-02 00:00:00+05:45"],
        columns=["arr"],
        dtype="datetime64[us, Asia/Kathmandu]",
    )
    tm.assert_frame_equal(expected, result)


def test_interchange_from_corrected_buffer_dtypes(monkeypatch) -> None:
    # [FUNCTION DOC]
    # Verifies that conversion uses the data buffer dtype reported by the column rather than the column dtype (issue 54781).
    # Replaces the data buffer dtype of a string column with unsigned 8-bit, wires the replaced column into the interchange object and patches the frame's interchange method (via the monkeypatch fixture) to return it, then checks the conversion works.
    # https://github.com/pandas-dev/pandas/issues/54781
    with tm.assert_produces_warning(match="Interchange"):
        df = pd.DataFrame({"a": ["foo", "bar"]}).__dataframe__()
        interchange = df.__dataframe__()
    column = interchange.get_column_by_name("a")
    buffers = column.get_buffers()
    buffers_data = buffers["data"]
    buffer_dtype = buffers_data[1]
    buffer_dtype = (
        DtypeKind.UINT,
        8,
        ArrowCTypes.UINT8,
        buffer_dtype[3],
    )
    buffers["data"] = (buffers_data[0], buffer_dtype)
    column.get_buffers = lambda: buffers
    interchange.get_column_by_name = lambda _: column
    monkeypatch.setattr(df, "__dataframe__", lambda allow_copy: interchange)
    with tm.assert_produces_warning(match="Interchange"):
        pd.api.interchange.from_dataframe(df)


def test_empty_string_column():
    # [FUNCTION DOC]
    # Verifies that an empty string column survives the interchange round trip unchanged (issue 56703).
    # https://github.com/pandas-dev/pandas/issues/56703
    df = pd.DataFrame({"a": []}, dtype=str)
    with tm.assert_produces_warning(match="Interchange"):
        df2 = df.__dataframe__()
        result = pd.api.interchange.from_dataframe(df2)
    tm.assert_frame_equal(df, result)


def test_large_string():
    # [FUNCTION DOC]
    # Verifies an Arrow-backed large string column converts to the default string dtype (issue 56702); stack level is not checked. Requires pyarrow.
    # GH#56702
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": ["x"]}, dtype="large_string[pyarrow]")
    # Don't check stacklevel as PyArrow calls the deprecated `__dataframe__` method.
    with tm.assert_produces_warning(match="Interchange", check_stacklevel=False):
        result = pd.api.interchange.from_dataframe(df.__dataframe__())
    expected = pd.DataFrame({"a": ["x"]}, dtype="str")
    tm.assert_frame_equal(result, expected)


def test_non_str_names():
    # [FUNCTION DOC]
    # Verifies that integer column names are exposed as strings by the interchange object (issue 56701).
    # https://github.com/pandas-dev/pandas/issues/56701
    df = pd.Series([1, 2, 3], name=0).to_frame()
    with tm.assert_produces_warning(match="Interchange"):
        names = df.__dataframe__().column_names()
    assert names == ["0"]


def test_non_str_names_w_duplicates():
    # [FUNCTION DOC]
    # Verifies that converting an interchange object whose string-converted column names collide (a text zero and an integer zero) raises a type error explaining a series was expected but a frame was received because of duplicate names (issue 56701).
    # https://github.com/pandas-dev/pandas/issues/56701
    df = pd.DataFrame({"0": [1, 2, 3], 0: [4, 5, 6]})
    with tm.assert_produces_warning(match="Interchange"):
        dfi = df.__dataframe__()
    with tm.assert_produces_warning(match="Interchange"):
        with pytest.raises(
            TypeError,
            match=(
                "Expected a Series, got a DataFrame. This likely happened because you "
                "called __dataframe__ on a DataFrame which, after converting column "
                r"names to string, resulted in duplicated names: Index\(\['0', '0'\], "
                r"dtype='(str|object)'\). Please rename these columns before using the "
                "interchange protocol."
            ),
        ):
            pd.api.interchange.from_dataframe(dfi, allow_copy=False)


@pytest.mark.parametrize(
    ("data", "dtype", "expected_dtype"),
    [
        ([1, 2, None], "Int64", "int64"),
        ([1, 2, None], "Int64[pyarrow]", "int64"),
        ([1, 2, None], "Int8", "int8"),
        ([1, 2, None], "Int8[pyarrow]", "int8"),
        (
            [1, 2, None],
            "UInt64",
            "uint64",
        ),
        (
            [1, 2, None],
            "UInt64[pyarrow]",
            "uint64",
        ),
        ([1.0, 2.25, None], "Float32", "float32"),
        ([1.0, 2.25, None], "Float32[pyarrow]", "float32"),
        ([True, False, None], "boolean", "bool"),
        ([True, False, None], "boolean[pyarrow]", "bool"),
        (["much ado", "about", None], pd.StringDtype(na_value=np.nan), "large_string"),
        (["much ado", "about", None], "string[pyarrow]", "large_string"),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), None],
            "timestamp[ns][pyarrow]",
            "timestamp[ns]",
        ),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), None],
            "timestamp[us][pyarrow]",
            "timestamp[us]",
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                None,
            ],
            "timestamp[us, Asia/Kathmandu][pyarrow]",
            "timestamp[us, tz=Asia/Kathmandu]",
        ),
    ],
)
def test_pandas_nullable_with_missing_values(
    data: list, dtype: str, expected_dtype: str
) -> None:
    # [FUNCTION DOC]
    # Verifies conversion of pandas nullable and Arrow-backed dtypes with a missing value to pyarrow (issues 57643 and 57664).
    # Parametrized by (data, pandas dtype, expected pyarrow type name): integer, unsigned, float, boolean, string and timestamp dtypes in numpy-backed nullable and pyarrow-backed forms. Expected: the pyarrow column has the expected type, the first two values equal the input and the third is none. The time zone case builds the expected pyarrow type explicitly. Requires pyarrow 14.
    # https://github.com/pandas-dev/pandas/issues/57643
    # https://github.com/pandas-dev/pandas/issues/57664
    pa = pytest.importorskip("pyarrow", "14.0.0")
    import pyarrow.interchange as pai

    if expected_dtype == "timestamp[us, tz=Asia/Kathmandu]":
        expected_dtype = pa.timestamp("us", "Asia/Kathmandu")

    df = pd.DataFrame({"a": data}, dtype=dtype)
    with tm.assert_produces_warning(match="Interchange"):
        result = pai.from_dataframe(df.__dataframe__())["a"]
    assert result.type == expected_dtype
    assert result[0].as_py() == data[0]
    assert result[1].as_py() == data[1]
    assert result[2].as_py() is None


@pytest.mark.parametrize(
    ("data", "dtype", "expected_dtype"),
    [
        ([1, 2, 3], "Int64", "int64"),
        ([1, 2, 3], "Int64[pyarrow]", "int64"),
        ([1, 2, 3], "Int8", "int8"),
        ([1, 2, 3], "Int8[pyarrow]", "int8"),
        (
            [1, 2, 3],
            "UInt64",
            "uint64",
        ),
        (
            [1, 2, 3],
            "UInt64[pyarrow]",
            "uint64",
        ),
        ([1.0, 2.25, 5.0], "Float32", "float32"),
        ([1.0, 2.25, 5.0], "Float32[pyarrow]", "float32"),
        ([True, False, False], "boolean", "bool"),
        ([True, False, False], "boolean[pyarrow]", "bool"),
        (
            ["much ado", "about", "nothing"],
            pd.StringDtype(na_value=np.nan),
            "large_string",
        ),
        (["much ado", "about", "nothing"], "string[pyarrow]", "large_string"),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "timestamp[ns][pyarrow]",
            "timestamp[ns]",
        ),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "timestamp[us][pyarrow]",
            "timestamp[us]",
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                datetime(2020, 1, 3, tzinfo=timezone.utc),
            ],
            "timestamp[us, Asia/Kathmandu][pyarrow]",
            "timestamp[us, tz=Asia/Kathmandu]",
        ),
    ],
)
def test_pandas_nullable_without_missing_values(
    data: list, dtype: str, expected_dtype: str
) -> None:
    # [FUNCTION DOC]
    # Same as the previous test but without missing values; all three values are compared with the input (issue 57643). Requires pyarrow 14.
    # https://github.com/pandas-dev/pandas/issues/57643
    pa = pytest.importorskip("pyarrow", "14.0.0")
    import pyarrow.interchange as pai

    if expected_dtype == "timestamp[us, tz=Asia/Kathmandu]":
        expected_dtype = pa.timestamp("us", "Asia/Kathmandu")

    df = pd.DataFrame({"a": data}, dtype=dtype)
    with tm.assert_produces_warning(match="Interchange"):
        result = pai.from_dataframe(df.__dataframe__())["a"]
    assert result.type == expected_dtype
    assert result[0].as_py() == data[0]
    assert result[1].as_py() == data[1]
    assert result[2].as_py() == data[2]


def test_string_validity_buffer() -> None:
    # [FUNCTION DOC]
    # Verifies that the validity buffer of an Arrow-backed large string column with no missing values is none (issue 57761). Requires pyarrow 11.
    # https://github.com/pandas-dev/pandas/issues/57761
    pytest.importorskip("pyarrow", "11.0.0")
    df = pd.DataFrame({"a": ["x"]}, dtype="large_string[pyarrow]")
    with tm.assert_produces_warning(match="Interchange"):
        result = df.__dataframe__().get_column_by_name("a").get_buffers()["validity"]
    assert result is None


def test_string_validity_buffer_no_missing() -> None:
    # [FUNCTION DOC]
    # Verifies that the validity buffer of an Arrow-backed large string column that does contain a missing value is present and describes a one-bit boolean (issue 57762). Requires pyarrow 11.
    # KNOWN ISSUE: the test name says no missing values but the data contains a missing value; its name and the name of the previous test appear swapped relative to what they check.
    # https://github.com/pandas-dev/pandas/issues/57762
    pytest.importorskip("pyarrow", "11.0.0")
    df = pd.DataFrame({"a": ["x", None]}, dtype="large_string[pyarrow]")
    with tm.assert_produces_warning(match="Interchange"):
        validity = df.__dataframe__().get_column_by_name("a").get_buffers()["validity"]
    assert validity is not None
    result = validity[1]
    expected = (DtypeKind.BOOL, 1, ArrowCTypes.BOOL, "=")
    assert result == expected


def test_empty_dataframe():
    # [FUNCTION DOC]
    # Verifies that an empty frame with an 8-bit integer column round trips with copying disallowed and keeps its dtype (issue 56700).
    # https://github.com/pandas-dev/pandas/issues/56700
    df = pd.DataFrame({"a": []}, dtype="int8")
    with tm.assert_produces_warning(match="Interchange"):
        dfi = df.__dataframe__()
        result = pd.api.interchange.from_dataframe(dfi, allow_copy=False)
    expected = pd.DataFrame({"a": []}, dtype="int8")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "expected_dtype", "expected_buffer_dtype"),
    [
        (
            pd.Series(["a", "b", "a"], dtype="category"),
            (DtypeKind.CATEGORICAL, 8, "c", "="),
            (DtypeKind.INT, 8, "c", "|"),
        ),
        (
            pd.Series(
                [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
                dtype="M8[ns]",
            ),
            (DtypeKind.DATETIME, 64, "tsn:", "="),
            (DtypeKind.INT, 64, ArrowCTypes.INT64, "="),
        ),
        (
            pd.Series(["a", "bc", None]),
            (DtypeKind.STRING, 8, ArrowCTypes.STRING, "="),
            (DtypeKind.UINT, 8, ArrowCTypes.UINT8, "="),
        ),
        (
            pd.Series([1, 2, 3]),
            (DtypeKind.INT, 64, ArrowCTypes.INT64, "="),
            (DtypeKind.INT, 64, ArrowCTypes.INT64, "="),
        ),
        (
            pd.Series([1.5, 2, 3]),
            (DtypeKind.FLOAT, 64, ArrowCTypes.FLOAT64, "="),
            (DtypeKind.FLOAT, 64, ArrowCTypes.FLOAT64, "="),
        ),
    ],
)
def test_buffer_dtype_categorical(
    data: pd.Series,
    expected_dtype: tuple[DtypeKind, int, str, str],
    expected_buffer_dtype: tuple[DtypeKind, int, str, str],
) -> None:
    # [FUNCTION DOC]
    # Verifies the column dtype and the data buffer dtype for several kinds of series.
    # Parametrized by (series, expected column dtype tuple, expected buffer dtype tuple), each tuple being kind, bit width, Arrow format string and byte order: categorical, datetime, string with a missing value, integer and float (issue 54781). Despite the name, not only categorical data is covered.
    # https://github.com/pandas-dev/pandas/issues/54781
    df = pd.DataFrame({"data": data})
    with tm.assert_produces_warning(match="Interchange"):
        dfi = df.__dataframe__()
    col = dfi.get_column_by_name("data")
    assert col.dtype == expected_dtype
    assert col.get_buffers()["data"][1] == expected_buffer_dtype


def test_from_dataframe_list_dtype():
    # [FUNCTION DOC]
    # Verifies converting a pyarrow table with a list-typed column to a frame preserves the lists. Requires pyarrow 14. Does not assert the interchange warning.
    pa = pytest.importorskip("pyarrow", "14.0.0")
    data = {"a": [[1, 2], [4, 5, 6]]}
    tbl = pa.table(data)
    result = from_dataframe(tbl)
    expected = pd.DataFrame(data)
    tm.assert_frame_equal(result, expected)
