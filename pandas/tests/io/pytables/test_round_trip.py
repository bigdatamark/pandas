# [MODULE DOC]
# Round-trip tests for the HDF5 store: write an object (series or frame) and read it back, checking equality, across API spellings of format/append, index types, dtypes, empty objects, hierarchical data, unicode and compression. Module marked single_cpu.
# Contains two private helpers (generic round trip through item assignment, and a table-format variant) used by many tests; both write to the supplied path in write mode.
import datetime
import re

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    HDFStore,
    Index,
    Series,
    _testing as tm,
    bdate_range,
    date_range,
    read_hdf,
)
from pandas.util import _test_decorators as td

pytestmark = [pytest.mark.single_cpu]


def test_conv_read_write(temp_h5_path):
    # [FUNCTION DOC]
    # Uses a nested helper that writes with to_hdf under a key and reads with read_hdf. Series with datetime and string indexes and a frame round-trip equal. A frame written with append through to_hdf then read with a where condition on the index returns just the matching rows.
    def roundtrip(key, obj, **kwargs):
        # [FUNCTION DOC]
        # HELPER: writes obj to the enclosing test's temp path under key (extra keyword arguments passed to the writer) and returns the object read back from that key.
        obj.to_hdf(temp_h5_path, key=key, **kwargs)
        return read_hdf(temp_h5_path, key)

    o = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    tm.assert_series_equal(o, roundtrip("series", o))

    o = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])
    tm.assert_series_equal(o, roundtrip("string_series", o))

    o = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    tm.assert_frame_equal(o, roundtrip("frame", o))

    # table
    df = DataFrame({"A": range(5), "B": range(5)})
    df.to_hdf(temp_h5_path, key="table", append=True)
    result = read_hdf(temp_h5_path, "table", where=["index>2"])
    tm.assert_frame_equal(df[df.index > 2], result)


def test_long_strings(temp_hdfstore):
    # [FUNCTION DOC]
    # GH6166. Fifty-character strings used as data column values and as index round-trip through a table.
    # GH6166
    data = ["a" * 50] * 10
    df = DataFrame({"a": data}, index=data)

    temp_hdfstore.append("df", df, data_columns=["a"])

    result = temp_hdfstore.select("df")
    tm.assert_frame_equal(df, result)


def test_api(temp_h5_path):
    # [FUNCTION DOC]
    # GH4584. to_hdf must accept both append and format arguments: two appended halves in table format reproduce the full frame; with append false for the first half and true for the second also reproduces it.
    # GH4584
    # API issue when to_hdf doesn't accept append AND format args
    path = temp_h5_path

    df = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key="df", append=True, format="table")
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    # append to False
    df.iloc[:10].to_hdf(path, key="df", append=False, format="table")
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)


def test_api_append(temp_h5_path):
    # [FUNCTION DOC]
    # to_hdf appends where the first call omits the format (table inferred from the append flag) and the second specifies table format, and the reverse; both reconstruct the full frame.
    path = temp_h5_path

    df = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key="df", append=True)
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    # append to False
    df.iloc[:10].to_hdf(path, key="df", append=False, format="table")
    df.iloc[10:].to_hdf(path, key="df", append=True)
    tm.assert_frame_equal(read_hdf(path, "df"), df)


def test_api_2(temp_h5_path):
    # [FUNCTION DOC]
    # to_hdf with append false and format fixed, short format code f, default format, and no arguments all write a frame that reads back equal.
    df = DataFrame(range(20))
    df.to_hdf(temp_h5_path, key="df", append=False, format="fixed")
    tm.assert_frame_equal(read_hdf(temp_h5_path, "df"), df)

    df.to_hdf(temp_h5_path, key="df", append=False, format="f")
    tm.assert_frame_equal(read_hdf(temp_h5_path, "df"), df)

    df.to_hdf(temp_h5_path, key="df", append=False)
    tm.assert_frame_equal(read_hdf(temp_h5_path, "df"), df)

    df.to_hdf(temp_h5_path, key="df")
    tm.assert_frame_equal(read_hdf(temp_h5_path, "df"), df)


def test_api_3(temp_hdfstore):
    # [FUNCTION DOC]
    # Store-level append with append flag and table format, with append false for the first half then true, and with the format omitted (none) on the second append; each variant reproduces the full frame (the removal between variants resets state).
    df = DataFrame(range(20))

    temp_hdfstore.append("df", df.iloc[:10], append=True, format="table")
    temp_hdfstore.append("df", df.iloc[10:], append=True, format="table")
    tm.assert_frame_equal(temp_hdfstore.select("df"), df)

    # append to False
    temp_hdfstore.remove("df")
    temp_hdfstore.append("df", df.iloc[:10], append=False, format="table")
    temp_hdfstore.append("df", df.iloc[10:], append=True, format="table")
    tm.assert_frame_equal(temp_hdfstore.select("df"), df)

    # formats
    temp_hdfstore.remove("df")
    temp_hdfstore.append("df", df.iloc[:10], append=False, format="table")
    temp_hdfstore.append("df", df.iloc[10:], append=True, format="table")
    tm.assert_frame_equal(temp_hdfstore.select("df"), df)

    temp_hdfstore.remove("df")
    temp_hdfstore.append("df", df.iloc[:10], append=False, format="table")
    temp_hdfstore.append("df", df.iloc[10:], append=True, format=None)
    tm.assert_frame_equal(temp_hdfstore.select("df"), df)


def test_api_invalid(temp_h5_path):
    # [FUNCTION DOC]
    # Appending in fixed format (long or short name) raises a value error that only tables can be appended to; an unknown format raises a type error saying the format is invalid, for append true or false. Reading from an empty path raises a file-not-found error.
    path = temp_h5_path
    # Invalid.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    msg = "Can only append to Tables"

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", append=True, format="f")

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", append=True, format="fixed")

    msg = r"invalid HDFStore format specified \[foo\]"

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key="df", append=True, format="foo")

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key="df", append=False, format="foo")

    # File path doesn't exist
    path = ""
    msg = f"File {path} does not exist"

    with pytest.raises(FileNotFoundError, match=msg):
        read_hdf(path, "df")


def test_get(temp_hdfstore):
    # [FUNCTION DOC]
    # The get method and item access return equal series for plain and slash-prefixed keys; getting a missing key raises a key error saying no object with that name.
    temp_hdfstore["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    left = temp_hdfstore.get("a")
    right = temp_hdfstore["a"]
    tm.assert_series_equal(left, right)

    left = temp_hdfstore.get("/a")
    right = temp_hdfstore["/a"]
    tm.assert_series_equal(left, right)

    with pytest.raises(KeyError, match="'No object named b in the file'"):
        temp_hdfstore.get("b")


def test_put_integer(temp_h5_path):
    # [FUNCTION DOC]
    # Round-trips a 50 by 100 frame with default integer labels using the shared helper.
    # non-date, non-string index
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    _check_roundtrip(df, tm.assert_frame_equal, temp_h5_path)


def test_table_values_dtypes_roundtrip(temp_hdfstore, using_infer_string):
    # [FUNCTION DOC]
    # Appending float64, int64 and float32 frames preserves dtypes. Appending a float frame to the existing integer table raises a value error stating the contents are not float but integer. A frame with many numeric widths, string, boolean and second/millisecond/nanosecond timestamp columns is appended and the counts of resulting dtypes (string as str under infer-string mode, else object) are asserted.
    df1 = DataFrame({"a": [1, 2, 3]}, dtype="f8")
    temp_hdfstore.append("df_f8", df1)
    tm.assert_series_equal(df1.dtypes, temp_hdfstore["df_f8"].dtypes)

    df2 = DataFrame({"a": [1, 2, 3]}, dtype="i8")
    temp_hdfstore.append("df_i8", df2)
    tm.assert_series_equal(df2.dtypes, temp_hdfstore["df_i8"].dtypes)

    # incompatible dtype
    msg = re.escape(
        "Cannot serialize the column [a] "
        "because its data contents are not [float] "
        "but [integer] object dtype"
    )
    with pytest.raises(ValueError, match=msg):
        temp_hdfstore.append("df_i8", df1)

    # check creation/storage/retrieval of float32 (a bit hacky to
    # actually create them thought)
    df1 = DataFrame(np.array([[1], [2], [3]], dtype="f4"), columns=["A"])
    temp_hdfstore.append("df_f4", df1)
    tm.assert_series_equal(df1.dtypes, temp_hdfstore["df_f4"].dtypes)
    assert df1.dtypes.iloc[0] == "float32"

    # check with mixed dtypes
    df1 = DataFrame(
        {
            c: Series(np.random.default_rng(2).integers(5), dtype=c)
            for c in ["float32", "float64", "int32", "int64", "int16", "int8"]
        }
    )
    df1["string"] = "foo"
    df1["float322"] = 1.0
    df1["float322"] = df1["float322"].astype("float32")
    df1["bool"] = df1["float32"] > 0
    df1["time_s_1"] = Timestamp("20130101").as_unit("s")
    df1["time_s_2"] = Timestamp("20130101 00:00:00").as_unit("s")
    df1["time_ms"] = Timestamp("20130101 00:00:00.000").as_unit("ms")
    df1["time_ns"] = Timestamp("20130102 00:00:00.000000000")

    temp_hdfstore.append("df_mixed_dtypes1", df1)
    result = temp_hdfstore.select("df_mixed_dtypes1").dtypes.value_counts()
    result.index = [str(i) for i in result.index]
    str_dtype = "str" if using_infer_string else "object"
    expected = Series(
        {
            "float32": 2,
            "float64": 1,
            "int32": 1,
            "bool": 1,
            "int16": 1,
            "int8": 1,
            "int64": 1,
            str_dtype: 1,
            "datetime64[s]": 2,
            "datetime64[ms]": 1,
            "datetime64[ns]": 1,
        },
        name="count",
    )
    result = result.sort_index()
    expected = expected.sort_index()
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_series(temp_h5_path):
    # [FUNCTION DOC]
    # Series with string, datetime, and datetime-valued index (two ways) round-trip using the helper; the last with index type checking disabled. Performance warnings ignored.
    s = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])
    _check_roundtrip(s, tm.assert_series_equal, path=temp_h5_path)

    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    _check_roundtrip(ts, tm.assert_series_equal, path=temp_h5_path)

    ts2 = Series(ts.index, Index(ts.index))
    _check_roundtrip(ts2, tm.assert_series_equal, path=temp_h5_path)

    ts3 = Series(ts.values, Index(np.asarray(ts.index)))
    _check_roundtrip(
        ts3, tm.assert_series_equal, path=temp_h5_path, check_index_type=False
    )


def test_float_index(temp_h5_path):
    # [FUNCTION DOC]
    # GH #454. A series with a random float index round-trips.
    # GH #454
    index = np.random.default_rng(2).standard_normal(10)
    s = Series(np.random.default_rng(2).standard_normal(10), index=index)
    _check_roundtrip(s, tm.assert_series_equal, path=temp_h5_path)


def test_tuple_index(temp_h5_path, performance_warning):
    # [FUNCTION DOC]
    # GH #492. A frame indexed by tuples round-trips while emitting the performance warning (expected-warning fixture).
    # GH #492
    col = np.arange(10)
    idx = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    data = np.random.default_rng(2).standard_normal(30).reshape((3, 10))
    DF = DataFrame(data, index=idx, columns=col)

    with tm.assert_produces_warning(performance_warning):
        _check_roundtrip(DF, tm.assert_frame_equal, path=temp_h5_path)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_index_types(temp_h5_path):
    # [FUNCTION DOC]
    # Round-trips series with a wide range of mixed-type indexes (integer and string, today's datetime and date with other types, float and string, mixed number types, integers, nanosecond datetimes and the same converted to second unit) using a comparator that checks index type. Performance warnings ignored.
    # KNOWN ISSUE: four of the mixed-index cases (integer and string, datetime and integer, string and integer, date and string) appear twice in succession, which is redundant.
    values = np.random.default_rng(2).standard_normal(2)

    func = lambda lhs, rhs: tm.assert_series_equal(lhs, rhs, check_index_type=True)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [1.23, "b"])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [1, 1.53])
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser = Series(values, [1, 5])
    _check_roundtrip(ser, func, path=temp_h5_path)

    dti = DatetimeIndex(["2012-01-01", "2012-01-02"], dtype="M8[ns]")
    ser = Series(values, index=dti)
    _check_roundtrip(ser, func, path=temp_h5_path)

    ser.index = ser.index.as_unit("s")
    _check_roundtrip(ser, func, path=temp_h5_path)


def test_timeseries_preepoch(temp_h5_path, request):
    # [FUNCTION DOC]
    # Round-trips a business-day series from 1940 to 1960. An overflow error on Windows platforms is converted into an expected failure via the request fixture and then re-raised.
    # KNOWN ISSUE: the expected-failure marker is created with the message string as its first (positional) argument, which pytest treats as the condition rather than the reason.
    dr = bdate_range("1/1/1940", "1/1/1960")
    ts = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    try:
        _check_roundtrip(ts, tm.assert_series_equal, path=temp_h5_path)
    except OverflowError:
        if is_platform_windows():
            request.applymarker(
                pytest.mark.xfail("known failure on some windows platforms")
            )
        raise


@pytest.mark.parametrize(
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_frame(compression, temp_h5_path):
    # [FUNCTION DOC]
    # Parametrized over compression off and on (on skipped for Windows). Frame with random missing values is round-tripped in table format and fixed format; a datetime-indexed frame round-trips; a frame with a newly added column stored with item assignment comes back consolidated; and an empty slice of the frame (with an empty index) round-trips.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    # put in some random NAs
    df.iloc[0, 0] = np.nan
    df.iloc[5, 3] = np.nan

    _check_roundtrip_table(
        df, tm.assert_frame_equal, path=temp_h5_path, compression=compression
    )
    _check_roundtrip(
        df, tm.assert_frame_equal, path=temp_h5_path, compression=compression
    )

    tdf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    _check_roundtrip(
        tdf, tm.assert_frame_equal, path=temp_h5_path, compression=compression
    )

    with HDFStore(temp_h5_path) as store:
        # not consolidated
        df["foo"] = np.random.default_rng(2).standard_normal(len(df))
        store["df"] = df
        recons = store["df"]
        assert recons._mgr.is_consolidated()

    # empty
    df2 = df[:0]
    # Prevent df2 from having index with inferred_type as string
    df2.index = Index([])
    _check_roundtrip(df2[:0], tm.assert_frame_equal, path=temp_h5_path)


def test_empty_series_frame(temp_h5_path):
    # [FUNCTION DOC]
    # Empty series (unnamed and named), empty frame, frame with only index and frame with only columns each round-trip.
    s0 = Series(dtype=object)
    s1 = Series(name="myseries", dtype=object)
    df0 = DataFrame()
    df1 = DataFrame(index=["a", "b", "c"])
    df2 = DataFrame(columns=["d", "e", "f"])

    _check_roundtrip(s0, tm.assert_series_equal, path=temp_h5_path)
    _check_roundtrip(s1, tm.assert_series_equal, path=temp_h5_path)
    _check_roundtrip(df0, tm.assert_frame_equal, path=temp_h5_path)
    _check_roundtrip(df1, tm.assert_frame_equal, path=temp_h5_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=temp_h5_path)


@pytest.mark.parametrize("dtype", [np.int64, np.float64, object, "m8[ns]", "M8[ns]"])
def test_empty_series(dtype, temp_h5_path):
    # [FUNCTION DOC]
    # Parametrized over int64, float64, object, timedelta and datetime dtypes: an empty series of the dtype round-trips.
    s = Series(dtype=dtype)
    _check_roundtrip(s, tm.assert_series_equal, path=temp_h5_path)


def test_can_serialize_dates(temp_h5_path):
    # [FUNCTION DOC]
    # A frame indexed by plain date objects (business days of one month) round-trips.
    rng = [x.date() for x in bdate_range("1/1/2000", "1/30/2000")]
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    _check_roundtrip(frame, tm.assert_frame_equal, path=temp_h5_path)


def test_store_hierarchical(
    temp_h5_path, using_infer_string, multiindex_dataframe_random_data
):
    # [FUNCTION DOC]
    # Uses the multi-index frame fixture. The frame, its transpose and a column series round-trip; storing the frame with item assignment returns an equal frame (names stored). The infer-string fixture is unused.
    frame = multiindex_dataframe_random_data

    _check_roundtrip(frame, tm.assert_frame_equal, path=temp_h5_path)
    _check_roundtrip(frame.T, tm.assert_frame_equal, path=temp_h5_path)
    _check_roundtrip(frame["A"], tm.assert_series_equal, path=temp_h5_path)

    # check that the names are stored
    with HDFStore(temp_h5_path) as store:
        store["frame"] = frame
        recons = store["frame"]
        tm.assert_frame_equal(recons, frame)


@pytest.mark.parametrize(
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_store_mixed(compression, temp_h5_path):
    # [FUNCTION DOC]
    # Parametrized over compression off and on (on skipped for Windows). Two consolidated frames with object, boolean and integer columns round-trip; overwriting the same key in a store returns the latest frame; and a string series, boolean series and integer series taken from the first frame round-trip with the compression setting applied.
    def _make_one():
        # [FUNCTION DOC]
        # HELPER: builds a 30 by 4 float frame (columns A to D, string row labels) plus two string columns, two boolean columns and two integer columns, and returns it consolidated.
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        df["obj1"] = "foo"
        df["obj2"] = "bar"
        df["bool1"] = df["A"] > 0
        df["bool2"] = df["B"] > 0
        df["int1"] = 1
        df["int2"] = 2
        return df._consolidate()

    df1 = _make_one()
    df2 = _make_one()

    _check_roundtrip(df1, tm.assert_frame_equal, path=temp_h5_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=temp_h5_path)

    with HDFStore(temp_h5_path) as store:
        store["obj"] = df1
        tm.assert_frame_equal(store["obj"], df1)
        store["obj"] = df2
        tm.assert_frame_equal(store["obj"], df2)

    # check that can store Series of all of these types
    _check_roundtrip(
        df1["obj1"],
        tm.assert_series_equal,
        path=temp_h5_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["bool1"],
        tm.assert_series_equal,
        path=temp_h5_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["int1"],
        tm.assert_series_equal,
        path=temp_h5_path,
        compression=compression,
    )


def _check_roundtrip(obj, comparator, path, compression=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: parameters obj (object to store), comparator (assertion function taking retrieved and original), path, compression (default off; when on uses the blosc library) and extra keyword arguments passed to the comparator. Opens a store at path in write mode (truncating it), stores obj via item assignment under a fixed key, reads it back and invokes the comparator. Returns nothing.
    options = {}
    if compression:
        options["complib"] = "blosc"

    with HDFStore(path, "w", **options) as store:
        store["obj"] = obj
        retrieved = store["obj"]
        comparator(retrieved, obj, **kwargs)


def _check_roundtrip_table(obj, comparator, path, compression=False):
    # [FUNCTION DOC]
    # HELPER: like the generic round-trip helper but stores with put in table format and does not forward extra comparator keywords. Parameters: obj, comparator, path, compression (default off; blosc when on). Opens the store in write mode, stores, reads back and calls the comparator.
    options = {}
    if compression:
        options["complib"] = "blosc"

    with HDFStore(path, "w", **options) as store:
        store.put("obj", obj, format="table")
        retrieved = store["obj"]

        comparator(retrieved, obj)


def test_unicode_index(temp_h5_path):
    # [FUNCTION DOC]
    # A series indexed by non-ascii characters round-trips.
    unicode_values = ["\u03c3", "\u03c3\u03c3"]

    s = Series(
        np.random.default_rng(2).standard_normal(len(unicode_values)),
        unicode_values,
    )
    _check_roundtrip(s, tm.assert_series_equal, path=temp_h5_path)


def test_unicode_longer_encoded(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 11234. Strings whose utf-8 encoding is longer than their character count (a single Greek character, and mixed with ascii strings) are put in table format with utf-8 encoding and read back equal.
    # GH 11234
    char = "\u0394"
    df = DataFrame({"A": [char]})
    temp_hdfstore.put("df", df, format="table", encoding="utf-8")
    result = temp_hdfstore.get("df")
    tm.assert_frame_equal(result, df)

    df = DataFrame({"A": ["a", char], "B": ["b", "b"]})
    temp_hdfstore.remove("df")
    temp_hdfstore.put("df", df, format="table", encoding="utf-8")
    result = temp_hdfstore.get("df")
    tm.assert_frame_equal(result, df)


def test_store_datetime_mixed(temp_h5_path):
    # [FUNCTION DOC]
    # A frame with integer, float, string and datetime columns round-trips.
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    df["d"] = ts.index[:3]
    _check_roundtrip(df, tm.assert_frame_equal, path=temp_h5_path)


def test_round_trip_equals(temp_h5_path):
    # [FUNCTION DOC]
    # GH 9330. A table-format round trip returns a frame that is equal to the original under the frame comparator and in both directions under the equality method.
    # GH 9330
    df = DataFrame({"B": [1, 2], "A": ["x", "y"]})

    df.to_hdf(temp_h5_path, key="df", format="table")
    other = read_hdf(temp_h5_path, "df")
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)


def test_infer_string_columns(temp_h5_path):
    # [FUNCTION DOC]
    # Requires pyarrow. Under the future infer-string option, a frame with a two-level index created from string columns round-trips in table format.
    # KNOWN ISSUE: the issue-number comment above the body is truncated (only the GH prefix, no number).
    # GH#
    pytest.importorskip("pyarrow")
    with pd.option_context("future.infer_string", True):
        df = DataFrame(1, columns=list("ABCD"), index=list(range(10))).set_index(
            ["A", "B"]
        )
        expected = df.copy()
        df.to_hdf(temp_h5_path, key="df", format="table")

        result = read_hdf(temp_h5_path, "df")
        tm.assert_frame_equal(result, expected)
