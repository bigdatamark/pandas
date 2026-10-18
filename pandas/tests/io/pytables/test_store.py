# [MODULE DOC]
# General tests of the HDFStore class: context-manager use, creation-time tracking, repr and info, containment, version attributes, group walking, attribute access, dropna and min_itemsize in to_hdf, error handling for encodings, table index creation, various selection by coordinates and start/stop, pathlib support, copying, column name restrictions, and several bug regressions.
# Skipped at import if tables is missing; module marked single_cpu. Uses temp_hdfstore, temp_h5_path and temp_file fixtures.
import contextlib
import datetime as dt
import hashlib
import tempfile
import time

import numpy as np
import pytest

from pandas.compat import PY312

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.api.types import (
    CategoricalDtype,
)
from pandas.util.version import Version

from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)

pytestmark = [pytest.mark.single_cpu]

tables = pytest.importorskip("tables")


def test_context(temp_h5_path):
    # [FUNCTION DOC]
    # Raising an exception inside a store context manager still closes the store cleanly; reopening and storing a frame works, the store has length one and item access returns a frame instance.
    try:
        with HDFStore(temp_h5_path) as tbl:
            raise ValueError("blah")
    except ValueError:
        pass
    with HDFStore(temp_h5_path) as tbl:
        tbl["a"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        assert len(tbl) == 1
        assert type(tbl["a"]) == DataFrame


@pytest.mark.xfail(
    Version(tables.hdf5_version) >= Version("2"),
    reason="track_times=False produces non-deterministic files with HDF5 >= 2",
)
def test_no_track_times(temp_h5_path):
    # [FUNCTION DOC]
    # GH 32682. Expected to fail with HDF5 version two or newer (files are non-deterministic). Creates tables with the track_times option off and on, twice with a one second sleep between so creation timestamps differ; asserts file checksums are equal when tracking is off and differ when on.
    # GH 32682
    # enables to set track_times (see `pytables` `create_table` documentation)

    def checksum(filename, hash_factory=hashlib.md5, chunk_num_blocks=128):
        # [FUNCTION DOC]
        # HELPER: returns the digest of a file's contents, hashing it in blocks of a given number of hash block sizes. Parameters: filename, hash_factory (default md5), chunk_num_blocks (default 128).
        h = hash_factory()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
                h.update(chunk)
        return h.digest()

    def create_h5_and_return_checksum(temp_h5_path, track_times):
        # [FUNCTION DOC]
        # HELPER: writes a one-row table as a data-column table with the given track_times setting into a new file (write mode) at the path, closes it, and returns the file checksum.
        df = DataFrame({"a": [1]})

        with HDFStore(temp_h5_path, mode="w") as hdf:
            hdf.put(
                "table",
                df,
                format="table",
                data_columns=True,
                index=None,
                track_times=track_times,
            )

        return checksum(temp_h5_path)

    checksum_0_tt_false = create_h5_and_return_checksum(temp_h5_path, track_times=False)
    checksum_0_tt_true = create_h5_and_return_checksum(temp_h5_path, track_times=True)

    # sleep is necessary to create h5 with different creation time
    time.sleep(1)

    checksum_1_tt_false = create_h5_and_return_checksum(temp_h5_path, track_times=False)
    checksum_1_tt_true = create_h5_and_return_checksum(temp_h5_path, track_times=True)

    # checksums are the same if track_time = False
    assert checksum_0_tt_false == checksum_1_tt_false

    # checksums are NOT same if track_time = True
    assert checksum_0_tt_true != checksum_1_tt_true


def test_iter_empty(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 12221. Iterating over an empty store yields nothing.
    # GH 12221
    assert list(temp_hdfstore) == []


def test_repr(temp_hdfstore, performance_warning, using_infer_string):
    # [FUNCTION DOC]
    # Calls repr and info on an empty store, then on a store containing series, a frame and a mixed-type frame (writing that one emits a performance warning about mapping to C types unless infer-string is on), plus a non-pandas group; asserts the filename is in both repr and str, and info still works.
    store = temp_hdfstore
    repr(store)
    store.info()
    store["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    store["b"] = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])
    store["c"] = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["bool1"] = df["A"] > 0
    df["bool2"] = df["B"] > 0
    df["bool3"] = True
    df["int1"] = 1
    df["int2"] = 2
    df["timestamp1"] = Timestamp("20010102")
    df["timestamp2"] = Timestamp("20010103")
    df["datetime1"] = dt.datetime(2001, 1, 2, 0, 0)
    df["datetime2"] = dt.datetime(2001, 1, 3, 0, 0)
    df.loc[df.index[3:6], ["obj1"]] = np.nan
    df = df._consolidate()

    warning = None if using_infer_string else performance_warning
    msg = "cannot\nmap directly to c-types .* dtype='object'"
    with tm.assert_produces_warning(warning, match=msg):
        store["df"] = df

    # make a random group in hdf space
    store._handle.create_group(store._handle.root, "bah")

    assert store.filename in repr(store)
    assert store.filename in str(store)
    store.info()


def test_repr_get_storer(temp_hdfstore):
    # [FUNCTION DOC]
    # repr and str of a table storer work without error.
    # storers
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    temp_hdfstore.append("df", df)

    s = temp_hdfstore.get_storer("df")
    repr(s)
    str(s)


def test_contains(temp_hdfstore):
    # [FUNCTION DOC]
    # Membership tests for plain keys, nested keys with and without a leading slash, and absent keys (partial paths are not members). A key with unusual characters triggers a natural-name warning (stack level not checked) but is still contained (gh-2694).
    store = temp_hdfstore
    store["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    store["b"] = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    store["foo/bar"] = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    assert "a" in store
    assert "b" in store
    assert "c" not in store
    assert "foo/bar" in store
    assert "/foo/bar" in store
    assert "/foo/b" not in store
    assert "bar" not in store

    # gh-2694: tables.NaturalNameWarning
    with tm.assert_produces_warning(tables.NaturalNameWarning, check_stacklevel=False):
        store["node())"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
    assert "node())" in store


def test_versioning(temp_hdfstore):
    # [FUNCTION DOC]
    # Stored objects carry the file format version attribute 0.15.2 for fixed and table objects. Clearing the version attribute of a table node makes select fail with an exception about a missing string method on none.
    store = temp_hdfstore
    store["a"] = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    store["b"] = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=20, freq="B"),
    )
    store.append("df1", df[:10])
    store.append("df1", df[10:])
    assert store.root.a._v_attrs.pandas_version == "0.15.2"
    assert store.root.b._v_attrs.pandas_version == "0.15.2"
    assert store.root.df1._v_attrs.pandas_version == "0.15.2"

    # write a file and wipe its versioning
    store.append("df2", df)

    # this is an error because its table_type is appendable, but no
    # version info
    store.get_node("df2")._v_attrs.pandas_version = None

    msg = "'NoneType' object has no attribute 'startswith'"

    with pytest.raises(Exception, match=msg):
        store.select("df2")


@pytest.mark.parametrize(
    "where, expected",
    [
        (
            "/",
            {
                "": ({"first_group", "second_group"}, set()),
                "/first_group": (set(), {"df1", "df2"}),
                "/second_group": ({"third_group"}, {"df3", "s1"}),
                "/second_group/third_group": (set(), {"df4"}),
            },
        ),
        (
            "/second_group",
            {
                "/second_group": ({"third_group"}, {"df3", "s1"}),
                "/second_group/third_group": (set(), {"df4"}),
            },
        ),
    ],
)
def test_walk(where, expected, temp_hdfstore):
    # [FUNCTION DOC]
    # GH10143. Parametrized over walking from the root and from a subgroup. Builds a hierarchy of frames, a series and non-pandas array and table nodes (which must be ignored), then asserts the number of visited groups, that each visited path has the expected subgroups and leaves, and that each leaf object retrieved equals the original.
    # GH10143
    objs = {
        "df1": DataFrame([1, 2, 3]),
        "df2": DataFrame([4, 5, 6]),
        "df3": DataFrame([6, 7, 8]),
        "df4": DataFrame([9, 10, 11]),
        "s1": Series([10, 9, 8]),
        # Next 3 items aren't pandas objects and should be ignored
        "a1": np.array([[1, 2, 3], [4, 5, 6]]),
        "tb1": np.array([(1, 2, 3), (4, 5, 6)], dtype="i,i,i"),
        "tb2": np.array([(7, 8, 9), (10, 11, 12)], dtype="i,i,i"),
    }

    store = temp_hdfstore
    store.put("/first_group/df1", objs["df1"])
    store.put("/first_group/df2", objs["df2"])
    store.put("/second_group/df3", objs["df3"])
    store.put("/second_group/s1", objs["s1"])
    store.put("/second_group/third_group/df4", objs["df4"])
    # Create non-pandas objects
    store._handle.create_array("/first_group", "a1", objs["a1"])
    store._handle.create_table("/first_group", "tb1", obj=objs["tb1"])
    store._handle.create_table("/second_group", "tb2", obj=objs["tb2"])

    assert len(list(store.walk(where=where))) == len(expected)
    for path, groups, leaves in store.walk(where=where):
        assert path in expected
        expected_groups, expected_frames = expected[path]
        assert expected_groups == set(groups)
        assert expected_frames == set(leaves)
        for leaf in leaves:
            frame_path = "/".join([path, leaf])
            obj = store.get(frame_path)
            if "df" in leaf:
                tm.assert_frame_equal(obj, objs[leaf])
            else:
                tm.assert_series_equal(obj, objs[leaf])


def test_getattr(temp_hdfstore):
    # [FUNCTION DOC]
    # Attribute-style access returns stored series and frames. Accessing attributes that do not correspond to keys (including public names for mode, path, handle and complib) raises an attribute error; the underscore-prefixed internal attributes exist.
    store = temp_hdfstore
    s = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    store["a"] = s

    # test attribute access
    result = store.a
    tm.assert_series_equal(result, s)
    result = store.a
    tm.assert_series_equal(result, s)

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    store["df"] = df
    result = store.df
    tm.assert_frame_equal(result, df)

    # errors
    for x in ["d", "mode", "path", "handle", "complib"]:
        msg = f"'HDFStore' object has no attribute '{x}'"
        with pytest.raises(AttributeError, match=msg):
            getattr(store, x)

    # not stores
    for x in ["mode", "path", "handle", "complib"]:
        getattr(store, f"_{x}")


def test_store_dropna(temp_h5_path):
    # [FUNCTION DOC]
    # Default write keeps rows with missing values, dropna false keeps them, and dropna true drops rows where all values are missing (as in issue 9382). Written with to_hdf in table format and read back.
    df_with_missing = DataFrame(
        {"col1": [0.0, np.nan, 2.0], "col2": [1.0, np.nan, np.nan]},
        index=list("abc"),
    )
    df_without_missing = DataFrame(
        {"col1": [0.0, 2.0], "col2": [1.0, np.nan]}, index=list("ac")
    )

    # # Test to make sure defaults are to not drop.
    # # Corresponding to Issue 9382
    df_with_missing.to_hdf(temp_h5_path, key="df", format="table")
    reloaded = read_hdf(temp_h5_path, "df")
    tm.assert_frame_equal(df_with_missing, reloaded)

    df_with_missing.to_hdf(temp_h5_path, key="df", format="table", dropna=False)
    reloaded = read_hdf(temp_h5_path, "df")
    tm.assert_frame_equal(df_with_missing, reloaded)

    df_with_missing.to_hdf(temp_h5_path, key="df", format="table", dropna=True)
    reloaded = read_hdf(temp_h5_path, "df")
    tm.assert_frame_equal(df_without_missing, reloaded)


def test_to_hdf_with_min_itemsize(temp_h5_path):
    # [FUNCTION DOC]
    # GH 10381. min_itemsize for the index in to_hdf, then appending a frame with a longer index string works for a frame and for a series; asserts concatenated results.
    # min_itemsize in index with to_hdf (GH 10381)
    df = DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"]),
            "D": date_range("20130101", periods=5),
        }
    ).set_index("C")
    df.to_hdf(temp_h5_path, key="ss3", format="table", min_itemsize={"index": 6})
    # just make sure there is a longer string:
    df2 = df.copy().reset_index().assign(C="longer").set_index("C")
    df2.to_hdf(temp_h5_path, key="ss3", append=True, format="table")
    tm.assert_frame_equal(read_hdf(temp_h5_path, "ss3"), concat([df, df2]))

    # same as above, with a Series
    df["B"].to_hdf(temp_h5_path, key="ss4", format="table", min_itemsize={"index": 6})
    df2["B"].to_hdf(temp_h5_path, key="ss4", append=True, format="table")
    tm.assert_series_equal(read_hdf(temp_h5_path, "ss4"), concat([df["B"], df2["B"]]))


@pytest.mark.parametrize("format", ["fixed", "table"])
def test_to_hdf_errors(format, temp_h5_path, using_infer_string):
    # [FUNCTION DOC]
    # GH 20835. Parametrized over fixed and table formats. A string with a lone surrogate character in data and index is written with the surrogatepass error handler and read back with the same handler. The expected dtype is object, or python-backed string with NaN under infer-string mode (surrogates fall back to python storage, PR 60993).
    data = ["\ud800foo"]
    ser = Series(data, index=Index(data, dtype="object"), dtype="object")
    # GH 20835
    ser.to_hdf(temp_h5_path, key="table", format=format, errors="surrogatepass")

    result = read_hdf(temp_h5_path, "table", errors="surrogatepass")

    if using_infer_string:
        # https://github.com/pandas-dev/pandas/pull/60993
        # Surrogates fallback to python storage.
        dtype = pd.StringDtype(storage="python", na_value=np.nan)
    else:
        dtype = "object"
    expected = Series(data, index=Index(data, dtype=dtype), dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_create_table_index(temp_hdfstore):
    # [FUNCTION DOC]
    # Appending with data columns indexes the index and the data columns; appending with an explicit index list indexes only the named column and not the index; asking to create a table index on a fixed-format key raises a type error.
    store = temp_hdfstore

    def col(t, column):
        # [FUNCTION DOC]
        # HELPER: returns the PyTables column object named column from the table of the stored key t in the enclosing test's store, to inspect its index flag.
        return getattr(store.get_storer(t).table.cols, column)

    # data columns
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df["string"] = "foo"
    df["string2"] = "bar"
    store.append("f", df, data_columns=["string", "string2"])
    assert col("f", "index").is_indexed is True
    assert col("f", "string").is_indexed is True
    assert col("f", "string2").is_indexed is True

    # specify index=columns
    store.append("f2", df, index=["string"], data_columns=["string", "string2"])
    assert col("f2", "index").is_indexed is False
    assert col("f2", "string").is_indexed is True
    assert col("f2", "string2").is_indexed is False

    # try to index a non-table
    store.put("f2", df)
    msg = "cannot create table index on a Fixed format store"
    with pytest.raises(TypeError, match=msg):
        store.create_table_index("f2")


def test_create_table_index_data_columns_argument(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 28156. With one data column, the index and that column are indexed; accessing a non-data column raises an attribute error on the columns object; asking create_table_index to index a non-data column raises an attribute error explaining it must be included in data_columns when loading.
    # GH 28156

    store = temp_hdfstore

    def col(t, column):
        # [FUNCTION DOC]
        # HELPER: returns the PyTables column object named column from the table of the stored key t in the enclosing test's store, to inspect its index flag.
        return getattr(store.get_storer(t).table.cols, column)

    # data columns
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df["string"] = "foo"
    df["string2"] = "bar"
    store.append("f", df, data_columns=["string"])
    assert col("f", "index").is_indexed is True
    assert col("f", "string").is_indexed is True

    msg = "'Cols' object has no attribute 'string2'"
    with pytest.raises(AttributeError, match=msg):
        col("f", "string2").is_indexed

    # try to index a col which isn't a data_column
    msg = (
        "column string2 is not a data_column.\n"
        "In order to read column string2 you must reload the dataframe \n"
        "into HDFStore and include string2 with the data_columns argument."
    )
    with pytest.raises(AttributeError, match=msg):
        store.create_table_index("f", columns=["string2"])


def test_mi_data_columns(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 14435. Appending a frame with a named two-level index as data columns lets where select by a level name; asserts the single matching row.
    # GH 14435
    idx = MultiIndex.from_arrays(
        [date_range("2000-01-01", periods=5), range(5)], names=["date", "id"]
    )
    df = DataFrame({"a": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=idx)

    temp_hdfstore.append("df", df, data_columns=True)

    actual = temp_hdfstore.select("df", where="id == 1")
    expected = df.iloc[[1], :]
    tm.assert_frame_equal(actual, expected)


def test_table_mixed_dtypes(temp_hdfstore):
    # [FUNCTION DOC]
    # Appending a consolidated frame with string, boolean, integer, timestamp and datetime columns (some missing strings) round-trips.
    # frame
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["bool1"] = df["A"] > 0
    df["bool2"] = df["B"] > 0
    df["bool3"] = True
    df["int1"] = 1
    df["int2"] = 2
    df["timestamp1"] = Timestamp("20010102").as_unit("ns")
    df["timestamp2"] = Timestamp("20010103").as_unit("ns")
    df["datetime1"] = Timestamp("20010102").as_unit("ns")
    df["datetime2"] = Timestamp("20010103").as_unit("ns")
    df.loc[df.index[3:6], ["obj1"]] = np.nan
    df = df._consolidate()

    temp_hdfstore.append("df1_mixed", df)
    tm.assert_frame_equal(temp_hdfstore.select("df1_mixed"), df)


def test_calendar_roundtrip_issue(temp_hdfstore):
    # [FUNCTION DOC]
    # Issue 8591. A series indexed by dates from a custom business-day offset (with weekmask and holidays of different types) stored in fixed and table format round-trips.
    # 8591
    # doc example from tseries holiday section
    weekmask_egypt = "Sun Mon Tue Wed Thu"
    holidays = [
        "2012-05-01",
        dt.datetime(2013, 5, 1),
        np.datetime64("2014-05-01"),
    ]
    bday_egypt = pd.offsets.CustomBusinessDay(
        holidays=holidays, weekmask=weekmask_egypt
    )
    mydt = dt.datetime(2013, 4, 30)
    dts = date_range(mydt, periods=5, freq=bday_egypt)

    s = Series(dts.weekday, dts).map(Series("Mon Tue Wed Thu Fri Sat Sun".split()))

    temp_hdfstore.put("fixed", s)
    result = temp_hdfstore.select("fixed")
    tm.assert_series_equal(result, s)

    temp_hdfstore.append("table", s)
    result = temp_hdfstore.select("table")
    tm.assert_series_equal(result, s)


def test_remove(temp_hdfstore):
    # [FUNCTION DOC]
    # Removing keys by name, nested name and parent group; removing a nonexistent key raises a key error; deleting by item deletion also removes objects; the store length is asserted after each step.
    store = temp_hdfstore
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    store["a"] = ts
    store["b"] = df
    store.remove("a")
    assert len(store) == 1
    tm.assert_frame_equal(df, store["b"])

    store.remove("b")
    assert len(store) == 0

    # nonexistence
    with pytest.raises(
        KeyError, match="'No object named a_nonexistent_store in the file'"
    ):
        store.remove("a_nonexistent_store")

    # pathing
    store["a"] = ts
    store["b/foo"] = df
    store.remove("b/foo")
    assert len(store) == 1

    store["a"] = ts
    store["b/foo"] = df
    store.remove("b")
    assert len(store) == 1

    # __delitem__
    store["a"] = ts
    store["b"] = df
    del store["a"]
    del store["b"]
    assert len(store) == 0


def test_same_name_scoping(temp_hdfstore):
    # [FUNCTION DOC]
    # Selecting with an index condition that refers to the datetime module attribute and to the bare datetime name returns the same rows as a direct comparison.
    # KNOWN ISSUE: a comment says the namespace is changed so that the lookup name points elsewhere ("technically an error, but allow it"), but no such change is made and the first two selections are identical.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((20, 2)),
        index=date_range("20130101", periods=20, unit="ns"),
    )
    temp_hdfstore.put("df", df, format="table")
    expected = df[df.index > Timestamp("20130105")]

    result = temp_hdfstore.select("df", "index>datetime.datetime(2013,1,5)")
    tm.assert_frame_equal(result, expected)

    # changes what 'datetime' points to in the namespace where
    #  'select' does the lookup

    # technically an error, but allow it
    result = temp_hdfstore.select("df", "index>datetime.datetime(2013,1,5)")
    tm.assert_frame_equal(result, expected)

    result = temp_hdfstore.select("df", "index>datetime(2013,1,5)")
    tm.assert_frame_equal(result, expected)


def test_store_index_name(temp_hdfstore):
    # [FUNCTION DOC]
    # An index name round-trips through item assignment.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    df.index.name = "foo"

    temp_hdfstore["frame"] = df
    recons = temp_hdfstore["frame"]
    tm.assert_frame_equal(recons, df)


@pytest.mark.parametrize("tz", [None, "US/Pacific"])
@pytest.mark.parametrize("table_format", ["table", "fixed"])
def test_store_index_name_numpy_str(table_format, temp_h5_path, unit, tz):
    # [FUNCTION DOC]
    # GH #13492. Parametrized over table/fixed formats, timezone (none or US/Pacific) and datetime unit. Index and column names containing non-ascii characters round-trip with check of names, and the retrieved names are native strings (not numpy strings).
    # GH #13492
    idx = (
        DatetimeIndex(
            [dt.date(2000, 1, 1), dt.date(2000, 1, 2)],
            name="cols\u05d2",
        )
        .tz_localize(tz)
        .as_unit(unit)
    )
    idx1 = (
        DatetimeIndex(
            [dt.date(2010, 1, 1), dt.date(2010, 1, 2)],
            name="rows\u05d0",
        )
        .as_unit(unit)
        .tz_localize(tz)
    )
    df = DataFrame(np.arange(4).reshape(2, 2), columns=idx, index=idx1)

    # This used to fail, returning numpy strings instead of python strings.
    df.to_hdf(temp_h5_path, key="df", format=table_format)
    df2 = read_hdf(temp_h5_path, "df")

    tm.assert_frame_equal(df, df2, check_names=True)

    assert isinstance(df2.index.name, str)
    assert isinstance(df2.columns.name, str)


def test_store_series_name(temp_hdfstore):
    # [FUNCTION DOC]
    # A named series round-trips through item assignment.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    series = df["A"]

    temp_hdfstore["series"] = series
    recons = temp_hdfstore["series"]
    tm.assert_series_equal(recons, series)


def test_overwrite_node(temp_hdfstore):
    # [FUNCTION DOC]
    # Assigning a series to a key that holds a frame replaces it; asserts the series is returned.
    temp_hdfstore["a"] = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    temp_hdfstore["a"] = ts

    tm.assert_series_equal(temp_hdfstore["a"], ts)


def test_coordinates(temp_hdfstore):
    # [FUNCTION DOC]
    # select_as_coordinates without a where returns all row positions.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    store = temp_hdfstore
    store.append("df", df)

    # all
    c = store.select_as_coordinates("df")
    assert (c.values == np.arange(len(df.index))).all()


def test_coordinates2(temp_hdfstore):
    # [FUNCTION DOC]
    # select_as_coordinates with conditions returns positions; passing them as where to select returns the corresponding rows; the coordinates object is an index.
    # get coordinates back & test vs frame
    store = temp_hdfstore

    df = DataFrame({"A": range(5), "B": range(5)})
    store.append("df", df)
    c = store.select_as_coordinates("df", ["index<3"])
    assert (c.values == np.arange(3)).all()
    result = store.select("df", where=c)
    expected = df.loc[0:2, :]
    tm.assert_frame_equal(result, expected)

    c = store.select_as_coordinates("df", ["index>=3", "index<=4"])
    assert (c.values == np.arange(2) + 3).all()
    result = store.select("df", where=c)
    expected = df.loc[3:4, :]
    tm.assert_frame_equal(result, expected)
    assert isinstance(c, Index)


def test_coordinates_multiple_tables(temp_hdfstore):
    # [FUNCTION DOC]
    # Coordinates computed from conditions on the first table are used to select the same rows from a second table; concatenated results equal the filtered concatenation of the frames.
    # KNOWN ISSUE: a FIXME notes that the frequency check is disabled (check_freq false) because some builds (mostly Windows) return no frequency instead of the expected business-day frequency.
    store = temp_hdfstore
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df2 = df1.copy().rename(columns="{}_2".format)
    store.append("df1", df1, data_columns=["A", "B"])
    store.append("df2", df2)

    c = store.select_as_coordinates("df1", ["A>0", "B>0"])
    df1_result = store.select("df1", c)
    df2_result = store.select("df2", c)
    result = concat([df1_result, df2_result], axis=1)

    expected = concat([df1, df2], axis=1)
    expected = expected[(expected.A > 0) & (expected.B > 0)]
    tm.assert_frame_equal(result, expected, check_freq=False)
    # FIXME: 2021-01-18 on some (mostly windows) builds we get freq=None
    #  but expect freq="18B"


def test_coordinates_array_mask(temp_hdfstore):
    # [FUNCTION DOC]
    # Selecting with where given as an index of row positions, a plain list of positions, and a boolean list returns the expected rows; arrays of the wrong dtype or length, or combined with start and stop, raise a type error about acceptable where types; a where using an index-membership expression works; start and stop slicing works.
    # KNOWN ISSUE: the sections commented as locations and boolean both run identical code with the same where index, so the boolean case is not actually tested separately until the later list-based boolean section.
    store = temp_hdfstore
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 2)),
        index=date_range("20000101", periods=1000),
    )
    store.append("df", df)
    c = store.select_column("df", "index")
    where = c[DatetimeIndex(c).month == 5].index
    expected = df.iloc[where]

    # locations
    result = store.select("df", where=where)
    tm.assert_frame_equal(result, expected)

    # boolean
    result = store.select("df", where=where)
    tm.assert_frame_equal(result, expected)

    # invalid
    msg = "where must be passed as a string, PyTablesExpr, or list-like of PyTablesExpr"
    with pytest.raises(TypeError, match=msg):
        store.select("df", where=np.arange(len(df), dtype="float64"))

    with pytest.raises(TypeError, match=msg):
        store.select("df", where=np.arange(len(df) + 1))

    with pytest.raises(TypeError, match=msg):
        store.select("df", where=np.arange(len(df)), start=5)

    with pytest.raises(TypeError, match=msg):
        store.select("df", where=np.arange(len(df)), start=5, stop=10)

    # selection with filter
    selection = date_range("20000101", periods=500)
    result = store.select("df", where="index in selection")
    expected = df[df.index.isin(selection)]
    tm.assert_frame_equal(result, expected)

    # list
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    store.append("df2", df)
    result = store.select("df2", where=[0, 3, 5])
    expected = df.iloc[[0, 3, 5]]
    tm.assert_frame_equal(result, expected)

    # boolean
    where = [True] * 10
    where[-2] = False
    result = store.select("df2", where=where)
    expected = df.loc[where]
    tm.assert_frame_equal(result, expected)

    # start/stop
    result = store.select("df2", start=5, stop=10)
    expected = df[5:10]
    tm.assert_frame_equal(result, expected)


def test_start_stop_table(temp_hdfstore):
    # [FUNCTION DOC]
    # Select with a column condition and start and stop returns the expected rows; an out-of-range window returns an empty frame.
    # table
    df = DataFrame(
        {
            "A": np.random.default_rng(2).random(20),
            "B": np.random.default_rng(2).random(20),
        }
    )
    temp_hdfstore.append("df", df)

    result = temp_hdfstore.select("df", "columns=['A']", start=0, stop=5)
    expected = df.loc[0:4, ["A"]]
    tm.assert_frame_equal(result, expected)

    # out of range
    result = temp_hdfstore.select("df", "columns=['A']", start=30, stop=40)
    assert len(result) == 0
    expected = df.loc[30:40, ["A"]]
    tm.assert_frame_equal(result, expected)


def test_start_stop_multiple(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 16209. append_to_multiple into a selector table and a data table; select_as_multiple with start and stop returns the first row with both columns.
    # GH 16209
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]})

    temp_hdfstore.append_to_multiple(
        {"selector": ["foo"], "data": None}, df, selector="selector"
    )
    result = temp_hdfstore.select_as_multiple(
        ["selector", "data"], selector="selector", start=0, stop=1
    )
    expected = df.loc[[0], ["foo", "bar"]]
    tm.assert_frame_equal(result, expected)


def test_start_stop_fixed(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 8287. Fixed-format frame and series with start and stop windows return the matching slices, including an out-of-range window (empty).
    # KNOWN ISSUE: the end of the body builds a frame with some missing values for a sparse case marked as not implemented, but never stores or asserts anything with it (dead code).
    # fixed, GH 8287
    store = temp_hdfstore
    df = DataFrame(
        {
            "A": np.random.default_rng(2).random(20),
            "B": np.random.default_rng(2).random(20),
        },
        index=date_range("20130101", periods=20),
    )
    store.put("df", df)

    result = store.select("df", start=0, stop=5)
    expected = df.iloc[0:5, :]
    tm.assert_frame_equal(result, expected)

    result = store.select("df", start=5, stop=10)
    expected = df.iloc[5:10, :]
    tm.assert_frame_equal(result, expected)

    # out of range
    result = store.select("df", start=30, stop=40)
    expected = df.iloc[30:40, :]
    tm.assert_frame_equal(result, expected)

    # series
    s = df.A
    store.put("s", s)
    result = store.select("s", start=0, stop=5)
    expected = s.iloc[0:5]
    tm.assert_series_equal(result, expected)

    result = store.select("s", start=5, stop=10)
    expected = s.iloc[5:10]
    tm.assert_series_equal(result, expected)

    # sparse; not implemented
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    df.iloc[3:5, 1:3] = np.nan
    df.iloc[8:10, -2] = np.nan


def test_select_filter_corner(temp_hdfstore, request):
    # [FUNCTION DOC]
    # Selecting columns by a condition that slices the frame's column labels (prefix, and prefix with step) returns the expected column subsets. Expected to fail with a value error on Python 3.12 or later (request fixture).
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    df.index = [f"{c:3d}" for c in df.index]
    df.columns = [f"{c:3d}" for c in df.columns]

    temp_hdfstore.put("frame", df, format="table")

    request.applymarker(
        pytest.mark.xfail(
            PY312,
            reason="AST change in PY312",
            raises=ValueError,
        )
    )
    crit = "columns=df.columns[:75]"
    result = temp_hdfstore.select("frame", [crit])
    tm.assert_frame_equal(result, df.loc[:, df.columns[:75]])

    crit = "columns=df.columns[:75:2]"
    result = temp_hdfstore.select("frame", [crit])
    tm.assert_frame_equal(result, df.loc[:, df.columns[:75:2]])


def test_path_pathlib(temp_h5_path):
    # [FUNCTION DOC]
    # Uses the testing round-trip helper with a path-like object to write with to_hdf and read with read_hdf; asserts equality.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    result = tm.round_trip_pathlib(
        lambda p: df.to_hdf(p, key="df"), lambda p: read_hdf(p, "df"), temp_h5_path
    )
    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize("start, stop", [(0, 2), (1, 2), (None, None)])
def test_contiguous_mixed_data_table(start, stop, temp_hdfstore):
    # [FUNCTION DOC]
    # GH 17021. Parametrized over start/stop pairs: selecting a slice from a table with an integer and a string column returns that slice of the frame.
    # GH 17021
    df = DataFrame(
        {
            "a": Series([20111010, 20111011, 20111012]),
            "b": Series(["ab", "cd", "ab"]),
        }
    )

    temp_hdfstore.append("test_dataset", df)

    result = temp_hdfstore.select("test_dataset", start=start, stop=stop)
    tm.assert_frame_equal(df[start:stop], result)


def test_path_pathlib_hdfstore(temp_h5_path):
    # [FUNCTION DOC]
    # Same as the path-like test but the writer and reader helpers use an HDFStore opened on the path.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    def writer(path):
        # [FUNCTION DOC]
        # HELPER: writes the enclosing test's frame to the given path through an HDFStore opened on that path (context managed).
        with HDFStore(path) as store:
            df.to_hdf(store, key="df")

    def reader(path):
        # [FUNCTION DOC]
        # HELPER: opens an HDFStore on the given path and returns the frame read from it.
        with HDFStore(path) as store:
            return read_hdf(store, "df")

    result = tm.round_trip_pathlib(writer, reader, temp_h5_path)
    tm.assert_frame_equal(df, result)


def test_pickle_path_localpath(temp_h5_path):
    # [FUNCTION DOC]
    # Round-trips a frame through to_hdf and read_hdf using the testing round-trip helper with a path object.
    # KNOWN ISSUE: the body is identical to the path-like round-trip test and does not use a legacy local-path object or pickling as the name suggests.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    result = tm.round_trip_pathlib(
        lambda p: df.to_hdf(p, key="df"), lambda p: read_hdf(p, "df"), temp_h5_path
    )
    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize("propindexes", [True, False])
def test_copy(propindexes, temp_file):
    # [FUNCTION DOC]
    # Parametrized over propindexes true/false. Writes a table with one data column, copies the store to a new temporary file (all keys), and asserts the copied store has the same keys, the same row counts for tables and, when index properties are propagated, the same indexed axes.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    with HDFStore(temp_file) as st:
        st.append("df", df, data_columns=["A"])
    with tempfile.NamedTemporaryFile() as new_f:
        with HDFStore(temp_file) as store:
            with contextlib.closing(
                store.copy(new_f.name, keys=None, propindexes=propindexes)
            ) as tstore:
                # check keys
                keys = store.keys()
                assert set(keys) == set(tstore.keys())
                # check indices & nrows
                for k in tstore.keys():
                    if tstore.get_storer(k).is_table:
                        new_t = tstore.get_storer(k)
                        orig_t = store.get_storer(k)

                        assert orig_t.nrows == new_t.nrows

                        # check propindixes
                        if propindexes:
                            for a in orig_t.axes:
                                if a.is_indexed:
                                    assert new_t[a.name].is_indexed


def test_duplicate_column_name(temp_h5_path):
    # [FUNCTION DOC]
    # A frame with duplicate column labels: fixed format raises a value error that the column index must be unique; table format round-trips and compares equal in both directions.
    df = DataFrame(columns=["a", "a"], data=[[0, 0]])

    msg = "Columns index has to be unique for fixed format"
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(temp_h5_path, key="df", format="fixed")

    df.to_hdf(temp_h5_path, key="df", format="table")
    other = read_hdf(temp_h5_path, "df")

    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)


def test_preserve_timedeltaindex_type(temp_hdfstore, unit):
    # [FUNCTION DOC]
    # GH9635. Parametrized unit. A frame with a named timedelta index round-trips via item assignment.
    # GH9635
    df = DataFrame(np.random.default_rng(2).normal(size=(10, 5)))
    df.index = timedelta_range(
        start="0s", periods=10, freq="1s", name="example", unit=unit
    )

    temp_hdfstore["df"] = df
    tm.assert_frame_equal(temp_hdfstore["df"], df)


def test_columns_multiindex_modified(temp_h5_path):
    # [FUNCTION DOC]
    # BUG 7212 and GH#10055. Writing a frame with a multi-level index through to_hdf with data columns and no index, then calling read_hdf with a list of columns to load does not modify that list.
    # BUG: 7212

    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    df.index.name = "letters"
    df = df.set_index(keys="E", append=True)

    data_columns = df.index.names + df.columns.tolist()
    df.to_hdf(
        temp_h5_path,
        key="df",
        mode="a",
        append=True,
        data_columns=data_columns,
        index=False,
    )
    cols2load = list("BCD")
    cols2load_original = list(cols2load)
    # GH#10055 make sure read_hdf call does not alter cols2load inplace
    read_hdf(temp_h5_path, "df", columns=cols2load)
    assert cols2load_original == cols2load


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "columns",
    [
        Index([0, 1], dtype=np.int64),
        Index([0.0, 1.0], dtype=np.float64),
        date_range("2020-01-01", periods=2),
        timedelta_range("1 day", periods=2),
        period_range("2020-01-01", periods=2, freq="D"),
    ],
)
def test_to_hdf_with_object_column_names_should_fail(temp_h5_path, columns):
    # [FUNCTION DOC]
    # GH9057. Parametrized over integer, float, datetime, timedelta and period column labels: writing in table format with all columns as data columns raises a value error that non-object labels cannot be data indexable columns. Period business-day deprecation warning is ignored.
    # GH9057
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=columns)
    msg = "cannot have non-object label DataIndexableCol"
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(temp_h5_path, key="df", format="table", data_columns=True)


@pytest.mark.parametrize("dtype", [None, "category"])
def test_to_hdf_with_object_column_names_should_run(temp_h5_path, dtype):
    # [FUNCTION DOC]
    # GH9057. Parametrized over object and categorical string column labels: writing as table with data columns works and a where condition on the index returns a non-empty result.
    # GH9057
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 2)),
        columns=Index(["a", "b"], dtype=dtype),
    )
    df.to_hdf(temp_h5_path, key="df", format="table", data_columns=True)
    result = read_hdf(temp_h5_path, "df", where=f"index = [{df.index[0]}]")
    assert len(result)


def test_hdfstore_strides(temp_hdfstore):
    # [FUNCTION DOC]
    # GH22073. The strides of a column's values are the same before storing and after retrieving.
    # GH22073
    df = DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    temp_hdfstore.put("df", df)
    assert df["a"].values.strides == temp_hdfstore["df"]["a"].values.strides


def test_store_bool_index(temp_h5_path):
    # [FUNCTION DOC]
    # GH#48667. A frame with boolean column and row labels round-trips in fixed format.
    # KNOWN ISSUE: a leftover comment above the write describes dropna defaults (issue 9382), which has no relation to this test.
    # GH#48667
    df = DataFrame([[1]], columns=[True], index=Index([False], dtype="bool"))
    expected = df.copy()

    # # Test to make sure defaults are to not drop.
    # # Corresponding to Issue 9382
    df.to_hdf(temp_h5_path, key="a")
    result = read_hdf(temp_h5_path, "a")
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("model", ["name", "longname", "verylongname"])
def test_select_categorical_string_columns(temp_hdfstore, model):
    # [FUNCTION DOC]
    # Bug 57608. Parametrized over category values. A categorical column with a fixed set of categories is stored as a data column; a condition comparing it with a local variable returns the matching rows.
    # Corresponding to BUG: 57608

    models = CategoricalDtype(categories=["name", "longname", "verylongname"])
    df = DataFrame(
        {"modelId": ["name", "longname", "longname"], "value": [1, 2, 3]}
    ).astype({"modelId": models, "value": int})

    temp_hdfstore.append("df", df, data_columns=["modelId"])

    result = temp_hdfstore.select("df", "modelId == model")
    expected = df[df["modelId"] == model]
    tm.assert_frame_equal(result, expected)


def test_to_hdf_multiindex_string_dtype_crash(temp_h5_path):
    # [FUNCTION DOC]
    # GH#63412. A frame with a named two-level string index written with to_hdf in default format reads back equal (dtype check disabled).
    # GH#63412
    index = MultiIndex.from_tuples([("a", "x"), ("b", "y")], names=["level1", "level2"])
    df = DataFrame({"value": [1, 2]}, index=index)
    df.to_hdf(temp_h5_path, key="test")
    result = read_hdf(temp_h5_path, key="test")
    tm.assert_frame_equal(df, result, check_dtype=False)
