# [MODULE DOC]
# Copy-on-write tests for DataFrame and Series methods. For each method the
# tests check whether the returned object lazily shares memory with the source
# (a shallow copy that holds references), whether the result has new index and
# columns objects, that writing to the result triggers copy on write for just the
# touched column/block and never alters the source, and (for in-place variants)
# that a live view forces a copy. Methods covered include copy, reset_index,
# rename, reindex, drop, filter, shift, pop, align, to_frame, set_index, prefix
# and suffix helpers, dropna, head/tail, infer_objects, truncate, take, sorting,
# round, level reordering, set_axis, tz conversion, putmask style assignment,
# where/mask, isetitem, xs, update, in-place arithmetic, transpose, transform,
# insert, eval, apply and reductions. Uses the get_array helper from the sibling
# util module and fixtures from the wider test suite.

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
from pandas.util.version import Version


def test_copy():
    # [FUNCTION DOC]
    # Deep copy of a frame: new index and columns objects that are identical (same
    # underlying object) to the originals, no memory sharing, no references on the
    # copy's blocks, and writing to the copy leaves the original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_copy = df.copy()

    # the deep copy by defaults takes a shallow copy of the Index
    assert df_copy.index is not df.index
    assert df_copy.columns is not df.columns
    assert df_copy.index.is_(df.index)
    assert df_copy.columns.is_(df.columns)

    # the deep copy doesn't share memory
    assert not np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    assert not df_copy._mgr.blocks[0].refs.has_reference()
    assert not df_copy._mgr.blocks[1].refs.has_reference()

    assert df_copy.index is not df.index
    assert df_copy.columns is not df.columns

    # mutating copy doesn't mutate original
    df_copy.iloc[0, 0] = 0
    assert df.iloc[0, 0] == 1


def test_copy_shallow():
    # [FUNCTION DOC]
    # Shallow copy: new index and columns objects (identical underlying), shared
    # memory and references on the blocks. Writing to the copy leaves the original
    # unchanged and detaches only the written column; other columns still share.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_copy = df.copy(deep=False)

    # the shallow copy also makes a shallow copy of the index
    assert df_copy.index is not df.index
    assert df_copy.columns is not df.columns
    assert df_copy.index.is_(df.index)
    assert df_copy.columns.is_(df.columns)

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    assert df_copy._mgr.blocks[0].refs.has_reference()
    assert df_copy._mgr.blocks[1].refs.has_reference()

    # mutating shallow copy doesn't mutate original
    df_copy.iloc[0, 0] = 0
    assert df.iloc[0, 0] == 1
    # mutating triggered a copy-on-write -> no longer shares memory
    assert not np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    # but still shares memory for the other columns/blocks
    assert np.shares_memory(get_array(df_copy, "c"), get_array(df, "c"))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("copy", [True, None, False])
@pytest.mark.parametrize(
    "method",
    [
        lambda df, copy: df.rename(columns=str.lower, copy=copy),
        lambda df, copy: df.reindex(columns=["a", "c"], copy=copy),
        lambda df, copy: df.reindex_like(df, copy=copy),
        lambda df, copy: df.align(df, copy=copy)[0],
        lambda df, copy: df.set_axis(["a", "b", "c"], axis="index", copy=copy),
        lambda df, copy: df.rename_axis(index="test", copy=copy),
        lambda df, copy: df.rename_axis(columns="test", copy=copy),
        lambda df, copy: df.astype({"b": "int64"}, copy=copy),
        # lambda df, copy: df.swaplevel(0, 0, copy=copy),
        lambda df, copy: df.truncate(0, 5, copy=copy),
        lambda df, copy: df.infer_objects(copy=copy),
        lambda df, copy: df.to_timestamp(copy=copy),
        lambda df, copy: df.to_period(freq="D", copy=copy),
        lambda df, copy: df.tz_localize("US/Central", copy=copy),
        lambda df, copy: df.tz_convert("US/Central", copy=copy),
        lambda df, copy: df.set_flags(allows_duplicate_labels=False, copy=copy),
    ],
    ids=[
        "rename",
        "reindex",
        "reindex_like",
        "align",
        "set_axis",
        "rename_axis0",
        "rename_axis1",
        "astype",
        # "swaplevel",  # only series
        "truncate",
        "infer_objects",
        "to_timestamp",
        "to_period",
        "tz_localize",
        "tz_convert",
        "set_flags",
    ],
)
def test_methods_copy_keyword(request, method, copy):
    # [FUNCTION DOC]
    # Parametrized over the copy keyword (true, none, false) and a set of frame
    # methods (rename, reindex, reindex_like, align, set_axis, rename_axis on both
    # axes, astype, truncate, infer_objects, to_timestamp, to_period, tz_localize,
    # tz_convert, set_flags), each given as a function of the frame and copy value.
    # Deprecation warnings are ignored. The test id from the request selects a
    # suitable index (period, datetime or tz-aware datetime) for the methods that
    # need one. For every combination the result must share memory with the source,
    # i.e. the copy keyword is ignored.
    index = None
    if "to_timestamp" in request.node.callspec.id:
        index = period_range("2012-01-01", freq="D", periods=3)
    elif "to_period" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_localize" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_convert" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3, tz="Europe/Brussels")

    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]}, index=index)
    df2 = method(df, copy=copy)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("copy", [True, None, False])
@pytest.mark.parametrize(
    "method",
    [
        lambda ser, copy: ser.rename(index={0: 100}, copy=copy),
        lambda ser, copy: ser.rename(None, copy=copy),
        lambda ser, copy: ser.reindex(index=ser.index, copy=copy),
        lambda ser, copy: ser.reindex_like(ser, copy=copy),
        lambda ser, copy: ser.align(ser, copy=copy)[0],
        lambda ser, copy: ser.set_axis(["a", "b", "c"], axis="index", copy=copy),
        lambda ser, copy: ser.rename_axis(index="test", copy=copy),
        lambda ser, copy: ser.astype("int64", copy=copy),
        lambda ser, copy: ser.swaplevel(0, 1, copy=copy),
        lambda ser, copy: ser.truncate(0, 5, copy=copy),
        lambda ser, copy: ser.infer_objects(copy=copy),
        lambda ser, copy: ser.to_timestamp(copy=copy),
        lambda ser, copy: ser.to_period(freq="D", copy=copy),
        lambda ser, copy: ser.tz_localize("US/Central", copy=copy),
        lambda ser, copy: ser.tz_convert("US/Central", copy=copy),
        lambda ser, copy: ser.set_flags(allows_duplicate_labels=False, copy=copy),
    ],
    ids=[
        "rename (dict)",
        "rename",
        "reindex",
        "reindex_like",
        "align",
        "set_axis",
        "rename_axis0",
        "astype",
        "swaplevel",
        "truncate",
        "infer_objects",
        "to_timestamp",
        "to_period",
        "tz_localize",
        "tz_convert",
        "set_flags",
    ],
)
def test_methods_series_copy_keyword(request, method, copy):
    # [FUNCTION DOC]
    # Series counterpart of the previous test, over the copy keyword and a set of
    # Series methods (rename with mapping and with none, reindex, reindex_like,
    # align, set_axis, rename_axis, astype, swaplevel, truncate, infer_objects,
    # to_timestamp, to_period, tz_localize, tz_convert, set_flags). The index is
    # chosen from the test id like in the frame variant (adding a two-level index for
    # swaplevel). The result must always share memory with the source.
    index = None
    if "to_timestamp" in request.node.callspec.id:
        index = period_range("2012-01-01", freq="D", periods=3)
    elif "to_period" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_localize" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_convert" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3, tz="Europe/Brussels")
    elif "swaplevel" in request.node.callspec.id:
        index = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])

    ser = Series([1, 2, 3], index=index)
    ser2 = method(ser, copy=copy)
    assert np.shares_memory(get_array(ser2), get_array(ser))


# -----------------------------------------------------------------------------
# DataFrame methods returning new DataFrame using shallow copy


def test_reset_index():
    # [FUNCTION DOC]
    # reset_index adds a column and returns a shallow copy: the old columns still
    # share memory, manager integrity holds, writing to one column copies only that
    # one and the original is unchanged.
    # Case: resetting the index (i.e. adding a new column) + mutating the
    # resulting dataframe
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]}, index=[10, 11, 12]
    )
    df_orig = df.copy()
    df2 = df.reset_index()
    df2._mgr._verify_integrity()

    # still shares memory (df2 is a shallow copy)
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 2] = 0
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("index", [pd.RangeIndex(0, 2), Index([1, 2])])
def test_reset_index_series_drop(index):
    # [FUNCTION DOC]
    # Parametrized over a range index and a plain integer index. Dropping the index
    # gives a Series sharing memory while the source keeps a reference; writing to
    # the result leaves the source unchanged.
    ser = Series([1, 2], index=index)
    ser_orig = ser.copy()
    ser2 = ser.reset_index(drop=True)
    assert np.shares_memory(get_array(ser), get_array(ser2))
    assert not ser._mgr._has_no_reference(0)

    ser2.iloc[0] = 100
    tm.assert_series_equal(ser, ser_orig)


def test_groupby_column_index_in_references():
    # [FUNCTION DOC]
    # Grouping a frame (indexed by a column) with one of its own columns as the key,
    # fetched as a Series, with observed set, gives the same sums as grouping by
    # the column name.
    df = DataFrame(
        {"A": ["a", "b", "c", "d"], "B": [1, 2, 3, 4], "C": ["a", "a", "b", "b"]}
    )
    df = df.set_index("A")
    key = df["C"]
    result = df.groupby(key, observed=True).sum()
    expected = df.groupby("C", observed=True).sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_modify_series():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: a groupby built with a Series key is
    # unaffected by later modification of that Series; summing afterwards uses the
    # original key values.
    # https://github.com/pandas-dev/pandas/issues/63219
    # Modifying a Series after using it to groupby should not impact
    # the groupby operation.
    ser = Series([1, 2, 1])
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gb = df.groupby(ser)
    ser.iloc[0] = 100
    result = gb.sum()
    expected = DataFrame({"a": [4, 2], "b": [10, 5]}, index=[1, 2])
    tm.assert_frame_equal(result, expected)


def test_rename_columns():
    # [FUNCTION DOC]
    # Renaming columns with a function returns a frame sharing memory; writing to a
    # column detaches just that column; result equals the expected renamed frame and
    # the original is unchanged.
    # Case: renaming columns returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.rename(columns=str.upper)

    assert np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "C"), get_array(df, "c"))
    expected = DataFrame({"A": [0, 2, 3], "B": [4, 5, 6], "C": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(df2, expected)
    tm.assert_frame_equal(df, df_orig)


def test_rename_columns_modify_parent():
    # [FUNCTION DOC]
    # After renaming columns, modifying the parent detaches the touched column in the
    # renamed frame (which keeps its old values) while untouched columns still share.
    # Case: renaming columns returns a new dataframe
    # + afterwards modifying the original (parent) dataframe
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df2 = df.rename(columns=str.upper)
    df2_orig = df2.copy()

    assert np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "C"), get_array(df, "c"))
    expected = DataFrame({"a": [0, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(df, expected)
    tm.assert_frame_equal(df2, df2_orig)


def test_pipe():
    # [FUNCTION DOC]
    # pipe with an identity function returns a frame sharing memory (the function
    # receives a shallow copy); writing to the result detaches only the written
    # column and leaves the original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    df_orig = df.copy()

    def testfunc(df):
        # [FUNCTION DOC]
        # HELPER: nested identity function passed to the pipe method in the test; returns
        # its argument unchanged.
        return df

    df2 = df.pipe(testfunc)

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column
    df2.iloc[0, 0] = 0
    tm.assert_frame_equal(df, df_orig)
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))


def test_pipe_modify_df():
    # [FUNCTION DOC]
    # pipe with a function that modifies its argument in place: since the function
    # receives a shallow copy, the original is unchanged and the written column no
    # longer shares memory with the result while the other column does.
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    df_orig = df.copy()

    def testfunc(df):
        # [FUNCTION DOC]
        # HELPER: nested function passed to the pipe method in the test; sets the first cell of
        # its argument to a new value and returns the argument.
        df.iloc[0, 0] = 100
        return df

    df2 = df.pipe(testfunc)

    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    tm.assert_frame_equal(df, df_orig)
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))


def test_reindex_columns():
    # [FUNCTION DOC]
    # Reindexing columns to a subset returns a shallow copy sharing memory; writing
    # detaches the touched column only; original unchanged.
    # Case: reindexing the column returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.reindex(columns=["a", "c"])

    # still shares memory (df2 is a shallow copy)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # mutating df2 triggers a copy-on-write for that column
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "index",
    [
        lambda idx: idx,
        lambda idx: idx.view(),
        lambda idx: idx.copy(),
        lambda idx: list(idx),
    ],
    ids=["identical", "view", "copy", "values"],
)
def test_reindex_rows(index):
    # [FUNCTION DOC]
    # Parametrized over four ways to supply the same row index (identical object,
    # view, copy, plain list; given as lambdas). Reindexing rows with an equal index
    # returns a shallow copy sharing memory; writing detaches only the touched column;
    # original unchanged.
    # Case: reindexing the rows with an index that matches the current index
    # can use a shallow copy
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.reindex(index=index(df.index))

    # still shares memory (df2 is a shallow copy)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # mutating df2 triggers a copy-on-write for that column
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


def test_drop_on_column():
    # [FUNCTION DOC]
    # Dropping a column keeps manager integrity and shares the remaining columns;
    # writing detaches only the touched column; original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.drop(columns="a")
    df2._mgr._verify_integrity()

    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


def test_select_dtypes():
    # [FUNCTION DOC]
    # Selecting integer columns returns a frame sharing memory with valid manager
    # integrity; writing detaches it; original unchanged.
    # Case: selecting columns using `select_dtypes()` returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.select_dtypes("int64")
    df2._mgr._verify_integrity()

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # mutating df2 triggers a copy-on-write for that column/block
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "filter_kwargs", [{"items": ["a"]}, {"like": "a"}, {"regex": "a"}]
)
def test_filter(filter_kwargs):
    # [FUNCTION DOC]
    # Parametrized over items, like and regex arguments. The filtered frame shares
    # memory; writing detaches it; the original is unchanged.
    # Case: selecting columns using `filter()` returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.filter(**filter_kwargs)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column/block
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_shift_no_op():
    # [FUNCTION DOC]
    # Shifting by zero periods returns a lazy copy: shares memory, new index and
    # columns objects. Modifying the original detaches only the touched column and
    # the result keeps the original values.
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df_orig = df.copy()
    df2 = df.shift(periods=0)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert df2.index is not df.index
    assert df2.columns is not df.columns

    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df2, df_orig)


def test_shift_index():
    # [FUNCTION DOC]
    # A real row shift computes new data: no sharing, with new index and columns
    # objects.
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df2 = df.shift(periods=1, axis=0)

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert df2.index is not df.index
    assert df2.columns is not df.columns


def test_shift_rows_freq():
    # [FUNCTION DOC]
    # Shifting rows with a frequency only moves the index: the data is shared with
    # the source until the source is modified, after which the result keeps the old
    # values with the shifted index.
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df_orig = df.copy()
    df_orig.index = date_range("2020-01-02", "2020-01-04")
    df2 = df.shift(periods=1, freq="1D")

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(df2, df_orig)


def test_shift_columns():
    # [FUNCTION DOC]
    # Shifting along columns by one period shares the data of the first column with
    # the shifted label; modifying the source detaches it. Result equals the expected frame
    # whose first column is all missing.
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]], columns=date_range("2020-01-01", "2020-01-02")
    )
    df2 = df.shift(periods=1, axis=1)

    assert np.shares_memory(get_array(df2, "2020-01-02"), get_array(df, "2020-01-01"))
    df.iloc[0, 0] = 0
    assert not np.shares_memory(
        get_array(df2, "2020-01-02"), get_array(df, "2020-01-01")
    )
    expected = DataFrame(
        [[np.nan, 1], [np.nan, 3], [np.nan, 5]],
        columns=date_range("2020-01-01", "2020-01-02"),
    )
    tm.assert_frame_equal(df2, expected)


def test_pop():
    # [FUNCTION DOC]
    # Popping a column returns a Series sharing memory with a prior view; the
    # remaining column also still shares with it. Writing to the popped Series or to the
    # frame detaches them; the earlier view equals the original.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    view_original = df[:]
    result = df.pop("a")

    assert np.shares_memory(result.values, get_array(view_original, "a"))
    assert np.shares_memory(get_array(df, "b"), get_array(view_original, "b"))

    result.iloc[0] = 0
    assert not np.shares_memory(result.values, get_array(view_original, "a"))
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df, "b"), get_array(view_original, "b"))
    tm.assert_frame_equal(view_original, df_orig)


@pytest.mark.parametrize(
    "func",
    [
        lambda x, y: x.align(y),
        lambda x, y: x.align(y.a, axis=0),
        lambda x, y: x.align(y.a.iloc[slice(0, 1)], axis=1),
    ],
)
def test_align_frame(func):
    # [FUNCTION DOC]
    # Parametrized over three align call forms (align with a frame, with a column
    # Series on axis zero, with a sliced Series on axis one; given as lambdas). The
    # aligned frame shares memory with the source; writing detaches; the original is
    # unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": "a"})
    df_orig = df.copy()
    df_changed = df[["b", "a"]].copy()
    df2, _ = func(df, df_changed)

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_align_series():
    # [FUNCTION DOC]
    # Aligning a Series with a copy of itself returns two Series sharing memory with
    # their respective sources; writing to each detaches that one; both sources
    # equal the original.
    ser = Series([1, 2])
    ser_orig = ser.copy()
    ser_other = ser.copy()
    ser2, ser_other_result = ser.align(ser_other)

    assert np.shares_memory(ser2.values, ser.values)
    assert np.shares_memory(ser_other_result.values, ser_other.values)
    ser2.iloc[0] = 0
    ser_other_result.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    assert not np.shares_memory(ser_other_result.values, ser_other.values)
    tm.assert_series_equal(ser, ser_orig)
    tm.assert_series_equal(ser_other, ser_orig)


def test_align_copy_false():
    # [FUNCTION DOC]
    # Aligning a frame with itself returns frames sharing memory with the source;
    # writing to either result leaves the original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_orig = df.copy()
    df2, df3 = df.align(df)

    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    df3.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged


def test_align_with_series_copy_false():
    # [FUNCTION DOC]
    # Aligning a frame with a Series along axis zero returns a frame and a Series
    # that share memory with their sources; writing to either leaves the originals unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = Series([1, 2, 3], name="x")
    ser_orig = ser.copy()
    df_orig = df.copy()
    df2, ser2 = df.align(ser, axis=0)

    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    assert np.shares_memory(get_array(ser, "x"), get_array(ser2, "x"))

    df2.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    ser2.loc[0] = 0
    tm.assert_series_equal(ser, ser_orig)  # Original is unchanged


def test_to_frame():
    # [FUNCTION DOC]
    # Converting a Series slice to a frame currently always shares memory; writing to
    # the frame detaches and leaves the Series unchanged; modifying the Series does
    # not change an earlier frame; the frame from a plain Series has a new index.
    # Case: converting a Series to a DataFrame with to_frame
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()

    df = ser[:].to_frame()

    # currently this always returns a "view"
    assert np.shares_memory(ser.values, get_array(df, 0))

    df.iloc[0, 0] = 0

    # mutating df triggers a copy-on-write for that column
    assert not np.shares_memory(ser.values, get_array(df, 0))
    tm.assert_series_equal(ser, ser_orig)

    # modify original series -> don't modify dataframe
    df = ser[:].to_frame()
    ser.iloc[0] = 0

    tm.assert_frame_equal(df, ser_orig.to_frame())

    df = ser.to_frame()
    assert df.index is not ser.index


@pytest.mark.parametrize(
    "method, idx",
    [
        (lambda df: df.copy(deep=False).copy(deep=False), 0),
        (lambda df: df.reset_index().reset_index(), 2),
        (lambda df: df.rename(columns=str.upper).rename(columns=str.lower), 0),
        (lambda df: df.copy(deep=False).select_dtypes(include="number"), 0),
    ],
    ids=["shallow-copy", "reset_index", "rename", "select_dtypes"],
)
def test_chained_methods(method, idx):
    # [FUNCTION DOC]
    # Parametrized over four chains of methods (double shallow copy, double
    # reset_index, rename then rename back, shallow copy then select_dtypes) with
    # a column offset idx. Modifying the chained result never modifies the
    # original, and modifying the original never changes a re-created result
    # (compared from offset idx onward).
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    # modify df2 -> don't modify df
    df2 = method(df)
    df2.iloc[0, idx] = 0
    tm.assert_frame_equal(df, df_orig)

    # modify df -> don't modify df2
    df2 = method(df)
    df.iloc[0, 0] = 0
    tm.assert_frame_equal(df2.iloc[:, idx:], df_orig)


@pytest.mark.parametrize("obj", [Series([1, 2], name="a"), DataFrame({"a": [1, 2]})])
def test_to_timestamp(obj):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame, both given a period index. Converting
    # the index to timestamps shares data; writing to the result detaches it; the
    # original is unchanged.
    obj.index = Index([Period("2012-1-1", freq="D"), Period("2012-1-2", freq="D")])

    obj_orig = obj.copy()
    obj2 = obj.to_timestamp()

    assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # mutating obj2 triggers a copy-on-write for that column / block
    obj2.iloc[0] = 0
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize("obj", [Series([1, 2], name="a"), DataFrame({"a": [1, 2]})])
def test_to_period(obj):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame with a timestamp index. Converting the
    # index to periods (yearly) shares the data; writing detaches; original
    # unchanged.
    obj.index = Index([Timestamp("2019-12-31"), Timestamp("2020-12-31")])

    obj_orig = obj.copy()
    obj2 = obj.to_period(freq="Y")

    assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # mutating obj2 triggers a copy-on-write for that column / block
    obj2.iloc[0] = 0
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


def test_set_index():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Making a column the index returns a frame
    # sharing the remaining columns; writing to the result detaches the written
    # column; the original is unchanged.
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.set_index("a")

    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


def test_set_index_mutating_parent_does_not_mutate_index():
    # [FUNCTION DOC]
    # After setting a column as the index, modifying that column in the parent does
    # not change the result.
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    result = df.set_index("a")
    expected = result.copy()

    df.iloc[0, 0] = 100
    tm.assert_frame_equal(result, expected)


def test_add_prefix():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Adding a prefix to labels shares memory;
    # writing detaches the touched column; the result equals the expected prefixed
    # frame and the original is unchanged.
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.add_prefix("CoW_")

    assert np.shares_memory(get_array(df2, "CoW_a"), get_array(df, "a"))
    df2.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(df2, "CoW_a"), get_array(df, "a"))

    assert np.shares_memory(get_array(df2, "CoW_c"), get_array(df, "c"))
    expected = DataFrame(
        {"CoW_a": [0, 2, 3], "CoW_b": [4, 5, 6], "CoW_c": [0.1, 0.2, 0.3]}
    )
    tm.assert_frame_equal(df2, expected)
    tm.assert_frame_equal(df, df_orig)


def test_add_suffix():
    # [FUNCTION DOC]
    # Same as the prefix test but with a suffix: shares memory, writing detaches only
    # the touched column, result as expected, original unchanged.
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.add_suffix("_CoW")
    assert np.shares_memory(get_array(df2, "a_CoW"), get_array(df, "a"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a_CoW"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "c_CoW"), get_array(df, "c"))
    expected = DataFrame(
        {"a_CoW": [0, 2, 3], "b_CoW": [4, 5, 6], "c_CoW": [0.1, 0.2, 0.3]}
    )
    tm.assert_frame_equal(df2, expected)
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis, val", [(0, 5.5), (1, np.nan)])
def test_dropna(axis, val):
    # [FUNCTION DOC]
    # Parametrized over axis and a value placed in the data (row-wise drop with a
    # regular value, column-wise drop with a missing value). The result shares
    # memory with the source; writing detaches; the original is unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, val, 6], "c": "d"})
    df_orig = df.copy()
    df2 = df.dropna(axis=axis)

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("val", [5, 5.5])
def test_dropna_series(val):
    # [FUNCTION DOC]
    # Parametrized over an integer or float middle value. Series dropna returns a
    # Series sharing memory; writing detaches; the source is unchanged.
    ser = Series([1, val, 4])
    ser_orig = ser.copy()
    ser2 = ser.dropna()
    assert np.shares_memory(ser2.values, ser.values)

    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df.head(),
        lambda df: df.head(2),
        lambda df: df.tail(),
        lambda df: df.tail(3),
    ],
)
def test_head_tail(method):
    # [FUNCTION DOC]
    # Parametrized over head/tail calls with default and explicit row counts (given as
    # lambdas). By design these make an eager copy: no memory sharing; manager
    # integrity holds; writes affect only the result; original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = method(df)
    df2._mgr._verify_integrity()

    # We are explicitly deviating for CoW here to make an eager copy (avoids
    # tracking references for very cheap ops)
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # modify df2 to trigger CoW for that block
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_infer_objects(using_infer_string):
    # [FUNCTION DOC]
    # Fixture using_infer_string selects the string behavior. infer_objects keeps
    # the integer column shared; the object column shares memory unless strings are
    # inferred to a pyarrow-backed dtype (when pyarrow is available), in which case
    # it is copied. Writing to both columns detaches them; original unchanged.
    df = DataFrame(
        {"a": [1, 2], "b": Series(["x", "y"], dtype=object), "c": 1, "d": "x"}
    )
    df_orig = df.copy()
    df2 = df.infer_objects()

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    if using_infer_string and HAS_PYARROW:
        assert not tm.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    else:
        assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    df2.iloc[0, 0] = 0
    df2.iloc[0, 1] = "d"
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    tm.assert_frame_equal(df, df_orig)


def test_infer_objects_no_reference(using_infer_string):
    # [FUNCTION DOC]
    # Fixture using_infer_string. After infer_objects on a frame with integer, object
    # string, object datetime and another object column, the result has no
    # references, so writes to the integer and datetime-object columns do not copy
    # (buffers stay shared). The string column is copied, unless it is
    # pyarrow-backed strings, whose array is updated in place and still appears to share
    # memory.
    df = DataFrame(
        {
            "a": [1, 2],
            "b": Series(["x", "y"], dtype=object),
            "c": 1,
            "d": Series(
                [Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype="object"
            ),
            "e": Series(["z", "w"], dtype=object),
        }
    )
    df = df.infer_objects()

    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")
    arr_d = get_array(df, "d")

    df.iloc[0, 0] = 0
    df.iloc[0, 1] = "d"
    df.iloc[0, 3] = Timestamp("2018-12-31")
    assert np.shares_memory(arr_a, get_array(df, "a"))
    if using_infer_string and HAS_PYARROW:
        # note that the underlying memory of arr_b has been copied anyway
        # because of the assignment, but the EA is updated inplace so still
        # appears the share memory
        assert tm.shares_memory(arr_b, get_array(df, "b"))
    else:
        # TODO(CoW): Block splitting causes references here
        assert not np.shares_memory(arr_b, get_array(df, "b"))
    assert np.shares_memory(arr_d, get_array(df, "d"))


def test_infer_objects_reference():
    # [FUNCTION DOC]
    # Same frame as the previous test but with a view alive before inference.
    # Writing to the integer and string columns copies them (no shared buffer), while
    # the datetime-object column still shares memory.
    df = DataFrame(
        {
            "a": [1, 2],
            "b": Series(["x", "y"], dtype=object),
            "c": 1,
            "d": Series(
                [Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype="object"
            ),
        }
    )
    view = df[:]  # noqa: F841
    df = df.infer_objects()

    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")
    arr_d = get_array(df, "d")

    df.iloc[0, 0] = 0
    df.iloc[0, 1] = "d"
    df.iloc[0, 3] = Timestamp("2018-12-31")
    assert not np.shares_memory(arr_a, get_array(df, "a"))
    assert not np.shares_memory(arr_b, get_array(df, "b"))
    assert np.shares_memory(arr_d, get_array(df, "d"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"before": "a", "after": "b", "axis": 1},
        {"before": 0, "after": 1, "axis": 0},
    ],
)
def test_truncate(kwargs):
    # [FUNCTION DOC]
    # Parametrized over truncating columns or rows. The result shares memory;
    # manager integrity holds; writing detaches; original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2})
    df_orig = df.copy()
    df2 = df.truncate(**kwargs)
    df2._mgr._verify_integrity()

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("method", ["assign", "drop_duplicates"])
def test_assign_drop_duplicates(method):
    # [FUNCTION DOC]
    # Parametrized over assign and drop_duplicates with no arguments. The result
    # shares memory; manager integrity holds; writing detaches; original unchanged.
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()
    df2 = getattr(df, method)()
    df2._mgr._verify_integrity()

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("obj", [Series([1, 2]), DataFrame({"a": [1, 2]})])
def test_take(obj):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame. Taking all rows in original order does
    # not copy; writing detaches; the source is unchanged.
    # Check that no copy is made when we take all rows in original order
    obj_orig = obj.copy()
    obj2 = obj.take([0, 1])
    assert np.shares_memory(obj2.values, obj.values)

    obj2.iloc[0] = 0
    assert not np.shares_memory(obj2.values, obj.values)
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize("obj", [Series([1, 2]), DataFrame({"a": [1, 2]})])
def test_between_time(obj):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame with a datetime index. A time-of-day
    # selection that keeps all rows shares memory; writing detaches; the source is
    # unchanged.
    obj.index = date_range("2018-04-09", periods=2, freq="1D20min")
    obj_orig = obj.copy()
    obj2 = obj.between_time("0:00", "1:00")
    assert np.shares_memory(obj2.values, obj.values)

    obj2.iloc[0] = 0
    assert not np.shares_memory(obj2.values, obj.values)
    tm.assert_equal(obj, obj_orig)


def test_reindex_like():
    # [FUNCTION DOC]
    # Reindexing a frame like another with the same columns in a different order
    # shares memory; writing detaches; original unchanged.
    df = DataFrame({"a": [1, 2], "b": "a"})
    other = DataFrame({"b": "a", "a": [1, 2]})

    df_orig = df.copy()
    df2 = df.reindex_like(other)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_sort_index():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Sorting an already sorted Series by index
    # shares memory; writing detaches; source unchanged.
    # GH 49473
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.sort_index()
    assert np.shares_memory(ser.values, ser2.values)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize(
    "obj, kwargs",
    [(Series([1, 2, 3], name="a"), {}), (DataFrame({"a": [1, 2, 3]}), {"by": "a"})],
)
def test_sort_values(obj, kwargs):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame sorted by a column. An already sorted
    # input shares memory with the result; writing detaches; source unchanged.
    obj_orig = obj.copy()
    obj2 = obj.sort_values(**kwargs)
    assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # mutating df triggers a copy-on-write for the column / block
    obj2.iloc[0] = 0
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize(
    "obj, kwargs",
    [(Series([1, 2, 3], name="a"), {}), (DataFrame({"a": [1, 2, 3]}), {"by": "a"})],
)
def test_sort_values_inplace(obj, kwargs):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame. In-place sort of already sorted data
    # keeps sharing memory with a view; writing then detaches and the view stays
    # equal to the original.
    obj_orig = obj.copy()
    view = obj[:]
    obj.sort_values(inplace=True, **kwargs)

    assert np.shares_memory(get_array(obj, "a"), get_array(view, "a"))

    # mutating obj triggers a copy-on-write for the column / block
    obj.iloc[0] = 0
    assert not np.shares_memory(get_array(obj, "a"), get_array(view, "a"))
    tm.assert_equal(view, obj_orig)


@pytest.mark.parametrize("decimals", [-1, 0, 1])
def test_round(decimals):
    # [FUNCTION DOC]
    # Parametrized over decimals (negative, zero, positive). The non-numeric column
    # always shares memory; the numeric column shares memory only for a
    # non-negative decimals value on older numpy versions (a no-op round returns a view
    # there) and otherwise is copied. Result has new index and columns; writes
    # detach; original unchanged.
    df = DataFrame({"a": [1, 2], "b": "c"})
    df_orig = df.copy()
    df2 = df.round(decimals=decimals)

    assert tm.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # TODO: Make inplace by using out parameter of ndarray.round?
    if decimals >= 0 and Version(np.__version__) < Version("2.4.0.dev0"):
        # Ensure lazy copy if no-op
        # TODO: Cannot rely on Numpy returning view after version 2.3
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert df2.index is not df.index
    assert df2.columns is not df.columns

    df2.iloc[0, 1] = "d"
    df2.iloc[0, 0] = 4
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_reorder_levels():
    # [FUNCTION DOC]
    # Reordering index levels of a frame shares data; writing detaches; original
    # unchanged.
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["one", "two"]
    )
    df = DataFrame({"a": [1, 2, 3, 4]}, index=index)
    df_orig = df.copy()
    df2 = df.reorder_levels(order=["two", "one"])
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_series_reorder_levels():
    # [FUNCTION DOC]
    # Series version of reorder_levels: shares memory; writing detaches; source
    # unchanged.
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["one", "two"]
    )
    ser = Series([1, 2, 3, 4], index=index)
    ser_orig = ser.copy()
    ser2 = ser.reorder_levels(order=["two", "one"])
    assert np.shares_memory(ser2.values, ser.values)

    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("obj", [Series([1, 2, 3]), DataFrame({"a": [1, 2, 3]})])
def test_swaplevel(obj):
    # [FUNCTION DOC]
    # Parametrized over a Series and a frame given a two-level index. Swapping
    # levels shares memory; writing detaches; original unchanged.
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["one", "two"])
    obj.index = index
    obj_orig = obj.copy()
    obj2 = obj.swaplevel()
    assert np.shares_memory(obj2.values, obj.values)

    obj2.iloc[0] = 0
    assert not np.shares_memory(obj2.values, obj.values)
    tm.assert_equal(obj, obj_orig)


def test_frame_set_axis():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Setting new row labels on a frame shares
    # memory; writing detaches; original unchanged.
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.set_axis(["a", "b", "c"], axis="index")

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_series_set_axis():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Setting new labels on a Series shares memory
    # (checked on the objects directly); writing detaches; source unchanged.
    # GH 49473
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.set_axis(["a", "b", "c"], axis="index")
    assert np.shares_memory(ser, ser2)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2, ser)
    tm.assert_series_equal(ser, ser_orig)


def test_set_flags():
    # [FUNCTION DOC]
    # Setting flags on a Series shares memory; writing detaches; source unchanged.
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.set_flags(allows_duplicate_labels=False)

    assert np.shares_memory(ser, ser2)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2, ser)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("kwargs", [{"mapper": "test"}, {"index": "test"}])
def test_rename_axis(kwargs):
    # [FUNCTION DOC]
    # Parametrized over the mapper and index keyword forms. Renaming the axis shares
    # memory; writing detaches; original unchanged.
    df = DataFrame({"a": [1, 2, 3, 4]}, index=Index([1, 2, 3, 4], name="a"))
    df_orig = df.copy()
    df2 = df.rename_axis(**kwargs)
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "func, tz", [("tz_convert", "Europe/Berlin"), ("tz_localize", None)]
)
def test_tz_convert_localize(func, tz):
    # [FUNCTION DOC]
    # Parametrized over tz_convert (on a tz-aware index) and tz_localize (naive
    # index). The Series result shares memory; writing detaches; source unchanged.
    # GH 49473
    ser = Series(
        [1, 2], index=date_range(start="2014-08-01 09:00", freq="h", periods=2, tz=tz)
    )
    ser_orig = ser.copy()
    ser2 = getattr(ser, func)("US/Central")
    assert np.shares_memory(ser.values, ser2.values)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


def test_droplevel():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Dropping an index level shares the data
    # columns; writing to one detaches only that column; original unchanged.
    # GH 49473
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["one", "two"])
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=index)
    df_orig = df.copy()
    df2 = df.droplevel(0)

    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    tm.assert_frame_equal(df, df_orig)


def test_squeeze():
    # [FUNCTION DOC]
    # Squeezing a single-column frame to a Series shares memory (it is just a
    # positional selection); writing detaches; frame unchanged.
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()
    series = df.squeeze()

    # Should share memory regardless of CoW since squeeze is just an iloc
    assert np.shares_memory(series.values, get_array(df, "a"))

    # mutating squeezed df triggers a copy-on-write for that column/block
    series.iloc[0] = 0
    assert not np.shares_memory(series.values, get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_items():
    # [FUNCTION DOC]
    # Iterates the columns of a frame twice (the second pass exercises any cache).
    # Each yielded Series shares memory with its column; writing to it detaches it
    # and leaves the frame unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()

    # Test this twice, since the second time, the item cache will be
    # triggered, and we want to make sure it still works then.
    for i in range(2):
        for name, ser in df.items():
            assert np.shares_memory(get_array(ser, name), get_array(df, name))

            # mutating df triggers a copy-on-write for that column / block
            ser.iloc[0] = 0

            assert not np.shares_memory(get_array(ser, name), get_array(df, name))
            tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_putmask(dtype):
    # [FUNCTION DOC]
    # Parametrized over int64 and nullable Int64. With a view alive, setting all
    # cells via a full boolean frame mask copies the data (no sharing with the view);
    # the view is unchanged.
    df = DataFrame({"a": [1, 2], "b": 1, "c": 2}, dtype=dtype)
    view = df[:]
    df_orig = df.copy()
    df[df == df] = 5

    assert not np.shares_memory(get_array(view, "a"), get_array(df, "a"))
    tm.assert_frame_equal(view, df_orig)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_putmask_no_reference(dtype):
    # [FUNCTION DOC]
    # Parametrized over int64 and Int64. The same full-mask assignment without any
    # other reference modifies in place: the buffer is kept.
    df = DataFrame({"a": [1, 2], "b": 1, "c": 2}, dtype=dtype)
    arr_a = get_array(df, "a")
    df[df == df] = 5
    assert np.shares_memory(arr_a, get_array(df, "a"))


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_putmask_aligns_rhs_no_reference(dtype):
    # [FUNCTION DOC]
    # Parametrized over float64 and Float64. Masked assignment of an aligned frame
    # (only some columns present) without other references keeps the buffer of the
    # modified column.
    df = DataFrame({"a": [1.5, 2], "b": 1.5}, dtype=dtype)
    arr_a = get_array(df, "a")
    df[df == df] = DataFrame({"a": [5.5, 5]})
    assert np.shares_memory(arr_a, get_array(df, "a"))


@pytest.mark.parametrize("val, exp, raises", [(5.5, True, True), (5, False, False)])
def test_putmask_dont_copy_some_blocks(val, exp, raises: bool):
    # [FUNCTION DOC]
    # Parametrized over (value, expected-shared flag, raises flag): a float value
    # raises a type error (invalid value for the integer column); an integer
    # value succeeds. In the success case, with a view alive, the modified integer column
    # does not share memory, the same-block integer column shares exactly as the flag
    # says (block is copied as a whole; a TODO notes splitting could avoid that), the
    # float column still shares and has references, and the view is unchanged.
    df = DataFrame({"a": [1, 2], "b": 1, "c": 1.5})
    view = df[:]
    df_orig = df.copy()
    indexer = DataFrame(
        [[True, False, False], [True, False, False]], columns=list("abc")
    )
    if raises:
        with pytest.raises(TypeError, match="Invalid value"):
            df[indexer] = val
    else:
        df[indexer] = val
        assert not np.shares_memory(get_array(view, "a"), get_array(df, "a"))
        # TODO(CoW): Could split blocks to avoid copying the whole block
        assert np.shares_memory(get_array(view, "b"), get_array(df, "b")) is exp
        assert np.shares_memory(get_array(view, "c"), get_array(df, "c"))
        assert df._mgr._has_no_reference(1) is not exp
        assert not df._mgr._has_no_reference(2)
        tm.assert_frame_equal(view, df_orig)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize(
    "func",
    [
        lambda ser: ser.where(ser > 0, 10),
        lambda ser: ser.mask(ser <= 0, 10),
    ],
)
def test_where_mask_noop(dtype, func):
    # [FUNCTION DOC]
    # Parametrized over int64 and Int64 and where/mask forms (lambdas) that change
    # nothing. Result shares memory and has a new index; writing detaches; source
    # unchanged.
    ser = Series([1, 2, 3], dtype=dtype)
    ser_orig = ser.copy()

    result = func(ser)
    assert np.shares_memory(get_array(ser), get_array(result))
    assert result.index is not ser.index

    result.iloc[0] = 10
    assert not np.shares_memory(get_array(ser), get_array(result))
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize(
    "func",
    [
        lambda ser: ser.where(ser < 0, 10),
        lambda ser: ser.mask(ser >= 0, 10),
    ],
)
def test_where_mask(dtype, func):
    # [FUNCTION DOC]
    # Parametrized over dtype and where/mask forms that do replace values. The result
    # does not share memory, has a new index, and the source is unchanged.
    ser = Series([1, 2, 3], dtype=dtype)
    ser_orig = ser.copy()

    result = func(ser)

    assert not np.shares_memory(get_array(ser), get_array(result))
    assert result.index is not ser.index
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("dtype, val", [("int64", 10.5), ("Int64", 10)])
@pytest.mark.parametrize(
    "func",
    [
        lambda df, val: df.where(df < 0, val),
        lambda df, val: df.mask(df >= 0, val),
    ],
)
def test_where_mask_noop_on_single_column(dtype, val, func):
    # [FUNCTION DOC]
    # Parametrized over (dtype, replacement value) and where/mask forms on a
    # frame. The column not touched by the condition shares memory while the
    # touched column does not; writing detaches the shared one; original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [-4, -5, -6]}, dtype=dtype)
    df_orig = df.copy()

    result = func(df, val)
    assert np.shares_memory(get_array(df, "b"), get_array(result, "b"))
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))

    result.iloc[0, 1] = 10
    assert not np.shares_memory(get_array(df, "b"), get_array(result, "b"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("func", ["mask", "where"])
def test_chained_where_mask(func):
    # [FUNCTION DOC]
    # Parametrized over mask and where. In-place use through a chained selection
    # (Series, then single-column frame) emits the chained assignment warning and
    # leaves the frame unchanged.
    df = DataFrame({"a": [1, 4, 2], "b": 1})
    df_orig = df.copy()
    with tm.raises_chained_assignment_error():
        getattr(df["a"], func)(df["a"] > 2, 5, inplace=True)
    tm.assert_frame_equal(df, df_orig)

    with tm.raises_chained_assignment_error():
        getattr(df[["a"]], func)(df["a"] > 2, 5, inplace=True)
    tm.assert_frame_equal(df, df_orig)


def test_asfreq_noop():
    # [FUNCTION DOC]
    # asfreq with the already existing frequency shares memory; writing detaches;
    # original unchanged.
    df = DataFrame(
        {"a": [0.0, None, 2.0, 3.0]},
        index=date_range("1/1/2000", periods=4, freq="min"),
    )
    df_orig = df.copy()
    df2 = df.asfreq(freq="min")
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_iterrows():
    # [FUNCTION DOC]
    # Writing to each row Series yielded by row iteration does not modify the frame.
    df = DataFrame({"a": 0, "b": 1}, index=[1, 2, 3])
    df_orig = df.copy()

    for _, sub in df.iterrows():
        sub.iloc[0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_interpolate_creates_copy():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. After an in-place forward fill (which fills
    # a gap) with a view alive, writing to the frame leaves the view unchanged.
    # GH#51126
    df = DataFrame({"a": [1.5, np.nan, 3]})
    view = df[:]
    expected = df.copy()

    df.ffill(inplace=True)
    df.iloc[0, 0] = 100.5
    tm.assert_frame_equal(view, expected)


def test_isetitem():
    # [FUNCTION DOC]
    # Setting a column by position on a shallow copy (works in place) leaves the
    # unrelated columns shared with the original; later writes to the copy do not
    # change the original and the untouched column keeps sharing.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()
    df2 = df.copy(deep=False)  # Trigger a CoW
    df2.isetitem(1, np.array([-1, -2, -3]))  # This is inplace
    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged
    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_isetitem_series(dtype):
    # [FUNCTION DOC]
    # Parametrized over single-block and mixed-block layouts. Setting a column by
    # position from a Series shares memory and leaves a reference; writing to the
    # frame does not change the Series, and modifying the Series does not change a
    # frame built the same way.
    df = DataFrame({"a": [1, 2, 3], "b": np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    ser_orig = ser.copy()
    df.isetitem(0, ser)

    assert np.shares_memory(get_array(df, "a"), get_array(ser))
    assert not df._mgr._has_no_reference(0)

    # mutating dataframe doesn't update series
    df.loc[0, "a"] = 0
    tm.assert_series_equal(ser, ser_orig)

    # mutating series doesn't update dataframe
    df = DataFrame({"a": [1, 2, 3], "b": np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    df.isetitem(0, ser)

    ser.loc[0] = 0
    expected = DataFrame({"a": [7, 8, 9], "b": np.array([4, 5, 6], dtype=dtype)})
    tm.assert_frame_equal(df, expected)


def test_isetitem_frame():
    # [FUNCTION DOC]
    # Setting two columns by position from a frame shares memory with both columns
    # and keeps references; later writes to the right-hand frame do not change the
    # target.
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2})
    rhs = DataFrame({"a": [4, 5, 6], "b": 2})
    df.isetitem([0, 1], rhs)
    assert np.shares_memory(get_array(df, "a"), get_array(rhs, "a"))
    assert np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))
    assert not df._mgr._has_no_reference(0)
    expected = df.copy()
    rhs.iloc[0, 0] = 100
    rhs.iloc[0, 1] = 100
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("key", ["a", ["a"]])
def test_get(key):
    # [FUNCTION DOC]
    # Parametrized over a scalar key and a list key. get returns an object sharing
    # memory; writing detaches; frame unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_orig = df.copy()

    result = df.get(key)

    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    result.iloc[0] = 0
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis, key", [(0, 0), (1, "a")])
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_xs(axis, key, dtype):
    # [FUNCTION DOC]
    # Parametrized over (axis, key) and dtype layout. Cross-section: along columns
    # or for single-block data the result shares memory; for a mixed-block row the
    # result owns its data (no references). The result index is a different object
    # from the corresponding axis labels; writing never changes the source.
    single_block = dtype == "int64"
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    result = df.xs(key, axis=axis)

    if axis == 1 or single_block:
        assert np.shares_memory(get_array(df, "a"), get_array(result))
    else:
        assert result._mgr._has_no_reference(0)
    if axis == 0:
        assert result.index is not df.columns
    else:
        assert result.index is not df.index

    result.iloc[0] = 0
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("key, level", [("l1", 0), (2, 1)])
def test_xs_multiindex(key, level, axis):
    # [FUNCTION DOC]
    # Parametrized over axis and (key, level). Cross-section on a multi-index (frame
    # transposed first for axis one): for the outer level the first column shares
    # memory; the result has new index and columns objects; writing leaves the source
    # unchanged.
    arr = np.arange(18).reshape(6, 3)
    index = MultiIndex.from_product([["l1", "l2"], [1, 2, 3]], names=["lev1", "lev2"])
    df = DataFrame(arr, index=index, columns=list("abc"))
    if axis == 1:
        df = df.transpose().copy()
    df_orig = df.copy()

    result = df.xs(key, level=level, axis=axis)

    if level == 0:
        assert np.shares_memory(
            get_array(df, df.columns[0]), get_array(result, result.columns[0])
        )
    assert result.index is not df.index
    assert result.columns is not df.columns

    result.iloc[0, 0] = 0
    tm.assert_frame_equal(df, df_orig)


def test_update_frame():
    # [FUNCTION DOC]
    # In-place update of a frame from a frame with a view alive: the frame has the
    # expected updated values; the view keeps the old values, still sharing the
    # untouched column but not the updated one.
    df1 = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df2 = DataFrame({"b": [100.0]}, index=[1])
    df1_orig = df1.copy()
    view = df1[:]
    df1.update(df2)

    expected = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 100.0, 6.0]})
    tm.assert_frame_equal(df1, expected)
    # df1 is updated, but its view not
    tm.assert_frame_equal(view, df1_orig)
    assert np.shares_memory(get_array(df1, "a"), get_array(view, "a"))
    assert not np.shares_memory(get_array(df1, "b"), get_array(view, "b"))


def test_update_series():
    # [FUNCTION DOC]
    # Series update with a view alive: the Series has the updated values and the view
    # stays at the original values.
    ser1 = Series([1.0, 2.0, 3.0])
    ser2 = Series([100.0], index=[1])
    ser1_orig = ser1.copy()
    view = ser1[:]

    ser1.update(ser2)

    expected = Series([1.0, 100.0, 3.0])
    tm.assert_series_equal(ser1, expected)
    # ser1 is updated, but its view not
    tm.assert_series_equal(view, ser1_orig)


def test_update_chained_assignment():
    # [FUNCTION DOC]
    # update through a chained selection (column Series, then single-column frame)
    # emits the chained assignment warning and leaves the frame unchanged.
    df = DataFrame({"a": [1, 2, 3]})
    ser2 = Series([100.0], index=[1])
    df_orig = df.copy()
    with tm.raises_chained_assignment_error():
        df["a"].update(ser2)
    tm.assert_frame_equal(df, df_orig)

    with tm.raises_chained_assignment_error():
        df[["a"]].update(ser2.to_frame())
    tm.assert_frame_equal(df, df_orig)


def test_inplace_arithmetic_series():
    # [FUNCTION DOC]
    # In-place multiplication of a Series no longer updates the buffer in place (a
    # GitHub pull request changed this): the new data is a different buffer and the
    # old buffer is unchanged.
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    data = get_array(ser)
    ser *= 2
    # https://github.com/pandas-dev/pandas/pull/55745
    # changed to NOT update inplace because there is no benefit (actual
    # operation already done non-inplace). This was only for the optics
    # of updating the backing array inplace, but we no longer want to make
    # that guarantee
    assert not np.shares_memory(get_array(ser), data)
    tm.assert_numpy_array_equal(data, get_array(ser_orig))


def test_inplace_arithmetic_series_with_reference():
    # [FUNCTION DOC]
    # In-place multiplication with a view alive does not share memory with the view,
    # and the view equals the original.
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    view = ser[:]
    ser *= 2
    assert not np.shares_memory(get_array(ser), get_array(view))
    tm.assert_series_equal(ser_orig, view)


def test_transpose():
    # [FUNCTION DOC]
    # Transposing a single-dtype frame shares memory; writing to the result leaves the
    # original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()
    result = df.transpose()
    assert np.shares_memory(get_array(df, "a"), get_array(result, 0))

    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_transpose_different_dtypes():
    # [FUNCTION DOC]
    # Transposing mixed dtypes cannot share memory; writing to the result leaves
    # the original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    df_orig = df.copy()
    result = df.T

    assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))
    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_transpose_ea_single_column():
    # [FUNCTION DOC]
    # Transposing a nullable single-column frame does not share memory.
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    result = df.T

    assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))


def test_transform_frame():
    # [FUNCTION DOC]
    # transform with a function that writes into the Series it is given must not
    # modify the original frame.
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()

    def func(ser):
        # [FUNCTION DOC]
        # HELPER: nested function passed to transform; sets the first cell of the given
        # Series to a new value and returns it.
        ser.iloc[0] = 100
        return ser

    df.transform(func)
    tm.assert_frame_equal(df, df_orig)


def test_transform_series():
    # [FUNCTION DOC]
    # Series transform with a function that writes into its argument must not modify
    # the original Series.
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()

    def func(ser):
        # [FUNCTION DOC]
        # HELPER: nested function passed to transform; sets the first cell of the given
        # Series to a new value and returns it.
        ser.iloc[0] = 100
        return ser

    ser.transform(func)
    tm.assert_series_equal(ser, ser_orig)


def test_count_read_only_array():
    # [FUNCTION DOC]
    # The result of counting non-missing values per column must be writable: setting
    # a cell in it works and gives the expected Series.
    df = DataFrame({"a": [1, 2], "b": 3})
    result = df.count()
    result.iloc[0] = 100
    expected = Series([100, 2], index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_insert_series():
    # [FUNCTION DOC]
    # Inserting a Series as a column at a position shares memory and keeps a
    # reference; writing to the frame leaves the Series unchanged.
    df = DataFrame({"a": [1, 2, 3]})
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    df.insert(loc=1, value=ser, column="b")
    assert np.shares_memory(get_array(ser), get_array(df, "b"))
    assert not df._mgr._has_no_reference(1)

    df.iloc[0, 1] = 100
    tm.assert_series_equal(ser, ser_orig)


def test_eval():
    # [FUNCTION DOC]
    # Evaluating an assignment expression returns a frame sharing the existing
    # columns; writing to the result leaves the original unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()

    result = df.eval("c = a+b")
    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))

    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_eval_inplace():
    # [FUNCTION DOC]
    # In-place evaluation of an assignment expression with a view alive keeps the
    # existing column shared; writing to the frame leaves the view unchanged.
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()
    df_view = df[:]

    df.eval("c = a+b", inplace=True)
    assert np.shares_memory(get_array(df, "a"), get_array(df_view, "a"))

    df.iloc[0, 0] = 100
    tm.assert_frame_equal(df_view, df_orig)


def test_apply_modify_row():
    # [FUNCTION DOC]
    # Row-wise apply with a function that writes into each row must not modify the
    # original frame. With mixed dtypes (rows as copies) no warning is raised and the
    # frame is still unchanged.
    # Case: applying a function on each row as a Series object, where the
    # function mutates the row object (which needs to trigger CoW if row is a view)
    df = DataFrame({"A": [1, 2], "B": [3, 4]})
    df_orig = df.copy()

    def transform(row):
        # [FUNCTION DOC]
        # HELPER: nested function applied to each row; sets one entry of the row to a
        # fixed value and returns the row.
        row["B"] = 100
        return row

    df.apply(transform, axis=1)

    tm.assert_frame_equal(df, df_orig)

    # row Series is a copy
    df = DataFrame({"A": [1, 2], "B": ["b", "c"]})
    df_orig = df.copy()

    with tm.assert_produces_warning(None):
        df.apply(transform, axis=1)

    tm.assert_frame_equal(df, df_orig)


def test_reduce():
    # [FUNCTION DOC]
    # Results of reductions (sum, grouped sum, scalar and list quantile) have index or
    # columns objects that are not the same objects as the source's labels.
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})

    result = df.sum()
    assert result.index is not df.columns

    result = df.groupby([0, 0, 1]).sum()
    assert result.columns is not df.columns

    result = df.quantile(0.5)
    assert result.index is not df.columns
    result = df.quantile([0.25, 0.5, 0.75])
    assert result.columns is not df.columns


def test_diff():
    # [FUNCTION DOC]
    # The difference result for a frame and for a Series has new index (and columns)
    # objects distinct from the source's.
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})

    result = df.diff()
    assert result.index is not df.index
    assert result.columns is not df.columns

    ser = Series([1, 2, 3])
    result = ser.diff()
    assert result.index is not ser.index
