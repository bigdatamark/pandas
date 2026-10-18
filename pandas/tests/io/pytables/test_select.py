# [MODULE DOC]
# Tests for HDFStore.select, read_hdf with where/columns, and select_as_multiple: column selection, duplicate column labels, dtype-specific conditions (timestamps, booleans, integer/float indexes, NaN handling), large selector lists, iterators and chunked reading (including GH 8014 range cases), complex boolean query expressions, scope resolution of local variables, invalid filters, string comparisons, nested special characters, float literal precision, column type comparisons, empty where arguments, large integers and non-nanosecond datetime indexes. Module marked single_cpu.
import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import PY312

import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    bdate_range,
    concat,
    date_range,
    isna,
    read_hdf,
)

from pandas.io.pytables import Term

pytestmark = [pytest.mark.single_cpu]


def test_select_columns_in_where(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 6169. A frame and a series with a two-level row index stored as tables: selecting columns via the columns argument or via a columns condition inside where returns the same single-column frame; for the series, the column condition returns the whole series with its multi-index recreated.
    # GH 6169
    # recreate multi-indexes when columns is passed
    # in the `where` argument
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["foo_name", "bar_name"],
    )

    # With a DataFrame
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        index=index,
        columns=["A", "B", "C"],
    )

    temp_hdfstore.put("df", df, format="table")
    expected = df[["A"]]

    tm.assert_frame_equal(temp_hdfstore.select("df", columns=["A"]), expected)

    tm.assert_frame_equal(temp_hdfstore.select("df", where="columns=['A']"), expected)

    # With a Series
    s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="A")
    temp_hdfstore.put("s", s, format="table")
    tm.assert_series_equal(temp_hdfstore.select("s", where="columns=['A']"), s)


def test_select_with_dups(temp_hdfstore):
    # [FUNCTION DOC]
    # A frame with duplicate column labels appended as a table: selecting everything, selecting by the full column index, and selecting a duplicated label all return matching frames (compared by blocks where applicable).
    # single dtypes
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "A", "B", "B"]
    )
    df.index = date_range("20130101 9:30", periods=10, freq="min", unit="ns")

    temp_hdfstore.append("df", df)

    result = temp_hdfstore.select("df")
    expected = df
    tm.assert_frame_equal(result, expected, by_blocks=True)

    result = temp_hdfstore.select("df", columns=df.columns)
    expected = df
    tm.assert_frame_equal(result, expected, by_blocks=True)

    result = temp_hdfstore.select("df", columns=["A"])
    expected = df.loc[:, ["A"]]
    tm.assert_frame_equal(result, expected)


def test_select_with_dups_across_dtypes(temp_hdfstore):
    # [FUNCTION DOC]
    # Duplicate float column labels combined with an integer frame that repeats label A: selecting all, by full columns, by one repeated label and by an out-of-order pair returns the expected column subsets (compared by blocks).
    df = concat(
        [
            DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=["A", "A", "B", "B"],
            ),
            DataFrame(
                np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                columns=["A", "C"],
            ),
        ],
        axis=1,
    )
    df.index = date_range("20130101 9:30", periods=10, freq="min", unit="ns")

    temp_hdfstore.append("df", df)

    result = temp_hdfstore.select("df")
    expected = df
    tm.assert_frame_equal(result, expected, by_blocks=True)

    result = temp_hdfstore.select("df", columns=df.columns)
    expected = df
    tm.assert_frame_equal(result, expected, by_blocks=True)

    expected = df.loc[:, ["A"]]
    result = temp_hdfstore.select("df", columns=["A"])
    tm.assert_frame_equal(result, expected, by_blocks=True)

    expected = df.loc[:, ["B", "A"]]
    result = temp_hdfstore.select("df", columns=["B", "A"])
    tm.assert_frame_equal(result, expected, by_blocks=True)


def test_select_with_dups_across_index_and_columns(temp_hdfstore):
    # [FUNCTION DOC]
    # The same duplicate-label frame appended twice (doubling the rows): selecting two columns returns the doubled rows with those columns, compared by blocks.
    df = concat(
        [
            DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=["A", "A", "B", "B"],
            ),
            DataFrame(
                np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                columns=["A", "C"],
            ),
        ],
        axis=1,
    )
    df.index = date_range("20130101 9:30", periods=10, freq="min", unit="ns")
    temp_hdfstore.append("df", df)
    temp_hdfstore.append("df", df)

    expected = df.loc[:, ["B", "A"]]
    expected = concat([expected, expected])
    result = temp_hdfstore.select("df", columns=["B", "A"])
    tm.assert_frame_equal(result, expected, by_blocks=True)


def test_select(temp_hdfstore):
    # [FUNCTION DOC]
    # Selecting with the columns argument, with an equivalent columns condition, with a data column condition and column subset, with all columns as data columns, and with a condition on a data column but different returned columns; each matches the expected reindexed frame.
    # select with columns=
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    temp_hdfstore.append("df", df)
    result = temp_hdfstore.select("df", columns=["A", "B"])
    expected = df.reindex(columns=["A", "B"])
    tm.assert_frame_equal(expected, result)

    # equivalently
    result = temp_hdfstore.select("df", ["columns=['A', 'B']"])
    expected = df.reindex(columns=["A", "B"])
    tm.assert_frame_equal(expected, result)

    # with a data column
    temp_hdfstore.remove("df")
    temp_hdfstore.append("df", df, data_columns=["A"])
    result = temp_hdfstore.select("df", ["A > 0"], columns=["A", "B"])
    expected = df[df.A > 0].reindex(columns=["A", "B"])
    tm.assert_frame_equal(expected, result)

    # all a data columns
    temp_hdfstore.remove("df")
    temp_hdfstore.append("df", df, data_columns=True)
    result = temp_hdfstore.select("df", ["A > 0"], columns=["A", "B"])
    expected = df[df.A > 0].reindex(columns=["A", "B"])
    tm.assert_frame_equal(expected, result)

    # with a data column, but different columns
    temp_hdfstore.remove("df")
    temp_hdfstore.append("df", df, data_columns=["A"])
    result = temp_hdfstore.select("df", ["A > 0"], columns=["C", "D"])
    expected = df[df.A > 0].reindex(columns=["C", "D"])
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_timestamp(temp_hdfstore):
    # [FUNCTION DOC]
    # GH #2637. A timestamp data column filtered with a timestamp comparison returns the matching rows.
    # with a Timestamp data column (GH #2637)
    df = DataFrame(
        {
            "ts": bdate_range("2012-01-01", periods=300, unit="ns"),
            "A": np.random.default_rng(2).standard_normal(300),
        }
    )
    temp_hdfstore.append("df", df, data_columns=["ts", "A"])

    result = temp_hdfstore.select("df", "ts>=Timestamp('2012-02-01')")
    expected = df[df.ts >= Timestamp("2012-02-01")]
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_bools(temp_hdfstore):
    # [FUNCTION DOC]
    # GH #2849. A boolean data column filtered by equality with true/false, their lowercase string forms and numeric 1/0 returns the matching rows restricted to two columns.
    # bool columns (GH #2849)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=["A", "B"])
    df["object"] = "foo"
    df.loc[4:5, "object"] = "bar"
    df["boolv"] = df["A"] > 0
    temp_hdfstore.append("df", df, data_columns=True)

    expected = df[df.boolv == True].reindex(columns=["A", "boolv"])  # noqa: E712
    for v in [True, "true", 1]:
        result = temp_hdfstore.select("df", f"boolv == {v}", columns=["A", "boolv"])
        tm.assert_frame_equal(expected, result)

    expected = df[df.boolv == False].reindex(columns=["A", "boolv"])  # noqa: E712
    for v in [False, "false", 0]:
        result = temp_hdfstore.select("df", f"boolv == {v}", columns=["A", "boolv"])
        tm.assert_frame_equal(expected, result)


def test_select_dtypes_integer_index(temp_hdfstore):
    # [FUNCTION DOC]
    # A condition on an integer index combined with a column restriction returns the first ten rows and one column.
    df = DataFrame(
        {
            "A": np.random.default_rng(2).random(20),
            "B": np.random.default_rng(2).random(20),
        }
    )
    temp_hdfstore.append("df_int", df)
    result = temp_hdfstore.select("df_int", "index<10 and columns=['A']")
    expected = df.reindex(index=list(df.index)[0:10], columns=["A"])
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_float_index(temp_hdfstore):
    # [FUNCTION DOC]
    # A frame with a float column literally named index: a condition comparing it with a float literal combined with a column restriction returns the first ten rows and one column.
    df = DataFrame(
        {
            "A": np.random.default_rng(2).random(20),
            "B": np.random.default_rng(2).random(20),
            "index": np.arange(20, dtype="f8"),
        }
    )
    temp_hdfstore.append("df_float", df)
    result = temp_hdfstore.select("df_float", "index<10.0 and columns=['A']")
    expected = df.reindex(index=list(df.index)[0:10], columns=["A"])
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_floats_without_nan(temp_hdfstore):
    # [FUNCTION DOC]
    # String column and float data columns: a numeric condition on the float data column returns the matching rows.
    df = DataFrame({"cols": range(11), "values": range(11)}, dtype="float64")
    df["cols"] = (df["cols"] + 10).apply(str)

    temp_hdfstore.append("df1", df, data_columns=True)
    result = temp_hdfstore.select("df1", where="values>2.0")
    expected = df[df["values"] > 2.0]
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_floats_with_nan(temp_hdfstore):
    # [FUNCTION DOC]
    # Same as above with the first row entirely missing and the table written without an index; the condition still returns the correct rows.
    df = DataFrame({"cols": range(11), "values": range(11)}, dtype="float64")
    df["cols"] = (df["cols"] + 10).apply(str)
    df.iloc[0] = np.nan
    expected = df[df["values"] > 2.0]

    temp_hdfstore.append("df2", df, data_columns=True, index=False)
    result = temp_hdfstore.select("df2", where="values>2.0")
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_floats_with_nan_not_first_position(temp_hdfstore):
    # [FUNCTION DOC]
    # Same as above with the missing row in second position; the condition still returns the correct rows.
    df = DataFrame({"cols": range(11), "values": range(11)}, dtype="float64")
    df["cols"] = (df["cols"] + 10).apply(str)

    df.iloc[1] = np.nan
    expected = df[df["values"] > 2.0]

    temp_hdfstore.append("df4", df, data_columns=True)
    result = temp_hdfstore.select("df4", where="values>2.0")
    tm.assert_frame_equal(expected, result)


def test_select_dtypes_comparison_with_numpy_scalar(temp_hdfstore, request):
    # [FUNCTION DOC]
    # GH 11283. A condition comparing a data column with a numpy scalar variable from the local scope returns the matching rows. Expected to fail with a value error on Python 3.12 or later (AST change; marked via the request fixture). The scalar variable is defined only to be resolved by name in the query.
    # GH 11283
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    expected = df[df["A"] > 0]

    temp_hdfstore.append("df", df, data_columns=True)
    request.applymarker(
        pytest.mark.xfail(
            PY312,
            reason="AST change in PY312",
            raises=ValueError,
        )
    )
    np_zero = np.float64(0)  # noqa: F841
    result = temp_hdfstore.select("df", where=["A>np_zero"])
    tm.assert_frame_equal(expected, result)


def test_select_with_many_inputs(temp_hdfstore):
    # [FUNCTION DOC]
    # Data columns for timestamp, float, integer and user (string) values. Asserts a timestamp condition, a combination with a small literal list of users, a combination with a long list given by a local variable, an integer column compared with a range of values (exactly 100 rows), and a timestamp column compared with an index of values (exactly 100 rows).
    df = DataFrame(
        {
            "ts": bdate_range("2012-01-01", periods=300, unit="ns"),
            "A": np.random.default_rng(2).standard_normal(300),
            "B": range(300),
            "users": ["a"] * 50
            + ["b"] * 50
            + ["c"] * 100
            + [f"a{i:03d}" for i in range(100)],
        }
    )
    temp_hdfstore.append("df", df, data_columns=["ts", "A", "B", "users"])

    # regular select
    result = temp_hdfstore.select("df", "ts>=Timestamp('2012-02-01')")
    expected = df[df.ts >= Timestamp("2012-02-01")]
    tm.assert_frame_equal(expected, result)

    # small selector
    result = temp_hdfstore.select(
        "df", "ts>=Timestamp('2012-02-01') & users=['a','b','c']"
    )
    expected = df[(df.ts >= Timestamp("2012-02-01")) & df.users.isin(["a", "b", "c"])]
    tm.assert_frame_equal(expected, result)

    # big selector along the columns
    selector = ["a", "b", "c"] + [f"a{i:03d}" for i in range(60)]
    result = temp_hdfstore.select(
        "df", "ts>=Timestamp('2012-02-01') and users=selector"
    )
    expected = df[(df.ts >= Timestamp("2012-02-01")) & df.users.isin(selector)]
    tm.assert_frame_equal(expected, result)

    selector = range(100, 200)
    result = temp_hdfstore.select("df", "B=selector")
    expected = df[df.B.isin(selector)]
    tm.assert_frame_equal(expected, result)
    assert len(result) == 100

    # big selector along the index
    selector = Index(df.ts[0:100].values)
    result = temp_hdfstore.select("df", "ts=selector")
    expected = df[df.ts.isin(selector.values)]
    tm.assert_frame_equal(expected, result)
    assert len(result) == 100


def test_select_iterator(temp_hdfstore):
    # [FUNCTION DOC]
    # Selecting with the iterator option and with a chunk size of two (five chunks) both concatenate to the full result; asserts equality.
    # single table
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    temp_hdfstore.append("df", df)

    expected = temp_hdfstore.select("df")

    results = list(temp_hdfstore.select("df", iterator=True))
    result = concat(results)
    tm.assert_frame_equal(expected, result)

    results = list(temp_hdfstore.select("df", chunksize=2))
    assert len(results) == 5
    result = concat(results)
    tm.assert_frame_equal(expected, result)

    results = list(temp_hdfstore.select("df", chunksize=2))
    result = concat(results)
    tm.assert_frame_equal(result, expected)


def test_select_iterator2(temp_h5_path):
    # [FUNCTION DOC]
    # read_hdf on a fixed-format (non-table) key with chunksize or iterator raises a type error that iterators and chunk sizes need a table.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    df.to_hdf(temp_h5_path, key="df_non_table")

    msg = "can only use an iterator or chunksize on a table"
    with pytest.raises(TypeError, match=msg):
        read_hdf(temp_h5_path, "df_non_table", chunksize=2)

    with pytest.raises(TypeError, match=msg):
        read_hdf(temp_h5_path, "df_non_table", iterator=True)


def test_select_iterator3(temp_h5_path):
    # [FUNCTION DOC]
    # Chunked read_hdf of a table gives five chunks that concatenate to the original frame and to a non-chunked read.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    df.to_hdf(temp_h5_path, key="df", format="table")

    results = list(read_hdf(temp_h5_path, "df", chunksize=2))
    result = concat(results)

    assert len(results) == 5
    tm.assert_frame_equal(result, df)
    tm.assert_frame_equal(result, read_hdf(temp_h5_path, "df"))


def test_select_iterator_multiple(temp_hdfstore):
    # [FUNCTION DOC]
    # select_as_multiple with a chunk size returns chunks whose concatenation equals the unchunked multiple select.
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    temp_hdfstore.append("df1", df1, data_columns=True)
    df2 = df1.copy().rename(columns="{}_2".format)
    df2["foo"] = "bar"
    temp_hdfstore.append("df2", df2)

    # full selection
    expected = temp_hdfstore.select_as_multiple(["df1", "df2"], selector="df1")
    results = list(
        temp_hdfstore.select_as_multiple(["df1", "df2"], selector="df1", chunksize=2)
    )
    result = concat(results)
    tm.assert_frame_equal(expected, result)


def test_select_iterator_complete_8014(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 8014. For a 100064-row second-frequency frame, selecting with no where clause and with where clauses covering the start, the end and the complete index range (no iterator) each return the whole frame.
    # GH 8014
    # using iterator and where clause
    # no iterator
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((100064, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100064, freq="s", unit="ns"),
    )
    temp_hdfstore.append("df", expected)

    beg_dt = expected.index[0]
    end_dt = expected.index[-1]

    # select w/o iteration and no where clause works
    result = temp_hdfstore.select("df")
    tm.assert_frame_equal(expected, result)

    # select w/o iterator and where clause, single term, begin
    # of range, works
    where = f"index >= '{beg_dt}'"
    result = temp_hdfstore.select("df", where=where)
    tm.assert_frame_equal(expected, result)

    # select w/o iterator and where clause, single term, end
    # of range, works
    where = f"index <= '{end_dt}'"
    result = temp_hdfstore.select("df", where=where)
    tm.assert_frame_equal(expected, result)

    # select w/o iterator and where clause, inclusive range,
    # works
    where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
    result = temp_hdfstore.select("df", where=where)
    tm.assert_frame_equal(expected, result)


def test_select_iterator_complete_8014_full_range(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 8014. Same where clauses as the complete case but using chunked iteration (chunk size 10000, a float); the concatenated chunks equal the whole frame.
    # GH 8014
    chunksize = 1e4
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((100064, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100064, freq="s", unit="ns"),
    )
    temp_hdfstore.append("df", expected)

    beg_dt = expected.index[0]
    end_dt = expected.index[-1]

    # select w/iterator and no where clause works
    results = list(temp_hdfstore.select("df", chunksize=chunksize))
    result = concat(results)
    tm.assert_frame_equal(expected, result)

    # select w/iterator and where clause, single term, begin of range
    where = f"index >= '{beg_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    tm.assert_frame_equal(expected, result)

    # select w/iterator and where clause, single term, end of range
    where = f"index <= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    tm.assert_frame_equal(expected, result)

    # select w/iterator and where clause, inclusive range
    where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    tm.assert_frame_equal(expected, result)


def test_select_iterator_non_complete_8014(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 8014. Chunked iteration with where clauses covering partial ranges (from the second row, to the second-to-last row, and between them); the concatenated chunks equal the corresponding slices.
    # GH 8014
    # using iterator and where clause
    chunksize = 1e4

    # with iterator, non complete range
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((100064, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100064, freq="s", unit="ns"),
    )
    temp_hdfstore.append("df", expected)

    beg_dt = expected.index[1]
    end_dt = expected.index[-2]

    # select w/iterator and where clause, single term, begin of range
    where = f"index >= '{beg_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    rexpected = expected[expected.index >= beg_dt]
    tm.assert_frame_equal(rexpected, result)

    # select w/iterator and where clause, single term, end of range
    where = f"index <= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    rexpected = expected[expected.index <= end_dt]
    tm.assert_frame_equal(rexpected, result)

    # select w/iterator and where clause, inclusive range
    where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
    tm.assert_frame_equal(rexpected, result)


def test_select_iterator_non_complete_8014_empty_where(temp_hdfstore):
    # [FUNCTION DOC]
    # A chunked select whose where clause matches nothing (after the last timestamp) yields no chunks at all.
    chunksize = 1e4
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((100064, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100064, freq="s", unit="ns"),
    )
    temp_hdfstore.append("df", expected)

    end_dt = expected.index[-1]

    # select w/iterator and where clause, single term, begin of range
    where = f"index > '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    assert 0 == len(results)


def test_select_iterator_many_empty_frames(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 8014. A where clause can match only the first chunk of a chunked scan; asserts that concatenated results equal the expected slices, that a range limited to the first chunk yields a single chunk, and that a clause selecting nothing yields no chunks.
    # KNOWN ISSUE: two comments say the chunk count "should be 1, is 10" and "should be []" but the assertions already require one chunk and zero chunks respectively, so the comments are stale or misleading.
    # GH 8014
    # using iterator and where clause can return many empty
    # frames.
    chunksize = 10_000

    # with iterator, range limited to the first chunk
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((100064, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100064, freq="s", unit="ns"),
    )
    temp_hdfstore.append("df", expected)

    beg_dt = expected.index[0]
    end_dt = expected.index[chunksize - 1]

    # select w/iterator and where clause, single term, begin of range
    where = f"index >= '{beg_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))
    result = concat(results)
    rexpected = expected[expected.index >= beg_dt]
    tm.assert_frame_equal(rexpected, result)

    # select w/iterator and where clause, single term, end of range
    where = f"index <= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))

    assert len(results) == 1
    result = concat(results)
    rexpected = expected[expected.index <= end_dt]
    tm.assert_frame_equal(rexpected, result)

    # select w/iterator and where clause, inclusive range
    where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))

    # should be 1, is 10
    assert len(results) == 1
    result = concat(results)
    rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
    tm.assert_frame_equal(rexpected, result)

    # select w/iterator and where clause which selects
    # *nothing*.
    #
    # To be consistent with Python idiom I suggest this should
    # return [] e.g. `for e in []: print True` never prints
    # True.

    where = f"index <= '{beg_dt}' & index >= '{end_dt}'"
    results = list(temp_hdfstore.select("df", where=where, chunksize=chunksize))

    # should be []
    assert len(results) == 0


def test_frame_select(temp_hdfstore, request):
    # [FUNCTION DOC]
    # A Term on the index referencing a local variable has that variable in its scope. Selecting with the index term and a column list, and with a single column, returns the expected slices (expected to fail with a type error on Python 3.12 or later). A comparison of the index with an integer raises a value error about a day out of range.
    # KNOWN ISSUE: a trailing block of commented-out code (selecting from a non-table store) is left in the body and is never executed.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )

    temp_hdfstore.put("frame", df, format="table")
    date = df.index[len(df) // 2]

    crit1 = Term("index>=date")
    assert crit1.env.scope["date"] == date

    crit2 = "columns=['A', 'D']"
    crit3 = "columns=A"

    request.applymarker(
        pytest.mark.xfail(
            PY312,
            reason="AST change in PY312",
            raises=TypeError,
        )
    )
    result = temp_hdfstore.select("frame", [crit1, crit2])
    expected = df.loc[date:, ["A", "D"]]
    tm.assert_frame_equal(result, expected)

    result = temp_hdfstore.select("frame", [crit3])
    expected = df.loc[:, ["A"]]
    tm.assert_frame_equal(result, expected)

    # invalid terms
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    temp_hdfstore.append("df_time", df)
    msg = "day is out of range for month: 0"
    with pytest.raises(ValueError, match=msg):
        temp_hdfstore.select("df_time", "index>0")

    # can't select if not written as table
    # temp_hdfstore['frame'] = df
    # with pytest.raises(ValueError):
    #     temp_hdfstore.select('frame', [crit1, crit2])


def test_frame_select_complex(temp_hdfstore):
    # [FUNCTION DOC]
    # Complex boolean where expressions combining index comparisons and a string data column using and, or and grouping; negation of a data column comparison works for filters but raises a not-implemented error when handed to numexpr for a condition; membership in a column list works.
    # select via complex criteria

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    df["string"] = "foo"
    df.loc[df.index[0:4], "string"] = "bar"

    temp_hdfstore.put("df", df, format="table", data_columns=["string"])

    # empty
    result = temp_hdfstore.select("df", 'index>df.index[3] & string="bar"')
    expected = df.loc[(df.index > df.index[3]) & (df.string == "bar")]
    tm.assert_frame_equal(result, expected)

    result = temp_hdfstore.select("df", 'index>df.index[3] & string="foo"')
    expected = df.loc[(df.index > df.index[3]) & (df.string == "foo")]
    tm.assert_frame_equal(result, expected)

    # or
    result = temp_hdfstore.select("df", 'index>df.index[3] | string="bar"')
    expected = df.loc[(df.index > df.index[3]) | (df.string == "bar")]
    tm.assert_frame_equal(result, expected)

    result = temp_hdfstore.select(
        "df", '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    )
    expected = df.loc[
        ((df.index > df.index[3]) & (df.index <= df.index[6])) | (df.string == "bar")
    ]
    tm.assert_frame_equal(result, expected)

    # invert
    result = temp_hdfstore.select("df", 'string!="bar"')
    expected = df.loc[df.string != "bar"]
    tm.assert_frame_equal(result, expected)

    # invert not implemented in numexpr :(
    msg = "cannot use an invert condition when passing to numexpr"
    with pytest.raises(NotImplementedError, match=msg):
        temp_hdfstore.select("df", '~(string="bar")')

    # invert ok for filters
    result = temp_hdfstore.select("df", "~(columns=['A','B'])")
    expected = df.loc[:, df.columns.difference(["A", "B"])]
    tm.assert_frame_equal(result, expected)

    # in
    result = temp_hdfstore.select("df", "index>df.index[3] & columns in ['A','B']")
    expected = df.loc[df.index > df.index[3]].reindex(columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


def test_frame_select_complex2(tmp_path):
    # [FUNCTION DOC]
    # Uses two files in the temp directory: a parameter table and a multi-index history table. Selecting history rows with conditions that reference local variables and expressions (a list, an index, an attribute, a method call and a conversion call) in both read_hdf and a store, asserting each equals the expected rows.
    pp = tmp_path / "params.hdf"
    hh = tmp_path / "hist.hdf"

    # use non-trivial selection criteria
    params = DataFrame({"A": [1, 1, 2, 2, 3]})
    params.to_hdf(pp, key="df", mode="w", format="table", data_columns=["A"])

    selection = read_hdf(pp, "df", where="A=[2,3]")
    hist = DataFrame(
        np.random.default_rng(2).standard_normal((25, 1)),
        columns=["data"],
        index=MultiIndex.from_tuples(
            [(i, j) for i in range(5) for j in range(5)], names=["l1", "l2"]
        ),
    )

    hist.to_hdf(hh, key="df", mode="w", format="table")

    expected = read_hdf(hh, "df", where="l1=[2, 3, 4]")

    # scope with list like
    l0 = selection.index.tolist()  # noqa: F841
    with HDFStore(hh) as store:
        result = store.select("df", where="l1=l0")
        tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=l0")
    tm.assert_frame_equal(result, expected)

    # index
    index = selection.index  # noqa: F841
    result = read_hdf(hh, "df", where="l1=index")
    tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=selection.index")
    tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=selection.index.tolist()")
    tm.assert_frame_equal(result, expected)

    result = read_hdf(hh, "df", where="l1=list(selection.index)")
    tm.assert_frame_equal(result, expected)

    # scope with index
    with HDFStore(hh) as store:
        result = store.select("df", where="l1=index")
        tm.assert_frame_equal(result, expected)

        result = store.select("df", where="l1=selection.index")
        tm.assert_frame_equal(result, expected)

        result = store.select("df", where="l1=selection.index.tolist()")
        tm.assert_frame_equal(result, expected)

        result = store.select("df", where="l1=list(selection.index)")
        tm.assert_frame_equal(result, expected)


def test_invalid_filtering(temp_hdfstore):
    # [FUNCTION DOC]
    # Combining more than one column filter with or or and raises a not-implemented error saying joint filters cannot be collapsed.
    # can't use more than one filter (atm)

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )

    temp_hdfstore.put("df", df, format="table")

    msg = "unable to collapse Joint Filters"
    # not implemented
    with pytest.raises(NotImplementedError, match=msg):
        temp_hdfstore.select("df", "columns=['A'] | columns=['B']")

    # in theory we could deal with this
    with pytest.raises(NotImplementedError, match=msg):
        temp_hdfstore.select("df", "columns=['A','B'] & columns=['C']")


def test_string_select(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 2973. String data column equality and inequality against a literal (including the empty string and missing values), and integer equality and inequality conditions on data columns.
    # GH 2973
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )

    # test string ==/!=
    df["x"] = "none"
    df.loc[df.index[2:7], "x"] = ""

    temp_hdfstore.append("df", df, data_columns=["x"])

    result = temp_hdfstore.select("df", "x=none")
    expected = df[df.x == "none"]
    tm.assert_frame_equal(result, expected)

    result = temp_hdfstore.select("df", "x!=none")
    expected = df[df.x != "none"]
    tm.assert_frame_equal(result, expected)

    df2 = df.copy()
    df2.loc[df2.x == "", "x"] = np.nan

    temp_hdfstore.append("df2", df2, data_columns=["x"])
    result = temp_hdfstore.select("df2", "x!=none")
    expected = df2[isna(df2.x)]
    tm.assert_frame_equal(result, expected)

    # int ==/!=
    df["int"] = 1
    df.loc[df.index[2:7], "int"] = 2

    temp_hdfstore.append("df3", df, data_columns=["int"])

    result = temp_hdfstore.select("df3", "int=2")
    expected = df[df.int == 2]
    tm.assert_frame_equal(result, expected)

    result = temp_hdfstore.select("df3", "int!=2")
    expected = df[df.int != 2]
    tm.assert_frame_equal(result, expected)


def test_select_as_multiple(temp_hdfstore):
    # [FUNCTION DOC]
    # Validates select_as_multiple: keys that are not a list or tuple raise a type error (including a list containing none); unknown table keys or an unknown selector raise key errors; a single-key list or plain string behaves like a normal select; multiple tables with the same selector or a different selector return the joined columns of the rows matching; tables with different row counts raise a value error.
    # KNOWN ISSUE: a FIXME comment records that the frequency check is disabled (check_freq false) on one assertion because the frequency intermittently differs on some builds; also the first type-error check with a none key is repeated verbatim after the tables are appended.
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B", unit="ns"),
    )
    df2 = df1.copy().rename(columns="{}_2".format)
    df2["foo"] = "bar"

    msg = "keys must be a list/tuple"
    # no tables stored
    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.select_as_multiple(None, where=["A>0", "B>0"], selector="df1")

    temp_hdfstore.append("df1", df1, data_columns=["A", "B"])
    temp_hdfstore.append("df2", df2)

    # exceptions
    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.select_as_multiple(None, where=["A>0", "B>0"], selector="df1")

    with pytest.raises(TypeError, match=msg):
        temp_hdfstore.select_as_multiple([None], where=["A>0", "B>0"], selector="df1")

    msg = "'No object named df3 in the file'"
    with pytest.raises(KeyError, match=msg):
        temp_hdfstore.select_as_multiple(
            ["df1", "df3"], where=["A>0", "B>0"], selector="df1"
        )

    with pytest.raises(KeyError, match=msg):
        temp_hdfstore.select_as_multiple(["df3"], where=["A>0", "B>0"], selector="df1")

    with pytest.raises(KeyError, match="'No object named df4 in the file'"):
        temp_hdfstore.select_as_multiple(
            ["df1", "df2"], where=["A>0", "B>0"], selector="df4"
        )

    # default select
    result = temp_hdfstore.select("df1", ["A>0", "B>0"])
    expected = temp_hdfstore.select_as_multiple(
        ["df1"], where=["A>0", "B>0"], selector="df1"
    )
    tm.assert_frame_equal(result, expected)
    expected = temp_hdfstore.select_as_multiple(
        "df1", where=["A>0", "B>0"], selector="df1"
    )
    tm.assert_frame_equal(result, expected)

    # multiple
    result = temp_hdfstore.select_as_multiple(
        ["df1", "df2"], where=["A>0", "B>0"], selector="df1"
    )
    expected = concat([df1, df2], axis=1)
    expected = expected[(expected.A > 0) & (expected.B > 0)]
    tm.assert_frame_equal(result, expected, check_freq=False)
    # FIXME: 2021-01-20 this is failing with freq None vs 4B on some builds

    # multiple (diff selector)
    result = temp_hdfstore.select_as_multiple(
        ["df1", "df2"], where="index>df2.index[4]", selector="df2"
    )
    expected = concat([df1, df2], axis=1)
    expected = expected[5:]
    tm.assert_frame_equal(result, expected)

    # test exception for diff rows
    df3 = df1.copy().head(2)
    temp_hdfstore.append("df3", df3)
    msg = "all tables must have exactly the same nrows!"
    with pytest.raises(ValueError, match=msg):
        temp_hdfstore.select_as_multiple(
            ["df1", "df3"], where=["A>0", "B>0"], selector="df1"
        )


def test_nan_selection_bug_4858(temp_hdfstore):
    # [FUNCTION DOC]
    # A table written with an index only on the string column and the first row missing: a numeric condition on the values data column returns the expected three rows.
    df = DataFrame({"cols": range(6), "values": range(6)}, dtype="float64")
    df["cols"] = (df["cols"] + 10).apply(str)
    df.iloc[0] = np.nan

    expected = DataFrame(
        {"cols": ["13.0", "14.0", "15.0"], "values": [3.0, 4.0, 5.0]},
        index=[3, 4, 5],
    )

    # write w/o the index on that particular column
    temp_hdfstore.append("df", df, data_columns=True, index=["cols"])
    result = temp_hdfstore.select("df", where="values>2.0")
    tm.assert_frame_equal(result, expected)


def test_query_with_nested_special_character(temp_hdfstore):
    # [FUNCTION DOC]
    # A string data value containing an ampersand can be matched with a quoted equality condition.
    df = DataFrame(
        {
            "a": ["a", "a", "c", "b", "test & test", "c", "b", "e"],
            "b": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    expected = df[df.a == "test & test"]
    temp_hdfstore.append("test", df, format="table", data_columns=True)
    result = temp_hdfstore.select("test", 'a = "test & test"')
    tm.assert_frame_equal(expected, result)


def test_query_long_float_literal(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 14241. Float literals with four decimals compared against values of magnitude one billion: a lower bound below all values returns nothing, a bound between values returns the last two rows, and an exact equality matches one row.
    # GH 14241
    df = DataFrame({"A": [1000000000.0009, 1000000000.0011, 1000000000.0015]})

    temp_hdfstore.append("test", df, format="table", data_columns=True)

    cutoff = 1000000000.0006
    result = temp_hdfstore.select("test", f"A < {cutoff:.4f}")
    assert result.empty

    cutoff = 1000000000.0010
    result = temp_hdfstore.select("test", f"A > {cutoff:.4f}")
    expected = df.loc[[1, 2], :]
    tm.assert_frame_equal(expected, result)

    exact = 1000000000.0011
    result = temp_hdfstore.select("test", f"A == {exact:.4f}")
    expected = df.loc[[1], :]
    tm.assert_frame_equal(expected, result)


def test_query_compare_column_type(temp_hdfstore):
    # [FUNCTION DOC]
    # GH 15492. For each comparison operator, comparing the string column with non-string values (float, boolean, timestamp, timedelta) raises a type error that the value cannot be compared to a string column; comparing integer, float and date columns with a non-convertible string raises a value error (date-specific message for the date column); comparing them with convertible strings returns the expected rows depending on the operator. A timestamp variable comparison with the date column also works.
    # GH 15492
    df = DataFrame(
        {
            "date": ["2014-01-01", "2014-01-02"],
            "real_date": date_range("2014-01-01", periods=2, unit="ns"),
            "float": [1.1, 1.2],
            "int": [1, 2],
        },
        columns=["date", "real_date", "float", "int"],
    )

    temp_hdfstore.append("test", df, format="table", data_columns=True)

    ts = Timestamp("2014-01-01")  # noqa: F841
    result = temp_hdfstore.select("test", where="real_date > ts")
    expected = df.loc[[1], :]
    tm.assert_frame_equal(expected, result)

    for op in ["<", ">", "=="]:
        # non strings to string column always fail
        for v in [2.1, True, Timestamp("2014-01-01"), pd.Timedelta(1, "s")]:
            query = f"date {op} v"
            msg = f"Cannot compare {v} of type {type(v)} to string column"
            with pytest.raises(TypeError, match=msg):
                temp_hdfstore.select("test", where=query)

        # strings to other columns must be convertible to type
        v = "a"
        for col in ["int", "float", "real_date"]:
            query = f"{col} {op} v"
            if col == "real_date":
                msg = 'Given date string "a" not likely a datetime'
            else:
                msg = "could not convert string to"
            with pytest.raises(ValueError, match=msg):
                temp_hdfstore.select("test", where=query)

        for v, col in zip(["1", "1.1", "2014-01-01"], ["int", "float", "real_date"]):
            query = f"{col} {op} v"
            result = temp_hdfstore.select("test", where=query)

            if op == "==":
                expected = df.loc[[0], :]
            elif op == ">":
                expected = df.loc[[1], :]
            else:
                expected = df.loc[[], :]
            tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("where", ["", (), (None,), [], [None]])
def test_select_empty_where(temp_hdfstore, where):
    # [FUNCTION DOC]
    # GH26610. Parametrized over empty strings, empty tuples/lists and ones containing none: read_hdf with such a where argument on a table returns the entire frame.
    # GH26610

    df = DataFrame([1, 2, 3])
    temp_hdfstore.put("df", df, "t")
    result = read_hdf(temp_hdfstore, "df", where=where)
    tm.assert_frame_equal(result, df)


def test_select_large_integer(temp_hdfstore):
    # [FUNCTION DOC]
    # A table with a large negative 64-bit integer data column written without index: an equality condition on the large literal returns the matching row and its value equals the original.
    df = DataFrame(
        zip(
            ["a", "b", "c", "d"],
            [-9223372036854775801, -9223372036854775802, -9223372036854775803, 123],
        ),
        columns=["x", "y"],
    )
    temp_hdfstore.append("data", df, data_columns=True, index=False)
    result = (
        temp_hdfstore.select("data", where="y==-9223372036854775801").get("y").get(0)
    )
    expected = df["y"][0]

    assert expected == result


@pytest.mark.parametrize("unit", ["us", "ns", "ms", "s"])
def test_select_where_datetime_index_with_non_ns_resolution(temp_hdfstore, unit):
    # [FUNCTION DOC]
    # GH#64310. Parametrized over datetime units (us, ns, ms, s): a where clause with an inclusive string date range on an hourly index of that unit returns the expected rows.
    # GH#64310
    idx = date_range("2020-01-01", periods=10, freq="h", unit=unit)
    df = DataFrame({"value": range(10)}, index=idx)

    temp_hdfstore.append("df", df)

    result = temp_hdfstore.select(
        "df", where="index >= '2020-01-02' & index <= '2020-01-03'"
    )
    expected = df[(df.index >= "2020-01-02") & (df.index <= "2020-01-03")]
    tm.assert_frame_equal(result, expected)
