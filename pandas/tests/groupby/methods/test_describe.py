# [MODULE DOC]
# Tests for groupby describe on Series and DataFrame: multi-key grouping, as_index
# handling, tuple-valued keys, duplicate column names, non-compiled code paths and empty
# inputs. Uses the as_index, tsframe and multiindex_dataframe_random_data fixtures.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def test_apply_describe_bug(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # Verifies describe on a frame grouped by an index level of a multi-indexed frame
    # runs without error (regression check; no value assertion).
    grouped = multiindex_dataframe_random_data.groupby(level="first")
    grouped.describe()  # it works!


def test_series_describe_multikey():
    # [FUNCTION DOC]
    # Verifies describe on a Series grouped by year and month callables has summary
    # columns equal to the matching individual aggregations (mean, std, min) with names
    # ignored.
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    tm.assert_series_equal(result["mean"], grouped.mean(), check_names=False)
    tm.assert_series_equal(result["std"], grouped.std(), check_names=False)
    tm.assert_series_equal(result["min"], grouped.min(), check_names=False)


def test_series_describe_single():
    # [FUNCTION DOC]
    # Verifies that applying describe per group on a Series grouped by month gives the
    # same result as the grouped describe stacked into a series.
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    grouped = ts.groupby(lambda x: x.month)
    result = grouped.apply(lambda x: x.describe())
    expected = grouped.describe().stack()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", ["key1", ["key1", "key2"]])
def test_series_describe_as_index(as_index, keys):
    # [FUNCTION DOC]
    # Verifies (GH#49256) describe on a single grouped column returns the expected
    # eight-statistic frame, with keys as index when as_index is true or as leading
    # columns otherwise. Parametrized over one key or two keys (the second key column is
    # a copy of the first) and the as_index fixture.
    # GH#49256
    df = DataFrame(
        {
            "key1": ["one", "two", "two", "three", "two"],
            "key2": ["one", "two", "two", "three", "two"],
            "foo2": [1, 2, 4, 4, 6],
        }
    )
    gb = df.groupby(keys, as_index=as_index)["foo2"]
    result = gb.describe()
    expected = DataFrame(
        {
            "key1": ["one", "three", "two"],
            "count": [1.0, 1.0, 3.0],
            "mean": [1.0, 4.0, 4.0],
            "std": [np.nan, np.nan, 2.0],
            "min": [1.0, 4.0, 2.0],
            "25%": [1.0, 4.0, 3.0],
            "50%": [1.0, 4.0, 4.0],
            "75%": [1.0, 4.0, 5.0],
            "max": [1.0, 4.0, 6.0],
        }
    )
    if len(keys) == 2:
        expected.insert(1, "key2", expected["key1"])
    if as_index:
        expected = expected.set_index(keys)
    tm.assert_frame_equal(result, expected)


def test_frame_describe_multikey(tsframe):
    # [FUNCTION DOC]
    # Verifies describe on a frame grouped by year and month equals the column-wise
    # concatenation of each column's describe, with a two-level column index where
    # duplicate levels are removed (GH 17464).
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    desc_groups = []
    for col in tsframe:
        group = grouped[col].describe()
        # GH 17464 - Remove duplicate MultiIndex levels
        group_col = MultiIndex(
            levels=[Index([col], dtype=tsframe.columns.dtype), group.columns],
            codes=[[0] * len(group.columns), range(len(group.columns))],
        )
        group = DataFrame(group.values, columns=group_col, index=group.index)
        desc_groups.append(group)
    expected = pd.concat(desc_groups, axis=1)
    tm.assert_frame_equal(result, expected)


def test_frame_describe_tupleindex():
    # [FUNCTION DOC]
    # Verifies describe on a frame grouped by a column of tuples (GH 14848) gives a
    # frame indexed by the tuples (no tuple splitting into levels) with one column group
    # x and eight statistics.
    # GH 14848 - regression from 0.19.0 to 0.19.1
    name = "k"
    df = DataFrame(
        {
            "x": [1, 2, 3, 4, 5] * 3,
            name: [(0, 0, 1), (0, 1, 0), (1, 0, 0)] * 5,
        }
    )
    result = df.groupby(name).describe()
    expected = DataFrame(
        [[5.0, 3.0, 1.581139, 1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
        index=Index([(0, 0, 1), (0, 1, 0), (1, 0, 0)], tupleize_cols=False, name=name),
        columns=MultiIndex.from_arrays(
            [["x"] * 8, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_frame_describe_unstacked_format():
    # [FUNCTION DOC]
    # Verifies describe on a single grouped series of volumes by price (GH 4792) gives a
    # frame indexed by price with the eight statistics as columns, equal to describing
    # each price subset.
    # GH 4792
    prices = {
        Timestamp("2011-01-06 10:59:05", tz=None): 24990,
        Timestamp("2011-01-06 12:43:33", tz=None): 25499,
        Timestamp("2011-01-06 12:54:09", tz=None): 25499,
    }
    volumes = {
        Timestamp("2011-01-06 10:59:05", tz=None): 1500000000,
        Timestamp("2011-01-06 12:43:33", tz=None): 5000000000,
        Timestamp("2011-01-06 12:54:09", tz=None): 100000000,
    }
    df = DataFrame({"PRICE": prices, "VOLUME": volumes})
    result = df.groupby("PRICE").VOLUME.describe()
    data = [
        df[df.PRICE == 24990].VOLUME.describe().values.tolist(),
        df[df.PRICE == 25499].VOLUME.describe().values.tolist(),
    ]
    expected = DataFrame(
        data,
        index=Index([24990, 25499], name="PRICE"),
        columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:"
    "indexing past lexsort depth may impact performance:"
    "pandas.errors.PerformanceWarning"
)
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_describe_with_duplicate_output_column_names(as_index, keys):
    # [FUNCTION DOC]
    # Verifies (GH 35314) describe of a frame with duplicate column names (b appears
    # twice) grouped by one or two keys gives repeated statistics per duplicate column,
    # with index (or reset columns when as_index is false) as expected.
    # Parametrized over keys and as_index; a lexsort performance warning is filtered.
    # The key column a2 is dropped when only a1 is the key.
    # GH 35314
    df = DataFrame(
        {
            "a1": [99, 99, 99, 88, 88, 88],
            "a2": [99, 99, 99, 88, 88, 88],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [10, 20, 30, 40, 50, 60],
        },
        columns=["a1", "a2", "b", "b"],
        copy=False,
    )
    if keys == ["a1"]:
        df = df.drop(columns="a2")

    expected = (
        DataFrame.from_records(
            [
                ("b", "count", 3.0, 3.0),
                ("b", "mean", 5.0, 2.0),
                ("b", "std", 1.0, 1.0),
                ("b", "min", 4.0, 1.0),
                ("b", "25%", 4.5, 1.5),
                ("b", "50%", 5.0, 2.0),
                ("b", "75%", 5.5, 2.5),
                ("b", "max", 6.0, 3.0),
                ("b", "count", 3.0, 3.0),
                ("b", "mean", 5.0, 2.0),
                ("b", "std", 1.0, 1.0),
                ("b", "min", 4.0, 1.0),
                ("b", "25%", 4.5, 1.5),
                ("b", "50%", 5.0, 2.0),
                ("b", "75%", 5.5, 2.5),
                ("b", "max", 6.0, 3.0),
            ],
        )
        .set_index([0, 1])
        .T
    )
    expected.columns.names = [None, None]
    if len(keys) == 2:
        expected.index = MultiIndex(
            levels=[[88, 99], [88, 99]], codes=[[0, 1], [0, 1]], names=["a1", "a2"]
        )
    else:
        expected.index = Index([88, 99], name="a1")

    if not as_index:
        expected = expected.reset_index()

    result = df.groupby(keys, as_index=as_index).describe()

    tm.assert_frame_equal(result, expected)


def test_describe_duplicate_columns():
    # [FUNCTION DOC]
    # Verifies (GH#50806) describe with no percentiles on a frame with duplicate column
    # labels gives a multi-index column result with one block of five statistics per
    # original column, indexed by the group key.
    # GH#50806
    df = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1])
    result = gb.describe(percentiles=[])

    columns = ["count", "mean", "std", "min", "max"]
    frames = [
        DataFrame([[1.0, val, np.nan, val, val]], index=[1], columns=columns)
        for val in (0.0, 2.0, 3.0)
    ]
    expected = pd.concat(frames, axis=1)
    expected.columns = MultiIndex(
        levels=[[0, 2], columns],
        codes=[5 * [0] + 5 * [1] + 5 * [0], 3 * list(range(5))],
    )
    expected.index.names = [1]
    tm.assert_frame_equal(result, expected)


def test_describe_non_cython_paths():
    # [FUNCTION DOC]
    # Verifies describe for non-compiled paths (GH#5610): a frame with a string column
    # and a numeric column with missing data, grouped by A; only the numeric column is
    # described, the grouper is not included, with as_index true and false.
    # GH#5610 non-cython calls should not include the grouper
    # Tests for code not expected to go through cython paths.
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
        columns=["A", "B", "C"],
    )
    gb = df.groupby("A")
    expected_index = Index([1, 3], name="A")
    expected_col = MultiIndex(
        levels=[["B"], ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]],
        codes=[[0] * 8, list(range(8))],
    )
    expected = DataFrame(
        [
            [1.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0],
            [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        index=expected_index,
        columns=expected_col,
    )
    result = gb.describe()
    tm.assert_frame_equal(result, expected)

    gni = df.groupby("A", as_index=False)
    expected = expected.reset_index()
    result = gni.describe()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, float, object])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"percentiles": [0.10, 0.20, 0.30], "include": "all", "exclude": None},
        {"percentiles": [0.10, 0.20, 0.30], "include": None, "exclude": ["int"]},
        {"percentiles": [0.10, 0.20, 0.30], "include": ["int"], "exclude": None},
    ],
)
def test_groupby_empty_dataset(dtype, kwargs):
    # [FUNCTION DOC]
    # Verifies (GH#41575) describe with percentiles and include/exclude options on a
    # frame sliced to zero rows gives an empty result with the same structure as the
    # non-empty result's columns, for the frame and for the single column B.
    # Parametrized over the construction dtype (int, float, object) and three option
    # dictionaries.
    # GH#41575
    df = DataFrame([[1, 2, 3]], columns=["A", "B", "C"], dtype=dtype)
    df["B"] = df["B"].astype(int)
    df["C"] = df["C"].astype(float)

    result = df.iloc[:0].groupby("A").describe(**kwargs)
    expected = df.groupby("A").describe(**kwargs).reset_index(drop=True).iloc[:0]
    tm.assert_frame_equal(result, expected)

    result = df.iloc[:0].groupby("A").B.describe(**kwargs)
    expected = df.groupby("A").B.describe(**kwargs).reset_index(drop=True).iloc[:0]
    expected.index = Index([], dtype=df.columns.dtype)
    tm.assert_frame_equal(result, expected)
