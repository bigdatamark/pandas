"""test with the .transform"""
# [MODULE DOC]
# Tests for the groupby transform machinery (DataFrameGroupBy and SeriesGroupBy transform),
# covering user-defined functions, string aliases, cython-backed cumulative/shift/fill
# operations, null group keys, categorical keys, dtype preservation and index alignment.
# Relies on fixtures from the groupby test conftest (frames, series, function-name lists)
# and on the shared helper that builds default arguments for groupby methods.
# No import-time side effects beyond imports.

import numpy as np
import pytest

from pandas._libs import lib
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import ensure_platform_int

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


def assert_fp_equal(a, b):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the transform tests.
    # Takes two array-like numeric objects a and b and asserts that every element-wise absolute
    # difference is below one trillionth. Returns nothing; fails with an assertion error otherwise.
    assert (np.abs(a - b) < 1e-12).all()


def test_transform():
    # [FUNCTION DOC]
    # Checks basic transform on a shuffled-index Series grouped by a function of the label
    # (the group-wise multiplied sum is looked up at a label), that DataFrame transform preserves
    # input order for an unsorted index (compared to a sorted-index computation), that demeaning
    # via transform equals demeaning via apply, that grouping by a monthly Grouper on a
    # business-day frame works, and that the builtin max passed to transform on a frame returns
    # only the non-key column.
    data = Series(np.arange(9) // 3, index=np.arange(9))

    index = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)

    grouped = data.groupby(lambda x: x // 3)

    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12

    # GH 8046
    # make sure that we preserve the input order

    df = DataFrame(
        np.arange(6, dtype="int64").reshape(3, 2), columns=["a", "b"], index=[0, 2, 1]
    )
    key = [0, 0, 1]
    expected = (
        df.sort_index()
        .groupby(key)
        .transform(lambda x: x - x.mean())
        .groupby(key)
        .mean()
    )
    result = df.groupby(key).transform(lambda x: x - x.mean()).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    def demean(arr):
        # [FUNCTION DOC]
        # HELPER: subtracts the column-wise mean from the input array; used as the transform function.
        return arr - arr.mean(axis=0)

    people = DataFrame(
        np.random.default_rng(2).standard_normal((5, 5)),
        columns=["a", "b", "c", "d", "e"],
        index=["Joe", "Steve", "Wes", "Jim", "Travis"],
    )
    key = ["one", "two", "one", "two", "one"]
    result = people.groupby(key).transform(demean).groupby(key).mean()
    expected = people.groupby(key, group_keys=False).apply(demean).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    # GH 8430
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    g = df.groupby(pd.Grouper(freq="ME"))
    g.transform(lambda x: x - 1)

    # GH 9700
    df = DataFrame({"a": range(5, 10), "b": range(5)})
    result = df.groupby("a").transform(max)
    expected = DataFrame({"b": range(5)})
    tm.assert_frame_equal(result, expected)


def test_transform_fast():
    # [FUNCTION DOC]
    # Verifies that transforming a grouped Series with a numpy mean function and with the string
    # alias mean both equal the group means repeated per row (same index, name preserved).
    df = DataFrame(
        {
            "id": np.arange(10) / 3,
            "val": np.random.default_rng(2).standard_normal(10),
        }
    )

    grp = df.groupby("id")["val"]

    values = np.repeat(grp.mean().values, ensure_platform_int(grp.count().values))
    expected = Series(values, index=df.index, name="val")

    result = grp.transform(np.mean)
    tm.assert_series_equal(result, expected)

    result = grp.transform("mean")
    tm.assert_series_equal(result, expected)


def test_transform_fast2():
    # [FUNCTION DOC]
    # Verifies that the "first" transform on a frame with datetime, float and int columns keeps
    # dtypes (datetime stays datetime64 nanosecond) and drops the key column, and also with a
    # column selection on the groupby.
    # GH 12737
    df = DataFrame(
        {
            "grouping": [0, 1, 1, 3],
            "f": [1.1, 2.1, 3.1, 4.5],
            "d": date_range("2014-1-1", "2014-1-4", unit="ns"),
            "i": [1, 2, 3, 4],
        },
        columns=["grouping", "f", "i", "d"],
    )
    result = df.groupby("grouping").transform("first")

    dates = Index(
        [
            Timestamp("2014-1-1"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-4"),
        ],
        dtype="M8[ns]",
    )
    expected = DataFrame(
        {"f": [1.1, 2.1, 2.1, 4.5], "d": dates, "i": [1, 2, 2, 4]},
        columns=["f", "i", "d"],
    )
    tm.assert_frame_equal(result, expected)

    # selection
    result = df.groupby("grouping")[["f", "i"]].transform("first")
    expected = expected[["f", "i"]]
    tm.assert_frame_equal(result, expected)


def test_transform_fast3():
    # [FUNCTION DOC]
    # Verifies that "first" transform works with duplicate column names: result equals the frame
    # without the grouping column.
    # dup columns
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["g", "a", "a"])
    result = df.groupby("g").transform("first")
    expected = df.drop("g", axis=1)
    tm.assert_frame_equal(result, expected)


def test_transform_broadcast(tsframe, ts):
    # [FUNCTION DOC]
    # Using the tsframe and ts fixtures, verifies that transforming by month with a mean function
    # broadcasts group means back over each group's rows, keeps the original index, for both
    # Series and DataFrame.
    grouped = ts.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)

    tm.assert_index_equal(result.index, ts.index)
    for _, gp in grouped:
        assert_fp_equal(result.reindex(gp.index), gp.mean())

    grouped = tsframe.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    for _, gp in grouped:
        agged = gp.mean(axis=0)
        res = result.reindex(gp.index)
        for col in tsframe:
            assert_fp_equal(res[col], agged[col])


def test_transform_axis_ts(tsframe):
    # [FUNCTION DOC]
    # Checks that subtracting a weekday-grouped mean transform equals applying demeaning per
    # group, for both a monotonic and a non-monotonic (two rows swapped) index.
    # make sure that we are setting the axes correctly
    # in the presence of a non-monotonic indexer
    # GH12713

    base = tsframe.iloc[0:5]
    r = len(base.index)
    c = len(base.columns)
    tso = DataFrame(
        np.random.default_rng(2).standard_normal((r, c)),
        index=base.index,
        columns=base.columns,
        dtype="float64",
    )
    # monotonic
    ts = tso
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)

    # non-monotonic
    ts = tso.iloc[[1, 0, *list(range(2, len(base)))]]
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)


def test_transform_dtype():
    # [FUNCTION DOC]
    # Verifies that a mean transform on an integer frame yields float results with the expected
    # frame shape (dtype of the output is float).
    # GH 9807
    # Check transform dtype output is preserved
    df = DataFrame([[1, 3], [2, 3]])
    result = df.groupby(1).transform("mean")
    expected = DataFrame([[1.5], [1.5]])
    tm.assert_frame_equal(result, expected)


def test_transform_bug():
    # [FUNCTION DOC]
    # Regression test: transform with a ranking function on a datetime-keyed grouping returns a
    # float Series named like the selected column.
    # GH 5712
    # transforming on a datetime column
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    result = df.groupby("A")["B"].transform(lambda x: x.rank(ascending=False))
    expected = Series(np.arange(5, 0, step=-1), name="B", dtype="float64")
    tm.assert_series_equal(result, expected)


def test_transform_numeric_to_boolean():
    # [FUNCTION DOC]
    # Verifies that a transform function returning a constant boolean produces a boolean Series
    # for both float and integer value columns.
    # GH 16875
    # inconsistency in transforming boolean values
    expected = Series([True, True], name="A")

    df = DataFrame({"A": [1.1, 2.2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)

    df = DataFrame({"A": [1, 2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_timedelta():
    # [FUNCTION DOC]
    # Verifies that date arithmetic inside transform (max minus min) can return timedelta values:
    # once with a base time added and subtracted, once directly; both yield timedelta zero Series.
    # GH 15429
    # transforming a datetime to timedelta
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    expected = Series(
        Timestamp("20130101") - Timestamp("20130101"), index=range(5), name="A"
    )

    # this does date math without changing result type in transform
    base_time = df["A"][0]
    result = (
        df.groupby("A")["A"].transform(lambda x: x.max() - x.min() + base_time)
        - base_time
    )
    tm.assert_series_equal(result, expected)

    # this does date math and causes the transform to return timedelta
    result = df.groupby("A")["A"].transform(lambda x: x.max() - x.min())
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_numeric():
    # [FUNCTION DOC]
    # Verifies that transforming a datetime column into day-of-week arithmetic gives a float Series
    # (mean subtraction) or an int32 Series (minimum subtraction).
    # GH 10972
    # convert dt to float
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.mean()
    )

    expected = Series([-0.5, 0.5], name="b")
    tm.assert_series_equal(result, expected)

    # convert dt to int
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.min()
    )

    expected = Series([0, 1], dtype=np.int32, name="b")
    tm.assert_series_equal(result, expected)


def test_transform_casting():
    # [FUNCTION DOC]
    # Regression test: a transform that applies diff on a datetime column grouped by a string key
    # must give timedelta dtype for both Series and DataFrame selections.
    # 13046
    times = [
        "13:43:27",
        "14:26:19",
        "14:29:01",
        "18:39:34",
        "18:40:18",
        "18:44:30",
        "18:46:00",
        "18:52:15",
        "18:59:59",
        "19:17:48",
        "19:21:38",
    ]
    df = DataFrame(
        {
            "A": [f"B-{i}" for i in range(11)],
            "ID3": np.take(
                ["a", "b", "c", "d", "e"], [0, 1, 2, 1, 3, 1, 1, 1, 4, 1, 1]
            ),
            "DATETIME": pd.to_datetime([f"2014-10-08 {time}" for time in times]),
        },
        index=pd.RangeIndex(11, name="idx"),
    )

    result = df.groupby("ID3")["DATETIME"].transform(lambda x: x.diff())
    assert lib.is_np_dtype(result.dtype, "m")

    result = df[["ID3", "DATETIME"]].groupby("ID3").transform(lambda x: x.diff())
    assert lib.is_np_dtype(result.DATETIME.dtype, "m")


def test_transform_multiple(ts):
    # [FUNCTION DOC]
    # Smoke test: transform by multiple function-based keys with a lambda and with a numpy mean
    # works without error (no result assertions).
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    grouped.transform(lambda x: x * 2)
    grouped.transform(np.mean)


def test_dispatch_transform(tsframe):
    # [FUNCTION DOC]
    # Verifies that the groupby forward-fill method gives the same result as transforming with a
    # forward-fill function, on a frame with every fifth row retained and the rest missing.
    df = tsframe[::5].reindex(tsframe.index)

    grouped = df.groupby(lambda x: x.month)

    filled = grouped.ffill()
    fillit = lambda x: x.ffill()
    expected = df.groupby(lambda x: x.month).transform(fillit)
    tm.assert_frame_equal(filled, expected)


def test_transform_transformation_func(transformation_func):
    # [FUNCTION DOC]
    # Parametrized over the transformation_func fixture (names of transformation methods).
    # Builds a frame with a string key and float column with NaNs on a daily index. Computes the
    # expected value by applying a mock operation to each group in group order, concatenating and
    # sorting by index (then restoring the index to drop frequency). Special cases cumcount and
    # ngroup, which return a Series built from a range or a running counter. Compares with the
    # result of transform by name; Series comparison for cumcount/ngroup, frame comparison else.
    # GH 30918
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "bar", "bar", "baz"],
            "B": [1, 2, np.nan, 3, 3, np.nan, 4],
        },
        index=date_range("2020-01-01", "2020-01-07"),
    )
    if transformation_func == "cumcount":
        test_op = lambda x: x.transform("cumcount")
        mock_op = lambda x: Series(range(len(x)), x.index)
    elif transformation_func == "ngroup":
        test_op = lambda x: x.transform("ngroup")
        counter = -1

        def mock_op(x):
            # [FUNCTION DOC]
            # HELPER: used by test_transform_transformation_func for ngroup. Increments the
            # enclosing counter (starting at minus one) on each call and returns a Series with
            # the counter value over the index of the passed group, mimicking group numbering.
            nonlocal counter
            counter += 1
            return Series(counter, index=x.index)

    else:
        test_op = lambda x: x.transform(transformation_func)
        mock_op = lambda x: getattr(x, transformation_func)()

    result = test_op(df.groupby("A"))

    # pass the group in same order as iterating `for ... in df.groupby(...)`
    # but reorder to match df's index since this is a transform
    groups = [df[["B"]].iloc[4:6], df[["B"]].iloc[6:], df[["B"]].iloc[:4]]
    expected = concat([mock_op(g) for g in groups]).sort_index()
    # sort_index does not preserve the freq
    expected = expected.set_axis(df.index)

    if transformation_func in ("cumcount", "ngroup"):
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)


def test_transform_select_columns(df):
    # [FUNCTION DOC]
    # Verifies that transform on a selected list of columns with a mean function equals the same
    # transform applied to the pre-selected sub-frame grouped by the key column.
    f = lambda x: x.mean()
    result = df.groupby("A")[["C", "D"]].transform(f)

    selection = df[["C", "D"]]
    expected = selection.groupby(df["A"]).transform(f)

    tm.assert_frame_equal(result, expected)


def test_transform_nuisance_raises(df, using_infer_string):
    # [FUNCTION DOC]
    # Verifies that transforming with a numpy mean over a non-numeric duplicated column "B"
    # raises a type error, from both SeriesGroupBy and DataFrameGroupBy; the message depends on
    # the using_infer_string fixture (reduction-on-string message when True, conversion message
    # otherwise).
    # case that goes through _transform_item_by_item

    df.columns = ["A", "B", "B", "D"]

    # this also tests orderings in transform between
    # series/frame to make sure it's consistent
    grouped = df.groupby("A")

    gbc = grouped["B"]
    msg = "Could not convert"
    if using_infer_string:
        msg = "Cannot perform reduction 'mean' with string dtype"
    with pytest.raises(TypeError, match=msg):
        gbc.transform(lambda x: np.mean(x))

    with pytest.raises(TypeError, match=msg):
        df.groupby("A").transform(lambda x: np.mean(x))


def test_transform_function_aliases(df):
    # [FUNCTION DOC]
    # Verifies that string "mean" with numeric_only equals the numpy mean transform on numeric
    # columns, and likewise for a single-column Series transform.
    result = df.groupby("A").transform("mean", numeric_only=True)
    expected = df.groupby("A")[["C", "D"]].transform(np.mean)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A")["C"].transform("mean")
    expected = df.groupby("A")["C"].transform(np.mean)
    tm.assert_series_equal(result, expected)


def test_series_fast_transform_date():
    # [FUNCTION DOC]
    # Regression test: "first" transform on a datetime column with a NaN group key gives NaT for
    # the NaN-key row and datetime64 nanosecond dtype overall.
    # GH 13191
    df = DataFrame(
        {
            "grouping": [np.nan, 1, 1, 3],
            "d": date_range("2014-1-1", "2014-1-4", unit="ns"),
        }
    )
    result = df.groupby("grouping")["d"].transform("first")
    dates = [
        pd.NaT,
        Timestamp("2014-1-2"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-4"),
    ]
    expected = Series(dates, name="d", dtype="M8[ns]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [lambda x: np.nansum(x), sum])
def test_transform_length(func):
    # [FUNCTION DOC]
    # Parametrized over a numpy nansum lambda and builtin sum. Verifies the transform result
    # length equals the input length for frame and Series selections; with builtin sum the group
    # containing NaN gives NaN, with nansum the result is the group sum.
    # GH 9697
    df = DataFrame({"col1": [1, 1, 2, 2], "col2": [1, 2, 3, np.nan]})
    if func is sum:
        expected = Series([3.0, 3.0, np.nan, np.nan])
    else:
        expected = Series([3.0] * 4)

    results = [
        df.groupby("col1").transform(func)["col2"],
        df.groupby("col1")["col2"].transform(func),
    ]
    for result in results:
        tm.assert_series_equal(result, expected, check_names=False)


def test_transform_coercion():
    # [FUNCTION DOC]
    # Verifies that a mean transform and a lambda computing the numpy mean along axis 0 give equal
    # frames (no coercion through assignment).
    # 14457
    # when we are transforming be sure to not coerce
    # via assignment
    df = DataFrame({"A": ["a", "a", "b", "b"], "B": [0, 1, 3, 4]})
    g = df.groupby("A")

    expected = g.transform(np.mean)
    result = g.transform(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_int(using_infer_string):
    # [FUNCTION DOC]
    # Covers upcasting in item-by-item transforms: standardizing float columns gives NaN for a
    # constant column; with an integer frame containing a string column the full-frame transform
    # raises a type error (message depends on using_infer_string) while column-selected transform
    # works; integer data requiring float conversion is standardized per group; and an identity
    # arithmetic transform of integer columns yields floats (no downcast).
    # GH 3740, make sure that we might upcast on item-by-item transform

    # floats
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": Series(1, dtype="float64"),
            "C": Series([1, 2, 3, 1, 2, 3], dtype="float64"),
            "D": "foo",
        }
    )
    with np.errstate(all="ignore"):
        result = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    expected = DataFrame(
        {"B": np.nan, "C": Series([-1, 0, 1, -1, 0, 1], dtype="float64")}
    )
    tm.assert_frame_equal(result, expected)

    # int case
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": 1,
            "C": [1, 2, 3, 1, 2, 3],
            "D": "foo",
        }
    )
    msg = "Could not convert"
    if using_infer_string:
        msg = "Cannot perform reduction 'mean' with string dtype"
    with np.errstate(all="ignore"):
        with pytest.raises(TypeError, match=msg):
            df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
        result = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    expected = DataFrame({"B": np.nan, "C": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    tm.assert_frame_equal(result, expected)

    # int that needs float conversion
    s = Series([2, 3, 4, 10, 5, -1])
    df = DataFrame({"A": [1, 1, 1, 2, 2, 2], "B": 1, "C": s, "D": "foo"})
    with np.errstate(all="ignore"):
        with pytest.raises(TypeError, match=msg):
            df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
        result = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    s1 = s.iloc[0:3]
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = s.iloc[3:6]
    s2 = (s2 - s2.mean()) / s2.std()
    expected = DataFrame({"B": np.nan, "C": concat([s1, s2])})
    tm.assert_frame_equal(result, expected)

    # int doesn't get downcasted
    result = df.groupby("A")[["B", "C"]].transform(lambda x: x * 2 / 2)
    expected = DataFrame({"B": 1.0, "C": [2.0, 3.0, 4.0, 10.0, 5.0, -1.0]})
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_nan_group():
    # [FUNCTION DOC]
    # Regression test: grouping by a column with NaN and transforming "a" with builtin max yields
    # NaN for the NaN-key row and group maxima elsewhere.
    # GH 9941
    df = DataFrame({"a": range(10), "b": [1, 1, 2, 3, np.nan, 4, 4, 5, 5, 5]})
    result = df.groupby(df.b)["a"].transform(max)
    expected = Series([1.0, 1.0, 2.0, 3.0, np.nan, 6.0, 6.0, 9.0, 9.0, 9.0], name="a")
    tm.assert_series_equal(result, expected)


def test_transform_mixed_type():
    # [FUNCTION DOC]
    # Verifies apply on a mixed-type frame with a multi-index: a nested function mutates a group
    # by adding a doubled column and returns its first row; result float dtype is preserved and
    # each group result equals the function applied to the group without the key column.
    index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "c": np.tile(["a", "b", "c"], 2),
            "v": np.arange(1.0, 7.0),
        },
        index=index,
    )

    def f(group):
        # [FUNCTION DOC]
        # HELPER: used by test_transform_mixed_type. Mutates the passed group by adding a column
        # named g equal to twice column d, and returns only the first row of the group.
        group["g"] = group["d"] * 2
        return group[:1]

    grouped = df.groupby("c")
    result = grouped.apply(f)

    assert result["d"].dtype == np.float64

    # this is by definition a mutating operation!
    for key, group in grouped:
        res = f(group.drop(columns="c"))
        tm.assert_frame_equal(res, result.loc[key])


@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),
        ("cumsum", (), lambda x: x.cumsum()),
        ("shift", (-1,), lambda x: x.shift(-1)),
        ("shift", (1,), lambda x: x.shift()),
    ],
)
def test_cython_transform_series(op, args, targop):
    # [FUNCTION DOC]
    # Parametrized over (op, args, targop) for cumprod, cumsum, shift backward by one and shift forward by one. For a
    # random Series, with and without missing values, grouped by random float labels, verifies
    # transform with the operation name plus args, the equivalent callable, and the direct
    # groupby method all agree.
    # GH 4095
    s = Series(np.random.default_rng(2).standard_normal(1000))
    s_missing = s.copy()
    s_missing.iloc[2:10] = np.nan
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)

    # series
    for data in [s, s_missing]:
        # print(data.head())
        expected = data.groupby(labels).transform(targop)

        tm.assert_series_equal(expected, data.groupby(labels).transform(op, *args))
        tm.assert_series_equal(expected, getattr(data.groupby(labels), op)(*args))


@pytest.mark.parametrize("op", ["cumprod", "cumsum"])
@pytest.mark.parametrize(
    "input, exp",
    [
        # When everything is NaN
        ({"key": ["b"] * 10, "value": np.nan}, Series([np.nan] * 10, name="value")),
        # When there is a single NaN
        (
            {"key": ["b"] * 10 + ["a"] * 2, "value": [3] * 3 + [np.nan] + [3] * 8},
            {
                ("cumprod", False): [3.0, 9.0, 27.0] + [np.nan] * 7 + [3.0, 9.0],
                ("cumprod", True): [
                    3.0,
                    9.0,
                    27.0,
                    np.nan,
                    81.0,
                    243.0,
                    729.0,
                    2187.0,
                    6561.0,
                    19683.0,
                    3.0,
                    9.0,
                ],
                ("cumsum", False): [3.0, 6.0, 9.0] + [np.nan] * 7 + [3.0, 6.0],
                ("cumsum", True): [
                    3.0,
                    6.0,
                    9.0,
                    np.nan,
                    12.0,
                    15.0,
                    18.0,
                    21.0,
                    24.0,
                    27.0,
                    3.0,
                    6.0,
                ],
            },
        ),
    ],
)
def test_groupby_cum_skipna(op, skipna, input, exp):
    # [FUNCTION DOC]
    # Parametrized over cumprod/cumsum, the skipna fixture, and input frames (all-NaN values,
    # or a single NaN). Verifies that the cumulative transform respects skipna: with skipna true
    # the NaN position stays NaN while accumulation continues; with false everything after NaN
    # in the group is NaN. Expected is looked up in a dictionary keyed by operation and skipna
    # when given a dictionary, else used directly.
    df = DataFrame(input)
    result = df.groupby("key")["value"].transform(op, skipna=skipna)
    if isinstance(exp, dict):
        expected = exp[(op, skipna)]
    else:
        expected = exp
    expected = Series(expected, name="value")
    tm.assert_series_equal(expected, result)


@pytest.fixture
def frame():
    # [FUNCTION DOC]
    # HELPER: fixture. Builds a 10-row frame with float, float-with-missing, int, datetime,
    # timedelta, string, string-with-missing and categorical columns, used by the cython
    # transform tests.
    floating = Series(np.random.default_rng(2).standard_normal(10))
    floating_missing = floating.copy()
    floating_missing.iloc[2:7] = np.nan
    strings = list("abcde") * 2
    strings_missing = strings[:]
    strings_missing[5] = np.nan

    df = DataFrame(
        {
            "float": floating,
            "float_missing": floating_missing,
            "int": [1, 1, 1, 1, 2] * 2,
            "datetime": date_range("1990-1-1", periods=10),
            "timedelta": pd.timedelta_range(1, freq="s", periods=10),
            "string": strings,
            "string_missing": strings_missing,
            "cat": Categorical(strings),
        },
    )
    return df


@pytest.fixture
def frame_mi(frame):
    # [FUNCTION DOC]
    # HELPER: fixture. Takes the frame fixture and assigns it a two-level product MultiIndex
    # (5 by 2). Mutates and returns the same frame.
    frame.index = MultiIndex.from_product([range(5), range(2)])
    return frame


@pytest.mark.slow
@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),
        ("cumsum", (), lambda x: x.cumsum()),
        ("shift", (-1,), lambda x: x.shift(-1)),
        ("shift", (1,), lambda x: x.shift()),
    ],
)
@pytest.mark.parametrize("df_fix", ["frame", "frame_mi"])
@pytest.mark.parametrize(
    "gb_target",
    [
        {"by": np.random.default_rng(2).integers(0, 50, size=10).astype(float)},
        {"level": 0},
        {"by": "string"},
        pytest.param({"by": "string_missing"}, marks=pytest.mark.xfail),
        {"by": ["int", "string"]},
    ],
)
def test_cython_transform_frame(request, op, args, targop, df_fix, gb_target):
    # [FUNCTION DOC]
    # Slow test, parametrized over operation (cumprod, cumsum, shift forward/back), frame fixture
    # (flat or MultiIndex), and grouping target (random float labels, level, string column,
    # string with missing column which is marked xfail, multi-column). Computes the expected frame
    # via apply of the equivalent callable (concatenating int and float columns separately when
    # dtype promotion is a concern) and compares to transform by name and to the direct method.
    # For shift, string columns' missing values are normalized to NaN.
    df = request.getfixturevalue(df_fix)
    gb = df.groupby(group_keys=False, **gb_target)

    if op != "shift" and "int" not in gb_target:
        # numeric apply fastpath promotes dtype so have
        # to apply separately and concat
        i = gb[["int"]].apply(targop)
        f = gb[["float", "float_missing"]].apply(targop)
        expected = concat([f, i], axis=1)
    else:
        expected = gb.apply(targop)

    expected = expected.sort_index(axis=1)
    if op == "shift":
        expected["string_missing"] = expected["string_missing"].fillna(np.nan)
        by = gb_target.get("by")
        if not isinstance(by, (str, list)) or (by != "string" and "string" not in by):
            expected["string"] = expected["string"].fillna(np.nan)

    result = gb[expected.columns].transform(op, *args).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)
    result = getattr(gb[expected.columns], op)(*args).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),
        ("cumsum", (), lambda x: x.cumsum()),
        ("shift", (-1,), lambda x: x.shift(-1)),
        ("shift", (1,), lambda x: x.shift()),
    ],
)
@pytest.mark.parametrize("df_fix", ["frame", "frame_mi"])
@pytest.mark.parametrize(
    "gb_target",
    [
        {"by": np.random.default_rng(2).integers(0, 50, size=10).astype(float)},
        {"level": 0},
        {"by": "string"},
        # TODO: create xfail condition given other params
        # {"by": 'string_missing'},
        {"by": ["int", "string"]},
    ],
)
@pytest.mark.parametrize(
    "column",
    [
        "float",
        "float_missing",
        "int",
        "datetime",
        "timedelta",
        "string",
        "string_missing",
    ],
)
def test_cython_transform_frame_column(
    request, op, args, targop, df_fix, gb_target, column
):
    # [FUNCTION DOC]
    # Slow test, parametrized like the frame variant plus a single column. For columns that are
    # not float/int/float-missing (except shift and timedelta cumsum) asserts a type error is
    # raised by both transform and the direct method with one of several unsupported-dtype
    # messages; otherwise compares transform and direct-method results against apply of the
    # callable (with NaN normalization for string columns).
    df = request.getfixturevalue(df_fix)
    gb = df.groupby(group_keys=False, **gb_target)
    c = column
    if (
        c not in ["float", "int", "float_missing"]
        and op != "shift"
        and not (c == "timedelta" and op == "cumsum")
    ):
        msg = "|".join(
            [
                "does not support .* operations",
                "does not support operation",
                ".* is not supported for object dtype",
                "is not implemented for this dtype",
                ".* is not supported for str dtype",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            gb[c].transform(op)
        with pytest.raises(TypeError, match=msg):
            getattr(gb[c], op)()
    else:
        expected = gb[c].apply(targop)
        expected.name = c
        if c in ["string_missing", "string"]:
            expected = expected.fillna(np.nan)

        res = gb[c].transform(op, *args)
        tm.assert_series_equal(expected, res)
        res2 = getattr(gb[c], op)(*args)
        tm.assert_series_equal(expected, res2)


@pytest.mark.parametrize(
    "cols,expected",
    [
        ("a", Series([1, 1, 1], name="a")),
        (
            ["a", "c"],
            DataFrame({"a": [1, 1, 1], "c": [1, 1, 1]}),
        ),
    ],
)
@pytest.mark.parametrize("agg_func", ["count", "rank", "size"])
def test_transform_numeric_ret(cols, expected, agg_func):
    # [FUNCTION DOC]
    # Parametrized over column selection (single column or list), expected output, and
    # aggregation (count, rank, size). Verifies transform on a datetime column returns numeric
    # results: ranks become floats, and size with several columns yields a Series rather than a
    # frame.
    # GH#19200 and GH#27469
    df = DataFrame(
        {"a": date_range("2018-01-01", periods=3), "b": range(3), "c": range(7, 10)}
    )
    result = df.groupby("b")[cols].transform(agg_func)

    if agg_func == "rank":
        expected = expected.astype("float")
    elif agg_func == "size" and cols == ["a", "c"]:
        # transform("size") returns a Series
        expected = expected["a"].rename(None)
    tm.assert_equal(result, expected)


def test_transform_ffill():
    # [FUNCTION DOC]
    # Verifies that "ffill" transform fills NaN within each key for both frame and Series
    # selections.
    # GH 24211
    data = [["a", 0.0], ["a", float("nan")], ["b", 1.0], ["b", float("nan")]]
    df = DataFrame(data, columns=["key", "values"])
    result = df.groupby("key").transform("ffill")
    expected = DataFrame({"values": [0.0, 0.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    result = df.groupby("key")["values"].transform("ffill")
    expected = Series([0.0, 0.0, 1.0, 1.0], name="values")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("mix_groupings", [True, False])
@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize("val1,val2", [("foo", "bar"), (1, 2), (1.0, 2.0)])
@pytest.mark.parametrize(
    "fill_method,limit,exp_vals",
    [
        (
            "ffill",
            None,
            [np.nan, np.nan, "val1", "val1", "val1", "val2", "val2", "val2"],
        ),
        ("ffill", 1, [np.nan, np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan]),
        (
            "bfill",
            None,
            ["val1", "val1", "val1", "val2", "val2", "val2", np.nan, np.nan],
        ),
        ("bfill", 1, [np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan, np.nan]),
    ],
)
def test_group_fill_methods(
    mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals
):
    # [FUNCTION DOC]
    # Parametrized over mixed vs sequential group layout, Series vs DataFrame, three value types
    # (string, int, float), and fill method (forward or backward) with a limit. Builds values with
    # NaNs, substitutes placeholders in the expected list with the actual values, interleaves
    # if groupings are mixed, then compares the groupby fill method result with the expected
    # Series or frame.
    vals = [np.nan, np.nan, val1, np.nan, np.nan, val2, np.nan, np.nan]
    _exp_vals = list(exp_vals)
    # Overwrite placeholder values
    for index, exp_val in enumerate(_exp_vals):
        if exp_val == "val1":
            _exp_vals[index] = val1
        elif exp_val == "val2":
            _exp_vals[index] = val2

    # Need to modify values and expectations depending on the
    # Series / DataFrame that we ultimately want to generate
    if mix_groupings:  # ['a', 'b', 'a, 'b', ...]
        keys = ["a", "b"] * len(vals)

        def interweave(list_obj):
            # [FUNCTION DOC]
            # HELPER: used by test_group_fill_methods. Returns a new list where every element of the
            # input is repeated twice consecutively.
            temp = []
            for x in list_obj:
                temp.extend([x, x])

            return temp

        _exp_vals = interweave(_exp_vals)
        vals = interweave(vals)
    else:  # ['a', 'a', 'a', ... 'b', 'b', 'b']
        keys = ["a"] * len(vals) + ["b"] * len(vals)
        _exp_vals = _exp_vals * 2
        vals = vals * 2

    df = DataFrame({"key": keys, "val": vals})
    if as_series:
        result = getattr(df.groupby("key")["val"], fill_method)(limit=limit)
        exp = Series(_exp_vals, name="val")
        tm.assert_series_equal(result, exp)
    else:
        result = getattr(df.groupby("key"), fill_method)(limit=limit)
        exp = DataFrame({"val": _exp_vals})
        tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("fill_method", ["ffill", "bfill"])
def test_pad_stable_sorting(fill_method):
    # [FUNCTION DOC]
    # Regression test: forward and backward fill by a constant key must keep original order
    # (stable sort); the result equals the frame without the key column.
    # GH 21207
    x = [0] * 20
    y = [np.nan] * 10 + [1] * 10

    if fill_method == "bfill":
        y = y[::-1]

    df = DataFrame({"x": x, "y": y})
    expected = df.drop("x", axis=1)

    result = getattr(df.groupby("x"), fill_method)()

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "freq",
    [
        None,
        pytest.param(
            "D",
            marks=pytest.mark.xfail(
                reason="GH#23918 before method uses freq in vectorized approach"
            ),
        ),
    ],
)
@pytest.mark.parametrize("periods", [1, -1])
def test_pct_change(frame_or_series, freq, periods):
    # [FUNCTION DOC]
    # Parametrized over freq (None, or "D" marked xfail because the vectorized method uses freq),
    # periods (1, -1), and frame_or_series. Verifies pct_change by group equals value divided by
    # the grouped shifted value minus one, for DataFrame (converted to one-column frame) and
    # Series groupby.
    # GH 21200, 21621, 30463
    vals = [3, np.nan, np.nan, np.nan, 1, 2, 4, 10, np.nan, 4]
    keys = ["a", "b"]
    key_v = np.repeat(keys, len(vals))
    df = DataFrame({"key": key_v, "vals": vals * 2})

    df_g = df
    grp = df_g.groupby(df.key)

    expected = grp["vals"].obj / grp["vals"].shift(periods) - 1

    gb = df.groupby("key")

    if frame_or_series is Series:
        gb = gb["vals"]
    else:
        expected = expected.to_frame("vals")

    result = gb.pct_change(periods=periods, freq=freq)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_status",
    [
        ("ffill", ["shrt", "shrt", "lng", np.nan, "shrt", "ntrl", "ntrl"]),
        ("bfill", ["shrt", "lng", "lng", "shrt", "shrt", "ntrl", np.nan]),
    ],
)
def test_ffill_bfill_non_unique_multilevel(func, expected_status):
    # [FUNCTION DOC]
    # Parametrized over ffill/bfill and expected status columns. Verifies that fill within groups
    # works with a non-unique MultiIndex (date, symbol), preserving the index.
    # GH 19437
    date = pd.to_datetime(
        [
            "2018-01-01",
            "2018-01-01",
            "2018-01-01",
            "2018-01-01",
            "2018-01-02",
            "2018-01-01",
            "2018-01-02",
        ]
    )
    symbol = ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL", "TSLA", "TSLA"]
    status = ["shrt", np.nan, "lng", np.nan, "shrt", "ntrl", np.nan]

    df = DataFrame({"date": date, "symbol": symbol, "status": status})
    df = df.set_index(["date", "symbol"])
    result = getattr(df.groupby("symbol")["status"], func)()

    index = MultiIndex.from_tuples(
        tuples=list(zip(*[date, symbol], strict=True)), names=["date", "symbol"]
    )
    expected = Series(expected_status, index=index, name="status")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [np.any, np.all])
def test_any_all_np_func(func):
    # [FUNCTION DOC]
    # Parametrized over numpy any and all. Verifies that transform with these numpy functions on
    # a boolean column with a NaN key gives NaN for the row whose key is NaN and True elsewhere.
    # GH 20653
    df = DataFrame(
        [["foo", True], [np.nan, True], ["foo", True]], columns=["key", "val"]
    )

    exp = Series([True, np.nan, True], name="val")

    res = df.groupby("key")["val"].transform(func)
    tm.assert_series_equal(res, exp)


def test_groupby_transform_rename():
    # [FUNCTION DOC]
    # Verifies that a transform function which renames output columns (appending a suffix on frames)
    # does not alter the resulting names: the frame result keeps the original column name and the
    # selected Series result matches.
    # https://github.com/pandas-dev/pandas/issues/23461
    def demean_rename(x):
        # [FUNCTION DOC]
        # HELPER: used by test_groupby_transform_rename. Subtracts the mean from the object; if the
        # object is a Series returns it, otherwise renames each column to add a demeaned suffix.
        result = x - x.mean()

        if isinstance(x, Series):
            return result

        result = result.rename(columns={c: f"{c}_demeaned" for c in result.columns})

        return result

    df = DataFrame({"group": list("ababa"), "value": [1, 1, 1, 2, 2]})
    expected = DataFrame({"value": [-1.0 / 3, -0.5, -1.0 / 3, 0.5, 2.0 / 3]})

    result = df.groupby("group").transform(demean_rename)
    tm.assert_frame_equal(result, expected)
    result_single = df.groupby("group").value.transform(demean_rename)
    tm.assert_series_equal(result_single, expected["value"])


@pytest.mark.parametrize("func", [min, max, np.min, np.max, "first", "last"])
def test_groupby_transform_timezone_column(func):
    # [FUNCTION DOC]
    # Parametrized over min, max, numpy min/max, "first", "last". Verifies that transform of a
    # timezone-aware datetime column keeps the timezone and equals the original value.
    # GH 24198
    ts = pd.to_datetime("now", utc=True).tz_convert("Asia/Singapore")
    result = DataFrame({"end_time": [ts], "id": [1]})
    result["max_end_time"] = result.groupby("id").end_time.transform(func)
    expected = DataFrame([[ts, 1, ts]], columns=["end_time", "id", "max_end_time"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"]),
        ("idxmax", ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"]),
    ],
)
def test_groupby_transform_with_datetimes(func, values):
    # [FUNCTION DOC]
    # Parametrized over idxmin/idxmax with expected dates. Verifies that transform with these
    # functions on a daily-indexed frame grouped by ISO week gives nanosecond datetime values
    # at the expected index labels.
    # GH 15306
    dates = date_range("1/1/2011", periods=10, freq="D", unit="ns")

    stocks = DataFrame({"price": np.arange(10.0)}, index=dates)
    stocks["week_id"] = dates.isocalendar().week

    result = stocks.groupby(stocks["week_id"])["price"].transform(func)

    expected = Series(
        data=pd.to_datetime(values).as_unit("ns"), index=dates, name="price"
    )

    tm.assert_series_equal(result, expected)


def test_groupby_transform_dtype():
    # [FUNCTION DOC]
    # Verifies that transform with a function mapping floats to strings returns string
    # (object-like) Series for plain Series, groupby, and object-dtype input.
    # GH 22243
    df = DataFrame({"a": [1], "val": [1.35]})

    result = df["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    expected1 = Series(["+1.35"], name="val")
    tm.assert_series_equal(result, expected1)

    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    tm.assert_series_equal(result, expected1)

    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+({y})"))
    expected2 = Series(["+(1.35)"], name="val")
    tm.assert_series_equal(result, expected2)

    df["val"] = df["val"].astype(object)
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    tm.assert_series_equal(result, expected1)


def test_transform_absent_categories(all_numeric_accumulations):
    # [FUNCTION DOC]
    # Uses the all_numeric_accumulations fixture. With a categorical key that has more
    # categories than rows (unobserved), the accumulation by group returns the original Series.
    # GH 16771
    # cython transforms with more groups than rows
    x_vals = [1]
    x_cats = range(2)
    y = [1]
    df = DataFrame({"x": Categorical(x_vals, x_cats), "y": y})
    result = getattr(df.y.groupby(df.x, observed=False), all_numeric_accumulations)()
    expected = df.y
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["ffill", "bfill", "shift"])
@pytest.mark.parametrize("key, val", [("level", 0), ("by", Series([0]))])
def test_ffill_not_in_axis(func, key, val):
    # [FUNCTION DOC]
    # Parametrized over ffill/bfill/shift and grouping via level or a by Series. Verifies that
    # filling or shifting a frame whose only column is NaN and is not a grouping column returns
    # the same frame.
    # GH 21521
    df = DataFrame([[np.nan]])
    result = getattr(df.groupby(**{key: val}), func)()
    expected = df

    tm.assert_frame_equal(result, expected)


def test_transform_invalid_name_raises():
    # [FUNCTION DOC]
    # Verifies that transform with a nonexistent function name raises a value error saying it is
    # not a valid function name, as does a name for an existing but non-transform method
    # (aggregate), for both DataFrameGroupBy and SeriesGroupBy.
    # GH#27486
    df = DataFrame({"a": [0, 1, 1, 2]})
    g = df.groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("some_arbitrary_name")

    # method exists on the object, but is not a valid transformation/agg
    assert hasattr(g, "aggregate")  # make sure the method exists
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("aggregate")

    # Test SeriesGroupBy
    g = df["a"].groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("some_arbitrary_name")


def test_transform_agg_by_name(request, reduction_func, frame_or_series):
    # [FUNCTION DOC]
    # Uses the reduction_func and frame_or_series fixtures. Verifies that transform by reduction
    # name gives a result with the same index as the input (and same columns for frames except
    # size and ngroup) and that values are constant within each group. For corrwith on Series
    # asserts the groupby has no such method and returns; for frames, corrwith emits the
    # Pandas4Warning deprecation, which is asserted.
    func = reduction_func

    obj = DataFrame(
        {"a": [0, 0, 0, 0, 1, 1, 1, 1], "b": range(8)},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    if frame_or_series is Series:
        obj = obj["a"]

    g = obj.groupby(np.repeat([0, 1], 4))

    if func == "corrwith" and isinstance(obj, Series):  # GH#32293
        # TODO: implement SeriesGroupBy.corrwith
        assert not hasattr(g, func)
        return

    args = get_groupby_method_args(reduction_func, obj)
    if func == "corrwith":
        warn = Pandas4Warning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    with tm.assert_produces_warning(warn, match=msg):
        result = g.transform(func, *args)

    # this is the *definition* of a transformation
    tm.assert_index_equal(result.index, obj.index)

    if func not in ("ngroup", "size") and obj.ndim == 2:
        # size/ngroup return a Series, unlike other transforms
        tm.assert_index_equal(result.columns, obj.columns)

    # verify that values were broadcasted across each group
    assert len(set(DataFrame(result).iloc[-4:, -1])) == 1


def test_transform_lambda_with_datetimetz():
    # [FUNCTION DOC]
    # Regression test: a transform lambda that localizes each group's datetime to the timezone
    # named in the group key returns timezone-aware datetimes with per-row zones preserved.
    # GH 27496
    df = DataFrame(
        {
            "time": [
                Timestamp("2010-07-15 03:14:45"),
                Timestamp("2010-11-19 18:47:06"),
            ],
            "timezone": ["Etc/GMT+4", "US/Eastern"],
        }
    )
    result = df.groupby(["timezone"])["time"].transform(
        lambda x: x.dt.tz_localize(x.name)
    )
    expected = Series(
        [
            Timestamp("2010-07-15 03:14:45", tz="Etc/GMT+4"),
            Timestamp("2010-11-19 18:47:06", tz="US/Eastern"),
        ],
        name="time",
    )
    tm.assert_series_equal(result, expected)


def test_transform_fastpath_raises():
    # [FUNCTION DOC]
    # Verifies that when the fast path for a transform raises (the function fails on frames) the
    # transform still succeeds via the slow path (function applied per column), producing the
    # expected doubled values. Also directly asserts the fast path raises not-implemented.
    # Uses private groupby internals (object with exclusions, grouper iterator, path definition).
    # GH#29631 case where fastpath defined in groupby.generic _choose_path
    #  raises, but slow_path does not

    df = DataFrame({"A": [1, 1, 2, 2], "B": [1, -1, 1, 2]})
    gb = df.groupby("A")

    def func(grp):
        # [FUNCTION DOC]
        # HELPER: used by test_transform_fastpath_raises. Raises a not-implemented error when given a
        # two-dimensional object, otherwise returns the input multiplied by two.
        # we want a function such that func(frame) fails but func.apply(frame)
        #  works
        if grp.ndim == 2:
            # Ensure that fast_path fails
            raise NotImplementedError("Don't cross the streams")
        return grp * 2

    # Check that the fastpath raises, see _transform_general
    obj = gb._obj_with_exclusions
    gen = gb._grouper.get_iterator(obj)
    fast_path, slow_path = gb._define_paths(func)
    _, group = next(gen)

    with pytest.raises(NotImplementedError, match="Don't cross the streams"):
        fast_path(group)

    result = gb.transform(func)

    expected = DataFrame([2, -2, 2, 4], columns=["B"])
    tm.assert_frame_equal(result, expected)


def test_transform_lambda_indexing():
    # [FUNCTION DOC]
    # Verifies that transform with a function selecting the last row of each group on a sorted
    # MultiIndex frame returns the last value repeated for each member of the group.
    # GH 7883
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "flux", "foo", "flux"],
            "B": ["one", "one", "two", "three", "two", "six", "five", "three"],
            "C": range(8),
            "D": range(8),
            "E": range(8),
        }
    )
    df = df.set_index(["A", "B"])
    df = df.sort_index()
    result = df.groupby(level="A").transform(lambda x: x.iloc[-1])
    expected = DataFrame(
        {
            "C": [3, 3, 7, 7, 4, 4, 4, 4],
            "D": [3, 3, 7, 7, 4, 4, 4, 4],
            "E": [3, 3, 7, 7, 4, 4, 4, 4],
        },
        index=MultiIndex.from_tuples(
            [
                ("bar", "one"),
                ("bar", "three"),
                ("flux", "six"),
                ("flux", "three"),
                ("foo", "five"),
                ("foo", "one"),
                ("foo", "two"),
                ("foo", "two"),
            ],
            names=["A", "B"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_categorical_and_not_categorical_key(observed):
    # [FUNCTION DOC]
    # Parametrized over the observed fixture. Verifies that grouping by a categorical and a
    # non-categorical key with a sum transform returns the input shape (no unobserved categories
    # added), for both frame and Series, equal to the non-categorical equivalent.
    # Checks that groupby-transform, when grouping by both a categorical
    # and a non-categorical key, doesn't try to expand the output to include
    # non-observed categories but instead matches the input shape.
    # GH 32494
    df_with_categorical = DataFrame(
        {
            "A": Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "B": [1, 2, 3],
            "C": ["a", "b", "a"],
        }
    )
    df_without_categorical = DataFrame(
        {"A": ["a", "b", "a"], "B": [1, 2, 3], "C": ["a", "b", "a"]}
    )

    # DataFrame case
    result = df_with_categorical.groupby(["A", "C"], observed=observed).transform("sum")
    expected = df_without_categorical.groupby(["A", "C"]).transform("sum")
    tm.assert_frame_equal(result, expected)
    expected_explicit = DataFrame({"B": [4, 2, 4]})
    tm.assert_frame_equal(result, expected_explicit)

    # Series case
    gb = df_with_categorical.groupby(["A", "C"], observed=observed)
    gbp = gb["B"]
    result = gbp.transform("sum")
    expected = df_without_categorical.groupby(["A", "C"])["B"].transform("sum")
    tm.assert_series_equal(result, expected)
    expected_explicit = Series([4, 2, 4], name="B")
    tm.assert_series_equal(result, expected_explicit)


def test_string_rank_grouping():
    # [FUNCTION DOC]
    # Verifies that the rank transform by group on a two-column frame gives float ranks.
    # GH 19354
    df = DataFrame({"A": [1, 1, 2], "B": [1, 2, 3]})
    result = df.groupby("A").transform("rank")
    expected = DataFrame({"B": [1.0, 2.0, 1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_cumcount():
    # [FUNCTION DOC]
    # Verifies that the groupby cumcount method and the "cumcount" transform name both give the
    # within-group counter Series.
    # GH 27472
    df = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    grp = df.groupby(np.repeat([0, 1], 3))

    result = grp.cumcount()
    expected = Series([0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(result, expected)

    result = grp.transform("cumcount")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", [["A1"], ["A1", "A2"]])
def test_null_group_lambda_self(sort, dropna, keys):
    # [FUNCTION DOC]
    # Parametrized over key lists, sort and dropna fixtures. Verifies that an identity transform
    # on a column with groups containing NaN keys returns the original values when dropna is false,
    # and NaN for rows whose group key is null when dropna is true.
    # GH 17093
    size = 50
    nulls1 = np.random.default_rng(2).choice([False, True], size)
    nulls2 = np.random.default_rng(2).choice([False, True], size)
    # Whether a group contains a null value or not
    nulls_grouper = nulls1 if len(keys) == 1 else nulls1 | nulls2

    a1 = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a1[nulls1] = np.nan
    a2 = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a2[nulls2] = np.nan
    values = np.random.default_rng(2).integers(0, 5, size=a1.shape)
    df = DataFrame({"A1": a1, "A2": a2, "B": values})

    expected_values = values
    if dropna and nulls_grouper.any():
        expected_values = expected_values.astype(float)
        expected_values[nulls_grouper] = np.nan
    expected = DataFrame(expected_values, columns=["B"])

    gb = df.groupby(keys, dropna=dropna, sort=sort)
    result = gb[["B"]].transform(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_null_group_str_reducer(request, dropna, reduction_func):
    # [FUNCTION DOC]
    # Uses dropna and reduction_func fixtures. Verifies transform by reducer name with NaN group
    # keys: the expected value is computed per group (with explicit values for first, last, nth,
    # size, corrwith); with dropna the null-key rows become NaN with float (or object for any/all)
    # dtype. corrwith is marked as an expected failure ("incorrectly raises").
    # GH 17093
    if reduction_func == "corrwith":
        msg = "incorrectly raises"
        request.applymarker(pytest.mark.xfail(reason=msg))

    index = [1, 2, 3, 4]  # test transform preserves non-standard index
    df = DataFrame({"A": [1, 1, np.nan, np.nan], "B": [1, 2, 2, 3]}, index=index)
    gb = df.groupby("A", dropna=dropna)

    args = get_groupby_method_args(reduction_func, df)

    # Manually handle reducers that don't fit the generic pattern
    # Set expected with dropna=False, then replace if necessary
    if reduction_func == "first":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "last":
        expected = DataFrame({"B": [2, 2, 3, 3]}, index=index)
    elif reduction_func == "nth":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = DataFrame({"B": [1.0, 1.0, 1.0, 1.0]}, index=index)
    else:
        expected_gb = df.groupby("A", dropna=False)
        buffer = []
        for idx, group in expected_gb:
            res = getattr(group["B"], reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer).to_frame("B")
    if dropna:
        dtype = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        if expected.ndim == 2:
            expected.iloc[[2, 3], 0] = np.nan
        else:
            expected.iloc[[2, 3]] = np.nan

    result = gb.transform(reduction_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_transformer(dropna, transformation_func):
    # [FUNCTION DOC]
    # Uses dropna and transformation_func fixtures. Verifies transform by name for a frame with a
    # NaN key: expected is built from applying the method to each group (cumcount and ngroup are
    # built by hand and returned as Series), plus a NaN row for the null key when dropna.
    # GH 17093
    df = DataFrame({"A": [1, 1, np.nan], "B": [1, 2, 2]}, index=[1, 2, 3])
    args = get_groupby_method_args(transformation_func, df)
    gb = df.groupby("A", dropna=dropna)

    buffer = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            # DataFrame has no cumcount method
            res = DataFrame({"B": range(len(group))}, index=group.index)
        elif transformation_func == "ngroup":
            res = DataFrame(len(group) * [k], index=group.index, columns=["B"])
        else:
            res = getattr(group[["B"]], transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype = object if transformation_func in ("any", "all") else None
        buffer.append(DataFrame([[np.nan]], index=[3], dtype=dtype, columns=["B"]))
    expected = concat(buffer)

    if transformation_func in ("cumcount", "ngroup"):
        # ngroup/cumcount always returns a Series as it counts the groups, not values
        expected = expected["B"].rename(None)

    result = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_reducer_series(request, dropna, reduction_func):
    # [FUNCTION DOC]
    # Series variant of the null-group reducer test with the dropna and reduction_func fixtures.
    # For corrwith, asserts SeriesGroupBy lacks the method and returns. Otherwise compares
    # transform by reducer name with a manually built expected Series, setting null-key rows to
    # NaN when dropna.
    # GH 17093
    index = [1, 2, 3, 4]  # test transform preserves non-standard index
    ser = Series([1, 2, 2, 3], index=index)
    gb = ser.groupby([1, 1, np.nan, np.nan], dropna=dropna)

    if reduction_func == "corrwith":
        # corrwith not implemented for SeriesGroupBy
        assert not hasattr(gb, reduction_func)
        return

    args = get_groupby_method_args(reduction_func, ser)

    # Manually handle reducers that don't fit the generic pattern
    # Set expected with dropna=False, then replace if necessary
    if reduction_func == "first":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "last":
        expected = Series([2, 2, 3, 3], index=index)
    elif reduction_func == "nth":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = Series([1, 1, 2, 2], index=index)
    else:
        expected_gb = ser.groupby([1, 1, np.nan, np.nan], dropna=False)
        buffer = []
        for idx, group in expected_gb:
            res = getattr(group, reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer)
    if dropna:
        dtype = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        expected.iloc[[2, 3]] = np.nan

    result = gb.transform(reduction_func, *args)
    tm.assert_series_equal(result, expected)


def test_null_group_str_transformer_series(dropna, transformation_func):
    # [FUNCTION DOC]
    # Series variant of the null-group transformer test (dropna and transformation_func
    # fixtures): compares transform by name with the concatenation of per-group results, adding
    # a NaN entry for the null key when dropna.
    # GH 17093
    ser = Series([1, 2, 2], index=[1, 2, 3])
    args = get_groupby_method_args(transformation_func, ser)
    gb = ser.groupby([1, 1, np.nan], dropna=dropna)

    buffer = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            # Series has no cumcount method
            res = Series(range(len(group)), index=group.index)
        elif transformation_func == "ngroup":
            res = Series(k, index=group.index)
        else:
            res = getattr(group, transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype = object if transformation_func in ("any", "all") else None
        buffer.append(Series([np.nan], index=[3], dtype=dtype))
    expected = concat(buffer)
    result = gb.transform(transformation_func, *args)

    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_values",
    [
        (Series.sort_values, [5, 4, 3, 2, 1]),
        (lambda x: x.head(1), [5.0, np.nan, 3, 2, np.nan]),
    ],
)
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
@pytest.mark.parametrize("keys_in_index", [True, False])
def test_transform_aligns(func, frame_or_series, expected_values, keys, keys_in_index):
    # [FUNCTION DOC]
    # Parametrized over function (sort_values, or a head-one lambda), frame_or_series, expected
    # values, key lists and whether keys are in the index. Verifies that transform aligns the
    # output to the input's index (values dropped by head become NaN).
    # GH#45648 - transform should align with the input's index
    df = DataFrame({"a1": [1, 1, 3, 2, 2], "b": [5, 4, 3, 2, 1]})
    if "a2" in keys:
        df["a2"] = df["a1"]
    if keys_in_index:
        df = df.set_index(keys, append=True)

    gb = df.groupby(keys)
    if frame_or_series is Series:
        gb = gb["b"]

    result = gb.transform(func)
    expected = DataFrame({"b": expected_values}, index=df.index)
    if frame_or_series is Series:
        expected = expected["b"]
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("keys", ["A", ["A", "B"]])
def test_as_index_no_change(keys, df, groupby_func):
    # [FUNCTION DOC]
    # Parametrized over key(s) and the groupby_func fixture. Verifies that the as_index setting
    # has no effect on transform output: results with true and false are equal. For single key "A"
    # drops string column "B". corrwith is expected to emit the Pandas4Warning deprecation.
    # GH#49834 - as_index should have no impact on DataFrameGroupBy.transform
    if keys == "A":
        # Column B is string dtype; will fail on some ops
        df = df.drop(columns="B")
    args = get_groupby_method_args(groupby_func, df)
    gb_as_index_true = df.groupby(keys, as_index=True)
    gb_as_index_false = df.groupby(keys, as_index=False)
    if groupby_func == "corrwith":
        warn = Pandas4Warning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    with tm.assert_produces_warning(warn, match=msg):
        result = gb_as_index_true.transform(groupby_func, *args)
    with tm.assert_produces_warning(warn, match=msg):
        expected = gb_as_index_false.transform(groupby_func, *args)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("how", ["idxmax", "idxmin"])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_transform_args(how, skipna, numeric_only):
    # [FUNCTION DOC]
    # Parametrized over idxmax/idxmin, skipna and numeric_only. Verifies positional arguments
    # given to transform are passed on to the method: with skipna true the positional call equals
    # the keyword call; with false a value error is expected stating that the method with
    # skipna false found an NA value.
    # GH#55268 - ensure *args are passed through when calling transform
    df = DataFrame({"a": [1, 1, 1, 2], "b": [3.0, 4.0, np.nan, 6.0], "c": list("abcd")})
    gb = df.groupby("a")
    if skipna:
        result = gb.transform(how, skipna, numeric_only)
        expected = gb.transform(how, skipna=skipna, numeric_only=numeric_only)
        tm.assert_frame_equal(result, expected)
    else:
        msg = f"{how} with skipna=False encountered an NA value"
        with pytest.raises(ValueError, match=msg):
            gb.transform(how, skipna, numeric_only)


def test_transform_sum_one_column_no_matching_labels():
    # [FUNCTION DOC]
    # Verifies a sum transform by an external Series key whose label matches nothing in a
    # one-column frame returns the frame unchanged.
    df = DataFrame({"X": [1.0]})
    series = Series(["Y"])
    result = df.groupby(series, as_index=False).transform("sum")
    expected = DataFrame({"X": [1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels():
    # [FUNCTION DOC]
    # Verifies a sum transform by an external Series of unique labels returns the input values
    # unchanged.
    df = DataFrame({"X": [1.0, -93204, 4935]})
    series = Series(["A", "B", "C"])

    result = df.groupby(series, as_index=False).transform("sum")
    expected = DataFrame({"X": [1.0, -93204, 4935]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels():
    # [FUNCTION DOC]
    # Verifies a sum transform by an external Series with repeated labels broadcasts the group
    # sum to each member.
    df = DataFrame({"X": [1.0, -93204, 4935]})
    series = Series(["A", "B", "A"])

    result = df.groupby(series, as_index=False).transform("sum")
    expected = DataFrame({"X": [4936.0, -93204, 4936.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_missing_labels():
    # [FUNCTION DOC]
    # Verifies a sum transform when the external key Series is shorter than the frame: rows without
    # a key label get NaN.
    df = DataFrame({"X": [1.0, -93204, 4935]})
    series = Series(["A", "C"])

    result = df.groupby(series, as_index=False).transform("sum")
    expected = DataFrame({"X": [1.0, -93204, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels_and_missing_labels():
    # [FUNCTION DOC]
    # Verifies a sum transform when the key Series has fewer entries than rows and repeated labels:
    # labeled rows get the group sum, the unlabeled row gets NaN.
    df = DataFrame({"X": [1.0, -93204, 4935]})
    series = Series(["A", "A"])

    result = df.groupby(series, as_index=False).transform("sum")
    expected = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_min_one_unobserved_category_no_type_coercion(dtype):
    # [FUNCTION DOC]
    # Parametrized over int32 and float32. Regression test: min transform with an unobserved
    # categorical key keeps the original dtype of the value column rather than coercing.
    # GH#58084
    df = DataFrame({"A": Categorical([1, 1, 2], categories=[1, 2, 3]), "B": [3, 4, 5]})
    df["B"] = df["B"].astype(dtype)
    gb = df.groupby("A", observed=False)
    result = gb.transform("min")

    expected = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_min_all_empty_data_no_type_coercion():
    # [FUNCTION DOC]
    # Regression test: min transform on an empty frame with a categorical key (all categories
    # unobserved) keeps the int32 dtype of the value column.
    # GH#58084
    df = DataFrame(
        {
            "X": Categorical(
                [],
                categories=[1, "randomcat", 100],
            ),
            "Y": [],
        }
    )
    df["Y"] = df["Y"].astype("int32")

    gb = df.groupby("X", observed=False)
    result = gb.transform("min")

    expected = DataFrame({"Y": []}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_min_one_dim_no_type_coercion():
    # [FUNCTION DOC]
    # Regression test: min transform when grouping by a categorical object (with unobserved
    # categories) with a single int32 column keeps int32 dtype and gives group minima per row.
    # GH#58084
    df = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    df["Y"] = df["Y"].astype("int32")
    categories = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])

    gb = df.groupby(categories, observed=False)
    result = gb.transform("min")

    expected = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_nan_in_cumsum_group_label():
    # [FUNCTION DOC]
    # Regression test: cumulative sum by a nullable Int16 key containing missing gives a nullable
    # Int16 Series with a missing value for the null-key row.
    # GH#58811
    df = DataFrame({"A": [1, None], "B": [2, 3]}, dtype="Int16")
    gb = df.groupby("A")["B"]
    result = gb.cumsum()
    expected = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)
