"""
test all other .agg behavior
"""
# [MODULE DOC]
# Tests for groupby aggregation behaviors not covered elsewhere: error propagation,
# dictionary and list specifications (including the unsupported nested renamer forms),
# callables, datetimes and time zones, structure-valued results, categorical grouping
# with numpy sums, and error suppression in the Python aggregation path. Uses the
# observed fixture and the specification error type.

import datetime as dt
from functools import partial

import numpy as np
import pytest

from pandas.errors import SpecificationError

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm

from pandas.io.formats.printing import pprint_thing


def test_agg_partial_failure_raises():
    # [FUNCTION DOC]
    # Verifies (GH#43741) that a function failing for non-numeric columns (max minus min
    # on strings) raises a type error mentioning unsupported operand types, both when
    # given in a list and on its own.
    # GH#43741

    df = DataFrame(
        {
            "data1": np.random.default_rng(2).standard_normal(5),
            "data2": np.random.default_rng(2).standard_normal(5),
            "key1": ["a", "a", "b", "b", "a"],
            "key2": ["one", "two", "one", "two", "one"],
        }
    )
    grouped = df.groupby("key1")

    def peak_to_peak(arr):
        # [FUNCTION DOC]
        # HELPER: reducer returning the maximum minus the minimum of its input; used to
        # trigger a type error on string columns.
        return arr.max() - arr.min()

    with pytest.raises(TypeError, match="unsupported operand type"):
        grouped.agg([peak_to_peak])

    with pytest.raises(TypeError, match="unsupported operand type"):
        grouped.agg(peak_to_peak)


def test_agg_datetimes_mixed():
    # [FUNCTION DOC]
    # Verifies grouping by a date column aggregated with sum gives the same number of
    # groups whether the dates are strings or date objects with a missing entry.
    data = [[1, "2012-01-01", 1.0], [2, "2012-01-02", 2.0], [3, None, 3.0]]

    df1 = DataFrame(
        {
            "key": [x[0] for x in data],
            "date": [x[1] for x in data],
            "value": [x[2] for x in data],
        }
    )

    data = [
        [
            row[0],
            (dt.datetime.strptime(row[1], "%Y-%m-%d").date() if row[1] else None),
            row[2],
        ]
        for row in data
    ]

    df2 = DataFrame(
        {
            "key": [x[0] for x in data],
            "date": [x[1] for x in data],
            "value": [x[2] for x in data],
        }
    )

    df1["weights"] = df1["value"] / df1["value"].sum()
    gb1 = df1.groupby("date").aggregate("sum")

    df2["weights"] = df1["value"] / df1["value"].sum()
    gb2 = df2.groupby("date").aggregate("sum")

    assert len(gb1) == len(gb2)


def test_agg_period_index():
    # [FUNCTION DOC]
    # Verifies sum grouped by index level zero of a period-indexed frame keeps a period
    # index, and (GH 3579) grouping a period-indexed frame by month can be iterated
    # without error. Only the index type and non-failure are checked.
    prng = period_range("2012-1-1", freq="M", periods=3)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)), index=prng)
    rs = df.groupby(level=0).sum()
    assert isinstance(rs.index, PeriodIndex)

    # GH 3579
    index = period_range(start="1999-01", periods=5, freq="M")
    s1 = Series(np.random.default_rng(2).random(len(index)), index=index)
    s2 = Series(np.random.default_rng(2).random(len(index)), index=index)
    df = DataFrame.from_dict({"s1": s1, "s2": s2})
    grouped = df.groupby(df.index.month)
    list(grouped)


def test_agg_dict_parameter_cast_result_dtypes():
    # [FUNCTION DOC]
    # Verifies (GH 12821) first, last, via method, string aggregation and dictionary
    # aggregation on a datetime column with missing values give matching expected frames
    # and series, and that length-based aggregation and size equal two per group while
    # count gives non-missing counts.
    # GH 12821

    df = DataFrame(
        {
            "class": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "time": date_range("1/1/2011", periods=8, freq="h"),
        }
    )
    df.loc[[0, 1, 2, 5], "time"] = None

    # test for `first` function
    exp = df.loc[[0, 3, 4, 6]].set_index("class")
    grouped = df.groupby("class")
    tm.assert_frame_equal(grouped.first(), exp)
    tm.assert_frame_equal(grouped.agg("first"), exp)
    tm.assert_frame_equal(grouped.agg({"time": "first"}), exp)
    tm.assert_series_equal(grouped.time.first(), exp["time"])
    tm.assert_series_equal(grouped.time.agg("first"), exp["time"])

    # test for `last` function
    exp = df.loc[[0, 3, 4, 7]].set_index("class")
    grouped = df.groupby("class")
    tm.assert_frame_equal(grouped.last(), exp)
    tm.assert_frame_equal(grouped.agg("last"), exp)
    tm.assert_frame_equal(grouped.agg({"time": "last"}), exp)
    tm.assert_series_equal(grouped.time.last(), exp["time"])
    tm.assert_series_equal(grouped.time.agg("last"), exp["time"])

    # count
    exp = Series([2, 2, 2, 2], index=Index(list("ABCD"), name="class"), name="time")
    tm.assert_series_equal(grouped.time.agg(len), exp)
    tm.assert_series_equal(grouped.time.size(), exp)

    exp = Series([0, 1, 1, 2], index=Index(list("ABCD"), name="class"), name="time")
    tm.assert_series_equal(grouped.time.count(), exp)


def test_agg_cast_results_dtypes():
    # [FUNCTION DOC]
    # Verifies (xref #11444) aggregating with the length function on a datetime column
    # equals the grouped count.
    # similar to GH12821
    # xref #11444
    u = [dt.datetime(2015, x + 1, 1) for x in range(12)]
    v = list("aaabbbbbbccd")
    df = DataFrame({"X": v, "Y": u})

    result = df.groupby("X")["Y"].agg(len)
    expected = df.groupby("X")["Y"].count()
    tm.assert_series_equal(result, expected)


def test_aggregate_float64_no_int64():
    # [FUNCTION DOC]
    # Verifies (gh-11199) grouped mean of integer columns returns float values, for one
    # and two selected columns.
    # see gh-11199
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 4, 5], "c": [1, 2, 3, 4, 5]})

    expected = DataFrame({"a": [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = "b"

    result = df.groupby("b")[["a"]].mean()
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"a": [1, 2.5, 4, 5], "c": [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = "b"

    result = df.groupby("b")[["a", "c"]].mean()
    tm.assert_frame_equal(result, expected)


def test_aggregate_api_consistency():
    # [FUNCTION DOC]
    # Verifies (GH 9052) aggregation with lists and dictionaries is consistent with
    # computing each statistic separately and concatenating: series with sum and mean,
    # frame with list, reordered column selection, a dictionary of single functions, and
    # that dictionary keys naming non-existing labels raise a key error listing them.
    # KNOWN ISSUE: the expected frame built for the dictionary of function lists is
    # never compared with a result, so that case is not actually checked.
    # GH 9052
    # make sure that the aggregates via dict
    # are consistent
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    grouped = df.groupby(["A", "B"])
    c_mean = grouped["C"].mean()
    c_sum = grouped["C"].sum()
    d_mean = grouped["D"].mean()
    d_sum = grouped["D"].sum()

    result = grouped["D"].agg(["sum", "mean"])
    expected = pd.concat([d_sum, d_mean], axis=1)
    expected.columns = ["sum", "mean"]
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped.agg(["sum", "mean"])
    expected = pd.concat([c_sum, c_mean, d_sum, d_mean], axis=1)
    expected.columns = MultiIndex.from_product([["C", "D"], ["sum", "mean"]])
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped[["D", "C"]].agg(["sum", "mean"])
    expected = pd.concat([d_sum, d_mean, c_sum, c_mean], axis=1)
    expected.columns = MultiIndex.from_product([["D", "C"], ["sum", "mean"]])
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped.agg({"C": "mean", "D": "sum"})
    expected = pd.concat([d_sum, c_mean], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped.agg({"C": ["mean", "sum"], "D": ["mean", "sum"]})
    expected = pd.concat([c_mean, c_sum, d_mean, d_sum], axis=1)
    expected.columns = MultiIndex.from_product([["C", "D"], ["mean", "sum"]])

    msg = r"Label\(s\) \['r', 'r2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        grouped[["D", "C"]].agg({"r": "sum", "r2": "mean"})


def test_agg_dict_renaming_deprecation():
    # [FUNCTION DOC]
    # Verifies (15931) that nested renaming dictionaries in aggregation raise the
    # specification error ("nested renamer is not supported"), a dictionary with a
    # non-existing label raises a key error, and a renamer dictionary on a series group
    # also raises the specification error.
    # 15931
    df = DataFrame({"A": [1, 1, 1, 2, 2], "B": range(5), "C": range(5)})

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        df.groupby("A").agg(
            {"B": {"foo": ["sum", "max"]}, "C": {"bar": ["count", "min"]}}
        )

    msg = r"Label\(s\) \['ma'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.groupby("A")[["B", "C"]].agg({"ma": "max"})

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        df.groupby("A").B.agg({"foo": "count"})


def test_agg_compat():
    # [FUNCTION DOC]
    # Verifies (GH 12334) that selecting a series and aggregating with dictionaries
    # keyed by other column names raises the specification error for nested renamers.
    # GH 12334
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    g = df.groupby(["A", "B"])

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"C": ["sum", "std"]})

    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"C": "sum", "D": "std"})


def test_agg_nested_dicts():
    # [FUNCTION DOC]
    # Verifies that nested dictionaries in groupby aggregation, including dictionaries
    # naming new result labels on a selected series and reusing the original column name
    # (GH9052), raise the specification error.
    # API change for disallowing these types of nested dicts
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    g = df.groupby(["A", "B"])

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        g.aggregate({"r1": {"C": ["mean", "sum"]}, "r2": {"D": ["mean", "sum"]}})

    with pytest.raises(SpecificationError, match=msg):
        g.agg({"C": {"ra": ["mean", "std"]}, "D": {"rb": ["mean", "std"]}})

    # same name as the original column
    # GH9052
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"result1": np.sum, "result2": np.mean})

    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"D": np.sum, "result2": np.mean})


def test_agg_item_by_item_raise_typeerror():
    # [FUNCTION DOC]
    # Verifies that a type error raised inside a user aggregation function propagates
    # unchanged out of the grouped aggregation.
    df = DataFrame(np.random.default_rng(2).integers(10, size=(20, 10)))

    def raiseException(df):
        # [FUNCTION DOC]
        # HELPER: aggregation function that prints the frame it receives and raises a
        # type error with message test.
        pprint_thing("----------------------------------------")
        pprint_thing(df.to_string())
        raise TypeError("test")

    with pytest.raises(TypeError, match="test"):
        df.groupby(0).agg(raiseException)


def test_series_agg_multikey():
    # [FUNCTION DOC]
    # Verifies string aggregation sum on a series grouped by year and month callables
    # equals the sum method.
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])

    result = grouped.agg("sum")
    expected = grouped.sum()
    tm.assert_series_equal(result, expected)


def test_series_agg_multi_pure_python():
    # [FUNCTION DOC]
    # Verifies the pure-Python aggregation path with a function that returns a constant
    # string for a frame grouped by two keys equals the result of a lambda returning the
    # same string. The helper also asserts that array data passed in has a base object
    # (is a view).
    data = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    def bad(x):
        # [FUNCTION DOC]
        # HELPER: aggregation function returning the constant string foo; when the input
        # values are a numpy array, asserts the array has a non-empty base (so the data
        # is a view, not a copy).
        if isinstance(x.values, np.ndarray):
            assert len(x.values.base) > 0
        return "foo"

    result = data.groupby(["A", "B"]).agg(bad)
    expected = data.groupby(["A", "B"]).agg(lambda x: "foo")
    tm.assert_frame_equal(result, expected)


def test_agg_consistency():
    # [FUNCTION DOC]
    # Verifies (GH 6715) that aggregating with a single function and with a one-element
    # list of the function gives the same values (after flattening the column levels).
    # Grouped by date.
    # agg with ([]) and () not consistent
    # GH 6715
    def P1(a):
        # [FUNCTION DOC]
        # HELPER: returns the first percentile of the input after dropping missing
        # values.
        return np.percentile(a.dropna(), q=1)

    df = DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": [10, 25, 26, 31],
            "date": [
                dt.date(2013, 2, 10),
                dt.date(2013, 2, 10),
                dt.date(2013, 2, 11),
                dt.date(2013, 2, 11),
            ],
        }
    )

    g = df.groupby("date")

    expected = g.agg([P1])
    expected.columns = expected.columns.levels[0]

    result = g.agg(P1)
    tm.assert_frame_equal(result, expected)


def test_agg_callables():
    # [FUNCTION DOC]
    # Verifies (GH 7929) that equivalent callables (builtin sum, numpy sum, lambdas, a
    # partial, and a callable class instance) give the same result as the string sum for
    # an integer frame.
    # GH 7929
    df = DataFrame({"foo": [1, 2], "bar": [3, 4]}).astype(np.int64)

    class fn_class:
        # [CLASS DOC]
        # HELPER: callable class whose call method returns the builtin sum of its
        # argument; one of the callables compared in the enclosing test.
        def __call__(self, x):
            # [FUNCTION DOC]
            # HELPER: returns the builtin sum of the given sequence.
            return sum(x)

    equiv_callables = [
        sum,
        np.sum,
        lambda x: sum(x),
        lambda x: x.sum(),
        partial(sum),
        fn_class(),
    ]

    expected = df.groupby("foo").agg("sum")
    for ecall in equiv_callables:
        result = df.groupby("foo").agg(ecall)
        tm.assert_frame_equal(result, expected)


def test_agg_over_numpy_arrays():
    # [FUNCTION DOC]
    # Verifies (GH 3788) summing object columns holding numpy arrays with numeric-only
    # disabled gives element-wise array sums, via method and via aggregate.
    # GH 3788
    df = DataFrame(
        [
            [1, np.array([10, 20, 30])],
            [1, np.array([40, 50, 60])],
            [2, np.array([20, 30, 40])],
        ],
        columns=["category", "arraydata"],
    )
    gb = df.groupby("category")

    expected_data = [[np.array([50, 70, 90])], [np.array([20, 30, 40])]]
    expected_index = Index([1, 2], name="category")
    expected_column = ["arraydata"]
    expected = DataFrame(expected_data, index=expected_index, columns=expected_column)

    alt = gb.sum(numeric_only=False)
    tm.assert_frame_equal(alt, expected)

    result = gb.agg("sum", numeric_only=False)
    tm.assert_frame_equal(result, expected)

    # FIXME: the original version of this test called `gb.agg(sum)`
    #  and that raises TypeError if `numeric_only=False` is passed


@pytest.mark.parametrize("as_period", [True, False])
def test_agg_tzaware_non_datetime_result(as_period):
    # [FUNCTION DOC]
    # Verifies (GH#29589, GH#29641) aggregation of tz-aware (or period) values: a
    # dtype-preserving function keeps the dtype, a function extracting the year gives
    # integers, and a subtraction function gives time deltas (or day offsets for
    # periods). Parametrized over whether the values are periods.
    # discussed in GH#29589, fixed in GH#29641, operating on tzaware values
    #  with function that is not dtype-preserving
    dti = date_range("2012-01-01", periods=4, tz="UTC")
    if as_period:
        dti = dti.tz_localize(None).to_period("D")

    df = DataFrame({"a": [0, 0, 1, 1], "b": dti})
    gb = df.groupby("a")

    # Case that _does_ preserve the dtype
    result = gb["b"].agg(lambda x: x.iloc[0])
    expected = Series(dti[::2], name="b")
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)

    # Cases that do _not_ preserve the dtype
    result = gb["b"].agg(lambda x: x.iloc[0].year)
    expected = Series([2012, 2012], name="b")
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)

    result = gb["b"].agg(lambda x: x.iloc[-1] - x.iloc[0])
    expected = Series(
        [pd.Timedelta(days=1), pd.Timedelta(days=1)], name="b", dtype="m8[us]"
    )
    expected.index.name = "a"
    if as_period:
        expected = Series([pd.offsets.Day(1), pd.offsets.Day(1)], name="b")
        expected.index.name = "a"
    tm.assert_series_equal(result, expected)


def test_agg_timezone_round_trip():
    # [FUNCTION DOC]
    # Verifies (GH 15426, GH#27110) time zone round trip: min via string, lambda and
    # method of tz-aware values returns the original timestamp; nth, head, first, last
    # and apply with positional selection return the right timestamps.
    # GH 15426
    ts = pd.Timestamp("2016-01-01 12:00:00", tz="US/Pacific")
    df = DataFrame({"a": 1, "b": [ts + dt.timedelta(minutes=nn) for nn in range(10)]})

    result1 = df.groupby("a")["b"].agg("min").iloc[0]
    result2 = df.groupby("a")["b"].agg(lambda x: np.min(x)).iloc[0]
    result3 = df.groupby("a")["b"].min().iloc[0]

    assert result1 == ts
    assert result2 == ts
    assert result3 == ts

    dates = [
        pd.Timestamp(f"2016-01-0{i:d} 12:00:00", tz="US/Pacific") for i in range(1, 5)
    ]
    df = DataFrame({"A": ["a", "b"] * 2, "B": dates})
    grouped = df.groupby("A")

    ts = df["B"].iloc[0]
    assert ts == grouped.nth(0)["B"].iloc[0]
    assert ts == grouped.head(1)["B"].iloc[0]
    assert ts == grouped.first()["B"].iloc[0]

    # GH#27110 applying iloc should return a DataFrame
    assert ts == grouped.apply(lambda x: x.iloc[0])["B"].iloc[0]

    ts = df["B"].iloc[2]
    assert ts == grouped.last()["B"].iloc[0]

    # GH#27110 applying iloc should return a DataFrame
    assert ts == grouped.apply(lambda x: x.iloc[-1])["B"].iloc[0]


def test_sum_uint64_overflow():
    # [FUNCTION DOC]
    # Verifies (gh-14758) summing object-dtype data near the 64-bit signed maximum does
    # not overflow, produces unsigned 64-bit index labels, and with numeric-only enabled
    # drops the non-numeric result column.
    # see gh-14758
    # Convert to uint64 and don't overflow
    df = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=object)
    df = df + 9223372036854775807

    index = Index(
        [9223372036854775808, 9223372036854775810, 9223372036854775812], dtype=np.uint64
    )
    expected = DataFrame(
        {1: [9223372036854775809, 9223372036854775811, 9223372036854775813]},
        index=index,
        dtype=object,
    )

    expected.index.name = 0
    result = df.groupby(0).sum(numeric_only=False)
    tm.assert_frame_equal(result, expected)

    # out column is non-numeric, so with numeric_only=True it is dropped
    result2 = df.groupby(0).sum(numeric_only=True)
    expected2 = expected[[]]
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize(
    "structure, cast_as",
    [
        (tuple, tuple),
        (list, list),
        (lambda x: tuple(x), tuple),
        (lambda x: list(x), list),
    ],
)
def test_agg_structs_dataframe(structure, cast_as):
    # [FUNCTION DOC]
    # Verifies aggregation of a frame grouped by two keys with tuple or list
    # constructors (or lambdas wrapping them) yields tuples or lists per group as cell
    # values. Parametrized over the structure callable and the expected container type.
    df = DataFrame(
        {"A": [1, 1, 1, 3, 3, 3], "B": [1, 1, 1, 4, 4, 4], "C": [1, 1, 1, 3, 4, 4]}
    )

    result = df.groupby(["A", "B"]).aggregate(structure)
    expected = DataFrame(
        {"C": {(1, 1): cast_as([1, 1, 1]), (3, 4): cast_as([3, 4, 4])}}
    )
    expected.index.names = ["A", "B"]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "structure, cast_as",
    [
        (tuple, tuple),
        (list, list),
        (lambda x: tuple(x), tuple),
        (lambda x: list(x), list),
    ],
)
def test_agg_structs_series(structure, cast_as):
    # [FUNCTION DOC]
    # Verifies (Issue #18079) the same structure aggregation on a selected series
    # grouped by one key.
    # Issue #18079
    df = DataFrame(
        {"A": [1, 1, 1, 3, 3, 3], "B": [1, 1, 1, 4, 4, 4], "C": [1, 1, 1, 3, 4, 4]}
    )

    result = df.groupby("A")["C"].aggregate(structure)
    expected = Series([cast_as([1, 1, 1]), cast_as([3, 4, 4])], index=[1, 3], name="C")
    expected.index.name = "A"
    tm.assert_series_equal(result, expected)


def test_agg_category_nansum(observed):
    # [FUNCTION DOC]
    # Verifies aggregation with the numpy nan-aware sum on a categorical key gives zero
    # for unobserved categories, which are dropped when observed is true. Uses the
    # observed fixture.
    categories = ["a", "b", "c"]
    df = DataFrame(
        {"A": pd.Categorical(["a", "a", "b"], categories=categories), "B": [1, 2, 3]}
    )
    result = df.groupby("A", observed=observed).B.agg(np.nansum)
    expected = Series(
        [3, 3, 0],
        index=pd.CategoricalIndex(["a", "b", "c"], categories=categories, name="A"),
        name="B",
    )
    if observed:
        expected = expected[expected != 0]
    tm.assert_series_equal(result, expected)


def test_agg_list_like_func():
    # [FUNCTION DOC]
    # Verifies (GH 18473) aggregation with a dictionary naming a lambda that returns a
    # list, with as_index false and no sorting, yields list cells and the key column.
    # GH 18473
    df = DataFrame({"A": [str(x) for x in range(3)], "B": [str(x) for x in range(3)]})
    grouped = df.groupby("A", as_index=False, sort=False)
    result = grouped.agg({"B": lambda x: list(x)})
    expected = DataFrame(
        {"A": [str(x) for x in range(3)], "B": [[str(x)] for x in range(3)]}
    )
    tm.assert_frame_equal(result, expected)


def test_agg_lambda_with_timezone():
    # [FUNCTION DOC]
    # Verifies (GH 23683) aggregation of a tz-aware column with a lambda taking the head
    # of each group keeps the timestamp and its UTC time zone.
    # GH 23683
    df = DataFrame(
        {
            "tag": [1, 1],
            "date": [
                pd.Timestamp("2018-01-01", tz="UTC"),
                pd.Timestamp("2018-01-02", tz="UTC"),
            ],
        }
    )
    result = df.groupby("tag").agg({"date": lambda e: e.head(1)})
    expected = DataFrame(
        [pd.Timestamp("2018-01-01", tz="UTC")],
        index=Index([1], name="tag"),
        columns=["date"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "err_cls",
    [
        NotImplementedError,
        RuntimeError,
        KeyError,
        IndexError,
        OSError,
        ValueError,
        ArithmeticError,
        AttributeError,
    ],
)
def test_groupby_agg_err_catching(err_cls):
    # [FUNCTION DOC]
    # Verifies the Python-path aggregation suppresses errors other than type or
    # assertion errors, using the decimal extension array so the numpy paths are not
    # taken. Parametrized over several exception classes (not-implemented, runtime, key,
    # index, OS, value, arithmetic, attribute errors); the aggregate takes the first
    # element per group.
    # KNOWN ISSUE: the helper function raises only for an empty group, and groups are
    # never empty here, so the exception is never raised and the suppression logic is
    # not actually exercised.
    # KNOWN ISSUE: the helper's comment says exceptions other than a type error or an
    # index error, while the leading comment says type or assertion error, and the
    # parametrization includes an index error.
    # make sure we suppress anything other than TypeError or AssertionError
    #  in _python_agg_general

    # Use a non-standard EA to make sure we don't go down ndarray paths
    from pandas.tests.extension.decimal.array import (
        DecimalArray,
        make_data,
        to_decimal,
    )

    data = make_data(5)
    df = DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    expected = Series(to_decimal([data[0], data[3]]))

    def weird_func(x):
        # [FUNCTION DOC]
        # HELPER: aggregation function for the enclosing test. Raises the parametrized
        # exception class when given an empty group; otherwise returns the first element
        # of the group.
        # weird function that raise something other than TypeError or IndexError
        #  in _python_agg_general
        if len(x) == 0:
            raise err_cls
        return x.iloc[0]

    result = df["decimals"].groupby(df["id1"]).agg(weird_func)
    tm.assert_series_equal(result, expected, check_names=False)
