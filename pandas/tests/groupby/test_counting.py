# [MODULE DOC]
# Tests for counting-related groupby operations: cumcount, ngroup and count on DataFrameGroupBy and SeriesGroupBy. Covers duplicate, multi and default indexes, empty inputs, group ordering, descending numbering, datetime-like and categorical columns, missing values, low precision integers and extension string arrays. Module-level count tests follow the TestCounting class.
from itertools import product
from string import ascii_lowercase

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestCounting:
    # [CLASS DOC]
    # Test class grouping the cumcount, ngroup and some count tests. Stateless; each method builds its own frames. Relies on the sort fixture in one method.
    def test_cumcount(self):
        # [FUNCTION DOC]
        # cumcount on a column grouped by A: numbers rows within each group in order of appearance (0, 1, 2, 0, 3). Both the frame groupby and its column selection must return the same Series with the original index.
        df = DataFrame([["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"])
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 1, 2, 0, 3])

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_cumcount_empty(self):
        # [FUNCTION DOC]
        # cumcount of an empty frame or empty object Series grouped by level 0 must return an empty int64 Series, not float.
        ge = DataFrame().groupby(level=0)
        se = Series(dtype=object).groupby(level=0)

        # edge case, as this is usually considered float
        e = Series(dtype="int64")

        tm.assert_series_equal(e, ge.cumcount())
        tm.assert_series_equal(e, se.cumcount())

    def test_cumcount_dupe_index(self):
        # [FUNCTION DOC]
        # cumcount with an all-zero (duplicated) index must retain that index and give 0, 1, 2, 0, 3.
        df = DataFrame(
            [["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=[0] * 5
        )
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 1, 2, 0, 3], index=[0] * 5)

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_cumcount_mi(self):
        # [FUNCTION DOC]
        # cumcount on a frame with a MultiIndex must keep that MultiIndex in the result with values 0, 1, 2, 0, 3.
        mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])
        df = DataFrame([["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=mi)
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 1, 2, 0, 3], index=mi)

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_cumcount_groupby_not_col(self):
        # [FUNCTION DOC]
        # cumcount when grouping by an external list (not a column) on a frame with duplicated index values: result has the original index and the same counts.
        df = DataFrame(
            [["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=[0] * 5
        )
        g = df.groupby([0, 0, 0, 1, 0])
        sg = g.A

        expected = Series([0, 1, 2, 0, 3], index=[0] * 5)

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_ngroup(self):
        # [FUNCTION DOC]
        # ngroup assigns each row the integer number of its (sorted) group: 0, 0, 0, 1, 0 for groups a and b, on both the frame groupby and column selection.
        df = DataFrame({"A": list("aaaba")})
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 0, 0, 1, 0])

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_distinct(self):
        # [FUNCTION DOC]
        # ngroup with all-distinct group keys gives 0 to 4 as int64.
        df = DataFrame({"A": list("abcde")})
        g = df.groupby("A")
        sg = g.A

        expected = Series(range(5), dtype="int64")

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_one_group(self):
        # [FUNCTION DOC]
        # ngroup with a single group gives zeros for all rows.
        df = DataFrame({"A": [0] * 5})
        g = df.groupby("A")
        sg = g.A

        expected = Series([0] * 5)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_empty(self):
        # [FUNCTION DOC]
        # ngroup on an empty frame or empty object Series grouped by level 0 gives an empty int64 Series.
        ge = DataFrame().groupby(level=0)
        se = Series(dtype=object).groupby(level=0)

        # edge case, as this is usually considered float
        e = Series(dtype="int64")

        tm.assert_series_equal(e, ge.ngroup())
        tm.assert_series_equal(e, se.ngroup())

    def test_ngroup_series_matches_frame(self):
        # [FUNCTION DOC]
        # ngroup using a Series as the grouper on a frame must equal ngroup on the Series grouped by itself.
        df = DataFrame({"A": list("aaaba")})
        s = Series(list("aaaba"))

        tm.assert_series_equal(df.groupby(s).ngroup(), s.groupby(s).ngroup())

    def test_ngroup_dupe_index(self):
        # [FUNCTION DOC]
        # ngroup with a duplicated index keeps the index in the result: 0, 0, 0, 1, 0.
        df = DataFrame({"A": list("aaaba")}, index=[0] * 5)
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 0, 0, 1, 0], index=[0] * 5)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_mi(self):
        # [FUNCTION DOC]
        # ngroup with a MultiIndex preserves it in the result.
        mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])
        df = DataFrame({"A": list("aaaba")}, index=mi)
        g = df.groupby("A")
        sg = g.A
        expected = Series([0, 0, 0, 1, 0], index=mi)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_groupby_not_col(self):
        # [FUNCTION DOC]
        # ngroup when grouping by an external list on a frame with a duplicated index preserves that index.
        df = DataFrame({"A": list("aaaba")}, index=[0] * 5)
        g = df.groupby([0, 0, 0, 1, 0])
        sg = g.A

        expected = Series([0, 0, 0, 1, 0], index=[0] * 5)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_descending(self):
        # [FUNCTION DOC]
        # Descending ngroup: with ascending false the group numbers are reversed (the last group gets 0); this must equal the number of groups minus one minus the ascending numbers. ascending true gives the default numbering.
        df = DataFrame(["a", "a", "b", "a", "b"], columns=["A"])
        g = df.groupby(["A"])

        ascending = Series([0, 0, 1, 0, 1])
        descending = Series([1, 1, 0, 1, 0])

        tm.assert_series_equal(descending, (g.ngroups - 1) - ascending)
        tm.assert_series_equal(ascending, g.ngroup(ascending=True))
        tm.assert_series_equal(descending, g.ngroup(ascending=False))

    def test_ngroup_matches_cumcount(self):
        # [FUNCTION DOC]
        # One hand-worked case: two key columns A and X; ngroup gives 0, 1, 2, 0, 3 and cumcount gives 0, 0, 0, 1, 0.
        # verify one manually-worked out case works
        df = DataFrame(
            [["a", "x"], ["a", "y"], ["b", "x"], ["a", "x"], ["b", "y"]],
            columns=["A", "X"],
        )
        g = df.groupby(["A", "X"])
        g_ngroup = g.ngroup()
        g_cumcount = g.cumcount()
        expected_ngroup = Series([0, 1, 2, 0, 3])
        expected_cumcount = Series([0, 0, 0, 1, 0])

        tm.assert_series_equal(g_ngroup, expected_ngroup)
        tm.assert_series_equal(g_cumcount, expected_cumcount)

    def test_ngroup_cumcount_pair(self):
        # [FUNCTION DOC]
        # Brute-force check over every length-4 sequence of values 0 to 2: ngroup must equal the index of the value in the sorted distinct values, and cumcount must equal the number of earlier occurrences of the same value.
        # brute force comparison for all small series
        for p in product(range(3), repeat=4):
            df = DataFrame({"a": p})
            g = df.groupby(["a"])

            order = sorted(set(p))
            ngroupd = [order.index(val) for val in p]
            cumcounted = [p[:i].count(val) for i, val in enumerate(p)]

            tm.assert_series_equal(g.ngroup(), Series(ngroupd))
            tm.assert_series_equal(g.cumcount(), Series(cumcounted))

    def test_ngroup_respects_groupby_order(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. For a random 100-row frame of letters, iterating the groupby defines group id (enumeration order) and position within group; ngroup and cumcount must equal these respectively, so numbering respects the sort option of the groupby.
        df = DataFrame({"a": np.random.default_rng(2).choice(list("abcdef"), 100)})
        g = df.groupby("a", sort=sort)
        df["group_id"] = -1
        df["group_index"] = -1

        for i, (_, group) in enumerate(g):
            df.loc[group.index, "group_id"] = i
            for j, ind in enumerate(group.index):
                df.loc[ind, "group_index"] = j

        tm.assert_series_equal(Series(df["group_id"].values), g.ngroup())
        tm.assert_series_equal(Series(df["group_index"].values), g.cumcount())

    @pytest.mark.parametrize(
        "datetimelike",
        [
            [Timestamp(f"2016-05-{i:02d} 20:09:25+00:00") for i in range(1, 4)],
            [Timestamp(f"2016-05-{i:02d} 20:09:25") for i in range(1, 4)],
            [Timestamp(f"2016-05-{i:02d} 20:09:25", tz="UTC") for i in range(1, 4)],
            [Timedelta(x, unit="h") for x in range(1, 4)],
            [Period(freq="2W", year=2017, month=x) for x in range(1, 4)],
        ],
    )
    def test_count_with_datetimelike(self, datetimelike):
        # [FUNCTION DOC]
        # Test for issue 13393: count on a frame with a datetime-like column (tz-aware, naive, UTC and timedelta, period values) grouped by x must give counts 2 and 1 in a frame indexed by x without raising.
        # test for #13393, where DataframeGroupBy.count() fails
        # when counting a datetimelike column.

        df = DataFrame({"x": ["a", "a", "b"], "y": datetimelike})
        res = df.groupby("x").count()
        expected = DataFrame({"y": [2, 1]}, index=["a", "b"])
        expected.index.name = "x"
        tm.assert_frame_equal(expected, res)

    def test_count_with_only_nans_in_first_group(self):
        # [FUNCTION DOC]
        # GH21956: grouping by two keys where the first key is all NaN leaves no groups; count on C must return an empty int64 Series named C with an empty two-level index (index type check disabled).
        # GH21956
        df = DataFrame({"A": [np.nan, np.nan], "B": ["a", "b"], "C": [1, 2]})
        result = df.groupby(["A", "B"]).C.count()
        mi = MultiIndex(levels=[[], ["a", "b"]], codes=[[], []], names=["A", "B"])
        expected = Series([], index=mi, dtype=np.int64, name="C")
        tm.assert_series_equal(result, expected, check_index_type=False)

    def test_count_groupby_column_with_nan_in_groupby_column(self):
        # [FUNCTION DOC]
        # Refs GitHub issue 32841: grouping by a column B with a NaN; count of the remaining column A gives counts of one for each non-NaN B value, indexed by the sorted B values, NaN group dropped.
        # https://github.com/pandas-dev/pandas/issues/32841
        df = DataFrame({"A": [1, 1, 1, 1, 1], "B": [5, 4, np.nan, 3, 0]})
        res = df.groupby(["B"]).count()
        expected = DataFrame(
            index=Index([0.0, 3.0, 4.0, 5.0], name="B"), data={"A": [1, 1, 1, 1]}
        )
        tm.assert_frame_equal(expected, res)

    def test_groupby_count_dateparseerror(self):
        # [FUNCTION DOC]
        # Counting a Series with a MultiIndex (datetimes first, ints second) grouped by a function of the integer level must give the same result as grouping with the levels swapped; checks no date-parsing error occurs when datetimes are the first level.
        dr = date_range(start="1/1/2012", freq="5min", periods=10)

        # BAD Example, datetimes first
        ser = Series(np.arange(10), index=[dr, np.arange(10)])
        grouped = ser.groupby(lambda x: x[1] % 2 == 0)
        result = grouped.count()

        ser = Series(np.arange(10), index=[np.arange(10), dr])
        grouped = ser.groupby(lambda x: x[0] % 2 == 0)
        expected = grouped.count()

        tm.assert_series_equal(result, expected)


def test_groupby_timedelta_cython_count():
    # [FUNCTION DOC]
    # count on a timedelta column grouped by a key: cython count must give 2 and 2 for groups a and b, as a Series named delta.
    df = DataFrame(
        {"g": list("ab" * 2), "delta": np.arange(4).astype("timedelta64[ns]")}
    )
    expected = Series([2, 2], index=Index(["a", "b"], name="g"), name="delta")
    result = df.groupby("g").delta.count()
    tm.assert_series_equal(expected, result)


def test_count():
    # [FUNCTION DOC]
    # Builds a large frame (32768 rows) with string, integer, float, datetime, timedelta and categorical columns, with NaN injected in about a tenth of the entries (except for the first, second and fourth columns). For keys first, second and both, count must equal applying the frame count function to each group.
    n = 1 << 15
    dr = date_range("2015-08-30", periods=n // 10, freq="min")

    df = DataFrame(
        {
            "1st": np.random.default_rng(2).choice(list(ascii_lowercase), n),
            "2nd": np.random.default_rng(2).integers(0, 5, n),
            "3rd": np.random.default_rng(2).standard_normal(n).round(3),
            "4th": np.random.default_rng(2).integers(-10, 10, n),
            "5th": np.random.default_rng(2).choice(dr, n),
            "6th": np.random.default_rng(2).standard_normal(n).round(3),
            "7th": np.random.default_rng(2).standard_normal(n).round(3),
            "8th": np.random.default_rng(2).choice(dr, n)
            - np.random.default_rng(2).choice(dr, 1),
            "9th": np.random.default_rng(2).choice(list(ascii_lowercase), n),
        }
    )

    for col in df.columns.drop(["1st", "2nd", "4th"]):
        df.loc[np.random.default_rng(2).choice(n, n // 10), col] = np.nan

    df["9th"] = df["9th"].astype("category")

    for key in ["1st", "2nd", ["1st", "2nd"]]:
        left = df.groupby(key).count()
        right = df.groupby(key).apply(DataFrame.count)
        tm.assert_frame_equal(left, right)


def test_count_non_nulls():
    # [FUNCTION DOC]
    # GH#5610: count counts non-null entries per group: result columns B and C have counts (1, 2) for group 1 and (0, 0) for group 3; as_index false gives the same data with the group column. A column selection gives the Series for B.
    # GH#5610
    # count counts non-nulls
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, np.nan]],
        columns=["A", "B", "C"],
    )

    count_as = df.groupby("A").count()
    count_not_as = df.groupby("A", as_index=False).count()

    expected = DataFrame([[1, 2], [0, 0]], columns=["B", "C"], index=[1, 3])
    expected.index.name = "A"
    tm.assert_frame_equal(count_not_as, expected.reset_index())
    tm.assert_frame_equal(count_as, expected)

    count_B = df.groupby("A")["B"].count()
    tm.assert_series_equal(count_B, expected["B"])


def test_count_object():
    # [FUNCTION DOC]
    # count of an object column grouped by an integer column returns 3 and 3 (no nulls).
    df = DataFrame({"a": ["a"] * 3 + ["b"] * 3, "c": [2] * 3 + [3] * 3})
    result = df.groupby("c").a.count()
    expected = Series([3, 3], index=Index([2, 3], name="c"), name="a")
    tm.assert_series_equal(result, expected)


def test_count_object_nan():
    # [FUNCTION DOC]
    # count of an object column containing NaN counts only non-null entries: 1 for the first group, 3 for the second.
    df = DataFrame({"a": ["a", np.nan, np.nan] + ["b"] * 3, "c": [2] * 3 + [3] * 3})
    result = df.groupby("c").a.count()
    expected = Series([1, 3], index=Index([2, 3], name="c"), name="a")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("typ", ["object", "float32"])
def test_count_cross_type(typ):
    # [FUNCTION DOC]
    # GH8169: parametrized over object and float32. After setting some entries to NaN, converting columns a and b to the parametrized dtype must not change the grouped count by two other columns.
    # GH8169
    # Set float64 dtype to avoid upcast when setting nan below
    vals = np.hstack(
        (
            np.random.default_rng(2).integers(0, 5, (10, 2)),
            np.random.default_rng(2).integers(0, 2, (10, 2)),
        )
    ).astype("float64")

    df = DataFrame(vals, columns=["a", "b", "c", "d"])
    df[df == 2] = np.nan
    expected = df.groupby(["c", "d"]).count()

    df["a"] = df["a"].astype(typ)
    df["b"] = df["b"].astype(typ)
    result = df.groupby(["c", "d"]).count()
    tm.assert_frame_equal(result, expected)


def test_lower_int_prec_count():
    # [FUNCTION DOC]
    # count of int8, uint32 and int16 columns grouped by a key gives two for each column and group; low-precision integer types must be handled.
    df = DataFrame(
        {
            "a": np.array([0, 1, 2, 100], np.int8),
            "b": np.array([1, 2, 3, 6], np.uint32),
            "c": np.array([4, 5, 6, 8], np.int16),
            "grp": list("ab" * 2),
        }
    )
    result = df.groupby("grp").count()
    expected = DataFrame(
        {"a": [2, 2], "b": [2, 2], "c": [2, 2]}, index=Index(list("ab"), name="grp")
    )
    tm.assert_frame_equal(result, expected)


def test_count_uses_size_on_exception():
    # [FUNCTION DOC]
    # Count must fall back to the size path when the comparison raises inside the compiled loop: a custom element type raises its own exception on equality; count of such object column grouped by key still gives 2 for each group.
    class RaisingObjectException(Exception):
        # [CLASS DOC]
        # HELPER: exception type raised by the custom object defined inside the test for count fallback.
        pass

    class RaisingObject:
        # [CLASS DOC]
        # HELPER: object type used to check that count still works when comparison of elements raises. It stores a message string (default text about raising inside Cython).
        def __init__(self, msg="I will raise inside Cython") -> None:
            # [FUNCTION DOC]
            # HELPER: initializer. Takes msg (a string, default message mentioning raising inside Cython), calls the base initializer and stores the message on the instance.
            super().__init__()
            self.msg = msg

        def __eq__(self, other):
            # [FUNCTION DOC]
            # HELPER: equality comparison that always raises the custom exception with the stored message, so that compiled code that tries to compare values hits an error.
            # gets called in Cython to check that raising calls the method
            raise RaisingObjectException(self.msg)

    df = DataFrame({"a": [RaisingObject() for _ in range(4)], "grp": list("ab" * 2)})
    result = df.groupby("grp").count()
    expected = DataFrame({"a": [2, 2]}, index=Index(list("ab"), name="grp"))
    tm.assert_frame_equal(result, expected)


def test_count_arrow_string_array(any_string_dtype):
    # [FUNCTION DOC]
    # GH#54751: parametrized over any string dtype (needs the pyarrow package, skipped otherwise). count of a string column grouped by an integer column gives 1 per group, in a frame with index named a.
    # GH#54751
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {"a": [1, 2, 3], "b": Series(["a", "b", "a"], dtype=any_string_dtype)}
    )
    result = df.groupby("a").count()
    expected = DataFrame({"b": 1}, index=Index([1, 2, 3], name="a"))
    tm.assert_frame_equal(result, expected)
