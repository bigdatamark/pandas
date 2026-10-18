# [MODULE DOC]
# Tests for window operations on grouped data (the groupby then rolling,
# expanding or ewm combinations): equivalence with applying the window to each
# group, result index construction (group keys combined with original index
# levels), handling of the grouping column, custom indexers, closed intervals,
# time-based windows and the on argument, monotonicity validation, named
# aggregation, empty and nan cases, ewm with times, and non-mutation of the
# grouped object. Organized into three test classes (rolling, expanding, ewm)
# plus two module-level tests, with two fixtures: a times frame and a
# group-and-values frame.
import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NamedAgg,
    Series,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby


@pytest.fixture
def times_frame():
    # [FUNCTION DOC]
    # Fixture returning a ten-row frame for ewm with times: column A holds group
    # labels a, b, c repeating, column B an integer sequence per position, and
    # column C irregular datetimes in January 2020.
    """Frame for testing times argument in EWM groupby."""
    return DataFrame(
        {
            "A": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "B": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            "C": to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-10",
                    "2020-01-22",
                    "2020-01-03",
                    "2020-01-23",
                    "2020-01-23",
                    "2020-01-04",
                ]
            ),
        }
    )


@pytest.fixture
def roll_frame():
    # [FUNCTION DOC]
    # Fixture returning a 40-row frame with group column A (20 ones, 12 twos, 8
    # threes) and column B holding the integers 0 to 39.
    return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})


class TestRolling:
    # [CLASS DOC]
    # Tests for grouped rolling windows. Most tests build expected results by applying
    # the equivalent rolling operation to every group with a group-wise apply and
    # rebuilding the expected two-level index (group label, original position). No
    # state.
    def test_groupby_unsupported_argument(self, roll_frame):
        # [FUNCTION DOC]
        # Verifies grouping with an unknown keyword argument raises a type error saying
        # the grouping call got an unexpected keyword argument named foo.
        msg = r"groupby\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            roll_frame.groupby("A", foo=1)

    def test_getitem(self, roll_frame):
        # [FUNCTION DOC]
        # Verifies selecting column B in four orders (after the aggregation, on the
        # rolling object, on the grouped object before rolling, and via a series grouped
        # by another series) all equal the group-wise mean of a two-observation rolling
        # window computed by an internal grouping helper.
        g = roll_frame.groupby("A")
        g_mutated = get_groupby(roll_frame, by="A")

        expected = g_mutated.B.apply(lambda x: x.rolling(2).mean())

        result = g.rolling(2).mean().B
        tm.assert_series_equal(result, expected)

        result = g.rolling(2).B.mean()
        tm.assert_series_equal(result, expected)

        result = g.B.rolling(2).mean()
        tm.assert_series_equal(result, expected)

        result = roll_frame.B.groupby(roll_frame.A).rolling(2).mean()
        tm.assert_series_equal(result, expected)

    def test_getitem_multiple(self, roll_frame):
        # [FUNCTION DOC]
        # Regression test for GH 13174. Verifies the count of a column selected from a
        # grouped rolling object (window 2, minimum periods 0) equals the group-wise
        # count; the same check is made twice to confirm repeating the call is stable.
        # GH 13174
        g = roll_frame.groupby("A")
        r = g.rolling(2, min_periods=0)
        g_mutated = get_groupby(roll_frame, by="A")
        expected = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())

        result = r.B.count()
        tm.assert_series_equal(result, expected)

        result = r.B.count()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "min",
            "max",
            "first",
            "last",
            "count",
            "kurt",
            "skew",
            "nunique",
        ],
    )
    def test_rolling(self, f, roll_frame):
        # [FUNCTION DOC]
        # Parametrized over ten methods (sum, mean, min, max, first, last, count,
        # kurtosis, skewness, unique count). Verifies a grouped rolling window of 4
        # equals the group-wise rolling result with the expected index being the group
        # label combined with the original row position (GH 39732).
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.rolling(4), f)())
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_rolling_ddof(self, f, roll_frame):
        # [FUNCTION DOC]
        # Parametrized over std and var with delta degrees of freedom 1. Same check as
        # the method test: grouped window of 4 equals the group-wise result with the
        # reconstructed index.
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result = getattr(r, f)(ddof=1)
        expected = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_rolling_quantile(self, interpolation, roll_frame):
        # [FUNCTION DOC]
        # Parametrized over five interpolation methods. Verifies the grouped rolling
        # quantile 0.4 (window 4) equals the group-wise result with the reconstructed
        # index.
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result = r.quantile(0.4, interpolation=interpolation)
        expected = g.apply(
            lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation)
        )
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f, expected_val", [["corr", 1], ["cov", 0.5]])
    def test_rolling_corr_cov_other_same_size_as_groups(self, f, expected_val):
        # [FUNCTION DOC]
        # Regression test for GH 42915. Parametrized over correlation (expected 1) and
        # covariance (expected 0.5). A frame indexed by two levels is grouped by the first
        # level and rolled with window 2 against an other frame whose length equals the
        # group size; verifies the result has missing first values per group and the
        # expected value afterwards, with a three-level index whose first two levels are
        # both named idx1.
        # GH 42915
        df = DataFrame(
            {"value": range(10), "idx1": [1] * 5 + [2] * 5, "idx2": [1, 2, 3, 4, 5] * 2}
        ).set_index(["idx1", "idx2"])
        other = DataFrame({"value": range(5), "idx2": [1, 2, 3, 4, 5]}).set_index(
            "idx2"
        )
        result = getattr(df.groupby(level=0).rolling(2), f)(other)
        expected_data = ([np.nan] + [expected_val] * 4) * 2
        expected = DataFrame(
            expected_data,
            columns=["value"],
            index=MultiIndex.from_arrays(
                [
                    [1] * 5 + [2] * 5,
                    [1] * 5 + [2] * 5,
                    list(range(1, 6)) * 2,
                ],
                names=["idx1", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f, roll_frame):
        # [FUNCTION DOC]
        # Parametrized over corr and cov. Verifies a grouped rolling window of 4 against
        # the full frame (different size from groups) equals the group-wise result,
        # except that the grouping column A is set to missing in the expectation (GH
        # 39591: the group-wise apply would insert zeros for covariance).
        g = roll_frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r, f)(roll_frame)

        def func(x):
            # [FUNCTION DOC]
            # HELPER: nested function applied per group. Takes a group frame x and returns
            # the window-4 rolling correlation or covariance (as chosen by the enclosing
            # test) of x against the full frame.
            return getattr(x.rolling(4), f)(roll_frame)

        expected = g.apply(func)
        # GH 39591: The grouped column should be all np.nan
        # (groupby.apply inserts 0s for cov)
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_pairwise(self, f, roll_frame):
        # [FUNCTION DOC]
        # Parametrized over corr and cov. Verifies the pairwise form of column B of a
        # grouped rolling window of 4 equals the group-wise application.
        g = roll_frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r.B, f)(pairwise=True)

        def func(x):
            # [FUNCTION DOC]
            # HELPER: nested function applied per group. Takes a group frame x and returns
            # the pairwise rolling window-4 correlation or covariance (as chosen by the
            # enclosing test) of its column B.
            return getattr(x.B.rolling(4), f)(pairwise=True)

        expected = g.apply(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, expected_values",
        [("cov", [[1.0, 1.0], [1.0, 4.0]]), ("corr", [[1.0, 0.5], [0.5, 1.0]])],
    )
    def test_rolling_corr_cov_unordered(self, func, expected_values):
        # [FUNCTION DOC]
        # Regression test for GH 43386. Parametrized over cov and corr with expected
        # 2 by 2 values. Verifies grouped rolling (window 3) pairwise results for a
        # frame whose groups are interleaved: the result has a three-level index (group,
        # original position, column name), with missing values except for the one full
        # window of group g1.
        # GH 43386
        df = DataFrame(
            {
                "a": ["g1", "g2", "g1", "g1"],
                "b": [0, 0, 1, 2],
                "c": [2, 0, 6, 4],
            }
        )
        rol = df.groupby("a").rolling(3)
        result = getattr(rol, func)()
        expected = DataFrame(
            {
                "b": 4 * [np.nan] + expected_values[0] + 2 * [np.nan],
                "c": 4 * [np.nan] + expected_values[1] + 2 * [np.nan],
            },
            index=MultiIndex.from_tuples(
                [
                    ("g1", 0, "b"),
                    ("g1", 0, "c"),
                    ("g1", 2, "b"),
                    ("g1", 2, "c"),
                    ("g1", 3, "b"),
                    ("g1", 3, "c"),
                    ("g2", 1, "b"),
                    ("g2", 1, "c"),
                ],
                names=["a", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply(self, raw, roll_frame):
        # [FUNCTION DOC]
        # Verifies grouped rolling apply of a sum function (raw fixture) equals the
        # group-wise apply with the reconstructed index (GH 39732).
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        # reduction
        result = r.apply(lambda x: x.sum(), raw=raw)
        expected = g.apply(lambda x: x.rolling(4).apply(lambda y: y.sum(), raw=raw))
        # GH 39732
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply_mutability(self):
        # [FUNCTION DOC]
        # Regression test for GH 14013. Verifies a grouped rolling sum (window 2) leaves
        # the grouping column out of the result, gives the expected values with a
        # two-level index, and that running an unrelated aggregation on the same grouped
        # object does not change a second run of the rolling sum.
        # GH 14013
        df = DataFrame({"A": ["foo"] * 3 + ["bar"] * 3, "B": [1] * 6})
        g = df.groupby("A")

        mi = MultiIndex.from_tuples(
            [("bar", 3), ("bar", 4), ("bar", 5), ("foo", 0), ("foo", 1), ("foo", 2)]
        )

        mi.names = ["A", None]
        # Grouped column should not be a part of the output
        expected = DataFrame([np.nan, 2.0, 2.0] * 2, columns=["B"], index=mi)

        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

        # Call an arbitrary function on the groupby
        g.sum()

        # Make sure nothing has been mutated
        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("expected_value,raw_value", [[1.0, True], [0.0, False]])
    def test_groupby_rolling(self, expected_value, raw_value):
        # [FUNCTION DOC]
        # Regression test for GH 31754. Parametrized over raw true (expecting 1.0) and
        # false (expecting 0.0). Verifies the object passed to the applied function is a
        # numpy array exactly when raw is true.
        # GH 31754

        def isnumpyarray(x):
            # [FUNCTION DOC]
            # HELPER: nested function. Takes any object x and returns the integer 1 if it is
            # a numpy array, otherwise 0.
            return int(isinstance(x, np.ndarray))

        df = DataFrame({"id": [1, 1, 1], "value": [1, 2, 3]})
        result = df.groupby("id").value.rolling(1).apply(isnumpyarray, raw=raw_value)
        expected = Series(
            [expected_value] * 3,
            index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=["id", None]),
            name="value",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self):
        # [FUNCTION DOC]
        # Regression test for GH 35552. Verifies centered grouped rolling means with
        # window 3: for series grouped by their own distinct values every window has one
        # observation so results are all missing (odd and even lengths); for a frame
        # with groups of 5 and 6 (or 5 and 5) rows, windows at the edges are missing
        # and interior positions take the neighbor mean.
        # GH 35552
        series = Series(range(1, 6))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 5,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))),
        )
        tm.assert_series_equal(result, expected)

        series = Series(range(1, 5))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 4,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))),
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 6, "b": range(11)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                    ("b", 10),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 5, "b": range(10)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_center_on(self):
        # [FUNCTION DOC]
        # Regression test for GH 37141. Verifies centered grouped rolling mean with a
        # window of 6, minimum periods 1 and on set to a date column gives the expected
        # values with an index of group label and date.
        # GH 37141
        df = DataFrame(
            data={
                "Date": date_range("2020-01-01", "2020-01-10"),
                "gb": ["group_1"] * 6 + ["group_2"] * 4,
                "value": range(10),
            }
        )
        result = (
            df.groupby("gb")
            .rolling(6, on="Date", center=True, min_periods=1)
            .value.mean()
        )
        mi = MultiIndex.from_arrays([df["gb"], df["Date"]], names=["gb", "Date"])
        expected = Series(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5],
            name="value",
            index=mi,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("min_periods", [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods):
        # [FUNCTION DOC]
        # Regression test for GH 36040. Parametrized over minimum periods 5, 4, 3 with
        # window 5 and centered windows over two groups of ten. Verifies the number of
        # missing results at each end of each group equals the larger of zero and
        # minimum periods minus 3, and the interior values match the expected means.
        # GH 36040
        df = DataFrame({"group": ["A"] * 10 + ["B"] * 10, "data": range(20)})

        window_size = 5
        result = (
            df.groupby("group")
            .rolling(window_size, center=True, min_periods=min_periods)
            .mean()
        )
        result = result.reset_index()[["group", "data"]]

        grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean = [x + 10.0 for x in grp_A_mean]

        num_nans = max(0, min_periods - 3)  # For window_size of 5
        nans = [np.nan] * num_nans
        grp_A_expected = nans + grp_A_mean[num_nans : 10 - num_nans] + nans
        grp_B_expected = nans + grp_B_mean[num_nans : 10 - num_nans] + nans

        expected = DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10, "data": grp_A_expected + grp_B_expected}
        )

        tm.assert_frame_equal(result, expected)

    def test_groupby_subselect_rolling(self):
        # [FUNCTION DOC]
        # Regression test for GH 35486. Verifies selecting column b (as a list or as a
        # single name) after grouping by a and rolling with window 2 max returns the
        # frame or series with the expected values and a two-level index.
        # GH 35486
        df = DataFrame(
            {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0], "c": [10, 20, 30, 20]}
        )
        result = df.groupby("a")[["b"]].rolling(2).max()
        expected = DataFrame(
            [np.nan, np.nan, 2.0, np.nan],
            columns=["b"],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
        )
        tm.assert_frame_equal(result, expected)

        result = df.groupby("a")["b"].rolling(2).max()
        expected = Series(
            [np.nan, np.nan, 2.0, np.nan],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
            name="b",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_custom_indexer(self):
        # [FUNCTION DOC]
        # Regression test for GH 35557. Verifies a grouped rolling sum with a custom
        # indexer of window size 3 and minimum periods 1 equals the same operation with
        # an integer window of 3.
        # GH 35557
        class SimpleIndexer(BaseIndexer):
            # [CLASS DOC]
            # Local indexer subclass producing trailing windows of the indexer's window size
            # ending at each position; used only by the test that defines it.
            def get_window_bounds(
                self,
                num_values=0,
                min_periods=None,
                center=None,
                closed=None,
                step=None,
            ):
                # [FUNCTION DOC]
                # Computes window bounds. Inputs: number of values (default 0), minimum periods,
                # center, closed and step (center, closed and step unused). Returns int64 start
                # and end arrays: the end is the position plus one, the start is the end minus
                # the window size, with negative starts clamped to a clamp value that is the
                # window size if minimum periods is none, else zero.
                # KNOWN ISSUE: the local variable named minimum periods is reused as the clamp value for negative starts, which is not a minimum periods concept; when rolling passes an explicit minimum periods the clamp becomes zero, otherwise the window size, so the clamp is a quirk of the test not a semantic of the argument.
                min_periods = self.window_size if min_periods is None else 0
                end = np.arange(num_values, dtype=np.int64) + 1
                start = end - self.window_size
                start[start < 0] = min_periods
                return start, end

        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5
        )
        result = (
            df.groupby(df.index)
            .rolling(SimpleIndexer(window_size=3), min_periods=1)
            .sum()
        )
        expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_subset_with_closed(self):
        # [FUNCTION DOC]
        # Regression test for GH 35549. Verifies a grouped rolling "1D" sum with on set
        # to a date column and closed on the left, selecting column1 afterwards, gives the
        # expected sums for both groups and an index of group label and date.
        # GH 35549
        df = DataFrame(
            {
                "column1": range(8),
                "column2": range(8),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )
        result = (
            df.groupby("group").rolling("1D", on="date", closed="left")["column1"].sum()
        )
        expected = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_agg_namedagg(self):
        # [FUNCTION DOC]
        # Regression test for GH#28333. Verifies named aggregation (output name with
        # column and built-in sum or min function) on a grouped rolling window of 2:
        # total weight and minimum height per window, with missing for the first row of
        # each group.
        # GH#28333
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0, 12.0, 8.0],
                "weight": [7.9, 7.5, 9.9, 198.0, 10.0, 42.0],
            }
        )
        result = (
            df.groupby("kind")
            .rolling(2)
            .agg(
                total_weight=NamedAgg(column="weight", aggfunc=sum),
                min_height=NamedAgg(column="height", aggfunc=min),
            )
        )
        expected = DataFrame(
            {
                "total_weight": [np.nan, 17.8, 19.9, np.nan, 205.5, 240.0],
                "min_height": [np.nan, 9.1, 9.5, np.nan, 6.0, 8.0],
            },
            index=MultiIndex(
                [["cat", "dog"], [0, 1, 2, 3, 4, 5]],
                [[0, 0, 0, 1, 1, 1], [0, 2, 4, 1, 3, 5]],
                names=["kind", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_subset_rolling_subset_with_closed(self):
        # [FUNCTION DOC]
        # Regression test for GH 35549. Same as the closed-left subset test but the
        # grouped object is first reduced to the two columns column1 and date before
        # rolling; the result is the same series.
        # GH 35549
        df = DataFrame(
            {
                "column1": range(8),
                "column2": range(8),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )

        result = (
            df.groupby("group")[["column1", "date"]]
            .rolling("1D", on="date", closed="left")["column1"]
            .sum()
        )
        expected = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_groupby_rolling_index_changed(self, func):
        # [FUNCTION DOC]
        # Regression test for GH: #36018 (index depth change). Parametrized over max and
        # min. Verifies grouping a series with a two-level index by its own values and
        # rolling with window 2 gives a three-level index starting with the group value.
        # GH: #36018 nlevels of MultiIndex changed
        ds = Series(
            [1, 2, 2],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("c", "z")], names=["1", "2"]
            ),
            name="a",
        )

        result = getattr(ds.groupby(ds).rolling(2), func)()
        expected = Series(
            [np.nan, np.nan, 2.0],
            index=MultiIndex.from_tuples(
                [(1, "a", "x"), (2, "a", "y"), (2, "c", "z")], names=["a", "1", "2"]
            ),
            name="a",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_empty_frame(self):
        # [FUNCTION DOC]
        # Regression test for GH 36197 (and 32262, GH-38057). Verifies a grouped rolling
        # sum on an empty frame with one or two group columns returns an empty frame
        # without the grouping columns and with a multi-level index whose levels are
        # empty float (group keys) and integer (position) indexes.
        # GH 36197
        expected = DataFrame({"s1": []})
        result = expected.groupby("s1").rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns="s1")
        # GH-38057 from_tuples gives empty object dtype, we now get float/int levels
        # expected.index = MultiIndex.from_tuples([], names=["s1", None])
        expected.index = MultiIndex.from_product(
            [Index([], dtype="float64"), Index([], dtype="int64")], names=["s1", None]
        )
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"s1": [], "s2": []})
        result = expected.groupby(["s1", "s2"]).rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns=["s1", "s2"])
        expected.index = MultiIndex.from_product(
            [
                Index([], dtype="float64"),
                Index([], dtype="float64"),
                Index([], dtype="int64"),
            ],
            names=["s1", "s2", None],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_string_index(self):
        # [FUNCTION DOC]
        # Regression test for GH: 36727. A frame with a string index, groups and event
        # times is rolled over "10D" on the event-time column; applying a row-count
        # function gives counts of rows in each window and an index of group and original
        # label, with groups in group order.
        # GH: 36727
        df = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9)],
                ["B", "group_1", Timestamp(2019, 1, 2, 9)],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9)],
                ["H", "group_1", Timestamp(2019, 1, 6, 9)],
                ["E", "group_2", Timestamp(2019, 1, 20, 9)],
            ],
            columns=["index", "group", "eventTime"],
        ).set_index("index")

        groups = df.groupby("group")
        df["count_to_date"] = groups.cumcount()
        rolling_groups = groups.rolling("10D", on="eventTime")
        result = rolling_groups.apply(lambda df: df.shape[0])
        expected = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9), 1.0],
                ["B", "group_1", Timestamp(2019, 1, 2, 9), 2.0],
                ["H", "group_1", Timestamp(2019, 1, 6, 9), 3.0],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9), 1.0],
                ["E", "group_2", Timestamp(2019, 1, 20, 9), 1.0],
            ],
            columns=["index", "group", "eventTime", "count_to_date"],
        ).set_index(["group", "index"])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_no_sort(self):
        # [FUNCTION DOC]
        # Regression test for GH 36889. Verifies grouped rolling min with window 1 on
        # data grouped without sorting keeps the group order of first appearance (2
        # before 1) and excludes the grouping column.
        # GH 36889
        result = (
            DataFrame({"foo": [2, 1], "bar": [2, 1]})
            .groupby("foo", sort=False)
            .rolling(1)
            .min()
        )
        expected = DataFrame(
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            columns=["foo", "bar"],
            index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=["foo", None]),
        )
        # GH 32262
        expected = expected.drop(columns="foo")
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_count_closed_on(self, unit):
        # [FUNCTION DOC]
        # Regression test for GH 35869 (time unit fixture). Verifies a grouped "3d"
        # rolling count with closed left on a date column raises the future-deprecation
        # warning that the lowercase day alias is deprecated, and gives counts missing
        # for the first value of each group then ones, with an index of group and date
        # at the given unit.
        # GH 35869
        df = DataFrame(
            {
                "column1": range(6),
                "column2": range(6),
                "group": 3 * ["A", "B"],
                "date": date_range(end="20190101", periods=6, unit=unit),
            }
        )
        msg = "'d' is deprecated and will be removed in a future version."

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = (
                df.groupby("group")
                .rolling("3d", on="date", closed="left")["column1"]
                .count()
            )
        dti = DatetimeIndex(
            [
                "2018-12-27",
                "2018-12-29",
                "2018-12-31",
                "2018-12-28",
                "2018-12-30",
                "2019-01-01",
            ],
            dtype=f"M8[{unit}]",
        )
        mi = MultiIndex.from_arrays(
            [
                ["A", "A", "A", "B", "B", "B"],
                dti,
            ],
            names=["group", "date"],
        )
        expected = Series(
            [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0],
            name="column1",
            index=mi,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        ("func", "kwargs", "expected_values"),
        [
            (
                "rolling",
                {"window": 2, "min_periods": 1},
                [np.nan, 0.5, np.nan, 0.5, 0.5],
            ),
            ("expanding", {}, [np.nan, 0.5, np.nan, 0.5, (1 / 3) ** 0.5]),
        ],
    )
    def test_groupby_rolling_sem(self, func, kwargs, expected_values):
        # [FUNCTION DOC]
        # Regression test for GH: 26476. Parametrized over rolling (window 2, minimum
        # periods 1) and expanding with expected standard error values. Verifies grouped
        # standard error of the mean excludes the grouping column and gives the expected
        # values (missing where fewer than two observations).
        # GH: 26476
        df = DataFrame(
            [["a", 1], ["a", 2], ["b", 1], ["b", 2], ["b", 3]], columns=["a", "b"]
        )
        result = getattr(df.groupby("a"), func)(**kwargs).sem()
        expected = DataFrame(
            {"a": [np.nan] * 5, "b": expected_values},
            index=MultiIndex.from_tuples(
                [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("b", 4)], names=["a", None]
            ),
        )
        # GH 32262
        expected = expected.drop(columns="a")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("rollings", "key"), [({"on": "a"}, "a"), ({"on": None}, "index")]
    )
    def test_groupby_rolling_nans_in_index(self, rollings, key):
        # [FUNCTION DOC]
        # Regression test for GH: 34617. Parametrized over using a column via on or the
        # index. Verifies creating a grouped time-based rolling object when the datetime
        # values contain a missing timestamp raises a value error saying the values must
        # not have NaT (message names the column as a or index).
        # GH: 34617
        df = DataFrame(
            {
                "a": to_datetime(["2020-06-01 12:00", "2020-06-01 14:00", np.nan]),
                "b": [1, 2, 3],
                "c": [1, 1, 1],
            }
        )
        if key == "index":
            df = df.set_index("a")
        with pytest.raises(ValueError, match=f"{key} values must not have NaT"):
            df.groupby("c").rolling("60min", **rollings)

    @pytest.mark.parametrize("group_keys", [True, False])
    def test_groupby_rolling_group_keys(self, group_keys):
        # [FUNCTION DOC]
        # Regression test for GH 37641 and GH 38523. Parametrized over group_keys true
        # and false. Verifies group_keys has no effect on the grouped rolling result,
        # which always repeats the group levels followed by the original index levels
        # (a four-level index with duplicated level names).
        # GH 37641
        # GH 38523: GH 37641 actually was not a bug.
        # group_keys only applies to groupby.apply directly
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        s = Series([1, 2, 3], index=index)
        result = s.groupby(["idx1", "idx2"], group_keys=group_keys).rolling(1).mean()
        expected = Series(
            [1.0, 2.0, 3.0],
            index=MultiIndex.from_tuples(
                [
                    ("val1", "val1", "val1", "val1"),
                    ("val1", "val1", "val1", "val1"),
                    ("val2", "val2", "val2", "val2"),
                ],
                names=["idx1", "idx2", "idx1", "idx2"],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_index_level_and_column_label(self):
        # [FUNCTION DOC]
        # Verifies grouping by an index level name and a column label at once and rolling
        # (window 1, mean) excludes the group column from the output and gives an index
        # of the group keys followed by the original index levels.
        # The groupby keys should not appear as a resulting column
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        df = DataFrame({"A": [1, 1, 2], "B": range(3)}, index=index)
        result = df.groupby(["idx1", "A"]).rolling(1).mean()
        expected = DataFrame(
            {"B": [0.0, 1.0, 2.0]},
            index=MultiIndex.from_tuples(
                [
                    ("val1", 1, "val1", "val1"),
                    ("val1", 1, "val1", "val1"),
                    ("val2", 2, "val2", "val2"),
                ],
                names=["idx1", "A", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_resulting_multiindex(self):
        # [FUNCTION DOC]
        # Verifies grouping by one column yields a two-level index (group key and
        # original position) listing groups in order with their original positions
        # (pull request 38057).
        # a few different cases checking the created MultiIndex of the result
        # https://github.com/pandas-dev/pandas/pull/38057

        # grouping by 1 columns -> 2-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4})
        result = df.groupby("b").rolling(3).mean()
        expected_index = MultiIndex.from_tuples(
            [(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)],
            names=["b", None],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex2(self):
        # [FUNCTION DOC]
        # Verifies grouping by two columns yields a three-level index (the two keys and
        # the original position) with the expected group ordering.
        # grouping by 2 columns -> 3-level MI as result
        df = DataFrame({"a": np.arange(12.0), "b": [1, 2] * 6, "c": [1, 2, 3, 4] * 3})
        result = df.groupby(["b", "c"]).rolling(2).sum()
        expected_index = MultiIndex.from_tuples(
            [
                (1, 1, 0),
                (1, 1, 4),
                (1, 1, 8),
                (1, 3, 2),
                (1, 3, 6),
                (1, 3, 10),
                (2, 2, 1),
                (2, 2, 5),
                (2, 2, 9),
                (2, 4, 3),
                (2, 4, 7),
                (2, 4, 11),
            ],
            names=["b", "c", None],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex3(self):
        # [FUNCTION DOC]
        # Verifies grouping with one key on a frame already having a two-level index
        # yields a three-level index (key followed by the two original levels), compared
        # with equivalent-type checking.
        # grouping with 1 level on dataframe with 2-level MI -> 3-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4, "c": [1, 2, 3, 4] * 2})
        df = df.set_index("c", append=True)
        result = df.groupby("b").rolling(3).mean()
        expected_index = MultiIndex.from_tuples(
            [
                (1, 0, 1),
                (1, 2, 3),
                (1, 4, 1),
                (1, 6, 3),
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 2),
                (2, 7, 4),
            ],
            names=["b", None, "c"],
        )
        tm.assert_index_equal(result.index, expected_index, exact="equiv")

    def test_groupby_rolling_object_doesnt_affect_groupby_apply(self, roll_frame):
        # [FUNCTION DOC]
        # Regression test for GH 39732. Verifies that merely creating a grouped rolling
        # object does not change the index produced by a later group-wise apply on the
        # same grouped object.
        # GH 39732
        g = roll_frame.groupby("A", group_keys=False)
        expected = g.apply(lambda x: x.rolling(4).sum()).index
        _ = g.rolling(window=4)
        result = g.apply(lambda x: x.rolling(4).sum()).index
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        ("window", "min_periods", "closed", "expected"),
        [
            (2, 0, "left", [None, 0.0, 1.0, 1.0, None, 0.0, 1.0, 1.0]),
            (2, 2, "left", [None, None, 1.0, 1.0, None, None, 1.0, 1.0]),
            (4, 4, "left", [None, None, None, None, None, None, None, None]),
            (4, 4, "right", [None, None, None, 5.0, None, None, None, 5.0]),
        ],
    )
    def test_groupby_rolling_var(self, window, min_periods, closed, expected):
        # [FUNCTION DOC]
        # Parametrized over window, minimum periods, closed side and expected variances
        # (zero delta degrees of freedom). Groups alternate over eight rows in two
        # groups; verifies grouped rolling variance equals the expected values, indexed
        # by group key and original position.
        df = DataFrame([1, 2, 3, 4, 5, 6, 7, 8])
        result = (
            df.groupby([1, 2, 1, 2, 1, 2, 1, 2])
            .rolling(window=window, min_periods=min_periods, closed=closed)
            .var(0)
        )
        expected_result = DataFrame(
            np.array(expected, dtype="float64"),
            index=MultiIndex(
                levels=[np.array([1, 2]), [0, 1, 2, 3, 4, 5, 6, 7]],
                codes=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 2, 4, 6, 1, 3, 5, 7]],
            ),
        )
        tm.assert_frame_equal(result, expected_result)

    @pytest.mark.parametrize(
        "columns", [MultiIndex.from_tuples([("A", ""), ("B", "C")]), ["A", "B"]]
    )
    def test_by_column_not_in_values(self, columns):
        # [FUNCTION DOC]
        # Regression test for GH 32262. Parametrized over a two-level column index and a
        # plain column list. Verifies a grouped rolling sum excludes the grouping column
        # from the result columns and does not modify the grouped object's data.
        # GH 32262
        df = DataFrame([[1, 0]] * 20 + [[2, 0]] * 12 + [[3, 0]] * 8, columns=columns)
        g = df.groupby("A")
        original_obj = g.obj.copy(deep=True)
        r = g.rolling(4)
        result = r.sum()
        assert "A" not in result.columns
        tm.assert_frame_equal(g.obj, original_obj)

    def test_groupby_level(self):
        # [FUNCTION DOC]
        # Regression test for GH 38523 and 38787. Verifies grouping by index level 0 and
        # rolling a selected column with window 2 gives a three-level index whose first
        # two levels are both named Animal, with sums of consecutive pairs.
        # GH 38523, 38787
        arrays = [
            ["Falcon", "Falcon", "Parrot", "Parrot"],
            ["Captive", "Wild", "Captive", "Wild"],
        ]
        index = MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
        df = DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
        result = df.groupby(level=0)["Max Speed"].rolling(2).sum()
        expected = Series(
            [np.nan, 740.0, np.nan, 50.0],
            index=MultiIndex.from_tuples(
                [
                    ("Falcon", "Falcon", "Captive"),
                    ("Falcon", "Falcon", "Wild"),
                    ("Parrot", "Parrot", "Captive"),
                    ("Parrot", "Parrot", "Wild"),
                ],
                names=["Animal", "Animal", "Type"],
            ),
            name="Max Speed",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "by, expected_data",
        [
            [["id"], {"num": [100.0, 150.0, 150.0, 200.0]}],
            [
                ["id", "index"],
                {
                    "date": [
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                    ],
                    "num": [100.0, 200.0, 150.0, 250.0],
                },
            ],
        ],
    )
    def test_as_index_false(self, by, expected_data, unit):
        # [FUNCTION DOC]
        # Regression test for GH 39433 (time unit fixture). Parametrized over grouping
        # by one or two keys (id; id and date index) with expected data. Verifies a
        # grouped rolling mean (window 2, minimum periods 1) with as_index false keeps
        # the original index and returns the group key columns as regular columns.
        # GH 39433
        data = [
            ["A", "2018-01-01", 100.0],
            ["A", "2018-01-02", 200.0],
            ["B", "2018-01-01", 150.0],
            ["B", "2018-01-02", 250.0],
        ]
        df = DataFrame(data, columns=["id", "date", "num"])
        df["date"] = df["date"].astype(f"M8[{unit}]")
        df = df.set_index(["date"])

        gp_by = [getattr(df, attr) for attr in by]
        result = (
            df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
        )

        expected = {"id": ["A", "A", "B", "B"]}
        expected.update(expected_data)
        expected = DataFrame(
            expected,
            index=df.index,
        )
        if "date" in expected_data:
            expected["date"] = expected["date"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(result, expected)

    def test_nan_and_zero_endpoints(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Regression test for an old issue (twosigma pandas 53), parametrized over
        # integer dtypes. A 1000-row series of a huge constant with a missing first value
        # and zero last value, grouped by an index with 999 zeros and a single one:
        # verifies a grouped rolling mean (window 10, minimum periods 1) preserves the
        # values without numerical drift and with an index made of the group key twice.
        # https://github.com/twosigma/pandas/issues/53
        typ = np.dtype(any_int_numpy_dtype).type
        size = 1000
        idx = np.repeat(typ(0), size)
        idx[-1] = 1

        val = 5e25
        arr = np.repeat(val, size)
        arr[0] = np.nan
        arr[-1] = 0

        df = DataFrame(
            {
                "index": idx,
                "adl2": arr,
            }
        ).set_index("index")
        result = df.groupby("index")["adl2"].rolling(window=10, min_periods=1).mean()
        expected = Series(
            arr,
            name="adl2",
            index=MultiIndex.from_arrays(
                [
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                ],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_non_monotonic(self):
        # [FUNCTION DOC]
        # Regression test for GH 43909. Verifies that creating a grouped time-based
        # rolling object over a non-monotonic on column raises a value error saying it
        # must be monotonic.
        # GH 43909

        shuffled = [3, 0, 1, 2]
        sec = 1_000
        df = DataFrame(
            [{"t": Timestamp(2 * x * sec), "x": x + 1, "c": 42} for x in shuffled]
        )
        with pytest.raises(ValueError, match=r".* must be monotonic"):
            df.groupby("c").rolling(on="t", window="3s")

    def test_groupby_monotonic(self):
        # [FUNCTION DOC]
        # Regression test for GH 15130. With date-sorted data over three people, verifies
        # a grouped "180D" rolling sum over the date column equals the group-wise rolling
        # sum on a date-indexed frame.
        # GH 15130
        # we don't need to validate monotonicity when grouping

        # GH 43909 we should raise an error here to match
        # behaviour of non-groupby rolling.

        data = [
            ["David", "1/1/2015", 100],
            ["David", "1/5/2015", 500],
            ["David", "5/30/2015", 50],
            ["David", "7/25/2015", 50],
            ["Ryan", "1/4/2014", 100],
            ["Ryan", "1/19/2015", 500],
            ["Ryan", "3/31/2016", 50],
            ["Joe", "7/1/2015", 100],
            ["Joe", "9/9/2015", 500],
            ["Joe", "10/15/2015", 50],
        ]

        df = DataFrame(data=data, columns=["name", "date", "amount"])
        df["date"] = to_datetime(df["date"])
        df = df.sort_values("date")

        expected = (
            df.set_index("date")
            .groupby("name")
            .apply(lambda x: x.rolling("180D")["amount"].sum())
        )
        result = df.groupby("name").rolling("180D", on="date")["amount"].sum()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_monotonic_within_each_group(self):
        # [FUNCTION DOC]
        # Regression test for GH 13966 and GH 46061. Verifies that an on column that
        # repeats its dates in two parts (monotonic within each group but not overall)
        # is accepted and gives the same result as group-wise rolling "4s" means.
        # GH 13966 (similar to #15130, closed by #15175)

        # superseded by 43909
        # GH 46061: OK if the on is monotonic relative to each each group

        dates = date_range(start="2016-01-01 09:30:00", periods=20, freq="s")
        df = DataFrame(
            {
                "A": [1] * 20 + [2] * 12 + [3] * 8,
                "B": np.concatenate((dates, dates)),
                "C": np.arange(40),
            }
        )

        expected = (
            df.set_index("B").groupby("A").apply(lambda x: x.rolling("4s")["C"].mean())
        )
        result = df.groupby("A").rolling("4s", on="B").C.mean()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_not_monotonic_within_each_group(self):
        # [FUNCTION DOC]
        # Regression test for GH 46061. Verifies creating a grouped time-based rolling
        # object whose on column is not monotonic inside a group raises a value error
        # saying each group within B must be monotonic.
        # GH 46061
        df = DataFrame(
            {
                "A": [1] * 3 + [2] * 3,
                "B": [Timestamp(year, 1, 1) for year in [2020, 2021, 2019]] * 2,
                "C": range(6),
            }
        )
        with pytest.raises(ValueError, match="Each group within B must be monotonic."):
            df.groupby("A").rolling("365D", on="B")


class TestExpanding:
    # [CLASS DOC]
    # Tests for grouped expanding windows. Mirrors the grouped rolling checks by
    # comparing with group-wise application. Defines its own frame fixture that
    # overrides the shared one. No state.
    @pytest.fixture
    def frame(self):
        # [FUNCTION DOC]
        # Fixture local to this class returning a 40-row frame with group column A (20
        # ones, 12 twos, 8 threes) and column B holding the integers 0 to 39. It
        # shadows the random-data frame fixture of the shared conftest.
        return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "min",
            "max",
            "first",
            "last",
            "count",
            "kurt",
            "skew",
            "nunique",
        ],
    )
    def test_expanding(self, f, frame):
        # [FUNCTION DOC]
        # Parametrized over ten methods (sum, mean, min, max, first, last, count,
        # kurtosis, skewness, unique count). Verifies grouped expanding equals the
        # group-wise expanding result with the reconstructed index of group label and
        # original position (GH 39732).
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.expanding(), f)())
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_expanding_ddof(self, f, frame):
        # [FUNCTION DOC]
        # Parametrized over std and var with delta degrees of freedom 0. Same check as
        # the method test.
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result = getattr(r, f)(ddof=0)
        expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_expanding_quantile(self, interpolation, frame):
        # [FUNCTION DOC]
        # Parametrized over five interpolation methods. Verifies the grouped expanding
        # quantile 0.4 equals the group-wise result with the reconstructed index.
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result = r.quantile(0.4, interpolation=interpolation)
        expected = g.apply(
            lambda x: x.expanding().quantile(0.4, interpolation=interpolation)
        )
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_expanding_corr_cov(self, f, frame):
        # [FUNCTION DOC]
        # Parametrized over corr and cov. Verifies grouped expanding results against the
        # full frame equal the group-wise application except that selected positions
        # (20 to 60 and 72 to 112 of the second column) are set missing, because the
        # group-wise apply returns 1 rather than missing for all-missing windows (GH
        # 39591), and the grouping column is set missing (apply inserts zeros). Also
        # verifies the pairwise form of column B equals the group-wise application.
        g = frame.groupby("A")
        r = g.expanding()

        result = getattr(r, f)(frame)

        def func_0(x):
            # [FUNCTION DOC]
            # HELPER: nested function applied per group. Takes a group frame x and returns
            # the expanding correlation or covariance (as chosen by the enclosing test) of x
            # against the full frame.
            return getattr(x.expanding(), f)(frame)

        expected = g.apply(func_0)
        # GH 39591: groupby.apply returns 1 instead of nan for windows
        # with all nan values
        null_idx = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan
        # GH 39591: The grouped column should be all np.nan
        # (groupby.apply inserts 0s for cov)
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

        result = getattr(r.B, f)(pairwise=True)

        def func_1(x):
            # [FUNCTION DOC]
            # HELPER: nested function applied per group. Takes a group frame x and returns
            # the pairwise expanding correlation or covariance of its column B.
            return getattr(x.B.expanding(), f)(pairwise=True)

        expected = g.apply(func_1)
        tm.assert_series_equal(result, expected)

    def test_expanding_apply(self, raw, frame):
        # [FUNCTION DOC]
        # Verifies grouped expanding apply of a sum function (raw fixture) equals the
        # group-wise apply with the reconstructed index (GH 39732).
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        # reduction
        result = r.apply(lambda x: x.sum(), raw=raw)
        expected = g.apply(lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw))
        # GH 39732
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_groupby_expanding_agg_namedagg(self):
        # [FUNCTION DOC]
        # Regression test for GH#28333. Verifies named aggregation (total weight using
        # the built-in sum, minimum height using the built-in min) on a grouped
        # expanding window with minimum periods 1 gives running totals and minima per
        # group with a two-level index.
        # GH#28333
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0, 12.0, 8.0],
                "weight": [7.9, 7.5, 9.9, 198.0, 10.0, 42.0],
            }
        )
        result = (
            df.groupby("kind")
            .expanding(1)
            .agg(
                total_weight=NamedAgg(column="weight", aggfunc=sum),
                min_height=NamedAgg(column="height", aggfunc=min),
            )
        )
        expected = DataFrame(
            {
                "total_weight": [7.9, 17.8, 27.8, 7.5, 205.5, 247.5],
                "min_height": [9.1, 9.1, 9.1, 6.0, 6.0, 6.0],
            },
            index=MultiIndex(
                [["cat", "dog"], [0, 1, 2, 3, 4, 5]],
                [[0, 0, 0, 1, 1, 1], [0, 2, 4, 1, 3, 5]],
                names=["kind", None],
            ),
        )
        tm.assert_frame_equal(result, expected)


class TestEWM:
    # [CLASS DOC]
    # Tests for grouped exponentially weighted windows (mean, standard deviation,
    # variance, correlation, covariance, named aggregation, times). No state.
    @pytest.mark.parametrize(
        "method, expected_data",
        [
            ["mean", [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666]],
            ["std", [np.nan, 0.707107, 0.963624, 1.177164]],
            ["var", [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]],
        ],
    )
    def test_methods(self, method, expected_data):
        # [FUNCTION DOC]
        # Regression test for GH 16037. Parametrized over mean, std and var with
        # expected values. Verifies grouped ewm (center of mass 1.0) on a single group
        # of four values gives the expected series with a two-level index.
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_ewm_agg_namedagg(self):
        # [FUNCTION DOC]
        # Regression test for GH#28333. Verifies named aggregation of mean, standard
        # deviation and variance of column B for a grouped ewm (center of mass 1.0)
        # gives the same numbers as the individual methods.
        # GH#28333
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = (
            df.groupby("A")
            .ewm(com=1.0)
            .agg(
                B_mean=NamedAgg(column="B", aggfunc="mean"),
                B_std=NamedAgg(column="B", aggfunc="std"),
                B_var=NamedAgg(column="B", aggfunc="var"),
            )
        )
        expected = DataFrame(
            {
                "B_mean": [
                    0.0,
                    0.6666666666666666,
                    1.4285714285714286,
                    2.2666666666666666,
                ],
                "B_std": [np.nan, 0.707107, 0.963624, 1.177164],
                "B_var": [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857],
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected_data",
        [["corr", [np.nan, 1.0, 1.0, 1]], ["cov", [np.nan, 0.5, 0.928571, 1.385714]]],
    )
    def test_pairwise_methods(self, method, expected_data):
        # [FUNCTION DOC]
        # Regression test for GH 16037. Parametrized over corr and cov with expected
        # values. Verifies the pairwise grouped ewm result has a three-level index
        # (group, position, column name) with the expected values, and equals the
        # group-wise application on the selected column.
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0, "B"),
                    ("a", 1, "B"),
                    ("a", 2, "B"),
                    ("a", 3, "B"),
                ],
                names=["A", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

        expected = df.groupby("A")[["B"]].apply(
            lambda x: getattr(x.ewm(com=1.0), method)()
        )
        tm.assert_frame_equal(result, expected)

    def test_times(self, times_frame):
        # [FUNCTION DOC]
        # Regression test for GH 40951 (and GH#42738). Using the times frame fixture with
        # its datetime column removed and passed as times, verifies a grouped ewm mean
        # with a 23 days half-life gives the expected values per group and original
        # position.
        # GH 40951
        halflife = "23 days"
        # GH#42738
        times = times_frame.pop("C")
        result = times_frame.groupby("A").ewm(halflife=halflife, times=times).mean()
        expected = DataFrame(
            {
                "B": [
                    0.0,
                    0.507534,
                    1.020088,
                    1.537661,
                    0.0,
                    0.567395,
                    1.221209,
                    0.0,
                    0.653141,
                    1.195003,
                ]
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 3),
                    ("a", 6),
                    ("a", 9),
                    ("b", 1),
                    ("b", 4),
                    ("b", 7),
                    ("c", 2),
                    ("c", 5),
                    ("c", 8),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_times_array(self, times_frame):
        # [FUNCTION DOC]
        # Regression test for GH 40951. Verifies passing times as a series gives the same
        # grouped ewm mean as passing the underlying array.
        # GH 40951
        halflife = "23 days"
        times = times_frame.pop("C")
        gb = times_frame.groupby("A")
        result = gb.ewm(halflife=halflife, times=times).mean()
        expected = gb.ewm(halflife=halflife, times=times.values).mean()
        tm.assert_frame_equal(result, expected)

    def test_dont_mutate_obj_after_slicing(self):
        # [FUNCTION DOC]
        # Regression test for GH 43355. With a grouped hourly rolling "1h" object on a
        # timestamp column, verifies counts for the frame, then for a selected column
        # (indexed by group and timestamp), and again for the frame, which must be
        # unchanged by the earlier slicing.
        # GH 43355
        df = DataFrame(
            {
                "id": ["a", "a", "b", "b", "b"],
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": range(5),
            }
        )
        grp = df.groupby("id").rolling("1h", on="timestamp")
        result = grp.count()
        expected_df = DataFrame(
            {
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": [1.0] * 5,
            },
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "b", "b"], list(range(5))], names=["id", None]
            ),
        )
        tm.assert_frame_equal(result, expected_df)

        result = grp["y"].count()
        expected_series = Series(
            [1.0] * 5,
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "b", "b", "b"],
                    date_range("2021-9-1", periods=5, freq="h"),
                ],
                names=["id", "timestamp"],
            ),
            name="y",
        )
        tm.assert_series_equal(result, expected_series)
        # This is the key test
        result = grp.count()
        tm.assert_frame_equal(result, expected_df)


def test_rolling_corr_with_single_integer_in_index():
    # [FUNCTION DOC]
    # Regression test for GH 44078. Grouping by a column of one-element tuples,
    # verifies grouped rolling correlation (window 2) against the whole frame gives
    # missing for the first value and 1.0 afterwards, indexed by the tuple key and
    # position.
    # GH 44078
    df = DataFrame({"a": [(1,), (1,), (1,)], "b": [4, 5, 6]})
    gb = df.groupby(["a"])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples([((1,), 0), ((1,), 1), ((1,), 2)], names=["a", None])
    expected = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_corr_with_tuples_in_index():
    # [FUNCTION DOC]
    # Regression test for GH 44078. Same as the single-integer tuple key case but
    # with two-element tuple keys.
    # GH 44078
    df = DataFrame(
        {
            "a": [
                (
                    1,
                    2,
                ),
                (
                    1,
                    2,
                ),
                (
                    1,
                    2,
                ),
            ],
            "b": [4, 5, 6],
        }
    )
    gb = df.groupby(["a"])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples(
        [((1, 2), 0), ((1, 2), 1), ((1, 2), 2)], names=["a", None]
    )
    expected = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)
