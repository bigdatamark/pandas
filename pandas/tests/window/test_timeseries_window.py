# [MODULE DOC]
# Tests for time-based (offset window) rolling on frames and series indexed by
# timestamps (GH13327): validation of the window, the on argument and index
# monotonicity, closed-side options, regular and irregularly spaced data for
# each aggregation, agreement with integer windows, frequency variants,
# decreasing indexes, empty input, multi-index levels, missing timestamps and
# arrow-backed datetime indexes. Defines two fixtures (a regularly and an
# irregularly spaced second-resolution frame) and one test class plus two
# module-level tests.
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm

from pandas.tseries import offsets


@pytest.fixture
def regular():
    # [FUNCTION DOC]
    # Fixture returning a frame with column B holding 0 to 4, indexed (and named A)
    # by five consecutive seconds starting at 2013-01-01 00:00:00.
    return DataFrame(
        {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
    ).set_index("A")


@pytest.fixture
def ragged():
    # [FUNCTION DOC]
    # Fixture returning a frame with column B holding 0 to 4 and an irregular
    # timestamp index on 2013-01-01 at 09:00:00, :02, :03, :05 and :06.
    df = DataFrame({"B": range(5)})
    df.index = [
        Timestamp("20130101 09:00:00"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:03"),
        Timestamp("20130101 09:00:05"),
        Timestamp("20130101 09:00:06"),
    ]
    return df


class TestRollingTS:
    # [CLASS DOC]
    # Test class for rolling with time offset windows (xref GH13327). No state; uses
    # the regular and ragged fixtures to build inputs and compares against integer
    # window computations or hand-computed results.
    # rolling time-series friendly
    # xref GH13327

    def test_doc_string(self):
        # [FUNCTION DOC]
        # Smoke test of the documentation example: a frame with a missing value and an
        # irregular timestamp index, rolled with a "2s" window and summed. Asserts
        # nothing; passes if no exception is raised.
        df = DataFrame(
            {"B": [0, 1, 2, np.nan, 4]},
            index=[
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ],
        )
        df
        df.rolling("2s").sum()

    def test_invalid_window_non_int(self, regular):
        # [FUNCTION DOC]
        # Verifies a nonsense string window on a datetime-indexed frame raises a value
        # error saying the window is not compatible with a datetimelike index, and on a
        # frame with a default integer index raises a value error saying the window must
        # be an integer.
        # not a valid freq
        msg = "passed window foobar is not compatible with a datetimelike index"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="foobar")
        # not a datetimelike index
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            regular.reset_index().rolling(window="foobar")

    @pytest.mark.parametrize("freq", ["2MS", offsets.MonthBegin(2)])
    def test_invalid_window_nonfixed(self, freq, regular):
        # [FUNCTION DOC]
        # Parametrized over a month-begin frequency given as string and as offset
        # object. Verifies a value error saying the (two month-begins) offset is a
        # non-fixed frequency.
        # non-fixed freqs
        msg = "\\<2 \\* MonthBegins\\> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window=freq)

    @pytest.mark.parametrize("freq", ["1D", offsets.Day(2), "2ms"])
    def test_valid_window(self, freq, regular):
        # [FUNCTION DOC]
        # Parametrized over "1D", a two-day offset and "2ms". Verifies creating a
        # rolling object with these fixed-frequency windows does not raise.
        regular.rolling(window=freq)

    @pytest.mark.parametrize("minp", [1.0, "foo", np.array([1, 2, 3])])
    def test_invalid_minp(self, minp, regular):
        # [FUNCTION DOC]
        # Parametrized over invalid minimum periods (float, string, array). Verifies a
        # value error is raised for a time-based window (either saying minimum periods
        # must be an integer, or the legacy message about an unbound local variable).
        # non-integer min_periods
        msg = (
            r"local variable 'minp' referenced before assignment|"
            "min_periods must be an integer"
        )
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="1D", min_periods=minp)

    def test_on(self, regular):
        # [FUNCTION DOC]
        # Verifies the on argument of rolling: an unknown column name raises a value
        # error saying on must be a column of the frame, an index or none; a datetime
        # column is accepted; an integer column with a string window raises a value
        # error saying the window must be an integer; and aggregating a single other
        # column works with on set to a column outside the selection.
        df = regular

        # not a valid column
        msg = (
            r"invalid on specified as foobar, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2s", on="foobar")

        # column is valid
        df = df.copy()
        df["C"] = date_range("20130101", periods=len(df))
        df.rolling(window="2D", on="C").sum()

        # invalid columns
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2d", on="B")

        # ok even though on non-selected
        df.rolling(window="2D", on="C").B.sum()

    def test_monotonic_on(self):
        # [FUNCTION DOC]
        # Verifies a monotonic increasing datetime column used via on, and a monotonic
        # index, both allow time-based rolling sums without error.
        # on/index must be monotonic
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )

        assert df.A.is_monotonic_increasing
        df.rolling("2s", on="A").sum()

        df = df.set_index("A")
        assert df.index.is_monotonic_increasing
        df.rolling("2s").sum()

    def test_non_monotonic_on(self):
        # [FUNCTION DOC]
        # Regression test for GH 19248. Verifies a non-monotonic datetime index makes a
        # time-based rolling sum raise a value error saying index values must be
        # monotonic, and that after resetting the index the datetime column (named A)
        # used via on raises a value error saying on must be a column of the frame, an
        # index or none.
        # GH 19248
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )
        df = df.set_index("A")
        non_monotonic_index = df.index.to_list()
        non_monotonic_index[0] = non_monotonic_index[3]
        df.index = non_monotonic_index

        assert not df.index.is_monotonic_increasing

        msg = "index values must be monotonic"
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s").sum()

        df = df.reset_index()

        msg = (
            r"invalid on specified as A, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s", on="A").sum()

    def test_frame_on(self):
        # [FUNCTION DOC]
        # Verifies rolling "2s" using a datetime column through on equals setting that
        # column as the index, rolling, and resetting the index (for a single selected
        # column and for a column list result, where the column order differs from the
        # simulated approach and is switched in the expected frame).
        df = DataFrame(
            {"B": range(5), "C": date_range("20130101 09:00:00", periods=5, freq="3s")}
        )

        df["A"] = [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130101 09:00:02"),
            Timestamp("20130101 09:00:03"),
            Timestamp("20130101 09:00:05"),
            Timestamp("20130101 09:00:06"),
        ]

        # we are doing simulating using 'on'
        expected = df.set_index("A").rolling("2s").B.sum().reset_index(drop=True)

        result = df.rolling("2s", on="A").B.sum()
        tm.assert_series_equal(result, expected)

        # test as a frame
        # we should be ignoring the 'on' as an aggregation column
        # note that the expected is setting, computing, and resetting
        # so the columns need to be switched compared
        # to the actual result where they are ordered as in the
        # original
        expected = (
            df.set_index("A").rolling("2s")[["B"]].sum().reset_index()[["B", "A"]]
        )

        result = df.rolling("2s", on="A")[["B"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_frame_on2(self, unit):
        # [FUNCTION DOC]
        # Uses the time unit fixture. With a frame with integer, float and datetime
        # columns, verifies a "2s" rolling sum on the datetime column sums the
        # non-datetime columns (excluding the on column from aggregation but keeping it
        # in the result in its original position), that selecting one column gives the
        # expected series, and that selecting all columns including the on column
        # keeps it in the result.
        # using multiple aggregation columns
        dti = DatetimeIndex(
            [
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ]
        ).as_unit(unit)
        df = DataFrame(
            {
                "A": [0, 1, 2, 3, 4],
                "B": [0, 1, 2, np.nan, 4],
                "C": dti,
            },
            columns=["A", "C", "B"],
        )

        expected1 = DataFrame(
            {"A": [0.0, 1, 3, 3, 7], "B": [0, 1, 3, np.nan, 4], "C": df["C"]},
            columns=["A", "C", "B"],
        )

        result = df.rolling("2s", on="C").sum()
        expected = expected1
        tm.assert_frame_equal(result, expected)

        expected = Series([0, 1, 3, np.nan, 4], name="B")
        result = df.rolling("2s", on="C").B.sum()
        tm.assert_series_equal(result, expected)

        expected = expected1[["A", "B", "C"]]
        result = df.rolling("2s", on="C")[["A", "B", "C"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_basic_regular(self, regular):
        # [FUNCTION DOC]
        # Verifies on daily and every-second-day indexed frames that a one-day (or
        # two-day) offset window equals an integer window of one observation with
        # minimum periods one, and that without minimum periods the same holds.
        df = regular.copy()

        df.index = date_range("20130101", periods=5, freq="D")
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="1D").sum()
        tm.assert_frame_equal(result, expected)

        df.index = date_range("20130101", periods=5, freq="2D")
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(window=1).sum()
        result = df.rolling(window="2D").sum()
        tm.assert_frame_equal(result, expected)

    def test_min_periods(self, regular):
        # [FUNCTION DOC]
        # Verifies that a "2s" window on a regular one-second frame equals an integer
        # window of 2 with minimum periods one, with or without explicit minimum periods
        # of one.
        # compare for min_periods
        df = regular

        # these slightly different
        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_closed(self, regular, unit):
        # [FUNCTION DOC]
        # Regression test for GH13965 (time unit fixture). An invalid closed value raises
        # a value error listing the allowed values. For five irregular timestamps of ones,
        # verifies a "2s" rolling sum for each closed option: right (also the default)
        # gives 1, 2, 2, 2, 1; both gives 1, 2, 3, 3, 2; left gives missing, 1, 2, 2, 1;
        # neither gives missing, 1, 1, 1, missing.
        # xref GH13965

        dti = DatetimeIndex(
            [
                Timestamp("20130101 09:00:01"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:04"),
                Timestamp("20130101 09:00:06"),
            ]
        ).as_unit(unit)

        df = DataFrame(
            {"A": [1] * 5},
            index=dti,
        )

        # closed must be 'right', 'left', 'both', 'neither'
        msg = "closed must be 'right', 'left', 'both' or 'neither'"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="2s", closed="blabla")

        expected = df.copy()
        expected["A"] = [1.0, 2, 2, 2, 1]
        result = df.rolling("2s", closed="right").sum()
        tm.assert_frame_equal(result, expected)

        # default should be 'right'
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [1.0, 2, 3, 3, 2]
        result = df.rolling("2s", closed="both").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 2, 2, 1]
        result = df.rolling("2s", closed="left").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 1, 1, np.nan]
        result = df.rolling("2s", closed="neither").sum()
        tm.assert_frame_equal(result, expected)

    def test_ragged_sum(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling sums on irregular timestamps for windows of 1s to
        # 5s with various minimum periods against hand-computed values (entries are
        # missing where fewer observations than the minimum periods lie in the window;
        # the default minimum periods for time windows is 1).
        df = ragged
        result = df.rolling(window="1s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 3, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=2).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, np.nan, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s").sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="4s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="4s", min_periods=3).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 10]
        tm.assert_frame_equal(result, expected)

    def test_ragged_mean(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling mean on irregular timestamps for 1s and 2s
        # windows against hand-computed values.
        df = ragged
        result = df.rolling(window="1s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_median(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling median on irregular timestamps for 1s and 2s
        # windows against hand-computed values.
        df = ragged
        result = df.rolling(window="1s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_quantile(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling quantile 0.5 on irregular timestamps for 1s and
        # 2s windows against hand-computed values.
        df = ragged
        result = df.rolling(window="1s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_std(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling standard deviation on irregular timestamps for
        # different windows and delta degrees of freedom: ddof 0 and window 1s gives
        # zeros; ddof 1 and window 1s gives missing; ddof 0 and 3s gives zero then 0.5;
        # ddof 1 and 5s gives hand-computed values.
        df = ragged
        result = df.rolling(window="1s", min_periods=1).std(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="1s", min_periods=1).std(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).std(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] + [0.5] * 4
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).std(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan, 0.707107, 1.0, 1.0, 1.290994]
        tm.assert_frame_equal(result, expected)

    def test_ragged_var(self, ragged):
        # [FUNCTION DOC]
        # Variance counterpart of the standard deviation test: zeros, missing, zero then
        # 0.25, and hand-computed unbiased values for 5s.
        df = ragged
        result = df.rolling(window="1s", min_periods=1).var(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="1s", min_periods=1).var(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).var(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] + [0.25] * 4
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).var(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan, 0.5, 1.0, 1.0, 1 + 2 / 3.0]
        tm.assert_frame_equal(result, expected)

    def test_ragged_skew(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling skewness on irregular timestamps: 3s window gives
        # all missing; 5s window gives two missing followed by zeros.
        df = ragged
        result = df.rolling(window="3s", min_periods=1).skew()
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).skew()
        expected = df.copy()
        expected["B"] = [np.nan] * 2 + [0.0, 0.0, 0.0]
        tm.assert_frame_equal(result, expected)

    def test_ragged_kurt(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling kurtosis on irregular timestamps: 3s window gives
        # all missing; 5s window gives four missing and then -1.2.
        df = ragged
        result = df.rolling(window="3s", min_periods=1).kurt()
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).kurt()
        expected = df.copy()
        expected["B"] = [np.nan] * 4 + [-1.2]
        tm.assert_frame_equal(result, expected)

    def test_ragged_count(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling count on irregular timestamps: 1s window gives
        # ones (with or without explicit minimum periods); 2s window with minimum
        # periods one gives 1, 1, 2, 1, 2; with minimum periods two gives missing where
        # the count is below two.
        df = ragged
        result = df.rolling(window="1s", min_periods=1).count()
        expected = df.copy()
        expected["B"] = [1.0, 1, 1, 1, 1]
        tm.assert_frame_equal(result, expected)

        df = ragged
        result = df.rolling(window="1s").count()
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).count()
        expected = df.copy()
        expected["B"] = [1.0, 1, 2, 1, 2]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=2).count()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 2, np.nan, 2]
        tm.assert_frame_equal(result, expected)

    def test_regular_min(self):
        # [FUNCTION DOC]
        # Verifies time-window rolling min on regular one-second data: a 1s window of
        # increasing values returns the values; for the values 5, 4, 3, 4, 5, a 2s
        # window gives 5, 4, 3, 3, 4 and a 5s window gives 5, 4, 3, 3, 3.
        # KNOWN ISSUE: after the second frame is created, the first result and expected are asserted equal again before the new result is computed; this repeated assertion adds nothing.
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [0.0, 1, 2, 3, 4]}
        ).set_index("A")
        result = df.rolling("1s").min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [5, 4, 3, 4, 5]}
        ).set_index("A")

        tm.assert_frame_equal(result, expected)
        result = df.rolling("2s").min()
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling("5s").min()
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 3]
        tm.assert_frame_equal(result, expected)

    def test_ragged_min(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling min on irregular timestamps for 1s, 2s and 5s
        # windows against hand-computed values.
        df = ragged

        result = df.rolling(window="1s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1, 3, 3]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 0, 0, 1, 1]
        tm.assert_frame_equal(result, expected)

    def test_perf_min(self):
        # [FUNCTION DOC]
        # Performance-style check over 10000 random values at one-second spacing: the
        # time-window min with "2s" and "200s" windows must be close to the integer
        # window min of 2 and 200 observations with minimum periods one.
        # KNOWN ISSUE: the comparison tests that the difference is below a tolerance without taking the absolute value, so results smaller than expected would pass; it is a one-sided check.
        N = 10000

        dfp = DataFrame(
            {"B": np.random.default_rng(2).standard_normal(N)},
            index=date_range("20130101", periods=N, freq="s"),
        )
        expected = dfp.rolling(2, min_periods=1).min()
        result = dfp.rolling("2s").min()
        assert ((result - expected) < 0.01).all().all()

        expected = dfp.rolling(200, min_periods=1).min()
        result = dfp.rolling("200s").min()
        assert ((result - expected) < 0.01).all().all()

    def test_ragged_max(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling max on irregular timestamps for 1s, 2s and 5s
        # windows: the result equals the original values because the data is increasing.
        df = ragged

        result = df.rolling(window="1s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_ragged_first(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling first on irregular timestamps for 1s, 2s and 5s
        # windows against hand-computed values (the first value inside each window).
        df = ragged

        result = df.rolling(window="1s", min_periods=1).first()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).first()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1, 3, 3]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).first()
        expected = df.copy()
        expected["B"] = [0.0, 0, 0, 1, 1]
        tm.assert_frame_equal(result, expected)

    def test_ragged_last(self, ragged):
        # [FUNCTION DOC]
        # Verifies time-window rolling last on irregular timestamps for 1s, 2s and 5s
        # windows: the result equals the original values (the last value inside each
        # window is the current one).
        df = ragged

        result = df.rolling(window="1s", min_periods=1).last()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).last()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).last()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, op, result_data",
        [
            ("ms", "min", [0.0] * 10),
            ("ms", "mean", [0.0] * 9 + [2.0 / 9]),
            ("ms", "max", [0.0] * 9 + [2.0]),
            ("s", "min", [0.0] * 10),
            ("s", "mean", [0.0] * 9 + [2.0 / 9]),
            ("s", "max", [0.0] * 9 + [2.0]),
            ("min", "min", [0.0] * 10),
            ("min", "mean", [0.0] * 9 + [2.0 / 9]),
            ("min", "max", [0.0] * 9 + [2.0]),
            ("h", "min", [0.0] * 10),
            ("h", "mean", [0.0] * 9 + [2.0 / 9]),
            ("h", "max", [0.0] * 9 + [2.0]),
            ("D", "min", [0.0] * 10),
            ("D", "mean", [0.0] * 9 + [2.0 / 9]),
            ("D", "max", [0.0] * 9 + [2.0]),
        ],
    )
    def test_freqs_ops(self, freq, op, result_data):
        # [FUNCTION DOC]
        # Regression test for GH 21096. Parametrized over frequencies ms, s, min, h, D,
        # operations min, mean, max, and expected data. Builds a float series of zeros
        # over ten steps at that frequency with the second value missing and the last
        # set to 2; a window of ten steps in the same unit yields zeros for min, and
        # for mean and max the last value is two ninths and two respectively.
        # GH 21096
        index = date_range(start="2018-1-1 01:00:00", freq=f"1{freq}", periods=10)
        # Explicit cast to float to avoid implicit cast when setting nan
        s = Series(data=0, index=index, dtype="float")
        s.iloc[1] = np.nan
        s.iloc[-1] = 2
        result = getattr(s.rolling(window=f"10{freq}"), op)()
        expected = Series(data=result_data, index=index)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "count",
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "min",
            "max",
            "first",
            "last",
        ],
    )
    def test_all(self, f, regular):
        # [FUNCTION DOC]
        # Parametrized over twelve aggregation names. Verifies that on the doubled
        # regular frame, a one-second time window equals an integer window of one
        # observation for the aggregation and for the median-like quantile 0.5.
        # simple comparison of integer vs time-based windowing
        df = regular * 2
        er = df.rolling(window=1)
        r = df.rolling(window="1s")

        result = getattr(r, f)()
        expected = getattr(er, f)()
        tm.assert_frame_equal(result, expected)

        result = r.quantile(0.5)
        expected = er.quantile(0.5)
        tm.assert_frame_equal(result, expected)

    def test_all2(self, arithmetic_win_operators):
        # [FUNCTION DOC]
        # Uses the arithmetic window operators fixture. Over 50 hourly values, restricts
        # to working hours (09:00 to 16:00), applies a five-hour time window, and
        # verifies the result equals computing a five-observation integer window
        # separately on each calendar day (restricted to the same hours) and dropping
        # the day level.
        f = arithmetic_win_operators
        # more sophisticated comparison of integer vs.
        # time-based windowing
        df = DataFrame(
            {"B": np.arange(50)}, index=date_range("20130101", periods=50, freq="h")
        )
        # in-range data
        dft = df.between_time("09:00", "16:00")

        r = dft.rolling(window="5h")

        result = getattr(r, f)()

        # we need to roll the days separately
        # to compare with a time-based roll
        # finally groupby-apply will return a multi-index
        # so we need to drop the day
        def agg_by_day(x):
            # [FUNCTION DOC]
            # HELPER: nested function applied per day. Takes a frame x, restricts it to
            # 09:00 to 16:00, and returns the integer-window (5 observations, minimum
            # periods 1) aggregation selected by the enclosing test's operator name.
            x = x.between_time("09:00", "16:00")
            return getattr(x.rolling(5, min_periods=1), f)()

        expected = (
            df.groupby(df.index.day).apply(agg_by_day).reset_index(level=0, drop=True)
        )

        tm.assert_frame_equal(result, expected)

    def test_rolling_cov_offset(self):
        # [FUNCTION DOC]
        # Regression test for GH16058. Verifies rolling covariance of a series with
        # itself using "2h" and "3h" windows on hourly data gives missing then 0.5 (or
        # missing, 0.5 then 1.0), equal to the integer windows 2 and 3 with minimum
        # periods one.
        # GH16058

        idx = date_range("2017-01-01", periods=24, freq="1h")
        ss = Series(np.arange(len(idx)), index=idx)

        result = ss.rolling("2h").cov()
        expected = Series([np.nan] + [0.5] * (len(idx) - 1), index=idx)
        tm.assert_series_equal(result, expected)

        expected2 = ss.rolling(2, min_periods=1).cov()
        tm.assert_series_equal(result, expected2)

        result = ss.rolling("3h").cov()
        expected = Series([np.nan, 0.5] + [1.0] * (len(idx) - 2), index=idx)
        tm.assert_series_equal(result, expected)

        expected2 = ss.rolling(3, min_periods=1).cov()
        tm.assert_series_equal(result, expected2)

    def test_rolling_on_decreasing_index(self, unit):
        # [FUNCTION DOC]
        # Regression test for GH-19248 and GH-32385 (time unit fixture). Verifies a "5s"
        # rolling min on a monotonic decreasing timestamp index works and gives the
        # expected values 3, 3, 4, 4, 6.
        # GH-19248, GH-32385
        index = DatetimeIndex(
            [
                Timestamp("20190101 09:00:30"),
                Timestamp("20190101 09:00:27"),
                Timestamp("20190101 09:00:20"),
                Timestamp("20190101 09:00:18"),
                Timestamp("20190101 09:00:10"),
            ]
        ).as_unit(unit)

        df = DataFrame({"column": [3, 4, 4, 5, 6]}, index=index)
        result = df.rolling("5s").min()
        expected = DataFrame({"column": [3.0, 3.0, 4.0, 4.0, 6.0]}, index=index)
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_empty(self):
        # [FUNCTION DOC]
        # Regression test for GH-32385: a "5s" rolling min of an empty frame with an
        # empty default index returns an equal empty frame.
        # GH-32385
        df = DataFrame({"column": []}, index=[])
        result = df.rolling("5s").min()
        expected = DataFrame({"column": []}, index=[])
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_multi_index_level(self):
        # [FUNCTION DOC]
        # Regression test for GH-15584. Verifies a "10D" rolling sum using the date
        # level values of a two-level index through on gives the cumulative sums 0, 1,
        # 3, 6, 10, 15 on the original index.
        # GH-15584
        df = DataFrame(
            {"column": range(6)},
            index=MultiIndex.from_product(
                [date_range("20190101", periods=3), range(2)], names=["date", "seq"]
            ),
        )
        result = df.rolling("10D", on=df.index.get_level_values("date")).sum()
        expected = DataFrame(
            {"column": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]}, index=df.index
        )
        tm.assert_frame_equal(result, expected)


def test_nat_axis_error():
    # [FUNCTION DOC]
    # Verifies a time-based rolling mean on an index that contains a missing
    # timestamp raises a value error saying index values must not have NaT.
    idx = [Timestamp("2020"), NaT]
    df = DataFrame(np.eye(2), index=idx)
    with pytest.raises(ValueError, match="index values must not have NaT"):
        df.rolling("D").mean()


@td.skip_if_no("pyarrow")
def test_arrow_datetime_axis():
    # [FUNCTION DOC]
    # Regression test for GH 55849. Requires pyarrow. Verifies a "1D" rolling sum on
    # a series with an arrow-backed datetime index returns the same series (windows
    # of one day contain only the current point).
    # GH 55849
    expected = Series(
        np.arange(5, dtype=np.float64),
        index=Index(
            date_range("2020-01-01", periods=5), dtype="timestamp[ns][pyarrow]"
        ),
    )
    result = expected.rolling("1D").sum()
    tm.assert_series_equal(result, expected)
