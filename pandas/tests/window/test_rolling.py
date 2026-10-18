# [MODULE DOC]
# General tests for fixed and time-based rolling windows: constructor validation,
# step and window-type restrictions, closed interval handling (fixed,
# datetime-like, centered, nonunique and decreasing indexes), min, max, median
# and quantile with closed sides, empty inputs, minimum periods, iteration over
# rolling objects, numerical accuracy regressions (compensated sums, variance,
# skewness and kurtosis), first/last, rank, unique count, quantile argument
# validation, numeric-only handling, time units and time zones, and rolling
# min/max with custom prescribed window bounds. Uses frame_or_series, closed,
# center, step, raw, ascending, arithmetic_win_operators, numeric_only and unit
# fixtures.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import (
    IS64,
)
from pandas.errors import Pandas4Warning

from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer

from pandas.tseries.offsets import BusinessDay


def test_doc_string():
    # [FUNCTION DOC]
    # Smoke test of the documentation example: a one-column frame with a missing
    # value rolled with a window of 2, summed with default and with minimum periods
    # 1. Asserts nothing; passes if no exception is raised.
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.rolling(2).sum()
    df.rolling(2, min_periods=1).sum()


def test_constructor(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 12669 and GH 13383, parametrized over series and
    # frame. Verifies valid constructor calls (window zero, window 2, minimum
    # periods, centering both ways) succeed, and that a negative window raises a
    # value error saying the window must be an integer zero or greater.
    # GH 12669

    c = frame_or_series(range(5)).rolling

    # valid
    c(0)
    c(window=2)
    c(window=2, min_periods=1)
    c(window=2, min_periods=1, center=True)
    c(window=2, min_periods=1, center=False)

    # GH 13383

    msg = "window must be an integer 0 or greater"

    with pytest.raises(ValueError, match=msg):
        c(-1)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_invalid_constructor(frame_or_series, w):
    # [FUNCTION DOC]
    # Parametrized over series/frame and invalid values (float, string, array).
    # Verifies each invalid value for window raises a value error (message about
    # the window being an integer, or a string not compatible with a datetimelike
    # index), for minimum periods a value error that it must be an integer, and for
    # center a value error that it must be a boolean.
    # not valid

    c = frame_or_series(range(5)).rolling

    msg = "|".join(
        [
            "window must be an integer",
            "passed window foo is not compatible with a datetimelike index",
        ]
    )
    with pytest.raises(ValueError, match=msg):
        c(window=w)

    msg = "min_periods must be an integer"
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)

    msg = "center must be a boolean"
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)


@pytest.mark.parametrize(
    "window",
    [
        timedelta(days=3),
        Timedelta(days=3),
        "3D",
        VariableOffsetWindowIndexer(
            index=date_range("2015-12-25", periods=5), offset=BusinessDay(1)
        ),
    ],
)
def test_freq_window_not_implemented(window):
    # [FUNCTION DOC]
    # Regression test for GH 15354. Parametrized over a timedelta, a pandas
    # timedelta, a "3D" string and a variable offset window indexer. Verifies rolling
    # with step 3 on a daily-indexed frame raises a not implemented error beginning
    # with "step not implemented" or "step is not supported".
    # GH 15354
    df = DataFrame(
        np.arange(10),
        index=date_range("2015-12-24", periods=10, freq="D"),
    )
    with pytest.raises(
        NotImplementedError, match="^step (not implemented|is not supported)"
    ):
        df.rolling(window, step=3).sum()


@pytest.mark.parametrize("agg", ["cov", "corr"])
def test_step_not_implemented_for_cov_corr(agg):
    # [FUNCTION DOC]
    # Regression test for GH 15354. Parametrized over cov and corr. Verifies a
    # rolling object with step 2 raises a not implemented error saying step is not
    # implemented for these methods.
    # GH 15354
    roll = DataFrame(range(2)).rolling(1, step=2)
    with pytest.raises(NotImplementedError, match="step not implemented"):
        getattr(roll, agg)()


@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window):
    # [FUNCTION DOC]
    # Regression test for GH 15440. Parametrized over a standard and a pandas
    # timedelta of three days. Verifies the rolling sum over ten daily values equals
    # the expected sums and equals the "3D" string window result.
    # GH 15440
    n = 10
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    expected_data = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))

    result = df.rolling(window=window).sum()
    expected = DataFrame(
        {"value": expected_data},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    tm.assert_frame_equal(result, expected)
    expected = df.rolling("3D").sum()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3), "3D"])
def test_constructor_timedelta_window_and_minperiods(window, raw):
    # [FUNCTION DOC]
    # Regression test for GH 15305. Parametrized over three window forms of three
    # days (and raw). Verifies the sum and the generic apply of a sum function with
    # minimum periods 2 both give the expected values (first missing).
    # GH 15305
    n = 10
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    expected = DataFrame(
        {"value": np.append([np.nan, 1.0], np.arange(3.0, 27.0, 3))},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    result_roll_sum = df.rolling(window=window, min_periods=2).sum()
    result_roll_generic = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)


def test_closed_fixed(closed, arithmetic_win_operators):
    # [FUNCTION DOC]
    # Regression test for GH 34315. Uses the closed and arithmetic window operator
    # fixtures. Verifies a fixed window of 2 observations with minimum periods 1 and
    # the given closed side equals a "2D" time window on daily data (index ignored).
    # GH 34315
    func_name = arithmetic_win_operators
    df_fixed = DataFrame({"A": [0, 1, 2, 3, 4]})
    df_time = DataFrame({"A": [0, 1, 2, 3, 4]}, index=date_range("2020", periods=5))

    result = getattr(
        df_fixed.rolling(2, closed=closed, min_periods=1),
        func_name,
    )()
    expected = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1),
        func_name,
    )().reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "closed, window_selections",
    [
        (
            "both",
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ],
        ),
        (
            "left",
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
        ),
        (
            "right",
            [
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
                [False, False, False, False, True],
            ],
        ),
        (
            "neither",
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
        ),
    ],
)
def test_datetimelike_centered_selections(
    closed, window_selections, arithmetic_win_operators
):
    # [FUNCTION DOC]
    # Regression test for GH 34315. Parametrized over closed side with a boolean
    # selection matrix describing which of five points each centered window of "2D"
    # includes, and over the operator fixture. Verifies the centered time-based
    # result equals the operator applied to each selected subset.
    # GH 34315
    func_name = arithmetic_win_operators
    df_time = DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range("2020", periods=5)
    )

    expected = DataFrame(
        {"A": [getattr(df_time["A"].iloc[s], func_name)() for s in window_selections]},
        index=date_range("2020", periods=5),
    )

    result = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1, center=True),
        func_name,
    )()

    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("3s", "right", [3.0, 3.0, 3.0]),
        ("3s", "both", [3.0, 3.0, 3.0]),
        ("3s", "left", [3.0, 3.0, 3.0]),
        ("3s", "neither", [3.0, 3.0, 3.0]),
        ("2s", "right", [3.0, 2.0, 2.0]),
        ("2s", "both", [3.0, 3.0, 3.0]),
        ("2s", "left", [1.0, 3.0, 3.0]),
        ("2s", "neither", [1.0, 2.0, 2.0]),
    ],
)
def test_datetimelike_centered_offset_covers_all(
    window, closed, expected, frame_or_series
):
    # [FUNCTION DOC]
    # Regression test for GH 42753. Parametrized over a window ("3s" or "2s"), closed
    # side and expected sums, for series and frame. With three timestamps (the last
    # two equal), verifies centered time-window sums equal the expected counts.
    # GH 42753

    index = [
        Timestamp("20130101 09:00:01"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:02"),
    ]
    df = frame_or_series([1, 1, 1], index=index)

    result = df.rolling(window, closed=closed, center=True).sum()
    expected = frame_or_series(expected, index=index)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("2D", "right", [4, 4, 4, 4, 4, 4, 2, 2]),
        ("2D", "left", [2, 2, 4, 4, 4, 4, 4, 4]),
        ("2D", "both", [4, 4, 6, 6, 6, 6, 4, 4]),
        ("2D", "neither", [2, 2, 2, 2, 2, 2, 2, 2]),
    ],
)
def test_datetimelike_nonunique_index_centering(
    window, closed, expected, frame_or_series
):
    # [FUNCTION DOC]
    # Parametrized over window "2D", closed side and expected counts, for series and
    # frame. With a datetime index in which every day occurs twice, verifies centered
    # window sums of ones equal the expected counts.
    index = DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
            "2020-01-03",
            "2020-01-04",
            "2020-01-04",
        ]
    )

    df = frame_or_series([1] * 8, index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    result = df.rolling(window, center=True, closed=closed).sum()

    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("left", [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]),
        ("neither", [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]),
        ("right", [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]),
        ("both", [0, 1, 3, 6, 10, 15, 20, 27, 26, 30]),
    ],
)
def test_variable_window_nonunique(closed, expected, frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 20712. Parametrized over closed side and expected sums.
    # With a nonunique datetime index, verifies a "2D" time window sum of ten values
    # for each closed side.
    # GH 20712
    index = DatetimeIndex(
        [
            "2011-01-01",
            "2011-01-01",
            "2011-01-02",
            "2011-01-02",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
        ]
    )

    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    result = df.rolling("2D", closed=closed).sum()

    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("left", [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 18, 21]),
        ("neither", [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 13, 8]),
        ("right", [0, 1, 3, 6, 10, 15, 21, 28, 21, 17]),
        ("both", [0, 1, 3, 6, 10, 15, 21, 28, 26, 30]),
    ],
)
def test_variable_offset_window_nonunique(closed, expected, frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 20712. Parametrized over closed side and expected sums.
    # With the same nonunique index, verifies a variable offset window indexer with
    # a two business day offset and minimum periods 1 gives the expected sums.
    # GH 20712
    index = DatetimeIndex(
        [
            "2011-01-01",
            "2011-01-01",
            "2011-01-02",
            "2011-01-02",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
        ]
    )

    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    offset = BusinessDay(2)
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result = df.rolling(indexer, closed=closed, min_periods=1).sum()

    tm.assert_equal(result, expected)


def test_even_number_window_alignment():
    # [FUNCTION DOC]
    # See discussion in GH 38780. Verifies a centered "2D" time window mean with
    # minimum periods 1 on three daily values gives 0.5, 1.5, 2. The comment notes
    # that integer windows align differently.
    # see discussion in GH 38780
    s = Series(range(3), index=date_range(start="2020-01-01", freq="D", periods=3))

    # behavior of index- and datetime-based windows differs here!
    # s.rolling(window=2, min_periods=1, center=True).mean()

    result = s.rolling(window="2D", min_periods=1, center=True).mean()

    expected = Series([0.5, 1.5, 2], index=s.index)

    tm.assert_series_equal(result, expected)


def test_closed_fixed_binary_col(center, step):
    # [FUNCTION DOC]
    # Regression test for GH 34315. Uses the center and step fixtures. Verifies a
    # fixed window as long as the data with closed left, minimum periods 1 and
    # the given centering gives the expected running means of a binary column
    # (sliced by step).
    # GH 34315
    data = [0, 1, 1, 0, 0, 1, 0, 1]
    df = DataFrame(
        {"binary_col": data},
        index=date_range(start="2020-01-01", freq="min", periods=len(data)),
    )

    if center:
        expected_data = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]

    expected = DataFrame(
        expected_data,
        columns=["binary_col"],
        index=date_range(start="2020-01-01", freq="min", periods=len(expected_data)),
    )[::step]

    rolling = df.rolling(
        window=len(df), closed="left", min_periods=1, center=center, step=step
    )
    result = rolling.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("closed", ["neither", "left"])
def test_closed_empty(closed, arithmetic_win_operators):
    # [FUNCTION DOC]
    # Regression test for GH 26005. Parametrized over closed neither and left and the
    # operator fixture. With points spaced two days apart and a "1D" window, every
    # window is empty so every result is missing.
    # GH 26005
    func_name = arithmetic_win_operators
    ser = Series(data=np.arange(5), index=date_range("2000", periods=5, freq="2D"))
    roll = ser.rolling("1D", closed=closed)

    result = getattr(roll, func_name)()
    expected = Series([np.nan] * 5, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry(func):
    # [FUNCTION DOC]
    # Regression test for GH24718. Parametrized over min and max. Verifies a
    # single-entry series rolled over "10D" with closed left gives missing.
    # GH24718
    ser = Series(data=[2], index=date_range("2000", periods=1))
    result = getattr(ser.rolling("10D", closed="left"), func)()
    tm.assert_series_equal(result, Series([np.nan], index=ser.index))


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry_groupby(func):
    # [FUNCTION DOC]
    # Regression test for GH24718. Parametrized over min and max. Verifies a grouped
    # (unsorted groups) "10D" rolling with closed left gives missing for the first
    # entry of each group and the earlier value for the second entry of group 1.
    # GH24718
    ser = DataFrame(
        data={"A": [1, 1, 2], "B": [3, 2, 1]},
        index=date_range("2000", periods=3),
    )
    result = getattr(
        ser.groupby("A", sort=False)["B"].rolling("10D", closed="left"), func
    )()
    exp_idx = MultiIndex.from_arrays(arrays=[[1, 1, 2], ser.index], names=("A", None))
    expected = Series(data=[np.nan, 3, np.nan], index=exp_idx, name="B")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("input_dtype", ["int", "float"])
@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "both", [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("min", "neither", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "left", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", "right", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "both", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "neither", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("max", "left", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_closed_min_max_datetime(input_dtype, func, closed, expected):
    # [FUNCTION DOC]
    # See gh-21704. Parametrized over input dtype (integer or float), min/max, closed
    # side and expected values. Verifies "3D" rolling min and max on ten daily values
    # for each closed side.
    # see gh-21704
    ser = Series(
        data=np.arange(10).astype(input_dtype),
        index=date_range("2000", periods=10),
    )

    result = getattr(ser.rolling("3D", closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)


def test_closed_uneven():
    # [FUNCTION DOC]
    # See gh-21704. Verifies a "3D" rolling min with closed left on a daily series
    # from which two entries were dropped gives the expected values.
    # see gh-21704
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))

    # uneven
    ser = ser.drop(index=ser.index[[1, 5]])
    result = ser.rolling("3D", closed="left").min()
    expected = Series([np.nan, 0, 0, 2, 3, 4, 6, 6], index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "both", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("min", "neither", [np.nan, np.nan, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "left", [np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("max", "right", [np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan, np.nan]),
        ("max", "both", [np.nan, 1, 2, 3, 4, 5, 6, 6, 6, np.nan]),
        ("max", "neither", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan]),
        ("max", "left", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan]),
    ],
)
def test_closed_min_max_minp(func, closed, expected):
    # [FUNCTION DOC]
    # See gh-21704. Parametrized over min/max, closed side and expected values. A
    # float daily series whose last three values are missing is rolled over "3D"
    # with minimum periods 2; verifies the expected values including missing where
    # too few observations.
    # see gh-21704
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))
    # Explicit cast to float to avoid implicit cast when setting nan
    ser = ser.astype("float")
    ser[ser.index[-3:]] = np.nan
    result = getattr(ser.rolling("3D", min_periods=2, closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("right", [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("both", [0, 0.5, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("neither", [np.nan, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("left", [np.nan, 0, 0.5, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_closed_median_quantile(closed, expected):
    # [FUNCTION DOC]
    # Regression test for GH 26005. Parametrized over closed side and expected
    # values. Verifies "3D" rolling median and quantile 0.5 give identical expected
    # values.
    # GH 26005
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))
    roll = ser.rolling("3D", closed=closed)
    expected = Series(expected, index=ser.index)

    result = roll.median()
    tm.assert_series_equal(result, expected)

    result = roll.quantile(0.5)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("roller", ["1s", 1])
def tests_empty_df_rolling(roller):
    # [FUNCTION DOC]
    # Regression test for GH 15819. Parametrized over a "1s" string window and an
    # integer window of 1. Verifies rolling sum on an empty frame, with and without
    # an empty datetime index, returns an equal empty frame. (The function name
    # begins with "tests", which pytest still collects.)
    # GH 15819 Verifies that datetime and integer rolling windows can be
    # applied to empty DataFrames
    expected = DataFrame()
    result = DataFrame().rolling(roller).sum()
    tm.assert_frame_equal(result, expected)

    # Verifies that datetime and integer rolling windows can be applied to
    # empty DataFrames with datetime index
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).rolling(roller).sum()
    tm.assert_frame_equal(result, expected)


def test_empty_window_median_quantile():
    # [FUNCTION DOC]
    # Regression test for GH 26005. Verifies rolling median and quantile with a
    # window of zero over three values return all missing.
    # GH 26005
    expected = Series([np.nan, np.nan, np.nan])
    roll = Series(np.arange(3)).rolling(0)

    result = roll.median()
    tm.assert_series_equal(result, expected)

    result = roll.quantile(0.1)
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero():
    # [FUNCTION DOC]
    # Regression test for pull request 18921 (rolling analogue of the expanding case).
    # Verifies rolling sum with window 1 and minimum periods zero over a single
    # missing value gives 0.0, and (in the part below) minimum periods one gives
    # missing.
    # https://github.com/pandas-dev/pandas/pull/18921
    # minp=0
    x = Series([np.nan])
    result = x.rolling(1, min_periods=0).sum()
    expected = Series([0.0])
    tm.assert_series_equal(result, expected)

    # minp=1
    result = x.rolling(1, min_periods=1).sum()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero_variable():
    # [FUNCTION DOC]
    # Regression test for pull request 18921. Verifies a time-based rolling sum with
    # a "2D" window and minimum periods zero over four missing values gives 0.0 at
    # every position.
    # https://github.com/pandas-dev/pandas/pull/18921
    x = Series(
        [np.nan] * 4,
        index=DatetimeIndex(["2017-01-01", "2017-01-04", "2017-01-06", "2017-01-07"]),
    )
    result = x.rolling(Timedelta("2D"), min_periods=0).sum()
    expected = Series(0.0, index=x.index)
    tm.assert_series_equal(result, expected)


def test_multi_index_names():
    # [FUNCTION DOC]
    # Regression test for GH 16789 and 16825. Verifies pairwise rolling covariance
    # (window 3) on a frame with two-level named columns keeps the original columns
    # and gives a result index whose level names are none followed by the two column
    # level names.
    # GH 16789, 16825
    cols = MultiIndex.from_product([["A", "B"], ["C", "D", "E"]], names=["1", "2"])
    df = DataFrame(np.ones((10, 6)), columns=cols)
    result = df.rolling(3).cov()

    tm.assert_index_equal(result.columns, df.columns)
    assert result.index.names == [None, "1", "2"]


def test_rolling_axis_sum():
    # [FUNCTION DOC]
    # See gh-23372. Verifies rolling sum with window 3 over a 10 by 20 frame of ones
    # gives two missing rows then 3.0 in every column.
    # see gh-23372.
    df = DataFrame(np.ones((10, 20)))
    expected = DataFrame({i: [np.nan] * 2 + [3.0] * 8 for i in range(20)})
    result = df.rolling(3).sum()
    tm.assert_frame_equal(result, expected)


def test_rolling_axis_count():
    # [FUNCTION DOC]
    # See gh-26055. Verifies rolling count with window 2 and minimum periods 0 gives
    # 1, 2, 2 per column.
    # see gh-26055
    df = DataFrame({"x": range(3), "y": range(3)})

    expected = DataFrame({"x": [1.0, 2.0, 2.0], "y": [1.0, 2.0, 2.0]})
    result = df.rolling(2, min_periods=0).count()
    tm.assert_frame_equal(result, expected)


def test_readonly_array():
    # [FUNCTION DOC]
    # Regression test for GH-27766. Verifies rolling mean works on a read-only numpy
    # array (window 2), producing missing wherever a window includes the missing
    # value.
    # GH-27766
    arr = np.array([1, 3, np.nan, 3, 5])
    arr.setflags(write=False)
    result = Series(arr).rolling(2).mean()
    expected = Series([np.nan, 2, np.nan, np.nan, 4])
    tm.assert_series_equal(result, expected)


def test_rolling_datetime(tz_naive_fixture):
    # [FUNCTION DOC]
    # Regression test for GH-28192, using the timezone-naive fixture (a set of time
    # zones including none). Verifies rolling "2D" sum along the columns (transposed)
    # of a frame whose column labels are dates gives 1.0 for the first date and 2.0
    # for later dates.
    # GH-28192
    tz = tz_naive_fixture
    df = DataFrame(
        {i: [1] * 2 for i in date_range("2019-8-01", "2019-08-03", freq="D", tz=tz)}
    )

    result = df.T.rolling("2D").sum().T
    expected = DataFrame(
        {
            **{
                i: [1.0] * 2
                for i in date_range("2019-8-01", periods=1, freq="D", tz=tz)
            },
            **{
                i: [2.0] * 2
                for i in date_range("2019-8-02", "2019-8-03", freq="D", tz=tz)
            },
        }
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_window_as_string(center):
    # [FUNCTION DOC]
    # See gh-22590. Uses the center fixture. Builds a daily frame over one year from
    # the current time; verifies a "21D" rolling max with minimum periods 2 and
    # closed left over a column of ones gives ones, with the first two missing when
    # not centered.
    # see gh-22590
    date_today = datetime.now()
    days = date_range(date_today, date_today + timedelta(365), freq="D")

    data = np.ones(len(days))
    df = DataFrame({"DateCol": days, "metric": data})

    df.set_index("DateCol", inplace=True)
    result = df.rolling(window="21D", min_periods=2, closed="left", center=center)[
        "metric"
    ].agg("max")

    index = days.rename("DateCol")
    index = index._with_freq(None)
    expected_data = np.ones(len(days), dtype=np.float64)
    if not center:
        expected_data[:2] = np.nan
    expected = Series(expected_data, index=index, name="metric")
    tm.assert_series_equal(result, expected)


def test_min_periods1():
    # [FUNCTION DOC]
    # Regression test for GH#6795. Verifies centered rolling max (window 3, minimum
    # periods 1) of 0, 1, 2, 1, 0 gives 1, 2, 2, 2, 1.
    # GH#6795
    df = DataFrame([0, 1, 2, 1, 0], columns=["a"])
    result = df["a"].rolling(3, center=True, min_periods=1).max()
    expected = Series([1.0, 2.0, 2.0, 2.0, 1.0], name="a")
    tm.assert_series_equal(result, expected)


def test_rolling_count_with_min_periods(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 26996, for series and frame. Verifies rolling count
    # with window 3 and minimum periods 3 gives missing, missing, 3, 3, 3.
    # GH 26996
    result = frame_or_series(range(5)).rolling(3, min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 3.0, 3.0])
    tm.assert_equal(result, expected)


def test_rolling_count_default_min_periods_with_null_values(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 26996 and GH 31302. Verifies rolling count with window
    # 3 and minimum periods 0 over values with a missing entry counts only
    # non-missing values.
    # GH 26996
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0]

    # GH 31302
    result = frame_or_series(values).rolling(3, min_periods=0).count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "df,expected,window,min_periods",
    [
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            3,
            None,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            2,
            1,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            2,
            2,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            1,
            1,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            1,
            0,
        ),
        ({"A": [1], "B": [4]}, [], 2, None),
        ({"A": [1], "B": [4]}, [], 2, 1),
        (None, [({}, [])], 2, None),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            3,
            2,
        ),
    ],
)
def test_iter_rolling_dataframe(df, expected, window, min_periods):
    # [FUNCTION DOC]
    # Regression test for GH 11704. Parametrized over frame data, expected sequence
    # of window frames with their index labels, window size and minimum periods
    # (including cases with fewer rows than the window, empty frames and missing
    # values). Verifies iterating over a rolling object yields the expected window
    # frames in order.
    # GH 11704
    df = DataFrame(df)
    expecteds = [DataFrame(values, index=index) for (values, index) in expected]

    for expected, actual in zip(
        expecteds, df.rolling(window, min_periods=min_periods), strict=False
    ):
        tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "expected,window",
    [
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            "2D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            "3D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            "1D",
        ),
    ],
)
def test_iter_rolling_on_dataframe(expected, window):
    # [FUNCTION DOC]
    # Regression test for GH 11704 and GH 40373. Parametrized over "2D", "3D" and
    # "1D" windows. Verifies iterating a time-based rolling object with the on
    # column set to a date column yields window frames indexed by that column and
    # excluding it from the values.
    # GH 11704, 40373
    df = DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [4, 5, 6, 7, 8],
            "C": date_range(start="2016-01-01", periods=5, freq="D"),
        }
    )

    expecteds = [
        DataFrame(values, index=df.loc[index, "C"]) for (values, index) in expected
    ]
    for expected, actual in zip(expecteds, df.rolling(window, on="C"), strict=False):
        tm.assert_frame_equal(actual, expected)


def test_iter_rolling_on_dataframe_unordered():
    # [FUNCTION DOC]
    # Regression test for GH 43386. Verifies iterating over a grouped rolling object
    # of window 2 yields frames matching selected rows in group order, including
    # the windows for the interleaved group.
    # GH 43386
    df = DataFrame({"a": ["x", "y", "x"], "b": [0, 1, 2]})
    results = list(df.groupby("a").rolling(2))
    expecteds = [df.iloc[idx, [1]] for idx in [[0], [0, 2], [1]]]
    for result, expected in zip(results, expecteds, strict=True):
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ser,expected,window, min_periods",
    [
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            None,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            2,
        ),
        (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 0),
        (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2, 0),
        (Series([], dtype="int64"), [], 2, 1),
    ],
)
def test_iter_rolling_series(ser, expected, window, min_periods):
    # [FUNCTION DOC]
    # Regression test for GH 11704. Parametrized over series, expected sequence of
    # windows, window size and minimum periods (including an empty series).
    # Verifies iteration yields equal series, strictly pairing with the expected
    # list.
    # GH 11704
    expecteds = [Series(values, index=index) for (values, index) in expected]

    for expected, actual in zip(
        expecteds, ser.rolling(window, min_periods=min_periods), strict=True
    ):
        tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "expected,expected_index,window",
    [
        (
            [[0], [1], [2], [3], [4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-02", periods=1, freq="D"),
                date_range("2020-01-03", periods=1, freq="D"),
                date_range("2020-01-04", periods=1, freq="D"),
                date_range("2020-01-05", periods=1, freq="D"),
            ],
            "1D",
        ),
        (
            [[0], [0, 1], [1, 2], [2, 3], [3, 4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-02", periods=2, freq="D"),
                date_range("2020-01-03", periods=2, freq="D"),
                date_range("2020-01-04", periods=2, freq="D"),
            ],
            "2D",
        ),
        (
            [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-01", periods=3, freq="D"),
                date_range("2020-01-02", periods=3, freq="D"),
                date_range("2020-01-03", periods=3, freq="D"),
            ],
            "3D",
        ),
    ],
)
def test_iter_rolling_datetime(expected, expected_index, window):
    # [FUNCTION DOC]
    # Regression test for GH 11704. Parametrized over "1D", "2D", "3D" windows with
    # expected values and date indexes of each window. Verifies iterating a
    # time-based rolling object on a daily series yields the expected windows.
    # GH 11704
    ser = Series(range(5), index=date_range(start="2020-01-01", periods=5, freq="D"))

    expecteds = [
        Series(values, index=idx)
        for (values, idx) in zip(expected, expected_index, strict=True)
    ]

    for expected, actual in zip(expecteds, ser.rolling(window), strict=True):
        tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "grouping,_index",
    [
        (
            {"level": 0},
            MultiIndex.from_tuples(
                [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None]
            ),
        ),
        (
            {"by": "X"},
            MultiIndex.from_tuples(
                [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=["X", None]
            ),
        ),
    ],
)
def test_rolling_positional_argument(grouping, _index, raw):
    # [FUNCTION DOC]
    # Regression test for GH 34605 (and GH 40341). Parametrized over grouping by
    # index level or by column X, with the expected index. Verifies grouped rolling
    # apply with extra positional arguments (a divisor of 2) passes them through to
    # the function; when grouping by a column, the grouping column is dropped from
    # the expected result.
    # GH 34605

    def scaled_sum(*args):
        # [FUNCTION DOC]
        # HELPER: nested user function taking any positional arguments. Raises a value
        # error saying two arguments are needed if given fewer than two; otherwise
        # treats them as an array and a scale and returns the sum of the array divided
        # by the scale.
        if len(args) < 2:
            raise ValueError("The function needs two arguments")
        array, scale = args
        return array.sum() / scale

    df = DataFrame(data={"X": range(5)}, index=[0, 0, 1, 1, 1])

    expected = DataFrame(data={"X": [0.0, 0.5, 1.0, 1.5, 2.0]}, index=_index)
    # GH 40341
    if "by" in grouping:
        expected = expected.drop(columns="X", errors="ignore")
    result = df.groupby(**grouping).rolling(1).apply(scaled_sum, raw=raw, args=(2,))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("add", [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add, unit):
    # [FUNCTION DOC]
    # Regression test for GH 36031 (compensated summation). Parametrized over an
    # added offset of 0.0 or 2.0 and the time unit fixture. Resamples three huge
    # values spaced three seconds apart with forward fill, rolls "3s" with closed
    # left and minimum periods 3 and verifies the mean matches the expected values
    # (first three missing, then 3002399751580330.5 and so on, ending at 0.0).
    # GH: 36031 implementing kahan summation
    dti = DatetimeIndex(
        [
            Timestamp("19700101 09:00:00"),
            Timestamp("19700101 09:00:03"),
            Timestamp("19700101 09:00:06"),
        ]
    ).as_unit(unit)
    df = DataFrame(
        {"A": [3002399751580331.0 + add, -0.0, -0.0]},
        index=dti,
    )
    result = (
        df.resample("1s").ffill().rolling("3s", closed="left", min_periods=3).mean()
    )
    dates = date_range("19700101 09:00:00", periods=7, freq="s", unit=unit)
    expected = DataFrame(
        {
            "A": [
                np.nan,
                np.nan,
                np.nan,
                3002399751580330.5,
                2001599834386887.25,
                1000799917193443.625,
                0.0,
            ]
        },
        index=dates,
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_numerical_accuracy_kahan_sum():
    # [FUNCTION DOC]
    # Regression test for GH 13254. Verifies rolling sum with window 3 of small
    # values with cancellation gives 0.539, -1.647, 0.0, 0.0 after the initial
    # missing values (exact zeros, no floating drift).
    # GH: 13254
    df = DataFrame([2.186, -1.647, 0.0, 0.0, 0.0, 0.0], columns=["x"])
    result = df["x"].rolling(3).sum()
    expected = Series([np.nan, np.nan, 0.539, -1.647, 0.0, 0.0], name="x")
    tm.assert_series_equal(result, expected)


def test_rolling_numerical_accuracy_jump():
    # [FUNCTION DOC]
    # Regression test for GH 32761. Verifies a "60s" rolling mean over one-minute
    # data followed by a jump of one day returns the data itself (each window holds
    # only the current point).
    # GH: 32761
    index = date_range(start="2020-01-01", end="2020-01-02", freq="60s").append(
        DatetimeIndex(["2020-01-03"])
    )
    data = np.random.default_rng(2).random(len(index))

    df = DataFrame({"data": data}, index=index)
    result = df.rolling("60s").mean()
    tm.assert_frame_equal(result, df[["data"]])


def test_rolling_numerical_accuracy_small_values():
    # [FUNCTION DOC]
    # Regression test for GH 10319. Verifies a rolling mean with window 1 over tiny
    # values (and negative zeros) returns the original series.
    # GH: 10319
    s = Series(
        data=[0.00012456, 0.0003, -0.0, -0.0],
        index=date_range("1999-02-03", "1999-02-06"),
    )
    result = s.rolling(1).mean()
    tm.assert_series_equal(result, s)


def test_rolling_numerical_too_large_numbers():
    # [FUNCTION DOC]
    # Regression test for GH 11645. Verifies a rolling mean (window 5) over a series
    # containing one huge negative value (minus 9e33) gives that value divided by 5
    # while the huge value is in the window and normal means afterwards.
    # GH: 11645
    dates = date_range("2015-01-01", periods=10, freq="D")
    ds = Series(data=range(10), index=dates, dtype=np.float64)
    ds.iloc[2] = -9e33
    result = ds.rolling(5).mean()
    expected = Series(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.8e33,
            -1.8e33,
            -1.8e33,
            5.0,
            6.0,
            7.0,
        ],
        index=dates,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    [
        (
            period_range(start="2020-01-01 08:00", end="2020-01-01 08:08", freq="min"),
            "2min",
        ),
        (
            period_range(
                start="2020-01-01 08:00", end="2020-01-01 12:00", freq="30min"
            ),
            "1h",
        ),
    ],
)
@pytest.mark.parametrize(
    ("func", "values"),
    [
        ("min", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("sum", [np.nan, 0, 1, 3, 5, 7, 9, 11, 13]),
    ],
)
def test_rolling_period_index(index, window, func, values):
    # [FUNCTION DOC]
    # Regression test for GH 34225. Parametrized over period indexes (per-minute with
    # a "2min" window, and 30-minute spaced with a "1h" window) and min, max, sum
    # with expected values. Verifies time-based rolling on period-indexed series
    # with closed left.
    # GH: 34225
    ds = Series([0, 1, 2, 3, 4, 5, 6, 7, 8], index=index)
    result = getattr(ds.rolling(window, closed="left"), func)()
    expected = Series(values, index=index)
    tm.assert_series_equal(result, expected)


def test_rolling_sem(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH 26476, for series and frame. Verifies rolling standard
    # error of the mean (window 2, minimum periods 1) of 0, 1, 2 gives missing, 0.5,
    # 0.5.
    # GH: 26476
    obj = frame_or_series([0, 1, 2])
    result = obj.rolling(2, min_periods=1).sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.5] * 2)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "values", "window", "ddof", "expected_values"),
    [
        ("var", [99999999999999999, 1, 1, 2, 3, 1, 1], 2, 1, [5e33, 0, 0.5, 0.5, 2, 0]),
        (
            "std",
            [99999999999999999, 1, 1, 2, 3, 1, 1],
            2,
            1,
            [7.071068e16, 0, 0.7071068, 0.7071068, 1.414214, 0],
        ),
        ("var", [99999999999999999, 1, 2, 2, 3, 1, 1], 2, 1, [5e33, 0.5, 0, 0.5, 2, 0]),
        (
            "std",
            [99999999999999999, 1, 2, 2, 3, 1, 1],
            2,
            1,
            [7.071068e16, 0.7071068, 0, 0.7071068, 1.414214, 0],
        ),
        (
            "std",
            [1.2e03, 1.3e17, 1.5e17, 1.995e03, 1.990e03],
            2,
            1,
            [9.192388e16, 1.414214e16, 1.060660e17, 3.535534e00],
        ),
        (
            "var",
            [
                0.00000000e00,
                0.00000000e00,
                3.16188252e-18,
                2.95781651e-16,
                2.23153542e-51,
                0.00000000e00,
                0.00000000e00,
                5.39943432e-48,
                1.38206260e-73,
                0.00000000e00,
            ],
            3,
            1,
            [
                3.33250036e-036,
                2.88538519e-032,
                2.88538519e-032,
                2.91622617e-032,
                1.65991678e-102,
                9.71796366e-096,
                9.71796366e-096,
                9.71796366e-096,
            ],
        ),
        (
            "std",
            [1, -1, 0, 1, 3, 2, -2, 10000000000, 1, 2, 0, -2, 1, 3, 0, 1],
            6,
            1,
            [
                1.41421356e00,
                1.87082869e00,
                4.08248290e09,
                4.08248290e09,
                4.08248290e09,
                4.08248290e09,
                4.08248290e09,
                4.08248290e09,
                1.72240142e00,
                1.75119007e00,
                1.64316767e00,
            ],
        ),
    ],
)
def test_rolling_var_correctness(func, values, window, ddof, expected_values):
    # [FUNCTION DOC]
    # Regression test for several issues (GH 37051, 42064, 54518, 52407, 47721).
    # Parametrized over var/std, input values (including very large and very small
    # magnitudes), window, ddof and expected results. Verifies the numerically
    # stable rolling variance and standard deviation: the valid portion of the
    # result (trimmed from the first to the last valid position when any exists)
    # matches the expected within a tiny absolute tolerance, and exact zeros occur
    # exactly where expected.
    # GH: 37051, 42064, 54518, 52407, 47721
    ts = Series(values)
    result = getattr(ts.rolling(window=window), func)(ddof=ddof)
    if result.last_valid_index():
        result = result[
            result.first_valid_index() : result.last_valid_index() + 1
        ].reset_index(drop=True)
    expected = Series(expected_values)
    tm.assert_series_equal(result, expected, atol=1e-55)
    # GH 42064
    tm.assert_series_equal(result == 0, expected == 0)


def test_timeoffset_as_window_parameter_for_corr(unit):
    # [FUNCTION DOC]
    # Regression test for GH 28266 (time unit fixture). Verifies pairwise rolling
    # correlation with a "3D" window on a two-column frame indexed by five
    # irregular timestamps gives the expected values, with a two-level index of
    # timestamp and column name.
    # GH: 28266
    dti = DatetimeIndex(
        [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130102 09:00:02"),
            Timestamp("20130103 09:00:03"),
            Timestamp("20130105 09:00:05"),
            Timestamp("20130106 09:00:06"),
        ]
    ).as_unit(unit)
    mi = MultiIndex.from_product([dti, ["B", "A"]])

    exp = DataFrame(
        {
            "B": [
                np.nan,
                np.nan,
                0.9999999999999998,
                -1.0,
                1.0,
                -0.3273268353539892,
                0.9999999999999998,
                1.0,
                0.9999999999999998,
                1.0,
            ],
            "A": [
                np.nan,
                np.nan,
                -1.0,
                1.0000000000000002,
                -0.3273268353539892,
                0.9999999999999966,
                1.0,
                1.0000000000000002,
                1.0,
                1.0000000000000002,
            ],
        },
        index=mi,
    )

    df = DataFrame(
        {"B": [0, 1, 2, 4, 3], "A": [7, 4, 6, 9, 3]},
        index=dti,
    )

    res = df.rolling(window="3D").corr()

    tm.assert_frame_equal(exp, res)


@pytest.mark.parametrize("method", ["var", "sum", "mean", "skew", "kurt", "min", "max"])
def test_rolling_decreasing_indices(method):
    # [FUNCTION DOC]
    # GH 36933. Parametrized over var, sum, mean, skew, kurt, min, max. Verifies a
    # rolling window of 5 over a frame whose index is in decreasing order gives the
    # same values (reversed, with the unfilled warm-up elements excluded) as over an
    # increasing index, with tolerance 1e-12.
    """
    Make sure that decreasing indices give the same results as increasing indices.

    GH 36933
    """
    df = DataFrame({"values": np.arange(-15, 10) ** 2})
    df_reverse = DataFrame({"values": df["values"][::-1]}, index=df.index[::-1])

    increasing = getattr(df.rolling(window=5), method)()
    decreasing = getattr(df_reverse.rolling(window=5), method)()

    tm.assert_almost_equal(
        decreasing.values[::-1][:-4], increasing.values[4:], atol=1e-12
    )


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("2s", "right", [1.0, 3.0, 5.0, 3.0]),
        ("2s", "left", [0.0, 1.0, 3.0, 5.0]),
        ("2s", "both", [1.0, 3.0, 6.0, 5.0]),
        ("2s", "neither", [0.0, 1.0, 2.0, 3.0]),
        ("3s", "right", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "left", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "both", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "neither", [1.0, 3.0, 6.0, 5.0]),
    ],
)
def test_rolling_decreasing_indices_centered(window, closed, expected, frame_or_series):
    # [FUNCTION DOC]
    # GH 43927. Parametrized over window ("2s" or "3s"), closed side and expected
    # sums, for series and frame. Verifies a centered time-window sum over a
    # decreasing index gives the same values (matched by position in the reversed
    # index) as over the increasing index.
    """
    Ensure that a symmetrical inverted index return same result as non-inverted.
    """
    #  GH 43927

    index = date_range("2020", periods=4, freq="1s")
    df_inc = frame_or_series(range(4), index=index)
    df_dec = frame_or_series(range(4), index=index[::-1])

    expected_inc = frame_or_series(expected, index=index)
    expected_dec = frame_or_series(expected, index=index[::-1])

    result_inc = df_inc.rolling(window, closed=closed, center=True).sum()
    result_dec = df_dec.rolling(window, closed=closed, center=True).sum()

    tm.assert_equal(result_inc, expected_inc)
    tm.assert_equal(result_dec, expected_dec)


@pytest.mark.parametrize(
    "window,expected",
    [
        ("1ns", [1.0, 1.0, 1.0, 1.0]),
        ("3ns", [2.0, 3.0, 3.0, 2.0]),
    ],
)
def test_rolling_center_nanosecond_resolution(
    window, closed, expected, frame_or_series
):
    # [FUNCTION DOC]
    # Parametrized over windows "1ns" and "3ns" with expected counts, the closed
    # fixture, and series/frame. Verifies centered rolling sum of ones on a
    # nanosecond-spaced index gives the expected counts (windows of one or three
    # points), independent of closed side.
    index = date_range("2020", periods=4, freq="1ns")
    df = frame_or_series([1, 1, 1, 1], index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            "var",
            [
                float("nan"),
                43.0,
                float("nan"),
                136.333333,
                43.5,
                94.966667,
                182.0,
                318.0,
            ],
        ),
        (
            "mean",
            [float("nan"), 7.5, float("nan"), 21.5, 6.0, 9.166667, 13.0, 17.5],
        ),
        (
            "sum",
            [float("nan"), 30.0, float("nan"), 86.0, 30.0, 55.0, 91.0, 140.0],
        ),
        (
            "skew",
            [
                float("nan"),
                0.709296,
                float("nan"),
                0.407073,
                0.984656,
                0.919184,
                0.874674,
                0.842418,
            ],
        ),
        (
            "kurt",
            [
                float("nan"),
                -0.5916711736073559,
                float("nan"),
                -1.0028993131317954,
                -0.06103844629409494,
                -0.254143227116194,
                -0.37362637362637585,
                -0.45439658241367054,
            ],
        ),
    ],
)
def test_rolling_non_monotonic(method, expected):
    # [FUNCTION DOC]
    # GH 36933. Parametrized over var, mean, sum, skew, kurt with expected values
    # (from version 1.1.3, with sum and mean verified by hand). Uses a custom indexer
    # that alternates expanding and fixed-size windows so that windows are not
    # monotonic, to cover the rare non-monotonic branch of the rolling kernels.
    """
    Make sure the (rare) branch of non-monotonic indices is covered by a test.

    output from 1.1.3 is assumed to be the expected output. Output of sum/mean has
    manually been verified.

    GH 36933.
    """
    # Based on an example found in computation.rst
    use_expanding = [True, False, True, False, True, True, True, True]
    df = DataFrame({"values": np.arange(len(use_expanding)) ** 2})

    class CustomIndexer(BaseIndexer):
        # [CLASS DOC]
        # Local indexer subclass whose per-row behavior is controlled by a list of
        # flags (use expanding) supplied to the constructor; used only by the test that
        # defines it.
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            # [FUNCTION DOC]
            # Computes window bounds. Inputs: number of values, minimum periods, center,
            # closed and step (the last four unused). Returns two int64 arrays: for flagged
            # rows the window starts at zero and ends just after the row; otherwise it
            # starts at the row and spans the indexer's window size.
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=4, use_expanding=use_expanding)

    result = getattr(df.rolling(indexer), method)()
    expected = DataFrame({"values": expected})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    [
        ([0, 1, 2, 3, 4], 2),
        (date_range("2001-01-01", freq="D", periods=5), "2D"),
    ],
)
def test_rolling_corr_timedelta_index(index, window):
    # [FUNCTION DOC]
    # Regression test for GH 31286. Parametrized over an integer index with window 2
    # and a daily index with a "2D" window. Verifies rolling correlation of a series
    # whose first two values are replaced by zeros with a copy of the original gives
    # missing, missing, 1, 1, 1.
    # GH: 31286
    x = Series([1, 2, 3, 4, 5], index=index)
    y = x.copy()
    x.iloc[0:2] = 0.0
    result = x.rolling(window).corr(y)
    expected = Series([np.nan, np.nan, 1, 1, 1], index=index)
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "values,method,expected",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "first",
            [float("nan"), float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "first",
            [float("nan")] * 10,
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "last",
            [float("nan"), float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "last",
            [float("nan")] * 10,
        ),
    ],
)
def test_rolling_first_last(values, method, expected):
    # [FUNCTION DOC]
    # Regression test for GH#33155. Parametrized over values with and without missing
    # entries, first or last, and expected results. With window 3 (default minimum
    # periods), verifies the first two results are missing, then first returns the
    # first element of each window and last the final element; when every window
    # contains a missing value the result is all missing; for series and frame.
    # GH#33155
    x = Series(values)
    result = getattr(x.rolling(3), method)()
    expected = Series(expected)
    tm.assert_almost_equal(result, expected)

    x = DataFrame({"A": values})
    result = getattr(x.rolling(3), method)()
    expected = DataFrame({"A": expected})
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "values,method,expected",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "first",
            [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "first",
            [1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "last",
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            "last",
            [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
        ),
    ],
)
def test_rolling_first_last_no_minp(values, method, expected):
    # [FUNCTION DOC]
    # Regression test for GH#33155. Same as the first/last test but with minimum
    # periods zero: results exist from the start; with missing entries first and last
    # return the first or last valid value in the window.
    # GH#33155
    x = Series(values)
    result = getattr(x.rolling(3, min_periods=0), method)()
    expected = Series(expected)
    tm.assert_almost_equal(result, expected)

    x = DataFrame({"A": values})
    result = getattr(x.rolling(3, min_periods=0), method)()
    expected = DataFrame({"A": expected})
    tm.assert_almost_equal(result, expected)


def test_groupby_rolling_nan_included():
    # [FUNCTION DOC]
    # Regression test for GH 35542. Verifies a grouped rolling mean (window 1,
    # minimum periods 1) with groups containing missing keys, grouping with
    # dropping of missing keys disabled, keeps the missing group (placed last) in a
    # multi-level index where the missing key lives in the level values.
    # GH 35542
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = DataFrame(data)
    result = df.groupby("group", dropna=False).rolling(1, min_periods=1).mean()
    expected = DataFrame(
        {"B": [0.0, 2.0, 3.0, 1.0, 4.0]},
        # GH-38057 from_tuples puts the NaNs in the codes, result expects them
        # to be in the levels, at the moment
        # index=MultiIndex.from_tuples(
        #     [("g1", 0), ("g1", 2), ("g2", 3), (np.nan, 1), (np.nan, 4)],
        #     names=["group", None],
        # ),
        index=MultiIndex(
            [["g1", "g2", np.nan], [0, 1, 2, 3, 4]],
            [[0, 0, 1, 2, 2], [0, 2, 3, 1, 4]],
            names=["group", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_rolling_skew_kurt_numerical_stability(method):
    # [FUNCTION DOC]
    # Regression test for GH#6929. Parametrized over skew and kurt. Verifies rolling
    # (window 3) results on random data are unchanged when 50000 is added, and that
    # the input series is not modified by the first computation.
    # GH#6929
    ser = Series(np.random.default_rng(2).random(10))
    ser_copy = ser.copy()
    expected = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(ser, ser_copy)
    ser = ser + 50000
    result = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("method", "data", "values"),
    [
        (
            "skew",
            [3000000, 1, 1, 2, 3, 4, 999],
            [np.nan] * 3 + [2.0, 0.854563, 0.0, 1.999984],
        ),
        (
            "skew",
            [1e6, -1e6, 1, 2, 3, 4, 5, 6],
            [np.nan] * 3 + [-5.51135192e-06, -2.0, 0.0, 0.0, 0.0],
        ),
        (
            "kurt",
            [3000000, 1, 1, 2, 3, 4, 999],
            [np.nan] * 3 + [4.0, -1.289256, -1.2, 3.999946],
        ),
        (
            "kurt",
            [1e6, -1e6, 1, 2, 3, 4, 5, 6],
            [np.nan] * 3 + [1.5, 4.0, -1.2, -1.2, -1.2],
        ),
    ],
)
def test_rolling_skew_kurt_large_value_range(method, data, values):
    # [FUNCTION DOC]
    # Regression test for GH: 37557, 47461 and 61416. Parametrized over skew/kurt,
    # data containing values of very different magnitudes, and expected values.
    # Verifies rolling window 4 results match the expected numbers.
    # GH: 37557, 47461, 61416
    s = Series(data)
    result = getattr(s.rolling(4), method)()
    expected = Series(values)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_same_result_with_different_lengths(method):
    # [FUNCTION DOC]
    # Regression test for GH-54380. Parametrized over skew and kurt. Verifies the
    # rolling statistic (window 8) over the first 10 values of a longer series is
    # exactly equal to the first 10 results over the 12-value series, so results do
    # not depend on later data.
    # GH-54380
    len_smaller = 10
    len_bigger = 12
    window_size = 8

    rng = np.random.default_rng(2)
    data = rng.normal(loc=0.0, scale=1e3, size=len_bigger)
    window_smaller = Series(data[:len_smaller]).rolling(window_size)
    window_bigger = Series(data).rolling(window_size)

    result_smaller = getattr(window_smaller, method)()
    result_bigger = getattr(window_bigger, method)()

    result_bigger_trimmed = result_bigger[:len_smaller]

    tm.assert_series_equal(result_smaller, result_bigger_trimmed, check_exact=True)


def test_invalid_method():
    # [FUNCTION DOC]
    # Verifies an unknown method argument to rolling raises a value error saying the
    # method must be table or single.
    with pytest.raises(ValueError, match="method must be 'table' or 'single"):
        Series(range(1)).rolling(1, method="foo")


def test_rolling_descending_date_order_with_offset(frame_or_series):
    # [FUNCTION DOC]
    # Regression test for GH#40002. For series and frame, verifies a "1d" time
    # window with closed left over a daily index gives missing, 1, 2 (and the
    # lower-case day alias emits the future deprecation warning that it is
    # deprecated), and rolling over the reversed index with "1D" gives missing, 3, 2.
    # GH#40002
    msg = "'d' is deprecated and will be removed in a future version."

    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        idx = date_range(start="2020-01-01", end="2020-01-03", freq="1d")
        obj = frame_or_series(range(1, 4), index=idx)
        result = obj.rolling("1d", closed="left").sum()

    expected = frame_or_series([np.nan, 1, 2], index=idx)
    tm.assert_equal(result, expected)

    result = obj.iloc[::-1].rolling("1D", closed="left").sum()
    idx = date_range(start="2020-01-03", end="2020-01-01", freq="-1D")
    expected = frame_or_series([np.nan, 3, 2], index=idx)
    tm.assert_equal(result, expected)


def test_rolling_var_floating_artifact_precision():
    # [FUNCTION DOC]
    # Regression test for GH 37051 and GH 42064. Verifies rolling variance (window 3)
    # of 7, 5, 5, 5 matches missing, missing, 4/3, 0 within a tiny tolerance and that
    # the exact-zero positions coincide with the expected.
    # GH 37051
    s = Series([7, 5, 5, 5])
    result = s.rolling(3).var()
    expected = Series([np.nan, np.nan, 4 / 3, 0])
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)
    # GH 42064
    # new `roll_var` will output 0.0 correctly
    tm.assert_series_equal(result == 0, expected == 0)


def test_rolling_std_small_values():
    # [FUNCTION DOC]
    # Regression test for GH 37051. Verifies rolling standard deviation (window 2) of
    # three tiny close values is about 7.07e-9 for each full window.
    # GH 37051
    s = Series(
        [
            0.00000054,
            0.00000053,
            0.00000054,
        ]
    )
    result = s.rolling(2).std()
    expected = Series([np.nan, 7.071068e-9, 7.071068e-9])
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)


@pytest.mark.parametrize(
    "start, exp_values",
    [
        (1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]),
        (2, [0.001, 0.001, 0.0015, 0.00366666]),
    ],
)
def test_rolling_mean_all_nan_window_floating_artifacts(start, exp_values):
    # [FUNCTION DOC]
    # Regression test for GH#41053. Parametrized over a start offset (1 or 2) with
    # expected leading values. A frame containing runs of missing values is rolled
    # (window 5, minimum periods 0) after skipping the first rows; verifies the means
    # (without floating drift leaving residue where the window holds only missing
    # values, which should be missing) match the expected values, indexed from the
    # start offset.
    # GH#41053
    df = DataFrame(
        [
            0.03,
            0.03,
            0.001,
            np.nan,
            0.002,
            0.008,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.005,
            0.2,
        ]
    )

    values = [
        *exp_values,
        0.00366666,
        0.005,
        0.005,
        0.008,
        np.nan,
        np.nan,
        0.005,
        0.102500,
    ]
    expected = DataFrame(
        values,
        index=list(range(start, len(values) + start)),
    )
    result = df.iloc[start:].rolling(5, min_periods=0).mean()
    tm.assert_frame_equal(result, expected)


def test_rolling_sum_all_nan_window_floating_artifacts():
    # [FUNCTION DOC]
    # Regression test for GH#41053. Verifies rolling sum (window 3, minimum periods 0)
    # over values followed by missing entries gives the running sums ending in exactly
    # 0.0 when the window holds only missing values.
    # GH#41053
    df = DataFrame([0.002, 0.008, 0.005, np.nan, np.nan, np.nan])
    result = df.rolling(3, min_periods=0).sum()
    expected = DataFrame([0.002, 0.010, 0.015, 0.013, 0.005, 0.0])
    tm.assert_frame_equal(result, expected)


def test_rolling_zero_window():
    # [FUNCTION DOC]
    # Regression test for GH 22719. Verifies rolling min with window zero over one
    # value gives missing.
    # GH 22719
    s = Series(range(1))
    result = s.rolling(0).min()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("method", ["min", "max", "average"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans"])
def test_rank(window, method, pct, ascending, test_data):
    # [FUNCTION DOC]
    # Parametrized over windows 1, 3, 10, 20, method (min, max, average), percentile
    # flag, the ascending fixture and data kind (random floats, duplicates, floats
    # with missing and infinite values) of length 20. Verifies rolling rank equals
    # applying a function that ranks the window with the same options and takes the
    # last rank.
    length = 20
    if test_data == "default":
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )

    expected = ser.rolling(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    result = ser.rolling(window).rank(method=method, pct=pct, ascending=ascending)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans", "precision"])
def test_nunique(window, test_data):
    # [FUNCTION DOC]
    # Parametrized over windows and data kind (random floats, duplicates, missing and
    # infinite values, and values built so nearly equal floats may differ in the last
    # bit). Verifies rolling unique count equals applying a function that counts
    # unique values in the window.
    length = 20
    if test_data == "default":
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )
    elif test_data == "precision":
        ser = Series(
            data=[
                0.3,
                0.1 * 3,  # Not necessarily exactly 0.3
                0.6,
                0.2 * 3,  # Not necessarily exactly 0.6
                0.9,
                0.3 * 3,  # Not necessarily exactly 0.9
                0.5,
                0.1 * 5,  # Not necessarily exactly 0.5
                0.8,
                0.2 * 4,  # Not necessarily exactly 0.8
            ],
            dtype=np.float64,
        )

    expected = ser.rolling(window).apply(lambda x: x.nunique())
    result = ser.rolling(window).nunique()

    tm.assert_series_equal(result, expected)


def test_rolling_quantile_np_percentile():
    # [FUNCTION DOC]
    # Regression test for 9413 about the default quantile interpolation being
    # analogous to numpy percentile. Compares frame quantiles at 0.25, 0.5, 0.75 (axis
    # zero) against numpy percentiles.
    # KNOWN ISSUE: despite the name and purpose, no rolling object is created here; the test only compares the frame quantile method to numpy's percentile, so it does not exercise rolling quantile.
    # #9413: Tests that rolling window's quantile default behavior
    # is analogous to Numpy's percentile
    row = 10
    col = 5
    idx = date_range("20100101", periods=row, freq="B")
    df = DataFrame(
        np.random.default_rng(2).random(row * col).reshape((row, -1)), index=idx
    )

    df_quantile = df.quantile([0.25, 0.5, 0.75], axis=0)
    np_percentile = np.percentile(df, [25, 50, 75], axis=0)

    tm.assert_almost_equal(df_quantile.values, np.array(np_percentile))


@pytest.mark.parametrize("quantile", [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0],
        [0.0, np.nan, 0.2, np.nan, 0.4],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5],
        [0.5],
        [np.nan, 0.7, 0.6],
    ],
)
def test_rolling_quantile_interpolation_options(quantile, interpolation, data):
    # [FUNCTION DOC]
    # Parametrized over quantiles 0.0, 0.1, 0.45, 0.5, 1, five interpolation options
    # and seven data lists (including missing values, an all-missing list and a
    # single element). Verifies an expanding quantile (minimum periods 1) at the last
    # position equals the series quantile for the same interpolation; both missing
    # counts as equal; on 32-bit platforms a relative tolerance is used.
    # KNOWN ISSUE: the comment says it tests the rolling window's quantile, but the code uses an expanding window.
    # Tests that rolling window's quantile behavior is analogous to
    # Series' quantile for each interpolation option
    s = Series(data)

    q1 = s.quantile(quantile, interpolation)
    q2 = s.expanding(min_periods=1).quantile(quantile, interpolation).iloc[-1]

    if np.isnan(q1):
        assert np.isnan(q2)
    elif not IS64:
        # Less precision on 32-bit
        assert np.allclose([q1], [q2], rtol=1e-07, atol=0)
    else:
        assert q1 == q2


def test_invalid_quantile_value():
    # [FUNCTION DOC]
    # Verifies an unsupported interpolation name makes rolling quantile raise a value
    # error saying the interpolation is not supported.
    data = np.arange(5)
    s = Series(data)

    msg = "Interpolation 'invalid' is not supported"
    with pytest.raises(ValueError, match=msg):
        s.rolling(len(data), min_periods=1).quantile(0.5, interpolation="invalid")


def test_rolling_quantile_param():
    # [FUNCTION DOC]
    # Verifies rolling quantile argument validation: values below zero or above one
    # raise a value error naming the value and the allowed range; a string raises a
    # type error saying a real number is required.
    ser = Series([0.0, 0.1, 0.5, 0.9, 1.0])
    msg = "quantile value -0.1 not in \\[0, 1\\]"
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(-0.1)

    msg = "quantile value 10.0 not in \\[0, 1\\]"
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(10.0)

    msg = "must be real number, not str"
    with pytest.raises(TypeError, match=msg):
        ser.rolling(3).quantile("foo")


def test_rolling_std_1obs():
    # [FUNCTION DOC]
    # Verifies rolling standard deviation of single-observation windows (minimum
    # periods 1) is missing with default ddof and zero with ddof zero, and that a
    # window with fewer than the minimum periods gives missing at the third
    # position.
    vals = Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = vals.rolling(1, min_periods=1).std()
    expected = Series([np.nan] * 5)
    tm.assert_series_equal(result, expected)

    result = vals.rolling(1, min_periods=1).std(ddof=0)
    expected = Series([0.0] * 5)
    tm.assert_series_equal(result, expected)

    result = Series([np.nan, np.nan, 3, 4, 5]).rolling(3, min_periods=2).std()
    assert np.isnan(result[2])


def test_rolling_std_neg_sqrt():
    # [FUNCTION DOC]
    # Unit test from Bottleneck for the negative square root case: rolling (window 3)
    # and ewm (span 3) standard deviations of tiny nearly-constant values must be
    # finite from the third position on.
    # unit test from Bottleneck

    # Test move_nanstd for neg sqrt.

    a = Series(
        [
            0.0011448196318903589,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
        ]
    )
    b = a.rolling(window=3).std()
    assert np.isfinite(b[2:]).all()

    b = a.ewm(span=3).std()
    assert np.isfinite(b[2:]).all()


def test_step_not_integer_raises():
    # [FUNCTION DOC]
    # Verifies a non-integer step makes rolling raise a value error saying step must
    # be an integer.
    with pytest.raises(ValueError, match="step must be an integer"):
        DataFrame(range(2)).rolling(1, step="foo")


def test_step_not_positive_raises():
    # [FUNCTION DOC]
    # Verifies a negative step makes rolling raise a value error saying step must be
    # at least 0.
    with pytest.raises(ValueError, match="step must be >= 0"):
        DataFrame(range(2)).rolling(1, step=-1)


@pytest.mark.parametrize(
    ["values", "window", "min_periods", "expected"],
    [
        [
            [20, 10, 10, np.inf, 1, 1, 2, 3],
            3,
            1,
            [np.nan, 50, 100 / 3, 0, 40.5, 0, 1 / 3, 1],
        ],
        [
            [20, 10, 10, np.nan, 10, 1, 2, 3],
            3,
            1,
            [np.nan, 50, 100 / 3, 0, 0, 40.5, 73 / 3, 1],
        ],
        [
            [np.nan, 5, 6, 7, 5, 5, 5],
            3,
            3,
            [np.nan] * 3 + [1, 1, 4 / 3, 0],
        ],
        [
            [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
            3,
            3,
            [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [1 / 3, 0],
        ],
        [
            [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
            3,
            3,
            [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [16 / 3, 0],
        ],
        [
            [5, 7] * 4,
            3,
            3,
            [np.nan] * 2 + [4 / 3] * 6,
        ],
        [
            [5, 7, 5, np.nan, 7, 5, 7],
            3,
            2,
            [np.nan, 2, 4 / 3] + [2] * 3 + [4 / 3],
        ],
    ],
)
def test_rolling_var_same_value_count_logic(values, window, min_periods, expected):
    # [FUNCTION DOC]
    # Regression test for GH 42064. Parametrized over data (with infinities, missing
    # values and repeated values), window, minimum periods and expected variances.
    # Verifies rolling variance is close to the expected values with exact zeros
    # matching where consecutive equal values are found, and that standard
    # deviation equals the square root of the expected variance with the same zeros.
    # GH 42064.

    expected = Series(expected)
    sr = Series(values)

    # With new algo implemented, result will be set to .0 in rolling var
    # if sufficient amount of consecutively same values are found.
    result_var = sr.rolling(window, min_periods=min_periods).var()

    # use `assert_series_equal` twice to check for equality,
    # because `check_exact=True` will fail in 32-bit tests due to
    # precision loss.

    # 1. result should be close to correct value
    # non-zero values can still differ slightly from "truth"
    # as the result of online algorithm
    tm.assert_series_equal(result_var, expected)
    # 2. zeros should be exactly the same since the new algo takes effect here
    tm.assert_series_equal(expected == 0, result_var == 0)

    # std should also pass as it's just a sqrt of var
    result_std = sr.rolling(window, min_periods=min_periods).std()
    tm.assert_series_equal(result_std, np.sqrt(expected))
    tm.assert_series_equal(expected == 0, result_std == 0)


def test_rolling_mean_sum_floating_artifacts():
    # [FUNCTION DOC]
    # Regression test for GH 42064. Verifies rolling (window 3) mean and sum of a
    # series ending in zeros are exactly zero for the last three results (no
    # leftover floating artifacts).
    # GH 42064.

    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(3)
    result = r.mean()
    assert (result[-3:] == 0).all()
    result = r.sum()
    assert (result[-3:] == 0).all()


def test_rolling_skew_kurt_floating_artifacts():
    # [FUNCTION DOC]
    # Regression test for GH 42064 and 46431. Verifies rolling (window 4) skewness
    # and kurtosis of a series ending in zeros give the expected values, with exactly
    # 0.0 skew and -3.0 kurtosis for all-zero windows.
    # GH 42064 46431

    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(4)
    result = r.skew()
    expected = Series([np.nan, np.nan, np.nan, 1.9619045191072484, 2.0, 0.0, 0.0])
    tm.assert_series_equal(result, expected)
    result = r.kurt()
    expected = Series([np.nan, np.nan, np.nan, 3.8636048803878786, 4.0, -3.0, -3.0])
    tm.assert_series_equal(result, expected)


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # [FUNCTION DOC]
    # Regression test for GH#46560. For each rolling operation, verifies a frame
    # with two numeric columns and one object column, for the numeric_only
    # fixture, equals the aggregation of the corresponding columns (object column
    # dropped when numeric_only is true) cast to float.
    # GH#46560
    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    rolling = df.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    result = op(numeric_only=numeric_only)

    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
    assert list(expected.columns) == columns

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over corr/cov, with or without the
    # other-frame argument, and numeric_only. Verifies the result equals the same
    # operation on a float-cast frame of the selected columns.
    # GH#46560
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    arg = (df,) if use_arg else ()
    rolling = df.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    result = op(*arg, numeric_only=numeric_only)

    # Compare result to op using float dtypes, dropping c when numeric_only is True
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    rolling2 = df2.rolling(2, min_periods=1)
    op2 = getattr(rolling2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over integer and object dtype.
    # Object dtype with numeric_only true must raise a not implemented error saying
    # the rolling operation does not implement numeric_only; otherwise the result
    # equals the series aggregate cast to float.
    # GH#46560
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    rolling = ser.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    if numeric_only and dtype is object:
        msg = f"Rolling.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result = op(numeric_only=numeric_only)
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over corr/cov, with or without an
    # other argument, dtype integer or object, and numeric_only. Object dtype with
    # numeric_only true must raise a not implemented error; otherwise the result
    # equals the same operation on a float-cast copy.
    # GH#46560
    ser = Series([1, 2, 3], dtype=dtype)
    arg = (ser,) if use_arg else ()
    rolling = ser.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    if numeric_only and dtype is object:
        msg = f"Rolling.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)

        ser2 = ser.astype(float)
        arg2 = (ser2,) if use_arg else ()
        rolling2 = ser2.rolling(2, min_periods=1)
        op2 = getattr(rolling2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Prague"])
def test_rolling_timedelta_window_non_nanoseconds(unit, tz):
    # [FUNCTION DOC]
    # Regression tests for GH#55106 and GH#55026. Parametrized over time zones (none,
    # UTC, Europe/Prague) and the time unit fixture. Verifies a "1s" rolling sum
    # gives the same result whether the index has nanosecond or the other unit, and
    # a rolling max over a timedelta of four days on a daily index gives the same
    # values for nanosecond and the other unit.
    # Test Sum, GH#55106
    df_time = DataFrame(
        {"A": range(5)},
        index=date_range("2013-01-01", freq="1s", periods=5, tz=tz, unit="ns"),
    )
    sum_in_nanosecs = df_time.rolling("1s").sum()
    # microseconds / milliseconds should not break the correct rolling
    df_time.index = df_time.index.as_unit(unit)
    sum_in_microsecs = df_time.rolling("1s").sum()
    sum_in_microsecs.index = sum_in_microsecs.index.as_unit("ns")
    tm.assert_frame_equal(sum_in_nanosecs, sum_in_microsecs)

    # Test max, GH#55026
    ref_dates = date_range("2023-01-01", "2023-01-10", unit="ns", tz=tz)
    ref_series = Series(0, index=ref_dates)
    ref_series.iloc[0] = 1
    ref_max_series = ref_series.rolling(Timedelta(days=4)).max()

    dates = date_range("2023-01-01", "2023-01-10", unit=unit, tz=tz)
    series = Series(0, index=dates)
    series.iloc[0] = 1
    max_series = series.rolling(Timedelta(days=4)).max()

    ref_df = DataFrame(ref_max_series)
    df = DataFrame(max_series)
    df.index = df.index.as_unit("ns")

    tm.assert_frame_equal(ref_df, df)


class PrescribedWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # HELPER: test indexer whose window bounds are given explicitly as two arrays
    # (start and end positions per row) at construction; extends the base indexer.
    # Bounds are clipped to the valid range when requested. State: start and end
    # arrays.
    def __init__(self, start, end):
        # [FUNCTION DOC]
        # HELPER: constructor. Takes start and end integer arrays, stores them in
        # private attributes, then runs the base class constructor with defaults.
        self._start = start
        self._end = end
        super().__init__()

    def get_window_bounds(
        self, num_values=None, min_periods=None, center=None, closed=None, step=None
    ):
        # [FUNCTION DOC]
        # HELPER: returns the stored bounds. Inputs: number of values (defaults to the
        # length of the stored start array when none), and unused minimum periods,
        # center, closed and step. Returns start and end arrays clipped to the range
        # zero through number of values.
        if num_values is None:
            num_values = len(self._start)
        start = np.clip(self._start, 0, num_values)
        end = np.clip(self._end, 0, num_values)
        return start, end


class TestMinMax:
    # [CLASS DOC]
    # Tests for rolling min and max with arbitrary (non-monotonic) window bounds
    # supplied through the prescribed-bounds indexer helper. No state.
    @pytest.mark.parametrize(
        "is_max, has_nan, exp_list",
        [
            (True, False, [3.0, 5.0, 2.0, 5.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            (True, True, [3.0, 4.0, 2.0, 4.0, 1.0, 4.0, 6.0, 7.0, 7.0, 9.0]),
            (False, False, [3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 7.0, 0.0]),
            (False, True, [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 6.0, 6.0, 7.0, 1.0]),
        ],
    )
    def test_minmax(self, is_max, has_nan, exp_list):
        # [FUNCTION DOC]
        # Parametrized over max/min, whether some data values are missing, and expected
        # results. A ten-row frame with prescribed start and end bounds per row is rolled;
        # the max or min per window (ignoring missing values) must equal the expected
        # series.
        nan_idx = [0, 5, 8]
        df = DataFrame(
            {
                "data": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0, 7.0, 8.0, 9.0],
                "start": [2, 0, 3, 0, 4, 0, 5, 5, 7, 3],
                "end": [3, 4, 4, 5, 5, 6, 7, 8, 9, 10],
            }
        )
        if has_nan:
            df.loc[nan_idx, "data"] = np.nan
        expected = Series(exp_list, name="data")
        r = df.data.rolling(
            PrescribedWindowIndexer(df.start.to_numpy(), df.end.to_numpy())
        )
        if is_max:
            result = r.max()
        else:
            result = r.min()

        tm.assert_series_equal(result, expected)

    def test_wrong_order(self):
        # [FUNCTION DOC]
        # Verifies that bounds whose start or end positions are not monotonic (the
        # fourth window ends where the third does and starts before the third's start)
        # make rolling max raise a value error saying the start/end ordering requirement
        # is violated at row 3.
        start = np.array(range(5), dtype=np.int64)
        end = start + 1
        end[3] = end[2]
        start[3] = start[2] - 1

        df = DataFrame({"data": start * 1.0, "start": start, "end": end})

        r = df.data.rolling(PrescribedWindowIndexer(start, end))
        with pytest.raises(
            ValueError, match="Start/End ordering requirement is violated at index 3"
        ):
            r.max()
