# [MODULE DOC]
# Tests for rolling skewness and kurtosis: agreement with the scipy statistics
# functions (unbiased variant) on series, frames and time-rule resampled data,
# missing value and minimum periods handling, centering, edge cases (constant
# data, too-small windows, hand-computed values) and floating point error on
# equal values. Tests comparing with scipy skip when scipy is missing. Each
# comparison test is parametrized over (scipy function, rolling method) pairs
# for kurtosis and skew.
from functools import partial

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    concat,
    isna,
    notna,
)
import pandas._testing as tm

from pandas.tseries import offsets


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_series(series, sp_func, roll_func):
    # [FUNCTION DOC]
    # Verifies rolling kurtosis or skewness (window 50) on the series fixture returns
    # a series whose last value equals the unbiased scipy statistic of the last 50
    # values.
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    result = getattr(series.rolling(50), roll_func)()
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[-50:]))


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_frame(raw, frame, sp_func, roll_func):
    # [FUNCTION DOC]
    # Frame counterpart (raw fixture): the last row equals the unbiased scipy
    # statistic applied column by column to the last 50 rows (names not compared).
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    result = getattr(frame.rolling(50), roll_func)()
    assert isinstance(result, DataFrame)
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_time_rule_series(series, sp_func, roll_func):
    # [FUNCTION DOC]
    # On the series fixture sampled every other point and resampled to business days,
    # verifies the last result (window 25, minimum periods 10) equals the unbiased
    # scipy statistic of the original data truncated to the last 25 business days.
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    win = 25
    ser = series[::2].resample("B").mean()
    series_result = getattr(ser.rolling(window=win, min_periods=10), roll_func)()
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_time_rule_frame(raw, frame, sp_func, roll_func):
    # [FUNCTION DOC]
    # Frame counterpart of the time-rule test (raw fixture): the final row equals
    # the scipy statistic applied column by column (names not compared).
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    win = 25
    frm = frame[::2].resample("B").mean()
    frame_result = getattr(frm.rolling(window=win, min_periods=10), roll_func)()
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_nans(sp_func, roll_func):
    # [FUNCTION DOC]
    # On random data with ten missing values at each end, verifies: window 50 with
    # minimum periods 30 gives the scipy statistic of the middle values at the end;
    # window 20 with minimum periods 15 first yields at position 24 and becomes
    # missing again five from the end; window 10 with minimum periods 5 first yields
    # at position 4; and minimum periods zero equals one.
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = getattr(obj.rolling(50, min_periods=30), roll_func)()
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))

    # min_periods is working correctly
    result = getattr(obj.rolling(20, min_periods=15), roll_func)()
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    result = getattr(obj2.rolling(10, min_periods=5), roll_func)()
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    result0 = getattr(obj.rolling(20, min_periods=0), roll_func)()
    result1 = getattr(obj.rolling(20, min_periods=1), roll_func)()
    tm.assert_almost_equal(result0, result1)


@pytest.mark.parametrize("minp", [0, 99, 100])
@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_min_periods(series, minp, roll_func, step):
    # [FUNCTION DOC]
    # Parametrized over kurt and skew, minimum periods 0, 99, 100 and step. Verifies
    # a window one longer than the series and a window equal to the length give
    # identical missing patterns and matching values where present.
    result = getattr(
        series.rolling(len(series) + 1, min_periods=minp, step=step), roll_func
    )()
    expected = getattr(
        series.rolling(len(series), min_periods=minp, step=step), roll_func
    )()
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_center(roll_func):
    # [FUNCTION DOC]
    # Verifies a centered rolling window of 20 equals the non-centered result on
    # data padded with nine trailing missing values, dropping the first nine results
    # and resetting the index (kurt and skew).
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = getattr(obj.rolling(20, center=True), roll_func)()
    expected = (
        getattr(concat([obj, Series([np.nan] * 9)]).rolling(20), roll_func)()
        .iloc[9:]
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_center_reindex_series(series, roll_func):
    # [FUNCTION DOC]
    # Verifies a centered rolling window of 25 on the series fixture equals the
    # non-centered result on the series padded with twelve extra labels, shifted back
    # by twelve and reindexed (kurt and skew).
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    series_xp = (
        getattr(
            series.reindex(list(series.index) + s).rolling(window=25),
            roll_func,
        )()
        .shift(-12)
        .reindex(series.index)
    )
    series_rs = getattr(series.rolling(window=25, center=True), roll_func)()
    tm.assert_series_equal(series_xp, series_rs)


@pytest.mark.slow
@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_center_reindex_frame(frame, roll_func):
    # [FUNCTION DOC]
    # Marked slow. Frame counterpart of the series centered-reindex test (kurt and
    # skew).
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    frame_xp = (
        getattr(
            frame.reindex(list(frame.index) + s).rolling(window=25),
            roll_func,
        )()
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = getattr(frame.rolling(window=25, center=True), roll_func)()
    tm.assert_frame_equal(frame_xp, frame_rs)


def test_rolling_skew_edge_cases(step):
    # [FUNCTION DOC]
    # Verifies skewness edge cases with step: constant data with window 5 gives
    # missing for the first four positions then 0.0 at the fifth; a window of 2
    # gives all missing; and a hand-computed five-value example with window 4 gives
    # missing, missing, missing, 0.177994, 1.548824.
    # KNOWN ISSUE: the first inline comment says the constant-data case yields all missing values, but the expectation (and the following comment) says the full-window value is 0.0, so the first comment is inaccurate.
    expected = Series([np.nan] * 4 + [0.0])[::step]
    # yields all NaN (0 variance)
    d = Series([1] * 5)
    x = d.rolling(window=5, step=step).skew()
    # index 4 should be 0 as it contains 5 same obs
    tm.assert_series_equal(expected, x)

    expected = Series([np.nan] * 5)[::step]
    # yields all NaN (window too small)
    d = Series(np.random.default_rng(2).standard_normal(5))
    x = d.rolling(window=2, step=step).skew()
    tm.assert_series_equal(expected, x)

    # yields [NaN, NaN, NaN, 0.177994, 1.548824]
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    expected = Series([np.nan, np.nan, np.nan, 0.177994, 1.548824])[::step]
    x = d.rolling(window=4, step=step).skew()
    tm.assert_series_equal(expected, x)


def test_rolling_kurt_edge_cases(step):
    # [FUNCTION DOC]
    # Verifies kurtosis edge cases with step: constant data with window 5 gives
    # missing for the first four positions then -3.0; a window of 3 gives all
    # missing; and a hand-computed five-value example with window 4 gives missing,
    # missing, missing, 1.224307, 2.671499.
    # KNOWN ISSUE: the first inline comment says the constant-data case yields all missing values, but the expected result ends with -3.0 at the full window, so the comment is inaccurate.
    expected = Series([np.nan] * 4 + [-3.0])[::step]

    # yields all NaN (0 variance)
    d = Series([1] * 5)
    x = d.rolling(window=5, step=step).kurt()
    tm.assert_series_equal(expected, x)

    # yields all NaN (window too small)
    expected = Series([np.nan] * 5)[::step]
    d = Series(np.random.default_rng(2).standard_normal(5))
    x = d.rolling(window=3, step=step).kurt()
    tm.assert_series_equal(expected, x)

    # yields [NaN, NaN, NaN, 1.224307, 2.671499]
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    expected = Series([np.nan, np.nan, np.nan, 1.224307, 2.671499])[::step]
    x = d.rolling(window=4, step=step).kurt()
    tm.assert_series_equal(expected, x)


def test_rolling_skew_eq_value_fperr(step):
    # [FUNCTION DOC]
    # Regression test for 18804 (updated by 46717). Verifies rolling skewness with
    # window 10 over 15 equal floating point values is zero (not missing) from the
    # tenth position on, and missing before it.
    # KNOWN ISSUE: the first inline comment still says equal values should give a missing result; the second comment and the assertions say zero.
    # #18804 all rolling skew for all equal values should return Nan
    # #46717 update: all equal values should return 0 instead of NaN
    a = Series([1.1] * 15).rolling(window=10, step=step).skew()
    assert (a[a.index >= 9] == 0).all()
    assert a[a.index < 9].isna().all()


def test_rolling_kurt_eq_value_fperr(step):
    # [FUNCTION DOC]
    # Regression test for 18804 (updated by 46717). Verifies rolling kurtosis with
    # window 10 over 15 equal floating point values is -3 from the tenth position on,
    # and missing before it.
    # KNOWN ISSUE: the first inline comment still says equal values should give a missing result; the second comment and the assertions say -3.
    # #18804 all rolling kurt for all equal values should return Nan
    # #46717 update: all equal values should return -3 instead of NaN
    a = Series([1.1] * 15).rolling(window=10, step=step).kurt()
    assert (a[a.index >= 9] == -3).all()
    assert a[a.index < 9].isna().all()
