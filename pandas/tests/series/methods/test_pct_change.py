# [MODULE DOC]
# Tests for Series.pct_change: periods, freq, handling of missing values and
# duplicate index. Uses the datetime_series fixture. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm


class TestSeriesPctChange:
    # [CLASS DOC]
    # Groups the main pct_change tests. Stateless.
    def test_pct_change(self, datetime_series):
        # [FUNCTION DOC]
        # pct_change equals the ratio to the shifted values minus one: for the default
        # period, for two periods (after forward fill of missing values) and for a day
        # frequency (5D).
        rs = datetime_series.pct_change()
        tm.assert_series_equal(rs, datetime_series / datetime_series.shift(1) - 1)

        rs = datetime_series.pct_change(2)
        filled = datetime_series.ffill()
        tm.assert_series_equal(rs, filled / filled.shift(2) - 1)

        rs = datetime_series.pct_change(freq="5D")
        filled = datetime_series.ffill()
        tm.assert_series_equal(
            rs, (filled / filled.shift(freq="5D") - 1).reindex_like(filled)
        )

    def test_pct_change_with_duplicate_axis(self):
        # [FUNCTION DOC]
        # GH#28664: pct_change with a business-day freq on a daily index gives NaN,
        # infinity, NaN, NaN, 3.0, as reasoned in the referenced pull request.
        # GH#28664
        common_idx = date_range("2019-11-14", periods=5, freq="D")
        result = Series(range(5), common_idx).pct_change(freq="B")

        # the reason that the expected should be like this is documented at PR 28681
        expected = Series([np.nan, np.inf, np.nan, np.nan, 3.0], common_idx)

        tm.assert_series_equal(result, expected)

    def test_pct_change_shift_over_nas(self):
        # [FUNCTION DOC]
        # A NaN breaks the chain: the result has NaN at the missing position and the
        # position after it (no fill), with 0.5 and 0.2 elsewhere.
        s = Series([1.0, 1.5, np.nan, 2.5, 3.0])
        chg = s.pct_change()
        expected = Series([np.nan, 0.5, np.nan, np.nan, 0.2])
        tm.assert_series_equal(chg, expected)

    @pytest.mark.parametrize("freq, periods", [("5B", 5), ("3B", 3), ("14B", 14)])
    def test_pct_change_periods_freq(self, freq, periods, datetime_series):
        # [FUNCTION DOC]
        # GH#7292, parametrized over business-day frequencies and matching periods:
        # pct_change with freq equals pct_change with the corresponding periods, also for a
        # Series of all-missing object data.
        # KNOWN ISSUE: the variable named empty is not empty: it is a Series with the
        # full index and no values (all missing).
        # GH#7292
        rs_freq = datetime_series.pct_change(freq=freq)
        rs_periods = datetime_series.pct_change(periods)
        tm.assert_series_equal(rs_freq, rs_periods)

        empty_ts = Series(index=datetime_series.index, dtype=object)
        rs_freq = empty_ts.pct_change(freq=freq)
        rs_periods = empty_ts.pct_change(periods)
        tm.assert_series_equal(rs_freq, rs_periods)


def test_pct_change_with_duplicated_indices():
    # [FUNCTION DOC]
    # GH30463: duplicated index labels do not affect the result, computed by
    # position.
    # GH30463
    s = Series([np.nan, 1, 2, 3, 9, 18], index=["a", "b"] * 3)
    result = s.pct_change()
    expected = Series([np.nan, np.nan, 1.0, 0.5, 2.0, 1.0], index=["a", "b"] * 3)
    tm.assert_series_equal(result, expected)


def test_pct_change_no_warning_na_beginning():
    # [FUNCTION DOC]
    # GH#54981: leading None values give NaN without emitting a warning.
    # GH#54981
    ser = Series([None, None, 1, 2, 3])
    result = ser.pct_change()
    expected = Series([np.nan, np.nan, np.nan, 1, 0.5])
    tm.assert_series_equal(result, expected)


def test_pct_change_empty():
    # [FUNCTION DOC]
    # GH 57056: an empty float Series with periods set to zero returns an equal
    # empty Series.
    # GH 57056
    ser = Series([], dtype="float64")
    expected = ser.copy()
    result = ser.pct_change(periods=0)
    tm.assert_series_equal(expected, result)
