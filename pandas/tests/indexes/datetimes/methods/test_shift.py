# [MODULE DOC]
# Tests for DatetimeIndex.shift (and Series.shift with a freq on a datetime index): empty and
# timezone-aware cases, frequency handling, errors without freq, DST crossing, and business-day
# and business-month offsets.
from datetime import datetime
import zoneinfo

import pytest

from pandas.errors import NullFrequencyError

import pandas as pd
from pandas import (
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndexShift:
    # [CLASS DOC]
    # Groups shift tests for DatetimeIndex; no state.
    # -------------------------------------------------------------
    # DatetimeIndex.shift is used in integer addition

    def test_dti_shift_tzaware(self, tz_naive_fixture, unit):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture and unit. Verifies shifting an empty index is a no-op, and shifting a
        # three-hour index by zero, three and minus three hours (passing an hourly freq) gives the
        # expected stamps with name, timezone and hourly freq kept.
        # GH#9903
        tz = tz_naive_fixture
        idx = DatetimeIndex([], name="xxx", tz=tz).as_unit(unit)
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        tm.assert_index_equal(idx.shift(3, freq="h"), idx)

        idx = DatetimeIndex(
            ["2011-01-01 10:00", "2011-01-01 11:00", "2011-01-01 12:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        exp = DatetimeIndex(
            ["2011-01-01 13:00", "2011-01-01 14:00", "2011-01-01 15:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        tm.assert_index_equal(idx.shift(3, freq="h"), exp)
        exp = DatetimeIndex(
            ["2011-01-01 07:00", "2011-01-01 08:00", "2011-01-01 09:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        tm.assert_index_equal(idx.shift(-3, freq="h"), exp)

    def test_dti_shift_freqs(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Verifies shifting a five-day range by one and minus one period moves values by a
        # day, and shifting by three with a two-day freq moves by six days; the resulting freq stays daily.
        # test shift for DatetimeIndex and non DatetimeIndex
        # GH#8083
        drange = date_range("20130101", periods=5, unit=unit)
        result = drange.shift(1)
        expected = DatetimeIndex(
            ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05", "2013-01-06"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)

        result = drange.shift(-1)
        expected = DatetimeIndex(
            ["2012-12-31", "2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)

        result = drange.shift(3, freq="2D")
        expected = DatetimeIndex(
            ["2013-01-07", "2013-01-08", "2013-01-09", "2013-01-10", "2013-01-11"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)

    def test_dti_shift_int(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Verifies that adding or subtracting a multiple of the freq offset equals shifting by
        # that many periods.
        rng = date_range("1/1/2000", periods=20, unit=unit)

        result = rng + 5 * rng.freq
        expected = rng.shift(5)
        tm.assert_index_equal(result, expected)

        result = rng - 5 * rng.freq
        expected = rng.shift(-5)
        tm.assert_index_equal(result, expected)

    def test_dti_shift_no_freq(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Regression test: shifting an index with no freq raises a null-frequency error
        # saying it cannot shift without a freq.
        # GH#19147
        dti = DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None).as_unit(unit)
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):
            dti.shift(2)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_shift_localized(self, tzstr, unit):
        # [FUNCTION DOC]
        # Parametrized over two timezone spellings. Verifies that shifting a localized weekly range by a
        # period and ten-minute frequency string keeps the timezone.
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI", unit=unit)
        dr_tz = dr.tz_localize(tzstr)

        result = dr_tz.shift(1, "10min")
        assert result.tz == dr_tz.tz

    def test_dti_shift_across_dst(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Regression test: shifting a Series on an hourly Chicago index by an hour across the
        # DST end gives the series with an index moved by one position.
        # GH 8616
        idx = date_range(
            "2013-11-03", tz="America/Chicago", periods=7, freq="h", unit=unit
        )
        ser = Series(index=idx[:-1], dtype=object)
        result = ser.shift(freq="h")
        expected = Series(index=idx[1:], dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "shift, result_time",
        [
            [0, "2014-11-14 00:00:00"],
            [-1, "2014-11-13 23:00:00"],
            [1, "2014-11-14 01:00:00"],
        ],
    )
    def test_dti_shift_near_midnight(self, shift, result_time, unit):
        # [FUNCTION DOC]
        # Parametrized over shift counts (0, -1, 1) and expected result time. Regression test: shifting a
        # Series indexed by a US/Eastern midnight datetime by hours yields the expected timestamp.
        # GH 8616
        tz = zoneinfo.ZoneInfo("US/Eastern")
        dt_est = datetime(2014, 11, 14, 0, tzinfo=tz)
        idx = DatetimeIndex([dt_est]).as_unit(unit)
        ser = Series(data=[1], index=idx)
        result = ser.shift(shift, freq="h")
        exp_index = DatetimeIndex([result_time], tz=tz).as_unit(unit)
        expected = Series(1, index=exp_index)
        tm.assert_series_equal(result, expected)

    def test_shift_periods(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Verifies shifting by zero given as the keyword periods or positionally returns the
        # same index.
        # GH#22458 : argument 'n' was deprecated in favor of 'periods'
        idx = date_range(
            start=datetime(2009, 1, 1), end=datetime(2010, 1, 1), periods=3, unit=unit
        )
        tm.assert_index_equal(idx.shift(periods=0), idx)
        tm.assert_index_equal(idx.shift(0), idx)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_shift_bday(self, freq, unit):
        # [FUNCTION DOC]
        # Parametrized over business-day frequencies B and C. Verifies shifting by five, minus five and
        # zero periods maps values to the corresponding positions of the original and retains freq.
        rng = date_range(
            datetime(2009, 1, 1), datetime(2010, 1, 1), freq=freq, unit=unit
        )
        shifted = rng.shift(5)
        assert shifted[0] == rng[5]
        assert shifted.freq == rng.freq

        shifted = rng.shift(-5)
        assert shifted[5] == rng[0]
        assert shifted.freq == rng.freq

        shifted = rng.shift(0)
        assert shifted[0] == rng[0]
        assert shifted.freq == rng.freq

    def test_shift_bmonth(self, performance_warning, unit):
        # [FUNCTION DOC]
        # Uses performance_warning and unit. Verifies shifting a business-month-end range with a
        # business-day offset moves the first stamp by one business day; with custom business day offset
        # the same holds but a performance warning is expected.
        rng = date_range(
            datetime(2009, 1, 1),
            datetime(2010, 1, 1),
            freq=pd.offsets.BMonthEnd(),
            unit=unit,
        )
        shifted = rng.shift(1, freq=pd.offsets.BDay())
        assert shifted[0] == rng[0] + pd.offsets.BDay()

        rng = date_range(
            datetime(2009, 1, 1),
            datetime(2010, 1, 1),
            freq=pd.offsets.BMonthEnd(),
            unit=unit,
        )
        with tm.assert_produces_warning(performance_warning):
            shifted = rng.shift(1, freq=pd.offsets.CDay())
            assert shifted[0] == rng[0] + pd.offsets.CDay()

    def test_shift_empty(self, unit):
        # [FUNCTION DOC]
        # Uses unit. Regression test: shifting a single-element business-month-end range returns the
        # same index.
        # GH#14811
        dti = date_range(start="2016-10-21", end="2016-10-21", freq="BME", unit=unit)
        result = dti.shift(1)
        tm.assert_index_equal(result, dti)
