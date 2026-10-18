# [MODULE DOC]
# Tests for resampling with period indexes and several time-zone, daylight-saving and
# closed/label edge cases for datetime resampling that were historically tested together.
# Covers frequency conversion, up- and downsampling with start/end conventions across many
# anchored frequencies, error cases for incompatible or invalid frequencies, fill with limit,
# missing period values, offsets, the private helper that computes period range edges,
# and behavior with time zones. Provides a fixture that returns a factory for random-valued
# period-indexed series and one test class plus a few module-level tests. Warnings about the
# deprecated business-day period frequency are suppressed in the factory and asserted in
# specific tests.

from datetime import (
    datetime,
    timezone,
)
import re
import warnings
import zoneinfo

import dateutil
import numpy as np
import pytest

from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
from pandas.core.resample import _get_period_range_edges

from pandas.tseries import offsets


@pytest.fixture
def simple_period_range_series():
    # [FUNCTION DOC]
    # Fixture returning a factory function that builds a series with a period index and random
    # normal data. The factory itself is the product of this fixture; see the nested factory.
    """
    Series with period range index and random data for test purposes.
    """

    def _simple_period_range_series(start, end, freq="D"):
        # [FUNCTION DOC]
        # HELPER: factory returned by the simple period series fixture. Parameters: start and end
        # of the range (any period-parsable values) and freq (default daily). Builds the period range
        # while suppressing future warnings about the business-day period; returns a series of random
        # normal values (seeded) indexed by that range, with one value per period.
        with warnings.catch_warnings():
            # suppress Period[B] deprecation warning
            msg = "|".join(["Period with BDay freq", r"PeriodDtype\[B\] is deprecated"])
            warnings.filterwarnings(
                "ignore",
                msg,
                category=FutureWarning,
            )
            rng = period_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    return _simple_period_range_series


class TestPeriodIndex:
    # [CLASS DOC]
    # Tests of resampling for period-indexed series and frames; also several datetime/time zone
    # resample cases. Stateless; most tests use the series factory fixture.
    @pytest.mark.parametrize("freq", ["2D", "1h", "2h"])
    def test_asfreq(self, frame_or_series, freq):
        # [FUNCTION DOC]
        # Regression test (GH 12884, GH 15944), parametrized by frequency (two days, one hour, two
        # hours) and frame-or-series fixture. Verifies the frequency conversion on a period-indexed
        # object equals converting through timestamps and reindexing on the target range (left
        # inclusive), and that round trip through timestamps and periods preserves it.
        # KNOWN ISSUE: the first expected value and the first result are computed with the identical expression, so that first comparison is trivially true.
        # GH 12884, 15944

        obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))

        expected = obj.to_timestamp().resample(freq).asfreq()
        result = obj.to_timestamp().resample(freq).asfreq()
        tm.assert_almost_equal(result, expected)

        start = obj.index[0].to_timestamp(how="start")
        end = (obj.index[-1] + obj.index.freq).to_timestamp(how="start")
        new_index = date_range(start=start, end=end, freq=freq, inclusive="left")
        expected = obj.to_timestamp().reindex(new_index).to_period(freq)

        result = obj.resample(freq).asfreq()
        tm.assert_almost_equal(result, expected)

        result = obj.resample(freq).asfreq().to_timestamp().to_period()
        tm.assert_almost_equal(result, expected)

    def test_asfreq_fill_value(self):
        # [FUNCTION DOC]
        # Regression test (issue 3715): frequency conversion with a fill value of 4.0 for a series and
        # 3.0 for a frame, both going through timestamps to hourly, equals reindexing with the fill
        # value.
        # test for fill value during resampling, issue 3715

        index = period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
        s = Series(range(len(index)), index=index)
        new_index = date_range(
            s.index[0].to_timestamp(how="start"),
            (s.index[-1]).to_timestamp(how="start"),
            freq="1h",
        )
        expected = s.to_timestamp().reindex(new_index, fill_value=4.0)
        result = s.to_timestamp().resample("1h").asfreq(fill_value=4.0)
        tm.assert_series_equal(result, expected)

        frame = s.to_frame("value")
        new_index = date_range(
            frame.index[0].to_timestamp(how="start"),
            (frame.index[-1]).to_timestamp(how="start"),
            freq="1h",
        )
        expected = frame.to_timestamp().reindex(new_index, fill_value=3.0)
        result = frame.to_timestamp().resample("1h").asfreq(fill_value=3.0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("freq", ["h", "12h", "2D", "W"])
    @pytest.mark.parametrize("kwargs", [{"on": "date"}, {"level": "d"}])
    def test_selection(self, freq, kwargs):
        # [FUNCTION DOC]
        # Regression test (GH 14008), parametrized by frequency and on/level keyword. Resampling a
        # period-indexed frame by a column or index level raises a not-implemented error saying the
        # selection with a period index is unsupported and to set the index. The source comment
        # calls this a bug since it should be implemented.
        # This is a bug, these should be implemented
        # GH 14008
        index = period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
        rng = np.arange(len(index), dtype=np.int64)
        df = DataFrame(
            {"date": index, "a": rng},
            index=pd.MultiIndex.from_arrays([rng, index], names=["v", "d"]),
        )
        msg = (
            "Resampling from level= or on= selection with a PeriodIndex is "
            r"not currently supported, use \.set_index\(\.\.\.\) to "
            "explicitly set index"
        )
        with pytest.raises(NotImplementedError, match=msg):
            df.resample(freq, **kwargs)

    @pytest.mark.parametrize("month", MONTHS)
    @pytest.mark.parametrize("meth", ["ffill", "bfill"])
    @pytest.mark.parametrize("conv", ["start", "end"])
    @pytest.mark.parametrize(
        ("offset", "period"), [("D", "D"), ("B", "B"), ("ME", "M"), ("QE", "Q")]
    )
    def test_annual_upsample_cases(
        self, offset, period, conv, meth, month, simple_period_range_series
    ):
        # [FUNCTION DOC]
        # Parametrized over twelve annual end months, ffill or bfill, start or end convention and
        # target (day, business day, month, quarter). Upsampling an annual period series by
        # convention then filling equals converting to timestamps and back with the same fill. For
        # business days a deprecation warning is asserted.
        ts = simple_period_range_series("1/1/1990", "12/31/1990", freq=f"Y-{month}")
        warn = FutureWarning if period == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = getattr(ts.resample(period, convention=conv), meth)()
            expected = result.to_timestamp(period, how=conv)
            expected = expected.asfreq(offset, meth).to_period()
        tm.assert_series_equal(result, expected)

    def test_basic_downsample(self, simple_period_range_series):
        # [FUNCTION DOC]
        # Verifies yearly downsample of a monthly series equals grouping by calendar year (with
        # year-end index), and that the alias without month gives the same result.
        ts = simple_period_range_series("1/1/1990", "6/30/1995", freq="M")
        result = ts.resample("Y-DEC").mean()

        expected = ts.groupby(ts.index.year).mean()
        expected.index = period_range("1/1/1990", "6/30/1995", freq="Y-DEC")
        tm.assert_series_equal(result, expected)

        # this is ok
        tm.assert_series_equal(ts.resample("Y-DEC").mean(), result)
        tm.assert_series_equal(ts.resample("Y").mean(), result)

    @pytest.mark.parametrize(
        "rule,expected_error_msg",
        [
            ("Y-DEC", "<YearEnd: month=12>"),
            ("Q-MAR", "<QuarterEnd: startingMonth=3>"),
            ("M", "<MonthEnd>"),
            ("W-THU", "<Week: weekday=3>"),
        ],
    )
    def test_not_subperiod(self, simple_period_range_series, rule, expected_error_msg):
        # [FUNCTION DOC]
        # Parametrized by target rule and the text naming that rule in the error. Resampling weekly
        # (Wednesday) periods to yearly, quarterly, monthly or another weekday rule raises an
        # incompatible frequency error saying the frequencies are not sub or super periods.
        # These are incompatible period rules for resampling
        ts = simple_period_range_series("1/1/1990", "6/30/1995", freq="W-WED")
        msg = (
            "Frequency <Week: weekday=2> cannot be resampled to "
            f"{expected_error_msg}, as they are not sub or super periods"
        )
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts.resample(rule).mean()

    @pytest.mark.parametrize("freq", ["D", "2D"])
    def test_basic_upsample(self, freq, simple_period_range_series):
        # [FUNCTION DOC]
        # Parametrized by daily and two-daily frequency. A yearly mean upsampled with end convention and
        # forward fill equals converting through timestamps and reindexing.
        ts = simple_period_range_series("1/1/1990", "6/30/1995", freq="M")
        result = ts.resample("Y-DEC").mean()

        resampled = result.resample(freq, convention="end").ffill()
        expected = result.to_timestamp(freq, how="end")
        expected = expected.asfreq(freq, "ffill").to_period(freq)
        tm.assert_series_equal(resampled, expected)

    def test_upsample_with_limit(self):
        # [FUNCTION DOC]
        # Verifies forward fill with a limit of two when upsampling yearly periods to months with the
        # end convention equals reindexing with the same limit.
        rng = period_range("1/1/2000", periods=5, freq="Y")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

        result = ts.resample("M", convention="end").ffill(limit=2)
        expected = ts.asfreq("M").reindex(result.index, method="ffill", limit=2)
        tm.assert_series_equal(result, expected)

    def test_annual_upsample(self, simple_period_range_series):
        # [FUNCTION DOC]
        # Verifies upsampling a one-column frame of yearly periods to daily with forward fill gives the
        # same column as upsampling the series.
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq="Y-DEC")
        df = DataFrame({"a": ts})
        rdf = df.resample("D").ffill()
        exp = df["a"].resample("D").ffill()
        tm.assert_series_equal(rdf["a"], exp)

    def test_annual_upsample2(self):
        # [FUNCTION DOC]
        # Verifies yearly to monthly forward fill upsample equals the start-aligned frequency
        # conversion reindexed with forward fill over 2000-01 to 2003-12.
        rng = period_range("2000", "2003", freq="Y-DEC")
        ts = Series([1, 2, 3, 4], index=rng)

        result = ts.resample("M").ffill()
        ex_index = period_range("2000-01", "2003-12", freq="M")

        expected = ts.asfreq("M", how="start").reindex(ex_index, method="ffill")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("month", MONTHS)
    @pytest.mark.parametrize("convention", ["start", "end"])
    @pytest.mark.parametrize(
        ("offset", "period"), [("D", "D"), ("B", "B"), ("ME", "M")]
    )
    def test_quarterly_upsample(
        self, month, offset, period, convention, simple_period_range_series
    ):
        # [FUNCTION DOC]
        # Parametrized by month, convention and target (day, business day, month). Upsampling
        # quarterly series (anchored at the month) with forward fill equals converting through
        # timestamps with the same convention; a deprecation warning is asserted for business days.
        freq = f"Q-{month}"
        ts = simple_period_range_series("1/1/1990", "12/31/1991", freq=freq)
        warn = FutureWarning if period == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = ts.resample(period, convention=convention).ffill()
            expected = result.to_timestamp(period, how=convention)
            expected = expected.asfreq(offset, "ffill").to_period()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("target", ["D", "B"])
    @pytest.mark.parametrize("convention", ["start", "end"])
    def test_monthly_upsample(self, target, convention, simple_period_range_series):
        # [FUNCTION DOC]
        # Parametrized by target (day or business day) and convention. Upsampling a monthly series with
        # forward fill equals the timestamp-based equivalent; business day emits a deprecation
        # warning that is asserted.
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq="M")

        warn = None if target == "D" else FutureWarning
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = ts.resample(target, convention=convention).ffill()
            expected = result.to_timestamp(target, how=convention)
            expected = expected.asfreq(target, "ffill").to_period()
        tm.assert_series_equal(result, expected)

    def test_resample_basic(self):
        # [FUNCTION DOC]
        # Regression test (GH3609): a second-frequency series with missing values resampled to minute
        # means gives 34.5 and 79.5 on a minute period index, whether computed via periods or via
        # datetime then converted to periods.
        # GH3609
        s = Series(
            range(100),
            index=date_range("20130101", freq="s", periods=100, name="idx"),
            dtype="float",
        )
        s[10:30] = np.nan
        index = PeriodIndex(
            [Period("2013-01-01 00:00", "min"), Period("2013-01-01 00:01", "min")],
            name="idx",
        )
        expected = Series([34.5, 79.5], index=index)
        result = s.to_period().resample("min").mean()
        tm.assert_series_equal(result, expected)
        result2 = s.resample("min").mean().to_period()
        tm.assert_series_equal(result2, expected)

    @pytest.mark.parametrize(
        "freq,expected_vals", [("M", [31, 29, 31, 9]), ("2M", [31 + 29, 31 + 9])]
    )
    def test_resample_count(self, freq, expected_vals):
        # [FUNCTION DOC]
        # Regression test (GH12774), parametrized by frequency and expected counts: a daily
        # period series of ones over 100 days counted per month gives 31, 29, 31, 9 and per two
        # months gives 60 and 40.
        # GH12774
        series = Series(1, index=period_range(start="2000", periods=100))
        result = series.resample(freq).count()
        expected_index = period_range(
            start="2000", freq=freq, periods=len(expected_vals)
        )
        expected = Series(expected_vals, index=expected_index)
        tm.assert_series_equal(result, expected)

    def test_resample_same_freq(self, resample_method):
        # [FUNCTION DOC]
        # Regression test (GH12770), parametrized by resample method fixture. Resampling a monthly
        # period series to the same frequency with any method returns the series unchanged.
        # GH12770
        series = Series(range(3), index=period_range(start="2000", periods=3, freq="M"))
        expected = series

        result = getattr(series.resample("M"), resample_method)()
        tm.assert_series_equal(result, expected)

    def test_resample_incompat_freq(self):
        # [FUNCTION DOC]
        # Verifies that resampling a monthly period series to weekly and computing the mean raises an
        # incompatible frequency error stating that the frequencies are not sub or super periods. A
        # to-do in the source asks whether the error should be raised at the resample call instead.
        msg = (
            "Frequency <MonthEnd> cannot be resampled to <Week: weekday=6>, "
            "as they are not sub or super periods"
        )
        pi = period_range(start="2000", periods=3, freq="M")
        ser = Series(range(3), index=pi)
        rs = ser.resample("W")
        with pytest.raises(IncompatibleFrequency, match=msg):
            # TODO: should this raise at the resample call instead of at the mean call?
            rs.mean()

    @pytest.mark.parametrize(
        "tz",
        [
            zoneinfo.ZoneInfo("America/Los_Angeles"),
            dateutil.tz.gettz("America/Los_Angeles"),
        ],
    )
    def test_with_local_timezone(self, tz):
        # [FUNCTION DOC]
        # Regression test (gh-5430), parametrized by time zone implementation (standard library zone
        # info or dateutil) for Los Angeles. Hourly series of ones over one UTC day is converted to
        # local time, resampled daily and converted to periods (a user warning about dropping the zone is
        # asserted). The expected period index is the UTC dates moved back one day.
        # see gh-5430
        local_timezone = tz

        start = datetime(
            year=2013, month=11, day=1, hour=0, minute=0, tzinfo=timezone.utc
        )
        # 1 day later
        end = datetime(
            year=2013, month=11, day=2, hour=0, minute=0, tzinfo=timezone.utc
        )

        index = date_range(start, end, freq="h", name="idx")

        series = Series(1, index=index)
        series = series.tz_convert(local_timezone)
        msg = "Converting to PeriodArray/Index representation will drop timezone"
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = series.resample("D").mean().to_period()

        # Create the expected series
        # Index is moved back a day with the timezone conversion from UTC to
        # Pacific
        expected_index = (
            period_range(start=start, end=end, freq="D", name="idx") - offsets.Day()
        )
        expected = Series(1.0, index=expected_index)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "tz",
        [
            zoneinfo.ZoneInfo("America/Los_Angeles"),
            dateutil.tz.gettz("America/Los_Angeles"),
        ],
    )
    def test_resample_with_tz(self, tz, unit):
        # [FUNCTION DOC]
        # Regression test (GH 13238), parametrized by time zone implementation and time unit fixture.
        # Daily mean of a 48-hour constant series in a local zone gives two days at the zone's local
        # midnights with value 2.0, with the given unit.
        # GH 13238
        dti = date_range("2017-01-01", periods=48, freq="h", tz=tz, unit=unit)
        ser = Series(2, index=dti)
        result = ser.resample("D").mean()
        exp_dti = pd.DatetimeIndex(
            ["2017-01-01", "2017-01-02"], tz=tz, freq="D"
        ).as_unit(unit)
        expected = Series(
            2.0,
            index=exp_dti,
        )
        tm.assert_series_equal(result, expected)

    def test_resample_nonexistent_time_bin_edge(self):
        # [FUNCTION DOC]
        # Regression test (GH 19375): resampling a series localized to US/Pacific across the daylight
        # saving gap with 900-second bins equals the original (zeros) with a 900-second frequency.
        # GH 19375
        index = date_range("2017-03-12", "2017-03-12 1:45:00", freq="15min")
        s = Series(np.zeros(len(index)), index=index)
        expected = s.tz_localize("US/Pacific")
        expected.index = pd.DatetimeIndex(expected.index, freq="900s")
        result = expected.resample("900s").mean()
        tm.assert_series_equal(result, expected)

    def test_resample_nonexistent_time_bin_edge2(self):
        # [FUNCTION DOC]
        # Regression test (GH 23742): grouping an hourly frame in the Sao Paulo zone by a daily grouper
        # and counting gives an index equal to the daily range with nonexistent midnights shifted
        # forward (left-inclusive).
        # GH 23742
        index = date_range(start="2017-10-10", end="2017-10-20", freq="1h")
        index = index.tz_localize("UTC").tz_convert("America/Sao_Paulo")
        df = DataFrame(data=list(range(len(index))), index=index)
        result = df.groupby(pd.Grouper(freq="1D")).count()
        expected = date_range(
            start="2017-10-09",
            end="2017-10-20",
            freq="D",
            tz="America/Sao_Paulo",
            nonexistent="shift_forward",
            inclusive="left",
        )
        tm.assert_index_equal(result.index, expected)

    def test_resample_ambiguous_time_bin_edge(self):
        # [FUNCTION DOC]
        # Regression test (GH 10117): resampling to thirty-minute means a series spanning an
        # ambiguous local time in London leaves it unchanged.
        # GH 10117
        idx = date_range(
            "2014-10-25 22:00:00",
            "2014-10-26 00:30:00",
            freq="30min",
            tz="Europe/London",
        )
        expected = Series(np.zeros(len(idx)), index=idx)
        result = expected.resample("30min").mean()
        tm.assert_series_equal(result, expected)

    def test_fill_method_and_how_upsample(self):
        # [FUNCTION DOC]
        # Regression test (GH2073): forward fill to month end of a quarterly series equals the result
        # of taking the last value of that forward-filled result at the same frequency.
        # GH2073
        s = Series(
            np.arange(9, dtype="int64"),
            index=date_range("2010-01-01", periods=9, freq="QE"),
        )
        last = s.resample("ME").ffill()
        both = s.resample("ME").ffill().resample("ME").last().astype("int64")
        tm.assert_series_equal(last, both)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("target", ["D", "B"])
    @pytest.mark.parametrize("convention", ["start", "end"])
    def test_weekly_upsample(self, day, target, convention, simple_period_range_series):
        # [FUNCTION DOC]
        # Parametrized over weekday anchors, target (day, business day) and convention. Weekly period
        # series upsampled with forward fill equals the timestamp-based equivalent; business day
        # emits a deprecation warning that is asserted.
        freq = f"W-{day}"
        ts = simple_period_range_series("1/1/1990", "07/31/1990", freq=freq)
        warn = None if target == "D" else FutureWarning
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = ts.resample(target, convention=convention).ffill()
            expected = result.to_timestamp(target, how=convention)
            expected = expected.asfreq(target, "ffill").to_period()
        tm.assert_series_equal(result, expected)

    def test_resample_to_timestamps(self, simple_period_range_series):
        # [FUNCTION DOC]
        # Verifies yearly mean converted to timestamps with the default equals converting with the
        # start option explicitly.
        ts = simple_period_range_series("1/1/1990", "12/31/1995", freq="M")

        result = ts.resample("Y-DEC").mean().to_timestamp()
        expected = ts.resample("Y-DEC").mean().to_timestamp(how="start")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("month", MONTHS)
    def test_resample_to_quarterly(self, simple_period_range_series, month):
        # [FUNCTION DOC]
        # Parametrized over twelve months. Resampling a yearly series to quarterly with forward fill
        # equals reindexing the start-timestamps series onto the quarter start dates, using the
        # quarter period index.
        ts = simple_period_range_series("1990", "1992", freq=f"Y-{month}")
        quar_ts = ts.resample(f"Q-{month}").ffill()

        stamps = ts.to_timestamp("D", how="start")
        qdates = period_range(
            ts.index[0].asfreq("D", "start"),
            ts.index[-1].asfreq("D", "end"),
            freq=f"Q-{month}",
        )

        expected = stamps.reindex(qdates.to_timestamp("D", "s"), method="ffill")
        expected.index = qdates

        tm.assert_series_equal(quar_ts, expected)

    @pytest.mark.parametrize("how", ["start", "end"])
    def test_resample_to_quarterly_start_end(self, simple_period_range_series, how):
        # [FUNCTION DOC]
        # Parametrized by start or end convention. Resampling a yearly (June) series to quarterly
        # (March) with forward fill equals converting the frequency with the same convention and
        # forward-fill reindexing on the result index.
        # KNOWN ISSUE: a comment block marked as a fixme about leaving code commented out remains in the test body (an unused timestamp-based comparison), i.e. leftover dead code.
        # conforms, but different month
        ts = simple_period_range_series("1990", "1992", freq="Y-JUN")
        result = ts.resample("Q-MAR", convention=how).ffill()
        expected = ts.asfreq("Q-MAR", how=how)
        expected = expected.reindex(result.index, method="ffill")

        # FIXME: don't leave commented-out
        # .to_timestamp('D')
        # expected = expected.resample('Q-MAR').ffill()

        tm.assert_series_equal(result, expected)

    def test_resample_fill_missing(self):
        # [FUNCTION DOC]
        # Verifies that with a period index with gaps (2000, 2005, 2007, 2009) the yearly forward fill
        # equals the timestamp resample result converted back to periods.
        rng = PeriodIndex([2000, 2005, 2007, 2009], freq="Y")

        s = Series(np.random.default_rng(2).standard_normal(4), index=rng)

        stamps = s.to_timestamp()
        filled = s.resample("Y").ffill()
        expected = stamps.resample("YE").ffill().to_period("Y")
        tm.assert_series_equal(filled, expected)

    def test_cant_fill_missing_dups(self):
        # [FUNCTION DOC]
        # Verifies forward filling with duplicated periods in the index raises an invalid index
        # error saying reindexing is only valid with uniquely valued indexes.
        rng = PeriodIndex([2000, 2005, 2005, 2007, 2007], freq="Y")
        s = Series(np.random.default_rng(2).standard_normal(5), index=rng)
        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            s.resample("Y").ffill()

    def test_resample_5minute(self):
        # [FUNCTION DOC]
        # Verifies minute-period series resampled to five minutes: mean, then to timestamps equals the
        # timestamp-based mean; and the period result equals that converted to five-minute periods,
        # also through a round trip.
        rng = period_range("1/1/2000", "1/5/2000", freq="min")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        expected = ts.to_timestamp().resample("5min").mean()
        result = ts.resample("5min").mean().to_timestamp()
        tm.assert_series_equal(result, expected)

        expected = expected.to_period("5min")
        result = ts.resample("5min").mean()
        tm.assert_series_equal(result, expected)
        result = ts.resample("5min").mean().to_timestamp().to_period()
        tm.assert_series_equal(result, expected)

    def test_upsample_daily_business_daily(self, simple_period_range_series):
        # [FUNCTION DOC]
        # Verifies upsampling a business-day period series to daily (frequency conversion) equals
        # the converted and reindexed series, and an hourly upsample with start convention equals
        # the equivalent reindexed on the hourly period range.
        ts = simple_period_range_series("1/1/2000", "2/1/2000", freq="B")

        result = ts.resample("D").asfreq()
        expected = ts.asfreq("D").reindex(period_range("1/3/2000", "2/1/2000"))
        tm.assert_series_equal(result, expected)

        ts = simple_period_range_series("1/1/2000", "2/1/2000")
        result = ts.resample("h", convention="s").asfreq()
        exp_rng = period_range("1/1/2000", "2/1/2000 23:00", freq="h")
        expected = ts.asfreq("h", how="s").reindex(exp_rng)
        tm.assert_series_equal(result, expected)

    def test_resample_irregular_sparse(self):
        # [FUNCTION DOC]
        # Verifies resampling an irregular sparse series: the 10-minute count (length) of a subset
        # equals the corresponding part of the full-series result.
        dr = date_range(start="1/1/2012", freq="5min", periods=1000)
        s = Series(np.array(100), index=dr)
        # subset the data.
        subset = s[:"2012-01-04 06:55"]

        result = subset.resample("10min").apply(len)
        expected = s.resample("10min").apply(len).loc[result.index]
        tm.assert_series_equal(result, expected)

    def test_resample_weekly_all_na(self):
        # [FUNCTION DOC]
        # Verifies weekly Thursday frequency conversion of a Wednesday-anchored series gives all
        # missing values, and the forward-filled version (without the last row) equals the converted
        # frequency forward-filled.
        rng = date_range("1/1/2000", periods=10, freq="W-WED")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        result = ts.resample("W-THU").asfreq()

        assert result.isna().all()

        result = ts.resample("W-THU").asfreq().ffill()[:-1]
        expected = ts.asfreq("W-THU").ffill()
        tm.assert_series_equal(result, expected)

    def test_resample_tz_localized(self, unit):
        # [FUNCTION DOC]
        # Parametrized by time unit fixture. Weekly mean of a Los Angeles localized series equals the
        # naive resample result localized back and given a weekly frequency; a daily mean works.
        dr = date_range(start="2012-4-13", end="2012-5-1", unit=unit)
        ts = Series(range(len(dr)), index=dr)

        ts_utc = ts.tz_localize("UTC")
        ts_local = ts_utc.tz_convert("America/Los_Angeles")

        result = ts_local.resample("W").mean()

        ts_local_naive = ts_local.copy()
        ts_local_naive.index = ts_local_naive.index.tz_localize(None)

        exp = ts_local_naive.resample("W").mean().tz_localize("America/Los_Angeles")
        exp.index = pd.DatetimeIndex(exp.index, freq="W")

        tm.assert_series_equal(result, exp)

        # it works
        result = ts_local.resample("D").mean()

    def test_resample_tz_localized2(self):
        # [FUNCTION DOC]
        # Regression test (#2245) in Sydney zone: daily right-closed right-labeled mean of two
        # minutes gives a missing then 1.5 on a daily index (behavior changed by GH#61985 to be
        # like calendar days); and daily mean converted to periods (a user warning is asserted) gives
        # 1.5 on a daily period index.
        # #2245
        idx = date_range(
            "2001-09-20 15:59", "2001-09-20 16:00", freq="min", tz="Australia/Sydney"
        )
        s = Series([1, 2], index=idx)

        # GH#61985 changed this to behave like "B" rather than "24h"
        result = s.resample("D", closed="right", label="right").mean()
        ex_index = date_range("2001-09-20", periods=2, freq="D", tz="Australia/Sydney")
        expected = Series([np.nan, 1.5], index=ex_index)

        tm.assert_series_equal(result, expected)

        # for good measure
        msg = "Converting to PeriodArray/Index representation will drop timezone "
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = s.resample("D").mean().to_period()
        ex_index = period_range("2001-09-20", periods=1, freq="D")
        expected = Series([1.5], index=ex_index)
        tm.assert_series_equal(result, expected)

    def test_resample_tz_localized3(self):
        # [FUNCTION DOC]
        # Regression test (GH 6397): hourly frame localized to EST with 20000 rows; aggregating yearly
        # with a dictionary of sum and mean equals computing the sums and means separately, ordering
        # columns first then second.
        # GH 6397
        # comparing an offset that doesn't propagate tz's
        rng = date_range("1/1/2011", periods=20000, freq="h")
        rng = rng.tz_localize("EST")
        ts = DataFrame(index=rng)
        ts["first"] = np.random.default_rng(2).standard_normal(len(rng))
        ts["second"] = np.cumsum(np.random.default_rng(2).standard_normal(len(rng)))
        expected = DataFrame(
            {
                "first": ts.resample("YE").sum()["first"],
                "second": ts.resample("YE").mean()["second"],
            },
            columns=["first", "second"],
        )
        result = (
            ts.resample("YE")
            .agg({"first": "sum", "second": "mean"})
            .reindex(columns=["first", "second"])
        )
        tm.assert_frame_equal(result, expected)

    def test_closed_left_corner(self):
        # [FUNCTION DOC]
        # Regression test (#1465): ten-minute left-closed means with right or left labels on a series
        # whose first value is missing equal the result of dropping the first point; left labels give
        # three expected bins starting at 9:30.
        # #1465
        s = Series(
            np.random.default_rng(2).standard_normal(21),
            index=date_range(start="1/1/2012 9:30", freq="1min", periods=21),
        )
        s.iloc[0] = np.nan

        result = s.resample("10min", closed="left", label="right").mean()
        exp = s[1:].resample("10min", closed="left", label="right").mean()
        tm.assert_series_equal(result, exp)

        result = s.resample("10min", closed="left", label="left").mean()
        exp = s[1:].resample("10min", closed="left", label="left").mean()

        ex_index = date_range(start="1/1/2012 9:30", freq="10min", periods=3)

        tm.assert_index_equal(result.index, ex_index)
        tm.assert_series_equal(result, exp)

    def test_quarterly_resampling(self):
        # [FUNCTION DOC]
        # Verifies yearly mean of a quarterly period series equals the timestamp-based yearly mean
        # converted to periods.
        rng = period_range("2000Q1", periods=10, freq="Q-DEC")
        ts = Series(np.arange(10), index=rng)

        result = ts.resample("Y").mean()
        exp = ts.to_timestamp().resample("YE").mean().to_period()
        tm.assert_series_equal(result, exp)

    def test_resample_weekly_bug_1726(self):
        # [FUNCTION DOC]
        # Regression test (bug 1726): weekly (Monday) left-closed left-labeled first on a daily
        # open/high/low/close/volume frame starting on a Monday must work without error.
        # 8/6/12 is a Monday
        ind = date_range(start="8/6/2012", end="8/26/2012", freq="D")
        n = len(ind)
        data = [[x] * 5 for x in range(n)]
        df = DataFrame(data, columns=["open", "high", "low", "close", "vol"], index=ind)

        # it works!
        df.resample("W-MON", closed="left", label="left").first()

    def test_resample_with_dst_time_change(self):
        # [FUNCTION DOC]
        # Regression test (GH 15549): resampling a Chicago-localized frame across the daylight saving
        # change to twelve-hour bins, right-closed and right-labeled, with last then forward fill
        # gives thirteen bins (ending with 2.0) whose offsets switch from minus six to minus five
        # hours and a twelve-hour frequency.
        # GH 15549
        index = (
            pd.DatetimeIndex([1457537600000000000, 1458059600000000000])
            .tz_localize("UTC")
            .tz_convert("America/Chicago")
        )
        df = DataFrame([1, 2], index=index)
        result = df.resample("12h", closed="right", label="right").last().ffill()

        expected_index_values = [
            "2016-03-09 12:00:00-06:00",
            "2016-03-10 00:00:00-06:00",
            "2016-03-10 12:00:00-06:00",
            "2016-03-11 00:00:00-06:00",
            "2016-03-11 12:00:00-06:00",
            "2016-03-12 00:00:00-06:00",
            "2016-03-12 12:00:00-06:00",
            "2016-03-13 00:00:00-06:00",
            "2016-03-13 13:00:00-05:00",
            "2016-03-14 01:00:00-05:00",
            "2016-03-14 13:00:00-05:00",
            "2016-03-15 01:00:00-05:00",
            "2016-03-15 13:00:00-05:00",
        ]
        index = (
            pd.to_datetime(expected_index_values, utc=True)
            .tz_convert("America/Chicago")
            .as_unit(index.unit)
        )
        index = pd.DatetimeIndex(index, freq="12h")
        expected = DataFrame(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            index=index,
        )
        tm.assert_frame_equal(result, expected)

    def test_resample_bms_2752(self):
        # [FUNCTION DOC]
        # Regression test (GH2753): business month start mean of a business-day series of missing
        # values starts at 2000-01-03 and after resampling to business days the first label is
        # unchanged.
        # GH2753
        timeseries = Series(
            index=pd.bdate_range("20000101", "20000201"), dtype=np.float64
        )
        res1 = timeseries.resample("BMS").mean()
        res2 = timeseries.resample("BMS").mean().resample("B").mean()
        assert res1.index[0] == Timestamp("20000103")
        assert res1.index[0] == res2.index[0]

    @pytest.mark.xfail(reason="Commented out for more than 3 years. Should this work?")
    def test_monthly_convention_span(self):
        # [FUNCTION DOC]
        # Expected-failure test (commented out for over three years; unclear if it should work). A
        # monthly period series resampled to daily mean should equal the end-aligned conversion
        # back-filled over the daily range. Uses a deprecated fill-method form.
        rng = period_range("2000-01", periods=3, freq="ME")
        ts = Series(np.arange(3), index=rng)

        # hacky way to get same thing
        exp_index = period_range("2000-01-01", "2000-03-31", freq="D")
        expected = ts.asfreq("D", how="end").reindex(exp_index)
        expected = expected.fillna(method="bfill")

        result = ts.resample("D").mean()

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "from_freq, to_freq", [("D", "ME"), ("QE", "YE"), ("ME", "QE"), ("D", "W")]
    )
    def test_default_right_closed_label(self, from_freq, to_freq):
        # [FUNCTION DOC]
        # Parametrized by source and target frequency pairs (day to month end, quarter to year, month to
        # quarter, day to week). Verifies that the default closed side and label side for these
        # end-anchored targets are right: results equal an explicit right closed, right labeled
        # resample.
        idx = date_range(start="8/15/2012", periods=100, freq=from_freq)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)

        resampled = df.resample(to_freq).mean()
        tm.assert_frame_equal(
            resampled, df.resample(to_freq, closed="right", label="right").mean()
        )

    @pytest.mark.parametrize(
        "from_freq, to_freq",
        [("D", "MS"), ("QE", "YS"), ("ME", "QS"), ("h", "D"), ("min", "h")],
    )
    def test_default_left_closed_label(self, from_freq, to_freq):
        # [FUNCTION DOC]
        # Parametrized by source and target pairs for start-anchored or small targets (month start,
        # year start, quarter start, hour to day, minute to hour). Verifies the default closed and
        # label sides are left.
        idx = date_range(start="8/15/2012", periods=100, freq=from_freq)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), idx)

        resampled = df.resample(to_freq).mean()
        tm.assert_frame_equal(
            resampled, df.resample(to_freq, closed="left", label="left").mean()
        )

    def test_all_values_single_bin(self):
        # [FUNCTION DOC]
        # Regression test (GH#2070): a yearly mean of a monthly period series with twelve months has
        # its single value equal to the overall mean.
        # GH#2070
        index = period_range(start="2012-01-01", end="2012-12-31", freq="M")
        ser = Series(np.random.default_rng(2).standard_normal(len(index)), index=index)

        result = ser.resample("Y").mean()
        tm.assert_almost_equal(result.iloc[0], ser.mean())

    def test_evenly_divisible_with_no_extra_bins(self):
        # [FUNCTION DOC]
        # Regression test (GH#4076): a nine-row frame resampled to five-day means has exactly two bins
        # (no extra bin when the frequency evenly divides), at 2000-01-01 and 2000-01-06.
        # GH#4076
        # when the frequency is evenly divisible, sometimes extra bins

        df = DataFrame(
            np.random.default_rng(2).standard_normal((9, 3)),
            index=date_range("2000-1-1", periods=9, unit="ns"),
        )
        result = df.resample("5D").mean()
        expected = pd.concat([df.iloc[0:5].mean(), df.iloc[5:].mean()], axis=1).T
        expected.index = pd.DatetimeIndex(
            [Timestamp("2000-1-1"), Timestamp("2000-1-6")], dtype="M8[ns]", freq="5D"
        )
        tm.assert_frame_equal(result, expected)

    def test_evenly_divisible_with_no_extra_bins2(self):
        # [FUNCTION DOC]
        # Verifies seven-day count and sum for a frame of two groups of 28 daily rows with duplicated
        # dates: counts of 14 per column per bin and the corresponding sums in four weekly bins.
        index = date_range(start="2001-5-4", periods=28)
        df = DataFrame(
            [
                {
                    "REST_KEY": 1,
                    "DLY_TRN_QT": 80,
                    "DLY_SLS_AMT": 90,
                    "COOP_DLY_TRN_QT": 30,
                    "COOP_DLY_SLS_AMT": 20,
                }
            ]
            * 28
            + [
                {
                    "REST_KEY": 2,
                    "DLY_TRN_QT": 70,
                    "DLY_SLS_AMT": 10,
                    "COOP_DLY_TRN_QT": 50,
                    "COOP_DLY_SLS_AMT": 20,
                }
            ]
            * 28,
            index=index.append(index),
        ).sort_index()

        index = date_range("2001-5-4", periods=4, freq="7D")
        expected = DataFrame(
            [
                {
                    "REST_KEY": 14,
                    "DLY_TRN_QT": 14,
                    "DLY_SLS_AMT": 14,
                    "COOP_DLY_TRN_QT": 14,
                    "COOP_DLY_SLS_AMT": 14,
                }
            ]
            * 4,
            index=index,
        )
        result = df.resample("7D").count()
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            [
                {
                    "REST_KEY": 21,
                    "DLY_TRN_QT": 1050,
                    "DLY_SLS_AMT": 700,
                    "COOP_DLY_TRN_QT": 560,
                    "COOP_DLY_SLS_AMT": 280,
                }
            ]
            * 4,
            index=index,
        )
        result = df.resample("7D").sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("freq, period_mult", [("h", 24), ("12h", 2)])
    def test_upsampling_ohlc(self, freq, period_mult):
        # [FUNCTION DOC]
        # Regression test (GH 13083), parametrized by hourly or twelve-hourly frequency with the
        # number of subperiods per day. Upsampling a daily period series with open-high-low-close
        # equals the timestamp-based result converted to periods and extended to all sub-periods of the
        # last day; also holds after a round trip through timestamps.
        # GH 13083
        pi = period_range(start="2000", freq="D", periods=10)
        s = Series(range(len(pi)), index=pi)
        expected = s.to_timestamp().resample(freq).ohlc().to_period(freq)

        # timestamp-based resampling doesn't include all sub-periods
        # of the last original period, so extend accordingly:
        new_index = period_range(start="2000", freq=freq, periods=period_mult * len(pi))
        expected = expected.reindex(new_index)
        result = s.resample(freq).ohlc()
        tm.assert_frame_equal(result, expected)

        result = s.resample(freq).ohlc().to_timestamp().to_period()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, values",
        [
            (
                [
                    pd.NaT,
                    "1970-01-01 00:00:00",
                    pd.NaT,
                    "1970-01-01 00:00:02",
                    "1970-01-01 00:00:03",
                ],
                [2, 3, 5, 7, 11],
            ),
            (
                [
                    pd.NaT,
                    pd.NaT,
                    "1970-01-01 00:00:00",
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    "1970-01-01 00:00:02",
                    "1970-01-01 00:00:03",
                    pd.NaT,
                    pd.NaT,
                ],
                [1, 2, 3, 5, 6, 8, 7, 11, 12, 13],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "freq, expected_values",
        [
            ("1s", [3, np.nan, 7, 11]),
            ("2s", [3, (7 + 11) / 2]),
            ("3s", [(3 + 7) / 2, 11]),
        ],
    )
    def test_resample_with_nat(self, periods, values, freq, expected_values):
        # [FUNCTION DOC]
        # Regression test (GH 13224), parametrized by period lists with not-a-time entries and values,
        # and by target frequency with expected means. A frame of values with missing times in the
        # index resampled to one, two and three seconds gives the expected means, ignoring entries
        # without a time.
        # GH 13224
        index = PeriodIndex(periods, freq="s")
        frame = DataFrame(values, index=index)

        expected_index = period_range(
            "1970-01-01 00:00:00", periods=len(expected_values), freq=freq
        )
        expected = DataFrame(expected_values, index=expected_index)
        result = frame.resample(freq).mean()
        tm.assert_frame_equal(result, expected)

    def test_resample_with_only_nat(self):
        # [FUNCTION DOC]
        # Regression test (GH 13224): a frame whose index has only not-a-time entries resampled to one
        # second gives an empty float frame with an empty period index of the original frequency.
        # GH 13224
        pi = PeriodIndex([pd.NaT] * 3, freq="s")
        frame = DataFrame([2, 3, 5], index=pi, columns=["a"])
        expected_index = PeriodIndex(data=[], freq=pi.freq)
        expected = DataFrame(index=expected_index, columns=["a"], dtype="float64")
        result = frame.resample("1s").mean()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "start,end,start_freq,end_freq,offset",
        [
            ("19910905", "19910909 03:00", "h", "24h", "10h"),
            ("19910905", "19910909 12:00", "h", "24h", "10h"),
            ("19910905", "19910909 23:00", "h", "24h", "10h"),
            ("19910905 10:00", "19910909", "h", "24h", "10h"),
            ("19910905 10:00", "19910909 10:00", "h", "24h", "10h"),
            ("19910905", "19910909 10:00", "h", "24h", "10h"),
            ("19910905 12:00", "19910909", "h", "24h", "10h"),
            ("19910905 12:00", "19910909 03:00", "h", "24h", "10h"),
            ("19910905 12:00", "19910909 12:00", "h", "24h", "10h"),
            ("19910905 12:00", "19910909 12:00", "h", "24h", "34h"),
            ("19910905 12:00", "19910909 12:00", "h", "17h", "10h"),
            ("19910905 12:00", "19910909 12:00", "h", "17h", "3h"),
            ("19910905", "19910913 06:00", "2h", "24h", "10h"),
            ("19910905", "19910905 01:39", "Min", "5Min", "3Min"),
            ("19910905", "19910905 03:18", "2Min", "5Min", "3Min"),
        ],
    )
    def test_resample_with_offset(self, start, end, start_freq, end_freq, offset):
        # [FUNCTION DOC]
        # Regression test (GH 23882, GH 31809), parametrized by start, end, source frequency, target
        # frequency and offset. Resampling a period series with an offset and mean then converting to
        # timestamps equals the timestamp-based resample with the same offset.
        # GH 23882 & 31809
        pi = period_range(start, end, freq=start_freq)
        ser = Series(np.arange(len(pi)), index=pi)
        result = ser.resample(end_freq, offset=offset).mean()
        result = result.to_timestamp(end_freq)

        expected = ser.to_timestamp().resample(end_freq, offset=offset).mean()
        tm.assert_series_equal(result, expected)

    def test_resample_with_offset_month(self):
        # [FUNCTION DOC]
        # Regression test (GH 23882, GH 31809): an hourly period series resampled to months and
        # converted to timestamps equals the timestamp-based month-end mean, except that the
        # expected index frequency is removed.
        # KNOWN ISSUE: the line clearing the frequency of the expected index carries an unresolved to-do asking whether being non-tick is the relevant characteristic, so the reason for the difference is undecided.
        # GH 23882 & 31809
        pi = period_range("19910905 12:00", "19910909 1:00", freq="h")
        ser = Series(np.arange(len(pi)), index=pi)
        result = ser.resample("M").mean()
        result = result.to_timestamp("M")
        expected = ser.to_timestamp().resample("ME").mean()
        # TODO: is non-tick the relevant characteristic? (GH 33815)
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "first,last,freq,freq_to_offset,exp_first,exp_last",
        [
            ("19910905", "19920406", "D", "D", "19910905", "19920406"),
            ("19910905 00:00", "19920406 06:00", "D", "D", "19910905", "19920406"),
            (
                "19910905 06:00",
                "19920406 06:00",
                "h",
                "h",
                "19910905 06:00",
                "19920406 06:00",
            ),
            ("19910906", "19920406", "M", "ME", "1991-09", "1992-04"),
            ("19910831", "19920430", "M", "ME", "1991-08", "1992-04"),
            ("1991-08", "1992-04", "M", "ME", "1991-08", "1992-04"),
        ],
    )
    def test_get_period_range_edges(
        self, first, last, freq, freq_to_offset, exp_first, exp_last
    ):
        # [FUNCTION DOC]
        # Parametrized by first and last periods, frequency, offset frequency, and expected first and
        # last. Tests the private helper that computes the period range edges: given first and last
        # periods and an offset, the returned pair of periods equals the expected pair at the period
        # frequency.
        first = Period(first)
        last = Period(last)

        exp_first = Period(exp_first, freq=freq)
        exp_last = Period(exp_last, freq=freq)

        freq = pd.tseries.frequencies.to_offset(freq_to_offset)
        result = _get_period_range_edges(first, last, freq)
        expected = (exp_first, exp_last)
        assert result == expected

    def test_sum_min_count(self):
        # [FUNCTION DOC]
        # Regression test (GH 19974): quarterly sum with minimum count one of six monthly values (the
        # last three missing) gives 3.0 for the first quarter and a missing value for the second.
        # GH 19974
        index = date_range(start="2018", freq="ME", periods=6)
        data = np.ones(6)
        data[3:6] = np.nan
        s = Series(data, index).to_period()
        result = s.resample("Q").sum(min_count=1)
        expected = Series(
            [3.0, np.nan], index=PeriodIndex(["2018Q1", "2018Q2"], freq="Q-DEC")
        )
        tm.assert_series_equal(result, expected)

    def test_resample_t_l_deprecated(self):
        # [FUNCTION DOC]
        # Regression test (GH#52536): an invalid millisecond-style frequency letter for a period range
        # raises a value error saying the frequency is invalid; resampling with the old minute letter
        # alias also raises an invalid frequency error.
        # GH#52536
        msg_t = "Invalid frequency: T"
        msg_l = "Invalid frequency: L"

        with pytest.raises(ValueError, match=msg_l):
            period_range(
                "2020-01-01 00:00:00 00:00", "2020-01-01 00:00:00 00:01", freq="L"
            )
        rng_l = period_range(
            "2020-01-01 00:00:00 00:00", "2020-01-01 00:00:00 00:01", freq="ms"
        )
        ser = Series(np.arange(len(rng_l)), index=rng_l)

        with pytest.raises(ValueError, match=msg_t):
            ser.resample("T").mean()

    @pytest.mark.parametrize(
        "freq, freq_depr, freq_depr_res",
        [
            ("2Q", "2q", "2y"),
            ("2M", "2m", "2q"),
        ],
    )
    def test_resample_lowercase_frequency_raises(self, freq, freq_depr, freq_depr_res):
        # [FUNCTION DOC]
        # Parametrized by quarter/month frequency, its lowercase variant and a lowercase target. Lowercase
        # frequency aliases raise value errors saying the frequency is invalid, both when creating a
        # period range and when resampling a valid period series.
        msg = f"Invalid frequency: {freq_depr}"
        with pytest.raises(ValueError, match=msg):
            period_range("2020-01-01", "2020-08-01", freq=freq_depr)

        msg = f"Invalid frequency: {freq_depr_res}"
        rng = period_range("2020-01-01", "2020-08-01", freq=freq)
        ser = Series(np.arange(len(rng)), index=rng)
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq_depr_res).mean()

    @pytest.mark.parametrize(
        "offset",
        [
            offsets.MonthBegin(),
            offsets.BYearBegin(2),
            offsets.BusinessHour(2),
        ],
    )
    def test_asfreq_invalid_period_offset(self, offset, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH#55785), parametrized by offsets (month begin, business year begin
        # twice, business hour twice) and the frame-or-series fixture. Converting frequency of a
        # period-indexed object to such an offset raises a value error saying it is not supported as a
        # period frequency.
        # GH#55785
        msg = re.escape(f"{offset} is not supported as period frequency")

        obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))
        with pytest.raises(ValueError, match=msg):
            obj.asfreq(freq=offset)


@pytest.mark.parametrize(
    "freq",
    [
        ("2ME"),
        ("2QE"),
        ("2QE-FEB"),
        ("2YE"),
        ("2YE-MAR"),
        ("2me"),
        ("2qe"),
        ("2ye-mar"),
    ],
)
def test_resample_frequency_ME_QE_YE_raises(frame_or_series, freq):
    # [FUNCTION DOC]
    # Regression test (GH#9586), parametrized by multiples of month, quarter and year end
    # frequencies in upper and lower case, and frame-or-series fixture. Resampling a
    # period-indexed object with those frequencies raises a value error saying the frequency is
    # invalid.
    # KNOWN ISSUE: the first message assignment (saying the frequency is not supported as period frequency) is immediately overwritten by the invalid frequency message and is dead code.
    # GH#9586
    msg = f"{freq[1:]} is not supported as period frequency"

    obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))
    msg = f"Invalid frequency: {freq}"
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq)


def test_corner_cases_period(simple_period_range_series):
    # [FUNCTION DOC]
    # Miscellaneous coverage: resampling an empty slice of a monthly period series to yearly
    # works and gives an empty result.
    # miscellaneous test coverage
    len0pts = simple_period_range_series("2007-01", "2010-05", freq="M")[:0]
    # it works
    result = len0pts.resample("Y-DEC").mean()
    assert len(result) == 0


@pytest.mark.parametrize("freq", ["2BME", "2CBME", "2SME", "2BQE-FEB", "2BYE-MAR"])
def test_resample_frequency_invalid_freq(frame_or_series, freq):
    # [FUNCTION DOC]
    # Regression test (GH#9586), parametrized by business, custom business, semi-month and
    # business-quarter/year end frequencies: resampling a period-indexed frame or series raises a
    # value error saying the frequency is invalid.
    # GH#9586
    msg = f"Invalid frequency: {freq}"

    obj = frame_or_series(range(5), index=period_range("2020-01-01", periods=5))
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq)
