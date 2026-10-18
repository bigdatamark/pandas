# [MODULE DOC]
# Tests for DatetimeIndex.to_period: explicit and inferred frequencies, frequency
# normalization of quarter/annual/month-ish offsets, errors for unsupported frequencies,
# timezone dropping warnings, sub-second periods and round trips.
from datetime import timezone

import dateutil.tz
from dateutil.tz import tzlocal
import pytest

from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToPeriod:
    # [CLASS DOC]
    # Groups to_period tests for DatetimeIndex; no state.
    def test_dti_to_period(self):
        # [FUNCTION DOC]
        # Verifies conversion of a month-end range with the default (monthly), daily and three-daily
        # frequencies: first and last periods and equality with the corresponding period ranges.
        # KNOWN ISSUE: the assertion on the last element of the three-day conversion uses a comma, so it
        # only checks truthiness of the element and treats the expected period as the failure message;
        # it never compares the values.
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")
        pi1 = dti.to_period()
        pi2 = dti.to_period(freq="D")
        pi3 = dti.to_period(freq="3D")

        assert pi1[0] == Period("Jan 2005", freq="M")
        assert pi2[0] == Period("1/31/2005", freq="D")
        assert pi3[0] == Period("1/31/2005", freq="3D")

        assert pi1[-1] == Period("Nov 2005", freq="M")
        assert pi2[-1] == Period("11/30/2005", freq="D")
        assert pi3[-1], Period("11/30/2005", freq="3D")

        tm.assert_index_equal(pi1, period_range("1/1/2005", "11/1/2005", freq="M"))
        tm.assert_index_equal(
            pi2, period_range("1/1/2005", "11/1/2005", freq="M").asfreq("D")
        )
        tm.assert_index_equal(
            pi3, period_range("1/1/2005", "11/1/2005", freq="M").asfreq("3D")
        )

    @pytest.mark.parametrize("month", MONTHS)
    def test_to_period_quarterly(self, month):
        # [FUNCTION DOC]
        # Parametrized over twelve month names. Verifies converting a quarterly period range anchored at
        # each month to timestamps and back with the same quarterly frequency returns the original.
        # make sure we can make the round trip
        freq = f"Q-{month}"
        rng = period_range("1989Q3", "1991Q3", freq=freq)
        stamps = rng.to_timestamp()
        result = stamps.to_period(freq)
        tm.assert_index_equal(rng, result)

    @pytest.mark.parametrize("off", ["BQE", "QS", "BQS"])
    def test_to_period_quarterlyish(self, off):
        # [FUNCTION DOC]
        # Parametrized over business-quarter-end, quarter-start and business-quarter-start frequencies.
        # Verifies the period frequency is quarterly ending in December.
        rng = date_range("01-Jan-2012", periods=8, freq=off)
        prng = rng.to_period()
        assert prng.freq == "QE-DEC"

    @pytest.mark.parametrize("off", ["BYE", "YS", "BYS"])
    def test_to_period_annualish(self, off):
        # [FUNCTION DOC]
        # Parametrized over business-year-end, year-start, business-year-start frequencies. Verifies the
        # period frequency is annual ending in December.
        rng = date_range("01-Jan-2012", periods=8, freq=off)
        prng = rng.to_period()
        assert prng.freq == "YE-DEC"

    def test_to_period_monthish(self):
        # [FUNCTION DOC]
        # Verifies month-start, business-month-end and month-end ranges convert to monthly period
        # frequency, and that the retired end-of-month alias raises a value error on range creation.
        offsets = ["MS", "BME"]
        for off in offsets:
            rng = date_range("01-Jan-2012", periods=8, freq=off)
            prng = rng.to_period()
            assert prng.freqstr == "M"

        rng = date_range("01-Jan-2012", periods=8, freq="ME")
        prng = rng.to_period()
        assert prng.freqstr == "M"

        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            date_range("01-Jan-2012", periods=8, freq="EOM")

    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("2ME", "2M"),
            (MonthEnd(2), MonthEnd(2)),
        ],
    )
    def test_dti_to_period_2monthish(self, freq_offset, freq_period):
        # [FUNCTION DOC]
        # Parametrized over a two-month-end string or offset with equivalent period frequency.
        # Verifies that converting a range gives the equal period range with two-month frequency.
        dti = date_range("2020-01-01", periods=3, freq=freq_offset)
        pi = dti.to_period()

        tm.assert_index_equal(pi, period_range("2020-01", "2020-05", freq=freq_period))

    @pytest.mark.parametrize(
        "freq", ["2ME", "1me", "2QE", "2QE-SEP", "1YE", "ye", "2YE-MAR"]
    )
    def test_to_period_frequency_M_Q_Y_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequency strings (multiples of month, quarter or year end, lowercase
        # variants). Verifies passing them to to_period raises a value error saying the frequency is
        # invalid.
        msg = f"Invalid frequency: {freq}"

        rng = date_range("01-Jan-2012", periods=8, freq="ME")
        with pytest.raises(ValueError, match=msg):
            rng.to_period(freq)

    def test_to_period_infer(self):
        # [FUNCTION DOC]
        # Regression test: converting a timezone-aware five-minute range with an explicit and with an
        # inferred frequency each emits a user warning about dropping timezone information, and the two
        # results are equal.
        # https://github.com/pandas-dev/pandas/issues/33358
        rng = date_range(
            start="2019-12-22 06:40:00+00:00",
            end="2019-12-22 08:45:00+00:00",
            freq="5min",
        )

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            pi1 = rng.to_period("5min")

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            pi2 = rng.to_period()

        tm.assert_index_equal(pi1, pi2)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_period_dt64_round_trip(self):
        # [FUNCTION DOC]
        # Ignores a deprecation warning about business-day period dtype. Verifies that converting a
        # business-day range to periods and back to timestamps gives the range in microsecond unit, also
        # with hourly period frequency.
        dti = date_range("1/1/2000", "1/7/2002", freq="B", unit="ns")
        pi = dti.to_period()
        tm.assert_index_equal(pi.to_timestamp(), dti.as_unit("us"))

        dti = date_range("1/1/2000", "1/7/2002", freq="B", unit="ns")
        pi = dti.to_period(freq="h")
        tm.assert_index_equal(pi.to_timestamp(), dti.as_unit("us"))

    def test_to_period_millisecond(self):
        # [FUNCTION DOC]
        # Verifies conversion of two UTC timestamps to millisecond-frequency periods (with a warning about
        # dropping timezone), truncating to milliseconds.
        index = DatetimeIndex(
            [
                Timestamp("2007-01-01 10:11:12.123456Z"),
                Timestamp("2007-01-01 10:11:13.789123Z"),
            ]
        )

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            period = index.to_period(freq="ms")
        assert 2 == len(period)
        assert period[0] == Period("2007-01-01 10:11:12.123Z", "ms")
        assert period[1] == Period("2007-01-01 10:11:13.789Z", "ms")

    def test_to_period_microsecond(self):
        # [FUNCTION DOC]
        # Same as the millisecond test with microsecond frequency.
        index = DatetimeIndex(
            [
                Timestamp("2007-01-01 10:11:12.123456Z"),
                Timestamp("2007-01-01 10:11:13.789123Z"),
            ]
        )

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            period = index.to_period(freq="us")
        assert 2 == len(period)
        assert period[0] == Period("2007-01-01 10:11:12.123456Z", "us")
        assert period[1] == Period("2007-01-01 10:11:13.789123Z", "us")

    @pytest.mark.parametrize(
        "tz",
        [
            "US/Eastern",
            timezone.utc,
            tzlocal(),
            "dateutil/US/Eastern",
            dateutil.tz.tzutc(),
        ],
    )
    def test_to_period_tz(self, tz):
        # [FUNCTION DOC]
        # Parametrized over five timezone specifications (named, UTC, local, dateutil named, dateutil UTC).
        # Verifies converting a timezone-aware daily range (with a warning about dropping the timezone)
        # gives the same first period as the scalar conversion, and the full conversion with an explicit
        # freq equals converting the naive range.
        ts = date_range("1/1/2000", "2/1/2000", tz=tz)

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            result = ts.to_period()[0]
            expected = ts[0].to_period(ts.freq)

        assert result == expected

        expected = date_range("1/1/2000", "2/1/2000").to_period()

        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            result = ts.to_period(ts.freq)

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", ["Etc/GMT-1", "Etc/GMT+1"])
    def test_to_period_tz_utc_offset_consistency(self, tz):
        # [FUNCTION DOC]
        # Parametrized over Etc/GMT-1 and Etc/GMT+1. Regression test: converting a range with a fixed
        # offset timezone to periods warns and the first period equals the scalar conversion.
        # KNOWN ISSUE: the tz parameter is never used; the range always uses the Etc/GMT-1 timezone, so
        # the second parametrized case duplicates the first.
        # GH#22905
        ts = date_range("1/1/2000", "2/1/2000", tz="Etc/GMT-1")
        with tm.assert_produces_warning(UserWarning, match="drop timezone info"):
            result = ts.to_period()[0]
            expected = ts[0].to_period(ts.freq)
            assert result == expected

    def test_to_period_nofreq(self):
        # [FUNCTION DOC]
        # Verifies converting an index with no freq raises a value error telling that a freq must be
        # passed; an index created with inferred freq converts to the daily PeriodIndex; and a regular
        # daily index without declared freq is also inferred when converting.
        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"])
        msg = "You must pass a freq argument as current index has none."
        with pytest.raises(ValueError, match=msg):
            idx.to_period()

        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"], freq="infer")
        assert idx.freqstr == "D"
        expected = PeriodIndex(["2000-01-01", "2000-01-02", "2000-01-03"], freq="D")
        tm.assert_index_equal(idx.to_period(), expected)

        # GH#7606
        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])
        assert idx.freqstr is None
        tm.assert_index_equal(idx.to_period(), expected)

    @pytest.mark.parametrize("freq", ["2BME", "SME-15", "2BMS"])
    def test_to_period_offsets_not_supported(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequencies not supported as period frequencies (business month end
        # multiples, semi-month, business month start multiples). Verifies to_period raises a value error
        # about an invalid frequency or about being unsupported.
        # GH#56243
        msg = "|".join(
            [
                f"Invalid frequency: {freq}",
                f"{freq} is not supported as period frequency",
            ]
        )

        ts = date_range("1/1/2012", periods=4, freq=freq)
        with pytest.raises(ValueError, match=msg):
            ts.to_period()
