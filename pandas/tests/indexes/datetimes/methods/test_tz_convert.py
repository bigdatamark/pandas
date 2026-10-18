# [MODULE DOC]
# Tests for DatetimeIndex.tz_convert: NaT handling, scalar/vector consistency, DST transitions
# (hour overflow regressions), round trips, local timezone, and freq behavior.
from datetime import datetime

import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._libs.tslibs import timezones

from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestTZConvert:
    # [CLASS DOC]
    # Groups tz_convert tests; no state.
    def test_tz_convert_nat(self):
        # [FUNCTION DOC]
        # Regression test: localizing then converting an index of only NaT, and of dates with NaT, across
        # Pacific, Eastern and UTC keeps NaT and shifts wall times correctly; adding an hour offset or a
        # numpy timedelta afterwards gives the expected wall times in the current zone.
        # GH#5546
        dates = [NaT]
        idx = DatetimeIndex(dates)
        idx = idx.tz_localize("US/Pacific")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Pacific"))
        idx = idx.tz_convert("US/Eastern")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Eastern"))
        idx = idx.tz_convert("UTC")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="UTC"))

        dates = ["2010-12-01 00:00", "2010-12-02 00:00", NaT]
        idx = DatetimeIndex(dates)
        idx = idx.tz_localize("US/Pacific")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Pacific"))
        idx = idx.tz_convert("US/Eastern")
        expected = ["2010-12-01 03:00", "2010-12-02 03:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))

        idx = idx + offsets.Hour(5)
        expected = ["2010-12-01 08:00", "2010-12-02 08:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))
        idx = idx.tz_convert("US/Pacific")
        expected = ["2010-12-01 05:00", "2010-12-02 05:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Pacific"))

        idx = idx + np.timedelta64(3, "h")
        expected = ["2010-12-01 08:00", "2010-12-02 08:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Pacific"))

        idx = idx.tz_convert("US/Eastern")
        expected = ["2010-12-01 11:00", "2010-12-02 11:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_tz_convert_compat_timestamp(self, prefix):
        # [FUNCTION DOC]
        # Parametrized over timezone name prefix (plain or dateutil). Verifies converting a single stamp
        # equals the first element of the converted index.
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
        idx = DatetimeIndex(strdates, tz=prefix + "US/Eastern")

        conv = idx[0].tz_convert(prefix + "US/Pacific")
        expected = idx.tz_convert(prefix + "US/Pacific")[0]

        assert conv == expected

    def test_dti_tz_convert_hour_overflow_dst(self):
        # [FUNCTION DOC]
        # Regression test: hours after converting Eastern-localized dates (sorted and unsorted, across
        # DST) to UTC and UTC to Eastern are as expected (int32 Index).
        # Regression test for GH#13306

        # sorted case US/Eastern -> UTC
        ts = ["2008-05-12 09:50:00", "2008-12-12 09:50:35", "2009-05-12 09:50:32"]
        tt = DatetimeIndex(ts).tz_localize("US/Eastern")
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # sorted case UTC -> US/Eastern
        ts = ["2008-05-12 13:50:00", "2008-12-12 14:50:35", "2009-05-12 13:50:32"]
        tt = DatetimeIndex(ts).tz_localize("UTC")
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case US/Eastern -> UTC
        ts = ["2008-05-12 09:50:00", "2008-12-12 09:50:35", "2008-05-12 09:50:32"]
        tt = DatetimeIndex(ts).tz_localize("US/Eastern")
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case UTC -> US/Eastern
        ts = ["2008-05-12 13:50:00", "2008-12-12 14:50:35", "2008-05-12 13:50:32"]
        tt = DatetimeIndex(ts).tz_localize("UTC")
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_convert_hour_overflow_dst_timestamps(self, tz):
        # [FUNCTION DOC]
        # Parametrized over Eastern timezone spelling. Same as the previous test but building the index
        # from timezone-aware Timestamps rather than localizing.
        # Regression test for GH#13306

        # sorted case US/Eastern -> UTC
        ts = [
            Timestamp("2008-05-12 09:50:00", tz=tz),
            Timestamp("2008-12-12 09:50:35", tz=tz),
            Timestamp("2009-05-12 09:50:32", tz=tz),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # sorted case UTC -> US/Eastern
        ts = [
            Timestamp("2008-05-12 13:50:00", tz="UTC"),
            Timestamp("2008-12-12 14:50:35", tz="UTC"),
            Timestamp("2009-05-12 13:50:32", tz="UTC"),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case US/Eastern -> UTC
        ts = [
            Timestamp("2008-05-12 09:50:00", tz=tz),
            Timestamp("2008-12-12 09:50:35", tz=tz),
            Timestamp("2008-05-12 09:50:32", tz=tz),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case UTC -> US/Eastern
        ts = [
            Timestamp("2008-05-12 13:50:00", tz="UTC"),
            Timestamp("2008-12-12 14:50:35", tz="UTC"),
            Timestamp("2008-05-12 13:50:32", tz="UTC"),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

    @pytest.mark.parametrize("freq, n", [("h", 1), ("min", 60), ("s", 3600)])
    def test_dti_tz_convert_trans_pos_plus_1__bug(self, freq, n):
        # [FUNCTION DOC]
        # Parametrized over (frequency, repetitions per hour) for hourly, minute and second ranges.
        # Regression test: converting a UTC range to Moscow time across a transition gives the expected
        # hour pattern (3, 4, 5 with repeated counts).
        # Regression test for tslib.tz_convert(vals, tz1, tz2).
        # See GH#4496 for details.
        idx = date_range(datetime(2011, 3, 26, 23), datetime(2011, 3, 27, 1), freq=freq)
        idx = idx.tz_localize("UTC")
        idx = idx.tz_convert("Europe/Moscow")

        expected = np.repeat(np.array([3, 4, 5]), np.array([n, n, 1]))
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

    def test_dti_tz_convert_dst(self):
        # [FUNCTION DOC]
        # Verifies hours after converting between UTC and US/Eastern ranges at hourly, minute and second
        # frequencies around the spring and fall DST transitions, and daily ranges across them.
        # KNOWN ISSUE: one end date string in the last case is written with a three-digit hour field
        # (zero-padded oddly); it parses but looks like a typo.
        for freq, n in [("h", 1), ("min", 60), ("s", 3600)]:
            # Start DST
            idx = date_range(
                "2014-03-08 23:00", "2014-03-09 09:00", freq=freq, tz="UTC"
            )
            idx = idx.tz_convert("US/Eastern")
            expected = np.repeat(
                np.array([18, 19, 20, 21, 22, 23, 0, 1, 3, 4, 5]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            idx = date_range(
                "2014-03-08 18:00", "2014-03-09 05:00", freq=freq, tz="US/Eastern"
            )
            idx = idx.tz_convert("UTC")
            expected = np.repeat(
                np.array([23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            # End DST
            idx = date_range(
                "2014-11-01 23:00", "2014-11-02 09:00", freq=freq, tz="UTC"
            )
            idx = idx.tz_convert("US/Eastern")
            expected = np.repeat(
                np.array([19, 20, 21, 22, 23, 0, 1, 1, 2, 3, 4]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            idx = date_range(
                "2014-11-01 18:00", "2014-11-02 05:00", freq=freq, tz="US/Eastern"
            )
            idx = idx.tz_convert("UTC")
            expected = np.repeat(
                np.array([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                np.array([n, n, n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

        # daily
        # Start DST
        idx = date_range("2014-03-08 00:00", "2014-03-09 00:00", freq="D", tz="UTC")
        idx = idx.tz_convert("US/Eastern")
        tm.assert_index_equal(idx.hour, Index([19, 19], dtype=np.int32))

        idx = date_range(
            "2014-03-08 00:00", "2014-03-09 00:00", freq="D", tz="US/Eastern"
        )
        idx = idx.tz_convert("UTC")
        tm.assert_index_equal(idx.hour, Index([5, 5], dtype=np.int32))

        # End DST
        idx = date_range("2014-11-01 00:00", "2014-11-02 00:00", freq="D", tz="UTC")
        idx = idx.tz_convert("US/Eastern")
        tm.assert_index_equal(idx.hour, Index([20, 20], dtype=np.int32))

        idx = date_range(
            "2014-11-01 00:00", "2014-11-02 000:00", freq="D", tz="US/Eastern"
        )
        idx = idx.tz_convert("UTC")
        tm.assert_index_equal(idx.hour, Index([4, 4], dtype=np.int32))

    def test_tz_convert_roundtrip(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Uses tz_aware_fixture. Verifies that converting UTC ranges (monthly, daily, hourly,
        # minutely) to a timezone and back to naive equals the original naive range, and equals the
        # UTC-converted naive range with inferred freq; the result has no tzinfo.
        tz = tz_aware_fixture
        idx1 = date_range(start="2014-01-01", end="2014-12-31", freq="ME", tz="UTC")
        exp1 = date_range(start="2014-01-01", end="2014-12-31", freq="ME")

        idx2 = date_range(start="2014-01-01", end="2014-12-31", freq="D", tz="UTC")
        exp2 = date_range(start="2014-01-01", end="2014-12-31", freq="D")

        idx3 = date_range(start="2014-01-01", end="2014-03-01", freq="h", tz="UTC")
        exp3 = date_range(start="2014-01-01", end="2014-03-01", freq="h")

        idx4 = date_range(start="2014-08-01", end="2014-10-31", freq="min", tz="UTC")
        exp4 = date_range(start="2014-08-01", end="2014-10-31", freq="min")

        for idx, expected in [(idx1, exp1), (idx2, exp2), (idx3, exp3), (idx4, exp4)]:
            converted = idx.tz_convert(tz)
            reset = converted.tz_convert(None)
            tm.assert_index_equal(reset, expected)
            assert reset.tzinfo is None
            expected = converted.tz_convert("UTC").tz_localize(None)
            expected = expected._with_freq("infer")
            tm.assert_index_equal(reset, expected)

    def test_dti_tz_convert_tzlocal(self):
        # [FUNCTION DOC]
        # Verifies converting UTC to the local dateutil timezone and local to naive does not change
        # the underlying integer values.
        # GH#13583
        # tz_convert doesn't affect to internal
        dti = date_range(start="2001-01-01", end="2001-03-01", tz="UTC")
        dti2 = dti.tz_convert(dateutil.tz.tzlocal())
        tm.assert_numpy_array_equal(dti2.asi8, dti.asi8)

        dti = date_range(start="2001-01-01", end="2001-03-01", tz=dateutil.tz.tzlocal())
        dti2 = dti.tz_convert(None)
        tm.assert_numpy_array_equal(dti2.asi8, dti.asi8)

    @pytest.mark.parametrize(
        "tz",
        [
            "US/Eastern",
            "dateutil/US/Eastern",
            "pytz/US/Eastern",
            gettz("US/Eastern"),
        ],
    )
    def test_dti_tz_convert_utc_to_local_no_modify(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone given as a name, a dateutil name, a pytz name (skipped if pytz is
        # unavailable) and a zone object. Verifies that converting a UTC hourly range to local time leaves
        # the integer values unchanged and the resulting timezone compares equal to the requested one.
        if isinstance(tz, str) and tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        rng = date_range("3/11/2012", "3/12/2012", freq="h", tz="utc")
        rng_eastern = rng.tz_convert(tz)

        # Values are unmodified
        tm.assert_numpy_array_equal(rng.asi8, rng_eastern.asi8)

        assert timezones.tz_compare(rng_eastern.tz, timezones.maybe_get_tz(tz))

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_tz_convert_unsorted(self, tzstr):
        # [FUNCTION DOC]
        # Parametrized over timezone spelling. Verifies that the hours of a reversed converted index equal
        # the reversed hours of the original (no ordering assumption).
        dr = date_range("2012-03-09", freq="h", periods=100, tz="utc")
        dr = dr.tz_convert(tzstr)

        result = dr[::-1].hour
        exp = dr.hour[::-1]
        tm.assert_almost_equal(result, exp)

    def test_dti_tz_convert_day_freq_not_preserved(self):
        # [FUNCTION DOC]
        # Regression test: converting a daily range in London to UTC keeps values equal but drops the
        # freq.
        # GH#51716
        dti = date_range("2020-3-28", periods=5, freq="D", tz="Europe/London")
        result = dti.tz_convert("UTC")
        assert (result == dti).all()
        assert result.freq is None
