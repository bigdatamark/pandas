"""
Tests for DatetimeIndex timezone-related methods
"""

# [MODULE DOC]
# Tests for DatetimeIndex timezone-related behavior: dropping, localization,
# conversion, equality across zones and construction with zone objects.
# Defines a small tzinfo subclass and a module-level fixed-offset instance.

from datetime import (
    datetime,
    timedelta,
    timezone,
    tzinfo,
)
import zoneinfo

from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._libs.tslibs import (
    conversion,
    timezones,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Timestamp,
    bdate_range,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm


class FixedOffset(tzinfo):
    # [CLASS DOC]
    # Minimal tzinfo implementation for tests: a fixed offset in minutes east of UTC
    # with an optional name, no daylight saving. Stores the offset as a duration and
    # the name privately.
    """Fixed offset in minutes east from UTC."""

    def __init__(self, offset, name) -> None:
        # [FUNCTION DOC]
        # Stores the offset (minutes) converted to a duration, and the zone name (may be none).
        self.__offset = timedelta(minutes=offset)
        self.__name = name

    def utcoffset(self, dt):
        # [FUNCTION DOC]
        # Returns the stored fixed offset regardless of the datetime argument.
        return self.__offset

    def tzname(self, dt):
        # [FUNCTION DOC]
        # Returns the stored zone name regardless of the datetime argument (may be none).
        return self.__name

    def dst(self, dt):
        # [FUNCTION DOC]
        # Returns a zero duration since there is no daylight saving.
        return timedelta(0)


fixed_off_no_name = FixedOffset(-330, None)


class TestDatetimeIndexTimezones:
    # [CLASS DOC]
    # Groups timezone-related tests for DatetimeIndex.
    # -------------------------------------------------------------
    # Unsorted

    def test_dti_drop_dont_lose_tz(self):
        # [FUNCTION DOC]
        # Verifies (GH#2621) dropping the last element of a UTC range keeps the timezone.
        # GH#2621
        ind = date_range("2012-12-01", periods=10, tz="utc")
        ind = ind.drop(ind[-1])

        assert ind.tz is not None

    def test_dti_tz_conversion_freq(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Parametrized over tz_naive_fixture. Verifies (GH25241) that localizing a naive
        # index and converting a UTC index keep the frequency unchanged.
        # GH25241
        t3 = DatetimeIndex(["2019-01-01 10:00"], freq="h")
        assert t3.tz_localize(tz=tz_naive_fixture).freq == t3.freq
        t4 = DatetimeIndex(["2019-01-02 12:00"], tz="UTC", freq="min")
        assert t4.tz_convert(tz="UTC").freq == t4.freq

    def test_drop_dst_boundary(self):
        # [FUNCTION DOC]
        # Verifies (gh-18031) that dropping the first element from a range spanning a
        # Brussels daylight saving end boundary gives the expected index in which the
        # repeated wall times are disambiguated by the ambiguous flags.
        # see gh-18031
        tz = "Europe/Brussels"
        freq = "15min"

        start = Timestamp("201710290100", tz=tz)
        end = Timestamp("201710290300", tz=tz)
        index = date_range(start=start, end=end, freq=freq, unit="ns")

        expected = DatetimeIndex(
            [
                "201710290115",
                "201710290130",
                "201710290145",
                "201710290200",
                "201710290215",
                "201710290230",
                "201710290245",
                "201710290200",
                "201710290215",
                "201710290230",
                "201710290245",
                "201710290300",
            ],
            dtype="M8[ns, Europe/Brussels]",
            freq=freq,
            ambiguous=[
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        )
        result = index.drop(index[0])
        tm.assert_index_equal(result, expected)

    def test_date_range_localize(self, unit):
        # [FUNCTION DOC]
        # Parametrized over unit. Verifies a range created with a US/Eastern zone across
        # the March 2012 DST transition equals a naive range localized afterwards (aside
        # from frequency), that the first element has the wall hour three, and that a
        # slice equals an explicitly constructed index of the same zone.
        rng = date_range(
            "3/11/2012 03:00", periods=15, freq="h", tz="US/Eastern", unit=unit
        )
        rng2 = DatetimeIndex(
            ["3/11/2012 03:00", "3/11/2012 04:00"], dtype=f"M8[{unit}, US/Eastern]"
        )
        rng3 = date_range("3/11/2012 03:00", periods=15, freq="h", unit=unit)
        rng3 = rng3.tz_localize("US/Eastern")

        tm.assert_index_equal(rng._with_freq(None), rng3)

        # DST transition time
        val = rng[0]
        exp = Timestamp("3/11/2012 03:00", tz="US/Eastern")

        assert val.hour == 3
        assert exp.hour == 3
        assert val == exp  # same UTC value
        tm.assert_index_equal(rng[:2], rng2)

    def test_date_range_localize2(self, unit):
        # [FUNCTION DOC]
        # Parametrized over unit. Verifies a range just before the March 2012 DST
        # transition matches an explicit index and its elements' hours, and that a longer
        # range skips a wall hour at the transition.
        # Right before the DST transition
        rng = date_range(
            "3/11/2012 00:00", periods=2, freq="h", tz="US/Eastern", unit=unit
        )
        rng2 = DatetimeIndex(
            ["3/11/2012 00:00", "3/11/2012 01:00"],
            dtype=f"M8[{unit}, US/Eastern]",
            freq="h",
        )
        tm.assert_index_equal(rng, rng2)
        exp = Timestamp("3/11/2012 00:00", tz="US/Eastern")
        assert exp.hour == 0
        assert rng[0] == exp
        exp = Timestamp("3/11/2012 01:00", tz="US/Eastern")
        assert exp.hour == 1
        assert rng[1] == exp

        rng = date_range(
            "3/11/2012 00:00", periods=10, freq="h", tz="US/Eastern", unit=unit
        )
        assert rng[2].hour == 3

    def test_timestamp_equality_different_timezones(self):
        # [FUNCTION DOC]
        # Verifies elements of a UTC range compare equal to the same instants in
        # Eastern and Berlin zones, elementwise and via vectorised comparison.
        utc_range = date_range("1/1/2000", periods=20, tz="UTC")
        eastern_range = utc_range.tz_convert("US/Eastern")
        berlin_range = utc_range.tz_convert("Europe/Berlin")

        for a, b, c in zip(utc_range, eastern_range, berlin_range):
            assert a == b
            assert b == c
            assert a == c

        assert (utc_range == eastern_range).all()
        assert (utc_range == berlin_range).all()
        assert (berlin_range == eastern_range).all()

    def test_dti_equals_with_tz(self):
        # [FUNCTION DOC]
        # Verifies two indexes with the same wall times in different zones are not equal.
        left = date_range("1/1/2011", periods=100, freq="h", tz="utc")
        right = date_range("1/1/2011", periods=100, freq="h", tz="US/Eastern")

        assert not left.equals(right)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_nat(self, tzstr):
        # [FUNCTION DOC]
        # Parametrized over zone name spellings. Verifies an index of a localized
        # timestamp and a missing value reports the second entry as missing and the first
        # as zone-aware.
        idx = DatetimeIndex([Timestamp("2013-1-1", tz=tzstr), pd.NaT])

        assert isna(idx[1])
        assert idx[0].tzinfo is not None

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_utc_box_timestamp_and_localize(self, tzstr):
        # [FUNCTION DOC]
        # Parametrized over zone name spellings. Verifies converting a UTC range to a
        # zone yields timestamps equal to astimezone conversions with identical tzinfo
        # and, for dateutil zones, the daylight tzinfo representation after DST starts.
        tz = timezones.maybe_get_tz(tzstr)

        rng = date_range("3/11/2012", "3/12/2012", freq="h", tz="utc")
        rng_eastern = rng.tz_convert(tzstr)

        expected = rng[-1].astimezone(tz)

        stamp = rng_eastern[-1]
        assert stamp == expected
        assert stamp.tzinfo == expected.tzinfo

        # right tzinfo
        rng = date_range("3/13/2012", "3/14/2012", freq="h", tz="utc")
        rng_eastern = rng.tz_convert(tzstr)
        if "dateutil" in tzstr:
            assert "EDT" in repr(rng_eastern[0].tzinfo) or "tzfile" in repr(
                rng_eastern[0].tzinfo
            )

    @pytest.mark.parametrize(
        "tz", [zoneinfo.ZoneInfo("US/Central"), gettz("US/Central")]
    )
    def test_with_tz(self, tz):
        # [FUNCTION DOC]
        # Parametrized over a standard-library and a dateutil Central zone. Verifies
        # business date range construction with timezone-aware start, naive endpoints
        # with a zone, and conversion to a zone keeping the identical zone object and the
        # localization of the first element. Also verifies that giving a tz-aware start
        # and a different explicit zone raises an error about different timezones.
        # just want it to work
        start = datetime(2011, 3, 12, tzinfo=timezone.utc)
        dr = bdate_range(start, periods=50, freq=pd.offsets.Hour())
        assert dr.tz is timezone.utc

        # DateRange with naive datetimes
        dr = bdate_range("1/1/2005", "1/1/2009", tz=timezone.utc)
        dr = bdate_range("1/1/2005", "1/1/2009", tz=tz)

        # normalized
        central = dr.tz_convert(tz)
        assert central.tz is tz
        naive = central[0].to_pydatetime().replace(tzinfo=None)
        comp = conversion.localize_pydatetime(naive, tz).tzinfo
        assert central[0].tz is comp

        # compare vs a localized tz
        naive = dr[0].to_pydatetime().replace(tzinfo=None)
        comp = conversion.localize_pydatetime(naive, tz).tzinfo
        assert central[0].tz is comp

        # datetimes with tzinfo set
        dr = bdate_range(
            datetime(2005, 1, 1, tzinfo=timezone.utc),
            datetime(2009, 1, 1, tzinfo=timezone.utc),
        )
        msg = "Start and end cannot both be tz-aware with different timezones"
        with pytest.raises(Exception, match=msg):
            bdate_range(datetime(2005, 1, 1, tzinfo=timezone.utc), "1/1/2009", tz=tz)

    @pytest.mark.parametrize(
        "tz", [zoneinfo.ZoneInfo("US/Eastern"), gettz("US/Eastern")]
    )
    def test_dti_convert_tz_aware_datetime_datetime(self, tz):
        # [FUNCTION DOC]
        # Parametrized over standard-library and dateutil Eastern zones. Verifies
        # (GH#1581) that building an index from localized datetimes keeps a comparable
        # zone, and that converting them to UTC gives integer values equal to the
        # timestamps' values and the UTC zone.
        # GH#1581
        dates = [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]

        dates_aware = [conversion.localize_pydatetime(x, tz) for x in dates]
        result = DatetimeIndex(dates_aware).as_unit("ns")
        assert timezones.tz_compare(result.tz, tz)

        converted = to_datetime(dates_aware, utc=True).as_unit("ns")
        ex_vals = np.array([Timestamp(x).as_unit("ns")._value for x in dates_aware])
        tm.assert_numpy_array_equal(converted.asi8, ex_vals)
        assert converted.tz is timezone.utc
