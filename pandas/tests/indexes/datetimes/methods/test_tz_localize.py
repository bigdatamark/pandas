# [MODULE DOC]
# Tests for DatetimeIndex.tz_localize: freq invalidation, copying, nonexistent and ambiguous times
# (raise, NaT, infer, flags, shift), conversions to UTC, round trips, and local timezone. Defines a
# module-level fixture of equivalent US/Eastern timezone objects.
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from zoneinfo import ZoneInfo

import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Timestamp,
    bdate_range,
    date_range,
    offsets,
    to_datetime,
)
import pandas._testing as tm


@pytest.fixture(params=["pytz/US/Eastern", gettz("US/Eastern"), ZoneInfo("US/Eastern")])
def tz(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over a pytz Eastern timezone (skipped if pytz is unavailable), a
    # dateutil Eastern timezone and a zoneinfo Eastern timezone; returns the timezone object.
    if isinstance(request.param, str) and request.param.startswith("pytz/"):
        pytz = pytest.importorskip("pytz")
        return pytz.timezone(request.param.removeprefix("pytz/"))
    return request.param


class TestTZLocalize:
    # [CLASS DOC]
    # Groups tz_localize tests; no state.
    def test_tz_localize_invalidates_freq(self):
        # [FUNCTION DOC]
        # Verifies localizing keeps hourly freq for no-op and UTC localization, drops freq (and does not
        # infer one) when localizing across a US/Eastern DST jump with shifted nonexistent times, and
        # keeps freq for a length-one index.
        # we only preserve freq in unambiguous cases

        # if localized to US/Eastern, this crosses a DST transition
        dti = date_range("2014-03-08 23:00", "2014-03-09 09:00", freq="h")
        assert dti.freq == "h"

        result = dti.tz_localize(None)  # no-op
        assert result.freq == "h"

        result = dti.tz_localize("UTC")  # unambiguous freq preservation
        assert result.freq == "h"

        result = dti.tz_localize("US/Eastern", nonexistent="shift_forward")
        assert result.freq is None
        assert result.inferred_freq is None  # i.e. we are not _too_ strict here

        # Case where we _can_ keep freq because we're length==1
        dti2 = dti[:1]
        result = dti2.tz_localize("US/Eastern")
        assert result.freq == "h"

    def test_tz_localize_utc_copies(self, utc_fixture):
        # [FUNCTION DOC]
        # Uses utc_fixture. Regression test: localizing naive data to UTC returns a new object that does
        # not share memory, for the index and for its array.
        # GH#46460
        times = ["2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 03:00"]
        index = DatetimeIndex(times)

        res = index.tz_localize(utc_fixture)
        assert not tm.shares_memory(res, index)

        res2 = index._data.tz_localize(utc_fixture)
        assert not tm.shares_memory(index._data, res2)

    def test_dti_tz_localize_nonexistent_raise_coerce(self):
        # [FUNCTION DOC]
        # Verifies localizing times that include a nonexistent spring-forward time raises a value error
        # listing the times by default and with nonexistent set to raise, and with NaT coercion gives NaT
        # in the nonexistent slot and the expected Eastern stamps elsewhere.
        # GH#13057
        times = ["2015-03-08 01:00", "2015-03-08 02:00", "2015-03-08 03:00"]
        index = DatetimeIndex(times)
        tz = "US/Eastern"
        with pytest.raises(ValueError, match="|".join(times)):
            index.tz_localize(tz=tz)

        with pytest.raises(ValueError, match="|".join(times)):
            index.tz_localize(tz=tz, nonexistent="raise")

        result = index.tz_localize(tz=tz, nonexistent="NaT")
        test_times = ["2015-03-08 01:00-05:00", "NaT", "2015-03-08 03:00-04:00"]
        dti = to_datetime(test_times, utc=True)
        expected = dti.tz_convert("US/Eastern")
        tm.assert_index_equal(result, expected)

    def test_dti_tz_localize_ambiguous_infer(self, tz):
        # [FUNCTION DOC]
        # Uses the tz fixture. Verifies inferring ambiguity fails with a value error saying the DST time
        # cannot be inferred when the range has no repeated hours around a fall-back date.
        # November 6, 2011, fall back, repeat 2 AM hour
        # With no repeated hours, we cannot infer the transition
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour())
        with pytest.raises(ValueError, match="Cannot infer dst time"):
            dr.tz_localize(tz)

    def test_dti_tz_localize_ambiguous_infer2(self, tz, unit):
        # [FUNCTION DOC]
        # Uses tz and unit. Verifies that with repeated hours the ambiguous time is inferred by
        # tz_localize and by the constructor, giving the expected range without freq.
        # With repeated hours, we can infer the transition
        dr = date_range(
            datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit
        )
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        di = DatetimeIndex(times).as_unit(unit)
        result = di.tz_localize(tz, ambiguous="infer")
        expected = dr._with_freq(None)
        tm.assert_index_equal(result, expected)
        result2 = DatetimeIndex(times, tz=tz, ambiguous="infer").as_unit(unit)
        tm.assert_index_equal(result2, expected)

    def test_dti_tz_localize_ambiguous_infer3(self, tz):
        # [FUNCTION DOC]
        # Uses tz. Verifies that infer on a range without a DST transition equals the plain localization.
        # When there is no dst transition, nothing special happens
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        localized = dr.tz_localize(tz)
        localized_infer = dr.tz_localize(tz, ambiguous="infer")
        tm.assert_index_equal(localized, localized_infer)

    def test_dti_tz_localize_ambiguous_times(self, tz):
        # [FUNCTION DOC]
        # Uses tz. Verifies localizing a range through the spring-forward gap raises a value error naming
        # the nonexistent time, and through the fall-back repeat raises the cannot-infer error.
        # KNOWN ISSUE: two of the ranges built below (one after the transition, one in UTC) are assigned
        # but never checked, so those parts of the test make no assertion.
        # March 13, 2011, spring forward, skip from 2 AM to 3 AM
        dr = date_range(datetime(2011, 3, 13, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(ValueError, match="2011-03-13 02:30:00"):
            dr.tz_localize(tz)

        # after dst transition, it works
        dr = date_range(
            datetime(2011, 3, 13, 3, 30), periods=3, freq=offsets.Hour(), tz=tz
        )

        # November 6, 2011, fall back, repeat 2 AM hour
        dr = date_range(datetime(2011, 11, 6, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(ValueError, match="Cannot infer dst time"):
            dr.tz_localize(tz)

        # UTC is OK
        dr = date_range(
            datetime(2011, 3, 13), periods=48, freq=offsets.Minute(30), tz=timezone.utc
        )

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_localize_pass_dates_to_utc(self, tzstr):
        # [FUNCTION DOC]
        # Parametrized over timezone spelling. Verifies localizing naive dates equals constructing with
        # the timezone (same tz and same UTC values).
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]

        idx = DatetimeIndex(strdates)
        conv = idx.tz_localize(tzstr)

        fromdates = DatetimeIndex(strdates, tz=tzstr)

        assert conv.tz == fromdates.tz
        tm.assert_numpy_array_equal(conv.values, fromdates.values)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_tz_localize(self, prefix):
        # [FUNCTION DOC]
        # Parametrized over name prefix. Verifies localizing a millisecond range to Eastern gives the same
        # UTC integer values as a UTC range five hours later, converting to Pacific keeps values, and
        # ranges within the ambiguous or nonexistent hour raise value errors.
        tzstr = prefix + "US/Eastern"
        dti = date_range(start="1/1/2005", end="1/1/2005 0:00:02.256", freq="ms")
        dti2 = dti.tz_localize(tzstr)

        dti_utc = date_range(
            start="1/1/2005 05:00", end="1/1/2005 5:00:02.256", freq="ms", tz="utc"
        )

        tm.assert_numpy_array_equal(dti2.values, dti_utc.values)

        dti3 = dti2.tz_convert(prefix + "US/Pacific")
        tm.assert_numpy_array_equal(dti3.values, dti_utc.values)

        dti = date_range(start="11/6/2011 1:59:59", end="11/6/2011 2:00", freq="ms")
        with pytest.raises(ValueError, match="Cannot infer dst time"):
            dti.tz_localize(tzstr)

        dti = date_range(start="3/13/2011 1:59:59", end="3/13/2011 2:00", freq="ms")
        with pytest.raises(ValueError, match="2011-03-13 02:00:00"):
            dti.tz_localize(tzstr)

    def test_dti_tz_localize_utc_conversion(self, tz):
        # [FUNCTION DOC]
        # Uses tz. Verifies localizing a half-hourly range converts to UTC (five hours ahead for the
        # pre-DST day) and that a range crossing spring forward raises a value error naming the
        # nonexistent time.
        # Localizing to time zone should:
        #  1) check for DST ambiguities
        #  2) convert to UTC

        rng = date_range("3/10/2012", "3/11/2012", freq="30min")

        converted = rng.tz_localize(tz)
        expected_naive = rng + offsets.Hour(5)
        tm.assert_numpy_array_equal(converted.asi8, expected_naive.asi8)

        # DST ambiguity, this should fail
        rng = date_range("3/11/2012", "3/12/2012", freq="30min")
        # Is this really how it should fail??
        with pytest.raises(ValueError, match="2012-03-11 02:00:00"):
            rng.tz_localize(tz)

    def test_dti_tz_localize_roundtrip(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Uses tz_aware_fixture. Verifies a localized index cannot be localized again (type error
        # suggesting tz_convert), and localizing to none restores the naive index with freq dropped.
        # note: this tz tests that a tz-naive index can be localized
        # and de-localized successfully, when there are no DST transitions
        # in the range.
        idx = date_range(start="2014-06-01", end="2014-08-30", freq="15min")
        tz = tz_aware_fixture
        localized = idx.tz_localize(tz)
        # can't localize a tz-aware object
        with pytest.raises(
            TypeError, match="Already tz-aware, use tz_convert to convert"
        ):
            localized.tz_localize(tz)
        reset = localized.tz_localize(None)
        assert reset.tzinfo is None
        expected = idx._with_freq(None)
        tm.assert_index_equal(reset, expected)

    def test_dti_tz_localize_naive(self):
        # [FUNCTION DOC]
        # Verifies localizing an hourly naive range to US/Pacific equals the timezone-aware range (freq
        # dropped).
        rng = date_range("1/1/2011", periods=100, freq="h")

        conv = rng.tz_localize("US/Pacific")
        exp = date_range("1/1/2011", periods=100, freq="h", tz="US/Pacific")

        tm.assert_index_equal(conv, exp._with_freq(None))

    def test_dti_tz_localize_tzlocal(self):
        # [FUNCTION DOC]
        # Regression test: localizing to the local dateutil timezone shifts the underlying integers by the
        # local UTC offset (and removing it shifts back).
        # GH#13583
        offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
        offset = int(offset.total_seconds() * 1000000000)

        dti = date_range(start="2001-01-01", end="2001-03-01", unit="ns")
        dti2 = dti.tz_localize(dateutil.tz.tzlocal())
        tm.assert_numpy_array_equal(dti2.asi8 + offset, dti.asi8)

        dti = date_range(
            start="2001-01-01", end="2001-03-01", tz=dateutil.tz.tzlocal(), unit="ns"
        )
        dti2 = dti.tz_localize(None)
        tm.assert_numpy_array_equal(dti2.asi8 - offset, dti.asi8)

    def test_dti_tz_localize_ambiguous_nat(self, tz):
        # [FUNCTION DOC]
        # Uses tz. Verifies ambiguous set to NaT turns the two ambiguous times into NaT.
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        di = DatetimeIndex(times)
        localized = di.tz_localize(tz, ambiguous="NaT")

        times = [
            "11/06/2011 00:00",
            np.nan,
            np.nan,
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]
        di_test = DatetimeIndex(times, tz="US/Eastern")

        # left dtype is datetime64[ns, US/Eastern]
        # right is datetime64[ns, tzfile('/usr/share/zoneinfo/US/Eastern')]
        tm.assert_numpy_array_equal(di_test.values, localized.values)

    def test_dti_tz_localize_ambiguous_flags(self, tz, unit):
        # [FUNCTION DOC]
        # Uses tz and unit. Verifies explicit ambiguity flags (list, numeric array, boolean array) select
        # the right transition for tz_localize and the constructor; a flag list of the wrong length raises
        # an error about length; for doubled times with a doubled flag array the result equals the range
        # appended to itself.
        # November 6, 2011, fall back, repeat 2 AM hour

        # Pass in flags to determine right dst transition
        dr = date_range(
            datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit
        )
        times = [
            "11/06/2011 00:00",
            "11/06/2011 01:00",
            "11/06/2011 01:00",
            "11/06/2011 02:00",
            "11/06/2011 03:00",
        ]

        # Test tz_localize
        di = DatetimeIndex(times).as_unit(unit)
        is_dst = [1, 1, 0, 0, 0]
        localized = di.tz_localize(tz, ambiguous=is_dst)
        expected = dr._with_freq(None)
        tm.assert_index_equal(expected, localized)

        result = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(result, expected)

        localized = di.tz_localize(tz, ambiguous=np.array(is_dst))
        tm.assert_index_equal(dr, localized)

        localized = di.tz_localize(tz, ambiguous=np.array(is_dst).astype("bool"))
        tm.assert_index_equal(dr, localized)

        # Test constructor
        localized = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(dr, localized)

        # Test duplicate times where inferring the dst fails
        times += times
        di = DatetimeIndex(times).as_unit(unit)

        # When the sizes are incompatible, make sure error is raised
        msg = "Length of ambiguous bool-array must be the same size as vals"
        with pytest.raises(Exception, match=msg):
            di.tz_localize(tz, ambiguous=is_dst)

        # When sizes are compatible and there are repeats ('infer' won't work)
        is_dst = np.hstack((is_dst, is_dst))
        localized = di.tz_localize(tz, ambiguous=is_dst)
        dr = dr.append(dr)
        tm.assert_index_equal(dr, localized)

    def test_dti_tz_localize_ambiguous_flags2(self, tz):
        # [FUNCTION DOC]
        # Uses tz. Verifies ambiguity flags are accepted when there is no transition and give the same
        # result as plain localization.
        # When there is no dst transition, nothing special happens
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        is_dst = np.array([1] * 10)
        localized = dr.tz_localize(tz)
        localized_is_dst = dr.tz_localize(tz, ambiguous=is_dst)
        tm.assert_index_equal(localized, localized_is_dst)

    def test_dti_tz_localize_bdate_range(self):
        # [FUNCTION DOC]
        # Verifies localizing a business-day range to UTC equals the business-day range created with the
        # UTC timezone.
        dr = bdate_range("1/1/2009", "1/1/2010")
        dr_utc = bdate_range("1/1/2009", "1/1/2010", tz=timezone.utc)
        localized = dr.tz_localize(timezone.utc)
        tm.assert_index_equal(dr_utc, localized)

    @pytest.mark.parametrize(
        "start_ts, tz, end_ts, shift",
        [
            ["2015-03-29 02:20:00", "Europe/Warsaw", "2015-03-29 03:00:00", "forward"],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:59:59.999999999",
                "backward",
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 03:20:00",
                timedelta(hours=1),
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:20:00",
                timedelta(hours=-1),
            ],
            ["2018-03-11 02:33:00", "US/Pacific", "2018-03-11 03:00:00", "forward"],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:59:59.999999999",
                "backward",
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 03:33:00",
                timedelta(hours=1),
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:33:00",
                timedelta(hours=-1),
            ],
        ],
    )
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    def test_dti_tz_localize_nonexistent_shift(
        self, start_ts, tz, end_ts, shift, tz_type, unit
    ):
        # [FUNCTION DOC]
        # Parametrized over (start, timezone, expected end, shift) for Warsaw and US Pacific gaps (shift
        # forward, shift backward, positive and negative timedelta), a timezone type prefix (plain or
        # dateutil), and unit. Verifies localizing a nonexistent time with the named shift gives the
        # expected shifted stamp.
        # GH#8917
        tz = tz_type + tz
        if isinstance(shift, str):
            shift = "shift_" + shift
        dti = DatetimeIndex([Timestamp(start_ts)]).as_unit(unit)
        result = dti.tz_localize(tz, nonexistent=shift)
        expected = DatetimeIndex([Timestamp(end_ts)]).tz_localize(tz).as_unit(unit)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("offset", [-1, 1])
    def test_dti_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
        # [FUNCTION DOC]
        # Parametrized over offsets of plus and minus one second; uses the warsaw timezone fixture.
        # Verifies a timedelta shift that leaves the time nonexistent raises a value error saying the
        # provided timedelta would relocalize on a nonexistent time.
        # GH#8917
        tz = warsaw
        dti = DatetimeIndex([Timestamp("2015-03-29 02:20:00")])
        msg = "The provided timedelta will relocalize on a nonexistent time"
        with pytest.raises(ValueError, match=msg):
            dti.tz_localize(tz, nonexistent=timedelta(seconds=offset))
