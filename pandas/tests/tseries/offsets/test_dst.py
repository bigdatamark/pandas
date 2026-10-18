"""
Tests for DateOffset additions over Daylight Savings Time
"""

# [MODULE DOC]
# Tests for date offsets applied to time-zone-aware timestamps around daylight saving time transitions: generic keyword offsets before spring-forward and fall-back, every month/quarter/year/week offset class on a DST date, ambiguous and nonexistent wall times raising errors.
# The module is skipped entirely when the pytz library is not installed (import-time skip) and imports pytz at module level. Uses the performance_warning fixture from the test configuration.
from datetime import timedelta

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CustomBusinessDay,
    DateOffset,
    Day,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    YearBegin,
    YearEnd,
)

from pandas import DatetimeIndex
import pandas._testing as tm

pytz = pytest.importorskip("pytz")


def get_utc_offset_hours(ts):
    # [FUNCTION DOC]
    # HELPER: converts a timestamp's UTC offset to a floating number of hours. Parameter: ts is a time-zone-aware timestamp. Returns the total offset in hours (days and seconds combined, divided by 3600).
    # take a Timestamp and compute total hours of utc offset
    o = ts.utcoffset()
    return (o.days * 24 * 3600 + o.seconds) / 3600.0


class TestDST:
    # [CLASS DOC]
    # Test collection for generic date offsets over DST. Class attributes: two timestamp strings one microsecond before the 2013 fall-back and spring-forward moments; a mapping from time zone (pytz US/Eastern, a dateutil US/Pacific name) to daylight and standard UTC offsets in hours; lists of valid singular and plural offset keyword names; and the offset_classes mapping from offset class to (start date, expected date) pairs. Helper methods drive the checks.
    # one microsecond before the DST transition
    ts_pre_fallback = "2013-11-03 01:59:59.999999"
    ts_pre_springfwd = "2013-03-10 01:59:59.999999"

    # test both basic names and dateutil timezones
    timezone_utc_offsets = {
        pytz.timezone("US/Eastern"): {
            "utc_offset_daylight": -4,
            "utc_offset_standard": -5,
        },
        "dateutil/US/Pacific": {"utc_offset_daylight": -7, "utc_offset_standard": -8},
    }
    valid_date_offsets_singular = [
        "weekday",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
    ]
    valid_date_offsets_plural = [
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
    ]

    def _test_all_offsets(self, n, performance_warning, **kwds):
        # [FUNCTION DOC]
        # HELPER: runs the offset check for every valid keyword name. Parameters: n is the offset count; performance_warning is the expected warning class (or none) for non-vectorized application; extra keyword arguments (start timestamp and expected UTC offset) are forwarded. Uses plural names when n is greater than 1, singular names otherwise. Returns nothing.
        valid_offsets = (
            self.valid_date_offsets_plural
            if n > 1
            else self.valid_date_offsets_singular
        )

        for name in valid_offsets:
            self._test_offset(
                offset_name=name,
                offset_n=n,
                performance_warning=performance_warning,
                **kwds,
            )

    def _test_offset(
        self, offset_name, offset_n, tstart, expected_utc_offset, performance_warning
    ):
        # [FUNCTION DOC]
        # HELPER: checks one keyword offset applied to a timestamp. Parameters: offset_name keyword (like hour or weeks); offset_n the amount; tstart the start timestamp; expected_utc_offset hours or none; performance_warning the warning class expected for non-vectorized application.
        # Special case: for singular hour/minute/second/microsecond with the Eastern fall-back start, the result is an ambiguous wall time so it expects a value error naming the ambiguous time, for scalar and for the vectorized index path (the latter must also emit the non-vectorized warning), then returns. Otherwise adds the offset; if an expected UTC offset is given the result's UTC offset must match. For weeks: date moves by seven days per week and weekday/hour/minute/second are unchanged. For days: date moves by the days and time of day unchanged. For singular names: the corresponding part of the result equals the offset value. Otherwise: result equals the same addition done in UTC then converted.
        # KNOWN ISSUE: the fallback comparison converts back to a hard-coded US/Pacific zone regardless of the start timestamp's zone; it still passes because timestamp equality ignores the zone, but it is misleading.
        offset = DateOffset(**{offset_name: offset_n})

        if (
            offset_name in ["hour", "minute", "second", "microsecond"]
            and offset_n == 1
            and tstart
            == Timestamp(
                "2013-11-03 01:59:59.999999-0500", tz=pytz.timezone("US/Eastern")
            )
        ):
            # This addition results in an ambiguous wall time
            err_msg = {
                "hour": "2013-11-03 01:59:59.999999",
                "minute": "2013-11-03 01:01:59.999999",
                "second": "2013-11-03 01:59:01.999999",
                "microsecond": "2013-11-03 01:59:59.000001",
            }[offset_name]
            with pytest.raises(ValueError, match=err_msg):
                tstart + offset
            # While we're here, let's check that we get the same behavior in a
            #  vectorized path
            dti = DatetimeIndex([tstart])
            warn_msg = "Non-vectorized DateOffset"
            with pytest.raises(ValueError, match=err_msg):
                with tm.assert_produces_warning(performance_warning, match=warn_msg):
                    dti + offset
            return

        t = tstart + offset
        if expected_utc_offset is not None:
            assert get_utc_offset_hours(t) == expected_utc_offset

        if offset_name == "weeks":
            # dates should match
            assert t.date() == timedelta(days=7 * offset.kwds["weeks"]) + tstart.date()
            # expect the same day of week, hour of day, minute, second, ...
            assert (
                t.dayofweek == tstart.dayofweek
                and t.hour == tstart.hour
                and t.minute == tstart.minute
                and t.second == tstart.second
            )
        elif offset_name == "days":
            # dates should match
            assert timedelta(offset.kwds["days"]) + tstart.date() == t.date()
            # expect the same hour of day, minute, second, ...
            assert (
                t.hour == tstart.hour
                and t.minute == tstart.minute
                and t.second == tstart.second
            )
        elif offset_name in self.valid_date_offsets_singular:
            # expect the singular offset value to match between tstart and t
            datepart_offset = getattr(
                t, offset_name if offset_name != "weekday" else "dayofweek"
            )
            assert datepart_offset == offset.kwds[offset_name]
        else:
            # the offset should be the same as if it was done in UTC
            assert t == (tstart.tz_convert("UTC") + offset).tz_convert(
                pytz.timezone("US/Pacific")
            )

    def _make_timestamp(self, string, hrs_offset, tz):
        # [FUNCTION DOC]
        # HELPER: builds a timestamp in a given zone. Parameters: string is the wall-clock date text; hrs_offset is the integer UTC offset in hours (negative for west); tz is the target zone. Appends a four-digit offset suffix, parses as a UTC-offset timestamp, and converts to tz. Returns the converted timestamp.
        if hrs_offset >= 0:
            offset_string = f"{hrs_offset:02d}00"
        else:
            offset_string = f"-{(hrs_offset * -1):02}00"
        return Timestamp(string + offset_string).tz_convert(tz)

    def test_springforward_plural(self, performance_warning):
        # [FUNCTION DOC]
        # Uses the performance_warning fixture. For each zone, starts one microsecond before spring-forward in standard time and applies plural offsets with n=3; each result must have the daylight UTC offset (or satisfy the per-keyword checks).
        # test moving from standard to daylight savings
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre = utc_offsets["utc_offset_standard"]
            hrs_post = utc_offsets["utc_offset_daylight"]
            self._test_all_offsets(
                n=3,
                performance_warning=performance_warning,
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                expected_utc_offset=hrs_post,
            )

    def test_fallback_singular(self, performance_warning):
        # [FUNCTION DOC]
        # Starts one microsecond before fall-back in standard time in each zone and applies singular offsets with n=1; no expected UTC offset is checked because the result zone may differ per keyword.
        # in the case of singular offsets, we don't necessarily know which utc
        # offset the new Timestamp will wind up in (the tz for 1 month may be
        # different from 1 second) so we don't specify an expected_utc_offset
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre = utc_offsets["utc_offset_standard"]
            self._test_all_offsets(
                n=1,
                performance_warning=performance_warning,
                tstart=self._make_timestamp(self.ts_pre_fallback, hrs_pre, tz),
                expected_utc_offset=None,
            )

    def test_springforward_singular(self, performance_warning):
        # [FUNCTION DOC]
        # Starts one microsecond before spring-forward in each zone and applies singular offsets with n=1; no expected UTC offset checked.
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre = utc_offsets["utc_offset_standard"]
            self._test_all_offsets(
                n=1,
                performance_warning=performance_warning,
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                expected_utc_offset=None,
            )

    offset_classes = {
        MonthBegin: ["11/2/2012", "12/1/2012"],
        MonthEnd: ["11/2/2012", "11/30/2012"],
        BMonthBegin: ["11/2/2012", "12/3/2012"],
        BMonthEnd: ["11/2/2012", "11/30/2012"],
        CBMonthBegin: ["11/2/2012", "12/3/2012"],
        CBMonthEnd: ["11/2/2012", "11/30/2012"],
        SemiMonthBegin: ["11/2/2012", "11/15/2012"],
        SemiMonthEnd: ["11/2/2012", "11/15/2012"],
        Week: ["11/2/2012", "11/9/2012"],
        YearBegin: ["11/2/2012", "1/1/2013"],
        YearEnd: ["11/2/2012", "12/31/2012"],
        BYearBegin: ["11/2/2012", "1/1/2013"],
        BYearEnd: ["11/2/2012", "12/31/2012"],
        QuarterBegin: ["11/2/2012", "12/1/2012"],
        QuarterEnd: ["11/2/2012", "12/31/2012"],
        BQuarterBegin: ["11/2/2012", "12/3/2012"],
        BQuarterEnd: ["11/2/2012", "12/31/2012"],
        Day: ["11/4/2012", "11/5/2012"],
    }.items()

    @pytest.mark.parametrize("tup", offset_classes)
    def test_all_offset_classes(self, tup):
        # [FUNCTION DOC]
        # Parametrized over offset_classes entries. Adding a default instance of each class (month, business month, semi-month, week, year, quarter variants and day) to an Eastern-time start date gives the expected Eastern-time date.
        offset, test_values = tup

        first = Timestamp(test_values[0], tz="US/Eastern") + offset()
        second = Timestamp(test_values[1], tz="US/Eastern")
        assert first == second


@pytest.mark.parametrize(
    "original_dt, target_dt, offset, tz",
    [
        (
            Timestamp("2021-10-01 01:15"),
            Timestamp("2021-10-31 01:15"),
            MonthEnd(1),
            "Europe/London",
        ),
        (
            Timestamp("2010-12-05 02:59"),
            Timestamp("2010-10-31 02:59"),
            SemiMonthEnd(-3),
            "Europe/Paris",
        ),
        (
            Timestamp("2021-10-31 01:20"),
            Timestamp("2021-11-07 01:20"),
            CustomBusinessDay(2, weekmask="Sun Mon"),
            "US/Eastern",
        ),
        (
            Timestamp("2020-04-03 01:30"),
            Timestamp("2020-11-01 01:30"),
            YearBegin(1, month=11),
            "America/Chicago",
        ),
    ],
)
def test_nontick_offset_with_ambiguous_time_error(original_dt, target_dt, offset, tz):
    # [FUNCTION DOC]
    # Parametrized over (original, target, offset, zone) for month end, semi-month end, custom business day and year begin offsets. Localizing the original and adding the offset raises a value error "Cannot infer dst time from <target>, try using the 'ambiguous' argument" because the target wall time is DST-ambiguous.
    # .apply for non-Tick offsets throws ValueError when the target dt
    # is dst-ambiguous
    localized_dt = original_dt.tz_localize(tz)

    msg = f"Cannot infer dst time from {target_dt}, try using the 'ambiguous' argument"
    with pytest.raises(ValueError, match=msg):
        localized_dt + offset


def test_add_day_nonexistent_raises():
    # [FUNCTION DOC]
    # Adding one calendar day to 2015-03-28 02:30 in Europe/Warsaw lands on a nonexistent time (spring-forward) and raises a value error "is a nonexistent time due to daylight savings time".
    # https://github.com/pandas-dev/pandas/issues/41943#issuecomment-1466978225

    ts = Timestamp("2015-03-28T2:30", tz="Europe/Warsaw")

    msg = "is a nonexistent time due to daylight savings time"
    with pytest.raises(ValueError, match=msg):
        ts + Day(1)
