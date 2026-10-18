# [MODULE DOC]
# ASV benchmarks for the timestamp scalar type: construction from many inputs,
# attribute access, operations, and replacement across a daylight saving
# transition. Public contents: classes TimestampConstruction,
# TimestampProperties, TimestampOps and TimestampAcrossDst. Uses the timezone
# list from the sibling tslib benchmark module. No import-time side effects
# beyond imports.
from datetime import (
    datetime,
    timezone,
)
import zoneinfo

import numpy as np

from pandas import Timestamp

from .tslib import _tzs


class TimestampConstruction:
    # [CLASS DOC]
    # Benchmarks constructing timestamps from strings, numeric stamps, numpy
    # datetimes, standard datetimes and existing timestamps. State set by
    # setup: a numpy datetime64, a naive datetime, a UTC-aware datetime and a
    # pandas timestamp, all at 2020-01-01 midnight.
    def setup(self):
        # [FUNCTION DOC]
        # Creates the four reference values described for the class.
        self.npdatetime64 = np.datetime64("2020-01-01 00:00:00")
        self.dttime_unaware = datetime(2020, 1, 1, 0, 0, 0)
        self.dttime_aware = datetime(2020, 1, 1, 0, 0, 0, 0, timezone.utc)
        self.ts = Timestamp("2020-01-01 00:00:00")

    def time_parse_iso8601_no_tz(self):
        # [FUNCTION DOC]
        # Times parsing an ISO 8601 string with no time zone.
        Timestamp("2017-08-25 08:16:14")

    def time_parse_iso8601_tz(self):
        # [FUNCTION DOC]
        # Times parsing an ISO 8601 string carrying a numeric UTC offset.
        Timestamp("2017-08-25 08:16:14-0500")

    def time_parse_dateutil(self):
        # [FUNCTION DOC]
        # Times parsing a non-ISO string (slashes and AM marker) that falls
        # back to the flexible third-party parser.
        Timestamp("2017/08/25 08:16:14 AM")

    def time_parse_today(self):
        # [FUNCTION DOC]
        # Times the special string for the current date.
        Timestamp("today")

    def time_parse_now(self):
        # [FUNCTION DOC]
        # Times the special string for the current moment.
        Timestamp("now")

    def time_fromordinal(self):
        # [FUNCTION DOC]
        # Times constructing from a proleptic Gregorian ordinal day number.
        Timestamp.fromordinal(730120)

    def time_fromtimestamp(self):
        # [FUNCTION DOC]
        # Times constructing from a POSIX timestamp in seconds (local time).
        Timestamp.fromtimestamp(1515448538)

    def time_from_npdatetime64(self):
        # [FUNCTION DOC]
        # Times constructing from the stored numpy datetime64.
        Timestamp(self.npdatetime64)

    def time_from_datetime_unaware(self):
        # [FUNCTION DOC]
        # Times constructing from the stored naive datetime.
        Timestamp(self.dttime_unaware)

    def time_from_datetime_aware(self):
        # [FUNCTION DOC]
        # Times constructing from the stored UTC-aware datetime.
        Timestamp(self.dttime_aware)

    def time_from_pd_timestamp(self):
        # [FUNCTION DOC]
        # Times constructing from the stored pandas timestamp.
        Timestamp(self.ts)


class TimestampProperties:
    # [CLASS DOC]
    # Benchmarks reading properties and simple name methods of a timestamp,
    # parameterized over the shared timezone list. State set by setup: one
    # timestamp at 2017-08-25 08:16:14 with the timezone attached.
    # KNOWN ISSUE: the parameter list is wrapped in an extra list, unlike the
    # sibling TimestampOps class; this works because asv treats the outer list as
    # the list of parameter axes, but it differs in style.
    params = [_tzs]
    param_names = ["tz"]

    def setup(self, tz):
        # [FUNCTION DOC]
        # Creates the timestamp from the string using the tzinfo keyword with
        # the given timezone (which may be none).
        self.ts = Timestamp("2017-08-25 08:16:14", tzinfo=tz)

    def time_tz(self, tz):
        # [FUNCTION DOC]
        # Times reading the timezone attribute.
        self.ts.tz

    def time_dayofweek(self, tz):
        # [FUNCTION DOC]
        # Times reading the day-of-week attribute.
        self.ts.dayofweek

    def time_dayofyear(self, tz):
        # [FUNCTION DOC]
        # Times reading the day-of-year attribute.
        self.ts.dayofyear

    def time_week(self, tz):
        # [FUNCTION DOC]
        # Times reading the week-of-year attribute.
        self.ts.week

    def time_quarter(self, tz):
        # [FUNCTION DOC]
        # Times reading the quarter attribute.
        self.ts.quarter

    def time_days_in_month(self, tz):
        # [FUNCTION DOC]
        # Times reading the days-in-month attribute.
        self.ts.days_in_month

    def time_is_month_start(self, tz):
        # [FUNCTION DOC]
        # Times reading the first-day-of-month flag.
        self.ts.is_month_start

    def time_is_month_end(self, tz):
        # [FUNCTION DOC]
        # Times reading the last-day-of-month flag.
        self.ts.is_month_end

    def time_is_quarter_start(self, tz):
        # [FUNCTION DOC]
        # Times reading the first-day-of-quarter flag.
        self.ts.is_quarter_start

    def time_is_quarter_end(self, tz):
        # [FUNCTION DOC]
        # Times reading the last-day-of-quarter flag.
        self.ts.is_quarter_end

    def time_is_year_start(self, tz):
        # [FUNCTION DOC]
        # Times reading the first-day-of-year flag.
        self.ts.is_year_start

    def time_is_year_end(self, tz):
        # [FUNCTION DOC]
        # Times reading the last-day-of-year flag.
        self.ts.is_year_end

    def time_is_leap_year(self, tz):
        # [FUNCTION DOC]
        # Times reading the leap-year flag.
        self.ts.is_leap_year

    def time_microsecond(self, tz):
        # [FUNCTION DOC]
        # Times reading the microsecond component.
        self.ts.microsecond

    def time_month_name(self, tz):
        # [FUNCTION DOC]
        # Times calling the method that returns the month name.
        self.ts.month_name()

    def time_weekday_name(self, tz):
        # [FUNCTION DOC]
        # Times calling the method that returns the weekday name.
        self.ts.day_name()


class TimestampOps:
    # [CLASS DOC]
    # Benchmarks timestamp methods (replace, conversion, rounding, time zone
    # conversion and localization), parameterized over the shared timezone list.
    # State set by setup: one timestamp at 2017-08-25 08:16:14 with the
    # timezone attached via the tz keyword.
    params = _tzs
    param_names = ["tz"]

    def setup(self, tz):
        # [FUNCTION DOC]
        # Creates the timestamp from the string with the given timezone (which
        # may be none).
        self.ts = Timestamp("2017-08-25 08:16:14", tz=tz)

    def time_replace_tz(self, tz):
        # [FUNCTION DOC]
        # Times replacing the timezone with the US Eastern zone.
        self.ts.replace(tzinfo=zoneinfo.ZoneInfo("US/Eastern"))

    def time_replace_None(self, tz):
        # [FUNCTION DOC]
        # Times replacing the timezone with none (making it naive).
        self.ts.replace(tzinfo=None)

    def time_to_pydatetime(self, tz):
        # [FUNCTION DOC]
        # Times converting to a standard library datetime.
        self.ts.to_pydatetime()

    def time_normalize(self, tz):
        # [FUNCTION DOC]
        # Times normalizing the timestamp to midnight.
        self.ts.normalize()

    def time_tz_convert(self, tz):
        # [FUNCTION DOC]
        # Times converting to the parameter timezone, but only when the
        # timestamp is timezone-aware; otherwise does nothing.
        if self.ts.tz is not None:
            self.ts.tz_convert(tz)

    def time_tz_localize(self, tz):
        # [FUNCTION DOC]
        # Times localizing to the parameter timezone, but only when the
        # timestamp is naive; otherwise does nothing.
        if self.ts.tz is None:
            self.ts.tz_localize(tz)

    def time_to_julian_date(self, tz):
        # [FUNCTION DOC]
        # Times conversion to a Julian date.
        self.ts.to_julian_date()

    def time_floor(self, tz):
        # [FUNCTION DOC]
        # Times rounding down to a five-minute boundary.
        self.ts.floor("5min")

    def time_ceil(self, tz):
        # [FUNCTION DOC]
        # Times rounding up to a five-minute boundary.
        self.ts.ceil("5min")


class TimestampAcrossDst:
    # [CLASS DOC]
    # Benchmarks replacing the timezone of a timestamp that lies at a daylight
    # saving transition. State set by setup: the Berlin timezone info and a
    # timestamp built from a naive datetime on 2016-03-27 at 01:00.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a naive datetime for 2016-03-27 01:00 (first fold), converts it
        # to Europe/Berlin to extract the resulting tzinfo, and keeps a
        # timestamp made from the original naive datetime.
        dt = datetime(2016, 3, 27, 1, fold=0)
        self.tzinfo = dt.astimezone(zoneinfo.ZoneInfo("Europe/Berlin")).tzinfo
        self.ts2 = Timestamp(dt)

    def time_replace_across_dst(self):
        # [FUNCTION DOC]
        # Times replacing the timezone of the naive timestamp with the Berlin
        # tzinfo.
        self.ts2.replace(tzinfo=self.tzinfo)
