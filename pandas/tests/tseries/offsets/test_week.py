"""
Tests for the following offsets:
- Week
- WeekOfMonth
- LastWeekOfMonth
"""

# [MODULE DOC]
# Tests for the weekly offsets: plain/anchored weekly offset, the nth-week-of-month offset, and the last-week-of-month offset.
# Covers representation, argument validation errors, table-driven application, on-offset checks (including regression cases with
# nanosecond timezone-aware timestamps), and an error when adding an incompatible offset. Uses shared helpers and a weekday enumeration
# from the offsets common test module; no import-time side effects other than building case tables on the test classes.

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    Day,
    LastWeekOfMonth,
    Week,
    WeekOfMonth,
)

from pandas.tests.tseries.offsets.common import (
    WeekDay,
    assert_is_on_offset,
    assert_offset_equal,
)


class TestWeek:
    # [CLASS DOC]
    # Tests for the weekly offset (optionally anchored to a weekday). Class attribute holds the offset application case table.
    def test_repr(self):
        # [FUNCTION DOC]
        # Representation includes the weekday, shows a minus one count as a product with the singular name, and a count of minus two with the plural name.
        assert repr(Week(weekday=0)) == "<Week: weekday=0>"
        assert repr(Week(n=-1, weekday=0)) == "<-1 * Week: weekday=0>"
        assert repr(Week(n=-2, weekday=0)) == "<-2 * Weeks: weekday=0>"

    def test_corner(self):
        # [FUNCTION DOC]
        # Constructing a weekly offset with a weekday of 7 or minus 1 raises a value error whose message starts with the day must be text.
        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=7)

        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=-1)

    offset_cases = []
    # not business week
    offset_cases.append(
        (
            Week(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 8),
                datetime(2008, 1, 4): datetime(2008, 1, 11),
                datetime(2008, 1, 5): datetime(2008, 1, 12),
                datetime(2008, 1, 6): datetime(2008, 1, 13),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # Mon
    offset_cases.append(
        (
            Week(weekday=0),
            {
                datetime(2007, 12, 31): datetime(2008, 1, 7),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # n=0 -> roll forward. Mon
    offset_cases.append(
        (
            Week(0, weekday=0),
            {
                datetime(2007, 12, 31): datetime(2007, 12, 31),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 7),
            },
        )
    )

    # n=0 -> roll forward. Mon
    offset_cases.append(
        (
            Week(-2, weekday=1),
            {
                datetime(2010, 4, 6): datetime(2010, 3, 23),
                datetime(2010, 4, 8): datetime(2010, 3, 30),
                datetime(2010, 4, 5): datetime(2010, 3, 23),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the case table (unanchored weekly offset, Monday-anchored, n of 0 which rolls forward, and n of minus two anchored to Tuesday). For each base date asserts the offset gives the expected date.
        # KNOWN ISSUE: the source comment above the case with n of minus two repeats the n=0 roll-forward comment, which does not describe that case.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("weekday", range(7))
    def test_is_on_offset(self, weekday):
        # [FUNCTION DOC]
        # Parametrized over weekdays 0 to 6. Iterates over January 1 to 7 2008 computing a date and whether it is the anchor weekday.
        # KNOWN ISSUE: the assertion call is placed after the loop instead of inside it, so only the last date of the loop (January 7) is actually checked against the on-offset helper.
        offset = Week(weekday=weekday)

        for day in range(1, 8):
            date = datetime(2008, 1, day)
            expected = day % 7 == weekday
        assert_is_on_offset(offset, date, expected)

    @pytest.mark.parametrize(
        "n,date",
        [
            (2, "1862-01-13 09:03:34.873477378+0210"),
            (-2, "1856-10-24 16:18:36.556360110-0717"),
        ],
    )
    def test_is_on_offset_weekday_none(self, n, date):
        # [FUNCTION DOC]
        # Regression test (GH 18510). Parametrized over n of 2 and minus 2 with old timezone-aware nanosecond timestamps. For an unanchored weekly offset without normalization the fast on-offset check must equal the slow definition (add then subtract returns the original).
        # GH 18510 Week with weekday = None, normalize = False
        # should always be is_on_offset
        offset = Week(n=n, weekday=None)
        ts = Timestamp(date, tz="Africa/Lusaka")
        fast = offset.is_on_offset(ts)
        slow = (ts + offset) - offset == ts
        assert fast == slow

    def test_week_add_invalid(self):
        # [FUNCTION DOC]
        # Adding a day offset to a weekly offset anchored to a weekday raises a type error saying it cannot add (rather than an attribute error).
        # Week with weekday should raise TypeError and _not_ AttributeError
        #  when adding invalid offset
        offset = Week(weekday=1)
        other = Day()
        with pytest.raises(TypeError, match="Cannot add"):
            offset + other


class TestWeekOfMonth:
    # [CLASS DOC]
    # Tests for the offset anchoring to the nth given weekday in a month. Class attribute holds the on-offset cases (week, weekday, date, expected).
    def test_constructor(self):
        # [FUNCTION DOC]
        # Invalid arguments raise value errors: week 4 or minus 1 gives an error beginning with Week; weekday minus 1 or minus 7 gives an error beginning with Day.
        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=4, weekday=0)

        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=-1, weekday=0)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-7)

    def test_repr(self):
        # [FUNCTION DOC]
        # Representation lists week then weekday values.
        assert (
            repr(WeekOfMonth(weekday=1, week=2)) == "<WeekOfMonth: week=2, weekday=1>"
        )

    def test_offset(self):
        # [FUNCTION DOC]
        # Uses four Tuesdays in January 2011 as bases. A table of (n, week, weekday, base, expected) cases is applied through the shared assertion helper, covering negative, zero and positive counts, and various weeks and weekdays. Also checks subtracting the offset from a datetime moves to the expected earlier anchor date (January 12 and February 2 2011).
        date1 = datetime(2011, 1, 4)  # 1st Tuesday of Month
        date2 = datetime(2011, 1, 11)  # 2nd Tuesday of Month
        date3 = datetime(2011, 1, 18)  # 3rd Tuesday of Month
        date4 = datetime(2011, 1, 25)  # 4th Tuesday of Month

        # see for loop for structure
        test_cases = [
            (-2, 2, 1, date1, datetime(2010, 11, 16)),
            (-2, 2, 1, date2, datetime(2010, 11, 16)),
            (-2, 2, 1, date3, datetime(2010, 11, 16)),
            (-2, 2, 1, date4, datetime(2010, 12, 21)),
            (-1, 2, 1, date1, datetime(2010, 12, 21)),
            (-1, 2, 1, date2, datetime(2010, 12, 21)),
            (-1, 2, 1, date3, datetime(2010, 12, 21)),
            (-1, 2, 1, date4, datetime(2011, 1, 18)),
            (0, 0, 1, date1, datetime(2011, 1, 4)),
            (0, 0, 1, date2, datetime(2011, 2, 1)),
            (0, 0, 1, date3, datetime(2011, 2, 1)),
            (0, 0, 1, date4, datetime(2011, 2, 1)),
            (0, 1, 1, date1, datetime(2011, 1, 11)),
            (0, 1, 1, date2, datetime(2011, 1, 11)),
            (0, 1, 1, date3, datetime(2011, 2, 8)),
            (0, 1, 1, date4, datetime(2011, 2, 8)),
            (0, 0, 1, date1, datetime(2011, 1, 4)),
            (0, 1, 1, date2, datetime(2011, 1, 11)),
            (0, 2, 1, date3, datetime(2011, 1, 18)),
            (0, 3, 1, date4, datetime(2011, 1, 25)),
            (1, 0, 0, date1, datetime(2011, 2, 7)),
            (1, 0, 0, date2, datetime(2011, 2, 7)),
            (1, 0, 0, date3, datetime(2011, 2, 7)),
            (1, 0, 0, date4, datetime(2011, 2, 7)),
            (1, 0, 1, date1, datetime(2011, 2, 1)),
            (1, 0, 1, date2, datetime(2011, 2, 1)),
            (1, 0, 1, date3, datetime(2011, 2, 1)),
            (1, 0, 1, date4, datetime(2011, 2, 1)),
            (1, 0, 2, date1, datetime(2011, 1, 5)),
            (1, 0, 2, date2, datetime(2011, 2, 2)),
            (1, 0, 2, date3, datetime(2011, 2, 2)),
            (1, 0, 2, date4, datetime(2011, 2, 2)),
            (1, 2, 1, date1, datetime(2011, 1, 18)),
            (1, 2, 1, date2, datetime(2011, 1, 18)),
            (1, 2, 1, date3, datetime(2011, 2, 15)),
            (1, 2, 1, date4, datetime(2011, 2, 15)),
            (2, 2, 1, date1, datetime(2011, 2, 15)),
            (2, 2, 1, date2, datetime(2011, 2, 15)),
            (2, 2, 1, date3, datetime(2011, 3, 15)),
            (2, 2, 1, date4, datetime(2011, 3, 15)),
        ]

        for n, week, weekday, dt, expected in test_cases:
            offset = WeekOfMonth(n, week=week, weekday=weekday)
            assert_offset_equal(offset, dt, expected)

        # try subtracting
        result = datetime(2011, 2, 1) - WeekOfMonth(week=1, weekday=2)
        assert result == datetime(2011, 1, 12)

        result = datetime(2011, 2, 3) - WeekOfMonth(week=0, weekday=2)
        assert result == datetime(2011, 2, 2)

    on_offset_cases = [
        (0, 0, datetime(2011, 2, 7), True),
        (0, 0, datetime(2011, 2, 6), False),
        (0, 0, datetime(2011, 2, 14), False),
        (1, 0, datetime(2011, 2, 14), True),
        (0, 1, datetime(2011, 2, 1), True),
        (0, 1, datetime(2011, 2, 8), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the on-offset cases; builds the offset from week and weekday and asserts its on-offset check equals expected.
        week, weekday, dt, expected = case
        offset = WeekOfMonth(week=week, weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    @pytest.mark.parametrize(
        "n,week,date,tz",
        [
            (2, 2, "1916-05-15 01:14:49.583410462+0422", "Asia/Qyzylorda"),
            (-3, 1, "1980-12-08 03:38:52.878321185+0500", "Asia/Oral"),
        ],
    )
    def test_is_on_offset_nanoseconds(self, n, week, date, tz):
        # [FUNCTION DOC]
        # Regression test (GH 18864). Parametrized over historical timezone-aware nanosecond timestamps; the fast on-offset check must equal the slow add-then-subtract definition, so nanoseconds do not break the check.
        # GH 18864
        # Make sure that nanoseconds don't trip up is_on_offset (and with it apply)
        offset = WeekOfMonth(n=n, week=week, weekday=0)
        ts = Timestamp(date, tz=tz)
        fast = offset.is_on_offset(ts)
        slow = (ts + offset) - offset == ts
        assert fast == slow


class TestLastWeekOfMonth:
    # [CLASS DOC]
    # Tests for the offset anchoring to the last given weekday of each month. Class attribute holds the on-offset cases.
    def test_constructor(self):
        # [FUNCTION DOC]
        # A count of zero raises a value error beginning with N cannot be 0; a weekday of minus 1 or 7 raises a value error beginning with Day.
        with pytest.raises(ValueError, match="^N cannot be 0"):
            LastWeekOfMonth(n=0, weekday=1)

        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=7)

    def test_offset(self):
        # [FUNCTION DOC]
        # Using last Saturday of August 2013, last Thursday of January 2013 and a Sunday-anchored offset: one day before an anchor date moves to it, one day after moves to the next month's anchor, adding on the anchor date moves to the next month's anchor, and other nearby offsets (three days before, two days after) behave accordingly. A Sunday offset applied to July 31 2013 gives August 25.
        # Saturday
        last_sat = datetime(2013, 8, 31)
        next_sat = datetime(2013, 9, 28)
        offset_sat = LastWeekOfMonth(n=1, weekday=5)

        one_day_before = last_sat + timedelta(days=-1)
        assert one_day_before + offset_sat == last_sat

        one_day_after = last_sat + timedelta(days=+1)
        assert one_day_after + offset_sat == next_sat

        # Test On that day
        assert last_sat + offset_sat == next_sat

        # Thursday

        offset_thur = LastWeekOfMonth(n=1, weekday=3)
        last_thurs = datetime(2013, 1, 31)
        next_thurs = datetime(2013, 2, 28)

        one_day_before = last_thurs + timedelta(days=-1)
        assert one_day_before + offset_thur == last_thurs

        one_day_after = last_thurs + timedelta(days=+1)
        assert one_day_after + offset_thur == next_thurs

        # Test on that day
        assert last_thurs + offset_thur == next_thurs

        three_before = last_thurs + timedelta(days=-3)
        assert three_before + offset_thur == last_thurs

        two_after = last_thurs + timedelta(days=+2)
        assert two_after + offset_thur == next_thurs

        offset_sunday = LastWeekOfMonth(n=1, weekday=WeekDay.SUN)
        assert datetime(2013, 7, 31) + offset_sunday == datetime(2013, 8, 25)

    on_offset_cases = [
        (WeekDay.SUN, datetime(2013, 1, 27), True),
        (WeekDay.SAT, datetime(2013, 3, 30), True),
        (WeekDay.MON, datetime(2013, 2, 18), False),  # Not the last Mon
        (WeekDay.SUN, datetime(2013, 2, 25), False),  # Not a SUN
        (WeekDay.MON, datetime(2013, 2, 25), True),
        (WeekDay.SAT, datetime(2013, 11, 30), True),
        (WeekDay.SAT, datetime(2006, 8, 26), True),
        (WeekDay.SAT, datetime(2007, 8, 25), True),
        (WeekDay.SAT, datetime(2008, 8, 30), True),
        (WeekDay.SAT, datetime(2009, 8, 29), True),
        (WeekDay.SAT, datetime(2010, 8, 28), True),
        (WeekDay.SAT, datetime(2011, 8, 27), True),
        (WeekDay.SAT, datetime(2019, 8, 31), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over (weekday, date, expected); asserts the offset's on-offset check equals expected.
        weekday, dt, expected = case
        offset = LastWeekOfMonth(weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    @pytest.mark.parametrize(
        "n,weekday,date,tz",
        [
            (4, 6, "1917-05-27 20:55:27.084284178+0200", "Europe/Warsaw"),
            (-4, 5, "2005-08-27 05:01:42.799392561-0500", "America/Rainy_River"),
        ],
    )
    def test_last_week_of_month_on_offset(self, n, weekday, date, tz):
        # [FUNCTION DOC]
        # Regression test (GH 19036, GH 18977), where the daylight saving adjustment was wrong. Parametrized over n/weekday/timezone-aware nanosecond timestamps; fast on-offset check equals slow add-then-subtract definition.
        # GH 19036, GH 18977 _adjust_dst was incorrect for LastWeekOfMonth
        offset = LastWeekOfMonth(n=n, weekday=weekday)
        ts = Timestamp(date, tz=tz)
        slow = (ts + offset) - offset == ts
        fast = offset.is_on_offset(ts)
        assert fast == slow

    def test_repr(self):
        # [FUNCTION DOC]
        # Representation of a count 2 Tuesday-anchored offset shows two times the plural class name and the weekday.
        assert (
            repr(LastWeekOfMonth(n=2, weekday=1)) == "<2 * LastWeekOfMonths: weekday=1>"
        )
