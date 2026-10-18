# [MODULE DOC]
# Tests for the abstract holiday calendar machinery and the US federal holiday
# calendar: holiday listing for different input types, rule caching across instances,
# observance stability, rule lookup by name, interplay with custom business day offsets,
# and calendars without rules. No import-time side effects.
from datetime import datetime

import pytest

from pandas import (
    DatetimeIndex,
    offsets,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    Timestamp,
    USFederalHolidayCalendar,
    USLaborDay,
    USThanksgivingDay,
    get_calendar,
)


@pytest.mark.parametrize(
    "transform", [lambda x: x, lambda x: x.strftime("%Y-%m-%d"), lambda x: Timestamp(x)]
)
def test_calendar(transform):
    # [FUNCTION DOC]
    # Parametrized over three input transforms (identity, ISO date string, Timestamp). Asks the US
    # federal calendar for holidays between 2012-01-01 and 2012-12-31 and expects the ten observed
    # federal holidays for 2012 (Jan 2, Jan 16, Feb 20, May 28, Jul 4, Sep 3, Oct 8, Nov 12,
    # Nov 22, Dec 25) as datetimes, independent of the input type.
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2012, 12, 31)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(transform(start_date), transform(end_date))

    expected = [
        datetime(2012, 1, 2),
        datetime(2012, 1, 16),
        datetime(2012, 2, 20),
        datetime(2012, 5, 28),
        datetime(2012, 7, 4),
        datetime(2012, 9, 3),
        datetime(2012, 10, 8),
        datetime(2012, 11, 12),
        datetime(2012, 11, 22),
        datetime(2012, 12, 25),
    ]

    assert list(holidays.to_pydatetime()) == expected


def test_calendar_caching():
    # [FUNCTION DOC]
    # see gh-9552. Defines a trivial calendar subclass, then two instances each with a single
    # one-year rule for 2015-01-01 and 2015-01-02. Verifies the holiday lists are independent (no cross-instance
    # cache leakage): the first gives only Jan 1, the second only Jan 2, both microsecond resolution.
    # see gh-9552.

    class TestCalendar(AbstractHolidayCalendar):
        # [CLASS DOC]
        # HELPER: local calendar subclass for the caching test; forwards an optional name and rules to the
        # abstract calendar constructor without adding behavior.
        def __init__(self, name=None, rules=None) -> None:
            # [FUNCTION DOC]
            # HELPER: constructor of the local test calendar; accepts a name and a list of rules (both
            # default to none) and passes them straight to the parent initializer. Returns nothing.
            super().__init__(name=name, rules=rules)

    jan1 = TestCalendar(rules=[Holiday("jan1", year=2015, month=1, day=1)])
    jan2 = TestCalendar(rules=[Holiday("jan2", year=2015, month=1, day=2)])

    # Getting holidays for Jan 1 should not alter results for Jan 2.
    expected = DatetimeIndex(["01-Jan-2015"]).as_unit("us")
    tm.assert_index_equal(jan1.holidays(), expected)

    expected2 = DatetimeIndex(["02-Jan-2015"]).as_unit("us")
    tm.assert_index_equal(jan2.holidays(), expected2)


def test_calendar_observance_dates():
    # [FUNCTION DOC]
    # see gh-11477. Gets the US federal calendar by name and queries holidays three times: a single
    # day range (July 3 2015, the observed Independence Day), a range July 3 to July 6, and the single day again.
    # All results must be equal, showing that a different earlier range does not change later output.
    # see gh-11477
    us_fed_cal = get_calendar("USFederalHolidayCalendar")
    holidays0 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 3)
    )  # <-- same start and end dates
    holidays1 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 6)
    )  # <-- different start and end dates
    holidays2 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 3)
    )  # <-- same start and end dates

    # These should all produce the same result.
    #
    # In addition, calling with different start and end
    # dates should not alter the output if we call the
    # function again with the same start and end date.
    tm.assert_index_equal(holidays0, holidays1)
    tm.assert_index_equal(holidays0, holidays2)


def test_rule_from_name():
    # [FUNCTION DOC]
    # Looking up the rule named "Thanksgiving Day" on the US federal calendar returns the predefined
    # Thanksgiving rule object.
    us_fed_cal = get_calendar("USFederalHolidayCalendar")
    assert us_fed_cal.rule_from_name("Thanksgiving Day") == USThanksgivingDay


def test_calendar_2031():
    # [FUNCTION DOC]
    # See gh-27790. Defines a calendar whose only rule is US Labor Day, builds a custom business day offset from it,
    # and adds zero business days to Saturday 2031-08-30. Expects the next working day
    # Tuesday 2031-09-02, because Labor Day 2031 (Monday Sep 1) is a holiday.
    # See gh-27790
    #
    # Labor Day 2031 is on September 1. Saturday before is August 30.
    # Next working day after August 30 ought to be Tuesday, September 2.

    class testCalendar(AbstractHolidayCalendar):
        # [CLASS DOC]
        # HELPER: local calendar subclass with a single rule, Labor Day, used by the 2031 test.
        rules = [USLaborDay]

    cal = testCalendar()
    workDay = offsets.CustomBusinessDay(calendar=cal)
    Sat_before_Labor_Day_2031 = to_datetime("2031-08-30")
    next_working_day = Sat_before_Labor_Day_2031 + 0 * workDay
    assert next_working_day == to_datetime("2031-09-02")


def test_no_holidays_calendar():
    # [FUNCTION DOC]
    # Issue #31415. A calendar subclass with no rules returns an empty datetime index for a one-year query.
    # Test for issue #31415

    class NoHolidaysCalendar(AbstractHolidayCalendar):
        # [CLASS DOC]
        # HELPER: local calendar subclass that defines no rules; used to test the empty calendar case.
        pass

    cal = NoHolidaysCalendar()
    holidays = cal.holidays(Timestamp("01-Jan-2020"), Timestamp("01-Jan-2021"))
    empty_index = DatetimeIndex([])  # Type is DatetimeIndex since return_name=False
    tm.assert_index_equal(holidays, empty_index)
