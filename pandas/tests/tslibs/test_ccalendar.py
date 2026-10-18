# [MODULE DOC]
# Tests for the low-level calendar helpers: day-of-year computation and ISO calendar (year, week, weekday)
# computation, including comparison with the standard library and a property-based test over the timestamp range.
# No import-time side effects.

from datetime import (
    date,
    datetime,
)

from hypothesis import given
import numpy as np
import pytest

from pandas._libs.tslibs import ccalendar

from pandas._testing._hypothesis import DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ


@pytest.mark.parametrize(
    "date_tuple,expected",
    [
        ((2001, 3, 1), 60),
        ((2004, 3, 1), 61),
        ((1907, 12, 31), 365),  # End-of-year, non-leap year.
        ((2004, 12, 31), 366),  # End-of-year, leap year.
    ],
)
def test_get_day_of_year_numeric(date_tuple, expected):
    # [FUNCTION DOC]
    # Parametrized over (year, month, day) tuples with expected day numbers: March 1 in a non-leap year is day 60, in a leap year day 61, and December 31 is day 365 or 366 depending on leap year.
    assert ccalendar.get_day_of_year(*date_tuple) == expected


def test_get_day_of_year_dt():
    # [FUNCTION DOC]
    # Picks a random date (seeded generator, ordinal up to about 4000 years) and checks the helper equals the number of days since January 1 of the same year plus one.
    dt = datetime.fromordinal(1 + np.random.default_rng(2).integers(365 * 4000))
    result = ccalendar.get_day_of_year(dt.year, dt.month, dt.day)

    expected = (dt - dt.replace(month=1, day=1)).days + 1
    assert result == expected


@pytest.mark.parametrize(
    "input_date_tuple, expected_iso_tuple",
    [
        [(2020, 1, 1), (2020, 1, 3)],
        [(2019, 12, 31), (2020, 1, 2)],
        [(2019, 12, 30), (2020, 1, 1)],
        [(2009, 12, 31), (2009, 53, 4)],
        [(2010, 1, 1), (2009, 53, 5)],
        [(2010, 1, 3), (2009, 53, 7)],
        [(2010, 1, 4), (2010, 1, 1)],
        [(2006, 1, 1), (2005, 52, 7)],
        [(2005, 12, 31), (2005, 52, 6)],
        [(2008, 12, 28), (2008, 52, 7)],
        [(2008, 12, 29), (2009, 1, 1)],
    ],
)
def test_dt_correct_iso_8601_year_week_and_day(input_date_tuple, expected_iso_tuple):
    # [FUNCTION DOC]
    # Parametrized over input date tuples and expected ISO (year, week, weekday) tuples, including year-boundary cases where the ISO year differs from the calendar year. The helper's result must equal both the standard library's ISO calendar result and the expected tuple.
    result = ccalendar.get_iso_calendar(*input_date_tuple)
    expected_from_date_isocalendar = date(*input_date_tuple).isocalendar()
    assert result == expected_from_date_isocalendar
    assert result == expected_iso_tuple


@pytest.mark.slow
@given(DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ)
def test_isocalendar(dt):
    # [FUNCTION DOC]
    # Property test (slow) over datetimes in the timestamp range without timezone: the helper's ISO calendar for the year, month and day equals the standard library's ISO calendar.
    expected = dt.isocalendar()
    result = ccalendar.get_iso_calendar(dt.year, dt.month, dt.day)
    assert result == expected
