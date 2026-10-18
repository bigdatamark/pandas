"""
Tests for helper functions in the cython tslibs.offsets
"""

# [MODULE DOC]
# Tests for helper functions of the compiled offsets library and calendar library: first and last business day of a month,
# month shifting with a day option, quarter-day rolling (with the month modulus of twelve and of three), and the roll convention
# helper. Defines a fixture over the four day-option strings. No import-time side effects.

from datetime import datetime

import pytest

from pandas._libs.tslibs.ccalendar import (
    get_firstbday,
    get_lastbday,
)
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday

from pandas import Timestamp


@pytest.fixture(params=["start", "end", "business_start", "business_end"])
def day_opt(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the day-option strings start, end, business_start and business_end; yields one string per test.
    return request.param


@pytest.mark.parametrize(
    "dt,exp_week_day,exp_last_day",
    [
        (datetime(2017, 11, 30), 3, 30),  # Business day.
        (datetime(1993, 10, 31), 6, 29),  # Non-business day.
    ],
)
def test_get_last_bday(dt, exp_week_day, exp_last_day):
    # [FUNCTION DOC]
    # Parametrized over (date, expected weekday number, expected last business day). First asserts the date's weekday is as stated, then that the last-business-day helper for its year and month equals the expected day: November 2017 (ends on a business day, 30) and October 1993 (ends on a Sunday, last business day 29).
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day


@pytest.mark.parametrize(
    "dt,exp_week_day,exp_first_day",
    [
        (datetime(2017, 4, 1), 5, 3),  # Non-weekday.
        (datetime(1993, 10, 1), 4, 1),  # Business day.
    ],
)
def test_get_first_bday(dt, exp_week_day, exp_first_day):
    # [FUNCTION DOC]
    # Parametrized over (date, expected weekday, expected first business day): April 2017 starts on a Saturday so the first business day is 3; October 1993 starts on a Friday so it is 1. Asserts the weekday then the helper result.
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day


@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (0, 15, datetime(2017, 11, 15)),
        (0, None, datetime(2017, 11, 30)),
        (1, "start", datetime(2017, 12, 1)),
        (-145, "end", datetime(2005, 10, 31)),
        (0, "business_end", datetime(2017, 11, 30)),
        (0, "business_start", datetime(2017, 11, 1)),
    ],
)
def test_shift_month_dt(months, day_opt, expected):
    # [FUNCTION DOC]
    # Parametrized over (months, day option, expected datetime). Shifting November 30 2017 by the number of months with the given day option (a specific day number, none to keep the day with clipping, start, end, business_end, business_start) equals the expected datetime.
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected


@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (1, "start", Timestamp("1929-06-01")),
        (-3, "end", Timestamp("1929-02-28")),
        (25, None, Timestamp("1931-06-5")),
        (-1, 31, Timestamp("1929-04-30")),
    ],
)
def test_shift_month_ts(months, day_opt, expected):
    # [FUNCTION DOC]
    # Parametrized as above but with a timestamp base of May 5 1929 and day options start, end, none (keep day) and 31 (clipped to month length). Asserts equality with the expected timestamp.
    ts = Timestamp("1929-05-05")
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected


def test_shift_month_error():
    # [FUNCTION DOC]
    # An invalid day option string raises a value error whose message contains that string.
    dt = datetime(2017, 11, 15)
    day_opt = "this should raise"

    with pytest.raises(ValueError, match=day_opt):
        liboffsets.shift_month(dt, 3, day_opt=day_opt)


@pytest.mark.parametrize(
    "other,expected",
    [
        # Before March 1.
        (datetime(2017, 2, 10), {2: 1, -7: -7, 0: 0}),
        # After March 1.
        (Timestamp("2014-03-15", tz="US/Eastern"), {2: 2, -7: -6, 0: 1}),
    ],
)
@pytest.mark.parametrize("n", [2, -7, 0])
def test_roll_qtrday_year(other, expected, n):
    # [FUNCTION DOC]
    # Parametrized over a date before or after March 1 (the latter timezone-aware) and over n of 2, minus 7, 0. With month 3, start day option and month modulus 12, the rolled count equals the expected value from the mapping (n is reduced by one when the date is before the anchor for positive n; increased by one after for negative n and for zero).
    month = 3
    day_opt = "start"  # `other` will be compared to March 1.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


@pytest.mark.parametrize(
    "other,expected",
    [
        # Before June 30.
        (datetime(1999, 6, 29), {5: 4, -7: -7, 0: 0}),
        # After June 30.
        (Timestamp(2072, 8, 24, 6, 17, 18), {5: 5, -7: -6, 0: 1}),
    ],
)
@pytest.mark.parametrize("n", [5, -7, 0])
def test_roll_qtrday_year2(other, expected, n):
    # [FUNCTION DOC]
    # Same structure as the previous test but anchored to June 30 (end day option, month 6, modulus 12), with a date before and after June 30 and n of 5, minus 7, 0.
    month = 6
    day_opt = "end"  # `other` will be compared to June 30.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


def test_get_day_of_month_error():
    # [FUNCTION DOC]
    # The day-of-month helper is only reachable through the quarter-day roll. With an invalid day option, positive n and an anchor month equal to the date's month, the roll raises a value error whose message contains the day option.
    # get_day_of_month is not directly exposed.
    # We test it via roll_qtrday.
    dt = datetime(2017, 11, 15)
    day_opt = "foo"

    with pytest.raises(ValueError, match=day_opt):
        # To hit the raising case we need month == dt.month and n > 0.
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)


@pytest.mark.parametrize("month", [3, 5])
# (other.month % 3) < (month % 3)  # (other.month % 3) > (month % 3)
@pytest.mark.parametrize("n", [4, -3])
def test_roll_qtr_day_not_mod_unequal(day_opt, month, n):
    # [FUNCTION DOC]
    # Parametrized over every day option, anchor months 3 and 5, and n of 4 and minus 3. For a Saturday timestamp in October with month modulus 3 where the date's month residue differs from the anchor's, the rolled n equals a table value (n unchanged or shifted by one depending on the residue relation).
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}

    other = Timestamp(2072, 10, 1, 6, 17, 18)  # Saturday.
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]


@pytest.mark.parametrize(
    "other,month,exp_dict",
    [
        # Monday.
        (datetime(1999, 5, 31), 2, {-1: {"start": 0, "business_start": 0}}),
        # Saturday.
        (
            Timestamp(2072, 10, 1, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1, "business_start": 1}},
        ),
        # First business day.
        (
            Timestamp(2072, 10, 3, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1}, -1: {"start": 0}},
        ),
    ],
)
@pytest.mark.parametrize("n", [2, -1])
def test_roll_qtr_day_mod_equal(other, month, exp_dict, n, day_opt):
    # [FUNCTION DOC]
    # Parametrized over three dates (a Monday May 31, a Saturday, and a first business day in October 2072), anchor months, per-day-option expected overrides, n of 2 and minus 1, and every day option. All cases have equal month residue modulo 3. The expected value is the override for the (n, day option) if present, else n itself; asserts the roll equals it.
    # All cases have (other.month % 3) == (month % 3).
    expected = exp_dict.get(n, {}).get(day_opt, n)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected


@pytest.mark.parametrize(
    "n,expected", [(42, {29: 42, 1: 42, 31: 41}), (-4, {29: -4, 1: -3, 31: -4})]
)
@pytest.mark.parametrize("compare", [29, 1, 31])
def test_roll_convention(n, expected, compare):
    # [FUNCTION DOC]
    # Parametrized over n (42, minus 4) and a comparison day (29, 1, 31). The roll convention helper called with day 29, n and the comparison day returns the expected adjusted count: for positive n it drops by one when the comparison day is later than 29; for negative n it rises by one when the comparison day is earlier.
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]
