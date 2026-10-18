"""
Tests for offsets.CustomBusinessHour
"""

# [MODULE DOC]
# Tests for the custom business hour offset (CustomBusinessHour): constructor validation, equality, repr, rolling, normalization, on-offset checks, application tables for several steps and times including nanosecond fractions, US federal calendar holidays and week masks combined with holidays.
# Defines a module-level list of holiday values (string, datetime, NumPy datetime64) around the end of June 2014 used by fixtures and tables, plus a comment showing the surrounding 2014 calendar. Uses the shared offsets assertion helper. No import-time side effects.
from __future__ import annotations

from datetime import (
    datetime,
    time as dt_time,
)

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    BusinessHour,
    CustomBusinessHour,
    Nano,
)

from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.holiday import USFederalHolidayCalendar

holidays = ["2014-06-27", datetime(2014, 6, 30), np.datetime64("2014-07-02")]


@pytest.fixture
def dt():
    # [FUNCTION DOC]
    # HELPER: fixture returning the base datetime 2014-07-01 10:00 (a Tuesday).
    return datetime(2014, 7, 1, 10, 00)


@pytest.fixture
def _offset():
    # [FUNCTION DOC]
    # HELPER: fixture returning the custom business hour class (not an instance).
    return CustomBusinessHour


# 2014 Calendar to check custom holidays
#   Sun Mon Tue Wed Thu Fri Sat
#  6/22  23  24  25  26  27  28
#    29  30 7/1   2   3   4   5
#     6   7   8   9  10  11  12
@pytest.fixture
def offset1():
    # [FUNCTION DOC]
    # HELPER: fixture returning a custom business hour offset whose week mask is Tuesday through Friday (Monday not a business day).
    return CustomBusinessHour(weekmask="Tue Wed Thu Fri")


@pytest.fixture
def offset2():
    # [FUNCTION DOC]
    # HELPER: fixture returning a custom business hour offset with the module's holiday list (June 27, June 30 and July 2, 2014).
    return CustomBusinessHour(holidays=holidays)


class TestCustomBusinessHour:
    # [CLASS DOC]
    # Test collection for the custom business hour offset (default hours 09:00 to 17:00); several case tables are class attributes.
    def test_constructor_errors(self):
        # [FUNCTION DOC]
        # Constructor validation of start time: a time object with seconds raises a value error "time data must be specified only with hour and minute"; an unparseable string and a string with seconds raise a value error "time data must match '%H:%M' format".
        msg = "time data must be specified only with hour and minute"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start=dt_time(11, 0, 5))
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start="AAA")
        msg = "time data must match '%H:%M' format"
        with pytest.raises(ValueError, match=msg):
            CustomBusinessHour(start="14:00:05")

    def test_different_normalize_equals(self, _offset):
        # [FUNCTION DOC]
        # GH#21404. An instance and an instance with normalization enabled compare unequal.
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(self, offset1, offset2):
        # [FUNCTION DOC]
        # Both the week-mask and holiday instances print as a custom business hour with hours 09:00-17:00 (week mask and holidays are not shown).
        assert repr(offset1) == "<CustomBusinessHour: cbh=09:00-17:00>"
        assert repr(offset2) == "<CustomBusinessHour: cbh=09:00-17:00>"

    def test_with_offset(self, dt):
        # [FUNCTION DOC]
        # Adding three hours (by multiplying a single hour offset by 3, or by constructing with step 3) to 2014-07-01 10:00 gives 2014-07-01 13:00.
        expected = Timestamp("2014-07-01 13:00")

        assert dt + CustomBusinessHour() * 3 == expected
        assert dt + CustomBusinessHour(n=3) == expected

    def test_eq(self, offset1, offset2):
        # [FUNCTION DOC]
        # Equality checks: both fixtures equal themselves; offsets differ when the step, start/end times, week mask or holidays differ; explicitly specifying the default start equals the default offset.
        for offset in [offset1, offset2]:
            assert offset == offset

        assert CustomBusinessHour() != CustomBusinessHour(-1)
        assert CustomBusinessHour(start="09:00") == CustomBusinessHour()
        assert CustomBusinessHour(start="09:00") != CustomBusinessHour(start="09:01")
        assert CustomBusinessHour(start="09:00", end="17:00") != CustomBusinessHour(
            start="17:00", end="09:01"
        )

        assert CustomBusinessHour(weekmask="Tue Wed Thu Fri") != CustomBusinessHour(
            weekmask="Mon Tue Wed Thu Fri"
        )
        assert CustomBusinessHour(holidays=["2014-06-27"]) != CustomBusinessHour(
            holidays=["2014-06-28"]
        )

    def test_hash(self, offset1, offset2):
        # [FUNCTION DOC]
        # Hash of each fixture instance is stable across calls.
        assert hash(offset1) == hash(offset1)
        assert hash(offset2) == hash(offset2)

    def test_add_dateime(self, dt, offset1, offset2):
        # [FUNCTION DOC]
        # Adding either fixture to 2014-07-01 10:00 gives 11:00 the same day.
        # KNOWN ISSUE: the test name is misspelled (dateime); it is the datetime addition test.
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 11)

    def testRollback1(self, dt, offset1, offset2):
        # [FUNCTION DOC]
        # Rolling back the in-hours base date returns it unchanged for both fixtures. From midnight on Tuesday July 1 2014, the week-mask offset rolls back to 2014-06-27 17:00 (Monday not a business day, so Friday close) and the holiday offset rolls back to 2014-06-26 17:00 (June 30 and June 27 are holidays).
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt

        d = datetime(2014, 7, 1, 0)

        # 2014/07/01 is Tuesday, 06/30 is Monday(holiday)
        assert offset1.rollback(d) == datetime(2014, 6, 27, 17)

        # 2014/6/30 and 2014/6/27 are holidays
        assert offset2.rollback(d) == datetime(2014, 6, 26, 17)

    def testRollback2(self, _offset):
        # [FUNCTION DOC]
        # A step -3 offset rolling back Saturday 2014-07-05 15:00 gives Friday 2014-07-04 17:00.
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(self, dt, offset1, offset2):
        # [FUNCTION DOC]
        # Rolling forward the in-hours base returns it unchanged for both fixtures; from midnight July 1 2014 both roll to 09:00 that day.
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt

        d = datetime(2014, 7, 1, 0)
        assert offset1.rollforward(d) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(d) == datetime(2014, 7, 1, 9)

    def testRollforward2(self, _offset):
        # [FUNCTION DOC]
        # A step -3 offset rolling forward Saturday 2014-07-05 16:00 gives Monday 2014-07-07 09:00.
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    def test_roll_date_object(self):
        # [FUNCTION DOC]
        # Uses the plain business hour offset rather than the custom one. Sunday 2014-07-06 15:00 rolls back to Friday 2014-07-04 17:00 and forward to Monday 2014-07-07 09:00.
        # KNOWN ISSUE: despite the name, no date object is used (a datetime is used) and the offset is the non-custom business hour class, not the class under test.
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    normalize_cases = [
        (
            CustomBusinessHour(normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 3),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        ),
        (
            CustomBusinessHour(-1, normalize=True, holidays=holidays),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 26),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 26),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        ),
        (
            CustomBusinessHour(
                1, normalize=True, start="17:00", end="04:00", holidays=holidays
            ),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 3),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        ),
    ]

    @pytest.mark.parametrize("norm_cases", normalize_cases)
    def test_normalize(self, norm_cases):
        # [FUNCTION DOC]
        # Parametrized over normalize_cases (three offsets with normalization enabled: forward step 1, step -1, and overnight hours 17:00-04:00, all with holidays). Applying the offset directly to each base datetime gives the expected midnight date.
        offset, cases = norm_cases
        for dt, expected in cases.items():
            assert offset._apply(dt) == expected

    @pytest.mark.parametrize(
        "dt, expected",
        [
            [datetime(2014, 7, 1, 9), False],
            [datetime(2014, 7, 1, 10), True],
            [datetime(2014, 7, 1, 15), True],
            [datetime(2014, 7, 1, 15, 1), False],
            [datetime(2014, 7, 5, 12), False],
            [datetime(2014, 7, 6, 12), False],
        ],
    )
    def test_is_on_offset(self, dt, expected):
        # [FUNCTION DOC]
        # Parametrized over (datetime, expected). With hours 10:00-15:00 and the holiday list: 09:00 is off, 10:00 and 15:00 on, 15:01 off, weekend days off.
        offset = CustomBusinessHour(start="10:00", end="15:00", holidays=holidays)
        assert offset.is_on_offset(dt) == expected

    apply_cases = [
        (
            CustomBusinessHour(holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 3, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 10),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            CustomBusinessHour(4, holidays=holidays),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 3, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 3, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 3, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
    ]

    @pytest.mark.parametrize("apply_case", apply_cases)
    def test_apply(self, apply_case):
        # [FUNCTION DOC]
        # Parametrized over apply_cases for step 1 and step 4 with holidays; checks times inside hours, after hours, before hours, holiday skipping, and weekends, including fractional seconds, using the assertion helper.
        offset, cases = apply_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    nano_cases = [
        (
            CustomBusinessHour(holidays=holidays),
            {
                Timestamp("2014-07-01 15:00") + Nano(5): Timestamp("2014-07-01 16:00")
                + Nano(5),
                Timestamp("2014-07-01 16:00") + Nano(5): Timestamp("2014-07-03 09:00")
                + Nano(5),
                Timestamp("2014-07-01 16:00") - Nano(5): Timestamp("2014-07-01 17:00")
                - Nano(5),
            },
        ),
        (
            CustomBusinessHour(-1, holidays=holidays),
            {
                Timestamp("2014-07-01 15:00") + Nano(5): Timestamp("2014-07-01 14:00")
                + Nano(5),
                Timestamp("2014-07-01 10:00") + Nano(5): Timestamp("2014-07-01 09:00")
                + Nano(5),
                Timestamp("2014-07-01 10:00") - Nano(5): Timestamp("2014-06-26 17:00")
                - Nano(5),
            },
        ),
    ]

    @pytest.mark.parametrize("nano_case", nano_cases)
    def test_apply_nanoseconds(self, nano_case):
        # [FUNCTION DOC]
        # Parametrized over nano_cases for steps 1 and -1 with holidays. Bases carry a 5 nanosecond offset (added or subtracted); the expected results carry the same nanosecond component.
        offset, cases = nano_case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_us_federal_holiday_with_datetime(self):
        # [FUNCTION DOC]
        # GH 16867. Offset using the US federal holiday calendar: eight business hours added to Friday 2014-01-17 15:00 gives Tuesday 2014-01-21 15:00 (skips the weekend and the holiday on the 20th).
        # GH 16867
        bhour_us = CustomBusinessHour(calendar=USFederalHolidayCalendar())
        t0 = datetime(2014, 1, 17, 15)
        result = t0 + bhour_us * 8
        expected = Timestamp("2014-01-21 15:00:00")
        assert result == expected


@pytest.mark.parametrize(
    "weekmask, expected_time, mult",
    [
        ["Mon Tue Wed Thu Fri Sat", "2018-11-10 09:00:00", 10],
        ["Tue Wed Thu Fri Sat", "2018-11-13 08:00:00", 18],
    ],
)
def test_custom_businesshour_weekmask_and_holidays(weekmask, expected_time, mult):
    # [FUNCTION DOC]
    # GH 23542. Parametrized over (weekmask, expected time, multiplier). With hours 08:00-17:00 and a holiday on 2018-11-09, adding the multiplier times the offset to 2018-11-08 08:00 gives the expected timestamp (Monday-Saturday work week, 10 hours later is 2018-11-10 09:00; Tuesday-Saturday, 18 hours later is 2018-11-13 08:00).
    # GH 23542
    holidays = ["2018-11-09"]
    bh = CustomBusinessHour(
        start="08:00", end="17:00", weekmask=weekmask, holidays=holidays
    )
    result = Timestamp("2018-11-08 08:00") + mult * bh
    expected = Timestamp(expected_time)
    assert result == expected
