# [MODULE DOC]
# Tests for the observance helper functions that move a holiday date relative to the weekend
# (next Monday, previous Friday, nearest workday, and similar).
# Defines module-level constants for consecutive dates in April 2014, from Wednesday the 9th to the following Wednesday the 16th,
# used as inputs and expected outputs. No import-time side effects.
from datetime import datetime

import pytest

from pandas.tseries.holiday import (
    after_nearest_workday,
    before_nearest_workday,
    nearest_workday,
    next_monday,
    next_monday_or_tuesday,
    next_workday,
    previous_friday,
    previous_workday,
    sunday_to_monday,
    weekend_to_monday,
)

_WEDNESDAY = datetime(2014, 4, 9)
_THURSDAY = datetime(2014, 4, 10)
_FRIDAY = datetime(2014, 4, 11)
_SATURDAY = datetime(2014, 4, 12)
_SUNDAY = datetime(2014, 4, 13)
_MONDAY = datetime(2014, 4, 14)
_TUESDAY = datetime(2014, 4, 15)
_NEXT_WEDNESDAY = datetime(2014, 4, 16)


@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_next_monday(day):
    # [FUNCTION DOC]
    # Parametrized over Saturday and Sunday. The next-Monday observance returns Monday for both.
    assert next_monday(day) == _MONDAY


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_MONDAY, _TUESDAY)]
)
def test_next_monday_or_tuesday(day, expected):
    # [FUNCTION DOC]
    # Parametrized over (day, expected): Saturday maps to Monday, Sunday to Tuesday, Monday to Tuesday.
    assert next_monday_or_tuesday(day) == expected


@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_previous_friday(day):
    # [FUNCTION DOC]
    # Parametrized over Saturday and Sunday. The previous-Friday observance returns the Friday before.
    assert previous_friday(day) == _FRIDAY


def test_sunday_to_monday():
    # [FUNCTION DOC]
    # A Sunday maps to the following Monday.
    assert sunday_to_monday(_SUNDAY) == _MONDAY


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _FRIDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)]
)
def test_nearest_workday(day, expected):
    # [FUNCTION DOC]
    # Parametrized: Saturday goes back to Friday, Sunday goes forward to Monday, Monday stays Monday.
    assert nearest_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)]
)
def test_weekend_to_monday(day, expected):
    # [FUNCTION DOC]
    # Parametrized: Saturday and Sunday both map to Monday; Monday is unchanged.
    assert weekend_to_monday(day) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        (_WEDNESDAY, _THURSDAY),
        (_THURSDAY, _FRIDAY),
        (_SATURDAY, _MONDAY),
        (_SUNDAY, _MONDAY),
        (_MONDAY, _TUESDAY),
        (_TUESDAY, _NEXT_WEDNESDAY),  # WED is same week as TUE
    ],
)
def test_next_workday(day, expected):
    # [FUNCTION DOC]
    # Parametrized over weekdays and weekend days: Wednesday goes to Thursday, Thursday to Friday, Saturday
    # and Sunday to Monday, Monday to Tuesday, Tuesday to the next Wednesday.
    assert next_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _FRIDAY), (_SUNDAY, _FRIDAY), (_TUESDAY, _MONDAY)]
)
def test_previous_workday(day, expected):
    # [FUNCTION DOC]
    # Parametrized: Saturday and Sunday map to Friday; Tuesday maps to Monday.
    assert previous_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        (_THURSDAY, _WEDNESDAY),
        (_FRIDAY, _THURSDAY),
        (_SATURDAY, _THURSDAY),
        (_SUNDAY, _FRIDAY),
        (_MONDAY, _FRIDAY),  # last week Friday
        (_TUESDAY, _MONDAY),
        (_NEXT_WEDNESDAY, _TUESDAY),  # WED is same week as TUE
    ],
)
def test_before_nearest_workday(day, expected):
    # [FUNCTION DOC]
    # Parametrized: the observance returns the workday before the nearest workday: Thursday to Wednesday, Friday to Thursday,
    # Saturday to Thursday, Sunday to Friday, Monday to the previous Friday, Tuesday to Monday, next Wednesday to Tuesday.
    assert before_nearest_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_FRIDAY, _MONDAY)]
)
def test_after_nearest_workday(day, expected):
    # [FUNCTION DOC]
    # Parametrized: the observance returns the workday after the nearest workday: Saturday to Monday, Sunday to Tuesday, Friday to Monday.
    assert after_nearest_workday(day) == expected
