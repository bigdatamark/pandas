"""
Assertion helpers and base class for offsets tests
"""

# [MODULE DOC]
# HELPER: shared test support for the offsets test modules.
# Provides two assertion helpers (checking offset arithmetic and on-offset
# membership) and a small constants class of weekday numbers. Used by the
# business-day/hour/month/quarter/year, custom-business and Easter test modules.
# No import-time side effects.
from __future__ import annotations


def assert_offset_equal(offset, base, expected):
    # [FUNCTION DOC]
    # HELPER: assertion helper for offset arithmetic tests.
    # Parameters: offset is a date offset; base is a timestamp or datetime;
    # expected is the timestamp the addition should produce.
    # Adds the offset to the base on the right, adds the base to the offset
    # (swapped operand order), and calls the offset's internal apply method
    # directly; all three results must equal expected.
    # Raises an assertion error with a message showing expected, actual, the
    # offset and the base date when any comparison fails (the message reports
    # only the plain right-hand addition result).
    # Returns nothing.
    actual = offset + base
    actual_swapped = base + offset
    actual_apply = offset._apply(base)
    try:
        assert actual == expected
        assert actual_swapped == expected
        assert actual_apply == expected
    except AssertionError as err:
        raise AssertionError(
            f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
            f"\nAt Date: {base}"
        ) from err


def assert_is_on_offset(offset, date, expected):
    # [FUNCTION DOC]
    # HELPER: assertion helper for on-offset membership tests.
    # Parameters: offset is a date offset; date is the timestamp to test;
    # expected is the boolean the offset's is-on-offset check should return.
    # Raises an assertion error with a descriptive message on mismatch.
    # Returns nothing.
    actual = offset.is_on_offset(date)
    assert actual == expected, (
        f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
        f"\nAt Date: {date}"
    )


class WeekDay:
    # [CLASS DOC]
    # HELPER: namespace of integer constants for weekdays, Monday is 0
    # through Sunday is 6, matching the numbering used by datetime weekday.
    # Plain class with only class attributes; never instantiated.
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6
