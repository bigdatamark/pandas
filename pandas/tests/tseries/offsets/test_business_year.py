"""
Tests for the following offsets:
- BYearBegin
- BYearEnd
"""

# [MODULE DOC]
# Tests for the business year begin and business year end offsets: argument validation of the month, application tables for steps -2 to 1 and zero, lagged (non-December) fiscal year ends, rolling, and on-offset checks. Uses the shared offsets assertion helpers. No import-time side effects.
from __future__ import annotations

from datetime import datetime

import pytest

from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BYearBegin,
    BYearEnd,
)


class TestBYearBegin:
    # [CLASS DOC]
    # Test collection for the business year begin offset; the case table is a class attribute.
    def test_misspecified(self):
        # [FUNCTION DOC]
        # A month outside the range 1 to 12 (13) raises a value error "Month must go from 1 to 12" for both the begin and end classes.
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearBegin(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)

    offset_cases = []
    offset_cases.append(
        (
            BYearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2011, 1, 1): datetime(2011, 1, 3),
                datetime(2011, 1, 3): datetime(2012, 1, 2),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 2),
                datetime(2009, 1, 4): datetime(2009, 1, 1),
                datetime(2009, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 2),
                datetime(2006, 12, 30): datetime(2006, 1, 2),
                datetime(2006, 1, 1): datetime(2005, 1, 3),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 3),
                datetime(2007, 6, 30): datetime(2006, 1, 2),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases for steps 1, 0, -1 and -2; checks each base date maps to the expected first business day of a year (for example 2011 starts on Monday January 3).
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestBYearEnd:
    # [CLASS DOC]
    # Test collection for the business year end offset (default December anchor); case tables are class attributes.
    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2006, 12, 29),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 30),
                datetime(2008, 6, 30): datetime(2006, 12, 29),
                datetime(2008, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases for steps 1, 0, -1, -2; each base maps to the expected last business day of a year (for example 2006's is Friday December 29).
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BYearEnd(), datetime(2007, 12, 31), True),
        (BYearEnd(), datetime(2008, 1, 1), False),
        (BYearEnd(), datetime(2006, 12, 31), False),
        (BYearEnd(), datetime(2006, 12, 29), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases: last business days (2007-12-31, 2006-12-29) are on offset; 2008-01-01 and the Sunday 2006-12-31 are not.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBYearEndLagged:
    # [CLASS DOC]
    # Test collection for the business year end offset anchored on a month other than December.
    def test_bad_month_fail(self):
        # [FUNCTION DOC]
        # Months 13 and 0 raise a value error "Month must go from 1 to 12".
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=0)

    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(month=6),
            {
                datetime(2008, 1, 1): datetime(2008, 6, 30),
                datetime(2007, 6, 30): datetime(2008, 6, 30),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(n=-1, month=6),
            {
                datetime(2008, 1, 1): datetime(2007, 6, 29),
                datetime(2007, 6, 30): datetime(2007, 6, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases: June-anchored forward (2008-01-01 and 2007-06-30 give 2008-06-30) and backward with step -1 (both give 2007-06-29, the last business day of June 2007).
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_roll(self):
        # [FUNCTION DOC]
        # For the June-anchored offset and 2009-11-30, rolling forward gives 2010-06-30 and rolling back gives 2009-06-30.
        offset = BYearEnd(month=6)
        date = datetime(2009, 11, 30)

        assert offset.rollforward(date) == datetime(2010, 6, 30)
        assert offset.rollback(date) == datetime(2009, 6, 30)

    on_offset_cases = [
        (BYearEnd(month=2), datetime(2007, 2, 28), True),
        (BYearEnd(month=6), datetime(2007, 6, 30), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized: February-anchored 2007-02-28 is on offset; June-anchored 2007-06-30 (a Saturday) is not.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
