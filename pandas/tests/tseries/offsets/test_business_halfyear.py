"""
Tests for the following offsets:
- BHalfYearBegin
- BHalfYearEnd
"""

# [MODULE DOC]
# Tests for the business half-year begin and business half-year end offsets: time-of-day preservation, agreement of the fast on-offset check with the add-then-subtract definition, repr, corner cases, application tables for several starting months and steps, and on-offset tables. Uses the shared offsets assertion helpers. No import-time side effects.
from __future__ import annotations

from datetime import datetime

import pytest

from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BHalfYearBegin,
    BHalfYearEnd,
)


@pytest.mark.parametrize("klass", (BHalfYearBegin, BHalfYearEnd))
def test_halfyearly_dont_normalize(klass):
    # [FUNCTION DOC]
    # Parametrized over both classes. Adding the default offset to 2012-03-31 05:30 preserves the time of day.
    date = datetime(2012, 3, 31, 5, 30)
    result = date + klass()
    assert result.time() == date.time()


@pytest.mark.parametrize("offset", [BHalfYearBegin(), BHalfYearEnd()])
@pytest.mark.parametrize(
    "date",
    [
        datetime(2016, m, d)
        for m in [7, 8, 9, 10, 11, 12]
        for d in [1, 2, 3, 28, 29, 30, 31]
        if not (m in {9, 11} and d == 31)
    ],
)
def test_on_offset(offset, date):
    # [FUNCTION DOC]
    # Parametrized over both default offsets and a set of dates in July through December 2016 (excluding the nonexistent Sept 31 and Nov 31). The direct on-offset answer must equal the slow definition: date equals (date plus offset) minus offset.
    res = offset.is_on_offset(date)
    slow_version = date == (date + offset) - offset
    assert res == slow_version


class TestBHalfYearBegin:
    # [CLASS DOC]
    # Test collection for the business half-year begin offset (default starting month 1).
    def test_repr(self):
        # [FUNCTION DOC]
        # Default prints with starting month 1, explicit starting month 3 prints with 3, explicit 1 prints with 1; class name shown is business half-year begin.
        expected = "<BusinessHalfYearBegin: startingMonth=1>"
        assert repr(BHalfYearBegin()) == expected
        expected = "<BusinessHalfYearBegin: startingMonth=3>"
        assert repr(BHalfYearBegin(startingMonth=3)) == expected
        expected = "<BusinessHalfYearBegin: startingMonth=1>"
        assert repr(BHalfYearBegin(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # [FUNCTION DOC]
        # A step of -1 with starting month 1 applied to 2010-02-01 gives 2010-01-01.
        # corner
        offset = BHalfYearBegin(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 1)

    offset_cases = []
    offset_cases.append(
        (
            BHalfYearBegin(startingMonth=1),
            {
                datetime(2007, 12, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 1): datetime(2008, 7, 1),
                datetime(2008, 2, 15): datetime(2008, 7, 1),
                datetime(2008, 2, 29): datetime(2008, 7, 1),
                datetime(2008, 3, 15): datetime(2008, 7, 1),
                datetime(2008, 3, 31): datetime(2008, 7, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 1): datetime(2008, 7, 1),
                datetime(2008, 7, 1): datetime(2009, 1, 1),
                datetime(2008, 7, 15): datetime(2009, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearBegin(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 2, 29): datetime(2008, 8, 1),
                datetime(2008, 3, 15): datetime(2008, 8, 1),
                datetime(2008, 3, 31): datetime(2008, 8, 1),
                datetime(2008, 4, 15): datetime(2008, 8, 1),
                datetime(2008, 4, 30): datetime(2008, 8, 1),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearBegin(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 12, 1): datetime(2009, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 7, 1),
                datetime(2008, 2, 29): datetime(2008, 7, 1),
                datetime(2008, 3, 15): datetime(2008, 7, 1),
                datetime(2008, 3, 31): datetime(2008, 7, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 30): datetime(2008, 7, 1),
                datetime(2008, 7, 1): datetime(2008, 7, 1),
                datetime(2008, 7, 15): datetime(2009, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearBegin(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 7, 2),
                datetime(2008, 1, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 1, 1),
                datetime(2008, 2, 29): datetime(2008, 1, 1),
                datetime(2008, 3, 15): datetime(2008, 1, 1),
                datetime(2008, 3, 31): datetime(2008, 1, 1),
                datetime(2008, 4, 15): datetime(2008, 1, 1),
                datetime(2008, 4, 30): datetime(2008, 1, 1),
                datetime(2008, 7, 1): datetime(2008, 1, 1),
                datetime(2008, 7, 15): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearBegin(startingMonth=1, n=2),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 2, 15): datetime(2009, 1, 1),
                datetime(2008, 2, 29): datetime(2009, 1, 1),
                datetime(2008, 3, 15): datetime(2009, 1, 1),
                datetime(2008, 3, 31): datetime(2009, 1, 1),
                datetime(2008, 4, 15): datetime(2009, 1, 1),
                datetime(2008, 4, 1): datetime(2009, 1, 1),
                datetime(2008, 7, 15): datetime(2009, 7, 1),
                datetime(2008, 7, 1): datetime(2009, 7, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases: starting months 1 and 2 with default step, step 0, -1 and 2; each base maps to the expected first business day of a half-year period.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BHalfYearBegin(1, startingMonth=1), datetime(2008, 1, 1), True),
        (BHalfYearBegin(1, startingMonth=1), datetime(2007, 12, 1), False),
        (BHalfYearBegin(1, startingMonth=1), datetime(2008, 2, 1), False),
        (BHalfYearBegin(1, startingMonth=1), datetime(2007, 3, 1), False),
        (BHalfYearBegin(1, startingMonth=1), datetime(2008, 4, 1), False),
        (BHalfYearBegin(1, startingMonth=1), datetime(2008, 5, 1), False),
        (BHalfYearBegin(1, startingMonth=1), datetime(2007, 6, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2008, 1, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2007, 12, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2008, 2, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2007, 3, 1), True),
        (BHalfYearBegin(1, startingMonth=3), datetime(2008, 4, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2008, 5, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2008, 5, 2), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2007, 6, 1), False),
        (BHalfYearBegin(1, startingMonth=3), datetime(2007, 6, 2), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2008, 1, 1), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2007, 12, 3), True),
        (BHalfYearBegin(1, startingMonth=6), datetime(2008, 2, 1), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2007, 3, 1), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2007, 3, 2), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2008, 4, 1), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2008, 5, 1), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2008, 5, 2), False),
        (BHalfYearBegin(1, startingMonth=6), datetime(2007, 6, 1), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases for starting months 1, 3 and 6: a date is on offset only when it is the first business day of a month in the half-year cycle for that starting month (for example month 6: 2007-12-03 and 2007-06-01).
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBHalfYearEnd:
    # [CLASS DOC]
    # Test collection for the business half-year end offset (default starting month 6).
    def test_repr(self):
        # [FUNCTION DOC]
        # Default prints with starting month 6, explicit 3 prints with 3, explicit 1 prints with 1; class name shown is business half-year end.
        expected = "<BusinessHalfYearEnd: startingMonth=6>"
        assert repr(BHalfYearEnd()) == expected
        expected = "<BusinessHalfYearEnd: startingMonth=3>"
        assert repr(BHalfYearEnd(startingMonth=3)) == expected
        expected = "<BusinessHalfYearEnd: startingMonth=1>"
        assert repr(BHalfYearEnd(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # [FUNCTION DOC]
        # A step of -1 with starting month 1 applied to 2010-01-30 gives 2010-01-29.
        # corner
        offset = BHalfYearEnd(n=-1, startingMonth=1)
        assert datetime(2010, 1, 30) + offset == datetime(2010, 1, 29)

    offset_cases = []
    offset_cases.append(
        (
            BHalfYearEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 7, 31): datetime(2009, 1, 30),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 8, 29),
                datetime(2008, 3, 15): datetime(2008, 8, 29),
                datetime(2008, 3, 31): datetime(2008, 8, 29),
                datetime(2008, 4, 15): datetime(2008, 8, 29),
                datetime(2008, 8, 28): datetime(2008, 8, 29),
                datetime(2008, 8, 29): datetime(2009, 2, 27),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearEnd(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 7, 31): datetime(2008, 7, 31),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearEnd(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 7, 31),
                datetime(2008, 1, 31): datetime(2007, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 1, 31),
                datetime(2008, 2, 29): datetime(2008, 1, 31),
                datetime(2008, 3, 15): datetime(2008, 1, 31),
                datetime(2008, 3, 31): datetime(2008, 1, 31),
                datetime(2008, 7, 15): datetime(2008, 1, 31),
                datetime(2008, 7, 30): datetime(2008, 1, 31),
                datetime(2008, 7, 31): datetime(2008, 1, 31),
                datetime(2008, 8, 1): datetime(2008, 7, 31),
            },
        )
    )

    offset_cases.append(
        (
            BHalfYearEnd(startingMonth=6, n=2),
            {
                datetime(2008, 1, 31): datetime(2008, 12, 31),
                datetime(2008, 2, 15): datetime(2008, 12, 31),
                datetime(2008, 2, 29): datetime(2008, 12, 31),
                datetime(2008, 3, 15): datetime(2008, 12, 31),
                datetime(2008, 3, 31): datetime(2008, 12, 31),
                datetime(2008, 4, 15): datetime(2008, 12, 31),
                datetime(2008, 4, 30): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2009, 6, 30),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases: starting months 1, 2, 1 with step 0, 1 with step -1, and 6 with step 2; each base maps to the expected last business day of a half-year period.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BHalfYearEnd(1, startingMonth=1), datetime(2008, 1, 31), True),
        (BHalfYearEnd(1, startingMonth=1), datetime(2007, 12, 31), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2008, 2, 29), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2007, 3, 30), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2007, 3, 31), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2008, 4, 30), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2008, 5, 30), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2008, 5, 31), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2007, 6, 29), False),
        (BHalfYearEnd(1, startingMonth=1), datetime(2007, 6, 30), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2008, 1, 31), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2007, 12, 31), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2008, 2, 29), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2007, 3, 30), True),
        (BHalfYearEnd(1, startingMonth=3), datetime(2007, 3, 31), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2008, 4, 30), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2008, 5, 30), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2008, 5, 31), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2007, 6, 29), False),
        (BHalfYearEnd(1, startingMonth=3), datetime(2007, 6, 30), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2008, 1, 31), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2007, 12, 31), True),
        (BHalfYearEnd(1, startingMonth=6), datetime(2008, 2, 29), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2007, 3, 30), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2007, 3, 31), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2008, 4, 30), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2008, 5, 30), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2008, 5, 31), False),
        (BHalfYearEnd(1, startingMonth=6), datetime(2007, 6, 29), True),
        (BHalfYearEnd(1, startingMonth=6), datetime(2007, 6, 30), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases for starting months 1, 3 and 6: a date is on offset only when it is the last business day of a month in the half-year cycle for that starting month (for example month 6: 2007-12-31 and 2007-06-29).
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
