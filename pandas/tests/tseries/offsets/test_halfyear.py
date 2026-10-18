"""
Tests for the following offsets:
- HalfYearBegin
- HalfYearEnd
"""

# [MODULE DOC]
# Tests for the half-year offsets (beginning-of-half-year and end-of-half-year anchored offsets).
# Covers non-normalization of time of day, agreement of the fast on-offset check with the slow
# definition, string representation, negative-offset corner cases, and table-driven application
# of offsets with different starting months and counts.
# Uses shared assertion helpers from the offsets common test module; no import-time side effects
# other than building case tables on the test classes.

from __future__ import annotations

from datetime import datetime

import pytest

from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    HalfYearBegin,
    HalfYearEnd,
)


@pytest.mark.parametrize("klass", (HalfYearBegin, HalfYearEnd))
def test_halfyearly_dont_normalize(klass):
    # [FUNCTION DOC]
    # Parametrized over both half-year offset classes. Adds a default-constructed offset to a datetime with
    # a non-midnight time and asserts the time of day of the result is unchanged (no normalization).
    date = datetime(2012, 3, 31, 5, 30)
    result = date + klass()
    assert result.time() == date.time()


@pytest.mark.parametrize("offset", [HalfYearBegin(), HalfYearEnd()])
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
    # Parametrized over a begin and an end offset and over many 2016 dates (months July to December, a
    # range of days, excluding the nonexistent 31st of September and November). Asserts that the fast
    # on-offset check equals the slow definition: the date equals itself after adding then subtracting the offset.
    res = offset.is_on_offset(date)
    slow_version = date == (date + offset) - offset
    assert res == slow_version


class TestHalfYearBegin:
    # [CLASS DOC]
    # Tests for the half-year-begin offset. Class attributes hold the table of offset application cases
    # (each a pair of an offset and a mapping from base date to expected result) and the on-offset table.
    def test_repr(self):
        # [FUNCTION DOC]
        # Asserts the representation shows the starting month, which defaults to 1 for this offset, and
        # reflects explicit starting months of 3 and 1.
        expected = "<HalfYearBegin: startingMonth=1>"
        assert repr(HalfYearBegin()) == expected
        expected = "<HalfYearBegin: startingMonth=3>"
        assert repr(HalfYearBegin(startingMonth=3)) == expected
        expected = "<HalfYearBegin: startingMonth=1>"
        assert repr(HalfYearBegin(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # [FUNCTION DOC]
        # Asserts that a negative one offset with starting month 1 applied to the first of February 2010 gives the first of January 2010.
        # corner
        offset = HalfYearBegin(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 1)

    offset_cases = []
    offset_cases.append(
        (
            HalfYearBegin(startingMonth=1),
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
            HalfYearBegin(startingMonth=2),
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
            HalfYearBegin(startingMonth=1, n=0),
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
            HalfYearBegin(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 7, 1),
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
            HalfYearBegin(startingMonth=1, n=2),
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
        # Parametrized over the offset case table (starting months 1 and 2, n of 0, minus one and two). For
        # each base date in the mapping asserts via the shared helper that applying the offset gives the expected date.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (HalfYearBegin(1, startingMonth=1), datetime(2008, 1, 1), True),
        (HalfYearBegin(1, startingMonth=1), datetime(2007, 12, 1), False),
        (HalfYearBegin(1, startingMonth=1), datetime(2008, 2, 1), False),
        (HalfYearBegin(1, startingMonth=1), datetime(2007, 3, 1), False),
        (HalfYearBegin(1, startingMonth=1), datetime(2008, 4, 1), False),
        (HalfYearBegin(1, startingMonth=1), datetime(2008, 5, 1), False),
        (HalfYearBegin(1, startingMonth=1), datetime(2007, 6, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2008, 1, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2007, 12, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2008, 2, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2007, 3, 1), True),
        (HalfYearBegin(1, startingMonth=3), datetime(2008, 4, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2008, 5, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2008, 5, 2), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2007, 6, 1), False),
        (HalfYearBegin(1, startingMonth=3), datetime(2007, 6, 2), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2008, 1, 1), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2007, 12, 1), True),
        (HalfYearBegin(1, startingMonth=6), datetime(2008, 2, 1), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2007, 3, 1), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2007, 3, 2), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2008, 4, 1), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2008, 5, 1), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2008, 5, 2), False),
        (HalfYearBegin(1, startingMonth=6), datetime(2007, 6, 1), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the on-offset table of (offset, date, expected); asserts the on-offset check matches expected.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestHalfYearEnd:
    # [CLASS DOC]
    # Tests for the half-year-end offset. Class attributes hold the offset application case table and the on-offset table.
    def test_repr(self):
        # [FUNCTION DOC]
        # Asserts the representation shows the starting month, which defaults to 6 for this offset, and
        # reflects explicit starting months of 3 and 1.
        expected = "<HalfYearEnd: startingMonth=6>"
        assert repr(HalfYearEnd()) == expected
        expected = "<HalfYearEnd: startingMonth=3>"
        assert repr(HalfYearEnd(startingMonth=3)) == expected
        expected = "<HalfYearEnd: startingMonth=1>"
        assert repr(HalfYearEnd(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # [FUNCTION DOC]
        # Asserts that a negative one offset with starting month 1 applied to the first of February 2010 gives the end of January 2010.
        # corner
        offset = HalfYearEnd(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 31)

    offset_cases = []
    offset_cases.append(
        (
            HalfYearEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 7, 31): datetime(2009, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            HalfYearEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 8, 31),
                datetime(2008, 3, 15): datetime(2008, 8, 31),
                datetime(2008, 3, 31): datetime(2008, 8, 31),
                datetime(2008, 4, 15): datetime(2008, 8, 31),
                datetime(2008, 8, 30): datetime(2008, 8, 31),
                datetime(2008, 8, 31): datetime(2009, 2, 28),
            },
        )
    )

    offset_cases.append(
        (
            HalfYearEnd(startingMonth=1, n=0),
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
            HalfYearEnd(startingMonth=1, n=-1),
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
            HalfYearEnd(startingMonth=6, n=2),
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
        # Parametrized over the offset case table (starting months 1, 2 and 6 with various n); for each base
        # date asserts the offset gives the expected month-end date.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (HalfYearEnd(1, startingMonth=1), datetime(2008, 1, 31), True),
        (HalfYearEnd(1, startingMonth=1), datetime(2007, 12, 31), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2008, 2, 29), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2007, 3, 30), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2007, 3, 31), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2008, 4, 30), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2008, 5, 30), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2008, 5, 31), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2007, 6, 29), False),
        (HalfYearEnd(1, startingMonth=1), datetime(2007, 6, 30), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2008, 1, 31), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2007, 12, 31), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2008, 2, 29), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2007, 3, 30), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2007, 3, 31), True),
        (HalfYearEnd(1, startingMonth=3), datetime(2008, 4, 30), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2008, 5, 30), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2008, 5, 31), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2007, 6, 29), False),
        (HalfYearEnd(1, startingMonth=3), datetime(2007, 6, 30), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2008, 1, 31), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2007, 12, 31), True),
        (HalfYearEnd(1, startingMonth=6), datetime(2008, 2, 29), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2007, 3, 30), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2007, 3, 31), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2008, 4, 30), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2008, 5, 30), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2008, 5, 31), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2007, 6, 29), False),
        (HalfYearEnd(1, startingMonth=6), datetime(2007, 6, 30), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the on-offset table of (offset, date, expected); asserts the on-offset check matches expected.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
