"""
Tests for the following offsets:
- QuarterBegin
- QuarterEnd
"""

# [MODULE DOC]
# Tests for the quarter offsets (quarter begin and quarter end): non-normalization of time of day, agreement of the fast
# on-offset check with the slow definition, representation, negative-offset corner cases, and table-driven application of
# offsets with varied starting months and counts. Uses shared helpers from the offsets common test module; no import-time side effects
# besides building case tables on the test classes.

from __future__ import annotations

from datetime import datetime

import pytest

from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    QuarterBegin,
    QuarterEnd,
)


@pytest.mark.parametrize("klass", (QuarterBegin, QuarterEnd))
def test_quarterly_dont_normalize(klass):
    # [FUNCTION DOC]
    # Parametrized over both quarter offset classes: adding a default offset to a datetime with a non-midnight time preserves the time of day.
    date = datetime(2012, 3, 31, 5, 30)
    result = date + klass()
    assert result.time() == date.time()


@pytest.mark.parametrize("offset", [QuarterBegin(), QuarterEnd()])
@pytest.mark.parametrize(
    "date",
    [
        datetime(2016, m, d)
        for m in [10, 11, 12]
        for d in [1, 2, 3, 28, 29, 30, 31]
        if not (m == 11 and d == 31)
    ],
)
def test_on_offset(offset, date):
    # [FUNCTION DOC]
    # Parametrized over a begin and an end offset and many dates in October to December 2016 (excluding November 31). Asserts the fast
    # on-offset check equals the slow definition: the date equals itself after adding and then subtracting the offset.
    res = offset.is_on_offset(date)
    slow_version = date == (date + offset) - offset
    assert res == slow_version


class TestQuarterBegin:
    # [CLASS DOC]
    # Tests for the quarter-begin offset. Class attribute holds the offset application case table.
    def test_repr(self):
        # [FUNCTION DOC]
        # The representation shows the starting month, defaulting to 3, and reflects explicit values 3 and 1.
        expected = "<QuarterBegin: startingMonth=3>"
        assert repr(QuarterBegin()) == expected
        expected = "<QuarterBegin: startingMonth=3>"
        assert repr(QuarterBegin(startingMonth=3)) == expected
        expected = "<QuarterBegin: startingMonth=1>"
        assert repr(QuarterBegin(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # [FUNCTION DOC]
        # A negative one offset with starting month 1 applied to February 1 2010 gives January 1 2010.
        # corner
        offset = QuarterBegin(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 1)

    offset_cases = []
    offset_cases.append(
        (
            QuarterBegin(startingMonth=1),
            {
                datetime(2007, 12, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 1): datetime(2008, 4, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 1): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 2, 29): datetime(2008, 5, 1),
                datetime(2008, 3, 15): datetime(2008, 5, 1),
                datetime(2008, 3, 31): datetime(2008, 5, 1),
                datetime(2008, 4, 15): datetime(2008, 5, 1),
                datetime(2008, 4, 30): datetime(2008, 5, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 12, 1): datetime(2009, 1, 1),
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 4, 1),
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 3, 15): datetime(2008, 4, 1),
                datetime(2008, 3, 31): datetime(2008, 4, 1),
                datetime(2008, 4, 15): datetime(2008, 7, 1),
                datetime(2008, 4, 30): datetime(2008, 7, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 1),
                datetime(2008, 1, 31): datetime(2008, 1, 1),
                datetime(2008, 2, 15): datetime(2008, 1, 1),
                datetime(2008, 2, 29): datetime(2008, 1, 1),
                datetime(2008, 3, 15): datetime(2008, 1, 1),
                datetime(2008, 3, 31): datetime(2008, 1, 1),
                datetime(2008, 4, 15): datetime(2008, 4, 1),
                datetime(2008, 4, 30): datetime(2008, 4, 1),
                datetime(2008, 7, 1): datetime(2008, 4, 1),
            },
        )
    )

    offset_cases.append(
        (
            QuarterBegin(startingMonth=1, n=2),
            {
                datetime(2008, 1, 1): datetime(2008, 7, 1),
                datetime(2008, 2, 15): datetime(2008, 7, 1),
                datetime(2008, 2, 29): datetime(2008, 7, 1),
                datetime(2008, 3, 15): datetime(2008, 7, 1),
                datetime(2008, 3, 31): datetime(2008, 7, 1),
                datetime(2008, 4, 15): datetime(2008, 10, 1),
                datetime(2008, 4, 1): datetime(2008, 10, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the case table (starting months 1 and 2; n of 0, minus one, two). For each base date asserts the offset gives the expected date.
        # KNOWN ISSUE: the case table for starting month 1 with n of 0 lists the key for January 1 2008 twice in the same mapping, so the duplicate entry is redundant (the later one silently overwrites the earlier).
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestQuarterEnd:
    # [CLASS DOC]
    # Tests for the quarter-end offset. Class attributes hold the offset application case table and the on-offset table.
    def test_repr(self):
        # [FUNCTION DOC]
        # The representation shows the starting month, defaulting to 3, and reflects explicit values 3 and 1.
        expected = "<QuarterEnd: startingMonth=3>"
        assert repr(QuarterEnd()) == expected
        expected = "<QuarterEnd: startingMonth=3>"
        assert repr(QuarterEnd(startingMonth=3)) == expected
        expected = "<QuarterEnd: startingMonth=1>"
        assert repr(QuarterEnd(startingMonth=1)) == expected

    def test_offset_corner_case(self):
        # [FUNCTION DOC]
        # A negative one offset with starting month 1 applied to February 1 2010 gives January 31 2010.
        # corner
        offset = QuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 31)

    offset_cases = []
    offset_cases.append(
        (
            QuarterEnd(startingMonth=1),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 4, 30),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 7, 31),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2008, 2, 15): datetime(2008, 2, 29),
                datetime(2008, 2, 29): datetime(2008, 5, 31),
                datetime(2008, 3, 15): datetime(2008, 5, 31),
                datetime(2008, 3, 31): datetime(2008, 5, 31),
                datetime(2008, 4, 15): datetime(2008, 5, 31),
                datetime(2008, 4, 30): datetime(2008, 5, 31),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2008, 2, 15): datetime(2008, 4, 30),
                datetime(2008, 2, 29): datetime(2008, 4, 30),
                datetime(2008, 3, 15): datetime(2008, 4, 30),
                datetime(2008, 3, 31): datetime(2008, 4, 30),
                datetime(2008, 4, 15): datetime(2008, 4, 30),
                datetime(2008, 4, 30): datetime(2008, 4, 30),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=-1),
            {
                datetime(2008, 1, 1): datetime(2007, 10, 31),
                datetime(2008, 1, 31): datetime(2007, 10, 31),
                datetime(2008, 2, 15): datetime(2008, 1, 31),
                datetime(2008, 2, 29): datetime(2008, 1, 31),
                datetime(2008, 3, 15): datetime(2008, 1, 31),
                datetime(2008, 3, 31): datetime(2008, 1, 31),
                datetime(2008, 4, 15): datetime(2008, 1, 31),
                datetime(2008, 4, 30): datetime(2008, 1, 31),
                datetime(2008, 7, 1): datetime(2008, 4, 30),
            },
        )
    )

    offset_cases.append(
        (
            QuarterEnd(startingMonth=1, n=2),
            {
                datetime(2008, 1, 31): datetime(2008, 7, 31),
                datetime(2008, 2, 15): datetime(2008, 7, 31),
                datetime(2008, 2, 29): datetime(2008, 7, 31),
                datetime(2008, 3, 15): datetime(2008, 7, 31),
                datetime(2008, 3, 31): datetime(2008, 7, 31),
                datetime(2008, 4, 15): datetime(2008, 7, 31),
                datetime(2008, 4, 30): datetime(2008, 10, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over the case table (starting months 1 and 2; n of 0, minus one, two). For each base date asserts the offset gives the expected month-end date.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (QuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False),
        (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 31), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False),
        (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), False),
        (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 31), True),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False),
        (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), True),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False),
        (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 31), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), False),
        (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over (offset, date, expected); asserts the on-offset check matches.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
