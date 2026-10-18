"""
Tests for the following offsets:
- BMonthBegin
- BMonthEnd
"""

# [MODULE DOC]
# Tests for the business month begin and business month end offsets: vectorized application to indexes and series, equality, normalization, application
# tables for steps -2 to 2 and zero, and on-offset checks. Uses the shared offsets assertion helpers. No import-time side effects.
from __future__ import annotations

from datetime import datetime

import pytest

import pandas as pd
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
)


@pytest.mark.parametrize("n", [-2, 1])
@pytest.mark.parametrize(
    "cls",
    [
        BMonthBegin,
        BMonthEnd,
    ],
)
def test_apply_index(cls, n):
    # [FUNCTION DOC]
    # Parametrized over both classes and steps -2 and 1. Adds the offset to a 100000-minute date range and to a series of it. The index result has no frequency; first and last elements equal the scalar sum of the offset with the corresponding input elements; the series result matches elementwise at first and last positions.
    offset = cls(n=n)
    rng = pd.date_range(start="1/1/2000", periods=100000, freq="min")
    ser = pd.Series(rng)

    res = rng + offset
    assert res.freq is None  # not retained
    assert res[0] == rng[0] + offset
    assert res[-1] == rng[-1] + offset
    res2 = ser + offset
    # apply_index is only for indexes, not series, so no res2_v2
    assert res2.iloc[0] == ser.iloc[0] + offset
    assert res2.iloc[-1] == ser.iloc[-1] + offset


class TestBMonthBegin:
    # [CLASS DOC]
    # Test collection for the business month begin offset; case tables are class attributes built at import (class definition) time.
    def test_offsets_compare_equal(self):
        # [FUNCTION DOC]
        # Root cause of #456: two default offsets are not unequal (the not-equal operator is false).
        # root cause of #456
        offset1 = BMonthBegin()
        offset2 = BMonthBegin()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 1): datetime(2006, 10, 2),
                datetime(2007, 1, 1): datetime(2007, 2, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 10, 2): datetime(2006, 10, 2),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 15): datetime(2006, 10, 2),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(2),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 3),
                datetime(2008, 1, 15): datetime(2008, 3, 3),
                datetime(2006, 12, 29): datetime(2007, 2, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 2),
                datetime(2008, 6, 1): datetime(2008, 5, 1),
                datetime(2008, 3, 10): datetime(2008, 3, 3),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 12, 30): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases, each (offset, mapping of start to expected). Checks each pair using the assertion helper for steps 1, 0, 2 and -1.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthBegin(), datetime(2007, 12, 31), False),
        (BMonthBegin(), datetime(2008, 1, 1), True),
        (BMonthBegin(), datetime(2001, 4, 2), True),
        (BMonthBegin(), datetime(2008, 3, 3), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases: first business day of a month is on offset (2008-01-01, 2001-04-02, 2008-03-03), the last day of the previous month is not.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBMonthEnd:
    # [CLASS DOC]
    # Test collection for the business month end offset; case tables are class attributes.
    def test_normalize(self):
        # [FUNCTION DOC]
        # A datetime at 03:00 plus an offset with normalization on equals the midnight-truncated datetime plus the default offset.
        dt = datetime(2007, 1, 1, 3)

        result = dt + BMonthEnd(normalize=True)
        expected = dt.replace(hour=0) + BMonthEnd()
        assert result == expected

    def test_offsets_compare_equal(self):
        # [FUNCTION DOC]
        # Root cause of #456: two default offsets are not unequal.
        # root cause of #456
        offset1 = BMonthEnd()
        offset2 = BMonthEnd()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 29),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 2, 28),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2008, 5, 30),
                datetime(2008, 12, 31): datetime(2008, 11, 28),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over offset_cases for steps 1, 0, 2 and -1; checks each start date against its expected last business day.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthEnd(), datetime(2007, 12, 31), True),
        (BMonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases: 2007-12-31 (a Monday, last business day) is on offset; 2008-01-01 is not.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
