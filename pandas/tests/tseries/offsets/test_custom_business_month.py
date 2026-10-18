"""
Tests for the following offsets:
- CustomBusinessMonthBase
- CustomBusinessMonthBegin
- CustomBusinessMonthEnd
"""

# [MODULE DOC]
# Tests for the custom business month begin and end offsets (CBMonthBegin, CBMonthEnd): equality, hashing, pickling, copying, repr, normalization inequality, addition to datetimes, rolling, on-offset, application tables for several steps, large steps, custom holidays, and an extra fixed time offset added after rolling. The custom business day offset is used in a few checks. Uses the shared offsets assertion helpers.
# No import-time side effects.
from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import (
    CBMonthBegin,
    CBMonthEnd,
    CDay,
)

import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries import offsets


@pytest.fixture
def dt():
    # [FUNCTION DOC]
    # HELPER: fixture returning the base date 2008-01-01.
    return datetime(2008, 1, 1)


class TestCommonCBM:
    # [CLASS DOC]
    # Tests behavior common to both the custom business month begin and end classes, parametrized over the class or a step-two instance.
    @pytest.mark.parametrize("offset2", [CBMonthBegin(2), CBMonthEnd(2)])
    def test_eq(self, offset2):
        # [FUNCTION DOC]
        # Parametrized over step-two begin and end instances; an instance equals itself.
        assert offset2 == offset2

    @pytest.mark.parametrize("offset2", [CBMonthBegin(2), CBMonthEnd(2)])
    def test_hash(self, offset2):
        # [FUNCTION DOC]
        # Parametrized over step-two begin and end instances; the hash is stable across calls.
        assert hash(offset2) == hash(offset2)

    @pytest.mark.parametrize("_offset", [CBMonthBegin, CBMonthEnd])
    def test_roundtrip_pickle(self, _offset, temp_file):
        # [FUNCTION DOC]
        # Parametrized over both classes; uses the temp_file fixture. Default, step-two and doubled instances each survive a pickle round trip equal to the original.
        def _check_roundtrip(obj):
            # [FUNCTION DOC]
            # HELPER: nested function for the pickle test; round-trips an object through pickle via the enclosing temp file and asserts equality with the original.
            unpickled = tm.round_trip_pickle(obj, temp_file)
            assert unpickled == obj

        _check_roundtrip(_offset())
        _check_roundtrip(_offset(2))
        _check_roundtrip(_offset() * 2)

    @pytest.mark.parametrize("_offset", [CBMonthBegin, CBMonthEnd])
    def test_copy(self, _offset):
        # [FUNCTION DOC]
        # GH 17452. Parametrized over both classes. An instance created with a Monday-Wednesday-Friday week mask equals its copy.
        # GH 17452
        off = _offset(weekmask="Mon Wed Fri")
        assert off == off.copy()


class TestCustomBusinessMonthBegin:
    # [CLASS DOC]
    # Test collection for the custom business month begin offset; case tables are class attributes.
    def test_different_normalize_equals(self):
        # [FUNCTION DOC]
        # GH#21404. An instance and an instance with normalization enabled compare unequal.
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = CBMonthBegin()
        offset2 = CBMonthBegin(normalize=True)
        assert offset != offset2

    def test_repr(self):
        # [FUNCTION DOC]
        # Default prints as a custom business month begin; step two prints with a multiplier and plural class name.
        assert repr(CBMonthBegin()) == "<CustomBusinessMonthBegin>"
        assert repr(CBMonthBegin(2)) == "<2 * CustomBusinessMonthBegins>"

    def test_add_datetime(self, dt):
        # [FUNCTION DOC]
        # Adding the step-two offset to 2008-01-01 gives 2008-03-03 (first business day of March).
        assert CBMonthBegin(2) + dt == datetime(2008, 3, 3)

    def testRollback1(self):
        # [FUNCTION DOC]
        # Rolls back 2007-12-31 with a step-ten custom business day offset (a different class from the one under test) and expects the unchanged date, since it is a business day.
        # KNOWN ISSUE: the test is inside the month-begin class but exercises the custom business day class rather than the month-begin offset, so it does not test the class named by the test collection.
        assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)

    def testRollback2(self, dt):
        # [FUNCTION DOC]
        # Rolling back the base date 2008-01-01 (already the first business day of its month) with a step-ten month begin offset returns it unchanged.
        assert CBMonthBegin(10).rollback(dt) == datetime(2008, 1, 1)

    def testRollforward1(self, dt):
        # [FUNCTION DOC]
        # Rolling forward 2008-01-01 (already on offset) returns it unchanged.
        assert CBMonthBegin(10).rollforward(dt) == datetime(2008, 1, 1)

    def test_roll_date_object(self):
        # [FUNCTION DOC]
        # A plain date (2012-09-15) rolls back to 2012-09-03 and forward to 2012-10-01 with the month begin offset; with the calendar day offset both rolls return the same day as a datetime.
        offset = CBMonthBegin()

        dt = date(2012, 9, 15)

        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 3)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 10, 1)

        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    on_offset_cases = [
        (CBMonthBegin(), datetime(2008, 1, 1), True),
        (CBMonthBegin(), datetime(2008, 1, 31), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases: 2008-01-01 is on offset; 2008-01-31 is not.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    apply_cases = [
        (
            CBMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 2, 7): datetime(2008, 3, 3),
            },
        ),
        (
            2 * CBMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 3),
                datetime(2008, 2, 7): datetime(2008, 4, 1),
            },
        ),
        (
            -CBMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2007, 12, 3),
                datetime(2008, 2, 8): datetime(2008, 2, 1),
            },
        ),
        (
            -2 * CBMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2007, 11, 1),
                datetime(2008, 2, 9): datetime(2008, 1, 1),
            },
        ),
        (
            CBMonthBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 7): datetime(2008, 2, 1),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case):
        # [FUNCTION DOC]
        # Parametrized over apply_cases for steps 1, 2, -1, -2 and 0; each base maps to the expected first business day of a month.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_large_n(self):
        # [FUNCTION DOC]
        # Ten months forward from 2012-10-23 is 2013-08-01; adding then subtracting 100 custom business days returns the start; six months back from 2012-01-01 is 2011-07-01; six months forward from 2011-12-18 is 2012-06-01.
        dt = datetime(2012, 10, 23)

        result = dt + CBMonthBegin(10)
        assert result == datetime(2013, 8, 1)

        result = dt + CDay(100) - CDay(100)
        assert result == dt

        off = CBMonthBegin() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 7, 1)
        assert rs == xp

        st = datetime(2011, 12, 18)
        rs = st + off

        xp = datetime(2012, 6, 1)
        assert rs == xp

    def test_holidays(self):
        # [FUNCTION DOC]
        # Holidays given as a string, datetime and NumPy datetime64 (2012-02-01, 2012-02-02, 2012-03-01). From 2012-01-01, one step lands on 2012-01-02 and two steps land on 2012-02-03 (skipping the holiday first business days).
        # Define a TradingDay offset
        holidays = ["2012-02-01", datetime(2012, 2, 2), np.datetime64("2012-03-01")]
        bm_offset = CBMonthBegin(holidays=holidays)
        dt = datetime(2012, 1, 1)

        assert dt + bm_offset == datetime(2012, 1, 2)
        assert dt + 2 * bm_offset == datetime(2012, 2, 3)

    @pytest.mark.parametrize(
        "case",
        [
            (
                CBMonthBegin(n=1, offset=timedelta(days=5)),
                {
                    datetime(2021, 3, 1): datetime(2021, 4, 1) + timedelta(days=5),
                    datetime(2021, 4, 17): datetime(2021, 5, 3) + timedelta(days=5),
                },
            ),
            (
                CBMonthBegin(n=2, offset=timedelta(days=40)),
                {
                    datetime(2021, 3, 10): datetime(2021, 5, 3) + timedelta(days=40),
                    datetime(2021, 4, 30): datetime(2021, 6, 1) + timedelta(days=40),
                },
            ),
            (
                CBMonthBegin(n=1, offset=timedelta(days=-5)),
                {
                    datetime(2021, 3, 1): datetime(2021, 4, 1) - timedelta(days=5),
                    datetime(2021, 4, 11): datetime(2021, 5, 3) - timedelta(days=5),
                },
            ),
            (
                -2 * CBMonthBegin(n=1, offset=timedelta(days=10)),
                {
                    datetime(2021, 3, 1): datetime(2021, 1, 1) + timedelta(days=10),
                    datetime(2021, 4, 3): datetime(2021, 3, 1) + timedelta(days=10),
                },
            ),
            (
                CBMonthBegin(n=0, offset=timedelta(days=1)),
                {
                    datetime(2021, 3, 2): datetime(2021, 4, 1) + timedelta(days=1),
                    datetime(2021, 4, 1): datetime(2021, 4, 1) + timedelta(days=1),
                },
            ),
            (
                CBMonthBegin(
                    n=1, holidays=["2021-04-01", "2021-04-02"], offset=timedelta(days=1)
                ),
                {
                    datetime(2021, 3, 2): datetime(2021, 4, 5) + timedelta(days=1),
                },
            ),
        ],
    )
    def test_apply_with_extra_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over cases that combine the month begin offset with an extra fixed time delta (positive, large, negative, with a negative multiplier, step zero, and with holidays). The expected result is the rolled first business day with the delta added afterward.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestCustomBusinessMonthEnd:
    # [CLASS DOC]
    # Test collection for the custom business month end offset; case tables are class attributes.
    def test_different_normalize_equals(self):
        # [FUNCTION DOC]
        # GH#21404. An instance and an instance with normalization enabled compare unequal.
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = CBMonthEnd()
        offset2 = CBMonthEnd(normalize=True)
        assert offset != offset2

    def test_repr(self):
        # [FUNCTION DOC]
        # Default prints as a custom business month end; step two prints with a multiplier and plural class name.
        assert repr(CBMonthEnd()) == "<CustomBusinessMonthEnd>"
        assert repr(CBMonthEnd(2)) == "<2 * CustomBusinessMonthEnds>"

    def test_add_datetime(self, dt):
        # [FUNCTION DOC]
        # Adding the step-two offset to 2008-01-01 gives 2008-02-29.
        assert CBMonthEnd(2) + dt == datetime(2008, 2, 29)

    def testRollback1(self):
        # [FUNCTION DOC]
        # Rolls back 2007-12-31 with a step-ten custom business day offset and expects the unchanged date.
        # KNOWN ISSUE: exercises the custom business day class, not the month-end offset this collection is for.
        assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)

    def testRollback2(self, dt):
        # [FUNCTION DOC]
        # Rolling back 2008-01-01 with a step-ten month end offset gives 2007-12-31.
        assert CBMonthEnd(10).rollback(dt) == datetime(2007, 12, 31)

    def testRollforward1(self, dt):
        # [FUNCTION DOC]
        # Rolling forward 2008-01-01 gives 2008-01-31.
        assert CBMonthEnd(10).rollforward(dt) == datetime(2008, 1, 31)

    def test_roll_date_object(self):
        # [FUNCTION DOC]
        # A plain date (2012-09-15) rolls back to 2012-08-31 and forward to 2012-09-28; with the calendar day offset both rolls return the same day as a datetime.
        offset = CBMonthEnd()

        dt = date(2012, 9, 15)

        result = offset.rollback(dt)
        assert result == datetime(2012, 8, 31)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 28)

        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    on_offset_cases = [
        (CBMonthEnd(), datetime(2008, 1, 31), True),
        (CBMonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases: 2008-01-31 is on offset; 2008-01-01 is not.
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    apply_cases = [
        (
            CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 2, 7): datetime(2008, 2, 29),
            },
        ),
        (
            2 * CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 2, 7): datetime(2008, 3, 31),
            },
        ),
        (
            -CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2007, 12, 31),
                datetime(2008, 2, 8): datetime(2008, 1, 31),
            },
        ),
        (
            -2 * CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2007, 11, 30),
                datetime(2008, 2, 9): datetime(2007, 12, 31),
            },
        ),
        (
            CBMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 2, 7): datetime(2008, 2, 29),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case):
        # [FUNCTION DOC]
        # Parametrized over apply_cases for steps 1, 2, -1, -2 and 0; each base maps to the expected last business day of a month.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_large_n(self):
        # [FUNCTION DOC]
        # Ten months forward from 2012-10-23 is 2013-07-31; adding then subtracting 100 custom business days returns the start; six months back from 2012-01-01 is 2011-07-29; six months forward from 2011-12-18 is 2012-05-31.
        dt = datetime(2012, 10, 23)

        result = dt + CBMonthEnd(10)
        assert result == datetime(2013, 7, 31)

        result = dt + CDay(100) - CDay(100)
        assert result == dt

        off = CBMonthEnd() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 7, 29)
        assert rs == xp

        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2012, 5, 31)
        assert rs == xp

    def test_holidays(self):
        # [FUNCTION DOC]
        # Holidays given as a string, datetime and NumPy datetime64 (2012-01-31, 2012-02-28, 2012-02-29). From 2012-01-01, one step lands on 2012-01-30 and two steps on 2012-02-27.
        # Define a TradingDay offset
        holidays = ["2012-01-31", datetime(2012, 2, 28), np.datetime64("2012-02-29")]
        bm_offset = CBMonthEnd(holidays=holidays)
        dt = datetime(2012, 1, 1)
        assert dt + bm_offset == datetime(2012, 1, 30)
        assert dt + 2 * bm_offset == datetime(2012, 2, 27)

    @pytest.mark.parametrize(
        "case",
        [
            (
                CBMonthEnd(n=1, offset=timedelta(days=5)),
                {
                    datetime(2021, 3, 1): datetime(2021, 3, 31) + timedelta(days=5),
                    datetime(2021, 4, 17): datetime(2021, 4, 30) + timedelta(days=5),
                },
            ),
            (
                CBMonthEnd(n=2, offset=timedelta(days=40)),
                {
                    datetime(2021, 3, 10): datetime(2021, 4, 30) + timedelta(days=40),
                    datetime(2021, 4, 30): datetime(2021, 6, 30) + timedelta(days=40),
                },
            ),
            (
                CBMonthEnd(n=1, offset=timedelta(days=-5)),
                {
                    datetime(2021, 3, 1): datetime(2021, 3, 31) - timedelta(days=5),
                    datetime(2021, 4, 11): datetime(2021, 4, 30) - timedelta(days=5),
                },
            ),
            (
                -2 * CBMonthEnd(n=1, offset=timedelta(days=10)),
                {
                    datetime(2021, 3, 1): datetime(2021, 1, 29) + timedelta(days=10),
                    datetime(2021, 4, 3): datetime(2021, 2, 26) + timedelta(days=10),
                },
            ),
            (
                CBMonthEnd(n=0, offset=timedelta(days=1)),
                {
                    datetime(2021, 3, 2): datetime(2021, 3, 31) + timedelta(days=1),
                    datetime(2021, 4, 1): datetime(2021, 4, 30) + timedelta(days=1),
                },
            ),
            (
                CBMonthEnd(n=1, holidays=["2021-03-31"], offset=timedelta(days=1)),
                {
                    datetime(2021, 3, 2): datetime(2021, 3, 30) + timedelta(days=1),
                },
            ),
        ],
    )
    def test_apply_with_extra_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over cases combining the month end offset with an extra fixed time delta (positive, large, negative, negative multiplier, step zero, with holidays). The expected result is the rolled last business day with the delta added afterward.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
