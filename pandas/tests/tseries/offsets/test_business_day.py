"""
Tests for offsets.BDay
"""

# [MODULE DOC]
# Tests for the business day offset (BDay): equality, repr, combination with time deltas,
# datetime/date/NumPy inputs, rolling to the nearest business day, application for positive, negative and zero
# step counts, on-offset checks and type errors. Shared fixtures supply a base date and the offset class so other modules
# (for instance the trading-day variant) could reuse the class; the assertion helpers come from the shared offsets test helper module.
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
    ApplyTypeError,
    BDay,
    BMonthEnd,
)

from pandas import (
    DatetimeIndex,
    Timedelta,
    _testing as tm,
)
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries import offsets


@pytest.fixture
def dt():
    # [FUNCTION DOC]
    # HELPER: fixture returning the base date Tuesday 2008-01-01 used by many tests in this module.
    return datetime(2008, 1, 1)


@pytest.fixture
def _offset():
    # [FUNCTION DOC]
    # HELPER: fixture returning the offset class under test (the business day class itself, not an instance) so tests can construct offsets with varying arguments.
    return BDay


@pytest.fixture
def offset(_offset):
    # [FUNCTION DOC]
    # HELPER: fixture returning a default (step one) instance of the class from the _offset fixture.
    return _offset()


@pytest.fixture
def offset2(_offset):
    # [FUNCTION DOC]
    # HELPER: fixture returning an instance with step two of the class from the _offset fixture.
    return _offset(2)


class TestBusinessDay:
    # [CLASS DOC]
    # Test collection for the business day offset; defines a table of application cases as a class attribute (see apply_cases) used by the apply test.
    def test_different_normalize_equals(self, _offset, offset2):
        # [FUNCTION DOC]
        # GH#21404. Offsets that differ only in the normalize flag compare unequal.
        # KNOWN ISSUE: the offset2 fixture parameter is unused and both local offsets are rebuilt inside the body, shadowing the fixture value.
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(self, offset, offset2):
        # [FUNCTION DOC]
        # Default offset prints as a business day, step two prints as two business days, and adding a one-day time delta gives a representation that includes the offset field with the timedelta of one day.
        assert repr(offset) == "<BusinessDay>"
        assert repr(offset2) == "<2 * BusinessDays>"

        expected = "<BusinessDay: offset=datetime.timedelta(days=1)>"
        assert repr(offset + timedelta(1)) == expected

    def test_with_offset(self, dt, offset):
        # [FUNCTION DOC]
        # Adding a two-hour time delta to the offset and adding that to 2008-01-01 yields 2008-01-02 02:00 (one business day plus the extra two hours).
        offset = offset + timedelta(hours=2)

        assert (dt + offset) == datetime(2008, 1, 2, 2)

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta(hours=2),
            Timedelta(hours=2).to_pytimedelta(),
            Timedelta(hours=2).to_timedelta64(),
        ],
        ids=lambda x: type(x),
    )
    def test_with_offset_index(self, td, dt, offset):
        # [FUNCTION DOC]
        # Parametrized over a two-hour Timedelta, python timedelta and NumPy timedelta (ids use the type). Adding the combination of time delta and offset (either operand order) to a one-element datetime index
        # equals an index holding 2008-01-02 02:00.
        dti = DatetimeIndex([dt])
        expected = DatetimeIndex([datetime(2008, 1, 2, 2)])

        result = dti + (td + offset)
        tm.assert_index_equal(result, expected)

        result = dti + (offset + td)
        tm.assert_index_equal(result, expected)

    def test_eq(self, offset2):
        # [FUNCTION DOC]
        # An offset compares equal to itself.
        assert offset2 == offset2

    def test_hash(self, offset2):
        # [FUNCTION DOC]
        # An offset's hash is stable across calls.
        assert hash(offset2) == hash(offset2)

    def test_add_datetime(self, dt, offset2):
        # [FUNCTION DOC]
        # Adding the step-two offset to 2008-01-01 (a datetime or a NumPy datetime64) gives 2008-01-03.
        assert offset2 + dt == datetime(2008, 1, 3)
        assert offset2 + np.datetime64("2008-01-01 00:00:00") == datetime(2008, 1, 3)

    def testRollback1(self, dt, _offset):
        # [FUNCTION DOC]
        # Rolling back a date that is already a business day, with a step of ten, returns the same date.
        assert _offset(10).rollback(dt) == dt

    def testRollback2(self, _offset):
        # [FUNCTION DOC]
        # Rolling back Saturday 2008-01-05 returns Friday 2008-01-04.
        assert _offset(10).rollback(datetime(2008, 1, 5)) == datetime(2008, 1, 4)

    def testRollforward1(self, dt, _offset):
        # [FUNCTION DOC]
        # Rolling forward a date that is already a business day returns the same date.
        assert _offset(10).rollforward(dt) == dt

    def testRollforward2(self, _offset):
        # [FUNCTION DOC]
        # Rolling forward Saturday 2008-01-05 returns Monday 2008-01-07.
        assert _offset(10).rollforward(datetime(2008, 1, 5)) == datetime(2008, 1, 7)

    def test_roll_date_object(self, offset):
        # [FUNCTION DOC]
        # A plain date object (2012-09-15, a Saturday) rolls back to datetime 2012-09-14 and forward to 2012-09-17 with the business day offset; with the calendar day offset both roll operations return the same day as a datetime.
        dt = date(2012, 9, 15)

        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 14)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 17)

        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    @pytest.mark.parametrize(
        "dt, expected",
        [
            (datetime(2008, 1, 1), True),
            (datetime(2008, 1, 5), False),
        ],
    )
    def test_is_on_offset(self, offset, dt, expected):
        # [FUNCTION DOC]
        # Parametrized over (date, expected): Tuesday 2008-01-01 is on the offset, Saturday 2008-01-05 is not.
        assert_is_on_offset(offset, dt, expected)

    apply_cases: list[tuple[int, dict[datetime, datetime]]] = [
        (
            1,
            {
                datetime(2008, 1, 1): datetime(2008, 1, 2),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 8),
            },
        ),
        (
            2,
            {
                datetime(2008, 1, 1): datetime(2008, 1, 3),
                datetime(2008, 1, 4): datetime(2008, 1, 8),
                datetime(2008, 1, 5): datetime(2008, 1, 8),
                datetime(2008, 1, 6): datetime(2008, 1, 8),
                datetime(2008, 1, 7): datetime(2008, 1, 9),
            },
        ),
        (
            -1,
            {
                datetime(2008, 1, 1): datetime(2007, 12, 31),
                datetime(2008, 1, 4): datetime(2008, 1, 3),
                datetime(2008, 1, 5): datetime(2008, 1, 4),
                datetime(2008, 1, 6): datetime(2008, 1, 4),
                datetime(2008, 1, 7): datetime(2008, 1, 4),
                datetime(2008, 1, 8): datetime(2008, 1, 7),
            },
        ),
        (
            -2,
            {
                datetime(2008, 1, 1): datetime(2007, 12, 28),
                datetime(2008, 1, 4): datetime(2008, 1, 2),
                datetime(2008, 1, 5): datetime(2008, 1, 3),
                datetime(2008, 1, 6): datetime(2008, 1, 3),
                datetime(2008, 1, 7): datetime(2008, 1, 3),
                datetime(2008, 1, 8): datetime(2008, 1, 4),
                datetime(2008, 1, 9): datetime(2008, 1, 7),
            },
        ),
        (
            0,
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 4): datetime(2008, 1, 4),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 7),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case, _offset):
        # [FUNCTION DOC]
        # Parametrized over apply_cases, each a (step, mapping of base date to expected date) pair covering steps 1, 2, -1, -2 and 0 across weekends. Builds the offset with the step and checks each base through the assertion helper.
        n, cases = case
        offset = _offset(n)
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_large_n(self, _offset):
        # [FUNCTION DOC]
        # Checks large steps: ten business days after 2012-10-23 is 2012-11-06; adding then subtracting 100 returns the start; subtracting six business days from 2012-01-01 gives 2011-12-23; adding six to Sunday 2011-12-18 gives 2011-12-26;
        # adding ten to Sunday 2014-01-05 gives 2014-01-17 (see #5890).
        dt = datetime(2012, 10, 23)

        result = dt + _offset(10)
        assert result == datetime(2012, 11, 6)

        result = dt + _offset(100) - _offset(100)
        assert result == dt

        off = _offset() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 12, 23)
        assert rs == xp

        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2011, 12, 26)
        assert rs == xp

        off = _offset() * 10
        rs = datetime(2014, 1, 5) + off  # see #5890
        xp = datetime(2014, 1, 17)
        assert rs == xp

    def test_apply_corner(self, _offset):
        # [FUNCTION DOC]
        # Applying the offset to a non-date offset object (business month end) raises ApplyTypeError. The expected message differs by class: "Only know how to combine business day with datetime or timedelta" for the business day class, otherwise a trading-day message.
        # KNOWN ISSUE: the fixture always supplies the business day class, so the trading-day message branch is unreachable in this module.
        if _offset is BDay:
            msg = "Only know how to combine business day with datetime or timedelta"
        else:
            msg = (
                "Only know how to combine trading day "
                "with datetime, datetime64 or timedelta"
            )
        with pytest.raises(ApplyTypeError, match=msg):
            _offset()._apply(BMonthEnd())
