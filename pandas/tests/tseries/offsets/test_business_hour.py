"""
Tests for offsets.BusinessHour
"""

# [MODULE DOC]
# Tests for the business hour offset (BusinessHour): constructor validation (single and multiple opening intervals), normalization, repr, equality and hashing, arithmetic with datetimes, rolling, on-offset checks, application tables (default and custom hours, overnight intervals, multiple intervals, large step counts, nanoseconds), plus tests of the business day offset with a time delta component and of the private opening-time helpers.
# Fixtures offset1 to offset10 provide a variety of instances (default, positive and negative steps, custom start/end as strings or times, overnight, multiple intervals). Uses the shared offsets assertion helper and the unit fixture from the test configuration. No import-time side effects.
from __future__ import annotations

from datetime import (
    datetime,
    time as dt_time,
)

import pytest

from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas._libs.tslibs.offsets import (
    BDay,
    BusinessHour,
    Nano,
)

from pandas import (
    DatetimeIndex,
    _testing as tm,
    date_range,
)
from pandas.tests.tseries.offsets.common import assert_offset_equal


@pytest.fixture
def dt():
    # [FUNCTION DOC]
    # HELPER: fixture returning the base datetime 2014-07-01 10:00 (a Tuesday).
    return datetime(2014, 7, 1, 10, 00)


@pytest.fixture
def _offset():
    # [FUNCTION DOC]
    # HELPER: fixture returning the business hour class (not an instance).
    return BusinessHour


@pytest.fixture
def offset1():
    # [FUNCTION DOC]
    # HELPER: fixture returning the default business hour offset (09:00-17:00, step 1).
    return BusinessHour()


@pytest.fixture
def offset2():
    # [FUNCTION DOC]
    # HELPER: fixture returning a default-hours business hour offset with step 3.
    return BusinessHour(n=3)


@pytest.fixture
def offset3():
    # [FUNCTION DOC]
    # HELPER: fixture returning a default-hours business hour offset with step -1.
    return BusinessHour(n=-1)


@pytest.fixture
def offset4():
    # [FUNCTION DOC]
    # HELPER: fixture returning a default-hours business hour offset with step -4.
    return BusinessHour(n=-4)


@pytest.fixture
def offset5():
    # [FUNCTION DOC]
    # HELPER: fixture returning a business hour offset with hours 11:00 to 14:30 given as time objects.
    return BusinessHour(start=dt_time(11, 0), end=dt_time(14, 30))


@pytest.fixture
def offset6():
    # [FUNCTION DOC]
    # HELPER: fixture returning a business hour offset with an overnight interval 20:00 to 05:00.
    return BusinessHour(start="20:00", end="05:00")


@pytest.fixture
def offset7():
    # [FUNCTION DOC]
    # HELPER: fixture returning an overnight offset with step -2, hours 21:30 to 06:30.
    return BusinessHour(n=-2, start=dt_time(21, 30), end=dt_time(6, 30))


@pytest.fixture
def offset8():
    # [FUNCTION DOC]
    # HELPER: fixture returning an offset with two intervals per day: 09:00-12:00 and 13:00-17:00.
    return BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"])


@pytest.fixture
def offset9():
    # [FUNCTION DOC]
    # HELPER: fixture returning a step-3 offset with two intervals: 09:00-13:00 and 22:00-03:00 (second one overnight).
    return BusinessHour(n=3, start=["09:00", "22:00"], end=["13:00", "03:00"])


@pytest.fixture
def offset10():
    # [FUNCTION DOC]
    # HELPER: fixture returning a step -1 offset with two intervals starting 23:00 and 13:00 and ending 02:00 and 17:00 (given unsorted).
    return BusinessHour(n=-1, start=["23:00", "13:00"], end=["02:00", "17:00"])


class TestBusinessHour:
    # [CLASS DOC]
    # Test collection for the business hour offset; several case tables (normalize, on-offset, apply, large step) are class attributes built at class definition.
    @pytest.mark.parametrize(
        "start,end,match",
        [
            (
                dt_time(11, 0, 5),
                "17:00",
                "time data must be specified only with hour and minute",
            ),
            ("AAA", "17:00", "time data must match '%H:%M' format"),
            ("14:00:05", "17:00", "time data must match '%H:%M' format"),
            ([], "17:00", "Must include at least 1 start time"),
            ("09:00", [], "Must include at least 1 end time"),
            (
                ["09:00", "11:00"],
                "17:00",
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["10:00"],
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["12:00", "20:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
            (
                ["12:00", "20:00"],
                ["09:00", "11:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
        ],
    )
    def test_constructor_errors(self, start, end, match):
        # [FUNCTION DOC]
        # Parametrized over (start, end, expected message). Construction raises a value error for: a time with seconds; strings that are not hour:minute; an empty start or end list ("Must include at least 1 start/end time"); lists of unequal length; and intervals that touch or overlap each other.
        with pytest.raises(ValueError, match=match):
            BusinessHour(start=start, end=end)

    def test_different_normalize_equals(self, _offset):
        # [FUNCTION DOC]
        # GH#21404. An instance and an instance with normalization enabled compare unequal.
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(
        self,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        # [FUNCTION DOC]
        # Uses the ten offset fixtures. The representation shows the step prefix when not 1 (with a plural class name when absolute step is greater than one) and the hours as start-end, with multiple intervals comma-separated and sorted by start time.
        assert repr(offset1) == "<BusinessHour: bh=09:00-17:00>"
        assert repr(offset2) == "<3 * BusinessHours: bh=09:00-17:00>"
        assert repr(offset3) == "<-1 * BusinessHour: bh=09:00-17:00>"
        assert repr(offset4) == "<-4 * BusinessHours: bh=09:00-17:00>"

        assert repr(offset5) == "<BusinessHour: bh=11:00-14:30>"
        assert repr(offset6) == "<BusinessHour: bh=20:00-05:00>"
        assert repr(offset7) == "<-2 * BusinessHours: bh=21:30-06:30>"
        assert repr(offset8) == "<BusinessHour: bh=09:00-12:00,13:00-17:00>"
        assert repr(offset9) == "<3 * BusinessHours: bh=09:00-13:00,22:00-03:00>"
        assert repr(offset10) == "<-1 * BusinessHour: bh=13:00-17:00,23:00-02:00>"

    def test_with_offset(self, dt):
        # [FUNCTION DOC]
        # Adding three hours (multiplying a single-hour offset by 3, or constructing with step 3) to 2014-07-01 10:00 gives 13:00.
        expected = Timestamp("2014-07-01 13:00")

        assert dt + BusinessHour() * 3 == expected
        assert dt + BusinessHour(n=3) == expected

    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_eq_attribute(self, offset_name, request):
        # [FUNCTION DOC]
        # Parametrized over fixture names; each offset equals itself.
        offset = request.getfixturevalue(offset_name)
        assert offset == offset

    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(start="09:00"), BusinessHour()),
            (
                BusinessHour(start=["23:00", "13:00"], end=["12:00", "17:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_eq(self, offset1, offset2):
        # [FUNCTION DOC]
        # Parametrized over pairs expected equal: explicit default start equals the default offset; the same two intervals given in different order are equal.
        assert offset1 == offset2

    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(), BusinessHour(-1)),
            (BusinessHour(start="09:00"), BusinessHour(start="09:01")),
            (
                BusinessHour(start="09:00", end="17:00"),
                BusinessHour(start="17:00", end="09:01"),
            ),
            (
                BusinessHour(start=["13:00", "23:00"], end=["18:00", "07:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_neq(self, offset1, offset2):
        # [FUNCTION DOC]
        # Parametrized over pairs expected unequal: different step, different start, different end, and different interval end lists.
        assert offset1 != offset2

    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_hash(self, offset_name, request):
        # [FUNCTION DOC]
        # Parametrized over fixture names.
        # KNOWN ISSUE: despite the name, this checks equality of the offset with itself and never computes a hash; it duplicates the attribute equality test.
        offset = request.getfixturevalue(offset_name)
        assert offset == offset

    def test_add_datetime(
        self,
        dt,
        offset1,
        offset2,
        offset3,
        offset4,
        offset8,
        offset9,
        offset10,
    ):
        # [FUNCTION DOC]
        # Adds each of several fixture offsets to 2014-07-01 10:00 and checks the result: one hour gives 11:00; three hours gives 13:00; minus one hour gives the previous day at 17:00 closing; minus four hours gives the previous day 14:00; the multi-interval cases give 11:00, 22:00 and 01:00 next day.
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 13)
        assert offset3 + dt == datetime(2014, 6, 30, 17)
        assert offset4 + dt == datetime(2014, 6, 30, 14)
        assert offset8 + dt == datetime(2014, 7, 1, 11)
        assert offset9 + dt == datetime(2014, 7, 1, 22)
        assert offset10 + dt == datetime(2014, 7, 1, 1)

    def test_sub(self, dt, offset2, _offset):
        # [FUNCTION DOC]
        # Subtracting a datetime from an offset raises a type error "Cannot subtract datetime from offset". Twice the step-3 offset minus itself equals itself, and subtracting it from the base equals adding the class built with step -3.
        off = offset2
        msg = "Cannot subtract datetime from offset"
        with pytest.raises(TypeError, match=msg):
            off - dt
        assert 2 * off - off == off

        assert dt - offset2 == dt + _offset(-3)

    def test_multiply_by_zero(self, dt, offset1, offset2):
        # [FUNCTION DOC]
        # Multiplying an offset by zero gives an offset that neither adds nor subtracts: base date unchanged for both fixtures under addition and subtraction.
        assert dt - 0 * offset1 == dt
        assert dt + 0 * offset1 == dt
        assert dt - 0 * offset2 == dt
        assert dt + 0 * offset2 == dt

    def testRollback1(
        self,
        dt,
        _offset,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        # [FUNCTION DOC]
        # Rolling back the in-hours base returns it unchanged for the default-hours and multi-interval offsets with positive or negative step; for offsets whose hours exclude 10:00 it rolls to the previous closing time (14:30 on the previous day, 05:00, 06:30, 02:00 for the overnight/multi-interval ones). From midnight, default hours roll to 17:00 the previous day, the overnight offsets and offset9, offset10 stay on the midnight date (it is within an overnight interval), the two-interval offset rolls to 17:00 the previous day. A step-5 instance also keeps the in-hours base.
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt
        assert offset3.rollback(dt) == dt
        assert offset4.rollback(dt) == dt
        assert offset5.rollback(dt) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(dt) == datetime(2014, 7, 1, 5, 0)
        assert offset7.rollback(dt) == datetime(2014, 7, 1, 6, 30)
        assert offset8.rollback(dt) == dt
        assert offset9.rollback(dt) == dt
        assert offset10.rollback(dt) == datetime(2014, 7, 1, 2)

        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset2.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset3.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset4.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset5.rollback(datet) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(datet) == datet
        assert offset7.rollback(datet) == datet
        assert offset8.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset9.rollback(datet) == datet
        assert offset10.rollback(datet) == datet

        assert _offset(5).rollback(dt) == dt

    def testRollback2(self, _offset):
        # [FUNCTION DOC]
        # A step -3 offset rolling back Saturday 2014-07-05 15:00 gives Friday 2014-07-04 17:00.
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(
        self,
        dt,
        _offset,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        # [FUNCTION DOC]
        # Rolling forward counterpart of the rollback test with the same fixtures: in-hours base unchanged; others move to the next opening (11:00, 20:00, 21:30, 13:00); from midnight the default-hours offsets give 09:00 the same day, 11:00 for offset5, 09:00 for offset8 and midnight unchanged for offsets in overnight intervals. A step-5 instance keeps the in-hours base.
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt
        assert offset3.rollforward(dt) == dt
        assert offset4.rollforward(dt) == dt
        assert offset5.rollforward(dt) == datetime(2014, 7, 1, 11, 0)
        assert offset6.rollforward(dt) == datetime(2014, 7, 1, 20, 0)
        assert offset7.rollforward(dt) == datetime(2014, 7, 1, 21, 30)
        assert offset8.rollforward(dt) == dt
        assert offset9.rollforward(dt) == dt
        assert offset10.rollforward(dt) == datetime(2014, 7, 1, 13)

        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset3.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset4.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset5.rollforward(datet) == datetime(2014, 7, 1, 11)
        assert offset6.rollforward(datet) == datet
        assert offset7.rollforward(datet) == datet
        assert offset8.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset9.rollforward(datet) == datet
        assert offset10.rollforward(datet) == datet

        assert _offset(5).rollforward(dt) == dt

    def testRollforward2(self, _offset):
        # [FUNCTION DOC]
        # A step -3 offset rolling forward Saturday 2014-07-05 16:00 gives Monday 2014-07-07 09:00.
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    def test_roll_date_object(self):
        # [FUNCTION DOC]
        # Sunday 2014-07-06 15:00 rolls back to Friday 17:00 and forward to Monday 09:00 with default hours.
        # KNOWN ISSUE: the name says date object but the input is a datetime, so a pure date input is not exercised.
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    normalize_cases = []
    normalize_cases.append(
        (
            BusinessHour(normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        )
    )

    normalize_cases.append(
        (
            BusinessHour(-1, normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 30),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        )
    )

    normalize_cases.append(
        (
            BusinessHour(1, normalize=True, start="17:00", end="04:00"),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 2),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        )
    )

    @pytest.mark.parametrize("case", normalize_cases)
    def test_normalize(self, case):
        # [FUNCTION DOC]
        # Parametrized over normalize_cases (offsets with normalization enabled, with various hours and steps). Applying the offset directly to each base datetime must give the expected midnight-truncated date.
        offset, cases = case
        for dt, expected in cases.items():
            assert offset._apply(dt) == expected

    on_offset_cases = []
    on_offset_cases.append(
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start="10:00", end="15:00"),
            {
                datetime(2014, 7, 1, 9): False,
                datetime(2014, 7, 1, 10): True,
                datetime(2014, 7, 1, 15): True,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start="19:00", end="05:00"),
            {
                datetime(2014, 7, 1, 9, 0): False,
                datetime(2014, 7, 1, 10, 0): False,
                datetime(2014, 7, 1, 15): False,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12, 0): False,
                datetime(2014, 7, 6, 12, 0): False,
                datetime(2014, 7, 1, 19, 0): True,
                datetime(2014, 7, 2, 0, 0): True,
                datetime(2014, 7, 4, 23): True,
                datetime(2014, 7, 5, 1): True,
                datetime(2014, 7, 5, 5, 0): True,
                datetime(2014, 7, 6, 23, 0): False,
                datetime(2014, 7, 7, 3, 0): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"]),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
                datetime(2014, 7, 1, 12, 30): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start=["19:00", "23:00"], end=["21:00", "05:00"]),
            {
                datetime(2014, 7, 1, 9, 0): False,
                datetime(2014, 7, 1, 10, 0): False,
                datetime(2014, 7, 1, 15): False,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12, 0): False,
                datetime(2014, 7, 6, 12, 0): False,
                datetime(2014, 7, 1, 19, 0): True,
                datetime(2014, 7, 2, 0, 0): True,
                datetime(2014, 7, 4, 23): True,
                datetime(2014, 7, 5, 1): True,
                datetime(2014, 7, 5, 5, 0): True,
                datetime(2014, 7, 6, 23, 0): False,
                datetime(2014, 7, 7, 3, 0): False,
                datetime(2014, 7, 4, 22): False,
            },
        )
    )

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # [FUNCTION DOC]
        # Parametrized over on_offset_cases: each is an offset with a mapping from datetime to expected boolean; the offset's on-offset check must match each expectation (opening and closing times inclusive, minutes outside the interval false, weekends false, overnight and multiple-interval cases).
        offset, cases = case
        for dt, expected in cases.items():
            assert offset.is_on_offset(dt) == expected

    apply_cases = [
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 2, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 12),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 10),
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
            BusinessHour(4),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
        (
            BusinessHour(-1),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 1, 15, 30, 15),
                datetime(2014, 7, 1, 9, 30, 15): datetime(2014, 6, 30, 16, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 5): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 10),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 16),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 16),
                datetime(2014, 7, 7, 9): datetime(2014, 7, 4, 16),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 16, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 16, 30, 30),
            },
        ),
        (
            BusinessHour(-4),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 30, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 13, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 13, 30, 30),
            },
        ),
        (
            BusinessHour(start="13:00", end="16:00"),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 15, 30, 15): datetime(2014, 7, 2, 13, 30, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 14),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 14),
            },
        ),
        (
            BusinessHour(n=2, start="13:00", end="16:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 2, 14, 30): datetime(2014, 7, 3, 13, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 14, 30): datetime(2014, 7, 7, 13, 30),
                datetime(2014, 7, 4, 14, 30, 30): datetime(2014, 7, 7, 13, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start="13:00", end="16:00"),
            {
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 15): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 16): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 13, 30, 15): datetime(2014, 7, 1, 15, 30, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 15),
                datetime(2014, 7, 7, 11): datetime(2014, 7, 4, 15),
            },
        ),
        (
            BusinessHour(n=-3, start="10:00", end="16:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 11, 30): datetime(2014, 7, 1, 14, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 4, 10): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 16): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 12, 30): datetime(2014, 7, 3, 15, 30),
                datetime(2014, 7, 4, 12, 30, 30): datetime(2014, 7, 3, 15, 30, 30),
            },
        ),
        (
            BusinessHour(start="19:00", end="05:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 20),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 4, 30): datetime(2014, 7, 2, 19, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 1),
                datetime(2014, 7, 4, 10): datetime(2014, 7, 4, 20),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5, 0),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 5, 1),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 7, 19),
                datetime(2014, 7, 5, 4, 30): datetime(2014, 7, 7, 19, 30),
                datetime(2014, 7, 5, 4, 30, 30): datetime(2014, 7, 7, 19, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start="19:00", end="05:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 4),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 20): datetime(2014, 7, 2, 5),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 19, 30): datetime(2014, 7, 2, 4, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 23),
                datetime(2014, 7, 3, 6): datetime(2014, 7, 3, 4),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 4, 23),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 5, 3),
                datetime(2014, 7, 7, 19, 30): datetime(2014, 7, 5, 4, 30),
                datetime(2014, 7, 7, 19, 30, 30): datetime(2014, 7, 5, 4, 30, 30),
            },
        ),
        (
            BusinessHour(n=4, start="00:00", end="23:00"),
            {
                datetime(2014, 7, 3, 22): datetime(2014, 7, 4, 3),
                datetime(2014, 7, 4, 22): datetime(2014, 7, 7, 3),
                datetime(2014, 7, 3, 22, 30): datetime(2014, 7, 4, 3, 30),
                datetime(2014, 7, 3, 22, 20): datetime(2014, 7, 4, 3, 20),
                datetime(2014, 7, 4, 22, 30, 30): datetime(2014, 7, 7, 3, 30, 30),
                datetime(2014, 7, 4, 22, 30, 20): datetime(2014, 7, 7, 3, 30, 20),
            },
        ),
        (
            BusinessHour(n=-4, start="00:00", end="23:00"),
            {
                datetime(2014, 7, 4, 3): datetime(2014, 7, 3, 22),
                datetime(2014, 7, 7, 3): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 4, 3, 30): datetime(2014, 7, 3, 22, 30),
                datetime(2014, 7, 4, 3, 20): datetime(2014, 7, 3, 22, 20),
                datetime(2014, 7, 7, 3, 30, 30): datetime(2014, 7, 4, 22, 30, 30),
                datetime(2014, 7, 7, 3, 30, 20): datetime(2014, 7, 4, 22, 30, 20),
            },
        ),
        (
            BusinessHour(start=["09:00", "14:00"], end=["12:00", "18:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 17),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 1, 17, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 14),
                # out of business hours
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 9),
                datetime(2014, 7, 4, 17, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 17, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            BusinessHour(n=4, start=["09:00", "14:00"], end=["12:00", "18:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 17),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 17),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 14),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 11, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 11, 30, 30),
            },
        ),
        (
            BusinessHour(n=-4, start=["09:00", "14:00"], end=["12:00", "18:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 15): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 10),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 11),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 12),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 4, 12),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 14, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 14, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start=["19:00", "03:00"], end=["01:00", "05:00"]),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 4),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 20): datetime(2014, 7, 2, 5),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 4): datetime(2014, 7, 2, 1),
                datetime(2014, 7, 2, 19, 30): datetime(2014, 7, 2, 4, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 23),
                datetime(2014, 7, 3, 6): datetime(2014, 7, 3, 4),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 4, 23),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 5, 0),
                datetime(2014, 7, 7, 3, 30): datetime(2014, 7, 5, 0, 30),
                datetime(2014, 7, 7, 19, 30): datetime(2014, 7, 7, 4, 30),
                datetime(2014, 7, 7, 19, 30, 30): datetime(2014, 7, 7, 4, 30, 30),
            },
        ),
    ]

    # long business hours (see gh-26381)

    # multiple business hours

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case):
        # [FUNCTION DOC]
        # Parametrized over apply_cases: each is an offset (default hours with steps 1, 2, 3, -1 and others; custom hours; overnight; multiple intervals) with a mapping from start datetime to expected result, covering in-hours, after-hours, before-hours, weekend and fractional-second cases. Checked with the assertion helper.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    apply_large_n_cases = [
        (
            # A week later
            BusinessHour(40),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 8, 11),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 8, 13),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 8, 15),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 8, 16),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 9, 9),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 9, 11),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 9, 9),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 14, 9),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 14, 9),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 14, 9, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 14, 9, 30, 30),
            },
        ),
        (
            # 3 days and 1 hour before
            BusinessHour(-25),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 26, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 26, 12),
                datetime(2014, 7, 1, 9): datetime(2014, 6, 25, 16),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 25, 17),
                datetime(2014, 7, 3, 11): datetime(2014, 6, 30, 10),
                datetime(2014, 7, 3, 8): datetime(2014, 6, 27, 16),
                datetime(2014, 7, 3, 19): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 3, 23): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 4, 9): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 1, 16, 30),
                datetime(2014, 7, 7, 10, 30, 30): datetime(2014, 7, 2, 9, 30, 30),
            },
        ),
        (
            # 5 days and 3 hours later
            BusinessHour(28, start="21:00", end="02:00"),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 9, 0),
                datetime(2014, 7, 1, 22): datetime(2014, 7, 9, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 9, 21),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 10, 0),
                datetime(2014, 7, 3, 21): datetime(2014, 7, 11, 0),
                datetime(2014, 7, 4, 1): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 2): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 4, 3): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 5, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 7, 1): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 7, 23, 30): datetime(2014, 7, 15, 21, 30),
            },
        ),
        (
            # large n for multiple opening hours (3 days and 1 hour before)
            BusinessHour(n=-25, start=["09:00", "14:00"], end=["12:00", "19:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 26, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 26, 11),
                datetime(2014, 7, 1, 9): datetime(2014, 6, 25, 18),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 25, 19),
                datetime(2014, 7, 3, 11): datetime(2014, 6, 30, 10),
                datetime(2014, 7, 3, 8): datetime(2014, 6, 27, 18),
                datetime(2014, 7, 3, 19): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 3, 23): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 4, 9): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 1, 18),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 1, 18),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 1, 18, 30),
                datetime(2014, 7, 7, 10, 30, 30): datetime(2014, 7, 2, 9, 30, 30),
            },
        ),
        (
            # 5 days and 3 hours later
            BusinessHour(28, start=["21:00", "03:00"], end=["01:00", "04:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 9, 0),
                datetime(2014, 7, 1, 22): datetime(2014, 7, 9, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 9, 21),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 9, 23),
                datetime(2014, 7, 3, 21): datetime(2014, 7, 11, 0),
                datetime(2014, 7, 4, 1): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 2): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 3): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 21): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 14, 22),
                datetime(2014, 7, 5, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 7, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 7, 23, 30): datetime(2014, 7, 15, 21, 30),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_large_n_cases)
    def test_apply_large_n(self, case):
        # [FUNCTION DOC]
        # Parametrized over apply_large_n_cases: large positive and negative step counts (forty hours, minus twenty-five hours, and multiple-interval offsets with large steps) mapped to expected datetimes about a week or several days away.
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_nanoseconds(self):
        # [FUNCTION DOC]
        # Offsets with step 1 and -1; bases carry plus or minus five nanoseconds, and the expected results carry the same nanosecond component (nanoseconds are preserved through the offset).
        tests = [
            (
                BusinessHour(),
                {
                    Timestamp("2014-07-04 15:00") + Nano(5): Timestamp(
                        "2014-07-04 16:00"
                    )
                    + Nano(5),
                    Timestamp("2014-07-04 16:00") + Nano(5): Timestamp(
                        "2014-07-07 09:00"
                    )
                    + Nano(5),
                    Timestamp("2014-07-04 16:00") - Nano(5): Timestamp(
                        "2014-07-04 17:00"
                    )
                    - Nano(5),
                },
            ),
            (
                BusinessHour(-1),
                {
                    Timestamp("2014-07-04 15:00") + Nano(5): Timestamp(
                        "2014-07-04 14:00"
                    )
                    + Nano(5),
                    Timestamp("2014-07-04 10:00") + Nano(5): Timestamp(
                        "2014-07-04 09:00"
                    )
                    + Nano(5),
                    Timestamp("2014-07-04 10:00") - Nano(5): Timestamp(
                        "2014-07-03 17:00"
                    )
                    - Nano(5),
                },
            ),
        ]

        for offset, cases in tests:
            for base, expected in cases.items():
                assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("td_unit", ["s", "ms", "us", "ns"])
    def test_bday_ignores_timedeltas(self, unit, td_unit):
        # [FUNCTION DOC]
        # GH#55608. Parametrized over the index unit fixture and the time delta unit (seconds to nanoseconds). Builds a 12-hourly index from 2010-02-01 to 2010-02-10 and a business day offset carrying a three-hour time delta. Adding the offset to the index must equal the expected index (weekend values collapse onto Monday) in the finer of the two units, and equal element-wise addition.
        # KNOWN ISSUE: the test tests the business day offset (not business hour) and the comment marks an unresolved TODO for the unit conversion; it lives in the business hour test class only for convenience.
        # GH#55608
        idx = date_range("2010/02/01", "2010/02/10", freq="12h", unit=unit)
        td = Timedelta(3, unit="h").as_unit(td_unit)
        off = BDay(offset=td)
        t1 = idx + off

        exp_unit = tm.get_finest_unit(td.unit, idx.unit)

        expected = DatetimeIndex(
            [
                "2010-02-02 03:00:00",
                "2010-02-02 15:00:00",
                "2010-02-03 03:00:00",
                "2010-02-03 15:00:00",
                "2010-02-04 03:00:00",
                "2010-02-04 15:00:00",
                "2010-02-05 03:00:00",
                "2010-02-05 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-09 03:00:00",
                "2010-02-09 15:00:00",
                "2010-02-10 03:00:00",
                "2010-02-10 15:00:00",
                "2010-02-11 03:00:00",
            ],
            freq=None,
        ).as_unit(exp_unit)
        tm.assert_index_equal(t1, expected)

        # TODO(GH#55564): as_unit will be unnecessary
        pointwise = DatetimeIndex([x + off for x in idx]).as_unit(exp_unit)
        tm.assert_index_equal(pointwise, expected)

    def test_add_bday_offset_nanos(self):
        # [FUNCTION DOC]
        # GH#55608. A nanosecond-resolution 12-hourly index plus a business day offset with a three-nanosecond time delta equals the element-wise sum.
        # GH#55608
        idx = date_range("2010/02/01", "2010/02/10", freq="12h", unit="ns")
        off = BDay(offset=Timedelta(3, unit="ns"))

        result = idx + off
        expected = DatetimeIndex([x + off for x in idx])
        tm.assert_index_equal(result, expected)


class TestOpeningTimes:
    # [CLASS DOC]
    # Tests the private opening-time helpers used by the business hour offset. The comment records that the opening time depends on the sign of the step, not on its magnitude, nor on the end time.
    # opening time should be affected by sign of n, not by n's value and end
    opening_time_cases = [
        (
            [
                BusinessHour(),
                BusinessHour(n=2),
                BusinessHour(n=4),
                BusinessHour(end="10:00"),
                BusinessHour(n=2, end="4:00"),
                BusinessHour(n=4, end="15:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                # if timestamp is on opening time, next opening time is
                # as it is
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 3, 9),
                    datetime(2014, 7, 2, 9),
                ),
                # 2014-07-05 is saturday
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 8, 9),
                    datetime(2014, 7, 7, 9),
                ),
            },
        ),
        (
            [
                BusinessHour(start="11:15"),
                BusinessHour(n=2, start="11:15"),
                BusinessHour(n=3, start="11:15"),
                BusinessHour(start="11:15", end="10:00"),
                BusinessHour(n=2, start="11:15", end="4:00"),
                BusinessHour(n=3, start="11:15", end="15:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 11, 15),
                    datetime(2014, 6, 30, 11, 15),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 11, 15): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 2, 11, 15, 1): (
                    datetime(2014, 7, 3, 11, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 11, 15),
                    datetime(2014, 7, 3, 11, 15),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
            },
        ),
        (
            [
                BusinessHour(-1),
                BusinessHour(n=-2),
                BusinessHour(n=-4),
                BusinessHour(n=-1, end="10:00"),
                BusinessHour(n=-2, end="4:00"),
                BusinessHour(n=-4, end="15:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 3, 9),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 7, 9): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 8, 9),
                ),
            },
        ),
        (
            [
                BusinessHour(start="17:00", end="05:00"),
                BusinessHour(n=3, start="17:00", end="03:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 6, 30, 17),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 4, 17): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 3, 17),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 7, 17, 1): (
                    datetime(2014, 7, 8, 17),
                    datetime(2014, 7, 7, 17),
                ),
            },
        ),
        (
            [
                BusinessHour(-1, start="17:00", end="05:00"),
                BusinessHour(n=-2, start="17:00", end="03:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 6, 30, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 16, 59): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 3, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 7, 18): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 8, 17),
                ),
            },
        ),
        (
            [
                BusinessHour(start=["11:15", "15:00"], end=["13:00", "20:00"]),
                BusinessHour(n=3, start=["11:15", "15:00"], end=["12:00", "20:00"]),
                BusinessHour(start=["11:15", "15:00"], end=["13:00", "17:00"]),
                BusinessHour(n=2, start=["11:15", "15:00"], end=["12:00", "03:00"]),
                BusinessHour(n=3, start=["11:15", "15:00"], end=["13:00", "16:00"]),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 11, 15),
                    datetime(2014, 6, 30, 15),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 11, 15): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 2, 11, 15, 1): (
                    datetime(2014, 7, 2, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 11, 15),
                    datetime(2014, 7, 3, 15),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 7, 12): (
                    datetime(2014, 7, 7, 15),
                    datetime(2014, 7, 7, 11, 15),
                ),
            },
        ),
        (
            [
                BusinessHour(n=-1, start=["17:00", "08:00"], end=["05:00", "10:00"]),
                BusinessHour(n=-2, start=["08:00", "17:00"], end=["10:00", "03:00"]),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 8),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 8),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 8),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 8),
                    datetime(2014, 7, 2, 8),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 8),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 16, 59): (
                    datetime(2014, 7, 2, 8),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 8),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 7, 18): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 8, 8),
                ),
            },
        ),
    ]

    @pytest.mark.parametrize("case", opening_time_cases)
    def test_opening_time(self, case):
        # [FUNCTION DOC]
        # Parametrized over opening_time_cases: a list of offsets with different steps and end times, and a mapping from a datetime to (expected next opening time, expected previous opening time). Every offset in the list must give the same next and previous opening times; a timestamp exactly on an opening time returns itself for both, and weekends skip to Monday or back to Friday.
        _offsets, cases = case
        for offset in _offsets:
            for dt, (exp_next, exp_prev) in cases.items():
                assert offset._next_opening_time(dt) == exp_next
                assert offset._prev_opening_time(dt) == exp_prev
