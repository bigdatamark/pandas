# [MODULE DOC]
# Tests for the overlaps method of the Interval scalar, across numeric, timedelta and timestamp-with-timedelta-shift
# endpoints and all combinations of closed sides. Defines a local fixture of (start, shift) pairs.

import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


@pytest.fixture(
    params=[
        (Timedelta("0 days"), Timedelta("1 day")),
        (Timestamp("2018-01-01"), Timedelta("1 day")),
        (0, 1),
    ],
    ids=lambda x: type(x[0]).__name__,
)
def start_shift(request):
    # [FUNCTION DOC]
    # Fixture parametrized over three (start, shift) pairs: timedelta start with timedelta shift, timestamp start with
    # timedelta shift, and integer start with integer shift. Returns the pair so tests can build endpoints by adding multiples of the shift to the start. Test ids come from the start type name.
    """
    Fixture for generating intervals of types from a start value and a shift
    value that can be added to start to generate an endpoint
    """
    return request.param


class TestOverlaps:
    # [CLASS DOC]
    # Groups tests of Interval overlap detection.
    def test_overlaps_self(self, start_shift, closed):
        # [FUNCTION DOC]
        # Any interval overlaps itself, for every closed side.
        start, shift = start_shift
        interval = Interval(start, start + shift, closed)
        assert interval.overlaps(interval)

    def test_overlaps_nested(self, start_shift, closed, other_closed):
        # [FUNCTION DOC]
        # An interval strictly nested inside another overlaps it, regardless of the closed side of either.
        start, shift = start_shift
        interval1 = Interval(start, start + 3 * shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # nested intervals should always overlap
        assert interval1.overlaps(interval2)

    def test_overlaps_disjoint(self, start_shift, closed, other_closed):
        # [FUNCTION DOC]
        # Two intervals separated by a gap never overlap, regardless of the closed sides.
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + 2 * shift, start + 3 * shift, closed)

        # disjoint intervals should never overlap
        assert not interval1.overlaps(interval2)

    def test_overlaps_endpoint(self, start_shift, closed, other_closed):
        # [FUNCTION DOC]
        # Two intervals touching at one shared endpoint overlap only when the first is closed on its right and the second on its left;
        # the expected value is computed from those two closed flags.
        start, shift = start_shift
        interval1 = Interval(start, start + shift, other_closed)
        interval2 = Interval(start + shift, start + 2 * shift, closed)

        # overlap if shared endpoint is closed for both (overlap at a point)
        result = interval1.overlaps(interval2)
        expected = interval1.closed_right and interval2.closed_left
        assert result == expected

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(self, other):
        # [FUNCTION DOC]
        # Passing something other than an interval (a number, boolean, string, timedelta or timestamp) must raise a type error stating that the other argument must be an Interval and giving the type name received.
        interval = Interval(0, 1)
        msg = f"`other` must be an Interval, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval.overlaps(other)
