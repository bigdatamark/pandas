# [MODULE DOC]
# Tests for membership checks (the in operator) on the Interval scalar, for both
# scalar values and other intervals, including infinite, zero-length and mixed-type endpoints.

import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


class TestContains:
    # [CLASS DOC]
    # Groups containment tests for Interval.
    def test_contains(self):
        # [FUNCTION DOC]
        # For a default (right-closed) interval a scalar is contained when strictly above the left end and at most the
        # right end. Checks the same for a both-closed interval (both ends included) and a neither-closed interval (both excluded).
        interval = Interval(0, 1)
        assert 0.5 in interval
        assert 1 in interval
        assert 0 not in interval

        interval_both = Interval(0, 1, "both")
        assert 0 in interval_both
        assert 1 in interval_both

        interval_neither = Interval(0, 1, closed="neither")
        assert 0 not in interval_neither
        assert 0.5 in interval_neither
        assert 1 not in interval_neither

    def test_contains_interval(self, inclusive_endpoints_fixture):
        # [FUNCTION DOC]
        # An interval always contains itself. A both-closed interval contains an interval with the same endpoints and any closed side;
        # the reverse holds only when the other interval is also both-closed. Parametrized over the inclusive endpoint fixture.
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(0, 1, inclusive_endpoints_fixture)
        assert interval1 in interval1
        assert interval2 in interval2
        assert interval2 in interval1
        assert interval1 not in interval2 or inclusive_endpoints_fixture == "both"

    def test_contains_infinite_length(self):
        # [FUNCTION DOC]
        # A finite both-closed interval is contained in an open interval from negative to positive infinity, but not the reverse.
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(float("-inf"), float("inf"), "neither")
        assert interval1 in interval2
        assert interval2 not in interval1

    def test_contains_zero_length(self):
        # [FUNCTION DOC]
        # Zero-length intervals: a point inside a range is contained in it, a point outside is not,
        # two different points do not contain each other, and a range is not contained in a point.
        interval1 = Interval(0, 1, "both")
        interval2 = Interval(-1, -1, "both")
        interval3 = Interval(0.5, 0.5, "both")
        assert interval2 not in interval1
        assert interval3 in interval1
        assert interval2 not in interval3 and interval3 not in interval2
        assert interval1 not in interval2 and interval1 not in interval3

    @pytest.mark.parametrize(
        "type1",
        [
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    @pytest.mark.parametrize(
        "type2",
        [
            (0, 1),
            (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)),
            (Timedelta("0h"), Timedelta("1h")),
        ],
    )
    def test_contains_mixed_types(self, type1, type2):
        # [FUNCTION DOC]
        # Parametrized over two endpoint pairs each of numbers, timestamps or timedeltas. When both pairs have the same kind,
        # containment of the first interval in the second must hold; otherwise it must raise a type error saying less-or-equal is not supported between the instances.
        interval1 = Interval(*type1)
        interval2 = Interval(*type2)
        if type1 == type2:
            assert interval1 in interval2
        else:
            msg = "^'<=' not supported between instances of"
            with pytest.raises(TypeError, match=msg):
                interval1 in interval2
