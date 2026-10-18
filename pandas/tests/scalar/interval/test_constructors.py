# [MODULE DOC]
# Tests for validation errors raised by the Interval constructor: invalid endpoint
# types, invalid closed option, reversed endpoints and mismatched time zones.

import pytest

from pandas import (
    Interval,
    Period,
    Timestamp,
)


class TestIntervalConstructors:
    # [CLASS DOC]
    # Groups constructor error tests for Interval.
    @pytest.mark.parametrize(
        "left, right",
        [
            ("a", "z"),
            (("a", "b"), ("c", "d")),
            (list("AB"), list("ab")),
            (Interval(0, 1), Interval(1, 2)),
            (Period("2018Q1", freq="Q"), Period("2018Q1", freq="Q")),
        ],
    )
    def test_construct_errors(self, left, right):
        # [FUNCTION DOC]
        # GH#23013. Constructing an interval from endpoints that are not numeric, timestamp or timedelta
        # (strings, tuples, lists, other intervals, periods) must raise a value error listing the allowed endpoint kinds.
        # Parametrized over pairs of left and right endpoints.
        # GH#23013
        msg = "Only numeric, Timestamp and Timedelta endpoints are allowed"
        with pytest.raises(ValueError, match=msg):
            Interval(left, right)

    def test_constructor_errors(self):
        # [FUNCTION DOC]
        # An unrecognized closed option must raise a value error naming the invalid option, and a left endpoint
        # greater than the right endpoint must raise a value error saying left must be less than or equal to right.
        msg = "invalid option for 'closed': foo"
        with pytest.raises(ValueError, match=msg):
            Interval(0, 1, closed="foo")

        msg = "left side of interval must be <= right side"
        with pytest.raises(ValueError, match=msg):
            Interval(1, 0)

    @pytest.mark.parametrize(
        "tz_left, tz_right", [(None, "UTC"), ("UTC", None), ("UTC", "US/Eastern")]
    )
    def test_constructor_errors_tz(self, tz_left, tz_right):
        # [FUNCTION DOC]
        # GH#18538. Builds timestamps with the given left and right time zones (None means naive).
        # Mixing naive and aware endpoints must raise a type error about comparing naive and aware timestamps;
        # two different time zones must raise a value error saying both sides must share a time zone.
        # GH#18538
        left = Timestamp("2017-01-01", tz=tz_left)
        right = Timestamp("2017-01-02", tz=tz_right)

        if tz_left is None or tz_right is None:
            error = TypeError
            msg = "Cannot compare tz-naive and tz-aware timestamps"
        else:
            error = ValueError
            msg = "left and right must have the same time zone"
        with pytest.raises(error, match=msg):
            Interval(left, right)
