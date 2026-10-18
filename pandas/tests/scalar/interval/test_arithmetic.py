# [MODULE DOC]
# Tests for arithmetic and comparison operators on the Interval scalar.
# Covers adding, subtracting, multiplying and dividing intervals by numbers and
# time deltas, error cases for unsupported operands, and equality/ordering.
# Uses the shared closed fixture for the closed-side parametrization.

from datetime import timedelta

import numpy as np
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestIntervalArithmetic:
    # [CLASS DOC]
    # Groups tests of arithmetic operators between an Interval and scalar operands.
    def test_interval_add(self, closed):
        # [FUNCTION DOC]
        # Verifies that adding an integer to an interval shifts both endpoints, for all closed
        # sides, on the left, on the right, and as an in-place update; the closed side is kept.
        # Adding an interval to itself or adding a string must raise a type error about unsupported operands.
        interval = Interval(0, 1, closed=closed)
        expected = Interval(1, 2, closed=closed)

        result = interval + 1
        assert result == expected

        result = 1 + interval
        assert result == expected

        result = interval
        result += 1
        assert result == expected

        msg = r"unsupported operand type\(s\) for \+"
        with pytest.raises(TypeError, match=msg):
            interval + interval

        with pytest.raises(TypeError, match=msg):
            interval + "foo"

    def test_interval_sub(self, closed):
        # [FUNCTION DOC]
        # Verifies subtracting an integer shifts both endpoints down (also in-place) and keeps the closed side.
        # Subtracting an interval from an interval, or a string from an interval, must raise a type error.
        interval = Interval(0, 1, closed=closed)
        expected = Interval(-1, 0, closed=closed)

        result = interval - 1
        assert result == expected

        result = interval
        result -= 1
        assert result == expected

        msg = r"unsupported operand type\(s\) for -"
        with pytest.raises(TypeError, match=msg):
            interval - interval

        with pytest.raises(TypeError, match=msg):
            interval - "foo"

    def test_interval_mult(self, closed):
        # [FUNCTION DOC]
        # Verifies multiplying an interval by an integer scales both endpoints, from either side and in-place.
        # Multiplying two intervals raises a type error for unsupported operands; multiplying by a string
        # raises a type error about being unable to multiply a sequence by a non-integer.
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 2, closed=closed)

        result = interval * 2
        assert result == expected

        result = 2 * interval
        assert result == expected

        result = interval
        result *= 2
        assert result == expected

        msg = r"unsupported operand type\(s\) for \*"
        with pytest.raises(TypeError, match=msg):
            interval * interval

        msg = r"can\'t multiply sequence by non-int"
        with pytest.raises(TypeError, match=msg):
            interval * "foo"

    def test_interval_div(self, closed):
        # [FUNCTION DOC]
        # Verifies true division of an interval by a float divides both endpoints (also in-place) and keeps the closed side.
        # Dividing an interval by an interval or by a string must raise a type error for unsupported operands.
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 0.5, closed=closed)

        result = interval / 2.0
        assert result == expected

        result = interval
        result /= 2.0
        assert result == expected

        msg = r"unsupported operand type\(s\) for /"
        with pytest.raises(TypeError, match=msg):
            interval / interval

        with pytest.raises(TypeError, match=msg):
            interval / "foo"

    def test_interval_floordiv(self, closed):
        # [FUNCTION DOC]
        # Verifies floor division of an interval by an integer floor-divides both endpoints (also in-place).
        # Floor-dividing by an interval or a string must raise a type error for unsupported operands.
        interval = Interval(1, 2, closed=closed)
        expected = Interval(0, 1, closed=closed)

        result = interval // 2
        assert result == expected

        result = interval
        result //= 2
        assert result == expected

        msg = r"unsupported operand type\(s\) for //"
        with pytest.raises(TypeError, match=msg):
            interval // interval

        with pytest.raises(TypeError, match=msg):
            interval // "foo"

    @pytest.mark.parametrize("method", ["__add__", "__sub__"])
    @pytest.mark.parametrize(
        "interval",
        [
            Interval(
                Timestamp("2017-01-01 00:00:00"), Timestamp("2018-01-01 00:00:00")
            ),
            Interval(Timedelta(days=7), Timedelta(days=14)),
        ],
    )
    @pytest.mark.parametrize(
        "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
    )
    def test_time_interval_add_subtract_timedelta(self, interval, delta, method):
        # [FUNCTION DOC]
        # Regression test for GH 32023. Parametrized over add and subtract special methods, timestamp-based
        # and timedelta-based intervals, and three delta kinds (Timedelta, standard library timedelta, numpy timedelta).
        # The result of applying the operation to the interval must equal an interval built from the operation applied
        # to each endpoint separately.
        # https://github.com/pandas-dev/pandas/issues/32023
        result = getattr(interval, method)(delta)
        left = getattr(interval.left, method)(delta)
        right = getattr(interval.right, method)(delta)
        expected = Interval(left, right)

        assert result == expected

    @pytest.mark.parametrize("interval", [Interval(1, 2), Interval(1.0, 2.0)])
    @pytest.mark.parametrize(
        "delta", [Timedelta(days=7), timedelta(7), np.timedelta64(7, "D")]
    )
    @pytest.mark.filterwarnings(
        "ignore:.*'generic' unit for NumPy timedelta:DeprecationWarning"
    )
    def test_numeric_interval_add_timedelta_raises(self, interval, delta):
        # [FUNCTION DOC]
        # Regression test for GH 32023. Adding a time delta (any of three kinds) to a purely numeric interval
        # (integer or float endpoints), in either operand order, must raise a type error or value error
        # whose message mentions unsupported operands, incompatible operands, or the allowed endpoint kinds.
        # Deprecation warnings about generic numpy timedelta units are filtered.
        # https://github.com/pandas-dev/pandas/issues/32023
        msg = "|".join(
            [
                "unsupported operand",
                "cannot use operands",
                "Only numeric, Timestamp and Timedelta endpoints are allowed",
            ]
        )
        with pytest.raises((TypeError, ValueError), match=msg):
            interval + delta

        with pytest.raises((TypeError, ValueError), match=msg):
            delta + interval

    @pytest.mark.parametrize(
        "delta",
        [timedelta(0), np.timedelta64(0, "ns"), Timedelta(0)],
    )
    def test_timedelta_add_timestamp_interval(self, delta):
        # [FUNCTION DOC]
        # Adding a zero-length delta (three kinds) to a timestamp interval, in either operand order, must return an
        # interval equal to the original.
        expected = Interval(Timestamp("2020-01-01"), Timestamp("2020-02-01"))

        result = delta + expected
        assert result == expected

        result = expected + delta
        assert result == expected


class TestIntervalComparisons:
    # [CLASS DOC]
    # Groups tests of equality and ordering comparisons between intervals and other objects.
    def test_interval_equal(self):
        # [FUNCTION DOC]
        # Intervals are equal only when endpoints and closed side match (default closed side is right);
        # an interval is not equal to a plain number.
        assert Interval(0, 1) == Interval(0, 1, closed="right")
        assert Interval(0, 1) != Interval(0, 1, closed="left")
        assert Interval(0, 1) != 0

    def test_interval_comparison(self):
        # [FUNCTION DOC]
        # Ordering an interval against a non-interval must raise a type error naming the unsupported comparison.
        # Between intervals, ordering compares left endpoints first, then right endpoints, and supports less-than,
        # less-or-equal, greater-than and greater-or-equal.
        msg = (
            "'<' not supported between instances of "
            "'pandas._libs.interval.Interval' and 'int'"
        )
        with pytest.raises(TypeError, match=msg):
            Interval(0, 1) < 2

        assert Interval(0, 1) < Interval(1, 2)
        assert Interval(0, 1) < Interval(0, 2)
        assert Interval(0, 1) < Interval(0.5, 1.5)
        assert Interval(0, 1) <= Interval(0, 1)
        assert Interval(0, 1) > Interval(-1, 2)
        assert Interval(0, 1) >= Interval(0, 1)

    def test_equality_comparison_broadcasts_over_array(self):
        # [FUNCTION DOC]
        # Regression test for GH 35931. Comparing an interval for equality to an object array containing the
        # interval twice must broadcast and return a boolean array of two true values.
        # https://github.com/pandas-dev/pandas/issues/35931
        interval = Interval(0, 1)
        arr = np.array([interval, interval])
        result = interval == arr
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
