# [MODULE DOC]
# Tests for comparison operators between Timestamp and other objects: numpy datetime64
# scalars and arrays, object arrays, python datetimes and dates, tz-naive versus
# tz-aware values, invalid types, NaT, out-of-bounds values and unsupported types that
# must defer via the not-implemented protocol.
from datetime import (
    datetime,
    timedelta,
)
import operator

import numpy as np
import pytest

from pandas import Timestamp
import pandas._testing as tm


class TestTimestampComparison:
    # [CLASS DOC]
    # Groups the Timestamp comparison tests. Stateless.
    def test_compare_non_nano_dt64(self):
        # [FUNCTION DOC]
        # Verifies a datetime64 scalar with a pre-nanosecond-bound date (year 1066)
        # compares equal to the Timestamp built from it, without raising during
        # conversion in the rich comparison.
        # don't raise when converting dt64 to Timestamp in __richcmp__
        dt = np.datetime64("1066-10-14")
        ts = Timestamp(dt)

        assert dt == ts

    def test_comparison_dt64_ndarray(self):
        # [FUNCTION DOC]
        # Verifies all six comparison operators between a Timestamp and a
        # two-dimensional datetime64 nanosecond array, in both operand orders, give
        # boolean arrays of the same shape with the expected elementwise values.
        ts = Timestamp("2021-01-01")
        ts2 = Timestamp("2019-04-05")
        arr = np.array([[ts.asm8, ts2.asm8]], dtype="M8[ns]")

        result = ts == arr
        expected = np.array([[True, False]], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = arr == ts
        tm.assert_numpy_array_equal(result, expected)

        result = ts != arr
        tm.assert_numpy_array_equal(result, ~expected)

        result = arr != ts
        tm.assert_numpy_array_equal(result, ~expected)

        result = ts2 < arr
        tm.assert_numpy_array_equal(result, expected)

        result = arr < ts2
        tm.assert_numpy_array_equal(result, np.array([[False, False]], dtype=bool))

        result = ts2 <= arr
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))

        result = arr <= ts2
        tm.assert_numpy_array_equal(result, ~expected)

        result = ts >= arr
        tm.assert_numpy_array_equal(result, np.array([[True, True]], dtype=bool))

        result = arr >= ts
        tm.assert_numpy_array_equal(result, np.array([[True, False]], dtype=bool))

    @pytest.mark.parametrize("reverse", [True, False])
    def test_comparison_dt64_ndarray_tzaware(self, reverse, comparison_op):
        # [FUNCTION DOC]
        # Parametrized over operand order and the comparison-operator fixture. A
        # tz-aware UTC timestamp compared with a naive datetime64 array: equality gives
        # all false, inequality all true, and any ordering operator raises a type error
        # about comparing naive and aware timestamps.
        ts = Timestamp("2021-01-01 00:00:00.00000", tz="UTC")
        arr = np.array([ts.asm8, ts.asm8], dtype="M8[ns]")

        left, right = ts, arr
        if reverse:
            left, right = arr, ts

        if comparison_op is operator.eq:
            expected = np.array([False, False], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        elif comparison_op is operator.ne:
            expected = np.array([True, True], dtype=bool)
            result = comparison_op(left, right)
            tm.assert_numpy_array_equal(result, expected)
        else:
            msg = "Cannot compare tz-naive and tz-aware timestamps"
            with pytest.raises(TypeError, match=msg):
                comparison_op(left, right)

    def test_comparison_object_array(self):
        # [FUNCTION DOC]
        # Regression test (GH 15183). Equality and inequality of an object-dtype array
        # of aware timestamps against a Timestamp work elementwise for one-dimensional
        # and two-dimensional arrays (keeping shape); an ordering comparison of an array
        # holding a naive timestamp with an aware one raises a type error.
        # GH#15183
        ts = Timestamp("2011-01-03 00:00:00-0500", tz="US/Eastern")
        other = Timestamp("2011-01-01 00:00:00-0500", tz="US/Eastern")
        naive = Timestamp("2011-01-01 00:00:00")

        arr = np.array([other, ts], dtype=object)
        res = arr == ts
        expected = np.array([False, True], dtype=bool)
        assert (res == expected).all()

        # 2D case
        arr = np.array([[other, ts], [ts, other]], dtype=object)
        res = arr != ts
        expected = np.array([[True, False], [False, True]], dtype=bool)
        assert res.shape == expected.shape
        assert (res == expected).all()

        # tzaware mismatch
        arr = np.array([naive], dtype=object)
        msg = "Cannot compare tz-naive and tz-aware timestamps"
        with pytest.raises(TypeError, match=msg):
            arr < ts

    def test_comparison(self):
        # [FUNCTION DOC]
        # Verifies the comparison operators on a Timestamp with itself, with an equal
        # python datetime, and with a Timestamp 100 nanoseconds later.
        # KNOWN ISSUE: the not-equal assertion against the later timestamp is repeated
        # twice and the not-equal check is never repeated in the reverse operand order;
        # one of the duplicates is redundant.
        # 5-18-2012 00:00:00.000
        stamp = 1337299200000000000

        val = Timestamp(stamp)

        assert val == val
        assert not val != val
        assert not val < val
        assert val <= val
        assert not val > val
        assert val >= val

        other = datetime(2012, 5, 18)
        assert val == other
        assert not val != other
        assert not val < other
        assert val <= other
        assert not val > other
        assert val >= other

        other = Timestamp(stamp + 100)

        assert val != other
        assert val != other
        assert val < other
        assert val <= other
        assert other > val
        assert other >= val

    def test_compare_invalid(self):
        # [FUNCTION DOC]
        # Regression test (GH 8058). A Timestamp compared for equality with strings,
        # floats, integers, lists, dicts and numpy scalars is never equal, and
        # inequality is always true; no error is raised.
        # GH#8058
        val = Timestamp("20130101 12:01:02")
        assert not val == "foo"
        assert not val == 10.0
        assert not val == 1
        assert not val == []
        assert not val == {"foo": 1}
        assert not val == np.float64(1)
        assert not val == np.int64(1)

        assert val != "foo"
        assert val != 10.0
        assert val != 1
        assert val != []
        assert val != {"foo": 1}
        assert val != np.float64(1)
        assert val != np.int64(1)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_compare_date(self, tz):
        # [FUNCTION DOC]
        # Parametrized over naive and US/Pacific timestamps. Comparing a Timestamp with
        # a plain date (either order): equality is false, inequality is true, and every
        # ordering operator raises a type error saying a Timestamp cannot be compared
        # with a date (disallowed since version 2.0; the first comment describing it as
        # deprecated is stale).
        # GH#36131 comparing Timestamp with date object is deprecated
        ts = Timestamp("2021-01-01 00:00:00.00000", tz=tz)
        dt = ts.to_pydatetime().date()
        # in 2.0 we disallow comparing pydate objects with Timestamps,
        #  following the stdlib datetime behavior.

        msg = "Cannot compare Timestamp with datetime.date"
        for left, right in [(ts, dt), (dt, ts)]:
            assert not left == right
            assert left != right

            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right
            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right

    def test_cant_compare_tz_naive_w_aware(self, utc_fixture):
        # [FUNCTION DOC]
        # Regression test (GH 1404). A naive and an aware (UTC fixture) timestamp, in
        # both operand orders: equality false, inequality true, orderings raise a type
        # error about naive and aware timestamps; equality with the other's python
        # datetime is also false.
        # see GH#1404
        a = Timestamp("3/12/2012")
        b = Timestamp("3/12/2012", tz=utc_fixture)

        msg = "Cannot compare tz-naive and tz-aware timestamps"
        assert not a == b
        assert a != b
        with pytest.raises(TypeError, match=msg):
            a < b
        with pytest.raises(TypeError, match=msg):
            a <= b
        with pytest.raises(TypeError, match=msg):
            a > b
        with pytest.raises(TypeError, match=msg):
            a >= b

        assert not b == a
        assert b != a
        with pytest.raises(TypeError, match=msg):
            b < a
        with pytest.raises(TypeError, match=msg):
            b <= a
        with pytest.raises(TypeError, match=msg):
            b > a
        with pytest.raises(TypeError, match=msg):
            b >= a

        assert not a == b.to_pydatetime()
        assert not a.to_pydatetime() == b

    def test_timestamp_compare_scalars(self):
        # [FUNCTION DOC]
        # Verifies comparisons between a zero-dimensional datetime64 and a current
        # Timestamp and NaT-valued Timestamp. For each operator paired with its mirrored
        # operator, the reflected comparison gives the same result as the direct one.
        # case where ndim == 0
        lhs = np.datetime64(datetime(2013, 12, 6))
        rhs = Timestamp("now")
        nat = Timestamp("nat")

        ops = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge", "eq": "eq", "ne": "ne"}

        for left, right in ops.items():
            left_f = getattr(operator, left)
            right_f = getattr(operator, right)
            expected = left_f(lhs, rhs)

            result = right_f(rhs, lhs)
            assert result == expected

            expected = left_f(rhs, nat)
            result = right_f(nat, rhs)
            assert result == expected

    def test_timestamp_compare_with_early_datetime(self):
        # [FUNCTION DOC]
        # Verifies comparisons with python datetimes that cannot be represented in
        # nanoseconds (minimum datetime, year 1600, year 2700): equality is false,
        # inequality true, and ordering is correct; and the minimum Timestamp's datetime
        # conversion (warnings suppressed) minus one microsecond is below the minimum
        # Timestamp.
        # e.g. datetime.min
        stamp = Timestamp("2012-01-01")

        assert not stamp == datetime.min
        assert not stamp == datetime(1600, 1, 1)
        assert not stamp == datetime(2700, 1, 1)
        assert stamp != datetime.min
        assert stamp != datetime(1600, 1, 1)
        assert stamp != datetime(2700, 1, 1)
        assert stamp > datetime(1600, 1, 1)
        assert stamp >= datetime(1600, 1, 1)
        assert stamp < datetime(2700, 1, 1)
        assert stamp <= datetime(2700, 1, 1)

        other = Timestamp.min.to_pydatetime(warn=False)
        assert other - timedelta(microseconds=1) < Timestamp.min

    def test_timestamp_compare_oob_dt64(self):
        # [FUNCTION DOC]
        # Verifies comparisons with out-of-bounds values. The minimum Timestamp is
        # greater than the minimum Timestamp rounded to microsecond datetime64 (the
        # comment notes numpy gets reversed comparison wrong); the maximum is greater
        # than its microsecond version but less than that plus one microsecond; and with
        # python datetimes in year 9999 and year 1 (GH 42794) the ordering is consistent
        # in both operand orders.
        us = np.timedelta64(1, "us")
        other = np.datetime64(Timestamp.min).astype("M8[us]")

        # This may change if the implementation bound is dropped to match
        #  DatetimeArray/DatetimeIndex GH#24124
        assert Timestamp.min > other
        # Note: numpy gets the reversed comparison wrong

        other = np.datetime64(Timestamp.max).astype("M8[us]")
        assert Timestamp.max > other  # not actually OOB
        assert other < Timestamp.max

        assert Timestamp.max < other + us
        # Note: numpy gets the reversed comparison wrong

        # GH-42794
        other = datetime(9999, 9, 9)
        assert Timestamp.min < other
        assert other > Timestamp.min
        assert Timestamp.max < other
        assert other > Timestamp.max

        other = datetime(1, 1, 1)
        assert Timestamp.max > other
        assert other < Timestamp.max
        assert Timestamp.min > other
        assert other < Timestamp.min

    def test_compare_zerodim_array(self, fixed_now_ts):
        # [FUNCTION DOC]
        # Regression test (GH 26916). A zero-dimensional datetime64 array compared with
        # a Timestamp using less-than yields a numpy true boolean scalar and with
        # greater-than a numpy false scalar (identity checked).
        # GH#26916
        ts = fixed_now_ts
        dt64 = np.datetime64("2016-01-01", "ns")
        arr = np.array(dt64)
        assert arr.ndim == 0

        result = arr < ts
        assert result is np.bool_(True)
        result = arr > ts
        assert result is np.bool_(False)


def test_rich_comparison_with_unsupported_type():
    # [FUNCTION DOC]
    # Verifies that a Timestamp comparison with an arbitrary user type returns the
    # not-implemented marker so the other operand's reflected methods are used, instead
    # of raising (GH 24011). Defines a local class whose ordering methods are defined to
    # behave like infinity; for both operand orders, strict and non-strict ordering
    # hold, equality is false and inequality true.
    # Comparisons with unsupported objects should return NotImplemented
    # (it previously raised TypeError, see #24011)

    class Inf:
        # [CLASS DOC]
        # HELPER: local test class modeling infinity. It is only ever equal to another
        # instance of itself, never less than anything, less-or-equal only to instances
        # of itself, greater than anything that is not an instance of itself, and always
        # greater-or-equal.
        def __lt__(self, o):
            # [FUNCTION DOC]
            # HELPER: less-than operator; always returns false regardless of the other
            # operand.
            return False

        def __le__(self, o):
            # [FUNCTION DOC]
            # HELPER: less-or-equal operator; returns true only when the other operand
            # is an instance of this class.
            return isinstance(o, Inf)

        def __gt__(self, o):
            # [FUNCTION DOC]
            # HELPER: greater-than operator; returns true when the other operand is not
            # an instance of this class.
            return not isinstance(o, Inf)

        def __ge__(self, o):
            # [FUNCTION DOC]
            # HELPER: greater-or-equal operator; always returns true.
            return True

        def __eq__(self, other) -> bool:
            # [FUNCTION DOC]
            # HELPER: equality operator; returns true only when the other operand is an
            # instance of this class, else false.
            return isinstance(other, Inf)

    inf = Inf()
    timestamp = Timestamp("2018-11-30")

    for left, right in [(inf, timestamp), (timestamp, inf)]:
        assert left > right or left < right
        assert left >= right or left <= right
        assert not left == right
        assert left != right
