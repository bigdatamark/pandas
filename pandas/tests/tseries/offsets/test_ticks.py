"""
Tests for offsets.Tick and subclasses
"""

# [MODULE DOC]
# Tests for the fixed-duration tick offsets (hour, minute, second, millisecond, microsecond, nanosecond): arithmetic between
# ticks, conversion from time deltas, equality and ordering, division and multiplication (including type promotion to
# finer or coarser tick classes), comparison with strings and time-delta-like values, and application to datetimes and timestamps.
# Defines a module-level list of the six tick classes used for parametrization. Some tests use Hypothesis and are marked slow.

from datetime import (
    datetime,
    timedelta,
)

from hypothesis import (
    assume,
    example,
    given,
)
import numpy as np
import pytest

from pandas._libs.tslibs.offsets import delta_to_tick

from pandas import (
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries import offsets
from pandas.tseries.offsets import (
    Hour,
    Micro,
    Milli,
    Minute,
    Nano,
    Second,
)

# ---------------------------------------------------------------------
# Test Helpers

tick_classes = [Hour, Minute, Second, Milli, Micro, Nano]


# ---------------------------------------------------------------------


def test_apply_ticks():
    # [FUNCTION DOC]
    # Adding an hour tick of 3 to an hour tick of 4 equals an hour tick of 7.
    result = offsets.Hour(3) + offsets.Hour(4)
    exp = offsets.Hour(7)
    assert result == exp


def test_delta_to_tick():
    # [FUNCTION DOC]
    # The conversion helper maps a three-day timedelta to an hour tick of 72 and a time delta of five nanoseconds to a nanosecond tick of 5.
    delta = timedelta(3)

    tick = delta_to_tick(delta)
    assert tick == offsets.Hour(72)

    td = Timedelta(nanoseconds=5)
    tick = delta_to_tick(td)
    assert tick == Nano(5)


@pytest.mark.slow
@pytest.mark.parametrize("cls", tick_classes)
@example(n=2, m=3)
@example(n=800, m=300)
@example(n=1000, m=5)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_add_sub(cls, n, m):
    # [FUNCTION DOC]
    # Property test (slow) over each tick class and integers from minus 999 to 999 (plus three explicit examples): sum of ticks n and m equals the tick of n plus m, and difference equals the tick of n minus m.
    # For all Tick subclasses and all integers n, m, we should have
    # tick(n) + tick(m) == tick(n+m)
    # tick(n) - tick(m) == tick(n-m)
    left = cls(n)
    right = cls(m)
    expected = cls(n + m)

    assert left + right == expected

    expected = cls(n - m)
    assert left - right == expected


@pytest.mark.slow
@pytest.mark.arm_slow
@pytest.mark.parametrize("cls", tick_classes)
@example(n=2, m=3)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_equality(cls, n, m):
    # [FUNCTION DOC]
    # Property test (slow) over each tick class and distinct integers n and m: ticks with different counts are unequal; ticks of equal count are equal (and not unequal); for nonzero n a tick differs from its negation.
    assume(m != n)
    # tick == tock iff tick.n == tock.n
    left = cls(n)
    right = cls(m)
    assert left != right

    right = cls(n)
    assert left == right
    assert not left != right

    if n != 0:
        assert cls(n) != cls(-n)


# ---------------------------------------------------------------------


def test_Hour():
    # [FUNCTION DOC]
    # Applying hour ticks (positive, negative, scaled by a multiplier on either side) to datetimes shifts by the expected hours; hour ticks add and subtract like counts, and different counts are unequal.
    assert_offset_equal(Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 1))
    assert_offset_equal(Hour(-1), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 2))
    assert_offset_equal(-1 * Hour(), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))

    assert Hour(3) + Hour(2) == Hour(5)
    assert Hour(3) - Hour(2) == Hour()

    assert Hour(4) != Hour(1)


def test_Minute():
    # [FUNCTION DOC]
    # Same checks as the hour test for minute ticks: shifting datetimes by minutes with positive, negative and scaled forms, tick addition and subtraction, and inequality of different counts.
    assert_offset_equal(Minute(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 1))
    assert_offset_equal(Minute(-1), datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Minute(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 2))
    assert_offset_equal(-1 * Minute(), datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))

    assert Minute(3) + Minute(2) == Minute(5)
    assert Minute(3) - Minute(2) == Minute()
    assert Minute(5) != Minute()


def test_Second():
    # [FUNCTION DOC]
    # Same checks as the hour test for second ticks: shifting datetimes, scaled forms, and tick addition and subtraction.
    assert_offset_equal(Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 1))
    assert_offset_equal(Second(-1), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(
        2 * Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 2)
    )
    assert_offset_equal(
        -1 * Second(), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1)
    )

    assert Second(3) + Second(2) == Second(5)
    assert Second(3) - Second(2) == Second()


def test_Millisecond():
    # [FUNCTION DOC]
    # Same checks for millisecond ticks (shift datetimes by thousands of microseconds, scaling by multiplier or count, tick addition and subtraction).
    assert_offset_equal(
        Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1000)
    )
    assert_offset_equal(
        Milli(-1), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1)
    )
    assert_offset_equal(
        Milli(2), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000)
    )
    assert_offset_equal(
        2 * Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000)
    )
    assert_offset_equal(
        -1 * Milli(), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1)
    )

    assert Milli(3) + Milli(2) == Milli(5)
    assert Milli(3) - Milli(2) == Milli()


def test_MillisecondTimestampArithmetic():
    # [FUNCTION DOC]
    # A one-millisecond tick shifts timestamps forward one millisecond, and its negative shifts back, verified with the shared offset assertion helper.
    assert_offset_equal(
        Milli(), Timestamp("2010-01-01"), Timestamp("2010-01-01 00:00:00.001")
    )
    assert_offset_equal(
        Milli(-1), Timestamp("2010-01-01 00:00:00.001"), Timestamp("2010-01-01")
    )


def test_Microsecond():
    # [FUNCTION DOC]
    # Same checks for microsecond ticks: datetime shifts of one and two microseconds, negative and scaled forms, tick addition and subtraction.
    assert_offset_equal(Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1))
    assert_offset_equal(
        Micro(-1), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1)
    )

    assert_offset_equal(
        2 * Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2)
    )
    assert_offset_equal(
        -1 * Micro(), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1)
    )

    assert Micro(3) + Micro(2) == Micro(5)
    assert Micro(3) - Micro(2) == Micro()


def test_NanosecondGeneric():
    # [FUNCTION DOC]
    # Adding a 10 nanosecond tick to a timestamp (either operand order) sets the nanosecond field of the result to 10 starting from zero.
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert timestamp.nanosecond == 0

    result = timestamp + Nano(10)
    assert result.nanosecond == 10

    reverse_result = Nano(10) + timestamp
    assert reverse_result.nanosecond == 10


def test_Nanosecond():
    # [FUNCTION DOC]
    # Shifting a timestamp by nanosecond ticks (positive, negative, scaled) matches adding a numpy timedelta; tick addition and subtraction; and regression cases (GH9284) where mixed nanosecond and microsecond ticks sum to a nanosecond tick of the combined count.
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert_offset_equal(Nano(), timestamp, timestamp + np.timedelta64(1, "ns"))
    assert_offset_equal(Nano(-1), timestamp + np.timedelta64(1, "ns"), timestamp)
    assert_offset_equal(2 * Nano(), timestamp, timestamp + np.timedelta64(2, "ns"))
    assert_offset_equal(-1 * Nano(), timestamp + np.timedelta64(1, "ns"), timestamp)

    assert Nano(3) + Nano(2) == Nano(5)
    assert Nano(3) - Nano(2) == Nano()

    # GH9284
    assert Nano(1) + Nano(10) == Nano(11)
    assert Nano(5) + Micro(1) == Nano(1005)
    assert Micro(5) + Nano(1) == Nano(5001)


@pytest.mark.parametrize(
    "kls, expected",
    [
        (Hour, Timedelta(hours=5)),
        (Minute, Timedelta(hours=2, minutes=3)),
        (Second, Timedelta(hours=2, seconds=3)),
        (Milli, Timedelta(hours=2, milliseconds=3)),
        (Micro, Timedelta(hours=2, microseconds=3)),
        (Nano, Timedelta(hours=2, nanoseconds=3)),
    ],
)
def test_tick_addition(kls, expected):
    # [FUNCTION DOC]
    # Parametrized over tick class and the expected total with a time delta of two hours. For a tick of count 3 and for the timedelta given as a pandas timedelta, a Python timedelta and a numpy timedelta, adding in either order returns a pandas timedelta equal to the expected sum.
    offset = kls(3)
    td = Timedelta(hours=2)

    for other in [td, td.to_pytimedelta(), td.to_timedelta64()]:
        result = offset + other
        assert isinstance(result, Timedelta)
        assert result == expected

        result = other + offset
        assert isinstance(result, Timedelta)
        assert result == expected


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_division(cls):
    # [FUNCTION DOC]
    # Parametrized over tick classes. For a tick of 10: dividing by a tick of 5 gives 2; by integer or float 2 gives a tick of 5; dividing by its own timedelta (or numpy form) gives 1; dividing by a nanosecond tick equals the ratio of timedeltas. For non-nanosecond classes dividing by 1000 gives a tick of a different, finer class with the correct duration. For classes finer than one second dividing by 0.001 gives a different, coarser tick class with the correct duration.
    off = cls(10)

    assert off / cls(5) == 2
    assert off / 2 == cls(5)
    assert off / 2.0 == cls(5)

    assert off / off._as_pd_timedelta == 1
    assert off / off._as_pd_timedelta.to_timedelta64() == 1

    assert off / Nano(1) == off._as_pd_timedelta / Nano(1)._as_pd_timedelta

    if cls is not Nano:
        # A case where we end up with a smaller class
        result = off / 1000
        assert isinstance(result, offsets.Tick)
        assert not isinstance(result, cls)
        assert result._as_pd_timedelta == off._as_pd_timedelta / 1000

    if cls._nanos_inc < Timedelta(seconds=1)._value:
        # Case where we end up with a bigger class
        result = off / 0.001
        assert isinstance(result, offsets.Tick)
        assert not isinstance(result, cls)
        assert result._as_pd_timedelta == off._as_pd_timedelta / 0.001


def test_tick_mul_float():
    # [FUNCTION DOC]
    # Multiplying a two-microsecond tick by 1.5 gives a microsecond tick of 3 (type retained); by 1.25 gives a nanosecond tick of 2500 (promoted to the finer class).
    off = Micro(2)

    # Case where we retain type
    result = off * 1.5
    expected = Micro(3)
    assert result == expected
    assert isinstance(result, Micro)

    # Case where we bump up to the next type
    result = off * 1.25
    expected = Nano(2500)
    assert result == expected
    assert isinstance(result, Nano)


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_rdiv(cls):
    # [FUNCTION DOC]
    # Parametrized over tick classes. Dividing an int or float by a tick raises a type error about unsupported operand types naming the tick's qualified class. A numpy timedelta scaled by 2.5 divided by the tick gives 2.5; for non-nanosecond classes a Python timedelta doubled divided by the tick gives 2; an array of two numpy timedeltas divided by the tick gives the float array of ratios.
    off = cls(10)
    delta = off._as_pd_timedelta
    td64 = delta.to_timedelta64()
    instance__type = ".".join([cls.__module__, cls.__name__])
    msg = (
        f"unsupported operand type\\(s\\) for \\/: 'int'|'float' and '{instance__type}'"
    )

    with pytest.raises(TypeError, match=msg):
        2 / off
    with pytest.raises(TypeError, match=msg):
        2.0 / off

    assert (td64 * 2.5) / off == 2.5

    if cls is not Nano:
        # skip pytimedelta for Nano since it gets dropped
        assert (delta.to_pytimedelta() * 2) / off == 2

    result = np.array([2 * td64, td64]) / off
    expected = np.array([2.0, 1.0])
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("cls1", tick_classes)
@pytest.mark.parametrize("cls2", tick_classes)
def test_tick_zero(cls1, cls2):
    # [FUNCTION DOC]
    # Parametrized over all pairs of tick classes. Zero ticks of any classes are equal, their sum equals the zero tick of the first class; adding a zero tick of any class to a count-2 tick of a non-nano first class keeps it unchanged; for the nanosecond class adding a nano zero keeps it unchanged.
    assert cls1(0) == cls2(0)
    assert cls1(0) + cls2(0) == cls1(0)

    if cls1 is not Nano:
        assert cls1(2) + cls2(0) == cls1(2)

    if cls1 is Nano:
        assert cls1(2) + Nano(0) == cls1(2)


@pytest.mark.parametrize("cls", tick_classes)
def test_tick_equalities(cls):
    # [FUNCTION DOC]
    # A default tick of any class equals the same class with count 1.
    assert cls() == cls(1)


@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks(cls):
    # [FUNCTION DOC]
    # Ordering and equality comparisons between ticks of the same class (3 versus 4) behave as for the counts.
    three = cls(3)
    four = cls(4)

    assert three < cls(4)
    assert cls(3) < four
    assert four > cls(3)
    assert cls(4) > three
    assert cls(3) == cls(3)
    assert cls(3) != cls(4)


@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks_to_strs(cls):
    # [FUNCTION DOC]
    # Regression test (GH#23524). A tick compared for equality with an arbitrary string is never equal (either operand order). Ordering comparisons (less, less-equal, greater, greater-equal) between a tick and a string, in either order, raise a type error mentioning the unsupported comparison between the str and the tick type.
    # GH#23524
    off = cls(19)

    # These tests should work with any strings, but we particularly are
    #  interested in "infer" as that comparison is convenient to make in
    #  Datetime/Timedelta Array/Index constructors
    assert not off == "infer"
    assert not "foo" == off

    instance_type = ".".join([cls.__module__, cls.__name__])
    msg = (
        "'<'|'<='|'>'|'>=' not supported between instances of "
        f"'str' and '{instance_type}'|'{instance_type}' and 'str'"
    )

    for left, right in [("infer", off), (off, "infer")]:
        with pytest.raises(TypeError, match=msg):
            left < right
        with pytest.raises(TypeError, match=msg):
            left <= right
        with pytest.raises(TypeError, match=msg):
            left > right
        with pytest.raises(TypeError, match=msg):
            left >= right


@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks_to_timedeltalike(cls):
    # [FUNCTION DOC]
    # A tick of 19 compared to its equivalent pandas timedelta and numpy timedelta (and a Python timedelta for non-nano classes) is equal, not unequal, neither less nor greater, and both less-or-equal and greater-or-equal.
    off = cls(19)

    td = off._as_pd_timedelta

    others = [td, td.to_timedelta64()]
    if cls is not Nano:
        others.append(td.to_pytimedelta())

    for other in others:
        assert off == other
        assert not off != other
        assert not off < other
        assert not off > other
        assert off <= other
        assert off >= other
