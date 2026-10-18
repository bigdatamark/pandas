# [MODULE DOC]
# Tests behavior shared by all offset classes, parametrized through a fixture listing 25 offset classes (business, custom business, month, semi-month, quarter, week, year, fiscal and generic date offset): out-of-range application with time zone preservation, equality of fresh instances, reflected subtraction and addition, subtraction errors, multiplication, and comparison with strings.
# Defines a private factory helper to build an instance for any class. Uses the time zone fixtures from the test configuration. No import-time side effects.
from datetime import datetime

from dateutil.tz.tz import tzlocal
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas.compat import (
    IS64,
    WASM,
    is_platform_windows,
)

from pandas.tseries.offsets import (
    FY5253,
    BDay,
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BusinessHour,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CDay,
    CustomBusinessHour,
    DateOffset,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    MonthEnd,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)


def _get_offset(klass, value=1, normalize=False):
    # [FUNCTION DOC]
    # HELPER: factory used by the shared tests. Parameters: klass is an offset class; value is the step count (default 1); normalize is the normalization flag (default false).
    # Builds an instance with class-specific required arguments: the 52-53 week fiscal year and its quarter variant get starting month 1, weekday 1, variation "last" (the quarter also an extra-week quarter 1); last week of month gets weekday 5; week-of-month gets week 1 and weekday 5; week gets weekday 5; the generic date offset gets the value as its days argument; every other class receives the value positionally. Returns the new instance.
    # create instance from offset class
    if klass is FY5253:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is FY5253Quarter:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            qtr_with_extra_week=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is LastWeekOfMonth:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is WeekOfMonth:
        klass = klass(n=value, week=1, weekday=5, normalize=normalize)
    elif klass is Week:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is DateOffset:
        klass = klass(days=value, normalize=normalize)
    else:
        klass = klass(value, normalize=normalize)
    return klass


@pytest.fixture(
    params=[
        BDay,
        BusinessHour,
        BMonthEnd,
        BMonthBegin,
        BQuarterEnd,
        BQuarterBegin,
        BYearEnd,
        BYearBegin,
        CDay,
        CustomBusinessHour,
        CBMonthEnd,
        CBMonthBegin,
        MonthEnd,
        MonthBegin,
        SemiMonthBegin,
        SemiMonthEnd,
        QuarterEnd,
        LastWeekOfMonth,
        WeekOfMonth,
        Week,
        YearBegin,
        YearEnd,
        FY5253,
        FY5253Quarter,
        DateOffset,
    ]
)
def _offset(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture yielding each offset class in turn from the 25-class list. Returns the class (not an instance).
    return request.param


@pytest.mark.skipif(WASM, reason="OverflowError received on WASM")
def test_apply_out_of_range(request, tz_naive_fixture, _offset):
    # [FUNCTION DOC]
    # Skipped on WASM (overflow errors there). Parametrized by the time zone fixture (including naive) and offset class. Builds an offset with a very large step (100000 for the hour-based classes, 10000 otherwise), adds it to a naive 2008-01-01 timestamp (result must be a datetime without a timezone), then to a tz-aware timestamp (result must be a datetime), and finally checks the timezone string is preserved.
    # Out-of-bounds results and invalid-offset errors (value, key, not implemented, overflow, os errors) are silently ignored, i.e. the test passes without asserting when the offset cannot be built or applied. Adds xfail markers for local-timezone cases past 2038 on 32-bit builds, and on Windows for quarter classes with local time zone.
    # KNOWN ISSUE: the 32-bit xfail condition also tests the WASM flag, but the whole test is skipped on WASM, so that part is unreachable.
    tz = tz_naive_fixture

    # try to create an out-of-bounds result timestamp; if we can't create
    # the offset skip
    try:
        if _offset in (BusinessHour, CustomBusinessHour):
            # Using 10000 in BusinessHour fails in tz check because of DST
            # difference
            offset = _get_offset(_offset, value=100000)
        else:
            offset = _get_offset(_offset, value=10000)

        result = Timestamp("20080101") + offset
        assert isinstance(result, datetime)
        assert result.tzinfo is None

        # Check tz is preserved
        t = Timestamp("20080101", tz=tz)
        result = t + offset
        assert isinstance(result, datetime)
        if tz is not None:
            assert t.tzinfo is not None

        if (
            isinstance(tz, tzlocal)
            and ((not IS64) or WASM)
            and _offset is not DateOffset
        ):
            # If we hit OutOfBoundsDatetime on non-64 bit machines
            # we'll drop out of the try clause before the next test
            request.applymarker(
                pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
            )
        elif (
            isinstance(tz, tzlocal)
            and is_platform_windows()
            and _offset in (QuarterEnd, BQuarterBegin, BQuarterEnd, FY5253Quarter)
        ):
            request.applymarker(
                pytest.mark.xfail(reason="After GH#49737 t.tzinfo is None on CI")
            )
        assert str(t.tzinfo) == str(result.tzinfo), (t.tzinfo, result.tzinfo)

    except OutOfBoundsDatetime:
        pass
    except (ValueError, KeyError, NotImplementedError, OverflowError, OSError):
        # we are creating an invalid offset
        # so ignore
        # - NotImplementedError is raised for tz-aware timestamps outside Python's
        #   range that are created by adding the offset
        # - OverflowError is raised in the same case on 32-bit systems with tzlocal
        # - OSError is raised in the same case on Windows with tzlocal
        pass


def test_offsets_compare_equal(_offset):
    # [FUNCTION DOC]
    # Root cause of GH#456. Two default instances of each class compare equal and are not unequal.
    # root cause of GH#456: __ne__ was not implemented
    offset1 = _offset()
    offset2 = _offset()
    assert not offset1 != offset2
    assert offset1 == offset2


@pytest.mark.parametrize(
    "date, offset2",
    [
        [Timestamp(2008, 1, 1), BDay(2)],
        [Timestamp(2014, 7, 1, 10, 00), BusinessHour(n=3)],
        [
            Timestamp(2014, 7, 1, 10),
            CustomBusinessHour(
                holidays=["2014-06-27", Timestamp(2014, 6, 30), Timestamp("2014-07-02")]
            ),
        ],
        [Timestamp(2008, 1, 2), SemiMonthEnd(2)],
        [Timestamp(2008, 1, 2), SemiMonthBegin(2)],
        [Timestamp(2008, 1, 2), Week(2)],
        [Timestamp(2008, 1, 2), WeekOfMonth(2)],
        [Timestamp(2008, 1, 2), LastWeekOfMonth(2)],
    ],
)
def test_rsub(date, offset2):
    # [FUNCTION DOC]
    # Parametrized over (timestamp, offset) pairs for business day, business hour, custom business hour with holidays, semi-month end/begin, week, week-of-month and last-week-of-month offsets. Subtracting the offset from the date must equal applying the negated offset to the date.
    assert date - offset2 == (-offset2)._apply(date)


@pytest.mark.parametrize(
    "date, offset2",
    [
        [Timestamp(2008, 1, 1), BDay(2)],
        [Timestamp(2014, 7, 1, 10, 00), BusinessHour(n=3)],
        [
            Timestamp(2014, 7, 1, 10),
            CustomBusinessHour(
                holidays=["2014-06-27", Timestamp(2014, 6, 30), Timestamp("2014-07-02")]
            ),
        ],
        [Timestamp(2008, 1, 2), SemiMonthEnd(2)],
        [Timestamp(2008, 1, 2), SemiMonthBegin(2)],
        [Timestamp(2008, 1, 2), Week(2)],
        [Timestamp(2008, 1, 2), WeekOfMonth(2)],
        [Timestamp(2008, 1, 2), LastWeekOfMonth(2)],
    ],
)
def test_radd(date, offset2):
    # [FUNCTION DOC]
    # Same parameter pairs as the reflected subtraction test. Adding date plus offset equals offset plus date.
    assert date + offset2 == offset2 + date


@pytest.mark.parametrize(
    "date, offset_box, offset2",
    [
        [Timestamp(2008, 1, 1), BDay, BDay(2)],
        [Timestamp(2008, 1, 2), SemiMonthEnd, SemiMonthEnd(2)],
        [Timestamp(2008, 1, 2), SemiMonthBegin, SemiMonthBegin(2)],
        [Timestamp(2008, 1, 2), Week, Week(2)],
        [Timestamp(2008, 1, 2), WeekOfMonth, WeekOfMonth(2)],
        [Timestamp(2008, 1, 2), LastWeekOfMonth, LastWeekOfMonth(2)],
    ],
)
def test_sub(date, offset_box, offset2):
    # [FUNCTION DOC]
    # Parametrized over (date, offset class, step-two instance) for six classes. Subtracting a date from an offset raises a type error "Cannot subtract datetime from offset". Also verifies two times the offset minus the offset equals the offset, the date minus the offset equals the date plus the class built with step -2, and equals the date minus (two times offset minus offset).
    off = offset2
    msg = "Cannot subtract datetime from offset"
    with pytest.raises(TypeError, match=msg):
        off - date

    assert 2 * off - off == off
    assert date - offset2 == date + offset_box(-2)
    assert date - offset2 == date - (2 * off - off)


@pytest.mark.parametrize(
    "offset_box, offset1",
    [
        [BDay, BDay()],
        [LastWeekOfMonth, LastWeekOfMonth()],
        [WeekOfMonth, WeekOfMonth()],
        [Week, Week()],
        [SemiMonthBegin, SemiMonthBegin()],
        [SemiMonthEnd, SemiMonthEnd()],
        [CustomBusinessHour, CustomBusinessHour(weekmask="Tue Wed Thu Fri")],
        [BusinessHour, BusinessHour()],
    ],
)
def test_Mult1(offset_box, offset1):
    # [FUNCTION DOC]
    # Parametrized over (class, default instance) for eight classes. Multiplying an instance by 10 or 5 and adding to 2008-01-02 equals adding the class built with that step.
    dt = Timestamp(2008, 1, 2)
    assert dt + 10 * offset1 == dt + offset_box(10)
    assert dt + 5 * offset1 == dt + offset_box(5)


def test_compare_str(_offset):
    # [FUNCTION DOC]
    # GH#23524. Each class's default instance is not equal to the string "infer" and is unequal to "foo"; comparing with strings that cannot be parsed as offsets must not raise. Ordering comparisons are tested elsewhere (ticks).
    # GH#23524
    # comparing to strings that cannot be cast to DateOffsets should
    #  not raise for __eq__ or __ne__
    off = _get_offset(_offset)

    assert not off == "infer"
    assert off != "foo"
    # Note: inequalities are only implemented for Tick subclasses;
    #  tests for this are in test_ticks
