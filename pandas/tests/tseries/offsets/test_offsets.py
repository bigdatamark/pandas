"""
Tests of pandas.tseries.offsets
"""

# [MODULE DOC]
# General tests of the offset classes in the time series offsets package. Contains a generic suite run
# against every public offset class (immutability, return types, multiplication, frequency strings,
# apply, rollforward, rollback, on-offset checks, addition to scalars and empty indexes, pickling,
# hashing, non-nanosecond arrays), tests of the generic date offset (arithmetic, equality, constructor
# arguments), name/alias lookup tests, argument validation tests, and a handful of regression tests.
# Defines the module constant listing date-offset keyword names used for arithmetic, a helper that
# builds a default instance of any offset class, and fixtures supplying offset classes and expected
# values. Imports private lookup helpers and the alias cache; some tests clear that cache.

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
    conversion,
    timezones,
)
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
    _get_offset,
    _offset_map,
    to_offset,
)
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay

from pandas.tseries import offsets
from pandas.tseries.offsets import (
    FY5253,
    BDay,
    BMonthEnd,
    BusinessHour,
    CustomBusinessDay,
    CustomBusinessHour,
    CustomBusinessMonthBegin,
    CustomBusinessMonthEnd,
    DateOffset,
    Day,
    Easter,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    Nano,
    Tick,
    Week,
    WeekOfMonth,
)

_ARITHMETIC_DATE_OFFSET = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
]


def _create_offset(klass, value=1, normalize=False):
    # [FUNCTION DOC]
    # HELPER: builds an instance of a given offset class for the generic tests in this module.
    # Parameters: klass is the offset class; value is the count n (default 1); normalize is passed through
    # (default false). Special-cases classes needing extra arguments: the fiscal year offset (starting month 1,
    # weekday 1, variation "last"), the fiscal quarter offset (additionally an extra-week quarter of 1),
    # the last-week-of-month offset (weekday 5), the week-of-month offset (week 1, weekday 5), the weekly offset
    # (weekday 5), and the generic date offset (which receives value as a number of days). All other classes get
    # value positionally. Returns the constructed offset.
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
        getattr(offsets, o)
        for o in offsets.__all__
        if issubclass(getattr(offsets, o), liboffsets.MonthOffset)
        and o != "MonthOffset"
    ]
)
def month_classes(request):
    # [FUNCTION DOC]
    # HELPER: fixture.
    # Parametrized over every month-based offset class exported by the offsets package that is a subclass of
    # the month offset base, excluding the base itself. Yields one class per test.
    """
    Fixture for month based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(
    params=[
        getattr(offsets, o) for o in offsets.__all__ if o not in ("Tick", "BaseOffset")
    ]
)
def offset_types(request):
    # [FUNCTION DOC]
    # HELPER: fixture.
    # Parametrized over every exported offset class except the tick base and the offset base class. Yields one class per test.
    """
    Fixture for all the datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture
def dt():
    # [FUNCTION DOC]
    # HELPER: fixture returning a timestamp for January 2, 2008 at midnight, used as the base value in date offset arithmetic tests.
    return Timestamp(datetime(2008, 1, 2))


@pytest.fixture
def expecteds():
    # [FUNCTION DOC]
    # HELPER: fixture returning a mapping from offset class name to the expected timestamp obtained by applying a
    # default instance (as built by the creation helper) to January 1, 2011 at 09:00 (a Saturday). Used by apply,
    # rollforward, add and on-offset tests; covers every offset class including business hour, fiscal and tick classes.
    # executed value created by _create_offset
    # are applied to 2011/01/01 09:00 (Saturday)
    # used for .apply and .rollforward
    return {
        "Day": Timestamp("2011-01-02 09:00:00"),
        "DateOffset": Timestamp("2011-01-02 09:00:00"),
        "BusinessDay": Timestamp("2011-01-03 09:00:00"),
        "CustomBusinessDay": Timestamp("2011-01-03 09:00:00"),
        "CustomBusinessMonthEnd": Timestamp("2011-01-31 09:00:00"),
        "CustomBusinessMonthBegin": Timestamp("2011-01-03 09:00:00"),
        "MonthBegin": Timestamp("2011-02-01 09:00:00"),
        "BusinessMonthBegin": Timestamp("2011-01-03 09:00:00"),
        "MonthEnd": Timestamp("2011-01-31 09:00:00"),
        "SemiMonthEnd": Timestamp("2011-01-15 09:00:00"),
        "SemiMonthBegin": Timestamp("2011-01-15 09:00:00"),
        "BusinessMonthEnd": Timestamp("2011-01-31 09:00:00"),
        "YearBegin": Timestamp("2012-01-01 09:00:00"),
        "BYearBegin": Timestamp("2011-01-03 09:00:00"),
        "YearEnd": Timestamp("2011-12-31 09:00:00"),
        "BYearEnd": Timestamp("2011-12-30 09:00:00"),
        "QuarterBegin": Timestamp("2011-03-01 09:00:00"),
        "BQuarterBegin": Timestamp("2011-03-01 09:00:00"),
        "QuarterEnd": Timestamp("2011-03-31 09:00:00"),
        "BQuarterEnd": Timestamp("2011-03-31 09:00:00"),
        "HalfYearBegin": Timestamp("2011-07-01 09:00:00"),
        "HalfYearEnd": Timestamp("2011-06-30 09:00:00"),
        "BHalfYearBegin": Timestamp("2011-01-03 09:00:00"),
        "BHalfYearEnd": Timestamp("2011-06-30 09:00:00"),
        "BusinessHour": Timestamp("2011-01-03 10:00:00"),
        "CustomBusinessHour": Timestamp("2011-01-03 10:00:00"),
        "WeekOfMonth": Timestamp("2011-01-08 09:00:00"),
        "LastWeekOfMonth": Timestamp("2011-01-29 09:00:00"),
        "FY5253Quarter": Timestamp("2011-01-25 09:00:00"),
        "FY5253": Timestamp("2011-01-25 09:00:00"),
        "Week": Timestamp("2011-01-08 09:00:00"),
        "Easter": Timestamp("2011-04-24 09:00:00"),
        "Hour": Timestamp("2011-01-01 10:00:00"),
        "Minute": Timestamp("2011-01-01 09:01:00"),
        "Second": Timestamp("2011-01-01 09:00:01"),
        "Milli": Timestamp("2011-01-01 09:00:00.001000"),
        "Micro": Timestamp("2011-01-01 09:00:00.000001"),
        "Nano": Timestamp("2011-01-01T09:00:00.000000001"),
    }


class TestCommon:
    # [CLASS DOC]
    # Generic tests parametrized over all offset classes via the offset types fixture. Holds one helper that
    # checks apply-like methods and the tests that use it.
    def test_immutable(self, offset_types):
        # [FUNCTION DOC]
        # Regression test (GH#21341): assigning to the normalize or n attributes of any offset instance raises an
        # attribute error with a message about non-writable or immutable objects.
        # GH#21341 check that __setattr__ raises
        offset = _create_offset(offset_types)
        msg = "objects is not writable|DateOffset objects are immutable"
        with pytest.raises(AttributeError, match=msg):
            offset.normalize = True
        with pytest.raises(AttributeError, match=msg):
            offset.n = 91

    def test_return_type(self, offset_types):
        # [FUNCTION DOC]
        # Adding an offset to a timestamp returns a timestamp. Adding or subtracting an offset with a missing-time
        # (NaT) value returns that same missing value, in both operand orders and through the private apply on a negated offset.
        offset = _create_offset(offset_types)

        # make sure that we are returning a Timestamp
        result = Timestamp("20080101") + offset
        assert isinstance(result, Timestamp)

        # make sure that we are returning NaT
        assert NaT + offset is NaT
        assert offset + NaT is NaT

        assert NaT - offset is NaT
        assert (-offset)._apply(NaT) is NaT

    def test_offset_n(self, offset_types):
        # [FUNCTION DOC]
        # A default offset has a count of one; multiplying by minus one gives a count of minus one; multiplying by three gives three.
        offset = _create_offset(offset_types)
        assert offset.n == 1

        neg_offset = offset * -1
        assert neg_offset.n == -1

        mul_offset = offset * 3
        assert mul_offset.n == 3

    def test_offset_timedelta64_arg(self, offset_types):
        # [FUNCTION DOC]
        # Constructing an offset of the same type, with the stored keyword arguments, and a numpy timedelta value for n
        # raises a type error saying the argument must be an integer.
        # check that offset._validate_n raises TypeError on a timedelt64
        #  object
        off = _create_offset(offset_types)

        td64 = np.timedelta64(4567, "s")
        with pytest.raises(TypeError, match="argument must be an integer"):
            type(off)(n=td64, **off.kwds)

    def test_offset_mul_ndarray(self, offset_types):
        # [FUNCTION DOC]
        # Multiplying an integer array by an offset (either operand order) returns an object array in which each
        # element is the offset scaled by the corresponding integer; compared with the expected object array.
        off = _create_offset(offset_types)

        expected = np.array([[off, off * 2], [off * 3, off * 4]])

        result = np.array([[1, 2], [3, 4]]) * off
        tm.assert_numpy_array_equal(result, expected)

        result = off * np.array([[1, 2], [3, 4]])
        tm.assert_numpy_array_equal(result, expected)

    def test_offset_freqstr(self, offset_types):
        # [FUNCTION DOC]
        # For offsets whose frequency string is not one of the special unparseable forms (easter, generic date offset
        # with days, last-week-of-month Saturday), parsing the frequency string yields an offset with the same rule code as the original.
        offset = _create_offset(offset_types)

        freqstr = offset.freqstr
        if freqstr not in ("<Easter: method=3>", "<DateOffset: days=1>", "LWOM-SAT"):
            code = _get_offset(freqstr)
            assert offset.rule_code == code

    def _check_offsetfunc_works(self, offset, funcname, dt, expected, normalize=False):
        # [FUNCTION DOC]
        # HELPER: shared checker for the apply, rollforward and rollback tests.
        # Parameters: offset is an offset class; funcname is the method name to look up on an instance; dt is the input
        # (a datetime or numpy datetime64); expected is the expected timestamp; normalize (default false) selects a
        # normalizing instance. Returns without checking when normalize is true and the class is a tick or day offset
        # (normalization is disallowed, GH#21427). Otherwise builds an instance via the creation helper and verifies:
        # the method on the raw input and on a timestamp of it returns a timestamp equal to expected; on a timestamp
        # with 5 extra nanoseconds (GH#14101, no warning allowed) the nanoseconds are preserved unless normalizing.
        # For non-numpy inputs it repeats the checks for several time zones (none, UTC, Tokyo, US/Eastern and
        # dateutil variants), localizing both the input and the expectation, again including the nanosecond case.
        if normalize and issubclass(offset, (Tick, Day)):
            # normalize=True disallowed for Tick subclasses GH#21427
            return

        offset_s = _create_offset(offset, normalize=normalize)
        func = getattr(offset_s, funcname)

        result = func(dt)
        assert isinstance(result, Timestamp)
        assert result == expected

        result = func(Timestamp(dt))
        assert isinstance(result, Timestamp)
        assert result == expected

        # see gh-14101
        ts = Timestamp(dt) + Nano(5)
        # test nanosecond is preserved
        with tm.assert_produces_warning(None):
            result = func(ts)

        assert isinstance(result, Timestamp)
        if normalize is False:
            assert result == expected + Nano(5)
        else:
            assert result == expected

        if isinstance(dt, np.datetime64):
            # test tz when input is datetime or Timestamp
            return

        for tz in [
            None,
            "UTC",
            "Asia/Tokyo",
            "US/Eastern",
            "dateutil/Asia/Tokyo",
            "dateutil/US/Pacific",
        ]:
            expected_localize = expected.tz_localize(tz)
            tz_obj = timezones.maybe_get_tz(tz)
            dt_tz = conversion.localize_pydatetime(dt, tz_obj)

            result = func(dt_tz)
            assert isinstance(result, Timestamp)
            assert result == expected_localize

            result = func(Timestamp(dt, tz=tz))
            assert isinstance(result, Timestamp)
            assert result == expected_localize

            # see gh-14101
            ts = Timestamp(dt, tz=tz) + Nano(5)
            # test nanosecond is preserved
            with tm.assert_produces_warning(None):
                result = func(ts)
            assert isinstance(result, Timestamp)
            if normalize is False:
                assert result == expected_localize + Nano(5)
            else:
                assert result == expected_localize

    def test_apply(self, offset_types, expecteds):
        # [FUNCTION DOC]
        # Uses the expecteds fixture for the class being tested with a datetime and a numpy datetime64 at 2011-01-01 09:00.
        # Verifies the private apply gives the expected timestamp, and for normalizing instances gives the expected date at midnight.
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        expected = expecteds[offset_types.__name__]
        expected_norm = Timestamp(expected.date())

        for dt in [sdt, ndt]:
            self._check_offsetfunc_works(offset_types, "_apply", dt, expected)

            self._check_offsetfunc_works(
                offset_types, "_apply", dt, expected_norm, normalize=True
            )

    def test_rollforward(self, offset_types, expecteds):
        # [FUNCTION DOC]
        # Builds the expected values from the expecteds fixture, overriding classes that do not move when the date is already
        # on offset (day, month/year/half-year begin, semi-month begin, week, ticks, generic date offset) to the unchanged
        # input, and the business hour classes to 09:00 on the next business day. Builds normalized expectations (midnight dates,
        # with explicit overrides for the unchanged classes). Checks rollforward with and without normalization for datetime and
        # datetime64 inputs via the shared checker.
        expecteds = expecteds.copy()

        # result will not be changed if the target is on the offset
        no_changes = [
            "Day",
            "MonthBegin",
            "SemiMonthBegin",
            "YearBegin",
            "HalfYearBegin",
            "Week",
            "Hour",
            "Minute",
            "Second",
            "Milli",
            "Micro",
            "Nano",
            "DateOffset",
        ]
        for n in no_changes:
            expecteds[n] = Timestamp("2011/01/01 09:00")

        expecteds["BusinessHour"] = Timestamp("2011-01-03 09:00:00")
        expecteds["CustomBusinessHour"] = Timestamp("2011-01-03 09:00:00")

        # but be changed when normalize=True
        norm_expected = expecteds.copy()
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())

        normalized = {
            "Day": Timestamp("2011-01-02 00:00:00"),
            "DateOffset": Timestamp("2011-01-02 00:00:00"),
            "MonthBegin": Timestamp("2011-02-01 00:00:00"),
            "SemiMonthBegin": Timestamp("2011-01-15 00:00:00"),
            "YearBegin": Timestamp("2012-01-01 00:00:00"),
            "HalfYearBegin": Timestamp("2011-07-01 00:00:00"),
            "Week": Timestamp("2011-01-08 00:00:00"),
            "Hour": Timestamp("2011-01-01 00:00:00"),
            "Minute": Timestamp("2011-01-01 00:00:00"),
            "Second": Timestamp("2011-01-01 00:00:00"),
            "Milli": Timestamp("2011-01-01 00:00:00"),
            "Micro": Timestamp("2011-01-01 00:00:00"),
        }
        norm_expected.update(normalized)

        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        for dt in [sdt, ndt]:
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, "rollforward", dt, expected)
            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(
                offset_types, "rollforward", dt, expected, normalize=True
            )

    def test_rollback(self, offset_types):
        # [FUNCTION DOC]
        # Defines a table of expected rollback results from 2011-01-01 09:00 per class name, overridden to the unchanged input
        # for classes already on offset at that date. Derives normalized expectations from the dates and checks rollback with and
        # without normalization for datetime and datetime64 inputs via the shared checker.
        expecteds = {
            "BusinessDay": Timestamp("2010-12-31 09:00:00"),
            "CustomBusinessDay": Timestamp("2010-12-31 09:00:00"),
            "CustomBusinessMonthEnd": Timestamp("2010-12-31 09:00:00"),
            "CustomBusinessMonthBegin": Timestamp("2010-12-01 09:00:00"),
            "BusinessMonthBegin": Timestamp("2010-12-01 09:00:00"),
            "MonthEnd": Timestamp("2010-12-31 09:00:00"),
            "SemiMonthEnd": Timestamp("2010-12-31 09:00:00"),
            "BusinessMonthEnd": Timestamp("2010-12-31 09:00:00"),
            "BYearBegin": Timestamp("2010-01-01 09:00:00"),
            "YearEnd": Timestamp("2010-12-31 09:00:00"),
            "BYearEnd": Timestamp("2010-12-31 09:00:00"),
            "QuarterBegin": Timestamp("2010-12-01 09:00:00"),
            "BQuarterBegin": Timestamp("2010-12-01 09:00:00"),
            "QuarterEnd": Timestamp("2010-12-31 09:00:00"),
            "BQuarterEnd": Timestamp("2010-12-31 09:00:00"),
            "HalfYearBegin": Timestamp("2010-07-01 09:00:00"),
            "HalfYearEnd": Timestamp("2010-12-31 09:00:00"),
            "BHalfYearBegin": Timestamp("2010-07-01 09:00:00"),
            "BHalfYearEnd": Timestamp("2010-12-31 09:00:00"),
            "BusinessHour": Timestamp("2010-12-31 17:00:00"),
            "CustomBusinessHour": Timestamp("2010-12-31 17:00:00"),
            "WeekOfMonth": Timestamp("2010-12-11 09:00:00"),
            "LastWeekOfMonth": Timestamp("2010-12-25 09:00:00"),
            "FY5253Quarter": Timestamp("2010-10-26 09:00:00"),
            "FY5253": Timestamp("2010-01-26 09:00:00"),
            "Easter": Timestamp("2010-04-04 09:00:00"),
        }

        # result will not be changed if the target is on the offset
        for n in [
            "Day",
            "MonthBegin",
            "SemiMonthBegin",
            "YearBegin",
            "HalfYearBegin",
            "Week",
            "Hour",
            "Minute",
            "Second",
            "Milli",
            "Micro",
            "Nano",
            "DateOffset",
        ]:
            expecteds[n] = Timestamp("2011/01/01 09:00")

        # but be changed when normalize=True
        norm_expected = expecteds.copy()
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())

        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        for dt in [sdt, ndt]:
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, "rollback", dt, expected)

            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(
                offset_types, "rollback", dt, expected, normalize=True
            )

    def test_is_on_offset(self, offset_types, expecteds):
        # [FUNCTION DOC]
        # Takes the expected apply result for the class and asserts a default offset considers it on offset. For non-tick,
        # non-day classes, a normalizing offset must not consider that timestamp (with its time of day) on offset; except for
        # business hour classes it must consider the plain date at midnight on offset.
        dt = expecteds[offset_types.__name__]
        offset_s = _create_offset(offset_types)
        assert offset_s.is_on_offset(dt)

        # when normalize=True, is_on_offset checks time is 00:00:00
        if issubclass(offset_types, (Tick, Day)):
            # normalize=True disallowed for Tick subclasses GH#21427
            return
        offset_n = _create_offset(offset_types, normalize=True)
        assert not offset_n.is_on_offset(dt)

        if offset_types in (BusinessHour, CustomBusinessHour):
            # In default BusinessHour (9:00-17:00), normalized time
            # cannot be in business hour range
            return
        date = datetime(dt.year, dt.month, dt.day)
        assert offset_n.is_on_offset(date)

    def test_add(self, offset_types, tz_naive_fixture, expecteds):
        # [FUNCTION DOC]
        # Parametrized over classes and the naive/aware timezone fixture. Adding an offset to a datetime or timestamp yields a
        # timestamp equal to the expected value, and adding it to a timezone-aware timestamp gives the localized expectation.
        # Repeats with normalizing offsets (expected truncated to the date) except for tick and day classes, for which it returns early.
        tz = tz_naive_fixture
        dt = datetime(2011, 1, 1, 9, 0)

        offset_s = _create_offset(offset_types)
        expected = expecteds[offset_types.__name__]

        result_dt = dt + offset_s
        result_ts = Timestamp(dt) + offset_s
        for result in [result_dt, result_ts]:
            assert isinstance(result, Timestamp)
            assert result == expected

        expected_localize = expected.tz_localize(tz)
        result = Timestamp(dt, tz=tz) + offset_s
        assert isinstance(result, Timestamp)
        assert result == expected_localize

        # normalize=True, disallowed for Tick subclasses GH#21427
        if issubclass(offset_types, (Tick, Day)):
            return
        offset_s = _create_offset(offset_types, normalize=True)
        expected = Timestamp(expected.date())

        result_dt = dt + offset_s
        result_ts = Timestamp(dt) + offset_s
        for result in [result_dt, result_ts]:
            assert isinstance(result, Timestamp)
            assert result == expected

        expected_localize = expected.tz_localize(tz)
        result = Timestamp(dt, tz=tz) + offset_s
        assert isinstance(result, Timestamp)
        assert result == expected_localize

    def test_add_empty_datetimeindex(
        self, performance_warning, offset_types, tz_naive_fixture
    ):
        # [FUNCTION DOC]
        # Regression test (GH#12724, GH#30336). Adds an offset, in both operand orders, to an empty nanosecond datetime index
        # and to its underlying array. Expects the result equal to the input. The performance warning fixture value is expected
        # only for offsets without an optimized vectorized path (easter, week-of-month, last-week-of-month, custom business day,
        # business hour variants, custom business month variants, fiscal offsets); otherwise no warning is expected. The warning
        # stack level is checked only when the timezone fixture is none, to keep the many variants fast.
        # GH#12724, GH#30336
        offset_s = _create_offset(offset_types)

        dti = DatetimeIndex([], tz=tz_naive_fixture).as_unit("ns")

        if not isinstance(
            offset_s,
            (
                Easter,
                WeekOfMonth,
                LastWeekOfMonth,
                CustomBusinessDay,
                BusinessHour,
                CustomBusinessHour,
                CustomBusinessMonthBegin,
                CustomBusinessMonthEnd,
                FY5253,
                FY5253Quarter,
            ),
        ):
            # We don't have an optimized apply_index
            performance_warning = False

        # stacklevel checking is slow, and we have ~800 of variants of this
        #  test, so let's only check the stacklevel in a subset of them
        check_stacklevel = tz_naive_fixture is None
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = dti + offset_s
        tm.assert_index_equal(result, dti)
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = offset_s + dti
        tm.assert_index_equal(result, dti)

        dta = dti._data
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = dta + offset_s
        tm.assert_equal(result, dta)
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = offset_s + dta
        tm.assert_equal(result, dta)

    def test_pickle_roundtrip(self, offset_types, temp_file):
        # [FUNCTION DOC]
        # Round-trips any offset through the temp-file pickle helper and asserts equality. For non generic offsets, asserts every
        # attribute listed in the offset's attribute names is equal after the round trip, skipping the calendar attribute (whose
        # equality is always false; holidays and weekmask are checked instead).
        off = _create_offset(offset_types)
        res = tm.round_trip_pickle(off, temp_file)
        assert off == res
        if type(off) is not DateOffset:
            for attr in off._attributes:
                if attr == "calendar":
                    # np.busdaycalendar __eq__ will return False;
                    #  we check holidays and weekmask attrs so are OK
                    continue
                # Make sure nothings got lost from _params (which __eq__) is based on
                assert getattr(off, attr) == getattr(res, attr)

    def test_pickle_dateoffset_odd_inputs(self, temp_file):
        # [FUNCTION DOC]
        # Regression test (GH#34511): a generic date offset of twelve months survives a pickle round trip and, added to a datetime,
        # gives the same result as the unpickled one.
        # GH#34511
        off = DateOffset(months=12)
        res = tm.round_trip_pickle(off, temp_file)
        assert off == res

        base_dt = datetime(2020, 1, 1)
        assert base_dt + off == base_dt + res

    def test_offsets_hashable(self, offset_types):
        # [FUNCTION DOC]
        # Regression test (GH#37267): hashing any offset does not fail.
        # GH: 37267
        off = _create_offset(offset_types)
        assert hash(off) is not None

    # TODO: belongs in arithmetic tests?
    @pytest.mark.filterwarnings(
        "ignore:Non-vectorized DateOffset being applied to Series or DatetimeIndex"
    )
    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_add_dt64_ndarray_non_nano(self, offset_types, unit):
        # [FUNCTION DOC]
        # Parametrized over classes and units second, millisecond and microsecond; a warning about non-vectorized application is ignored.
        # Adds the offset to a 35-day daily range of that unit and strips the frequency. The expected value is built by adding the offset to
        # each element separately; for tick offsets with a finer resolution than the range, the expected unit is the offset's own unit
        # (cast to higher resolution as with a timedelta scalar). Asserts the index results are equal.
        # check that the result with non-nano matches nano
        off = _create_offset(offset_types)

        dti = date_range("2016-01-01", periods=35, freq="D", unit=unit)

        result = (dti + off)._with_freq(None)

        exp_unit = unit
        if isinstance(off, Tick) and off._creso > dti._data._creso:
            # cast to higher reso like we would with Timedelta scalar
            exp_unit = Timedelta(off).unit
        # TODO(GH#55564): as_unit will be unnecessary
        expected = DatetimeIndex([x + off for x in dti]).as_unit(exp_unit)

        tm.assert_index_equal(result, expected)


class TestDateOffset:
    # [CLASS DOC]
    # Tests for the generic date offset, which wraps relativedelta-style keywords. Setup clears the offset alias cache before each test.
    def setup_method(self):
        # [FUNCTION DOC]
        # Pytest per-test hook: clears the global offset alias cache so each test starts with no cached offsets. No parameters, no return.
        _offset_map.clear()

    def test_repr(self):
        # [FUNCTION DOC]
        # Smoke test: obtaining the representation of several generic date offsets (default, with count, scaled, with months) does not raise.
        repr(DateOffset())
        repr(DateOffset(2))
        repr(2 * DateOffset())
        repr(2 * DateOffset(months=2))

    def test_mul(self):
        # [FUNCTION DOC]
        # Multiplying a count-one date offset by two (either order) equals a date offset constructed with count two.
        assert DateOffset(2) == 2 * DateOffset(1)
        assert DateOffset(2) == DateOffset(1) * 2

    @pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
    def test_constructor(self, kwd, request):
        # [FUNCTION DOC]
        # Parametrized over every keyword in the sorted relativedelta keyword set. Constructing an offset with that keyword set to 2 stores it
        # in the keyword mapping and exposes it as an attribute. The singular millisecond keyword is marked as an expected failure with a not
        # implemented error because constructing with it is unsupported.
        if kwd == "millisecond":
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason="Constructing DateOffset object with `millisecond` is not "
                    "yet supported.",
                )
            )
        offset = DateOffset(**{kwd: 2})
        assert offset.kwds == {kwd: 2}
        assert getattr(offset, kwd) == 2

    def test_default_constructor(self, dt):
        # [FUNCTION DOC]
        # A date offset constructed with a positional count of two, added to January 2 2008, gives January 4 2008 (the count is treated as days).
        assert (dt + DateOffset(2)) == datetime(2008, 1, 4)

    def test_copy(self):
        # [FUNCTION DOC]
        # Copies of date offsets with months or milliseconds compare equal to equivalent newly built offsets.
        assert DateOffset(months=2).copy() == DateOffset(months=2)
        assert DateOffset(milliseconds=1).copy() == DateOffset(milliseconds=1)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                "2009-01-02",
                "2008-02-02",
                "2008-01-09",
                "2008-01-03",
                "2008-01-02 01:00:00",
                "2008-01-02 00:01:00",
                "2008-01-02 00:00:01",
                "2008-01-02 00:00:00.001000000",
                "2008-01-02 00:00:00.000001000",
            ],
            strict=True,
        ),
    )
    def test_add(self, arithmatic_offset_type, expected, dt):
        # [FUNCTION DOC]
        # Parametrized pairing each unit keyword (years through microseconds) with an expected timestamp when adding one unit to January 2 2008.
        # Asserts both operand orders give the expected timestamp.
        assert DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                "2007-01-02",
                "2007-12-02",
                "2007-12-26",
                "2008-01-01",
                "2008-01-01 23:00:00",
                "2008-01-01 23:59:00",
                "2008-01-01 23:59:59",
                "2008-01-01 23:59:59.999000000",
                "2008-01-01 23:59:59.999999000",
            ],
            strict=True,
        ),
    )
    def test_sub(self, arithmatic_offset_type, expected, dt):
        # [FUNCTION DOC]
        # Parametrized as for addition. Subtracting one unit from the base gives the expected earlier timestamp; subtracting the base timestamp from
        # the offset raises a type error saying a datetime cannot be subtracted from an offset.
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)
        with pytest.raises(TypeError, match="Cannot subtract datetime from offset"):
            DateOffset(**{arithmatic_offset_type: 1}) - dt

    @pytest.mark.parametrize(
        "arithmatic_offset_type, n, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            range(1, 10),
            [
                "2009-01-02",
                "2008-03-02",
                "2008-01-23",
                "2008-01-06",
                "2008-01-02 05:00:00",
                "2008-01-02 00:06:00",
                "2008-01-02 00:00:07",
                "2008-01-02 00:00:00.008000000",
                "2008-01-02 00:00:00.000009000",
            ],
            strict=True,
        ),
    )
    def test_mul_add(self, arithmatic_offset_type, n, expected, dt):
        # [FUNCTION DOC]
        # Parametrized with unit keyword, multiplier 1 to 9 and expected timestamp. Asserts that scaling the one-unit offset by the multiplier
        # (either operand order) and adding it to the base (either order) all give the expected timestamp.
        assert DateOffset(**{arithmatic_offset_type: 1}) * n + dt == Timestamp(expected)
        assert n * DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt + n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, n, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            range(1, 10),
            [
                "2007-01-02",
                "2007-11-02",
                "2007-12-12",
                "2007-12-29",
                "2008-01-01 19:00:00",
                "2008-01-01 23:54:00",
                "2008-01-01 23:59:53",
                "2008-01-01 23:59:59.992000000",
                "2008-01-01 23:59:59.999991000",
            ],
            strict=True,
        ),
    )
    def test_mul_sub(self, arithmatic_offset_type, n, expected, dt):
        # [FUNCTION DOC]
        # Parametrized as for the multiplication test. Subtracting a scaled one-unit offset (either operand order of the scaling) from the base gives the expected timestamp.
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt - n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    def test_leap_year(self):
        # [FUNCTION DOC]
        # Adding one month to January 31 2008 gives February 29 2008 (day clipped to month end in a leap year).
        d = datetime(2008, 1, 31)
        assert (d + DateOffset(months=1)) == datetime(2008, 2, 29)

    def test_eq(self):
        # [FUNCTION DOC]
        # Date offsets with different days or different milliseconds compare unequal.
        offset1 = DateOffset(days=1)
        offset2 = DateOffset(days=365)

        assert offset1 != offset2

        assert DateOffset(milliseconds=3) != DateOffset(milliseconds=7)

    @pytest.mark.parametrize(
        "offset_kwargs, expected_arg",
        [
            ({"microseconds": 1, "milliseconds": 1}, "2022-01-01 00:00:00.001001"),
            ({"seconds": 1, "milliseconds": 1}, "2022-01-01 00:00:01.001"),
            ({"minutes": 1, "milliseconds": 1}, "2022-01-01 00:01:00.001"),
            ({"hours": 1, "milliseconds": 1}, "2022-01-01 01:00:00.001"),
            ({"days": 1, "milliseconds": 1}, "2022-01-02 00:00:00.001"),
            ({"weeks": 1, "milliseconds": 1}, "2022-01-08 00:00:00.001"),
            ({"months": 1, "milliseconds": 1}, "2022-02-01 00:00:00.001"),
            ({"years": 1, "milliseconds": 1}, "2023-01-01 00:00:00.001"),
        ],
    )
    def test_milliseconds_combination(self, offset_kwargs, expected_arg):
        # [FUNCTION DOC]
        # Regression test (GH 49897). Parametrized over keyword combinations that include milliseconds alongside another unit (microseconds
        # through years). Adding the offset to 2022-01-01 gives the expected timestamp with the millisecond fraction included.
        # GH 49897
        offset = DateOffset(**offset_kwargs)
        ts = Timestamp("2022-01-01")
        result = ts + offset
        expected = Timestamp(expected_arg)

        assert result == expected

    def test_offset_invalid_arguments(self):
        # [FUNCTION DOC]
        # Constructing a date offset with picoseconds raises a value error whose message begins with the invalid argument or bad combination text.
        msg = "^Invalid argument/s or bad combination of arguments"
        with pytest.raises(ValueError, match=msg):
            DateOffset(picoseconds=1)


class TestOffsetNames:
    # [CLASS DOC]
    # Tests for frequency string names of offsets.
    def test_get_offset_name(self):
        # [FUNCTION DOC]
        # Asserts frequency strings: business day gives B (with count 2 gives 2B), business month end gives BME, weekly offsets for weekdays Monday to
        # Friday give W-MON through W-FRI, and a last-week-of-month offset on Sunday gives LWOM-SUN.
        assert BDay().freqstr == "B"
        assert BDay(2).freqstr == "2B"
        assert BMonthEnd().freqstr == "BME"
        assert Week(weekday=0).freqstr == "W-MON"
        assert Week(weekday=1).freqstr == "W-TUE"
        assert Week(weekday=2).freqstr == "W-WED"
        assert Week(weekday=3).freqstr == "W-THU"
        assert Week(weekday=4).freqstr == "W-FRI"

        assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == "LWOM-SUN"


def test_get_offset():
    # [FUNCTION DOC]
    # Verifies the private offset lookup by name. Unknown names (gibberish, or a quarter alias with an invalid suffix) raise a value error with the
    # invalid frequency message. Known names (B, BME, W-MON to W-FRI) return offsets equal to the expected ones, with a descriptive failure message.
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset("gibberish")
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset("QS-JAN-B")

    pairs = [
        ("B", BDay()),
        ("BME", BMonthEnd()),
        ("W-MON", Week(weekday=0)),
        ("W-TUE", Week(weekday=1)),
        ("W-WED", Week(weekday=2)),
        ("W-THU", Week(weekday=3)),
        ("W-FRI", Week(weekday=4)),
    ]

    for name, expected in pairs:
        offset = _get_offset(name)
        assert offset == expected, (
            f"Expected {name!r} to yield {expected!r} (actual: {offset!r})"
        )


def test_get_offset_legacy():
    # [FUNCTION DOC]
    # Verifies the legacy-style alias for weekly Saturday is no longer accepted: the private lookup raises a value error with the invalid frequency message.
    pairs = [("w@Sat", Week(weekday=5))]
    for name, expected in pairs:
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            _get_offset(name)


class TestOffsetAliases:
    # [CLASS DOC]
    # Tests for offset alias handling and the alias cache; setup clears the cache.
    def setup_method(self):
        # [FUNCTION DOC]
        # Pytest per-test hook: clears the global offset alias cache. No parameters, no return.
        _offset_map.clear()

    def test_alias_equality(self):
        # [FUNCTION DOC]
        # Intended to check that each cached alias entry (with a non-missing value) equals a copy of itself.
        # KNOWN ISSUE: the setup hook empties the alias cache immediately before this test and nothing populates it, so the loop body never executes and the test asserts nothing.
        for k, v in _offset_map.items():
            if v is None:
                continue
            assert k == v.copy()

    def test_rule_code(self):
        # [FUNCTION DOC]
        # For simple aliases (month end/start, business month end/start, day, business day, hour, minute, second, milli, micro) the offset obtained from the
        # name has a rule code equal to the name, the name is then present in the cache, and a three-times multiple has the same rule code. Repeats for weekly
        # aliases with each weekday suffix (multiple of five) and for year, business year, quarter, business quarter, half-year and business half-year end/start
        # bases combined with each month suffix.
        lst = ["ME", "MS", "BME", "BMS", "D", "B", "h", "min", "s", "ms", "us"]
        for k in lst:
            assert k == _get_offset(k).rule_code
            # should be cached - this is kind of an internals test...
            assert k in _offset_map
            assert k == (_get_offset(k) * 3).rule_code

        suffix_lst = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        base = "W"
        for v in suffix_lst:
            alias = "-".join([base, v])
            assert alias == _get_offset(alias).rule_code
            assert alias == (_get_offset(alias) * 5).rule_code

        suffix_lst = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
        base_lst = [
            "YE",
            "YS",
            "BYE",
            "BYS",
            "QE",
            "QS",
            "BQE",
            "BQS",
            "HYE",
            "HYS",
            "BHYE",
            "BHYS",
        ]
        for base in base_lst:
            for v in suffix_lst:
                alias = "-".join([base, v])
                assert alias == _get_offset(alias).rule_code
                assert alias == (_get_offset(alias) * 5).rule_code


def test_freq_offsets():
    # [FUNCTION DOC]
    # A business day offset with a 30 minute positive time offset has frequency string B+30Min; with a negative 30 minute offset it is B-30Min.
    off = BDay(1, offset=timedelta(0, 1800))
    assert off.freqstr == "B+30Min"

    off = BDay(1, offset=timedelta(0, -1800))
    assert off.freqstr == "B-30Min"


class TestReprNames:
    # [CLASS DOC]
    # Tests for representations of named offsets.
    def test_str_for_named_is_name(self):
        # [FUNCTION DOC]
        # Clears the alias cache, builds a list of alias names (year, quarter, half-year anchors with month suffixes, weekly aliases for each weekday, and
        # week-of-month aliases with weeks one to four and each weekday), and asserts that the frequency string of the offset obtained for each name equals that name.
        # look at all the amazing combinations!
        month_prefixes = [
            "YE",
            "YS",
            "BYE",
            "BYS",
            "QE",
            "BQE",
            "BQS",
            "QS",
            "HYE",
            "HYS",
            "BHYE",
            "BHYS",
        ]
        names = [
            prefix + "-" + month
            for prefix in month_prefixes
            for month in [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        ]
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        names += ["W-" + day for day in days]
        names += ["WOM-" + week + day for week in ("1", "2", "3", "4") for day in days]
        _offset_map.clear()
        for name in names:
            offset = _get_offset(name)
            assert offset.freqstr == name


# ---------------------------------------------------------------------


def test_valid_default_arguments(offset_types):
    # [FUNCTION DOC]
    # Regression test (GH#19142): every offset class can be constructed with no arguments.
    # GH#19142 check that the calling the constructors without passing
    # any keyword arguments produce valid offsets
    cls = offset_types
    cls()


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_month_attributes(kwd, month_classes):
    # [FUNCTION DOC]
    # Regression test (GH#18226). Parametrized over each relativedelta keyword and over the month offset classes. Constructing a month offset with that keyword raises a type
    # error saying the initializer got an unexpected keyword argument of that name.
    # GH#18226
    cls = month_classes
    # check that we cannot create e.g. MonthEnd(weeks=3)
    msg = rf"__init__\(\) got an unexpected keyword argument '{kwd}'"
    with pytest.raises(TypeError, match=msg):
        cls(**{kwd: 3})


def test_month_offset_name(month_classes):
    # [FUNCTION DOC]
    # Regression test (GH#33757): the name attribute of a month offset with count 2 can be read and equals that of the count 1 offset.
    # GH#33757 off.name with n != 1 should not raise AttributeError
    obj = month_classes(1)
    obj2 = month_classes(2)
    assert obj2.name == obj.name


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_relativedelta_kwargs(kwd, request):
    # [FUNCTION DOC]
    # Parametrized over each relativedelta keyword; checks that every keyword in the module's set is a valid argument to the date offset constructor. The singular millisecond keyword is
    # marked as an expected failure (not implemented error).
    if kwd == "millisecond":
        request.applymarker(
            pytest.mark.xfail(
                raises=NotImplementedError,
                reason="Constructing DateOffset object with `millisecond` is not "
                "yet supported.",
            )
        )
    # Check that all the arguments specified in liboffsets._relativedelta_kwds
    # are in fact valid relativedelta keyword args
    DateOffset(**{kwd: 1})


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_tick_attributes(kwd, tick_classes):
    # [FUNCTION DOC]
    # Regression test (GH#18226). Parametrized over relativedelta keywords and tick classes: constructing a tick with that keyword raises a type error about an unexpected keyword argument.
    # GH#18226
    cls = tick_classes
    # check that we cannot create e.g. Hour(weeks=3)
    msg = rf"__init__\(\) got an unexpected keyword argument '{kwd}'"
    with pytest.raises(TypeError, match=msg):
        cls(**{kwd: 3})


def test_validate_n_error():
    # [FUNCTION DOC]
    # Constructing offsets with a non-integer count (a string for the generic offset, a timedelta for month begin, an integer array for business day) raises a type error saying the argument must be an integer.
    with pytest.raises(TypeError, match="argument must be an integer"):
        DateOffset(n="Doh!")

    with pytest.raises(TypeError, match="argument must be an integer"):
        MonthBegin(n=timedelta(1))

    with pytest.raises(TypeError, match="argument must be an integer"):
        BDay(n=np.array([1, 2], dtype=np.int64))


def test_require_integers(offset_types):
    # [FUNCTION DOC]
    # Constructing any offset with a float count of 1.5 raises a value error saying the argument must be an integer.
    cls = offset_types
    with pytest.raises(ValueError, match="argument must be an integer"):
        cls(n=1.5)


def test_tick_normalize_raises(tick_classes):
    # [FUNCTION DOC]
    # Regression test (GH#21427): constructing any tick class with normalize enabled raises a value error saying tick offsets with normalize are not allowed.
    # check that trying to create a Tick object with normalize=True raises
    # GH#21427
    cls = tick_classes
    msg = "Tick offset with `normalize=True` are not allowed."
    with pytest.raises(ValueError, match=msg):
        cls(n=3, normalize=True)


@pytest.mark.parametrize(
    "offset_kwargs, expected_arg",
    [
        ({"nanoseconds": 1}, "1970-01-01 00:00:00.000000001"),
        ({"nanoseconds": 5}, "1970-01-01 00:00:00.000000005"),
        ({"nanoseconds": -1}, "1969-12-31 23:59:59.999999999"),
        ({"microseconds": 1}, "1970-01-01 00:00:00.000001"),
        ({"microseconds": -1}, "1969-12-31 23:59:59.999999"),
        ({"seconds": 1}, "1970-01-01 00:00:01"),
        ({"seconds": -1}, "1969-12-31 23:59:59"),
        ({"minutes": 1}, "1970-01-01 00:01:00"),
        ({"minutes": -1}, "1969-12-31 23:59:00"),
        ({"hours": 1}, "1970-01-01 01:00:00"),
        ({"hours": -1}, "1969-12-31 23:00:00"),
        ({"days": 1}, "1970-01-02 00:00:00"),
        ({"days": -1}, "1969-12-31 00:00:00"),
        ({"weeks": 1}, "1970-01-08 00:00:00"),
        ({"weeks": -1}, "1969-12-25 00:00:00"),
        ({"months": 1}, "1970-02-01 00:00:00"),
        ({"months": -1}, "1969-12-01 00:00:00"),
        ({"years": 1}, "1971-01-01 00:00:00"),
        ({"years": -1}, "1969-01-01 00:00:00"),
    ],
)
def test_dateoffset_add_sub(offset_kwargs, expected_arg):
    # [FUNCTION DOC]
    # Parametrized over single-unit keyword offsets (nanoseconds to years, positive and negative) with expected timestamps relative to the epoch. Adding the offset to the epoch timestamp
    # gives the expected value, subtracting it returns the epoch, and adding in the reverse operand order gives the expected value.
    offset = DateOffset(**offset_kwargs)
    ts = Timestamp(0)
    result = ts + offset
    expected = Timestamp(expected_arg)
    assert result == expected
    result -= offset
    assert result == ts
    result = offset + ts
    assert result == expected


def test_dateoffset_add_sub_timestamp_with_nano():
    # [FUNCTION DOC]
    # An offset of two minutes and nine nanoseconds added to a timestamp of 4 nanoseconds gives the expected timestamp, subtracting recovers the original, and reverse order addition works.
    # An offset that also sets an absolute hour uses the relativedelta path and, added without emitting any warning about discarding nonzero nanoseconds, gives the expected result.
    offset = DateOffset(minutes=2, nanoseconds=9)
    ts = Timestamp(4)
    result = ts + offset
    expected = Timestamp("1970-01-01 00:02:00.000000013")
    assert result == expected
    result -= offset
    assert result == ts
    result = offset + ts
    assert result == expected

    offset2 = DateOffset(minutes=2, nanoseconds=9, hour=1)
    assert offset2._use_relativedelta
    with tm.assert_produces_warning(None):
        # no warning about Discarding nonzero nanoseconds
        result2 = ts + offset2
    expected2 = Timestamp("1970-01-01 01:02:00.000000013")
    assert result2 == expected2


@pytest.mark.parametrize(
    "attribute",
    [
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ],
)
def test_dateoffset_immutable(attribute):
    # [FUNCTION DOC]
    # Parametrized over hours, days, weeks, months, years: setting that attribute on a date offset raises an attribute error saying date offset objects are immutable.
    offset = DateOffset(**{attribute: 0})
    msg = "DateOffset objects are immutable"
    with pytest.raises(AttributeError, match=msg):
        setattr(offset, attribute, 5)


def test_dateoffset_misc():
    # [FUNCTION DOC]
    # Smoke tests: the frequency string of a months-and-days date offset can be accessed, and a date offset compared to the integer two is not equal.
    oset = offsets.DateOffset(months=2, days=4)
    # it works
    oset.freqstr

    assert not offsets.DateOffset(months=2) == 2


@pytest.mark.parametrize("n", [-1, 1, 3])
def test_construct_int_arg_no_kwargs_assumed_days(n):
    # [FUNCTION DOC]
    # Regression test (GH 45890, 45643). Parametrized over n of minus one, one, three. A date offset built from a lone positional integer stores a one-day internal offset and adding it to a timestamp advances by n days.
    # GH 45890, 45643
    offset = DateOffset(n)
    assert offset._offset == timedelta(1)
    result = Timestamp(2022, 1, 2) + offset
    expected = Timestamp(2022, 1, 2 + n)
    assert result == expected


@pytest.mark.parametrize(
    "offset, expected",
    [
        (
            DateOffset(minutes=7, nanoseconds=18),
            Timestamp("2022-01-01 00:07:00.000000018"),
        ),
        (DateOffset(nanoseconds=3), Timestamp("2022-01-01 00:00:00.000000003")),
    ],
)
def test_dateoffset_add_sub_timestamp_series_with_nano(offset, expected):
    # [FUNCTION DOC]
    # Regression test (GH 47856). Parametrized over two offsets with nanosecond components. Adding the offset to a one-element series of a timestamp gives the expected timestamp, subtracting returns the start, and adding in reverse order gives the expected value.
    # GH 47856
    start_time = Timestamp("2022-01-01")
    teststamp = start_time
    testseries = Series([start_time])
    testseries = testseries + offset
    assert testseries[0] == expected
    testseries -= offset
    assert testseries[0] == teststamp
    testseries = offset + testseries
    assert testseries[0] == expected


@pytest.mark.parametrize(
    "n_months, scaling_factor, start_timestamp, expected_timestamp",
    [
        (1, 2, "2020-01-30", "2020-03-30"),
        (2, 1, "2020-01-30", "2020-03-30"),
        (1, 0, "2020-01-30", "2020-01-30"),
        (2, 0, "2020-01-30", "2020-01-30"),
        (1, -1, "2020-01-30", "2019-12-30"),
        (2, -1, "2020-01-30", "2019-11-30"),
    ],
)
def test_offset_multiplication(
    n_months, scaling_factor, start_timestamp, expected_timestamp
):
    # [FUNCTION DOC]
    # Regression test (GH 47953). Parametrized over month counts, scaling factors and expected results. Multiplying a month date offset by the factor and adding to a timestamp scalar, and to a one-element series, gives the expected timestamp (equality for the scalar, series equality for the series).
    # GH 47953
    mo1 = DateOffset(months=n_months)

    startscalar = Timestamp(start_timestamp)
    startarray = Series([startscalar])

    resultscalar = startscalar + (mo1 * scaling_factor)
    resultarray = startarray + (mo1 * scaling_factor)

    expectedscalar = Timestamp(expected_timestamp)
    expectedarray = Series([expectedscalar])
    assert resultscalar == expectedscalar

    tm.assert_series_equal(resultarray, expectedarray)


def test_offset_deprecated_error():
    # [FUNCTION DOC]
    # Creating a date range with the removed uppercase hour frequency alias raises a value error suggesting the lowercase form.
    with pytest.raises(ValueError, match="Did you mean h"):
        date_range("2012-01-01", periods=3, freq="H")


def test_dateoffset_operations_on_dataframes(performance_warning):
    # [FUNCTION DOC]
    # Regression test (GH 47953). Builds frames with a timestamp column and a date offset object column. Multiplying the offset column by 26 and adding it to the timestamp column gives a timestamp 26 months later (June 30 2021); the two-row version is evaluated under the performance warning fixture expectation. Both results are checked.
    # GH 47953
    df = DataFrame({"T": [Timestamp("2019-04-30")], "D": [DateOffset(months=1)]})
    frameresult1 = df["T"] + 26 * df["D"]
    df2 = DataFrame(
        {
            "T": [Timestamp("2019-04-30"), Timestamp("2019-04-30")],
            "D": [DateOffset(months=1), DateOffset(months=1)],
        }
    )
    expecteddate = Timestamp("2021-06-30")
    with tm.assert_produces_warning(performance_warning):
        frameresult2 = df2["T"] + 26 * df2["D"]

    assert frameresult1[0] == expecteddate
    assert frameresult2[0] == expecteddate


def test_is_yqm_start_end():
    # [FUNCTION DOC]
    # Table-driven check of the is month/quarter/year start/end predicates of offsets obtained from frequency strings (month end, business month end, quarter end and start with February anchors, business quarter end and start, year start anchored in November) applied to various timestamps, comparing each result to an expected zero/one value; also checks the days-in-month attribute for February of a leap and non-leap year.
    freq_m = to_offset("ME")
    bm = to_offset("BME")
    qfeb = to_offset("QE-FEB")
    qsfeb = to_offset("QS-FEB")
    bq = to_offset("BQE")
    bqs_apr = to_offset("BQS-APR")
    as_nov = to_offset("YS-NOV")

    tests = [
        (freq_m.is_month_start(Timestamp("2013-06-01")), 1),
        (bm.is_month_start(Timestamp("2013-06-01")), 0),
        (freq_m.is_month_start(Timestamp("2013-06-03")), 0),
        (bm.is_month_start(Timestamp("2013-06-03")), 1),
        (qfeb.is_month_end(Timestamp("2013-02-28")), 1),
        (qfeb.is_quarter_end(Timestamp("2013-02-28")), 1),
        (qfeb.is_year_end(Timestamp("2013-02-28")), 1),
        (qfeb.is_month_start(Timestamp("2013-03-01")), 1),
        (qfeb.is_quarter_start(Timestamp("2013-03-01")), 1),
        (qfeb.is_year_start(Timestamp("2013-03-01")), 1),
        (qsfeb.is_month_end(Timestamp("2013-03-31")), 1),
        (qsfeb.is_quarter_end(Timestamp("2013-03-31")), 0),
        (qsfeb.is_year_end(Timestamp("2013-03-31")), 0),
        (qsfeb.is_month_start(Timestamp("2013-02-01")), 1),
        (qsfeb.is_quarter_start(Timestamp("2013-02-01")), 1),
        (qsfeb.is_year_start(Timestamp("2013-02-01")), 1),
        (bq.is_month_end(Timestamp("2013-06-30")), 0),
        (bq.is_quarter_end(Timestamp("2013-06-30")), 0),
        (bq.is_year_end(Timestamp("2013-06-30")), 0),
        (bq.is_month_end(Timestamp("2013-06-28")), 1),
        (bq.is_quarter_end(Timestamp("2013-06-28")), 1),
        (bq.is_year_end(Timestamp("2013-06-28")), 0),
        (bqs_apr.is_month_end(Timestamp("2013-06-30")), 0),
        (bqs_apr.is_quarter_end(Timestamp("2013-06-30")), 0),
        (bqs_apr.is_year_end(Timestamp("2013-06-30")), 0),
        (bqs_apr.is_month_end(Timestamp("2013-06-28")), 1),
        (bqs_apr.is_quarter_end(Timestamp("2013-06-28")), 1),
        (bqs_apr.is_year_end(Timestamp("2013-03-29")), 1),
        (as_nov.is_year_start(Timestamp("2013-11-01")), 1),
        (as_nov.is_year_end(Timestamp("2013-10-31")), 1),
        (Timestamp("2012-02-01").days_in_month, 29),
        (Timestamp("2013-02-01").days_in_month, 28),
    ]

    for ts, value in tests:
        assert ts == value


@pytest.mark.parametrize("left", [DateOffset(1), Nano(1)])
@pytest.mark.parametrize("right", [DateOffset(1), Nano(1)])
def test_multiply_dateoffset_typeerror(left, right):
    # [FUNCTION DOC]
    # Parametrized over all pairs of a date offset or a nanosecond tick on each side. Multiplying two offsets raises a type error saying they cannot be multiplied.
    with pytest.raises(TypeError, match="Cannot multiply"):
        left * right
