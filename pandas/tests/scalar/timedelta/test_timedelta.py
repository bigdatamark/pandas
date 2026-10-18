"""test the scalar Timedelta"""
# [MODULE DOC]
# Tests for the scalar Timedelta: behavior at non-nanosecond resolutions (seconds,
# milliseconds, microseconds), unary operators, string and numeric conversions,
# component fields, hashing, pickling, implementation limits, resolution properties and
# unit deprecations. Contains a few module-level tests (class-level min/max/resolution,
# truthiness, attribute precision, large values, week suffix). Some tests use hypothesis
# and the temp file fixture.

from datetime import timedelta
import sys

from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
    NaT,
    iNaT,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import (
    OutOfBoundsTimedelta,
    Pandas4Warning,
)

from pandas import (
    Timedelta,
    to_timedelta,
)
import pandas._testing as tm


class TestNonNano:
    # [CLASS DOC]
    # Groups tests of Timedelta objects with second, millisecond or microsecond
    # resolution that are out of bounds for nanoseconds (about 106752 days). The
    # fixtures build these values; each test checks resolution preservation across
    # operations. Stateless.
    @pytest.fixture(params=["s", "ms", "us"])
    def unit_str(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over unit strings s, ms and us; returns the unit
        # string.
        return request.param

    @pytest.fixture
    def unit(self, unit_str):
        # [FUNCTION DOC]
        # HELPER: fixture converting the unit string to the numpy datetime unit code (7,
        # 8 and 9 for seconds, milliseconds, microseconds).
        # 7, 8, 9 correspond to second, millisecond, and microsecond, respectively
        attr = f"NPY_FR_{unit_str}"
        return getattr(NpyDatetimeUnit, attr).value

    @pytest.fixture
    def val(self, unit):
        # [FUNCTION DOC]
        # HELPER: fixture giving the integer value in the requested unit for the
        # microsecond count 9223372800000000, which is just out of bounds for
        # nanoseconds (divided by one thousand for milliseconds, one million for
        # seconds).
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        if unit == NpyDatetimeUnit.NPY_FR_us.value:
            value = us
        elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
            value = us // 1000
        else:
            value = us // 1_000_000
        return value

    @pytest.fixture
    def td(self, unit, val):
        # [FUNCTION DOC]
        # HELPER: fixture building a Timedelta from the value and unit through the
        # private value-and-resolution constructor.
        return Timedelta._from_value_and_reso(val, unit)

    def test_from_value_and_reso(self, unit, val):
        # [FUNCTION DOC]
        # Verifies the fixtures: the built Timedelta has the stored value, the
        # resolution code and 106752 days.
        # Just checking that the fixture is giving us what we asked for
        td = Timedelta._from_value_and_reso(val, unit)
        assert td._value == val
        assert td._creso == unit
        assert td.days == 106752

    def test_unary_non_nano(self, td, unit):
        # [FUNCTION DOC]
        # Verifies absolute value, negation and unary plus keep the resolution.
        assert abs(td)._creso == unit
        assert (-td)._creso == unit
        assert (+td)._creso == unit

    def test_sub_preserves_reso(self, td, unit):
        # [FUNCTION DOC]
        # Verifies subtracting a Timedelta from itself gives a zero Timedelta with the
        # same resolution.
        res = td - td
        expected = Timedelta._from_value_and_reso(0, unit)
        assert res == expected
        assert res._creso == unit

    def test_mul_preserves_reso(self, td, unit):
        # [FUNCTION DOC]
        # Verifies multiplying by two doubles the stored value and keeps the resolution
        # (the fixture is far from the bound, so no overflow).
        # The td fixture should always be far from the implementation
        #  bound, so doubling does not risk overflow.
        res = td * 2
        assert res._value == td._value * 2
        assert res._creso == unit

    def test_cmp_cross_reso(self, td):
        # [FUNCTION DOC]
        # Verifies comparison against a nanosecond-resolution Timedelta of 106751 days
        # is correct (numpy gets this wrong due to silent overflow).
        # numpy gets this wrong because of silent overflow
        other = Timedelta(days=106751)
        assert other < td
        assert td > other
        assert not other == td
        assert td != other

    def test_to_pytimedelta(self, td):
        # [FUNCTION DOC]
        # Verifies conversion to a python timedelta returns an exact standard library
        # timedelta of 106752 days.
        res = td.to_pytimedelta()
        expected = timedelta(days=106752)
        assert type(res) is timedelta
        assert res == expected

    def test_to_timedelta64(self, td, unit):
        # [FUNCTION DOC]
        # Verifies the three conversions to numpy (the timedelta64 method, the to numpy
        # method and the numpy view property) return a numpy timedelta64 whose integer
        # view equals the stored value and whose dtype has the matching unit.
        for res in [td.to_timedelta64(), td.to_numpy(), td.asm8]:
            assert isinstance(res, np.timedelta64)
            assert res.view("i8") == td._value
            if unit == NpyDatetimeUnit.NPY_FR_s.value:
                assert res.dtype == "m8[s]"
            elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
                assert res.dtype == "m8[ms]"
            elif unit == NpyDatetimeUnit.NPY_FR_us.value:
                assert res.dtype == "m8[us]"

    def test_truediv_timedeltalike(self, td):
        # [FUNCTION DOC]
        # Verifies true division by timedeltas: a Timedelta divided by itself is one,
        # 2.5 times it divided by it is 2.5; dividing by a nanosecond-resolution
        # Timedelta with the same integer raises an out-of-bounds timedelta error about
        # casting; the reflected division from the python timedelta succeeds
        # (microsecond resolution); and when no overflow occurs the result uses the
        # finer unit (50 microseconds over 50 milliseconds is 0.001 and the reverse is
        # 1000).
        assert td / td == 1
        assert (2.5 * td) / td == 2.5

        other = Timedelta(td._value)
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow."
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td / other

        # Timedelta(other.to_pytimedelta()) has microsecond resolution,
        #  so the division doesn't require casting all the way to nanos,
        #  so succeeds
        res = other.to_pytimedelta() / td
        expected = other.to_pytimedelta() / td.to_pytimedelta()
        assert res == expected

        # if there's no overflow, we cast to the higher reso
        left = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_us.value)
        right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
        result = left / right
        assert result == 0.001

        result = right / left
        assert result == 1000

    def test_truediv_numeric(self, td):
        # [FUNCTION DOC]
        # Verifies true division by numbers: dividing by NaN gives NaT; dividing by 2 or
        # 2.0 halves the stored value and keeps the resolution.
        assert td / np.nan is NaT

        res = td / 2
        assert res._value == td._value / 2
        assert res._creso == td._creso

        res = td / 2.0
        assert res._value == td._value / 2
        assert res._creso == td._creso

    def test_truediv_na_type_not_supported(self, td):
        # [FUNCTION DOC]
        # Verifies dividing a Timedelta by the NA singleton, or NA by a Timedelta,
        # raises a type error about unsupported operand types.
        msg_td_floordiv_na = (
            r"unsupported operand type\(s\) for /: 'Timedelta' and 'NAType'"
        )
        with pytest.raises(TypeError, match=msg_td_floordiv_na):
            td / NA

        msg_na_floordiv_td = (
            r"unsupported operand type\(s\) for /: 'NAType' and 'Timedelta'"
        )
        with pytest.raises(TypeError, match=msg_na_floordiv_td):
            NA / td

    def test_floordiv_timedeltalike(self, td):
        # [FUNCTION DOC]
        # Verifies floor division by timedeltas: itself gives one, 2.5 times itself
        # gives two; with a nanosecond Timedelta of the same integer it raises an
        # out-of-bounds timedelta error; the python timedelta reflected floor division
        # gives zero; and mixed micro/millisecond operands use the finer unit (50050
        # microseconds over 50 milliseconds is one, the reverse is zero).
        assert td // td == 1
        assert (2.5 * td) // td == 2

        other = Timedelta(td._value)
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td // other

        # Timedelta(other.to_pytimedelta()) has microsecond resolution,
        #  so the floordiv doesn't require casting all the way to nanos,
        #  so succeeds
        res = other.to_pytimedelta() // td
        assert res == 0

        # if there's no overflow, we cast to the higher reso
        left = Timedelta._from_value_and_reso(50050, NpyDatetimeUnit.NPY_FR_us.value)
        right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
        result = left // right
        assert result == 1
        result = right // left
        assert result == 0

    def test_floordiv_numeric(self, td):
        # [FUNCTION DOC]
        # Verifies floor division by numbers: NaN (scalar or zero-dimensional array)
        # gives NaT; by 2, 2.0 and numpy zero-dimensional arrays of 2 and 2.0 halves the
        # stored value (floored) and keeps the resolution.
        assert td // np.nan is NaT

        res = td // 2
        assert res._value == td._value // 2
        assert res._creso == td._creso

        res = td // 2.0
        assert res._value == td._value // 2
        assert res._creso == td._creso

        assert td // np.array(np.nan) is NaT

        res = td // np.array(2)
        assert res._value == td._value // 2
        assert res._creso == td._creso

        res = td // np.array(2.0)
        assert res._value == td._value // 2
        assert res._creso == td._creso

    def test_floordiv_na_type_not_supported(self, td):
        # [FUNCTION DOC]
        # Verifies floor-dividing a Timedelta by NA, or NA by a Timedelta, raises a type
        # error about unsupported operand types.
        msg_td_floordiv_na = (
            r"unsupported operand type\(s\) for //: 'Timedelta' and 'NAType'"
        )
        with pytest.raises(TypeError, match=msg_td_floordiv_na):
            td // NA

        msg_na_floordiv_td = (
            r"unsupported operand type\(s\) for //: 'NAType' and 'Timedelta'"
        )
        with pytest.raises(TypeError, match=msg_na_floordiv_td):
            NA // td

    def test_addsub_mismatched_reso(self, td):
        # [FUNCTION DOC]
        # Verifies addition and subtraction with a one-day microsecond Timedelta (either
        # order): the result takes the other operand's resolution and the day count
        # changes by plus or minus one; with a nanosecond Timedelta of 500 nanoseconds
        # all four operations raise an out-of-bounds timedelta error about casting to
        # nanoseconds.
        # need to cast to since td is out of bounds for ns, so
        #  so we would raise OverflowError without casting
        other = Timedelta(days=1).as_unit("us")

        # td is out of bounds for ns
        result = td + other
        assert result._creso == other._creso
        assert result.days == td.days + 1

        result = other + td
        assert result._creso == other._creso
        assert result.days == td.days + 1

        result = td - other
        assert result._creso == other._creso
        assert result.days == td.days - 1

        result = other - td
        assert result._creso == other._creso
        assert result.days == 1 - td.days

        other2 = Timedelta(500)
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td + other2
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            other2 + td
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td - other2
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            other2 - td

    def test_min(self, td):
        # [FUNCTION DOC]
        # Verifies the min attribute on an instance has the same resolution, is not
        # above the instance, and has the smallest representable integer (one above the
        # NaT sentinel).
        assert td.min <= td
        assert td.min._creso == td._creso
        assert td.min._value == NaT._value + 1

    def test_max(self, td):
        # [FUNCTION DOC]
        # Verifies the max attribute on an instance has the same resolution, is not
        # below the instance, and has the largest 64-bit integer value.
        assert td.max >= td
        assert td.max._creso == td._creso
        assert td.max._value == np.iinfo(np.int64).max

    def test_resolution(self, td):
        # [FUNCTION DOC]
        # Verifies the resolution property on an instance is a one-unit Timedelta in the
        # instance's own resolution.
        expected = Timedelta._from_value_and_reso(1, td._creso)
        result = td.resolution
        assert result == expected
        assert result._creso == expected._creso

    def test_hash(self) -> None:
        # [FUNCTION DOC]
        # Regression test (GH 54037). The maximum Timedelta at second resolution has a
        # truthy hash (hashing does not fail).
        # GH#54037
        second_resolution_max = Timedelta(0).as_unit("s").max

        assert hash(second_resolution_max)


def test_timedelta_class_min_max_resolution():
    # [FUNCTION DOC]
    # Verifies the min, max and resolution accessed on the class are in nanoseconds: min
    # is one above the NaT sentinel, max the largest 64-bit integer and resolution one
    # nanosecond.
    # when accessed on the class (as opposed to an instance), we default
    #  to nanoseconds
    assert Timedelta.min == Timedelta(NaT._value + 1)
    assert Timedelta.min._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timedelta.max == Timedelta(np.iinfo(np.int64).max)
    assert Timedelta.max._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timedelta.resolution == Timedelta(1)
    assert Timedelta.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value


class TestTimedeltaUnaryOps:
    # [CLASS DOC]
    # Groups tests of Timedelta unary operators. Stateless.
    def test_invert(self):
        # [FUNCTION DOC]
        # Verifies bitwise inversion on a Timedelta raises a type error about the unary
        # operand, matching a python timedelta; a numpy timedelta64 raises a type error
        # about the ufunc not supporting the input types.
        td = Timedelta(10, unit="D")

        msg = "bad operand type for unary ~"
        with pytest.raises(TypeError, match=msg):
            ~td

        # check this matches pytimedelta and timedelta64
        with pytest.raises(TypeError, match=msg):
            ~(td.to_pytimedelta())

        umsg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=umsg):
            ~(td.to_timedelta64())

    def test_unary_ops(self):
        # [FUNCTION DOC]
        # Verifies negation, unary plus and absolute value on a ten-day Timedelta,
        # compared with constructed Timedeltas from numbers and strings.
        td = Timedelta(10, unit="D")

        # __neg__, __pos__
        assert -td == Timedelta(-10, unit="D")
        assert -td == Timedelta("-10D")
        assert +td == Timedelta(10, unit="D")

        # __abs__, __abs__(__neg__)
        assert abs(td) == td
        assert abs(-td) == td
        assert abs(-td) == Timedelta("10D")


class TestTimedeltas:
    # [CLASS DOC]
    # Groups assorted general Timedelta tests (construction rounding, fields,
    # conversions, hashing, limits, deprecations). Stateless.
    @pytest.mark.parametrize(
        "unit, value, expected",
        [
            ("us", 9.999, 9999),
            ("ms", 9.999999, 9999999),
            ("s", 9.999999999, 9999999999),
        ],
    )
    def test_rounding_on_int_unit_construction(self, unit, value, expected):
        # [FUNCTION DOC]
        # Regression test (GH 12690). Parametrized over unit, float value and expected
        # integer. A float with more digits than the unit holds is truncated when
        # constructing from a number with the unit and when parsing a string with the
        # same suffix; both give the expected nanosecond count.
        # GH 12690
        result = Timedelta(value, unit=unit)
        assert result._value == expected
        result = Timedelta(str(value) + unit)
        assert result._value == expected

    def test_total_seconds_scalar(self):
        # [FUNCTION DOC]
        # Regression test (gh-10939). Total seconds of a Timedelta with nanosecond
        # digits equals the computed float (almost equal); NaN gives NaN.
        # see gh-10939
        rng = Timedelta("1 days, 10:11:12.100123456")
        expt = 1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9
        tm.assert_almost_equal(rng.total_seconds(), expt)

        rng = Timedelta(np.nan)
        assert np.isnan(rng.total_seconds())

    def test_conversion(self):
        # [FUNCTION DOC]
        # Verifies conversions between Timedelta and python timedelta and numpy
        # timedelta64: round trips and equality hold for values without sub-microsecond
        # parts, the python value is a plain timedelta (not a Timedelta), and a value
        # with nanoseconds is not equal to its python timedelta.
        for td in [Timedelta(10, unit="D"), Timedelta("1 days, 10:11:12.012345")]:
            td = td.as_unit("ns")
            pydt = td.to_pytimedelta()
            assert td == Timedelta(pydt)
            assert td == pydt
            assert isinstance(pydt, timedelta) and not isinstance(pydt, Timedelta)

            assert td == np.timedelta64(td._value, "ns")
            td64 = td.to_timedelta64()

            assert td64 == np.timedelta64(td._value, "ns")
            assert td == td64

            assert isinstance(td64, np.timedelta64)

        # this is NOT equal and cannot be roundtripped (because of the nanos)
        td = Timedelta("1 days, 10:11:12.012345678")
        assert td != td.to_pytimedelta()

    def test_fields(self):
        # [FUNCTION DOC]
        # Verifies component fields. Days, seconds, microseconds and nanoseconds follow
        # the standard library convention and are ints (GH 10050); hours, minutes and
        # milliseconds are not attributes and raise an attribute error. Negative values
        # normalize as days minus one plus a positive remainder, and the string,
        # absolute value and negation behave accordingly; the components attribute gives
        # day through nanosecond parts, with correct values for minus one microsecond
        # and minus one day plus one microsecond.
        def check(value):
            # [FUNCTION DOC]
            # HELPER: asserts that the given value is a plain integer.
            # that we are int
            assert isinstance(value, int)

        # compat to datetime.timedelta
        rng = to_timedelta("1 days, 10:11:12")
        assert rng.days == 1
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        assert rng.microseconds == 0
        assert rng.nanoseconds == 0

        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # GH 10050
        check(rng.days)
        check(rng.seconds)
        check(rng.microseconds)
        check(rng.nanoseconds)

        td = Timedelta("-1 days, 10:11:12")
        assert abs(td) == Timedelta("13:48:48")
        assert str(td) == "-1 days +10:11:12"
        assert -td == Timedelta("0 days 13:48:48")
        assert -Timedelta("-1 days, 10:11:12")._value == 49728000000
        assert Timedelta("-1 days, 10:11:12")._value == -49728000000

        rng = to_timedelta("-1 days, 10:11:12.100123456")
        assert rng.days == -1
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        assert rng.microseconds == 100 * 1000 + 123
        assert rng.nanoseconds == 456
        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # components
        tup = to_timedelta(-1, "us").components
        assert tup.days == -1
        assert tup.hours == 23
        assert tup.minutes == 59
        assert tup.seconds == 59
        assert tup.milliseconds == 999
        assert tup.microseconds == 999
        assert tup.nanoseconds == 0

        # GH 10050
        check(tup.days)
        check(tup.hours)
        check(tup.minutes)
        check(tup.seconds)
        check(tup.milliseconds)
        check(tup.microseconds)
        check(tup.nanoseconds)

        tup = Timedelta("-1 days 1 us").components
        assert tup.days == -2
        assert tup.hours == 23
        assert tup.minutes == 59
        assert tup.seconds == 59
        assert tup.milliseconds == 999
        assert tup.microseconds == 999
        assert tup.nanoseconds == 0

    # TODO: this is a test of to_timedelta string parsing
    def test_iso_conversion(self):
        # [FUNCTION DOC]
        # Regression test (GH 21877). A string in ISO 8601 duration format for one
        # second equals a one-second Timedelta.
        # KNOWN ISSUE: a TODO notes that this really tests the string-parsing function
        # rather than the scalar, so it is in an odd location.
        # GH #21877
        expected = Timedelta(1, unit="s")
        assert to_timedelta("P0DT0H0M1S") == expected

    # TODO: this is a test of to_timedelta returning NaT
    def test_nat_converters(self):
        # [FUNCTION DOC]
        # Verifies parsing the strings nat and nan returns a missing value whose numpy
        # conversion has a datetime kind and whose integer view equals the NaT sentinel.
        # KNOWN ISSUE: a TODO notes that this tests the conversion function returning
        # NaT rather than the scalar; also the numpy dtype asserted has a datetime kind,
        # not a timedelta kind.
        result = to_timedelta("nat").to_numpy()
        assert result.dtype.kind == "M"
        assert result.astype("int64") == iNaT

        result = to_timedelta("nan").to_numpy()
        assert result.dtype.kind == "M"
        assert result.astype("int64") == iNaT

    def test_numeric_conversions(self):
        # [FUNCTION DOC]
        # Verifies a Timedelta built from an integer or from a number with a unit
        # (nanoseconds through days) equals the corresponding numpy timedelta64.
        assert Timedelta(0) == np.timedelta64(0, "ns")
        assert Timedelta(10) == np.timedelta64(10, "ns")
        assert Timedelta(10, unit="ns") == np.timedelta64(10, "ns")

        assert Timedelta(10, unit="us") == np.timedelta64(10, "us")
        assert Timedelta(10, unit="ms") == np.timedelta64(10, "ms")
        assert Timedelta(10, unit="s") == np.timedelta64(10, "s")
        assert Timedelta(10, unit="D") == np.timedelta64(10, "D")

    def test_timedelta_conversions(self):
        # [FUNCTION DOC]
        # Verifies Timedeltas from python timedeltas of one second, one microsecond and
        # one day equal the corresponding numpy values cast to nanoseconds.
        assert Timedelta(timedelta(seconds=1)) == np.timedelta64(1, "s").astype(
            "m8[ns]"
        )
        assert Timedelta(timedelta(microseconds=1)) == np.timedelta64(1, "us").astype(
            "m8[ns]"
        )
        assert Timedelta(timedelta(days=1)) == np.timedelta64(1, "D").astype("m8[ns]")

    def test_to_numpy_alias(self):
        # [FUNCTION DOC]
        # GH 24653, GH 44460. The to numpy method equals the timedelta64 conversion;
        # passing a dtype or a copy request raises a value error saying those arguments
        # are ignored.
        # GH 24653: alias .to_numpy() for scalars
        td = Timedelta("10m7s")
        assert td.to_timedelta64() == td.to_numpy()

        # GH#44460
        msg = "dtype and copy arguments are ignored"
        with pytest.raises(ValueError, match=msg):
            td.to_numpy("m8[s]")
        with pytest.raises(ValueError, match=msg):
            td.to_numpy(copy=True)

    def test_identity(self):
        # [FUNCTION DOC]
        # Verifies a Timedelta is an instance of both Timedelta and the standard library
        # timedelta.
        td = Timedelta(10, unit="D")
        assert isinstance(td, Timedelta)
        assert isinstance(td, timedelta)

    def test_short_format_converters(self):
        # [FUNCTION DOC]
        # Verifies short string parsing: bare numbers and the ns, us (mixed case), ms,
        # s, D (and negative, whitespace-padded) suffixes equal the right numpy values;
        # the uppercase NS and lowercase d suffixes emit Pandas4 deprecation warnings;
        # an unknown suffix raises a value error about an invalid unit abbreviation and
        # a unit without a number raises a value error about that.
        def conv(v):
            # [FUNCTION DOC]
            # HELPER: converts a numpy timedelta64 to nanosecond resolution so it can be
            # compared with a parsed Timedelta.
            return v.astype("m8[ns]")

        assert Timedelta("10") == np.timedelta64(10, "ns")
        assert Timedelta("10ns") == np.timedelta64(10, "ns")
        assert Timedelta("100") == np.timedelta64(100, "ns")
        assert Timedelta("100ns") == np.timedelta64(100, "ns")

        assert Timedelta("1000") == np.timedelta64(1000, "ns")
        assert Timedelta("1000ns") == np.timedelta64(1000, "ns")

        msg = "'NS' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            assert Timedelta("1000NS") == np.timedelta64(1000, "ns")

        assert Timedelta("10us") == np.timedelta64(10000, "ns")
        assert Timedelta("100us") == np.timedelta64(100000, "ns")
        assert Timedelta("1000us") == np.timedelta64(1000000, "ns")
        assert Timedelta("1000Us") == np.timedelta64(1000000, "ns")
        assert Timedelta("1000uS") == np.timedelta64(1000000, "ns")

        assert Timedelta("1ms") == np.timedelta64(1000000, "ns")
        assert Timedelta("10ms") == np.timedelta64(10000000, "ns")
        assert Timedelta("100ms") == np.timedelta64(100000000, "ns")
        assert Timedelta("1000ms") == np.timedelta64(1000000000, "ns")

        assert Timedelta("-1s") == -np.timedelta64(1000000000, "ns")
        assert Timedelta("1s") == np.timedelta64(1000000000, "ns")
        assert Timedelta("10s") == np.timedelta64(10000000000, "ns")
        assert Timedelta("100s") == np.timedelta64(100000000000, "ns")
        assert Timedelta("1000s") == np.timedelta64(1000000000000, "ns")

        msg = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            assert Timedelta("1d") == conv(np.timedelta64(1, "D"))
        assert Timedelta("-1D") == -conv(np.timedelta64(1, "D"))
        assert Timedelta("1D") == conv(np.timedelta64(1, "D"))
        assert Timedelta("10D") == conv(np.timedelta64(10, "D"))
        assert Timedelta("100D") == conv(np.timedelta64(100, "D"))
        assert Timedelta("1000D") == conv(np.timedelta64(1000, "D"))
        assert Timedelta("10000D") == conv(np.timedelta64(10000, "D"))

        # space
        assert Timedelta(" 10000D ") == conv(np.timedelta64(10000, "D"))
        assert Timedelta(" - 10000D ") == -conv(np.timedelta64(10000, "D"))

        # invalid
        msg = "invalid unit abbreviation"
        with pytest.raises(ValueError, match=msg):
            Timedelta("1foo")
        msg = "unit abbreviation w/o a number"
        with pytest.raises(ValueError, match=msg):
            Timedelta("foo")

    def test_full_format_converters(self):
        # [FUNCTION DOC]
        # Verifies full-format string parsing: day words (with trailing comma and
        # negative sign), clock times with and without fractions, combined days and
        # times; a negative day string with only an hour part and no minutes raises a
        # value error saying there are leftover units.
        def conv(v):
            # [FUNCTION DOC]
            # HELPER: converts a numpy timedelta64 to nanosecond resolution so it can be
            # compared with a parsed Timedelta.
            return v.astype("m8[ns]")

        d1 = np.timedelta64(1, "D")

        assert Timedelta("1days") == conv(d1)
        assert Timedelta("1days,") == conv(d1)
        assert Timedelta("- 1days,") == -conv(d1)

        assert Timedelta("00:00:01") == conv(np.timedelta64(1, "s"))
        assert Timedelta("06:00:01") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        assert Timedelta("06:00:01.0") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        assert Timedelta("06:00:01.01") == conv(
            np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        assert Timedelta("- 1days, 00:00:01") == conv(-d1 + np.timedelta64(1, "s"))
        assert Timedelta("1days, 06:00:01") == conv(
            d1 + np.timedelta64(6 * 3600 + 1, "s")
        )
        assert Timedelta("1days, 06:00:01.01") == conv(
            d1 + np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        # invalid
        msg = "have leftover units"
        with pytest.raises(ValueError, match=msg):
            Timedelta("- 1days, 00")

    def test_pickle(self, temp_file):
        # [FUNCTION DOC]
        # Verifies pickling a Timedelta (using the temp file fixture and round trip
        # helper) gives an equal value.
        v = Timedelta("1 days 10:11:12.0123456")
        v_p = tm.round_trip_pickle(v, temp_file)
        assert v == v_p

    def test_timedelta_hash_equality(self):
        # [FUNCTION DOC]
        # Regression test (GH 11129). A one-day Timedelta hashes like the equal python
        # timedelta and works as a dictionary key; twenty Timedeltas with day and second
        # parts hash like their python versions; a one-nanosecond Timedelta hashes
        # differently from its (truncated) python version.
        # GH 11129
        v = Timedelta(1, "D")
        td = timedelta(days=1)
        assert hash(v) == hash(td)

        d = {td: 2}
        assert d[v] == 2

        tds = [Timedelta(seconds=1) + Timedelta(days=n) for n in range(20)]
        assert all(hash(td) == hash(td.to_pytimedelta()) for td in tds)

        # python timedeltas drop ns resolution
        ns_td = Timedelta(1, "ns")
        assert hash(ns_td) != hash(ns_td.to_pytimedelta())

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="pd.Timedelta violates the Python hash invariant (GH#44504).",
    )
    @given(
        st.integers(
            min_value=(-sys.maxsize - 1) // 500,
            max_value=sys.maxsize // 500,
        )
    )
    def test_hash_equality_invariance(self, half_microseconds: int) -> None:
        # [FUNCTION DOC]
        # Slow hypothesis test marked as an expected failure (GH 44504: Timedelta
        # violates the hash invariant). For integers scaled by 500 nanoseconds, a
        # Timedelta and numpy timedelta64 that compare equal should hash equal; the test
        # is expected to fail for some values.
        # GH#44504

        nanoseconds = half_microseconds * 500

        pandas_timedelta = Timedelta(nanoseconds)
        numpy_timedelta = np.timedelta64(nanoseconds)

        # See: https://docs.python.org/3/glossary.html#term-hashable
        # Hashable objects which compare equal must have the same hash value.
        assert pandas_timedelta != numpy_timedelta or hash(pandas_timedelta) == hash(
            numpy_timedelta
        )

    def test_implementation_limits(self):
        # [FUNCTION DOC]
        # Verifies bounds (GH 12727). The minimum and maximum Timedelta have the extreme
        # integer values; one nanosecond below the minimum yields NaT; two below or one
        # above the maximum raise an overflow error about the integer being too large;
        # the same with internal values: constructing from one below gives NaT, and two
        # below or one above raise an out-of-bounds timedelta error with a message
        # naming the value.
        min_td = Timedelta(Timedelta.min)
        max_td = Timedelta(Timedelta.max)

        # GH 12727
        # timedelta limits correspond to int64 boundaries
        assert min_td._value == iNaT + 1
        assert max_td._value == lib.i8max

        # Beyond lower limit, a NAT before the Overflow
        assert (min_td - Timedelta(1, "ns")) is NaT

        msg = "int too (large|big) to convert"
        with pytest.raises(OverflowError, match=msg):
            min_td - Timedelta(2, "ns")

        with pytest.raises(OverflowError, match=msg):
            max_td + Timedelta(1, "ns")

        # Same tests using the internal nanosecond values
        td = Timedelta(min_td._value - 1, "ns")
        assert td is NaT

        msg = "Cannot cast -9223372036854775809 from ns to 'ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta(min_td._value - 2, "ns")

        msg = "Cannot cast 9223372036854775808 from ns to 'ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta(max_td._value + 1, "ns")

    def test_total_seconds_precision(self):
        # [FUNCTION DOC]
        # Regression test (GH 19458). Total seconds for various strings equals the exact
        # float; the two difference assertions use a one-sided bound.
        # KNOWN ISSUE: the last two assertions only check that the signed difference is
        # below a tiny positive number, so they pass for any total that is larger than
        # expected; they do not bound the absolute error.
        # GH 19458
        assert Timedelta("30s").total_seconds() == 30.0
        assert Timedelta("0").total_seconds() == 0.0
        assert Timedelta("-2s").total_seconds() == -2.0
        assert Timedelta("5.324s").total_seconds() == 5.324
        assert (Timedelta("30s").total_seconds() - 30.0) < 1e-20
        assert (30.0 - Timedelta("30s").total_seconds()) < 1e-20

    def test_resolution_string(self):
        # [FUNCTION DOC]
        # Verifies the resolution string property: whole days gives D, and adding a
        # smaller unit component gives the finest nonzero component (hours h, minutes
        # min, seconds s, milliseconds ms, microseconds us, nanoseconds ns).
        assert Timedelta(days=1).resolution_string == "D"
        assert Timedelta(days=1, hours=6).resolution_string == "h"
        assert Timedelta(days=1, minutes=6).resolution_string == "min"
        assert Timedelta(days=1, seconds=6).resolution_string == "s"
        assert Timedelta(days=1, milliseconds=6).resolution_string == "ms"
        assert Timedelta(days=1, microseconds=6).resolution_string == "us"
        assert Timedelta(days=1, nanoseconds=6).resolution_string == "ns"

    def test_resolution_deprecated(self):
        # [FUNCTION DOC]
        # Regression test (GH 21344). The instance resolution property of a Timedelta
        # with days and hours is one microsecond while the class attribute is one
        # nanosecond.
        # KNOWN ISSUE: the name says deprecated but nothing is checked for deprecation
        # or warnings; the test only compares instance and class resolutions (the
        # instance value, microsecond, differs from the class value, nanosecond, because
        # the instance is microsecond resolution).
        # GH#21344
        td = Timedelta(days=4, hours=3)
        result = td.resolution
        assert result == Timedelta(microseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib timedelta behavior
        result = Timedelta.resolution
        assert result == Timedelta(nanoseconds=1)

    @pytest.mark.parametrize(
        "unit,unit_depr",
        [
            ("W", "w"),
            ("D", "d"),
            ("min", "MIN"),
            ("s", "S"),
            ("h", "H"),
            ("ms", "MS"),
            ("us", "US"),
        ],
    )
    def test_unit_deprecated(self, unit, unit_depr):
        # [FUNCTION DOC]
        # Regression test (GH 59051). Parametrized over unit and its deprecated
        # alternate-case spelling (week, day, minute, second, hour, millisecond,
        # microsecond). Constructing with the deprecated spelling emits a Pandas4
        # warning saying it is deprecated and gives a value equal to the standard
        # spelling.
        # GH#59051
        msg = f"'{unit_depr}' is deprecated and will be removed in a future version."

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = Timedelta(1, unit_depr)
        assert result == Timedelta(1, unit)


@pytest.mark.parametrize(
    "value, expected",
    [
        (Timedelta("10s"), True),
        (Timedelta("-10s"), True),
        (Timedelta(10, unit="ns"), True),
        (Timedelta(0, unit="ns"), False),
        (Timedelta(-10, unit="ns"), True),
        (Timedelta(None), True),
        (NaT, True),
    ],
)
def test_truthiness(value, expected):
    # [FUNCTION DOC]
    # Regression test for GH 21484. Parametrized over several values: a Timedelta is
    # truthy unless it is exactly zero, and NaT and the none-constructed Timedelta are
    # truthy.
    # https://github.com/pandas-dev/pandas/issues/21484
    assert bool(value) is expected


def test_timedelta_attribute_precision():
    # [FUNCTION DOC]
    # Regression test (GH 31354). Reconstructing the integer nanoseconds from days,
    # seconds, microseconds and nanoseconds fields gives the stored value for a large
    # value.
    # GH 31354
    td = Timedelta(1552211999999999872, unit="ns")
    result = td.days * 86400
    result += td.seconds
    result *= 1000000
    result += td.microseconds
    result *= 1000
    result += td.nanoseconds
    expected = td._value
    assert result == expected


def test_to_pytimedelta_large_values():
    # [FUNCTION DOC]
    # Verifies conversion of a large nanosecond value to a python timedelta truncates to
    # microseconds and gives the expected days, seconds and microseconds.
    td = Timedelta(1152921504609987375, unit="ns")
    result = td.to_pytimedelta()
    expected = timedelta(days=13343, seconds=86304, microseconds=609987)
    assert result == expected


def test_timedelta_week_suffix():
    # [FUNCTION DOC]
    # Regression test (GH 12691). Constructing from the unit W with value 1, or the
    # string 1W, equals the seven-day Timedelta.
    # GH#12691 ensure 'W' suffix works as a string passed to Timedelta
    expected = Timedelta("7 days")
    result = Timedelta(1, unit="W")
    assert result == expected

    result = Timedelta("1W")
    assert result == expected
