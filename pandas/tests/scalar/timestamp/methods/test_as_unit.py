# [MODULE DOC]
# Tests for the Timestamp method that converts a timestamp to a different resolution
# (unit). Covers lossless conversion, overflow, rounding and conversions between
# non-nanosecond units. Uses the private value and resolution attributes to check the
# stored integer and unit code.
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime

from pandas import Timestamp


class TestTimestampAsUnit:
    # [CLASS DOC]
    # Groups the tests of the unit-conversion method on Timestamp. Stateless.
    def test_as_unit(self):
        # [FUNCTION DOC]
        # Verifies converting a nanosecond timestamp to microsecond, millisecond and
        # second units. Expected: converting to the same unit returns the identical
        # object; the stored integer is the floor division of the original by the unit
        # ratio; the resolution code matches the target unit; converting back to
        # nanoseconds restores the original integer value and resolution.
        ts = Timestamp("1970-01-01").as_unit("ns")
        assert ts.unit == "ns"

        assert ts.as_unit("ns") is ts

        res = ts.as_unit("us")
        assert res._value == ts._value // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_us.value

        rt = res.as_unit("ns")
        assert rt._value == ts._value
        assert rt._creso == ts._creso

        res = ts.as_unit("ms")
        assert res._value == ts._value // 1_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

        rt = res.as_unit("ns")
        assert rt._value == ts._value
        assert rt._creso == ts._creso

        res = ts.as_unit("s")
        assert res._value == ts._value // 1_000_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_s.value

        rt = res.as_unit("ns")
        assert rt._value == ts._value
        assert rt._creso == ts._creso

    def test_as_unit_overflows(self):
        # [FUNCTION DOC]
        # Verifies overflow handling. A microsecond-resolution timestamp just past the
        # nanosecond upper bound is created through the private constructor. Expected:
        # converting to nanoseconds raises an out-of-bounds datetime error with a
        # message naming the value and unit; converting to milliseconds succeeds with
        # the integer divided by one thousand.
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        ts = Timestamp._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value, None)

        msg = "Cannot cast 2262-04-12 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ts.as_unit("ns")

        res = ts.as_unit("ms")
        assert res._value == us // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

    def test_as_unit_rounding(self):
        # [FUNCTION DOC]
        # Verifies rounding behavior when converting to a coarser unit. A timestamp of
        # 1.5 milliseconds expressed in nanoseconds is converted to milliseconds.
        # Expected: the result equals the timestamp for one millisecond (value truncated
        # to 1, resolution millisecond); passing the flag that disables rounding raises
        # a value error saying the units cannot be losslessly converted.
        ts = Timestamp(1_500_000)  # i.e. 1500 microseconds
        res = ts.as_unit("ms")

        expected = Timestamp(1_000_000)  # i.e. 1 millisecond
        assert res == expected

        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res._value == 1

        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            ts.as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # [FUNCTION DOC]
        # Verifies conversion where neither source nor target is nanoseconds
        # (milliseconds to seconds). Expected: date fields stay 1970-01-02 with all
        # sub-day fields zero, and after converting to seconds the stored value equals
        # the number of seconds in one day with the same calendar fields.
        # case where we are going neither to nor from nano
        ts = Timestamp("1970-01-02").as_unit("ms")
        assert ts.year == 1970
        assert ts.month == 1
        assert ts.day == 2
        assert ts.hour == ts.minute == ts.second == ts.microsecond == ts.nanosecond == 0

        res = ts.as_unit("s")
        assert res._value == 24 * 3600
        assert res.year == 1970
        assert res.month == 1
        assert res.day == 2
        assert (
            res.hour
            == res.minute
            == res.second
            == res.microsecond
            == res.nanosecond
            == 0
        )
