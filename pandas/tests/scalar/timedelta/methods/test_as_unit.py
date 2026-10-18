# [MODULE DOC]
# Tests for the Timedelta method that converts a timedelta to a different time resolution (as_unit):
# value rescaling, overflow detection, rounding control, and internal field consistency for non-nanosecond units.
# Relies on private attributes for the stored integer value and resolution code.

import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta

from pandas import Timedelta


class TestAsUnit:
    # [CLASS DOC]
    # Groups tests of unit conversion on Timedelta.
    def test_as_unit(self):
        # [FUNCTION DOC]
        # Starting from a one-day timedelta (microsecond resolution), converting to its own unit returns the same object.
        # Converting to nanoseconds multiplies the stored value by one thousand, to milliseconds and seconds divides it
        # (by one thousand and one million), and sets the matching resolution code. Converting each result back to
        # microseconds restores the original value and resolution.
        td = Timedelta(days=1)

        assert td.as_unit("us") is td

        res = td.as_unit("ns")
        assert res._value == td._value * 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ns.value

        rt = res.as_unit("us")
        assert rt._value == td._value
        assert rt._creso == td._creso

        res = td.as_unit("ms")
        assert res._value == td._value // 1_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

        rt = res.as_unit("us")
        assert rt._value == td._value
        assert rt._creso == td._creso

        res = td.as_unit("s")
        assert res._value == td._value // 1_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_s.value

        rt = res.as_unit("us")
        assert rt._value == td._value
        assert rt._creso == td._creso

    def test_as_unit_overflows(self):
        # [FUNCTION DOC]
        # A microsecond count just beyond the nanosecond bound is built through the private value-and-resolution factory.
        # Converting it to nanoseconds must raise the out-of-bounds timedelta error with a message naming the days value and
        # the target unit; converting it to milliseconds succeeds with the value divided by one thousand.
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        td = Timedelta._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value)

        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td.as_unit("ns")

        res = td.as_unit("ms")
        assert res._value == us // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

    def test_as_unit_rounding(self):
        # [FUNCTION DOC]
        # Converting 1500 microseconds to milliseconds rounds to one millisecond (equal to a one millisecond timedelta,
        # stored value one, millisecond resolution). Passing the option that disallows rounding must raise a value error
        # saying the units cannot be converted losslessly.
        td = Timedelta(microseconds=1500)
        res = td.as_unit("ms")

        expected = Timedelta(milliseconds=1)
        assert res == expected

        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res._value == 1

        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            td.as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # [FUNCTION DOC]
        # Conversion between two non-nanosecond units (milliseconds to microseconds). Checks that days, stored value,
        # components, the private day/hour fields and total seconds stay consistent before and after conversion.
        # case where we are going neither to nor from nano
        td = Timedelta(days=1).as_unit("ms")
        assert td.days == 1
        assert td._value == 86_400_000
        assert td.components.days == 1
        assert td._d == 1
        assert td.total_seconds() == 86400

        res = td.as_unit("us")
        assert res._value == 86_400_000_000
        assert res.components.days == 1
        assert res.components.hours == 0
        assert res._d == 1
        assert res._h == 0
        assert res.total_seconds() == 86400
