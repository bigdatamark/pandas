"""
Tests for scalar Timedelta arithmetic ops
"""

# [MODULE DOC]
# Tests for arithmetic and comparison operators on the Timedelta scalar: addition and subtraction with datetimes, timestamps, timedeltas, numpy values, offsets,
# arrays and the not-a-time scalar; multiplication, division, floor division, modulo and divmod (and their reflected forms) with numbers, time-like scalars, arrays and missing values;
# rich comparisons including out-of-range standard library timedeltas; and generic unsupported-operand behavior. Uses the unit and tick-class fixtures and the private operator helpers for reflected operators.

from datetime import (
    datetime,
    timedelta,
)
import operator

import numpy as np
import pytest

from pandas.errors import (
    OutOfBoundsTimedelta,
    Pandas4Warning,
)

import pandas as pd
from pandas import (
    NaT,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm
from pandas.core import ops


class TestTimedeltaAdditionSubtraction:
    # [CLASS DOC]
    # Groups tests of addition and subtraction (forward and reflected) involving a Timedelta.
    """
    Tests for Timedelta methods:

        __add__, __radd__,
        __sub__, __rsub__
    """

    def test_td_add_sub_pydatetime(self, unit):
        # [FUNCTION DOC]
        # GH#53643. Parametrized over the resolution fixture. Adding a Timedelta of 23 hours to a standard library datetime, in either order, gives the datetime 23 hours later; subtracting it gives 23 hours earlier than the day before.
        # GH#53643
        td = Timedelta(hours=23).as_unit(unit)
        dt = datetime(2016, 1, 1)

        expected = datetime(2016, 1, 1, 23)
        result = dt + td
        assert result == expected
        result = td + dt
        assert result == expected

        expected = datetime(2015, 12, 31, 1)
        result = dt - td
        assert result == expected

    @pytest.mark.parametrize(
        "ten_seconds",
        [
            Timedelta(10, unit="s"),
            timedelta(seconds=10),
            np.timedelta64(10, "s"),
            np.timedelta64(10000000000, "ns"),
            offsets.Second(10),
        ],
    )
    def test_td_add_sub_ten_seconds(self, ten_seconds):
        # [FUNCTION DOC]
        # GH#6808. Parametrized over five representations of ten seconds (Timedelta, standard timedelta, numpy timedelta in seconds and in nanoseconds, second offset). Adding to and subtracting from a timestamp shifts it by ten seconds.
        # GH#6808
        base = Timestamp("20130101 09:01:12.123456")
        expected_add = Timestamp("20130101 09:01:22.123456")
        expected_sub = Timestamp("20130101 09:01:02.123456")

        result = base + ten_seconds
        assert result == expected_add

        result = base - ten_seconds
        assert result == expected_sub

    @pytest.mark.parametrize(
        "one_day_ten_secs",
        [
            Timedelta("1 day, 00:00:10"),
            Timedelta("1 days, 00:00:10"),
            timedelta(days=1, seconds=10),
            np.timedelta64(1, "D") + np.timedelta64(10, "s"),
            offsets.Day() + offsets.Second(10),
        ],
    )
    def test_td_add_sub_one_day_ten_seconds(self, one_day_ten_secs):
        # [FUNCTION DOC]
        # GH#6808. Parametrized over five representations of one day plus ten seconds. Adding to and subtracting from a timestamp shifts it by that amount.
        # GH#6808
        base = Timestamp("20130102 09:01:12.123456")
        expected_add = Timestamp("20130103 09:01:22.123456")
        expected_sub = Timestamp("20130101 09:01:02.123456")

        result = base + one_day_ten_secs
        assert result == expected_add

        result = base - one_day_ten_secs
        assert result == expected_sub

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_datetimelike_scalar(self, op):
        # [FUNCTION DOC]
        # GH#19738. Parametrized over forward and reflected addition. A ten-day timedelta combined with a datetime, a timestamp or a numpy datetime gives a Timestamp with the right value; with NaT gives NaT.
        # For forward addition the datetime case is also checked to be a Timestamp (not a plain datetime).
        # GH#19738
        td = Timedelta(10, unit="D")

        result = op(td, datetime(2016, 1, 1))
        if op is operator.add:
            # datetime + Timedelta does _not_ call Timedelta.__radd__,
            # so we get a datetime back instead of a Timestamp
            assert isinstance(result, Timestamp)
        assert result == Timestamp(2016, 1, 11)

        result = op(td, Timestamp("2018-01-12 18:09"))
        assert isinstance(result, Timestamp)
        assert result == Timestamp("2018-01-22 18:09")

        result = op(td, np.datetime64("2018-01-12"))
        assert isinstance(result, Timestamp)
        assert result == Timestamp("2018-01-22")

        result = op(td, NaT)
        assert result is NaT

    def test_td_add_timestamp_overflow(self):
        # [FUNCTION DOC]
        # Adding a day-scale Timedelta or standard timedelta to a nanosecond-resolution timestamp in 1700 whose result is out of range must raise the out-of-bounds timedelta error stating the cast to nanoseconds without overflow.
        ts = Timestamp("1700-01-01").as_unit("ns")
        msg = "Cannot cast 259987 days 00:00:00 to unit='ns' without overflow."
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            ts + Timedelta(13 * 19999, unit="D")

        msg = "Cannot cast 259987 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            ts + timedelta(days=13 * 19999)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_td(self, op):
        # [FUNCTION DOC]
        # Parametrized over forward and reflected addition. Two Timedeltas add to a Timedelta of the summed duration.
        td = Timedelta(10, unit="D")

        result = op(td, Timedelta(days=10))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=20)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_pytimedelta(self, op):
        # [FUNCTION DOC]
        # Parametrized over forward and reflected addition. A Timedelta plus a standard timedelta gives a Timedelta of the summed duration.
        td = Timedelta(10, unit="D")
        result = op(td, timedelta(days=9))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=19)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_timedelta64(self, op):
        # [FUNCTION DOC]
        # Parametrized over forward and reflected addition. A Timedelta plus a numpy timedelta (negative days) gives a Timedelta of the summed duration.
        td = Timedelta(10, unit="D")
        result = op(td, np.timedelta64(-4, "D"))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=6)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_offset(self, op):
        # [FUNCTION DOC]
        # Parametrized over forward and reflected addition. A Timedelta plus an hour offset gives a Timedelta including the hours.
        td = Timedelta(10, unit="D")

        result = op(td, offsets.Hour(6))
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=10, hours=6)

    def test_td_sub_td(self):
        # [FUNCTION DOC]
        # Subtracting a Timedelta from itself gives a zero Timedelta.
        td = Timedelta(10, unit="D")
        expected = Timedelta(0, unit="ns")
        result = td - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_pytimedelta(self):
        # [FUNCTION DOC]
        # Subtracting a standard timedelta from an equal Timedelta, in both orders, gives a zero Timedelta.
        td = Timedelta(10, unit="D")
        expected = Timedelta(0, unit="ns")

        result = td - td.to_pytimedelta()
        assert isinstance(result, Timedelta)
        assert result == expected

        result = td.to_pytimedelta() - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_timedelta64(self):
        # [FUNCTION DOC]
        # Subtracting a numpy timedelta from an equal Timedelta, in both orders, gives a zero Timedelta.
        td = Timedelta(10, unit="D")
        expected = Timedelta(0, unit="ns")

        result = td - td.to_timedelta64()
        assert isinstance(result, Timedelta)
        assert result == expected

        result = td.to_timedelta64() - td
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_td_sub_nat(self):
        # [FUNCTION DOC]
        # Subtracting NaT from a Timedelta gives NaT (NaT is treated as time-delta-like here).
        # In this context pd.NaT is treated as timedelta-like
        td = Timedelta(10, unit="D")
        result = td - NaT
        assert result is NaT

    def test_td_sub_td64_nat(self):
        # [FUNCTION DOC]
        # Subtracting a missing numpy timedelta from a Timedelta, in either order, gives NaT.
        td = Timedelta(10, unit="D")
        td_nat = np.timedelta64("NaT", "ns")

        result = td - td_nat
        assert result is NaT

        result = td_nat - td
        assert result is NaT

    def test_td_sub_offset(self):
        # [FUNCTION DOC]
        # Subtracting an hour offset from a ten-day Timedelta gives 239 hours.
        td = Timedelta(10, unit="D")
        result = td - offsets.Hour(1)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(239, unit="h")

    def test_td_add_sub_numeric_raises(self):
        # [FUNCTION DOC]
        # Adding or subtracting a plain integer or float (Python or numpy), in either operand order, must raise a type error for unsupported operand types.
        td = Timedelta(10, unit="D")
        msg = "unsupported operand type"
        for other in [2, 2.0, np.int64(2), np.float64(2)]:
            with pytest.raises(TypeError, match=msg):
                td + other
            with pytest.raises(TypeError, match=msg):
                other + td
            with pytest.raises(TypeError, match=msg):
                td - other
            with pytest.raises(TypeError, match=msg):
                other - td

    def test_td_add_sub_int_ndarray(self):
        # [FUNCTION DOC]
        # Adding or subtracting a one-element integer array and a Timedelta, in both orders, must raise a type error with message about unsupported operand types (the array-plus-Timedelta message may alternatively be about concatenation not being implemented for arrays).
        td = Timedelta("1 day")
        other = np.array([1])

        msg = r"unsupported operand type\(s\) for \+: 'Timedelta' and 'int'"
        with pytest.raises(TypeError, match=msg):
            td + np.array([1])

        msg = "|".join(
            [
                (
                    r"unsupported operand type\(s\) for \+: 'numpy.ndarray' "
                    "and 'Timedelta'"
                ),
                # This message goes on to say "Please do not rely on this error;
                #  it may not be given on all Python implementations"
                "Concatenation operation is not implemented for NumPy arrays",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            other + td
        msg = r"unsupported operand type\(s\) for -: 'Timedelta' and 'int'"
        with pytest.raises(TypeError, match=msg):
            td - other
        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            other - td

    def test_td_rsub_nat(self):
        # [FUNCTION DOC]
        # Subtracting a Timedelta from NaT, or from a missing numpy datetime, gives NaT.
        td = Timedelta(10, unit="D")
        result = NaT - td
        assert result is NaT

        result = np.datetime64("NaT", "ns") - td
        assert result is NaT

    def test_td_rsub_offset(self):
        # [FUNCTION DOC]
        # Subtracting a Timedelta from an hour offset gives a Timedelta of minus 239 hours.
        result = offsets.Hour(1) - Timedelta(10, unit="D")
        assert isinstance(result, Timedelta)
        assert result == Timedelta(-239, unit="h")

    def test_td_sub_timedeltalike_object_dtype_array(self):
        # [FUNCTION DOC]
        # GH#21980. An object array of timestamps minus a Timedelta gives an object array of timestamps shifted back by that duration.
        # GH#21980
        arr = np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")])
        exp = np.array([Timestamp("20121231 9:01"), Timestamp("20121229 9:02")])
        res = arr - Timedelta("1D")
        tm.assert_numpy_array_equal(res, exp)

    def test_td_sub_mixed_most_timedeltalike_object_dtype_array(self):
        # [FUNCTION DOC]
        # GH#21980. An object array mixing a timestamp, a Timedelta and a numpy timedelta, minus a Timedelta, subtracts element-wise: the timestamp shifts, the Timedelta becomes zero, and the numpy timedelta is reduced.
        # GH#21980
        now = Timestamp("2021-11-09 09:54:00")
        arr = np.array([now, Timedelta("1D"), np.timedelta64(2, "h")])
        exp = np.array(
            [
                now - Timedelta("1D"),
                Timedelta("0D"),
                np.timedelta64(2, "h") - Timedelta("1D"),
            ]
        )
        res = arr - Timedelta("1D")
        tm.assert_numpy_array_equal(res, exp)

    def test_td_rsub_mixed_most_timedeltalike_object_dtype_array(self):
        # [FUNCTION DOC]
        # GH#21980. Subtracting a mixed object array (containing a timestamp) from a Timedelta must raise a type error naming the unsupported operand types.
        # GH#21980
        now = Timestamp("2021-11-09 09:54:00")
        arr = np.array([now, Timedelta("1D"), np.timedelta64(2, "h")])
        msg = r"unsupported operand type\(s\) for \-: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            Timedelta("1D") - arr

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_timedeltalike_object_dtype_array(self, op):
        # [FUNCTION DOC]
        # GH#21980. Parametrized over forward and reflected addition. An object array of timestamps plus a Timedelta gives an object array of shifted timestamps.
        # GH#21980
        arr = np.array([Timestamp("20130101 9:01"), Timestamp("20121230 9:02")])
        exp = np.array([Timestamp("20130102 9:01"), Timestamp("20121231 9:02")])
        res = op(arr, Timedelta("1D"))
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_td_add_mixed_timedeltalike_object_dtype_array(self, op):
        # [FUNCTION DOC]
        # GH#21980. Parametrized over forward and reflected addition. An object array mixing a timestamp and a Timedelta plus a Timedelta adds element-wise.
        # GH#21980
        now = Timestamp("2021-11-09 09:54:00")
        arr = np.array([now, Timedelta("1D")])
        exp = np.array([now + Timedelta("1D"), Timedelta("2D")])
        res = op(arr, Timedelta("1D"))
        tm.assert_numpy_array_equal(res, exp)

    def test_td_add_sub_td64_ndarray(self):
        # [FUNCTION DOC]
        # A numpy timedelta array of one element plus or minus a Timedelta, in both orders, returns a numpy array: the doubled duration for sums and zero for differences.
        td = Timedelta("1 day")

        other = np.array([td.to_timedelta64()])
        expected = np.array([Timedelta("2 Days").to_timedelta64()])

        result = td + other
        tm.assert_numpy_array_equal(result, expected)
        result = other + td
        tm.assert_numpy_array_equal(result, expected)

        result = td - other
        tm.assert_numpy_array_equal(result, expected * 0)
        result = other - td
        tm.assert_numpy_array_equal(result, expected * 0)

    def test_td_add_sub_dt64_ndarray(self):
        # [FUNCTION DOC]
        # A numpy datetime array plus a Timedelta (either order) is shifted forward by a day; subtracting the Timedelta (and adding its negation) shifts it back by a day. Results are numpy arrays of the same datetime dtype.
        td = Timedelta("1 day")
        other = np.array(["2000-01-01"], dtype="M8[ns]")

        expected = np.array(["2000-01-02"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(td + other, expected)
        tm.assert_numpy_array_equal(other + td, expected)

        expected = np.array(["1999-12-31"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(-td + other, expected)
        tm.assert_numpy_array_equal(other - td, expected)

    def test_td_add_sub_ndarray_0d(self):
        # [FUNCTION DOC]
        # A zero-dimensional array holding a numpy timedelta, added or subtracted with a Timedelta in either order, gives a Timedelta scalar (not an array): double the duration for sums and zero for differences.
        td = Timedelta("1 day")
        other = np.array(td.asm8)

        result = td + other
        assert isinstance(result, Timedelta)
        assert result == 2 * td

        result = other + td
        assert isinstance(result, Timedelta)
        assert result == 2 * td

        result = other - td
        assert isinstance(result, Timedelta)
        assert result == 0 * td

        result = td - other
        assert isinstance(result, Timedelta)
        assert result == 0 * td


class TestTimedeltaMultiplicationDivision:
    # [CLASS DOC]
    # Groups tests of multiplication, division, floor division, modulo and divmod (forward and reflected) involving a Timedelta.
    """
    Tests for Timedelta methods:

        __mul__, __rmul__,
        __div__, __rdiv__,
        __truediv__, __rtruediv__,
        __floordiv__, __rfloordiv__,
        __mod__, __rmod__,
        __divmod__, __rdivmod__
    """

    # ---------------------------------------------------------------
    # Timedelta.__mul__, __rmul__

    @pytest.mark.parametrize("td_nat", [NaT, np.timedelta64("NaT", "ns")])
    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_nat(self, op, td_nat):
        # [FUNCTION DOC]
        # GH#19819. Parametrized over multiplication forward and reflected, and over NaT and a missing numpy timedelta. Multiplying a Timedelta by a time-like missing value must raise a type error about unsupported operand types or a numpy multiply ufunc that cannot use these operand types.
        # GH#19819
        td = Timedelta(10, unit="D")
        typs = "|".join(["numpy.timedelta64", "NaTType", "Timedelta"])
        msg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typs}' and '{typs}'",
                r"ufunc '?multiply'? cannot use operands with types",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            op(td, td_nat)

    @pytest.mark.parametrize("nan", [np.nan, np.float64("NaN"), float("nan")])
    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_nan(self, op, nan):
        # [FUNCTION DOC]
        # Parametrized over forward and reflected multiplication and over three kinds of nan. Multiplying a Timedelta by nan gives NaT (the numpy float variant is checked to not be treated as an array).
        # np.float64('NaN') has a 'dtype' attr, avoid treating as array
        td = Timedelta(10, unit="D")
        result = op(td, nan)
        assert result is NaT

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul])
    def test_td_mul_scalar(self, op):
        # [FUNCTION DOC]
        # GH#19738. Parametrized over forward and reflected multiplication. Multiplying by integers and floats scales the duration (including negative factors, compared on the internal value), by nan gives NaT; multiplying by a timestamp or by another Timedelta must raise a type error for unsupported operands.
        # GH#19738
        td = Timedelta(minutes=3)

        result = op(td, 2)
        assert result == Timedelta(minutes=6)

        result = op(td, 1.5)
        assert result == Timedelta(minutes=4, seconds=30)

        assert op(td, np.nan) is NaT

        assert op(-1, td)._value == -1 * td._value
        assert op(-1.0, td)._value == -1.0 * td._value

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            # timedelta * datetime is gibberish
            op(td, Timestamp(2016, 1, 2))

        with pytest.raises(TypeError, match=msg):
            # invalid multiply with another timedelta
            op(td, td)

    def test_td_mul_numeric_ndarray(self):
        # [FUNCTION DOC]
        # Multiplying a Timedelta by a one-element integer array, in either order, gives a numpy timedelta array with the scaled duration.
        td = Timedelta("1 day")
        other = np.array([2])
        expected = np.array([Timedelta("2 Days").to_timedelta64()])

        result = td * other
        tm.assert_numpy_array_equal(result, expected)

        result = other * td
        tm.assert_numpy_array_equal(result, expected)

    def test_td_mul_numeric_ndarray_0d(self):
        # [FUNCTION DOC]
        # Multiplying a Timedelta by a zero-dimensional integer array, in either order, gives a plain Timedelta scalar equal to the scaled duration.
        td = Timedelta("1 day")
        other = np.array(2, dtype=np.int64)
        assert other.ndim == 0
        expected = Timedelta("2 days")

        res = td * other
        assert type(res) is Timedelta
        assert res == expected

        res = other * td
        assert type(res) is Timedelta
        assert res == expected

    def test_td_mul_td64_ndarray_invalid(self):
        # [FUNCTION DOC]
        # Multiplying a Timedelta by an array of numpy timedeltas, in either order, must raise a type error whose message says the multiply ufunc cannot use operands with two timedelta microsecond dtypes.
        td = Timedelta("1 day")
        other = np.array([Timedelta("2 Days").to_timedelta64()])

        msg = (
            "ufunc '?multiply'? cannot use operands with types "
            rf"dtype\('{tm.ENDIAN}m8\[us\]'\) and dtype\('{tm.ENDIAN}m8\[us\]'\)"
        )
        with pytest.raises(TypeError, match=msg):
            td * other
        with pytest.raises(TypeError, match=msg):
            other * td

    # ---------------------------------------------------------------
    # Timedelta.__div__, __truediv__

    def test_td_div_timedeltalike_scalar(self):
        # [FUNCTION DOC]
        # GH#19738. True division of a Timedelta by a time-like value (hour offset, Timedelta, numpy timedelta) gives the ratio as a float (for example 240, 1 and 4); dividing by NaT gives not-a-number.
        # GH#19738
        td = Timedelta(10, unit="D")

        result = td / offsets.Hour(1)
        assert result == 240

        assert td / td == 1
        assert td / np.timedelta64(60, "h") == 4

        assert np.isnan(td / NaT)

    def test_td_div_td64_non_nano(self):
        # [FUNCTION DOC]
        # True division and floor division by numpy timedeltas of day, second and nanosecond units give ratios computed from the internal nanosecond value (floor division gives integers). The Timedelta has a nanosecond component so resolution is nanoseconds.
        # truediv
        td = Timedelta("1 days 2 hours 3 ns")
        result = td / np.timedelta64(1, "D")
        assert result == td._value / (86400 * 10**9)
        result = td / np.timedelta64(1, "s")
        assert result == td._value / 10**9
        result = td / np.timedelta64(1, "ns")
        assert result == td._value

        # floordiv
        td = Timedelta("1 days 2 hours 3 ns")
        result = td // np.timedelta64(1, "D")
        assert result == 1
        result = td // np.timedelta64(1, "s")
        assert result == 93600
        result = td // np.timedelta64(1, "ns")
        assert result == td._value

    def test_td_div_numeric_scalar(self):
        # [FUNCTION DOC]
        # GH#19738. Dividing a Timedelta by an integer gives a Timedelta of the divided duration.
        # GH#19738
        td = Timedelta(10, unit="D")

        result = td / 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=5)

        result = td / 5
        assert isinstance(result, Timedelta)
        assert result == Timedelta(days=2)

    @pytest.mark.parametrize(
        "nan",
        [
            np.nan,
            np.float64("NaN"),
            float("nan"),
        ],
    )
    def test_td_div_nan(self, nan):
        # [FUNCTION DOC]
        # Parametrized over three kinds of nan. True division and floor division of a Timedelta by nan both give NaT (the numpy float variant is checked to not be treated as an array).
        # np.float64('NaN') has a 'dtype' attr, avoid treating as array
        td = Timedelta(10, unit="D")
        result = td / nan
        assert result is NaT

        result = td // nan
        assert result is NaT

    def test_td_div_td64_ndarray(self):
        # [FUNCTION DOC]
        # A Timedelta divided by an array of numpy timedeltas gives a float array of ratios (one half for a double duration); the reverse division gives the inverse ratios (four times the previous expected array).
        td = Timedelta("1 day")

        other = np.array([Timedelta("2 Days").to_timedelta64()])
        expected = np.array([0.5])

        result = td / other
        tm.assert_numpy_array_equal(result, expected)

        result = other / td
        tm.assert_numpy_array_equal(result, expected * 4)

    def test_td_div_ndarray_0d(self):
        # [FUNCTION DOC]
        # Dividing a Timedelta by a zero-dimensional integer array holding one returns a Timedelta equal to the original (scalar result, not an array).
        td = Timedelta("1 day")

        other = np.array(1)
        res = td / other
        assert isinstance(res, Timedelta)
        assert res == td

    # ---------------------------------------------------------------
    # Timedelta.__rdiv__

    def test_td_rdiv_timedeltalike_scalar(self):
        # [FUNCTION DOC]
        # GH#19738. Reflected true division: an hour offset divided by a ten-day Timedelta and a numpy timedelta divided by it give the expected float ratios.
        # GH#19738
        td = Timedelta(10, unit="D")
        result = offsets.Hour(1) / td
        assert result == 1 / 240.0

        assert np.timedelta64(60, "h") / td == 0.25

    def test_td_rdiv_na_scalar(self):
        # [FUNCTION DOC]
        # GH#31869. Dividing NaT, None, or a missing numpy timedelta by a Timedelta gives not-a-number; dividing a missing numpy datetime or a float nan by it must raise a type error for unsupported operands.
        # GH#31869 None gets cast to NaT
        td = Timedelta(10, unit="D")

        result = NaT / td
        assert np.isnan(result)

        result = None / td
        assert np.isnan(result)

        result = np.timedelta64("NaT", "ns") / td
        assert np.isnan(result)

        msg = r"unsupported operand type\(s\) for /: 'numpy.datetime64' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.datetime64("NaT", "ns") / td

        msg = r"unsupported operand type\(s\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            np.nan / td

    def test_td_rdiv_ndarray(self):
        # [FUNCTION DOC]
        # Dividing an object array holding a Timedelta by a Timedelta gives a float array of ones; an array holding None gives not-a-number; an object array holding a float nan or a float array of nan must raise a type error (unsupported operand or ufunc operand dtype).
        td = Timedelta(10, unit="D")

        arr = np.array([td], dtype=object)
        result = arr / td
        expected = np.array([1], dtype=np.float64)
        tm.assert_numpy_array_equal(result, expected)

        arr = np.array([None])
        result = arr / td
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

        arr = np.array([np.nan], dtype=object)
        msg = r"unsupported operand type\(s\) for /: 'float' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            arr / td

        arr = np.array([np.nan], dtype=np.float64)
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            arr / td

    def test_td_rdiv_ndarray_0d(self):
        # [FUNCTION DOC]
        # A zero-dimensional numpy timedelta array divided by an equal Timedelta gives one.
        td = Timedelta(10, unit="D")

        arr = np.array(td.asm8)

        assert arr / td == 1

    # ---------------------------------------------------------------
    # Timedelta.__floordiv__

    def test_td_floordiv_timedeltalike_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. Floor division of a Timedelta by a Timedelta, standard timedelta and numpy timedelta gives integers (including a negative dividend rounding toward minus infinity).
        # GH#18846
        td = Timedelta(hours=3, minutes=4)
        scalar = Timedelta(hours=3, minutes=3)

        assert td // scalar == 1
        assert -td // scalar.to_pytimedelta() == -2
        assert (2 * td) // scalar.to_timedelta64() == 2

    def test_td_floordiv_null_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. Floor division by nan gives NaT; by NaT or a missing numpy timedelta gives not-a-number.
        # GH#18846
        td = Timedelta(hours=3, minutes=4)

        assert td // np.nan is NaT
        assert np.isnan(td // NaT)
        assert np.isnan(td // np.timedelta64("NaT", "ns"))

    def test_td_floordiv_offsets(self):
        # [FUNCTION DOC]
        # GH#19738. Floor division of a Timedelta by hour and minute offsets gives integer counts.
        # GH#19738
        td = Timedelta(hours=3, minutes=4)
        assert td // offsets.Hour(1) == 3
        assert td // offsets.Minute(2) == 92

    def test_td_floordiv_invalid_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. Floor division by a numpy datetime must raise a type error (message varies by numpy version) about an invalid dtype or unsupported operand types.
        # GH#18846
        td = Timedelta(hours=3, minutes=4)

        msg = "|".join(
            [
                r"Invalid dtype datetime64\[D\] for __floordiv__",
                "'dtype' is an invalid keyword argument for this function",
                "this function got an unexpected keyword argument 'dtype'",
                r"ufunc '?floor_divide'? cannot use operands with types",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            td // np.datetime64("2016-01-01", dtype="datetime64[us]")

    def test_td_floordiv_numeric_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. Floor division of a Timedelta by integers and floats of several types (Python and numpy) gives a Timedelta of half the duration.
        # GH#18846
        td = Timedelta(hours=3, minutes=4)

        expected = Timedelta(hours=1, minutes=32)
        assert td // 2 == expected
        assert td // 2.0 == expected
        assert td // np.float64(2.0) == expected
        assert td // np.int32(2.0) == expected
        assert td // np.uint8(2.0) == expected

    def test_td_floordiv_timedeltalike_array(self):
        # [FUNCTION DOC]
        # GH#18846. Floor division by zero-dimensional and one-dimensional numpy timedelta arrays gives an integer or integer array; an array containing a missing value gives a float array with nan for the missing element.
        # GH#18846
        td = Timedelta(hours=3, minutes=4)
        scalar = Timedelta(hours=3, minutes=3)

        # Array-like others
        assert td // np.array(scalar.to_timedelta64()) == 1

        res = (3 * td) // np.array([scalar.to_timedelta64()])
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)

        res = (10 * td) // np.array(
            [scalar.to_timedelta64(), np.timedelta64("NaT", "ns")]
        )
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_floordiv_numeric_series(self):
        # [FUNCTION DOC]
        # GH#18846. Floor division of a Timedelta by an integer series gives a series of timedelta dtype.
        # GH#18846
        td = Timedelta(hours=3, minutes=4)
        ser = pd.Series([1], dtype=np.int64)
        res = td // ser
        assert res.dtype.kind == "m"

    # ---------------------------------------------------------------
    # Timedelta.__rfloordiv__

    def test_td_rfloordiv_timedeltalike_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. Calls the reflected floor-division method directly with a Timedelta, standard timedelta and numpy timedelta, returning integers (including negative dividends rounding toward minus infinity).
        # GH#18846
        td = Timedelta(hours=3, minutes=3)
        scalar = Timedelta(hours=3, minutes=4)

        # scalar others
        # x // Timedelta is defined only for timedelta-like x. int-like,
        # float-like, and date-like, in particular, should all either
        # a) raise TypeError directly or
        # b) return NotImplemented, following which the reversed
        #    operation will raise TypeError.
        assert td.__rfloordiv__(scalar) == 1
        assert (-td).__rfloordiv__(scalar.to_pytimedelta()) == -2
        assert (2 * td).__rfloordiv__(scalar.to_timedelta64()) == 0

    def test_td_rfloordiv_null_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. The reflected floor-division method with NaT or a missing numpy timedelta returns not-a-number.
        # GH#18846
        td = Timedelta(hours=3, minutes=3)

        assert np.isnan(td.__rfloordiv__(NaT))
        assert np.isnan(td.__rfloordiv__(np.timedelta64("NaT", "ns")))

    def test_td_rfloordiv_offsets(self):
        # [FUNCTION DOC]
        # GH#19738. An hour offset floor-divided by a 25-minute Timedelta gives two.
        # GH#19738
        assert offsets.Hour(1) // Timedelta(minutes=25) == 2

    def test_td_rfloordiv_invalid_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. The reflected floor-division method with a numpy datetime returns the not-implemented marker, and the operator itself raises a type error for unsupported operand types.
        # GH#18846
        td = Timedelta(hours=3, minutes=3)

        dt64 = np.datetime64("2016-01-01", "us")

        assert td.__rfloordiv__(dt64) is NotImplemented

        msg = (
            r"unsupported operand type\(s\) for //: 'numpy.datetime64' and 'Timedelta'"
        )
        with pytest.raises(TypeError, match=msg):
            dt64 // td

    def test_td_rfloordiv_numeric_scalar(self):
        # [FUNCTION DOC]
        # GH#18846. The reflected floor-division method with numbers (floats, ints, numpy numbers, nan) returns the not-implemented marker, and the operator raises a type error (the integer case was deprecated in GH#19761 and enforced in GH#29797).
        # GH#18846
        td = Timedelta(hours=3, minutes=3)

        assert td.__rfloordiv__(np.nan) is NotImplemented
        assert td.__rfloordiv__(3.5) is NotImplemented
        assert td.__rfloordiv__(2) is NotImplemented
        assert td.__rfloordiv__(np.float64(2.0)) is NotImplemented
        assert td.__rfloordiv__(np.uint8(9)) is NotImplemented
        assert td.__rfloordiv__(np.int32(2.0)) is NotImplemented

        msg = r"unsupported operand type\(s\) for //: '.*' and 'Timedelta"
        with pytest.raises(TypeError, match=msg):
            np.float64(2.0) // td
        with pytest.raises(TypeError, match=msg):
            np.uint8(9) // td
        with pytest.raises(TypeError, match=msg):
            # deprecated GH#19761, enforced GH#29797
            np.int32(2.0) // td

    def test_td_rfloordiv_timedeltalike_array(self):
        # [FUNCTION DOC]
        # GH#18846. The reflected floor-division method applied to zero-dimensional, one-dimensional and nan-containing numpy timedelta arrays returns integer or float results as appropriate.
        # GH#18846
        td = Timedelta(hours=3, minutes=3)
        scalar = Timedelta(hours=3, minutes=4)

        # Array-like others
        assert td.__rfloordiv__(np.array(scalar.to_timedelta64())) == 1

        res = td.__rfloordiv__(np.array([(3 * scalar).to_timedelta64()]))
        expected = np.array([3], dtype=np.int64)
        tm.assert_numpy_array_equal(res, expected)

        arr = np.array([(10 * scalar).to_timedelta64(), np.timedelta64("NaT", "ns")])
        res = td.__rfloordiv__(arr)
        expected = np.array([10, np.nan])
        tm.assert_numpy_array_equal(res, expected)

    def test_td_rfloordiv_intarray(self):
        # [FUNCTION DOC]
        # Deprecated in GH#19761 and enforced in GH#29797. Floor-dividing an integer array by a Timedelta must raise a type error about an invalid dtype.
        # deprecated GH#19761, enforced GH#29797
        ints = np.array([1349654400, 1349740800, 1349827200, 1349913600]) * 10**9

        msg = "Invalid dtype"
        with pytest.raises(TypeError, match=msg):
            ints // Timedelta(1, unit="s")

    def test_td_rfloordiv_numeric_series(self):
        # [FUNCTION DOC]
        # GH#18846. The reflected floor-division method with an integer series returns the not-implemented marker, and the operator raises a type error about an invalid dtype (deprecated GH#19761, enforced GH#29797).
        # GH#18846
        td = Timedelta(hours=3, minutes=3)
        ser = pd.Series([1], dtype=np.int64)
        res = td.__rfloordiv__(ser)
        assert res is NotImplemented

        msg = "Invalid dtype"
        with pytest.raises(TypeError, match=msg):
            # Deprecated GH#19761, enforced GH#29797
            ser // td

    # ----------------------------------------------------------------
    # Timedelta.__mod__, __rmod__

    def test_mod_timedeltalike(self):
        # [FUNCTION DOC]
        # GH#19365. Modulo of a 37-hour Timedelta by a Timedelta or standard timedelta gives a Timedelta remainder; by NaT gives NaT.
        # GH#19365
        td = Timedelta(hours=37)

        # Timedelta-like others
        result = td % Timedelta(hours=6)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

        result = td % timedelta(minutes=60)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        result = td % NaT
        assert result is NaT

    def test_mod_timedelta64_nat(self):
        # [FUNCTION DOC]
        # GH#19365. Modulo by a missing numpy timedelta gives NaT.
        # GH#19365
        td = Timedelta(hours=37)

        result = td % np.timedelta64("NaT", "ns")
        assert result is NaT

    def test_mod_timedelta64(self):
        # [FUNCTION DOC]
        # GH#19365. Modulo by a numpy timedelta gives a Timedelta remainder.
        # GH#19365
        td = Timedelta(hours=37)

        result = td % np.timedelta64(2, "h")
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=1)

    def test_mod_offset(self):
        # [FUNCTION DOC]
        # GH#19365. Modulo by an hour offset gives a Timedelta remainder.
        # GH#19365
        td = Timedelta(hours=37)

        result = td % offsets.Hour(5)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(hours=2)

    def test_mod_numeric(self):
        # [FUNCTION DOC]
        # GH#19365. Modulo by numbers (an integer, a float, an integer-valued float count) treats the number as nanoseconds and gives a Timedelta remainder.
        # GH#19365
        td = Timedelta(hours=37)

        # Numeric Others
        result = td % 2
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        result = td % 1e9
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

        result = td % int(1e9)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=3, seconds=20)

    def test_mod_invalid(self):
        # [FUNCTION DOC]
        # GH#19365. Modulo by a timestamp or by a list must raise a type error for unsupported operands.
        # GH#19365
        td = Timedelta(hours=37)
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            td % Timestamp("2018-01-22")

        with pytest.raises(TypeError, match=msg):
            td % []

    def test_rmod_pytimedelta(self):
        # [FUNCTION DOC]
        # GH#19365. A standard timedelta modulo a Timedelta gives a Timedelta remainder.
        # GH#19365
        td = Timedelta(minutes=3)

        result = timedelta(minutes=4) % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=1)

    def test_rmod_timedelta64(self):
        # [FUNCTION DOC]
        # GH#19365. A numpy timedelta modulo a Timedelta gives a Timedelta remainder.
        # GH#19365
        td = Timedelta(minutes=3)
        result = np.timedelta64(5, "m") % td
        assert isinstance(result, Timedelta)
        assert result == Timedelta(minutes=2)

    def test_rmod_invalid(self):
        # [FUNCTION DOC]
        # GH#19365. A timestamp, integer or float modulo a Timedelta must raise a type error for unsupported operands; an integer numpy array must raise a type error about an invalid integer dtype.
        # GH#19365
        td = Timedelta(minutes=3)

        msg = "unsupported operand"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2018-01-22") % td

        with pytest.raises(TypeError, match=msg):
            15 % td

        with pytest.raises(TypeError, match=msg):
            16.0 % td

        msg = "Invalid dtype int"
        with pytest.raises(TypeError, match=msg):
            np.array([22, 24]) % td

    # ----------------------------------------------------------------
    # Timedelta.__divmod__, __rdivmod__

    def test_divmod_numeric(self):
        # [FUNCTION DOC]
        # GH#19365. Divmod by a number treats it as nanoseconds: the quotient is a Timedelta and the remainder a Timedelta; divmod by nan gives NaT for both.
        # KNOWN ISSUE: the line asserting the truthiness of the result is a no-op (a non-empty tuple is always true) and checks nothing.
        # GH#19365
        td = Timedelta(days=2, hours=6)

        result = divmod(td, 53 * 3600 * 1e6)
        assert result[0] == Timedelta(1, unit="us").as_unit("us")
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=1)

        assert result
        result = divmod(td, np.nan)
        assert result[0] is NaT
        assert result[1] is NaT

    def test_divmod(self):
        # [FUNCTION DOC]
        # GH#19365. Divmod by a standard timedelta gives an integer quotient and Timedelta remainder; by an integer gives a Timedelta quotient; by NaT gives not-a-number and NaT.
        # GH#19365
        td = Timedelta(days=2, hours=6)

        result = divmod(td, timedelta(days=1))
        assert result[0] == 2
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

        result = divmod(td, 54)
        assert result[0] == Timedelta(hours=1)
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(0)

        result = divmod(td, NaT)
        assert np.isnan(result[0])
        assert result[1] is NaT

    def test_divmod_offset(self):
        # [FUNCTION DOC]
        # GH#19365. Divmod by a negative hour offset gives a negative integer quotient and a negative Timedelta remainder.
        # GH#19365
        td = Timedelta(days=2, hours=6)

        result = divmod(td, offsets.Hour(-4))
        assert result[0] == -14
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_divmod_invalid(self):
        # [FUNCTION DOC]
        # GH#19365. Divmod by a timestamp must raise a type error mentioning the floor-division operator with the Timedelta and Timestamp types.
        # GH#19365
        td = Timedelta(days=2, hours=6)

        msg = r"unsupported operand type\(s\) for //: 'Timedelta' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            divmod(td, Timestamp("2018-01-22"))

    def test_rdivmod_pytimedelta(self):
        # [FUNCTION DOC]
        # GH#19365. Reflected divmod with a standard timedelta numerator gives an integer quotient and Timedelta remainder.
        # GH#19365
        result = divmod(timedelta(days=2, hours=6), Timedelta(days=1))
        assert result[0] == 2
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=6)

    def test_rdivmod_offset(self):
        # [FUNCTION DOC]
        # Reflected divmod with an hour offset numerator and a negative-hour Timedelta gives a negative integer quotient and negative Timedelta remainder.
        result = divmod(offsets.Hour(54), Timedelta(hours=-4))
        assert result[0] == -14
        assert isinstance(result[1], Timedelta)
        assert result[1] == Timedelta(hours=-2)

    def test_rdivmod_invalid(self):
        # [FUNCTION DOC]
        # GH#19365. Divmod with a timestamp, integer or float numerator and a Timedelta must raise a type error for unsupported operands; with an integer array it must raise a type error about an invalid integer dtype.
        # GH#19365
        td = Timedelta(minutes=3)
        msg = "unsupported operand type"

        with pytest.raises(TypeError, match=msg):
            divmod(Timestamp("2018-01-22"), td)

        with pytest.raises(TypeError, match=msg):
            divmod(15, td)

        with pytest.raises(TypeError, match=msg):
            divmod(16.0, td)

        msg = "Invalid dtype int"
        with pytest.raises(TypeError, match=msg):
            divmod(np.array([22, 24]), td)

    # ----------------------------------------------------------------

    @pytest.mark.parametrize(
        "op", [operator.mul, ops.rmul, operator.truediv, ops.rdiv, ops.rsub]
    )
    @pytest.mark.parametrize(
        "arr",
        [
            [Timestamp("20130101 9:01"), Timestamp("20121230 9:02")],
            [Timestamp("2021-11-09 09:54:00"), Timedelta("1D")],
        ],
    )
    def test_td_op_timedelta_timedeltalike_array(self, op, arr):
        # [FUNCTION DOC]
        # Parametrized over multiplication, reflected multiplication, true division, reflected division and reflected subtraction, and two object arrays containing timestamps (and a Timedelta). Applying the operator between the array and a Timedelta must raise a type error for unsupported operands or an unsupported ufunc operand type.
        arr = np.array(arr)
        msg = "unsupported operand type|cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            op(arr, Timedelta("1D"))

    def test_mul_bool_invalid(self):
        # [FUNCTION DOC]
        # GH#62316. Multiplying a Timedelta by a boolean must raise a type error telling the user to cast to an integer explicitly.
        # GH#62316
        msg = "Cannot multiply Timedelta by bool. Explicitly cast to integer"
        with pytest.raises(TypeError, match=msg):
            Timedelta("1 day") * True


class TestTimedeltaComparison:
    # [CLASS DOC]
    # Groups tests of the rich comparison operators on Timedelta.
    def test_compare_pytimedelta_bounds(self):
        # [FUNCTION DOC]
        # GH#49021. Comparing the maximum and minimum Timedelta at nanosecond and microsecond resolution with the extreme standard timedelta values must not overflow: the Timedelta extremes are strictly inside. For millisecond and second resolution the extremes extend beyond the standard bounds.
        # All six comparison operators are checked.
        # GH#49021 don't overflow on comparison with very large pytimedeltas

        for unit in ["ns", "us"]:
            tdmax = Timedelta.max.as_unit(unit).max
            tdmin = Timedelta.min.as_unit(unit).min

            assert tdmax < timedelta.max
            assert tdmax <= timedelta.max
            assert not tdmax > timedelta.max
            assert not tdmax >= timedelta.max
            assert tdmax != timedelta.max
            assert not tdmax == timedelta.max

            assert tdmin > timedelta.min
            assert tdmin >= timedelta.min
            assert not tdmin < timedelta.min
            assert not tdmin <= timedelta.min
            assert tdmin != timedelta.min
            assert not tdmin == timedelta.min

        # But the "ms" and "s"-reso bounds extend pass pytimedelta
        for unit in ["ms", "s"]:
            tdmax = Timedelta.max.as_unit(unit).max
            tdmin = Timedelta.min.as_unit(unit).min

            assert tdmax > timedelta.max
            assert tdmax >= timedelta.max
            assert not tdmax < timedelta.max
            assert not tdmax <= timedelta.max
            assert tdmax != timedelta.max
            assert not tdmax == timedelta.max

            assert tdmin < timedelta.min
            assert tdmin <= timedelta.min
            assert not tdmin > timedelta.min
            assert not tdmin >= timedelta.min
            assert tdmin != timedelta.min
            assert not tdmin == timedelta.min

    def test_compare_pytimedelta_bounds2(self):
        # [FUNCTION DOC]
        # A standard timedelta of nearly the maximum days (outside microsecond bounds) is rebuilt as a Timedelta via a numpy sum of days and seconds, with matching days and seconds fields and equal under all comparisons. One second less compares as smaller and unequal.
        # a pytimedelta outside the microsecond bounds
        pytd = timedelta(days=999999999, seconds=86399)
        # NB: np.timedelta64(td, "s"") incorrectly overflows
        td64 = np.timedelta64(pytd.days, "D") + np.timedelta64(pytd.seconds, "s")
        td = Timedelta(td64)
        assert td.days == pytd.days
        assert td.seconds == pytd.seconds

        assert td == pytd
        assert not td != pytd
        assert not td < pytd
        assert not td > pytd
        assert td <= pytd
        assert td >= pytd

        td2 = td - Timedelta(seconds=1).as_unit("s")
        assert td2 != pytd
        assert not td2 == pytd
        assert td2 < pytd
        assert td2 <= pytd
        assert not td2 > pytd
        assert not td2 >= pytd

    def test_compare_tick(self, tick_classes):
        # [FUNCTION DOC]
        # Parametrized over the fixed-length offset class fixture. A Timedelta made from an offset of four units compares equal, less-or-equal and greater-or-equal to it, and orders properly against twice the offset.
        cls = tick_classes

        off = cls(4)
        td = off._as_pd_timedelta
        assert isinstance(td, Timedelta)

        assert td == off
        assert not td != off
        assert td <= off
        assert td >= off
        assert not td < off
        assert not td > off

        assert not td == 2 * off
        assert td != 2 * off
        assert td <= 2 * off
        assert td < 2 * off
        assert not td >= 2 * off
        assert not td > 2 * off

    def test_comparison_object_array(self):
        # [FUNCTION DOC]
        # Comparing an object array of Timedeltas (one- and two-dimensional) with a Timedelta by equality and inequality gives boolean arrays of the same shape with the expected pattern (analogous to GH#15183).
        # analogous to GH#15183
        td = Timedelta("2 days")
        other = Timedelta("3 hours")

        arr = np.array([other, td], dtype=object)
        res = arr == td
        expected = np.array([False, True], dtype=bool)
        assert (res == expected).all()

        # 2D case
        arr = np.array([[other, td], [td, other]], dtype=object)
        res = arr != td
        expected = np.array([[True, False], [False, True]], dtype=bool)
        assert res.shape == expected.shape
        assert (res == expected).all()

    def test_compare_timedelta_ndarray(self):
        # [FUNCTION DOC]
        # GH#11835. Comparing one element of an object array of equal Timedeltas with the whole array using greater-than gives all false.
        # GH#11835
        periods = [Timedelta("0 days 01:00:00"), Timedelta("0 days 01:00:00")]
        arr = np.array(periods)
        result = arr[0] > arr
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_compare_td64_ndarray(self):
        # [FUNCTION DOC]
        # GG#33441 (sic). Equality and inequality between a Timedelta and a numpy timedelta array, in either order, give boolean arrays flagging only the matching element.
        # GG#33441
        arr = np.arange(5).astype("timedelta64[ns]")
        td = Timedelta(arr[1])

        expected = np.array([False, True, False, False, False], dtype=bool)

        result = td == arr
        tm.assert_numpy_array_equal(result, expected)

        result = arr == td
        tm.assert_numpy_array_equal(result, expected)

        result = td != arr
        tm.assert_numpy_array_equal(result, ~expected)

        result = arr != td
        tm.assert_numpy_array_equal(result, ~expected)

    def test_compare_custom_object(self):
        # [FUNCTION DOC]
        # GH#20829. Unsupported comparisons return the not-implemented marker so the other operand's reflected method decides. Uses a local helper class whose comparison results are configurable. Unequal to strings, integers, and to helper instances that defer or return false;
        # less than a helper that says true; equal to a helper that returns true.
        """
        Make sure non supported operations on Timedelta returns NonImplemented
        and yields to other operand (GH#20829).
        """

        class CustomClass:
            # [CLASS DOC]
            # HELPER: local class used by the custom-object comparison test. Stores a configurable result and defines equality and greater-than that return that result, or the not-implemented marker when none is configured.
            def __init__(self, cmp_result=None) -> None:
                # [FUNCTION DOC]
                # HELPER: constructor storing the configured comparison result (default none, meaning defer to the other operand).
                self.cmp_result = cmp_result

            def generic_result(self):
                # [FUNCTION DOC]
                # HELPER: returns the stored comparison result, or the not-implemented marker when none is stored.
                if self.cmp_result is None:
                    return NotImplemented
                else:
                    return self.cmp_result

            def __eq__(self, other):
                # [FUNCTION DOC]
                # HELPER: equality operator returning the generic result regardless of the other operand.
                return self.generic_result()

            def __gt__(self, other):
                # [FUNCTION DOC]
                # HELPER: greater-than operator returning the generic result regardless of the other operand (used when a Timedelta is compared with less-than).
                return self.generic_result()

        t = Timedelta("1s")

        assert t != "string"
        assert t != 1
        assert t != CustomClass()
        assert t != CustomClass(cmp_result=False)

        assert t < CustomClass(cmp_result=True)
        assert not t < CustomClass(cmp_result=False)

        assert t == CustomClass(cmp_result=True)

    @pytest.mark.parametrize("val", ["string", 1])
    def test_compare_unknown_type(self, val):
        # [FUNCTION DOC]
        # GH#20829. Parametrized over a string and an integer. Ordering comparisons (all four) with a Timedelta must raise a type error naming the unsupported comparison between the types.
        # GH#20829
        t = Timedelta("1s")
        msg = "not supported between instances of 'Timedelta' and '(int|str)'"
        with pytest.raises(TypeError, match=msg):
            t >= val
        with pytest.raises(TypeError, match=msg):
            t > val
        with pytest.raises(TypeError, match=msg):
            t <= val
        with pytest.raises(TypeError, match=msg):
            t < val


def test_ops_notimplemented():
    # [FUNCTION DOC]
    # For an unrelated plain object, the special methods for addition, subtraction, true division, multiplication and floor division of a Timedelta each return the not-implemented marker.
    class Other:
        # [CLASS DOC]
        # HELPER: empty local class representing an unrelated type, used by the not-implemented test above.
        pass

    other = Other()

    td = Timedelta("1 day")
    assert td.__add__(other) is NotImplemented
    assert td.__sub__(other) is NotImplemented
    assert td.__truediv__(other) is NotImplemented
    assert td.__mul__(other) is NotImplemented
    assert td.__floordiv__(other) is NotImplemented


def test_ops_error_str():
    # [FUNCTION DOC]
    # GH#13624. A Timedelta and a string, in either order: addition must raise a type error (message varies by Python version), ordering comparison must raise a type error, equality is false and inequality is true.
    # GH#13624
    td = Timedelta("1 day")

    for left, right in [(td, "a"), ("a", td)]:
        msg = "|".join(
            [
                "unsupported operand type",
                r'can only concatenate str \(not "Timedelta"\) to str',
                "must be str, not Timedelta",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            left + right

        msg = "not supported between instances of"
        with pytest.raises(TypeError, match=msg):
            left > right

        assert not left == right
        assert left != right


@pytest.mark.parametrize("box", [True, False])
def test_ops_str_deprecated(box):
    # [FUNCTION DOC]
    # GH#59653. Parametrized over a plain string and a one-element object array containing it. Adding, subtracting and reflected operations between a Timedelta and the numeric-looking string emit the major-version deprecation warning about scalar operations with strings.
    # For the plain string, true and floor division in both orders also warn. For the array, the remaining divisions must raise a type error (message varies by operator and numpy version).
    # GH#59653
    td = Timedelta("1 day")
    item = "1"
    if box:
        item = np.array([item], dtype=object)

    msg = "Scalar operations between Timedelta and string are deprecated"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        td + item
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        item + td
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        td - item
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        item - td
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        item / td
    if not box:
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            td / item
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            item // td
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            td // item
    else:
        msg = "|".join(
            [
                "ufunc 'divide' cannot use operands",
                "Invalid dtype object for __floordiv__",
                r"unsupported operand type\(s\) for /: 'int' and 'str'",
                r"unsupported operand type\(s\) for /: 'datetime.timedelta' and 'str'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            td / item
        with pytest.raises(TypeError, match=msg):
            item // td
        with pytest.raises(TypeError, match=msg):
            td // item
