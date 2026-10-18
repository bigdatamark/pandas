# [MODULE DOC]
# Tests for TimedeltaArray: non-nanosecond units (fields, conversion, total
# seconds, arithmetic with NaT, datetime scalars, multiplication and division),
# astype, setitem, searchsorted type validation and unary operators. No
# import-time side effects.

from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)


class TestNonNano:
    # [CLASS DOC]
    # Tests of TimedeltaArray holding non-nanosecond units (second, millisecond,
    # microsecond); several compare results with those of the nanosecond equivalent.
    @pytest.fixture(params=["s", "ms", "us"])
    def unit(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over the units second, millisecond and
        # microsecond. Returns the unit string.
        return request.param

    @pytest.fixture
    def tda(self, unit):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns a TimedeltaArray of the integers 0 to 4 with the
        # fixture unit, built via the private simple constructor.
        arr = np.arange(5, dtype=np.int64).view(f"m8[{unit}]")
        return TimedeltaArray._simple_new(arr, dtype=arr.dtype)

    def test_non_nano(self, unit):
        # [FUNCTION DOC]
        # Verifies a non-nano array built from integer values keeps its dtype and its
        # elements report the unit.
        arr = np.arange(5, dtype=np.int64).view(f"m8[{unit}]")
        tda = TimedeltaArray._simple_new(arr, dtype=arr.dtype)

        assert tda.dtype == arr.dtype
        assert tda[0].unit == unit

    def test_as_unit_raises(self, tda):
        # [FUNCTION DOC]
        # Regression test (GH#50616). Verifies as_unit with an unsupported unit (day)
        # raises a value error listing supported units, for the array and for an Index.
        # GH#50616
        with pytest.raises(ValueError, match="Supported units"):
            tda.as_unit("D")

        tdi = pd.Index(tda)
        with pytest.raises(ValueError, match="Supported units"):
            tdi.as_unit("D")

    @pytest.mark.parametrize("field", TimedeltaArray._field_ops)
    def test_fields(self, tda, field):
        # [FUNCTION DOC]
        # Parametrized over timedelta field properties. Verifies the field of the non-nano
        # array equals the field of the same data in nanoseconds.
        as_nano = tda._ndarray.astype("m8[ns]")
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)

        result = getattr(tda, field)
        expected = getattr(tda_nano, field)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_pytimedelta(self, tda):
        # [FUNCTION DOC]
        # Verifies conversion to Python timedeltas equals that of the nanosecond version.
        as_nano = tda._ndarray.astype("m8[ns]")
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)

        result = tda.to_pytimedelta()
        expected = tda_nano.to_pytimedelta()
        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, unit, tda):
        # [FUNCTION DOC]
        # Verifies total_seconds equals that of the nanosecond version. The unit
        # parameter is unused.
        as_nano = tda._ndarray.astype("m8[ns]")
        tda_nano = TimedeltaArray._simple_new(as_nano, dtype=as_nano.dtype)

        result = tda.total_seconds()
        expected = tda_nano.total_seconds()
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta_array_total_seconds(self):
        # [FUNCTION DOC]
        # Regression test (GH34290). Verifies total_seconds on an array built via the
        # array factory from a two-minute Timedelta gives the scalar's total seconds.
        # GH34290
        expected = Timedelta("2 min").total_seconds()

        result = pd.array([Timedelta("2 min")]).total_seconds()[0]
        assert result == expected

    def test_total_seconds_nanoseconds(self):
        # [FUNCTION DOC]
        # Regression test (issue #48521). Verifies total_seconds on a Series difference of
        # datetimes far in the future matches division by one second of the raw timedelta
        # values.
        # issue #48521
        start_time = pd.Series(["2145-11-02 06:00:00"]).astype("datetime64[ns]")
        end_time = pd.Series(["2145-11-02 07:06:00"]).astype("datetime64[ns]")
        expected = (end_time - start_time).values / np.timedelta64(1, "s")
        result = (end_time - start_time).dt.total_seconds().values
        assert result == expected

    @pytest.mark.parametrize(
        "nat", [np.datetime64("NaT", "ns"), np.datetime64("NaT", "us")]
    )
    def test_add_nat_datetimelike_scalar(self, nat, tda):
        # [FUNCTION DOC]
        # Parametrized over numpy datetime64 NaT of ns and us units. Verifies adding it to
        # the array on either side gives an all-missing DatetimeArray keeping the array
        # resolution.
        result = tda + nat
        assert isinstance(result, DatetimeArray)
        assert result._creso == tda._creso
        assert result.isna().all()

        result = nat + tda
        assert isinstance(result, DatetimeArray)
        assert result._creso == tda._creso
        assert result.isna().all()

    def test_add_pdnat(self, tda):
        # [FUNCTION DOC]
        # Verifies adding pandas NaT on either side gives an all-missing TimedeltaArray
        # keeping the array resolution.
        result = tda + pd.NaT
        assert isinstance(result, TimedeltaArray)
        assert result._creso == tda._creso
        assert result.isna().all()

        result = pd.NaT + tda
        assert isinstance(result, TimedeltaArray)
        assert result._creso == tda._creso
        assert result.isna().all()

    # TODO: 2022-07-11 this is the only test that gets to DTA.tz_convert
    #  or tz_localize with non-nano; implement tests specific to that.
    def test_add_datetimelike_scalar(self, tda, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture. Verifies adding a (possibly tz-aware) nanosecond
        # timestamp on either side equals the nanosecond addition, including a case where
        # the timestamp cannot be losslessly cast to the array unit.
        # KNOWN ISSUE: the TODO above the test says this is the only test reaching
        # tz_convert / tz_localize with non-nano datetimes and that specific tests are
        # still needed.
        ts = pd.Timestamp("2016-01-01", tz=tz_naive_fixture).as_unit("ns")

        expected = tda.as_unit("ns") + ts
        res = tda + ts
        tm.assert_extension_array_equal(res, expected)
        res = ts + tda
        tm.assert_extension_array_equal(res, expected)

        ts += Timedelta(1)  # case where we can't cast losslessly

        exp_values = tda._ndarray + ts.asm8
        expected = (
            DatetimeArray._simple_new(exp_values, dtype=exp_values.dtype)
            .tz_localize("UTC")
            .tz_convert(ts.tz)
        )

        result = tda + ts
        tm.assert_extension_array_equal(result, expected)

        result = ts + tda
        tm.assert_extension_array_equal(result, expected)

    def test_mul_scalar(self, tda):
        # [FUNCTION DOC]
        # Verifies multiplication by an integer scalar multiplies the stored values and
        # keeps dtype and resolution.
        other = 2
        result = tda * other
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_mul_listlike(self, tda):
        # [FUNCTION DOC]
        # Verifies multiplication by an integer numpy array elementwise keeps dtype and
        # resolution.
        other = np.arange(len(tda))
        result = tda * other
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_mul_listlike_object(self, tda):
        # [FUNCTION DOC]
        # Verifies multiplication by an object-dtype integer numpy array elementwise
        # keeps dtype and resolution.
        other = np.arange(len(tda))
        result = tda * other.astype(object)
        expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_div_numeric_scalar(self, tda):
        # [FUNCTION DOC]
        # Verifies division by an integer scalar divides the stored values and keeps
        # dtype and resolution.
        other = 2
        result = tda / other
        expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_div_td_scalar(self, tda):
        # [FUNCTION DOC]
        # Verifies dividing by a Python timedelta of one second gives a float numpy array
        # equal to the raw values divided by one second.
        other = timedelta(seconds=1)
        result = tda / other
        expected = tda._ndarray / np.timedelta64(1, "s")
        tm.assert_numpy_array_equal(result, expected)

    def test_div_numeric_array(self, tda):
        # [FUNCTION DOC]
        # Verifies division by an integer numpy array elementwise keeps dtype and
        # resolution.
        other = np.arange(len(tda))
        result = tda / other
        expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
        tm.assert_extension_array_equal(result, expected)
        assert result._creso == tda._creso

    def test_div_td_array(self, tda):
        # [FUNCTION DOC]
        # Verifies dividing by a timedelta numpy array gives the float ratio array.
        other = tda._ndarray + tda._ndarray[-1]
        result = tda / other
        expected = tda._ndarray / other
        tm.assert_numpy_array_equal(result, expected)

    def test_add_timedeltaarraylike(self, tda):
        # [FUNCTION DOC]
        # Verifies adding and subtracting a same-values nanosecond array gives results
        # in the finer resolution: doubled values for addition on either side and zeros
        # for subtraction on either side.
        tda_nano = tda.astype("m8[ns]")

        expected = tda_nano * 2
        res = tda_nano + tda
        tm.assert_extension_array_equal(res, expected)
        res = tda + tda_nano
        tm.assert_extension_array_equal(res, expected)

        expected = tda_nano * 0
        res = tda - tda_nano
        tm.assert_extension_array_equal(res, expected)

        res = tda_nano - tda
        tm.assert_extension_array_equal(res, expected)


class TestTimedeltaArray:
    # [CLASS DOC]
    # General tests of TimedeltaArray.
    def test_astype_int(self, any_int_numpy_dtype):
        # [FUNCTION DOC]
        # Uses the any_int_numpy_dtype fixture. Verifies astype to any integer dtype other
        # than int64 raises a type error advising int64, and int64 gives the integer view of
        # the values.
        arr = TimedeltaArray._from_sequence(
            [Timedelta("1h"), Timedelta("2h")], dtype="m8[ns]"
        )

        if np.dtype(any_int_numpy_dtype) != np.int64:
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(any_int_numpy_dtype)
            return

        result = arr.astype(any_int_numpy_dtype)
        expected = arr._ndarray.view("i8")
        tm.assert_numpy_array_equal(result, expected)

    def test_setitem_clears_freq(self):
        # [FUNCTION DOC]
        # Verifies setting an element clears the array frequency.
        a = pd.timedelta_range("1h", periods=2, freq="h")._data
        a[0] = Timedelta("1h")
        assert a.freq is None

    @pytest.mark.parametrize(
        "obj",
        [
            Timedelta(seconds=1),
            Timedelta(seconds=1).to_timedelta64(),
            Timedelta(seconds=1).to_pytimedelta(),
        ],
    )
    def test_setitem_objects(self, obj):
        # [FUNCTION DOC]
        # Parametrized over a Timedelta, a numpy timedelta64 and a Python timedelta.
        # Verifies each can be assigned and reads back as the one second Timedelta.
        # make sure we accept timedelta64 and timedelta in addition to Timedelta
        tdi = pd.timedelta_range("2 Days", periods=4, freq="h")
        arr = tdi._data

        arr[0] = obj
        assert arr[0] == Timedelta(seconds=1)

    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.datetime64("NaT", "ns"),
            pd.Timestamp("2021-01-01"),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10**9,
            (np.arange(10) * 24 * 3600 * 10**9).view("datetime64[ns]"),
            pd.Timestamp("2021-01-01").to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        # [FUNCTION DOC]
        # Parametrized over many invalid values (numbers, datetime NaT, Timestamp, the
        # string "invalid", integer and datetime arrays, a period) and over index. Verifies
        # searchsorted raises a type error with one of two accepted messages.
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.TimedeltaIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timedelta', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)


class TestUnaryOps:
    # [CLASS DOC]
    # Tests of unary operators on TimedeltaArray.
    def test_abs(self):
        # [FUNCTION DOC]
        # Verifies absolute value (builtin and numpy) of an array with a negative value
        # and NaT gives the positive values with NaT kept.
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals, dtype=vals.dtype)

        evals = np.array([3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        expected = TimedeltaArray._from_sequence(evals, dtype=evals.dtype)

        result = abs(arr)
        tm.assert_timedelta_array_equal(result, expected)

        result2 = np.abs(arr)
        tm.assert_timedelta_array_equal(result2, expected)

    def test_pos(self):
        # [FUNCTION DOC]
        # Verifies unary plus (operator and numpy positive) returns an equal array that
        # does not share memory with the original.
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals, dtype=vals.dtype)

        result = +arr
        tm.assert_timedelta_array_equal(result, arr)
        assert not tm.shares_memory(result, arr)

        result2 = np.positive(arr)
        tm.assert_timedelta_array_equal(result2, arr)
        assert not tm.shares_memory(result2, arr)

    def test_neg(self):
        # [FUNCTION DOC]
        # Verifies negation (operator and numpy negative) flips signs and keeps NaT.
        vals = np.array([-3600 * 10**9, "NaT", 7200 * 10**9], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(vals, dtype=vals.dtype)

        evals = np.array([3600 * 10**9, "NaT", -7200 * 10**9], dtype="m8[ns]")
        expected = TimedeltaArray._from_sequence(evals)

        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

        result2 = np.negative(arr)
        tm.assert_timedelta_array_equal(result2, expected)

    def test_neg_freq(self):
        # [FUNCTION DOC]
        # Verifies negation of an array with a frequency equals the negated array.
        tdi = pd.timedelta_range("2 Days", periods=4, freq="h")
        arr = tdi._data

        expected = -tdi._data

        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

        result2 = np.negative(arr)
        tm.assert_timedelta_array_equal(result2, expected)
