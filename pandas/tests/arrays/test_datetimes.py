"""
Tests for DatetimeArray
"""
# [MODULE DOC]
# Tests for DatetimeArray: non-nanosecond resolution behavior (TestNonNano),
# comparisons against index and list-likes (TestDatetimeArrayComparisons),
# and a general class (TestDatetimeArray) covering astype, setitem, fill, numpy
# array protocol, searchsorted, shift, timezone handling, and frequency alias
# validation in date_range. Also one module-level test of factorize with sort.
# Shares behavior tests with the datetime-like shared tests module. No
# import-time side effects.

from __future__ import annotations

from datetime import timedelta
import operator

import numpy as np
import pytest

from pandas._libs.tslibs import tz_compare
from pandas.compat.numpy import (
    is_numpy_dev,
    np_version_gt2_5,
)
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)


class TestNonNano:
    # [CLASS DOC]
    # Tests of DatetimeArray holding non-nanosecond units (second, millisecond,
    # microsecond). Fixtures parametrize unit and timezone; most tests compare
    # the non-nano array with its nanosecond DatetimeIndex counterpart.
    @pytest.fixture(params=["s", "ms", "us"])
    def unit(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over the units second, millisecond and
        # microsecond. Returns the unit string.
        """Fixture returning parametrized time units"""
        return request.param

    @pytest.fixture
    def dtype(self, unit, tz_naive_fixture):
        # [FUNCTION DOC]
        # HELPER: fixture. Combines the unit and tz_naive_fixture fixtures: returns a
        # numpy datetime64 dtype of that unit for tz-naive data, otherwise a timezone-aware
        # datetime dtype of that unit and timezone.
        tz = tz_naive_fixture
        if tz is None:
            return np.dtype(f"datetime64[{unit}]")
        else:
            return DatetimeTZDtype(unit=unit, tz=tz)

    @pytest.fixture
    def dta_dti(self, unit, dtype):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns a pair of equal logical data: a DatetimeArray of 55
        # daily timestamps from 2016-01-01 in the fixture unit and dtype (built via the
        # private simple constructor from UTC wall values), and the nanosecond DatetimeIndex
        # of the same timestamps for comparison.
        tz = getattr(dtype, "tz", None)

        dti = pd.date_range("2016-01-01", periods=55, freq="D", tz=tz, unit="ns")
        if tz is None:
            arr = np.asarray(dti).astype(f"M8[{unit}]")
        else:
            arr = np.asarray(dti.tz_convert("UTC").tz_localize(None)).astype(
                f"M8[{unit}]"
            )

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        return dta, dti

    @pytest.fixture
    def dta(self, dta_dti):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns only the array from the dta_dti fixture.
        dta, dti = dta_dti
        return dta

    def test_non_nano(self, unit, dtype):
        # [FUNCTION DOC]
        # Verifies a non-nano array built from integer values has the expected dtype,
        # its elements retain the unit and timezone, and the first element equals the
        # slice of length one.
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dta = DatetimeArray._simple_new(arr, dtype=dtype)

        assert dta.dtype == dtype
        assert dta[0].unit == unit
        assert tz_compare(dta.tz, dta[0].tz)
        assert (dta[0] == dta[:1]).all()

    @pytest.mark.parametrize(
        "field", DatetimeArray._field_ops + DatetimeArray._bool_ops
    )
    def test_fields(self, unit, field, dtype, dta_dti):
        # [FUNCTION DOC]
        # Parametrized over all integer field and boolean field properties. Verifies the
        # non-nano array equals the index elementwise and its property values equal the
        # nanosecond index property values.
        dta, dti = dta_dti

        assert (dti == dta).all()

        res = getattr(dta, field)
        expected = getattr(dti._data, field)
        tm.assert_numpy_array_equal(res, expected)

    def test_normalize(self, unit):
        # [FUNCTION DOC]
        # Parametrized by unit. Verifies the array is not normalized when it holds
        # times of day and that normalize gives midnight values equal to the nanosecond
        # index normalized result converted to the unit.
        dti = pd.date_range("2016-01-01 06:00:00", periods=55, freq="D")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        assert not dta.is_normalized

        # TODO: simplify once we can just .astype to other unit
        exp = np.asarray(dti.normalize()).astype(f"M8[{unit}]")
        expected = DatetimeArray._simple_new(exp, dtype=exp.dtype)

        res = dta.normalize()
        tm.assert_extension_array_equal(res, expected)

    def test_normalize_overflow_raises(self):
        # [FUNCTION DOC]
        # Regression test (GH#60583). Verifies normalizing the minimum representable
        # timestamp raises a value error ("Cannot normalize Timestamp without integer
        # overflow").
        # GH#60583
        ts = pd.Timestamp.min
        dta = DatetimeArray._from_sequence([ts], dtype="M8[ns]")

        msg = "Cannot normalize Timestamp without integer overflow"
        with pytest.raises(ValueError, match=msg):
            dta.normalize()

    def test_simple_new_requires_match(self, unit):
        # [FUNCTION DOC]
        # Verifies the private simple constructor accepts a timezone dtype whose unit
        # matches the data, and raises an assertion error (empty message) for a dtype of
        # a different unit.
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dtype = DatetimeTZDtype(unit, "UTC")

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        assert dta.dtype == dtype

        wrong = DatetimeTZDtype("ns", "UTC")
        with pytest.raises(AssertionError, match="^$"):
            DatetimeArray._simple_new(arr, dtype=wrong)

    def test_std_non_nano(self, unit):
        # [FUNCTION DOC]
        # Verifies the standard deviation of a non-nano array has the same resolution as
        # the array and equals the nanosecond standard deviation floored to the unit.
        dti = pd.date_range("2016-01-01", periods=55, freq="D", unit="ns")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        # we should match the nano-reso std, but floored to our reso.
        res = dta.std()
        assert res._creso == dta._creso
        assert res == dti.std().floor(unit)

    @pytest.mark.filterwarnings("ignore:Converting to PeriodArray.*:UserWarning")
    def test_to_period(self, dta_dti):
        # [FUNCTION DOC]
        # Verifies conversion to daily periods equals that of the nanosecond array. A
        # warning about converting to a period is ignored.
        dta, dti = dta_dti
        result = dta.to_period("D")
        expected = dti._data.to_period("D")

        tm.assert_extension_array_equal(result, expected)

    def test_iter(self, dta):
        # [FUNCTION DOC]
        # Verifies iteration yields Timestamp objects (exactly that class) that keep the
        # integer value, resolution and equality of the first element.
        res = next(iter(dta))
        expected = dta[0]

        assert type(res) is pd.Timestamp
        assert res._value == expected._value
        assert res._creso == expected._creso
        assert res == expected

    def test_astype_object(self, dta):
        # [FUNCTION DOC]
        # Verifies casting to object gives Timestamps keeping the array resolution and
        # equal to the array elements.
        result = dta.astype(object)
        assert all(x._creso == dta._creso for x in result)
        assert all(x == y for x, y in zip(result, dta, strict=True))

    def test_to_pydatetime(self, dta_dti):
        # [FUNCTION DOC]
        # Verifies conversion to Python datetimes equals the index result.
        dta, dti = dta_dti

        result = dta.to_pydatetime()
        expected = dti.to_pydatetime()
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("meth", ["time", "timetz", "date"])
    def test_time_date(self, dta_dti, meth):
        # [FUNCTION DOC]
        # Parametrized over time, timetz and date. Verifies the array property
        # equals the index property.
        dta, dti = dta_dti

        result = getattr(dta, meth)
        expected = getattr(dti, meth)
        tm.assert_numpy_array_equal(result, expected)

    def test_format_native_types(self, unit, dtype, dta_dti):
        # [FUNCTION DOC]
        # Verifies the private native-type formatting gives the same strings as the
        # nanosecond index data. The unit and dtype parameters are unused.
        # In this case we should get the same formatted values with our nano
        #  version dti._data as we do with the non-nano dta
        dta, dti = dta_dti

        res = dta._format_native_types()
        exp = dti._data._format_native_types()
        tm.assert_numpy_array_equal(res, exp)

    def test_repr(self, dta_dti, unit):
        # [FUNCTION DOC]
        # Verifies the repr equals the nanosecond index repr with the unit text replaced.
        dta, dti = dta_dti

        assert repr(dta) == repr(dti._data).replace("[ns", f"[{unit}")

    # TODO: tests with td64
    def test_compare_mismatched_resolutions(self, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison_op fixture. With values near the int64 limits where numpy
        # overflows silently, verifies comparisons between a nanosecond array and a
        # second-unit array (array to array and element to array) give the correct
        # results for each operator. For order comparisons also verifies that the raw
        # numpy comparison raises an overflow error on new numpy versions, or still gets
        # the wrong answer on older ones.
        # KNOWN ISSUE: the TODO above the test says tests with timedelta are missing.
        # comparison that numpy gets wrong bc of silent overflows
        op = comparison_op

        iinfo = np.iinfo(np.int64)
        vals = np.array([iinfo.min, iinfo.min + 1, iinfo.max], dtype=np.int64)

        # Construct so that arr2[1] < arr[1] < arr[2] < arr2[2]
        arr = np.array(vals).view("M8[ns]")
        arr2 = arr.view("M8[s]")

        left = DatetimeArray._simple_new(arr, dtype=arr.dtype)
        right = DatetimeArray._simple_new(arr2, dtype=arr2.dtype)

        if comparison_op is operator.eq:
            expected = np.array([False, False, False])
        elif comparison_op is operator.ne:
            expected = np.array([True, True, True])
        elif comparison_op in [operator.lt, operator.le]:
            expected = np.array([False, False, True])
        else:
            expected = np.array([False, True, False])

        result = op(left, right)
        tm.assert_numpy_array_equal(result, expected)

        result = op(left[1], right)
        tm.assert_numpy_array_equal(result, expected)

        if op not in [operator.eq, operator.ne]:
            if is_numpy_dev or np_version_gt2_5:
                # numpy now raises instead of silently overflowing
                # https://github.com/numpy/numpy/pull/31085
                with pytest.raises(OverflowError, match="Overflow"):
                    op(left._ndarray, right._ndarray)
            else:
                # check that numpy still gets this wrong; if it is fixed we may
                #  be able to remove compare_mismatched_resolutions
                np_res = op(left._ndarray, right._ndarray)
                tm.assert_numpy_array_equal(np_res[1:], ~expected[1:])

    def test_add_mismatched_reso_doesnt_downcast(self):
        # [FUNCTION DOC]
        # Verifies adding a microsecond timedelta to a microsecond array keeps unit
        # microsecond even if the result values are whole days (no downcasting).
        # https://github.com/pandas-dev/pandas/pull/48748#issuecomment-1260181008
        td = pd.Timedelta(microseconds=1)
        dti = pd.date_range("2016-01-01", periods=3) - td
        dta = dti._data.as_unit("us")

        res = dta + td.as_unit("us")
        # even though the result is an even number of days
        #  (so we _could_ downcast to unit="s"), we do not.
        assert res.unit == "us"

    @pytest.mark.parametrize(
        "scalar",
        [
            timedelta(hours=2),
            pd.Timedelta(hours=2),
            np.timedelta64(2, "h"),
            np.timedelta64(2 * 3600 * 1000, "ms"),
            pd.offsets.Minute(120),
            pd.offsets.Hour(2),
        ],
    )
    def test_add_timedeltalike_scalar_mismatched_reso(self, dta_dti, scalar):
        # [FUNCTION DOC]
        # Parametrized over several equal two-hour scalars (timedelta, Timedelta, numpy
        # timedelta64 in hours and ms, minute and hour offsets). Verifies adding on either
        # side and subtracting give results equal to the nanosecond index operations,
        # converted to the finer of the two units.
        dta, dti = dta_dti

        td = pd.Timedelta(scalar)
        exp_unit = tm.get_finest_unit(dta.unit, td.unit)

        expected = (dti + td)._data.as_unit(exp_unit)
        result = dta + scalar
        tm.assert_extension_array_equal(result, expected)

        result = scalar + dta
        tm.assert_extension_array_equal(result, expected)

        expected = (dti - td)._data.as_unit(exp_unit)
        result = dta - scalar
        tm.assert_extension_array_equal(result, expected)

    def test_sub_datetimelike_scalar_mismatch(self):
        # [FUNCTION DOC]
        # Verifies subtracting a second-unit timestamp from a microsecond array gives a
        # timedelta array of microsecond unit with the expected values.
        dti = pd.date_range("2016-01-01", periods=3)
        dta = dti._data.as_unit("us")

        ts = dta[0].as_unit("s")

        result = dta - ts
        expected = (dti - dti[0])._data.as_unit("us")
        assert result.dtype == "m8[us]"
        tm.assert_extension_array_equal(result, expected)

    def test_sub_datetime64_reso_mismatch(self):
        # [FUNCTION DOC]
        # Verifies subtracting datetime arrays of different resolutions (seconds minus
        # milliseconds, and the reverse) gives zero timedeltas in the finer unit.
        dti = pd.date_range("2016-01-01", periods=3)
        left = dti._data.as_unit("s")
        right = left.as_unit("ms")

        result = left - right
        exp_values = np.array([0, 0, 0], dtype="m8[ms]")
        expected = TimedeltaArray._simple_new(
            exp_values,
            dtype=exp_values.dtype,
        )
        tm.assert_extension_array_equal(result, expected)
        result2 = right - left
        tm.assert_extension_array_equal(result2, expected)


class TestDatetimeArrayComparisons:
    # [CLASS DOC]
    # Tests of comparison operators on DatetimeArray. A TODO says it should be
    # merged into the datetime64 arithmetic tests when those are robust enough.
    # TODO: merge this into tests/arithmetic/test_datetime64 once it is
    #  sufficiently robust

    def test_cmp_dt64_arraylike_tznaive(self, comparison_op):
        # [FUNCTION DOC]
        # Uses the comparison_op fixture. For a tz-naive array of nine month starts,
        # verifies the array has the same frequency and timezone as the index, and that
        # comparing it to itself and to the index, numpy array, list, tuple and object
        # Index (on either side) gives all true for equality-like operators, and all false for
        # not-equal, greater and less.
        # arbitrary tz-naive DatetimeIndex
        op = comparison_op

        dti = pd.date_range("2016-01-1", freq="MS", periods=9, tz=None)
        arr = dti._data
        assert arr.freq == dti.freq
        assert arr.tz == dti.tz

        right = dti

        expected = np.ones(len(arr), dtype=bool)
        if comparison_op.__name__ in ["ne", "gt", "lt"]:
            # for these the comparisons should be all-False
            expected = ~expected

        result = op(arr, arr)
        tm.assert_numpy_array_equal(result, expected)
        for other in [
            right,
            np.array(right),
            list(right),
            tuple(right),
            right.astype(object),
        ]:
            result = op(arr, other)
            tm.assert_numpy_array_equal(result, expected)

            result = op(other, arr)
            tm.assert_numpy_array_equal(result, expected)


class TestDatetimeArray:
    # [CLASS DOC]
    # General tests of DatetimeArray (as opposed to non-nano specific tests).
    def test_astype_ns_to_ms_near_bounds(self):
        # [FUNCTION DOC]
        # Regression test (GH#55979). Verifies casting a nanosecond array holding a
        # timestamp near the lower bound to millisecond gives the correct truncated value,
        # matching construction directly with the millisecond unit.
        # GH#55979
        ts = pd.Timestamp("1677-09-21 00:12:43.145225")
        target = ts.as_unit("ms")

        dta = DatetimeArray._from_sequence([ts], dtype="M8[ns]")
        assert (dta.view("i8") == ts.as_unit("ns").value).all()

        result = dta.astype("M8[ms]")
        assert result[0] == target

        expected = DatetimeArray._from_sequence([ts], dtype="M8[ms]")
        assert (expected.view("i8") == target._value).all()

        tm.assert_datetime_array_equal(result, expected)

    def test_astype_non_nano_tznaive(self):
        # [FUNCTION DOC]
        # Verifies astype to a non-nano naive dtype on an index and an array gives the
        # requested dtype and the array result is a DatetimeArray.
        dti = pd.date_range("2016-01-01", periods=3)

        res = dti.astype("M8[s]")
        assert res.dtype == "M8[s]"

        dta = dti._data
        res = dta.astype("M8[s]")
        assert res.dtype == "M8[s]"
        assert isinstance(res, pd.core.arrays.DatetimeArray)  # used to be ndarray

    def test_astype_non_nano_tzaware(self):
        # [FUNCTION DOC]
        # Verifies astype to a non-nano timezone-aware dtype on index and array, then
        # from non-nano to non-nano: a default cast copies (no shared memory) and copy false
        # shares memory.
        # KNOWN ISSUE: after the copy-false cast the assertion on the dtype is made on the
        # earlier result rather than the new one.
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")

        res = dti.astype("M8[s, US/Pacific]")
        assert res.dtype == "M8[s, US/Pacific]"

        dta = dti._data
        res = dta.astype("M8[s, US/Pacific]")
        assert res.dtype == "M8[s, US/Pacific]"

        # from non-nano to non-nano, preserving reso
        res2 = res.astype("M8[s, UTC]")
        assert res2.dtype == "M8[s, UTC]"
        assert not tm.shares_memory(res2, res)

        res3 = res.astype("M8[s, UTC]", copy=False)
        assert res2.dtype == "M8[s, UTC]"
        assert tm.shares_memory(res3, res)

    def test_astype_to_same(self):
        # [FUNCTION DOC]
        # Verifies casting to an equal timezone dtype with copy false returns the same object.
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        result = arr.astype(DatetimeTZDtype(tz="US/Central"), copy=False)
        assert result is arr

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[ns, UTC]"])
    @pytest.mark.parametrize(
        "other", ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, CET]"]
    )
    def test_astype_copies(self, dtype, other):
        # [FUNCTION DOC]
        # Parametrized over source and target datetime64 nanosecond dtypes (naive, UTC,
        # CET). Regression test for pull request 32490. Casting between naive and aware
        # raises a type error (message suggests tz_localize, or says aware to naive); other
        # casts return a copy so changing the result leaves the original Series unchanged.
        # https://github.com/pandas-dev/pandas/pull/32490
        ser = pd.Series([1, 2], dtype=dtype)
        orig = ser.copy()

        err = False
        if (dtype == "datetime64[ns]") ^ (other == "datetime64[ns]"):
            # deprecated in favor of tz_localize
            err = True

        if err:
            if dtype == "datetime64[ns]":
                msg = "Use obj.tz_localize instead or series.dt.tz_localize instead"
            else:
                msg = "from timezone-aware dtype to timezone-naive dtype"
            with pytest.raises(TypeError, match=msg):
                ser.astype(other)
        else:
            t = ser.astype(other)
            t[:] = pd.NaT
            tm.assert_series_equal(ser, orig)

    @pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
    def test_astype_int(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over integer dtypes. Verifies casting to anything other than int64
        # raises a type error advising to cast to int64, and casting to int64 gives
        # the integer view of the values.
        arr = DatetimeArray._from_sequence(
            [pd.Timestamp("2000"), pd.Timestamp("2001")], dtype="M8[ns]"
        )

        if np.dtype(dtype) != np.int64:
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(dtype)
            return

        result = arr.astype(dtype)
        expected = arr._ndarray.view("i8")
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_to_sparse_dt64(self):
        # [FUNCTION DOC]
        # Regression test (GH#50082). Verifies casting to a sparse datetime dtype
        # yields that dtype and equal values.
        # GH#50082
        dti = pd.date_range("2016-01-01", periods=4)
        dta = dti._data
        result = dta.astype("Sparse[datetime64[ns]]")

        assert result.dtype == "Sparse[datetime64[ns]]"
        assert (result == dta).all()

    def test_tz_setter_raises(self):
        # [FUNCTION DOC]
        # Verifies assigning the tz attribute raises an attribute error mentioning tz_localize.
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(AttributeError, match="tz_localize"):
            arr.tz = "UTC"

    def test_setitem_str_impute_tz(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture. Verifies setting a tz-naive string (scalar and list-like)
        # on a possibly timezone-aware array interprets the string in the array timezone,
        # equal to assigning the localized timestamp.
        # Like for getitem, if we are passed a naive-like string, we impute
        #  our own timezone.
        tz = tz_naive_fixture

        data = np.array([1, 2, 3], dtype="M8[ns]")
        dtype = data.dtype if tz is None else DatetimeTZDtype(tz=tz)
        arr = DatetimeArray._from_sequence(data, dtype=dtype)
        expected = arr.copy()

        ts = pd.Timestamp("2020-09-08 16:50").tz_localize(tz)
        setter = str(ts.tz_localize(None))

        # Setting a scalar tznaive string
        expected[0] = ts
        arr[0] = setter
        tm.assert_equal(arr, expected)

        # Setting a listlike of tznaive strings
        expected[1] = ts
        arr[:2] = [setter, setter]
        tm.assert_equal(arr, expected)

    def test_setitem_different_tz_raises(self):
        # [FUNCTION DOC]
        # Verifies setting a naive timestamp on a tz-aware array raises a type error
        # (cannot compare tz-naive and tz-aware), while setting a timestamp in another
        # timezone is allowed and stored converted to the array timezone.
        # pre-2.0 we required exact tz match, in 2.0 we require only
        #  tzawareness-match
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray._from_sequence(
            data, copy=False, dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="Cannot compare tz-naive and tz-aware"):
            arr[0] = pd.Timestamp("2000")

        ts = pd.Timestamp("2000", tz="US/Eastern")
        arr[0] = ts
        assert arr[0] == ts.tz_convert("US/Central")

    def test_setitem_clears_freq(self):
        # [FUNCTION DOC]
        # Verifies setting an element clears the array frequency.
        a = pd.date_range("2000", periods=2, freq="D", tz="US/Central")._data
        a[0] = pd.Timestamp("2000", tz="US/Central")
        assert a.freq is None

    @pytest.mark.parametrize(
        "obj",
        [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01").to_datetime64(),
            pd.Timestamp("2021-01-01").to_pydatetime(),
        ],
    )
    def test_setitem_objects(self, obj):
        # [FUNCTION DOC]
        # Parametrized over a Timestamp, a numpy datetime64 and a Python datetime.
        # Verifies each can be assigned and compares equal when read.
        # make sure we accept datetime64 and datetime in addition to Timestamp
        dti = pd.date_range("2000", periods=2, freq="D")
        arr = dti._data

        arr[0] = obj
        assert arr[0] == obj

    def test_repeat_preserves_tz(self):
        # [FUNCTION DOC]
        # Verifies repeat keeps timezone and values but drops frequency.
        dti = pd.date_range("2000", periods=2, freq="D", tz="US/Central")
        arr = dti._data

        repeated = arr.repeat([1, 1])

        # preserves tz and values, but not freq
        expected = DatetimeArray._from_sequence(arr.asi8, dtype=arr.dtype)
        tm.assert_equal(repeated, expected)

    def test_value_counts_preserves_tz(self):
        # [FUNCTION DOC]
        # Verifies value_counts keeps the timezone in its index, and with a NaT included
        # and dropna false gives counts for each value and for NaT.
        dti = pd.date_range("2000", periods=2, freq="D", tz="US/Central")
        arr = dti._data.repeat([4, 3])

        result = arr.value_counts()

        # Note: not tm.assert_index_equal, since `freq`s do not match
        assert result.index.equals(dti)

        arr[-2] = pd.NaT
        result = arr.value_counts(dropna=False)
        expected = pd.Series([4, 2, 1], index=[dti[0], dti[1], pd.NaT], name="count")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_preserves_tz(self, method):
        # [FUNCTION DOC]
        # Parametrized over pad and backfill. Verifies the private pad-or-backfill
        # fills the missing middle value from the previous or next neighbor keeping the
        # timezone, and does not modify the array or the index.
        dti = pd.date_range(
            "2000-01-01", periods=5, freq="D", tz="US/Central", unit="ns"
        )
        arr = DatetimeArray._from_sequence(dti, dtype=dti.dtype, copy=True)
        arr[2] = pd.NaT

        fill_val = dti[1] if method == "pad" else dti[3]
        expected = DatetimeArray._from_sequence(
            [dti[0], dti[1], fill_val, dti[3], dti[4]],
            dtype=DatetimeTZDtype(tz="US/Central"),
        )

        result = arr._pad_or_backfill(method=method)
        tm.assert_extension_array_equal(result, expected)

        # assert that arr and dti were not modified in-place
        assert arr[2] is pd.NaT
        assert dti[2] == pd.Timestamp("2000-01-03", tz="US/Central")

    def test_fillna_2d(self):
        # [FUNCTION DOC]
        # Verifies pad-or-backfill on a 2-d tz-aware array in both directions fills along
        # the first axis, behaves identically for Fortran-ordered backing data, and that
        # DataFrame ffill and bfill give matching results.
        # KNOWN ISSUE: the expected array for backfill is created by copying twice in
        # immediate succession; the duplicate copy is redundant.
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        dta = dti._data.reshape(3, 2).copy()
        dta[0, 1] = pd.NaT
        dta[1, 0] = pd.NaT

        res1 = dta._pad_or_backfill(method="pad")
        expected1 = dta.copy()
        expected1[1, 0] = dta[0, 0]
        tm.assert_extension_array_equal(res1, expected1)

        res2 = dta._pad_or_backfill(method="backfill")
        expected2 = dta.copy()
        expected2 = dta.copy()
        expected2[1, 0] = dta[2, 0]
        expected2[0, 1] = dta[1, 1]
        tm.assert_extension_array_equal(res2, expected2)

        # with different ordering for underlying ndarray; behavior should
        #  be unchanged
        dta2 = dta._from_backing_data(dta._ndarray.copy(order="F"))
        assert dta2._ndarray.flags["F_CONTIGUOUS"]
        assert not dta2._ndarray.flags["C_CONTIGUOUS"]
        tm.assert_extension_array_equal(dta, dta2)

        res3 = dta2._pad_or_backfill(method="pad")
        tm.assert_extension_array_equal(res3, expected1)

        res4 = dta2._pad_or_backfill(method="backfill")
        tm.assert_extension_array_equal(res4, expected2)

        # test the DataFrame method while we're here
        df = pd.DataFrame(dta)
        res = df.ffill()
        expected = pd.DataFrame(expected1)
        tm.assert_frame_equal(res, expected)

        res = df.bfill()
        expected = pd.DataFrame(expected2)
        tm.assert_frame_equal(res, expected)

    def test_array_interface_tz(self):
        # [FUNCTION DOC]
        # Verifies the numpy array protocol on a tz-aware array: default and object dtype
        # give Timestamp objects with the timezone; datetime64 nanosecond dtype gives UTC
        # naive values.
        tz = "US/Central"
        data = pd.date_range("2017", periods=2, tz=tz, unit="ns")._data
        result = np.asarray(data)

        expected = np.array(
            [
                pd.Timestamp("2017-01-01T00:00:00", tz=tz),
                pd.Timestamp("2017-01-02T00:00:00", tz=tz),
            ],
            dtype=object,
        )
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(data, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(data, dtype="M8[ns]")

        expected = np.array(
            ["2017-01-01T06:00:00", "2017-01-02T06:00:00"], dtype="M8[ns]"
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self):
        # [FUNCTION DOC]
        # Verifies the numpy array protocol for a tz-naive array: default gives datetime64
        # nanosecond values and object dtype gives Timestamp objects.
        data = pd.date_range("2017", periods=2, unit="ns")._data
        expected = np.array(
            ["2017-01-01T00:00:00", "2017-01-02T00:00:00"], dtype="datetime64[ns]"
        )

        result = np.asarray(data)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(data, dtype=object)
        expected = np.array(
            [pd.Timestamp("2017-01-01T00:00:00"), pd.Timestamp("2017-01-02T00:00:00")],
            dtype=object,
        )
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_different_tz(self, index):
        # [FUNCTION DOC]
        # Parametrized over index (true uses an Index wrapper). Verifies searchsorted of a
        # scalar and an array in a different timezone gives the same positions as the same
        # instants in the array timezone.
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.DatetimeIndex(data, freq="D")._data.tz_localize("Asia/Tokyo")
        if index:
            arr = pd.Index(arr)

        expected = arr.searchsorted(arr[2])
        result = arr.searchsorted(arr[2].tz_convert("UTC"))
        assert result == expected

        expected = arr.searchsorted(arr[2:6])
        result = arr.searchsorted(arr[2:6].tz_convert("UTC"))
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_tzawareness_compat(self, index):
        # [FUNCTION DOC]
        # Parametrized over index. Verifies searchsorted with a tz-aware scalar/array on
        # naive data and the reverse raises a type error (cannot compare tz-naive and
        # tz-aware).
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.DatetimeIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        mismatch = arr.tz_localize("Asia/Tokyo")

        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(mismatch[0])
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(mismatch)

        with pytest.raises(TypeError, match=msg):
            mismatch.searchsorted(arr[0])
        with pytest.raises(TypeError, match=msg):
            mismatch.searchsorted(arr)

    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.timedelta64("NaT", "ns"),
            pd.Timedelta(days=2),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10**9,
            np.arange(10, dtype="i8").view("timedelta64[ns]") * 24 * 3600 * 10**9,
            pd.Timestamp("2021-01-01").to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        # [FUNCTION DOC]
        # Parametrized over many invalid search values (numbers, timedelta objects,
        # the string "invalid", integer and timedelta arrays, a period) and index.
        # Verifies searchsorted raises a type error with one of two accepted messages.
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        arr = pd.DatetimeIndex(data, freq="D")._data
        if index:
            arr = pd.Index(arr)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timestamp', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)

    def test_shift_fill_value(self):
        # [FUNCTION DOC]
        # Verifies shift by one with a fill value (Timestamp, Python datetime, numpy
        # datetime64) rolls the data and fills with the last value; the same for
        # tz-aware data (without the numpy datetime64 case).
        dti = pd.date_range("2016-01-01", periods=3)

        dta = dti._data
        expected = DatetimeArray._from_sequence(
            np.roll(dta._ndarray, 1), dtype=dti.dtype
        )

        fv = dta[-1]
        for fill_value in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            result = dta.shift(1, fill_value=fill_value)
            tm.assert_datetime_array_equal(result, expected)

        dta = dta.tz_localize("UTC")
        expected = expected.tz_localize("UTC")
        fv = dta[-1]
        for fill_value in [fv, fv.to_pydatetime()]:
            result = dta.shift(1, fill_value=fill_value)
            tm.assert_datetime_array_equal(result, expected)

    def test_shift_value_tzawareness_mismatch(self):
        # [FUNCTION DOC]
        # Verifies shifting with a fill value whose timezone awareness does not match
        # the array raises a type error ("Cannot compare"), in both directions and for
        # timestamp, datetime and datetime64 inputs.
        dti = pd.date_range("2016-01-01", periods=3)

        dta = dti._data

        fv = dta[-1].tz_localize("UTC")
        for invalid in [fv, fv.to_pydatetime()]:
            with pytest.raises(TypeError, match="Cannot compare"):
                dta.shift(1, fill_value=invalid)

        dta = dta.tz_localize("UTC")
        fv = dta[-1].tz_localize(None)
        for invalid in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
            with pytest.raises(TypeError, match="Cannot compare"):
                dta.shift(1, fill_value=invalid)

    def test_shift_requires_tzmatch(self):
        # [FUNCTION DOC]
        # Verifies a fill value in a different timezone is accepted on shift and converted
        # (only tz-awareness must match since version 2.0).
        # pre-2.0 we required exact tz match, in 2.0 we require just
        #  matching tzawareness
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")
        dta = dti._data

        fill_value = pd.Timestamp("2020-10-18 18:44", tz="US/Pacific")

        result = dta.shift(1, fill_value=fill_value)
        expected = dta.shift(1, fill_value=fill_value.tz_convert("UTC"))
        tm.assert_equal(result, expected)

    def test_tz_localize_t2d(self):
        # [FUNCTION DOC]
        # Verifies tz_localize to naive on a 2-d array equals the flattened result
        # reshaped, and localizing back round trips to the original.
        dti = pd.date_range("1994-05-12", periods=12, tz="US/Pacific")
        dta = dti._data.reshape(3, 4)
        result = dta.tz_localize(None)

        expected = dta.ravel().tz_localize(None).reshape(dta.shape)
        tm.assert_datetime_array_equal(result, expected)

        roundtrip = expected.tz_localize("US/Pacific")
        tm.assert_datetime_array_equal(roundtrip, dta)

    @pytest.mark.parametrize(
        "tz", ["US/Eastern", "dateutil/US/Eastern", "pytz/US/Eastern"]
    )
    def test_iter_zoneinfo_fold(self, tz):
        # [FUNCTION DOC]
        # Regression test (GH#49684). Parametrized over timezone specs (zoneinfo,
        # dateutil, pytz; pytz skipped if not installed). With times around a daylight
        # saving fold, verifies a scalar taken from the array, an element from iterating
        # the array, and an element from casting to object have equal text and UTC offsets.
        # GH#49684
        if tz.startswith("pytz/"):
            pytz = pytest.importorskip("pytz")
            tz = pytz.timezone(tz.removeprefix("pytz/"))
        utc_vals = np.array(
            [1320552000, 1320555600, 1320559200, 1320562800], dtype=np.int64
        )
        utc_vals *= 1_000_000_000

        dta = (
            DatetimeArray._from_sequence(utc_vals, dtype=np.dtype("M8[ns]"))
            .tz_localize("UTC")
            .tz_convert(tz)
        )

        left = dta[2]
        right = list(dta)[2]
        assert str(left) == str(right)
        # previously there was a bug where with non-pytz right would be
        #  Timestamp('2011-11-06 01:00:00-0400', tz='US/Eastern')
        # while left would be
        #  Timestamp('2011-11-06 01:00:00-0500', tz='US/Eastern')
        # The .value's would match (so they would compare as equal),
        #  but the folds would not
        assert left.utcoffset() == right.utcoffset()

        # The same bug in ints_to_pydatetime affected .astype, so we test
        #  that here.
        right2 = dta.astype(object)[2]
        assert str(left) == str(right2)
        assert left.utcoffset() == right2.utcoffset()

    @pytest.mark.parametrize(
        "freq",
        ["2M", "2SM", "2sm", "2Q", "2Q-SEP", "1Y", "2Y-MAR", "2m", "2q-sep", "2y"],
    )
    def test_date_range_frequency_M_Q_Y_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over month, semi-month, quarter and year aliases in invalid
        # spellings. Verifies date_range raises a value error "Invalid frequency".
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)

    @pytest.mark.parametrize("freq_depr", ["2MIN", "2nS", "2Us"])
    def test_date_range_uppercase_frequency_deprecated(self, freq_depr):
        # [FUNCTION DOC]
        # Regression test (GH#9586, GH#54939). Parametrized over uppercase minute,
        # nano and micro unit aliases. Verifies date_range with the uppercase alias emits
        # the Pandas4Warning deprecation (suggesting lowercase) and equals the result of the
        # lowercase alias.
        # GH#9586, GH#54939
        depr_msg = (
            f"'{freq_depr[1:]}' is deprecated and will be removed in a "
            f"future version, please use '{freq_depr.lower()[1:]}' instead."
        )

        expected = pd.date_range("1/1/2000", periods=4, freq=freq_depr.lower())
        with tm.assert_produces_warning(Pandas4Warning, match=depr_msg):
            result = pd.date_range("1/1/2000", periods=4, freq=freq_depr)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq",
        [
            "2ye-mar",
            "2ys",
            "2qe",
            "2qs-feb",
            "2bqs",
            "2sms",
            "2bms",
            "2cbme",
            "2me",
        ],
    )
    def test_date_range_lowercase_frequency_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over lowercase anchored aliases. Verifies date_range raises a value
        # error "Invalid frequency".
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)

    def test_date_range_lowercase_frequency_deprecated(self):
        # [FUNCTION DOC]
        # Regression test (GH#9586, GH#54939). Verifies the lowercase week alias emits a
        # Pandas4Warning deprecation and equals the result of the uppercase alias.
        # GH#9586, GH#54939
        depr_msg = "'w' is deprecated and will be removed in a future version"

        expected = pd.date_range("1/1/2000", periods=4, freq="2W")
        with tm.assert_produces_warning(Pandas4Warning, match=depr_msg):
            result = pd.date_range("1/1/2000", periods=4, freq="2w")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq", ["1A", "2A-MAR", "2a-mar"])
    def test_date_range_frequency_A_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over legacy year aliases. Verifies date_range raises a value error
        # "Invalid frequency".
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)

    @pytest.mark.parametrize("freq", ["2H", "2CBH", "2S"])
    def test_date_range_uppercase_frequency_raises(self, freq):
        # [FUNCTION DOC]
        # Parametrized over removed uppercase aliases for hour, business hour and second.
        # Verifies date_range raises a value error "Invalid frequency".
        msg = f"Invalid frequency: {freq}"

        with pytest.raises(ValueError, match=msg):
            pd.date_range("1/1/2000", periods=4, freq=freq)


def test_factorize_sort_without_freq():
    # [FUNCTION DOC]
    # Verifies factorize with sorting on a datetime array without frequency raises a
    # not-implemented error telling users to call the top-level factorize function, and
    # the same for a TimedeltaArray derived from it.
    dta = DatetimeArray._from_sequence([0, 2, 1], dtype="M8[ns]")

    msg = r"call pd.factorize\(obj, sort=True\) instead"
    with pytest.raises(NotImplementedError, match=msg):
        dta.factorize(sort=True)

    # Do TimedeltaArray while we're here
    tda = dta - dta[0]
    with pytest.raises(NotImplementedError, match=msg):
        tda.factorize(sort=True)
