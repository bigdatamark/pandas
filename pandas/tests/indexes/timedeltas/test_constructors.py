# [MODULE DOC]
# Tests for constructing a timedelta index (and the related range and conversion functions): invalid inputs, copy semantics, frequency inference and mismatch handling, float handling, mixed-type element parsing, ISO-8601 strings, unit and precision errors, categorical input, and deprecation of upper-case unit abbreviations.

from datetime import timedelta

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray


class TestTimedeltaIndex:
    # [CLASS DOC]
    # Test class for timedelta-index construction.
    def test_array_of_dt64_nat_raises(self):
        # [FUNCTION DOC]
        # An object array containing a datetime NaT passed to the index constructor, the array's private sequence constructor (with nanosecond dtype) or the timedelta conversion function raises a type error "Invalid type for timedelta scalar".
        # GH#39462
        nat = np.datetime64("NaT", "ns")
        arr = np.array([nat], dtype=object)

        msg = "Invalid type for timedelta scalar"
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(arr)

        with pytest.raises(TypeError, match=msg):
            TimedeltaArray._from_sequence(arr, dtype="m8[ns]")

        with pytest.raises(TypeError, match=msg):
            to_timedelta(arr)

    def test_int64_nocopy(self):
        # [FUNCTION DOC]
        # Constructing from an int64 array with copy disabled shares memory: the internal array's base is the passed array.
        # GH#23539 check that a copy isn't made when we pass int64 data
        #  and copy=False
        arr = np.arange(10, dtype=np.int64)
        tdi = TimedeltaIndex(arr, copy=False)
        assert tdi._data._ndarray.base is arr

    def test_infer_from_tdi(self):
        # [FUNCTION DOC]
        # For a very large (ten million) one-second range, constructing with frequency inference keeps the existing frequency without running the (expensive) inference: the inferred-frequency entry is absent from the object's cache.
        # GH#23539
        # fast-path for inferring a frequency if the passed data already
        #  has one
        tdi = timedelta_range("1 second", periods=10**7, freq="1s")

        result = TimedeltaIndex(tdi, freq="infer")
        assert result.freq == tdi.freq

        # check that inferred_freq was not called by checking that the
        #  value has not been cached
        assert "inferred_freq" not in getattr(result, "_cache", {})

    def test_infer_from_tdi_mismatch(self):
        # [FUNCTION DOC]
        # Constructing from an index or its array with a daily frequency that conflicts with the data's second frequency raises a value error saying the inferred frequency does not conform.
        # GH#23539
        # fast-path for invalidating a frequency if the passed data already
        #  has one and it does not match the `freq` input
        tdi = timedelta_range("1 second", periods=100, freq="1s")

        msg = (
            "Inferred frequency .* from passed values does "
            "not conform to passed frequency"
        )
        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(tdi, freq="D")

        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(tdi._data, freq="D")

    def test_dt64_data_invalid(self):
        # [FUNCTION DOC]
        # Passing a tz-aware datetime index, a naive one or a numpy datetime array raises a type error saying it cannot be converted to timedelta64.
        # GH#23539
        # passing tz-aware DatetimeIndex raises, naive or ndarray[datetime64]
        #  raise as of GH#29794
        dti = pd.date_range("2016-01-01", periods=3)

        msg = "cannot be converted to timedelta64"
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(dti.tz_localize("Europe/Brussels"))

        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(dti)

        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex(np.asarray(dti))

    def test_float64_ns_rounded(self):
        # [FUNCTION DOC]
        # Without a unit, floats are nanoseconds and fractions are truncated (2.3 and 9.7 become 2 and 9); integral floats are exact; NaN becomes NaT.
        # GH#23539 without specifying a unit, floats are regarded as nanos,
        #  and fractional portions are truncated
        tdi = TimedeltaIndex([2.3, 9.7])
        expected = TimedeltaIndex([2, 9])
        tm.assert_index_equal(tdi, expected)

        # integral floats are non-lossy
        tdi = TimedeltaIndex([2.0, 9.0])
        expected = TimedeltaIndex([2, 9])
        tm.assert_index_equal(tdi, expected)

        # NaNs get converted to NaT
        tdi = TimedeltaIndex([2.0, np.nan])
        expected = TimedeltaIndex([Timedelta(nanoseconds=2), pd.NaT])
        tm.assert_index_equal(tdi, expected)

    def test_float64_unit_conversion(self):
        # [FUNCTION DOC]
        # Converting floats 1.5 and 2.25 with unit day gives timedeltas of that many days (nanosecond resolution).
        # GH#23539
        tdi = to_timedelta([1.5, 2.25], unit="D")
        expected = TimedeltaIndex(
            [Timedelta(days=1.5), Timedelta(days=2.25)], dtype="m8[ns]"
        )
        tm.assert_index_equal(tdi, expected)

    def test_construction_base_constructor(self):
        # [FUNCTION DOC]
        # The generic Index constructor on a list or array of timedeltas with NaT (or NaN) infers the same result as the timedelta index constructor.
        arr = [Timedelta("1 days"), pd.NaT, Timedelta("3 days")]
        tm.assert_index_equal(pd.Index(arr), TimedeltaIndex(arr))
        tm.assert_index_equal(pd.Index(np.array(arr)), TimedeltaIndex(np.array(arr)))

        arr = [np.nan, pd.NaT, Timedelta("1 days")]
        tm.assert_index_equal(pd.Index(arr), TimedeltaIndex(arr))
        tm.assert_index_equal(pd.Index(np.array(arr)), TimedeltaIndex(np.array(arr)))

    def test_constructor(self):
        # [FUNCTION DOC]
        # A list mixing strings, a numpy timedelta, a standard timedelta and an offset (Second 3) parses into the expected timedelta index.
        expected = TimedeltaIndex(
            [
                "1 days",
                "1 days 00:00:05",
                "2 days",
                "2 days 00:00:02",
                "0 days 00:00:03",
            ]
        )
        result = TimedeltaIndex(
            [
                "1 days",
                "1 days, 00:00:05",
                np.timedelta64(2, "D"),
                timedelta(days=2, seconds=2),
                pd.offsets.Second(3),
            ]
        )
        tm.assert_index_equal(result, expected)

    def test_constructor_iso(self):
        # [FUNCTION DOC]
        # ISO-8601 duration strings of seconds convert to the same values as a one-second-frequency range.
        # GH #21877
        expected = timedelta_range("1s", periods=9, freq="s")
        durations = [f"P0DT0H0M{i}S" for i in range(1, 10)]
        result = to_timedelta(durations)
        tm.assert_index_equal(result, expected)

    def test_timedelta_range_fractional_period(self):
        # [FUNCTION DOC]
        # A non-integer periods argument to the range function raises a type error ("periods must be an integer").
        msg = "periods must be an integer"
        with pytest.raises(TypeError, match=msg):
            timedelta_range("1 days", periods=10.5)

    def test_constructor_coverage(self):
        # [FUNCTION DOC]
        # Misc constructor errors and forms: a string periods raises a type error; a scalar string where a collection is required raises a type error; a generator works; a numpy string array parses (to microsecond unit); integer nanosecond values rebuild an equal nanosecond index; data with a non-conforming frequency (days one, two, four with daily frequency) raises a value error; a range with only periods and freq raises a value error that exactly three of start, end, periods and freq must be specified.
        msg = "periods must be an integer, got foo"
        with pytest.raises(TypeError, match=msg):
            timedelta_range(start="1 days", periods="foo", freq="D")

        msg = (
            r"TimedeltaIndex\(\.\.\.\) must be called with a collection of some kind, "
            "'1 days' was passed"
        )
        with pytest.raises(TypeError, match=msg):
            TimedeltaIndex("1 days")

        # generator expression
        gen = (timedelta(i) for i in range(10))
        result = TimedeltaIndex(gen)
        expected = TimedeltaIndex([timedelta(i) for i in range(10)])
        tm.assert_index_equal(result, expected)

        # NumPy string array
        strings = np.array(["1 days", "2 days", "3 days"])
        result = TimedeltaIndex(strings)
        expected = to_timedelta([1, 2, 3], unit="D").as_unit("us")
        tm.assert_index_equal(result, expected)

        from_ints = TimedeltaIndex(expected.as_unit("ns").asi8)
        tm.assert_index_equal(from_ints, expected.as_unit("ns"))

        # non-conforming freq
        msg = (
            "Inferred frequency None from passed values does not conform to "
            "passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(["1 days", "2 days", "4 days"], freq="D")

        msg = (
            "Of the four parameters: start, end, periods, and freq, exactly "
            "three must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            timedelta_range(periods=10, freq="D")

    def test_constructor_name(self):
        # [FUNCTION DOC]
        # The constructor keeps the name given to a range and a new name overrides when building from an index.
        idx = timedelta_range(start="1 days", periods=1, freq="D", name="TEST")
        assert idx.name == "TEST"

        # GH10025
        idx2 = TimedeltaIndex(idx, name="something else")
        assert idx2.name == "something else"

    def test_constructor_no_precision_raises(self):
        # [FUNCTION DOC]
        # A unit-less timedelta64 dtype raises a value error ("with no precision is not allowed") in the timedelta constructor and a value error saying the dtype has no unit in the generic constructor.
        # GH-24753, GH-24739

        msg = "with no precision is not allowed"
        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(["2000"], dtype="timedelta64")

        msg = "The 'timedelta64' dtype has no unit. Please pass in"
        with pytest.raises(ValueError, match=msg):
            pd.Index(["2000"], dtype="timedelta64")

    def test_constructor_wrong_precision_raises(self):
        # [FUNCTION DOC]
        # A day-resolution dtype raises a value error listing supported resolutions; microsecond dtype is accepted and kept.
        msg = "Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            TimedeltaIndex(["2000"], dtype="timedelta64[D]")

        # "timedelta64[us]" was unsupported pre-2.0, but now this works.
        tdi = TimedeltaIndex(["2000"], dtype="timedelta64[us]")
        assert tdi.dtype == "m8[us]"

    def test_explicit_none_freq(self):
        # [FUNCTION DOC]
        # Passing freq as none explicitly, from an index or from its array, removes the frequency of the result.
        # Explicitly passing freq=None is respected
        tdi = timedelta_range(1, periods=5)
        assert tdi.freq is not None

        result = TimedeltaIndex(tdi, freq=None)
        assert result.freq is None

        result = TimedeltaIndex(tdi._data, freq=None)
        assert result.freq is None

    def test_from_categorical(self):
        # [FUNCTION DOC]
        # A categorical or categorical index of timedeltas converts to the equal timedelta index.
        tdi = timedelta_range(1, periods=5)

        cat = pd.Categorical(tdi)

        result = TimedeltaIndex(cat)
        tm.assert_index_equal(result, tdi)

        ci = pd.CategoricalIndex(tdi)
        result = TimedeltaIndex(ci)
        tm.assert_index_equal(result, tdi)

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
        # Parametrized over (unit, deprecated spelling) pairs (week, day, minute, second, hour, millisecond, microsecond). Constructing from strings with the deprecated spelling, or converting with that spelling as unit, emits the Pandas 4 warning "'...' is deprecated and will be removed in a future version" and gives the same values as the modern spelling (resolution seconds unless the unit is already second, millisecond or microsecond).
        # GH#52536, GH#59051
        msg = f"'{unit_depr}' is deprecated and will be removed in a future version."

        expected = TimedeltaIndex([f"1{unit}", f"2{unit}"])
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = TimedeltaIndex([f"1{unit_depr}", f"2{unit_depr}"])
        tm.assert_index_equal(result, expected)

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            tdi = to_timedelta([1, 2], unit=unit_depr)
        exp_unit = unit if unit in ["s", "ms", "us"] else "s"
        tm.assert_index_equal(tdi, expected.as_unit(exp_unit))
