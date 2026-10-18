# [MODULE DOC]
# Tests for the cast method (astype) of the timedelta index: to object (with and without missing values), integer, string, unsigned integer (rejected), other timedelta units, unsupported day unit, category, float/datetime (rejected), and boolean fallback. Several tests also run on series via the index_or_series fixture.

from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltaIndex:
    # [CLASS DOC]
    # Test class for timedelta-index casting.
    def test_astype_object(self):
        # [FUNCTION DOC]
        # Casting a named daily timedelta index to object gives an object index of timedelta scalars with the same name; converting the original to a list gives the same scalars.
        idx = timedelta_range(start="1 days", periods=4, freq="D", name="idx")
        expected_list = [
            Timedelta("1 days"),
            Timedelta("2 days"),
            Timedelta("3 days"),
            Timedelta("4 days"),
        ]
        result = idx.astype(object)
        expected = Index(expected_list, dtype=object, name="idx")
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype_object_with_nat(self):
        # [FUNCTION DOC]
        # Same with a missing value in the middle: the object result keeps NaT in that position.
        idx = TimedeltaIndex(
            [timedelta(days=1), timedelta(days=2), NaT, timedelta(days=4)], name="idx"
        )
        expected_list = [
            Timedelta("1 days"),
            Timedelta("2 days"),
            NaT,
            Timedelta("4 days"),
        ]
        result = idx.astype(object)
        expected = Index(expected_list, dtype=object, name="idx")
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype(self, using_infer_string):
        # [FUNCTION DOC]
        # Uses the infer-string flag. For an index of one value of 1e14 nanoseconds and three missing values: cast to object gives timedeltas and NaT; cast to int64 gives the integer value then the minimum 64-bit integer for each missing; cast to str gives string forms (missing as None under string inference with str dtype; otherwise str of NaT with object dtype); cast of a range to 64-bit integers gives its integer representation, equal to the stored integer values.
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan], name="idx")

        result = idx.astype(object)
        expected = Index(
            [Timedelta("1 days 03:46:40")] + [NaT] * 3, dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(np.int64)
        expected = Index(
            [100000000000000] + [-9223372036854775808] * 3, dtype=np.int64, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(str)
        if using_infer_string:
            expected = Index(
                [str(x) if x is not NaT else None for x in idx], name="idx", dtype="str"
            )
        else:
            expected = Index([str(x) for x in idx], name="idx", dtype=object)
        tm.assert_index_equal(result, expected)

        rng = timedelta_range("1 days", periods=10)
        result = rng.astype("i8")
        tm.assert_index_equal(result, Index(rng.asi8))
        tm.assert_numpy_array_equal(rng.asi8, result.values)

    def test_astype_uint(self):
        # [FUNCTION DOC]
        # Casting to unsigned 64 or 32 bit integer raises a type error suggesting to cast to int64 instead.
        arr = timedelta_range("1h", periods=2)

        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint64")
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint32")

    def test_astype_timedelta64(self):
        # [FUNCTION DOC]
        # Casting to the unit-less timedelta64 raises a value error listing supported resolutions. Casting to nanosecond timedelta returns an equal but distinct object by default, and the identical object with copy disabled.
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan])

        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        with pytest.raises(ValueError, match=msg):
            idx.astype("timedelta64")

        result = idx.astype("timedelta64[ns]")
        tm.assert_index_equal(result, idx)
        assert result is not idx

        result = idx.astype("timedelta64[ns]", copy=False)
        tm.assert_index_equal(result, idx)
        assert result is idx

    def test_astype_to_td64d_raises(self, index_or_series):
        # [FUNCTION DOC]
        # Uses index_or_series. Casting nanosecond timedeltas to day resolution raises a value error stating that conversion is unsupported and listing supported resolutions.
        # We don't support "D" reso
        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )
        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64\[D\]. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        with pytest.raises(ValueError, match=msg):
            td.astype("timedelta64[D]")

    def test_astype_ms_to_s(self, index_or_series):
        # [FUNCTION DOC]
        # Uses index_or_series. Casting a nanosecond timedelta container to seconds equals one constructed from the values truncated to second resolution through the private array constructor, and has second dtype. (The test name says milliseconds but the source resolution is nanoseconds.)
        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )

        exp_values = np.asarray(td).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype)
        expected = index_or_series(exp_tda)
        assert expected.dtype == "m8[s]"
        result = td.astype("timedelta64[s]")
        tm.assert_equal(result, expected)

    def test_astype_freq_conversion(self):
        # [FUNCTION DOC]
        # Casting a daily-frequency timedelta index to seconds converts the values to second resolution and keeps the frequency; the same cast on the underlying array equals the index values, and on a series the values equal those without frequency.
        # pre-2.0 td64 astype converted to float64. now for supported units
        #  (s, ms, us, ns) this converts to the requested dtype.
        # This matches TDA and Series
        tdi = timedelta_range("1 Day", periods=30)

        res = tdi.astype("m8[s]")
        exp_values = np.asarray(tdi).astype("m8[s]")
        exp_tda = TimedeltaArray._simple_new(
            exp_values, dtype=exp_values.dtype, freq=tdi.freq
        )
        expected = Index(exp_tda)
        assert expected.dtype == "m8[s]"
        tm.assert_index_equal(res, expected)

        # check this matches Series and TimedeltaArray
        res = tdi._data.astype("m8[s]")
        tm.assert_equal(res, expected._values)

        res = tdi.to_series().astype("m8[s]")
        tm.assert_equal(res._values, expected._values._with_freq(None))

    @pytest.mark.parametrize("dtype", [float, "datetime64", "datetime64[ns]"])
    def test_astype_raises(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over float, unit-less datetime64 and nanosecond datetime64. Casting raises a type error "Cannot cast TimedeltaIndex to dtype".
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.nan])
        msg = "Cannot cast TimedeltaIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_astype_category(self):
        # [FUNCTION DOC]
        # Casting an hourly range to category gives a categorical index of the same timedeltas; the underlying array's cast gives the equal categorical.
        obj = timedelta_range("1h", periods=2, freq="h")

        result = obj.astype("category")
        expected = pd.CategoricalIndex([Timedelta("1h"), Timedelta("2h")])
        tm.assert_index_equal(result, expected)

        result = obj._data.astype("category")
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    def test_astype_array_fallback(self):
        # [FUNCTION DOC]
        # Casting non-zero timedeltas to boolean gives an all-true boolean index, and likewise numpy array from the underlying array.
        obj = timedelta_range("1h", periods=2)
        result = obj.astype(bool)
        expected = Index(np.array([True, True]))
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
