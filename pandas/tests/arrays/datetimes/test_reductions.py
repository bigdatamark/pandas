# [MODULE DOC]
# Tests for reductions (min, max, median, mean) on DatetimeArray, with and without time zones, empty
# arrays, two-dimensional arrays, axis arguments and skipna. Uses the fixtures tz_naive_fixture, unit and
# skipna from the shared conftest. No import-time side effects.

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestReductions:
    # [CLASS DOC]
    # Groups reduction tests for DatetimeArray; provides the arr1d fixture. Stateless.
    @pytest.fixture
    def arr1d(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # HELPER: fixture. Returns a six element DatetimeArray (dates in January 2000 including one missing value and a
        # duplicate), tz-naive nanosecond or tz-aware depending on the timezone fixture.
        """Fixture returning DatetimeArray with parametrized timezones"""
        tz = tz_naive_fixture
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-03",
                "2000-01-03",
                "NaT",
                "2000-01-02",
                "2000-01-05",
                "2000-01-04",
            ],
            dtype=dtype,
        )
        return arr

    def test_min_max(self, arr1d, unit):
        # [FUNCTION DOC]
        # Min and max of the array converted to the unit fixture equal the earliest and latest timestamps (same unit); with
        # skipna disabled the result is NaT.
        arr = arr1d
        arr = arr.as_unit(unit)
        tz = arr.tz

        result = arr.min()
        expected = pd.Timestamp("2000-01-02", tz=tz).as_unit(unit)
        assert result == expected
        assert result.unit == expected.unit

        result = arr.max()
        expected = pd.Timestamp("2000-01-05", tz=tz).as_unit(unit)
        assert result == expected
        assert result.unit == expected.unit

        result = arr.min(skipna=False)
        assert result is NaT

        result = arr.max(skipna=False)
        assert result is NaT

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_min_max_empty(self, skipna, tz):
        # [FUNCTION DOC]
        # Parametrized over no timezone and US/Central, with the skipna fixture. Min and max of an empty array are NaT.
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        arr = DatetimeArray._from_sequence([], dtype=dtype)
        result = arr.min(skipna=skipna)
        assert result is NaT

        result = arr.max(skipna=skipna)
        assert result is NaT

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_median_empty(self, skipna, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone, with the skipna fixture. Median of an empty array is NaT; reshaped to zero rows by three
        # columns, median along rows gives three NaT and along columns gives an empty array.
        dtype = DatetimeTZDtype(tz=tz) if tz is not None else np.dtype("M8[ns]")
        arr = DatetimeArray._from_sequence([], dtype=dtype)
        result = arr.median(skipna=skipna)
        assert result is NaT

        arr = arr.reshape(0, 3)
        result = arr.median(axis=0, skipna=skipna)
        expected = type(arr)._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected)

        result = arr.median(axis=1, skipna=skipna)
        expected = type(arr)._from_sequence([], dtype=arr.dtype)
        tm.assert_equal(result, expected)

    def test_median(self, arr1d):
        # [FUNCTION DOC]
        # Median skipping missing equals the first element; with skipna disabled it is NaT; dropping missing first gives a valid median; explicit axis zero works.
        arr = arr1d

        result = arr.median()
        assert result == arr[0]
        result = arr.median(skipna=False)
        assert result is NaT

        result = arr.dropna().median(skipna=False)
        assert result == arr[0]

        result = arr.median(axis=0)
        assert result == arr[0]

    def test_median_axis(self, arr1d):
        # [FUNCTION DOC]
        # Median along axis zero equals the default; an axis beyond the dimensions raises a value error about axis range.
        arr = arr1d
        assert arr.median(axis=0) == arr.median()
        assert arr.median(axis=0, skipna=False) is NaT

        msg = r"abs\(axis\) must be less than ndim"
        with pytest.raises(ValueError, match=msg):
            arr.median(axis=1)

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test_median_2d(self, arr1d):
        # [FUNCTION DOC]
        # The array reshaped to one row: overall, per-column and per-row medians are as for the 1D case; all-NaT column stays NaT; the
        # all-NaN slice runtime warning is filtered.
        arr = arr1d.reshape(1, -1)

        # axis = None
        assert arr.median() == arr1d.median()
        assert arr.median(skipna=False) is NaT

        # axis = 0
        result = arr.median(axis=0)
        expected = arr1d
        tm.assert_equal(result, expected)

        # Since column 3 is all-NaT, we get NaT there with or without skipna
        result = arr.median(axis=0, skipna=False)
        expected = arr1d
        tm.assert_equal(result, expected)

        # axis = 1
        result = arr.median(axis=1)
        expected = type(arr)._from_sequence([arr1d.median()], dtype=arr.dtype)
        tm.assert_equal(result, expected)

        result = arr.median(axis=1, skipna=False)
        expected = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected)

    def test_mean(self, arr1d):
        # [FUNCTION DOC]
        # Mean skipping missing equals the first element plus 0.4 days (manually verified); skipna disabled gives NaT; dropna first gives the valid mean; axis zero works.
        arr = arr1d

        # manually verified result
        expected = arr[0] + 0.4 * pd.Timedelta(days=1)

        result = arr.mean()
        assert result == expected
        result = arr.mean(skipna=False)
        assert result is NaT

        result = arr.dropna().mean(skipna=False)
        assert result == expected

        result = arr.mean(axis=0)
        assert result == expected

    def test_mean_2d(self):
        # [FUNCTION DOC]
        # A 3 by 2 tz-aware array: mean along each axis equals the expected middle row, row midpoints, and the mean of the whole index when axis is none.
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific", unit="ns")
        dta = dti._data.reshape(3, 2)

        result = dta.mean(axis=0)
        expected = dta[1]
        tm.assert_datetime_array_equal(result, expected)

        result = dta.mean(axis=1)
        expected = dta[:, 0] + pd.Timedelta(hours=12)
        tm.assert_datetime_array_equal(result, expected)

        result = dta.mean(axis=None)
        expected = dti.mean()
        assert result == expected

    def test_mean_empty(self, arr1d, skipna):
        # [FUNCTION DOC]
        # With the skipna fixture. Mean of an empty array is NaT; the two-dimensional zero-by-three version gives three NaT along
        # axis zero, an empty array along axis one and NaT for all.
        arr = arr1d[:0]

        assert arr.mean(skipna=skipna) is NaT

        arr2d = arr.reshape(0, 3)
        result = arr2d.mean(axis=0, skipna=skipna)
        expected = DatetimeArray._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
        tm.assert_datetime_array_equal(result, expected)

        result = arr2d.mean(axis=1, skipna=skipna)
        expected = arr  # i.e. 1D, empty
        tm.assert_datetime_array_equal(result, expected)

        result = arr2d.mean(axis=None, skipna=skipna)
        assert result is NaT
