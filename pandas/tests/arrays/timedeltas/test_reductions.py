# [MODULE DOC]
# Tests for reductions (std, min, max, median, mean, sum) on TimedeltaArray and
# TimedeltaIndex, including empty inputs, skipna handling, 2-d arrays and the
# equivalent datetime standard deviation. Uses the skipna and unit fixtures from
# the suite conftest. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays import TimedeltaArray


class TestReductions:
    # [CLASS DOC]
    # Groups tests of TimedeltaArray reductions.
    @pytest.mark.parametrize("name", ["std", "min", "max", "median", "mean"])
    def test_reductions_empty(self, name, skipna):
        # [FUNCTION DOC]
        # Parametrized over reductions std, min, max, median, mean; uses the skipna fixture.
        # Verifies on an empty index and array each reduction returns NaT.
        tdi = pd.TimedeltaIndex([])
        arr = tdi.array

        result = getattr(tdi, name)(skipna=skipna)
        assert result is pd.NaT

        result = getattr(arr, name)(skipna=skipna)
        assert result is pd.NaT

    def test_sum_empty(self, skipna):
        # [FUNCTION DOC]
        # Uses the skipna fixture. Verifies the sum of an empty index or array is a
        # zero Timedelta whatever skipna is.
        tdi = pd.TimedeltaIndex([])
        arr = tdi.array

        result = tdi.sum(skipna=skipna)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        result = arr.sum(skipna=skipna)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

    def test_min_max(self, unit):
        # [FUNCTION DOC]
        # Uses the unit fixture. Verifies min and max skip NaT by default (2h and 5h) and
        # return NaT when not skipping.
        dtype = f"m8[{unit}]"
        arr = TimedeltaArray._from_sequence(
            ["3h", "3h", "NaT", "2h", "5h", "4h"], dtype=dtype
        )

        result = arr.min()
        expected = Timedelta("2h")
        assert result == expected

        result = arr.max()
        expected = Timedelta("5h")
        assert result == expected

        result = arr.min(skipna=False)
        assert result is pd.NaT

        result = arr.max(skipna=False)
        assert result is pd.NaT

    def test_sum(self):
        # [FUNCTION DOC]
        # Verifies sum on array and index: 17 hours when skipping NaT, NaT when not
        # skipping, NaT when min_count (9) exceeds the number of valid values and 17 hours
        # with min_count 1; the result is a Timedelta.
        tdi = pd.TimedeltaIndex(["3h", "3h", "NaT", "2h", "5h", "4h"])
        arr = tdi.array

        result = arr.sum(skipna=True)
        expected = Timedelta(hours=17)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.sum(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = arr.sum(skipna=False)
        assert result is pd.NaT

        result = tdi.sum(skipna=False)
        assert result is pd.NaT

        result = arr.sum(min_count=9)
        assert result is pd.NaT

        result = tdi.sum(min_count=9)
        assert result is pd.NaT

        result = arr.sum(min_count=1)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.sum(min_count=1)
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_npsum(self):
        # [FUNCTION DOC]
        # Regression test (GH#25282, GH#25335). Verifies numpy sum of an index or array
        # returns a Timedelta (17 hours), not a numpy timedelta64.
        # GH#25282, GH#25335 np.sum should return a Timedelta, not timedelta64
        tdi = pd.TimedeltaIndex(["3h", "3h", "2h", "5h", "4h"])
        arr = tdi.array

        result = np.sum(tdi)
        expected = Timedelta(hours=17)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = np.sum(arr)
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_sum_2d_skipna_false(self):
        # [FUNCTION DOC]
        # Verifies sum of a 2-d array with NaT not skipping: whole array gives NaT, axis
        # zero gives a per-column sum with NaT in the column with the missing value, axis
        # one gives per-row sums with NaT for the last row.
        arr = np.arange(8).astype(np.int64).view("m8[s]").astype("m8[ns]").reshape(4, 2)
        arr[-1, -1] = "Nat"

        tda = TimedeltaArray._from_sequence(arr)

        result = tda.sum(skipna=False)
        assert result is pd.NaT

        result = tda.sum(axis=0, skipna=False)
        expected = pd.TimedeltaIndex(
            [Timedelta(seconds=12), pd.NaT], dtype="m8[ns]"
        )._values
        tm.assert_timedelta_array_equal(result, expected)

        result = tda.sum(axis=1, skipna=False)
        expected = pd.TimedeltaIndex(
            [
                Timedelta(seconds=1),
                Timedelta(seconds=5),
                Timedelta(seconds=9),
                pd.NaT,
            ],
            dtype="m8[ns]",
        )._values
        tm.assert_timedelta_array_equal(result, expected)

    # Adding a Timestamp makes this a test for DatetimeArray.std
    @pytest.mark.parametrize(
        "add",
        [
            Timedelta(0).as_unit("us"),
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-01-01", tz="Asia/Tokyo"),
        ],
    )
    def test_std(self, add):
        # [FUNCTION DOC]
        # Parametrized over an added offset (zero microsecond timedelta, naive timestamp,
        # UTC and Tokyo timestamps), so with a timestamp this tests the datetime standard
        # deviation. Verifies the standard deviation (skipna true) of the offsets 0, 4, NaT,
        # 4, 0, 2 hours is a two-hour Timedelta in microsecond unit for array and index,
        # equals the nanops function result (numpy timedelta64) for tz-naive data, and
        # is NaT when not skipping (NaT numpy value from nanops).
        tdi = pd.TimedeltaIndex(["0h", "4h", "NaT", "4h", "0h", "2h"]) + add
        arr = tdi.array

        result = arr.std(skipna=True)
        expected = Timedelta(hours=2).as_unit("us")
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.std(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        if getattr(arr, "tz", None) is None:
            result = nanops.nanstd(np.asarray(arr), skipna=True)
            assert isinstance(result, np.timedelta64)
            assert result == expected

        result = arr.std(skipna=False)
        assert result is pd.NaT

        result = tdi.std(skipna=False)
        assert result is pd.NaT

        if getattr(arr, "tz", None) is None:
            result = nanops.nanstd(np.asarray(arr), skipna=False)
            assert isinstance(result, np.timedelta64)
            assert np.isnat(result)

    def test_median(self):
        # [FUNCTION DOC]
        # Verifies the median skipping NaT of the sample (0, 3, 5h06m, 0, 2 hours) is two
        # hours for array and index, and NaT when not skipping.
        tdi = pd.TimedeltaIndex(["0h", "3h", "NaT", "5h06m", "0h", "2h"])
        arr = tdi.array

        result = arr.median(skipna=True)
        expected = Timedelta(hours=2)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.median(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = arr.median(skipna=False)
        assert result is pd.NaT

        result = tdi.median(skipna=False)
        assert result is pd.NaT

    def test_mean(self):
        # [FUNCTION DOC]
        # Verifies mean equals the Timedelta of the numpy mean of the non-missing values;
        # NaT when not skipping; equal when applied after dropping missing values; and
        # with axis zero equal to the plain mean.
        tdi = pd.TimedeltaIndex(["0h", "3h", "NaT", "5h06m", "0h", "2h"])
        arr = tdi._data

        # manually verified result
        expected = Timedelta(arr.dropna()._ndarray.mean())

        result = arr.mean()
        assert result == expected
        result = arr.mean(skipna=False)
        assert result is pd.NaT

        result = arr.dropna().mean(skipna=False)
        assert result == expected

        result = arr.mean(axis=0)
        assert result == expected

    def test_mean_2d(self):
        # [FUNCTION DOC]
        # Verifies mean on a 2-d array of days: axis zero gives the middle row, axis one
        # gives the first column plus twelve hours (microsecond unit), and axis none equals
        # the index mean.
        tdi = pd.timedelta_range("14 days", periods=6)
        tda = tdi._data.reshape(3, 2)

        result = tda.mean(axis=0)
        expected = tda[1]
        tm.assert_timedelta_array_equal(result, expected)

        result = tda.mean(axis=1)
        expected = tda[:, 0] + Timedelta(hours=12).as_unit("us")
        tm.assert_timedelta_array_equal(result, expected)

        result = tda.mean(axis=None)
        expected = tdi.mean()
        assert result == expected
