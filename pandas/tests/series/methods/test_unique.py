# [MODULE DOC]
# Tests for the Series method returning the unique values in order of appearance, across dtypes (unsigned ints, floats, object, categorical, interval).
# No import-time side effects.

import numpy as np

from pandas import (
    Categorical,
    IntervalIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestUnique:
    # [CLASS DOC]
    # Groups tests of Series uniqueness. Stateless.
    def test_unique_uint64(self):
        # [FUNCTION DOC]
        # Verifies uniqueness of an unsigned 64-bit Series with values above the signed range returns a uint64 array of the three distinct values in order.
        ser = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        res = ser.unique()
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(res, exp)

    def test_unique_data_ownership(self):
        # [FUNCTION DOC]
        # Smoke test (issue 1807): the array returned by uniqueness can be wrapped in a new Series and sorted without error. No assertions.
        # it works! GH#1807
        Series(Series(["a", "c", "b"]).unique()).sort_values()

    def test_unique(self):
        # [FUNCTION DOC]
        # Verifies that a float Series with every other entry set to NaN has exactly two unique values (the number and NaN), for default float64 and for float32 (issue 714).
        # GH#714 also, dtype=float
        ser = Series([1.2345] * 100)
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

        # explicit f4 dtype
        ser = Series([1.2345] * 100, dtype="f4")
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

    def test_unique_nan_object_dtype(self):
        # [FUNCTION DOC]
        # Verifies that an object-dtype Series of one string with every other entry set to NaN has two unique values (issue 714).
        # NAs in object arrays GH#714
        ser = Series(["foo"] * 100, dtype="O")
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

    def test_unique_none(self):
        # [FUNCTION DOC]
        # Verifies that for an object Series containing integers and repeated None, the unique result keeps a single None as an object array in order of appearance.
        # decision about None
        ser = Series([1, 2, 3, None, None, None], dtype=object)
        result = ser.unique()
        expected = np.array([1, 2, 3, None], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_unique_categorical(self):
        # [FUNCTION DOC]
        # Verifies uniqueness of a Series backed by an empty categorical and by a categorical with only NaN returns a categorical equal to the original (issue 18051).
        # GH#18051
        cat = Categorical([])
        ser = Series(cat)
        result = ser.unique()
        tm.assert_categorical_equal(result, cat)

        cat = Categorical([np.nan])
        ser = Series(cat)
        result = ser.unique()
        tm.assert_categorical_equal(result, cat)

    def test_tz_unique(self):
        # [FUNCTION DOC]
        # Verifies that the unique values of interval-valued Series built from naive and tz-aware datetime breaks equal the Series' own array, and that the two results have different dtypes (issue 46128).
        # GH 46128
        dti1 = date_range("2016-01-01", periods=3)
        ii1 = IntervalIndex.from_breaks(dti1)
        ser1 = Series(ii1)
        uni1 = ser1.unique()
        tm.assert_interval_array_equal(ser1.array, uni1)

        dti2 = date_range("2016-01-01", periods=3, tz="US/Eastern")
        ii2 = IntervalIndex.from_breaks(dti2)
        ser2 = Series(ii2)
        uni2 = ser2.unique()
        tm.assert_interval_array_equal(ser2.array, uni2)

        assert uni1.dtype != uni2.dtype
