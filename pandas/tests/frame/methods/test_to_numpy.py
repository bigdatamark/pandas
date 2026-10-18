# [MODULE DOC]
# Tests for DataFrame.to_numpy: dtype conversion, copy semantics, string conversion
# of mixed dtypes and datetime missing-value handling.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestToNumpy:
    # [CLASS DOC]
    # Test container for DataFrame.to_numpy.
    def test_to_numpy(self):
        # [FUNCTION DOC]
        # A mixed integer and float frame converts to a two-dimensional float array with the same values.
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})
        expected = np.array([[1, 3], [2, 4.5]])
        result = df.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_dtype(self):
        # [FUNCTION DOC]
        # Passing an integer dtype converts the same mixed frame to an int64 array, truncating 4.5 to 4.
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})
        expected = np.array([[1, 3], [2, 4]], dtype="int64")
        result = df.to_numpy(dtype="int64")
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_copy(self):
        # [FUNCTION DOC]
        # Builds a frame from a random array and checks base-array identity: the frame's values do not
        # have the source array as base, requesting no copy returns an array sharing the frame's base,
        # and requesting a copy does not return the source array as base.
        # KNOWN ISSUE: the trailing comment talks about passing a missing-value marker, but the code repeats the same no-copy check without passing any such argument.
        arr = np.random.default_rng(2).standard_normal((4, 3))
        df = DataFrame(arr)
        assert df.values.base is not arr
        assert df.to_numpy(copy=False).base is df.values.base
        assert df.to_numpy(copy=True).base is not arr

        # we still don't want a copy when na_value=np.nan is passed,
        #  and that can be respected because we are already numpy-float
        assert df.to_numpy(copy=False).base is df.values.base

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_to_numpy_mixed_dtype_to_str(self):
        # [FUNCTION DOC]
        # Regression test (GH 35455). Converting a frame mixing a timestamp and a float to the
        # string dtype gives a string array with the formatted timestamp and "100.0". A cast
        # RuntimeWarning is filtered out.
        # https://github.com/pandas-dev/pandas/issues/35455
        df = DataFrame([[Timestamp("2020-01-01 00:00:00"), 100.0]])
        result = df.to_numpy(dtype=str)
        expected = np.array([["2020-01-01 00:00:00", "100.0"]], dtype=str)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_datetime_with_na(self):
        # [FUNCTION DOC]
        # Regression test (GH 53115). Converting datetime data containing NaT to float with the
        # missing-value argument set to NaN yields NaN for NaT and epoch nanoseconds otherwise.
        # Second part uses a frame with second-resolution timestamp columns (with NaT / NaN) and a numeric
        # column; expects epoch seconds as floats and NaN in missing positions.
        # GH #53115
        dti = date_range("2016-01-01", periods=3, unit="ns")
        df = DataFrame(dti)
        df.iloc[0, 0] = NaT
        expected = np.array([[np.nan], [1.45169280e18], [1.45177920e18]])
        result = df.to_numpy(float, na_value=np.nan)
        tm.assert_numpy_array_equal(result, expected)

        df = DataFrame(
            {
                "a": [
                    Timestamp("1970-01-01").as_unit("s"),
                    Timestamp("1970-01-02").as_unit("s"),
                    NaT,
                ],
                "b": [
                    Timestamp("1970-01-01").as_unit("s"),
                    np.nan,
                    Timestamp("1970-01-02").as_unit("s"),
                ],
                "c": [
                    1,
                    np.nan,
                    2,
                ],
            }
        )
        expected = np.array(
            [
                [0.00e00, 0.00e00, 1.00e00],
                [8.64e04, np.nan, np.nan],
                [np.nan, 8.64e04, 2.00e00],
            ]
        )
        result = df.to_numpy(float, na_value=np.nan)
        tm.assert_numpy_array_equal(result, expected)
