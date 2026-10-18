# [MODULE DOC]
# Tests for Series.searchsorted: sides, scalar and vector inputs, datetime
# values with mixed time zones, sorter, and invalid frame input. No import-time
# side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import is_scalar


class TestSeriesSearchSorted:
    # [CLASS DOC]
    # Groups the tests of Series.searchsorted. Stateless.
    def test_searchsorted(self):
        # [FUNCTION DOC]
        # For sorted integers, the left side of 1 gives position 0 and the right side
        # gives 1, both as scalars.
        ser = Series([1, 2, 3])

        result = ser.searchsorted(1, side="left")
        assert is_scalar(result)
        assert result == 0

        result = ser.searchsorted(1, side="right")
        assert is_scalar(result)
        assert result == 1

    def test_searchsorted_numeric_dtypes_scalar(self):
        # [FUNCTION DOC]
        # A float Series with large values: a scalar needle returns a scalar position
        # (2 for 30), and a list needle returns an array of positions.
        ser = Series([1, 2, 90, 1000, 3e9])
        res = ser.searchsorted(30)
        assert is_scalar(res)
        assert res == 2

        res = ser.searchsorted([30])
        exp = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_numeric_dtypes_vector(self):
        # [FUNCTION DOC]
        # A list of needles gives an array of positions with the platform integer type.
        ser = Series([1, 2, 90, 1000, 3e9])
        res = ser.searchsorted([91, 2e6])
        exp = np.array([3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_datetime64_scalar(self):
        # [FUNCTION DOC]
        # A Timestamp needle on a datetime Series returns the scalar position 1.
        ser = Series(date_range("20120101", periods=10, freq="2D"))
        val = Timestamp("20120102")
        res = ser.searchsorted(val)
        assert is_scalar(res)
        assert res == 1

    def test_searchsorted_datetime64_scalar_mixed_timezones(self):
        # [FUNCTION DOC]
        # GH 30086: a Timestamp needle in another time zone is compared by instant,
        # position 1.
        # GH 30086
        ser = Series(date_range("20120101", periods=10, freq="2D", tz="UTC"))
        val = Timestamp("20120102", tz="America/New_York")
        res = ser.searchsorted(val)
        assert is_scalar(res)
        assert res == 1

    def test_searchsorted_datetime64_list(self):
        # [FUNCTION DOC]
        # A list of Timestamps gives positions as an array.
        ser = Series(date_range("20120101", periods=10, freq="2D"))
        vals = [Timestamp("20120102"), Timestamp("20120104")]
        res = ser.searchsorted(vals)
        exp = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_sorter(self):
        # [FUNCTION DOC]
        # GH8490: with the sorter argument (argsort of the Series), positions refer to
        # the sorted order.
        # GH8490
        ser = Series([3, 1, 2])
        res = ser.searchsorted([0, 3], sorter=np.argsort(ser))
        exp = np.array([0, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(res, exp)

    def test_searchsorted_dataframe_fail(self):
        # [FUNCTION DOC]
        # GH#49620: searching with a DataFrame needle raises a value error saying the
        # value must be one-dimensional array-like or scalar.
        # GH#49620
        ser = Series([1, 2, 3, 4, 5])
        vals = pd.DataFrame([[1, 2], [3, 4]])
        msg = "Value must be 1-D array-like or scalar, DataFrame is not supported"
        with pytest.raises(ValueError, match=msg):
            ser.searchsorted(vals)
