# [MODULE DOC]
# Test for the internal Series method that selects numeric data. No import-time
# side effects.

from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm


class TestGetNumericData:
    # [CLASS DOC]
    # Groups tests of the internal numeric-data selection for Series. Stateless.
    def test_get_numeric_data_preserve_dtype(self):
        # [FUNCTION DOC]
        # For an integer or boolean Series the numeric data equals the Series and is a
        # shallow copy (modifying it does not change the original). For mixed object data
        # and datetime data the result is an empty Series of the same dtype.
        # get the numeric data
        obj = Series([1, 2, 3])
        result = obj._get_numeric_data()
        tm.assert_series_equal(result, obj)

        # returned object is a shallow copy
        result.iloc[0] = 0
        assert obj.iloc[0] == 1

        obj = Series([1, "2", 3.0])
        result = obj._get_numeric_data()
        expected = Series([], dtype=object)
        tm.assert_series_equal(result, expected)

        obj = Series([True, False, True])
        result = obj._get_numeric_data()
        tm.assert_series_equal(result, obj)

        obj = Series(date_range("20130101", periods=3, unit="ns"))
        result = obj._get_numeric_data()
        expected = Series([], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)
