# [MODULE DOC]
# Tests for Series.between: inclusive options on datetime, object and period data
# and rejection of invalid inclusive values. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Series,
    bdate_range,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestBetween:
    # [CLASS DOC]
    # Groups the tests of Series.between. Stateless.
    def test_between(self):
        # [FUNCTION DOC]
        # Between two datetime values (default inclusive of both ends) equals the logical
        # and of greater-or-equal and less-or-equal comparisons.
        series = Series(date_range("1/1/2000", periods=10))
        left, right = series[[2, 7]]

        result = series.between(left, right)
        expected = (series >= left) & (series <= right)
        tm.assert_series_equal(result, expected)

    def test_between_datetime_object_dtype(self):
        # [FUNCTION DOC]
        # Object-dtype Series of business dates with every second entry NaN: between with
        # default inclusion selects labels 3 to 17 without missing values; with neither
        # end included it selects labels 5 to 15 without missing values.
        ser = Series(bdate_range("1/1/2000", periods=20), dtype=object)
        ser[::2] = np.nan

        result = ser[ser.between(ser[3], ser[17])]
        expected = ser[3:18].dropna()
        tm.assert_series_equal(result, expected)

        result = ser[ser.between(ser[3], ser[17], inclusive="neither")]
        expected = ser[5:16].dropna()
        tm.assert_series_equal(result, expected)

    def test_between_period_values(self):
        # [FUNCTION DOC]
        # Between on a daily period-valued Series equals the combination of the two
        # comparison operators.
        ser = Series(period_range("2000-01-01", periods=10, freq="D"))
        left, right = ser[[2, 7]]
        result = ser.between(left, right)
        expected = (ser >= left) & (ser <= right)
        tm.assert_series_equal(result, expected)

    def test_between_inclusive_string(self):
        # [FUNCTION DOC]
        # Regression test (GH 40628): the string options for inclusive (both, left,
        # right, neither) each match the corresponding mix of strict and non-strict
        # comparisons.
        # GH 40628
        series = Series(date_range("1/1/2000", periods=10))
        left, right = series[[2, 7]]

        result = series.between(left, right, inclusive="both")
        expected = (series >= left) & (series <= right)
        tm.assert_series_equal(result, expected)

        result = series.between(left, right, inclusive="left")
        expected = (series >= left) & (series < right)
        tm.assert_series_equal(result, expected)

        result = series.between(left, right, inclusive="right")
        expected = (series > left) & (series <= right)
        tm.assert_series_equal(result, expected)

        result = series.between(left, right, inclusive="neither")
        expected = (series > left) & (series < right)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("inclusive", ["yes", True, False])
    def test_between_error_args(self, inclusive):
        # [FUNCTION DOC]
        # Regression test (GH 40628), parametrized over an invalid string and the booleans
        # True and False: between raises a value error saying inclusive must be one of
        # the four allowed strings.
        # KNOWN ISSUE: the Series is created twice (the second assignment is
        # redundant).
        # GH 40628
        series = Series(date_range("1/1/2000", periods=10))
        left, right = series[[2, 7]]

        value_error_msg = (
            "Inclusive has to be either string of 'both','left', 'right', or 'neither'."
        )

        series = Series(date_range("1/1/2000", periods=10))
        with pytest.raises(ValueError, match=value_error_msg):
            series.between(left, right, inclusive=inclusive)
