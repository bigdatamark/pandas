# [MODULE DOC]
# Tests for comparison operations on categorical data. Contains one test class.

import numpy as np

from pandas import (
    Categorical,
    Series,
)
import pandas._testing as tm


class TestCategoricalComparisons:
    # [CLASS DOC]
    # Test class for equality comparisons involving Categorical-backed Series.
    def test_categorical_nan_equality(self):
        # [FUNCTION DOC]
        # Checks that comparing a categorical series containing a missing value to itself gives True for
        # the valid entries and False for the missing entry (nan is not equal to nan).
        cat = Series(Categorical(["a", "b", "c", np.nan]))
        expected = Series([True, True, True, False])
        result = cat == cat
        tm.assert_series_equal(result, expected)

    def test_categorical_tuple_equality(self):
        # [FUNCTION DOC]
        # Regression test (GH 18050): a series of tuples compared to a tuple scalar gives an element-wise
        # boolean series, both for object dtype and after conversion to category dtype; results are equal.
        # GH 18050
        ser = Series([(0, 0), (0, 1), (0, 0), (1, 0), (1, 1)])
        expected = Series([True, False, True, False, False])
        result = ser == (0, 0)
        tm.assert_series_equal(result, expected)

        result = ser.astype("category") == (0, 0)
        tm.assert_series_equal(result, expected)
