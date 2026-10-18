"""
Tests shared by MaskedArray subclasses.
"""

# [MODULE DOC]
# HELPER: shared test mixins used by the masked-array test modules (integer, floating, boolean). Provides ComparisonOps (scalar and
# array comparison checks, built on the extension-test operator utility base class) and NumericOps (numeric-only tests such as searchsorted with
# NaN, mask sharing, array comparisons, boolean-array comparisons, string comparison, ufuncs with an out argument, and multiplication with timedelta
# arrays). Concrete test classes inherit from these and supply the dtype, data and comparison_op fixtures. Imports the operator utility base class
# from the extension test package. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil


class ComparisonOps(BaseOpsUtil):
    # [CLASS DOC]
    # HELPER: mixin (subclass of the extension operator-test utility class) holding comparison tests shared by masked arrays. Stateless; subclasses provide fixtures and may override the scalar test to parametrize it.
    def _compare_other(self, data, op, other):
        # [FUNCTION DOC]
        # HELPER: compares an operator applied to a masked array and another operand with the same operator applied to the raw data, as a nullable boolean Series with missing restored at masked positions; checks both the array form and
        # the Series form. Takes the data array, operator function and the other operand. Asserts equality; returns nothing.
        # array
        result = pd.Series(op(data, other))
        expected = pd.Series(op(data._data, other), dtype="boolean")

        # fill the nan locations
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

        # series
        ser = pd.Series(data)
        result = op(ser, other)

        # Set nullable dtype here to avoid upcasting when setting to pd.NA below
        expected = op(pd.Series(data._data), other).astype("boolean")

        # fill the nan locations
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

    # subclass will override to parametrize 'other'
    def test_scalar(self, other, comparison_op, dtype):
        # [FUNCTION DOC]
        # Template test whose other-operand parametrization is supplied by subclasses. Compares a three-element array (with missing) to a scalar with the comparison_op fixture: with NA the result is all NA, else a boolean array
        # with the same mask; also checks that mutating the result does not alter the input.
        op = comparison_op
        left = pd.array([1, 0, None], dtype=dtype)

        result = op(left, other)

        if other is pd.NA:
            expected = pd.array([None, None, None], dtype="boolean")
        else:
            values = op(left._data, other)
            expected = pd.arrays.BooleanArray(values, left._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(left, pd.array([1, 0, None], dtype=dtype))


class NumericOps:
    # [CLASS DOC]
    # HELPER: mixin of tests shared by integer and floating masked arrays (not boolean). Stateless; subclasses provide the dtype and data fixtures.
    # Shared by IntegerArray and FloatingArray, not BooleanArray

    def test_searchsorted_nan(self, dtype):
        # [FUNCTION DOC]
        # Searching for NaN in a range array returns the end position (ten) for both sides, unlike the object-cast base behavior.
        # The base class casts to object dtype, for which searchsorted returns
        #  0 from the left and 10 from the right.
        arr = pd.array(range(10), dtype=dtype)

        assert arr.searchsorted(np.nan, side="left") == 10
        assert arr.searchsorted(np.nan, side="right") == 10

    def test_no_shared_mask(self, data):
        # [FUNCTION DOC]
        # The result of adding a scalar to the data does not share memory (data or mask) with the input.
        result = data + 1
        assert not tm.shares_memory(result, data)

    def test_array(self, comparison_op, dtype):
        # [FUNCTION DOC]
        # Comparing two arrays with different missing positions gives a boolean array with the union of masks; inputs are not mutated by changing the result.
        op = comparison_op

        left = pd.array([0, 1, 2, None, None, None], dtype=dtype)
        right = pd.array([0, 1, None, 0, 1, None], dtype=dtype)

        result = op(left, right)
        values = op(left._data, right._data)
        mask = left._mask | right._mask

        expected = pd.arrays.BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(
            left, pd.array([0, 1, 2, None, None, None], dtype=dtype)
        )
        tm.assert_extension_array_equal(
            right, pd.array([0, 1, None, 0, 1, None], dtype=dtype)
        )

    def test_compare_with_booleanarray(self, comparison_op, dtype):
        # [FUNCTION DOC]
        # Comparing a nullable boolean array with a numeric masked array equals comparing it with the equivalent boolean array, in both operand orders.
        op = comparison_op

        left = pd.array([True, False, None] * 3, dtype="boolean")
        right = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype=dtype)
        other = pd.array([False] * 3 + [True] * 3 + [None] * 3, dtype="boolean")

        expected = op(left, other)
        result = op(left, right)
        tm.assert_extension_array_equal(result, expected)

        # reversed op
        expected = op(other, left)
        result = op(right, left)
        tm.assert_extension_array_equal(result, expected)

    def test_compare_to_string(self, dtype):
        # [FUNCTION DOC]
        # Equality of a numeric Series with a string gives false, with NA where the value is missing, as nullable boolean.
        # GH#28930
        ser = pd.Series([1, None], dtype=dtype)
        result = ser == "a"
        expected = pd.Series([False, pd.NA], dtype="boolean")

        tm.assert_series_equal(result, expected)

    def test_ufunc_with_out(self, dtype):
        # [FUNCTION DOC]
        # Checks a ufunc with an out argument: in-place or-ing a numpy boolean array with a mask works when no missing values exist and raises a value error asking for an na_value otherwise; add gives the expected array;
        # passing the input as out modifies it in place and returns it.
        arr = pd.array([1, 2, 3], dtype=dtype)
        arr2 = pd.array([1, 2, pd.NA], dtype=dtype)

        mask = arr == arr
        mask2 = arr2 == arr2

        result = np.zeros(3, dtype=bool)
        result |= mask
        # If MaskedArray.__array_ufunc__ handled "out" appropriately,
        #  `result` should still be an ndarray.
        assert isinstance(result, np.ndarray)
        assert result.all()

        # result |= mask worked because mask could be cast losslessly to
        #  boolean ndarray. mask2 can't, so this raises
        result = np.zeros(3, dtype=bool)
        msg = "Specify an appropriate 'na_value' for this dtype"
        with pytest.raises(ValueError, match=msg):
            result |= mask2

        # addition
        res = np.add(arr, arr2)
        expected = pd.array([2, 4, pd.NA], dtype=dtype)
        tm.assert_extension_array_equal(res, expected)

        # when passing out=arr, we will modify 'arr' inplace.
        res = np.add(arr, arr2, out=arr)
        assert res is arr
        tm.assert_extension_array_equal(res, expected)
        tm.assert_extension_array_equal(arr, expected)

    def test_mul_td64_array(self, dtype):
        # [FUNCTION DOC]
        # Multiplying a numeric masked array by a timedelta numpy array gives a timedelta extension array with NaT for missing.
        # GH#45622
        arr = pd.array([1, 2, pd.NA], dtype=dtype)
        other = np.arange(3, dtype=np.int64).view("m8[ns]")

        result = arr * other
        expected = pd.array([pd.Timedelta(0), pd.Timedelta(2), pd.NaT])
        tm.assert_extension_array_equal(result, expected)
