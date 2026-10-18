# [MODULE DOC]
# Tests for comparison and equality of the nullable floating array. Reuses the shared comparison and numeric
# operator test mixins from the masked-array shared test module. Uses fixtures comparison_op, dtype and
# using_nan_is_na. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.tests.arrays.masked_shared import (
    ComparisonOps,
    NumericOps,
)


class TestComparisonOps(NumericOps, ComparisonOps):
    # [CLASS DOC]
    # Test class composed of the shared numeric-operator and comparison-operator mixins (inherits all their tests
    # for the floating array); only adds scalar parametrization and an integer-array comparison.
    @pytest.mark.parametrize("other", [True, False, pd.NA, -1.0, 0.0, 1])
    def test_scalar(self, other, comparison_op, dtype):
        # [FUNCTION DOC]
        # Parametrized over booleans, NA, floats and an integer scalar. Delegates to the shared comparison mixin scalar test with the given operator fixture and dtype.
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_compare_with_integerarray(self, comparison_op):
        # [FUNCTION DOC]
        # Comparing a nullable integer array with a nullable float array equals comparing it with the float array cast to integer, in both operand orders.
        op = comparison_op
        a = pd.array([0, 1, None] * 3, dtype="Int64")
        b = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype="Float64")
        other = b.astype("Int64")
        expected = op(a, other)
        result = op(a, b)
        tm.assert_extension_array_equal(result, expected)
        expected = op(other, a)
        result = op(b, a)
        tm.assert_extension_array_equal(result, expected)


def test_equals():
    # [FUNCTION DOC]
    # Two arrays with the same values but different float dtypes (64-bit and 32-bit) are not equal according to the equality method.
    # GH-30652
    # equals is generally tested in /tests/extension/base/methods, but this
    # specifically tests that two arrays of the same class but different dtype
    # do not evaluate equal
    a1 = pd.array([1, 2, None], dtype="Float64")
    a2 = pd.array([1, 2, None], dtype="Float32")
    assert a1.equals(a2) is False


def test_equals_nan_vs_na(using_nan_is_na):
    # [FUNCTION DOC]
    # Constructing arrays directly from data and mask. Arrays with a NaN in the data compare equal to themselves and
    # copies; without NaN-is-NA the NaN differs from a masked entry, with it the constructor-level mask makes them equal.
    # Entries that are masked do not take part in the equality.
    # GH#44382

    mask = np.zeros(3, dtype=bool)
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    if using_nan_is_na:
        # Under PDEP16, all callers of the FloatingArray constructor should
        #  ensure that mask[np.isnan(data)] = True
        mask[1] = True

    left = FloatingArray(data, mask)
    assert left.equals(left)
    tm.assert_extension_array_equal(left, left)

    assert left.equals(left.copy())
    assert left.equals(FloatingArray(data.copy(), mask.copy()))

    mask2 = np.array([False, True, False], dtype=bool)
    data2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    right = FloatingArray(data2, mask2)
    assert right.equals(right)
    tm.assert_extension_array_equal(right, right)

    if not using_nan_is_na:
        assert not left.equals(right)
    else:
        # the constructor will set the NaN locations to NA
        assert left.equals(right)

    # with mask[1] = True, the only difference is data[1], which should
    #  not matter for equals
    mask[1] = True
    assert left.equals(right)
