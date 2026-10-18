# [MODULE DOC]
# Tests for comparison and equality of the nullable integer array. Reuses the shared comparison and numeric
# operator mixins from the masked-array shared module. Uses fixtures comparison_op and dtype. No import-time side effects.

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.arrays.masked_shared import (
    ComparisonOps,
    NumericOps,
)


class TestComparisonOps(NumericOps, ComparisonOps):
    # [CLASS DOC]
    # Test class composed of the shared numeric-operator and comparison-operator mixins (inherits all their tests); adds integer scalar parametrization and a comparison to a python integer.
    @pytest.mark.parametrize("other", [True, False, pd.NA, -1, 0, 1])
    def test_scalar(self, other, comparison_op, dtype):
        # [FUNCTION DOC]
        # Parametrized over booleans, NA and integers. Delegates to the shared mixin scalar comparison test.
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_compare_to_int(self, dtype, comparison_op):
        # [FUNCTION DOC]
        # Comparing a nullable integer Series with the integer two gives the same result as comparing a float Series with NaN, converted to nullable boolean with NA where missing.
        # GH 28930
        op_name = f"__{comparison_op.__name__}__"
        s1 = pd.Series([1, None, 3], dtype=dtype)
        s2 = pd.Series([1, None, 3], dtype="float")

        method = getattr(s1, op_name)
        result = method(2)

        method = getattr(s2, op_name)
        expected = method(2).astype("boolean")
        expected[s2.isna()] = pd.NA

        tm.assert_series_equal(result, expected)


def test_equals():
    # [FUNCTION DOC]
    # Two integer arrays with same values but different integer dtypes (64-bit and 32-bit) are not equal according to the equality method.
    # GH-30652
    # equals is generally tested in /tests/extension/base/methods, but this
    # specifically tests that two arrays of the same class but different dtype
    # do not evaluate equal
    a1 = pd.array([1, 2, None], dtype="Int64")
    a2 = pd.array([1, 2, None], dtype="Int32")
    assert a1.equals(a2) is False
