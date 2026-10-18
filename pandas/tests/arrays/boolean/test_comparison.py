# [MODULE DOC]
# Tests for comparison operators on the nullable boolean array, reusing the shared masked-array comparison test mixin.
# Defines local fixtures for sample data and dtype that the mixin consumes.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.tests.arrays.masked_shared import ComparisonOps


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # Fixture returning a boolean extension array of eleven elements: alternating true and false with missing values at two positions.
    """Fixture returning boolean array with valid and missing data"""
    return pd.array(
        [True, False] * 2 + [np.nan] + [True, False] + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.fixture
def dtype():
    # [FUNCTION DOC]
    # Fixture returning an instance of the boolean extension dtype, used by the shared comparison mixin.
    """Fixture returning BooleanDtype"""
    return pd.BooleanDtype()


class TestComparisonOps(ComparisonOps):
    # [CLASS DOC]
    # Comparison test class for boolean arrays; inherits the generic comparison tests from the shared masked-array mixin and adds boolean-specific cases.
    def test_compare_scalar(self, data, comparison_op):
        # [FUNCTION DOC]
        # Compares the data fixture against a scalar true using the parametrized comparison operator via the mixin's helper.
        self._compare_other(data, comparison_op, True)

    def test_compare_array(self, data, comparison_op):
        # [FUNCTION DOC]
        # Compares the data fixture with equal-length other operands (a boolean array, a NumPy bool array, and a series) of all true using the parametrized comparison operator via the mixin helper.
        other = pd.array([True] * len(data), dtype="boolean")
        self._compare_other(data, comparison_op, other)
        other = np.array([True] * len(data))
        self._compare_other(data, comparison_op, other)
        other = pd.Series([True] * len(data))
        self._compare_other(data, comparison_op, other)

    @pytest.mark.parametrize("other", [True, False, pd.NA])
    def test_scalar(self, other, comparison_op, dtype):
        # [FUNCTION DOC]
        # Parametrized over other in true, false, and the NA singleton; delegates to the mixin's scalar comparison test with the given operator and dtype.
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_array(self, comparison_op):
        # [FUNCTION DOC]
        # Verifies elementwise comparison of two nine-element boolean arrays covering all true/false/missing pairings. The expected result is a boolean array built from the operator applied to the raw data and the union of the two masks.
        # Also checks that comparing does not mutate the inputs: after setting an element of the result to missing, both operands remain unchanged.
        op = comparison_op
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")

        result = op(a, b)

        values = op(a._data, b._data)
        mask = a._mask | b._mask
        expected = BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = None
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )
