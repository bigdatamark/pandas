# [MODULE DOC]
# Tests for the array-level logical and comparison operation helpers in the core ops module
# (the null-aware logical op and the comparison op). Contains three test functions.

import operator

import numpy as np
import pytest

from pandas.core.dtypes.missing import isna

import pandas._testing as tm
from pandas.core.ops.array_ops import (
    comparison_op,
    na_logical_op,
)


def test_na_logical_op_2d():
    # [FUNCTION DOC]
    # Checks that the null-aware logical-or on a two-dimensional integer array and an object array with a
    # NaN falls back to the element-wise path: the direct operator raises a type error (unsupported operand
    # type), while the helper returns the object array unchanged as expected.
    left = np.arange(8).reshape(4, 2)
    right = left.astype(object)
    right[0, 0] = np.nan

    # Check that we fall back to the vec_binop branch
    with pytest.raises(TypeError, match="unsupported operand type"):
        operator.or_(left, right)

    result = na_logical_op(left, right, operator.or_)
    expected = right
    tm.assert_numpy_array_equal(result, expected)


def test_object_comparison_2d():
    # [FUNCTION DOC]
    # Checks that comparison of a 3x3 object array with its transpose gives an identity matrix for
    # equality and its complement for inequality, including when the right operand is read-only (as from
    # broadcasting), which the compiled code must tolerate.
    left = np.arange(9).reshape(3, 3).astype(object)
    right = left.T

    result = comparison_op(left, right, operator.eq)
    expected = np.eye(3).astype(bool)
    tm.assert_numpy_array_equal(result, expected)

    # Ensure that cython doesn't raise on non-writeable arg, which
    #  we can get from np.broadcast_to
    right.flags.writeable = False
    result = comparison_op(left, right, operator.ne)
    tm.assert_numpy_array_equal(result, ~expected)


@pytest.mark.parametrize("rvalues", [1, [1, 1, 1], np.nan, None])
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
)
def test_comparison_for_subclasses(rvalues, op):
    # [FUNCTION DOC]
    # GH#63205: comparison operations must handle ndarray subclasses correctly. Parametrized by the right
    # values (scalar, list, NaN, None) and by each of six comparison operators. For both plain arrays and
    # a local subclass, the result must equal the expected result built with special handling: a zero-dimensional
    # right side that is missing yields all-True for inequality and all-False otherwise.
    # GH#63205 Ensure subclasses of ndarray are correctly handled in comparison_op
    # Define a custom ndarray subclass
    class TestArray(np.ndarray):
        # [CLASS DOC]
        # HELPER: ndarray subclass used to test subclass handling; views input data as itself and sets a
        # marker attribute when finalized.
        def __new__(cls, input_array):
            # [FUNCTION DOC]
            # HELPER: constructor that converts the input array to a numpy array viewed as this subclass.
            return np.asarray(input_array).view(cls)

        def __array_finalize__(self, obj) -> None:
            # [FUNCTION DOC]
            # HELPER: array finalize hook that sets a marker attribute on the new instance; ignores the source object.
            self._is_test_array = True

    def expected_with_na_handling(lvalues, rvalues, op):
        # [FUNCTION DOC]
        # HELPER: computes the expected comparison result for lvalues, rvalues and op; if rvalues is zero
        # dimensional and holds a missing value, returns an all-True (for not-equal) or all-False boolean array
        # shaped like lvalues, since numpy does not compare against none; otherwise applies op directly.
        # Similar to comparison_op, handle zerodim arrays with na value separately
        if (rvalues.ndim == 0) and isna(rvalues.item()):
            # numpy does not like comparisons vs None
            if op is operator.ne:
                return np.ones(lvalues.shape, dtype=bool)
            else:
                return np.zeros(lvalues.shape, dtype=bool)
        return op(lvalues, rvalues)

    # Define test data
    lvalues = [1, 2, 3]

    # Test with both ndarray and TestArray
    result = comparison_op(np.array(lvalues), np.array(rvalues), op)
    expected = expected_with_na_handling(np.array(lvalues), np.array(rvalues), op)
    tm.assert_numpy_array_equal(result, expected)

    result = comparison_op(TestArray(lvalues), TestArray(rvalues), op)
    expected = expected_with_na_handling(TestArray(lvalues), TestArray(rvalues), op)
    tm.assert_numpy_array_equal(result, expected)
