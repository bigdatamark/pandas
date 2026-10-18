# [MODULE DOC]
# Tests for unary operators (positive, negative, absolute value, invert) on
# SparseArray, including effects on the fill value, and invert on Series and
# DataFrame wrapping sparse data. No import-time side effects.

import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import SparseArray


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("fill_value", [0, np.nan])
@pytest.mark.parametrize("op", [operator.pos, operator.neg])
def test_unary_op(op, fill_value):
    # [FUNCTION DOC]
    # Parametrized over fill_value (0, NaN) and operator (positive, negative).
    # Verifies applying the operator to a SparseArray gives a SparseArray equal to
    # building one from the operator applied to the dense data, with the fill value
    # also transformed by the operator. Cast RuntimeWarnings are ignored.
    arr = np.array([0, 1, np.nan, 2])
    sparray = SparseArray(arr, fill_value=fill_value)
    result = op(sparray)
    expected = SparseArray(op(arr), fill_value=op(fill_value))
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("fill_value", [True, False])
def test_invert(fill_value):
    # [FUNCTION DOC]
    # Parametrized over boolean fill_value. Verifies inverting a boolean SparseArray
    # inverts values and the fill value, and that inverting a Series or DataFrame
    # holding the sparse array gives the equivalent sparse result.
    arr = np.array([True, False, False, True])
    sparray = SparseArray(arr, fill_value=fill_value)
    result = ~sparray
    expected = SparseArray(~arr, fill_value=not fill_value)
    tm.assert_sp_array_equal(result, expected)

    result = ~pd.Series(sparray)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)

    result = ~pd.DataFrame({"A": sparray})
    expected = pd.DataFrame({"A": expected})
    tm.assert_frame_equal(result, expected)


class TestUnaryMethods:
    # [CLASS DOC]
    # Tests unary operator methods on SparseArray with integer and boolean dtypes.
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_neg_operator(self):
        # [FUNCTION DOC]
        # Verifies negation of int8 SparseArrays with NaN fill and with fill value -1:
        # values and fill value are negated, dtype preserved.
        arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
        res = -arr
        exp = SparseArray([1, 2, np.nan, -3], fill_value=np.nan, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

        arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)
        res = -arr
        exp = SparseArray([1, 2, -1, -3], fill_value=1, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_abs_operator(self):
        # [FUNCTION DOC]
        # Verifies the absolute value of int8 SparseArrays with NaN fill and fill value
        # -1: values and fill value become absolute, dtype preserved.
        arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
        res = abs(arr)
        exp = SparseArray([1, 2, np.nan, 3], fill_value=np.nan, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

        arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)
        res = abs(arr)
        exp = SparseArray([1, 2, 1, 3], fill_value=1, dtype=np.int8)
        tm.assert_sp_array_equal(exp, res)

    def test_invert_operator(self):
        # [FUNCTION DOC]
        # Verifies bitwise invert of a boolean SparseArray (fill value flips) and of an
        # int32 SparseArray with fill value 0 (becomes -1, values become bitwise
        # complements).
        arr = SparseArray([False, True, False, True], fill_value=False, dtype=np.bool_)
        exp = SparseArray(
            np.invert([False, True, False, True]), fill_value=True, dtype=np.bool_
        )
        res = ~arr
        tm.assert_sp_array_equal(exp, res)

        arr = SparseArray([0, 1, 0, 2, 3, 0], fill_value=0, dtype=np.int32)
        res = ~arr
        exp = SparseArray([-1, -2, -1, -3, -4, -1], fill_value=-1, dtype=np.int32)
        tm.assert_sp_array_equal(exp, res)
