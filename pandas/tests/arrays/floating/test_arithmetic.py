# [MODULE DOC]
# Tests for arithmetic on the nullable floating array (FloatingArray): the basic binary operators, division
# by zero, powers with scalars, arrays and missing values, zero-dimensional numpy operands, error messages
# for invalid operands, cross-type arithmetic, unary operators and bitwise operators. Uses fixtures dtype,
# data (from the local conftest) and using_nan_is_na, all_arithmetic_operators, float_ea_dtype (from the
# shared conftest). No import-time side effects.

import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray

# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [1.1, 2.2, None, None, 5.5]),
        ("mul", [0.1, 0.4, None, None, 2.5]),
        ("sub", [0.9, 1.8, None, None, 4.5]),
        ("truediv", [10.0, 10.0, None, None, 10.0]),
        ("floordiv", [9.0, 9.0, None, None, 10.0]),
        ("mod", [0.1, 0.2, None, None, 0.0]),
    ],
    ids=["add", "mul", "sub", "div", "floordiv", "mod"],
)
def test_array_op(dtype, opname, exp):
    # [FUNCTION DOC]
    # Parametrized over operator names (add, mul, sub, truediv, floordiv, mod) and expected lists. Applies the operator to
    # two five-element float arrays with missing values at different places; result has missing wherever either
    # operand is missing and the expected values otherwise.
    a = pd.array([1.0, 2.0, None, 4.0, 5.0], dtype=dtype)
    b = pd.array([0.1, 0.2, 0.3, None, 0.5], dtype=dtype)

    op = getattr(operator, opname)

    result = op(a, b)
    expected = pd.array(exp, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("zero, negative", [(0, False), (0.0, False), (-0.0, True)])
def test_divide_by_zero(dtype, zero, negative, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over zero values (integer zero, float zero, negative float zero). Dividing an array with zero,
    # positive, negative and missing entries by zero gives not-a-number, positive infinity, negative infinity and
    # missing, with signs flipped for negative zero; when NaN is treated as NA the zero-over-zero position is also masked.
    # TODO pending NA/NaN discussion
    # https://github.com/pandas-dev/pandas/issues/32265/
    a = pd.array([0, 1, -1, None], dtype=dtype)
    result = a / zero
    exp_mask = np.array([False, False, False, True])
    if using_nan_is_na:
        exp_mask[[0, -1]] = True
    expected = FloatingArray(
        np.array([np.nan, np.inf, -np.inf, np.nan], dtype=dtype.numpy_dtype),
        exp_mask,
    )
    if negative:
        expected *= -1
    tm.assert_extension_array_equal(result, expected)


def test_pow_scalar(dtype, using_nan_is_na):
    # [FUNCTION DOC]
    # Powers with scalars: exponent zero gives one everywhere (even missing), exponent one is identity, exponent NA
    # or NaN gives missing except base one; reversed (scalar base, array exponent) cases for zero, one, NA and NaN
    # bases. NaN-is-NA mode changes whether results are masked or real NaN.
    a = pd.array([-1, 0, 1, None, 2], dtype=dtype)
    result = a**0
    expected = pd.array([1, 1, 1, 1, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = a**1
    expected = pd.array([-1, 0, 1, None, 2], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = a**pd.NA
    expected = pd.array([None, None, 1, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = a**np.nan
    if using_nan_is_na:
        expected = pd.array([None, None, 1, None, None], dtype=dtype)
    else:
        # TODO np.nan should be converted to pd.NA / missing before operation?
        expected = FloatingArray(
            np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype=dtype.numpy_dtype),
            mask=a._mask,
        )
    tm.assert_extension_array_equal(result, expected)

    # reversed
    a = a[1:]  # Can't raise integers to negative powers.

    result = 0**a
    expected = pd.array([1, 0, None, 0], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = 1**a
    expected = pd.array([1, 1, 1, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = pd.NA**a
    expected = pd.array([1, None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = np.nan**a
    if not using_nan_is_na:
        # Otherwise the previous `expected` can be reused
        expected = FloatingArray(
            np.array([1, np.nan, np.nan, np.nan], dtype=dtype.numpy_dtype), mask=a._mask
        )
    tm.assert_extension_array_equal(result, expected)


def test_pow_array(dtype):
    # [FUNCTION DOC]
    # Element-wise power between arrays with all combinations of zero, one and missing, following the rule that zero exponent and unit base give one.
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None], dtype=dtype)
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None], dtype=dtype)
    result = a**b
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_rpow_one_to_na():
    # [FUNCTION DOC]
    # A numpy array raised to an all-missing nullable array gives one where the base is one and missing elsewhere.
    # https://github.com/pandas-dev/pandas/issues/22022
    # https://github.com/pandas-dev/pandas/issues/29997
    arr = pd.array([pd.NA, pd.NA], dtype="Float64")
    result = np.array([1.0, 2.0]) ** arr
    expected = pd.array([1.0, pd.NA], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("other", [0, 0.5])
def test_arith_zero_dim_ndarray(other):
    # [FUNCTION DOC]
    # Parametrized over two scalars. Adding a zero-dimensional numpy array equals adding the scalar itself.
    arr = pd.array([1, None, 2], dtype="Float64")
    result = arr + np.array(other)
    expected = arr + other
    tm.assert_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    # [FUNCTION DOC]
    # Uses the all_arithmetic_operators fixture. Applying the operator on a Series of floats with a string, a timestamp, a
    # string Series or a datetime Series raises a type error matching one of many acceptable messages (they differ by
    # operator and numpy version).
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    msg = "|".join(
        [
            r"can only perform ops with numeric values",
            r"FloatingArray cannot perform the operation mod",
            "unsupported operand type",
            "not all arguments converted during string formatting",
            "can't multiply sequence by non-int of type 'float'",
            "ufunc 'subtract' cannot use operands with types dtype",
            r"can only concatenate str \(not \"float\"\) to str",
            "ufunc '.*' not supported for the input types, and the inputs could not",
            "ufunc '.*' did not contain a loop with signature matching types",
            "Concatenation operation is not implemented for NumPy arrays",
            "has no kernel",
            "not implemented",
            "not supported for dtype",
            "Can only string multiply by an integer",
        ]
    )
    with pytest.raises(TypeError, match=msg):
        ops("foo")
    with pytest.raises(TypeError, match=msg):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    with pytest.raises(TypeError, match=msg):
        ops(pd.Series("foo", index=s.index))

    msg = "|".join(
        [
            "can only perform ops with numeric values",
            "cannot perform .* with this index type: DatetimeArray",
            "Addition/subtraction of integers and integer-arrays "
            "with DatetimeArray is no longer supported. *",
            "unsupported operand type",
            "not all arguments converted during string formatting",
            "can't multiply sequence by non-int of type 'float'",
            "ufunc 'subtract' cannot use operands with types dtype",
            (
                "ufunc 'add' cannot use operands with types "
                rf"dtype\('{tm.ENDIAN}M8\[ns\]'\)"
            ),
            r"ufunc 'add' cannot use operands with types dtype\('float\d{2}'\)",
            "cannot subtract DatetimeArray from ndarray",
            "has no kernel",
            "not implemented",
            "not supported for dtype",
        ]
    )
    with pytest.raises(TypeError, match=msg):
        ops(pd.Series(pd.date_range("20180101", periods=len(s), unit="ns")))


# Various
# -----------------------------------------------------------------------------


def test_cross_type_arithmetic():
    # [FUNCTION DOC]
    # Adding nullable Float64, Float32 and plain numpy float columns gives nullable Float64 results with missing propagating; comparison of the result to a number gives a nullable boolean Series.
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, pd.NA], dtype="Float64"),
            "B": pd.array([1, pd.NA, 3], dtype="Float32"),
            "C": np.array([1, 2, 3], dtype="float64"),
        }
    )

    result = df.A + df.C
    expected = pd.Series([2, 4, pd.NA], dtype="Float64")
    tm.assert_series_equal(result, expected)

    result = (df.A + df.C) * 3 == 12
    expected = pd.Series([False, True, None], dtype="boolean")
    tm.assert_series_equal(result, expected)

    result = df.A + df.B
    expected = pd.Series([2, pd.NA, pd.NA], dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "source, neg_target, abs_target",
    [
        ([1.1, 2.2, 3.3], [-1.1, -2.2, -3.3], [1.1, 2.2, 3.3]),
        ([1.1, 2.2, None], [-1.1, -2.2, None], [1.1, 2.2, None]),
        ([-1.1, 0.0, 1.1], [1.1, 0.0, -1.1], [1.1, 0.0, 1.1]),
    ],
)
def test_unary_float_operators(float_ea_dtype, source, neg_target, abs_target):
    # [FUNCTION DOC]
    # Parametrized over (source, negated target, absolute target) and the float_ea_dtype fixture. Negation and absolute
    # value give the expected arrays (missing preserved); unary plus returns an equal array that does not share memory.
    # GH38794
    dtype = float_ea_dtype
    arr = pd.array(source, dtype=dtype)
    neg_result, pos_result, abs_result = -arr, +arr, abs(arr)
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)

    tm.assert_extension_array_equal(neg_result, neg_target)
    tm.assert_extension_array_equal(pos_result, arr)
    assert not tm.shares_memory(pos_result, arr)
    tm.assert_extension_array_equal(abs_result, abs_target)


def test_bitwise(dtype):
    # [FUNCTION DOC]
    # Or, and and exclusive-or between float arrays raise a type error for unsupported operands.
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)

    with pytest.raises(TypeError, match="unsupported operand type"):
        left | right
    with pytest.raises(TypeError, match="unsupported operand type"):
        left & right
    with pytest.raises(TypeError, match="unsupported operand type"):
        left ^ right
