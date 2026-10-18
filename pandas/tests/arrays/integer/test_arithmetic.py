# [MODULE DOC]
# Tests for arithmetic on the nullable integer array (IntegerArray): add, multiply, subtract, true and floor
# division, modulo, division by zero, powers (scalar, array and reversed), zero-dimensional numpy operands,
# invalid operand errors, scalar coercion, cross-type arithmetic, reductions to float, unary operators, NA
# multiplication on large Series and bitwise operators. Uses fixtures dtype, data (from the local conftest) and
# using_nan_is_na, all_arithmetic_operators, any_int_ea_dtype, any_signed_int_ea_dtype, using_python_scalars
# (shared conftest). No import-time side effects.

import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray

# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [("add", [1, 3, None, None, 9]), ("mul", [0, 2, None, None, 20])],
    ids=["add", "mul"],
)
def test_add_mul(dtype, opname, exp):
    # [FUNCTION DOC]
    # Parametrized over addition and multiplication with expected values. Two five-element integer arrays with missing at different
    # places give missing where either side is missing; checked for the operator function and the reflected operator function from the ops module.
    a = pd.array([0, 1, None, 3, 4], dtype=dtype)
    b = pd.array([1, 2, 3, None, 5], dtype=dtype)

    # array / array
    expected = pd.array(exp, dtype=dtype)

    op = getattr(operator, opname)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)

    op = getattr(ops, "r" + opname)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)


def test_sub(dtype):
    # [FUNCTION DOC]
    # Array subtraction propagates missing from either side.
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a - b
    expected = pd.array([1, 1, None, None, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_div(dtype):
    # [FUNCTION DOC]
    # True division of integer arrays gives a nullable Float64 array; division by zero gives positive infinity; missing propagates.
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a / b
    expected = pd.array([np.inf, 2, None, None, 1.25], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("zero, negative", [(0, False), (0.0, False), (-0.0, True)])
def test_divide_by_zero(zero, negative, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over integer zero, float zero and negative float zero. Dividing an Int64 array containing zero, positive, negative and missing by
    # zero gives NaN, positive infinity, negative infinity and missing (signs flipped for negative zero); in NaN-is-NA mode the zero-over-zero entry is also masked.
    # https://github.com/pandas-dev/pandas/issues/27398, GH#22793
    a = pd.array([0, 1, -1, None], dtype="Int64")
    result = a / zero
    exp_mask = np.array([False, False, False, True])
    if using_nan_is_na:
        exp_mask[0] = True
    expected = FloatingArray(
        np.array([np.nan, np.inf, -np.inf, 1], dtype="float64"),
        exp_mask,
    )
    if negative:
        expected *= -1
    tm.assert_extension_array_equal(result, expected)


def test_floordiv(dtype):
    # [FUNCTION DOC]
    # Floor division of integer arrays gives zero for division by zero (unlike Series, which would give infinity), and missing propagates.
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a // b
    # Series op sets 1//0 to np.inf, which IntegerArray does not do (yet)
    expected = pd.array([0, 2, None, None, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_floordiv_by_int_zero_no_mask(any_int_ea_dtype):
    # [FUNCTION DOC]
    # Using any_int_ea_dtype. Integer one floor-divided by a nullable Series containing zero gives infinity then one as Float64; the same on the plain numpy-typed Series gives the equal float64 result.
    # GH 48223: Aligns with non-masked floordiv
    # but differs from numpy
    # https://github.com/pandas-dev/pandas/issues/30188#issuecomment-564452740
    ser = pd.Series([0, 1], dtype=any_int_ea_dtype)
    result = 1 // ser
    expected = pd.Series([np.inf, 1.0], dtype="Float64")
    tm.assert_series_equal(result, expected)

    ser_non_nullable = ser.astype(ser.dtype.numpy_dtype)
    result = 1 // ser_non_nullable
    expected = expected.astype(np.float64)
    tm.assert_series_equal(result, expected)


def test_mod(dtype):
    # [FUNCTION DOC]
    # Modulo of integer arrays gives zero for modulo by zero and propagates missing.
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a % b
    expected = pd.array([0, 0, None, None, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_pow_scalar(using_nan_is_na):
    # [FUNCTION DOC]
    # Powers on an Int64 array with a missing value: exponent zero gives one everywhere, exponent one is identity, exponent NA gives missing except base one,
    # exponent NaN gives floats (masked in NaN-is-NA mode); reversed scalar-base cases for zero, one, NA and NaN bases.
    a = pd.array([-1, 0, 1, None, 2], dtype="Int64")
    result = a**0
    expected = pd.array([1, 1, 1, 1, 1], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = a**1
    expected = pd.array([-1, 0, 1, None, 2], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = a**pd.NA
    expected = pd.array([None, None, 1, None, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = a**np.nan
    if using_nan_is_na:
        expected = expected.astype("Float64")
    else:
        expected = FloatingArray(
            np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype="float64"),
            np.array([False, False, False, True, False]),
        )
    tm.assert_extension_array_equal(result, expected)

    # reversed
    a = a[1:]  # Can't raise integers to negative powers.

    result = 0**a
    expected = pd.array([1, 0, None, 0], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = 1**a
    expected = pd.array([1, 1, 1, 1], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = pd.NA**a
    expected = pd.array([1, None, None, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = np.nan**a
    if using_nan_is_na:
        expected = expected.astype("Float64")
    else:
        expected = FloatingArray(
            np.array([1, np.nan, np.nan, np.nan], dtype="float64"),
            np.array([False, False, True, False]),
        )
    tm.assert_extension_array_equal(result, expected)


def test_pow_array():
    # [FUNCTION DOC]
    # Element-wise power of two integer arrays covering zero, one and missing in both positions; zero exponent and unit base give one.
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None])
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None])
    result = a**b
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None])
    tm.assert_extension_array_equal(result, expected)


def test_rpow_one_to_na():
    # [FUNCTION DOC]
    # A numpy float array raised to an all-missing nullable integer array gives Float64 with one where the base is one and missing elsewhere.
    # https://github.com/pandas-dev/pandas/issues/22022
    # https://github.com/pandas-dev/pandas/issues/29997
    arr = pd.array([pd.NA, pd.NA], dtype="Int64")
    result = np.array([1.0, 2.0]) ** arr
    expected = pd.array([1.0, pd.NA], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("other", [0, 0.5])
def test_numpy_zero_dim_ndarray(other):
    # [FUNCTION DOC]
    # Parametrized over an integer and a float scalar. Adding a zero-dimensional numpy array gives the same result as adding the scalar.
    arr = pd.array([1, None, 2])
    result = arr + np.array(other)
    expected = arr + other
    tm.assert_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    # [FUNCTION DOC]
    # Uses data and all_arithmetic_operators fixtures. Operating on a Series with a string or timestamp raises a type error (of external origin); with a string
    # Series multiplication works by repeating the string, other operators raise; with a datetime Series it raises.
    # KNOWN ISSUE: a commented-out context manager line and comment remain in the body and have no effect.
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    with tm.external_error_raised(TypeError):
        ops("foo")
    with tm.external_error_raised(TypeError):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    str_ser = pd.Series("foo", index=s.index)
    # with pytest.raises(TypeError, match=msg):
    if all_arithmetic_operators in [
        "__mul__",
        "__rmul__",
    ]:  # (data[~data.isna()] >= 0).all():
        res = ops(str_ser)
        expected = pd.Series(["foo" * x for x in data], index=s.index)
        tm.assert_series_equal(res, expected)
    else:
        with tm.external_error_raised(TypeError):
            ops(str_ser)

    with tm.external_error_raised(TypeError):
        ops(pd.Series(pd.date_range("20180101", periods=len(s))))


# Various
# -----------------------------------------------------------------------------


# TODO test unsigned overflow


def test_arith_coerce_scalar(data, all_arithmetic_operators, using_nan_is_na):
    # [FUNCTION DOC]
    # Using data and all_arithmetic_operators. Operating a Series of nullable integers with a float scalar equals the same operation on the float conversion
    # cast to Float64, with missing restored (and for the reflected modulo, entries where the original value was zero are unmasked because they become real NaN).
    op = tm.get_op_from_name(all_arithmetic_operators)
    s = pd.Series(data)
    other = 0.01

    result = op(s, other)
    expected = op(s.astype(float), other)
    expected = expected.astype("Float64")
    if not using_nan_is_na:
        expected[s.isna()] = pd.NA

    # rmod results in NaN that wasn't NA in original nullable Series -> unmask it
    if all_arithmetic_operators == "__rmod__" and not using_nan_is_na:
        mask = (s == 0).fillna(False).to_numpy(bool)
        expected.array._mask[mask] = False

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("other", [1.0, np.array(1.0)])
def test_arithmetic_conversion(all_arithmetic_operators, other):
    # [FUNCTION DOC]
    # Parametrized over a python float and a zero-dimensional float array. Any arithmetic operator with a float operand on an Int64 Series gives a Float64 result.
    # if we have a float operand we should have a float result
    # if that is equal to an integer
    op = tm.get_op_from_name(all_arithmetic_operators)

    s = pd.Series([1, 2, 3], dtype="Int64")
    result = op(s, other)
    assert result.dtype == "Float64"


def test_cross_type_arithmetic():
    # [FUNCTION DOC]
    # Adding Int64, UInt8 and plain integer columns gives nullable Int64 with missing propagating; a comparison gives nullable boolean.
    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, pd.NA], dtype="Int64"),
            "B": pd.Series([1, pd.NA, 3], dtype="UInt8"),
            "C": [1, 2, 3],
        }
    )

    result = df.A + df.C
    expected = pd.Series([2, 4, pd.NA], dtype="Int64")
    tm.assert_series_equal(result, expected)

    result = (df.A + df.C) * 3 == 12
    expected = pd.Series([False, True, None], dtype="boolean")
    tm.assert_series_equal(result, expected)

    result = df.A + df.B
    expected = pd.Series([2, pd.NA, pd.NA], dtype="Int64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", ["mean"])
def test_reduce_to_float(op, using_python_scalars):
    # [FUNCTION DOC]
    # Parametrized over the mean. A mean of a nullable integer column is a float scalar (numpy float, or python float when enabled), and groupby mean keeps
    # the nullable Float64 dtype for it.
    # some reduce ops always return float, even if the result
    # is a rounded number
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([1, None, 3], dtype="Int64"),
        }
    )

    # op
    result = getattr(df.C, op)()
    if using_python_scalars:
        assert type(result) == float
    else:
        assert isinstance(result, np.float64)

    # groupby
    result = getattr(df.groupby("A"), op)()

    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "source, neg_target, abs_target",
    [
        ([1, 2, 3], [-1, -2, -3], [1, 2, 3]),
        ([1, 2, None], [-1, -2, None], [1, 2, None]),
        ([-1, 0, 1], [1, 0, -1], [1, 0, 1]),
    ],
)
def test_unary_int_operators(any_signed_int_ea_dtype, source, neg_target, abs_target):
    # [FUNCTION DOC]
    # Using any_signed_int_ea_dtype. Negation and absolute value give expected values (missing preserved); unary plus returns an equal copy without shared memory.
    dtype = any_signed_int_ea_dtype
    arr = pd.array(source, dtype=dtype)
    neg_result, pos_result, abs_result = -arr, +arr, abs(arr)
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)

    tm.assert_extension_array_equal(neg_result, neg_target)
    tm.assert_extension_array_equal(pos_result, arr)
    assert not tm.shares_memory(pos_result, arr)
    tm.assert_extension_array_equal(abs_result, abs_target)


def test_values_multiplying_large_series_by_NA():
    # [FUNCTION DOC]
    # Multiplying NA by a Series of ten thousand and one zeros gives a Series of NA (regression test for large input).
    # GH#33701

    result = pd.NA * pd.Series(np.zeros(10001))
    expected = pd.Series([pd.NA] * 10001)

    tm.assert_series_equal(result, expected)


def test_bitwise(dtype):
    # [FUNCTION DOC]
    # Or, and and exclusive-or between integer arrays give element-wise results with missing wherever either is missing; combining with a Float64 array raises a type
    # error for unsupported operands.
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)

    result = left | right
    expected = pd.array([None, None, 3 | 5, 4 | 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = left & right
    expected = pd.array([None, None, 3 & 5, 4 & 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = left ^ right
    expected = pd.array([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # TODO: desired behavior when operating with boolean?  defer?

    floats = right.astype("Float64")
    with pytest.raises(TypeError, match="unsupported operand type"):
        left | floats
    with pytest.raises(TypeError, match="unsupported operand type"):
        left & floats
    with pytest.raises(TypeError, match="unsupported operand type"):
        left ^ floats
