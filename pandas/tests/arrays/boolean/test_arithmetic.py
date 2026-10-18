# [MODULE DOC]
# Tests for arithmetic operators on the nullable boolean array.
# Defines fixtures for sample data and for a left/right operand pair that together cover every true/false/missing pairing, then checks which operators work, which raise, and error handling for invalid operands.
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # Fixture returning an eleven-element boolean array of alternating true and false with missing values at two positions.
    """Fixture returning boolean array with valid and missing values."""
    return pd.array(
        [True, False] * 2 + [np.nan] + [True, False] + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.fixture
def left_array():
    # [FUNCTION DOC]
    # Fixture returning the left operand of nine elements: three true, three false, three missing.
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")


@pytest.fixture
def right_array():
    # [FUNCTION DOC]
    # Fixture returning the right operand of nine elements: the cycle true, false, missing repeated three times, so that paired with the left operand it covers all nine combinations.
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True, False, None] * 3, dtype="boolean")


# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [True, True, None, True, False, None, None, None, None]),
        ("mul", [True, False, None, False, False, None, None, None, None]),
    ],
    ids=["add", "mul"],
)
def test_add_mul(left_array, right_array, opname, exp):
    # [FUNCTION DOC]
    # Parametrized over addition and multiplication with explicit expected lists. Verifies that these operators on two boolean arrays return a boolean extension array where addition acts as logical or and multiplication as logical and, with missing propagated when the result is not determined by the other operand.
    op = getattr(operator, opname)
    result = op(left_array, right_array)
    expected = pd.array(exp, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_sub(left_array, right_array):
    # [FUNCTION DOC]
    # Verifies that subtracting one boolean array from another raises a type error whose message says the NumPy boolean subtract operator is deprecated or not supported and suggests bitwise or logical xor.
    msg = (
        r"numpy boolean subtract, the `-` operator, is (?:deprecated|not supported), "
        r"use the bitwise_xor, the `\^` operator, or the logical_xor function instead\."
    )
    with pytest.raises(TypeError, match=msg):
        left_array - right_array


def test_div(left_array, right_array):
    # [FUNCTION DOC]
    # Verifies that true division of boolean arrays raises a not-implemented error saying the operator is not implemented for bool dtypes, both for the arrays and, for comparison with non-masked behavior, for series built from the raw data arrays.
    msg = "operator '.*' not implemented for bool dtypes"
    with pytest.raises(NotImplementedError, match=msg):
        # check that we are matching the non-masked Series behavior
        pd.Series(left_array._data) / pd.Series(right_array._data)

    with pytest.raises(NotImplementedError, match=msg):
        left_array / right_array


@pytest.mark.parametrize(
    "opname",
    [
        "floordiv",
        "mod",
        "pow",
    ],
)
def test_op_int8(left_array, right_array, opname):
    # [FUNCTION DOC]
    # Parametrized over floor division, modulo, and power. Floor division and power must raise a not-implemented error about bool dtypes. For modulo the result must equal modulo computed after casting both operands to the Int8 nullable dtype.
    op = getattr(operator, opname)
    if opname != "mod":
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            result = op(left_array, right_array)
        return
    result = op(left_array, right_array)
    expected = op(left_array.astype("Int8"), right_array.astype("Int8"))
    tm.assert_extension_array_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    # [FUNCTION DOC]
    # Takes every arithmetic operator by name from a shared fixture and applies it, via a series wrapping the data, to invalid operands: a string scalar, a timestamp scalar, and (except for multiplication, whose reflected and plain forms are excluded because NumPy treats booleans as numbers) a series of strings. Each must raise a type error matching one of several accepted messages that vary by NumPy version and operator.
    # invalid ops
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    msg = (
        "did not contain a loop with signature matching types|"
        "BooleanArray cannot perform the operation|"
        "not supported for the input types, and the inputs could not be safely coerced "
        "to any supported types according to the casting rule ''safe''|"
        "not supported for dtype"
    )
    with pytest.raises(TypeError, match=msg):
        ops("foo")
    msg = "|".join(
        [
            r"unsupported operand type\(s\) for",
            "Concatenation operation is not implemented for NumPy arrays",
            "has no kernel",
            "not supported for dtype",
        ]
    )
    with pytest.raises(TypeError, match=msg):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    if op not in ("__mul__", "__rmul__"):
        # TODO(extension) numpy's mul with object array sees booleans as numbers
        msg = "|".join(
            [
                r"unsupported operand type\(s\) for",
                "can only concatenate str",
                "not all arguments converted during string formatting",
                "has no kernel",
                "not implemented",
                "not supported for dtype",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            ops(pd.Series("foo", index=s.index))
