# [MODULE DOC]
# Tests for logical operators (and, or, xor and their reflected forms) on the nullable boolean array, including three-valued (Kleene) logic with missing values, scalar and array operands, and error cases.
# Also tests the low-level mask-operation helpers for the Kleene operations directly.
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
    kleene_and,
    kleene_or,
    kleene_xor,
)
from pandas.tests.extension.base import BaseOpsUtil


class TestLogicalOps(BaseOpsUtil):
    # [CLASS DOC]
    # Test class for logical operators on boolean arrays; inherits the operator-testing utility base from the extension test suite. Uses a shared fixture providing every logical operator method name.
    def test_numpy_scalars_ok(self, all_logical_operators):
        # [FUNCTION DOC]
        # Verifies that calling the named logical operator method with a NumPy bool scalar gives the same result as with the equivalent Python bool, for true and false.
        a = pd.array([True, False, None], dtype="boolean")
        op = getattr(a, all_logical_operators)

        tm.assert_extension_array_equal(op(True), op(np.bool_(True)))
        tm.assert_extension_array_equal(op(False), op(np.bool_(False)))

    def get_op_from_name(self, op_name):
        # [FUNCTION DOC]
        # HELPER: maps a dunder operator method name to a binary callable for the base test utility.
        # Strips underscores from the name, appends a trailing underscore unless the name contains xor (so and becomes and_ and or becomes or_, matching the operator module), and looks it up in the operator module. If missing, the name is taken to be a reflected form (leading r): it looks up the unreflected operator and returns a function that swaps its two arguments. Returns the callable.
        short_opname = op_name.strip("_")
        short_opname = short_opname if "xor" in short_opname else short_opname + "_"
        try:
            op = getattr(operator, short_opname)
        except AttributeError:
            # Assume it is the reverse operator
            rop = getattr(operator, short_opname[1:])
            op = lambda x, y: rop(y, x)

        return op

    def test_empty_ok(self, all_logical_operators):
        # [FUNCTION DOC]
        # Verifies that applying each logical operator to an empty boolean array with true, false, or NA returns an empty boolean array equal to the input.
        a = pd.array([], dtype="boolean")
        op_name = all_logical_operators
        result = getattr(a, op_name)(True)
        tm.assert_extension_array_equal(a, result)

        result = getattr(a, op_name)(False)
        tm.assert_extension_array_equal(a, result)

        result = getattr(a, op_name)(pd.NA)
        tm.assert_extension_array_equal(a, result)

    @pytest.mark.parametrize(
        "other", ["a", pd.Timestamp(2017, 1, 1, 12), np.timedelta64(4, "ns")]
    )
    def test_eq_mismatched_type(self, other):
        # [FUNCTION DOC]
        # Parametrized over a string, a timestamp, and a timedelta. Verifies that equality of a boolean array against an operand of an unrelated type gives all false and inequality gives all true, as boolean arrays, rather than raising.
        # GH-44499
        arr = pd.array([True, False])
        result = arr == other
        expected = pd.array([False, False])
        tm.assert_extension_array_equal(result, expected)

        result = arr != other
        expected = pd.array([True, True])
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("other", [[True, False], [True, False, True, False]])
    def test_logical_length_mismatch_raises(self, other, all_logical_operators):
        # [FUNCTION DOC]
        # Parametrized over a list shorter than the array and a list longer than it. For the given logical operator, verifies that a value error saying lengths must match is raised when the operand is a list, a NumPy array, or a boolean array of the wrong length.
        op_name = all_logical_operators
        a = pd.array([True, False, None], dtype="boolean")
        msg = "Lengths must match"

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(other)

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(np.array(other))

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(pd.array(other, dtype="boolean"))

    def test_logical_nan_raises(self, all_logical_operators):
        # [FUNCTION DOC]
        # Verifies that a logical operator with a float NaN scalar operand raises a type error whose message says a float was received instead of a bool or NA.
        op_name = all_logical_operators
        a = pd.array([True, False, None], dtype="boolean")
        msg = "Got float instead"

        with pytest.raises(TypeError, match=msg):
            getattr(a, op_name)(np.nan)

    @pytest.mark.parametrize("other", ["a", 1])
    def test_non_bool_or_na_other_raises(self, other, all_logical_operators):
        # [FUNCTION DOC]
        # Parametrized over a string and an integer. Verifies a logical operator with such an operand raises a type error mentioning the operand's type name.
        a = pd.array([True, False], dtype="boolean")
        with pytest.raises(TypeError, match=str(type(other).__name__)):
            getattr(a, all_logical_operators)(other)

    def test_kleene_or(self):
        # [FUNCTION DOC]
        # Verifies Kleene or across all nine combinations of true, false, and missing: the result is true if either side is true, false only if both are false, missing otherwise. Checks symmetry of the operands and that neither input is mutated.
        # A clear test of behavior.
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a | b
        expected = pd.array(
            [True, True, True, True, False, None, True, None, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        result = b | a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [True, None, None]),
            (True, [True, True, True]),
            (np.bool_(True), [True, True, True]),
            (False, [True, False, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_or_scalar(self, other, expected):
        # [FUNCTION DOC]
        # Parametrized over scalar operands (NA, true, false, and NumPy bool equivalents) with expected outputs for a three-element array of true, false, missing.
        # KNOWN ISSUE: a leftover TODO comment in the body says to test true with false, which this test does not do.
        # Verifies or with the scalar on either side gives the expected array and the original array is unchanged.
        # TODO: test True & False
        a = pd.array([True, False, None], dtype="boolean")
        result = a | other
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        result = other | a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    def test_kleene_and(self):
        # [FUNCTION DOC]
        # Verifies Kleene and across all nine combinations: false if either side is false, true only if both true, missing otherwise. Checks symmetry and that inputs are not mutated.
        # A clear test of behavior.
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a & b
        expected = pd.array(
            [True, False, None, False, False, False, None, False, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        result = b & a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [None, False, None]),
            (True, [True, False, None]),
            (False, [False, False, False]),
            (np.bool_(True), [True, False, None]),
            (np.bool_(False), [False, False, False]),
        ],
    )
    def test_kleene_and_scalar(self, other, expected):
        # [FUNCTION DOC]
        # Parametrized over scalar operands (NA, true, false, and NumPy bool variants) with expected outputs. Verifies and with the scalar on either side gives the expected array and the input is unchanged.
        a = pd.array([True, False, None], dtype="boolean")
        result = a & other
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        result = other & a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    def test_kleene_xor(self):
        # [FUNCTION DOC]
        # Verifies exclusive or across all nine combinations: missing if either side is missing, otherwise true when the values differ. Checks symmetry and that inputs are not mutated.
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a ^ b
        expected = pd.array(
            [False, True, None, True, False, None, None, None, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        result = b ^ a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [None, None, None]),
            (True, [False, True, None]),
            (np.bool_(True), [False, True, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_xor_scalar(self, other, expected):
        # [FUNCTION DOC]
        # Parametrized over scalar operands (NA, true, NumPy true, NumPy false) with expected arrays. Verifies xor with the scalar on either side gives the expected array and the input is unchanged.
        a = pd.array([True, False, None], dtype="boolean")
        result = a ^ other
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        result = other ^ a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    @pytest.mark.parametrize("other", [True, False, pd.NA, [True, False, None] * 3])
    def test_no_masked_assumptions(self, other, all_logical_operators):
        # [FUNCTION DOC]
        # Parametrized over true, false, NA, and a nine-element list. Verifies logical operators do not assume the data under a masked position is false: an array whose masked slots hold arbitrary values must give the same result as a reference array with the same mask. When the operand is a boolean array it additionally fills the operand's masked slots with true and the first array's masked slots with false, and checks the result again is unchanged.
        # This test mutates the operand array it builds.
        # The logical operations should not assume that masked values are False!
        a = pd.arrays.BooleanArray(
            np.array([True, True, True, False, False, False, True, False, True]),
            np.array([False] * 6 + [True, True, True]),
        )
        b = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        if isinstance(other, list):
            other = pd.array(other, dtype="boolean")

        result = getattr(a, all_logical_operators)(other)
        expected = getattr(b, all_logical_operators)(other)
        tm.assert_extension_array_equal(result, expected)

        if isinstance(other, BooleanArray):
            other._data[other._mask] = True
            a._data[a._mask] = False

            result = getattr(a, all_logical_operators)(other)
            expected = getattr(b, all_logical_operators)(other)
            tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("operation", [kleene_or, kleene_xor, kleene_and])
def test_error_both_scalar(operation):
    # [FUNCTION DOC]
    # Parametrized over the Kleene or, xor, and and helpers. Verifies that calling a helper with two scalar values (and two dummy zero-filled masks) raises a type error saying either left or right must be an ndarray; masks must be non-None to avoid infinite recursion.
    msg = r"Either `left` or `right` need to be an np\.ndarray."
    with pytest.raises(TypeError, match=msg):
        # masks need to be non-None, otherwise it ends up in an infinite recursion
        operation(True, True, np.zeros(1), np.zeros(1))
