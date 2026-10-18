# [MODULE DOC]
# Tests for the logical (and, or, xor) operators on Series: alignment and fill semantics, mixed dtypes (bool, integer, float, string, object), scalars, array-likes and Index operands, reflected operators, and consistency with frame behavior.
# Uses the internal reflected-operator helper functions from the ops module. No import-time side effects.

from datetime import datetime
import operator

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    bdate_range,
)
import pandas._testing as tm
from pandas.core import ops


class TestSeriesLogicalOps:
    # [CLASS DOC]
    # Groups tests of Series logical operators. Stateless. Key expected rules: operations align on labels and the result index is the union; positions missing on one side are treated as false; dtype-less sequences (lists, tuples) are rejected with a type error; scalars must be compatible with the dtype.
    @pytest.mark.parametrize("bool_op", [operator.and_, operator.or_, operator.xor])
    def test_bool_operators_with_nas(self, bool_op):
        # [FUNCTION DOC]
        # Verifies and, or and xor (parametrized) on object Series built from ten business dates with every other entry set to NaN: comparisons yield boolean results where comparison with missing is false; the result equals the same operation on a filled copy with positions that were missing forced to false.
        # boolean &, |, ^ should work with object arrays and propagate NAs
        ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
        ser[::2] = np.nan

        mask = ser.isna()
        filled = ser.fillna(ser[0])

        result = bool_op(ser < ser[9], ser > ser[3])

        expected = bool_op(filled < filled[9], filled > filled[3])
        expected[mask] = False
        tm.assert_series_equal(result, expected)

    def test_logical_operators_bool_dtype_with_empty(self):
        # [FUNCTION DOC]
        # Verifies (issue 9016) that bitwise and/or of a boolean Series (index b, c, a) with an empty object Series aligns to the union: and gives all false, or gives the original values, both sorted by index.
        # GH#9016: support bitwise op for integer types
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)
        s_empty = Series([], dtype=object)

        res = s_tft & s_empty
        expected = s_fff.sort_index()
        tm.assert_series_equal(res, expected)

        res = s_tft | s_empty
        expected = s_tft.sort_index()
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_int_dtype(self):
        # [FUNCTION DOC]
        # Verifies (issue 9016) bitwise and/or between integer Series: same-dtype operations give the expected int64 results; and with an int8 operand gives int64; or between int16 and int32 gives int32.
        # GH#9016: support bitwise op for integer types

        s_0123 = Series(range(4), dtype="int64")
        s_3333 = Series([3] * 4)
        s_4444 = Series([4] * 4)

        res = s_0123 & s_3333
        expected = Series(range(4), dtype="int64")
        tm.assert_series_equal(res, expected)

        res = s_0123 | s_4444
        expected = Series(range(4, 8), dtype="int64")
        tm.assert_series_equal(res, expected)

        s_1111 = Series([1] * 4, dtype="int8")
        res = s_0123 & s_1111
        expected = Series([0, 1, 0, 1], dtype="int64")
        tm.assert_series_equal(res, expected)

        res = s_0123.astype(np.int16) | s_1111.astype(np.int32)
        expected = Series([1, 1, 3, 3], dtype="int32")
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_int_scalar(self):
        # [FUNCTION DOC]
        # Verifies (issue 9016) bitwise and of an int64 Series with integer scalars zero and one gives elementwise integer results.
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        res = s_0123 & 0
        expected = Series([0] * 4)
        tm.assert_series_equal(res, expected)

        res = s_0123 & 1
        expected = Series([0, 1, 0, 1])
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_float(self):
        # [FUNCTION DOC]
        # Verifies (issue 9016) that bitwise and of an integer Series with a float operand raises a type error: NaN or a float scalar gives the dtyped-array-and-scalar message; a list of floats gives the dtype-less-sequence message; a float array or float Series gives an unsupported-operand message.
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        err_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.nan
        with pytest.raises(TypeError, match=msg):
            s_0123 & 3.14
        msg = "unsupported operand type.+for &:"
        with pytest.raises(TypeError, match=err_msg):
            s_0123 & [0.1, 4, 3.14, 2]
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.array([0.1, 4, 3.14, 2])
        with pytest.raises(TypeError, match=msg):
            s_0123 & Series([0.1, 4, -3.14, 2])

    def test_logical_operators_int_dtype_with_str(self):
        # [FUNCTION DOC]
        # Verifies that bitwise and of an int8 Series with a string scalar raises a type error about performing the operation with a dtyped array and scalar of that type, and with a list of strings raises the dtype-less-sequence error.
        s_1111 = Series([1] * 4, dtype="int8")

        err_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        msg = "Cannot perform 'and_' with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s_1111 & "a"
        with pytest.raises(TypeError, match=err_msg):
            s_1111 & ["a", "b", "c", "d"]

    def test_logical_operators_int_dtype_with_bool(self):
        # [FUNCTION DOC]
        # Verifies (issue 9016) that and of an integer Series with scalar false gives all false; lists and tuples of booleans raise the dtype-less-sequence type error; xor with false gives true for non-zero positions and false at zero.
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        expected = Series([False] * 4)

        result = s_0123 & False
        tm.assert_series_equal(result, expected)

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )
        with pytest.raises(TypeError, match=msg):
            s_0123 & [False]

        with pytest.raises(TypeError, match=msg):
            s_0123 & (False,)

        result = s_0123 ^ False
        expected = Series([False, True, True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_operators_int_dtype_with_object(self):
        # [FUNCTION DOC]
        # Verifies (issue 9016) that and of an integer Series with a Series of false and NaN gives all false, while and with a Series of strings and NaN raises a type error (unsupported operand integer and string, or the reflected and operation being unsupported).
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        result = s_0123 & Series([False, np.nan, False, False])
        expected = Series([False] * 4)
        tm.assert_series_equal(result, expected)

        s_abNd = Series(["a", "b", np.nan, "d"])
        with pytest.raises(
            TypeError, match="unsupported.* 'int' and 'str'|'rand_' not supported"
        ):
            s_0123 & s_abNd

    def test_logical_operators_bool_dtype_with_int(self):
        # [FUNCTION DOC]
        # Verifies that and of a boolean Series with integer scalars zero and one gives all false and the original values respectively.
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)

        res = s_tft & 0
        expected = s_fff
        tm.assert_series_equal(res, expected)

        res = s_tft & 1
        expected = s_tft
        tm.assert_series_equal(res, expected)

    def test_logical_ops_bool_dtype_with_ndarray(self):
        # [FUNCTION DOC]
        # Verifies that and, or and xor of a boolean Series with list data containing None and NaN raise the dtype-less-sequence error for a list, while a NumPy array, an Index and a Series of the same values each give the same expected result (missing treated as false for and and xor expectations as listed, true where left is true for or).
        # make sure we operate on ndarray the same as Series
        left = Series([True, True, True, False, True])
        right = [True, False, None, True, np.nan]

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        expected = Series([True, False, False, False, False])
        with pytest.raises(TypeError, match=msg):
            left & right
        result = left & np.array(right)
        tm.assert_series_equal(result, expected)
        result = left & Index(right)
        tm.assert_series_equal(result, expected)
        result = left & Series(right)
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, True, True, True])
        with pytest.raises(TypeError, match=msg):
            left | right
        result = left | np.array(right)
        tm.assert_series_equal(result, expected)
        result = left | Index(right)
        tm.assert_series_equal(result, expected)
        result = left | Series(right)
        tm.assert_series_equal(result, expected)

        expected = Series([False, True, True, True, True])
        with pytest.raises(TypeError, match=msg):
            left ^ right
        result = left ^ np.array(right)
        tm.assert_series_equal(result, expected)
        result = left ^ Index(right)
        tm.assert_series_equal(result, expected)
        result = left ^ Series(right)
        tm.assert_series_equal(result, expected)

    def test_logical_operators_int_dtype_with_bool_dtype_and_reindex(self):
        # [FUNCTION DOC]
        # Verifies (issues 9016, 52538) alignment of a boolean Series with an integer Series with different indexes: the boolean-on-the-left operation produces an all-false boolean result over the union index; the integer-on-the-left operation raises a type error about unsupported operand types float and bool (the integers are reindexed to float and no longer cast to object). A single-element Series on a subset of labels gives the reindexed results for and and or.
        # KNOWN ISSUE: the first boolean Series is created twice with identical arguments, so the second assignment is redundant.
        # GH#9016: support bitwise op for integer types

        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_tft = Series([True, False, True], index=index)
        s_tff = Series([True, False, False], index=index)

        s_0123 = Series(range(4), dtype="int64")

        # s_0123 will be all false now because of reindexing like s_tft
        expected = Series([False] * 7, index=[0, 1, 2, 3, "a", "b", "c"])
        result = s_tft & s_0123
        tm.assert_series_equal(result, expected)

        # GH#52538: no longer to object type when reindex is needed;
        # matches DataFrame behavior
        msg = r"unsupported operand type\(s\) for &: 'float' and 'bool'"
        with pytest.raises(TypeError, match=msg):
            s_0123 & s_tft

        s_a0b1c0 = Series([1], list("b"))

        res = s_tft & s_a0b1c0
        expected = s_tff.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

        res = s_tft | s_a0b1c0
        expected = s_tft.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

    def test_scalar_na_logical_ops_corners(self):
        # [FUNCTION DOC]
        # Verifies logical-op corner cases with missing and datetime elements: and with a datetime scalar on an integer Series raises a type error about a dtyped array and scalar. For an object Series with every other value missing, and with a list of its own values raises the dtype-less-sequence error.
        # KNOWN ISSUE: an expected Series is built (true except false at every other position) but never used in any assertion.
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, 10])

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s & datetime(2005, 1, 1)

        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan

        expected = Series(True, index=s.index)
        expected[::2] = False

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )
        with pytest.raises(TypeError, match=msg):
            s & list(s)

    def test_scalar_na_logical_ops_corners_aligns(self):
        # [FUNCTION DOC]
        # Verifies that and between an object Series (with NaN and a datetime) and a one-column frame, in either operand order, returns an all-false frame whose columns are the union of the frame column name and the Series index positions.
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan
        d = DataFrame({"A": s})

        expected = DataFrame(False, index=range(9), columns=["A", *list(range(9))])

        result = s & d
        tm.assert_frame_equal(result, expected)

        result = d & s
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
    def test_logical_ops_with_index(self, op):
        # [FUNCTION DOC]
        # Verifies (issues 22092, 19792) that and, or and xor (parametrized) between a boolean Series and a boolean Index or an integer Index (zeros and ones) work elementwise and return a Series: the boolean index case equals elementwise scalar operations; the integer index case is cast to boolean dtype.
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

        result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)

        result = op(ser, idx2)
        tm.assert_series_equal(result, expected)

    def test_reversed_xor_with_index_returns_series(self):
        # [FUNCTION DOC]
        # Verifies that xor with the Index on the left and the Series on the right (boolean and integer index cases) returns a Series with elementwise xor values rather than a set operation, as it was before version 2.0.
        # GH#22092, GH#19792 pre-2.0 these were aliased to setops
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False], dtype=bool)
        idx2 = Index([1, 0, 1, 0])

        expected = Series([False, True, True, False])
        result = idx1 ^ ser
        tm.assert_series_equal(result, expected)

        result = idx2 ^ ser
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op",
        [
            ops.rand_,
            ops.ror_,
        ],
    )
    def test_reversed_logical_op_with_index_returns_series(self, op):
        # [FUNCTION DOC]
        # Verifies the reflected and and or helper functions from the ops module (parametrized) with a boolean Series and Index: for a boolean Index the result equals the helper on raw values wrapped in a Series; for an integer Index it equals the helper applied to the Series and a Series built from the index.
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        expected = Series(op(idx1.values, ser.values))
        result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        expected = op(ser, Series(idx2))
        result = op(ser, idx2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (ops.rand_, [False, False]),
            (ops.ror_, [True, True]),
            (ops.rxor, [True, True]),
        ],
    )
    def test_reverse_ops_with_index(self, op, expected):
        # [FUNCTION DOC]
        # Verifies the reflected and, or and xor helper functions (parametrized with the expected two-element list) applied to a boolean Series and a boolean Index with distinct values, avoiding duplicates because multi-set Index ops are unreliable (pull request 23628, issue 49503): the result is a Series equal to the expected list.
        # https://github.com/pandas-dev/pandas/pull/23628
        # multi-set Index ops are buggy, so let's avoid duplicates...
        # GH#49503
        ser = Series([True, False])
        idx = Index([False, True])

        result = op(ser, idx)
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    def test_logical_ops_label_based(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies (issue 4947) that logical operators are label based: and/or/xor of Series with permuted indexes align by label; a larger right operand gives the union index with absent left labels false; combining with an empty Series or a non-matching single label fills with false while or keeps true values; selection by the or-ed mask returns the original for several right-hand identity cases (empty, non-matching, all-NaN on either index). With a string-valued right operand: when the infer-string option fixture is on, selection raises a type error (not supported for dtype or unsupported operand); otherwise it equals the original selection.
        # Scalar sections: or with true, one, two gives all true; or with false/zero and and with true/one keep the original; and with false/zero gives all false; NaN or the string foo raise the dtyped-array-and-scalar type error (the string only checked for or; NaN checked for and as well).
        # KNOWN ISSUE: a comment says the infer-string behavior needs a decision (a referenced TODO), so the type error expected there may be temporary.
        # GH#4947
        # logical ops should be label based

        a = Series([True, False, True], list("bca"))
        b = Series([False, True, False], list("abc"))

        expected = Series([False, True, False], list("abc"))
        result = a & b
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, False], list("abc"))
        result = a | b
        tm.assert_series_equal(result, expected)

        expected = Series([True, False, False], list("abc"))
        result = a ^ b
        tm.assert_series_equal(result, expected)

        # rhs is bigger
        a = Series([True, False, True], list("bca"))
        b = Series([False, True, False, True], list("abcd"))

        expected = Series([False, True, False, False], list("abcd"))
        result = a & b
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, False, False], list("abcd"))
        result = a | b
        tm.assert_series_equal(result, expected)

        # filling

        # vs empty
        empty = Series([], dtype=object)

        result = a & empty
        expected = Series([False, False, False], list("abc"))
        tm.assert_series_equal(result, expected)

        result = a | empty
        expected = Series([True, True, False], list("abc"))
        tm.assert_series_equal(result, expected)

        # vs non-matching
        result = a & Series([1], ["z"])
        expected = Series([False, False, False, False], list("abcz"))
        tm.assert_series_equal(result, expected)

        result = a | Series([1], ["z"])
        expected = Series([True, True, False, False], list("abcz"))
        tm.assert_series_equal(result, expected)

        # identity
        # we would like s[s|e] == s to hold for any e, whether empty or not
        for e in [
            empty.copy(),
            Series([1], ["z"]),
            Series(np.nan, b.index),
            Series(np.nan, a.index),
        ]:
            result = a[a | e]
            tm.assert_series_equal(result, a[a])

        for e in [Series(["z"])]:
            if using_infer_string:
                # TODO(infer_string) should this behave differently?
                # -> https://github.com/pandas-dev/pandas/issues/60234
                with pytest.raises(
                    TypeError, match="not supported for dtype|unsupported operand type"
                ):
                    result = a[a | e]
            else:
                result = a[a | e]
            tm.assert_series_equal(result, a[a])

        # vs scalars
        index = list("bca")
        t = Series([True, False, True])

        for v in [True, 1, 2]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, True, True], index=index)
            tm.assert_series_equal(result, expected)

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        for v in [np.nan, "foo"]:
            with pytest.raises(TypeError, match=msg):
                t | v

        for v in [False, 0]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, False, True], index=index)
            tm.assert_series_equal(result, expected)

        for v in [True, 1]:
            result = Series([True, False, True], index=index) & v
            expected = Series([True, False, True], index=index)
            tm.assert_series_equal(result, expected)

        for v in [False, 0]:
            result = Series([True, False, True], index=index) & v
            expected = Series([False, False, False], index=index)
            tm.assert_series_equal(result, expected)
        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        for v in [np.nan]:
            with pytest.raises(TypeError, match=msg):
                t & v

    def test_logical_ops_df_compat(self):
        # [FUNCTION DOC]
        # Verifies (issue 1134) that Series logical ops fill missing alignment positions with false, while the equivalent frame operations do not: with two same-name boolean Series on partially overlapping indexes, and gives false outside the intersection in either order; or gives true where either side is true and false where the only value present is false; the frame versions agree with the Series versions in the expected frames used. A second pair with differing lengths is tested similarly.
        # KNOWN ISSUE: the first frame expectation is rebound to a new frame of true, true, NaN, NaN that is never used, since the following assertions use other expectations; and the comments about NaN or-ing are copied between the two blocks in a way that does not match the data in the second block.
        # GH#1134
        s1 = Series([True, False, True], index=list("ABC"), name="x")
        s2 = Series([True, True, False], index=list("ABD"), name="x")

        exp = Series([True, False, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 & s2, exp)
        tm.assert_series_equal(s2 & s1, exp)

        # True | np.nan => True
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 | s2, exp_or1)
        # np.nan | True => np.nan, filled with False
        exp_or = Series([True, True, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s2 | s1, exp_or)

        # DataFrame doesn't fill nan with False
        tm.assert_frame_equal(s1.to_frame() & s2.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s2.to_frame() & s1.to_frame(), exp.to_frame())

        exp = DataFrame({"x": [True, True, np.nan, np.nan]}, index=list("ABCD"))
        tm.assert_frame_equal(s1.to_frame() | s2.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s2.to_frame() | s1.to_frame(), exp_or.to_frame())

        # different length
        s3 = Series([True, False, True], index=list("ABC"), name="x")
        s4 = Series([True, True, True, True], index=list("ABCD"), name="x")

        exp = Series([True, False, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 & s4, exp)
        tm.assert_series_equal(s4 & s3, exp)

        # np.nan | True => np.nan, filled with False
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 | s4, exp_or1)
        # True | np.nan => True
        exp_or = Series([True, True, True, True], index=list("ABCD"), name="x")
        tm.assert_series_equal(s4 | s3, exp_or)

        tm.assert_frame_equal(s3.to_frame() & s4.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s4.to_frame() & s3.to_frame(), exp.to_frame())

        tm.assert_frame_equal(s3.to_frame() | s4.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s4.to_frame() | s3.to_frame(), exp_or.to_frame())

    def test_int_dtype_different_index_not_bool(self):
        # [FUNCTION DOC]
        # Verifies (issue 52500) that the NumPy bitwise xor function and the xor operator on two integer Series with the same labels in different order align by label and give integer results, not boolean, indexed in the first Series' order.
        # GH 52500
        ser1 = Series([1, 2, 3], index=[10, 11, 23], name="a")
        ser2 = Series([10, 20, 30], index=[11, 10, 23], name="a")
        result = np.bitwise_xor(ser1, ser2)
        expected = Series([21, 8, 29], index=[10, 11, 23], name="a")
        tm.assert_series_equal(result, expected)

        result = ser1 ^ ser2
        tm.assert_series_equal(result, expected)
