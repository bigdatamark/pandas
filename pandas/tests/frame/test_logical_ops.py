# [MODULE DOC]
# Tests for the element-wise logical operators (and, or, exclusive or, and the
# bitwise inversion) on the tabular container, including missing values, empty
# frames, boolean/integer mixes, unsupported dtypes, and categorical columns.
# Uses fixtures from the test-suite conftest; no import-time side effects.
import operator
import re

import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DataFrame,
    Interval,
    Series,
    isnull,
)
import pandas._testing as tm


class TestDataFrameLogicalOperators:
    # [CLASS DOC]
    # Test group for logical operators on tabular data. Stateless.
    # &, |, ^

    @pytest.mark.parametrize(
        "left, right, op, expected",
        [
            (
                [True, False, np.nan],
                [True, False, True],
                operator.and_,
                [True, False, False],
            ),
            (
                [True, False, True],
                [True, False, np.nan],
                operator.and_,
                [True, False, False],
            ),
            (
                [True, False, np.nan],
                [True, False, True],
                operator.or_,
                [True, False, False],
            ),
            (
                [True, False, True],
                [True, False, np.nan],
                operator.or_,
                [True, False, True],
            ),
        ],
    )
    def test_logical_operators_nans(self, left, right, op, expected, frame_or_series):
        # [FUNCTION DOC]
        # GH#13896: parametrized over (left, right, operator, expected) and over the
        # container kind (frame or Series). Logical operators on object-type boolean data
        # with a missing value in either operand give a boolean result where the missing
        # position is treated as false for and, and for or per the listed expectations.
        # GH#13896
        result = op(frame_or_series(left), frame_or_series(right))
        expected = frame_or_series(expected)

        tm.assert_equal(result, expected)

    def test_logical_ops_empty_frame(self):
        # [FUNCTION DOC]
        # GH#5808: logical and/or on empty-column frames. With no columns the result
        # equals the input frame; with differing indexes the result takes the union
        # index; and for a frame with a single all-missing column, and gives all false
        # values.
        # GH#5808
        # empty frames, non-mixed dtype
        df = DataFrame(index=[1])

        result = df & df
        tm.assert_frame_equal(result, df)

        result = df | df
        tm.assert_frame_equal(result, df)

        df2 = DataFrame(index=[1, 2])
        result = df & df2
        tm.assert_frame_equal(result, df2)

        dfa = DataFrame(index=[1], columns=["A"])

        result = dfa & dfa
        expected = DataFrame(False, index=[1], columns=["A"])
        tm.assert_frame_equal(result, expected)

    def test_logical_ops_bool_frame(self):
        # [FUNCTION DOC]
        # GH#5808: and/or of an all-true boolean frame with itself equals that frame.
        # GH#5808
        df1a_bool = DataFrame(True, index=[1], columns=["A"])

        result = df1a_bool & df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

        result = df1a_bool | df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

    def test_logical_ops_int_frame(self):
        # [FUNCTION DOC]
        # GH#5808: or of an integer frame and a boolean frame equals the boolean frame,
        # matching the same operation on the single columns as Series.
        # GH#5808
        df1a_int = DataFrame(1, index=[1], columns=["A"])
        df1a_bool = DataFrame(True, index=[1], columns=["A"])

        result = df1a_int | df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

        # Check that this matches Series behavior
        res_ser = df1a_int["A"] | df1a_bool["A"]
        tm.assert_series_equal(res_ser, df1a_bool["A"])

    def test_logical_ops_invalid(self, using_infer_string):
        # [FUNCTION DOC]
        # GH#5808: or between a float frame and a boolean frame raises a type error for
        # unsupported operands; likewise a string frame with a boolean frame, where the
        # message differs when the new string dtype with pyarrow storage is active.
        # GH#5808

        df1 = DataFrame(1.0, index=[1], columns=["A"])
        df2 = DataFrame(True, index=[1], columns=["A"])
        msg = re.escape("unsupported operand type(s) for |: 'float' and 'bool'")
        with pytest.raises(TypeError, match=msg):
            df1 | df2

        df1 = DataFrame("foo", index=[1], columns=["A"])
        df2 = DataFrame(True, index=[1], columns=["A"])
        if using_infer_string and df1["A"].dtype.storage == "pyarrow":
            msg = "operation 'or_' not supported for dtype 'str'"
        else:
            msg = re.escape("unsupported operand type(s) for |: 'str' and 'bool'")
        with pytest.raises(TypeError, match=msg):
            df1 | df2

    def test_logical_operators(self):
        # [FUNCTION DOC]
        # Compares frame and, or, exclusive-or against the operation on the raw value
        # arrays (result must be boolean dtype), and unary inversion likewise, for two
        # 5x5 boolean frames built from nested dictionaries.
        def _check_bin_op(op):
            # [FUNCTION DOC]
            # HELPER: for the logical operator test. Applies a binary operator to the
            # enclosing test's two frames, compares to the operator applied to the raw
            # values wrapped in a frame, and asserts boolean dtype.
            result = op(df1, df2)
            expected = DataFrame(
                op(df1.values, df2.values), index=df1.index, columns=df1.columns
            )
            assert result.values.dtype == np.bool_
            tm.assert_frame_equal(result, expected)

        def _check_unary_op(op):
            # [FUNCTION DOC]
            # HELPER: for the logical operator test. Applies a unary operator to the
            # enclosing test's first frame, compares to the operator applied to the raw
            # values, and asserts boolean dtype.
            result = op(df1)
            expected = DataFrame(op(df1.values), index=df1.index, columns=df1.columns)
            assert result.values.dtype == np.bool_
            tm.assert_frame_equal(result, expected)

        df1 = {
            "a": {"a": True, "b": False, "c": False, "d": True, "e": True},
            "b": {"a": False, "b": True, "c": False, "d": False, "e": False},
            "c": {"a": False, "b": False, "c": True, "d": False, "e": False},
            "d": {"a": True, "b": False, "c": False, "d": True, "e": True},
            "e": {"a": True, "b": False, "c": False, "d": True, "e": True},
        }

        df2 = {
            "a": {"a": True, "b": False, "c": True, "d": False, "e": False},
            "b": {"a": False, "b": True, "c": False, "d": False, "e": False},
            "c": {"a": True, "b": False, "c": True, "d": False, "e": False},
            "d": {"a": False, "b": False, "c": False, "d": True, "e": False},
            "e": {"a": False, "b": False, "c": False, "d": False, "e": True},
        }

        df1 = DataFrame(df1)
        df2 = DataFrame(df2)

        _check_bin_op(operator.and_)
        _check_bin_op(operator.or_)
        _check_bin_op(operator.xor)

        _check_unary_op(operator.inv)  # TODO: belongs elsewhere

    def test_logical_with_nas(self):
        # [FUNCTION DOC]
        # GH4947 / GH4604: or of a column with a missing value and a true column gives
        # a boolean Series (missing treated as false); after filling missing with false
        # the result is all true. The final check repeats the preceding one.
        d = DataFrame({"a": [np.nan, False], "b": [True, True]})

        # GH4947
        # bool comparisons should return bool
        result = d["a"] | d["b"]
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)

        # GH4604, automatic casting here
        result = d["a"].fillna(False) | d["b"]
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)
        result = d["a"].fillna(False) | d["b"]
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_ops_categorical_columns(self):
        # [FUNCTION DOC]
        # GH#38367: or of an all-false mask with the null mask of a frame whose columns
        # are a categorical index of intervals (with an extra unused category) works,
        # giving true where the data are missing and preserving the columns.
        # GH#38367
        intervals = [Interval(1, 2), Interval(3, 4)]
        data = DataFrame(
            [[1, np.nan], [2, np.nan]],
            columns=CategoricalIndex(
                intervals, categories=[*intervals, Interval(5, 6)]
            ),
        )
        mask = DataFrame(
            [[False, False], [False, False]], columns=data.columns, dtype=bool
        )
        result = mask | isnull(data)
        expected = DataFrame(
            [[False, True], [False, True]],
            columns=CategoricalIndex(
                intervals, categories=[*intervals, Interval(5, 6)]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_int_dtype_different_index_not_bool(self):
        # [FUNCTION DOC]
        # GH 52500: bitwise exclusive-or (both via the NumPy ufunc and the operator) on
        # integer frames with differently ordered indexes aligns the indexes and returns
        # integer values (not booleans) in the first frame's index order.
        # GH 52500
        df1 = DataFrame([1, 2, 3], index=[10, 11, 23], columns=["a"])
        df2 = DataFrame([10, 20, 30], index=[11, 10, 23], columns=["a"])
        result = np.bitwise_xor(df1, df2)
        expected = DataFrame([21, 8, 29], index=[10, 11, 23], columns=["a"])
        tm.assert_frame_equal(result, expected)

        result = df1 ^ df2
        tm.assert_frame_equal(result, expected)

    def test_different_dtypes_different_index_raises(self):
        # [FUNCTION DOC]
        # GH 52538: and of two single-column integer frames whose string indexes only
        # partially overlap (so alignment introduces missing values) raises a type error about unsupported
        # operands.
        # GH 52538
        df1 = DataFrame([1, 2], index=["a", "b"])
        df2 = DataFrame([3, 4], index=["b", "c"])
        with pytest.raises(TypeError, match="unsupported operand type"):
            df1 & df2
