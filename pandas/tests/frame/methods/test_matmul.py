# [MODULE DOC]
# Tests for the matrix multiplication operator on DataFrame (GH#10259) in its different operand combinations and error messages.

import operator

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestMatMul:
    # [CLASS DOC]
    # Groups the DataFrame matrix multiplication tests.
    def test_matmul(self):
        # [FUNCTION DOC]
        # Using random frames with labeled axes, verifies the matmul operator: frame by frame gives a frame labeled by the left index and right columns; frame by series gives a series; array by frame gives a frame with default row index and right-hand columns; nested list by frame works through the reflected operator; mixed and uniform integer dtypes give the same values as a numpy dot product; when the inner labels do not align a value error mentioning alignment is raised.
        # matmul test is for GH#10259
        a = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["a", "b", "c"],
            columns=["p", "q", "r", "s"],
        )
        b = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            index=["p", "q", "r", "s"],
            columns=["one", "two"],
        )

        # DataFrame @ DataFrame
        result = operator.matmul(a, b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)

        # DataFrame @ Series
        result = operator.matmul(a, b.one)
        expected = Series(np.dot(a.values, b.one.values), index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        # np.array @ DataFrame
        result = operator.matmul(a.values, b)
        assert isinstance(result, DataFrame)
        assert result.columns.equals(b.columns)
        assert result.index.equals(Index(range(3)))
        expected = np.dot(a.values, b.values)
        tm.assert_almost_equal(result.values, expected)

        # nested list @ DataFrame (__rmatmul__)
        result = operator.matmul(a.values.tolist(), b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_almost_equal(result.values, expected.values)

        # mixed dtype DataFrame @ DataFrame
        a["q"] = a.q.round().astype(int)
        result = operator.matmul(a, b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)

        # different dtypes DataFrame @ DataFrame
        a = a.astype(int)
        result = operator.matmul(a, b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)

        # unaligned
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=[1, 2, 3],
            columns=range(4),
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=range(5),
            columns=[1, 2, 3],
        )

        with pytest.raises(ValueError, match="aligned"):
            operator.matmul(df, df2)

    def test_matmul_message_shapes(self):
        # [FUNCTION DOC]
        # GH#21581. Verifies the error raised for incompatible shapes between an array (or nested list) and a frame is a value error whose message reports the original shapes, not transposed ones.
        # GH#21581 exception message should reflect original shapes,
        #  not transposed shapes
        a = np.random.default_rng(2).random((10, 4))
        b = np.random.default_rng(2).random((5, 3))

        df = DataFrame(b)

        msg = r"shapes \(10, 4\) and \(5, 3\) not aligned"
        with pytest.raises(ValueError, match=msg):
            a @ df
        with pytest.raises(ValueError, match=msg):
            a.tolist() @ df
