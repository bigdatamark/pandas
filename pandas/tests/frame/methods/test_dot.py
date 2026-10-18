# [MODULE DOC]
# Tests for matrix multiplication of series and frames with frames, series or arrays
# (dot). A shared abstract test base defines the common tests and two concrete
# subclasses supply the fixtures for a series left operand and a frame left operand; two
# module-level tests check arrow and nullable extension dtype results. Uses the testing
# helpers module. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class DotSharedTests:
    # [CLASS DOC]
    # HELPER: abstract base class of the dot tests (not collected on its own as it has
    # no test-class name). Subclasses must provide three fixtures (the left operand, a
    # right-hand frame indexed so the product is valid, and the expected product) and a
    # class method asserting equality of results with one fewer dimension. Contains the
    # shared test methods: values equivalence, array arguments, series argument, index
    # alignment, shape mismatch and misaligned errors.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: abstract fixture for the left operand; subclasses override. The base
        # version raises not-implemented.
        raise NotImplementedError

    @pytest.fixture
    def other(self) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: abstract fixture for the right-hand frame, indexed so that the left
        # operand dot it is valid. The base version raises not-implemented.
        """
        other is a DataFrame that is indexed so that obj.dot(other) is valid
        """
        raise NotImplementedError

    @pytest.fixture
    def expected(self, obj, other) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: abstract fixture for the expected product of left operand and
        # right-hand frame. Parameters: the two fixtures above. Base version raises
        # not-implemented.
        """
        The expected result of obj.dot(other)
        """
        raise NotImplementedError

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        # [FUNCTION DOC]
        # HELPER: abstract class method asserting equality of results that have one
        # fewer dimension than the left operand. Parameters: result, expected. Base
        # version raises not-implemented.
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        raise NotImplementedError

    def test_dot_equiv_values_dot(self, obj, other, expected):
        # [FUNCTION DOC]
        # The product with the frame equals the expected object (built from the
        # raw-array product).
        # `expected` is constructed from obj.values.dot(other.values)
        result = obj.dot(other)
        tm.assert_equal(result, expected)

    def test_dot_2d_ndarray(self, obj, other, expected):
        # [FUNCTION DOC]
        # Passing the raw values of the right-hand frame gives matching values; labels
        # may differ so only values are compared.
        # Check ndarray argument; in this case we get matching values,
        #  but index/columns may not match
        result = obj.dot(other.values)
        assert np.all(result == expected.values)

    def test_dot_1d_ndarray(self, obj, expected):
        # [FUNCTION DOC]
        # Passing a one-dimensional array equal to a row (or to the series itself) gives
        # the same as passing that row as a series, compared with the reduced-dimension
        # assertion.
        # can pass correct-length array
        row = obj.iloc[0] if obj.ndim == 2 else obj

        result = obj.dot(row.values)
        expected = obj.dot(row)
        self.reduced_dim_assert(result, expected)

    def test_dot_series(self, obj, other, expected):
        # [FUNCTION DOC]
        # Product with one column of the right-hand frame, as a series, equals the
        # matching expected column.
        # Check series argument
        result = obj.dot(other["1"])
        self.reduced_dim_assert(result, expected["1"])

    def test_dot_series_alignment(self, obj, other, expected):
        # [FUNCTION DOC]
        # A series argument whose index is in reversed order is aligned before
        # multiplying, giving the same expected column.
        result = obj.dot(other.iloc[::-1]["1"])
        self.reduced_dim_assert(result, expected["1"])

    def test_dot_aligns(self, obj, other, expected):
        # [FUNCTION DOC]
        # Reversing the right-hand frame's rows does not change the result since rows
        # are aligned on labels.
        # Check index alignment
        other2 = other.iloc[::-1]
        result = obj.dot(other2)
        tm.assert_equal(result, expected)

    def test_dot_shape_mismatch(self, obj):
        # [FUNCTION DOC]
        # An array whose length does not match raises an error whose message says the
        # dot product shapes mismatch.
        msg = "Dot product shape mismatch"
        # exception raised is of type Exception
        with pytest.raises(Exception, match=msg):
            obj.dot(obj.values[:3])

    def test_dot_misaligned(self, obj, other):
        # [FUNCTION DOC]
        # Passing the transposed right-hand frame (labels not matching) raises a value
        # error saying the matrices are not aligned.
        msg = "matrices are not aligned"
        with pytest.raises(ValueError, match=msg):
            obj.dot(other.T)


class TestSeriesDot(DotSharedTests):
    # [CLASS DOC]
    # Concrete dot tests where the left operand is a series of four labeled floats.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a random series of length four with index p, q, r,
        # s.
        return Series(
            np.random.default_rng(2).standard_normal(4), index=["p", "q", "r", "s"]
        )

    @pytest.fixture
    def other(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a random frame with rows p, q, r, s and columns 1,
        # 2, 3 (a transposed 3 by 4 random frame).
        return DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["1", "2", "3"],
            columns=["p", "q", "r", "s"],
        ).T

    @pytest.fixture
    def expected(self, obj, other):
        # [FUNCTION DOC]
        # HELPER: fixture returning the series of the raw array product with the other
        # frame columns as index.
        return Series(np.dot(obj.values, other.values), index=other.columns)

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        # [FUNCTION DOC]
        # HELPER: for series left operand, results with fewer dimensions are scalars;
        # compares them approximately.
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        tm.assert_almost_equal(result, expected)


class TestDataFrameDot(DotSharedTests):
    # [CLASS DOC]
    # Concrete dot tests where the left operand is a 3 by 4 random frame.
    @pytest.fixture
    def obj(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a random 3 by 4 frame with rows a, b, c and columns
        # p, q, r, s.
        return DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["a", "b", "c"],
            columns=["p", "q", "r", "s"],
        )

    @pytest.fixture
    def other(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a random 4 by 2 frame with rows p, q, r, s and
        # columns 1, 2.
        return DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            index=["p", "q", "r", "s"],
            columns=["1", "2"],
        )

    @pytest.fixture
    def expected(self, obj, other):
        # [FUNCTION DOC]
        # HELPER: fixture returning the frame of the raw array product with the left
        # operand index and the right-hand frame columns.
        return DataFrame(
            np.dot(obj.values, other.values), index=obj.index, columns=other.columns
        )

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        # [FUNCTION DOC]
        # HELPER: for frame left operand, results with fewer dimensions are series;
        # compares them ignoring names and asserts the result has no name.
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name is None


@pytest.mark.parametrize(
    "dtype,exp_dtype",
    [("Float32", "Float64"), ("Int16", "Int32"), ("float[pyarrow]", "double[pyarrow]")],
)
def test_arrow_dtype(dtype, exp_dtype):
    # [FUNCTION DOC]
    # Parametrized over three (right-hand dtype, expected dtype) pairs: nullable
    # float32, nullable int16, arrow float. Requires pyarrow. Multiplying an int32 frame
    # by an identity matrix of the given dtype yields the expected widened dtype for all
    # columns.
    pytest.importorskip("pyarrow")

    cols = ["a", "b"]
    df_a = DataFrame([[1, 2], [3, 4], [5, 6]], columns=cols, dtype="int32")
    df_b = DataFrame([[1, 0], [0, 1]], index=cols, dtype=dtype)
    result = df_a.dot(df_b)
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=exp_dtype)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype,exp_dtype",
    [("Float32", "Float64"), ("Int16", "Int32"), ("float[pyarrow]", "double[pyarrow]")],
)
def test_arrow_dtype_series(dtype, exp_dtype):
    # [FUNCTION DOC]
    # Same dtype pairs; a series of int32 times the frame of the given dtype yields a
    # series of the expected dtype.
    pytest.importorskip("pyarrow")

    cols = ["a", "b"]
    series_a = Series([1, 2], index=cols, dtype="int32")
    df_b = DataFrame([[1, 0], [0, 1]], index=cols, dtype=dtype)
    result = series_a.dot(df_b)
    expected = Series([1, 2], dtype=exp_dtype)

    tm.assert_series_equal(result, expected)
