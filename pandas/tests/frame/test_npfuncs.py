"""
Tests for np.foo applied to DataFrame, not necessarily ufuncs.
"""
# [MODULE DOC]
# Tests for NumPy functions (array conversion, square root, sum, ravel) applied
# to the tabular container, whether or not they are true ufuncs. Uses fixtures
# from the test-suite conftest; no import-time side effects.

import numpy as np

from pandas import (
    Categorical,
    DataFrame,
)
import pandas._testing as tm


class TestAsArray:
    # [CLASS DOC]
    # Test group for converting or reducing frames through NumPy functions.
    def test_asarray_homogeneous(self):
        # [FUNCTION DOC]
        # Converting a frame of two categorical columns to an array yields an object
        # dtype array of the category values (noted in-test as possibly changing in the
        # future).
        df = DataFrame({"A": Categorical([1, 2]), "B": Categorical([1, 2])})
        result = np.asarray(df)
        # may change from object in the future
        expected = np.array([[1, 1], [2, 2]], dtype="object")
        tm.assert_numpy_array_equal(result, expected)

    def test_np_sqrt(self, float_frame):
        # [FUNCTION DOC]
        # Applying the NumPy square root to the float fixture frame (with floating
        # warnings suppressed) returns a frame of the same type whose index and columns
        # are the identical objects as the input, and equals applying the function
        # column-wise.
        with np.errstate(all="ignore"):
            result = np.sqrt(float_frame)
        assert isinstance(result, type(float_frame))
        assert result.index.is_(float_frame.index)
        assert result.columns.is_(float_frame.columns)

        tm.assert_frame_equal(result, float_frame.apply(np.sqrt))

    def test_sum_axis_behavior(self):
        # [FUNCTION DOC]
        # Applying the NumPy sum to a random 4x3 frame must reduce over both axes and
        # equal the sum of the underlying array with no axis (GH#52042).
        # GH#52042 df.sum(axis=None) now reduces over both axes, which gets
        #  called when we do np.sum(df)

        arr = np.random.default_rng(2).standard_normal((4, 3))
        df = DataFrame(arr)

        res = np.sum(df)
        expected = df.to_numpy().sum(axis=None)
        assert res == expected

    def test_np_ravel(self):
        # [FUNCTION DOC]
        # GH26247: flattening with the NumPy ravel works for a list of single-row frames,
        # a single-row frame with named columns, and a list of single-row named-column
        # frames; each time the result is a flat float array of the row values in order.
        # GH26247
        arr = np.array(
            [
                [0.11197053, 0.44361564, -0.92589452],
                [0.05883648, -0.00948922, -0.26469934],
            ]
        )

        result = np.ravel([DataFrame(batch.reshape(1, 3)) for batch in arr])
        expected = np.array(
            [
                0.11197053,
                0.44361564,
                -0.92589452,
                0.05883648,
                -0.00948922,
                -0.26469934,
            ]
        )
        tm.assert_numpy_array_equal(result, expected)

        result = np.ravel(DataFrame(arr[0].reshape(1, 3), columns=["x1", "x2", "x3"]))
        expected = np.array([0.11197053, 0.44361564, -0.92589452])
        tm.assert_numpy_array_equal(result, expected)

        result = np.ravel(
            [
                DataFrame(batch.reshape(1, 3), columns=["x1", "x2", "x3"])
                for batch in arr
            ]
        )
        expected = np.array(
            [
                0.11197053,
                0.44361564,
                -0.92589452,
                0.05883648,
                -0.00948922,
                -0.26469934,
            ]
        )
        tm.assert_numpy_array_equal(result, expected)
