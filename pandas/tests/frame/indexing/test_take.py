# [MODULE DOC]
# Tests for the positional row/column selection method of the frame class (take).
# Verifies slice rejection, result equivalence to reindexing with taken labels, negative
# indices, out-of-bounds errors and behavior on homogeneous, mixed-object and mixed
# numeric frames. Relies on shared frame fixtures from the frame test conftest and the
# testing helpers module. No import-time side effects.

import pytest

import pandas._testing as tm


class TestDataFrameTake:
    # [CLASS DOC]
    # Group of tests for the frame take method; stateless pytest class.
    def test_take_slices_not_supported(self, float_frame):
        # [FUNCTION DOC]
        # Passing a slice object as the indices for either axis must raise a type error
        # whose message mentions slices.
        # GH#51539
        df = float_frame

        slc = slice(0, 4, 1)
        with pytest.raises(TypeError, match="slice"):
            df.take(slc, axis=0)
        with pytest.raises(TypeError, match="slice"):
            df.take(slc, axis=1)

    def test_take(self, float_frame):
        # [FUNCTION DOC]
        # Uses the float frame. Taking row positions must equal reindexing with the
        # index taken at those positions; taking column positions must equal selecting
        # the permuted column labels. Repeats with negative positions. Out-of-range
        # positions (positive or negative) on either axis must raise an index error
        # stating indices are out-of-bounds.
        # homogeneous
        order = [3, 1, 2, 0]
        for df in [float_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["D", "B", "C", "A"]]
            tm.assert_frame_equal(result, expected, check_names=False)

        # negative indices
        order = [2, 1, -1]
        for df in [float_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            result = df.take(order, axis=0)
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["C", "B", "D"]]
            tm.assert_frame_equal(result, expected, check_names=False)

        # illegal indices
        msg = "indices are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, 30], axis=0)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, -31], axis=0)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, 5], axis=1)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, -5], axis=1)

    def test_take_mixed_type(self, float_string_frame):
        # [FUNCTION DOC]
        # Same checks as the basic take test on the float-plus-string frame: row take
        # equals reindex, column take equals label selection, including negative
        # positions.
        # mixed-dtype
        order = [4, 1, 2, 0, 3]
        for df in [float_string_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["foo", "B", "C", "A", "D"]]
            tm.assert_frame_equal(result, expected)

        # negative indices
        order = [4, 1, -2]
        for df in [float_string_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["foo", "B", "D"]]
            tm.assert_frame_equal(result, expected)

    def test_take_mixed_numeric(self, mixed_float_frame, mixed_int_frame):
        # [FUNCTION DOC]
        # Runs take on rows and columns for the mixed-float and mixed-int frames and
        # compares to reindex and label selection respectively.
        # by dtype
        order = [1, 2, 0, 3]
        for df in [mixed_float_frame, mixed_int_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["B", "C", "A", "D"]]
            tm.assert_frame_equal(result, expected)
