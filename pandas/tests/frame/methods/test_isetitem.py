# [MODULE DOC]
# Tests for DataFrame isetitem, which sets columns by integer position with a value (including extension-dtype frames).

import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameSetItem:
    # [CLASS DOC]
    # Groups the isetitem tests.
    def test_isetitem_ea_df(self):
        # [FUNCTION DOC]
        # GH#49922. Replaces two columns by positions with an Int64 frame. Verifies the frame is modified in place and the replaced columns keep the nullable integer dtype while the untouched column is unchanged.
        # GH#49922
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        rhs = DataFrame([[11, 12], [13, 14]], dtype="Int64")

        df.isetitem([0, 1], rhs)
        expected = DataFrame(
            {
                0: Series([11, 13], dtype="Int64"),
                1: Series([12, 14], dtype="Int64"),
                2: [3, 6],
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_isetitem_ea_df_scalar_indexer(self):
        # [FUNCTION DOC]
        # GH#49922. Sets a single column by a scalar position using a one-column Int64 frame. Verifies the column takes the nullable integer dtype and other columns are unchanged.
        # GH#49922
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        rhs = DataFrame([[11], [13]], dtype="Int64")

        df.isetitem(2, rhs)
        expected = DataFrame(
            {
                0: [1, 4],
                1: [2, 5],
                2: Series([11, 13], dtype="Int64"),
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_isetitem_dimension_mismatch(self):
        # [FUNCTION DOC]
        # GH#51701. Verifies isetitem raises a value error when the number of positions differs from the number of columns of the frame value; the message states the number of positions and columns (3 columns and 1 column cases).
        # GH#51701
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        value = df.copy()
        with pytest.raises(ValueError, match="Got 2 positions but value has 3 columns"):
            df.isetitem([1, 2], value)

        value = df.copy()
        with pytest.raises(ValueError, match="Got 2 positions but value has 1 columns"):
            df.isetitem([1, 2], value[["a"]])
