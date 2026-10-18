# [MODULE DOC]
# Tests for the private DataFrame method that splits a frame into a mapping from dtype
# name to a per-dtype sub-frame.

from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm


class TestToDictOfBlocks:
    # [CLASS DOC]
    # Test container for the private method that returns a mapping of dtype name to sub-frame.
    def test_no_copy_blocks(self, float_frame):
        # [FUNCTION DOC]
        # Regression test (GH 9607). Copies the float fixture frame, obtains the block mapping,
        # and adds one to the first column inside each sub-frame that contains it.
        # Expects the last sub-frame's column to then differ from the original frame's column,
        # confirming that mutation of a block does not write through to the source frame.
        # GH#9607
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]

        _last_df = None
        # use the copy=False, change a column
        blocks = df._to_dict_of_blocks()
        for _df in blocks.values():
            _last_df = _df
            if column in _df:
                _df.loc[:, column] = _df[column] + 1
        assert _last_df is not None and not _last_df[column].equals(df[column])


def test_set_change_dtype_slice():
    # [FUNCTION DOC]
    # Regression test (GH 8850). Builds a frame with a three-entry two-level column index
    # holding mixed float and integer data, then multiplies one column by a float so that
    # it changes dtype. Expects the block mapping keys to be exactly the float and integer
    # dtype names, the float entry to hold the two float columns with the expected values
    # and the integer entry to hold the remaining integer column.
    # GH#8850
    cols = MultiIndex.from_tuples([("1st", "a"), ("2nd", "b"), ("3rd", "c")])
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    df["2nd"] = df["2nd"] * 2.0

    blocks = df._to_dict_of_blocks()
    assert sorted(blocks.keys()) == ["float64", "int64"]
    tm.assert_frame_equal(
        blocks["float64"], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2])
    )
    tm.assert_frame_equal(blocks["int64"], DataFrame([[3], [6]], columns=cols[2:]))
