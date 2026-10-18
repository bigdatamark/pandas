# [MODULE DOC]
# Tests for the frame equality method (equals): dtype and block layout differences,
# values with missing entries, index and column comparison, multi-level indexes and
# categorical dtype ordering. Uses the string inference fixture and the testing helpers
# module. No import-time side effects.

import numpy as np

from pandas import (
    Categorical,
    DataFrame,
    date_range,
)
import pandas._testing as tm


class TestEquals:
    # [CLASS DOC]
    # Stateless pytest class of equals tests.
    def test_dataframe_not_equal(self):
        # [FUNCTION DOC]
        # Frames with the same values but swapped column dtypes are not equal.
        # see GH#28839
        df1 = DataFrame({"a": [1, 2], "b": ["s", "d"]})
        df2 = DataFrame({"a": ["s", "d"], "b": [1, 2]})
        assert df1.equals(df2) is False

    def test_equals_different_blocks(self, using_infer_string):
        # [FUNCTION DOC]
        # Takes string inference fixture. A frame and a column-reordered copy with
        # different block layout compare equal both ways and in the testing helper;
        # unless string inference is on, the test asserts the first blocks have
        # different dtypes to confirm that block rearrangement happened.
        # GH#9330
        df0 = DataFrame({"A": ["x", "y"], "B": [1, 2], "C": ["w", "z"]})
        df1 = df0.reset_index()[["A", "B", "C"]]
        if not using_infer_string:
            # this assert verifies that the above operations have
            # induced a block rearrangement
            assert df0._mgr.blocks[0].dtype != df1._mgr.blocks[0].dtype

        # do the real tests
        tm.assert_frame_equal(df0, df1)
        assert df0.equals(df1)
        assert df1.equals(df0)

    def test_equals(self):
        # [FUNCTION DOC]
        # Builds a frame of floats, strings, datetimes, timedeltas and object booleans
        # with missing rows. A copy is equal (columnwise too) and it is not equal to a
        # non-frame object. Changing a dtype, index values, or column order makes frames
        # unequal. Equality also holds with a datetime index, with a multi-level index
        # including a column with missing labels, and differs when different levels are
        # appended.
        # Add object dtype column with nans
        index = np.random.default_rng(2).random(10)
        df1 = DataFrame(
            np.random.default_rng(2).random(10), index=index, columns=["floats"]
        )
        df1["text"] = "the sky is so blue. we could use more chocolate.".split()
        df1["start"] = date_range("2000-1-1", periods=10, freq="min")
        df1["end"] = date_range("2000-1-1", periods=10, freq="D")
        df1["diff"] = df1["end"] - df1["start"]
        # Explicitly cast to object, to avoid implicit cast when setting np.nan
        df1["bool"] = (np.arange(10) % 3 == 0).astype(object)
        df1.loc[::2] = np.nan
        df2 = df1.copy()
        assert df1["text"].equals(df2["text"])
        assert df1["start"].equals(df2["start"])
        assert df1["end"].equals(df2["end"])
        assert df1["diff"].equals(df2["diff"])
        assert df1["bool"].equals(df2["bool"])
        assert df1.equals(df2)
        assert not df1.equals(object)

        # different dtype
        different = df1.copy()
        different["floats"] = different["floats"].astype("float32")
        assert not df1.equals(different)

        # different index
        different_index = -index
        different = df2.set_index(different_index)
        assert not df1.equals(different)

        # different columns
        different = df2.copy()
        different.columns = df2.columns[::-1]
        assert not df1.equals(different)

        # DatetimeIndex
        index = date_range("2000-1-1", periods=10, freq="min")
        df1 = df1.set_index(index)
        df2 = df1.copy()
        assert df1.equals(df2)

        # MultiIndex
        df3 = df1.set_index(["text"], append=True)
        df2 = df1.set_index(["text"], append=True)
        assert df3.equals(df2)

        df2 = df1.set_index(["floats"], append=True)
        assert not df3.equals(df2)

        # NaN in index
        df3 = df1.set_index(["floats"], append=True)
        df2 = df1.set_index(["floats"], append=True)
        assert df3.equals(df2)

    def test_equals_categorical_categories_order(self):
        # [FUNCTION DOC]
        # Categoricals that differ only in the order of the categories are equal; with
        # an extra category they are not.
        cat1 = Categorical(["a", "b", "a"], categories=["a", "b"])
        cat2 = Categorical(["a", "b", "a"], categories=["b", "a"])
        df1 = DataFrame({"c": cat1})
        df2 = DataFrame({"c": cat2})

        assert df1.equals(df2)

        cat3 = Categorical(["a", "b", "a"], categories=["a", "b", "c"])
        df3 = DataFrame({"c": cat3})

        assert not df1.equals(df3)
