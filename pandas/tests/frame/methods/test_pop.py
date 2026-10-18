# [MODULE DOC]
# Tests for DataFrame pop (remove a column and return it).

import numpy as np

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestDataFramePop:
    # [CLASS DOC]
    # Groups the DataFrame pop tests.
    def test_pop(self, float_frame):
        # [FUNCTION DOC]
        # Using float_frame, verifies pop removes the column from the frame, keeps the columns axis name, and (gh-10912) that modifying the popped series in place does not alter the remaining frame or leave a stale cache.
        float_frame.columns.name = "baz"

        float_frame.pop("A")
        assert "A" not in float_frame

        float_frame["foo"] = "bar"
        float_frame.pop("foo")
        assert "foo" not in float_frame
        assert float_frame.columns.name == "baz"

        # gh-10912: inplace ops cause caching issue
        a = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"], index=["X", "Y"])
        b = a.pop("B")
        b += 1

        # original frame
        expected = DataFrame([[1, 3], [4, 6]], columns=["A", "C"], index=["X", "Y"])
        tm.assert_frame_equal(a, expected)

        # result
        expected = Series([2, 5], index=["X", "Y"], name="B") + 1
        tm.assert_series_equal(b, expected)

    def test_pop_non_unique_cols(self):
        # [FUNCTION DOC]
        # Verifies popping a label that appears twice in the columns returns a frame holding both columns and leaves only the other column.
        df = DataFrame({0: [0, 1], 1: [0, 1], 2: [4, 5]})
        df.columns = ["a", "b", "a"]

        res = df.pop("a")
        assert type(res) == DataFrame
        assert len(res) == 2
        assert len(df.columns) == 1
        assert "b" in df.columns
        assert "a" not in df.columns
        assert len(df.index) == 2

    def test_mixed_depth_pop(self):
        # [FUNCTION DOC]
        # Frame with three-level columns of mixed depth. Verifies popping by the top-level label alone returns the same series as popping by the full tuple (named by the first level), leaves identical remaining frames, and popping a label with sub-columns returns a frame of those columns.
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        df1 = df.copy()
        df2 = df.copy()
        result = df1.pop("a")
        expected = df2.pop(("a", "", ""))
        tm.assert_series_equal(expected, result, check_names=False)
        tm.assert_frame_equal(df1, df2)
        assert result.name == "a"

        expected = df1["top"]
        df1 = df1.drop(["top"], axis=1)
        result = df2.pop("top")
        tm.assert_frame_equal(expected, result)
        tm.assert_frame_equal(df1, df2)
