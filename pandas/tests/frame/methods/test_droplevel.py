# [MODULE DOC]
# Tests for removing a level from a multi-level index or columns (droplevel) on frames
# and series. Uses the frame-or-series fixture and the testing helpers module. No
# import-time side effects.

import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm


class TestDropLevel:
    # [CLASS DOC]
    # Stateless pytest class of droplevel tests.
    def test_droplevel(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Builds a frame with two-level rows and columns
        # (reduced to one column for series). Dropping a row level equals resetting that
        # level with drop; for frames, dropping a column level keeps the other level as
        # a named column index; for series, dropping on the columns axis raises a value
        # error since only one axis exists.
        # GH#20342
        cols = MultiIndex.from_tuples(
            [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        )
        mi = MultiIndex.from_tuples([(1, 2), (5, 6), (9, 10)], names=["a", "b"])
        df = DataFrame([[3, 4], [7, 8], [11, 12]], index=mi, columns=cols)
        if frame_or_series is not DataFrame:
            df = df.iloc[:, 0]

        # test that dropping of a level in index works
        expected = df.reset_index("a", drop=True)
        result = df.droplevel("a", axis="index")
        tm.assert_equal(result, expected)

        if frame_or_series is DataFrame:
            # test that dropping of a level in columns works
            expected = df.copy()
            expected.columns = Index(["c", "d"], name="level_1")
            result = df.droplevel("level_2", axis="columns")
            tm.assert_equal(result, expected)
        else:
            # test that droplevel raises ValueError on axis != 0
            with pytest.raises(ValueError, match="No axis named columns"):
                df.droplevel(1, axis="columns")
