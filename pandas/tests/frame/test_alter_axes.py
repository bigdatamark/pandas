# [MODULE DOC]
# Tests for assigning to the index and columns attributes of DataFrame directly
# (attribute assignment rather than set_axis or rename).

from datetime import (
    datetime,
    timezone,
)

from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameAlterAxes:
    # [CLASS DOC]
    # Test container for direct attribute assignment of the index and columns of a frame.
    # Tests for setting index/columns attributes directly (i.e. __setattr__)

    def test_set_axis_setattr_index(self):
        # [FUNCTION DOC]
        # Regression test (GH 6785). Assigns a timezone-aware datetime column to the index
        # attribute and removes the column; expects the result to equal the frame produced
        # by setting that column as index through the method.
        # GH 6785
        # set the index manually

        df = DataFrame([{"ts": datetime(2014, 4, 1, tzinfo=timezone.utc), "foo": 1}])
        expected = df.set_index("ts")
        df.index = df["ts"]
        df.pop("ts")
        tm.assert_frame_equal(df, expected)

    # Renaming

    def test_assign_columns(self, float_frame):
        # [FUNCTION DOC]
        # Adds a string column to the float fixture frame, copies it and assigns a new list of
        # column labels to the copy. Expects data in the original frame under the old labels
        # to match the copy under the new labels (names ignored), i.e. only labels change.
        float_frame["hi"] = "there"

        df = float_frame.copy()
        df.columns = ["foo", "bar", "baz", "quux", "foo2"]
        tm.assert_series_equal(float_frame["C"], df["baz"], check_names=False)
        tm.assert_series_equal(float_frame["hi"], df["foo2"], check_names=False)
