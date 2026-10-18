# [MODULE DOC]
# Tests for counting non-missing values per row or column of a frame (count). Uses the
# mixed string/float frame fixture and the testing helpers module. No import-time side
# effects.

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameCount:
    # [CLASS DOC]
    # Stateless pytest class of count tests.
    def test_count(self):
        # [FUNCTION DOC]
        # Corner cases: counting an empty frame along either axis returns a series; a
        # frame with only rows counts zero for each row; a frame with only columns
        # counts zero for each column; a frame with neither returns an empty int64
        # series.
        # corner case
        frame = DataFrame()
        ct1 = frame.count(1)
        assert isinstance(ct1, Series)

        ct2 = frame.count(0)
        assert isinstance(ct2, Series)

        # GH#423
        df = DataFrame(index=range(10))
        result = df.count(1)
        expected = Series(0, index=df.index)
        tm.assert_series_equal(result, expected)

        df = DataFrame(columns=range(10))
        result = df.count(0)
        expected = Series(0, index=df.columns)
        tm.assert_series_equal(result, expected)

        df = DataFrame()
        result = df.count()
        expected = Series(dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_count_objects(self, float_string_frame):
        # [FUNCTION DOC]
        # Builds two frames from the series dict of the mixed fixture and checks that
        # counts along both axes are equal.
        # KNOWN ISSUE: both frames are constructed from the same input in the same way,
        # so the comparison is effectively between identical objects and cannot fail for
        # a meaningful reason.
        dm = DataFrame(float_string_frame._series)
        df = DataFrame(float_string_frame._series)

        tm.assert_series_equal(dm.count(), df.count())
        tm.assert_series_equal(dm.count(1), df.count(1))
