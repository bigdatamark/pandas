# [MODULE DOC]
# Tests for DataFrame reindex_like, including the deprecated method argument and subclass handling.

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameReindexLike:
    # [CLASS DOC]
    # Groups the reindex_like tests.
    def test_reindex_like(self, float_frame):
        # [FUNCTION DOC]
        # Verifies reindexing the float_frame like a subset frame (first ten rows, two reordered columns) equals that subset.
        other = float_frame.reindex(index=float_frame.index[:10], columns=["C", "B"])

        tm.assert_frame_equal(other, float_frame.reindex_like(other))

    @pytest.mark.parametrize(
        "method,expected_values",
        [
            ("nearest", [0, 1, 1, 2]),
            ("pad", [np.nan, 0, 1, 1]),
            ("backfill", [0, 1, 2, 2]),
        ],
    )
    def test_reindex_like_methods(self, method, expected_values):
        # [FUNCTION DOC]
        # Parametrized over method names (nearest, pad, backfill) and an expected-values list.
        # Reindexes a frame like itself with the method and a zero tolerance (scalar and list), expecting a Pandas4Warning (deprecation) each time and the unchanged frame as result.
        # KNOWN ISSUE: the expected_values parameter is never used by the test body.
        df = DataFrame({"x": list(range(5))})

        with tm.assert_produces_warning(Pandas4Warning):
            result = df.reindex_like(df, method=method, tolerance=0)
        tm.assert_frame_equal(df, result)
        with tm.assert_produces_warning(Pandas4Warning):
            result = df.reindex_like(df, method=method, tolerance=[0, 0, 0, 0])
        tm.assert_frame_equal(df, result)

    def test_reindex_like_subclass(self):
        # [FUNCTION DOC]
        # Issue 31925. Defines a trivial DataFrame subclass inside the test. Verifies reindex_like on an empty subclass instance with an empty frame works and compares equal to an empty frame.
        # https://github.com/pandas-dev/pandas/issues/31925
        class MyDataFrame(DataFrame):
            # [CLASS DOC]
            # HELPER: local empty DataFrame subclass used only by the subclass reindex_like test.
            pass

        expected = DataFrame()
        df = MyDataFrame()
        result = df.reindex_like(expected)

        tm.assert_frame_equal(result, expected)
