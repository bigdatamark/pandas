# [MODULE DOC]
# Tests for the element-wise combination method of the frame class that applies a user
# function to aligned columns of two frames (combine). Covers datetime-like data,
# built-in binary functions with misaligned frames and non-unique columns. Uses the
# float frame fixture and the testing helpers module. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestCombine:
    # [CLASS DOC]
    # Stateless pytest class of combine tests.
    @pytest.mark.parametrize(
        "data",
        [
            pd.date_range("2000", periods=4),
            pd.date_range("2000", periods=4, tz="US/Central"),
            pd.period_range("2000", periods=4),
            pd.timedelta_range(0, periods=4),
        ],
    )
    def test_combine_datetlike_udf(self, data):
        # [FUNCTION DOC]
        # Parametrized over datetime, tz-aware datetime, period and timedelta data.
        # Combining a frame that has a missing entry with a full copy using a function
        # that returns the second argument must give the copy back (dtype preserved).
        # GH#23079
        df = pd.DataFrame({"A": data})
        other = df.copy()
        df.iloc[1, 0] = None

        def combiner(a, b):
            # [FUNCTION DOC]
            # HELPER: nested user function that ignores its first argument and returns
            # the second; used as the combine function.
            return b

        result = df.combine(other, combiner)
        tm.assert_frame_equal(result, other)

    def test_combine_generic(self, float_frame):
        # [FUNCTION DOC]
        # Combining the float frame with a smaller frame (fewer rows and columns) using
        # numpy addition in both orders: the column absent from the smaller frame is
        # entirely missing; for the overlapping block the result equals twice the
        # original values.
        df1 = float_frame
        df2 = float_frame.loc[float_frame.index[:-5], ["A", "B", "C"]]

        combined = df1.combine(df2, np.add)
        combined2 = df2.combine(df1, np.add)
        assert combined["D"].isna().all()
        assert combined2["D"].isna().all()

        chunk = combined.loc[combined.index[:-5], ["A", "B", "C"]]
        chunk2 = combined2.loc[combined2.index[:-5], ["A", "B", "C"]]

        exp = (
            float_frame.loc[float_frame.index[:-5], ["A", "B", "C"]].reindex_like(chunk)
            * 2
        )
        tm.assert_frame_equal(chunk, exp)
        tm.assert_frame_equal(chunk2, exp)

    def test_combine_nonunique_columns(self):
        # [FUNCTION DOC]
        # With duplicated column labels and a missing row in the first frame, combining
        # with a copy via a function returning the second argument gives the copy
        # converted to float.
        # GH#51340

        df = pd.DataFrame({"A": range(5), "B": range(5)})
        df.columns = ["A", "A"]

        other = df.copy()
        df.iloc[1, :] = None

        def combiner(a, b):
            # [FUNCTION DOC]
            # HELPER: nested user function that ignores its first argument and returns
            # the second; used as the combine function.
            return b

        result = df.combine(other, combiner)
        expected = other.astype("float64")
        tm.assert_frame_equal(result, expected)
