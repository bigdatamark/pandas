# [MODULE DOC]
# Tests for DataFrame.to_period and Series.to_period (convert datetime index/columns to periods).

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    PeriodIndex,
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToPeriod:
    # [CLASS DOC]
    # Test container for to_period on DataFrame and Series.
    def test_to_period(self, frame_or_series):
        # [FUNCTION DOC]
        # Uses a daily-indexed frame (with an added string column) wrapped as frame or series
        # by the frame_or_series fixture. Default conversion must yield a daily period index
        # spanning the same dates; passing the monthly frequency yields the index converted to monthly frequency.
        K = 5

        dr = date_range("1/1/2000", "1/1/2001", freq="D")
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(dr), K)),
            index=dr,
            columns=["A", "B", "C", "D", "E"],
        )
        obj["mix"] = "a"
        obj = tm.get_obj(obj, frame_or_series)

        pts = obj.to_period()
        exp = obj.copy()
        exp.index = period_range("1/1/2000", "1/1/2001")
        tm.assert_equal(pts, exp)

        pts = obj.to_period("M")
        exp.index = exp.index.asfreq("M")
        tm.assert_equal(pts, exp)

    def test_to_period_without_freq(self, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 7606): a datetime index without a set frequency has its frequency inferred
        # (daily) on conversion. For frames also converts the columns when axis is 1, with the columns equal to
        # the same daily period index.
        # GH#7606 without freq
        idx = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"])
        exp_idx = PeriodIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], freq="D"
        )

        obj = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), index=idx, columns=idx
        )
        obj = tm.get_obj(obj, frame_or_series)
        expected = obj.copy()
        expected.index = exp_idx
        tm.assert_equal(obj.to_period(), expected)

        if frame_or_series is DataFrame:
            expected = obj.copy()
            expected.columns = exp_idx
            tm.assert_frame_equal(obj.to_period(axis=1), expected)

    def test_to_period_columns(self):
        # [FUNCTION DOC]
        # Transposes a daily-indexed frame and converts along the column axis. Default gives a daily
        # period column index; passing monthly frequency gives the monthly-converted columns.
        dr = date_range("1/1/2000", "1/1/2001")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)
        df["mix"] = "a"

        df = df.T
        pts = df.to_period(axis=1)
        exp = df.copy()
        exp.columns = period_range("1/1/2000", "1/1/2001")
        tm.assert_frame_equal(pts, exp)

        pts = df.to_period("M", axis=1)
        tm.assert_index_equal(pts.columns, exp.columns.asfreq("M"))

    def test_to_period_invalid_axis(self):
        # [FUNCTION DOC]
        # Passing axis 2 must raise a value error saying no axis named 2 exists for DataFrame.
        dr = date_range("1/1/2000", "1/1/2001")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)
        df["mix"] = "a"

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.to_period(axis=2)

    def test_to_period_raises(self, index, frame_or_series):
        # [FUNCTION DOC]
        # Regression test (GH 33327). Takes the shared index fixture (all index types), builds an empty
        # object-dtype series (or one-column frame). For any non-datetime index, conversion must
        # raise a type error saying the index type is unsupported. Datetime indexes are not checked.
        # https://github.com/pandas-dev/pandas/issues/33327
        obj = Series(index=index, dtype=object)
        if frame_or_series is DataFrame:
            obj = obj.to_frame()

        if not isinstance(index, DatetimeIndex):
            msg = f"unsupported Type {type(index).__name__}"
            with pytest.raises(TypeError, match=msg):
                obj.to_period()
