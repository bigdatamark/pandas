# [MODULE DOC]
# Tests for the pipe method of groupby objects (frame and series): chaining with plain functions and passing arguments.
import numpy as np

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


def test_pipe():
    # [FUNCTION DOC]
    # Issue 17871: pipe on a DataFrameGroupBy followed by pipe on the resulting series; checks the numeric result of the max minus the minimum of minima, squared.
    # Test the pipe method of DataFrameGroupBy.
    # Issue #17871

    random_state = np.random.default_rng(2)

    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": random_state.standard_normal(8),
            "C": random_state.standard_normal(8),
        }
    )

    def f(dfgb):
        # [FUNCTION DOC]
        # HELPER: pipe function taking a groupby and returning the maximum of one column minus the overall minimum of another.
        return dfgb.B.max() - dfgb.C.min().min()

    def square(srs):
        # [FUNCTION DOC]
        # HELPER: pipe function squaring a series.
        return srs**2

    # Note that the transformations are
    # GroupBy -> Series
    # Series -> Series
    # This then chains the GroupBy.pipe and the
    # NDFrame.pipe methods
    result = df.groupby("A").pipe(f).pipe(square)

    index = Index(["bar", "foo"], name="A")
    expected = pd.Series([3.749306591013693, 6.717707873081384], name="B", index=index)

    tm.assert_series_equal(expected, result)


def test_pipe_args():
    # [FUNCTION DOC]
    # Issue 17871: pipe with positional arguments through a chain of three functions (filtering by group mean returning a new groupby, normalizing sums and adding a constant, then combining columns), plus pipe on a SeriesGroupBy with a lambda.
    # Test passing args to the pipe method of DataFrameGroupBy.
    # Issue #17871

    df = DataFrame(
        {
            "group": ["A", "A", "B", "B", "C"],
            "x": [1.0, 2.0, 3.0, 2.0, 5.0],
            "y": [10.0, 100.0, 1000.0, -100.0, -1000.0],
        }
    )

    def f(dfgb, arg1):
        # [FUNCTION DOC]
        # HELPER: pipe function that filters groups by mean of one column exceeding an argument (keeping the rows as missing) and regroups by the group column.
        filtered = dfgb.filter(lambda grp: grp.y.mean() > arg1, dropna=False)
        return filtered.groupby("group")

    def g(dfgb, arg2):
        # [FUNCTION DOC]
        # HELPER: pipe function dividing group sums by the total sum and adding an argument.
        return dfgb.sum() / dfgb.sum().sum() + arg2

    def h(df, arg3):
        # [FUNCTION DOC]
        # HELPER: pipe function on a frame returning the sum of two columns minus an argument.
        return df.x + df.y - arg3

    result = df.groupby("group").pipe(f, 0).pipe(g, 10).pipe(h, 100)

    # Assert the results here
    index = Index(["A", "B"], name="group")
    expected = pd.Series([-79.5160891089, -78.4839108911], index=index)

    tm.assert_series_equal(result, expected)

    # test SeriesGroupby.pipe
    ser = pd.Series([1, 1, 2, 2, 3, 3])
    result = ser.groupby(ser).pipe(lambda grp: grp.sum() * grp.count())

    expected = pd.Series([4, 8, 12], index=Index([1, 2, 3], dtype=np.int64))

    tm.assert_series_equal(result, expected)
