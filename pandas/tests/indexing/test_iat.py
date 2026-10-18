# [MODULE DOC]
# Tests for the scalar positional accessor (the iat attribute) on frames and series.
import numpy as np

from pandas import (
    DataFrame,
    Series,
    period_range,
)


def test_iat(float_frame):
    # [FUNCTION DOC]
    # For every row and column position of the shared float frame fixture, the positional scalar
    # equals the label-based scalar for the corresponding labels.
    for i, row in enumerate(float_frame.index):
        for j, col in enumerate(float_frame.columns):
            result = float_frame.iat[i, j]
            expected = float_frame.at[row, col]
            assert result == expected


def test_iat_duplicate_columns():
    # https://github.com/pandas-dev/pandas/issues/11754
    # [FUNCTION DOC]
    # Issue 11754. Positional scalar access works on a frame with duplicate column labels and
    # returns the first value.
    df = DataFrame([[1, 2]], columns=["x", "x"])
    assert df.iat[0, 0] == 1


def test_iat_getitem_series_with_period_index():
    # GH#4390, iat incorrectly indexing
    # [FUNCTION DOC]
    # GH 4390. Positional scalar access on a series with a period index returns the same value as
    # label-based getitem with the first label.
    index = period_range("1/1/2001", periods=10)
    ser = Series(np.random.default_rng(2).standard_normal(10), index=index)
    expected = ser[index[0]]
    result = ser.iat[0]
    assert expected == result
