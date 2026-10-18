# [MODULE DOC]
# Tests for reductions (sum, prod, min, max, mean, median, var, std, skew, kurt, any, all) on nullable integer data in
# Series, DataFrame, groupby and mixed-dtype frames, checking the result types (integer, Float64 or boolean). No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    array,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", np.int64(3)],
        ["prod", np.int64(2)],
        ["min", np.int64(1)],
        ["max", np.int64(2)],
        ["mean", np.float64(1.5)],
        ["median", np.float64(1.5)],
        ["var", np.float64(0.5)],
        ["std", np.float64(0.5**0.5)],
        ["skew", pd.NA],
        ["kurt", pd.NA],
        ["any", True],
        ["all", True],
    ],
)
def test_series_reductions(op, expected):
    # [FUNCTION DOC]
    # Parametrized over reductions with expected scalars. Reducing a two-element Int64 Series gives numpy integers for integer ops, floats for mean-like ones, NA for skew and kurtosis, and true for any and all.
    ser = Series([1, 2], dtype="Int64")
    result = getattr(ser, op)()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", Series([3], index=["a"], dtype="Int64")],
        ["prod", Series([2], index=["a"], dtype="Int64")],
        ["min", Series([1], index=["a"], dtype="Int64")],
        ["max", Series([2], index=["a"], dtype="Int64")],
        ["mean", Series([1.5], index=["a"], dtype="Float64")],
        ["median", Series([1.5], index=["a"], dtype="Float64")],
        ["var", Series([0.5], index=["a"], dtype="Float64")],
        ["std", Series([0.5**0.5], index=["a"], dtype="Float64")],
        ["skew", Series([pd.NA], index=["a"], dtype="Float64")],
        ["kurt", Series([pd.NA], index=["a"], dtype="Float64")],
        ["any", Series([True], index=["a"], dtype="boolean")],
        ["all", Series([True], index=["a"], dtype="boolean")],
    ],
)
def test_dataframe_reductions(op, expected):
    # [FUNCTION DOC]
    # Parametrized over reductions with expected Series. Reducing a one-column Int64 frame gives a Series indexed by column name with Int64, Float64 or boolean dtype as appropriate.
    df = DataFrame({"a": array([1, 2], dtype="Int64")})
    result = getattr(df, op)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", array([1, 3], dtype="Int64")],
        ["prod", array([1, 3], dtype="Int64")],
        ["min", array([1, 3], dtype="Int64")],
        ["max", array([1, 3], dtype="Int64")],
        ["mean", array([1, 3], dtype="Float64")],
        ["median", array([1, 3], dtype="Float64")],
        ["var", array([pd.NA], dtype="Float64")],
        ["std", array([pd.NA], dtype="Float64")],
        ["skew", array([pd.NA], dtype="Float64")],
        ["any", array([True, True], dtype="boolean")],
        ["all", array([True, True], dtype="boolean")],
    ],
)
def test_groupby_reductions(op, expected):
    # [FUNCTION DOC]
    # Parametrized over reductions with expected arrays. Group-wise reduction of an Int64 column with a missing value gives a frame with the expected dtype per operation (Float64 for mean, median, variance and the like; boolean for any and all).
    df = DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": array([1, None, 3], dtype="Int64"),
        }
    )
    result = getattr(df.groupby("A"), op)()
    expected = DataFrame(expected, index=pd.Index(["a", "b"], name="A"), columns=["B"])

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", Series([4, 4], index=["B", "C"], dtype="Float64")],
        ["prod", Series([3, 3], index=["B", "C"], dtype="Float64")],
        ["min", Series([1, 1], index=["B", "C"], dtype="Float64")],
        ["max", Series([3, 3], index=["B", "C"], dtype="Float64")],
        ["mean", Series([2, 2], index=["B", "C"], dtype="Float64")],
        ["median", Series([2, 2], index=["B", "C"], dtype="Float64")],
        ["var", Series([2, 2], index=["B", "C"], dtype="Float64")],
        ["std", Series([2**0.5, 2**0.5], index=["B", "C"], dtype="Float64")],
        ["skew", Series([np.nan, pd.NA], index=["B", "C"], dtype="Float64")],
        ["kurt", Series([np.nan, pd.NA], index=["B", "C"], dtype="Float64")],
        ["any", Series([True, True, True], index=["A", "B", "C"], dtype="boolean")],
        ["all", Series([True, True, True], index=["A", "B", "C"], dtype="boolean")],
    ],
)
def test_mixed_reductions(op, expected):
    # [FUNCTION DOC]
    # Parametrized over reductions with expected Series. A frame with string, plain float and Int64 columns: the Series reduction equals the expected value for column C, and the frame reduction
    # (restricted to numeric columns except for any and all) gives a Float64 Series for the numeric columns.
    df = DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": array([1, None, 3], dtype="Int64"),
        }
    )

    # series
    result = getattr(df.C, op)()
    tm.assert_equal(result, expected["C"])

    # frame
    if op in ["any", "all"]:
        result = getattr(df, op)()
    else:
        result = getattr(df, op)(numeric_only=True)
    tm.assert_series_equal(result, expected)
