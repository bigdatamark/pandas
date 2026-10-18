# [MODULE DOC]
# Tests for the anti-join modes of the merge function (left-anti keeps rows of the
# left frame whose key has no match on the right; right-anti the converse). In both
# cases the result carries the columns of both frames with missing values for the
# side without matches, and the original row labels of the retained side are kept.
# Covers index-based and column-based keys, non-unique keys, NaN keys,
# tz-aware datetimes, multi-level indexes, extension dtypes, empty frames and mixed
# dtypes. Imports merge from the reshape implementation module.
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm
from pandas.core.reshape.merge import merge


def test_merge_antijoin():
    # [FUNCTION DOC]
    # Verifies left-anti and right-anti merges on the indexes of two frames: the left
    # result holds the row only present on the left, the right result the row only on
    # the right (GH 42916).
    # GH#42916
    left = DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])
    right = DataFrame({"B": [1, 2, 4]}, index=["a", "b", "d"])

    result = merge(left, right, how="left_anti", left_index=True, right_index=True)
    expected = DataFrame({"A": [3], "B": [np.nan]}, index=["c"])
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_index=True, right_index=True)
    expected = DataFrame({"A": [np.nan], "B": [4]}, index=["d"])
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_on_different_columns():
    # [FUNCTION DOC]
    # Verifies anti-joins with different key column names on each side (object dtype
    # keys): result has all four columns, with the unmatched side's columns missing and
    # the original row label retained.
    left = DataFrame({"A": [1.0, 2.0, 3.0], "B": ["a", "b", "c"]}).astype({"B": object})
    right = DataFrame({"C": [1.0, 2.0, 4.0], "D": ["a", "d", "b"]}).astype(
        {"D": object}
    )

    result = merge(left, right, how="left_anti", left_on="B", right_on="D")
    expected = DataFrame(
        {
            "A": [3.0],
            "B": ["c"],
            "C": [np.nan],
            "D": [np.nan],
        },
        index=[2],
    ).astype({"B": object, "D": object})
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_on="B", right_on="D")
    expected = DataFrame(
        {
            "A": [np.nan],
            "B": [np.nan],
            "C": [2.0],
            "D": ["d"],
        },
        index=[1],
    ).astype({"B": object, "D": object})
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_nonunique_keys():
    # [FUNCTION DOC]
    # Verifies anti-joins when keys repeat: all left rows whose key never appears on
    # the right are returned; for right-anti every right row without match (both
    # duplicates), with their original labels.
    left = DataFrame({"A": [1.0, 2.0, 3.0], "B": ["a", "b", "b"]}).astype({"B": object})
    right = DataFrame({"C": [1.0, 2.0, 4.0], "D": ["b", "d", "d"]}).astype(
        {"D": object}
    )

    result = merge(left, right, how="left_anti", left_on="B", right_on="D")
    expected = DataFrame(
        {
            "A": [1.0],
            "B": ["a"],
            "C": [np.nan],
            "D": [np.nan],
        },
        index=[0],
    ).astype({"B": object, "D": object})
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_on="B", right_on="D")
    expected = DataFrame(
        {
            "A": [np.nan, np.nan],
            "B": [np.nan, np.nan],
            "C": [2.0, 4.0],
            "D": ["d", "d"],
        },
        index=[2, 3],
    ).astype({"B": object, "D": object})
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_same_df():
    # [FUNCTION DOC]
    # Verifies that left-anti merging a frame with itself on the index gives an empty
    # frame with suffixed columns of the same integer dtype (index type not checked).
    left = DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"], dtype=np.int64)
    result = merge(left, left, how="left_anti", left_index=True, right_index=True)
    expected = DataFrame([], columns=["A_x", "A_y"], dtype=np.int64)
    tm.assert_frame_equal(result, expected, check_index_type=False)


def test_merge_antijoin_nans():
    # [FUNCTION DOC]
    # Verifies that NaN keys match each other in the join, so the NaN-keyed left row
    # is excluded from the left-anti result.
    left = DataFrame({"A": [1.0, 2.0, np.nan], "C": ["a", "b", "c"]}).astype(
        {"C": object}
    )
    right = DataFrame({"A": [3.0, 2.0, np.nan], "D": ["d", "e", "f"]}).astype(
        {"D": object}
    )
    result = merge(left, right, how="left_anti", on="A")
    expected = DataFrame({"A": [1.0], "C": ["a"], "D": [np.nan]}).astype(
        {"C": object, "D": object}
    )
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_on_datetime64tz():
    # [FUNCTION DOC]
    # Verifies left-anti and right-anti merges on a tz-aware datetime key column with
    # overlapping values: columns with the same name get the default suffixes and the
    # datetime dtype is kept (GH 11405).
    # GH11405
    left = DataFrame(
        {
            "key": pd.date_range("20151010", periods=2, tz="US/Eastern"),
            "value": [1.0, 2.0],
        }
    )
    right = DataFrame(
        {
            "key": pd.date_range("20151011", periods=3, tz="US/Eastern"),
            "value": [1.0, 2.0, 3.0],
        }
    )

    expected = DataFrame(
        {
            "key": pd.date_range("20151010", periods=1, tz="US/Eastern"),
            "value_x": [1.0],
            "value_y": [np.nan],
        },
        index=[0],
    )
    result = merge(left, right, on="key", how="left_anti")
    tm.assert_frame_equal(result, expected)

    expected = DataFrame(
        {
            "key": pd.date_range("20151012", periods=2, tz="US/Eastern"),
            "value_x": [np.nan, np.nan],
            "value_y": [2.0, 3.0],
        },
        index=[1, 2],
    )
    result = merge(left, right, on="key", how="right_anti")
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_multiindex():
    # [FUNCTION DOC]
    # Verifies anti-joins on identically named two-level indexes: the result keeps the
    # multi-level index (with names) of the retained rows.
    left = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        },
        index=MultiIndex.from_tuples(
            [("a", "x"), ("b", "y"), ("c", "z")], names=["first", "second"]
        ),
    )
    right = DataFrame(
        {
            "C": [7, 8, 9],
            "D": [10, 11, 12],
        },
        index=MultiIndex.from_tuples(
            [("a", "x"), ("b", "y"), ("c", "w")], names=["first", "second"]
        ),
    )

    result = merge(left, right, how="left_anti", left_index=True, right_index=True)
    expected = DataFrame(
        {
            "A": [3],
            "B": [6],
            "C": [np.nan],
            "D": [np.nan],
        },
        index=MultiIndex.from_tuples([("c", "z")], names=["first", "second"]),
    )
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_index=True, right_index=True)
    expected = DataFrame(
        {
            "A": [np.nan],
            "B": [np.nan],
            "C": [9],
            "D": [12],
        },
        index=MultiIndex.from_tuples([("c", "w")], names=["first", "second"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("timestamp[s][pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ],
)
def test_merge_antijoin_extension_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over nullable integer and (if pyarrow is installed) pyarrow integer,
    # timestamp and string dtypes for the key column. Verifies left-anti merge on that
    # column returns the unmatched left rows with the extension dtype kept for the key
    # and missing values on the right-side column.
    left = DataFrame(
        {
            "join_col": [1, 3, 5],
            "left_val": [1, 2, 3],
        }
    )
    right = DataFrame(
        {
            "join_col": [2, 3, 4],
            "right_val": [1, 2, 3],
        }
    )
    left = left.astype({"join_col": dtype})
    right = right.astype({"join_col": dtype})
    result = merge(left, right, how="left_anti", on="join_col")
    expected = DataFrame(
        {
            "join_col": [1, 5],
            "left_val": [1, 3],
            "right_val": [np.nan, np.nan],
        },
        index=[0, 2],
    )
    expected = expected.astype({"join_col": dtype})
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_empty_dataframe():
    # [FUNCTION DOC]
    # Verifies that anti-joins of two empty frames return an empty frame with all four
    # columns, for both directions.
    left = DataFrame({"A": [], "B": []})
    right = DataFrame({"C": [], "D": []})

    result = merge(left, right, how="left_anti", left_on="A", right_on="C")
    expected = DataFrame({"A": [], "B": [], "C": [], "D": []})
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_on="A", right_on="C")
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_no_common_elements():
    # [FUNCTION DOC]
    # Verifies that when no keys match, left-anti returns all left rows (right columns
    # missing) and right-anti returns all right rows (left columns missing).
    left = DataFrame({"A": [1, 2, 3]})
    right = DataFrame({"B": [4, 5, 6]})

    result = merge(left, right, how="left_anti", left_on="A", right_on="B")
    expected = DataFrame({"A": [1, 2, 3], "B": [np.nan, np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_on="A", right_on="B")
    expected = DataFrame({"A": [np.nan, np.nan, np.nan], "B": [4, 5, 6]})
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_with_null_values():
    # [FUNCTION DOC]
    # Verifies that missing keys in the two frames match each other (so are excluded);
    # left-anti returns the other unmatched left rows, right-anti the unmatched right
    # row, with original labels retained.
    left = DataFrame({"A": [1.0, 2.0, None, 4.0]})
    right = DataFrame({"B": [2.0, None, 5.0]})

    result = merge(left, right, how="left_anti", left_on="A", right_on="B")
    expected = DataFrame({"A": [1.0, 4.0], "B": [np.nan, np.nan]}, index=[0, 3])
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_on="A", right_on="B")
    expected = DataFrame({"A": [np.nan], "B": [5.0]}, index=[2])
    tm.assert_frame_equal(result, expected)


def test_merge_antijoin_with_mixed_dtypes():
    # [FUNCTION DOC]
    # Verifies anti-joins on object columns containing mixed integers, strings and
    # floats compare by value across types: results are object-typed with the expected
    # unmatched rows.
    left = DataFrame({"A": [1, "2", 3.0]})
    right = DataFrame({"B": ["2", 3.0, 4]})

    result = merge(left, right, how="left_anti", left_on="A", right_on="B")
    expected = DataFrame({"A": [1], "B": [np.nan]}, dtype=object)
    tm.assert_frame_equal(result, expected)

    result = merge(left, right, how="right_anti", left_on="A", right_on="B")
    expected = DataFrame({"A": [np.nan], "B": [4]}, dtype=object, index=[2])
    tm.assert_frame_equal(result, expected)
