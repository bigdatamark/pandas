# [MODULE DOC]
# Copy-on-write tests for top-level combining functions: concat, merge and
# join. Verifies that results lazily share memory with the inputs (columns,
# indexes) wherever no data had to be rewritten, that writing to the result or
# to an input triggers a copy only for the touched columns, and that inputs are
# never modified. Uses the get_array helper from the sibling util module.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    concat,
    merge,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_concat_frames():
    # [FUNCTION DOC]
    # Column-wise concatenation of two object-dtype frames: both columns share
    # memory with their inputs; writing to each result column copies only that
    # column; the inputs are unchanged.
    df = DataFrame({"b": ["a"] * 3}, dtype=object)
    df2 = DataFrame({"a": ["a"] * 3}, dtype=object)
    df_orig = df.copy()
    result = concat([df, df2], axis=1)

    assert np.shares_memory(get_array(result, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    result.iloc[0, 0] = "d"
    assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    result.iloc[0, 1] = "d"
    assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_concat_frames_updating_input():
    # [FUNCTION DOC]
    # After column-wise concatenation, writing to an input frame detaches only that
    # input's column from the result while the other stays shared; the result's
    # values are unchanged by the input mutations.
    df = DataFrame({"b": ["a"] * 3}, dtype=object)
    df2 = DataFrame({"a": ["a"] * 3}, dtype=object)
    result = concat([df, df2], axis=1)

    assert np.shares_memory(get_array(result, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    expected = result.copy()
    df.iloc[0, 0] = "d"
    assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))

    df2.iloc[0, 0] = "d"
    assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(result, expected)


def test_concat_series():
    # [FUNCTION DOC]
    # Column-wise concatenation of two Series shares memory with each; writing to a
    # result column copies just that column; the input Series stay unchanged.
    ser = Series([1, 2], name="a")
    ser2 = Series([3, 4], name="b")
    ser_orig = ser.copy()
    ser2_orig = ser2.copy()
    result = concat([ser, ser2], axis=1)

    assert np.shares_memory(get_array(result, "a"), ser.values)
    assert np.shares_memory(get_array(result, "b"), ser2.values)

    result.iloc[0, 0] = 100
    assert not np.shares_memory(get_array(result, "a"), ser.values)
    assert np.shares_memory(get_array(result, "b"), ser2.values)

    result.iloc[0, 1] = 1000
    assert not np.shares_memory(get_array(result, "b"), ser2.values)
    tm.assert_series_equal(ser, ser_orig)
    tm.assert_series_equal(ser2, ser2_orig)


def test_concat_frames_chained():
    # [FUNCTION DOC]
    # Nested column-wise concatenation of three frames shares memory with all
    # inputs; modifying an input afterwards does not change the result.
    df1 = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df2 = DataFrame({"c": [4, 5, 6]})
    df3 = DataFrame({"d": [4, 5, 6]})
    result = concat([concat([df1, df2], axis=1), df3], axis=1)
    expected = result.copy()

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "c"), get_array(df2, "c"))
    assert np.shares_memory(get_array(result, "d"), get_array(df3, "d"))

    df1.iloc[0, 0] = 100
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))

    tm.assert_frame_equal(result, expected)


def test_concat_series_chained():
    # [FUNCTION DOC]
    # Nested column-wise concatenation of three Series shares memory with all
    # inputs; modifying an input afterwards does not change the result.
    ser1 = Series([1, 2, 3], name="a")
    ser2 = Series([4, 5, 6], name="c")
    ser3 = Series([4, 5, 6], name="d")
    result = concat([concat([ser1, ser2], axis=1), ser3], axis=1)
    expected = result.copy()

    assert np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))
    assert np.shares_memory(get_array(result, "c"), get_array(ser2, "c"))
    assert np.shares_memory(get_array(result, "d"), get_array(ser3, "d"))

    ser1.iloc[0] = 100
    assert not np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))

    tm.assert_frame_equal(result, expected)


def test_concat_series_updating_input():
    # [FUNCTION DOC]
    # After concatenating two Series by columns, writing to each input in turn
    # detaches only that column from the result, and the result keeps its original
    # values.
    ser = Series([1, 2], name="a")
    ser2 = Series([3, 4], name="b")
    expected = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = concat([ser, ser2], axis=1)

    assert np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))

    ser.iloc[0] = 100
    assert not np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    tm.assert_frame_equal(result, expected)

    ser2.iloc[0] = 1000
    assert not np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    tm.assert_frame_equal(result, expected)


def test_concat_mixed_series_frame():
    # [FUNCTION DOC]
    # Column-wise concatenation of a frame and a Series shares memory with both;
    # modifying either input afterwards leaves the result unchanged.
    df = DataFrame({"a": [1, 2, 3], "c": 1})
    ser = Series([4, 5, 6], name="d")
    result = concat([df, ser], axis=1)
    expected = result.copy()

    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(result, "c"), get_array(df, "c"))
    assert np.shares_memory(get_array(result, "d"), get_array(ser, "d"))

    ser.iloc[0] = 100
    assert not np.shares_memory(get_array(result, "d"), get_array(ser, "d"))

    df.iloc[0, 0] = 100
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    tm.assert_frame_equal(result, expected)


def test_concat_copy_keyword():
    # [FUNCTION DOC]
    # Concatenating two frames of different dtypes by columns with default arguments
    # shares memory with both inputs (verifying the default no-copy behavior).
    df = DataFrame({"a": [1, 2]})
    df2 = DataFrame({"b": [1.5, 2.5]})

    result = concat([df, df2], axis=1)

    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(result, "b"))


@pytest.mark.parametrize(
    "func",
    [
        lambda df1, df2, **kwargs: df1.merge(df2, **kwargs),
        lambda df1, df2, **kwargs: merge(df1, df2, **kwargs),
    ],
)
def test_merge_on_key(func):
    # [FUNCTION DOC]
    # Parametrized over the frame merge method and the top-level merge function
    # (defined as inline lambdas). Merging two object-keyed frames on the key shares
    # the value columns with their inputs, the key with the left input only. Writes
    # to the result copy only the touched column; inputs are unchanged.
    df1 = DataFrame({"key": Series(["a", "b", "c"], dtype=object), "a": [1, 2, 3]})
    df2 = DataFrame({"key": Series(["a", "b", "c"], dtype=object), "b": [4, 5, 6]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = func(df1, df2, on="key")

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    assert np.shares_memory(get_array(result, "key"), get_array(df1, "key"))
    assert not np.shares_memory(get_array(result, "key"), get_array(df2, "key"))

    result.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 2] = 0
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


def test_merge_on_index():
    # [FUNCTION DOC]
    # Merging on both indexes shares both value columns with the inputs; writing to
    # each result column copies only that column; inputs are unchanged.
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame({"b": [4, 5, 6]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = merge(df1, df2, left_index=True, right_index=True)

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


@pytest.mark.parametrize(
    "func, how",
    [
        (lambda df1, df2, **kwargs: merge(df2, df1, on="key", **kwargs), "right"),
        (lambda df1, df2, **kwargs: merge(df1, df2, on="key", **kwargs), "left"),
    ],
)
def test_merge_on_key_enlarging_one(func, how):
    # [FUNCTION DOC]
    # Parametrized over a function and how (right, left; lambdas perform the merge
    # with swapped arguments). The smaller frame is enlarged by the merge, so its
    # column is a new copy with no references, while the larger frame's value column
    # is shared. The key column is shared with the left frame only for a left join.
    # Writing to the result leaves the inputs unchanged.
    df1 = DataFrame({"key": Series(["a", "b", "c"], dtype=object), "a": [1, 2, 3]})
    df2 = DataFrame({"key": Series(["a", "b"], dtype=object), "b": [4, 5]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = func(df1, df2, how=how)

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    assert df2._mgr._has_no_reference(1)
    assert df2._mgr._has_no_reference(0)
    assert np.shares_memory(get_array(result, "key"), get_array(df1, "key")) is (
        how == "left"
    )
    assert not np.shares_memory(get_array(result, "key"), get_array(df2, "key"))

    if how == "left":
        result.iloc[0, 1] = 0
    else:
        result.iloc[0, 2] = 0
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


def test_merge_copy_keyword():
    # [FUNCTION DOC]
    # Default merge on indexes of a frame with another of different dtype shares
    # memory for both value columns.
    df = DataFrame({"a": [1, 2]})
    df2 = DataFrame({"b": [3, 4.5]})

    result = df.merge(df2, left_index=True, right_index=True)

    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(result, "b"))


def test_merge_upcasting_no_copy():
    # [FUNCTION DOC]
    # Merging where the shared key column must be upcast (integer with object) in
    # either argument order: the non-key column of the left frame is still shared,
    # the key column is not, and the inputs are unchanged.
    left = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    left_copy = left.copy()
    right = DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]}, dtype=object)
    result = merge(left, right, on="a")
    assert np.shares_memory(get_array(result, "b"), get_array(left, "b"))
    assert not np.shares_memory(get_array(result, "a"), get_array(left, "a"))
    tm.assert_frame_equal(left, left_copy)

    result = merge(right, left, on="a")
    assert np.shares_memory(get_array(result, "b"), get_array(left, "b"))
    assert not np.shares_memory(get_array(result, "a"), get_array(left, "a"))
    tm.assert_frame_equal(left, left_copy)


def test_merge_indicator_no_deep_copy():
    # [FUNCTION DOC]
    # Merging with the indicator option adds a column without deep-copying: the
    # value columns of both inputs remain shared with the result.
    left = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    right = DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
    result = merge(left, right, on="a", indicator=True)
    assert np.shares_memory(get_array(result, "b"), get_array(left, "b"))
    assert np.shares_memory(get_array(result, "c"), get_array(right, "c"))


@pytest.mark.parametrize("dtype", [object, "str"])
def test_join_on_key(dtype):
    # [FUNCTION DOC]
    # Parametrized over index dtype (object, string). Joining on the key index name
    # shares both value columns, shares the result index with the left frame only;
    # writes to the result copy only the touched column; inputs unchanged.
    df_index = Index(["a", "b", "c"], name="key", dtype=dtype)

    df1 = DataFrame({"a": [1, 2, 3]}, index=df_index.copy(deep=True))
    df2 = DataFrame({"b": [4, 5, 6]}, index=df_index.copy(deep=True))

    df1_orig = df1.copy()
    df2_orig = df2.copy()

    result = df1.join(df2, on="key")

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    assert tm.shares_memory(get_array(result.index), get_array(df1.index))
    assert not np.shares_memory(get_array(result.index), get_array(df2.index))

    result.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    result.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


def test_join_multiple_dataframes_on_key():
    # [FUNCTION DOC]
    # Joining a frame with a list of frames: all value columns and the index (with
    # the first frame only) are shared with the inputs. Writes to each result column
    # in turn detach only that column; the inputs equal their saved copies.
    df_index = Index(["a", "b", "c"], name="key", dtype=object)

    df1 = DataFrame({"a": [1, 2, 3]}, index=df_index.copy(deep=True))
    dfs_list = [
        DataFrame({"b": [4, 5, 6]}, index=df_index.copy(deep=True)),
        DataFrame({"c": [7, 8, 9]}, index=df_index.copy(deep=True)),
    ]

    df1_orig = df1.copy()
    dfs_list_orig = [df.copy() for df in dfs_list]

    result = df1.join(dfs_list)

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
    assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))
    assert np.shares_memory(get_array(result.index), get_array(df1.index))
    assert not np.shares_memory(get_array(result.index), get_array(dfs_list[0].index))
    assert not np.shares_memory(get_array(result.index), get_array(dfs_list[1].index))

    result.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    assert np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
    assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    result.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
    assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    result.iloc[0, 2] = 0
    assert not np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    tm.assert_frame_equal(df1, df1_orig)
    for df, df_orig in zip(dfs_list, dfs_list_orig, strict=True):
        tm.assert_frame_equal(df, df_orig)
