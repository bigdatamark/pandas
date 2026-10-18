# [MODULE DOC]
# Copy-on-write tests for assigning values into a DataFrame: setting a column
# from an array, Series, Index or frame, overwriting columns, broadcasting to
# multi-level columns and in-place arithmetic on a column. Verifies which sources are
# copied eagerly (arrays, indexes) versus shared lazily (Series, frames).

import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

# -----------------------------------------------------------------------------
# Copy/view behaviour for the values that are set in a DataFrame


def test_set_column_with_array():
    # [FUNCTION DOC]
    # Setting a numpy array as a new column copies the data, so later changes to
    # the array do not alter the frame.
    # Case: setting an array as a new column (df[col] = arr) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = np.array([1, 2, 3], dtype="int64")

    df["c"] = arr

    # the array data is copied
    assert not np.shares_memory(get_array(df, "c"), arr)
    # and thus modifying the array does not modify the DataFrame
    arr[0] = 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_series():
    # [FUNCTION DOC]
    # Setting a Series as a new column shares memory lazily; changing the Series
    # afterwards does not change the frame column.
    # Case: setting a series as a new column (df[col] = s) copies that data
    # (with delayed copy with CoW)
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = Series([1, 2, 3])

    df["c"] = ser

    assert np.shares_memory(get_array(df, "c"), get_array(ser))

    # and modifying the series does not modify the DataFrame
    ser.iloc[0] = 0
    assert ser.iloc[0] == 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_index():
    # [FUNCTION DOC]
    # Setting an Index (generic and range) as a new column copies the data: no
    # memory sharing with the Index values.
    # Case: setting an index as a new column (df[col] = idx) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    idx = Index([1, 2, 3])

    df["c"] = idx

    # the index data is copied
    assert not np.shares_memory(get_array(df, "c"), idx.values)

    idx = RangeIndex(1, 4)
    arr = idx.values

    df["d"] = idx

    assert not np.shares_memory(get_array(df, "d"), arr)


def test_set_columns_with_dataframe():
    # [FUNCTION DOC]
    # Setting a frame into several new columns via a list of labels shares memory
    # with the source; modifying the source afterwards does not change the target.
    # Case: setting a DataFrame as new columns copies that data
    # (with delayed copy with CoW)
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})

    df[["c", "d"]] = df2

    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    # and modifying the set DataFrame does not modify the original DataFrame
    df2.iloc[0, 0] = 0
    tm.assert_series_equal(df["c"], Series([7, 8, 9], name="c"))


def test_setitem_series_no_copy():
    # [FUNCTION DOC]
    # Adding a Series as a new column shares memory; writing to the frame leaves the
    # Series unchanged.
    # Case: setting a Series as column into a DataFrame can delay copying that data
    df = DataFrame({"a": [1, 2, 3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    # adding a new column
    df["b"] = rhs
    assert np.shares_memory(get_array(rhs), get_array(df, "b"))

    df.iloc[0, 1] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_no_copy_single_block():
    # [FUNCTION DOC]
    # Overwriting an existing single-block column with a Series shares memory; writing to
    # the frame leaves the Series unchanged.
    # Overwriting an existing column that is a single block
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    df["a"] = rhs
    assert np.shares_memory(get_array(rhs), get_array(df, "a"))

    df.iloc[0, 0] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_no_copy_split_block():
    # [FUNCTION DOC]
    # Overwriting a column that is part of a larger block with a Series shares
    # memory; writing to the frame leaves the Series unchanged.
    # Overwriting an existing column that is part of a larger block
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    df["b"] = rhs
    assert np.shares_memory(get_array(rhs), get_array(df, "b"))

    df.iloc[0, 1] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_column_midx_broadcasting():
    # [FUNCTION DOC]
    # Assigning a Series to a top-level key of multi-level columns repeats the data
    # across the sub-columns. The data is currently copied eagerly: no memory
    # sharing and no references.
    # Setting a Series to multiple columns will repeat the data
    # (currently copying the data eagerly)
    df = DataFrame(
        [[1, 2, 3], [3, 4, 5]],
        columns=MultiIndex.from_arrays([["a", "a", "b"], [1, 2, 3]]),
    )
    rhs = Series([10, 11])
    df["a"] = rhs
    assert not np.shares_memory(get_array(rhs), df._get_column_array(0))
    assert df._mgr._has_no_reference(0)


def test_set_column_with_inplace_operator():
    # [FUNCTION DOC]
    # In-place addition applied directly to a column item raises no warning. Doing
    # the same on a separately held Series from selecting the column is the case
    # that would warn, but nothing is asserted for it.
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # this should not raise any warning
    with tm.assert_produces_warning(None):
        df["a"] += 1

    # when it is not in a chain, then it should produce a warning
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = df["a"]
    ser += 1
