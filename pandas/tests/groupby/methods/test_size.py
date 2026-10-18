# [MODULE DOC]
# Tests for the size method of DataFrameGroupBy and SeriesGroupBy: correctness against group lengths, sorting, empty and all-null groups, period and categorical keys, masked and string dtypes.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    PeriodIndex,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize("by", ["A", "B", ["A", "B"]])
def test_size(df, by):
    # [FUNCTION DOC]
    # Parametrized over grouping by column A, column B, or both. For every group key, the size result must equal the length of that group. Uses the df fixture.
    grouped = df.groupby(by=by)
    result = grouped.size()
    for key, group in grouped:
        assert result[key] == len(group)


@pytest.mark.parametrize("by", ["A", "B", ["A", "B"]])
def test_size_sort(sort, by):
    # [FUNCTION DOC]
    # Parametrized over the grouping specification and the sort fixture. On a random 1000 by 3 integer frame, the size result must equal the count obtained by applying a row-count function to column C per group, ignoring names.
    df = DataFrame(np.random.default_rng(2).choice(20, (1000, 3)), columns=list("ABC"))
    left = df.groupby(by=by, sort=sort).size()
    right = df.groupby(by=by, sort=sort)["C"].apply(lambda a: a.shape[0])
    tm.assert_series_equal(left, right, check_names=False)


def test_size_series_dataframe():
    # [FUNCTION DOC]
    # Refs GitHub issue 11699: grouping an empty two-column frame by A and taking size must give an empty int64 Series whose index is an empty index named A.
    # https://github.com/pandas-dev/pandas/issues/11699
    df = DataFrame(columns=["A", "B"])
    out = Series(dtype="int64", index=Index([], name="A"))
    tm.assert_series_equal(df.groupby("A").size(), out)


def test_size_groupby_all_null():
    # [FUNCTION DOC]
    # Refs GitHub issue 23050: grouping by an all-null column drops all rows, so size must give an empty int64 Series with an empty index named A rather than raising a length mismatch error.
    # https://github.com/pandas-dev/pandas/issues/23050
    # Assert no 'Value Error : Length of passed values is 2, index implies 0'
    df = DataFrame({"A": [None, None]})  # all-null groups
    result = df.groupby("A").size()
    expected = Series(dtype="int64", index=Index([], name="A"))
    tm.assert_series_equal(result, expected)


def test_size_period_index():
    # [FUNCTION DOC]
    # Refs GitHub issue 34010: a Series indexed by a daily PeriodIndex named A, grouped by that level, has a size equal to the original Series (all groups of size one).
    # https://github.com/pandas-dev/pandas/issues/34010
    ser = Series([1], index=PeriodIndex(["2000"], name="A", freq="D"))
    grp = ser.groupby(level="A")
    result = grp.size()
    tm.assert_series_equal(result, ser)


def test_size_on_categorical(as_index):
    # [FUNCTION DOC]
    # Parametrized over the as_index fixture. Groups a frame by a categorical column and an integer column with observed false, so unobserved combinations appear with size zero. Expected is a frame with a size column (indexed by A and B and reduced to an unnamed Series when as_index is true), with A categorical.
    df = DataFrame([[1, 1], [2, 2]], columns=["A", "B"])
    df["A"] = df["A"].astype("category")
    result = df.groupby(["A", "B"], as_index=as_index, observed=False).size()

    expected = DataFrame(
        [[1, 1, 1], [1, 2, 0], [2, 1, 0], [2, 2, 1]], columns=["A", "B", "size"]
    )
    expected["A"] = expected["A"].astype("category")
    if as_index:
        expected = expected.set_index(["A", "B"])["size"].rename(None)

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_size_series_masked_type_returns_Int64(dtype):
    # [FUNCTION DOC]
    # GH 54132: for masked dtypes Int64, Float64 and boolean, size of a level-0 grouping must return an Int64 Series (nullable) holding the group counts.
    # GH 54132
    ser = Series([1, 1, 1], index=["a", "a", "b"], dtype=dtype)
    result = ser.groupby(level=0).size()
    expected = Series([2, 1], dtype="Int64", index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_size_strings(any_string_dtype, using_infer_string):
    # [FUNCTION DOC]
    # GH#55627: size of a single column grouped by a string-dtype column across any_string_dtype. The count dtype is Int64 for pyarrow-backed strings and int64 otherwise; the index dtype is the string dtype, or str when infer-string mode is on and the dtype is object.
    # GH#55627
    dtype = any_string_dtype
    df = DataFrame({"a": ["a", "a", "b"], "b": "a"}, dtype=dtype)
    result = df.groupby("a")["b"].size()
    exp_dtype = "Int64" if dtype == "string[pyarrow]" else "int64"
    exp_index_dtype = "str" if using_infer_string and dtype == "object" else dtype
    expected = Series(
        [2, 1],
        index=Index(["a", "b"], name="a", dtype=exp_index_dtype),
        name="b",
        dtype=exp_dtype,
    )
    tm.assert_series_equal(result, expected)
