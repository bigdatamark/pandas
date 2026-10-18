# [MODULE DOC]
# Main tests for the merge function and the frame merge method: key selection
# (on, left_on/right_on, index flags, array keys), join kinds, suffix handling,
# empty frames, dtype preservation (integer, bool, datetime with and without
# timezone, period, timedelta, categorical, extension, nullable), indicator column,
# validation of one-to-one/one-to-many relations, sorting, merging on indexes, and
# various bug regressions. Defines a test-data helper, shared fixtures
# (dfs_for_indicator, left) and several test classes: TestMerge, TestMergeDtypes,
# TestMergeCategorical, TestMergeOnIndexes, plus module-level helpers and tests.
# Uses conftest fixtures join_type, index, sort, using_infer_string and others.
from datetime import (
    date,
    datetime,
    timedelta,
)
import re

import numpy as np
import pytest

from pandas.compat import PY314

from pandas.core.dtypes.common import (
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)


def get_test_data(ngroups=8, n=50):
    # [FUNCTION DOC]
    # HELPER: builds a one-dimensional integer array of length n made by repeating the
    # group labels zero through ngroups minus one, topping up with the first labels if
    # the tiling is short, then shuffling in place with a fixed-seed generator.
    # Parameters: ngroups (default 8), n (default 50). Returns the shuffled array.
    # Same behavior as the helper of the same name in the join tests module.
    unique_groups = list(range(ngroups))
    arr = np.asarray(np.tile(unique_groups, n // ngroups))

    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])

    np.random.default_rng(2).shuffle(arr)
    return arr


@pytest.fixture
def dfs_for_indicator():
    # [FUNCTION DOC]
    # HELPER: fixture returning a pair of frames for indicator tests. The first has
    # columns col1, col_conflict and col_left (two rows); the second has col1,
    # col_conflict and col_right (five rows).
    df1 = DataFrame({"col1": [0, 1], "col_conflict": [1, 2], "col_left": ["a", "b"]})
    df2 = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col_conflict": [1, 2, 3, 4, 5],
            "col_right": [2, 2, 2, 2, 2],
        }
    )
    return df1, df2


class TestMerge:
    # [CLASS DOC]
    # Collection of general merge tests. Provides fixtures df, df2 and left giving test
    # frames. Stateless between tests.
    @pytest.fixture
    def df(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a frame of 50 rows with two integer key columns from the
        # key generator and two random float columns, keeping only rows where the second key
        # exceeds one.
        df = DataFrame(
            {
                "key1": get_test_data(),
                "key2": get_test_data(),
                "data1": np.random.default_rng(2).standard_normal(50),
                "data2": np.random.default_rng(2).standard_normal(50),
            }
        )

        # exclude a couple keys for fun
        df = df[df["key2"] > 1]
        return df

    @pytest.fixture
    def df2(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a ten-row frame with two integer key columns (eight and
        # four groups) and a random float column named value.
        return DataFrame(
            {
                "key1": get_test_data(n=10),
                "key2": get_test_data(ngroups=4, n=10),
                "value": np.random.default_rng(2).standard_normal(10),
            }
        )

    @pytest.fixture
    def left(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a seven-row frame with a string key column having
        # repeated keys and a random float column v1. (A module-level fixture with the same
        # name exists later in the module.)
        return DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )

    def test_merge_inner_join_empty(self):
        # [FUNCTION DOC]
        # Verifies merging an empty frame with a non-empty one on both indexes gives an
        # empty integer-column frame (GH 15328).
        # GH 15328
        df_empty = DataFrame()
        df_a = DataFrame({"a": [1, 2]}, index=[0, 1], dtype="int64")
        result = merge(df_empty, df_a, left_index=True, right_index=True)
        expected = DataFrame({"a": []}, dtype="int64")
        tm.assert_frame_equal(result, expected)

    def test_merge_common(self, df, df2):
        # [FUNCTION DOC]
        # Verifies merge with no key arguments uses the common columns (equals merging on
        # all of them).
        joined = merge(df, df2)
        exp = merge(df, df2, on=["key1", "key2"])
        tm.assert_frame_equal(joined, exp)

    def test_merge_non_string_columns(self):
        # [FUNCTION DOC]
        # Verifies merging frames with integer column labels (all common) works and
        # returns the left values (GH 17962).
        # https://github.com/pandas-dev/pandas/issues/17962
        # Checks that method runs for non string column names
        left = DataFrame(
            {0: [1, 0, 1, 0], 1: [0, 1, 0, 0], 2: [0, 0, 2, 0], 3: [1, 0, 0, 3]}
        )

        right = left.astype(float)
        expected = left
        result = merge(left, right)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df, df2):
        # [FUNCTION DOC]
        # Verifies that an index level name can be given as the key argument for both
        # sides (GH 14355).
        # GH14355

        left = df.set_index("key1")
        right = df2.set_index("key1")
        result = merge(left, right, on="key1")
        expected = merge(df, df2, on="key1").set_index("key1")
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self):
        # [FUNCTION DOC]
        # Verifies that a left merge on a column against the right index equals the mirrored
        # right merge, with and without sorting.
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=False
        )
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=False
        )
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=True
        )
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=True
        )
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self):
        # [FUNCTION DOC]
        # Verifies inner merge of a column key against an index equals the matching rows of
        # the index-based join, in both argument orders.
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

        # inner join
        result = merge(left, right, left_on="key", right_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected)

        result = merge(right, left, right_on="key", left_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

    def test_merge_misspecified(self, df, df2, left):
        # [FUNCTION DOC]
        # Verifies merge errors: missing right or left key specification when only the other
        # side's index flag is set; combining the on argument with left_on/right_on; and
        # unequal lengths of left_on and right_on lists (a value error).
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )
        msg = "Must pass right_on or right_index=True"
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, left_index=True)
        msg = "Must pass left_on or left_index=True"
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, right_index=True)

        msg = (
            'Can only pass argument "on" OR "left_on" and "right_on", not '
            "a combination of both"
        )
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on="key", on="key")

        msg = r"len\(right_on\) must equal len\(left_on\)"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=["key1"], right_on=["key1", "key2"])

    def test_index_and_on_parameters_confusion(self, df, df2):
        # [FUNCTION DOC]
        # Verifies a value error is raised when index flags are given as lists instead of
        # booleans, naming the offending parameter and its type.
        msg = "right_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=False,
                right_index=["key1", "key2"],
            )
        msg = "left_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=False,
            )
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=["key1", "key2"],
            )

    def test_merge_overlap(self, left):
        # [FUNCTION DOC]
        # Verifies merging a frame with itself on its key repeats rows per squared key count
        # and adds default suffixes to the shared column.
        merged = merge(left, left, on="key")
        exp_len = (left["key"].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert "v1_x" in merged
        assert "v1_y" in merged

    def test_merge_different_column_key_names(self):
        # [FUNCTION DOC]
        # Verifies an outer sorted merge with different key column names keeps both key
        # columns and suffixes the duplicated value columns, checked series by series.
        left = DataFrame({"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 4]})
        right = DataFrame({"rkey": ["foo", "bar", "qux", "foo"], "value": [5, 6, 7, 8]})

        merged = left.merge(
            right, left_on="lkey", right_on="rkey", how="outer", sort=True
        )

        exp = Series(["bar", "baz", "foo", "foo", "foo", "foo", np.nan], name="lkey")
        tm.assert_series_equal(merged["lkey"], exp)

        exp = Series(["bar", np.nan, "foo", "foo", "foo", "foo", "qux"], name="rkey")
        tm.assert_series_equal(merged["rkey"], exp)

        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name="value_x")
        tm.assert_series_equal(merged["value_x"], exp)

        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name="value_y")
        tm.assert_series_equal(merged["value_y"], exp)

    def test_merge_copy(self):
        # [FUNCTION DOC]
        # Verifies that modifying the merged result does not alter the inputs.
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame({"c": "foo", "d": "bar"}, index=range(10))

        merged = merge(left, right, left_index=True, right_index=True)

        merged["a"] = 6
        assert (left["a"] == 0).all()

        merged["d"] = "peekaboo"
        assert (right["d"] == "bar").all()

    def test_merge_nocopy(self, using_infer_string):
        # [FUNCTION DOC]
        # Takes using_infer_string. Verifies merged numeric columns share memory with the
        # inputs (no copy), and for object columns too unless string inference is enabled.
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame({"c": "foo", "d": "bar"}, index=range(10))

        merged = merge(left, right, left_index=True, right_index=True)

        assert np.shares_memory(merged["a"]._values, left["a"]._values)
        if not using_infer_string:
            assert np.shares_memory(merged["d"]._values, right["d"]._values)

    def test_intelligently_handle_join_key(self):
        # [FUNCTION DOC]
        # Verifies an outer merge on a key where the left frame orders columns value then key
        # keeps that column order and fills missing values (GH 733).
        # #733, be a bit more 1337 about not returning unconsolidated DataFrame

        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": list(range(5))}, columns=["value", "key"]
        )
        right = DataFrame({"key": [1, 1, 2, 3, 4, 5], "rvalue": list(range(6))})

        joined = merge(left, right, on="key", how="outer")
        expected = DataFrame(
            {
                "key": [1, 1, 1, 1, 2, 2, 3, 4, 5],
                "value": np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
                "rvalue": [0, 1, 0, 1, 2, 2, 3, 4, 5],
            },
            columns=["value", "key", "rvalue"],
        )
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self):
        # [FUNCTION DOC]
        # Verifies the key dtype is preserved on outer merge for integer and bool keys, and
        # that merging on arrays creates an integer key column named with the default key
        # label (GH 8596, GH 13169, GH 40073).
        # #8596

        df1 = DataFrame({"key": [1], "v1": [10]})
        df2 = DataFrame({"key": [2], "v1": [20]})
        df = merge(df1, df2, how="outer")
        assert df["key"].dtype == "int64"

        df1 = DataFrame({"key": [True], "v1": [1]})
        df2 = DataFrame({"key": [False], "v1": [0]})
        df = merge(df1, df2, how="outer")

        # GH13169
        # GH#40073
        assert df["key"].dtype == "bool"

        df1 = DataFrame({"val": [1]})
        df2 = DataFrame({"val": [2]})
        lkey = np.array([1])
        rkey = np.array([2])
        df = merge(df1, df2, left_on=lkey, right_on=rkey, how="outer")
        assert df["key_0"].dtype == np.dtype(int)

    def test_handle_join_key_pass_array(self):
        # [FUNCTION DOC]
        # Verifies passing arrays (or a mix of column name and array) as join keys: the
        # resulting key column is not missing and is the same regardless of argument order;
        # unnamed array keys create a default-named key column, including with a left index.
        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": np.arange(5)},
            columns=["value", "key"],
            dtype="int64",
        )
        right = DataFrame({"rvalue": np.arange(6)}, dtype="int64")
        key = np.array([1, 1, 2, 3, 4, 5], dtype="int64")

        merged = merge(left, right, left_on="key", right_on=key, how="outer")
        merged2 = merge(right, left, left_on=key, right_on="key", how="outer")

        tm.assert_series_equal(merged["key"], merged2["key"])
        assert merged["key"].notna().all()
        assert merged2["key"].notna().all()

        left = DataFrame({"value": np.arange(5)}, columns=["value"])
        right = DataFrame({"rvalue": np.arange(6)})
        lkey = np.array([1, 1, 2, 2, 3])
        rkey = np.array([1, 1, 2, 3, 4, 5])

        merged = merge(left, right, left_on=lkey, right_on=rkey, how="outer")
        expected = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name="key_0")
        tm.assert_series_equal(merged["key_0"], expected)

        left = DataFrame({"value": np.arange(3)})
        right = DataFrame({"rvalue": np.arange(6)})

        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left, right, left_index=True, right_on=key, how="outer")
        tm.assert_series_equal(merged["key_0"], Series(key, name="key_0"))

    def test_no_overlap_more_informative_error(self):
        # [FUNCTION DOC]
        # Verifies a merge error saying there are no common columns and listing the
        # merge options, when frames with only non-column overlap (index values) are
        # merged without keys.
        dt = datetime.now()
        df1 = DataFrame({"x": ["a"]}, index=[dt])

        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])

        msg = (
            "No common columns to perform merge on. "
            f"Merge options: left_on={None}, right_on={None}, "
            f"left_index={False}, right_index={False}"
        )

        with pytest.raises(MergeError, match=msg):
            merge(df1, df2)

    def test_merge_non_unique_indexes(self):
        # [FUNCTION DOC]
        # Verifies index merges with non-unique, non-monotonic indexes by comparing to the
        # by-hand expected result via the module's merge-check helper.
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        dt4 = datetime(2012, 5, 4)

        df1 = DataFrame({"x": ["a"]}, index=[dt])
        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])
        _check_merge(df1, df2)

        # Not monotonic
        df1 = DataFrame({"x": ["a", "b", "q"]}, index=[dt2, dt, dt4])
        df2 = DataFrame(
            {"y": ["c", "d", "e", "f", "g", "h"]}, index=[dt3, dt3, dt2, dt2, dt, dt]
        )
        _check_merge(df1, df2)

        df1 = DataFrame({"x": ["a", "b"]}, index=[dt, dt])
        df2 = DataFrame({"y": ["c", "d"]}, index=[dt, dt])
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self):
        # [FUNCTION DOC]
        # Same check for many-to-many cases with duplicated index values on both sides.
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        df1 = DataFrame({"x": ["a", "b", "c", "d"]}, index=[dt2, dt2, dt, dt])
        df2 = DataFrame(
            {"y": ["e", "f", "g", " h", "i"]}, index=[dt2, dt2, dt3, dt, dt]
        )
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self):
        # [FUNCTION DOC]
        # Verifies left merge with an empty right frame returns the left frame, and right
        # merge with an empty left returns the right frame.
        left = DataFrame({"key": [1], "value": [2]})
        right = DataFrame({"key": []})

        result = merge(left, right, on="key", how="left")
        tm.assert_frame_equal(result, left)

        result = merge(right, left, on="key", how="right")
        tm.assert_frame_equal(result, left)

    def test_merge_empty_dataframe(self, index, join_type):
        # [FUNCTION DOC]
        # Takes index and join_type fixtures. Verifies joining an empty frame (with the
        # fixture index emptied) with a copy of itself returns an equal frame (GH 52777).
        # GH52777
        left = DataFrame([], index=index[:0])
        right = left.copy()

        result = left.join(right, how=join_type)
        tm.assert_frame_equal(result, left)

    @pytest.mark.parametrize(
        "kwarg",
        [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
            {"left_on": "a", "right_index": True},
            {"left_on": "a", "right_on": "x"},
        ],
    )
    def test_merge_left_empty_right_empty(self, join_type, kwarg):
        # [FUNCTION DOC]
        # Parametrized over key specifications (index flags/columns) and join_type. Verifies
        # merging two empty frames gives an empty frame with all columns of object dtype
        # (GH 10824).
        # GH 10824
        left = DataFrame(columns=["a", "b", "c"])
        right = DataFrame(columns=["x", "y", "z"])

        exp_in = DataFrame(columns=["a", "b", "c", "x", "y", "z"], dtype=object)

        result = merge(left, right, how=join_type, **kwarg)
        tm.assert_frame_equal(result, exp_in)

    def test_merge_left_empty_right_notempty(self):
        # [FUNCTION DOC]
        # Verifies merges of an empty left frame with a non-empty right frame for the four
        # key specifications: inner/left give empty frames, right/outer give the right rows
        # with object-typed missing left columns (and index-derived values for a left key
        # column used with the right index) (GH 10824). The nested helpers check two join
        # kinds each against an expected frame.
        # GH 10824
        left = DataFrame(columns=["a", "b", "c"])
        right = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["x", "y", "z"])

        exp_out = DataFrame(
            {
                "a": np.array([np.nan] * 3, dtype=object),
                "b": np.array([np.nan] * 3, dtype=object),
                "c": np.array([np.nan] * 3, dtype=object),
                "x": [1, 4, 7],
                "y": [2, 5, 8],
                "z": [3, 6, 9],
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype

        def check1(exp, kwarg):
            # [FUNCTION DOC]
            # HELPER: nested in test_merge_left_empty_right_notempty. Merges with the inner and
            # left kinds using the given key keyword arguments and asserts both equal the
            # expected frame.
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp, kwarg):
            # [FUNCTION DOC]
            # HELPER: nested in test_merge_left_empty_right_notempty. Merges with the right and
            # outer kinds using the given key keyword arguments and asserts both equal the
            # expected frame.
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

        for kwarg in [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
        ]:
            check1(exp_in, kwarg)
            check2(exp_out, kwarg)

        kwarg = {"left_on": "a", "right_index": True}
        check1(exp_in, kwarg)
        exp_out["a"] = [0, 1, 2]
        check2(exp_out, kwarg)

        kwarg = {"left_on": "a", "right_on": "x"}
        check1(exp_in, kwarg)
        exp_out["a"] = np.array([np.nan] * 3, dtype=object)
        check2(exp_out, kwarg)

    def test_merge_left_notempty_right_empty(self):
        # [FUNCTION DOC]
        # Intended to verify merges of a non-empty left frame with an empty right frame for
        # several key specifications (GH 10824), using nested helpers.
        # KNOWN ISSUE: the
        # loop that applies the helpers is indented inside the second helper instead of the
        # test body (a TODO notes that un-indenting breaks the test), and the helpers are
        # never called from the test body, so the test executes no merges and verifies
        # nothing; if the second helper were called it would recurse into itself.
        # GH 10824
        left = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
        right = DataFrame(columns=["x", "y", "z"])

        exp_out = DataFrame(
            {
                "a": [1, 4, 7],
                "b": [2, 5, 8],
                "c": [3, 6, 9],
                "x": np.array([np.nan] * 3, dtype=object),
                "y": np.array([np.nan] * 3, dtype=object),
                "z": np.array([np.nan] * 3, dtype=object),
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype
        # result will have object dtype
        exp_in.index = exp_in.index.astype(object)

        def check1(exp, kwarg):
            # [FUNCTION DOC]
            # HELPER: nested in test_merge_left_notempty_right_empty. Merges with the inner and
            # right kinds using the given key keyword arguments and asserts both equal the
            # expected frame.
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp, kwarg):
            # [FUNCTION DOC]
            # HELPER: nested in test_merge_left_notempty_right_empty. Merges with the left and
            # outer kinds and asserts equality with the expected frame, then loops over four key
            # specifications calling the helpers (this loop is why the helper never terminates
            # if called).
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

            # TODO: should the next loop be un-indented? doing so breaks this test
            for kwarg in [
                {"left_index": True, "right_index": True},
                {"left_index": True, "right_on": "x"},
                {"left_on": "a", "right_index": True},
                {"left_on": "a", "right_on": "x"},
            ]:
                check1(exp_in, kwarg)
                check2(exp_out, kwarg)

    @pytest.mark.parametrize(
        "series_of_dtype",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    @pytest.mark.parametrize(
        "series_of_dtype2",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    def test_merge_empty_frame(self, series_of_dtype, series_of_dtype2):
        # [FUNCTION DOC]
        # Parametrized over two series dtypes (integer, nullable integer, float, string,
        # bool, naive and tz-aware datetime). Verifies merging an empty slice of a
        # two-column frame with the full frame on the key gives an empty frame with the
        # key dtype and suffixed value columns of the value dtype (GH 25183).
        # GH 25183
        df = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype2},
            columns=["key", "value"],
        )
        df_empty = df[:0]
        expected = DataFrame(
            {
                "key": Series(dtype=df.dtypes["key"]),
                "value_x": Series(dtype=df.dtypes["value"]),
                "value_y": Series(dtype=df.dtypes["value"]),
            },
            columns=["key", "value_x", "value_y"],
        )
        actual = df_empty.merge(df, on="key")
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "series_of_dtype",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    @pytest.mark.parametrize(
        "series_of_dtype_all_na",
        [
            Series([pd.NA], dtype="Int64"),
            Series([np.nan], dtype="float"),
            Series([np.nan], dtype="object"),
            Series([pd.NaT]),
        ],
    )
    def test_merge_all_na_column(self, series_of_dtype, series_of_dtype_all_na):
        # [FUNCTION DOC]
        # Parametrized over key dtypes and all-missing column dtypes. Verifies that merging
        # frames with an all-missing value column on the key keeps the column dtypes with
        # suffixes (GH 25183).
        # GH 25183
        df_left = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        df_right = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        expected = DataFrame(
            {
                "key": series_of_dtype,
                "value_x": series_of_dtype_all_na,
                "value_y": series_of_dtype_all_na,
            },
            columns=["key", "value_x", "value_y"],
        )
        actual = df_left.merge(df_right, on="key")
        tm.assert_frame_equal(actual, expected)

    def test_merge_nosort(self):
        # [FUNCTION DOC]
        # Verifies that with sorting disabled the merged result keeps the order of first
        # appearance of the left keys and equals the method call form (GH 2098).
        # GH#2098

        d = {
            "var1": np.random.default_rng(2).integers(0, 10, size=10),
            "var2": np.random.default_rng(2).integers(0, 10, size=10),
            "var3": [
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2010, 2, 3),
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2012, 4, 3),
                datetime(2012, 3, 4),
                datetime(2008, 5, 1),
                datetime(2010, 2, 3),
                datetime(2012, 2, 3),
            ],
        }
        df = DataFrame.from_dict(d)
        var3 = df.var3.unique()
        var3 = np.sort(var3)
        new = DataFrame.from_dict(
            {"var3": var3, "var8": np.random.default_rng(2).random(7)}
        )

        result = df.merge(new, on="var3", sort=False)
        exp = merge(df, new, on="var3", sort=False)
        tm.assert_frame_equal(result, exp)

        assert (df.var3.unique() == result.var3.unique()).all()

    @pytest.mark.parametrize(
        ("sort", "values"), [(False, [1, 1, 0, 1, 1]), (True, [0, 1, 1, 1, 1])]
    )
    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_same_order_left_right(self, sort, values, how):
        # [FUNCTION DOC]
        # Parametrized over (sort flag, expected values) and join kind (left or right).
        # Verifies that self-merging a frame with duplicate keys yields the same order for
        # left and right joins, sorted when requested (GH 35382).
        # GH#35382
        df = DataFrame({"a": [1, 0, 1]})

        result = df.merge(df, on="a", how=how, sort=sort)
        expected = DataFrame(values, columns=["a"])
        tm.assert_frame_equal(result, expected)

    def test_merge_nan_right(self):
        # [FUNCTION DOC]
        # Verifies the frame join with a right suffix where one key has no match: missing
        # values appear in right columns, which become float (GH regression).
        df1 = DataFrame({"i1": [0, 1], "i2": [0, 1]})
        df2 = DataFrame({"i1": [0], "i3": [0]})
        result = df1.join(df2, on="i1", rsuffix="_")
        expected = (
            DataFrame(
                {
                    "i1": {0: 0.0, 1: 1},
                    "i2": {0: 0, 1: 1},
                    "i1_": {0: 0, 1: np.nan},
                    "i3": {0: 0.0, 1: np.nan},
                    None: {0: 0, 1: 0},
                },
                columns=Index(["i1", "i2", "i1_", "i3", None], dtype=object),
            )
            .set_index(None)
            .reset_index()[["i1", "i2", "i1_", "i3"]]
        )
        result.columns = result.columns.astype("object")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_nan_right2(self):
        # [FUNCTION DOC]
        # Same with float columns: verifies columns order and missing values for the
        # unmatched key.
        df1 = DataFrame({"i1": [0, 1], "i2": [0.5, 1.5]})
        df2 = DataFrame({"i1": [0], "i3": [0.7]})
        result = df1.join(df2, rsuffix="_", on="i1")
        expected = DataFrame(
            {
                "i1": {0: 0, 1: 1},
                "i1_": {0: 0.0, 1: np.nan},
                "i2": {0: 0.5, 1: 1.5},
                "i3": {0: 0.69999999999999996, 1: np.nan},
            }
        )[["i1", "i2", "i1_", "i3"]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, df, df2):
        # [FUNCTION DOC]
        # Ignores block-manager deprecation warnings. Verifies that merging a frame subclass
        # (defined locally, overriding its constructor property) with a plain frame returns
        # an instance of the subclass.
        class NotADataFrame(DataFrame):
            # [CLASS DOC]
            # HELPER: local DataFrame subclass used by test_merge_type; its constructor
            # property returns the subclass so operations preserve the type.
            @property
            def _constructor(self):
                # [FUNCTION DOC]
                # HELPER: property of the local subclass returning the subclass itself (used by
                # pandas to construct results).
                return NotADataFrame

        nad = NotADataFrame(df)
        result = nad.merge(df2, on="key1")

        assert isinstance(result, NotADataFrame)

    def test_join_append_timedeltas(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a frame having object columns with a frame holding a
        # datetime and a timedelta keeps both values and gives object dtype (GH 5695).
        # Although grouped with join tests, it uses the concatenate function only.
        # timedelta64 issues with join/merge
        # GH 5695

        d = DataFrame.from_dict(
            {"d": [datetime(2013, 11, 5, 5, 56)], "t": [timedelta(0, 22500)]}
        )
        df = DataFrame(columns=list("dt"))
        df = concat([df, d], ignore_index=True)
        result = concat([df, d], ignore_index=True)
        expected = DataFrame(
            {
                "d": [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)],
                "t": [timedelta(0, 22500), timedelta(0, 22500)],
            },
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_join_append_timedeltas2(self):
        # [FUNCTION DOC]
        # Verifies a left join with a right suffix of timedelta frames fills missing
        # timedeltas with NaT (GH 5695).
        # timedelta64 issues with join/merge
        # GH 5695
        td = np.timedelta64(300000000, "ns")
        lhs = DataFrame(Series([td, td], index=["A", "B"]))
        rhs = DataFrame(Series([td], index=["A"]))

        result = lhs.join(rhs, rsuffix="r", how="left")
        expected = DataFrame(
            {
                "0": Series([td, td], index=list("AB")),
                "0r": Series([td, pd.NaT], index=list("AB")),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_datetime_unit(self, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime resolution units. Verifies merging a frame with a
        # datetime column of all-missing values keeps the resolution (units day, hour and
        # minute are cast to seconds) (GH 13389).
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"datetime64[{unit}]"

        if unit in ["D", "h", "m"]:
            # not supported so we cast to the nearest supported unit, seconds
            exp_dtype = "datetime64[s]"
        else:
            exp_dtype = dtype
        df2 = ser.astype(exp_dtype).to_frame("days")
        assert df2["days"].dtype == exp_dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        days = np.array(["nat", "nat"], dtype=exp_dtype)
        days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
        exp = DataFrame(
            {
                "entity_id": [101, 102],
                "days": days,
            },
            columns=["entity_id", "days"],
        )
        assert exp["days"].dtype == exp_dtype
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_timedelta_unit(self, unit):
        # [FUNCTION DOC]
        # Parametrized over timedelta resolution units. For units day, hour and minute,
        # verifies casting raises a value error listing supported resolutions, then uses
        # seconds; verifies merging keeps the resolution of the all-missing timedelta
        # column (GH 13389).
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"m8[{unit}]"
        if unit in ["D", "h", "m"]:
            # We cannot astype, instead do nearest supported unit, i.e. "s"
            msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)

            df2 = ser.astype("m8[s]").to_frame("days")
        else:
            df2 = ser.astype(dtype).to_frame("days")
            assert df2["days"].dtype == dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        exp = DataFrame(
            {"entity_id": [101, 102], "days": np.array(["nat", "nat"], dtype=dtype)},
            columns=["entity_id", "days"],
        )
        tm.assert_frame_equal(result, exp)

    def test_overlapping_columns_error_message(self):
        # [FUNCTION DOC]
        # Verifies merging frames with duplicate-named columns that are non-overlapping
        # works, and when the merge would produce duplicate columns that are not unique a
        # merge error about data columns not being unique is raised (GH 2649, GH 10639).
        df = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})
        df2 = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})

        df.columns = ["key", "foo", "foo"]
        df2.columns = ["key", "bar", "bar"]
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "v1": [4, 5, 6],
                "v2": [7, 8, 9],
                "v3": [4, 5, 6],
                "v4": [7, 8, 9],
            }
        )
        expected.columns = ["key", "foo", "foo", "bar", "bar"]
        tm.assert_frame_equal(merge(df, df2), expected)

        # #2649, #10639
        df2.columns = ["key1", "foo", "foo"]
        msg = r"Data columns not unique: Index\(\['foo'\], dtype='object|str'\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    def test_merge_on_datetime64tz(self):
        # [FUNCTION DOC]
        # Verifies outer merge on a tz-aware datetime key gives the union of keys with
        # suffixed value columns (GH 11405).
        # GH11405
        left = DataFrame(
            {
                "key": pd.date_range("20151010", periods=2, tz="US/Eastern"),
                "value": [1, 2],
            }
        )
        right = DataFrame(
            {
                "key": pd.date_range("20151011", periods=3, tz="US/Eastern"),
                "value": [1, 2, 3],
            }
        )

        expected = DataFrame(
            {
                "key": pd.date_range("20151010", periods=4, tz="US/Eastern"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_values(self):
        # [FUNCTION DOC]
        # Verifies outer merge on an integer key with tz-aware datetime value columns keeps
        # the tz-aware dtype with NaT fill.
        left = DataFrame(
            {
                "key": [1, 2],
                "value": pd.date_range(
                    "20151010", periods=2, tz="US/Eastern", unit="ns"
                ),
            }
        )
        right = DataFrame(
            {
                "key": [2, 3],
                "value": pd.date_range(
                    "20151011", periods=2, tz="US/Eastern", unit="ns"
                ),
            }
        )
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": [
                    *list(
                        pd.date_range("20151010", periods=2, tz="US/Eastern", unit="ns")
                    ),
                    pd.NaT,
                ],
                "value_y": [
                    pd.NaT,
                    *list(
                        pd.date_range("20151011", periods=2, tz="US/Eastern", unit="ns")
                    ),
                ],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
        assert result["value_x"].dtype == "datetime64[ns, US/Eastern]"
        assert result["value_y"].dtype == "datetime64[ns, US/Eastern]"

    def test_merge_on_datetime64tz_empty(self):
        # [FUNCTION DOC]
        # Verifies merging an empty frame (with a tz-aware key) with the full frame on the
        # key keeps the datetime-with-timezone dtypes for the key and extra column (GH 25014).
        # https://github.com/pandas-dev/pandas/issues/25014
        dtz = pd.DatetimeTZDtype(tz="UTC")
        right = DataFrame(
            {
                "date": DatetimeIndex(["2018"], dtype=dtz),
                "value": [4.0],
                "date2": DatetimeIndex(["2019"], dtype=dtz),
            },
            columns=["date", "value", "date2"],
        )
        left = right[:0]
        result = left.merge(right, on="date")
        expected = DataFrame(
            {
                "date": Series(dtype=dtz),
                "value_x": Series(dtype=float),
                "date2_x": Series(dtype=dtz),
                "value_y": Series(dtype=float),
                "date2_y": Series(dtype=dtz),
            },
            columns=["date", "value_x", "date2_x", "value_y", "date2_y"],
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_with_dst_transition(self):
        # [FUNCTION DOC]
        # Verifies an outer merge on tz-aware dates spanning a DST change (one side built by
        # converting from UTC) aligns identical instants (GH 18885).
        # GH 18885
        df1 = DataFrame(
            pd.date_range("2017-10-29 01:00", periods=4, freq="h", tz="Europe/Madrid"),
            columns=["date"],
        )
        df1["value"] = 1
        df2 = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2017-10-29 03:00:00",
                        "2017-10-29 04:00:00",
                        "2017-10-29 05:00:00",
                    ]
                ),
                "value": 2,
            }
        )
        df2["date"] = df2["date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid")
        result = merge(df1, df2, how="outer", on="date")
        expected = DataFrame(
            {
                "date": pd.date_range(
                    "2017-10-29 01:00", periods=7, freq="h", tz="Europe/Madrid"
                ),
                "value_x": [1] * 4 + [np.nan] * 3,
                "value_y": [np.nan] * 4 + [2] * 3,
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_non_unique_period_index(self):
        # [FUNCTION DOC]
        # Verifies inner index merge with a duplicated period index gives pairs with default
        # suffixes (GH 16871).
        # GH #16871
        index = pd.period_range("2016-01-01", periods=16, freq="M")
        df = DataFrame(list(range(len(index))), index=index, columns=["pnum"])
        df2 = concat([df, df])
        result = df.merge(df2, left_index=True, right_index=True, how="inner")
        expected = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=["pnum_x", "pnum_y"],
            index=df2.sort_index().index,
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self):
        # [FUNCTION DOC]
        # Verifies outer merge on a period key.
        left = DataFrame(
            {"key": pd.period_range("20151010", periods=2, freq="D"), "value": [1, 2]}
        )
        right = DataFrame(
            {
                "key": pd.period_range("20151011", periods=3, freq="D"),
                "value": [1, 2, 3],
            }
        )

        expected = DataFrame(
            {
                "key": pd.period_range("20151010", periods=4, freq="D"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self):
        # [FUNCTION DOC]
        # Verifies outer merge on integer key with period values keeps the daily period
        # dtype with NaT fill.
        left = DataFrame(
            {"key": [1, 2], "value": pd.period_range("20151010", periods=2, freq="D")}
        )
        right = DataFrame(
            {"key": [2, 3], "value": pd.period_range("20151011", periods=2, freq="D")}
        )

        exp_x = pd.period_range("20151010", periods=2, freq="D")
        exp_y = pd.period_range("20151011", periods=2, freq="D")
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": [*list(exp_x), pd.NaT],
                "value_y": [pd.NaT, *list(exp_y)],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
        assert result["value_x"].dtype == "Period[D]"
        assert result["value_y"].dtype == "Period[D]"

    def test_indicator(self, dfs_for_indicator):
        # [FUNCTION DOC]
        # Uses the dfs_for_indicator fixture. Verifies an outer merge with the indicator
        # option adds a categorical column named for the default indicator label with
        # categories left-only, right-only, both, holding the origin of each row; the
        # function and method forms agree, inputs are not modified, and a custom string
        # name for the indicator column is used when given (PR 10054, GH 7412, GH 8790).
        # PR #10054. xref #7412 and closes #8790.
        df1, df2 = dfs_for_indicator
        df1_copy = df1.copy()

        df2_copy = df2.copy()

        df_result = DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5],
                "col_conflict_x": [1, 2, np.nan, np.nan, np.nan, np.nan],
                "col_left": ["a", "b", np.nan, np.nan, np.nan, np.nan],
                "col_conflict_y": [np.nan, 1, 2, 3, 4, 5],
                "col_right": [np.nan, 2, 2, 2, 2, 2],
            }
        )
        df_result["_merge"] = Categorical(
            [
                "left_only",
                "both",
                "right_only",
                "right_only",
                "right_only",
                "right_only",
            ],
            categories=["left_only", "right_only", "both"],
        )

        df_result = df_result[
            [
                "col1",
                "col_conflict_x",
                "col_left",
                "col_conflict_y",
                "col_right",
                "_merge",
            ]
        ]

        test = merge(df1, df2, on="col1", how="outer", indicator=True)
        tm.assert_frame_equal(test, df_result)
        test = df1.merge(df2, on="col1", how="outer", indicator=True)
        tm.assert_frame_equal(test, df_result)

        # No side effects
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # Check with custom name
        df_result_custom_name = df_result
        df_result_custom_name = df_result_custom_name.rename(
            columns={"_merge": "custom_name"}
        )

        test_custom_name = merge(
            df1, df2, on="col1", how="outer", indicator="custom_name"
        )
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)
        test_custom_name = df1.merge(
            df2, on="col1", how="outer", indicator="custom_name"
        )
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

    def test_merge_indicator_arg_validation(self, dfs_for_indicator):
        # [FUNCTION DOC]
        # Verifies a value error saying the indicator option accepts only boolean or string
        # arguments when given an integer, for function and method forms.
        # Check only accepts strings and booleans
        df1, df2 = dfs_for_indicator

        msg = "indicator option can only accept boolean or string arguments"
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on="col1", how="outer", indicator=5)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df2, on="col1", how="outer", indicator=5)

    def test_merge_indicator_result_integrity(self, dfs_for_indicator):
        # [FUNCTION DOC]
        # Verifies for left, right and inner merges that the indicator column never holds
        # right-only (left merge), never left-only (right merge), and is always both (inner
        # merge), for function and method forms.
        # Check result integrity
        df1, df2 = dfs_for_indicator

        test2 = merge(df1, df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()
        test2 = df1.merge(df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()

        test3 = merge(df1, df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()
        test3 = df1.merge(df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()

        test4 = merge(df1, df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()
        test4 = df1.merge(df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()

    def test_merge_indicator_invalid(self, dfs_for_indicator):
        # [FUNCTION DOC]
        # Verifies value errors when the indicator name collides with an existing column:
        # for the three reserved default/internal names the message says an existing
        # column name cannot be used, and for a custom name present in the data too.
        # Check if working name in df
        df1, _ = dfs_for_indicator

        for i in ["_right_indicator", "_left_indicator", "_merge"]:
            df_badcolumn = DataFrame({"col1": [1, 2], i: [2, 2]})

            msg = (
                "Cannot use `indicator=True` option when data contains a "
                f"column named {i}|"
                "Cannot use name of an existing column for indicator column"
            )
            with pytest.raises(ValueError, match=msg):
                merge(df1, df_badcolumn, on="col1", how="outer", indicator=True)
            with pytest.raises(ValueError, match=msg):
                df1.merge(df_badcolumn, on="col1", how="outer", indicator=True)

        # Check for name conflict with custom name
        df_badcolumn = DataFrame({"col1": [1, 2], "custom_column_name": [2, 2]})

        msg = "Cannot use name of an existing column for indicator column"
        with pytest.raises(ValueError, match=msg):
            merge(
                df1,
                df_badcolumn,
                on="col1",
                how="outer",
                indicator="custom_column_name",
            )
        with pytest.raises(ValueError, match=msg):
            df1.merge(
                df_badcolumn, on="col1", how="outer", indicator="custom_column_name"
            )

    def test_merge_indicator_multiple_columns(self):
        # [FUNCTION DOC]
        # Verifies the indicator column when merging on two key columns matches a
        # hand-coded result, for function and method forms.
        # Merge on multiple columns
        df3 = DataFrame({"col1": [0, 1], "col2": ["a", "b"]})

        df4 = DataFrame({"col1": [1, 1, 3], "col2": ["b", "x", "y"]})

        hand_coded_result = DataFrame(
            {"col1": [0, 1, 1, 3], "col2": ["a", "b", "x", "y"]}
        )
        hand_coded_result["_merge"] = Categorical(
            ["left_only", "both", "right_only", "right_only"],
            categories=["left_only", "right_only", "both"],
        )

        test5 = merge(df3, df4, on=["col1", "col2"], how="outer", indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=["col1", "col2"], how="outer", indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self):
        # [FUNCTION DOC]
        # Verifies the validate argument: valid one-to-one merges (index, columns, mixed)
        # leave inputs unmodified and produce the expected frames; duplicates on the right,
        # left or both trigger merge errors with messages stating which dataset is not
        # unique and which relation was requested; an unknown validate value raises a value
        # error listing the valid choices; and a two-column key that is jointly unique
        # passes one-to-one validation while a single duplicated column fails.
        left = DataFrame(
            {"a": ["a", "b", "c", "d"], "b": ["cat", "dog", "weasel", "horse"]},
            index=range(4),
        )

        right = DataFrame(
            {
                "a": ["a", "b", "c", "d", "e"],
                "c": ["meow", "bark", "um... weasel noise?", "nay", "chirp"],
            },
            index=range(5),
        )

        # Make sure no side effects.
        left_copy = left.copy()
        right_copy = right.copy()

        result = merge(left, right, left_index=True, right_index=True, validate="1:1")
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)

        # make sure merge still correct
        expected = DataFrame(
            {
                "a_x": ["a", "b", "c", "d"],
                "b": ["cat", "dog", "weasel", "horse"],
                "a_y": ["a", "b", "c", "d"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            index=range(4),
            columns=["a_x", "b", "a_y", "c"],
        )

        result = merge(
            left, right, left_index=True, right_index=True, validate="one_to_one"
        )
        tm.assert_frame_equal(result, expected)

        expected_2 = DataFrame(
            {
                "a": ["a", "b", "c", "d"],
                "b": ["cat", "dog", "weasel", "horse"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            index=range(4),
        )

        result = merge(left, right, on="a", validate="1:1")
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)

        result = merge(left, right, on="a", validate="one_to_one")
        tm.assert_frame_equal(result, expected_2)

        # One index, one column
        expected_3 = DataFrame(
            {
                "b": ["cat", "dog", "weasel", "horse"],
                "a": ["a", "b", "c", "d"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            columns=["b", "a", "c"],
            index=range(4),
        )

        left_index_reset = left.set_index("a")
        result = merge(
            left_index_reset,
            right,
            left_index=True,
            right_on="a",
            validate="one_to_one",
        )
        tm.assert_frame_equal(result, expected_3)

        # Dups on right
        right_w_dups = concat([right, DataFrame({"a": ["e"], "c": ["moo"]}, index=[4])])
        merge(
            left,
            right_w_dups,
            left_index=True,
            right_index=True,
            validate="one_to_many",
        )

        msg = "Merge keys are not unique in right dataset; not a one-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left,
                right_w_dups,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on="a", validate="one_to_one")

        # Dups on left
        left_w_dups = concat(
            [left, DataFrame({"a": ["a"], "c": ["cow"]}, index=[3])], sort=True
        )
        merge(
            left_w_dups,
            right,
            left_index=True,
            right_index=True,
            validate="many_to_one",
        )

        msg = "Merge keys are not unique in left dataset; not a one-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left_w_dups,
                right,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on="a", validate="one_to_one")

        # Dups on both
        merge(left_w_dups, right_w_dups, on="a", validate="many_to_many")

        msg = "Merge keys are not unique in right dataset; not a many-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left_w_dups,
                right_w_dups,
                left_index=True,
                right_index=True,
                validate="many_to_one",
            )

        msg = "Merge keys are not unique in left dataset; not a one-to-many merge"
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on="a", validate="one_to_many")

        # Check invalid arguments
        msg = (
            '"jibberish" is not a valid argument. '
            "Valid arguments are:\n"
            '- "1:1"\n'
            '- "1:m"\n'
            '- "m:1"\n'
            '- "m:m"\n'
            '- "one_to_one"\n'
            '- "one_to_many"\n'
            '- "many_to_one"\n'
            '- "many_to_many"'
        )
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on="a", validate="jibberish")

        # Two column merge, dups in both, but jointly no dups.
        left = DataFrame(
            {
                "a": ["a", "a", "b", "b"],
                "b": [0, 1, 0, 1],
                "c": ["cat", "dog", "weasel", "horse"],
            },
            index=range(4),
        )

        right = DataFrame(
            {
                "a": ["a", "a", "b"],
                "b": [0, 1, 0],
                "d": ["meow", "bark", "um... weasel noise?"],
            },
            index=range(3),
        )

        expected_multi = DataFrame(
            {
                "a": ["a", "a", "b"],
                "b": [0, 1, 0],
                "c": ["cat", "dog", "weasel"],
                "d": ["meow", "bark", "um... weasel noise?"],
            },
            index=range(3),
        )

        msg = (
            "Merge keys are not unique in either left or right dataset; "
            "not a one-to-one merge"
        )
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on="a", validate="1:1")

        result = merge(left, right, on=["a", "b"], validate="1:1")
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_validate_error_message(self):
        # [FUNCTION DOC]
        # Verifies that the validation merge error message includes a Duplicates section
        # naming the side and showing the duplicated key rows (GH 62742).
        # GH#62742
        left = DataFrame({"key": [1, 1, 2]})
        right = DataFrame({"key": [1, 2, 2]})

        with pytest.raises(MergeError, match="Duplicates in left:\n  key\n   1 ...\n"):
            merge(left, right, validate="1:1")
        with pytest.raises(MergeError, match="Duplicates in left:\n  key\n   1 ..."):
            merge(left, right, validate="1:m")
        with pytest.raises(MergeError, match="Duplicates in right:\n  key\n   2 ..."):
            merge(left, right, validate="1:1")
        with pytest.raises(MergeError, match="Duplicates in right:\n  key\n   2 ..."):
            merge(left, right, validate="m:1")

        right = DataFrame({"key": [1, 2, 3]})
        with pytest.raises(MergeError, match="Duplicates in left:\n  key\n   1 ..."):
            merge(left, right, validate="1:1")
        with pytest.raises(MergeError, match="Duplicates in right:\n  key\n   1 ..."):
            merge(right, left, validate="1:1")

    def test_merge_two_empty_df_no_division_error(self):
        # [FUNCTION DOC]
        # Verifies merging two empty frames on a tuple of keys does not raise a floating
        # point division error (GH 17776, PR 17846).
        # GH17776, PR #17846
        a = DataFrame({"a": [], "b": [], "c": []})
        with np.errstate(divide="raise"):
            merge(a, a, on=("a", "b"))

    @pytest.mark.parametrize("how", ["right", "outer"])
    @pytest.mark.parametrize(
        "index,expected_index",
        [
            (
                CategoricalIndex([1, 2, 4]),
                CategoricalIndex([1, 2, 4, None, None, None]),
            ),
            (
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03"], dtype="M8[ns]"
                ),
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03", pd.NaT, pd.NaT, pd.NaT],
                    dtype="M8[ns]",
                ),
            ),
            *[
                (
                    Index([1, 2, 3], dtype=dtyp),
                    Index([1, 2, 3, None, None, None], dtype=np.float64),
                )
                for dtyp in tm.ALL_REAL_NUMPY_DTYPES
            ],
            (
                IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
                IntervalIndex.from_tuples(
                    [(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan]
                ),
            ),
            (
                PeriodIndex(["2001-01-01", "2001-01-02", "2001-01-03"], freq="D"),
                PeriodIndex(
                    ["2001-01-01", "2001-01-02", "2001-01-03", pd.NaT, pd.NaT, pd.NaT],
                    freq="D",
                ),
            ),
            (
                TimedeltaIndex(["1D", "2D", "3D"]),
                TimedeltaIndex(["1D", "2D", "3D", pd.NaT, pd.NaT, pd.NaT]),
            ),
        ],
    )
    def test_merge_on_index_with_more_values(self, how, index, expected_index):
        # [FUNCTION DOC]
        # Parametrized over join kind (right, outer) and index types (categorical, datetime,
        # each real numpy dtype, interval, period, timedelta) with the expected result
        # index. Verifies that merging a column key against a longer right index fills the
        # left index with missing values rather than treating the missing indexer as the
        # last element (GH 24212, GH 41626 for float32).
        # GH 24212
        # pd.merge gets [0, 1, 2, -1, -1, -1] as left_indexer, ensure that
        # -1 is interpreted as a missing value instead of the last element
        if index.dtype == "float32" and expected_index.dtype == "float64":
            # GH#41626
            expected_index = expected_index.astype("float32")
        df1 = DataFrame({"a": [0, 1, 2], "key": [0, 1, 2]}, index=index)
        df2 = DataFrame({"b": [0, 1, 2, 3, 4, 5]})
        result = df1.merge(df2, left_on="key", right_index=True, how=how)
        expected = DataFrame(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [np.nan, 3, 3],
                [np.nan, 4, 4],
                [np.nan, 5, 5],
            ],
            columns=["a", "key", "b"],
        )
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self):
        # [FUNCTION DOC]
        # Regression test for GH 24897: right merge on a column against the right index
        # duplicates a key and leaves an unmatched row with missing index.
        # KNOWN ISSUE: the
        # test's own comment states the expected output is probably incorrect (see
        # issue 17257); it only pins current behavior.
        # Note: the expected output here is probably incorrect.
        # See https://github.com/pandas-dev/pandas/issues/17257 for more.
        # We include this as a regression test for GH-24897.
        left = DataFrame({"a": [1, 2, 3], "key": [0, 1, 1]})
        right = DataFrame({"b": [1, 2, 3]})

        expected = DataFrame(
            {"a": [1, 2, 3, None], "key": [0, 1, 1, 2], "b": [1, 2, 2, 3]},
            columns=["a", "key", "b"],
            index=[0, 1, 2, np.nan],
        )
        result = left.merge(right, left_on="key", right_index=True, how="right")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_preserves_row_order(self, how):
        # [FUNCTION DOC]
        # Parametrized over left and right join. Verifies that a merge on all columns keeps
        # the order of the retained frame's rows (GH 27453).
        # GH 27453
        left_df = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})
        right_df = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        result = left_df.merge(right_df, on=["animal", "max_speed"], how=how)
        if how == "right":
            expected = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        else:
            expected = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self):
        # [FUNCTION DOC]
        # Verifies a right merge of a categorical key column against a categorical right
        # index extends the key categories and fills the missing left value with a missing
        # index label (GH 24212).
        # GH 24212
        left = DataFrame(
            {
                "a": [1, 2, 3],
                "key": Categorical(["a", "a", "b"], categories=list("abc")),
            }
        )
        right = DataFrame({"b": [1, 2, 3]}, index=CategoricalIndex(["a", "b", "c"]))
        result = left.merge(right, left_on="key", right_index=True, how="right")
        expected = DataFrame(
            {
                "a": [1, 2, 3, None],
                "key": Categorical(["a", "a", "b", "c"]),
                "b": [1, 1, 2, 3],
            },
            index=[0, 1, 2, np.nan],
        )
        expected = expected.reindex(columns=["a", "key", "b"])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self):
        # [FUNCTION DOC]
        # Smoke test: merging frames whose underlying arrays are made read-only does not
        # raise (GH 27943).
        # https://github.com/pandas-dev/pandas/issues/27943
        data1 = DataFrame(
            np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
        )
        data2 = DataFrame(
            np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"]
        )

        # make each underlying block array / column array read-only
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False

        data1.merge(data2)  # no error

    def test_merge_how_validation(self):
        # [FUNCTION DOC]
        # Verifies a value error listing valid merge types (including anti joins, cross and
        # asof) when the how argument is not valid (GH 59422).
        # https://github.com/pandas-dev/pandas/issues/59422
        data1 = DataFrame(
            np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
        )
        data2 = DataFrame(
            np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"]
        )
        msg = (
            "'full' is not a valid Merge type: left, right, inner, outer, "
            "left_anti, right_anti, cross, asof"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            data1.merge(data2, how="full")


def _check_merge(x, y):
    # [FUNCTION DOC]
    # HELPER: compares the frame join to the merge on reset indexes. For each of the
    # inner, left, outer kinds and both sort settings, joins x and y, merges the reset
    # frames, restores the index column as index, and asserts equality ignoring names
    # (a TODO notes the names are not checked).
    for how in ["inner", "left", "outer"]:
        for sort in [True, False]:
            result = x.join(y, how=how, sort=sort)

            expected = merge(x.reset_index(), y.reset_index(), how=how, sort=sort)
            expected = expected.set_index("index")

            # TODO check_names on merge?
            tm.assert_frame_equal(result, expected, check_names=False)


class TestMergeDtypes:
    # [CLASS DOC]
    # Collection of merge tests about key and value dtype handling (compatible and
    # incompatible dtypes, extension and string dtypes, small integer types). Stateless.
    @pytest.mark.parametrize("dtype", [object, "category"])
    def test_different(self, dtype):
        # [FUNCTION DOC]
        # Parametrized on right column dtype (object or categorical). Verifies merging on
        # an object column with a column of object or categorical dtype works and the result
        # key is object or string (GH 9780).
        left = DataFrame(
            {
                "A": ["foo", "bar"],
                "B": Series(["foo", "bar"]).astype("category"),
                "C": [1, 2],
                "D": [1.0, 2.0],
                "E": Series([1, 2], dtype="uint64"),
                "F": Series([1, 2], dtype="int32"),
            }
        )
        right_vals = Series(["foo", "bar"], dtype=dtype)
        right = DataFrame({"A": right_vals})

        # GH 9780
        # We allow merging on object and categorical cols and cast
        # categorical cols to object
        result = merge(left, right, on="A")
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize("d2", [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(self, any_int_numpy_dtype, d2):
        # [FUNCTION DOC]
        # Takes any_int_numpy_dtype and a parameter for the right value dtype (int64 or
        # floats). Verifies joining on two key columns against a multi-level index gives the
        # right-value column with matching rows and missing otherwise (integers become float
        # 64), and the same with sorting (rows ordered by keys, stable).
        dtype1 = np.dtype(any_int_numpy_dtype)
        dtype2 = np.dtype(d2)

        left = DataFrame(
            {
                "k1": np.array([0, 1, 2] * 8, dtype=dtype1),
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": np.array([5, 7], dtype=dtype2)}, index=index)

        result = left.join(right, on=["k1", "k2"])

        expected = left.copy()

        if dtype2.kind == "i":
            dtype2 = np.dtype("float64")
        expected["v2"] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=["k1", "k2"], sort=True)
        expected.sort_values(["k1", "k2"], kind="mergesort", inplace=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "int_vals, float_vals, exp_vals",
        [
            ([1, 2, 3], [1.0, 2.0, 3.0], {"X": [1, 2, 3], "Y": [1.0, 2.0, 3.0]}),
            ([1, 2, 3], [1.0, 3.0], {"X": [1, 3], "Y": [1.0, 3.0]}),
            ([1, 2], [1.0, 2.0, 3.0], {"X": [1, 2], "Y": [1.0, 2.0]}),
        ],
    )
    def test_merge_on_ints_floats(self, int_vals, float_vals, exp_vals):
        # [FUNCTION DOC]
        # Parametrized over (left ints, right floats, expected data). Verifies merging an
        # integer column with a float column on equality keeps the float column as float and
        # the integer column as integer, in either order (GH 16572).
        # GH 16572
        # Check that float column is not cast to object if
        # merging on float and int columns
        A = DataFrame({"X": int_vals})
        B = DataFrame({"Y": float_vals})
        expected = DataFrame(exp_vals)

        result = A.merge(B, left_on="X", right_on="Y")
        tm.assert_frame_equal(result, expected)

        result = B.merge(A, left_on="Y", right_on="X")
        tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_key_dtype_cast(self):
        # [FUNCTION DOC]
        # Verifies a left merge of a float key with an integer key keeps the key as float
        # and the right value becomes float (GH 17044).
        # GH 17044
        df1 = DataFrame({"key": [1.0, 2.0], "v1": [10, 20]}, columns=["key", "v1"])
        df2 = DataFrame({"key": [2], "v2": [200]}, columns=["key", "v2"])
        result = df1.merge(df2, on="key", how="left")
        expected = DataFrame(
            {"key": [1.0, 2.0], "v1": [10, 20], "v2": [np.nan, 200.0]},
            columns=["key", "v1", "v2"],
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_ints_floats_warning(self):
        # [FUNCTION DOC]
        # Verifies a UserWarning saying float values are not equal to their integer
        # representation when merging integer with float keys that are not integral, in
        # both orders; and no warning if the float column holds only whole numbers or
        # missing values (GH 16572).
        # GH 16572
        # merge will produce a warning when merging on int and
        # float columns where the float values are not exactly
        # equal to their int representation
        A = DataFrame({"X": [1, 2, 3]})
        B = DataFrame({"Y": [1.1, 2.5, 3.0]})
        expected = DataFrame({"X": [3], "Y": [3.0]})

        msg = "the float values are not equal to their int representation"
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = A.merge(B, left_on="X", right_on="Y")
            tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(UserWarning, match=msg):
            result = B.merge(A, left_on="Y", right_on="X")
            tm.assert_frame_equal(result, expected[["Y", "X"]])

        # test no warning if float has NaNs
        B = DataFrame({"Y": [np.nan, np.nan, 3.0]})

        with tm.assert_produces_warning(None):
            result = B.merge(A, left_on="Y", right_on="X")
            tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_incompat_infer_boolean_object(self):
        # [FUNCTION DOC]
        # Verifies merging an object column of booleans with a bool column works and results
        # in object dtype, in both orders (GH 21119).
        # GH21119: bool + object bool merge OK
        df1 = DataFrame({"key": Series([True, False], dtype=object)})
        df2 = DataFrame({"key": [True, False]})

        expected = DataFrame({"key": [True, False]}, dtype=object)
        result = merge(df1, df2, on="key")
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on="key")
        tm.assert_frame_equal(result, expected)

    def test_merge_incompat_infer_boolean_object_with_missing(self):
        # [FUNCTION DOC]
        # Same as above but the object column contains a missing value, which is dropped by
        # the inner merge (GH 21119).
        # GH21119: bool + object bool merge OK
        # with missing value
        df1 = DataFrame({"key": Series([True, False, np.nan], dtype=object)})
        df2 = DataFrame({"key": [True, False]})

        expected = DataFrame({"key": [True, False]}, dtype=object)
        result = merge(df1, df2, on="key")
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on="key")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals",
        [
            # merge on category coerces to object
            ([0, 1, 2], Series(["a", "b", "a"]).astype("category")),
            ([0.0, 1.0, 2.0], Series(["a", "b", "a"]).astype("category")),
            # no not infer
            ([0, 1], Series([False, True], dtype=object)),
            ([0, 1], Series([False, True], dtype=bool)),
        ],
    )
    def test_merge_incompat_dtypes_are_ok(self, df1_vals, df2_vals):
        # [FUNCTION DOC]
        # Parametrized over explicitly allowed incompatible key pairs (numeric with
        # categorical strings, numeric with bool or object bool values). Verifies the merge
        # succeeds and the key dtype is object (or string for the reversed order when string
        # inference applies).
        # these are explicitly allowed incompat merges, that pass thru
        # the result type is dependent on if the values on the rhs are
        # inferred, otherwise these will be coerced to object

        df1 = DataFrame({"A": df1_vals})
        df2 = DataFrame({"A": df2_vals})

        result = merge(df1, df2, on=["A"])
        assert is_object_dtype(result.A.dtype)
        result = merge(df2, df1, on=["A"])
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals",
        [
            # do not infer to numeric
            (Series([1, 2], dtype="uint64"), ["a", "b", "c"]),
            (Series([1, 2], dtype="int32"), ["a", "b", "c"]),
            ([0, 1, 2], ["0", "1", "2"]),
            ([0.0, 1.0, 2.0], ["0", "1", "2"]),
            (
                pd.date_range("1/1/2011", periods=2, freq="D"),
                ["2011-01-01", "2011-01-02"],
            ),
            (pd.date_range("1/1/2011", periods=2, freq="D"), [0, 1]),
            (pd.date_range("1/1/2011", periods=2, freq="D"), [0.0, 1.0]),
            (
                pd.date_range("20130101", periods=3),
                pd.date_range("20130101", periods=3, tz="US/Eastern"),
            ),
        ],
    )
    def test_merge_incompat_dtypes_error(self, df1_vals, df2_vals):
        # [FUNCTION DOC]
        # Parametrized over key value pairs of incompatible dtypes (numbers with strings,
        # datetimes with strings or numbers, naive with tz-aware datetimes). Verifies a value
        # error naming both dtypes and the key, advising concatenation, in both frame orders,
        # and, for equal-length inputs, that the first incompatible key among several is
        # named when merging on all columns or a subset (GH 9780, GH 15800).
        # GH 9780, GH 15800
        # Raise a ValueError when a user tries to merge on
        # dtypes that are incompatible (e.g., obj and int/float)

        df1 = DataFrame({"A": df1_vals})
        df2 = DataFrame({"A": df2_vals})

        msg = (
            f"You are trying to merge on {df1['A'].dtype} and {df2['A'].dtype} "
            "columns for key 'A'. If you wish to proceed you should use pd.concat"
        )
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on=["A"])

        # Check that error still raised when swapping order of dataframes
        msg = (
            f"You are trying to merge on {df2['A'].dtype} and {df1['A'].dtype} "
            "columns for key 'A'. If you wish to proceed you should use pd.concat"
        )
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df2, df1, on=["A"])

        # Check that error still raised when merging on multiple columns
        # The error message should mention the first incompatible column
        if len(df1_vals) == len(df2_vals):
            # Column A in df1 and df2 is of compatible (the same) dtype
            # Columns B and C in df1 and df2 are of incompatible dtypes
            df3 = DataFrame({"A": df2_vals, "B": df1_vals, "C": df1_vals})
            df4 = DataFrame({"A": df2_vals, "B": df2_vals, "C": df2_vals})

            # Check that error raised correctly when merging all columns A, B, and C
            # The error message should mention key 'B'
            msg = (
                f"You are trying to merge on {df3['B'].dtype} and {df4['B'].dtype} "
                "columns for key 'B'. If you wish to proceed you should use pd.concat"
            )
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4)

            # Check that error raised correctly when merging columns A and C
            # The error message should mention key 'C'
            msg = (
                f"You are trying to merge on {df3['C'].dtype} and {df4['C'].dtype} "
                "columns for key 'C'. If you wish to proceed you should use pd.concat"
            )
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4, on=["A", "C"])

    @pytest.mark.parametrize(
        "expected_data, how",
        [
            ([1, 2], "outer"),
            ([], "inner"),
            ([2], "right"),
            ([1], "left"),
        ],
    )
    def test_merge_EA_dtype(self, any_numeric_ea_dtype, how, expected_data):
        # [FUNCTION DOC]
        # Takes any_numeric_ea_dtype fixture; parametrized over (expected values, join
        # kind). Verifies merging two one-row frames with a nullable numeric key keeps the
        # extension dtype and gives the expected rows with a range index (GH 40073).
        # GH#40073
        d1 = DataFrame([(1,)], columns=["id"], dtype=any_numeric_ea_dtype)
        d2 = DataFrame([(2,)], columns=["id"], dtype=any_numeric_ea_dtype)
        result = merge(d1, d2, how=how)
        exp_index = RangeIndex(len(expected_data))
        expected = DataFrame(
            expected_data, index=exp_index, columns=["id"], dtype=any_numeric_ea_dtype
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "expected_data, how",
        [
            (["a", "b"], "outer"),
            ([], "inner"),
            (["b"], "right"),
            (["a"], "left"),
        ],
    )
    def test_merge_string_dtype(self, how, expected_data, any_string_dtype):
        # [FUNCTION DOC]
        # Same structure with string keys using the any_string_dtype fixture; verifies the
        # string dtype is preserved in the result column (GH 40073).
        # GH#40073
        d1 = DataFrame([("a",)], columns=["id"], dtype=any_string_dtype)
        d2 = DataFrame([("b",)], columns=["id"], dtype=any_string_dtype)
        result = merge(d1, d2, how=how)
        exp_idx = RangeIndex(len(expected_data))
        expected = DataFrame(
            expected_data, index=exp_idx, columns=["id"], dtype=any_string_dtype
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "how, expected_data",
        [
            ("inner", [[True, 1, 4], [False, 5, 3]]),
            ("outer", [[False, 5, 3], [True, 1, 4]]),
            ("left", [[True, 1, 4], [False, 5, 3]]),
            ("right", [[False, 5, 3], [True, 1, 4]]),
        ],
    )
    def test_merge_bool_dtype(self, how, expected_data):
        # [FUNCTION DOC]
        # Parametrized over join kind and expected rows. Verifies merging on a bool key
        # column keeps bool dtype and orders rows as expected (GH 40073).
        # GH#40073
        df1 = DataFrame({"A": [True, False], "B": [1, 5]})
        df2 = DataFrame({"A": [False, True], "C": [3, 4]})
        result = merge(df1, df2, how=how)
        expected = DataFrame(expected_data, columns=["A", "B", "C"])
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_with_string(self, join_type, string_dtype):
        # [FUNCTION DOC]
        # Takes join_type and string_dtype fixtures. Verifies merging frames with
        # two-level column labels on one tuple key column, where the left uses the default
        # string extension dtype and the right a fixture string dtype: inputs unchanged and
        # result dtypes are object for the key, string extension for the left-only
        # column and object for the right-only column (GH 43734).
        # GH 43734 Avoid the use of `assign` with multi-index
        df1 = DataFrame(
            data={
                ("lvl0", "lvl1-a"): ["1", "2", "3", "4", None],
                ("lvl0", "lvl1-b"): ["4", "5", "6", "7", "8"],
            },
            dtype=pd.StringDtype(),
        )
        df1_copy = df1.copy()
        df2 = DataFrame(
            data={
                ("lvl0", "lvl1-a"): ["1", "2", "3", pd.NA, "5"],
                ("lvl0", "lvl1-c"): ["7", "8", "9", pd.NA, "11"],
            },
            dtype=string_dtype,
        )
        df2_copy = df2.copy()
        merged = merge(left=df1, right=df2, on=[("lvl0", "lvl1-a")], how=join_type)

        # No change in df1 and df2
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # Check the expected types for the merged data frame
        expected = Series(
            [np.dtype("O"), pd.StringDtype(), np.dtype("O")],
            index=MultiIndex.from_tuples(
                [("lvl0", "lvl1-a"), ("lvl0", "lvl1-b"), ("lvl0", "lvl1-c")]
            ),
        )
        tm.assert_series_equal(merged.dtypes, expected)

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    def test_merge_empty(self, left_empty, how, exp):
        # [FUNCTION DOC]
        # Parametrized over (left empty, join kind, expected outcome). Verifies merges where
        # one side is empty: left or right rows with missing fill, empty frames with the
        # combined columns (cross join columns carry suffixes), and outer results sorted by
        # the key.
        left = DataFrame({"A": [2, 1], "B": [3, 4]})
        right = DataFrame({"A": [1], "C": [5]}, dtype="int64")

        if left_empty:
            left = left.head(0)
        else:
            right = right.head(0)

        result = left.merge(right, how=how)

        if exp == "left":
            expected = DataFrame({"A": [2, 1], "B": [3, 4], "C": [np.nan, np.nan]})
        elif exp == "right":
            expected = DataFrame({"A": [1], "B": [np.nan], "C": [5]})
        elif exp == "empty":
            expected = DataFrame(columns=["A", "B", "C"], dtype="int64")
        elif exp == "empty_cross":
            expected = DataFrame(columns=["A_x", "B", "A_y", "C"], dtype="int64")

        if how == "outer":
            expected = expected.sort_values("A", ignore_index=True)

        tm.assert_frame_equal(result, expected)

    def test_merge_with_uintc_columns(self):
        # [FUNCTION DOC]
        # Verifies an outer merge on a string column and a C unsigned int column preserves
        # the unsigned C int dtype and sorted order of the keys.
        df1 = DataFrame({"a": ["foo", "bar"], "b": np.array([1, 2], dtype=np.uintc)})
        df2 = DataFrame({"a": ["foo", "baz"], "b": np.array([3, 4], dtype=np.uintc)})
        result = df1.merge(df2, how="outer")
        expected = DataFrame(
            {
                "a": ["bar", "baz", "foo", "foo"],
                "b": np.array([2, 4, 1, 3], dtype=np.uintc),
            }
        )
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_merge_with_intc_columns(self):
        # [FUNCTION DOC]
        # Same as above for the C int dtype.
        df1 = DataFrame({"a": ["foo", "bar"], "b": np.array([1, 2], dtype=np.intc)})
        df2 = DataFrame({"a": ["foo", "baz"], "b": np.array([3, 4], dtype=np.intc)})
        result = df1.merge(df2, how="outer")
        expected = DataFrame(
            {
                "a": ["bar", "baz", "foo", "foo"],
                "b": np.array([2, 4, 1, 3], dtype=np.intc),
            }
        )
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_merge_intc_non_monotonic(self):
        # [FUNCTION DOC]
        # Verifies a left merge on a non-monotonic C int key preserves left order and the
        # dtype.
        df = DataFrame({"join_key": Series([0, 2, 1], dtype=np.intc)})
        df_details = DataFrame(
            {"join_key": Series([0, 1, 2], dtype=np.intc), "value": ["a", "b", "c"]}
        )
        merged = df.merge(df_details, on="join_key", how="left")
        expected = DataFrame(
            {"join_key": np.array([0, 2, 1], dtype=np.intc), "value": ["a", "c", "b"]}
        )
        tm.assert_frame_equal(merged.reset_index(drop=True), expected)


@pytest.fixture
def left():
    # [FUNCTION DOC]
    # HELPER: module-level fixture returning a ten-row frame with a categorical
    # column X (categories foo, bar; random values) and a random string column Y
    # (one, two, three). Used by the categorical merge tests.
    return DataFrame(
        {
            "X": Series(
                np.random.default_rng(2).choice(["foo", "bar"], size=(10,))
            ).astype(CategoricalDtype(["foo", "bar"])),
            "Y": np.random.default_rng(2).choice(["one", "two", "three"], size=(10,)),
        }
    )


@pytest.fixture
def right():
    # [FUNCTION DOC]
    # HELPER: module-level fixture returning a two-row frame with categorical column X
    # (categories foo, bar; values foo and bar) and integer column Z.
    return DataFrame(
        {
            "X": Series(["foo", "bar"]).astype(CategoricalDtype(["foo", "bar"])),
            "Z": [1, 2],
        }
    )


class TestMergeCategorical:
    # [CLASS DOC]
    # Collection of tests of merging frames with categorical columns: dtype
    # preservation, equal and unequal category sets, ordering and non-key columns.
    # Uses module-level fixtures left and right.
    def test_identical(self, left, using_infer_string):
        # [FUNCTION DOC]
        # Verifies merging a frame with itself on a categorical key preserves the
        # categorical dtype for the key and the object (or str) dtype for other columns.
        # merging on the same, should preserve dtypes
        merged = merge(left, left, on="X")
        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series(
            [CategoricalDtype(categories=["foo", "bar"]), dtype, dtype],
            index=["X", "Y_x", "Y_y"],
        )
        tm.assert_series_equal(result, expected)

    def test_basic(self, left, right, using_infer_string):
        # [FUNCTION DOC]
        # Verifies merging on a categorical key with identical categories keeps categorical
        # dtype for the key, object (or str) for the string column and integer for Z.
        # we have matching Categorical dtypes in X
        # so should preserve the merged column
        merged = merge(left, right, on="X")
        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series(
            [
                CategoricalDtype(categories=["foo", "bar"]),
                dtype,
                np.dtype("int64"),
            ],
            index=["X", "Y", "Z"],
        )
        tm.assert_series_equal(result, expected)

    def test_merge_categorical(self):
        # [FUNCTION DOC]
        # Verifies left merges on object columns with object or categorical columns (GH 9426):
        # the right category column is propagated as categorical with its categories even
        # when no rows match, for combinations of object and categorical keys.
        # GH 9426

        right = DataFrame(
            {
                "c": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
                "d": {0: "null", 1: "null", 2: "null", 3: "null", 4: "null"},
            }
        )
        left = DataFrame(
            {
                "a": {0: "f", 1: "f", 2: "f", 3: "f", 4: "f"},
                "b": {0: "g", 1: "g", 2: "g", 3: "g", 4: "g"},
            }
        )
        df = merge(left, right, how="left", left_on="b", right_on="c")

        # object-object
        expected = df.copy()

        # object-cat
        # note that we propagate the category
        # because we don't have any matching rows
        cright = right.copy()
        cright["d"] = cright["d"].astype("category")
        result = merge(left, cright, how="left", left_on="b", right_on="c")
        expected["d"] = expected["d"].astype(CategoricalDtype(["null"]))
        tm.assert_frame_equal(result, expected)

        # cat-object
        cleft = left.copy()
        cleft["b"] = cleft["b"].astype("category")
        result = merge(cleft, cright, how="left", left_on="b", right_on="c")
        tm.assert_frame_equal(result, expected)

        # cat-cat
        cright = right.copy()
        cright["d"] = cright["d"].astype("category")
        cleft = left.copy()
        cleft["b"] = cleft["b"].astype("category")
        result = merge(cleft, cright, how="left", left_on="b", right_on="c")
        tm.assert_frame_equal(result, expected)

    def tests_merge_categorical_unordered_equal(self):
        # [FUNCTION DOC]
        # Verifies merging on categorical columns with the same categories in different
        # order works and the result keeps the left category order (GH 19551).
        # GH-19551
        df1 = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
                "Left": ["A0", "B0", "C0"],
            }
        )

        df2 = DataFrame(
            {
                "Foo": Categorical(["C", "B", "A"], categories=["C", "B", "A"]),
                "Right": ["C1", "B1", "A1"],
            }
        )
        result = merge(df1, df2, on=["Foo"])
        expected = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"]),
                "Left": ["A0", "B0", "C0"],
                "Right": ["A1", "B1", "C1"],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_multiindex_merge_with_unordered_categoricalindex(self, ordered):
        # [FUNCTION DOC]
        # Parametrized by ordered flag. Verifies a left merge on indexes (id and a
        # categorical level with unsorted categories) gives the expected rows and missing
        # values (GH 36973).
        # GH 36973
        pcat = CategoricalDtype(categories=["P2", "P1"], ordered=ordered)
        df1 = DataFrame(
            {
                "id": ["C", "C", "D"],
                "p": Categorical(["P2", "P1", "P2"], dtype=pcat),
                "a": [0, 1, 2],
            }
        ).set_index(["id", "p"])
        df2 = DataFrame(
            {
                "id": ["A", "C", "C"],
                "p": Categorical(["P2", "P2", "P1"], dtype=pcat),
                "d1": [10, 11, 12],
            }
        ).set_index(["id", "p"])
        result = merge(df1, df2, how="left", left_index=True, right_index=True)
        expected = DataFrame(
            {
                "id": ["C", "C", "D"],
                "p": Categorical(["P2", "P1", "P2"], dtype=pcat),
                "a": [0, 1, 2],
                "d1": [11.0, 12.0, np.nan],
            }
        ).set_index(["id", "p"])
        tm.assert_frame_equal(result, expected)

    def test_other_columns(self, left, right, using_infer_string):
        # [FUNCTION DOC]
        # Verifies non-key categorical columns remain categorical in the merge result and
        # the categories are preserved up to permutation.
        # non-merge columns should preserve if possible
        right = right.assign(Z=right.Z.astype("category"))

        merged = merge(left, right, on="X")
        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series(
            [
                CategoricalDtype(categories=["foo", "bar"]),
                dtype,
                CategoricalDtype(categories=[1, 2]),
            ],
            index=["X", "Y", "Z"],
        )
        tm.assert_series_equal(result, expected)

        # categories are preserved
        assert left.X.values._categories_match_up_to_permutation(merged.X.values)
        assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)

    @pytest.mark.parametrize(
        "change",
        [
            lambda x: x,
            lambda x: x.astype(CategoricalDtype(["foo", "bar", "bah"])),
            lambda x: x.astype(CategoricalDtype(ordered=True)),
        ],
    )
    def test_dtype_on_merged_different(
        self, change, join_type, left, right, using_infer_string
    ):
        # [FUNCTION DOC]
        # Takes the join_type fixture; parametrized over three transforms (identity, a
        # categorical with extra category, ordered categorical) applied to the right key
        # cast to object. Verifies that when key dtypes differ the merged key column and
        # the other string column are object (or str) and Z stays integer. The comment
        # contains a disabled assertion.
        # our merging columns, X now has 2 different dtypes
        # so we must be object as a result

        X = change(right.X.astype("object"))
        right = right.assign(X=X)
        assert isinstance(left.X.values.dtype, CategoricalDtype)
        # assert not left.X.values._categories_match_up_to_permutation(right.X.values)

        merged = merge(left, right, on="X", how=join_type)

        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series([dtype, dtype, np.dtype("int64")], index=["X", "Y", "Z"])
        tm.assert_series_equal(result, expected)

    def test_self_join_multiple_categories(self):
        # [FUNCTION DOC]
        # Verifies that self-merging a frame whose four columns are all categorical on all
        # columns returns the frame unchanged (GH 16767).
        # GH 16767
        # non-duplicates should work with multiple categories
        m = 5
        df = DataFrame(
            {
                "a": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * m,
                "b": ["t", "w", "x", "y", "z"] * 2 * m,
                "c": [
                    letter
                    for each in ["m", "n", "u", "p", "o"]
                    for letter in [each] * 2 * m
                ],
                "d": [
                    letter
                    for each in [
                        "aa",
                        "bb",
                        "cc",
                        "dd",
                        "ee",
                        "ff",
                        "gg",
                        "hh",
                        "ii",
                        "jj",
                    ]
                    for letter in [each] * m
                ],
            }
        )

        # change them all to categorical variables
        df = df.apply(lambda x: x.astype("category"))

        # self-join should equal ourselves
        result = merge(df, df, on=list(df.columns))

        tm.assert_frame_equal(result, df)

    def test_dtype_on_categorical_dates(self):
        # [FUNCTION DOC]
        # Verifies outer and inner merges on a categorical column of date objects keep dates
        # (not integers) in the result (GH 16900).
        # GH 16900
        # dates should not be coerced to ints

        df = DataFrame(
            [[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=["date", "num2"]
        )
        df["date"] = df["date"].astype("category")

        df2 = DataFrame(
            [[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=["date", "num4"]
        )
        df2["date"] = df2["date"].astype("category")

        expected_outer = DataFrame(
            [
                [pd.Timestamp("2001-01-01").date(), 1.1, 1.3],
                [pd.Timestamp("2001-01-02").date(), 1.3, np.nan],
                [pd.Timestamp("2001-01-03").date(), np.nan, 1.4],
            ],
            columns=["date", "num2", "num4"],
        )
        result_outer = merge(df, df2, how="outer", on=["date"])
        tm.assert_frame_equal(result_outer, expected_outer)

        expected_inner = DataFrame(
            [[pd.Timestamp("2001-01-01").date(), 1.1, 1.3]],
            columns=["date", "num2", "num4"],
        )
        result_inner = merge(df, df2, how="inner", on=["date"])
        tm.assert_frame_equal(result_inner, expected_inner)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "category_column,categories,expected_categories",
        [
            ([False, True, True, False], [True, False], [True, False]),
            ([2, 1, 1, 2], [1, 2], [1, 2]),
            (["False", "True", "True", "False"], ["True", "False"], ["True", "False"]),
        ],
    )
    def test_merging_with_bool_or_int_cateorical_column(
        self, category_column, categories, expected_categories, ordered
    ):
        # [FUNCTION DOC]
        # Parametrized by ordered flag and by categorical column data (bools, ints, bool-like
        # strings) with categories. Verifies that merging on the shared id column keeps a
        # categorical column of bool/int/string categories with its categories and
        # ordering (GH 17187). The function name misspells categorical.
        # GH 17187
        # merging with a boolean/int categorical column
        df1 = DataFrame({"id": [1, 2, 3, 4], "cat": category_column})
        df1["cat"] = df1["cat"].astype(CategoricalDtype(categories, ordered=ordered))
        df2 = DataFrame({"id": [2, 4], "num": [1, 9]})
        result = df1.merge(df2)
        expected = DataFrame({"id": [2, 4], "cat": expected_categories, "num": [1, 9]})
        expected["cat"] = expected["cat"].astype(
            CategoricalDtype(categories, ordered=ordered)
        )
        tm.assert_frame_equal(expected, result)

    def test_merge_on_int_array(self):
        # [FUNCTION DOC]
        # Verifies a self-merge on a nullable integer column containing a missing value
        # matches the missing value with itself and keeps the nullable dtype (GH 23020).
        # GH 23020
        df = DataFrame({"A": Series([1, 2, pd.NA], dtype="Int64"), "B": 1})
        result = merge(df, df, on="A")
        expected = DataFrame(
            {"A": Series([1, 2, pd.NA], dtype="Int64"), "B_x": 1, "B_y": 1}
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_category_index_levels_stay_category(self):
        # [FUNCTION DOC]
        # Verifies an outer merge on indexes of two grouped frames (multi-level indexes
        # with categorical levels) keeps categorical dtypes for all levels, with missing
        # codes in the level absent from the left (GH 37480).
        # GH#37480
        df1 = DataFrame(
            {
                "x": list(range(1, 10)),
                "y": list(range(1, 10)),
                "z": list(range(1, 10)),
                "d": list(range(1, 10)),
            }
        )

        df2 = df1.astype({"x": "category", "y": "category", "z": "category"})

        df3 = df2.iloc[:4, :].groupby(["z", "x"], observed=True).agg({"d": "sum"})
        df4 = df2.iloc[7:, :].groupby(["z", "x", "y"], observed=True).agg({"d": "sum"})
        result = merge(df3, df4, left_index=True, right_index=True, how="outer")

        cats = list(range(1, 10))
        z = Categorical(list(range(1, 5)) + list(range(8, 10)), categories=cats)
        x = Categorical(list(range(1, 5)) + list(range(8, 10)), categories=cats)
        y = Categorical([np.nan] * 4 + list(range(8, 10)), categories=cats)
        idx = MultiIndex.from_arrays([z, x, y], names=["z", "x", "y"])
        expected = DataFrame(
            {
                "d_x": list(map(float, range(1, 5))) + [np.nan] * 2,
                "d_y": [np.nan] * 4 + list(map(float, range(8, 10))),
            },
            index=idx,
        )
        tm.assert_frame_equal(result, expected)


class TestMergeOnIndexes:
    # [CLASS DOC]
    # Collection of tests of merges using both indexes as keys. Stateless.
    @pytest.mark.parametrize(
        "how, sort, expected",
        [
            ("inner", False, DataFrame({"a": [20, 10], "b": [200, 100]}, index=[2, 1])),
            ("inner", True, DataFrame({"a": [10, 20], "b": [100, 200]}, index=[1, 2])),
            (
                "left",
                False,
                DataFrame({"a": [20, 10, 0], "b": [200, 100, np.nan]}, index=[2, 1, 0]),
            ),
            (
                "left",
                True,
                DataFrame({"a": [0, 10, 20], "b": [np.nan, 100, 200]}, index=[0, 1, 2]),
            ),
            (
                "right",
                False,
                DataFrame(
                    {"a": [np.nan, 10, 20], "b": [300, 100, 200]}, index=[3, 1, 2]
                ),
            ),
            (
                "right",
                True,
                DataFrame(
                    {"a": [10, 20, np.nan], "b": [100, 200, 300]}, index=[1, 2, 3]
                ),
            ),
            (
                "outer",
                False,
                DataFrame(
                    {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                    index=[0, 1, 2, 3],
                ),
            ),
            (
                "outer",
                True,
                DataFrame(
                    {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                    index=[0, 1, 2, 3],
                ),
            ),
        ],
    )
    def test_merge_on_indexes(self, how, sort, expected):
        # [FUNCTION DOC]
        # Parametrized over (join kind, sort flag, expected frame). Verifies merging two
        # frames on their indexes gives the expected rows: order of the left (or right) frame
        # without sorting, ascending key order with sorting, and for outer join always
        # sorted.
        left_df = DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
        right_df = DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])
        result = merge(
            left_df, right_df, left_index=True, right_index=True, how=how, sort=sort
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        Index([1, 2, 4], dtype=dtyp, name="index_col")
        for dtyp in tm.ALL_REAL_NUMPY_DTYPES
    ]
    + [
        CategoricalIndex(["A", "B", "C"], categories=["A", "B", "C"], name="index_col"),
        RangeIndex(start=0, stop=3, name="index_col"),
        DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-03"], name="index_col"),
    ],
    ids=lambda x: f"{type(x).__name__}[{x.dtype}]",
)
def test_merge_index_types(index):
    # [FUNCTION DOC]
    # Parametrized over index kinds (every real numpy dtype, categorical, range,
    # datetime), all named index_col. Verifies merging on the index name works for each
    # type and the result keeps the index (GH 20777).
    # gh-20777
    # assert key access is consistent across index types
    left = DataFrame({"left_data": [1, 2, 3]}, index=index)
    right = DataFrame({"right_data": [1.0, 2.0, 3.0]}, index=index)

    result = left.merge(right, on=["index_col"])

    expected = DataFrame(
        {"left_data": [1, 2, 3], "right_data": [1.0, 2.0, 3.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "on,left_on,right_on,left_index,right_index,nm",
    [
        (["outer", "inner"], None, None, False, False, "B"),
        (None, None, None, True, True, "B"),
        (None, ["outer", "inner"], None, False, True, "B"),
        (None, None, ["outer", "inner"], True, False, "B"),
        (["outer", "inner"], None, None, False, False, None),
        (None, None, None, True, True, None),
        (None, ["outer", "inner"], None, False, True, None),
        (None, None, ["outer", "inner"], True, False, None),
    ],
)
def test_merge_series(on, left_on, right_on, left_index, right_index, nm):
    # [FUNCTION DOC]
    # Parametrized over key specifications (on, left_on, right_on, index flags) and a
    # series name or no name. Verifies merging a frame with a named series, using the
    # multi-level index names, gives the expected frame; an unnamed series raises a
    # value error saying a series cannot be merged without a name (GH 21220).
    # GH 21220
    a = DataFrame(
        {"A": [1, 2, 3, 4]},
        index=MultiIndex.from_product([["a", "b"], [0, 1]], names=["outer", "inner"]),
    )
    b = Series(
        [1, 2, 3, 4],
        index=MultiIndex.from_product([["a", "b"], [1, 2]], names=["outer", "inner"]),
        name=nm,
    )
    expected = DataFrame(
        {"A": [2, 4], "B": [1, 3]},
        index=MultiIndex.from_product([["a", "b"], [1]], names=["outer", "inner"]),
    )
    if nm is not None:
        result = merge(
            a,
            b,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
        )
        tm.assert_frame_equal(result, expected)
    else:
        msg = "Cannot merge a Series without a name"
        with pytest.raises(ValueError, match=msg):
            result = merge(
                a,
                b,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
            )


def test_merge_series_multilevel():
    # [FUNCTION DOC]
    # Verifies a merge error saying merging between different levels is not allowed when
    # the series name is a tuple (multi-level) and the frame has flat columns
    # (GH 47946, GH 40993).
    # GH#47946
    # GH 40993: For raising, enforced in 2.0
    a = DataFrame(
        {"A": [1, 2, 3, 4]},
        index=MultiIndex.from_product([["a", "b"], [0, 1]], names=["outer", "inner"]),
    )
    b = Series(
        [1, 2, 3, 4],
        index=MultiIndex.from_product([["a", "b"], [1, 2]], names=["outer", "inner"]),
        name=("B", "C"),
    )
    with pytest.raises(
        MergeError, match="Not allowed to merge between different levels"
    ):
        merge(a, b, on=["outer", "inner"])


@pytest.mark.parametrize(
    "col1, col2, kwargs, expected_cols",
    [
        (0, 0, {"suffixes": ("", "_dup")}, ["0", "0_dup"]),
        (0, 0, {"suffixes": (None, "_dup")}, [0, "0_dup"]),
        (0, 0, {"suffixes": ("_x", "_y")}, ["0_x", "0_y"]),
        (0, 0, {"suffixes": ["_x", "_y"]}, ["0_x", "0_y"]),
        ("a", 0, {"suffixes": (None, "_y")}, ["a", 0]),
        (0.0, 0.0, {"suffixes": ("_x", None)}, ["0.0_x", 0.0]),
        ("b", "b", {"suffixes": (None, "_y")}, ["b", "b_y"]),
        ("a", "a", {"suffixes": ("_x", None)}, ["a_x", "a"]),
        ("a", "b", {"suffixes": ("_x", None)}, ["a", "b"]),
        ("a", "a", {"suffixes": (None, "_x")}, ["a", "a_x"]),
        (0, 0, {"suffixes": ("_a", None)}, ["0_a", 0]),
        ("a", "a", {}, ["a_x", "a_y"]),
        (0, 0, {}, ["0_x", "0_y"]),
    ],
)
def test_merge_suffix(col1, col2, kwargs, expected_cols):
    # [FUNCTION DOC]
    # Parametrized over column labels of the two frames, suffix arguments (including
    # none entries, empty strings, lists) and expected output labels. Verifies the
    # suffix rules for overlapping labels: suffix applied only to overlapping labels, a
    # none suffix leaves that side unchanged, defaults apply when not given, non-string
    # labels become strings when suffixed; both the method and function forms agree
    # (GH 24782).
    # issue: 24782
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [4, 5, 6]})

    expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=expected_cols)

    result = a.merge(b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)

    result = merge(a, b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "how,expected",
    [
        (
            "right",
            {"A": [100, 200, 300], "B1": [60, 70, np.nan], "B2": [600, 700, 800]},
        ),
        (
            "outer",
            {
                "A": [1, 100, 200, 300],
                "B1": [80, 60, 70, np.nan],
                "B2": [np.nan, 600, 700, 800],
            },
        ),
    ],
)
def test_merge_duplicate_suffix(how, expected):
    # [FUNCTION DOC]
    # Parametrized over right and outer join. Verifies that identical suffixes on both
    # sides yield duplicate column labels with the expected values.
    left_df = DataFrame({"A": [100, 200, 1], "B": [60, 70, 80]})
    right_df = DataFrame({"A": [100, 200, 300], "B": [600, 700, 800]})
    result = merge(left_df, right_df, on="A", how=how, suffixes=("_x", "_x"))
    expected = DataFrame(expected)
    expected.columns = ["A", "B_x", "B_x"]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "col1, col2, suffixes",
    [("a", "a", (None, None)), ("a", "a", ("", None)), (0, 0, (None, ""))],
)
def test_merge_suffix_error(col1, col2, suffixes):
    # [FUNCTION DOC]
    # Parametrized over overlapping labels with suffix pairs lacking a suffix on both
    # sides effectively. Verifies a value error saying columns overlap but no suffix
    # was specified (GH 24782). The code has a TODO to reconsider the behavior.
    # issue: 24782
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [3, 4, 5]})

    # TODO: might reconsider current raise behaviour, see issue 24782
    msg = "columns overlap but no suffix specified"
    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


@pytest.mark.parametrize("suffixes", [{"left", "right"}, {"left": 0, "right": 0}])
def test_merge_suffix_raises(suffixes):
    # [FUNCTION DOC]
    # Parametrized over a set and a mapping used as suffixes. Verifies a type error that
    # suffixes cannot be passed as that type.
    a = DataFrame({"a": [1, 2, 3]})
    b = DataFrame({"b": [3, 4, 5]})

    with pytest.raises(TypeError, match="Passing 'suffixes' as a"):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


TWO_GOT_THREE = "2, got 3" if PY314 else "2"


@pytest.mark.parametrize(
    "col1, col2, suffixes, msg",
    [
        (
            "a",
            "a",
            ("a", "b", "c"),
            (rf"too many values to unpack \(expected {TWO_GOT_THREE}\)"),
        ),
        ("a", "a", tuple("a"), r"not enough values to unpack \(expected 2, got 1\)"),
    ],
)
def test_merge_suffix_length_error(col1, col2, suffixes, msg):
    # [FUNCTION DOC]
    # Parametrized over a three-element and a one-element suffix tuple. Verifies a value
    # error from unpacking the suffix tuple (too many or not enough values; the message
    # includes the actual count on Python 3.14 and later through the module constant).
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [3, 4, 5]})

    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


@pytest.mark.parametrize("cat_dtype", ["one", "two"])
@pytest.mark.parametrize("reverse", [True, False])
def test_merge_equal_cat_dtypes(cat_dtype, reverse):
    # [FUNCTION DOC]
    # Parametrized over which equal categorical dtype object the right frame uses and
    # whether its rows are reversed. Verifies merging on categorical indexes with equal
    # dtypes aligns by key and keeps the index dtype (GH 22501).
    # see gh-22501
    cat_dtypes = {
        "one": CategoricalDtype(categories=["a", "b", "c"], ordered=False),
        "two": CategoricalDtype(categories=["a", "b", "c"], ordered=False),
    }

    df1 = DataFrame(
        {"foo": Series(["a", "b", "c"]).astype(cat_dtypes["one"]), "left": [1, 2, 3]}
    ).set_index("foo")

    data_foo = ["a", "b", "c"]
    data_right = [1, 2, 3]

    if reverse:
        data_foo.reverse()
        data_right.reverse()

    df2 = DataFrame(
        {"foo": Series(data_foo).astype(cat_dtypes[cat_dtype]), "right": data_right}
    ).set_index("foo")

    result = df1.merge(df2, left_index=True, right_index=True)

    expected = DataFrame(
        {
            "left": [1, 2, 3],
            "right": [1, 2, 3],
            "foo": Series(["a", "b", "c"]).astype(cat_dtypes["one"]),
        }
    ).set_index("foo")

    tm.assert_frame_equal(result, expected)


def test_merge_equal_cat_dtypes2():
    # [FUNCTION DOC]
    # Verifies inner merge on categorical indexes with identical unordered dtypes, with
    # different subsets of categories present, gives the matched rows and keeps the
    # categorical index (GH 22501).
    # see gh-22501
    cat_dtype = CategoricalDtype(categories=["a", "b", "c"], ordered=False)

    # Test Data
    df1 = DataFrame(
        {"foo": Series(["a", "b"]).astype(cat_dtype), "left": [1, 2]}
    ).set_index("foo")

    df2 = DataFrame(
        {"foo": Series(["a", "b", "c"]).astype(cat_dtype), "right": [3, 2, 1]}
    ).set_index("foo")

    result = df1.merge(df2, left_index=True, right_index=True)

    expected = DataFrame(
        {"left": [1, 2], "right": [3, 2], "foo": Series(["a", "b"]).astype(cat_dtype)}
    ).set_index("foo")

    tm.assert_frame_equal(result, expected)


def test_merge_on_cat_and_ext_array():
    # [FUNCTION DOC]
    # Verifies merging a categorical column of intervals with an interval-dtype column
    # on that column gives the interval-dtype values (GH 28668).
    # GH 28668
    right = DataFrame(
        {"a": Series([pd.Interval(0, 1), pd.Interval(1, 2)], dtype="interval")}
    )
    left = right.copy()
    left["a"] = left["a"].astype("category")

    result = merge(left, right, how="inner", on="a")
    expected = right.copy()

    tm.assert_frame_equal(result, expected)


def test_merge_multiindex_columns():
    # [FUNCTION DOC]
    # Verifies merging empty frames with two-level column labels on a column with default
    # suffixes gives suffixed first-level labels with inner labels kept (GH 28518).
    # Issue #28518
    # Verify that merging two dataframes give the expected labels
    # The original cause of this issue come from a bug lexsort_depth and is tested in
    # test_lexsort_depth

    letters = ["a", "b", "c", "d"]
    numbers = ["1", "2", "3"]
    index = MultiIndex.from_product((letters, numbers), names=["outer", "inner"])

    frame_x = DataFrame(columns=index)
    frame_x["id"] = ""
    frame_y = DataFrame(columns=index)
    frame_y["id"] = ""

    l_suf = "_x"
    r_suf = "_y"
    result = frame_x.merge(frame_y, on="id", suffixes=((l_suf, r_suf)))

    # Constructing the expected results
    tuples = [(letter + l_suf, num) for letter in letters for num in numbers]
    tuples += [("id", "")]
    tuples += [(letter + r_suf, num) for letter in letters for num in numbers]

    expected_index = MultiIndex.from_tuples(tuples, names=["outer", "inner"])
    expected = DataFrame(columns=expected_index)

    tm.assert_frame_equal(result, expected, check_dtype=False)


def test_merge_datetime_upcast_dtype():
    # [FUNCTION DOC]
    # Verifies a left merge where the right datetime column has no match for one row
    # keeps datetime dtype with NaT (GH 31208).
    # https://github.com/pandas-dev/pandas/issues/31208
    df1 = DataFrame({"x": ["a", "b", "c"], "y": ["1", "2", "4"]})
    df2 = DataFrame(
        {"y": ["1", "2", "3"], "z": pd.to_datetime(["2000", "2001", "2002"])}
    )
    result = merge(df1, df2, how="left", on="y")
    expected = DataFrame(
        {
            "x": ["a", "b", "c"],
            "y": ["1", "2", "4"],
            "z": pd.to_datetime(["2000", "2001", "NaT"]),
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n_categories", [5, 128])
def test_categorical_non_unique_monotonic(n_categories):
    # [FUNCTION DOC]
    # Parametrized over the number of categories (5 and 128, to exercise 8-bit and
    # 16-bit category code paths). Verifies a left merge on indexes where the left
    # categorical index is non-unique and monotonic gives the expected values (GH 28189).
    # GH 28189
    # With n_categories as 5, we test the int8 case is hit in libjoin,
    # with n_categories as 128 we test the int16 case.
    left_index = CategoricalIndex([0, *list(range(n_categories))])
    df1 = DataFrame(range(n_categories + 1), columns=["value"], index=left_index)
    df2 = DataFrame(
        [[6]],
        columns=["value"],
        index=CategoricalIndex([0], categories=list(range(n_categories))),
    )

    result = merge(df1, df2, how="left", left_index=True, right_index=True)
    expected = DataFrame(
        [[i, 6.0] if i < 2 else [i, np.nan] for i in range(n_categories + 1)],
        columns=["value_x", "value_y"],
        index=left_index,
    )
    tm.assert_frame_equal(expected, result)


def test_merge_join_categorical_multiindex():
    # [FUNCTION DOC]
    # Verifies that joining a frame with two columns (categorical and integer) onto a
    # series with a two-level index (categorical, integer) on those columns equals a left
    # merge on the reset index with the extra key columns dropped; repeated for ordered
    # categoricals with a non-alphabetical category order (issue 16627).
    # From issue 16627
    a = {
        "Cat1": Categorical(["a", "b", "a", "c", "a", "b"], ["a", "b", "c"]),
        "Int1": [0, 1, 0, 1, 0, 0],
    }
    a = DataFrame(a)

    b = {
        "Cat": Categorical(["a", "b", "c", "a", "b", "c"], ["a", "b", "c"]),
        "Int": [0, 0, 0, 1, 1, 1],
        "Factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    b = DataFrame(b).set_index(["Cat", "Int"])["Factor"]

    expected = merge(
        a,
        b.reset_index(),
        left_on=["Cat1", "Int1"],
        right_on=["Cat", "Int"],
        how="left",
    )
    expected = expected.drop(["Cat", "Int"], axis=1)
    result = a.join(b, on=["Cat1", "Int1"])
    tm.assert_frame_equal(expected, result)

    # Same test, but with ordered categorical
    a = {
        "Cat1": Categorical(
            ["a", "b", "a", "c", "a", "b"], ["b", "a", "c"], ordered=True
        ),
        "Int1": [0, 1, 0, 1, 0, 0],
    }
    a = DataFrame(a)

    b = {
        "Cat": Categorical(
            ["a", "b", "c", "a", "b", "c"], ["b", "a", "c"], ordered=True
        ),
        "Int": [0, 0, 0, 1, 1, 1],
        "Factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    b = DataFrame(b).set_index(["Cat", "Int"])["Factor"]

    expected = merge(
        a,
        b.reset_index(),
        left_on=["Cat1", "Int1"],
        right_on=["Cat", "Int"],
        how="left",
    )
    expected = expected.drop(["Cat", "Int"], axis=1)
    result = a.join(b, on=["Cat1", "Int1"])
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        ({"left_on": "a", "left_index": True}, ["left_on", "left_index"]),
        ({"right_on": "a", "right_index": True}, ["right_on", "right_index"]),
    ],
)
def test_merge_join_cols_error_reporting_duplicates(func, kwargs, err_msg):
    # [FUNCTION DOC]
    # Parametrized over the merge function name (merge and the as-of variant) and over
    # conflicting keyword pairs (column key with index flag on one side). Verifies a
    # merge error that only one of the two arguments may be passed, naming both (GH 16228).
    # GH: 16228
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    msg = rf'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)


@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        ({"left_on": "a"}, ["right_on", "right_index"]),
        ({"right_on": "a"}, ["left_on", "left_index"]),
    ],
)
def test_merge_join_cols_error_reporting_missing(func, kwargs, err_msg):
    # [FUNCTION DOC]
    # Same parametrization with only one side's column key given. Verifies a merge error
    # that one of the required key arguments of the other side must be passed (GH 16228).
    # GH: 16228
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    msg = rf'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)


@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"right_index": True},
        {"left_index": True},
    ],
)
def test_merge_join_cols_error_reporting_on_and_index(func, kwargs):
    # [FUNCTION DOC]
    # Parametrized over the function name and one index flag. Verifies a merge error when
    # the on argument is combined with an index flag (GH 16228).
    # GH: 16228
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    msg = (
        r'Can only pass argument "on" OR "left_index" '
        r'and "right_index", not a combination of both\.'
    )
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, on="a", **kwargs)


def test_merge_right_left_index():
    # [FUNCTION DOC]
    # Verifies a right merge using the left index and a right key column keeps the key
    # as the first column and suffixes the duplicated columns (GH 38616).
    # GH#38616
    left = DataFrame({"x": [1, 1], "z": ["foo", "foo"]})
    right = DataFrame({"x": [1, 1], "z": ["foo", "foo"]})
    result = merge(left, right, how="right", left_index=True, right_on="x")
    expected = DataFrame(
        {
            "x": [1, 1],
            "x_x": [1, 1],
            "z_x": ["foo", "foo"],
            "x_y": [1, 1],
            "z_y": ["foo", "foo"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_result_empty_index_and_on():
    # [FUNCTION DOC]
    # Verifies merging frames indexed by multi-level and single index on a key mixing
    # columns and index gives an empty integer frame in either order (GH 33814).
    # GH#33814
    df1 = DataFrame({"a": [1], "b": [2]}).set_index(["a", "b"])
    df2 = DataFrame({"b": [1]}).set_index(["b"])
    expected = DataFrame({"a": [], "b": []}, dtype=np.int64).set_index(["a", "b"])
    result = merge(df1, df2, left_on=["b"], right_index=True)
    tm.assert_frame_equal(result, expected)

    result = merge(df2, df1, left_index=True, right_on=["b"])
    tm.assert_frame_equal(result, expected)


def test_merge_suffixes_produce_dup_columns_raises():
    # [FUNCTION DOC]
    # Verifies a merge error saying the given suffixes cause duplicate columns when a
    # suffixed name would collide with an existing column, in both orders (GH 22818).
    # GH#22818; Enforced in 2.0
    left = DataFrame({"a": [1, 2, 3], "b": 1, "b_x": 2})
    right = DataFrame({"a": [1, 2, 3], "b": 2})

    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on="a")

    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(right, left, on="a", suffixes=("_y", "_x"))


def test_merge_duplicate_columns_with_suffix_no_warning():
    # [FUNCTION DOC]
    # Verifies no error when the duplicate labels already exist in the input (suffix
    # yields duplicates only because the input had duplicates) (GH 22818).
    # GH#22818
    # Do not raise warning when duplicates are caused by duplicates in origin
    left = DataFrame([[1, 1, 1], [2, 2, 2]], columns=["a", "b", "b"])
    right = DataFrame({"a": [1, 3], "b": 2})
    result = merge(left, right, on="a")
    expected = DataFrame([[1, 1, 1, 2]], columns=["a", "b_x", "b_x", "b_y"])
    tm.assert_frame_equal(result, expected)


def test_merge_duplicate_columns_with_suffix_causing_another_duplicate_raises():
    # [FUNCTION DOC]
    # Verifies a merge error when suffixing an input with duplicate labels causes a
    # further collision (GH 22818).
    # GH#22818, Enforced in 2.0
    # This should raise warning because suffixes cause another collision
    left = DataFrame([[1, 1, 1, 1], [2, 2, 2, 2]], columns=["a", "b", "b", "b_x"])
    right = DataFrame({"a": [1, 3], "b": 2})
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on="a")


def test_merge_string_float_column_result():
    # [FUNCTION DOC]
    # Verifies a merge on indexes where one frame's columns mix a string and a float
    # label keeps the column labels in order with a mixed object index (GH 13353).
    # GH 13353
    df1 = DataFrame([[1, 2], [3, 4]], columns=Index(["a", 114.0]))
    df2 = DataFrame([[9, 10], [11, 12]], columns=["x", "y"])
    result = merge(df2, df1, how="inner", left_index=True, right_index=True)
    expected = DataFrame(
        [[9, 10, 1, 2], [11, 12, 3, 4]], columns=Index(["x", "y", "a", 114.0])
    )
    tm.assert_frame_equal(result, expected)


def test_mergeerror_on_left_index_mismatched_dtypes():
    # [FUNCTION DOC]
    # Verifies a merge error saying only certain argument combinations may be passed
    # when on is combined with left_index (GH 22449).
    # GH 22449
    df_1 = DataFrame(data=["X"], columns=["C"], index=[22])
    df_2 = DataFrame(data=["X"], columns=["C"], index=[999])
    with pytest.raises(MergeError, match="Can only pass argument"):
        merge(df_1, df_2, on=["C"], left_index=True)


def test_merge_on_left_categoricalindex():
    # [FUNCTION DOC]
    # Verifies passing a categorical index as the left key does not raise, and the
    # result equals passing its underlying categorical array (GH 48464).
    # GH#48464 don't raise when left_on is a CategoricalIndex
    ci = CategoricalIndex(range(3))

    right = DataFrame({"A": ci, "B": range(3)})
    left = DataFrame({"C": range(3, 6)})

    res = merge(left, right, left_on=ci, right_on="A")
    expected = merge(left, right, left_on=ci._data, right_on="A")
    tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize("dtype", [None, "Int64"])
def test_merge_outer_with_NaN(dtype):
    # [FUNCTION DOC]
    # Parametrized on dtype (default or nullable integer). Verifies an outer merge on a
    # key column holding only missing values on one side matches missing keys as
    # separate rows (not joined) in both argument orders (GH 43550).
    # GH#43550
    item = np.nan if dtype is None else pd.NA
    left = DataFrame({"key": [1, 2], "col1": [1, 2]}, dtype=dtype)
    right = DataFrame({"key": [item, item], "col2": [3, 4]}, dtype=dtype)
    result = merge(left, right, on="key", how="outer")
    expected = DataFrame(
        {
            "key": [1, 2, item, item],
            "col1": [1, 2, item, item],
            "col2": [item, item, 3, 4],
        },
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)

    # switch left and right
    result = merge(right, left, on="key", how="outer")
    expected = DataFrame(
        {
            "key": [1, 2, item, item],
            "col2": [item, item, 3, 4],
            "col1": [1, 2, item, item],
        },
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_merge_different_index_names():
    # [FUNCTION DOC]
    # Verifies merging with left_on and right_on naming differently named index levels
    # gives a default index and suffixed overlapping columns (GH 45094).
    # GH#45094
    left = DataFrame({"a": [1]}, index=Index([1], name="c"))
    right = DataFrame({"a": [1]}, index=Index([1], name="d"))
    result = merge(left, right, left_on="c", right_on="d")
    expected = DataFrame({"a_x": [1], "a_y": 1})
    tm.assert_frame_equal(result, expected)


def test_merge_ea(any_numeric_ea_dtype, join_type):
    # [FUNCTION DOC]
    # Takes any_numeric_ea_dtype and join_type fixtures. Verifies merging frames of the
    # same nullable numeric dtype on the shared column keeps the dtype (GH 44240).
    # GH#44240
    left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
    right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype)
    result = left.merge(right, how=join_type)
    expected = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2}, dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)


def test_merge_ea_and_non_ea(any_numeric_ea_dtype, join_type):
    # [FUNCTION DOC]
    # Same with the right frame using the corresponding lower-case numpy dtype name;
    # verifies each column keeps its own dtype and the key keeps the left extension
    # dtype (GH 44240).
    # GH#44240
    left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
    right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype.lower())
    result = left.merge(right, how=join_type)
    expected = DataFrame(
        {
            "a": Series([1, 2, 3], dtype=any_numeric_ea_dtype),
            "b": Series([1, 1, 1], dtype=any_numeric_ea_dtype),
            "c": Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower()),
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "int64[pyarrow]"])
def test_merge_arrow_and_numpy_dtypes(dtype):
    # [FUNCTION DOC]
    # Parametrized over a numpy integer and a pyarrow integer dtype; skipped without
    # pyarrow. Verifies merging a frame with another of pyarrow integer dtype returns
    # the left frame in each order (GH 52406).
    # GH#52406
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [1, 2]}, dtype=dtype)
    df2 = DataFrame({"a": [1, 2]}, dtype="int64[pyarrow]")
    result = df.merge(df2)
    expected = df.copy()
    tm.assert_frame_equal(result, expected)

    result = df2.merge(df)
    expected = df2.copy()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "America/Chicago"])
def test_merge_datetime_different_resolution(tz, join_type):
    # [FUNCTION DOC]
    # Parametrized by timezone (none or Chicago) and join_type. Verifies merging on
    # datetime columns of nanosecond and second resolution matches equal instants and
    # the result key uses the nanosecond unit, for each join kind (GH 53200).
    # https://github.com/pandas-dev/pandas/issues/53200
    vals = [
        pd.Timestamp(2023, 5, 12, tz=tz),
        pd.Timestamp(2023, 5, 13, tz=tz),
        pd.Timestamp(2023, 5, 14, tz=tz),
    ]
    df1 = DataFrame({"t": vals[:2], "a": [1.0, 2.0]})
    df1["t"] = df1["t"].dt.as_unit("ns")
    df2 = DataFrame({"t": vals[1:], "b": [1.0, 2.0]})
    df2["t"] = df2["t"].dt.as_unit("s")

    expected = DataFrame({"t": vals, "a": [1.0, 2.0, np.nan], "b": [np.nan, 1.0, 2.0]})
    expected["t"] = expected["t"].dt.as_unit("ns")
    if join_type == "inner":
        expected = expected.iloc[[1]].reset_index(drop=True)
    elif join_type == "left":
        expected = expected.iloc[[0, 1]]
    elif join_type == "right":
        expected = expected.iloc[[1, 2]].reset_index(drop=True)

    result = df1.merge(df2, on="t", how=join_type)
    tm.assert_frame_equal(result, expected)


def test_merge_multiindex_single_level():
    # [FUNCTION DOC]
    # Verifies a left merge of a column key against a one-level multi-level index works
    # and fills missing for the unmatched key (GH 52331).
    # GH52331
    df = DataFrame({"col": ["A", "B"]})
    df2 = DataFrame(
        data={"b": [100]},
        index=MultiIndex.from_tuples([("A",), ("C",)], names=["col"]),
    )
    expected = DataFrame({"col": ["A", "B"], "b": [100, np.nan]})

    result = df.merge(df2, left_on=["col"], right_index=True, how="left")
    tm.assert_frame_equal(result, expected)


def test_merge_multiindex_reset_index_mixed():
    # [FUNCTION DOC]
    # Verifies merging frames with multi-level column labels after resetting the index,
    # on a plain label and on a tuple label, emits a PerformanceWarning in the first
    # case and gives the cross-product of matching rows (GH 62150).
    # GH#62150
    df = DataFrame(
        {("column_1", ""): [1, 1], ("column_2", ""): [2, 2]},
        index=MultiIndex.from_arrays(
            [[1, 1], ["metadata_1", "metadata_2"]], names=["index", "metadata"]
        ),
    )

    df2 = DataFrame(
        data=[1, 1],
        index=Index([1, 1], name="index"),
        columns=MultiIndex.from_product([["new_data"], [""]]),
    )

    with tm.assert_produces_warning(pd.errors.PerformanceWarning):
        result = df.reset_index().merge(df2.reset_index(), on="index")

    expected = DataFrame(
        {
            ("index", ""): [1, 1, 1, 1],
            ("metadata", ""): ["metadata_1", "metadata_1", "metadata_2", "metadata_2"],
            ("column_1", ""): [1, 1, 1, 1],
            ("column_2", ""): [2, 2, 2, 2],
            ("new_data", ""): [1, 1, 1, 1],
        }
    )
    expected.columns = MultiIndex.from_tuples(expected.columns)

    tm.assert_frame_equal(result, expected)

    result2 = df.reset_index().merge(df2.reset_index(), on=[("index", "")])
    tm.assert_frame_equal(result2, expected)


@pytest.mark.parametrize("on_index", [True, False])
@pytest.mark.parametrize("left_unique", [True, False])
@pytest.mark.parametrize("left_monotonic", [True, False])
@pytest.mark.parametrize("right_unique", [True, False])
@pytest.mark.parametrize("right_monotonic", [True, False])
def test_merge_combinations(
    join_type,
    sort,
    on_index,
    left_unique,
    left_monotonic,
    right_unique,
    right_monotonic,
):
    # [FUNCTION DOC]
    # Parametrized over index versus column keys, uniqueness and monotonicity of each
    # side's keys, plus the join_type and sort fixtures (GH 54611). Builds three-row key
    # columns for both sides, merges, and compares to a reference built with repeat
    # counts: left/right/inner keep the retained side's rows (filtered for inner, sorted
    # if requested) repeated by the other side's key counts; outer takes products of
    # counts, always sorted. Key position (index or column) controls how the expected
    # frame is indexed.
    how = join_type
    # GH 54611
    left = [2, 3]
    if left_unique:
        left.append(4 if left_monotonic else 1)
    else:
        left.append(3 if left_monotonic else 2)

    right = [2, 3]
    if right_unique:
        right.append(4 if right_monotonic else 1)
    else:
        right.append(3 if right_monotonic else 2)

    left = DataFrame({"key": left})
    right = DataFrame({"key": right})

    if on_index:
        left = left.set_index("key")
        right = right.set_index("key")
        on_kwargs = {"left_index": True, "right_index": True}
    else:
        on_kwargs = {"on": "key"}

    result = merge(left, right, how=how, sort=sort, **on_kwargs)

    if on_index:
        left = left.reset_index()
        right = right.reset_index()

    if how in ["left", "right", "inner"]:
        if how in ["left", "inner"]:
            expected, other, other_unique = left, right, right_unique
        else:
            expected, other, other_unique = right, left, left_unique
        if how == "inner":
            keep_values = set(left["key"].values).intersection(right["key"].values)
            keep_mask = expected["key"].isin(keep_values)
            expected = expected[keep_mask]
        if sort:
            expected = expected.sort_values("key")
        if not other_unique:
            other_value_counts = other["key"].value_counts()
            repeats = other_value_counts.reindex(expected["key"].values, fill_value=1)
            repeats = repeats.astype(np.intp)
            expected = expected["key"].repeat(repeats.values)
            expected = expected.to_frame()
    elif how == "outer":
        left_counts = left["key"].value_counts()
        right_counts = right["key"].value_counts()
        expected_counts = left_counts.mul(right_counts, fill_value=1)
        expected_counts = expected_counts.astype(np.intp)
        expected = expected_counts.index.values.repeat(expected_counts.values)
        expected = DataFrame({"key": expected})
        expected = expected.sort_values("key")

    if on_index:
        expected = expected.set_index("key")
    else:
        expected = expected.reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


def test_merge_ea_int_and_float_numpy():
    # [FUNCTION DOC]
    # Verifies merging a nullable integer column with a numpy float column: a UserWarning
    # that mixed types are merged when the floats are not integral (empty result) in
    # both orders (result dtype follows the left frame), and no warning for integral
    # floats giving a single matching row (GH 46178).
    # GH#46178
    df1 = DataFrame([1.0, pd.NA], dtype=pd.Int64Dtype())
    df2 = DataFrame([1.5])
    expected = DataFrame(columns=[0], dtype="Int64")

    with tm.assert_produces_warning(UserWarning, match="You are merging"):
        result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(UserWarning, match="You are merging"):
        result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype("float64"))

    df2 = DataFrame([1.0])
    expected = DataFrame([1], columns=[0], dtype="Int64")
    result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)

    result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype("float64"))


def test_merge_arrow_string_index(any_string_dtype):
    # [FUNCTION DOC]
    # Takes any_string_dtype; skipped without pyarrow. Verifies a left merge of a string
    # column against a string-dtype index keeps the string dtype and fills missing
    # (GH 54894).
    # GH#54894
    pytest.importorskip("pyarrow")
    left = DataFrame({"a": ["a", "b"]}, dtype=any_string_dtype)
    right = DataFrame({"b": 1}, index=Index(["a", "c"], dtype=any_string_dtype))
    result = left.merge(right, left_on="a", right_index=True, how="left")
    expected = DataFrame(
        {"a": Series(["a", "b"], dtype=any_string_dtype), "b": [1, np.nan]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("left_empty", [True, False])
@pytest.mark.parametrize("right_empty", [True, False])
def test_merge_empty_frames_column_order(left_empty, right_empty):
    # [FUNCTION DOC]
    # Parametrized over left/right being empty. Verifies the outer merge result has
    # columns in the order key, left-only, right-only, with missing fill for the empty
    # side's columns, and is empty if both are (GH 51929).
    # GH 51929
    df1 = DataFrame(1, index=[0], columns=["A", "B"])
    df2 = DataFrame(1, index=[0], columns=["A", "C", "D"])

    if left_empty:
        df1 = df1.iloc[:0]
    if right_empty:
        df2 = df2.iloc[:0]

    result = merge(df1, df2, on=["A"], how="outer")
    expected = DataFrame(1, index=range(1), columns=["A", "B", "C", "D"])
    if left_empty and right_empty:
        expected = expected.iloc[:0]
    elif left_empty:
        expected["B"] = np.nan
    elif right_empty:
        expected[["C", "D"]] = np.nan
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_merge_datetime_and_timedelta(how):
    # [FUNCTION DOC]
    # Parametrized over join kind. Verifies a value error naming both dtypes and the key
    # when merging a datetime column with a timedelta column, in both orders.
    left = DataFrame({"key": Series([1, None], dtype="datetime64[ns]")})
    right = DataFrame({"key": Series([1], dtype="timedelta64[ns]")})

    msg = (
        f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} "
        "columns for key 'key'. If you wish to proceed you should use pd.concat"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        left.merge(right, on="key", how=how)

    msg = (
        f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} "
        "columns for key 'key'. If you wish to proceed you should use pd.concat"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        right.merge(left, on="key", how=how)


def test_merge_on_all_nan_column():
    # [FUNCTION DOC]
    # Verifies an outer merge on keys where one key column is entirely missing matches
    # the all-missing keys without out-of-bounds access (GH 59421).
    # GH#59421
    left = DataFrame({"x": [1, 2, 3], "y": [np.nan, np.nan, np.nan], "z": [4, 5, 6]})
    right = DataFrame({"x": [1, 2, 3], "y": [np.nan, np.nan, np.nan], "zz": [4, 5, 6]})
    result = left.merge(right, on=["x", "y"], how="outer")
    # Should not trigger array bounds eerror with bounds checking or asan enabled.
    expected = DataFrame(
        {"x": [1, 2, 3], "y": [np.nan, np.nan, np.nan], "z": [4, 5, 6], "zz": [4, 5, 6]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("suffixes", [("_dup", ""), ("", "_dup")])
def test_merge_for_suffix_collisions(suffixes):
    # [FUNCTION DOC]
    # Parametrized over two suffix pairs. Verifies a merge error mentioning duplicate
    # columns when the suffixed overlapping column collides with an existing column
    # (GH 61402).
    # GH#61402
    df1 = DataFrame({"col1": [1], "col2": [2]})
    df2 = DataFrame({"col1": [1], "col2": [2], "col2_dup": [3]})
    with pytest.raises(MergeError, match="duplicate columns"):
        merge(df1, df2, on="col1", suffixes=suffixes)


def test_merge_categorical_key_recursion():
    # [FUNCTION DOC]
    # Verifies an outer merge on categorical keys with different category dtypes (int64
    # and float64 categories) completes (no infinite recursion) and equals merging the
    # plain-dtype versions (GH 56376).
    # GH#56376
    lt = CategoricalDtype(categories=np.asarray([1, 2, 3], dtype="int64"))
    rt = CategoricalDtype(categories=np.asarray([1, 2, 3], dtype="float64"))
    left = DataFrame({"key": Series([1, 2], dtype=lt)})
    right = DataFrame({"key": Series([1, 3], dtype=rt)})

    result = left.merge(right, on="key", how="outer")
    expected = left.astype("int64").merge(
        right.astype("float64"), on="key", how="outer"
    )
    tm.assert_frame_equal(result, expected)


def test_merge_pyarrow_datetime_duplicates():
    # [FUNCTION DOC]
    # Skipped without pyarrow. Verifies a left merge on a pyarrow datetime key with
    # duplicates on the right repeats left rows and keeps pyarrow dtypes (GH 61926).
    # GH#61926
    pytest.importorskip("pyarrow")

    t = pd.date_range("2025-07-06", periods=3, freq="h")
    df1 = DataFrame({"time": t, "val1": [1, 2, 3]})
    df1 = df1.convert_dtypes(dtype_backend="pyarrow")

    df2 = DataFrame({"time": t.repeat(2), "val2": [10, 20, 30, 40, 50, 60]})
    df2 = df2.convert_dtypes(dtype_backend="pyarrow")

    result = merge(df1, df2, on="time", how="left")

    expected = DataFrame(
        {
            "time": t.repeat(2),
            "val1": [1, 1, 2, 2, 3, 3],
            "val2": [10, 20, 30, 40, 50, 60],
        }
    )
    expected = expected.convert_dtypes(dtype_backend="pyarrow")
    tm.assert_frame_equal(result, expected)


def test_merge_right_on_and_right_index():
    # [FUNCTION DOC]
    # Verifies a merge error is raised when right_on and right_index are both given.
    df1 = DataFrame({"col": [1, 2, 3]})
    df2 = DataFrame({"col": [2, 3, 4]})

    with pytest.raises(pd.errors.MergeError):
        df1.merge(df2, left_on="col", right_on="col", right_index=True)
