# [MODULE DOC]
# Tests for DataFrame join: index-based and key-based joins of several kinds, validation modes, suffixes, joining lists of frames or series, multi-index, timezone, period and categorical indexes.
# Defines four fixtures with and without duplicated keys that build on one another.

from datetime import datetime
import re
import zoneinfo

import numpy as np
import pytest

from pandas.errors import MergeError

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat


@pytest.fixture
def left_no_dup():
    # [FUNCTION DOC]
    # HELPER: fixture returning the left frame without duplicate keys (columns a and b with four rows).
    return DataFrame(
        {"a": ["a", "b", "c", "d"], "b": ["cat", "dog", "weasel", "horse"]},
        index=range(4),
    )


@pytest.fixture
def right_no_dup():
    # [FUNCTION DOC]
    # HELPER: fixture returning the right frame without duplicate keys, indexed by column a (five rows, one column c).
    return DataFrame(
        {
            "a": ["a", "b", "c", "d", "e"],
            "c": ["meow", "bark", "um... weasel noise?", "nay", "chirp"],
        },
        index=range(5),
    ).set_index("a")


@pytest.fixture
def left_w_dups(left_no_dup):
    # [FUNCTION DOC]
    # HELPER: fixture returning the left frame with a repeated key a, made by concatenating an extra row (sorted columns).
    return concat(
        [left_no_dup, DataFrame({"a": ["a"], "b": ["cow"]}, index=[3])], sort=True
    )


@pytest.fixture
def right_w_dups(right_no_dup):
    # [FUNCTION DOC]
    # HELPER: fixture returning the right frame with a repeated key e in its a-index, made by concatenating an extra row and setting the index.
    return concat(
        [right_no_dup, DataFrame({"a": ["e"], "c": ["moo"]}, index=[3])]
    ).set_index("a")


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
            DataFrame({"a": [np.nan, 10, 20], "b": [300, 100, 200]}, index=[3, 1, 2]),
        ),
        (
            "right",
            True,
            DataFrame({"a": [10, 20, np.nan], "b": [100, 200, 300]}, index=[1, 2, 3]),
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
def test_join(how, sort, expected):
    # [FUNCTION DOC]
    # Parametrized over join type (inner, left, right, outer), sort flag and expected frame. Joins two small integer-indexed frames with a one-to-one validation. Verifies the result rows and order: the order follows the left frame when not sorting and is sorted when sorting; for outer joins the result is sorted either way.
    left = DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])
    right = DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])
    result = left.join(right, how=how, sort=sort, validate="1:1")
    tm.assert_frame_equal(result, expected)


def test_suffix_on_list_join():
    # [FUNCTION DOC]
    # Verifies joining a list of frames with a left or right suffix raises a value error that suffixes are not supported for multiple frames; overlapping values among the list raise a value error about overlapping indexes; joining a one-element list equals joining the frame itself.
    first = DataFrame({"key": [1, 2, 3, 4, 5]})
    second = DataFrame({"key": [1, 8, 3, 2, 5], "v1": [1, 2, 3, 4, 5]})
    third = DataFrame({"keys": [5, 2, 3, 4, 1], "v2": [1, 2, 3, 4, 5]})

    # check proper errors are raised
    msg = "Suffixes not supported when joining multiple DataFrames"
    with pytest.raises(ValueError, match=msg):
        first.join([second], lsuffix="y")
    with pytest.raises(ValueError, match=msg):
        first.join([second, third], rsuffix="x")
    with pytest.raises(ValueError, match=msg):
        first.join([second, third], lsuffix="y", rsuffix="x")
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        first.join([second, third])

    # no errors should be raised
    arr_joined = first.join([third])
    norm_joined = first.join(third)
    tm.assert_frame_equal(arr_joined, norm_joined)


def test_join_invalid_validate(left_no_dup, right_no_dup):
    # [FUNCTION DOC]
    # GH 46622. Verifies an invalid validate string raises a value error listing the valid argument names.
    # GH 46622
    # Check invalid arguments
    msg = (
        '"invalid" is not a valid argument. '
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
        left_no_dup.merge(right_no_dup, on="a", validate="invalid")


@pytest.mark.parametrize("dtype", ["object", "string[pyarrow]"])
def test_join_on_single_col_dup_on_right(left_no_dup, right_w_dups, dtype):
    # [FUNCTION DOC]
    # GH 46622. Parametrized over object and pyarrow string dtype. Verifies duplicates on the right are allowed with one-to-many validation and raise a merge error stating the right keys are not unique for one-to-one validation.
    # GH 46622
    # Dups on right allowed by one_to_many constraint
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    left_no_dup = left_no_dup.astype(dtype)
    right_w_dups.index = right_w_dups.index.astype(dtype)
    left_no_dup.join(
        right_w_dups,
        on="a",
        validate="one_to_many",
    )

    # Dups on right not allowed by one_to_one constraint
    msg = "Merge keys are not unique in right dataset; not a one-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_no_dup.join(
            right_w_dups,
            on="a",
            validate="one_to_one",
        )


def test_join_on_single_col_dup_on_left(left_w_dups, right_no_dup):
    # [FUNCTION DOC]
    # GH 46622. Verifies duplicates on the left are allowed with many-to-one validation and raise a merge error about the left dataset for one-to-one validation.
    # GH 46622
    # Dups on left allowed by many_to_one constraint
    left_w_dups.join(
        right_no_dup,
        on="a",
        validate="many_to_one",
    )

    # Dups on left not allowed by one_to_one constraint
    msg = "Merge keys are not unique in left dataset; not a one-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_no_dup,
            on="a",
            validate="one_to_one",
        )


def test_join_on_single_col_dup_on_both(left_w_dups, right_w_dups):
    # [FUNCTION DOC]
    # GH 46622. Verifies duplicates on both sides are allowed for many-to-many, and raise merge errors for many-to-one (message about the right dataset) and one-to-many (message about the left dataset).
    # GH 46622
    # Dups on both allowed by many_to_many constraint
    left_w_dups.join(right_w_dups, on="a", validate="many_to_many")

    # Dups on both not allowed by many_to_one constraint
    msg = "Merge keys are not unique in right dataset; not a many-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_w_dups,
            on="a",
            validate="many_to_one",
        )

    # Dups on both not allowed by one_to_many constraint
    msg = "Merge keys are not unique in left dataset; not a one-to-many merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_w_dups,
            on="a",
            validate="one_to_many",
        )


def test_join_on_multi_col_check_dup():
    # [FUNCTION DOC]
    # GH 46622. Two-key join where each single key has duplicates but the pair is unique. Verifies an inner join with one-to-one validation succeeds and gives the expected frame.
    # GH 46622
    # Two column join, dups in both, but jointly no dups
    left = DataFrame(
        {
            "a": ["a", "a", "b", "b"],
            "b": [0, 1, 0, 1],
            "c": ["cat", "dog", "weasel", "horse"],
        },
        index=range(4),
    ).set_index(["a", "b"])

    right = DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": [0, 1, 0],
            "d": ["meow", "bark", "um... weasel noise?"],
        },
        index=range(3),
    ).set_index(["a", "b"])

    expected_multi = DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": [0, 1, 0],
            "c": ["cat", "dog", "weasel"],
            "d": ["meow", "bark", "um... weasel noise?"],
        },
        index=range(3),
    ).set_index(["a", "b"])

    # Jointly no dups allowed by one_to_one constraint
    result = left.join(right, how="inner", validate="1:1")
    tm.assert_frame_equal(result, expected_multi)


def test_join_index(float_frame):
    # [FUNCTION DOC]
    # Uses float_frame split into overlapping row and column parts. Verifies the index of the result and columns for default (left), left, right, inner, outer, left-anti and right-anti joins; an unknown join type raises a value error listing valid merge types; joining a frame with itself without suffix raises a value error about overlapping columns.
    # left / right

    f = float_frame.loc[float_frame.index[:10], ["A", "B"]]
    f2 = float_frame.loc[float_frame.index[5:], ["C", "D"]].iloc[::-1]

    joined = f.join(f2)
    tm.assert_index_equal(f.index, joined.index)
    expected_columns = Index(["A", "B", "C", "D"])
    tm.assert_index_equal(joined.columns, expected_columns)

    joined = f.join(f2, how="left")
    tm.assert_index_equal(joined.index, f.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    joined = f.join(f2, how="right")
    tm.assert_index_equal(joined.index, f2.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    # inner

    joined = f.join(f2, how="inner")
    tm.assert_index_equal(joined.index, f.index[5:10])
    tm.assert_index_equal(joined.columns, expected_columns)

    # outer

    joined = f.join(f2, how="outer")
    tm.assert_index_equal(joined.index, float_frame.index.sort_values())
    tm.assert_index_equal(joined.columns, expected_columns)

    # left anti
    joined = f.join(f2, how="left_anti")
    tm.assert_index_equal(joined.index, float_frame.index[:5])
    tm.assert_index_equal(joined.columns, expected_columns)

    # right anti
    joined = f.join(f2, how="right_anti")
    tm.assert_index_equal(joined.index, float_frame.index[10:][::-1])
    tm.assert_index_equal(joined.columns, expected_columns)

    join_msg = (
        "'foo' is not a valid Merge type: left, right, inner, outer, "
        "left_anti, right_anti, cross, asof"
    )
    with pytest.raises(ValueError, match=re.escape(join_msg)):
        f.join(f2, how="foo")

    # corner case - overlapping columns
    msg = "columns overlap but no suffix"
    for how in ("outer", "left", "inner"):
        with pytest.raises(ValueError, match=msg):
            float_frame.join(float_frame, how=how)


def test_join_index_more(float_frame):
    # [FUNCTION DOC]
    # Verifies joining two column subsets of the float frame where one has every second row: the default and right joins give the expected rows, and the right join with swapped operands gives the same data with swapped column order.
    af = float_frame.loc[:, ["A", "B"]]
    bf = float_frame.loc[::2, ["C", "D"]]

    expected = af.copy()
    expected["C"] = float_frame["C"][::2]
    expected["D"] = float_frame["D"][::2]

    result = af.join(bf)
    tm.assert_frame_equal(result, expected)

    result = af.join(bf, how="right")
    tm.assert_frame_equal(result, expected[::2])

    result = bf.join(af, how="right")
    tm.assert_frame_equal(result, expected.loc[:, result.columns])


def test_join_index_series(float_frame):
    # [FUNCTION DOC]
    # Verifies joining a series (popped last column) back to its frame reproduces the frame, and that a series without a name raises a value error saying it must have a name.
    df = float_frame.copy()
    ser = df.pop(float_frame.columns[-1])
    joined = df.join(ser)

    tm.assert_frame_equal(joined, float_frame)

    ser.name = None
    with pytest.raises(ValueError, match="must have a name"):
        df.join(ser)


def test_join_overlap(float_frame):
    # [FUNCTION DOC]
    # Verifies overlapping columns with left and right suffixes: suffixed columns plus non-overlapping ones equal the expected frame (column order not enforced).
    df1 = float_frame.loc[:, ["A", "B", "C"]]
    df2 = float_frame.loc[:, ["B", "C", "D"]]

    joined = df1.join(df2, lsuffix="_df1", rsuffix="_df2")
    df1_suf = df1.loc[:, ["B", "C"]].add_suffix("_df1")
    df2_suf = df2.loc[:, ["B", "C"]].add_suffix("_df2")

    no_overlap = float_frame.loc[:, ["A", "D"]]
    expected = df1_suf.join(df2_suf).join(no_overlap)

    # column order not necessarily sorted
    tm.assert_frame_equal(joined, expected.loc[:, joined.columns])


def test_join_period_index():
    # [FUNCTION DOC]
    # Verifies joining a frame with a period index to a version with renamed columns gives the combined columns and unchanged index.
    frame_with_period_index = DataFrame(
        data=np.arange(20).reshape(4, 5),
        columns=list("abcde"),
        index=period_range(start="2000", freq="Y", periods=4),
    )
    other = frame_with_period_index.rename(columns=lambda key: f"{key}{key}")

    joined_values = np.concatenate([frame_with_period_index.values] * 2, axis=1)

    joined_cols = frame_with_period_index.columns.append(other.columns)

    joined = frame_with_period_index.join(other)
    expected = DataFrame(
        data=joined_values, columns=joined_cols, index=frame_with_period_index.index
    )

    tm.assert_frame_equal(joined, expected)


def test_join_left_sequence_non_unique_index():
    # [FUNCTION DOC]
    # Issue 19607. Joins a list of two frames with left join where the right frames have duplicated or missing keys. Verifies the left rows are repeated for duplicate matches and missing values are filled.
    # https://github.com/pandas-dev/pandas/issues/19607
    df1 = DataFrame({"a": [0, 10, 20]}, index=[1, 2, 3])
    df2 = DataFrame({"b": [100, 200, 300]}, index=[4, 3, 2])
    df3 = DataFrame({"c": [400, 500, 600]}, index=[2, 2, 4])

    joined = df1.join([df2, df3], how="left")

    expected = DataFrame(
        {
            "a": [0, 10, 10, 20],
            "b": [np.nan, 300, 300, 200],
            "c": [np.nan, 400, 500, np.nan],
        },
        index=[1, 2, 2, 3],
    )

    tm.assert_frame_equal(joined, expected)


def test_join_list_series(float_frame):
    # [FUNCTION DOC]
    # GH#46850. Verifies a frame can be joined with a list containing both a series and a frame, restoring the original float frame.
    # GH#46850
    # Join a DataFrame with a list containing both a Series and a DataFrame
    left = float_frame.A.to_frame()
    right = [float_frame.B, float_frame[["C", "D"]]]
    result = left.join(right)
    tm.assert_frame_equal(result, float_frame)


class TestDataFrameJoin:
    # [CLASS DOC]
    # Groups the join tests that were written as methods of a class.
    def test_join(self, multiindex_dataframe_random_data):
        # [FUNCTION DOC]
        # Uses the multiindex_dataframe_random_data fixture. Joins two partial column subsets of a multi-indexed frame with an outer join and reindexes. Verifies the values match the original with missing values where the partial pieces do not overlap, and that the result is not entirely missing.
        frame = multiindex_dataframe_random_data

        a = frame.loc[frame.index[:5], ["A"]]
        b = frame.loc[frame.index[2:], ["B", "C"]]

        joined = a.join(b, how="outer").reindex(frame.index)
        expected = frame.copy().values.copy()
        expected[np.isnan(joined.values)] = np.nan
        expected = DataFrame(expected, index=frame.index, columns=frame.columns)

        assert not np.isnan(joined.values).all()

        tm.assert_frame_equal(joined, expected)

    def test_join_segfault(self):
        # [FUNCTION DOC]
        # GH#1532. Regression: joining two frames indexed by overlapping pairs with left, right and outer joins must not crash. Nothing is asserted.
        # GH#1532
        df1 = DataFrame({"a": [1, 1], "b": [1, 2], "x": [1, 2]})
        df2 = DataFrame({"a": [2, 2], "b": [1, 2], "y": [1, 2]})
        df1 = df1.set_index(["a", "b"])
        df2 = df2.set_index(["a", "b"])
        # it works!
        for how in ["left", "right", "outer"]:
            df1.join(df2, how=how)

    def test_join_str_datetime(self):
        # [FUNCTION DOC]
        # Joins a frame whose column holds date strings to a frame whose index holds those strings and whose columns are datetimes, using the key column. Verifies only that the result has three columns.
        str_dates = ["20120209", "20120222"]
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        A = DataFrame(str_dates, index=range(2), columns=["aa"])
        C = DataFrame([[1, 2], [3, 4]], index=str_dates, columns=dt_dates)

        tst = A.join(C, on="aa")

        assert len(tst.columns) == 3

    def test_join_multiindex_leftright(self):
        # [FUNCTION DOC]
        # GH 10741. Joins a two-level index frame with a one-level frame in both directions and with left and right join types. Verifies the expected values, that results are identical up to column order between the two directions, and that the right join keeps only the shared first-level keys.
        # GH 10741
        df1 = DataFrame(
            [
                ["a", "x", 0.471780],
                ["a", "y", 0.774908],
                ["a", "z", 0.563634],
                ["b", "x", -0.353756],
                ["b", "y", 0.368062],
                ["b", "z", -1.721840],
                ["c", "x", 1],
                ["c", "y", 2],
                ["c", "z", 3],
            ],
            columns=["first", "second", "value1"],
        ).set_index(["first", "second"])

        df2 = DataFrame([["a", 10], ["b", 20]], columns=["first", "value2"]).set_index(
            ["first"]
        )

        exp = DataFrame(
            [
                [0.471780, 10],
                [0.774908, 10],
                [0.563634, 10],
                [-0.353756, 20],
                [0.368062, 20],
                [-1.721840, 20],
                [1.000000, np.nan],
                [2.000000, np.nan],
                [3.000000, np.nan],
            ],
            index=df1.index,
            columns=["value1", "value2"],
        )

        # these must be the same results (but columns are flipped)
        tm.assert_frame_equal(df1.join(df2, how="left"), exp)
        tm.assert_frame_equal(df2.join(df1, how="right"), exp[["value2", "value1"]])

        exp_idx = MultiIndex.from_product(
            [["a", "b"], ["x", "y", "z"]], names=["first", "second"]
        )
        exp = DataFrame(
            [
                [0.471780, 10],
                [0.774908, 10],
                [0.563634, 10],
                [-0.353756, 20],
                [0.368062, 20],
                [-1.721840, 20],
            ],
            index=exp_idx,
            columns=["value1", "value2"],
        )

        tm.assert_frame_equal(df1.join(df2, how="right"), exp)
        tm.assert_frame_equal(df2.join(df1, how="left"), exp[["value2", "value1"]])

    def test_join_multiindex_dates(self):
        # [FUNCTION DOC]
        # GH 33692. Verifies joining a frame to a list of two others with a multi-index containing a date level gives a combined frame with all three columns.
        # GH 33692
        date = pd.Timestamp(2000, 1, 1).date()

        df1_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df1 = DataFrame({"col1": [0]}, index=df1_index)
        df2_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df2 = DataFrame({"col2": [0]}, index=df2_index)
        df3_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df3 = DataFrame({"col3": [0]}, index=df3_index)

        result = df1.join([df2, df3])

        expected_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        expected = DataFrame(
            {"col1": [0], "col2": [0], "col3": [0]}, index=expected_index
        )

        tm.assert_equal(result, expected)

    def test_merge_join_different_levels_raises(self):
        # [FUNCTION DOC]
        # GH#9455 and GH 40993. Verifies merge and join on a column between flat columns and a two-level column frame raise a merge error that merging between different levels is not allowed.
        # GH#9455
        # GH 40993: For raising, enforced in 2.0

        # first dataframe
        df1 = DataFrame(columns=["a", "b"], data=[[1, 11], [0, 22]])

        # second dataframe
        columns = MultiIndex.from_tuples([("a", ""), ("c", "c1")])
        df2 = DataFrame(columns=columns, data=[[1, 33], [0, 44]])

        # merge
        with pytest.raises(
            MergeError, match="Not allowed to merge between different levels"
        ):
            pd.merge(df1, df2, on="a")

        # join, see discussion in GH#12219
        with pytest.raises(
            MergeError, match="Not allowed to merge between different levels"
        ):
            df1.join(df2, on="a")

    def test_frame_join_tzaware(self):
        # [FUNCTION DOC]
        # Verifies an outer join of two frames with timezone-aware indexes of different frequencies gives the union of the indexes with the timezone preserved (US/Central).
        tz = zoneinfo.ZoneInfo("US/Central")
        test1 = DataFrame(
            np.zeros((6, 3)),
            index=date_range("2012-11-15 00:00:00", periods=6, freq="100ms", tz=tz),
        )
        test2 = DataFrame(
            np.zeros((3, 3)),
            index=date_range("2012-11-15 00:00:00", periods=3, freq="250ms", tz=tz),
            columns=range(3, 6),
        )

        result = test1.join(test2, how="outer")
        expected = test1.index.union(test2.index)

        tm.assert_index_equal(result.index, expected)
        assert result.index.tz.key == "US/Central"

    def test_frame_join_categorical_index(self):
        # [FUNCTION DOC]
        # GH 61675. Joins two frames with a categorical (ordered, nullable-integer categories) index and categorical columns. Verifies the result keeps the categorical index and combines the categorical columns of both frames.
        # GH 61675
        cat_data = pd.Categorical(
            [3, 4],
            categories=pd.Series([2, 3, 4, 5], dtype="Int64"),
            ordered=True,
        )
        values1 = "a b".split()
        values2 = "foo bar".split()
        df1 = DataFrame({"hr": cat_data, "values1": values1}).set_index("hr")
        df2 = DataFrame({"hr": cat_data, "values2": values2}).set_index("hr")
        df1.columns = pd.CategoricalIndex([4], dtype=cat_data.dtype, name="other_hr")
        df2.columns = pd.CategoricalIndex([3], dtype=cat_data.dtype, name="other_hr")

        df_joined = df1.join(df2)
        expected = DataFrame(
            {"hr": cat_data, "values1": values1, "values2": values2}
        ).set_index("hr")
        expected.columns = pd.CategoricalIndex(
            [4, 3], dtype=cat_data.dtype, name="other_hr"
        )

        tm.assert_frame_equal(df_joined, expected)
