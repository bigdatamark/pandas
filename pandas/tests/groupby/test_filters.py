# [MODULE DOC]
# Tests for the filter method of SeriesGroupBy and DataFrameGroupBy: boolean-returning predicates per group, dropna true (drop failing rows) and false (keep shape with NaN), non-scalar/non-boolean predicate errors, NaN and NA results treated as false, equivalence with the transform-and-mask workaround, ordering and duplicated indexes of various types, and regression cases.
from string import ascii_lowercase

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
)
import pandas._testing as tm


def test_filter_series():
    # [FUNCTION DOC]
    # Series filter with groups by parity: groups whose mean is below 10 (odd values) or above 10 (even values) are kept with their original index. With dropna false the result is reindexed to the full original index with NaN for filtered-out rows.
    s = Series([1, 3, 20, 5, 22, 24, 7])
    expected_odd = Series([1, 3, 5, 7], index=[0, 1, 3, 6])
    expected_even = Series([20, 22, 24], index=[2, 4, 5])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() < 10), expected_odd)
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() > 10), expected_even)
    # Test dropna=False.
    tm.assert_series_equal(
        grouped.filter(lambda x: x.mean() < 10, dropna=False),
        expected_odd.reindex(s.index),
    )
    tm.assert_series_equal(
        grouped.filter(lambda x: x.mean() > 10, dropna=False),
        expected_even.reindex(s.index),
    )


def test_filter_single_column_df():
    # [FUNCTION DOC]
    # Same as the Series filter test but for a single-column DataFrame: frame rows of groups passing the mean test are kept; dropna false reindexes with NaN rows.
    df = DataFrame([1, 3, 20, 5, 22, 24, 7])
    expected_odd = DataFrame([1, 3, 5, 7], index=[0, 1, 3, 6])
    expected_even = DataFrame([20, 22, 24], index=[2, 4, 5])
    grouper = df[0].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    tm.assert_frame_equal(grouped.filter(lambda x: x.mean() < 10), expected_odd)
    tm.assert_frame_equal(grouped.filter(lambda x: x.mean() > 10), expected_even)
    # Test dropna=False.
    tm.assert_frame_equal(
        grouped.filter(lambda x: x.mean() < 10, dropna=False),
        expected_odd.reindex(df.index),
    )
    tm.assert_frame_equal(
        grouped.filter(lambda x: x.mean() > 10, dropna=False),
        expected_even.reindex(df.index),
    )


def test_filter_multi_column_df():
    # [FUNCTION DOC]
    # Two-column frame grouped by parity of A. The predicate (sum of A minus sum of B greater than 10) keeps only the group with the two 12 values, with their original index 1 and 2.
    df = DataFrame({"A": [1, 12, 12, 1], "B": [1, 1, 1, 1]})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    expected = DataFrame({"A": [12, 12], "B": [1, 1]}, index=[1, 2])
    tm.assert_frame_equal(
        grouped.filter(lambda x: x["A"].sum() - x["B"].sum() > 10), expected
    )


def test_filter_mixed_df():
    # [FUNCTION DOC]
    # Mixed-dtype frame (ints and strings) with a predicate on column A sum; keeps the rows of the group passing the test, preserving both columns.
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    expected = DataFrame({"A": [12, 12], "B": ["b", "c"]}, index=[1, 2])
    tm.assert_frame_equal(grouped.filter(lambda x: x["A"].sum() > 10), expected)


def test_filter_out_all_groups():
    # [FUNCTION DOC]
    # A predicate that is false for every group returns an empty Series (for Series) or an empty frame (for DataFrame) with the same dtypes.
    s = Series([1, 3, 20, 5, 22, 24, 7])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    tm.assert_series_equal(grouped.filter(lambda x: x.mean() > 1000), s[[]])
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    tm.assert_frame_equal(grouped.filter(lambda x: x["A"].sum() > 1000), df.loc[[]])


def test_filter_out_no_groups():
    # [FUNCTION DOC]
    # A predicate true for every group returns the Series unchanged.
    s = Series([1, 3, 20, 5, 22, 24, 7])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    filtered = grouped.filter(lambda x: x.mean() > 0)
    tm.assert_series_equal(filtered, s)


def test_filter_out_no_groups_dataframe():
    # [FUNCTION DOC]
    # A predicate true for every group returns the DataFrame unchanged.
    df = DataFrame({"A": [1, 12, 12, 1], "B": "a b c d".split()})
    grouper = df["A"].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    filtered = grouped.filter(lambda x: x["A"].mean() > 0)
    tm.assert_frame_equal(filtered, df)


def test_filter_out_all_groups_in_df():
    # [FUNCTION DOC]
    # GH12768: with dropna false and every group filtered out, the result has the original shape filled with NaN (floats) in both columns.
    # GH12768
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 0]})
    res = df.groupby("a")
    res = res.filter(lambda x: x["b"].sum() > 5, dropna=False)
    expected = DataFrame({"a": [np.nan] * 3, "b": [np.nan] * 3})
    tm.assert_frame_equal(expected, res)


def test_filter_out_all_groups_in_df_dropna_true():
    # [FUNCTION DOC]
    # GH12768: with dropna true and every group filtered out, the result is an empty frame of int64 columns a and b.
    # GH12768
    df = DataFrame({"a": [1, 1, 2], "b": [1, 2, 0]})
    res = df.groupby("a")
    res = res.filter(lambda x: x["b"].sum() > 5, dropna=True)
    expected = DataFrame({"a": [], "b": []}, dtype="int64")
    tm.assert_frame_equal(expected, res)


def test_filter_condition_raises():
    # [FUNCTION DOC]
    # When the predicate itself raises (for a group whose sum is zero) the filter wraps the failure into a type error saying the filter must return a boolean result.
    def raise_if_sum_is_zero(x):
        # [FUNCTION DOC]
        # HELPER: predicate used by the test. Raises a value error without a message when the sum of the group is zero, otherwise returns whether the sum is positive.
        if x.sum() == 0:
            raise ValueError
        return x.sum() > 0

    s = Series([-1, 0, 1, 2])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        grouped.filter(raise_if_sum_is_zero)


def test_filter_bad_shapes():
    # [FUNCTION DOC]
    # Predicates returning non-scalar results: identity, an elementwise comparison, or an outer product. For the DataFrameGroupBy, the first two raise a type error saying the filter function returned a DataFrame but expected a scalar bool; the outer product raises a type error from multiplying strings. For the SeriesGroupBy all three raise a type error saying the filter must return a boolean result.
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    s = df["B"]
    g_df = df.groupby("B")
    g_s = s.groupby(s)

    f = lambda x: x
    msg = "filter function returned a DataFrame, but expected a scalar bool"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)

    f = lambda x: x == 1
    msg = "filter function returned a DataFrame, but expected a scalar bool"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)

    f = lambda x: np.outer(x, x)
    msg = "can't multiply sequence by non-int of type 'str'"
    with pytest.raises(TypeError, match=msg):
        g_df.filter(f)
    msg = "the filter must return a boolean result"
    with pytest.raises(TypeError, match=msg):
        g_s.filter(f)


def test_filter_nan_is_false():
    # [FUNCTION DOC]
    # A predicate returning NaN is treated as false: both frame and Series filters return an empty selection of the original.
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    s = df["B"]
    g_df = df.groupby(df["B"])
    g_s = s.groupby(s)

    f = lambda x: np.nan
    tm.assert_frame_equal(g_df.filter(f), df.loc[[]])
    tm.assert_series_equal(g_s.filter(f), s[[]])


def test_filter_pdna_is_false():
    # [FUNCTION DOC]
    # A predicate returning the NA value is treated as false and must not raise when truth value of NA is evaluated; frame and Series filters return empty selections.
    # in particular, dont raise in filter trying to call bool(pd.NA)
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    ser = df["B"]
    g_df = df.groupby(df["B"])
    g_s = ser.groupby(ser)

    func = lambda x: pd.NA
    res = g_df.filter(func)
    tm.assert_frame_equal(res, df.loc[[]])
    res = g_s.filter(func)
    tm.assert_series_equal(res, ser[[]])


def test_filter_against_workaround_ints():
    # [FUNCTION DOC]
    # Random integer Series grouped by values rounded to tens: filter by group mean greater than 10 must equal selecting rows where the transform of the same predicate is true (the old workaround); both sorted by value before comparison.
    # Series of ints
    s = Series(np.random.default_rng(2).integers(0, 100, 10))
    grouper = s.apply(lambda x: np.round(x, -1))
    grouped = s.groupby(grouper)
    f = lambda x: x.mean() > 10

    old_way = s[grouped.transform(f).astype("bool")]
    new_way = grouped.filter(f)
    tm.assert_series_equal(new_way.sort_values(), old_way.sort_values())


def test_filter_against_workaround_floats():
    # [FUNCTION DOC]
    # Same workaround equivalence for a Series of floats scaled to 0 to 100.
    # Series of floats
    s = 100 * Series(np.random.default_rng(2).random(10))
    grouper = s.apply(lambda x: np.round(x, -1))
    grouped = s.groupby(grouper)
    f = lambda x: x.mean() > 10
    old_way = s[grouped.transform(f).astype("bool")]
    new_way = grouped.filter(f)
    tm.assert_series_equal(new_way.sort_values(), old_way.sort_values())


def test_filter_against_workaround_dataframe():
    # [FUNCTION DOC]
    # Workaround equivalence on a ten-row frame of ints, floats and letters in three configurations: group by ints and filter on mean of floats, group by rounded floats and filter on count of letters, group by letters and filter on mean of ints.
    # Set up DataFrame of ints, floats, strings.
    letters = np.array(list(ascii_lowercase))
    N = 10
    random_letters = letters.take(
        np.random.default_rng(2).integers(0, 26, N, dtype=int)
    )
    df = DataFrame(
        {
            "ints": Series(np.random.default_rng(2).integers(0, 10, N)),
            "floats": N / 10 * Series(np.random.default_rng(2).random(N)),
            "letters": Series(random_letters),
        }
    )

    # Group by ints; filter on floats.
    grouped = df.groupby("ints")
    old_way = df[grouped.floats.transform(lambda x: x.mean() > N / 2).astype("bool")]
    new_way = grouped.filter(lambda x: x["floats"].mean() > N / 2)
    tm.assert_frame_equal(new_way, old_way)

    # Group by floats (rounded); filter on strings.
    grouper = df.floats.apply(lambda x: np.round(x, -1))
    grouped = df.groupby(grouper)
    old_way = df[grouped.letters.transform(lambda x: len(x) < N / 2).astype("bool")]
    new_way = grouped.filter(lambda x: len(x.letters) < N / 2)
    tm.assert_frame_equal(new_way, old_way)

    # Group by strings; filter on ints.
    grouped = df.groupby("letters")
    old_way = df[grouped.ints.transform(lambda x: x.mean() > N / 2).astype("bool")]
    new_way = grouped.filter(lambda x: x["ints"].mean() > N / 2)
    tm.assert_frame_equal(new_way, old_way)


def test_filter_using_len():
    # [FUNCTION DOC]
    # GH 4447: filter by group length greater than 2 keeps the four rows of group b with their index 2 to 5; length greater than 4 keeps nothing.
    # GH 4447
    df = DataFrame({"A": np.arange(8), "B": list("aabbbbcc"), "C": np.arange(8)})
    grouped = df.groupby("B")
    actual = grouped.filter(lambda x: len(x) > 2)
    expected = DataFrame(
        {"A": np.arange(2, 6), "B": list("bbbb"), "C": np.arange(2, 6)},
        index=range(2, 6),
    )
    tm.assert_frame_equal(actual, expected)

    actual = grouped.filter(lambda x: len(x) > 4)
    expected = df.loc[[]]
    tm.assert_frame_equal(actual, expected)


def test_filter_using_len_series():
    # [FUNCTION DOC]
    # GH 4447: Series version of the length filter, grouped by itself; keeps four b values with index 2 to 5, none for length greater than 4.
    # GH 4447
    s = Series(list("aabbbbcc"), name="B")
    grouped = s.groupby(s)
    actual = grouped.filter(lambda x: len(x) > 2)
    expected = Series(4 * ["b"], index=range(2, 6), name="B")
    tm.assert_series_equal(actual, expected)

    actual = grouped.filter(lambda x: len(x) > 4)
    expected = s[[]]
    tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "index", [range(8), range(7, -1, -1), [0, 2, 1, 3, 4, 6, 5, 7]]
)
def test_filter_maintains_ordering(index):
    # [FUNCTION DOC]
    # GH 4621: parametrized over three index orderings (ascending, descending, shuffled). Filtering for groups of more than one row preserves the original row order: positional rows 1, 2, 4 and 7 for the frame and for the Series grouped by the tag column.
    # GH 4621
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    s = df["pid"]
    grouped = df.groupby("tag")
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)

    grouped = s.groupby(df["tag"])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)


def test_filter_multiple_timestamp():
    # [FUNCTION DOC]
    # GH 10114: grouping by a string column and a constant Timestamp column. Filter with an always-true predicate returns the input; transform with length gives counts 2, 3, 2, 3, 3; frame transform with sum and with length give the expected per-row group sums and counts.
    # GH 10114
    df = DataFrame(
        {
            "A": np.arange(5, dtype="int64"),
            "B": ["foo", "bar", "foo", "bar", "bar"],
            "C": Timestamp("20130101"),
        }
    )

    grouped = df.groupby(["B", "C"])

    result = grouped["A"].filter(lambda x: True)
    tm.assert_series_equal(df["A"], result)

    result = grouped["A"].transform(len)
    expected = Series([2, 3, 2, 3, 3], name="A")
    tm.assert_series_equal(result, expected)

    result = grouped.filter(lambda x: True)
    tm.assert_frame_equal(df, result)

    result = grouped.transform("sum")
    expected = DataFrame({"A": [2, 8, 2, 8, 8]})
    tm.assert_frame_equal(result, expected)

    result = grouped.transform(len)
    expected = DataFrame({"A": [2, 3, 2, 3, 3]})
    tm.assert_frame_equal(result, expected)


def test_filter_and_transform_with_non_unique_int_index():
    # [FUNCTION DOC]
    # GH4620: frame and Series with a non-unique integer index. Filter keeps rows of groups with more than one row (positions 1, 2, 4, 7); with dropna false it returns all rows with NaN for removed ones (float64); transform of length gives group sizes per row; the transform through column attribute access matches too.
    # GH4620
    index = [1, 1, 1, 2, 1, 1, 0, 1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_multiple_non_unique_int_index():
    # [FUNCTION DOC]
    # GH4620: same as the previous test but with a different duplicated integer index (more repeated labels such as zero), checking the same filter/transform semantics.
    # GH4620
    index = [1, 1, 1, 2, 0, 0, 0, 1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_non_unique_float_index():
    # [FUNCTION DOC]
    # GH4620: same filter/transform semantics with a duplicated float index.
    # GH4620
    index = np.array([1, 1, 1, 2, 1, 1, 0, 1], dtype=float)
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_non_unique_timestamp_index():
    # [FUNCTION DOC]
    # GH4620: same filter/transform semantics with a duplicated Timestamp index.
    # GH4620
    t0 = Timestamp("2013-09-30 00:05:00")
    t1 = Timestamp("2013-10-30 00:05:00")
    t2 = Timestamp("2013-11-30 00:05:00")
    index = [t1, t1, t1, t2, t1, t1, t0, t1]
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_and_transform_with_non_unique_string_index():
    # [FUNCTION DOC]
    # GH4620: same filter/transform semantics with a duplicated string index.
    # GH4620
    index = list("bbbcbbab")
    df = DataFrame(
        {"pid": [1, 1, 1, 2, 2, 3, 3, 3], "tag": [23, 45, 62, 24, 45, 34, 25, 62]},
        index=index,
    )
    grouped_df = df.groupby("tag")
    ser = df["pid"]
    grouped_ser = ser.groupby(df["tag"])
    expected_indexes = [1, 2, 4, 7]

    # Filter DataFrame
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)

    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    # Cast to avoid upcast when setting nan below
    expected = df.copy().astype("float64")
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)

    # Filter Series
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)

    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name="pid")
    # ^ made manually because this can get confusing!
    tm.assert_series_equal(actual, expected)

    # Transform Series
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name="pid")
    tm.assert_series_equal(actual, expected)

    # Transform (a column from) DataFrameGroupBy
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)


def test_filter_has_access_to_grouped_cols():
    # [FUNCTION DOC]
    # The predicate receives the full group including the grouping column A (the comment says previously it did not); filtering for sum of A equal to 2 keeps the first two rows.
    df = DataFrame([[1, 2], [1, 3], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")
    # previously didn't have access to col A #????
    filt = g.filter(lambda x: x["A"].sum() == 2)
    tm.assert_frame_equal(filt, df.iloc[[0, 1]])


def test_filter_enforces_scalarness():
    # [FUNCTION DOC]
    # Predicate returning a boolean Series (not a scalar) must raise a type error whose message starts with filter function returned a.
    df = DataFrame(
        [
            ["best", "a", "x"],
            ["worst", "b", "y"],
            ["best", "c", "x"],
            ["best", "d", "y"],
            ["worst", "d", "y"],
            ["worst", "d", "y"],
            ["best", "d", "z"],
        ],
        columns=["a", "b", "c"],
    )
    with pytest.raises(TypeError, match="filter function returned a.*"):
        df.groupby("c").filter(lambda g: g["a"] == "best")


def test_filter_non_bool_raises():
    # [FUNCTION DOC]
    # Predicate returning a non-boolean number (the mean) must raise a type error whose message starts with filter function returned a.
    df = DataFrame(
        [
            ["best", "a", 1],
            ["worst", "b", 1],
            ["best", "c", 1],
            ["best", "d", 1],
            ["worst", "d", 1],
            ["worst", "d", 1],
            ["best", "d", 1],
        ],
        columns=["a", "b", "c"],
    )
    with pytest.raises(TypeError, match="filter function returned a.*"):
        df.groupby("a").filter(lambda g: g.c.mean())


def test_filter_dropna_with_empty_groups():
    # [FUNCTION DOC]
    # GH 10780: Series with three groups of three; a predicate that is false for all groups gives an all-NaN Series of original index with dropna false, and an empty float64 Series with an empty integer index with dropna true.
    # GH 10780
    data = Series(np.random.default_rng(2).random(9), index=np.repeat([1, 2, 3], 3))
    grouped = data.groupby(level=0)
    result_false = grouped.filter(lambda x: x.mean() > 1, dropna=False)
    expected_false = Series([np.nan] * 9, index=np.repeat([1, 2, 3], 3))
    tm.assert_series_equal(result_false, expected_false)

    result_true = grouped.filter(lambda x: x.mean() > 1, dropna=True)
    expected_true = Series(index=pd.Index([], dtype=int), dtype=np.float64)
    tm.assert_series_equal(result_true, expected_true)


def test_filter_consistent_result_before_after_agg_func():
    # [FUNCTION DOC]
    # GH 17091: filter with an always-true predicate gives the same frame before and after the same groupby object has had sum computed on it (no cached state alters the filter).
    # GH 17091
    df = DataFrame({"data": range(6), "key": list("ABCABC")})
    grouper = df.groupby("key")
    result = grouper.filter(lambda x: True)
    expected = DataFrame({"data": range(6), "key": list("ABCABC")})
    tm.assert_frame_equal(result, expected)

    grouper.sum()
    result = grouper.filter(lambda x: True)
    tm.assert_frame_equal(result, expected)


def test_filter_with_non_values():
    # [FUNCTION DOC]
    # GH 62501: grouping on a column containing None with dropna false, an always-true filter must return the original frame including the missing-key row.
    # GH 62501
    df = DataFrame(
        [
            [1],
            [None],
        ],
        columns=["a"],
    )

    result = df.groupby("a", dropna=False).filter(lambda x: True)
    tm.assert_frame_equal(result, df)


def test_filter_with_non_values_multi_index():
    # [FUNCTION DOC]
    # GH 62501: grouping on two columns with missing keys in various rows with dropna false; an always-true filter returns the whole original frame.
    # GH 62501
    df = DataFrame(
        [
            [1, 2],
            [3, None],
            [None, 4],
            [None, None],
        ],
        columns=["a", "b"],
    )

    result = df.groupby(["a", "b"], dropna=False).filter(lambda x: True)
    tm.assert_frame_equal(result, df)
