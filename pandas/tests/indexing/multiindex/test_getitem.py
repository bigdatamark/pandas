# [MODULE DOC]
# Tests for getitem and loc selection on Series and DataFrame with multi-level indexes: partial indexing by levels, scalar results, error messages, generators as keys, tuple column names, empty levels, missing values in levels, duplicate-labelled columns and empty selections. Sections are separated by comments for series, frame, and duplicate-index cases. Relies on the multiindex_dataframe_random_data and year-month-day frame fixtures and on the indexer_sl fixture.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.indexing import IndexingError

# ----------------------------------------------------------------------------
# test indexing of Series with multi-level Index
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "access_method",
    [lambda s, x: s[:, x], lambda s, x: s.loc[:, x], lambda s, x: s.xs(x, level=1)],
)
@pytest.mark.parametrize(
    "level1_value, expected",
    [(0, Series([1], index=[0])), (1, Series([2, 3], index=[1, 2]))],
)
def test_series_getitem_multiindex(access_method, level1_value, expected):
    # [FUNCTION DOC]
    # Parametrized over access methods (bracket with slice and value, label-based, cross-section at level one) and (level-one value, expected series). Selecting rows by the second-level value returns a series indexed by the first level (name "A").
    # GH 6018
    # series regression getitem with a multi-index

    mi = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 1)], names=["A", "B"])
    ser = Series([1, 2, 3], index=mi)
    expected.index.name = "A"

    result = access_method(ser, level1_value)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("level0_value", ["D", "A"])
def test_series_getitem_duplicates_multiindex(level0_value):
    # [FUNCTION DOC]
    # Parametrized over the first-level label (D or A). Builds a frame with duplicated first-level labels. Selecting a missing key raises a key error with just the key text (the A check is skipped when A is a real label); selecting an existing first-level label returns a series of its second-level values with the right name.
    # GH 5725 the 'A' happens to be a valid Timestamp so the doesn't raise
    # the appropriate error, only in PY3 of course!

    index = MultiIndex(
        levels=[[level0_value, "B", "C"], [0, 26, 27, 37, 57, 67, 75, 82]],
        codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]],
        names=["tag", "day"],
    )
    arr = np.random.default_rng(2).standard_normal((len(index), 1))
    df = DataFrame(arr, index=index, columns=["val"])

    # confirm indexing on missing value raises KeyError
    if level0_value != "A":
        with pytest.raises(KeyError, match=r"^'A'$"):
            df.val["A"]

    with pytest.raises(KeyError, match=r"^'X'$"):
        df.val["X"]

    result = df.val[level0_value]
    expected = Series(
        arr.ravel()[0:3], name="val", index=Index([26, 37, 57], name="day")
    )
    tm.assert_series_equal(result, expected)


def test_series_getitem(multiindex_year_month_day_dataframe_random_data, indexer_sl):
    # [FUNCTION DOC]
    # Using the year-month-day random frame's first column and indexer_sl, selecting with year and month returns a series indexed by day, equal to the matching slice with the first two levels dropped.
    s = multiindex_year_month_day_dataframe_random_data["A"]
    expected = s.reindex(s.index[42:65])
    expected.index = expected.index.droplevel(0).droplevel(0)

    result = indexer_sl(s)[2000, 3]
    tm.assert_series_equal(result, expected)


def test_series_getitem_returns_scalar(
    multiindex_year_month_day_dataframe_random_data, indexer_sl
):
    # [FUNCTION DOC]
    # A full key (year, month, day) returns the scalar at the matching position.
    s = multiindex_year_month_day_dataframe_random_data["A"]
    expected = s.iloc[49]

    result = indexer_sl(s)[2000, 3, 10]
    assert result == expected


@pytest.mark.parametrize(
    "indexer,expected_error,expected_error_msg",
    [
        (lambda s: s.__getitem__((2000, 3, 4)), KeyError, r"^\(2000, 3, 4\)$"),
        (lambda s: s[(2000, 3, 4)], KeyError, r"^\(2000, 3, 4\)$"),
        (lambda s: s.loc[(2000, 3, 4)], KeyError, r"^\(2000, 3, 4\)$"),
        (lambda s: s.loc[(2000, 3, 4, 5)], IndexingError, "Too many indexers"),
        (
            lambda s: s.__getitem__(len(s)),
            KeyError,
            "100",
        ),  # match should include len(s)
        (lambda s: s[len(s)], KeyError, "100"),  # match should include len(s)
        (
            lambda s: s.iloc[len(s)],
            IndexError,
            "single positional indexer is out-of-bounds",
        ),
    ],
)
def test_series_getitem_indexing_errors(
    multiindex_year_month_day_dataframe_random_data,
    indexer,
    expected_error,
    expected_error_msg,
):
    # [FUNCTION DOC]
    # Parametrized over (accessor, exception type, message). A missing full key raises a key error with the tuple text (via getitem and loc); too many indexers raises the indexing error "Too many indexers"; integer positions beyond the length raise a key error (getitem) or an index error ("single positional indexer is out-of-bounds") for positional access.
    s = multiindex_year_month_day_dataframe_random_data["A"]
    with pytest.raises(expected_error, match=expected_error_msg):
        indexer(s)


def test_series_getitem_corner_generator(
    multiindex_year_month_day_dataframe_random_data,
):
    # [FUNCTION DOC]
    # Indexing a series with a generator of booleans equals indexing with the equivalent boolean series.
    s = multiindex_year_month_day_dataframe_random_data["A"]
    result = s[(x > 0 for x in s)]
    expected = s[s > 0]
    tm.assert_series_equal(result, expected)


# ----------------------------------------------------------------------------
# test indexing of DataFrame with multi-level Index
# ----------------------------------------------------------------------------


def test_getitem_simple(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # On the transposed random frame, selecting a column by a two-element tuple gives the first column's values.
    df = multiindex_dataframe_random_data.T
    expected = df.values[:, 0]
    result = df["foo", "one"].values
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected_error_msg",
    [
        (lambda df: df[("foo", "four")], r"^\('foo', 'four'\)$"),
        (lambda df: df["foobar"], r"^'foobar'$"),
    ],
)
def test_frame_getitem_simple_key_error(
    multiindex_dataframe_random_data, indexer, expected_error_msg
):
    # [FUNCTION DOC]
    # Parametrized over (accessor, message). A missing tuple key or a missing top-level label raises a key error with the key text.
    df = multiindex_dataframe_random_data.T
    with pytest.raises(KeyError, match=expected_error_msg):
        indexer(df)


def test_tuple_string_column_names():
    # [FUNCTION DOC]
    # A frame with two-level columns and an added single-level column is flattened to tuple column labels (plus a new scalar column); selecting a list of one tuple label and the new column name gives the expected frame whose column index mixes a tuple and a string.
    # GH#50372
    mi = MultiIndex.from_tuples([("a", "aa"), ("a", "ab"), ("b", "ba"), ("b", "bb")])
    df = DataFrame([range(4), range(1, 5), range(2, 6)], columns=mi)
    df["single_index"] = 0

    df_flat = df.copy()
    df_flat.columns = df_flat.columns.to_flat_index()
    df_flat["new_single_index"] = 0

    result = df_flat[[("a", "aa"), "new_single_index"]]
    expected = DataFrame(
        [[0, 0], [1, 0], [2, 0]], columns=Index([("a", "aa"), "new_single_index"])
    )
    tm.assert_frame_equal(result, expected)


def test_frame_getitem_multicolumn_empty_level():
    # [FUNCTION DOC]
    # A frame with three-level columns where one second-level label is an empty string: selecting the first-level label returns a frame whose columns are the remaining deepest level only (empty labels dropped).
    df = DataFrame({"a": ["1", "2", "3"], "b": ["2", "3", "4"]})
    df.columns = [
        ["level1 item1", "level1 item2"],
        ["", "level2 item2"],
        ["level3 item1", "level3 item2"],
    ]

    result = df["level1 item1"]
    expected = DataFrame(
        [["1"], ["2"], ["3"]], index=df.index, columns=["level3 item1"]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected_slice",
    [
        (lambda df: df["foo"], slice(3)),
        (lambda df: df["bar"], slice(3, 5)),
        (lambda df: df.loc[:, "bar"], slice(3, 5)),
    ],
)
def test_frame_getitem_toplevel(
    multiindex_dataframe_random_data, indexer, expected_slice
):
    # [FUNCTION DOC]
    # Parametrized over (accessor, expected slice of columns): selecting a top-level label by bracket or label-based column selection returns the sub-columns with that level dropped.
    df = multiindex_dataframe_random_data.T
    expected = df.reindex(columns=df.columns[expected_slice])
    expected.columns = expected.columns.droplevel(0)
    result = indexer(df)
    tm.assert_frame_equal(result, expected)


def test_frame_mixed_depth_get():
    # [FUNCTION DOC]
    # Frame with columns of mixed depth (padded with empty strings). Selecting a shallow label returns a series named for it; selecting a two-element tuple returns the series of the deeper key, renamed to the shorter tuple.
    arrays = [
        ["a", "top", "top", "routine1", "routine1", "routine2"],
        ["", "OD", "OD", "result1", "result2", "result1"],
        ["", "wx", "wy", "", "", ""],
    ]

    tuples = sorted(zip(*arrays))
    index = MultiIndex.from_tuples(tuples)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

    result = df["a"]
    expected = df["a", "", ""].rename("a")
    tm.assert_series_equal(result, expected)

    result = df["routine1", "result1"]
    expected = df["routine1", "result1", ""]
    expected = expected.rename(("routine1", "result1"))
    tm.assert_series_equal(result, expected)


def test_frame_getitem_nan_multiindex(nulls_fixture):
    # [FUNCTION DOC]
    # Uses the nulls fixture. A frame indexed by two columns with the second level always the null value: slicing by label with a tuple endpoint on either side or both returns the expected contiguous rows, with the data column cast to int64.
    # GH#29751
    # loc on a multiindex containing nan values
    n = nulls_fixture  # for code readability
    cols = ["a", "b", "c"]
    df = DataFrame(
        [[11, n, 13], [21, n, 23], [31, n, 33], [41, n, 43]],
        columns=cols,
    ).set_index(["a", "b"])
    df["c"] = df["c"].astype("int64")

    idx = (21, n)
    result = df.loc[:idx]
    expected = DataFrame([[11, n, 13], [21, n, 23]], columns=cols).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)

    result = df.loc[idx:]
    expected = DataFrame(
        [[21, n, 23], [31, n, 33], [41, n, 43]], columns=cols
    ).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)

    idx1, idx2 = (21, n), (31, n)
    result = df.loc[idx1:idx2]
    expected = DataFrame([[21, n, 23], [31, n, 33]], columns=cols).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected",
    [
        (
            (["b"], ["bar", np.nan]),
            (
                DataFrame(
                    [[2, 3], [5, 6]],
                    columns=MultiIndex.from_tuples([("b", "bar"), ("b", np.nan)]),
                    dtype="int64",
                )
            ),
        ),
        (
            (["a", "b"]),
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6]],
                    columns=MultiIndex.from_tuples(
                        [("a", "foo"), ("b", "bar"), ("b", np.nan)]
                    ),
                    dtype="int64",
                )
            ),
        ),
        (
            (["b"]),
            (
                DataFrame(
                    [[2, 3], [5, 6]],
                    columns=MultiIndex.from_tuples([("b", "bar"), ("b", np.nan)]),
                    dtype="int64",
                )
            ),
        ),
        (
            (["b"], ["bar"]),
            (
                DataFrame(
                    [[2], [5]],
                    columns=MultiIndex.from_tuples([("b", "bar")]),
                    dtype="int64",
                )
            ),
        ),
        (
            (["b"], [np.nan]),
            (
                DataFrame(
                    [[3], [6]],
                    columns=MultiIndex(
                        codes=[[1], [-1]], levels=[["a", "b"], ["bar", "foo"]]
                    ),
                    dtype="int64",
                )
            ),
        ),
        (("b", np.nan), Series([3, 6], dtype="int64", name=("b", np.nan))),
    ],
)
def test_frame_getitem_nan_cols_multiindex(
    indexer,
    expected,
    nulls_fixture,
):
    # [FUNCTION DOC]
    # Parametrized over several column selectors (list-of-list selectors, a single list, a tuple key with NaN) and their expected frame or series; uses the nulls fixture for the NaN-like entry in the columns' second level. Label-based column selection returns the expected object.
    # Slicing MultiIndex including levels with nan values, for more information
    # see GH#25154
    df = DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        columns=MultiIndex.from_tuples(
            [("a", "foo"), ("b", "bar"), ("b", nulls_fixture)]
        ),
        dtype="int64",
    )

    result = df.loc[:, indexer]
    tm.assert_equal(result, expected)


# ----------------------------------------------------------------------------
# test indexing of DataFrame with multi-level Index with duplicates
# ----------------------------------------------------------------------------


@pytest.fixture
def dataframe_with_duplicate_index():
    # [FUNCTION DOC]
    # HELPER: fixture returning a frame with a three-row string index and two-level columns whose labels include a duplicate (A2 appears twice under A) and columns named main and sub, used for regression tests of two GitHub issues.
    """Fixture for DataFrame used in tests for gh-4145 and gh-4146"""
    data = [["a", "d", "e", "c", "f", "b"], [1, 4, 5, 3, 6, 2], [1, 4, 5, 3, 6, 2]]
    index = ["h1", "h3", "h5"]
    columns = MultiIndex(
        levels=[["A", "B"], ["A1", "A2", "B1", "B2"]],
        codes=[[0, 0, 0, 1, 1, 1], [0, 3, 3, 0, 1, 2]],
        names=["main", "sub"],
    )
    return DataFrame(data, index=index, columns=columns)


@pytest.mark.parametrize(
    "indexer", [lambda df: df[("A", "A1")], lambda df: df.loc[:, ("A", "A1")]]
)
def test_frame_mi_access(dataframe_with_duplicate_index, indexer):
    # [FUNCTION DOC]
    # Parametrized over bracket tuple access and label-based access. Selecting the unique key ("A", "A1") returns a frame transposed such that the expected single column is selected.
    # GH 4145
    df = dataframe_with_duplicate_index
    index = Index(["h1", "h3", "h5"])
    columns = MultiIndex.from_tuples([("A", "A1")], names=["main", "sub"])
    expected = DataFrame([["a", 1, 1]], index=columns, columns=index).T

    result = indexer(df)
    tm.assert_frame_equal(result, expected)


def test_frame_mi_access_returns_series(dataframe_with_duplicate_index):
    # [FUNCTION DOC]
    # Chained selection of a unique second-level key returns a series named for the key with the row labels as index.
    # GH 4146, not returning a block manager when selecting a unique index
    # from a duplicate index
    # as of 4879, this returns a Series (which is similar to what happens
    # with a non-unique)
    df = dataframe_with_duplicate_index
    expected = Series(["a", 1, 1], index=["h1", "h3", "h5"], name="A1")
    result = df["A"]["A1"]
    tm.assert_series_equal(result, expected)


def test_frame_mi_access_returns_frame(dataframe_with_duplicate_index):
    # [FUNCTION DOC]
    # Selecting a duplicated second-level key returns a frame with both matching columns, with the sub-level name on the columns.
    # selecting a non_unique from the 2nd level
    df = dataframe_with_duplicate_index
    expected = DataFrame(
        [["d", 4, 4], ["e", 5, 5]],
        index=Index(["B2", "B2"], name="sub"),
        columns=["h1", "h3", "h5"],
    ).T
    result = df["A"]["B2"]
    tm.assert_frame_equal(result, expected)


def test_frame_mi_empty_slice():
    # [FUNCTION DOC]
    # Selecting with an empty list of columns returns a frame with the original rows and an empty two-level column index.
    # GH 15454
    df = DataFrame(0, index=range(2), columns=MultiIndex.from_product([[1], [2]]))
    result = df[[]]
    expected = DataFrame(
        index=[0, 1], columns=MultiIndex(levels=[[1], [2]], codes=[[], []])
    )
    tm.assert_frame_equal(result, expected)


def test_loc_empty_multiindex():
    # [FUNCTION DOC]
    # A label-based selection with an empty multi-level index (from a condition that matches nothing) equals selection with an all-false mask; assigning a value to those empty rows leaves the frame unchanged.
    # GH#36936
    arrays = [["a", "a", "b", "a"], ["a", "a", "b", "b"]]
    index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))
    df = DataFrame([1, 2, 3, 4], index=index, columns=["value"])

    # loc on empty multiindex == loc with False mask
    empty_multiindex = df.loc[df.loc[:, "value"] == 0, :].index
    result = df.loc[empty_multiindex, :]
    expected = df.loc[[False] * len(df.index), :]
    tm.assert_frame_equal(result, expected)

    # replacing value with loc on empty multiindex
    df.loc[df.loc[df.loc[:, "value"] == 0].index, "value"] = 5
    result = df
    expected = DataFrame([1, 2, 3, 4], index=index, columns=["value"])
    tm.assert_frame_equal(result, expected)
