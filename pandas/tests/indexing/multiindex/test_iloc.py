# [MODULE DOC]
# Tests for position-based (iloc) indexing on frames and series that have a
# MultiIndex on rows and/or columns. Uses the shared multiindex_dataframe_random_data
# fixture from the test configuration plus one local fixture.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.fixture
def simple_multiindex_dataframe():
    # [FUNCTION DOC]
    # HELPER: fixture for the tests in this module.
    # Returns a 3 x 3 frame of seeded random normal values whose rows have the two-level
    # index with first level 4, 4, 8 and second level 8, 10, 12, and whose columns have
    # the two-level index with first level 2, 2, 4 and second level 6, 8, 10.
    """
    Factory function to create simple 3 x 3 dataframe with
    both columns and row MultiIndex using supplied data or
    random data by default.
    """

    data = np.random.default_rng(2).standard_normal((3, 3))
    return DataFrame(
        data, columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]]
    )


@pytest.mark.parametrize(
    "indexer, expected",
    [
        (
            lambda df: df.iloc[0],
            lambda arr: Series(arr[0], index=[[2, 2, 4], [6, 8, 10]], name=(4, 8)),
        ),
        (
            lambda df: df.iloc[2],
            lambda arr: Series(arr[2], index=[[2, 2, 4], [6, 8, 10]], name=(8, 12)),
        ),
        (
            lambda df: df.iloc[:, 2],
            lambda arr: Series(arr[:, 2], index=[[4, 4, 8], [8, 10, 12]], name=(4, 10)),
        ),
    ],
)
def test_iloc_returns_series(indexer, expected, simple_multiindex_dataframe):
    # [FUNCTION DOC]
    # Parametrized over three (indexer, expected-builder) pairs: integer row 0, integer row 2,
    # and all rows of integer column 2. The indexer is applied to the fixture frame and the
    # result must equal a series built from the raw array, with the other axis' MultiIndex as
    # the index and the selected row/column tuple as name.
    df = simple_multiindex_dataframe
    arr = df.values
    result = indexer(df)
    expected = expected(arr)
    tm.assert_series_equal(result, expected)


def test_iloc_returns_dataframe(simple_multiindex_dataframe):
    # [FUNCTION DOC]
    # Selecting rows at positions 0 and 1 returns a frame equal to the cross-section on
    # first-level label 4 with the level kept.
    df = simple_multiindex_dataframe
    result = df.iloc[[0, 1]]
    expected = df.xs(4, drop_level=False)
    tm.assert_frame_equal(result, expected)


def test_iloc_returns_scalar(simple_multiindex_dataframe):
    # [FUNCTION DOC]
    # Selecting a single position pair on both axes returns the scalar from the raw array.
    df = simple_multiindex_dataframe
    arr = df.values
    result = df.iloc[2, 2]
    expected = arr[2, 2]
    assert result == expected


def test_iloc_getitem_multiple_items():
    # GH 5528
    # [FUNCTION DOC]
    # GH 5528. On a 4-row frame with a two-level index, selecting positions 2 and 3 equals
    # the cross-section for first-level label b with the level kept.
    tup = zip(*[["a", "a", "b", "b"], ["x", "y", "x", "y"]])
    index = MultiIndex.from_tuples(tup)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=index)
    result = df.iloc[[2, 3]]
    expected = df.xs("b", drop_level=False)
    tm.assert_frame_equal(result, expected)


def test_iloc_getitem_labels():
    # this is basically regular indexing
    # [FUNCTION DOC]
    # Positional scalar access on a frame with duplicated MultiIndex labels on both axes
    # behaves like ordinary indexing: row 2, column 2 returns the matching raw array value.
    arr = np.random.default_rng(2).standard_normal((4, 3))
    df = DataFrame(
        arr,
        columns=[["i", "i", "j"], ["A", "A", "B"]],
        index=[["i", "i", "j", "k"], ["X", "X", "Y", "Y"]],
    )
    result = df.iloc[2, 2]
    expected = arr[2, 2]
    assert result == expected


def test_frame_getitem_slice(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # A positional slice of the first four rows equals the plain slice of the frame.
    df = multiindex_dataframe_random_data
    result = df.iloc[:4]
    expected = df[:4]
    tm.assert_frame_equal(result, expected)


def test_frame_setitem_slice(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # Assigning a scalar zero through a positional slice of the first four rows sets exactly
    # those rows to zero in the underlying values and leaves all other values non-zero.
    df = multiindex_dataframe_random_data
    df.iloc[:4] = 0

    assert (df.values[:4] == 0).all()
    assert (df.values[4:] != 0).all()


def test_indexing_ambiguity_bug_1678():
    # GH 1678
    # [FUNCTION DOC]
    # GH 1678. With a column MultiIndex, selecting the column at position 1 for all rows
    # equals label-based selection of the matching tuple, so positional access is not
    # confused by the MultiIndex.
    columns = MultiIndex.from_tuples(
        [("Ohio", "Green"), ("Ohio", "Red"), ("Colorado", "Green")]
    )
    index = MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])

    df = DataFrame(np.arange(12).reshape((4, 3)), index=index, columns=columns)

    result = df.iloc[:, 1]
    expected = df.loc[:, ("Ohio", "Red")]
    tm.assert_series_equal(result, expected)


def test_iloc_integer_locations():
    # GH 13797
    # [FUNCTION DOC]
    # GH 13797. For a frame with a MultiIndex containing duplicate labels, reading each
    # cell by integer row and column positions reproduces the original data when
    # reassembled into a frame.
    data = [
        ["str00", "str01"],
        ["str10", "str11"],
        ["str20", "srt21"],
        ["str30", "str31"],
        ["str40", "str41"],
    ]

    index = MultiIndex.from_tuples(
        [("CC", "A"), ("CC", "B"), ("CC", "B"), ("BB", "a"), ("BB", "b")]
    )

    expected = DataFrame(data)
    df = DataFrame(data, index=index)

    result = DataFrame([[df.iloc[r, c] for c in range(2)] for r in range(5)])

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, indexes, values, expected_k",
    [
        # test without indexer value in first level of MultiIndex
        ([[2, 22, 5], [2, 33, 6]], [0, -1, 1], [2, 3, 1], [7, 10]),
        # test like code sample 1 in the issue
        ([[1, 22, 555], [1, 33, 666]], [0, -1, 1], [200, 300, 100], [755, 1066]),
        # test like code sample 2 in the issue
        ([[1, 3, 7], [2, 4, 8]], [0, -1, 1], [10, 10, 1000], [17, 1018]),
        # test like code sample 3 in the issue
        ([[1, 11, 4], [2, 22, 5], [3, 33, 6]], [0, -1, 1], [4, 7, 10], [8, 15, 13]),
    ],
)
def test_iloc_setitem_int_multiindex_series(data, indexes, values, expected_k):
    # GH17148
    # [FUNCTION DOC]
    # GH 17148. Parametrized over frame data, positions (including a negative one), and
    # increments. A series with a two-level integer MultiIndex has the increments added
    # in place through positional augmented assignment; the result must equal the
    # expected values column, so integer labels in the index do not cause label-based
    # interpretation.
    df = DataFrame(data=data, columns=["i", "j", "k"])
    df = df.set_index(["i", "j"])

    series = df.k.copy()
    for i, v in zip(indexes, values):
        series.iloc[i] += v

    df["k"] = expected_k
    expected = df.k
    tm.assert_series_equal(series, expected)


def test_getitem_iloc(multiindex_dataframe_random_data):
    # [FUNCTION DOC]
    # Positional row selection at position 2 equals the cross-section taken with that
    # row's full index label.
    df = multiindex_dataframe_random_data
    result = df.iloc[2]
    expected = df.xs(df.index[2])
    tm.assert_series_equal(result, expected)
