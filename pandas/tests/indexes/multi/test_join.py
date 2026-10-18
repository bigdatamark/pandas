# [MODULE DOC]
# Tests for join on MultiIndex: joining on a level, self joins, joining a
# MultiIndex with a flat index or another MultiIndex (overlapping names), indexer
# return values, interval levels, extension dtypes, NaN keys and level preservation.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Interval,
    MultiIndex,
    Series,
    StringDtype,
)
import pandas._testing as tm


@pytest.mark.parametrize("other", [["three", "one", "two"], ["one"], ["one", "three"]])
def test_join_level(idx, other, join_type):
    # [FUNCTION DOC]
    # Parametrized over three flat indexes and the join-type fixture. Joins the flat
    # index to the fixture MultiIndex on the level named "second" with indexers.
    # Checks the result keeps the first level, joins the second level as a flat join
    # would, keeps only rows whose second-level value survives, and for inner/outer
    # that the reversed join gives the same index and swapped indexers.
    other = Index(other)
    join_index, lidx, ridx = other.join(
        idx, how=join_type, level="second", return_indexers=True
    )

    exp_level = other.join(idx.levels[1], how=join_type)
    assert join_index.levels[0].equals(idx.levels[0])
    assert join_index.levels[1].equals(exp_level)

    # pare down levels
    mask = np.array([x[1] in exp_level for x in idx], dtype=bool)
    exp_values = idx.values[mask]
    tm.assert_numpy_array_equal(join_index.values, exp_values)

    if join_type in ("outer", "inner"):
        join_index2, ridx2, lidx2 = idx.join(
            other, how=join_type, level="second", return_indexers=True
        )

        assert join_index.equals(join_index2)
        tm.assert_numpy_array_equal(lidx, lidx2)
        if ridx is None:
            assert ridx == ridx2
        else:
            tm.assert_numpy_array_equal(ridx, ridx2)
        tm.assert_numpy_array_equal(join_index2.values, exp_values)


def test_join_level_corner_case(idx):
    # [FUNCTION DOC]
    # Joining a flat index to a MultiIndex on a named level yields a MultiIndex;
    # joining a MultiIndex to itself on a numeric level is ambiguous and raises a
    # type error.
    # some corner cases
    index = Index(["three", "one", "two"])
    result = index.join(idx, level="second")
    assert isinstance(result, MultiIndex)

    with pytest.raises(TypeError, match="Join.*MultiIndex.*ambiguous"):
        idx.join(idx, level=1)


def test_join_self(idx, join_type):
    # [FUNCTION DOC]
    # Parametrized over the join-type fixture. Joining the fixture index with itself
    # returns the index itself (sorted for outer joins).
    result = idx.join(idx, how=join_type)
    expected = idx
    if join_type == "outer":
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)


def test_join_multi():
    # [FUNCTION DOC]
    # Joins a two-level index and a flat index sharing a level name (GH 10665):
    # inner keeps matching rows with expected left/right indexers, in both
    # orientations; left/right joins keep the MultiIndex with a missing left indexer
    # and the expected right indexer.
    # GH 10665
    midx = MultiIndex.from_product([np.arange(4), np.arange(4)], names=["a", "b"])
    idx = Index([1, 2, 5], name="b")

    # inner
    jidx, lidx, ridx = midx.join(idx, how="inner", return_indexers=True)
    exp_idx = MultiIndex.from_product([np.arange(4), [1, 2]], names=["a", "b"])
    exp_lidx = np.array([1, 2, 5, 6, 9, 10, 13, 14], dtype=np.intp)
    exp_ridx = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.intp)
    tm.assert_index_equal(jidx, exp_idx)
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)
    # flip
    jidx, ridx, lidx = idx.join(midx, how="inner", return_indexers=True)
    tm.assert_index_equal(jidx, exp_idx)
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)

    # keep MultiIndex
    jidx, lidx, ridx = midx.join(idx, how="left", return_indexers=True)
    exp_ridx = np.array(
        [-1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1], dtype=np.intp
    )
    tm.assert_index_equal(jidx, midx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)
    # flip
    jidx, ridx, lidx = idx.join(midx, how="right", return_indexers=True)
    tm.assert_index_equal(jidx, midx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_multi_wrong_order():
    # [FUNCTION DOC]
    # Joining two MultiIndexes with the same names in a different order returns the
    # left index with no left indexer and an all-missing right indexer (GH 25760,
    # GH 28956).
    # GH 25760
    # GH 28956

    midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["b", "a"])

    join_idx, lidx, ridx = midx1.join(midx2, return_indexers=True)

    exp_ridx = np.array([-1, -1, -1, -1], dtype=np.intp)

    tm.assert_index_equal(midx1, join_idx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_multi_return_indexers():
    # [FUNCTION DOC]
    # Joining a three-level index with a two-level index with indexers disabled
    # returns just the index, equal to the left one (GH 34074).
    # GH 34074

    midx1 = MultiIndex.from_product([[1, 2], [3, 4], [5, 6]], names=["a", "b", "c"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])

    result = midx1.join(midx2, return_indexers=False)
    tm.assert_index_equal(result, midx1)


def test_join_overlapping_interval_level():
    # [FUNCTION DOC]
    # Outer join of two MultiIndexes with interval levels given in different
    # order returns the sorted union (GH 44096).
    # GH 44096
    idx_1 = MultiIndex.from_tuples(
        [
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (1, Interval(2.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),  # interval limit is here at 3.0, not at 2.0
            (2, Interval(3.0, 5.0)),
        ],
        names=["num", "interval"],
    )

    idx_2 = MultiIndex.from_tuples(
        [
            (1, Interval(2.0, 5.0)),
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (2, Interval(3.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),
        ],
        names=["num", "interval"],
    )

    expected = MultiIndex.from_tuples(
        [
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (1, Interval(2.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),
            (2, Interval(3.0, 5.0)),
        ],
        names=["num", "interval"],
    )
    result = idx_1.join(idx_2, how="outer")

    tm.assert_index_equal(result, expected)


def test_join_midx_ea():
    # [FUNCTION DOC]
    # Inner join of two MultiIndexes with nullable integer levels and one shared name
    # keeps extension dtypes and adds the new level (GH#49277).
    # GH#49277
    midx = MultiIndex.from_arrays(
        [Series([1, 1, 3], dtype="Int64"), Series([1, 2, 3], dtype="Int64")],
        names=["a", "b"],
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1], dtype="Int64"), Series([3], dtype="Int64")], names=["a", "c"]
    )
    result = midx.join(midx2, how="inner")
    expected = MultiIndex.from_arrays(
        [
            Series([1, 1], dtype="Int64"),
            Series([1, 2], dtype="Int64"),
            Series([3, 3], dtype="Int64"),
        ],
        names=["a", "b", "c"],
    )
    tm.assert_index_equal(result, expected)


def test_join_midx_string():
    # [FUNCTION DOC]
    # Same as the nullable integer case but for string-dtype levels (GH#49277).
    # GH#49277
    midx = MultiIndex.from_arrays(
        [
            Series(["a", "a", "c"], dtype=StringDtype()),
            Series(["a", "b", "c"], dtype=StringDtype()),
        ],
        names=["a", "b"],
    )
    midx2 = MultiIndex.from_arrays(
        [Series(["a"], dtype=StringDtype()), Series(["c"], dtype=StringDtype())],
        names=["a", "c"],
    )
    result = midx.join(midx2, how="inner")
    expected = MultiIndex.from_arrays(
        [
            Series(["a", "a"], dtype=StringDtype()),
            Series(["a", "b"], dtype=StringDtype()),
            Series(["c", "c"], dtype=StringDtype()),
        ],
        names=["a", "b", "c"],
    )
    tm.assert_index_equal(result, expected)


def test_join_multi_with_nan():
    # [FUNCTION DOC]
    # Joining frames whose MultiIndexes differ by NaN in a level keeps the left index
    # and gives NaN in the right column where no match exists (GH29252).
    # GH29252
    df1 = DataFrame(
        data={"col1": [1.1, 1.2]},
        index=MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"]),
    )
    df2 = DataFrame(
        data={"col2": [2.1, 2.2]},
        index=MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"]),
    )
    result = df1.join(df2)
    expected = DataFrame(
        data={"col1": [1.1, 1.2], "col2": [np.nan, 2.2]},
        index=MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", [0, 5])
def test_join_dtypes(any_numeric_ea_dtype, val):
    # [FUNCTION DOC]
    # Parametrized over a value (zero or five) and numeric extension dtypes. Outer join
    # of MultiIndexes preserves the extension dtype of the first level and returns a
    # sorted union (GH#49830).
    # GH#49830
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [3, 4]])
    midx2 = MultiIndex.from_arrays(
        [Series([1, val, val], dtype=any_numeric_ea_dtype), [3, 4, 4]]
    )
    result = midx.join(midx2, how="outer")
    expected = MultiIndex.from_arrays(
        [Series([val, val, 1, 2], dtype=any_numeric_ea_dtype), [4, 4, 3, 4]]
    ).sort_values()
    tm.assert_index_equal(result, expected)


def test_join_dtypes_all_nan(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Same with an all-NaN second level; result keeps the extension dtype and all NaN
    # second level values (GH#49830).
    # GH#49830
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [np.nan, np.nan]]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 0, 0], dtype=any_numeric_ea_dtype), [np.nan, np.nan, np.nan]]
    )
    result = midx.join(midx2, how="outer")
    expected = MultiIndex.from_arrays(
        [
            Series([0, 0, 1, 2], dtype=any_numeric_ea_dtype),
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    tm.assert_index_equal(result, expected)


def test_join_index_levels():
    # [FUNCTION DOC]
    # Outer join with duplicate left entries keeps duplicates, sorted, and the second
    # result level equals the expected levels (GH#53093).
    # GH#53093
    midx = MultiIndex.from_tuples([("a", "2019-02-01"), ("a", "2019-02-01")])
    midx2 = MultiIndex.from_tuples([("a", "2019-01-31")])
    result = midx.join(midx2, how="outer")
    expected = MultiIndex.from_tuples(
        [("a", "2019-01-31"), ("a", "2019-02-01"), ("a", "2019-02-01")]
    )
    tm.assert_index_equal(result.levels[1], expected.levels[1])
    tm.assert_index_equal(result, expected)
