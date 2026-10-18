# [MODULE DOC]
# Tests for MultiIndex.reindex and related behavior: type and names of the result,
# reindexing on a level, name and dtype preservation with empty or list-like
# targets, non-unique errors and the limit argument.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


def test_reindex(idx):
    # [FUNCTION DOC]
    # Reindexing to a list of tuples returns a MultiIndex with the original names and
    # level names; reindexing to a list of all existing values returns no indexer
    # (none) since nothing changed.
    result, indexer = idx.reindex(list(idx[:4]))
    assert isinstance(result, MultiIndex)
    assert result.names == ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]

    result, indexer = idx.reindex(list(idx))
    assert isinstance(result, MultiIndex)
    assert indexer is None
    assert result.names == ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]


def test_reindex_level(idx):
    # [FUNCTION DOC]
    # Reindex on the named level "second" with a flat target. Compares the result and
    # indexer with a right/left join on that level (both orientations); passing a fill
    # method together with a level raises a type error ("Fill method not supported").
    index = Index(["one"])

    target, indexer = idx.reindex(index, level="second")
    target2, indexer2 = index.reindex(idx, level="second")

    exp_index = idx.join(index, level="second", how="right")
    exp_index2 = idx.join(index, level="second", how="left")

    assert target.equals(exp_index)
    exp_indexer = np.array([0, 2, 4])
    tm.assert_numpy_array_equal(indexer, exp_indexer, check_dtype=False)

    assert target2.equals(exp_index2)
    exp_indexer2 = np.array([0, -1, 0, -1, 0, -1])
    tm.assert_numpy_array_equal(indexer2, exp_indexer2, check_dtype=False)

    with pytest.raises(TypeError, match="Fill method not supported"):
        idx.reindex(idx, method="pad", level="second")


def test_reindex_preserves_names_when_target_is_list_or_ndarray(idx):
    # [FUNCTION DOC]
    # Reindexing to empty, list-of-tuples or array targets (same or different dtypes)
    # keeps
    # the index names, for both unnamed and named indexes (GH6552).
    # GH6552
    idx = idx.copy()
    target = idx.copy()
    idx.names = target.names = [None, None]

    other_dtype = MultiIndex.from_product([[1, 2], [3, 4]])

    # list & ndarray cases
    assert idx.reindex([])[0].names == [None, None]
    assert idx.reindex(np.array([]))[0].names == [None, None]
    assert idx.reindex(target.tolist())[0].names == [None, None]
    assert idx.reindex(target.values)[0].names == [None, None]
    assert idx.reindex(other_dtype.tolist())[0].names == [None, None]
    assert idx.reindex(other_dtype.values)[0].names == [None, None]

    idx.names = ["foo", "bar"]
    assert idx.reindex([])[0].names == ["foo", "bar"]
    assert idx.reindex(np.array([]))[0].names == ["foo", "bar"]
    assert idx.reindex(target.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(target.values)[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.values)[0].names == ["foo", "bar"]


def test_reindex_lvl_preserves_names_when_target_is_list_or_array():
    # [FUNCTION DOC]
    # Reindexing an empty list on either level keeps the names (GH7774).
    # GH7774
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]], names=["foo", "bar"])
    assert idx.reindex([], level=0)[0].names == ["foo", "bar"]
    assert idx.reindex([], level=1)[0].names == ["foo", "bar"]


def test_reindex_lvl_preserves_type_if_target_is_empty_list_or_array(
    using_infer_string,
):
    # [FUNCTION DOC]
    # Reindexing an empty list on a level keeps the level dtype (integer, object or
    # string depending on the infer-string fixture; categorical and timezone-aware
    # datetime extension dtypes too) (GH7774).
    # GH7774
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]])
    assert idx.reindex([], level=0)[0].levels[0].dtype.type == np.int64
    exp = np.object_ if not using_infer_string else str
    assert idx.reindex([], level=1)[0].levels[1].dtype.type == exp

    # case with EA levels
    cat = pd.Categorical(["foo", "bar"])
    dti = pd.date_range("2016-01-01", periods=2, tz="US/Pacific")
    mi = MultiIndex.from_product([cat, dti])
    assert mi.reindex([], level=0)[0].levels[0].dtype == cat.dtype
    assert mi.reindex([], level=1)[0].levels[1].dtype == dti.dtype


def test_reindex_base(idx):
    # [FUNCTION DOC]
    # get_indexer of an index against itself is the identity positions; an invalid
    # fill method raises a value error.
    # KNOWN ISSUE: the name suggests a reindex test, but the body only exercises get_indexer.
    expected = np.arange(idx.size, dtype=np.intp)

    actual = idx.get_indexer(idx)
    tm.assert_numpy_array_equal(expected, actual)

    with pytest.raises(ValueError, match="Invalid fill method"):
        idx.get_indexer(idx, method="invalid")


def test_reindex_non_unique():
    # [FUNCTION DOC]
    # Reindexing a series with a non-unique MultiIndex raises a value error saying
    # non-unique multi-index cannot be handled.
    idx = MultiIndex.from_tuples([(0, 0), (1, 1), (1, 1), (2, 2)])
    a = pd.Series(np.arange(4), index=idx)
    new_idx = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)])

    msg = "cannot handle a non-unique multi-index!"
    with pytest.raises(ValueError, match=msg):
        a.reindex(new_idx)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # [FUNCTION DOC]
    # Parametrized over two value layouts (non-empty and empty). Reindexing to a new
    # level
    # value yields an index with that level value and no rows plus an empty indexer
    # (GH41170).
    # GH41170
    idx = MultiIndex.from_arrays(values)
    result, result_indexer = idx.reindex(np.array(["b"]), level=0)
    expected = MultiIndex(levels=[["b"], values[1]], codes=[[], []])
    expected_indexer = np.array([], dtype=result_indexer.dtype)
    tm.assert_index_equal(result, expected)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_not_all_tuples():
    # [FUNCTION DOC]
    # Reindexing a MultiIndex to a flat index containing a non-tuple key returns the
    # target as is and indexer with -1 for the non-tuple key.
    keys = [("i", "i"), ("i", "j"), ("j", "i"), "j"]
    mi = MultiIndex.from_tuples(keys[:-1])
    idx = Index(keys)
    res, indexer = mi.reindex(idx)

    tm.assert_index_equal(res, idx)
    expected = np.array([0, 1, 2, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)


def test_reindex_limit_arg_with_multiindex():
    # [FUNCTION DOC]
    # Passing a limit without a fill method when reindexing a series with a MultiIndex
    # raises a value error (GH21247).
    # GH21247

    idx = MultiIndex.from_tuples([(3, "A"), (4, "A"), (4, "B")])

    df = pd.Series([0.02, 0.01, 0.012], index=idx)

    new_idx = MultiIndex.from_tuples(
        [
            (3, "A"),
            (3, "B"),
            (4, "A"),
            (4, "B"),
            (4, "C"),
            (5, "B"),
            (5, "C"),
            (6, "B"),
            (6, "C"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="limit argument only valid if doing pad, backfill or nearest reindexing",
    ):
        df.reindex(new_idx, fill_value=0, limit=1)


def test_reindex_with_none_in_nested_multiindex():
    # [FUNCTION DOC]
    # Reindexing a frame like another, when index tuples contain nested tuples with
    # None,
    # yields a frame equal to the other (GH42883).
    # GH42883
    index = MultiIndex.from_tuples([(("a", None), 1), (("b", None), 2)])
    index2 = MultiIndex.from_tuples([(("b", None), 2), (("a", None), 1)])
    df1_dtype = pd.DataFrame([1, 2], index=index)
    df2_dtype = pd.DataFrame([2, 1], index=index2)

    result = df1_dtype.reindex_like(df2_dtype)
    expected = df2_dtype
    tm.assert_frame_equal(result, expected)
