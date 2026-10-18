# [MODULE DOC]
# Tests for reindex on CategoricalIndex: duplicate-label errors, duplicate targets, empty
# index, and added categories.
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    Interval,
)
import pandas._testing as tm


class TestReindex:
    # [CLASS DOC]
    # Groups reindex tests for CategoricalIndex; no state.
    def test_reindex_list_non_unique(self):
        # [FUNCTION DOC]
        # Verifies reindexing an index that has duplicate labels with a list raises a value error
        # that the axis has duplicate labels.
        # GH#11586
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(["a", "c"])

    def test_reindex_categorical_non_unique(self):
        # [FUNCTION DOC]
        # Same as the list case but the target is a Categorical array.
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(Categorical(["a", "c"]))

    def test_reindex_list_non_unique_unused_category(self):
        # [FUNCTION DOC]
        # Same as the list case with an index that also has an unused category.
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(["a", "c"])

    def test_reindex_categorical_non_unique_unused_category(self):
        # [FUNCTION DOC]
        # Same as the Categorical case with an index that also has an unused category.
        msg = "cannot reindex on an axis with duplicate labels"
        ci = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        with pytest.raises(ValueError, match=msg):
            ci.reindex(Categorical(["a", "c"]))

    def test_reindex_duplicate_target(self):
        # [FUNCTION DOC]
        # Regression test: reindexing a unique index with a list target containing duplicates gives a
        # plain Index and a position indexer; with a CategoricalIndex target the result stays
        # categorical with the target's categories.
        # See GH25459
        cat = CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c", "d"])
        res, indexer = cat.reindex(["a", "c", "c"])
        exp = Index(["a", "c", "c"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))

        res, indexer = cat.reindex(
            CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        )
        exp = CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))

    def test_reindex_empty_index(self):
        # [FUNCTION DOC]
        # Verifies reindexing an empty CategoricalIndex onto new labels returns the labels as a plain
        # Index and an all minus-one indexer.
        # See GH16770
        c = CategoricalIndex([])
        res, indexer = c.reindex(["a", "b"])
        tm.assert_index_equal(res, Index(["a", "b"]), exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([-1, -1], dtype=np.intp))

    def test_reindex_categorical_added_category(self):
        # [FUNCTION DOC]
        # Regression test: reindexing an ordered interval-valued CategoricalIndex onto a superset
        # categorical index returns that target.
        # GH 42424
        ci = CategoricalIndex(
            [Interval(0, 1, closed="right"), Interval(1, 2, closed="right")],
            ordered=True,
        )
        ci_add = CategoricalIndex(
            [
                Interval(0, 1, closed="right"),
                Interval(1, 2, closed="right"),
                Interval(2, 3, closed="right"),
                Interval(3, 4, closed="right"),
            ],
            ordered=True,
        )
        result, _ = ci.reindex(ci_add)
        expected = ci_add
        tm.assert_index_equal(expected, result)
