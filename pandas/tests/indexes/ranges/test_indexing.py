# [MODULE DOC]
# Tests for indexing methods of RangeIndex: get_indexer with fill methods, limit
# and decreasing steps, take with fill values and bounds, and where/putmask casting.
import numpy as np
import pytest

from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm


class TestGetIndexer:
    # [CLASS DOC]
    # Groups get_indexer tests.
    def test_get_indexer(self):
        # [FUNCTION DOC]
        # Exact-match positions between a stepped range and a unit range, -1 for absent
        # values.
        index = RangeIndex(start=0, stop=20, step=2)
        target = RangeIndex(10)
        indexer = index.get_indexer(target)
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_pad(self):
        # [FUNCTION DOC]
        # With forward fill, each target gets the preceding entry.
        index = RangeIndex(start=0, stop=20, step=2)
        target = RangeIndex(10)
        indexer = index.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_backfill(self):
        # [FUNCTION DOC]
        # With backfill, each target gets the following entry.
        index = RangeIndex(start=0, stop=20, step=2)
        target = RangeIndex(10)
        indexer = index.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_limit(self):
        # [FUNCTION DOC]
        # A limit of one on forward fill leaves targets more than one step past the end
        # as missing (GH#28631).
        # GH#28631
        idx = RangeIndex(4)
        target = RangeIndex(6)
        result = idx.get_indexer(target, method="pad", limit=1)
        expected = np.array([0, 1, 2, 3, 3, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("stop", [0, -1, -2])
    def test_get_indexer_decreasing(self, stop):
        # [FUNCTION DOC]
        # Parametrized over stop values. For a decreasing range, plain target values map
        # to the expected positions (GH#28678).
        # GH#28678
        index = RangeIndex(7, stop, -3)
        result = index.get_indexer(range(9))
        expected = np.array([-1, 2, -1, -1, 1, -1, -1, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_missing_value_casting_string_dtype(self):
        # [FUNCTION DOC]
        # A string index with a missing value finds none; a mixed target with a boolean
        # gives -1 for it (GH#55833).
        # KNOWN ISSUE: this test builds a string Index rather than a RangeIndex despite living in the RangeIndex test module.
        # GH#55833
        idx = Index(["a", "b", None])
        result = idx.get_indexer([None])
        expected = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = idx.get_indexer([None, True])
        expected = np.array([2, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)


class TestTake:
    # [CLASS DOC]
    # Groups take tests.
    def test_take_preserve_name(self):
        # [FUNCTION DOC]
        # take keeps the name.
        index = RangeIndex(1, 5, name="foo")
        taken = index.take([3, 0, 1])
        assert index.name == taken.name

    def test_take_fill_value(self):
        # [FUNCTION DOC]
        # Wrapping of negative positions works; fill enabled raises a value error that
        # RangeIndex cannot contain NA; fill disabled wraps; invalid negative positions
        # with fill raise the same value error (GH#12631).
        # GH#12631
        idx = RangeIndex(1, 4, name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = Index([2, 1, 3], dtype=np.int64, name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2, 1, 3], dtype=np.int64, name="xxx")
        tm.assert_index_equal(result, expected)

        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

    def test_take_raises_index_error(self):
        # [FUNCTION DOC]
        # Out-of-bounds negative positions raise index errors; the lowest valid negative
        # position works.
        idx = RangeIndex(1, 4, name="xxx")

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

        msg = "index -4 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -4]))

        # no errors
        result = idx.take(np.array([1, -3]))
        expected = Index([2, 1], dtype=np.int64, name="xxx")
        tm.assert_index_equal(result, expected)

    def test_take_accepts_empty_array(self):
        # [FUNCTION DOC]
        # Taking an empty array from a non-empty and an empty range yields an empty
        # integer index with the name.
        idx = RangeIndex(1, 4, name="foo")
        result = idx.take(np.array([]))
        expected = Index([], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

        # empty index
        idx = RangeIndex(0, name="foo")
        result = idx.take(np.array([]))
        expected = Index([], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

    def test_take_accepts_non_int64_array(self):
        # [FUNCTION DOC]
        # Unsigned 32-bit positions are accepted and give 64-bit integer results.
        idx = RangeIndex(1, 4, name="foo")
        result = idx.take(np.array([2, 1], dtype=np.uint32))
        expected = Index([3, 2], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

    def test_take_when_index_has_step(self):
        # [FUNCTION DOC]
        # take on a stepped range yields the right values, including negative positions.
        idx = RangeIndex(1, 11, 3, name="foo")  # [1, 4, 7, 10]
        result = idx.take(np.array([1, 0, -1, -4]))
        expected = Index([4, 1, 10, 1], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

    def test_take_when_index_has_negative_step(self):
        # [FUNCTION DOC]
        # Same for a descending range.
        idx = RangeIndex(11, -4, -2, name="foo")  # [11, 9, 7, 5, 3, 1, -1, -3]
        result = idx.take(np.array([1, 0, -1, -8]))
        expected = Index([9, 11, -3, 11], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)


class TestWhere:
    # [CLASS DOC]
    # Groups where/putmask tests.
    def test_where_putmask_range_cast(self):
        # [FUNCTION DOC]
        # putmask and where with scalar replacement produce an integer index (not a
        # range) with the name (GH#43240).
        # GH#43240
        idx = RangeIndex(0, 5, name="test")

        mask = np.array([True, True, False, False, False])
        result = idx.putmask(mask, 10)
        expected = Index([10, 10, 2, 3, 4], dtype=np.int64, name="test")
        tm.assert_index_equal(result, expected)

        result = idx.where(~mask, 10)
        tm.assert_index_equal(result, expected)
