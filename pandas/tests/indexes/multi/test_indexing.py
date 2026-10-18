# [MODULE DOC]
# Tests for lookup and indexing behavior of MultiIndex: slice_locs, putmask,
# get_indexer (with fill methods, missing values, categorical levels), element
# access, get_loc / get_loc_level, where, membership tests, slice bounds with
# missing values, the integer-packing lookup engine and get_locs ordering.
# Uses the shared "idx" fixture from the multi-index conftest plus
# pandas testing helpers. No import-time side effects.
from collections import namedtuple
from datetime import timedelta
import re

import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.errors import InvalidIndexError

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


class TestSliceLocs:
    # [CLASS DOC]
    # Groups tests of the slice_locs method on a MultiIndex.
    def test_slice_locs_partial(self, idx):
        # [FUNCTION DOC]
        # Sorts the fixture index by its first level, then checks that slice_locs with
        # partial tuples, a missing start/end (none), and bare first-level labels
        # returns the expected (start, stop) positions.
        sorted_idx, _ = idx.sortlevel(0)

        result = sorted_idx.slice_locs(("foo", "two"), ("qux", "one"))
        assert result == (1, 5)

        result = sorted_idx.slice_locs(None, ("qux", "one"))
        assert result == (0, 5)

        result = sorted_idx.slice_locs(("foo", "two"), None)
        assert result == (1, len(sorted_idx))

        result = sorted_idx.slice_locs("bar", "baz")
        assert result == (2, 4)

    def test_slice_locs(self):
        # [FUNCTION DOC]
        # Stacks a business-day frame to build a two-level index, then verifies that
        # slice_locs on timestamps (exact, and shifted by seconds so they fall between
        # labels) yields positions matching the equivalent row-slice of the frame.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((50, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=50, freq="B"),
        )
        stacked = df.stack()
        idx = stacked.index

        slob = slice(*idx.slice_locs(df.index[5], df.index[15]))
        sliced = stacked[slob]
        expected = df[5:16].stack()
        tm.assert_almost_equal(sliced.values, expected.values)

        slob = slice(
            *idx.slice_locs(
                df.index[5] + timedelta(seconds=30),
                df.index[15] - timedelta(seconds=30),
            )
        )
        sliced = stacked[slob]
        expected = df[6:15].stack()
        tm.assert_almost_equal(sliced.values, expected.values)

    def test_slice_locs_with_type_mismatch(self):
        # [FUNCTION DOC]
        # Verifies slice_locs raises a type error beginning with "Level type mismatch"
        # when a key's type does not match the level (tuples too short, a timedelta
        # against a string level, etc.), for both datetime-based and string-based
        # stacked indexes.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs((1, 3))
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(df.index[5] + timedelta(seconds=30), (5, 2))
        df = DataFrame(
            np.ones((5, 5)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(5)], name="a"),
        )
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(timedelta(seconds=30))
        # TODO: Try creating a UnicodeDecodeError in exception message
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(df.index[1], (16, "a"))

    def test_slice_locs_not_sorted(self):
        # [FUNCTION DOC]
        # On a MultiIndex that is not lexsorted, slice_locs with a key longer than the
        # lexsort depth must raise a key error; after sorting by level 0 the same call
        # must succeed (only checked for no error).
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        msg = "[Kk]ey length.*greater than MultiIndex lexsort depth"
        with pytest.raises(KeyError, match=msg):
            index.slice_locs((1, 0, 1), (2, 1, 0))

        # works
        sorted_index, _ = index.sortlevel(0)
        # should there be a test case here???
        sorted_index.slice_locs((1, 0, 1), (2, 1, 0))

    def test_slice_locs_not_contained(self):
        # [FUNCTION DOC]
        # Checks slice_locs when the bounds are not present in the index (searchsorted
        # semantics), for tuple and scalar bounds and for bounds outside the whole
        # range; asserts the exact (start, stop) pairs.
        # some searchsorted action

        index = MultiIndex(
            levels=[[0, 2, 4, 6], [0, 2, 4]],
            codes=[[0, 0, 0, 1, 1, 2, 3, 3, 3], [0, 1, 2, 1, 2, 2, 0, 1, 2]],
        )

        result = index.slice_locs((1, 0), (5, 2))
        assert result == (3, 6)

        result = index.slice_locs(1, 5)
        assert result == (3, 6)

        result = index.slice_locs((2, 2), (5, 2))
        assert result == (3, 6)

        result = index.slice_locs(2, 5)
        assert result == (3, 6)

        result = index.slice_locs((1, 0), (6, 3))
        assert result == (3, 8)

        result = index.slice_locs(-1, 10)
        assert result == (0, len(index))

    @pytest.mark.parametrize(
        "index_arr,expected,start_idx,end_idx",
        [
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, None),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, "b"),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, ("b", "e")),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), None),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), "c"),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), ("c", "e")),
        ],
    )
    def test_slice_locs_with_missing_value(
        self, index_arr, expected, start_idx, end_idx
    ):
        # [FUNCTION DOC]
        # Parametrized over index arrays containing missing values, expected bounds and
        # start/end keys (scalar missing, tuples, none). Verifies slice_locs returns
        # the expected pair (GH 19132).
        # issue 19132
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.slice_locs(start=start_idx, end=end_idx)
        assert result == expected


class TestPutmask:
    # [CLASS DOC]
    # Groups tests of MultiIndex.putmask.
    def test_putmask_with_wrong_mask(self, idx):
        # [FUNCTION DOC]
        # A mask whose length differs from the index (longer, shorter) or a non-array
        # string must raise a value error about mask and data needing the same size
        # (GH18368).
        # GH18368

        msg = "putmask: mask and data must be the same size"
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) + 1, np.bool_), 1)

        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) - 1, np.bool_), 1)

        with pytest.raises(ValueError, match=msg):
            idx.putmask("foo", 1)

    def test_putmask_multiindex_other(self):
        # [FUNCTION DOC]
        # putmask where the replacement value is itself a MultiIndex: masked positions
        # take values from the other index and unmasked positions keep original values
        # (GH#43212).
        # GH#43212 `value` is also a MultiIndex

        left = MultiIndex.from_tuples([(np.nan, 6), (np.nan, 6), ("a", 4)])
        right = MultiIndex.from_tuples([("a", 1), ("a", 1), ("d", 1)])
        mask = np.array([True, True, False])

        result = left.putmask(mask, right)

        expected = MultiIndex.from_tuples([right[0], right[1], left[2]])
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Parametrized over numeric extension dtypes. putmask with a same-length
        # MultiIndex replacement must preserve the extension dtype of the first level
        # (GH#49830).
        # GH#49830
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5, 6, 7], dtype=any_numeric_ea_dtype), [-1, -2, -3]]
        )
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype_shorter_value(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Same as the keep-dtype test but the replacement MultiIndex has length one;
        # it is broadcast over the masked positions and dtype is preserved (GH#49830).
        # GH#49830
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5], dtype=any_numeric_ea_dtype), [-1]]
        )
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        tm.assert_index_equal(result, expected)


class TestGetIndexer:
    # [CLASS DOC]
    # Groups tests of MultiIndex.get_indexer.
    def test_get_indexer(self):
        # [FUNCTION DOC]
        # Exercises get_indexer between MultiIndex objects with exact matching and with
        # pad/ffill and backfill/bfill methods (including reversed targets), passing a
        # plain array of tuples or list, and checks that duplicated non-Multi indexes
        # raise an invalid-index error about reindexing only valid with uniquely
        # valued Index objects.
        major_axis = Index(np.arange(4))
        minor_axis = Index(np.arange(2))

        major_codes = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.intp)
        minor_codes = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.intp)

        index = MultiIndex(
            levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
        )
        idx1 = index[:5]
        idx2 = index[[1, 3, 5]]

        r1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, np.array([1, 3, -1], dtype=np.intp))

        r1 = idx2.get_indexer(idx1, method="pad")
        e1 = np.array([-1, 0, 0, 1, 1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

        r2 = idx2.get_indexer(idx1[::-1], method="pad")
        tm.assert_almost_equal(r2, e1[::-1])

        rffill1 = idx2.get_indexer(idx1, method="ffill")
        tm.assert_almost_equal(r1, rffill1)

        r1 = idx2.get_indexer(idx1, method="backfill")
        e1 = np.array([0, 0, 1, 1, 2], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

        r2 = idx2.get_indexer(idx1[::-1], method="backfill")
        tm.assert_almost_equal(r2, e1[::-1])

        rbfill1 = idx2.get_indexer(idx1, method="bfill")
        tm.assert_almost_equal(r1, rbfill1)

        # pass non-MultiIndex
        r1 = idx1.get_indexer(idx2.values)
        rexp1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, rexp1)

        r1 = idx1.get_indexer([1, 2, 3])
        assert (r1 == [-1, -1, -1]).all()

        # create index with duplicates
        idx1 = Index(list(range(10)) + list(range(10)))
        idx2 = Index(list(range(20)))

        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            idx1.get_indexer(idx2)

    def test_get_indexer_nearest(self):
        # [FUNCTION DOC]
        # The nearest method and the tolerance argument are not implemented for
        # MultiIndex and must raise not-implemented errors with specific messages.
        midx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        msg = (
            "method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365"
        )
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(["a"], method="nearest")
        msg = "tolerance not implemented yet for MultiIndex"
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(["a"], method="pad", tolerance=2)

    def test_get_indexer_categorical_time(self):
        # [FUNCTION DOC]
        # get_indexer of a MultiIndex built from categorical levels (one of them
        # datetimes) against itself returns the identity positions (GH 21390).
        # https://github.com/pandas-dev/pandas/issues/21390
        midx = MultiIndex.from_product(
            [
                Categorical(["a", "b", "c"]),
                Categorical(date_range("2012-01-01", periods=3, freq="h")),
            ]
        )
        result = midx.get_indexer(midx)
        tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))

    @pytest.mark.parametrize(
        "index_arr,labels,expected",
        [
            (
                [[1, np.nan, 2], [3, 4, 5]],
                [1, np.nan, 2],
                np.array([-1, -1, -1], dtype=np.intp),
            ),
            ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)),
            ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)),
            (
                [[1, 2, 3], [np.nan, 4, 5]],
                [np.nan, 4, 5],
                np.array([-1, -1, -1], dtype=np.intp),
            ),
        ],
    )
    def test_get_indexer_with_missing_value(self, index_arr, labels, expected):
        # [FUNCTION DOC]
        # Parametrized over index arrays with missing values, target labels and the
        # expected positions; verifies missing values match only when given inside full
        # tuples and that non-tuple labels yield -1 (GH 19132).
        # issue 19132
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.get_indexer(labels)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_methods(self):
        # [FUNCTION DOC]
        # Checks get_indexer without a fill method, with backfill/bfill and with
        # pad/ffill for a target index falling between entries of a 2-level source;
        # legacy aliases must give identical results (GH 29896).
        # https://github.com/pandas-dev/pandas/issues/29896
        # test getting an indexer for another index with different methods
        # confirms that getting an indexer without a filling method, getting an
        # indexer and backfilling, and getting an indexer and padding all behave
        # correctly in the case where all of the target values fall in between
        # several levels in the MultiIndex into which they are getting an indexer
        #
        # visually, the MultiIndexes used in this test are:
        # mult_idx_1:
        #  0: -1 0
        #  1:    2
        #  2:    3
        #  3:    4
        #  4:  0 0
        #  5:    2
        #  6:    3
        #  7:    4
        #  8:  1 0
        #  9:    2
        # 10:    3
        # 11:    4
        #
        # mult_idx_2:
        #  0: 0 1
        #  1:   3
        #  2:   4
        mult_idx_1 = MultiIndex.from_product([[-1, 0, 1], [0, 2, 3, 4]])
        mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, 6, 7], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="backfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        # ensure the legacy "bfill" option functions identically to "backfill"
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="pad")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

        # ensure the legacy "ffill" option functions identically to "pad"
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    @pytest.mark.parametrize("method", ["pad", "ffill", "backfill", "bfill", "nearest"])
    def test_get_indexer_methods_raise_for_non_monotonic(self, method):
        # [FUNCTION DOC]
        # Parametrized over the fill methods. On a non-monotonic MultiIndex, pad, ffill,
        # backfill and bfill raise a value error requiring a monotonic index, while
        # nearest raises a not-implemented error.
        # 53452
        mi = MultiIndex.from_arrays([[0, 4, 2], [0, 4, 2]])
        if method == "nearest":
            err = NotImplementedError
            msg = "not implemented yet for MultiIndex"
        else:
            err = ValueError
            msg = "index must be monotonic increasing or decreasing"
        with pytest.raises(err, match=msg):
            mi.get_indexer([(1, 1)], method=method)

    def test_get_indexer_three_or_more_levels(self):
        # [FUNCTION DOC]
        # Verifies get_indexer with no fill, backfill and pad between two sorted unique
        # three-level indexes, in both directions, and asserts the ordering
        # relationships between the entries first as a sanity check (GH 29896).
        # KNOWN ISSUE: the explanatory comment lists the fifth row of the second index as (2, 7, 6) but the code constructs (2, 7, 7).
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests get_indexer() on MultiIndexes with 3+ levels
        # visually, these are
        # mult_idx_1:
        #  0: 1 2 5
        #  1:     7
        #  2:   4 5
        #  3:     7
        #  4:   6 5
        #  5:     7
        #  6: 3 2 5
        #  7:     7
        #  8:   4 5
        #  9:     7
        # 10:   6 5
        # 11:     7
        #
        # mult_idx_2:
        #  0: 1 1 8
        #  1: 1 5 9
        #  2: 1 6 7
        #  3: 2 1 6
        #  4: 2 7 6
        #  5: 2 7 8
        #  6: 3 6 8
        mult_idx_1 = MultiIndex.from_product([[1, 3], [2, 4, 6], [5, 7]])
        mult_idx_2 = MultiIndex.from_tuples(
            [
                (1, 1, 8),
                (1, 5, 9),
                (1, 6, 7),
                (2, 1, 6),
                (2, 7, 7),
                (2, 7, 8),
                (3, 6, 8),
            ]
        )
        # sanity check
        assert mult_idx_1.is_monotonic_increasing
        assert mult_idx_1.is_unique
        assert mult_idx_2.is_monotonic_increasing
        assert mult_idx_2.is_unique

        # show the relationships between the two
        assert mult_idx_2[0] < mult_idx_1[0]
        assert mult_idx_1[3] < mult_idx_2[1] < mult_idx_1[4]
        assert mult_idx_1[5] == mult_idx_2[2]
        assert mult_idx_1[5] < mult_idx_2[3] < mult_idx_1[6]
        assert mult_idx_1[5] < mult_idx_2[4] < mult_idx_1[6]
        assert mult_idx_1[5] < mult_idx_2[5] < mult_idx_1[6]
        assert mult_idx_1[-1] < mult_idx_2[6]

        indexer_no_fill = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, -1, 5, -1, -1, -1, -1], dtype=indexer_no_fill.dtype)
        tm.assert_almost_equal(expected, indexer_no_fill)

        # test with backfilling
        indexer_backfilled = mult_idx_1.get_indexer(mult_idx_2, method="backfill")
        expected = np.array([0, 4, 5, 6, 6, 6, -1], dtype=indexer_backfilled.dtype)
        tm.assert_almost_equal(expected, indexer_backfilled)

        # now, the same thing, but forward-filled (aka "padded")
        indexer_padded = mult_idx_1.get_indexer(mult_idx_2, method="pad")
        expected = np.array([-1, 3, 5, 5, 5, 5, 11], dtype=indexer_padded.dtype)
        tm.assert_almost_equal(expected, indexer_padded)

        # now, do the indexing in the other direction
        assert mult_idx_2[0] < mult_idx_1[0] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[1] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[2] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[3] < mult_idx_2[1]
        assert mult_idx_2[1] < mult_idx_1[4] < mult_idx_2[2]
        assert mult_idx_2[2] == mult_idx_1[5]
        assert mult_idx_2[5] < mult_idx_1[6] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[7] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[8] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[9] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[10] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[11] < mult_idx_2[6]

        indexer = mult_idx_2.get_indexer(mult_idx_1)
        expected = np.array(
            [-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1], dtype=indexer.dtype
        )
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_2.get_indexer(mult_idx_1, method="bfill")
        expected = np.array(
            [1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6], dtype=backfill_indexer.dtype
        )
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_2.get_indexer(mult_idx_1, method="pad")
        expected = np.array(
            [0, 0, 0, 0, 1, 2, 5, 5, 5, 5, 5, 5], dtype=pad_indexer.dtype
        )
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_crossing_levels(self):
        # [FUNCTION DOC]
        # Corner case where get_indexer with bfill/ffill must carry across levels with
        # correct tuple ordering on 4-level indexes; checks exact, backfilled and
        # padded results (GH 29896).
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests a corner case with get_indexer() with MultiIndexes where, when we
        # need to "carry" across levels, proper tuple ordering is respected
        #
        # the MultiIndexes used in this test, visually, are:
        # mult_idx_1:
        #  0: 1 1 1 1
        #  1:       2
        #  2:     2 1
        #  3:       2
        #  4: 1 2 1 1
        #  5:       2
        #  6:     2 1
        #  7:       2
        #  8: 2 1 1 1
        #  9:       2
        # 10:     2 1
        # 11:       2
        # 12: 2 2 1 1
        # 13:       2
        # 14:     2 1
        # 15:       2
        #
        # mult_idx_2:
        #  0: 1 3 2 2
        #  1: 2 3 2 2
        mult_idx_1 = MultiIndex.from_product([[1, 2]] * 4)
        mult_idx_2 = MultiIndex.from_tuples([(1, 3, 2, 2), (2, 3, 2, 2)])

        # show the tuple orderings, which get_indexer() should respect
        assert mult_idx_1[7] < mult_idx_2[0] < mult_idx_1[8]
        assert mult_idx_1[-1] < mult_idx_2[1]

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, -1], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([8, -1], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([7, 15], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_kwarg_validation(self):
        # [FUNCTION DOC]
        # Passing a limit or a tolerance without a fill method must raise a value error
        # saying the argument is only valid for pad, backfill or nearest (GH#41918).
        # GH#41918
        mi = MultiIndex.from_product([range(3), ["A", "B"]])

        msg = "limit argument only valid if doing pad, backfill or nearest"
        with pytest.raises(ValueError, match=msg):
            mi.get_indexer(mi[:-1], limit=4)

        msg = "tolerance argument only valid if doing pad, backfill or nearest"
        with pytest.raises(ValueError, match=msg):
            mi.get_indexer(mi[:-1], tolerance="piano")

    def test_get_indexer_nan(self):
        # [FUNCTION DOC]
        # Indexers between two product indexes differing by a missing value in the
        # second level return -1 for the NaN row and the matching position otherwise,
        # in both directions (GH#37222). Dtype of the result is not compared.
        # GH#37222
        idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
        idx2 = MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"])
        expected = np.array([-1, 1])
        result = idx2.get_indexer(idx1)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        result = idx1.get_indexer(idx2)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)


def test_getitem(idx):
    # [FUNCTION DOC]
    # Tests element access on the fixture index: scalar position returns a tuple,
    # slices and integer/boolean list or array selections return indexes equal to
    # the expected subset.
    # scalar
    assert idx[2] == ("bar", "one")

    # slice
    result = idx[2:5]
    expected = idx[[2, 3, 4]]
    assert result.equals(expected)

    # boolean
    result = idx[[True, False, True, False, True, True]]
    result2 = idx[np.array([True, False, True, False, True, True])]
    expected = idx[[0, 2, 4, 5]]
    assert result.equals(expected)
    assert result2.equals(expected)


def test_getitem_group_select(idx):
    # [FUNCTION DOC]
    # After sorting by level 0, get_loc of a first-level label returns the slice
    # covering that group.
    sorted_idx, _ = idx.sortlevel(0)
    assert sorted_idx.get_loc("baz") == slice(3, 4)
    assert sorted_idx.get_loc("foo") == slice(0, 2)


@pytest.mark.parametrize("box", [list, Index])
def test_getitem_bool_index_all(box):
    # [FUNCTION DOC]
    # Parametrized over list and Index as the boolean container. Selecting with an
    # all-true mask returns the full index and a mixed mask returns the matching
    # subset (GH#22533).
    # GH#22533
    ind1 = box([True] * 5)
    idx = MultiIndex.from_tuples([(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)])
    tm.assert_index_equal(idx[ind1], idx)

    ind2 = box([True, False, True, False, False])
    expected = MultiIndex.from_tuples([(10, 1), (30, 3)])
    tm.assert_index_equal(idx[ind2], expected)


@pytest.mark.parametrize("box", [list, Index])
def test_getitem_bool_index_single(box):
    # [FUNCTION DOC]
    # Parametrized over list and Index. On a one-element MultiIndex, a true mask
    # returns the same index and a false mask returns an empty MultiIndex with
    # empty integer levels (GH#22533).
    # GH#22533
    ind1 = box([True])
    idx = MultiIndex.from_tuples([(10, 1)])
    tm.assert_index_equal(idx[ind1], idx)

    ind2 = box([False])
    expected = MultiIndex(
        levels=[np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
        codes=[[], []],
    )
    tm.assert_index_equal(idx[ind2], expected)


class TestGetLoc:
    # [CLASS DOC]
    # Groups tests of MultiIndex.get_loc and get_loc_level.
    def test_get_loc(self, idx):
        # [FUNCTION DOC]
        # Full-key lookups return an integer position, partial keys return slices, and
        # missing keys raise key errors whose messages show the key; covers the fixture
        # index and a three-level index.
        assert idx.get_loc(("foo", "two")) == 1
        assert idx.get_loc(("baz", "two")) == 3
        with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
            idx.get_loc(("bar", "two"))
        with pytest.raises(KeyError, match=r"^'quux'$"):
            idx.get_loc("quux")

        # 3 levels
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        with pytest.raises(KeyError, match=r"^\(1, 1\)$"):
            index.get_loc((1, 1))
        assert index.get_loc((2, 0)) == slice(3, 5)

    def test_get_loc_duplicates(self):
        # [FUNCTION DOC]
        # Checks get_loc on a flat Index with duplicates (slice for a monotonic
        # duplicate run, integer for unique, key error for missing).
        # KNOWN ISSUE: despite living in the MultiIndex test module, this test builds plain Index objects, not MultiIndex.
        index = Index([2, 2, 2, 2])
        result = index.get_loc(2)
        expected = slice(0, 4)
        assert result == expected

        index = Index(["c", "a", "a", "b", "b"])
        rs = index.get_loc("c")
        xp = 0
        assert rs == xp

        with pytest.raises(KeyError, match="2"):
            index.get_loc(2)

    def test_get_loc_level(self):
        # [FUNCTION DOC]
        # Checks get_loc_level with partial and full keys: returned location (slice or
        # integer), the remaining index (none when all levels are consumed), key errors
        # for missing and unused labels (GH 22221), and a slice key on a lower level.
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        loc, new_index = index.get_loc_level((0, 1))
        expected = slice(1, 2)
        exp_index = index[expected].droplevel(0).droplevel(0)
        assert loc == expected
        assert new_index.equals(exp_index)

        loc, new_index = index.get_loc_level((0, 1, 0))
        expected = 1
        assert loc == expected
        assert new_index is None

        with pytest.raises(KeyError, match=r"^\(2, 2\)$"):
            index.get_loc_level((2, 2))
        # GH 22221: unused label
        with pytest.raises(KeyError, match=r"^2$"):
            index.drop(2).get_loc_level(2)
        # Unused label on unsorted level:
        with pytest.raises(KeyError, match=r"^2$"):
            index.drop(1, level=2).get_loc_level(2, level=2)

        index = MultiIndex(
            levels=[[2000], list(range(4))],
            codes=[np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])],
        )
        result, new_index = index.get_loc_level((2000, slice(None, None)))
        expected = slice(None, None)
        assert result == expected
        assert new_index.equals(index.droplevel(0))

    @pytest.mark.parametrize("dtype1", [int, float, bool, str])
    @pytest.mark.parametrize("dtype2", [int, float, bool, str])
    def test_get_loc_multiple_dtypes(self, dtype1, dtype2):
        # [FUNCTION DOC]
        # Parametrized over pairs of level dtypes (int, float, bool, str). get_loc of
        # an element taken from the index itself returns its position (GH 18520).
        # GH 18520
        levels = [np.array([0, 1]).astype(dtype1), np.array([0, 1]).astype(dtype2)]
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(idx[2]) == 2

    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("dtypes", [[int, float], [float, int]])
    def test_get_loc_implicit_cast(self, level, dtypes):
        # [FUNCTION DOC]
        # Parametrized over level and dtype combinations. A key whose component has a
        # different numeric type (int vs float) than the level still locates the right
        # row (GH 18818, GH 15994).
        # GH 18818, GH 15994 : as flat index, cast int to float and vice-versa
        levels = [["a", "b"], ["c", "d"]]
        key = ["b", "d"]
        lev_dtype, key_dtype = dtypes
        levels[level] = np.array([0, 1], dtype=lev_dtype)
        key[level] = key_dtype(1)
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(tuple(key)) == 3

    @pytest.mark.parametrize("dtype", [bool, object])
    def test_get_loc_cast_bool(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over bool and object dtype for the first level. For bool dtype
        # integers are not cast to bool and lookups raise key errors; for object dtype
        # python equality makes zero equal false and one equal true. Lookups with bool
        # keys against integer second level always raise key errors (GH 19086).
        # GH 19086 : int is casted to bool, but not vice-versa (for object dtype)
        #  With bool dtype, we don't cast in either direction.
        levels = [Index([False, True], dtype=dtype), np.arange(2, dtype="int64")]
        idx = MultiIndex.from_product(levels)

        if dtype is bool:
            with pytest.raises(KeyError, match=r"^\(0, 1\)$"):
                assert idx.get_loc((0, 1)) == 1
            with pytest.raises(KeyError, match=r"^\(1, 0\)$"):
                assert idx.get_loc((1, 0)) == 2
        else:
            # We use python object comparisons, which treat 0 == False and 1 == True
            assert idx.get_loc((0, 1)) == 1
            assert idx.get_loc((1, 0)) == 2

        with pytest.raises(KeyError, match=r"^\(False, True\)$"):
            idx.get_loc((False, True))
        with pytest.raises(KeyError, match=r"^\(True, False\)$"):
            idx.get_loc((True, False))

    @pytest.mark.parametrize("level", [0, 1])
    def test_get_loc_nan(self, level, nulls_fixture):
        # [FUNCTION DOC]
        # Parametrized over level position and the null-values fixture. A key with a
        # null in either level locates the row correctly (GH 18485).
        # GH 18485 : NaN in MultiIndex
        levels = [["a", "b"], ["c", "d"]]
        key = ["b", "d"]
        levels[level] = np.array([0, nulls_fixture], dtype=type(nulls_fixture))
        key[level] = nulls_fixture
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(tuple(key)) == 3

    def test_get_loc_missing_nan(self):
        # [FUNCTION DOC]
        # Partial scalar key returns a slice; absent scalar or NaN raises key errors;
        # a list key raises an invalid-index error (GH 8569).
        # GH 8569
        idx = MultiIndex.from_arrays([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(idx.get_loc(1), slice)
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        with pytest.raises(KeyError, match=r"^nan$"):
            idx.get_loc(np.nan)
        with pytest.raises(InvalidIndexError, match=r"\[nan\]"):
            # listlike/non-hashable raises TypeError
            idx.get_loc([np.nan])

    def test_get_loc_with_values_including_missing_values(self):
        # [FUNCTION DOC]
        # get_loc with NaN as partial key returns a slice for product indexes and a
        # boolean mask for a single-level index; tuple keys with NaN return slices.
        # issue 19132
        idx = MultiIndex.from_product([[np.nan, 1]] * 2)
        expected = slice(0, 2, None)
        assert idx.get_loc(np.nan) == expected

        idx = MultiIndex.from_arrays([[np.nan, 1, 2, np.nan]])
        expected = np.array([True, False, False, True])
        tm.assert_numpy_array_equal(idx.get_loc(np.nan), expected)

        idx = MultiIndex.from_product([[np.nan, 1]] * 3)
        expected = slice(2, 4, None)
        assert idx.get_loc((np.nan, 1)) == expected

    def test_get_loc_duplicates2(self):
        # [FUNCTION DOC]
        # Partial key on a first level with a repeated label returns the slice of that
        # label.
        # KNOWN ISSUE: the in-code TODO suggests this duplicates test_get_loc_duplicates, which does not actually use a MultiIndex.
        # TODO: de-duplicate with test_get_loc_duplicates above?
        index = MultiIndex(
            levels=[["D", "B", "C"], [0, 26, 27, 37, 57, 67, 75, 82]],
            codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]],
            names=["tag", "day"],
        )

        assert index.get_loc("D") == slice(0, 3)

    def test_get_loc_past_lexsort_depth(self, performance_warning):
        # [FUNCTION DOC]
        # Looking up a key longer than the lexsort depth works but emits a performance
        # warning (via the fixture); the result is the single-row slice (GH#30053).
        # GH#30053
        idx = MultiIndex(
            levels=[["a"], [0, 7], [1]],
            codes=[[0, 0], [1, 0], [0, 0]],
            names=["x", "y", "z"],
            sortorder=0,
        )
        key = ("a", 7)

        with tm.assert_produces_warning(performance_warning):
            # PerformanceWarning: indexing past lexsort depth may impact performance
            result = idx.get_loc(key)

        assert result == slice(0, 1, None)

    def test_multiindex_get_loc_list_raises(self):
        # [FUNCTION DOC]
        # get_loc with an empty list raises an invalid-index error (GH#35878).
        # GH#35878
        idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        msg = r"\[\]"
        with pytest.raises(InvalidIndexError, match=msg):
            idx.get_loc([])

    def test_get_loc_nested_tuple_raises_keyerror(self):
        # [FUNCTION DOC]
        # A key containing a nested tuple must raise a key error (not a type error)
        # whose message matches the key text.
        # raise KeyError, not TypeError
        mi = MultiIndex.from_product([range(3), range(4), range(5), range(6)])
        key = ((2, 3, 4), "foo")

        with pytest.raises(KeyError, match=re.escape(str(key))):
            mi.get_loc(key)


class TestWhere:
    # [CLASS DOC]
    # Groups tests of MultiIndex.where.
    def test_where(self):
        # [FUNCTION DOC]
        # where is unsupported on MultiIndex and raises a not-implemented error.
        i = MultiIndex.from_tuples([("A", 1), ("A", 2)])

        msg = r"\.where is not supported for MultiIndex operations"
        with pytest.raises(NotImplementedError, match=msg):
            i.where(True)

    def test_where_array_like(self, listlike_box):
        # [FUNCTION DOC]
        # Same as above, with the condition supplied via each list-like container from
        # the listlike_box fixture.
        mi = MultiIndex.from_tuples([("A", 1), ("A", 2)])
        cond = [False, True]
        msg = r"\.where is not supported for MultiIndex operations"
        with pytest.raises(NotImplementedError, match=msg):
            mi.where(listlike_box(cond))


class TestContains:
    # [CLASS DOC]
    # Groups tests of membership (the in operator) on MultiIndex.
    def test_contains_top_level(self):
        # [FUNCTION DOC]
        # A top-level label is contained in the index but not in the lookup engine.
        midx = MultiIndex.from_product([["A", "B"], [1, 2]])
        assert "A" in midx
        assert "A" not in midx._engine

    def test_contains_with_nat(self):
        # [FUNCTION DOC]
        # Index containing a NaT in a datetime level: full-key membership of the
        # timestamp and of every value in the index holds.
        # MI with a NaT
        mi = MultiIndex(
            levels=[["C"], date_range("2012-01-01", periods=5)],
            codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]],
            names=[None, "B"],
        )
        assert ("C", pd.Timestamp("2012-01-01")) in mi
        for val in mi.values:
            assert val in mi

    def test_contains(self, idx):
        # [FUNCTION DOC]
        # Membership of present and absent tuples and of none on the fixture index.
        assert ("foo", "two") in idx
        assert ("bar", "two") not in idx
        assert None not in idx

    def test_contains_with_missing_value(self):
        # [FUNCTION DOC]
        # Missing values as top-level or tuple components are contained only where
        # actually present (GH#19132).
        # GH#19132
        idx = MultiIndex.from_arrays([[1, np.nan, 2]])
        assert np.nan in idx

        idx = MultiIndex.from_arrays([[1, 2], [np.nan, 3]])
        assert np.nan not in idx
        assert (1, np.nan) in idx

    def test_multiindex_contains_dropped(self):
        # [FUNCTION DOC]
        # After dropping a label, it is no longer contained in the index although the
        # levels still keep it; checked for integers and strings (GH#19027).
        # GH#19027
        # test that dropped MultiIndex levels are not in the MultiIndex
        # despite continuing to be in the MultiIndex's levels
        idx = MultiIndex.from_product([[1, 2], [3, 4]])
        assert 2 in idx
        idx = idx.drop(2)

        # drop implementation keeps 2 in the levels
        assert 2 in idx.levels[0]
        # but it should no longer be in the index itself
        assert 2 not in idx

        # also applies to strings
        idx = MultiIndex.from_product([["a", "b"], ["c", "d"]])
        assert "a" in idx
        idx = idx.drop("a")
        assert "a" in idx.levels[0]
        assert "a" not in idx

    def test_contains_td64_level(self):
        # [FUNCTION DOC]
        # Membership with timedelta level: timestamp-like element and string
        # representation present, unrelated string absent (GH#24570).
        # GH#24570
        tx = pd.timedelta_range("09:30:00", "16:00:00", freq="30 min")
        idx = MultiIndex.from_arrays([tx, np.arange(len(tx))])
        assert tx[0] in idx
        assert "element_not_exit" not in idx
        assert "0 day 09:30:00" in idx

    def test_large_mi_contains(self, monkeypatch):
        # [FUNCTION DOC]
        # With the engine size cutoff patched small via monkeypatch, a large index
        # does not contain an absent tuple (GH#10645). Patch is undone afterwards.
        # GH#10645
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 10)
            result = MultiIndex.from_arrays([range(10), range(10)])
            assert (10, 0) not in result


def test_timestamp_multiindex_indexer():
    # [FUNCTION DOC]
    # Label-based loc slicing with a string date slice on a datetime first level
    # of a three-level index returns the expected series (GH 26944).
    # https://github.com/pandas-dev/pandas/issues/26944
    idx = MultiIndex.from_product(
        [
            date_range("2019-01-01T00:15:33", periods=100, freq="h", name="date"),
            ["x"],
            [3],
        ]
    )
    df = DataFrame({"foo": np.arange(len(idx))}, idx)
    result = df.loc[pd.IndexSlice["2019-1-2":, "x", :], "foo"]
    qidx = MultiIndex.from_product(
        [
            date_range(
                start="2019-01-02T00:15:33",
                end="2019-01-05T03:15:33",
                freq="h",
                name="date",
            ),
            ["x"],
            [3],
        ]
    )
    should_be = pd.Series(data=np.arange(24, len(qidx) + 24), index=qidx, name="foo")
    tm.assert_series_equal(result, should_be)


@pytest.mark.parametrize(
    "index_arr,expected,target,algo",
    [
        ([[np.nan, "a", "b"], ["c", "d", "e"]], 0, np.nan, "left"),
        ([[np.nan, "a", "b"], ["c", "d", "e"]], 1, (np.nan, "c"), "right"),
        ([["a", "b", "c"], ["d", np.nan, "d"]], 1, ("b", np.nan), "left"),
    ],
)
def test_get_slice_bound_with_missing_value(index_arr, expected, target, algo):
    # [FUNCTION DOC]
    # Parametrized over index arrays with missing values, expected bounds, targets
    # and sides. Checks get_slice_bound handles NaN labels (issue 19132).
    # issue 19132
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.get_slice_bound(target, side=algo)
    assert result == expected


@pytest.mark.parametrize(
    "index_arr,expected,start_idx,end_idx",
    [
        ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 2, None), np.nan, 1),
        ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 3, None), np.nan, (2, 5)),
        ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), 3),
        ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), (3, 5)),
    ],
)
def test_slice_indexer_with_missing_value(index_arr, expected, start_idx, end_idx):
    # [FUNCTION DOC]
    # Parametrized over index arrays with missing values and start/end keys. Checks
    # slice_indexer returns the expected slice (issue 19132).
    # issue 19132
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.slice_indexer(start=start_idx, end=end_idx)
    assert result == expected


@pytest.mark.parametrize(
    "N, expected_dtype",
    [
        (1, "uint8"),  # 2*4*N = 8
        (2, "uint16"),  # 2*4*N = 16
        (4, "uint32"),  # 2*4*N = 32
        (8, "uint64"),  # 2*4*N = 64
        (10, "object"),  # 2*4*N = 80
    ],
)
def test_pyint_engine(N, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over multiplier N and expected engine dtype. Builds keys whose
    # combined level codes need 8 to over 64 bits and verifies the engine's value
    # dtype grows (uint8 up to uint64, then Python object), and that get_loc and
    # get_indexer return correct positions without collisions, including for a
    # missing key (GH#18519).
    # GH#18519 : when combinations of codes cannot be represented in 64
    # bits, the index underlying the MultiIndex engine works with Python
    # integers, rather than uint64.
    keys = [
        tuple(arr)
        for arr in [
            [0] * 4 * N,
            [1] * 4 * N,
            [np.nan] * N + [0] * 3 * N,
            [0] * N + [1] * 3 * N,
            [np.nan] * N + [1] * 2 * N + [0] * N,
        ]
    ]
    # Each level contains 3 elements (NaN, 0, 1), and it's represented
    # in 2 bits to store 4 possible values (0=notfound, 1=NaN, 2=0, 3=1), for
    # a total of 2*N*4 = 80 > 64 bits where N=10 and the number of levels is N*4.
    # If we were using a 64 bit engine and truncating the first levels, the
    # fourth and fifth keys would collide; if truncating the last levels, the
    # fifth and sixth; if rotating bits rather than shifting, the third and fifth.

    index = MultiIndex.from_tuples(keys)
    assert index._engine.values.dtype == expected_dtype

    for idx, key_value in enumerate(keys):
        assert index.get_loc(key_value) == idx

        expected = np.arange(idx + 1, dtype=np.intp)
        result = index.get_indexer([keys[i] for i in expected])
        tm.assert_numpy_array_equal(result, expected)

    # With missing key:
    idces = range(len(keys))
    expected = np.array([-1, *list(idces)], dtype=np.intp)
    missing = tuple([0, 1, 0, 1] * N)
    result = index.get_indexer([missing] + [keys[i] for i in idces])
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "keys,expected",
    [
        ((slice(None), [5, 4]), [1, 0]),
        ((slice(None), [4, 5]), [0, 1]),
        (([True, False, True], [4, 6]), [0, 2]),
        (([True, False, True], [6, 4]), [0, 2]),
        ((2, [4, 5]), [0, 1]),
        ((2, [5, 4]), [1, 0]),
        (([2], [4, 5]), [0, 1]),
        (([2], [5, 4]), [1, 0]),
    ],
)
def test_get_locs_reordering(keys, expected):
    # [FUNCTION DOC]
    # Parametrized over key tuples and expected positions. get_locs honors the
    # order of list-like selectors rather than index order (GH48384).
    # GH48384
    idx = MultiIndex.from_arrays(
        [
            [2, 2, 1],
            [4, 5, 6],
        ]
    )
    result = idx.get_locs(keys)
    expected = np.array(expected, dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_get_indexer_for_multiindex_with_nans(nulls_fixture):
    # [FUNCTION DOC]
    # Using the null-values fixture, get_indexer between indexes that differ by a
    # null in the second level returns -1 for the null row, in both directions
    # (GH37222).
    # GH37222
    idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
    idx2 = MultiIndex.from_product([["A"], [nulls_fixture, 2.0]], names=["id1", "id2"])

    result = idx2.get_indexer(idx1)
    expected = np.array([-1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)

    result = idx1.get_indexer(idx2)
    expected = np.array([-1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_get_loc_namedtuple_behaves_like_tuple():
    # [FUNCTION DOC]
    # Lookups with named tuples and plain tuples find the same positions, for both
    # a MultiIndex built from named tuples or plain tuples and its flat
    # (tuple-object) index (GH57922).
    # GH57922
    NamedIndex = namedtuple("NamedIndex", ("a", "b"))
    multi_idx = MultiIndex.from_tuples(
        [NamedIndex("i1", "i2"), NamedIndex("i3", "i4"), NamedIndex("i5", "i6")]
    )
    for idx in (multi_idx, multi_idx.to_flat_index()):
        assert idx.get_loc(NamedIndex("i1", "i2")) == 0
        assert idx.get_loc(NamedIndex("i3", "i4")) == 1
        assert idx.get_loc(NamedIndex("i5", "i6")) == 2
        assert idx.get_loc(("i1", "i2")) == 0
        assert idx.get_loc(("i3", "i4")) == 1
        assert idx.get_loc(("i5", "i6")) == 2
    multi_idx = MultiIndex.from_tuples([("i1", "i2"), ("i3", "i4"), ("i5", "i6")])
    for idx in (multi_idx, multi_idx.to_flat_index()):
        assert idx.get_loc(NamedIndex("i1", "i2")) == 0
        assert idx.get_loc(NamedIndex("i3", "i4")) == 1
        assert idx.get_loc(NamedIndex("i5", "i6")) == 2
        assert idx.get_loc(("i1", "i2")) == 0
        assert idx.get_loc(("i3", "i4")) == 1
        assert idx.get_loc(("i5", "i6")) == 2
