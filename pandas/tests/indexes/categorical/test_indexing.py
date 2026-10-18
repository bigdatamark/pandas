# [MODULE DOC]
# Tests for indexing operations on CategoricalIndex: take (with fill values), get_loc,
# get_indexer and variants, where, and containment checks including NA and intervals.
import numpy as np
import pytest

from pandas.errors import InvalidIndexError

import pandas as pd
from pandas import (
    CategoricalIndex,
    Index,
    IntervalIndex,
    Timestamp,
)
import pandas._testing as tm


class TestTake:
    # [CLASS DOC]
    # Groups take tests for CategoricalIndex; no state.
    def test_take_fill_value(self):
        # [FUNCTION DOC]
        # Verifies take with numeric and string categories: plain take preserves name, categories and
        # orderedness; fill_value true turns the -1 position into a missing value; allow_fill false
        # treats -1 as a regular negative position; positions below -1 with fill raise a value error;
        # positions out of bounds raise an index error.
        # GH 12631

        # numeric category
        idx = CategoricalIndex([1, 2, 3], name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = CategoricalIndex([2, 1, 3], name="xxx")
        tm.assert_index_equal(result, expected)
        tm.assert_categorical_equal(result.values, expected.values)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = CategoricalIndex([2, 1, np.nan], categories=[1, 2, 3], name="xxx")
        tm.assert_index_equal(result, expected)
        tm.assert_categorical_equal(result.values, expected.values)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = CategoricalIndex([2, 1, 3], name="xxx")
        tm.assert_index_equal(result, expected)
        tm.assert_categorical_equal(result.values, expected.values)

        # object category
        idx = CategoricalIndex(
            list("CBA"), categories=list("ABC"), ordered=True, name="xxx"
        )
        result = idx.take(np.array([1, 0, -1]))
        expected = CategoricalIndex(
            list("BCA"), categories=list("ABC"), ordered=True, name="xxx"
        )
        tm.assert_index_equal(result, expected)
        tm.assert_categorical_equal(result.values, expected.values)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = CategoricalIndex(
            ["B", "C", np.nan], categories=list("ABC"), ordered=True, name="xxx"
        )
        tm.assert_index_equal(result, expected)
        tm.assert_categorical_equal(result.values, expected.values)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = CategoricalIndex(
            list("BCA"), categories=list("ABC"), ordered=True, name="xxx"
        )
        tm.assert_index_equal(result, expected)
        tm.assert_categorical_equal(result.values, expected.values)

        msg = (
            "When allow_fill=True and fill_value is not None, all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    def test_take_fill_value_datetime(self):
        # [FUNCTION DOC]
        # Same checks as the fill-value test but for datetime-valued categories: fill produces NaT
        # while keeping all original categories; invalid positions raise the same errors.
        # datetime category
        idx = pd.DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx")
        idx = CategoricalIndex(idx)
        result = idx.take(np.array([1, 0, -1]))
        expected = pd.DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx"
        )
        expected = CategoricalIndex(expected)
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = pd.DatetimeIndex(["2011-02-01", "2011-01-01", "NaT"], name="xxx")
        exp_cats = pd.DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"])
        expected = CategoricalIndex(expected, categories=exp_cats)
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = pd.DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx"
        )
        expected = CategoricalIndex(expected)
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    def test_take_invalid_kwargs(self):
        # [FUNCTION DOC]
        # Verifies that take rejects unsupported keyword arguments: an unknown keyword raises a type
        # error, while the out and mode parameters raise value errors saying they are not supported.
        idx = CategoricalIndex([1, 2, 3], name="foo")
        indices = [1, 0, -1]

        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)

        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode="clip")


class TestGetLoc:
    # [CLASS DOC]
    # Groups get_loc tests for CategoricalIndex; no state.
    def test_get_loc(self):
        # [FUNCTION DOC]
        # Verifies get_loc matches a plain object Index in three layouts: unique labels give an integer,
        # non-unique non-contiguous labels give a boolean mask, non-unique contiguous labels give a
        # slice; absent labels (or labels that are only categories) raise a key error naming the label.
        # GH 12531
        cidx1 = CategoricalIndex(list("abcde"), categories=list("edabc"))
        idx1 = Index(list("abcde"))
        assert cidx1.get_loc("a") == idx1.get_loc("a")
        assert cidx1.get_loc("e") == idx1.get_loc("e")

        for i in [cidx1, idx1]:
            with pytest.raises(KeyError, match="'NOT-EXIST'"):
                i.get_loc("NOT-EXIST")

        # non-unique
        cidx2 = CategoricalIndex(list("aacded"), categories=list("edabc"))
        idx2 = Index(list("aacded"))

        # results in bool array
        res = cidx2.get_loc("d")
        tm.assert_numpy_array_equal(res, idx2.get_loc("d"))
        tm.assert_numpy_array_equal(
            res, np.array([False, False, False, True, False, True])
        )
        # unique element results in scalar
        res = cidx2.get_loc("e")
        assert res == idx2.get_loc("e")
        assert res == 4

        for i in [cidx2, idx2]:
            with pytest.raises(KeyError, match="'NOT-EXIST'"):
                i.get_loc("NOT-EXIST")

        # non-unique, sliceable
        cidx3 = CategoricalIndex(list("aabbb"), categories=list("abc"))
        idx3 = Index(list("aabbb"))

        # results in slice
        res = cidx3.get_loc("a")
        assert res == idx3.get_loc("a")
        assert res == slice(0, 2, None)

        res = cidx3.get_loc("b")
        assert res == idx3.get_loc("b")
        assert res == slice(2, 5, None)

        for i in [cidx3, idx3]:
            with pytest.raises(KeyError, match="'c'"):
                i.get_loc("c")

    def test_get_loc_unique(self):
        # [FUNCTION DOC]
        # Verifies get_loc on a unique index returns an integer position.
        cidx = CategoricalIndex(list("abc"))
        result = cidx.get_loc("b")
        assert result == 1

    def test_get_loc_monotonic_nonunique(self):
        # [FUNCTION DOC]
        # Verifies get_loc on a monotonic non-unique index returns a slice.
        cidx = CategoricalIndex(list("abbc"))
        result = cidx.get_loc("b")
        expected = slice(1, 3, None)
        assert result == expected

    def test_get_loc_nonmonotonic_nonunique(self):
        # [FUNCTION DOC]
        # Verifies get_loc on a non-monotonic non-unique index returns a boolean mask.
        cidx = CategoricalIndex(list("abcb"))
        result = cidx.get_loc("b")
        expected = np.array([False, True, False, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_loc_nan(self):
        # [FUNCTION DOC]
        # Regression test: get_loc with NaN returns the position of the missing entry.
        # GH#41933
        ci = CategoricalIndex(["A", "B", np.nan])
        res = ci.get_loc(np.nan)

        assert res == 2


class TestGetIndexer:
    # [CLASS DOC]
    # Groups get_indexer tests for CategoricalIndex; no state.
    def test_get_indexer_base(self):
        # [FUNCTION DOC]
        # Verifies self indexing returns the identity positions as platform integers, and an invalid
        # fill method name raises a value error.
        # Determined by cat ordering.
        idx = CategoricalIndex(list("cab"), categories=list("cab"))
        expected = np.arange(len(idx), dtype=np.intp)

        actual = idx.get_indexer(idx)
        tm.assert_numpy_array_equal(expected, actual)

        with pytest.raises(ValueError, match="Invalid fill method"):
            idx.get_indexer(idx, method="invalid")

    def test_get_indexer_requires_unique(self):
        # [FUNCTION DOC]
        # Verifies get_indexer with a target with duplicates (random samples or an exact duplicate
        # list) raises an invalid-index error saying reindexing needs uniquely valued indexes.
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        oidx = Index(np.array(ci))

        msg = "Reindexing only valid with uniquely valued Index objects"

        for n in [1, 2, 5, len(ci)]:
            finder = oidx[np.random.default_rng(2).integers(0, len(ci), size=n)]

            with pytest.raises(InvalidIndexError, match=msg):
                ci.get_indexer(finder)

        # see gh-17323
        #
        # Even when indexer is equal to the
        # members in the index, we should
        # respect duplicates instead of taking
        # the fast-track path.
        for finder in [list("aabbca"), list("aababca")]:
            with pytest.raises(InvalidIndexError, match=msg):
                ci.get_indexer(finder)

    def test_get_indexer_non_unique(self):
        # [FUNCTION DOC]
        # Verifies that for categorical, list and plain Index targets get_indexer raises the
        # unique-index error, whereas the non-unique variant returns the matched positions with minus
        # one for absent labels.
        idx1 = CategoricalIndex(list("aabcde"), categories=list("edabc"))
        idx2 = CategoricalIndex(list("abf"))

        for indexer in [idx2, list("abf"), Index(list("abf"))]:
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                idx1.get_indexer(indexer)

            r1, _ = idx1.get_indexer_non_unique(indexer)
            expected = np.array([0, 1, 2, -1], dtype=np.intp)
            tm.assert_almost_equal(r1, expected)

    def test_get_indexer_method(self):
        # [FUNCTION DOC]
        # Verifies that requesting pad, backfill or nearest methods raises a not-implemented error
        # naming the method and CategoricalIndex.
        idx1 = CategoricalIndex(list("aabcde"), categories=list("edabc"))
        idx2 = CategoricalIndex(list("abf"))

        msg = "method pad not yet implemented for CategoricalIndex"
        with pytest.raises(NotImplementedError, match=msg):
            idx2.get_indexer(idx1, method="pad")
        msg = "method backfill not yet implemented for CategoricalIndex"
        with pytest.raises(NotImplementedError, match=msg):
            idx2.get_indexer(idx1, method="backfill")

        msg = "method nearest not yet implemented for CategoricalIndex"
        with pytest.raises(NotImplementedError, match=msg):
            idx2.get_indexer(idx1, method="nearest")

    def test_get_indexer_array(self):
        # [FUNCTION DOC]
        # Verifies get_indexer with an object array of timestamps against a timestamp-categories index
        # gives the matching positions.
        arr = np.array(
            [Timestamp("1999-12-31 00:00:00"), Timestamp("2000-12-31 00:00:00")],
            dtype=object,
        )
        cats = [Timestamp("1999-12-31 00:00:00"), Timestamp("2000-12-31 00:00:00")]
        ci = CategoricalIndex(cats, categories=cats, ordered=False, dtype="category")
        result = ci.get_indexer(arr)
        expected = np.array([0, 1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_same_categories_same_order(self):
        # [FUNCTION DOC]
        # Verifies get_indexer with a target categorical sharing the same categories in the same order
        # returns positions by label.
        ci = CategoricalIndex(["a", "b"], categories=["a", "b"])

        result = ci.get_indexer(CategoricalIndex(["b", "b"], categories=["a", "b"]))
        expected = np.array([1, 1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_same_categories_different_order(self):
        # [FUNCTION DOC]
        # Regression test: target categories in a different order still match by label, not by code.
        # https://github.com/pandas-dev/pandas/issues/19551
        ci = CategoricalIndex(["a", "b"], categories=["a", "b"])

        result = ci.get_indexer(CategoricalIndex(["b", "b"], categories=["b", "a"]))
        expected = np.array([1, 1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_nans_in_index_and_target(self):
        # [FUNCTION DOC]
        # Regression test: get_indexer handles missing values in both index and target, matching NaN to
        # NaN and giving minus one for absent values.
        # GH 45361
        ci = CategoricalIndex([1, 2, np.nan, 3])
        other1 = [2, 3, 4, np.nan]
        res1 = ci.get_indexer(other1)
        expected1 = np.array([1, 3, -1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(res1, expected1)
        other2 = [1, 4, 2, 3]
        res2 = ci.get_indexer(other2)
        expected2 = np.array([0, -1, 1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(res2, expected2)


class TestWhere:
    # [CLASS DOC]
    # Groups where tests for CategoricalIndex; no state.
    def test_where(self, listlike_box):
        # [FUNCTION DOC]
        # Uses the listlike_box fixture (list, tuple, array types, Series) to wrap the boolean
        # condition. Verifies an all-True condition returns the index unchanged and a condition with
        # first element False yields a missing first value while keeping categories.
        klass = listlike_box

        i = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        cond = [True] * len(i)
        expected = i
        result = i.where(klass(cond))
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * (len(i) - 1)
        expected = CategoricalIndex([np.nan, *i[1:].tolist()], categories=i.categories)
        result = i.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_where_non_categories(self):
        # [FUNCTION DOC]
        # Verifies where with a replacement outside the categories casts the index to object, while
        # the Categorical private method raises a type error about a new category.
        ci = CategoricalIndex(["a", "b", "c", "d"])
        mask = np.array([True, False, True, False])

        result = ci.where(mask, 2)
        expected = Index(["a", 2, "c", 2], dtype=object)
        tm.assert_index_equal(result, expected)

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            # Test the Categorical method directly
            ci._data._where(mask, 2)


class TestContains:
    # [CLASS DOC]
    # Groups containment tests for CategoricalIndex; no state.
    def test_contains(self):
        # [FUNCTION DOC]
        # Verifies membership: present labels are contained; unused categories, absent labels, NaN and
        # integer codes are not.
        ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"), ordered=False)

        assert "a" in ci
        assert "z" not in ci
        assert "e" not in ci
        assert np.nan not in ci

        # assert codes NOT in index
        assert 0 not in ci
        assert 1 not in ci

    def test_contains_nan(self):
        # [FUNCTION DOC]
        # Verifies NaN is contained in an index holding a missing value.
        ci = CategoricalIndex([*list("aabbca"), np.nan], categories=list("cabdef"))
        assert np.nan in ci

    @pytest.mark.parametrize("unwrap", [True, False])
    def test_contains_na_dtype(self, unwrap):
        # [FUNCTION DOC]
        # Parametrized over whether to test the index or its underlying Categorical. Verifies that for
        # categories of datetimes, timedeltas and periods, generic NA values (NaN, None, NaT) are
        # contained, while the NaT of a mismatched kind (numpy datetime or timedelta) is contained only
        # for the matching kind.
        dti = pd.date_range("2016-01-01", periods=100).insert(0, pd.NaT)
        pi = dti.to_period("D")
        tdi = dti - dti[-1]
        ci = CategoricalIndex(dti)

        obj = ci
        if unwrap:
            obj = ci._data

        assert np.nan in obj
        assert None in obj
        assert pd.NaT in obj
        assert np.datetime64("NaT", "ns") in obj
        assert np.timedelta64("NaT", "ns") not in obj

        obj2 = CategoricalIndex(tdi)
        if unwrap:
            obj2 = obj2._data

        assert np.nan in obj2
        assert None in obj2
        assert pd.NaT in obj2
        assert np.datetime64("NaT", "ns") not in obj2
        assert np.timedelta64("NaT", "ns") in obj2

        obj3 = CategoricalIndex(pi)
        if unwrap:
            obj3 = obj3._data

        assert np.nan in obj3
        assert None in obj3
        assert pd.NaT in obj3
        assert np.datetime64("NaT", "ns") not in obj3
        assert np.timedelta64("NaT", "ns") not in obj3

    @pytest.mark.parametrize(
        "item, expected",
        [
            (pd.Interval(0, 1), True),
            (1.5, True),
            (pd.Interval(0.5, 1.5), False),
            ("a", False),
            (Timestamp(1), False),
            (pd.Timedelta(1), False),
        ],
        ids=str,
    )
    def test_contains_interval(self, item, expected):
        # [FUNCTION DOC]
        # Parametrized over (item, expected) with ids from string form. Verifies membership for an
        # interval-categories index: an exact interval, a point inside an interval, and non-matching
        # interval, string, Timestamp and Timedelta values; the result must be a real boolean.
        # GH 23705
        ci = CategoricalIndex(IntervalIndex.from_breaks(range(3)))
        result = item in ci
        assert result is expected

    def test_contains_list(self):
        # [FUNCTION DOC]
        # Regression test: a string absent from the index is not contained, and testing membership of a
        # list raises a type error about an unhashable type.
        # GH#21729
        idx = CategoricalIndex([1, 2, 3])

        assert "a" not in idx

        with pytest.raises(TypeError, match="unhashable type"):
            ["a"] in idx

        with pytest.raises(TypeError, match="unhashable type"):
            ["a", "b"] in idx
