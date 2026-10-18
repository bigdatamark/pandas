# [MODULE DOC]
# Tests for Series.sort_index: name, random order, descending, in place, level,
# MultiIndex, algorithm kind, NaN positions, interval index, ignore_index,
# ascending lists, invalid ascending, and the key argument. Defines a fixture for
# sort algorithms. Uses the sort_by_key fixture from conftest. No import-time
# side effects.

import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.fixture(params=["quicksort", "mergesort", "heapsort", "stable"])
def sort_kind(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the sort algorithm names quicksort,
    # mergesort, heapsort and stable; returns the name.
    return request.param


class TestSeriesSortIndex:
    # [CLASS DOC]
    # Groups the tests of Series.sort_index without key. Stateless.
    def test_sort_index_name(self, datetime_series):
        # [FUNCTION DOC]
        # sort_index keeps the Series name.
        result = datetime_series.sort_index(ascending=False)
        assert result.name == datetime_series.name

    def test_sort_index(self, datetime_series):
        # [FUNCTION DOC]
        # A Series with shuffled labels is sorted ascending and descending (and with
        # level 0 or axis 0 as compatibility options); an invalid axis raises a value
        # error that no such axis exists.
        # KNOWN ISSUE: one of the invalid-axis checks calls the value sort method
        # rather than the index sort method.
        datetime_series.index = datetime_series.index._with_freq(None)

        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)

        random_order = datetime_series.reindex(rindex)
        sorted_series = random_order.sort_index()
        tm.assert_series_equal(sorted_series, datetime_series)

        # descending
        sorted_series = random_order.sort_index(ascending=False)
        tm.assert_series_equal(
            sorted_series, datetime_series.reindex(datetime_series.index[::-1])
        )

        # compat on level
        sorted_series = random_order.sort_index(level=0)
        tm.assert_series_equal(sorted_series, datetime_series)

        # compat on axis
        sorted_series = random_order.sort_index(axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)

        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            random_order.sort_values(axis=1)

        sorted_series = random_order.sort_index(level=0, axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)

        with pytest.raises(ValueError, match=msg):
            random_order.sort_index(level=0, axis=1)

    def test_sort_index_inplace(self, datetime_series):
        # [FUNCTION DOC]
        # GH#11402: in place sorting, ascending and descending, of a shuffled Series
        # returns None and matches the sorted version (frequency cleared).
        datetime_series.index = datetime_series.index._with_freq(None)

        # For GH#11402
        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)

        # descending
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=False, inplace=True)

        assert result is None
        expected = datetime_series.reindex(datetime_series.index[::-1])
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)

        # ascending
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=True, inplace=True)

        assert result is None
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)

    def test_sort_index_level(self):
        # [FUNCTION DOC]
        # For a three-level MultiIndex, sort by level name or list of levels sorts
        # the rows; with sort_remaining off, the order is unchanged.
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        res = s.sort_index(level="A")
        tm.assert_series_equal(backwards, res)

        res = s.sort_index(level=["A", "B"])
        tm.assert_series_equal(backwards, res)

        res = s.sort_index(level="A", sort_remaining=False)
        tm.assert_series_equal(s, res)

        res = s.sort_index(level=["A", "B"], sort_remaining=False)
        tm.assert_series_equal(s, res)

    @pytest.mark.parametrize("level", ["A", 0])  # GH#21052
    def test_sort_index_multiindex(self, level):
        # [FUNCTION DOC]
        # GH#21052, parametrized over level name or position: sort by a level with
        # implicit remaining sorting reverses the two rows; with sort_remaining off the
        # order is unchanged (GH#13496).
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        # implicit sort_remaining=True
        res = s.sort_index(level=level)
        tm.assert_series_equal(backwards, res)

        # GH#13496
        # sort has no effect without remaining lvls
        res = s.sort_index(level=level, sort_remaining=False)
        tm.assert_series_equal(s, res)

    def test_sort_index_kind(self, sort_kind):
        # [FUNCTION DOC]
        # GH#14444, GH#13589, parametrized over sort kinds: sorting an index of
        # integers with duplicates on an object Series with no values gives the sorted
        # index.
        # GH#14444 & GH#13589:  Add support for sort algo choosing
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(kind=sort_kind)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position(self):
        # [FUNCTION DOC]
        # NaN in the index goes first or last according to na_position.
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(na_position="first")
        tm.assert_series_equal(expected_series_first, index_sorted_series)

        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)

        index_sorted_series = series.sort_index(na_position="last")
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_sort_index_intervals(self):
        # [FUNCTION DOC]
        # Interval index: sorting ascending keeps order, descending reverses it.
        s = Series(
            [np.nan, 1, 2, 3], IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4])
        )

        result = s.sort_index()
        expected = s
        tm.assert_series_equal(result, expected)

        result = s.sort_index(ascending=False)
        expected = Series(
            [3, 2, 1, np.nan], IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1])
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_list, sorted_list, ascending, ignore_index, output_index",
        [
            ([2, 3, 6, 1], [2, 3, 6, 1], True, True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [2, 3, 6, 1], True, False, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [1, 6, 3, 2], False, True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [1, 6, 3, 2], False, False, [3, 2, 1, 0]),
        ],
    )
    def test_sort_index_ignore_index(
        self, inplace, original_list, sorted_list, ascending, ignore_index, output_index
    ):
        # [FUNCTION DOC]
        # GH 30114, parametrized over in place, ascending flag, and ignore_index: with
        # ignore_index the result has a fresh range index; otherwise the labels follow
        # values; the original Series stays unchanged in the non-in-place and copy case.
        # GH 30114
        ser = Series(original_list)
        expected = Series(sorted_list, index=output_index)
        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        if inplace:
            result_ser = ser.copy()
            result_ser.sort_index(**kwargs)
        else:
            result_ser = ser.sort_index(**kwargs)

        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_sort_index_ascending_list(self):
        # [FUNCTION DOC]
        # GH#16934: sorting a three-level MultiIndex by two levels with ascending as
        # a bool or as a list of bools per level.
        # GH#16934

        # Set up a Series with a three level MultiIndex
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
            [4, 3, 2, 1, 4, 3, 2, 1],
        ]
        tuples = zip(*arrays, strict=True)
        mi = MultiIndex.from_tuples(tuples, names=["first", "second", "third"])
        ser = Series(range(8), index=mi)

        # Sort with boolean ascending
        result = ser.sort_index(level=["third", "first"], ascending=False)
        expected = ser.iloc[[4, 0, 5, 1, 6, 2, 7, 3]]
        tm.assert_series_equal(result, expected)

        # Sort with list of boolean ascending
        result = ser.sort_index(level=["third", "first"], ascending=[False, True])
        expected = ser.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [
            None,
            (True, None),
            (False, "True"),
        ],
    )
    def test_sort_index_ascending_bad_value_raises(self, ascending):
        # [FUNCTION DOC]
        # Parametrized over None, a tuple with None and a tuple with a string: a non-bool
        # ascending value raises a value error that the argument must be of bool type.
        ser = Series(range(10), index=[0, 3, 2, 1, 4, 5, 7, 6, 8, 9])
        match = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            ser.sort_index(ascending=ascending)


class TestSeriesSortIndexKey:
    # [CLASS DOC]
    # Groups the tests of Series.sort_index with the key argument. Stateless.
    def test_sort_index_multiindex_key(self):
        # [FUNCTION DOC]
        # Sorting by a level with a negating key reverses the order; the identity key
        # changes nothing.
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        result = s.sort_index(level="C", key=lambda x: -x)
        tm.assert_series_equal(s, result)

        result = s.sort_index(level="C", key=lambda x: x)  # nothing happens
        tm.assert_series_equal(backwards, result)

    def test_sort_index_multiindex_key_multi_level(self):
        # [FUNCTION DOC]
        # Same for a key applied to several levels.
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]

        result = s.sort_index(level=["A", "C"], key=lambda x: -x)
        tm.assert_series_equal(s, result)

        result = s.sort_index(level=["A", "C"], key=lambda x: x)  # nothing happens
        tm.assert_series_equal(backwards, result)

    def test_sort_index_key(self):
        # [FUNCTION DOC]
        # Sort by index with and without a lowercase key, ascending and descending, for
        # a mixed-case string index.
        series = Series(np.arange(6, dtype="int64"), index=list("aaBBca"))

        result = series.sort_index()
        expected = series.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_series_equal(result, expected)

        result = series.sort_index(key=lambda x: x.str.lower())
        expected = series.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_series_equal(result, expected)

        result = series.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = series.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_key_int(self):
        # [FUNCTION DOC]
        # Key functions on an integer index: negation sorts descending, doubling leaves
        # order.
        series = Series(np.arange(6, dtype="int64"), index=np.arange(6, dtype="int64"))

        result = series.sort_index()
        tm.assert_series_equal(result, series)

        result = series.sort_index(key=lambda x: -x)
        expected = series.sort_index(ascending=False)
        tm.assert_series_equal(result, expected)

        result = series.sort_index(key=lambda x: 2 * x)
        tm.assert_series_equal(result, series)

    def test_sort_index_kind_key(self, sort_kind, sort_by_key):
        # [FUNCTION DOC]
        # GH #14444 and #13589, parametrized over sort kinds and key fixture: the sorted
        # index with a key is the same as without.
        # GH #14444 & #13589:  Add support for sort algo choosing
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(kind=sort_kind, key=sort_by_key)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_kind_neg_key(self, sort_kind):
        # [FUNCTION DOC]
        # GH #14444 and #13589, parametrized over sort kinds: a negating key sorts the
        # index descending.
        # GH #14444 & #13589:  Add support for sort algo choosing
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[4, 3, 3, 2, 1], dtype=object)

        index_sorted_series = series.sort_index(kind=sort_kind, key=lambda x: -x)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position_key(self, sort_by_key):
        # [FUNCTION DOC]
        # Parametrized over key fixtures: NaN placement with a key follows na_position.
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)

        index_sorted_series = series.sort_index(na_position="first", key=sort_by_key)
        tm.assert_series_equal(expected_series_first, index_sorted_series)

        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)

        index_sorted_series = series.sort_index(na_position="last", key=sort_by_key)
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_changes_length_raises(self):
        # [FUNCTION DOC]
        # A key that changes the number of entries raises a value error that the key must
        # not change the shape.
        s = Series([1, 2, 3])
        with pytest.raises(ValueError, match="change the shape"):
            s.sort_index(key=lambda x: x[:1])

    def test_sort_values_key_type(self):
        # [FUNCTION DOC]
        # Keys that use datetime attributes (month, day, year, month name) sort a
        # datetime index in the corresponding order.
        # KNOWN ISSUE: the name says sort_values but the test exercises sort_index with
        # a key function.
        s = Series([1, 2, 3], DatetimeIndex(["2008-10-24", "2008-11-23", "2007-12-22"]))

        result = s.sort_index(key=lambda x: x.month)
        expected = s.iloc[[0, 1, 2]]
        tm.assert_series_equal(result, expected)

        result = s.sort_index(key=lambda x: x.day)
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

        result = s.sort_index(key=lambda x: x.year)
        expected = s.iloc[[2, 0, 1]]
        tm.assert_series_equal(result, expected)

        result = s.sort_index(key=lambda x: x.month_name())
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [
            [True, False],
            [False, True],
        ],
    )
    def test_sort_index_multi_already_monotonic(self, ascending):
        # [FUNCTION DOC]
        # GH 56049, parametrized over [True, False] and [False, True]: for an already
        # lexicographically monotonic MultiIndex, per-level ascending lists still reorder
        # rows accordingly.
        # GH 56049
        mi = MultiIndex.from_product([[1, 2], [3, 4]])
        ser = Series(range(len(mi)), index=mi)
        result = ser.sort_index(ascending=ascending)
        if ascending == [True, False]:
            expected = ser.take([1, 0, 3, 2])
        elif ascending == [False, True]:
            expected = ser.take([2, 3, 0, 1])
        tm.assert_series_equal(result, expected)
