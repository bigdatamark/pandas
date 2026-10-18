# [MODULE DOC]
# Tests for the sorting helpers in the core sorting module and the safe-sort function in the algorithms module, plus
# integration tests that exercise 64-bit integer overflow of combined group indexes in groupby and merge. Contains
# classes for sorting/groupby, merge, and safe sorting, and module-level tests for group-index deconstruction and
# sorting of mixed or null-containing values. Some tests are marked slow.
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    MultiIndex,
    Series,
    array,
    concat,
    merge,
)
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
    _decons_group_index,
    get_group_index,
    is_int64_overflow_possible,
    lexsort_indexer,
    nargsort,
)


@pytest.fixture
def left_right():
    # [FUNCTION DOC]
    # HELPER: fixture. Builds a frame of about one million rows with seven random integer key columns (A to G, values from
    # minus 1024 to 1023) and a "left" column holding the row sum, then a second frame that is a full random shuffle of it
    # with the last column renamed "right" and negated. Returns the pair (left, right).
    low, high, n = -1 << 10, 1 << 10, 1 << 20
    left = DataFrame(
        np.random.default_rng(2).integers(low, high, (n, 7)), columns=list("ABCDEFG")
    )
    left["left"] = left.sum(axis=1)
    right = left.sample(
        frac=1, random_state=np.random.default_rng(2), ignore_index=True
    )
    right.columns = [*right.columns[:-1].tolist(), "right"]
    right["right"] *= -1
    return left, right


class TestSorting:
    # [CLASS DOC]
    # Tests of groupby ordering under integer overflow of the group index, and of the lexicographic and argument-sort helpers.
    @pytest.mark.slow
    def test_int64_overflow(self):
        # [FUNCTION DOC]
        # Marked slow. Builds a 2500-row frame with eight key columns whose group-count product overflows 64 bits. Groups by the keys
        # in forward and reverse order and sums values; verifies each result index is sorted, and that every group sum agrees between the
        # two orders and with grouping by an array of key tuples; and the lengths agree.
        B = np.concatenate((np.arange(1000), np.arange(1000), np.arange(500)))
        A = np.arange(2500)
        df = DataFrame(
            {
                "A": A,
                "B": B,
                "C": A,
                "D": B,
                "E": A,
                "F": B,
                "G": A,
                "H": B,
                "values": np.random.default_rng(2).standard_normal(2500),
            }
        )

        lg = df.groupby(["A", "B", "C", "D", "E", "F", "G", "H"])
        rg = df.groupby(["H", "G", "F", "E", "D", "C", "B", "A"])

        left = lg.sum()["values"]
        right = rg.sum()["values"]

        exp_index, _ = left.index.sortlevel()
        tm.assert_index_equal(left.index, exp_index)

        exp_index, _ = right.index.sortlevel(0)
        tm.assert_index_equal(right.index, exp_index)

        tups = list(map(tuple, df[["A", "B", "C", "D", "E", "F", "G", "H"]].values))
        tups = com.asarray_tuplesafe(tups)

        expected = df.groupby(tups).sum()["values"]

        for k, v in expected.items():
            assert left[k] == right[k[::-1]]
            assert left[k] == v
        assert len(left) == len(right)

    def test_int64_overflow_groupby_large_range(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Groups a frame of four identical columns with 55109 distinct values and verifies the group count equals the number of rows.
        # GH9096
        values = range(55109)
        data = DataFrame.from_dict({"a": values, "b": values, "c": values, "d": values})
        grouped = data.groupby(["a", "b", "c", "d"])
        assert len(grouped) == len(values)

    @pytest.mark.slow
    @pytest.mark.parametrize("agg", ["mean", "median"])
    def test_int64_overflow_groupby_large_df_shuffled(self, agg):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over the mean and median aggregations. Builds a shuffled frame of random integer keys with duplicated rows and two zero value columns;
        # first asserts the grouping really can overflow the 64-bit group index; verifies the aggregation equals a zero frame indexed by the sorted unique key rows.
        rs = np.random.default_rng(2)
        arr = rs.integers(-1 << 12, 1 << 12, (1 << 15, 5))
        i = rs.choice(len(arr), len(arr) * 4)
        arr = np.vstack((arr, arr[i]))  # add some duplicate rows

        i = rs.permutation(len(arr))
        arr = arr[i]  # shuffle rows

        df = DataFrame(arr, columns=list("abcde"))
        df["jim"], df["joe"] = np.zeros((2, len(df)))
        gr = df.groupby(list("abcde"))

        # verify this is testing what it is supposed to test!
        assert is_int64_overflow_possible(
            tuple(ping.ngroups for ping in gr._grouper.groupings)
        )

        mi = MultiIndex.from_arrays(
            [ar.ravel() for ar in np.array_split(np.unique(arr, axis=0), 5, axis=1)],
            names=list("abcde"),
        )

        res = DataFrame(
            np.zeros((len(mi), 2)), columns=["jim", "joe"], index=mi
        ).sort_index()

        tm.assert_frame_equal(getattr(gr, agg)(), res)

    @pytest.mark.parametrize(
        "order, na_position, exp",
        [
            [
                True,
                "last",
                list(range(5, 105)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                True,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(5, 105)),
            ],
            [
                False,
                "last",
                list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                False,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1)),
            ],
        ],
    )
    def test_lexsort_indexer(self, order, na_position, exp):
        # [FUNCTION DOC]
        # Parametrized over (ascending order flag, NaN position, expected indexer). With a single key of five NaNs, one hundred numbers and five NaNs, verifies the lexicographic
        # sort indexer places NaNs first or last as requested, reversing the non-NaN part when descending, and keeping NaNs in original order. Result must equal the expected array of platform integers.
        keys = [[np.nan] * 5 + list(range(100)) + [np.nan] * 5]
        result = lexsort_indexer(keys, orders=order, na_position=na_position)
        tm.assert_numpy_array_equal(result, np.array(exp, dtype=np.intp))

    @pytest.mark.parametrize(
        "ascending, na_position, exp",
        [
            [
                True,
                "last",
                list(range(5, 105)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                True,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(5, 105)),
            ],
            [
                False,
                "last",
                list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                False,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1)),
            ],
        ],
    )
    def test_nargsort(self, ascending, na_position, exp):
        # [FUNCTION DOC]
        # Parametrized over (ascending, NaN position, expected indexer) with the same layout as the lexsort test. Verifies the argument sort with a stable merge sort on an object-dtype array gives the expected positions (dtype not checked).
        # list places NaNs last, np.array(..., dtype="O") may not place NaNs first
        items = np.array([np.nan] * 5 + list(range(100)) + [np.nan] * 5, dtype="O")

        # mergesort is the most difficult to get right because we want it to be
        # stable.

        # According to numpy/core/tests/test_multiarray, """The number of
        # sorted items must be greater than ~50 to check the actual algorithm
        # because quick and merge sort fall over to insertion sort for small
        # arrays."""

        result = nargsort(
            items, kind="mergesort", ascending=ascending, na_position=na_position
        )
        tm.assert_numpy_array_equal(result, np.array(exp), check_dtype=False)


class TestMerge:
    # [CLASS DOC]
    # Merge tests involving integer overflow of the combined key index.
    def test_int64_overflow_outer_merge(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue about combinatorial explosion. Outer-merges two random thousand-row frames sharing six key columns and verifies the result has two thousand rows (no matches).
        # #2690, combinatorial explosion
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 7)),
            columns=[*list("ABCDEF"), "G1"],
        )
        df2 = DataFrame(
            np.random.default_rng(3).standard_normal((1000, 7)),
            columns=[*list("ABCDEF"), "G2"],
        )
        result = merge(df1, df2, how="outer")
        assert len(result) == 2000

    @pytest.mark.slow
    def test_int64_overflow_check_sum_col(self, left_right):
        # [FUNCTION DOC]
        # Marked slow. Using the large left/right fixture, verifies an outer merge keeps the row count, that the left sum column equals the negated right column, and equals the sum of the key columns of the merged result (name ignored); result Series has no name.
        left, right = left_right

        out = merge(left, right, how="outer")
        assert len(out) == len(left)
        tm.assert_series_equal(out["left"], -out["right"], check_names=False)
        result = out.iloc[:, :-2].sum(axis=1)
        tm.assert_series_equal(out["left"], result, check_names=False)
        assert result.name is None

    @pytest.mark.slow
    def test_int64_overflow_how_merge(self, left_right, join_type):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over the join type fixture. Verifies a sorted merge of the shuffled fixture frames for the given join type equals the sorted outer merge (all rows match, so every join type gives the same set).
        left, right = left_right

        out = merge(left, right, how="outer")
        out.sort_values(out.columns.tolist(), inplace=True)
        tm.assert_frame_equal(out, merge(left, right, how=join_type, sort=True))

    @pytest.mark.slow
    def test_int64_overflow_sort_false_order(self, left_right):
        # [FUNCTION DOC]
        # Marked slow. Verifies a left merge without sorting keeps the left frame's row order and values for the left frame's columns, and the same in the swapped direction.
        left, right = left_right

        # check that left merge w/ sort=False maintains left frame order
        out = merge(left, right, how="left", sort=False)
        tm.assert_frame_equal(left, out[left.columns.tolist()])

        out = merge(right, left, how="left", sort=False)
        tm.assert_frame_equal(right, out[right.columns.tolist()])

    @pytest.mark.slow
    def test_int64_overflow_one_to_many_none_match(self, join_type, sort):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over the join type and sort fixtures. Builds a left frame of random integer keys (first checking the key cardinality product can overflow), duplicates it, builds a right frame with duplicates and some overlap with the left, adds random value columns, and shuffles both. Computes the expected merge manually by dictionaries of key to values and a cross product per key, with NaN where no match; derives the row mask for the join type, then verifies the merge equals the manual result (with sort requested, also that the key columns are sorted stably).
        # one-2-many/none match
        how = join_type
        low, high, n = -1 << 10, 1 << 10, 1 << 11
        left = DataFrame(
            np.random.default_rng(2).integers(low, high, (n, 7)).astype("int64"),
            columns=list("ABCDEFG"),
        )

        # confirm that this is checking what it is supposed to check
        shape = left.apply(Series.nunique).values
        assert is_int64_overflow_possible(shape)

        # add duplicates to left frame
        left = concat([left, left], ignore_index=True)

        right = DataFrame(
            np.random.default_rng(3).integers(low, high, (n // 2, 7)).astype("int64"),
            columns=list("ABCDEFG"),
        )

        # add duplicates & overlap with left to the right frame
        i = np.random.default_rng(4).choice(len(left), n)
        right = concat([right, right, left.iloc[i]], ignore_index=True)

        left["left"] = np.random.default_rng(2).standard_normal(len(left))
        right["right"] = np.random.default_rng(2).standard_normal(len(right))

        # shuffle left & right frames
        left = left.sample(
            frac=1, ignore_index=True, random_state=np.random.default_rng(5)
        )
        right = right.sample(
            frac=1, ignore_index=True, random_state=np.random.default_rng(6)
        )

        # manually compute outer merge
        ldict, rdict = defaultdict(list), defaultdict(list)

        for idx, row in left.set_index(list("ABCDEFG")).iterrows():
            ldict[idx].append(row["left"])

        for idx, row in right.set_index(list("ABCDEFG")).iterrows():
            rdict[idx].append(row["right"])

        vals = []
        for k, lval in ldict.items():
            rval = rdict.get(k, [np.nan])
            for lv, rv in product(lval, rval):
                vals.append((*k, lv, rv))

        for k, rval in rdict.items():
            if k not in ldict:
                vals.extend((*k, np.nan, rv) for rv in rval)

        out = DataFrame(vals, columns=[*list("ABCDEFG"), "left", "right"])
        out = out.sort_values(out.columns.to_list(), ignore_index=True)

        jmask = {
            "left": out["left"].notna(),
            "right": out["right"].notna(),
            "inner": out["left"].notna() & out["right"].notna(),
            "outer": np.ones(len(out), dtype="bool"),
        }

        mask = jmask[how]
        frame = out[mask].sort_values(out.columns.to_list(), ignore_index=True)
        assert mask.all() ^ mask.any() or how == "outer"

        res = merge(left, right, how=how, sort=sort)
        if sort:
            kcols = list("ABCDEFG")
            tm.assert_frame_equal(
                res[kcols], res[kcols].sort_values(kcols, kind="mergesort")
            )

        # as in GH9092 dtypes break with outer/right join
        # 2021-12-18: dtype does not break anymore
        tm.assert_frame_equal(
            frame, res.sort_values(res.columns.to_list(), ignore_index=True)
        )


@pytest.mark.parametrize(
    "codes_list, shape",
    [
        [
            [
                np.tile([0, 1, 2, 3, 0, 1, 2, 3], 100).astype(np.int64),
                np.tile([0, 2, 4, 3, 0, 1, 2, 3], 100).astype(np.int64),
                np.tile([5, 1, 0, 2, 3, 0, 5, 4], 100).astype(np.int64),
            ],
            (4, 5, 6),
        ],
        [
            [
                np.tile(np.arange(10000, dtype=np.int64), 5),
                np.tile(np.arange(10000, dtype=np.int64), 5),
            ],
            (10000, 10000),
        ],
    ],
)
def test_decons(codes_list, shape):
    # [FUNCTION DOC]
    # Parametrized over two (list of integer code arrays, shape) cases including one whose combined size overflows. Verifies that building the combined group index from codes, then deconstructing it with the same shape, returns the original code arrays.
    group_index = get_group_index(codes_list, shape, sort=True, xnull=True)
    codes_list2 = _decons_group_index(group_index, shape)

    for a, b in zip(codes_list, codes_list2, strict=True):
        tm.assert_numpy_array_equal(a, b)


class TestSafeSort:
    # [CLASS DOC]
    # Tests for the safe-sort function, which sorts values (including mixed types) and optionally remaps integer codes.
    @pytest.mark.parametrize(
        "arg, exp",
        [
            [[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]],
            [
                np.array(list("baaacb"), dtype=object),
                np.array(list("aaabbc"), dtype=object),
            ],
            [[], []],
        ],
    )
    def test_basic_sort(self, arg, exp):
        # [FUNCTION DOC]
        # Parametrized over an integer list, an object array of letters, and an empty list. Verifies the sorted array result.
        result = safe_sort(np.array(arg))
        expected = np.array(exp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("verify", [True, False])
    @pytest.mark.parametrize(
        "codes, exp_codes",
        [
            [[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]],
            [[], []],
        ],
    )
    def test_codes(self, verify, codes, exp_codes):
        # [FUNCTION DOC]
        # Parametrized over the verify flag and two code lists (one with a missing sentinel, one empty). Verifies sorting an integer array with codes returns the sorted array and codes remapped to the new positions, keeping the sentinel.
        values = np.array([3, 1, 2, 0, 4])
        expected = np.array([0, 1, 2, 3, 4])

        result, result_codes = safe_sort(
            values, codes, use_na_sentinel=True, verify=verify
        )
        expected_codes = np.array(exp_codes, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_codes_out_of_bound(self):
        # [FUNCTION DOC]
        # Verifies that codes beyond the number of values become the missing sentinel while valid codes are remapped.
        values = np.array([3, 1, 2, 0, 4])
        expected = np.array([0, 1, 2, 3, 4])

        # out of bound indices
        codes = [0, 101, 102, 2, 3, 0, 99, 4]
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True)
        expected_codes = np.array([3, -1, -1, 2, 0, 3, -1, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    @pytest.mark.parametrize("codes", [[-1, -1], [2, -1], [2, 2]])
    def test_codes_empty_array_out_of_bound(self, codes):
        # [FUNCTION DOC]
        # Parametrized over code lists. Verifies that with an empty values array all codes come out as the missing sentinel.
        empty_values = np.array([])
        expected_codes = -np.ones_like(codes, dtype=np.intp)
        _, result_codes = safe_sort(empty_values, codes)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_mixed_integer(self):
        # [FUNCTION DOC]
        # Verifies an object array of integers and strings sorts with the numbers first, then the strings.
        values = np.array(["b", 1, 0, "a", 0, "b"], dtype=object)
        result = safe_sort(values)
        expected = np.array([0, 0, 1, "a", "b", "b"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_integer_with_codes(self):
        # [FUNCTION DOC]
        # Verifies mixed integer and string sorting with codes remaps the codes consistently, keeping the missing sentinel.
        values = np.array(["b", 1, 0, "a"], dtype=object)
        codes = [0, 1, 2, 3, 0, -1, 1]
        result, result_codes = safe_sort(values, codes)
        expected = np.array([0, 1, "a", "b"], dtype=object)
        expected_codes = np.array([3, 1, 0, 2, 3, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_unsortable(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies an object array mixing numbers with a datetime raises a type error saying comparison is not supported between instances.
        # GH 13714
        arr = np.array([1, 2, datetime.now(), 0, 3], dtype=object)
        msg = "'[<>]' not supported between instances of .*"
        with pytest.raises(TypeError, match=msg):
            safe_sort(arr)

    @pytest.mark.parametrize(
        "arg, codes, err, msg",
        [
            [1, None, TypeError, "Only np.ndarray, ExtensionArray, and Index"],
            [np.array([0, 1, 2]), 1, TypeError, "Only list-like objects or None"],
            [np.array([0, 1, 2, 1]), [0, 1], ValueError, "values should be unique"],
        ],
    )
    def test_exceptions(self, arg, codes, err, msg):
        # [FUNCTION DOC]
        # Parametrized over invalid argument combinations. Verifies: a non-array values argument raises a type error naming the accepted types; a non-list-like codes argument raises a type error; and non-unique values together with codes raises a value error.
        with pytest.raises(err, match=msg):
            safe_sort(values=arg, codes=codes)

    @pytest.mark.parametrize(
        "arg, exp", [[[1, 3, 2], [1, 2, 3]], [[1, 3, NA, 2], [1, 2, 3, NA]]]
    )
    def test_extension_array(self, arg, exp):
        # [FUNCTION DOC]
        # Parametrized over lists without and with a missing value. Verifies sorting a nullable integer array returns a nullable integer array sorted with the missing value last.
        a = array(arg, dtype="Int64")
        result = safe_sort(a)
        expected = array(exp, dtype="Int64")
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("verify", [True, False])
    def test_extension_array_codes(self, verify):
        # [FUNCTION DOC]
        # Parametrized over the verify flag. Verifies sorting a nullable integer array with codes returns the sorted array and remapped codes, keeping the sentinel.
        a = array([1, 3, 2], dtype="Int64")
        result, codes = safe_sort(a, [0, 1, -1, 2], use_na_sentinel=True, verify=verify)
        expected_values = array([1, 2, 3], dtype="Int64")
        expected_codes = np.array([0, 2, -1, 1], dtype=np.intp)
        tm.assert_extension_array_equal(result, expected_values)
        tm.assert_numpy_array_equal(codes, expected_codes)


def test_mixed_str_null(nulls_fixture):
    # [FUNCTION DOC]
    # Uses the null-values fixture. Verifies sorting an object array of strings with one null object puts the null last after the sorted strings.
    values = np.array(["b", nulls_fixture, "a", "b"], dtype=object)
    result = safe_sort(values)
    expected = np.array(["a", "b", "b", nulls_fixture], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_safe_sort_multiindex():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies safe sort of a MultiIndex with a nullable integer level containing missing values sorts rows with missing entries last.
    # GH#48412
    arr1 = Series([2, 1, NA, NA], dtype="Int64")
    arr2 = [2, 1, 3, 3]
    midx = MultiIndex.from_arrays([arr1, arr2])
    result = safe_sort(midx)
    expected = MultiIndex.from_arrays(
        [Series([1, 2, NA, NA], dtype="Int64"), [1, 2, 3, 3]]
    )
    tm.assert_index_equal(result, expected)
