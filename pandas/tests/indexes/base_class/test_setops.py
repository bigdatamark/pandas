# [MODULE DOC]
# Set-operation tests (union, intersection, difference, symmetric difference) for the base
# object-dtype Index, including sort handling, mixed types, name preservation and structured
# (record) arrays.
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Series,
)
import pandas._testing as tm
from pandas.core.algorithms import safe_sort


def equal_contents(arr1, arr2) -> bool:
    # [FUNCTION DOC]
    # HELPER: compares two iterables by their sets of unique elements; returns True when equal.
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)


class TestIndexSetOps:
    # [CLASS DOC]
    # Groups set-operation tests for the base Index; no state.
    @pytest.mark.parametrize(
        "method", ["union", "intersection", "difference", "symmetric_difference"]
    )
    def test_setops_sort_validation(self, method):
        # [FUNCTION DOC]
        # Parametrized over the four set operations. Verifies an invalid sort argument raises a value
        # error saying the sort keyword has restricted values, and that sort=True is accepted.
        idx1 = Index(["a", "b"])
        idx2 = Index(["b", "c"])

        with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
            getattr(idx1, method)(idx2, sort=2)

        # sort=True is supported as of GH#??
        getattr(idx1, method)(idx2, sort=True)

    def test_setops_preserve_object_dtype(self):
        # [FUNCTION DOC]
        # Verifies intersection and union of an object-dtype integer Index keep object dtype, for both
        # monotonic and reversed second operands, including the private union helper returning an
        # ndarray.
        idx = Index([1, 2, 3], dtype=object)
        result = idx.intersection(idx[1:])
        expected = idx[1:]
        tm.assert_index_equal(result, expected)

        # if other is not monotonic increasing, intersection goes through
        #  a different route
        result = idx.intersection(idx[1:][::-1])
        tm.assert_index_equal(result, expected)

        result = idx._union(idx[1:], sort=None)
        expected = idx
        tm.assert_numpy_array_equal(result, expected.values)

        result = idx.union(idx[1:], sort=None)
        tm.assert_index_equal(result, expected)

        # if other is not monotonic increasing, _union goes through
        #  a different route
        result = idx._union(idx[1:][::-1], sort=None)
        tm.assert_numpy_array_equal(result, expected.values)

        result = idx.union(idx[1:][::-1], sort=None)
        tm.assert_index_equal(result, expected)

    def test_union_base(self):
        # [FUNCTION DOC]
        # Verifies union of overlapping mixed int/str indexes gives the sorted-by-type result.
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[3:]
        second = index[:5]

        result = first.union(second)

        expected = Index([0, 1, 2, "a", "b", "c"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    def test_union_different_type_base(self, klass):
        # [FUNCTION DOC]
        # Parametrized over array, Series and list. Verifies union with a non-Index second operand
        # has the same contents as the full index.
        # GH 10149
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[3:]
        second = index[:5]

        result = first.union(klass(second.values))

        assert equal_contents(result, index)

    def test_union_sort_other_incomparable(self):
        # [FUNCTION DOC]
        # Regression test: union with incomparable elements (int and Timestamp) emits a runtime warning
        # about unsupported comparison for default and sort=None, and none for sort=False; the result
        # equals the original.
        # https://github.com/pandas-dev/pandas/issues/24959
        idx = Index([1, pd.Timestamp("2000")])
        # default (sort=None)
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            result = idx.union(idx[:1])

        tm.assert_index_equal(result, idx)

        # sort=None
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            result = idx.union(idx[:1], sort=None)
        tm.assert_index_equal(result, idx)

        # sort=False
        result = idx.union(idx[:1], sort=False)
        tm.assert_index_equal(result, idx)

    def test_union_sort_other_incomparable_true(self):
        # [FUNCTION DOC]
        # Verifies that union with sort=True on incomparable elements raises a type error.
        idx = Index([1, pd.Timestamp("2000")])
        with pytest.raises(TypeError, match=".*"):
            idx.union(idx[:1], sort=True)

    def test_intersection_equal_sort_true(self):
        # [FUNCTION DOC]
        # Verifies the intersection of an Index with itself and sort=True returns it sorted.
        idx = Index(["c", "a", "b"])
        sorted_ = Index(["a", "b", "c"])
        tm.assert_index_equal(idx.intersection(idx, sort=True), sorted_)

    def test_intersection_base(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Verifies intersection of mixed int/str indexes: sorted when sort is
        # None, otherwise in original order.
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:5]
        second = index[:3]

        expected = Index([0, 1, "a"]) if sort is None else Index([0, "a", 1])
        result = first.intersection(second, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    def test_intersection_different_type_base(self, klass, sort):
        # [FUNCTION DOC]
        # Parametrized over array, Series, list and the sort fixture. Verifies intersection with a
        # non-Index operand has the same contents as the second operand.
        # GH 10149
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:5]
        second = index[:3]

        result = first.intersection(klass(second.values), sort=sort)
        assert equal_contents(result, second)

    def test_intersection_nosort(self):
        # [FUNCTION DOC]
        # Verifies intersection with a list keeps the order of the first operand (no sorting) for
        # string labels.
        result = Index(["c", "b", "a"]).intersection(["b", "a"])
        expected = Index(["b", "a"])
        tm.assert_index_equal(result, expected)

    def test_intersection_equal_sort(self):
        # [FUNCTION DOC]
        # Verifies intersection of an Index with itself returns it unchanged for sort False and None.
        idx = Index(["c", "a", "b"])
        tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
        tm.assert_index_equal(idx.intersection(idx, sort=None), idx)

    def test_intersection_str_dates(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Verifies intersection of a datetime-object Index with a string Index
        # is empty.
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        i1 = Index(dt_dates, dtype=object)
        i2 = Index(["aa"], dtype=object)
        result = i2.intersection(i1, sort=sort)

        assert len(result) == 0

    @pytest.mark.parametrize(
        "index2,expected_arr",
        [(["B", "D"], ["B"]), (["B", "D", "A"], ["A", "B"])],
    )
    def test_intersection_non_monotonic_non_unique(self, index2, expected_arr, sort):
        # [FUNCTION DOC]
        # Parametrized over a second index and expected labels, plus the sort fixture. Verifies
        # intersection of a non-monotonic non-unique Index returns unique matches, sorted when sort is
        # None.
        # non-monotonic non-unique
        index1 = Index(["A", "B", "A", "C"])
        expected = Index(expected_arr)
        result = index1.intersection(Index(index2), sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_difference_base(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Verifies difference of overlapping mixed indexes; when sort is None the
        # expected value is sorted with the safe sorting helper.
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:4]
        second = index[3:]

        result = first.difference(second, sort)
        expected = Index([0, "a", 1])
        if sort is None:
            expected = Index(safe_sort(expected))
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self):
        # [FUNCTION DOC]
        # Verifies symmetric difference of overlapping mixed int/str indexes gives the sorted
        # union of non-shared labels.
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:4]
        second = index[3:]

        result = first.symmetric_difference(second)
        expected = Index([0, 1, 2, "a", "c"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "method,expected,sort",
        [
            (
                "intersection",
                np.array(
                    [(1, "A"), (2, "A"), (1, "B"), (2, "B")],
                    dtype=[("num", int), ("let", "S1")],
                ),
                False,
            ),
            (
                "intersection",
                np.array(
                    [(1, "A"), (1, "B"), (2, "A"), (2, "B")],
                    dtype=[("num", int), ("let", "S1")],
                ),
                None,
            ),
            (
                "union",
                np.array(
                    [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")],
                    dtype=[("num", int), ("let", "S1")],
                ),
                None,
            ),
        ],
    )
    def test_tuple_union_bug(self, method, expected, sort):
        # [FUNCTION DOC]
        # Parametrized over (method, expected record array, sort). Regression test: set operations on
        # Indexes built from structured numpy arrays (numeric plus one-byte string fields) return a
        # one-dimensional result equal to the expected, with order depending on sort.
        index1 = Index(
            np.array(
                [(1, "A"), (2, "A"), (1, "B"), (2, "B")],
                dtype=[("num", int), ("let", "S1")],
            )
        )
        index2 = Index(
            np.array(
                [(1, "A"), (2, "A"), (1, "B"), (2, "B"), (1, "C"), (2, "C")],
                dtype=[("num", int), ("let", "S1")],
            )
        )

        result = getattr(index1, method)(index2, sort=sort)
        assert result.ndim == 1

        expected = Index(expected)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("first_list", [["b", "a"], []])
    @pytest.mark.parametrize("second_list", [["a", "b"], []])
    @pytest.mark.parametrize(
        "first_name, second_name, expected_name",
        [("A", "B", None), (None, "B", None), ("A", None, None)],
    )
    def test_union_name_preservation(
        self, first_list, second_list, first_name, second_name, expected_name, sort
    ):
        # [FUNCTION DOC]
        # Parametrized over first/second lists (including empty), names, and the sort fixture.
        # Verifies union name handling: the name is kept only when both names agree (otherwise None),
        # and the union is sorted when sort is None and both lists are non-empty; otherwise compared
        # after sorting values.
        first = Index(first_list, name=first_name)
        second = Index(second_list, name=second_name)
        union = first.union(second, sort=sort)

        vals = set(first_list).union(second_list)

        if sort is None and len(first_list) > 0 and len(second_list) > 0:
            expected = Index(sorted(vals), name=expected_name)
            tm.assert_index_equal(union, expected)
        else:
            expected = Index(vals, name=expected_name)
            tm.assert_index_equal(union.sort_values(), expected.sort_values())

    @pytest.mark.parametrize(
        "diff_type, expected",
        [["difference", [1, "B"]], ["symmetric_difference", [1, 2, "B", "C"]]],
    )
    def test_difference_object_type(self, diff_type, expected):
        # [FUNCTION DOC]
        # Parametrized over difference and symmetric_difference with expected labels. Regression
        # test: set operations on object indexes mixing integers and strings give the expected labels.
        # GH 13432
        idx1 = Index([0, 1, "A", "B"])
        idx2 = Index([0, 2, "A", "C"])
        result = getattr(idx1, diff_type)(idx2)
        expected = Index(expected)
        tm.assert_index_equal(result, expected)
