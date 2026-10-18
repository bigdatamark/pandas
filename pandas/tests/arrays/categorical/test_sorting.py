# [MODULE DOC]
# Tests for sorting Categorical arrays: argsort (method and numpy function) and sort_values with the
# ascending, inplace and na_position options. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Categorical,
    Index,
)
import pandas._testing as tm


class TestCategoricalSort:
    # [CLASS DOC]
    # Groups sorting tests for Categorical. Stateless.
    def test_argsort(self):
        # [FUNCTION DOC]
        # Argsort of an ordered integer Categorical gives the positions that sort by category order, ascending or reversed (index dtype not checked).
        c = Categorical([5, 3, 1, 4, 2], ordered=True)

        expected = np.array([2, 4, 1, 3, 0])
        tm.assert_numpy_array_equal(
            c.argsort(ascending=True), expected, check_dtype=False
        )

        expected = expected[::-1]
        tm.assert_numpy_array_equal(
            c.argsort(ascending=False), expected, check_dtype=False
        )

    def test_numpy_argsort(self):
        # [FUNCTION DOC]
        # The numpy argsort function works on a Categorical, accepts a kind argument, and raises a value error for the
        # unsupported axis and order parameters.
        c = Categorical([5, 3, 1, 4, 2], ordered=True)

        expected = np.array([2, 4, 1, 3, 0])
        tm.assert_numpy_array_equal(np.argsort(c), expected, check_dtype=False)

        tm.assert_numpy_array_equal(
            np.argsort(c, kind="mergesort"), expected, check_dtype=False
        )

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(c, axis=0)

        msg = "the 'order' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(c, order="C")

    def test_sort_values(self):
        # [FUNCTION DOC]
        # Unordered Categoricals are sortable. Ordered ones sort ascending and descending by category order keeping
        # categories; in-place sorting reuses the same codes array and sorts it; reversed with duplicates works.
        # KNOWN ISSUE: after the in-place sort the categories assertion uses the earlier result object instead of the in-place sorted array, so it does not verify the in-place result.
        # unordered cats are sortable
        cat = Categorical(["a", "b", "b", "a"], ordered=False)
        cat.sort_values()

        cat = Categorical(["a", "c", "b", "d"], ordered=True)

        # sort_values
        res = cat.sort_values()
        exp = np.array(["a", "b", "c", "d"], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        cat = Categorical(
            ["a", "c", "b", "d"], categories=["a", "b", "c", "d"], ordered=True
        )
        res = cat.sort_values()
        exp = np.array(["a", "b", "c", "d"], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        res = cat.sort_values(ascending=False)
        exp = np.array(["d", "c", "b", "a"], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        # sort (inplace order)
        cat1 = cat.copy()
        orig_codes = cat1._codes
        cat1.sort_values(inplace=True)
        assert cat1._codes is orig_codes
        exp = np.array(["a", "b", "c", "d"], dtype=object)
        tm.assert_numpy_array_equal(cat1.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        # reverse
        cat = Categorical(["a", "c", "c", "b", "d"], ordered=True)
        res = cat.sort_values(ascending=False)
        exp_val = np.array(["d", "c", "c", "b", "a"], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)

    def test_sort_values_na_position(self):
        # [FUNCTION DOC]
        # Numeric and string ordered Categoricals with missing values: default puts missing last; first or last
        # placement and ascending or descending produce the expected order, and categories are unchanged.
        # see gh-12882
        cat = Categorical([5, 2, np.nan, 2, np.nan], ordered=True)
        exp_categories = Index([2, 5])

        exp = np.array([2.0, 2.0, 5.0, np.nan, np.nan])
        res = cat.sort_values()  # default arguments
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([np.nan, np.nan, 2.0, 2.0, 5.0])
        res = cat.sort_values(ascending=True, na_position="first")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([np.nan, np.nan, 5.0, 2.0, 2.0])
        res = cat.sort_values(ascending=False, na_position="first")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([2.0, 2.0, 5.0, np.nan, np.nan])
        res = cat.sort_values(ascending=True, na_position="last")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([5.0, 2.0, 2.0, np.nan, np.nan])
        res = cat.sort_values(ascending=False, na_position="last")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        cat = Categorical(["a", "c", "b", "d", np.nan], ordered=True)
        res = cat.sort_values(ascending=False, na_position="last")
        exp_val = np.array(["d", "c", "b", "a", np.nan], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)

        cat = Categorical(["a", "c", "b", "d", np.nan], ordered=True)
        res = cat.sort_values(ascending=False, na_position="first")
        exp_val = np.array([np.nan, "d", "c", "b", "a"], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)
