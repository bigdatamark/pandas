# [MODULE DOC]
# Tests for the public and semi-private API of the Categorical extension array: ordering flags,
# renaming, reordering, adding, removing and setting categories, describe, code immutability and the
# internal code recoding helper.
# Imports the categorical recoding helper from the core categorical array module and uses the shared
# testing utilities. The ordered fixture comes from the surrounding conftest. No import-time side effects.

import re

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    Series,
    StringDtype,
)
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories


class TestCategoricalAPI:
    # [CLASS DOC]
    # Groups tests of the category-manipulation methods of Categorical (ordered flags, rename, reorder,
    # add, set and remove categories). Stateless; each test builds its own arrays.
    def test_ordered_api(self):
        # [FUNCTION DOC]
        # Verifies that the ordered argument of the constructor controls the ordered attribute, and that
        # categories are sorted when not given and kept in the given order when given. Four combinations of
        # explicit categories and ordered flag are checked through index and attribute comparisons.
        # GH 9347
        cat1 = Categorical(list("acb"), ordered=False)
        tm.assert_index_equal(cat1.categories, Index(["a", "b", "c"]))
        assert not cat1.ordered

        cat2 = Categorical(list("acb"), categories=list("bca"), ordered=False)
        tm.assert_index_equal(cat2.categories, Index(["b", "c", "a"]))
        assert not cat2.ordered

        cat3 = Categorical(list("acb"), ordered=True)
        tm.assert_index_equal(cat3.categories, Index(["a", "b", "c"]))
        assert cat3.ordered

        cat4 = Categorical(list("acb"), categories=list("bca"), ordered=True)
        tm.assert_index_equal(cat4.categories, Index(["b", "c", "a"]))
        assert cat4.ordered

    def test_set_ordered(self):
        # [FUNCTION DOC]
        # Checks conversion between ordered and unordered via as_ordered, as_unordered and set_ordered, each
        # returning a new object with the matching flag. Also checks that assigning the ordered attribute
        # directly raises an attribute error because the property has no setter.
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        cat2 = cat.as_unordered()
        assert not cat2.ordered
        cat2 = cat.as_ordered()
        assert cat2.ordered

        assert cat2.set_ordered(True).ordered
        assert not cat2.set_ordered(False).ordered

        # removed in 0.19.0
        msg = "property 'ordered' of 'Categorical' object has no setter"
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = True
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = False

    def test_rename_categories(self):
        # [FUNCTION DOC]
        # Verifies renaming categories from a list returns a new Categorical with the new categories and
        # leaves the original values and categories unchanged. Also verifies that a callable mapping applied
        # to each category is accepted (upper-casing strings).
        cat = Categorical(["a", "b", "c", "a"])

        # inplace=False: the old one must not be changed
        res = cat.rename_categories([1, 2, 3])
        tm.assert_numpy_array_equal(
            res.__array__(), np.array([1, 2, 3, 1], dtype=np.int64)
        )
        tm.assert_index_equal(res.categories, Index([1, 2, 3]))

        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        exp_cat = Index(["a", "b", "c"])
        tm.assert_index_equal(cat.categories, exp_cat)

        # GH18862 (let rename_categories take callables)
        result = cat.rename_categories(lambda x: x.upper())
        expected = Categorical(["A", "B", "C", "A"])
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("new_categories", [[1, 2, 3, 4], [1, 2]])
    def test_rename_categories_wrong_length_raises(self, new_categories):
        # [FUNCTION DOC]
        # Parametrized over a list that is too long and one that is too short. Expects a value error saying
        # the new categories must have the same number of items as the old ones.
        cat = Categorical(["a", "b", "c", "a"])
        msg = (
            "new categories need to have the same number of items as the "
            "old categories!"
        )
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    def test_rename_categories_series(self):
        # [FUNCTION DOC]
        # Checks that a Series, used as a mapping from old to new categories through its index, is accepted
        # by rename_categories and produces the mapped categories.
        # https://github.com/pandas-dev/pandas/issues/17981
        c = Categorical(["a", "b"])
        result = c.rename_categories(Series([0, 1], index=["a", "b"]))
        expected = Categorical([0, 1])
        tm.assert_categorical_equal(result, expected)

    def test_rename_categories_dict(self):
        # [FUNCTION DOC]
        # Checks dictionary renaming: a full mapping; a partial mapping keeps unmapped categories as they are;
        # a mapping with extra keys ignores the extras; a mapping with no matching keys leaves categories
        # unchanged. Compares the resulting categories index only.
        # GH 17336
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 4, "b": 3, "c": 2, "d": 1})
        expected = Index([4, 3, 2, 1])
        tm.assert_index_equal(res.categories, expected)

        # Test for dicts of smaller length
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 1, "c": 3})

        expected = Index([1, "b", 3, "d"])
        tm.assert_index_equal(res.categories, expected)

        # Test for dicts with bigger length
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6})
        expected = Index([1, 2, 3, 4])
        tm.assert_index_equal(res.categories, expected)

        # Test for dicts with no items from old categories
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"f": 1, "g": 3})

        expected = Index(["a", "b", "c", "d"])
        tm.assert_index_equal(res.categories, expected)

    def test_reorder_categories(self):
        # [FUNCTION DOC]
        # Reorders categories of an ordered Categorical to the reverse order; checks the original is unchanged
        # and the result equals a Categorical constructed with that category order.
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        old = cat.copy()
        new = Categorical(
            ["a", "b", "c", "a"], categories=["c", "b", "a"], ordered=True
        )

        res = cat.reorder_categories(["c", "b", "a"])
        # cat must be the same as before
        tm.assert_categorical_equal(cat, old)
        # only res is changed
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize(
        "new_categories",
        [
            ["a"],  # not all "old" included in "new"
            ["a", "b", "d"],  # still not all "old" in "new"
            ["a", "b", "c", "d"],  # all "old" included in "new", but too long
        ],
    )
    def test_reorder_categories_raises(self, new_categories):
        # [FUNCTION DOC]
        # Parametrized over new category lists that are a strict subset, partially different, or a superset
        # of the old categories. Expects a value error because the new categories must contain exactly the
        # same items as the old ones.
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        msg = "items in new_categories are not the same as in old categories"
        with pytest.raises(ValueError, match=msg):
            cat.reorder_categories(new_categories)

    def test_add_categories(self):
        # [FUNCTION DOC]
        # Adds a category given as a scalar and as a list; original unchanged and result has the new category
        # appended at the end. Also checks that Series, array, Index and list inputs of several new
        # categories are all accepted and give equal results.
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        old = cat.copy()
        new = Categorical(
            ["a", "b", "c", "a"], categories=["a", "b", "c", "d"], ordered=True
        )

        res = cat.add_categories("d")
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        res = cat.add_categories(["d"])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        # GH 9927
        cat = Categorical(list("abc"), ordered=True)
        expected = Categorical(list("abc"), categories=list("abcde"), ordered=True)
        # test with Series, np.array, index, list
        res = cat.add_categories(Series(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(np.array(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(Index(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(["d", "e"])
        tm.assert_categorical_equal(res, expected)

    def test_add_categories_existing_raises(self):
        # [FUNCTION DOC]
        # Adding a category that already exists must raise a value error stating that new categories must
        # not include old categories, naming the offending one.
        # new is in old categories
        cat = Categorical(["a", "b", "c", "d"], ordered=True)
        msg = re.escape("new categories must not include old categories: {'d'}")
        with pytest.raises(ValueError, match=msg):
            cat.add_categories(["d"])

    def test_add_categories_losing_dtype_information(self):
        # [FUNCTION DOC]
        # Regression test: adding categories supplied as nullable integer or string dtype Series must keep
        # that dtype in the resulting categories, so the result equals a Categorical built with the extended
        # categories of the same dtype.
        # GH#48812
        cat = Categorical(Series([1, 2], dtype="Int64"))
        ser = Series([4], dtype="Int64")
        result = cat.add_categories(ser)
        expected = Categorical(
            Series([1, 2], dtype="Int64"), categories=Series([1, 2, 4], dtype="Int64")
        )
        tm.assert_categorical_equal(result, expected)

        cat = Categorical(Series(["a", "b", "a"], dtype=StringDtype()))
        ser = Series(["d"], dtype=StringDtype())
        result = cat.add_categories(ser)
        expected = Categorical(
            Series(["a", "b", "a"], dtype=StringDtype()),
            categories=Series(["a", "b", "d"], dtype=StringDtype()),
        )
        tm.assert_categorical_equal(result, expected)

    def test_set_categories(self):
        # [FUNCTION DOC]
        # Broad test of set_categories: reassigning categories in a different order leaves values intact
        # and original untouched; dropping categories turns their values into missing (code minus one);
        # adding categories works; inspects internal integer codes (int8) after recoding; checks min and max
        # respect the new order; checks the ordered argument is applied when given and preserved otherwise.
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        exp_categories = Index(["c", "b", "a"])
        exp_values = np.array(["a", "b", "c", "a"], dtype=np.object_)

        cat = cat.set_categories(["c", "b", "a"])
        res = cat.set_categories(["a", "b", "c"])
        # cat must be the same as before
        tm.assert_index_equal(cat.categories, exp_categories)
        tm.assert_numpy_array_equal(cat.__array__(), exp_values)
        # only res is changed
        exp_categories_back = Index(["a", "b", "c"])
        tm.assert_index_equal(res.categories, exp_categories_back)
        tm.assert_numpy_array_equal(res.__array__(), exp_values)

        # not all "old" included in "new" -> all not included ones are now
        # np.nan
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        res = cat.set_categories(["a"])
        tm.assert_numpy_array_equal(res.codes, np.array([0, -1, -1, 0], dtype=np.int8))

        # still not all "old" in "new"
        res = cat.set_categories(["a", "b", "d"])
        tm.assert_numpy_array_equal(res.codes, np.array([0, 1, -1, 0], dtype=np.int8))
        tm.assert_index_equal(res.categories, Index(["a", "b", "d"]))

        # all "old" included in "new"
        cat = cat.set_categories(["a", "b", "c", "d"])
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_index_equal(cat.categories, exp_categories)

        # internals...
        c = Categorical([1, 2, 3, 4, 1], categories=[1, 2, 3, 4], ordered=True)
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, 2, 3, 0], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([1, 2, 3, 4]))

        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)

        # all "pointers" to '4' must be changed from 3 to 0,...
        c = c.set_categories([4, 3, 2, 1])

        # positions are changed
        tm.assert_numpy_array_equal(c._codes, np.array([3, 2, 1, 0, 3], dtype=np.int8))

        # categories are now in new order
        tm.assert_index_equal(c.categories, Index([4, 3, 2, 1]))

        # output is the same
        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        assert c.min() == 4
        assert c.max() == 1

        # set_categories should set the ordering if specified
        c2 = c.set_categories([4, 3, 2, 1], ordered=False)
        assert not c2.ordered

        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

        # set_categories should pass thru the ordering
        c2 = c.set_ordered(False).set_categories([4, 3, 2, 1])
        assert not c2.ordered

        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

    @pytest.mark.parametrize(
        "values, categories, new_categories",
        [
            # No NaNs, same cats, same order
            (["a", "b", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["a", "b", "a"], ["a", "b"], ["b", "a"]),
            # Same, unsorted
            (["b", "a", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["b", "a", "a"], ["a", "b"], ["b", "a"]),
            # NaNs
            (["a", "b", "c"], ["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "b"], ["b", "a"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            (["b", "a", "c"], ["a", "b"], ["b", "a"]),
            # Introduce NaNs
            (["a", "b", "c"], ["a", "b"], ["a"]),
            (["a", "b", "c"], ["a", "b"], ["b"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            (["b", "a", "c"], ["a", "b"], ["b"]),
            # No overlap
            (["a", "b", "c"], ["a", "b"], ["d", "e"]),
        ],
    )
    def test_set_categories_many(self, values, categories, new_categories, ordered):
        # [FUNCTION DOC]
        # Parametrized over (values, categories, new_categories) covering same, reordered, subset and
        # disjoint categories, and the ordered fixture. Builds expected Categoricals directly (expecting the
        # Pandas4Warning about values not in the dtype categories when applicable) and checks that
        # set_categories with the ordered flag produces an equal Categorical.
        msg = "Constructing a Categorical with a dtype and values containing"

        warn1 = Pandas4Warning if set(values).difference(categories) else None
        with tm.assert_produces_warning(warn1, match=msg):
            c = Categorical(values, categories)

        warn2 = Pandas4Warning if set(values).difference(new_categories) else None
        with tm.assert_produces_warning(warn2, match=msg):
            expected = Categorical(values, new_categories, ordered)

        result = c.set_categories(new_categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_rename_less(self):
        # [FUNCTION DOC]
        # With rename enabled, setting fewer categories than exist renames the first ones and the leftover
        # value becomes missing.
        # GH 24675
        cat = Categorical(["A", "B"])
        result = cat.set_categories(["A"], rename=True)
        expected = Categorical(["A", np.nan])
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_private(self):
        # [FUNCTION DOC]
        # Tests the private categories setter, with and without the fastpath flag: categories are replaced
        # by the new list and the codes are kept, so values change accordingly (values mapped by position).
        cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        cat._set_categories(["a", "c", "d", "e"])
        expected = Categorical(["a", "c", "d"], categories=list("acde"))
        tm.assert_categorical_equal(cat, expected)

        # fastpath
        cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        cat._set_categories(["a", "c", "d", "e"], fastpath=True)
        expected = Categorical(["a", "c", "d"], categories=list("acde"))
        tm.assert_categorical_equal(cat, expected)

    def test_remove_categories(self):
        # [FUNCTION DOC]
        # Removes a category given as scalar or list; original unchanged and values in the removed category
        # become missing, with remaining categories and ordered flag kept.
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        old = cat.copy()
        new = Categorical(["a", "b", np.nan, "a"], categories=["a", "b"], ordered=True)

        res = cat.remove_categories("c")
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        res = cat.remove_categories(["c"])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize("removals", [["c"], ["c", np.nan], "c", ["c", "c"]])
    def test_remove_categories_raises(self, removals):
        # [FUNCTION DOC]
        # Parametrized over removals that include a category not present (also a missing value, a bare
        # scalar, or duplicates). Expects a value error that removals must all be in the old categories.
        cat = Categorical(["a", "b", "a"])
        message = re.escape("removals must all be in old categories: {'c'}")

        with pytest.raises(ValueError, match=message):
            cat.remove_categories(removals)

    def test_remove_unused_categories(self):
        # [FUNCTION DOC]
        # Checks removal of categories with no values: the result drops unused categories while the original
        # is unchanged; handles missing values (codes stay minus one and remaining codes are renumbered);
        # handles unordered categories keeping their order; and checks a large random object array round-trips
        # to the same list of values.
        c = Categorical(["a", "b", "c", "d", "a"], categories=["a", "b", "c", "d", "e"])
        exp_categories_all = Index(["a", "b", "c", "d", "e"])
        exp_categories_dropped = Index(["a", "b", "c", "d"])

        tm.assert_index_equal(c.categories, exp_categories_all)

        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, exp_categories_dropped)
        tm.assert_index_equal(c.categories, exp_categories_all)

        # with NaN values (GH11599)
        c = Categorical(["a", "b", "c", np.nan], categories=["a", "b", "c", "d", "e"])
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, Index(np.array(["a", "b", "c"])))
        exp_codes = np.array([0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(res.codes, exp_codes)
        tm.assert_index_equal(c.categories, exp_categories_all)

        val = ["F", np.nan, "D", "B", "D", "F", np.nan]
        cat = Categorical(values=val, categories=list("ABCDEFG"))
        out = cat.remove_unused_categories()
        tm.assert_index_equal(out.categories, Index(["B", "D", "F"]))
        exp_codes = np.array([2, -1, 1, 0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(out.codes, exp_codes)
        assert out.tolist() == val

        alpha = list("abcdefghijklmnopqrstuvwxyz")
        val = np.random.default_rng(2).choice(alpha[::2], 10000).astype("object")
        val[np.random.default_rng(2).choice(len(val), 100)] = np.nan

        cat = Categorical(values=val, categories=alpha)
        out = cat.remove_unused_categories()
        assert out.tolist() == val.tolist()


class TestCategoricalAPIWithFactor:
    # [CLASS DOC]
    # Tests of describe on Categorical, named for the historical Factor type. Stateless.
    def test_describe(self):
        # [FUNCTION DOC]
        # Checks describe returns a frame indexed by a categorical index named categories with count and
        # frequency columns: for strings, with an unused category (count zero), for integers, and with missing
        # values (reported as an additional row after the categories with counts and frequencies including
        # missing).
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        # string type
        desc = factor.describe()
        assert factor.ordered
        exp_index = CategoricalIndex(
            ["a", "b", "c"], name="categories", ordered=factor.ordered
        )
        expected = DataFrame(
            {"counts": [3, 2, 3], "freqs": [3 / 8.0, 2 / 8.0, 3 / 8.0]}, index=exp_index
        )
        tm.assert_frame_equal(desc, expected)

        # check unused categories
        cat = factor.copy()
        cat = cat.set_categories(["a", "b", "c", "d"])
        desc = cat.describe()

        exp_index = CategoricalIndex(
            list("abcd"), ordered=factor.ordered, name="categories"
        )
        expected = DataFrame(
            {"counts": [3, 2, 3, 0], "freqs": [3 / 8.0, 2 / 8.0, 3 / 8.0, 0]},
            index=exp_index,
        )
        tm.assert_frame_equal(desc, expected)

        # check an integer one
        cat = Categorical([1, 2, 3, 1, 2, 3, 3, 2, 1, 1, 1])
        desc = cat.describe()
        exp_index = CategoricalIndex([1, 2, 3], ordered=cat.ordered, name="categories")
        expected = DataFrame(
            {"counts": [5, 3, 3], "freqs": [5 / 11.0, 3 / 11.0, 3 / 11.0]},
            index=exp_index,
        )
        tm.assert_frame_equal(desc, expected)

        # https://github.com/pandas-dev/pandas/issues/3678
        # describe should work with NaN
        cat = Categorical([np.nan, 1, 2, 2])
        desc = cat.describe()
        expected = DataFrame(
            {"counts": [1, 2, 1], "freqs": [1 / 4.0, 2 / 4.0, 1 / 4.0]},
            index=CategoricalIndex(
                [1, 2, np.nan], categories=[1, 2], name="categories"
            ),
        )
        tm.assert_frame_equal(desc, expected)


class TestPrivateCategoricalAPI:
    # [CLASS DOC]
    # Tests of private behavior: read-only codes and the recode helper. Stateless.
    def test_codes_immutable(self):
        # [FUNCTION DOC]
        # Verifies the codes attribute is read only: assignment raises an attribute error, and writing into
        # the returned array raises a value error about a read-only destination. Setting a value on the
        # Categorical or writing into the private codes array still works and is reflected in the codes.
        # Codes should be read only
        c = Categorical(["a", "b", "c", "a", np.nan])
        exp = np.array([0, 1, 2, 0, -1], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)

        # Assignments to codes should raise
        msg = "property 'codes' of 'Categorical' object has no setter"
        with pytest.raises(AttributeError, match=msg):
            c.codes = np.array([0, 1, 2, 0, 1], dtype="int8")

        # changes in the codes array should raise
        codes = c.codes

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            codes[4] = 1

        # But even after getting the codes, the original array should still be
        # writeable!
        c[4] = "a"
        exp = np.array([0, 1, 2, 0, 0], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)
        c._codes[4] = 2
        exp = np.array([0, 1, 2, 0, 2], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)

    @pytest.mark.parametrize(
        "codes, old, new, expected",
        [
            ([0, 1], ["a", "b"], ["a", "b"], [0, 1]),
            ([0, 1], ["b", "a"], ["b", "a"], [0, 1]),
            ([0, 1], ["a", "b"], ["b", "a"], [1, 0]),
            ([0, 1], ["b", "a"], ["a", "b"], [1, 0]),
            ([0, 1, 0, 1], ["a", "b"], ["a", "b", "c"], [0, 1, 0, 1]),
            ([0, 1, 2, 2], ["a", "b", "c"], ["a", "b"], [0, 1, -1, -1]),
            ([0, 1, -1], ["a", "b", "c"], ["a", "b", "c"], [0, 1, -1]),
            ([0, 1, -1], ["a", "b", "c"], ["b"], [-1, 0, -1]),
            ([0, 1, -1], ["a", "b", "c"], ["d"], [-1, -1, -1]),
            ([0, 1, -1], ["a", "b", "c"], [], [-1, -1, -1]),
            ([-1, -1], [], ["a", "b"], [-1, -1]),
            ([1, 0], ["b", "a"], ["a", "b"], [0, 1]),
        ],
    )
    def test_recode_to_categories(self, codes, old, new, expected):
        # [FUNCTION DOC]
        # Parametrized over (codes, old, new, expected). Calls the recoding helper with copy enabled and checks
        # the output integer codes map each old category to its position in the new categories, using minus
        # one for missing or removed categories. Codes and expected are 8-bit integers.
        codes = np.asanyarray(codes, dtype=np.int8)
        expected = np.asanyarray(expected, dtype=np.int8)
        old = Index(old)
        new = Index(new)
        result = recode_for_categories(codes, old, new, copy=True)
        tm.assert_numpy_array_equal(result, expected)

    def test_recode_to_categories_large(self):
        # [FUNCTION DOC]
        # Recoding a 1000-element code array where the new categories are the old ones reversed should give
        # the reversed positions, as 16-bit integers (verifies dtype widening beyond the small-int range).
        N = 1000
        codes = np.arange(N)
        old = Index(codes)
        expected = np.arange(N - 1, -1, -1, dtype=np.int16)
        new = Index(expected)
        result = recode_for_categories(codes, old, new, copy=True)
        tm.assert_numpy_array_equal(result, expected)
