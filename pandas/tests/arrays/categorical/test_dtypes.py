# [MODULE DOC]
# Tests for dtype-related behavior of Categorical: comparing category sets up to permutation, the
# private dtype-setting method, the integer width of the codes, the types yielded when iterating, and
# interval-index categories. Uses the ordered fixture from the shared conftest. No import-time side
# effects.

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestCategoricalDtypes:
    # [CLASS DOC]
    # Groups the dtype tests of Categorical. Stateless.
    def test_categories_match_up_to_permutation(self):
        # [FUNCTION DOC]
        # Exercises the private method that checks whether another object has the same categories (in any order)
        # and the same ordered flag. Compares a Categorical with itself, reordered categories, ordered versus
        # unordered, plain index, object conversion, categorical index and Series of categorical data, expecting
        # true only where categories and ordered flag agree.
        # test dtype comparisons between cats

        c1 = Categorical(list("aabca"), categories=list("abc"), ordered=False)
        c2 = Categorical(list("aabca"), categories=list("cab"), ordered=False)
        c3 = Categorical(list("aabca"), categories=list("cab"), ordered=True)
        assert c1._categories_match_up_to_permutation(c1)
        assert c2._categories_match_up_to_permutation(c2)
        assert c3._categories_match_up_to_permutation(c3)
        assert c1._categories_match_up_to_permutation(c2)
        assert not c1._categories_match_up_to_permutation(c3)
        assert not c1._categories_match_up_to_permutation(Index(list("aabca")))
        assert not c1._categories_match_up_to_permutation(c1.astype(object))
        assert c1._categories_match_up_to_permutation(CategoricalIndex(c1))
        assert c1._categories_match_up_to_permutation(
            CategoricalIndex(c1, categories=list("cab"))
        )
        assert not c1._categories_match_up_to_permutation(
            CategoricalIndex(c1, ordered=True)
        )

        # GH 16659
        s1 = Series(c1)
        s2 = Series(c2)
        s3 = Series(c3)
        assert c1._categories_match_up_to_permutation(s1)
        assert c2._categories_match_up_to_permutation(s2)
        assert c3._categories_match_up_to_permutation(s3)
        assert c1._categories_match_up_to_permutation(s2)
        assert not c1._categories_match_up_to_permutation(s3)
        assert not c1._categories_match_up_to_permutation(s1.astype(object))

    def test_set_dtype_same(self):
        # [FUNCTION DOC]
        # Private dtype setter with an identical dtype and copy enabled returns an equal Categorical.
        c = Categorical(["a", "b", "c"])
        result = c._set_dtype(CategoricalDtype(["a", "b", "c"]), copy=True)
        tm.assert_categorical_equal(result, c)

    def test_set_dtype_new_categories(self):
        # [FUNCTION DOC]
        # Setting a dtype with an extra category keeps the codes and extends the dtype categories.
        c = Categorical(["a", "b", "c"])
        result = c._set_dtype(CategoricalDtype(list("abcd")), copy=True)
        tm.assert_numpy_array_equal(result.codes, c.codes)
        tm.assert_index_equal(result.dtype.categories, Index(list("abcd")))

    @pytest.mark.parametrize(
        "values, categories, new_categories, warn",
        [
            # No NaNs, same cats, same order
            (["a", "b", "a"], ["a", "b"], ["a", "b"], None),
            # No NaNs, same cats, different order
            (["a", "b", "a"], ["a", "b"], ["b", "a"], None),
            # Same, unsorted
            (["b", "a", "a"], ["a", "b"], ["a", "b"], None),
            # No NaNs, same cats, different order
            (["b", "a", "a"], ["a", "b"], ["b", "a"], None),
            # NaNs
            (["a", "b", "c"], ["a", "b"], ["a", "b"], None),
            (["a", "b", "c"], ["a", "b"], ["b", "a"], None),
            (["b", "a", "c"], ["a", "b"], ["a", "b"], None),
            (["b", "a", "c"], ["a", "b"], ["b", "a"], None),
            # Introduce NaNs
            (["a", "b", "c"], ["a", "b"], ["a"], Pandas4Warning),
            (["a", "b", "c"], ["a", "b"], ["b"], Pandas4Warning),
            (["b", "a", "c"], ["a", "b"], ["a"], Pandas4Warning),
            (["b", "a", "c"], ["a", "b"], ["b"], Pandas4Warning),
            # No overlap
            (["a", "b", "c"], ["a", "b"], ["d", "e"], Pandas4Warning),
        ],
    )
    def test_set_dtype_many(self, values, categories, new_categories, warn, ordered):
        # [FUNCTION DOC]
        # Parametrized over (values, categories, new_categories, warn) and the ordered fixture. Sets the dtype of a
        # Categorical to the dtype of a directly constructed one and expects equality.
        # KNOWN ISSUE: the warn parameter is accepted but never used; the expected warning is recomputed inside the test from the set difference of values and categories.
        msg = "Constructing a Categorical with a dtype and values containing"
        warn1 = Pandas4Warning if set(values).difference(categories) else None
        with tm.assert_produces_warning(warn1, match=msg):
            c = Categorical(values, categories)
        warn2 = Pandas4Warning if set(values).difference(new_categories) else None
        with tm.assert_produces_warning(warn2, match=msg):
            expected = Categorical(values, new_categories, ordered)
        result = c._set_dtype(expected.dtype, copy=True)
        tm.assert_categorical_equal(result, expected)

    def test_set_dtype_no_overlap(self):
        # [FUNCTION DOC]
        # Setting a dtype whose categories share nothing with the values yields all-missing values with the new categories.
        msg = "Constructing a Categorical with a dtype and values containing"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            c = Categorical(["a", "b", "c"], ["d", "e"])
        result = c._set_dtype(CategoricalDtype(["a", "b"]), copy=True)
        expected = Categorical([None, None, None], categories=["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_codes_dtypes(self):
        # [FUNCTION DOC]
        # The integer type of the codes widens with the number of categories (8, 16, 32 bits); adding many categories
        # widens it and removing them narrows it again.
        # GH 8453
        result = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == "int8"

        result = Categorical([f"foo{i:05d}" for i in range(400)])
        assert result.codes.dtype == "int16"

        result = Categorical([f"foo{i:05d}" for i in range(40000)])
        assert result.codes.dtype == "int32"

        # adding cats
        result = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == "int8"
        result = result.add_categories([f"foo{i:05d}" for i in range(400)])
        assert result.codes.dtype == "int16"

        # removing cats
        result = result.remove_categories([f"foo{i:05d}" for i in range(300)])
        assert result.codes.dtype == "int8"

    def test_iter_python_types(self):
        # [FUNCTION DOC]
        # Iterating a Categorical of integers and converting it to a list yield python integers.
        # GH-19909
        cat = Categorical([1, 2])
        assert isinstance(next(iter(cat)), int)
        assert isinstance(cat.tolist()[0], int)

    def test_iter_python_types_datetime(self):
        # [FUNCTION DOC]
        # Iterating a Categorical of timestamps and converting it to a list yield Timestamp objects.
        cat = Categorical([Timestamp("2017-01-01"), Timestamp("2017-01-02")])
        assert isinstance(next(iter(cat)), Timestamp)
        assert isinstance(cat.tolist()[0], Timestamp)

    def test_interval_index_category(self):
        # [FUNCTION DOC]
        # A categorical index built from an unsigned-integer interval index keeps its interval dtype as categories.
        # GH 38316
        index = IntervalIndex.from_breaks(np.arange(3, dtype="uint64"))

        result = CategoricalIndex(index).dtype.categories
        expected = IntervalIndex.from_arrays(
            [0, 1], [1, 2], dtype="interval[uint64, right]"
        )
        tm.assert_index_equal(result, expected)
