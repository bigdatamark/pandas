# [MODULE DOC]
# Tests for equality (the equals method and comparison operators) of CategoricalIndex.
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    MultiIndex,
)


class TestEquals:
    # [CLASS DOC]
    # Groups equality tests for CategoricalIndex; no state.
    def test_equals_categorical(self):
        # [FUNCTION DOC]
        # Verifies equals semantics: self and object-cast equality, element-wise comparison operators
        # (equal, ordering) against itself, scalars, plain Index and values; length mismatch raises a
        # value error; differing categories or orderedness in comparison raises a type error;
        # equals respects category set, orderedness (but not category order for unordered), a non-index
        # list, and missing values.
        ci1 = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)
        ci2 = CategoricalIndex(["a", "b"], categories=["a", "b", "c"], ordered=True)

        assert ci1.equals(ci1)
        assert not ci1.equals(ci2)
        assert ci1.equals(ci1.astype(object))
        assert ci1.astype(object).equals(ci1)

        assert (ci1 == ci1).all()
        assert not (ci1 != ci1).all()
        assert not (ci1 > ci1).all()
        assert not (ci1 < ci1).all()
        assert (ci1 <= ci1).all()
        assert (ci1 >= ci1).all()

        assert not (ci1 == 1).all()
        assert (ci1 == Index(["a", "b"])).all()
        assert (ci1 == ci1.values).all()

        # invalid comparisons
        with pytest.raises(ValueError, match="Lengths must match"):
            ci1 == Index(["a", "b", "c"])

        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):
            ci1 == ci2
        with pytest.raises(TypeError, match=msg):
            ci1 == Categorical(ci1.values, ordered=False)
        with pytest.raises(TypeError, match=msg):
            ci1 == Categorical(ci1.values, categories=list("abc"))

        # tests
        # make sure that we are testing for category inclusion properly
        ci = CategoricalIndex(list("aabca"), categories=["c", "a", "b"])
        assert not ci.equals(list("aabca"))
        # Same categories, but different order
        # Unordered
        assert ci.equals(CategoricalIndex(list("aabca")))
        # Ordered
        assert not ci.equals(CategoricalIndex(list("aabca"), ordered=True))
        assert ci.equals(ci.copy())

        ci = CategoricalIndex([*list("aabca"), np.nan], categories=["c", "a", "b"])
        assert not ci.equals(list("aabca"))
        assert not ci.equals(CategoricalIndex(list("aabca")))
        assert ci.equals(ci.copy())

        ci = CategoricalIndex([*list("aabca"), np.nan], categories=["c", "a", "b"])
        assert not ci.equals([*list("aabca"), np.nan])
        assert ci.equals(CategoricalIndex([*list("aabca"), np.nan]))
        assert not ci.equals(CategoricalIndex([*list("aabca"), np.nan], ordered=True))
        assert ci.equals(ci.copy())

    def test_equals_categorical_unordered(self):
        # [FUNCTION DOC]
        # Regression test: unordered CategoricalIndexes with the same labels but permuted categories
        # are equal, while one with a missing value instead is not equal to either.
        # https://github.com/pandas-dev/pandas/issues/16603
        a = CategoricalIndex(["A"], categories=["A", "B"])
        b = CategoricalIndex(["A"], categories=["B", "A"])
        c = CategoricalIndex([None], categories=["B", "A"])
        assert a.equals(b)
        assert not a.equals(c)
        assert not b.equals(c)

    def test_equals_non_category(self):
        # [FUNCTION DOC]
        # Regression test: an index with missing values is not equal to a plain Index that contains a
        # value outside its categories at a position where the index has a missing value.
        # GH#37667 Case where other contains a value not among ci's
        #  categories ("D") and also contains np.nan
        ci = CategoricalIndex(["A", "B", np.nan, np.nan])
        other = Index(["A", "B", "D", np.nan])

        assert not ci.equals(other)

    def test_equals_multiindex(self):
        # [FUNCTION DOC]
        # Verifies a categorical flat-tuple index is not equal to a MultiIndex and the comparison does
        # not raise.
        # dont raise NotImplementedError when calling is_dtype_compat

        mi = MultiIndex.from_arrays([["A", "B", "C", "D"], range(4)])
        ci = mi.to_flat_index().astype("category")

        assert not ci.equals(mi)

    def test_equals_string_dtype(self, any_string_dtype):
        # [FUNCTION DOC]
        # Uses the any_string_dtype fixture. Verifies a CategoricalIndex of strings equals a plain
        # Index of the same strings in any string dtype (names equal).
        # GH#55364
        idx = CategoricalIndex(list("abc"), name="B")
        other = Index(["a", "b", "c"], name="B", dtype=any_string_dtype)
        assert idx.equals(other)
