# [MODULE DOC]
# Tests for appending to a CategoricalIndex (same categories, empty, mismatched categories,
# plain object Index, and Index-level concatenation).
import pytest

from pandas import (
    CategoricalIndex,
    Index,
)
import pandas._testing as tm


class TestAppend:
    # [CLASS DOC]
    # Groups append tests for CategoricalIndex; the class provides a local fixture for the index.
    @pytest.fixture
    def ci(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an unordered CategoricalIndex of six labels (a, a, b, b, c, a) with
        # categories ordered c, a, b. Used by the append tests.
        categories = list("cab")
        return CategoricalIndex(list("aabbca"), categories=categories, ordered=False)

    def test_append(self, ci):
        # [FUNCTION DOC]
        # Verifies appending two slices, and a list of slices, reconstructs the original index exactly
        # (categorical dtype kept).
        # append cats with the same categories
        result = ci[:3].append(ci[3:])
        tm.assert_index_equal(result, ci, exact=True)

        foos = [ci[:1], ci[1:3], ci[3:]]
        result = foos[0].append(foos[1:])
        tm.assert_index_equal(result, ci, exact=True)

    def test_append_empty(self, ci):
        # [FUNCTION DOC]
        # Verifies appending an empty list returns an index exactly equal to the original.
        # empty
        result = ci.append([])
        tm.assert_index_equal(result, ci, exact=True)

    def test_append_mismatched_categories(self, ci):
        # [FUNCTION DOC]
        # Verifies appending raw Categorical arrays whose categories differ or are reordered raises a
        # type error saying all inputs must be Index objects.
        # appending with different categories or reordered is not ok
        msg = "all inputs must be Index"
        with pytest.raises(TypeError, match=msg):
            ci.append(ci.values.set_categories(list("abcd")))
        with pytest.raises(TypeError, match=msg):
            ci.append(ci.values.reorder_categories(list("abc")))

    def test_append_category_objects(self, ci):
        # [FUNCTION DOC]
        # Verifies appending a plain object Index of values present in the categories yields a plain
        # object Index (not categorical) of the concatenated values.
        # GH#41626 pre-3.0 this used to cast the object-dtype index to
        #  ci.dtype
        # with objects
        result = ci.append(Index(["c", "a"]))
        expected = Index(list("aabbcaca"))
        tm.assert_index_equal(result, expected, exact=True)

    def test_append_non_categories(self, ci):
        # [FUNCTION DOC]
        # Verifies appending a plain Index containing a value not among the categories gives an
        # object Index of all values.
        # invalid objects -> cast to object via concat_compat
        result = ci.append(Index(["a", "d"]))
        expected = Index(["a", "a", "b", "b", "c", "a", "a", "d"])
        tm.assert_index_equal(result, expected, exact=True)

    def test_append_object(self, ci):
        # [FUNCTION DOC]
        # Verifies that appending a CategoricalIndex to a plain Index coerces the result to object.
        # GH#14298 - if base object is not categorical -> coerce to object
        result = Index(["c", "a"]).append(ci)
        expected = Index(list("caaabbca"))
        tm.assert_index_equal(result, expected, exact=True)

    def test_append_to_another(self):
        # [FUNCTION DOC]
        # Verifies concatenating a plain string Index with a CategoricalIndex of new labels gives a plain
        # Index of all labels in order.
        # hits Index._concat
        fst = Index(["a", "b"])
        snd = CategoricalIndex(["d", "e"])
        result = fst.append(snd)
        expected = Index(["a", "b", "d", "e"])
        tm.assert_index_equal(result, expected)
