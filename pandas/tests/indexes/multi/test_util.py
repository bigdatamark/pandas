# [MODULE DOC]
# Tests for the cartesian product helper in the multi-index module, which produces
# repeated arrays for each input factor.
import numpy as np
import pytest

from pandas import (
    Index,
    date_range,
)
import pandas._testing as tm
from pandas.core.indexes.multi import cartesian_product


class TestCartesianProduct:
    # [CLASS DOC]
    # Groups tests of the cartesian product function.
    def test_simple(self):
        # [FUNCTION DOC]
        # Product of a string list and an integer list gives two arrays with the first
        # repeated per element and the second tiled.
        x, y = list("ABC"), [1, 22]
        result1, result2 = cartesian_product([x, y])
        expected1 = np.array(["A", "A", "B", "B", "C", "C"])
        expected2 = np.array([1, 22, 1, 22, 1, 22])
        tm.assert_numpy_array_equal(result1, expected1)
        tm.assert_numpy_array_equal(result2, expected2)

    def test_datetimeindex(self):
        # [FUNCTION DOC]
        # The order of products involving a datetime index is consistent (GitHub issue
        # #6439).
        # regression test for GitHub issue #6439
        # make sure that the ordering on datetimeindex is consistent
        x = date_range("2000-01-01", periods=2)
        result1, result2 = (Index(y).day for y in cartesian_product([x, x]))
        expected1 = Index([1, 1, 2, 2], dtype=np.int32)
        expected2 = Index([1, 2, 1, 2], dtype=np.int32)
        tm.assert_index_equal(result1, expected1)
        tm.assert_index_equal(result2, expected2)

    @pytest.mark.parametrize("x, y", [[[], []], [[0, 1], []], [[], ["a", "b", "c"]]])
    def test_empty(self, x, y):
        # [FUNCTION DOC]
        # Parametrized over empty factor combinations. Results are empty arrays carrying
        # the
        # dtype of the corresponding input.
        # product of empty factors
        expected1 = np.array([], dtype=np.asarray(x).dtype)
        expected2 = np.array([], dtype=np.asarray(y).dtype)
        result1, result2 = cartesian_product([x, y])
        tm.assert_numpy_array_equal(result1, expected1)
        tm.assert_numpy_array_equal(result2, expected2)

    def test_empty_input(self):
        # [FUNCTION DOC]
        # An empty list of factors gives an empty list.
        # empty product (empty input):
        result = cartesian_product([])
        expected = []
        assert result == expected

    @pytest.mark.parametrize(
        "X", [1, [1], [1, 2], [[1], 2], "a", ["a"], ["a", "b"], [["a"], "b"]]
    )
    def test_invalid_input(self, X):
        # [FUNCTION DOC]
        # Parametrized over scalars, strings and mixed lists. Inputs that are not
        # list-likes of
        # list-likes raise a type error.
        msg = "Input must be a list-like of list-likes"

        with pytest.raises(TypeError, match=msg):
            cartesian_product(X=X)

    def test_exceed_product_space(self):
        # [FUNCTION DOC]
        # A product whose total size exceeds the allocatable space raises a value error
        # saying
        # the product space is too large (GH31355).
        # GH31355: raise useful error when produce space is too large
        msg = "Product space too large to allocate arrays!"

        dims = [np.arange(0, 22, dtype=np.int16) for i in range(12)] + [
            (np.arange(15128, dtype=np.int16)),
        ]
        with pytest.raises(ValueError, match=msg):
            cartesian_product(X=dims)
