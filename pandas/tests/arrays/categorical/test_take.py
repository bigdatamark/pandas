# [MODULE DOC]
# Tests for Categorical take: default and explicit allow_fill behavior, bounds errors, empty arrays,
# fill values, and preservation of categories and ordering. Defines a small boolean parametrized fixture
# and uses the ordered fixture from the shared conftest. No import-time side effects.

import numpy as np
import pytest

from pandas import Categorical
import pandas._testing as tm


@pytest.fixture(params=[True, False])
def allow_fill(request):
    # [FUNCTION DOC]
    # HELPER: pytest fixture, parametrized over true and false, supplying the allow_fill argument for take tests.
    """Boolean 'allow_fill' parameter for Categorical.take"""
    return request.param


class TestTake:
    # [CLASS DOC]
    # Groups take tests for Categorical. Stateless.
    # https://github.com/pandas-dev/pandas/issues/20664

    def test_take_default_allow_fill(self):
        # [FUNCTION DOC]
        # Taking with a negative index and default arguments emits no warning and returns an object equal to the source (default is fill mode, so negative means missing, which here equals last by values).
        cat = Categorical(["a", "b"])
        with tm.assert_produces_warning(None):
            result = cat.take([0, -1])

        assert result.equals(cat)

    def test_take_positive_no_warning(self):
        # [FUNCTION DOC]
        # Taking only non-negative indices emits no warning.
        cat = Categorical(["a", "b"])
        with tm.assert_produces_warning(None):
            cat.take([0, 0])

    def test_take_bounds(self, allow_fill):
        # [FUNCTION DOC]
        # Parametrized over allow_fill. Out-of-range indices raise an index error; message says indices are out-of-bounds
        # with fill and a numpy-style out-of-bounds message without.
        # https://github.com/pandas-dev/pandas/issues/20664
        cat = Categorical(["a", "b", "a"])
        if allow_fill:
            msg = "indices are out-of-bounds"
        else:
            msg = "index 4 is out of bounds for( axis 0 with)? size 3"
        with pytest.raises(IndexError, match=msg):
            cat.take([4, 5], allow_fill=allow_fill)

    def test_take_empty(self, allow_fill):
        # [FUNCTION DOC]
        # Parametrized over allow_fill. Taking from an empty Categorical raises an index error (out-of-bounds in fill mode,
        # a message about non-empty take from empty axes otherwise).
        # https://github.com/pandas-dev/pandas/issues/20664
        cat = Categorical([], categories=["a", "b"])
        if allow_fill:
            msg = "indices are out-of-bounds"
        else:
            msg = "cannot do a non-empty take from an empty axes"
        with pytest.raises(IndexError, match=msg):
            cat.take([0], allow_fill=allow_fill)

    def test_positional_take(self, ordered):
        # [FUNCTION DOC]
        # Uses the ordered fixture. Positional take without fill keeps categories and ordered flag.
        cat = Categorical(["a", "a", "b", "b"], categories=["b", "a"], ordered=ordered)
        result = cat.take([0, 1, 2], allow_fill=False)
        expected = Categorical(
            ["a", "a", "b"], categories=cat.categories, ordered=ordered
        )
        tm.assert_categorical_equal(result, expected)

    def test_positional_take_unobserved(self, ordered):
        # [FUNCTION DOC]
        # Uses the ordered fixture. Positional take keeps categories that never occur in the values.
        cat = Categorical(["a", "b"], categories=["a", "b", "c"], ordered=ordered)
        result = cat.take([1, 0], allow_fill=False)
        expected = Categorical(["b", "a"], categories=cat.categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_take_allow_fill(self):
        # [FUNCTION DOC]
        # With fill allowed, negative indices become missing values and categories are kept.
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "a", "b"])
        result = cat.take([0, -1, -1], allow_fill=True)
        expected = Categorical(["a", np.nan, np.nan], categories=["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_take_fill_with_negative_one(self):
        # [FUNCTION DOC]
        # When minus one is itself a category, fill value minus one fills with that category rather than missing.
        # -1 was a category
        cat = Categorical([-1, 0, 1])
        result = cat.take([0, -1, 1], allow_fill=True, fill_value=-1)
        expected = Categorical([-1, -1, 0], categories=[-1, 0, 1])
        tm.assert_categorical_equal(result, expected)

    def test_take_fill_value(self):
        # [FUNCTION DOC]
        # A fill value that is an existing category fills the negative-index positions.
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "b", "c"])
        result = cat.take([0, 1, -1], fill_value="a", allow_fill=True)
        expected = Categorical(["a", "b", "a"], categories=["a", "b", "c"])
        tm.assert_categorical_equal(result, expected)

    def test_take_fill_value_new_raises(self):
        # [FUNCTION DOC]
        # A fill value that is not an existing category raises a type error about a new category.
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "b", "c"])
        xpr = r"Cannot setitem on a Categorical with a new category \(d\)"
        with pytest.raises(TypeError, match=xpr):
            cat.take([0, 1, -1], fill_value="d", allow_fill=True)
