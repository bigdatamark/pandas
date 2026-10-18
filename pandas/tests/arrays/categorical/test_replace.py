# [MODULE DOC]
# Tests for replacing values in Series holding categorical data (one-to-one, many-to-one, order
# preservation, extension-dtype categories). All functions are module-level. No import-time side effects.

import pytest

import pandas as pd
from pandas import Categorical
import pandas._testing as tm


@pytest.mark.parametrize(
    "to_replace,value,expected",
    [
        # one-to-one
        (4, 1, [1, 2, 3]),
        (3, 1, [1, 2, 1]),
        # many-to-one
        ((5, 6), 2, [1, 2, 3]),
        ((3, 2), 1, [1, 1, 1]),
    ],
)
def test_replace_categorical_series(to_replace, value, expected):
    # [FUNCTION DOC]
    # Parametrized over (to_replace, value, expected). Replacing an existing category value, or several at once,
    # in a categorical Series gives a categorical Series with the same categories (one to three) and the
    # replaced values.
    # GH 31720
    ser = pd.Series([1, 2, 3], dtype="category")
    result = ser.replace(to_replace, value)
    expected = pd.Series(Categorical(expected, categories=[1, 2, 3]))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "to_replace,value",
    [
        # one-to-one
        (3, 5),
        # many-to-one
        ((3, 2), 5),
    ],
)
def test_replace_categorical_series_new_category_raises(to_replace, value):
    # [FUNCTION DOC]
    # Parametrized over a single value and a pair, replacing with a value that is not an existing category. Expects
    # a type error about setting a new category.
    # GH 31720
    ser = pd.Series([1, 2, 3], dtype="category")
    with pytest.raises(
        TypeError, match="Cannot setitem on a Categorical with a new category"
    ):
        ser.replace(to_replace, value)


def test_replace_maintain_ordering():
    # [FUNCTION DOC]
    # Replacing in an ordered categorical Series keeps dtype and category order (checked strictly).
    # GH51016
    dtype = pd.CategoricalDtype([0, 1, 2], ordered=True)
    ser = pd.Series([0, 1, 2], dtype=dtype)
    result = ser.replace(0, 2)
    expected = pd.Series([2, 1, 2], dtype=dtype)
    tm.assert_series_equal(expected, result, check_category_order=True)


def test_replace_categorical_ea_dtype():
    # [FUNCTION DOC]
    # Categorical with string-extension-dtype categories: replacing two values with an existing third gives a
    # Categorical with all original categories and the string dtype.
    # GH49404
    cat = Categorical(pd.array(["a", "b", "c"], dtype="string"))
    result = pd.Series(cat).replace(["a", "b"], ["c", "c"])._values
    expected = Categorical(
        pd.array(["c"] * 3, dtype="string"),
        categories=pd.array(["a", "b", "c"], dtype="string"),
    )
    tm.assert_categorical_equal(result, expected)


def test_replace_categorical_ea_dtype_different_cats_raises():
    # [FUNCTION DOC]
    # Replacing with a value that is not a category (or NA) raises a type error about a new category.
    # GH49404
    cat = Categorical(pd.array(["a", "b"], dtype="string"))
    with pytest.raises(
        TypeError, match="Cannot setitem on a Categorical with a new category"
    ):
        pd.Series(cat).replace(["a", "b"], ["c", pd.NA])
