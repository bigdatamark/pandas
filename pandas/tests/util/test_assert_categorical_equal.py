# [MODULE DOC]
# Tests for the testing-utility function that asserts two categoricals are equal: identical values, differing category order (with and without the order check), differing categories, differing codes, differing orderedness, and a custom object label in the message.

import pytest

from pandas import Categorical
import pandas._testing as tm


@pytest.mark.parametrize("c", [None, [1, 2, 3, 4, 5]])
def test_categorical_equal(c):
    # [FUNCTION DOC]
    # Parametrized over default and explicit categories. A categorical compared to itself passes.
    c = Categorical([1, 2, 3, 4], categories=c)
    tm.assert_categorical_equal(c, c)


@pytest.mark.parametrize("check_category_order", [True, False])
def test_categorical_equal_order_mismatch(check_category_order):
    # [FUNCTION DOC]
    # Parametrized over the category-order check flag. Two categoricals with identical values but reversed category order: with the check on, an assertion error with a message listing differing categories (100 percent) is raised; with the check off, they compare equal.
    c1 = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 4], categories=[4, 3, 2, 1])
    kwargs = {"check_category_order": check_category_order}

    if check_category_order:
        msg = """Categorical\\.categories are different

Categorical\\.categories values are different \\(100\\.0 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[4, 3, 2, 1\\], dtype='int64'\\)"""
        with pytest.raises(AssertionError, match=msg):
            tm.assert_categorical_equal(c1, c2, **kwargs)
    else:
        tm.assert_categorical_equal(c1, c2, **kwargs)


def test_categorical_equal_categories_mismatch():
    # [FUNCTION DOC]
    # Categoricals with a differing last category raise an assertion error whose message shows the categories differ in 25 percent of values with the left and right index listings.
    msg = """Categorical\\.categories are different

Categorical\\.categories values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 5\\], dtype='int64'\\)"""

    c1 = Categorical([1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 5])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)


def test_categorical_equal_codes_mismatch():
    # [FUNCTION DOC]
    # Categoricals with the same categories but two swapped values raise an assertion error showing the codes differ in 50 percent of values.
    categories = [1, 2, 3, 4]
    msg = """Categorical\\.codes are different

Categorical\\.codes values are different \\(50\\.0 %\\)
\\[left\\]:  \\[0, 1, 3, 2\\]
\\[right\\]: \\[0, 1, 2, 3\\]"""

    c1 = Categorical([1, 2, 4, 3], categories=categories)
    c2 = Categorical([1, 2, 3, 4], categories=categories)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)


def test_categorical_equal_ordered_mismatch():
    # [FUNCTION DOC]
    # Categoricals identical except for orderedness raise an assertion error saying the ordered attribute is different (left false, right true).
    data = [1, 2, 3, 4]
    msg = """Categorical are different

Attribute "ordered" are different
\\[left\\]:  False
\\[right\\]: True"""

    c1 = Categorical(data, ordered=False)
    c2 = Categorical(data, ordered=True)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)


@pytest.mark.parametrize("obj", ["index", "foo", "pandas"])
def test_categorical_equal_object_override(obj):
    # [FUNCTION DOC]
    # Parametrized over object label strings. The assertion error message for an ordered mismatch begins with the given label followed by "are different", when the label is passed through the obj argument.
    data = [1, 2, 3, 4]
    msg = f"""{obj} are different

Attribute "ordered" are different
\\[left\\]:  False
\\[right\\]: True"""

    c1 = Categorical(data, ordered=False)
    c2 = Categorical(data, ordered=True)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2, obj=obj)
