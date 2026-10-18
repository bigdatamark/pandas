# [MODULE DOC]
# Tests for algorithm-style operations on the categorical array: factorize, membership tests, differencing, and hashing of a categorical dtype built from a read-only index.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("categories", [["b", "a", "c"], ["a", "b", "c", "d"]])
def test_factorize(categories, ordered):
    # [FUNCTION DOC]
    # Parametrized over two category lists (one a permutation of the data's values, one with an extra unused category) and a shared ordered flag fixture. Factorizes a categorical with a missing value and verifies codes follow order of appearance (with minus one for missing) and the uniques are a categorical of the observed values in order of appearance keeping the same categories and ordered flag.
    cat = pd.Categorical(
        ["b", "b", "a", "c", None], categories=categories, ordered=ordered
    )
    codes, uniques = pd.factorize(cat)
    expected_codes = np.array([0, 0, 1, 2, -1], dtype=np.intp)
    expected_uniques = pd.Categorical(
        ["b", "a", "c"], categories=categories, ordered=ordered
    )

    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)


def test_factorized_sort():
    # [FUNCTION DOC]
    # Verifies that factorizing with sorting enabled gives codes by sorted category order (missing is minus one) and uniques in sorted order, as an unordered categorical.
    cat = pd.Categorical(["b", "b", None, "a"])
    codes, uniques = pd.factorize(cat, sort=True)
    expected_codes = np.array([1, 1, -1, 0], dtype=np.intp)
    expected_uniques = pd.Categorical(["a", "b"])

    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)


def test_factorized_sort_ordered():
    # [FUNCTION DOC]
    # Verifies sorted factorize on an ordered categorical with explicit category order: codes and uniques follow the category order, not alphabetical order, and the ordered flag and categories are preserved.
    cat = pd.Categorical(
        ["b", "b", None, "a"], categories=["c", "b", "a"], ordered=True
    )

    codes, uniques = pd.factorize(cat, sort=True)
    expected_codes = np.array([0, 0, -1, 1], dtype=np.intp)
    expected_uniques = pd.Categorical(
        ["b", "a"], categories=["c", "b", "a"], ordered=True
    )

    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)


def test_isin_cats():
    # [FUNCTION DOC]
    # Verifies membership testing on a categorical with a missing value: a values list containing NaN matches the missing element, and unmatched values give false.
    # GH2003
    cat = pd.Categorical(["a", "b", np.nan])

    result = cat.isin(["a", np.nan])
    expected = np.array([True, False, True], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)

    result = cat.isin(["a", "c"])
    expected = np.array([True, False, False], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)


@pytest.mark.parametrize("value", [[""], [None, ""], [pd.NaT, ""]])
def test_isin_cats_corner_cases(value):
    # [FUNCTION DOC]
    # Parametrized over value lists containing the empty string with None or NaT. Verifies that a categorical holding only the empty string reports membership true, i.e. missing-value markers in the argument do not interfere with matching the empty string.
    # GH36550
    cat = pd.Categorical([""])
    result = cat.isin(value)
    expected = np.array([True], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)


@pytest.mark.parametrize("empty", [[], pd.Series(dtype=object), np.array([])])
def test_isin_empty(empty):
    # [FUNCTION DOC]
    # Parametrized over an empty list, an empty series, and an empty NumPy array. Verifies that membership of a two-element categorical in an empty collection is all false.
    s = pd.Categorical(["a", "b"])
    expected = np.array([False, False], dtype=bool)

    result = s.isin(empty)
    tm.assert_numpy_array_equal(expected, result)


def test_diff():
    # [FUNCTION DOC]
    # Verifies that differencing a categorical-dtype series or a frame built from it raises a type error with a message advising conversion to a suitable dtype.
    ser = pd.Series([1, 2, 3], dtype="category")

    msg = "Convert to a suitable dtype"
    with pytest.raises(TypeError, match=msg):
        ser.diff()

    df = ser.to_frame(name="A")
    with pytest.raises(TypeError, match=msg):
        df.diff()


def test_hash_read_only_categorical():
    # [FUNCTION DOC]
    # Verifies that a categorical dtype built from the values of an object index (a read-only underlying array) hashes consistently: hashing the dtype twice gives the same value. Regression test for a bug about hashing read-only categoricals.
    # GH#58481
    idx = pd.Index(pd.Index(["a", "b", "c"], dtype="object").values)
    cat = pd.CategoricalDtype(idx)
    arr = pd.Series(["a", "b"], dtype=cat).values
    assert hash(arr.dtype) == hash(arr.dtype)
