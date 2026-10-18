# [MODULE DOC]
# Tests for the public function that validates and converts an array-like indexer for use on
# an array (check_array_indexer, imported from the public indexers API): valid inputs, missing
# values, length mismatches, invalid dtypes and pass-through of non-array indexers.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer


@pytest.mark.parametrize(
    "indexer, expected",
    [
        # integer
        ([1, 2], np.array([1, 2], dtype=np.intp)),
        (np.array([1, 2], dtype="int64"), np.array([1, 2], dtype=np.intp)),
        (pd.array([1, 2], dtype="Int32"), np.array([1, 2], dtype=np.intp)),
        (pd.Index([1, 2]), np.array([1, 2], dtype=np.intp)),
        # boolean
        ([True, False, True], np.array([True, False, True], dtype=np.bool_)),
        (np.array([True, False, True]), np.array([True, False, True], dtype=np.bool_)),
        (
            pd.array([True, False, True], dtype="boolean"),
            np.array([True, False, True], dtype=np.bool_),
        ),
        # other
        ([], np.array([], dtype=np.intp)),
    ],
)
def test_valid_input(indexer, expected):
    # [FUNCTION DOC]
    # Parametrized over integer and boolean indexers (lists, numpy arrays, nullable arrays, index
    # objects) and an empty list. The function returns a numpy array of platform integers or booleans
    # with the same content.
    arr = np.array([1, 2, 3])
    result = check_array_indexer(arr, indexer)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "indexer", [[True, False, None], pd.array([True, False, None], dtype="boolean")]
)
def test_boolean_na_returns_indexer(indexer):
    # https://github.com/pandas-dev/pandas/issues/31503
    # [FUNCTION DOC]
    # Issue 31503. Parametrized over a list and a nullable boolean array containing missing values.
    # Missing entries are treated as false in the returned boolean array.
    arr = np.array([1, 2, 3])

    result = check_array_indexer(arr, indexer)
    expected = np.array([True, False, False], dtype=bool)

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "indexer",
    [
        [True, False],
        pd.array([True, False], dtype="boolean"),
        np.array([True, False], dtype=np.bool_),
    ],
)
def test_bool_raise_length(indexer):
    # [FUNCTION DOC]
    # Parametrized over boolean list, nullable array, and numpy array shorter than the target array.
    # Raises an index error saying the boolean index has the wrong length.
    arr = np.array([1, 2, 3])

    msg = "Boolean index has wrong length"
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)


@pytest.mark.parametrize(
    "indexer", [[0, 1, None], pd.array([0, 1, pd.NA], dtype="Int64")]
)
def test_int_raise_missing_values(indexer):
    # [FUNCTION DOC]
    # Integer indexers containing missing values (list with None, nullable array with NA) raise a
    # value error saying an integer indexer cannot contain NA values.
    arr = np.array([1, 2, 3])

    msg = "Cannot index with an integer indexer containing NA values"
    with pytest.raises(ValueError, match=msg):
        check_array_indexer(arr, indexer)


@pytest.mark.parametrize(
    "indexer",
    [
        [0.0, 1.0],
        np.array([1.0, 2.0], dtype="float64"),
        np.array([True, False], dtype=object),
        pd.Index([True, False], dtype=object),
    ],
)
def test_raise_invalid_array_dtypes(indexer):
    # [FUNCTION DOC]
    # Parametrized over float list, float array, object-dtype boolean array and index. Raises an
    # index error saying arrays used as indices must be integer or boolean type.
    arr = np.array([1, 2, 3])

    msg = "arrays used as indices must be of integer or boolean type"
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)


def test_raise_nullable_string_dtype(nullable_string_dtype):
    # [FUNCTION DOC]
    # Uses the nullable string dtype fixture. A string array indexer raises the same index error
    # about integer or boolean type.
    indexer = pd.array(["a", "b"], dtype=nullable_string_dtype)
    arr = np.array([1, 2, 3])

    msg = "arrays used as indices must be of integer or boolean type"
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)


@pytest.mark.parametrize("indexer", [None, Ellipsis, slice(0, 3), (None,)])
def test_pass_through_non_array_likes(indexer):
    # [FUNCTION DOC]
    # Parametrized over None, ellipsis, a slice and a tuple with None. These indexers are returned
    # unchanged (equal to the input).
    arr = np.array([1, 2, 3])

    result = check_array_indexer(arr, indexer)
    assert result == indexer
