# [MODULE DOC]
# Tests of sanitize_array for building numpy arrays from one-dimensional input,
# preserving missing values, and for object conversion of datetime-like arrays.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.construction import sanitize_array


@pytest.mark.parametrize(
    "values, dtype, expected",
    [
        ([1, 2, 3], None, np.array([1, 2, 3], dtype=np.int64)),
        (np.array([1, 2, 3]), None, np.array([1, 2, 3])),
        (["1", "2", None], None, np.array(["1", "2", None])),
        (["1", "2", None], np.dtype("str"), np.array(["1", "2", None])),
        ([1, 2, None], np.dtype("str"), np.array(["1", "2", None])),
    ],
)
def test_construct_1d_ndarray_preserving_na(
    values, dtype, expected, using_infer_string
):
    # [FUNCTION DOC]
    # Parametrized over (values, dtype, expected) cases: integer list, integer array,
    # strings with None (without and with a string dtype) and integers with None
    # converted to string. Fixture using_infer_string: when strings are inferred and
    # the expected array is object with no dtype requested, the result must equal a
    # string-dtype extension array; otherwise the result must equal the expected
    # numpy array.
    result = sanitize_array(values, index=None, dtype=dtype)
    if using_infer_string and expected.dtype == object and dtype is None:
        tm.assert_extension_array_equal(result, pd.array(expected, dtype="str"))
    else:
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]"])
def test_construct_1d_ndarray_preserving_na_datetimelike(dtype):
    # [FUNCTION DOC]
    # Parametrized over m8[ns] and M8[ns]. Sanitizing a datetime-like array with
    # requested object dtype yields an object array whose elements are the scalar
    # type of the original array's elements (numpy scalars), compared to the expected
    # object array.
    arr = np.arange(5, dtype=np.int64).view(dtype)
    expected = np.array(list(arr), dtype=object)
    assert all(isinstance(x, type(arr[0])) for x in expected)

    result = sanitize_array(arr, index=None, dtype=np.dtype(object))
    tm.assert_numpy_array_equal(result, expected)
