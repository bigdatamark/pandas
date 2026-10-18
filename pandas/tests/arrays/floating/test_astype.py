# [MODULE DOC]
# Tests for converting the nullable floating array with astype to numpy dtypes, other nullable dtypes,
# string and object, and for copy semantics of data and mask. Uses the dtype fixture from the local conftest
# and using_infer_string from the shared conftest. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_astype():
    # [FUNCTION DOC]
    # With missing values, casting to a 64-bit integer or bool raises a value error, while casting to float gives NaN for
    # missing. Without missing values, casting to integer truncates and to bool is true for nonzero.
    # with missing values
    arr = pd.array([0.1, 0.2, None], dtype="Float64")

    with pytest.raises(ValueError, match="cannot convert NA to integer"):
        arr.astype("int64")

    with pytest.raises(ValueError, match="cannot convert float NaN to bool"):
        arr.astype("bool")

    result = arr.astype("float64")
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    # no missing values
    arr = pd.array([0.0, 1.0, 0.5], dtype="Float64")
    result = arr.astype("int64")
    expected = np.array([0, 1, 0], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("bool")
    expected = np.array([False, True, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_to_floating_array():
    # [FUNCTION DOC]
    # Casting to the same dtype (by name or instance) returns an equal array; casting to 32-bit nullable float keeps the missing value.
    # astype to FloatingArray
    arr = pd.array([0.0, 1.0, None], dtype="Float64")

    result = arr.astype("Float64")
    tm.assert_extension_array_equal(result, arr)
    result = arr.astype(pd.Float64Dtype())
    tm.assert_extension_array_equal(result, arr)
    result = arr.astype("Float32")
    expected = pd.array([0.0, 1.0, None], dtype="Float32")
    tm.assert_extension_array_equal(result, expected)


def test_astype_to_boolean_array():
    # [FUNCTION DOC]
    # Casting to nullable boolean (by name or instance) gives false, true and missing.
    # astype to BooleanArray
    arr = pd.array([0.0, 1.0, None], dtype="Float64")

    result = arr.astype("boolean")
    expected = pd.array([False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = arr.astype(pd.BooleanDtype())
    tm.assert_extension_array_equal(result, expected)


def test_astype_to_integer_array():
    # [FUNCTION DOC]
    # Casting to nullable Int64 truncates fractional values and keeps missing.
    # astype to IntegerArray
    arr = pd.array([0.0, 1.5, None], dtype="Float64")

    result = arr.astype("Int64")
    expected = pd.array([0, 1, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_str(using_infer_string):
    # [FUNCTION DOC]
    # Casting to the string type gives a string extension array with NaN as missing if string inference is on, else a numpy array of text with the marker for NA in place of missing.
    a = pd.array([0.1, 0.2, None], dtype="Float64")

    if using_infer_string:
        expected = pd.array(["0.1", "0.2", None], dtype=pd.StringDtype(na_value=np.nan))

        tm.assert_extension_array_equal(a.astype(str), expected)
        tm.assert_extension_array_equal(a.astype("str"), expected)
    else:
        expected = np.array(["0.1", "0.2", "<NA>"], dtype="U32")

        tm.assert_numpy_array_equal(a.astype(str), expected)
        tm.assert_numpy_array_equal(a.astype("str"), expected)


def test_astype_copy():
    # [FUNCTION DOC]
    # Copy true gives an independent array whose data and mask are not shared (mutations do not reach the original);
    # copy false on the same dtype returns the same object sharing both; casting to a different dtype always copies, including the mask,
    # even with copy disabled.
    arr = pd.array([0.1, 0.2, None], dtype="Float64")
    orig = pd.array([0.1, 0.2, None], dtype="Float64")

    # copy=True -> ensure both data and mask are actual copies
    result = arr.astype("Float64", copy=True)
    assert result is not arr
    assert not tm.shares_memory(result, arr)
    result[0] = 10
    tm.assert_extension_array_equal(arr, orig)
    result[0] = pd.NA
    tm.assert_extension_array_equal(arr, orig)

    # copy=False
    result = arr.astype("Float64", copy=False)
    assert result is arr
    assert np.shares_memory(result._data, arr._data)
    assert np.shares_memory(result._mask, arr._mask)
    result[0] = 10
    assert arr[0] == 10
    result[0] = pd.NA
    assert arr[0] is pd.NA

    # astype to different dtype -> always needs a copy -> even with copy=False
    # we need to ensure that also the mask is actually copied
    arr = pd.array([0.1, 0.2, None], dtype="Float64")
    orig = pd.array([0.1, 0.2, None], dtype="Float64")

    result = arr.astype("Float32", copy=False)
    assert not tm.shares_memory(result, arr)
    result[0] = 10
    tm.assert_extension_array_equal(arr, orig)
    result[0] = pd.NA
    tm.assert_extension_array_equal(arr, orig)


def test_astype_object(dtype):
    # [FUNCTION DOC]
    # Casting to object gives python floats and the NA singleton for missing.
    arr = pd.array([1.0, pd.NA], dtype=dtype)

    result = arr.astype(object)
    expected = np.array([1.0, pd.NA], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    # check exact element types
    assert isinstance(result[0], float)
    assert result[1] is pd.NA


def test_Float64_conversion():
    # [FUNCTION DOC]
    # A Series of numeric strings with object dtype cast to the nullable Float64 dtype parses the strings.
    # GH#40729
    testseries = pd.Series(["1", "2", "3", "4"], dtype="object")
    result = testseries.astype(pd.Float64Dtype())

    expected = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=pd.Float64Dtype())

    tm.assert_series_equal(result, expected)
