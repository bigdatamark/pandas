# [MODULE DOC]
# Tests for NumPy universal functions applied to the nullable boolean array and series, plus value counting and differencing.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "ufunc", [np.add, np.logical_or, np.logical_and, np.logical_xor]
)
def test_ufuncs_binary(ufunc):
    # [FUNCTION DOC]
    # Parametrized over the binary ufuncs add, logical or, logical and, and logical xor. Verifies each ufunc applied to boolean arrays, to a series with an array, to an array with a NumPy bool array in either order, and to an array with a scalar true in either order, returns a boolean extension array (or series) equal to the ufunc on the raw data with the mask positions set to missing.
    # Also verifies that a ufunc with an unsupported operand (a string) raises a type error saying all operand types returned not-implemented from the array-ufunc protocol.
    # two BooleanArrays
    a = pd.array([True, False, None], dtype="boolean")
    result = ufunc(a, a)
    expected = pd.array(ufunc(a._data, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s, a)
    expected = pd.Series(ufunc(a._data, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)

    # Boolean with numpy array
    arr = np.array([True, True, False])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a._data, arr), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    # BooleanArray with scalar
    result = ufunc(a, True)
    expected = pd.array(ufunc(a._data, True), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(True, a)
    expected = pd.array(ufunc(True, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    # not handled types
    msg = r"operand type\(s\) all returned NotImplemented from __array_ufunc__"
    with pytest.raises(TypeError, match=msg):
        ufunc(a, "test")


@pytest.mark.parametrize("ufunc", [np.logical_not])
def test_ufuncs_unary(ufunc):
    # [FUNCTION DOC]
    # Parametrized over logical not. Verifies the ufunc on a boolean array and on a series returns boolean-dtype output equal to applying it to the raw data with masked positions set to missing.
    a = pd.array([True, False, None], dtype="boolean")
    result = ufunc(a)
    expected = pd.array(ufunc(a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    ser = pd.Series(a)
    result = ufunc(ser)
    expected = pd.Series(ufunc(a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)


def test_ufunc_numeric():
    # [FUNCTION DOC]
    # Verifies that a numeric ufunc (square root) on a boolean array yields a Float32 nullable array with 1, 0, and missing, noting that NumPy gives half-precision floats for bool input and the library upcasts to single precision since no half-precision nullable dtype exists.
    # np.sqrt on np.bool_ returns float16, which we upcast to Float32
    #  bc we do not have Float16
    arr = pd.array([True, False, None], dtype="boolean")

    res = np.sqrt(arr)

    expected = pd.array([1, 0, None], dtype="Float32")
    tm.assert_extension_array_equal(res, expected)


@pytest.mark.parametrize("values", [[True, False], [True, None]])
def test_ufunc_reduce_raises(values):
    # [FUNCTION DOC]
    # Parametrized over a list with no missing values and a list ending in a missing value.
    # KNOWN ISSUE: the name says the reduction raises, but the test expects no exception: it performs an add reduction through the ufunc and compares the result to the missing singleton when the last element is missing, otherwise to the sum of the raw data (approximate comparison).
    # Verifies the reduction result accordingly.
    arr = pd.array(values, dtype="boolean")

    res = np.add.reduce(arr)
    if arr[-1] is pd.NA:
        expected = pd.NA
    else:
        expected = arr._data.sum()
    tm.assert_almost_equal(res, expected)


def test_value_counts_na():
    # [FUNCTION DOC]
    # Verifies that counting values of a three-element boolean array (true, false, NA) returns a series with the count name, Int64 dtype, and a boolean-dtype index: all three values counted once when missing is kept, and only the two non-missing values when missing is dropped.
    arr = pd.array([True, False, pd.NA], dtype="boolean")
    result = arr.value_counts(dropna=False)
    expected = pd.Series([1, 1, 1], index=arr, dtype="Int64", name="count")
    assert expected.index.dtype == arr.dtype
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([1, 1], index=arr[:-1], dtype="Int64", name="count")
    assert expected.index.dtype == arr.dtype
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    # [FUNCTION DOC]
    # Verifies that counting values with normalization on a boolean series containing one NA returns proportions of one half each (Float64, named proportion) over the non-missing values, with a boolean-dtype index.
    ser = pd.Series([True, False, pd.NA], dtype="boolean")
    result = ser.value_counts(normalize=True)
    expected = pd.Series([1, 1], index=ser[:-1], dtype="Float64", name="proportion") / 2
    assert expected.index.dtype == "boolean"
    tm.assert_series_equal(result, expected)


def test_diff():
    # [FUNCTION DOC]
    # Verifies differencing a boolean array with a step of one through the internal algorithms function and the series method: the result is a boolean array where each element is the xor of adjacent values, and the first element and any element involving a missing neighbor are missing.
    a = pd.array(
        [True, True, False, False, True, None, True, None, False], dtype="boolean"
    )
    result = pd.core.algorithms.diff(a, 1)
    expected = pd.array(
        [None, False, True, False, True, None, None, None, None], dtype="boolean"
    )
    tm.assert_extension_array_equal(result, expected)

    ser = pd.Series(a)
    result = ser.diff()
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)
