# [MODULE DOC]
# Tests for converting the nullable floating array (directly and boxed in a Series) to numpy with to_numpy and
# the array protocol: dtype, na_value, copy, errors for missing values, string conversion and read-only handling.
# Uses the using_nan_is_na and dtype fixtures. No import-time side effects.

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy(box, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over Series and array boxes. Default conversion without missing values gives a float64 numpy array; with
    # missing values it gives NaN when NaN-is-NA mode is on, else an object array holding the NA singleton.
    # KNOWN ISSUE: the comment says the default is object dtype with or without missing values, but the first assertion expects float64 when there is no missing value.
    con = pd.Series if box else pd.array

    # default (with or without missing values) -> object dtype
    arr = con([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy()
    expected = np.array([0.1, 0.2, 0.3], dtype="float64")
    # TODO: should this be object with `not using_nan_is_na` to avoid
    #  values-dependent behavior?
    tm.assert_numpy_array_equal(result, expected)

    arr = con([0.1, 0.2, None], dtype="Float64")
    result = arr.to_numpy()
    if using_nan_is_na:
        expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    else:
        expected = np.array([0.1, 0.2, pd.NA], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_float(box):
    # [FUNCTION DOC]
    # Parametrized over boxes. Converting to float64 works without missing values and uses NaN for missing; an explicit NaN
    # na_value gives the same.
    # KNOWN ISSUE: the comment says conversion raises otherwise, but the test expects NaN for missing values.
    con = pd.Series if box else pd.array

    # no missing values -> can convert to float, otherwise raises
    arr = con([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy(dtype="float64")
    expected = np.array([0.1, 0.2, 0.3], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([0.1, 0.2, None], dtype="Float64")
    result = arr.to_numpy(dtype="float64")
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="float64", na_value=np.nan)
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_int(box):
    # [FUNCTION DOC]
    # Parametrized over boxes. Integer conversion of whole floats works; with a missing value it raises a value error; fractional values are truncated.
    con = pd.Series if box else pd.array

    # no missing values -> can convert to int, otherwise raises
    arr = con([1.0, 2.0, 3.0], dtype="Float64")
    result = arr.to_numpy(dtype="int64")
    expected = np.array([1, 2, 3], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([1.0, 2.0, None], dtype="Float64")
    with pytest.raises(ValueError, match="cannot convert to 'int64'-dtype"):
        result = arr.to_numpy(dtype="int64")

    # automatic casting (floors the values)
    arr = con([0.1, 0.9, 1.1], dtype="Float64")
    result = arr.to_numpy(dtype="int64")
    expected = np.array([0, 0, 1], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_na_value(box):
    # [FUNCTION DOC]
    # Parametrized over boxes. A custom na_value replaces missing entries for object (none), bool (false) and integer (minus ninety-nine) outputs.
    con = pd.Series if box else pd.array

    arr = con([0.0, 1.0, None], dtype="Float64")
    result = arr.to_numpy(dtype=object, na_value=None)
    expected = np.array([0.0, 1.0, None], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype=bool, na_value=False)
    expected = np.array([False, True, False], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="int64", na_value=-99)
    expected = np.array([0, 1, -99], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_na_value_with_nan(using_nan_is_na):
    # [FUNCTION DOC]
    # Array built from data with a real NaN and a mask: only masked entries are replaced by na_value; a NaN is also replaced when NaN-is-NA mode marks it missing.
    # array with both NaN and NA -> only fill NA with `na_value`
    mask = np.array([False, False, True])
    if using_nan_is_na:
        mask[1] = True
    arr = FloatingArray(np.array([0.0, np.nan, 0.0]), mask)
    result = arr.to_numpy(dtype="float64", na_value=-1)
    if using_nan_is_na:
        # the NaN passed to the constructor is considered as NA
        expected = np.array([0.0, -1.0, -1.0], dtype="float64")
    else:
        expected = np.array([0.0, np.nan, -1.0], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_dtype(box, dtype):
    # [FUNCTION DOC]
    # Parametrized over boxes and numpy dtypes (floats, integers, bool). Conversion without missing values gives the expected values in that dtype.
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0], dtype="Float64")

    result = arr.to_numpy(dtype=dtype)
    expected = np.array([0, 1], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_na_raises(box, dtype):
    # [FUNCTION DOC]
    # Parametrized over boxes and integer or bool dtypes. Converting with a missing value raises a value error mentioning the dtype.
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0, None], dtype="Float64")
    with pytest.raises(ValueError, match=dtype):
        arr.to_numpy(dtype=dtype)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_string(box, dtype):
    # [FUNCTION DOC]
    # Parametrized over boxes. Conversion to the string kind gives a fixed-width unicode numpy array in native byte order with the NA marker for missing.
    # KNOWN ISSUE: the test requests the dtype fixture but never uses it, which needlessly runs it for each float dtype.
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0, None], dtype="Float64")

    result = arr.to_numpy(dtype="str")
    expected = np.array([0.0, 1.0, pd.NA], dtype=f"{tm.ENDIAN}U32")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_copy():
    # [FUNCTION DOC]
    # Without missing values conversion to float64 is zero-copy by default (mutating the result changes the array); with copy enabled it is independent.
    # to_numpy can be zero-copy if no missing values
    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy(dtype="float64")
    result[0] = 10
    tm.assert_extension_array_equal(arr, pd.array([10, 0.2, 0.3], dtype="Float64"))

    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy(dtype="float64", copy=True)
    result[0] = 10
    tm.assert_extension_array_equal(arr, pd.array([0.1, 0.2, 0.3], dtype="Float64"))


def test_to_numpy_readonly():
    # [FUNCTION DOC]
    # When the array is flagged read-only, a same-dtype no-copy conversion is read-only; a copy, a different dtype, or an object conversion
    # is writeable.
    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    arr._readonly = True
    result = arr.to_numpy(dtype="float64")
    assert not result.flags.writeable

    result = arr.to_numpy(dtype="float64", copy=True)
    assert result.flags.writeable

    result = arr.to_numpy(dtype="float32")
    assert result.flags.writeable

    result = arr.to_numpy(dtype="object")
    assert result.flags.writeable


@pytest.mark.skipif(not np_version_gt2, reason="copy keyword introduced in np 2.0")
@pytest.mark.parametrize("dtype", [None, "float64"])
def test_asarray_readonly(dtype):
    # [FUNCTION DOC]
    # Parametrized over dtype none and float64. Skipped for old numpy. For a read-only array the numpy array protocol gives a read-only
    # result by default and with copy disabled, and a writeable one with copy enabled.
    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    arr._readonly = True

    result = np.asarray(arr, dtype=dtype)
    assert not result.flags.writeable

    result = np.asarray(arr, dtype=dtype, copy=True)
    assert result.flags.writeable

    result = np.asarray(arr, dtype=dtype, copy=False)
    assert not result.flags.writeable
