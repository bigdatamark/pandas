# [MODULE DOC]
# Tests for constructing the nullable floating array: the direct constructor with data and mask, the generic
# array factory with dtype inference or keyword, errors for invalid input, bool and integer inputs, and Series
# construction. Uses the data fixture from the local conftest and using_nan_is_na. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


def test_uses_pandas_na():
    # [FUNCTION DOC]
    # Missing values in a nullable float array are returned as the pandas NA singleton.
    a = pd.array([1, None], dtype=Float64Dtype())
    assert a[1] is pd.NA


def test_floating_array_constructor():
    # [FUNCTION DOC]
    # Constructing from a float data array and boolean mask gives the expected array and stores the very arrays; lists,
    # integer data or a non-boolean mask raise a type error recommending the array factory; omitting the mask raises a type error about the missing argument.
    values = np.array([1, 2, 3, 4], dtype="float64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = FloatingArray(values, mask)
    expected = pd.array([1, 2, 3, pd.NA], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(result._data, values)
    tm.assert_numpy_array_equal(result._mask, mask)

    msg = r".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values.tolist(), mask)

    with pytest.raises(TypeError, match=msg):
        FloatingArray(values, mask.tolist())

    with pytest.raises(TypeError, match=msg):
        FloatingArray(values.astype(int), mask)

    msg = r"__init__\(\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values)


def test_floating_array_disallows_float16():
    # [FUNCTION DOC]
    # A 16-bit float data array raises a type error saying that dtype is unsupported.
    # GH#44715
    arr = np.array([1, 2], dtype=np.float16)
    mask = np.array([False, False])

    msg = "FloatingArray does not support np.float16 dtype"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(arr, mask)


def test_floating_array_disallows_Float16_dtype(request):
    # [FUNCTION DOC]
    # Asking the array factory for the dtype name Float16 raises a type error that the data type is not understood.
    # KNOWN ISSUE: the request fixture parameter is accepted but never used.
    # GH#44715
    with pytest.raises(TypeError, match="data type 'Float16' not understood"):
        pd.array([1.0, 2.0], dtype="Float16")


def test_floating_array_constructor_copy():
    # [FUNCTION DOC]
    # By default the constructor keeps references to the given data and mask; with copy enabled both are copied.
    values = np.array([1, 2, 3, 4], dtype="float64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = FloatingArray(values, mask)
    assert result._data is values
    assert result._mask is mask

    result = FloatingArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask


def test_to_array():
    # [FUNCTION DOC]
    # Array factory on a list of floats without dtype gives the 64-bit nullable dtype.
    result = pd.array([0.1, 0.2, 0.3, 0.4])
    expected = pd.array([0.1, 0.2, 0.3, 0.4], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        ([1, None], [1, pd.NA]),
        ([None], [pd.NA]),
        ([None, np.nan], [pd.NA, pd.NA]),
        ([1, np.nan], [1, pd.NA]),
        ([np.nan], [pd.NA]),
    ],
)
def test_to_array_none_is_nan(a, b, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over lists with None and NaN. Both become missing (NA); without NaN-is-NA mode a trailing NaN stays a
    # real NaN in the data (unmasked) in the result.
    result = pd.array(a, dtype="Float64")
    expected = pd.array(b, dtype="Float64")
    if not using_nan_is_na and a[-1] is np.nan:
        assert np.isnan(result[-1])
        expected._mask[-1] = False
    tm.assert_extension_array_equal(result, expected)


def test_to_array_mixed_integer_float():
    # [FUNCTION DOC]
    # A list mixing integers, floats and None converts to nullable Float64.
    result = pd.array([1, 2.0])
    expected = pd.array([1.0, 2.0], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = pd.array([1, None, 2.0])
    expected = pd.array([1.0, None, 2.0], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        ["foo", "bar"],
        "foo",
        1,
        1.0,
        pd.date_range("20130101", periods=2),
        np.array(["foo"]),
        [[1, 2], [3, 4]],
        [np.nan, {"a": 1}],
        # GH#44514 all-NA case used to get quietly swapped out before checking ndim
        np.array([pd.NA] * 6, dtype=object).reshape(3, 2),
    ],
)
def test_to_array_error(values):
    # [FUNCTION DOC]
    # Parametrized over invalid inputs (strings, scalars, datetimes, nested lists, dict element, a two-dimensional
    # all-missing object array). Requesting Float64 raises a type or value error with one of several messages.
    # error in converting existing arrays to FloatingArray
    msg = "|".join(
        [
            "cannot be converted to FloatingDtype",
            "values must be a 1D list-like",
            "Cannot pass scalar",
            r"float\(\) argument must be a string or a (real )?number, not 'dict'",
            "could not convert string to float: 'foo'",
            r"could not convert string to float: np\.str_\('foo'\)",
        ]
    )
    with pytest.raises((TypeError, ValueError), match=msg):
        pd.array(values, dtype="Float64")


@pytest.mark.parametrize("values", [["1", "2", None], ["1.5", "2", None]])
def test_construct_from_float_strings(values):
    # [FUNCTION DOC]
    # Strings holding numbers (and None) convert to Float64 both through the array factory and the private sequence constructor.
    # see also test_to_integer_array_str
    expected = pd.array([float(values[0]), 2, None], dtype="Float64")

    res = pd.array(values, dtype="Float64")
    tm.assert_extension_array_equal(res, expected)

    res = FloatingArray._from_sequence(values)
    tm.assert_extension_array_equal(res, expected)


def test_to_array_inferred_dtype():
    # [FUNCTION DOC]
    # With a numpy float32 input the dtype is 32-bit nullable; with a plain list of floats it is 64-bit.
    # if values has dtype -> respect it
    result = pd.array(np.array([1, 2], dtype="float32"))
    assert result.dtype == Float32Dtype()

    # if values have no dtype -> always float64
    result = pd.array([1.0, 2.0])
    assert result.dtype == Float64Dtype()


def test_to_array_dtype_keyword():
    # [FUNCTION DOC]
    # An explicit dtype keyword is respected, overriding the numpy input dtype.
    result = pd.array([1, 2], dtype="Float32")
    assert result.dtype == Float32Dtype()

    # if values has dtype -> override it
    result = pd.array(np.array([1, 2], dtype="float32"), dtype="Float64")
    assert result.dtype == Float64Dtype()


def test_to_array_integer():
    # [FUNCTION DOC]
    # Integers convert to Float64; the integer item size is not preserved (32-bit integer input still gives 64-bit float).
    result = pd.array([1, 2], dtype="Float64")
    expected = pd.array([1.0, 2.0], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # for integer dtypes, the itemsize is not preserved
    # TODO can we specify "floating" in general?
    result = pd.array(np.array([1, 2], dtype="int32"), dtype="Float64")
    assert result.dtype == Float64Dtype()


@pytest.mark.parametrize(
    "bool_values, values, target_dtype, expected_dtype",
    [
        ([False, True], [0, 1], Float64Dtype(), Float64Dtype()),
        ([False, True], [0, 1], "Float64", Float64Dtype()),
        ([False, True, np.nan], [0, 1, np.nan], Float64Dtype(), Float64Dtype()),
    ],
)
def test_to_array_bool(bool_values, values, target_dtype, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over bool lists (with and without NaN) and dtype given as instance or name. Booleans convert to 0/1 floats of Float64.
    result = pd.array(bool_values, dtype=target_dtype)
    assert result.dtype == expected_dtype
    expected = pd.array(values, dtype=target_dtype)
    tm.assert_extension_array_equal(result, expected)


def test_series_from_float(data, using_nan_is_na):
    # [FUNCTION DOC]
    # Using the data fixture. Building a Series from a numpy float array (with missing as NaN, or NA objects without
    # NaN-is-NA mode) or from a python list with the dtype name equals a Series built directly from the data.
    # construct from our dtype & string dtype
    dtype = data.dtype

    # from float
    expected = pd.Series(data)
    np_res = data.to_numpy(na_value=np.nan, dtype="float")
    if not using_nan_is_na:
        np_res = np_res.astype(object)
        np_res[data.isna()] = pd.NA
    result = pd.Series(np_res, dtype=str(dtype))
    tm.assert_series_equal(result, expected)

    # from list
    expected = pd.Series(data)
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    tm.assert_series_equal(result, expected)
