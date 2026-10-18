# [MODULE DOC]
# Tests for constructing the nullable integer array: the data/mask constructor, the array factory and private sequence
# constructor with dtype inference and keywords, floats, strings, booleans, errors, and conversions to object and Series. Uses
# fixtures data and data_missing from the local conftest and using_nan_is_na. Defines a constructor fixture. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int32Dtype,
    Int64Dtype,
)


@pytest.fixture(params=[pd.array, IntegerArray._from_sequence])
def constructor(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the array factory and the private sequence constructor of the integer array; returns the callable under test.
    """Fixture returning parametrized IntegerArray from given sequence.

    Used to test dtype conversions.
    """
    return request.param


def test_uses_pandas_na():
    # [FUNCTION DOC]
    # Missing values in a nullable integer array are returned as the NA singleton.
    a = pd.array([1, None], dtype=Int64Dtype())
    assert a[1] is pd.NA


def test_from_dtype_from_float(data, using_nan_is_na):
    # [FUNCTION DOC]
    # Using the data fixture. Building a Series with the dtype name from a float array with NaN works when NaN-is-NA mode is on, else raises a value
    # error about casting NaN; building from a python list or from an array without missing values works.
    # construct from our dtype & string dtype
    dtype = data.dtype

    # from float
    expected = pd.Series(data)
    arr = data.to_numpy(na_value=np.nan, dtype="float")
    if using_nan_is_na:
        result = pd.Series(arr, dtype=str(dtype))
        tm.assert_series_equal(result, expected)
    else:
        msg = "Cannot cast NaN value to Integer dtype"
        with pytest.raises(ValueError, match=msg):
            pd.Series(arr, dtype=str(dtype))

    # from int / list
    expected = pd.Series(data)
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    tm.assert_series_equal(result, expected)

    # from int / array
    expected = pd.Series(data).dropna().reset_index(drop=True)
    dropped = np.array(data.dropna()).astype(np.dtype(dtype.type))
    result = pd.Series(dropped, dtype=str(dtype))
    tm.assert_series_equal(result, expected)


def test_conversions(data_missing):
    # [FUNCTION DOC]
    # Converting a Series with missing data to object gives NA and a python-level integer; elementwise types and values are compared against an expected object array.
    # astype to object series
    df = pd.DataFrame({"A": data_missing})
    result = df["A"].astype("object")
    expected = pd.Series(np.array([pd.NA, 1], dtype=object), name="A")
    tm.assert_series_equal(result, expected)

    # convert to object ndarray
    # we assert that we are exactly equal
    # including type conversions of scalars
    result = df["A"].astype("object").values
    expected = np.array([pd.NA, 1], dtype=object)
    tm.assert_numpy_array_equal(result, expected)

    for r, e in zip(result, expected, strict=True):
        if pd.isnull(r):
            assert pd.isnull(e)
        elif is_integer(r):
            assert r == e
            assert is_integer(e)
        else:
            assert r == e
            assert type(r) == type(e)


def test_integer_array_constructor():
    # [FUNCTION DOC]
    # Constructing from integer data and boolean mask gives the expected array; lists, a list mask, or float data raise a type error recommending the array factory;
    # omitting the mask raises a type error about the missing argument.
    values = np.array([1, 2, 3, 4], dtype="int64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = IntegerArray(values, mask)
    expected = pd.array([1, 2, 3, pd.NA], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    msg = r".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):
        IntegerArray(values.tolist(), mask)

    with pytest.raises(TypeError, match=msg):
        IntegerArray(values, mask.tolist())

    with pytest.raises(TypeError, match=msg):
        IntegerArray(values.astype(float), mask)
    msg = r"__init__\(\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):
        IntegerArray(values)


def test_integer_array_constructor_copy():
    # [FUNCTION DOC]
    # By default the constructor keeps references to the given data and mask; with copy enabled both are copied.
    values = np.array([1, 2, 3, 4], dtype="int64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = IntegerArray(values, mask)
    assert result._data is values
    assert result._mask is mask

    result = IntegerArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask


@pytest.mark.parametrize(
    "a, b",
    [
        ([1, None], [1, np.nan]),
        ([None], [np.nan]),
        ([None, np.nan], [np.nan, np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan]),
    ],
)
def test_to_integer_array_none_is_nan(a, b, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over lists with None and NaN. In NaN-is-NA mode, a list is converted to Int64 equal to the all-NaN form; otherwise creating from NaN raises a value error.
    # KNOWN ISSUE: the second branch builds from the second parameter only, so the first parameter is not used in that mode.
    if using_nan_is_na:
        result = pd.array(a, dtype="Int64")
        expected = pd.array(b, dtype="Int64")
        tm.assert_extension_array_equal(result, expected)
    else:
        msg = "Cannot cast NaN value to Integer dtype"
        with pytest.raises(ValueError, match=msg):
            pd.array(b, dtype="Int64")


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
    ],
)
def test_to_integer_array_error(values):
    # [FUNCTION DOC]
    # Parametrized over invalid inputs (strings, scalars, datetimes, nested lists, dict element). The factory and the private constructor both raise a type or value error with one of several messages.
    # error in converting existing arrays to IntegerArrays
    msg = "|".join(
        [
            "cannot convert float NaN to integer",  # with not using_nan_is_na
            r"cannot be converted to IntegerDtype",
            r"invalid literal for int\(\) with base 10:",
            r"values must be a 1D list-like",
            r"Cannot pass scalar",
            r"int\(\) argument must be a string",
        ]
    )
    with pytest.raises((ValueError, TypeError), match=msg):
        pd.array(values, dtype="Int64")

    with pytest.raises((ValueError, TypeError), match=msg):
        IntegerArray._from_sequence(values)


def test_to_integer_array_inferred_dtype(constructor):
    # [FUNCTION DOC]
    # Using the constructor fixture. A numpy integer input keeps its width as nullable dtype; a plain list of integers gives 64-bit.
    # if values has dtype -> respect it
    result = constructor(np.array([1, 2], dtype="int8"))
    assert result.dtype == Int8Dtype()
    result = constructor(np.array([1, 2], dtype="int32"))
    assert result.dtype == Int32Dtype()

    # if values have no dtype -> always int64
    result = constructor([1, 2])
    assert result.dtype == Int64Dtype()


def test_to_integer_array_dtype_keyword(constructor):
    # [FUNCTION DOC]
    # Using the constructor fixture. An explicit dtype argument is respected and overrides the numpy input dtype.
    result = constructor([1, 2], dtype="Int8")
    assert result.dtype == Int8Dtype()

    # if values has dtype -> override it
    result = constructor(np.array([1, 2], dtype="int8"), dtype="Int32")
    assert result.dtype == Int32Dtype()


def test_to_integer_array_float():
    # [FUNCTION DOC]
    # Whole floats convert to Int64; non-integral floats raise a type error about a non-safe cast; float32 input still gives the requested 64-bit dtype.
    result = IntegerArray._from_sequence([1.0, 2.0], dtype="Int64")
    expected = pd.array([1, 2], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    with pytest.raises(TypeError, match="cannot safely cast non-equivalent"):
        IntegerArray._from_sequence([1.5, 2.0], dtype="Int64")

    # for float dtypes, the itemsize is not preserved
    result = IntegerArray._from_sequence(
        np.array([1.0, 2.0], dtype="float32"), dtype="Int64"
    )
    assert result.dtype == Int64Dtype()


def test_to_integer_array_str():
    # [FUNCTION DOC]
    # Numeric strings convert (with None as missing); an empty string or a decimal string raises a value error about an invalid integer literal.
    result = IntegerArray._from_sequence(["1", "2", None], dtype="Int64")
    expected = pd.array([1, 2, pd.NA], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: .*"
    ):
        IntegerArray._from_sequence(["1", "2", ""], dtype="Int64")

    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: .*"
    ):
        IntegerArray._from_sequence(["1.5", "2.0"], dtype="Int64")


@pytest.mark.parametrize(
    "bool_values, int_values, target_dtype, expected_dtype",
    [
        ([False, True], [0, 1], Int64Dtype(), Int64Dtype()),
        ([False, True], [0, 1], "Int64", Int64Dtype()),
        ([False, True, np.nan], [0, 1, np.nan], Int64Dtype(), Int64Dtype()),
    ],
)
def test_to_integer_array_bool(
    constructor, bool_values, int_values, target_dtype, expected_dtype, using_nan_is_na
):
    # [FUNCTION DOC]
    # Using the constructor fixture. Booleans convert to zero and one; booleans with NaN work in NaN-is-NA mode, otherwise both the constructor and the
    # factory with integers/NaN raise a value error about casting NaN.
    if not using_nan_is_na and np.isnan(bool_values[-1]):
        msg = "Cannot cast NaN value to Integer dtype"
        with pytest.raises(ValueError, match=msg):
            constructor(bool_values, dtype=target_dtype)
        with pytest.raises(ValueError, match=msg):
            pd.array(int_values, dtype=target_dtype)
        return

    result = constructor(bool_values, dtype=target_dtype)
    assert result.dtype == expected_dtype
    expected = pd.array(int_values, dtype=target_dtype)
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values, to_dtype, result_dtype",
    [
        (np.array([1], dtype="int64"), None, Int64Dtype),
        (np.array([1, np.nan]), None, Int64Dtype),
        (np.array([1, np.nan]), "int8", Int8Dtype),
    ],
)
def test_to_integer_array(values, to_dtype, result_dtype, using_nan_is_na):
    # [FUNCTION DOC]
    # Parametrized over numpy arrays and target dtype. Conversion keeps the given integer width or int64 when none is specified, matching the factory; with NaN it
    # raises a value error unless NaN-is-NA mode is on.
    # convert existing arrays to IntegerArrays
    if not using_nan_is_na and np.isnan(values[-1]):
        msg = "Cannot cast NaN value to Integer dtype"
        with pytest.raises(ValueError, match=msg):
            IntegerArray._from_sequence(values, dtype=to_dtype)
        with pytest.raises(ValueError, match=msg):
            pd.array(values, dtype=result_dtype())
        return

    result = IntegerArray._from_sequence(values, dtype=to_dtype)
    assert result.dtype == result_dtype()
    expected = pd.array(values, dtype=result_dtype())
    tm.assert_extension_array_equal(result, expected)


def test_integer_array_from_boolean():
    # [FUNCTION DOC]
    # A boolean array (as numpy bool or object) converts to Int64 with zero and one.
    # GH31104
    expected = pd.array(np.array([True, False]), dtype="Int64")
    result = pd.array(np.array([True, False], dtype=object), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)
