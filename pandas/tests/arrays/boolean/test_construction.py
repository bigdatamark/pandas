# [MODULE DOC]
# Tests for constructing the nullable boolean array: the direct constructor from value and mask arrays, the public array factory with dtype boolean from many inputs, the internal coerce-to-array helper, conversion from strings, and conversion to NumPy (including copy and readonly behavior).
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array


def test_boolean_array_constructor():
    # [FUNCTION DOC]
    # Verifies the constructor with a bool values array and a bool mask equals the array factory result with the masked element missing. Verifies type errors ("values should be boolean numpy array" / "mask should be boolean numpy array") for list values, list mask, integer-typed values, and a None mask, and value errors about mismatched shapes when either argument is made two-dimensional.
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    result = BooleanArray(values, mask)
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    with pytest.raises(TypeError, match="values should be boolean numpy array"):
        BooleanArray(values.tolist(), mask)

    with pytest.raises(TypeError, match="mask should be boolean numpy array"):
        BooleanArray(values, mask.tolist())

    with pytest.raises(TypeError, match="values should be boolean numpy array"):
        BooleanArray(values.astype(int), mask)

    with pytest.raises(TypeError, match="mask should be boolean numpy array"):
        BooleanArray(values, None)

    with pytest.raises(ValueError, match="values.shape must match mask.shape"):
        BooleanArray(values.reshape(1, -1), mask)

    with pytest.raises(ValueError, match="values.shape must match mask.shape"):
        BooleanArray(values, mask.reshape(1, -1))


def test_boolean_array_constructor_copy():
    # [FUNCTION DOC]
    # Verifies the constructor stores the given values and mask arrays without copying (identity preserved), and that the copy option makes new arrays for both.
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    result = BooleanArray(values, mask)
    assert result._data is values
    assert result._mask is mask

    result = BooleanArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask


def test_to_boolean_array():
    # [FUNCTION DOC]
    # Verifies the array factory with dtype boolean builds the expected array from a list, NumPy bool array, and object array of bools, and with a missing entry from a list and an object array containing None.
    expected = BooleanArray(
        np.array([True, False, True]), np.array([False, False, False])
    )

    result = pd.array([True, False, True], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, True]), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, True], dtype=object), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    expected = BooleanArray(
        np.array([True, False, True]), np.array([False, False, True])
    )

    result = pd.array([True, False, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([True, False, None], dtype=object), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_all_none():
    # [FUNCTION DOC]
    # Verifies an all-None list or object array becomes a boolean array that is fully masked (the underlying data are true for masked slots).
    expected = BooleanArray(np.array([True, True, True]), np.array([True, True, True]))

    result = pd.array([None, None, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([None, None, None], dtype=object), dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        ([True, False, None, np.nan, pd.NA], [True, False, None, None, None]),
        ([True, np.nan], [True, None]),
        ([True, pd.NA], [True, None]),
        ([np.nan, np.nan], [None, None]),
        (np.array([np.nan, np.nan], dtype=float), [None, None]),
    ],
)
def test_to_boolean_array_missing_indicators(a, b):
    # [FUNCTION DOC]
    # Parametrized over inputs containing different missing indicators (None, NaN, NA, float NaN array) paired with the expected list with None. Verifies all indicators are treated as missing.
    result = pd.array(a, dtype="boolean")
    expected = pd.array(b, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        ["foo", "bar"],
        ["1", "2"],
        # "foo",
        [1, 2],
        [1.0, 2.0],
        pd.date_range("20130101", periods=2),
        np.array(["foo"]),
        np.array([1, 2]),
        np.array([1.0, 2.0]),
        [np.nan, {"a": 1}],
    ],
)
def test_to_boolean_array_error(values):
    # [FUNCTION DOC]
    # Parametrized over inputs that are not bool-like (strings, numeric strings, integers other than 0 and 1 as written lists, floats, datetimes, NumPy string/int/float arrays, a list with a dict). Verifies the factory raises a type error "Need to pass bool-like value".
    # error in converting existing arrays to BooleanArray
    msg = "Need to pass bool-like value"
    with pytest.raises(TypeError, match=msg):
        pd.array(values, dtype="boolean")


def test_to_boolean_array_from_integer_array():
    # [FUNCTION DOC]
    # Verifies an integer NumPy array of ones and zeros converts to true/false boolean array, and None in an object-dtype integer array becomes missing.
    result = pd.array(np.array([1, 0, 1, 0]), dtype="boolean")
    expected = pd.array([True, False, True, False], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    result = pd.array(np.array([1, 0, 1, None]), dtype="boolean")
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_from_float_array():
    # [FUNCTION DOC]
    # Verifies a float NumPy array of ones and zeros converts to true/false and NaN becomes missing.
    result = pd.array(np.array([1.0, 0.0, 1.0, 0.0]), dtype="boolean")
    expected = pd.array([True, False, True, False], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    result = pd.array(np.array([1.0, 0.0, 1.0, np.nan]), dtype="boolean")
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_integer_like():
    # [FUNCTION DOC]
    # Verifies a list of Python integers zero and one converts to booleans, and None becomes missing.
    # integers of 0's and 1's
    result = pd.array([1, 0, 1, 0], dtype="boolean")
    expected = pd.array([True, False, True, False], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # with missing values
    result = pd.array([1, 0, 1, None], dtype="boolean")
    expected = pd.array([True, False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_coerce_to_array():
    # [FUNCTION DOC]
    # Exercises the internal coerce-to-array helper (not public API) whose output tuple is fed to the constructor. Verifies that values and mask are kept by identity unless copy is requested; that missing values in the values argument (None) combine with a given mask (as an array, an object array, or a list) so the result is masked where either marks it; that two-dimensional values with no mask are accepted; and that a value error "values.shape and mask.shape must match" is raised when only one of them is two-dimensional.
    # TODO this is currently not public API
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    expected = BooleanArray(values, mask)
    tm.assert_extension_array_equal(result, expected)
    assert result._data is values
    assert result._mask is mask
    result = BooleanArray(*coerce_to_array(values, mask=mask, copy=True))
    expected = BooleanArray(values, mask)
    tm.assert_extension_array_equal(result, expected)
    assert result._data is not values
    assert result._mask is not mask

    # mixed missing from values and mask
    values = [True, False, None, False]
    mask = np.array([False, False, False, True], dtype="bool")
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    expected = BooleanArray(
        np.array([True, False, True, True]), np.array([False, False, True, True])
    )
    tm.assert_extension_array_equal(result, expected)
    result = BooleanArray(*coerce_to_array(np.array(values, dtype=object), mask=mask))
    tm.assert_extension_array_equal(result, expected)
    result = BooleanArray(*coerce_to_array(values, mask=mask.tolist()))
    tm.assert_extension_array_equal(result, expected)

    # raise errors for wrong dimension
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    # passing 2D values is OK as long as no mask
    coerce_to_array(values.reshape(1, -1))

    with pytest.raises(ValueError, match="values.shape and mask.shape must match"):
        coerce_to_array(values.reshape(1, -1), mask=mask)

    with pytest.raises(ValueError, match="values.shape and mask.shape must match"):
        coerce_to_array(values, mask=mask.reshape(1, -1))


def test_coerce_to_array_from_boolean_array():
    # [FUNCTION DOC]
    # Verifies the helper with a boolean array input returns its data and mask by identity (copy only if requested), and raises a value error "cannot pass mask for BooleanArray input" when a mask is also given.
    # passing BooleanArray to coerce_to_array
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")
    arr = BooleanArray(values, mask)
    result = BooleanArray(*coerce_to_array(arr))
    tm.assert_extension_array_equal(result, arr)
    # no copy
    assert result._data is arr._data
    assert result._mask is arr._mask

    result = BooleanArray(*coerce_to_array(arr), copy=True)
    tm.assert_extension_array_equal(result, arr)
    assert result._data is not arr._data
    assert result._mask is not arr._mask

    with pytest.raises(ValueError, match="cannot pass mask for BooleanArray input"):
        coerce_to_array(arr, mask=mask)


def test_coerce_to_numpy_array():
    # [FUNCTION DOC]
    # Verifies conversion of a boolean array to a NumPy array: with missing values it gives an object array holding bools and the NA singleton; without missing values it gives a bool array; forcing bool dtype gives the same bool array, and raises a value error about missing values and na_value when the array has missing values.
    # KNOWN ISSUE: the inline comment for the case without missing values says object dtype, but the expected result is a bool dtype array.
    # with missing values -> object dtype
    arr = pd.array([True, False, None], dtype="boolean")
    result = np.array(arr)
    expected = np.array([True, False, pd.NA], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    # also with no missing values -> object dtype
    arr = pd.array([True, False, True], dtype="boolean")
    result = np.array(arr)
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    # force bool dtype
    result = np.array(arr, dtype="bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)
    # with missing values will raise error
    arr = pd.array([True, False, None], dtype="boolean")
    msg = (
        "cannot convert to 'bool'-dtype NumPy array with missing values. "
        "Specify an appropriate 'na_value' for this dtype."
    )
    with pytest.raises(ValueError, match=msg):
        np.array(arr, dtype="bool")


def test_to_boolean_array_from_strings():
    # [FUNCTION DOC]
    # Verifies the private from-sequence-of-strings constructor maps "True", "False", "1", "1.0", "0", "0.0" to true, false, true, true, false, false and NaN to missing.
    result = BooleanArray._from_sequence_of_strings(
        np.array(["True", "False", "1", "1.0", "0", "0.0", np.nan], dtype=object),
        dtype=pd.BooleanDtype(),
    )
    expected = BooleanArray(
        np.array([True, False, True, True, False, False, False]),
        np.array([False, False, False, False, False, False, True]),
    )

    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_from_strings_invalid_string():
    # [FUNCTION DOC]
    # Verifies the from-strings constructor raises a value error "cannot be cast" for an unrecognized string.
    with pytest.raises(ValueError, match="cannot be cast"):
        BooleanArray._from_sequence_of_strings(["donkey"], dtype=pd.BooleanDtype())


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy(box):
    # [FUNCTION DOC]
    # Parametrized over series versus array container (ids series, array). Verifies the to-numpy method: default yields bool when no missing, object with NA when missing; string dtype yields fixed-width unicode; dtype bool works without missing and raises a value error with missing; with na_value, object/bool/int64/float64 conversions fill the missing slot (None, false, -99, NaN); converting to int64 without na_value when missing raises a value error.
    # KNOWN ISSUE: the comment for the default case claims object dtype with or without missing values, but the first expectation is a bool array.
    con = pd.Series if box else pd.array
    # default (with or without missing values) -> object dtype
    arr = con([True, False, True], dtype="boolean")
    result = arr.to_numpy()
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy()
    expected = np.array([True, False, pd.NA], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy(dtype="str")
    expected = np.array([True, False, pd.NA], dtype=f"{tm.ENDIAN}U5")
    tm.assert_numpy_array_equal(result, expected)

    # no missing values -> can convert to bool, otherwise raises
    arr = con([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype="bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    with pytest.raises(ValueError, match="cannot convert to 'bool'-dtype"):
        result = arr.to_numpy(dtype="bool")

    # specify dtype and na_value
    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy(dtype=object, na_value=None)
    expected = np.array([True, False, None], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype=bool, na_value=False)
    expected = np.array([True, False, False], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="int64", na_value=-99)
    expected = np.array([1, 0, -99], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="float64", na_value=np.nan)
    expected = np.array([1, 0, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    # converting to int or float without specifying na_value raises
    with pytest.raises(ValueError, match="cannot convert to 'int64'-dtype"):
        arr.to_numpy(dtype="int64")


def test_to_numpy_copy():
    # [FUNCTION DOC]
    # Verifies that to-numpy with bool dtype and no copy shares memory with the array (writing to the result changes the array), while requesting a copy leaves the array unchanged.
    # to_numpy can be zero-copy if no missing values
    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype=bool)
    result[0] = False
    tm.assert_extension_array_equal(
        arr, pd.array([False, False, True], dtype="boolean")
    )

    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype=bool, copy=True)
    result[0] = False
    tm.assert_extension_array_equal(arr, pd.array([True, False, True], dtype="boolean"))


def test_to_numpy_readonly():
    # [FUNCTION DOC]
    # Verifies that when the array is flagged readonly, to-numpy to bool yields a non-writeable NumPy array, while converting to int64 (which copies) yields a writeable one. Sets the private readonly flag on the array.
    arr = pd.array([True, False, True], dtype="boolean")
    arr._readonly = True
    result = arr.to_numpy(dtype=bool)
    assert not result.flags.writeable

    result = arr.to_numpy(dtype="int64")
    assert result.flags.writeable
