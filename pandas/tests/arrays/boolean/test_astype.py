# [MODULE DOC]
# Tests for casting the nullable boolean array to other dtypes (NumPy int, bool, float, str, and the nullable boolean and integer extension dtypes).
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_astype(using_infer_string):
    # [FUNCTION DOC]
    # Verifies casts of a boolean array. With missing values: casting to int64 raises a value error about converting NA to integer, casting to bool raises a value error about converting float NaN, casting to float64 yields 1.0, 0.0, NaN, and casting to str yields strings "True", "False", and either "<NA>" in a fixed-width unicode NumPy array or, under the infer-string option (fixture), a string-dtype extension array with a missing entry.
    # Without missing values: casting to int64 gives 1, 0, 1 and casting to bool gives a bool array of the same truth values.
    # with missing values
    arr = pd.array([True, False, None], dtype="boolean")

    with pytest.raises(ValueError, match="cannot convert NA to integer"):
        arr.astype("int64")

    with pytest.raises(ValueError, match="cannot convert float NaN to"):
        arr.astype("bool")

    result = arr.astype("float64")
    expected = np.array([1, 0, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("str")
    if using_infer_string:
        expected = pd.array(
            ["True", "False", None], dtype=pd.StringDtype(na_value=np.nan)
        )
        tm.assert_extension_array_equal(result, expected)
    else:
        expected = np.array(["True", "False", "<NA>"], dtype=f"{tm.ENDIAN}U5")
        tm.assert_numpy_array_equal(result, expected)

    # no missing values
    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.astype("int64")
    expected = np.array([1, 0, 1], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_to_boolean_array():
    # [FUNCTION DOC]
    # Verifies that casting a boolean array to the boolean dtype, given as a string name or as a dtype instance, returns an array equal to the original.
    # astype to BooleanArray
    arr = pd.array([True, False, None], dtype="boolean")

    result = arr.astype("boolean")
    tm.assert_extension_array_equal(result, arr)
    result = arr.astype(pd.BooleanDtype())
    tm.assert_extension_array_equal(result, arr)


def test_astype_to_integer_array():
    # [FUNCTION DOC]
    # Verifies that casting a boolean array with a missing value to the nullable Int64 dtype yields an integer extension array with 1, 0, and missing.
    # astype to IntegerArray
    arr = pd.array([True, False, None], dtype="boolean")

    result = arr.astype("Int64")
    expected = pd.array([1, 0, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)
