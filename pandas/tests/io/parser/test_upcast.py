# [MODULE DOC]
# Tests for the private parser helper that upcasts numpy arrays produced by the C tokenizer into nullable extension arrays when a nullable dtype backend is requested (GH#36712).
# Covers integer/float, boolean, all-missing and object (string) inputs. Relies on the per-dtype sentinel missing-value table from the parser extension module.
import numpy as np
import pytest

from pandas._libs.parsers import (
    _maybe_upcast,
    na_values,
)

import pandas as pd
from pandas import NA
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    BooleanArray,
    FloatingArray,
    IntegerArray,
)


def test_maybe_upcast(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Parametrized over every real numpy dtype: builds a three-element array whose last item is the dtype's missing sentinel, upcasts with the nullable backend and asserts the result is an integer array (for integer dtypes) or floating array (otherwise) with only the last element masked.
    # GH#36712

    dtype = np.dtype(any_real_numpy_dtype)
    na_value = na_values[dtype]
    arr = np.array([1, 2, na_value], dtype=dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, True])
    if issubclass(dtype.type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)

    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcast_no_na(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # Parametrized over every real numpy dtype: an array without sentinel values upcasts to an integer or floating extension array with an all-false mask.
    # GH#36712
    arr = np.array([1, 2, 3], dtype=any_real_numpy_dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, False])
    if issubclass(np.dtype(any_real_numpy_dtype).type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)

    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcaste_bool():
    # [FUNCTION DOC]
    # Boolean array whose last element holds the boolean missing sentinel (built through an unsigned byte view) upcasts to a boolean extension array with the last element masked.
    # GH#36712
    dtype = np.bool_
    na_value = na_values[dtype]
    arr = np.array([True, False, na_value], dtype="uint8").view(dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, True])
    expected = BooleanArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcaste_bool_no_nan():
    # [FUNCTION DOC]
    # Boolean array without the sentinel upcasts to a boolean extension array with an all-false mask.
    # GH#36712
    dtype = np.bool_
    arr = np.array([True, False, False], dtype="uint8").view(dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, False])
    expected = BooleanArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcaste_all_nan():
    # [FUNCTION DOC]
    # A 64-bit integer array consisting only of the missing sentinel upcasts to an integer extension array with every element masked.
    # GH#36712
    dtype = np.int64
    na_value = na_values[dtype]
    arr = np.array([na_value, na_value], dtype=dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([True, True])
    expected = IntegerArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("val", [na_values[np.object_], "c"])
def test_maybe_upcast_object(val, string_storage):
    # [FUNCTION DOC]
    # Parametrized over the object-dtype missing sentinel and the string "c", and over the string storage option (via fixture); requires pyarrow. Sets the string storage option, upcasts an object array of two strings plus the value, and asserts the result is a python-backed string array (missing becomes NA) or an Arrow string array (missing becomes null) accordingly.
    # GH#36712
    pa = pytest.importorskip("pyarrow")

    with pd.option_context("mode.string_storage", string_storage):
        arr = np.array(["a", "b", val], dtype=np.object_)
        result = _maybe_upcast(arr, use_dtype_backend=True)

        if string_storage == "python":
            exp_val = "c" if val == "c" else NA
            expected = pd.array(["a", "b", exp_val], dtype=pd.StringDtype())
        else:
            exp_val = "c" if val == "c" else None
            expected = ArrowStringArray(pa.array(["a", "b", exp_val]))
        tm.assert_extension_array_equal(result, expected)
