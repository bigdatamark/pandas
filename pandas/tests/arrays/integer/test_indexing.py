# [MODULE DOC]
# Tests for setting items and filtering with nullable boolean masks on the nullable integer array. No import-time side effects.

import pandas as pd
import pandas._testing as tm


def test_array_setitem_nullable_boolean_mask():
    # [FUNCTION DOC]
    # Series where with a mask from comparing a nullable Series replaces non-matching entries with NA, keeping Int64.
    # GH 31446
    ser = pd.Series([1, 2], dtype="Int64")
    result = ser.where(ser > 1)
    expected = pd.Series([pd.NA, 2], dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_array_setitem():
    # [FUNCTION DOC]
    # Assigning a scalar through a nullable boolean mask from a comparison on the array updates only matching positions.
    # GH 31446
    arr = pd.array([1, 2], dtype="Int64")
    arr[arr > 1] = 1

    expected = pd.array([1, 1], dtype="Int64")
    tm.assert_extension_array_equal(arr, expected)
