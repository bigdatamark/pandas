# [PACKAGE DOC]
# Package of tests for the string accessor (the str namespace of Series and Index) and its implementations for object, string and Arrow-backed dtypes.
# Contains shared helpers defined in this file for the sibling test modules, plus a conftest with a fixture enumerating all string methods. No import-time side effects beyond importing NumPy and pandas.

import numpy as np

import pandas as pd


def is_object_or_nan_string_dtype(dtype):
    # [FUNCTION DOC]
    # HELPER: determines whether a dtype follows missing-value semantics where the missing value is NaN: returns true for the NumPy object dtype or for any dtype whose missing-value marker attribute is NaN (for example the NaN variant of the string dtype), and false otherwise (for example the NA variant). Takes a single dtype argument.
    # KNOWN ISSUE: for a NumPy dtype other than object the first operand is false and the second operand reads a missing-value marker attribute that NumPy dtypes do not have, so such input would raise an attribute error instead of returning false.
    """
    Check if string-like dtype is following NaN semantics, i.e. is object
    dtype or a NaN-variant of the StringDtype.
    """
    return (isinstance(dtype, np.dtype) and dtype == "object") or (
        dtype.na_value is np.nan
    )


def _convert_na_value(ser, expected):
    # [FUNCTION DOC]
    # HELPER: adjusts the expected missing-value marker in an expected result for the dtype of the tested Series. Takes the Series under test and an expected Series. For object dtype the expected is returned unchanged; for other dtypes it returns the expected with missing values replaced by NaN if the dtype's marker is NaN, or by the pandas NA marker otherwise (issue 18463). Returns a new filled Series.
    if ser.dtype != object:
        if ser.dtype.na_value is np.nan:
            expected = expected.fillna(np.nan)
        else:
            # GH#18463
            expected = expected.fillna(pd.NA)
    return expected
