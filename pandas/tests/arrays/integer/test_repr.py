# [MODULE DOC]
# Tests for the dtype and array string representations of the nullable integer array, including long arrays and frames.
# Uses the data_missing fixture from the local conftest. No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)


def test_dtypes(dtype):
    # [FUNCTION DOC]
    # Smoke test: signed dtypes have numpy kind i, unsigned ones kind u, and every dtype has a name (same as in the dtype tests).
    # smoke tests on auto dtype construction

    if dtype.is_signed_integer:
        assert np.dtype(dtype.type).kind == "i"
    else:
        assert np.dtype(dtype.type).kind == "u"
    assert dtype.name is not None


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (Int8Dtype(), "Int8Dtype()"),
        (Int16Dtype(), "Int16Dtype()"),
        (Int32Dtype(), "Int32Dtype()"),
        (Int64Dtype(), "Int64Dtype()"),
        (UInt8Dtype(), "UInt8Dtype()"),
        (UInt16Dtype(), "UInt16Dtype()"),
        (UInt32Dtype(), "UInt32Dtype()"),
        (UInt64Dtype(), "UInt64Dtype()"),
    ],
)
def test_repr_dtype(dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over the eight dtype instances. Representation is the class name followed by empty parentheses.
    assert repr(dtype) == expected


def test_repr_array():
    # [FUNCTION DOC]
    # Short integer array repr shows a header with the class name, values with the NA marker, and a length and dtype line.
    result = repr(pd.array([1, None, 3]))
    expected = "<IntegerArray>\n[1, <NA>, 3]\nLength: 3, dtype: Int64"
    assert result == expected


def test_repr_array_long():
    # [FUNCTION DOC]
    # A 3000-element array repr is truncated to the first and last ten elements with an ellipsis line and right-aligned values.
    data = pd.array([1, 2, None] * 1000)
    expected = (
        "<IntegerArray>\n"
        "[   1,    2, <NA>,    1,    2, <NA>,    1,    2, <NA>,    1,\n"
        " ...\n"
        " <NA>,    1,    2, <NA>,    1,    2, <NA>,    1,    2, <NA>]\n"
        "Length: 3000, dtype: Int64"
    )
    result = repr(data)
    assert result == expected


def test_frame_repr(data_missing):
    # [FUNCTION DOC]
    # A frame with an integer column with missing renders the NA marker and the number in a text table.
    df = pd.DataFrame({"A": data_missing})
    result = repr(df)
    expected = "      A\n0  <NA>\n1     1"
    assert result == expected
