# [MODULE DOC]
# Tests for the dtype and array string representations of the nullable floating array, including long
# arrays and frames. Uses the dtype and data_missing fixtures from the local conftest. No import-time side
# effects.

import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


def test_dtypes(dtype):
    # [FUNCTION DOC]
    # Smoke test on the automatically constructed float dtype: its name is not none.
    # KNOWN ISSUE: the statement checking that the numpy kind is float is a bare comparison with no assertion, so it checks nothing.
    # smoke tests on auto dtype construction

    np.dtype(dtype.type).kind == "f"
    assert dtype.name is not None


@pytest.mark.parametrize(
    "dtype, expected",
    [(Float32Dtype(), "Float32Dtype()"), (Float64Dtype(), "Float64Dtype()")],
)
def test_repr_dtype(dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over the 32-bit and 64-bit dtype instances; the repr is the class name followed by empty parentheses.
    assert repr(dtype) == expected


def test_repr_array():
    # [FUNCTION DOC]
    # Short array repr shows a header with the class name, the values with the NA marker, and the length and dtype line.
    result = repr(pd.array([1.0, None, 3.0]))
    expected = "<FloatingArray>\n[1.0, <NA>, 3.0]\nLength: 3, dtype: Float64"
    assert result == expected


def test_repr_array_long():
    # [FUNCTION DOC]
    # A 3000-element array repr is truncated to the first and last ten elements, separated by an ellipsis line, with right-aligned values.
    data = pd.array([1.0, 2.0, None] * 1000)
    expected = """<FloatingArray>
[ 1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,
 ...
 <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>]
Length: 3000, dtype: Float64"""
    result = repr(data)
    assert result == expected


def test_frame_repr(data_missing):
    # [FUNCTION DOC]
    # A frame with a column from the data_missing fixture shows the NA marker and value aligned in a text table.
    df = pd.DataFrame({"A": data_missing})
    result = repr(df)
    expected = "      A\n0  <NA>\n1   0.1"
    assert result == expected
