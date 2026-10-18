# [MODULE DOC]
# Tests for rounding, list conversion and numpy conversion on masked arrays of integer and float dtypes. Builds a module-level
# list of arrays (one per integer and float extension dtype with a trailing missing value) used by the data fixture.

import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray

arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
arrays += [
    pd.array([0.141, -0.268, 5.895, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES
]


@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the module-level integer and float masked arrays; ids are dtype names.
    """
    Fixture returning parametrized 'data' array with different integer and
    floating point types
    """
    return request.param


@pytest.fixture
def numpy_dtype(data):
    # [FUNCTION DOC]
    # HELPER: fixture that derives, from the data array, the numpy scalar type to use for reference computations: float for integer arrays (since missing needs NaN), otherwise the array's own scalar type.
    """
    Fixture returning numpy dtype from 'data' input array.
    """
    # For integer dtype, the numpy conversion must be done to float
    if is_integer_dtype(data):
        numpy_dtype = float
    else:
        numpy_dtype = data.dtype.type
    return numpy_dtype


def test_round(data, numpy_dtype):
    # [FUNCTION DOC]
    # Rounding with no argument and with two decimals equals numpy rounding of the converted data, with missing positions restored to NA in the same dtype.
    # No arguments
    result = data.round()
    np_result = np.round(data.to_numpy(dtype=numpy_dtype, na_value=None))
    exp_np = np_result.astype(object)
    exp_np[data.isna()] = pd.NA
    expected = pd.array(exp_np, dtype=data.dtype)
    tm.assert_extension_array_equal(result, expected)

    # Decimals argument
    result = data.round(decimals=2)
    np_result = np.round(data.to_numpy(dtype=numpy_dtype, na_value=None), decimals=2)
    exp_np = np_result.astype(object)
    exp_np[data.isna()] = pd.NA
    expected = pd.array(exp_np, dtype=data.dtype)
    tm.assert_extension_array_equal(result, expected)


def test_tolist(data):
    # [FUNCTION DOC]
    # List conversion equals the list obtained by iterating over the array.
    result = data.tolist()
    expected = list(data)
    tm.assert_equal(result, expected)


def test_to_numpy():
    # [FUNCTION DOC]
    # A custom masked array subclass defined inside the test (string dtype, NA fill value) converts to numpy as an array where the masked position holds NA (regression for base-class conversion).
    # GH#56991

    class MyStringArray(BaseMaskedArray):
        # [CLASS DOC]
        # HELPER: minimal subclass of the masked array base used only by this test; sets the string dtype, its dtype class and NA as internal fill value, to check generic conversion in the base class.
        dtype = pd.StringDtype()
        _dtype_cls = pd.StringDtype
        _internal_fill_value = pd.NA

    arr = MyStringArray(
        values=np.array(["a", "b", "c"]), mask=np.array([False, True, False])
    )
    result = arr.to_numpy()
    expected = np.array(["a", pd.NA, "c"])
    tm.assert_numpy_array_equal(result, expected)
