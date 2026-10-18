# [MODULE DOC]
# HELPER: pytest fixtures shared by the floating-array tests in this directory (dtype, data, data with
# missing values, and a parametrized selection of both). Imports the nullable float dtype classes from the core
# floating array module. No import-time side effects other than fixture registration.

import pytest

import pandas as pd
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


@pytest.fixture(params=[Float32Dtype, Float64Dtype])
def dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the 32-bit and 64-bit nullable float dtype classes; returns an instance of the dtype.
    """Parametrized fixture returning a float 'dtype'"""
    return request.param()


@pytest.fixture
def data(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a ten element nullable float array of the given dtype with values from 0.1 to 10.0 and two missing values at positions four and seven.
    """Fixture returning 'data' array according to parametrized float 'dtype'"""
    return pd.array(
        [0.1, 0.2, 0.3, 0.4, pd.NA, 1.0, 1.1, pd.NA, 9.9, 10.0],
        dtype=dtype,
    )


@pytest.fixture
def data_missing(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two element nullable float array of the given dtype: missing first, then 0.1.
    """
    Fixture returning array with missing data according to parametrized float
    'dtype'.
    """
    return pd.array([pd.NA, 0.1], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the names data and data_missing; returns the corresponding array so tests run
    # with and without missing values. Returns nothing for any other parameter value (not reachable).
    """Parametrized fixture returning 'data' or 'data_missing' float arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing
