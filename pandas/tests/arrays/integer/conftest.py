# [MODULE DOC]
# HELPER: pytest fixtures shared by the integer-array tests in this directory (dtype over all eight signed and
# unsigned nullable integer types, data, data with missing values, and a selection of both). Imports the dtype
# classes from the core integer array module. No import-time side effects other than fixture registration.

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


@pytest.fixture(
    params=[
        Int8Dtype,
        Int16Dtype,
        Int32Dtype,
        Int64Dtype,
        UInt8Dtype,
        UInt16Dtype,
        UInt32Dtype,
        UInt64Dtype,
    ]
)
def dtype(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the eight nullable integer dtype classes (8, 16, 32, 64 bits, signed and unsigned); returns an instance.
    """Parametrized fixture returning integer 'dtype'"""
    return request.param()


@pytest.fixture
def data(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a ten element nullable integer array of the given dtype containing zero to one hundred values and two missing values at positions four and seven.
    """
    Fixture returning 'data' array with valid and missing values according to
    parametrized integer 'dtype'.

    Used to test dtype conversion with and without missing values.
    """
    return pd.array(
        [0, 1, 2, 3, pd.NA, 10, 11, pd.NA, 99, 100],
        dtype=dtype,
    )


@pytest.fixture
def data_missing(dtype):
    # [FUNCTION DOC]
    # HELPER: fixture returning a two element nullable integer array of the given dtype with a missing value followed by one.
    """
    Fixture returning array with exactly one NaN and one valid integer,
    according to parametrized integer 'dtype'.

    Used to test dtype conversion with and without missing values.
    """
    return pd.array([pd.NA, 1], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the names data and data_missing; returns the corresponding array so tests cover arrays with and
    # without leading missing values.
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing
