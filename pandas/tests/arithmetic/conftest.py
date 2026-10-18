# [MODULE DOC]
# HELPER: pytest fixtures for the arithmetic tests, providing scalar and array variants of
# integer one, zeros and several timedelta-like objects. Also defines a module-level list of zero
# variants and a private list of offsets that do not match daily frequency. No other import-time effects.

import numpy as np
import pytest

import pandas as pd
from pandas import Index


@pytest.fixture(params=[1, np.array(1, dtype=np.int64)])
def one(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the integer one and a zero-dimensional int64 array holding one.
    # Returns the current parameter.
    """
    Several variants of integer value 1. The zero-dim integer array
    behaves like an integer.

    This fixture can be used to check that datetimelike indexes handle
    addition and subtraction of integers and zero-dimensional arrays
    of integers.

    Examples
    --------
    dti = pd.date_range('2016-01-01', periods=2, freq='h')
    dti
    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-01 01:00:00'],
    dtype='datetime64[ns]', freq='h')
    dti + one
    DatetimeIndex(['2016-01-01 01:00:00', '2016-01-01 02:00:00'],
    dtype='datetime64[ns]', freq='h')
    """
    return request.param


zeros = [
    box_cls([0] * 5, dtype=dtype)
    for box_cls in [Index, np.array, pd.array]
    for dtype in [np.int64, np.uint64, np.float64]
]
zeros.extend([box_cls([-0.0] * 5, dtype=np.float64) for box_cls in [Index, np.array]])
zeros.extend([np.array(0, dtype=dtype) for dtype in [np.int64, np.uint64, np.float64]])
zeros.extend([np.array(-0.0, dtype=np.float64)])
zeros.extend([0, 0.0, -0.0])


@pytest.fixture(params=zeros)
def zero(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the module-level list of zeros: length-5 Index, numpy and pandas
    # arrays of int64, uint64 and float64 zeros, negative-zero float arrays, zero-dimensional arrays, and
    # the scalars zero, zero float and negative zero float. Returns the current parameter.
    """
    Several types of scalar zeros and length 5 vectors of zeros.

    This fixture can be used to check that numeric-dtype indexes handle
    division by any zero numeric-dtype.

    Uses vector of length 5 for broadcasting with `numeric_idx` fixture,
    which creates numeric-dtype vectors also of length 5.

    Examples
    --------
    arr = RangeIndex(5)
    arr / zeros
    Index([nan, inf, inf, inf, inf], dtype='float64')
    """
    return request.param


# ------------------------------------------------------------------
# Scalar Fixtures


@pytest.fixture(
    params=[
        pd.Timedelta("10m7s").to_pytimedelta(),
        pd.Timedelta("10m7s"),
        pd.Timedelta("10m7s").to_timedelta64(),
    ],
    ids=lambda x: type(x).__name__,
)
def scalar_td(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over three equal timedeltas of ten minutes seven seconds as a python
    # timedelta, a pandas Timedelta and a numpy timedelta64. Ids are the type names. Returns the parameter.
    """
    Several variants of Timedelta scalars representing 10 minutes and 7 seconds.
    """
    return request.param


@pytest.fixture(
    params=[
        pd.offsets.Day(3),
        pd.offsets.Hour(72),
        pd.Timedelta(days=3).to_pytimedelta(),
        pd.Timedelta("72:00:00"),
        np.timedelta64(3, "D"),
        np.timedelta64(72, "h"),
    ],
    ids=lambda x: type(x).__name__,
)
def three_days(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over six objects each representing three days: a Day offset, an Hour
    # offset of 72, a python timedelta, a pandas Timedelta, and numpy timedelta64 values. Returns the parameter.
    """
    Several timedelta-like and DateOffset objects that each represent
    a 3-day timedelta
    """
    return request.param


@pytest.fixture(
    params=[
        pd.offsets.Hour(2),
        pd.offsets.Minute(120),
        pd.Timedelta(hours=2).to_pytimedelta(),
        pd.Timedelta(seconds=2 * 3600),
        np.timedelta64(2, "h"),
        np.timedelta64(120, "m"),
    ],
    ids=lambda x: type(x).__name__,
)
def two_hours(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over six objects each representing two hours: Hour and Minute offsets,
    # python timedelta, pandas Timedelta, and numpy timedelta64 values. Returns the parameter.
    """
    Several timedelta-like and DateOffset objects that each represent
    a 2-hour timedelta
    """
    return request.param


_common_mismatch = [
    pd.offsets.YearBegin(2),
    pd.offsets.MonthBegin(1),
    pd.offsets.Minute(),
]


@pytest.fixture(
    params=[
        np.timedelta64(4, "h"),
        pd.Timedelta(hours=23).to_pytimedelta(),
        pd.Timedelta("23:00:00"),
        *_common_mismatch,
    ]
)
def not_daily(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over timedelta-like and offset objects that are incompatible with daily
    # frequency: a four-hour numpy timedelta, a 23-hour python timedelta and Timedelta, and the shared
    # mismatching offsets (year begin, month begin, minute). Returns the parameter.
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Daily frequencies.
    """
    return request.param
