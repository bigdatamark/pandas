# [MODULE DOC]
# Tests for the function that finds a common dtype for a sequence of dtypes:
# numpy numeric, bool, datetime-like and object combinations; categorical,
# tz-aware datetime, period and interval extension dtypes; and the error for
# empty input. Defines a module-level list of interval dtypes used for
# parametrizing the interval tests.

import numpy as np
import pytest

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

from pandas import (
    Categorical,
    Index,
)


@pytest.mark.parametrize(
    "source_dtypes,expected_common_dtype",
    [
        ((np.int64,), np.int64),
        ((np.uint64,), np.uint64),
        ((np.float32,), np.float32),
        ((object,), object),
        # Into ints.
        ((np.int16, np.int64), np.int64),
        ((np.int32, np.uint32), np.int64),
        ((np.uint16, np.uint64), np.uint64),
        # Into floats.
        ((np.float16, np.float32), np.float32),
        ((np.float16, np.int16), np.float32),
        ((np.float32, np.int16), np.float32),
        ((np.uint64, np.int64), np.float64),
        ((np.int16, np.float64), np.float64),
        ((np.float16, np.int64), np.float64),
        # Into others.
        ((np.complex128, np.int32), np.complex128),
        ((object, np.float32), object),
        ((object, np.int16), object),
        # Bool with int.
        ((np.dtype("bool"), np.int64), object),
        ((np.dtype("bool"), np.int32), object),
        ((np.dtype("bool"), np.int16), object),
        ((np.dtype("bool"), np.int8), object),
        ((np.dtype("bool"), np.uint64), object),
        ((np.dtype("bool"), np.uint32), object),
        ((np.dtype("bool"), np.uint16), object),
        ((np.dtype("bool"), np.uint8), object),
        # Bool with float.
        ((np.dtype("bool"), np.float64), object),
        ((np.dtype("bool"), np.float32), object),
        (
            (np.dtype("datetime64[ns]"), np.dtype("datetime64[ns]")),
            np.dtype("datetime64[ns]"),
        ),
        (
            (np.dtype("timedelta64[ns]"), np.dtype("timedelta64[ns]")),
            np.dtype("timedelta64[ns]"),
        ),
        (
            (np.dtype("datetime64[ns]"), np.dtype("datetime64[ms]")),
            np.dtype("datetime64[ns]"),
        ),
        (
            (np.dtype("timedelta64[ms]"), np.dtype("timedelta64[ns]")),
            np.dtype("timedelta64[ns]"),
        ),
        ((np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]")), object),
        ((np.dtype("datetime64[ns]"), np.int64), object),
    ],
)
def test_numpy_dtypes(source_dtypes, expected_common_dtype):
    # [FUNCTION DOC]
    # Parametrized over (source dtype tuple, expected common dtype) pairs covering
    # single dtypes, integer widening (including mixed signedness going to 64-bit
    # integer or float), float widening, integer with float, complex, object, bool
    # with any numeric (object), datetime64 with different resolutions (finer
    # resolution wins), timedelta64 resolutions, and datetime mixed with timedelta
    # or integer (object). Source entries are converted to dtype objects and the
    # common type must equal the expected dtype.
    source_dtypes = [pandas_dtype(x) for x in source_dtypes]
    assert find_common_type(source_dtypes) == expected_common_dtype


def test_raises_empty_input():
    # [FUNCTION DOC]
    # An empty list of dtypes raises a value error saying no types were given.
    with pytest.raises(ValueError, match="no types given"):
        find_common_type([])


@pytest.mark.parametrize(
    "dtypes,exp_type",
    [
        ([CategoricalDtype()], "category"),
        ([object, CategoricalDtype()], object),
        ([CategoricalDtype(), CategoricalDtype()], "category"),
    ],
)
def test_categorical_dtype(dtypes, exp_type):
    # [FUNCTION DOC]
    # Parametrized: a lone categorical dtype stays category; categorical with object
    # gives object; two categorical dtypes give category.
    assert find_common_type(dtypes) == exp_type


def test_datetimetz_dtype_match():
    # [FUNCTION DOC]
    # Two identical tz-aware datetime dtypes give that same dtype (as the nanosecond
    # US/Eastern datetime dtype).
    dtype = DatetimeTZDtype(unit="ns", tz="US/Eastern")
    assert find_common_type([dtype, dtype]) == "datetime64[ns, US/Eastern]"


@pytest.mark.parametrize(
    "dtype2",
    [
        DatetimeTZDtype(unit="ns", tz="Asia/Tokyo"),
        np.dtype("datetime64[ns]"),
        object,
        np.int64,
    ],
)
def test_datetimetz_dtype_mismatch(dtype2):
    # [FUNCTION DOC]
    # Parametrized over a second dtype (different timezone, naive datetime64,
    # object, int64). The common type with a US/Eastern tz-aware dtype is object in
    # both argument orders.
    dtype = DatetimeTZDtype(unit="ns", tz="US/Eastern")
    assert find_common_type([dtype, dtype2]) == object
    assert find_common_type([dtype2, dtype]) == object


def test_period_dtype_match():
    # [FUNCTION DOC]
    # Two identical daily period dtypes give the daily period dtype.
    dtype = PeriodDtype(freq="D")
    assert find_common_type([dtype, dtype]) == "period[D]"


@pytest.mark.parametrize(
    "dtype2",
    [
        DatetimeTZDtype(unit="ns", tz="Asia/Tokyo"),
        PeriodDtype(freq="2D"),
        PeriodDtype(freq="h"),
        np.dtype("datetime64[ns]"),
        object,
        np.int64,
    ],
)
def test_period_dtype_mismatch(dtype2):
    # [FUNCTION DOC]
    # Parametrized over a second dtype (tz-aware datetime, period with another
    # frequency, naive datetime64, object, int64). The common type with a daily
    # period dtype is object in both argument orders.
    dtype = PeriodDtype(freq="D")
    assert find_common_type([dtype, dtype2]) == object
    assert find_common_type([dtype2, dtype]) == object


interval_dtypes = [
    IntervalDtype(np.int64, "right"),
    IntervalDtype(np.float64, "right"),
    IntervalDtype(np.uint64, "right"),
    IntervalDtype(DatetimeTZDtype(unit="ns", tz="US/Eastern"), "right"),
    IntervalDtype("M8[ns]", "right"),
    IntervalDtype("m8[ns]", "right"),
]


@pytest.mark.parametrize("left", interval_dtypes)
@pytest.mark.parametrize("right", interval_dtypes)
def test_interval_dtype(left, right):
    # [FUNCTION DOC]
    # Parametrized over every pair of the module-level interval dtypes. Identical
    # objects give that very object. Two numeric-subtype intervals give a float64
    # subtype interval (closed right); any other combination gives object.
    result = find_common_type([left, right])

    if left is right:
        assert result is left

    elif left.subtype.kind in ["i", "u", "f"]:
        # i.e. numeric
        if right.subtype.kind in ["i", "u", "f"]:
            # both numeric -> common numeric subtype
            expected = IntervalDtype(np.float64, "right")
            assert result == expected
        else:
            assert result == object

    else:
        assert result == object


@pytest.mark.parametrize("dtype", interval_dtypes)
def test_interval_dtype_with_categorical(dtype):
    # [FUNCTION DOC]
    # Parametrized over each interval dtype. The common type of an interval dtype and
    # the dtype of a categorical whose categories have that interval dtype equals
    # the interval dtype.
    obj = Index([], dtype=dtype)

    cat = Categorical([], categories=obj)

    result = find_common_type([dtype, cat.dtype])
    assert result == dtype
