# [MODULE DOC]
# Tests for dtype inference from scalars and arrays in the dtype cast module
# (scalar inference, array inference and the combined entry point), covering
# numpy scalars, python scalars, datetimes, timedeltas, periods, intervals,
# timezone-aware timestamps, zero-dimensional arrays and error cases.

from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.core.dtypes.cast import (
    infer_dtype_from,
    infer_dtype_from_array,
    infer_dtype_from_scalar,
)
from pandas.core.dtypes.common import is_dtype_equal

from pandas import (
    Categorical,
    Interval,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)


def test_infer_dtype_from_int_scalar(any_int_numpy_dtype):
    # [FUNCTION DOC]
    # Fixture any_int_numpy_dtype. A numpy integer scalar of that type is inferred
    # as exactly its own type.
    # Test that infer_dtype_from_scalar is
    # returning correct dtype for int and float.
    data = np.dtype(any_int_numpy_dtype).type(12)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == type(data)


def test_infer_dtype_from_float_scalar(float_numpy_dtype):
    # [FUNCTION DOC]
    # Fixture float_numpy_dtype. A numpy float scalar is inferred as exactly its own
    # type.
    float_numpy_dtype = np.dtype(float_numpy_dtype).type
    data = float_numpy_dtype(12)

    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == float_numpy_dtype


@pytest.mark.parametrize(
    "data,exp_dtype", [(12, np.int64), (np.float64(12), np.float64)]
)
def test_infer_dtype_from_python_scalar(data, exp_dtype):
    # [FUNCTION DOC]
    # Parametrized over a python integer (int64) and a numpy float64 (float64).
    # Inferred dtype equals the expected one.
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == exp_dtype


@pytest.mark.parametrize("bool_val", [True, False])
def test_infer_dtype_from_boolean(bool_val):
    # [FUNCTION DOC]
    # Parametrized over true and false: inferred dtype is numpy bool.
    dtype, val = infer_dtype_from_scalar(bool_val)
    assert dtype == np.bool_


def test_infer_dtype_from_complex(complex_dtype):
    # [FUNCTION DOC]
    # Fixture complex_dtype. Any numpy complex scalar is inferred as complex128.
    data = np.dtype(complex_dtype).type(1)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == np.complex128


def test_infer_dtype_from_datetime():
    # [FUNCTION DOC]
    # Numpy datetime64 and Timestamp scalars infer nanosecond datetime; a python
    # datetime infers microsecond resolution.
    dt64 = np.datetime64(1, "ns")
    dtype, val = infer_dtype_from_scalar(dt64)
    assert dtype == "M8[ns]"

    ts = Timestamp(1)
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == "M8[ns]"

    dt = datetime(2000, 1, 1, 0, 0)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == "M8[us]"


def test_infer_dtype_from_timedelta():
    # [FUNCTION DOC]
    # Numpy timedelta64 and Timedelta scalars infer nanosecond timedelta; a python
    # timedelta infers microsecond resolution.
    td64 = np.timedelta64(1, "ns")
    dtype, val = infer_dtype_from_scalar(td64)
    assert dtype == "m8[ns]"

    pytd = timedelta(1)
    dtype, val = infer_dtype_from_scalar(pytd)
    assert dtype == "m8[us]"

    td = Timedelta(1)
    dtype, val = infer_dtype_from_scalar(td)
    assert dtype == "m8[ns]"


@pytest.mark.parametrize("freq", ["M", "D"])
def test_infer_dtype_from_period(freq):
    # [FUNCTION DOC]
    # Parametrized over monthly and daily frequency. A Period scalar infers the
    # matching period dtype and the returned value is the period itself.
    p = Period("2011-01-01", freq=freq)
    dtype, val = infer_dtype_from_scalar(p)

    exp_dtype = f"period[{freq}]"

    assert dtype == exp_dtype
    assert val == p


def test_infer_dtype_misc():
    # [FUNCTION DOC]
    # A plain date object infers object dtype; a tz-aware Timestamp infers the
    # nanosecond tz-aware datetime dtype for its zone.
    dt = date(2000, 1, 1)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == np.object_

    ts = Timestamp(1, tz="US/Eastern")
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == "datetime64[ns, US/Eastern]"


@pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo"])
def test_infer_from_scalar_tz(tz):
    # [FUNCTION DOC]
    # Parametrized over UTC, US/Eastern, Asia/Tokyo. A tz-aware Timestamp infers the
    # nanosecond datetime dtype with that zone and the value returned unchanged.
    dt = Timestamp(1, tz=tz)
    dtype, val = infer_dtype_from_scalar(dt)

    exp_dtype = f"datetime64[ns, {tz}]"

    assert dtype == exp_dtype
    assert val == dt


@pytest.mark.parametrize(
    "left, right, subtype",
    [
        (0, 1, "int64"),
        (0.0, 1.0, "float64"),
        (Timestamp(0), Timestamp(1), "datetime64[ns]"),
        (Timestamp(0, tz="UTC"), Timestamp(1, tz="UTC"), "datetime64[ns, UTC]"),
        (Timedelta(0), Timedelta(1), "timedelta64[ns]"),
    ],
)
def test_infer_from_interval(left, right, subtype, closed):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over (left, right, subtype
    # name) for integers, floats, naive and UTC timestamps, timedeltas; fixture
    # closed supplies the closure. An Interval infers an interval dtype with the
    # matching subtype and closure, and the returned value is the interval.
    # GH 30337
    interval = Interval(left, right, closed)
    result_dtype, result_value = infer_dtype_from_scalar(interval)
    expected_dtype = f"interval[{subtype}, {closed}]"
    assert result_dtype == expected_dtype
    assert result_value == interval


def test_infer_dtype_from_scalar_errors():
    # [FUNCTION DOC]
    # Passing a non-zero-dimensional ndarray to the scalar inference function raises
    # a value error saying the ndarray is invalid.
    msg = "invalid ndarray passed to infer_dtype_from_scalar"

    with pytest.raises(ValueError, match=msg):
        infer_dtype_from_scalar(np.array([1]))


@pytest.mark.parametrize(
    "value, expected",
    [
        ("foo", np.object_),
        (b"foo", np.object_),
        (1, np.int64),
        (1.5, np.float64),
        (np.datetime64("2016-01-01"), np.dtype("M8[s]")),
        (Timestamp("20160101").as_unit("s"), np.dtype("M8[s]")),
        (Timestamp("20160101", tz="UTC").as_unit("s"), "datetime64[s, UTC]"),
    ],
)
def test_infer_dtype_from_scalar(value, expected, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over (value, expected dtype): string and bytes give object
    # (string dtype for str when strings are inferred, via the using_infer_string
    # fixture), python int and float, numpy datetime and second-resolution Timestamps
    # (naive and UTC). Scalar inference must equal the expected dtype. Then the array
    # inference function must raise a type error saying the argument must be list-like.
    dtype, _ = infer_dtype_from_scalar(value)
    if using_infer_string and value == "foo":
        expected = "string"
    assert is_dtype_equal(dtype, expected)

    with pytest.raises(TypeError, match="must be list-like"):
        infer_dtype_from_array(value)


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([1], np.dtype(int)),
        (np.array([1], dtype=np.int64), np.int64),
        ([np.nan, 1, ""], np.object_),
        (np.array([[1.0, 2.0]]), np.float64),
        (Categorical(list("aabc")), "category"),
        (Categorical([1, 2, 3]), "category"),
        (date_range("20160101", periods=3, unit="ns"), np.dtype("=M8[ns]")),
        (
            date_range("20160101", periods=3, tz="US/Eastern", unit="ns"),
            "datetime64[ns, US/Eastern]",
        ),
        (Series([1.0, 2, 3]), np.float64),
        (Series(list("abc")), np.object_),
        (
            Series(date_range("20160101", periods=3, tz="US/Eastern", unit="ns")),
            "datetime64[ns, US/Eastern]",
        ),
    ],
)
def test_infer_dtype_from_array(arr, expected, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over (array-like, expected dtype): lists, numpy arrays, mixed
    # object list, 2-D float array, categoricals, naive and tz-aware date ranges, and
    # Series of floats, strings, tz-aware datetimes. The string Series expects the
    # string dtype when strings are inferred (fixture using_infer_string). Inferred
    # dtype must equal the expected one.
    dtype, _ = infer_dtype_from_array(arr)
    if (
        using_infer_string
        and isinstance(arr, Series)
        and arr.tolist() == ["a", "b", "c"]
    ):
        expected = "string"
    assert is_dtype_equal(dtype, expected)


@pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
def test_infer_dtype_from_scalar_zerodim_datetimelike(cls):
    # [FUNCTION DOC]
    # Parametrized over numpy datetime64 and timedelta64. A zero-dimensional array
    # holding a nanosecond value infers a dtype of that type with an unboxed value of
    # the same numpy scalar type (not an integer), through both the scalar-inference
    # function and the combined function.
    # ndarray.item() can incorrectly return int instead of td64/dt64
    val = cls(1234, "ns")
    arr = np.array(val)

    dtype, res = infer_dtype_from_scalar(arr)
    assert dtype.type is cls
    assert isinstance(res, cls)

    dtype, res = infer_dtype_from(arr)
    assert dtype.type is cls
