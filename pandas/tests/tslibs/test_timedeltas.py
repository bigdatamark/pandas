# [MODULE DOC]
# Tests for timedelta helpers in the compiled timedelta library and the timedelta constructor: conversion of timedelta-like objects to integer nanoseconds (including errors), unsupported ambiguous units, overflow cases, keyword validation, array conversion errors, and conversion of integer arrays to Python timedelta objects.

import re

import numpy as np
import pytest

from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,
    delta_to_nanoseconds,
    ints_to_pytimedelta,
)

from pandas import (
    Timedelta,
    offsets,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "obj,expected",
    [
        (np.timedelta64(14, "D"), 14 * 24 * 3600 * 1e9),
        (Timedelta(minutes=-7), -7 * 60 * 1e9),
        (Timedelta(minutes=-7).to_pytimedelta(), -7 * 60 * 1e9),
        (Timedelta(seconds=1234e-9), 1234),  # GH43764, GH40946
        (
            Timedelta(seconds=1e-9, milliseconds=1e-5, microseconds=1e-1),
            111,
        ),  # GH43764
        (
            Timedelta(days=1, seconds=1e-9, milliseconds=1e-5, microseconds=1e-1),
            24 * 3600e9 + 111,
        ),  # GH43764
        (offsets.Nano(125), 125),
    ],
)
def test_delta_to_nanoseconds(obj, expected):
    # [FUNCTION DOC]
    # Parametrized over timedelta-like inputs (numpy timedelta, pandas timedelta, Python timedelta, fractional sub-nanosecond constructions, and a nano offset) with expected nanosecond totals (regressions GH43764, GH40946); the helper returns the expected value.
    result = delta_to_nanoseconds(obj)
    assert result == expected


def test_delta_to_nanoseconds_error():
    # [FUNCTION DOC]
    # Passing a numpy array raises a type error mentioning the array class; passing a float, a Python int, or numpy integers raises a type error mentioning the type name.
    obj = np.array([123456789], dtype="m8[ns]")

    with pytest.raises(TypeError, match="<class 'numpy.ndarray'>"):
        delta_to_nanoseconds(obj)

    with pytest.raises(TypeError, match="float"):
        delta_to_nanoseconds(1.5)
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(1)
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(np.int64(2))
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(np.int32(3))


def test_delta_to_nanoseconds_td64_MY_raises():
    # [FUNCTION DOC]
    # Passing numpy timedeltas with year or month units raises a value error saying those units are not supported because their nanosecond duration is ambiguous.
    msg = (
        "delta_to_nanoseconds does not support Y or M units, "
        "as their duration in nanoseconds is ambiguous"
    )

    td = np.timedelta64(1234, "Y")

    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)

    td = np.timedelta64(1234, "M")

    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)


@pytest.mark.parametrize("unit", ["Y", "M"])
def test_unsupported_td64_unit_raises(unit):
    # [FUNCTION DOC]
    # Regression test (GH 52806), parametrized over year and month. Constructing a Timedelta from a numpy timedelta with that unit raises a value error listing the allowed units.
    # GH 52806
    with pytest.raises(
        ValueError,
        match=f"Unit {unit} is not supported. "
        "Only unambiguous timedelta values durations are supported. "
        "Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'",
    ):
        Timedelta(np.timedelta64(1, unit))


def test_huge_nanoseconds_overflow():
    # [FUNCTION DOC]
    # Regression test (GH 32402): a Timedelta from 1e10 (default unit and nanoseconds keyword) converts to exactly 1e10 nanoseconds.
    # GH 32402
    assert delta_to_nanoseconds(Timedelta(1e10)) == 1e10
    assert delta_to_nanoseconds(Timedelta(nanoseconds=1e10)) == 1e10


@pytest.mark.parametrize(
    "kwargs", [{"Seconds": 1}, {"seconds": 1, "Nanoseconds": 1}, {"Foo": 2}]
)
def test_kwarg_assertion(kwargs):
    # [FUNCTION DOC]
    # Parametrized over wrong-case or unknown keyword dictionaries. The Timedelta constructor raises a value error listing the allowed keywords; also (GH#53801) a misspelled unit keyword raises the same error.
    err_message = (
        "cannot construct a Timedelta from the passed arguments, "
        "allowed keywords are "
        "[weeks, days, hours, minutes, seconds, "
        "milliseconds, microseconds, nanoseconds]"
    )

    with pytest.raises(ValueError, match=re.escape(err_message)):
        Timedelta(**kwargs)

    with pytest.raises(ValueError, match=re.escape(err_message)):
        # GH#53801 'unit' misspelled as 'units'
        Timedelta(1, units="hours")


class TestArrayToTimedelta64:
    # [CLASS DOC]
    # Tests for error handling of the array to timedelta conversion routine.
    def test_array_to_timedelta64_string_with_unit_2d_raises(self):
        # [FUNCTION DOC]
        # Exercises the path where a unit is given but errors is not coerce, with two-dimensional object values: raises a value error that unit must not be specified.
        # check the 'unit is not None and errors != "coerce"' path
        #  in array_to_timedelta64 raises correctly with 2D values
        values = np.array([["1", 2], [3, "4"]], dtype=object)
        with pytest.raises(ValueError, match="unit must not be specified"):
            array_to_timedelta64(values, unit="s")

    def test_array_to_timedelta64_non_object_raises(self):
        # [FUNCTION DOC]
        # A non-object integer array raises a type error that values must have object dtype (instead of crashing).
        # check we raise, not segfault
        values = np.arange(5)

        msg = "'values' must have object dtype"
        with pytest.raises(TypeError, match=msg):
            array_to_timedelta64(values)


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_ints_to_pytimedelta(unit):
    # [FUNCTION DOC]
    # Parametrized over units second, millisecond, microsecond. With boxing disabled the result equals numpy's conversion to object (Python timedeltas); with boxing enabled the result is an object array of pandas Timedeltas of each element.
    # tests for non-nanosecond cases
    arr = np.arange(6, dtype=np.int64).view(f"m8[{unit}]")

    res = ints_to_pytimedelta(arr, box=False)
    # For non-nanosecond, .astype(object) gives pytimedelta objects
    #  instead of integers
    expected = arr.astype(object)
    tm.assert_numpy_array_equal(res, expected)

    res = ints_to_pytimedelta(arr, box=True)
    expected = np.array([Timedelta(x) for x in arr], dtype=object)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize("unit", ["Y", "M", "ps", "fs", "as"])
def test_ints_to_pytimedelta_unsupported(unit):
    # [FUNCTION DOC]
    # Parametrized over year, month, picosecond, femtosecond, attosecond. Without boxing a not-implemented error is raised (message with a number); with boxing a not-implemented error says only resolutions s, ms, us and ns are supported.
    arr = np.arange(6, dtype=np.int64).view(f"m8[{unit}]")

    with pytest.raises(NotImplementedError, match=r"\d{1,2}"):
        ints_to_pytimedelta(arr, box=False)
    msg = "Only resolutions 's', 'ms', 'us', 'ns' are supported"
    with pytest.raises(NotImplementedError, match=msg):
        ints_to_pytimedelta(arr, box=True)
