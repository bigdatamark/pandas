# [MODULE DOC]
# Tests for the scalar boxing helpers in the dtype cast module: converting numpy
# and datetime scalars to their native or pandas equivalents (box native) and
# converting numpy scalars to python or pandas scalars depending on a global
# option (unbox numpy scalar).

from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.core.dtypes.cast import (
    maybe_box_native,
    maybe_unbox_numpy_scalar,
)

from pandas import (
    Interval,
    Period,
    Timedelta,
    Timestamp,
)


@pytest.mark.parametrize(
    "obj,expected_dtype",
    [
        (b"\x00\x10", bytes),
        (4, int),
        (np.uint(4), int),
        (np.int32(-4), int),
        (np.uint8(4), int),
        (float(454.98), float),
        (np.float16(0.4), float),
        (np.float64(1.4), float),
        (np.bool_(False), bool),
        (datetime(2005, 2, 25), datetime),
        (np.datetime64("2005-02-25"), Timestamp),
        (Timestamp("2005-02-25"), Timestamp),
        (np.timedelta64(1, "D"), Timedelta),
        (Timedelta(1, "D"), Timedelta),
        (Interval(0, 1), Interval),
        (Period("4Q2005"), Period),
    ],
)
def test_maybe_box_native(obj, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over (object, expected type) pairs covering bytes, python and numpy
    # integers, floats and bool, datetime objects, numpy datetime/timedelta scalars,
    # Timestamp, Timedelta, Interval and Period. The type of the boxed result must be
    # exactly the expected type (numpy scalars become python scalars; numpy
    # datetime-like scalars become pandas scalars; others unchanged).
    boxed_obj = maybe_box_native(obj)
    result_dtype = type(boxed_obj)
    assert result_dtype is expected_dtype


@pytest.mark.parametrize("typecode", np.typecodes["All"])
def test_maybe_unbox_numpy_scalar(typecode, using_python_scalars):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Parametrized over every numpy type
    # code; fixture using_python_scalars selects the option state. A python value
    # suited to the code (boolean, integer, float, complex, bytes, string, datetime
    # or timedelta) is stored in a one-element array of that type and the element
    # extracted. With python scalars enabled the unboxed result must have exactly the
    # mapped python/pandas type (integers, floats, complex, bytes, string, Timestamp,
    # Timedelta; the object code maps to integer); otherwise the very same numpy
    # scalar object is returned. An unrecognized type code raises a value error.
    # https://github.com/pandas-dev/pandas/pull/63016
    if typecode == "?":
        scalar = False
        expected = bool
    elif typecode in "bhilqnpBHILQNP":
        scalar = 0
        expected = int
    elif typecode in "efdg":
        scalar = 0.0
        expected = float
    elif typecode in "FDG":
        scalar = 0.0 + 0.0j
        expected = complex
    elif typecode in "SV":
        scalar = b""
        expected = bytes
    elif typecode == "U":
        scalar = ""
        expected = str
    elif typecode == "O":
        scalar = 0
        expected = int
    elif typecode == "M":
        scalar = datetime(2025, 1, 1)
        expected = Timestamp
    elif typecode == "m":
        scalar = timedelta(seconds=3)
        expected = Timedelta
    else:
        raise ValueError(f"typecode {typecode} not recognized")
    value = np.array([scalar], dtype=typecode)[0]
    result = maybe_unbox_numpy_scalar(value)
    if using_python_scalars:
        assert type(result) == expected
    else:
        assert result is value


def test_maybe_unbox_numpy_scalar_timestamp(unit, using_python_scalars):
    # [FUNCTION DOC]
    # Fixture unit gives a datetime resolution; fixture using_python_scalars the
    # option. A numpy datetime scalar unboxes to an equal Timestamp of the same
    # resolution when python scalars are enabled, otherwise it stays the numpy scalar;
    # the result type must match the expectation.
    # https://github.com/pandas-dev/pandas/pull/63016
    value = np.datetime64(1, unit)
    expected = Timestamp(1, unit=unit) if using_python_scalars else value
    result = maybe_unbox_numpy_scalar(value)
    assert result == expected
    assert type(result) == type(expected)


def test_maybe_unbox_numpy_scalar_datetime(unit, using_python_scalars):
    # [FUNCTION DOC]
    # Same as the previous test but for a numpy timedelta scalar: with python scalars
    # enabled the result is an equal Timedelta, otherwise the numpy scalar is kept.
    # KNOWN ISSUE: the name says datetime but the test exercises timedelta scalars, and the previous test with timestamp in its name covers datetimes.
    # https://github.com/pandas-dev/pandas/pull/63016
    value = np.timedelta64(1, unit)
    expected = Timedelta(1, unit=unit) if using_python_scalars else value
    result = maybe_unbox_numpy_scalar(value)
    assert result == expected
    assert type(result) == type(expected)
