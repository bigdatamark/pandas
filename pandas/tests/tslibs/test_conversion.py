# [MODULE DOC]
# Tests for timezone and unit conversion helpers: consistency of vectorized UTC-to-local and local-to-UTC conversion with the pointwise versions,
# copy semantics, handling of missing sentinels and read-only input, overflow-safe dtype casting (zero-length, big-endian, and timedelta overflow), and
# localizing standard datetimes including subclasses. Defines two private comparison helpers and a datetime subclass used by a test.

from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsTimedelta,
    astype_overflowsafe,
    conversion,
    iNaT,
    timezones,
    tz_convert_from_utc,
    tzconversion,
)

from pandas import (
    Timestamp,
    date_range,
)
import pandas._testing as tm


def _compare_utc_to_local(tz_didx):
    # [FUNCTION DOC]
    # HELPER: used by the timezone conversion tests.
    # Takes a timezone-aware datetime index. Converts its integer UTC values to local time with the vectorized conversion function, and builds the expected
    # values by applying the single-value conversion function to each integer with np vectorize; asserts the two integer arrays are equal. Returns nothing.
    def f(x):
        # [FUNCTION DOC]
        # HELPER: nested function of the comparison helper that converts one integer UTC value to local time using the index's timezone (single-value conversion). Returns the converted integer.
        return tzconversion.tz_convert_from_utc_single(x, tz_didx.tz)

    result = tz_convert_from_utc(tz_didx.asi8, tz_didx.tz)
    expected = np.vectorize(f)(tz_didx.asi8)

    tm.assert_numpy_array_equal(result, expected)


def _compare_local_to_utc(tz_didx, naive_didx):
    # [FUNCTION DOC]
    # HELPER: used by the timezone conversion tests.
    # Takes a timezone-aware index (for its timezone) and a naive index. Verifies that localizing to UTC behaves the same vectorized and pointwise: computes the vectorized result of the localization routine on integer values and the expected result by localizing each element individually, capturing any exception from each. If the vectorized call raised, the pointwise path must raise an exception of the same type; otherwise the pointwise path must not raise and the arrays must be equal.
    # Check that tz_localize behaves the same vectorized and pointwise.
    err1 = err2 = None
    try:
        result = tzconversion.tz_localize_to_utc(naive_didx.asi8, tz_didx.tz)
        err1 = None
    except Exception as err:
        err1 = err

    try:
        expected = naive_didx.map(lambda x: x.tz_localize(tz_didx.tz)).asi8
    except Exception as err:
        err2 = err

    if err1 is not None:
        assert type(err1) == type(err2)
    else:
        assert err2 is None
        tm.assert_numpy_array_equal(result, expected)


def test_tz_localize_to_utc_copies():
    # [FUNCTION DOC]
    # Regression test (GH#46460): converting an integer array from UTC with the UTC timezone, and with no timezone, returns an equal array that does not share memory with the input.
    # GH#46460
    arr = np.arange(5, dtype="i8")
    result = tz_convert_from_utc(arr, tz=timezone.utc)
    tm.assert_numpy_array_equal(result, arr)
    assert not np.shares_memory(arr, result)

    result = tz_convert_from_utc(arr, tz=None)
    tm.assert_numpy_array_equal(result, arr)
    assert not np.shares_memory(arr, result)


def test_tz_convert_single_matches_tz_convert_hourly(tz_aware_fixture):
    # [FUNCTION DOC]
    # Using the timezone-aware fixture, builds hourly nanosecond ranges for March 2014, aware and naive, and runs both comparison helpers to check vectorized and pointwise conversions agree.
    tz = tz_aware_fixture
    tz_didx = date_range("2014-03-01", "2014-04-01", freq="h", tz=tz, unit="ns")
    naive_didx = date_range("2014-03-01", "2014-04-01", freq="h", unit="ns")

    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)


@pytest.mark.parametrize("freq", ["D", "YE"])
def test_tz_convert_single_matches_tz_convert(tz_aware_fixture, freq):
    # [FUNCTION DOC]
    # Parametrized over daily and year-end frequencies with the timezone-aware fixture; same consistency check over a multi-year range.
    tz = tz_aware_fixture
    tz_didx = date_range("2018-01-01", "2020-01-01", freq=freq, tz=tz, unit="ns")
    naive_didx = date_range("2018-01-01", "2020-01-01", freq=freq, unit="ns")

    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param([], id="empty"),
        pytest.param([iNaT], id="all_nat"),
    ],
)
def test_tz_convert_corner(arr):
    # [FUNCTION DOC]
    # Parametrized over an empty list and a list of just the missing sentinel.
    # KNOWN ISSUE: the body immediately overwrites the parameter with an array containing only the missing sentinel, so the empty-array case is never actually exercised and both parametrizations run the same check.
    # Converting that array from UTC with the Tokyo timezone returns an array equal to the input (missing sentinel preserved).
    arr = np.array([iNaT], dtype=np.int64)
    result = tz_convert_from_utc(arr, timezones.maybe_get_tz("Asia/Tokyo"))
    tm.assert_numpy_array_equal(result, arr)


def test_tz_convert_readonly():
    # [FUNCTION DOC]
    # Regression test (GH#35530): converting a read-only integer array from UTC with the UTC timezone works and returns an equal array.
    # GH#35530
    arr = np.array([0], dtype=np.int64)
    arr.setflags(write=False)
    result = tz_convert_from_utc(arr, timezone.utc)
    tm.assert_numpy_array_equal(result, arr)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("dtype", ["M8[ns]", "M8[s]"])
def test_length_zero_copy(dtype, copy):
    # [FUNCTION DOC]
    # Parametrized over source dtype (nanosecond or second datetime64) and the copy flag. Casting an empty array to nanosecond unit with the overflow-safe cast: with copy, the result never shares memory with the input; without copy, if the dtype is unchanged the same object is returned, otherwise the result does not share memory.
    arr = np.array([], dtype=dtype)
    result = astype_overflowsafe(arr, copy=copy, dtype=np.dtype("M8[ns]"))
    if copy:
        assert not np.shares_memory(result, arr)
    elif arr.dtype == result.dtype:
        assert result is arr
    else:
        assert not np.shares_memory(result, arr)


def test_ensure_datetime64ns_bigendian():
    # [FUNCTION DOC]
    # Regression test (GH#29684): casting a big-endian millisecond datetime64 array to nanosecond gives the native-endian nanosecond array with the same value.
    # GH#29684
    arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")
    result = astype_overflowsafe(arr, dtype=np.dtype("M8[ns]"))

    expected = np.array([np.datetime64(1, "ms")], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


def test_ensure_timedelta64ns_overflows():
    # [FUNCTION DOC]
    # Casting year-unit timedeltas (0 to 900 years) to nanoseconds with the overflow-safe cast raises the out-of-bounds timedelta error with a message that 300 years cannot be converted without overflow.
    arr = np.arange(10).astype("m8[Y]") * 100
    msg = r"Cannot convert 300 years to timedelta64\[ns\] without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        astype_overflowsafe(arr, dtype=np.dtype("m8[ns]"))


class SubDatetime(datetime):
    # [CLASS DOC]
    # HELPER: trivial subclass of the standard datetime type used only for the localization test.
    pass


@pytest.mark.parametrize(
    "dt, expected",
    [
        pytest.param(
            Timestamp("2000-01-01"),
            Timestamp("2000-01-01", tz=timezone.utc),
            id="timestamp",
        ),
        pytest.param(
            datetime(2000, 1, 1),
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            id="datetime",
        ),
        pytest.param(
            SubDatetime(2000, 1, 1),
            SubDatetime(2000, 1, 1, tzinfo=timezone.utc),
            id="subclassed_datetime",
        ),
    ],
)
def test_localize_pydatetime_dt_types(dt, expected):
    # [FUNCTION DOC]
    # Regression test (GH 25851). Parametrized over a timestamp, a datetime and a datetime subclass instance at midnight January 1 2000. Localizing to UTC with the localization function gives a value equal to the same value with the UTC timezone attached.
    # GH 25851
    # ensure that subclassed datetime works with
    # localize_pydatetime
    result = conversion.localize_pydatetime(dt, timezone.utc)
    assert result == expected
