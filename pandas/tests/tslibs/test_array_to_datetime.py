# [MODULE DOC]
# Tests for the low-level array-to-datetime conversion routines in the internal timestamp parsing library: resolution inference
# (unit chosen when the requested resolution is generic) for all-missing, datetime, date, datetime64, timestamp, string and mixed
# inputs; the timezone-aware variant; parsing of valid date strings; timezone offsets in strings; mixed offsets error; out-of-nanosecond-bounds
# handling; coercion of invalid strings; boundary checks; and datetime subclasses. Defines a module constant holding the generic-resolution
# code used to request unit inference, and a datetime subclass used by one test.

from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

from dateutil.tz.tz import tzoffset
import numpy as np
import pytest

from pandas._libs import (
    NaT,
    iNaT,
    tslib,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

from pandas import Timestamp
import pandas._testing as tm

creso_infer = NpyDatetimeUnit.NPY_FR_GENERIC.value


class TestArrayToDatetimeResolutionInference:
    # [CLASS DOC]
    # Tests that the conversion function infers the output datetime64 unit from the input objects when the generic resolution is requested. Every test passes an object array and expects a tuple of the converted array and the timezone (always none here). There is a TODO in the source about adding cases with time zones and integers.
    # TODO: tests that include tzs, ints

    def test_infer_all_nat(self):
        # [FUNCTION DOC]
        # An array holding only missing values (NaT and NaN) converts to second-unit datetime64 with no timezone.
        arr = np.array([NaT, np.nan], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        assert result.dtype == "M8[s]"

    def test_infer_homogeoneous_datetimes(self):
        # [FUNCTION DOC]
        # Three identical datetimes with microsecond precision yield a microsecond-unit array equal to the expected values, with no timezone.
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        arr = np.array([dt, dt, dt], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, dt, dt], dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_date_objects(self):
        # [FUNCTION DOC]
        # A leading None followed by identical date objects yields a second-unit array with the first element missing and no timezone.
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt2 = dt.date()
        arr = np.array([None, dt2, dt2, dt2], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([np.datetime64("NaT", "ns"), dt2, dt2, dt2], dtype="M8[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_dt64(self):
        # [FUNCTION DOC]
        # A leading None followed by identical millisecond datetime64 values infers millisecond unit; the first element is missing.
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        dt64 = np.datetime64(dt, "ms")
        arr = np.array([None, dt64, dt64, dt64], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(
            [np.datetime64("NaT", "ns"), dt64, dt64, dt64], dtype="M8[ms]"
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_timestamps(self):
        # [FUNCTION DOC]
        # A leading None followed by identical nanosecond timestamps infers nanosecond unit; the first element is missing.
        dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
        ts = Timestamp(dt).as_unit("ns")
        arr = np.array([None, ts, ts, ts], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(
            [np.datetime64("NaT", "ns")] + [ts.asm8] * 3, dtype="M8[ns]"
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_homogeoneous_datetimes_strings(self):
        # [FUNCTION DOC]
        # A leading None followed by identical strings with microsecond precision infers microsecond unit; the first element is missing.
        item = "2023-10-27 18:03:05.678000"
        arr = np.array([None, item, item, item], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(
            [np.datetime64("NaT", "ns"), item, item, item], dtype="M8[us]"
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_infer_heterogeneous(self):
        # [FUNCTION DOC]
        # Strings with different precisions (microseconds, milliseconds, seconds) plus a None are inferred to the finest unit needed, microsecond; the same holds for the reversed array, giving the reversed expected result. No timezone.
        dtstr = "2023-10-27 18:03:05.678000"

        arr = np.array([dtstr, dtstr[:-3], dtstr[:-7], None], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array(arr, dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

        result, tz = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz is None
        tm.assert_numpy_array_equal(result, expected[::-1])

    @pytest.mark.parametrize(
        "item", [float("nan"), NaT.value, float(NaT.value), "NaT", ""]
    )
    def test_infer_with_nat_int_float_str(self, item):
        # [FUNCTION DOC]
        # Parametrized over missing-like items (NaN, the integer missing value, its float form, the string NaT, and empty string). An array of a datetime and the item infers microsecond unit with the item becoming missing (such numbers are only treated as nanoseconds unless they are NaN or the missing sentinel); the reversed array gives the reversed result.
        # floats/ints get inferred to nanos *unless* they are NaN/iNaT,
        # similar NaT string gets treated like NaT scalar (ignored for resolution)
        dt = datetime(2023, 11, 15, 15, 5, 6)

        arr = np.array([dt, item], dtype=object)
        result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
        assert tz is None
        expected = np.array([dt, np.datetime64("NaT", "ns")], dtype="M8[us]")
        tm.assert_numpy_array_equal(result, expected)

        result2, tz2 = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
        assert tz2 is None
        tm.assert_numpy_array_equal(result2, expected[::-1])


class TestArrayToDatetimeWithTZResolutionInference:
    # [CLASS DOC]
    # Tests unit inference for the timezone-aware variant of the conversion.
    def test_array_to_datetime_with_tz_resolution(self):
        # [FUNCTION DOC]
        # With a fixed-offset timezone and generic resolution (other flags false), the result unit is microsecond for strings with milliseconds plus NaT, microsecond for datetimes plus NaT, and second for NaT with a datetime64 in seconds.
        tz = tzoffset("custom", 3600)
        vals = np.array(["2016-01-01 02:03:04.567", NaT], dtype=object)
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        assert res.dtype == "M8[us]"

        vals2 = np.array([datetime(2016, 1, 1, 2, 3, 4), NaT], dtype=object)
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        assert res2.dtype == "M8[us]"

        vals3 = np.array([NaT, np.datetime64(12345, "s")], dtype=object)
        res3 = tslib.array_to_datetime_with_tz(vals3, tz, False, False, creso_infer)
        assert res3.dtype == "M8[s]"

    def test_array_to_datetime_with_tz_resolution_all_nat(self):
        # [FUNCTION DOC]
        # All-missing inputs (a single NaT string, and two NaT objects) give second-unit results.
        tz = tzoffset("custom", 3600)
        vals = np.array(["NaT"], dtype=object)
        res = tslib.array_to_datetime_with_tz(vals, tz, False, False, creso_infer)
        assert res.dtype == "M8[s]"

        vals2 = np.array([NaT, NaT], dtype=object)
        res2 = tslib.array_to_datetime_with_tz(vals2, tz, False, False, creso_infer)
        assert res2.dtype == "M8[s]"


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            ["01-01-2013", "01-02-2013"],
            [
                "2013-01-01T00:00:00.000000000",
                "2013-01-02T00:00:00.000000000",
            ],
        ),
        (
            ["Mon Sep 16 2013", "Tue Sep 17 2013"],
            [
                "2013-09-16T00:00:00.000000000",
                "2013-09-17T00:00:00.000000000",
            ],
        ),
    ],
)
def test_parsing_valid_dates(data, expected):
    # [FUNCTION DOC]
    # Parametrized over two lists of date strings (numeric with dashes, and weekday-month-day-year text). Converting returns microsecond-unit values equal to the expected midnight dates.
    arr = np.array(data, dtype=object)
    result, _ = tslib.array_to_datetime(arr)

    expected = np.array(expected, dtype="M8[us]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "dt_string, expected_tz",
    [
        ["01-01-2013 08:00:00+08:00", 480],
        ["2013-01-01T08:00:00.000000000+0800", 480],
        ["2012-12-31T16:00:00.000000000-0800", -480],
        ["12-31-2012 23:00:00-01:00", -60],
    ],
)
def test_parsing_timezone_offsets(dt_string, expected_tz):
    # [FUNCTION DOC]
    # Parametrized over strings that all represent the same instant with different UTC offsets (and expected offset minutes). The expected value is the conversion of the naive midnight string, cast to nanosecond when the string has nine fractional digits. The conversion of the offset string must equal the expected array and return a timezone equal to a fixed offset of the given number of minutes.
    # All of these datetime strings with offsets are equivalent
    # to the same datetime after the timezone offset is added.
    arr = np.array(["01-01-2013 00:00:00"], dtype=object)
    expected, _ = tslib.array_to_datetime(arr)
    if "000000000" in dt_string:
        expected = expected.astype("M8[ns]")

    arr = np.array([dt_string], dtype=object)
    result, result_tz = tslib.array_to_datetime(arr)

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz == timezone(timedelta(minutes=expected_tz))


def test_parsing_non_iso_timezone_offset():
    # [FUNCTION DOC]
    # A non-ISO string with a zero offset converts with no warning (GH#50949, no deprecation warning about local timezone), gives a nanosecond array of the expected date, and returns the UTC timezone object itself.
    dt_string = "01-01-2013T00:00:00.000000000+0000"
    arr = np.array([dt_string], dtype=object)

    with tm.assert_produces_warning(None):
        # GH#50949 should not get tzlocal-deprecation warning here
        result, result_tz = tslib.array_to_datetime(arr)
    expected = np.array([np.datetime64("2013-01-01 00:00:00.000000000")])

    tm.assert_numpy_array_equal(result, expected)
    assert result_tz is timezone.utc


def test_parsing_different_timezone_offsets():
    # [FUNCTION DOC]
    # Regression test (gh-17697): two strings with different UTC offsets raise a value error saying mixed timezones were detected and to pass utc.
    # see gh-17697
    data = ["2015-11-18 15:30:00+05:30", "2015-11-18 15:30:00+06:30"]
    data = np.array(data, dtype=object)

    msg = "Mixed timezones detected. Pass utc=True in to_datetime"
    with pytest.raises(ValueError, match=msg):
        tslib.array_to_datetime(data)


@pytest.mark.parametrize(
    "invalid_date,exp_unit",
    [
        (date(1000, 1, 1), "s"),
        (datetime(1000, 1, 1), "us"),
        ("1000-01-01", "us"),
        ("Jan 1, 1000", "us"),
        (np.datetime64("1000-01-01"), "s"),
    ],
)
@pytest.mark.parametrize("errors", ["coerce", "raise"])
def test_coerce_outside_ns_bounds(invalid_date, exp_unit, errors):
    # [FUNCTION DOC]
    # Parametrized over dates outside the nanosecond range in several forms (date, datetime, ISO string, text string, datetime64) with the expected unit, and over error handling modes coerce and raise. The result array has the expected unit, matching the unit of a timestamp built from the same value, and equals the timestamp's integer value in that unit.
    arr = np.array([invalid_date], dtype="object")

    result, _ = tslib.array_to_datetime(arr, errors=errors)
    out_reso = np.datetime_data(result.dtype)[0]
    assert out_reso == exp_unit
    ts = Timestamp(invalid_date)
    assert ts.unit == exp_unit

    expected = np.array([ts._value], dtype=f"M8[{exp_unit}]")
    tm.assert_numpy_array_equal(result, expected)


def test_coerce_outside_ns_bounds_one_valid():
    # [FUNCTION DOC]
    # With coercion, an array of an out-of-nanosecond-range string and an in-range string converts to microsecond-unit values for both dates.
    arr = np.array(["1/1/1000", "1/1/2000"], dtype=object)
    result, _ = tslib.array_to_datetime(arr, errors="coerce")

    expected = ["1000-01-01T00:00:00.000000000", "2000-01-01T00:00:00.000000000"]
    expected = np.array(expected, dtype="M8[us]")

    tm.assert_numpy_array_equal(result, expected)


def test_coerce_of_invalid_datetimes():
    # [FUNCTION DOC]
    # With coercion, a valid string and two invalid strings (text and a bare digit) convert to microsecond-unit array with the invalid ones as the missing sentinel.
    # KNOWN ISSUE: the second half of the test repeats the first half exactly (same call, same expectation), so it adds no coverage.
    arr = np.array(["01-01-2013", "not_a_date", "1"], dtype=object)
    # With coercing, the invalid dates becomes iNaT
    result, _ = tslib.array_to_datetime(arr, errors="coerce")
    expected = ["2013-01-01T00:00:00.000000000", iNaT, iNaT]
    tm.assert_numpy_array_equal(result, np.array(expected, dtype="M8[us]"))

    # With coercing, the invalid dates becomes iNaT
    result, _ = tslib.array_to_datetime(arr, errors="coerce")
    expected = ["2013-01-01T00:00:00.000000000", iNaT, iNaT]

    tm.assert_numpy_array_equal(result, np.array(expected, dtype="M8[us]"))


def test_to_datetime_barely_out_of_bounds():
    # [FUNCTION DOC]
    # Regression test (gh-19382, gh-19529): a string one nanosecond beyond the upper bound raises the out-of-bounds datetime error with a message giving the out-of-bounds timestamp truncated to whole seconds.
    # see gh-19382, gh-19529
    #
    # Close enough to bounds that dropping nanos
    # would result in an in-bounds datetime.
    arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)
    msg = "^Out of bounds nanosecond timestamp: 2262-04-11 23:47:16$"

    with pytest.raises(tslib.OutOfBoundsDatetime, match=msg):
        tslib.array_to_datetime(arr)


@pytest.mark.parametrize(
    "timestamp",
    [
        # Close enough to bounds that scaling micros to nanos overflows
        # but adding nanos would result in an in-bounds datetime.
        "1677-09-21T00:12:43.145224193",
        "1677-09-21T00:12:43.145224999",
        # this always worked
        "1677-09-21T00:12:43.145225000",
    ],
)
def test_to_datetime_barely_inside_bounds(timestamp):
    # [FUNCTION DOC]
    # Regression test (gh-57150), parametrized over three timestamps near the lower nanosecond bound (two where scaling microseconds would overflow, one that always worked). Each converts without error to a nanosecond-unit array equal to the expected value.
    # see gh-57150
    result, _ = tslib.array_to_datetime(np.array([timestamp], dtype=object))
    tm.assert_numpy_array_equal(result, np.array([timestamp], dtype="M8[ns]"))


class SubDatetime(datetime):
    # [CLASS DOC]
    # HELPER: trivial subclass of the standard datetime type used only to test subclass handling in the conversion.
    pass


@pytest.mark.parametrize("klass", [SubDatetime, datetime, Timestamp])
def test_datetime_subclass(klass):
    # [FUNCTION DOC]
    # Regression test (GH 25851), parametrized over the subclass, plain datetime and timestamp. An object array of one January 1 2000 instance converts to a microsecond-unit array of that date.
    # GH 25851
    # ensure that subclassed datetime works with
    # array_to_datetime

    arr = np.array([klass(2000, 1, 1)], dtype=object)
    result, _ = tslib.array_to_datetime(arr)

    expected = np.array(["2000-01-01T00:00:00.000000"], dtype="M8[us]")
    tm.assert_numpy_array_equal(result, expected)
