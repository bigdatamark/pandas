"""test to_datetime"""

# [MODULE DOC]
# Tests for the public datetime conversion function (to_datetime), the largest tools test module. Organized into classes: explicit formats, the main scalar/array conversion tests, epoch units, assembling from frame columns, miscellaneous behaviors (ISO 8601 handling, constructors, NaT strings), format guessing, format inference, day-of-month validity, parsing wrappers, the origin parameter, and the caching heuristic; followed by module-level tests of ISO 8601 and mixed formats and mixed time zone/awareness handling.
# Defines constants for the parse failure advice message regex and for version-dependent text of day-out-of-range errors (depending on the Python version being 3.14 or later), a fixture for epoch units, a fixture for Julian dates, and two module-level values (a UTC-offset date string and its Timestamp) used in a parametrization late in the file. Uses the cache, index_or_series and other fixtures from the test configuration. Several tests are skipped or xfailed by locale, platform or optional library (pyarrow, pytz, psycopg2, lxml).
# No import-time side effects other than building those constants and evaluating parametrization arguments (including conversion calls, for example when parametrizing with a converted value).
import calendar
from collections import deque
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal
import locale
import zoneinfo

from dateutil.parser import parse
import numpy as np
import pytest

from pandas._libs import tslib
from pandas._libs.tslibs import (
    iNaT,
    parsing,
)
from pandas.compat import (
    PY314,
    PY315,
    WASM,
)
from pandas.errors import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_datetime64_ns_dtype

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timestamp,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at

PARSING_ERR_MSG = (
    r"You might want to try:\n"
    r"    - passing `format` if your strings have a consistent format;\n"
    r"    - passing `format=\'ISO8601\'` if your strings are all ISO8601 "
    r"but not necessarily in exactly the same format;\n"
    r"    - passing `format=\'mixed\'`, and the format will be inferred "
    r"for each element individually. You might want to use `dayfirst` "
    r"alongside this."
)

if PY314:
    NOT_99 = ", not 99"
    DAY_IS_OUT_OF_RANGE = (
        r"day \d{1,2} must be in range 1\.\.\d{1,2} for "
        r"month \d{1,2} in year \d{4}"
    )
else:
    NOT_99 = ""
    DAY_IS_OUT_OF_RANGE = "day is out of range for month"


class TestTimeConversionFormats:
    # [CLASS DOC]
    # Tests of the datetime conversion function with explicit format strings: numeric date formats, missing values, locale-dependent directives, partial matching, time zone directives, week directives and error cases. Uses the cache and index_or_series fixtures from the test configuration.
    def test_to_datetime_readonly(self, writable):
        # [FUNCTION DOC]
        # GH#34857. Parametrized by the writable fixture. An empty object array whose write flag follows the fixture converts to an index equal to converting an empty list.
        # GH#34857
        arr = np.array([], dtype=object)
        arr.setflags(write=writable)
        result = to_datetime(arr)
        expected = to_datetime([])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "format, expected",
        [
            [
                "%d/%m/%Y",
                [Timestamp("20000101"), Timestamp("20000201"), Timestamp("20000301")],
            ],
            [
                "%m/%d/%Y",
                [Timestamp("20000101"), Timestamp("20000102"), Timestamp("20000103")],
            ],
        ],
    )
    def test_to_datetime_format(self, cache, index_or_series, format, expected):
        # [FUNCTION DOC]
        # Parametrized over (format, expected timestamps), the cache fixture and a container fixture (index or series). Slash-separated strings "1/1/2000", "1/2/2000", "1/3/2000" parse as day-first (1 Jan, 1 Feb, 1 Mar) or month-first (1 Jan, 2 Jan, 3 Jan) depending on the format; the result equals the expected values in the same container type.
        values = index_or_series(["1/1/2000", "1/2/2000", "1/3/2000"])
        result = to_datetime(values, format=format, cache=cache)
        expected = index_or_series(expected)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "arg, expected, format",
        [
            ["1/1/2000", "20000101", "%d/%m/%Y"],
            ["1/1/2000", "20000101", "%m/%d/%Y"],
            ["1/2/2000", "20000201", "%d/%m/%Y"],
            ["1/2/2000", "20000102", "%m/%d/%Y"],
            ["1/3/2000", "20000301", "%d/%m/%Y"],
            ["1/3/2000", "20000103", "%m/%d/%Y"],
        ],
    )
    def test_to_datetime_format_scalar(self, cache, arg, expected, format):
        # [FUNCTION DOC]
        # Parametrized over (string, expected compact date text, format) and the cache fixture. Parsing a scalar with the given day-first or month-first format returns the expected Timestamp.
        result = to_datetime(arg, format=format, cache=cache)
        expected = Timestamp(expected)
        assert result == expected

    def test_to_datetime_format_YYYYMMDD(self, cache):
        # [FUNCTION DOC]
        # A Series of eight-digit integers (two values and five of another) parses with the compact year-month-day format to the expected timestamps, whether the input is integers or their string form.
        ser = Series([19801222, 19801222] + [19810105] * 5)
        expected = Series([Timestamp(x) for x in ser.apply(str)])

        result = to_datetime(ser, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

        result = to_datetime(ser.apply(str), format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_with_nat(self, cache):
        # [FUNCTION DOC]
        # A float Series with a NaN converts with the compact format to timestamps (microsecond resolution) with NaT in the NaN slot. The string form of the data with the text "nat" in the slot raises a value error: unconverted data remains (".0") since float strings carry a decimal suffix (issue 50051).
        # Explicit cast to float to explicit cast when setting np.nan
        ser = Series([19801222, 19801222] + [19810105] * 5, dtype="float")
        # with NaT
        expected = Series(
            [Timestamp("19801222"), Timestamp("19801222")]
            + [Timestamp("19810105")] * 5,
            dtype="M8[us]",
        )
        expected[2] = np.nan
        ser[2] = np.nan

        result = to_datetime(ser, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

        # string with NaT
        ser2 = ser.apply(str)
        ser2[2] = "nat"
        with pytest.raises(
            ValueError,
            match=(
                'unconverted data remains when parsing with format "%Y%m%d": ".0". '
            ),
        ):
            # https://github.com/pandas-dev/pandas/issues/50051
            to_datetime(ser2, format="%Y%m%d", cache=cache)

    def test_to_datetime_format_YYYYMM_with_nat(self, cache):
        # [FUNCTION DOC]
        # Issue 50237. A float Series of six-digit year-month values with a NaN converts with the year-month format to first-of-month timestamps with NaT in the NaN slot.
        # https://github.com/pandas-dev/pandas/issues/50237
        # Explicit cast to float to explicit cast when setting np.nan
        ser = Series([198012, 198012] + [198101] * 5, dtype="float")
        expected = Series(
            [Timestamp("19801201"), Timestamp("19801201")]
            + [Timestamp("19810101")] * 5,
            dtype="M8[us]",
        )
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format="%Y%m", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_oob_for_ns(self, cache):
        # [FUNCTION DOC]
        # GH 7930, GH 14487. Dates through year 9999 (outside the nanosecond range) convert with the compact format in raise mode to a microsecond-resolution Series with the exact dates.
        # coercion
        # GH 7930, GH 14487
        ser = Series([20121231, 20141231, 99991231])
        result = to_datetime(ser, format="%Y%m%d", errors="raise", cache=cache)
        expected = Series(
            np.array(["2012-12-31", "2014-12-31", "9999-12-31"], dtype="M8[s]"),
            dtype="M8[us]",
        )
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_coercion(self, cache):
        # [FUNCTION DOC]
        # GH 7930. In coerce mode, an integer too large to be a date (a huge value) becomes NaT while valid dates convert; the result has microsecond resolution.
        # coercion
        # GH 7930
        ser = Series([20121231, 20141231, 999999999999999999999999999991231])
        result = to_datetime(ser, format="%Y%m%d", errors="coerce", cache=cache)
        expected = Series(["20121231", "20141231", "NaT"], dtype="M8[us]")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s",
        [
            # Null values with Strings
            ["19801222", "20010112", None],
            ["19801222", "20010112", np.nan],
            ["19801222", "20010112", NaT],
            ["19801222", "20010112", "NaT"],
            # Null values with Integers
            [19801222, 20010112, None],
            [19801222, 20010112, np.nan],
            [19801222, 20010112, NaT],
            [19801222, 20010112, "NaT"],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_with_none(self, input_s):
        # [FUNCTION DOC]
        # GH 30011. Parametrized over lists of two valid compact dates (as strings or integers) plus a null-like third element (None, NaN, NaT, or the string "NaT"). Converting with the compact format gives the two timestamps and NaT.
        # GH 30011
        # format='%Y%m%d'
        # with None
        expected = Series([Timestamp("19801222"), Timestamp("20010112"), NaT])
        result = Series(to_datetime(input_s, format="%Y%m%d"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s, expected",
        [
            # NaN before strings with invalid date values
            [
                ["19801222", np.nan, "20010012", "10019999"],
                [Timestamp("19801222"), np.nan, np.nan, np.nan],
            ],
            # NaN after strings with invalid date values
            [
                ["19801222", "20010012", "10019999", np.nan],
                [Timestamp("19801222"), np.nan, np.nan, np.nan],
            ],
            # NaN before integers with invalid date values
            [
                [20190813, np.nan, 20010012, 20019999],
                [Timestamp("20190813"), np.nan, np.nan, np.nan],
            ],
            # NaN after integers with invalid date values
            [
                [20190813, 20010012, np.nan, 20019999],
                [Timestamp("20190813"), np.nan, np.nan, np.nan],
            ],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_overflow(self, input_s, expected):
        # [FUNCTION DOC]
        # GH 25512. Parametrized over strings or integers where a NaN is placed before or after invalid dates. In coerce mode, valid dates convert and everything else (NaN and invalid) is NaT.
        # GH 25512
        # format='%Y%m%d', errors='coerce'
        input_s = Series(input_s)
        result = to_datetime(input_s, format="%Y%m%d", errors="coerce")
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data, format, expected",
        [
            ([pd.NA], "%Y%m%d%H%M%S", ["NaT"]),
            ([pd.NA], None, ["NaT"]),
            (
                [pd.NA, "20210202202020"],
                "%Y%m%d%H%M%S",
                ["NaT", "2021-02-02 20:20:20"],
            ),
            (["201010", pd.NA], "%y%m%d", ["2020-10-10", "NaT"]),
            (["201010", pd.NA], "%d%m%y", ["2010-10-20", "NaT"]),
            ([None, np.nan, pd.NA], None, ["NaT", "NaT", "NaT"]),
            ([None, np.nan, pd.NA], "%Y%m%d", ["NaT", "NaT", "NaT"]),
        ],
    )
    def test_to_datetime_with_NA(self, data, format, expected):
        # [FUNCTION DOC]
        # GH#42957. Parametrized over (data containing the NA scalar, format or none, expected text). NA values convert to NaT, with or without a format, along with valid strings in various two-digit-year formats (day/month ordering as the format states).
        # GH#42957
        result = to_datetime(data, format=format)
        expected = DatetimeIndex(expected)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_with_NA_with_warning(self):
        # [FUNCTION DOC]
        # GH#42957. Converting a list of an ambiguous six-digit string and NA without a format gives the day-first parse (2010-10-20) and NaT.
        # KNOWN ISSUE: the name says with warning, but no warning is asserted or captured.
        # GH#42957
        result = to_datetime(["201010", pd.NA])
        expected = DatetimeIndex(["2010-10-20", "NaT"])
        tm.assert_index_equal(result, expected)

    def test_to_datetime_format_integer(self, cache):
        # [FUNCTION DOC]
        # GH 10178. A Series of four-digit integer years converts with the year format to January 1 timestamps; a Series of six-digit integers converts with the year-month format to first-of-month timestamps.
        # GH 10178
        ser = Series([2000, 2001, 2002])
        expected = Series([Timestamp(x) for x in ser.apply(str)])

        result = to_datetime(ser, format="%Y", cache=cache)
        tm.assert_series_equal(result, expected)

        ser = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + "-" + x[4:]) for x in ser.apply(str)])

        result = to_datetime(ser, format="%Y%m", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_microsecond(self, cache):
        # [FUNCTION DOC]
        # A string with an abbreviated month name and fractional seconds (three digits), parsed with a day-month name-year-time-fraction format, equals the result of the standard library parser for the same string and format.
        month_abbr = calendar.month_abbr[4]
        val = f"01-{month_abbr}-2011 00:00:01.978"

        format = "%d-%b-%Y %H:%M:%S.%f"
        result = to_datetime(val, format=format, cache=cache)
        exp = datetime.strptime(val, format)
        assert result == exp

    @pytest.mark.parametrize(
        "value, format, dt",
        [
            ["01/10/2010 15:20", "%m/%d/%Y %H:%M", Timestamp("2010-01-10 15:20")],
            ["01/10/2010 05:43", "%m/%d/%Y %I:%M", Timestamp("2010-01-10 05:43")],
            [
                "01/10/2010 13:56:01",
                "%m/%d/%Y %H:%M:%S",
                Timestamp("2010-01-10 13:56:01"),
            ],
            # The 3 tests below are locale-dependent.
            # They pass, except when the machine locale is zh_CN or it_IT .
            pytest.param(
                "01/10/2010 08:14 PM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 20:14"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 07:40 AM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 07:40"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 09:12:56 AM",
                "%m/%d/%Y %I:%M:%S %p",
                Timestamp("2010-01-10 09:12:56"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
        ],
    )
    def test_to_datetime_format_time(self, cache, value, format, dt):
        # [FUNCTION DOC]
        # Parametrized over (string, format, expected Timestamp) with 24-hour, 12-hour and AM/PM variants. The AM/PM cases are marked as expected failures (non-strict) when the machine locale is Chinese or Italian. The scalar result equals the expected Timestamp.
        assert to_datetime(value, format=format, cache=cache) == dt

    @td.skip_if_not_us_locale
    def test_to_datetime_with_non_exact(self, cache):
        # [FUNCTION DOC]
        # GH 10834, 8904. Skipped unless the locale is US English. With exact matching disabled, strings with extra leading or trailing text (like a prefix or a time suffix) parse by searching for the format, giving the same result as first extracting the date substring with a regular expression and parsing that exactly.
        # GH 10834
        # 8904
        # exact kw
        ser = Series(
            ["19MAY11", "foobar19MAY11", "19MAY11:00:00:00", "19MAY11 00:00:00Z"]
        )
        result = to_datetime(ser, format="%d%b%y", exact=False, cache=cache)
        expected = to_datetime(
            ser.str.extract(r"(\d+\w+\d+)", expand=False), format="%d%b%y", cache=cache
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "format, expected",
        [
            ("%Y-%m-%d", Timestamp(2000, 1, 3)),
            ("%Y-%d-%m", Timestamp(2000, 3, 1)),
            ("%Y-%m-%d %H", Timestamp(2000, 1, 3, 12)),
            ("%Y-%d-%m %H", Timestamp(2000, 3, 1, 12)),
            ("%Y-%m-%d %H:%M", Timestamp(2000, 1, 3, 12, 34)),
            ("%Y-%d-%m %H:%M", Timestamp(2000, 3, 1, 12, 34)),
            ("%Y-%m-%d %H:%M:%S", Timestamp(2000, 1, 3, 12, 34, 56)),
            ("%Y-%d-%m %H:%M:%S", Timestamp(2000, 3, 1, 12, 34, 56)),
            ("%Y-%m-%d %H:%M:%S.%f", Timestamp(2000, 1, 3, 12, 34, 56, 123456)),
            ("%Y-%d-%m %H:%M:%S.%f", Timestamp(2000, 3, 1, 12, 34, 56, 123456)),
            (
                "%Y-%m-%d %H:%M:%S.%f%z",
                Timestamp(2000, 1, 3, 12, 34, 56, 123456, tz="UTC+01:00"),
            ),
            (
                "%Y-%d-%m %H:%M:%S.%f%z",
                Timestamp(2000, 3, 1, 12, 34, 56, 123456, tz="UTC+01:00"),
            ),
        ],
    )
    def test_non_exact_doesnt_parse_whole_string(self, cache, format, expected):
        # [FUNCTION DOC]
        # Issue 50412. Parametrized over formats that alternate between ISO 8601 and non-ISO forms, with expected Timestamps. With exact matching disabled, parsing a string with date, time, fraction and offset using a format that covers only a leading part returns just the parsed part (including the UTC+01:00 time zone when the format has the offset directive). Both parsing paths are exercised.
        # KNOWN ISSUE: the cache fixture parameter is accepted but not passed to the conversion call.
        # https://github.com/pandas-dev/pandas/issues/50412
        # the formats alternate between ISO8601 and non-ISO8601 to check both paths
        result = to_datetime(
            "2000-01-03 12:34:56.123456+01:00", format=format, exact=False
        )
        assert result == expected

    @pytest.mark.parametrize(
        "arg",
        [
            "2012-01-01 09:00:00.000000001",
            "2012-01-01 09:00:00.000001",
            "2012-01-01 09:00:00.001",
            "2012-01-01 09:00:00.001000",
            "2012-01-01 09:00:00.001000000",
        ],
    )
    def test_parse_nanoseconds_with_formula(self, cache, arg):
        # [FUNCTION DOC]
        # GH8989. Parametrized over strings with sub-second parts of increasing precision (nanosecond to millisecond). Parsing with an explicit fractional-seconds format equals parsing without a format (nanoseconds are not truncated away when a format is given).
        # GH8989
        # truncating the nanoseconds when a format was provided
        expected = to_datetime(arg, cache=cache)
        result = to_datetime(arg, format="%Y-%m-%d %H:%M:%S.%f", cache=cache)
        assert result == expected

    @pytest.mark.parametrize(
        "value,fmt,expected",
        [
            ["2009324", "%Y%W%w", "2009-08-13"],
            ["2013020", "%Y%U%w", "2013-01-13"],
        ],
    )
    def test_to_datetime_format_weeks(self, value, fmt, expected, cache):
        # [FUNCTION DOC]
        # Parametrized over (value, format, expected date) with the week-of-year (Monday-first and Sunday-first) and weekday directives. The scalar parse equals the expected Timestamp.
        assert to_datetime(value, format=fmt, cache=cache) == Timestamp(expected)

    @pytest.mark.parametrize(
        "fmt,dates,expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                ["2010-01-01 12:00:00 UTC"] * 2,
                [Timestamp("2010-01-01 12:00:00", tz="UTC")] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S%z",
                ["2010-01-01 12:00:00+0100"] * 2,
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))
                    )
                ]
                * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100"] * 2,
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))
                    )
                ]
                * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 Z", "2010-01-01 12:00:00 Z"],
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))
                    ),
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))
                    ),
                ],
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset(self, fmt, dates, expected_dates):
        # [FUNCTION DOC]
        # GH 13486. Parametrized over (format, strings, expected timestamps) for a named UTC zone, a numeric offset with no space, with a space, and the letter Z. The result index equals an index of the expected time-zone-aware timestamps.
        # GH 13486
        result = to_datetime(dates, format=fmt)
        expected = Index(expected_dates)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt,dates,expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                [
                    "2010-01-01 12:00:00 UTC",
                    "2010-01-01 12:00:00 GMT",
                    "2010-01-01 12:00:00 US/Pacific",
                ],
                [
                    Timestamp("2010-01-01 12:00:00", tz="UTC"),
                    Timestamp("2010-01-01 12:00:00", tz="GMT"),
                    Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                ],
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                [
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))
                    ),
                    Timestamp(
                        "2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))
                    ),
                ],
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(
        self, fmt, dates, expected_dates
    ):
        # [FUNCTION DOC]
        # GH#13486, GH#50887, GH#57275. Parametrized over formats and strings with differing zone names or offsets. Without the utc option, a value error "Mixed timezones detected. Pass utc=True in to_datetime" is raised.
        # KNOWN ISSUE: the expected_dates parameter is unused in the body.
        # GH#13486, GH#50887, GH#57275
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self):
        # [FUNCTION DOC]
        # GH 32792. Four strings with different numeric offsets parsed with the offset format and the utc option give the corresponding UTC wall times as a UTC-aware index.
        # GH 32792
        dates = [
            "2010-01-01 12:00:00 +0100",
            "2010-01-01 12:00:00 -0100",
            "2010-01-01 12:00:00 +0300",
            "2010-01-01 12:00:00 +0400",
        ]
        expected_dates = [
            "2010-01-01 11:00:00+00:00",
            "2010-01-01 13:00:00+00:00",
            "2010-01-01 09:00:00+00:00",
            "2010-01-01 08:00:00+00:00",
        ]
        fmt = "%Y-%m-%d %H:%M:%S %z"

        result = to_datetime(dates, format=fmt, utc=True)
        expected = DatetimeIndex(expected_dates)
        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(
        not PY315, reason="%:z directive not supported prior to 3.15", raises=ValueError
    )
    def test_to_datetime_colon_z_offset(self):
        # [FUNCTION DOC]
        # Expected to fail (with a value error) before Python 3.15, where the colon-offset directive is unsupported. Strings with colon-separated offsets parsed with that directive and the utc option give the corresponding UTC times.
        dates = [
            "2010-01-01 12:00:00+04:00",
            "2010-01-01 12:00:00+04:30",
            "2010-01-01 12:00:00-05:00",
        ]
        expected_dates = [
            "2010-01-01 08:00:00+00:00",
            "2010-01-01 07:30:00+00:00",
            "2010-01-01 17:00:00+00:00",
        ]
        fmt = "%Y-%m-%d %H:%M:%S%:z"

        result = to_datetime(dates, format=fmt, utc=True)
        expected = DatetimeIndex(expected_dates)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_missing_colon_z_offset(self):
        # [FUNCTION DOC]
        # Adapted from a CPython test. A string whose offset has extra digits after the colon, parsed with the colon-offset directive and utc option, raises a value error. The expected message depends on the Python version: a missing colon message on 3.15 and later, otherwise a bad directive message.
        # test adapted from python/cpython#136961
        dates = ["+04:0030"]
        fmt = "%:z"

        if PY315:
            msg = r"Missing colon in %:z before '30', got '\+04:0030'"
        else:
            msg = "':' is a bad directive in format '%:z'"

        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt, utc=True)

    @pytest.mark.parametrize(
        "offset", ["+0", "-1foo", "UTCbar", ":10", "+01:000:01", ""]
    )
    def test_to_datetime_parse_timezone_malformed(self, offset):
        # [FUNCTION DOC]
        # Parametrized over malformed offset suffixes (like a lone sign and digit, trailing letters, a colon prefix, too many digits, empty). Parsing a string with such a suffix with the offset format raises a value error: either the text does not match the format or unconverted data remains, followed by the standard advice about passing a format.
        fmt = "%Y-%m-%d %H:%M:%S %z"
        date = "2010-01-01 12:00:00 " + offset

        msg = "|".join(
            [
                r'^time data ".*" doesn\'t match format ".*". ' f"{PARSING_ERR_MSG}$",
                r'^unconverted data remains when parsing with format ".*": ".*". '
                f"{PARSING_ERR_MSG}$",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self):
        # [FUNCTION DOC]
        # GH 21697. Converting an Index named foo with a Z suffix and the offset format gives a UTC-aware datetime index that keeps the name.
        # GH 21697
        fmt = "%Y-%m-%d %H:%M:%S %z"
        arg = Index(["2010-01-01 12:00:00 Z"], name="foo")
        result = to_datetime(arg, format=fmt)
        expected = DatetimeIndex(["2010-01-01 12:00:00"], tz="UTC", name="foo")
        tm.assert_index_equal(result, expected)


class TestToDatetime:
    # [CLASS DOC]
    # Main collection of tests for the datetime conversion function on many input kinds: strings, scalars, NumPy and Python datetimes, arrays, time zones and offsets, units of epoch numbers, caching, and error modes (this class covers the first part of those, other classes below cover units, frames, miscellany, and more).
    def test_to_datetime_mixed_string_resos(self):
        # [FUNCTION DOC]
        # GH#62801. Four ISO strings with 0, 3, 6 and 9 fractional digits convert to a nanosecond-resolution index when passed to the index constructor, or to the conversion function with the ISO 8601 format or the mixed format; all equal the per-element nanosecond Timestamps.
        # GH#62801
        vals = [
            "2016-01-01 01:02:03",
            "2016-01-01 01:02:03.001",
            "2016-01-01 01:02:03.001002",
            "2016-01-01 01:02:03.001002003",
        ]
        expected = DatetimeIndex([Timestamp(x).as_unit("ns") for x in vals])

        result1 = DatetimeIndex(vals)
        tm.assert_index_equal(result1, expected)

        result2 = to_datetime(vals, format="ISO8601")
        tm.assert_index_equal(result2, expected)

        result3 = to_datetime(vals, format="mixed")
        tm.assert_index_equal(result3, expected)

    def test_to_datetime_none(self):
        # [FUNCTION DOC]
        # GH#23055. Converting None returns the NaT singleton.
        # GH#23055
        assert to_datetime(None) is NaT

    @pytest.mark.filterwarnings("ignore:Could not infer format")
    def test_to_datetime_overflow(self):
        # [FUNCTION DOC]
        # Filters the could-not-infer-format warning. A digit string that overflows datetime parsing raises OutOfBoundsDatetime with a message about parsing overflow, for scalar and list input; in coerce mode the scalar becomes NaT and the list becomes a second-resolution all-NaT index.
        # we should get an OutOfBoundsDatetime, NOT OverflowError
        # TODO: Timestamp raises ValueError("could not convert string to Timestamp")
        #  can we make these more consistent?
        arg = "08335394550"
        msg = 'Parsing "08335394550" to datetime overflows'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arg)

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime([arg])

        res = to_datetime(arg, errors="coerce")
        assert res is NaT
        res = to_datetime([arg], errors="coerce")
        exp = Index([NaT], dtype="M8[s]")
        tm.assert_index_equal(res, exp)

    def test_to_datetime_mixed_datetime_and_string(self):
        # [FUNCTION DOC]
        # GH#47018. A list with a string carrying a -0100 offset and a datetime with that fixed offset converts to an index equal to converting two datetimes, expressed in the fixed offset time zone.
        # GH#47018 adapted old doctest with new behavior
        d1 = datetime(2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1)))
        d2 = datetime(2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1)))
        res = to_datetime(["2020-01-01 17:00 -0100", d2])
        expected = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
        tm.assert_index_equal(res, expected)

    def test_to_datetime_mixed_string_and_numeric(self):
        # [FUNCTION DOC]
        # GH#55780. A list with an ISO date string and the number zero parses (with the mixed format) as 2016-01-01 and the epoch 1970-01-01, in either order, and the index constructor gives the same (the number must not be cast to text).
        # GH#55780 np.array(vals) would incorrectly cast the number to str
        vals = ["2016-01-01", 0]
        expected = DatetimeIndex([Timestamp(x) for x in vals])
        result = to_datetime(vals, format="mixed")
        result2 = to_datetime(vals[::-1], format="mixed")[::-1]
        result3 = DatetimeIndex(vals)
        result4 = DatetimeIndex(vals[::-1])[::-1]

        tm.assert_index_equal(result, expected)
        tm.assert_index_equal(result2, expected)
        tm.assert_index_equal(result3, expected)
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize(
        "format", ["%Y-%m-%d", "%Y-%d-%m"], ids=["ISO8601", "non-ISO8601"]
    )
    def test_to_datetime_mixed_date_and_string(self, format):
        # [FUNCTION DOC]
        # Issue 50108. Parametrized over an ISO and a non-ISO format. A list with a date string and a date object converts with the format to a microsecond-resolution index of both dates.
        # https://github.com/pandas-dev/pandas/issues/50108
        d1 = date(2020, 1, 2)
        res = to_datetime(["2020-01-01", d1], format=format)
        expected = DatetimeIndex(["2020-01-01", "2020-01-02"], dtype="M8[us]")
        tm.assert_index_equal(res, expected)

    @pytest.mark.parametrize(
        "fmt",
        ["%Y-%d-%m %H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"],
        ids=["non-ISO8601 format", "ISO8601 format"],
    )
    @pytest.mark.parametrize(
        "utc, args, expected",
        [
            pytest.param(
                True,
                ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00-08:00"],
                DatetimeIndex(
                    ["2000-01-01 09:00:00+00:00", "2000-01-01 10:00:00+00:00"],
                    dtype="datetime64[us, UTC]",
                ),
                id="all tz-aware, with utc",
            ),
            pytest.param(
                False,
                ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                DatetimeIndex(
                    ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                ).as_unit("us"),
                id="all tz-aware, without utc",
            ),
            pytest.param(
                True,
                ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00+00:00"],
                DatetimeIndex(
                    ["2000-01-01 09:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                    dtype="datetime64[us, UTC]",
                ),
                id="all tz-aware, mixed offsets, with utc",
            ),
            pytest.param(
                True,
                ["2000-01-01 01:00:00", "2000-01-01 02:00:00+00:00"],
                DatetimeIndex(
                    ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                    dtype="datetime64[us, UTC]",
                ),
                id="tz-aware string, naive pydatetime, with utc",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "constructor",
        [Timestamp, lambda x: Timestamp(x).to_pydatetime()],
    )
    def test_to_datetime_mixed_datetime_and_string_with_format(
        self, fmt, utc, args, expected, constructor
    ):
        # [FUNCTION DOC]
        # Issues 49298, 50254. Parametrized over a format (non-ISO and ISO, with offset), the (utc flag, two inputs, expected index) cases, and a constructor that makes the first input a Timestamp or a Python datetime. Converts [first, second string] with the format and utc flag; the result equals the expected index (UTC-aware when utc is true, offset preserved when all share the same offset). When the constructor yields Timestamp, the expected is cast to microsecond resolution.
        # https://github.com/pandas-dev/pandas/issues/49298
        # https://github.com/pandas-dev/pandas/issues/50254
        # note: ISO8601 formats go down a fastpath, so we need to check both
        # an ISO8601 format and a non-ISO8601 one
        ts1 = constructor(args[0])
        ts2 = args[1]
        result = to_datetime([ts1, ts2], format=fmt, utc=utc)
        if constructor is Timestamp:
            expected = expected.as_unit("us")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt",
        ["%Y-%d-%m %H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"],
        ids=["non-ISO8601 format", "ISO8601 format"],
    )
    @pytest.mark.parametrize(
        "constructor",
        [Timestamp, lambda x: Timestamp(x).to_pydatetime()],
    )
    def test_to_datetime_mixed_dt_and_str_with_format_mixed_offsets_utc_false_removed(
        self, fmt, constructor
    ):
        # [FUNCTION DOC]
        # GH#57275. Parametrized over format and constructor. A naive datetime-like and a string with +00:00 offset converted with utc false raises a value error "Mixed timezones detected. Pass utc=True in to_datetime".
        # https://github.com/pandas-dev/pandas/issues/49298
        # https://github.com/pandas-dev/pandas/issues/50254
        # GH#57275
        # note: ISO8601 formats go down a fastpath, so we need to check both
        # an ISO8601 format and a non-ISO8601 one
        args = ["2000-01-01 01:00:00", "2000-01-01 02:00:00+00:00"]
        ts1 = constructor(args[0])
        ts2 = args[1]
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"

        with pytest.raises(ValueError, match=msg):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            pytest.param(
                "%Y-%m-%d %H:%M:%S%z",
                [
                    Timestamp("2000-01-01 09:00:00+0100", tz="UTC+01:00"),
                    Timestamp("2000-01-02 02:00:00+0200", tz="UTC+02:00"),
                    NaT,
                ],
                id="ISO8601, non-UTC",
            ),
            pytest.param(
                "%Y-%d-%m %H:%M:%S%z",
                [
                    Timestamp("2000-01-01 09:00:00+0100", tz="UTC+01:00"),
                    Timestamp("2000-02-01 02:00:00+0200", tz="UTC+02:00"),
                    NaT,
                ],
                id="non-ISO8601, non-UTC",
            ),
        ],
    )
    def test_to_datetime_mixed_offsets_with_none_tz_utc_false_removed(
        self, fmt, expected
    ):
        # [FUNCTION DOC]
        # Issue 50071, GH#57275. Parametrized over ISO and non-ISO formats. Strings with +01:00 and +02:00 offsets plus None, with utc false, raise a value error "Mixed timezones detected. Pass utc=True in to_datetime".
        # KNOWN ISSUE: the expected parameter is accepted but never used in the body.
        # https://github.com/pandas-dev/pandas/issues/50071
        # GH#57275
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"

        with pytest.raises(ValueError, match=msg):
            to_datetime(
                ["2000-01-01 09:00:00+01:00", "2000-01-02 02:00:00+02:00", None],
                format=fmt,
                utc=False,
            )

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            pytest.param(
                "%Y-%m-%d %H:%M:%S%z",
                DatetimeIndex(
                    ["2000-01-01 08:00:00+00:00", "2000-01-02 00:00:00+00:00", "NaT"],
                    dtype="datetime64[us, UTC]",
                ),
                id="ISO8601, UTC",
            ),
            pytest.param(
                "%Y-%d-%m %H:%M:%S%z",
                DatetimeIndex(
                    ["2000-01-01 08:00:00+00:00", "2000-02-01 00:00:00+00:00", "NaT"],
                    dtype="datetime64[us, UTC]",
                ),
                id="non-ISO8601, UTC",
            ),
        ],
    )
    def test_to_datetime_mixed_offsets_with_none(self, fmt, expected):
        # [FUNCTION DOC]
        # Issue 50071. Same input with utc true: results are UTC-converted values (08:00 on January 1, 00:00 on the 2nd or February 1st depending on day/month order) and NaT, as a microsecond UTC index.
        # https://github.com/pandas-dev/pandas/issues/50071
        result = to_datetime(
            ["2000-01-01 09:00:00+01:00", "2000-01-02 02:00:00+02:00", None],
            format=fmt,
            utc=True,
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt",
        ["%Y-%d-%m %H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"],
        ids=["non-ISO8601 format", "ISO8601 format"],
    )
    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(
                ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00-07:00"],
                id="all tz-aware, mixed timezones, without utc",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "constructor",
        [Timestamp, lambda x: Timestamp(x).to_pydatetime()],
    )
    def test_to_datetime_mixed_datetime_and_string_with_format_raises(
        self, fmt, args, constructor
    ):
        # [FUNCTION DOC]
        # Issue 49298. Parametrized over formats and a Timestamp/datetime constructor. Two inputs with different offsets (-08:00 and -07:00) and utc false raise a value error that they cannot be converted to datetime64 unless utc is true.
        # https://github.com/pandas-dev/pandas/issues/49298
        # note: ISO8601 formats go down a fastpath, so we need to check both
        # an ISO8601 format and a non-ISO8601 one
        ts1 = constructor(args[0])
        ts2 = constructor(args[1])
        with pytest.raises(
            ValueError, match="cannot be converted to datetime64 unless utc=True"
        ):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    def test_to_datetime_np_str(self):
        # [FUNCTION DOC]
        # GH#32264, GH#48969. A NumPy string scalar with an offset parses to a UTC Timestamp, whether passed alone, as an element of a list (index result), or in a Series (series result).
        # GH#32264
        # GH#48969
        value = np.str_("2019-02-04 10:18:46.297000+0000")

        ser = Series([value])

        exp = Timestamp("2019-02-04 10:18:46.297000", tz="UTC")

        assert to_datetime(value) == exp
        assert to_datetime(ser.iloc[0]) == exp

        res = to_datetime([value])
        expected = Index([exp])
        tm.assert_index_equal(res, expected)

        res = to_datetime(ser)
        expected = Series(expected)
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize(
        "s, _format, dt",
        [
            ["2015-1-1", "%G-%V-%u", datetime(2014, 12, 29, 0, 0)],
            ["2015-1-4", "%G-%V-%u", datetime(2015, 1, 1, 0, 0)],
            ["2015-1-7", "%G-%V-%u", datetime(2015, 1, 4, 0, 0)],
            ["2024-52-1", "%G-%V-%u", datetime(2024, 12, 23, 0, 0)],
            ["2024-52-7", "%G-%V-%u", datetime(2024, 12, 29, 0, 0)],
            ["2025-1-1", "%G-%V-%u", datetime(2024, 12, 30, 0, 0)],
            ["2020-53-1", "%G-%V-%u", datetime(2020, 12, 28, 0, 0)],
        ],
    )
    def test_to_datetime_iso_week_year_format(self, s, _format, dt):
        # [FUNCTION DOC]
        # GH#16607. Parametrized over (string, format, expected datetime) using ISO year, ISO week and ISO weekday directives, including year boundaries (week 1 may begin in December, week 53 exists in 2020). The scalar parse equals the expected datetime.
        # See GH#16607
        assert to_datetime(s, format=_format) == dt

    @pytest.mark.parametrize(
        "msg, s, _format",
        [
            [
                "Week 53 does not exist in ISO year 2024",
                "2024 53 1",
                "%G %V %u",
            ],
            [
                "Week 53 does not exist in ISO year 2023",
                "2023 53 1",
                "%G %V %u",
            ],
        ],
    )
    def test_invalid_iso_week_53(self, msg, s, _format):
        # [FUNCTION DOC]
        # GH#60885. Parametrized over (message, string, format) for years 2024 and 2023 asking for week 53. Parsing raises a value error "Week 53 does not exist in ISO year ...".
        # See GH#60885
        with pytest.raises(ValueError, match=msg):
            to_datetime(s, format=_format)

    @pytest.mark.parametrize(
        "msg, s, _format",
        [
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 50",
                "%Y %V",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 51",
                "%G %V",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 Monday",
                "%G %A",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 Mon",
                "%G %a",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 6",
                "%G %w",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 6",
                "%G %u",
            ],
            [
                "ISO year directive '%G' must be used with the ISO week directive "
                "'%V' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "2051",
                "%G",
            ],
            [
                "Day of the year directive '%j' is not compatible with ISO year "
                "directive '%G'. Use '%Y' instead.",
                "1999 51 6 256",
                "%G %V %u %j",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 Sunday",
                "%Y %V %A",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 Sun",
                "%Y %V %a",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 1",
                "%Y %V %w",
            ],
            [
                "ISO week directive '%V' is incompatible with the year directive "
                "'%Y'. Use the ISO year '%G' instead.",
                "1999 51 1",
                "%Y %V %u",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "20",
                "%V",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 51 Sunday",
                "%V %A",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 51 Sun",
                "%V %a",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 51 1",
                "%V %w",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "1999 51 1",
                "%V %u",
            ],
            [
                "Day of the year directive '%j' is not compatible with ISO year "
                "directive '%G'. Use '%Y' instead.",
                "1999 50",
                "%G %j",
            ],
            [
                "ISO week directive '%V' must be used with the ISO year directive "
                "'%G' and a weekday directive '%A', '%a', '%w', or '%u'.",
                "20 Monday",
                "%V %A",
            ],
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    def test_error_iso_week_year(self, msg, s, _format, errors):
        # [FUNCTION DOC]
        # GH#16607, GH#50308. Parametrized over (message, string, format) for invalid combinations of ISO year, ISO week, ordinary year, weekday and day-of-year directives, and over errors raise or coerce. Parsing raises a value error with the matching explanation (even in coerce mode). Only executed when the locale is not Chinese or Italian UTF-8, because the messages and test data are English.
        # See GH#16607, GH#50308
        # This test checks for errors thrown when giving the wrong format
        # However, as discussed on PR#25541, overriding the locale
        # causes a different error to be thrown due to the format being
        # locale specific, but the test data is in english.
        # Therefore, the tests only run when locale is not overwritten,
        # as a sort of solution to this problem.
        if locale.getlocale() != ("zh_CN", "UTF-8") and locale.getlocale() != (
            "it_IT",
            "UTF-8",
        ):
            with pytest.raises(ValueError, match=msg):
                to_datetime(s, format=_format, errors=errors)

    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_to_datetime_dtarr(self, tz):
        # [FUNCTION DOC]
        # Parametrized over naive and US/Central. Passing a datetime array object returns the very same object (identity).
        # DatetimeArray
        dti = date_range("1965-04-03", periods=19, freq="2W", tz=tz)
        arr = dti._data

        result = to_datetime(arr)
        assert result is arr

    # Doesn't work on Windows since tzpath not set correctly
    @td.skip_if_windows
    @pytest.mark.parametrize("utc", [True, False])
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_to_datetime_arrow(self, tz, utc, index_or_series):
        # [FUNCTION DOC]
        # Skipped on Windows (time zone path issue) and without pyarrow. Parametrized over time zone, utc flag and container (index or series). Casts a datetime container to a pyarrow timestamp dtype with the time zone and converts. The result equals the original converted and cast to the pyarrow dtype (in UTC when utc is requested). When utc is false and the container is an index, the result is the very same object.
        pa = pytest.importorskip("pyarrow")

        dti = date_range("1965-04-03", periods=19, freq="2W", tz=tz)
        dti = index_or_series(dti)

        dti_arrow = dti.astype(pd.ArrowDtype(pa.timestamp(unit="ns", tz=tz)))

        result = to_datetime(dti_arrow, utc=utc)
        expected = to_datetime(dti, utc=utc).astype(
            pd.ArrowDtype(pa.timestamp(unit="ns", tz=tz if not utc else "UTC"))
        )
        if not utc and index_or_series is not Series:
            # Doesn't hold for utc=True, since that will astype
            # to_datetime also returns a new object for series
            assert result is dti_arrow
        if index_or_series is Series:
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result, expected)

    def test_to_datetime_pydatetime(self):
        # [FUNCTION DOC]
        # A Python datetime converts to an equal value.
        actual = to_datetime(datetime(2008, 1, 15))
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_YYYYMMDD(self):
        # [FUNCTION DOC]
        # The string "20080115" parses to 2008-01-15 without a format.
        actual = to_datetime("20080115")
        assert actual == datetime(2008, 1, 15)

    @td.skip_if_windows  # `tm.set_timezone` does not work in windows
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_to_datetime_now(self):
        # [FUNCTION DOC]
        # See GH#18666. Skipped on Windows and WASM. Sets the process time zone to US/Eastern for the duration. Converting "now" (scalar and in a list) is within 10 seconds of the current time (comparing internal integers) and the results are naive (no time zone).
        # See GH#18666
        with tm.set_timezone("US/Eastern"):
            # GH#18705
            now = Timestamp("now")
            pdnow = to_datetime("now")
            pdnow2 = to_datetime(["now"])[0]

            # These should all be equal with infinite perf; this gives
            # a generous margin of 10 seconds
            assert abs(pdnow._value - now._value) < 1e10
            assert abs(pdnow2._value - now._value) < 1e10

            assert pdnow.tzinfo is None
            assert pdnow2.tzinfo is None

    @td.skip_if_windows  # `tm.set_timezone` does not work on Windows
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    @pytest.mark.parametrize("tz", ["Pacific/Auckland", "US/Samoa"])
    def test_to_datetime_today(self, tz):
        # [FUNCTION DOC]
        # See GH#18666. Skipped on Windows and WASM. Parametrized over a zone far ahead of UTC and one far behind, set as the process zone. Converting "today" (scalar and list) normalizes to the same day as the NumPy today value, matches the Timestamp today values within 10 seconds, and is naive. Documents that the test cannot detect the regression during a short window when all zones share a day.
        # See GH#18666
        # Test with one timezone far ahead of UTC and another far behind, so
        # one of these will _almost_ always be in a different day from UTC.
        # Unfortunately this test between 12 and 1 AM Samoa time
        # this both of these timezones _and_ UTC will all be in the same day,
        # so this test will not detect the regression introduced in #18666.
        with tm.set_timezone(tz):
            nptoday = np.datetime64("today").astype("datetime64[us]").astype(np.int64)
            pdtoday = to_datetime("today")
            pdtoday2 = to_datetime(["today"])[0]

            tstoday = Timestamp("today")
            tstoday2 = Timestamp.today()

            # These should all be equal with infinite perf; this gives
            # a generous margin of 10 seconds
            assert abs(pdtoday.normalize()._value - nptoday) < 1e10
            assert abs(pdtoday2.normalize()._value - nptoday) < 1e10
            assert abs(pdtoday._value - tstoday._value) < 1e10
            assert abs(pdtoday._value - tstoday2._value) < 1e10

            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

    @pytest.mark.parametrize("arg", ["now", "today"])
    def test_to_datetime_today_now_unicode_bytes(self, arg):
        # [FUNCTION DOC]
        # Parametrized over "now" and "today". Passing a list holding the keyword must not raise (checks text/bytes handling of the keywords).
        to_datetime([arg])

    @pytest.mark.filterwarnings(
        "ignore:Timestamp.utcnow is deprecated:DeprecationWarning"
    )
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    @pytest.mark.parametrize(
        "format, expected_ds",
        [
            ("%Y-%m-%d %H:%M:%S%z", "2020-01-03"),
            ("%Y-%d-%m %H:%M:%S%z", "2020-03-01"),
            (None, "2020-01-03"),
        ],
    )
    @pytest.mark.parametrize(
        "string, attribute",
        [
            ("now", "utcnow"),
            ("today", "today"),
        ],
    )
    def test_to_datetime_now_with_format(self, format, expected_ds, string, attribute):
        # [FUNCTION DOC]
        # Issue 50359. Ignores a deprecation warning from the utcnow function; skipped on WASM. Parametrized over format and expected date text, and over the keyword "now" or "today". Converting a UTC string and the keyword, with a format and utc true, gives a second-resolution UTC index; the maximum difference of the expected index minus the result is under one second.
        # https://github.com/pandas-dev/pandas/issues/50359
        result = to_datetime(["2020-01-03 00:00:00Z", string], format=format, utc=True)
        expected = DatetimeIndex(
            [expected_ds, getattr(Timestamp, attribute)()], dtype="datetime64[s, UTC]"
        )
        assert (expected - result).max().total_seconds() < 1

    @pytest.mark.parametrize(
        "dt", [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    )
    def test_to_datetime_dt64s(self, cache, dt):
        # [FUNCTION DOC]
        # Parametrized over two NumPy day-resolution datetimes and the cache fixture. The conversion equals the Timestamp constructed from the same value.
        assert to_datetime(dt, cache=cache) == Timestamp(dt)

    @pytest.mark.parametrize(
        "arg, format",
        [
            ("2001-01-01", "%Y-%m-%d"),
            ("01-01-2001", "%d-%m-%Y"),
        ],
    )
    def test_to_datetime_dt64s_and_str(self, arg, format):
        # [FUNCTION DOC]
        # Issue 50036. Parametrized over a string with its format. A list with the string and a NumPy datetime converts to the two expected dates.
        # https://github.com/pandas-dev/pandas/issues/50036
        result = to_datetime([arg, np.datetime64("2020-01-01")], format=format)
        expected = DatetimeIndex(["2001-01-01", "2020-01-01"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "dt", [np.datetime64("1000-01-01"), np.datetime64("5000-01-02")]
    )
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    def test_to_datetime_dt64s_out_of_ns_bounds(self, cache, dt, errors):
        # [FUNCTION DOC]
        # GH#50369. Parametrized over datetimes in years 1000 and 5000, both errors modes and the cache fixture. The conversion (and the Timestamp constructor) gives a Timestamp with second resolution (the nearest supported unit) whose NumPy value equals the input.
        # GH#50369 We cast to the nearest supported reso, i.e. "s"
        ts = to_datetime(dt, errors=errors, cache=cache)
        assert isinstance(ts, Timestamp)
        assert ts.unit == "s"
        assert ts.asm8 == dt

        ts = Timestamp(dt)
        assert ts.unit == "s"
        assert ts.asm8 == dt

    def test_to_datetime_dt64d_out_of_bounds(self, cache):
        # [FUNCTION DOC]
        # The largest 64-bit integer as a day-resolution NumPy datetime raises OutOfBoundsDatetime ("Out of bounds second timestamp") in the Timestamp constructor and in raise mode; coerce mode returns NaT.
        dt64 = np.datetime64(np.iinfo(np.int64).max, "D")

        msg = "Out of bounds second timestamp: 25252734927768524-07-27"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(dt64)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt64, errors="raise", cache=cache)

        assert to_datetime(dt64, errors="coerce", cache=cache) is NaT

    @pytest.mark.parametrize("unit", ["s", "D"])
    def test_to_datetime_array_of_dt64s(self, cache, unit):
        # [FUNCTION DOC]
        # Issue 31491. Parametrized over second and day units and the cache fixture. A list of sixty NumPy datetimes (enough to trigger the cache, which needs at least 50) converts to a second-resolution index equal to the per-element Timestamp NumPy values. Then appends an out-of-bounds datetime (year 9999) to the list: raise mode does not raise (GH#51978) and coerce mode returns an index equal to the array cast to second resolution (the out-of-nanosecond-range value is kept).
        # https://github.com/pandas-dev/pandas/issues/31491
        # Need at least 50 to ensure cache is used.
        dts = [
            np.datetime64("2000-01-01", unit),
            np.datetime64("2000-01-02", unit),
        ] * 30
        # Assuming all datetimes are in bounds, to_datetime() returns
        # an array that is equal to Timestamp() parsing
        result = to_datetime(dts, cache=cache)
        expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype="M8[s]")

        tm.assert_index_equal(result, expected)

        # A list of datetimes where the last one is out of bounds
        dts_with_oob = [*dts, np.datetime64("9999-01-01")]

        # As of GH#51978 we do not raise in this case
        to_datetime(dts_with_oob, errors="raise")

        result = to_datetime(dts_with_oob, errors="coerce", cache=cache)
        expected = DatetimeIndex(np.array(dts_with_oob, dtype="M8[s]"))
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz(self, cache):
        # [FUNCTION DOC]
        # Issue 8260. A list of two Timestamps with the same US/Pacific zone converts to a datetime index with that zone and the same wall times.
        # xref 8260
        # uniform returns a DatetimeIndex
        arr = [
            Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
            Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific"),
        ]
        result = to_datetime(arr, cache=cache)
        expected = DatetimeIndex(
            ["2013-01-01 13:00:00", "2013-01-02 14:00:00"], tz="US/Pacific"
        )
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz_mixed(self, cache):
        # [FUNCTION DOC]
        # Issue 50585. Two Timestamps in different zones (US/Pacific and US/Eastern) raise a value error saying a tz-aware datetime cannot be converted unless utc is true; in coerce mode the first keeps its zone and the second becomes NaT (microsecond resolution, Pacific zone dtype).
        # mixed tzs will raise if errors='raise'
        # https://github.com/pandas-dev/pandas/issues/50585
        arr = [
            Timestamp("2013-01-01 13:00:00", tz="US/Pacific"),
            Timestamp("2013-01-02 14:00:00", tz="US/Eastern"),
        ]
        msg = (
            "Tz-aware datetime.datetime cannot be "
            "converted to datetime64 unless utc=True"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

        result = to_datetime(arr, cache=cache, errors="coerce")
        expected = DatetimeIndex(
            ["2013-01-01 13:00:00-08:00", "NaT"], dtype="datetime64[us, US/Pacific]"
        )
        tm.assert_index_equal(result, expected)

    def test_to_datetime_different_offsets_removed(self, cache):
        # [FUNCTION DOC]
        # GH-26097, GH#57275. Ten strings of two different offsets (+0400, +0500) raise a value error "Mixed timezones detected. Pass utc=True in to_datetime" (inspired by an asv benchmark).
        # inspired by asv timeseries.ToDatetimeNONISO8601 benchmark
        # see GH-26097 for more
        # GH#57275
        ts_string_1 = "March 1, 2018 12:00:00+0400"
        ts_string_2 = "March 1, 2018 12:00:00+0500"
        arr = [ts_string_1] * 5 + [ts_string_2] * 5
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

    def test_to_datetime_tz_pytz(self, cache):
        # [FUNCTION DOC]
        # see gh-8260. Skipped without pytz. A NumPy object array of two pytz-localized US/Eastern datetimes (winter and summer) converts with utc true to UTC times 08:00 and 07:00, as a microsecond UTC index with no frequency.
        # see gh-8260
        pytz = pytest.importorskip("pytz")
        us_eastern = pytz.timezone("US/Eastern")
        arr = np.array(
            [
                us_eastern.localize(
                    datetime(year=2000, month=1, day=1, hour=3, minute=0)
                ),
                us_eastern.localize(
                    datetime(year=2000, month=6, day=1, hour=3, minute=0)
                ),
            ],
            dtype=object,
        )
        result = to_datetime(arr, utc=True, cache=cache)
        expected = DatetimeIndex(
            ["2000-01-01 08:00:00+00:00", "2000-06-01 07:00:00+00:00"],
            dtype="datetime64[us, UTC]",
            freq=None,
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "init_constructor, end_constructor",
        [
            (Index, DatetimeIndex),
            (list, DatetimeIndex),
            (np.array, DatetimeIndex),
            (Series, Series),
        ],
    )
    def test_to_datetime_utc_true(self, cache, init_constructor, end_constructor):
        # [FUNCTION DOC]
        # See gh-11934 and gh-6415. Parametrized over input container (Index, list, array, Series) and the expected output container (datetime index, or Series for Series input). Compact date-time strings parsed with the format and utc true give UTC-aware timestamps in the matching container.
        # See gh-11934 & gh-6415
        data = ["20100102 121314", "20100102 121315"]
        expected_data = [
            Timestamp("2010-01-02 12:13:14", tz="utc"),
            Timestamp("2010-01-02 12:13:15", tz="utc"),
        ]

        result = to_datetime(
            init_constructor(data), format="%Y%m%d %H%M%S", utc=True, cache=cache
        )
        expected = end_constructor(expected_data)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "scalar, expected",
        [
            ["20100102 121314", Timestamp("2010-01-02 12:13:14", tz="utc")],
            ["20100102 121315", Timestamp("2010-01-02 12:13:15", tz="utc")],
        ],
    )
    def test_to_datetime_utc_true_scalar(self, cache, scalar, expected):
        # [FUNCTION DOC]
        # Parametrized over two compact date-time strings; the scalar parse with format and utc true equals the UTC-aware Timestamp.
        # Test scalar case as well
        result = to_datetime(scalar, format="%Y%m%d %H%M%S", utc=True, cache=cache)
        assert result == expected

    def test_to_datetime_utc_true_with_series_single_value(self, cache):
        # [FUNCTION DOC]
        # GH 15760. A Series holding the float 1.5e18 (nanoseconds since epoch) converts with utc true to a Series with the equal UTC Timestamp.
        # GH 15760 UTC=True with Series
        ts = 1.5e18
        result = to_datetime(Series([ts]), utc=True, cache=cache)
        expected = Series([Timestamp(ts, tz="utc")])
        tm.assert_series_equal(result, expected)

    def test_to_datetime_utc_true_with_series_tzaware_string(self, cache):
        # [FUNCTION DOC]
        # A Series of three strings with a -01:00 offset converts with utc true to a Series of three UTC-aware timestamps one hour later.
        ts = "2013-01-01 00:00:00-01:00"
        expected_ts = "2013-01-01 01:00:00"
        data = Series([ts] * 3)
        result = to_datetime(data, utc=True, cache=cache)
        expected = Series([Timestamp(expected_ts, tz="utc")] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "date, dtype",
        [
            ("2013-01-01 01:00:00", "datetime64[ns]"),
            ("2013-01-01 01:00:00", "datetime64[ns, UTC]"),
        ],
    )
    def test_to_datetime_utc_true_with_series_datetime_ns(self, cache, date, dtype):
        # [FUNCTION DOC]
        # Parametrized over a date string with dtype naive nanosecond or UTC nanosecond. A Series of that dtype converts with utc true to a nanosecond UTC Series with the expected single value.
        expected = Series(
            [Timestamp("2013-01-01 01:00:00", tz="UTC")], dtype="M8[ns, UTC]"
        )
        result = to_datetime(Series([date], dtype=dtype), utc=True, cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_tz_psycopg2(self, request, cache):
        # [FUNCTION DOC]
        # Issue 8260. Skipped without psycopg2. Takes the request and cache fixtures. Datetimes with fixed-offset zones from psycopg2 (minus 300 and minus 240 minutes) in an object array convert in coerce mode with utc true to UTC times 08:00 and 07:00 (microsecond UTC index, no frequency). A localized index in a psycopg2 zone is not nanosecond dtype; converting it returns an equal index, and with utc true shifts it to 13:00 UTC.
        # xref 8260
        psycopg2_tz = pytest.importorskip("psycopg2.tz")

        # misc cases
        tz1 = psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)
        tz2 = psycopg2_tz.FixedOffsetTimezone(offset=-240, name=None)
        arr = np.array(
            [
                datetime(2000, 1, 1, 3, 0, tzinfo=tz1),
                datetime(2000, 6, 1, 3, 0, tzinfo=tz2),
            ],
            dtype=object,
        )

        result = to_datetime(arr, errors="coerce", utc=True, cache=cache)
        expected = DatetimeIndex(
            ["2000-01-01 08:00:00+00:00", "2000-06-01 07:00:00+00:00"],
            dtype="datetime64[us, UTC]",
            freq=None,
        )
        tm.assert_index_equal(result, expected)

        # dtype coercion
        i = DatetimeIndex(
            ["2000-01-01 08:00:00"],
            tz=psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None),
        ).as_unit("us")
        assert not is_datetime64_ns_dtype(i)

        # tz coercion
        result = to_datetime(i, errors="coerce", cache=cache)
        tm.assert_index_equal(result, i)

        result = to_datetime(i, errors="coerce", utc=True, cache=cache)
        expected = DatetimeIndex(["2000-01-01 13:00:00"], dtype="datetime64[us, UTC]")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("arg", [True, False])
    def test_datetime_bool(self, cache, arg):
        # [FUNCTION DOC]
        # GH13176. Parametrized over true and false. A boolean scalar raises a type error (dtype bool cannot be converted to datetime64 nanoseconds); in coerce mode it gives NaT.
        # GH13176
        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)
        assert to_datetime(arg, errors="coerce", cache=cache) is NaT

    def test_datetime_bool_arrays_mixed(self, cache):
        # [FUNCTION DOC]
        # Mixed lists: a list with False and a datetime raises a type error that the type of the cache fixture value is not convertible; a list with a date string and True raises a value error that the text "True" does not match the inferred format; in coerce mode, zero, False, NaT and 0.0 give the epoch timestamp for numeric zeros and NaT for the others.
        msg = f"{type(cache)} is not convertible to datetime"
        with pytest.raises(TypeError, match=msg):
            to_datetime([False, datetime.today()], cache=cache)
        with pytest.raises(
            ValueError,
            match=(
                r'^time data "True" doesn\'t match format "%Y%m%d". '
                f"{PARSING_ERR_MSG}$"
            ),
        ):
            to_datetime(["20130101", True], cache=cache)
        tm.assert_index_equal(
            to_datetime([0, False, NaT, 0.0], errors="coerce", cache=cache),
            DatetimeIndex(
                [to_datetime(0, cache=cache), NaT, NaT, to_datetime(0, cache=cache)]
            ),
        )

    @pytest.mark.parametrize("arg", [bool, to_datetime])
    def test_datetime_invalid_datatype(self, arg):
        # [FUNCTION DOC]
        # GH13176. Parametrized over the bool class and the conversion function itself. Passing them raises a type error "is not convertible to datetime".
        # GH13176
        msg = "is not convertible to datetime"
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)

    @pytest.mark.parametrize("errors", ["coerce", "raise"])
    def test_invalid_format_raises(self, errors):
        # [FUNCTION DOC]
        # Issue 50255. Parametrized over errors raise and coerce. A malformed format with misplaced percent signs raises a value error about a bad directive; the message depends on the Python version (it names the colon from 3.15 on, otherwise ':' ), in both error modes.
        # https://github.com/pandas-dev/pandas/issues/50255
        if PY315:
            msg = r"':M' is a bad directive in format 'H%:M%:S%"
        else:
            msg = "':' is a bad directive in format 'H%:M%:S%"

        with pytest.raises(ValueError, match=msg):
            to_datetime(["00:00:00"], format="H%:M%:S%", errors=errors)

    @pytest.mark.parametrize("value", ["a", "00:01:99"])
    @pytest.mark.parametrize("format", [None, "%H:%M:%S"])
    def test_datetime_invalid_scalar(self, value, format):
        # [FUNCTION DOC]
        # GH24763. Parametrized over an invalid string and an out-of-range time, with and without a format. Coerce mode returns NaT; raise mode raises a value error matching one of several messages (not matching the format, not likely a datetime, unconverted data remains, second out of range).
        # GH24763
        res = to_datetime(value, errors="coerce", format=format)
        assert res is NaT

        msg = "|".join(
            [
                r'^time data "a" doesn\'t match format "%H:%M:%S". '
                f"{PARSING_ERR_MSG}$",
                r'^Given date string "a" not likely a datetime$',
                r'^unconverted data remains when parsing with format "%H:%M:%S": "9". '
                f"{PARSING_ERR_MSG}$",
                rf"^second must be in 0..59{NOT_99}: 00:01:99$",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(value, errors="raise", format=format)

    @pytest.mark.parametrize("value", ["3000/12/11 00:00:00"])
    @pytest.mark.parametrize("format", [None, "%H:%M:%S"])
    def test_datetime_outofbounds_scalar(self, value, format):
        # [FUNCTION DOC]
        # GH24763. Parametrized over format none or a time format with a date value beyond the nanosecond bounds. Without a format the string parses to the equal Timestamp in both error modes; with a time-only format, coerce mode returns NaT and raise mode raises a value error that the data does not match the format.
        # GH24763
        res = to_datetime(value, errors="coerce", format=format)
        if format is None:
            assert isinstance(res, Timestamp)
            assert res == Timestamp(value)
        else:
            assert res is NaT

        if format is not None:
            msg = r'^time data ".*" doesn\'t match format ".*"'
            with pytest.raises(ValueError, match=msg):
                to_datetime(value, errors="raise", format=format)
        else:
            res = to_datetime(value, errors="raise", format=format)
            assert isinstance(res, Timestamp)
            assert res == Timestamp(value)

    @pytest.mark.parametrize(
        ("values"), [(["a"]), (["00:01:99"]), (["a", "b", "99:00:00"])]
    )
    @pytest.mark.parametrize("format", [(None), ("%H:%M:%S")])
    def test_datetime_invalid_index(self, values, format):
        # [FUNCTION DOC]
        # GH24763. Parametrized over lists of bad strings and formats. In coerce mode returns an all-NaT index (a could-not-infer-format user warning is expected only when no format is given and the list has more than one element, other warnings are tolerated); in raise mode raises a value error matching one of several messages.
        # GH24763
        # Not great to have logic in tests, but this one's hard to
        # parametrise over
        if format is None and len(values) > 1:
            warn = UserWarning
        else:
            warn = None

        with tm.assert_produces_warning(
            warn, match="Could not infer format", raise_on_extra_warnings=False
        ):
            res = to_datetime(values, errors="coerce", format=format)
        tm.assert_index_equal(res, DatetimeIndex([NaT] * len(values)))

        msg = "|".join(
            [
                r'^Given date string "a" not likely a datetime$',
                r'^time data "a" doesn\'t match format "%H:%M:%S". '
                f"{PARSING_ERR_MSG}$",
                r'^unconverted data remains when parsing with format "%H:%M:%S": "9". '
                f"{PARSING_ERR_MSG}$",
                rf"^second must be in 0..59{NOT_99}: 00:01:99$",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                warn, match="Could not infer format", raise_on_extra_warnings=False
            ):
                to_datetime(values, errors="raise", format=format)

    @pytest.mark.parametrize("utc", [True, None])
    @pytest.mark.parametrize("format", ["%Y%m%d %H:%M:%S", None])
    @pytest.mark.parametrize("constructor", [list, tuple, np.array, Index, deque])
    def test_to_datetime_cache(self, utc, format, constructor):
        # [FUNCTION DOC]
        # Parametrized over utc (true or none), format (explicit or none) and container type (list, tuple, array, index, deque). With 100000 identical date strings, converting with caching on equals converting with caching off.
        date = "20130101 00:00:00"
        test_dates = [date] * 10**5
        data = constructor(test_dates)

        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)

        tm.assert_index_equal(result, expected)

    def test_to_datetime_from_deque(self):
        # [FUNCTION DOC]
        # GH 29403. A deque of 51 identical Timestamps converts to the same index as a list of them.
        # GH 29403
        result = to_datetime(deque([Timestamp("2010-06-02 09:30:00")] * 51))
        expected = to_datetime([Timestamp("2010-06-02 09:30:00")] * 51)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("utc", [True, None])
    @pytest.mark.parametrize("format", ["%Y%m%d %H:%M:%S", None])
    def test_to_datetime_cache_series(self, utc, format):
        # [FUNCTION DOC]
        # Parametrized over utc and format. A Series of 100000 identical strings converts with caching on equal to converting with caching off.
        date = "20130101 00:00:00"
        test_dates = [date] * 10**5
        data = Series(test_dates)
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_cache_scalar(self):
        # [FUNCTION DOC]
        # A scalar string converts with caching enabled to the equal Timestamp.
        date = "20130101 00:00:00"
        result = to_datetime(date, cache=True)
        expected = Timestamp("20130101 00:00:00")
        assert result == expected

    @pytest.mark.parametrize(
        "datetimelikes,expected_values,exp_unit",
        (
            (
                (None, np.nan) + (NaT,) * start_caching_at,
                (NaT,) * (start_caching_at + 2),
                "s",
            ),
            (
                (None, Timestamp("2012-07-26").as_unit("s"))
                + (NaT,) * start_caching_at,
                (NaT, Timestamp("2012-07-26").as_unit("s")) + (NaT,) * start_caching_at,
                "s",
            ),
            (
                (None,)
                + (NaT,) * start_caching_at
                + ("2012 July 26", Timestamp("2012-07-26")),
                (NaT,) * (start_caching_at + 1)
                + (Timestamp("2012-07-26"), Timestamp("2012-07-26")),
                "us",
            ),
        ),
    )
    def test_convert_object_to_datetime_with_cache(
        self, datetimelikes, expected_values, exp_unit
    ):
        # [FUNCTION DOC]
        # GH#39882. Parametrized over input tuples (all missing values; missing with a Timestamp; missing, text and Timestamp, each long enough to trigger caching), expected values and the expected resolution (seconds or microseconds). An object Series converted in coerce mode equals a Series of the expected values with that datetime dtype.
        # GH#39882
        ser = Series(
            datetimelikes,
            dtype="object",
        )
        result_series = to_datetime(ser, errors="coerce")
        expected_series = Series(
            expected_values,
            dtype=f"datetime64[{exp_unit}]",
        )
        tm.assert_series_equal(result_series, expected_series)

    @pytest.mark.parametrize(
        "input",
        [
            Series([NaT] * 20 + [None] * 20, dtype="object"),
            Series([NaT] * 60 + [None] * 60, dtype="object"),
            Series([None] * 20),
            Series([None] * 60),
            Series([""] * 20),
            Series([""] * 60),
            Series([pd.NA] * 20),
            Series([pd.NA] * 60),
            Series([np.nan] * 20),
            Series([np.nan] * 60),
        ],
    )
    def test_to_datetime_converts_null_like_to_nat(self, cache, input):
        # [FUNCTION DOC]
        # GH35888. Parametrized over object Series of lengths 20 and 60 filled with NaT, None, empty strings, NA or NaN (some mixes). Conversion with the cache fixture gives an all-NaT second-resolution Series.
        # GH35888
        expected = Series([NaT] * len(input), dtype="M8[s]")
        result = to_datetime(input, cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "date, format",
        [
            ("2017-20", "%Y-%W"),
            ("20 Sunday", "%W %A"),
            ("20 Sun", "%W %a"),
            ("2017-21", "%Y-%U"),
            ("20 Sunday", "%U %A"),
            ("20 Sun", "%U %a"),
        ],
    )
    def test_week_without_day_and_calendar_year(self, date, format):
        # [FUNCTION DOC]
        # GH16774. Parametrized over (string, format) using week-of-year directives (Monday and Sunday based) without both a weekday and a year. Parsing raises a value error "Cannot use '%W' or '%U' without day and year".
        # GH16774

        msg = "Cannot use '%W' or '%U' without day and year"
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, format=format)

    def test_to_datetime_coerce(self):
        # [FUNCTION DOC]
        # GH#26122, GH#57275. Two strings with different offsets plus an invalid date, in coerce mode, still raise a value error "Mixed timezones detected. Pass utc=True in to_datetime" (coerce does not suppress this error).
        # GH#26122, GH#57275
        ts_strings = [
            "March 1, 2018 12:00:00+0400",
            "March 1, 2018 12:00:00+0500",
            "20100240",
        ]
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(ts_strings, errors="coerce")

    @pytest.mark.parametrize(
        "string_arg, format",
        [("March 1, 2018", "%B %d, %Y"), ("2018-03-01", "%Y-%m-%d")],
    )
    @pytest.mark.parametrize(
        "outofbounds",
        [
            datetime(9999, 1, 1),
            date(9999, 1, 1),
            np.datetime64("9999-01-01"),
            "January 1, 9999",
            "9999-01-01",
        ],
    )
    def test_to_datetime_coerce_oob(self, string_arg, format, outofbounds):
        # [FUNCTION DOC]
        # Issue 50255. Parametrized over (string, format) and out-of-bounds (year 9999) values given as datetime, date, NumPy datetime, long text and ISO text. In coerce mode, a text value that does not match the given format becomes NaT; in the other cases the value is kept; the result is a microsecond index with the first string parsed as 2018-03-01.
        # KNOWN ISSUE: the last two branches of the expected-value computation (a datetime input, and everything else) construct identical results, so the second branch is redundant.
        # https://github.com/pandas-dev/pandas/issues/50255
        ts_strings = [string_arg, outofbounds]
        result = to_datetime(ts_strings, errors="coerce", format=format)
        if isinstance(outofbounds, str) and (
            format.startswith("%B") ^ outofbounds.startswith("J")
        ):
            # the strings don't match the given format, so they raise and we coerce
            expected = DatetimeIndex([datetime(2018, 3, 1), NaT], dtype="M8[us]")
        elif isinstance(outofbounds, datetime):
            expected = DatetimeIndex(
                [datetime(2018, 3, 1), outofbounds], dtype="M8[us]"
            )
        else:
            expected = DatetimeIndex(
                [datetime(2018, 3, 1), outofbounds], dtype="M8[us]"
            )
        tm.assert_index_equal(result, expected)

    def test_to_datetime_malformed_no_raise(self):
        # [FUNCTION DOC]
        # GH 28299, GH 48633. Two malformed strings in coerce mode give an all-NaT second-resolution index and emit a user warning "Could not infer format" (other warnings tolerated). Carries a TODO asking whether the default resolution should be second.
        # GH 28299
        # GH 48633
        ts_strings = ["200622-12-31", "111111-24-11"]
        with tm.assert_produces_warning(
            UserWarning, match="Could not infer format", raise_on_extra_warnings=False
        ):
            result = to_datetime(ts_strings, errors="coerce")
        # TODO: should Index get "s" by default here?
        exp = Index([NaT, NaT], dtype="M8[s]")
        tm.assert_index_equal(result, exp)

    def test_to_datetime_malformed_raise(self):
        # [FUNCTION DOC]
        # GH 48633. The same malformed strings in raise mode raise a value error about an invalid time zone offset in the parsed string (must be within 24 hours), and also emit the could-not-infer-format user warning.
        # GH 48633
        ts_strings = ["200622-12-31", "111111-24-11"]
        msg = (
            'Parsed string "200622-12-31" gives an invalid tzoffset, which must '
            r"be between -timedelta\(hours=24\) and timedelta\(hours=24\)"
        )
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            with tm.assert_produces_warning(
                UserWarning, match="Could not infer format"
            ):
                to_datetime(
                    ts_strings,
                    errors="raise",
                )

    def test_iso_8601_strings_with_same_offset(self):
        # [FUNCTION DOC]
        # GH 17697, 11736. A string with a +05:30 offset converts to the equal Timestamp; a list of two such strings converts to the equal index, also via the index constructor.
        # GH 17697, 11736
        ts_str = "2015-11-18 15:30:00+05:30"
        result = to_datetime(ts_str)
        expected = Timestamp(ts_str)
        assert result == expected

        expected = DatetimeIndex([Timestamp(ts_str)] * 2)
        result = to_datetime([ts_str] * 2)
        tm.assert_index_equal(result, expected)

        result = DatetimeIndex([ts_str] * 2)
        tm.assert_index_equal(result, expected)

    def test_iso_8601_strings_with_different_offsets_removed(self):
        # [FUNCTION DOC]
        # GH#17697, GH#11736, GH#50887, GH#57275. ISO strings with +05:30 and +06:30 plus NaT, with the default utc setting, raise a value error "Mixed timezones detected. Pass utc=True in to_datetime".
        # GH#17697, GH#11736, GH#50887, GH#57275
        ts_strings = ["2015-11-18 15:30:00+05:30", "2015-11-18 16:30:00+06:30", NaT]
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(ts_strings)

    def test_iso_8601_strings_with_different_offsets_utc(self):
        # [FUNCTION DOC]
        # The same inputs with utc true give two equal UTC timestamps (10:00 on 2015-11-18) and NaT, as a UTC index.
        ts_strings = ["2015-11-18 15:30:00+05:30", "2015-11-18 16:30:00+06:30", NaT]
        result = to_datetime(ts_strings, utc=True)
        expected = DatetimeIndex(
            [Timestamp(2015, 11, 18, 10), Timestamp(2015, 11, 18, 10), NaT], tz="UTC"
        )
        tm.assert_index_equal(result, expected)

    def test_mixed_offsets_with_native_datetime_utc_false_raises(self):
        # [FUNCTION DOC]
        # GH#25978, GH#57275. A Series mixing the string "nan", a naive Timestamp, ISO strings with different offsets, None, and the keywords today and now raises a value error "Mixed timezones detected. Pass utc=True in to_datetime". Also asserts the Series stores the original object references (GH#40111).
        # GH#25978, GH#57275

        vals = [
            "nan",
            Timestamp("1990-01-01"),
            "2015-03-14T16:15:14.123-08:00",
            "2019-03-04T21:56:32.620-07:00",
            None,
            "today",
            "now",
        ]
        ser = Series(vals)
        assert all(ser[i] is vals[i] for i in range(len(vals)))  # GH#40111

        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser)

    def test_non_iso_strings_with_tz_offset(self):
        # [FUNCTION DOC]
        # Two copies of a non-ISO string with a +0400 offset convert to an index of two datetimes with a fixed 240-minute offset.
        result = to_datetime(["March 1, 2018 12:00:00+0400"] * 2)
        expected = DatetimeIndex(
            [datetime(2018, 3, 1, 12, tzinfo=timezone(timedelta(minutes=240)))] * 2
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "ts, expected",
        [
            (Timestamp("2018-01-01"), Timestamp("2018-01-01", tz="UTC")),
            (
                Timestamp("2018-01-01", tz="US/Pacific"),
                Timestamp("2018-01-01 08:00", tz="UTC"),
            ),
        ],
    )
    def test_timestamp_utc_true(self, ts, expected):
        # [FUNCTION DOC]
        # GH 24415. Parametrized over a naive Timestamp and a Pacific Timestamp. Converting with utc true yields the UTC equivalent (a naive value is taken as UTC; the Pacific value shifts by 8 hours).
        # GH 24415
        result = to_datetime(ts, utc=True)
        assert result == expected

    @pytest.mark.parametrize("dt_str", ["00010101", "13000101", "30000101", "99990101"])
    def test_to_datetime_with_format_out_of_bounds(self, dt_str):
        # [FUNCTION DOC]
        # GH 9107. Parametrized over compact date strings from year 1 to year 9999. Parsing with the compact format equals the Timestamp from the standard library parser result, with the same resolution unit.
        # GH 9107
        res = to_datetime(dt_str, format="%Y%m%d")
        dtobj = datetime.strptime(dt_str, "%Y%m%d")
        expected = Timestamp(dtobj)
        assert res == expected
        assert res.unit == expected.unit

    def test_to_datetime_utc(self):
        # [FUNCTION DOC]
        # An object array holding a parsed UTC datetime converts with utc true to an index whose time zone is the standard library UTC object (identity).
        arr = np.array([parse("2012-06-13T01:39:00Z")], dtype=object)

        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

    def test_to_datetime_fixed_offset(self):
        # [FUNCTION DOC]
        # Imports a fixed-offset time zone class from the datetime index time zone tests. Three datetimes with a -07:00 fixed offset convert to an index whose time zone equals that fixed offset.
        from pandas.tests.indexes.datetimes.test_timezones import FixedOffset

        fixed_off = FixedOffset(-420, "-07:00")

        dates = [
            datetime(2000, 1, 1, tzinfo=fixed_off),
            datetime(2000, 1, 2, tzinfo=fixed_off),
            datetime(2000, 1, 3, tzinfo=fixed_off),
        ]
        result = to_datetime(dates)
        assert result.tz == fixed_off

    @pytest.mark.parametrize(
        "date",
        [
            ["2020-10-26 00:00:00+06:00", "2020-10-26 00:00:00+01:00"],
            ["2020-10-26 00:00:00+06:00", Timestamp("2018-01-01", tz="US/Pacific")],
            [
                "2020-10-26 00:00:00+06:00",
                datetime(2020, 1, 1, 18).astimezone(
                    zoneinfo.ZoneInfo("Australia/Melbourne")
                ),
            ],
        ],
    )
    def test_to_datetime_mixed_offsets_with_utc_false_removed(self, date):
        # [FUNCTION DOC]
        # GH#50887, GH#57275. Parametrized over lists mixing a +06:00 string with another offset string, a Pacific Timestamp, or a Melbourne zone datetime. With utc false, raises a value error "Mixed timezones detected. Pass utc=True in to_datetime".
        # GH#50887, GH#57275
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, utc=False)


class TestToDatetimeUnit:
    # [CLASS DOC]
    # Tests of the conversion function's unit parameter for numeric epoch values (years, months, days, seconds, milliseconds and so on), including floats, missing values, big values, overflow, and error modes.
    @pytest.mark.parametrize("unit", ["Y", "M"])
    @pytest.mark.parametrize("item", [150, float(150)])
    def test_to_datetime_month_or_year_unit_int(self, cache, unit, item, request):
        # [FUNCTION DOC]
        # GH#50870. Parametrized over unit year or month and over an integer or equal float 150. The expected result is a second-resolution index holding the Timestamp built with the same value and unit. The conversion (list, object array, numeric array, and numeric array with a NaN where the NaN becomes NaT) must equal it. The request fixture is accepted but unused.
        # GH#50870 Note we have separate tests that pd.Timestamp gets these right
        ts = Timestamp(item, unit=unit)
        dtype = "M8[s]"
        expected = DatetimeIndex([ts], dtype=dtype)

        result = to_datetime([item], unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)

        result = to_datetime(np.array([item], dtype=object), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)

        result = to_datetime(np.array([item]), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)

        # with a nan!
        result = to_datetime(np.array([item, np.nan]), unit=unit, cache=cache)
        assert result.isna()[1]
        tm.assert_index_equal(result[:1], expected.astype("M8[s]"))

    @pytest.mark.parametrize("unit", ["Y", "M"])
    def test_to_datetime_month_or_year_unit_non_round_float(self, cache, unit):
        # [FUNCTION DOC]
        # GH#50301. Parametrized over year and month units. Matches the Timestamp behavior of disallowing non-round floats with these units: raise mode raises a value error that the conversion is ambiguous (list and array); a string "1.5" raises a value error "not likely a datetime" since a string is parsed as it would be without a unit; coerce mode gives NaT for both. A round float is treated like an integer and gives the second-resolution result.
        # GH#50301
        # Match Timestamp behavior in disallowing non-round floats with
        #  Y or M unit
        msg = f"Conversion of non-round float with unit={unit} is ambiguous"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1.5], unit=unit, errors="raise")
        with pytest.raises(ValueError, match=msg):
            to_datetime(np.array([1.5]), unit=unit, errors="raise")

        msg = r"Given date string \"1.5\" not likely a datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(["1.5"], unit=unit, errors="raise")

        res = to_datetime([1.5], unit=unit, errors="coerce")
        expected = Index([NaT], dtype="M8[ns]")
        tm.assert_index_equal(res, expected)

        # In 3.0, the string "1.5" is parsed as as it would be without unit,
        #  which fails. With errors="coerce" this becomes NaT.
        res = to_datetime(["1.5"], unit=unit, errors="coerce")
        expected = to_datetime([NaT])
        tm.assert_index_equal(res, expected)

        # round floats are OK; treated like integers to give
        #  closest-to-supported unit
        res = to_datetime([1.0], unit=unit)
        expected = to_datetime([1], unit=unit).as_unit("s")
        tm.assert_index_equal(res, expected)

    def test_unit(self, cache):
        # [FUNCTION DOC]
        # GH 11758. Passing both a format and a unit raises a value error "cannot specify both format and unit".
        # GH 11758
        # test proper behavior with errors
        msg = "cannot specify both format and unit"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1], unit="D", format="%Y%m%d", cache=cache)

    def test_unit_array_mixed_nans(self, cache):
        # [FUNCTION DOC]
        # A list mixing a huge integer, 1, 1.0, the integer NaT sentinel, NaT, NaN, the string "NaT" and an empty string with unit days: coerce mode gives NaT for the huge value and missing entries and 1970-01-02 for the ones (second resolution); raise mode raises OutOfBoundsDatetime "cannot convert input ... with the unit 'D'".
        values = [11111111111111111, 1, 1.0, iNaT, NaT, np.nan, "NaT", ""]

        result = to_datetime(values, unit="D", errors="coerce", cache=cache)
        expected = DatetimeIndex(
            ["NaT", "1970-01-02", "1970-01-02", "NaT", "NaT", "NaT", "NaT", "NaT"],
            dtype="M8[s]",
        )
        tm.assert_index_equal(result, expected)

        msg = "cannot convert input 11111111111111111 with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, unit="D", errors="raise", cache=cache)

    def test_unit_array_mixed_nans_large_int(self, cache):
        # [FUNCTION DOC]
        # A list of an enormous integer and missing entries with unit seconds: coerce mode gives all NaT (second resolution); raise mode raises OutOfBoundsDatetime naming the input and the unit 's'.
        values = [1420043460000000000000000, iNaT, NaT, np.nan, "NaT"]

        result = to_datetime(values, errors="coerce", unit="s", cache=cache)
        expected = DatetimeIndex(["NaT", "NaT", "NaT", "NaT", "NaT"], dtype="M8[s]")
        tm.assert_index_equal(result, expected)

        msg = "cannot convert input 1420043460000000000000000 with the unit 's'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, errors="raise", unit="s", cache=cache)

    def test_to_datetime_invalid_str_not_out_of_bounds_valuerror(self, cache):
        # [FUNCTION DOC]
        # A non-date string with a unit raises a value error "Unknown datetime string format, unable to parse: foo" and not OutOfBoundsDatetime.
        # if we have a string, then we raise a ValueError
        # and NOT an OutOfBoundsDatetime
        msg = "Unknown datetime string format, unable to parse: foo"
        with pytest.raises(ValueError, match=msg):
            to_datetime("foo", errors="raise", unit="s", cache=cache)

    @pytest.mark.parametrize("error", ["raise", "coerce"])
    def test_unit_consistency(self, cache, error):
        # [FUNCTION DOC]
        # Parametrized over raise and coerce. The integer 11111111 with unit seconds gives the Timestamp 1970-05-09 14:25:11 in both modes.
        # consistency of conversions
        expected = Timestamp("1970-05-09 14:25:11")
        result = to_datetime(11111111, unit="s", errors=error, cache=cache)
        assert result == expected
        assert isinstance(result, Timestamp)

    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    @pytest.mark.parametrize("dtype", ["float64", "int64"])
    def test_unit_with_numeric(self, cache, errors, dtype):
        # [FUNCTION DOC]
        # GH 13180. Parametrized over errors mode and float64 or int64 arrays of epoch nanoseconds (1.434692e18, 1.432766e18). Converting without a unit gives the expected nanosecond datetime index of June 19 and May 27, 2015.
        # GH 13180
        # coercions from floats/ints are ok
        expected = DatetimeIndex(
            ["2015-06-19 05:33:20", "2015-05-27 22:33:20"], dtype="M8[ns]"
        )
        arr = np.array([1.434692e18, 1.432766e18]).astype(dtype)
        result = to_datetime(arr, errors=errors, cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "exp, arr, warning",
        [
            [
                ["NaT", "2015-06-19 05:33:20", "2015-05-27 22:33:20"],
                ["foo", 1.434692e18, 1.432766e18],
                UserWarning,
            ],
            [
                ["2015-06-19 05:33:20", "2015-05-27 22:33:20", "NaT", "NaT"],
                [1.434692e18, 1.432766e18, "foo", "NaT"],
                None,
            ],
        ],
    )
    def test_unit_with_numeric_coerce(self, cache, exp, arr, warning):
        # [FUNCTION DOC]
        # Parametrized over (expected text, input list, expected warning). A list of a string and floats (with a user warning "Could not infer format") or floats with strings (no warning) converts in coerce mode to the numeric timestamps with NaT for the unparseable entries (nanosecond resolution).
        # but we want to make sure that we are coercing
        # if we have ints/strings
        expected = DatetimeIndex(exp, dtype="M8[ns]")
        with tm.assert_produces_warning(warning, match="Could not infer format"):
            result = to_datetime(arr, errors="coerce", cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            [Timestamp("20130101"), 1.434692e18, 1.432766e18],
            [1.434692e18, 1.432766e18, Timestamp("20130101")],
        ],
    )
    def test_unit_mixed(self, cache, arr):
        # [FUNCTION DOC]
        # GH#50453, GH#49037. Parametrized over lists mixing a Timestamp and large floats (either order). Coerce mode, raise mode and the index constructor all give a nanosecond index equal to the per-element Timestamps (floats treated as nanoseconds), where earlier versions coerced to NaT or raised.
        # GH#50453 pre-2.0 with mixed numeric/datetimes and errors="coerce"
        #  the numeric entries would be coerced to NaT, was never clear exactly
        #  why.
        # mixed integers/datetimes
        expected = Index([Timestamp(x) for x in arr], dtype="M8[ns]")
        result = to_datetime(arr, errors="coerce", cache=cache)
        tm.assert_index_equal(result, expected)

        # GH#49037 pre-2.0 this raised, but it always worked with Series,
        #  was never clear why it was disallowed
        result = to_datetime(arr, errors="raise", cache=cache)
        tm.assert_index_equal(result, expected)

        result = DatetimeIndex(arr)
        tm.assert_index_equal(result, expected)

    def test_unit_rounding(self, cache):
        # [FUNCTION DOC]
        # GH 14156 and GH 20445. A float with fractional seconds (1434743731.8770001) and unit seconds gives the Timestamp 2015-06-19 19:55:31.877000093 (floating error but no premature rounding), and the Timestamp constructor agrees.
        # GH 14156 & GH 20445: argument will incur floating point errors
        # but no premature rounding
        value = 1434743731.8770001
        result = to_datetime(value, unit="s", cache=cache)
        expected = Timestamp("2015-06-19 19:55:31.877000093")
        assert result == expected

        alt = Timestamp(value, unit="s")
        assert alt == result

    @pytest.mark.parametrize("dtype", [int, float])
    def test_to_datetime_unit(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over integer and float dtypes. A Series of twenty consecutive epoch seconds converts with unit seconds to a second-resolution Series starting 2013-06-09 02:42:28.
        epoch = 1370745748
        ser = Series([epoch + t for t in range(20)]).astype(dtype)
        result = to_datetime(ser, unit="s")
        unit = "s"
        expected = Series(
            [
                Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t)
                for t in range(20)
            ],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("null", [iNaT, np.nan])
    def test_to_datetime_unit_with_nulls(self, null):
        # [FUNCTION DOC]
        # Parametrized over the integer NaT sentinel and NaN. A Series of twenty epoch seconds plus a null value converts with unit seconds to a second-resolution Series with NaT at the end.
        # KNOWN ISSUE: the in-body comment says a NaN makes the list a float array that always gets nanosecond resolution, but the test expects second resolution.
        epoch = 1370745748
        ser = Series([epoch + t for t in range(20)] + [null])
        result = to_datetime(ser, unit="s")
        # With np.nan, the list gets cast to a float64 array, which always
        #  gets ns unit.
        unit = "s"
        expected = Series(
            [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
            + [NaT],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

    def test_to_datetime_unit_fractional_seconds(self):
        # [FUNCTION DOC]
        # GH13834, GH20455. A float Series of epoch seconds with quarter-second steps and a missing sentinel, with unit seconds, rounded to milliseconds afterward, equals the expected nanosecond Series with NaT at the end (floating errors but no premature rounding).
        # GH13834
        epoch = 1370745748
        ser = Series([epoch + t for t in np.arange(0, 2, 0.25)] + [iNaT]).astype(float)
        result = to_datetime(ser, unit="s")
        expected = Series(
            [
                Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t)
                for t in np.arange(0, 2, 0.25)
            ]
            + [NaT],
            dtype="M8[ns]",
        )
        # GH20455 argument will incur floating point errors but no premature rounding
        result = result.dt.round("ms")
        tm.assert_series_equal(result, expected)

    def test_to_datetime_unit_na_values(self):
        # [FUNCTION DOC]
        # A list of 1, 2, "NaT", NaT and NaN with unit days gives 1970-01-02, 1970-01-03 and three NaT in second resolution.
        result = to_datetime([1, 2, "NaT", NaT, np.nan], unit="D")
        expected = DatetimeIndex(
            [Timestamp("1970-01-02"), Timestamp("1970-01-03")] + ["NaT"] * 3,
            dtype="M8[s]",
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("bad_val", ["foo", 111111111111111])
    def test_to_datetime_unit_invalid(self, bad_val):
        # [FUNCTION DOC]
        # Parametrized over "foo" and an out-of-range integer. With unit days, a list containing it raises a value error: "Unknown datetime string format" for the string, "cannot convert input ... with the unit 'D'" for the integer.
        if bad_val == "foo":
            msg = f"Unknown datetime string format, unable to parse: {bad_val}"
        else:
            msg = "cannot convert input 111111111111111 with the unit 'D'"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, 2, bad_val], unit="D")

    @pytest.mark.parametrize("bad_val", ["foo", 111111111111111])
    def test_to_timestamp_unit_coerce(self, bad_val):
        # [FUNCTION DOC]
        # Parametrized over the same two bad values. In coerce mode with unit days the valid entries convert and the bad one becomes NaT (second resolution).
        # coerce we can process
        expected = DatetimeIndex(
            [Timestamp("1970-01-02"), Timestamp("1970-01-03")] + ["NaT"] * 1,
            dtype="M8[s]",
        )
        result = to_datetime([1, 2, bad_val], unit="D", errors="coerce")
        tm.assert_index_equal(result, expected)

    def test_float_to_datetime_raise_near_bounds(self):
        # [FUNCTION DOC]
        # GH50183. Builds floats of day counts just inside (minus a small margin of 0.005 days) and just outside the bounds of the 64-bit nanosecond range. The inside values convert in raise and coerce modes, matching the expected integer nanoseconds approximately; values just outside raise OutOfBoundsDatetime "cannot convert input with unit 'D'" in raise mode.
        # GH50183
        msg = "cannot convert input with unit 'D'"
        oneday_in_ns = 1e9 * 60 * 60 * 24
        tsmax_in_days = 2**63 / oneday_in_ns  # 2**63 ns, in days
        # just in bounds
        should_succeed = Series(
            [0, tsmax_in_days - 0.005, -tsmax_in_days + 0.005], dtype=float
        )
        expected = (should_succeed * oneday_in_ns).astype(np.int64)
        for error_mode in ["raise", "coerce"]:
            result1 = to_datetime(should_succeed, unit="D", errors=error_mode)
            # Cast to `np.float64` so that `rtol` and inexact checking kick in
            # (`check_exact` doesn't take place for integer dtypes)
            tm.assert_almost_equal(
                result1.astype(np.int64).astype(np.float64),
                expected.astype(np.float64),
                rtol=1e-10,
            )
        # just out of bounds
        should_fail1 = Series([0, tsmax_in_days + 0.005], dtype=float)
        should_fail2 = Series([0, -tsmax_in_days - 0.005], dtype=float)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail1, unit="D", errors="raise")
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail2, unit="D", errors="raise")


class TestToDatetimeDataFrame:
    # [CLASS DOC]
    # Tests converting a DataFrame or a mapping of component columns (year, month, day, optional hour, minute, second, millisecond, microsecond, nanosecond, with accepted aliases) to a Series of timestamps, including errors for missing, extra and duplicate keys.
    @pytest.fixture
    def df(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a two-row DataFrame with integer component columns year, month, day, hour, minute, second, ms, us, ns (2015-02-04 06:58:10.001002003 and 2016-03-05 07:59:11.001002003).
        return DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "hour": [6, 7],
                "minute": [58, 59],
                "second": [10, 11],
                "ms": [1, 1],
                "us": [2, 2],
                "ns": [3, 3],
            }
        )

    def test_dataframe(self, df, cache):
        # [FUNCTION DOC]
        # Assembles from a dict of the year, month and day columns (series keep the index) and from a plain dictionary of those columns: both give a Series of midnight timestamps 2015-02-04 and 2016-03-05, with the same index.
        result = to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]}, cache=cache
        )
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:0:00")]
        )
        tm.assert_series_equal(result, expected)

        # dict-like
        result = to_datetime(df[["year", "month", "day"]].to_dict(), cache=cache)
        expected.index = Index([0, 1])
        tm.assert_series_equal(result, expected)

    def test_dataframe_dict_with_constructable(self, df, cache):
        # [FUNCTION DOC]
        # A mapping where month is a scalar (2) while year and day are columns gives timestamps in February for both rows (the scalar is broadcast).
        # dict but with constructable
        df2 = df[["year", "month", "day"]].to_dict()
        df2["month"] = 2
        result = to_datetime(df2, cache=cache)
        expected2 = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160205 00:0:00")],
            index=Index([0, 1]),
        )
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize(
        "unit",
        [
            {
                "year": "years",
                "month": "months",
                "day": "days",
                "hour": "hours",
                "minute": "minutes",
                "second": "seconds",
            },
            {
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "minute": "minute",
                "second": "second",
            },
        ],
    )
    def test_dataframe_field_aliases_column_subset(self, df, cache, unit):
        # [FUNCTION DOC]
        # Parametrized over two mappings of column name to alias (plural names, or singular names). Selecting the year to second columns and renaming with the alias mapping yields a microsecond-resolution Series of two timestamps (2015-02-04 06:58:10 and 2016-03-05 07:59:11).
        # unit mappings
        result = to_datetime(df[list(unit.keys())].rename(columns=unit), cache=cache)
        expected = Series(
            [Timestamp("20150204 06:58:10"), Timestamp("20160305 07:59:11")],
            dtype="M8[us]",
        )
        tm.assert_series_equal(result, expected)

    def test_dataframe_field_aliases(self, df, cache):
        # [FUNCTION DOC]
        # Renaming columns to themselves (year through ns, including the millisecond, microsecond, nanosecond aliases) assembles full timestamps with fractional seconds .001002003.
        d = {
            "year": "year",
            "month": "month",
            "day": "day",
            "hour": "hour",
            "minute": "minute",
            "second": "second",
            "ms": "ms",
            "us": "us",
            "ns": "ns",
        }

        result = to_datetime(df.rename(columns=d), cache=cache)
        expected = Series(
            [
                Timestamp("20150204 06:58:10.001002003"),
                Timestamp("20160305 07:59:11.001002003"),
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_dataframe_str_dtype(self, df, cache):
        # [FUNCTION DOC]
        # Converting the frame cast to strings gives the same full timestamps (strings are coerced back to integers).
        # coerce back to int
        result = to_datetime(df.astype(str), cache=cache)
        expected = Series(
            [
                Timestamp("20150204 06:58:10.001002003"),
                Timestamp("20160305 07:59:11.001002003"),
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_dataframe_float32_dtype(self, df, cache):
        # [FUNCTION DOC]
        # GH#60506. Converting the frame cast to 32-bit floats gives the same full timestamps (coerced to 64-bit floats first).
        # GH#60506
        # coerce to float64
        result = to_datetime(df.astype(np.float32), cache=cache)
        expected = Series(
            [
                Timestamp("20150204 06:58:10.001002003"),
                Timestamp("20160305 07:59:11.001002003"),
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_dataframe_coerce(self, cache):
        # [FUNCTION DOC]
        # A frame with month 20 is invalid: default mode raises a value error "cannot assemble the datetimes: time data ... doesn't match format"; coerce mode gives NaT for that row and the valid timestamp for the other.
        # passing coerce
        df2 = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})

        msg = (
            r'^cannot assemble the datetimes: time data ".+" doesn\'t '
            r'match format "%Y%m%d"\.'
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

        result = to_datetime(df2, errors="coerce", cache=cache)
        expected = Series([Timestamp("20150204 00:00:00"), NaT])
        tm.assert_series_equal(result, expected)

    def test_dataframe_extra_keys_raises(self, df, cache):
        # [FUNCTION DOC]
        # A frame with an extra column named foo raises a value error "extra keys have been passed to the datetime assemblage: [foo]".
        # extra columns
        msg = r"extra keys have been passed to the datetime assemblage: \[foo\]"
        df2 = df.copy()
        df2["foo"] = 1
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    @pytest.mark.parametrize(
        "cols",
        [
            ["year"],
            ["year", "month"],
            ["year", "month", "second"],
            ["month", "day"],
            ["year", "day", "second"],
        ],
    )
    def test_dataframe_missing_keys_raises(self, df, cache, cols):
        # [FUNCTION DOC]
        # Parametrized over column subsets lacking year, month or day. Raises a value error that assembling mappings requires at least year, month and day, naming the missing ones.
        # not enough
        msg = (
            r"to assemble mappings requires at least that \[year, month, "
            r"day\] be specified: \[.+\] is missing"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(df[cols], cache=cache)

    def test_dataframe_duplicate_columns_raises(self, cache):
        # [FUNCTION DOC]
        # Frames with a duplicated year column or a duplicated day column raise a value error "cannot assemble with duplicate keys".
        # duplicates
        msg = "cannot assemble with duplicate keys"
        df2 = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})
        df2.columns = ["year", "year", "day"]
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

        df2 = DataFrame(
            {"year": [2015, 2016], "month": [2, 20], "day": [4, 5], "hour": [4, 5]}
        )
        df2.columns = ["year", "month", "day", "day"]
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    def test_dataframe_int16(self, cache):
        # [FUNCTION DOC]
        # GH#13451. A frame of 16-bit integer columns year, month, day assembles to two midnight timestamps.
        # GH#13451
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})

        # int16
        result = to_datetime(df.astype("int16"), cache=cache)
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
        )
        tm.assert_series_equal(result, expected)

    def test_dataframe_mixed(self, cache):
        # [FUNCTION DOC]
        # A frame with mixed integer widths (64-bit year, 8-bit month and day) assembles to two midnight timestamps.
        # mixed dtypes
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        df["month"] = df["month"].astype("int8")
        df["day"] = df["day"].astype("int8")
        result = to_datetime(df, cache=cache)
        expected = Series(
            [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
        )
        tm.assert_series_equal(result, expected)

    def test_dataframe_float(self, cache):
        # [FUNCTION DOC]
        # A non-integral float month (1.5) raises a value error "cannot assemble the datetimes: unconverted data remains when parsing...".
        # float
        df = DataFrame({"year": [2000, 2001], "month": [1.5, 1], "day": [1, 1]})
        msg = (
            r"^cannot assemble the datetimes: unconverted data remains when parsing "
            r'with format ".*": "1".'
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(df, cache=cache)

    def test_dataframe_utc_true(self):
        # [FUNCTION DOC]
        # GH#23760. A frame of year/month/day converted with utc true gives a UTC-localized Series of microsecond resolution.
        # GH#23760
        df = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        result = to_datetime(df, utc=True)
        expected = Series(
            np.array(["2015-02-04", "2016-03-05"], dtype="datetime64[us]")
        ).dt.tz_localize("UTC")
        tm.assert_series_equal(result, expected)


class TestToDatetimeMisc:
    # [CLASS DOC]
    # Miscellaneous conversion tests: out-of-bounds values, ISO 8601 format handling with exact and non-exact matching, separators, non-padded values, dayfirst/yearfirst handling, empty and missing strings, caching, NumPy units, constructor interplay, and parse warnings.
    def test_to_datetime_barely_out_of_bounds(self):
        # [FUNCTION DOC]
        # GH#19529, GH#19382. A string just past the nanosecond upper bound (with sub-nanosecond digits) in an object array raises OutOfBoundsDatetime "Out of bounds nanosecond timestamp".
        # GH#19529
        # GH#19382 close enough to bounds that dropping nanos would result
        # in an in-bounds datetime
        arr = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)

        msg = "^Out of bounds nanosecond timestamp: .*"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arr)

    @pytest.mark.parametrize(
        "arg, exp_str",
        [
            ["2012-01-01 00:00:00", "2012-01-01 00:00:00"],
            ["20121001", "2012-10-01"],  # bad iso 8601
        ],
    )
    def test_to_datetime_iso8601(self, cache, arg, exp_str):
        # [FUNCTION DOC]
        # Parametrized over a standard ISO string and a compact date string flagged as a bad ISO 8601 form. A list with one such string converts to the expected Timestamp (first element checked).
        result = to_datetime([arg], cache=cache)
        exp = Timestamp(exp_str)
        assert result[0] == exp

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012", "%Y-%m"),
            ("2012-01", "%Y-%m-%d"),
            ("2012-01-01", "%Y-%m-%d %H"),
            ("2012-01-01 10", "%Y-%m-%d %H:%M"),
            ("2012-01-01 10:00", "%Y-%m-%d %H:%M:%S"),
            ("2012-01-01 10:00:00", "%Y-%m-%d %H:%M:%S.%f"),
            ("2012-01-01 10:00:00.123", "%Y-%m-%d %H:%M:%S.%f%z"),
            (0, "%Y-%m-%d"),
        ],
    )
    @pytest.mark.parametrize("exact", [True, False])
    def test_to_datetime_iso8601_fails(self, input, format, exact):
        # [FUNCTION DOC]
        # Issue 12649. Parametrized over (input, format) pairs where the format is longer than the input, and over exact true or false. Parsing raises a value error "time data ... doesn't match format ..." regardless of exact.
        # https://github.com/pandas-dev/pandas/issues/12649
        # `format` is longer than the string, so this fails regardless of `exact`
        with pytest.raises(
            ValueError,
            match=(rf"time data \"{input}\" doesn't match format " rf"\"{format}\""),
        ):
            to_datetime(input, format=format, exact=exact)

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012-01-01", "%Y-%m"),
            ("2012-01-01 10", "%Y-%m-%d"),
            ("2012-01-01 10:00", "%Y-%m-%d %H"),
            ("2012-01-01 10:00:00", "%Y-%m-%d %H:%M"),
            (0, "%Y-%m-%d"),
        ],
    )
    def test_to_datetime_iso8601_exact_fails(self, input, format):
        # [FUNCTION DOC]
        # Issue 12649. Parametrized over inputs longer than the format. With the default exact matching, a value error is raised (unconverted data remains, or the text does not match) followed by the standard advice about passing a format.
        # https://github.com/pandas-dev/pandas/issues/12649
        # `format` is shorter than the date string, so only fails with `exact=True`
        msg = "|".join(
            [
                '^unconverted data remains when parsing with format ".*": ".*". '
                f"{PARSING_ERR_MSG}$",
                f'^time data ".*" doesn\'t match format ".*". {PARSING_ERR_MSG}$',
            ]
        )
        with pytest.raises(
            ValueError,
            match=(msg),
        ):
            to_datetime(input, format=format)

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012-01-01", "%Y-%m"),
            ("2012-01-01 00", "%Y-%m-%d"),
            ("2012-01-01 00:00", "%Y-%m-%d %H"),
            ("2012-01-01 00:00:00", "%Y-%m-%d %H:%M"),
        ],
    )
    def test_to_datetime_iso8601_non_exact(self, input, format):
        # [FUNCTION DOC]
        # Issue 12649. Parametrized over (input, shorter format). With exact false the partial match succeeds and gives 2012-01-01.
        # https://github.com/pandas-dev/pandas/issues/12649
        expected = Timestamp(2012, 1, 1)
        result = to_datetime(input, format=format, exact=False)
        assert result == expected

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01", "%Y/%m"),
            ("2020-01-01", "%Y/%m/%d"),
            ("2020-01-01 00", "%Y/%m/%dT%H"),
            ("2020-01-01T00", "%Y/%m/%d %H"),
            ("2020-01-01 00:00", "%Y/%m/%dT%H:%M"),
            ("2020-01-01T00:00", "%Y/%m/%d %H:%M"),
            ("2020-01-01 00:00:00", "%Y/%m/%dT%H:%M:%S"),
            ("2020-01-01T00:00:00", "%Y/%m/%d %H:%M:%S"),
        ],
    )
    def test_to_datetime_iso8601_separator(self, input, format):
        # [FUNCTION DOC]
        # Issue 12649. Parametrized over inputs and formats that differ in the separator (dash vs slash, space vs T). Parsing raises a value error that the data does not match the format.
        # https://github.com/pandas-dev/pandas/issues/12649
        with pytest.raises(
            ValueError,
            match=(rf"time data \"{input}\" doesn\'t match format " rf"\"{format}\""),
        ):
            to_datetime(input, format=format)

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01", "%Y-%m"),
            ("2020-01-01", "%Y-%m-%d"),
            ("2020-01-01 00", "%Y-%m-%d %H"),
            ("2020-01-01T00", "%Y-%m-%dT%H"),
            ("2020-01-01 00:00", "%Y-%m-%d %H:%M"),
            ("2020-01-01T00:00", "%Y-%m-%dT%H:%M"),
            ("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
            ("2020-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"),
            ("2020-01-01T00:00:00.000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-01-01T00:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-01-01T00:00:00.000000000", "%Y-%m-%dT%H:%M:%S.%f"),
        ],
    )
    def test_to_datetime_iso8601_valid(self, input, format):
        # [FUNCTION DOC]
        # Issue 12649. Parametrized over valid ISO inputs with matching formats at various precisions (date, hour, minute, second, fractional digits). All parse to 2020-01-01.
        # https://github.com/pandas-dev/pandas/issues/12649
        expected = Timestamp(2020, 1, 1)
        result = to_datetime(input, format=format)
        assert result == expected

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-1", "%Y-%m"),
            ("2020-1-1", "%Y-%m-%d"),
            ("2020-1-1 0", "%Y-%m-%d %H"),
            ("2020-1-1T0", "%Y-%m-%dT%H"),
            ("2020-1-1 0:0", "%Y-%m-%d %H:%M"),
            ("2020-1-1T0:0", "%Y-%m-%dT%H:%M"),
            ("2020-1-1 0:0:0", "%Y-%m-%d %H:%M:%S"),
            ("2020-1-1T0:0:0", "%Y-%m-%dT%H:%M:%S"),
            ("2020-1-1T0:0:0.000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-1-1T0:0:0.000000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-1-1T0:0:0.000000000", "%Y-%m-%dT%H:%M:%S.%f"),
        ],
    )
    def test_to_datetime_iso8601_non_padded(self, input, format):
        # [FUNCTION DOC]
        # Issue 21422. Parametrized over inputs with non-zero-padded month, day, hour, minute, second. Parsing with the matching format gives 2020-01-01.
        # https://github.com/pandas-dev/pandas/issues/21422
        expected = Timestamp(2020, 1, 1)
        result = to_datetime(input, format=format)
        assert result == expected

    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01-01T00:00:00.000000000+00:00", "%Y-%m-%dT%H:%M:%S.%f%z"),
            ("2020-01-01T00:00:00+00:00", "%Y-%m-%dT%H:%M:%S%z"),
            ("2020-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),
        ],
    )
    def test_to_datetime_iso8601_with_timezone_valid(self, input, format):
        # [FUNCTION DOC]
        # Issue 12649. Parametrized over strings with +00:00 or Z suffix and offset formats. Parsing gives the UTC Timestamp for 2020-01-01.
        # https://github.com/pandas-dev/pandas/issues/12649
        expected = Timestamp(2020, 1, 1, tzinfo=timezone.utc)
        result = to_datetime(input, format=format)
        assert result == expected

    def test_to_datetime_default(self, cache):
        # [FUNCTION DOC]
        # Converting the year string "2001" gives datetime 2001-01-01.
        rs = to_datetime("2001", cache=cache)
        xp = datetime(2001, 1, 1)
        assert rs == xp

    @pytest.mark.xfail(reason="fails to enforce dayfirst=True, which would raise")
    def test_to_datetime_respects_dayfirst(self, cache):
        # [FUNCTION DOC]
        # Marked as an expected failure: the dayfirst option is not enforced (the call is expected to raise for an impossible month 13 under dayfirst, with a "Provide format" user warning, but it does not).
        # KNOWN ISSUE: the in-body comment says dayfirst is essentially broken; the test documents behavior that does not yet match the intent (hence the expected-failure mark).
        # dayfirst is essentially broken

        # The msg here is not important since it isn't actually raised yet.
        msg = "Invalid date specified"
        with pytest.raises(ValueError, match=msg):
            # if dayfirst is respected, then this would parse as month=13, which
            #  would raise
            with tm.assert_produces_warning(UserWarning, match="Provide format"):
                to_datetime("01-13-2012", dayfirst=True, cache=cache)

    def test_to_datetime_on_datetime64_series(self, cache):
        # [FUNCTION DOC]
        # #2699. Converting an already-datetime Series returns values equal to the input (first element checked).
        # #2699
        ser = Series(date_range("1/1/2000", periods=10))

        result = to_datetime(ser, cache=cache)
        assert result[0] == ser[0]

    def test_to_datetime_with_space_in_series(self, cache):
        # [FUNCTION DOC]
        # GH 6428. A Series with a whitespace-only string raises a value error (text " " does not match the inferred month/day/year format) in raise mode; coerce mode gives NaT for it and datetimes for the others.
        # GH 6428
        ser = Series(["10/18/2006", "10/18/2008", " "])
        msg = (
            r'^time data " " doesn\'t match format "%m/%d/%Y". ' rf"{PARSING_ERR_MSG}$"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, errors="raise", cache=cache)
        result_coerce = to_datetime(ser, errors="coerce", cache=cache)
        expected_coerce = Series([datetime(2006, 10, 18), datetime(2008, 10, 18), NaT])
        tm.assert_series_equal(result_coerce, expected_coerce)

    @td.skip_if_not_us_locale
    def test_to_datetime_with_apply(self, cache):
        # [FUNCTION DOC]
        # Only run with a US or default locale. GH 5195. Applying the conversion per element with a month-abbreviation/two-digit-year format gives the same result as converting the Series at once.
        # this is only locale tested with US/None locales
        # GH 5195
        # with a format and coerce a single item to_datetime fails
        td = Series(["May 04", "Jun 02", "Dec 11"], index=[1, 2, 3])
        expected = to_datetime(td, format="%b %y", cache=cache)
        result = td.apply(to_datetime, format="%b %y", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_timezone_name(self):
        # [FUNCTION DOC]
        # Issue 49748. The string "2020-01-01 00:00:00UTC" with a time zone name directive gives the UTC-localized Timestamp.
        # https://github.com/pandas-dev/pandas/issues/49748
        result = to_datetime("2020-01-01 00:00:00UTC", format="%Y-%m-%d %H:%M:%S%Z")
        expected = Timestamp(2020, 1, 1).tz_localize("UTC")
        assert result == expected

    @td.skip_if_not_us_locale
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    def test_to_datetime_with_apply_with_empty_str(self, cache, errors):
        # [FUNCTION DOC]
        # Only with a US or default locale. Parametrized over errors mode. GH 5195, GH50251. Applying per element in coerce mode over a Series with an empty string gives the same result as converting the whole Series with the given errors mode (empty becomes NaT).
        # this is only locale tested with US/None locales
        # GH 5195, GH50251
        # with a format and coerce a single item to_datetime fails
        td = Series(["May 04", "Jun 02", ""], index=[1, 2, 3])
        expected = to_datetime(td, format="%b %y", errors=errors, cache=cache)

        result = td.apply(
            lambda x: to_datetime(x, format="%b %y", errors="coerce", cache=cache)
        )
        tm.assert_series_equal(result, expected)

    def test_to_datetime_empty_stt(self, cache):
        # [FUNCTION DOC]
        # An empty string scalar gives NaT.
        # empty string
        result = to_datetime("", cache=cache)
        assert result is NaT

    def test_to_datetime_empty_str_list(self, cache):
        # [FUNCTION DOC]
        # A list of two empty strings gives all-missing results.
        result = to_datetime(["", ""], cache=cache)
        assert isna(result).all()

    def test_to_datetime_zero(self, cache):
        # [FUNCTION DOC]
        # The integer zero converts to the Timestamp at the epoch, equal to constructing a Timestamp from zero.
        # ints
        result = Timestamp(0)
        expected = to_datetime(0, cache=cache)
        assert result == expected

    def test_to_datetime_strings(self, cache):
        # [FUNCTION DOC]
        # GH 3888. Scalar year string conversion equals the first element of converting a one-element list.
        # GH 3888 (strings)
        expected = to_datetime(["2012"], cache=cache)[0]
        result = to_datetime("2012", cache=cache)
        assert result == expected

    def test_to_datetime_strings_variation(self, cache):
        # [FUNCTION DOC]
        # Three string forms (year, compact date, compact date with time) converted by the function agree approximately with the Timestamp constructor results (the names result and expected are swapped in the body, harmless).
        array = ["2012", "20120101", "20120101 12:01:01"]
        expected = [to_datetime(dt_str, cache=cache) for dt_str in array]
        result = [Timestamp(date_str) for date_str in array]
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("result", [Timestamp("2012"), to_datetime("2012")])
    def test_to_datetime_strings_vs_constructor(self, result):
        # [FUNCTION DOC]
        # Parametrized over a Timestamp from "2012" and the conversion result for "2012" (both evaluated at import time). Each equals Timestamp for 2012-01-01.
        expected = Timestamp(2012, 1, 1)
        assert result == expected

    def test_to_datetime_unprocessable_input(self, cache):
        # [FUNCTION DOC]
        # GH 4928, GH 21864. A list with the integer 1 and string "1" in raise mode raises a value error 'Given date string "1" not likely a datetime'.
        # GH 4928
        # GH 21864
        msg = '^Given date string "1" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, "1"], errors="raise", cache=cache)

    def test_to_datetime_other_datetime64_units(self):
        # [FUNCTION DOC]
        # A NumPy microsecond-resolution datetime scalar (2012-05-25), put in a datetime index or turned into a Timestamp, equals its object form.
        # KNOWN ISSUE: despite its name, this does not call the conversion function; it exercises the index and Timestamp constructors.
        # 5/25/2012
        scalar = np.int64(1337904000000000).view("M8[us]")
        as_obj = scalar.astype("O")

        index = DatetimeIndex([scalar])
        assert index[0] == scalar.astype("O")

        value = Timestamp(scalar)
        assert value == as_obj

    def test_to_datetime_list_of_integers(self):
        # [FUNCTION DOC]
        # A list of integer nanosecond values gives, through the datetime index constructor, an index equal to the original nanosecond range.
        # KNOWN ISSUE: despite its name, this does not call the conversion function; it exercises the index constructor.
        rng = date_range("1/1/2000", periods=20, unit="ns")
        rng = DatetimeIndex(rng.values)

        ints = list(rng.asi8)

        result = DatetimeIndex(ints)

        tm.assert_index_equal(rng, result)

    def test_to_datetime_overflow(self):
        # [FUNCTION DOC]
        # gh-17637. Creating a business-day range of 100000 periods starting 1700 with nanosecond unit raises OutOfBoundsTimedelta "Cannot cast ... days to unit='ns' without overflow".
        # KNOWN ISSUE: despite its name, this tests the date range constructor and not the conversion function; it also shares its name with another test in a different class.
        # gh-17637
        # we are overflowing Timedelta range here
        msg = "Cannot cast 139999 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            date_range(start="1/1/1700", freq="B", periods=100000, unit="ns")

    def test_to_datetime_float_with_nans_floating_point_error(self):
        # [FUNCTION DOC]
        # GH#58419. A float64 Series of 1000 NaN values followed by an epoch-second value converts in coerce mode with unit seconds to second-resolution NaT values and the exact timestamp 2024-04-04 08:23:53.
        # GH#58419
        ser = Series([np.nan] * 1000 + [1712219033.0], dtype=np.float64)
        result = to_datetime(ser, unit="s", errors="coerce")
        expected = Series(
            [NaT] * 1000 + [Timestamp("2024-04-04 08:23:53")], dtype="datetime64[s]"
        )
        tm.assert_series_equal(result, expected)

    def test_string_invalid_operation(self, cache):
        # [FUNCTION DOC]
        # GH #51084. An object array with a malformed string raises a value error "Unknown datetime string format".
        invalid = np.array(["87156549591102612381000001219H5"], dtype=object)
        # GH #51084

        with pytest.raises(ValueError, match="Unknown datetime string format"):
            to_datetime(invalid, errors="raise", cache=cache)

    def test_string_na_nat_conversion(self, cache):
        # [FUNCTION DOC]
        # GH 999, 858. A four-element object array of slash dates with a NaN: the low-level array conversion function gives approximately the expected microsecond NumPy array with the missing sentinel, and the conversion function returns a datetime index with identical values.
        # GH #999, #858

        strings = np.array(["1/1/2000", "1/2/2000", np.nan, "1/4/2000"], dtype=object)

        expected = np.empty(4, dtype="M8[us]")
        for i, val in enumerate(strings):
            if isna(val):
                expected[i] = iNaT
            else:
                expected[i] = parse(val)

        result = tslib.array_to_datetime(strings)[0]
        tm.assert_almost_equal(result, expected)

        result2 = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        tm.assert_numpy_array_equal(result, result2.values)

    def test_string_na_nat_conversion_malformed(self, cache):
        # [FUNCTION DOC]
        # GH 10636. An object array with an invalid day (100) and NaN raises a value error "Unknown datetime string format" (the default is raise).
        # KNOWN ISSUE: the same raising assertion is written twice in a row, identically.
        malformed = np.array(["1/100/2000", np.nan], dtype=object)

        # GH 10636, default is now 'raise'
        msg = r"Unknown datetime string format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

    def test_string_na_nat_conversion_with_name(self, cache):
        # [FUNCTION DOC]
        # A named Series of date strings and NaN converts to a datetime Series with NaT for NaN, preserving the index and name foo; a Series of already-converted datetimes with NaN converts to the same result and name.
        idx = ["a", "b", "c", "d", "e"]
        series = Series(
            ["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo"
        )
        dseries = Series(
            [
                to_datetime("1/1/2000", cache=cache),
                np.nan,
                to_datetime("1/3/2000", cache=cache),
                np.nan,
                to_datetime("1/5/2000", cache=cache),
            ],
            index=idx,
            name="foo",
        )

        result = to_datetime(series, cache=cache)
        dresult = to_datetime(dseries, cache=cache)

        expected = Series(np.empty(5, dtype="M8[us]"), index=idx)
        for i in range(5):
            x = series.iloc[i]
            if isna(x):
                expected.iloc[i] = NaT
            else:
                expected.iloc[i] = to_datetime(x, cache=cache)

        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name == "foo"

        tm.assert_series_equal(dresult, expected, check_names=False)
        assert dresult.name == "foo"

    @pytest.mark.parametrize(
        "unit",
        ["h", "m", "s", "ms", "us", "ns"],
    )
    def test_dti_constructor_numpy_timeunits(self, cache, unit):
        # [FUNCTION DOC]
        # GH 9114. Parametrized over NumPy datetime units hour to nanosecond. Values cast to that unit (hour and minute are cast to the closest supported unit, seconds) produce equal indexes through both the index constructor and the conversion function, with the expected dtype.
        # GH 9114
        dtype = np.dtype(f"M8[{unit}]")
        base = to_datetime(["2000-01-01T00:00", "2000-01-02T00:00", "NaT"], cache=cache)

        values = base.values.astype(dtype)

        if unit in ["h", "m"]:
            # we cast to closest supported unit
            unit = "s"
        exp_dtype = np.dtype(f"M8[{unit}]")
        expected = DatetimeIndex(base.astype(exp_dtype))
        assert expected.dtype == exp_dtype

        tm.assert_index_equal(DatetimeIndex(values), expected)
        tm.assert_index_equal(to_datetime(values, cache=cache), expected)

    def test_dayfirst(self, cache):
        # [FUNCTION DOC]
        # GH 5917. Day-first dates in text, supplied as list, array, Index or Series, convert with the day-first option to February 10, 11 and 12, 2014 through both the index constructor and the conversion function.
        # GH 5917
        arr = ["10/02/2014", "11/02/2014", "12/02/2014"]
        expected = DatetimeIndex(
            [datetime(2014, 2, 10), datetime(2014, 2, 11), datetime(2014, 2, 12)]
        )
        idx1 = DatetimeIndex(arr, dayfirst=True)
        idx2 = DatetimeIndex(np.array(arr), dayfirst=True)
        idx3 = to_datetime(arr, dayfirst=True, cache=cache)
        idx4 = to_datetime(np.array(arr), dayfirst=True, cache=cache)
        idx5 = DatetimeIndex(Index(arr), dayfirst=True)
        idx6 = DatetimeIndex(Series(arr), dayfirst=True)
        tm.assert_index_equal(expected, idx1)
        tm.assert_index_equal(expected, idx2)
        tm.assert_index_equal(expected, idx3)
        tm.assert_index_equal(expected, idx4)
        tm.assert_index_equal(expected, idx5)
        tm.assert_index_equal(expected, idx6)

    def test_dayfirst_warnings_valid_input(self):
        # [FUNCTION DOC]
        # GH 12585. Dates valid in day-first form (31/12/2014, 10/03/2011): with the day-first option true the index equals the expected microsecond dates with no warning; with it false, the same dates are parsed anyway and a user warning "Parsing dates in ... format when dayfirst=... was specified" is emitted.
        # GH 12585
        warning_msg = (
            "Parsing dates in .* format when dayfirst=.* was specified. "
            "Pass `dayfirst=.*` or specify a format to silence this warning."
        )

        # CASE 1: valid input
        arr = ["31/12/2014", "10/03/2011"]
        expected = DatetimeIndex(
            ["2014-12-31", "2011-03-10"], dtype="datetime64[us]", freq=None
        )

        # A. dayfirst arg correct, no warning
        res1 = to_datetime(arr, dayfirst=True)
        tm.assert_index_equal(expected, res1)

        # B. dayfirst arg incorrect, warning
        with tm.assert_produces_warning(UserWarning, match=warning_msg):
            res2 = to_datetime(arr, dayfirst=False)
        tm.assert_index_equal(expected, res2)

    def test_dayfirst_warnings_invalid_input(self):
        # [FUNCTION DOC]
        # Day-first and month-first dates mixed in a list cannot be handled with one format: raises a value error that the second text does not match the day/month/year format, followed by the standard advice.
        # CASE 2: invalid input
        # cannot consistently process with single format
        # ValueError *always* raised

        # first in DD/MM/YYYY, second in MM/DD/YYYY
        arr = ["31/12/2014", "03/30/2011"]

        with pytest.raises(
            ValueError,
            match=(
                r'^time data "03/30/2011" doesn\'t match format '
                rf'"%d/%m/%Y". {PARSING_ERR_MSG}$'
            ),
        ):
            to_datetime(arr, dayfirst=True)

    @pytest.mark.parametrize("klass", [DatetimeIndex, DatetimeArray._from_sequence])
    def test_to_datetime_dta_tz(self, klass):
        # [FUNCTION DOC]
        # GH#27733. Parametrized over the datetime index class and the array sequence constructor. A naive named daily range wrapped in that class converts with utc true to the same class holding the UTC-localized values.
        # GH#27733
        dti = date_range("2015-04-05", periods=3).rename("foo")
        expected = dti.tz_localize("UTC")

        obj = klass(dti)
        expected = klass(expected)

        result = to_datetime(obj, utc=True)
        tm.assert_equal(result, expected)


class TestGuessDatetimeFormat:
    # [CLASS DOC]
    # Tests of the private function that guesses a single format string for an array of date strings.
    @pytest.mark.parametrize(
        "test_list",
        [
            [
                "2011-12-30 00:00:00.000000",
                "2011-12-30 00:00:00.000000",
                "2011-12-30 00:00:00.000000",
            ],
            [np.nan, np.nan, "2011-12-30 00:00:00.000000"],
            ["", "2011-12-30 00:00:00.000000"],
            ["NaT", "2011-12-30 00:00:00.000000"],
            ["2011-12-30 00:00:00.000000", "random_string"],
            ["now", "2011-12-30 00:00:00.000000"],
            ["today", "2011-12-30 00:00:00.000000"],
        ],
    )
    def test_guess_datetime_format_for_array(self, test_list):
        # [FUNCTION DOC]
        # Parametrized over arrays of date strings in which the first usable element sets the format (missing values, empty strings, NaT text, keywords now and today, or a trailing unparseable string are skipped or ignored). The guessed format is the year-month-day hour:minute:second.fraction format.
        expected_format = "%Y-%m-%d %H:%M:%S.%f"
        test_array = np.array(test_list, dtype=object)
        assert tools._guess_datetime_format_for_array(test_array) == expected_format

    @td.skip_if_not_us_locale
    def test_guess_datetime_format_for_array_all_nans(self):
        # [FUNCTION DOC]
        # Only with a US or default locale. An array with only NaN values gives no format (None).
        format_for_string_of_nans = tools._guess_datetime_format_for_array(
            np.array([np.nan, np.nan, np.nan], dtype="O")
        )
        assert format_for_string_of_nans is None


class TestToDatetimeInferFormat:
    # [CLASS DOC]
    # Tests that format inference from the first element gives consistent results and errors when strings are inconsistent.
    @pytest.mark.parametrize(
        "test_format", ["%m-%d-%Y", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"]
    )
    def test_to_datetime_infer_datetime_format_consistent_format(
        self, cache, test_format
    ):
        # [FUNCTION DOC]
        # Parametrized over three string formats. A Series of fifty hourly timestamps rendered with the format converts to the same Series whether the format is passed explicitly or inferred.
        ser = Series(date_range("20000101", periods=50, freq="h"))

        s_as_dt_strings = ser.apply(lambda x: x.strftime(test_format))

        with_format = to_datetime(s_as_dt_strings, format=test_format, cache=cache)
        without_format = to_datetime(s_as_dt_strings, cache=cache)

        # Whether the format is explicitly passed, or
        # it is inferred, the results should all be the same
        tm.assert_series_equal(with_format, without_format)

    def test_to_datetime_inconsistent_format(self, cache):
        # [FUNCTION DOC]
        # Strings in different formats (slashes, dashes, ISO with T) raise a value error that the second string does not match the format inferred from the first, plus the standard advice.
        data = ["01/01/2011 00:00:00", "01-02-2011 00:00:00", "2011-01-03T00:00:00"]
        ser = Series(np.array(data))
        msg = (
            r'^time data "01-02-2011 00:00:00" doesn\'t match format '
            rf'"%m/%d/%Y %H:%M:%S". {PARSING_ERR_MSG}$'
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, cache=cache)

    def test_to_datetime_consistent_format(self, cache):
        # [FUNCTION DOC]
        # Strings with an abbreviated month name and slashes infer the format and give the expected first-of-month dates (microsecond resolution).
        data = ["Jan/01/2011", "Feb/01/2011", "Mar/01/2011"]
        ser = Series(np.array(data))
        result = to_datetime(ser, cache=cache)
        expected = Series(
            ["2011-01-01", "2011-02-01", "2011-03-01"], dtype="datetime64[us]"
        )
        tm.assert_series_equal(result, expected)

    def test_to_datetime_series_with_nans(self, cache):
        # [FUNCTION DOC]
        # An object Series with NaN values in between valid slash dates converts to a microsecond Series with NaT in the NaN positions.
        ser = Series(
            np.array(
                ["01/01/2011 00:00:00", np.nan, "01/03/2011 00:00:00", np.nan],
                dtype=object,
            )
        )
        result = to_datetime(ser, cache=cache)
        expected = Series(
            ["2011-01-01", NaT, "2011-01-03", NaT], dtype="datetime64[us]"
        )
        tm.assert_series_equal(result, expected)

    def test_to_datetime_series_start_with_nans(self, cache):
        # [FUNCTION DOC]
        # Leading NaN values do not prevent format inference; result has NaT for them followed by the three parsed dates.
        ser = Series(
            np.array(
                [
                    np.nan,
                    np.nan,
                    "01/01/2011 00:00:00",
                    "01/02/2011 00:00:00",
                    "01/03/2011 00:00:00",
                ],
                dtype=object,
            )
        )

        result = to_datetime(ser, cache=cache)
        expected = Series(
            [NaT, NaT, "2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[us]"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "tz_name, offset",
        [("UTC", 0), ("UTC-3", 180), ("UTC+3", -180)],
    )
    def test_infer_datetime_format_tz_name(self, tz_name, offset):
        # [FUNCTION DOC]
        # GH 33133. Parametrized over (zone name, offset in minutes): "UTC", "UTC-3" and "UTC+3" suffixes. A one-element Series converts to a Series with the equal fixed-offset time zone (UTC-3 means 180 minutes, UTC+3 means -180, using the POSIX sign convention).
        # GH 33133
        ser = Series([f"2019-02-02 08:07:13 {tz_name}"])
        result = to_datetime(ser)
        tz = timezone(timedelta(minutes=offset))
        expected = Series([Timestamp("2019-02-02 08:07:13").tz_localize(tz)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ts,zero_tz",
        [
            ("2019-02-02 08:07:13", "Z"),
            ("2019-02-02 08:07:13", ""),
            ("2019-02-02 08:07:13.012345", "Z"),
            ("2019-02-02 08:07:13.012345", ""),
        ],
    )
    def test_infer_datetime_format_zero_tz(self, ts, zero_tz):
        # [FUNCTION DOC]
        # GH 41047. Parametrized over strings with and without fractional seconds and with a Z suffix or none. The result is UTC-aware when Z is present, otherwise naive.
        # GH 41047
        ser = Series([ts + zero_tz])
        result = to_datetime(ser)
        tz = timezone.utc if zero_tz == "Z" else None
        expected = Series([Timestamp(ts, tz=tz)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("format", [None, "%Y-%m-%d"])
    def test_to_datetime_iso8601_noleading_0s(self, cache, format):
        # [FUNCTION DOC]
        # GH 11871. Parametrized over no format and a year-month-day format. Strings without zero padding in month and day convert to the expected timestamps.
        # GH 11871
        ser = Series(["2014-1-1", "2014-2-2", "2015-3-3"])
        expected = Series(
            [
                Timestamp("2014-01-01"),
                Timestamp("2014-02-02"),
                Timestamp("2015-03-03"),
            ]
        )
        result = to_datetime(ser, format=format, cache=cache)
        tm.assert_series_equal(result, expected)


class TestDaysInMonth:
    # [CLASS DOC]
    # Tests for dates that do not exist in a month (issue 10154).
    # tests for issue #10154

    @pytest.mark.parametrize(
        "arg, format",
        [
            ["2015-02-29", None],
            ["2015-02-29", "%Y-%m-%d"],
            ["2015-02-32", "%Y-%m-%d"],
            ["2015-04-31", "%Y-%m-%d"],
        ],
    )
    def test_day_not_in_month_coerce(self, cache, arg, format):
        # [FUNCTION DOC]
        # Parametrized over (string, format) with February 29, 2015, February 32 and April 31. In coerce mode each yields a missing value.
        assert isna(to_datetime(arg, errors="coerce", format=format, cache=cache))

    def test_day_not_in_month_raise(self, cache):
        # [FUNCTION DOC]
        # Raise mode on "2015-02-29" raises a value error whose text depends on the Python version (a day-range message from Python 3.14 on, otherwise day is out of range for month).
        if PY314:
            msg = "day 29 must be in range 1..28 for month 2 in year 2015: 2015-02-29"
        else:
            msg = "day is out of range for month: 2015-02-29"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-02-29", errors="raise", cache=cache)

    @pytest.mark.parametrize(
        "arg, format, msg",
        [
            (
                "2015-02-29",
                "%Y-%m-%d",
                f"^{DAY_IS_OUT_OF_RANGE}. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-29-02",
                "%Y-%d-%m",
                f"^{DAY_IS_OUT_OF_RANGE}. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-02-32",
                "%Y-%m-%d",
                '^unconverted data remains when parsing with format "%Y-%m-%d": "2". '
                f"{PARSING_ERR_MSG}$",
            ),
            (
                "2015-32-02",
                "%Y-%d-%m",
                '^time data "2015-32-02" doesn\'t match format "%Y-%d-%m". '
                f"{PARSING_ERR_MSG}$",
            ),
            (
                "2015-04-31",
                "%Y-%m-%d",
                f"^{DAY_IS_OUT_OF_RANGE}. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-31-04",
                "%Y-%d-%m",
                f"^{DAY_IS_OUT_OF_RANGE}. {PARSING_ERR_MSG}$",
            ),
        ],
    )
    def test_day_not_in_month_raise_value(self, cache, arg, format, msg):
        # [FUNCTION DOC]
        # Issue 50462. Parametrized over (string, format, message) with invalid days. With raise mode and a format, a value error is raised with the matching text: day out of range, unconverted data remains, or text does not match the format; each followed by the standard advice.
        # https://github.com/pandas-dev/pandas/issues/50462
        with pytest.raises(ValueError, match=msg):
            to_datetime(arg, errors="raise", format=format, cache=cache)


class TestDatetimeParsingWrappers:
    # [CLASS DOC]
    # Tests that the string parsing wrappers (the low-level parse function with resolution, the conversion function with list, array and scalar input, the index constructor, Timestamp and date range) agree on a collection of date string forms.
    @pytest.mark.parametrize(
        "date_str, expected",
        [
            ("2011-01-01", datetime(2011, 1, 1)),
            ("2Q2005", datetime(2005, 4, 1)),
            ("2Q05", datetime(2005, 4, 1)),
            ("2005Q1", datetime(2005, 1, 1)),
            ("05Q1", datetime(2005, 1, 1)),
            ("2011Q3", datetime(2011, 7, 1)),
            ("11Q3", datetime(2011, 7, 1)),
            ("3Q2011", datetime(2011, 7, 1)),
            ("3Q11", datetime(2011, 7, 1)),
            # quarterly without space
            ("2000Q4", datetime(2000, 10, 1)),
            ("00Q4", datetime(2000, 10, 1)),
            ("4Q2000", datetime(2000, 10, 1)),
            ("4Q00", datetime(2000, 10, 1)),
            ("2000q4", datetime(2000, 10, 1)),
            ("2000-Q4", datetime(2000, 10, 1)),
            ("00-Q4", datetime(2000, 10, 1)),
            ("4Q-2000", datetime(2000, 10, 1)),
            ("4Q-00", datetime(2000, 10, 1)),
            ("00q4", datetime(2000, 10, 1)),
            ("2005", datetime(2005, 1, 1)),
            ("2005-11", datetime(2005, 11, 1)),
            ("2005 11", datetime(2005, 11, 1)),
            ("11-2005", datetime(2005, 11, 1)),
            ("11 2005", datetime(2005, 11, 1)),
            ("200511", datetime(2020, 5, 11)),
            ("20051109", datetime(2005, 11, 9)),
            ("20051109 10:15", datetime(2005, 11, 9, 10, 15)),
            ("20051109 08H", datetime(2005, 11, 9, 8, 0)),
            ("2005-11-09 10:15", datetime(2005, 11, 9, 10, 15)),
            ("2005-11-09 08H", datetime(2005, 11, 9, 8, 0)),
            ("2005/11/09 10:15", datetime(2005, 11, 9, 10, 15)),
            ("2005/11/09 10:15:32", datetime(2005, 11, 9, 10, 15, 32)),
            ("2005/11/09 10:15:32 AM", datetime(2005, 11, 9, 10, 15, 32)),
            ("2005/11/09 10:15:32 PM", datetime(2005, 11, 9, 22, 15, 32)),
            ("2005/11/09 08H", datetime(2005, 11, 9, 8, 0)),
            ("Thu Sep 25 10:36:28 2003", datetime(2003, 9, 25, 10, 36, 28)),
            ("Thu Sep 25 2003", datetime(2003, 9, 25)),
            ("Sep 25 2003", datetime(2003, 9, 25)),
            ("January 1 2014", datetime(2014, 1, 1)),
            # GH#10537
            ("2014-06", datetime(2014, 6, 1)),
            ("06-2014", datetime(2014, 6, 1)),
            ("2014-6", datetime(2014, 6, 1)),
            ("6-2014", datetime(2014, 6, 1)),
            ("20010101 12", datetime(2001, 1, 1, 12)),
            ("20010101 1234", datetime(2001, 1, 1, 12, 34)),
            ("20010101 123456", datetime(2001, 1, 1, 12, 34, 56)),
        ],
    )
    def test_parsers(self, date_str, expected, cache):
        # [FUNCTION DOC]
        # Parametrized over many date strings (quarters, years, months, compact dates, times, AM/PM, textual dates) with expected datetimes and the cache fixture. Uses year-first parsing. The low-level parser and scalar conversion equal the expected datetime; the list, array, index constructor, Index and Series based results equal a one-element index in the resolution reported by the low-level parser (nanoseconds if the string has nanosecond digits, else microseconds).
        # KNOWN ISSUE: the year-first flag is fixed to true, so the final branch (guarded by a not-year-first check, containing the Timestamp and date range checks) never executes; it also uses a deprecated seconds frequency alias.
        # dateutil >= 2.5.0 defaults to yearfirst=True
        # https://github.com/dateutil/dateutil/issues/217
        yearfirst = True

        result1, reso_attrname = parsing.parse_datetime_string_with_reso(
            date_str, yearfirst=yearfirst
        )

        reso = {
            "nanosecond": "ns",
        }.get(reso_attrname, "us")
        result2 = to_datetime(date_str, yearfirst=yearfirst)
        result3 = to_datetime([date_str], yearfirst=yearfirst)
        # result5 is used below
        result4 = to_datetime(
            np.array([date_str], dtype=object), yearfirst=yearfirst, cache=cache
        )
        result6 = DatetimeIndex([date_str], yearfirst=yearfirst)
        # result7 is used below
        result8 = DatetimeIndex(Index([date_str]), yearfirst=yearfirst)
        result9 = DatetimeIndex(Series([date_str]), yearfirst=yearfirst)

        for res in [result1, result2]:
            assert res == expected
        for res in [result3, result4, result6, result8, result9]:
            exp = DatetimeIndex([Timestamp(expected)]).as_unit(reso)
            tm.assert_index_equal(res, exp)

        # these really need to have yearfirst, but we don't support
        if not yearfirst:
            result5 = Timestamp(date_str)
            assert result5 == expected
            result7 = date_range(date_str, freq="S", periods=1, yearfirst=yearfirst)
            assert result7 == expected

    def test_na_values_with_cache(
        self, cache, unique_nulls_fixture, unique_nulls_fixture2
    ):
        # [FUNCTION DOC]
        # GH22305. Uses the cache fixture and two null fixtures. A list of two null values of different types converts to a second-resolution all-NaT index.
        # GH22305
        expected = Index([NaT, NaT], dtype="datetime64[s]")
        result = to_datetime([unique_nulls_fixture, unique_nulls_fixture2], cache=cache)
        tm.assert_index_equal(result, expected)

    def test_parsers_nat(self):
        # [FUNCTION DOC]
        # The text "NaT" converts to the NaT singleton through the low-level parser, the conversion function, Timestamp and the index constructor (identity checks).
        # Test that each of several string-accepting methods return pd.NaT
        result1, _ = parsing.parse_datetime_string_with_reso("NaT")
        result2 = to_datetime("NaT")
        result3 = Timestamp("NaT")
        result4 = DatetimeIndex(["NaT"])[0]
        assert result1 is NaT
        assert result2 is NaT
        assert result3 is NaT
        assert result4 is NaT

    @pytest.mark.parametrize(
        "date_str, dayfirst, yearfirst, expected",
        [
            ("10-11-12", False, False, datetime(2012, 10, 11)),
            ("10-11-12", True, False, datetime(2012, 11, 10)),
            ("10-11-12", False, True, datetime(2010, 11, 12)),
            ("10-11-12", True, True, datetime(2010, 12, 11)),
            ("20/12/21", False, False, datetime(2021, 12, 20)),
            ("20/12/21", True, False, datetime(2021, 12, 20)),
            ("20/12/21", False, True, datetime(2020, 12, 21)),
            ("20/12/21", True, True, datetime(2020, 12, 21)),
            # GH 58859
            ("20201012", True, False, datetime(2020, 12, 10)),
        ],
    )
    def test_parsers_dayfirst_yearfirst(
        self, cache, date_str, dayfirst, yearfirst, expected
    ):
        # [FUNCTION DOC]
        # GH 58859 among others. Parametrized over (string, dayfirst, yearfirst, expected). The expected result equals the dateutil parse result with the same options; the low-level parser, the conversion function and index constructor all agree. The plain Timestamp constructor is checked only when neither option is set. The long comment block records dateutil versions' behaviors for these inputs.
        # OK
        # 2.5.1 10-11-12   [dayfirst=0, yearfirst=0] -> 2012-10-11 00:00:00
        # 2.5.2 10-11-12   [dayfirst=0, yearfirst=1] -> 2012-10-11 00:00:00
        # 2.5.3 10-11-12   [dayfirst=0, yearfirst=0] -> 2012-10-11 00:00:00

        # OK
        # 2.5.1 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.2 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.3 10-11-12   [dayfirst=0, yearfirst=1] -> 2010-11-12 00:00:00

        # bug fix in 2.5.2
        # 2.5.1 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-11-12 00:00:00
        # 2.5.2 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-12-11 00:00:00
        # 2.5.3 10-11-12   [dayfirst=1, yearfirst=1] -> 2010-12-11 00:00:00

        # OK
        # 2.5.1 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00
        # 2.5.2 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00
        # 2.5.3 10-11-12   [dayfirst=1, yearfirst=0] -> 2012-11-10 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.2 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.3 20/12/21   [dayfirst=0, yearfirst=0] -> 2021-12-20 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.2 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.3 20/12/21   [dayfirst=0, yearfirst=1] -> 2020-12-21 00:00:00

        # revert of bug in 2.5.2
        # 2.5.1 20/12/21   [dayfirst=1, yearfirst=1] -> 2020-12-21 00:00:00
        # 2.5.2 20/12/21   [dayfirst=1, yearfirst=1] -> month must be in 1..12
        # 2.5.3 20/12/21   [dayfirst=1, yearfirst=1] -> 2020-12-21 00:00:00

        # OK
        # 2.5.1 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.2 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00
        # 2.5.3 20/12/21   [dayfirst=1, yearfirst=0] -> 2021-12-20 00:00:00

        # str : dayfirst, yearfirst, expected

        # compare with dateutil result
        dateutil_result = parse(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        assert dateutil_result == expected

        result1, _ = parsing.parse_datetime_string_with_reso(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst
        )

        # we don't support dayfirst/yearfirst here:
        if not dayfirst and not yearfirst:
            result2 = Timestamp(date_str)
            assert result2 == expected

        result3 = to_datetime(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache
        )

        result4 = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]

        assert result1 == expected
        assert result3 == expected
        assert result4 == expected

    @pytest.mark.parametrize(
        "date_str, exp_def",
        [["10:15", datetime(1, 1, 1, 10, 15)], ["9:05", datetime(1, 1, 1, 9, 5)]],
    )
    def test_parsers_timestring(self, date_str, exp_def):
        # [FUNCTION DOC]
        # Parametrized over time-only strings. The low-level parser returns the time on the minimal default date (year 1) while the conversion function, list conversion, Timestamp and index constructor all return the same value as dateutil (which uses today's date).
        # must be the same as dateutil result
        exp_now = parse(date_str)

        result1, _ = parsing.parse_datetime_string_with_reso(date_str)
        result2 = to_datetime(date_str)
        result3 = to_datetime([date_str])
        result4 = Timestamp(date_str)
        result5 = DatetimeIndex([date_str])[0]
        # parse time string return time string based on default date
        # others are not, and can't be changed because it is used in
        # time series plot
        assert result1 == exp_def
        assert result2 == exp_now
        assert result3 == exp_now
        assert result4 == exp_now
        assert result5 == exp_now

    @pytest.mark.parametrize(
        "dt_string, tz, dt_string_repr",
        [
            (
                "2013-01-01 05:45+0545",
                timezone(timedelta(minutes=345)),
                "Timestamp('2013-01-01 05:45:00+0545', tz='UTC+05:45')",
            ),
            (
                "2013-01-01 05:30+0530",
                timezone(timedelta(minutes=330)),
                "Timestamp('2013-01-01 05:30:00+0530', tz='UTC+05:30')",
            ),
        ],
    )
    def test_parsers_timezone_minute_offsets_roundtrip(
        self, cache, dt_string, tz, dt_string_repr
    ):
        # [FUNCTION DOC]
        # GH11708. Parametrized over strings with +0545 and +0530 offsets, the equivalent fixed-offset zone and the expected repr. Converting equals the UTC base converted to that zone, and the repr of the result matches the expected text.
        # GH11708
        base = to_datetime("2013-01-01 00:00:00", cache=cache)
        base = base.tz_localize("UTC").tz_convert(tz)
        dt_time = to_datetime(dt_string, cache=cache)
        assert base == dt_time
        assert dt_string_repr == repr(dt_time)


@pytest.fixture(params=["D", "s", "ms", "us", "ns"])
def units(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the units D, s, ms, us, ns (day and some time units), returning the current unit. Has its own docstring listing them.
    """Day and some time units.

    * D
    * s
    * ms
    * us
    * ns
    """
    return request.param


@pytest.fixture
def julian_dates():
    # [FUNCTION DOC]
    # HELPER: fixture returning the values of ten daily dates starting 2014-01-01 as Julian date floats.
    return date_range("2014-1-1", periods=10).to_julian_date().values


class TestOrigin:
    # [CLASS DOC]
    # Tests of the origin parameter: unix, julian, numeric and datetime-like epochs, invalid combinations, and time zone checks.
    def test_origin_and_unit(self):
        # [FUNCTION DOC]
        # GH#42624. Converting 1 with unit seconds and numeric origin 1 gives 1970-01-01 00:00:02; with origin 1000000000 gives 2001-09-09 01:46:41.
        # GH#42624
        ts = to_datetime(1, unit="s", origin=1)
        expected = Timestamp("1970-01-01 00:00:02")
        assert ts == expected

        ts = to_datetime(1, unit="s", origin=1_000_000_000)
        expected = Timestamp("2001-09-09 01:46:41")
        assert ts == expected

    def test_julian(self, julian_dates):
        # [FUNCTION DOC]
        # gh-11276, gh-11745. Converting Julian date values with unit days and origin julian equals converting the same values minus the Julian date of the epoch with unit days.
        # gh-11276, gh-11745
        # for origin as julian

        result = Series(to_datetime(julian_dates, unit="D", origin="julian"))
        expected = Series(
            to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit="D")
        )
        tm.assert_series_equal(result, expected)

    def test_unix(self):
        # [FUNCTION DOC]
        # Values 0, 1, 2 with unit days and unix origin give 1970-01-01, 02, 03 in second resolution.
        result = Series(to_datetime([0, 1, 2], unit="D", origin="unix"))
        expected = Series(
            [Timestamp("1970-01-01"), Timestamp("1970-01-02"), Timestamp("1970-01-03")],
            dtype="M8[s]",
        )
        tm.assert_series_equal(result, expected)

    def test_julian_round_trip(self):
        # [FUNCTION DOC]
        # The value 2456658 with julian origin gives a Timestamp whose Julian date is the same number; the value 1 is out of bounds and raises a value error "1 is Out of Bounds for origin='julian'".
        result = to_datetime(2456658, origin="julian", unit="D")
        assert result.to_julian_date() == 2456658

        # out-of-bounds
        msg = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin="julian", unit="D")

    def test_invalid_unit(self, units, julian_dates):
        # [FUNCTION DOC]
        # Parametrized by the unit fixture and Julian date values. For any unit other than days, the julian origin raises a value error "unit must be 'D' for origin='julian'". For days nothing is asserted.
        # checking for invalid combination of origin='julian' and unit != D
        if units != "D":
            msg = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin="julian")

    @pytest.mark.parametrize("unit", ["ns", "D"])
    def test_invalid_origin(self, unit):
        # [FUNCTION DOC]
        # Parametrized over units ns and D. A string input with an origin raises a value error that it must be numeric with a unit specified.
        # need to have a numeric specified
        msg = "it must be numeric with a unit specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2005-01-01", origin="1960-01-01", unit=unit)

    @pytest.mark.parametrize(
        "epochs",
        [
            Timestamp(1960, 1, 1),
            datetime(1960, 1, 1),
            "1960-01-01",
            np.datetime64("1960-01-01"),
        ],
    )
    def test_epoch(self, units, epochs):
        # [FUNCTION DOC]
        # Parametrized by the unit fixture and four representations of 1960-01-01 (Timestamp, datetime, string, NumPy datetime). Converting integers 0 to 4 with the unit and origin gives a Series of the origin plus each multiple of the unit; resolution is seconds for days, otherwise the unit itself.
        epoch_1960 = Timestamp(1960, 1, 1)
        units_from_epochs = np.arange(5, dtype=np.int64)
        exp_unit = "s" if units == "D" else units
        expected = Series(
            [pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs],
            dtype=f"M8[{exp_unit}]",
        )

        result = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "origin, exc",
        [
            ("random_string", ValueError),
            ("epoch", ValueError),
            ("13-24-1990", ValueError),
            (datetime(1, 1, 1), OutOfBoundsDatetime),
        ],
    )
    def test_invalid_origins(self, origin, exc, units):
        # [FUNCTION DOC]
        # Parametrized over (origin, exception class) with unit fixture. Origins that are not parseable, the text "epoch", or year 1 (out of bounds) raise the given exception, whose message matches one of: origin out of bounds, origin cannot be converted to a Timestamp, or overflow casting to nanoseconds.
        msg = "|".join(
            [
                f"origin {origin} is Out of Bounds",
                f"origin {origin} cannot be converted to a Timestamp",
                "Cannot cast .* to unit='ns' without overflow",
            ]
        )
        with pytest.raises(exc, match=msg):
            to_datetime(list(range(5)), unit=units, origin=origin)

    def test_invalid_origins_tzinfo(self):
        # [FUNCTION DOC]
        # GH16842. A time-zone-aware origin raises a value error "must be tz-naive".
        # GH16842
        with pytest.raises(ValueError, match="must be tz-naive"):
            to_datetime(1, unit="D", origin=datetime(2000, 1, 1, tzinfo=timezone.utc))

    def test_incorrect_value_exception(self):
        # [FUNCTION DOC]
        # GH47495. A list holding the keyword today and the text "yesterday" raises a value error "Unknown datetime string format, unable to parse: yesterday".
        # GH47495
        msg = "Unknown datetime string format, unable to parse: yesterday"
        with pytest.raises(ValueError, match=msg):
            to_datetime(["today", "yesterday"])

    @pytest.mark.parametrize(
        "format, warning",
        [
            (None, UserWarning),
            ("%Y-%m-%d %H:%M:%S", None),
            ("%Y-%d-%m %H:%M:%S", None),
        ],
    )
    def test_to_datetime_out_of_bounds_with_format_arg(self, format, warning):
        # [FUNCTION DOC]
        # see gh-23830. Parametrized over (format, warning). Without a format, a far-future string converts to a Timestamp in 2417 (October 10). With a format that lacks the fractional part, a value error "unconverted data remains" is raised.
        # KNOWN ISSUE: the warning parameter is accepted but never used, so the expected user warning for the no-format case is not checked.
        # see gh-23830
        if format is None:
            res = to_datetime("2417-10-10 00:00:00.00", format=format)
            assert isinstance(res, Timestamp)
            assert res.year == 2417
            assert res.month == 10
            assert res.day == 10
        else:
            msg = "unconverted data remains when parsing with format.*"
            with pytest.raises(ValueError, match=msg):
                to_datetime("2417-10-10 00:00:00.00", format=format)

    @pytest.mark.parametrize(
        "arg, origin, expected_str",
        [
            [200 * 365, "unix", "2169-11-13 00:00:00"],
            [200 * 365, "1870-01-01", "2069-11-13 00:00:00"],
            [300 * 365, "1870-01-01", "2169-10-20 00:00:00"],
        ],
    )
    def test_processing_order(self, arg, origin, expected_str):
        # [FUNCTION DOC]
        # Parametrized over (days, origin, expected text) using unix origin and a 1870 origin. Out-of-bounds handling must happen before date construction: converting 73000 or 109500 days from the given origin gives the exact expected Timestamps (the same checks are repeated on fixed values after the call).
        # make sure we handle out-of-bounds *before*
        # constructing the dates

        result = to_datetime(arg, unit="D", origin=origin)
        expected = Timestamp(expected_str)
        assert result == expected

        result = to_datetime(200 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2069-11-13 00:00:00")
        assert result == expected

        result = to_datetime(300 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2169-10-20 00:00:00")
        assert result == expected

    @pytest.mark.parametrize(
        "offset,utc,exp",
        [
            ["Z", True, "2019-01-01T00:00:00.000Z"],
            ["Z", None, "2019-01-01T00:00:00.000Z"],
            ["-01:00", True, "2019-01-01T01:00:00.000Z"],
            ["-01:00", None, "2019-01-01T00:00:00.000-01:00"],
        ],
    )
    def test_arg_tz_ns_unit(self, offset, utc, exp):
        # [FUNCTION DOC]
        # GH 25546. Parametrized over (offset suffix, utc flag, expected ISO text). A string with the Z or -01:00 suffix converted with unit nanoseconds (which applies to numbers only, so strings are parsed normally) gives the expected index in microsecond resolution; UTC conversion when utc is true, offset preserved when utc is none.
        # GH 25546
        arg = "2019-01-01T00:00:00.000" + offset
        result = to_datetime([arg], unit="ns", utc=utc)
        expected = to_datetime([exp]).as_unit("us")
        tm.assert_index_equal(result, expected)


class TestShouldCache:
    # [CLASS DOC]
    # Tests of the helper that decides whether caching of unique values is worthwhile for a list-like.
    @pytest.mark.parametrize(
        "listlike,do_caching",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False),
            ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True),
        ],
    )
    def test_should_cache(self, listlike, do_caching):
        # [FUNCTION DOC]
        # Parametrized over (list, expected decision). With check count equal to the length and a unique share threshold of 0.7, a list of all unique values returns false and a list with many duplicates returns true.
        assert (
            tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7)
            == do_caching
        )

    @pytest.mark.parametrize(
        "unique_share,check_count, err_message",
        [
            (0.5, 11, r"check_count must be in next bounds: \[0; len\(arg\)\]"),
            (10, 2, r"unique_share must be in next bounds: \(0; 1\)"),
        ],
    )
    def test_should_cache_errors(self, unique_share, check_count, err_message):
        # [FUNCTION DOC]
        # Parametrized over (unique share, check count, message). A check count larger than the list length, or a unique share outside the open interval from 0 to 1, raises AssertionError with a message describing the allowed bounds.
        arg = [5] * 10

        with pytest.raises(AssertionError, match=err_message):
            tools.should_cache(arg, unique_share, check_count)

    @pytest.mark.parametrize(
        "listlike",
        [
            (deque([Timestamp("2010-06-02 09:30:00")] * 51)),
            ([Timestamp("2010-06-02 09:30:00")] * 51),
            (tuple([Timestamp("2010-06-02 09:30:00")] * 51)),
        ],
    )
    def test_no_slicing_errors_in_should_cache(self, listlike):
        # [FUNCTION DOC]
        # GH#29403. Parametrized over a deque, a list and a tuple of 51 identical Timestamps. The helper returns true for each without raising (no slicing problems with non-sliceable types).
        # GH#29403
        assert tools.should_cache(listlike) is True


def test_nullable_integer_to_datetime():
    # [FUNCTION DOC]
    # Test for #30050. A nullable Int64 Series with missing values and a large integer (2 to the 61st power) converts with unit nanoseconds to a nanosecond Series with NaT for the missing entries; the input Series is not mutated.
    # Test for #30050
    ser = Series([1, 2, None, 2**61, None], dtype="Int64")
    ser_copy = ser.copy()

    res = to_datetime(ser, unit="ns")

    expected = Series(
        [
            np.datetime64("1970-01-01 00:00:00.000000001"),
            np.datetime64("1970-01-01 00:00:00.000000002"),
            np.datetime64("NaT", "ns"),
            np.datetime64("2043-01-25 23:56:49.213693952"),
            np.datetime64("NaT", "ns"),
        ]
    )
    tm.assert_series_equal(res, expected)
    # Check that ser isn't mutated
    tm.assert_series_equal(ser, ser_copy)


@pytest.mark.parametrize("klass", [np.array, list])
def test_na_to_datetime(nulls_fixture, klass):
    # [FUNCTION DOC]
    # Parametrized over list and array containers and the nulls fixture. A container holding a single null-like value converts to an index whose first element is NaT, except for Decimal NaN, which raises a type error "not convertible to datetime".
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match="not convertible to datetime"):
            to_datetime(klass([nulls_fixture]))

    else:
        result = to_datetime(klass([nulls_fixture]))

        assert result[0] is NaT


@pytest.mark.parametrize("errors", ["raise", "coerce"])
@pytest.mark.parametrize(
    "args, format",
    [
        (["03/24/2016", "03/25/2016", ""], "%m/%d/%Y"),
        (["2016-03-24", "2016-03-25", ""], "%Y-%m-%d"),
    ],
    ids=["non-ISO8601", "ISO8601"],
)
def test_empty_string_datetime(errors, args, format):
    # [FUNCTION DOC]
    # GH13044, GH50251. Parametrized over errors mode and (list with two dates and an empty string, format) for non-ISO and ISO formats. The empty string becomes NaT and the other two convert, in both modes, giving a microsecond Series.
    # GH13044, GH50251
    td = Series(args)

    # coerce empty string to pd.NaT
    result = to_datetime(td, format=format, errors=errors)
    expected = Series(["2016-03-24", "2016-03-25", NaT], dtype="datetime64[us]")
    tm.assert_series_equal(expected, result)


def test_empty_string_datetime_coerce__unit():
    # [FUNCTION DOC]
    # GH13044. A list with the integer 1 and an empty string with unit seconds gives 1970-01-01 00:00:01 and NaT in second resolution in coerce mode; raise mode gives the same result without an exception.
    # GH13044
    # coerce empty string to pd.NaT
    result = to_datetime([1, ""], unit="s", errors="coerce")
    expected = DatetimeIndex(["1970-01-01 00:00:01", "NaT"], dtype="datetime64[s]")
    tm.assert_index_equal(expected, result)

    # verify that no exception is raised even when errors='raise' is set
    result = to_datetime([1, ""], unit="s", errors="raise")
    tm.assert_index_equal(expected, result)


def test_to_datetime_monotonic_increasing_index(cache):
    # [FUNCTION DOC]
    # GH28238. Uses the cache fixture. A Series of yearly-start timestamps, sampled in shuffled order and indexed by a float index derived from the original positions, converts to a Series equal to the input (index preserved, order preserved) when the length is at least the cache-start threshold.
    # GH28238
    cstart = start_caching_at
    times = date_range(Timestamp("1980"), periods=cstart, freq="YS")
    times = times.to_frame(index=False, name="DT").sample(n=cstart, random_state=1)
    times.index = times.index.to_series().astype(float) / 1000
    result = to_datetime(times.iloc[:, 0], cache=cache)
    expected = times.iloc[:, 0]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "series_length",
    [40, start_caching_at, (start_caching_at + 1), (start_caching_at + 5)],
)
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length):
    # [FUNCTION DOC]
    # GH#45319. Parametrized over series lengths around the cache-start threshold. An object Series with one out-of-nanosecond-bounds UTC datetime (year 1446) followed by many in-bounds ones converts with utc true in coerce mode and in raise mode to the same microsecond UTC Series equal to the element-wise Timestamps.
    # GH#45319
    ser = Series(
        [datetime.fromisoformat("1446-04-12 00:00:00+00:00")]
        + ([datetime.fromisoformat("1991-10-20 00:00:00+00:00")] * series_length),
        dtype=object,
    )
    result1 = to_datetime(ser, errors="coerce", utc=True)

    expected1 = Series([Timestamp(x) for x in ser])
    assert expected1.dtype == "M8[us, UTC]"
    tm.assert_series_equal(result1, expected1)

    result3 = to_datetime(ser, errors="raise", utc=True)
    tm.assert_series_equal(result3, expected1)


def test_to_datetime_format_f_parse_nanos():
    # [FUNCTION DOC]
    # GH 48767. The string with nine fractional digits parsed with the fractional-seconds directive keeps the nanoseconds (123456 microseconds and 789 nanoseconds).
    # GH 48767
    timestamp = "15/02/2020 02:03:04.123456789"
    timestamp_format = "%d/%m/%Y %H:%M:%S.%f"
    result = to_datetime(timestamp, format=timestamp_format)
    expected = Timestamp(
        year=2020,
        month=2,
        day=15,
        hour=2,
        minute=3,
        second=4,
        microsecond=123456,
        nanosecond=789,
    )
    assert result == expected


def test_to_datetime_mixed_iso8601():
    # [FUNCTION DOC]
    # Issue 50411. A date-only and a date-time ISO string with the ISO 8601 format convert to midnight and 05:00.
    # https://github.com/pandas-dev/pandas/issues/50411
    result = to_datetime(["2020-01-01", "2020-01-01 05:00:00"], format="ISO8601")
    expected = DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 05:00:00"])
    tm.assert_index_equal(result, expected)


def test_to_datetime_mixed_other():
    # [FUNCTION DOC]
    # Issue 50411. Non-uniform formats ("01/11/2000" and "12 January 2000") with the mixed format convert to January 11 and January 12, 2000.
    # https://github.com/pandas-dev/pandas/issues/50411
    result = to_datetime(["01/11/2000", "12 January 2000"], format="mixed")
    expected = DatetimeIndex(["2000-01-11", "2000-01-12"])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("format", ["ISO8601", "mixed"])
def test_to_datetime_mixed_or_iso_exact(exact, format):
    # [FUNCTION DOC]
    # Parametrized over exact true or false and format ISO8601 or mixed. Combining exact with those formats raises a value error "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'".
    msg = "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'"
    with pytest.raises(ValueError, match=msg):
        to_datetime(["2020-01-01"], exact=exact, format=format)


def test_to_datetime_mixed_not_necessarily_iso8601_raise():
    # [FUNCTION DOC]
    # Issue 50411. With the ISO 8601 format, a non-ISO string (day-month-year dashes) raises a value error "Time data 01-01-2000 is not ISO8601 format".
    # https://github.com/pandas-dev/pandas/issues/50411
    with pytest.raises(ValueError, match="Time data 01-01-2000 is not ISO8601 format"):
        to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601")


def test_to_datetime_mixed_not_necessarily_iso8601_coerce():
    # [FUNCTION DOC]
    # Issue 50411. The same input in coerce mode gives the first date and NaT.
    # https://github.com/pandas-dev/pandas/issues/50411
    result = to_datetime(
        ["2020-01-01", "01-01-2000"], format="ISO8601", errors="coerce"
    )
    tm.assert_index_equal(result, DatetimeIndex(["2020-01-01 00:00:00", NaT]))


def test_to_datetime_iso8601_utc_single_naive():
    # [FUNCTION DOC]
    # GH#61389. A naive ISO string with utc true and the ISO 8601 format is taken as UTC.
    # GH#61389
    result = to_datetime("2023-10-15T14:30:00", utc=True, format="ISO8601")
    expected = Timestamp("2023-10-15 14:30:00+00:00")
    assert result == expected


def test_to_datetime_iso8601_utc_mixed_negative_offset():
    # [FUNCTION DOC]
    # GH#61389. A string with offset -12:00 and a naive string with utc true and ISO format: the first is shifted to UTC (22:30), the second is taken as UTC (14:30).
    # GH#61389
    data = ["2023-10-15T10:30:00-12:00", "2023-10-15T14:30:00"]
    result = to_datetime(data, utc=True, format="ISO8601")

    expected = DatetimeIndex(
        [Timestamp("2023-10-15 22:30:00+00:00"), Timestamp("2023-10-15 14:30:00+00:00")]
    )
    tm.assert_index_equal(result, expected)


def test_to_datetime_iso8601_utc_mixed_positive_offset():
    # [FUNCTION DOC]
    # GH#61389. Same with a +08:00 offset: the first is shifted to 02:30 UTC, the second stays at 14:30.
    # GH#61389
    data = ["2023-10-15T10:30:00+08:00", "2023-10-15T14:30:00"]
    result = to_datetime(data, utc=True, format="ISO8601")

    expected = DatetimeIndex(
        [Timestamp("2023-10-15 02:30:00+00:00"), Timestamp("2023-10-15 14:30:00+00:00")]
    )
    tm.assert_index_equal(result, expected)


def test_to_datetime_iso8601_utc_mixed_both_offsets():
    # [FUNCTION DOC]
    # GH#61389. Strings with +08:00, -05:00 and no offset with utc true and ISO format give 02:30, 17:30 and 14:30 UTC.
    # GH#61389
    data = [
        "2023-10-15T10:30:00+08:00",
        "2023-10-15T12:30:00-05:00",
        "2023-10-15T14:30:00",
    ]
    result = to_datetime(data, utc=True, format="ISO8601")

    expected = DatetimeIndex(
        [
            Timestamp("2023-10-15 02:30:00+00:00"),
            Timestamp("2023-10-15 17:30:00+00:00"),
            Timestamp("2023-10-15 14:30:00+00:00"),
        ]
    )
    tm.assert_index_equal(result, expected)


def test_unknown_tz_raises():
    # [FUNCTION DOC]
    # GH#18702, GH#51476. A string with the unrecognized time zone name FAKE raises a value error "un-recognized timezone" from the Timestamp constructor, the scalar conversion and the list conversion.
    # GH#18702, GH#51476
    dtstr = "2014 Jan 9 05:15 FAKE"
    msg = '.*un-recognized timezone "FAKE".'
    with pytest.raises(ValueError, match=msg):
        Timestamp(dtstr)

    with pytest.raises(ValueError, match=msg):
        to_datetime(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime([dtstr])


def test_unformatted_input_raises():
    # [FUNCTION DOC]
    # A Series of valid ISO dates at the cache-start threshold length plus one invalid entry "N", converted with exact format and caching on, raises a value error that "N" does not match the format.
    valid, invalid = "2024-01-01", "N"
    ser = Series([valid] * start_caching_at + [invalid])
    msg = 'time data "N" doesn\'t match format "%Y-%m-%d"'

    with pytest.raises(ValueError, match=msg):
        to_datetime(ser, format="%Y-%m-%d", exact=True, cache=True)


def test_from_numeric_arrow_dtype(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # GH 52425. Skipped without pyarrow. Parametrized over numeric extension dtypes. A Series of 1 and 2 with the pyarrow dtype converts to a nanosecond datetime Series (the numbers are nanoseconds since epoch).
    # GH 52425
    pytest.importorskip("pyarrow")
    ser = Series([1, 2], dtype=f"{any_numeric_ea_dtype.lower()}[pyarrow]")
    result = to_datetime(ser)
    expected = Series([1, 2], dtype="datetime64[ns]")
    tm.assert_series_equal(result, expected)


def test_to_datetime_with_empty_str_utc_false_format_mixed():
    # [FUNCTION DOC]
    # GH 50887. A UTC-aware string and an empty string with the mixed format give a microsecond UTC index with NaT; the plain conversion and the index constructor give the same.
    # GH 50887
    vals = ["2020-01-01 00:00+00:00", ""]
    result = to_datetime(vals, format="mixed")
    expected = Index([Timestamp("2020-01-01 00:00+00:00"), "NaT"], dtype="M8[us, UTC]")
    tm.assert_index_equal(result, expected)

    # Check that a couple of other similar paths work the same way
    alt = to_datetime(vals)
    tm.assert_index_equal(alt, expected)
    alt2 = DatetimeIndex(vals)
    tm.assert_index_equal(alt2, expected)


def test_to_datetime_with_empty_str_utc_false_offsets_and_format_mixed():
    # [FUNCTION DOC]
    # GH#50887, GH#57275. Two strings with different offsets and an empty string with the mixed format raise a value error "Mixed timezones detected. Pass utc=True in to_datetime".
    # GH#50887, GH#57275
    msg = "Mixed timezones detected. Pass utc=True in to_datetime"

    with pytest.raises(ValueError, match=msg):
        to_datetime(
            ["2020-01-01 00:00+00:00", "2020-01-01 00:00+02:00", ""], format="mixed"
        )


def test_to_datetime_mixed_tzs_mixed_types():
    # [FUNCTION DOC]
    # GH#55793, GH#55693. A list with a Pacific Timestamp and a string with a +01 offset raises a value error about mixed time zones (with advice to pass utc=True or tz='UTC'), for the default conversion, the mixed format, and the index constructor.
    # GH#55793, GH#55693 mismatched tzs but one is str and other is
    #  datetime object
    ts = Timestamp("2016-01-02 03:04:05", tz="US/Pacific")
    dtstr = "2023-10-30 15:06+01"
    arr = [ts, dtstr]

    msg = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' "
        "in DatetimeIndex to convert to a common timezone"
    )
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr)
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr, format="mixed")
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(arr)


def test_to_datetime_mixed_types_matching_tzs():
    # [FUNCTION DOC]
    # GH#55793. A Timestamp with a -07:00 offset and its string form (same offset) convert to a two-element index of that Timestamp, in either order, with the mixed format, and through the index constructor.
    # GH#55793
    dtstr = "2023-11-01 09:22:03-07:00"
    ts = Timestamp(dtstr)
    arr = [ts, dtstr]
    res1 = to_datetime(arr)
    res2 = to_datetime(arr[::-1])[::-1]
    res3 = to_datetime(arr, format="mixed")
    res4 = DatetimeIndex(arr)

    expected = DatetimeIndex([ts, ts])
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


dtstr = "2020-01-01 00:00+00:00"
ts = Timestamp(dtstr)


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
@pytest.mark.parametrize(
    "aware_val",
    [dtstr, Timestamp(dtstr)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize(
    "naive_val",
    [dtstr[:-6], ts.tz_localize(None), ts.date(), ts.asm8, ts.value, float(ts.value)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize("naive_first", [True, False])
def test_to_datetime_mixed_awareness_mixed_types(aware_val, naive_val, naive_first):
    # [FUNCTION DOC]
    # GH#55793, GH#55693, GH#57275. Filters the could-not-infer-format user warning. Parametrized over an aware value (string or Timestamp), a naive value (string without offset, naive Timestamp, date, NumPy datetime, integer, float), and the order of the two. Builds a list of the two values plus an empty string and checks a decision tree of expected value errors by value types and order: when the first non-empty value is not a string the mixed-timezones message is expected (or tz-aware datetime / cannot mix messages for particular type combinations); numeric values after an aware string produce a does-not-match-format error (also with utc true); string pairs produce does-not-match-format or unconverted-data-remains errors depending on order; otherwise the mixed message, and utc true succeeds. The mixed format and the index constructor then raise the mixed message (or the particular tz-aware messages for Timestamp-first combinations).
    # Module-level values defined just before this test: a UTC-offset datetime string and its Timestamp, used to build the parametrization.
    # GH#55793, GH#55693, GH#57275
    # Empty string parses to NaT
    vals = [aware_val, naive_val, ""]

    vec = vals
    if naive_first:
        # alas, the behavior is order-dependent, so we test both ways
        vec = [naive_val, aware_val, ""]

    # both_strs-> paths that were previously already deprecated with warning
    #  issued in _array_to_datetime_object
    both_strs = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric = isinstance(naive_val, (int, float))
    both_datetime = isinstance(naive_val, datetime) and isinstance(aware_val, datetime)

    mixed_msg = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' "
        "in DatetimeIndex to convert to a common timezone"
    )

    first_non_null = next(x for x in vec if x != "")
    # if first_non_null is a not a string, _guess_datetime_format_for_array
    #  doesn't guess a format so we don't go through array_strptime
    if not isinstance(first_non_null, str):
        # that case goes through array_strptime which has different behavior
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)

        # No warning/error with utc=True
        to_datetime(vec, utc=True)

    elif has_numeric and vec.index(aware_val) < vec.index(naive_val):
        msg = "time data .* doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)

    elif both_strs and vec.index(aware_val) < vec.index(naive_val):
        msg = r"time data \"2020-01-01 00:00\" doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)

    elif both_strs and vec.index(naive_val) < vec.index(aware_val):
        msg = "unconverted data remains when parsing with format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)

    else:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)

        # No warning/error with utc=True
        to_datetime(vec, utc=True)

    if both_strs:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, format="mixed")
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(vec)
    else:
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format="mixed")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format="mixed")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)


def test_to_datetime_wrapped_datetime64_ps():
    # [FUNCTION DOC]
    # GH#60341. A list with a picosecond-resolution NumPy datetime converts to a nanosecond index (precision truncated to 1970-01-01 00:00:01.901901901).
    # GH#60341
    result = to_datetime([np.datetime64(1901901901901, "ps")])
    expected = DatetimeIndex(
        ["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None
    )
    tm.assert_index_equal(result, expected)


def test_to_datetime_lxml_elementunicoderesult_with_format(cache):
    # [FUNCTION DOC]
    # Skipped without lxml. Uses the cache fixture. A string subclass instance obtained from an XML text node, in a Series, converts with an explicit format to the equal Timestamp.
    etree = pytest.importorskip("lxml.etree")

    s = "2025-02-05 16:59:57"
    node = etree.XML(f"<date>{s}</date>")
    val = node.xpath("/date/node()")[0]  # _ElementUnicodeResult

    out = to_datetime(Series([val]), format="%Y-%m-%d %H:%M:%S", cache=cache)
    assert out.iloc[0] == Timestamp(s)


def test_to_datetime_missing_component_no_runtime_warning():
    # [FUNCTION DOC]
    # A frame with year, month, day and hour columns, where the hour has a NaN, converts without emitting any warning; the row with the missing component is NaT.
    df = DataFrame(
        {
            "year": [2023, 2023],
            "month": [12, 2],
            "day": [1, 2],
            "hour": [2, np.nan],
        }
    )

    with tm.assert_produces_warning(None):
        result = to_datetime(df)

    assert result.iloc[1] is NaT
