"""
Tests for Timestamp parsing, aimed at pandas/_libs/tslibs/parsing.pyx
"""

# [MODULE DOC]
# Tests for timestamp string parsing in the compiled parsing library (and, for convenience, the timestamp constructor): parsing with resolution, quarter
# strings (with dashes and fiscal frequencies), the check for whether a string looks like a datetime, guessing a strptime format from a string (valid,
# invalid, day-first, locale, wrong types, missing padding), parsing arrays of dates with a custom parser, the ISO-format check, and agreement with dateutil under display options for day-first and year-first.
# Imports platform flags and test decorators; no import-time side effects.

from datetime import datetime
import re

from dateutil.parser import parse as du_parse
import numpy as np
import pytest

from pandas._libs.tslibs import (
    parsing,
    strptime,
)
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import (
    ISMUSL,
    WASM,
    is_platform_windows,
)
import pandas.util._test_decorators as td

# Usually we wouldn't want this import in this test file (which is targeted at
#  tslibs.parsing), but it is convenient to test the Timestamp constructor at
#  the same time as the other parsing functions.
from pandas import (
    Timestamp,
    option_context,
)
import pandas._testing as tm


@pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
@pytest.mark.skipif(
    is_platform_windows() or ISMUSL,
    reason="TZ setting incorrect on Windows and MUSL Linux",
)
def test_parsing_tzlocal_deprecated():
    # [FUNCTION DOC]
    # Regression test (GH#50791). Skipped on WebAssembly (no timezone setting), Windows and musl Linux. With the local timezone set to US/Eastern, parsing a string with the EST abbreviation through the resolution parser, the Python-visible string parser, and the timestamp constructor each raise a value error saying parsing as the local timezone is no longer supported (or that the string included an unrecognized timezone).
    # KNOWN ISSUE: the name says deprecated but the behavior being tested is a hard error, as the message itself states the usage is no longer supported.
    # GH#50791
    msg = "|".join(
        [
            r"Parsing 'EST' as tzlocal \(dependent on system timezone\) "
            r"is no longer supported\. "
            "Pass the 'tz' keyword or call tz_localize after construction instead",
            ".*included an un-recognized timezone",
        ]
    )
    dtstr = "Jan 15 2004 03:00 EST"

    with tm.set_timezone("US/Eastern"):
        with pytest.raises(ValueError, match=msg):
            parse_datetime_string_with_reso(dtstr)

        with pytest.raises(ValueError, match=msg):
            parsing.py_parse_datetime_string(dtstr)

        with pytest.raises(ValueError, match=msg):
            Timestamp(dtstr)


def test_parse_datetime_string_with_reso():
    # [FUNCTION DOC]
    # Parsing the quarter string "4Q1984" and its lowercase form gives the same parsed value and the same resolution.
    (parsed, reso) = parse_datetime_string_with_reso("4Q1984")
    (parsed_lower, reso_lower) = parse_datetime_string_with_reso("4q1984")

    assert reso == reso_lower
    assert parsed == parsed_lower


def test_parse_datetime_string_with_reso_nanosecond_reso():
    # [FUNCTION DOC]
    # Regression test (GH#46811): parsing a string with nine fractional digits reports resolution nanosecond.
    # GH#46811
    parsed, reso = parse_datetime_string_with_reso("2022-04-20 09:19:19.123456789")
    assert reso == "nanosecond"


def test_parse_datetime_string_with_reso_invalid_type():
    # [FUNCTION DOC]
    # Passing a tuple instead of a string raises a type error saying the argument has an incorrect type (expected str, got tuple), rather than returning the input.
    # Raise on invalid input, don't just return it
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((4, 5))


@pytest.mark.parametrize(
    "dashed,normal", [("1988-Q2", "1988Q2"), ("2Q-1988", "2Q1988")]
)
def test_parse_time_quarter_with_dash(dashed, normal):
    # [FUNCTION DOC]
    # Regression test (gh-9688), parametrized over dashed and undashed quarter strings: both parse to the same value and resolution.
    # see gh-9688
    (parsed_dash, reso_dash) = parse_datetime_string_with_reso(dashed)
    (parsed, reso) = parse_datetime_string_with_reso(normal)

    assert parsed_dash == parsed
    assert reso_dash == reso


@pytest.mark.parametrize("dashed", ["-2Q1992", "2-Q1992", "4-4Q1992"])
def test_parse_time_quarter_with_dash_error(dashed):
    # [FUNCTION DOC]
    # Malformed dashed quarter strings raise the date parse error saying the datetime string format is unknown and cannot be parsed.
    msg = f"Unknown datetime string format, unable to parse: {dashed}"

    with pytest.raises(parsing.DateParseError, match=msg):
        parse_datetime_string_with_reso(dashed)


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("123.1234", False),
        ("-50000", False),
        ("999", False),
        ("m", False),
        ("T", False),
        ("Mon Sep 16, 2013", True),
        ("2012-01-01", True),
        ("01/01/2012", True),
        ("01012012", True),
        ("0101", True),
        ("1-1", True),
    ],
)
def test_does_not_convert_mixed_integer(date_string, expected):
    # [FUNCTION DOC]
    # Parametrized over strings with the expected truth value: the private looks-like-datetime check returns exactly False for number-like or single-letter strings and exactly True for date-like text and numeric date forms.
    assert parsing._does_string_look_like_datetime(date_string) is expected


@pytest.mark.parametrize(
    "date_str,kwargs,msg",
    [
        (
            "2013Q5",
            {},
            (
                "Incorrect quarterly string is given, "
                "quarter must be between 1 and 4: 2013Q5"
            ),
        ),
        # see gh-5418
        (
            "2013Q1",
            {"freq": "INVLD-L-DEC-SAT"},
            ("Unable to retrieve month information from given freq: INVLD-L-DEC-SAT"),
        ),
    ],
)
def test_parsers_quarterly_with_freq_error(date_str, kwargs, msg):
    # [FUNCTION DOC]
    # Parametrized over invalid quarter strings: a quarter number of 5 gives the date parse error that the quarter must be between 1 and 4; (gh-5418) an invalid frequency string gives the error that the month information cannot be retrieved from the frequency.
    with pytest.raises(parsing.DateParseError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str, **kwargs)


@pytest.mark.parametrize(
    "date_str,freq,expected",
    [
        ("2013Q2", None, datetime(2013, 4, 1)),
        ("2013Q2", "Y-APR", datetime(2012, 8, 1)),
        ("2013-Q2", "Y-DEC", datetime(2013, 4, 1)),
    ],
)
def test_parsers_quarterly_with_freq(date_str, freq, expected):
    # [FUNCTION DOC]
    # Parametrized over quarter strings and fiscal-year frequencies (none, anchored April, anchored December); the first returned value equals the expected first day of the quarter's first month.
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq=freq)
    assert result == expected


@pytest.mark.parametrize(
    "date_str", ["2Q 2005", "2Q-200Y", "2Q-200", "22Q2005", "2Q200.", "6Q-20"]
)
def test_parsers_quarter_invalid(date_str):
    # [FUNCTION DOC]
    # Parametrized over malformed quarter strings. The string with quarter 6 gives the error that the quarter must be between 1 and 4; the others give the error that the format is unknown and cannot be parsed. Both are value errors.
    if date_str == "6Q-20":
        msg = (
            "Incorrect quarterly string is given, quarter "
            f"must be between 1 and 4: {date_str}"
        )
    else:
        msg = f"Unknown datetime string format, unable to parse: {date_str}"

    with pytest.raises(ValueError, match=msg):
        parsing.parse_datetime_string_with_reso(date_str)


@pytest.mark.parametrize(
    "date_str,expected",
    [("201101", datetime(2011, 1, 1, 0, 0)), ("200005", datetime(2000, 5, 1, 0, 0))],
)
def test_parsers_month_freq(date_str, expected):
    # [FUNCTION DOC]
    # Parametrized over six-digit year-month strings; parsed with a month-end frequency they return the first day of that month.
    result, _ = parsing.parse_datetime_string_with_reso(date_str, freq="ME")
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("20111230", "%Y%m%d"),
        ("201112300000", "%Y%m%d%H%M"),
        ("20111230000000", "%Y%m%d%H%M%S"),
        ("20111230T00", "%Y%m%dT%H"),
        ("20111230T0000", "%Y%m%dT%H%M"),
        ("20111230T000000", "%Y%m%dT%H%M%S"),
        ("2011-12-30", "%Y-%m-%d"),
        ("2011", "%Y"),
        ("2011-01", "%Y-%m"),
        ("30-12-2011", "%d-%m-%Y"),
        ("2011-12-30 00:00:00", "%Y-%m-%d %H:%M:%S"),
        ("2011-12-30T00:00:00", "%Y-%m-%dT%H:%M:%S"),
        ("2011-12-30T00:00:00UTC", "%Y-%m-%dT%H:%M:%S%Z"),
        ("2011-12-30T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+9", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+090", None),
        ("2011-12-30T00:00:00+0900", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00-0900", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:00", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:000", None),
        ("2011-12-30T00:00:00+9:0", "%Y-%m-%dT%H:%M:%S%z"),
        ("2011-12-30T00:00:00+09:", None),
        ("2011-12-30T00:00:00.000000UTC", "%Y-%m-%dT%H:%M:%S.%f%Z"),
        ("2011-12-30T00:00:00.000000Z", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+9", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+090", None),
        ("2011-12-30T00:00:00.000000+0900", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000-0900", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:00", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:000", None),
        ("2011-12-30T00:00:00.000000+9:0", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("2011-12-30T00:00:00.000000+09:", None),
        ("2011-12-30 00:00:00.000000", "%Y-%m-%d %H:%M:%S.%f"),
        ("Tue 24 Aug 2021 01:30:48", "%a %d %b %Y %H:%M:%S"),
        ("Tuesday 24 Aug 2021 01:30:48", "%A %d %b %Y %H:%M:%S"),
        ("Tue 24 Aug 2021 01:30:48 AM", "%a %d %b %Y %I:%M:%S %p"),
        ("Tuesday 24 Aug 2021 01:30:48 AM", "%A %d %b %Y %I:%M:%S %p"),
        ("27.03.2003 14:55:00.000", "%d.%m.%Y %H:%M:%S.%f"),  # GH50317
        ("2023-11-09T20:23:46Z", "%Y-%m-%dT%H:%M:%S%z"),  # GH57452
    ],
)
def test_guess_datetime_format_with_parseable_formats(string, fmt):
    # [FUNCTION DOC]
    # Skipped unless the locale is US. Parametrized over many strings and their expected format strings (compact, dashed, T-separated, with timezone names or numeric offsets of various lengths, with fractional seconds, weekday names and AM/PM, dotted day-first forms; some malformed offsets expect none). Asserts the guessed format equals the expected value, allowing a user warning to be produced when the format has day before month.
    with tm.maybe_produces_warning(
        UserWarning, fmt is not None and re.search(r"%d.*%m", fmt)
    ):
        result = parsing.guess_datetime_format(string)
    assert result == fmt


@pytest.mark.parametrize("dayfirst,expected", [(True, "%d/%m/%Y"), (False, "%m/%d/%Y")])
def test_guess_datetime_format_with_dayfirst(dayfirst, expected):
    # [FUNCTION DOC]
    # For the ambiguous string 01/01/2011, the guessed format is day/month/year when day-first is true and month/day/year when false.
    ambiguous_string = "01/01/2011"
    result = parsing.guess_datetime_format(ambiguous_string, dayfirst=dayfirst)
    assert result == expected


@td.skip_if_not_us_locale
@pytest.mark.parametrize(
    "string,fmt",
    [
        ("30/Dec/2011", "%d/%b/%Y"),
        ("30/December/2011", "%d/%B/%Y"),
        ("30/Dec/2011 00:00:00", "%d/%b/%Y %H:%M:%S"),
    ],
)
def test_guess_datetime_format_with_locale_specific_formats(string, fmt):
    # [FUNCTION DOC]
    # Skipped unless the locale is US. Strings with abbreviated or full month names (and one with a time) yield the matching format strings.
    result = parsing.guess_datetime_format(string)
    assert result == fmt


@pytest.mark.parametrize(
    "invalid_dt",
    [
        "01/2013",
        "12:00:00",
        "1/1/1/1",
        "this_is_not_a_datetime",
        "51a",
        "13/2019",
        "202001",  # YYYYMM isn't ISO8601
        "2020/01",  # YYYY/MM isn't ISO8601 either
        "87156549591102612381000001219H5",
    ],
)
def test_guess_datetime_format_invalid_inputs(invalid_dt):
    # [FUNCTION DOC]
    # Parametrized over strings that lack a year, month or day, are not datetime-like, or are six-digit/slash year-month forms; the guesser returns none.
    # A datetime string must include a year, month and a day for it to be
    # guessable, in addition to being a string that looks like a datetime.
    assert parsing.guess_datetime_format(invalid_dt) is None


@pytest.mark.parametrize("invalid_type_dt", [9, datetime(2011, 1, 1)])
def test_guess_datetime_format_wrong_type_inputs(invalid_type_dt):
    # [FUNCTION DOC]
    # Parametrized over an integer and a datetime object. Passing a non-string raises a type error saying the argument has an incorrect type (expected str).
    # KNOWN ISSUE: the explanatory comment in the source (a datetime string must include a year, month and day) is copied from the invalid-input test and does not describe this type check.
    # A datetime string must include a year, month and a day for it to be
    # guessable, in addition to being a string that looks like a datetime.
    with pytest.raises(
        TypeError,
        match=r"^Argument 'dt_str' has incorrect type \(expected str, got .*\)$",
    ):
        parsing.guess_datetime_format(invalid_type_dt)


@pytest.mark.parametrize(
    "string,fmt,dayfirst,warning",
    [
        ("2011-1-1", "%Y-%m-%d", False, None),
        ("2011-1-1", "%Y-%d-%m", True, None),
        ("1/1/2011", "%m/%d/%Y", False, None),
        ("1/1/2011", "%d/%m/%Y", True, None),
        ("30-1-2011", "%d-%m-%Y", False, UserWarning),
        ("30-1-2011", "%d-%m-%Y", True, None),
        ("2011-1-1 0:0:0", "%Y-%m-%d %H:%M:%S", False, None),
        ("2011-1-1 0:0:0", "%Y-%d-%m %H:%M:%S", True, None),
        ("2011-1-3T00:00:0", "%Y-%m-%dT%H:%M:%S", False, None),
        ("2011-1-3T00:00:0", "%Y-%d-%mT%H:%M:%S", True, None),
        ("2011-1-1 00:00:00", "%Y-%m-%d %H:%M:%S", False, None),
        ("2011-1-1 00:00:00", "%Y-%d-%m %H:%M:%S", True, None),
    ],
)
def test_guess_datetime_format_no_padding(string, fmt, dayfirst, warning):
    # [FUNCTION DOC]
    # Regression test (gh-11142). Parametrized over strings without zero padding, the expected format, the day-first flag and the expected warning (a user warning only for a day-first-looking string when day-first is false). Asserts the warning (matching a message suggesting passing day-first or specifying a format) and that the guessed format equals the expected one.
    # see gh-11142
    msg = (
        rf"Parsing dates in {fmt} format when dayfirst=False \(the default\) "
        "was specified. "
        "Pass `dayfirst=True` or specify a format to silence this warning."
    )
    with tm.assert_produces_warning(warning, match=msg):
        result = parsing.guess_datetime_format(string, dayfirst=dayfirst)
    assert result == fmt


def test_try_parse_dates():
    # [FUNCTION DOC]
    # Parses an object array of three slash-separated date strings with a custom parser built on dateutil with day-first enabled, and asserts the resulting object array equals the dateutil parse of each string.
    arr = np.array(["5/1/2000", "6/1/2000", "7/1/2000"], dtype=object)
    result = parsing.try_parse_dates(arr, parser=lambda x: du_parse(x, dayfirst=True))

    expected = np.array([du_parse(d, dayfirst=True) for d in arr])
    tm.assert_numpy_array_equal(result, expected)


def test_parse_datetime_string_with_reso_check_instance_type_raise_exception():
    # [FUNCTION DOC]
    # Issue 20684: a tuple input raises a type error (expected str, got tuple); the string "2019" parses to January 1 2019 with resolution year.
    # issue 20684
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((1, 2, 3))

    result = parse_datetime_string_with_reso("2019")
    expected = (datetime(2019, 1, 1), "year")
    assert result == expected


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("%Y %m %d %H:%M:%S", True),
        ("%Y/%m/%d %H:%M:%S", True),
        (r"%Y\%m\%d %H:%M:%S", True),
        ("%Y-%m-%d %H:%M:%S", True),
        ("%Y.%m.%d %H:%M:%S", True),
        ("%Y%m%d %H:%M:%S", True),
        ("%Y-%m-%dT%H:%M:%S", True),
        ("%Y-%m-%dT%H:%M:%S%z", True),
        ("%Y-%m-%dT%H:%M:%S%Z", False),
        ("%Y-%m-%dT%H:%M:%S.%f", True),
        ("%Y-%m-%dT%H:%M:%S.%f%z", True),
        ("%Y-%m-%dT%H:%M:%S.%f%Z", False),
        ("%Y%m%d", True),
        ("%Y%m", False),
        ("%Y", True),
        ("%Y-%m-%d", True),
        ("%Y-%m", True),
    ],
)
def test_is_iso_format(fmt, expected):
    # [FUNCTION DOC]
    # Regression test (gh-41047), parametrized over format strings: the private check says whether the format is ISO-like. Formats with separators of space, slash, backslash, dash or dot or none, numeric offset specifiers and fractional seconds are ISO; those with timezone-name specifiers and year-month without separator are not.
    # see gh-41047
    result = strptime._test_format_is_iso(fmt)
    assert result == expected


@pytest.mark.parametrize(
    "input",
    [
        "2018-01-01T00:00:00.123456789",
        "2018-01-01T00:00:00.123456",
        "2018-01-01T00:00:00.123",
    ],
)
def test_guess_datetime_format_f(input):
    # [FUNCTION DOC]
    # Regression test (issue 49043): ISO strings with 3, 6 or 9 fractional digits all guess the format with the generic fractional-second specifier.
    # https://github.com/pandas-dev/pandas/issues/49043
    result = parsing.guess_datetime_format(input)
    expected = "%Y-%m-%dT%H:%M:%S.%f"
    assert result == expected


def _helper_hypothesis_delimited_date(call, date_string, **kwargs):
    # [FUNCTION DOC]
    # HELPER: for the day-first/year-first comparison tests.
    # Calls the given parser function with the date string and keyword arguments. Returns a pair of (error message or none, result or none): when the call raises a value error its message string is returned with no result; otherwise no message and the result.
    msg, result = None, None
    try:
        result = call(date_string, **kwargs)
    except ValueError as err:
        msg = str(err)
    return msg, result


@pytest.mark.parametrize("input", ["21-01-01", "01-01-21"])
@pytest.mark.parametrize("dayfirst", [True, False])
def test_parse_datetime_string_with_reso_dayfirst(dayfirst, input):
    # [FUNCTION DOC]
    # Parametrized over the day-first option and two ambiguous numeric strings. With the display date-day-first option set accordingly, the pandas parser and dateutil (called with a default of year 1 and the same day-first, year-first off) must raise the same error message (or none) and the first returned value of the pandas parser must equal the dateutil value.
    with option_context("display.date_dayfirst", dayfirst):
        except_out_dateutil, result = _helper_hypothesis_delimited_date(
            parsing.parse_datetime_string_with_reso, input
        )

        except_in_dateutil, expected = _helper_hypothesis_delimited_date(
            du_parse,
            input,
            default=datetime(1, 1, 1),
            dayfirst=dayfirst,
            yearfirst=False,
        )
        assert except_out_dateutil == except_in_dateutil
        assert result[0] == expected


@pytest.mark.parametrize("input", ["21-01-01", "01-01-21"])
@pytest.mark.parametrize("yearfirst", [True, False])
def test_parse_datetime_string_with_reso_yearfirst(yearfirst, input):
    # [FUNCTION DOC]
    # Same comparison as the day-first test but with the display date-year-first option set and dateutil called with year-first, day-first off.
    with option_context("display.date_yearfirst", yearfirst):
        except_out_dateutil, result = _helper_hypothesis_delimited_date(
            parsing.parse_datetime_string_with_reso, input
        )
        except_in_dateutil, expected = _helper_hypothesis_delimited_date(
            du_parse,
            input,
            default=datetime(1, 1, 1),
            dayfirst=False,
            yearfirst=yearfirst,
        )
        assert except_out_dateutil == except_in_dateutil
        assert result[0] == expected
