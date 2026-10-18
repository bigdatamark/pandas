# [MODULE DOC]
# Tests for the function converting frequency strings and other objects (offsets, time deltas, tuples) to offset objects: compound strings, negative and signed counts,
# invalid strings, whitespace handling, leading zeros, anchored shortcuts, conversion from timedeltas, and the rejected or deprecated lowercase and uppercase aliases.

import re

import pytest

from pandas._libs.tslibs import (
    Timedelta,
    offsets,
    to_offset,
)
from pandas.errors import Pandas4Warning

import pandas._testing as tm


@pytest.mark.parametrize(
    "freq_input,expected",
    [
        (to_offset("10us"), offsets.Micro(10)),
        (offsets.Hour(), offsets.Hour()),
        ("2h30min", offsets.Minute(150)),
        ("2h 30min", offsets.Minute(150)),
        ("2h30min15s", offsets.Second(150 * 60 + 15)),
        ("2h 60min", offsets.Hour(3)),
        ("2h 20.5min", offsets.Second(8430)),
        ("1.5min", offsets.Second(90)),
        ("0.5s", offsets.Milli(500)),
        ("15ms500us", offsets.Micro(15500)),
        ("10s75ms", offsets.Milli(10075)),
        ("1s0.25ms", offsets.Micro(1000250)),
        ("2800ns", offsets.Nano(2800)),
        ("2SME", offsets.SemiMonthEnd(2)),
        ("2SME-16", offsets.SemiMonthEnd(2, day_of_month=16)),
        ("2SMS-14", offsets.SemiMonthBegin(2, day_of_month=14)),
        ("2SMS-15", offsets.SemiMonthBegin(2)),
        ("LWOM-MON", offsets.LastWeekOfMonth()),
    ],
)
def test_to_offset(freq_input, expected):
    # [FUNCTION DOC]
    # Parametrized over inputs (an already-built offset, an offset instance, many compound and fractional strings such as hours with minutes, seconds with milliseconds, semi-monthly strings with day suffixes, and last-week-of-month) and the expected offset. The converted result equals the expected offset (combining units into the finest unit needed).
    result = to_offset(freq_input)
    assert result == expected


@pytest.mark.parametrize(
    "freqstr,expected", [("-1s", -1), ("-2SME", -2), ("-1SMS", -1), ("-5min10s", -310)]
)
def test_to_offset_negative(freqstr, expected):
    # [FUNCTION DOC]
    # Parametrized over strings with negative counts, simple or compound; the resulting offset's count equals the expected integer (compound negatives are totaled in the finest unit).
    result = to_offset(freqstr)
    assert result.n == expected


@pytest.mark.filterwarnings("ignore:.*'m' is deprecated.*:FutureWarning")
@pytest.mark.parametrize(
    "freqstr",
    [
        "2h20m",
        "us1",
        "-us",
        "3us1",
        "-2-3us",
        "-2D:3h",
        "1.5.0s",
        "2SMS-15-15",
        "2SMS-15D",
        "100foo",
        # Invalid leading +/- signs.
        "+-1D",
        "-+1h",
        "+1",
        "-7",
        "+D",
        "-m",
        # Invalid shortcut anchors.
        "SME-0",
        "SME-28",
        "SME-29",
        "SME-FOO",
        "BSM",
        "SME--1",
        "SMS-1",
        "SMS-28",
        "SMS-30",
        "SMS-BAR",
        "SMS-BYR",
        "BSMS",
        "SMS--2",
    ],
)
def test_to_offset_invalid(freqstr):
    # [FUNCTION DOC]
    # Regression test (gh-13930), ignoring the deprecation warning about the minute alias. Parametrized over malformed strings (bad unit order, repeated suffixes, bad anchors, bad signs, unknown units, invalid semi-month anchors). Raises a value error with the invalid frequency message quoting the string (escaped for regex).
    # see gh-13930

    # We escape string because some of our
    # inputs contain regex special characters.
    msg = re.escape(f"Invalid frequency: {freqstr}")
    with pytest.raises(ValueError, match=msg):
        to_offset(freqstr)


def test_to_offset_no_evaluate():
    # [FUNCTION DOC]
    # Passing a pair of empty strings raises a type error whose message is the string form of that tuple.
    msg = str(("", ""))
    with pytest.raises(TypeError, match=msg):
        to_offset(("", ""))


def test_to_offset_tuple_unsupported():
    # [FUNCTION DOC]
    # Passing a tuple of count and unit raises a type error telling the caller to pass a string instead.
    with pytest.raises(TypeError, match="pass as a string instead"):
        to_offset((5, "T"))


@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("2D 3h", offsets.Hour(51)),
        ("2 D3 h", offsets.Hour(51)),
        ("2 D 3 h", offsets.Hour(51)),
        ("  2 D 3 h  ", offsets.Hour(51)),
        ("   h    ", offsets.Hour()),
        (" 3  h    ", offsets.Hour(3)),
    ],
)
def test_to_offset_whitespace(freqstr, expected):
    # [FUNCTION DOC]
    # Parametrized over strings with various embedded and surrounding whitespace; the result equals the expected hour offset (days converted to hours).
    result = to_offset(freqstr)
    assert result == expected


@pytest.mark.parametrize(
    "freqstr,expected", [("00h 00min 01s", 1), ("-00h 03min 14s", -194)]
)
def test_to_offset_leading_zero(freqstr, expected):
    # [FUNCTION DOC]
    # Parametrized over strings with zero-padded components; the offset's count equals the expected total in seconds (positive or negative).
    result = to_offset(freqstr)
    assert result.n == expected


@pytest.mark.parametrize(
    "freqstr,expected,wrn", [("+1d", 1, Pandas4Warning), ("+2h30min", 150, None)]
)
def test_to_offset_leading_plus(freqstr, expected, wrn):
    # [FUNCTION DOC]
    # Parametrized over a plus-signed day string (which warns with the future-removal warning that the lowercase d alias is deprecated) and a plus-signed hours-minutes string (no warning). The count equals the expected value; the warning is asserted via the warning helper.
    msg = "'d' is deprecated and will be removed in a future version."

    with tm.assert_produces_warning(wrn, match=msg):
        result = to_offset(freqstr)
    assert result.n == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"days": 1, "seconds": 1}, offsets.Second(86401)),
        ({"days": -1, "seconds": 1}, offsets.Second(-86399)),
        ({"hours": 1, "minutes": 10}, offsets.Minute(70)),
        ({"hours": 1, "minutes": -10}, offsets.Minute(50)),
        ({"weeks": 1}, offsets.Hour(168)),
        ({"hours": 1}, offsets.Hour(1)),
        ({"hours": 1}, to_offset("60min")),
        ({"microseconds": 1}, offsets.Micro(1)),
        ({"microseconds": 0}, offsets.Nano(0)),
    ],
)
def test_to_offset_pd_timedelta(kwargs, expected):
    # [FUNCTION DOC]
    # Regression test (gh-9064). Parametrized over keyword arguments for a Timedelta and the expected offset; converting the timedelta gives an offset in the largest unit that represents it exactly (zero microseconds gives a zero nano tick).
    # see gh-9064
    td = Timedelta(**kwargs)
    result = to_offset(td)
    assert result == expected


@pytest.mark.parametrize(
    "shortcut,expected",
    [
        ("W", offsets.Week(weekday=6)),
        ("W-SUN", offsets.Week(weekday=6)),
        ("QE", offsets.QuarterEnd(startingMonth=12)),
        ("QE-DEC", offsets.QuarterEnd(startingMonth=12)),
        ("QE-MAY", offsets.QuarterEnd(startingMonth=5)),
        ("SME", offsets.SemiMonthEnd(day_of_month=15)),
        ("SME-15", offsets.SemiMonthEnd(day_of_month=15)),
        ("SME-1", offsets.SemiMonthEnd(day_of_month=1)),
        ("SME-27", offsets.SemiMonthEnd(day_of_month=27)),
        ("SMS-2", offsets.SemiMonthBegin(day_of_month=2)),
        ("SMS-27", offsets.SemiMonthBegin(day_of_month=27)),
    ],
)
def test_anchored_shortcuts(shortcut, expected):
    # [FUNCTION DOC]
    # Parametrized over shortcut strings (weekly, quarter end with and without month, semi-month end and begin with day numbers) and expected offsets; the converted result equals the expected one.
    result = to_offset(shortcut)
    assert result == expected


@pytest.mark.parametrize(
    "freq_depr",
    [
        "2ye-mar",
        "2ys",
        "2qe",
        "2qs-feb",
        "2bqs",
        "2sms",
        "1sme",
        "2bms",
        "2cbme",
        "2me",
    ],
)
def test_to_offset_lowercase_frequency_raises(freq_depr):
    # [FUNCTION DOC]
    # Parametrized over lowercase aliases of year, quarter, business quarter, semi-month, business month and custom business month frequencies; each raises a value error with the invalid frequency message.
    msg = f"Invalid frequency: {freq_depr}"

    with pytest.raises(ValueError, match=msg):
        to_offset(freq_depr)


@pytest.mark.parametrize("freq_depr", ["2MIN", "2Us", "2NS"])
def test_to_offset_uppercase_frequency_deprecated(freq_depr):
    # [FUNCTION DOC]
    # GH#54939. Parametrized over uppercase minute, microsecond and nanosecond aliases (with a count). Conversion emits the future-removal warning saying the string (without the count) is deprecated and suggesting the lowercase form.
    # GH#54939
    depr_msg = (
        f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version, please use '{freq_depr.lower()[1:]}' instead."
    )

    with tm.assert_produces_warning(Pandas4Warning, match=depr_msg):
        to_offset(freq_depr)


@pytest.mark.parametrize(
    "freq_depr,expected",
    [
        ("2w", offsets.Week(2, weekday=6)),
        ("2b", offsets.BusinessDay(2)),
        ("2d", offsets.Day(2)),
    ],
)
def test_to_offset_lowercase_frequency_deprecated(freq_depr, expected):
    # [FUNCTION DOC]
    # GH#54939, GH#58998. Parametrized over lowercase week, business day, and day aliases with expected offsets. Conversion emits the future-removal warning that the unit is deprecated, and returns the expected offset.
    # GH#54939, GH#58998
    msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a future version."

    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        result = to_offset(freq_depr)
    assert result == expected


@pytest.mark.parametrize("freq", ["2H", "2BH", "2S"])
def test_to_offset_uppercase_frequency_raises(freq):
    # [FUNCTION DOC]
    # Parametrized over removed uppercase aliases for hour, business hour, second; raises a value error with the invalid frequency message.
    msg = f"Invalid frequency: {freq}"

    with pytest.raises(ValueError, match=msg):
        to_offset(freq)
