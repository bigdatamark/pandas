# [MODULE DOC]
# General tests for the Period scalar: rejected frequencies, construction from strings, integers, ordinals, dates and keyword fields,
# frequency inference and multiples, deprecated and lowercase frequency aliases, methods (pickling, hashing, conversion to timestamp, text rendering),
# properties (leap year, start and end time, calendar fields for each frequency), sorting, immutability and edge ordinals.
# Defines a module-level message for the business-day deprecation warning; the test classes are TestPeriodDisallowedFreqs, TestPeriodConstruction,
# TestPeriodMethods, TestPeriodProperties and TestPeriodComparisons.

from datetime import (
    date,
    datetime,
    timedelta,
)
import re

import numpy as np
import pytest

from pandas._libs.tslibs import (
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import PY314
from pandas.errors import (
    OutOfBoundsDatetime,
    Pandas4Warning,
)

from pandas import (
    NaT,
    Period,
    Timedelta,
    Timestamp,
    offsets,
)
import pandas._testing as tm

bday_msg = "Period with BDay freq is deprecated"


class TestPeriodDisallowedFreqs:
    # [CLASS DOC]
    # Groups tests that the Period constructor rejects offsets that are not valid period frequencies.
    @pytest.mark.parametrize(
        "freq, freq_msg",
        [
            (offsets.BYearBegin(), "BYearBegin"),
            (offsets.YearBegin(2), "YearBegin"),
            (offsets.QuarterBegin(startingMonth=12), "QuarterBegin"),
            (offsets.BusinessMonthEnd(2), "BusinessMonthEnd"),
        ],
    )
    def test_offsets_not_supported(self, freq, freq_msg):
        # [FUNCTION DOC]
        # GH#55785. Parametrized over begin-anchored and business offsets with the text that names them. Constructing a period with such an offset as frequency
        # must raise a value error saying the offset is not supported as a period frequency.
        # GH#55785
        msg = re.escape(f"{freq} is not supported as period frequency")
        with pytest.raises(ValueError, match=msg):
            Period(year=2014, freq=freq)

    def test_custom_business_day_freq_raises(self):
        # [FUNCTION DOC]
        # GH#52534. The custom business day frequency, given as a string or as an offset object, must raise a value error saying it is not supported as a period frequency.
        # GH#52534
        msg = "C is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq="C")
        msg = f"{offsets.CustomBusinessDay().base} is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq=offsets.CustomBusinessDay())

    def test_invalid_frequency_error_message(self):
        # [FUNCTION DOC]
        # A week-of-month frequency string must raise a value error saying it is not supported as a period frequency.
        msg = "WOM-1MON is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="WOM-1MON")

    def test_invalid_frequency_period_error_message(self):
        # [FUNCTION DOC]
        # The month-end alias is not a period frequency; constructing with it must raise a value error saying the frequency is invalid.
        msg = "Invalid frequency: ME"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="ME")


class TestPeriodConstruction:
    # [CLASS DOC]
    # Groups tests of the Period constructor with its different argument combinations.
    def test_from_td64nat_raises(self):
        # [FUNCTION DOC]
        # GH#44507. A missing timedelta numpy value, with or without a frequency, is not an acceptable value and must raise a value error listing the accepted kinds.
        # GH#44507
        td = NaT.to_numpy("m8[ns]")

        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(td)

        with pytest.raises(ValueError, match=msg):
            Period(td, freq="D")

    def test_construction(self):
        # [FUNCTION DOC]
        # Broad construction checks: equivalence of string spellings, default annual frequency, the now constructor with string and offset frequencies,
        # deprecated uppercase minute and lowercase day aliases emitting a deprecation warning, keyword-field construction,
        # sub-second string inference (millisecond, microsecond) and explicit frequencies, an ordinal without frequency raising a value error, and an unknown frequency letter raising a value error.
        i1 = Period("1/1/2005", freq="M")
        i2 = Period("Jan 2005")

        assert i1 == i2

        # GH#54105 - Period can be confusingly instantiated with lowercase freq
        # TODO: raise in the future an error when passing lowercase freq
        i1 = Period("2005", freq="Y")
        i2 = Period("2005")

        assert i1 == i2

        i4 = Period("2005", freq="M")
        assert i1 != i4

        i1 = Period.now(freq="Q")
        i2 = Period(datetime.now(), freq="Q")

        assert i1 == i2

        # Pass in freq as a keyword argument sometimes as a test for
        # https://github.com/pandas-dev/pandas/issues/53369
        i1 = Period.now(freq="D")
        i2 = Period(datetime.now(), freq="D")
        i3 = Period.now(offsets.Day())

        assert i1 == i2
        assert i1 == i3

        i1 = Period("1982", freq="min")
        msg = "'MIN' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            i2 = Period("1982", freq="MIN")
        assert i1 == i2

        i1 = Period(year=2005, month=3, day=1, freq="D")
        i2 = Period("3/1/2005", freq="D")
        assert i1 == i2

        msg = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            i3 = Period(year=2005, month=3, day=1, freq="d")
        assert i1 == i3

        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

        msg = "Must supply freq for ordinal value"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=200701)

        msg = "Invalid frequency: X"
        with pytest.raises(ValueError, match=msg):
            Period("2007-1-1", freq="X")

    def test_tuple_freq_disallowed(self):
        # [FUNCTION DOC]
        # GH#34703. A frequency given as a tuple (name and multiplier) must raise a type error telling the caller to pass a string instead, for both keyword and positional frequency.
        # GH#34703 tuple freq disallowed
        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("1982", freq=("Min", 1))

        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("2006-12-31", ("w", 1))

    def test_construction_from_timestamp_nanos(self):
        # [FUNCTION DOC]
        # GH#46811. A nanosecond-frequency period built from a timestamp or from a numpy datetime value must keep the nanoseconds, so converting back to a timestamp round-trips exactly.
        # GH#46811 don't drop nanos from Timestamp
        ts = Timestamp("2022-04-20 09:23:24.123456789")
        per = Period(ts, freq="ns")

        # should losslessly round-trip, not lose the 789
        rt = per.to_timestamp()
        assert rt == ts

        # same thing but from a datetime64 object
        dt64 = ts.asm8
        per2 = Period(dt64, freq="ns")
        rt2 = per2.to_timestamp()
        assert rt2.asm8 == dt64

    @pytest.mark.parametrize("freq", ["ms", "us", "ns"])
    def test_construction_from_min_timestamp(self, freq):
        # [FUNCTION DOC]
        # GH-63278. Parametrized over millisecond, microsecond and nanosecond units. A period built from the minimum timestamp value interpreted in that unit has year, month, day, hour, minute
        # and second equal to those of the timestamp. (Converting back to a timestamp is deliberately not tested because it would be out of bounds.)
        # GH-63278
        ts = Timestamp(Timestamp.min.value, unit=freq)
        per = Period(ts, freq=freq)

        # pandas.errors.OutOfBoundsDatetime: Out of bounds nanosecond timestamp:
        # -290308-12-21 19:59:05
        # rt = per.to_timestamp()

        assert per.year == ts.year
        assert per.month == ts.month
        assert per.day == ts.day
        assert per.hour == ts.hour
        assert per.minute == ts.minute
        assert per.second == ts.second

    def test_construction_bday(self):
        # [FUNCTION DOC]
        # Business-day construction rolls weekend dates forward to Monday: Saturday, Sunday and Monday strings all equal the daily period converted to business frequency, the lowercase alias works
        # (with a deprecation warning, extra warnings allowed), and keyword construction on a Saturday equals the Monday period.
        # Biz day construction, roll forward if non-weekday
        with tm.assert_produces_warning(
            Pandas4Warning, match="'b' is deprecated", raise_on_extra_warnings=False
        ):
            i1 = Period("3/10/12", freq="B")
            i2 = Period("3/10/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/11/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/12/12", freq="D")
            assert i1 == i2.asfreq("B")

            i3 = Period("3/10/12", freq="b")
            assert i1 == i3

            i1 = Period(year=2012, month=3, day=10, freq="B")
            i2 = Period("3/12/12", freq="B")
            assert i1 == i2

    def test_construction_quarter(self):
        # [FUNCTION DOC]
        # Quarterly construction from year and quarter, from date strings, and from quarter-string spellings (year-first, quarter-first, two-digit years, lowercase) all agree; a four-digit year string is parsed to the right year.
        i1 = Period(year=2005, quarter=1, freq="Q")
        i2 = Period("1/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period(year=2005, quarter=3, freq="Q")
        i2 = Period("9/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period("2005Q1")
        i2 = Period(year=2005, quarter=1, freq="Q")
        i3 = Period("2005q1")
        assert i1 == i2
        assert i1 == i3

        i1 = Period("05Q1")
        assert i1 == i2
        lower = Period("05q1")
        assert i1 == lower

        i1 = Period("1Q2005")
        assert i1 == i2
        lower = Period("1q2005")
        assert i1 == lower

        i1 = Period("1Q05")
        assert i1 == i2
        lower = Period("1q05")
        assert i1 == lower

        i1 = Period("4Q1984")
        assert i1.year == 1984
        lower = Period("4q1984")
        assert i1 == lower

    def test_construction_month(self):
        # [FUNCTION DOC]
        # Monthly construction from compact strings, integers, ordinals (whose year is computed from the ordinal), dates, datetimes, numpy datetimes and datetime strings with trailing zeros, all agree.
        expected = Period("2007-01", freq="M")
        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period(200701, freq="M")
        assert i1 == expected

        i1 = Period(ordinal=200701, freq="M")
        assert i1.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period("2007-01-01 00:00:00", freq="M")
        i5 = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

    def test_period_constructor_offsets(self):
        # [FUNCTION DOC]
        # Offset objects as the frequency are equivalent to the matching frequency strings for year, month, quarter, day (with multiples) and business-day (with a deprecation warning) periods, for string, integer, ordinal and keyword construction.
        # KNOWN ISSUE: the later part of this test (date, datetime, numpy and sub-second checks) duplicates checks in the monthly and general construction tests.
        assert Period("1/1/2005", freq=offsets.MonthEnd()) == Period(
            "1/1/2005", freq="M"
        )
        assert Period("2005", freq=offsets.YearEnd()) == Period("2005", freq="Y")
        assert Period("2005", freq=offsets.MonthEnd()) == Period("2005", freq="M")
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period("3/10/12", freq=offsets.BusinessDay()) == Period(
                "3/10/12", freq="B"
            )
        assert Period("3/10/12", freq=offsets.Day()) == Period("3/10/12", freq="D")

        assert Period(
            year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=1, freq="Q")
        assert Period(
            year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=2, freq="Q")

        assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(
            year=2005, month=3, day=1, freq="D"
        )
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(
                year=2012, month=3, day=10, freq="B"
            )

        expected = Period("2005-03-01", freq="3D")
        assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
        assert Period(year=2005, month=3, day=1, freq="3D") == expected

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(
                year=2012, month=3, day=10, freq="3B"
            )

        assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq="M")

        i1 = Period(ordinal=200701, freq=offsets.MonthEnd())
        i2 = Period(ordinal=200701, freq="M")
        assert i1 == i2
        assert i1.year == 18695
        assert i2.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3 = Period(np.datetime64("2007-01-01"), freq="M")
        i4 = Period("2007-01-01 00:00:00", freq="M")
        i5 = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

        i1 = Period("2007-01-01 09:00:00.001")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

    def test_invalid_arguments(self):
        # [FUNCTION DOC]
        # Invalid argument combinations raise value errors: missing frequency for a datetime or date, a float value or float ordinal, supplying both value and ordinal, a missing value and a missing frequency,
        # a negative year string, a year zero string (message depends on the Python version) and an unparsable date string (raised as a date parse error).
        msg = "Must supply freq for datetime value"
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now())
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now().date())

        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(1.6, freq="D")
        msg = "Ordinal must be an integer"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1.6, freq="D")
        msg = "Only value or ordinal but not both should be given but not both"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=2, value=1, freq="D")

        msg = "If value is None, freq cannot be None"
        with pytest.raises(ValueError, match=msg):
            Period(month=1)

        msg = '^Given date string "-2000" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            Period("-2000", "Y")
        if PY314:
            msg = "day 0 must be in range 1..31 for month 1 in year 1: 0"
        else:
            msg = "day is out of range for month"
        with pytest.raises(DateParseError, match=msg):
            Period("0", "Y")
        msg = "Unknown datetime string format, unable to parse"
        with pytest.raises(DateParseError, match=msg):
            Period("1/1/-2000", "Y")

    def test_constructor_corner(self):
        # [FUNCTION DOC]
        # Corner cases: year and month with a multiple frequency equal the string form; a missing value yields not-a-time; a period converted by passing a different frequency to the constructor equals the annual period.
        expected = Period("2007-01", freq="2M")
        assert Period(year=2007, month=1, freq="2M") == expected

        assert Period(None) is NaT

        p = Period("2007-01-01", freq="D")

        result = Period(p, freq="Y")
        exp = Period("2007", freq="Y")
        assert result == exp

    def test_constructor_infer_freq(self):
        # [FUNCTION DOC]
        # The frequency is inferred from the precision of the string: day, hour, minute, second, millisecond (three fractional digits) and microsecond (six or more fractional digits, even if zeros).
        p = Period("2007-01-01")
        assert p.freq == "D"

        p = Period("2007-01-01 07")
        assert p.freq == "h"

        p = Period("2007-01-01 07:10")
        assert p.freq == "min"

        p = Period("2007-01-01 07:10:15")
        assert p.freq == "s"

        p = Period("2007-01-01 07:10:15.123")
        assert p.freq == "ms"

        # We see that there are 6 digits after the decimal, so get microsecond
        #  even though they are all zeros.
        p = Period("2007-01-01 07:10:15.123000")
        assert p.freq == "us"

        p = Period("2007-01-01 07:10:15.123400")
        assert p.freq == "us"

    def test_multiples(self):
        # [FUNCTION DOC]
        # A two-year frequency shares its ordinal with the one-year period, has frequency strings that include the December anchor, equals the matching year-end offset with a multiple, and addition or subtraction of an integer steps by that multiple.
        result1 = Period("1989", freq="2Y")
        result2 = Period("1989", freq="Y")
        assert result1.ordinal == result2.ordinal
        assert result1.freqstr == "2Y-DEC"
        assert result2.freqstr == "Y-DEC"
        assert result1.freq == offsets.YearEnd(2)
        assert result2.freq == offsets.YearEnd()

        assert (result1 + 1).ordinal == result1.ordinal + 2
        assert (1 + result1).ordinal == result1.ordinal + 2
        assert (result1 - 1).ordinal == result2.ordinal - 2
        assert (-1 + result1).ordinal == result2.ordinal - 2

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_quarterly(self, month):
        # [FUNCTION DOC]
        # Parametrized over every month. For a quarterly frequency anchored at that month, converting a period to the end timestamp (with plain or multiple day frequency) and constructing a period back gives the original period; the string form contains the quarter text.
        # bugs in scikits.timeseries
        freq = f"Q-{month}"
        exp = Period("1989Q3", freq=freq)
        assert "1989Q3" in str(exp)
        stamp = exp.to_timestamp("D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

        stamp = exp.to_timestamp("3D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_annual(self, month):
        # [FUNCTION DOC]
        # Parametrized over every month. An annual period with that anchor, converted to its end timestamp plus thirty days and reconstructed, equals the next period and is a Period.
        # bugs in scikits.timeseries
        freq = f"Y-{month}"
        exp = Period("1989", freq=freq)
        stamp = exp.to_timestamp("D", how="end") + timedelta(days=30)
        p = Period(stamp, freq=freq)

        assert p == exp + 1
        assert isinstance(p, Period)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("num", range(10, 17))
    def test_period_cons_weekly(self, num, day):
        # [FUNCTION DOC]
        # Parametrized over days of the week and day numbers ten to sixteen. A weekly period built directly from a date string equals the daily period converted to that weekly frequency.
        daystr = f"2011-02-{num}"
        freq = f"W-{day}"

        result = Period(daystr, freq=freq)
        expected = Period(daystr, freq="D").asfreq(freq)
        assert result == expected
        assert isinstance(result, Period)

    def test_parse_week_str_roundstrip(self):
        # [FUNCTION DOC]
        # GH#50803. A week range string yields a weekly frequency anchored on the weekday of the end date (Sunday or Monday); a range that is not six days apart raises a value error saying it could not be parsed as weekly.
        # GH#50803
        per = Period("2017-01-23/2017-01-29")
        assert per.freq.freqstr == "W-SUN"

        per = Period("2017-01-24/2017-01-30")
        assert per.freq.freqstr == "W-MON"

        msg = "Could not parse as weekly-freq Period"
        with pytest.raises(ValueError, match=msg):
            # not 6 days apart
            Period("2016-01-23/2017-01-29")

    def test_period_from_ordinal(self):
        # [FUNCTION DOC]
        # The private ordinal constructor given an ordinal and frequency from an existing period returns an equal Period.
        p = Period("2011-01", freq="M")
        res = Period._from_ordinal(p.ordinal, freq=p.freq)
        assert p == res
        assert isinstance(res, Period)

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    def test_construct_from_nat_string_and_freq(self, freq):
        # [FUNCTION DOC]
        # Parametrized over year, month, day and hour frequencies. A not-a-time string with the frequency, or with a multiple of it, returns the not-a-time singleton.
        per = Period("NaT", freq=freq)
        assert per is NaT

        per = Period("NaT", freq="2" + freq)
        assert per is NaT

        per = Period("NaT", freq="3" + freq)
        assert per is NaT

    def test_period_cons_nat(self):
        # [FUNCTION DOC]
        # Not-a-time strings and the missing-integer sentinel return the not-a-time singleton for various frequencies (weekly, multiple day, combined day and hour) and for no frequency.
        p = Period("nat", freq="W-SUN")
        assert p is NaT

        p = Period(iNaT, freq="D")
        assert p is NaT

        p = Period(iNaT, freq="3D")
        assert p is NaT

        p = Period(iNaT, freq="1D1h")
        assert p is NaT

        p = Period("NaT")
        assert p is NaT

        p = Period(iNaT)
        assert p is NaT

    def test_period_cons_mult(self):
        # [FUNCTION DOC]
        # A three-month period shares the ordinal of the one-month period, has the multiple month-end offset and frequency string, steps by three months with integer addition and subtraction keeping its frequency,
        # and a negative or zero multiple raises a value error saying the frequency must be positive.
        p1 = Period("2011-01", freq="3M")
        p2 = Period("2011-01", freq="M")
        assert p1.ordinal == p2.ordinal

        assert p1.freq == offsets.MonthEnd(3)
        assert p1.freqstr == "3M"

        assert p2.freq == offsets.MonthEnd()
        assert p2.freqstr == "M"

        result = p1 + 1
        assert result.ordinal == (p2 + 3).ordinal

        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        result = p1 - 1
        assert result.ordinal == (p2 - 3).ordinal
        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        msg = "Frequency must be positive, because it represents span: -3M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-3M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0M")

    def test_period_cons_combined(self):
        # [FUNCTION DOC]
        # Combined day-plus-hour frequencies (either order), built from a string or from an ordinal, equal a twenty-five-hour frequency with the same ordinal as plain hourly; integer arithmetic steps by twenty-five hours.
        # Negative or zero combined frequencies raise a value error saying the frequency must be positive; combining a week with a day raises a value error saying the frequency is invalid.
        p = [
            (
                Period("2011-01", freq="1D1h"),
                Period("2011-01", freq="1h1D"),
                Period("2011-01", freq="h"),
            ),
            (
                Period(ordinal=1, freq="1D1h"),
                Period(ordinal=1, freq="1h1D"),
                Period(ordinal=1, freq="h"),
            ),
        ]

        for p1, p2, p3 in p:
            assert p1.ordinal == p3.ordinal
            assert p2.ordinal == p3.ordinal

            assert p1.freq == offsets.Hour(25)
            assert p1.freqstr == "25h"

            assert p2.freq == offsets.Hour(25)
            assert p2.freqstr == "25h"

            assert p3.freq == offsets.Hour()
            assert p3.freqstr == "h"

            result = p1 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

            result = p1 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

        msg = "Frequency must be positive, because it represents span: -25h"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1h1D")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1h1D")

        msg = "Frequency must be positive, because it represents span: 0D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0D0h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="0D0h")

        # You can only combine together day and intraday offsets
        msg = "Invalid frequency: 1W1D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1W1D")
        msg = "Invalid frequency: 1D1W"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1D1W")

    @pytest.mark.parametrize("day", ["1970/01/01 ", "2020-12-31 ", "1981/09/13 "])
    @pytest.mark.parametrize("hour", ["00:00:00", "00:00:01", "23:59:59", "12:00:59"])
    @pytest.mark.parametrize(
        "sec_float, expected",
        [
            (".000000001", 1),
            (".000000999", 999),
            (".123456789", 789),
            (".999999999", 999),
            (".999999000", 0),
            # Test femtoseconds, attoseconds, picoseconds are dropped like Timestamp
            (".999999001123", 1),
            (".999999001123456", 1),
            (".999999001123456789", 1),
        ],
    )
    def test_period_constructor_nanosecond(self, day, hour, sec_float, expected):
        # [FUNCTION DOC]
        # GH 34621. Parametrized over dates, times and fractional second strings with expected nanosecond component. The start time of the period built from the combined string has the expected nanosecond value;
        # digits beyond nanosecond precision are dropped like a timestamp does.
        # GH 34621

        assert Period(day + hour + sec_float).start_time.nanosecond == expected

    @pytest.mark.parametrize("hour", range(24))
    def test_period_large_ordinal(self, hour):
        # [FUNCTION DOC]
        # Issue #36430. Parametrized over hour 0 to 23. An hourly period with an ordinal beyond the maximum timestamp's must not overflow and has the expected hour.
        # Issue #36430
        # Integer overflow for Period over the maximum timestamp
        p = Period(ordinal=2562048 + hour, freq="1h")
        assert p.hour == hour

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize(
        "freq,freq_depr",
        [("2W", "2w"), ("2W-FRI", "2w-fri"), ("2D", "2d"), ("2B", "2b")],
    )
    def test_period_deprecated_lowercase_freq(self, freq, freq_depr):
        # [FUNCTION DOC]
        # GH#58998. Parametrized over a frequency and its lowercase form. Constructing with the lowercase form emits the deprecation warning (message uses the form without its first character; extra warnings allowed) and equals the period with the canonical frequency. Business-day warnings are filtered.
        # GH#58998
        msg = (
            f"'{freq_depr[1:]}' is deprecated and will be removed in a future version."
        )

        with tm.assert_produces_warning(
            Pandas4Warning, match=msg, raise_on_extra_warnings=False
        ):
            result = Period("2016-03-01 09:00", freq=freq_depr)

        expected = Period("2016-03-01 09:00", freq=freq)
        assert result == expected


class TestPeriodMethods:
    # [CLASS DOC]
    # Groups tests of Period methods: pickling, hashing, conversion to timestamp and text rendering.
    def test_round_trip(self, temp_file):
        # [FUNCTION DOC]
        # Pickle round trip: a quarterly period written to the temporary file fixture and read back equals the original.
        p = Period("2000Q1")
        new_p = tm.round_trip_pickle(p, temp_file)
        assert new_p == p

    def test_hash(self):
        # [FUNCTION DOC]
        # Equal periods hash equal; different frequency, different multiple or different position give different hashes.
        assert hash(Period("2011-01", freq="M")) == hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01-01", freq="D")) != hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01", freq="3M")) != hash(Period("2011-01", freq="2M"))

        assert hash(Period("2011-01", freq="M")) != hash(Period("2011-02", freq="M"))

    # --------------------------------------------------------------
    # to_timestamp

    def test_to_timestamp_mult(self):
        # [FUNCTION DOC]
        # Conversion to a timestamp with the start convention gives the first instant of the period and with the end convention the last microsecond before the next period, also for a multiple-month frequency.
        p = Period("2011-01", freq="M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-02-01") - Timedelta(1, unit="us")
        assert p.to_timestamp(how="E") == expected

        p = Period("2011-01", freq="3M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-04-01") - Timedelta(1, unit="us")
        assert p.to_timestamp(how="E") == expected

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_to_timestamp(self):
        # [FUNCTION DOC]
        # Conversion to timestamp: accepts case-insensitive aliases for start and end conventions, a multiple-day frequency argument does not change the result, and for every base frequency a round trip through timestamp and back gives the original period;
        # start and end time properties match the start convention and the expected last microsecond. Also checks converting an annual period with hourly, minutely, secondly and multiple frequencies for both conventions.
        p = Period("1982", freq="Y")
        start_ts = p.to_timestamp(how="S")
        aliases = ["s", "StarT", "BEGIn"]
        for a in aliases:
            assert start_ts == p.to_timestamp("D", how=a)
            # freq with mult should not affect to the result
            assert start_ts == p.to_timestamp("3D", how=a)

        end_ts = p.to_timestamp(how="E")
        aliases = ["e", "end", "FINIsH"]
        for a in aliases:
            assert end_ts == p.to_timestamp("D", how=a)
            assert end_ts == p.to_timestamp("3D", how=a)

        from_lst = ["Y", "Q", "M", "W", "B", "D", "h", "Min", "s"]

        def _ex(p):
            # [FUNCTION DOC]
            # HELPER: computes the expected end time of a period for the timestamp test above. For business-day frequency it is the start time plus one day minus one microsecond; otherwise it is the microsecond before the start of the next period.
            if p.freq == "B":
                return p.start_time + Timedelta(days=1) - Timedelta(microseconds=1)
            return Timestamp((p + p.freq).start_time._value - 1, unit="us")

        for fcode in from_lst:
            p = Period("1982", freq=fcode)
            result = p.to_timestamp().to_period(fcode)
            assert result == p

            assert p.start_time == p.to_timestamp(how="S")

            assert p.end_time == _ex(p)

        # Frequency other than daily

        p = Period("1985", freq="Y")

        result = p.to_timestamp("h", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, unit="us")
        assert result == expected
        result = p.to_timestamp("3h", how="end")
        assert result == expected

        result = p.to_timestamp("min", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, unit="us")
        assert result == expected
        result = p.to_timestamp("2min", how="end")
        assert result == expected

        result = p.to_timestamp(how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, unit="us")
        assert result == expected

        expected = datetime(1985, 1, 1)
        result = p.to_timestamp("h", how="start")
        assert result == expected
        result = p.to_timestamp("min", how="start")
        assert result == expected
        result = p.to_timestamp("s", how="start")
        assert result == expected
        result = p.to_timestamp("3h", how="start")
        assert result == expected
        result = p.to_timestamp("5s", how="start")
        assert result == expected

    def test_to_timestamp_business_end(self):
        # [FUNCTION DOC]
        # A Friday business-day period converted to timestamp with business frequency and the end convention gives the last microsecond of Saturday; construction emits the business-day deprecation warning.
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period("1990-01-05", "B")  # Friday
            result = per.to_timestamp("B", how="E")

        expected = Timestamp("1990-01-06") - Timedelta(1, unit="us")
        assert result == expected

    @pytest.mark.parametrize(
        "ts, expected",
        [
            ("1970-01-01 00:00:00", 0),
            ("1970-01-01 00:00:00.000001", 1),
            ("1970-01-01 00:00:00.00001", 10),
            ("1970-01-01 00:00:00.499", 499000),
            ("1999-12-31 23:59:59.999", 999000),
            ("1999-12-31 23:59:59.999999", 999999),
            ("2050-12-31 23:59:59.5", 500000),
            ("2050-12-31 23:59:59.500001", 500001),
            ("2050-12-31 23:59:59.123456", 123456),
        ],
    )
    @pytest.mark.parametrize("freq", [None, "us", "ns"])
    def test_to_timestamp_microsecond(self, ts, expected, freq):
        # [FUNCTION DOC]
        # GH 24444. Parametrized over timestamp strings with expected microsecond component and over no frequency, microsecond and nanosecond frequency arguments. The microsecond field of the resulting timestamp equals the expected value.
        # GH 24444
        result = Period(ts).to_timestamp(freq=freq).microsecond
        assert result == expected

    # --------------------------------------------------------------
    # Rendering: __repr__, strftime, etc

    @pytest.mark.parametrize(
        "str_ts,freq,str_res,str_freq",
        (
            ("Jan-2000", None, "2000-01", "M"),
            ("2000-12-15", None, "2000-12-15", "D"),
            (
                "2000-12-15 13:45:26.123456789",
                "ns",
                "2000-12-15 13:45:26.123456789",
                "ns",
            ),
            ("2000-12-15 13:45:26.123456789", "us", "2000-12-15 13:45:26.123456", "us"),
            ("2000-12-15 13:45:26.123456", None, "2000-12-15 13:45:26.123456", "us"),
            ("2000-12-15 13:45:26.123456789", "ms", "2000-12-15 13:45:26.123", "ms"),
            ("2000-12-15 13:45:26.123", None, "2000-12-15 13:45:26.123", "ms"),
            ("2000-12-15 13:45:26", "s", "2000-12-15 13:45:26", "s"),
            ("2000-12-15 13:45:26", "min", "2000-12-15 13:45", "min"),
            ("2000-12-15 13:45:26", "h", "2000-12-15 13:00", "h"),
            ("2000-12-15", "Y", "2000", "Y-DEC"),
            ("2000-12-15", "Q", "2000Q4", "Q-DEC"),
            ("2000-12-15", "M", "2000-12", "M"),
            ("2000-12-15", "W", "2000-12-11/2000-12-17", "W-SUN"),
            ("2000-12-15", "D", "2000-12-15", "D"),
            ("2000-12-15", "B", "2000-12-15", "B"),
        ),
    )
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_repr(self, str_ts, freq, str_res, str_freq):
        # [FUNCTION DOC]
        # Parametrized over a timestamp string, a frequency (or none), the expected string form and the expected frequency text. The string of the period equals the expected form and the detailed representation names the class, string form and frequency text.
        # Business-day deprecation warnings are filtered.
        p = Period(str_ts, freq=freq)
        assert str(p) == str_res
        assert repr(p) == f"Period('{str_res}', '{str_freq}')"

    def test_repr_nat(self):
        # [FUNCTION DOC]
        # The representation of a not-a-time period (monthly) contains the representation of the not-a-time scalar.
        p = Period("nat", freq="M")
        assert repr(NaT) in repr(p)

    def test_strftime(self):
        # [FUNCTION DOC]
        # GH#3363. Formatting a seconds-frequency period with a format string returns a plain string with the formatted date and time.
        # GH#3363
        p = Period("2000-1-1 12:34:12", freq="s")
        res = p.strftime("%Y-%m-%d %H:%M:%S")
        assert res == "2000-01-01 12:34:12"
        assert isinstance(res, str)


class TestPeriodProperties:
    # [CLASS DOC]
    # Groups tests of Period properties such as year, month, weekday, start and end time.
    """Test properties such as year, month, weekday, etc...."""

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    def test_is_leap_year(self, freq):
        # [FUNCTION DOC]
        # GH 13727. Parametrized over year, month, day and hour frequencies. The leap-year property is a boolean: true for 2000 and 2004, false for 1999 and 2100.
        # GH 13727
        p = Period("2000-01-01 00:00:00", freq=freq)
        assert p.is_leap_year
        assert isinstance(p.is_leap_year, bool)

        p = Period("1999-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

        p = Period("2004-01-01 00:00:00", freq=freq)
        assert p.is_leap_year

        p = Period("2100-01-01 00:00:00", freq=freq)
        assert not p.is_leap_year

    def test_quarterly_negative_ordinals(self):
        # [FUNCTION DOC]
        # Periods with negative ordinals (before 1970) have correct year, quarter and month: ordinals negative one and two for quarterly give 1969 quarters four and three; negative two for monthly gives November 1969.
        p = Period(ordinal=-1, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 4
        assert isinstance(p, Period)

        p = Period(ordinal=-2, freq="Q-DEC")
        assert p.year == 1969
        assert p.quarter == 3
        assert isinstance(p, Period)

        p = Period(ordinal=-2, freq="M")
        assert p.year == 1969
        assert p.month == 11
        assert isinstance(p, Period)

    def test_freq_str(self):
        # [FUNCTION DOC]
        # A period built with the capitalized minute alias has a minute offset frequency and the canonical frequency string.
        i1 = Period("1982", freq="Min")
        assert i1.freq == offsets.Minute()
        assert i1.freqstr == "min"

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_period_deprecated_freq(self):
        # [FUNCTION DOC]
        # Long-form and mixed-case frequency aliases (monthly, business, daily, hourly, minutely, secondly, and so on) must raise a value error with the invalid-frequency message, both for string and ordinal construction; the supported short aliases still build periods.
        # KNOWN ISSUE: the name and the variable names refer to deprecated aliases, but the aliases are now rejected outright rather than deprecated.
        cases = {
            "M": ["MTH", "MONTH", "MONTHLY", "Mth", "month", "monthly"],
            "B": ["BUS", "BUSINESS", "BUSINESSLY", "WEEKDAY", "bus"],
            "D": ["DAY", "DLY", "DAILY", "Day", "Dly", "Daily"],
            "h": ["HR", "HOUR", "HRLY", "HOURLY", "hr", "Hour", "HRly"],
            "min": ["minute", "MINUTE", "MINUTELY", "minutely"],
            "s": ["sec", "SEC", "SECOND", "SECONDLY", "second"],
            "ms": ["MILLISECOND", "MILLISECONDLY", "millisecond"],
            "us": ["MICROSECOND", "MICROSECONDLY", "microsecond"],
            "ns": ["NANOSECOND", "NANOSECONDLY", "nanosecond"],
        }

        msg = INVALID_FREQ_ERR_MSG
        for exp, freqs in cases.items():
            for freq in freqs:
                with pytest.raises(ValueError, match=msg):
                    Period("2016-03-01 09:00", freq=freq)
                with pytest.raises(ValueError, match=msg):
                    Period(ordinal=1, freq=freq)

            # check supported freq-aliases still works
            p1 = Period("2016-03-01 09:00", freq=exp)
            p2 = Period(ordinal=1, freq=exp)
            assert isinstance(p1, Period)
            assert isinstance(p2, Period)

    @staticmethod
    def _period_constructor(bound, offset):
        # [FUNCTION DOC]
        # HELPER: static method building a microsecond-frequency period from a timestamp bound, using its year, month, day, hour and minute and its second shifted by an offset (in seconds). Used by the two bounds tests below.
        return Period(
            year=bound.year,
            month=bound.month,
            day=bound.day,
            hour=bound.hour,
            minute=bound.minute,
            second=bound.second + offset,
            freq="us",
        )

    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    def test_outer_bounds_start_and_end_time(self, bound, offset, period_property):
        # [FUNCTION DOC]
        # GH #13346. Parametrized over the minimum and maximum timestamp (with an offset of minus or plus one second) and over the start-time and end-time properties. Reading these properties on the period shifted just outside the bound no longer raises
        # (since GH#63760). A millisecond period built from the bound's ordinal far outside the range must raise: an overflow error for the end time of the maximum case, otherwise an out-of-bounds datetime error saying the microsecond is out of bounds.
        # GH #13346
        period = TestPeriodProperties._period_constructor(bound, offset)
        # post-GH#63760 this no longer raises OutOfBoundsDatetime
        getattr(period, period_property)

        per = Period._from_ordinal(bound.value, freq=to_offset("ms"))

        if period_property == "end_time" and offset == 1:
            err = OverflowError
            msg = "value too large"
        else:
            err = OutOfBoundsDatetime
            msg = "Out of bounds microsecond"
        # But if we do the same thing with a much-more-remote date, we still
        #  raise OutOfBoundsDatetime (as opposed to silently giving a wrong result)
        with pytest.raises(err, match=msg):
            getattr(per, period_property)

    @pytest.mark.parametrize("bound, offset", [(Timestamp.min, -1), (Timestamp.max, 1)])
    @pytest.mark.parametrize("period_property", ["start_time", "end_time"])
    def test_inner_bounds_start_and_end_time(self, bound, offset, period_property):
        # [FUNCTION DOC]
        # GH #13346. Parametrized over the minimum and maximum bounds and the start/end properties. A period just inside the bound has start and end times that agree with the period's timestamp after rounding to seconds, and equal to the bound shifted by one second after flooring to seconds.
        # GH #13346
        period = TestPeriodProperties._period_constructor(bound, -offset)
        expected = period.to_timestamp().round(freq="s")
        assert getattr(period, period_property).round(freq="s") == expected
        expected = (bound - offset * Timedelta(1, unit="s")).floor("s")
        assert getattr(period, period_property).floor("s") == expected

    def test_start_time(self):
        # [FUNCTION DOC]
        # The start time of 2012 periods in year, quarter, month, day, hour, minute and second frequency is the first of January; the business-day period starts on the second (a Monday, with deprecation warning), and the weekly period starts on the Monday before.
        freq_lst = ["Y", "Q", "M", "D", "h", "min", "s"]
        xp = datetime(2012, 1, 1)
        for f in freq_lst:
            p = Period("2012", freq=f)
            assert p.start_time == xp
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period("2012", freq="B").start_time == datetime(2012, 1, 2)
        assert Period("2012", freq="W").start_time == datetime(2011, 12, 26)

    def test_end_time(self):
        # [FUNCTION DOC]
        # The end time is the last microsecond of the period for year, quarter, month, day, hour, business day, week, multiple-day (GH 11738) and combined day-and-hour frequencies.
        p = Period("2012", freq="Y")

        exp = Timestamp(2012, 12, 31, 23, 59, 59, 999999)
        assert p.end_time == exp

        p = Period("2012", freq="Q")
        exp = Timestamp(2012, 3, 31, 23, 59, 59, 999999)
        assert p.end_time == exp

        p = Period("2012", freq="M")
        exp = Timestamp(2012, 1, 31, 23, 59, 59, 999999)
        assert p.end_time == exp

        p = Period("2012", freq="D")
        exp = Timestamp(2012, 1, 1, 23, 59, 59, 999999)
        assert p.end_time == exp

        p = Period("2012", freq="h")
        exp = Timestamp(2012, 1, 1, 0, 59, 59, 999999)
        assert p.end_time == exp

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            p = Period("2012", freq="B")
            exp = Timestamp(2012, 1, 2, 23, 59, 59, 999999)
            assert p.end_time == exp

        p = Period("2012", freq="W")
        exp = Timestamp(2012, 1, 1, 23, 59, 59, 999999)
        assert p.end_time == exp

        # Test for GH 11738
        p = Period("2012", freq="15D")
        exp = Timestamp(2012, 1, 15, 23, 59, 59, 999999)
        assert p.end_time == exp

        p = Period("2012", freq="1D1h")
        exp = Timestamp(2012, 1, 2, 0, 59, 59, 999999)
        assert p.end_time == exp

        p = Period("2012", freq="1h1D")
        exp = Timestamp(2012, 1, 2, 0, 59, 59, 999999)
        assert p.end_time == exp

    def test_end_time_business_friday(self):
        # [FUNCTION DOC]
        # GH#34449. A Friday business-day period's end time is the last microsecond of Saturday; construction emits the business-day warning.
        # GH#34449
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period("1990-01-05", "B")
            result = per.end_time

        expected = Timestamp("1990-01-06") - Timedelta(1, unit="us")
        assert result == expected

    def test_anchor_week_end_time(self):
        # [FUNCTION DOC]
        # A weekly period anchored on Saturday has an end time at the last microsecond of that Saturday.
        p = Period("2013-1-1", "W-SAT")
        exp = Timestamp(2013, 1, 5, 23, 59, 59, 999999)
        assert p.end_time == exp

    def test_properties_annually(self):
        # [FUNCTION DOC]
        # An annual period reports its year.
        # Test properties on Periods with annually frequency.
        a_date = Period(freq="Y", year=2007)
        assert a_date.year == 2007

    def test_properties_quarterly(self):
        # [FUNCTION DOC]
        # Quarterly periods with December, January and June fiscal year-ends: the fiscal year is 2007 and the quarter is one plus the offset for the first three steps.
        # KNOWN ISSUE: the comment says daily frequency but the test exercises quarterly periods (copy-paste).
        # Test properties on Periods with daily frequency.
        qedec_date = Period(freq="Q-DEC", year=2007, quarter=1)
        qejan_date = Period(freq="Q-JAN", year=2007, quarter=1)
        qejun_date = Period(freq="Q-JUN", year=2007, quarter=1)
        for x in range(3):
            for qd in (qedec_date, qejan_date, qejun_date):
                assert (qd + x).qyear == 2007
                assert (qd + x).quarter == x + 1

    def test_properties_monthly(self):
        # [FUNCTION DOC]
        # Monthly periods stepped over the first eleven months report year 2007, the right month and quarter.
        # KNOWN ISSUE: the comment says daily frequency but the test exercises monthly periods, and the loop stops before December.
        # Test properties on Periods with daily frequency.
        m_date = Period(freq="M", year=2007, month=1)
        for x in range(11):
            m_ival_x = m_date + x
            assert m_ival_x.year == 2007
            if 1 <= x + 1 <= 3:
                assert m_ival_x.quarter == 1
            elif 4 <= x + 1 <= 6:
                assert m_ival_x.quarter == 2
            elif 7 <= x + 1 <= 9:
                assert m_ival_x.quarter == 3
            elif 10 <= x + 1 <= 12:
                assert m_ival_x.quarter == 4
            assert m_ival_x.month == x + 1

    def test_properties_weekly(self):
        # [FUNCTION DOC]
        # A weekly period reports year, quarter, month, week number (one, and fifty-two one week earlier) and days in month (including leap February).
        # KNOWN ISSUE: the comment says daily frequency but the test exercises weekly periods.
        # Test properties on Periods with daily frequency.
        w_date = Period(freq="W", year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31
        assert Period(freq="W", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_weekly_legacy(self):
        # [FUNCTION DOC]
        # Same property checks as the weekly test, plus the legacy week alias raising a value error with the invalid-frequency message.
        # KNOWN ISSUE: the property assertions duplicate the previous test, and the comment wrongly says daily frequency.
        # Test properties on Periods with daily frequency.
        w_date = Period(freq="W", year=2007, month=1, day=7)
        assert w_date.year == 2007
        assert w_date.quarter == 1
        assert w_date.month == 1
        assert w_date.week == 1
        assert (w_date - 1).week == 52
        assert w_date.days_in_month == 31

        exp = Period(freq="W", year=2012, month=2, day=1)
        assert exp.days_in_month == 29

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=7)

    def test_properties_daily(self):
        # [FUNCTION DOC]
        # Business-day (with deprecation warning) and daily periods on the first of January 2007 report year, quarter, month, day, weekday zero, day of year one and days in month; leap February has twenty-nine days.
        # Test properties on Periods with daily frequency.
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            b_date = Period(freq="B", year=2007, month=1, day=1)
        assert b_date.year == 2007
        assert b_date.quarter == 1
        assert b_date.month == 1
        assert b_date.day == 1
        assert b_date.weekday == 0
        assert b_date.dayofyear == 1
        assert b_date.days_in_month == 31
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(freq="B", year=2012, month=2, day=1).days_in_month == 29

        d_date = Period(freq="D", year=2007, month=1, day=1)

        assert d_date.year == 2007
        assert d_date.quarter == 1
        assert d_date.month == 1
        assert d_date.day == 1
        assert d_date.weekday == 0
        assert d_date.dayofyear == 1
        assert d_date.days_in_month == 31
        assert Period(freq="D", year=2012, month=2, day=1).days_in_month == 29

    def test_properties_hourly(self):
        # [FUNCTION DOC]
        # Hourly periods, with a one-hour and a two-hour frequency, report the calendar fields and hour; leap February has twenty-nine days.
        # Test properties on Periods with hourly frequency.
        h_date1 = Period(freq="h", year=2007, month=1, day=1, hour=0)
        h_date2 = Period(freq="2h", year=2007, month=1, day=1, hour=0)

        for h_date in [h_date1, h_date2]:
            assert h_date.year == 2007
            assert h_date.quarter == 1
            assert h_date.month == 1
            assert h_date.day == 1
            assert h_date.weekday == 0
            assert h_date.dayofyear == 1
            assert h_date.hour == 0
            assert h_date.days_in_month == 31
            assert (
                Period(freq="h", year=2012, month=2, day=1, hour=0).days_in_month == 29
            )

    def test_properties_minutely(self):
        # [FUNCTION DOC]
        # A minute-frequency period reports calendar fields, hour and minute.
        # KNOWN ISSUE: the final leap-February check builds a daily period (with hour and minute fields) rather than a minutely one.
        # Test properties on Periods with minutely frequency.
        t_date = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
        assert t_date.quarter == 1
        assert t_date.month == 1
        assert t_date.day == 1
        assert t_date.weekday == 0
        assert t_date.dayofyear == 1
        assert t_date.hour == 0
        assert t_date.minute == 0
        assert t_date.days_in_month == 31
        assert (
            Period(freq="D", year=2012, month=2, day=1, hour=0, minute=0).days_in_month
            == 29
        )

    def test_properties_secondly(self):
        # [FUNCTION DOC]
        # Reports calendar fields and the second field for a period.
        # KNOWN ISSUE: despite the name, the periods are built with minute frequency rather than seconds frequency.
        # Test properties on Periods with secondly frequency.
        s_date = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        assert s_date.year == 2007
        assert s_date.quarter == 1
        assert s_date.month == 1
        assert s_date.day == 1
        assert s_date.weekday == 0
        assert s_date.dayofyear == 1
        assert s_date.hour == 0
        assert s_date.minute == 0
        assert s_date.second == 0
        assert s_date.days_in_month == 31
        assert (
            Period(
                freq="Min", year=2012, month=2, day=1, hour=0, minute=0, second=0
            ).days_in_month
            == 29
        )


class TestPeriodComparisons:
    # [CLASS DOC]
    # Groups ordering tests for Period.
    def test_sort_periods(self):
        # [FUNCTION DOC]
        # Sorting a list of monthly periods in arbitrary order yields chronological order.
        jan = Period("2000-01", "M")
        feb = Period("2000-02", "M")
        mar = Period("2000-03", "M")
        periods = [mar, jan, feb]
        correctPeriods = [jan, feb, mar]
        assert sorted(periods) == correctPeriods


def test_period_immutable():
    # [FUNCTION DOC]
    # see gh-17116. Assigning to the ordinal or the frequency of a period must raise an attribute error saying the attribute is not writable.
    # see gh-17116
    msg = "not writable"

    per = Period("2014Q1")
    with pytest.raises(AttributeError, match=msg):
        per.ordinal = 14

    freq = per.freq
    with pytest.raises(AttributeError, match=msg):
        per.freq = 2 * freq


def test_small_year_parsing():
    # [FUNCTION DOC]
    # A daily period in year one parses its year and day correctly.
    per1 = Period("0001-01-07", "D")
    assert per1.year == 1
    assert per1.day == 7


def test_negone_ordinals():
    # [FUNCTION DOC]
    # Periods with ordinal negative one for year, month, quarter, day, hour, minute, second, business day (with deprecation warning) and week frequency can be represented without errors; the daily one converted to each frequency can also be represented; non-business ones report year 1969.
    freqs = ["Y", "M", "Q", "D", "h", "min", "s"]

    period = Period(ordinal=-1, freq="D")
    for freq in freqs:
        repr(period.asfreq(freq))

    for freq in freqs:
        period = Period(ordinal=-1, freq=freq)
        repr(period)
        assert period.year == 1969

    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        period = Period(ordinal=-1, freq="B")
    repr(period)
    period = Period(ordinal=-1, freq="W")
    repr(period)
