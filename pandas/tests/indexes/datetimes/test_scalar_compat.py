"""
Tests for DatetimeIndex methods behaving like their Timestamp counterparts
"""

# [MODULE DOC]
# Tests that DatetimeIndex accessors and name methods agree with the Timestamp
# counterparts: date and time parts, calendar fields, boundary flags, day and
# month names (including locales), and an optional property-based comparison.
# Uses hypothesis for one slow test; reads the available locales at collection time.

import calendar
from datetime import (
    date,
    datetime,
    time,
)
import locale
import unicodedata

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import pytest

from pandas._libs.tslibs import timezones

from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestDatetimeIndexOps:
    # [CLASS DOC]
    # Groups tests of DatetimeIndex field and name accessors.
    def test_dti_no_millisecond_field(self):
        # [FUNCTION DOC]
        # Verifies that the class and instances of DatetimeIndex raise an attribute error
        # for a millisecond field that does not exist.
        msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex.millisecond

        msg = "'DatetimeIndex' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex([]).millisecond

    def test_dti_time(self):
        # [FUNCTION DOC]
        # Verifies the time accessor of a 12-minute range equals the list of the elements'
        # time-of-day values.
        rng = date_range("1/1/2000", freq="12min", periods=10)
        result = Index(rng).time
        expected = [t.time() for t in rng]
        assert (result == expected).all()

    def test_dti_date(self):
        # [FUNCTION DOC]
        # Verifies the date accessor of a 12-hour range equals the list of the elements'
        # dates.
        rng = date_range("1/1/2000", freq="12h", periods=10)
        result = Index(rng).date
        expected = [t.date() for t in rng]
        assert (result == expected).all()

    @pytest.mark.parametrize(
        "dtype",
        [None, "datetime64[ns, CET]", "datetime64[ns, EST]", "datetime64[ns, UTC]"],
    )
    def test_dti_date2(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype (naive, CET, EST, UTC). Regression test (GH#21230)
        # that the date accessor returns an object array of the date and the
        # missing-time marker for a missing entry.
        # Regression test for GH#21230
        expected = np.array([date(2018, 6, 4), NaT])

        index = DatetimeIndex(["2018-06-04 10:00:00", NaT], dtype=dtype)
        result = index.date

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [None, "datetime64[ns, CET]", "datetime64[ns, EST]", "datetime64[ns, UTC]"],
    )
    def test_dti_time2(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype (naive, CET, EST, UTC). Regression test (GH#21267)
        # that the time accessor returns an object array of the time-of-day and the
        # missing-time marker for a missing entry.
        # Regression test for GH#21267
        expected = np.array([time(10, 20, 30), NaT])

        index = DatetimeIndex(["2018-06-04 10:20:30", NaT], dtype=dtype)
        result = index.time

        tm.assert_numpy_array_equal(result, expected)

    def test_dti_timetz(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Parametrized over tz_naive_fixture. Verifies (GH#21358) the timetz accessor
        # returns time-of-day objects carrying the index timezone, with the missing
        # marker preserved.
        # GH#21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        expected = np.array([time(10, 20, 30, tzinfo=tz), NaT])

        index = DatetimeIndex(["2018-06-04 10:20:30", NaT], tz=tz)
        result = index.timetz

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "field",
        [
            "dayofweek",
            "day_of_week",
            "dayofyear",
            "day_of_year",
            "quarter",
            "days_in_month",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ],
    )
    def test_dti_timestamp_fields(self, field):
        # [FUNCTION DOC]
        # Parametrized over fields (weekday, ordinal-day, quarter, days in month,
        # and start/end flags). Verifies the index field at the last position equals the
        # same field of the corresponding Timestamp.
        # extra fields from DatetimeIndex like quarter and week
        idx = date_range("2020-01-01", periods=10)
        expected = getattr(idx, field)[-1]

        result = getattr(Timestamp(idx[-1]), field)
        assert result == expected

    def test_dti_nanosecond(self):
        # [FUNCTION DOC]
        # Verifies the nanosecond accessor of an index built from small integers returns
        # an index of 32-bit integers equal to those values.
        dti = DatetimeIndex(np.arange(10))
        expected = Index(np.arange(10, dtype=np.int32))

        tm.assert_index_equal(dti.nanosecond, expected)

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_hour_tzaware(self, prefix):
        # [FUNCTION DOC]
        # Parametrized over a timezone prefix (pytz-style or dateutil). Verifies hour
        # is zero for midnight values in US/Eastern, and (GH#1946) that an hourly range
        # in an unusual zone (Atikokan) yields hours zero through nine as a 32-bit
        # integer index.
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
        rng = DatetimeIndex(strdates, tz=prefix + "US/Eastern")
        assert (rng.hour == 0).all()

        # a more unusual time zone, GH#1946
        dr = date_range(
            "2011-10-02 00:00", freq="h", periods=10, tz=prefix + "America/Atikokan"
        )

        expected = Index(np.arange(10, dtype=np.int32))
        tm.assert_index_equal(dr.hour, expected)

    # GH#12806
    @pytest.mark.parametrize(
        "time_locale",
        [None, *tm.get_locales()],
    )
    def test_day_name_month_name(self, time_locale):
        # [FUNCTION DOC]
        # Parametrized over locale (none plus all available locales). Verifies
        # (GH#12805, GH#12806, GH#11128) that day and month names match the locale's
        # calendar names (English when no locale), for the index and for Timestamp
        # objects, and that missing entries give a missing float.
        # Names are normalized to a canonical Unicode form before comparing to work
        # around differing normalization schemes (GH#22342).
        # Sets the time locale temporarily; the setting is restored on exit.
        # KNOWN ISSUE: in the per-element month name loop the expected value is rebuilt from the already normalized result instead of from the expected month name, so the final assertion compares the result with itself and can never fail.
        # Test Monday -> Sunday and January -> December, in that sequence
        if time_locale is None:
            # If the time_locale is None, day-name and month_name should
            # return the english attributes
            expected_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            expected_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = calendar.day_name[:]
                expected_months = calendar.month_name[1:]

        # GH#11128
        dti = date_range(freq="D", start=datetime(1998, 1, 1), periods=365)
        english_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name = name.capitalize()
            assert dti.day_name(locale=time_locale)[day] == name
            assert dti.day_name(locale=None)[day] == eng_name
            ts = Timestamp(datetime(2016, 4, day))
            assert ts.day_name(locale=time_locale) == name
        dti = dti.append(DatetimeIndex([NaT]))
        assert np.isnan(dti.day_name(locale=time_locale)[-1])
        ts = Timestamp(NaT)
        assert np.isnan(ts.day_name(locale=time_locale))

        # GH#12805
        dti = date_range(freq="ME", start="2012", end="2013")
        result = dti.month_name(locale=time_locale)
        expected = Index([month.capitalize() for month in expected_months])

        # work around different normalization schemes GH#22342
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        tm.assert_index_equal(result, expected)

        for item, expected in zip(dti, expected_months):
            result = item.month_name(locale=time_locale)
            expected = expected.capitalize()

            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", result)

            assert result == expected
        dti = dti.append(DatetimeIndex([NaT]))
        assert np.isnan(dti.month_name(locale=time_locale)[-1])

    def test_dti_week(self):
        # [FUNCTION DOC]
        # Verifies (GH#6538) that the ISO week of dates near new year in the Brussels
        # zone is identical for the index and its timestamps, with expected weeks 52, 1, 1.
        # GH#6538: Check that DatetimeIndex and its TimeStamp elements
        # return the same weekofyear accessor close to new year w/ tz
        dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
        dates = DatetimeIndex(dates, tz="Europe/Brussels")
        expected = [52, 1, 1]
        assert dates.isocalendar().week.tolist() == expected
        assert [d.weekofyear for d in dates] == expected

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_dti_fields(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (none or US/Eastern). Verifies (GH#13303) the
        # calendar field accessors of a daily nanosecond range over 1998 at selected
        # positions: year, month, day, time parts, weekday, ordinal day, ISO week,
        # quarter, days in month, and start/end flags; that every accessor has 365
        # entries; that integer-like accessors return an index retaining the name and
        # boolean ones return plain arrays; and boolean indexing by quarter start and by
        # leap year (empty result) selects the expected elements.
        # GH#13303
        dti = date_range(
            freq="D", start=datetime(1998, 1, 1), periods=365, tz=tz, unit="ns"
        )
        assert dti.year[0] == 1998
        assert dti.month[0] == 1
        assert dti.day[0] == 1
        assert dti.hour[0] == 0
        assert dti.minute[0] == 0
        assert dti.second[0] == 0
        assert dti.microsecond[0] == 0
        assert dti.dayofweek[0] == 3

        assert dti.dayofyear[0] == 1
        assert dti.dayofyear[120] == 121

        assert dti.isocalendar().week.iloc[0] == 1
        assert dti.isocalendar().week.iloc[120] == 18

        assert dti.quarter[0] == 1
        assert dti.quarter[120] == 2

        assert dti.days_in_month[0] == 31
        assert dti.days_in_month[90] == 30

        assert dti.is_month_start[0]
        assert not dti.is_month_start[1]
        assert dti.is_month_start[31]
        assert dti.is_quarter_start[0]
        assert dti.is_quarter_start[90]
        assert dti.is_year_start[0]
        assert not dti.is_year_start[364]
        assert not dti.is_month_end[0]
        assert dti.is_month_end[30]
        assert not dti.is_month_end[31]
        assert dti.is_month_end[364]
        assert not dti.is_quarter_end[0]
        assert not dti.is_quarter_end[30]
        assert dti.is_quarter_end[89]
        assert dti.is_quarter_end[364]
        assert not dti.is_year_end[0]
        assert dti.is_year_end[364]

        assert len(dti.year) == 365
        assert len(dti.month) == 365
        assert len(dti.day) == 365
        assert len(dti.hour) == 365
        assert len(dti.minute) == 365
        assert len(dti.second) == 365
        assert len(dti.microsecond) == 365
        assert len(dti.dayofweek) == 365
        assert len(dti.dayofyear) == 365
        assert len(dti.isocalendar()) == 365
        assert len(dti.quarter) == 365
        assert len(dti.is_month_start) == 365
        assert len(dti.is_month_end) == 365
        assert len(dti.is_quarter_start) == 365
        assert len(dti.is_quarter_end) == 365
        assert len(dti.is_year_start) == 365
        assert len(dti.is_year_end) == 365

        dti.name = "name"

        # non boolean accessors -> return Index
        for accessor in DatetimeArray._field_ops:
            res = getattr(dti, accessor)
            assert len(res) == 365
            assert isinstance(res, Index)
            assert res.name == "name"

        # boolean accessors -> return array
        for accessor in DatetimeArray._bool_ops:
            res = getattr(dti, accessor)
            assert len(res) == 365
            assert isinstance(res, np.ndarray)

        # test boolean indexing
        res = dti[dti.is_quarter_start]
        exp = dti[[0, 90, 181, 273]]
        tm.assert_index_equal(res, exp)
        res = dti[dti.is_leap_year]
        exp = DatetimeIndex([], freq="D", tz=dti.tz, name="name").as_unit("ns")
        tm.assert_index_equal(res, exp)

    def test_dti_is_year_quarter_start(self):
        # [FUNCTION DOC]
        # Verifies for a business-quarter-end range with a February fiscal year that
        # there are no quarter or year starts, four quarter ends and exactly one year end.
        dti = date_range(freq="BQE-FEB", start=datetime(1998, 1, 1), periods=4)

        assert sum(dti.is_quarter_start) == 0
        assert sum(dti.is_quarter_end) == 4
        assert sum(dti.is_year_start) == 0
        assert sum(dti.is_year_end) == 1

    def test_dti_is_month_start(self):
        # [FUNCTION DOC]
        # Verifies the month-start flag is truthy for the first element of a short index
        # starting on the first of a month.
        dti = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])

        assert dti.is_month_start[0] == 1

    def test_dti_is_month_start_custom(self):
        # [FUNCTION DOC]
        # Verifies that asking for the month-start flag on a custom business day
        # range raises a value error stating custom business days are not supported.
        # Ensure is_start/end accessors throw ValueError for CustomBusinessDay,
        bday_egypt = offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")
        dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
        msg = "Custom business days is not supported by is_month_start"
        with pytest.raises(ValueError, match=msg):
            dti.is_month_start

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "MS", 3, np.array([False, True, False])),
            ("2017-12-01", "QS", 3, np.array([True, False, False])),
            ("2017-12-01", "YS", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_year_start(self, timestamp, freq, periods, expected_values):
        # [FUNCTION DOC]
        # Parametrized over start timestamp, frequency, number of periods and expected
        # flags. Verifies (GH57377) the year-start flag of month, quarter and year start
        # ranges beginning in December.
        # GH57377
        result = date_range(timestamp, freq=freq, periods=periods).is_year_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "ME", 3, np.array([True, False, False])),
            ("2017-12-01", "QE", 3, np.array([True, False, False])),
            ("2017-12-01", "YE", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_year_end(self, timestamp, freq, periods, expected_values):
        # [FUNCTION DOC]
        # Parametrized over start timestamp, frequency, periods and expected flags.
        # Verifies (GH57377) the year-end flag of month, quarter and year end ranges
        # beginning in December.
        # GH57377
        result = date_range(timestamp, freq=freq, periods=periods).is_year_end
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "MS", 3, np.array([False, True, False])),
            ("2017-12-01", "QS", 3, np.array([True, True, True])),
            ("2017-12-01", "YS", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_quarter_start(self, timestamp, freq, periods, expected_values):
        # [FUNCTION DOC]
        # Parametrized over start timestamp, frequency, periods and expected flags.
        # Verifies (GH57377) the quarter-start flag for month, quarter and year start
        # ranges beginning in December.
        # GH57377
        result = date_range(timestamp, freq=freq, periods=periods).is_quarter_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "ME", 3, np.array([True, False, False])),
            ("2017-12-01", "QE", 3, np.array([True, True, True])),
            ("2017-12-01", "YE", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_quarter_end(self, timestamp, freq, periods, expected_values):
        # [FUNCTION DOC]
        # Parametrized over start timestamp, frequency, periods and expected flags.
        # Verifies (GH57377) the quarter-end flag for month, quarter and year end ranges
        # beginning in December.
        # GH57377
        result = date_range(timestamp, freq=freq, periods=periods).is_quarter_end
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "MS", 3, np.array([True, True, True])),
            ("2017-12-01", "QS", 3, np.array([True, True, True])),
            ("2017-12-01", "YS", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_month_start(self, timestamp, freq, periods, expected_values):
        # [FUNCTION DOC]
        # Parametrized over start timestamp, frequency, periods and expected flags.
        # Verifies (GH57377) the month-start flag is always true for month, quarter and
        # year start ranges.
        # GH57377
        result = date_range(timestamp, freq=freq, periods=periods).is_month_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        "timestamp, freq, periods, expected_values",
        [
            ("2017-12-01", "ME", 3, np.array([True, True, True])),
            ("2017-12-01", "QE", 3, np.array([True, True, True])),
            ("2017-12-01", "YE", 3, np.array([True, True, True])),
        ],
    )
    def test_dti_dr_is_month_end(self, timestamp, freq, periods, expected_values):
        # [FUNCTION DOC]
        # Parametrized over start timestamp, frequency, periods and expected flags.
        # Verifies (GH57377) the month-end flag is always true for month, quarter and
        # year end ranges.
        # GH57377
        result = date_range(timestamp, freq=freq, periods=periods).is_month_end
        tm.assert_numpy_array_equal(result, expected_values)

    def test_dti_is_year_quarter_start_doubledigit_freq(self):
        # [FUNCTION DOC]
        # Verifies (GH#58523) that year and quarter start flags are true for every
        # element when the frequency has a multi-digit multiple of a year or quarter start.
        # GH#58523
        dr = date_range("2017-01-01", periods=2, freq="10YS")
        assert all(dr.is_year_start)

        dr = date_range("2017-01-01", periods=2, freq="10QS")
        assert all(dr.is_quarter_start)

    def test_dti_is_year_start_freq_custom_business_day_with_digit(self):
        # [FUNCTION DOC]
        # Verifies (GH#58664) that asking for the year-start flag on a custom business
        # day range with a numeric multiple raises a value error about custom business days.
        # GH#58664
        dr = date_range("2020-01-01", periods=2, freq="2C")
        msg = "Custom business days is not supported by is_year_start"
        with pytest.raises(ValueError, match=msg):
            dr.is_year_start

    @pytest.mark.parametrize("freq", ["3BMS", offsets.BusinessMonthBegin(3)])
    def test_dti_is_year_quarter_start_freq_business_month_begin(self, freq):
        # [FUNCTION DOC]
        # Parametrized over a three-month business-month-begin frequency given as a
        # string or an offset. Verifies (GH#58729) the year-start flags of the elements
        # follow the pattern true, false, false, false, true over five periods, and that
        # all quarter-start flags are true over four periods.
        # KNOWN ISSUE: the second per-element list of quarter-start flags is computed into a variable that is never used; the assertion checks the index-level flags instead, so the per-element path is not verified.
        # GH#58729
        dr = date_range("2020-01-01", periods=5, freq=freq)
        result = [x.is_year_start for x in dr]
        assert result == [True, False, False, False, True]

        dr = date_range("2020-01-01", periods=4, freq=freq)
        result = [x.is_quarter_start for x in dr]
        assert all(dr.is_quarter_start)

    def test_dti_is_year_start_freq_two_business_days(self):
        # [FUNCTION DOC]
        # Verifies (GH#58524) the year-start flags for a two-business-day range starting
        # on a January first are true then false.
        # GH#58524
        dr = date_range("2017-01-01", periods=2, freq="2B")
        result = dr.is_year_start
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)


@given(
    dt=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
    n=st.integers(min_value=1, max_value=10),
    freq=st.sampled_from(["MS", "QS", "YS"]),
)
@pytest.mark.slow
def test_against_scalar_parametric(freq, dt, n):
    # [FUNCTION DOC]
    # Slow, property-based test over generated start datetimes (1960-1980), a
    # multiplier from one to ten, and month/quarter/year start frequencies. Verifies
    # (see issue 49606) that the vectorised year-start flags of a three-period range
    # equal the flags of its individual elements.
    # https://github.com/pandas-dev/pandas/issues/49606
    freq = f"{n}{freq}"
    d = date_range(dt, periods=3, freq=freq)
    result = list(d.is_year_start)
    expected = [x.is_year_start for x in d]
    assert result == expected
