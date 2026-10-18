# [MODULE DOC]
# Tests for the datetime-like accessor of Series: properties and methods for datetime,
# timezone-aware, timedelta and period values, rounding, name accessors with locales,
# string formatting, missing values, categorical delegation, ISO calendar and period end
# time. Defines module-level lists of the allowed properties and methods per dtype
# (taken from the array classes for properties) and a helper to list the public accessor
# names.
import calendar
from datetime import (
    date,
    datetime,
    time,
)
import locale
import unicodedata

import numpy as np
import pytest

from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
)

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)

ok_for_period = PeriodArray._datetimelike_ops
ok_for_period_methods = ["strftime", "to_timestamp", "asfreq"]
ok_for_dt = DatetimeArray._datetimelike_ops
ok_for_dt_methods = [
    "to_period",
    "to_pydatetime",
    "tz_localize",
    "tz_convert",
    "normalize",
    "strftime",
    "round",
    "floor",
    "ceil",
    "day_name",
    "month_name",
    "isocalendar",
    "as_unit",
]
ok_for_td = TimedeltaArray._datetimelike_ops
ok_for_td_methods = [
    "components",
    "to_pytimedelta",
    "total_seconds",
    "round",
    "floor",
    "ceil",
    "as_unit",
]


def get_dir(ser):
    # [FUNCTION DOC]
    # HELPER: returns the sorted unique list of public (not underscore-prefixed) names
    # in the accessor namespace of the given series, used to check the limited display
    # API. Input: ser (a series with a datetime-like dtype).
    # check limited display api
    results = [r for r in ser.dt.__dir__() if not r.startswith("_")]
    return sorted(set(results))


class TestSeriesDatetimeValues:
    # [CLASS DOC]
    # Groups tests of the datetime accessor of Series. Stateless apart from a comparison
    # helper method.
    def _compare(self, ser, name):
        # [FUNCTION DOC]
        # HELPER: compares the accessor property of the given name on a series with the
        # same property on an index built from the same values. Input: ser (series with
        # datetime-like values), name (property name). The expected value is the index
        # property, wrapped in a series with the original index and name when it is
        # array-like (integer arrays converted to int64), a frame stays a frame, and
        # scalars stay scalars. Scalars are compared with equality, frames and series
        # with the testing comparison helpers.
        # GH 7207, 11128
        # test .dt namespace accessor

        def get_expected(ser, prop):
            # [FUNCTION DOC]
            # HELPER: computes the expected property value from the index equivalent of
            # the series values. Input: ser, prop (name). Arrays become series with the
            # same index and name (integer arrays as int64); frames and scalars are
            # returned as they are.
            result = getattr(Index(ser._values), prop)
            if isinstance(result, np.ndarray):
                if is_integer_dtype(result):
                    result = result.astype("int64")
            elif not is_list_like(result) or isinstance(result, DataFrame):
                return result
            return Series(result, index=ser.index, name=ser.name)

        left = getattr(ser.dt, name)
        right = get_expected(ser, name)
        if not (is_list_like(left) and is_list_like(right)):
            assert left == right
        elif isinstance(left, DataFrame):
            tm.assert_frame_equal(left, right)
        else:
            tm.assert_series_equal(left, right)

    @pytest.mark.parametrize("freq", ["D", "s", "ms"])
    def test_dt_namespace_accessor_datetime64(self, freq):
        # [FUNCTION DOC]
        # Regression test (GH 7207, GH 11128). Parametrized over daily, second and
        # millisecond frequencies. For a datetime series: every property (except
        # frequency) matches the index equivalent; every method name is reachable;
        # python datetime conversion gives an object-dtype series; tz localize gives the
        # expected tz-aware series and the timezone is US/Eastern; the frequency matches
        # an inferred one; localize to UTC then convert to US/Eastern gives the expected
        # series.
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # datetimeindex
        dti = date_range("20130101", periods=5, freq=freq)
        ser = Series(dti, name="xxx")

        for prop in ok_for_dt:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)

        for prop in ok_for_dt_methods:
            getattr(ser.dt, prop)

        result = ser.dt.to_pydatetime()
        assert isinstance(result, Series)
        assert result.dtype == object

        result = ser.dt.tz_localize("US/Eastern")
        exp_values = DatetimeIndex(ser.values).tz_localize("US/Eastern")
        expected = Series(exp_values, index=ser.index, name="xxx")
        tm.assert_series_equal(result, expected)

        tz_result = result.dt.tz
        assert str(tz_result) == "US/Eastern"
        freq_result = ser.dt.freq
        assert freq_result == DatetimeIndex(ser.values, freq="infer").freq

        # let's localize, then convert
        result = ser.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        exp_values = (
            DatetimeIndex(ser.values).tz_localize("UTC").tz_convert("US/Eastern")
        )
        expected = Series(exp_values, index=ser.index, name="xxx")
        tm.assert_series_equal(result, expected)

    def test_dt_namespace_accessor_datetime64tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 7207, GH 11128). Same checks for a tz-aware US/Eastern
        # series: properties match, methods reachable, python datetime conversion gives
        # objects, converting to CET gives the expected series with a CET timezone, and
        # the frequency is inferred.
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # datetimeindex with tz
        dti = date_range("20130101", periods=5, tz="US/Eastern")
        ser = Series(dti, name="xxx")
        for prop in ok_for_dt:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)

        for prop in ok_for_dt_methods:
            getattr(ser.dt, prop)

        result = ser.dt.to_pydatetime()
        assert isinstance(result, Series)
        assert result.dtype == object

        result = ser.dt.tz_convert("CET")
        expected = Series(ser._values.tz_convert("CET"), index=ser.index, name="xxx")
        tm.assert_series_equal(result, expected)

        tz_result = result.dt.tz
        assert str(tz_result) == "CET"
        freq_result = ser.dt.freq
        assert freq_result == DatetimeIndex(ser.values, freq="infer").freq

    def test_dt_namespace_accessor_timedelta(self):
        # [FUNCTION DOC]
        # Regression test (GH 7207, GH 11128). Three timedelta series (day, second and
        # millisecond spacing, one with string index): properties match the index
        # versions, methods reachable, components is a frame with the series index,
        # conversion to python timedelta emits a Pandas4 deprecation warning and gives
        # an object array, total seconds is a float64 series, and frequency matches the
        # inferred frequency.
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # timedelta index
        cases = [
            Series(
                timedelta_range("1 day", periods=5), index=list("abcde"), name="xxx"
            ),
            Series(timedelta_range("1 day 01:23:45", periods=5, freq="s"), name="xxx"),
            Series(
                timedelta_range("2 days 01:23:45.012345", periods=5, freq="ms"),
                name="xxx",
            ),
        ]
        for ser in cases:
            for prop in ok_for_td:
                # we test freq below
                if prop != "freq":
                    self._compare(ser, prop)

            for prop in ok_for_td_methods:
                getattr(ser.dt, prop)

            result = ser.dt.components
            assert isinstance(result, DataFrame)
            tm.assert_index_equal(result.index, ser.index)

            msg = "The behavior of TimedeltaProperties.to_pytimedelta is deprecated"
            with tm.assert_produces_warning(Pandas4Warning, match=msg):
                result = ser.dt.to_pytimedelta()
            assert isinstance(result, np.ndarray)
            assert result.dtype == object

            result = ser.dt.total_seconds()
            assert isinstance(result, Series)
            assert result.dtype == "float64"

            freq_result = ser.dt.freq
            assert freq_result == TimedeltaIndex(ser.values, freq="infer").freq

    def test_dt_namespace_accessor_period(self):
        # [FUNCTION DOC]
        # Regression test (GH 7207, GH 11128). For a daily period series: properties
        # match, methods reachable, and the frequency equals that of the period index.
        # GH#7207, GH#11128
        # test .dt namespace accessor

        # periodindex
        pi = period_range("20130101", periods=5, freq="D")
        ser = Series(pi, name="xxx")

        for prop in ok_for_period:
            # we test freq below
            if prop != "freq":
                self._compare(ser, prop)

        for prop in ok_for_period_methods:
            getattr(ser.dt, prop)

        freq_result = ser.dt.freq
        assert freq_result == PeriodIndex(ser.values).freq

    def test_dt_namespace_accessor_index_and_values(self):
        # [FUNCTION DOC]
        # Verifies that with a datetime series having a different index than its values,
        # year, month and second properties return series with the series index, name
        # and int32 dtype, and normalizing keeps the index.
        # both
        index = date_range("20130101", periods=3, freq="D")
        dti = date_range("20140204", periods=3, freq="s")
        ser = Series(dti, index=index, name="xxx")
        exp = Series(
            np.array([2014, 2014, 2014], dtype="int32"), index=index, name="xxx"
        )
        tm.assert_series_equal(ser.dt.year, exp)

        exp = Series(np.array([2, 2, 2], dtype="int32"), index=index, name="xxx")
        tm.assert_series_equal(ser.dt.month, exp)

        exp = Series(np.array([0, 1, 2], dtype="int32"), index=index, name="xxx")
        tm.assert_series_equal(ser.dt.second, exp)

        exp = Series([ser.iloc[0]] * 3, index=index, name="xxx")
        tm.assert_series_equal(ser.dt.normalize(), exp)

    def test_dt_accessor_limited_display_api(self):
        # [FUNCTION DOC]
        # Verifies the namespace listing (public names) of the accessor equals the union
        # of the allowed properties and methods for tz-naive, tz-aware (UTC converted to
        # Chicago, over a year at minute frequency) and period series.
        # tznaive
        ser = Series(date_range("20130101", periods=5, freq="D"), name="xxx")
        results = get_dir(ser)
        tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))

        # tzaware
        ser = Series(date_range("2015-01-01", "2016-01-01", freq="min"), name="xxx")
        ser = ser.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")
        results = get_dir(ser)
        tm.assert_almost_equal(results, sorted(set(ok_for_dt + ok_for_dt_methods)))

        # Period
        idx = period_range("20130101", periods=5, freq="D", name="xxx")
        ser = Series(idx)
        results = get_dir(ser)
        tm.assert_almost_equal(
            results, sorted(set(ok_for_period + ok_for_period_methods))
        )

    def test_dt_accessor_ambiguous_freq_conversions(self):
        # [FUNCTION DOC]
        # Regression test (GH 11295). Localizing to UTC and converting to Chicago on a
        # minute-frequency series for a year gives the same values as converting the
        # index (with the frequency removed), with no ambiguous time errors.
        # GH#11295
        # ambiguous time error on the conversions
        ser = Series(date_range("2015-01-01", "2016-01-01", freq="min"), name="xxx")
        ser = ser.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")

        exp_values = date_range(
            "2015-01-01", "2016-01-01", freq="min", tz="UTC"
        ).tz_convert("America/Chicago")
        # freq not preserved by tz_localize above
        exp_values = exp_values._with_freq(None)
        expected = Series(exp_values, name="xxx")
        tm.assert_series_equal(ser, expected)

    def test_dt_accessor_not_writeable(self):
        # [FUNCTION DOC]
        # Verifies setting an accessor property raises a value error saying
        # modifications are not allowed, and setting an element of the returned copy
        # raises the chained assignment error.
        # no setting allowed
        ser = Series(date_range("20130101", periods=5, freq="D"), name="xxx")
        with pytest.raises(ValueError, match="modifications"):
            ser.dt.hour = 5

        # trying to set a copy
        with tm.raises_chained_assignment_error():
            ser.dt.hour[0] = 5

    @pytest.mark.parametrize(
        "method, dates",
        [
            ["round", ["2012-01-02", "2012-01-02", "2012-01-01"]],
            ["floor", ["2012-01-01", "2012-01-01", "2012-01-01"]],
            ["ceil", ["2012-01-02", "2012-01-02", "2012-01-02"]],
        ],
    )
    def test_dt_round(self, method, dates):
        # [FUNCTION DOC]
        # Parametrized over round, floor and ceil with expected dates. Applying the
        # daily rounding to three times on 2012-01-01 gives the expected dates in a
        # series that keeps the name.
        # round
        ser = Series(
            pd.to_datetime(
                ["2012-01-01 13:00:00", "2012-01-01 12:01:00", "2012-01-01 08:00:00"]
            ),
            name="xxx",
        )
        result = getattr(ser.dt, method)("D")
        expected = Series(pd.to_datetime(dates), name="xxx")
        tm.assert_series_equal(result, expected)

    def test_dt_round_tz(self):
        # [FUNCTION DOC]
        # Verifies daily rounding of a series localized to UTC, converted to US/Eastern
        # gives the expected aware series (all on 2012-01-01 midnight US/Eastern).
        ser = Series(
            pd.to_datetime(
                ["2012-01-01 13:00:00", "2012-01-01 12:01:00", "2012-01-01 08:00:00"]
            ),
            name="xxx",
        )
        result = ser.dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.round("D")

        exp_values = pd.to_datetime(
            ["2012-01-01", "2012-01-01", "2012-01-01"]
        ).tz_localize("US/Eastern")
        expected = Series(exp_values, name="xxx")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    def test_dt_round_tz_ambiguous(self, method):
        # [FUNCTION DOC]
        # Regression test (GH 18946). Parametrized over ceil, round, floor with hourly
        # rounding on Madrid times around the fall-back transition: ambiguous inference
        # and an explicit bool array leave the series unchanged; the NaT option turns
        # the first two (ambiguous) elements into NaT; the raise option raises a value
        # error from the external library.
        # GH 18946 round near "fall back" DST
        df1 = DataFrame(
            [
                pd.to_datetime("2017-10-29 02:00:00+02:00", utc=True),
                pd.to_datetime("2017-10-29 02:00:00+01:00", utc=True),
                pd.to_datetime("2017-10-29 03:00:00+01:00", utc=True),
            ],
            columns=["date"],
        )
        df1["date"] = df1["date"].dt.tz_convert("Europe/Madrid")
        # infer
        result = getattr(df1.date.dt, method)("h", ambiguous="infer")
        expected = df1["date"]
        tm.assert_series_equal(result, expected)

        # bool-array
        result = getattr(df1.date.dt, method)("h", ambiguous=[True, False, False])
        tm.assert_series_equal(result, expected)

        # NaT
        result = getattr(df1.date.dt, method)("h", ambiguous="NaT")
        expected = df1["date"].copy()
        expected.iloc[0:2] = pd.NaT
        tm.assert_series_equal(result, expected)

        # raise
        with tm.external_error_raised(ValueError):
            getattr(df1.date.dt, method)("h", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2h"],
        ],
    )
    def test_dt_round_tz_nonexistent(self, method, ts_str, freq):
        # [FUNCTION DOC]
        # Regression test (GH 23324). Parametrized over method, input time and frequency
        # near the Chicago spring-forward gap. The shift-forward option gives 03:00
        # Chicago (second unit), the NaT option gives NaT with the timezone kept, and
        # the raise option raises a value error mentioning the nonexistent time.
        # GH 23324 round near "spring forward" DST
        ser = Series([pd.Timestamp(ts_str, tz="America/Chicago").as_unit("s")])
        result = getattr(ser.dt, method)(freq, nonexistent="shift_forward")
        expected = Series(
            [pd.Timestamp("2018-03-11 03:00:00", tz="America/Chicago").as_unit("s")]
        )
        tm.assert_series_equal(result, expected)

        result = getattr(ser.dt, method)(freq, nonexistent="NaT")
        expected = Series([pd.NaT]).dt.tz_localize(result.dt.tz)
        tm.assert_series_equal(result, expected)

        with pytest.raises(ValueError, match="2018-03-11 02:00:00"):
            getattr(ser.dt, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize("freq", ["ns", "us", "1000us"])
    def test_dt_round_nonnano_higher_resolution_no_op(self, freq):
        # [FUNCTION DOC]
        # Regression test (GH 52761). Parametrized over nanosecond and microsecond
        # frequencies on a millisecond series. Rounding to a finer unit returns an equal
        # series that does not share memory with the original.
        # GH 52761
        ser = Series(
            ["2020-05-31 08:00:00", "2000-12-31 04:00:05", "1800-03-14 07:30:20"],
            dtype="datetime64[ms]",
        )
        expected = ser.copy()
        result = ser.dt.round(freq)
        tm.assert_series_equal(result, expected)

        assert not np.shares_memory(ser.array._ndarray, result.array._ndarray)

    def test_dt_namespace_accessor_categorical(self):
        # [FUNCTION DOC]
        # Regression test (GH 19468). Year property of a categorical series of repeated
        # dates gives int32 values with the series name.
        # GH 19468
        dti = DatetimeIndex(["20171111", "20181212"]).repeat(2)
        ser = Series(pd.Categorical(dti), name="foo")
        result = ser.dt.year
        expected = Series([2017, 2017, 2018, 2018], dtype="int32", name="foo")
        tm.assert_series_equal(result, expected)

    def test_dt_tz_localize_categorical(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Regression test (GH 27952). Localizing a categorical series of datetimes to
        # each aware timezone fixture equals localizing the original.
        # GH 27952
        tz = tz_aware_fixture
        datetimes = Series(
            ["2019-01-01", "2019-01-01", "2019-01-02"], dtype="datetime64[ns]"
        )
        categorical = datetimes.astype("category")
        result = categorical.dt.tz_localize(tz)
        expected = datetimes.dt.tz_localize(tz)
        tm.assert_series_equal(result, expected)

    def test_dt_tz_convert_categorical(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Regression test (GH 27952). Converting a categorical series of datetimes with
        # a MET timezone to each aware timezone fixture equals converting the original.
        # GH 27952
        tz = tz_aware_fixture
        datetimes = Series(
            ["2019-01-01", "2019-01-01", "2019-01-02"], dtype="datetime64[ns, MET]"
        )
        categorical = datetimes.astype("category")
        result = categorical.dt.tz_convert(tz)
        expected = datetimes.dt.tz_convert(tz)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("accessor", ["year", "month", "day"])
    def test_dt_other_accessors_categorical(self, accessor):
        # [FUNCTION DOC]
        # Regression test (GH 27952). Parametrized over year, month and day. The
        # property on a categorical datetime series equals the property on the original.
        # GH 27952
        datetimes = Series(
            ["2018-01-01", "2018-01-01", "2019-01-02"], dtype="datetime64[ns]"
        )
        categorical = datetimes.astype("category")
        result = getattr(categorical.dt, accessor)
        expected = getattr(datetimes.dt, accessor)
        tm.assert_series_equal(result, expected)

    def test_dt_accessor_no_new_attributes(self):
        # [FUNCTION DOC]
        # Regression test for GH 10673. Setting a new attribute on the datetime accessor
        # raises an attribute error saying new attributes cannot be added.
        # https://github.com/pandas-dev/pandas/issues/10673
        ser = Series(date_range("20130101", periods=5, freq="D"))
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):
            ser.dt.xlabel = "a"

    @pytest.mark.parametrize(
        "time_locale",
        [None, *tm.get_locales()],
    )
    def test_dt_accessor_datetime_name_accessors(self, time_locale):
        # [FUNCTION DOC]
        # Parametrized over no locale plus all available system locales. For a year of
        # daily dates, the day names for seven days equal the locale's (capitalized)
        # names and the default locale gives English names; a missing date gives NaN.
        # For monthly dates, month names (normalized to NFD form to work around GH
        # 22342) equal the locale's names, both in the accessor and on each scalar
        # timestamp; a missing date gives NaN.
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

        ser = Series(date_range(freq="D", start=datetime(1998, 1, 1), periods=365))
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
            assert ser.dt.day_name(locale=time_locale)[day] == name
            assert ser.dt.day_name(locale=None)[day] == eng_name
        ser = pd.concat([ser, Series([pd.NaT])])
        assert np.isnan(ser.dt.day_name(locale=time_locale).iloc[-1])

        ser = Series(date_range(freq="ME", start="2012", end="2013"))
        result = ser.dt.month_name(locale=time_locale)
        expected = Series([month.capitalize() for month in expected_months])

        # work around https://github.com/pandas-dev/pandas/issues/22342
        result = result.str.normalize("NFD")
        expected = expected.str.normalize("NFD")

        tm.assert_series_equal(result, expected)

        for s_date, expected in zip(ser, expected_months):
            result = s_date.month_name(locale=time_locale)
            expected = expected.capitalize()

            result = unicodedata.normalize("NFD", result)
            expected = unicodedata.normalize("NFD", expected)

            assert result == expected

        ser = pd.concat([ser, Series([pd.NaT])])
        assert np.isnan(ser.dt.month_name(locale=time_locale).iloc[-1])

    def test_strftime(self):
        # [FUNCTION DOC]
        # Regression test (GH 10086). String formatting with a pattern gives the
        # expected strings for datetime series (date only, and time with the fraction
        # dropped) and for period series (daily and seconds).
        # GH 10086
        ser = Series(date_range("20130101", periods=5))
        result = ser.dt.strftime("%Y/%m/%d")
        expected = Series(
            ["2013/01/01", "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        tm.assert_series_equal(result, expected)

        ser = Series(date_range("2015-02-03 11:22:33.4567", periods=5))
        result = ser.dt.strftime("%Y/%m/%d %H-%M-%S")
        expected = Series(
            [
                "2015/02/03 11-22-33",
                "2015/02/04 11-22-33",
                "2015/02/05 11-22-33",
                "2015/02/06 11-22-33",
                "2015/02/07 11-22-33",
            ]
        )
        tm.assert_series_equal(result, expected)

        ser = Series(period_range("20130101", periods=5))
        result = ser.dt.strftime("%Y/%m/%d")
        expected = Series(
            ["2013/01/01", "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        tm.assert_series_equal(result, expected)

        ser = Series(period_range("2015-02-03 11:22:33.4567", periods=5, freq="s"))
        result = ser.dt.strftime("%Y/%m/%d %H-%M-%S")
        expected = Series(
            [
                "2015/02/03 11-22-33",
                "2015/02/03 11-22-34",
                "2015/02/03 11-22-35",
                "2015/02/03 11-22-36",
                "2015/02/03 11-22-37",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_strftime_dt64_days(self):
        # [FUNCTION DOC]
        # Verifies string formatting of a day-resolution datetime series with a missing
        # first element gives NaN there and the formatted dates elsewhere, and the index
        # version gives the expected index of formatted strings.
        # KNOWN ISSUE: the comment about the dtype being S10 or U10 depending on python
        # version is stale; no dtype is asserted.
        ser = Series(date_range("20130101", periods=5))
        ser.iloc[0] = pd.NaT
        result = ser.dt.strftime("%Y/%m/%d")
        expected = Series(
            [np.nan, "2013/01/02", "2013/01/03", "2013/01/04", "2013/01/05"]
        )
        tm.assert_series_equal(result, expected)

        datetime_index = date_range("20150301", periods=5)
        result = datetime_index.strftime("%Y/%m/%d")

        expected = Index(
            ["2015/03/01", "2015/03/02", "2015/03/03", "2015/03/04", "2015/03/05"],
        )
        # dtype may be S10 or U10 depending on python version
        tm.assert_index_equal(result, expected)

    def test_strftime_period_days(self, using_infer_string):
        # [FUNCTION DOC]
        # Verifies string formatting of a daily period index returns an index of
        # formatted dates with fixed-width unicode dtype, or the string dtype with NaN
        # as missing when the infer-string option is on.
        period_index = period_range("20150301", periods=5)
        result = period_index.strftime("%Y/%m/%d")
        expected = Index(
            ["2015/03/01", "2015/03/02", "2015/03/03", "2015/03/04", "2015/03/05"],
            dtype="=U10",
        )
        if using_infer_string:
            expected = expected.astype(StringDtype(na_value=np.nan))
        tm.assert_index_equal(result, expected)

    def test_strftime_dt64_microsecond_resolution(self):
        # [FUNCTION DOC]
        # Verifies string formatting of a series of python datetimes (microsecond
        # resolution) with a time pattern.
        ser = Series([datetime(2013, 1, 1, 2, 32, 59), datetime(2013, 1, 2, 14, 32, 1)])
        result = ser.dt.strftime("%Y-%m-%d %H:%M:%S")
        expected = Series(["2013-01-01 02:32:59", "2013-01-02 14:32:01"])
        tm.assert_series_equal(result, expected)

    def test_strftime_period_hours(self):
        # [FUNCTION DOC]
        # Verifies string formatting of an hourly period series with date and time
        # pattern gives hour-aligned strings.
        ser = Series(period_range("20130101", periods=4, freq="h"))
        result = ser.dt.strftime("%Y/%m/%d %H:%M:%S")
        expected = Series(
            [
                "2013/01/01 00:00:00",
                "2013/01/01 01:00:00",
                "2013/01/01 02:00:00",
                "2013/01/01 03:00:00",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_strftime_period_minutes(self):
        # [FUNCTION DOC]
        # Verifies string formatting with the milliseconds directive on a
        # millisecond-frequency period series.
        # KNOWN ISSUE: the name says minutes but the period frequency is milliseconds
        # and the pattern shows milliseconds.
        ser = Series(period_range("20130101", periods=4, freq="ms"))
        result = ser.dt.strftime("%Y/%m/%d %H:%M:%S.%l")
        expected = Series(
            [
                "2013/01/01 00:00:00.000",
                "2013/01/01 00:00:00.001",
                "2013/01/01 00:00:00.002",
                "2013/01/01 00:00:00.003",
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            DatetimeIndex(["2019-01-01", pd.NaT]),
            PeriodIndex(["2019-01-01", pd.NaT], dtype="period[D]"),
        ],
    )
    def test_strftime_nat(self, data):
        # [FUNCTION DOC]
        # Regression test (GH 29578). Parametrized over a datetime index and a period
        # index each with one missing value. Formatting gives the formatted date and NaN
        # for the missing one.
        # GH 29578
        ser = Series(data)
        result = ser.dt.strftime("%Y-%m-%d")
        expected = Series(["2019-01-01", np.nan])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data", [DatetimeIndex([pd.NaT]), PeriodIndex([pd.NaT], dtype="period[D]")]
    )
    def test_strftime_all_nat(self, data):
        # [FUNCTION DOC]
        # Regression test (GH 45858). Parametrized over datetime and period indexes made
        # entirely of NaT. Formatting emits no warning and gives a string-dtype series
        # of NaN.
        # https://github.com/pandas-dev/pandas/issues/45858
        ser = Series(data)
        with tm.assert_produces_warning(None):
            result = ser.dt.strftime("%Y-%m-%d")
        expected = Series([np.nan], dtype="str")
        tm.assert_series_equal(result, expected)

    def test_valid_dt_with_missing_values(self):
        # [FUNCTION DOC]
        # Regression test (GH 8689). With a missing element, the microsecond,
        # nanosecond, second, minute, hour and day properties give the same values with
        # NaN at the missing position; date gives python dates with NaT at that position
        # (object dtype) and time gives times with NaT (object dtype).
        # GH 8689
        ser = Series(date_range("20130101", periods=5, freq="D"))
        ser.iloc[2] = pd.NaT

        for attr in ["microsecond", "nanosecond", "second", "minute", "hour", "day"]:
            expected = getattr(ser.dt, attr).copy()
            expected.iloc[2] = np.nan
            result = getattr(ser.dt, attr)
            tm.assert_series_equal(result, expected)

        result = ser.dt.date
        expected = Series(
            [
                date(2013, 1, 1),
                date(2013, 1, 2),
                pd.NaT,
                date(2013, 1, 4),
                date(2013, 1, 5),
            ],
            dtype="object",
        )
        tm.assert_series_equal(result, expected)

        result = ser.dt.time
        expected = Series([time(0), time(0), pd.NaT, time(0), time(0)], dtype="object")
        tm.assert_series_equal(result, expected)

    def test_dt_accessor_api(self):
        # [FUNCTION DOC]
        # Regression test (GH 9322). The class attribute is the combined datetime-like
        # properties class and an instance on a datetime series is the datetime
        # properties class.
        # GH 9322
        from pandas.core.indexes.accessors import (
            CombinedDatetimelikeProperties,
            DatetimeProperties,
        )

        assert Series.dt is CombinedDatetimelikeProperties

        ser = Series(date_range("2000-01-01", periods=3))
        assert isinstance(ser.dt, DatetimeProperties)

    @pytest.mark.parametrize(
        "data",
        [
            np.arange(5),
            list("abcde"),
            np.random.default_rng(2).standard_normal(5),
        ],
    )
    def test_dt_accessor_invalid(self, data):
        # [FUNCTION DOC]
        # Regression test (GH 9322). Parametrized over integers, strings and floats. The
        # accessor raises an attribute error saying only datetime-like values are
        # allowed, and the attribute is reported absent.
        # GH#9322 check that series with incorrect dtypes don't have attr
        ser = Series(data)
        with pytest.raises(AttributeError, match="only use .dt accessor"):
            ser.dt
        assert not hasattr(ser, "dt")

    def test_dt_accessor_updates_on_inplace(self):
        # [FUNCTION DOC]
        # Verifies the accessor reflects an in-place fill: after setting an element to
        # missing and filling in place with a timestamp (which returns the series
        # itself), the first and third dates are equal.
        ser = Series(date_range("2018-01-01", periods=10))
        ser[2] = None
        result = ser.fillna(pd.Timestamp("2018-01-01"), inplace=True)
        assert result is ser
        result = ser.dt.date
        assert result[0] == result[2]

    def test_date_tz(self):
        # [FUNCTION DOC]
        # Regression test (GH 11757). The date property of an aware series equals the
        # expected python dates and the apply of a date function gives the same series.
        # GH11757
        rng = DatetimeIndex(
            ["2014-04-04 23:56", "2014-07-18 21:24", "2015-11-22 22:14"],
            tz="US/Eastern",
        )
        ser = Series(rng)
        expected = Series([date(2014, 4, 4), date(2014, 7, 18), date(2015, 11, 22)])
        tm.assert_series_equal(ser.dt.date, expected)
        tm.assert_series_equal(ser.apply(lambda x: x.date()), expected)

    def test_dt_timetz_accessor(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Regression test (GH 21358). Over the naive/timezone fixture, the timetz
        # property of a series gives python times with the timezone attached.
        # GH21358
        tz = maybe_get_tz(tz_naive_fixture)

        dtindex = DatetimeIndex(
            ["2014-04-04 23:56", "2014-07-18 21:24", "2015-11-22 22:14"], tz=tz
        )
        ser = Series(dtindex)
        expected = Series(
            [time(23, 56, tzinfo=tz), time(21, 24, tzinfo=tz), time(22, 14, tzinfo=tz)]
        )
        result = ser.dt.timetz
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_series, expected_output",
        [
            [["2020-01-01"], [[2020, 1, 3]]],
            [[pd.NaT], [[None, None, None]]],
            [["2019-12-31", "2019-12-29"], [[2020, 1, 2], [2019, 52, 7]]],
            [["2010-01-01", pd.NaT], [[2009, 53, 5], [None, None, None]]],
            # see GH#36032
            [["2016-01-08", "2016-01-04"], [[2016, 1, 5], [2016, 1, 1]]],
            [["2016-01-07", "2016-01-01"], [[2016, 1, 4], [2015, 53, 5]]],
        ],
    )
    def test_isocalendar(self, input_series, expected_output):
        # [FUNCTION DOC]
        # Parametrized over inputs (including NaT) with expected year, week and day
        # triples (GH 36032 for year edges). The ISO calendar frame has unsigned 32-bit
        # integer columns year, week and day with missing values where the input is
        # missing.
        result = pd.to_datetime(Series(input_series)).dt.isocalendar()
        expected_frame = DataFrame(
            expected_output, columns=["year", "week", "day"], dtype="UInt32"
        )
        tm.assert_frame_equal(result, expected_frame)

    def test_hour_index(self):
        # [FUNCTION DOC]
        # Verifies the hour property of a categorical hourly series with a non-default
        # index returns an int32 series with the same index.
        dt_series = Series(
            date_range(start="2021-01-01", periods=5, freq="h"),
            index=[2, 6, 7, 8, 11],
            dtype="category",
        )
        result = dt_series.dt.hour
        expected = Series(
            [0, 1, 2, 3, 4],
            dtype="int32",
            index=[2, 6, 7, 8, 11],
        )
        tm.assert_series_equal(result, expected)


class TestSeriesPeriodValuesDtAccessor:
    # [CLASS DOC]
    # Groups tests of the datetime accessor on period series. Stateless.
    @pytest.mark.parametrize(
        "input_vals",
        [
            [Period("2016-01", freq="M"), Period("2016-02", freq="M")],
            [Period("2016-01-01", freq="D"), Period("2016-01-02", freq="D")],
            [
                Period("2016-01-01 00:00:00", freq="h"),
                Period("2016-01-01 01:00:00", freq="h"),
            ],
            [
                Period("2016-01-01 00:00:00", freq="M"),
                Period("2016-01-01 00:01:00", freq="M"),
            ],
            [
                Period("2016-01-01 00:00:00", freq="s"),
                Period("2016-01-01 00:00:01", freq="s"),
            ],
        ],
    )
    def test_end_time_timevalues(self, input_vals):
        # [FUNCTION DOC]
        # Regression test (GH 17157). Parametrized over period arrays with several
        # frequencies. The end time property equals the end time computed per element by
        # apply.
        # GH#17157
        # Check that the time part of the Period is adjusted by end_time
        # when using the dt accessor on a Series
        dtype = pd.PeriodDtype(input_vals[0].freq)
        input_vals = PeriodArray._from_sequence(np.asarray(input_vals), dtype=dtype)

        ser = Series(input_vals)
        result = ser.dt.end_time
        expected = ser.apply(lambda x: x.end_time)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("input_vals", [("2001"), ("NaT")])
    def test_to_period(self, input_vals):
        # [FUNCTION DOC]
        # Regression test (GH 21205). Parametrized over a year string and NaT.
        # Converting a datetime series to a daily period series equals the expected
        # period series.
        # GH#21205
        expected = Series([input_vals], dtype="Period[D]")
        result = Series([input_vals], dtype="datetime64[ns]").dt.to_period("D")
        tm.assert_series_equal(result, expected)


def test_normalize_pre_epoch_dates():
    # [FUNCTION DOC]
    # Regression test (GH 36294). Normalizing a series with a pre-epoch date and a
    # current date gives midnights of the same days.
    # GH: 36294
    ser = pd.to_datetime(Series(["1969-01-01 09:00:00", "2016-01-01 09:00:00"]))
    result = ser.dt.normalize()
    expected = pd.to_datetime(Series(["1969-01-01", "2016-01-01"]))
    tm.assert_series_equal(result, expected)


def test_day_attribute_non_nano_beyond_int32():
    # [FUNCTION DOC]
    # Regression test (GH 52386). A timedelta series with second resolution and very
    # large values gives day counts beyond the 32-bit range correctly.
    # GH 52386
    data = np.array(
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        dtype="timedelta64[s]",
    )
    ser = Series(data)
    result = ser.dt.days
    expected = Series([1579371003, 1559453522, 2839645203, 2586, 27, 42066, 0])
    tm.assert_series_equal(result, expected)
