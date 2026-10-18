# [MODULE DOC]
# Tests for the matplotlib unit converters and tick formatters of pandas
# plotting (datetime, period and timedelta converters, the time formatter, the
# quarterly tick finder) and for registration of the converters with the
# matplotlib units registry. Public contents: module-level tests plus the test
# classes TestRegistration, TestDateTimeConverter, TestPeriodConverter and
# TestTimeDeltaConverter. Import-time behavior: skips the module if the
# matplotlib pyplot, dates or units modules are unavailable, and imports the
# pandas matplotlib converter module only afterwards.

from datetime import (
    date,
    datetime,
)
import subprocess
import sys

import numpy as np
import pytest

import pandas._config.config as cf

from pandas._libs.tslibs import to_offset

from pandas import (
    Index,
    Period,
    PeriodIndex,
    Series,
    Timestamp,
    arrays,
    date_range,
)
import pandas._testing as tm

from pandas.plotting import (
    deregister_matplotlib_converters,
    register_matplotlib_converters,
)
from pandas.tseries.offsets import (
    Day,
    Micro,
    Milli,
    Second,
)

plt = pytest.importorskip("matplotlib.pyplot")
dates = pytest.importorskip("matplotlib.dates")
units = pytest.importorskip("matplotlib.units")

from pandas.plotting._matplotlib import converter


@pytest.mark.single_cpu
def test_registry_mpl_resets():
    # [FUNCTION DOC]
    # Marked to run in a single process. Runs a fresh interpreter that records the
    # number of registered converters, registers and then deregisters the pandas
    # converters, and asserts the count is unchanged (issue 27481). The subprocess
    # failing raises a called-process error.
    # Check that Matplotlib converters are properly reset (see issue #27481)
    code = (
        "import matplotlib.units as units; "
        "import matplotlib.dates as mdates; "
        "n_conv = len(units.registry); "
        "import pandas as pd; "
        "pd.plotting.register_matplotlib_converters(); "
        "pd.plotting.deregister_matplotlib_converters(); "
        "assert len(units.registry) == n_conv"
    )
    call = [sys.executable, "-c", code]
    subprocess.check_output(call)


def test_timtetonum_accepts_unicode():
    # [FUNCTION DOC]
    # Checks the time-to-number conversion of a time string.
    # KNOWN ISSUE: the test name has a typo ("timtetonum") and it compares the
    # result of the identical call with itself, so it cannot fail unless the call
    # raises; it does not actually check the unicode behavior.
    assert converter.time2num("00:01") == converter.time2num("00:01")


class TestRegistration:
    # [CLASS DOC]
    # Tests of registering and deregistering the pandas converters in matplotlib.
    @pytest.mark.single_cpu
    def test_dont_register_by_default(self):
        # [FUNCTION DOC]
        # Marked to run in a single process. In a fresh subprocess, importing pandas
        # must not register its timestamp type in the matplotlib units registry.
        # Run in subprocess to ensure a clean state
        code = (
            "import matplotlib.units; "
            "import pandas as pd; "
            "units = dict(matplotlib.units.registry); "
            "assert pd.Timestamp not in units"
        )
        call = [sys.executable, "-c", code]
        assert subprocess.check_call(call) == 0

    def test_registering_no_warning(self):
        # [FUNCTION DOC]
        # After explicit registration, plotting datetime index values on a matplotlib
        # axes runs without error (no warning check is made explicitly).
        s = Series(range(12), index=date_range("2017", periods=12))
        _, ax = plt.subplots()

        # Set to the "warn" state, in case this isn't the first test run
        register_matplotlib_converters()
        ax.plot(s.index, s.values)

    def test_pandas_plots_register(self):
        # [FUNCTION DOC]
        # Plotting a series with a date index through the pandas plot method must not
        # emit any warning (checked with the testing warning helper expecting none).
        # KNOWN ISSUE: the comment says to set the warn state first, but the test does
        # nothing to do so.
        s = Series(range(12), index=date_range("2017", periods=12))
        # Set to the "warn" state, in case this isn't the first test run
        with tm.assert_produces_warning(None) as w:
            s.plot()

        assert len(w) == 0

    def test_matplotlib_formatters(self):
        # [FUNCTION DOC]
        # Using the option context for the register converters option: turning it on
        # then off removes the timestamp converter from the units registry and the outer
        # context restores it.
        # Can't make any assertion about the start state.
        # We we check that toggling converters off removes it, and toggling it
        # on restores it.

        with cf.option_context("plotting.matplotlib.register_converters", True):
            with cf.option_context("plotting.matplotlib.register_converters", False):
                assert Timestamp not in units.registry
            assert Timestamp in units.registry

    def test_option_no_warning(self):
        # [FUNCTION DOC]
        # Plotting datetime values with the register option disabled, both before and
        # after explicit registration, must not raise.
        s = Series(range(12), index=date_range("2017", periods=12))
        _, ax = plt.subplots()

        # Test without registering first, no warning
        with cf.option_context("plotting.matplotlib.register_converters", False):
            ax.plot(s.index, s.values)

        # Now test with registering
        register_matplotlib_converters()
        with cf.option_context("plotting.matplotlib.register_converters", False):
            ax.plot(s.index, s.values)

    def test_registry_resets(self):
        # [FUNCTION DOC]
        # Saves the matplotlib registry, clears it and installs the matplotlib date
        # converter for datetime and date. Registering pandas converters replaces the
        # date entry; deregistering restores the original converter object. In all
        # cases the original registry content is restored in a finally block (side
        # effect: mutates and then restores global state).
        # make a copy, to reset to
        original = dict(units.registry)

        try:
            # get to a known state
            units.registry.clear()
            date_converter = dates.DateConverter()
            units.registry[datetime] = date_converter
            units.registry[date] = date_converter

            register_matplotlib_converters()
            assert units.registry[date] is not date_converter
            deregister_matplotlib_converters()
            assert units.registry[date] is date_converter

        finally:
            # restore original stater
            units.registry.clear()
            for k, v in original.items():
                units.registry[k] = v


class TestDateTimeConverter:
    # [CLASS DOC]
    # Tests of the pandas datetime converter used by matplotlib.
    @pytest.fixture
    def dtc(self):
        # [FUNCTION DOC]
        # Fixture returning a new datetime converter object.
        return converter.DatetimeConverter()

    def test_convert_accepts_unicode(self, dtc):
        # [FUNCTION DOC]
        # Converting the same date string twice gives equal results.
        # KNOWN ISSUE: both results come from the identical call, so the assertion is
        # trivially true and does not verify unicode handling.
        r1 = dtc.convert("2000-01-01 12:22", None, None)
        r2 = dtc.convert("2000-01-01 12:22", None, None)
        assert r1 == r2, "DatetimeConverter.convert should accept unicode"

    def test_conversion(self, dtc):
        # [FUNCTION DOC]
        # Converts many input kinds (list of date string, date strings, a date, a
        # timestamp, a timezone-aware timestamp, a native datetime, string array, index
        # of timestamps, native datetime array) and expects each to equal the matplotlib
        # number for the first of January 2012 (or the second element for index and
        # array inputs).
        rs = dtc.convert(["2012-1-1"], None, None)[0]
        xp = dates.date2num(datetime(2012, 1, 1))
        assert rs == xp

        rs = dtc.convert("2012-1-1", None, None)
        assert rs == xp

        rs = dtc.convert(date(2012, 1, 1), None, None)
        assert rs == xp

        rs = dtc.convert("2012-1-1", None, None)
        assert rs == xp

        rs = dtc.convert(Timestamp("2012-1-1"), None, None)
        assert rs == xp

        # also testing datetime64 dtype (GH8614)
        rs = dtc.convert("2012-01-01", None, None)
        assert rs == xp

        rs = dtc.convert("2012-01-01 00:00:00+0000", None, None)
        assert rs == xp

        rs = dtc.convert(
            np.array(["2012-01-01 00:00:00+0000", "2012-01-02 00:00:00+0000"]),
            None,
            None,
        )
        assert rs[0] == xp

        # we have a tz-aware date (constructed to that when we turn to utc it
        # is the same as our sample)
        ts = Timestamp("2012-01-01").tz_localize("UTC").tz_convert("US/Eastern")
        rs = dtc.convert(ts, None, None)
        assert rs == xp

        rs = dtc.convert(ts.to_pydatetime(), None, None)
        assert rs == xp

        rs = dtc.convert(Index([ts - Day(1), ts]), None, None)
        assert rs[1] == xp

        rs = dtc.convert(Index([ts - Day(1), ts]).to_pydatetime(), None, None)
        assert rs[1] == xp

    def test_conversion_float(self, dtc):
        # [FUNCTION DOC]
        # Checks conversion of timestamps in different time zones and a native datetime
        # to float date numbers that are almost equal within a relative tolerance of
        # about half a billionth.
        rtol = 0.5 * 10**-9

        rs = dtc.convert(Timestamp("2012-1-1 01:02:03", tz="UTC"), None, None)
        xp = dates.date2num(Timestamp("2012-1-1 01:02:03", tz="UTC"))
        tm.assert_almost_equal(rs, xp, rtol=rtol)

        rs = dtc.convert(
            Timestamp("2012-1-1 09:02:03", tz="Asia/Hong_Kong"), None, None
        )
        tm.assert_almost_equal(rs, xp, rtol=rtol)

        rs = dtc.convert(datetime(2012, 1, 1, 1, 2, 3), None, None)
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    @pytest.mark.parametrize(
        "values",
        [
            [date(1677, 1, 1), date(1677, 1, 2)],
            [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)],
        ],
    )
    def test_conversion_outofbounds_datetime(self, dtc, values):
        # [FUNCTION DOC]
        # Issue 2579. Parametrized over lists of dates and datetimes before the
        # nanosecond range (year 1677). Converting a list gives the same array as the
        # matplotlib date-to-number function; converting the first element alone gives
        # the scalar equivalent.
        # 2579
        rs = dtc.convert(values, None, None)
        xp = dates.date2num(values)
        tm.assert_numpy_array_equal(rs, xp)
        rs = dtc.convert(values[0], None, None)
        xp = dates.date2num(values[0])
        assert rs == xp

    @pytest.mark.parametrize(
        "time,format_expected",
        [
            (0, "00:00"),  # time2num(datetime.time.min)
            (86399.999999, "23:59:59.999999"),  # time2num(datetime.time.max)
            (90000, "01:00"),
            (3723, "01:02:03"),
            (39723.2, "11:02:03.200"),
        ],
    )
    def test_time_formatter(self, time, format_expected):
        # [FUNCTION DOC]
        # Issue 18478. Parametrized over (seconds, expected text). The time formatter,
        # called with a number of seconds since midnight, returns text without sub-second
        # digits unless needed (hours and minutes form, then seconds, then fractional).
        # issue 18478
        result = converter.TimeFormatter(None)(time)
        assert result == format_expected

    @pytest.mark.parametrize("freq", ("B", "ms", "s"))
    def test_dateindex_conversion(self, freq, dtc):
        # [FUNCTION DOC]
        # Parametrized over frequency (business day, milliseconds, seconds). Converting
        # a date range gives numbers almost equal (tight tolerance) to the matplotlib
        # conversion of its matplotlib representation.
        rtol = 10**-9
        dateindex = date_range("2020-01-01", periods=10, freq=freq)
        rs = dtc.convert(dateindex, None, None)
        xp = dates.date2num(dateindex._mpl_repr())
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    @pytest.mark.parametrize("offset", [Second(), Milli(), Micro(50)])
    def test_resolution(self, offset, dtc):
        # [FUNCTION DOC]
        # Parametrized over offsets (one second, one millisecond, 50 microseconds).
        # Converts a timestamp and the same plus the offset, and requires the second
        # number to be strictly larger, i.e. the float representation resolves the
        # offset.
        # Matplotlib's time representation using floats cannot distinguish
        # intervals smaller than ~10 microsecond in the common range of years.
        ts1 = Timestamp("2012-1-1")
        ts2 = ts1 + offset
        val1 = dtc.convert(ts1, None, None)
        val2 = dtc.convert(ts2, None, None)
        if not val1 < val2:
            raise AssertionError(f"{val1} is not less than {val2}.")

    def test_convert_nested(self, dtc):
        # [FUNCTION DOC]
        # Converting a nested list of timestamps gives the same result as converting
        # each inner list separately.
        inner = [Timestamp("2017-01-01"), Timestamp("2017-01-02")]
        data = [inner, inner]
        result = dtc.convert(data, None, None)
        expected = [dtc.convert(x, None, None) for x in data]
        assert (np.array(result) == expected).all()


class TestPeriodConverter:
    # [CLASS DOC]
    # Tests of the pandas period converter used by matplotlib.
    @pytest.fixture
    def pc(self):
        # [FUNCTION DOC]
        # Fixture returning a new period converter object.
        return converter.PeriodConverter()

    @pytest.fixture
    def axis(self):
        # [FUNCTION DOC]
        # Fixture returning a simple object that has a daily frequency attribute,
        # standing in for a matplotlib axis.
        class Axis:
            # [CLASS DOC]
            # HELPER: empty placeholder class used to build a fake axis for the period
            # converter; attributes are attached after construction.
            pass

        axis = Axis()
        axis.freq = "D"
        return axis

    def test_convert_accepts_unicode(self, pc, axis):
        # [FUNCTION DOC]
        # Converting the same string twice gives equal results.
        # KNOWN ISSUE: both results come from the identical call, so the assertion is
        # trivially true and does not verify unicode handling.
        r1 = pc.convert("2012-1-1", None, axis)
        r2 = pc.convert("2012-1-1", None, axis)
        assert r1 == r2

    def test_conversion(self, pc, axis):
        # [FUNCTION DOC]
        # Converts several kinds of input (date string lists and strings, dates,
        # timestamps, a tz-suffixed string, a datetime64 array) with a daily axis and
        # expects each to equal the ordinal of the daily period for 1 January 2012.
        rs = pc.convert(["2012-1-1"], None, axis)[0]
        xp = Period("2012-1-1").ordinal
        assert rs == xp

        rs = pc.convert("2012-1-1", None, axis)
        assert rs == xp

        rs = pc.convert([date(2012, 1, 1)], None, axis)[0]
        assert rs == xp

        rs = pc.convert(date(2012, 1, 1), None, axis)
        assert rs == xp

        rs = pc.convert([Timestamp("2012-1-1")], None, axis)[0]
        assert rs == xp

        rs = pc.convert(Timestamp("2012-1-1"), None, axis)
        assert rs == xp

        rs = pc.convert("2012-01-01", None, axis)
        assert rs == xp

        rs = pc.convert("2012-01-01 00:00:00+0000", None, axis)
        assert rs == xp

        rs = pc.convert(
            np.array(
                ["2012-01-01 00:00:00", "2012-01-02 00:00:00"],
                dtype="datetime64[ns]",
            ),
            None,
            axis,
        )
        assert rs[0] == xp

    def test_integer_passthrough(self, pc, axis):
        # [FUNCTION DOC]
        # GH9012. Integer inputs are passed through unchanged by the period converter.
        # GH9012
        rs = pc.convert([0, 1], None, axis)
        xp = [0, 1]
        assert rs == xp

    def test_convert_nested(self, pc, axis):
        # [FUNCTION DOC]
        # Converting a nested list of date strings gives the same as converting each
        # inner list separately.
        data = ["2012-1-1", "2012-1-2"]
        r1 = pc.convert([data, data], None, axis)
        r2 = [pc.convert(data, None, axis) for _ in range(2)]
        assert r1 == r2


class TestTimeDeltaConverter:
    # [CLASS DOC]
    # Tests of the timedelta tick formatter.
    """Test timedelta converter"""

    @pytest.mark.parametrize(
        "x, decimal, format_expected",
        [
            (0.0, 0, "00:00:00"),
            (3972320000000, 1, "01:06:12.3"),
            (713233432000000, 2, "8 days 06:07:13.43"),
            (32423432000000, 4, "09:00:23.4320"),
        ],
    )
    def test_format_timedelta_ticks(self, x, decimal, format_expected):
        # [FUNCTION DOC]
        # Parametrized over (value in nanoseconds, decimals, expected text). The static
        # formatting function with a nanosecond exponent produces clock-style text with
        # the given number of decimals, and includes a day count when over a day.
        tdc = converter.TimeSeries_TimedeltaFormatter
        result = tdc.format_timedelta_ticks(x, pos=None, n_decimals=decimal, exp=9)
        assert result == format_expected

    @pytest.mark.parametrize("view_interval", [(1, 2), (2, 1)])
    def test_call_w_different_view_intervals(self, view_interval, monkeypatch):
        # [FUNCTION DOC]
        # Parametrized over view intervals in normal and reversed order (GH37454).
        # Replaces the axis of a timedelta formatter with a fake whose view interval is
        # the given tuple, then calls the formatter; it must not raise even when the
        # limits are reversed.
        # previously broke on reversed xlmits; see GH37454
        class mock_axis:
            # [CLASS DOC]
            # HELPER: fake axis class used to give the formatter a view interval; the
            # interval comes from the enclosing test's parameter.
            def get_view_interval(self):
                # [FUNCTION DOC]
                # HELPER: returns the view interval tuple supplied by the enclosing test.
                return view_interval

        tdc = converter.TimeSeries_TimedeltaFormatter()
        monkeypatch.setattr(tdc, "axis", mock_axis())
        tdc(0.0, 0)


@pytest.mark.parametrize("year_span", [11.25, 30, 80, 150, 400, 800, 1500, 2500, 3500])
# The range is limited to 11.25 at the bottom by if statements in
# the _quarterly_finder() function
def test_quarterly_finder(year_span):
    # [FUNCTION DOC]
    # Parametrized over year spans from 11.25 to 3500. Computes a range of quarterly
    # period positions for the given span, runs the quarterly tick finder and
    # verifies that major ticks fall on the first quarter of years that are multiples
    # of the default major annual spacing, and minor ticks on first quarters of
    # years that are multiples of the minor spacing. The test skips when the span
    # is under 45 quarters.
    # KNOWN ISSUE: the skip can never trigger, since the smallest parametrized
    # span (11.25 years) gives 46 quarters.
    vmin = -1000
    vmax = vmin + year_span * 4
    span = vmax - vmin + 1
    if span < 45:
        pytest.skip("the quarterly finder is only invoked if the span is >= 45")
    nyears = span / 4
    (min_anndef, maj_anndef) = converter._get_default_annual_spacing(nyears)
    result = converter._quarterly_finder(vmin, vmax, to_offset("QE"))
    quarters = PeriodIndex(
        arrays.PeriodArray(np.array([x[0] for x in result]), dtype="period[Q]")
    )
    majors = np.array([x[1] for x in result])
    minors = np.array([x[2] for x in result])
    major_quarters = quarters[majors]
    minor_quarters = quarters[minors]
    check_major_years = major_quarters.year % maj_anndef == 0
    check_minor_years = minor_quarters.year % min_anndef == 0
    check_major_quarters = major_quarters.quarter == 1
    check_minor_quarters = minor_quarters.quarter == 1
    assert np.all(check_major_years)
    assert np.all(check_minor_years)
    assert np.all(check_major_quarters)
    assert np.all(check_minor_quarters)
