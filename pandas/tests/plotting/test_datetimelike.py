"""Test cases for time series specific (freq conversion, etc)"""

# [MODULE DOC]
# Tests for plotting of time series: datetime, period and timedelta indexes,
# inferred and irregular frequencies, mixed-frequency overlays and resampling
# on shared axes, secondary axes and legends, tick finders and tick label
# formats, x-axis limits, time-of-day indexes, NaT handling and pickling of
# figures. Public contents: the class TestTSPlot and a module-level private
# helper that runs a plotting callable on two subplots. Pandas matplotlib
# converter module is imported under the alias conv after the matplotlib
# import-or-skip, so the whole module is skipped when matplotlib is absent.
# No other import-time side effects.

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
import pickle

import numpy as np
import pytest

from pandas._libs.tslibs import (
    BaseOffset,
    to_offset,
)

from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    DataFrame,
    Index,
    NaT,
    Series,
    concat,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    bdate_range,
    date_range,
)
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props

from pandas.tseries.offsets import WeekOfMonth

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

import pandas.plotting._matplotlib.converter as conv


class TestTSPlot:
    # [CLASS DOC]
    # Test class for time series plotting behavior (series and frames with
    # datetime-like indexes). All tests use matplotlib figures created within the
    # test and rely on the plotting cleanup fixtures.
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ts_plot_with_tz(self, tz_aware_fixture):
        # [FUNCTION DOC]
        # Takes the timezone fixture. GH2877, GH17173, GH31205, GH31580. A two-point
        # hourly series with a timezone-aware index plots (user warnings ignored); the
        # first and last x data points show hour/minute 0:00 and 1:00 of local time.
        # GH2877, GH17173, GH31205, GH31580
        tz = tz_aware_fixture
        index = date_range("1/1/2011", periods=2, freq="h", tz=tz)
        ts = Series([188.5, 328.25], index=index)
        _check_plot_works(ts.plot)
        ax = ts.plot()
        xdata = next(iter(ax.get_lines())).get_xdata()
        # Check first and last points' labels are correct
        assert (xdata[0].hour, xdata[0].minute) == (0, 0)
        assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)

    def test_fontsize_set_correctly(self):
        # [FUNCTION DOC]
        # Issue 8765. Plotting with font size 2 gives x and y tick labels of size 2.
        # For issue #8765
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)), index=range(10)
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            assert label.get_fontsize() == 2

    def test_frame_inferred(self):
        # [FUNCTION DOC]
        # Frames with a monthly index whose frequency attribute was removed (inferred),
        # and with an index missing some entries, plot without error.
        # inferred freq
        idx = date_range("1/1/1987", freq="MS", periods=10)
        idx = DatetimeIndex(idx.values, freq=None)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df.plot)

        # axes freq
        idx = idx[0:4].union(idx[6:])
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df2.plot)

    def test_frame_inferred_n_gt_1(self):
        # [FUNCTION DOC]
        # A frame with a 15-minute index without frequency attribute plots.
        # N > 1
        idx = date_range("2008-1-1 00:15:00", freq="15min", periods=10)
        idx = DatetimeIndex(idx.values, freq=None)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx
        )
        _check_plot_works(df.plot)

    def test_is_error_nozeroindex(self):
        # [FUNCTION DOC]
        # GH11858. Plotting a frame with an integer index not starting at zero works
        # with the frame as x error values and as y error values.
        # GH11858
        i = np.array([1, 2, 3])
        a = DataFrame(i, index=i)
        _check_plot_works(a.plot, xerr=a)
        _check_plot_works(a.plot, yerr=a)

    def test_nonnumeric_exclude(self):
        # [FUNCTION DOC]
        # A frame with a non-numeric and a numeric column on a yearly index plots only
        # the numeric one (one line).
        idx = date_range("1/1/1987", freq="YE", periods=3)
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)

        fig, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)  # it works
        assert len(ax.get_lines()) == 1  # B was plotted

    def test_nonnumeric_exclude_error(self):
        # [FUNCTION DOC]
        # Plotting the non-numeric series alone raises a type error saying there is no
        # numeric data to plot.
        idx = date_range("1/1/1987", freq="YE", periods=3)
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]}, idx)
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            df["A"].plot()

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_tsplot_period(self, freq):
        # [FUNCTION DOC]
        # Parametrized over period frequencies. A series with a period index plots on a
        # given axes and passes the plot checks.
        idx = period_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_tsplot_datetime(self, freq):
        # [FUNCTION DOC]
        # Parametrized over date range frequencies (including a business-hour like
        # combined offset). A series with a datetime index plots on given axes.
        idx = date_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(ser.plot, ax=ax)

    def test_tsplot(self):
        # [FUNCTION DOC]
        # Plotting a datetime-indexed series with the black style gives a line with the
        # black RGBA color.
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        _, ax = mpl.pyplot.subplots()
        ts.plot(style="k", ax=ax)
        color = (0.0, 0.0, 0.0, 1)
        assert color == ax.get_lines()[0].get_color()

    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=10)])
    def test_both_style_and_color(self, index):
        # [FUNCTION DOC]
        # Parametrized over no index or date index. A style with a color symbol together
        # with the color keyword raises a value error describing the conflict.
        ts = Series(np.arange(10, dtype=np.float64), index=index)
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' "
            "keyword argument. Please use one or the other or pass 'style' "
            "without a color symbol"
        )
        with pytest.raises(ValueError, match=msg):
            ts.plot(style="b-", color="#000099")

    @pytest.mark.parametrize("freq", ["ms", "us"])
    def test_high_freq(self, freq):
        # [FUNCTION DOC]
        # Parametrized over millisecond and microsecond frequencies. Series plots on
        # given axes.
        _, ax = mpl.pyplot.subplots()
        rng = date_range("1/1/2012", periods=10, freq=freq)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _check_plot_works(ser.plot, ax=ax)

    def test_get_datevalue(self):
        # [FUNCTION DOC]
        # Tests the private converter helper that converts axis limit values: none gives
        # none; an integer passes through; a period converts to the ordinal in the
        # target frequency (end of year for the monthly target); a date string converts
        # to the daily ordinal.
        assert conv._get_datevalue(None, "D") is None
        assert conv._get_datevalue(1987, "Y") == 1987
        assert (
            conv._get_datevalue(Period(1987, "Y"), "M")
            == Period("1987-12", "M").ordinal
        )
        assert conv._get_datevalue("1/1/1987", "D") == Period("1987-1-1", "D").ordinal

    @pytest.mark.parametrize(
        "freq, expected_string",
        [["YE-DEC", "t = 2014  y = 1.000000"], ["D", "t = 2014-01-01  y = 1.000000"]],
    )
    def test_ts_plot_format_coord(self, freq, expected_string):
        # [FUNCTION DOC]
        # Parametrized over (frequency, expected string). The axes' coordinate
        # formatter, given the first point's ordinal, returns a string with the date
        # (year only for annual data) and the y value.
        ser = Series(1, index=date_range("2014-01-01", periods=3, freq=freq))
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        first_line = ax.get_lines()[0]
        first_x = first_line.get_xdata()[0].ordinal
        first_y = first_line.get_ydata()[0]
        assert expected_string == ax.format_coord(first_x, first_y)

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "M", "Q", "Y"])
    def test_line_plot_period_series(self, freq):
        # [FUNCTION DOC]
        # Parametrized over period frequencies. Series plot with the frequency check
        # against the index frequency.
        idx = period_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq)

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    def test_line_plot_period_mlt_series(self, frqncy):
        # [FUNCTION DOC]
        # Issue 14763. Parametrized over multiples of frequencies. Plots a period
        # indexed series and checks the axes frequency matches the rule code.
        # test period index line plot for series with multiples (`mlt`) of the
        # frequency (`frqncy`) rule code. tests resolution of issue #14763
        idx = period_range("12/31/1999", freq=frqncy, periods=10)
        s = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(s.plot, s.index.freq.rule_code)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_datetime_series(self, freq):
        # [FUNCTION DOC]
        # Parametrized over date range frequencies. Plots and checks the axes frequency
        # against the index frequency's rule code.
        idx = date_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        _check_plot_works(ser.plot, ser.index.freq.rule_code)

    @pytest.mark.parametrize("freq", ["s", "min", "h", "D", "W", "ME", "QE", "YE"])
    def test_line_plot_period_frame(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequencies. Plots a frame with a datetime index and checks
        # the axes frequency against the index frequency.
        # KNOWN ISSUE: the name says period frame but the index is a datetime range, not
        # a period range.
        idx = date_range("12/31/1999", freq=freq, periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        _check_plot_works(df.plot, df.index.freq)

    @pytest.mark.parametrize(
        "frqncy", ["1s", "3s", "5min", "7h", "4D", "8W", "11M", "3Y"]
    )
    def test_line_plot_period_mlt_frame(self, frqncy):
        # [FUNCTION DOC]
        # Issue 14763. Frame version of the multiples-of-frequency test with a period
        # index and rule code check.
        # test period index line plot for DataFrames with multiples (`mlt`)
        # of the frequency (`frqncy`) rule code. tests resolution of issue
        # #14763
        idx = period_range("12/31/1999", freq=frqncy, periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        freq = df.index.freq.rule_code
        _check_plot_works(df.plot, freq)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_datetime_frame(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequencies; ignores a deprecation warning about the
        # business period dtype. Plots a frame with a datetime index and checks the axes
        # frequency against the matching period frequency.
        idx = date_range("12/31/1999", freq=freq, periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )
        freq = PeriodDtype(df.index.freq)._freqstr
        freq = df.index.to_period(freq).freq
        _check_plot_works(df.plot, freq)

    @pytest.mark.parametrize(
        "freq", ["s", "min", "h", "D", "W", "ME", "QE-DEC", "YE", "1B30Min"]
    )
    def test_line_plot_inferred_freq(self, freq):
        # [FUNCTION DOC]
        # Parametrized over frequencies. Plots a series whose index has no frequency
        # attribute and checks the inferred frequency; then plots an irregular subset of
        # the series.
        idx = date_range("12/31/1999", freq=freq, periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser = Series(ser.values, Index(np.asarray(ser.index)))
        _check_plot_works(ser.plot, ser.index.inferred_freq)

        ser = ser.iloc[[0, 3, 5, 6]]
        _check_plot_works(ser.plot)

    def test_fake_inferred_business(self):
        # [FUNCTION DOC]
        # A daily series with a gap (inferred business-like) plots on the given axes
        # without the axes getting a frequency attribute.
        _, ax = mpl.pyplot.subplots()
        rng = date_range("2001-1-1", "2001-1-10")
        ts = Series(range(len(rng)), index=rng)
        ts = concat([ts[:3], ts[5:]])
        ts.plot(ax=ax)
        assert not hasattr(ax, "freq")

    def test_plot_offset_freq(self):
        # [FUNCTION DOC]
        # A series on a daily index plots.
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        _check_plot_works(ser.plot)

    def test_plot_offset_freq_business(self):
        # [FUNCTION DOC]
        # A series on a business quarter-start index plots.
        dr = date_range("2023-01-01", freq="BQS", periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        _check_plot_works(ser.plot)

    def test_plot_multiple_inferred_freq(self):
        # [FUNCTION DOC]
        # A series on an index of three native datetime values five days apart plots.
        dr = Index([datetime(2000, 1, 1), datetime(2000, 1, 6), datetime(2000, 1, 11)])
        ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
        _check_plot_works(ser.plot)

    def test_irreg_hf(self):
        # [FUNCTION DOC]
        # A high-frequency (seconds) irregular selection of rows plots with x
        # differences matching the true time gaps in days, within a tolerance.
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )

        irreg = df.iloc[[0, 1, 3, 4]]
        _, ax = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()

        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-8).all()

    def test_irreg_hf_object(self):
        # [FUNCTION DOC]
        # A seconds index converted to object dtype plots with evenly spaced x values.
        idx = date_range("2012-6-22 21:59:51", freq="s", periods=10)
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx
        )
        _, ax = mpl.pyplot.subplots()
        df2.index = df2.index.astype(object)
        df2.plot(ax=ax)
        diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
        sec = 1.0 / 24 / 60 / 60
        assert (np.fabs(diffs[1:] - sec) < 1e-8).all()

    def test_irregular_datetime64_repr_bug(self):
        # [FUNCTION DOC]
        # An irregular datetime series plots and the line's x data equal the index.
        ser = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ser = ser.iloc[[0, 1, 2, 7]]

        _, ax = mpl.pyplot.subplots()

        ret = ser.plot(ax=ax)
        assert ret is not None

        for rs, xp in zip(ax.get_lines()[0].get_xdata(), ser.index, strict=True):
            assert rs == xp

    def test_business_freq(self):
        # [FUNCTION DOC]
        # Creates a series with a business-frequency period index (emitting a
        # deprecation warning about the business period dtype). Plots it; the first x
        # value equals the first period's ordinal and the x data convert back to a
        # business frequency period index (with the same warning).
        bts = Series(range(5), period_range("2020-01-01", periods=5))
        msg = r"PeriodDtype\[B\] is deprecated"
        dt = bts.index[0].to_timestamp()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            bts.index = period_range(start=dt, periods=len(bts), freq="B")
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == bts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert PeriodIndex(data=idx).freqstr == "B"

    def test_business_freq_convert(self):
        # [FUNCTION DOC]
        # A series resampled to business month end plots as monthly periods: the first x
        # value equals the monthly ordinal and the x data have monthly frequency.
        bts = Series(
            np.arange(50, dtype=np.float64),
            index=date_range("2020-01-01", periods=50, freq="B"),
        ).asfreq("BME")
        ts = bts.to_period("M")
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        assert ax.get_lines()[0].get_xydata()[0, 0] == ts.index[0].ordinal
        idx = ax.get_lines()[0].get_xdata()
        assert PeriodIndex(data=idx).freqstr == "M"

    def test_freq_with_no_period_alias(self):
        # [FUNCTION DOC]
        # GH34487. A series with a week-of-month frequency (no period alias) plots; the
        # x data cannot be converted to a period index and raise a value error saying
        # the frequency cannot be inferred.
        # GH34487
        freq = WeekOfMonth()
        bts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        ).asfreq(freq)
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)

        idx = ax.get_lines()[0].get_xdata()
        msg = "freq not specified and cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(data=idx)

    def test_nonzero_base(self):
        # [FUNCTION DOC]
        # GH2571. An hourly index offset by 30 minutes keeps non-normalized x data
        # after plotting.
        # GH2571
        idx = date_range("2012-12-20", periods=24, freq="h") + timedelta(minutes=30)
        df = DataFrame(np.arange(24), index=idx)
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)
        rs = ax.get_lines()[0].get_xdata()
        assert not Index(rs).is_normalized

    def test_dataframe(self):
        # [FUNCTION DOC]
        # A one-column frame plots with x data equal to the index as periods.
        bts = DataFrame(
            {
                "a": Series(
                    np.arange(10, dtype=np.float64),
                    index=date_range("2020-01-01", periods=10),
                )
            }
        )
        _, ax = mpl.pyplot.subplots()
        bts.plot(ax=ax)
        idx = ax.get_lines()[0].get_xdata()
        tm.assert_index_equal(bts.index.to_period(), PeriodIndex(idx))

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize(
        "obj",
        [
            Series(
                np.arange(10, dtype=np.float64),
                index=date_range("2020-01-01", periods=10),
            ),
            DataFrame(
                {
                    "a": Series(
                        np.arange(10, dtype=np.float64),
                        index=date_range("2020-01-01", periods=10),
                    ),
                    "b": Series(
                        np.arange(10, dtype=np.float64),
                        index=date_range("2020-01-01", periods=10),
                    )
                    + 1,
                }
            ),
        ],
    )
    def test_axis_limits(self, obj):
        # [FUNCTION DOC]
        # Parametrized over a series and a frame with daily dates; ignores a deprecation
        # warning. After plotting, x limits can be set by numbers (reflected exactly),
        # by date strings and by datetimes (converted to ordinals of the axes frequency).
        _, ax = mpl.pyplot.subplots()
        obj.plot(ax=ax)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 5, xlim[1] + 10)
        result = ax.get_xlim()
        assert result[0] == xlim[0] - 5
        assert result[1] == xlim[1] + 10

        # string
        expected = (Period("1/1/2000", ax.freq), Period("4/1/2000", ax.freq))
        ax.set_xlim("1/1/2000", "4/1/2000")
        result = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal

        # datetime
        expected = (Period("1/1/2000", ax.freq), Period("4/1/2000", ax.freq))
        ax.set_xlim(datetime(2000, 1, 1), datetime(2000, 4, 1))
        result = ax.get_xlim()
        assert int(result[0]) == expected[0].ordinal
        assert int(result[1]) == expected[1].ordinal

    def test_get_finder(self):
        # [FUNCTION DOC]
        # The tick finder lookup returns the daily finder for business and daily and
        # weekly offsets, the monthly finder for month-end, quarterly for quarter-end and
        # annual for year-end.
        assert conv.get_finder(to_offset("B")) == conv._daily_finder
        assert conv.get_finder(to_offset("D")) == conv._daily_finder
        assert conv.get_finder(to_offset("ME")) == conv._monthly_finder
        assert conv.get_finder(to_offset("QE")) == conv._quarterly_finder
        assert conv.get_finder(to_offset("YE")) == conv._annual_finder
        assert conv.get_finder(to_offset("W")) == conv._daily_finder

    def test_finder_daily(self):
        # [FUNCTION DOC]
        # For several business-day series lengths, the first major tick location equals
        # the ordinal of the first business day, both for the default limits and after
        # shifting the lower limit by 0.9; expects a deprecation warning when building the
        # expected period.
        day_lst = [10, 40, 252, 400, 950, 2750, 10000]

        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            xpl1 = xpl2 = [Period("1999-1-1", freq="B").ordinal] * len(day_lst)
        rs1 = []
        rs2 = []
        for n in day_lst:
            rng = bdate_range("1999-1-1", periods=n)
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])

            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_quarterly(self):
        # [FUNCTION DOC]
        # For two series lengths (years), the first major tick is the first quarter of
        # 1988 with default limits and after shifting the lower limit by 0.9.
        yrs = [3.5, 11]

        xpl1 = xpl2 = [Period("1988Q1").ordinal] * len(yrs)
        rs1 = []
        rs2 = []
        for n in yrs:
            rng = period_range("1987Q2", periods=int(n * 4), freq="Q")
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])

            (vmin, vmax) = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly(self):
        # [FUNCTION DOC]
        # For several series lengths, the first major tick is January 1988 with default
        # limits and after shifting the lower x limit by 0.9.
        yrs = [1.15, 2.5, 4, 11]

        xpl1 = xpl2 = [Period("Jan 1988").ordinal] * len(yrs)
        rs1 = []
        rs2 = []
        for n in yrs:
            rng = period_range("1987Q2", periods=int(n * 12), freq="M")
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs1.append(xaxis.get_majorticklocs()[0])

            vmin, vmax = ax.get_xlim()
            ax.set_xlim(vmin + 0.9, vmax)
            rs2.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs1 == xpl1
        assert rs2 == xpl2

    def test_finder_monthly_long(self):
        # [FUNCTION DOC]
        # A long monthly series (24 years) has the first major tick at the first month
        # of 1989.
        rng = period_range("1988Q1", periods=24 * 12, freq="M")
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period("1989Q1", "M").ordinal
        assert rs == xp

    def test_finder_annual(self):
        # [FUNCTION DOC]
        # For several series lengths of years, the first major tick equals the
        # expected year (1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170) as annual ordinal.
        xp = [1987, 1988, 1990, 1990, 1995, 2020, 2070, 2170]
        xp = [Period(x, freq="Y").ordinal for x in xp]
        rs = []
        for nyears in [5, 10, 19, 49, 99, 199, 599, 1001]:
            rng = period_range("1987", periods=nyears, freq="Y")
            ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
            _, ax = mpl.pyplot.subplots()
            ser.plot(ax=ax)
            xaxis = ax.get_xaxis()
            rs.append(xaxis.get_majorticklocs()[0])
            mpl.pyplot.close(ax.get_figure())

        assert rs == xp

    @pytest.mark.slow
    def test_finder_minutely(self):
        # [FUNCTION DOC]
        # Marked slow. A day of minute data has its first major tick at the first minute.
        nminutes = 1 * 24 * 60
        rng = date_range("1/1/1999", freq="Min", periods=nminutes)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period("1/1/1999", freq="Min").ordinal

        assert rs == xp

    def test_finder_hourly(self):
        # [FUNCTION DOC]
        # Twenty-three hourly points: the first major tick is the first hour.
        nhours = 23
        rng = date_range("1/1/1999", freq="h", periods=nhours)
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        xaxis = ax.get_xaxis()
        rs = xaxis.get_majorticklocs()[0]
        xp = Period("1/1/1999", freq="h").ordinal

        assert rs == xp

    def test_gaps(self):
        # [FUNCTION DOC]
        # A series with a missing stretch plots to one line whose data have masked
        # values at the positions of the gap.
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ts.iloc[5:7] = np.nan
        _, ax = mpl.pyplot.subplots()
        ts.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()

        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[5:7, 1].all()

    def test_gaps_irregular(self):
        # [FUNCTION DOC]
        # Same as the gap test for an irregular series: gap positions are masked.
        # irregular
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        ts = ts.iloc[[0, 1, 2, 5, 7, 9, 12, 15, 20]]
        ts.iloc[2:5] = np.nan
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()

        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[2:5, 1].all()

    def test_gaps_non_ts(self):
        # [FUNCTION DOC]
        # Same as the gap test for a non-time-series (integer index) series.
        # non-ts
        idx = [0, 1, 2, 5, 7, 9, 12, 15, 20]
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        ser.iloc[2:5] = np.nan
        _, ax = mpl.pyplot.subplots()
        ser.plot(ax=ax)
        lines = ax.get_lines()
        assert len(lines) == 1
        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[2:5, 1].all()

    def test_gap_upsample(self):
        # [FUNCTION DOC]
        # A low frequency series with a gap plotted, then a higher frequency series on a
        # secondary axis: one line on each axes and the low series still has masked gap
        # values.
        low = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        low.iloc[5:7] = np.nan
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)

        idxh = date_range(low.index[0], low.index[-1], freq="12h")
        s = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        s.plot(secondary_y=True)
        lines = ax.get_lines()
        assert len(lines) == 1
        assert len(ax.right_ax.get_lines()) == 1

        line = lines[0]
        data = line.get_xydata()
        data = np.ma.MaskedArray(data, mask=isna(data), fill_value=np.nan)

        assert isinstance(data, np.ma.core.MaskedArray)
        mask = data.mask
        assert mask[5:7, 1].all()

    def test_secondary_y(self):
        # [FUNCTION DOC]
        # A plain series plotted with secondary y returns an axes marked as having a left
        # twin, not a right one; the line data equal the series; y ticks are on the right;
        # the original axes y axis is hidden.
        ser = Series(np.random.default_rng(2).standard_normal(10))
        fig, _ = mpl.pyplot.subplots()
        ax = ser.plot(secondary_y=True)
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        axes = fig.get_axes()
        line = ax.get_lines()[0]
        xp = Series(line.get_ydata(), line.get_xdata())
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == "right"
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_yaxis(self):
        # [FUNCTION DOC]
        # Plotting a series normally leaves the y ticks on the left.
        # KNOWN ISSUE: the first statement constructs a series and discards it (dead
        # code).
        Series(np.random.default_rng(2).standard_normal(10))
        ser2 = Series(np.random.default_rng(2).standard_normal(10))
        _, ax2 = mpl.pyplot.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == "left"

    def test_secondary_both(self):
        # [FUNCTION DOC]
        # A normal and a secondary-y series on the same figure: the first axes keeps a
        # visible y axis and gets a right twin; the second returned axes has the left
        # twin reference.
        ser = Series(np.random.default_rng(2).standard_normal(10))
        ser2 = Series(np.random.default_rng(2).standard_normal(10))
        ax = ser2.plot()
        ax2 = ser.plot(secondary_y=True)
        assert ax.get_yaxis().get_visible()
        assert not hasattr(ax, "left_ax")
        assert hasattr(ax, "right_ax")
        assert hasattr(ax2, "left_ax")
        assert not hasattr(ax2, "right_ax")

    def test_secondary_y_ts(self):
        # [FUNCTION DOC]
        # As the plain secondary-y test but with a nanosecond date index; line data are
        # period data converted to timestamps at nanosecond unit and equal the series.
        idx = date_range("1/1/2000", periods=10, unit="ns")
        ser = Series(np.random.default_rng(2).standard_normal(10), idx)
        fig, _ = mpl.pyplot.subplots()
        ax = ser.plot(secondary_y=True)
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        axes = fig.get_axes()
        line = ax.get_lines()[0]
        xp = Series(line.get_ydata(), line.get_xdata()).to_timestamp()
        xp.index = xp.index.as_unit("ns")
        tm.assert_series_equal(ser, xp)
        assert ax.get_yaxis().get_ticks_position() == "right"
        assert not axes[0].get_yaxis().get_visible()

    def test_secondary_y_ts_yaxis(self):
        # [FUNCTION DOC]
        # Plotting a time series normally leaves the y ticks on the left.
        idx = date_range("1/1/2000", periods=10)
        ser2 = Series(np.random.default_rng(2).standard_normal(10), idx)
        _, ax2 = mpl.pyplot.subplots()
        ser2.plot(ax=ax2)
        assert ax2.get_yaxis().get_ticks_position() == "left"

    def test_secondary_y_ts_visible(self):
        # [FUNCTION DOC]
        # Plotting a time series normally leaves the y axis visible.
        idx = date_range("1/1/2000", periods=10)
        ser2 = Series(np.random.default_rng(2).standard_normal(10), idx)
        ax = ser2.plot()
        assert ax.get_yaxis().get_visible()

    def test_secondary_kde(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot with secondary y: the axes has a left twin and the
        # second axes has the y ticks on the right.
        pytest.importorskip("scipy")
        ser = Series(np.random.default_rng(2).standard_normal(10))
        fig, ax = mpl.pyplot.subplots()
        ax = ser.plot(secondary_y=True, kind="density", ax=ax)
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        axes = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == "right"

    def test_secondary_bar(self):
        # [FUNCTION DOC]
        # Bar plot with secondary y: the second axes has y ticks on the right.
        ser = Series(np.random.default_rng(2).standard_normal(10))
        fig, ax = mpl.pyplot.subplots()
        ser.plot(secondary_y=True, kind="bar", ax=ax)
        axes = fig.get_axes()
        assert axes[1].get_yaxis().get_ticks_position() == "right"

    def test_secondary_frame(self):
        # [FUNCTION DOC]
        # Frame with secondary y columns a and c and subplots: tick positions are right,
        # left, right.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["a", "b", "c"]
        )
        axes = df.plot(secondary_y=["a", "c"], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == "right"
        assert axes[1].get_yaxis().get_ticks_position() == "left"
        assert axes[2].get_yaxis().get_ticks_position() == "right"

    def test_secondary_bar_frame(self):
        # [FUNCTION DOC]
        # As above for bar plots of the frame.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["a", "b", "c"]
        )
        axes = df.plot(kind="bar", secondary_y=["a", "c"], subplots=True)
        assert axes[0].get_yaxis().get_ticks_position() == "right"
        assert axes[1].get_yaxis().get_ticks_position() == "left"
        assert axes[2].get_yaxis().get_ticks_position() == "right"

    def test_mixed_freq_regular_first(self):
        # [FUNCTION DOC]
        # A regular business-day series then an irregular subset on the same axes: both
        # lines' x data convert to business period indexes (with the deprecation warning),
        # and the axes limits cover the full range.
        # TODO
        s1 = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, freq="B"),
        )
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]

        # it works!
        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)

        ax2 = s2.plot(style="g", ax=ax)
        lines = ax2.get_lines()
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())

            tm.assert_index_equal(idx1, s1.index.to_period("B"))
            tm.assert_index_equal(idx2, s2.index.to_period("B"))

            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first(self):
        # [FUNCTION DOC]
        # The irregular subset plotted first then the regular series: the axes has no
        # frequency attribute and the x data are the index values as objects.
        s1 = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15]]
        _, ax = mpl.pyplot.subplots()
        s2.plot(style="g", ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, "freq")
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_regular_first_df(self):
        # [FUNCTION DOC]
        # GH 9852. Frame version of the regular-first mixed frequency test.
        # GH 9852
        s1 = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20, freq="B"),
        ).to_frame()
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        ax2 = s2.plot(style="g", ax=ax)
        lines = ax2.get_lines()
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(lines[0].get_xdata())
            idx2 = PeriodIndex(lines[1].get_xdata())
            assert idx1.equals(s1.index.to_period("B"))
            assert idx2.equals(s2.index.to_period("B"))
            left, right = ax2.get_xlim()
            pidx = s1.index.to_period()
        assert left <= pidx[0].ordinal
        assert right >= pidx[-1].ordinal

    def test_mixed_freq_irregular_first_df(self):
        # [FUNCTION DOC]
        # GH 9852. Frame version of the irregular-first mixed frequency test.
        # GH 9852
        s1 = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        ).to_frame()
        s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
        _, ax = mpl.pyplot.subplots()
        s2.plot(style="g", ax=ax)
        s1.plot(ax=ax)
        assert not hasattr(ax, "freq")
        lines = ax.get_lines()
        x1 = lines[0].get_xdata()
        tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
        x2 = lines[1].get_xdata()
        tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)

    def test_mixed_freq_hf_first(self):
        # [FUNCTION DOC]
        # A daily series then a monthly series on one axes: all lines have daily period
        # frequency.
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "D"

    def test_mixed_freq_alignment(self):
        # [FUNCTION DOC]
        # An hourly series and its minute-interpolated version drawn on one axes begin at
        # the same x value.
        ts_ind = date_range("2012-01-01 13:00", "2012-01-02", freq="h")
        ts_data = np.random.default_rng(2).standard_normal(12)

        ts = Series(ts_data, index=ts_ind)
        ts2 = ts.asfreq("min").interpolate()

        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax)
        ts2.plot(style="r", ax=ax)

        assert ax.lines[0].get_xdata()[0] == ax.lines[1].get_xdata()[0]

    def test_mixed_freq_lf_first(self):
        # [FUNCTION DOC]
        # Low frequency then high frequency series with legends: all lines convert to
        # daily frequency and the legend has two entries; closes the figure at the end.
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(legend=True, ax=ax)
        high.plot(legend=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "D"
        leg = ax.get_legend()
        assert len(leg.texts) == 2
        mpl.pyplot.close(ax.get_figure())

    def test_mixed_freq_lf_first_hourly(self):
        # [FUNCTION DOC]
        # Hourly then minute series: all lines convert to minute frequency.
        idxh = date_range("1/1/1999", periods=240, freq="min")
        idxl = date_range("1/1/1999", periods=4, freq="h")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "min"

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_mixed_freq_irreg_period(self):
        # [FUNCTION DOC]
        # Ignoring the business period deprecation, an irregular datetime series then a
        # business period series are plotted on one axes without error (the period
        # creation emits the expected deprecation warning).
        ts = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        irreg = ts.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 29]]
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rng = period_range("1/3/2000", periods=30, freq="B")
        ps = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        irreg.plot(ax=ax)
        ps.plot(ax=ax)

    def test_mixed_freq_shared_ax(self):
        # [FUNCTION DOC]
        # GH13341. Two series with different gaps in two subplots with shared x: both axes
        # have monthly frequency and the same first x value.
        # GH13341, using sharex=True
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)

        _, (ax1, ax2) = mpl.pyplot.subplots(nrows=2, sharex=True)
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)

        assert ax1.freq == "M"
        assert ax2.freq == "M"
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_mixed_freq_shared_ax_twin_x(self):
        # [FUNCTION DOC]
        # GH13341. Same series on an axes and its twin x axes: same first x value.
        # GH13341, using sharex=True
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)
        # using twinx
        _, ax1 = mpl.pyplot.subplots()
        ax2 = ax1.twinx()
        s1.plot(ax=ax1)
        s2.plot(ax=ax2)

        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    @pytest.mark.xfail(reason="TODO (GH14330, GH14322)")
    def test_mixed_freq_shared_ax_twin_x_irregular_first(self):
        # [FUNCTION DOC]
        # Expected to fail (GH14330, GH14322). Irregular series on the main axes and
        # regular on the twin axes should start at the same x value.
        # GH13341, using sharex=True
        idx1 = date_range("2015-01-01", periods=3, freq="ME")
        idx2 = idx1[:1].union(idx1[2:])
        s1 = Series(range(len(idx1)), idx1)
        s2 = Series(range(len(idx2)), idx2)
        _, ax1 = mpl.pyplot.subplots()
        ax2 = ax1.twinx()
        s2.plot(ax=ax1)
        s1.plot(ax=ax2)
        assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]

    def test_nat_handling(self):
        # [FUNCTION DOC]
        # A datetime index containing a missing timestamp plots; the line's x data stay
        # within the bounds of the index.
        _, ax = mpl.pyplot.subplots()

        dti = DatetimeIndex(["2015-01-01", NaT, "2015-01-03"])
        s = Series(range(len(dti)), dti)
        s.plot(ax=ax)
        xdata = ax.get_lines()[0].get_xdata()
        # plot x data is bounded by index values
        assert s.index.min() <= Series(xdata).min()
        assert Series(xdata).max() <= s.index.max()

    def test_to_weekly_resampling_disallow_how_kwd(self):
        # [FUNCTION DOC]
        # Plotting a low frequency series onto an axes with a weekly series with the
        # how keyword raises a value error saying how is not valid and to resample
        # manually.
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)

        msg = (
            "'how' is not a valid keyword for plotting functions. If plotting "
            "multiple objects on shared axes, resample manually first."
        )
        with pytest.raises(ValueError, match=msg):
            low.plot(ax=ax, how="foo")

    def test_to_weekly_resampling(self):
        # [FUNCTION DOC]
        # Weekly series then monthly series on one axes: all lines have the weekly
        # frequency.
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq

    def test_from_weekly_resampling(self):
        # [FUNCTION DOC]
        # Monthly then weekly series on one axes: all lines have weekly frequency; the
        # monthly line's raw x data equal the expected weekly ordinals of the month ends
        # and the weekly line's equal its period integers.
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)

        expected_h = idxh.to_period().asi8.astype(np.float64)
        expected_l = np.array(
            [1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562],
            dtype=np.float64,
        )
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            xdata = line.get_xdata(orig=False)
            if len(xdata) == 12:  # idxl lines
                tm.assert_numpy_array_equal(xdata, expected_l)
            else:
                tm.assert_numpy_array_equal(xdata, expected_h)

    @pytest.mark.parametrize("kind1, kind2", [("line", "area"), ("area", "line")])
    def test_from_resampling_area_line_mixed(self, kind1, kind2):
        # [FUNCTION DOC]
        # Parametrized over (line, area) orders. Stacked frames of low then high
        # frequency data on one axes: the first three lines are the low data cumulated in
        # stacking order with x at the weekly ordinals; the next three are the high data
        # cumulated.
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = DataFrame(
            np.random.default_rng(2).random((len(idxh), 3)),
            index=idxh,
            columns=[0, 1, 2],
        )
        low = DataFrame(
            np.random.default_rng(2).random((len(idxl), 3)),
            index=idxl,
            columns=[0, 1, 2],
        )

        _, ax = mpl.pyplot.subplots()
        low.plot(kind=kind1, stacked=True, ax=ax)
        high.plot(kind=kind2, stacked=True, ax=ax)

        # check low dataframe result
        expected_x = np.array(
            [
                1514,
                1519,
                1523,
                1527,
                1531,
                1536,
                1540,
                1544,
                1549,
                1553,
                1558,
                1562,
            ],
            dtype=np.float64,
        )
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[i]
            assert PeriodIndex(line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            # check stacked values are correct
            expected_y += low[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

        # check high dataframe result
        expected_x = idxh.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[3 + i]
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

    @pytest.mark.parametrize("kind1, kind2", [("line", "area"), ("area", "line")])
    def test_from_resampling_area_line_mixed_high_to_low(self, kind1, kind2):
        # [FUNCTION DOC]
        # Parametrized over order. Same as the previous test but plotting the high
        # frequency frame first, then the low frequency frame.
        idxh = date_range("1/1/1999", periods=52, freq="W")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = DataFrame(
            np.random.default_rng(2).random((len(idxh), 3)),
            index=idxh,
            columns=[0, 1, 2],
        )
        low = DataFrame(
            np.random.default_rng(2).random((len(idxl), 3)),
            index=idxl,
            columns=[0, 1, 2],
        )
        _, ax = mpl.pyplot.subplots()
        high.plot(kind=kind1, stacked=True, ax=ax)
        low.plot(kind=kind2, stacked=True, ax=ax)

        # check high dataframe result
        expected_x = idxh.to_period().asi8.astype(np.float64)
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            line = ax.lines[i]
            assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(line.get_xdata(orig=False), expected_x)
            expected_y += high[i].values
            tm.assert_numpy_array_equal(line.get_ydata(orig=False), expected_y)

        # check low dataframe result
        expected_x = np.array(
            [
                1514,
                1519,
                1523,
                1527,
                1531,
                1536,
                1540,
                1544,
                1549,
                1553,
                1558,
                1562,
            ],
            dtype=np.float64,
        )
        expected_y = np.zeros(len(expected_x), dtype=np.float64)
        for i in range(3):
            lines = ax.lines[3 + i]
            assert PeriodIndex(data=lines.get_xdata()).freq == idxh.freq
            tm.assert_numpy_array_equal(lines.get_xdata(orig=False), expected_x)
            expected_y += low[i].values
            tm.assert_numpy_array_equal(lines.get_ydata(orig=False), expected_y)

    def test_mixed_freq_second_millisecond(self):
        # [FUNCTION DOC]
        # GH 7772, GH 7760. Seconds series then 100 ms series: two lines, both with
        # millisecond frequency.
        # GH 7772, GH 7760
        idxh = date_range("2014-07-01 09:00", freq="s", periods=5)
        idxl = date_range("2014-07-01 09:00", freq="100ms", periods=50)
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # high to low
        _, ax = mpl.pyplot.subplots()
        high.plot(ax=ax)
        low.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "ms"

    def test_mixed_freq_second_millisecond_low_to_high(self):
        # [FUNCTION DOC]
        # Same as above in the reverse plotting order.
        # GH 7772, GH 7760
        idxh = date_range("2014-07-01 09:00", freq="s", periods=5)
        idxl = date_range("2014-07-01 09:00", freq="100ms", periods=50)
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        # low to high
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        high.plot(ax=ax)
        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            assert PeriodIndex(data=line.get_xdata()).freq == "ms"

    def test_irreg_dtypes(self):
        # [FUNCTION DOC]
        # A frame indexed by native date objects (object dtype) plots.
        # date
        idx = [date(2000, 1, 1), date(2000, 1, 5), date(2000, 1, 20)]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(idx), 3)),
            Index(idx, dtype=object),
        )
        _check_plot_works(df.plot)

    def test_irreg_dtypes_dt64(self):
        # [FUNCTION DOC]
        # A frame indexed by datetime values of object dtype, irregularly spaced, plots
        # on given axes.
        # np.datetime64
        idx = date_range("1/1/2000", periods=10)
        idx = idx[[0, 2, 5, 9]].astype(object)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(df.plot, ax=ax)

    def test_time(self):
        # [FUNCTION DOC]
        # A frame indexed by time-of-day values plots; the x tick labels are formatted
        # as hour:minute (with seconds when nonzero) matching each tick position.
        t = datetime(1, 1, 1, 3, 30, 0)
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)

        # verify tick labels
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels, strict=True):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs

    def test_time_change_xlim(self):
        # [FUNCTION DOC]
        # As the time test, then x limits are set with time strings and the tick labels
        # are checked again.
        t = datetime(1, 1, 1, 3, 30, 0)
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        _, ax = mpl.pyplot.subplots()
        df.plot(ax=ax)

        # verify tick labels
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels, strict=True):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs

        # change xlim
        ax.set_xlim("1:30", "5:00")

        # check tick labels again
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels, strict=True):
            m, s = divmod(int(_tick), 60)
            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if s != 0:
                    xp = time(h, m, s).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s).strftime("%H:%M")
                assert xp == rs

    def test_time_musec(self):
        # [FUNCTION DOC]
        # Time-of-day index with microsecond spacing: tick labels include fractional
        # seconds as needed (microseconds, milliseconds, seconds or minutes only).
        t = datetime(1, 1, 1, 3, 30, 0)
        deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
        ts = np.array([(t + timedelta(microseconds=int(x))).time() for x in deltas])
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(len(ts)),
                "b": np.random.default_rng(2).standard_normal(len(ts)),
            },
            index=ts,
        )
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)

        # verify tick labels
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        for _tick, _label in zip(ticks, labels, strict=True):
            m, s = divmod(int(_tick), 60)

            us = round((_tick - int(_tick)) * 1e6)

            h, m = divmod(m, 60)
            rs = _label.get_text()
            if len(rs) > 0:
                if (us % 1000) != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S.%f")
                elif (us // 1000) != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S.%f")[:-3]
                elif s != 0:
                    xp = time(h, m, s, us).strftime("%H:%M:%S")
                else:
                    xp = time(h, m, s, us).strftime("%H:%M")
                assert xp == rs

    def test_secondary_upsample(self):
        # [FUNCTION DOC]
        # Low frequency series, then a high frequency series on a secondary axis: all
        # lines get daily frequency, the returned axes has a left twin and no right twin.
        idxh = date_range("1/1/1999", periods=365, freq="D")
        idxl = date_range("1/1/1999", periods=12, freq="ME")
        high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
        low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
        _, ax = mpl.pyplot.subplots()
        low.plot(ax=ax)
        ax = high.plot(secondary_y=True, ax=ax)
        for line in ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == "D"
        assert hasattr(ax, "left_ax")
        assert not hasattr(ax, "right_ax")
        for line in ax.left_ax.get_lines():
            assert PeriodIndex(line.get_xdata()).freq == "D"

    def test_secondary_legend(self):
        # [FUNCTION DOC]
        # Frame of business-day data with two columns on the secondary axis: the legend
        # has four lines in order with the right-axis suffix for the secondary columns;
        # the right axes has no legend of its own; four distinct colors.
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)

        # ts
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df.plot(secondary_y=["A", "B"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == "A (right)"
        assert leg.get_texts()[1].get_text() == "B (right)"
        assert leg.get_texts()[2].get_text() == "C"
        assert leg.get_texts()[3].get_text() == "D"
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4

    def test_secondary_legend_right(self):
        # [FUNCTION DOC]
        # With mark_right disabled, legend labels have no suffix.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        df.plot(secondary_y=["A", "C"], mark_right=False, ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert leg.get_texts()[0].get_text() == "A"
        assert leg.get_texts()[1].get_text() == "B"
        assert leg.get_texts()[2].get_text() == "C"
        assert leg.get_texts()[3].get_text() == "D"

    def test_secondary_legend_bar(self):
        # [FUNCTION DOC]
        # Bar plot with one secondary column: first label has the suffix, second does not.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig, ax = mpl.pyplot.subplots()
        df.plot(kind="bar", secondary_y=["A"], ax=ax)
        leg = ax.get_legend()
        assert leg.get_texts()[0].get_text() == "A (right)"
        assert leg.get_texts()[1].get_text() == "B"

    def test_secondary_legend_bar_right(self):
        # [FUNCTION DOC]
        # Bar plot with mark_right disabled: labels have no suffix.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig, ax = mpl.pyplot.subplots()
        df.plot(kind="bar", secondary_y=["A"], mark_right=False, ax=ax)
        leg = ax.get_legend()
        assert leg.get_texts()[0].get_text() == "A"
        assert leg.get_texts()[1].get_text() == "B"

    def test_secondary_legend_multi_col(self):
        # [FUNCTION DOC]
        # Secondary columns C and D: four legend lines in the main axes legend, none in
        # the right axes, four distinct colors.
        # KNOWN ISSUE: the frame is built twice and the first one is discarded.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        ax = df.plot(secondary_y=["C", "D"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4

    def test_secondary_legend_nonts(self):
        # [FUNCTION DOC]
        # Non-time-series (string index) frame with two secondary columns: four legend
        # lines, no legend on the right axes, four distinct colors.
        # non-ts
        df = DataFrame(
            1.1 * np.arange(40).reshape((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(10)], dtype=object),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        ax = df.plot(secondary_y=["A", "B"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4

    def test_secondary_legend_nonts_multi_col(self):
        # [FUNCTION DOC]
        # As above with columns C and D on the secondary axis.
        # non-ts
        df = DataFrame(
            1.1 * np.arange(40).reshape((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(10)], dtype=object),
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_subplot(211)
        ax = df.plot(secondary_y=["C", "D"], ax=ax)
        leg = ax.get_legend()
        assert len(leg.get_lines()) == 4
        assert ax.right_ax.get_legend() is None
        colors = set()
        for line in leg.get_lines():
            colors.add(line.get_color())

        # TODO: color cycle problems
        assert len(colors) == 4

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_format_date_axis(self):
        # [FUNCTION DOC]
        # Expected to fail (matplotlib API changed in 3.6.0). Monthly data plotted; every
        # nonempty x tick label should be rotated 30 degrees.
        rng = date_range("1/1/2012", periods=12, freq="ME")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)
        xaxis = ax.get_xaxis()
        for line in xaxis.get_ticklabels():
            if len(line.get_text()) > 0:
                assert line.get_rotation() == 30

    def test_ax_plot(self):
        # [FUNCTION DOC]
        # Plotting dates directly with matplotlib (no pandas plotting) produces x data
        # equal to the datetime index.
        x = date_range(start="2012-01-02", periods=10, freq="D")
        y = list(range(len(x)))
        _, ax = mpl.pyplot.subplots()
        lines = ax.plot(x, y, label="Y")
        tm.assert_index_equal(DatetimeIndex(lines[0].get_xdata()), x)

    def test_mpl_nopandas(self):
        # [FUNCTION DOC]
        # Plotting ordinal day numbers via matplotlib with two data sets in one call
        # returns two lines whose x data equal the ordinal numbers as floats.
        dates = [date(2008, 12, 31), date(2009, 1, 31)]
        values1 = np.arange(10.0, 11.0, 0.5)
        values2 = np.arange(11.0, 12.0, 0.5)

        _, ax = mpl.pyplot.subplots()
        (
            line1,
            line2,
        ) = ax.plot(
            [x.toordinal() for x in dates],
            values1,
            "-",
            [x.toordinal() for x in dates],
            values2,
            "-",
            linewidth=4,
        )

        exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
        tm.assert_numpy_array_equal(line1.get_xydata()[:, 0], exp)
        tm.assert_numpy_array_equal(line2.get_xydata()[:, 0], exp)

    def test_irregular_ts_shared_ax_xlim(self):
        # [FUNCTION DOC]
        # GH 2960. The left then right sections of an irregular series plotted on one
        # axes: the axis limits cover the full index range.
        # GH 2960
        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]

        # plot the left section of the irregular series, then the right section
        _, ax = mpl.pyplot.subplots()
        ts_irregular[:5].plot(ax=ax)
        ts_irregular[5:].plot(ax=ax)

        # check that axis limits are correct
        left, right = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), "", ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), "", ax)

    def test_secondary_y_non_ts_xlim(self):
        # [FUNCTION DOC]
        # GH 3490. A non-time series plotted, then another on the secondary axis with a
        # higher index range: the lower limit does not increase and the upper limit
        # grows.
        # GH 3490 - non-timeseries with secondary y
        index_1 = [1, 2, 3, 4]
        index_2 = [5, 6, 7, 8]
        s1 = Series(1, index=index_1)
        s2 = Series(2, index=index_2)

        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()

        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_regular_ts_xlim(self):
        # [FUNCTION DOC]
        # GH 3490. As above for regular daily time series.
        # GH 3490 - regular-timeseries with secondary y
        index_1 = date_range(start="2000-01-01", periods=4, freq="D")
        index_2 = date_range(start="2000-01-05", periods=4, freq="D")
        s1 = Series(1, index=index_1)
        s2 = Series(2, index=index_2)

        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        s2.plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()

        assert left_before >= left_after
        assert right_before < right_after

    def test_secondary_y_mixed_freq_ts_xlim(self):
        # [FUNCTION DOC]
        # GH 3490. A minute series then its daily mean on the secondary axis: neither
        # limit changes (downsample).
        # GH 3490 - mixed frequency timeseries with secondary y
        rng = date_range("2000-01-01", periods=10, freq="min")
        ts = Series(1, index=rng)

        _, ax = mpl.pyplot.subplots()
        ts.plot(ax=ax)
        left_before, right_before = ax.get_xlim()
        ts.resample("D").mean().plot(secondary_y=True, ax=ax)
        left_after, right_after = ax.get_xlim()

        # a downsample should not have changed either limit
        assert left_before == left_after
        assert right_before == right_after

    def test_secondary_y_irregular_ts_xlim(self):
        # [FUNCTION DOC]
        # GH 3490. Irregular series plotted partly on primary and secondary axes then
        # primary again: the secondary limits are not overwritten and cover the full
        # range.
        # GH 3490 - irregular-timeseries with secondary y
        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]

        _, ax = mpl.pyplot.subplots()
        ts_irregular[:5].plot(ax=ax)
        # plot higher-x values on secondary axis
        ts_irregular[5:].plot(secondary_y=True, ax=ax)
        # ensure secondary limits aren't overwritten by plot on primary
        ts_irregular[:5].plot(ax=ax)

        left, right = ax.get_xlim()
        assert left <= conv.DatetimeConverter.convert(ts_irregular.index.min(), "", ax)
        assert right >= conv.DatetimeConverter.convert(ts_irregular.index.max(), "", ax)

    def test_plot_outofbounds_datetime(self):
        # [FUNCTION DOC]
        # Issue 2579. Plotting dates and datetimes outside the nanosecond range with
        # matplotlib does not raise.
        # 2579 - checking this does not raise
        values = [date(1677, 1, 1), date(1677, 1, 2)]
        _, ax = mpl.pyplot.subplots()
        ax.plot(values)

        values = [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]
        ax.plot(values)

    def test_format_timedelta_ticks_narrow(self):
        # [FUNCTION DOC]
        # Nanosecond-spaced timedelta index plots with small font; the tick labels are
        # the zero-padded nanosecond time strings.
        expected_labels = [f"00:00:00.0000000{i:0>2d}" for i in np.arange(10)]

        rng = timedelta_range("0", periods=10, freq="ns")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax = mpl.pyplot.subplots()
        df.plot(fontsize=2, ax=ax)
        mpl.pyplot.draw()
        labels = ax.get_xticklabels()

        result_labels = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_format_timedelta_ticks_wide(self, unit):
        # [FUNCTION DOC]
        # Takes the unit fixture. Daily timedelta index plots; the tick labels are the
        # expected strings with a day count and clock time at the chosen resolution.
        expected_labels = [
            "00:00:00",
            "1 days 03:46:40",
            "2 days 07:33:20",
            "3 days 11:20:00",
            "4 days 15:06:40",
            "5 days 18:53:20",
            "6 days 22:40:00",
            "8 days 02:26:40",
            "9 days 06:13:20",
        ]

        rng = timedelta_range("0", periods=10, freq="1 D", unit=unit)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(fontsize=2, ax=ax)
        mpl.pyplot.draw()
        labels = ax.get_xticklabels()

        result_labels = [x.get_text() for x in labels]
        assert len(result_labels) == len(expected_labels)
        assert result_labels == expected_labels

    def test_timedelta_plot(self):
        # [FUNCTION DOC]
        # Issue 8711. A series with a timedelta index plots.
        # test issue #8711
        s = Series(range(5), timedelta_range("1day", periods=5))
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_long_period(self):
        # [FUNCTION DOC]
        # Timedelta index with day spacing plots.
        # test long period
        index = timedelta_range("1 day 2 hr 30 min 10 s", periods=10, freq="1 D")
        s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_timedelta_short_period(self):
        # [FUNCTION DOC]
        # Timedelta index with nanosecond spacing plots.
        # test short period
        index = timedelta_range("1 day 2 hr 30 min 10 s", periods=10, freq="1 ns")
        s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        _, ax = mpl.pyplot.subplots()
        _check_plot_works(s.plot, ax=ax)

    def test_hist(self):
        # [FUNCTION DOC]
        # Matplotlib issue 8459. A matplotlib histogram of two datetime index objects with
        # weights does not raise.
        # https://github.com/matplotlib/matplotlib/issues/8459
        rng = date_range("1/1/2011", periods=10, freq="h")
        x = rng
        w1 = np.arange(0, 1, 0.1)
        w2 = np.arange(0, 1, 0.1)[::-1]
        _, ax = mpl.pyplot.subplots()
        ax.hist([x, x], weights=[w1, w2])

    def test_overlapping_datetime(self):
        # [FUNCTION DOC]
        # GB 6608. A series, a second series with partly overlapping dates, then the first
        # again on the same axes do not raise.
        # GB 6608
        s1 = Series(
            [1, 2, 3],
            index=[
                datetime(1995, 12, 31),
                datetime(2000, 12, 31),
                datetime(2005, 12, 31),
            ],
        )
        s2 = Series(
            [1, 2, 3],
            index=[
                datetime(1997, 12, 31),
                datetime(2003, 12, 31),
                datetime(2008, 12, 31),
            ],
        )

        # plot first series, then add the second series to those axes,
        # then try adding the first series again
        _, ax = mpl.pyplot.subplots()
        s1.plot(ax=ax)
        s2.plot(ax=ax)
        s1.plot(ax=ax)

    @pytest.mark.xfail(reason="GH9053 matplotlib does not use ax.xaxis.converter")
    def test_add_matplotlib_datetime64(self):
        # [FUNCTION DOC]
        # Expected to fail (GH9053). After a pandas plot, adding a matplotlib plot of
        # datetime64 data (emitting a deprecation warning) should give two lines with the
        # same data; fails because matplotlib overrides the axis converter.
        # GH9053 - ensure that a plot with PeriodConverter still understands
        # datetime64 data. This still fails because matplotlib overrides the
        # ax.xaxis.converter with a DatetimeConverter
        s = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1970-01-02", periods=10),
        )
        ax = s.plot()
        with tm.assert_produces_warning(DeprecationWarning):
            # multi-dimensional indexing
            ax.plot(s.index, s.values, color="g")
        l1, l2 = ax.lines
        tm.assert_numpy_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_matplotlib_scatter_datetime64(self):
        # [FUNCTION DOC]
        # Matplotlib issue 11391. Scatter with a datetime column by name gives a first x
        # tick label of the first date.
        # https://github.com/matplotlib/matplotlib/issues/11391
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=["x", "y"])
        df["time"] = date_range("2018-01-01", periods=10, freq="D")
        _, ax = mpl.pyplot.subplots()
        ax.scatter(x="time", y="y", data=df)
        mpl.pyplot.draw()
        label = ax.get_xticklabels()[0]
        expected = "2018-01-01"
        assert label.get_text() == expected

    def test_check_xticks_rot(self):
        # [FUNCTION DOC]
        # Issue 29460. A regular daily time series frame plot has x tick rotation 0.
        # https://github.com/pandas-dev/pandas/issues/29460
        # regular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        axes = df.plot(x="x", y="y")
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_irregular(self):
        # [FUNCTION DOC]
        # An irregular time series has x tick rotation 30.
        # irregular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        axes = df.plot(x="x", y="y")
        _check_ticks_props(axes, xrot=30)

    def test_check_xticks_rot_use_idx(self):
        # [FUNCTION DOC]
        # With the time series index used, rotation is 30; when the index is not used,
        # rotation is 0.
        # irregular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # use timeseries index or not
        axes = df.set_index("x").plot(y="y", use_index=True)
        _check_ticks_props(axes, xrot=30)
        axes = df.set_index("x").plot(y="y", use_index=False)
        _check_ticks_props(axes, xrot=0)

    def test_check_xticks_rot_sharex(self):
        # [FUNCTION DOC]
        # Subplots with sharex true give rotation 30; with sharex false give 0.
        # irregular time series
        x = to_datetime(["2020-05-01", "2020-05-02", "2020-05-04"])
        df = DataFrame({"x": x, "y": [1, 2, 3]})
        # separate subplots
        axes = df.plot(x="x", y="y", subplots=True, sharex=True)
        _check_ticks_props(axes, xrot=30)
        axes = df.plot(x="x", y="y", subplots=True, sharex=False)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize(
        "idx",
        [
            date_range("2020-01-01", periods=5),
            date_range("2020-01-01", periods=5, tz="UTC"),
            timedelta_range("1 day", periods=5, freq="D"),
            period_range("2020-01-01", periods=5, freq="D"),
            Index([date(2000, 1, i) for i in [1, 3, 6, 20, 22]], dtype=object),
            range(5),
        ],
    )
    def test_pickle_fig(self, temp_file, frame_or_series, idx):
        # [FUNCTION DOC]
        # Takes the temp file and frame-or-series fixtures. Parametrized over several
        # index kinds (dates, timezone-aware dates, timedeltas, periods, date objects,
        # range). GH18439, GH#24088. Plots on a figure and pickles the figure to a temp
        # file; must not raise.
        # GH18439, GH#24088, statsmodels#4772
        df = frame_or_series(range(5), index=idx)
        fig, ax = plt.subplots(1, 1)
        df.plot(ax=ax)
        with temp_file.open(mode="wb") as path:
            pickle.dump(fig, path)

    @pytest.mark.parametrize("kind", ["bar", "barh"])
    def test_bar_plot_datetime_index_inferred_freq(self, kind):
        # [FUNCTION DOC]
        # GH#66771. Parametrized over bar and barh. A datetime index without frequency
        # attribute but with an inferable frequency: bar plots work and the category axis
        # tick labels are the full date and time strings.
        # GH#66771 - the index freq attribute is unset but inferable, so the
        # bar plot must resolve the freq instead of raising AttributeError
        idx = DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
        assert idx.freq is None
        df = DataFrame({"A": [1, 2, 3]}, index=idx)

        ax = df.plot(kind=kind)

        ax.get_figure().canvas.draw()
        axis = ax.get_yaxis() if kind == "barh" else ax.get_xaxis()
        labels = [t.get_text() for t in axis.get_ticklabels()]
        assert labels == [
            "2020-01-01 00:00:00",
            "2020-01-02 00:00:00",
            "2020-01-03 00:00:00",
        ]

    @pytest.mark.parametrize("kind", ["bar", "barh"])
    def test_bar_plot_datetime_index_freq_from_axes(self, kind):
        # [FUNCTION DOC]
        # GH#66771. Parametrized over bar and barh. After a line plot sets the axes
        # frequency, bar plot of a series with an index without frequency or inferred
        # frequency (the axes frequency, stored as an alias string, must be normalized):
        # bar centers equal the daily ordinals.
        # GH#66771 - the index carries no freq of its own, but the axes was
        # already decorated by a line plot; that freq is stored as a period
        # alias string, so it has to be normalized to an offset before it
        # reaches the converter
        _, ax = plt.subplots()
        Series(
            np.arange(10.0), index=date_range("2020-01-01", periods=10, freq="D")
        ).plot(ax=ax)

        idx = DatetimeIndex(["2020-01-02", "2020-01-05"])
        assert idx.freq is None
        assert idx.inferred_freq is None
        Series([1.0, 2.0], index=idx).plot(kind=kind, ax=ax)

        # the bars are centered on the same daily ordinals as the line
        if kind == "bar":
            centers = [p.get_x() + p.get_width() / 2 for p in ax.patches]
        else:
            centers = [p.get_y() + p.get_height() / 2 for p in ax.patches]
        assert centers == [
            Period("2020-01-02", freq="D").ordinal,
            Period("2020-01-05", freq="D").ordinal,
        ]


def _check_plot_works(f, freq=None, series=None, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: used by the tests of this module (a local version that differs from
    # the shared plotting helper). Parameters: f is a plotting callable; freq is an
    # optional expected frequency; series is an optional series whose index
    # frequency (or rule code) is compared with the axes frequency; extra positional
    # arguments and keyword arguments are forwarded to f (the axes keyword is
    # removed from the keywords and not passed on the first call). Behavior: clears
    # the current figure, adds a top subplot of a 2x1 grid, calls f once (the result
    # must not be none), checks the frequency conditions against the axes only if the
    # original axes had no frequency attribute, then adds the bottom subplot, sets it
    # as the axes keyword and calls f again, asserting a result. Returns nothing.
    # Side effects: clears and draws on the current figure.
    # KNOWN ISSUE: the first call to f does not receive the axes keyword that
    # callers pass (it is removed beforehand), and the local axes variable assigned
    # from the first subplot is overwritten without being used.
    fig = plt.gcf()

    fig.clf()
    ax = fig.add_subplot(211)
    orig_ax = kwargs.pop("ax", plt.gca())
    orig_axfreq = getattr(orig_ax, "freq", None)

    ret = f(*args, **kwargs)
    assert ret is not None  # do something more intelligent

    ax = kwargs.pop("ax", plt.gca())
    if series is not None:
        dfreq = series.index.freq
        if isinstance(dfreq, BaseOffset):
            dfreq = dfreq.rule_code
        if orig_axfreq is None:
            assert ax.freq == dfreq

    if freq is not None and orig_axfreq is None:
        assert to_offset(ax.freq, is_period=True) == freq

    ax = fig.add_subplot(212)
    kwargs["ax"] = ax
    ret = f(*args, **kwargs)
    assert ret is not None  # TODO: do something more intelligent
