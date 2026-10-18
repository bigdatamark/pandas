"""Test cases for Series.plot"""

# [MODULE DOC]
# Tests for the plotting accessor of Series (line, bar, area, pie, kernel density, box,
# histogram and other kinds). Skipped if matplotlib is missing. A module-level mark ignores two
# numeric runtime warnings (divide by zero and invalid multiply in scalar operations).
# Provides two fixtures (a date-indexed float series named ts and a string-indexed float series
# named series) and a single test class. Covers plot keywords (scales, rotation, labels,
# colors, ticks, limits), secondary axes and legends, error bars, tables, pie handling of
# missing values and negatives, log bar axes, kernel density keywords, invalid data errors,
# standard-color helper behavior and tick label behavior. Uses the shared plotting test
# helpers and the private datetime converter and color helper of the matplotlib backend.

from datetime import datetime
from itertools import chain

import numpy as np
import pytest

from pandas.compat import is_platform_linux
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    period_range,
    plotting,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_grid_settings,
    _check_has_errorbars,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    _unpack_cycler,
    get_y_axis,
)
from pandas.util.version import Version

from pandas.tseries.offsets import CustomBusinessDay

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

from pandas.plotting._matplotlib.converter import DatetimeConverter
from pandas.plotting._matplotlib.style import get_standard_colors

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar multiply:RuntimeWarning"
    ),
]


@pytest.fixture
def ts():
    # [FUNCTION DOC]
    # Fixture: float series of ten values counting up from zero, indexed by ten consecutive
    # days starting 2020-01-01, named ts.
    return Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10),
        name="ts",
    )


@pytest.fixture
def series():
    # [FUNCTION DOC]
    # Fixture: float series of ten values counting up from zero, indexed by the strings i_0 to
    # i_9, named series.
    return Series(
        range(10), dtype=np.float64, name="series", index=[f"i_{i}" for i in range(10)]
    )


class TestSeriesPlots:
    # [CLASS DOC]
    # Tests for plotting a Series through its plot accessor and related backend helpers.
    # Stateless; plots are created in matplotlib global state.
    @pytest.mark.slow
    @pytest.mark.parametrize("kwargs", [{"label": "foo"}, {"use_index": False}])
    def test_plot(self, ts, kwargs):
        # [FUNCTION DOC]
        # Slow smoke test, parametrized by keywords (a custom label, or turning off use of the
        # index): plotting the date-indexed series works.
        _check_plot_works(ts.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_tick_props(self, ts):
        # [FUNCTION DOC]
        # Slow test: plotting with a rotation of zero applies a zero x tick rotation.
        axes = _check_plot_works(ts.plot, rot=0)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "scale, exp_scale",
        [
            [{"logy": True}, {"yaxis": "log"}],
            [{"logx": True}, {"xaxis": "log"}],
            [{"loglog": True}, {"xaxis": "log", "yaxis": "log"}],
        ],
    )
    def test_plot_scales(self, ts, scale, exp_scale):
        # [FUNCTION DOC]
        # Slow test, parametrized by scale keyword and expected axis scales (log y, log x, both).
        # Plots with a dotted style and verifies the axes scales are logarithmic as expected.
        ax = _check_plot_works(ts.plot, style=".", **scale)
        _check_ax_scales(ax, **exp_scale)

    @pytest.mark.slow
    def test_plot_ts_bar(self, ts):
        # [FUNCTION DOC]
        # Slow smoke test: a bar plot of the first ten entries of the date-indexed series works.
        _check_plot_works(ts[:10].plot.bar)

    @pytest.mark.slow
    def test_plot_ts_area_stacked(self, ts):
        # [FUNCTION DOC]
        # Slow smoke test: an unstacked area plot of the date-indexed series works.
        _check_plot_works(ts.plot.area, stacked=False)

    def test_plot_iseries(self):
        # [FUNCTION DOC]
        # Smoke test: plotting a series with a period index works.
        ser = Series(range(5), period_range("2020-01-01", periods=5))
        _check_plot_works(ser.plot)

    @pytest.mark.parametrize(
        "kind",
        [
            "line",
            "bar",
            "barh",
            pytest.param("kde", marks=td.skip_if_no("scipy")),
            "hist",
            "box",
        ],
    )
    def test_plot_series_kinds(self, series, kind):
        # [FUNCTION DOC]
        # Parametrized over line, bar, horizontal bar, kernel density (needs scipy), histogram and
        # box kinds. Smoke test that the first five entries plot with each kind.
        _check_plot_works(series[:5].plot, kind=kind)

    def test_plot_series_barh(self, series):
        # [FUNCTION DOC]
        # Smoke test: horizontal bar plot of the ten-entry series works.
        _check_plot_works(series[:10].plot.barh)

    def test_plot_series_bar_ax(self):
        # [FUNCTION DOC]
        # Verifies a bar plot of ten random values with the color black has a black face color on
        # its first patch.
        ax = _check_plot_works(
            Series(np.random.default_rng(2).standard_normal(10)).plot.bar, color="black"
        )
        _check_colors([ax.patches[0]], facecolors=["black"])

    @pytest.mark.parametrize("kwargs", [{}, {"layout": (-1, 1)}, {"layout": (1, -1)}])
    def test_plot_6951(self, ts, kwargs):
        # [FUNCTION DOC]
        # Regression test (GH 6951), parametrized by layout keyword (none, minus-one by one, one by
        # minus-one). Plotting with subplots enabled for a single series gives one axes in a
        # one-by-one layout.
        # GH 6951
        ax = _check_plot_works(ts.plot, subplots=True, **kwargs)
        _check_axes_shape(ax, axes_num=1, layout=(1, 1))

    def test_plot_figsize_and_title(self, series):
        # [FUNCTION DOC]
        # Verifies the title and figure size arguments: the axes has the given title and the
        # requested figure size with a single axes in a one-by-one layout.
        # figsize and title
        _, ax = mpl.pyplot.subplots()
        ax = series.plot(title="Test", figsize=(16, 8), ax=ax)
        _check_text_labels(ax.title, "Test")
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self):
        # [FUNCTION DOC]
        # Regression test (GH 8242): plotting a series must not change the global property cycle
        # setting of matplotlib.
        # GH 8242
        key = "axes.prop_cycle"
        colors = mpl.pyplot.rcParams[key]
        _, ax = mpl.pyplot.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        assert colors == mpl.pyplot.rcParams[key]

    @pytest.mark.parametrize("kwargs", [{}, {"secondary_y": True}])
    def test_ts_line_lim(self, ts, kwargs):
        # [FUNCTION DOC]
        # Parametrized by extra keywords (none or secondary y axis). Verifies the x limits of the
        # axes enclose the first line's x data.
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax, **kwargs)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self, ts):
        # [FUNCTION DOC]
        # Verifies an unstacked area plot has x limits enclosing the line data and x tick rotation
        # zero.
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_area_lim_xcompat(self, ts):
        # [FUNCTION DOC]
        # Regression test (GH 7471): same as the area limits test but with x_compat enabled; the
        # x limits enclose the data and x tick rotation is 30 degrees.
        # GH 7471
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=30)

    def test_ts_tz_area_lim_xcompat(self, ts):
        # [FUNCTION DOC]
        # Verifies an unstacked area plot with x_compat on a series whose index was localized to
        # GMT and converted to CET: x limits enclose the data and rotation is zero.
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_tz_area_lim_xcompat_secondary_y(self, ts):
        # [FUNCTION DOC]
        # Same as the previous test but plotted on a secondary y axis instead of using x_compat:
        # x limits enclose the data and rotation is zero.
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_area_sharey_dont_overwrite(self, ts):
        # [FUNCTION DOC]
        # Regression test (GH37942): two area plots drawn on two subplots created with shared y
        # axes must leave the y axes joined (not overwritten).
        # GH37942
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)

        abs(ts).plot(ax=ax1, kind="area")
        abs(ts).plot(ax=ax2, kind="area")

        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    def test_label(self):
        # [FUNCTION DOC]
        # Verifies that a legend with an explicit label shows that label.
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(label="LABEL", legend=True, ax=ax)
        _check_legend_labels(ax, labels=["LABEL"])

    def test_label_none(self):
        # [FUNCTION DOC]
        # Verifies that an unnamed series with no label shows an empty legend label.
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=[""])

    def test_label_ser_name(self):
        # [FUNCTION DOC]
        # Verifies that the series name is used as the legend label by default.
        s = Series([1, 2], name="NAME")
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=["NAME"])

    def test_label_ser_name_override(self):
        # [FUNCTION DOC]
        # Verifies an explicit label overrides the series name in the legend.
        s = Series([1, 2], name="NAME")
        # override the default
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, label="LABEL", ax=ax)
        _check_legend_labels(ax, labels=["LABEL"])

    def test_label_ser_name_override_dont_draw(self):
        # [FUNCTION DOC]
        # Verifies that when the legend is disabled but a label is given, no legend exists after
        # plotting, and drawing a legend afterwards shows the label.
        s = Series([1, 2], name="NAME")
        # Add lebel info, but don't draw
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=False, label="LABEL", ax=ax)
        assert ax.get_legend() is None  # Hasn't been drawn
        ax.legend()  # draw it
        _check_legend_labels(ax, labels=["LABEL"])

    def test_boolean(self):
        # [FUNCTION DOC]
        # Regression test (GH 23719): a boolean series plots when including boolean data is
        # requested, and otherwise raises a type error saying there is no numeric data to plot.
        # GH 23719
        s = Series([False, False, True])
        _check_plot_works(s.plot, include_bool=True)

        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            _check_plot_works(s.plot)

    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=4)])
    def test_line_area_nan_series(self, index):
        # [FUNCTION DOC]
        # Parametrized by index (default or dates). Plots a series with a missing value as a line:
        # the y data is a masked array with the missing point masked (values one, two, three after
        # removal), and with stacking or as an area plot (stacked or not) the missing value is
        # drawn as zero.
        values = [1, 2, np.nan, 3]
        d = Series(values, index=index)
        ax = _check_plot_works(d.plot)
        masked = ax.lines[0].get_ydata()
        # remove nan for comparison purpose
        exp = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
        tm.assert_numpy_array_equal(masked.mask, np.array([False, False, True, False]))

        expected = np.array([1, 2, 0, 3], dtype=np.float64)
        ax = _check_plot_works(d.plot, stacked=True)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area, stacked=False)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    def test_line_use_index_false(self):
        # [FUNCTION DOC]
        # Verifies that a line plot with index use disabled has an empty x label even when the
        # index is named.
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(use_index=False, ax=ax)
        label = ax.get_xlabel()
        assert label == ""

    def test_line_use_index_false_diff_var(self):
        # [FUNCTION DOC]
        # Same as above but for a bar plot: the x label is empty when the index is not used.
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = mpl.pyplot.subplots()
        ax2 = s.plot.bar(use_index=False, ax=ax)
        label2 = ax2.get_xlabel()
        assert label2 == ""

    @pytest.mark.xfail(
        is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize("axis, meth", [("yaxis", "bar"), ("xaxis", "barh")])
    def test_bar_log(self, axis, meth):
        # [FUNCTION DOC]
        # Parametrized by axis name and bar method (vertical bar with y axis, horizontal bar with
        # x axis). Expected to fail on linux (non-strict) because of rounding issues. Plots a
        # two-value series on a log axis and checks the tick locations are the decades; with
        # matplotlib 3.11 or later the outermost tick is dropped (GH#65918).
        # matplotlib 3.11 no longer pads log-axis ticks a decade past the
        #  view limits, so the outermost tick is dropped (GH#65918)
        expected = (
            np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
            if Version(mpl.__version__) < Version("3.11.0rc1")
            else np.array([1e-1, 1e0, 1e1, 1e2, 1e3])
        )

        _, ax = mpl.pyplot.subplots()
        ax = getattr(Series([200, 500]).plot, meth)(log=True, ax=ax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    @pytest.mark.xfail(
        is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize(
        "axis, kind, res_meth",
        [["yaxis", "bar", "get_ylim"], ["xaxis", "barh", "get_xlim"]],
    )
    def test_bar_log_kind_bar(self, axis, kind, res_meth):
        # [FUNCTION DOC]
        # Regression test (GH 9905), parametrized by axis, kind and limit accessor. Expected to fail
        # on linux (non-strict). Plots a small series with a log value axis and verifies the view
        # limits against fixed numbers and the tick locations against decade lists, which differ for
        # matplotlib 3.11 or later (GH#65918).
        # GH 9905
        # matplotlib 3.11 no longer pads log-axis ticks a decade past the
        #  view limits, so the outermost ticks are dropped (GH#65918)
        expected = (
            np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
            if Version(mpl.__version__) < Version("3.11.0rc1")
            else np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        )

        _, ax = mpl.pyplot.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind=kind, ax=ax)
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = getattr(ax, res_meth)()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    def test_bar_ignore_index(self):
        # [FUNCTION DOC]
        # Verifies a bar plot with index use disabled labels the x ticks with integer positions as
        # text, zero to three.
        df = Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.bar(use_index=False, ax=ax)
        _check_text_labels(ax.get_xticklabels(), ["0", "1", "2", "3"])

    def test_bar_user_colors(self):
        # [FUNCTION DOC]
        # Verifies a bar plot with a list of user colors gives the patches those colors as RGBA
        # tuples in order.
        s = Series([1, 2, 3, 4])
        ax = s.plot.bar(color=["red", "blue", "blue", "red"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_rotation_default(self):
        # [FUNCTION DOC]
        # Verifies the default tick rotation of a data frame line plot is zero.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # Default rot 0
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(ax=ax)
        _check_ticks_props(axes, xrot=0)

    def test_rotation_30(self):
        # [FUNCTION DOC]
        # Verifies a rotation argument of 30 is applied to the x tick labels.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(rot=30, ax=ax)
        _check_ticks_props(axes, xrot=30)

    def test_irregular_datetime(self):
        # [FUNCTION DOC]
        # Verifies an irregular datetime index plot: after setting x limits to 1999 and 2001, the
        # lower limit equals the converter's conversion of January 1 1999, and tick rotation is 30
        # degrees.
        rng = date_range("1/1/2000", "1/15/2000")
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xp = DatetimeConverter.convert(datetime(1999, 1, 1), "", ax)
        ax.set_xlim("1/1/1999", "1/1/2001")
        assert xp == ax.get_xlim()[0]
        _check_ticks_props(ax, xrot=30)

    def test_unsorted_index_xlim(self):
        # [FUNCTION DOC]
        # Verifies plotting a series with an unsorted, missing-containing float index sets x limits
        # enclosing the minimum and maximum (ignoring missing) of the plotted x data.
        ser = Series(
            [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
        )
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
        assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])

    def test_pie_series(self):
        # [FUNCTION DOC]
        # Verifies a pie plot of a five-entry series: the text labels are the index entries and the y
        # label is empty.
        # KNOWN ISSUE: the series is built from one random integer scalar broadcast across the five-entry index, so all values are equal, and the leading comment about sums under one describes a case that is not constructed here.
        # if sum of values is less than 1.0, pie handle them as rate and draw
        # semicircle.
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        ax = _check_plot_works(series.plot.pie)
        _check_text_labels(ax.texts, series.index)
        assert ax.get_ylabel() == ""

    def test_pie_arrow_type(self):
        # [FUNCTION DOC]
        # Regression test (GH 59192): a pie plot works for a series with a pyarrow-backed integer
        # dtype (skipped if pyarrow is missing).
        # GH 59192
        pytest.importorskip("pyarrow")
        ser = Series([1, 2, 3, 4], dtype="int32[pyarrow]")
        _check_plot_works(ser.plot.pie)

    def test_pie_series_no_label(self):
        # [FUNCTION DOC]
        # Verifies that disabling labels leaves five empty text labels on the pie plot.
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        ax = _check_plot_works(series.plot.pie, labels=None)
        _check_text_labels(ax.texts, [""] * 5)

    def test_pie_series_less_colors_than_elements(self):
        # [FUNCTION DOC]
        # Verifies that when fewer colors than elements are given (three for five slices) the colors
        # cycle: red, green, blue, red, green.
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        color_args = ["r", "g", "b"]
        ax = _check_plot_works(series.plot.pie, colors=color_args)

        color_expected = ["r", "g", "b", "r", "g"]
        _check_colors(ax.patches, facecolors=color_expected)

    def test_pie_series_labels_and_colors(self):
        # [FUNCTION DOC]
        # Verifies a pie plot with explicit labels and colors uses them for text labels and patch face
        # colors.
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # with labels and colors
        labels = ["A", "B", "C", "D", "E"]
        color_args = ["r", "g", "b", "c", "m"]
        ax = _check_plot_works(series.plot.pie, labels=labels, colors=color_args)
        _check_text_labels(ax.texts, labels)
        _check_colors(ax.patches, facecolors=color_args)

    def test_pie_series_autopct_and_fontsize(self):
        # [FUNCTION DOC]
        # Verifies a pie plot with colors, a two-decimal percentage format and font size seven:
        # the texts consist of each label and its percentage, with ordering depending on the
        # matplotlib version (before 3.11 interleaved pairs; from 3.11 all labels then all
        # percentages), and every text has font size seven.
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        color_args = ["r", "g", "b", "c", "m"]
        ax = _check_plot_works(
            series.plot.pie, colors=color_args, autopct="%.2f", fontsize=7
        )
        pcts = [f"{s * 100:.2f}" for s in series.values / series.sum()]
        expected_texts = (
            list(chain.from_iterable(zip(series.index, pcts, strict=True)))
            if Version(mpl.__version__) < Version("3.11.0rc1")
            else list(chain.from_iterable((series.index, pcts)))
        )
        _check_text_labels(ax.texts, expected_texts)
        for t in ax.texts:
            assert t.get_fontsize() == 7

    def test_pie_series_negative_raises(self):
        # [FUNCTION DOC]
        # Verifies a pie plot with a negative value raises a value error saying negative values are
        # not allowed.
        # includes negative value
        series = Series([1, 2, 0, 4, -1], index=["a", "b", "c", "d", "e"])
        with pytest.raises(ValueError, match="pie plot doesn't allow negative values"):
            series.plot.pie()

    def test_pie_series_nan(self):
        # [FUNCTION DOC]
        # Verifies a pie plot with a missing value gives an empty text label for the missing slice.
        # includes nan
        series = Series([1, 2, np.nan, 4], index=["a", "b", "c", "d"], name="YLABEL")
        ax = _check_plot_works(series.plot.pie)
        _check_text_labels(ax.texts, ["a", "b", "", "d"])

    def test_pie_nan(self):
        # [FUNCTION DOC]
        # Verifies a pie plot with a legend and a missing value: the texts are the index positions
        # with the missing one empty.
        s = Series([1, np.nan, 1, 1])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot.pie(legend=True, ax=ax)
        expected = ["0", "", "2", "3"]
        result = [x.get_text() for x in ax.texts]
        assert result == expected

    def test_df_series_secondary_legend(self):
        # [FUNCTION DOC]
        # Regression test (GH 9779): a series with a secondary axis drawn onto an axes holding a
        # frame's primary plot shows all legends on the left axes with the series labeled as right,
        # and both y axes are visible.
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(10), name="x")

        # primary -> secondary (without passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are drawn on left ax
        # left and right axis must be visible
        _check_legend_labels(ax, labels=["a", "b", "c", "x (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both(self):
        # [FUNCTION DOC]
        # Regression test (GH 9779): frame and series both on secondary axes: all legend entries on
        # the left axes labeled as right, left y axis hidden, right y axis visible.
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(10), name="x")
        # secondary -> secondary (without passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a (right)", "b (right)", "c (right)", "x (right)"]
        _check_legend_labels(ax.left_ax, labels=expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both_with_axis_2(self):
        # [FUNCTION DOC]
        # Regression test (GH 9779): like the previous test but with right-marking disabled for the
        # frame, so the frame entries have no suffix and the series entry is labeled as right.
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(10), name="x")
        # secondary -> secondary (with passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, mark_right=False, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a", "b", "c", "x (right)"]
        _check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    @pytest.mark.parametrize(
        "input_logy, expected_scale", [(True, "log"), ("sym", "symlog")]
    )
    @pytest.mark.parametrize("secondary_kwarg", [{}, {"secondary_y": True}])
    def test_secondary_logy(self, input_logy, expected_scale, secondary_kwarg):
        # [FUNCTION DOC]
        # Regression test (GH 25545, GH 24980): the log y argument value true gives a log scale and
        # the symmetric string gives a symmetric log scale, with and without a secondary y axis.
        # Parametrized by both.
        # GH 25545, GH 24980
        s1 = Series(np.random.default_rng(2).standard_normal(10))
        ax1 = s1.plot(logy=input_logy, **secondary_kwarg)
        assert ax1.get_yscale() == expected_scale

    def test_plot_fails_with_dupe_color_and_style(self):
        # [FUNCTION DOC]
        # Verifies that giving both a style string with a color letter and a color keyword raises a
        # value error saying only one should be used or the style should omit the color symbol.
        x = Series(np.random.default_rng(2).standard_normal(2))
        _, ax = mpl.pyplot.subplots()
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            x.plot(style="k--", color="k", ax=ax)

    @pytest.mark.parametrize(
        "bw_method, ind",
        [
            ["scott", 20],
            [None, 20],
            [None, np.int_(20)],
            [0.5, np.linspace(-100, 100, 20)],
        ],
    )
    def test_kde_kwargs(self, ts, bw_method, ind):
        # [FUNCTION DOC]
        # Smoke test, parametrized by bandwidth method and sample points (a count, an integer
        # scalar, or an explicit array): the kernel density plot accepts them (skipped without scipy).
        pytest.importorskip("scipy")
        _check_plot_works(ts.plot.kde, bw_method=bw_method, ind=ind)

    @pytest.mark.parametrize(
        "bw_method, ind, weights",
        [
            ["scott", 20, None],
            [None, 20, None],
            [None, np.int_(20), None],
            [0.5, np.linspace(-100, 100, 20), None],
            ["scott", 40, np.linspace(0.0, 2.0, 50)],
        ],
    )
    def test_kde_kwargs_weights(self, bw_method, ind, weights):
        # [FUNCTION DOC]
        # Regression test (GH59337): kernel density plot with weights, on a fifty-value uniform
        # series; parametrized by bandwidth method, sample points and weights (none or fifty
        # weights). Skipped without scipy.
        # GH59337
        pytest.importorskip("scipy")
        s = Series(np.random.default_rng(2).uniform(size=50))
        _check_plot_works(s.plot.kde, bw_method=bw_method, ind=ind, weights=weights)

    def test_density_kwargs(self, ts):
        # [FUNCTION DOC]
        # Smoke test: the density plot accessor accepts a bandwidth and an explicit sample array
        # (skipped without scipy).
        pytest.importorskip("scipy")
        sample_points = np.linspace(-100, 100, 20)
        _check_plot_works(ts.plot.density, bw_method=0.5, ind=sample_points)

    def test_kde_kwargs_check_axes(self, ts):
        # [FUNCTION DOC]
        # Verifies a kernel density plot with log y has a log y scale and the y axis label is
        # density (skipped without scipy).
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        sample_points = np.linspace(-100, 100, 20)
        ax = ts.plot.kde(logy=True, bw_method=0.5, ind=sample_points, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        _check_text_labels(ax.yaxis.get_label(), "Density")

    def test_kde_missing_vals(self):
        # [FUNCTION DOC]
        # Regression test (gh-14821): a kernel density plot of a series with a missing value still
        # has non-missing x data (skipped without scipy).
        pytest.importorskip("scipy")
        s = Series(np.random.default_rng(2).uniform(size=50))
        s[0] = np.nan
        axes = _check_plot_works(s.plot.kde)

        # gh-14821: check if the values have any missing values
        assert any(~np.isnan(axes.lines[0].get_xdata()))

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_boxplot_series(self, ts):
        # [FUNCTION DOC]
        # Expected-failure test (matplotlib API change in 3.6.0). Box plot with log y; checks log
        # scale, the single x tick label is the series name, and y tick labels are blank.
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.box(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [ts.name])
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    @pytest.mark.parametrize(
        "kind",
        plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds,
    )
    def test_kind_kwarg(self, kind):
        # [FUNCTION DOC]
        # Parametrized over common and series-specific plot kinds. Plots a three-element series with
        # the kind given as a keyword and closes the figure; skipped without scipy.
        pytest.importorskip("scipy")
        s = Series(range(3))
        _, ax = mpl.pyplot.subplots()
        s.plot(kind=kind, ax=ax)
        mpl.pyplot.close()

    @pytest.mark.parametrize(
        "kind",
        plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds,
    )
    def test_kind_attr(self, kind):
        # [FUNCTION DOC]
        # Parametrized over common and series-specific kinds. Plots with the kind used as accessor
        # attribute and closes the figure; skipped without scipy.
        # KNOWN ISSUE: the axes created at the start is never used by the attribute call, so the attribute plot draws on a different axes.
        pytest.importorskip("scipy")
        s = Series(range(3))
        _, ax = mpl.pyplot.subplots()
        getattr(s.plot, kind)()
        mpl.pyplot.close()

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_invalid_plot_data(self, kind):
        # [FUNCTION DOC]
        # Parametrized over common kinds. A series of strings raises a type error saying there is
        # no numeric data to plot.
        s = Series(list("abcd"))
        _, ax = mpl.pyplot.subplots()
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            s.plot(kind=kind, ax=ax)

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_valid_object_plot(self, kind):
        # [FUNCTION DOC]
        # Parametrized over common kinds. A series of integers stored with object dtype plots
        # without error (skipped without scipy).
        pytest.importorskip("scipy")
        s = Series(range(10), dtype=object)
        _check_plot_works(s.plot, kind=kind)

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_partially_invalid_plot_data(self, kind):
        # [FUNCTION DOC]
        # Parametrized over common kinds. A series mixing strings and numbers raises a type error
        # saying there is no numeric data to plot.
        s = Series(["a", "b", 1.0, 2])
        _, ax = mpl.pyplot.subplots()
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            s.plot(kind=kind, ax=ax)

    def test_invalid_kind(self):
        # [FUNCTION DOC]
        # Verifies an unknown plot kind name raises a value error saying it is not a valid plot kind.
        s = Series([1, 2])
        with pytest.raises(ValueError, match="invalid_kind is not a valid plot kind"):
            s.plot(kind="invalid_kind")

    def test_dup_datetime_index_plot(self):
        # [FUNCTION DOC]
        # Smoke test: plotting a series whose datetime index has duplicates works.
        dr1 = date_range("1/1/2009", periods=4)
        dr2 = date_range("1/2/2009", periods=4)
        index = dr1.append(dr2)
        values = np.random.default_rng(2).standard_normal(index.size)
        s = Series(values, index=index)
        _check_plot_works(s.plot)

    def test_errorbar_asymmetrical(self):
        # [FUNCTION DOC]
        # Regression test (GH9536): asymmetric error bars (a two-row array) for both y and x:
        # the plotted vertical segment endpoints equal the values minus and plus the corresponding
        # error rows.
        # GH9536
        s = Series(np.arange(10), name="x")
        err = np.random.default_rng(2).random((2, 10))

        ax = s.plot(yerr=err, xerr=err)

        result = np.vstack([i.vertices[:, 1] for i in ax.collections[1].get_paths()])
        expected = (err.T * np.array([-1, 1])) + s.to_numpy().reshape(-1, 1)
        tm.assert_numpy_array_equal(result, expected)

    def test_errorbar_asymmetrical_error(self):
        # [FUNCTION DOC]
        # Regression test (GH9536): an asymmetric error array whose second dimension differs from
        # the series length raises a value error naming the required shape.
        # GH9536
        s = Series(np.arange(10), name="x")
        msg = (
            "Asymmetrical error bars should be provided "
            f"with the shape \\(2, {len(s)}\\)"
        )
        with pytest.raises(ValueError, match=msg):
            s.plot(yerr=np.random.default_rng(2).random((2, 11)))

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar"])
    @pytest.mark.parametrize(
        "yerr",
        [
            Series(np.abs(np.random.default_rng(2).standard_normal(10))),
            np.abs(np.random.default_rng(2).standard_normal(10)),
            list(np.abs(np.random.default_rng(2).standard_normal(10))),
            DataFrame(
                np.abs(np.random.default_rng(2).standard_normal((10, 2))),
                columns=["x", "y"],
            ),
        ],
    )
    def test_errorbar_plot(self, kind, yerr):
        # [FUNCTION DOC]
        # Slow test, parametrized by line or bar kind and by y error specifications (series, array,
        # list, frame with two columns). Verifies error bars are drawn for y only.
        s = Series(np.arange(10), name="x")
        ax = _check_plot_works(s.plot, yerr=yerr, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_yerr_0(self):
        # [FUNCTION DOC]
        # Slow test: giving only x errors yields x error bars and no y error bars.
        s = Series(np.arange(10), name="x")
        s_err = np.abs(np.random.default_rng(2).standard_normal(10))
        ax = _check_plot_works(s.plot, xerr=s_err)
        _check_has_errorbars(ax, xerr=1, yerr=0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "yerr",
        [
            Series(np.abs(np.random.default_rng(2).standard_normal(12))),
            DataFrame(
                np.abs(np.random.default_rng(2).standard_normal((12, 2))),
                columns=["x", "y"],
            ),
        ],
    )
    def test_errorbar_plot_ts(self, yerr):
        # [FUNCTION DOC]
        # Slow test, parametrized by series or frame of errors. Re-indexes the errors (in place)
        # to a monthly date index equal to the plotted series' index, plots a time series with y errors
        # and verifies y error bars are present and x ones absent.
        # test time series plotting
        ix = date_range("1/1/2000", "1/1/2001", freq="ME")
        ts = Series(np.arange(12), index=ix, name="x")
        yerr.index = ix

        ax = _check_plot_works(ts.plot, yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr_shape(self):
        # [FUNCTION DOC]
        # Slow test: a y error array with wrong length raises a value error from the plotting
        # library (matplotlib), checked as an externally raised error.
        s = Series(np.arange(10), name="x")
        # check incorrect lengths and types
        with tm.external_error_raised(ValueError):
            s.plot(yerr=np.arange(11))

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr(self):
        # [FUNCTION DOC]
        # Slow test: string errors raise a type error from the plotting library.
        s = Series(np.arange(10), name="x")
        s_err = ["zzz"] * 10
        with tm.external_error_raised(TypeError):
            s.plot(yerr=s_err)

    @pytest.mark.slow
    def test_table_true(self, series):
        # [FUNCTION DOC]
        # Slow smoke test: plotting with a table enabled works.
        _check_plot_works(series.plot, table=True)

    @pytest.mark.slow
    def test_table_self(self, series):
        # [FUNCTION DOC]
        # Slow smoke test: plotting with the table argument given as the series itself works.
        _check_plot_works(series.plot, table=series)

    @pytest.mark.slow
    def test_series_grid_settings(self):
        # [FUNCTION DOC]
        # Slow regression test (GH 9792): helper-based check that plots default to the global grid
        # setting for all series and common kinds (skipped without scipy).
        # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792
        pytest.importorskip("scipy")
        _check_grid_settings(
            Series([1, 2, 3]),
            plotting.PlotAccessor._series_kinds + plotting.PlotAccessor._common_kinds,
        )

    @pytest.mark.parametrize("c", ["r", "red", "green", "#FF0000"])
    def test_standard_colors(self, c):
        # [FUNCTION DOC]
        # Parametrized by color (letter, name, name, hex). Verifies the standard color helper returns
        # that color wrapped in a list for one color given as scalar or list, and repeated three
        # times when three are requested.
        result = get_standard_colors(1, color=c)
        assert result == [c]

        result = get_standard_colors(1, color=[c])
        assert result == [c]

        result = get_standard_colors(3, color=c)
        assert result == [c] * 3

        result = get_standard_colors(3, color=[c])
        assert result == [c] * 3

    def test_standard_colors_all(self):
        # [FUNCTION DOC]
        # Verifies the standard color helper for every named matplotlib color and every single
        # letter color: returns the color repeated per the requested count (one or three), for scalar
        # and list inputs.
        # multiple colors like mediumaquamarine
        for c in mpl.colors.cnames:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

        # single letter colors like k
        for c in mpl.colors.ColorConverter.colors:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

    def test_series_plot_color_kwargs(self):
        # [FUNCTION DOC]
        # Regression test (GH1890): a series plot with the color green has green lines.
        # GH1890
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1).plot(color="green", ax=ax)
        _check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_kwargs(self):
        # [FUNCTION DOC]
        # Regression test (GH1890): same as above for a date-indexed series.
        # #1890
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1, index=date_range("1/1/2000", periods=12)).plot(
            color="green", ax=ax
        )
        _check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_with_empty_kwargs(self):
        # [FUNCTION DOC]
        # Verifies that plotting the same date-indexed series three times on the same axes uses the
        # first three default colors of matplotlib's color cycle.
        def_colors = _unpack_cycler(mpl.rcParams)
        index = date_range("1/1/2000", periods=12)
        s = Series(np.arange(1, 13), index=index)

        ncolors = 3

        _, ax = mpl.pyplot.subplots()
        for i in range(ncolors):
            ax = s.plot(ax=ax)
        _check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])

    def test_xticklabels(self):
        # [FUNCTION DOC]
        # Regression test (GH11529): x tick positions limited to four values show the matching
        # index strings as labels.
        # GH11529
        s = Series(np.arange(10), index=[f"P{i:02d}" for i in range(10)])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
        exp = [f"P{i:02d}" for i in [0, 3, 5, 9]]
        _check_text_labels(ax.get_xticklabels(), exp)

    def test_xtick_barPlot(self):
        # [FUNCTION DOC]
        # Regression test (GH28172): a bar plot with custom x tick positions has exactly those tick
        # positions.
        # GH28172
        s = Series(range(10), index=[f"P{i:02d}" for i in range(10)])
        ax = s.plot.bar(xticks=range(0, 11, 2))
        exp = np.array(list(range(0, 11, 2)))
        tm.assert_numpy_array_equal(exp, ax.get_xticks())

    def test_custom_business_day_freq(self):
        # [FUNCTION DOC]
        # Regression test (GH7222): plotting a series indexed by a custom business-day frequency
        # with a holiday works.
        # GH7222
        s = Series(
            range(100, 121),
            index=pd.bdate_range(
                start="2014-05-01",
                end="2014-06-01",
                freq=CustomBusinessDay(holidays=["2014-05-26"]),
            ),
        )

        _check_plot_works(s.plot)

    @pytest.mark.xfail(
        reason="GH#24426, see also "
        "github.com/pandas-dev/pandas/commit/"
        "ef1bd69fa42bbed5d09dd17f08c44fc8bfc2b685#r61470674"
    )
    def test_plot_accessor_updates_on_inplace(self):
        # [FUNCTION DOC]
        # Expected-failure test (GH#24426). Intended to verify that the plot accessor reflects an
        # in-place drop on the series.
        # KNOWN ISSUE: the second tick locations are taken from a freshly created, empty axes instead of re-plotting the modified series, so the comparison does not actually test the accessor; the test is marked as an expected failure.
        ser = Series([1, 2, 3, 4])
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        before = ax.xaxis.get_ticklocs()

        ser.drop([0, 1], inplace=True)
        _, ax = mpl.pyplot.subplots()
        after = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(before, after)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_plot_xlim_for_series(self, kind):
        # [FUNCTION DOC]
        # Regression test (GH 27686), parametrized by line or area: the x limits of a two-point
        # series extend below zero and above one.
        # test if xlim is also correctly plotted in Series for line and area
        # GH 27686
        s = Series([2, 3])
        _, ax = mpl.pyplot.subplots()
        s.plot(kind=kind, ax=ax)
        xlims = ax.get_xlim()

        assert xlims[0] < 0
        assert xlims[1] > 1

    def test_plot_no_rows(self):
        # [FUNCTION DOC]
        # Regression test (GH 27758): plotting an empty integer series gives one line with no x and
        # no y data.
        # GH 27758
        df = Series(dtype=int)
        assert df.empty
        ax = df.plot()
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self):
        # [FUNCTION DOC]
        # Verifies plotting a series of strings raises a type error saying there is no numeric
        # data to plot.
        df = Series(["a", "b", "c"])
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    @pytest.mark.parametrize(
        "data, index",
        [
            ([1, 2, 3, 4], [3, 2, 1, 0]),
            ([10, 50, 20, 30], [1910, 1920, 1980, 1950]),
        ],
    )
    def test_plot_order(self, data, index):
        # [FUNCTION DOC]
        # Regression test (GH38865), parametrized by data and index: in a bar plot, the bars sorted
        # by horizontal position have heights equal to the series values in original order (the
        # index order is not changed).
        # GH38865 Verify plot order of a Series
        ser = Series(data=data, index=index)
        ax = ser.plot(kind="bar")

        expected = ser.tolist()
        result = [
            patch.get_bbox().ymax
            for patch in sorted(ax.patches, key=lambda patch: patch.get_bbox().xmax)
        ]
        assert expected == result

    def test_style_single_ok(self):
        # [FUNCTION DOC]
        # Verifies a style of square marker plus a color keyword uses that color for the line.
        s = Series([1, 2])
        ax = s.plot(style="s", color="C3")
        assert ax.lines[0].get_color() == "C3"

    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [(None, "", "new"), ("old", "old", "new"), (None, "", "")],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar", "barh", "hist"])
    def test_xlabel_ylabel_series(self, kind, index_name, old_label, new_label):
        # [FUNCTION DOC]
        # Regression test (GH 9093), parametrized by kind, index name, old label and new label.
        # By default x label is the index name and y label empty, except for horizontal bars (swapped)
        # and histograms (empty x label and frequency y label). Explicit x and y label arguments
        # override the defaults.
        # GH 9093
        ser = Series([1, 2, 3, 4])
        ser.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name (reverse for barh)
        ax = ser.plot(kind=kind)
        if kind == "barh":
            assert ax.get_xlabel() == ""
            assert ax.get_ylabel() == old_label
        elif kind == "hist":
            assert ax.get_xlabel() == ""
            assert ax.get_ylabel() == "Frequency"
        else:
            assert ax.get_ylabel() == ""
            assert ax.get_xlabel() == old_label

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        ax = ser.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == new_label
        assert ax.get_xlabel() == new_label

    @pytest.mark.parametrize(
        "index",
        [
            pd.timedelta_range(start=0, periods=2, freq="D"),
            [pd.Timedelta(days=1), pd.Timedelta(days=2)],
        ],
    )
    def test_timedelta_index(self, index):
        # [FUNCTION DOC]
        # Regression test (GH37454), parametrized by a timedelta index built two ways. Plotting
        # with given x limits that are reversed (three then one) keeps those limits.
        # GH37454
        xlims = (3, 1)
        ax = Series([1, 2], index=index).plot(xlim=(xlims))
        assert ax.get_xlim() == (3, 1)

    def test_series_none_color(self):
        # [FUNCTION DOC]
        # Regression test (GH51953): passing no color to the plot is the same as the default
        # first color of matplotlib's cycle.
        # GH51953
        series = Series([1, 2, 3])
        ax = series.plot(color=None)
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:1]
        _check_colors(ax.get_lines(), linecolors=expected)

    @pytest.mark.slow
    def test_plot_no_warning(self, ts):
        # [FUNCTION DOC]
        # Slow regression test (GH 55138): plotting the date-indexed series produces no warnings.
        # The source notes a to-do that this can be removed once the business-period deprecation is
        # enforced.
        # GH 55138
        # TODO(3.0): this can be removed once Period[B] deprecation is enforced
        with tm.assert_produces_warning(False):
            _ = ts.plot()

    def test_secondary_y_subplot_axis_labels(self):
        # [FUNCTION DOC]
        # Regression test (GH#14102): plot a series plus a secondary-axis series on one subplot and
        # another on a second subplot; the first subplot has no minor x ticks and still has x tick
        # labels.
        # GH#14102
        s1 = Series([5, 7, 6, 8, 7], index=[1, 2, 3, 4, 5])
        s2 = Series([6, 4, 5, 3, 4], index=[1, 2, 3, 4, 5])

        ax = plt.subplot(2, 1, 1)
        s1.plot(ax=ax)
        s2.plot(ax=ax, secondary_y=True)
        ax2 = plt.subplot(2, 1, 2)
        s1.plot(ax=ax2)
        assert len(ax.xaxis.get_minor_ticks()) == 0
        assert len(ax.get_xticklabels()) > 0

    def test_bar_line_plot(self):
        # [FUNCTION DOC]
        # Regression test (GH61161): plots a bar chart then a line chart of a series with an annual
        # period index on the same axes. Verifies the year tick labels are the same for both, that
        # there is one per period, and that the x limits contain the first and last tick positions.
        """
        Test that bar and line plots with the same x values are superposed
        and that the x limits are set such that the plots are visible.
        """
        # GH61161
        index = period_range("2023", periods=3, freq="Y")
        years = set(index.year.astype(str))
        s = Series([1, 2, 3], index=index)
        ax = plt.subplot()
        s.plot(kind="bar", ax=ax)
        bar_xticks = [
            label for label in ax.get_xticklabels() if label.get_text() in years
        ]
        s.plot(kind="line", ax=ax, color="r")
        line_xticks = [
            label for label in ax.get_xticklabels() if label.get_text() in years
        ]
        assert len(bar_xticks) == len(index)
        assert bar_xticks == line_xticks
        x_limits = ax.get_xlim()
        assert x_limits[0] <= bar_xticks[0].get_position()[0]
        assert x_limits[1] >= bar_xticks[-1].get_position()[0]
