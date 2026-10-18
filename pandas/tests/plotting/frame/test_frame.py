"""Test cases for DataFrame.plot"""

# [MODULE DOC]
# Main tests for DataFrame.plot (and the plot accessor kinds): line, area, bar,
# barh, pie, scatter, hexbin, hist, box, kde and related options such as
# stacking, log scales, secondary axes, error bars, titles, tick settings,
# x_compat parameters, nullable data, and misc regressions. Defines the main
# test class TestDataFramePlots and a few auxiliary test classes. Applies
# module-wide ignore filters for two numeric runtime warnings (division by zero
# in scalar divide, invalid value in scalar multiply). Skipped if matplotlib is
# missing (import-or-skip at module level). No other import-time side effects.

from datetime import (
    date,
    datetime,
)
import itertools
import re
import string

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.api import is_list_like

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    Series,
    bdate_range,
    date_range,
    option_context,
    plotting,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_box_return_type,
    _check_colors,
    _check_data,
    _check_grid_settings,
    _check_has_errorbars,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    _check_visible,
    get_y_axis,
)
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar multiply:RuntimeWarning"
    ),
]


class TestDataFramePlots:
    # [CLASS DOC]
    # Primary test class for DataFrame.plot. Contains one private helper for
    # comparing stacked and unstacked line y coordinates.
    @pytest.mark.slow
    def test_plot(self):
        # [FUNCTION DOC]
        # Marked slow. A business-day indexed random frame plots with the grid disabled.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        _check_plot_works(df.plot, grid=False)

    @pytest.mark.slow
    def test_plot_subplots(self):
        # [FUNCTION DOC]
        # Marked slow. Plot with subplots and default axes gives four axes in a column.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # _check_plot_works adds an ax so use default_axes=True to avoid warning
        axes = _check_plot_works(df.plot, default_axes=True, subplots=True)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    @pytest.mark.slow
    def test_plot_subplots_negative_layout(self):
        # [FUNCTION DOC]
        # Marked slow. Layout with minus one rows and two columns gives a 2x2 arrangement
        # of four axes.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            subplots=True,
            layout=(-1, 2),
        )
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_plot_subplots_use_index(self):
        # [FUNCTION DOC]
        # Marked slow. Subplots with index usage disabled give four axes in a column and
        # x tick rotation zero.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            subplots=True,
            use_index=False,
        )
        _check_ticks_props(axes, xrot=0)
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    @pytest.mark.slow
    def test_plot_invalid_arg(self):
        # [FUNCTION DOC]
        # Expected to fail (marked xfail since the matplotlib API changed in 3.6.0) and
        # slow. Passing an unknown keyword to the line plot should raise an attribute
        # error about the missing line property.
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        msg = "'Line2D' object has no property 'blarg'"
        with pytest.raises(AttributeError, match=msg):
            df.plot.line(blarg=True)

    @pytest.mark.slow
    def test_plot_tick_props(self):
        # [FUNCTION DOC]
        # Marked slow. A string-indexed frame plotted using the index has x tick
        # rotation zero.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        ax = _check_plot_works(df.plot, use_index=True)
        _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"yticks": [1, 5, 10]},
            {"xticks": [1, 5, 10]},
            {"ylim": (-100, 100), "xlim": (-100, 100)},
            {"default_axes": True, "subplots": True, "title": "blah"},
        ],
    )
    def test_plot_other_args(self, kwargs):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over keyword sets (y ticks, x ticks, limits, default
        # axes with subplots and title). The plot works for each.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        _check_plot_works(df.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_visible_ax(self):
        # [FUNCTION DOC]
        # Marked slow. Subplots with a title: the first two axes hide their x tick
        # labels and x label (axis itself visible); the last shows them with x rotation
        # zero. Redone without the helper because the helper's second plot (with an
        # axes given) would not hide labels (issue 9737).
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        # We have to redo it here because _check_plot_works does two plots,
        # once without an ax kwarg and once with an ax kwarg and the new sharex
        # behaviour does not remove the visibility of the latter axis (as ax is
        # present).  see: https://github.com/pandas-dev/pandas/issues/9737

        axes = df.plot(subplots=True, title="blah")
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes[:2]:
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible([ax.xaxis.get_label()], visible=False)
        for ax in [axes[2]]:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible([ax.xaxis.get_label()])
            _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    def test_plot_title(self):
        # [FUNCTION DOC]
        # Marked slow. Plot with a title works.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        _check_plot_works(df.plot, title="blah")

    @pytest.mark.slow
    def test_plot_multiindex(self):
        # [FUNCTION DOC]
        # Marked slow. Frame with a two-level index plots using the index; x rotation
        # zero.
        tuples = zip(string.ascii_letters[:10], range(10), strict=True)
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        ax = _check_plot_works(df.plot, use_index=True)
        _check_ticks_props(ax, xrot=0)

    @pytest.mark.slow
    def test_plot_multiindex_unicode(self):
        # [FUNCTION DOC]
        # Marked slow. Frame with unicode multi-level index and column labels and a
        # unicode title plots.
        # unicode
        index = MultiIndex.from_tuples(
            [
                ("\u03b1", 0),
                ("\u03b1", 1),
                ("\u03b2", 2),
                ("\u03b2", 3),
                ("\u03b3", 4),
                ("\u03b3", 5),
                ("\u03b4", 6),
                ("\u03b4", 7),
            ],
            names=["i0", "i1"],
        )
        columns = MultiIndex.from_tuples(
            [("bar", "\u0394"), ("bar", "\u0395")], names=["c0", "c1"]
        )
        df = DataFrame(
            np.random.default_rng(2).integers(0, 10, (8, 2)),
            columns=columns,
            index=index,
        )
        _check_plot_works(df.plot, title="\u03a3")

    @pytest.mark.slow
    @pytest.mark.parametrize("layout", [None, (-1, 1)])
    def test_plot_single_column_bar(self, layout):
        # [FUNCTION DOC]
        # Marked slow. GH 6951. Parametrized over layout none or (-1, 1). A one-column bar
        # plot with subplots gives one axes in a one-by-one layout.
        # GH 6951
        # Test with single column
        df = DataFrame({"x": np.random.default_rng(2).random(10)})
        axes = _check_plot_works(df.plot.bar, subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_plot_passed_ax(self):
        # [FUNCTION DOC]
        # Marked slow. With an axes supplied and one column, the bar plot with subplots
        # returns a list of the single supplied axes.
        # When ax is supplied and required number of axes is 1,
        # passed ax should be used:
        df = DataFrame({"x": np.random.default_rng(2).random(10)})
        _, ax = mpl.pyplot.subplots()
        axes = df.plot.bar(subplots=True, ax=ax)
        assert len(axes) == 1
        result = ax.axes
        assert result is axes[0]

    @pytest.mark.parametrize(
        "cols, x, y",
        [
            [list("ABCDE"), "A", "B"],
            [["A", "B"], "A", "B"],
            [["C", "A"], "C", "A"],
            [["A", "C"], "A", "C"],
            [["B", "C"], "B", "C"],
            [["A", "D"], "A", "D"],
            [["A", "E"], "A", "E"],
        ],
    )
    def test_nullable_int_plot(self, cols, x, y):
        # [FUNCTION DOC]
        # GH 32073. Parametrized over column subsets and x/y names. A frame with integer,
        # object (with missing), and datetime-as-integer columns (naive and UTC) plots for
        # the chosen x and y pair.
        # GH 32073
        dates = ["2008", "2009", None, "2011", "2012"]
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1, 2, 3, 4, 5],
                "C": np.array([7, 5, np.nan, 3, 2], dtype=object),
                "D": pd.to_datetime(dates, format="%Y").view("i8"),
                "E": pd.to_datetime(dates, format="%Y", utc=True).view("i8"),
            }
        )

        _check_plot_works(df[cols].plot, x=x, y=y)

    @pytest.mark.slow
    @pytest.mark.parametrize("plot", ["line", "bar", "hist", "pie"])
    def test_integer_array_plot_series(self, plot):
        # [FUNCTION DOC]
        # Marked slow. GH 25587. Parametrized over line, bar, hist and pie. A series of
        # unsigned 32-bit nullable integer values plots with each kind.
        # GH 25587
        arr = pd.array([1, 2, 3, 4], dtype="UInt32")

        s = Series(arr)
        _check_plot_works(getattr(s.plot, plot))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "plot, kwargs",
        [
            ["line", {}],
            ["bar", {}],
            ["hist", {}],
            ["pie", {"y": "y"}],
            ["scatter", {"x": "x", "y": "y"}],
            ["hexbin", {"x": "x", "y": "y"}],
        ],
    )
    def test_integer_array_plot_df(self, plot, kwargs):
        # [FUNCTION DOC]
        # Marked slow. GH 25587. Parametrized over kind and keywords (including pie with
        # y and scatter / hexbin with x and y). A frame of unsigned 32-bit nullable
        # integers plots with each kind.
        # GH 25587
        arr = pd.array([1, 2, 3, 4], dtype="UInt32")
        df = DataFrame({"x": arr, "y": arr})
        _check_plot_works(getattr(df.plot, plot), **kwargs)

    def test_nonnumeric_exclude(self):
        # [FUNCTION DOC]
        # A frame with a string column and a numeric column plots just one line.
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})
        ax = df.plot()
        assert len(ax.get_lines()) == 1  # B was plotted

    def test_implicit_label(self):
        # [FUNCTION DOC]
        # Plotting column b against column a labels the x axis with a.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        ax = df.plot(x="a", y="b")
        _check_text_labels(ax.xaxis.get_label(), "a")

    def test_donot_overwrite_index_name(self):
        # [FUNCTION DOC]
        # GH 8494. Plotting with a y column and a label keyword does not change the
        # frame's index name.
        # GH 8494
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)), columns=["a", "b"]
        )
        df.index.name = "NAME"
        df.plot(y="b", label="LABEL")
        assert df.index.name == "NAME"

    def test_plot_xy(self):
        # [FUNCTION DOC]
        # Plotting using column positions or names for x and y gives the same line data
        # as plotting the equivalent column series or frame set to that index (string
        # column labels).
        # columns.inferred_type == 'string'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        _check_data(df.plot(x=0, y=1), df.set_index("A")["B"].plot())
        _check_data(df.plot(x=0), df.set_index("A").plot())
        _check_data(df.plot(y=0), df.B.plot())
        _check_data(df.plot(x="A", y="B"), df.set_index("A").B.plot())
        _check_data(df.plot(x="A"), df.set_index("A").plot())
        _check_data(df.plot(y="B"), df.B.plot())

    def test_plot_xy_int_cols(self):
        # [FUNCTION DOC]
        # Same as above for integer column labels.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        # columns.inferred_type == 'integer'
        df.columns = np.arange(1, len(df.columns) + 1)
        _check_data(df.plot(x=1, y=2), df.set_index(1)[2].plot())
        _check_data(df.plot(x=1), df.set_index(1).plot())
        _check_data(df.plot(y=1), df[1].plot())

    def test_plot_xy_figsize_and_title(self):
        # [FUNCTION DOC]
        # Plot with x, y, title and figsize sets the title and a figure of 16 by 8 inches
        # with one axes. (A TODO comment notes missing mixed column labels test.)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        # figsize and title
        ax = df.plot(x=1, y=2, title="Test", figsize=(16, 8))
        _check_text_labels(ax.title, "Test")
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16.0, 8.0))

        # columns.inferred_type == 'mixed'
        # TODO add MultiIndex test

    @pytest.mark.parametrize(
        "input_log, expected_log", [(True, "log"), ("sym", "symlog")]
    )
    def test_logscales(self, input_log, expected_log):
        # [FUNCTION DOC]
        # Parametrized over (log argument, expected scale): true gives log, "sym" gives
        # symlog. The logy, logx and loglog keywords set the corresponding axis scale.
        df = DataFrame({"a": np.arange(100)}, index=np.arange(100))

        ax = df.plot(logy=input_log)
        _check_ax_scales(ax, yaxis=expected_log)
        assert ax.get_yscale() == expected_log

        ax = df.plot(logx=input_log)
        _check_ax_scales(ax, xaxis=expected_log)
        assert ax.get_xscale() == expected_log

        ax = df.plot(loglog=input_log)
        _check_ax_scales(ax, xaxis=expected_log, yaxis=expected_log)
        assert ax.get_xscale() == expected_log
        assert ax.get_yscale() == expected_log

    @pytest.mark.parametrize("input_param", ["logx", "logy", "loglog"])
    def test_invalid_logscale(self, input_param):
        # [FUNCTION DOC]
        # GH 24867. Parametrized over the three log keywords. An invalid string (not bool,
        # none or sym) raises a value error naming the keyword. A pie plot ignoring the
        # keyword emits a user warning.
        # GH: 24867
        df = DataFrame({"a": np.arange(100)}, index=np.arange(100))

        msg = f"keyword '{input_param}' should be bool, None, or 'sym', not 'sm'"
        with pytest.raises(ValueError, match=msg):
            df.plot(**{input_param: "sm"})

        msg = f"PiePlot ignores the '{input_param}' keyword"
        with tm.assert_produces_warning(UserWarning, match=msg):
            df.plot.pie(subplots=True, **{input_param: True})

    def test_xcompat(self):
        # [FUNCTION DOC]
        # Plotting with the x compatibility option true: x data are not period index and
        # x tick rotation is 30.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        ax = df.plot(x_compat=True)
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params(self):
        # [FUNCTION DOC]
        # Sets the global plot parameter for x axis compatibility to true (side effect:
        # the setting is not reset by this test) then plots: x data are not periods and
        # rotation is 30.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        plotting.plot_params["xaxis.compat"] = True
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params_x_compat(self):
        # [FUNCTION DOC]
        # Sets the global x_compat parameter to false (side effect: not reset), plots; x
        # data are not period index objects, but converting the x data to a period index
        # works (with the business period deprecation warning).
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        plotting.plot_params["x_compat"] = False

        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)

    def test_xcompat_plot_params_context_manager(self):
        # [FUNCTION DOC]
        # Uses the plot parameters context manager to enable x compatibility within a
        # block: x data are not periods and rotation is 30.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # useful if you're plotting a bunch together
        with plotting.plot_params.use("x_compat", True):
            ax = df.plot()
            lines = ax.get_lines()
            assert not isinstance(lines[0].get_xdata(), PeriodIndex)
            _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_period(self):
        # [FUNCTION DOC]
        # Default plotting of business-day data: x data are not periods but convertible
        # (with deprecation warning); x rotation is zero.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        msg = r"PeriodDtype\[B\] is deprecated "
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)
        _check_ticks_props(ax, xrot=0)

    def test_period_compat(self):
        # [FUNCTION DOC]
        # GH 9012. Plotting a business-day frame and then drawing a horizontal line with
        # matplotlib on the current axes does not raise.
        # GH 9012
        # period-array conversions
        df = DataFrame(
            np.random.default_rng(2).random((21, 2)),
            index=bdate_range(datetime(2000, 1, 1), datetime(2000, 1, 31)),
            columns=["a", "b"],
        )

        df.plot()
        mpl.pyplot.axhline(y=0)

    @pytest.mark.parametrize("index_dtype", [np.int64, np.float64])
    def test_unsorted_index(self, index_dtype):
        # [FUNCTION DOC]
        # Parametrized over index dtype (int64 and float64). A frame with a descending
        # index plots in index order: the line data equal the original series.
        df = DataFrame(
            {"y": np.arange(100)},
            index=Index(np.arange(99, -1, -1), dtype=index_dtype),
            dtype=np.int64,
        )
        ax = df.plot()
        lines = ax.get_lines()[0]
        rs = lines.get_xydata()
        rs = Series(rs[:, 1], rs[:, 0], dtype=np.int64, name="y")
        tm.assert_series_equal(rs, df.y, check_index_type=False)

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame({"y": [0.0, 1.0, 2.0, 3.0]}, index=[1.0, 0.0, 3.0, 2.0]),
            DataFrame(
                {"y": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]},
                index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
            ),
        ],
    )
    def test_unsorted_index_lims(self, df):
        # [FUNCTION DOC]
        # Parametrized over frames with unsorted (and missing) float index values. The x
        # limits cover the line's x data range ignoring missing.
        ax = df.plot()
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

    def test_unsorted_index_lims_x_y(self):
        # [FUNCTION DOC]
        # With unsorted x column values, the limits contain the plotted x range.
        df = DataFrame({"y": [0.0, 1.0, 2.0, 3.0], "z": [91.0, 90.0, 93.0, 92.0]})
        ax = df.plot(x="z", y="y")
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        assert xmax >= np.nanmax(lines[0].get_data()[0])

    def test_negative_log(self):
        # [FUNCTION DOC]
        # An all-negative frame with an area plot and log y or both log axes raises a
        # value error saying log y scales are not supported in area plot.
        df = -DataFrame(
            np.random.default_rng(2).random((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        msg = "Log-y scales are not supported in area plot"
        with pytest.raises(ValueError, match=msg):
            df.plot.area(logy=True)
        with pytest.raises(ValueError, match=msg):
            df.plot.area(loglog=True)

    def _compare_stacked_y_cood(self, normal_lines, stacked_lines):
        # [FUNCTION DOC]
        # HELPER: used by the stacking tests. Parameters: normal_lines and stacked_lines
        # are sequences of lines of equal length (a mismatch raises an error). Starting
        # from a zero array, adds each normal line's y data cumulatively and asserts the
        # running total equals the corresponding stacked line's y data.
        base = np.zeros(len(normal_lines[0].get_data()[1]))
        for nl, sl in zip(normal_lines, stacked_lines, strict=True):
            base += nl.get_data()[1]  # get y coordinates
            sy = sl.get_data()[1]
            tm.assert_numpy_array_equal(base, sy)

    @pytest.mark.parametrize("kind", ["line", "area"])
    @pytest.mark.parametrize("mult", [1, -1])
    def test_line_area_stacked(self, kind, mult):
        # [FUNCTION DOC]
        # Parametrized over kind (line, area) and sign multiplier (positive, negative).
        # Stacked plot lines equal the cumulative sums of the unstacked plot's lines.
        df = mult * DataFrame(
            np.random.default_rng(2).random((6, 4)), columns=["w", "x", "y", "z"]
        )

        ax1 = _check_plot_works(df.plot, kind=kind, stacked=False)
        ax2 = _check_plot_works(df.plot, kind=kind, stacked=True)
        self._compare_stacked_y_cood(ax1.lines, ax2.lines)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_line_area_stacked_sep_df(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind. A frame whose columns are each all positive or all
        # negative: stacking is done separately within the positive columns and the
        # negative columns.
        # each column has either positive or negative value
        sep_df = DataFrame(
            {
                "w": np.random.default_rng(2).random(6),
                "x": np.random.default_rng(2).random(6),
                "y": -np.random.default_rng(2).random(6),
                "z": -np.random.default_rng(2).random(6),
            }
        )
        ax1 = _check_plot_works(sep_df.plot, kind=kind, stacked=False)
        ax2 = _check_plot_works(sep_df.plot, kind=kind, stacked=True)
        self._compare_stacked_y_cood(ax1.lines[:2], ax2.lines[:2])
        self._compare_stacked_y_cood(ax1.lines[2:], ax2.lines[2:])

    def test_line_area_stacked_mixed(self):
        # [FUNCTION DOC]
        # A frame with mixed-sign columns plots unstacked fine, but stacked raises a value
        # error naming the first column with both signs.
        mixed_df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["w", "x", "y", "z"],
        )
        _check_plot_works(mixed_df.plot, stacked=False)

        msg = (
            "When stacked is True, each column must be either all positive or "
            "all negative. Column 'w' contains both positive and negative "
            "values"
        )
        with pytest.raises(ValueError, match=msg):
            mixed_df.plot(stacked=True)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_line_area_stacked_positive_idx(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind. With a strictly positive index and log x, stacked plots
        # work (avoids a matplotlib warning about x limits).
        df = DataFrame(
            np.random.default_rng(2).random((6, 4)), columns=["w", "x", "y", "z"]
        )
        # Use an index with strictly positive values, preventing
        #  matplotlib from warning about ignoring xlim
        df2 = df.set_index(df.index + 1)
        _check_plot_works(df2.plot, kind=kind, logx=True, stacked=True)

    @pytest.mark.parametrize(
        "idx", [range(4), date_range("2023-01-1", freq="D", periods=4)]
    )
    def test_line_area_nan_df(self, idx):
        # [FUNCTION DOC]
        # Parametrized over range and date indexes. A frame with missing values plots; the
        # masked y data of each line hide the missing positions and the rest of the
        # values match.
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        df = DataFrame({"a": values1, "b": values2}, index=idx)

        ax = _check_plot_works(df.plot)
        masked1 = ax.lines[0].get_ydata()
        masked2 = ax.lines[1].get_ydata()
        # remove nan for comparison purpose

        exp = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked1.data, 2), exp)

        exp = np.array([3, 2, 1], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked2.data, 1), exp)
        tm.assert_numpy_array_equal(masked1.mask, np.array([False, False, True, False]))
        tm.assert_numpy_array_equal(masked2.mask, np.array([False, True, False, False]))

    @pytest.mark.parametrize(
        "idx", [range(4), date_range("2023-01-1", freq="D", periods=4)]
    )
    def test_line_area_nan_df_stacked(self, idx):
        # [FUNCTION DOC]
        # Parametrized over index type. Stacked plot treats missing as zero: first line
        # equals the first column filled with zeros and second line equals the sum.
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        df = DataFrame({"a": values1, "b": values2}, index=idx)

        expected1 = np.array([1, 2, 0, 3], dtype=np.float64)
        expected2 = np.array([3, 0, 2, 1], dtype=np.float64)

        ax = _check_plot_works(df.plot, stacked=True)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

    @pytest.mark.parametrize(
        "idx", [range(4), date_range("2023-01-1", freq="D", periods=4)]
    )
    @pytest.mark.parametrize("kwargs", [{}, {"stacked": False}])
    def test_line_area_nan_df_stacked_area(self, idx, kwargs):
        # [FUNCTION DOC]
        # Parametrized over index type and keywords (default or stacked false). The area
        # plot's first line equals first column with zero fill; for stacked default the
        # second line is the cumulative sum, for unstacked the second column filled with
        # zero; then an explicit unstacked area plot is checked.
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        df = DataFrame({"a": values1, "b": values2}, index=idx)

        expected1 = np.array([1, 2, 0, 3], dtype=np.float64)
        expected2 = np.array([3, 0, 2, 1], dtype=np.float64)

        ax = _check_plot_works(df.plot.area, **kwargs)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        if kwargs:
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)
        else:
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

        ax = _check_plot_works(df.plot.area, stacked=False)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)

    @pytest.mark.parametrize("kwargs", [{}, {"secondary_y": True}])
    def test_line_lim(self, kwargs):
        # [FUNCTION DOC]
        # Parametrized over default and secondary y. The x limits cover the first and last
        # x values of the first line.
        df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=["x", "y", "z"])
        ax = df.plot(**kwargs)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]

    def test_line_lim_subplots(self):
        # [FUNCTION DOC]
        # Subplots with secondary y: three axes in a column, each having a left twin and
        # no right twin and x limits covering the first line's data.
        df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=["x", "y", "z"])
        axes = df.plot(secondary_y=True, subplots=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes:
            assert hasattr(ax, "left_ax")
            assert not hasattr(ax, "right_ax")
            xmin, xmax = ax.get_xlim()
            lines = ax.get_lines()
            assert xmin <= lines[0].get_data()[0][0]
            assert xmax >= lines[0].get_data()[0][-1]

    @pytest.mark.xfail(
        strict=False,
        reason="2020-12-01 this has been failing periodically on the "
        "ymin==0 assertion for a week or so.",
    )
    @pytest.mark.parametrize("stacked", [True, False])
    def test_area_lim(self, stacked):
        # [FUNCTION DOC]
        # Parametrized over stacked flag; non-strict expected failure (marked as flaky).
        # Area plot of positive data has x limits covering the data and lower y limit
        # zero; area plot of negative data has upper y limit zero.
        df = DataFrame(
            np.random.default_rng(2).random((6, 4)), columns=["x", "y", "z", "four"]
        )

        neg_df = -df

        ax = _check_plot_works(df.plot.area, stacked=stacked)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data()[0][0]
        assert xmax >= lines[0].get_data()[0][-1]
        assert ymin == 0

        ax = _check_plot_works(neg_df.plot.area, stacked=stacked)
        ymin, ymax = ax.get_ylim()
        assert ymax == 0

    def test_area_sharey_dont_overwrite(self):
        # [FUNCTION DOC]
        # GH37942. Two area plots drawn on two axes with shared y keep the y axes
        # joined.
        # GH37942
        df = DataFrame(np.random.default_rng(2).random((4, 2)), columns=["x", "y"])
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)

        df.plot(ax=ax1, kind="area")
        df.plot(ax=ax2, kind="area")

        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    @pytest.mark.parametrize("stacked", [True, False])
    def test_bar_linewidth(self, stacked):
        # [FUNCTION DOC]
        # Parametrized over stacked flag. Bar plot with line width 2 gives every patch a
        # line width of 2.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax = df.plot.bar(stacked=stacked, linewidth=2)
        for r in ax.patches:
            assert r.get_linewidth() == 2

    def test_bar_linewidth_subplots(self):
        # [FUNCTION DOC]
        # Bar plot with line width 2 and subplots: five axes in a column layout.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # subplots
        axes = df.plot.bar(linewidth=2, subplots=True)
        _check_axes_shape(axes, axes_num=5, layout=(5, 1))
        for ax in axes:
            for r in ax.patches:
                assert r.get_linewidth() == 2

    @pytest.mark.parametrize(
        "meth, dim", [("bar", "get_width"), ("barh", "get_height")]
    )
    @pytest.mark.parametrize("stacked", [True, False])
    def test_bar_barwidth(self, meth, dim, stacked):
        # [FUNCTION DOC]
        # Parametrized over bar/barh and stacked flag. With width 0.9, each bar's
        # thickness (width or height by kind) equals the width divided by the number of
        # columns when unstacked, or the full width when stacked.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        width = 0.9

        ax = getattr(df.plot, meth)(stacked=stacked, width=width)
        for r in ax.patches:
            if not stacked:
                assert getattr(r, dim)() == width / len(df.columns)
            else:
                assert getattr(r, dim)() == width

    @pytest.mark.parametrize(
        "meth, dim", [("bar", "get_width"), ("barh", "get_height")]
    )
    def test_barh_barwidth_subplots(self, meth, dim):
        # [FUNCTION DOC]
        # Parametrized over bar/barh. With width 0.9 and subplots, every patch has the
        # full width.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        width = 0.9

        axes = getattr(df.plot, meth)(width=width, subplots=True)
        for ax in axes:
            for r in ax.patches:
                assert getattr(r, dim)() == width

    def test_bar_bottom_left_bottom(self):
        # [FUNCTION DOC]
        # Bar plot with a scalar bottom for unstacked bars starts all 25 bars at y equal to
        # the bottom; with a list of bottoms and stacked, the first layer of bars starts at
        # those values.
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.bar(stacked=False, bottom=1)
        result = [p.get_y() for p in ax.patches]
        assert result == [1] * 25

        ax = df.plot.bar(stacked=True, bottom=[-1, -2, -3, -4, -5])
        result = [p.get_y() for p in ax.patches[:5]]
        assert result == [-1, -2, -3, -4, -5]

    def test_bar_bottom_left_left(self):
        # [FUNCTION DOC]
        # Horizontal bar plot with left offset as an array, unstacked: all bars start at 1;
        # stacked with a list: first layer starts at the list values.
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.barh(stacked=False, left=np.array([1, 1, 1, 1, 1]))
        result = [p.get_x() for p in ax.patches]
        assert result == [1] * 25

        ax = df.plot.barh(stacked=True, left=[1, 2, 3, 4, 5])
        result = [p.get_x() for p in ax.patches[:5]]
        assert result == [1, 2, 3, 4, 5]

    def test_bar_bottom_left_subplots(self):
        # [FUNCTION DOC]
        # Bar plot with subplots and bottom of minus one: every patch starts at minus one;
        # horizontal subplots with left offsets start at 1.
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        axes = df.plot.bar(subplots=True, bottom=-1)
        for ax in axes:
            result = [p.get_y() for p in ax.patches]
            assert result == [-1] * 5

        axes = df.plot.barh(subplots=True, left=np.array([1, 1, 1, 1, 1]))
        for ax in axes:
            result = [p.get_x() for p in ax.patches]
            assert result == [1] * 5

    def test_bar_nan(self):
        # [FUNCTION DOC]
        # Bar heights for a frame with a missing value are the values with zero in place
        # of the missing value, column by column.
        df = DataFrame({"A": [10, np.nan, 20], "B": [5, 10, 20], "C": [1, 2, 3]})
        ax = df.plot.bar()
        expected = [10, 0, 20, 5, 10, 20, 1, 2, 3]
        result = [p.get_height() for p in ax.patches]
        assert result == expected

    def test_bar_nan_stacked(self):
        # [FUNCTION DOC]
        # As above for stacked bars; additionally the bar bottoms equal the cumulative
        # sums of the lower layers (missing counted as zero).
        df = DataFrame({"A": [10, np.nan, 20], "B": [5, 10, 20], "C": [1, 2, 3]})
        ax = df.plot.bar(stacked=True)
        expected = [10, 0, 20, 5, 10, 20, 1, 2, 3]
        result = [p.get_height() for p in ax.patches]
        assert result == expected

        result = [p.get_y() for p in ax.patches]
        expected = [0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 15.0, 10.0, 40.0]
        assert result == expected

    def test_bar_stacked_label_position_with_zero_height(self):
        # [FUNCTION DOC]
        # GH 59429. Stacked bars with zero-height segments: the labels added for the top
        # container are positioned at the total height of each stack. Closes all figures.
        # GH 59429
        df = DataFrame({"A": [3, 0, 1], "B": [0, 2, 4], "C": [5, 0, 2]})
        ax = df.plot.bar(stacked=True)
        ax.bar_label(ax.containers[-1])
        expected = [8.0, 2.0, 7.0]
        result = [text.xy[1] for text in ax.texts]
        tm.assert_almost_equal(result, expected)
        plt.close("all")

    @pytest.mark.parametrize("idx", [Index, pd.CategoricalIndex])
    def test_bar_categorical(self, idx):
        # [FUNCTION DOC]
        # GH 13019. Parametrized over plain and categorical index types for rows and
        # columns. Grouped bar plot has ticks at 0..5, x limits -0.5 to 5.5 and expected
        # first and last bar left edges; stacked version has limits and edges for stacked
        # bars.
        # KNOWN ISSUE: for the stacked plot the tick locations are not recomputed, so the
        # tick assertion repeats the check on the earlier plot's ticks.
        # GH 13019
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 5)),
            index=idx(list("ABCDEF")),
            columns=idx(list("abcde")),
        )

        ax = df.plot.bar()
        ticks = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
        assert ax.get_xlim() == (-0.5, 5.5)
        # check left-edge of bars
        assert ax.patches[0].get_x() == -0.25
        assert ax.patches[-1].get_x() == 5.15

        ax = df.plot.bar(stacked=True)
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
        assert ax.get_xlim() == (-0.5, 5.5)
        assert ax.patches[0].get_x() == -0.25
        assert ax.patches[-1].get_x() == 4.75

    @pytest.mark.parametrize("x, y", [("x", "y"), (1, 2)])
    def test_plot_scatter(self, x, y):
        # [FUNCTION DOC]
        # Parametrized over (x, y) as names or positions. Scatter plot works.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )

        _check_plot_works(df.plot.scatter, x=x, y=y)

    def test_plot_scatter_error(self):
        # [FUNCTION DOC]
        # Scatter without y or without x raises a type error that the corresponding
        # required argument is missing.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        msg = re.escape("scatter() missing 1 required positional argument: 'y'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(x="x")
        msg = re.escape("scatter() missing 1 required positional argument: 'x'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(y="y")

    def test_plot_scatter_shape(self):
        # [FUNCTION DOC]
        # GH 6951. Scatter plot with subplots gives one axes in a one-by-one layout.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        # GH 6951
        axes = df.plot(x="x", y="y", kind="scatter", subplots=True)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_scatter_on_datetime_time_data(self):
        # [FUNCTION DOC]
        # A scatter plot with a column of time-of-day values on the x axis works (a
        # converter exists in the scatter plot class).
        # datetime.time type is now supported in scatter, since a converter
        # is implemented in ScatterPlot
        df = DataFrame(np.random.default_rng(2).standard_normal(10), columns=["a"])
        df["dtime"] = date_range(start="2014-01-01", freq="h", periods=10).time
        df.plot(kind="scatter", x="dtime", y="a")

    def test_scatter_line_xticks(self):
        # [FUNCTION DOC]
        # GH#61005. A scatter plot and a line plot of the same datetime x data on shared
        # axes have the same first and last x tick.
        # GH#61005
        df = DataFrame(
            [(datetime(year=2025, month=1, day=1, hour=n), n) for n in range(3)],
            columns=["datetime", "y"],
        )
        fig, ax = plt.subplots(2, sharex=True)
        df.plot.scatter(x="datetime", y="y", ax=ax[0])
        scatter_xticks = ax[0].get_xticks()
        df.plot(x="datetime", y="y", ax=ax[1])
        line_xticks = ax[1].get_xticks()
        assert scatter_xticks[0] == line_xticks[0]
        assert scatter_xticks[-1] == line_xticks[-1]

    @pytest.mark.parametrize("x, y", [("dates", "vals"), (0, 1)])
    def test_scatterplot_datetime_data(self, x, y):
        # [FUNCTION DOC]
        # GH 30391. Parametrized over (names, positions). Scatter with a datetime x
        # column works.
        # GH 30391
        dates = date_range(start=date(2019, 1, 1), periods=12, freq="W")
        vals = np.random.default_rng(2).normal(0, 1, len(dates))
        df = DataFrame({"dates": dates, "vals": vals})

        _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    @pytest.mark.parametrize("x, y", [("a", "b"), (0, 1)])
    @pytest.mark.parametrize("b_col", [[2, 3, 4], ["a", "b", "c"]])
    def test_scatterplot_object_data(self, b_col, x, y, infer_string):
        # [FUNCTION DOC]
        # GH 18755. Parametrized over future string inference on/off (the on case needs
        # pyarrow), x/y by name or position, and an integer or string second column.
        # Scatter plot with object data works inside the option context.
        # GH 18755
        with option_context("future.infer_string", infer_string):
            df = DataFrame({"a": ["A", "B", "C"], "b": b_col})

            _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "categories",
        (["setosa", "versicolor", "virginica"], ["versicolor", "virginica", "setosa"]),
    )
    def test_scatterplot_color_by_categorical(self, ordered, categories):
        # [FUNCTION DOC]
        # Parametrized over ordered flag and category order. Scatter colored by a
        # categorical column creates a single collection with a colorbar having ticks at
        # 0.5, 1.5, 2.5, boundaries 0 to 3 and y tick labels in the category order.
        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        df["species"] = pd.Categorical(
            ["setosa", "setosa", "virginica", "virginica", "versicolor"],
            ordered=ordered,
            categories=categories,
        )
        ax = df.plot.scatter(x=0, y=1, c="species")
        (colorbar_collection,) = ax.collections
        colorbar = colorbar_collection.colorbar

        expected_ticks = np.array([0.5, 1.5, 2.5])
        result_ticks = colorbar.get_ticks()
        tm.assert_numpy_array_equal(result_ticks, expected_ticks)

        expected_boundaries = np.array([0.0, 1.0, 2.0, 3.0])
        result_boundaries = colorbar._boundaries
        tm.assert_numpy_array_equal(result_boundaries, expected_boundaries)

        expected_yticklabels = categories
        result_yticklabels = [i.get_text() for i in colorbar.ax.get_ymajorticklabels()]
        assert all(
            i == j
            for i, j in zip(result_yticklabels, expected_yticklabels, strict=True)
        )

    @pytest.mark.parametrize("x, y", [("x", "y"), ("y", "x"), ("y", "y")])
    def test_plot_scatter_with_categorical_data(self, x, y):
        # [FUNCTION DOC]
        # Parametrized over (x, y) combinations including a categorical column. Scatter
        # with categorical data works.
        # after fixing GH 18755, should be able to plot categorical data
        df = DataFrame({"x": [1, 2, 3, 4], "y": pd.Categorical(["a", "b", "a", "c"])})

        _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize("x, y, c", [("x", "y", "z"), (0, 1, 2)])
    def test_plot_scatter_with_c(self, x, y, c):
        # [FUNCTION DOC]
        # Parametrized over names or positions. Scatter with a color column uses the
        # Greys colormap by default and the colorbar label is the color column name z.
        df = DataFrame(
            np.random.default_rng(2).integers(low=0, high=100, size=(6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )

        ax = df.plot.scatter(x=x, y=y, c=c)
        # default to Greys
        assert ax.collections[0].cmap.name == "Greys"

        assert ax.collections[0].colorbar.ax.get_ylabel() == "z"

    def test_plot_scatter_with_c_props(self):
        # [FUNCTION DOC]
        # Scatter with a colormap name uses that colormap; turning off the colorbar leaves
        # no colorbar; a solid color name gives no colorbar and red face colors.
        df = DataFrame(
            np.random.default_rng(2).integers(low=0, high=100, size=(6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        cm = "cubehelix"
        ax = df.plot.scatter(x="x", y="y", c="z", colormap=cm)
        assert ax.collections[0].cmap.name == cm

        # verify turning off colorbar works
        ax = df.plot.scatter(x="x", y="y", c="z", colorbar=False)
        assert ax.collections[0].colorbar is None

        # verify that we can still plot a solid color
        ax = df.plot.scatter(x=0, y=1, c="red")
        assert ax.collections[0].colorbar is None
        _check_colors(ax.collections, facecolors=["r"])

    def test_plot_scatter_with_c_array(self):
        # [FUNCTION DOC]
        # Passing an RGBA array as colors is passed through to matplotlib: face colors
        # equal the array exactly (issue 8852); passing a float array with a colormap
        # works (colors not checked).
        # Ensure that we can pass an np.array straight through to matplotlib,
        # this functionality was accidentally removed previously.
        # See https://github.com/pandas-dev/pandas/issues/8852 for bug report
        #
        # Exercise colormap path and non-colormap path as they are independent
        #
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        red_rgba = [1.0, 0.0, 0.0, 1.0]
        green_rgba = [0.0, 1.0, 0.0, 1.0]
        rgba_array = np.array([red_rgba, green_rgba])
        ax = df.plot.scatter(x="A", y="B", c=rgba_array)
        # expect the face colors of the points in the non-colormap path to be
        # identical to the values we supplied, normally we'd be on shaky ground
        # comparing floats for equality but here we expect them to be
        # identical.
        tm.assert_numpy_array_equal(ax.collections[0].get_facecolor(), rgba_array)
        # we don't test the colors of the faces in this next plot because they
        # are dependent on the spring colormap, which may change its colors
        # later.
        float_array = np.array([0.0, 1.0])
        df.plot.scatter(x="A", y="B", c=float_array, cmap="spring")

    def test_plot_scatter_with_s(self):
        # [FUNCTION DOC]
        # GH 32904. The size column is used as marker sizes.
        # this refers to GH 32904
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)) * 100, columns=["a", "b", "c"]
        )

        ax = df.plot.scatter(x="a", y="b", s="c")
        tm.assert_numpy_array_equal(df["c"].values, right=ax.collections[0].get_sizes())

    def test_plot_scatter_with_norm(self):
        # [FUNCTION DOC]
        # GH 45809. A passed norm object is the one used by the collection.
        # added while fixing GH 45809
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)) * 100, columns=["a", "b", "c"]
        )
        norm = mpl.colors.LogNorm()
        ax = df.plot.scatter(x="a", y="b", c="c", norm=norm)
        assert ax.collections[0].norm is norm

    def test_plot_scatter_without_norm(self):
        # [FUNCTION DOC]
        # GH 45809. Default normalization of the color data matches a default linear
        # normalization between the column's minimum and maximum, for every value.
        # added while fixing GH 45809
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)) * 100, columns=["a", "b", "c"]
        )
        ax = df.plot.scatter(x="a", y="b", c="c")
        plot_norm = ax.collections[0].norm
        color_min_max = (df.c.min(), df.c.max())
        default_norm = mpl.colors.Normalize(*color_min_max)
        for value in df.c:
            assert plot_norm(value) == default_norm(value)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"legend": False},
            {"default_axes": True, "subplots": True},
            {"stacked": True},
        ],
    )
    def test_plot_bar(self, kwargs):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over keyword sets (default, no legend, default axes with
        # subplots, stacked). The bar plot works.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )

        _check_plot_works(df.plot.bar, **kwargs)

    @pytest.mark.slow
    def test_plot_bar_int_col(self):
        # [FUNCTION DOC]
        # Marked slow. Bar plot of a frame with integer column labels works.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 15)),
            index=list(string.ascii_letters[:10]),
            columns=range(15),
        )
        _check_plot_works(df.plot.bar)

    @pytest.mark.slow
    def test_plot_bar_ticks(self):
        # [FUNCTION DOC]
        # Marked slow. Bar plot x tick rotation defaults to 90; with rotation 35 and font
        # size 10, the rotation and the label sizes are applied.
        df = DataFrame({"a": [0, 1], "b": [1, 0]})
        ax = _check_plot_works(df.plot.bar)
        _check_ticks_props(ax, xrot=90)

        ax = df.plot.bar(rot=35, fontsize=10)
        _check_ticks_props(ax, xrot=35, xlabelsize=10, ylabelsize=10)

    @pytest.mark.slow
    def test_plot_barh_ticks(self):
        # [FUNCTION DOC]
        # Marked slow. Horizontal bar plot y tick rotation defaults to 0; with rotation 55
        # and font size 11, they are applied.
        df = DataFrame({"a": [0, 1], "b": [1, 0]})
        ax = _check_plot_works(df.plot.barh)
        _check_ticks_props(ax, yrot=0)

        ax = df.plot.barh(rot=55, fontsize=11)
        _check_ticks_props(ax, yrot=55, ylabelsize=11, xlabelsize=11)

    def test_boxplot(self, hist_df):
        # [FUNCTION DOC]
        # Takes the shared hist_df fixture. Box plot of the numeric columns: x tick labels are the
        # numeric column names, tick positions are 1..n and each column draws 7 lines.
        df = hist_df
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]

        ax = _check_plot_works(df.plot.box)
        _check_text_labels(ax.get_xticklabels(), labels)
        tm.assert_numpy_array_equal(
            ax.xaxis.get_ticklocs(), np.arange(1, len(numeric_cols) + 1)
        )
        assert len(ax.lines) == 7 * len(numeric_cols)

    def test_boxplot_series(self, hist_df):
        # [FUNCTION DOC]
        # Box plot of a series with rotation 40: x rotation 40, y rotation 0; a plain
        # series box plot works.
        df = hist_df
        series = df["height"]
        axes = series.plot.box(rot=40)
        _check_ticks_props(axes, xrot=40, yrot=0)

        _check_plot_works(series.plot.box)

    def test_boxplot_series_positions(self, hist_df):
        # [FUNCTION DOC]
        # Box plot with explicit positions: tick positions equal the positions, labels
        # are the numeric column names and each column draws 7 lines.
        # KNOWN ISSUE: the name says series, but the test plots the whole frame.
        df = hist_df
        positions = np.array([1, 6, 7])
        ax = df.plot.box(positions=positions)
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]
        _check_text_labels(ax.get_xticklabels(), labels)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), positions)
        assert len(ax.lines) == 7 * len(numeric_cols)

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    @pytest.mark.xfail(
        Version(mpl.__version__) >= Version("3.10"),
        reason="Fails starting with matplotlib 3.10",
    )
    def test_boxplot_vertical(self, hist_df):
        # [FUNCTION DOC]
        # Ignores a tick label warning; expected to fail with matplotlib 3.10 or later.
        # Horizontal box plot (orientation keyword depends on version) with rotation 50
        # and font size 8: y labels rotated, x unrotated, labels are the column names,
        # 7 lines each.
        df = hist_df
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]

        # if horizontal, yticklabels are rotated
        kwargs = (
            {"vert": False}
            if Version(mpl.__version__) < Version("3.10")
            else {"orientation": "horizontal"}
        )
        ax = df.plot.box(rot=50, fontsize=8, **kwargs)
        _check_ticks_props(ax, xrot=0, yrot=50, ylabelsize=8)
        _check_text_labels(ax.get_yticklabels(), labels)
        assert len(ax.lines) == 7 * len(numeric_cols)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.xfail(
        Version(mpl.__version__) >= Version("3.10"),
        reason="Fails starting with matplotlib version 3.10",
    )
    def test_boxplot_vertical_subplots(self, hist_df):
        # [FUNCTION DOC]
        # Expected to fail with matplotlib 3.10 or later. Horizontal box plot with
        # subplots and log x gives three axes in a row with log x scale, one label each and
        # 7 lines per axes.
        df = hist_df
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]
        kwargs = (
            {"vert": False}
            if Version(mpl.__version__) < Version("3.10")
            else {"orientation": "horizontal"}
        )
        axes = _check_plot_works(
            df.plot.box, default_axes=True, subplots=True, logx=True, **kwargs
        )
        _check_axes_shape(axes, axes_num=3, layout=(1, 3))
        _check_ax_scales(axes, xaxis="log")
        for ax, label in zip(axes, labels, strict=True):
            _check_text_labels(ax.get_yticklabels(), [label])
            assert len(ax.lines) == 7

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    @pytest.mark.xfail(
        Version(mpl.__version__) >= Version("3.10"),
        reason="Fails starting with matplotlib 3.10",
    )
    def test_boxplot_vertical_positions(self, hist_df):
        # [FUNCTION DOC]
        # Expected to fail with matplotlib 3.10 or later. Horizontal box plot with
        # positions: y tick locations equal the positions and the labels are the column
        # names.
        df = hist_df
        numeric_cols = df._get_numeric_data().columns
        labels = [pprint_thing(c) for c in numeric_cols]
        positions = np.array([3, 2, 8])
        kwargs = (
            {"vert": False}
            if Version(mpl.__version__) < Version("3.10")
            else {"orientation": "horizontal"}
        )
        ax = df.plot.box(positions=positions, **kwargs)
        _check_text_labels(ax.get_yticklabels(), labels)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), positions)
        assert len(ax.lines) == 7 * len(numeric_cols)

    def test_boxplot_return_type_invalid(self):
        # [FUNCTION DOC]
        # An invalid box plot return type raises a value error listing the allowed values.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        msg = "return_type must be {None, 'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.plot.box(return_type="not_a_type")

    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_boxplot_return_type_invalid_type(self, return_type):
        # [FUNCTION DOC]
        # Parametrized over the valid return types (dict, axes, both). The box plot result
        # has the matching type.
        # KNOWN ISSUE: the name suggests invalid types, but only valid return types are
        # tested.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        result = df.plot.box(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_kde_df(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot: legend labels are the column names, x rotation 0.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        ax = _check_plot_works(df.plot, kind="kde")
        expected = [pprint_thing(c) for c in df.columns]
        _check_legend_labels(ax, labels=expected)
        _check_ticks_props(ax, xrot=0)

    def test_kde_df_rot(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot with rotation 20 and font size 5 applies both.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        ax = df.plot(kind="kde", rot=20, fontsize=5)
        _check_ticks_props(ax, xrot=20, xlabelsize=5, ylabelsize=5)

    def test_kde_df_subplots(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot with subplots gives four axes in a column.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            kind="kde",
            subplots=True,
        )
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    def test_kde_df_logy(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot with log y and subplots sets a log y scale.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        axes = df.plot(kind="kde", logy=True, subplots=True)
        _check_ax_scales(axes, yaxis="log")

    def test_kde_missing_vals(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot of data containing a missing value works.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).uniform(size=(100, 4)))
        df.loc[0, 0] = np.nan
        _check_plot_works(df.plot, kind="kde")

    def test_hist_df(self):
        # [FUNCTION DOC]
        # Histogram of a frame: legend labels are the column names; with subplots and log
        # y: four axes in a column with log y scales.
        df = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))

        ax = _check_plot_works(df.plot.hist)
        expected = [pprint_thing(c) for c in df.columns]
        _check_legend_labels(ax, labels=expected)

        axes = _check_plot_works(
            df.plot.hist,
            default_axes=True,
            subplots=True,
            logy=True,
        )
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))
        _check_ax_scales(axes, yaxis="log")

    def test_hist_df_series(self):
        # [FUNCTION DOC]
        # Series histogram with rotation 40: x rotation 40, y rotation 0.
        series = Series(np.random.default_rng(2).random(10))
        axes = series.plot.hist(rot=40)
        _check_ticks_props(axes, xrot=40, yrot=0)

    def test_hist_df_series_cumulative_density(self):
        # [FUNCTION DOC]
        # Cumulative density histogram with four bins: the last rectangle's height is 1.
        series = Series(np.random.default_rng(2).random(10))
        ax = series.plot.hist(cumulative=True, bins=4, density=True)
        # height of last bin (index 5) must be 1.0
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    def test_hist_df_series_cumulative(self):
        # [FUNCTION DOC]
        # Cumulative histogram with four bins: the second to last rectangle's height is
        # the number of data points (10).
        series = Series(np.random.default_rng(2).random(10))
        ax = series.plot.hist(cumulative=True, bins=4)
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]

        tm.assert_almost_equal(rects[-2].get_height(), 10.0)

    def test_hist_df_orientation(self):
        # [FUNCTION DOC]
        # Horizontal histogram with rotation 50 and font size 8: x rotation 0, y rotation
        # 50, y label size 8.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # if horizontal, yticklabels are rotated
        axes = df.plot.hist(rot=50, fontsize=8, orientation="horizontal")
        _check_ticks_props(axes, xrot=0, yrot=50, ylabelsize=8)

    @pytest.mark.parametrize("weight_shape", [(100,), (100, 2)])
    def test_hist_weights(self, weight_shape):
        # [FUNCTION DOC]
        # GH 33173. Parametrized over weights of shape (100,) and (100, 2). Weighted
        # histogram heights equal 0.1 times the unweighted heights.
        # GH 33173
        weights = 0.1 * np.ones(shape=weight_shape)
        df = DataFrame(
            dict(
                zip(
                    ["A", "B"],
                    np.random.default_rng(2).standard_normal((2, 100)),
                    strict=True,
                )
            )
        )

        ax1 = _check_plot_works(df.plot, kind="hist", weights=weights)
        ax2 = _check_plot_works(df.plot, kind="hist")

        patch_height_with_weights = [patch.get_height() for patch in ax1.patches]

        # original heights with no weights, and we manually multiply with example
        # weights, so after multiplication, they should be almost same
        expected_patch_height = [0.1 * patch.get_height() for patch in ax2.patches]

        tm.assert_almost_equal(patch_height_with_weights, expected_patch_height)

    def _check_box_coord(
        self,
        patches,
        expected_y=None,
        expected_h=None,
        expected_x=None,
        expected_w=None,
    ):
        # [FUNCTION DOC]
        # HELPER: asserts coordinates of rectangle patches. Parameters: patches is a
        # sequence of rectangles; expected_y, expected_h, expected_x, expected_w are
        # optional expected arrays of bottom coordinate, height, left coordinate and
        # width (checked only when given, without checking dtype).
        result_y = np.array([p.get_y() for p in patches])
        result_height = np.array([p.get_height() for p in patches])
        result_x = np.array([p.get_x() for p in patches])
        result_width = np.array([p.get_width() for p in patches])
        # dtype is depending on above values, no need to check

        if expected_y is not None:
            tm.assert_numpy_array_equal(result_y, expected_y, check_dtype=False)
        if expected_h is not None:
            tm.assert_numpy_array_equal(result_height, expected_h, check_dtype=False)
        if expected_x is not None:
            tm.assert_numpy_array_equal(result_x, expected_x, check_dtype=False)
        if expected_w is not None:
            tm.assert_numpy_array_equal(result_width, expected_w, check_dtype=False)

    @pytest.mark.parametrize(
        "data",
        [
            {
                "A": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([10, 9, 8, 7, 6])),
                "B": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([8, 8, 8, 8, 8])),
                "C": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])),
            },
            {
                "A": np.repeat(
                    np.array([np.nan, 1, 2, 3, 4, 5]), np.array([3, 10, 9, 8, 7, 6])
                ),
                "B": np.repeat(
                    np.array([1, np.nan, 2, 3, 4, 5]), np.array([8, 3, 8, 8, 8, 8])
                ),
                "C": np.repeat(
                    np.array([1, 2, 3, np.nan, 4, 5]), np.array([6, 7, 8, 3, 9, 10])
                ),
            },
        ],
    )
    def test_hist_df_coord(self, data):
        # [FUNCTION DOC]
        # Parametrized over data without and with missing values. Checks rectangle
        # coordinates of five-bin histograms: unstacked (all bars start at zero with
        # the counts as heights), stacked (each layer starts at the cumulative sum of
        # the earlier layers), stacked with subplots (each axes starts at zero), and the
        # horizontal variants where x and width are checked instead.
        df = DataFrame(data)

        ax = df.plot.hist(bins=5)
        self._check_box_coord(
            ax.patches[:5],
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([10, 9, 8, 7, 6]),
        )
        self._check_box_coord(
            ax.patches[5:10],
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([8, 8, 8, 8, 8]),
        )
        self._check_box_coord(
            ax.patches[10:],
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([6, 7, 8, 9, 10]),
        )

        ax = df.plot.hist(bins=5, stacked=True)
        self._check_box_coord(
            ax.patches[:5],
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([10, 9, 8, 7, 6]),
        )
        self._check_box_coord(
            ax.patches[5:10],
            expected_y=np.array([10, 9, 8, 7, 6]),
            expected_h=np.array([8, 8, 8, 8, 8]),
        )
        self._check_box_coord(
            ax.patches[10:],
            expected_y=np.array([18, 17, 16, 15, 14]),
            expected_h=np.array([6, 7, 8, 9, 10]),
        )

        axes = df.plot.hist(bins=5, stacked=True, subplots=True)
        self._check_box_coord(
            axes[0].patches,
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([10, 9, 8, 7, 6]),
        )
        self._check_box_coord(
            axes[1].patches,
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([8, 8, 8, 8, 8]),
        )
        self._check_box_coord(
            axes[2].patches,
            expected_y=np.array([0, 0, 0, 0, 0]),
            expected_h=np.array([6, 7, 8, 9, 10]),
        )

        # horizontal
        ax = df.plot.hist(bins=5, orientation="horizontal")
        self._check_box_coord(
            ax.patches[:5],
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([10, 9, 8, 7, 6]),
        )
        self._check_box_coord(
            ax.patches[5:10],
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([8, 8, 8, 8, 8]),
        )
        self._check_box_coord(
            ax.patches[10:],
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([6, 7, 8, 9, 10]),
        )

        ax = df.plot.hist(bins=5, stacked=True, orientation="horizontal")
        self._check_box_coord(
            ax.patches[:5],
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([10, 9, 8, 7, 6]),
        )
        self._check_box_coord(
            ax.patches[5:10],
            expected_x=np.array([10, 9, 8, 7, 6]),
            expected_w=np.array([8, 8, 8, 8, 8]),
        )
        self._check_box_coord(
            ax.patches[10:],
            expected_x=np.array([18, 17, 16, 15, 14]),
            expected_w=np.array([6, 7, 8, 9, 10]),
        )

        axes = df.plot.hist(
            bins=5, stacked=True, subplots=True, orientation="horizontal"
        )
        self._check_box_coord(
            axes[0].patches,
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([10, 9, 8, 7, 6]),
        )
        self._check_box_coord(
            axes[1].patches,
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([8, 8, 8, 8, 8]),
        )
        self._check_box_coord(
            axes[2].patches,
            expected_x=np.array([0, 0, 0, 0, 0]),
            expected_w=np.array([6, 7, 8, 9, 10]),
        )

    def test_plot_int_columns(self):
        # [FUNCTION DOC]
        # A cumulative-sum frame with integer column labels plots with a legend.
        df = DataFrame(np.random.default_rng(2).standard_normal((100, 4))).cumsum()
        _check_plot_works(df.plot, legend=True)

    @pytest.mark.parametrize(
        "markers",
        [
            {0: "^", 1: "+", 2: "o"},
            {0: "^", 1: "+"},
            ["^", "+", "o"],
            ["^", "+"],
        ],
    )
    def test_style_by_column(self, markers):
        # [FUNCTION DOC]
        # Parametrized over marker styles given as dict by column (full or partial) or list
        # (full or partial). Starts with a fresh figure; each plotted line (up to the
        # number of styles) has the marker given for its position or column.
        fig = plt.gcf()
        fig.clf()
        fig.add_subplot(111)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
        ax = df.plot(style=markers)
        for idx, line in enumerate(ax.get_lines()[: len(markers)]):
            assert line.get_marker() == markers[idx]

    def test_line_label_none(self):
        # [FUNCTION DOC]
        # A series plot has no legend by default; with the legend option the legend's
        # only text is empty because the series has no name.
        s = Series([1, 2])
        ax = s.plot()
        assert ax.get_legend() is None

        ax = s.plot(legend=True)
        assert ax.get_legend().get_texts()[0].get_text() == ""

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd_plot_box(self, props, expected):
        # [FUNCTION DOC]
        # GH 30346. Parametrized over property keyword (box, whisker, cap, median
        # properties) and the part. The color given through the keyword applies to the
        # first element of the chosen part in the box plot accessor.
        # GH 30346
        df = DataFrame({k: np.random.default_rng(2).random(100) for k in "ABC"})
        kwd = {props: {"color": "C1"}}
        result = df.plot.box(return_type="dict", **kwd)

        assert result[expected][0].get_color() == "C1"

    def test_unordered_ts(self):
        # [FUNCTION DOC]
        # GH#2609, GH#55906. A frame indexed by unsorted dates: the line keeps the data
        # in the original order (x data are the dates as objects, y the values), while the
        # tick positions and labels are in increasing order.
        # GH#2609, GH#55906
        index = [date(2012, 10, 1), date(2012, 9, 1), date(2012, 8, 1)]
        values = [3.0, 2.0, 1.0]
        df = DataFrame(
            np.array(values),
            index=index,
            columns=["test"],
        )
        ax = df.plot()
        xticks = ax.lines[0].get_xdata()
        tm.assert_numpy_array_equal(xticks, np.array(index, dtype=object))
        ydata = ax.lines[0].get_ydata()
        tm.assert_numpy_array_equal(ydata, np.array(values))

        # even though we don't sort the data before passing it to matplotlib,
        # the ticks are sorted
        xticks = ax.xaxis.get_ticklabels()
        xlocs = [x.get_position()[0] for x in xticks]
        assert Index(xlocs).is_monotonic_increasing
        xlabels = [x.get_text() for x in xticks]
        assert pd.to_datetime(xlabels, format="%Y-%m-%d").is_monotonic_increasing

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_kind_both_ways(self, kind):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over the common plot kinds. Plot by kind keyword and
        # by accessor method both work on a one-column frame.
        pytest.importorskip("scipy")
        df = DataFrame({"x": [1, 2, 3]})
        df.plot(kind=kind)
        getattr(df.plot, kind)()

    @pytest.mark.parametrize("kind", ["scatter", "hexbin"])
    def test_kind_both_ways_x_y(self, kind):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over scatter and hexbin. Plot with x and y given
        # positionally by keyword kind and by accessor method.
        pytest.importorskip("scipy")
        df = DataFrame({"x": [1, 2, 3]})
        df.plot("x", "x", kind=kind)
        getattr(df.plot, kind)("x", "x")

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_all_invalid_plot_data(self, kind):
        # [FUNCTION DOC]
        # Parametrized over the common plot kinds. A frame of strings raises a type error
        # saying there is no numeric data to plot.
        df = DataFrame(list("abcd"))
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            df.plot(kind=kind)

    @pytest.mark.parametrize(
        "kind", [*list(plotting.PlotAccessor._common_kinds), "area"]
    )
    def test_partially_invalid_plot_data_numeric(self, kind):
        # [FUNCTION DOC]
        # Parametrized over the common kinds plus area. An object-dtype frame in which a
        # random subset of rows were replaced by a string raises a type error saying there
        # is no numeric data to plot.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            dtype=object,
        )
        df[np.random.default_rng(2).random(df.shape[0]) > 0.5] = "a"
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            df.plot(kind=kind)

    def test_invalid_kind(self):
        # [FUNCTION DOC]
        # An unknown plot kind raises a value error naming the kind as not valid.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        msg = "invalid_plot_kind is not a valid plot kind"
        with pytest.raises(ValueError, match=msg):
            df.plot(kind="invalid_plot_kind")

    @pytest.mark.parametrize(
        "x,y,lbl",
        [
            (["B", "C"], "A", "a"),
            (["A"], ["B", "C"], ["b", "c"]),
        ],
    )
    def test_invalid_xy_args(self, x, y, lbl):
        # [FUNCTION DOC]
        # GH 18671, 19699. Parametrized over invalid x arguments (list-like). Raises a
        # value error saying x must be a label or position, even with list-like y.
        # GH 18671, 19699 allows y to be list-like but not x
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        with pytest.raises(ValueError, match="x must be a label or position"):
            df.plot(x=x, y=y, label=lbl)

    def test_bad_label(self):
        # [FUNCTION DOC]
        # A label string that is not list-like with the same length as list-like y raises
        # a value error.
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        msg = "label should be list-like and same length as y"
        with pytest.raises(ValueError, match=msg):
            df.plot(x="A", y=["B", "C"], label="bad_label")

    @pytest.mark.parametrize("x,y", [("A", "B"), (["A"], "B")])
    def test_invalid_xy_args_dup_cols(self, x, y):
        # [FUNCTION DOC]
        # GH 18671, 19699. With duplicate column labels, an x that selects an ambiguous
        # or list-like column raises a value error saying x must be a label or position.
        # GH 18671, 19699 allows y to be list-like but not x
        df = DataFrame([[1, 3, 5], [2, 4, 6]], columns=list("AAB"))
        with pytest.raises(ValueError, match="x must be a label or position"):
            df.plot(x=x, y=y)

    @pytest.mark.parametrize(
        "x,y,lbl,colors",
        [
            ("A", ["B"], ["b"], ["red"]),
            ("A", ["B", "C"], ["b", "c"], ["red", "blue"]),
            (0, [1, 2], ["bokeh", "cython"], ["green", "yellow"]),
        ],
    )
    def test_y_listlike(self, x, y, lbl, colors):
        # [FUNCTION DOC]
        # GH 19699. Parametrized over (x, list-like y, labels, colors). Plotting several y
        # columns works with labels, and with colors there are as many lines as y columns
        # with those colors.
        # KNOWN ISSUE: the first check uses the literal column A as x instead of the
        # parametrized x value.
        # GH 19699: tests list-like y and verifies lbls & colors
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        _check_plot_works(df.plot, x="A", y=y, label=lbl)

        ax = df.plot(x=x, y=y, label=lbl, color=colors)
        assert len(ax.lines) == len(y)
        _check_colors(ax.get_lines(), linecolors=colors)

    @pytest.mark.parametrize("x,y,colnames", [(0, 1, ["A", "B"]), (1, 0, [0, 1])])
    def test_xy_args_integer(self, x, y, colnames):
        # [FUNCTION DOC]
        # GH 20056. Parametrized over positions and column labels (strings or integers).
        # Integer x and y arguments are treated as positions or labels and the plot works.
        # GH 20056: tests integer args for xy and checks col names
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        df.columns = colnames
        _check_plot_works(df.plot, x=x, y=y)

    def test_hexbin_basic(self):
        # [FUNCTION DOC]
        # A hexbin plot yields exactly one collection (a TODO notes the check is only
        # existence).
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )

        ax = df.plot.hexbin(x="A", y="B", gridsize=10)
        # TODO: need better way to test. This just does existence.
        assert len(ax.collections) == 1

    def test_hexbin_basic_subplots(self):
        # [FUNCTION DOC]
        # GH 6951. Hexbin with subplots has two axes in the figure (plot and colorbar) and
        # the return value is a single axes in a one-by-one layout.
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        # GH 6951
        axes = df.plot.hexbin(x="A", y="B", subplots=True)
        # hexbin should have 2 axes in the figure, 1 for plotting and another
        # is colorbar
        assert len(axes[0].figure.axes) == 2
        # return value is single axes
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.parametrize("reduce_C", [None, np.std])
    def test_hexbin_with_c(self, reduce_C):
        # [FUNCTION DOC]
        # Parametrized over the reduce function (none or standard deviation). Hexbin with
        # a color column and the reduce function gives one collection.
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )

        ax = df.plot.hexbin(x="A", y="B", C="C", reduce_C_function=reduce_C)
        assert len(ax.collections) == 1

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "BuGn"),  # default cmap
            ({"colormap": "cubehelix"}, "cubehelix"),
            ({"cmap": "YlGn"}, "YlGn"),
        ],
    )
    def test_hexbin_cmap(self, kwargs, expected):
        # [FUNCTION DOC]
        # Parametrized over keywords. Default colormap is BuGn; the colormap or cmap
        # keyword selects the named colormap.
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        ax = df.plot.hexbin(x="A", y="B", **kwargs)
        assert ax.collections[0].cmap.name == expected

    def test_pie_df_err(self):
        # [FUNCTION DOC]
        # A pie plot of a frame without y and without subplots raises a value error saying
        # pie requires either a y column or subplots.
        df = DataFrame(
            np.random.default_rng(2).random((5, 3)),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        msg = "pie requires either y column or 'subplots=True'"
        with pytest.raises(ValueError, match=msg):
            df.plot.pie()

    @pytest.mark.parametrize("y", ["Y", 2])
    def test_pie_df(self, y):
        # [FUNCTION DOC]
        # Parametrized over y by name or position. Pie plot text labels are the index.
        df = DataFrame(
            np.random.default_rng(2).random((5, 3)),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        ax = _check_plot_works(df.plot.pie, y=y)
        _check_text_labels(ax.texts, df.index)

    def test_pie_df_subplots(self):
        # [FUNCTION DOC]
        # Pie plot with subplots makes one axes per column; each axes has the index as text
        # labels and an empty y label.
        df = DataFrame(
            np.random.default_rng(2).random((5, 3)),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        axes = _check_plot_works(
            df.plot.pie,
            default_axes=True,
            subplots=True,
        )
        assert len(axes) == len(df.columns)
        for ax in axes:
            _check_text_labels(ax.texts, df.index)
        for ax, ylabel in zip(axes, df.columns, strict=True):
            assert ax.get_ylabel() == ""

    def test_pie_df_labels_colors(self):
        # [FUNCTION DOC]
        # Pie with subplots and given labels and colors: text labels equal the labels and
        # wedge face colors equal the colors on every axes.
        df = DataFrame(
            np.random.default_rng(2).random((5, 3)),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        labels = ["A", "B", "C", "D", "E"]
        color_args = ["r", "g", "b", "c", "m"]
        axes = _check_plot_works(
            df.plot.pie,
            default_axes=True,
            subplots=True,
            labels=labels,
            colors=color_args,
        )
        assert len(axes) == len(df.columns)

        for ax in axes:
            _check_text_labels(ax.texts, labels)
            _check_colors(ax.patches, facecolors=color_args)

    def test_pie_df_nan(self):
        # [FUNCTION DOC]
        # GH 37668. Pie plot with subplots, normalization and legend on existing axes for a
        # frame with a diagonal of missing values emits no warning; text labels exclude
        # the missing wedge (empty label) and legend labels omit the missing entries
        # (issue 8390).
        df = DataFrame(np.random.default_rng(2).random((4, 4)))
        for i in range(4):
            df.iloc[i, i] = np.nan
        _, axes = mpl.pyplot.subplots(ncols=4)

        # GH 37668
        kwargs = {"normalize": True}

        with tm.assert_produces_warning(None):
            df.plot.pie(subplots=True, ax=axes, legend=True, **kwargs)

        base_expected = ["0", "1", "2", "3"]
        for i, ax in enumerate(axes):
            expected = list(base_expected)  # force copy
            expected[i] = ""
            result = [x.get_text() for x in ax.texts]
            assert result == expected

            # legend labels
            # NaN's not included in legend with subplots
            # see https://github.com/pandas-dev/pandas/issues/8390
            result_labels = [x.get_text() for x in ax.get_legend().get_texts()]
            expected_labels = base_expected[:i] + base_expected[i + 1 :]
            assert result_labels == expected_labels

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"logy": True},
            {"logx": True, "logy": True},
            {"loglog": True},
        ],
    )
    def test_errorbar_plot(self, kwargs):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over log-scale keyword sets. A line plot with error
        # data frame gives two y error bar containers.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}
        df_err = DataFrame(d_err)

        # check line plots
        ax = _check_plot_works(df.plot, yerr=df_err, **kwargs)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    def test_errorbar_plot_bar(self):
        # [FUNCTION DOC]
        # Marked slow. Bar plot with both y and x errors and log scale has two x and two y
        # error bar containers.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}
        df_err = DataFrame(d_err)
        ax = _check_plot_works(
            (df + 1).plot, yerr=df_err, xerr=df_err, kind="bar", log=True
        )
        _check_has_errorbars(ax, xerr=2, yerr=2)

    @pytest.mark.slow
    def test_errorbar_plot_yerr_array(self):
        # [FUNCTION DOC]
        # Marked slow. A series plot with a raw error array has one error bar; a frame
        # plot with a two-by-n error array (asymmetric) has two.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        # yerr is raw error values
        ax = _check_plot_works(df["y"].plot, yerr=np.ones(12) * 0.4)
        _check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(df.plot, yerr=np.ones((2, 12)) * 0.4)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("yerr", ["yerr", "誤差"])
    def test_errorbar_plot_column_name(self, yerr):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over ASCII and non-ASCII error column names. Using a
        # column of the frame as error yields two y error bars (the error column is also
        # plotted as a column); with x and y given, one.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        df[yerr] = np.ones(12) * 0.2

        ax = _check_plot_works(df.plot, yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(df.plot, y="y", x="x", yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_external_valueerror(self):
        # [FUNCTION DOC]
        # Marked slow. An error array of the wrong length makes the underlying plotting
        # library raise a value error (checked with the external error helper).
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        with tm.external_error_raised(ValueError):
            df.plot(yerr=np.random.default_rng(2).standard_normal(11))

    @pytest.mark.slow
    def test_errorbar_plot_external_typeerror(self):
        # [FUNCTION DOC]
        # Marked slow. String error data makes the underlying library raise a type error.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        df_err = DataFrame({"x": ["zzz"] * 12, "y": ["zzz"] * 12})
        with tm.external_error_raised(TypeError):
            df.plot(yerr=df_err)

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    @pytest.mark.parametrize(
        "y_err",
        [
            Series(np.ones(12) * 0.2, name="x"),
            DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}),
        ],
    )
    def test_errorbar_plot_different_yerr(self, kind, y_err):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over kind and error as series or frame. Two y error
        # bars with no x errors.
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})

        ax = _check_plot_works(df.plot, yerr=y_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    @pytest.mark.parametrize(
        "y_err, x_err",
        [
            (
                DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}),
                DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}),
            ),
            (Series(np.ones(12) * 0.2, name="x"), Series(np.ones(12) * 0.2, name="x")),
            (0.2, 0.2),
        ],
    )
    def test_errorbar_plot_different_yerr_xerr(self, kind, y_err, x_err):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over kind and error types (frames, series, scalar).
        # Two x and two y error bars.
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})
        ax = _check_plot_works(df.plot, yerr=y_err, xerr=x_err, kind=kind)
        _check_has_errorbars(ax, xerr=2, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    def test_errorbar_plot_different_yerr_xerr_subplots(self, kind):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over kind. With subplots, each axes has one x and one y
        # error bar.
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})
        df_err = DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4})
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            yerr=df_err,
            xerr=df_err,
            subplots=True,
            kind=kind,
        )
        _check_has_errorbars(axes, xerr=1, yerr=1)

    @pytest.mark.xfail(reason="Iterator is consumed", raises=ValueError)
    def test_errorbar_plot_iterator(self):
        # [FUNCTION DOC]
        # Expected to fail with a value error (the iterator error data is consumed). Error
        # data given as an iterator should produce two y error bars.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)

        # yerr is iterator
        ax = _check_plot_works(df.plot, yerr=itertools.repeat(0.1, len(df)))
        _check_has_errorbars(ax, xerr=0, yerr=2)

    def test_errorbar_with_integer_column_names(self):
        # [FUNCTION DOC]
        # Integer column names: a frame as error gives two error bars; an integer y and
        # integer yerr selecting a column gives one.
        # test with integer column names
        df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))))
        df_err = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))))
        ax = _check_plot_works(df.plot, yerr=df_err)
        _check_has_errorbars(ax, xerr=0, yerr=2)
        ax = _check_plot_works(df.plot, y=0, yerr=1)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar"])
    def test_errorbar_with_partial_columns_kind(self, kind):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over line and bar. Error frame covering only some
        # columns gives two error bars.
        df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
        df_err = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2]
        )
        ax = _check_plot_works(df.plot, yerr=df_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    def test_errorbar_with_partial_columns_dti(self):
        # [FUNCTION DOC]
        # Marked slow. Same as above with a monthly date index on both frames.
        df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
        df_err = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2]
        )
        ix = date_range("1/1/2000", periods=10, freq="ME")
        df.set_index(ix, inplace=True)
        df_err.set_index(ix, inplace=True)
        ax = _check_plot_works(df.plot, yerr=df_err, kind="line")
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("err_box", [lambda x: x, DataFrame])
    def test_errorbar_with_partial_columns_box(self, err_box):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over a dict or frame as error with column labels only
        # partly matching: only the matching column gets one error bar.
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        err = err_box({"x": np.ones(12) * 0.2, "z": np.ones(12) * 0.4})
        ax = _check_plot_works(df.plot, yerr=err)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    def test_errorbar_timeseries(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind. On a monthly time-series frame: errors as a frame, as a
        # dict, as a series with y, as a column name with y, and with subplots all give the
        # expected number of error bars (two, two, one, one, two, one per axes).
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}

        # check time-series plots
        ix = date_range("1/1/2000", "1/1/2001", freq="ME")
        tdf = DataFrame(d, index=ix)
        tdf_err = DataFrame(d_err, index=ix)

        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(tdf.plot, yerr=d_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(tdf.plot, y="y", yerr=tdf_err["x"], kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(tdf.plot, y="y", yerr="x", kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

        axes = _check_plot_works(
            tdf.plot,
            default_axes=True,
            kind=kind,
            yerr=tdf_err,
            subplots=True,
        )
        _check_has_errorbars(axes, xerr=0, yerr=1)

    def test_errorbar_asymmetrical(self):
        # [FUNCTION DOC]
        # Three-dimensional error array of shape (3, 2, 5) on both axes: the error bar line
        # vertices match the first column's lower/upper errors; passing the transposed
        # array raises a value error stating the expected shape.
        err = np.random.default_rng(2).random((3, 2, 5))

        # each column is [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]...
        df = DataFrame(np.arange(15).reshape(3, 5)).T

        ax = df.plot(yerr=err, xerr=err / 2)

        yerr_0_0 = ax.collections[1].get_paths()[0].vertices[:, 1]
        expected_0_0 = err[0, :, 0] * np.array([-1, 1])
        tm.assert_almost_equal(yerr_0_0, expected_0_0)

        msg = re.escape(
            "Asymmetrical error bars should be provided with the shape (3, 2, 5)"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(yerr=err.T)

    def test_table(self):
        # [FUNCTION DOC]
        # Plot with table true or with a frame works. GH 35945: drawing a table on axes
        # manually emits no warning, starting with no tables and ending with one.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        _check_plot_works(df.plot, table=True)
        _check_plot_works(df.plot, table=df)

        # GH 35945 UserWarning
        with tm.assert_produces_warning(None):
            ax = df.plot()
            assert len(ax.tables) == 0
            plotting.table(ax, df.T)
            assert len(ax.tables) == 1

    def test_errorbar_scatter(self):
        # [FUNCTION DOC]
        # Scatter plot with no, x, y and both error frames has the corresponding error bar
        # containers.
        df = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((5, 2))),
            index=range(5),
            columns=["x", "y"],
        )
        df_err = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((5, 2))) / 5,
            index=range(5),
            columns=["x", "y"],
        )

        ax = _check_plot_works(df.plot.scatter, x="x", y="y")
        _check_has_errorbars(ax, xerr=0, yerr=0)
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", xerr=df_err)
        _check_has_errorbars(ax, xerr=1, yerr=0)

        ax = _check_plot_works(df.plot.scatter, x="x", y="y", yerr=df_err)
        _check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", xerr=df_err, yerr=df_err)
        _check_has_errorbars(ax, xerr=1, yerr=1)

    def test_errorbar_scatter_color(self):
        # [FUNCTION DOC]
        # GH 8081. Scatter with error columns and a color, or a color keyword: one x and
        # one y error bar, and the error bar lines take the scatter color.
        def _check_errorbar_color(containers, expected, has_err="has_xerr"):
            # [FUNCTION DOC]
            # HELPER: nested function of the scatter error bar color test. Parameters:
            # containers (unused), expected color, has_err name of the container attribute to
            # look for (default x errors). Finds the first container flagged as having that
            # kind of errors on the enclosing axes, collects its lines (flattening list-like
            # entries), keeps those that are also in the axes collections, and asserts that
            # their line colors equal the expected color.
            # KNOWN ISSUE: the containers parameter is ignored; the function reads the axes
            # from the enclosing scope instead.
            lines = []
            errs = next(c.lines for c in ax.containers if getattr(c, has_err, False))
            for el in errs:
                if is_list_like(el):
                    lines.extend(el)
                else:
                    lines.append(el)
            err_lines = [x for x in lines if x in ax.collections]
            _check_colors(err_lines, linecolors=np.array([expected] * len(err_lines)))

        # GH 8081
        df = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((10, 5))),
            columns=["a", "b", "c", "d", "e"],
        )
        ax = df.plot.scatter(x="a", y="b", xerr="d", yerr="e", c="red")
        _check_has_errorbars(ax, xerr=1, yerr=1)
        _check_errorbar_color(ax.containers, "red", has_err="has_xerr")
        _check_errorbar_color(ax.containers, "red", has_err="has_yerr")

        ax = df.plot.scatter(x="a", y="b", yerr="e", color="green")
        _check_has_errorbars(ax, xerr=0, yerr=1)
        _check_errorbar_color(ax.containers, "green", has_err="has_yerr")

    def test_scatter_unknown_colormap(self):
        # [FUNCTION DOC]
        # GH#48726. An unknown colormap name for scatter raises a value or key error
        # saying the colormap is not known.
        # GH#48726
        df = DataFrame({"a": [1, 2, 3], "b": 4})
        with pytest.raises((ValueError, KeyError), match="'unknown' is not a"):
            df.plot(x="a", y="b", colormap="unknown", kind="scatter")

    def test_sharex_and_ax(self):
        # [FUNCTION DOC]
        # Issue 9737 (gridspec axes ordering). Takes four axes from the 2x2 helper
        # (order: top-left, bottom-left, top-right, bottom-right) and plots the
        # same frame in each with sharex true, then applies tight layout: the two
        # top axes (list positions 0 and 2) hide x tick labels, the two bottom ones
        # (positions 1 and 3) show them, all keep y tick labels and one line each.
        # Then closes figures, creates the axes again and plots with subplots and
        # sharex true on them, expecting a user warning that sharex and sharey are
        # ignored, and the same visibility pattern.
        # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
        # the axis in fig.get_axis() are sorted differently than pandas
        # expected them, so make sure that only the right ones are removed
        gs, axes = _generate_4_axes_via_gridspec()

        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        def _check(axes):
            # [FUNCTION DOC]
            # HELPER: nested function of the sharex test. Takes the list of four axes. Each has
            # one line and visible y tick labels; the axes at positions 0 and 2 have hidden x tick
            # labels (major and minor), those at positions 1 and 3 have visible ones.
            for ax in axes:
                assert len(ax.lines) == 1
                _check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[0], axes[2]]:
                _check_visible(ax.get_xticklabels(), visible=False)
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            for ax in [axes[1], axes[3]]:
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)

        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax, sharex=True)
        gs.tight_layout(plt.gcf())
        _check(axes)
        plt.close("all")

        gs, axes = _generate_4_axes_via_gridspec()
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        _check(axes)

    def test_sharex_false_and_ax(self):
        # [FUNCTION DOC]
        # Issue 9737. Without sharex, plotting the same frame on four gridspec axes leaves
        # all labels (y, x major and minor) visible and one line in each axes.
        # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
        # the axis in fig.get_axis() are sorted differently than pandas
        # expected them, so make sure that only the right ones are removed
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )
        gs, axes = _generate_4_axes_via_gridspec()
        # without sharex, no labels should be touched!
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax)

        gs.tight_layout(plt.gcf())
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    def test_sharey_and_ax(self):
        # [FUNCTION DOC]
        # Issue 9737. Four gridspec axes plotted with sharey true: axes at positions 0 and 1
        # keep y labels, positions 2 and 3 hide them; all keep x labels and one line. Then
        # the same via subplots on the existing axes with a user warning about sharex and
        # sharey being ignored.
        # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
        # the axis in fig.get_axis() are sorted differently than pandas
        # expected them, so make sure that only the right ones are removed
        gs, axes = _generate_4_axes_via_gridspec()

        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        def _check(axes):
            # [FUNCTION DOC]
            # HELPER: nested function of the sharey test. Takes the list of four axes. Each has
            # one line and visible x tick labels (major and minor); the axes at positions 0 and 1
            # show y tick labels, positions 2 and 3 hide them.
            for ax in axes:
                assert len(ax.lines) == 1
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            for ax in [axes[0], axes[1]]:
                _check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[2], axes[3]]:
                _check_visible(ax.get_yticklabels(), visible=False)

        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax, sharey=True)
        gs.tight_layout(plt.gcf())
        _check(axes)
        plt.close("all")

        gs, axes = _generate_4_axes_via_gridspec()
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharey=True)

        gs.tight_layout(plt.gcf())
        _check(axes)

    def test_sharey_and_ax_tight(self):
        # [FUNCTION DOC]
        # Issue 9737. Plots the same frame on four gridspec axes with tight layout; every
        # axes keeps one line and all labels visible.
        # KNOWN ISSUE: despite the name it never uses sharey, and its body duplicates the
        # sharex-false test (even the comment says without sharex).
        # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )
        gs, axes = _generate_4_axes_via_gridspec()
        # without sharex, no labels should be touched!
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax)

        gs.tight_layout(plt.gcf())
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._all_kinds)
    def test_memory_leak(self, kind):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over all plot kinds. Builds a suitable frame for the
        # kind (x/y for hexbin, scatter, pie; absolute values for area; time series
        # otherwise), plots, and checks (issue 9003) that for line and area plots the cached
        # plot data are copies not sharing memory but equal in values; for other kinds the
        # axes has no cached plot data attribute.
        """Check that every plot type gets properly collected."""
        pytest.importorskip("scipy")
        args = {}
        if kind in ["hexbin", "scatter", "pie"]:
            df = DataFrame(
                {
                    "A": np.random.default_rng(2).uniform(size=20),
                    "B": np.random.default_rng(2).uniform(size=20),
                    "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
                }
            )
            args = {"x": "A", "y": "B"}
        elif kind == "area":
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=date_range("2000-01-01", periods=10, freq="B"),
            ).abs()
        else:
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=date_range("2000-01-01", periods=10, freq="B"),
            )

        ax = df.plot(kind=kind, **args)
        # https://github.com/pandas-dev/pandas/issues/9003#issuecomment-70544889
        if kind in ["line", "area"]:
            for i, (cached_data, _, _) in enumerate(ax._plot_data):
                ser = df.iloc[:, i]
                assert not tm.shares_memory(ser, cached_data)
                tm.assert_numpy_array_equal(ser._values, cached_data._values)
        else:
            assert not hasattr(ax, "_plot_data")

    def test_df_gridspec_patterns_vert_horiz(self):
        # [FUNCTION DOC]
        # GH 10819. Plots a series on the big gridspec cell and a frame on the small one, in
        # vertical and horizontal layouts: one and two lines and all labels visible. With
        # subplots over the two axes: one line each, labels visible. With subplots and sharex
        # and sharey (user warning expected): vertical layout hides the top axes' x labels and
        # keeps y labels; horizontal layout hides the right axes' y labels and keeps x labels.
        # Closes all figures between cases.
        # GH 10819
        ts = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1/1/2000", periods=10),
        )

        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=ts.index,
            columns=list("AB"),
        )

        def _get_vertical_grid():
            # [FUNCTION DOC]
            # HELPER: nested function building a figure with a three-row grid and returning a
            # large top axes (two rows) and a bottom axes.
            gs = mpl.gridspec.GridSpec(3, 1)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :])
            ax2 = fig.add_subplot(gs[2, :])
            return ax1, ax2

        def _get_horizontal_grid():
            # [FUNCTION DOC]
            # HELPER: nested function building a figure with a three-column grid and returning a
            # wide left axes (two columns) and a right axes.
            gs = mpl.gridspec.GridSpec(1, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:, :2])
            ax2 = fig.add_subplot(gs[:, 2])
            return ax1, ax2

        for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
            ax1 = ts.plot(ax=ax1)
            assert len(ax1.lines) == 1
            ax2 = df.plot(ax=ax2)
            assert len(ax2.lines) == 2
            for ax in [ax1, ax2]:
                _check_visible(ax.get_yticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            plt.close("all")

        # subplots=True
        for ax1, ax2 in [_get_vertical_grid(), _get_horizontal_grid()]:
            axes = df.plot(subplots=True, ax=[ax1, ax2])
            assert len(ax1.lines) == 1
            assert len(ax2.lines) == 1
            for ax in axes:
                _check_visible(ax.get_yticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            plt.close("all")

        # vertical / subplots / sharex=True / sharey=True
        ax1, ax2 = _get_vertical_grid()
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        for ax in [ax1, ax2]:
            # yaxis are visible because there is only one column
            _check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of axes0 (top) are hidden
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)
        plt.close("all")

        # horizontal / subplots / sharex=True / sharey=True
        ax1, ax2 = _get_horizontal_grid()
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=[ax1, ax2], sharex=True, sharey=True)
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == 1
        _check_visible(axes[0].get_yticklabels(), visible=True)
        # yaxis of axes1 (right) are hidden
        _check_visible(axes[1].get_yticklabels(), visible=False)
        for ax in [ax1, ax2]:
            # xaxis are visible because there is only one column
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close("all")

    def test_df_gridspec_patterns_boxed(self):
        # [FUNCTION DOC]
        # GH 10819. Four axes in a boxed three by three grid; subplots on a four-column frame:
        # without sharing all labels are visible with one line each; with sharex and sharey
        # (user warning expected) left column keeps y labels, right column hides them, top row
        # hides x labels, bottom row keeps them.
        # GH 10819
        ts = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1/1/2000", periods=10),
        )

        # boxed
        def _get_boxed_grid():
            # [FUNCTION DOC]
            # HELPER: nested function building a figure with a 3x3 grid and returning four axes: a
            # large top-left, a top-right, a bottom-left and a bottom-right.
            gs = mpl.gridspec.GridSpec(3, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :2])
            ax2 = fig.add_subplot(gs[:2, 2])
            ax3 = fig.add_subplot(gs[2, :2])
            ax4 = fig.add_subplot(gs[2, 2])
            return ax1, ax2, ax3, ax4

        axes = _get_boxed_grid()
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=ts.index,
            columns=list("ABCD"),
        )
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            # axis are visible because these are not shared
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close("all")

        # subplots / sharex=True / sharey=True
        axes = _get_boxed_grid()
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharex=True, sharey=True)
        for ax in axes:
            assert len(ax.lines) == 1
        for ax in [axes[0], axes[2]]:  # left column
            _check_visible(ax.get_yticklabels(), visible=True)
        for ax in [axes[1], axes[3]]:  # right column
            _check_visible(ax.get_yticklabels(), visible=False)
        for ax in [axes[0], axes[1]]:  # top row
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
        for ax in [axes[2], axes[3]]:  # bottom row
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        plt.close("all")

    def test_df_grid_settings(self):
        # [FUNCTION DOC]
        # GH 9792. Checks the plot grid defaults for all frame plot kinds via the shared grid
        # helper using column a as x and b as y.
        # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792
        _check_grid_settings(
            DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
            plotting.PlotAccessor._dataframe_kinds,
            kws={"x": "a", "y": "b"},
        )

    def test_plain_axes(self):
        # [FUNCTION DOC]
        # GH11556. A series plots onto a subplot axes although the figure also contains a
        # plain (non-subplot) axes.
        # supplied ax itself is a SubplotAxes, but figure contains also
        # a plain Axes object (GH11556)
        fig, ax = mpl.pyplot.subplots()
        fig.add_axes([0.2, 0.2, 0.2, 0.2])
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)

    def test_plain_axes_df(self):
        # [FUNCTION DOC]
        # GH11520. A scatter plot with colormap onto a plain axes (created by position) works
        # although a colorbar axes is created.
        # supplied ax itself is a plain Axes, but because the cmap keyword
        # a new ax is created for the colorbar -> also multiples axes (GH11520)
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(8),
                "b": np.random.default_rng(2).standard_normal(8),
            }
        )
        fig = mpl.pyplot.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        df.plot(kind="scatter", ax=ax, x="a", y="b", c="a", cmap="hsv")

    def test_plain_axes_make_axes_locatable(self):
        # [FUNCTION DOC]
        # Plotting series onto an axes and a colorbar-like axes appended through the axes
        # divider toolkit works.
        # other examples
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        Series(np.random.default_rng(2).random(10)).plot(ax=cax)

    def test_plain_axes_make_inset_axes(self):
        # [FUNCTION DOC]
        # Plotting series onto an axes and an inset axes created through the inset locator
        # toolkit works.
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        iax = inset_axes(ax, width="30%", height=1.0, loc=3)
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        Series(np.random.default_rng(2).random(10)).plot(ax=iax)

    @pytest.mark.parametrize("method", ["line", "barh", "bar"])
    def test_secondary_axis_font_size(self, method):
        # [FUNCTION DOC]
        # GH: 12565. Parametrized over line, barh, bar. With secondary y columns and font size
        # 20, the right axes' y tick label size is 20.
        # GH: 12565
        df = (
            DataFrame(
                np.random.default_rng(2).standard_normal((15, 2)), columns=list("AB")
            )
            .assign(C=lambda df: df.B.cumsum())
            .assign(D=lambda df: df.C * 1.1)
        )

        fontsize = 20
        sy = ["C", "D"]

        kwargs = {"secondary_y": sy, "fontsize": fontsize, "mark_right": True}
        ax = getattr(df.plot, method)(**kwargs)
        _check_ticks_props(axes=ax.right_ax, ylabelsize=fontsize)

    def test_x_string_values_ticks(self):
        # [FUNCTION DOC]
        # GH 7612, 22334. Area plot with a string column as x: after setting x limits wider
        # than the data, the tick labels remain at positions 0, 1, 2 for the three days.
        # Test if string plot index have a fixed xtick position
        # GH: 7612, GH: 22334
        df = DataFrame(
            {
                "sales": [3, 2, 3],
                "visits": [20, 42, 28],
                "day": ["Monday", "Tuesday", "Wednesday"],
            }
        )
        ax = df.plot.area(x="day")
        ax.set_xlim(-1, 3)
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        labels_position = dict(zip(xticklabels, ax.get_xticks(), strict=False))
        # Testing if the label stayed at the right position
        assert labels_position["Monday"] == 0.0
        assert labels_position["Tuesday"] == 1.0
        assert labels_position["Wednesday"] == 2.0

    def test_x_multiindex_values_ticks(self):
        # [FUNCTION DOC]
        # GH 15912. Line plot with a two-level index: after widening x limits, tick labels
        # formatted as tuples sit at positions 0..3.
        # Test if multiindex plot index have a fixed xtick position
        # GH: 15912
        index = MultiIndex.from_product([[2012, 2013], [1, 2]])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            columns=["A", "B"],
            index=index,
        )
        ax = df.plot()
        ax.set_xlim(-1, 4)
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        labels_position = dict(zip(xticklabels, ax.get_xticks(), strict=False))
        # Testing if the label stayed at the right position
        assert labels_position["(2012, 1)"] == 0.0
        assert labels_position["(2012, 2)"] == 1.0
        assert labels_position["(2013, 1)"] == 2.0
        assert labels_position["(2013, 2)"] == 3.0

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_xlim_plot_line(self, kind):
        # [FUNCTION DOC]
        # GH 27686. Parametrized over line and area. X limits extend beyond the data range on
        # both sides.
        # test if xlim is set correctly in plot.line and plot.area
        # GH 27686
        df = DataFrame([2, 4], index=[1, 2])
        ax = df.plot(kind=kind)
        xlims = ax.get_xlim()
        assert xlims[0] < 1
        assert xlims[1] > 2

    def test_xlim_plot_line_correctly_in_mixed_plot_type(self):
        # [FUNCTION DOC]
        # GH 27686. Bar plot and a secondary-y line plot on one axes: x limits extend beyond
        # the categories on both sides and the tick labels are the index labels.
        # test if xlim is set correctly when ax contains multiple different kinds
        # of plots, GH 27686
        fig, ax = mpl.pyplot.subplots()

        indexes = ["k1", "k2", "k3", "k4"]
        df = DataFrame(
            {
                "s1": [1000, 2000, 1500, 2000],
                "s2": [900, 1400, 2000, 3000],
                "s3": [1500, 1500, 1600, 1200],
                "secondary_y": [1, 3, 4, 3],
            },
            index=indexes,
        )
        df[["s1", "s2", "s3"]].plot.bar(ax=ax, stacked=False)
        df[["secondary_y"]].plot(ax=ax, secondary_y=True)

        xlims = ax.get_xlim()
        assert xlims[0] < 0
        assert xlims[1] > 3

        # make sure axis labels are plotted correctly as well
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xticklabels == indexes

    def test_plot_no_rows(self):
        # [FUNCTION DOC]
        # GH 27758. An empty frame with a column still plots one line with no x and y data.
        # GH 27758
        df = DataFrame(columns=["foo"], dtype=int)
        assert df.empty
        ax = df.plot()
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self):
        # [FUNCTION DOC]
        # A frame of strings raises a type error saying there is no numeric data to plot.
        df = DataFrame(["a", "b", "c"])
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    @pytest.mark.parametrize(
        "kind", ("line", "bar", "barh", "hist", "kde", "density", "area", "pie")
    )
    def test_group_subplot(self, kind):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over line, bar, barh, hist, kde, density, area and pie.
        # Subplots given as a list of column groups (b and e; c and d) gives three axes (two
        # groups plus the leftover column a); legend labels equal the group columns (not for
        # pie) and line plots have as many lines as group columns.
        pytest.importorskip("scipy")
        d = {
            "a": np.arange(10),
            "b": np.arange(10) + 1,
            "c": np.arange(10) + 1,
            "d": np.arange(10),
            "e": np.arange(10),
        }
        df = DataFrame(d)

        axes = df.plot(subplots=[("b", "e"), ("c", "d")], kind=kind)
        assert len(axes) == 3  # 2 groups + single column a

        expected_labels = (["b", "e"], ["c", "d"], ["a"])
        for ax, labels in zip(axes, expected_labels, strict=True):
            if kind != "pie":
                _check_legend_labels(ax, labels=labels)
            if kind == "line":
                assert len(ax.lines) == len(labels)

    def test_group_subplot_series_notimplemented(self):
        # [FUNCTION DOC]
        # An iterable value for subplots with a series raises a not-implemented error.
        ser = Series(range(1))
        msg = "An iterable subplots for a Series"
        with pytest.raises(NotImplementedError, match=msg):
            ser.plot(subplots=[("a",)])

    def test_group_subplot_multiindex_notimplemented(self):
        # [FUNCTION DOC]
        # An iterable subplots value with multi-level columns raises a not-implemented error.
        df = DataFrame(np.eye(2), columns=MultiIndex.from_tuples([(0, 1), (1, 2)]))
        msg = "An iterable subplots for a DataFrame with a MultiIndex"
        with pytest.raises(NotImplementedError, match=msg):
            df.plot(subplots=[(0, 1)])

    def test_group_subplot_nonunique_cols_notimplemented(self):
        # [FUNCTION DOC]
        # An iterable subplots value with non-unique column labels raises a not-implemented
        # error.
        df = DataFrame(np.eye(2), columns=["a", "a"])
        msg = "An iterable subplots for a DataFrame with non-unique"
        with pytest.raises(NotImplementedError, match=msg):
            df.plot(subplots=[("a",)])

    @pytest.mark.parametrize(
        "subplots, expected_msg",
        [
            (123, "subplots should be a bool or an iterable"),
            ("a", "each entry should be a list/tuple"),  # iterable of non-iterable
            ((1,), "each entry should be a list/tuple"),  # iterable of non-iterable
            (("a",), "each entry should be a list/tuple"),  # iterable of strings
        ],
    )
    def test_group_subplot_bad_input(self, subplots, expected_msg):
        # [FUNCTION DOC]
        # Parametrized over invalid subplots values and expected message: a non-iterable
        # number gives a value error that subplots should be a bool or an iterable; a string,
        # or an iterable of non-iterables or strings, gives a value error that each entry
        # should be a list or tuple.
        # Make sure error is raised when subplots is not a properly
        # formatted iterable. Only iterables of iterables are permitted, and
        # entries should not be strings.
        d = {"a": np.arange(10), "b": np.arange(10)}
        df = DataFrame(d)

        with pytest.raises(ValueError, match=expected_msg):
            df.plot(subplots=subplots)

    def test_group_subplot_invalid_column_name(self):
        # [FUNCTION DOC]
        # A group naming a missing column raises a value error listing the missing label; the
        # expected message format depends on whether numpy is older than 2.0 (plain string) or
        # newer (numpy string repr).
        d = {"a": np.arange(10), "b": np.arange(10)}
        df = DataFrame(d)

        if Version(np.__version__) < Version("2.0.0"):
            with pytest.raises(ValueError, match=r"Column label\(s\) \['bad_name'\]"):
                df.plot(subplots=[("a", "bad_name")])
        else:
            with pytest.raises(
                ValueError, match=r"Column label\(s\) \[np\.str\_\('bad_name'\)\]"
            ):
                df.plot(subplots=[("a", "bad_name")])

    def test_group_subplot_duplicated_column(self):
        # [FUNCTION DOC]
        # A column in two groups raises a value error saying it should be in only one subplot.
        d = {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
        df = DataFrame(d)

        with pytest.raises(ValueError, match="should be in only one subplot"):
            df.plot(subplots=[("a", "b"), ("a", "c")])

    @pytest.mark.parametrize("kind", ("box", "scatter", "hexbin"))
    def test_group_subplot_invalid_kind(self, kind):
        # [FUNCTION DOC]
        # Parametrized over box, scatter and hexbin. An iterable subplots value with these
        # kinds raises a value error listing the allowed kinds.
        d = {"a": np.arange(10), "b": np.arange(10)}
        df = DataFrame(d)
        with pytest.raises(
            ValueError, match="When subplots is an iterable, kind must be one of"
        ):
            df.plot(subplots=[("a", "b")], kind=kind)

    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [
            (None, "", "new"),
            ("old", "old", "new"),
            (None, "", ""),
            (None, "", 1),
            (None, "", [1, 2]),
        ],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar"])
    def test_xlabel_ylabel_dataframe_single_plot(
        self, kind, index_name, old_label, new_label
    ):
        # [FUNCTION DOC]
        # GH 9093. Parametrized over index name with old and new label and kind (line, area,
        # bar). Default: no y label and x label equal to the index name (or empty); passing
        # x and y labels overrides both with their string form.
        # GH 9093
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        df.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name
        ax = df.plot(kind=kind)
        assert ax.get_xlabel() == old_label
        assert ax.get_ylabel() == ""

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        ax = df.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == str(new_label)
        assert ax.get_xlabel() == str(new_label)

    @pytest.mark.parametrize(
        "xlabel, ylabel",
        [
            (None, None),
            ("X Label", None),
            (None, "Y Label"),
            ("X Label", "Y Label"),
        ],
    )
    @pytest.mark.parametrize("kind", ["scatter", "hexbin"])
    def test_xlabel_ylabel_dataframe_plane_plot(self, kind, xlabel, ylabel):
        # [FUNCTION DOC]
        # GH 37001. Parametrized over given or missing x and y labels and kind (scatter,
        # hexbin). Axis labels default to the column names and are overridden by the keywords.
        # GH 37001
        xcol = "Type A"
        ycol = "Type B"
        df = DataFrame([[1, 2], [2, 5]], columns=[xcol, ycol])

        # default is the labels are column names
        ax = df.plot(kind=kind, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel)
        assert ax.get_xlabel() == (xcol if xlabel is None else xlabel)
        assert ax.get_ylabel() == (ycol if ylabel is None else ylabel)

    @pytest.mark.parametrize("secondary_y", (False, True))
    def test_secondary_y(self, secondary_y):
        # [FUNCTION DOC]
        # Parametrized over secondary y false/true. With y label Y, y limits (0, 100) and y
        # ticks starting at 99: every visible y axis in the figure has the label, limits and
        # first tick.
        ax_df = DataFrame([0]).plot(
            secondary_y=secondary_y, ylabel="Y", ylim=(0, 100), yticks=[99]
        )
        for ax in ax_df.figure.axes:
            if ax.yaxis.get_visible():
                assert ax.get_ylabel() == "Y"
                assert ax.get_ylim() == (0, 100)
                assert ax.get_yticks()[0] == 99

    @pytest.mark.slow
    def test_plot_no_warning(self):
        # [FUNCTION DOC]
        # Marked slow. GH 55138. Plotting a business-day frame and its transpose emits no
        # warnings (the TODO says it can be removed once the business period deprecation is
        # enforced).
        # GH 55138
        # TODO(3.0): this can be removed once Period[B] deprecation is enforced
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        with tm.assert_produces_warning(False):
            _ = df.plot()
            _ = df.T.plot()

    @pytest.mark.parametrize("freq", ["h", "7h", "60min", "120min", "3M"])
    def test_plot_period_index_makes_no_right_shift(self, freq):
        # [FUNCTION DOC]
        # GH#57587. Parametrized over frequencies (hourly, multiples, minutes, 3 months). X
        # data of the line for a period indexed frame equal the index values (compared as
        # strings).
        # GH#57587
        idx = pd.period_range("01/01/2000", freq=freq, periods=4)
        df = DataFrame(
            np.array([0, 1, 0, 1]),
            index=idx,
            columns=["A"],
        )
        expected = idx.values

        ax = df.plot()
        result = ax.get_lines()[0].get_xdata()
        assert all(str(result[i]) == str(expected[i]) for i in range(4))

    def test_plot_display_xlabel_and_xticks(self):
        # [FUNCTION DOC]
        # GH#44050. Hexbin plot shows its x label and x tick labels.
        # GH#44050
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=["a", "b"])
        ax = df.plot.hexbin(x="a", y="b")

        _check_visible([ax.xaxis.get_label()], visible=True)
        _check_visible(ax.get_xticklabels(), visible=True)


def _generate_4_axes_via_gridspec():
    # [FUNCTION DOC]
    # HELPER: used by the shared axes tests of this module. Creates a 2x2 grid on the
    # current figure and four subplots in the order top-left, bottom-left, top-right,
    # bottom-right. Returns the grid specification and the list of four axes in that order
    # (so index 0 and 1 form the left column and 2 and 3 the right column).
    gs = mpl.gridspec.GridSpec(2, 2)
    ax_tl = plt.subplot(gs[0, 0])
    ax_ll = plt.subplot(gs[1, 0])
    ax_tr = plt.subplot(gs[0, 1])
    ax_lr = plt.subplot(gs[1, 1])

    return gs, [ax_tl, ax_ll, ax_tr, ax_lr]
