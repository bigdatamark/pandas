"""Test cases for DataFrame.plot"""

# [MODULE DOC]
# Tests for frame plotting with subplots: tick/label visibility with shared
# axes, layouts, passing existing axes, time series and duplicate columns,
# log-scale bar plots, box plot return types, axis labels and bar alignment.
# Public contents: the TestDataFramePlotsSubplots class. The hist_df fixture
# comes from the plotting conftest. Skipped if matplotlib is missing
# (import-or-skip at module level). No other import-time side effects.

import string

import numpy as np
import pytest

from pandas.compat import is_platform_linux

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_box_return_type,
    _check_legend_labels,
    _check_ticks_props,
    _check_visible,
    _flatten_visible,
)
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


class TestDataFramePlotsSubplots:
    # [CLASS DOC]
    # Test class for subplot-related behavior of DataFrame.plot, with one private
    # helper method for the bar alignment checks.
    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots(self, kind):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over kind (bar, barh, line, area). Plots three
        # columns with subplots and shared x: three axes in a column layout, one axes
        # per column with legend label equal to the column name. Axes other than the last
        # two hide x tick labels and x axis label (minor tick labels too, except for
        # bar plots, issue 26714) while y labels stay visible; the last axes shows all.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        assert axes.shape == (3,)

        for ax, column in zip(axes, df.columns, strict=True):
            _check_legend_labels(ax, labels=[pprint_thing(column)])

        for ax in axes[:-2]:
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            if kind != "bar":
                # change https://github.com/pandas-dev/pandas/issues/26714
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())

        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots_no_share_x(self, kind):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over kind. With sharex disabled, every subplot
        # shows x axis, x tick labels (major and minor), x label and y tick labels.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(kind=kind, subplots=True, sharex=False)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])
    def test_subplots_no_legend(self, kind):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over kind. With legend disabled for subplots, no
        # axes has a legend.
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(kind=kind, subplots=True, legend=False)
        for ax in axes:
            assert ax.get_legend() is None

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind (line, area). Time-indexed frame with subplots and
        # shared x (GH 7801): same visibility pattern as for the first test (only
        # bottom shows x labels) and x tick rotation zero.
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)

        axes = df.plot(kind=kind, subplots=True, sharex=True)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))

        for ax in axes[:-2]:
            # GH 7801
            _check_visible(ax.xaxis)  # xaxis must be visible for grid
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            _check_visible(ax.xaxis.get_label(), visible=False)
            _check_visible(ax.get_yticklabels())

        _check_visible(axes[-1].xaxis)
        _check_visible(axes[-1].get_xticklabels())
        _check_visible(axes[-1].get_xticklabels(minor=True))
        _check_visible(axes[-1].xaxis.get_label())
        _check_visible(axes[-1].get_yticklabels())
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries_rot(self, kind):
        # [FUNCTION DOC]
        # Parametrized over kind (line, area). Time series subplots without sharing, with
        # rotation 45 and font size 7: all x elements visible and tick properties match.
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
        for ax in axes:
            _check_visible(ax.xaxis)
            _check_visible(ax.get_xticklabels())
            _check_visible(ax.get_xticklabels(minor=True))
            _check_visible(ax.xaxis.get_label())
            _check_visible(ax.get_yticklabels())
            _check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)

    @pytest.mark.parametrize(
        "col", ["numeric", "timedelta", "datetime_no_tz", "datetime_all_tz"]
    )
    def test_subplots_timeseries_y_axis(self, col):
        # [FUNCTION DOC]
        # GH16953. Parametrized over column type (numeric, timedelta, naive datetime,
        # tz-aware datetime). Plotting that column on the y axis yields line data equal
        # to the column values.
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "timedelta": [
                pd.Timedelta(-10, unit="s"),
                pd.Timedelta(10, unit="m"),
                pd.Timedelta(10, unit="h"),
            ],
            "datetime_no_tz": [
                pd.to_datetime("2017-08-01 00:00:00"),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
            "datetime_all_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00", utc=True),
                pd.to_datetime("2017-08-02 00:00:00", utc=True),
            ],
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)

        ax = testdata.plot(y=col)
        result = ax.get_lines()[0].get_data()[1]
        expected = testdata[col].values
        assert (result == expected).all()

    def test_subplots_timeseries_y_text_error(self):
        # [FUNCTION DOC]
        # GH16953. Plotting a text column as y raises a type error saying there is no
        # numeric data to plot.
        # GH16953
        data = {
            "numeric": np.array([1, 2, 5]),
            "text": ["This", "should", "fail"],
        }
        testdata = DataFrame(data)
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            testdata.plot(y="text")

    @pytest.mark.xfail(reason="not support for period, categorical, datetime_mixed_tz")
    def test_subplots_timeseries_y_axis_not_supported(self):
        # [FUNCTION DOC]
        # Expected to fail (xfail: unsupported period, categorical and mixed-timezone
        # data on the y axis). Plots those y types and asserts the line data equals the
        # column values.
        """
        This test will fail for:
            period:
                since period isn't yet implemented in ``select_dtypes``
                and because it will need a custom value converter +
                tick formatter (as was done for x-axis plots)

            categorical:
                 because it will need a custom value converter +
                 tick formatter (also doesn't work for x-axis, as of now)

            datetime_mixed_tz:
                because of the way how pandas handles ``Series`` of
                ``datetime`` objects with different timezone,
                generally converting ``datetime`` objects in a tz-aware
                form could help with this problem
        """
        data = {
            "numeric": np.array([1, 2, 5]),
            "period": [
                pd.Period("2017-08-01 00:00:00", freq="h"),
                pd.Period("2017-08-01 02:00", freq="h"),
                pd.Period("2017-08-02 00:00:00", freq="h"),
            ],
            "categorical": pd.Categorical(
                ["c", "b", "a"], categories=["a", "b", "c"], ordered=False
            ),
            "datetime_mixed_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
        }
        testdata = DataFrame(data)
        ax_period = testdata.plot(x="numeric", y="period")
        assert (
            ax_period.get_lines()[0].get_data()[1] == testdata["period"].values
        ).all()
        ax_categorical = testdata.plot(x="numeric", y="categorical")
        assert (
            ax_categorical.get_lines()[0].get_data()[1]
            == testdata["categorical"].values
        ).all()
        ax_datetime_mixed_tz = testdata.plot(x="numeric", y="datetime_mixed_tz")
        assert (
            ax_datetime_mixed_tz.get_lines()[0].get_data()[1]
            == testdata["datetime_mixed_tz"].values
        ).all()

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 2), (2, 2)],
            [(-1, 2), (2, 2)],
            [(2, -1), (2, 2)],
            [(1, 4), (1, 4)],
            [(-1, 4), (1, 4)],
            [(4, -1), (4, 1)],
        ],
    )
    def test_subplots_layout_multi_column(self, layout, exp_layout):
        # [FUNCTION DOC]
        # GH 6667. Parametrized over (requested layout, expected layout). With three
        # columns, a requested layout (with -1 meaning inferred) results in the
        # expected shape of axes array and layout.
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(subplots=True, layout=layout)
        _check_axes_shape(axes, axes_num=3, layout=exp_layout)
        assert axes.shape == exp_layout

    def test_subplots_layout_multi_column_error(self):
        # [FUNCTION DOC]
        # GH 6667. A layout too small for three subplots raises a value error giving
        # the size; a layout without positive dimension raises a value error.
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        msg = "Layout of 1x1 must be larger than required size 3"

        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(1, 1))

        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(-1, -1))

    @pytest.mark.parametrize(
        "kwargs, expected_axes_num, expected_layout, expected_shape",
        [
            ({}, 1, (1, 1), (1,)),
            ({"layout": (3, 3)}, 1, (3, 3), (3, 3)),
        ],
    )
    def test_subplots_layout_single_column(
        self, kwargs, expected_axes_num, expected_layout, expected_shape
    ):
        # [FUNCTION DOC]
        # GH 6667. Parametrized over keywords. For one column with subplots: default
        # gives one axes with shape of one; a (3, 3) layout gives a 3 by 3 axes array
        # with only one axes in use.
        # GH 6667
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )
        axes = df.plot(subplots=True, **kwargs)
        _check_axes_shape(
            axes,
            axes_num=expected_axes_num,
            layout=expected_layout,
        )
        assert axes.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("idx", [range(5), date_range("1/1/2000", periods=5)])
    def test_subplots_warnings(self, idx):
        # [FUNCTION DOC]
        # Marked slow. GH 9464. Parametrized over integer and date index. Plotting with
        # subplots and a layout larger than needed emits no warning.
        # GH 9464
        with tm.assert_produces_warning(None):
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=idx)
            df.plot(subplots=True, layout=(3, 2))

    def test_subplots_multiple_axes(self):
        # [FUNCTION DOC]
        # GH 5353, 6970, 7069. Draws the same frame twice on rows of a 2x3 grid of
        # existing axes with subplots: each call returns three axes in the same figure.
        # GH 5353, 6970, GH 7069
        fig, axes = mpl.pyplot.subplots(2, 3)
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        # draw on second row
        returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        assert returned.shape == (3,)
        assert returned[0].figure is fig
        _check_axes_shape(axes, axes_num=6, layout=(2, 3))

    def test_subplots_multiple_axes_error(self):
        # [FUNCTION DOC]
        # GH 5353, 6970, 7069. Passing six axes for three subplots raises a value
        # error saying the number of passed axes must be 3.
        # GH 5353, 6970, GH 7069
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        msg = "The number of passed axes must be 3, the same as the output plot"
        _, axes = mpl.pyplot.subplots(2, 3)

        with pytest.raises(ValueError, match=msg):
            # pass different number of axes from required
            df.plot(subplots=True, ax=axes)

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 1), (2, 2)],
            [(2, -1), (2, 2)],
            [(-1, 2), (2, 2)],
        ],
    )
    def test_subplots_multiple_axes_2_dim(self, layout, exp_layout):
        # [FUNCTION DOC]
        # Parametrized over (layout, expected layout). Passing a 2x2 array of axes with
        # an invalid layout emits a user warning that the layout keyword is ignored and
        # returns the four axes in a flat array.
        # GH 5353, 6970, GH 7069
        # pass 2-dim axes and invalid layout
        # invalid layout should not affect to input and return value
        # (show warning is tested in
        # TestDataFrameGroupByPlots.test_grouped_box_multiple_axes
        _, axes = mpl.pyplot.subplots(2, 2)
        df = DataFrame(
            np.random.default_rng(2).random((10, 4)),
            index=list(string.ascii_letters[:10]),
        )
        with tm.assert_produces_warning(UserWarning, match="layout keyword is ignored"):
            returned = df.plot(
                subplots=True, ax=axes, layout=layout, sharex=False, sharey=False
            )
            _check_axes_shape(returned, axes_num=4, layout=exp_layout)
            assert returned.shape == (4,)

    def test_subplots_multiple_axes_single_col(self):
        # [FUNCTION DOC]
        # Single column plotted into a one-element list of axes with subplots: returns
        # a shape-one array.
        # GH 5353, 6970, GH 7069
        # single column
        _, axes = mpl.pyplot.subplots(1, 1)
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )

        axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        assert axes.shape == (1,)

    def test_subplots_ts_share_axes(self):
        # [FUNCTION DOC]
        # GH 3964. Nine time series drawn on a shared 3x3 grid of axes: only the bottom
        # row shows x tick labels and only the first column shows y tick labels.
        # GH 3964
        _, axes = mpl.pyplot.subplots(3, 3, sharex=True, sharey=True)
        mpl.pyplot.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)),
            index=date_range(start="2014-07-01", freq="ME", periods=10),
        )
        for i, ax in enumerate(axes.ravel()):
            df[i].plot(ax=ax, fontsize=5)

        # Rows other than bottom should not be visible
        for ax in axes[0:-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=False)

        # Bottom row should be visible
        for ax in axes[-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=True)

        # First column should be visible
        for ax in axes[[0, 1, 2], [0]].ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

        # Other columns should not be visible
        for ax in axes[[0, 1, 2], [1]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)
        for ax in axes[[0, 1, 2], [2]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)

    def test_subplots_sharex_axes_existing_axes(self):
        # [FUNCTION DOC]
        # GH 9158. Plotting two columns as subplots then a third on a secondary axis of
        # the first axes: the first axes keeps x labels hidden, second visible, all y
        # labels visible.
        # GH 9158
        d = {"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0], "C": [5, 1, 3, 4]}
        df = DataFrame(d, index=date_range("2014 10 11", "2014 10 14"))

        axes = df[["A", "B"]].plot(subplots=True)
        df["C"].plot(ax=axes[0], secondary_y=True)

        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        for ax in axes.ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

    def test_subplots_dup_columns(self):
        # [FUNCTION DOC]
        # GH 10962. Frame with duplicate column labels plotted as subplots: each axes
        # has legend label a and one line.
        # GH 10962
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        axes = df.plot(subplots=True)
        for ax in axes:
            _check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y(self):
        # [FUNCTION DOC]
        # GH 10962. As above with secondary y for the duplicated column: per-subplot
        # legend label has no right-axis suffix (suffix only without subplots), one line each.
        # GH 10962
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        axes = df.plot(subplots=True, secondary_y="a")
        for ax in axes:
            # (right) is only attached when subplots=False
            _check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y_no_subplot(self):
        # [FUNCTION DOC]
        # GH 10962. Duplicated columns with secondary y and no subplots: the main axes
        # has no lines, the right axes has five lines, and the legend has five labels
        # carrying the right-axis suffix.
        # GH 10962
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        ax = df.plot(secondary_y="a")
        _check_legend_labels(ax, labels=["a (right)"] * 5)
        assert len(ax.lines) == 0
        assert len(ax.right_ax.lines) == 5

    @pytest.mark.xfail(
        is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    def test_bar_log_no_subplots(self):
        # [FUNCTION DOC]
        # GH3254, GH3298. Non-strict xfail on Linux (rounding issues). Bar plot with a
        # log scale and grid: the y axis tick locations equal the expected decade
        # values, with the last tick dropped for matplotlib 3.11 and later (GH#65918).
        # GH3254, GH3298 matplotlib/matplotlib#1882, #1892
        # regressions in 1.2.1
        # matplotlib 3.11 no longer pads log-axis ticks a decade past the
        #  view limits, so the outermost tick is dropped (GH#65918)
        expected = (
            np.array([0.1, 1.0, 10.0, 100])
            if Version(mpl.__version__) < Version("3.11.0rc1")
            else np.array([0.1, 1.0, 10.0])
        )

        # no subplots
        df = DataFrame({"A": [3] * 5, "B": list(range(1, 6))}, index=range(5))
        ax = df.plot.bar(grid=True, log=True)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)

    @pytest.mark.xfail(
        is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    def test_bar_log_subplots(self):
        # [FUNCTION DOC]
        # Non-strict xfail on Linux. Log-scale bar plot with subplots: every subplot's
        # y tick locations equal the expected decade values (one fewer for matplotlib
        # 3.11 and later).
        # matplotlib 3.11 no longer pads log-axis ticks a decade past the
        #  view limits, so the outermost tick is dropped (GH#65918)
        expected = (
            np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 1e4])
            if Version(mpl.__version__) < Version("3.11.0rc1")
            else np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        )

        ax = DataFrame([Series([200, 300]), Series([300, 500])]).plot.bar(
            log=True, subplots=True
        )

        tm.assert_numpy_array_equal(ax[0].yaxis.get_ticklocs(), expected)
        tm.assert_numpy_array_equal(ax[1].yaxis.get_ticklocs(), expected)

    def test_boxplot_subplots_return_type_default(self, hist_df):
        # [FUNCTION DOC]
        # Box plot with subplots and no return type gives a series of axes keyed by
        # the plotted columns.
        df = hist_df

        # normal style: return_type=None
        result = df.plot.box(subplots=True)
        assert isinstance(result, Series)
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.parametrize("rt", ["dict", "axes", "both"])
    def test_boxplot_subplots_return_type(self, hist_df, rt):
        # [FUNCTION DOC]
        # Parametrized over return type (dict, axes, both). Box plot with subplots
        # returns the matching type per column (axes titles are not checked).
        df = hist_df
        returned = df.plot.box(return_type=rt, subplots=True)
        _check_box_return_type(
            returned,
            rt,
            expected_keys=["height", "weight", "category"],
            check_ax_title=False,
        )

    def test_df_subplots_patterns_minorticks(self):
        # [FUNCTION DOC]
        # GH 10657. Two shared subplots from existing axes of a time series: each axes
        # has one line and visible y labels; the first axes hides its x labels (major and
        # minor) and the second shows them.
        # GH 10657
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )

        # shared subplots
        _, axes = plt.subplots(2, 1, sharex=True)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of 1st ax must be hidden
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_1st_ax_hidden(self):
        # [FUNCTION DOC]
        # GH 10657. Unshared existing axes with sharex requested: a user warning that
        # sharex and sharey are ignored is emitted; result has first axes hiding x labels
        # and second showing them.
        # GH 10657
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )
        _, axes = plt.subplots(2, 1)
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        # xaxis of 1st ax must be hidden
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_not_shared(self):
        # [FUNCTION DOC]
        # GH 10657. Unshared existing axes with no sharing requested: all labels
        # (x major and minor, y) are visible.
        # GH 10657
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )
        # not shared
        _, axes = plt.subplots(2, 1)
        axes = df.plot(subplots=True, ax=axes)
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    def test_subplots_sharex_false(self):
        # [FUNCTION DOC]
        # GH 25160. With sharex disabled and data in different row ranges per column
        # (missing values), the two axes have different x ticks.
        # test when sharex is set to False, two plots should have different
        # labels, GH 25160
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        df.iloc[5:, 1] = np.nan
        df.iloc[:5, 0] = np.nan

        _, axs = mpl.pyplot.subplots(2, 1)
        df.plot.line(ax=axs, subplots=True, sharex=False)

        expected_ax1 = np.arange(4.5, 10, 0.5)
        expected_ax2 = np.arange(-0.5, 5, 0.5)

        tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
        tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)

    def test_subplots_constrained_layout(self, temp_file):
        # [FUNCTION DOC]
        # GH 25261. Takes the temp file fixture. Plots on a figure with constrained
        # layout (if supported by the installed matplotlib) and saves the figure to a
        # temp file; no warning may be emitted.
        # GH 25261
        idx = date_range(start="now", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        kwargs = {}
        if hasattr(mpl.pyplot.Figure, "get_constrained_layout"):
            kwargs["constrained_layout"] = True
        _, axes = mpl.pyplot.subplots(2, **kwargs)
        with tm.assert_produces_warning(None):
            df.plot(ax=axes[0])
            with temp_file.open(mode="wb") as path:
                mpl.pyplot.savefig(path)

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
    def test_xlabel_ylabel_dataframe_subplots(
        self, kind, index_name, old_label, new_label
    ):
        # [FUNCTION DOC]
        # GH 9093. Parametrized over index name with old and new labels and over kind
        # (line, area, bar). Subplots show no y label and the x label equal to the old
        # label (the index name or empty); passing x and y labels overrides them with
        # their string forms.
        # GH 9093
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        df.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name
        axes = df.plot(kind=kind, subplots=True)
        assert all(ax.get_ylabel() == "" for ax in axes)
        assert all(ax.get_xlabel() == old_label for ax in axes)

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        axes = df.plot(kind=kind, ylabel=new_label, xlabel=new_label, subplots=True)
        assert all(ax.get_ylabel() == str(new_label) for ax in axes)
        assert all(ax.get_xlabel() == str(new_label) for ax in axes)

    @pytest.mark.parametrize(
        "kwargs",
        [
            # stacked center
            {"kind": "bar", "stacked": True},
            {"kind": "bar", "stacked": True, "width": 0.9},
            {"kind": "barh", "stacked": True},
            {"kind": "barh", "stacked": True, "width": 0.9},
            # center
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": False, "width": 0.9},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": False, "width": 0.9},
            # subplots center
            {"kind": "bar", "subplots": True},
            {"kind": "bar", "subplots": True, "width": 0.9},
            {"kind": "barh", "subplots": True},
            {"kind": "barh", "subplots": True, "width": 0.9},
            # align edge
            {"kind": "bar", "stacked": True, "align": "edge"},
            {"kind": "bar", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": True, "align": "edge"},
            {"kind": "barh", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "bar", "stacked": False, "align": "edge"},
            {"kind": "bar", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": False, "align": "edge"},
            {"kind": "barh", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "bar", "subplots": True, "align": "edge"},
            {"kind": "bar", "subplots": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "subplots": True, "align": "edge"},
            {"kind": "barh", "subplots": True, "width": 0.9, "align": "edge"},
        ],
    )
    def test_bar_align_multiple_columns(self, kwargs):
        # [FUNCTION DOC]
        # GH2157. Parametrized over many keyword sets (bar and barh; stacked, side by
        # side or subplots; default or wider width; center or edge alignment). Checks
        # bar alignment for a two-column frame.
        # GH2157
        df = DataFrame({"A": [3] * 5, "B": list(range(5))}, index=range(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": True},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": True},
            {"kind": "bar", "subplots": True},
            {"kind": "barh", "subplots": True},
        ],
    )
    def test_bar_align_single_column(self, kwargs):
        # [FUNCTION DOC]
        # Parametrized over bar and barh with stacked, unstacked or subplots. Checks bar
        # alignment for a one-column random frame.
        df = DataFrame(np.random.default_rng(2).standard_normal(5))
        self._check_bar_alignment(df, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": True},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": True},
            {"kind": "bar", "subplots": True},
            {"kind": "barh", "subplots": True},
        ],
    )
    def test_bar_barwidth_position(self, kwargs):
        # [FUNCTION DOC]
        # Parametrized as above. Checks bar alignment with width 0.9 and position 0.2
        # for a five-column random frame.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        self._check_bar_alignment(df, width=0.9, position=0.2, **kwargs)

    @pytest.mark.parametrize("w", [1, 1.0])
    def test_bar_barwidth_position_int(self, w):
        # [FUNCTION DOC]
        # GH 12979. Parametrized over integer and float width 1. Stacked bar plot: tick
        # locations are the five integers, the x limits are -0.75 to 4.75, and the first
        # and last bar left edges are -0.5 and 3.5.
        # GH 12979
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(stacked=True, width=w)
        ticks = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4]))
        assert ax.get_xlim() == (-0.75, 4.75)
        # check left-edge of bars
        assert ax.patches[0].get_x() == -0.5
        assert ax.patches[-1].get_x() == 3.5

    @pytest.mark.parametrize(
        "kind, kwargs",
        [
            ["bar", {"stacked": True}],
            ["barh", {"stacked": False}],
            ["barh", {"stacked": True}],
            ["bar", {"subplots": True}],
            ["barh", {"subplots": True}],
        ],
    )
    def test_bar_barwidth_position_int_width_1(self, kind, kwargs):
        # [FUNCTION DOC]
        # GH 12979. Parametrized over kind and keywords. Checks bar alignment with
        # integer width 1.
        # GH 12979
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        self._check_bar_alignment(df, kind=kind, width=1, **kwargs)

    def _check_bar_alignment(
        self,
        df,
        kind="bar",
        stacked=False,
        subplots=False,
        align="center",
        width=0.5,
        position=0.5,
    ):
        # [FUNCTION DOC]
        # HELPER: verifies the bar position/alignment of a bar or horizontal bar plot
        # of the frame. Parameters: df is the frame; kind is bar or barh (default bar);
        # stacked, subplots (default false); align (center or edge, default center);
        # width (default 0.5); position (default 0.5). Plots with grid enabled, then for
        # each visible axes: the axis limits must extend 0.25 beyond the outermost bar
        # edges (GH 7498), tick locations must equal the integer range of the row count,
        # and the first tick must sit at the center of the first bar (center
        # alignment) or at its edge (edge alignment), computing the center from the bar
        # width, the position and, for unstacked bars, the number of columns. Raises a
        # value error for an unsupported kind or alignment. Returns the flattened
        # visible axes.
        axes = df.plot(
            kind=kind,
            stacked=stacked,
            subplots=subplots,
            align=align,
            width=width,
            position=position,
            grid=True,
        )

        axes = _flatten_visible(axes)

        for ax in axes:
            if kind == "bar":
                axis = ax.xaxis
                ax_min, ax_max = ax.get_xlim()
                min_edge = min(p.get_x() for p in ax.patches)
                max_edge = max(p.get_x() + p.get_width() for p in ax.patches)
            elif kind == "barh":
                axis = ax.yaxis
                ax_min, ax_max = ax.get_ylim()
                min_edge = min(p.get_y() for p in ax.patches)
                max_edge = max(p.get_y() + p.get_height() for p in ax.patches)
            else:
                raise ValueError

            # GH 7498
            # compare margins between lim and bar edges
            tm.assert_almost_equal(ax_min, min_edge - 0.25)
            tm.assert_almost_equal(ax_max, max_edge + 0.25)

            p = ax.patches[0]
            if kind == "bar" and (stacked is True or subplots is True):
                edge = p.get_x()
                center = edge + p.get_width() * position
            elif kind == "bar" and stacked is False:
                center = p.get_x() + p.get_width() * len(df.columns) * position
                edge = p.get_x()
            elif kind == "barh" and (stacked is True or subplots is True):
                center = p.get_y() + p.get_height() * position
                edge = p.get_y()
            elif kind == "barh" and stacked is False:
                center = p.get_y() + p.get_height() * len(df.columns) * position
                edge = p.get_y()
            else:
                raise ValueError

            # Check the ticks locates on integer
            assert (axis.get_ticklocs() == np.arange(len(df))).all()

            if align == "center":
                # Check whether the bar locates on center
                tm.assert_almost_equal(axis.get_ticklocs()[0], center)
            elif align == "edge":
                # Check whether the bar's edge starts from the tick
                tm.assert_almost_equal(axis.get_ticklocs()[0], edge)
            else:
                raise ValueError

        return axes
