"""Test cases for .boxplot method"""

# [MODULE DOC]
# Tests for the box plot methods of frames, series and group-by objects
# (legacy boxplot entry points and the box plot kind of the plot accessor):
# return types, axes limits and sharing, figure size, font size, colors and
# properties keywords, axis labels for horizontal/vertical orientation, grouped
# layouts, multiple axes and tick labels. Defines a module-level list of
# orientation keyword dictionaries chosen by the installed matplotlib version
# (older versions use the vert flag; versions from 3.10 use an orientation
# string) and a fixture over it. The hist_df fixture comes from the plotting
# conftest. Skipped if matplotlib is missing (import-or-skip at module level).

from __future__ import annotations

import itertools
import string

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    plotting,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_box_return_type,
    _check_plot_works,
    _check_ticks_props,
    _check_visible,
)
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


def _check_ax_limits(col, ax):
    # [FUNCTION DOC]
    # HELPER: asserts that an axes' y limits enclose a column's data. Parameters:
    # col is a series of numeric values; ax is the axes. Asserts the lower y
    # limit is at most the data minimum and the upper limit at least the maximum.
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()
    assert y_max >= col.max()


if Version(mpl.__version__) < Version("3.10"):
    verts: list[dict[str, bool | str]] = [{"vert": False}, {"vert": True}]
else:
    verts = [{"orientation": "horizontal"}, {"orientation": "vertical"}]


@pytest.fixture(params=verts)
def vert(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the version-dependent orientation keyword
    # dictionaries (vertical and horizontal variants). Yields one dictionary of
    # keyword arguments for box plot calls.
    return request.param


class TestDataFramePlots:
    # [CLASS DOC]
    # Tests of frame level box plotting (boxplot method and box kind).
    def test_stacked_boxplot_set_axis(self):
        # [FUNCTION DOC]
        # GH2980. A stacked bar plot of integer data: x tick labels are the index values
        # in order; after setting ticks every tenth position and redrawing, the labels
        # are those positions.
        # KNOWN ISSUE: despite its name this test exercises stacked bar plots, not box
        # plots.
        # GH2980
        n = 30
        df = DataFrame(
            {
                "Clinical": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Confirmed": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Discarded": np.random.default_rng(2).choice([0, 1, 2, 3], n),
            },
            index=np.arange(0, n),
        )
        ax = df.plot(kind="bar", stacked=True)
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()
        ax.set_xticks(np.arange(0, n, 10))
        plt.draw()  # Update changes
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(
            np.arange(0, n, 10)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs, warn",
        [
            [{"return_type": "dict"}, None],
            [{"column": ["one", "two"]}, None],
            [{"column": ["one", "two"], "by": "indic"}, UserWarning],
            [{"column": ["one"], "by": ["indic", "indic2"]}, None],
            [{"by": "indic"}, UserWarning],
            [{"by": ["indic", "indic2"]}, UserWarning],
            [{"notch": 1}, None],
            [{"by": "indic", "notch": 1}, UserWarning],
        ],
    )
    def test_boxplot_legacy1(self, kwargs, warn):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over keyword arguments and expected warning
        # (user warning when grouping with by, since helper adds an axes: GH 13188).
        # Draws the frame boxplot with return type, column selection, grouping, or
        # notch options on random data with two extra grouping columns and checks the
        # plot works under the expected warning.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        df["indic"] = ["foo", "bar"] * 3
        df["indic2"] = ["foo", "bar", "foo"] * 2

        # _check_plot_works can add an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            _check_plot_works(df.boxplot, **kwargs)

    def test_boxplot_legacy1_series(self):
        # [FUNCTION DOC]
        # Checks the module-level box plot function accepts a series and returns a
        # dict result.
        ser = Series(np.random.default_rng(2).standard_normal(6))
        _check_plot_works(plotting._core.boxplot, data=ser, return_type="dict")

    def test_boxplot_legacy2(self):
        # [FUNCTION DOC]
        # Box plot grouped by a column on a frame with two groups warns with a user
        # warning (axes added by the helper) and runs.
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.boxplot, by="X")

    def test_boxplot_legacy2_with_ax(self):
        # [FUNCTION DOC]
        # When an axes is given and only one axes is needed, the returned axes is the
        # supplied one.
        # KNOWN ISSUE: the asserted identity compares the axes' own axes attribute with
        # the result, which is trivially satisfied for any axes object.
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        # When ax is supplied and required number of axes is 1,
        # passed ax should be used:
        _, ax = mpl.pyplot.subplots()
        axes = df.boxplot("Col1", by="X", ax=ax)
        ax_axes = ax.axes
        assert ax_axes is axes

    def test_boxplot_legacy2_with_ax_return_type(self):
        # [FUNCTION DOC]
        # Group-by box plot with an axes supplied and return type axes: the returned
        # mapping for group A is the supplied axes.
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        fig, ax = mpl.pyplot.subplots()
        axes = df.groupby("Y").boxplot(ax=ax, return_type="axes")
        ax_axes = ax.axes
        assert ax_axes is axes["A"]

    def test_boxplot_legacy2_with_multi_col(self):
        # [FUNCTION DOC]
        # Box plot of multiple columns grouped by with an axes supplied emits a user
        # warning that the figure containing the passed axes is being cleared, and the
        # returned axes belong to the same figure.
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        # Multiple columns with an ax argument should use same figure
        fig, ax = mpl.pyplot.subplots()
        msg = "the figure containing the passed axes is being cleared"
        with tm.assert_produces_warning(UserWarning, match=msg):
            axes = df.boxplot(
                column=["Col1", "Col2"], by="X", ax=ax, return_type="axes"
            )
        assert axes["Col1"].get_figure() is fig

    def test_boxplot_legacy2_by_none(self):
        # [FUNCTION DOC]
        # With no grouping, the dict return type's lines (all values flattened)
        # account for every line drawn on the axes.
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        # When by is None, check that all relevant lines are present in the
        # dict
        _, ax = mpl.pyplot.subplots()
        d = df.boxplot(ax=ax, return_type="dict")
        lines = list(itertools.chain.from_iterable(d.values()))
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self, hist_df):
        # [FUNCTION DOC]
        # GH 12216. Box plot of a frame with no return type and no grouping returns an
        # axes.
        # GH 12216; return_type=None & by=None -> axes
        result = hist_df.boxplot()
        assert isinstance(result, mpl.pyplot.Axes)

    def test_boxplot_return_type_legacy(self):
        # [FUNCTION DOC]
        # API change from pull 7096. An invalid return type raises a value error saying
        # the return type must be one of axes, dict, both; the default returns an axes.
        # API change in https://github.com/pandas-dev/pandas/pull/7096

        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        msg = "return_type must be {'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type="NOT_A_TYPE")

        result = df.boxplot()
        _check_box_return_type(result, "axes")

    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_boxplot_return_type_legacy_return_type(self, return_type):
        # [FUNCTION DOC]
        # Parametrized over return type (dict, axes, both). Box plot with that return
        # type emits no warning and returns the matching type.
        # API change in https://github.com/pandas-dev/pandas/pull/7096

        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_boxplot_axis_limits(self, hist_df):
        # [FUNCTION DOC]
        # Box plots of two columns grouped by category (one row of axes): each axes' y
        # limits cover its column's data, and the second axes shares y with the first.
        df = hist_df.copy()
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        # One full row
        height_ax, weight_ax = df.boxplot(["height", "weight"], by="category")
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        assert weight_ax._sharey == height_ax

    def test_boxplot_axis_limits_two_rows(self, hist_df):
        # [FUNCTION DOC]
        # Box plots of three columns grouped by category lay out in two rows with one
        # empty axes: limits cover data; all three used axes share y with the first;
        # the unused fourth axes has no shared y.
        df = hist_df.copy()
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        # Two rows, one partial
        p = df.boxplot(["height", "weight", "age"], by="category")
        height_ax, weight_ax, age_ax = p[0, 0], p[0, 1], p[1, 0]
        dummy_ax = p[1, 1]

        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        _check_ax_limits(df["age"], age_ax)
        assert weight_ax._sharey == height_ax
        assert age_ax._sharey == height_ax
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self):
        # [FUNCTION DOC]
        # A frame with an all-missing column still plots with return type axes.
        df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)))
        df.loc[:, 0] = np.nan
        _check_plot_works(df.boxplot, return_type="axes")

    def test_figsize(self):
        # [FUNCTION DOC]
        # Box plot with figsize twelve by eight produces a figure of that size in
        # inches.
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["A", "B", "C", "D", "E"]
        )
        result = df.boxplot(return_type="axes", figsize=(12, 8))
        assert result.figure.bbox_inches.width == 12
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self):
        # [FUNCTION DOC]
        # Box plot with font size 16 has x and y tick labels of size 16.
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        _check_ticks_props(df.boxplot("a", fontsize=16), xlabelsize=16, ylabelsize=16)

    def test_boxplot_numeric_data(self):
        # [FUNCTION DOC]
        # GH 22799. The box kind of the plot accessor drops non-numeric columns
        # (dates, strings, timezone-aware dates, timedeltas), so only columns b and c
        # appear on the x axis.
        # GH 22799
        df = DataFrame(
            {
                "a": date_range("2012-01-01", periods=10),
                "b": np.random.default_rng(2).standard_normal(10),
                "c": np.random.default_rng(2).standard_normal(10) + 2,
                "d": date_range("2012-01-01", periods=10).astype(str),
                "e": date_range("2012-01-01", periods=10, tz="UTC"),
                "f": timedelta_range("1 days", periods=10),
            }
        )
        ax = df.plot(kind="box")
        assert [x.get_text() for x in ax.get_xticklabels()] == ["b", "c"]

    @pytest.mark.parametrize(
        "colors_kwd, expected",
        [
            (
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
            ),
            ({"boxes": "r"}, {"boxes": "r"}),
            ("r", {"boxes": "r", "whiskers": "r", "medians": "r", "caps": "r"}),
        ],
    )
    def test_color_kwd(self, colors_kwd, expected):
        # [FUNCTION DOC]
        # GH 26214. Parametrized over the color keyword (a full dict, a partial dict, or
        # a single color applying to all parts) and expected per-part colors. The first
        # line of each part with the expected key has the expected color.
        # GH: 26214
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        result = df.boxplot(color=colors_kwd, return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "scheme,expected",
        [
            (
                "dark_background",
                {
                    "boxes": "#8dd3c7",
                    "whiskers": "#8dd3c7",
                    "medians": "#bfbbd9",
                    "caps": "#8dd3c7",
                },
            ),
            (
                "default",
                {
                    "boxes": "#1f77b4",
                    "whiskers": "#1f77b4",
                    "medians": "#2ca02c",
                    "caps": "#1f77b4",
                },
            ),
        ],
    )
    def test_colors_in_theme(self, scheme, expected):
        # [FUNCTION DOC]
        # GH 40769. Parametrized over matplotlib style names (dark background and
        # default) with expected colors for boxes, whiskers, medians and caps. Applies
        # the style globally (side effect: changes global style until cleanup) and
        # checks the first element of each part has that color.
        # GH: 40769
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        plt.style.use(scheme)
        result = df.plot.box(return_type="dict")
        for k, v in expected.items():
            assert mpl.colors.same_color(result[k][0].get_color(), v)

    @pytest.mark.parametrize(
        "dict_colors, msg",
        [({"boxes": "r", "invalid_key": "r"}, "invalid key 'invalid_key'")],
    )
    def test_color_kwd_errors(self, dict_colors, msg):
        # [FUNCTION DOC]
        # GH 26214. A color dict containing an invalid key raises a value error
        # mentioning the invalid key.
        # GH: 26214
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        with pytest.raises(ValueError, match=msg):
            df.boxplot(color=dict_colors, return_type="dict")

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd(self, props, expected):
        # [FUNCTION DOC]
        # GH 30346. Parametrized over the property keyword (box, whisker, cap, median
        # properties) and the part it affects. The color given via that keyword is
        # applied to the first element of that part.
        # GH 30346
        df = DataFrame({k: np.random.default_rng(2).random(10) for k in "ABC"})
        kwd = {props: {"color": "C1"}}
        result = df.boxplot(return_type="dict", **kwd)

        assert result[expected][0].get_color() == "C1"

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_plot_xlabel_ylabel(self, vert):
        # [FUNCTION DOC]
        # Takes the orientation fixture; ignores a set-tick-labels user warning.
        # Passing x and y labels to the box kind sets them on the axes (labels are not
        # swapped by orientation).
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        xlabel, ylabel = "x", "y"
        ax = df.plot(kind="box", xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_plot_box(self, vert):
        # [FUNCTION DOC]
        # GH 54941. Two frames drawn on two side by side axes with shared y and the
        # orientation keywords: each axes shows the supplied x and y labels.
        # GH 54941
        rng = np.random.default_rng(2)
        df1 = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list("ABCD"))
        df2 = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list("ABCD"))

        xlabel, ylabel = "x", "y"
        _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
        df1.plot.box(ax=axs[0], xlabel=xlabel, ylabel=ylabel, **vert)
        df2.plot.box(ax=axs[1], xlabel=xlabel, ylabel=ylabel, **vert)
        for ax in axs:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_xlabel_ylabel(self, vert):
        # [FUNCTION DOC]
        # Takes the orientation fixture. The boxplot method with x and y labels sets
        # them on the axes.
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        xlabel, ylabel = "x", "y"
        ax = df.boxplot(xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_group_xlabel_ylabel(self, vert):
        # [FUNCTION DOC]
        # Takes the orientation fixture. The boxplot method grouped by a column with x
        # and y labels sets them on every subplot.
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        xlabel, ylabel = "x", "y"
        ax = df.boxplot(by="group", xlabel=xlabel, ylabel=ylabel, **vert)
        for subplot in ax:
            assert subplot.get_xlabel() == xlabel
            assert subplot.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_group_no_xlabel_ylabel(self, vert, request):
        # [FUNCTION DOC]
        # Takes the orientation fixture and a request object. Marked to ignore the
        # tick label warning. Expected to fail for horizontal orientation with matplotlib
        # 3.10 or later. When grouped by a column without labels, the label on the
        # categorical axis (x for vertical, y otherwise) equals the formatted group
        # column name list.
        if Version(mpl.__version__) >= Version("3.10") and vert == {
            "orientation": "horizontal"
        }:
            request.applymarker(
                pytest.mark.xfail(reason=f"{vert} fails starting with matplotlib 3.10")
            )
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        ax = df.boxplot(by="group", **vert)
        for subplot in ax:
            target_label = (
                subplot.get_xlabel()
                if vert in ({"vert": True}, {"orientation": "vertical"})
                else subplot.get_ylabel()
            )
            assert target_label == pprint_thing(["group"])


class TestDataFrameGroupByPlots:
    # [CLASS DOC]
    # Tests of box plots through group-by objects and grouped box plot layouts.
    def test_boxplot_legacy1(self, hist_df):
        # [FUNCTION DOC]
        # Group-by box plot with return type axes warns (user warning) and gives two
        # axes (one per gender) in a one-by-two layout.
        grouped = hist_df.groupby(by="gender")
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        _check_axes_shape(list(axes.values), axes_num=2, layout=(1, 2))

    def test_boxplot_legacy1_return_type(self, hist_df):
        # [FUNCTION DOC]
        # With subplots disabled, group-by box plot gives a single axes in a one-by-one
        # layout.
        grouped = hist_df.groupby(by="gender")
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_boxplot_legacy2(self):
        # [FUNCTION DOC]
        # Marked slow. Frame with a two-level index grouped by the second level (10
        # groups): box plot works with a user warning and yields 10 axes in a 4 by 3
        # layout.
        tuples = zip(string.ascii_letters[:10], range(10), strict=True)
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        grouped = df.groupby(level=1)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        _check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))

    @pytest.mark.slow
    def test_boxplot_legacy2_return_type(self):
        # [FUNCTION DOC]
        # Marked slow. Same data as before but with subplots disabled: one axes with a
        # one-by-one layout.
        tuples = zip(string.ascii_letters[:10], range(10), strict=True)
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        grouped = df.groupby(level=1)
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def test_grouped_plot_fignums(self):
        # [FUNCTION DOC]
        # Grouped plot makes one figure per group (two figures, two results); the
        # grouped box plot makes a single figure with two axes.
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        gb = df.groupby("gender")

        res = gb.plot()
        assert len(mpl.pyplot.get_fignums()) == 2
        assert len(res) == 2
        plt.close("all")

        res = gb.boxplot(return_type="axes")
        assert len(mpl.pyplot.get_fignums()) == 1
        assert len(res) == 2

    def test_grouped_plot_fignums_excluded_col(self):
        # [FUNCTION DOC]
        # Grouped histogram works when the grouping column is excluded from the plotted
        # columns (GH 5610).
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        # now works with GH 5610 as gender is excluded
        df.groupby("gender").hist()

    @pytest.mark.slow
    def test_grouped_box_return_type(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. The frame box plot grouped by a column with no return type gives
        # an array of axes; the return value check uses the plotted column names as keys.
        df = hist_df

        # old style: return_type=None
        result = df.boxplot(by="gender")
        assert isinstance(result, np.ndarray)
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    def test_grouped_box_return_type_groupby(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. The group-by box plot with return type dict returns a mapping
        # keyed by group labels.
        df = hist_df
        # now for groupby
        result = df.groupby("gender").boxplot(return_type="dict")
        _check_box_return_type(result, "dict", expected_keys=["Male", "Female"])

    @pytest.mark.slow
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_grouped_box_return_type_arg(self, hist_df, return_type):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over return type. Checks result types and keys both
        # for group-by box plots (keys are group labels) and the frame method with by
        # (keys are plotted columns).
        df = hist_df

        returned = df.groupby("classroom").boxplot(return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=["A", "B", "C"])

        returned = df.boxplot(by="classroom", return_type=return_type)
        _check_box_return_type(
            returned, return_type, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_grouped_box_return_type_arg_duplcate_cats(self, return_type):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over return type. Checks return types with a
        # frame whose grouping column repeats the categories, for the group-by and by
        # variants.
        # KNOWN ISSUE: the test name contains a typo ("duplcate").
        columns2 = "X B C D A".split()
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((6, 5)), columns=columns2
        )
        categories2 = "A B".split()
        df2["category"] = categories2 * 3

        returned = df2.groupby("category").boxplot(return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=categories2)

        returned = df2.boxplot(by="category", return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=columns2)

    @pytest.mark.slow
    def test_grouped_box_layout_too_small(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. Layout (1, 1) is too small for two plotted columns: raises a
        # value error saying the layout is smaller than the required size.
        df = hist_df

        msg = "Layout of 1x1 must be larger than required size 2"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(1, 1))

    @pytest.mark.slow
    def test_grouped_box_layout_needs_by(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. A layout without a by grouping raises a value error saying the
        # layout keyword is not supported when by is none.
        df = hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(
                column=["height", "weight", "category"],
                layout=(2, 1),
                return_type="dict",
            )

    @pytest.mark.slow
    def test_grouped_box_layout_positive_layout(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. A layout with no positive dimension raises a value error.
        df = hist_df
        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(-1, -1))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "gb_key, axes_num, rows",
        [["gender", 2, 1], ["category", 4, 2], ["classroom", 3, 2]],
    )
    def test_grouped_box_layout_positive_layout_axes(
        self, hist_df, gb_key, axes_num, rows
    ):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over (grouping key, axes count, rows). Group-by box
        # plot of one column warns with a user warning and draws the expected number of
        # axes in the expected number of rows with two columns.
        df = hist_df
        # _check_plot_works adds an ax so catch warning. see GH #13188 GH 6769
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(
                df.groupby(gb_key).boxplot, column="height", return_type="dict"
            )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=axes_num, layout=(rows, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "col, visible", [["height", False], ["weight", True], ["category", True]]
    )
    def test_grouped_box_layout_visible(self, hist_df, col, visible):
        # [FUNCTION DOC]
        # Marked slow. GH 5897. Parametrized over column and visibility. Plot with
        # three columns by gender on a 2x2 grid; checks that the x tick labels and x
        # axis label of each axes are visible except for the first plot (hidden).
        df = hist_df
        # GH 5897
        axes = df.boxplot(
            column=["height", "weight", "category"], by="gender", return_type="axes"
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))
        ax = axes[col]
        _check_visible(ax.get_xticklabels(), visible=visible)
        _check_visible([ax.xaxis.get_label()], visible=visible)

    @pytest.mark.slow
    def test_grouped_box_layout_shape(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. Group-by box plot of three columns by classroom produces three
        # axes in a two by two layout.
        df = hist_df
        df.groupby("classroom").boxplot(
            column=["height", "weight", "category"], return_type="dict"
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize("cols", [2, -1])
    def test_grouped_box_layout_works(self, hist_df, cols):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over column count (2 or minus one meaning
        # automatic). With a (3, cols) layout, the plot warns and draws four axes in a
        # 3 by 2 layout.
        df = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(
                df.groupby("category").boxplot,
                column="height",
                layout=(3, cols),
                return_type="dict",
            )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(3, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize("rows, res", [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_rows(self, hist_df, rows, res):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over (rows, resulting rows). Box plot by gender
        # with a (rows, 1) layout: three axes; a rows value of minus one is inferred to 3.
        df = hist_df
        df.boxplot(
            column=["height", "weight", "category"], by="gender", layout=(rows, 1)
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(res, 1))

    @pytest.mark.slow
    @pytest.mark.parametrize("cols, res", [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_cols_groupby(self, hist_df, cols, res):
        # [FUNCTION DOC]
        # Marked slow. Parametrized over (columns, resulting columns). Group-by box plot
        # with a (1, cols) layout draws three axes; minus one for columns is inferred.
        df = hist_df
        df.groupby("classroom").boxplot(
            column=["height", "weight", "category"],
            layout=(1, cols),
            return_type="dict",
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(1, res))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. GH 6970, 7069. Passing an array of four axes for a group-by box
        # plot emits a user warning that sharex and sharey are ignored, and the figure
        # has four axes in a 2x2 layout.
        # GH 6970, GH 7069
        df = hist_df

        # check warning to ignore sharex / sharey
        # this check should be done in the first function which
        # passes multiple axes to plot, hist or boxplot
        # location should be changed if other test is added
        # which has earlier alphabetical order
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            _, axes = mpl.pyplot.subplots(2, 2)
            df.groupby("category").boxplot(column="height", return_type="axes", ax=axes)
            _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_on_fig(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. GH 6970, 7069. With a figure of two rows of three axes, the
        # frame box plot with by drawn on the first row and the group-by version on the
        # second row each warn that sharex and sharey are ignored, return those axes
        # and keep them on the same figure.
        # GH 6970, GH 7069
        df = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            returned = df.boxplot(
                column=["height", "weight", "category"],
                by="gender",
                return_type="axes",
                ax=axes[0],
            )
        returned = np.array(list(returned.values))
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig

        # draw on second row
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            returned = df.groupby("classroom").boxplot(
                column=["height", "weight", "category"], return_type="axes", ax=axes[1]
            )
        returned = np.array(list(returned.values))
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_ax_error(self, hist_df):
        # [FUNCTION DOC]
        # Marked slow. GH 6970, 7069. Passing six axes where three groups are drawn
        # raises a value error that the number of axes must be 3 (with the sharex
        # warning emitted first).
        # GH 6970, GH 7069
        df = hist_df
        msg = "The number of passed axes must be 3, the same as the output plot"
        _, axes = mpl.pyplot.subplots(2, 3)
        with pytest.raises(ValueError, match=msg):
            # pass different number of axes from required
            with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
                axes = df.groupby("classroom").boxplot(ax=axes)

    def test_fontsize(self):
        # [FUNCTION DOC]
        # Box plot grouped via by with a font size sets the tick labels to size 16.
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]})
        _check_ticks_props(
            df.boxplot("a", by="b", fontsize=16), xlabelsize=16, ylabelsize=16
        )

    @pytest.mark.parametrize(
        "col, expected_xticklabel",
        [
            ("v", ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),
            (["v"], ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),
            ("v1", ["(a, v1)", "(b, v1)", "(c, v1)", "(d, v1)", "(e, v1)"]),
            (
                ["v", "v1"],
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
            (
                None,
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
        ],
    )
    def test_groupby_boxplot_subplots_false(self, col, expected_xticklabel):
        # [FUNCTION DOC]
        # GH 16748. Parametrized over column choices (single, list, none) and
        # expected x tick labels. With subplots disabled the group-by box plot puts all
        # groups and columns on one axes with tuple-formatted labels.
        # GH 16748
        df = DataFrame(
            {
                "cat": np.random.default_rng(2).choice(list("abcde"), 100),
                "v": np.random.default_rng(2).random(100),
                "v1": np.random.default_rng(2).random(100),
            }
        )
        grouped = df.groupby("cat")

        axes = _check_plot_works(
            grouped.boxplot, subplots=False, column=col, return_type="axes"
        )

        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    def test_groupby_boxplot_object(self, hist_df):
        # [FUNCTION DOC]
        # GH 43480. Group-by box plot on object-dtype data raises a value error saying
        # the box plot needs numerical columns.
        # GH 43480
        df = hist_df.astype("object")
        grouped = df.groupby("gender")
        msg = "boxplot method requires numerical columns, nothing to plot"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(grouped.boxplot, subplots=False)

    def test_boxplot_multiindex_column(self):
        # [FUNCTION DOC]
        # GH 16748. A box plot of selected columns from frame with a two-level column
        # index labels the x axis with tuple-formatted column names.
        # GH 16748
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays, strict=True))
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )

        col = [("bar", "one"), ("bar", "two")]
        axes = _check_plot_works(df.boxplot, column=col, return_type="axes")

        expected_xticklabel = ["(bar, one)", "(bar, two)"]
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    @pytest.mark.parametrize("group", ["X", ["X", "Y"]])
    def test_boxplot_multi_groupby_groups(self, group):
        # [FUNCTION DOC]
        # GH 14701. Parametrized over a single group key or a list of two keys. The
        # frame boxplot with by, the box kind with by, and the group-by boxplot all
        # work on random data with default axes.
        # GH 14701
        rows = 20
        df = DataFrame(
            np.random.default_rng(12).normal(size=(rows, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(np.repeat(["A", "B"], int(rows / 2)))
        df["Y"] = Series(np.tile(["C", "D"], int(rows / 2)))
        grouped = df.groupby(group)
        _check_plot_works(df.boxplot, by=group, default_axes=True)
        _check_plot_works(df.plot.box, by=group, default_axes=True)
        _check_plot_works(grouped.boxplot, default_axes=True)
