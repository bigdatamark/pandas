"""Test cases for GroupBy.plot"""

# [MODULE DOC]
# Tests for plotting through group-by objects (line, histogram, scatter plots
# of series and frame groupings, float indexes, legends). Mostly smoke tests
# that expect no error. Skipped if matplotlib is missing (import-or-skip at
# module level). No other import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_legend_labels,
)

pytest.importorskip("matplotlib")


class TestDataFrameGroupByPlots:
    # [CLASS DOC]
    # Tests for plot and hist methods of grouped series and frames.
    def test_series_groupby_plotting_nominally_works(self):
        # [FUNCTION DOC]
        # Smoke test: plotting a series of random weights grouped by random gender
        # labels runs without error.
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)

        weight.groupby(gender).plot()

    def test_series_groupby_plotting_nominally_works_hist(self):
        # [FUNCTION DOC]
        # Smoke test: histogram of a series grouped by random labels runs.
        n = 10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        height.groupby(gender).hist()

    def test_series_groupby_plotting_nominally_works_alpha(self):
        # [FUNCTION DOC]
        # Regression for GH8733: grouped series plot accepts an alpha keyword.
        n = 10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        # Regression test for GH8733
        height.groupby(gender).plot(alpha=0.5)

    def test_plotting_with_float_index_works(self):
        # [FUNCTION DOC]
        # GH 7025. Grouped plot of a column in a frame with a repeated float index runs
        # without error.
        # GH 7025
        df = DataFrame(
            {
                "def": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "val": np.random.default_rng(2).standard_normal(9),
            },
            index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )

        df.groupby("def")["val"].plot()

    def test_plotting_with_float_index_works_apply(self):
        # [FUNCTION DOC]
        # GH 7025. Same data as the previous test but plotting through the grouped apply
        # with a function that plots each group.
        # GH 7025
        df = DataFrame(
            {
                "def": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "val": np.random.default_rng(2).standard_normal(9),
            },
            index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )
        df.groupby("def")["val"].apply(lambda x: x.plot())

    def test_hist_single_row(self):
        # [FUNCTION DOC]
        # GH10214. Histogram of one column by another column on a two-row frame with
        # explicit bin edges runs.
        # GH10214
        bins = np.arange(80, 100 + 2, 1)
        df = DataFrame({"Name": ["AAA", "BBB"], "ByCol": [1, 2], "Mark": [85, 89]})
        df["Mark"].hist(by=df["ByCol"], bins=bins)

    def test_hist_single_row_single_bycol(self):
        # [FUNCTION DOC]
        # GH10214. Same as previous test for a frame with a single row.
        # GH10214
        bins = np.arange(80, 100 + 2, 1)
        df = DataFrame({"Name": ["AAA"], "ByCol": [1], "Mark": [85]})
        df["Mark"].hist(by=df["ByCol"], bins=bins)

    def test_plot_submethod_works(self):
        # [FUNCTION DOC]
        # Plotting through the scatter accessor of a grouped frame, with x and y
        # column names, runs.
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})
        df.groupby("z").plot.scatter("x", "y")

    def test_plot_submethod_works_line(self):
        # [FUNCTION DOC]
        # Plotting through the line accessor of a grouped column selection runs.
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})
        df.groupby("z")["x"].plot.line()

    def test_plot_kwargs(self):
        # [FUNCTION DOC]
        # Grouped plot with the kind keyword set to scatter returns a result mapping
        # group labels to axes; the axes of group a holds exactly one collection (the
        # scatter points) (GH11805).
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})

        res = df.groupby("z").plot(kind="scatter", x="x", y="y")
        # check that a scatter plot is effectively plotted: the axes should
        # contain a PathCollection from the scatter plot (GH11805)
        assert len(res["a"].collections) == 1

    def test_plot_kwargs_scatter(self):
        # [FUNCTION DOC]
        # Same as previous test but using the scatter accessor; the axes for group a
        # has one collection.
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})
        res = df.groupby("z").plot.scatter(x="x", y="y")
        assert len(res["a"].collections) == 1

    @pytest.mark.parametrize("column, expected_axes_num", [(None, 2), ("b", 1)])
    def test_groupby_hist_frame_with_legend(self, column, expected_axes_num):
        # [FUNCTION DOC]
        # GH 6279. Parametrized over column (none or b) and expected number of axes.
        # Frame group-by histogram with legend true: each result has the expected axes
        # count in a single-row layout and legend labels matching columns a and b (or
        # only column b when selected).
        # GH 6279 - DataFrameGroupBy histogram can have a legend
        expected_layout = (1, expected_axes_num)
        expected_labels = column or [["a"], ["b"]]

        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        for axes in g.hist(legend=True, column=column):
            _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
            for ax, expected_label in zip(axes[0], expected_labels, strict=True):
                _check_legend_labels(ax, expected_label)

    @pytest.mark.parametrize("column", [None, "b"])
    def test_groupby_hist_frame_with_legend_raises(self, column):
        # [FUNCTION DOC]
        # GH 6279. Parametrized over column. A grouped frame histogram with both the
        # legend and a label keyword raises a value error saying both cannot be used.
        # GH 6279 - DataFrameGroupBy histogram with legend and label raises
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            g.hist(legend=True, column=column, label="d")

    def test_groupby_hist_series_with_legend(self):
        # [FUNCTION DOC]
        # GH 6279. Series group-by histogram with legend true gives one axes per call
        # with legend labels equal to the group names.
        # GH 6279 - SeriesGroupBy histogram can have a legend
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        for ax in g["a"].hist(legend=True):
            _check_axes_shape(ax, axes_num=1, layout=(1, 1))
            _check_legend_labels(ax, ["1", "2"])

    def test_groupby_hist_series_with_legend_raises(self):
        # [FUNCTION DOC]
        # GH 6279. Group-by histogram with legend and label given raises a value error
        # saying both cannot be used.
        # GH 6279 - SeriesGroupBy histogram with legend and label raises
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            g.hist(legend=True, label="d")
