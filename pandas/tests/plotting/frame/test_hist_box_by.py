# [MODULE DOC]
# Tests for the by argument of histogram and box plot kinds of the frame
# plot accessor (GH 15079): titles, legends, tick labels, subplot layouts,
# errors for empty or invalid keys or layouts, shared axes and figure size.
# Public contents are the local hist_df fixture and the TestHistWithBy and
# TestBoxWithBy classes. Skipped if matplotlib is missing (import-or-skip at
# module level). No other import-time side effects.

import re

import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_plot_works,
    get_x_axis,
    get_y_axis,
)

pytest.importorskip("matplotlib")


@pytest.fixture
def hist_df():
    # [FUNCTION DOC]
    # Fixture returning a frame with 30 rows: two random normal columns A and B and
    # two random categorical columns C and D with values a, b, c (fixed seed).
    # KNOWN ISSUE: this fixture has the same name as a fixture in the plotting
    # conftest module but returns different data and shadows that one for this
    # module.
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 2)), columns=["A", "B"]
    )
    df["C"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    df["D"] = np.random.default_rng(2).choice(["a", "b", "c"], 30)
    return df


class TestHistWithBy:
    # [CLASS DOC]
    # Tests for histograms drawn with the by argument.
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            ("C", "A", ["a", "b", "c"], [["A"]] * 3),
            ("C", ["A", "B"], ["a", "b", "c"], [["A", "B"]] * 3),
            ("C", None, ["a", "b", "c"], [["A", "B"]] * 3),
            (
                ["C", "D"],
                "A",
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A"]] * 3,
            ),
            (
                ["C", "D"],
                ["A", "B"],
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A", "B"]] * 3,
            ),
            (
                ["C", "D"],
                None,
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A", "B"]] * 3,
            ),
        ],
    )
    def test_hist_plot_by_argument(self, by, column, titles, legends, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Marked slow; parametrized over by (one or two columns), column
        # (single, list or none), expected titles and legend texts. Plots histograms
        # with default axes, expects one axes per group (titles are the group labels,
        # or tuple-style labels for two keys) and legends listing the plotted columns.
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.hist, column=column, by=by, default_axes=True
        )
        result_titles = [ax.get_title() for ax in axes]
        result_legends = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]

        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column, titles, legends",
        [
            (0, "A", ["a", "b", "c"], [["A"]] * 3),
            (0, None, ["a", "b", "c"], [["A", "B"]] * 3),
            (
                [0, "D"],
                "A",
                [
                    "(a, a)",
                    "(b, b)",
                    "(c, c)",
                ],
                [["A"]] * 3,
            ),
        ],
    )
    def test_hist_plot_by_0(self, by, column, titles, legends, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Parametrized by key (integer 0 or a list mixing 0 and a column
        # name). Renames column C to the integer 0 and checks titles and legends as in
        # the previous test, so integer column labels work as by keys.
        # GH 15079
        df = hist_df.copy()
        df = df.rename(columns={"C": 0})

        axes = _check_plot_works(df.plot.hist, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_legends = [
            [legend.get_text() for legend in ax.get_legend().texts] for ax in axes
        ]

        assert result_legends == legends
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ([], ["A", "B"]),
            ((), None),
            ((), ["A", "B"]),
        ],
    )
    def test_hist_plot_empty_list_string_tuple_by(self, by, column, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Parametrized over empty list or tuple by keys and column choices.
        # Plotting with such a by value must raise a value error saying no group keys
        # were passed.
        # GH 15079
        msg = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(
                hist_df.plot.hist, default_axes=True, column=column, by=by
            )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (2, 2), 3),
            ("C", "A", (2, 2), 3),
            (["C"], ["A"], (1, 3), 3),
            ("C", None, (3, 1), 3),
            ("C", ["A", "B"], (3, 1), 3),
            (["C", "D"], "A", (9, 1), 3),
            (["C", "D"], "A", (3, 3), 3),
            (["C", "D"], ["A"], (5, 2), 3),
            (["C", "D"], ["A", "B"], (9, 1), 3),
            (["C", "D"], None, (9, 1), 3),
            (["C", "D"], ["A", "B"], (5, 2), 3),
        ],
    )
    def test_hist_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Marked slow; parametrized over (by, column, layout, axes_num). The
        # plot is expected to emit a user warning (because the helper adds an axes,
        # GH 13188); the resulting axes number and layout shape are checked against
        # the given values.
        # GH 15079
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(
                hist_df.plot.hist, column=column, by=by, layout=layout
            )
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    def test_hist_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Parametrized over (message, by, layout): a layout too small for the
        # groups, a layout that is not a two-tuple, or a layout without a positive
        # dimension. Each raises a value error with the matching message.
        # GH 15079, test if error is raised when invalid layout is given

        with pytest.raises(ValueError, match=msg):
            hist_df.plot.hist(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.slow
    def test_axis_share_x_with_by(self, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Marked slow. With sharex true, the three axes share the x axis
        # (pairwise joined) and none shares the y axis.
        # GH 15079
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharex=True)

        # share x
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)
        assert get_x_axis(ax3).joined(ax1, ax3)
        assert get_x_axis(ax3).joined(ax2, ax3)

        # don't share y
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)
        assert not get_y_axis(ax3).joined(ax1, ax3)
        assert not get_y_axis(ax3).joined(ax2, ax3)

    @pytest.mark.slow
    def test_axis_share_y_with_by(self, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Marked slow. With sharey true, the three axes share the y axis and
        # none shares the x axis.
        # GH 15079
        ax1, ax2, ax3 = hist_df.plot.hist(column="A", by="C", sharey=True)

        # share y
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        assert get_y_axis(ax3).joined(ax1, ax3)
        assert get_y_axis(ax3).joined(ax2, ax3)

        # don't share x
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)
        assert not get_x_axis(ax3).joined(ax1, ax3)
        assert not get_x_axis(ax3).joined(ax2, ax3)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Parametrized over figsize. The three axes are drawn in a figure of
        # the requested size.
        # GH 15079
        axes = hist_df.plot.hist(column="A", by="C", figsize=figsize)
        _check_axes_shape(axes, axes_num=3, figsize=figsize)


class TestBoxWithBy:
    # [CLASS DOC]
    # Tests for box plots drawn with the by argument.
    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            ("C", "A", ["A"], [["a", "b", "c"]]),
            (
                ["C", "D"],
                "A",
                ["A"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ],
            ),
            ("C", ["A", "B"], ["A", "B"], [["a", "b", "c"]] * 2),
            (
                ["C", "D"],
                ["A", "B"],
                ["A", "B"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ]
                * 2,
            ),
            (["C"], None, ["A", "B"], [["a", "b", "c"]] * 2),
        ],
    )
    def test_box_plot_by_argument(self, by, column, titles, xticklabels, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Parametrized over (by, column, titles, xticklabels). Box plots with
        # default axes: one axes per plotted column titled by the column name, x tick
        # labels are the group labels (tuple-style for two keys).
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.box, default_axes=True, column=column, by=by
        )
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]

        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column, titles, xticklabels",
        [
            (0, "A", ["A"], [["a", "b", "c"]]),
            (
                [0, "D"],
                "A",
                ["A"],
                [
                    [
                        "(a, a)",
                        "(b, b)",
                        "(c, c)",
                    ]
                ],
            ),
            (0, None, ["A", "B"], [["a", "b", "c"]] * 2),
        ],
    )
    def test_box_plot_by_0(self, by, column, titles, xticklabels, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Same as the previous test with an integer 0 column label used as
        # a by key (alone or together with a name).
        # GH 15079
        df = hist_df.copy()
        df = df.rename(columns={"C": 0})

        axes = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [
            [label.get_text() for label in ax.get_xticklabels()] for ax in axes
        ]

        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize(
        "by, column",
        [
            ([], ["A"]),
            ((), "A"),
            ([], None),
            ((), ["A", "B"]),
        ],
    )
    def test_box_plot_with_none_empty_list_by(self, by, column, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Empty list or tuple by values raise a value error saying no group
        # keys were passed.
        # GH 15079
        msg = "No group keys passed"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, column, layout, axes_num",
        [
            (["C"], "A", (1, 1), 1),
            ("C", "A", (1, 1), 1),
            ("C", None, (2, 1), 2),
            ("C", ["A", "B"], (1, 2), 2),
            (["C", "D"], "A", (1, 1), 1),
            (["C", "D"], None, (1, 2), 2),
        ],
    )
    def test_box_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Marked slow; parametrized over (by, column, layout, axes_num).
        # Checks the number of axes and layout of grouped box plots.
        # GH 15079
        axes = _check_plot_works(
            hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout
        )
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize(
        "msg, by, layout",
        [
            ("larger than required size", ["C", "D"], (1, 1)),
            (re.escape("Layout must be a tuple of (rows, columns)"), "C", (1,)),
            ("At least one dimension of layout must be positive", "C", (-1, -1)),
        ],
    )
    def test_box_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        # [FUNCTION DOC]
        # GH 15079. A too small layout, a layout not a two-tuple, or a layout without
        # a positive dimension each raises a value error with the matching message.
        # GH 15079, test if error is raised when invalid layout is given

        with pytest.raises(ValueError, match=msg):
            hist_df.plot.box(column=["A", "B"], by=by, layout=layout)

    @pytest.mark.parametrize("figsize", [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        # [FUNCTION DOC]
        # GH 15079. Parametrized over figsize. A box plot by group yields one axes in a
        # figure of the requested size.
        # KNOWN ISSUE: the test name says histogram but it tests box plots, and it has
        # the same name as a test in the histogram class.
        # GH 15079
        axes = hist_df.plot.box(column="A", by="C", figsize=figsize)
        _check_axes_shape(axes, axes_num=1, figsize=figsize)
