"""Test cases for DataFrame.plot"""

# [MODULE DOC]
# Tests for color handling in frame plotting: the color and colormap keywords
# and style strings for line, bar, area, histogram, density, scatter, hexbin and
# box plots, default color cycle behavior, error cases and scatter color bar
# behavior. Defines one private helper to check box plot part colors and the
# class TestDataFrameColor. Skipped if matplotlib, its pyplot or color map
# modules are unavailable (import-or-skip at module level). No other import-time
# side effects.

import re

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_colors,
    _check_plot_works,
    _unpack_cycler,
)

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
cm = pytest.importorskip("matplotlib.cm")


def _check_colors_box(bp, box_c, whiskers_c, medians_c, caps_c="k", fliers_c=None):
    # [FUNCTION DOC]
    # HELPER: checks all parts of a box plot result dictionary. Parameters: bp is
    # the box plot dict with boxes, whiskers, medians, fliers and caps; the expected
    # line colors for boxes, whiskers and medians are required; caps_c defaults to
    # black; fliers_c defaults to black when none is given. For each part, every
    # artist must have the expected line color.
    if fliers_c is None:
        fliers_c = "k"
    _check_colors(bp["boxes"], linecolors=[box_c] * len(bp["boxes"]))
    _check_colors(bp["whiskers"], linecolors=[whiskers_c] * len(bp["whiskers"]))
    _check_colors(bp["medians"], linecolors=[medians_c] * len(bp["medians"]))
    _check_colors(bp["fliers"], linecolors=[fliers_c] * len(bp["fliers"]))
    _check_colors(bp["caps"], linecolors=[caps_c] * len(bp["caps"]))


class TestDataFrameColor:
    # [CLASS DOC]
    # Test class for the color keywords of DataFrame.plot.
    @pytest.mark.parametrize("color", list(range(10)))
    def test_mpl2_color_cycle_str(self, color):
        # [FUNCTION DOC]
        # GH 15516. Parametrized over the ten matplotlib cycle names like C0 to C9.
        # Plotting a frame with that color string works.
        # GH 15516
        color = f"C{color}"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        _check_plot_works(df.plot, color=color)

    def test_color_single_series_list(self):
        # [FUNCTION DOC]
        # GH 3486. A one-column frame plotted with a single-element color list works.
        # GH 3486
        df = DataFrame({"A": [1, 2, 3]})
        _check_plot_works(df.plot, color=["red"])

    @pytest.mark.parametrize("color", [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color):
        # [FUNCTION DOC]
        # GH 16695. Parametrized over RGB and RGBA tuples. Plotting x against y with a
        # tuple color works.
        # GH 16695
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        _check_plot_works(df.plot, x="x", y="y", color=color)

    def test_color_empty_string(self):
        # [FUNCTION DOC]
        # An empty string color raises a value error saying the color argument is
        # invalid.
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with pytest.raises(ValueError, match="Invalid color argument:"):
            df.plot(color="")

    def test_color_and_style_arguments(self):
        # [FUNCTION DOC]
        # Color and style keywords may be combined when style strings have no color
        # symbol: line styles and colors are applied. When the style strings contain
        # color symbols, a value error says to use one or the other.
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        # passing both 'color' and 'style' arguments should be allowed
        # if there is no color symbol in the style strings:
        ax = df.plot(color=["red", "black"], style=["-", "--"])
        # check that the linestyles are correctly set:
        linestyle = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ["-", "--"]
        # check that the colors are correctly set:
        color = [line.get_color() for line in ax.lines]
        assert color == ["red", "black"]
        # passing both 'color' and 'style' arguments should not be allowed
        # if there is a color symbol in the style strings:
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(color=["red", "black"], style=["k-", "r--"])

    @pytest.mark.parametrize(
        "color, expected",
        [
            ("green", ["green"] * 4),
            (["yellow", "red", "green", "blue"], ["yellow", "red", "green", "blue"]),
        ],
    )
    def test_color_and_marker(self, color, expected):
        # [FUNCTION DOC]
        # GH 21003. Parametrized over a single color and a list of colors. With a style
        # giving a marker and dashed line, line colors match the colors, all line
        # styles are dashed and all markers are the diamond.
        # GH 21003
        df = DataFrame(np.random.default_rng(2).random((7, 4)))
        ax = df.plot(color=color, style="d--")
        # check colors
        result = [i.get_color() for i in ax.lines]
        assert result == expected
        # check markers and linestyles
        assert all(i.get_linestyle() == "--" for i in ax.lines)
        assert all(i.get_marker() == "d" for i in ax.lines)

    def test_color_and_style(self):
        # [FUNCTION DOC]
        # Color and style given as dictionaries keyed by column name apply per line.
        color = {"g": "black", "h": "brown"}
        style = {"g": "-", "h": "--"}
        expected_color = ["black", "brown"]
        expected_style = ["-", "--"]
        df = DataFrame({"g": [1, 2], "h": [2, 3]}, index=[1, 2])
        ax = df.plot.line(color=color, style=style)
        color = [i.get_color() for i in ax.lines]
        style = [i.get_linestyle() for i in ax.lines]
        assert color == expected_color
        assert style == expected_style

    def test_bar_colors(self):
        # [FUNCTION DOC]
        # Bar plot of a five-column frame: every fifth patch (one per column) takes the
        # default color cycle colors.
        default_colors = _unpack_cycler(plt.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar()
        _check_colors(ax.patches[::5], facecolors=default_colors[:5])

    def test_bar_colors_custom(self):
        # [FUNCTION DOC]
        # Bar plot with a string of single-letter colors applies them to the columns.
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(color=custom_colors)
        _check_colors(ax.patches[::5], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_bar_colors_cmap(self, colormap):
        # [FUNCTION DOC]
        # Parametrized over colormap given by name and as object. Bar colors follow the
        # jet colormap sampled at five evenly spaced points.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax = df.plot.bar(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::5], facecolors=rgba_colors)

    def test_bar_colors_single_col(self):
        # [FUNCTION DOC]
        # A single-column frame with a named color string yields a bar of that color.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.bar(color="DodgerBlue")
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_bar_colors_green(self):
        # [FUNCTION DOC]
        # Bar plot with kind keyword and a single color string colors every column.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind="bar", color="green")
        _check_colors(ax.patches[::5], facecolors=["green"] * 5)

    def test_bar_user_colors(self):
        # [FUNCTION DOC]
        # Colors given as a column of color names work only when the y column is
        # specified: each bar takes the color from the matching row.
        df = DataFrame(
            {"A": range(4), "B": range(1, 5), "color": ["red", "blue", "blue", "red"]}
        )
        # This should *only* work when `y` is specified, else
        # we use one color per column
        ax = df.plot.bar(y="A", color=df["color"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self):
        # [FUNCTION DOC]
        # Issue 10611. A scatter plot with a color column (colorbar) must have the same
        # x axis tick label and label visibility as one without.
        # addressing issue #10611, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax1 = df.plot.scatter(x="A label", y="B label")
        ax2 = df.plot.scatter(x="A label", y="B label", c="C label")

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        assert vis1 == vis2

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        assert vis1 == vis2

        assert (
            ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()
        )

    def test_if_hexbin_xaxis_label_is_visible(self):
        # [FUNCTION DOC]
        # Issue 10678. A hexbin plot keeps its x axis tick labels and label visible.
        # addressing issue #10678, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax = df.plot.hexbin("A label", "B label", gridsize=12)
        assert all(vis.get_visible() for vis in ax.xaxis.get_minorticklabels())
        assert all(vis.get_visible() for vis in ax.xaxis.get_majorticklabels())
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self):
        # [FUNCTION DOC]
        # Two scatter plots with colorbars on adjacent axes: after tight layout, the
        # distance between the two parent axes equals the distance between their
        # colorbars (within tolerance).
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        fig, axes = plt.subplots(1, 2)
        df.plot.scatter("A label", "B label", c="C label", ax=axes[0])
        df.plot.scatter("A label", "B label", c="C label", ax=axes[1])
        plt.tight_layout()

        points = np.array([ax.get_position().get_points() for ax in fig.axes])
        axes_x_coords = points[:, :, 0]
        parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
        colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-7).all()

    @pytest.mark.parametrize("cmap", [None, "Greys"])
    def test_scatter_with_c_column_name_with_colors(self, cmap):
        # [FUNCTION DOC]
        # Issue 34316. Parametrized over cmap (none or Greys). Scatter with the color
        # argument naming a column of color names: three unique face colors in
        # red/green/blue order, no colorbar; a user warning is emitted when cmap is also
        # given.
        # https://github.com/pandas-dev/pandas/issues/34316

        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        df["species"] = ["r", "r", "g", "g", "b"]
        if cmap is not None:
            with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
                ax = df.plot.scatter(x=0, y=1, cmap=cmap, c="species")
        else:
            ax = df.plot.scatter(x=0, y=1, c="species", cmap=cmap)

        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 3  # r/g/b
        assert (
            np.unique(ax.collections[0].get_facecolor(), axis=0)
            == np.array(
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
            )  # r/g/b
        ).all()
        assert ax.collections[0].colorbar is None

    def test_scatter_with_c_column_name_without_colors(self):
        # [FUNCTION DOC]
        # Scatter with the color argument naming a column of non-color labels (four
        # categories) yields four unique face colors; with a mix of valid colors and an
        # invalid string yields three unique colors.
        # Given
        colors = ["NY", "MD", "MA", "CA"]
        color_count = 4  # 4 unique colors

        # When
        df = DataFrame(
            {
                "dataX": range(100),
                "dataY": range(100),
                "color": (colors[i % len(colors)] for i in range(100)),
            }
        )

        # Then
        ax = df.plot.scatter("dataX", "dataY", c="color")
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count

        # Given
        colors = ["r", "g", "not-a-color"]
        color_count = 3
        # Also, since not all are mpl-colors, points matching 'r' or 'g'
        # are not necessarily red or green

        # When
        df = DataFrame(
            {
                "dataX": range(100),
                "dataY": range(100),
                "color": (colors[i % len(colors)] for i in range(100)),
            }
        )

        # Then
        ax = df.plot.scatter("dataX", "dataY", c="color")
        assert len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == color_count

    def test_scatter_colors(self):
        # [FUNCTION DOC]
        # Giving both the color argument and the color keyword to scatter raises a type
        # error asking for exactly one of them.
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        with pytest.raises(TypeError, match="Specify exactly one of `c` and `color`"):
            df.plot.scatter(x="a", y="b", c="c", color="green")

    def test_scatter_colors_not_raising_warnings(self):
        # [FUNCTION DOC]
        # GH-53908. Scatter with a single color name as the color argument must emit no
        # warning, and the face color is one unique blue.
        # GH-53908. Do not raise UserWarning: No data for colormapping
        # provided via 'c'. Parameters 'cmap' will be ignored
        df = DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        with tm.assert_produces_warning(None):
            ax = df.plot.scatter(x="x", y="y", c="b")
            assert (
                len(np.unique(ax.collections[0].get_facecolor(), axis=0)) == 1
            )  # blue
            assert (
                np.unique(ax.collections[0].get_facecolor(), axis=0)
                == np.array([[0.0, 0.0, 1.0, 1.0]])
            ).all()  # blue

    def test_scatter_colors_default(self):
        # [FUNCTION DOC]
        # Scatter with a numeric color column and no explicit color: the first point's
        # face color is the first color of the default cycle.
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        ax = df.plot.scatter(x="a", y="b", c="c")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array(mpl.colors.ColorConverter.to_rgba(default_colors[0])),
        )

    def test_scatter_colors_white(self):
        # [FUNCTION DOC]
        # Scatter with the white color keyword has white face colors.
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        ax = df.plot.scatter(x="a", y="b", color="white")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array([1, 1, 1, 1], dtype=np.float64),
        )

    def test_scatter_colorbar_different_cmap(self):
        # [FUNCTION DOC]
        # GH 33389. Two scatter plots on one axes with different colormaps keep their
        # own colormap names.
        # GH 33389
        df = DataFrame({"x": [1, 2, 3], "y": [1, 3, 2], "c": [1, 2, 3]})
        df["x2"] = df["x"] + 1

        _, ax = plt.subplots()
        df.plot("x", "y", c="c", kind="scatter", cmap="cividis", ax=ax)
        df.plot("x2", "y", c="c", kind="scatter", cmap="magma", ax=ax)

        assert ax.collections[0].cmap.name == "cividis"
        assert ax.collections[1].cmap.name == "magma"

    def test_line_colors(self):
        # [FUNCTION DOC]
        # Line plot with a string of single-letter colors applies them to lines; a second
        # plot with same colors gives identical line colors after closing figures.
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

        plt.close("all")

        ax2 = df.plot(color=custom_colors)
        lines2 = ax2.get_lines()

        for l1, l2 in zip(ax.get_lines(), lines2, strict=True):
            assert l1.get_color() == l2.get_color()

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_line_colors_cmap(self, colormap):
        # [FUNCTION DOC]
        # Parametrized over colormap name and object. Line colors follow the jet
        # colormap sampled at the number of rows.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_line_colors_single_col(self):
        # [FUNCTION DOC]
        # A single-column frame with a named color string colors the line with it.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        ax = df.loc[:, [0]].plot(color="DodgerBlue")
        _check_colors(ax.lines, linecolors=["DodgerBlue"])

    def test_line_colors_single_color(self):
        # [FUNCTION DOC]
        # A single color string colors all five lines.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(color="red")
        _check_colors(ax.get_lines(), linecolors=["red"] * 5)

    def test_line_colors_hex(self):
        # [FUNCTION DOC]
        # GH 10299. Hex color strings are applied to lines in order.
        # GH 10299
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    def test_dont_modify_colors(self):
        # [FUNCTION DOC]
        # Plotting with a color list does not mutate the passed list (length stays 3).
        colors = ["r", "g", "b"]
        DataFrame(np.random.default_rng(2).random((10, 2))).plot(color=colors)
        assert len(colors) == 3

    def test_line_colors_and_styles_subplots(self):
        # [FUNCTION DOC]
        # GH 9894. Plot with subplots and no color: each subplot's line uses the
        # corresponding default cycle color.
        # GH 9894
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        axes = df.plot(subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("color", ["k", "green"])
    def test_line_colors_and_styles_subplots_single_color_str(self, color):
        # [FUNCTION DOC]
        # Parametrized over single color strings. With subplots all lines use that color.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(subplots=True, color=color)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[color])

    @pytest.mark.parametrize("color", ["rgcby", list("rgcby")])
    def test_line_colors_and_styles_subplots_custom_colors(self, color):
        # [FUNCTION DOC]
        # GH 9894. Parametrized over a string and list of color letters. With subplots
        # each subplot's line uses the corresponding color.
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(color=color, subplots=True)
        for ax, c in zip(axes, list(color), strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_colormap_hex(self):
        # [FUNCTION DOC]
        # GH 9894, GH 10299. With subplots and hex colors, each subplot's line takes
        # the corresponding color.
        # KNOWN ISSUE: the name mentions a colormap but no colormap is used.
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # GH 10299
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        axes = df.plot(color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors), strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("cmap", ["jet", cm.jet])
    def test_line_colors_and_styles_subplots_colormap_subplot(self, cmap):
        # [FUNCTION DOC]
        # GH 9894. Parametrized over colormap name and object. With subplots, each
        # line takes the jet colormap sample at the corresponding position.
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(colormap=cmap, subplots=True)
        for ax, c in zip(axes, rgba_colors, strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_single_col(self):
        # [FUNCTION DOC]
        # GH 9894. Single-column frame with a named color and subplots colors the line.
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        axes = df.loc[:, [0]].plot(color="DodgerBlue", subplots=True)
        _check_colors(axes[0].lines, linecolors=["DodgerBlue"])

    def test_line_colors_and_styles_subplots_single_char(self):
        # [FUNCTION DOC]
        # GH 9894. A single-character style with subplots colors each subplot's line
        # with that color.
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # single character style
        axes = df.plot(style="r", subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=["r"])

    def test_line_colors_and_styles_subplots_list_styles(self):
        # [FUNCTION DOC]
        # GH 9894. A list of single-character styles with subplots colors each subplot's
        # line with the corresponding letter.
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # list of styles
        styles = list("rgcby")
        axes = df.plot(style=styles, subplots=True)
        for ax, c in zip(axes, styles, strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_area_colors(self):
        # [FUNCTION DOC]
        # Area plot with custom colors: lines, filled polygons and legend handles use
        # the colors, and legend handle alpha is unset.
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).random((5, 5)))

        ax = df.plot.area(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)
        poly = [
            o
            for o in ax.get_children()
            if isinstance(o, mpl.collections.PolyCollection)
        ]
        _check_colors(poly, facecolors=custom_colors)

        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=custom_colors)

        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_poly(self):
        # [FUNCTION DOC]
        # Area plot with the jet colormap: lines, polygons and legend handles use jet
        # samples and legend alpha is unset.
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.area(colormap="jet")
        jet_colors = [mpl.cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        poly = [
            o
            for o in ax.get_children()
            if isinstance(o, mpl.collections.PolyCollection)
        ]
        _check_colors(poly, facecolors=jet_colors)

        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=jet_colors)
        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_stacked_false(self):
        # [FUNCTION DOC]
        # Area plot with stacked disabled and a colormap object: lines use the colors;
        # polygons and legend handles use the same colors with alpha 0.5, and each
        # handle's alpha is 0.5.
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        jet_colors = [mpl.cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # When stacked=False, alpha is set to 0.5
        ax = df.plot.area(colormap=mpl.cm.jet, stacked=False)
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        poly = [
            o
            for o in ax.get_children()
            if isinstance(o, mpl.collections.PolyCollection)
        ]
        jet_with_alpha = [(c[0], c[1], c[2], 0.5) for c in jet_colors]
        _check_colors(poly, facecolors=jet_with_alpha)

        handles, _ = ax.get_legend_handles_labels()
        linecolors = jet_with_alpha
        _check_colors(handles[: len(jet_colors)], linecolors=linecolors)
        for h in handles:
            assert h.get_alpha() == 0.5

    def test_hist_colors(self):
        # [FUNCTION DOC]
        # Histogram: every tenth patch (one per column) uses the default cycle colors.
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.hist()
        _check_colors(ax.patches[::10], facecolors=default_colors[:5])

    def test_hist_colors_single_custom(self):
        # [FUNCTION DOC]
        # Histogram with custom color letters uses them per column.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = "rgcby"
        ax = df.plot.hist(color=custom_colors)
        _check_colors(ax.patches[::10], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_hist_colors_cmap(self, colormap):
        # [FUNCTION DOC]
        # Parametrized over colormap name and object. Histogram colors follow jet
        # samples at five points.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.hist(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::10], facecolors=rgba_colors)

    def test_hist_colors_single_col(self):
        # [FUNCTION DOC]
        # Single-column histogram with a named color uses it.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.hist(color="DodgerBlue")
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_hist_colors_single_color(self):
        # [FUNCTION DOC]
        # Histogram with a single color string colors all columns.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind="hist", color="green")
        _check_colors(ax.patches[::10], facecolors=["green"] * 5)

    def test_kde_colors(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plot with custom colors colors lines accordingly.
        pytest.importorskip("scipy")
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).random((5, 5)))

        ax = df.plot.kde(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_kde_colors_cmap(self, colormap):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over colormap. Density lines follow jet samples.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.kde(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_kde_colors_and_styles_subplots(self):
        # [FUNCTION DOC]
        # Requires scipy. Density plots with subplots use the default color cycle, one
        # per subplot.
        pytest.importorskip("scipy")
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        axes = df.plot(kind="kde", subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("colormap", ["k", "red"])
    def test_kde_colors_and_styles_subplots_single_col_str(self, colormap):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over single color strings. Density subplots all
        # use that color.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(kind="kde", color=colormap, subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[colormap])

    def test_kde_colors_and_styles_subplots_custom_color(self):
        # [FUNCTION DOC]
        # Requires scipy. Density subplots with a color letter string use one letter per
        # subplot.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = "rgcby"
        axes = df.plot(kind="kde", color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors), strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_kde_colors_and_styles_subplots_cmap(self, colormap):
        # [FUNCTION DOC]
        # Requires scipy. Parametrized over colormap. Density subplots use jet samples
        # per subplot.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(kind="kde", colormap=colormap, subplots=True)
        for ax, c in zip(axes, rgba_colors, strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_kde_colors_and_styles_subplots_single_col(self):
        # [FUNCTION DOC]
        # Requires scipy. Single-column density plot with named color and subplots.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        axes = df.loc[:, [0]].plot(kind="kde", color="DodgerBlue", subplots=True)
        _check_colors(axes[0].lines, linecolors=["DodgerBlue"])

    def test_kde_colors_and_styles_subplots_single_char(self):
        # [FUNCTION DOC]
        # Requires scipy. Density subplots with a single-character style color each
        # subplot's line with it.
        # KNOWN ISSUE: a stale comment before the call mentions a list of styles though
        # only a single character style is used.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # list of styles
        # single character style
        axes = df.plot(kind="kde", style="r", subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=["r"])

    def test_kde_colors_and_styles_subplots_list(self):
        # [FUNCTION DOC]
        # Requires scipy. Density subplots with a list of single-character styles color
        # subplots in order.
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # list of styles
        styles = list("rgcby")
        axes = df.plot(kind="kde", style=styles, subplots=True)
        for ax, c in zip(axes, styles, strict=True):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_boxplot_colors(self):
        # [FUNCTION DOC]
        # Box plot with default colors: boxes, whiskers and caps use the first cycle
        # color; medians use the third; fliers black.
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp = df.plot.box(return_type="dict")
        _check_colors_box(
            bp,
            default_colors[0],
            default_colors[0],
            default_colors[2],
            default_colors[0],
        )

    def test_boxplot_colors_dict_colors(self):
        # [FUNCTION DOC]
        # Box plot with a dict of colors for boxes, whiskers, medians and caps and an
        # outlier symbol of red plus: parts take the dict colors and fliers are red.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        dict_colors = {
            "boxes": "#572923",
            "whiskers": "#982042",
            "medians": "#804823",
            "caps": "#123456",
        }
        bp = df.plot.box(color=dict_colors, sym="r+", return_type="dict")
        _check_colors_box(
            bp,
            dict_colors["boxes"],
            dict_colors["whiskers"],
            dict_colors["medians"],
            dict_colors["caps"],
            "r",
        )

    def test_boxplot_colors_default_color(self):
        # [FUNCTION DOC]
        # Box plot with partial color dict (whiskers and medians): those use the given
        # colors, the rest default.
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # partial colors
        dict_colors = {"whiskers": "c", "medians": "m"}
        bp = df.plot.box(color=dict_colors, return_type="dict")
        _check_colors_box(bp, default_colors[0], "c", "m", default_colors[0])

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_boxplot_colors_cmap(self, colormap):
        # [FUNCTION DOC]
        # Parametrized over colormap name and object. Box plot parts use jet samples at
        # three points (boxes, whiskers, caps the first; medians the last).
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp = df.plot.box(colormap=colormap, return_type="dict")
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, 3)]
        _check_colors_box(
            bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0]
        )

    def test_boxplot_colors_single(self):
        # [FUNCTION DOC]
        # A string color applies to all box parts except fliers.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # string color is applied to all artists except fliers
        bp = df.plot.box(color="DodgerBlue", return_type="dict")
        _check_colors_box(bp, "DodgerBlue", "DodgerBlue", "DodgerBlue", "DodgerBlue")

    def test_boxplot_colors_tuple(self):
        # [FUNCTION DOC]
        # A tuple color applies to all parts except fliers, which use the outlier
        # symbol color.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # tuple is also applied to all artists except fliers
        bp = df.plot.box(color=(0, 1, 0), sym="#123456", return_type="dict")
        _check_colors_box(bp, (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), "#123456")

    def test_boxplot_colors_invalid(self):
        # [FUNCTION DOC]
        # A box color dict with an invalid key raises a value error listing the allowed
        # keys.
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        msg = re.escape(
            "color dict contains invalid key 'xxxx'. The key must be either "
            "['boxes', 'whiskers', 'medians', 'caps']"
        )
        with pytest.raises(ValueError, match=msg):
            # Color contains invalid key results in ValueError
            df.plot.box(color={"boxes": "red", "xxxx": "blue"})

    def test_default_color_cycle(self):
        # [FUNCTION DOC]
        # Sets the global color cycle to four letter colors (side effect: modifies
        # global plot parameters) and checks a three-column line plot uses the first
        # three.
        import cycler

        colors = list("rgbk")
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        ax = df.plot()

        expected = _unpack_cycler(plt.rcParams)[:3]
        _check_colors(ax.get_lines(), linecolors=expected)

    def test_no_color_bar(self):
        # [FUNCTION DOC]
        # Hexbin plot with the colorbar option set to none has no colorbar.
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        ax = df.plot.hexbin(x="A", y="B", colorbar=None)
        assert ax.collections[0].colorbar is None

    def test_mixing_cmap_and_colormap_raises(self):
        # [FUNCTION DOC]
        # Giving both cmap and colormap to hexbin raises a type error asking for only
        # one.
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        msg = "Only specify one of `cmap` and `colormap`"
        with pytest.raises(TypeError, match=msg):
            df.plot.hexbin(x="A", y="B", cmap="YlGn", colormap="BuGn")

    def test_passed_bar_colors(self):
        # [FUNCTION DOC]
        # A bar plot with a custom listed colormap object uses its colors for the
        # three bars.
        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        colormap = mpl.colors.ListedColormap(color_tuples)
        barplot = DataFrame([[1, 2, 3]]).plot(kind="bar", cmap=colormap)
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_rcParams_bar_colors(self):
        # [FUNCTION DOC]
        # Inside a rc context setting the color cycle, bar plot colors match the cycle.
        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        with mpl.rc_context(rc={"axes.prop_cycle": mpl.cycler("color", color_tuples)}):
            barplot = DataFrame([[1, 2, 3]]).plot(kind="bar")
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_colors_of_columns_with_same_name(self):
        # [FUNCTION DOC]
        # Issue 11136. Two frames with a duplicate column label are concatenated;
        # each legend handle color equals the color of the corresponding line.
        # ISSUE 11136 -> https://github.com/pandas-dev/pandas/issues/11136
        # Creating a DataFrame with duplicate column labels and testing colors of them.
        df = DataFrame({"b": [0, 1, 0], "a": [1, 2, 3]})
        df1 = DataFrame({"a": [2, 4, 6]})
        df_concat = pd.concat([df, df1], axis=1)
        result = df_concat.plot()
        legend = result.get_legend()
        handles = legend.legend_handles
        for legend, line in zip(handles, result.lines, strict=True):
            assert legend.get_color() == line.get_color()

    def test_invalid_colormap(self):
        # [FUNCTION DOC]
        # Plotting with an unknown colormap name raises a value or key error saying the
        # value is invalid or not a known colormap.
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 2)), columns=["A", "B"]
        )
        msg = "(is not a valid value)|(is not a known colormap)"
        with pytest.raises((ValueError, KeyError), match=msg):
            df.plot(colormap="invalid_colormap")

    def test_dataframe_none_color(self):
        # [FUNCTION DOC]
        # GH51953. Passing no color gives the default cycle colors for lines.
        # GH51953
        df = DataFrame([[1, 2, 3]])
        ax = df.plot(color=None)
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:3]
        _check_colors(ax.get_lines(), linecolors=expected)
