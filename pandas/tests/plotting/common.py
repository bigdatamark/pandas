"""
Module consolidating common testing functions for checking plotting.
"""

# [MODULE DOC]
# Shared helper functions for the plotting tests: assertion helpers that
# inspect matplotlib axes, artists, legends, ticks, colors, scales, layout,
# error bars, box plot return types and grid settings, plus small accessors for
# shared axes and generators that drive a plotting function on one or two
# subplots. Almost everything here is a private helper imported by the other
# plotting test modules; the two public names are the shared-axis accessors and
# the plot return object validator. Matplotlib is imported lazily inside the
# functions, and only typing-only imports are used at module level, so
# importing this module does not require matplotlib. No import-time side
# effects.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.api import is_list_like

import pandas as pd
from pandas import Series
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _check_legend_labels(axes, labels=None, visible=True):
    # [FUNCTION DOC]
    # HELPER: assertion used by plotting tests. Checks legend labels of one axes or
    # a collection of axes (only visible axes are examined). Parameters: axes is
    # an axes object or list-like of them; labels is the list-like of expected
    # legend text (default none); visible says whether a legend is expected
    # (default true). Raises a value error if a legend is expected but no labels are
    # given. When visible, each axes must have a legend and its texts must equal the
    # labels; otherwise each axes must have no legend. Returns nothing.
    """
    Check each axes has expected legend labels

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    labels : list-like
        expected legend labels
    visible : bool
        expected legend visibility. labels are checked only when visible is
        True
    """
    if visible and (labels is None):
        raise ValueError("labels must be specified when visible is True")
    axes = _flatten_visible(axes)
    for ax in axes:
        if visible:
            assert ax.get_legend() is not None
            _check_text_labels(ax.get_legend().get_texts(), labels)
        else:
            assert ax.get_legend() is None


def _check_legend_marker(ax, expected_markers=None, visible=True):
    # [FUNCTION DOC]
    # HELPER: assertion for legend markers of a single axes. Parameters: ax is the
    # axes; expected_markers is the list of expected marker symbols (default none);
    # visible is whether a legend is expected (default true). Raises a value error
    # if visible and no markers given. When visible, asserts the marker of each
    # legend handle equals the expected list; otherwise asserts there is no legend.
    # KNOWN ISSUE: the docstring says labels are checked only when visible, but
    # the function checks markers, not labels.
    """
    Check ax has expected legend markers

    Parameters
    ----------
    ax : matplotlib Axes object
    expected_markers : list-like
        expected legend markers
    visible : bool
        expected legend visibility. labels are checked only when visible is
        True
    """
    if visible and (expected_markers is None):
        raise ValueError("Markers must be specified when visible is True")
    if visible:
        handles, _ = ax.get_legend_handles_labels()
        markers = [handle.get_marker() for handle in handles]
        assert markers == expected_markers
    else:
        assert ax.get_legend() is None


def _check_data(xp, rs):
    # [FUNCTION DOC]
    # HELPER: compares two axes line by line. Parameters xp (expected) and rs
    # (result) are axes. Asserts both have the same number of lines and that the
    # plotted x/y data of each corresponding pair of lines are almost equal.
    """
    Check each axes has identical lines

    Parameters
    ----------
    xp : matplotlib Axes object
    rs : matplotlib Axes object
    """
    xp_lines = xp.get_lines()
    rs_lines = rs.get_lines()

    assert len(xp_lines) == len(rs_lines)
    for xpl, rsl in zip(xp_lines, rs_lines, strict=True):
        xpdata = xpl.get_xydata()
        rsdata = rsl.get_xydata()
        tm.assert_almost_equal(xpdata, rsdata)


def _check_visible(collections, visible=True):
    # [FUNCTION DOC]
    # HELPER: checks the visibility of artists. Parameters: collections is a single
    # artist or a list-like of artists (a matplotlib collection counts as a single
    # artist); visible is the expected visibility (default true). Asserts each
    # artist's visibility equals the expectation.
    """
    Check each artist is visible or not

    Parameters
    ----------
    collections : matplotlib Artist or its list-like
        target Artist or its list or collection
    visible : bool
        expected visibility
    """
    from matplotlib.collections import Collection

    if not isinstance(collections, Collection) and not is_list_like(collections):
        collections = [collections]

    for patch in collections:
        assert patch.get_visible() == visible


def _check_patches_all_filled(axes: Axes | Sequence[Axes], filled: bool = True) -> None:
    # [FUNCTION DOC]
    # HELPER: checks fill state of patches. Parameters: axes is an axes or a
    # sequence of them (flattened to visible axes); filled is the expected fill
    # flag (default true). Asserts every patch on every visible axes has that fill.
    """
    Check for each artist whether it is filled or not

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    filled : bool
        expected filling
    """

    axes = _flatten_visible(axes)
    for ax in axes:
        for patch in ax.patches:
            assert patch.fill == filled


def _get_colors_mapped(series, colors):
    # [FUNCTION DOC]
    # HELPER for the color checks: given a series of grouping keys and a list of
    # colors, maps each distinct key (in order of first appearance) to a color with
    # pairing truncated to the shorter of the two, and returns a list with the
    # mapped color for every value of the series (a key without a color raises a
    # key error).
    unique = series.unique()
    # unique and colors length can be differed
    # depending on slice value
    mapped = dict(zip(unique, colors))
    return [mapped[v] for v in series.values]


def _check_colors(collections, linecolors=None, facecolors=None, mapping=None):
    # [FUNCTION DOC]
    # HELPER: checks line and face colors of artists. Parameters: collections is a
    # list-like of artists; linecolors and facecolors are optional lists of
    # expected colors with the same length as collections; mapping is an optional
    # series used as grouping key so that the color lists are expanded per group
    # (used for the Andrews curves, parallel coordinates and radviz tests) and then
    # truncated to the number of artists. Colors are compared after conversion to
    # RGBA tuples. Line colors: for line objects the line color, for polygon or line
    # collections the first edge color, otherwise the edge color. Face colors: for
    # collections the first face color, otherwise the face color. Raises assertion
    # errors on length or color mismatch.
    """
    Check each artist has expected line colors and face colors

    Parameters
    ----------
    collections : list-like
        list or collection of target artist
    linecolors : list-like which has the same length as collections
        list of expected line colors
    facecolors : list-like which has the same length as collections
        list of expected face colors
    mapping : Series
        Series used for color grouping key
        used for andrew_curves, parallel_coordinates, radviz test
    """
    from matplotlib import colors
    from matplotlib.collections import (
        Collection,
        LineCollection,
        PolyCollection,
    )
    from matplotlib.lines import Line2D

    conv = colors.ColorConverter
    if linecolors is not None:
        if mapping is not None:
            linecolors = _get_colors_mapped(mapping, linecolors)
            linecolors = linecolors[: len(collections)]

        assert len(collections) == len(linecolors)
        for patch, color in zip(collections, linecolors, strict=True):
            if isinstance(patch, Line2D):
                result = patch.get_color()
                # Line2D may contains string color expression
                result = conv.to_rgba(result)
            elif isinstance(patch, (PolyCollection, LineCollection)):
                result = tuple(patch.get_edgecolor()[0])
            else:
                result = patch.get_edgecolor()

            expected = conv.to_rgba(color)
            assert result == expected

    if facecolors is not None:
        if mapping is not None:
            facecolors = _get_colors_mapped(mapping, facecolors)
            facecolors = facecolors[: len(collections)]

        assert len(collections) == len(facecolors)
        for patch, color in zip(collections, facecolors, strict=True):
            if isinstance(patch, Collection):
                # returned as list of np.array
                result = patch.get_facecolor()[0]
            else:
                result = patch.get_facecolor()

            if isinstance(result, np.ndarray):
                result = tuple(result)

            expected = conv.to_rgba(color)
            assert result == expected


def _check_text_labels(texts, expected):
    # [FUNCTION DOC]
    # HELPER: checks text artists. Parameters: texts is a single text object or a
    # list-like of them; expected is a string (for single text) or list-like of the
    # same length. Asserts the text contents equal the expectation, including the
    # count for list inputs.
    """
    Check each text has expected labels

    Parameters
    ----------
    texts : matplotlib Text object, or its list-like
        target text, or its list
    expected : str or list-like which has the same length as texts
        expected text label, or its list
    """
    if not is_list_like(texts):
        assert texts.get_text() == expected
    else:
        labels = [t.get_text() for t in texts]
        assert len(labels) == len(expected)
        for label, e in zip(labels, expected, strict=True):
            assert label == e


def _check_ticks_props(axes, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None):
    # [FUNCTION DOC]
    # HELPER: checks tick label properties. Parameters: axes (axes or list-like);
    # xlabelsize, xrot, ylabelsize, yrot are optional expected font sizes and
    # rotations for x and y tick labels (none means not checked). For each visible
    # axes, collects the major tick labels and also the minor ones unless the minor
    # formatter is the null formatter (in which case rotation and size are not
    # kept), then asserts each label's size or rotation is almost equal to the
    # expected value.
    """
    Check each axes has expected tick properties

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xlabelsize : number
        expected xticks font size
    xrot : number
        expected xticks rotation
    ylabelsize : number
        expected yticks font size
    yrot : number
        expected yticks rotation
    """
    from matplotlib.ticker import NullFormatter

    axes = _flatten_visible(axes)
    for ax in axes:
        if xlabelsize is not None or xrot is not None:
            if isinstance(ax.xaxis.get_minor_formatter(), NullFormatter):
                # If minor ticks has NullFormatter, rot / fontsize are not
                # retained
                labels = ax.get_xticklabels()
            else:
                labels = ax.get_xticklabels() + ax.get_xticklabels(minor=True)

            for label in labels:
                if xlabelsize is not None:
                    tm.assert_almost_equal(label.get_fontsize(), xlabelsize)
                if xrot is not None:
                    tm.assert_almost_equal(label.get_rotation(), xrot)

        if ylabelsize is not None or yrot is not None:
            if isinstance(ax.yaxis.get_minor_formatter(), NullFormatter):
                labels = ax.get_yticklabels()
            else:
                labels = ax.get_yticklabels() + ax.get_yticklabels(minor=True)

            for label in labels:
                if ylabelsize is not None:
                    tm.assert_almost_equal(label.get_fontsize(), ylabelsize)
                if yrot is not None:
                    tm.assert_almost_equal(label.get_rotation(), yrot)


def _check_ax_scales(axes, xaxis="linear", yaxis="linear"):
    # [FUNCTION DOC]
    # HELPER: checks axis scales. Parameters: axes (axes or list-like) and the
    # expected scale names for x and y (default linear; log is the other value).
    # Asserts every visible axes has those scales.
    """
    Check each axes has expected scales

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xaxis : {'linear', 'log'}
        expected xaxis scale
    yaxis : {'linear', 'log'}
        expected yaxis scale
    """
    axes = _flatten_visible(axes)
    for ax in axes:
        assert ax.xaxis.get_scale() == xaxis
        assert ax.yaxis.get_scale() == yaxis


def _check_axes_shape(axes, axes_num=None, layout=None, figsize=None):
    # [FUNCTION DOC]
    # HELPER: checks number and layout of axes. Parameters: axes (axes or
    # list-like); axes_num optional expected count of visible axes (each must have
    # drawn children); layout optional (rows, columns) tuple estimated from the
    # distinct x and y positions of all flattened axes; figsize optional expected
    # size in inches, defaulting to matplotlib's default of 6.4 by 4.8. Asserts the
    # figure size of the first visible axes equals the expected size.
    """
    Check expected number of axes is drawn in expected layout

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    axes_num : number
        expected number of axes. Unnecessary axes should be set to
        invisible.
    layout : tuple
        expected layout, (expected number of rows , columns)
    figsize : tuple
        expected figsize. default is matplotlib default
    """
    from pandas.plotting._matplotlib.tools import flatten_axes

    if figsize is None:
        figsize = (6.4, 4.8)
    visible_axes = _flatten_visible(axes)

    if axes_num is not None:
        assert len(visible_axes) == axes_num
        for ax in visible_axes:
            # check something drawn on visible axes
            assert len(ax.get_children()) > 0

    if layout is not None:
        x_set = set()
        y_set = set()
        for ax in flatten_axes(axes):
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        result = (len(y_set), len(x_set))
        assert result == layout

    tm.assert_numpy_array_equal(
        visible_axes[0].figure.get_size_inches(),
        np.array(figsize, dtype=np.float64),
    )


def _flatten_visible(axes: Axes | Sequence[Axes]) -> Sequence[Axes]:
    # [FUNCTION DOC]
    # HELPER: returns the visible axes from an axes or a sequence/array of axes
    # (possibly nested/2D). Flattens the input with the plotting tools function and
    # returns a list containing only the visible ones.
    """
    Flatten axes, and filter only visible

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like

    """
    from pandas.plotting._matplotlib.tools import flatten_axes

    axes_ndarray = flatten_axes(axes)
    axes = [ax for ax in axes_ndarray if ax.get_visible()]
    return axes


def _check_has_errorbars(axes, xerr=0, yerr=0):
    # [FUNCTION DOC]
    # HELPER: counts error bar containers. Parameters: axes (axes or list-like);
    # xerr and yerr are the expected number of containers having horizontal and
    # vertical error bars (default zero). For each visible axes, asserts the counts
    # of containers flagged as having x errors and y errors equal the expectations.
    """
    Check axes has expected number of errorbars

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    xerr : number
        expected number of x errorbar
    yerr : number
        expected number of y errorbar
    """
    axes = _flatten_visible(axes)
    for ax in axes:
        containers = ax.containers
        xerr_count = 0
        yerr_count = 0
        for c in containers:
            has_xerr = getattr(c, "has_xerr", False)
            has_yerr = getattr(c, "has_yerr", False)
            if has_xerr:
                xerr_count += 1
            if has_yerr:
                yerr_count += 1
        assert xerr == xerr_count
        assert yerr == yerr_count


def _check_box_return_type(
    returned, return_type, expected_keys=None, check_ax_title=True
):
    # [FUNCTION DOC]
    # HELPER: validates the return type of box plot or plot calls. Parameters:
    # returned is the object to check; return_type is the return type requested
    # (none, or one of the names for dict, axes or both); expected_keys is the
    # optional list of group labels (when absent a single axes is assumed);
    # check_ax_title says whether to verify each axes' title equals its key (should
    # be disabled for the normal plot function, which sets no title). With no keys:
    # a missing return type is treated as dict, and the returned object must be an
    # instance of the matching type (for both: a tuple with an axes and a dict of
    # lines). With keys: a missing return type means every visible element must be
    # an axes and the function returns; otherwise the result must be a series whose
    # sorted keys equal the sorted expected keys and whose values have the
    # requested type, with titles checked (for dict results the title is read from
    # the axes of the first median line). An unexpected return type raises an
    # assertion error.
    """
    Check box returned type is correct

    Parameters
    ----------
    returned : object to be tested, returned from boxplot
    return_type : str
        return_type passed to boxplot
    expected_keys : list-like, optional
        group labels in subplot case. If not passed,
        the function checks assuming boxplot uses single ax
    check_ax_title : bool
        Whether to check the ax.title is the same as expected_key
        Intended to be checked by calling from ``boxplot``.
        Normal ``plot`` doesn't attach ``ax.title``, it must be disabled.
    """
    from matplotlib.axes import Axes

    types = {"dict": dict, "axes": Axes, "both": tuple}
    if expected_keys is None:
        # should be fixed when the returning default is changed
        if return_type is None:
            return_type = "dict"

        assert isinstance(returned, types[return_type])
        if return_type == "both":
            assert isinstance(returned.ax, Axes)
            assert isinstance(returned.lines, dict)
    else:
        # should be fixed when the returning default is changed
        if return_type is None:
            for r in _flatten_visible(returned):
                assert isinstance(r, Axes)
            return

        assert isinstance(returned, Series)

        assert sorted(returned.keys()) == sorted(expected_keys)
        for key, value in returned.items():
            assert isinstance(value, types[return_type])
            # check returned dict has correct mapping
            if return_type == "axes":
                if check_ax_title:
                    assert value.get_title() == key
            elif return_type == "both":
                if check_ax_title:
                    assert value.ax.get_title() == key
                assert isinstance(value.ax, Axes)
                assert isinstance(value.lines, dict)
            elif return_type == "dict":
                line = value["medians"][0]
                axes = line.axes
                if check_ax_title:
                    assert axes.get_title() == key
            else:
                raise AssertionError


def _check_grid_settings(obj, kinds, kws=None):
    # [FUNCTION DOC]
    # HELPER: checks the grid defaults of plotting (GH 9792). Parameters: obj is a
    # series or frame with a plot method; kinds is a list of plot kind names; kws is
    # optional extra keyword arguments (default none). For every kind it draws on
    # consecutive subplots of one figure, with the global grid setting off then on:
    # with the grid keyword false the grid must be off even if the setting is on;
    # for kinds other than pie, hexbin and scatter, the setting alone must turn the
    # grid on, and requesting the grid true must turn it on even if the setting is
    # off. Side effects: changes the global axes grid setting and clears the current
    # figure after each case; the setting is not restored.
    # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792

    import matplotlib as mpl

    def is_grid_on():
        # [FUNCTION DOC]
        # HELPER: inner function of the grid check. Looks at the major ticks of the
        # current axes and returns true unless all x and all y grid lines are invisible.
        xticks = mpl.pyplot.gca().xaxis.get_major_ticks()
        yticks = mpl.pyplot.gca().yaxis.get_major_ticks()
        xoff = all(not g.gridline.get_visible() for g in xticks)
        yoff = all(not g.gridline.get_visible() for g in yticks)

        return not (xoff and yoff)

    if kws is None:
        kws = {}
    spndx = 1
    for kind in kinds:
        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc("axes", grid=False)
        obj.plot(kind=kind, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()

        mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
        spndx += 1
        mpl.rc("axes", grid=True)
        obj.plot(kind=kind, grid=False, **kws)
        assert not is_grid_on()
        mpl.pyplot.clf()

        if kind not in ["pie", "hexbin", "scatter"]:
            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc("axes", grid=True)
            obj.plot(kind=kind, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()

            mpl.pyplot.subplot(1, 4 * len(kinds), spndx)
            spndx += 1
            mpl.rc("axes", grid=False)
            obj.plot(kind=kind, grid=True, **kws)
            assert is_grid_on()
            mpl.pyplot.clf()


def _unpack_cycler(rcParams, field="color"):
    # [FUNCTION DOC]
    # HELPER: returns the list of values of one property (default color) from the
    # property cycle in the given rc parameters mapping, in order.
    """
    Auxiliary function for correctly unpacking cycler after MPL >= 1.5
    """
    return [v[field] for v in rcParams["axes.prop_cycle"]]


def get_x_axis(ax):
    # [FUNCTION DOC]
    # Returns the group of axes sharing the x axis with the given axes (from the
    # axes' shared-axes registry). Uses a private matplotlib attribute.
    return ax._shared_axes["x"]


def get_y_axis(ax):
    # [FUNCTION DOC]
    # Returns the group of axes sharing the y axis with the given axes (from the
    # axes' shared-axes registry). Uses a private matplotlib attribute.
    return ax._shared_axes["y"]


def assert_is_valid_plot_return_object(objs) -> None:
    # [FUNCTION DOC]
    # Validates the object returned from a plot call. If objs is a series or array,
    # every element must be an axes or a dict (series are unwrapped to their
    # values and arrays flattened); otherwise objs must be an artist, tuple or
    # dict. Raises assertion errors with descriptive messages.
    # KNOWN ISSUE: the message for the non-array case contains the typo
    # "ArtistArtist" (doubled word).
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    if isinstance(objs, (Series, np.ndarray)):
        if isinstance(objs, Series):
            objs = objs._values
        for el in objs.reshape(-1):
            msg = (
                "one of 'objs' is not a matplotlib Axes instance, "
                f"type encountered {type(el).__name__!r}"
            )
            assert isinstance(el, (Axes, dict)), msg
    else:
        msg = (
            "objs is neither an ndarray of Artist instances nor a single "
            "ArtistArtist instance, tuple, or dict, 'objs' is a "
            f"{type(objs).__name__!r}"
        )
        assert isinstance(objs, (Artist, tuple, dict)), msg


def _check_plot_works(f, default_axes=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: runs a plotting callable and checks that its results are valid.
    # Parameters: f is the plotting function; default_axes (default false) selects
    # whether to just call it with the keyword arguments (letting it create all axes
    # itself) or, when false, to call it twice: first on a new top subplot of a 2x1
    # grid (unless an axes is passed) and then on a new bottom subplot; all other
    # keyword arguments are forwarded to f. Clears the figure (taken from the
    # figure keyword or the current figure) before plotting, validates every
    # returned object with the plot return validator, and returns the object
    # returned by the last call. Side effects: clears and draws on a figure.
    """
    Create plot and ensure that plot return object is valid.

    Parameters
    ----------
    f : func
        Plotting function.
    default_axes : bool, optional
        If False (default):
            - If `ax` not in `kwargs`, then create subplot(211) and plot there
            - Create new subplot(212) and plot there as well
            - Mind special corner case for bootstrap_plot (see `_gen_two_subplots`)
        If True:
            - Simply run plotting function with kwargs provided
            - All required axes instances will be created automatically
            - It is recommended to use it when the plotting function
            creates multiple axes itself. It helps avoid warnings like
            'UserWarning: To output multiple subplots,
            the figure containing the passed axes is being cleared'
    **kwargs
        Keyword arguments passed to the plotting function.

    Returns
    -------
    Plot object returned by the last plotting.
    """
    import matplotlib.pyplot as plt

    if default_axes:
        gen_plots = _gen_default_plot
    else:
        gen_plots = _gen_two_subplots

    ret = None
    fig = kwargs.get("figure", plt.gcf())
    fig.clf()

    for ret in gen_plots(f, fig, **kwargs):
        assert_is_valid_plot_return_object(ret)

    return ret


def _gen_default_plot(f, fig, **kwargs):
    # [FUNCTION DOC]
    # HELPER: generator for the default-axes mode of the plot works check. Yields
    # the result of calling the function once with the keyword arguments; the figure
    # argument is not used.
    """
    Create plot in a default way.
    """
    yield f(**kwargs)


def _gen_two_subplots(f, fig, **kwargs):
    # [FUNCTION DOC]
    # HELPER: generator for the two-subplot mode of the plot works check. Adds a
    # top subplot if no axes was given and yields the function's result; then, for
    # the bootstrap plot function only, asserts no axes was given, otherwise sets
    # the axes keyword to a new bottom subplot; yields the function's result again.
    # Side effects: adds subplots to the figure and mutates the keyword mapping.
    """
    Create plot on two subplots forcefully created.
    """
    if "ax" not in kwargs:
        fig.add_subplot(211)
    yield f(**kwargs)

    if f is pd.plotting.bootstrap_plot:
        assert "ax" not in kwargs
    else:
        kwargs["ax"] = fig.add_subplot(212)
    yield f(**kwargs)
