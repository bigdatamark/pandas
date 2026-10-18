# being a bit too dynamic
# [MODULE DOC]
# HELPER: matplotlib helper utilities for the pandas plotting backend: figure layout adjustment, tables,
# subplot grid sizing and creation (adapted from matplotlib's subplot creation), shared-axis label hiding,
# flattening of axes containers, tick property setting and line/limit queries. Used by core, hist, misc and
# timeseries plotting modules. Imports matplotlib at import time; no other side effects.

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
import warnings

import matplotlib as mpl
import numpy as np

from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
    )

    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.table import Table

    from pandas import (
        DataFrame,
        Series,
    )


def do_adjust_figure(fig: Figure) -> bool:
    # [FUNCTION DOC]
    # Returns whether the figure's layout may be adjusted manually: false when the figure lacks the constrained-layout
    # getter, otherwise true exactly when constrained layout is not enabled.
    """Whether fig has constrained_layout enabled."""
    if not hasattr(fig, "get_constrained_layout"):
        return False
    return not fig.get_constrained_layout()


def maybe_adjust_figure(fig: Figure, *args, **kwargs) -> None:
    # [FUNCTION DOC]
    # Calls the figure's subplot adjustment with the given positional and keyword arguments, but only when manual
    # adjustment is allowed (see the previous function). Returns nothing.
    """Call fig.subplots_adjust unless fig has constrained_layout enabled."""
    if do_adjust_figure(fig):
        fig.subplots_adjust(*args, **kwargs)


def format_date_labels(ax: Axes, rot) -> None:
    # [FUNCTION DOC]
    # HELPER: mini version of the automatic date label formatting. For each x tick label of ax, sets right horizontal alignment
    # and the rotation rot. If the axes has a figure, shrinks the bottom margin to 0.2 when manual adjustment is allowed.
    # Returns nothing.
    # mini version of autofmt_xdate
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        # should always be a Figure but can technically be None
        maybe_adjust_figure(fig, bottom=0.2)  # type: ignore[arg-type]


def table(
    ax, data: DataFrame | Series, rowLabels=None, colLabels=None, **kwargs
) -> Table:
    # [FUNCTION DOC]
    # Builds a matplotlib table on ax from data. data must be a DataFrame or Series (a Series is converted to a one-column
    # frame); anything else raises a value error. rowLabels and colLabels default to the index and columns of the data; extra
    # keyword arguments go to the matplotlib table function. The cell text is the data's values. Returns the table object.
    if isinstance(data, ABCSeries):
        data = data.to_frame()
    elif isinstance(data, ABCDataFrame):
        pass
    else:
        raise ValueError("Input data must be DataFrame or Series")

    if rowLabels is None:
        rowLabels = data.index

    if colLabels is None:
        colLabels = data.columns

    cellText = data.values

    # error: Argument "cellText" to "table" has incompatible type "ndarray[Any,
    # Any]"; expected "Sequence[Sequence[str]] | None"
    return mpl.table.table(
        ax,
        cellText=cellText,  # type: ignore[arg-type]
        rowLabels=rowLabels,
        colLabels=colLabels,
        **kwargs,
    )


def _get_layout(
    nplots: int,
    layout: tuple[int, int] | None = None,
    layout_type: str = "box",
) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: computes the (rows, columns) subplot grid for nplots plots. layout is an optional tuple or list of two
    # integers; layout_type (default box) selects the shape when no layout is given.
    # With a layout: it must be a two-element tuple or list, else a value error. A minus one in one dimension is filled in as the ceiling of
    # nplots divided by the other (positive) dimension; if both dimensions are non-positive and not the above, a value error is
    # raised (at least one dimension must be positive); if the grid is smaller than nplots a value error is raised. The (possibly
    # list-typed) layout is returned.
    # Without a layout: single gives one by one, horizontal gives one row, vertical gives one column; box gives a fixed grid for one to
    # four plots (one by one, one by two, two by two, two by two) and otherwise the smallest near-square grid: k by k where k is the
    # smallest integer whose square is at least nplots, reduced to k by k minus one when that still fits.
    if layout is not None:
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError("Layout must be a tuple of (rows, columns)")

        nrows, ncols = layout

        if nrows == -1 and ncols > 0:
            layout = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            msg = "At least one dimension of layout must be positive"
            raise ValueError(msg)

        nrows, ncols = layout
        if nrows * ncols < nplots:
            raise ValueError(
                f"Layout of {nrows}x{ncols} must be larger than required size {nplots}"
            )

        return layout

    if layout_type == "single":
        return (1, 1)
    elif layout_type == "horizontal":
        return (1, nplots)
    elif layout_type == "vertical":
        return (nplots, 1)

    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[nplots]
    except KeyError:
        k = 1
        while k**2 < nplots:
            k += 1

        if (k - 1) * k >= nplots:
            return k, (k - 1)
        else:
            return k, k


# copied from matplotlib/pyplot.py and modified for pandas.plotting


def create_subplots(
    naxes: int,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw=None,
    ax=None,
    layout=None,
    layout_type: str = "box",
    **fig_kw,
):
    # [FUNCTION DOC]
    # Creates a figure with a grid of subplots (adapted from matplotlib's pyplot function). naxes is the number of axes needed
    # (extra grid cells are hidden); sharex and sharey share axes with the first subplot; squeeze says whether to drop length-one
    # dimensions of the returned axes; subplot_kw are keywords for each add-subplot call; ax is an optional existing axes or
    # sequence of axes to reuse; layout and layout_type determine the grid; fig_kw go to the figure creation.
    # Returns a tuple of figure and axes (a single axes, a one-dimensional or a two-dimensional object array).
    # If ax is a list-like of axes: when squeeze is true it is flattened to a one-dimensional object array; warnings (user warning)
    # are emitted when layout is given or sharex/sharey are set because they are ignored; the number of axes must equal naxes,
    # else a value error is raised; the existing figure and axes are returned as is.
    # If ax is a single axes: when naxes is one the axes is returned directly (or as a flat object array when squeeze is false);
    # when naxes is greater than one a user warning is emitted and the figure is cleared so that new subplots can be added.
    # Otherwise a new figure is created. Subplots are created one by one (grid cells beyond naxes get no shared axes and are hidden),
    # then the shared-axis label handling runs, and the result is reshaped according to squeeze.
    # KNOWN ISSUE: the docstring says naxes defaults to rows times columns, but it is a required parameter with no default.
    # KNOWN ISSUE: the docstring lists only box, horizontal and vertical for layout_type but the layout helper also accepts single.
    # KNOWN ISSUE: the docstring promises a two-dimensional array when squeeze is false, but the existing-single-axes path returns a flat one-dimensional array.
    # KNOWN ISSUE: when a list of axes is passed and squeeze is false, the list is not converted to an array and the size attribute lookup that follows would fail for a plain list.
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is
      nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    layout : tuple
      Number of rows and columns of the subplot grid.
      If not specified, calculated from naxes and layout_type

    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'
      Specify how to layout the subplot grid.

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.

    Returns
    -------
    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    Examples
    --------
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
    import matplotlib.pyplot as plt

    if subplot_kw is None:
        subplot_kw = {}

    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        if is_list_like(ax):
            if squeeze:
                ax = np.fromiter(flatten_axes(ax), dtype=object)
            if layout is not None:
                warnings.warn(
                    "When passing multiple axes, layout keyword is ignored.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            if sharex or sharey:
                warnings.warn(
                    "When passing multiple axes, sharex and sharey "
                    "are ignored. These settings must be specified when creating axes.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return fig, ax
            else:
                raise ValueError(
                    f"The number of passed axes must be {naxes}, the "
                    "same as the output plot"
                )

        fig = ax.get_figure()
        # if ax is passed and a number of subplots is 1, return ax as it is
        if naxes == 1:
            if squeeze:
                return fig, ax
            else:
                return fig, np.fromiter(flatten_axes(ax), dtype=object)
        else:
            warnings.warn(
                "To output multiple subplots, the figure containing "
                "the passed axes is being cleared.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            fig.clear()

    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)

    if sharex:
        subplot_kw["sharex"] = ax0
    if sharey:
        subplot_kw["sharey"] = ax0
    axarr[0] = ax0

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        # Set sharex and sharey to None for blank/dummy axes, these can
        # interfere with proper axis limits on the visible axes if
        # they share axes e.g. issue #7528
        if i >= naxes:
            kwds["sharex"] = None
            kwds["sharey"] = None
        ax = fig.add_subplot(nrows, ncols, i + 1, **kwds)
        axarr[i] = ax

    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)

    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            axes = axarr[0]
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        axes = axarr.reshape(nrows, ncols)

    return fig, axes


def _remove_labels_from_axis(axis: Axis) -> None:
    # [FUNCTION DOC]
    # HELPER: hides the tick labels and axis label of the given axis. Hides major tick labels; ensures minor tick labels can be
    # hidden by replacing a null minor locator with an automatic one and a null minor formatter with an empty-format formatter;
    # hides minor tick labels and the axis label. Mutates the axis; returns nothing.
    for t in axis.get_majorticklabels():
        t.set_visible(False)

    # set_visible will not be effective if
    # minor axis has NullLocator and NullFormatter (default)
    if isinstance(axis.get_minor_locator(), mpl.ticker.NullLocator):
        axis.set_minor_locator(mpl.ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), mpl.ticker.NullFormatter):
        axis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    for t in axis.get_minorticklabels():
        t.set_visible(False)

    axis.get_label().set_visible(False)


def _has_externally_shared_axis(ax1: Axes, compare_axis: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true when axis ax1 shares the x or y axis (compare_axis must be the letter x or y, otherwise a value error is
    # raised) with another axes at a different position (external sharing, as opposed to twinning, where the position is identical).
    # Looks at the group of sibling axes of the chosen shared axis and compares position corner points.
    """
    Return whether an axis is externally shared.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Axis to query.
    compare_axis : str
        `"x"` or `"y"` according to whether the X-axis or Y-axis is being
        compared.

    Returns
    -------
    bool
        `True` if the axis is externally shared. Otherwise `False`.

    Notes
    -----
    If two axes with different positions are sharing an axis, they can be
    referred to as *externally* sharing the common axis.

    If two axes sharing an axis also have the same position, they can be
    referred to as *internally* sharing the common axis (a.k.a twinning).

    _handle_shared_axes() is only interested in axes externally sharing an
    axis, regardless of whether either of the axes is also internally sharing
    with a third axis.
    """
    if compare_axis == "x":
        axes = ax1.get_shared_x_axes()
    elif compare_axis == "y":
        axes = ax1.get_shared_y_axes()
    else:
        raise ValueError(
            "_has_externally_shared_axis() needs 'x' or 'y' as a second parameter"
        )

    axes_siblings = axes.get_siblings(ax1)

    # Retain ax1 and any of its siblings which aren't in the same position as it
    ax1_points = ax1.get_position().get_points()

    for ax2 in axes_siblings:
        if not np.array_equal(ax1_points, ax2.get_position().get_points()):
            return True

    return False


def handle_shared_axes(
    axarr: Iterable[Axes],
    nplots: int,
    naxes: int,
    nrows: int,
    ncols: int,
    sharex: bool,
    sharey: bool,
) -> None:
    # [FUNCTION DOC]
    # HELPER: hides tick labels on shared axes so only the outer subplots show labels. axarr is the iterable of axes in the grid,
    # nplots the number of grid cells, naxes the number of visible plots, nrows and ncols the grid shape, sharex and sharey the
    # sharing flags. Does nothing for a single plot. When there is more than one row: builds a layout map of visible axes and removes x
    # labels of every axes that has a visible subplot directly below it, if x is shared (by flag or externally); if the map indexing
    # fails (grid specs that do not match) falls back to hiding x labels on every axes that is not in the last row. When there is more than
    # one column: removes y labels from every axes not in the first column if y is shared. Mutates the axes; returns nothing.
    if nplots > 1:
        row_num = lambda x: x.get_subplotspec().rowspan.start
        col_num = lambda x: x.get_subplotspec().colspan.start

        is_first_col = lambda x: x.get_subplotspec().is_first_col()

        if nrows > 1:
            try:
                # first find out the ax layout,
                # so that we can correctly handle 'gaps"
                layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
                for ax in axarr:
                    layout[row_num(ax), col_num(ax)] = ax.get_visible()

                for ax in axarr:
                    # only the last row of subplots should get x labels -> all
                    # other off layout handles the case that the subplot is
                    # the last in the column, because below is no subplot/gap.
                    if not layout[row_num(ax) + 1, col_num(ax)]:
                        continue
                    if sharex or _has_externally_shared_axis(ax, "x"):
                        _remove_labels_from_axis(ax.xaxis)

            except IndexError:
                # if gridspec is used, ax.rowNum and ax.colNum may different
                # from layout shape. in this case, use last_row logic
                is_last_row = lambda x: x.get_subplotspec().is_last_row()
                for ax in axarr:
                    if is_last_row(ax):
                        continue
                    if sharex or _has_externally_shared_axis(ax, "x"):
                        _remove_labels_from_axis(ax.xaxis)

        if ncols > 1:
            for ax in axarr:
                # only the first column should get y labels -> set all other to
                # off as we only have labels in the first column and we always
                # have a subplot there, we can skip the layout test
                if is_first_col(ax):
                    continue
                if sharey or _has_externally_shared_axis(ax, "y"):
                    _remove_labels_from_axis(ax.yaxis)


def flatten_axes(axes: Axes | Iterable[Axes]) -> Generator[Axes]:
    # [FUNCTION DOC]
    # HELPER: generator that yields individual axes. A scalar (non list-like) is yielded as is; numpy arrays and indexes are flattened
    # and their elements yielded; any other iterable has its elements yielded directly.
    if not is_list_like(axes):
        yield axes  # type: ignore[misc]
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        yield from np.asarray(axes).reshape(-1)
    else:
        yield from axes  # type: ignore[misc]


def set_ticks_props(
    axes: Axes | Iterable[Axes],
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
):
    # [FUNCTION DOC]
    # Sets tick label properties on all axes in axes (a single axes or iterable, flattened). xlabelsize and ylabelsize are font sizes
    # for x and y tick labels; xrot and yrot are rotations; each is applied only when not none. Returns the original axes argument.
    for ax in flatten_axes(axes):
        if xlabelsize is not None:
            mpl.artist.setp(ax.get_xticklabels(), fontsize=xlabelsize)  # type: ignore[arg-type]
        if xrot is not None:
            mpl.artist.setp(ax.get_xticklabels(), rotation=xrot)  # type: ignore[arg-type]
        if ylabelsize is not None:
            mpl.artist.setp(ax.get_yticklabels(), fontsize=ylabelsize)  # type: ignore[arg-type]
        if yrot is not None:
            mpl.artist.setp(ax.get_yticklabels(), rotation=yrot)  # type: ignore[arg-type]
    return axes


def get_all_lines(ax: Axes) -> list[Line2D]:
    # [FUNCTION DOC]
    # HELPER: returns the lines of ax; if it has right-axis or left-axis companions (secondary y handling), their lines are appended,
    # extending the list obtained from the primary axes. Returns the combined list of line objects.
    lines = ax.get_lines()

    if hasattr(ax, "right_ax"):
        lines += ax.right_ax.get_lines()

    if hasattr(ax, "left_ax"):
        lines += ax.left_ax.get_lines()

    return lines


def get_xlim(lines: Iterable[Line2D]) -> tuple[float, float]:
    # [FUNCTION DOC]
    # HELPER: returns the minimum and maximum x data over all given lines as a pair, ignoring NaN. Uses the original (non-transformed)
    # x data. If there are no lines returns positive infinity and negative infinity.
    left, right = np.inf, -np.inf
    for line in lines:
        x = line.get_xdata(orig=False)
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    return left, right
