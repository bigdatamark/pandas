# [MODULE DOC]
# Matplotlib implementation of histogram and kernel density plots. Defines the plot classes HistPlot
# (a LinePlot subclass handling bins, stacking and the by option) and KdePlot (derived from HistPlot, drawing a
# Gaussian kernel density estimate), plus helpers for grouped plots and the two entry functions hist_series
# and hist_frame used by the public plotting accessor. Depends on the core, groupby, misc and tools sibling modules.
# Imports matplotlib-related modules at import time through them; scipy is imported lazily for density plots.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    final,
)

import numpy as np

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
)
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import (
    create_iter_data_given_by,
    reformat_hist_y_given_by,
)
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
    set_ticks_props,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    from pandas._typing import PlottingOrientation

    from pandas import (
        DataFrame,
        Series,
    )


class HistPlot(LinePlot):
    # [CLASS DOC]
    # Histogram plot (kinds hist), subclass of LinePlot but with its own initializer. State: bottom offset(s), bin range, weights, axis labels, and the bins (either a single array of bin edges or, with by,
    # one array per group). Draws one histogram per column (or per group), supporting stacking.
    @property
    def _kind(self) -> Literal["hist", "kde"]:
        # [FUNCTION DOC]
        # Property: the kind is hist (the declared type also allows kde for the subclass).
        return "hist"

    def __init__(
        self,
        data,
        bins: int | np.ndarray | list[np.ndarray] = 10,
        bottom: int | np.ndarray = 0,
        *,
        range=None,
        weights=None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: data; bins (integer count, array of edges, or list of arrays; default 10); bottom (scalar or list-like, converted to an array; default 0); keyword-only range (limits for bin calculation) and weights; other keywords go to the base initializer.
        # The line plot initializer is deliberately skipped (it could fill missing values); the base plot initializer is called instead. After that, integer bins are converted to edges. Also records the xlabel and ylabel keywords.
        if is_list_like(bottom):
            bottom = np.array(bottom)
        self.bottom = bottom

        self._bin_range = range
        self.weights = weights

        self.xlabel = kwargs.get("xlabel")
        self.ylabel = kwargs.get("ylabel")
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)

        self.bins = self._adjust_bins(bins)

    def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]):
        # [FUNCTION DOC]
        # HELPER: converts an integer bin count to bin edge arrays so all histograms share edges. With by: one array per group computed from that group's selected columns; without by: a single array from the whole data. Non-integer bins are returned unchanged.
        if is_integer(bins):
            if self.by is not None:
                by_modified = unpack_single_str_list(self.by)
                grouped = self.data.groupby(by_modified)[self.columns]
                bins = [self._calculate_bins(group, bins) for key, group in grouped]
            else:
                bins = self._calculate_bins(self.data, bins)
        return bins

    def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: computes bin edges for data (series or frame): infers object types, keeps numeric columns, flattens, removes missing values, and returns the edges from numpy for the given bins and the stored range option.
        """Calculate bins given data"""
        nd_values = data.infer_objects()._get_numeric_data()
        values = nd_values.values
        if nd_values.ndim == 2:
            values = values.reshape(-1)
        values = values[~isna(values)]

        return np.histogram_bin_edges(values, bins=bins, range=self._bin_range)

    # error: Signature of "_plot" incompatible with supertype "LinePlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bottom: int | np.ndarray = 0,
        column_num: int = 0,
        stacking_id=None,
        *,
        bins,
        **kwds,
        # might return a subset from the possible return types of Axes.hist(...)[2]?
    ) -> BarContainer | Polygon | list[BarContainer | Polygon]:
        # [FUNCTION DOC]
        # Class method drawing one histogram on ax: y the values, bottom the base, column_num and stacking_id for stacking, keyword-only bins (edges), other keywords (label is required) go to the matplotlib hist function. The style argument is ignored.
        # On the first column initializes the stacking state with one slot per bin; the bottom is increased by the stacked totals; after drawing the stack totals are updated with the counts. Returns the patches.
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])
        # ignore style
        n, bins, patches = ax.hist(y, bins=bins, bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _make_plot(self, fig: Figure) -> None:
        # [FUNCTION DOC]
        # Draws all histograms. Gets colors and the stacking id; when by is given, the data is re-organized by group. For each column (group): chooses axes, copies keywords (adding the color option), builds and marks the label, applies style and color, adds the bottom and bins keywords (through the hook); with by, selects that group's bins,
        # uses the grouped columns as the label and drops the color; adds per-column weights when weights are given; removes missing values from y (with by, from each column); draws; with by sets the subplot title to the label; records the legend handle and label.
        colors = self._get_colors()
        stacking_id = self._get_stacking_id()

        # Re-create iterated data if `by` is assigned by users
        data = (
            create_iter_data_given_by(self.data, self._kind)
            if self.by is not None
            else self.data
        )
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)

            kwds = self.kwds.copy()
            if self.color is not None:
                kwds["color"] = self.color

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            self._make_plot_keywords(kwds, y)

            # the bins is multi-dimension array now and each plot need only 1-d and
            # when by is applied, label should be columns that are grouped
            if self.by is not None:
                kwds["bins"] = kwds["bins"][i]
                kwds["label"] = self.columns
                kwds.pop("color")

            if self.weights is not None:
                kwds["weights"] = type(self)._get_column_weights(self.weights, i, y)

            y = reformat_hist_y_given_by(y, self.by)

            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)

            # when by is applied, show title for subplots to know which group it is
            if self.by is not None:
                ax.set_title(pprint_thing(label))

            # error: Value of type "Polygon" is not indexable
            self._append_legend_handles_labels(artists[0], label)  # type: ignore[index,arg-type]

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        # [FUNCTION DOC]
        # HELPER: hook adding plot-specific keywords: sets the bottom and bins keywords from the stored values. The y argument is unused here (used by the density subclass). The docstring is imprecise about what is merged.
        """merge BoxPlot/KdePlot properties to passed kwds"""
        # y is required for KdePlot
        kwds["bottom"] = self.bottom
        kwds["bins"] = self.bins

    @final
    @staticmethod
    def _get_column_weights(weights, i: int, y):
        # [FUNCTION DOC]
        # HELPER: static, final. Selects weights for column i. A one-dimensional weights array (or one with a single column) is used for every column; a multi-dimensional array contributes its column i, raising a value error (weights must have the same shape as data or be a single column) if that index fails.
        # The selected weights are then filtered to the positions where y is not missing. None weights stay none.
        # We allow weights to be a multi-dimensional array, e.g. a (10, 2) array,
        # and each sub-array (10,) will be called in each iteration. If users only
        # provide 1D array, we assume the same weights is used for all iterations
        if weights is not None:
            if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
                try:
                    weights = weights[:, i]
                except IndexError as err:
                    raise ValueError(
                        "weights must have the same shape as data, "
                        "or be a single column"
                    ) from err
            weights = weights[~isna(y)]
        return weights

    def _post_plot_logic(self, ax: Axes, data) -> None:
        # [FUNCTION DOC]
        # Sets axis labels: the count axis is labeled Frequency unless a label was given (y for vertical orientation, x for horizontal); the value axis uses the other label option.
        if self.orientation == "horizontal":
            # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible
            # type "Hashable"; expected "str"
            ax.set_xlabel(
                "Frequency" if self.xlabel is None else self.xlabel  # type: ignore[arg-type]
            )
            ax.set_ylabel(self.ylabel)  # type: ignore[arg-type]
        else:
            ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]
            ax.set_ylabel(
                "Frequency" if self.ylabel is None else self.ylabel  # type: ignore[arg-type]
            )

    @property
    def orientation(self) -> PlottingOrientation:
        # [FUNCTION DOC]
        # Property: horizontal when the orientation keyword is horizontal, otherwise vertical.
        if self.kwds.get("orientation", None) == "horizontal":
            return "horizontal"
        else:
            return "vertical"


class KdePlot(HistPlot):
    # [CLASS DOC]
    # Kernel density estimate plot (kind kde), subclass of HistPlot reusing its drawing loop with density-specific keywords. State: bandwidth method, evaluation points, weights. Requires scipy at draw time.
    @property
    def _kind(self) -> Literal["kde"]:
        # [FUNCTION DOC]
        # Property returning the string kde.
        return "kde"

    @property
    def orientation(self) -> Literal["vertical"]:
        # [FUNCTION DOC]
        # Property: orientation is always vertical.
        return "vertical"

    def __init__(
        self, data, bw_method=None, ind=None, *, weights=None, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: data; bw_method (bandwidth rule or value); ind (evaluation points or an integer count); keyword-only weights; other keywords go to the base plot initializer (skipping the histogram initializer). Stores the three options.
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)
        self.bw_method = bw_method
        self.ind = ind
        self.weights = weights

    @staticmethod
    def _get_ind(y: np.ndarray, ind):
        # [FUNCTION DOC]
        # HELPER: static. Determines evaluation points. None gives 1000 evenly spaced points over the data range extended by half the range on each side (ignoring missing values); an integer gives that many points over the same extended range; otherwise the given points are returned.
        if ind is None:
            # np.nanmax() and np.nanmin() ignores the missing values
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                1000,
            )
        elif is_integer(ind):
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                ind,
            )
        return ind

    @classmethod
    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bw_method=None,
        weights=None,
        ind=None,
        column_num=None,
        stacking_id: int | None = None,
        **kwds,
    ):
        # [FUNCTION DOC]
        # Class method evaluating a Gaussian kernel density estimate (scipy, imported lazily) of y with missing values removed, bw_method and weights, at the points ind (asserted not none), and drawing the curve as a line with the base plotting method (points as x, density as y). Returns the lines.
        # The column_num and stacking_id parameters are accepted but unused.
        from scipy.stats import gaussian_kde

        y = remove_na_arraylike(y)
        gkde = gaussian_kde(y, bw_method=bw_method, weights=weights)

        # gaussian_kde.evaluate(None) raises TypeError, so pyright requires this check
        assert ind is not None
        y = gkde.evaluate(ind)
        lines = MPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        # [FUNCTION DOC]
        # HELPER: hook adding keywords: sets bw_method from the stored value and ind from the evaluation point computation for the current data.
        kwds["bw_method"] = self.bw_method
        kwds["ind"] = type(self)._get_ind(y, ind=self.ind)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        # [FUNCTION DOC]
        # Sets the y axis label to Density.
        ax.set_ylabel("Density")


def _grouped_plot(
    plotf,
    data: Series | DataFrame,
    column=None,
    by=None,
    numeric_only: bool = True,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = True,
    layout=None,
    rot: float = 0,
    ax=None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: draws one plot per group. plotf is called with the group data and its axes plus extra keywords; data is a series or frame; column (optional) selects columns; by the grouping; numeric_only (default true) keeps only numeric columns of frame groups; figsize, sharex, sharey, layout, ax size and place the subplots.
    # Raises a value error if figsize is the string default (no longer supported). Creates the subplots (one per group) and titles each with the printed group key. Returns the figure and axes.
    # KNOWN ISSUE: the rot parameter is accepted but never used.
    # error: Non-overlapping equality check (left operand type: "Optional[Tuple[float,
    # float]]", right operand type: "Literal['default']")
    if figsize == "default":  # type: ignore[comparison-overlap]
        # allowed to specify mpl default with 'default'
        raise ValueError(
            "figsize='default' is no longer supported. "
            "Specify figure size by tuple instead"
        )

    grouped = data.groupby(by)
    if column is not None:
        grouped = grouped[column]

    naxes = len(grouped)
    fig, axes = create_subplots(
        naxes=naxes, figsize=figsize, sharex=sharex, sharey=sharey, ax=ax, layout=layout
    )

    for ax, (key, group) in zip(flatten_axes(axes), grouped, strict=False):
        if numeric_only and isinstance(group, ABCDataFrame):
            group = group._get_numeric_data()
        plotf(group, ax, **kwargs)
        ax.set_title(pprint_thing(key))

    return fig, axes


def _grouped_hist(
    data: Series | DataFrame,
    column=None,
    by=None,
    ax=None,
    bins: int = 50,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = False,
    rot: float = 90,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    legend: bool = False,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: grouped histogram. Parameters follow the public API: data, column, by, ax, bins (default 50), figsize, layout, sharex, sharey, rot (default 90), grid (default true), tick label sizes and rotations, legend (default false), and keywords for the histogram call.
    # When legend is true a label keyword must not be present (asserted) and one is set from the series name, the frame columns or the column option. Each group is histogrammed (missing values dropped) with the shared bins and a legend if requested. The x rotation defaults to rot.
    # After plotting, tick properties are set and the figure margins adjusted. Returns the array of axes.
    # KNOWN ISSUE: the grid parameter is accepted but never applied.
    # KNOWN ISSUE: the docstring has a typo in the legend entry and does not document several parameters such as the tick settings.
    """
    Grouped histogram

    Parameters
    ----------
    data : Series/DataFrame
    column : object, optional
    by : object, optional
    ax : axes, optional
    bins : int, default 50
    figsize : tuple, optional
    layout : optional
    sharex : bool, default False
    sharey : bool, default False
    rot : float, default 90
    grid : bool, default True
    legend: : bool, default False
    kwargs : dict, keyword arguments passed to matplotlib.Axes.hist

    Returns
    -------
    collection of Matplotlib Axes
    """
    if legend:
        assert "label" not in kwargs
        if data.ndim == 1:
            kwargs["label"] = data.name
        elif column is None:
            kwargs["label"] = data.columns
        else:
            kwargs["label"] = column

    def plot_group(group, ax) -> None:
        # [FUNCTION DOC]
        # HELPER: draws a single group on its axes: histogram of the group values without missing entries using the enclosing bins and keywords, and a legend if requested.
        ax.hist(group.dropna().values, bins=bins, **kwargs)
        if legend:
            ax.legend()

    if xrot is None:
        xrot = rot

    fig, axes = _grouped_plot(
        plot_group,
        data,
        column=column,
        by=by,
        sharex=sharex,
        sharey=sharey,
        ax=ax,
        figsize=figsize,
        layout=layout,
        rot=rot,
    )

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )

    maybe_adjust_figure(
        fig, bottom=0.15, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.3
    )
    return axes


def hist_series(
    self: Series,
    by=None,
    ax=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    figsize: tuple[float, float] | None = None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    # [FUNCTION DOC]
    # Entry point for histogram of a Series. Parameters: self (the series), by, ax, grid, tick label sizes and rotations, figsize, bins (default 10), legend, and histogram keywords. Raises a value error when both legend and label are given.
    # Without by: raises a value error if a layout is given; gets the figure from a figure keyword or the current figure (creating one with the size if none exists), resizes it if figsize differs; uses the given axes (which must belong to the figure, else an assertion error) or the figure's current axes;
    # draws the histogram of non-missing values (with the series name as label when legend is requested), shows the grid, and sets tick properties. With by: a figure keyword is rejected with a value error; delegates to the grouped histogram. Returns the axes (a single axes if the result is a one-element array).
    # KNOWN ISSUE: the default for the figure keyword lookup is evaluated eagerly, so a figure may be created or fetched even when a figure keyword is supplied.
    import matplotlib.pyplot as plt

    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")

    if by is None:
        if kwds.get("layout", None) is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        # hack until the plotting interface is a bit more unified
        fig = kwds.pop(
            "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
        )
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")
        values = self.dropna().values
        if legend:
            kwds["label"] = self.name
        ax.hist(values, bins=bins, **kwds)
        if legend:
            ax.legend()
        ax.grid(grid)
        axes = np.array([ax])

        set_ticks_props(
            axes,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
        )

    else:
        if "figure" in kwds:
            raise ValueError(
                "Cannot pass 'figure' when using the "
                "'by' argument, since a new 'Figure' instance will be created"
            )
        axes = _grouped_hist(
            self,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )

    if hasattr(axes, "ndim"):
        if axes.ndim == 1 and len(axes) == 1:
            return axes[0]
    return axes


def hist_frame(
    data: DataFrame,
    column=None,
    by=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    ax=None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    layout=None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    # [FUNCTION DOC]
    # Entry point for histograms of DataFrame columns. Parameters: data, column, by, grid, tick label sizes/rotations, ax, sharex, sharey, figsize, layout, bins (default 10), legend, histogram keywords. A legend together with a label raises a value error. With by: delegates to the grouped histogram and returns its axes.
    # Otherwise selects the given columns, keeps numeric, datetime and tz-aware datetime columns (excluding timedelta) (GH32590); if no column remains raises a value error saying hist requires numerical or datetime columns. Creates a grid of subplots (one per column, not squeezed),
    # draws each column's histogram of non-missing values with the column name as title and optional per-column label for the legend, sets grid, tick properties and figure margins. Returns the two-dimensional array of axes.
    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")
    if by is not None:
        axes = _grouped_hist(
            data,
            column=column,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout=layout,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
        return axes

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndex)):
            column = [column]
        data = data[column]
    # GH32590
    data = data.select_dtypes(
        include=(np.number, "datetime64", "datetimetz"), exclude="timedelta"
    )
    naxes = len(data.columns)

    if naxes == 0:
        raise ValueError(
            "hist method requires numerical or datetime columns, nothing to plot."
        )

    fig, axes = create_subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
    )
    can_set_label = "label" not in kwds

    for ax, col in zip(flatten_axes(axes), data.columns, strict=False):
        if legend and can_set_label:
            kwds["label"] = col
        ax.hist(data[col].dropna().values, bins=bins, **kwds)
        ax.set_title(col)
        ax.grid(grid)
        if legend:
            ax.legend()

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )
    maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)

    return axes
