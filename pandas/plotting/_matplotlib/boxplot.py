# [MODULE DOC]
# Matplotlib implementation of box plots for the pandas plotting backend.
# Public contents: the box plot class (a line-plot subclass used for the plot
# accessor box kind), the stand-alone box plot function for frames and series,
# the method-style frame function, the grouped-frame function, and a color
# helper that paints the artists of a drawn box plot.
# Relationships: builds on the line plot and base plot classes of the core
# module, uses subplot-creation utilities from the tools module, color
# selection from the style module, and grouped-data helpers from the groupby
# module. The generic front end in the plotting core module dispatches here.
# Import-time side effects: imports matplotlib and pandas at load time; none
# other.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
)
import warnings

import matplotlib as mpl
import numpy as np

from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike

import pandas as pd
import pandas.core.common as com
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from pandas._typing import MatplotlibColor


def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None:
    # [FUNCTION DOC]
    # HELPER: sets tick labels on the value-axis-orthogonal axis of a box plot.
    # Inputs: ax is the target axes; labels is the list of label strings;
    # is_vertical selects the horizontal axis when true, else the vertical one;
    # extra keywords (such as rotation) go to the matplotlib label setter.
    # Works around a matplotlib issue with repeated fixed ticks: when the number of
    # current ticks differs from the labels count, the label list is repeated by the
    # integer ratio (older matplotlib versions assert the ratio is exact; newer
    # ones skip that check). Mutates the axes; returns nothing.
    """Set the tick labels of a given axis.

    Due to https://github.com/matplotlib/matplotlib/pull/17266, we need to handle the
    case of repeated ticks (due to `FixedLocator`) and thus we duplicate the number of
    labels.
    """
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    if len(ticks) != len(labels):
        i, remainder = divmod(len(ticks), len(labels))
        if Version(mpl.__version__) < Version("3.10"):
            assert remainder == 0, remainder
        labels *= i
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)


class BoxPlot(LinePlot):
    # [CLASS DOC]
    # Box plot implementation (subclass of the line plot class). Kind is box;
    # subplot layout is horizontal. State: the validated return type (none, axes,
    # dict or both) plus the usual plot state. A nested named-tuple class pairs an
    # axes with the dict of drawn line artists for the both return type. Key
    # behavior: draws with matplotlib's box plot per column (or per by-group
    # subplot), recolors artists, sets tick labels from column names, and returns an
    # object chosen by the return type.
    @property
    def _kind(self) -> Literal["box"]:
        # [FUNCTION DOC]
        # Property returning the constant kind name box.
        return "box"

    _layout_type = "horizontal"

    _valid_return_types = (None, "axes", "dict", "both")

    class BP(NamedTuple):
        # [CLASS DOC]
        # Named tuple used as the result type when the return type is both: field ax
        # holds the axes and field lines holds the dictionary of line artists lists
        # (boxes, whiskers, medians, caps, fliers) returned by matplotlib.
        # namedtuple to hold results
        ax: Axes
        lines: dict[str, list[Line2D]]

    def __init__(self, data, return_type: str = "axes", **kwargs) -> None:
        # [FUNCTION DOC]
        # Constructor. Inputs: data to plot; return_type (default axes; allowed none,
        # axes, dict, both); extra keywords for the base plot class. Raises a value
        # error for any other return type. Deliberately calls the base plot class
        # initializer directly instead of the line plot one, to avoid its missing-value
        # filling. When subplots are requested, disables sharing along the axis that
        # carries the labels (x for vertical boxes, y for horizontal) so each subplot
        # keeps its own labels.
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")

        self.return_type = return_type
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)

        if self.subplots:
            # Disable label ax sharing. Otherwise, all subplots shows last
            # column label
            if self.orientation == "vertical":
                self.sharex = False
            else:
                self.sharey = False

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Axes, y: np.ndarray, column_num=None, return_type: str = "axes", **kwds
    ):
        # [FUNCTION DOC]
        # Class-level drawing step for one axes (signature intentionally differs from
        # the base class). Inputs: ax, y (a one- or two-dimensional array of values),
        # column_num (unused), return_type (default axes), and keywords for matplotlib's
        # box plot. For two-dimensional input each column is stripped of missing values,
        # and empty columns are replaced by a single missing value because matplotlib
        # fails on empty arrays; one-dimensional input just drops missing values.
        # Draws and returns a pair: first the object to hand back to the user (the line
        # dictionary for dict, the named tuple for both, otherwise the axes), second the
        # raw dictionary of artists.
        ys: np.ndarray | list[np.ndarray]
        if y.ndim == 2:
            ys = [remove_na_arraylike(v) for v in y]
            # Boxplot fails with empty arrays, so need to add a NaN
            #   if any cols are empty
            # GH 8181
            ys = [v if v.size > 0 else np.array([np.nan]) for v in ys]
        else:
            ys = remove_na_arraylike(y)
        bp = ax.boxplot(ys, **kwds)

        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _validate_color_args(self, color, colormap):
        # [FUNCTION DOC]
        # HELPER: validates the color argument for this plot kind (overrides the base
        # class). Returns none when the argument is the unset sentinel. Otherwise emits
        # a user warning if a colormap was also given (the color is used). When color is
        # a dictionary, raises a value error for any key outside boxes, whiskers,
        # medians, caps. Returns the color unchanged.
        if color is lib.no_default:
            return None

        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        if isinstance(color, dict):
            valid_keys = ["boxes", "whiskers", "medians", "caps"]
            for key in color:
                if key not in valid_keys:
                    raise ValueError(
                        f"color dict contains invalid key '{key}'. "
                        f"The key must be either {valid_keys}"
                    )
        return color

    @cache_readonly
    def _color_attrs(self):
        # [FUNCTION DOC]
        # HELPER: cached property computing three default colors from the standard color
        # selection (honoring the colormap): index 0 for box, whisker and cap, index 2
        # for the median. Flier colors are controlled by the symbol keyword instead.
        # get standard colors for default
        # use 2 colors by default, for box/whisker and median
        # flier colors isn't needed here
        # because it can be specified by ``sym`` kw
        return get_standard_colors(num_colors=3, colormap=self.colormap, color=None)

    @cache_readonly
    def _boxes_c(self):
        # [FUNCTION DOC]
        # HELPER: cached default color of the boxes (first default color).
        return self._color_attrs[0]

    @cache_readonly
    def _whiskers_c(self):
        # [FUNCTION DOC]
        # HELPER: cached default color of the whiskers (first default color).
        return self._color_attrs[0]

    @cache_readonly
    def _medians_c(self):
        # [FUNCTION DOC]
        # HELPER: cached default color of the medians (third default color).
        return self._color_attrs[2]

    @cache_readonly
    def _caps_c(self):
        # [FUNCTION DOC]
        # HELPER: cached default color of the caps (first default color).
        return self._color_attrs[0]

    def _get_colors(
        self,
        num_colors=None,
        color_kwds: dict[str, MatplotlibColor]
        | MatplotlibColor
        | Collection[MatplotlibColor]
        | None = "color",
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: intentionally empty override of the base class color generation, so
        # that no per-series color list is computed for box plots (colors are applied
        # after drawing). Parameters are ignored; returns nothing.
        pass

    def maybe_color_bp(self, bp) -> None:
        # [FUNCTION DOC]
        # Recolors the artists of a drawn box plot dictionary. If the plot color is a
        # dictionary, each of boxes, whiskers, medians, caps takes the user value or
        # falls back to the cached default; otherwise a single user color (anything
        # accepted by matplotlib) is used for all four, or the defaults when it is
        # unset. Then calls the module-level coloring function with the colors and the
        # plot keywords (so explicitly given property dictionaries win). Mutates
        # artists; returns nothing.
        if isinstance(self.color, dict):
            boxes = self.color.get("boxes", self._boxes_c)
            whiskers = self.color.get("whiskers", self._whiskers_c)
            medians = self.color.get("medians", self._medians_c)
            caps = self.color.get("caps", self._caps_c)
        else:
            # Other types are forwarded to matplotlib
            # If None, use default colors
            boxes = self.color or self._boxes_c
            whiskers = self.color or self._whiskers_c
            medians = self.color or self._medians_c
            caps = self.color or self._caps_c

        color_tup = (boxes, whiskers, medians, caps)
        maybe_color_bp(bp, color_tup=color_tup, **self.kwds)

    def _make_plot(self, fig: Figure) -> None:
        # [FUNCTION DOC]
        # Draws the box plot(s) on the figure (called by the base class generation).
        # With subplots: when a by-grouping was requested, the data is rebuilt into
        # per-group form; each iterated item gets its own axes, a copy of the keywords,
        # and is drawn with the drawing step and recolored. For by-grouping the values
        # are transposed, the subplot title is the group label and tick labels are the
        # grouped values from the first level of the column index; otherwise the tick
        # label is the column name. The returned objects are collected into an object
        # series indexed by label and stored as the return object. Without subplots:
        # all columns are drawn on the first axes as a transposed value array, the
        # return object is the draw result, and tick labels are the column names (or
        # positional numbers when the index is not used). Tick labels are set via the
        # module helper.
        if self.subplots:
            obj_axes = []
            obj_labels = []

            # Re-create iterated data if `by` is assigned by users
            data = (
                create_iter_data_given_by(self.data, self._kind)
                if self.by is not None
                else self.data
            )

            for i, (label, y) in enumerate(self._iter_data(data=data)):
                ax = self._get_ax(i)
                kwds = self.kwds.copy()

                # When by is applied, show title for subplots to know which group it is
                # just like df.boxplot, and need to apply T on y to provide right input
                if self.by is not None:
                    y = y.T
                    ax.set_title(pprint_thing(label))

                    # When `by` is assigned, the ticklabels will become unique grouped
                    # values, instead of label which is used as subtitle in this case.
                    # error: "Index" has no attribute "levels"; maybe "nlevels"?
                    levels = self.data.columns.levels  # type: ignore[attr-defined]
                    ticklabels = [pprint_thing(col) for col in levels[0]]
                else:
                    ticklabels = [pprint_thing(label)]

                ret, bp = self._plot(
                    ax, y, column_num=i, return_type=self.return_type, **kwds
                )
                self.maybe_color_bp(bp)
                obj_axes.append(ret)
                obj_labels.append(label)
                _set_ticklabels(
                    ax=ax, labels=ticklabels, is_vertical=self.orientation == "vertical"
                )
            self._return_obj = pd.Series(obj_axes, index=obj_labels, dtype=object)
        else:
            y = self.data.values.T
            ax = self._get_ax(0)
            kwds = self.kwds.copy()

            ret, bp = self._plot(
                ax, y, column_num=0, return_type=self.return_type, **kwds
            )
            self.maybe_color_bp(bp)
            self._return_obj = ret

            labels = [pprint_thing(left) for left in self.data.columns]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            _set_ticklabels(
                ax=ax, labels=labels, is_vertical=self.orientation == "vertical"
            )

    def _make_legend(self) -> None:
        # [FUNCTION DOC]
        # Intentionally does nothing: box plots have no legend.
        pass

    def _post_plot_logic(self, ax: Axes, data) -> None:
        # [FUNCTION DOC]
        # Post-drawing hook: sets the x and y axis labels (printed form) when the plot
        # has them, since box plots would otherwise ignore them. The data argument is
        # unused.
        # GH 45465: make sure that the boxplot doesn't ignore xlabel/ylabel
        if self.xlabel:
            ax.set_xlabel(pprint_thing(self.xlabel))
        if self.ylabel:
            ax.set_ylabel(pprint_thing(self.ylabel))

    @property
    def orientation(self) -> Literal["horizontal", "vertical"]:
        # [FUNCTION DOC]
        # Property: returns vertical unless the keyword vert is given and false, in which
        # case horizontal.
        if self.kwds.get("vert", True):
            return "vertical"
        else:
            return "horizontal"

    @property
    def result(self):
        # [FUNCTION DOC]
        # Property: the plot result. When the return type is none it defers to the base
        # class result (axes); otherwise returns the stored return object.
        if self.return_type is None:
            return super().result
        else:
            return self._return_obj


def maybe_color_bp(bp, color_tup, **kwds) -> None:
    # [FUNCTION DOC]
    # Recolors a drawn box plot dictionary. Inputs: bp is the dictionary of artists
    # (boxes, whiskers, medians, caps); color_tup is a four-item sequence of colors in
    # that order; extra keywords are the user's matplotlib properties. For each of
    # the four artist groups, sets color (with full opacity) only when the user did
    # not supply the corresponding property dictionary keyword (box, whisker,
    # median, cap properties). Mutates artists; returns nothing.
    # GH#30346, when users specifying those arguments explicitly, our defaults
    # for these four kwargs should be overridden; if not, use Pandas settings
    if not kwds.get("boxprops"):
        mpl.artist.setp(bp["boxes"], color=color_tup[0], alpha=1)
    if not kwds.get("whiskerprops"):
        mpl.artist.setp(bp["whiskers"], color=color_tup[1], alpha=1)
    if not kwds.get("medianprops"):
        mpl.artist.setp(bp["medians"], color=color_tup[2], alpha=1)
    if not kwds.get("capprops"):
        mpl.artist.setp(bp["caps"], color=color_tup[3], alpha=1)


def _grouped_plot_by_column(
    plotf,
    data,
    columns=None,
    by=None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] | None = None,
    ax=None,
    layout=None,
    return_type=None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: draws one box plot per selected column, grouped by key columns, each in
    # its own subplot. Inputs: plotf is the per-axes drawing callable taking keys,
    # values and axes; data is the frame; columns (default none, meaning all numeric
    # columns except the grouping ones); by is the grouping spec;
    # KNOWN ISSUE: numeric_only (default true) is accepted but never used in the body.
    # grid (default false) toggles axes grid; figsize; ax; layout;
    # return_type; extra keywords (sharex and sharey default true, xlabel, ylabel
    # popped; the rest passed on). Groups the data (not dropping unobserved
    # categories), creates the subplots grid, puts the by label on the category
    # axis when the user did not give one (x for vertical boxes else y), draws each
    # column and titles each axes with the column. Builds a series of drawing
    # results indexed by column; when return_type is none the axes array is
    # returned instead. Sets the figure super title naming the grouping and adjusts
    # the figure margins. Returns the series or axes array.
    grouped = data.groupby(by, observed=False)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    naxes = len(columns)
    fig, axes = create_subplots(
        naxes=naxes,
        sharex=kwargs.pop("sharex", True),
        sharey=kwargs.pop("sharey", True),
        figsize=figsize,
        ax=ax,
        layout=layout,
    )

    # GH 45465: move the "by" label based on "vert"
    xlabel, ylabel = kwargs.pop("xlabel", None), kwargs.pop("ylabel", None)
    if kwargs.get("vert", True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by

    ax_values = []

    for ax, col in zip(flatten_axes(axes), columns, strict=False):
        gp_col = grouped[col]
        keys, values = zip(*gp_col, strict=True)
        re_plotf = plotf(keys, values, ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax.set_title(col)
        ax_values.append(re_plotf)
        ax.grid(grid)

    result = pd.Series(ax_values, index=columns, copy=False)

    # Return axes in multiplot case, maybe revisit later # 985
    if return_type is None:
        result = axes

    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f"Boxplot grouped by {byline}")
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)

    return result


def boxplot(
    data,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    # [FUNCTION DOC]
    # Backend box plot function behind the frame and series box plot front ends.
    # Inputs: data (frame or series; a series is converted to a one-column frame),
    # column (label or list or none), by (grouping or none), ax, fontsize (tick
    # label size), rot (label rotation, default 0), grid (default true), figsize,
    # layout, return_type (none, axes, dict, both), extra keywords for matplotlib.
    # KNOWN ISSUE: the invalid return type message below lists only axes, dict and both although none is also accepted.
    # Raises a value error for an invalid return type, for a layout given
    # without by, and when there are no numeric columns.
    # With by: delegates to the grouped helper and returns its result (series of
    # results, or axes array when return_type is none). Without by: return type
    # defaults to axes, draws on the given axes or the current one (with figsize
    # applied through a temporary rc context), uses numeric columns (restricted to
    # the requested ones), draws all in one axes and applies the grid. Returns the
    # object chosen by return_type.
    import matplotlib.pyplot as plt

    # validate return_type:
    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")

    if isinstance(data, ABCSeries):
        data = data.to_frame("x")
        column = "x"

    def _get_colors():
        # [FUNCTION DOC]
        # HELPER: builds the four-color array (boxes, whiskers, medians, caps) for the
        # enclosing box plot function. Defaults come from the standard colors with
        # caps black. Pops the color keyword from the enclosing keywords; a dict-like
        # value replaces entries by the keys boxes, whiskers, medians, caps (value error
        # for any other key); any other truthy value fills all four entries. Returns the
        # array.
        #  num_colors=3 is required as method maybe_color_bp takes the colors
        #  in positions 0 and 2.
        #  if colors not provided, use same defaults as DataFrame.plot.box
        result_list = get_standard_colors(num_colors=3)
        result = np.take(result_list, [0, 0, 2])
        result = np.append(result, "k")

        colors = kwds.pop("color", None)
        if colors:
            if is_dict_like(colors):
                # replace colors in result array with user-specified colors
                # taken from the colors dict parameter
                # "boxes" value placed in position 0, "whiskers" in 1, etc.
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                key_to_index = dict(zip(valid_keys, range(4), strict=True))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
            else:
                result.fill(colors)

        return result

    def plot_group(keys, values, ax: Axes, **kwds):
        # [FUNCTION DOC]
        # HELPER: draws one set of boxes on one axes for the enclosing function.
        # Inputs: keys (labels), values (arrays per label), ax, extra keywords. Pops
        # xlabel and ylabel and applies them if set. Drops missing values from each
        # array, draws with matplotlib, applies the enclosing font size to ticks if
        # set, sets tick labels from the printed keys with the enclosing rotation, and
        # recolors the artists using the enclosing colors. Returns the artist dictionary
        # for dict, the named tuple for both, otherwise the axes.
        # GH 45465: xlabel/ylabel need to be popped out before plotting happens
        xlabel, ylabel = kwds.pop("xlabel", None), kwds.pop("ylabel", None)
        if xlabel:
            ax.set_xlabel(pprint_thing(xlabel))
        if ylabel:
            ax.set_ylabel(pprint_thing(ylabel))

        keys = [pprint_thing(x) for x in keys]
        values = [remove_na_arraylike(v) for v in values]
        bp = ax.boxplot(values, **kwds)
        if fontsize is not None:
            ax.tick_params(axis="both", labelsize=fontsize)

        # GH 45465: x/y are flipped when "vert" changes
        _set_ticklabels(
            ax=ax, labels=keys, is_vertical=kwds.get("vert", True), rotation=rot
        )
        maybe_color_bp(bp, color_tup=colors, **kwds)

        # Return axes in multiplot case, maybe revisit later # 985
        if return_type == "dict":
            return bp
        elif return_type == "both":
            return BoxPlot.BP(ax=ax, lines=bp)
        else:
            return ax

    colors = _get_colors()
    if column is None:
        columns = None
    elif isinstance(column, (list, tuple)):
        columns = column
    else:
        columns = [column]

    if by is not None:
        # Prefer array return type for 2-D plots to match the subplot layout
        # https://github.com/pandas-dev/pandas/pull/12216#issuecomment-241175580
        result = _grouped_plot_by_column(
            plot_group,
            data,
            columns=columns,
            by=by,
            grid=grid,
            figsize=figsize,
            ax=ax,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
    else:
        if return_type is None:
            return_type = "axes"
        if layout is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")

        if ax is None:
            # Any: rc_context's accepted key type narrowed in matplotlib 3.11
            rc: Any = {"figure.figsize": figsize} if figsize is not None else {}
            with mpl.rc_context(rc):
                ax = plt.gca()
        data = data._get_numeric_data()
        naxes = len(data.columns)
        if naxes == 0:
            raise ValueError(
                "boxplot method requires numerical columns, nothing to plot."
            )
        if columns is None:
            columns = data.columns
        else:
            data = data[columns]

        result = plot_group(columns, data.values.T, ax, **kwds)
        ax.grid(grid)

    return result


def boxplot_frame(
    self,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    # [FUNCTION DOC]
    # Backend frame box plot method. Takes the same inputs as the box plot function
    # (column, by, ax, fontsize, rot, grid, figsize, layout, return_type and extra
    # keywords) with self as the frame; calls it, then asks pyplot to redraw if
    # interactive, and returns its result.
    import matplotlib.pyplot as plt

    ax = boxplot(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        grid=grid,
        rot=rot,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwds,
    )
    plt.draw_if_interactive()
    return ax


def boxplot_frame_groupby(
    grouped,
    subplots: bool = True,
    column=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax=None,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds,
):
    # [FUNCTION DOC]
    # Backend grouped box plot. Inputs: grouped (grouped frame object), subplots
    # (default true), column, fontsize, rot, grid, ax, figsize, layout, sharex
    # (default false), sharey (default true), extra keywords. With subplots true:
    # one subplot per group, each drawn by the group's own box plot method with the
    # group key as title; returns a series of results indexed by key and adjusts the
    # figure margins. Otherwise concatenates all groups side by side with keys as an
    # outer column level, converts the column selection to pairs of group key and
    # column (when given) and draws one box plot; returns that result.
    if subplots is True:
        naxes = len(grouped)
        fig, axes = create_subplots(
            naxes=naxes,
            squeeze=False,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
        )
        data = {}
        for (key, group), ax in zip(grouped, flatten_axes(axes), strict=False):
            d = group.boxplot(
                ax=ax, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds
            )
            ax.set_title(pprint_thing(key))
            data[key] = d
        ret = pd.Series(data)
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        keys, frames = zip(*grouped, strict=True)
        df = pd.concat(frames, keys=keys, axis=1)

        # GH 16748, DataFrameGroupby fails when subplots=False and `column` argument
        # is assigned, and in this case, since `df` here becomes MI after groupby,
        # so we need to couple the keys (grouped values) and column (original df
        # column) together to search for subset to plot
        if column is not None:
            column = com.convert_to_list_like(column)
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        ret = df.boxplot(
            column=column,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            ax=ax,
            figsize=figsize,
            layout=layout,
            **kwds,
        )
    return ret
