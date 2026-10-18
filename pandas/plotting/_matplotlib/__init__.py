# [PACKAGE DOC]
# HELPER: internal matplotlib plotting backend used by the public plotting
# package (selected through the backend dispatch in the plotting core module).
# Submodules: core (base plot classes and line, bar, area, pie, scatter,
# hexbin plots), boxplot, hist (histogram and kde), misc (stand-alone plots),
# timeseries, converter (unit converters for dates), tools, style, groupby.
# Exports: the backend API required of a plotting backend (plot, hist_frame,
# hist_series, boxplot, boxplot_frame, boxplot_frame_groupby, register,
# deregister, table, andrews_curves, autocorrelation_plot, bootstrap_plot,
# lag_plot, parallel_coordinates, radviz, scatter_matrix) plus the mapping
# from kind names to plot classes.
# Import-time side effects: imports all backend submodules; the converter
# module imports matplotlib-related machinery but registration of converters
# is not performed here.
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.plotting._matplotlib.boxplot import (
    BoxPlot,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
)
from pandas.plotting._matplotlib.converter import (
    deregister,
    register,
)
from pandas.plotting._matplotlib.core import (
    AreaPlot,
    BarhPlot,
    BarPlot,
    HexBinPlot,
    LinePlot,
    PiePlot,
    ScatterPlot,
)
from pandas.plotting._matplotlib.hist import (
    HistPlot,
    KdePlot,
    hist_frame,
    hist_series,
)
from pandas.plotting._matplotlib.misc import (
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    lag_plot,
    parallel_coordinates,
    radviz,
    scatter_matrix,
)
from pandas.plotting._matplotlib.tools import table

if TYPE_CHECKING:
    from pandas.plotting._matplotlib.core import MPLPlot

PLOT_CLASSES: dict[str, type[MPLPlot]] = {
    "line": LinePlot,
    "bar": BarPlot,
    "barh": BarhPlot,
    "box": BoxPlot,
    "hist": HistPlot,
    "kde": KdePlot,
    "area": AreaPlot,
    "pie": PiePlot,
    "scatter": ScatterPlot,
    "hexbin": HexBinPlot,
}


def plot(data, kind, **kwargs):
    # [FUNCTION DOC]
    # Backend entry point for the generic plot call on a series or frame.
    # Inputs: data is the series or frame to plot; kind is one of the keys of
    # the plot class mapping (line, bar, barh, box, hist, kde, area, pie,
    # scatter, hexbin); remaining keyword arguments go to the plot class.
    # A reuse_plot keyword (default false) is removed from the arguments; when
    # true and no axes were given while a pyplot figure exists, the current
    # axes are used (preferring the left axes of a secondary-axis pair).
    # Imports pyplot lazily so that converters are registered first.
    # Builds the plot object, generates it, triggers an interactive redraw if
    # interactive mode is on, and returns the result stored on the plot object
    # (axes or array of axes, depending on the kind).
    # Raises a key error for an unknown kind.
    # Importing pyplot at the top of the file (before the converters are
    # registered) causes problems in matplotlib 2 (converters seem to not
    # work)
    import matplotlib.pyplot as plt

    if kwargs.pop("reuse_plot", False):
        ax = kwargs.get("ax")
        if ax is None and len(plt.get_fignums()) > 0:
            with plt.rc_context():
                ax = plt.gca()
            kwargs["ax"] = getattr(ax, "left_ax", ax)
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    plot_obj.generate()
    plt.draw_if_interactive()
    return plot_obj.result


__all__ = [
    "andrews_curves",
    "autocorrelation_plot",
    "bootstrap_plot",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "deregister",
    "hist_frame",
    "hist_series",
    "lag_plot",
    "parallel_coordinates",
    "plot",
    "radviz",
    "register",
    "scatter_matrix",
    "table",
]
