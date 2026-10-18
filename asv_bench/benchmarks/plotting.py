# [MODULE DOC]
# Benchmarks for plotting: series and frame plots of many kinds, time-series
# plotting variants, the Andrews-curves helper, and plotting backend discovery.
# Public contents are the benchmark classes SeriesPlotting, FramePlotting,
# TimeseriesPlotting, Misc and BackendLoading. Requires matplotlib at import time.
# Import-time side effects: imports the Andrews-curves helper from the public
# plotting package (falling back to the old tools location), imports the private
# backend lookup function, and forces matplotlib to use its non-interactive Agg
# rendering backend globally (so no windows are opened). Final line re-exports the
# common asv setup hook.
import contextlib
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import tempfile
from unittest import mock

import matplotlib
import numpy as np

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
)

try:
    from pandas.plotting import andrews_curves
except ImportError:
    from pandas.tools.plotting import andrews_curves

from pandas.plotting._core import _get_plot_backend

matplotlib.use("Agg")


class SeriesPlotting:
    # [CLASS DOC]
    # Benchmark of plotting a random series with each supported plot kind.
    # Parametrized by kind. State: s.
    params = [["line", "bar", "area", "barh", "hist", "kde", "pie"]]
    param_names = ["kind"]

    def setup(self, kind):
        # [FUNCTION DOC]
        # Chooses a size by kind: 100 for bar, horizontal bar and pie; 10000 for
        # density estimate; one million otherwise. Builds a random normal series
        # of that size as s, converted to absolute values for the area and pie kinds
        # (which need non-negative data).
        if kind in ["bar", "barh", "pie"]:
            n = 100
        elif kind in ["kde"]:
            n = 10000
        else:
            n = 1000000

        self.s = Series(np.random.randn(n))
        if kind in ["area", "pie"]:
            self.s = self.s.abs()

    def time_series_plot(self, kind):
        # [FUNCTION DOC]
        # Times plotting the series with the plot kind (drawing happens on the
        # non-interactive backend; the figure is not closed here).
        self.s.plot(kind=kind)


class FramePlotting:
    # [CLASS DOC]
    # Benchmark of plotting one column against another of a two-column random
    # frame with each supported plot kind. Parametrized by kind. State: x, y and
    # df.
    params = [
        ["line", "bar", "area", "barh", "hist", "kde", "pie", "scatter", "hexbin"]
    ]
    param_names = ["kind"]

    def setup(self, kind):
        # [FUNCTION DOC]
        # Chooses a size by kind: 100 for bar, horizontal bar and pie; 10000 for
        # density estimate, scatter and hexagonal binning; one million otherwise.
        # Builds two independent random normal series x and y of that size
        # (absolute values for the area and pie kinds) and a frame df with them as
        # columns x and y.
        if kind in ["bar", "barh", "pie"]:
            n = 100
        elif kind in ["kde", "scatter", "hexbin"]:
            n = 10000
        else:
            n = 1000000

        self.x = Series(np.random.randn(n))
        self.y = Series(np.random.randn(n))
        if kind in ["area", "pie"]:
            self.x = self.x.abs()
            self.y = self.y.abs()
        self.df = DataFrame({"x": self.x, "y": self.y})

    def time_frame_plot(self, kind):
        # [FUNCTION DOC]
        # Times plotting column y against column x with the plot kind.
        self.df.plot(x="x", y="y", kind=kind)


class TimeseriesPlotting:
    # [CLASS DOC]
    # Benchmark of plotting a five-column frame with a daily datetime index,
    # regular or with missing days, with compatibility and table options. No
    # parameters. State: df (regular) and df2 (irregular).
    def setup(self):
        # [FUNCTION DOC]
        # Builds a daily index of 2000 entries from 1975-01-01 and a 2000 by 5
        # random normal frame df. Builds an irregular index by removing two entries
        # (positions 10 and 11) and a random frame df2 on it.
        N = 2000
        M = 5
        idx = date_range("1/1/1975", periods=N)
        self.df = DataFrame(np.random.randn(N, M), index=idx)

        idx_irregular = DatetimeIndex(
            np.concatenate((idx.values[0:10], idx.values[12:]))
        )
        self.df2 = DataFrame(
            np.random.randn(len(idx_irregular), M), index=idx_irregular
        )

    def time_plot_regular(self):
        # [FUNCTION DOC]
        # Times plotting the regular frame.
        self.df.plot()

    def time_plot_regular_compat(self):
        # [FUNCTION DOC]
        # Times plotting the regular frame with the matplotlib-compatible x axis.
        self.df.plot(x_compat=True)

    def time_plot_irregular(self):
        # [FUNCTION DOC]
        # Times plotting the irregular frame.
        self.df2.plot()

    def time_plot_table(self):
        # [FUNCTION DOC]
        # Times plotting the regular frame with a data table drawn below.
        self.df.plot(table=True)


class Misc:
    # [CLASS DOC]
    # Benchmark of the Andrews-curves plotting helper. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 500 by 10 random normal frame with an added class column Name
        # holding the constant A, stored as df.
        N = 500
        M = 10
        self.df = DataFrame(np.random.randn(N, M))
        self.df["Name"] = ["A"] * N

    def time_plot_andrews_curves(self):
        # [FUNCTION DOC]
        # Times drawing Andrews curves with the Name column as the class column.
        andrews_curves(self.df, "Name")


class BackendLoading:
    # [CLASS DOC]
    # Benchmark of looking up a plotting backend from installed entry points, with
    # one lookup hitting the first entry point and one scanning through all of them
    # before falling back to importing a module. Runs once per measurement
    # (single repeat, single call, no warm-up). State: stack, a saved exit stack
    # that undoes the temporary global changes made in setup.
    repeat = 1
    number = 1
    warmup_time = 0

    def setup(self):
        # [FUNCTION DOC]
        # Builds a dummy module named pandas_dummy_backend whose plot function
        # returns one. Inside an exit stack: temporarily registers the dummy module
        # in the module table, creates a temporary directory, prepends it to the
        # module search path, and writes a package metadata directory containing an
        # entry-points file that declares eleven plotting backend entry points
        # (names my_ep_backend and my_ep_backend0 to my_ep_backend9) all pointing at
        # the dummy module. The stack is detached (so cleanup is deferred) and
        # saved as stack; global state (module table, search path, temporary
        # directory) stays modified until teardown.
        mod = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec("pandas_dummy_backend", None)
        )
        mod.plot = lambda *args, **kwargs: 1

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(sys.modules, {"pandas_dummy_backend": mod})
            )
            tmp_path = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))

            sys.path.insert(0, os.fsdecode(tmp_path))
            stack.callback(sys.path.remove, os.fsdecode(tmp_path))

            dist_info = tmp_path / "my_backend-0.0.0.dist-info"
            dist_info.mkdir()
            (dist_info / "entry_points.txt").write_bytes(
                b"[pandas_plotting_backends]\n"
                b"my_ep_backend = pandas_dummy_backend\n"
                b"my_ep_backend0 = pandas_dummy_backend\n"
                b"my_ep_backend1 = pandas_dummy_backend\n"
                b"my_ep_backend2 = pandas_dummy_backend\n"
                b"my_ep_backend3 = pandas_dummy_backend\n"
                b"my_ep_backend4 = pandas_dummy_backend\n"
                b"my_ep_backend5 = pandas_dummy_backend\n"
                b"my_ep_backend6 = pandas_dummy_backend\n"
                b"my_ep_backend7 = pandas_dummy_backend\n"
                b"my_ep_backend8 = pandas_dummy_backend\n"
                b"my_ep_backend9 = pandas_dummy_backend\n"
            )
            self.stack = stack.pop_all()

    def teardown(self):
        # [FUNCTION DOC]
        # asv teardown hook. Closes the saved exit stack, which restores the module
        # table and search path and deletes the temporary directory.
        self.stack.close()

    def time_get_plot_backend(self):
        # [FUNCTION DOC]
        # Times looking up the backend by the name of the first entry point.
        # finds the first my_ep_backend
        _get_plot_backend("my_ep_backend")

    def time_get_plot_backend_fallback(self):
        # [FUNCTION DOC]
        # Times looking up the backend by a module name that matches none of the
        # entry-point names, so it scans all entry points before falling back to
        # importing the module by name.
        # iterates through all the my_ep_backend[0-9] before falling back
        # to importlib.import_module
        _get_plot_backend("pandas_dummy_backend")


from .pandas_vb_common import setup  # noqa: F401 isort:skip
