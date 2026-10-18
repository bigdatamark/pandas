# [MODULE DOC]
# Benchmarks for window operations: rolling, expanding and exponentially weighted
# windows with built-in statistics, apply with user functions, the numba engine,
# variable and forward-looking windows, pairwise correlation, quantile, rank,
# group-wise windows, and table-wise methods. Public contents are the benchmark
# classes Methods, Apply, NumbaEngineMethods, NumbaEngineApply, EWMMethods,
# VariableWindowMethods, Pairwise, Quantile, Rank, PeakMemFixedWindowMinMax,
# ForwardWindowMethods, Groupby, GroupbyLargeGroups, GroupbyEWM, GroupbyEWMEngine
# and TableMethod, plus the helper function table_method_func. Import-time side
# effect: the parameter list of the exponentially weighted benchmark builds a
# 100000-entry datetime index when the module is imported. Final line re-exports
# the common asv setup hook.
import warnings

import numpy as np

import pandas as pd


class Methods:
    # [CLASS DOC]
    # Benchmark of built-in window statistics on 100000 random numbers for
    # frames/series, window specs (rolling with 10, rolling with 1000, expanding),
    # integer/float data and many statistic names. Parametrized by constructor
    # name, (window kind, keyword arguments) pair, dtype and method. Timing for
    # both speed and peak memory. State: window. Subclassed by the variable window
    # benchmark.
    params = (
        ["DataFrame", "Series"],
        [("rolling", {"window": 10}), ("rolling", {"window": 1000}), ("expanding", {})],
        ["int", "float"],
        [
            "median",
            "mean",
            "max",
            "min",
            "std",
            "count",
            "skew",
            "kurt",
            "sum",
            "sem",
            "nunique",
        ],
    )
    param_names = ["constructor", "window_kwargs", "dtype", "method"]

    def setup(self, constructor, window_kwargs, dtype, method):
        # [FUNCTION DOC]
        # Builds 100000 random numbers scaled to 0..100 and cast to the dtype,
        # wraps them in the frame or series class, and creates the window object
        # of the requested kind with the keyword arguments; stores it as window.
        # The method parameter is unused here.
        N = 10**5
        window, kwargs = window_kwargs
        arr = (100 * np.random.random(N)).astype(dtype)
        obj = getattr(pd, constructor)(arr)
        self.window = getattr(obj, window)(**kwargs)

    def time_method(self, constructor, window_kwargs, dtype, method):
        # [FUNCTION DOC]
        # Times calling the named statistic on the window object.
        getattr(self.window, method)()

    def peakmem_method(self, constructor, window_kwargs, dtype, method):
        # [FUNCTION DOC]
        # Peak-memory benchmark of calling the named statistic.
        getattr(self.window, method)()


class Apply:
    # [CLASS DOC]
    # Benchmark of applying a user function over rolling windows, in raw or object
    # mode. Parametrized by constructor, window size (3, 300), dtype, function
    # (the built-in sum, the numpy sum, or a lambda adding five to the numpy sum)
    # and raw flag. State: roll.
    params = (
        ["DataFrame", "Series"],
        [3, 300],
        ["int", "float"],
        [sum, np.sum, lambda x: np.sum(x) + 5],
        [True, False],
    )
    param_names = ["constructor", "window", "dtype", "function", "raw"]

    def setup(self, constructor, window, dtype, function, raw):
        # [FUNCTION DOC]
        # Builds 1000 random numbers scaled to 0..100, cast to the dtype, wraps
        # them in the frame or series class and creates a rolling window of the
        # given size as roll. The function and raw parameters are unused here.
        N = 10**3
        arr = (100 * np.random.random(N)).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    def time_rolling(self, constructor, window, dtype, function, raw):
        # [FUNCTION DOC]
        # Times applying the function over the windows, passing arrays (raw) or
        # series according to the flag.
        self.roll.apply(function, raw=raw)


class NumbaEngineMethods:
    # [CLASS DOC]
    # Intended benchmark of built-in window statistics with the numba engine, for
    # one or many columns, serial or parallel. Parametrized by constructor, dtype,
    # (window kind, kwargs), method, parallel flag and number of columns.
    # KNOWN ISSUE: the only benchmark method is named with a test prefix rather
    # than the time prefix required by asv, so asv does not collect it and the
    # class never runs as a benchmark.
    # State: window.
    params = (
        ["DataFrame", "Series"],
        ["int", "float"],
        [("rolling", {"window": 10}), ("expanding", {})],
        ["sum", "max", "min", "median", "mean", "var", "std"],
        [True, False],
        [None, 100],
    )
    param_names = [
        "constructor",
        "dtype",
        "window_kwargs",
        "method",
        "parallel",
        "cols",
    ]

    def setup(self, constructor, dtype, window_kwargs, method, parallel, cols):
        # [FUNCTION DOC]
        # Builds 1000 random numbers scaled to 0..100 (shaped 1000 by cols for a
        # frame when cols is given, else one-dimensional), cast to the dtype, wraps
        # them in the class, creates the window object, and warms the numba
        # compilation cache by running the statistic once with the numba engine and
        # the parallel option, with warnings (for example parallel not being
        # applicable to one-dimensional data) recorded and discarded. Stores the
        # window object.
        N = 10**3
        window, kwargs = window_kwargs
        shape = (N, cols) if cols is not None and constructor != "Series" else N
        arr = (100 * np.random.random(shape)).astype(dtype)
        data = getattr(pd, constructor)(arr)

        # Warm the cache
        with warnings.catch_warnings(record=True):
            # Catch parallel=True not being applicable e.g. 1D data
            self.window = getattr(data, window)(**kwargs)
            getattr(self.window, method)(
                engine="numba", engine_kwargs={"parallel": parallel}
            )

    def test_method(self, constructor, dtype, window_kwargs, method, parallel, cols):
        # [FUNCTION DOC]
        # Would time running the statistic with the numba engine, with warnings
        # recorded and discarded. Not collected by asv because of the method name
        # (see the class note).
        with warnings.catch_warnings(record=True):
            getattr(self.window, method)(
                engine="numba", engine_kwargs={"parallel": parallel}
            )


class NumbaEngineApply:
    # [CLASS DOC]
    # Intended benchmark of applying a user function over windows with the numba
    # engine, for one or many columns, serial or parallel. Parametrized by
    # constructor, dtype, (window kind, kwargs), function (numpy sum, or a lambda
    # adding five to it), parallel flag and number of columns.
    # KNOWN ISSUE: the only benchmark method is named with a test prefix rather
    # than the time prefix required by asv, so asv does not collect it.
    # State: window.
    params = (
        ["DataFrame", "Series"],
        ["int", "float"],
        [("rolling", {"window": 10}), ("expanding", {})],
        [np.sum, lambda x: np.sum(x) + 5],
        [True, False],
        [None, 100],
    )
    param_names = [
        "constructor",
        "dtype",
        "window_kwargs",
        "function",
        "parallel",
        "cols",
    ]

    def setup(self, constructor, dtype, window_kwargs, function, parallel, cols):
        # [FUNCTION DOC]
        # Builds 1000 random numbers as in the previous numba class, creates the
        # window object, and warms the numba cache by applying the function in raw
        # mode with the numba engine and the parallel option, with warnings
        # recorded and discarded. Stores the window object.
        N = 10**3
        window, kwargs = window_kwargs
        shape = (N, cols) if cols is not None and constructor != "Series" else N
        arr = (100 * np.random.random(shape)).astype(dtype)
        data = getattr(pd, constructor)(arr)

        # Warm the cache
        with warnings.catch_warnings(record=True):
            # Catch parallel=True not being applicable e.g. 1D data
            self.window = getattr(data, window)(**kwargs)
            self.window.apply(
                function, raw=True, engine="numba", engine_kwargs={"parallel": parallel}
            )

    def test_method(self, constructor, dtype, window_kwargs, function, parallel, cols):
        # [FUNCTION DOC]
        # Would time applying the function in raw mode with the numba engine, with
        # warnings recorded and discarded. Not collected by asv because of the
        # method name.
        with warnings.catch_warnings(record=True):
            self.window.apply(
                function, raw=True, engine="numba", engine_kwargs={"parallel": parallel}
            )


class EWMMethods:
    # [CLASS DOC]
    # Benchmark of exponentially weighted window statistics (mean and standard
    # deviation) with a half-life of 10 or 1000 observations, or a time-based
    # half-life with explicit times. Parametrized by constructor, (keyword
    # arguments, method) pair and dtype. State: method and ewm. The time-based
    # parameter embeds a 100000-entry datetime index built at import.
    params = (
        ["DataFrame", "Series"],
        [
            ({"halflife": 10}, "mean"),
            ({"halflife": 10}, "std"),
            ({"halflife": 1000}, "mean"),
            ({"halflife": 1000}, "std"),
            (
                {
                    "halflife": "1 Day",
                    "times": pd.date_range("1900", periods=10**5, freq="23s"),
                },
                "mean",
            ),
        ],
        ["int", "float"],
    )
    param_names = ["constructor", "kwargs_method", "dtype"]

    def setup(self, constructor, kwargs_method, dtype):
        # [FUNCTION DOC]
        # Builds 100000 random numbers scaled to 0..100, cast to the dtype, wraps
        # them in the class, and creates the exponentially weighted window with the
        # keyword arguments; stores the window as ewm and the statistic name as
        # method.
        N = 10**5
        kwargs, method = kwargs_method
        arr = (100 * np.random.random(N)).astype(dtype)
        self.method = method
        self.ewm = getattr(pd, constructor)(arr).ewm(**kwargs)

    def time_ewm(self, constructor, kwargs_method, dtype):
        # [FUNCTION DOC]
        # Times calling the stored statistic on the weighted window.
        getattr(self.ewm, self.method)()


class VariableWindowMethods(Methods):
    # [CLASS DOC]
    # Variant of the window statistic benchmark using time-based (offset string)
    # windows on a datetime-indexed object. Inherits the timing methods
    # (time_method and peakmem_method) from the parent class and overrides the
    # parameters and setup. Parametrized by constructor, window offset (50 seconds,
    # 1 hour, 1 day), dtype and method.
    params = (
        ["DataFrame", "Series"],
        ["50s", "1h", "1d"],
        ["int", "float"],
        ["median", "mean", "max", "min", "std", "count", "skew", "kurt", "sum", "sem"],
    )
    param_names = ["constructor", "window", "dtype", "method"]

    def setup(self, constructor, window, dtype, method):
        # [FUNCTION DOC]
        # Builds 100000 random numbers scaled to 0..100 cast to the dtype, indexed
        # by timestamps every five seconds from 2017-01-01, wraps them in the class
        # and creates a rolling window with the time-offset size; stores as window.
        # The method parameter is unused here.
        N = 10**5
        arr = (100 * np.random.random(N)).astype(dtype)
        index = pd.date_range("2017-01-01", periods=N, freq="5s")
        self.window = getattr(pd, constructor)(arr, index=index).rolling(window)


class Pairwise:
    # [CLASS DOC]
    # Benchmark of rolling or expanding correlation and covariance of a frame with
    # itself, ungrouped and grouped, with the pairwise option on or off.
    # Parametrized by (keyword arguments, window kind) pair, method (corr, cov) and
    # the pairwise flag. State: df, window and window_group.
    params = (
        [({"window": 10}, "rolling"), ({"window": 1000}, "rolling"), ({}, "expanding")],
        ["corr", "cov"],
        [True, False],
    )
    param_names = ["window_kwargs", "method", "pairwise"]

    def setup(self, kwargs_window, method, pairwise):
        # [FUNCTION DOC]
        # Builds 10000 random numbers as a one-column frame df; creates the
        # ungrouped window object of the requested kind with the keyword
        # arguments as window; and creates the same window kind on the same data
        # grouped by twenty cyclic group labels as window_group. The method and
        # pairwise parameters are unused here.
        N = 10**4
        n_groups = 20
        kwargs, window = kwargs_window
        groups = [i for _ in range(N // n_groups) for i in range(n_groups)]
        arr = np.random.random(N)
        self.df = pd.DataFrame(arr)
        self.window = getattr(self.df, window)(**kwargs)
        self.window_group = getattr(
            pd.DataFrame({"A": groups, "B": arr}).groupby("A"), window
        )(**kwargs)

    def time_pairwise(self, kwargs_window, method, pairwise):
        # [FUNCTION DOC]
        # Times calling the correlation or covariance of the window against the
        # frame with the pairwise option.
        getattr(self.window, method)(self.df, pairwise=pairwise)

    def time_groupby(self, kwargs_window, method, pairwise):
        # [FUNCTION DOC]
        # Times calling the correlation or covariance of the grouped window against
        # the frame with the pairwise option.
        getattr(self.window_group, method)(self.df, pairwise=pairwise)


class Quantile:
    # [CLASS DOC]
    # Benchmark of rolling quantiles for frames and series.
    # KNOWN ISSUE: the parameter name list has four names but there are five
    # parameter lists (the interpolation list is unnamed).
    # Parametrized by constructor, window size, dtype, quantile and interpolation
    # method. State: roll.
    params = (
        ["DataFrame", "Series"],
        [10, 1000],
        ["int", "float"],
        [0, 0.5, 1],
        ["linear", "nearest", "lower", "higher", "midpoint"],
    )
    param_names = ["constructor", "window", "dtype", "percentile"]

    def setup(self, constructor, window, dtype, percentile, interpolation):
        # [FUNCTION DOC]
        # Builds 100000 uniform random numbers cast to the dtype (so integer dtypes
        # become all zeros), wraps them in the class and creates a rolling window of
        # the given size as roll. The percentile and interpolation parameters are
        # unused here.
        N = 10**5
        arr = np.random.random(N).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    def time_quantile(self, constructor, window, dtype, percentile, interpolation):
        # [FUNCTION DOC]
        # Times the rolling quantile for the given quantile and interpolation.
        self.roll.quantile(percentile, interpolation=interpolation)


class Rank:
    # [CLASS DOC]
    # Benchmark of rolling rank with percentile output on or off, ascending or
    # descending, and different tie methods. Parametrized by constructor, window
    # size, dtype, percentile flag, ascending flag and tie method. State: roll.
    params = (
        ["DataFrame", "Series"],
        [10, 1000],
        ["int", "float"],
        [True, False],
        [True, False],
        ["min", "max", "average"],
    )
    param_names = [
        "constructor",
        "window",
        "dtype",
        "percentile",
        "ascending",
        "method",
    ]

    def setup(self, constructor, window, dtype, percentile, ascending, method):
        # [FUNCTION DOC]
        # Builds 100000 uniform random numbers cast to the dtype (integer dtypes
        # become all zeros), wraps them in the class and creates a rolling window of
        # the given size as roll. The remaining parameters are unused here.
        N = 10**5
        arr = np.random.random(N).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    def time_rank(self, constructor, window, dtype, percentile, ascending, method):
        # [FUNCTION DOC]
        # Times the rolling rank with the percentile output flag, direction and tie
        # method.
        self.roll.rank(pct=percentile, ascending=ascending, method=method)


class PeakMemFixedWindowMinMax:
    # [CLASS DOC]
    # Peak-memory benchmark of repeated rolling minimum or maximum over a
    # two-element window of a million-element series. Parametrized by operation
    # name; there is no parameter name list. State: roll.
    params = ["min", "max"]

    def setup(self, operation):
        # [FUNCTION DOC]
        # Builds a million random numbers as a series and creates a rolling window
        # of two elements as roll. The operation parameter is unused here.
        N = 10**6
        arr = np.random.random(N)
        self.roll = pd.Series(arr).rolling(2)

    def peakmem_fixed(self, operation):
        # [FUNCTION DOC]
        # Calls the named statistic on the window five times, discarding results.
        for x in range(5):
            getattr(self.roll, operation)()


class ForwardWindowMethods:
    # [CLASS DOC]
    # Benchmark of window statistics using a forward-looking fixed window.
    # Parametrized by constructor, window size, dtype and method. Timing for both
    # speed and peak memory. State: roll.
    params = (
        ["DataFrame", "Series"],
        [10, 1000],
        ["int", "float"],
        ["median", "mean", "max", "min", "kurt", "sum"],
    )
    param_names = ["constructor", "window_size", "dtype", "method"]

    def setup(self, constructor, window_size, dtype, method):
        # [FUNCTION DOC]
        # Builds 100000 uniform random numbers cast to the dtype (integer dtypes
        # become all zeros), creates a forward-looking fixed window indexer of the
        # given size, wraps the data in the class and creates a rolling window
        # using the indexer as roll. The method parameter is unused here.
        N = 10**5
        arr = np.random.random(N).astype(dtype)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        self.roll = getattr(pd, constructor)(arr).rolling(window=indexer)

    def time_rolling(self, constructor, window_size, dtype, method):
        # [FUNCTION DOC]
        # Times calling the named statistic on the forward window.
        getattr(self.roll, method)()

    def peakmem_rolling(self, constructor, window_size, dtype, method):
        # [FUNCTION DOC]
        # Peak-memory benchmark of calling the named statistic.
        getattr(self.roll, method)()


class Groupby:
    # [CLASS DOC]
    # Benchmark of window statistics on grouped data, for rolling windows of two
    # observations or 30 seconds and expanding windows. Parametrized by method and
    # (window kind, kwargs) pair; there is no parameter name list.
    # KNOWN ISSUE: the method list contains the name sum twice, producing
    # duplicate parameter combinations.
    # State: groupby_window.
    params = (
        ["sum", "median", "mean", "max", "min", "kurt", "sum"],
        [
            ("rolling", {"window": 2}),
            ("rolling", {"window": "30s"}),
            ("expanding", {}),
        ],
    )

    def setup(self, method, window_kwargs):
        # [FUNCTION DOC]
        # Builds a 10000-row frame of 1000 string group labels each repeated ten
        # times (cycled) and integer values. If the window is given as a string
        # (a time offset) gives the frame a minute-frequency datetime index from
        # 1900-01-01. Groups by the label and creates the window object of the
        # requested kind; stores as groupby_window. The method parameter is unused
        # here.
        N = 1000
        window, kwargs = window_kwargs
        df = pd.DataFrame(
            {
                "A": [str(i) for i in range(N)] * 10,
                "B": list(range(N)) * 10,
            }
        )
        if isinstance(kwargs.get("window", None), str):
            df.index = pd.date_range(start="1900-01-01", freq="1min", periods=N * 10)
        self.groupby_window = getattr(df.groupby("A"), window)(**kwargs)

    def time_method(self, method, window_kwargs):
        # [FUNCTION DOC]
        # Times calling the named statistic on the grouped window.
        getattr(self.groupby_window, method)()


class GroupbyLargeGroups:
    # [CLASS DOC]
    # Benchmark (issue 38038) of a rolling mean over a frame with few but large
    # groups, where the window computation is cheap but building the result's
    # multi-index can be expensive. No parameters. State: df.
    # https://github.com/pandas-dev/pandas/issues/38038
    # specific example where the rolling operation on a larger dataframe
    # is relatively cheap (few but large groups), but creation of
    # MultiIndex of result can be expensive

    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows with a group column A alternating between
        # one and two and a random normal column B.
        N = 100000
        self.df = pd.DataFrame({"A": [1, 2] * (N // 2), "B": np.random.randn(N)})

    def time_rolling_multiindex_creation(self):
        # [FUNCTION DOC]
        # Times grouping by A and taking the mean over rolling windows of three.
        self.df.groupby("A").rolling(3).mean()


class GroupbyEWM:
    # [CLASS DOC]
    # Benchmark of group-wise exponentially weighted statistics with fifty groups
    # of one row each. Parametrized by method (var, std, cov, corr). State: gb_ewm.
    params = ["var", "std", "cov", "corr"]
    param_names = ["method"]

    def setup(self, method):
        # [FUNCTION DOC]
        # Builds a 50-row frame with columns A and B equal to the integers below 50,
        # groups by A and creates an exponentially weighted window with centre of
        # mass one; stores as gb_ewm. The method parameter is unused here.
        df = pd.DataFrame({"A": range(50), "B": range(50)})
        self.gb_ewm = df.groupby("A").ewm(com=1.0)

    def time_groupby_method(self, method):
        # [FUNCTION DOC]
        # Times calling the named statistic on the grouped weighted window.
        getattr(self.gb_ewm, method)()


class GroupbyEWMEngine:
    # [CLASS DOC]
    # Benchmark of the group-wise exponentially weighted mean with the cython and
    # numba engines. Parametrized by engine. State: gb_ewm.
    params = ["cython", "numba"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Builds the same 50-row grouped weighted window as the previous class and
        # stores it as gb_ewm. The engine parameter is unused here.
        df = pd.DataFrame({"A": range(50), "B": range(50)})
        self.gb_ewm = df.groupby("A").ewm(com=1.0)

    def time_groupby_mean(self, engine):
        # [FUNCTION DOC]
        # Times the grouped weighted mean with the chosen engine.
        self.gb_ewm.mean(engine=engine)


def table_method_func(x):
    # [FUNCTION DOC]
    # HELPER: user function for the table-method benchmark class. Takes a
    # two-dimensional block of values, sums it along the row axis (producing one sum
    # per column) and adds one; returns the resulting array.
    return np.sum(x, axis=0) + 1


class TableMethod:
    # [CLASS DOC]
    # Benchmark of window operations with the numba engine applied per column
    # (single) or to the whole table at once (table). Parametrized by method
    # (single, table). State: df.
    params = ["single", "table"]
    param_names = ["method"]

    def setup(self, method):
        # [FUNCTION DOC]
        # Builds a 10 by 1000 random normal frame as df. The method parameter is
        # unused here.
        self.df = pd.DataFrame(np.random.randn(10, 1000))

    def time_apply(self, method):
        # [FUNCTION DOC]
        # Times applying the table helper over rolling windows of two rows in raw
        # mode with the numba engine and the single or table method.
        self.df.rolling(2, method=method).apply(
            table_method_func, raw=True, engine="numba"
        )

    def time_ewm_mean(self, method):
        # [FUNCTION DOC]
        # Times the exponentially weighted mean (alpha-style parameter one) with the
        # numba engine and the single or table method.
        self.df.ewm(1, method=method).mean(engine="numba")


from .pandas_vb_common import setup  # noqa: F401 isort:skip
