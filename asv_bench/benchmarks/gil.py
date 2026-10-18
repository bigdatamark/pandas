# [MODULE DOC]
# Benchmarks that measure how well operations release the global interpreter lock,
# by running the same operation in several threads at once and comparing against
# running it in a loop. Public contents are the decorator factory run_parallel and
# the benchmark classes ParallelGroupbyMethods, ParallelGroups, ParallelTake1D,
# ParallelKth, ParallelDatetimeFields, ParallelRolling, ParallelReadCSV and
# ParallelFactorize. It imports the 1-D take helper from the core algorithms
# module, the low-level algorithms module (from the compiled library package, with
# a fallback for old versions) and the file-benchmark base class from the shared
# helper module. At import time it probes for the old standalone rolling
# functions and records the result in the module flag have_rolling_methods. The
# final line re-exports the common asv setup hook.
from functools import wraps
import threading

import numpy as np

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    factorize,
    read_csv,
)
from pandas.core.algorithms import take_nd

try:
    from pandas import (
        rolling_kurt,
        rolling_max,
        rolling_mean,
        rolling_median,
        rolling_min,
        rolling_skew,
        rolling_std,
        rolling_var,
    )

    have_rolling_methods = True
except ImportError:
    have_rolling_methods = False
try:
    from pandas._libs import algos
except ImportError:
    from pandas import algos

from .pandas_vb_common import BaseIO  # isort:skip


def run_parallel(num_threads=2, kwargs_list=None):
    # [FUNCTION DOC]
    # Decorator factory. Given a thread count and an optional list of per-thread
    # keyword-argument mappings, returns a decorator that wraps a function so that
    # calling the wrapper runs the original function once in each of that many
    # threads concurrently and waits for all to finish.
    # Inputs: num_threads, a positive integer (default two) giving how many threads
    # to start; kwargs_list, an optional list with exactly one mapping per thread
    # whose entries are merged over the call keyword arguments for that thread
    # (default none, meaning every thread gets the call keyword arguments).
    # Output: a decorator; the wrapped function returns nothing (return values of
    # the threads are discarded). Exceptions: assertion errors at decoration time
    # if num_threads is not positive or if the list length differs from the
    # thread count. Exceptions raised inside threads are not propagated to the
    # caller. Adapted from scikit-image (see the docstring below).
    """
    Decorator to run the same function multiple times in parallel.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.
    kwargs_list : list of dicts, optional
        The list of kwargs to update original
        function kwargs on different threads.

    Notes
    -----
    This decorator does not pass the return value of the decorated function.

    Original from scikit-image:

    https://github.com/scikit-image/scikit-image/pull/1519

    """
    assert num_threads > 0
    has_kwargs_list = kwargs_list is not None
    if has_kwargs_list:
        assert len(kwargs_list) == num_threads

    def wrapper(func):
        # [FUNCTION DOC]
        # HELPER: the decorator returned by the decorator factory run_parallel.
        # Takes the function to parallelize and returns the wrapper that preserves
        # the original function metadata.
        @wraps(func)
        def inner(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: the wrapper produced by the decorator in run_parallel.
            # Accepts any positional and keyword arguments. Builds one thread per
            # requested thread count, all targeting the original function with the
            # same positional arguments; the keyword arguments are either the
            # call's keyword arguments (chosen through an inline selector that
            # ignores the thread index) or, when a per-thread list was supplied,
            # the call's keyword arguments overlaid with that thread's mapping.
            # Starts all threads, then waits for all of them. Returns nothing.
            if has_kwargs_list:
                update_kwargs = lambda i: dict(kwargs, **kwargs_list[i])
            else:
                update_kwargs = lambda i: kwargs
            threads = []
            for i in range(num_threads):
                updated_kwargs = update_kwargs(i)
                thread = threading.Thread(target=func, args=args, kwargs=updated_kwargs)
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        return inner

    return wrapper


class ParallelGroupbyMethods:
    # [CLASS DOC]
    # Benchmark comparing a group-wise reduction run in several threads at once
    # versus the same number of sequential runs. Parametrized by thread count
    # (2, 4, 8) and reduction method name. State: parallel (threaded callable) and
    # loop (single-thread callable).
    params = ([2, 4, 8], ["count", "last", "max", "mean", "min", "prod", "sum", "var"])
    param_names = ["threads", "method"]

    def setup(self, threads, method):
        # [FUNCTION DOC]
        # Builds a frame of one million rows with an integer grouping key (one
        # thousand groups) and a random normal data column. Defines two callables
        # that group by the key and apply the named reduction to the data column:
        # parallel runs it in the given number of threads, loop runs it once. Stores
        # them as parallel and loop.
        N = 10**6
        ngroups = 10**3
        df = DataFrame(
            {"key": np.random.randint(0, ngroups, size=N), "data": np.random.randn(N)}
        )

        @run_parallel(num_threads=threads)
        def parallel():
            # [FUNCTION DOC]
            # HELPER: threaded body for the setup method. Groups the frame by the
            # key and applies the named reduction to the data column; the result is
            # discarded.
            getattr(df.groupby("key")["data"], method)()

        self.parallel = parallel

        def loop():
            # [FUNCTION DOC]
            # HELPER: single-thread body for the setup method. Performs the same
            # grouped reduction as the threaded body, once; result discarded.
            getattr(df.groupby("key")["data"], method)()

        self.loop = loop

    def time_parallel(self, threads, method):
        # [FUNCTION DOC]
        # Times running the reduction concurrently in the requested threads.
        self.parallel()

    def time_loop(self, threads, method):
        # [FUNCTION DOC]
        # Times running the reduction sequentially once per requested thread
        # count, as the single-threaded baseline.
        for i in range(threads):
            self.loop()


class ParallelGroups:
    # [CLASS DOC]
    # Benchmark of computing the group-label mapping of a series grouped by
    # itself, in multiple threads. Parametrized by thread count. State:
    # get_groups.
    params = [2, 4, 8]
    param_names = ["threads"]

    def setup(self, threads):
        # [FUNCTION DOC]
        # Builds a series of about 4.2 million random integers below one thousand
        # and defines a threaded callable that groups the series by itself and
        # reads the groups mapping. Stores it as get_groups.
        size = 2**22
        ngroups = 10**3
        data = Series(np.random.randint(0, ngroups, size=size))

        @run_parallel(num_threads=threads)
        def get_groups():
            # [FUNCTION DOC]
            # HELPER: threaded body for the setup method. Groups the series by
            # itself and reads the groups mapping; result discarded.
            data.groupby(data).groups

        self.get_groups = get_groups

    def time_get_groups(self, threads):
        # [FUNCTION DOC]
        # Times running the threaded group-mapping computation.
        self.get_groups()


class ParallelTake1D:
    # [CLASS DOC]
    # Benchmark of the one-dimensional take routine run concurrently in two
    # threads. Parametrized by dtype name. State: parallel_take1d.
    params = ["int64", "float64"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a frame with one column of a million values of the dtype, an
        # indexer covering all positions except the first and last hundred, and a
        # two-thread callable that takes those positions from the column's array.
        # Stores the callable as parallel_take1d.
        N = 10**6
        df = DataFrame({"col": np.arange(N, dtype=dtype)})
        indexer = np.arange(100, len(df) - 100)

        @run_parallel(num_threads=2)
        def parallel_take1d():
            # [FUNCTION DOC]
            # HELPER: threaded body for the setup method. Performs the
            # one-dimensional take of the indexer positions from the column
            # values; result discarded.
            take_nd(df["col"].values, indexer)

        self.parallel_take1d = parallel_take1d

    def time_take1d(self, dtype):
        # [FUNCTION DOC]
        # Times running the two-thread take operation.
        self.parallel_take1d()


class ParallelKth:
    # [CLASS DOC]
    # Benchmark of the low-level k-th smallest selection routine run concurrently
    # in two threads, each on its own array. The code depends only on the compiled
    # library and could live in the libs benchmark module. One call per
    # measurement and five repeats. State: parallel_kth_smallest.
    # This depends exclusively on code in _libs/, could go in libs.py

    number = 1
    repeat = 5

    def setup(self):
        # [FUNCTION DOC]
        # Creates two independent arrays of ten million random normal numbers and
        # a two-thread callable that selects the 500000th smallest value; each
        # thread receives one array through its keyword arguments. Stores the
        # callable as parallel_kth_smallest.
        N = 10**7
        k = 5 * 10**5
        kwargs_list = [{"arr": np.random.randn(N)}, {"arr": np.random.randn(N)}]

        @run_parallel(num_threads=2, kwargs_list=kwargs_list)
        def parallel_kth_smallest(arr):
            # [FUNCTION DOC]
            # HELPER: threaded body for the setup method. Takes the array for this
            # thread and runs the low-level k-th smallest selection with the
            # enclosing rank k; result discarded.
            algos.kth_smallest(arr, k)

        self.parallel_kth_smallest = parallel_kth_smallest

    def time_kth_smallest(self):
        # [FUNCTION DOC]
        # Times running the two-thread selection.
        self.parallel_kth_smallest()


class ParallelDatetimeFields:
    # [CLASS DOC]
    # Benchmark of datetime field extraction and conversion between datetime and
    # period representations, each run in two threads at once. No parameters.
    # State: dti and period.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a minute-frequency datetime index of one million entries from
        # 1900-01-01 as dti, and its daily period conversion as period.
        N = 10**6
        self.dti = date_range("1900-01-01", periods=N, freq="min")
        self.period = self.dti.to_period("D")

    def time_datetime_field_year(self):
        # [FUNCTION DOC]
        # Times reading the year field of the datetime index in two threads.
        @run_parallel(num_threads=2)
        def run(dti):
            # [FUNCTION DOC]
            # HELPER: threaded body for this timing method. Reads the year
            # field of the given datetime index; result discarded.
            dti.year

        run(self.dti)

    def time_datetime_field_day(self):
        # [FUNCTION DOC]
        # Times reading the day field of the datetime index in two threads.
        @run_parallel(num_threads=2)
        def run(dti):
            # [FUNCTION DOC]
            # HELPER: threaded body for this timing method. Reads the day field
            # of the given datetime index; result discarded.
            dti.day

        run(self.dti)

    def time_datetime_field_daysinmonth(self):
        # [FUNCTION DOC]
        # Times reading the days-in-month field of the datetime index in two
        # threads.
        @run_parallel(num_threads=2)
        def run(dti):
            # [FUNCTION DOC]
            # HELPER: threaded body for this timing method. Reads the
            # days-in-month field of the given datetime index; result discarded.
            dti.days_in_month

        run(self.dti)

    def time_datetime_field_normalize(self):
        # [FUNCTION DOC]
        # Times normalizing the datetime index to midnight in two threads.
        @run_parallel(num_threads=2)
        def run(dti):
            # [FUNCTION DOC]
            # HELPER: threaded body for this timing method. Normalizes the given
            # datetime index to midnight; result discarded.
            dti.normalize()

        run(self.dti)

    def time_datetime_to_period(self):
        # [FUNCTION DOC]
        # Times converting the datetime index to per-second periods in two threads.
        @run_parallel(num_threads=2)
        def run(dti):
            # [FUNCTION DOC]
            # HELPER: threaded body for this timing method. Converts the given
            # datetime index to second-frequency periods; result discarded.
            dti.to_period("s")

        run(self.dti)

    def time_period_to_datetime(self):
        # [FUNCTION DOC]
        # Times converting the period index back to timestamps in two threads.
        @run_parallel(num_threads=2)
        def run(period):
            # [FUNCTION DOC]
            # HELPER: threaded body for this timing method. Converts the given
            # period index to timestamps; result discarded.
            period.to_timestamp()

        run(self.period)


class ParallelRolling:
    # [CLASS DOC]
    # Benchmark of rolling-window statistics run concurrently in two threads.
    # Parametrized by statistic name (median, mean, min, max, var, skew, kurt,
    # std). Works with both the modern rolling API and the legacy standalone
    # functions. State: parallel_rolling.
    params = ["median", "mean", "min", "max", "var", "skew", "kurt", "std"]
    param_names = ["method"]

    def setup(self, method):
        # [FUNCTION DOC]
        # Creates 100000 uniform random numbers and a window of 100. If the
        # frame class has a rolling method, builds a rolling-window object over
        # the data and defines a two-thread callable that applies the named
        # statistic to it. Otherwise, if the legacy standalone functions were
        # imported, defines a callable that applies the matching legacy function
        # to the raw array and window. Raises a not-implemented error if neither
        # is available. Stores the callable as parallel_rolling.
        win = 100
        arr = np.random.rand(100000)
        if hasattr(DataFrame, "rolling"):
            df = DataFrame(arr).rolling(win)

            @run_parallel(num_threads=2)
            def parallel_rolling():
                # [FUNCTION DOC]
                # HELPER: threaded body for the setup method. Applies the named
                # statistic to the rolling-window object; result discarded.
                getattr(df, method)()

            self.parallel_rolling = parallel_rolling
        elif have_rolling_methods:
            rolling = {
                "median": rolling_median,
                "mean": rolling_mean,
                "min": rolling_min,
                "max": rolling_max,
                "var": rolling_var,
                "skew": rolling_skew,
                "kurt": rolling_kurt,
                "std": rolling_std,
            }

            @run_parallel(num_threads=2)
            def parallel_rolling():
                # [FUNCTION DOC]
                # HELPER: threaded body for the setup method (legacy path). Calls
                # the legacy function matching the statistic name with the raw
                # array and window; result discarded.
                rolling[method](arr, win)

            self.parallel_rolling = parallel_rolling
        else:
            raise NotImplementedError

    def time_rolling(self, method):
        # [FUNCTION DOC]
        # Times running the two-thread rolling computation.
        self.parallel_rolling()


class ParallelReadCSV(BaseIO):
    # [CLASS DOC]
    # Benchmark of reading a comma-separated file concurrently in two threads.
    # Inherits from the shared file-benchmark base class. One call per
    # measurement and five repeats. Parametrized by data kind (float, object,
    # datetime). State: fname and parallel_read_csv.
    number = 1
    repeat = 5
    params = ["float", "object", "datetime"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a 10000-row frame by kind and writes it to a file in the current
        # directory named with the kind (side effect: file I/O, never removed
        # here). Float: 50 random normal columns. Datetime: the same with a daily
        # datetime index. Object: the constant string foo with five column labels.
        # Raises a not-implemented error for other kinds. Defines a two-thread
        # callable that reads the file, stored as parallel_read_csv.
        # KNOWN ISSUE: the object-kind column labels use a percent-format string
        # that is never formatted, so all five labels are the same literal text
        # with a placeholder, producing duplicate column names.
        rows = 10000
        cols = 50
        if dtype == "float":
            df = DataFrame(np.random.randn(rows, cols))
        elif dtype == "datetime":
            df = DataFrame(
                np.random.randn(rows, cols), index=date_range("1/1/2000", periods=rows)
            )
        elif dtype == "object":
            df = DataFrame(
                "foo", index=range(rows), columns=["object%03d" for _ in range(5)]
            )
        else:
            raise NotImplementedError

        self.fname = f"__test_{dtype}__.csv"
        df.to_csv(self.fname)

        @run_parallel(num_threads=2)
        def parallel_read_csv():
            # [FUNCTION DOC]
            # HELPER: threaded body for the setup method. Reads the file written
            # by setup into a frame; result discarded.
            read_csv(self.fname)

        self.parallel_read_csv = parallel_read_csv

    def time_read_csv(self, dtype):
        # [FUNCTION DOC]
        # Times running the two-thread file read.
        self.parallel_read_csv()


class ParallelFactorize:
    # [CLASS DOC]
    # Benchmark of factorizing a string index concurrently versus sequentially.
    # One call per measurement and five repeats. Parametrized by thread count
    # (2, 4, 8). State: parallel and loop.
    number = 1
    repeat = 5
    params = [2, 4, 8]
    param_names = ["threads"]

    def setup(self, threads):
        # [FUNCTION DOC]
        # Builds an object index of 100000 strings and defines a threaded
        # callable (parallel) and a single-thread callable (loop) that each
        # factorize it. Stores them as parallel and loop.
        strings = Index([f"i-{i}" for i in range(100000)], dtype=object)

        @run_parallel(num_threads=threads)
        def parallel():
            # [FUNCTION DOC]
            # HELPER: threaded body for the setup method. Factorizes the string
            # index; result discarded.
            factorize(strings)

        self.parallel = parallel

        def loop():
            # [FUNCTION DOC]
            # HELPER: single-thread body for the setup method. Factorizes the
            # string index once; result discarded.
            factorize(strings)

        self.loop = loop

    def time_parallel(self, threads):
        # [FUNCTION DOC]
        # Times running the factorization concurrently in the requested threads.
        self.parallel()

    def time_loop(self, threads):
        # [FUNCTION DOC]
        # Times running the factorization sequentially once per thread count as
        # the baseline.
        for i in range(threads):
            self.loop()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
