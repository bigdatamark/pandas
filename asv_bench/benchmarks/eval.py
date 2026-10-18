# [MODULE DOC]
# Benchmarks for the expression evaluator and the frame query method. Public
# contents are the benchmark classes Eval and Query. Imports the expression
# module from its current location, falling back to the old location for older
# library versions; it is used to change the numexpr thread count. The final line
# re-exports the common asv setup hook.
import numpy as np

import pandas as pd

try:
    import pandas.core.computation.expressions as expr
except ImportError:
    import pandas.computation.expressions as expr


class Eval:
    # [CLASS DOC]
    # Benchmark of evaluating arithmetic, logical and chained comparison
    # expressions on four large frames with the numexpr or python engine and with
    # either one thread or the default (all). Parametrized by engine and threads.
    # State: df, df2, df3 and df4. Setup may change a global thread setting and
    # teardown resets it.
    params = [["numexpr", "python"], [1, "all"]]
    param_names = ["engine", "threads"]

    def setup(self, engine, threads):
        # [FUNCTION DOC]
        # Creates four independent 20000 by 100 random normal frames. Side effect:
        # if threads is one, sets the numexpr thread count to one globally. The
        # engine parameter is unused here.
        self.df = pd.DataFrame(np.random.randn(20000, 100))
        self.df2 = pd.DataFrame(np.random.randn(20000, 100))
        self.df3 = pd.DataFrame(np.random.randn(20000, 100))
        self.df4 = pd.DataFrame(np.random.randn(20000, 100))

        if threads == 1:
            expr.set_numexpr_threads(1)

    def time_add(self, engine, threads):
        # [FUNCTION DOC]
        # Times evaluating the sum of the four frames with the given engine. The
        # expression refers to the instance attributes by name.
        pd.eval("self.df + self.df2 + self.df3 + self.df4", engine=engine)

    def time_and(self, engine, threads):
        # [FUNCTION DOC]
        # Times evaluating the logical and of four greater-than-zero comparisons
        # of the frames with the given engine.
        pd.eval(
            "(self.df > 0) & (self.df2 > 0) & (self.df3 > 0) & (self.df4 > 0)",
            engine=engine,
        )

    def time_chained_cmp(self, engine, threads):
        # [FUNCTION DOC]
        # Times evaluating a chained less-than comparison across the four frames
        # with the given engine.
        pd.eval("self.df < self.df2 < self.df3 < self.df4", engine=engine)

    def time_mult(self, engine, threads):
        # [FUNCTION DOC]
        # Times evaluating the product of the four frames with the given engine.
        pd.eval("self.df * self.df2 * self.df3 * self.df4", engine=engine)

    def teardown(self, engine, threads):
        # [FUNCTION DOC]
        # asv teardown hook. Resets the numexpr thread count to its default
        # setting (global side effect). Parameters are unused.
        expr.set_numexpr_threads()


class Query:
    # [CLASS DOC]
    # Benchmark of the frame query method using a timestamp and a numeric range,
    # on a frame of one million rows indexed by per-minute timestamps. No
    # parameters. State: ts, df, min_val and max_val.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a minute-frequency datetime index of one million entries from
        # 2001-01-01, takes the element just before the midpoint as ts, builds a
        # frame df with a random normal column a and a datetime column dates
        # indexed by the datetime index, and takes the minimum and maximum of an
        # independent set of random normal numbers as min_val and max_val.
        # KNOWN ISSUE: min_val and max_val come from a separate random sample, not
        # from column a of the frame, so the selection range is unrelated to the
        # frame data (it still selects most rows with high probability).
        N = 10**6
        halfway = (N // 2) - 1
        index = pd.date_range("20010101", periods=N, freq="min")
        s = pd.Series(index)
        self.ts = s.iloc[halfway]
        self.df = pd.DataFrame({"a": np.random.randn(N), "dates": index}, index=index)
        data = np.random.randn(N)
        self.min_val = data.min()
        self.max_val = data.max()

    def time_query_datetime_index(self):
        # [FUNCTION DOC]
        # Times a query selecting rows whose index is before the timestamp ts.
        self.df.query("index < @self.ts")

    def time_query_datetime_column(self):
        # [FUNCTION DOC]
        # Times a query selecting rows whose dates column is before ts.
        self.df.query("dates < @self.ts")

    def time_query_with_boolean_selection(self):
        # [FUNCTION DOC]
        # Times a query selecting rows where column a lies between min_val and
        # max_val inclusive.
        self.df.query("(a >= @self.min_val) & (a <= @self.max_val)")


from .pandas_vb_common import setup  # noqa: F401 isort:skip
