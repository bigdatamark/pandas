# [MODULE DOC]
# Benchmarks for statistical reductions and related operations: frame and series
# reductions (plain, mixed dtypes, multi-index), ranking, correlation and
# covariance. Public contents are the module-level list ops (names of nine
# reduction methods: mean, sum, median, std, skew, kurt, prod, sem, var) and the
# benchmark classes FrameOps, FrameMixedDtypesOps, FrameMultiIndexOps, SeriesOps,
# SeriesMultiIndexOps, Rank, Correlation and Covariance. No import-time side
# effects; the final line re-exports the common asv setup hook.
import numpy as np

import pandas as pd

ops = ["mean", "sum", "median", "std", "skew", "kurt", "prod", "sem", "var"]


class FrameOps:
    # [CLASS DOC]
    # Benchmark of reductions over a tall four-column frame, along either axis or
    # over all values. Parametrized by operation name, dtype (float, int, nullable
    # Int64) and axis (0, 1, none). State: df_func (the bound reduction method).
    params = [ops, ["float", "int", "Int64"], [0, 1, None]]
    param_names = ["op", "dtype", "axis"]

    def setup(self, op, dtype, axis):
        # [FUNCTION DOC]
        # Builds 100000 by 4 random normal values (truncated to integers for the
        # nullable integer dtype so the conversion works), makes a frame converted to
        # the dtype, and stores the named reduction method of the frame as df_func.
        # The axis parameter is unused here.
        values = np.random.randn(100000, 4)
        if dtype == "Int64":
            values = values.astype(int)
        df = pd.DataFrame(values).astype(dtype)
        self.df_func = getattr(df, op)

    def time_op(self, op, dtype, axis):
        # [FUNCTION DOC]
        # Times calling the prepared reduction along the axis.
        self.df_func(axis=axis)


class FrameMixedDtypesOps:
    # [CLASS DOC]
    # Benchmark of reductions over a frame with float, integer and datetime
    # columns. Only the combinations for which datetime aggregation is implemented
    # run. Parametrized by operation name and axis. State: df_func.
    params = [ops, [0, 1, None]]
    param_names = ["op", "axis"]

    def setup(self, op, axis):
        # [FUNCTION DOC]
        # Raises a not-implemented error (skipping the case) for operations that are
        # not implemented for datetimes: the sum, skew, kurtosis, product, standard
        # error and variance reductions on any axis, and the mean, median and
        # standard deviation on axis one or over all values. Otherwise builds a frame
        # of a million rows with a random normal column f, a random integer column i
        # and an hourly timestamp column ts, and stores the named reduction method as
        # df_func.
        if op in ("sum", "skew", "kurt", "prod", "sem", "var") or (
            (op, axis)
            in (
                ("mean", 1),
                ("mean", None),
                ("median", 1),
                ("median", None),
                ("std", 1),
                ("std", None),
            )
        ):
            # Skipping cases where datetime aggregations are not implemented
            raise NotImplementedError

        N = 1_000_000
        df = pd.DataFrame(
            {
                "f": np.random.normal(0.0, 1.0, N),
                "i": np.random.randint(0, N, N),
                "ts": pd.date_range(start="1/1/2000", periods=N, freq="h"),
            }
        )

        self.df_func = getattr(df, op)

    def time_op(self, op, axis):
        # [FUNCTION DOC]
        # Times calling the prepared reduction along the axis.
        self.df_func(axis=axis)


class FrameMultiIndexOps:
    # [CLASS DOC]
    # Benchmark of reductions over a frame with a three-level multi-index.
    # Parametrized by operation name. State: df_func.
    params = [ops]
    param_names = ["op"]

    def setup(self, op):
        # [FUNCTION DOC]
        # Builds a three-level multi-index with 100000 entries (levels of 10, 100 and
        # 100 integers with explicit codes), a four-column random normal frame on it,
        # and stores the named reduction method as df_func.
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [
            np.arange(10).repeat(10000),
            np.tile(np.arange(100).repeat(100), 10),
            np.tile(np.tile(np.arange(100), 100), 10),
        ]
        index = pd.MultiIndex(levels=levels, codes=codes)
        df = pd.DataFrame(np.random.randn(len(index), 4), index=index)
        self.df_func = getattr(df, op)

    def time_op(self, op):
        # [FUNCTION DOC]
        # Times calling the prepared reduction with default settings.
        self.df_func()


class SeriesOps:
    # [CLASS DOC]
    # Benchmark of reductions over a series of 100000 random numbers, float or
    # integer. Parametrized by operation name and dtype. State: s_func.
    params = [ops, ["float", "int"]]
    param_names = ["op", "dtype"]

    def setup(self, op, dtype):
        # [FUNCTION DOC]
        # Builds a random normal series of 100000 elements converted to the dtype and
        # stores the named reduction method as s_func.
        s = pd.Series(np.random.randn(100000)).astype(dtype)
        self.s_func = getattr(s, op)

    def time_op(self, op, dtype):
        # [FUNCTION DOC]
        # Times calling the prepared reduction.
        self.s_func()


class SeriesMultiIndexOps:
    # [CLASS DOC]
    # Benchmark of reductions over a series with a three-level multi-index.
    # Parametrized by operation name. State: s_func.
    params = [ops]
    param_names = ["op"]

    def setup(self, op):
        # [FUNCTION DOC]
        # Builds the same three-level multi-index with 100000 entries as the frame
        # variant and a random normal series on it, and stores the named reduction
        # method as s_func.
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [
            np.arange(10).repeat(10000),
            np.tile(np.arange(100).repeat(100), 10),
            np.tile(np.tile(np.arange(100), 100), 10),
        ]
        index = pd.MultiIndex(levels=levels, codes=codes)
        s = pd.Series(np.random.randn(len(index)), index=index)
        self.s_func = getattr(s, op)

    def time_op(self, op):
        # [FUNCTION DOC]
        # Times calling the prepared reduction.
        self.s_func()


class Rank:
    # [CLASS DOC]
    # Benchmark of ranking random data held in a frame or series, with or without
    # percentage output, plus the older way of dividing by the length. Parametrized
    # by constructor name and the pct flag. State: data.
    params = [["DataFrame", "Series"], [True, False]]
    param_names = ["constructor", "pct"]

    def setup(self, constructor, pct):
        # [FUNCTION DOC]
        # Builds 100000 random normal values wrapped in the frame or series class as
        # data. The pct flag is unused here.
        values = np.random.randn(10**5)
        self.data = getattr(pd, constructor)(values)

    def time_rank(self, constructor, pct):
        # [FUNCTION DOC]
        # Times ranking the data, with percentage output according to the flag.
        self.data.rank(pct=pct)

    def time_average_old(self, constructor, pct):
        # [FUNCTION DOC]
        # Times the older way of getting percentage ranks: rank the data and divide
        # by the number of rows.
        self.data.rank(pct=pct) / len(self.data)


class Correlation:
    # [CLASS DOC]
    # Benchmark of correlation of frames, wide frames (with and without missing
    # values), series, and frame-to-frame column/row correlation, for three
    # correlation methods. Parametrized by method (spearman, kendall, pearson).
    # State: df, df2, df_wide, df_wide_nans, s, s2.
    params = [["spearman", "kendall", "pearson"]]
    param_names = ["method"]

    def setup(self, method):
        # [FUNCTION DOC]
        # Builds random normal data: df and df2 (500 by 15), df_wide (500 by 100),
        # df_wide_nans (a copy of df_wide with about ten percent of the values
        # replaced by missing), and two 500-element series s and s2. The method
        # parameter is unused here.
        self.df = pd.DataFrame(np.random.randn(500, 15))
        self.df2 = pd.DataFrame(np.random.randn(500, 15))
        self.df_wide = pd.DataFrame(np.random.randn(500, 100))
        self.df_wide_nans = self.df_wide.where(np.random.random((500, 100)) < 0.9)
        self.s = pd.Series(np.random.randn(500))
        self.s2 = pd.Series(np.random.randn(500))

    def time_corr(self, method):
        # [FUNCTION DOC]
        # Times the pairwise column correlation of the narrow frame.
        self.df.corr(method=method)

    def time_corr_wide(self, method):
        # [FUNCTION DOC]
        # Times the pairwise column correlation of the wide frame.
        self.df_wide.corr(method=method)

    def time_corr_wide_nans(self, method):
        # [FUNCTION DOC]
        # Times the pairwise column correlation of the wide frame with missing values.
        self.df_wide_nans.corr(method=method)

    def peakmem_corr_wide(self, method):
        # [FUNCTION DOC]
        # Peak-memory benchmark of the wide frame correlation.
        self.df_wide.corr(method=method)

    def time_corr_series(self, method):
        # [FUNCTION DOC]
        # Times the correlation between the two series.
        self.s.corr(self.s2, method=method)

    def time_corrwith_cols(self, method):
        # [FUNCTION DOC]
        # Times the correlation of matching columns of the two frames.
        self.df.corrwith(self.df2, method=method)

    def time_corrwith_rows(self, method):
        # [FUNCTION DOC]
        # Times the correlation of matching rows of the two frames.
        self.df.corrwith(self.df2, axis=1, method=method)


class Covariance:
    # [CLASS DOC]
    # Benchmark of the covariance between two long series. No parameters (the
    # parameter lists are empty). State: s and s2.
    params = []
    param_names = []

    def setup(self):
        # [FUNCTION DOC]
        # Builds two independent random normal series of 100000 elements.
        self.s = pd.Series(np.random.randn(100000))
        self.s2 = pd.Series(np.random.randn(100000))

    def time_cov_series(self):
        # [FUNCTION DOC]
        # Times the covariance of the two series.
        self.s.cov(self.s2)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
