# [MODULE DOC]
# Benchmarks for arithmetic and comparison operations on frames, series, indexes,
# categoricals, timestamps and date offsets. Public contents are the benchmark
# classes IntFrameWithScalar, OpWithFillValue, MixedFrameWithSeriesAxis,
# FrameWithFrameWide, Ops, Ops2, Timeseries, IrregularOps, TimedeltaOps,
# CategoricalComparisons, IndexArithmetic, NumericInferOps, DateInferOps,
# OffsetArrayArithmetic, ApplyIndex and BinaryOpsMultiIndex, plus the
# module-level lists non_apply, other_offsets, offsets and the calendar hcal that
# parametrize the offset benchmarks. Imports the expression-evaluation module from
# the new or the old location (for older library versions) and the numeric dtype
# list from the shared helper module. Import-time side effects: builds a US
# federal holiday calendar and many offset objects, and optionally imports the
# holiday module. The final line re-exports the common asv setup hook.
import operator
import warnings

import numpy as np

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    to_timedelta,
)

from .pandas_vb_common import numeric_dtypes

try:
    import pandas.core.computation.expressions as expr
except ImportError:
    import pandas.computation.expressions as expr
try:
    import pandas.tseries.holiday
except ImportError:
    pass


class IntFrameWithScalar:
    # [CLASS DOC]
    # Benchmark of binary arithmetic and comparison operators between a large
    # numeric frame and a scalar. Parametrized by frame dtype, scalar value (of
    # several numeric types) and operator function. State: df.
    params = [
        [np.float64, np.int64],
        [2, 3.0, np.int32(4), np.float64(5)],
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.floordiv,
            operator.pow,
            operator.mod,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
            operator.lt,
            operator.le,
        ],
    ]
    param_names = ["dtype", "scalar", "op"]

    def setup(self, dtype, scalar, op):
        # [FUNCTION DOC]
        # Creates a 20000 by 100 frame of random normal numbers cast to dtype and
        # stores it as df. Parameters scalar and op are unused here.
        arr = np.random.randn(20000, 100)
        self.df = DataFrame(arr.astype(dtype))

    def time_frame_op_with_scalar(self, dtype, scalar, op):
        # [FUNCTION DOC]
        # Times applying the operator function to the frame and the scalar.
        # The dtype parameter is unused. Result is discarded.
        op(self.df, scalar)


class OpWithFillValue:
    # [CLASS DOC]
    # Benchmark of the named add methods with a fill value on a frame and a
    # series that contain no missing values (regression benchmark for GH 31300).
    # No parameters. State: df and ser.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a one-column frame of one million integers and its single column
        # as a series; stores them as df and ser.
        # GH#31300
        arr = np.arange(10**6)
        df = DataFrame({"A": arr})
        ser = df["A"]

        self.df = df
        self.ser = ser

    def time_frame_op_with_fill_value_no_nas(self):
        # [FUNCTION DOC]
        # Times adding the frame to itself with a fill value of four.
        self.df.add(self.df, fill_value=4)

    def time_series_op_with_fill_value_no_nas(self):
        # [FUNCTION DOC]
        # Times adding the series to itself with a fill value of four.
        self.ser.add(self.ser, fill_value=4)


class MixedFrameWithSeriesAxis:
    # [CLASS DOC]
    # Benchmark of binary operations between a mixed-dtype frame and a series,
    # either aligned on the index (axis zero, via the named method) or on the
    # columns (via the operator module). Parametrized by operation name. The axis
    # one timing method overrides its parameter list to drop the six comparison
    # operations (the first six names), which do not align and would raise.
    # State: df, ser (first column) and row (first row).
    params = [
        [
            "eq",
            "ne",
            "lt",
            "le",
            "ge",
            "gt",
            "add",
            "sub",
            "truediv",
            "floordiv",
            "mul",
            "pow",
        ]
    ]
    param_names = ["opname"]

    def setup(self, opname):
        # [FUNCTION DOC]
        # Builds a frame from a thousand-row integer matrix of one million
        # elements, adds a float column named C filled with one, and stores the
        # frame as df, its first column as ser and its first row as row.
        arr = np.arange(10**6).reshape(1000, -1)
        df = DataFrame(arr)
        df["C"] = 1.0
        self.df = df
        self.ser = df[0]
        self.row = df.iloc[0]

    def time_frame_op_with_series_axis0(self, opname):
        # [FUNCTION DOC]
        # Times calling the frame method named by opname with the series aligned
        # on the index.
        getattr(self.df, opname)(self.ser, axis=0)

    def time_frame_op_with_series_axis1(self, opname):
        # [FUNCTION DOC]
        # Times calling the operator-module function of that name with the frame
        # and the series, which aligns the series to the columns. Only valid for
        # the arithmetic names (see the parameter override below).
        getattr(operator, opname)(self.df, self.ser)

    # exclude comparisons from the params for time_frame_op_with_series_axis1
    #  since they do not do alignment so raise
    time_frame_op_with_series_axis1.params = [params[0][6:]]


class FrameWithFrameWide:
    # [CLASS DOC]
    # Many-columns, mixed-dtype benchmark of binary operators between two frames
    # whose internal blocks do or do not line up. Parametrized by operator
    # (add, floor division, greater-than) and shape (rows, columns). See GH 32779
    # for the choice of operators. State: left and right frames.

    params = [
        [
            # GH#32779 has discussion of which operators are included here
            operator.add,
            operator.floordiv,
            operator.gt,
        ],
        [
            # (n_rows, n_columns)
            (1_000_000, 10),
            (100_000, 100),
            (10_000, 1000),
            (1000, 10_000),
        ],
    ]
    param_names = ["op", "shape"]

    def setup(self, op, shape):
        # [FUNCTION DOC]
        # Builds two frames. The left frame has two consolidated blocks of
        # random float64 and float32 data, each half the columns. The right frame
        # is made of three blocks: float64 (a quarter of the columns, at least
        # three), integer-cast data (half) and float64 (a quarter), so its block
        # layout does not match the left. Row count is cut to a tenth for floor
        # division since that operation is much slower. Both frames are
        # consolidated in place. Stores them as left and right.
        # we choose dtypes so as to make the blocks
        #  a) not perfectly match between right and left
        #  b) appreciably bigger than single columns
        n_rows, n_cols = shape

        if op is operator.floordiv:
            # floordiv is much slower than the other operations -> use less data
            n_rows = n_rows // 10

        # construct dataframe with 2 blocks
        arr1 = np.random.randn(n_rows, n_cols // 2).astype("f8")
        arr2 = np.random.randn(n_rows, n_cols // 2).astype("f4")
        df = pd.concat([DataFrame(arr1), DataFrame(arr2)], axis=1, ignore_index=True)
        # should already be the case, but just to be sure
        df._consolidate_inplace()

        # TODO: GH#33198 the setting here shouldn't need two steps
        arr1 = np.random.randn(n_rows, max(n_cols // 4, 3)).astype("f8")
        arr2 = np.random.randn(n_rows, n_cols // 2).astype("i8")
        arr3 = np.random.randn(n_rows, n_cols // 4).astype("f8")
        df2 = pd.concat(
            [DataFrame(arr1), DataFrame(arr2), DataFrame(arr3)],
            axis=1,
            ignore_index=True,
        )
        # should already be the case, but just to be sure
        df2._consolidate_inplace()

        self.left = df
        self.right = df2

    def time_op_different_blocks(self, op, shape):
        # [FUNCTION DOC]
        # Times the operator between left and right, whose blocks and dtypes
        # are not aligned. The shape parameter is unused.
        # blocks (and dtypes) are not aligned
        op(self.left, self.right)

    def time_op_same_blocks(self, op, shape):
        # [FUNCTION DOC]
        # Times the operator between left and itself, so blocks and dtypes are
        # aligned. The shape parameter is unused.
        # blocks (and dtypes) are aligned
        op(self.left, self.left)


class Ops:
    # [CLASS DOC]
    # Benchmark of frame operators with the optional numexpr acceleration engine
    # switched on or off and with a default or single thread count.
    # Parametrized by use_numexpr and threads. State: df and df2. The setup
    # mutates global expression settings and teardown restores them.
    params = [[True, False], ["default", 1]]
    param_names = ["use_numexpr", "threads"]

    def setup(self, use_numexpr, threads):
        # [FUNCTION DOC]
        # Creates two 20000 by 100 random frames (df and df2). Side effects on
        # global state: if threads is not the string default, sets the numexpr
        # thread count to it; if use_numexpr is false, disables numexpr usage.
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))

        if threads != "default":
            expr.set_numexpr_threads(threads)
        if not use_numexpr:
            expr.set_use_numexpr(False)

    def time_frame_add(self, use_numexpr, threads):
        # [FUNCTION DOC]
        # Times adding the two frames. Parameters are unused.
        self.df + self.df2

    def time_frame_mult(self, use_numexpr, threads):
        # [FUNCTION DOC]
        # Times multiplying the two frames. Parameters are unused.
        self.df * self.df2

    def time_frame_multi_and(self, use_numexpr, threads):
        # [FUNCTION DOC]
        # Times selecting from the first frame with a boolean mask that is the
        # logical and of two comparisons against zero. Parameters are unused.
        self.df[(self.df > 0) & (self.df2 > 0)]

    def time_frame_comparison(self, use_numexpr, threads):
        # [FUNCTION DOC]
        # Times an element-wise greater-than comparison of the two frames.
        # Parameters are unused.
        self.df > self.df2

    def teardown(self, use_numexpr, threads):
        # [FUNCTION DOC]
        # asv teardown hook. Restores global state changed by setup: turns numexpr
        # usage back on and resets the numexpr thread count to its default.
        # Parameters are unused.
        expr.set_use_numexpr(True)
        expr.set_numexpr_threads()


class Ops2:
    # [CLASS DOC]
    # Benchmark of division, modulo and dot product on square float and integer
    # frames and a series. No parameters. State: df, df2 (random floats),
    # df_int, df2_int (random 16-bit-range integers) and s (series), all
    # sized one thousand.
    def setup(self):
        # [FUNCTION DOC]
        # Creates 1000 by 1000 frames of random normal floats (df, df2), 1000 by
        # 1000 frames of random integers across the 16-bit signed range (df_int,
        # df2_int) and a 1000-element random series s.
        N = 10**3
        self.df = DataFrame(np.random.randn(N, N))
        self.df2 = DataFrame(np.random.randn(N, N))

        self.df_int = DataFrame(
            np.random.randint(
                np.iinfo(np.int16).min, np.iinfo(np.int16).max, size=(N, N)
            )
        )
        self.df2_int = DataFrame(
            np.random.randint(
                np.iinfo(np.int16).min, np.iinfo(np.int16).max, size=(N, N)
            )
        )

        self.s = Series(np.random.randn(N))

    # Division

    def time_frame_float_div(self):
        # [FUNCTION DOC]
        # Times floor division of the float frames.
        # KNOWN ISSUE: the name says true division but the operation is floor
        # division.
        self.df // self.df2

    def time_frame_float_div_by_zero(self):
        # [FUNCTION DOC]
        # Times true division of the float frame by the scalar zero.
        self.df / 0

    def time_frame_float_floor_by_zero(self):
        # [FUNCTION DOC]
        # Times floor division of the float frame by the scalar zero.
        self.df // 0

    def time_frame_int_div_by_zero(self):
        # [FUNCTION DOC]
        # Times true division of the integer frame by the scalar zero.
        self.df_int / 0

    # Modulo

    def time_frame_int_mod(self):
        # [FUNCTION DOC]
        # Times the element-wise remainder of the two integer frames.
        self.df_int % self.df2_int

    def time_frame_float_mod(self):
        # [FUNCTION DOC]
        # Times the element-wise remainder of the two float frames.
        self.df % self.df2

    # Dot product

    def time_frame_dot(self):
        # [FUNCTION DOC]
        # Times the matrix product of the two float frames.
        self.df.dot(self.df2)

    def time_series_dot(self):
        # [FUNCTION DOC]
        # Times the dot product of the series with itself.
        self.s.dot(self.s)

    def time_frame_series_dot(self):
        # [FUNCTION DOC]
        # Times the product of the float frame with the series.
        self.df.dot(self.s)


class Timeseries:
    # [CLASS DOC]
    # Benchmark of comparisons and differencing on datetime series, naive or in
    # the US Eastern timezone. Parametrized by tz. State: s (per-minute series),
    # ts (a scalar timestamp from its middle), s2 (per-second series) and
    # ts_different_reso (a timestamp with a different resolution).
    params = [None, "US/Eastern"]
    param_names = ["tz"]

    def setup(self, tz):
        # [FUNCTION DOC]
        # Builds a one-million element minute-frequency datetime series s starting
        # 2001-01-01 in timezone tz, takes the element just before its midpoint
        # as ts, builds a second-frequency series s2 and a timestamp for the
        # second day (2001-01-02) as ts_different_reso.
        N = 10**6
        halfway = (N // 2) - 1
        self.s = Series(date_range("20010101", periods=N, freq="min", tz=tz))
        self.ts = self.s[halfway]

        self.s2 = Series(date_range("20010101", periods=N, freq="s", tz=tz))
        self.ts_different_reso = Timestamp("2001-01-02", tz=tz)

    def time_series_timestamp_compare(self, tz):
        # [FUNCTION DOC]
        # Times a less-or-equal comparison of the series against the scalar.
        self.s <= self.ts

    def time_series_timestamp_different_reso_compare(self, tz):
        # [FUNCTION DOC]
        # Times a less-or-equal comparison of the series against the scalar
        # with a different resolution.
        self.s <= self.ts_different_reso

    def time_timestamp_series_compare(self, tz):
        # [FUNCTION DOC]
        # Times the reflected comparison: scalar greater-or-equal series.
        self.ts >= self.s

    def time_timestamp_ops_diff(self, tz):
        # [FUNCTION DOC]
        # Times the successive difference of the per-second series.
        self.s2.diff()

    def time_timestamp_ops_diff_with_shift(self, tz):
        # [FUNCTION DOC]
        # Times subtracting the series from its one-step shifted copy.
        self.s - self.s.shift()


class IrregularOps:
    # [CLASS DOC]
    # Benchmark of adding two series whose datetime indexes contain the same
    # labels in different random orders (forcing alignment). No parameters.
    # State: left and right.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a per-second datetime-indexed random series of 100000 points and
        # stores two independent full shuffled samples of it as left and right.
        N = 10**5
        idx = date_range(start="1/1/2000", periods=N, freq="s")
        s = Series(np.random.randn(N), index=idx)
        self.left = s.sample(frac=1)
        self.right = s.sample(frac=1)

    def time_add(self):
        # [FUNCTION DOC]
        # Times adding left to right with index alignment.
        self.left + self.right


class TimedeltaOps:
    # [CLASS DOC]
    # Benchmark of adding a timestamp to an array of timedeltas. No parameters.
    # State: td (timedelta index) and ts (timestamp).
    def setup(self):
        # [FUNCTION DOC]
        # Builds a timedelta index from the integers below one million and a
        # timestamp for the year 2000; stores them as td and ts.
        self.td = to_timedelta(np.arange(1000000))
        self.ts = Timestamp("2000")

    def time_add_td_ts(self):
        # [FUNCTION DOC]
        # Times adding the timestamp to the timedelta index.
        self.td + self.ts


class CategoricalComparisons:
    # [CLASS DOC]
    # Benchmark of comparison dunder methods on an ordered categorical.
    # Parametrized by the method name. State: cat.
    params = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]
    param_names = ["op"]

    def setup(self, op):
        # [FUNCTION DOC]
        # Builds an ordered categorical from the repeated letters a, a, b, b, c, d
        # repeated one hundred thousand times; stores it as cat. The op
        # parameter is unused.
        N = 10**5
        self.cat = pd.Categorical(list("aabbcd") * N, ordered=True)

    def time_categorical_op(self, op):
        # [FUNCTION DOC]
        # Times calling the named comparison method of the categorical with the
        # category label b.
        getattr(self.cat, op)("b")


class IndexArithmetic:
    # [CLASS DOC]
    # Benchmark of scalar arithmetic on a one-million element numeric Index.
    # Parametrized by dtype kind: float or int. State: index.
    params = ["float", "int"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds an index over the integers below one million as float64 when
        # dtype is float or as int64 when dtype is int; stores it as index. Any
        # other value leaves index unset.
        N = 10**6
        if dtype == "float":
            self.index = Index(np.arange(N), dtype=np.float64)
        elif dtype == "int":
            self.index = Index(np.arange(N), dtype=np.int64)

    def time_add(self, dtype):
        # [FUNCTION DOC]
        # Times adding the scalar two to the index.
        self.index + 2

    def time_subtract(self, dtype):
        # [FUNCTION DOC]
        # Times subtracting the scalar two from the index.
        self.index - 2

    def time_multiply(self, dtype):
        # [FUNCTION DOC]
        # Times multiplying the index by the scalar two.
        self.index * 2

    def time_divide(self, dtype):
        # [FUNCTION DOC]
        # Times dividing the index by the scalar two.
        self.index / 2

    def time_modulo(self, dtype):
        # [FUNCTION DOC]
        # Times the remainder of the index modulo two.
        self.index % 2


class NumericInferOps:
    # [CLASS DOC]
    # Benchmark of column-wise arithmetic between two numeric columns of a frame
    # for each dtype in the shared numeric dtype list (from GH 7332).
    # Parametrized by dtype. State: df with columns A and B.
    # from GH 7332
    params = numeric_dtypes
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a frame with columns A and B, each the integers below 500000 cast
        # to dtype; stores it as df.
        N = 5 * 10**5
        self.df = DataFrame(
            {"A": np.arange(N).astype(dtype), "B": np.arange(N).astype(dtype)}
        )

    def time_add(self, dtype):
        # [FUNCTION DOC]
        # Times adding column B to column A.
        self.df["A"] + self.df["B"]

    def time_subtract(self, dtype):
        # [FUNCTION DOC]
        # Times subtracting column B from column A.
        self.df["A"] - self.df["B"]

    def time_multiply(self, dtype):
        # [FUNCTION DOC]
        # Times multiplying column A by column B.
        self.df["A"] * self.df["B"]

    def time_divide(self, dtype):
        # [FUNCTION DOC]
        # Times dividing column A by column B.
        self.df["A"] / self.df["B"]

    def time_modulo(self, dtype):
        # [FUNCTION DOC]
        # Times the remainder of column A modulo column B.
        self.df["A"] % self.df["B"]


class DateInferOps:
    # [CLASS DOC]
    # Benchmark of arithmetic between datetime and timedelta columns (from
    # GH 7332). Data is created once by the asv cache hook and passed to the
    # timing methods as df. No instance state.
    # from GH 7332
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a frame with a datetime column made from the
        # integers below 500000 interpreted as milliseconds, and a timedelta column
        # equal to the datetime column minus itself (all zero).
        N = 5 * 10**5
        df = DataFrame({"datetime64": np.arange(N).astype("datetime64[ms]")})
        df["timedelta"] = df["datetime64"] - df["datetime64"]
        return df

    def time_subtract_datetimes(self, df):
        # [FUNCTION DOC]
        # Times subtracting the datetime column from itself.
        df["datetime64"] - df["datetime64"]

    def time_timedelta_plus_datetime(self, df):
        # [FUNCTION DOC]
        # Times adding the datetime column to the timedelta column.
        df["timedelta"] + df["datetime64"]

    def time_add_timedeltas(self, df):
        # [FUNCTION DOC]
        # Times adding the timedelta column to itself.
        df["timedelta"] + df["timedelta"]


hcal = pd.tseries.holiday.USFederalHolidayCalendar()
# These offsets currently raise a NotImplementedError with .apply_index()
non_apply = [
    pd.offsets.Day(),
    pd.offsets.BYearEnd(),
    pd.offsets.BYearBegin(),
    pd.offsets.BQuarterEnd(),
    pd.offsets.BQuarterBegin(),
    pd.offsets.BMonthEnd(),
    pd.offsets.BMonthBegin(),
    pd.offsets.CustomBusinessDay(),
    pd.offsets.CustomBusinessDay(calendar=hcal),
    pd.offsets.CustomBusinessMonthBegin(calendar=hcal),
    pd.offsets.CustomBusinessMonthEnd(calendar=hcal),
    pd.offsets.CustomBusinessMonthEnd(calendar=hcal),
]
other_offsets = [
    pd.offsets.YearEnd(),
    pd.offsets.YearBegin(),
    pd.offsets.QuarterEnd(),
    pd.offsets.QuarterBegin(),
    pd.offsets.MonthEnd(),
    pd.offsets.MonthBegin(),
    pd.offsets.DateOffset(months=2, days=2),
    pd.offsets.BusinessDay(),
    pd.offsets.SemiMonthEnd(),
    pd.offsets.SemiMonthBegin(),
]
offsets = non_apply + other_offsets


class OffsetArrayArithmetic:
    # [CLASS DOC]
    # Benchmark of adding a date offset to a datetime series and to a datetime
    # index, for every offset in the combined module-level offsets list.
    # Parametrized by offset. State: rng (index) and ser (series of the index).
    params = offsets
    param_names = ["offset"]

    def setup(self, offset):
        # [FUNCTION DOC]
        # Builds a per-minute datetime index of 10000 entries from 2000-01-01 as
        # rng, and a series made from it as ser. The offset parameter is unused.
        N = 10000
        rng = date_range(start="1/1/2000", periods=N, freq="min")
        self.rng = rng
        self.ser = Series(rng)

    def time_add_series_offset(self, offset):
        # [FUNCTION DOC]
        # Times adding the offset to the series, recording and discarding any
        # warnings emitted (for example performance warnings from non-vectorized
        # offsets).
        with warnings.catch_warnings(record=True):
            self.ser + offset

    def time_add_dti_offset(self, offset):
        # [FUNCTION DOC]
        # Times adding the offset to the datetime index, recording and discarding
        # any warnings emitted.
        with warnings.catch_warnings(record=True):
            self.rng + offset


class ApplyIndex:
    # [CLASS DOC]
    # Benchmark of adding vectorizable date offsets (the other_offsets list) to
    # a datetime index. Parametrized by offset. State: rng.
    params = other_offsets
    param_names = ["offset"]

    def setup(self, offset):
        # [FUNCTION DOC]
        # Builds a per-minute datetime index of 10000 entries from 2000-01-01 and
        # stores it as rng. The offset parameter is unused.
        N = 10000
        rng = date_range(start="1/1/2000", periods=N, freq="min")
        self.rng = rng

    def time_apply_index(self, offset):
        # [FUNCTION DOC]
        # Times adding the offset to the index (no warning capture, unlike the
        # similar class above).
        self.rng + offset


class BinaryOpsMultiIndex:
    # [CLASS DOC]
    # Benchmark of frame arithmetic methods with a level argument, aligning a
    # frame with a two-level multi-index against a frame indexed by the first
    # level. Parametrized by the method name (sub, add, mul, div).
    # State: df and arg_df.
    params = ["sub", "add", "mul", "div"]
    param_names = ["func"]

    def setup(self, func):
        # [FUNCTION DOC]
        # Builds a multi-index from the product of thirty string labels and one
        # day of per-second timestamps (86401 entries). df has two random float
        # columns (col_1, col_2) on that index. arg_df has the same two columns,
        # random integers from one to nine, indexed by the thirty labels. The func
        # parameter is unused here.
        array = date_range("20200101 00:00", "20200102 0:00", freq="s")
        level_0_names = [str(i) for i in range(30)]

        index = pd.MultiIndex.from_product([level_0_names, array])
        column_names = ["col_1", "col_2"]

        self.df = DataFrame(
            np.random.rand(len(index), 2), index=index, columns=column_names
        )

        self.arg_df = DataFrame(
            np.random.randint(1, 10, (len(level_0_names), 2)),
            index=level_0_names,
            columns=column_names,
        )

    def time_binary_op_multiindex(self, func):
        # [FUNCTION DOC]
        # Times calling the frame method named by func with arg_df, aligned on
        # level zero.
        getattr(self.df, func)(self.arg_df, level=0)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
