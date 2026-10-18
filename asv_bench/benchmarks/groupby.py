# [MODULE DOC]
# Benchmarks for group-by operations: apply with various callables, group and
# index mappings, nth selection, aggregations (string names, multi-column, cython
# and numba engines, extension dtypes), size counts, shift, fill, cumulative
# operations, ranking with ties, categorical keys, datelike keys, boolean and
# timedelta sums, transforms (including engine variants), sampling and
# resampling. Public contents are the module-level tables method_blocklist (method
# names to skip per key dtype) and _numba_unsupported_methods (aggregation names
# without a numba kernel), and the benchmark classes ApplyDictReturn, Apply,
# ApplyNonUniqueUnsortedIndex, Groups, GroupManyLabels, Nth, DateAttributes,
# Int64, CountMultiDtype, CountMultiInt, AggFunctions, GroupStrings, MultiColumn,
# Size, Shift, Fillna, GroupByMethods, GroupByCythonAgg, GroupByNumbaAgg,
# GroupByCythonAggEaDtypes, Cumulative, RankWithTies, Float32, String, Categories,
# MultipleCategories, Datelike, SumBools, SumMultiLevel, SumTimeDelta, Transform,
# TransformBools, TransformNaN, TransformEngine, AggEngine, Sample and Resample.
# No import-time side effects other than building the module-level tables; the
# final line re-exports the common asv setup hook.
from functools import partial
from itertools import product
from string import ascii_letters

import numpy as np

from pandas import (
    NA,
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_timedelta,
)

method_blocklist = {
    "object": {
        "diff",
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "cummin",
        "mean",
        "max",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "min",
        "var",
        "describe",
        "std",
        "quantile",
    },
    "datetime": {
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "mean",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "var",
        "describe",
        "std",
    },
}

# These aggregations don't have a kernel implemented for them yet
_numba_unsupported_methods = [
    "all",
    "any",
    "bfill",
    "count",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "ffill",
    "first",
    "head",
    "idxmax",
    "idxmin",
    "last",
    "median",
    "nunique",
    "pct_change",
    "prod",
    "quantile",
    "rank",
    "sem",
    "shift",
    "size",
    "skew",
    "tail",
    "unique",
    "value_counts",
]


class ApplyDictReturn:
    # [CLASS DOC]
    # Benchmark of group-wise apply where the callable returns a dictionary for
    # each group. No parameters. State: labels and data.
    def setup(self):
        # [FUNCTION DOC]
        # Creates labels (the integers below 1000, each repeated ten times) and
        # data (a random normal series of the same length, 10000).
        self.labels = np.arange(1000).repeat(10)
        self.data = Series(np.random.randn(len(self.labels)))

    def time_groupby_apply_dict_return(self):
        # [FUNCTION DOC]
        # Times grouping the series by the labels and applying a function that
        # returns a dictionary holding the first and last value of each group.
        self.data.groupby(self.labels).apply(
            lambda x: {"first": x.values[0], "last": x.values[-1]}
        )


class Apply:
    # [CLASS DOC]
    # Benchmark of group-wise apply with scalar-returning and copying functions
    # on a mixed frame, for small-group and large-group layouts. Parametrized by
    # factor (4 or 5): the row count is ten to the factor. State: df.
    param_names = ["factor"]
    params = [4, 5]

    def setup(self, factor):
        # [FUNCTION DOC]
        # Builds a frame with ten to the factor rows: key (random integers below
        # 2000 when the factor is four, otherwise below 20), key2 (random integers
        # below 3), value1 (random normal) and value2 (the strings foo, bar, baz,
        # qux cycled). This gives small groups of about five rows in the first case
        # and large groups of about five thousand in the second. Stores as df.
        N = 10**factor
        # two cases:
        # - small groups: small data (N**4) + many labels (2000) -> average group
        #   size of 5 (-> larger overhead of slicing method)
        # - larger groups: larger data (N**5) + fewer labels (20) -> average group
        #   size of 5000
        labels = np.random.randint(0, 2000 if factor == 4 else 20, size=N)
        labels2 = np.random.randint(0, 3, size=N)
        df = DataFrame(
            {
                "key": labels,
                "key2": labels2,
                "value1": np.random.randn(N),
                "value2": ["foo", "bar", "baz", "qux"] * (N // 4),
            }
        )
        self.df = df

    def time_scalar_function_multi_col(self, factor):
        # [FUNCTION DOC]
        # Times grouping by two keys and applying a function that returns the
        # constant one for each group.
        self.df.groupby(["key", "key2"]).apply(lambda x: 1)

    def time_scalar_function_single_col(self, factor):
        # [FUNCTION DOC]
        # Times grouping by one key and applying a function that returns the
        # constant one for each group.
        self.df.groupby("key").apply(lambda x: 1)

    @staticmethod
    def df_copy_function(g):
        # [FUNCTION DOC]
        # HELPER: callable used by the copy timing methods of this class.
        # Static method. Input g is one group (a frame). Reads the group name
        # attribute to ensure it is available (issue 15062) and returns a copy of
        # the group.
        # ensure that the group name is available (see GH #15062)
        g.name
        return g.copy()

    def time_copy_function_multi_col(self, factor):
        # [FUNCTION DOC]
        # Times grouping by two keys and applying the copying helper to each group.
        self.df.groupby(["key", "key2"]).apply(self.df_copy_function)

    def time_copy_overhead_single_col(self, factor):
        # [FUNCTION DOC]
        # Times grouping by one key and applying the copying helper to each group.
        self.df.groupby("key").apply(self.df_copy_function)


class ApplyNonUniqueUnsortedIndex:
    # [CLASS DOC]
    # Benchmark (issue 46527) of group-wise apply with an identity function on a
    # frame whose index is both unsorted and non-unique. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds an index of the integers 99 down to 0, each repeated 200 times,
        # named key; the frame df has 10 random normal columns on this index.
        # GH 46527
        # unsorted and non-unique index
        idx = np.arange(100)[::-1]
        idx = Index(np.repeat(idx, 200), name="key")
        self.df = DataFrame(np.random.randn(len(idx), 10), index=idx)

    def time_groupby_apply_non_unique_unsorted_index(self):
        # [FUNCTION DOC]
        # Times grouping by the index name without adding group keys and
        # applying an identity function.
        self.df.groupby("key", group_keys=False).apply(lambda x: x)


class Groups:
    # [CLASS DOC]
    # Benchmark of the groups and indices mappings of a series grouped by itself.
    # Parametrized by key kind (small or large number of distinct integer or
    # string keys). Data for all kinds is built once by the cache hook; setup
    # picks one. State: ser.
    param_names = ["key"]
    params = ["int64_small", "int64_large", "object_small", "object_large"]

    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook run once. Returns a mapping from the four key-kind names
        # to series of one million elements: random integers below 100 or below
        # 10000, and random picks of 100 or 10000 distinct strings.
        size = 10**6
        data = {
            "int64_small": Series(np.random.randint(0, 100, size=size)),
            "int64_large": Series(np.random.randint(0, 10000, size=size)),
            "object_small": Series(
                Index([f"i-{i}" for i in range(100)], dtype=object).take(
                    np.random.randint(0, 100, size=size)
                )
            ),
            "object_large": Series(
                Index([f"i-{i}" for i in range(10000)], dtype=object).take(
                    np.random.randint(0, 10000, size=size)
                )
            ),
        }
        return data

    def setup(self, data, key):
        # [FUNCTION DOC]
        # Selects the series for the requested key kind from the cached mapping
        # and stores it as ser. Note the first parameter is the cached mapping, not
        # a parametrized value.
        self.ser = data[key]

    def time_series_groups(self, data, key):
        # [FUNCTION DOC]
        # Times grouping the series by itself and reading the groups mapping.
        self.ser.groupby(self.ser).groups

    def time_series_indices(self, data, key):
        # [FUNCTION DOC]
        # Times grouping the series by itself and reading the indices mapping.
        self.ser.groupby(self.ser).indices


class GroupManyLabels:
    # [CLASS DOC]
    # Benchmark of summing a frame grouped by a label array, with one column or
    # many columns. Parametrized by column count (1 or 1000). State: labels and
    # df.
    params = [1, 1000]
    param_names = ["ncols"]

    def setup(self, ncols):
        # [FUNCTION DOC]
        # Builds a 1000-row random normal frame with ncols columns and 1000
        # random integer labels below 100.
        N = 1000
        data = np.random.randn(N, ncols)
        self.labels = np.random.randint(0, 100, size=N)
        self.df = DataFrame(data)

    def time_sum(self, ncols):
        # [FUNCTION DOC]
        # Times grouping by the label array and summing each group.
        self.df.groupby(self.labels).sum()


class Nth:
    # [CLASS DOC]
    # Benchmark of selecting the first row of each group (nth with position zero)
    # with different missing-value handling, for frames and series. Parametrized
    # by value dtype: float32, float64, datetime, object. State: df with columns
    # key and values.
    param_names = ["dtype"]
    params = ["float32", "float64", "datetime", "object"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a frame with 100000 rows: key is the integers below that count (all
        # distinct) and values is per-second datetimes (dtype datetime, issue
        # 7555), the constant string foo (object) or the integers cast to the
        # numeric dtype. Then sets row one of the values column to missing.
        N = 10**5
        # with datetimes (GH7555)
        if dtype == "datetime":
            values = date_range("1/1/2011", periods=N, freq="s")
        elif dtype == "object":
            values = ["foo"] * N
        else:
            values = np.arange(N).astype(dtype)

        key = np.arange(N)
        self.df = DataFrame({"key": key, "values": values})
        self.df.iloc[1, 1] = np.nan  # insert missing data

    def time_frame_nth_any(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the first row per group of the frame, dropping rows
        # that have any missing value.
        self.df.groupby("key").nth(0, dropna="any")

    def time_groupby_nth_all(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the first row per group of the frame, dropping rows
        # that are entirely missing.
        self.df.groupby("key").nth(0, dropna="all")

    def time_frame_nth(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the first row per group of the frame without dropping.
        self.df.groupby("key").nth(0)

    def time_series_nth_any(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the first element per group of the values series,
        # dropping missing values (any mode).
        self.df["values"].groupby(self.df["key"]).nth(0, dropna="any")

    def time_series_nth_all(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the first element per group of the values series,
        # dropping missing values (all mode).
        self.df["values"].groupby(self.df["key"]).nth(0, dropna="all")

    def time_series_nth(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the first element per group of the values series
        # without dropping.
        self.df["values"].groupby(self.df["key"]).nth(0)


class DateAttributes:
    # [CLASS DOC]
    # Benchmark of the number of groups when grouping an hourly series by three
    # integer arrays (year, month, day). No parameters. State: year, month, day,
    # ts.
    def setup(self):
        # [FUNCTION DOC]
        # Builds an hourly datetime index from 2000-01-01 to 2005-12-31, stores
        # its year, month and day arrays, and a random normal series on that index
        # as ts.
        rng = date_range("1/1/2000", "12/31/2005", freq="h")
        self.year, self.month, self.day = rng.year, rng.month, rng.day
        self.ts = Series(np.random.randn(len(rng)), index=rng)

    def time_len_groupby_object(self):
        # [FUNCTION DOC]
        # Times grouping the series by the three arrays and taking the number of
        # groups.
        len(self.ts.groupby([self.year, self.month, self.day]))


class Int64:
    # [CLASS DOC]
    # Benchmark of grouping by five integer columns whose combined key space
    # would overflow a naive packed key. No parameters. State: cols and df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 131072 by 5 matrix of random integers between minus 4096 and
        # 4095, appends 655360 randomly chosen existing rows (duplicates), shuffles
        # all rows, and makes a frame with columns a to e plus two random normal
        # columns scaled by ten, jim and joe. Stores the column list as cols and
        # the frame as df.
        arr = np.random.randint(-1 << 12, 1 << 12, (1 << 17, 5))
        i = np.random.choice(len(arr), len(arr) * 5)
        arr = np.vstack((arr, arr[i]))
        i = np.random.permutation(len(arr))
        arr = arr[i]
        self.cols = list("abcde")
        self.df = DataFrame(arr, columns=self.cols)
        self.df["jim"], self.df["joe"] = np.random.randn(2, len(self.df)) * 10

    def time_overflow(self):
        # [FUNCTION DOC]
        # Times grouping by the five integer columns and taking the maximum.
        self.df.groupby(self.cols).max()


class CountMultiDtype:
    # [CLASS DOC]
    # Benchmark of group-wise count over a frame with many dtypes and many missing
    # values. Data is built once by the cache hook and passed to the timing method
    # as df. No instance state.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a frame of 10000 rows with integer keys key1
        # (below 500) and key2 (below 100), datetimes (now plus random offsets,
        # about half set to not-a-time), value2 (random normal, about half missing),
        # value3 (random normal), ints (below 1000), obj (random a/b strings with
        # missing entries) and offsets (random timedeltas, about half not-a-time).
        # Note that the offsets are blanked after the datetimes are computed from
        # them, and that the object mask uses a normal draw, not a uniform one.
        n = 10000
        offsets = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates = np.datetime64("now") + offsets
        dates[np.random.rand(n) > 0.5] = np.datetime64("nat")
        offsets[np.random.rand(n) > 0.5] = np.timedelta64("NaT", "ns")
        value2 = np.random.randn(n)
        value2[np.random.rand(n) > 0.5] = np.nan
        obj = np.random.choice(list("ab"), size=n).astype(object)
        obj[np.random.randn(n) > 0.5] = np.nan
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "dates": dates,
                "value2": value2,
                "value3": np.random.randn(n),
                "ints": np.random.randint(0, 1000, size=n),
                "obj": obj,
                "offsets": offsets,
            }
        )
        return df

    def time_multi_count(self, df):
        # [FUNCTION DOC]
        # Times grouping by key1 and key2 and counting non-missing values per
        # column.
        df.groupby(["key1", "key2"]).count()


class CountMultiInt:
    # [CLASS DOC]
    # Benchmark of group-wise count and distinct count on an all-integer frame.
    # Data is built once by the cache hook and passed to the timing methods as df.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a frame of 10000 rows with integer columns key1
        # (below 500), key2 (below 100), ints and ints2 (each below 1000).
        n = 10000
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "ints": np.random.randint(0, 1000, size=n),
                "ints2": np.random.randint(0, 1000, size=n),
            }
        )
        return df

    def time_multi_int_count(self, df):
        # [FUNCTION DOC]
        # Times grouping by key1 and key2 and counting values.
        df.groupby(["key1", "key2"]).count()

    def time_multi_int_nunique(self, df):
        # [FUNCTION DOC]
        # Times grouping by key1 and key2 and counting distinct values.
        df.groupby(["key1", "key2"]).nunique()


class AggFunctions:
    # [CLASS DOC]
    # Benchmark of aggregating with string function names, per column or over all
    # columns, for two string keys. Data is built once by the cache hook and
    # passed to the timing methods as df.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a frame of 100000 rows with object keys key1
        # (A, B, C picked at random) and key2 (one, two picked at random) and
        # three random normal columns value1, value2, value3.
        N = 10**5
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        df = DataFrame(
            {
                "key1": fac1.take(np.random.randint(0, 3, size=N)),
                "key2": fac2.take(np.random.randint(0, 2, size=N)),
                "value1": np.random.randn(N),
                "value2": np.random.randn(N),
                "value3": np.random.randn(N),
            }
        )
        return df

    def time_different_str_functions(self, df):
        # [FUNCTION DOC]
        # Times grouping by both keys and aggregating value1 by mean, value2 by
        # variance and value3 by sum using a mapping of column to function name.
        df.groupby(["key1", "key2"]).agg(
            {"value1": "mean", "value2": "var", "value3": "sum"}
        )

    def time_different_str_functions_multicol(self, df):
        # [FUNCTION DOC]
        # Times grouping by both keys and applying a list of three function names
        # (sum, min, max) to every column.
        df.groupby(["key1", "key2"]).agg(["sum", "min", "max"])

    def time_different_str_functions_singlecol(self, df):
        # [FUNCTION DOC]
        # Times grouping by key1 only and aggregating with the per-column mapping
        # of mean, variance and sum.
        df.groupby("key1").agg({"value1": "mean", "value2": "var", "value3": "sum"})


class GroupStrings:
    # [CLASS DOC]
    # Benchmark of grouping by four string columns with many distinct
    # combinations. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Takes all four-letter combinations of ASCII letters and samples, without
        # replacement, a 40000 by 4 array of such four-letter strings. Repeats each
        # row five times (200000 rows) and makes a frame with columns a to d. Adds
        # a rounded random column joe, then shuffles the rows and resets the
        # index. Stores as df.
        n = 2 * 10**5
        alpha = list(map("".join, product(ascii_letters, repeat=4)))
        data = np.random.choice(alpha, (n // 5, 4), replace=False)
        data = np.repeat(data, 5, axis=0)
        self.df = DataFrame(data, columns=list("abcd"))
        self.df["joe"] = (np.random.randn(len(self.df)) * 10).round(3)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def time_multi_columns(self):
        # [FUNCTION DOC]
        # Times grouping by the four string columns and taking the maximum.
        self.df.groupby(list("abcd")).max()


class MultiColumn:
    # [CLASS DOC]
    # Benchmark of two-key grouping with a Python lambda aggregation versus the
    # compiled sum, for the whole frame and for a single selected column. Data is
    # built once by the cache hook and passed to the timing methods as df.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a frame of 100000 rows with two independently
        # shuffled object-dtype key columns (each the integers below 100 tiled 1000
        # times) and two random normal data columns.
        N = 10**5
        key1 = np.tile(np.arange(100, dtype=object), 1000)
        key2 = key1.copy()
        np.random.shuffle(key1)
        np.random.shuffle(key2)
        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "data1": np.random.randn(N),
                "data2": np.random.randn(N),
            }
        )
        return df

    def time_lambda_sum(self, df):
        # [FUNCTION DOC]
        # Times grouping by both keys and aggregating with a function that sums
        # the underlying values of each column group.
        df.groupby(["key1", "key2"]).agg(lambda x: x.values.sum())

    def time_cython_sum(self, df):
        # [FUNCTION DOC]
        # Times grouping by both keys and using the built-in sum.
        df.groupby(["key1", "key2"]).sum()

    def time_col_select_lambda_sum(self, df):
        # [FUNCTION DOC]
        # Times selecting column data1 per group and aggregating with the
        # values-summing function.
        df.groupby(["key1", "key2"])["data1"].agg(lambda x: x.values.sum())

    def time_col_select_str_sum(self, df):
        # [FUNCTION DOC]
        # Times selecting column data1 per group and aggregating with the sum
        # given by name.
        df.groupby(["key1", "key2"])["data1"].agg("sum")


class Size:
    # [CLASS DOC]
    # Benchmark of computing group sizes for two integer keys and for a
    # categorical grouper. No parameters. State: df, draws and cats.
    def setup(self):
        # [FUNCTION DOC]
        # With 100000 rows: builds df with integer keys key1 (below 500) and key2
        # (below 100), three random normal value columns and a dates column of the
        # current time plus random nanosecond offsets; draws, a random normal
        # series; and cats, a categorical series of the four labels foo, bar, baz,
        # qux repeated.
        n = 10**5
        offsets = np.random.randint(n, size=n).astype("timedelta64[ns]")
        dates = np.datetime64("now") + offsets
        self.df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "value1": np.random.randn(n),
                "value2": np.random.randn(n),
                "value3": np.random.randn(n),
                "dates": dates,
            }
        )
        self.draws = Series(np.random.randn(n))
        labels = Series(["foo", "bar", "baz", "qux"] * (n // 4))
        self.cats = labels.astype("category")

    def time_multi_size(self):
        # [FUNCTION DOC]
        # Times the group sizes when grouping by both integer keys.
        self.df.groupby(["key1", "key2"]).size()

    def time_category_size(self):
        # [FUNCTION DOC]
        # Times the group sizes of the draws series grouped by the categorical,
        # limited to observed categories.
        self.draws.groupby(self.cats, observed=True).size()


class Shift:
    # [CLASS DOC]
    # Benchmark of group-wise shift on a tiny frame. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds an 18-row frame with a group column g alternating a and b and a
        # value column v with the integers 0 to 17.
        N = 18
        self.df = DataFrame({"g": ["a", "b"] * 9, "v": list(range(N))})

    def time_defaults(self):
        # [FUNCTION DOC]
        # Times the group-wise shift with default settings.
        self.df.groupby("g").shift()

    def time_fill_value(self):
        # [FUNCTION DOC]
        # Times the group-wise shift filling vacated slots with 99.
        self.df.groupby("g").shift(fill_value=99)


class Fillna:
    # [CLASS DOC]
    # Benchmark of group-wise forward and backward fill. No parameters. State:
    # df, indexed by the group column.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of 200 rows: group is 1 for the first 100 rows and 2 for
        # the rest (set as the index) and value alternates missing and 1.0.
        N = 100
        self.df = DataFrame(
            {"group": [1] * N + [2] * N, "value": [np.nan, 1.0] * N}
        ).set_index("group")

    def time_df_ffill(self):
        # [FUNCTION DOC]
        # Times forward fill within groups of the frame, grouped by index name.
        self.df.groupby("group").ffill()

    def time_df_bfill(self):
        # [FUNCTION DOC]
        # Times backward fill within groups of the frame.
        self.df.groupby("group").bfill()

    def time_srs_ffill(self):
        # [FUNCTION DOC]
        # Times forward fill within groups of the value column as a series.
        self.df.groupby("group")["value"].ffill()

    def time_srs_bfill(self):
        # [FUNCTION DOC]
        # Times backward fill within groups of the value column as a series.
        self.df.groupby("group")["value"].bfill()


class GroupByMethods:
    # [CLASS DOC]
    # Broad benchmark of group-by methods across key dtypes, methods, application
    # style (direct or transform), column counts and engines (cython, numba).
    # KNOWN ISSUE: the parameter name list has four entries but there are five
    # parameter lists (the engine list is unnamed), and the setup method raises a
    # not-implemented error for multi-column, transformation and datetime
    # combinations, so those parameter values are always skipped and the
    # transformation and datetime code paths in setup are unreachable.
    # State: as_group_method and as_field_method (callables).
    param_names = ["dtype", "method", "application", "ncols"]
    params = [
        ["int", "int16", "float", "object", "datetime", "uint"],
        [
            "all",
            "any",
            "bfill",
            "count",
            "cumcount",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "describe",
            "diff",
            "ffill",
            "first",
            "head",
            "last",
            "max",
            "min",
            "median",
            "mean",
            "nunique",
            "pct_change",
            "prod",
            "quantile",
            "rank",
            "sem",
            "shift",
            "size",
            "skew",
            "std",
            "sum",
            "tail",
            "unique",
            "value_counts",
            "var",
        ],
        ["direct", "transformation"],
        [1, 5],
        ["cython", "numba"],
    ]

    def setup(self, dtype, method, application, ncols, engine):
        # [FUNCTION DOC]
        # Prepares one combination. Raises a not-implemented error (asv skips the
        # benchmark) when: the method is in the per-dtype blocklist; there are
        # several columns for the methods unique and value_counts; the application
        # is transformation for describe, head, tail, unique, value_counts or
        # size; or (the effective filter) the engine is numba with a method that
        # lacks a numba kernel, or ncols is above one, or the application is
        # transformation, or the dtype is datetime.
        # KNOWN ISSUE: the last filter subsumes the multi-column check and the
        # transformation check above it (they are redundant) and makes the
        # multi-column, transformation and datetime handling in the construction
        # code below unreachable: only direct, single-column, non-datetime runs
        # ever get past this point.
        # Otherwise, picks the number of groups (20 for describe, 100 for skew, else
        # 1000) and twice as many rows. Value columns hold group ids sampled
        # randomly with repeats. The key column depends on the dtype: random
        # integers (plain, 16-bit or unsigned) below the row count; a concatenation
        # of small and large random floats; the constant string foo; or per-second
        # datetimes. Builds a frame of value columns plus the key; selects a single
        # column name when there is only one column; passes the engine keyword only
        # for numba. Stores as_group_method (the named method of the value columns
        # grouped by the key) and as_field_method (the named method of the key
        # column grouped by the value column) as zero-argument callables (partial
        # objects, or transform lambdas in the unreachable transformation path).
        if method in method_blocklist.get(dtype, {}):
            raise NotImplementedError  # skip benchmark

        if ncols != 1 and method in ["value_counts", "unique"]:
            # DataFrameGroupBy doesn't have these methods
            raise NotImplementedError

        if application == "transformation" and method in [
            "describe",
            "head",
            "tail",
            "unique",
            "value_counts",
            "size",
        ]:
            # DataFrameGroupBy doesn't have these methods
            raise NotImplementedError

        # Numba currently doesn't support
        # multiple transform functions or strs for transform,
        # grouping on multiple columns
        # and we lack kernels for a bunch of methods
        if (
            (engine == "numba" and method in _numba_unsupported_methods)
            or ncols > 1
            or application == "transformation"
            or dtype == "datetime"
        ):
            raise NotImplementedError

        if method == "describe":
            ngroups = 20
        elif method == "skew":
            ngroups = 100
        else:
            ngroups = 1000
        size = ngroups * 2
        rng = np.arange(ngroups).reshape(-1, 1)
        rng = np.broadcast_to(rng, (len(rng), ncols))
        taker = np.random.randint(0, ngroups, size=size)
        values = rng.take(taker, axis=0)
        if dtype == "int":
            key = np.random.randint(0, size, size=size)
        elif dtype in ("int16", "uint"):
            key = np.random.randint(0, size, size=size, dtype=dtype)
        elif dtype == "float":
            key = np.concatenate(
                [np.random.random(ngroups) * 0.1, np.random.random(ngroups) * 10.0]
            )
        elif dtype == "object":
            key = ["foo"] * size
        elif dtype == "datetime":
            key = date_range("1/1/2011", periods=size, freq="s")

        cols = [f"values{n}" for n in range(ncols)]
        df = DataFrame(values, columns=cols)
        df["key"] = key

        if len(cols) == 1:
            cols = cols[0]

        # Not everything supports the engine keyword yet
        kwargs = {}
        if engine == "numba":
            kwargs["engine"] = engine

        if application == "transformation":
            self.as_group_method = lambda: df.groupby("key")[cols].transform(
                method, **kwargs
            )
            self.as_field_method = lambda: df.groupby(cols)["key"].transform(
                method, **kwargs
            )
        else:
            self.as_group_method = partial(
                getattr(df.groupby("key")[cols], method), **kwargs
            )
            self.as_field_method = partial(
                getattr(df.groupby(cols)["key"], method), **kwargs
            )

    def time_dtype_as_group(self, dtype, method, application, ncols, engine):
        # [FUNCTION DOC]
        # Times the prepared method applied to the value columns grouped by the
        # key column. Parameters are unused here.
        self.as_group_method()

    def time_dtype_as_field(self, dtype, method, application, ncols, engine):
        # [FUNCTION DOC]
        # Times the prepared method applied to the key column grouped by the value
        # column. Parameters are unused here.
        self.as_field_method()


class GroupByCythonAgg:
    # [CLASS DOC]
    # Benchmarks specifically targeting the cython aggregation algorithms, using a
    # large frame with a simple key so most time is spent in the grouped
    # aggregation. Parametrized by dtype (float64 only) and aggregation name.
    # State: df. Subclassed by the numba variant.
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    param_names = ["dtype", "method"]
    params = [
        ["float64"],
        [
            "sum",
            "prod",
            "min",
            "max",
            "idxmin",
            "idxmax",
            "mean",
            "median",
            "var",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype, method):
        # [FUNCTION DOC]
        # Builds a frame of one million rows by ten random normal columns named a
        # to j plus an integer key column with 100 groups. The dtype and method
        # parameters are unused here. Stores as df.
        N = 1_000_000
        df = DataFrame(np.random.randn(N, 10), columns=list("abcdefghij"))
        df["key"] = np.random.randint(0, 100, size=N)
        self.df = df

    def time_frame_agg(self, dtype, method):
        # [FUNCTION DOC]
        # Times grouping by the key and aggregating all columns with the named
        # method using the default engine.
        self.df.groupby("key").agg(method)


class GroupByNumbaAgg(GroupByCythonAgg):
    # [CLASS DOC]
    # Numba-engine variant of the compiled-aggregation benchmark. Inherits the
    # parameters and frame construction from the cython benchmark class and
    # overrides setup (to skip unsupported methods) and the timing method (to use
    # the numba engine).
    """
    Benchmarks specifically targeting our numba aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    def setup(self, dtype, method):
        # [FUNCTION DOC]
        # Raises a not-implemented error (skipping the benchmark) when the method
        # has no numba kernel; otherwise delegates to the parent setup.
        if method in _numba_unsupported_methods:
            raise NotImplementedError
        super().setup(dtype, method)

    def time_frame_agg(self, dtype, method):
        # [FUNCTION DOC]
        # Times grouping by the key and aggregating all columns with the named
        # method using the numba engine.
        self.df.groupby("key").agg(method, engine="numba")


class GroupByCythonAggEaDtypes:
    # [CLASS DOC]
    # Benchmarks targeting the cython aggregation algorithms for nullable
    # extension dtypes, using a large frame with a simple key. Parametrized by
    # dtype (Float64, Int64, Int32) and aggregation name. State: df.
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    param_names = ["dtype", "method"]
    params = [
        ["Float64", "Int64", "Int32"],
        [
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "median",
            "var",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype, method):
        # [FUNCTION DOC]
        # Builds a one million by ten frame of random integers below 100 in the
        # nullable dtype, with columns a to j, sets every fifth row starting at
        # one to missing in all those columns, and adds an integer key column with
        # 100 groups. Stores as df. The method parameter is unused here.
        N = 1_000_000
        df = DataFrame(
            np.random.randint(0, high=100, size=(N, 10)),
            columns=list("abcdefghij"),
            dtype=dtype,
        )
        df.loc[list(range(1, N, 5)), list("abcdefghij")] = NA
        df["key"] = np.random.randint(0, 100, size=N)
        self.df = df

    def time_frame_agg(self, dtype, method):
        # [FUNCTION DOC]
        # Times grouping by the key and aggregating all columns with the named
        # method.
        self.df.groupby("key").agg(method)


class Cumulative:
    # [CLASS DOC]
    # Benchmark of group-wise cumulative minimum, maximum and sum through the
    # transform method, with and without missing values. Parametrized by dtype
    # (float64, int64, Float64, Int64), method and a with_nans flag. State: df.
    param_names = ["dtype", "method", "with_nans"]
    params = [
        ["float64", "int64", "Float64", "Int64"],
        ["cummin", "cummax", "cumsum"],
        [True, False],
    ]

    def setup(self, dtype, method, with_nans):
        # [FUNCTION DOC]
        # Raises a not-implemented error for the plain int64 dtype with missing
        # values (the frame cannot be built). Creates 500000 random group keys
        # below 100 and a 500000 by 5 matrix of random integers between minus ten
        # and nine. With missing values: converts to float, sets every second and
        # every third row to NaN; for nullable dtypes converts to object and uses
        # the missing marker instead; builds the frame with columns a to e in the
        # dtype. Without: builds the frame and casts to the dtype without copying.
        # Adds the key column and stores as df. The method parameter is unused
        # here.
        if with_nans and dtype == "int64":
            raise NotImplementedError("Construction of df would raise")

        N = 500_000
        keys = np.random.randint(0, 100, size=N)
        vals = np.random.randint(-10, 10, (N, 5))

        if with_nans:
            null_vals = vals.astype(float, copy=True)
            null_vals[::2, :] = np.nan
            null_vals[::3, :] = np.nan
            if dtype in ["Int64", "Float64"]:
                null_vals = null_vals.astype(object)
                null_vals[::2, :] = NA
                null_vals[::3, :] = NA
            df = DataFrame(null_vals, columns=list("abcde"), dtype=dtype)
            df["key"] = keys
            self.df = df
        else:
            df = DataFrame(vals, columns=list("abcde")).astype(dtype, copy=False)
            df["key"] = keys
            self.df = df

    def time_frame_transform(self, dtype, method, with_nans):
        # [FUNCTION DOC]
        # Times the group-wise transform with the named cumulative method.
        self.df.groupby("key").transform(method)


class RankWithTies:
    # [CLASS DOC]
    # Benchmark (issue 21237) of group-wise ranking where all values tie.
    # Parametrized by dtype (float64, float32, int64, datetime64) and the
    # tie-breaking method (first, average, dense, min, max). State: df.
    # GH 21237
    param_names = ["dtype", "tie_method"]
    params = [
        ["float64", "float32", "int64", "datetime64"],
        ["first", "average", "dense", "min", "max"],
    ]

    def setup(self, dtype, tie_method):
        # [FUNCTION DOC]
        # Builds a frame of 10000 rows with a values column in which every entry
        # is identical (a repeated timestamp for datetime64, otherwise ones in the
        # dtype) and a key column of the constant foo, so a single group of all
        # ties. Stores as df.
        N = 10**4
        if dtype == "datetime64":
            data = np.array([Timestamp("2011/01/01")] * N, dtype=dtype)
        else:
            data = np.ones(N, dtype=dtype)
        self.df = DataFrame({"values": data, "key": ["foo"] * N})

    def time_rank_ties(self, dtype, tie_method):
        # [FUNCTION DOC]
        # Times the group-wise rank with the given tie-breaking method.
        self.df.groupby("key").rank(method=tie_method)


class Float32:
    # [CLASS DOC]
    # Benchmark (issue 13335) of grouping 32-bit floats by themselves and
    # summing. No parameters. State: df.
    # GH 13335
    def setup(self):
        # [FUNCTION DOC]
        # Builds 10000 small (below 0.1) and 10000 larger (below 10) random
        # 32-bit floats, concatenates them, repeats each ten times, and makes a
        # frame with two identical columns a and b.
        tmp1 = (np.random.random(10000) * 0.1).astype(np.float32)
        tmp2 = (np.random.random(10000) * 10.0).astype(np.float32)
        tmp = np.concatenate((tmp1, tmp2))
        arr = np.repeat(tmp, 10)
        self.df = DataFrame({"a": arr, "b": arr})

    def time_sum(self):
        # [FUNCTION DOC]
        # Times grouping by column a and summing column b.
        self.df.groupby(["a"])["b"].sum()


class String:
    # [CLASS DOC]
    # Benchmark (issue 41596) of group-wise aggregation on string-typed columns.
    # Parametrized by string dtype name and aggregation name. State: df.
    # GH#41596
    param_names = ["dtype", "method"]
    params = [
        ["str", "string[python]"],
        [
            "sum",
            "min",
            "max",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype, method):
        # [FUNCTION DOC]
        # Builds a 10000-row frame with ten columns (letters a, b, c, d, e, f, g,
        # h, j, k, l) from random integers below 100 converted to the given string
        # dtype. Stores as df. The method parameter is unused here.
        cols = list("abcdefghjkl")
        self.df = DataFrame(
            np.random.randint(0, 100, size=(10_000, len(cols))),
            columns=cols,
            dtype=dtype,
        )

    def time_str_func(self, dtype, method):
        # [FUNCTION DOC]
        # Times grouping by column a and aggregating all other columns with the
        # named method.
        self.df.groupby("a")[self.df.columns[1:]].agg(method)


class Categories:
    # [CLASS DOC]
    # Benchmark of counting with a categorical key, with the observed-only option
    # on or off and with sorting on or off, for unordered, ordered, and
    # extra-unused-category key columns. Parametrized by the observed flag. State:
    # df, df_ordered and df_extra_cat.
    params = [True, False]
    param_names = ["observed"]

    def setup(self, observed):
        # [FUNCTION DOC]
        # With 100000 rows and a shared random float column b: df has categorical
        # key a built from random integers below 10000; df_ordered has the same
        # style key made ordered; df_extra_cat has a key drawn from the first 100
        # values but declared with 10000 categories (many unused). The observed
        # flag is unused here.
        N = 10**5
        arr = np.random.random(N)
        data = {"a": Categorical(np.random.randint(10000, size=N)), "b": arr}
        self.df = DataFrame(data)
        data = {
            "a": Categorical(np.random.randint(10000, size=N), ordered=True),
            "b": arr,
        }
        self.df_ordered = DataFrame(data)
        data = {
            "a": Categorical(
                np.random.randint(100, size=N), categories=np.arange(10000)
            ),
            "b": arr,
        }
        self.df_extra_cat = DataFrame(data)

    def time_groupby_sort(self, observed):
        # [FUNCTION DOC]
        # Times counting b grouped by the unordered categorical key (sorted keys).
        self.df.groupby("a", observed=observed)["b"].count()

    def time_groupby_nosort(self, observed):
        # [FUNCTION DOC]
        # Times the same count without sorting the group keys.
        self.df.groupby("a", observed=observed, sort=False)["b"].count()

    def time_groupby_ordered_sort(self, observed):
        # [FUNCTION DOC]
        # Times counting b grouped by the ordered categorical key (sorted keys).
        self.df_ordered.groupby("a", observed=observed)["b"].count()

    def time_groupby_ordered_nosort(self, observed):
        # [FUNCTION DOC]
        # Times the ordered-key count without sorting the group keys.
        self.df_ordered.groupby("a", observed=observed, sort=False)["b"].count()

    def time_groupby_extra_cat_sort(self, observed):
        # [FUNCTION DOC]
        # Times counting b grouped by the key with many unused categories (sorted).
        self.df_extra_cat.groupby("a", observed=observed)["b"].count()

    def time_groupby_extra_cat_nosort(self, observed):
        # [FUNCTION DOC]
        # Times the many-unused-categories count without sorting.
        self.df_extra_cat.groupby("a", observed=observed, sort=False)["b"].count()


class MultipleCategories:
    # [CLASS DOC]
    # Benchmark of counting and cumulative sum with two categorical keys, always
    # including unobserved categories. No parameters. State: df, df_ordered and
    # df_extra_cat.
    def setup(self):
        # [FUNCTION DOC]
        # With 1000 rows and a shared random float column b: df has two categorical
        # keys a1 and a2 from random integers below 10000; df_ordered makes them
        # ordered; df_extra_cat draws the keys from the first 100 values but declares
        # 1000 categories each.
        N = 10**3
        arr = np.random.random(N)
        data = {
            "a1": Categorical(np.random.randint(10000, size=N)),
            "a2": Categorical(np.random.randint(10000, size=N)),
            "b": arr,
        }
        self.df = DataFrame(data)
        data = {
            "a1": Categorical(np.random.randint(10000, size=N), ordered=True),
            "a2": Categorical(np.random.randint(10000, size=N), ordered=True),
            "b": arr,
        }
        self.df_ordered = DataFrame(data)
        data = {
            "a1": Categorical(np.random.randint(100, size=N), categories=np.arange(N)),
            "a2": Categorical(np.random.randint(100, size=N), categories=np.arange(N)),
            "b": arr,
        }
        self.df_extra_cat = DataFrame(data)

    def time_groupby_sort(self):
        # [FUNCTION DOC]
        # Times counting b grouped by both unordered keys, including unobserved
        # combinations, with sorting.
        self.df.groupby(["a1", "a2"], observed=False)["b"].count()

    def time_groupby_nosort(self):
        # [FUNCTION DOC]
        # Times the same count without sorting.
        self.df.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    def time_groupby_ordered_sort(self):
        # [FUNCTION DOC]
        # Times counting b grouped by both ordered keys with sorting.
        self.df_ordered.groupby(["a1", "a2"], observed=False)["b"].count()

    def time_groupby_ordered_nosort(self):
        # [FUNCTION DOC]
        # Times the ordered-keys count without sorting.
        self.df_ordered.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    def time_groupby_extra_cat_sort(self):
        # [FUNCTION DOC]
        # Times counting b grouped by the two keys with many unused categories,
        # with sorting.
        self.df_extra_cat.groupby(["a1", "a2"], observed=False)["b"].count()

    def time_groupby_extra_cat_nosort(self):
        # [FUNCTION DOC]
        # Times the many-unused-categories count without sorting.
        self.df_extra_cat.groupby(["a1", "a2"], observed=False, sort=False)["b"].count()

    def time_groupby_transform(self):
        # [FUNCTION DOC]
        # Times the group-wise cumulative sum of b over the two keys with many
        # unused categories.
        self.df_extra_cat.groupby(["a1", "a2"], observed=False)["b"].cumsum()


class Datelike:
    # [CLASS DOC]
    # Benchmark (issue 14338) of grouping a frame by an externally supplied
    # period or datetime range grouper. Parametrized by grouper kind
    # (period_range, date_range, date_range_tz). State: grouper and df.
    # GH 14338
    params = ["period_range", "date_range", "date_range_tz"]
    param_names = ["grouper"]

    def setup(self, grouper):
        # [FUNCTION DOC]
        # Builds a daily range of 10000 entries from 1900-01-01 using the chosen
        # constructor (period range, date range, or date range in the US Central
        # zone) as grouper, and a 10000 by 2 random normal frame as df.
        N = 10**4
        rng_map = {
            "period_range": period_range,
            "date_range": date_range,
            "date_range_tz": partial(date_range, tz="US/Central"),
        }
        self.grouper = rng_map[grouper]("1900-01-01", freq="D", periods=N)
        self.df = DataFrame(np.random.randn(10**4, 2))

    def time_sum(self, grouper):
        # [FUNCTION DOC]
        # Times grouping the frame by the range and summing each group.
        self.df.groupby(self.grouper).sum()


class SumBools:
    # [CLASS DOC]
    # Benchmark (issue 2692) of summing boolean data per group. No parameters.
    # State: df.
    # GH 2692
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 500-row frame with a unique integer group column ii and a
        # boolean column bb that is always true.
        N = 500
        self.df = DataFrame({"ii": range(N), "bb": [True] * N})

    def time_groupby_sum_booleans(self):
        # [FUNCTION DOC]
        # Times grouping by ii and summing.
        self.df.groupby("ii").sum()


class SumMultiLevel:
    # [CLASS DOC]
    # Benchmark (issue 9049) of summing a frame grouped by both levels of a
    # two-level index. The timeout is raised to 120 seconds. No parameters.
    # State: df.
    # GH 9049
    timeout = 120.0

    def setup(self):
        # [FUNCTION DOC]
        # Builds a 100-row frame with columns A (0 to 49, twice), B (0 to 99) and
        # C (constant one), then sets A and B as a two-level index.
        N = 50
        self.df = DataFrame(
            {"A": list(range(N)) * 2, "B": range(N * 2), "C": 1}
        ).set_index(["A", "B"])

    def time_groupby_sum_multiindex(self):
        # [FUNCTION DOC]
        # Times grouping by both index levels and summing.
        self.df.groupby(level=[0, 1]).sum()


class SumTimeDelta:
    # [CLASS DOC]
    # Benchmark (issue 20660) of summing timedelta data and integer data grouped by
    # a function of the index. No parameters. State: df and df_int.
    # GH 20660
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 10000 by 100 frame of random integers between 1000 and 99999
        # indexed by random integers below 200, converted to nanosecond
        # timedeltas as df. df_int is a copy converted to 64-bit integers.
        N = 10**4
        self.df = DataFrame(
            np.random.randint(1000, 100000, (N, 100)),
            index=np.random.randint(200, size=(N,)),
        ).astype("timedelta64[ns]")
        self.df_int = self.df.copy().astype("int64")

    def time_groupby_sum_timedelta(self):
        # [FUNCTION DOC]
        # Times grouping the timedelta frame by an identity function of the index
        # label and summing.
        self.df.groupby(lambda x: x).sum()

    def time_groupby_sum_int(self):
        # [FUNCTION DOC]
        # Times grouping the integer frame by an identity function of the index
        # label and summing.
        self.df_int.groupby(lambda x: x).sum()


class Transform:
    # [CLASS DOC]
    # Benchmark of group-wise transform with a Python function versus a named
    # reduction, on a multi-indexed frame, wide and tall frames, and multi-key
    # frames with different duplication patterns. No parameters. State: df,
    # df_wide, df_tall, df1, df2, df3, df4.
    def setup(self):
        # [FUNCTION DOC]
        # Builds df (400 by 250 rows, three columns col1, col20, col3, with a few
        # NaN entries, indexed by a two-level index lev1 and lev2); df_wide
        # (1000 by 1000 random, indexed by random integers below 10); df_tall (a
        # million by 3 random, indexed by random integers below 5); df1 (20000 rows
        # of three integer columns jim, joe, jolie of random values below 20000),
        # df2 (copy of df1 with jim replaced by joe), df3 (similar to df1 but
        # values below 2000) and df4 (copy of df3 with jim replaced by joe).
        n1 = 400
        n2 = 250
        index = MultiIndex(
            levels=[np.arange(n1), Index([f"i-{i}" for i in range(n2)], dtype=object)],
            codes=[np.repeat(range(n1), n2).tolist(), list(range(n2)) * n1],
            names=["lev1", "lev2"],
        )
        arr = np.random.randn(n1 * n2, 3)
        arr[::10000, 0] = np.nan
        arr[1::10000, 1] = np.nan
        arr[2::10000, 2] = np.nan
        data = DataFrame(arr, index=index, columns=["col1", "col20", "col3"])
        self.df = data

        n = 1000
        self.df_wide = DataFrame(
            np.random.randn(n, n),
            index=np.random.choice(range(10), n),
        )

        n = 1_000_000
        self.df_tall = DataFrame(
            np.random.randn(n, 3),
            index=np.random.randint(0, 5, n),
        )

        n = 20000
        self.df1 = DataFrame(
            np.random.randint(1, n, (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df2 = self.df1.copy()
        self.df2["jim"] = self.df2["joe"]

        self.df3 = DataFrame(
            np.random.randint(1, (n / 10), (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df4 = self.df3.copy()
        self.df4["jim"] = self.df4["joe"]

    def time_transform_lambda_max(self):
        # [FUNCTION DOC]
        # Times a transform by the first index level using the built-in maximum
        # applied through a function.
        self.df.groupby(level="lev1").transform(lambda x: max(x))

    def time_transform_str_max(self):
        # [FUNCTION DOC]
        # Times a transform by the first index level using the maximum given by
        # name.
        self.df.groupby(level="lev1").transform("max")

    def time_transform_lambda_max_tall(self):
        # [FUNCTION DOC]
        # Times a transform of the tall frame by index level zero using a function
        # that takes the column-wise maximum.
        self.df_tall.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    def time_transform_lambda_max_wide(self):
        # [FUNCTION DOC]
        # Times a transform of the wide frame by index level zero using a function
        # that takes the column-wise maximum.
        self.df_wide.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    def time_transform_multi_key1(self):
        # [FUNCTION DOC]
        # Times a maximum transform of jolie grouped by jim and joe (df1).
        self.df1.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key2(self):
        # [FUNCTION DOC]
        # Times a maximum transform of jolie grouped by jim and joe (df2, where
        # the two keys are identical).
        self.df2.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key3(self):
        # [FUNCTION DOC]
        # Times a maximum transform of jolie grouped by jim and joe (df3, fewer
        # distinct values).
        self.df3.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key4(self):
        # [FUNCTION DOC]
        # Times a maximum transform of jolie grouped by jim and joe (df4, fewer
        # distinct values and identical keys).
        self.df4.groupby(["jim", "joe"])["jolie"].transform("max")


class TransformBools:
    # [CLASS DOC]
    # Benchmark of a mean transform grouped by a run-length-style integer array
    # derived from random boolean transition points. No parameters. State: g and
    # df.
    def setup(self):
        # [FUNCTION DOC]
        # With 120000 positions, picks 1400 random transition positions (with
        # possible repeats), marks them in a boolean array, and takes the running
        # sum as group ids g (so groups are consecutive runs). Builds df with a
        # random uniform signal column.
        N = 120000
        transition_points = np.sort(np.random.choice(np.arange(N), 1400))
        transitions = np.zeros(N, dtype=np.bool_)
        transitions[transition_points] = True
        self.g = transitions.cumsum()
        self.df = DataFrame({"signal": np.random.rand(N)})

    def time_transform_mean(self):
        # [FUNCTION DOC]
        # Times grouping the signal by the run ids and transforming with the mean.
        self.df["signal"].groupby(self.g).transform("mean")


class TransformNaN:
    # [CLASS DOC]
    # Benchmark (issue 12737) of a first-value transform on mostly-missing data.
    # No parameters. State: df_nans.
    # GH 12737
    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of 10000 rows with a key (the integers below 1000, each
        # repeated ten times) and two all-missing columns B and C, then sets the
        # fifth row of each ten-row block in B and C to the value five.
        self.df_nans = DataFrame(
            {"key": np.repeat(np.arange(1000), 10), "B": np.nan, "C": np.nan}
        )
        self.df_nans.loc[4::10, "B":"C"] = 5

    def time_first(self):
        # [FUNCTION DOC]
        # Times grouping by key and transforming with the first non-missing value.
        self.df_nans.groupby("key").transform("first")


class TransformEngine:
    # [CLASS DOC]
    # Benchmark of the transform method with the numba and cython engines for a
    # series and a frame grouper. Parametrized by the numba parallel flag. State:
    # parallel and grouper.
    param_names = ["parallel"]
    params = [[True, False]]

    def setup(self, parallel):
        # [FUNCTION DOC]
        # Builds a 100000-row frame with column zero a string key (100 distinct
        # values cycled) and column one an integer (0 to 99 cycled), groups it by
        # column zero. Stores the parallel flag and the grouper.
        N = 10**3
        data = DataFrame(
            {0: [str(i) for i in range(100)] * N, 1: list(range(100)) * N},
            columns=[0, 1],
        )
        self.parallel = parallel
        self.grouper = data.groupby(0)

    def time_series_numba(self, parallel):
        # [FUNCTION DOC]
        # Times transforming column one per group with a numba-compiled function
        # (multiplying by five), passing the parallel flag as an engine option.
        def function(values, index):
            # [FUNCTION DOC]
            # HELPER: transform function for this timing method. Takes the group
            # values and index (the index is required by the numba engine
            # signature but unused) and returns the values multiplied by five.
            return values * 5

        self.grouper[1].transform(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_series_cython(self, parallel):
        # [FUNCTION DOC]
        # Times transforming column one per group with a plain Python function
        # (multiplying by five) on the default compiled engine. The parameter is
        # unused.
        def function(values):
            # [FUNCTION DOC]
            # HELPER: transform function for this timing method. Returns the group
            # values multiplied by five.
            return values * 5

        self.grouper[1].transform(function, engine="cython")

    def time_dataframe_numba(self, parallel):
        # [FUNCTION DOC]
        # Times transforming the whole grouped frame with a numba-compiled
        # function (multiplying by five), passing the parallel flag.
        def function(values, index):
            # [FUNCTION DOC]
            # HELPER: transform function for this timing method. Takes group
            # values and index (unused, required by the numba signature) and
            # returns the values multiplied by five.
            return values * 5

        self.grouper.transform(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_dataframe_cython(self, parallel):
        # [FUNCTION DOC]
        # Times transforming the whole grouped frame with a plain Python function
        # (multiplying by five) on the cython engine. The parameter is unused.
        def function(values):
            # [FUNCTION DOC]
            # HELPER: transform function for this timing method. Returns the group
            # values multiplied by five.
            return values * 5

        self.grouper.transform(function, engine="cython")


class AggEngine:
    # [CLASS DOC]
    # Benchmark of the aggregate method with the numba and cython engines for a
    # series and a frame grouper, using a function with a Python loop. Parametrized
    # by the numba parallel flag. State: parallel and grouper.
    param_names = ["parallel"]
    params = [[True, False]]

    def setup(self, parallel):
        # [FUNCTION DOC]
        # Builds a 100000-row frame with column zero a string key (100 distinct
        # values cycled) and column one an integer (0 to 99 cycled), groups it by
        # column zero. Stores the parallel flag and the grouper.
        N = 10**3
        data = DataFrame(
            {0: [str(i) for i in range(100)] * N, 1: list(range(100)) * N},
            columns=[0, 1],
        )
        self.parallel = parallel
        self.grouper = data.groupby(0)

    def time_series_numba(self, parallel):
        # [FUNCTION DOC]
        # Times aggregating column one per group with a numba-compiled loop
        # function, passing the parallel flag as an engine option.
        def function(values, index):
            # [FUNCTION DOC]
            # HELPER: aggregation function for this timing method. Takes group
            # values and index (index unused, required by the numba signature).
            # Walks the values: for odd positions adds the value plus five to a
            # running total, for even positions adds twice the value. Returns the
            # total.
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper[1].agg(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_series_cython(self, parallel):
        # [FUNCTION DOC]
        # Times aggregating column one per group with the same loop function on
        # the cython engine. The parameter is unused.
        def function(values):
            # [FUNCTION DOC]
            # HELPER: aggregation function for this timing method. Walks the
            # values: for odd positions adds the value plus five to a running
            # total, for even positions adds twice the value. Returns the total.
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper[1].agg(function, engine="cython")

    def time_dataframe_numba(self, parallel):
        # [FUNCTION DOC]
        # Times aggregating the whole grouped frame with a numba-compiled loop
        # function, passing the parallel flag.
        def function(values, index):
            # [FUNCTION DOC]
            # HELPER: aggregation function for this timing method. Takes group
            # values and index (index unused, required by the numba signature).
            # Walks the values: odd positions add the value plus five, even
            # positions add twice the value. Returns the total.
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper.agg(
            function, engine="numba", engine_kwargs={"parallel": self.parallel}
        )

    def time_dataframe_cython(self, parallel):
        # [FUNCTION DOC]
        # Times aggregating the whole grouped frame with the loop function on the
        # cython engine. The parameter is unused.
        def function(values):
            # [FUNCTION DOC]
            # HELPER: aggregation function for this timing method. Walks the
            # values: odd positions add the value plus five, even positions add
            # twice the value. Returns the total.
            total = 0
            for i, value in enumerate(values):
                if i % 2:
                    total += value + 5
                else:
                    total += value * 2
            return total

        self.grouper.agg(function, engine="cython")


class Sample:
    # [CLASS DOC]
    # Benchmark of drawing one random row per group, with and without sampling
    # weights. No parameters. State: df, groups and weights.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000-row frame with a zero column a, group labels (each row its
        # own group) and a weights array of ones.
        N = 10**3
        self.df = DataFrame({"a": np.zeros(N)})
        self.groups = np.arange(0, N)
        self.weights = np.ones(N)

    def time_sample(self):
        # [FUNCTION DOC]
        # Times sampling one row per group.
        self.df.groupby(self.groups).sample(n=1)

    def time_sample_weights(self):
        # [FUNCTION DOC]
        # Times sampling one row per group using the weights.
        self.df.groupby(self.groups).sample(n=1, weights=self.weights)


class Resample:
    # [CLASS DOC]
    # Benchmark (issue 28635) of resampling within groups, using a timedelta
    # column or a timedelta index level. No parameters. State: df (timedeltas as a
    # column) and df_multiindex (timedeltas as an index level).
    # GH 28635
    def setup(self):
        # [FUNCTION DOC]
        # Builds a two-level index from three group ids and 20000 one-second
        # timedeltas (names groups and timedeltas), random integer data below
        # 1000, and two frames: df with the timedelta level moved into a column and
        # df_multiindex keeping both levels.
        num_timedeltas = 20_000
        num_groups = 3

        index = MultiIndex.from_product(
            [
                np.arange(num_groups),
                to_timedelta(np.arange(num_timedeltas), unit="s"),
            ],
            names=["groups", "timedeltas"],
        )
        data = np.random.randint(0, 1000, size=(len(index)))

        self.df = DataFrame(data, index=index).reset_index("timedeltas")
        self.df_multiindex = DataFrame(data, index=index)

    def time_resample(self):
        # [FUNCTION DOC]
        # Times grouping by the group level and resampling the timedelta column
        # to ten-second bins, taking the mean.
        self.df.groupby(level="groups").resample("10s", on="timedeltas").mean()

    def time_resample_multiindex(self):
        # [FUNCTION DOC]
        # Times grouping by the group level and resampling by the timedelta index
        # level to ten-second bins, taking the mean.
        self.df_multiindex.groupby(level="groups").resample(
            "10s", level="timedeltas"
        ).mean()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
