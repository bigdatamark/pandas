# [MODULE DOC]
# Benchmarks for reindexing and alignment: reindexing frames and multi-indexed
# series, fill-method reindexing, level alignment, duplicate removal and string
# index alignment. Public contents are the benchmark classes Reindex,
# ReindexMethod, LevelAlign, DropDuplicates and Align. No import-time side
# effects; the final line re-exports the common asv setup hook.
import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
    period_range,
)


class Reindex:
    # [CLASS DOC]
    # Benchmark of reindexing frames by dates and by columns, and of reindexing
    # multi-indexed series with and without the cached values of the index. No
    # parameters. State: df, rng_subset, df2, s, s_subset, s_subset_no_cache, s2
    # and s2_subset.
    def setup(self):
        # [FUNCTION DOC]
        # Builds: df (10000 rows of 10 random columns on a minute datetime index
        # from 1970, plus a constant string column foo); rng_subset (every second
        # timestamp as an index); df2 (10000 by 30 random on a range index with
        # integer columns); s (a random series of one million entries on a
        # two-level string index, 5000 by 200); s_subset (every second entry of s);
        # s_subset_no_cache (a copy of that); s2 (a random series on the product of
        # the minute timestamps and a range of 100); and s2_subset (a copy of every
        # second entry of s2).
        rng = date_range(start="1/1/1970", periods=10000, freq="1min")
        self.df = DataFrame(np.random.rand(10000, 10), index=rng, columns=range(10))
        self.df["foo"] = "bar"
        self.rng_subset = Index(rng[::2])
        self.df2 = DataFrame(
            index=range(10000), data=np.random.rand(10000, 30), columns=range(30)
        )
        N = 5000
        K = 200
        level1 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        level2 = np.tile(Index([f"i-{i}" for i in range(K)], dtype=object).values, N)
        index = MultiIndex.from_arrays([level1, level2])
        self.s = Series(np.random.randn(N * K), index=index)
        self.s_subset = self.s[::2]
        self.s_subset_no_cache = self.s[::2].copy()

        mi = MultiIndex.from_product([rng, range(100)])
        self.s2 = Series(np.random.randn(len(mi)), index=mi)
        self.s2_subset = self.s2[::2].copy()

    def time_reindex_dates(self):
        # [FUNCTION DOC]
        # Times reindexing the date-indexed frame to every second timestamp.
        self.df.reindex(self.rng_subset)

    def time_reindex_columns(self):
        # [FUNCTION DOC]
        # Times reindexing the second frame to the columns at positions 1 to 4 of
        # the first frame's columns.
        self.df2.reindex(columns=self.df.columns[1:5])

    def time_reindex_multiindex_with_cache(self):
        # [FUNCTION DOC]
        # Times reindexing the series to the subset index; the subset index keeps
        # its cached values array between runs.
        # MultiIndex._values gets cached
        self.s.reindex(self.s_subset.index)

    def time_reindex_multiindex_no_cache(self):
        # [FUNCTION DOC]
        # Times reindexing the series to a fresh copy of the subset index so the
        # values array is not cached from earlier runs.
        # Copy to avoid MultiIndex._values getting cached
        self.s.reindex(self.s_subset_no_cache.index.copy())

    def time_reindex_multiindex_no_cache_dates(self):
        # [FUNCTION DOC]
        # Times reindexing the subset of the datetime-level series to the full
        # index, copied to avoid cached values.
        # Copy to avoid MultiIndex._values getting cached
        self.s2_subset.reindex(self.s2.index.copy())


class ReindexMethod:
    # [CLASS DOC]
    # Benchmark of reindexing a half-empty series with a forward or backward fill
    # method, for datetime and period indexes. Parametrized by method (pad,
    # backfill) and index constructor function (date range, period range). State:
    # idx and ts.
    params = [["pad", "backfill"], [date_range, period_range]]
    param_names = ["method", "constructor"]

    def setup(self, method, constructor):
        # [FUNCTION DOC]
        # Builds a 100000-entry minute-frequency index from 2000-01-01 using the
        # constructor, and ts, a random series on it keeping only every second
        # element. The method parameter is unused here.
        N = 100000
        self.idx = constructor("1/1/2000", periods=N, freq="1min")
        self.ts = Series(np.random.randn(N), index=self.idx)[::2]

    def time_reindex_method(self, method, constructor):
        # [FUNCTION DOC]
        # Times reindexing ts to the full index with the fill method.
        self.ts.reindex(self.idx, method=method)


class LevelAlign:
    # [CLASS DOC]
    # Benchmark of aligning and reindexing a frame to a three-level multi-index
    # by its middle level. No parameters. State: index, df and df_level.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a three-level multi-index (levels of 10, 100 and 100 integers)
        # with 100000 entries from explicit codes; df, a four-column random frame
        # on it; and df_level, a four-column random frame of 100 rows indexed by
        # the second level values.
        self.index = MultiIndex(
            levels=[np.arange(10), np.arange(100), np.arange(100)],
            codes=[
                np.arange(10).repeat(10000),
                np.tile(np.arange(100).repeat(100), 10),
                np.tile(np.tile(np.arange(100), 100), 10),
            ],
        )
        self.df = DataFrame(np.random.randn(len(self.index), 4), index=self.index)
        self.df_level = DataFrame(np.random.randn(100, 4), index=self.index.levels[1])

    def time_align_level(self):
        # [FUNCTION DOC]
        # Times aligning the multi-indexed frame with the single-index frame on
        # level one without copying.
        self.df.align(self.df_level, level=1, copy=False)

    def time_reindex_level(self):
        # [FUNCTION DOC]
        # Times reindexing the single-index frame to the full multi-index, matching
        # on level one.
        self.df_level.reindex(self.index, level=1)


class DropDuplicates:
    # [CLASS DOC]
    # Benchmark of duplicate-row removal for frames and series of various dtypes,
    # in place or not. Parametrized by the inplace flag. State: df, df_nan, s,
    # s_str, df_int and df_bool.
    params = [True, False]
    param_names = ["inplace"]

    def setup(self, inplace):
        # [FUNCTION DOC]
        # Builds: df (100000 rows: two identical string key columns, each distinct
        # value repeated ten times, and a random value column); df_nan (a copy with
        # the first 10000 rows set to missing); s (10000 random integers below
        # 1000); s_str (1000 distinct strings tiled ten times); df_int (a million
        # rows of one integer key below 10000); and df_bool (10000 rows by 10
        # random boolean columns). The inplace flag is unused here.
        N = 10000
        K = 10
        key1 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        key2 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        self.df = DataFrame(
            {"key1": key1, "key2": key2, "value": np.random.randn(N * K)}
        )
        self.df_nan = self.df.copy()
        self.df_nan.iloc[:10000, :] = np.nan

        self.s = Series(np.random.randint(0, 1000, size=10000))
        self.s_str = Series(
            np.tile(Index([f"i-{i}" for i in range(1000)], dtype=object).values, 10)
        )

        N = 1000000
        K = 10000
        key1 = np.random.randint(0, K, size=N)
        self.df_int = DataFrame({"key1": key1})
        self.df_bool = DataFrame(np.random.randint(0, 2, size=(K, 10), dtype=bool))

    def time_frame_drop_dups(self, inplace):
        # [FUNCTION DOC]
        # Times dropping duplicate rows judged on the two key columns (mutates the
        # frame if inplace).
        self.df.drop_duplicates(["key1", "key2"], inplace=inplace)

    def time_frame_drop_dups_na(self, inplace):
        # [FUNCTION DOC]
        # Times dropping duplicates on the two key columns of the frame with
        # missing rows (mutates the frame if inplace).
        self.df_nan.drop_duplicates(["key1", "key2"], inplace=inplace)

    def time_series_drop_dups_int(self, inplace):
        # [FUNCTION DOC]
        # Times dropping duplicates in the integer series (mutates if inplace).
        self.s.drop_duplicates(inplace=inplace)

    def time_series_drop_dups_string(self, inplace):
        # [FUNCTION DOC]
        # Times dropping duplicates in the string series (mutates if inplace).
        self.s_str.drop_duplicates(inplace=inplace)

    def time_frame_drop_dups_int(self, inplace):
        # [FUNCTION DOC]
        # Times dropping duplicate rows in the single-integer-column frame
        # (mutates if inplace).
        self.df_int.drop_duplicates(inplace=inplace)

    def time_frame_drop_dups_bool(self, inplace):
        # [FUNCTION DOC]
        # Times dropping duplicate rows in the boolean frame (mutates if inplace).
        self.df_bool.drop_duplicates(inplace=inplace)


class Align:
    # [CLASS DOC]
    # Benchmark of adding two string-indexed series where one covers only part of
    # the other in a random order (the case described in a blog post titled "pandas
    # escaped the zoo"). No parameters. State: x and y.
    # blog "pandas escaped the zoo"
    def setup(self):
        # [FUNCTION DOC]
        # Builds a string index of 50000 entries and a random series x on it. y is a
        # random series on a random sample of 40000 distinct labels from the index
        # (unsorted).
        n = 50000
        indices = Index([f"i-{i}" for i in range(n)], dtype=object)
        subsample_size = 40000
        self.x = Series(np.random.randn(n), indices)
        self.y = Series(
            np.random.randn(subsample_size),
            index=np.random.choice(indices, subsample_size, replace=False),
        )

    def time_align_series_irregular_string(self):
        # [FUNCTION DOC]
        # Times adding the two series, which aligns their string indexes.
        self.x + self.y


from .pandas_vb_common import setup  # noqa: F401 isort:skip
