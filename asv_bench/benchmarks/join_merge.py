# [MODULE DOC]
# Benchmarks for combining data: concatenation, joins, merges (on keys, on
# extension dtypes, on integers, datetimes and categoricals, ordered and as-of
# merges, multi-index merges) and index alignment. Public contents are the
# benchmark classes Concat, ConcatDataFrames, ConcatIndexDtype, Join, JoinIndex,
# JoinMultiindexSubset, JoinEmpty, JoinNonUnique, Merge, MergeEA, I8Merge,
# UniqueMerge, MergeDatetime, MergeCategoricals, MergeOrdered, MergeAsof,
# MergeMultiIndex and Align. At import time it binds the ordered merge function,
# falling back to its old name for older library versions. Final line re-exports
# the common asv setup hook.
import string

import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    array,
    concat,
    date_range,
    merge,
    merge_asof,
)

try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered


class Concat:
    # [CLASS DOC]
    # Benchmark of concatenating many series, many small frames, and frames where
    # one input is empty or has fewer rows. Parametrized by axis. State: series,
    # small_frames, empty_left, empty_right and mixed_ndims.
    params = [0, 1]
    param_names = ["axis"]

    def setup(self, axis):
        # [FUNCTION DOC]
        # With N equal to 1000: builds a string-indexed series of the constant N
        # and nine trimmed slices of it (drop i elements from both ends for i from
        # one to nine), repeated 50 times (450 series); 1000 references to one 5 by
        # 4 random frame; and a one-column frame df of 1000 rows on a per-second
        # datetime index. Lists: empty_left (an empty frame then df), empty_right
        # (df then an empty frame) and mixed_ndims (df and its first half).
        N = 1000
        s = Series(N, index=Index([f"i-{i}" for i in range(N)], dtype=object))
        self.series = [s[i:-i] for i in range(1, 10)] * 50
        self.small_frames = [DataFrame(np.random.randn(5, 4))] * 1000
        df = DataFrame(
            {"A": range(N)}, index=date_range("20130101", periods=N, freq="s")
        )
        self.empty_left = [DataFrame(), df]
        self.empty_right = [df, DataFrame()]
        self.mixed_ndims = [df, df.head(N // 2)]

    def time_concat_series(self, axis):
        # [FUNCTION DOC]
        # Times concatenating the series along the axis without sorting.
        concat(self.series, axis=axis, sort=False)

    def time_concat_small_frames(self, axis):
        # [FUNCTION DOC]
        # Times concatenating the 1000 small frames along the axis.
        concat(self.small_frames, axis=axis)

    def time_concat_empty_right(self, axis):
        # [FUNCTION DOC]
        # Times concatenating a frame followed by an empty frame.
        concat(self.empty_right, axis=axis)

    def time_concat_empty_left(self, axis):
        # [FUNCTION DOC]
        # Times concatenating an empty frame followed by a frame.
        concat(self.empty_left, axis=axis)

    def time_concat_mixed_ndims(self, axis):
        # [FUNCTION DOC]
        # Times concatenating a frame with a shorter frame (first half) of the same
        # columns. Despite the name, both inputs are frames.
        concat(self.mixed_ndims, axis=axis)


class ConcatDataFrames:
    # [CLASS DOC]
    # Benchmark of concatenating twenty large float32 frames whose memory layout
    # is row-major or column-major. Parametrized by axis and the ignore_index flag.
    # State: frame_c and frame_f (lists of twenty references to one frame each).
    params = ([0, 1], [True, False])
    param_names = ["axis", "ignore_index"]

    def setup(self, axis, ignore_index):
        # [FUNCTION DOC]
        # Builds a 10000 by 200 zero float32 frame from a C-ordered array and a
        # second from a Fortran-ordered array, and lists of twenty references to
        # each as frame_c and frame_f.
        frame_c = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="C"))
        self.frame_c = [frame_c] * 20
        frame_f = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="F"))
        self.frame_f = [frame_f] * 20

    def time_c_ordered(self, axis, ignore_index):
        # [FUNCTION DOC]
        # Times concatenating the row-major frames.
        concat(self.frame_c, axis=axis, ignore_index=ignore_index)

    def time_f_ordered(self, axis, ignore_index):
        # [FUNCTION DOC]
        # Times concatenating the column-major frames.
        concat(self.frame_f, axis=axis, ignore_index=ignore_index)


class ConcatIndexDtype:
    # [CLASS DOC]
    # Benchmark of concatenating five series with overlapping indexes of various
    # dtypes and orderings (including a leading missing label). Parametrized by
    # index dtype, structure (monotonic, non_monotonic, has_na), axis and sort
    # flag. State: series.
    params = (
        [
            "datetime64[ns]",
            "int64",
            "Int64",
            "int64[pyarrow]",
            "string[python]",
            "string[pyarrow]",
        ],
        ["monotonic", "non_monotonic", "has_na"],
        [0, 1],
        [True, False],
    )
    param_names = ["dtype", "structure", "axis", "sort"]

    def setup(self, dtype, structure, axis, sort):
        # [FUNCTION DOC]
        # Builds 10000 label values for the dtype (daily dates from 1970, integers,
        # or distinct strings; raises a not-implemented error for other dtypes) and
        # an index of that dtype. For monotonic structure sorts it; for
        # non_monotonic reverses it; for has_na raises a not-implemented error if the
        # index cannot hold missing values, otherwise prepends a missing label (the
        # index grows by one); any other structure raises a not-implemented error.
        # Builds five series, the i-th (i from one to five) holding the value i over
        # the index with its last i labels dropped, stored as series. The axis and
        # sort parameters are unused here.
        N = 10_000
        if dtype == "datetime64[ns]":
            vals = date_range("1970-01-01", periods=N)
        elif dtype in ("int64", "Int64", "int64[pyarrow]"):
            vals = np.arange(N, dtype=np.int64)
        elif dtype in ("string[python]", "string[pyarrow]"):
            vals = Index([f"i-{i}" for i in range(N)], dtype=object)
        else:
            raise NotImplementedError

        idx = Index(vals, dtype=dtype)

        if structure == "monotonic":
            idx = idx.sort_values()
        elif structure == "non_monotonic":
            idx = idx[::-1]
        elif structure == "has_na":
            if not idx._can_hold_na:
                raise NotImplementedError
            idx = Index([None], dtype=dtype).append(idx)
        else:
            raise NotImplementedError

        self.series = [Series(i, idx[:-i]) for i in range(1, 6)]

    def time_concat_series(self, dtype, structure, axis, sort):
        # [FUNCTION DOC]
        # Times concatenating the five series along the axis with the sort option.
        concat(self.series, axis=axis, sort=sort)


class Join:
    # [CLASS DOC]
    # Benchmark of joining a 100000-row frame to indexed frames by one or two keys,
    # including shuffled input and a cross join. Parametrized by the sort flag.
    # State: df_multi, key1, key2, df, df_key1, df_key2 and df_shuf.
    params = [True, False]
    param_names = ["sort"]

    def setup(self, sort):
        # [FUNCTION DOC]
        # Builds two string levels (10 and 1000 distinct values) and a two-level
        # index (df_multi) covering every combination (10000 rows, four random
        # columns A to D). Builds key arrays by tiling the level values; df has
        # 100000 rows with two random columns and the two key columns. df_key1 and
        # df_key2 are random four-column frames indexed by each level's values.
        # df_shuf is df with its rows reordered by a random permutation of the
        # positions.
        level1 = Index([f"i-{i}" for i in range(10)], dtype=object).values
        level2 = Index([f"i-{i}" for i in range(1000)], dtype=object).values
        codes1 = np.arange(10).repeat(1000)
        codes2 = np.tile(np.arange(1000), 10)
        index2 = MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        self.df_multi = DataFrame(
            np.random.randn(len(index2), 4), index=index2, columns=["A", "B", "C", "D"]
        )

        self.key1 = np.tile(level1.take(codes1), 10)
        self.key2 = np.tile(level2.take(codes2), 10)
        self.df = DataFrame(
            {
                "data1": np.random.randn(100000),
                "data2": np.random.randn(100000),
                "key1": self.key1,
                "key2": self.key2,
            }
        )

        self.df_key1 = DataFrame(
            np.random.randn(len(level1), 4), index=level1, columns=["A", "B", "C", "D"]
        )
        self.df_key2 = DataFrame(
            np.random.randn(len(level2), 4), index=level2, columns=["A", "B", "C", "D"]
        )

        shuf = np.arange(100000)
        np.random.shuffle(shuf)
        self.df_shuf = self.df.reindex(self.df.index[shuf])

    def time_join_dataframe_index_multi(self, sort):
        # [FUNCTION DOC]
        # Times joining df to the multi-indexed frame on both key columns.
        self.df.join(self.df_multi, on=["key1", "key2"], sort=sort)

    def time_join_dataframe_index_single_key_bigger(self, sort):
        # [FUNCTION DOC]
        # Times joining df to the frame indexed by the 1000 distinct second keys.
        self.df.join(self.df_key2, on="key2", sort=sort)

    def time_join_dataframe_index_single_key_small(self, sort):
        # [FUNCTION DOC]
        # Times joining df to the frame indexed by the 10 distinct first keys.
        self.df.join(self.df_key1, on="key1", sort=sort)

    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort):
        # [FUNCTION DOC]
        # Times joining the shuffled frame to the 1000-key frame on the second key.
        self.df_shuf.join(self.df_key2, on="key2", sort=sort)

    def time_join_dataframes_cross(self, sort):
        # [FUNCTION DOC]
        # Times a cross join of the first rows of df (label slice up to 2000,
        # inclusive) with the small frame.
        self.df.loc[:2000].join(self.df_key1, how="cross", sort=sort)


class JoinIndex:
    # [CLASS DOC]
    # Benchmark of a left join onto an indexed frame with many duplicate keys. No
    # parameters. State: left and right.
    def setup(self):
        # [FUNCTION DOC]
        # Builds left (5000 rows, columns jim and joe of random integers from 1 to
        # 99) and right (5000 rows, columns jolie and jolia of random integers from
        # 1 to 99, indexed by jolie).
        N = 5000
        self.left = DataFrame(
            np.random.randint(1, N / 50, (N, 2)), columns=["jim", "joe"]
        )
        self.right = DataFrame(
            np.random.randint(1, N / 50, (N, 2)), columns=["jolie", "jolia"]
        ).set_index("jolie")

    def time_left_outer_join_index(self):
        # [FUNCTION DOC]
        # Times joining left to right using the jim column against the right index
        # (default left join).
        self.left.join(self.right, on="jim")


class JoinMultiindexSubset:
    # [CLASS DOC]
    # Benchmark of joining a four-level multi-indexed frame to a two-level
    # multi-indexed frame on the shared levels. No parameters. State: left and
    # right.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 100000-entry multi-index of four identical levels named a to d
        # for left (single column col1 equal to one) and a two-level one named a, b
        # for right (single column col2 equal to two).
        N = 100_000
        mi1 = MultiIndex.from_arrays([np.arange(N)] * 4, names=["a", "b", "c", "d"])
        mi2 = MultiIndex.from_arrays([np.arange(N)] * 2, names=["a", "b"])
        self.left = DataFrame({"col1": 1}, index=mi1)
        self.right = DataFrame({"col2": 2}, index=mi2)

    def time_join_multiindex_subset(self):
        # [FUNCTION DOC]
        # Times joining left to right on the overlapping index levels.
        self.left.join(self.right)


class JoinEmpty:
    # [CLASS DOC]
    # Benchmark of inner joins where one side is an empty frame. No parameters.
    # State: df and df_empty.
    def setup(self):
        # [FUNCTION DOC]
        # Builds df (one column A of 100000 integers) and df_empty (an empty frame
        # with columns B and C of 64-bit integer dtype).
        N = 100_000
        self.df = DataFrame({"A": np.arange(N)})
        self.df_empty = DataFrame(columns=["B", "C"], dtype="int64")

    def time_inner_join_left_empty(self):
        # [FUNCTION DOC]
        # Times an inner join with the empty frame on the left.
        self.df_empty.join(self.df, how="inner")

    def time_inner_join_right_empty(self):
        # [FUNCTION DOC]
        # Times an inner join with the empty frame on the right.
        self.df.join(self.df_empty, how="inner")


class JoinNonUnique:
    # [CLASS DOC]
    # Benchmark (issue 6329) of multiplying two series whose indexes contain
    # repeated timestamps (an outer-join-like alignment of non-unique indexes). No
    # parameters. State: fracofday and temp.
    # outer join of non-unique
    # GH 6329
    def setup(self):
        # [FUNCTION DOC]
        # Builds minute timestamps from 2013-01-01 to 2013-01-23 and, for each, the
        # midnight of its day (second resolution). fracofday is the fraction of the
        # day elapsed for each timestamp (as a float) indexed by the repeated
        # midnight timestamps. temp is a series of ones over the daily range
        # selected at those repeated midnights (so with the same non-unique index).
        date_index = date_range("01-Jan-2013", "23-Jan-2013", freq="min")
        daily_dates = date_index.to_period("D").to_timestamp("s", "s")
        self.fracofday = date_index.values - daily_dates.values
        self.fracofday = self.fracofday.astype("timedelta64[ns]")
        self.fracofday = self.fracofday.astype(np.float64) / 86_400_000_000_000
        self.fracofday = Series(self.fracofday, daily_dates)
        index = date_range(date_index.min(), date_index.max(), freq="D")
        self.temp = Series(1.0, index)[self.fracofday.index]

    def time_join_non_unique_equal(self):
        # [FUNCTION DOC]
        # Times multiplying the two series with identical non-unique indexes.
        self.fracofday * self.temp


class Merge:
    # [CLASS DOC]
    # Benchmark of merging frames on one or two string/integer keys, including
    # empty sides and a cross merge. Parametrized by the sort flag. State: left,
    # right, df, df2 and df3.
    params = [True, False]
    param_names = ["sort"]

    def setup(self, sort):
        # [FUNCTION DOC]
        # Builds left (80000 rows: two identical string key columns tiled from
        # the first 8000 labels ten times, plus random values) and right (8000
        # rows: the labels from position 2000 onward in the same two key columns,
        # plus random values). Builds df (10000 rows of two integer keys with
        # repetition and random values), df2 (500 rows keyed by the first integer
        # key) and df3 (the first half of df).
        N = 10000
        indices = Index([f"i-{i}" for i in range(N)], dtype=object).values
        indices2 = Index([f"i-{i}" for i in range(N)], dtype=object).values
        key = np.tile(indices[:8000], 10)
        key2 = np.tile(indices2[:8000], 10)
        self.left = DataFrame(
            {"key": key, "key2": key2, "value": np.random.randn(80000)}
        )
        self.right = DataFrame(
            {
                "key": indices[2000:],
                "key2": indices2[2000:],
                "value2": np.random.randn(8000),
            }
        )

        self.df = DataFrame(
            {
                "key1": np.tile(np.arange(500).repeat(10), 2),
                "key2": np.tile(np.arange(250).repeat(10), 4),
                "value": np.random.randn(10000),
            }
        )
        self.df2 = DataFrame({"key1": np.arange(500), "value2": np.random.randn(500)})
        self.df3 = self.df[:5000]

    def time_merge_2intkey(self, sort):
        # [FUNCTION DOC]
        # Times merging left and right on their shared columns (key and key2).
        # Despite the name, the keys are strings.
        merge(self.left, self.right, sort=sort)

    def time_merge_dataframe_integer_2key(self, sort):
        # [FUNCTION DOC]
        # Times merging df and its first half on their shared integer key columns.
        merge(self.df, self.df3, sort=sort)

    def time_merge_dataframe_integer_key(self, sort):
        # [FUNCTION DOC]
        # Times merging df with df2 on the integer column key1.
        merge(self.df, self.df2, on="key1", sort=sort)

    def time_merge_dataframe_empty_right(self, sort):
        # [FUNCTION DOC]
        # Times merging left with an empty slice of right.
        merge(self.left, self.right.iloc[:0], sort=sort)

    def time_merge_dataframe_empty_left(self, sort):
        # [FUNCTION DOC]
        # Times merging an empty slice of left with right.
        merge(self.left.iloc[:0], self.right, sort=sort)

    def time_merge_dataframes_cross(self, sort):
        # [FUNCTION DOC]
        # Times a cross merge of the first rows (label slice to 2000, inclusive) of
        # left and right.
        merge(self.left.loc[:2000], self.right.loc[:2000], how="cross", sort=sort)


class MergeEA:
    # [CLASS DOC]
    # Benchmark of merging on a nullable integer or float key, with sorted or
    # unsorted inputs. Parametrized by nullable dtype and a monotonic flag. State:
    # left and right.
    params = [
        [
            "Int64",
            "Int32",
            "Int16",
            "UInt64",
            "UInt32",
            "UInt16",
            "Float64",
            "Float32",
        ],
        [True, False],
    ]
    param_names = ["dtype", "monotonic"]

    def setup(self, dtype, monotonic):
        # [FUNCTION DOC]
        # Takes the integers 1 to 9999. left has 80000 rows: a key from the first
        # 8000 of them tiled ten times in the nullable dtype and random values.
        # right has 7999 rows: a key from the integers from position 2000 onward in
        # the dtype and random values. If monotonic is true sorts both frames by
        # key. Stores as left and right.
        N = 10_000
        indices = np.arange(1, N)
        key = np.tile(indices[:8000], 10)
        self.left = DataFrame(
            {"key": Series(key, dtype=dtype), "value": np.random.randn(80000)}
        )
        self.right = DataFrame(
            {
                "key": Series(indices[2000:], dtype=dtype),
                "value2": np.random.randn(7999),
            }
        )
        if monotonic:
            self.left = self.left.sort_values("key")
            self.right = self.right.sort_values("key")

    def time_merge(self, dtype, monotonic):
        # [FUNCTION DOC]
        # Times merging left and right on their shared key column.
        merge(self.left, self.right)


class I8Merge:
    # [CLASS DOC]
    # Benchmark of merging two million-row integer frames on all shared integer
    # columns, for each join type. Parametrized by how. State: left and right.
    params = ["inner", "outer", "left", "right"]
    param_names = ["how"]

    def setup(self, how):
        # [FUNCTION DOC]
        # Builds left as a million rows by seven columns (A to G) of random integers
        # between minus 1000 and 999 plus a column left holding the row sum. right is
        # a fully shuffled sample of left with the sum column renamed right,
        # re-indexed from zero, with the sum column negated.
        low, high, n = -1000, 1000, 10**6
        self.left = DataFrame(
            np.random.randint(low, high, (n, 7)), columns=list("ABCDEFG")
        )
        self.left["left"] = self.left.sum(axis=1)
        self.right = self.left.sample(frac=1).rename({"left": "right"}, axis=1)
        self.right = self.right.reset_index(drop=True)
        self.right["right"] *= -1

    def time_i8merge(self, how):
        # [FUNCTION DOC]
        # Times merging left and right with the join type, on all shared columns.
        merge(self.left, self.right, how=how)


class UniqueMerge:
    # [CLASS DOC]
    # Benchmark of an inner merge on an integer key where the right key is unique,
    # for a large and a smaller key range. Parametrized by unique_elements. State:
    # left and right.
    params = [4_000_000, 1_000_000]
    param_names = ["unique_elements"]

    def setup(self, unique_elements):
        # [FUNCTION DOC]
        # Builds left and right frames of one million rows with a column a of random
        # integers from 1 to unique_elements minus one. Then makes right's key
        # unique by keeping the distinct values it contains and filling the
        # remaining rows with zero and descending negative integers, so no
        # duplicates remain. Stores as left and right.
        N = 1_000_000
        self.left = DataFrame({"a": np.random.randint(1, unique_elements, (N,))})
        self.right = DataFrame({"a": np.random.randint(1, unique_elements, (N,))})
        uniques = self.right.a.drop_duplicates()
        self.right["a"] = concat(
            [uniques, Series(np.arange(0, -(N - len(uniques)), -1))], ignore_index=True
        )

    def time_unique_merge(self, unique_elements):
        # [FUNCTION DOC]
        # Times an inner merge of left and right on column a.
        merge(self.left, self.right, how="inner")


class MergeDatetime:
    # [CLASS DOC]
    # Benchmark of merging on a datetime key with different or equal time units,
    # naive or timezone-aware, sorted or not. Parametrized by (left unit, right
    # unit) pair, timezone and monotonic flag. State: left and right.
    params = [
        [
            ("ns", "ns"),
            ("ms", "ms"),
            ("ns", "ms"),
        ],
        [None, "Europe/Brussels"],
        [True, False],
    ]
    param_names = ["units", "tz", "monotonic"]

    def setup(self, units, tz, monotonic):
        # [FUNCTION DOC]
        # Builds 10000 minute timestamps from 2012-01-01 in the timezone. left has
        # 100000 rows: keys sampled with replacement from those, converted to the
        # left unit, plus random values. right has 8000 rows: the first 8000 keys
        # converted to the right unit, plus random values. If monotonic is true
        # sorts both by key.
        unit_left, unit_right = units
        N = 10_000
        keys = Series(date_range("2012-01-01", freq="min", periods=N, tz=tz))
        self.left = DataFrame(
            {
                "key": keys.sample(N * 10, replace=True).dt.as_unit(unit_left),
                "value1": np.random.randn(N * 10),
            }
        )
        self.right = DataFrame(
            {
                "key": keys[:8000].dt.as_unit(unit_right),
                "value2": np.random.randn(8000),
            }
        )
        if monotonic:
            self.left = self.left.sort_values("key")
            self.right = self.right.sort_values("key")

    def time_merge(self, units, tz, monotonic):
        # [FUNCTION DOC]
        # Times merging left and right on their shared key column.
        merge(self.left, self.right)


class MergeCategoricals:
    # [CLASS DOC]
    # Benchmark of merging on an integer key with payload columns of object or
    # categorical type, and of merging on a categorical key column or index. No
    # parameters. State: left_object, right_object, left_cat, right_cat,
    # left_cat_col, right_cat_col, left_cat_idx and right_cat_idx.
    def setup(self):
        # [FUNCTION DOC]
        # Builds two 10000-row object frames: left with an integer X (random below
        # 10) and a string Y from three words; right with X and a string Z from
        # three other words. Derives left_cat/right_cat (payload column converted to
        # categorical), left_cat_col/right_cat_col (X converted to categorical)
        # and left_cat_idx/right_cat_idx (those with X set as the index).
        self.left_object = DataFrame(
            {
                "X": np.random.choice(range(10), size=(10000,)),
                "Y": np.random.choice(["one", "two", "three"], size=(10000,)),
            }
        )

        self.right_object = DataFrame(
            {
                "X": np.random.choice(range(10), size=(10000,)),
                "Z": np.random.choice(["jjj", "kkk", "sss"], size=(10000,)),
            }
        )

        self.left_cat = self.left_object.assign(
            Y=self.left_object["Y"].astype("category")
        )
        self.right_cat = self.right_object.assign(
            Z=self.right_object["Z"].astype("category")
        )

        self.left_cat_col = self.left_object.astype({"X": "category"})
        self.right_cat_col = self.right_object.astype({"X": "category"})

        self.left_cat_idx = self.left_cat_col.set_index("X")
        self.right_cat_idx = self.right_cat_col.set_index("X")

    def time_merge_object(self):
        # [FUNCTION DOC]
        # Times merging the object frames on X.
        merge(self.left_object, self.right_object, on="X")

    def time_merge_cat(self):
        # [FUNCTION DOC]
        # Times merging the frames with categorical payload columns on X.
        merge(self.left_cat, self.right_cat, on="X")

    def time_merge_on_cat_col(self):
        # [FUNCTION DOC]
        # Times merging on the categorical column X.
        merge(self.left_cat_col, self.right_cat_col, on="X")

    def time_merge_on_cat_idx(self):
        # [FUNCTION DOC]
        # Times merging on X where X is a categorical index level.
        merge(self.left_cat_idx, self.right_cat_idx, on="X")


class MergeOrdered:
    # [CLASS DOC]
    # Benchmark of the ordered (fill-friendly) merge with grouping on the left. No
    # parameters. State: left and right.
    def setup(self):
        # [FUNCTION DOC]
        # Builds left (50000 rows: ten group labels each repeated 5000 times, an
        # even-number key column tiled ten times, random values) and right (10000
        # rows with an ascending integer key and random values).
        groups = Index([f"i-{i}" for i in range(10)], dtype=object).values
        self.left = DataFrame(
            {
                "group": groups.repeat(5000),
                "key": np.tile(np.arange(0, 10000, 2), 10),
                "lvalue": np.random.randn(50000),
            }
        )
        self.right = DataFrame(
            {"key": np.arange(10000), "rvalue": np.random.randn(10000)}
        )

    def time_merge_ordered(self):
        # [FUNCTION DOC]
        # Times the ordered merge on key, grouped on the left by group.
        merge_ordered(self.left, self.right, on="key", left_by="group")


class MergeAsof:
    # [CLASS DOC]
    # Benchmark of the as-of merge (nearest-key join) on integer time keys of
    # several widths, with and without exact-match grouping keys. Parametrized by
    # direction (backward, forward, nearest) and tolerance (none or 5). State:
    # df1a..df1f and df2a..df2f, pairs of left/right frames for each variant.
    params = [["backward", "forward", "nearest"], [None, 5]]
    param_names = ["direction", "tolerance"]

    def setup(self, direction, tolerance):
        # [FUNCTION DOC]
        # Builds a left frame of 200000 rows and a right frame of one million rows,
        # each with a random integer time column (range scaled by one twentieth of
        # the row count), a random uppercase letter key, a random small integer key
        # key2 (below 25) and a random value column, both sorted by time. Adds a
        # 32-bit time column and an unsigned 64-bit time column. Selects the column
        # subsets for each benchmark variant: a (time only), b (time with letter
        # key), c (time with integer key), d (32-bit time), e (time with both keys)
        # and f (unsigned 64-bit time), each on both sides. The direction and
        # tolerance parameters are unused here.
        one_count = 200000
        two_count = 1000000

        df1 = DataFrame(
            {
                "time": np.random.randint(0, one_count / 20, one_count),
                "key": np.random.choice(list(string.ascii_uppercase), one_count),
                "key2": np.random.randint(0, 25, one_count),
                "value1": np.random.randn(one_count),
            }
        )
        df2 = DataFrame(
            {
                "time": np.random.randint(0, two_count / 20, two_count),
                "key": np.random.choice(list(string.ascii_uppercase), two_count),
                "key2": np.random.randint(0, 25, two_count),
                "value2": np.random.randn(two_count),
            }
        )

        df1 = df1.sort_values("time")
        df2 = df2.sort_values("time")

        df1["time32"] = np.int32(df1.time)
        df2["time32"] = np.int32(df2.time)

        df1["timeu64"] = np.uint64(df1.time)
        df2["timeu64"] = np.uint64(df2.time)

        self.df1a = df1[["time", "value1"]]
        self.df2a = df2[["time", "value2"]]
        self.df1b = df1[["time", "key", "value1"]]
        self.df2b = df2[["time", "key", "value2"]]
        self.df1c = df1[["time", "key2", "value1"]]
        self.df2c = df2[["time", "key2", "value2"]]
        self.df1d = df1[["time32", "value1"]]
        self.df2d = df2[["time32", "value2"]]
        self.df1e = df1[["time", "key", "key2", "value1"]]
        self.df2e = df2[["time", "key", "key2", "value2"]]
        self.df1f = df1[["timeu64", "value1"]]
        self.df2f = df2[["timeu64", "value2"]]

    def time_on_int(self, direction, tolerance):
        # [FUNCTION DOC]
        # Times an as-of merge on the default-width integer time column.
        merge_asof(
            self.df1a, self.df2a, on="time", direction=direction, tolerance=tolerance
        )

    def time_on_int32(self, direction, tolerance):
        # [FUNCTION DOC]
        # Times an as-of merge on the 32-bit integer time column.
        merge_asof(
            self.df1d, self.df2d, on="time32", direction=direction, tolerance=tolerance
        )

    def time_on_uint64(self, direction, tolerance):
        # [FUNCTION DOC]
        # Times an as-of merge on the unsigned 64-bit time column.
        merge_asof(
            self.df1f, self.df2f, on="timeu64", direction=direction, tolerance=tolerance
        )

    def time_by_object(self, direction, tolerance):
        # [FUNCTION DOC]
        # Times an as-of merge on time, matching exactly on the string key.
        merge_asof(
            self.df1b,
            self.df2b,
            on="time",
            by="key",
            direction=direction,
            tolerance=tolerance,
        )

    def time_by_int(self, direction, tolerance):
        # [FUNCTION DOC]
        # Times an as-of merge on time, matching exactly on the integer key.
        merge_asof(
            self.df1c,
            self.df2c,
            on="time",
            by="key2",
            direction=direction,
            tolerance=tolerance,
        )

    def time_multiby(self, direction, tolerance):
        # [FUNCTION DOC]
        # Times an as-of merge on time, matching exactly on both the string and
        # integer keys.
        merge_asof(
            self.df1e,
            self.df2e,
            on="time",
            by=["key", "key2"],
            direction=direction,
            tolerance=tolerance,
        )


class MergeMultiIndex:
    # [CLASS DOC]
    # Benchmark of merging two frames on their two-level multi-indexes (for
    # several level dtype pairs and join types) with half-overlapping ranges.
    # Parametrized by the level dtype pair and how. State: df1 and df2.
    params = [
        [
            ("int64", "int64"),
            ("datetime64[ns]", "int64"),
            ("Int64", "Int64"),
        ],
        ["left", "right", "inner", "outer"],
    ]
    param_names = ["dtypes", "how"]

    def setup(self, dtypes, how):
        # [FUNCTION DOC]
        # Builds two multi-indexes of 100000 entries whose levels are arrays of the
        # two dtypes: the first from 0 up, the second offset by 50000 (so half
        # overlap). df1 and df2 are single-column frames (constant one and two) on
        # them. The how parameter is unused here.
        n = 100_000
        offset = 50_000
        mi1 = MultiIndex.from_arrays(
            [
                array(np.arange(n), dtype=dtypes[0]),
                array(np.arange(n), dtype=dtypes[1]),
            ]
        )
        mi2 = MultiIndex.from_arrays(
            [
                array(np.arange(offset, n + offset), dtype=dtypes[0]),
                array(np.arange(offset, n + offset), dtype=dtypes[1]),
            ]
        )
        self.df1 = DataFrame({"col1": 1}, index=mi1)
        self.df2 = DataFrame({"col2": 2}, index=mi2)

    def time_merge_sorted_multiindex(self, dtypes, how):
        # [FUNCTION DOC]
        # Times merging copies of df1 and df2 on both indexes with the join type.
        # Copies are made inside the timing to avoid the cached values of the
        # multi-index persisting between runs.
        # copy to avoid MultiIndex._values caching
        df1 = self.df1.copy()
        df2 = self.df2.copy()
        merge(df1, df2, how=how, left_index=True, right_index=True)


class Align:
    # [CLASS DOC]
    # Benchmark of aligning two large series with partially overlapping sorted
    # integer-timestamp indexes. No parameters. State: ts1 and ts2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 1,000,000 candidate stamps from the current time (integer view) plus
        # multiples of ten million up to ten trillion, picks 500000 distinct sorted
        # stamps twice independently as the two indexes, and creates random series
        # ts1 and ts2 on them.
        size = 5 * 10**5
        rng = np.arange(0, 10**13, 10**7)
        stamps = np.datetime64("now").view("i8") + rng
        idx1 = np.sort(np.random.choice(stamps, size, replace=False))
        idx2 = np.sort(np.random.choice(stamps, size, replace=False))
        self.ts1 = Series(np.random.randn(size), idx1)
        self.ts2 = Series(np.random.randn(size), idx2)

    def time_series_align_int64_index(self):
        # [FUNCTION DOC]
        # Times adding the two series, which aligns their indexes first.
        self.ts1 + self.ts2

    def time_series_align_left_monotonic(self):
        # [FUNCTION DOC]
        # Times aligning ts1 to ts2 with a left join.
        self.ts1.align(self.ts2, join="left")


from .pandas_vb_common import setup  # noqa: F401 isort:skip
