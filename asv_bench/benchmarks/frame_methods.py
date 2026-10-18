# [MODULE DOC]
# Benchmarks for DataFrame methods: dtype conversion, clipping, selection of
# numeric columns, reindex, rename, iteration, text/HTML/dict/numpy/record
# conversion, repr, masking and null detection, fill/drop of missing values,
# counting, apply, equality, interpolation, shift, nunique, duplicates, label
# lookup, sorting, quantile, rank, dtype counting, n-largest/smallest, describe,
# memory usage, rounding, where, valid index lookup and update. Each public
# benchmark class (AsType, Clip, GetNumericData, Reindex, Rename, Iteration,
# ToString, ToHTML, ToDict, ToNumpy, ToRecords, Repr, MaskBool, Isnull, Fillna,
# Dropna, Isna, Count, Apply, Dtypes, Equals, Interpolate, Shift, Nunique,
# SeriesNuniqueWithNan, Duplicated, XS, SortValues, SortMultiKey, Quantile, Rank,
# GetDtypeCounts, NSort, Describe, MemoryUsage, Round, Where, FindValidIndex,
# Update) builds data in setup using random numbers and times one method per
# timing function. No import-time side effects other than importing; the final
# line re-exports the common asv setup hook, which seeds the global random state
# (so most data is reproducible; the Update class uses its own unseeded random
# generator).
import string
import warnings

import numpy as np

from pandas import (
    NA,
    DataFrame,
    Index,
    MultiIndex,
    NaT,
    Series,
    date_range,
    isnull,
    period_range,
    timedelta_range,
)


class AsType:
    # [CLASS DOC]
    # Benchmark of frame dtype conversion between numpy dtypes and numpy-backed or
    # arrow-backed nullable dtypes, with and without copying. Parametrized by a
    # (source dtype, target dtype) pair and a copy flag. State: df.
    params = [
        [
            # from_dtype == to_dtype
            ("Float64", "Float64"),
            ("float64[pyarrow]", "float64[pyarrow]"),
            # from non-EA to EA
            ("float64", "Float64"),
            ("float64", "float64[pyarrow]"),
            # from EA to non-EA
            ("Float64", "float64"),
            ("float64[pyarrow]", "float64"),
            # from EA to EA
            ("Int64", "Float64"),
            ("int64[pyarrow]", "float64[pyarrow]"),
        ],
        [False, True],
    ]
    param_names = ["from_to_dtypes", "copy"]

    def setup(self, from_to_dtypes, copy):
        # [FUNCTION DOC]
        # Builds a 100 by 100 frame with the source dtype (first tuple element).
        # Float source names use random normal data; integer source names use
        # random integers below 1000. Raises a not-implemented error for any other
        # source dtype. Stores the frame as df. The copy flag is unused here.
        from_dtype = from_to_dtypes[0]
        if from_dtype in ("float64", "Float64", "float64[pyarrow]"):
            data = np.random.randn(100, 100)
        elif from_dtype in ("int64", "Int64", "int64[pyarrow]"):
            data = np.random.randint(0, 1000, (100, 100))
        else:
            raise NotImplementedError
        self.df = DataFrame(data, dtype=from_dtype)

    def time_astype(self, from_to_dtypes, copy):
        # [FUNCTION DOC]
        # Times converting the frame to the target dtype (second tuple element)
        # with the copy flag as given.
        self.df.astype(from_to_dtypes[1], copy=copy)


class Clip:
    # [CLASS DOC]
    # Benchmark of clipping all values of a frame to a fixed range. Parametrized
    # by dtype (numpy float, nullable float, arrow float). State: df.
    params = [
        ["float64", "Float64", "float64[pyarrow]"],
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a 100000 by 10 frame of random normal numbers in the given dtype
        # and stores it as df.
        data = np.random.randn(100_000, 10)
        df = DataFrame(data, dtype=dtype)
        self.df = df

    def time_clip(self, dtype):
        # [FUNCTION DOC]
        # Times clipping all values to the range from minus one to plus one.
        self.df.clip(-1.0, 1.0)


class GetNumericData:
    # [CLASS DOC]
    # Benchmark of the internal selection of numeric columns from a frame that
    # also holds two string columns. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 10000 by 25 random frame, adds two constant string columns (foo
        # and bar) and consolidates the internal blocks. Stores as df.
        self.df = DataFrame(np.random.randn(10000, 25))
        self.df["foo"] = "bar"
        self.df["bar"] = "baz"
        self.df = self.df._consolidate()

    def time_frame_get_numeric_data(self):
        # [FUNCTION DOC]
        # Times the internal call that returns only numeric columns.
        self.df._get_numeric_data()


class Reindex:
    # [CLASS DOC]
    # Benchmark of reindexing a wide frame along rows, columns and both, and of a
    # reindex that upcasts a frame of mixed integer and boolean blocks. No
    # parameters. State: df, idx, idx_cols and df2.
    def setup(self):
        # [FUNCTION DOC]
        # With N equal to 1000: builds df (10000 by 1000 random normal), idx (an
        # integer range 4000 to 6999 used as new labels, partly outside the
        # existing range), idx_cols (1000 random column labels below N, with
        # repeats), and df2, a 1000-column frame in which each column is randomly
        # chosen among boolean, 16-bit, 32-bit and 64-bit integer data of length
        # 1000.
        N = 10**3
        self.df = DataFrame(np.random.randn(N * 10, N))
        self.idx = np.arange(4 * N, 7 * N)
        self.idx_cols = np.random.randint(0, N, N)
        self.df2 = DataFrame(
            {
                c: {
                    0: np.random.randint(0, 2, N).astype(np.bool_),
                    1: np.random.randint(0, N, N).astype(np.int16),
                    2: np.random.randint(0, N, N).astype(np.int32),
                    3: np.random.randint(0, N, N).astype(np.int64),
                }[np.random.randint(0, 4)]
                for c in range(N)
            }
        )

    def time_reindex_axis0(self):
        # [FUNCTION DOC]
        # Times reindexing the rows to idx.
        self.df.reindex(self.idx)

    def time_reindex_axis1(self):
        # [FUNCTION DOC]
        # Times reindexing the columns to the random column labels.
        self.df.reindex(columns=self.idx_cols)

    def time_reindex_axis1_missing(self):
        # [FUNCTION DOC]
        # Times reindexing the columns to idx, whose labels are mostly missing
        # from the columns.
        self.df.reindex(columns=self.idx)

    def time_reindex_both_axes(self):
        # [FUNCTION DOC]
        # Times reindexing rows to idx and columns to the random labels together.
        self.df.reindex(index=self.idx, columns=self.idx_cols)

    def time_reindex_upcast(self):
        # [FUNCTION DOC]
        # Times reindexing the mixed-block frame to a random permutation of the
        # integers below 1200, introducing missing rows that force upcasting.
        self.df2.reindex(np.random.permutation(range(1200)))


class Rename:
    # [CLASS DOC]
    # Benchmark of renaming labels of a wide frame with an identity mapping.
    # No parameters. State: df, idx, dict_idx and df2.
    def setup(self):
        # [FUNCTION DOC]
        # With N equal to 1000: builds df (10000 by 1000 random normal), idx (the
        # integers 4000 to 6999), dict_idx (a mapping of each of those integers to
        # itself) and df2, a mixed-dtype frame built as in the reindex benchmark
        # (df2 is not used by any timing method).
        N = 10**3
        self.df = DataFrame(np.random.randn(N * 10, N))
        self.idx = np.arange(4 * N, 7 * N)
        self.dict_idx = {k: k for k in self.idx}
        self.df2 = DataFrame(
            {
                c: {
                    0: np.random.randint(0, 2, N).astype(np.bool_),
                    1: np.random.randint(0, N, N).astype(np.int16),
                    2: np.random.randint(0, N, N).astype(np.int32),
                    3: np.random.randint(0, N, N).astype(np.int64),
                }[np.random.randint(0, 4)]
                for c in range(N)
            }
        )

    def time_rename_single(self):
        # [FUNCTION DOC]
        # Times renaming a single row label (zero to zero).
        self.df.rename({0: 0})

    def time_rename_axis0(self):
        # [FUNCTION DOC]
        # Times renaming the row labels with the identity mapping.
        self.df.rename(self.dict_idx)

    def time_rename_axis1(self):
        # [FUNCTION DOC]
        # Times renaming the column labels with the identity mapping.
        self.df.rename(columns=self.dict_idx)

    def time_rename_both_axes(self):
        # [FUNCTION DOC]
        # Times renaming both row and column labels with the identity mapping.
        self.df.rename(index=self.dict_idx, columns=self.dict_idx)

    def time_dict_rename_both_axes(self):
        # [FUNCTION DOC]
        # Times renaming both row and column labels with the identity mapping.
        # KNOWN ISSUE: this body is identical to the previous method that renames
        # both axes, so the two benchmarks measure the same operation.
        self.df.rename(index=self.dict_idx, columns=self.dict_idx)


class Iteration:
    # [CLASS DOC]
    # Benchmark of the different frame iteration methods (columns, tuples, raw
    # tuples, rows), measuring time and memory. No parameters. The timeout
    # attribute (120 seconds) is raised because the memory benchmarks for tuple
    # iteration are slow. State: df (10000 by 1000), df2 (50000 by 10), df3
    # (1000 by 5000 with string labels) and df4 (one million by 10), all random
    # normal.
    # mem_itertuples_* benchmarks are slow
    timeout = 120

    def setup(self):
        # [FUNCTION DOC]
        # Builds the four random normal frames described in the class note. df2 is
        # not used by any timing method.
        N = 1000
        self.df = DataFrame(np.random.randn(N * 10, N))
        self.df2 = DataFrame(np.random.randn(N * 50, 10))
        self.df3 = DataFrame(
            np.random.randn(N, 5 * N), columns=["C" + str(c) for c in range(N * 5)]
        )
        self.df4 = DataFrame(np.random.randn(N * 1000, 10))

    def time_items(self):
        # [FUNCTION DOC]
        # Times looping over all (label, column) pairs of the first frame without
        # using them, to monitor that no copy is made.
        # (monitor no-copying behaviour)
        for name, col in self.df.items():
            pass

    def time_iteritems_indexing(self):
        # [FUNCTION DOC]
        # Times looping over the column labels of the wide frame and selecting
        # each column by label.
        for col in self.df3:
            self.df3[col]

    def time_itertuples_start(self):
        # [FUNCTION DOC]
        # Times creating (not consuming) the named-tuple iterator of the big frame.
        self.df4.itertuples()

    def time_itertuples_read_first(self):
        # [FUNCTION DOC]
        # Times creating the named-tuple iterator and reading its first row.
        next(self.df4.itertuples())

    def time_itertuples(self):
        # [FUNCTION DOC]
        # Times consuming the whole named-tuple iterator one row at a time.
        for row in self.df4.itertuples():
            pass

    def time_itertuples_to_list(self):
        # [FUNCTION DOC]
        # Times materializing all named tuples into a list.
        list(self.df4.itertuples())

    def mem_itertuples_start(self):
        # [FUNCTION DOC]
        # Memory benchmark: returns the freshly created named-tuple iterator so
        # asv measures the size of the returned object.
        return self.df4.itertuples()

    def peakmem_itertuples_start(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of creating the named-tuple iterator.
        self.df4.itertuples()

    def mem_itertuples_read_first(self):
        # [FUNCTION DOC]
        # Memory benchmark: returns the first named tuple of the iterator.
        return next(self.df4.itertuples())

    def peakmem_itertuples(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of consuming the named-tuple iterator row by row.
        for row in self.df4.itertuples():
            pass

    def mem_itertuples_to_list(self):
        # [FUNCTION DOC]
        # Memory benchmark: returns the list of all named tuples.
        return list(self.df4.itertuples())

    def peakmem_itertuples_to_list(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of materializing all named tuples into a list.
        list(self.df4.itertuples())

    def time_itertuples_raw_start(self):
        # [FUNCTION DOC]
        # Times creating the iterator of plain tuples without the index.
        self.df4.itertuples(index=False, name=None)

    def time_itertuples_raw_read_first(self):
        # [FUNCTION DOC]
        # Times creating the plain-tuple iterator and reading its first row.
        next(self.df4.itertuples(index=False, name=None))

    def time_itertuples_raw_tuples(self):
        # [FUNCTION DOC]
        # Times consuming the plain-tuple iterator one row at a time.
        for row in self.df4.itertuples(index=False, name=None):
            pass

    def time_itertuples_raw_tuples_to_list(self):
        # [FUNCTION DOC]
        # Times materializing all plain tuples into a list.
        list(self.df4.itertuples(index=False, name=None))

    def mem_itertuples_raw_start(self):
        # [FUNCTION DOC]
        # Memory benchmark: returns the freshly created plain-tuple iterator.
        return self.df4.itertuples(index=False, name=None)

    def peakmem_itertuples_raw_start(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of creating the plain-tuple iterator.
        self.df4.itertuples(index=False, name=None)

    def peakmem_itertuples_raw_read_first(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of creating the plain-tuple iterator and reading
        # its first row.
        next(self.df4.itertuples(index=False, name=None))

    def peakmem_itertuples_raw(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of consuming the plain-tuple iterator row by row.
        for row in self.df4.itertuples(index=False, name=None):
            pass

    def mem_itertuples_raw_to_list(self):
        # [FUNCTION DOC]
        # Memory benchmark: returns the list of all plain tuples.
        return list(self.df4.itertuples(index=False, name=None))

    def peakmem_itertuples_raw_to_list(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of materializing all plain tuples into a list.
        list(self.df4.itertuples(index=False, name=None))

    def time_iterrows(self):
        # [FUNCTION DOC]
        # Times consuming the (label, row series) iterator of the first frame.
        for row in self.df.iterrows():
            pass


class ToString:
    # [CLASS DOC]
    # Benchmark of rendering a small float frame as a text table. No parameters.
    # State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 100 by 10 random normal frame as df.
        self.df = DataFrame(np.random.randn(100, 10))

    def time_to_string_floats(self):
        # [FUNCTION DOC]
        # Times converting the frame to a text table.
        self.df.to_string()


class ToHTML:
    # [CLASS DOC]
    # Benchmark of rendering a mixed frame (period column, integer column and
    # float columns) as an HTML table. No parameters. State: df2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 500 by 10 random normal frame, replaces column zero with an
        # annual period range and column one with the integers below 500. Stores
        # the frame as df2.
        nrows = 500
        self.df2 = DataFrame(np.random.randn(nrows, 10))
        self.df2[0] = period_range("2000", periods=nrows)
        self.df2[1] = range(nrows)

    def time_to_html_mixed(self):
        # [FUNCTION DOC]
        # Times converting the mixed frame to an HTML table string.
        self.df2.to_html()


class ToDict:
    # [CLASS DOC]
    # Benchmark of converting a frame to a dictionary structure for each
    # orientation, using integer data and timedelta data. Parametrized by
    # orientation name. State: int_df and datetimelike_df.
    params = [["dict", "list", "series", "split", "records", "index"]]
    param_names = ["orient"]

    def setup(self, orient):
        # [FUNCTION DOC]
        # Builds a 10000 by 4 frame of random integers below 1000 as int_df and the
        # same data converted to nanosecond timedelta dtype as datetimelike_df.
        # The orient parameter is unused here.
        data = np.random.randint(0, 1000, size=(10000, 4))
        self.int_df = DataFrame(data)
        self.datetimelike_df = self.int_df.astype("timedelta64[ns]")

    def time_to_dict_ints(self, orient):
        # [FUNCTION DOC]
        # Times converting the integer frame to a dictionary with the orientation.
        self.int_df.to_dict(orient=orient)

    def time_to_dict_datetimelike(self, orient):
        # [FUNCTION DOC]
        # Times converting the timedelta frame to a dictionary with the
        # orientation.
        self.datetimelike_df.to_dict(orient=orient)


class ToNumpy:
    # [CLASS DOC]
    # Benchmark of converting float and mixed-dtype frames, tall (10000 by 10) and
    # wide (10 by 10000), to numpy arrays using both the to-numpy method and the
    # values attribute. No parameters. State: df_tall, df_wide, df_mixed_tall,
    # df_mixed_wide.
    def setup(self):
        # [FUNCTION DOC]
        # Builds random normal tall and wide frames. Each mixed copy gets a constant
        # string column foo, column zero replaced by an annual period range and
        # column one replaced by the integers from zero.
        N = 10000
        M = 10
        self.df_tall = DataFrame(np.random.randn(N, M))
        self.df_wide = DataFrame(np.random.randn(M, N))
        self.df_mixed_tall = self.df_tall.copy()
        self.df_mixed_tall["foo"] = "bar"
        self.df_mixed_tall[0] = period_range("2000", periods=N)
        self.df_mixed_tall[1] = range(N)
        self.df_mixed_wide = self.df_wide.copy()
        self.df_mixed_wide["foo"] = "bar"
        self.df_mixed_wide[0] = period_range("2000", periods=M)
        self.df_mixed_wide[1] = range(M)

    def time_to_numpy_tall(self):
        # [FUNCTION DOC]
        # Times the to-numpy conversion of the tall float frame.
        self.df_tall.to_numpy()

    def time_to_numpy_wide(self):
        # [FUNCTION DOC]
        # Times the to-numpy conversion of the wide float frame.
        self.df_wide.to_numpy()

    def time_to_numpy_mixed_tall(self):
        # [FUNCTION DOC]
        # Times the to-numpy conversion of the tall mixed frame.
        self.df_mixed_tall.to_numpy()

    def time_to_numpy_mixed_wide(self):
        # [FUNCTION DOC]
        # Times the to-numpy conversion of the wide mixed frame.
        self.df_mixed_wide.to_numpy()

    def time_values_tall(self):
        # [FUNCTION DOC]
        # Times reading the values attribute of the tall float frame.
        self.df_tall.values

    def time_values_wide(self):
        # [FUNCTION DOC]
        # Times reading the values attribute of the wide float frame.
        self.df_wide.values

    def time_values_mixed_tall(self):
        # [FUNCTION DOC]
        # Times reading the values attribute of the tall mixed frame.
        self.df_mixed_tall.values

    def time_values_mixed_wide(self):
        # [FUNCTION DOC]
        # Times reading the values attribute of the wide mixed frame.
        self.df_mixed_wide.values


class ToRecords:
    # [CLASS DOC]
    # Benchmark of converting a frame to a numpy record array, with a plain
    # index and with a two-level multi-index. No parameters. State: df and df_mi.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 100000 by 2 random normal data, a multi-index from an integer
        # range and a millisecond datetime range, and frames df (default index) and
        # df_mi (multi-index).
        N = 100_000
        data = np.random.randn(N, 2)
        mi = MultiIndex.from_arrays(
            [
                np.arange(N),
                date_range("1970-01-01", periods=N, freq="ms"),
            ]
        )
        self.df = DataFrame(data)
        self.df_mi = DataFrame(data, index=mi)

    def time_to_records(self):
        # [FUNCTION DOC]
        # Times converting the plain-index frame to records, including the index.
        self.df.to_records(index=True)

    def time_to_records_multiindex(self):
        # [FUNCTION DOC]
        # Times converting the multi-indexed frame to records, including the
        # index levels.
        self.df_mi.to_records(index=True)


class Repr:
    # [CLASS DOC]
    # Benchmark of text and HTML representations of large frames that get
    # truncated for display. No parameters. State: df3 (multi-index rows), df4
    # (float index), df_tall and df_wide.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 10000 rows by 10 columns of random normal data. df3 uses a
        # three-level multi-index built from three arrays that are each 100
        # random values tiled; df4 uses 10000 random float labels as the index;
        # df_tall is a 10000 by 10 random frame and df_wide is its 10 by 10000
        # counterpart with new random data.
        nrows = 10000
        data = np.random.randn(nrows, 10)
        arrays = np.tile(np.random.randn(3, nrows // 100), 100)
        idx = MultiIndex.from_arrays(arrays)
        self.df3 = DataFrame(data, index=idx)
        self.df4 = DataFrame(data, index=np.random.randn(nrows))
        self.df_tall = DataFrame(np.random.randn(nrows, 10))
        self.df_wide = DataFrame(np.random.randn(10, nrows))

    def time_html_repr_trunc_mi(self):
        # [FUNCTION DOC]
        # Times the HTML representation of the multi-indexed frame.
        self.df3._repr_html_()

    def time_html_repr_trunc_si(self):
        # [FUNCTION DOC]
        # Times the HTML representation of the single-indexed frame.
        self.df4._repr_html_()

    def time_repr_tall(self):
        # [FUNCTION DOC]
        # Times the text representation of the tall frame.
        repr(self.df_tall)

    def time_frame_repr_wide(self):
        # [FUNCTION DOC]
        # Times the text representation of the wide frame.
        repr(self.df_wide)


class MaskBool:
    # [CLASS DOC]
    # Benchmark of masking a boolean frame, versus the same frame as floats,
    # using a boolean mask. No parameters. State: bools and mask.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000 by 500 random normal frame, blanks all values that are not
        # positive to missing, then stores bools (true where the value is positive)
        # and mask (true where the value is missing, i.e. the complement).
        data = np.random.randn(1000, 500)
        df = DataFrame(data)
        df = df.where(df > 0)
        self.bools = df > 0
        self.mask = isnull(df)

    def time_frame_mask_bools(self):
        # [FUNCTION DOC]
        # Times masking the boolean frame with the null mask.
        self.bools.mask(self.mask)

    def time_frame_mask_floats(self):
        # [FUNCTION DOC]
        # Times masking the boolean frame converted to floats with the null mask.
        self.bools.astype(float).mask(self.mask)


class Isnull:
    # [CLASS DOC]
    # Benchmark of the top-level null detection function over a float frame
    # without nulls, a float frame with nulls, a string frame and an object frame
    # of mixed null kinds. No parameters. State: df_no_null, df, df_strings and
    # df_obj, all 1000 by 1000.
    def setup(self):
        # [FUNCTION DOC]
        # Builds four 1000 by 1000 frames: random normal floats (no nulls); a random
        # choice of missing and the float one; a random choice of ASCII letters
        # and whitespace characters; and an object frame drawn from a mix of
        # missing markers (NaT, NaN, none, datetime and timedelta not-a-time) and
        # ordinary values (integers, a float and strings).
        N = 10**3
        self.df_no_null = DataFrame(np.random.randn(N, N))

        sample = np.array([np.nan, 1.0])
        data = np.random.choice(sample, (N, N))
        self.df = DataFrame(data)

        sample = np.array(list(string.ascii_letters + string.whitespace))
        data = np.random.choice(sample, (N, N))
        self.df_strings = DataFrame(data)

        sample = np.array(
            [
                NaT,
                np.nan,
                None,
                np.datetime64("NaT"),
                np.timedelta64("NaT", "ns"),
                0,
                1,
                2.0,
                "",
                "abcd",
            ]
        )
        data = np.random.choice(sample, (N, N))
        self.df_obj = DataFrame(data)

    def time_isnull_floats_no_null(self):
        # [FUNCTION DOC]
        # Times null detection on the frame without nulls.
        isnull(self.df_no_null)

    def time_isnull(self):
        # [FUNCTION DOC]
        # Times null detection on the float frame with missing values.
        isnull(self.df)

    def time_isnull_strngs(self):
        # [FUNCTION DOC]
        # Times null detection on the string frame.
        isnull(self.df_strings)

    def time_isnull_obj(self):
        # [FUNCTION DOC]
        # Times null detection on the mixed object frame.
        isnull(self.df_obj)


class Fillna:
    # [CLASS DOC]
    # Benchmark of filling missing values by a column-to-value mapping, forward
    # fill and backward fill, in place or not. Parametrized by inplace flag and
    # dtype (numeric, object, nullable, datetime, timezone-aware datetime,
    # timedelta). State: df and fill_values.
    params = (
        [True, False],
        [
            "float64",
            "float32",
            "object",
            "Int64",
            "Float64",
            "datetime64[ns]",
            "datetime64[ns, tz]",
            "timedelta64[ns]",
        ],
    )
    param_names = ["inplace", "dtype"]

    def setup(self, inplace, dtype):
        # [FUNCTION DOC]
        # Builds a frame of 10000 rows and 100 columns in which every second row is
        # missing. For datetime-like dtypes each column is the same hourly (or
        # daily for timedelta) range and the even rows are set to none. For other
        # dtypes random normal data has even rows set to NaN; for the nullable
        # integer dtype the values are rounded, converted to object and the even
        # rows set to the missing marker; the frame is built with the dtype.
        # fill_values is the mapping of column to value from the first valid row.
        # The inplace flag is unused here.
        N, M = 10000, 100
        if dtype in ("datetime64[ns]", "datetime64[ns, tz]", "timedelta64[ns]"):
            data = {
                "datetime64[ns]": date_range("2011-01-01", freq="h", periods=N),
                "datetime64[ns, tz]": date_range(
                    "2011-01-01", freq="h", periods=N, tz="Asia/Tokyo"
                ),
                "timedelta64[ns]": timedelta_range(start="1 day", periods=N, freq="1D"),
            }
            self.df = DataFrame({f"col_{i}": data[dtype] for i in range(M)})
            self.df[::2] = None
        else:
            values = np.random.randn(N, M)
            values[::2] = np.nan
            if dtype == "Int64":
                values = values.round()
                values = values.astype(object)
                values[::2] = NA
            self.df = DataFrame(values, dtype=dtype)
        self.fill_values = self.df.iloc[self.df.first_valid_index()].to_dict()

    def time_fillna(self, inplace, dtype):
        # [FUNCTION DOC]
        # Times filling missing values with the per-column mapping, optionally in
        # place (mutating the frame).
        self.df.fillna(value=self.fill_values, inplace=inplace)

    def time_ffill(self, inplace, dtype):
        # [FUNCTION DOC]
        # Times forward fill, optionally in place (mutating the frame).
        self.df.ffill(inplace=inplace)

    def time_bfill(self, inplace, dtype):
        # [FUNCTION DOC]
        # Times backward fill, optionally in place (mutating the frame).
        self.df.bfill(inplace=inplace)


class Dropna:
    # [CLASS DOC]
    # Benchmark of dropping rows or columns that contain missing values, for a
    # float frame and for the same frame with an added string column.
    # Parametrized by how (all, any) and axis (0, 1). State: df and df_mixed.
    params = (["all", "any"], [0, 1])
    param_names = ["how", "axis"]

    def setup(self, how, axis):
        # [FUNCTION DOC]
        # Builds a 10000 by 1000 random normal frame and sets three regions to
        # missing: rows 50 to 999 in columns 20 to 49, all of rows 2000 to 2999,
        # and all rows in columns 60 to 69. df_mixed is a copy with an added
        # constant string column named foo.
        self.df = DataFrame(np.random.randn(10000, 1000))
        self.df.iloc[50:1000, 20:50] = np.nan
        self.df.iloc[2000:3000] = np.nan
        self.df.iloc[:, 60:70] = np.nan
        self.df_mixed = self.df.copy()
        self.df_mixed["foo"] = "bar"

    def time_dropna(self, how, axis):
        # [FUNCTION DOC]
        # Times dropping missing values from the float frame.
        self.df.dropna(how=how, axis=axis)

    def time_dropna_axis_mixed_dtypes(self, how, axis):
        # [FUNCTION DOC]
        # Times dropping missing values from the mixed-dtype frame.
        self.df_mixed.dropna(how=how, axis=axis)


class Isna:
    # [CLASS DOC]
    # Benchmark of the frame null-detection method for numpy float, nullable
    # float and arrow float dtypes. Parametrized by dtype. State: df.
    params = ["float64", "Float64", "float64[pyarrow]"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds 10000 by 1000 random normal data, sets columns 600 to 799 to
        # missing in all rows, and attempts to set a block of rows 800 to 999 in
        # columns 4000 to 4999 to missing.
        # KNOWN ISSUE: the data has only 1000 columns, so the column slice from
        # 4000 to 5000 selects nothing and the intended partially missing columns
        # are never created.
        data = np.random.randn(10000, 1000)
        # all-na columns
        data[:, 600:800] = np.nan
        # partial-na columns
        data[800:1000, 4000:5000] = np.nan
        self.df = DataFrame(data, dtype=dtype)

    def time_isna(self, dtype):
        # [FUNCTION DOC]
        # Times the frame null-detection method.
        self.df.isna()


class Count:
    # [CLASS DOC]
    # Benchmark of counting non-missing values per row or per column for a float
    # frame and a mixed frame. Parametrized by axis. State: df and df_mixed.
    params = [0, 1]
    param_names = ["axis"]

    def setup(self, axis):
        # [FUNCTION DOC]
        # Builds the same 10000 by 1000 frame with the same three missing regions
        # as the drop benchmark, and df_mixed with an added constant string column
        # foo. The axis parameter is unused here.
        self.df = DataFrame(np.random.randn(10000, 1000))
        self.df.iloc[50:1000, 20:50] = np.nan
        self.df.iloc[2000:3000] = np.nan
        self.df.iloc[:, 60:70] = np.nan
        self.df_mixed = self.df.copy()
        self.df_mixed["foo"] = "bar"

    def time_count(self, axis):
        # [FUNCTION DOC]
        # Times counting non-missing values in the float frame along the axis.
        self.df.count(axis=axis)

    def time_count_mixed_dtypes(self, axis):
        # [FUNCTION DOC]
        # Times counting non-missing values in the mixed frame along the axis.
        self.df_mixed.count(axis=axis)


class Apply:
    # [CLASS DOC]
    # Benchmark of the frame apply method with different callables and axes. No
    # parameters. State: df, s, df2 and df3.
    def setup(self):
        # [FUNCTION DOC]
        # Builds df (1000 by 100 random normal), s (the floats 0 to 1027 as a
        # series), df2 (a frame with 1028 columns that all equal s) and df3 (1000
        # by 3 random normal with columns A, B, C).
        self.df = DataFrame(np.random.randn(1000, 100))

        self.s = Series(np.arange(1028.0))
        self.df2 = DataFrame(dict.fromkeys(range(1028), self.s))
        self.df3 = DataFrame(np.random.randn(1000, 3), columns=list("ABC"))

    def time_apply_user_func(self):
        # [FUNCTION DOC]
        # Times applying, to each column of df2, a function that returns the
        # correlation coefficient between that column and s.
        self.df2.apply(lambda x: np.corrcoef(x, self.s)[(0, 1)])

    def time_apply_axis_1(self):
        # [FUNCTION DOC]
        # Times applying a function that adds one to each row.
        self.df.apply(lambda x: x + 1, axis=1)

    def time_apply_lambda_mean(self):
        # [FUNCTION DOC]
        # Times applying a function that takes the mean of each column.
        self.df.apply(lambda x: x.mean())

    def time_apply_str_mean(self):
        # [FUNCTION DOC]
        # Times applying the string name of the mean reduction, which lets the
        # frame dispatch to its own fast method.
        self.df.apply("mean")

    def time_apply_pass_thru(self):
        # [FUNCTION DOC]
        # Times applying an identity function to each column.
        self.df.apply(lambda x: x)

    def time_apply_ref_by_name(self):
        # [FUNCTION DOC]
        # Times applying, to each row of df3, a function that adds the values of
        # columns A and B looked up by label.
        self.df3.apply(lambda x: x["A"] + x["B"], axis=1)


class Dtypes:
    # [CLASS DOC]
    # Benchmark of reading the dtypes attribute of a wide frame. No parameters.
    # State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000 by 1000 random normal frame as df.
        self.df = DataFrame(np.random.randn(1000, 1000))

    def time_frame_dtypes(self):
        # [FUNCTION DOC]
        # Times reading the dtypes of the frame.
        self.df.dtypes


class Equals:
    # [CLASS DOC]
    # Benchmark of the frame equality check for float, object and duplicate-column
    # frames, in equal and unequal cases. No parameters. State: float_df,
    # float_df_nan, object_df, object_df_nan, nonunique_cols and
    # nonunique_cols_nan.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000 by 1000 random normal frame and a copy whose last cell is
        # missing; a 1000 by 1000 frame of the constant string foo and a copy whose
        # last cell is missing; and a copy of the string frame with all columns
        # labelled A (duplicate labels), plus a copy of that with the last cell
        # missing.
        N = 10**3
        self.float_df = DataFrame(np.random.randn(N, N))
        self.float_df_nan = self.float_df.copy()
        self.float_df_nan.iloc[-1, -1] = np.nan

        self.object_df = DataFrame("foo", index=range(N), columns=range(N))
        self.object_df_nan = self.object_df.copy()
        self.object_df_nan.iloc[-1, -1] = np.nan

        self.nonunique_cols = self.object_df.copy()
        self.nonunique_cols.columns = ["A"] * len(self.nonunique_cols.columns)
        self.nonunique_cols_nan = self.nonunique_cols.copy()
        self.nonunique_cols_nan.iloc[-1, -1] = np.nan

    def time_frame_float_equal(self):
        # [FUNCTION DOC]
        # Times comparing the float frame with itself.
        self.float_df.equals(self.float_df)

    def time_frame_float_unequal(self):
        # [FUNCTION DOC]
        # Times comparing the float frame with the copy that differs in one cell.
        self.float_df.equals(self.float_df_nan)

    def time_frame_nonunique_equal(self):
        # [FUNCTION DOC]
        # Times comparing the duplicate-column frame with itself.
        self.nonunique_cols.equals(self.nonunique_cols)

    def time_frame_nonunique_unequal(self):
        # [FUNCTION DOC]
        # Times comparing the duplicate-column frame with its differing copy.
        self.nonunique_cols.equals(self.nonunique_cols_nan)

    def time_frame_object_equal(self):
        # [FUNCTION DOC]
        # Times comparing the object frame with itself.
        self.object_df.equals(self.object_df)

    def time_frame_object_unequal(self):
        # [FUNCTION DOC]
        # Times comparing the object frame with the copy that differs in one cell.
        self.object_df.equals(self.object_df_nan)


class Interpolate:
    # [CLASS DOC]
    # Benchmark of interpolating missing values in a wide float frame (worst
    # case, every column has gaps) and in a four-column mixed frame. No
    # parameters. State: df and df2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds df from 10000 by 100 random normal data with every second row set
        # to missing, and df2 with columns A (integer range as float after
        # blanking), B (random integers below 100), C and D (random normal) where
        # every fifth row starting at one in columns A and C is set to missing.
        N = 10000
        # this is the worst case, where every column has NaNs.
        arr = np.random.randn(N, 100)
        arr[::2] = np.nan

        self.df = DataFrame(arr)

        self.df2 = DataFrame(
            {
                "A": np.arange(0, N),
                "B": np.random.randint(0, 100, N),
                "C": np.random.randn(N),
                "D": np.random.randn(N),
            }
        )
        self.df2.loc[1::5, "A"] = np.nan
        self.df2.loc[1::5, "C"] = np.nan

    def time_interpolate(self):
        # [FUNCTION DOC]
        # Times interpolating the wide frame with default settings.
        self.df.interpolate()

    def time_interpolate_some_good(self):
        # [FUNCTION DOC]
        # Times interpolating the mixed frame with default settings.
        self.df2.interpolate()


class Shift:
    # [CLASS DOC]
    # Benchmark of shifting a frame by one position along an axis (speedup issue
    # 5609). Parametrized by axis. State: df.
    # frame shift speedup issue-5609
    params = [0, 1]
    param_names = ["axis"]

    def setup(self, axis):
        # [FUNCTION DOC]
        # Builds a 10000 by 500 uniform random frame as df. The axis parameter is
        # unused here.
        self.df = DataFrame(np.random.rand(10000, 500))

    def time_shift(self, axis):
        # [FUNCTION DOC]
        # Times shifting the frame by one along the axis.
        self.df.shift(1, axis=axis)


class Nunique:
    # [CLASS DOC]
    # Benchmark of counting distinct values per column of a wide frame. No
    # parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 10000 by 1000 random normal frame as df.
        self.df = DataFrame(np.random.randn(10000, 1000))

    def time_frame_nunique(self):
        # [FUNCTION DOC]
        # Times counting distinct values in each column.
        self.df.nunique()


class SeriesNuniqueWithNan:
    # [CLASS DOC]
    # Benchmark of counting distinct values in a float series that contains many
    # missing values. No parameters. State: ser.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a pattern of 100 missing values followed by the integers 0 to 99,
        # tiled 10000 times, as a float series ser.
        values = 100 * [np.nan] + list(range(100))
        self.ser = Series(np.tile(values, 10000), dtype=float)

    def time_series_nunique_nan(self):
        # [FUNCTION DOC]
        # Times counting distinct non-missing values of the series.
        self.ser.nunique()


class Duplicated:
    # [CLASS DOC]
    # Benchmark of marking duplicate rows in a tall three-column mixed frame, in a
    # wide string-valued frame, and restricted to one column subset. No
    # parameters. State: df and df2.
    def setup(self):
        # [FUNCTION DOC]
        # With n equal to 1048576 (two to the twentieth): df has an integer
        # column a (random between minus 256 and 255), a datetime column b (random
        # picks from n/64 per-second timestamps) and a float column c (random
        # picks from n/64 values rounded to two decimals). df2 is the transpose of
        # a 1000 by 100 random normal frame converted to text, giving 100 rows by
        # 1000 columns.
        n = 1 << 20
        t = date_range("2015-01-01", freq="s", periods=(n // 64))
        xs = np.random.randn(n // 64).round(2)
        self.df = DataFrame(
            {
                "a": np.random.randint(-1 << 8, 1 << 8, n),
                "b": np.random.choice(t, n),
                "c": np.random.choice(xs, n),
            }
        )
        self.df2 = DataFrame(np.random.randn(1000, 100).astype(str)).T

    def time_frame_duplicated(self):
        # [FUNCTION DOC]
        # Times marking duplicate rows of the tall frame.
        self.df.duplicated()

    def time_frame_duplicated_wide(self):
        # [FUNCTION DOC]
        # Times marking duplicate rows of the wide string frame.
        self.df2.duplicated()

    def time_frame_duplicated_subset(self):
        # [FUNCTION DOC]
        # Times marking duplicates considering only column a.
        self.df.duplicated(subset=["a"])


class XS:
    # [CLASS DOC]
    # Benchmark of selecting one cross-section by label from a square frame along
    # either axis. Parametrized by axis. State: N and df.
    params = [0, 1]
    param_names = ["axis"]

    def setup(self, axis):
        # [FUNCTION DOC]
        # Sets N to 10000 and builds a square random normal frame df of that
        # size. The axis parameter is unused here.
        self.N = 10**4
        self.df = DataFrame(np.random.randn(self.N, self.N))

    def time_frame_xs(self, axis):
        # [FUNCTION DOC]
        # Times selecting the label at the midpoint along the axis. The label is
        # computed with true division so it is a float (5000.0), which matches an
        # integer label by value.
        self.df.xs(self.N / 2, axis=axis)


class SortValues:
    # [CLASS DOC]
    # Benchmark of sorting a million-row two-column frame by one column.
    # Parametrized by ascending flag. State: df.
    params = [True, False]
    param_names = ["ascending"]

    def setup(self, ascending):
        # [FUNCTION DOC]
        # Builds a one million by 2 random normal frame with columns A and B as
        # df. The ascending flag is unused here.
        self.df = DataFrame(np.random.randn(1000000, 2), columns=list("AB"))

    def time_frame_sort_values(self, ascending):
        # [FUNCTION DOC]
        # Times sorting by column A in the requested direction.
        self.df.sort_values(by="A", ascending=ascending)


class SortMultiKey:
    # [CLASS DOC]
    # Benchmark of sorting by two string keys, by column values and by a
    # multi-index. Parametrized by whether the data is already sorted
    # (monotonic). State: df_by_columns and df_by_index.
    params = [True, False]
    param_names = ["monotonic"]

    def setup(self, monotonic):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows from 10000 distinct strings each repeated
        # ten times as two key columns (key1 and key2 are identical) and a random
        # normal value column. If monotonic is true the frame is first sorted by
        # the two keys. Stores it as df_by_columns and a copy indexed by both keys
        # as df_by_index.
        N = 10000
        K = 10
        df = DataFrame(
            {
                "key1": Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(
                    K
                ),
                "key2": Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(
                    K
                ),
                "value": np.random.randn(N * K),
            }
        )
        if monotonic:
            df = df.sort_values(["key1", "key2"])
        self.df_by_columns = df
        self.df_by_index = df.set_index(["key1", "key2"])

    def time_sort_values(self, monotonic):
        # [FUNCTION DOC]
        # Times sorting the column-keyed frame by the two key columns.
        self.df_by_columns.sort_values(by=["key1", "key2"])

    def time_sort_index(self, monotonic):
        # [FUNCTION DOC]
        # Times sorting the multi-indexed frame by its index.
        self.df_by_index.sort_index()


class Quantile:
    # [CLASS DOC]
    # Benchmark of computing two quantiles of a three-column frame along either
    # axis. Parametrized by axis. State: df.
    params = [0, 1]
    param_names = ["axis"]

    def setup(self, axis):
        # [FUNCTION DOC]
        # Builds a 1000 by 3 random normal frame with columns A, B, C as df. The
        # axis parameter is unused here.
        self.df = DataFrame(np.random.randn(1000, 3), columns=list("ABC"))

    def time_frame_quantile(self, axis):
        # [FUNCTION DOC]
        # Times computing the 0.1 and 0.5 quantiles along the axis.
        self.df.quantile([0.1, 0.5], axis=axis)


class Rank:
    # [CLASS DOC]
    # Benchmark of ranking frame columns for integer, unsigned integer, float
    # and object dtypes. Parametrized by dtype name. State: df.
    param_names = ["dtype"]
    params = [
        ["int", "uint", "float", "object"],
    ]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a 10000 by 10 frame of random normal numbers converted to the
        # dtype (so integer dtypes truncate the data) with integer column labels,
        # and stores it as df.
        self.df = DataFrame(
            np.random.randn(10000, 10).astype(dtype), columns=range(10), dtype=dtype
        )

    def time_rank(self, dtype):
        # [FUNCTION DOC]
        # Times ranking each column with default settings.
        self.df.rank()


class GetDtypeCounts:
    # [CLASS DOC]
    # Benchmark of counting columns per dtype and of the summary info method on a
    # frame with many columns (issue 2807). No parameters. State: df.
    # 2807
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 10 by 10000 random normal frame as df.
        self.df = DataFrame(np.random.randn(10, 10000))

    def time_frame_get_dtype_counts(self):
        # [FUNCTION DOC]
        # Times counting the frame columns per dtype using the dtype series, with
        # warnings recorded and discarded.
        with warnings.catch_warnings(record=True):
            self.df.dtypes.value_counts()

    def time_info(self):
        # [FUNCTION DOC]
        # Times printing the summary info of the frame to the default output.
        self.df.info()


class NSort:
    # [CLASS DOC]
    # Benchmark of selecting the 100 largest or smallest rows by one or two
    # columns. Parametrized by the keep policy for ties (first, last, all).
    # State: df.
    params = ["first", "last", "all"]
    param_names = ["keep"]

    def setup(self, keep):
        # [FUNCTION DOC]
        # Builds a 100000 by 3 random normal frame with columns A, B, C as df.
        # The keep parameter is unused here.
        self.df = DataFrame(np.random.randn(100000, 3), columns=list("ABC"))

    def time_nlargest_one_column(self, keep):
        # [FUNCTION DOC]
        # Times selecting the 100 largest rows by column A.
        self.df.nlargest(100, "A", keep=keep)

    def time_nlargest_two_columns(self, keep):
        # [FUNCTION DOC]
        # Times selecting the 100 largest rows by columns A then B.
        self.df.nlargest(100, ["A", "B"], keep=keep)

    def time_nsmallest_one_column(self, keep):
        # [FUNCTION DOC]
        # Times selecting the 100 smallest rows by column A.
        self.df.nsmallest(100, "A", keep=keep)

    def time_nsmallest_two_columns(self, keep):
        # [FUNCTION DOC]
        # Times selecting the 100 smallest rows by columns A then B.
        self.df.nsmallest(100, ["A", "B"], keep=keep)


class Describe:
    # [CLASS DOC]
    # Benchmark of summary statistics for one column and for a whole integer
    # frame. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of one million rows with three columns a, b, c of random
        # integers below 100, stored as df.
        self.df = DataFrame(
            {
                "a": np.random.randint(0, 100, 10**6),
                "b": np.random.randint(0, 100, 10**6),
                "c": np.random.randint(0, 100, 10**6),
            }
        )

    def time_series_describe(self):
        # [FUNCTION DOC]
        # Times describing column a as a series.
        self.df["a"].describe()

    def time_dataframe_describe(self):
        # [FUNCTION DOC]
        # Times describing the whole frame.
        self.df.describe()


class MemoryUsage:
    # [CLASS DOC]
    # Benchmark of the deep memory usage report for a float frame and for a frame
    # with one object column. No parameters. State: df and df2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 100000 by 2 random normal frame with columns A and B as df, and
        # a copy whose column A is converted to object dtype as df2.
        self.df = DataFrame(np.random.randn(100000, 2), columns=list("AB"))
        self.df2 = self.df.copy()
        self.df2["A"] = self.df2["A"].astype("object")

    def time_memory_usage(self):
        # [FUNCTION DOC]
        # Times the deep memory usage report of the float frame.
        self.df.memory_usage(deep=True)

    def time_memory_usage_object_dtype(self):
        # [FUNCTION DOC]
        # Times the deep memory usage report of the frame with an object column.
        self.df2.memory_usage(deep=True)


class Round:
    # [CLASS DOC]
    # Benchmark of rounding a frame and its transpose, measuring time and peak
    # memory. No parameters. State: df and df_t.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 10000 by 10 random normal frame as df and a copied transpose as
        # df_t.
        self.df = DataFrame(np.random.randn(10000, 10))
        self.df_t = self.df.transpose(copy=True)

    def time_round(self):
        # [FUNCTION DOC]
        # Times rounding the frame to integers.
        self.df.round()

    def time_round_transposed(self):
        # [FUNCTION DOC]
        # Times rounding the transposed frame.
        self.df_t.round()

    def peakmem_round(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of rounding the frame.
        self.df.round()

    def peakmem_round_transposed(self):
        # [FUNCTION DOC]
        # Peak-memory benchmark of rounding the transposed frame.
        self.df_t.round()


class Where:
    # [CLASS DOC]
    # Benchmark of conditional replacement in a frame, in place or not, for
    # numpy float, nullable float and arrow float dtypes. Parametrized by inplace
    # flag and dtype. State: df and mask.
    # KNOWN ISSUE: the parameter name list has only one entry (dtype) although
    # there are two parameter lists (inplace and dtype), so the first axis is
    # unnamed.
    params = (
        [True, False],
        ["float64", "Float64", "float64[pyarrow]"],
    )
    param_names = ["dtype"]

    def setup(self, inplace, dtype):
        # [FUNCTION DOC]
        # Builds a 100000 by 10 random normal frame in the given dtype as df and a
        # boolean mask of negative values as mask. The inplace flag is unused
        # here.
        self.df = DataFrame(np.random.randn(100_000, 10), dtype=dtype)
        self.mask = self.df < 0

    def time_where(self, inplace, dtype):
        # [FUNCTION DOC]
        # Times keeping values where the mask is true and replacing the rest with
        # zero, in place (mutating df) or on a copy according to inplace.
        self.df.where(self.mask, other=0.0, inplace=inplace)


class FindValidIndex:
    # [CLASS DOC]
    # Benchmark of finding the first and last non-missing row labels in a frame
    # with missing values at both ends. Parametrized by dtype. State: df.
    param_names = ["dtype"]
    params = [
        ["float", "Float64", "float64[pyarrow]"],
    ]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a 100000 by 2 random normal frame with columns A and B in the
        # dtype. Sets the first 100 and last 100 rows of A and the first 200 and
        # last 200 rows of B to missing. Stores the frame as df.
        df = DataFrame(
            np.random.randn(100000, 2),
            columns=list("AB"),
            dtype=dtype,
        )
        df.iloc[:100, 0] = None
        df.iloc[:200, 1] = None
        df.iloc[-100:, 0] = None
        df.iloc[-200:, 1] = None
        self.df = df

    def time_first_valid_index(self, dtype):
        # [FUNCTION DOC]
        # Times finding the label of the first row with any non-missing value.
        self.df.first_valid_index()

    def time_last_valid_index(self, dtype):
        # [FUNCTION DOC]
        # Times finding the label of the last row with any non-missing value.
        self.df.last_valid_index()


class Update:
    # [CLASS DOC]
    # Benchmark of in-place update of a frame from another frame aligned on
    # labels, for different size relationships and index orders. No parameters.
    # State: df, df_random and df_sample. The update methods mutate the frames.
    def setup(self):
        # [FUNCTION DOC]
        # Uses a new random generator. Builds df (one million by 10 uniform
        # values), df_random (the same data re-indexed by a random permutation of
        # the integers below one million, via alignment) and df_sample (100000 by
        # 2 uniform values indexed by a random unique sample of row labels with two
        # randomly chosen column labels).
        rng = np.random.default_rng()
        self.df = DataFrame(rng.uniform(size=(1_000_000, 10)))

        idx = rng.choice(range(1_000_000), size=1_000_000, replace=False)
        self.df_random = DataFrame(self.df, index=idx)

        idx = rng.choice(range(1_000_000), size=100_000, replace=False)
        cols = rng.choice(range(10), size=2, replace=False)
        self.df_sample = DataFrame(
            rng.uniform(size=(100_000, 2)), index=idx, columns=cols
        )

    def time_to_update_big_frame_small_arg(self):
        # [FUNCTION DOC]
        # Times updating the big frame in place from the small sample frame.
        self.df.update(self.df_sample)

    def time_to_update_random_indices(self):
        # [FUNCTION DOC]
        # Times updating the randomly indexed frame in place from the sample
        # frame.
        self.df_random.update(self.df_sample)

    def time_to_update_small_frame_big_arg(self):
        # [FUNCTION DOC]
        # Times updating the small sample frame in place from the big frame.
        self.df_sample.update(self.df)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
