"""
These benchmarks are for Series and DataFrame indexing methods.  For the
lower-level methods directly on Index and subclasses, see index_object.py,
indexing_engine.py, and index_cached.py
"""

# [MODULE DOC]
# Benchmarks for Series and DataFrame indexing and assignment: positional and
# label selection on numeric, string, datetime, period, interval, categorical and
# multi-level indexes, boolean selection, take, cross-section selection, column
# insertion and assignment, single-row lookup, chained assignment and block
# assignment. Public contents are the benchmark classes NumericSeriesIndexing,
# NumericMaskedIndexing, NonNumericSeriesIndexing, DataFrameStringIndexing,
# DataFrameNumericIndexing, Take, MultiIndexing, IntervalIndexing,
# DatetimeIndexIndexing, SortedAndUnsortedDatetimeIndexLoc, CategoricalIndexIndexing,
# MethodLookup, GetItemSingleColumn, IndexSingleRow, AssignTimeseriesIndex,
# InsertColumns, Setitem, SetitemObjectDtype, SeriesSetitem, ChainIndexing and
# Block. No import-time side effects; the final line re-exports the common asv
# setup hook.
# KNOWN ISSUE: the module docstring above refers to files named the indexing
# engine module and the index cached module with names that do not match the real
# files (the actual files are the plural indexing_engines and
# index_cached_properties modules).

from datetime import datetime
import warnings

import numpy as np

from pandas import (
    NA,
    CategoricalIndex,
    DataFrame,
    Index,
    IntervalIndex,
    MultiIndex,
    Series,
    concat,
    date_range,
    option_context,
    period_range,
)


class NumericSeriesIndexing:
    # [CLASS DOC]
    # Benchmark of selection from a million-element series with a numeric index,
    # via the plain item operator, positional accessor and label accessor, with
    # scalar, slice, list-like, array and list selectors. Parametrized by index
    # dtype and index structure (unique increasing, or non-unique increasing with
    # one duplicate). State: data, array and array_list.
    params = [
        (np.int64, np.uint64, np.float64),
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),
    ]
    param_names = ["dtype", "index_structure"]

    def setup(self, dtype, index_structure):
        # [FUNCTION DOC]
        # Builds both candidate indexes of a million labels in the dtype (a plain
        # range, and a range in which the label 54 is duplicated and the final
        # label is skipped) and a series of random uniform values on the chosen
        # one as data. Also builds array (the integers below 10000) and the same
        # as a list.
        N = 10**6
        indices = {
            "unique_monotonic_inc": Index(range(N), dtype=dtype),
            "nonunique_monotonic_inc": Index(
                [*list(range(55)), 54, *list(range(55, N - 1))], dtype=dtype
            ),
        }
        self.data = Series(np.random.rand(N), index=indices[index_structure])
        self.array = np.arange(10000)
        self.array_list = self.array.tolist()

    def time_getitem_scalar(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access by the scalar label 800000. The first parameter is
        # named index but carries the dtype; both parameters are unused here.
        self.data[800000]

    def time_getitem_slice(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access with a slice up to 800000.
        self.data[:800000]

    def time_getitem_list_like(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access with a one-element list holding 800000.
        self.data[[800000]]

    def time_getitem_array(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access with the integer array of the first 10000 integers.
        self.data[self.array]

    def time_getitem_lists(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access with the same selectors as a plain list.
        self.data[self.array_list]

    def time_iloc_array(self, index, index_structure):
        # [FUNCTION DOC]
        # Times positional selection with the integer array.
        self.data.iloc[self.array]

    def time_iloc_list_like(self, index, index_structure):
        # [FUNCTION DOC]
        # Times positional selection with a one-element list holding 800000.
        self.data.iloc[[800000]]

    def time_iloc_scalar(self, index, index_structure):
        # [FUNCTION DOC]
        # Times positional selection of the single position 800000.
        self.data.iloc[800000]

    def time_iloc_slice(self, index, index_structure):
        # [FUNCTION DOC]
        # Times positional selection with a slice up to 800000.
        self.data.iloc[:800000]

    def time_loc_array(self, index, index_structure):
        # [FUNCTION DOC]
        # Times label selection with the integer array.
        self.data.loc[self.array]

    def time_loc_list_like(self, index, index_structure):
        # [FUNCTION DOC]
        # Times label selection with a one-element list holding 800000.
        self.data.loc[[800000]]

    def time_loc_scalar(self, index, index_structure):
        # [FUNCTION DOC]
        # Times label selection of the single label 800000.
        self.data.loc[800000]

    def time_loc_slice(self, index, index_structure):
        # [FUNCTION DOC]
        # Times label selection with a slice up to 800000.
        self.data.loc[:800000]


class NumericMaskedIndexing:
    # [CLASS DOC]
    # Benchmark of indexer computation (position lookup of many labels) on
    # nullable numeric indexes, monotonic or not, and the same with duplicates.
    # Parametrized by nullable dtype name and monotonic flag. The class attributes
    # monotonic_list (a million increasing integers) and non_monotonic_list
    # (the same with positions 50 to 54 permuted) are built when the module is
    # imported. State: data, indexer and data_dups.
    monotonic_list = list(range(10**6))
    non_monotonic_list = [*list(range(50)), 54, 53, 52, 51, *list(range(55, 10**6 - 1))]

    params = [
        ("Int64", "UInt64", "Float64"),
        (True, False),
    ]
    param_names = ["dtype", "monotonic"]

    def setup(self, dtype, monotonic):
        # [FUNCTION DOC]
        # Builds a nullable index in the dtype from the monotonic list, or from the
        # non-monotonic list with a missing value appended, and picks the requested
        # one as data. Also builds indexer (the integers 300 to 999) and data_dups
        # (data appended to itself, producing duplicates).
        indices = {
            True: Index(self.monotonic_list, dtype=dtype),
            False: Index(self.non_monotonic_list, dtype=dtype).append(
                Index([NA], dtype=dtype)
            ),
        }
        self.data = indices[monotonic]
        self.indexer = np.arange(300, 1_000)
        self.data_dups = self.data.append(self.data)

    def time_get_indexer(self, dtype, monotonic):
        # [FUNCTION DOC]
        # Times looking up the positions of the indexer labels in data.
        # KNOWN ISSUE: the data_dups index built in setup is never used here or in
        # the next method, which also uses data, so the duplicate case is not
        # actually measured.
        self.data.get_indexer(self.indexer)

    def time_get_indexer_dups(self, dtype, monotonic):
        # [FUNCTION DOC]
        # Times the lookup variant that tolerates duplicates, on data (not the
        # duplicated index) with the indexer labels.
        self.data.get_indexer_for(self.indexer)


class NonNumericSeriesIndexing:
    # [CLASS DOC]
    # Benchmark of selection from a million-element series indexed by strings,
    # datetimes or periods, for unique increasing, non-unique increasing and
    # non-monotonic structures. Parametrized by index kind and structure. State:
    # s and lbl.
    params = [
        ("string", "datetime", "period"),
        ("unique_monotonic_inc", "nonunique_monotonic_inc", "non_monotonic"),
    ]
    param_names = ["index_dtype", "index_structure"]

    def setup(self, index, index_structure):
        # [FUNCTION DOC]
        # Builds a million-element index of the requested kind (object strings,
        # per-second datetimes from 1900, or per-second periods from 1900), sorts
        # it, and asserts it is unique and monotonic increasing. For the
        # non-unique structure, duplicates the third label and drops the last
        # element. For the non-monotonic structure, appends the odd-position
        # elements after the even-position ones (asserting the length is
        # unchanged). Builds a random series on it as s, sets lbl to the label at
        # position 80000, and reads that label once to warm up the index lookup
        # mapping so its construction is not timed. Note the first parameter name
        # is reused for the built index.
        N = 10**6
        if index == "string":
            index = Index([f"i-{i}" for i in range(N)], dtype=object)
        elif index == "datetime":
            index = date_range("1900", periods=N, freq="s")
        elif index == "period":
            index = period_range("1900", periods=N, freq="s")
        index = index.sort_values()
        assert index.is_unique and index.is_monotonic_increasing
        if index_structure == "nonunique_monotonic_inc":
            index = index.insert(item=index[2], loc=2)[:-1]
        elif index_structure == "non_monotonic":
            index = index[::2].append(index[1::2])
            assert len(index) == N
        self.s = Series(np.random.rand(N), index=index)
        self.lbl = index[80000]
        # warm up index mapping
        self.s[self.lbl]

    def time_getitem_label_slice(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access with a slice up to the label lbl.
        self.s[: self.lbl]

    def time_getitem_pos_slice(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access with the positional slice up to 80000.
        self.s[:80000]

    def time_getitem_scalar(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access by the scalar label lbl.
        self.s[self.lbl]

    def time_getitem_list_like(self, index, index_structure):
        # [FUNCTION DOC]
        # Times item access by a one-element list holding the label lbl.
        self.s[[self.lbl]]


class DataFrameStringIndexing:
    # [CLASS DOC]
    # Benchmark of scalar and boolean selection from a small frame with string row
    # and column labels. No parameters. State: df, idx_scalar, col_scalar,
    # bool_indexer, bool_obj_indexer and boolean_indexer.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000 by 30 random normal frame with string labels (warnings
        # during construction are recorded and discarded), picks the row label at
        # position 100 and column label at position 10 as scalars, and builds three
        # row selectors from the sign of that column: boolean numpy-backed, object
        # dtype and nullable boolean dtype.
        index = Index([f"i-{i}" for i in range(1000)], dtype=object)
        columns = Index([f"i-{i}" for i in range(30)], dtype=object)
        with warnings.catch_warnings(record=True):
            self.df = DataFrame(np.random.randn(1000, 30), index=index, columns=columns)
        self.idx_scalar = index[100]
        self.col_scalar = columns[10]
        self.bool_indexer = self.df[self.col_scalar] > 0
        self.bool_obj_indexer = self.bool_indexer.astype(object)
        self.boolean_indexer = (self.df[self.col_scalar] > 0).astype("boolean")

    def time_loc(self):
        # [FUNCTION DOC]
        # Times label-based selection of one cell.
        self.df.loc[self.idx_scalar, self.col_scalar]

    def time_at(self):
        # [FUNCTION DOC]
        # Times the fast scalar accessor for one cell.
        self.df.at[self.idx_scalar, self.col_scalar]

    def time_at_setitem(self):
        # [FUNCTION DOC]
        # Times setting one cell to 0.0 through the fast scalar accessor
        # (mutates the frame).
        self.df.at[self.idx_scalar, self.col_scalar] = 0.0

    def time_getitem_scalar(self):
        # [FUNCTION DOC]
        # Times chained item access: first the column, then the row label.
        self.df[self.col_scalar][self.idx_scalar]

    def time_boolean_rows(self):
        # [FUNCTION DOC]
        # Times row selection with the boolean series.
        self.df[self.bool_indexer]

    def time_boolean_rows_object(self):
        # [FUNCTION DOC]
        # Times row selection with the object-dtype boolean series.
        self.df[self.bool_obj_indexer]

    def time_boolean_rows_boolean(self):
        # [FUNCTION DOC]
        # Times row selection with the nullable-boolean series.
        self.df[self.boolean_indexer]


class DataFrameNumericIndexing:
    # [CLASS DOC]
    # Benchmark of selection from frames with a numeric index, including a
    # duplicated-index frame. Parametrized by index dtype and index structure
    # (unique, or non-unique with one duplicated label). State: idx_dupe, df,
    # df_dup and bool_indexer.
    params = [
        (np.int64, np.uint64, np.float64),
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),
    ]
    param_names = ["dtype", "index_structure"]

    def setup(self, dtype, index_structure):
        # [FUNCTION DOC]
        # With 100000 rows: builds both candidate indexes (range, or range with
        # label 54 duplicated and the last label skipped); idx_dupe (thirty
        # multiples of 99); df (a random normal frame with five columns on the
        # chosen index); df_dup (df, twice df and three times df stacked, giving
        # an index that repeats); and bool_indexer (a list whose first half is true
        # and second half false).
        N = 10**5
        indices = {
            "unique_monotonic_inc": Index(range(N), dtype=dtype),
            "nonunique_monotonic_inc": Index(
                [*list(range(55)), 54, *list(range(55, N - 1))], dtype=dtype
            ),
        }
        self.idx_dupe = np.array(range(30)) * 99
        self.df = DataFrame(np.random.randn(N, 5), index=indices[index_structure])
        self.df_dup = concat([self.df, 2 * self.df, 3 * self.df])
        self.bool_indexer = [True] * (N // 2) + [False] * (N - N // 2)

    def time_iloc_dups(self, index, index_structure):
        # [FUNCTION DOC]
        # Times positional selection of the thirty positions from the stacked frame.
        self.df_dup.iloc[self.idx_dupe]

    def time_loc_dups(self, index, index_structure):
        # [FUNCTION DOC]
        # Times label selection of the thirty labels from the stacked frame, where
        # each label occurs several times.
        self.df_dup.loc[self.idx_dupe]

    def time_iloc(self, index, index_structure):
        # [FUNCTION DOC]
        # Times positional selection of the first 100 rows of column position zero.
        self.df.iloc[:100, 0]

    def time_loc(self, index, index_structure):
        # [FUNCTION DOC]
        # Times label selection of rows up to label 100 in column label zero.
        self.df.loc[:100, 0]

    def time_bool_indexer(self, index, index_structure):
        # [FUNCTION DOC]
        # Times row selection with the boolean list.
        self.df[self.bool_indexer]


class Take:
    # [CLASS DOC]
    # Benchmark of position-based take on a series with an integer or datetime
    # index. Parametrized by index kind. State: s and indexer.
    params = ["int", "datetime"]
    param_names = ["index"]

    def setup(self, index):
        # [FUNCTION DOC]
        # Builds a 100000-element index of the requested kind (64-bit integers, or
        # per-second timestamps from 2011-01-01), a random series on it as s, and
        # indexer (100000 random positions with repeats). The parameter name is
        # reused for the built index.
        N = 100000
        indexes = {
            "int": Index(np.arange(N), dtype=np.int64),
            "datetime": date_range("2011-01-01", freq="s", periods=N),
        }
        index = indexes[index]
        self.s = Series(np.random.rand(N), index=index)
        self.indexer = np.random.randint(0, N, size=N)

    def time_take(self, index):
        # [FUNCTION DOC]
        # Times taking the elements at the random positions.
        self.s.take(self.indexer)


class MultiIndexing:
    # [CLASS DOC]
    # Benchmark of selection and cross-section extraction on a two-level
    # multi-index frame with unique or repeating levels. Parametrized by the
    # unique_levels flag. State: nlevels, df and the target selectors tgt_slice,
    # tgt_null_slice, tgt_list, tgt_scalar and tgt_bool_indexer.
    params = [True, False]
    param_names = ["unique_levels"]

    def setup(self, unique_levels):
        # [FUNCTION DOC]
        # Sets two levels. If unique_levels, builds the index from two identical
        # arrays of a million integers (a million unique pairs); otherwise from the
        # product of two ranges of 1000 (a million pairs, with each level
        # repeating). Builds a random one-column frame on it. Defines selectors:
        # a slice 200 to 800, the null slice, a list of every tenth integer below
        # 1000, the scalar 500, and a boolean mask that is true at every hundredth
        # position.
        self.nlevels = 2
        if unique_levels:
            mi = MultiIndex.from_arrays([range(1000000)] * self.nlevels)
        else:
            mi = MultiIndex.from_product([range(1000)] * self.nlevels)
        self.df = DataFrame(np.random.randn(len(mi)), index=mi)

        self.tgt_slice = slice(200, 800)
        self.tgt_null_slice = slice(None)
        self.tgt_list = list(range(0, 1000, 10))
        self.tgt_scalar = 500

        bool_indexer = np.zeros(len(mi), dtype=np.bool_)
        bool_indexer[slice(0, len(mi), 100)] = True
        self.tgt_bool_indexer = bool_indexer

    def time_loc_partial_key_slice(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection of the first level by a slice (all columns).
        self.df.loc[self.tgt_slice, :]

    def time_loc_partial_key_null_slice(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection of the first level by a null slice.
        self.df.loc[self.tgt_null_slice, :]

    def time_loc_partial_key_list(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection of the first level by a list of labels.
        self.df.loc[self.tgt_list, :]

    def time_loc_partial_key_scalar(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection of the first level by a scalar label.
        self.df.loc[self.tgt_scalar, :]

    def time_loc_partial_key_bool_indexer(self, unique_levels):
        # [FUNCTION DOC]
        # Times row selection by the boolean mask.
        self.df.loc[self.tgt_bool_indexer, :]

    def time_loc_all_slices(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with a slice for every level (a tuple key).
        target = tuple([self.tgt_slice] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_null_slices(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with a null slice for every level.
        target = tuple([self.tgt_null_slice] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_lists(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with a list of labels for every level.
        target = tuple([self.tgt_list] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_scalars(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with a scalar label for every level (a full key).
        target = tuple([self.tgt_scalar] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_bool_indexers(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with the boolean mask for every level.
        target = tuple([self.tgt_bool_indexer] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_slice_plus_null_slice(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with a slice on the first level and a null slice on
        # the second.
        target = (self.tgt_slice, self.tgt_null_slice)
        self.df.loc[target, :]

    def time_loc_null_slice_plus_slice(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection with a null slice on the first level and a slice on
        # the second.
        target = (self.tgt_null_slice, self.tgt_slice)
        self.df.loc[target, :]

    def time_loc_multiindex(self, unique_levels):
        # [FUNCTION DOC]
        # Times label selection using every tenth element of the frame's own
        # multi-index as the target.
        target = self.df.index[::10]
        self.df.loc[target]

    def time_xs_level_0(self, unique_levels):
        # [FUNCTION DOC]
        # Times a cross-section by the scalar on the first level.
        target = self.tgt_scalar
        self.df.xs(target, level=0)

    def time_xs_level_1(self, unique_levels):
        # [FUNCTION DOC]
        # Times a cross-section by the scalar on the second level.
        target = self.tgt_scalar
        self.df.xs(target, level=1)

    def time_xs_full_key(self, unique_levels):
        # [FUNCTION DOC]
        # Times a cross-section by a full key (the scalar for every level).
        target = tuple([self.tgt_scalar] * self.nlevels)
        self.df.xs(target)


class IntervalIndexing:
    # [CLASS DOC]
    # Benchmark of scalar and slice selection from a series with a million
    # contiguous unit-interval index entries. Data is built once by the cache hook
    # and passed to each timing method as the series.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Builds an interval index from the breaks 0 to 1000000 and
        # a series of the integers below a million on it; returns the series.
        idx = IntervalIndex.from_breaks(np.arange(1000001))
        monotonic = Series(np.arange(1000000), index=idx)
        return monotonic

    def time_getitem_scalar(self, monotonic):
        # [FUNCTION DOC]
        # Times item access by the scalar 80000, which falls inside one interval.
        monotonic[80000]

    def time_loc_scalar(self, monotonic):
        # [FUNCTION DOC]
        # Times label access by the scalar 80000.
        monotonic.loc[80000]

    def time_getitem_list(self, monotonic):
        # [FUNCTION DOC]
        # Times item access with a slice starting at 80000 (despite the name,
        # it is a slice).
        monotonic[80000:]

    def time_loc_list(self, monotonic):
        # [FUNCTION DOC]
        # Times label access with a slice starting at 80000 (despite the name, it
        # is a slice).
        monotonic.loc[80000:]


class DatetimeIndexIndexing:
    # [CLASS DOC]
    # Benchmark of position lookup between two timezone-aware datetime indexes
    # with different zones. No parameters. State: dti and dti2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a daily index of 10000 entries in the US Pacific zone as dti and
        # the same instants converted to UTC as dti2.
        dti = date_range("2016-01-01", periods=10000, tz="US/Pacific")
        dti2 = dti.tz_convert("UTC")
        self.dti = dti
        self.dti2 = dti2

    def time_get_indexer_mismatched_tz(self):
        # [FUNCTION DOC]
        # Times looking up the positions of dti2 in dti, where the time zones
        # differ. This path is reached when selecting a series by an index of
        # another zone.
        # reached via e.g.
        #  ser = Series(range(len(dti)), index=dti)
        #  ser[dti2]
        self.dti.get_indexer(self.dti2)


class SortedAndUnsortedDatetimeIndexLoc:
    # [CLASS DOC]
    # Benchmark of partial-string date lookups on frames indexed by raw datetime
    # arrays that are sorted or contain one out-of-order element. No parameters.
    # State: df_unsorted and df_sort.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 10000 daily timestamps (Pacific zone) and extracts the underlying
        # raw datetime array. df_sort uses it as the index; df_unsorted uses a copy
        # where the element at position ten is replaced by the element at position
        # twenty (making the index unsorted and non-unique). Both frames have a
        # constant column a.
        dti = date_range("2016-01-01", periods=10000, tz="US/Pacific")
        index = np.array(dti)

        unsorted_index = index.copy()
        unsorted_index[10] = unsorted_index[20]

        self.df_unsorted = DataFrame(index=unsorted_index, data={"a": 1})
        self.df_sort = DataFrame(index=index, data={"a": 1})

    def time_loc_unsorted(self):
        # [FUNCTION DOC]
        # Times a partial-string date lookup on the unsorted frame.
        self.df_unsorted.loc["2016-6-11"]

    def time_loc_sorted(self):
        # [FUNCTION DOC]
        # Times a partial-string date lookup on the sorted frame.
        self.df_sort.loc["2016-6-11"]


class CategoricalIndexIndexing:
    # [CLASS DOC]
    # Benchmark of item access and lookups on a categorical index of 300000
    # elements in monotonic increasing, monotonic decreasing and non-monotonic
    # layouts. Parametrized by layout. State: data, data_unique, int_scalar,
    # int_list, cat_scalar, cat_list.
    params = ["monotonic_incr", "monotonic_decr", "non_monotonic"]
    param_names = ["index"]

    def setup(self, index):
        # [FUNCTION DOC]
        # Builds three categorical indexes from the letters a, b, c (100000 each):
        # grouped ascending, grouped descending, and cycled; picks the one for the
        # layout as data. data_unique is a categorical index of 300000 distinct
        # strings. Defines selectors: the position 10000, the first 10000
        # positions, the category b, and the categories 1 and 3 for a lookup list.
        N = 10**5
        values = list("a" * N + "b" * N + "c" * N)
        indices = {
            "monotonic_incr": CategoricalIndex(values),
            "monotonic_decr": CategoricalIndex(reversed(values)),
            "non_monotonic": CategoricalIndex(list("abc" * N)),
        }
        self.data = indices[index]
        self.data_unique = CategoricalIndex([str(i) for i in range(N * 3)])

        self.int_scalar = 10000
        self.int_list = list(range(10000))

        self.cat_scalar = "b"
        self.cat_list = ["1", "3"]

    def time_getitem_scalar(self, index):
        # [FUNCTION DOC]
        # Times item access at position 10000.
        self.data[self.int_scalar]

    def time_getitem_slice(self, index):
        # [FUNCTION DOC]
        # Times item access with a slice of the first 10000 positions.
        self.data[: self.int_scalar]

    def time_getitem_list_like(self, index):
        # [FUNCTION DOC]
        # Times item access with a one-element list holding position 10000.
        self.data[[self.int_scalar]]

    def time_getitem_list(self, index):
        # [FUNCTION DOC]
        # Times item access with a list of the first 10000 positions.
        self.data[self.int_list]

    def time_getitem_bool_array(self, index):
        # [FUNCTION DOC]
        # Times item access with a boolean mask comparing the index to the label b.
        self.data[self.data == self.cat_scalar]

    def time_get_loc_scalar(self, index):
        # [FUNCTION DOC]
        # Times finding the location of the label b.
        self.data.get_loc(self.cat_scalar)

    def time_get_indexer_list(self, index):
        # [FUNCTION DOC]
        # Times looking up the positions of the two labels in the unique index.
        self.data_unique.get_indexer(self.cat_list)


class MethodLookup:
    # [CLASS DOC]
    # Benchmark of the cost of reading the positional and label accessor
    # attributes of an empty series. Data is built once by the cache hook and
    # passed to the timing methods.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns a new empty series.
        s = Series()
        return s

    def time_lookup_iloc(self, s):
        # [FUNCTION DOC]
        # Times reading the positional accessor attribute.
        s.iloc

    def time_lookup_loc(self, s):
        # [FUNCTION DOC]
        # Times reading the label accessor attribute.
        s.loc


class GetItemSingleColumn:
    # [CLASS DOC]
    # Benchmark of selecting a single column by string label or by integer label.
    # No parameters. State: df_string_col and df_int_col.
    def setup(self):
        # [FUNCTION DOC]
        # Builds two 3000 by 1 random normal frames: one with the column label A
        # and one with the default integer label.
        self.df_string_col = DataFrame(np.random.randn(3000, 1), columns=["A"])
        self.df_int_col = DataFrame(np.random.randn(3000, 1))

    def time_frame_getitem_single_column_label(self):
        # [FUNCTION DOC]
        # Times selecting the column A from the string-labelled frame.
        self.df_string_col["A"]

    def time_frame_getitem_single_column_int(self):
        # [FUNCTION DOC]
        # Times selecting the column zero from the integer-labelled frame.
        self.df_int_col[0]


class IndexSingleRow:
    # [CLASS DOC]
    # Benchmark of single-row lookup on a ten-million-cell mixed-dtype frame,
    # with unique or duplicate column labels. Parametrized by the unique_cols
    # flag. State: df.
    params = [True, False]
    param_names = ["unique_cols"]

    def setup(self, unique_cols):
        # [FUNCTION DOC]
        # Builds a frame from the integers below ten million reshaped to ten
        # columns, then converts each column to a different dtype in order
        # (unsigned 8, 16, 32, 64, signed 8, 16, 32, 64, 64-bit float, 32-bit
        # float). If unique_cols is false, renames the first two columns both to A
        # (issue 33032: single-row lookup with non-unique columns was fifteen times
        # slower). Stores as df.
        arr = np.arange(10**7).reshape(-1, 10)
        df = DataFrame(arr)
        dtypes = ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f8", "f4"]
        for i, d in enumerate(dtypes):
            df[i] = df[i].astype(d)

        if not unique_cols:
            # GH#33032 single-row lookups with non-unique columns were
            #  15x slower than with unique columns
            df.columns = ["A", "A", *list(df.columns[2:])]

        self.df = df

    def time_iloc_row(self, unique_cols):
        # [FUNCTION DOC]
        # Times positional selection of the row at position 10000.
        self.df.iloc[10000]

    def time_loc_row(self, unique_cols):
        # [FUNCTION DOC]
        # Times label selection of the row labelled 10000.
        self.df.loc[10000]


class AssignTimeseriesIndex:
    # [CLASS DOC]
    # Benchmark of assigning a datetime index as a new column. No parameters.
    # State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds an hourly datetime index of 100000 entries and a one-column random
        # frame A on it, stored as df.
        N = 100000
        idx = date_range("1/1/2000", periods=N, freq="h")
        self.df = DataFrame(np.random.randn(N, 1), columns=["A"], index=idx)

    def time_frame_assign_timeseries_index(self):
        # [FUNCTION DOC]
        # Times adding a column named date equal to the index (mutates the frame).
        self.df["date"] = self.df.index


class InsertColumns:
    # [CLASS DOC]
    # Benchmark of adding columns to frames by insertion and assignment. The
    # timing methods mutate the frames, so repeated runs on the same instance see
    # growing frames unless asv re-runs setup. No parameters. State: N, df (empty
    # with 1000 rows) and df2 (1000 by 2 random).
    def setup(self):
        # [FUNCTION DOC]
        # Sets N to 1000, builds an empty frame with a 1000-row range index as
        # df, and a 1000 by 2 random normal frame as df2.
        self.N = 10**3
        self.df = DataFrame(index=range(self.N))
        self.df2 = DataFrame(np.random.randn(self.N, 2))

    def time_insert(self):
        # [FUNCTION DOC]
        # Times inserting 100 random columns at the front of df, labelled by the
        # loop counter, allowing duplicate labels (mutates df).
        for i in range(100):
            self.df.insert(0, i, np.random.randn(self.N), allow_duplicates=True)

    def time_insert_middle(self):
        # [FUNCTION DOC]
        # Times inserting a random column labelled colname at position one of df2
        # 100 times, allowing duplicates (mutates df2). Inserting in the middle
        # avoids the fast paths for front and back insertion.
        # same as time_insert but inserting to a middle column rather than
        #  front or back (which have fast-paths)
        for i in range(100):
            self.df2.insert(
                1, "colname", np.random.randn(self.N), allow_duplicates=True
            )

    def time_assign_with_setitem(self):
        # [FUNCTION DOC]
        # Times assigning 100 random columns labelled 0 to 99 to df, one at a time
        # (mutates df).
        for i in range(100):
            self.df[i] = np.random.randn(self.N)

    def time_assign_list_like_with_setitem(self):
        # [FUNCTION DOC]
        # Times assigning a block of 100 random columns to the list of labels 0 to
        # 99 in one operation (mutates df).
        self.df[list(range(100))] = np.random.randn(self.N, 100)

    def time_assign_list_of_columns_concat(self):
        # [FUNCTION DOC]
        # Times building a 1000 by 100 random frame and concatenating it to df
        # column-wise (the build is included in the timing).
        df = DataFrame(np.random.randn(self.N, 100))
        concat([self.df, df], axis=1)


class Setitem:
    # [CLASS DOC]
    # Benchmark of assigning a scalar to one column or a list of columns of a
    # wide frame. No parameters. State: df (500000 by 500).
    def setup(self):
        # [FUNCTION DOC]
        # Builds a 500000 by 500 uniform random frame as df.
        N = 500_000
        cols = 500
        self.df = DataFrame(np.random.rand(N, cols))

    def time_setitem(self):
        # [FUNCTION DOC]
        # Times setting column 100 to the scalar 100 (mutates df).
        self.df[100] = 100

    def time_setitem_list(self):
        # [FUNCTION DOC]
        # Times setting columns 100, 200 and 300 to the scalar 100 (mutates df).
        self.df[[100, 200, 300]] = 100


class SetitemObjectDtype:
    # [CLASS DOC]
    # Benchmark (issue 19299) of setting a single cell in an object-dtype frame.
    # No parameters. State: df.
    # GH#19299

    def setup(self):
        # [FUNCTION DOC]
        # Builds a 1000 by 500 frame of object dtype with range labels (all
        # entries missing) as df.
        N = 1000
        cols = 500
        self.df = DataFrame(index=range(N), columns=range(cols), dtype=object)

    def time_setitem_object_dtype(self):
        # [FUNCTION DOC]
        # Times setting the cell at row label 0, column label 1 to 1.0 (mutates df).
        self.df.loc[0, 1] = 1.0


class SeriesSetitem:
    # [CLASS DOC]
    # Benchmark of assigning a whole array over a null slice of a series. The
    # default string dtype is exempt from the defensive copy done for null slices,
    # so a fixed-width arrow float dtype is also used to keep that cost visible
    # (issue 67990). Parametrized by dtype. State: s, arr and arr_obj.
    # "str" is exempt from the null-slice defensive copy, so a fixed-width
    #  pyarrow dtype is needed to keep that cost on the dashboard (GH#67990)
    params = ["str", "double[pyarrow]"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a series of 500000 random uniform values in the dtype as s, takes
        # its underlying extension array as arr, and an object-dtype numpy copy of
        # that array as arr_obj.
        N = 500_000
        self.s = Series(np.random.rand(N), dtype=dtype)
        self.arr = self.s.array
        self.arr_obj = np.asarray(self.s.array, dtype=object)

    def time_setitem_slice_array(self, dtype):
        # [FUNCTION DOC]
        # Times assigning the extension array over the entire series via a null
        # slice (mutates s). See pull request 64530.
        # https://github.com/pandas-dev/pandas/pull/64530
        self.s[:] = self.arr

    def time_setitem_slice_array_infer(self, dtype):
        # [FUNCTION DOC]
        # Times assigning the object-dtype array over the entire series via a null
        # slice, requiring dtype inference (mutates s).
        self.s[:] = self.arr_obj


class ChainIndexing:
    # [CLASS DOC]
    # Benchmark of the chained-assignment pattern (assigning into a filtered copy)
    # under different chained-assignment option modes. Parametrized by mode (none
    # or warn). State: N and df.
    params = [None, "warn"]
    param_names = ["mode"]

    def setup(self, mode):
        # [FUNCTION DOC]
        # Sets N to a million and builds a frame with column A (the integers below
        # N) and column B (constant string foo). The mode parameter is unused here.
        self.N = 1000000
        self.df = DataFrame({"A": np.arange(self.N), "B": "foo"})

    def time_chained_indexing(self, mode):
        # [FUNCTION DOC]
        # Times selecting the rows with A above half of N and then assigning a new
        # column on the result, inside a context with the chained-assignment option
        # set to the mode (global option restored on exit) and warnings recorded
        # and discarded.
        df = self.df
        N = self.N
        with warnings.catch_warnings(record=True):
            with option_context("mode.chained_assignment", mode):
                df2 = df[df.A > N // 2]
                df2["C"] = 1.0


class Block:
    # [CLASS DOC]
    # Benchmark of assigning a scalar to a date-range slice of a wide boolean
    # frame. No parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a boolean frame filled with false, with 500 columns (labels are
        # the integers as strings) and a daily index from 2010-01-01 to
        # 2011-01-01. Stores as df.
        self.df = DataFrame(
            False,
            columns=np.arange(500).astype(str),
            index=date_range("2010-01-01", "2011-01-01"),
        )

    def time_test(self):
        # [FUNCTION DOC]
        # Times setting all columns for the rows between 2010-05-01 and 2010-09-01
        # (inclusive) to true (mutates df).
        start = datetime(2010, 5, 1)
        end = datetime(2010, 9, 1)
        self.df.loc[start:end, :] = True


from .pandas_vb_common import setup  # noqa: F401 isort:skip
