# [MODULE DOC]
# Benchmarks for Index objects: set operations across index kinds, disjoint
# datetime differences, unions with duplicates, range index operations, equality
# checks, appending, positional/boolean/label lookup, float label lookup, interval
# index methods and garbage-collection behavior of index engines. Public contents
# are the benchmark classes SetOperations, SetDisjoint, UnionWithDuplicates, Range,
# IndexEquals, IndexAppend, Indexing, Float64IndexMethod, IntervalIndexMethod and
# GC. No import-time side effects; the final line re-exports the common asv setup
# hook.
import gc

import numpy as np

from pandas import (
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    date_range,
)


class SetOperations:
    # [CLASS DOC]
    # Benchmark of intersection, union and symmetric difference between an index
    # and a shortened copy of it, for several index kinds and for monotonic or
    # reversed data. Parametrized by index structure, dtype kind and method name.
    # State: left and right indexes.
    params = (
        ["monotonic", "non_monotonic"],
        ["datetime", "date_string", "int", "strings", "ea_int"],
        ["intersection", "union", "symmetric_difference"],
    )
    param_names = ["index_structure", "dtype", "method"]

    def setup(self, index_structure, dtype, method):
        # [FUNCTION DOC]
        # Builds five indexes of 100000 elements (minute datetimes, the same as
        # formatted strings, integers, object strings, nullable integers). If the
        # structure is non_monotonic, reverses each one. For each kind pairs the
        # full index (left) with the index missing its last element (right) and
        # stores the pair for the requested dtype. The method parameter is unused
        # here.
        N = 10**5
        dates_left = date_range("1/1/2000", periods=N, freq="min")
        fmt = "%Y-%m-%d %H:%M:%S"
        date_str_left = Index(dates_left.strftime(fmt))
        int_left = Index(np.arange(N))
        ea_int_left = Index(np.arange(N), dtype="Int64")
        str_left = Index([f"i-{i}" for i in range(N)], dtype=object)

        data = {
            "datetime": dates_left,
            "date_string": date_str_left,
            "int": int_left,
            "strings": str_left,
            "ea_int": ea_int_left,
        }

        if index_structure == "non_monotonic":
            data = {k: mi[::-1] for k, mi in data.items()}

        data = {k: {"left": idx, "right": idx[:-1]} for k, idx in data.items()}

        self.left = data[dtype]["left"]
        self.right = data[dtype]["right"]

    def time_operation(self, index_structure, dtype, method):
        # [FUNCTION DOC]
        # Times calling the set operation named by method on left with right.
        # The other parameters are unused here.
        getattr(self.left, method)(self.right)


class SetDisjoint:
    # [CLASS DOC]
    # Benchmark of the difference of two datetime indexes with no common values.
    # No parameters. State: datetime_left and datetime_right.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a datetime index from the integers 0 to 99999 (nanosecond
        # offsets from the epoch) and another from 100000 to 119999.
        N = 10**5
        B = N + 20000
        self.datetime_left = DatetimeIndex(range(N))
        self.datetime_right = DatetimeIndex(range(N, B))

    def time_datetime_difference_disjoint(self):
        # [FUNCTION DOC]
        # Times the difference of the left index minus the disjoint right index.
        self.datetime_left.difference(self.datetime_right)


class UnionWithDuplicates:
    # [CLASS DOC]
    # Benchmark of the union of two integer indexes that contain many duplicates.
    # No parameters. State: left and right.
    def setup(self):
        # [FUNCTION DOC]
        # Builds left from the integers 0 to 999 each repeated 100 times, and
        # right from the integers 500 to 1499 tiled 50 times.
        self.left = Index(np.repeat(np.arange(1000), 100))
        self.right = Index(np.tile(np.arange(500, 1500), 50))

    def time_union_with_duplicates(self):
        # [FUNCTION DOC]
        # Times the union of left and right.
        self.left.union(self.right)


class Range:
    # [CLASS DOC]
    # Benchmark of range-index operations (extremes, label lookup, iteration,
    # sorting) for an increasing and a decreasing range of step three. No
    # parameters. State: idx_inc and idx_dec.
    def setup(self):
        # [FUNCTION DOC]
        # Builds idx_inc (0 up to a million exclusive, step three) and idx_dec
        # (a million down to zero, step minus three).
        self.idx_inc = RangeIndex(start=0, stop=10**6, step=3)
        self.idx_dec = RangeIndex(start=10**6, stop=-1, step=-3)

    def time_max(self):
        # [FUNCTION DOC]
        # Times the maximum of the increasing range (requires reading the end).
        self.idx_inc.max()

    def time_max_trivial(self):
        # [FUNCTION DOC]
        # Times the maximum of the decreasing range (the first element, trivial).
        self.idx_dec.max()

    def time_min(self):
        # [FUNCTION DOC]
        # Times the minimum of the decreasing range (requires reading the end).
        self.idx_dec.min()

    def time_min_trivial(self):
        # [FUNCTION DOC]
        # Times the minimum of the increasing range (the first element, trivial).
        self.idx_inc.min()

    def time_get_loc_inc(self):
        # [FUNCTION DOC]
        # Times finding the position of the label 900000 in the increasing range.
        self.idx_inc.get_loc(900_000)

    def time_get_loc_dec(self):
        # [FUNCTION DOC]
        # Times finding the position of the label 100000 in the decreasing range.
        self.idx_dec.get_loc(100_000)

    def time_iter_inc(self):
        # [FUNCTION DOC]
        # Times iterating over every element of the increasing range.
        for _ in self.idx_inc:
            pass

    def time_iter_dec(self):
        # [FUNCTION DOC]
        # Times iterating over every element of the decreasing range.
        for _ in self.idx_dec:
            pass

    def time_sort_values_asc(self):
        # [FUNCTION DOC]
        # Times sorting the increasing range in ascending order (already sorted).
        self.idx_inc.sort_values()

    def time_sort_values_des(self):
        # [FUNCTION DOC]
        # Times sorting the increasing range in descending order.
        self.idx_inc.sort_values(ascending=False)


class IndexEquals:
    # [CLASS DOC]
    # Benchmark of comparing a non-object index against a large multi-index. No
    # parameters. State: mi_large_slow and idx_non_object.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a multi-index as the product of a 100000-element range index and
        # a one-element datetime index (large and slow to materialize), and a
        # one-element range index.
        idx_large_fast = RangeIndex(100_000)
        idx_small_slow = date_range(start="1/1/2012", periods=1)
        self.mi_large_slow = MultiIndex.from_product([idx_large_fast, idx_small_slow])

        self.idx_non_object = RangeIndex(1)

    def time_non_object_equals_multiindex(self):
        # [FUNCTION DOC]
        # Times comparing the small range index with the large multi-index (should
        # return quickly because the lengths differ).
        self.idx_non_object.equals(self.mi_large_slow)


class IndexAppend:
    # [CLASS DOC]
    # Benchmark of appending a list of about ten thousand indexes of 100 elements
    # to one index of 100 elements, for range, integer and string indexes. No
    # parameters. State: range_idx, int_idx, obj_idx, range_idxs, int_idxs,
    # object_idxs and same_range_idx.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a base range index 0 to 99 with integer and string copies, then
        # 9999 consecutive 100-element range indexes (starting at 100) with
        # integer and string versions of each, and a list of 10000 references to
        # the base range index.
        N = 10_000
        self.range_idx = RangeIndex(0, 100)
        self.int_idx = self.range_idx.astype(int)
        self.obj_idx = self.int_idx.astype(str)
        self.range_idxs = []
        self.int_idxs = []
        self.object_idxs = []
        for i in range(1, N):
            r_idx = RangeIndex(i * 100, (i + 1) * 100)
            self.range_idxs.append(r_idx)
            i_idx = r_idx.astype(int)
            self.int_idxs.append(i_idx)
            o_idx = i_idx.astype(str)
            self.object_idxs.append(o_idx)
        self.same_range_idx = [self.range_idx] * N

    def time_append_range_list(self):
        # [FUNCTION DOC]
        # Times appending the list of consecutive range indexes to the base range
        # index.
        self.range_idx.append(self.range_idxs)

    def time_append_range_list_same(self):
        # [FUNCTION DOC]
        # Times appending the list of identical range indexes to the base range
        # index.
        self.range_idx.append(self.same_range_idx)

    def time_append_int_list(self):
        # [FUNCTION DOC]
        # Times appending the list of integer indexes to the integer index.
        self.int_idx.append(self.int_idxs)

    def time_append_obj_list(self):
        # [FUNCTION DOC]
        # Times appending the list of string indexes to the string index.
        self.obj_idx.append(self.object_idxs)


class Indexing:
    # [CLASS DOC]
    # Benchmark of selection and label lookup on a million-element index of
    # strings, floats or integers. Parametrized by kind. State: idx, array_mask,
    # series_mask, sorted, non_unique, non_unique_sorted and key.
    params = ["String", "Float", "Int"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds idx of a million elements by kind (object strings, float64 or
        # int64; an unknown kind leaves idx unset). Also: array_mask (true for
        # every third position) and the same as a series; sorted (idx sorted);
        # non_unique (the first half of idx appended to itself); non_unique_sorted
        # (the first half of the sorted index with each element repeated twice);
        # and key (the element a quarter of the way into the sorted index).
        N = 10**6
        if dtype == "String":
            self.idx = Index([f"i-{i}" for i in range(N)], dtype=object)
        elif dtype == "Float":
            self.idx = Index(np.arange(N), dtype=np.float64)
        elif dtype == "Int":
            self.idx = Index(np.arange(N), dtype=np.int64)
        self.array_mask = (np.arange(N) % 3) == 0
        self.series_mask = Series(self.array_mask)
        self.sorted = self.idx.sort_values()
        half = N // 2
        self.non_unique = self.idx[:half].append(self.idx[:half])
        self.non_unique_sorted = self.sorted[:half].repeat(2)
        self.key = self.sorted[N // 4]

    def time_boolean_array(self, dtype):
        # [FUNCTION DOC]
        # Times selecting with the boolean numpy mask.
        self.idx[self.array_mask]

    def time_boolean_series(self, dtype):
        # [FUNCTION DOC]
        # Times selecting with the boolean series mask.
        self.idx[self.series_mask]

    def time_get(self, dtype):
        # [FUNCTION DOC]
        # Times selecting the element at position one.
        self.idx[1]

    def time_slice(self, dtype):
        # [FUNCTION DOC]
        # Times selecting all but the last element with a slice.
        self.idx[:-1]

    def time_slice_step(self, dtype):
        # [FUNCTION DOC]
        # Times selecting every second element with a stepped slice.
        self.idx[::2]

    def time_get_loc(self, dtype):
        # [FUNCTION DOC]
        # Times finding the position of the key in the unsorted unique index.
        self.idx.get_loc(self.key)

    def time_get_loc_sorted(self, dtype):
        # [FUNCTION DOC]
        # Times finding the position of the key in the sorted unique index.
        self.sorted.get_loc(self.key)

    def time_get_loc_non_unique(self, dtype):
        # [FUNCTION DOC]
        # Times finding the positions of the key in the non-unique index.
        self.non_unique.get_loc(self.key)

    def time_get_loc_non_unique_sorted(self, dtype):
        # [FUNCTION DOC]
        # Times finding the positions of the key in the sorted non-unique index.
        self.non_unique_sorted.get_loc(self.key)


class Float64IndexMethod:
    # [CLASS DOC]
    # Benchmark (issue 13166) of label lookup in a float index with tiny
    # irregular spacing. No parameters. State: ind.
    # GH 13166
    def setup(self):
        # [FUNCTION DOC]
        # Builds a float index of 100000 values equal to the integers times a tiny
        # constant (about 4.8e-8) and stores it as ind.
        N = 100_000
        a = np.arange(N, dtype=np.float64)
        self.ind = Index(a * 4.8000000418824129e-08)

    def time_get_loc(self):
        # [FUNCTION DOC]
        # Times finding the position of the label zero.
        self.ind.get_loc(0)


class IntervalIndexMethod:
    # [CLASS DOC]
    # Benchmark (issue 24813) of interval index properties and set operations,
    # including indexes with one duplicated interval. Parametrized by size N (1000
    # or 100000); there is no parameter name list. State: intv, intv2, left and
    # right.
    # GH 24813
    params = [10**3, 10**5]

    def setup(self, N):
        # [FUNCTION DOC]
        # Builds intv from intervals (i, i+1) for i below N plus one extra interval
        # (0, 1), which duplicates the first; intv2 as intv shifted by one; left
        # as contiguous unit intervals from breaks 0 to N-1; right as contiguous
        # unit intervals from breaks N-3 to 2N-4 (overlapping the end of left).
        # Touches the internal lookup engine of intv and intv2 so engine
        # construction is excluded from the timings.
        left = np.append(np.arange(N), np.array(0))
        right = np.append(np.arange(1, N + 1), np.array(1))
        self.intv = IntervalIndex.from_arrays(left, right)
        self.intv._engine

        self.intv2 = IntervalIndex.from_arrays(left + 1, right + 1)
        self.intv2._engine

        self.left = IntervalIndex.from_breaks(np.arange(N))
        self.right = IntervalIndex.from_breaks(np.arange(N - 3, 2 * N - 3))

    def time_monotonic_inc(self, N):
        # [FUNCTION DOC]
        # Times reading the increasing-monotonic property of intv.
        self.intv.is_monotonic_increasing

    def time_is_unique(self, N):
        # [FUNCTION DOC]
        # Times reading the uniqueness property of intv.
        self.intv.is_unique

    def time_intersection(self, N):
        # [FUNCTION DOC]
        # Times intersecting left with right (unique, overlapping).
        self.left.intersection(self.right)

    def time_intersection_one_duplicate(self, N):
        # [FUNCTION DOC]
        # Times intersecting intv (which has a duplicate) with right.
        self.intv.intersection(self.right)

    def time_intersection_both_duplicate(self, N):
        # [FUNCTION DOC]
        # Times intersecting intv with the shifted intv2 (both with a duplicate).
        self.intv.intersection(self.intv2)


class GC:
    # [CLASS DOC]
    # Benchmark of peak memory when many large indexes are created and dropped
    # with the garbage collector disabled. Parametrized by the number of
    # repetitions (1, 2, 5); there is no parameter name list. No state.
    params = [1, 2, 5]

    def create_use_drop(self):
        # [FUNCTION DOC]
        # HELPER: workload for the peak-memory benchmark. Builds an index of a
        # million integers from a list, touches its internal lookup engine, then
        # lets the index go out of scope when the call returns. Not collected by
        # asv because the name has no benchmark prefix.
        idx = Index(list(range(1_000_000)))
        idx._engine

    def peakmem_gc_instances(self, N):
        # [FUNCTION DOC]
        # Peak-memory benchmark. Disables the garbage collector, runs the create
        # and drop helper N times, and always re-enables the collector in a
        # final block (global state change restored even on error).
        try:
            gc.disable()

            for _ in range(N):
                self.create_use_drop()
        finally:
            gc.enable()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
