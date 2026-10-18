# [MODULE DOC]
# Benchmarks for MultiIndex objects: label lookup, duplicate handling, integer
# indexers, sorting, values extraction, categorical levels, equality, set
# operations, uniqueness, membership, mask-based replacement and appending. Public
# contents are the benchmark classes GetLoc, GetLocs, Duplicates, Integer,
# Duplicated, Sortlevel, SortValues, Values, CategoricalLevel, Equals,
# SetOperations, Difference, Unique, Isin, Putmask and Append. No import-time side
# effects; the final line re-exports the common asv setup hook.
import string

import numpy as np

from pandas import (
    NA,
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    array,
    date_range,
)


class GetLoc:
    # [CLASS DOC]
    # Benchmark of single-key label lookup in multi-indexes of three sizes, cold
    # and with 1000 repeated lookups. No parameters. State: mi_large (1000 by 20 by
    # 52 entries), mi_med (1000 by 10 by 1) and mi_small (100 by 1 by 1).
    def setup(self):
        # [FUNCTION DOC]
        # Builds three three-level product multi-indexes named one, two, three: a
        # large one (integers below 1000, integers below 20, ASCII letters), a
        # medium one (integers below 1000, integers below 10, the letter A) and a
        # small one (integers below 100, the letter A, the letter A).
        self.mi_large = MultiIndex.from_product(
            [np.arange(1000), np.arange(20), list(string.ascii_letters)],
            names=["one", "two", "three"],
        )
        self.mi_med = MultiIndex.from_product(
            [np.arange(1000), np.arange(10), list("A")], names=["one", "two", "three"]
        )
        self.mi_small = MultiIndex.from_product(
            [np.arange(100), list("A"), list("A")], names=["one", "two", "three"]
        )

    def time_large_get_loc(self):
        # [FUNCTION DOC]
        # Times locating the last key of the large index (first lookup includes
        # building the lookup tables).
        self.mi_large.get_loc((999, 19, "Z"))

    def time_large_get_loc_warm(self):
        # [FUNCTION DOC]
        # Times 1000 repeated lookups of the last key of the large index.
        for _ in range(1000):
            self.mi_large.get_loc((999, 19, "Z"))

    def time_med_get_loc(self):
        # [FUNCTION DOC]
        # Times locating the last key of the medium index.
        self.mi_med.get_loc((999, 9, "A"))

    def time_med_get_loc_warm(self):
        # [FUNCTION DOC]
        # Times 1000 repeated lookups of the last key of the medium index.
        for _ in range(1000):
            self.mi_med.get_loc((999, 9, "A"))

    def time_string_get_loc(self):
        # [FUNCTION DOC]
        # Times locating the last key of the small index.
        self.mi_small.get_loc((99, "A", "A"))

    def time_small_get_loc_warm(self):
        # [FUNCTION DOC]
        # Times 1000 repeated lookups of the last key of the small index.
        for _ in range(1000):
            self.mi_small.get_loc((99, "A", "A"))


class GetLocs:
    # [CLASS DOC]
    # Benchmark of multi-level label list lookup (positions matching a list of one
    # label per level) in multi-indexes of three sizes. No parameters. State:
    # mi_large, mi_med and mi_small, built as in the single-key lookup benchmark.
    def setup(self):
        # [FUNCTION DOC]
        # Builds the same three product multi-indexes as the single-key lookup
        # benchmark: large (integers below 1000, integers below 20, ASCII
        # letters), medium (integers below 1000, integers below 10, letter A) and
        # small (integers below 100, letter A, letter A), named one, two, three.
        self.mi_large = MultiIndex.from_product(
            [np.arange(1000), np.arange(20), list(string.ascii_letters)],
            names=["one", "two", "three"],
        )
        self.mi_med = MultiIndex.from_product(
            [np.arange(1000), np.arange(10), list("A")], names=["one", "two", "three"]
        )
        self.mi_small = MultiIndex.from_product(
            [np.arange(100), list("A"), list("A")], names=["one", "two", "three"]
        )

    def time_large_get_locs(self):
        # [FUNCTION DOC]
        # Times locating positions for the labels 999, 19 and Z in the large index.
        self.mi_large.get_locs([999, 19, "Z"])

    def time_med_get_locs(self):
        # [FUNCTION DOC]
        # Times locating positions for the labels 999, 9 and A in the medium index.
        self.mi_med.get_locs([999, 9, "A"])

    def time_small_get_locs(self):
        # [FUNCTION DOC]
        # Times locating positions for the labels 99, A and A in the small index.
        self.mi_small.get_locs([99, "A", "A"])


class Duplicates:
    # [CLASS DOC]
    # Benchmark of removing unused levels from a randomly sub-selected
    # multi-index. No parameters. State: mi_unused_levels.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a two-level multi-index of 65536 entries from random integers
        # below 8192 and below 1024, then keeps only a random subset (about ten
        # percent) so many level values become unused. Stores it as
        # mi_unused_levels.
        size = 65536
        arrays = [np.random.randint(0, 8192, size), np.random.randint(0, 1024, size)]
        mask = np.random.rand(size) < 0.1
        self.mi_unused_levels = MultiIndex.from_arrays(arrays)
        self.mi_unused_levels = self.mi_unused_levels[mask]

    def time_remove_unused_levels(self):
        # [FUNCTION DOC]
        # Times removing the unused level values.
        self.mi_unused_levels.remove_unused_levels()


class Integer:
    # [CLASS DOC]
    # Benchmark of position lookup in an integer multi-index with exact keys and
    # with fill methods for non-matching keys, and of monotonicity. No parameters.
    # State: mi_int, obj_index and other_mi_many_mismatches.
    def setup(self):
        # [FUNCTION DOC]
        # Builds mi_int (the product of two ranges of 1000, named one and two);
        # obj_index (an object array of ten pairs (0, 10) to (0, 19)); and
        # other_mi_many_mismatches (a multi-index of eighteen tuples with some
        # values that are absent from mi_int, including negative numbers, fractional
        # numbers and values beyond the range).
        self.mi_int = MultiIndex.from_product(
            [np.arange(1000), np.arange(1000)], names=["one", "two"]
        )
        self.obj_index = np.array(
            [
                (0, 10),
                (0, 11),
                (0, 12),
                (0, 13),
                (0, 14),
                (0, 15),
                (0, 16),
                (0, 17),
                (0, 18),
                (0, 19),
            ],
            dtype=object,
        )
        self.other_mi_many_mismatches = MultiIndex.from_tuples(
            [
                (-7, 41),
                (-2, 3),
                (-0.7, 5),
                (0, 0),
                (0, 1.5),
                (0, 340),
                (0, 1001),
                (1, -4),
                (1, 20),
                (1, 1040),
                (432, -5),
                (432, 17),
                (439, 165.5),
                (998, -4),
                (998, 24065),
                (999, 865.2),
                (999, 1000),
                (1045, -843),
            ]
        )

    def time_get_indexer(self):
        # [FUNCTION DOC]
        # Times looking up the positions of the ten pairs in mi_int.
        self.mi_int.get_indexer(self.obj_index)

    def time_get_indexer_and_backfill(self):
        # [FUNCTION DOC]
        # Times looking up the mismatched tuples with the backward-fill method.
        self.mi_int.get_indexer(self.other_mi_many_mismatches, method="backfill")

    def time_get_indexer_and_pad(self):
        # [FUNCTION DOC]
        # Times looking up the mismatched tuples with the forward-fill method.
        self.mi_int.get_indexer(self.other_mi_many_mismatches, method="pad")

    def time_is_monotonic(self):
        # [FUNCTION DOC]
        # Times reading the increasing-monotonic property of mi_int.
        self.mi_int.is_monotonic_increasing


class Duplicated:
    # [CLASS DOC]
    # Benchmark of duplicate detection in a three-level multi-index with a million
    # entries drawn at random from small levels. No parameters. State: mi.
    def setup(self):
        # [FUNCTION DOC]
        # With n equal to 200 and k equal to 5000: builds three levels of 200
        # values (integers, strings, integers offset by 1000) and, for each, one
        # million random codes. Builds the multi-index from the levels and codes and
        # stores it as mi.
        n, k = 200, 5000
        levels = [
            np.arange(n),
            Index([f"i-{i}" for i in range(n)], dtype=object).values,
            1000 + np.arange(n),
        ]
        codes = [np.random.choice(n, (k * n)) for lev in levels]
        self.mi = MultiIndex(levels=levels, codes=codes)

    def time_duplicated(self):
        # [FUNCTION DOC]
        # Times marking duplicate entries.
        self.mi.duplicated()


class Sortlevel:
    # [CLASS DOC]
    # Benchmark of sorting a multi-index by a level, for five integer levels and
    # for two integer levels. No parameters. State: mi_int and mi.
    def setup(self):
        # [FUNCTION DOC]
        # Builds mi_int: five levels of 1182720 entries each made by repeating
        # random integers in blocks of 11, 7, 5, 3 and 1 positions, with the rows
        # shuffled. Builds mi: a two-level index (each of 100 values of the first
        # level combined with 1000 values of the second) with 100000 rows shuffled.
        n = 1182720
        low, high = -4096, 4096
        arrs = [
            np.repeat(np.random.randint(low, high, (n // k)), k)
            for k in [11, 7, 5, 3, 1]
        ]
        self.mi_int = MultiIndex.from_arrays(arrs)[np.random.permutation(n)]

        a = np.repeat(np.arange(100), 1000)
        b = np.tile(np.arange(1000), 100)
        self.mi = MultiIndex.from_arrays([a, b])
        self.mi = self.mi.take(np.random.permutation(np.arange(100000)))

    def time_sortlevel_int64(self):
        # [FUNCTION DOC]
        # Times sorting the five-level integer index by its default level.
        self.mi_int.sortlevel()

    def time_sortlevel_zero(self):
        # [FUNCTION DOC]
        # Times sorting the two-level index by level zero.
        self.mi.sortlevel(0)

    def time_sortlevel_one(self):
        # [FUNCTION DOC]
        # Times sorting the two-level index by level one.
        self.mi.sortlevel(1)


class SortValues:
    # [CLASS DOC]
    # Benchmark of sorting the values of a two-level index of integers, plain or
    # nullable. Parametrized by dtype (int64, Int64). State: mi.
    params = ["int64", "Int64"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds two arrays of 100000 values in the dtype: the integers below 100
        # tiled 1000 times and the integers below 1000 tiled 100 times, and makes a
        # two-level index from them as mi.
        a = array(np.tile(np.arange(100), 1000), dtype=dtype)
        b = array(np.tile(np.arange(1000), 100), dtype=dtype)
        self.mi = MultiIndex.from_arrays([a, b])

    def time_sort_values(self, dtype):
        # [FUNCTION DOC]
        # Times sorting the index values. The dtype parameter is unused.
        self.mi.sort_values()


class Values:
    # [CLASS DOC]
    # Benchmark of extracting the values of a multi-index with a datetime level,
    # whole after a copy and sliced. Data is built once by the cache hook and
    # passed to the timing methods.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook. Returns the product multi-index of integers below 1000
        # and 100 daily timestamps from 2012-01-01.
        level1 = range(1000)
        level2 = date_range(start="1/1/2012", periods=100)
        mi = MultiIndex.from_product([level1, level2])
        return mi

    def time_datetime_level_values_copy(self, mi):
        # [FUNCTION DOC]
        # Times copying the index and materializing its values array.
        mi.copy().values

    def time_datetime_level_values_sliced(self, mi):
        # [FUNCTION DOC]
        # Times materializing the values of the first ten entries (a slice).
        mi[:10].values


class CategoricalLevel:
    # [CLASS DOC]
    # Benchmark of building a multi-index from two categorical columns. No
    # parameters. State: df.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of a million rows with a 32-bit integer column a, a 64-bit
        # integer column b and a float column c, then converts a and b to
        # categorical dtype. Stores as df.
        self.df = DataFrame(
            {
                "a": np.arange(1_000_000, dtype=np.int32),
                "b": np.arange(1_000_000, dtype=np.int64),
                "c": np.arange(1_000_000, dtype=float),
            }
        ).astype({"a": "category", "b": "category"})

    def time_categorical_level(self):
        # [FUNCTION DOC]
        # Times setting the two categorical columns as a multi-index.
        self.df.set_index(["a", "b"])


class Equals:
    # [CLASS DOC]
    # Benchmark of equality checks of a large multi-index against a deep copy and
    # against a non-object index. No parameters. State: mi, mi_deepcopy and
    # idx_non_object.
    def setup(self):
        # [FUNCTION DOC]
        # Builds mi as the product of 1000 daily timestamps from 2000-01-01 and a
        # range index of 1000, a deep copy of it, and a one-element range index.
        self.mi = MultiIndex.from_product(
            [
                date_range("2000-01-01", periods=1000),
                RangeIndex(1000),
            ]
        )
        self.mi_deepcopy = self.mi.copy(deep=True)
        self.idx_non_object = RangeIndex(1)

    def time_equals_deepcopy(self):
        # [FUNCTION DOC]
        # Times comparing the index with its deep copy (equal).
        self.mi.equals(self.mi_deepcopy)

    def time_equals_non_object_index(self):
        # [FUNCTION DOC]
        # Times comparing the multi-index with the one-element range index
        # (unequal, different lengths and kinds).
        self.mi.equals(self.idx_non_object)


class SetOperations:
    # [CLASS DOC]
    # Benchmark of intersection, union and symmetric difference between a
    # multi-index and a shortened copy, for several level types, monotonic or
    # reversed order, and two sort settings. Parametrized by index structure,
    # dtype kind, method and sort. State: left and right.
    params = [
        ("monotonic", "non_monotonic"),
        ("datetime", "int", "string", "ea_int"),
        ("intersection", "union", "symmetric_difference"),
        (False, None),
    ]
    param_names = ["index_structure", "dtype", "method", "sort"]

    def setup(self, index_structure, dtype, method, sort):
        # [FUNCTION DOC]
        # Builds four product multi-indexes of 100000 entries: the first level is
        # the integers below 1000 and the second level (100 values) is daily
        # timestamps from 2000-01-01, integers, strings, or nullable integers. For
        # the non-monotonic structure reverses them all. Pairs each full index
        # (left) with the index missing its last entry (right) and stores the pair
        # for the requested dtype kind as left and right.
        N = 10**5
        level1 = range(1000)

        level2 = date_range(start="1/1/2000", periods=N // 1000)
        dates_left = MultiIndex.from_product([level1, level2])

        level2 = range(N // 1000)
        int_left = MultiIndex.from_product([level1, level2])

        level2 = Index([f"i-{i}" for i in range(N // 1000)], dtype=object).values
        str_left = MultiIndex.from_product([level1, level2])

        level2 = range(N // 1000)
        ea_int_left = MultiIndex.from_product([level1, Series(level2, dtype="Int64")])

        data = {
            "datetime": dates_left,
            "int": int_left,
            "string": str_left,
            "ea_int": ea_int_left,
        }

        if index_structure == "non_monotonic":
            data = {k: mi[::-1] for k, mi in data.items()}

        data = {k: {"left": mi, "right": mi[:-1]} for k, mi in data.items()}
        self.left = data[dtype]["left"]
        self.right = data[dtype]["right"]

    def time_operation(self, index_structure, dtype, method, sort):
        # [FUNCTION DOC]
        # Times calling the set operation named by method on left with right and the
        # sort setting.
        getattr(self.left, method)(self.right, sort=sort)


class Difference:
    # [CLASS DOC]
    # Benchmark of the set difference between a multi-index of 20000 entries and
    # its first five entries, for several level types. Parametrized by dtype kind.
    # State: left and right.
    params = [
        ("datetime", "int", "string", "ea_int"),
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds four product multi-indexes of 20000 entries: the first level is the
        # integers below 1000 and the second level (20 values) is daily timestamps,
        # integers, strings, or nullable integers whose first value is set to
        # missing. Pairs each with its first five entries and stores the pair for the
        # requested kind as left and right.
        N = 10**4 * 2
        level1 = range(1000)

        level2 = date_range(start="1/1/2000", periods=N // 1000)
        dates_left = MultiIndex.from_product([level1, level2])

        level2 = range(N // 1000)
        int_left = MultiIndex.from_product([level1, level2])

        level2 = Series(range(N // 1000), dtype="Int64")
        level2[0] = NA
        ea_int_left = MultiIndex.from_product([level1, level2])

        level2 = Index([f"i-{i}" for i in range(N // 1000)], dtype=object).values
        str_left = MultiIndex.from_product([level1, level2])

        data = {
            "datetime": dates_left,
            "int": int_left,
            "ea_int": ea_int_left,
            "string": str_left,
        }

        data = {k: {"left": mi, "right": mi[:5]} for k, mi in data.items()}
        self.left = data[dtype]["left"]
        self.right = data[dtype]["right"]

    def time_difference(self, dtype):
        # [FUNCTION DOC]
        # Times the set difference of left minus right.
        self.left.difference(self.right)


class Unique:
    # [CLASS DOC]
    # Benchmark of computing the unique entries of a two-level multi-index with a
    # missing or zero value present, nullable versus plain integers. Parametrized
    # by a (dtype, special value) pair, given as one tuple option. State: midx and
    # midx_dups.
    params = [
        (("Int64", NA), ("int64", 0)),
    ]
    param_names = ["dtype_val"]

    def setup(self, dtype_val):
        # [FUNCTION DOC]
        # Builds a level series in the dtype from one, two, the special value
        # twice, then the integers below a million, and uses it for both levels of
        # midx. Builds a second level series with the same four leading values then
        # two repetitions of the integers below 500000, and uses it for both levels
        # of midx_dups (so it has repeated entries).
        level = Series(
            [1, 2, dtype_val[1], dtype_val[1], *list(range(1000000))],
            dtype=dtype_val[0],
        )
        self.midx = MultiIndex.from_arrays([level, level])

        level_dups = Series(
            [1, 2, dtype_val[1], dtype_val[1]] + list(range(500_000)) * 2,
            dtype=dtype_val[0],
        )

        self.midx_dups = MultiIndex.from_arrays([level_dups, level_dups])

    def time_unique(self, dtype_val):
        # [FUNCTION DOC]
        # Times the unique entries of midx (few duplicates).
        self.midx.unique()

    def time_unique_dups(self, dtype_val):
        # [FUNCTION DOC]
        # Times the unique entries of midx_dups (many duplicates).
        self.midx_dups.unique()


class Isin:
    # [CLASS DOC]
    # Benchmark of membership testing of a 100000-entry multi-index against small
    # and large sets of its own entries, for several level types. Parametrized by
    # dtype kind. State: midx, values_small and values_large.
    params = [
        ("string", "int", "datetime"),
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds three product multi-indexes of 100000 entries (integers below 1000
        # crossed with 100 strings, integers or daily timestamps), selects the one
        # for the requested kind as midx, and takes its first 100 entries as
        # values_small and the remaining entries as values_large.
        N = 10**5
        level1 = range(1000)

        level2 = date_range(start="1/1/2000", periods=N // 1000)
        dates_midx = MultiIndex.from_product([level1, level2])

        level2 = range(N // 1000)
        int_midx = MultiIndex.from_product([level1, level2])

        level2 = Index([f"i-{i}" for i in range(N // 1000)], dtype=object).values
        str_midx = MultiIndex.from_product([level1, level2])

        data = {
            "datetime": dates_midx,
            "int": int_midx,
            "string": str_midx,
        }

        self.midx = data[dtype]
        self.values_small = self.midx[:100]
        self.values_large = self.midx[100:]

    def time_isin_small(self, dtype):
        # [FUNCTION DOC]
        # Times testing membership of every entry against the small set.
        self.midx.isin(self.values_small)

    def time_isin_large(self, dtype):
        # [FUNCTION DOC]
        # Times testing membership of every entry against the large set.
        self.midx.isin(self.values_large)


class Putmask:
    # [CLASS DOC]
    # Benchmark of replacing selected entries of a multi-index with entries from
    # another multi-index, where the replacements share or do not share the level
    # values. No parameters. State: midx, midx_values, midx_values_different and
    # mask.
    def setup(self):
        # [FUNCTION DOC]
        # Builds midx (integers below 1000 crossed with 100 daily timestamps from
        # 2000-01-01), midx_values (integers 1000 to 1999 crossed with the same
        # timestamps) and midx_values_different (the same integers crossed with
        # timestamps from 2010-01-01), plus a boolean mask alternating true and
        # false over the full length.
        N = 10**5
        level1 = range(1_000)

        level2 = date_range(start="1/1/2000", periods=N // 1000)
        self.midx = MultiIndex.from_product([level1, level2])

        level1 = range(1_000, 2_000)
        self.midx_values = MultiIndex.from_product([level1, level2])

        level2 = date_range(start="1/1/2010", periods=N // 1000)
        self.midx_values_different = MultiIndex.from_product([level1, level2])
        self.mask = np.array([True, False] * (N // 2))

    def time_putmask(self):
        # [FUNCTION DOC]
        # Times the masked replacement using values that share the second level.
        self.midx.putmask(self.mask, self.midx_values)

    def time_putmask_all_different(self):
        # [FUNCTION DOC]
        # Times the masked replacement using values whose second level differs
        # entirely.
        self.midx.putmask(self.mask, self.midx_values_different)


class Append:
    # [CLASS DOC]
    # Benchmark of appending two multi-indexes with disjoint first levels and a
    # shared second level of datetimes, integers or strings. Parametrized by the
    # dtype of the second level. State: left and right.
    params = ["datetime64[ns]", "int64", "string"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds left from integers 0 to 999 crossed with a 500-value second level,
        # and right from integers 1000 to 1999 crossed with the same second level.
        # The second level is daily timestamps from 2000-01-01, integers, or
        # strings depending on dtype; raises a not-implemented error for any other
        # dtype.
        N1 = 1000
        N2 = 500
        left_level1 = range(N1)
        right_level1 = range(N1, N1 + N1)

        if dtype == "datetime64[ns]":
            level2 = date_range(start="2000-01-01", periods=N2)
        elif dtype == "int64":
            level2 = range(N2)
        elif dtype == "string":
            level2 = Index([f"i-{i}" for i in range(N2)], dtype=object)
        else:
            raise NotImplementedError

        self.left = MultiIndex.from_product([left_level1, level2])
        self.right = MultiIndex.from_product([right_level1, level2])

    def time_append(self, dtype):
        # [FUNCTION DOC]
        # Times appending right to left.
        self.left.append(self.right)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
