# [MODULE DOC]
# Benchmarks for categorical data: construction, type conversion, concatenation
# and union, value counting, category set changes, ranking, monotonicity checks,
# membership, slicing, index operations and sorted search. Public contents are the
# benchmark classes Constructor, AsType, Concat, ValueCounts, Repr, SetCategories,
# RemoveCategories, Rank, IsMonotonic, Contains, CategoricalSlicing, Indexing and
# SearchSorted. At import time it tries to import the union-categoricals helper
# from the public types API, falling back to the old types module location and
# otherwise leaving the name undefined. The final line re-exports the common asv
# setup hook.
import string
import sys
import warnings

import numpy as np

import pandas as pd

try:
    from pandas.api.types import union_categoricals
except ImportError:
    try:
        from pandas.types.concat import union_categoricals
    except ImportError:
        pass


class Constructor:
    # [CLASS DOC]
    # Benchmark of the categorical constructor from several input kinds (plain
    # values, codes, datetimes, missing values, existing categoricals and
    # series). No parameters. State: categories, cat_idx, values, codes,
    # datetimes, datetimes_with_nat, values_some_nan, values_all_nan,
    # values_all_int8, categorical, series and intervals.
    def setup(self):
        # [FUNCTION DOC]
        # Builds the inputs for all timing methods. With N equal to 100000: five
        # categories (a to e) and an index of them; a values array of the
        # categories tiled N times and the matching integer codes; a series of
        # N/10 per-second timestamps and a copy whose last element is set to
        # missing; a list of the categories plus one missing value tiled N times;
        # a list of missing values as long as the values array; an int8 array of
        # ones of length N; a categorical and a series built from the values; and
        # an interval range of N/10 intervals between zero and one (intervals is
        # not used by any timing method).
        N = 10**5
        self.categories = list("abcde")
        self.cat_idx = pd.Index(self.categories)
        self.values = np.tile(self.categories, N)
        self.codes = np.tile(range(len(self.categories)), N)

        self.datetimes = pd.Series(
            pd.date_range("1995-01-01 00:00:00", periods=N // 10, freq="s")
        )
        self.datetimes_with_nat = self.datetimes.copy()
        self.datetimes_with_nat.iloc[-1] = pd.NaT

        self.values_some_nan = list(np.tile([*self.categories, np.nan], N))
        self.values_all_nan = [np.nan] * len(self.values)
        self.values_all_int8 = np.ones(N, "int8")
        self.categorical = pd.Categorical(self.values, self.categories)
        self.series = pd.Series(self.categorical)
        self.intervals = pd.interval_range(0, 1, periods=N // 10)

    def time_regular(self):
        # [FUNCTION DOC]
        # Times building a categorical from values and an explicit category list.
        pd.Categorical(self.values, self.categories)

    def time_fastpath(self):
        # [FUNCTION DOC]
        # Times the internal no-validation constructor given integer codes and a
        # dtype built from the category index.
        dtype = pd.CategoricalDtype(categories=self.cat_idx)
        pd.Categorical._simple_new(self.codes, dtype)

    def time_datetimes(self):
        # [FUNCTION DOC]
        # Times building a categorical from the datetime series, inferring the
        # categories.
        pd.Categorical(self.datetimes)

    def time_interval(self):
        # [FUNCTION DOC]
        # Times building a categorical from the datetime series using the same
        # series as the explicit categories.
        # KNOWN ISSUE: the name suggests an interval-based categorical, and an
        # intervals attribute is prepared in setup, but this method uses the
        # datetime series for both values and categories, so intervals are never
        # exercised.
        pd.Categorical(self.datetimes, categories=self.datetimes)

    def time_datetimes_with_nat(self):
        # [FUNCTION DOC]
        # Times building a categorical from the datetime series that has a missing
        # last element.
        pd.Categorical(self.datetimes_with_nat)

    def time_with_nan(self):
        # [FUNCTION DOC]
        # Times building a categorical from a list in which every sixth element is
        # missing.
        pd.Categorical(self.values_some_nan)

    def time_all_nan(self):
        # [FUNCTION DOC]
        # Times building a categorical from a list of only missing values.
        pd.Categorical(self.values_all_nan)

    def time_from_codes_all_int8(self):
        # [FUNCTION DOC]
        # Times the from-codes constructor using int8 codes and the category list.
        pd.Categorical.from_codes(self.values_all_int8, self.categories)

    def time_existing_categorical(self):
        # [FUNCTION DOC]
        # Times building a categorical from an existing categorical.
        pd.Categorical(self.categorical)

    def time_existing_series(self):
        # [FUNCTION DOC]
        # Times building a categorical from an existing categorical-typed series.
        pd.Categorical(self.series)


class AsType:
    # [CLASS DOC]
    # Intended benchmark of converting categorical columns back to other dtypes
    # (string, integer, float, timezone-aware datetime).
    # KNOWN ISSUE: none of the benchmark methods start with the time prefix that
    # asv requires, so asv does not collect them and they never run.
    # State: df, a frame with categorical columns str, int, float, timestamp
    # and the string forms int_as_str, float_as_str and timestamp_as_str.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows. Four columns are random draws (via a new
        # random generator) from category pools: ASCII letters (str), 154 random
        # integers below 65536 (int), 38 large random floats (float) and 578
        # timestamps made from random second counts (timestamp). Adds string
        # versions of the int, float and timestamp columns with the suffix as_str.
        # Finally converts every column to the categorical dtype. Stores as df.
        N = 10**5

        random_pick = np.random.default_rng().choice

        categories = {
            "str": list(string.ascii_letters),
            "int": np.random.randint(2**16, size=154),
            "float": sys.maxsize * np.random.random((38,)),
            "timestamp": [
                pd.Timestamp(x, unit="s") for x in np.random.randint(2**18, size=578)
            ],
        }

        self.df = pd.DataFrame(
            {col: random_pick(cats, N) for col, cats in categories.items()}
        )

        for col in ("int", "float", "timestamp"):
            self.df[f"{col}_as_str"] = self.df[col].astype(str)

        for col in self.df.columns:
            self.df[col] = self.df[col].astype("category")

    def astype_str(self):
        # [FUNCTION DOC]
        # Converts the int, float and timestamp categorical columns to string
        # dtype, collecting the results in a discarded list. Not collected by asv
        # (see class note).
        [self.df[col].astype("str") for col in "int float timestamp".split()]

    def astype_int(self):
        # [FUNCTION DOC]
        # Converts the int_as_str and timestamp categorical columns to integer
        # dtype, discarding the results. Not collected by asv (see class note).
        [self.df[col].astype("int") for col in "int_as_str timestamp".split()]

    def astype_float(self):
        # [FUNCTION DOC]
        # Converts the float_as_str, int, int_as_str and timestamp categorical
        # columns to float dtype, discarding the results. Not collected by asv
        # (see class note).
        [
            self.df[col].astype("float")
            for col in "float_as_str int int_as_str timestamp".split()
        ]

    def astype_datetime(self):
        # [FUNCTION DOC]
        # Converts the float categorical column to a datetime dtype in the US
        # Pacific timezone, discarding the result. Not collected by asv (see
        # class note).
        self.df["float"].astype(pd.DatetimeTZDtype(tz="US/Pacific"))


class Concat:
    # [CLASS DOC]
    # Benchmark of concatenating and unioning categorical data, including
    # categorical indexes and frames indexed by them. No parameters. State: s,
    # a, b, idx_a, idx_b, df_a and df_b.
    def setup(self):
        # [FUNCTION DOC]
        # With N equal to 100000: builds a categorical-typed series s from the
        # letters a, a, b, b, c, d repeated N times; categoricals a (same
        # letters) and b (letters b, b, c, d, j, k repeated N times, so partially
        # different categories); categorical indexes idx_a over N integers and
        # idx_b over N plus one integers (categories equal to the values); and
        # one-column frames df_a and df_b using those indexes.
        N = 10**5
        self.s = pd.Series(list("aabbcd") * N).astype("category")

        self.a = pd.Categorical(list("aabbcd") * N)
        self.b = pd.Categorical(list("bbcdjk") * N)

        self.idx_a = pd.CategoricalIndex(range(N), range(N))
        self.idx_b = pd.CategoricalIndex(range(N + 1), range(N + 1))
        self.df_a = pd.DataFrame(range(N), columns=["a"], index=self.idx_a)
        self.df_b = pd.DataFrame(range(N + 1), columns=["a"], index=self.idx_b)

    def time_concat(self):
        # [FUNCTION DOC]
        # Times concatenating the categorical series with itself.
        pd.concat([self.s, self.s])

    def time_union(self):
        # [FUNCTION DOC]
        # Times the union-categoricals helper on the two categoricals with
        # different category sets.
        union_categoricals([self.a, self.b])

    def time_append_overlapping_index(self):
        # [FUNCTION DOC]
        # Times appending a categorical index to itself (identical categories).
        self.idx_a.append(self.idx_a)

    def time_append_non_overlapping_index(self):
        # [FUNCTION DOC]
        # Times appending a categorical index to one with a different category
        # set.
        self.idx_a.append(self.idx_b)

    def time_concat_overlapping_index(self):
        # [FUNCTION DOC]
        # Times concatenating the first frame with itself, so the categorical
        # indexes share categories.
        pd.concat([self.df_a, self.df_a])

    def time_concat_non_overlapping_index(self):
        # [FUNCTION DOC]
        # Times concatenating the two frames whose categorical indexes have
        # different categories.
        pd.concat([self.df_a, self.df_b])


class ValueCounts:
    # [CLASS DOC]
    # Benchmark of counting values in a large categorical series. Parametrized
    # by the dropna flag. State: ts.
    params = [True, False]
    param_names = ["dropna"]

    def setup(self, dropna):
        # [FUNCTION DOC]
        # Builds 500000 strings with the letter s and a four-digit zero-padded
        # random integer below 50000, and stores them as a categorical-typed
        # series ts. The dropna parameter is unused here.
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    def time_value_counts(self, dropna):
        # [FUNCTION DOC]
        # Times counting the values, with missing values dropped according to the
        # parameter.
        self.ts.value_counts(dropna=dropna)


class Repr:
    # [CLASS DOC]
    # Benchmark of rendering a tiny categorical series as text. No parameters.
    # State: sel.
    def setup(self):
        # [FUNCTION DOC]
        # Creates a one-element categorical series holding the string s1234 and
        # stores it as sel.
        self.sel = pd.Series(["s1234"]).astype("category")

    def time_rendering(self):
        # [FUNCTION DOC]
        # Times converting the series to its text representation.
        str(self.sel)


class SetCategories:
    # [CLASS DOC]
    # Benchmark of replacing the category set of a large categorical series with
    # half of its categories. No parameters. State: ts.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 500000 formatted random strings (as in the value-counts
        # benchmark) as a categorical-typed series ts.
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    def time_set_categories(self):
        # [FUNCTION DOC]
        # Times setting the categories to every second existing category.
        self.ts.cat.set_categories(self.ts.cat.categories[::2])


class RemoveCategories:
    # [CLASS DOC]
    # Benchmark of removing half of the categories of a large categorical series.
    # No parameters. State: ts.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 500000 formatted random strings as a categorical-typed series ts.
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    def time_remove_categories(self):
        # [FUNCTION DOC]
        # Times removing every second existing category.
        self.ts.cat.remove_categories(self.ts.cat.categories[::2])


class Rank:
    # [CLASS DOC]
    # Benchmark of ranking series of strings and integers stored as plain,
    # unordered categorical and ordered categorical data. No parameters. State:
    # s_str, s_str_cat, s_str_cat_ordered, s_int, s_int_cat and s_int_cat_ordered.
    def setup(self):
        # [FUNCTION DOC]
        # With 100000 elements and 15 distinct values: builds a string series of
        # random integers cast to text, an unordered categorical copy and an
        # ordered categorical copy (using the set of values as categories, with
        # warnings recorded and discarded); then the same three forms for a
        # series of random integers.
        N = 10**5
        ncats = 15

        self.s_str = pd.Series(np.random.randint(0, ncats, size=N).astype(str))
        self.s_str_cat = pd.Series(self.s_str, dtype="category")
        with warnings.catch_warnings(record=True):
            str_cat_type = pd.CategoricalDtype(set(self.s_str), ordered=True)
            self.s_str_cat_ordered = self.s_str.astype(str_cat_type)

        self.s_int = pd.Series(np.random.randint(0, ncats, size=N))
        self.s_int_cat = pd.Series(self.s_int, dtype="category")
        with warnings.catch_warnings(record=True):
            int_cat_type = pd.CategoricalDtype(set(self.s_int), ordered=True)
            self.s_int_cat_ordered = self.s_int.astype(int_cat_type)

    def time_rank_string(self):
        # [FUNCTION DOC]
        # Times ranking the plain string series.
        self.s_str.rank()

    def time_rank_string_cat(self):
        # [FUNCTION DOC]
        # Times ranking the unordered categorical string series.
        self.s_str_cat.rank()

    def time_rank_string_cat_ordered(self):
        # [FUNCTION DOC]
        # Times ranking the ordered categorical string series.
        self.s_str_cat_ordered.rank()

    def time_rank_int(self):
        # [FUNCTION DOC]
        # Times ranking the plain integer series.
        self.s_int.rank()

    def time_rank_int_cat(self):
        # [FUNCTION DOC]
        # Times ranking the unordered categorical integer series.
        self.s_int_cat.rank()

    def time_rank_int_cat_ordered(self):
        # [FUNCTION DOC]
        # Times ranking the ordered categorical integer series.
        self.s_int_cat_ordered.rank()


class IsMonotonic:
    # [CLASS DOC]
    # Benchmark of the monotonicity properties of a categorical index and of a
    # series built from it. No parameters. State: c (index) and s (series).
    def setup(self):
        # [FUNCTION DOC]
        # Builds a categorical index of 1000 a values followed by 1000 b values
        # and 1000 c values (already sorted) as c, and a series from it as s.
        N = 1000
        self.c = pd.CategoricalIndex(list("a" * N + "b" * N + "c" * N))
        self.s = pd.Series(self.c)

    def time_categorical_index_is_monotonic_increasing(self):
        # [FUNCTION DOC]
        # Times reading the increasing-monotonic property of the index.
        self.c.is_monotonic_increasing

    def time_categorical_index_is_monotonic_decreasing(self):
        # [FUNCTION DOC]
        # Times reading the decreasing-monotonic property of the index.
        self.c.is_monotonic_decreasing

    def time_categorical_series_is_monotonic_increasing(self):
        # [FUNCTION DOC]
        # Times reading the increasing-monotonic property of the series.
        self.s.is_monotonic_increasing

    def time_categorical_series_is_monotonic_decreasing(self):
        # [FUNCTION DOC]
        # Times reading the decreasing-monotonic property of the series.
        self.s.is_monotonic_decreasing


class Contains:
    # [CLASS DOC]
    # Benchmark of the membership operator on a categorical index and on the
    # categorical array it wraps. No parameters. State: ci, c and key.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a categorical index over 100000 integers as ci, takes its values
        # as the categorical c, and the first category as key.
        N = 10**5
        self.ci = pd.CategoricalIndex(np.arange(N))
        self.c = self.ci.values
        self.key = self.ci.categories[0]

    def time_categorical_index_contains(self):
        # [FUNCTION DOC]
        # Times testing whether the key is in the categorical index.
        self.key in self.ci

    def time_categorical_contains(self):
        # [FUNCTION DOC]
        # Times testing whether the key is in the categorical array.
        self.key in self.c


class CategoricalSlicing:
    # [CLASS DOC]
    # Benchmark of selecting from a categorical of three million elements by
    # scalar, slice, list-like, list and boolean mask. Parametrized by the code
    # layout: monotonic increasing, monotonic decreasing or non-monotonic.
    # State: data, scalar (position 10000), list (first 10000 positions) and
    # cat_scalar (the label b).
    params = ["monotonic_incr", "monotonic_decr", "non_monotonic"]
    param_names = ["index"]

    def setup(self, index):
        # [FUNCTION DOC]
        # Builds integer codes for the categories a, b, c: each code repeated one
        # million times in increasing order, in decreasing order, or the sequence
        # tiled one million times. Raises a value error for any other layout name.
        # Creates the categorical from the codes and stores the helper selectors.
        N = 10**6
        categories = ["a", "b", "c"]
        if index == "monotonic_incr":
            codes = np.repeat([0, 1, 2], N)
        elif index == "monotonic_decr":
            codes = np.repeat([2, 1, 0], N)
        elif index == "non_monotonic":
            codes = np.tile([0, 1, 2], N)
        else:
            raise ValueError(f"Invalid index param: {index}")

        self.data = pd.Categorical.from_codes(codes, categories=categories)
        self.scalar = 10000
        self.list = list(range(10000))
        self.cat_scalar = "b"

    def time_getitem_scalar(self, index):
        # [FUNCTION DOC]
        # Times selecting the single element at the scalar position.
        self.data[self.scalar]

    def time_getitem_slice(self, index):
        # [FUNCTION DOC]
        # Times selecting the leading slice up to the scalar position.
        self.data[: self.scalar]

    def time_getitem_list_like(self, index):
        # [FUNCTION DOC]
        # Times selecting with a one-element list holding the scalar position.
        self.data[[self.scalar]]

    def time_getitem_list(self, index):
        # [FUNCTION DOC]
        # Times selecting with the list of the first 10000 positions.
        self.data[self.list]

    def time_getitem_bool_array(self, index):
        # [FUNCTION DOC]
        # Times selecting with a boolean mask from comparing the data with the
        # label b.
        self.data[self.data == self.cat_scalar]


class Indexing:
    # [CLASS DOC]
    # Benchmark of label lookups and set-like operations on a categorical index.
    # No parameters. State: index, series (sorted by index) and category.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a categorical index over 100000 integers (categories equal to the
        # values) as index, a series of the integers using it sorted by index as
        # series, and the element at position 500 as category.
        N = 10**5
        self.index = pd.CategoricalIndex(range(N), range(N))
        self.series = pd.Series(range(N), index=self.index).sort_index()
        self.category = self.index[500]

    def time_get_loc(self):
        # [FUNCTION DOC]
        # Times finding the position of one label in the index.
        self.index.get_loc(self.category)

    def time_shallow_copy(self):
        # [FUNCTION DOC]
        # Times creating a view of the index through the internal view method.
        self.index._view()

    def time_align(self):
        # [FUNCTION DOC]
        # Times building a frame from the full series and its first 500 rows,
        # which forces index alignment.
        pd.DataFrame({"a": self.series, "b": self.series[:500]})

    def time_intersection(self):
        # [FUNCTION DOC]
        # Times intersecting the first 750 index entries with entries from 250 on.
        self.index[:750].intersection(self.index[250:])

    def time_unique(self):
        # [FUNCTION DOC]
        # Times computing the unique values of the index.
        self.index.unique()

    def time_reindex(self):
        # [FUNCTION DOC]
        # Times reindexing the index to its first 500 entries.
        self.index.reindex(self.index[:500])

    def time_reindex_missing(self):
        # [FUNCTION DOC]
        # Times reindexing the index to four letters not present in it.
        self.index.reindex(["a", "b", "c", "d"])

    def time_sort_values(self):
        # [FUNCTION DOC]
        # Times sorting the index in descending order.
        self.index.sort_values(ascending=False)


class SearchSorted:
    # [CLASS DOC]
    # Benchmark of binary search on a sorted categorical index and on the
    # categorical array it wraps. No parameters. State: ci, c and key.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a sorted categorical index over 100000 integers as ci, takes its
        # values as c, and the second category as key.
        N = 10**5
        self.ci = pd.CategoricalIndex(np.arange(N)).sort_values()
        self.c = self.ci.values
        self.key = self.ci.categories[1]

    def time_categorical_index_contains(self):
        # [FUNCTION DOC]
        # Times a sorted search of the key in the categorical index.
        # KNOWN ISSUE: the method name says contains but it performs a sorted
        # search (name copied from the membership benchmark class).
        self.ci.searchsorted(self.key)

    def time_categorical_contains(self):
        # [FUNCTION DOC]
        # Times a sorted search of the key in the categorical array.
        # KNOWN ISSUE: the method name says contains but it performs a sorted
        # search (name copied from the membership benchmark class).
        self.c.searchsorted(self.key)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
