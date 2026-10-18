# [MODULE DOC]
# Benchmarks that stress the hash-table based routines (unique, factorize,
# group-by with float keys, label-based slicing on non-unique numeric indexes)
# with data that is awkward for hashing. Public contents are the benchmark classes
# UniqueForLargePyObjectInts, Float64GroupIndex, UniqueAndFactorizeArange, Unique,
# NumericSeriesIndexing and NumericSeriesIndexingShuffled. No import-time side
# effects.
import numpy as np

import pandas as pd


class UniqueForLargePyObjectInts:
    # [CLASS DOC]
    # Benchmark of computing unique values of an object array of large Python
    # integers (all with identical low bits). No parameters. State: arr.
    def setup(self):
        # [FUNCTION DOC]
        # Creates 5000 integers equal to the range values shifted left by 32 bits
        # and stores them as an object-dtype array arr.
        lst = [x << 32 for x in range(5000)]
        self.arr = np.array(lst, dtype=np.object_)

    def time_unique(self):
        # [FUNCTION DOC]
        # Times the top-level unique function on the array.
        pd.unique(self.arr)


class Float64GroupIndex:
    # [CLASS DOC]
    # Benchmark (issue 28303) of grouping a frame by float keys that are rounded
    # epoch-second values. No parameters. State: df and group_index.
    # GH28303
    def setup(self):
        # [FUNCTION DOC]
        # Builds a one-million-element datetime index spanning one day from
        # 2018-01-01 to 2018-01-02, converts it into a frame (the index becomes the
        # column and the index) as df, and computes group_index by converting the
        # index to integer nanoseconds, dividing by a billion and rounding (float
        # second values).
        self.df = pd.date_range(
            start="1/1/2018", end="1/2/2018", periods=10**6
        ).to_frame()
        self.group_index = np.round(self.df.index.astype(int) / 10**9)

    def time_groupby(self):
        # [FUNCTION DOC]
        # Times grouping the frame by the float keys and taking the last row of
        # each group.
        self.df.groupby(self.group_index).last()


class UniqueAndFactorizeArange:
    # [CLASS DOC]
    # Benchmark of factorize and unique on float arrays made of a range shifted by
    # ten to a power, each value repeated 100 times. Parametrized by the exponent
    # (4 to 15). State: a2.
    params = range(4, 16)
    param_names = ["exponent"]

    def setup(self, exponent):
        # [FUNCTION DOC]
        # Builds 10000 float values from zero, adds ten to the exponent, repeats
        # each value 100 times consecutively and stores the result as a2.
        a = np.arange(10**4, dtype="float64")
        self.a2 = (a + 10**exponent).repeat(100)

    def time_factorize(self, exponent):
        # [FUNCTION DOC]
        # Times the top-level factorize function on a2. The exponent parameter is
        # unused.
        pd.factorize(self.a2)

    def time_unique(self, exponent):
        # [FUNCTION DOC]
        # Times the top-level unique function on a2. The exponent parameter is
        # unused.
        pd.unique(self.a2)


class Unique:
    # [CLASS DOC]
    # Benchmark of computing unique values of nullable-integer and nullable-float
    # series. Parametrized by dtype name. State: ser (with duplicates) and
    # ser_unique (all distinct plus one missing value).
    params = ["Int64", "Float64"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds ser from the sequence one, missing, two, then the integers below
        # 100000, repeated three times (so with duplicates), and ser_unique from the
        # integers below 300000 followed by one missing value (no duplicates).
        # Both use the nullable dtype.
        self.ser = pd.Series(([1, pd.NA, 2, *range(100000)]) * 3, dtype=dtype)
        self.ser_unique = pd.Series([*range(300000), pd.NA], dtype=dtype)

    def time_unique_with_duplicates(self, exponent):
        # [FUNCTION DOC]
        # Times the top-level unique function on the series with duplicates.
        # The parameter (named exponent here, though it carries the dtype name) is
        # unused.
        pd.unique(self.ser)

    def time_unique(self, exponent):
        # [FUNCTION DOC]
        # Times the top-level unique function on the all-distinct series.
        # The parameter (named exponent here, though it carries the dtype name) is
        # unused.
        pd.unique(self.ser_unique)


class NumericSeriesIndexing:
    # [CLASS DOC]
    # Benchmark of label-based slicing on a series whose numeric index is
    # non-unique (one duplicated label) and in increasing order. Parametrized by
    # dtype and length. State: data.
    params = [
        (np.int64, np.uint64, np.float64),
        (10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6),
    ]
    param_names = ["dtype", "N"]

    def setup(self, dtype, N):
        # [FUNCTION DOC]
        # Builds index labels: the integers 0 to 54, a repeat of 54, then 55 up to
        # N minus 2, cast to dtype (length N, with 54 duplicated and the value N
        # minus 1 skipped). Creates a series of the integers below N indexed by them
        # and stores it as data.
        vals = np.array([*range(55), 54, *range(55, N - 1)], dtype=dtype)
        indices = pd.Index(vals)
        self.data = pd.Series(np.arange(N), index=indices)

    def time_loc_slice(self, index, N):
        # [FUNCTION DOC]
        # Times a label-based slice up to label 800, which forces building of the
        # index hash mapping. The first parameter (named index here, carrying the
        # dtype) and N are unused.
        # trigger building of mapping
        self.data.loc[:800]


class NumericSeriesIndexingShuffled:
    # [CLASS DOC]
    # Same as the previous slicing benchmark but with the index labels randomly
    # shuffled (so the index is neither unique nor monotonic). Parametrized by
    # dtype and length. State: data.
    params = [
        (np.int64, np.uint64, np.float64),
        (10**4, 10**5, 5 * 10**5, 10**6, 5 * 10**6),
    ]
    param_names = ["dtype", "N"]

    def setup(self, dtype, N):
        # [FUNCTION DOC]
        # Builds the same label array as the unshuffled variant, shuffles it in
        # place using the global random state, and creates a series of the integers
        # below N indexed by it; stores it as data.
        vals = np.array([*range(55), 54, *range(55, N - 1)], dtype=dtype)
        np.random.shuffle(vals)
        indices = pd.Index(vals)
        self.data = pd.Series(np.arange(N), index=indices)

    def time_loc_slice(self, index, N):
        # [FUNCTION DOC]
        # Times a label-based slice up to label 800, which forces building of the
        # index hash mapping. The first parameter (named index here, carrying the
        # dtype) and N are unused.
        # trigger building of mapping
        self.data.loc[:800]
