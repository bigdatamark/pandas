# [MODULE DOC]
# Benchmarks for the membership test method of Series across many dtypes, sizes
# and value distributions. Public contents are the benchmark classes IsIn,
# IsinAlmostFullWithRandomInt, IsinWithRandomFloat, IsinWithArangeSorted,
# IsinWithArange, IsInFloat64, IsInForObjects, IsInLongSeriesLookUpDominates,
# IsInLongSeriesValuesDominate, IsInWithLongTupples and IsInIndexes. Each class
# builds data in setup (using the global numpy random state, unseeded) and times
# the membership test in its time_ methods. Uses numpy and the top-level pandas
# namespace only. No import-time side effects.
import numpy as np

from pandas import (
    Categorical,
    Index,
    NaT,
    Series,
    date_range,
)


class IsIn:
    # [CLASS DOC]
    # Benchmark of Series membership testing across a wide set of dtypes
    # (numeric, object, nullable, boolean, datetime, categorical, string).
    # Parametrized by dtype name. State set in setup: series (the data), values
    # (the candidate values to test against), mismatched (values of a mismatched
    # type) and cat_values (a categorical built from values).
    params = [
        "int64",
        "uint64",
        "object",
        "Int64",
        "boolean",
        "bool",
        "datetime64[ns]",
        "category[object]",
        "category[int]",
        "str",
        "string[python]",
        "string[pyarrow]",
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds the benchmark inputs for one dtype name.
        # Input: dtype, a string naming the target dtype. Output: none; sets the
        # series, values, mismatched and cat_values attributes. The size is 10000
        # elements by default; datetime uses a 115777-element range, categorical
        # uses 500000 elements with 100 sampled values. Boolean dtypes use random
        # zeros and ones and the values true and false. Datetime uses every third
        # element as values and integers as the mismatched values. String dtypes
        # build from object strings and use the first two elements as values; if
        # the string backend is not installed an import error is converted into a
        # not-implemented error so asv skips the case. Any other dtype uses random
        # integers from one to nine with the values one and two. The mismatched
        # default is a pair of NaT converted to a datetime64 scalar.
        N = 10000

        self.mismatched = [NaT.to_datetime64()] * 2

        if dtype in ["boolean", "bool"]:
            self.series = Series(np.random.randint(0, 2, N)).astype(dtype)
            self.values = [True, False]

        elif dtype == "datetime64[ns]":
            # Note: values here is much larger than non-dt64ns cases

            # dti has length=115777
            dti = date_range(start="2015-10-26", end="2016-01-01", freq="50s")
            self.series = Series(dti)
            self.values = self.series._values[::3]
            self.mismatched = [1, 2]

        elif dtype in ["category[object]", "category[int]"]:
            # Note: sizes are different in this case than others
            n = 5 * 10**5
            sample_size = 100

            arr = list(np.random.randint(0, n // 10, size=n))
            if dtype == "category[object]":
                arr = [f"s{i:04d}" for i in arr]

            self.values = np.random.choice(arr, sample_size)
            self.series = Series(arr).astype("category")

        elif dtype in ["str", "string[python]", "string[pyarrow]"]:
            try:
                self.series = Series(
                    Index([f"i-{i}" for i in range(N)], dtype=object)._values,
                    dtype=dtype,
                )
            except ImportError as err:
                raise NotImplementedError from err
            self.values = list(self.series[:2])

        else:
            self.series = Series(np.random.randint(1, 10, N)).astype(dtype)
            self.values = [1, 2]

        self.cat_values = Categorical(self.values)

    def time_isin(self, dtype):
        # [FUNCTION DOC]
        # Times membership testing of the series against the list-like values.
        # The dtype parameter is unused. Result is discarded.
        self.series.isin(self.values)

    def time_isin_categorical(self, dtype):
        # [FUNCTION DOC]
        # Times membership testing of the series against a categorical built from
        # the values. The dtype parameter is unused. Result is discarded.
        self.series.isin(self.cat_values)

    def time_isin_empty(self, dtype):
        # [FUNCTION DOC]
        # Times membership testing against an empty list. The dtype parameter is
        # unused. Result is discarded.
        self.series.isin([])

    def time_isin_mismatched_dtype(self, dtype):
        # [FUNCTION DOC]
        # Times membership testing against values whose type does not match the
        # series dtype (datetime-like nulls or plain integers). The dtype
        # parameter is unused. Result is discarded.
        self.series.isin(self.mismatched)


class IsinAlmostFullWithRandomInt:
    # [CLASS DOC]
    # Benchmark of membership testing where the hash table built from the values
    # is nearly full (about 0.77 load, the maximum share of occupied buckets).
    # Parametrized by numeric or object dtype, a size exponent from ten to twenty
    # and whether the values lie inside or outside the series range. State: series
    # and values.
    params = [
        [np.float64, np.int64, np.uint64, np.object_],
        range(10, 21),
        ["inside", "outside"],
    ]
    param_names = ["dtype", "exponent", "title"]

    def setup(self, dtype, exponent, title):
        # [FUNCTION DOC]
        # Builds a series of random integers with size three times two to the
        # power of exponent minus two, cast to dtype, and a values array of the
        # same size. For title inside the values overlap the series range; for
        # outside they are shifted by the size so none can match. Raises a value
        # error for any other title. Sets series and values.
        M = 3 * 2 ** (exponent - 2)
        # 0.77-the maximal share of occupied buckets
        self.series = Series(np.random.randint(0, M, M)).astype(dtype)

        values = np.random.randint(0, M, M).astype(dtype)
        if title == "inside":
            self.values = values
        elif title == "outside":
            self.values = values + M
        else:
            raise ValueError(title)

    def time_isin(self, dtype, exponent, title):
        # [FUNCTION DOC]
        # Times membership testing of the series against the values array. All
        # parameters are unused here. Result is discarded.
        self.series.isin(self.values)


class IsinWithRandomFloat:
    # [CLASS DOC]
    # Benchmark of membership testing for random floating point data of various
    # sizes, as float or object dtype, with values that either match (inside) or
    # are shifted so they do not match (outside). State: series and values.
    params = [
        [np.float64, np.object_],
        [
            1_300,
            2_000,
            7_000,
            8_000,
            70_000,
            80_000,
            750_000,
            900_000,
        ],
        ["inside", "outside"],
    ]
    param_names = ["dtype", "size", "title"]

    def setup(self, dtype, size, title):
        # [FUNCTION DOC]
        # Creates size uniform random floats as values, builds the series from
        # them cast to dtype, then shuffles the values in place (the series is
        # built before shuffling, so it keeps the original order). For title
        # outside the values are shifted by a tenth so none match. Any other
        # title behaves like inside. Sets series and values.
        self.values = np.random.rand(size)
        self.series = Series(self.values).astype(dtype)
        np.random.shuffle(self.values)

        if title == "outside":
            self.values = self.values + 0.1

    def time_isin(self, dtype, size, title):
        # [FUNCTION DOC]
        # Times membership testing of the series against the values array. All
        # parameters are unused here. Result is discarded.
        self.series.isin(self.values)


class IsinWithArangeSorted:
    # [CLASS DOC]
    # Benchmark of membership testing where both series and values are the same
    # sorted range of integers cast to a dtype. Parametrized by dtype and size.
    # State: series and values.
    params = [
        [np.float64, np.int64, np.uint64, np.object_],
        [
            1_000,
            2_000,
            8_000,
            100_000,
            1_000_000,
        ],
    ]
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        # [FUNCTION DOC]
        # Sets series and values to the integer range of the given size, each cast
        # to dtype.
        self.series = Series(np.arange(size)).astype(dtype)
        self.values = np.arange(size).astype(dtype)

    def time_isin(self, dtype, size):
        # [FUNCTION DOC]
        # Times membership testing of the series against the values array.
        # Parameters are unused. Result is discarded.
        self.series.isin(self.values)


class IsinWithArange:
    # [CLASS DOC]
    # Benchmark of membership testing of one million random integers against a
    # small range of values, where the series range is shifted relative to the
    # values range by a multiple of the values size. Parametrized by dtype, the
    # values size and the offset factor (negative, zero, positive). State: series
    # and values.
    params = [
        [np.float64, np.int64, np.uint64, np.object_],
        [
            1_000,
            2_000,
            8_000,
        ],
        [-2, 0, 2],
    ]
    param_names = ["dtype", "M", "offset_factor"]

    def setup(self, dtype, M, offset_factor):
        # [FUNCTION DOC]
        # Computes an integer offset as M times offset_factor, draws one million
        # random integers from the offset up to M plus offset as the series (cast
        # to dtype) and uses the range from zero to M cast to dtype as values.
        # Sets series and values.
        offset = int(M * offset_factor)
        tmp = Series(np.random.randint(offset, M + offset, 10**6))
        self.series = tmp.astype(dtype)
        self.values = np.arange(M).astype(dtype)

    def time_isin(self, dtype, M, offset_factor):
        # [FUNCTION DOC]
        # Times membership testing of the series against the values array.
        # Parameters are unused. Result is discarded.
        self.series.isin(self.values)


class IsInFloat64:
    # [CLASS DOC]
    # Benchmark of membership testing for a tiny float series against large
    # float value arrays, so runtime is dominated by building the lookup table.
    # Parametrized by dtype (numpy float or nullable Float64) and a title
    # selecting the values: many distinct values, few distinct values (all
    # zeros) or only missing values. State: series and values.
    params = [
        [np.float64, "Float64"],
        ["many_different_values", "few_different_values", "only_nans_values"],
    ]
    param_names = ["dtype", "title"]

    def setup(self, dtype, title):
        # [FUNCTION DOC]
        # Builds a two-element series (one and two) of the given dtype and a
        # float values array selected by title: a range of 100000 values, a
        # million zeros, or a million missing values. Raises a value error for an
        # unknown title. Sets series and values.
        N_many = 10**5
        N_few = 10**6
        self.series = Series([1, 2], dtype=dtype)

        if title == "many_different_values":
            # runtime is dominated by creation of the lookup-table
            self.values = np.arange(N_many, dtype=np.float64)
        elif title == "few_different_values":
            # runtime is dominated by creation of the lookup-table
            self.values = np.zeros(N_few, dtype=np.float64)
        elif title == "only_nans_values":
            # runtime is dominated by creation of the lookup-table
            self.values = np.full(N_few, np.nan, dtype=np.float64)
        else:
            raise ValueError(title)

    def time_isin(self, dtype, title):
        # [FUNCTION DOC]
        # Times membership testing of the series against the values array.
        # Parameters are unused. Result is discarded.
        self.series.isin(self.values)


class IsInForObjects:
    # [CLASS DOC]
    # Benchmark of membership testing on object-dtype data for every pairing of
    # four variants of series content and values content (nans, short, long,
    # long_floats). State: series (object dtype) and values (object array).
    """
    A subset of the cartesian product of cases have special motivations:

    "nans" x "nans"
        if nan-objects are different objects,
        this has the potential to trigger O(n^2) running time

    "short" x "long"
        running time dominated by the preprocessing

    "long" x "short"
        running time dominated by look-up

    "long" x "long"
        no dominating part

    "long_floats" x "long_floats"
        because of nans floats are special
        no dominating part

    """

    variants = ["nans", "short", "long", "long_floats"]

    params = [variants, variants]
    param_names = ["series_type", "vals_type"]

    def setup(self, series_type, vals_type):
        # [FUNCTION DOC]
        # Builds the object-dtype series from the variant series_type and the
        # object values array from the variant vals_type. Variants: nans is ten
        # thousand missing floats, short is the integers zero and one, long is
        # a range of 100000 integers, long_floats is the same range as floats.
        # KNOWN ISSUE: there is no fallback branch, so an unrecognized variant
        # leaves the local array unset and would raise an unbound-variable error;
        # in practice only the four listed variants are ever passed.
        N_many = 10**5

        if series_type == "nans":
            ser_vals = np.full(10**4, np.nan)
        elif series_type == "short":
            ser_vals = np.arange(2)
        elif series_type == "long":
            ser_vals = np.arange(N_many)
        elif series_type == "long_floats":
            ser_vals = np.arange(N_many, dtype=np.float64)

        self.series = Series(ser_vals).astype(object)

        if vals_type == "nans":
            values = np.full(10**4, np.nan)
        elif vals_type == "short":
            values = np.arange(2)
        elif vals_type == "long":
            values = np.arange(N_many)
        elif vals_type == "long_floats":
            values = np.arange(N_many, dtype=np.float64)

        self.values = values.astype(object)

    def time_isin(self, series_type, vals_type):
        # [FUNCTION DOC]
        # Times membership testing of the object series against the object
        # values. Parameters are unused. Result is discarded.
        self.series.isin(self.values)


class IsInLongSeriesLookUpDominates:
    # [CLASS DOC]
    # Benchmark of membership testing of a ten-million element series against a
    # small set of values, so runtime is dominated by lookups. Parametrized by
    # dtype name (numpy and nullable), the number of distinct values
    # (MaxNumber) and the series pattern (random or monotone, hits or misses).
    # State: series and values.
    params = [
        ["int64", "int32", "float64", "float32", "object", "Int64", "Float64"],
        [5, 1000],
        ["random_hits", "random_misses", "monotone_hits", "monotone_misses"],
    ]
    param_names = ["dtype", "MaxNumber", "series_type"]

    def setup(self, dtype, MaxNumber, series_type):
        # [FUNCTION DOC]
        # Creates ten million integers according to series_type: random hits are
        # drawn below MaxNumber; random misses are those shifted up by MaxNumber;
        # monotone hits repeat each value of the range up to MaxNumber evenly;
        # monotone misses are an increasing range shifted by MaxNumber. The series
        # is cast to dtype and the values are the range below MaxNumber cast to
        # the lowercase form of the dtype name (so the nullable names map to the
        # numpy dtypes). The conditions are independent, not exclusive; an
        # unknown series_type would leave the array unset. Sets series and values.
        N = 10**7

        if series_type == "random_hits":
            array = np.random.randint(0, MaxNumber, N)
        if series_type == "random_misses":
            array = np.random.randint(0, MaxNumber, N) + MaxNumber
        if series_type == "monotone_hits":
            array = np.repeat(np.arange(MaxNumber), N // MaxNumber)
        if series_type == "monotone_misses":
            array = np.arange(N) + MaxNumber

        self.series = Series(array).astype(dtype)

        self.values = np.arange(MaxNumber).astype(dtype.lower())

    def time_isin(self, dtypes, MaxNumber, series_type):
        # [FUNCTION DOC]
        # Times membership testing of the long series against the small values
        # array. Parameters are unused (the first is named differently from the
        # parameter name list but is passed positionally). Result is discarded.
        self.series.isin(self.values)


class IsInLongSeriesValuesDominate:
    # [CLASS DOC]
    # Benchmark of membership testing of a one-million element series against
    # a ten-million element values array, so runtime is dominated by building the
    # table from the values. Parametrized by dtype name and a random or monotone
    # values pattern. State: series and values.
    params = [
        ["int64", "int32", "float64", "float32", "object", "Int64", "Float64"],
        ["random", "monotone"],
    ]
    param_names = ["dtype", "series_type"]

    def setup(self, dtype, series_type):
        # [FUNCTION DOC]
        # Creates ten million values: random integers below ten times that count,
        # or an increasing range, cast to the lowercase form of the dtype name.
        # The series is a range of 1000001 integers cast to dtype. An unknown
        # series_type would leave the array unset. Sets series and values.
        N = 10**7

        if series_type == "random":
            vals = np.random.randint(0, 10 * N, N)
        if series_type == "monotone":
            vals = np.arange(N)

        self.values = vals.astype(dtype.lower())
        M = 10**6 + 1
        self.series = Series(np.arange(M)).astype(dtype)

    def time_isin(self, dtypes, series_type):
        # [FUNCTION DOC]
        # Times membership testing of the series against the large values array.
        # Parameters are unused. Result is discarded.
        self.series.isin(self.values)


class IsInWithLongTupples:
    # [CLASS DOC]
    # Benchmark of membership testing for a series of identical long tuples
    # against a values list containing that tuple. No parameters. State: series
    # and values.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a tuple of the integers below one thousand, a series holding one
        # thousand copies of it, and a values list containing the tuple. Sets
        # series and values.
        t = tuple(range(1000))
        self.series = Series([t] * 1000)
        self.values = [t]

    def time_isin(self):
        # [FUNCTION DOC]
        # Times membership testing of the tuple series against the list of tuples.
        # Result is discarded.
        self.series.isin(self.values)


class IsInIndexes:
    # [CLASS DOC]
    # Benchmark of membership testing of an integer series against Index objects
    # (a range index and a regular integer index). No parameters. State:
    # range_idx, index and series.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a range-based index and an integer index, both over zero to 999,
        # and a series of one thousand random integers below one hundred
        # thousand. Sets range_idx, index and series.
        self.range_idx = Index(range(1000))
        self.index = Index(list(range(1000)))
        self.series = Series(np.random.randint(100_000, size=1000))

    def time_isin_range_index(self):
        # [FUNCTION DOC]
        # Times membership testing of the series against the range index. Result
        # is discarded.
        self.series.isin(self.range_idx)

    def time_isin_index(self):
        # [FUNCTION DOC]
        # Times membership testing of the series against the integer index.
        # Result is discarded.
        self.series.isin(self.index)
