# [MODULE DOC]
# Benchmarks for constructing Series, Index, DatetimeIndex and MultiIndex
# objects from a variety of input containers. Public contents are the benchmark
# classes SeriesConstructors, SeriesDtypesConstructors, MultiIndexConstructor and
# DatetimeIndexConstructor, plus module-level converter functions (no_change,
# list_of_str, gen_of_str, arr_dict, list_of_tuples, gen_of_tuples, list_of_lists,
# list_of_tuples_with_none, list_of_lists_with_none) that turn a numpy array into
# differently shaped input data for the Series constructor benchmark. No
# import-time side effects other than importing; the final line re-exports the
# common asv setup hook.
import numpy as np

from pandas import (
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)


def no_change(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns the given array unchanged.
    return arr


def list_of_str(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Converts the array elements to their text form and returns them as a list
    # of strings.
    return list(arr.astype(str))


def gen_of_str(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a one-shot generator that yields the elements of the array
    # converted to text. The generator is exhausted after one use.
    return (x for x in arr.astype(str))


def arr_dict(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a mapping from each position (zero up to the array length) to the
    # element at that position. Uses strict pairing, which would raise a value
    # error if the two sequences had different lengths (they cannot here).
    return dict(zip(range(len(arr)), arr, strict=True))


def list_of_tuples(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a list of pairs, each holding an element and its negation.
    return [(i, -i) for i in arr]


def gen_of_tuples(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a one-shot generator of pairs, each holding an element and its
    # negation.
    return ((i, -i) for i in arr)


def list_of_lists(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a list of two-element lists, each holding an element and its
    # negation.
    return [[i, -i] for i in arr]


def list_of_tuples_with_none(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a list of element/negation pairs for all but the last element, with
    # the last slot replaced by a none value.
    return [*[(i, -i) for i in arr][:-1], None]


def list_of_lists_with_none(arr):
    # [FUNCTION DOC]
    # HELPER: input-format converter for the series constructor benchmark.
    # Returns a list of two-element lists for all but the last element, with the
    # last slot replaced by a none value.
    return [*[[i, -i] for i in arr][:-1], None]


class SeriesConstructors:
    # [CLASS DOC]
    # Benchmark of the Series constructor for many input container formats, with
    # and without an explicit index, for float and integer source data.
    # Parametrized by data format converter, a with_index flag and dtype kind.
    # Because generator inputs get exhausted on use, the class sets the call
    # count per measurement to one and a repeat tuple so setup runs before every
    # call. State: data and index.
    param_names = ["data_fmt", "with_index", "dtype"]
    params = [
        [
            no_change,
            list,
            list_of_str,
            gen_of_str,
            arr_dict,
            list_of_tuples,
            gen_of_tuples,
            list_of_lists,
            list_of_tuples_with_none,
            list_of_lists_with_none,
        ],
        [False, True],
        ["float", "int"],
    ]

    # Generators get exhausted on use, so run setup before every call
    number = 1
    repeat = (3, 250, 10)

    def setup(self, data_fmt, with_index, dtype):
        # [FUNCTION DOC]
        # Builds the constructor inputs. Raises a not-implemented error (which
        # asv treats as a skip) when a generator format is combined with an index,
        # since the constructor does not support that. Creates 10000 random normal
        # floats when dtype is float, otherwise a range of 10000 integers, applies
        # the converter to get data, and sets index to a range of 10000 integers
        # when with_index is true, otherwise none.
        if data_fmt in (gen_of_str, gen_of_tuples) and with_index:
            raise NotImplementedError(
                "Series constructors do not support using generators with indexes"
            )
        N = 10**4
        if dtype == "float":
            arr = np.random.randn(N)
        else:
            arr = np.arange(N)
        self.data = data_fmt(arr)
        self.index = np.arange(N) if with_index else None

    def time_series_constructor(self, data_fmt, with_index, dtype):
        # [FUNCTION DOC]
        # Times constructing a series from the prepared data and index. The
        # parameters are unused here. Result is discarded.
        Series(self.data, index=self.index)


class SeriesDtypesConstructors:
    # [CLASS DOC]
    # Benchmark of constructing Index and DatetimeIndex objects from arrays and
    # series of various dtypes. No parameters. State: arr, arr_str and s.
    def setup(self):
        # [FUNCTION DOC]
        # With N equal to 10000: creates arr (random normal floats), arr_str (an
        # object array of the three strings foo, bar, baz) and s (a series of three
        # repeating timestamps from 2011, 2012 and 2013 repeated ten times N, i.e.
        # 300000 elements).
        N = 10**4
        self.arr = np.random.randn(N)
        self.arr_str = np.array(["foo", "bar", "baz"], dtype=object)
        self.s = Series(
            [Timestamp("20110101"), Timestamp("20120101"), Timestamp("20130101")]
            * N
            * 10
        )

    def time_index_from_array_string(self):
        # [FUNCTION DOC]
        # Times building an index from the string object array.
        Index(self.arr_str)

    def time_index_from_array_floats(self):
        # [FUNCTION DOC]
        # Times building an index from the float array.
        Index(self.arr)

    def time_dtindex_from_series(self):
        # [FUNCTION DOC]
        # Times building a datetime index from the timestamp series.
        DatetimeIndex(self.s)

    def time_dtindex_from_index_with_series(self):
        # [FUNCTION DOC]
        # Times building a general index from the timestamp series, which infers
        # a datetime index.
        Index(self.s)


class MultiIndexConstructor:
    # [CLASS DOC]
    # Benchmark of building a multi-index as the Cartesian product of two
    # iterables. No parameters. State: iterables.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a list of two iterables: an object index of 10000 strings and a
        # range of twenty integers; stores it as iterables.
        N = 10**4
        self.iterables = [Index([f"i-{i}" for i in range(N)], dtype=object), range(20)]

    def time_multiindex_from_iterables(self):
        # [FUNCTION DOC]
        # Times building the product multi-index from the iterables.
        MultiIndex.from_product(self.iterables)


class DatetimeIndexConstructor:
    # [CLASS DOC]
    # Benchmark of building a datetime index from lists of timestamps, dates,
    # datetimes and date strings. No parameters. State: list_of_timestamps,
    # list_of_dates, list_of_datetimes and list_of_str.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a daily datetime index of 20000 entries from 1900-01-01 and
        # derives four lists from it: timestamps, calendar dates, native datetime
        # objects and strings in year-month-day format.
        N = 20_000
        dti = date_range("1900-01-01", periods=N)

        self.list_of_timestamps = dti.tolist()
        self.list_of_dates = dti.date.tolist()
        self.list_of_datetimes = dti.to_pydatetime().tolist()
        self.list_of_str = dti.strftime("%Y-%m-%d").tolist()

    def time_from_list_of_timestamps(self):
        # [FUNCTION DOC]
        # Times building a datetime index from the list of timestamps.
        DatetimeIndex(self.list_of_timestamps)

    def time_from_list_of_dates(self):
        # [FUNCTION DOC]
        # Times building a datetime index from the list of date objects.
        DatetimeIndex(self.list_of_dates)

    def time_from_list_of_datetimes(self):
        # [FUNCTION DOC]
        # Times building a datetime index from the list of native datetimes.
        DatetimeIndex(self.list_of_datetimes)

    def time_from_list_of_str(self):
        # [FUNCTION DOC]
        # Times building a datetime index from the list of date strings.
        DatetimeIndex(self.list_of_str)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
