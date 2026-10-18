# [MODULE DOC]
# Benchmarks for Series methods and construction: construction from dict or no
# data, frame conversion, n-largest/smallest, missing-value dropping and filling,
# sorted search, mapping, clipping, value counts and mode, attribute listing and
# lookup, boolean reductions, numeric reductions, ranking, iteration, numpy
# conversion and replacement. Public contents are the benchmark classes
# SeriesConstructor, ToFrame, NSort, Dropna, Fillna, SearchSorted, Map, Clip,
# ClipDt, ValueCounts, ValueCountsEA, ValueCountsObjectDropNAFalse, Mode,
# ModeObjectDropNAFalse, Dir, SeriesGetattr, All, Any, NanOps, Rank, Iter, ToNumpy
# and Replace. No import-time side effects; the final line re-exports the common
# asv setup hook.
from datetime import datetime

import numpy as np

from pandas import (
    NA,
    Index,
    NaT,
    Series,
    date_range,
)


class SeriesConstructor:
    # [CLASS DOC]
    # Benchmark of building a series from a dictionary with a datetime index and
    # from no data. No parameters. State: idx, data, array and idx2.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a datetime index from 2015-10-26 to 2016-01-01 every 50 seconds
        # (115777 entries) as idx, a dictionary mapping each timestamp to its
        # position as data, a three-element integer array and a three-letter index.
        # The array and idx2 attributes are not used by any timing method.
        self.idx = date_range(
            start=datetime(2015, 10, 26), end=datetime(2016, 1, 1), freq="50s"
        )
        self.data = dict(zip(self.idx, range(len(self.idx)), strict=True))
        self.array = np.array([1, 2, 3])
        self.idx2 = Index(["a", "b", "c"])

    def time_constructor_dict(self):
        # [FUNCTION DOC]
        # Times building a series from the timestamp-keyed dictionary with the same
        # index supplied.
        Series(data=self.data, index=self.idx)

    def time_constructor_no_data(self):
        # [FUNCTION DOC]
        # Times building an all-missing series from no data on the index.
        Series(data=None, index=self.idx)


class ToFrame:
    # [CLASS DOC]
    # Benchmark of converting a series to a one-column frame with an optional
    # column name. Parametrized by dtype (int64, datetime64[ns], category, Int64)
    # and name (none or foo). State: ser.
    params = [["int64", "datetime64[ns]", "category", "Int64"], [None, "foo"]]
    param_names = ["dtype", "name"]

    def setup(self, dtype, name):
        # [FUNCTION DOC]
        # Builds a series from the integers below 100000 converted to the dtype
        # (for datetime, interpreted as nanosecond offsets). The name parameter is
        # unused here.
        arr = np.arange(10**5)
        ser = Series(arr, dtype=dtype)
        self.ser = ser

    def time_to_frame(self, dtype, name):
        # [FUNCTION DOC]
        # Times converting the series to a frame named by the name parameter (or
        # the default when none).
        self.ser.to_frame(name)


class NSort:
    # [CLASS DOC]
    # Benchmark of selecting the three largest or smallest values with different
    # tie policies. Parametrized by keep (first, last, all). State: s.
    params = ["first", "last", "all"]
    param_names = ["keep"]

    def setup(self, keep):
        # [FUNCTION DOC]
        # Builds a series of 100000 random integers from 1 to 9 (many ties). The keep
        # parameter is unused here.
        self.s = Series(np.random.randint(1, 10, 100000))

    def time_nlargest(self, keep):
        # [FUNCTION DOC]
        # Times selecting the three largest values with the tie policy.
        self.s.nlargest(3, keep=keep)

    def time_nsmallest(self, keep):
        # [FUNCTION DOC]
        # Times selecting the three smallest values with the tie policy.
        self.s.nsmallest(3, keep=keep)


class Dropna:
    # [CLASS DOC]
    # Benchmark of dropping missing values from a million-element integer or
    # datetime series (the datetime series has a hundred missing entries).
    # Parametrized by dtype kind (int, datetime). State: s.
    params = ["int", "datetime"]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a series of a million random integers from 1 to 9, or a million
        # per-second timestamps from 2000-01-01. For the datetime kind, sets 100
        # randomly chosen positions to missing (not-a-time). Stores as s.
        N = 10**6
        data = {
            "int": np.random.randint(1, 10, N),
            "datetime": date_range("2000-01-01", freq="s", periods=N),
        }
        self.s = Series(data[dtype])
        if dtype == "datetime":
            self.s[np.random.randint(1, N, 100)] = NaT

    def time_dropna(self, dtype):
        # [FUNCTION DOC]
        # Times dropping missing values.
        self.s.dropna()


class Fillna:
    # [CLASS DOC]
    # Benchmark of constant, forward and backward filling of a million-element
    # series in which every second value is missing. Parametrized by dtype name
    # (datetime, floats, nullable, arrow integer, strings). State: ser and
    # fill_value.
    params = [
        [
            "datetime64[ns]",
            "float32",
            "float64",
            "Float64",
            "Int64",
            "int64[pyarrow]",
            "string",
            "string[pyarrow]",
        ],
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a million values suited to the dtype (per-second timestamps with
        # not-a-time as the missing marker; random normal floats with NaN; integers
        # with the NA marker; five-times repeated decimal strings with NA). Raises a
        # not-implemented error for any other dtype. Takes the first value as the
        # fill value, creates the series in the dtype, sets every second element to
        # the missing marker, and stores as ser and fill_value.
        # KNOWN ISSUE: the float32 dtype is listed as a parameter but no branch
        # handles it (only float64 and the nullable float are handled), so that
        # parameter always raises the not-implemented error and is skipped.
        N = 10**6
        if dtype == "datetime64[ns]":
            data = date_range("2000-01-01", freq="s", periods=N)
            na_value = NaT
        elif dtype in ("float64", "Float64"):
            data = np.random.randn(N)
            na_value = np.nan
        elif dtype in ("Int64", "int64[pyarrow]"):
            data = np.arange(N)
            na_value = NA
        elif dtype in ("string", "string[pyarrow]"):
            data = np.array([str(i) * 5 for i in range(N)], dtype=object)
            na_value = NA
        else:
            raise NotImplementedError
        fill_value = data[0]
        ser = Series(data, dtype=dtype)
        ser[::2] = na_value
        self.ser = ser
        self.fill_value = fill_value

    def time_fillna(self, dtype):
        # [FUNCTION DOC]
        # Times filling missing values with the stored fill value.
        self.ser.fillna(value=self.fill_value)

    def time_ffill(self, dtype):
        # [FUNCTION DOC]
        # Times forward filling.
        self.ser.ffill()

    def time_bfill(self, dtype):
        # [FUNCTION DOC]
        # Times backward filling.
        self.ser.bfill()


class SearchSorted:
    # [CLASS DOC]
    # Benchmark of binary search on a sorted series for many numeric dtypes and
    # strings. Parametrized by dtype name. The goal_time attribute (0.2) is a
    # legacy asv setting. State: s.
    goal_time = 0.2
    params = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "str",
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a sorted series of 300000 values: 100000 each of one, two, three,
        # converted to the dtype.
        N = 10**5
        data = np.array([1] * N + [2] * N + [3] * N).astype(dtype)
        self.s = Series(data)

    def time_searchsorted(self, dtype):
        # [FUNCTION DOC]
        # Times the sorted search of the value two (as text for the string dtype).
        key = "2" if dtype == "str" else 2
        self.s.searchsorted(key)


class Map:
    # [CLASS DOC]
    # Benchmark of mapping series values through a dictionary, a series or a
    # function, with and without ignoring missing values. Parametrized by mapper
    # kind, dtype (object, category, int) and na_action. State: map_data (the
    # mapper) and s.
    params = (
        ["dict", "Series", "lambda"],
        ["object", "category", "int"],
        [None, "ignore"],
    )
    param_names = ["mapper", "dtype", "na_action"]

    def setup(self, mapper, dtype, na_action):
        # [FUNCTION DOC]
        # Builds a mapping source of 1000 values from 1000 down to 1 in the dtype
        # and converts it to the mapper kind: the series itself, a dictionary, or a
        # function that looks up the value in the dictionary (raises a not-implemented
        # error for other kinds). Builds s as 10000 random integers below 1000 in the
        # dtype. The na_action parameter is unused here.
        map_size = 1000
        map_data = Series(map_size - np.arange(map_size), dtype=dtype)

        # construct mapper
        if mapper == "Series":
            self.map_data = map_data
        elif mapper == "dict":
            self.map_data = map_data.to_dict()
        elif mapper == "lambda":
            map_dict = map_data.to_dict()
            self.map_data = lambda x: map_dict[x]
        else:
            raise NotImplementedError

        self.s = Series(np.random.randint(0, map_size, 10000), dtype=dtype)

    def time_map(self, mapper, dtype, na_action):
        # [FUNCTION DOC]
        # Times mapping the series values through the mapper with the missing-value
        # policy.
        self.s.map(self.map_data, na_action=na_action)


class Clip:
    # [CLASS DOC]
    # Benchmark of clipping a random series to the interval from zero to one.
    # Parametrized by length (50, 1000, 100000). State: s.
    params = [50, 1000, 10**5]
    param_names = ["n"]

    def setup(self, n):
        # [FUNCTION DOC]
        # Builds a random normal series of length n.
        self.s = Series(np.random.randn(n))

    def time_clip(self, n):
        # [FUNCTION DOC]
        # Times clipping the series to the range zero to one.
        self.s.clip(0, 1)


class ClipDt:
    # [CLASS DOC]
    # Benchmark of clipping a timezone-aware datetime series with an element-wise
    # upper bound. No parameters. State: clipper_dt and s.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 100000 per-second UTC timestamps from 2022-01-01 as s, and the first
        # 1000 of them each repeated 100 times (length 100000) as the upper bound
        # clipper_dt.
        dr = date_range("20220101", periods=100_000, freq="s", tz="UTC")
        self.clipper_dt = dr[0:1_000].repeat(100)
        self.s = Series(dr)

    def time_clip(self):
        # [FUNCTION DOC]
        # Times clipping the series with the element-wise upper bound.
        self.s.clip(upper=self.clipper_dt)


class ValueCounts:
    # [CLASS DOC]
    # Benchmark of counting occurrences of each distinct value for several dtypes
    # and distinct-value counts. Parametrized by N (1000, 10000, 100000) and dtype
    # (int, uint, float, object). State: s with 10 times N elements.
    params = [[10**3, 10**4, 10**5], ["int", "uint", "float", "object"]]
    param_names = ["N", "dtype"]

    def setup(self, N, dtype):
        # [FUNCTION DOC]
        # Builds a series of 10 times N random integers below N, converted to the
        # dtype.
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype(dtype)

    def time_value_counts(self, N, dtype):
        # [FUNCTION DOC]
        # Times counting the distinct values.
        self.s.value_counts()


class ValueCountsEA:
    # [CLASS DOC]
    # Benchmark of counting values in a nullable-integer series that has one
    # missing value, with missing values dropped or kept. Parametrized by N and the
    # dropna flag. State: s.
    params = [[10**3, 10**4, 10**5], [True, False]]
    param_names = ["N", "dropna"]

    def setup(self, N, dropna):
        # [FUNCTION DOC]
        # Builds a nullable 64-bit integer series of 10 times N random integers
        # below N and sets the element at label one to missing. The dropna flag is
        # unused here.
        self.s = Series(np.random.randint(0, N, size=10 * N), dtype="Int64")
        self.s.loc[1] = NA

    def time_value_counts(self, N, dropna):
        # [FUNCTION DOC]
        # Times counting the distinct values with the missing-value policy.
        self.s.value_counts(dropna=dropna)


class ValueCountsObjectDropNAFalse:
    # [CLASS DOC]
    # Benchmark of counting values in an object-dtype series with missing values
    # kept. Parametrized by N. State: s.
    params = [10**3, 10**4, 10**5]
    param_names = ["N"]

    def setup(self, N):
        # [FUNCTION DOC]
        # Builds an object-dtype series of 10 times N random integers below N.
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype("object")

    def time_value_counts(self, N):
        # [FUNCTION DOC]
        # Times counting the distinct values, keeping missing values in the result.
        self.s.value_counts(dropna=False)


class Mode:
    # [CLASS DOC]
    # Benchmark of computing the most frequent values for several dtypes and
    # distinct-value counts. Parametrized by N and dtype (int, uint, float,
    # object). State: s with 10 times N elements.
    params = [[10**3, 10**4, 10**5], ["int", "uint", "float", "object"]]
    param_names = ["N", "dtype"]

    def setup(self, N, dtype):
        # [FUNCTION DOC]
        # Builds a series of 10 times N random integers below N, converted to the
        # dtype.
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype(dtype)

    def time_mode(self, N, dtype):
        # [FUNCTION DOC]
        # Times computing the most frequent values.
        self.s.mode()


class ModeObjectDropNAFalse:
    # [CLASS DOC]
    # Benchmark of computing the most frequent values of an object-dtype series
    # with missing values kept. Parametrized by N. State: s.
    params = [10**3, 10**4, 10**5]
    param_names = ["N"]

    def setup(self, N):
        # [FUNCTION DOC]
        # Builds an object-dtype series of 10 times N random integers below N.
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype("object")

    def time_mode(self, N):
        # [FUNCTION DOC]
        # Times computing the most frequent values, counting missing values too.
        self.s.mode(dropna=False)


class Dir:
    # [CLASS DOC]
    # Benchmark of listing the attributes of a series that has a large string
    # index (the listing includes the labels). No parameters. State: s.
    def setup(self):
        # [FUNCTION DOC]
        # Builds an all-missing series on an object index of 10000 distinct strings.
        self.s = Series(index=Index([f"i-{i}" for i in range(10000)], dtype=object))

    def time_dir_strings(self):
        # [FUNCTION DOC]
        # Times listing the attribute names of the series.
        dir(self.s)


class SeriesGetattr:
    # [CLASS DOC]
    # Benchmark (issue 19764) of looking up a non-existent attribute on a series
    # with a large datetime index. No parameters. State: s.
    # https://github.com/pandas-dev/pandas/issues/19764
    def setup(self):
        # [FUNCTION DOC]
        # Builds a series of the constant one on a million per-second timestamps from
        # 2012-01-01.
        self.s = Series(1, index=date_range("2012-01-01", freq="s", periods=10**6))

    def time_series_datetimeindex_repr(self):
        # [FUNCTION DOC]
        # Times a failing attribute lookup (for the name a, returning the default
        # none). Despite the method name, no representation is computed.
        getattr(self.s, "a", None)


class All:
    # [CLASS DOC]
    # Benchmark of the all-true reduction for plain and nullable booleans where the
    # answer is found immediately (fast) or only after scanning everything (slow).
    # Parametrized by N, case and dtype. State: s.
    params = [[10**3, 10**6], ["fast", "slow"], ["bool", "boolean"]]
    param_names = ["N", "case", "dtype"]

    def setup(self, N, case, dtype):
        # [FUNCTION DOC]
        # Builds a boolean series of N identical values: false for the fast case
        # (first element decides) and true for the slow case.
        val = case != "fast"
        self.s = Series([val] * N, dtype=dtype)

    def time_all(self, N, case, dtype):
        # [FUNCTION DOC]
        # Times the all-true reduction.
        self.s.all()


class Any:
    # [CLASS DOC]
    # Benchmark of the any-true reduction for plain and nullable booleans where the
    # answer is found immediately (fast) or only after scanning everything (slow).
    # Parametrized by N, case and dtype. State: s.
    params = [[10**3, 10**6], ["fast", "slow"], ["bool", "boolean"]]
    param_names = ["N", "case", "dtype"]

    def setup(self, N, case, dtype):
        # [FUNCTION DOC]
        # Builds a boolean series of N identical values: true for the fast case and
        # false for the slow case.
        val = case == "fast"
        self.s = Series([val] * N, dtype=dtype)

    def time_any(self, N, case, dtype):
        # [FUNCTION DOC]
        # Times the any-true reduction.
        self.s.any()


class NanOps:
    # [CLASS DOC]
    # Benchmark of missing-value-aware reductions over a series of ones, for many
    # functions, two sizes and several dtypes. Parametrized by function name, N and
    # dtype. State: s and func (the bound method).
    params = [
        [
            "var",
            "mean",
            "median",
            "max",
            "min",
            "sum",
            "std",
            "sem",
            "argmax",
            "skew",
            "kurt",
            "prod",
        ],
        [10**3, 10**6],
        ["int8", "int32", "int64", "float64", "Int64", "boolean"],
    ]
    param_names = ["func", "N", "dtype"]

    def setup(self, func, N, dtype):
        # [FUNCTION DOC]
        # Raises a not-implemented error for the position-of-maximum function with the
        # nullable integer and nullable boolean dtypes (not supported yet, issue
        # 24382). Builds a series of N ones in the dtype as s and looks up the named
        # method on it as func.
        if func == "argmax" and dtype in {"Int64", "boolean"}:
            # Skip argmax for nullable int since this doesn't work yet (GH-24382)
            raise NotImplementedError
        self.s = Series(np.ones(N), dtype=dtype)
        self.func = getattr(self.s, func)

    def time_func(self, func, N, dtype):
        # [FUNCTION DOC]
        # Times calling the prepared reduction. Parameters are unused here.
        self.func()


class Rank:
    # [CLASS DOC]
    # Benchmark of ranking a series of many ties for several dtypes. Parametrized
    # by dtype (int, uint, float, object). State: s.
    param_names = ["dtype"]
    params = [
        ["int", "uint", "float", "object"],
    ]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds a series of 100000 random integers below 1000 in the dtype.
        self.s = Series(np.random.randint(0, 1000, size=100000), dtype=dtype)

    def time_rank(self, dtype):
        # [FUNCTION DOC]
        # Times ranking the series.
        self.s.rank()


class Iter:
    # [CLASS DOC]
    # Benchmark of iterating over a series element by element, for several dtypes.
    # Parametrized by dtype. State: s.
    param_names = ["dtype"]
    params = [
        "bool",
        "boolean",
        "int64",
        "Int64",
        "float64",
        "Float64",
        "datetime64[ns]",
    ]

    def setup(self, dtype):
        # [FUNCTION DOC]
        # Builds 100000 values suited to the dtype (half true then half false;
        # integers; random normal floats; per-second timestamps from 2000-01-01);
        # raises a not-implemented error for any other dtype. Creates the series in
        # the dtype as s.
        N = 10**5
        if dtype in ["bool", "boolean"]:
            data = np.repeat([True, False], N // 2)
        elif dtype in ["int64", "Int64"]:
            data = np.arange(N)
        elif dtype in ["float64", "Float64"]:
            data = np.random.randn(N)
        elif dtype == "datetime64[ns]":
            data = date_range("2000-01-01", freq="s", periods=N)
        else:
            raise NotImplementedError

        self.s = Series(data, dtype=dtype)

    def time_iter(self, dtype):
        # [FUNCTION DOC]
        # Times looping over every element of the series without using it.
        for v in self.s:
            pass


class ToNumpy:
    # [CLASS DOC]
    # Benchmark of converting a float series to a numpy array with different copy
    # and dtype options. No parameters. State: ser.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a series of a million random normal numbers as ser.
        N = 1_000_000
        self.ser = Series(
            np.random.randn(
                N,
            )
        )

    def time_to_numpy(self):
        # [FUNCTION DOC]
        # Times conversion with default settings.
        self.ser.to_numpy()

    def time_to_numpy_double_copy(self):
        # [FUNCTION DOC]
        # Times conversion requesting a 64-bit float result and a copy.
        self.ser.to_numpy(dtype="float64", copy=True)

    def time_to_numpy_copy(self):
        # [FUNCTION DOC]
        # Times conversion requesting a copy.
        self.ser.to_numpy(copy=True)

    def time_to_numpy_float_with_nan(self):
        # [FUNCTION DOC]
        # Times conversion to 64-bit float with NaN as the missing-value
        # placeholder.
        self.ser.to_numpy(dtype="float64", na_value=np.nan)


class Replace:
    # [CLASS DOC]
    # Benchmark of replacing values given as a dictionary or as two lists, in a
    # million-element float series, for time and peak memory. Parametrized by the
    # number of values to replace (100, 1000). State: arr, arr1, ser,
    # to_replace_list, values_list and replace_dict.
    param_names = ["num_to_replace"]
    params = [100, 1000]

    def setup(self, num_to_replace):
        # [FUNCTION DOC]
        # Builds a million random normal numbers as arr, a shuffled copy as arr1, and
        # a series ser from arr. Draws num_to_replace values with repeats from arr
        # (to_replace_list) and from arr1 (values_list), and pairs them in a
        # dictionary (replace_dict).
        N = 1_000_000
        self.arr = np.random.randn(N)
        self.arr1 = self.arr.copy()
        np.random.shuffle(self.arr1)
        self.ser = Series(self.arr)

        self.to_replace_list = np.random.choice(self.arr, num_to_replace)
        self.values_list = np.random.choice(self.arr1, num_to_replace)

        self.replace_dict = dict(
            zip(self.to_replace_list, self.values_list, strict=True)
        )

    def time_replace_dict(self, num_to_replace):
        # [FUNCTION DOC]
        # Times replacing values using the dictionary.
        self.ser.replace(self.replace_dict)

    def peakmem_replace_dict(self, num_to_replace):
        # [FUNCTION DOC]
        # Peak-memory benchmark of replacing values using the dictionary.
        self.ser.replace(self.replace_dict)

    def time_replace_list(self, num_to_replace):
        # [FUNCTION DOC]
        # Times replacing values using the list of values and the list of
        # replacements.
        self.ser.replace(self.to_replace_list, self.values_list)

    def peakmem_replace_list(self, num_to_replace):
        # [FUNCTION DOC]
        # Peak-memory benchmark of replacing values using the two lists.
        self.ser.replace(self.to_replace_list, self.values_list)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
