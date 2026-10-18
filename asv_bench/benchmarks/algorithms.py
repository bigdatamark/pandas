# [MODULE DOC]
# Benchmarks for core algorithms: factorization, duplicate detection, object
# hashing, quantiles and argsort of masked integer arrays. Public contents are the
# benchmark classes Factorize, Duplicated, DuplicatedMaskedArray, Hashing,
# Quantile and SortIntegerArray. At import time it tries to import a hashing
# module, first from the util package and then from the older tools hashing
# location, binding the first success to the module-level name hashing (if both
# fail the name stays undefined and the Hashing benchmarks would fail). The last
# line re-exports the common setup hook from the shared benchmark helper module so
# asv runs it before each benchmark.
from importlib import import_module

import numpy as np

import pandas as pd

for imp in ["pandas.util", "pandas.tools.hashing"]:
    try:
        hashing = import_module(imp)
        break
    except (ImportError, TypeError, ValueError):
        pass


class Factorize:
    # [CLASS DOC]
    # Benchmark of the top-level factorize function over many dtypes, with
    # unique or repeated data and sorted or unsorted codes. Parametrized by
    # unique flag, sort flag and dtype name. State: data (an index or array).
    params = [
        [True, False],
        [True, False],
        [
            "int64",
            "uint64",
            "float64",
            "object",
            "object_str",
            "datetime64[ns]",
            "datetime64[ns, tz]",
            "Int64",
            "boolean",
            "string[pyarrow]",
        ],
    ]
    param_names = ["unique", "sort", "dtype"]

    def setup(self, unique, sort, dtype):
        # [FUNCTION DOC]
        # Builds one hundred thousand elements of the requested dtype and stores
        # them as data. Integer and object kinds use an index over a range; float
        # uses random normal values; boolean uses a masked boolean array of random
        # zeros and ones; datetime uses an hourly range (naive or in the Tokyo
        # timezone); object_str uses object-dtype strings; string[pyarrow] uses an
        # arrow-backed string array. Raises a not-implemented error for any other
        # dtype. If unique is false every element is repeated five times. The sort
        # parameter is unused here.
        N = 10**5

        if dtype in ["int64", "uint64", "Int64", "object"]:
            data = pd.Index(np.arange(N), dtype=dtype)
        elif dtype == "float64":
            data = pd.Index(np.random.randn(N), dtype=dtype)
        elif dtype == "boolean":
            data = pd.array(np.random.randint(0, 2, N), dtype=dtype)
        elif dtype == "datetime64[ns]":
            data = pd.date_range("2011-01-01", freq="h", periods=N)
        elif dtype == "datetime64[ns, tz]":
            data = pd.date_range("2011-01-01", freq="h", periods=N, tz="Asia/Tokyo")
        elif dtype == "object_str":
            data = pd.Index([f"i-{i}" for i in range(N)], dtype=object)
        elif dtype == "string[pyarrow]":
            data = pd.array(
                pd.Index([f"i-{i}" for i in range(N)], dtype=object),
                dtype="string[pyarrow]",
            )
        else:
            raise NotImplementedError

        if not unique:
            data = data.repeat(5)
        self.data = data

    def time_factorize(self, unique, sort, dtype):
        # [FUNCTION DOC]
        # Times factorizing the data, with codes sorted according to the sort
        # parameter. The unique and dtype parameters are unused. Result discarded.
        pd.factorize(self.data, sort=sort)

    def peakmem_factorize(self, unique, sort, dtype):
        # [FUNCTION DOC]
        # Measures peak memory use of factorizing the data, with the sort
        # parameter controlling code ordering. Other parameters are unused.
        pd.factorize(self.data, sort=sort)


class Duplicated:
    # [CLASS DOC]
    # Benchmark of the duplicate-marking method of Index across dtypes, with
    # unique or repeated data and each keep policy (first, last, none).
    # Parametrized by unique flag, keep and dtype. State: idx.
    params = [
        [True, False],
        ["first", "last", False],
        [
            "int64",
            "uint64",
            "float64",
            "string",
            "datetime64[ns]",
            "datetime64[ns, tz]",
            "timestamp[ms][pyarrow]",
            "duration[s][pyarrow]",
        ],
    ]
    param_names = ["unique", "keep", "dtype"]

    def setup(self, unique, keep, dtype):
        # [FUNCTION DOC]
        # Builds an index of one hundred thousand elements of the given dtype
        # (string means object-dtype strings; the arrow dtypes are built from an
        # integer range). Raises a not-implemented error for unknown dtypes. If
        # unique is false every element is repeated five times. Stores the index
        # as idx and touches the uniqueness property once so that its cached
        # value does not count towards the timing. The keep parameter is unused.
        N = 10**5
        if dtype in ["int64", "uint64"]:
            data = pd.Index(np.arange(N), dtype=dtype)
        elif dtype == "float64":
            data = pd.Index(np.random.randn(N), dtype="float64")
        elif dtype == "string":
            data = pd.Index([f"i-{i}" for i in range(N)], dtype=object)
        elif dtype == "datetime64[ns]":
            data = pd.date_range("2011-01-01", freq="h", periods=N)
        elif dtype == "datetime64[ns, tz]":
            data = pd.date_range("2011-01-01", freq="h", periods=N, tz="Asia/Tokyo")
        elif dtype in ["timestamp[ms][pyarrow]", "duration[s][pyarrow]"]:
            data = pd.Index(np.arange(N), dtype=dtype)
        else:
            raise NotImplementedError
        if not unique:
            data = data.repeat(5)
        self.idx = data
        # cache is_unique
        self.idx.is_unique

    def time_duplicated(self, unique, keep, dtype):
        # [FUNCTION DOC]
        # Times marking duplicates in the index using the keep policy. Other
        # parameters are unused. Result is discarded.
        self.idx.duplicated(keep=keep)


class DuplicatedMaskedArray:
    # [CLASS DOC]
    # Benchmark of the duplicate-marking method of Series backed by masked
    # (nullable) arrays, with some missing values. Parametrized by unique flag,
    # keep policy and nullable dtype (Int64 or Float64). State: ser.
    params = [
        [True, False],
        ["first", "last", False],
        ["Int64", "Float64"],
    ]
    param_names = ["unique", "keep", "dtype"]

    def setup(self, unique, keep, dtype):
        # [FUNCTION DOC]
        # Builds a series of one hundred thousand integers in the nullable dtype,
        # sets every hundredth element starting at position one to missing, and
        # repeats every element five times if unique is false. Stores it as ser
        # and touches the uniqueness property once to cache it. The keep
        # parameter is unused.
        N = 10**5
        data = pd.Series(np.arange(N), dtype=dtype)
        data[list(range(1, N, 100))] = pd.NA
        if not unique:
            data = data.repeat(5)
        self.ser = data
        # cache is_unique
        self.ser.is_unique

    def time_duplicated(self, unique, keep, dtype):
        # [FUNCTION DOC]
        # Times marking duplicates in the series using the keep policy. Other
        # parameters are unused. Result is discarded.
        self.ser.duplicated(keep=keep)


class Hashing:
    # [CLASS DOC]
    # Benchmark of row-wise hashing of pandas objects using the hashing module
    # located at import time. Data is built once by the asv cache hook and passed
    # to each timing method as df. No per-run state on the instance.
    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook run once per benchmark class. Takes no inputs and returns
        # a frame of one hundred thousand rows with columns of strings (sampled
        # from ten thousand distinct values), random floats, integers, per-second
        # datetimes, per-second timedeltas and a categorical column derived from
        # the strings. Rows ten to nineteen are then overwritten with missing
        # values in every column.
        N = 10**5

        df = pd.DataFrame(
            {
                "strings": pd.Series(
                    pd.Index([f"i-{i}" for i in range(10000)], dtype=object).take(
                        np.random.randint(0, 10000, size=N)
                    )
                ),
                "floats": np.random.randn(N),
                "ints": np.arange(N),
                "dates": pd.date_range("20110101", freq="s", periods=N),
                "timedeltas": pd.timedelta_range("1 day", freq="s", periods=N),
            }
        )
        df["categories"] = df["strings"].astype("category")
        df.iloc[10:20] = np.nan
        return df

    def time_frame(self, df):
        # [FUNCTION DOC]
        # Times hashing every row of the whole frame. Input df is the cached frame.
        hashing.hash_pandas_object(df)

    def time_series_int(self, df):
        # [FUNCTION DOC]
        # Times hashing the integer column of the cached frame.
        hashing.hash_pandas_object(df["ints"])

    def time_series_string(self, df):
        # [FUNCTION DOC]
        # Times hashing the string column of the cached frame.
        hashing.hash_pandas_object(df["strings"])

    def time_series_float(self, df):
        # [FUNCTION DOC]
        # Times hashing the float column of the cached frame.
        hashing.hash_pandas_object(df["floats"])

    def time_series_categorical(self, df):
        # [FUNCTION DOC]
        # Times hashing the categorical column of the cached frame.
        hashing.hash_pandas_object(df["categories"])

    def time_series_timedeltas(self, df):
        # [FUNCTION DOC]
        # Times hashing the timedelta column of the cached frame.
        hashing.hash_pandas_object(df["timedeltas"])

    def time_series_dates(self, df):
        # [FUNCTION DOC]
        # Times hashing the datetime column of the cached frame.
        hashing.hash_pandas_object(df["dates"])


class Quantile:
    # [CLASS DOC]
    # Benchmark of Series quantile computation. Parametrized by quantile
    # (0, 0.5, 1), interpolation method and dtype (float64, int64, uint64).
    # State: ser.
    params = [
        [0, 0.5, 1],
        ["linear", "nearest", "lower", "higher", "midpoint"],
        ["float64", "int64", "uint64"],
    ]
    param_names = ["quantile", "interpolation", "dtype"]

    def setup(self, quantile, interpolation, dtype):
        # [FUNCTION DOC]
        # Builds a series from one hundred thousand values of the dtype (a range
        # for integer dtypes, random normal values for float64), each repeated five
        # times. Raises a not-implemented error for other dtypes. Stores it as ser.
        N = 10**5
        if dtype in ["int64", "uint64"]:
            data = np.arange(N, dtype=dtype)
        elif dtype == "float64":
            data = np.random.randn(N)
        else:
            raise NotImplementedError
        self.ser = pd.Series(data.repeat(5))

    def time_quantile(self, quantile, interpolation, dtype):
        # [FUNCTION DOC]
        # Times computing the given quantile of the series with the given
        # interpolation method. The dtype parameter is unused.
        self.ser.quantile(quantile, interpolation=interpolation)


class SortIntegerArray:
    # [CLASS DOC]
    # Benchmark of argsort on a nullable Int64 array containing one missing
    # value. Parametrized by array length. State: array.
    params = [10**3, 10**5]

    def setup(self, N):
        # [FUNCTION DOC]
        # Builds an object array of N floats cast to object, sets position forty
        # to missing and converts it to a nullable Int64 array stored as array.
        # KNOWN ISSUE: the position forty is assumed to exist; it fails for N
        # below forty-one, though the parameters never go that low.
        data = np.arange(N, dtype=float).astype(object)
        data[40] = pd.NA
        self.array = pd.array(data, dtype="Int64")

    def time_argsort(self, N):
        # [FUNCTION DOC]
        # Times computing the sorting permutation of the array. N is unused.
        self.array.argsort()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
