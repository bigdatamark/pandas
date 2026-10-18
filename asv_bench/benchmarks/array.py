# [MODULE DOC]
# Benchmarks for constructing and manipulating extension arrays: boolean,
# integer, interval, string (python-backed and arrow-backed) and general arrow
# extension arrays. Public contents are the benchmark classes BooleanArray,
# IntegerArray, IntervalArray, StringArray, ArrowStringArray and
# ArrowExtensionArray. The arrow-based classes need the optional pyarrow package
# and signal a skip through a not-implemented error when it is missing. No
# import-time side effects beyond importing numpy and pandas.
import numpy as np

import pandas as pd


class BooleanArray:
    # [CLASS DOC]
    # Benchmark of building a nullable boolean array directly from data and mask
    # arrays and via the generic array constructor from bool, float, integer and
    # list inputs. No parameters. State: values_bool, values_float,
    # values_integer, values_integer_like (a plain list), data and mask.
    def setup(self):
        # [FUNCTION DOC]
        # Creates tiny four-element inputs representing true, false, true, false
        # in boolean, float, integer and list forms, plus the data array and a
        # mask array marking the third element as missing.
        self.values_bool = np.array([True, False, True, False])
        self.values_float = np.array([1.0, 0.0, 1.0, 0.0])
        self.values_integer = np.array([1, 0, 1, 0])
        self.values_integer_like = [1, 0, 1, 0]
        self.data = np.array([True, False, True, False])
        self.mask = np.array([False, False, True, False])

    def time_constructor(self):
        # [FUNCTION DOC]
        # Times the direct constructor of the boolean array from data and mask.
        pd.arrays.BooleanArray(self.data, self.mask)

    def time_from_bool_array(self):
        # [FUNCTION DOC]
        # Times creating a boolean extension array from a boolean numpy array.
        pd.array(self.values_bool, dtype="boolean")

    def time_from_integer_array(self):
        # [FUNCTION DOC]
        # Times creating a boolean extension array from an integer numpy array.
        pd.array(self.values_integer, dtype="boolean")

    def time_from_integer_like(self):
        # [FUNCTION DOC]
        # Times creating a boolean extension array from a list of integers.
        pd.array(self.values_integer_like, dtype="boolean")

    def time_from_float_array(self):
        # [FUNCTION DOC]
        # Times creating a boolean extension array from a float numpy array.
        pd.array(self.values_float, dtype="boolean")


class IntegerArray:
    # [CLASS DOC]
    # Benchmark of building a nullable integer array directly and through the
    # generic array constructor. No parameters. State: values_integer, data and
    # mask, each one million elements long.
    def setup(self):
        # [FUNCTION DOC]
        # Tiles a four-element pattern 250000 times to make a million-element
        # plain integer array, a 64-bit data array (one to four repeated) and a
        # mask that marks every third element of each pattern as missing.
        N = 250_000
        self.values_integer = np.tile(np.array([1, 0, 1, 0]), N)
        self.data = np.tile(np.array([1, 2, 3, 4], dtype="int64"), N)
        self.mask = np.tile(np.array([False, False, True, False]), N)

    def time_constructor(self):
        # [FUNCTION DOC]
        # Times the direct constructor of the integer array from data and mask.
        pd.arrays.IntegerArray(self.data, self.mask)

    def time_from_integer_array(self):
        # [FUNCTION DOC]
        # Times creating a nullable 64-bit integer array from the integer array.
        pd.array(self.values_integer, dtype="Int64")


class IntervalArray:
    # [CLASS DOC]
    # Benchmark of building an interval array from a list of tuples. No
    # parameters. State: tuples.
    def setup(self):
        # [FUNCTION DOC]
        # Creates 10000 consecutive pairs (i, i plus one) and stores them as
        # tuples.
        N = 10_000
        self.tuples = [(i, i + 1) for i in range(N)]

    def time_from_tuples(self):
        # [FUNCTION DOC]
        # Times the interval array class constructor from a list of tuples.
        pd.arrays.IntervalArray.from_tuples(self.tuples)


class StringArray:
    # [CLASS DOC]
    # Benchmark of creating a string extension array from three input forms:
    # object numpy array, fixed-width unicode numpy array and a plain list. No
    # parameters. State: values_obj, values_str and values_list.
    def setup(self):
        # [FUNCTION DOC]
        # Builds 100000 decimal strings of the integers below that count as an
        # object array, then derives an object-typed copy (values_obj), a
        # fixed-width unicode array (values_str) and a list (values_list).
        N = 100_000
        values = np.array([str(i) for i in range(N)], dtype=object)
        self.values_obj = np.array(values, dtype="object")
        self.values_str = np.array(values, dtype="U")
        self.values_list = values.tolist()

    def time_from_np_object_array(self):
        # [FUNCTION DOC]
        # Times creating a string array from the object numpy array.
        pd.array(self.values_obj, dtype="string")

    def time_from_np_str_array(self):
        # [FUNCTION DOC]
        # Times creating a string array from the unicode numpy array.
        pd.array(self.values_str, dtype="string")

    def time_from_list(self):
        # [FUNCTION DOC]
        # Times creating a string array from the list.
        pd.array(self.values_list, dtype="string")


class ArrowStringArray:
    # [CLASS DOC]
    # Benchmark of item assignment and list conversion on the arrow-backed string
    # array. Parametrized by whether the underlying arrow storage has many
    # chunks (True) or a single chunk (False). State: array. Requires pyarrow.
    params = [False, True]
    param_names = ["multiple_chunks"]

    def setup(self, multiple_chunks):
        # [FUNCTION DOC]
        # Imports pyarrow, raising a not-implemented error (to skip the benchmark)
        # if the import fails. Builds 10000 decimal strings. If multiple_chunks
        # is true, splits them into chunks of one hundred and wraps a chunked
        # arrow array; otherwise wraps a single arrow array. Stores the arrow
        # string array as array.
        try:
            import pyarrow as pa
        except ImportError as err:
            raise NotImplementedError from err
        strings = np.array([str(i) for i in range(10_000)], dtype=object)
        if multiple_chunks:
            chunks = [strings[i : i + 100] for i in range(0, len(strings), 100)]
            self.array = pd.arrays.ArrowStringArray(pa.chunked_array(chunks))
        else:
            self.array = pd.arrays.ArrowStringArray(pa.array(strings))

    def time_setitem(self, multiple_chunks):
        # [FUNCTION DOC]
        # Times assigning the string foo to each of the first 200 positions one at
        # a time. Mutates the array held in state. The parameter is unused.
        for i in range(200):
            self.array[i] = "foo"

    def time_setitem_list(self, multiple_chunks):
        # [FUNCTION DOC]
        # Times assigning foo to a list of positions (the first fifty followed by
        # every fiftieth position counted from the end over the last thousand)
        # using a list indexer. Mutates the array. The parameter is unused.
        indexer = list(range(50)) + list(range(-1000, 0, 50))
        self.array[indexer] = ["foo"] * len(indexer)

    def time_setitem_slice(self, multiple_chunks):
        # [FUNCTION DOC]
        # Times assigning foo to every tenth position using a stepped slice.
        # Mutates the array. The parameter is unused.
        self.array[::10] = "foo"

    def time_setitem_null_slice(self, multiple_chunks):
        # [FUNCTION DOC]
        # Times assigning foo to the whole array through a full slice. Mutates
        # the array. The parameter is unused.
        self.array[:] = "foo"

    def time_tolist(self, multiple_chunks):
        # [FUNCTION DOC]
        # Times converting the array to a Python list. The parameter is unused.
        self.array.tolist()


class ArrowExtensionArray:
    # [CLASS DOC]
    # Benchmark of converting arrow-backed extension arrays to numpy, for several
    # arrow dtypes with and without missing values. Parametrized by dtype name
    # and a hasna flag. State: arr. Requires pyarrow.
    params = [
        [
            "boolean[pyarrow]",
            "float64[pyarrow]",
            "int64[pyarrow]",
            "string[pyarrow]",
            "timestamp[ns][pyarrow]",
        ],
        [False, True],
    ]
    param_names = ["dtype", "hasna"]

    def setup(self, dtype, hasna):
        # [FUNCTION DOC]
        # Generates 100000 values appropriate to dtype: random booleans, random
        # normal floats, a range of integers, decimal strings, or per-second
        # timestamps. Raises a not-implemented error for any other dtype. Builds
        # the arrow-backed array from them, and if hasna is true sets every
        # second element to missing. Stores it as arr.
        N = 100_000
        if dtype == "boolean[pyarrow]":
            data = np.random.choice([True, False], N, replace=True)
        elif dtype == "float64[pyarrow]":
            data = np.random.randn(N)
        elif dtype == "int64[pyarrow]":
            data = np.arange(N)
        elif dtype == "string[pyarrow]":
            data = np.array([str(i) for i in range(N)], dtype=object)
        elif dtype == "timestamp[ns][pyarrow]":
            data = pd.date_range("2000-01-01", freq="s", periods=N)
        else:
            raise NotImplementedError

        arr = pd.array(data, dtype=dtype)
        if hasna:
            arr[::2] = pd.NA
        self.arr = arr

    def time_to_numpy(self, dtype, hasna):
        # [FUNCTION DOC]
        # Times converting the array to a numpy array. Parameters are unused.
        self.arr.to_numpy()
