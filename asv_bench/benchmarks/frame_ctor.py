# [MODULE DOC]
# Benchmarks for DataFrame construction from dictionaries, series, records,
# arrays, lists, ranges and scalars. Public contents are the benchmark classes
# FromDicts, FromSeries, FromDictwithTimestamp, FromRecords, FromNDArray,
# FromLists, FromRange, FromScalar and FromArrays. At import time it imports the
# Hour and Nano offsets from the offsets module, falling back to the old datetools
# module for older library versions. The final line re-exports the common asv
# setup hook.
import numpy as np

import pandas as pd
from pandas import (
    NA,
    Categorical,
    DataFrame,
    Float64Dtype,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)

try:
    from pandas.tseries.offsets import (
        Hour,
        Nano,
    )
except ImportError:
    # For compatibility with older versions
    from pandas.core.datetools import (
        Hour,
        Nano,
    )


class FromDicts:
    # [CLASS DOC]
    # Benchmark of building a frame from dictionary-shaped inputs: a list of
    # records, nested dictionaries with optional index and columns, nested
    # dictionaries with integer keys, and a dictionary of categoricals. No
    # parameters. State: index, columns, data, dict_list, data2 and
    # dict_of_categoricals.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a random 5000 by 50 frame with string labels and derives data
        # (nested column-to-index mapping) and dict_list (list of per-row
        # records) from it; also data2 (2000 outer keys, each mapping 100 integer
        # keys to floats) and dict_of_categoricals (50 categoricals of 5000
        # integers, which are not consolidated).
        N, K = 5000, 50
        self.index = pd.Index([f"i-{i}" for i in range(N)], dtype=object)
        self.columns = pd.Index([f"i-{i}" for i in range(K)], dtype=object)
        frame = DataFrame(np.random.randn(N, K), index=self.index, columns=self.columns)
        self.data = frame.to_dict()
        self.dict_list = frame.to_dict(orient="records")
        self.data2 = {i: {j: float(j) for j in range(100)} for i in range(2000)}

        # arrays which we won't consolidate
        self.dict_of_categoricals = {i: Categorical(np.arange(N)) for i in range(K)}

    def time_list_of_dict(self):
        # [FUNCTION DOC]
        # Times building a frame from the list of record dictionaries.
        DataFrame(self.dict_list)

    def time_nested_dict(self):
        # [FUNCTION DOC]
        # Times building a frame from the nested dictionary.
        DataFrame(self.data)

    def time_nested_dict_index(self):
        # [FUNCTION DOC]
        # Times building a frame from the nested dictionary with an explicit row
        # index.
        DataFrame(self.data, index=self.index)

    def time_nested_dict_columns(self):
        # [FUNCTION DOC]
        # Times building a frame from the nested dictionary with explicit columns.
        DataFrame(self.data, columns=self.columns)

    def time_nested_dict_index_columns(self):
        # [FUNCTION DOC]
        # Times building a frame from the nested dictionary with both explicit
        # index and columns.
        DataFrame(self.data, index=self.index, columns=self.columns)

    def time_nested_dict_int64(self):
        # [FUNCTION DOC]
        # Times building a frame from the nested dictionary with integer keys
        # (a regression benchmark described in an old issue numbered 621).
        # nested dict, integer indexes, regression described in #621
        DataFrame(self.data2)

    def time_dict_of_categoricals(self):
        # [FUNCTION DOC]
        # Times building a frame from the dictionary of categoricals, which cannot
        # be consolidated into one block.
        # dict of arrays that we won't consolidate
        DataFrame(self.dict_of_categoricals)


class FromSeries:
    # [CLASS DOC]
    # Benchmark of building a frame from a single series with a two-level
    # multi-index. No parameters. State: s.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a multi-index from the product of two ranges of 100 and a
        # random normal series of 10000 elements using it; stores it as s.
        mi = MultiIndex.from_product([range(100), range(100)])
        self.s = Series(np.random.randn(10000), index=mi)

    def time_mi_series(self):
        # [FUNCTION DOC]
        # Times building a frame from the multi-indexed series.
        DataFrame(self.s)


class FromDictwithTimestamp:
    # [CLASS DOC]
    # Benchmark of building a frame from a dictionary whose keys are timestamps,
    # generated with a nanosecond or hourly frequency. Parametrized by offset.
    # State: d.
    params = [Nano(1), Hour(1)]
    param_names = ["offset"]

    def setup(self, offset):
        # [FUNCTION DOC]
        # Builds a datetime index of 1000 entries starting at 1900-01-01 with the
        # offset as frequency, a random 1000 by 10 frame on it, and stores the
        # nested dictionary form of the frame as d.
        N = 10**3
        idx = date_range(Timestamp("1/1/1900"), freq=offset, periods=N)
        df = DataFrame(np.random.randn(N, 10), index=idx)
        self.d = df.to_dict()

    def time_dict_with_timestamp_offsets(self, offset):
        # [FUNCTION DOC]
        # Times building a frame from the dictionary d. The offset parameter is
        # unused here.
        DataFrame(self.d)


class FromRecords:
    # [CLASS DOC]
    # Benchmark of building a frame from a generator of tuples with and without a
    # row limit (issue 6700). Parametrized by nrows (none or 1000). Because the
    # generator is consumed by a run, the class sets one call per measurement
    # and a repeat tuple so setup reruns before every call. State: gen, and df
    # after timing.
    params = [None, 1000]
    param_names = ["nrows"]

    # Generators get exhausted on use, so run setup before every call
    number = 1
    repeat = (3, 250, 10)

    def setup(self, nrows):
        # [FUNCTION DOC]
        # Creates a generator of 100000 triples (x, twenty times x, one hundred
        # times x) as gen. The nrows parameter is unused here.
        N = 100000
        self.gen = ((x, (x * 20), (x * 100)) for x in range(N))

    def time_frame_from_records_generator(self, nrows):
        # [FUNCTION DOC]
        # Times building a frame from the generator, limited to nrows rows when
        # given. Stores the result as df (side effect on the instance) so the
        # object is kept alive during timing.
        # issue-6700
        self.df = DataFrame.from_records(self.gen, nrows=nrows)


class FromNDArray:
    # [CLASS DOC]
    # Benchmark of building a frame from a one-dimensional numpy array. No
    # parameters. State: data, and df after timing.
    def setup(self):
        # [FUNCTION DOC]
        # Creates 100000 random normal values as data.
        N = 100000
        self.data = np.random.randn(N)

    def time_frame_from_ndarray(self):
        # [FUNCTION DOC]
        # Times building a frame from the array and stores it as df.
        self.df = DataFrame(self.data)


class FromLists:
    # [CLASS DOC]
    # Benchmark of building a frame from a list of lists of integers. No
    # parameters. The goal_time attribute (0.2) is a legacy asv setting. State:
    # data, and df after timing.
    goal_time = 0.2

    def setup(self):
        # [FUNCTION DOC]
        # Creates 1000 rows, each a list of the integers below 100, as data.
        N = 1000
        M = 100
        self.data = [list(range(M)) for i in range(N)]

    def time_frame_from_lists(self):
        # [FUNCTION DOC]
        # Times building a frame from the list of lists and stores it as df.
        self.df = DataFrame(self.data)


class FromRange:
    # [CLASS DOC]
    # Benchmark of building a frame from a range of one million integers. No
    # parameters. The goal_time attribute (0.2) is a legacy asv setting. State:
    # data, and df after timing.
    goal_time = 0.2

    def setup(self):
        # [FUNCTION DOC]
        # Creates a range object of one million integers as data.
        N = 1_000_000
        self.data = range(N)

    def time_frame_from_range(self):
        # [FUNCTION DOC]
        # Times building a frame from the range and stores it as df.
        self.df = DataFrame(self.data)


class FromScalar:
    # [CLASS DOC]
    # Benchmark of broadcasting a scalar into a frame of nullable 64-bit float
    # dtype. No parameters. State: nrows.
    def setup(self):
        # [FUNCTION DOC]
        # Sets the row count nrows to 100000.
        self.nrows = 100_000

    def time_frame_from_scalar_ea_float64(self):
        # [FUNCTION DOC]
        # Times building a frame of nrows rows and columns a, b, c filled with the
        # float one in the nullable float dtype.
        DataFrame(
            1.0,
            index=range(self.nrows),
            columns=list("abc"),
            dtype=Float64Dtype(),
        )

    def time_frame_from_scalar_ea_float64_na(self):
        # [FUNCTION DOC]
        # Times building a frame of nrows rows and columns a, b, c filled with the
        # missing-value marker in the nullable float dtype.
        DataFrame(
            NA,
            index=range(self.nrows),
            columns=list("abc"),
            dtype=Float64Dtype(),
        )


class FromArrays:
    # [CLASS DOC]
    # Benchmark of the internal frame constructor from a list of column arrays
    # (floats, nullable integers, sparse), without integrity verification. No
    # parameters. The goal_time attribute (0.2) is a legacy asv setting. State:
    # float_arrays, sparse_arrays, int_arrays, index, columns, and df after
    # timing.
    goal_time = 0.2

    def setup(self):
        # [FUNCTION DOC]
        # Creates 1000 arrays of 1000 elements for each of three kinds: random
        # normal floats; sparse float arrays of random zeros and ones; nullable
        # 64-bit integer arrays of random integers below 1000. Also creates range
        # indexes of 1000 for rows and columns.
        N_rows = 1000
        N_cols = 1000
        self.float_arrays = [np.random.randn(N_rows) for _ in range(N_cols)]
        self.sparse_arrays = [
            pd.arrays.SparseArray(np.random.randint(0, 2, N_rows), dtype="float64")
            for _ in range(N_cols)
        ]
        self.int_arrays = [
            pd.array(np.random.randint(1000, size=N_rows), dtype="Int64")
            for _ in range(N_cols)
        ]
        self.index = pd.Index(range(N_rows))
        self.columns = pd.Index(range(N_cols))

    def time_frame_from_arrays_float(self):
        # [FUNCTION DOC]
        # Times the internal from-arrays constructor with the float arrays, the
        # prepared labels and integrity verification off. Stores the result as df.
        self.df = DataFrame._from_arrays(
            self.float_arrays,
            index=self.index,
            columns=self.columns,
            verify_integrity=False,
        )

    def time_frame_from_arrays_int(self):
        # [FUNCTION DOC]
        # Same as the float variant but with the nullable integer arrays. Stores
        # the result as df.
        self.df = DataFrame._from_arrays(
            self.int_arrays,
            index=self.index,
            columns=self.columns,
            verify_integrity=False,
        )

    def time_frame_from_arrays_sparse(self):
        # [FUNCTION DOC]
        # Same as the float variant but with the sparse arrays. Stores the result
        # as df.
        self.df = DataFrame._from_arrays(
            self.sparse_arrays,
            index=self.index,
            columns=self.columns,
            verify_integrity=False,
        )


from .pandas_vb_common import setup  # noqa: F401 isort:skip
