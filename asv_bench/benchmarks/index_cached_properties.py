# [MODULE DOC]
# Benchmarks of first-access cost of cached properties on many Index types. Public
# contents are the benchmark class IndexCache. No import-time side effects.
import pandas as pd


class IndexCache:
    # [CLASS DOC]
    # Benchmark of computing cached index properties (values, shape, monotonicity,
    # uniqueness, engine, inferred type) from a cold cache. Parametrized by index
    # type name. One call per measurement with a repeat tuple so setup (which
    # clears the cache) runs before every call. State: idx.
    number = 1
    repeat = (3, 100, 20)

    params = [
        [
            "CategoricalIndex",
            "DatetimeIndex",
            "Float64Index",
            "IntervalIndex",
            "Int64Index",
            "MultiIndex",
            "PeriodIndex",
            "RangeIndex",
            "TimedeltaIndex",
            "UInt64Index",
        ]
    ]
    param_names = ["index_type"]

    def setup(self, index_type):
        # [FUNCTION DOC]
        # Builds an index of 100000 entries of the named type: a multi-index
        # (product of 50000 minute timestamps and the letters a and b), a minute
        # datetime index, a 64-bit integer index, a minute period index, a range
        # index, an interval index of consecutive unit intervals, a timedelta index
        # of nanosecond counts, a float index, an unsigned integer index, or a
        # categorical index with matching categories. The old class-style names
        # (such as the integer and float index names) are selected by string and
        # mapped to dtype-specific indexes. Raises a value error for an unknown
        # name and an assertion error if the length is not 100000. Finally resets
        # the internal cache mapping of the index to an empty mapping so each
        # property is computed from scratch. Stores the index as idx.
        N = 10**5
        if index_type == "MultiIndex":
            self.idx = pd.MultiIndex.from_product(
                [pd.date_range("1/1/2000", freq="min", periods=N // 2), ["a", "b"]]
            )
        elif index_type == "DatetimeIndex":
            self.idx = pd.date_range("1/1/2000", freq="min", periods=N)
        elif index_type == "Int64Index":
            self.idx = pd.Index(range(N), dtype="int64")
        elif index_type == "PeriodIndex":
            self.idx = pd.period_range("1/1/2000", freq="min", periods=N)
        elif index_type == "RangeIndex":
            self.idx = pd.RangeIndex(start=0, stop=N)
        elif index_type == "IntervalIndex":
            self.idx = pd.IntervalIndex.from_arrays(range(N), range(1, N + 1))
        elif index_type == "TimedeltaIndex":
            self.idx = pd.TimedeltaIndex(range(N))
        elif index_type == "Float64Index":
            self.idx = pd.Index(range(N), dtype="float64")
        elif index_type == "UInt64Index":
            self.idx = pd.Index(range(N), dtype="uint64")
        elif index_type == "CategoricalIndex":
            self.idx = pd.CategoricalIndex(range(N), range(N))
        else:
            raise ValueError
        assert len(self.idx) == N
        self.idx._cache = {}

    def time_values(self, index_type):
        # [FUNCTION DOC]
        # Times reading the internal values of the index from a cold cache.
        self.idx._values

    def time_shape(self, index_type):
        # [FUNCTION DOC]
        # Times reading the shape of the index.
        self.idx.shape

    def time_is_monotonic_decreasing(self, index_type):
        # [FUNCTION DOC]
        # Times reading the decreasing-monotonic property of the index.
        self.idx.is_monotonic_decreasing

    def time_is_monotonic_increasing(self, index_type):
        # [FUNCTION DOC]
        # Times reading the increasing-monotonic property of the index.
        self.idx.is_monotonic_increasing

    def time_is_unique(self, index_type):
        # [FUNCTION DOC]
        # Times reading the uniqueness property of the index.
        self.idx.is_unique

    def time_engine(self, index_type):
        # [FUNCTION DOC]
        # Times building the internal lookup engine of the index.
        self.idx._engine

    def time_inferred_type(self, index_type):
        # [FUNCTION DOC]
        # Times reading the inferred type of the index.
        self.idx.inferred_type
