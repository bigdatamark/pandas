"""
Timedelta benchmarks with non-tslibs dependencies.  See
benchmarks.tslibs.timedelta for benchmarks that rely only on tslibs.
"""

# [MODULE DOC]
# ASV benchmark module for timedelta-related behavior that depends on the wider
# pandas package (series accessors, indexes, alignment), as opposed to the
# tslibs-only benchmarks in the tslibs subpackage.
# Public contents: the benchmark classes DatetimeAccessor (despite the name it
# benchmarks the timedelta accessor attributes) and TimedeltaIndexing.
# Import-time side effects: none beyond importing pandas names.
# KNOWN ISSUE: the class named DatetimeAccessor actually benchmarks the
# timedelta properties of a timedelta series, so its name is misleading.

from pandas import (
    DataFrame,
    Series,
    timedelta_range,
)


class DatetimeAccessor:
    # [CLASS DOC]
    # Benchmark class for the accessor properties of a timedelta-valued series
    # (days, seconds, microseconds, nanoseconds). Uses the asv cache mechanism:
    # the cache setup method builds one series and asv passes the cached value to
    # every benchmark method as its argument.
    def setup_cache(self):
        # [FUNCTION DOC]
        # Builds and returns a series of 100000 hourly timedeltas starting at
        # one day. Called once by the asv runner; the returned series is passed
        # to each timing method as the series argument.
        N = 100000
        series = Series(timedelta_range("1 days", periods=N, freq="h"))
        return series

    def time_dt_accessor(self, series):
        # [FUNCTION DOC]
        # Times creating the accessor object on the cached series. The result
        # is discarded.
        series.dt

    def time_timedelta_days(self, series):
        # [FUNCTION DOC]
        # Times extracting the whole-days component of the cached series.
        series.dt.days

    def time_timedelta_seconds(self, series):
        # [FUNCTION DOC]
        # Times extracting the seconds component (within the day) of the cached
        # series.
        series.dt.seconds

    def time_timedelta_microseconds(self, series):
        # [FUNCTION DOC]
        # Times extracting the microseconds component of the cached series.
        series.dt.microseconds

    def time_timedelta_nanoseconds(self, series):
        # [FUNCTION DOC]
        # Times extracting the nanoseconds component of the cached series.
        series.dt.nanoseconds


class TimedeltaIndexing:
    # [CLASS DOC]
    # Benchmark class for indexing, alignment and set operations on a
    # timedelta index. State set by setup: two daily timedelta indexes of length
    # 1000 (starting one year apart), a series of integers indexed by the first,
    # and a scalar timedelta taken from the middle of the first index.
    def setup(self):
        # [FUNCTION DOC]
        # Creates index and index2 (1000 daily timedeltas starting at 1985 days
        # and 1986 days respectively), a series of the integers 0..999 indexed
        # by the first index, and the scalar element at position 500.
        self.index = timedelta_range(start="1985", periods=1000, freq="D")
        self.index2 = timedelta_range(start="1986", periods=1000, freq="D")
        self.series = Series(range(1000), index=self.index)
        self.timedelta = self.index[500]

    def time_get_loc(self):
        # [FUNCTION DOC]
        # Times looking up the integer position of the scalar in the index.
        self.index.get_loc(self.timedelta)

    def time_shallow_copy(self):
        # [FUNCTION DOC]
        # Times creating a view of the index through the private view method
        # (the shallow copy path).
        self.index._view()

    def time_series_loc(self):
        # [FUNCTION DOC]
        # Times label-based lookup of the scalar in the series.
        self.series.loc[self.timedelta]

    def time_align(self):
        # [FUNCTION DOC]
        # Times building a two-column frame from the full series and its first
        # 500 entries, which forces index alignment.
        DataFrame({"a": self.series, "b": self.series[:500]})

    def time_intersection(self):
        # [FUNCTION DOC]
        # Times the set intersection of the two indexes.
        self.index.intersection(self.index2)

    def time_union(self):
        # [FUNCTION DOC]
        # Times the set union of the two indexes.
        self.index.union(self.index2)

    def time_unique(self):
        # [FUNCTION DOC]
        # Times computing the unique values of the index.
        self.index.unique()
