"""
Period benchmarks with non-tslibs dependencies.  See
benchmarks.tslibs.period for benchmarks that rely only on tslibs.
"""

# [MODULE DOC]
# Benchmarks for period-related functionality that depends on more than the
# time-series compiled library: period index construction from several inputs,
# assigning and indexing by period columns, duplicate removal and value counts on
# period data, and period index lookups and set operations. Public contents are the
# benchmark classes PeriodIndexConstructor, DataFramePeriodColumn, Algorithms and
# Indexing. No import-time side effects; does not re-export the common asv setup
# hook.
from pandas import (
    DataFrame,
    Period,
    PeriodIndex,
    Series,
    date_range,
    period_range,
)

from pandas.tseries.frequencies import to_offset


class PeriodIndexConstructor:
    # [CLASS DOC]
    # Benchmark of building a period index from a datetime range, native datetimes,
    # integers (years) and daily integers (year-month-day). Parametrized by
    # frequency (daily only) and an is_offset flag. State: rng, rng2, ints,
    # daily_ints and freq.
    params = [["D"], [True, False]]
    param_names = ["freq", "is_offset"]

    def setup(self, freq, is_offset):
        # [FUNCTION DOC]
        # Builds 1000 daily timestamps from 1985 (rng) and the same as native
        # datetimes (rng2); the integers 2000 to 2999 (ints); and 1000 daily dates
        # from 2000-01-01 formatted as compact integers (daily_ints). Stores in
        # self.freq either the frequency string or its offset object, according to
        # is_offset.
        self.rng = date_range("1985", periods=1000)
        self.rng2 = date_range("1985", periods=1000).to_pydatetime()
        self.ints = list(range(2000, 3000))
        self.daily_ints = (
            date_range("1/1/2000", periods=1000, freq=freq).strftime("%Y%m%d").map(int)
        )
        if is_offset:
            self.freq = to_offset(freq)
        else:
            self.freq = freq

    def time_from_date_range(self, freq, is_offset):
        # [FUNCTION DOC]
        # Times building a period index from the datetime range.
        # KNOWN ISSUE: the timing methods of this class pass the frequency
        # parameter string, not the prepared offset-or-string attribute, so the
        # is_offset flag has no effect and both parameter values measure the same
        # thing.
        PeriodIndex(self.rng, freq=freq)

    def time_from_pydatetime(self, freq, is_offset):
        # [FUNCTION DOC]
        # Times building a period index from the native datetimes, with the
        # frequency parameter string.
        PeriodIndex(self.rng2, freq=freq)

    def time_from_ints(self, freq, is_offset):
        # [FUNCTION DOC]
        # Times building a period index from the integers (interpreted as years),
        # with the frequency parameter string.
        PeriodIndex(self.ints, freq=freq)

    def time_from_ints_daily(self, freq, is_offset):
        # [FUNCTION DOC]
        # Times building a period index from the compact daily integers, with the
        # frequency parameter string.
        PeriodIndex(self.daily_ints, freq=freq)


class DataFramePeriodColumn:
    # [CLASS DOC]
    # Benchmark of assigning a period array as a column and then setting it as the
    # index. No parameters. State: rng (20000 per-second periods from 1990-01-01)
    # and df (an empty frame with a matching row count).
    def setup(self):
        # [FUNCTION DOC]
        # Builds a per-second period range of 20000 entries as rng and an empty
        # frame of the same length as df.
        self.rng = period_range(start="1/1/1990", freq="s", periods=20000)
        self.df = DataFrame(index=range(len(self.rng)))

    def time_setitem_period_column(self):
        # [FUNCTION DOC]
        # Times assigning the period range as a new column (mutates df).
        self.df["col"] = self.rng

    def time_set_index(self):
        # [FUNCTION DOC]
        # Times assigning the period range as a column and setting it as an extra
        # index level (the assignment mutates df; the timing is limited by
        # comparisons of period objects, issue 21582).
        # GH#21582 limited by comparisons of Period objects
        self.df["col2"] = self.rng
        self.df.set_index("col2", append=True)


class Algorithms:
    # [CLASS DOC]
    # Benchmark of duplicate removal and value counting on period data held in an
    # index or a series. Parametrized by container kind (index, series). State:
    # vector.
    params = ["index", "series"]
    param_names = ["typ"]

    def setup(self, typ):
        # [FUNCTION DOC]
        # Builds four monthly periods (January to April 2011), repeats them 1000
        # times (4000 elements), and stores them as a period index or a series
        # depending on typ (as vector). Any other kind leaves vector unset.
        data = [
            Period("2011-01", freq="M"),
            Period("2011-02", freq="M"),
            Period("2011-03", freq="M"),
            Period("2011-04", freq="M"),
        ]

        if typ == "index":
            self.vector = PeriodIndex(data * 1000, freq="M")
        elif typ == "series":
            self.vector = Series(data * 1000)

    def time_drop_duplicates(self, typ):
        # [FUNCTION DOC]
        # Times dropping duplicate elements.
        self.vector.drop_duplicates()

    def time_value_counts(self, typ):
        # [FUNCTION DOC]
        # Times counting occurrences of each distinct period.
        self.vector.value_counts()


class Indexing:
    # [CLASS DOC]
    # Benchmark of lookup, copying, alignment and set operations on a daily period
    # index. No parameters. State: index, series and period.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a daily period index of 1000 entries from 1985, a series of the
        # integers below 1000 on it, and the period at position 500 as period.
        self.index = period_range(start="1985", periods=1000, freq="D")
        self.series = Series(range(1000), index=self.index)
        self.period = self.index[500]

    def time_get_loc(self):
        # [FUNCTION DOC]
        # Times locating the stored period in the index.
        self.index.get_loc(self.period)

    def time_shallow_copy(self):
        # [FUNCTION DOC]
        # Times creating a view of the index through the internal view method.
        self.index._view()

    def time_series_loc(self):
        # [FUNCTION DOC]
        # Times label selection of the stored period from the series.
        self.series.loc[self.period]

    def time_align(self):
        # [FUNCTION DOC]
        # Times building a frame from the full series and its first 500 elements,
        # which aligns their indexes.
        DataFrame({"a": self.series, "b": self.series[:500]})

    def time_intersection(self):
        # [FUNCTION DOC]
        # Times intersecting the first 750 index entries with the entries from 250
        # on.
        self.index[:750].intersection(self.index[250:])

    def time_unique(self):
        # [FUNCTION DOC]
        # Times computing the distinct entries of the index.
        self.index.unique()
