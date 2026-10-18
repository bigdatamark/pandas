# [MODULE DOC]
# ASV benchmark module for time series functionality: datetime index
# operations, time zone localization, index reset, frequency inference, the
# plotting date converter, iteration over time-like indexes, resampling,
# as-of lookups, sorting, scalar lookup and the datetime accessor.
# Public contents: classes DatetimeIndex, TzLocalize, ResetIndex, InferFreq,
# TimeDatetimeConverter, Iteration, ResampleDataFrame, ResampleSeries,
# ResampleDatetetime64 (name misspelled in the source), AsOf, SortIndex,
# Lookup and DatetimeAccessor. Relationships: the shared setup hook from the
# sibling common module is imported last (linter skip comment). Import-time
# side effects: a try/except selects the date converter from the matplotlib
# plotting module, falling back to the older converter module; the shared setup
# function is registered at module level by the final import.
from datetime import timedelta

import dateutil
import numpy as np

from pandas import (
    DataFrame,
    Series,
    date_range,
    period_range,
    timedelta_range,
)

from pandas.tseries.frequencies import infer_freq

try:
    from pandas.plotting._matplotlib.converter import DatetimeConverter
except ImportError:
    from pandas.tseries.converter import DatetimeConverter


class DatetimeIndex:
    # [CLASS DOC]
    # Benchmarks common operations on a datetime index of several flavors,
    # selected by the parameter index_type: a one-hour span across a daylight
    # saving fall-back (dst), a repeated-values index (repeated), a US/Eastern
    # zone-aware index (tz_aware), a local-zone index (tz_local) and a naive
    # index (tz_naive). State set by setup: the chosen index.
    # (Note the class name shadows the pandas datetime index name inside this
    # module's namespace, but it is not imported here.)
    params = ["dst", "repeated", "tz_aware", "tz_local", "tz_naive"]
    param_names = ["index_type"]

    def setup(self, index_type):
        # [FUNCTION DOC]
        # Builds all five candidate indexes eagerly and keeps the one named by
        # index_type. The dst index has one-second steps from 01:00:00 to 01:59:59
        # on 2000-10-29. The repeated index has 10000 distinct seconds each
        # repeated ten times. The three others have 100000 one-second steps from
        # the year 2000 (US/Eastern, local zone, naive).
        N = 100000
        dtidxes = {
            "dst": date_range(
                start="10/29/2000 1:00:00", end="10/29/2000 1:59:59", freq="s"
            ),
            "repeated": date_range(start="2000", periods=N // 10, freq="s").repeat(10),
            "tz_aware": date_range(start="2000", periods=N, freq="s", tz="US/Eastern"),
            "tz_local": date_range(
                start="2000", periods=N, freq="s", tz=dateutil.tz.tzlocal()
            ),
            "tz_naive": date_range(start="2000", periods=N, freq="s"),
        }
        self.index = dtidxes[index_type]

    def time_add_timedelta(self, index_type):
        # [FUNCTION DOC]
        # Times adding a two-minute timedelta to the index.
        self.index + timedelta(minutes=2)

    def time_normalize(self, index_type):
        # [FUNCTION DOC]
        # Times normalizing all timestamps to midnight.
        self.index.normalize()

    def time_unique(self, index_type):
        # [FUNCTION DOC]
        # Times computing the unique values.
        self.index.unique()

    def time_to_time(self, index_type):
        # [FUNCTION DOC]
        # Times extracting the time-of-day objects.
        self.index.time

    def time_get(self, index_type):
        # [FUNCTION DOC]
        # Times retrieving the first element by position.
        self.index[0]

    def time_timeseries_is_month_start(self, index_type):
        # [FUNCTION DOC]
        # Times computing the first-day-of-month flags.
        self.index.is_month_start

    def time_to_date(self, index_type):
        # [FUNCTION DOC]
        # Times extracting the date objects.
        self.index.date

    def time_to_pydatetime(self, index_type):
        # [FUNCTION DOC]
        # Times converting the index to an array of standard library datetimes.
        self.index.to_pydatetime()

    def time_is_dates_only(self, index_type):
        # [FUNCTION DOC]
        # Times reading the private flag saying whether all values are at
        # midnight.
        self.index._is_dates_only


class TzLocalize:
    # [CLASS DOC]
    # Benchmarks localizing a naive index containing ambiguous daylight saving
    # times to a zone, inferring ambiguity. Parameterized over the zone (none,
    # US/Eastern, UTC, a dateutil UTC object). State set by setup: the naive
    # index.
    # KNOWN ISSUE: param_names is a bare string rather than a list of names;
    # with a single parameter axis asv tolerates this but it deviates from the
    # other classes.
    params = [None, "US/Eastern", "UTC", dateutil.tz.tzutc()]
    param_names = "tz"

    def setup(self, tz):
        # [FUNCTION DOC]
        # Builds a naive one-second index on 2000-10-29 made by concatenating,
        # in order: the span 00:00:00 to 00:59:59, the span 01:00:00 to 01:59:59
        # twice (to create the ambiguous repeat around the fall-back), and the
        # span 02:00:00 to 03:00:00. Parameter tz is unused here.
        dst_rng = date_range(
            start="10/29/2000 1:00:00", end="10/29/2000 1:59:59", freq="s"
        )
        self.index = date_range(start="10/29/2000", end="10/29/2000 00:59:59", freq="s")
        self.index = self.index.append(dst_rng)
        self.index = self.index.append(dst_rng)
        self.index = self.index.append(
            date_range(start="10/29/2000 2:00:00", end="10/29/2000 3:00:00", freq="s")
        )

    def time_infer_dst(self, tz):
        # [FUNCTION DOC]
        # Times localizing the index to the zone with ambiguous times resolved by
        # inference from the ordering.
        self.index.tz_localize(tz, ambiguous="infer")


class ResetIndex:
    # [CLASS DOC]
    # Benchmarks moving a datetime index into a column, parameterized over zone
    # (none or US/Eastern). State set by setup: a two-column frame indexed by a
    # datetime index.
    # KNOWN ISSUE: param_names is a bare string rather than a list.
    params = [None, "US/Eastern"]
    param_names = "tz"

    def setup(self, tz):
        # [FUNCTION DOC]
        # Builds an hourly index of 1000 entries starting 2000-01-01 in the
        # zone, and a frame of 1000 rows by 2 columns of standard normal random
        # numbers using it as the index.
        idx = date_range(start="1/1/2000", periods=1000, freq="h", tz=tz)
        self.df = DataFrame(np.random.randn(1000, 2), index=idx)

    def time_reset_datetimeindex(self, tz):
        # [FUNCTION DOC]
        # Times resetting the index of the frame. Parameter tz is unused here.
        self.df.reset_index()


class InferFreq:
    # [CLASS DOC]
    # Benchmarks frequency inference, which depends mostly on low-level code,
    # the offsets code and the unique algorithm (per the source comment).
    # Parameterized over the frequency used to build the index: none (daily
    # index with its stored frequency cleared), daily, or business daily.
    # State set by setup: a 10000-element index from the year 1700.
    # This depends mostly on code in _libs/, tseries/, and core.algos.unique
    params = [None, "D", "B"]
    param_names = ["freq"]

    def setup(self, freq):
        # [FUNCTION DOC]
        # Builds a 10000-element index starting 1700-01-01. For a none freq the
        # index is daily and its internal stored frequency attribute is cleared
        # (by writing to the private data object) so that inference must
        # compute it; otherwise the index is built with the given frequency.
        if freq is None:
            self.idx = date_range(start="1/1/1700", freq="D", periods=10000)
            self.idx._data._freq = None
        else:
            self.idx = date_range(start="1/1/1700", freq=freq, periods=10000)

    def time_infer_freq(self, freq):
        # [FUNCTION DOC]
        # Times inferring the frequency of the index. Parameter freq is unused
        # here.
        infer_freq(self.idx)


class TimeDatetimeConverter:
    # [CLASS DOC]
    # Benchmarks the plotting date converter on a large minute-frequency index.
    # No parameters. State set by setup: the index.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a minute-frequency index of 100000 entries from 2000-01-01.
        N = 100000
        self.rng = date_range(start="1/1/2000", periods=N, freq="min")

    def time_convert(self):
        # [FUNCTION DOC]
        # Times the converter's static convert method on the index with no unit
        # and no axis information.
        DatetimeConverter.convert(self.rng, None, None)


class Iteration:
    # [CLASS DOC]
    # Benchmarks iterating over a time-like index. Parameterized over the index
    # constructor itself (datetime, period, timedelta range builders). State set
    # by setup: a million-element minute-frequency index and a loop exit
    # position of 10000.
    params = [date_range, period_range, timedelta_range]
    param_names = ["time_index"]

    def setup(self, time_index):
        # [FUNCTION DOC]
        # Builds one million minute-frequency entries with the given builder:
        # starting at zero for the timedelta builder, otherwise at 2014-01-01.
        # Also stores the early-exit position 10000.
        N = 10**6
        if time_index is timedelta_range:
            self.idx = time_index(start=0, freq="min", periods=N)
        else:
            self.idx = time_index(start="20140101", freq="min", periods=N)
        self.exit = 10000

    def time_iter(self, time_index):
        # [FUNCTION DOC]
        # Times a full loop over all elements, discarding each.
        for _ in self.idx:
            pass

    def time_iter_preexit(self, time_index):
        # [FUNCTION DOC]
        # Times looping with a counter and stopping once the counter exceeds the
        # exit position, i.e. after visiting 10002 elements.
        for i, _ in enumerate(self.idx):
            if i > self.exit:
                break


class ResampleDataFrame:
    # [CLASS DOC]
    # Benchmarks aggregating a frame resampled to one-second bins, parameterized
    # over the aggregation (max, mean, min). State set by setup: a bound
    # aggregation callable of the resampler.
    params = ["max", "mean", "min"]
    param_names = ["method"]

    def setup(self, method):
        # [FUNCTION DOC]
        # Builds a frame of 100000 rows by 2 normal random columns indexed at
        # 50-millisecond steps from 2013-01-01, resamples to one second, and
        # stores the aggregation method looked up by name.
        rng = date_range(start="20130101", periods=100000, freq="50ms")
        df = DataFrame(np.random.randn(100000, 2), index=rng)
        self.resample = getattr(df.resample("1s"), method)

    def time_method(self, method):
        # [FUNCTION DOC]
        # Times calling the stored aggregation. Parameter method is unused here.
        self.resample()


class ResampleSeries:
    # [CLASS DOC]
    # Benchmarks aggregating a resampled series, parameterized over index kind
    # (period or datetime), target frequency (five minutes or one day) and
    # aggregation (mean or open-high-low-close). State set by setup: a bound
    # aggregation callable.
    params = (["period", "datetime"], ["5min", "1D"], ["mean", "ohlc"])
    param_names = ["index", "freq", "method"]

    def setup(self, index, freq, method):
        # [FUNCTION DOC]
        # Builds both candidate minute-frequency indexes spanning 2000-01-01 to
        # 2001-01-01 eagerly, keeps the one named by index, builds a series of
        # standard normal values on it, resamples to freq and stores the
        # aggregation method looked up by name.
        indexes = {
            "period": period_range(start="1/1/2000", end="1/1/2001", freq="min"),
            "datetime": date_range(start="1/1/2000", end="1/1/2001", freq="min"),
        }
        idx = indexes[index]
        ts = Series(np.random.randn(len(idx)), index=idx)
        self.resample = getattr(ts.resample(freq), method)

    def time_resample(self, index, freq, method):
        # [FUNCTION DOC]
        # Times calling the stored aggregation. Parameters are unused here.
        self.resample()


class ResampleDatetetime64:
    # [CLASS DOC]
    # Benchmarks resampling a series of datetime64 values (regression check for
    # issue 7754). No parameters. State set by setup: the series.
    # GH 7754
    def setup(self):
        # [FUNCTION DOC]
        # Builds an index from 2000-01-01 00:00:00 to 10:00:00 at 555000
        # microsecond steps and a series whose values are all the integer five
        # interpreted as nanosecond datetime64 values.
        rng3 = date_range(
            start="2000-01-01 00:00:00", end="2000-01-01 10:00:00", freq="555000us"
        )
        self.dt_ts = Series(5, rng3, dtype="datetime64[ns]")

    def time_resample(self):
        # [FUNCTION DOC]
        # Times resampling to one-second bins and taking the last value in each.
        self.dt_ts.resample("1s").last()


class AsOf:
    # [CLASS DOC]
    # Benchmarks the as-of lookup (last valid value at or before a label) on a
    # frame or series, parameterized over the container kind. State set by
    # setup: three containers (full data; missing block in the middle; missing
    # tail), a long array of query dates, and three scalar dates (first, last
    # and ten days before the first).
    params = ["DataFrame", "Series"]
    param_names = ["constructor"]

    def setup(self, constructor):
        # [FUNCTION DOC]
        # Creates both candidate containers eagerly (10000 rows; the frame has 10
        # columns), keeps the one named by constructor, and gives it a 53-second
        # frequency index from 1990-01-01. Makes copies with rows 250 to 4999
        # set to missing and with the last 5000 rows set to missing. Builds
        # 100000 query dates at 5-second steps from the same start, and records
        # the first date, the last date, and a date ten days before the first.
        N = 10000
        M = 10
        rng = date_range(start="1/1/1990", periods=N, freq="53s")
        data = {
            "DataFrame": DataFrame(np.random.randn(N, M)),
            "Series": Series(np.random.randn(N)),
        }
        self.ts = data[constructor]
        self.ts.index = rng
        self.ts2 = self.ts.copy()
        self.ts2.iloc[250:5000] = np.nan
        self.ts3 = self.ts.copy()
        self.ts3.iloc[-5000:] = np.nan
        self.dates = date_range(start="1/1/1990", periods=N * 10, freq="5s")
        self.date = self.dates[0]
        self.date_last = self.dates[-1]
        self.date_early = self.date - timedelta(10)

    # test speed of pre-computing NAs.
    def time_asof(self, constructor):
        # [FUNCTION DOC]
        # Times the as-of lookup of the whole date array on the full-data
        # container (includes pre-computing missing-value information).
        self.ts.asof(self.dates)

    # should be roughly the same as above.
    def time_asof_nan(self, constructor):
        # [FUNCTION DOC]
        # Times the as-of lookup of the date array on the container with a
        # missing block; expected to take about as long as the plain case.
        self.ts2.asof(self.dates)

    # test speed of the code path for a scalar index
    # without *while* loop
    def time_asof_single(self, constructor):
        # [FUNCTION DOC]
        # Times a single-label as-of lookup of the first date on the full-data
        # container (the path with no scanning loop).
        self.ts.asof(self.date)

    # test speed of the code path for a scalar index
    # before the start. should be the same as above.
    def time_asof_single_early(self, constructor):
        # [FUNCTION DOC]
        # Times a single-label lookup of a date before the start of the data;
        # expected to match the previous case in speed.
        self.ts.asof(self.date_early)

    # test the speed of the code path for a scalar index
    # with a long *while* loop. should still be much
    # faster than pre-computing all the NAs.
    def time_asof_nan_single(self, constructor):
        # [FUNCTION DOC]
        # Times a single-label lookup of the last date on the container whose
        # tail is missing, forcing a long backwards scan.
        self.ts3.asof(self.date_last)


class SortIndex:
    # [CLASS DOC]
    # Benchmarks sorting and slicing a series with a datetime index,
    # parameterized over whether the index is already monotonic. State set by
    # setup: a series of 100000 values.
    params = [True, False]
    param_names = ["monotonic"]

    def setup(self, monotonic):
        # [FUNCTION DOC]
        # Builds a series of standard normal values on a one-second index of
        # 100000 entries; when monotonic is false the series is shuffled by
        # sampling all rows in random order.
        N = 10**5
        idx = date_range(start="1/1/2000", periods=N, freq="s")
        self.s = Series(np.random.randn(N), index=idx)
        if not monotonic:
            self.s = self.s.sample(frac=1)

    def time_sort_index(self, monotonic):
        # [FUNCTION DOC]
        # Times sorting the series by its index.
        self.s.sort_index()

    def time_get_slice(self, monotonic):
        # [FUNCTION DOC]
        # Times taking the first 10000 rows by positional slice.
        self.s[:10000]


class Lookup:
    # [CLASS DOC]
    # Benchmarks a single-label lookup on a very large datetime-indexed series
    # followed by cleanup of the index engine. No parameters.
    def setup(self):
        # [FUNCTION DOC]
        # Builds a series of 1500000 ones on a one-second index from 2000-01-01
        # and records the label at the middle position.
        N = 1500000
        rng = date_range(start="1/1/2000", periods=N, freq="s")
        self.ts = Series(1, index=rng)
        self.lookup_val = rng[N // 2]

    def time_lookup_and_cleanup(self):
        # [FUNCTION DOC]
        # Times looking up the stored label and then calling the private index
        # cleanup method, so the engine built by the lookup is released each run.
        self.ts[self.lookup_val]
        self.ts.index._cleanup()


class DatetimeAccessor:
    # [CLASS DOC]
    # Benchmarks the datetime accessor of a series, parameterized over zone
    # (none, US/Eastern, UTC, a dateutil UTC object). State set by setup: a
    # series of 100000 minute-spaced timestamps.
    # KNOWN ISSUE: param_names is a bare string rather than a list.
    params = [None, "US/Eastern", "UTC", dateutil.tz.tzutc()]
    param_names = "tz"

    def setup(self, tz):
        # [FUNCTION DOC]
        # Builds a series of 100000 timestamps at one-minute steps from
        # 2000-01-01 in the given zone.
        N = 100000
        self.series = Series(date_range(start="1/1/2000", periods=N, freq="min", tz=tz))

    def time_dt_accessor(self, tz):
        # [FUNCTION DOC]
        # Times creating the accessor object.
        self.series.dt

    def time_dt_accessor_normalize(self, tz):
        # [FUNCTION DOC]
        # Times normalizing all values to midnight.
        self.series.dt.normalize()

    def time_dt_accessor_month_name(self, tz):
        # [FUNCTION DOC]
        # Times computing month names.
        self.series.dt.month_name()

    def time_dt_accessor_day_name(self, tz):
        # [FUNCTION DOC]
        # Times computing weekday names.
        self.series.dt.day_name()

    def time_dt_accessor_time(self, tz):
        # [FUNCTION DOC]
        # Times extracting time-of-day objects.
        self.series.dt.time

    def time_dt_accessor_date(self, tz):
        # [FUNCTION DOC]
        # Times extracting date objects.
        self.series.dt.date

    def time_dt_accessor_year(self, tz):
        # [FUNCTION DOC]
        # Times extracting the year numbers.
        self.series.dt.year


from .pandas_vb_common import setup  # noqa: F401 isort:skip
