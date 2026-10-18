"""
Period benchmarks that rely only on tslibs. See benchmarks.period for
Period benchmarks that rely on other parts of pandas.
"""

# [MODULE DOC]
# ASV benchmarks for the period scalar type and period/datetime64 array
# conversion helpers from tslibs.
# Public contents: classes PeriodProperties, PeriodUnaryMethods,
# PeriodConstructor, TimePeriodArrToDT64Arr and TimeDT64ArrToPeriodArr, plus the
# module-level list _freq_ints of integer frequency codes (annual, quarterly,
# weekly and finer, including some anchored variants). Relationships: uses the
# size, timezone and local timezone object from the sibling tslib benchmark
# module, and the offset-from-string helper. Import-time side effects: a
# try/except picks the datetime-to-period array function from the vectorized
# module, falling back to the period module for older versions.

import numpy as np

from pandas._libs.tslibs.period import (
    Period,
    periodarr_to_dt64arr,
)

from pandas.tseries.frequencies import to_offset

from .tslib import (
    _sizes,
    _tzs,
    tzlocal_obj,
)

try:
    from pandas._libs.tslibs.vectorized import dt64arr_to_periodarr
except ImportError:
    from pandas._libs.tslibs.period import dt64arr_to_periodarr


class PeriodProperties:
    # [CLASS DOC]
    # Benchmarks reading a single attribute of a period scalar, parameterized
    # over frequency (monthly or minutely) and the attribute name (fifteen
    # attributes such as year, quarter, week, start time and end time).
    # State set by setup: a period at 2012-06-01.
    params = (
        ["M", "min"],
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "is_leap_year",
            "quarter",
            "qyear",
            "week",
            "daysinmonth",
            "dayofweek",
            "dayofyear",
            "start_time",
            "end_time",
        ],
    )
    param_names = ["freq", "attr"]

    def setup(self, freq, attr):
        # [FUNCTION DOC]
        # Creates the period for 2012-06-01 with the given frequency. Parameter
        # attr is unused here.
        self.per = Period("2012-06-01", freq=freq)

    def time_property(self, freq, attr):
        # [FUNCTION DOC]
        # Times reading the attribute named by attr from the period using
        # dynamic attribute lookup. Parameter freq is unused here.
        getattr(self.per, attr)


class PeriodUnaryMethods:
    # [CLASS DOC]
    # Benchmarks conversion and formatting methods on a period scalar,
    # parameterized over frequency (monthly or minutely). State set by setup:
    # the period at 2012-06-01 and the default format string for the frequency.
    params = ["M", "min"]
    param_names = ["freq"]

    def setup(self, freq):
        # [FUNCTION DOC]
        # Creates the period and stores the default format: year-month for
        # monthly, year-month-day hour:minute for minutely. For other
        # frequencies no default format is stored.
        self.per = Period("2012-06-01", freq=freq)
        if freq == "M":
            self.default_fmt = "%Y-%m"
        elif freq == "min":
            self.default_fmt = "%Y-%m-%d %H:%M"

    def time_to_timestamp(self, freq):
        # [FUNCTION DOC]
        # Times converting the period to a timestamp.
        self.per.to_timestamp()

    def time_now(self, freq):
        # [FUNCTION DOC]
        # Times the class-level constructor for the current period, called
        # through the instance, with the given frequency.
        self.per.now(freq)

    def time_asfreq(self, freq):
        # [FUNCTION DOC]
        # Times converting the period to the yearly frequency.
        self.per.asfreq("Y")

    def time_str(self, freq):
        # [FUNCTION DOC]
        # Times the builtin string conversion of the period.
        str(self.per)

    def time_repr(self, freq):
        # [FUNCTION DOC]
        # Times the builtin representation of the period.
        repr(self.per)

    def time_strftime_default(self, freq):
        # [FUNCTION DOC]
        # Times formatting the period with no format given (default format).
        self.per.strftime(None)

    def time_strftime_default_explicit(self, freq):
        # [FUNCTION DOC]
        # Times formatting the period with the explicit default format string
        # computed in setup.
        self.per.strftime(self.default_fmt)

    def time_strftime_custom(self, freq):
        # [FUNCTION DOC]
        # Times formatting the period with a custom format giving abbreviated
        # month, day, year and full weekday name.
        self.per.strftime("%b. %d, %Y was a %A")


class PeriodConstructor:
    # [CLASS DOC]
    # Benchmarks the period constructor with a frequency given either as a
    # string or as an offset object. Parameterized over frequency (daily only)
    # and a flag saying whether to convert it to an offset first.
    params = [["D"], [True, False]]
    param_names = ["freq", "is_offset"]

    def setup(self, freq, is_offset):
        # [FUNCTION DOC]
        # Stores on the instance either the offset parsed from the frequency
        # string (when is_offset is true) or the plain string itself.
        if is_offset:
            self.freq = to_offset(freq)
        else:
            self.freq = freq

    def time_period_constructor(self, freq, is_offset):
        # [FUNCTION DOC]
        # Times constructing a period for 2012-06-01 from the parameter freq.
        # KNOWN ISSUE: the call uses the raw parameter freq (always the string)
        # rather than the value prepared in setup and stored on the instance, so
        # the offset-object variant is never actually benchmarked.
        Period("2012-06-01", freq=freq)


_freq_ints = [
    1000,
    1011,  # Annual - November End
    2000,
    2011,  # Quarterly - November End
    3000,
    4000,
    4006,  # Weekly - Saturday End
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    11000,
    12000,
]


class TimePeriodArrToDT64Arr:
    # [CLASS DOC]
    # Benchmarks converting an int64 array of period ordinals to datetime64
    # values. Parameterized over array size and integer frequency code. State
    # set by setup: the ordinal array.
    params = [
        _sizes,
        _freq_ints,
    ]
    param_names = ["size", "freq"]

    def setup(self, size, freq):
        # [FUNCTION DOC]
        # Creates an int64 array of the ordinals zero to nine, each repeated
        # size divided by ten times. Parameter freq is unused here.
        arr = np.arange(10, dtype="i8").repeat(size // 10)
        self.i8values = arr

    def time_periodarray_to_dt64arr(self, size, freq):
        # [FUNCTION DOC]
        # Times converting the ordinal array to datetime64 values for the
        # integer frequency code. Parameter size is unused here.
        periodarr_to_dt64arr(self.i8values, freq)


class TimeDT64ArrToPeriodArr:
    # [CLASS DOC]
    # Benchmarks converting an int64 array of datetime64 values to period
    # ordinals, parameterized over array size, integer frequency code and
    # timezone. State set by setup: the int64 array.
    params = [
        _sizes,
        _freq_ints,
        _tzs,
    ]
    param_names = ["size", "freq", "tz"]

    def setup(self, size, freq, tz):
        # [FUNCTION DOC]
        # Creates an int64 array of ten consecutive values starting at two to
        # the power fifty-five, each repeated size divided by ten times. Large
        # values are used because small ones make the conversion return the
        # error sentinel (minus one) for the yearly frequency, which would
        # artificially slow the functions under test.
        # Raises a not-implemented error (skipping the benchmark) when size is
        # one million and tz is the local timezone object, since that is very slow.
        if size == 10**6 and tz is tzlocal_obj:
            # tzlocal is cumbersomely slow, so skip to keep runtime in check
            raise NotImplementedError

        # we pick 2**55 because smaller values end up returning
        # -1 from npy_datetimestruct_to_datetime with NPY_FR_Y frequency
        # this artificially slows down functions since -1 is also the
        # error sentinel
        arr = np.arange(2**55, 2**55 + 10, dtype="i8").repeat(size // 10)
        self.i8values = arr

    def time_dt64arr_to_periodarr(self, size, freq, tz):
        # [FUNCTION DOC]
        # Times converting the int64 datetime values to period ordinals for the
        # integer frequency code in the given timezone. Size is unused here.
        dt64arr_to_periodarr(self.i8values, freq, tz)
