# [MODULE DOC]
# ASV benchmark module measuring string formatting performance for datetime,
# period and business-hour offset data.
# Public contents: the benchmark classes DatetimeStrftime, PeriodStrftime and
# BusinessHourStrftime. Each class builds a frame in its setup method and each
# method whose name starts with the time prefix is timed by the asv runner.
# Relationships: exercises the datetime accessor string formatting method, the
# string conversion of columns, and string conversion of offset objects from the
# top-level pandas package.
# Import-time side effects: none beyond importing numpy and pandas.
import numpy as np

import pandas as pd
from pandas import offsets


class DatetimeStrftime:
    # [CLASS DOC]
    # Benchmark class for formatting datetime64 values as strings.
    # Class attributes: timeout (seconds allowed per benchmark), params (the list
    # of row counts) and param_names (a single name, the observation count).
    # State set by setup: a data frame with a datetime column holding a
    # timestamp with a time part, a column of midnight dates, and a float column.
    # Each benchmark method formats one of the two datetime columns.
    timeout = 1500
    params = [1000, 10000]
    param_names = ["nobs"]

    def setup(self, nobs):
        # [FUNCTION DOC]
        # Builds the data frame stored on the instance. Parameter nobs is the
        # number of rows. Column dt repeats one datetime with a time of day,
        # column d repeats one date at midnight, column r repeats one random
        # float drawn once. Returns nothing.
        d = "2018-11-29"
        dt = "2018-11-26 11:18:27.0"
        self.data = pd.DataFrame(
            {
                "dt": [np.datetime64(dt)] * nobs,
                "d": [np.datetime64(d)] * nobs,
                "r": [np.random.uniform()] * nobs,
            }
        )

    def time_frame_date_to_str(self, nobs):
        # [FUNCTION DOC]
        # Times converting the midnight date column to string dtype with the
        # generic type conversion. Parameter nobs is unused. Result discarded.
        self.data["d"].astype(str)

    def time_frame_date_formatting_default(self, nobs):
        # [FUNCTION DOC]
        # Times the datetime accessor formatting method on the date column with
        # no format given (default format). Parameter nobs is unused.
        self.data["d"].dt.strftime(date_format=None)

    def time_frame_date_formatting_default_explicit(self, nobs):
        # [FUNCTION DOC]
        # Times formatting the date column with an explicit year-month-day
        # format equal to the default output. Parameter nobs is unused.
        self.data["d"].dt.strftime(date_format="%Y-%m-%d")

    def time_frame_date_formatting_custom(self, nobs):
        # [FUNCTION DOC]
        # Times formatting the date column with a custom format using triple
        # dashes as separators. Parameter nobs is unused.
        self.data["d"].dt.strftime(date_format="%Y---%m---%d")

    def time_frame_datetime_to_str(self, nobs):
        # [FUNCTION DOC]
        # Times converting the datetime column (with time of day) to string
        # dtype with the generic type conversion. Parameter nobs is unused.
        self.data["dt"].astype(str)

    def time_frame_datetime_formatting_default(self, nobs):
        # [FUNCTION DOC]
        # Times the datetime accessor formatting method on the datetime column
        # with no format given. Parameter nobs is unused.
        self.data["dt"].dt.strftime(date_format=None)

    def time_frame_datetime_formatting_default_explicit_date_only(self, nobs):
        # [FUNCTION DOC]
        # Times formatting the datetime column with an explicit date-only format
        # (year-month-day). Parameter nobs is unused.
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d")

    def time_frame_datetime_formatting_default_explicit(self, nobs):
        # [FUNCTION DOC]
        # Times formatting the datetime column with an explicit date plus
        # seconds-resolution time format. Parameter nobs is unused.
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S")

    def time_frame_datetime_formatting_default_with_float(self, nobs):
        # [FUNCTION DOC]
        # Times formatting the datetime column with a format that adds
        # microseconds after the seconds. Parameter nobs is unused.
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")

    def time_frame_datetime_formatting_custom(self, nobs):
        # [FUNCTION DOC]
        # Times formatting the datetime column with a custom format containing
        # literal dashes between the date and time parts. Parameter nobs is unused.
        self.data["dt"].dt.strftime(date_format="%Y-%m-%d --- %H:%M:%S")


class PeriodStrftime:
    # [CLASS DOC]
    # Benchmark class for formatting period values as strings.
    # Parameterized over row count (1000 or 10000) and period frequency (daily
    # or hourly). State set by setup: a data frame with a period column, a float
    # column, and an index that is a copy of the period column, plus the
    # attribute holding the default text format for the chosen frequency.
    timeout = 1500
    params = ([1000, 10000], ["D", "h"])
    param_names = ["nobs", "freq"]

    def setup(self, nobs, freq):
        # [FUNCTION DOC]
        # Builds a frame with a period column of nobs consecutive periods of
        # the given frequency starting at the year 2000, a float column repeating
        # one random draw, and a column named i copied from the periods which is
        # then set as the index in place. Stores the default format string:
        # date only for daily, date plus hour with zero minutes for hourly.
        # KNOWN ISSUE: for any frequency other than the two listed the default
        # format attribute is never assigned, so later benchmarks would fail;
        # only the two parameterized frequencies are used so this is latent.
        self.data = pd.DataFrame(
            {
                "p": pd.period_range(start="2000-01-01", periods=nobs, freq=freq),
                "r": [np.random.uniform()] * nobs,
            }
        )
        self.data["i"] = self.data["p"]
        self.data.set_index("i", inplace=True)
        if freq == "D":
            self.default_fmt = "%Y-%m-%d"
        elif freq == "h":
            self.default_fmt = "%Y-%m-%d %H:00"

    def time_frame_period_to_str(self, nobs, freq):
        # [FUNCTION DOC]
        # Times converting the period column to string dtype with the generic
        # type conversion. Parameters are unused.
        self.data["p"].astype(str)

    def time_frame_period_formatting_default(self, nobs, freq):
        # [FUNCTION DOC]
        # Times the accessor formatting method on the period column with no
        # format given. Parameters are unused.
        self.data["p"].dt.strftime(date_format=None)

    def time_frame_period_formatting_default_explicit(self, nobs, freq):
        # [FUNCTION DOC]
        # Times formatting the period column with the explicit default format
        # string computed in setup for this frequency. Parameter nobs is unused.
        self.data["p"].dt.strftime(date_format=self.default_fmt)

    def time_frame_period_formatting_custom(self, nobs, freq):
        # [FUNCTION DOC]
        # Times formatting the period column with a custom date and time format
        # containing literal dashes. Parameters are unused.
        self.data["p"].dt.strftime(date_format="%Y-%m-%d --- %H:%M:%S")

    def time_frame_period_formatting_iso8601_strftime_Z(self, nobs, freq):
        # [FUNCTION DOC]
        # Times formatting the period column with an ISO 8601 style format ending
        # in a literal Z character. Parameters are unused.
        self.data["p"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ")

    def time_frame_period_formatting_iso8601_strftime_offset(self, nobs, freq):
        # [FUNCTION DOC]
        # Times formatting the period column with an ISO 8601 style format that
        # ends with a UTC offset directive. Per the existing docstring this path
        # is not optimized since the offset directive is not supported by the
        # internal format conversion helper. Parameters are unused.
        """Not optimized yet as %z is not supported by `convert_strftime_format`"""
        self.data["p"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%S%z")


class BusinessHourStrftime:
    # [CLASS DOC]
    # Benchmark class for converting business-hour offset objects to text.
    # Parameterized over row count (1000 or 10000). State set by setup: a data
    # frame with one object column of identical business-hour offset instances.
    timeout = 1500
    params = [1000, 10000]
    param_names = ["nobs"]

    def setup(self, nobs):
        # [FUNCTION DOC]
        # Builds a one-column frame whose column off holds nobs references to a
        # single default business-hour offset. Returns nothing.
        self.data = pd.DataFrame(
            {
                "off": [offsets.BusinessHour()] * nobs,
            }
        )

    def time_frame_offset_str(self, nobs):
        # [FUNCTION DOC]
        # Times applying the builtin string conversion to every offset in the
        # column. Parameter nobs is unused.
        self.data["off"].apply(str)

    def time_frame_offset_repr(self, nobs):
        # [FUNCTION DOC]
        # Times applying the builtin representation function to every offset in
        # the column. Parameter nobs is unused.
        self.data["off"].apply(repr)
