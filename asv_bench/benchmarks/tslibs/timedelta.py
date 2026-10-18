"""
Timedelta benchmarks that rely only on tslibs. See benchmarks.timedeltas for
Timedelta benchmarks that rely on other parts of pandas.
"""

# [MODULE DOC]
# ASV benchmarks for the timedelta scalar type using only tslibs. Public
# contents: classes TimedeltaConstructor (construction from many input kinds)
# and TimedeltaProperties (component attributes). No import-time side effects
# beyond imports.

import datetime

import numpy as np

from pandas import Timedelta


class TimedeltaConstructor:
    # [CLASS DOC]
    # Benchmarks the timedelta constructor with different kinds of input.
    # State set by setup: a numpy timedelta64 of 3600 seconds, a standard library
    # timedelta of 3600 seconds and a pandas timedelta of 3600 seconds.
    def setup(self):
        # [FUNCTION DOC]
        # Creates the three one-hour timedelta values described for the class.
        self.nptimedelta64 = np.timedelta64(3600, "s")
        self.dttimedelta = datetime.timedelta(seconds=3600)
        self.td = Timedelta(3600, unit="s")

    def time_from_int(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta from a bare integer (nanoseconds).
        Timedelta(123456789)

    def time_from_unit(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta from the integer one with the day unit.
        Timedelta(1, unit="D")

    def time_from_components(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta from keyword components: 1 day, 2 hours,
        # 3 minutes, 4 seconds, 5 milliseconds, 6 microseconds, 7 nanoseconds.
        Timedelta(
            days=1,
            hours=2,
            minutes=3,
            seconds=4,
            milliseconds=5,
            microseconds=6,
            nanoseconds=7,
        )

    def time_from_datetime_timedelta(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta from a standard library timedelta.
        Timedelta(self.dttimedelta)

    def time_from_np_timedelta(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta from a numpy timedelta64.
        Timedelta(self.nptimedelta64)

    def time_from_string(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta by parsing a human-readable string.
        Timedelta("1 days")

    def time_from_iso_format(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta by parsing an ISO 8601 duration string.
        Timedelta("P4DT12H30M5S")

    def time_from_missing(self):
        # [FUNCTION DOC]
        # Times the constructor with the missing-value string, which yields the
        # not-a-time singleton rather than a real timedelta.
        Timedelta("nat")

    def time_from_pd_timedelta(self):
        # [FUNCTION DOC]
        # Times constructing a timedelta from an existing pandas timedelta.
        Timedelta(self.td)


class TimedeltaProperties:
    # [CLASS DOC]
    # Benchmarks reading component attributes of a timedelta scalar. Uses the
    # asv cache mechanism: the cache setup method creates the timedelta and asv
    # passes it to each timing method.
    def setup_cache(self):
        # [FUNCTION DOC]
        # Builds and returns a timedelta of 365 days, 35 minutes, 25 seconds and
        # 35 milliseconds. Called once by the asv runner.
        td = Timedelta(days=365, minutes=35, seconds=25, milliseconds=35)
        return td

    def time_timedelta_days(self, td):
        # [FUNCTION DOC]
        # Times reading the days component of the cached timedelta.
        td.days

    def time_timedelta_seconds(self, td):
        # [FUNCTION DOC]
        # Times reading the seconds component of the cached timedelta.
        td.seconds

    def time_timedelta_microseconds(self, td):
        # [FUNCTION DOC]
        # Times reading the microseconds component of the cached timedelta.
        td.microseconds

    def time_timedelta_nanoseconds(self, td):
        # [FUNCTION DOC]
        # Times reading the nanoseconds component of the cached timedelta.
        td.nanoseconds
