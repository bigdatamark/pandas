# [MODULE DOC]
# ASV benchmarks for the datetime and timedelta field extraction routines in the
# tslibs fields extension module, operating on raw int64 arrays.
# Public contents: classes TimeGetTimedeltaField, TimeGetDateField and
# TimeGetStartEndField. Relationships: imports the size list from the sibling
# tslib benchmark module, and imports the shared setup hook from the parent
# benchmarks package at the very end (the import-ordering linter is told to
# skip that line). Import-time side effects: none beyond imports; the final
# import makes asv call the shared setup before each benchmark.
import numpy as np

from pandas._libs.tslibs.fields import (
    get_date_field,
    get_start_end_field,
    get_timedelta_field,
)

from .tslib import _sizes


class TimeGetTimedeltaField:
    # [CLASS DOC]
    # Benchmarks extraction of a timedelta component from an int64 array.
    # Parameterized over array size (from the shared size list) and the field
    # name (seconds, microseconds or nanoseconds). State set by setup: one array
    # of small non-negative values and one array of negative values within one day
    # in nanoseconds.
    params = [
        _sizes,
        ["seconds", "microseconds", "nanoseconds"],
    ]
    param_names = ["size", "field"]

    def setup(self, size, field):
        # [FUNCTION DOC]
        # Creates two int64 arrays of length size: random integers in the
        # range zero to nine, and random negative values between minus one day
        # in nanoseconds and zero. Parameter field is unused here.
        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr
        arr = np.random.randint(-86400 * 1_000_000_000, 0, size=size, dtype="i8")
        self.i8data_negative = arr

    def time_get_timedelta_field(self, size, field):
        # [FUNCTION DOC]
        # Times extracting the named field from the non-negative array.
        # Parameter size is unused here.
        get_timedelta_field(self.i8data, field)

    def time_get_timedelta_field_negative_td(self, size, field):
        # [FUNCTION DOC]
        # Times extracting the named field from the negative timedelta array.
        # Parameter size is unused here.
        get_timedelta_field(self.i8data_negative, field)


class TimeGetDateField:
    # [CLASS DOC]
    # Benchmarks extraction of a calendar field from an int64 array of
    # datetimes. Parameterized over array size and over fourteen field names:
    # year, month, day, hour, minute, second, microsecond, nanosecond, day of
    # year, day of week, week of year, quarter, days in month, is leap year.
    # State set by setup: one random int64 array.
    params = [
        _sizes,
        [
            "Y",
            "M",
            "D",
            "h",
            "m",
            "s",
            "us",
            "ns",
            "doy",
            "dow",
            "woy",
            "q",
            "dim",
            "is_leap_year",
        ],
    ]
    param_names = ["size", "field"]

    def setup(self, size, field):
        # [FUNCTION DOC]
        # Creates an int64 array of length size with random integers from zero
        # to nine (tiny nanosecond offsets from the epoch). Field is unused.
        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr

    def time_get_date_field(self, size, field):
        # [FUNCTION DOC]
        # Times extracting the named field from the array. Size is unused.
        get_date_field(self.i8data, field)


class TimeGetStartEndField:
    # [CLASS DOC]
    # Benchmarks the start-or-end-of-period boolean field extraction.
    # Parameterized over array size, side (start or end), period (month, quarter
    # or year), frequency string (business day, none, quarter start) and the
    # fiscal month keyword (12, 3 or 5). State set by setup: a random int64
    # array and the attribute name built from period and side.
    params = [
        _sizes,
        ["start", "end"],
        ["month", "quarter", "year"],
        ["B", None, "QS"],
        [12, 3, 5],
    ]
    param_names = ["size", "side", "period", "freqstr", "month_kw"]

    def setup(self, size, side, period, freqstr, month_kw):
        # [FUNCTION DOC]
        # Creates the int64 array (random integers zero to nine) and stores the
        # field name formed as the prefix is, then the period, an underscore and
        # the side. Parameters freqstr and month_kw are unused here.
        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr

        self.attrname = f"is_{period}_{side}"

    def time_get_start_end_field(self, size, side, period, freqstr, month_kw):
        # [FUNCTION DOC]
        # Times extracting the boolean start/end field named by the attribute
        # from the array, passing the frequency string and the fiscal month
        # keyword. Parameters size, side and period are unused here.
        get_start_end_field(self.i8data, self.attrname, freqstr, month_kw=month_kw)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
