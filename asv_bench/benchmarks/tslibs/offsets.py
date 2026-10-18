"""
offsets benchmarks that rely only on tslibs.  See benchmarks.offset for
offsets benchmarks that rely on other parts of pandas.
"""

# [MODULE DOC]
# ASV benchmarks for date offset objects using only tslibs and the offsets and
# holiday modules. Public module-level data: hcal (a US federal holiday
# calendar instance), non_apply (offsets which, per the source comment, raise a
# not-implemented error for the vectorized apply path), other_offsets (offsets
# that support it) and offset_objs (the concatenation of both lists, used as
# the parameter list of the benchmark classes). Public classes: OnOffset and
# OffestDatetimeArithmetic (the class name is misspelled in the source).
# Import-time side effects: imports the holiday module (swallowing an import
# error) and constructs offset objects and a holiday calendar.
# KNOWN ISSUE: if the holiday module import fails, the error is swallowed but the
# next statement uses the module anyway and would raise a name error, so the
# guard is ineffective.
# KNOWN ISSUE: the non_apply list contains the custom business month end
# offset with the holiday calendar twice, producing a duplicate parameter.

from datetime import datetime

import numpy as np

from pandas import offsets

try:
    import pandas.tseries.holiday
except ImportError:
    pass

hcal = pandas.tseries.holiday.USFederalHolidayCalendar()
# These offsets currently raise a NotImplementedError with .apply_index()
non_apply = [
    offsets.Day(),
    offsets.BYearEnd(),
    offsets.BYearBegin(),
    offsets.BQuarterEnd(),
    offsets.BQuarterBegin(),
    offsets.BMonthEnd(),
    offsets.BMonthBegin(),
    offsets.CustomBusinessDay(),
    offsets.CustomBusinessDay(calendar=hcal),
    offsets.CustomBusinessMonthBegin(calendar=hcal),
    offsets.CustomBusinessMonthEnd(calendar=hcal),
    offsets.CustomBusinessMonthEnd(calendar=hcal),
]
other_offsets = [
    offsets.YearEnd(),
    offsets.YearBegin(),
    offsets.QuarterEnd(),
    offsets.QuarterBegin(),
    offsets.MonthEnd(),
    offsets.MonthBegin(),
    offsets.DateOffset(months=2, days=2),
    offsets.BusinessDay(),
    offsets.SemiMonthEnd(),
    offsets.SemiMonthBegin(),
]
offset_objs = non_apply + other_offsets


class OnOffset:
    # [CLASS DOC]
    # Benchmarks the on-offset membership check of each offset in the shared
    # offset list (one benchmark instance per offset). State set by setup: a
    # list of calendar dates.
    params = offset_objs
    param_names = ["offset"]

    def setup(self, offset):
        # [FUNCTION DOC]
        # Builds the list of dates for October, November and December 2016 on
        # days 1, 2, 3, 28, 29, 30 and 31, excluding the nonexistent November
        # 31. Parameter offset is unused.
        self.dates = [
            datetime(2016, m, d)
            for m in [10, 11, 12]
            for d in [1, 2, 3, 28, 29, 30, 31]
            if not (m == 11 and d == 31)
        ]

    def time_on_offset(self, offset):
        # [FUNCTION DOC]
        # Times checking, for every prepared date, whether the date lies on the
        # given offset. Results are discarded.
        for date in self.dates:
            offset.is_on_offset(date)


class OffestDatetimeArithmetic:
    # [CLASS DOC]
    # Benchmarks adding and subtracting each offset (single and scaled by ten)
    # to and from a standard datetime, and adding to a numpy datetime scalar.
    # State set by setup: one datetime and one numpy datetime64 value.
    params = offset_objs
    param_names = ["offset"]

    def setup(self, offset):
        # [FUNCTION DOC]
        # Stores a datetime for 2011-01-01 and a numpy datetime64 for 2011-01-01
        # at 09:00 UTC. Parameter offset is unused.
        self.date = datetime(2011, 1, 1)
        self.dt64 = np.datetime64("2011-01-01 09:00Z")

    def time_add_np_dt64(self, offset):
        # [FUNCTION DOC]
        # Times adding the numpy datetime64 value to the offset (offset on the
        # left side).
        offset + self.dt64

    def time_add(self, offset):
        # [FUNCTION DOC]
        # Times adding the offset to the datetime (datetime on the left side).
        self.date + offset

    def time_add_10(self, offset):
        # [FUNCTION DOC]
        # Times adding ten multiples of the offset to the datetime.
        self.date + (10 * offset)

    def time_subtract(self, offset):
        # [FUNCTION DOC]
        # Times subtracting the offset from the datetime.
        self.date - offset

    def time_subtract_10(self, offset):
        # [FUNCTION DOC]
        # Times subtracting ten multiples of the offset from the datetime.
        self.date - (10 * offset)
