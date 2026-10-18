"""
ipython analogue:

tr = TimeIntsToPydatetime()
mi = pd.MultiIndex.from_product(
    tr.params[:-1] + ([str(x) for x in tr.params[-1]],)
)
df = pd.DataFrame(np.nan, index=mi, columns=["mean", "stdev"])
for box in tr.params[0]:
    for size in tr.params[1]:
        for tz in tr.params[2]:
            tr.setup(box, size, tz)
            key = (box, size, str(tz))
            print(key)
            val = %timeit -o tr.time_ints_to_pydatetime(box, size, tz)
            df.loc[key] = (val.average, val.stdev)
"""

# [MODULE DOC]
# ASV benchmark for converting int64 values to Python date-like objects, and the
# home of the shared parameter data used by the other tslibs benchmark modules.
# Public contents: tzlocal_obj (the local timezone object), _tzs (list of
# timezones: none, UTC, a fixed one-hour offset, US/Pacific, Asia/Tokyo via
# dateutil, and the local zone), _sizes (array lengths 0, 1, 100, ten thousand
# and one million) and class TimeIntsToPydatetime. The conversion function is
# imported from the tslibs package with a fallback to the older module location.
# Import-time side effects: constructs timezone objects, including a local
# timezone lookup, and a try/except on the import.

from datetime import (
    timedelta,
    timezone,
)
import zoneinfo

from dateutil.tz import (
    gettz,
    tzlocal,
)
import numpy as np

try:
    from pandas._libs.tslibs import ints_to_pydatetime
except ImportError:
    from pandas._libs.tslib import ints_to_pydatetime

tzlocal_obj = tzlocal()
_tzs = [
    None,
    timezone.utc,
    timezone(timedelta(minutes=60)),
    zoneinfo.ZoneInfo("US/Pacific"),
    gettz("Asia/Tokyo"),
    tzlocal_obj,
]
_sizes = [0, 1, 100, 10**4, 10**6]


class TimeIntsToPydatetime:
    # [CLASS DOC]
    # Benchmarks converting an int64 array to arrays of time, date, datetime or
    # timestamp objects. Parameterized over the box kind, array size and
    # timezone. State set by setup: the int64 array.
    params = (
        ["time", "date", "datetime", "timestamp"],
        _sizes,
        _tzs,
    )
    param_names = ["box", "size", "tz"]
    # TODO: fold?

    def setup(self, box, size, tz):
        # [FUNCTION DOC]
        # Creates an int64 array of the given size with random integers from
        # zero to nine.
        # Raises a not-implemented error (asv skip) in two cases: the date box
        # combined with any timezone (the timezone is ignored for dates, so the
        # case is redundant), and size one million with the last timezone in the
        # shared list (the local zone, too slow).
        if box == "date" and tz is not None:
            # tz is ignored, so avoid running redundant benchmarks
            raise NotImplementedError  # skip benchmark
        if size == 10**6 and tz is _tzs[-1]:
            # This is cumbersomely-slow, so skip to trim runtime
            raise NotImplementedError  # skip benchmark

        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr

    def time_ints_to_pydatetime(self, box, size, tz):
        # [FUNCTION DOC]
        # Times converting the int64 array to the requested box kind in the
        # given timezone. Parameter size is unused here.
        ints_to_pydatetime(self.i8data, tz, box=box)
