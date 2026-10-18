"""
ipython analogue:

tr = TimeResolution()
mi = pd.MultiIndex.from_product(tr.params[:-1] + ([str(x) for x in tr.params[-1]],))
df = pd.DataFrame(np.nan, index=mi, columns=["mean", "stdev"])

for unit in tr.params[0]:
    for size in tr.params[1]:
        for tz in tr.params[2]:
            tr.setup(unit, size, tz)
            key = (unit, size, str(tz))
            print(key)

            val = %timeit -o tr.time_get_resolution(unit, size, tz)

            df.loc[key] = (val.average, val.stdev)

"""

# [MODULE DOC]
# ASV benchmark for the function that infers the coarsest resolution of an
# int64 timestamp array. Public contents: class TimeResolution. The resolution
# function is imported from the tslibs package with a fallback to its own
# submodule for older versions (import-time try/except; no other side effects).
# Uses the sizes, timezones and local timezone object from the sibling tslib
# benchmark module.

import numpy as np

try:
    from pandas._libs.tslibs import get_resolution
except ImportError:
    from pandas._libs.tslibs.resolution import get_resolution

from .tslib import (
    _sizes,
    _tzs,
    tzlocal_obj,
)


class TimeResolution:
    # [CLASS DOC]
    # Benchmark class parameterized over datetime unit (day, hour, minute,
    # second, microsecond, nanosecond), array size and timezone. State set by
    # setup: an int64 array of nanosecond-based values.
    params = (
        ["D", "h", "m", "s", "us", "ns"],
        _sizes,
        _tzs,
    )
    param_names = ["unit", "size", "tz"]

    def setup(self, unit, size, tz):
        # [FUNCTION DOC]
        # Creates random integers from zero to nine of the requested size,
        # reinterprets them as datetime64 values of the given unit, converts
        # them to nanosecond datetime64 and stores the int64 view. Thus the data
        # has the coarseness implied by the unit.
        # Raises a not-implemented error (skipping the benchmark) when size is
        # one million and tz is the local timezone object, as that is too slow.
        if size == 10**6 and tz is tzlocal_obj:
            # tzlocal is cumbersomely slow, so skip to keep runtime in check
            raise NotImplementedError

        arr = np.random.randint(0, 10, size=size, dtype="i8")
        arr = arr.view(f"M8[{unit}]").astype("M8[ns]").view("i8")
        self.i8data = arr

    def time_get_resolution(self, unit, size, tz):
        # [FUNCTION DOC]
        # Times computing the resolution of the prepared array in the given
        # timezone. Parameters unit and size are unused here.
        get_resolution(self.i8data, tz)
