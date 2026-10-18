# [MODULE DOC]
# ASV benchmarks for time zone conversion and localization of int64 timestamp
# arrays. Public contents: class TimeTZConvert and the module flag old_sig,
# which records whether the imported conversion function has the old signature
# (taking a source zone as well); the function is imported from the tslibs
# package, falling back to the conversion module and then to an older function
# name, all via nested try/except at import time (the only side effects).
# Uses the sizes, timezones and local timezone object from the sibling tslib
# benchmark module.
from datetime import timezone

import numpy as np

from pandas._libs.tslibs.tzconversion import tz_localize_to_utc

from .tslib import (
    _sizes,
    _tzs,
    tzlocal_obj,
)

try:
    old_sig = False
    from pandas._libs.tslibs import tz_convert_from_utc
except ImportError:
    try:
        old_sig = False
        from pandas._libs.tslibs.tzconversion import tz_convert_from_utc
    except ImportError:
        old_sig = True
        from pandas._libs.tslibs.tzconversion import tz_convert as tz_convert_from_utc


class TimeTZConvert:
    # [CLASS DOC]
    # Benchmarks converting from UTC and localizing to UTC for int64 arrays.
    # Parameterized over array size and every timezone in the shared list except
    # none. State set by setup: an int64 array.
    params = [
        _sizes,
        [x for x in _tzs if x is not None],
    ]
    param_names = ["size", "tz"]

    def setup(self, size, tz):
        # [FUNCTION DOC]
        # Creates an int64 array of the given size with random integers from
        # zero to nine. Raises a not-implemented error (asv skip) when size is
        # one million and tz is the local timezone object, which is too slow.
        if size == 10**6 and tz is tzlocal_obj:
            # tzlocal is cumbersomely slow, so skip to keep runtime in check
            raise NotImplementedError

        arr = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data = arr

    def time_tz_convert_from_utc(self, size, tz):
        # [FUNCTION DOC]
        # Times converting the array from UTC to the timezone, i.e. the work
        # done when removing a timezone from a datetime index. When the old
        # function signature was detected at import time the source zone (UTC)
        # is passed explicitly, otherwise only the destination. Size is unused.
        # effectively:
        #  dti = DatetimeIndex(self.i8data, tz=tz)
        #  dti.tz_localize(None)
        if old_sig:
            tz_convert_from_utc(self.i8data, timezone.utc, tz)
        else:
            tz_convert_from_utc(self.i8data, tz)

    def time_tz_localize_to_utc(self, size, tz):
        # [FUNCTION DOC]
        # Times localizing the naive array in the timezone and converting to UTC,
        # with ambiguous and nonexistent local times mapped to the missing
        # sentinel. Size is unused.
        # effectively:
        #  dti = DatetimeIndex(self.i8data)
        #  dti.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        tz_localize_to_utc(self.i8data, tz, ambiguous="NaT", nonexistent="NaT")
