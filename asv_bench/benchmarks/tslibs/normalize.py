# [MODULE DOC]
# ASV benchmarks for normalizing int64 timestamps to midnight and for checking
# whether an int64 timestamp array is already normalized.
# Public contents: class Normalize. The two low-level functions are imported
# from the tslibs package, falling back to the conversion submodule for older
# versions (hence the try/except ImportError at import time; no other side
# effects). Relationships: uses the size, timezone list and local timezone
# object from the sibling tslib benchmark module.
try:
    from pandas._libs.tslibs import (
        is_date_array_normalized,
        normalize_i8_timestamps,
    )
except ImportError:
    from pandas._libs.tslibs.conversion import (
        normalize_i8_timestamps,
        is_date_array_normalized,
    )

import pandas as pd

from .tslib import (
    _sizes,
    _tzs,
    tzlocal_obj,
)


class Normalize:
    # [CLASS DOC]
    # Benchmark class parameterized over array size and timezone. State set by
    # setup: an int64 array of nanosecond timestamps made of ten consecutive
    # midnights repeated to fill the requested size.
    params = [
        _sizes,
        _tzs,
    ]
    param_names = ["size", "tz"]

    def setup(self, size, tz):
        # [FUNCTION DOC]
        # Builds ten daily midnight timestamps from 2016-01-01 in the given
        # timezone, repeats each size divided by ten times, and stores the
        # underlying int64 values. Using normalized data prevents the
        # normalization check from short-circuiting early.
        # Raises a not-implemented error (which asv treats as a skip) when size
        # is one million and tz is the local timezone object, because the local
        # timezone is very slow.
        # use an array that will have is_date_array_normalized give True,
        #  so we do not short-circuit early.
        dti = pd.date_range("2016-01-01", periods=10, tz=tz).repeat(size // 10)
        self.i8data = dti.asi8

        if size == 10**6 and tz is tzlocal_obj:
            # tzlocal is cumbersomely slow, so skip to keep runtime in check
            raise NotImplementedError

    def time_normalize_i8_timestamps(self, size, tz):
        # [FUNCTION DOC]
        # Times normalizing the int64 timestamps to midnight in the timezone
        # using the nanosecond unit code (ten). Size is unused here.
        # 10 i.e. NPY_FR_ns
        normalize_i8_timestamps(self.i8data, tz, 10)

    def time_is_date_array_normalized(self, size, tz):
        # [FUNCTION DOC]
        # Times checking whether the int64 timestamps are all at midnight in the
        # timezone, with the nanosecond unit code (ten). Size is unused here.
        # TODO: cases with different levels of short-circuiting
        # 10 i.e. NPY_FR_ns
        is_date_array_normalized(self.i8data, tz, 10)
