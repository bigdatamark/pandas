# [MODULE DOC]
# Tests for the compiled check of whether a datetime64 integer array is normalized (all values at midnight). Defines a module constant
# holding a normalized day-unit datetime64 array of ten values.

import numpy as np

from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas._libs.tslibs.vectorized import is_date_array_normalized

# a datetime64 ndarray which *is* normalized
day_arr = np.arange(10, dtype="i8").view("M8[D]")


class TestIsDateArrayNormalized:
    # [CLASS DOC]
    # Tests for the normalization check on integer views of datetime64 arrays.
    def test_is_date_array_normalized_day(self):
        # [FUNCTION DOC]
        # A day-unit array viewed as integers, with no timezone and the day unit code, is reported normalized (exactly True).
        arr = day_arr
        abbrev = "D"
        unit = abbrev_to_npy_unit(abbrev)
        result = is_date_array_normalized(arr.view("i8"), None, unit)
        assert result is True

    def test_is_date_array_normalized_seconds(self):
        # [FUNCTION DOC]
        # The day array cast to seconds is normalized (True). After adding one second to its first element it is not normalized (exactly False). Note this mutates the freshly cast copy only.
        abbrev = "s"
        arr = day_arr.astype(f"M8[{abbrev}]")
        unit = abbrev_to_npy_unit(abbrev)
        result = is_date_array_normalized(arr.view("i8"), None, unit)
        assert result is True

        arr[0] += np.timedelta64(1, abbrev)
        result2 = is_date_array_normalized(arr.view("i8"), None, unit)
        assert result2 is False
