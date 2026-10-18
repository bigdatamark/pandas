# [MODULE DOC]
# Tests for the shift method of the timedelta index, which is also what addition and subtraction of offsets rely on.

import pytest

from pandas.errors import NullFrequencyError

import pandas as pd
from pandas import TimedeltaIndex
import pandas._testing as tm


class TestTimedeltaIndexShift:
    # [CLASS DOC]
    # Test class for shift.
    # -------------------------------------------------------------
    # TimedeltaIndex.shift is used by __add__/__sub__

    def test_tdi_shift_empty(self):
        # [FUNCTION DOC]
        # Shifting an empty named index by zero or three hours returns an equal empty index.
        # GH#9903
        idx = TimedeltaIndex([], name="xxx")
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        tm.assert_index_equal(idx.shift(3, freq="h"), idx)

    def test_tdi_shift_hours(self):
        # [FUNCTION DOC]
        # Shifting by zero returns the same values; by three hours forward and back add or subtract three hours from each; name preserved.
        # GH#9903
        idx = TimedeltaIndex(["5 hours", "6 hours", "9 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        exp = TimedeltaIndex(["8 hours", "9 hours", "12 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(3, freq="h"), exp)
        exp = TimedeltaIndex(["2 hours", "3 hours", "6 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(-3, freq="h"), exp)

    def test_tdi_shift_minutes(self):
        # [FUNCTION DOC]
        # Same with a minute frequency: shifting by three minutes moves each value by three minutes in either direction.
        # GH#9903
        idx = TimedeltaIndex(["5 hours", "6 hours", "9 hours"], name="xxx")
        tm.assert_index_equal(idx.shift(0, freq="min"), idx)
        exp = TimedeltaIndex(["05:03:00", "06:03:00", "9:03:00"], name="xxx")
        tm.assert_index_equal(idx.shift(3, freq="min"), exp)
        exp = TimedeltaIndex(["04:57:00", "05:57:00", "8:57:00"], name="xxx")
        tm.assert_index_equal(idx.shift(-3, freq="min"), exp)

    def test_tdi_shift_int(self):
        # [FUNCTION DOC]
        # A daily range plus an hour has a frequency; shifting by one period with no explicit frequency adds one day and gives second-resolution values with a daily frequency.
        # GH#8083
        tdi = pd.to_timedelta(range(5), unit="D")
        trange = tdi._with_freq("infer") + pd.offsets.Hour(1)
        result = trange.shift(1)
        expected = TimedeltaIndex(
            [
                "1 days 01:00:00",
                "2 days 01:00:00",
                "3 days 01:00:00",
                "4 days 01:00:00",
                "5 days 01:00:00",
            ],
            freq="D",
            dtype="m8[s]",
        )
        tm.assert_index_equal(result, expected)

    def test_tdi_shift_nonstandard_freq(self):
        # [FUNCTION DOC]
        # Shifting by three periods with a compound frequency of two days and one second adds three times that amount to each value, giving second-resolution values.
        # GH#8083
        tdi = pd.to_timedelta(range(5), unit="D")
        trange = tdi._with_freq("infer") + pd.offsets.Hour(1)
        result = trange.shift(3, freq="2D 1s")
        expected = TimedeltaIndex(
            [
                "6 days 01:00:03",
                "7 days 01:00:03",
                "8 days 01:00:03",
                "9 days 01:00:03",
                "10 days 01:00:03",
            ],
            freq="D",
            dtype="m8[s]",
        )
        tm.assert_index_equal(result, expected)

    def test_shift_no_freq(self):
        # [FUNCTION DOC]
        # Shifting an index with no frequency raises the null-frequency error with message "Cannot shift with no freq".
        # GH#19147
        tdi = TimedeltaIndex(["1 days 01:00:00", "2 days 01:00:00"], freq=None)
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):
            tdi.shift(2)
