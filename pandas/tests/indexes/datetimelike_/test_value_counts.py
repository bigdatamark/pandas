# [MODULE DOC]
# Tests for value_counts and unique on DatetimeIndex, TimedeltaIndex and PeriodIndex (and their
# Series), with repeated values and with missing values and the dropna option.
import numpy as np

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestValueCounts:
    # [CLASS DOC]
    # Groups value_counts/unique tests for datetime-like indexes. Contains helper check methods
    # shared by the concrete test methods.
    # GH#7735

    def test_value_counts_unique_datetimeindex(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture. Runs the repeated-values check on an hourly datetime range of ten
        # entries in the given timezone.
        tz = tz_naive_fixture
        orig = date_range("2011-01-01 09:00", freq="h", periods=10, tz=tz)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_timedeltaindex(self):
        # [FUNCTION DOC]
        # Runs the repeated-values check on an hourly timedelta range of ten entries.
        orig = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_periodindex(self):
        # [FUNCTION DOC]
        # Runs the repeated-values check on an hourly period range of ten entries.
        orig = period_range("2011-01-01 09:00", freq="h", periods=10)
        self._check_value_counts_with_repeats(orig)

    def _check_value_counts_with_repeats(self, orig):
        # [FUNCTION DOC]
        # HELPER: takes an index of ten distinct sorted values (orig). Builds an index where the n-th
        # value is repeated n+1 times, then verifies value_counts for both the index and a Series of it
        # gives counts from ten down to one, with the original values in reverse order as the index
        # (freq dropped except for periods; counts named count with int64 dtype), and that unique
        # returns the original index.
        # create repeated values, 'n'th element is repeated by n+1 times
        idx = type(orig)(
            np.repeat(orig._values, range(1, len(orig) + 1)), dtype=orig.dtype
        )

        exp_idx = orig[::-1]
        if not isinstance(exp_idx, PeriodIndex):
            exp_idx = exp_idx._with_freq(None)
        expected = Series(range(10, 0, -1), index=exp_idx, dtype="int64", name="count")

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)

        tm.assert_index_equal(idx.unique(), orig)

    def test_value_counts_unique_datetimeindex2(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture. Builds a datetime index with three, two and one-NaT entries and runs
        # the NaT/dropna check.
        tz = tz_naive_fixture
        idx = DatetimeIndex(
            [
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 08:00",
                "2013-01-01 08:00",
                NaT,
            ],
            tz=tz,
        )
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_timedeltaindex2(self):
        # [FUNCTION DOC]
        # Runs the NaT/dropna check on a TimedeltaIndex with the same repetition pattern.
        idx = TimedeltaIndex(
            [
                "1 days 09:00:00",
                "1 days 09:00:00",
                "1 days 09:00:00",
                "1 days 08:00:00",
                "1 days 08:00:00",
                NaT,
            ]
        )
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_periodindex2(self):
        # [FUNCTION DOC]
        # Runs the NaT/dropna check on an hourly PeriodIndex with the same repetition pattern.
        idx = PeriodIndex(
            [
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 08:00",
                "2013-01-01 08:00",
                NaT,
            ],
            freq="h",
        )
        self._check_value_counts_dropna(idx)

    def _check_value_counts_dropna(self, idx):
        # [FUNCTION DOC]
        # HELPER: takes an index of three repeats, two repeats and one NaT. Verifies value_counts of the
        # index and Series by default excludes NaT (counts 3 and 2) and with dropna False includes it
        # (counts 3, 2, 1), and that unique returns the distinct values including NaT in order of first
        # appearance (values at positions 2, 3 and the last).
        exp_idx = idx[[2, 3]]
        expected = Series([3, 2], index=exp_idx, name="count")

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)

        exp_idx = idx[[2, 3, -1]]
        expected = Series([3, 2, 1], index=exp_idx, name="count")

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(dropna=False), expected)

        tm.assert_index_equal(idx.unique(), exp_idx)
