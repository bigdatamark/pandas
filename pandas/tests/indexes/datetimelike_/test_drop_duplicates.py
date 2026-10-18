# [MODULE DOC]
# Tests for duplicated/drop_duplicates on the datetime-like index types (PeriodIndex,
# DatetimeIndex, TimedeltaIndex). A shared mixin defines the tests; one concrete class per
# index type supplies the idx fixture.
import numpy as np
import pytest

from pandas import (
    PeriodIndex,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class DropDuplicates:
    # [CLASS DOC]
    # Mixin (not collected itself) holding the duplicate-dropping tests shared by the concrete
    # classes below. Requires an idx fixture from the subclass returning an index of ten
    # entries named idx.
    def test_drop_duplicates_metadata(self, idx):
        # [FUNCTION DOC]
        # Verifies drop_duplicates on a unique index returns an equal index with the same freq; for an
        # index appended to itself the result keeps the original values with freq reset to None for
        # datetime and timedelta indexes, but retains the freq for PeriodIndex.
        # GH#10115
        result = idx.drop_duplicates()
        tm.assert_index_equal(idx, result)
        assert idx.freq == result.freq

        idx_dup = idx.append(idx)
        result = idx_dup.drop_duplicates()

        expected = idx
        if not isinstance(idx, PeriodIndex):
            # freq is reset except for PeriodIndex
            assert idx_dup.freq is None
            assert result.freq is None
            expected = idx._with_freq(None)
        else:
            assert result.freq == expected.freq

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "keep, expected, index",
        [
            (
                "first",
                np.concatenate(([False] * 10, [True] * 5)),
                np.arange(0, 10, dtype=np.int64),
            ),
            (
                "last",
                np.concatenate(([True] * 5, [False] * 10)),
                np.arange(5, 15, dtype=np.int64),
            ),
            (
                False,
                np.concatenate(([True] * 5, [False] * 5, [True] * 5)),
                np.arange(5, 10, dtype=np.int64),
            ),
        ],
    )
    def test_drop_duplicates(self, keep, expected, index, idx):
        # [FUNCTION DOC]
        # Parametrized over keep option (first, last, False), expected duplicated mask and the expected
        # Series index labels. Appends the first five elements again, verifies duplicated gives the mask,
        # drop_duplicates on the index equals the masked-out selection, and the same on a Series
        # gives the matching values with the expected labels.
        # to check Index/Series compat
        idx = idx.append(idx[:5])

        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected)
        expected = idx[~expected]

        result = idx.drop_duplicates(keep=keep)
        tm.assert_index_equal(result, expected)

        result = Series(idx).drop_duplicates(keep=keep)
        expected = Series(expected, index=index)
        tm.assert_series_equal(result, expected)


class TestDropDuplicatesPeriodIndex(DropDuplicates):
    # [CLASS DOC]
    # Concrete test class for PeriodIndex; adds a freq fixture and the idx fixture.
    @pytest.fixture(params=["D", "3D", "h", "2h", "min", "2min", "s", "3s"])
    def freq(self, request):
        # [FUNCTION DOC]
        # HELPER: fixture parametrized over eight period frequency strings (days, hours, minutes,
        # seconds, with multiples); returns the current frequency.
        """
        Fixture to test for different frequencies for PeriodIndex.
        """
        return request.param

    @pytest.fixture
    def idx(self, freq):
        # [FUNCTION DOC]
        # HELPER: fixture returning a ten-period PeriodIndex named idx starting on 2011-01-01 at the
        # frequency from the freq fixture.
        """
        Fixture to get PeriodIndex for 10 periods for different frequencies.
        """
        return period_range("2011-01-01", periods=10, freq=freq, name="idx")


class TestDropDuplicatesDatetimeIndex(DropDuplicates):
    # [CLASS DOC]
    # Concrete test class for DatetimeIndex; provides the idx fixture.
    @pytest.fixture
    def idx(self, freq_sample):
        # [FUNCTION DOC]
        # HELPER: fixture returning a ten-element daterange named idx at the frequency given by the
        # shared freq_sample fixture.
        """
        Fixture to get DatetimeIndex for 10 periods for different frequencies.
        """
        return date_range("2011-01-01", freq=freq_sample, periods=10, name="idx")


class TestDropDuplicatesTimedeltaIndex(DropDuplicates):
    # [CLASS DOC]
    # Concrete test class for TimedeltaIndex; provides the idx fixture.
    @pytest.fixture
    def idx(self, freq_sample):
        # [FUNCTION DOC]
        # HELPER: fixture returning a ten-element timedelta range named idx starting at one day with
        # the frequency from the freq_sample fixture.
        """
        Fixture to get TimedeltaIndex for 10 periods for different frequencies.
        """
        return timedelta_range("1 day", periods=10, freq=freq_sample, name="idx")
