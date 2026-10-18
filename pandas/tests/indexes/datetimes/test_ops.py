# [MODULE DOC]
# Tests for miscellaneous DatetimeIndex operations: frequency inference from
# raw integers and behavior of business-day indexes.
# Uses the freq_sample fixture from the shared conftest.

from datetime import datetime

import pytest

from pandas import (
    DatetimeIndex,
    Index,
    bdate_range,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndexOps:
    # [CLASS DOC]
    # Groups general operation tests for DatetimeIndex.
    def test_infer_freq(self, freq_sample):
        # [FUNCTION DOC]
        # Parametrized over freq_sample. Builds a nanosecond-unit range of ten periods,
        # reconstructs an index from its raw integer values asking for frequency
        # inference, and verifies equality with the original and that the inferred
        # frequency matches the sample frequency (GH 11018).
        # GH 11018
        idx = date_range("2011-01-01 09:00:00", freq=freq_sample, periods=10, unit="ns")
        result = DatetimeIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample


@pytest.mark.parametrize("freq", ["B", "C"])
class TestBusinessDatetimeIndex:
    # [CLASS DOC]
    # Groups tests run for both business-day frequencies (class parametrized over
    # the two frequencies, applying to each method) on a business date range.
    @pytest.fixture
    def rng(self, freq):
        # [FUNCTION DOC]
        # HELPER: fixture for the business-day tests. Returns a business date range
        # covering the year 2009 for the class-level frequency parameter.
        START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)
        return bdate_range(START, END, freq=freq)

    def test_comparison(self, rng):
        # [FUNCTION DOC]
        # Verifies the elementwise greater-than comparison against an element is true
        # for a later position and false for an earlier one.
        d = rng[10]

        comp = rng > d
        assert comp[11]
        assert not comp[9]

    def test_copy(self, rng):
        # [FUNCTION DOC]
        # Verifies a copy of the range is equal to the original.
        cp = rng.copy()
        tm.assert_index_equal(cp, rng)

    def test_identical(self, rng):
        # [FUNCTION DOC]
        # Verifies the identical check: copies are identical; after renaming one, the
        # indexes are still equal but not identical; renaming the other to match makes
        # them identical; an index rebuilt from raw values (losing frequency) is equal
        # but not identical.
        t1 = rng.copy()
        t2 = rng.copy()
        assert t1.identical(t2)

        # name
        t1 = t1.rename("foo")
        assert t1.equals(t2)
        assert not t1.identical(t2)
        t2 = t2.rename("foo")
        assert t1.identical(t2)

        # freq
        t2v = Index(t2.values)
        assert t1.equals(t2v)
        assert not t1.identical(t2v)
