# [MODULE DOC]
# Tests for pickling round trips of DatetimeIndex, including missing values,
# timezones and frequency retention.
# Uses the temp_file fixture for the round-trip helper.

import pytest

from pandas import (
    NaT,
    date_range,
    to_datetime,
)
import pandas._testing as tm


class TestPickle:
    # [CLASS DOC]
    # Groups pickle round-trip tests for DatetimeIndex.
    def test_pickle(self, temp_file):
        # [FUNCTION DOC]
        # Verifies (GH#4606) round-tripping an index with a missing value preserves
        # values and keeps the missing element as the missing-time singleton.
        # GH#4606
        idx = to_datetime(["2013-01-01", NaT, "2014-01-06"])
        idx_p = tm.round_trip_pickle(idx, temp_file)
        assert idx_p[0] == idx[0]
        assert idx_p[1] is NaT
        assert idx_p[2] == idx[2]

    def test_pickle_dont_infer_freq(self, temp_file):
        # [FUNCTION DOC]
        # Verifies (GH#11002) pickling a long weekly range round-trips without frequency
        # inference changing the result.
        # GH#11002
        # don't infer freq
        idx = date_range("1750-1-1", "2050-1-1", freq="7D")
        idx_p = tm.round_trip_pickle(idx, temp_file)
        tm.assert_index_equal(idx, idx_p)

    def test_pickle_after_set_freq(self, temp_file):
        # [FUNCTION DOC]
        # Verifies a timezone-aware, named index with frequency cleared round-trips to
        # an equal index.
        dti = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        dti = dti._with_freq(None)

        res = tm.round_trip_pickle(dti, temp_file)
        tm.assert_index_equal(res, dti)

    def test_roundtrip_pickle_with_tz(self, temp_file):
        # [FUNCTION DOC]
        # Verifies (GH#8367) a timezone-aware named index round-trips equal.
        # GH#8367
        # round-trip of timezone
        index = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        unpickled = tm.round_trip_pickle(index, temp_file)
        tm.assert_index_equal(index, unpickled)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_pickle_unpickle(self, freq, temp_file):
        # [FUNCTION DOC]
        # Parametrized over business frequencies. Verifies the unpickled range keeps the
        # original frequency.
        rng = date_range("2009-01-01", "2010-01-01", freq=freq)
        unpickled = tm.round_trip_pickle(rng, temp_file)
        assert unpickled.freq == freq
