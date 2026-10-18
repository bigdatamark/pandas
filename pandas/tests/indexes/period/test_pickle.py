# [MODULE DOC]
# Pickle round-trip tests for PeriodIndex; they use the temp-file fixture.
import numpy as np
import pytest

from pandas import (
    NaT,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestPickle:
    # [CLASS DOC]
    # Groups pickle tests.
    @pytest.mark.parametrize("freq", ["D", "M", "Y"])
    def test_pickle_round_trip(self, freq, temp_file):
        # [FUNCTION DOC]
        # Parametrized over day, month and year frequencies. A pickled and restored
        # index with NaT-like values equals the original.
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq=freq)
        result = tm.round_trip_pickle(idx, temp_file)
        tm.assert_index_equal(result, idx)

    def test_pickle_freq(self, temp_file):
        # [FUNCTION DOC]
        # A restored monthly range keeps its month-end frequency object and string
        # (GH#2891).
        # GH#2891
        prng = period_range("1/1/2011", "1/1/2012", freq="M")
        new_prng = tm.round_trip_pickle(prng, temp_file)
        assert new_prng.freq == offsets.MonthEnd()
        assert new_prng.freqstr == "M"
