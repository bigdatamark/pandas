# [MODULE DOC]
# Tests for frequency inference on the timedelta index.

from pandas import (
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexOps:
    # [CLASS DOC]
    # Test class for index operations.
    def test_infer_freq(self, freq_sample):
        # [FUNCTION DOC]
        # Uses the freq_sample fixture. A range built with that frequency, rebuilt from its integer representation with frequency inference, equals the original and has that frequency.
        # GH#11018
        idx = timedelta_range("1", freq=freq_sample, periods=10)
        result = TimedeltaIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample
