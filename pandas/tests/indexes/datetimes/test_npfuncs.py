# [MODULE DOC]
# Tests for numpy function interoperability with DatetimeIndex.
# Pure tests module; no import-time side effects.

import numpy as np

from pandas import date_range
import pandas._testing as tm


class TestSplit:
    # [CLASS DOC]
    # Groups tests of the numpy split function applied to a DatetimeIndex.
    def test_split_non_utc(self):
        # [FUNCTION DOC]
        # Verifies (GH#14042) that splitting a timezone-aware index (fixed offset)
        # without split points returns a single piece equal to the original with its
        # frequency cleared.
        # GH#14042
        indices = date_range("2016-01-01 00:00:00+0200", freq="s", periods=10)
        result = np.split(indices, indices_or_sections=[])[0]
        expected = indices._with_freq(None)
        tm.assert_index_equal(result, expected)
