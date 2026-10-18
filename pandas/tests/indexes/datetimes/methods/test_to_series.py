# [MODULE DOC]
# Test of DatetimeIndex.to_series.
import numpy as np

from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm


class TestToSeries:
    # [CLASS DOC]
    # Groups to_series tests; no state.
    def test_to_series(self):
        # [FUNCTION DOC]
        # Verifies to_series with an explicit integer index on a US/Pacific index gives a Series named like
        # the index whose values are the timezone-aware stamps (dtype matches the index dtype).
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")

        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        result = idx.to_series(index=range(2))
        assert expected.dtype == idx.dtype
        tm.assert_series_equal(result, expected)
