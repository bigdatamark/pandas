# [MODULE DOC]
# Tests for Series.count (number of non-missing values). Uses the
# datetime_series fixture. No import-time side effects.

import numpy as np

from pandas import (
    Categorical,
    Series,
)


class TestSeriesCount:
    # [CLASS DOC]
    # Groups the tests of Series.count. Stateless.
    def test_count(self, datetime_series):
        # [FUNCTION DOC]
        # Count of a full Series equals its length; after setting every second value
        # to NaN it equals the number of finite values.
        assert datetime_series.count() == len(datetime_series)

        datetime_series[::2] = np.nan

        assert datetime_series.count() == np.isfinite(datetime_series).sum()

    def test_count_categorical(self):
        # [FUNCTION DOC]
        # An ordered categorical Series with two NaN entries among four values has a
        # count of 2.
        ser = Series(
            Categorical(
                [np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True
            )
        )
        result = ser.count()
        assert result == 2
