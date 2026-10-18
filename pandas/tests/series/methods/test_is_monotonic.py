# [MODULE DOC]
# Tests for the is_monotonic_increasing and is_monotonic_decreasing properties of
# Series. No import-time side effects.

import numpy as np

from pandas import (
    Series,
    date_range,
)


class TestIsMonotonic:
    # [CLASS DOC]
    # Groups the tests of the Series monotonic properties. Stateless.
    def test_is_monotonic_numeric(self):
        # [FUNCTION DOC]
        # Random integers are not monotonic increasing; an ascending range is increasing
        # and a descending range is decreasing.
        # KNOWN ISSUE: the assertion that the ascending Series is increasing is
        # repeated twice.
        ser = Series(np.random.default_rng(2).integers(0, 10, size=1000))
        assert not ser.is_monotonic_increasing
        ser = Series(np.arange(1000))
        assert ser.is_monotonic_increasing is True
        assert ser.is_monotonic_increasing is True
        ser = Series(np.arange(1000, 0, -1))
        assert ser.is_monotonic_decreasing is True

    def test_is_monotonic_dt64(self):
        # [FUNCTION DOC]
        # A daily date Series is increasing; reversed it is not increasing but is
        # decreasing.
        # KNOWN ISSUE: the assertion that the ascending Series is increasing is
        # repeated twice.
        ser = Series(date_range("20130101", periods=10))
        assert ser.is_monotonic_increasing is True
        assert ser.is_monotonic_increasing is True

        ser = Series(list(reversed(ser)))
        assert ser.is_monotonic_increasing is False
        assert ser.is_monotonic_decreasing is True
