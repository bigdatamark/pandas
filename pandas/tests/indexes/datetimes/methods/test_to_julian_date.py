# [MODULE DOC]
# Tests for DatetimeIndex.to_julian_date against the scalar Timestamp conversion at several
# resolutions.
import numpy as np

from pandas import (
    Index,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDateTimeIndexToJulianDate:
    # [CLASS DOC]
    # Groups Julian-date conversion tests; each test builds a five-element range and compares the
    # vectorized result with scalar conversion; no state.
    def test_1700(self):
        # [FUNCTION DOC]
        # Daily range starting in 1710: vectorized result is a float64 Index equal to scalar results.
        dr = date_range(start=Timestamp("1710-10-01"), periods=5, freq="D")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_2000(self):
        # [FUNCTION DOC]
        # Daily range starting in 2000-02-27 (spanning a leap day): float64 Index equal to scalar results.
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="D")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_hour(self):
        # [FUNCTION DOC]
        # Hourly range: float64 Index equal to scalar results.
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="h")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_minute(self):
        # [FUNCTION DOC]
        # Minute-frequency range: float64 Index equal to scalar results.
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="min")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_second(self):
        # [FUNCTION DOC]
        # Second-frequency range: float64 Index equal to scalar results.
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="s")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)
