# [MODULE DOC]
# Tests for repeating DatetimeIndex values (method and numpy function), including freq reset,
# units, NaT and unsupported axis argument.
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestRepeat:
    # [CLASS DOC]
    # Groups repeat tests for DatetimeIndex; no state.
    def test_repeat_range(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Verifies repeating a year-long daily range five times has no freq and five times the length.
        # KNOWN ISSUE: the tz_naive_fixture argument is requested but never used, so the test body is
        # repeated for each timezone value without any difference.
        rng = date_range("1/1/2000", "1/1/2001")

        result = rng.repeat(5)
        assert result.freq is None
        assert len(result) == 5 * len(rng)

    def test_repeat_range2(self, tz_naive_fixture, unit):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture and unit. Verifies repeating a two-element daily index twice, via the
        # method and via the numpy function, repeats each element consecutively and drops freq.
        tz = tz_naive_fixture
        index = date_range("2001-01-01", periods=2, freq="D", tz=tz, unit=unit)
        exp = DatetimeIndex(
            ["2001-01-01", "2001-01-01", "2001-01-02", "2001-01-02"], tz=tz
        ).as_unit(unit)
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat_range3(self, tz_naive_fixture, unit):
        # [FUNCTION DOC]
        # Same as the previous test but with a two-day frequency base.
        tz = tz_naive_fixture
        index = date_range("2001-01-01", periods=2, freq="2D", tz=tz, unit=unit)
        exp = DatetimeIndex(
            ["2001-01-01", "2001-01-01", "2001-01-03", "2001-01-03"], tz=tz
        ).as_unit(unit)
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat_range4(self, tz_naive_fixture, unit):
        # [FUNCTION DOC]
        # Verifies repeating three times for an index with NaT repeats NaT as well and drops freq,
        # for the method and numpy function.
        tz = tz_naive_fixture
        index = DatetimeIndex(["2001-01-01", "NaT", "2003-01-01"], tz=tz).as_unit(unit)
        exp = DatetimeIndex(
            [
                "2001-01-01",
                "2001-01-01",
                "2001-01-01",
                "NaT",
                "NaT",
                "NaT",
                "2003-01-01",
                "2003-01-01",
                "2003-01-01",
            ],
            tz=tz,
        ).as_unit(unit)
        for res in [index.repeat(3), np.repeat(index, 3)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat(self, tz_naive_fixture, unit):
        # [FUNCTION DOC]
        # Uses tz_naive_fixture and unit. Verifies repeat on a half-hourly index equals the expected
        # repeated stamps with no freq, matches the numpy function, and that passing the axis parameter
        # to the numpy function raises a value error saying axis is not supported.
        tz = tz_naive_fixture
        reps = 2
        msg = "the 'axis' parameter is not supported"

        rng = date_range(start="2016-01-01", periods=2, freq="30Min", tz=tz, unit=unit)

        expected_rng = DatetimeIndex(
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
            ]
        ).as_unit(unit)

        res = rng.repeat(reps)
        tm.assert_index_equal(res, expected_rng)
        assert res.freq is None

        tm.assert_index_equal(np.repeat(rng, reps), expected_rng)
        with pytest.raises(ValueError, match=msg):
            np.repeat(rng, reps, axis=1)
