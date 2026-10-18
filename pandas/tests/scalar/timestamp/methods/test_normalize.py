# [MODULE DOC]
# Tests for the Timestamp method that truncates the time of day to midnight. Covers
# parametrized units and timezones, pre-epoch dates and the overflow error.
import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit


class TestTimestampNormalize:
    # [CLASS DOC]
    # Groups the tests of the normalize method on Timestamp. Stateless.
    @pytest.mark.parametrize("arg", ["2013-11-30", "2013-11-30 12:00:00"])
    def test_normalize(self, tz_naive_fixture, arg, unit):
        # [FUNCTION DOC]
        # Verifies that normalizing a timestamp (given with or without a time of day, in
        # each tz-naive fixture timezone and each unit fixture) yields midnight of the
        # same date in the same timezone, and that the resolution of the result matches
        # the requested unit.
        tz = tz_naive_fixture
        ts = Timestamp(arg, tz=tz).as_unit(unit)
        result = ts.normalize()
        expected = Timestamp("2013-11-30", tz=tz)
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    def test_normalize_pre_epoch_dates(self):
        # [FUNCTION DOC]
        # Regression test for dates before the epoch (GH 36294): normalizing a pre-1970
        # timestamp with a morning time must give midnight of that same day (not the
        # previous day).
        # GH: 36294
        result = Timestamp("1969-01-01 09:00:00").normalize()
        expected = Timestamp("1969-01-01 00:00:00")
        assert result == expected

    def test_normalize_overflow_raises(self):
        # [FUNCTION DOC]
        # Regression test (GH 60583): normalizing the minimum representable timestamp
        # would need to move before the lower bound, so a value error stating that the
        # timestamp cannot be normalized without integer overflow is expected.
        # GH#60583
        ts = Timestamp.min

        msg = "Cannot normalize Timestamp without integer overflow"
        with pytest.raises(ValueError, match=msg):
            ts.normalize()
