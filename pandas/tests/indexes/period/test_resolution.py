# [MODULE DOC]
# Tests for the resolution property of PeriodIndex.
import pytest

import pandas as pd


class TestResolution:
    # [CLASS DOC]
    # Groups resolution tests.
    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("Y", "year"),
            ("Q", "quarter"),
            ("M", "month"),
            ("D", "day"),
            ("h", "hour"),
            ("min", "minute"),
            ("s", "second"),
            ("ms", "millisecond"),
            ("us", "microsecond"),
        ],
    )
    def test_resolution(self, freq, expected):
        # [FUNCTION DOC]
        # Parametrized over frequency and expected resolution name (year through
        # microsecond). A range with that frequency reports the expected resolution.
        idx = pd.period_range(start="2013-04-01", periods=30, freq=freq)
        assert idx.resolution == expected
