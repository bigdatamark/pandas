# [MODULE DOC]
# Tests for PeriodIndex.repeat.
import numpy as np
import pytest

from pandas import (
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestRepeat:
    # [CLASS DOC]
    # Groups repeat tests.
    @pytest.mark.parametrize("use_numpy", [True, False])
    @pytest.mark.parametrize(
        "index",
        [
            period_range("2000-01-01", periods=3, freq="D"),
            period_range("2001-01-01", periods=3, freq="2D"),
            PeriodIndex(["2001-01", "NaT", "2003-01"], freq="M"),
        ],
    )
    def test_repeat_freqstr(self, index, use_numpy):
        # [FUNCTION DOC]
        # Parametrized over using the numpy function or the method and over three
        # indexes (daily, 2-day multiple, monthly with NaT). Each element is repeated
        # three times consecutively and the frequency string is preserved (GH#10183).
        # GH#10183
        expected = PeriodIndex([per for per in index for _ in range(3)])
        result = np.repeat(index, 3) if use_numpy else index.repeat(3)
        tm.assert_index_equal(result, expected)
        assert result.freqstr == index.freqstr
