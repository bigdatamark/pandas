# [MODULE DOC]
# Tests for cumulative operations (the private accumulate method) on
# TimedeltaArray. No import-time side effects.

import pytest

import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestAccumulator:
    # [CLASS DOC]
    # Groups tests of accumulation on TimedeltaArray.
    def test_accumulators_disallowed(self):
        # [FUNCTION DOC]
        # Regression test (GH#50297). Verifies cumulative product raises a type error
        # ("cumprod not supported").
        # GH#50297
        arr = TimedeltaArray._from_sequence(["1D", "2D"], dtype="m8[ns]")
        with pytest.raises(TypeError, match="cumprod not supported"):
            arr._accumulate("cumprod")

    def test_cumsum(self, unit):
        # [FUNCTION DOC]
        # Regression test (GH#50297). Uses the unit fixture. Verifies cumulative sum of
        # one-day and two-day values gives one and three days in the same unit.
        # GH#50297
        dtype = f"m8[{unit}]"
        arr = TimedeltaArray._from_sequence(["1D", "2D"], dtype=dtype)
        result = arr._accumulate("cumsum")
        expected = TimedeltaArray._from_sequence(["1D", "3D"], dtype=dtype)
        tm.assert_timedelta_array_equal(result, expected)
