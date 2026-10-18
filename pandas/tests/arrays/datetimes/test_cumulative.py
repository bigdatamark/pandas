# [MODULE DOC]
# Tests for cumulative operations (private accumulate method) on DatetimeArray. No import-time side
# effects.

import pytest

import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestAccumulator:
    # [CLASS DOC]
    # Groups accumulation tests for DatetimeArray. Stateless.
    def test_accumulators_freq(self):
        # [FUNCTION DOC]
        # Cumulative minimum and maximum of a daily array with an inferred frequency give the running min (all first date) and the unchanged increasing sequence.
        # GH#50297
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="M8[ns]",
        )._with_freq("infer")
        result = arr._accumulate("cummin")
        expected = DatetimeArray._from_sequence(["2000-01-01"] * 3, dtype="M8[ns]")
        tm.assert_datetime_array_equal(result, expected)

        result = arr._accumulate("cummax")
        expected = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="M8[ns]",
        )
        tm.assert_datetime_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["cumsum", "cumprod"])
    def test_accumulators_disallowed(self, func):
        # [FUNCTION DOC]
        # Parametrized over cumsum and cumprod. Both raise a type error that the accumulation is not supported for datetimes.
        # GH#50297
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
            ],
            dtype="M8[ns]",
        )._with_freq("infer")
        with pytest.raises(TypeError, match=f"Accumulation {func}"):
            arr._accumulate(func)
