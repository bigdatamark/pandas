# [MODULE DOC]
# Tests for min and max reductions on PeriodArray. Uses the skipna fixture from the shared conftest. No import-time side effects.

import pandas as pd
from pandas.core.arrays import period_array


class TestReductions:
    # [CLASS DOC]
    # Groups reduction tests for PeriodArray. Stateless.
    def test_min_max(self):
        # [FUNCTION DOC]
        # Min and max of an array with a NaT give the earliest and latest periods; with skipna disabled they give NaT.
        arr = period_array(
            [
                "2000-01-03",
                "2000-01-03",
                "NaT",
                "2000-01-02",
                "2000-01-05",
                "2000-01-04",
            ],
            freq="D",
        )

        result = arr.min()
        expected = pd.Period("2000-01-02", freq="D")
        assert result == expected

        result = arr.max()
        expected = pd.Period("2000-01-05", freq="D")
        assert result == expected

        result = arr.min(skipna=False)
        assert result is pd.NaT

        result = arr.max(skipna=False)
        assert result is pd.NaT

    def test_min_max_empty(self, skipna):
        # [FUNCTION DOC]
        # Using the skipna fixture. Min and max of an empty array are NaT.
        arr = period_array([], freq="D")
        result = arr.min(skipna=skipna)
        assert result is pd.NaT

        result = arr.max(skipna=skipna)
        assert result is pd.NaT
