# [MODULE DOC]
# Tests for PeriodIndex.fillna.
from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
)
import pandas._testing as tm


class TestFillNA:
    # [CLASS DOC]
    # Groups fillna tests.
    def test_fillna_period(self):
        # [FUNCTION DOC]
        # An hourly index with a missing value: filling with an hourly Period keeps
        # period dtype; filling with a string or a period of a different frequency
        # produces an object index (GH#11343).
        # GH#11343
        idx = PeriodIndex(["2011-01-01 09:00", NaT, "2011-01-01 11:00"], freq="h")

        exp = PeriodIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], freq="h"
        )
        result = idx.fillna(Period("2011-01-01 10:00", freq="h"))
        tm.assert_index_equal(result, exp)

        exp = Index(
            [
                Period("2011-01-01 09:00", freq="h"),
                "x",
                Period("2011-01-01 11:00", freq="h"),
            ],
            dtype=object,
        )
        result = idx.fillna("x")
        tm.assert_index_equal(result, exp)

        exp = Index(
            [
                Period("2011-01-01 09:00", freq="h"),
                Period("2011-01-01", freq="D"),
                Period("2011-01-01 11:00", freq="h"),
            ],
            dtype=object,
        )
        result = idx.fillna(Period("2011-01-01", freq="D"))
        tm.assert_index_equal(result, exp)
