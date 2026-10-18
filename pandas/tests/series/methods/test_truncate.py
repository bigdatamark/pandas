# [MODULE DOC]
# Tests for the Series method that slices by index labels before and after given bounds, using datetime, period and repeated-label indexes.
# No import-time side effects.

from datetime import datetime

import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm


class TestTruncate:
    # [CLASS DOC]
    # Groups tests of truncating a Series by index bounds. Stateless.
    def test_truncate_datetimeindex_tz(self):
        # [FUNCTION DOC]
        # Verifies truncating a Series with a timezone-aware daily index (issue 9243): naive datetime bounds raise a type error about comparing tz-naive and tz-aware values (strict timezone-awareness compatibility, issue 36148), while bounds converted from the index's own aware timestamps return the matching three-element slice.
        # GH 9243
        idx = date_range("4/1/2005", "4/30/2005", freq="D", tz="US/Pacific")
        s = Series(range(len(idx)), index=idx)
        with pytest.raises(TypeError, match="Cannot compare tz-naive"):
            # GH#36148 as of 2.0 we require tzawareness compat
            s.truncate(datetime(2005, 4, 2), datetime(2005, 4, 4))

        lb = idx[1]
        ub = idx[3]
        result = s.truncate(lb.to_pydatetime(), ub.to_pydatetime())
        expected = Series([1, 2, 3], index=idx[1:4])
        tm.assert_series_equal(result, expected)

    def test_truncate_periodindex(self):
        # [FUNCTION DOC]
        # Verifies truncation on a period index with duplicate labels (issue 17717): truncating after a period on an unsorted-with-duplicates index keeps both duplicate entries, and for a sorted version of a different index keeps only the single matching entry. Results are compared to hand-built series.
        # GH 17717
        idx1 = pd.PeriodIndex(
            [pd.Period("2017-09-02"), pd.Period("2017-09-02"), pd.Period("2017-09-03")]
        )
        series1 = Series([1, 2, 3], index=idx1)
        result1 = series1.truncate(after="2017-09-02")

        expected_idx1 = pd.PeriodIndex(
            [pd.Period("2017-09-02"), pd.Period("2017-09-02")]
        )
        tm.assert_series_equal(result1, Series([1, 2], index=expected_idx1))

        idx2 = pd.PeriodIndex(
            [pd.Period("2017-09-03"), pd.Period("2017-09-02"), pd.Period("2017-09-03")]
        )
        series2 = Series([1, 2, 3], index=idx2)
        result2 = series2.sort_index().truncate(after="2017-09-02")

        expected_idx2 = pd.PeriodIndex([pd.Period("2017-09-02")])
        tm.assert_series_equal(result2, Series([2], index=expected_idx2))

    def test_truncate_one_element_series(self):
        # [FUNCTION DOC]
        # Verifies that truncating a single-element datetime-indexed Series with bounds enclosing its only label returns an equal Series (issue 35544).
        # GH 35544
        series = Series([0.1], index=pd.DatetimeIndex(["2020-08-04"]))
        before = pd.Timestamp("2020-08-02")
        after = pd.Timestamp("2020-08-04")

        result = series.truncate(before=before, after=after)

        # the input Series and the expected Series are the same
        tm.assert_series_equal(result, series)

    def test_truncate_index_only_one_unique_value(self):
        # [FUNCTION DOC]
        # Verifies truncating a Series whose index holds only one unique date repeated five times, with bounds enclosing it, returns the entire Series unchanged (issue 42365).
        # GH 42365
        obj = Series(0, index=date_range("2021-06-30", "2021-06-30")).repeat(5)

        truncated = obj.truncate("2021-06-28", "2021-07-01")

        tm.assert_series_equal(truncated, obj)
