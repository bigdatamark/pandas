"""Tests for PeriodIndex behaving like a vectorized Period scalar"""
# [MODULE DOC]
# Tests that PeriodIndex exposes scalar Period attributes in vectorized form:
# start and end timestamps.

import pytest

from pandas import (
    Timedelta,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndexOps:
    # [CLASS DOC]
    # Groups the tests of vectorized scalar attributes.
    def test_start_time(self):
        # [FUNCTION DOC]
        # start_time of a monthly range equals the month-start date range (GH#17157).
        # GH#17157
        index = period_range(freq="M", start="2016-01-01", end="2016-05-31")
        expected_index = date_range("2016-01-01", end="2016-05-31", freq="MS")
        tm.assert_index_equal(index.start_time, expected_index)

    def test_end_time(self):
        # [FUNCTION DOC]
        # end_time of a monthly range equals month-end dates plus one day minus one
        # microsecond (GH#17157).
        # GH#17157
        index = period_range(freq="M", start="2016-01-01", end="2016-05-31")
        expected_index = date_range("2016-01-01", end="2016-05-31", freq="ME")
        expected_index += Timedelta(1, "D") - Timedelta(1, "us")
        tm.assert_index_equal(index.end_time, expected_index)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_end_time_business_friday(self):
        # [FUNCTION DOC]
        # For a one-element business-day range on a Friday, end_time is the end of that
        # day, not Monday (GH#34449). Business-day deprecation warnings are ignored.
        # GH#34449
        pi = period_range("1990-01-05", freq="B", periods=1)
        result = pi.end_time

        dti = date_range("1990-01-05", freq="D", periods=1)._with_freq(None)
        expected = dti + Timedelta(1, "D") - Timedelta(1, "us")
        tm.assert_index_equal(result, expected)
