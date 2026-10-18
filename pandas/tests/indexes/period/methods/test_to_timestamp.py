# [MODULE DOC]
# Tests for PeriodIndex.to_timestamp and round trips through to_period.
from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToTimestamp:
    # [CLASS DOC]
    # Groups to_timestamp tests.
    def test_to_timestamp_non_contiguous(self):
        # [FUNCTION DOC]
        # For strided, reversed and strided-then-reversed views of a daily period index
        # (and its backing array) the converted timestamps equal the corresponding views
        # of the source date index in microsecond resolution; frequency round-trip is
        # not checked for arrays (GH#44100).
        # KNOWN ISSUE: a TODO notes the frequency does not round-trip for the array comparisons, so those comparisons skip the frequency check.
        # GH#44100
        dti = date_range("2021-10-18", periods=9, freq="D", unit="ns")
        pi = dti.to_period()

        result = pi[::2].to_timestamp()
        expected = dti[::2].as_unit("us")
        tm.assert_index_equal(result, expected)

        result = pi._data[::2].to_timestamp()
        expected = dti._data[::2].as_unit("us")
        # TODO: can we get the freq to round-trip?
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

        result = pi[::-1].to_timestamp()
        expected = dti[::-1].as_unit("us")
        tm.assert_index_equal(result, expected)

        result = pi._data[::-1].to_timestamp()
        expected = dti._data[::-1].as_unit("us")
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

        result = pi[::2][::-1].to_timestamp()
        expected = dti[::2][::-1].as_unit("us")
        tm.assert_index_equal(result, expected)

        result = pi._data[::2][::-1].to_timestamp()
        expected = dti._data[::2][::-1].as_unit("us")
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

    def test_to_timestamp_freq(self):
        # [FUNCTION DOC]
        # Yearly periods convert to timestamps at the start of each year with the
        # year-start frequency.
        idx = period_range("2017", periods=12, freq="Y-DEC")
        result = idx.to_timestamp()
        expected = date_range("2017", periods=12, freq="YS-JAN")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_nat(self):
        # [FUNCTION DOC]
        # Missing periods become NaT and the name is kept; round trip back to monthly
        # and triple-monthly periods; a negative frequency raises a value error that
        # frequency must be positive (GH#7228).
        # GH#7228
        index = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="M", name="idx")

        result = index.to_timestamp("D")
        expected = DatetimeIndex(
            [NaT, datetime(2011, 1, 1), datetime(2011, 2, 1)],
            dtype="M8[us]",
            name="idx",
        )
        tm.assert_index_equal(result, expected)
        assert result.name == "idx"

        result2 = result.to_period(freq="M")
        tm.assert_index_equal(result2, index)
        assert result2.name == "idx"

        result3 = result.to_period(freq="3M")
        exp = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="3M", name="idx")
        tm.assert_index_equal(result3, exp)
        assert result3.freqstr == "3M"

        msg = "Frequency must be positive, because it represents span: -2Y"
        with pytest.raises(ValueError, match=msg):
            result.to_period(freq="-2Y")

    def test_to_timestamp_preserve_name(self):
        # [FUNCTION DOC]
        # The name survives conversion.
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009", name="foo")
        assert index.name == "foo"

        conv = index.to_timestamp("D")
        assert conv.name == "foo"

    def test_to_timestamp_quarterly_bug(self):
        # [FUNCTION DOC]
        # Index built from year and quarter fields converts with end-of-period option to
        # the same stamps and frequency as converting each element.
        years = np.arange(1960, 2000).repeat(4)
        quarters = np.tile(list(range(1, 5)), 40)

        pindex = PeriodIndex.from_fields(year=years, quarter=quarters)

        stamps = pindex.to_timestamp("D", "end")
        expected = DatetimeIndex([x.to_timestamp("D", "end") for x in pindex])
        tm.assert_index_equal(stamps, expected)
        assert stamps.freq == expected.freq

    def test_to_timestamp_pi_mult(self):
        # [FUNCTION DOC]
        # Index with a 2-month multiple converts to the first day by default and, with
        # the end option, to the last instant of the span.
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="2M", name="idx")

        result = idx.to_timestamp()
        expected = DatetimeIndex(
            ["2011-01-01", "NaT", "2011-02-01"], dtype="M8[us]", name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E")
        expected = DatetimeIndex(
            ["2011-02-28", "NaT", "2011-03-31"], dtype="M8[us]", name="idx"
        )
        expected = expected + Timedelta(1, "D") - Timedelta(1, "us")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_combined(self):
        # [FUNCTION DOC]
        # Combined day-plus-hour frequency: start timestamps, end-of-span timestamps and
        # end timestamps with explicit hourly freq.
        idx = period_range(start="2011", periods=2, freq="1D1h", name="idx")

        result = idx.to_timestamp()
        expected = DatetimeIndex(
            ["2011-01-01 00:00", "2011-01-02 01:00"], dtype="M8[us]", name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E")
        expected = DatetimeIndex(
            ["2011-01-02 00:59:59", "2011-01-03 01:59:59"], name="idx", dtype="M8[us]"
        )
        expected = expected + Timedelta(1, "s") - Timedelta(1, "us")
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E", freq="h")
        expected = DatetimeIndex(
            ["2011-01-02 00:00", "2011-01-03 01:00"], dtype="M8[us]", name="idx"
        )
        expected = expected + Timedelta(1, "h") - Timedelta(1, "us")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_1703(self):
        # [FUNCTION DOC]
        # The first timestamp of a daily range equals the matching date (GitHub 1703).
        index = period_range("1/1/2012", periods=4, freq="D")

        result = index.to_timestamp()
        assert result[0] == Timestamp("1/1/2012")


def test_ms_to_timestamp_error_message():
    # [FUNCTION DOC]
    # Passing the month-start alias to to_timestamp raises a value error suggesting the
    # month-end alias instead.
    # https://github.com/pandas-dev/pandas/issues/58974#issuecomment-2164265446
    ix = period_range("2000", periods=3, freq="M")
    with pytest.raises(ValueError, match="for Period, please use 'M' instead of 'MS'"):
        ix.to_timestamp("MS")
