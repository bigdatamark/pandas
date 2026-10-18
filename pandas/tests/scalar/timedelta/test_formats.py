# [MODULE DOC]
# Tests for the text formats of Timedelta: its representation, ISO 8601 duration output and the private
# base-representation helper in each of its formats (default, sub-day, long, all).
# Note that the base-representation tests bind a throwaway lambda to a local name; lambdas cannot carry doc blocks.

import pytest

from pandas import Timedelta


@pytest.mark.parametrize(
    "td, expected_repr",
    [
        (Timedelta(10, unit="D"), "Timedelta('10 days 00:00:00')"),
        (Timedelta(10, unit="s"), "Timedelta('0 days 00:00:10')"),
        (Timedelta(10, unit="ms"), "Timedelta('0 days 00:00:00.010000')"),
        (Timedelta(-10, unit="ms"), "Timedelta('-1 days +23:59:59.990000')"),
    ],
)
def test_repr(td, expected_repr):
    # [FUNCTION DOC]
    # Parametrized over timedeltas and expected representation strings. The representation must name the class and show
    # days and time of day, using the negative-days-plus-time form for negative values.
    assert repr(td) == expected_repr


@pytest.mark.parametrize(
    "td, expected_iso",
    [
        (
            Timedelta(
                days=6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
            "P6DT0H50M3.010010012S",
        ),
        (Timedelta(days=4, hours=12, minutes=30, seconds=5), "P4DT12H30M5S"),
        (Timedelta(nanoseconds=123), "P0DT0H0M0.000000123S"),
        # trim nano
        (Timedelta(microseconds=10), "P0DT0H0M0.00001S"),
        # trim micro
        (Timedelta(milliseconds=1), "P0DT0H0M0.001S"),
        # don't strip every 0
        (Timedelta(minutes=1), "P0DT0H1M0S"),
    ],
)
def test_isoformat(td, expected_iso):
    # [FUNCTION DOC]
    # Parametrized over timedeltas with a full mix of components and the expected ISO duration string. Trailing zeros of
    # the fractional seconds are trimmed to whole microsecond or millisecond groups but a zero seconds field is kept.
    assert td.isoformat() == expected_iso


class TestReprBase:
    # [CLASS DOC]
    # Groups tests of the private base-representation method with different format arguments.
    def test_none(self):
        # [FUNCTION DOC]
        # Default format: whole days omit the time of day, otherwise the time of day follows; fractional seconds show six digits
        # only when non-zero; negative values use the negative-days-plus-time form.
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1s = Timedelta(1, unit="s")
        delta_500ms = Timedelta(500, unit="ms")

        drepr = lambda x: x._repr_base()
        assert drepr(delta_1d) == "1 days"
        assert drepr(-delta_1d) == "-1 days"
        assert drepr(delta_0d) == "0 days"
        assert drepr(delta_1s) == "0 days 00:00:01"
        assert drepr(delta_500ms) == "0 days 00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    def test_sub_day(self):
        # [FUNCTION DOC]
        # The sub-day format: like the default, but a zero timedelta shows only the time of day and whole days show only days.
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1s = Timedelta(1, unit="s")
        delta_500ms = Timedelta(500, unit="ms")

        drepr = lambda x: x._repr_base(format="sub_day")
        assert drepr(delta_1d) == "1 days"
        assert drepr(-delta_1d) == "-1 days"
        assert drepr(delta_0d) == "00:00:00"
        assert drepr(delta_1s) == "00:00:01"
        assert drepr(delta_500ms) == "00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    def test_long(self):
        # [FUNCTION DOC]
        # The long format always shows days and a time of day (and fractional part when non-zero), including for whole days and zero.
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1s = Timedelta(1, unit="s")
        delta_500ms = Timedelta(500, unit="ms")

        drepr = lambda x: x._repr_base(format="long")
        assert drepr(delta_1d) == "1 days 00:00:00"
        assert drepr(-delta_1d) == "-1 days +00:00:00"
        assert drepr(delta_0d) == "0 days 00:00:00"
        assert drepr(delta_1s) == "0 days 00:00:01"
        assert drepr(delta_500ms) == "0 days 00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    def test_all(self):
        # [FUNCTION DOC]
        # The all format always shows days and a time of day with nine fractional digits (nanosecond precision),
        # for whole days, zero, and a single nanosecond, and in the negative-days-plus-time form for negatives.
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1ns = Timedelta(1, unit="ns")

        drepr = lambda x: x._repr_base(format="all")
        assert drepr(delta_1d) == "1 days 00:00:00.000000000"
        assert drepr(-delta_1d) == "-1 days +00:00:00.000000000"
        assert drepr(delta_0d) == "0 days 00:00:00.000000000"
        assert drepr(delta_1ns) == "0 days 00:00:00.000000001"
        assert drepr(-delta_1d + delta_1ns) == "-1 days +00:00:00.000000001"
