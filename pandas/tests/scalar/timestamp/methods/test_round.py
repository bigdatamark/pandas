# [MODULE DOC]
# Tests for the Timestamp rounding methods (round to nearest, floor, ceil) to a
# frequency. Covers zero and invalid frequencies, ordinary and sub-second frequencies,
# timezone-aware inputs, daylight-saving ambiguous and nonexistent handling,
# integer-precision accuracy, implementation bounds with out-of-bounds errors, and a
# property-based sanity test.
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

import pandas._testing as tm


class TestTimestampRound:
    # [CLASS DOC]
    # Groups the Timestamp rounding tests. Stateless.
    def test_round_division_by_zero_raises(self):
        # [FUNCTION DOC]
        # Verifies that rounding to a zero-length frequency raises a value error saying
        # there was a division by zero in rounding.
        ts = Timestamp("2016-01-01")

        msg = "Division by zero in rounding"
        with pytest.raises(ValueError, match=msg):
            ts.round("0ns")

    @pytest.mark.parametrize(
        "timestamp, freq, expected",
        [
            ("20130101 09:10:11", "D", "20130101"),
            ("20130101 19:10:11", "D", "20130102"),
            ("20130201 12:00:00", "D", "20130202"),
            ("20130104 12:00:00", "D", "20130105"),
            ("2000-01-05 05:09:15.13", "D", "2000-01-05 00:00:00"),
            ("2000-01-05 05:09:15.13", "h", "2000-01-05 05:00:00"),
            ("2000-01-05 05:09:15.13", "s", "2000-01-05 05:09:15"),
        ],
    )
    def test_round_frequencies(self, timestamp, freq, expected):
        # [FUNCTION DOC]
        # Parametrized over timestamp, frequency (daily, hourly, per-second) and
        # expected result. Rounding to nearest gives the expected timestamp (the half-day
        # cases listed round to the later day).
        dt = Timestamp(timestamp)
        result = dt.round(freq)
        expected = Timestamp(expected)
        assert result == expected

    def test_round_tzaware(self):
        # [FUNCTION DOC]
        # Verifies rounding a US/Eastern aware timestamp to a day gives midnight of that
        # date in the same zone, and rounding to seconds when there is no sub-second
        # part returns an equal timestamp.
        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("D")
        expected = Timestamp("20130101", tz="US/Eastern")
        assert result == expected

        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("s")
        assert result == dt

    def test_round_30min(self):
        # [FUNCTION DOC]
        # Verifies rounding 12:32 to a thirty-minute frequency gives 12:30.
        # round
        dt = Timestamp("20130104 12:32:00")
        result = dt.round("30Min")
        expected = Timestamp("20130104 12:30:00")
        assert result == expected

    def test_round_subsecond(self):
        # [FUNCTION DOC]
        # Regression test (GH 14440, GH 15578). Rounding to milliseconds rounds up at
        # exactly half and down below half; rounding a millisecond-precision value to
        # microsecond or nanosecond leaves it unchanged; rounding to ten nanoseconds
        # works on a nanosecond-precision value.
        # GH#14440 & GH#15578
        result = Timestamp("2016-10-17 12:00:00.0015").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.002000")
        assert result == expected

        result = Timestamp("2016-10-17 12:00:00.00149").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.001000")
        assert result == expected

        ts = Timestamp("2016-10-17 12:00:00.0015")
        for freq in ["us", "ns"]:
            assert ts == ts.round(freq)

        result = Timestamp("2016-10-17 12:00:00.001501031").round("10ns")
        expected = Timestamp("2016-10-17 12:00:00.001501030")
        assert result == expected

    def test_round_nonstandard_freq(self):
        # [FUNCTION DOC]
        # Verifies rounding to a frequency that is not a divisor of a larger unit (1010
        # nanoseconds) completes without emitting any warning.
        with tm.assert_produces_warning(False):
            Timestamp("2016-10-17 12:00:00.001501031").round("1010ns")

    def test_round_invalid_arg(self):
        # [FUNCTION DOC]
        # Verifies an unrecognized frequency string raises a value error with the
        # invalid-frequency message.
        stamp = Timestamp("2000-01-05 05:09:15.13")
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            stamp.round("foo")

    @pytest.mark.parametrize(
        "test_input, rounder, freq, expected",
        [
            ("2117-01-01 00:00:45", "floor", "15s", "2117-01-01 00:00:45"),
            ("2117-01-01 00:00:45", "ceil", "15s", "2117-01-01 00:00:45"),
            (
                "2117-01-01 00:00:45.000000012",
                "floor",
                "10ns",
                "2117-01-01 00:00:45.000000010",
            ),
            (
                "1823-01-01 00:00:01.000000012",
                "ceil",
                "10ns",
                "1823-01-01 00:00:01.000000020",
            ),
            ("1823-01-01 00:00:01", "floor", "1s", "1823-01-01 00:00:01"),
            ("1823-01-01 00:00:01", "ceil", "1s", "1823-01-01 00:00:01"),
            ("NaT", "floor", "1s", "NaT"),
            ("NaT", "ceil", "1s", "NaT"),
        ],
    )
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        # [FUNCTION DOC]
        # Parametrized over input, rounder (floor or ceil), frequency and expected
        # value, including edge cases of values already on the grid and NaT. Floor or
        # ceil gives the expected timestamp, and NaT input gives NaT.
        dt = Timestamp(test_input)
        func = getattr(dt, rounder)
        result = func(freq)

        if dt is NaT:
            assert result is NaT
        else:
            expected = Timestamp(expected)
            assert result == expected

    @pytest.mark.parametrize(
        "test_input, freq, expected",
        [
            ("2018-01-01 00:02:06", "2s", "2018-01-01 00:02:06"),
            ("2018-01-01 00:02:00", "2min", "2018-01-01 00:02:00"),
            ("2018-01-01 00:04:00", "4min", "2018-01-01 00:04:00"),
            ("2018-01-01 00:15:00", "15min", "2018-01-01 00:15:00"),
            ("2018-01-01 00:20:00", "20min", "2018-01-01 00:20:00"),
            ("2018-01-01 03:00:00", "3h", "2018-01-01 03:00:00"),
        ],
    )
    @pytest.mark.parametrize("rounder", ["ceil", "floor", "round"])
    def test_round_minute_freq(self, test_input, freq, expected, rounder):
        # [FUNCTION DOC]
        # Regression test (GH 21262). Parametrized over inputs that are already exact
        # multiples of minute-like frequencies and over the three rounders (ceil, floor,
        # round). Expected: the timestamp is returned unchanged for every combination.
        # Ensure timestamps that shouldn't round dont!
        # GH#21262

        dt = Timestamp(test_input)
        expected = Timestamp(expected)
        func = getattr(dt, rounder)
        result = func(freq)
        assert result == expected

    def test_ceil(self, unit):
        # [FUNCTION DOC]
        # Verifies ceiling a timestamp to a day for each unit fixture gives the next
        # midnight and preserves the resolution code.
        dt = Timestamp("20130101 09:10:11").as_unit(unit)
        result = dt.ceil("D")
        expected = Timestamp("20130102")
        assert result == expected
        assert result._creso == dt._creso

    def test_floor(self, unit):
        # [FUNCTION DOC]
        # Verifies flooring a timestamp to a day for each unit fixture gives the
        # same-day midnight and preserves the resolution code.
        dt = Timestamp("20130101 09:10:11").as_unit(unit)
        result = dt.floor("D")
        expected = Timestamp("20130101")
        assert result == expected
        assert result._creso == dt._creso

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    def test_round_dst_border_ambiguous(self, method, unit):
        # [FUNCTION DOC]
        # Regression test (GH 18946) for rounding near a fall-back transition in Madrid.
        # Parametrized over method (ceil, round, floor) and unit. With ambiguous true
        # the result equals the input and keeps the resolution; with false it equals the
        # next hour as UTC converted to Madrid; with the NaT option it returns NaT; with
        # the raise option a value error saying dst cannot be inferred is raised.
        # GH 18946 round near "fall back" DST
        ts = Timestamp("2017-10-29 00:00:00", tz="UTC").tz_convert("Europe/Madrid")
        ts = ts.as_unit(unit)
        result = getattr(ts, method)("h", ambiguous=True)
        assert result == ts
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = getattr(ts, method)("h", ambiguous=False)
        expected = Timestamp("2017-10-29 01:00:00", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = getattr(ts, method)("h", ambiguous="NaT")
        assert result is NaT

        msg = "Cannot infer dst time"
        with pytest.raises(ValueError, match=msg):
            getattr(ts, method)("h", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2h"],
        ],
    )
    def test_round_dst_border_nonexistent(self, method, ts_str, freq, unit):
        # [FUNCTION DOC]
        # Regression test (GH 23324) for rounding near a spring-forward gap in Chicago.
        # Parametrized over method, input string and frequency, and unit. The
        # shift-forward option returns 03:00 local time and keeps the resolution; the
        # NaT option returns NaT; the raise option raises a value error mentioning the
        # nonexistent 02:00 time.
        # GH 23324 round near "spring forward" DST
        ts = Timestamp(ts_str, tz="America/Chicago").as_unit(unit)
        result = getattr(ts, method)(freq, nonexistent="shift_forward")
        expected = Timestamp("2018-03-11 03:00:00", tz="America/Chicago")
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = getattr(ts, method)(freq, nonexistent="NaT")
        assert result is NaT

        msg = "2018-03-11 02:00:00"
        with pytest.raises(ValueError, match=msg):
            getattr(ts, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2018-01-01 0:0:0.124999360",
            "2018-01-01 0:0:0.125000367",
            "2018-01-01 0:0:0.125500",
            "2018-01-01 0:0:0.126500",
            "2018-01-01 12:00:00",
            "2019-01-01 12:00:00",
        ],
    )
    @pytest.mark.parametrize(
        "freq",
        [
            "2ns",
            "3ns",
            "4ns",
            "5ns",
            "6ns",
            "7ns",
            "250ns",
            "500ns",
            "750ns",
            "1us",
            "19us",
            "250us",
            "500us",
            "750us",
            "1s",
            "2s",
            "3s",
            "1D",
        ],
    )
    def test_round_int64(self, timestamp, freq):
        # [FUNCTION DOC]
        # Regression test (GH 22591). Parametrized over several timestamps (including
        # ones with fine sub-second parts) and many frequencies from nanoseconds to a
        # day. Checks on the nanosecond integer: floor lies on the grid and is within
        # one step below; ceil lies on the grid and within one step above; round lies on
        # the grid and within half a step; when the step is even and the distance is
        # exactly half a step, the result's multiple index is even (round half to even).
        # check that all rounding modes are accurate to int64 precision
        # see GH#22591
        dt = Timestamp(timestamp).as_unit("ns")
        unit = to_offset(freq).nanos

        # test floor
        result = dt.floor(freq)
        assert result._value % unit == 0, f"floor not a {freq} multiple"
        assert 0 <= dt._value - result._value < unit, "floor error"

        # test ceil
        result = dt.ceil(freq)
        assert result._value % unit == 0, f"ceil not a {freq} multiple"
        assert 0 <= result._value - dt._value < unit, "ceil error"

        # test round
        result = dt.round(freq)
        assert result._value % unit == 0, f"round not a {freq} multiple"
        assert abs(result._value - dt._value) <= unit // 2, "round error"
        if unit % 2 == 0 and abs(result._value - dt._value) == unit // 2:
            # round half to even
            assert result._value // unit % 2 == 0, "round half to even error"

    def test_round_implementation_bounds(self):
        # [FUNCTION DOC]
        # Verifies the boundaries of what can be rounded. Ceil of the minimum timestamp
        # to seconds and floor of the maximum succeed with known values (the maximum
        # floor equals the maximum minus the leftover nanoseconds). Floor and round of
        # the minimum, and ceil and round of the maximum, raise an out-of-bounds
        # datetime error saying the value cannot be rounded to a second frequency.
        # See also: analogous test for Timedelta
        result = Timestamp.min.ceil("s")
        expected = Timestamp(1677, 9, 21, 0, 12, 44)
        assert result == expected

        result = Timestamp.max.floor("s")
        expected = Timestamp.max - Timedelta(854775807)
        assert result == expected

        msg = "Cannot round 1677-09-21 00:12:43.145224193 to freq=<Second>"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.floor("s")

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.round("s")

        msg = "Cannot round 2262-04-11 23:47:16.854775807 to freq=<Second>"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.ceil("s")

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.round("s")

    @pytest.mark.slow
    @given(val=st.integers(iNaT + 1, lib.i8max))
    @pytest.mark.parametrize(
        "method", [Timestamp.round, Timestamp.floor, Timestamp.ceil]
    )
    def test_round_sanity(self, val, method):
        # [FUNCTION DOC]
        # Property-based test (marked slow) over any valid 64-bit integer and each of
        # round, floor and ceil. A nested checker is run for units from nanoseconds to a
        # day. Expected: where the exact result would leave the supported range the
        # method raises an out-of-bounds datetime error mentioning overflow; otherwise
        # the result is a multiple of the step, within one step of the input, within
        # half a step for round, not above the input for floor, and not below it for
        # ceil.
        cls = Timestamp
        err_cls = OutOfBoundsDatetime

        val = np.int64(val)
        ts = cls(val)

        def checker(ts, nanos, unit):
            # [FUNCTION DOC]
            # HELPER: checker used by the sanity test for a single unit. Inputs: ts
            # (timestamp), nanos (step size in nanoseconds), unit (frequency string).
            # Closes over the method under test and the error class. For steps larger
            # than one nanosecond it computes the remainder, the lower and upper
            # multiples with unbounded integers and, when the method would have to go
            # outside the representable bounds, asserts the overflow error is raised and
            # returns. Otherwise it applies the method and asserts the result is a
            # multiple of the step, within one step of the input, and satisfies the
            # method-specific ordering or half-step bound. Returns nothing.
            # First check that we do raise in cases where we should
            if nanos == 1:
                pass
            else:
                div, mod = divmod(ts._value, nanos)
                diff = int(nanos - mod)
                lb = ts._value - mod
                assert lb <= ts._value  # i.e. no overflows with python ints
                ub = ts._value + diff
                assert ub > ts._value  # i.e. no overflows with python ints

                msg = "without overflow"
                if mod == 0:
                    # We should never be raising in this
                    pass
                elif method is cls.ceil:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif method is cls.floor:
                    if lb < cls.min._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif mod >= diff:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif lb < cls.min._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return

            res = method(ts, unit)

            td = res - ts
            diff = abs(td._value)
            assert diff < nanos
            assert res._value % nanos == 0

            if method is cls.round:
                assert diff <= nanos / 2
            elif method is cls.floor:
                assert res <= ts
            elif method is cls.ceil:
                assert res >= ts

        nanos = 1
        checker(ts, nanos, "ns")

        nanos = 1000
        checker(ts, nanos, "us")

        nanos = 1_000_000
        checker(ts, nanos, "ms")

        nanos = 1_000_000_000
        checker(ts, nanos, "s")

        nanos = 60 * 1_000_000_000
        checker(ts, nanos, "min")

        nanos = 60 * 60 * 1_000_000_000
        checker(ts, nanos, "h")

        nanos = 24 * 60 * 60 * 1_000_000_000
        checker(ts, nanos, "D")
