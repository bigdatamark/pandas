# [MODULE DOC]
# Tests for the Timedelta rounding methods (round, floor, ceil): correctness for many frequencies,
# invalid frequency errors, overflow at the representable bounds, a property-based sanity check, preservation of
# non-nanosecond resolution, and frequencies finer than the resolution.

from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import iNaT
from pandas.errors import OutOfBoundsTimedelta

from pandas import Timedelta


class TestTimedeltaRound:
    # [CLASS DOC]
    # Groups tests of Timedelta rounding.
    @pytest.mark.parametrize(
        "freq,s1,s2",
        [
            # This first case has s1, s2 being the same as t1,t2 below
            (
                "ns",
                "1 days 02:34:56.789123456",
                "-1 days 02:34:56.789123456",
            ),
            (
                "us",
                "1 days 02:34:56.789123000",
                "-1 days 02:34:56.789123000",
            ),
            (
                "ms",
                "1 days 02:34:56.789000000",
                "-1 days 02:34:56.789000000",
            ),
            ("s", "1 days 02:34:57", "-1 days 02:34:57"),
            ("2s", "1 days 02:34:56", "-1 days 02:34:56"),
            ("5s", "1 days 02:34:55", "-1 days 02:34:55"),
            ("min", "1 days 02:35:00", "-1 days 02:35:00"),
            ("12min", "1 days 02:36:00", "-1 days 02:36:00"),
            ("h", "1 days 03:00:00", "-1 days 03:00:00"),
            ("D", "1 days", "-1 days"),
        ],
    )
    def test_round(self, freq, s1, s2):
        # [FUNCTION DOC]
        # Parametrized over a frequency string and the expected rounded values for a positive and a negative timedelta
        # with a nanosecond-precision fractional part. Rounding each of them to the frequency must give the expected values
        # (frequencies include ns, us, ms, multiples of seconds and minutes, hours and days).
        s1 = Timedelta(s1)
        s2 = Timedelta(s2)
        t1 = Timedelta("1 days 02:34:56.789123456")
        t2 = Timedelta("-1 days 02:34:56.789123456")

        r1 = t1.round(freq)
        assert r1 == s1
        r2 = t2.round(freq)
        assert r2 == s2

    def test_round_invalid(self):
        # [FUNCTION DOC]
        # Rounding to a non-fixed frequency (year end, month end) must raise a value error saying the frequency is non-fixed,
        # and an unrecognized frequency string must raise a value error saying the frequency is invalid.
        t1 = Timedelta("1 days 02:34:56.789123456")

        for freq, msg in [
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),
            ("ME", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ]:
            with pytest.raises(ValueError, match=msg):
                t1.round(freq)

    def test_round_implementation_bounds(self):
        # [FUNCTION DOC]
        # GH#38964. Near the representable limits, ceiling the minimum timedelta to seconds and flooring the maximum to seconds
        # give results offset from the extremes by the sub-second remainder. Flooring or rounding the minimum, and ceiling or
        # rounding the maximum, to seconds must raise the out-of-bounds timedelta error mentioning that rounding overflows.
        # See also: analogous test for Timestamp
        # GH#38964
        result = Timedelta.min.ceil("s")
        expected = Timedelta.min + Timedelta(seconds=1) - Timedelta(145224193)
        assert result == expected

        result = Timedelta.max.floor("s")
        expected = Timedelta.max - Timedelta(854775807)
        assert result == expected

        msg = (
            r"Cannot round -106752 days \+00:12:43.145224193 to freq=s without overflow"
        )
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.min.floor("s")
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.min.round("s")

        msg = "Cannot round 106751 days 23:47:16.854775807 to freq=s without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.max.ceil("s")
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta.max.round("s")

    @pytest.mark.slow
    @given(val=st.integers(min_value=iNaT + 1, max_value=lib.i8max))
    @pytest.mark.parametrize(
        "method", [Timedelta.round, Timedelta.floor, Timedelta.ceil]
    )
    def test_round_sanity(self, val, method):
        # [FUNCTION DOC]
        # Slow property-based test (hypothesis integers across the valid nanosecond range, parametrized over round, floor
        # and ceil). Builds a timedelta from the integer and runs the nested checker for units from nanoseconds to days.
        cls = Timedelta
        err_cls = OutOfBoundsTimedelta

        val = np.int64(val)
        td = cls(val)

        def checker(ts, nanos, unit):
            # [FUNCTION DOC]
            # HELPER: nested checker for the sanity test above.
            # Arguments: the timedelta, the number of nanoseconds per unit, and the unit string. It first predicts, using
            # exact integer arithmetic, whether the lower or upper rounding boundary would exceed the representable range and,
            # if so, asserts the out-of-bounds error is raised and returns. Otherwise it applies the method and asserts the
            # result is a multiple of the unit, differs from the input by less than one unit, is within half a unit for round,
            # is not above the input for floor, and is not below it for ceil. A unit of one nanosecond skips the overflow prediction.
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
        checker(td, nanos, "ns")

        nanos = 1000
        checker(td, nanos, "us")

        nanos = 1_000_000
        checker(td, nanos, "ms")

        nanos = 1_000_000_000
        checker(td, nanos, "s")

        nanos = 60 * 1_000_000_000
        checker(td, nanos, "min")

        nanos = 60 * 60 * 1_000_000_000
        checker(td, nanos, "h")

        nanos = 24 * 60 * 60 * 1_000_000_000
        checker(td, nanos, "D")

    def test_round_non_nano(self, unit):
        # [FUNCTION DOC]
        # Parametrized over the unit fixture. A timedelta of one day, two hours, thirty-four minutes and fifty-seven seconds converted
        # to the given unit is rounded, floored and ceiled to minutes; results must equal the expected nanosecond-resolution
        # timedelta values and keep the original resolution.
        td = Timedelta("1 days 02:34:57").as_unit(unit)

        res = td.round("min")
        assert res == Timedelta("1 days 02:35:00")
        assert res._creso == td._creso

        res = td.floor("min")
        assert res == Timedelta("1 days 02:34:00")
        assert res._creso == td._creso

        res = td.ceil("min")
        assert res == Timedelta("1 days 02:35:00")
        assert res._creso == td._creso

    def test_round_freq_finer_than_resolution(self):
        # [FUNCTION DOC]
        # GH#64828. For a timedelta at second resolution, rounding, flooring and ceiling to a frequency of one hundred
        # milliseconds (finer than the resolution) must leave the value unchanged and the unit remains seconds.
        # GH#64828
        td = Timedelta(1.0, unit="days").as_unit("s")
        assert td.unit == "s"
        assert td.round("100ms") == Timedelta("1 days 00:00:00")
        assert td.floor("100ms") == Timedelta("1 days 00:00:00")
        assert td.ceil("100ms") == Timedelta("1 days 00:00:00")
