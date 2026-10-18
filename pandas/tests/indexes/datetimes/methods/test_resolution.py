# [MODULE DOC]
# Test of the resolution property of DatetimeIndex.
from dateutil.tz import tzlocal
import pytest

from pandas.compat import (
    IS64,
    WASM,
)

from pandas import date_range


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("YE", "day"),
        ("QE", "day"),
        ("ME", "day"),
        ("D", "day"),
        ("h", "hour"),
        ("min", "minute"),
        ("s", "second"),
        ("ms", "millisecond"),
        ("us", "microsecond"),
    ],
)
@pytest.mark.skipif(WASM, reason="OverflowError received on WASM")
def test_dti_resolution(request, tz_naive_fixture, freq, expected):
    # [FUNCTION DOC]
    # Parametrized over frequency and expected resolution (year, quarter and month ends report day;
    # smaller frequencies report themselves up to microsecond) and tz_naive_fixture. Skipped on WASM
    # (overflow errors). Builds a 30-period range from 2013-04-01 and asserts the resolution name; for
    # year-end frequency with local timezone on 32-bit or WASM platforms the test is marked as an
    # expected failure because of overflow in the local timezone past the year 2038.
    tz = tz_naive_fixture
    if freq == "YE" and ((not IS64) or WASM) and isinstance(tz, tzlocal):
        request.applymarker(
            pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
        )

    idx = date_range(start="2013-04-01", periods=30, freq=freq, tz=tz)
    assert idx.resolution == expected
