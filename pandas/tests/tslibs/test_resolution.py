# [MODULE DOC]
# Tests for resolution detection of integer datetime arrays and for the resolution name lookup from frequency strings, including rejection of the removed uppercase hour and second aliases.

import datetime

import numpy as np
import pytest

from pandas._libs.tslibs import (
    Resolution,
    get_resolution,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit


def test_get_resolution_nano():
    # [FUNCTION DOC]
    # An int64 array with a default unit gets resolution nanosecond (not the fallback day).
    # don't return the fallback RESO_DAY
    arr = np.array([1], dtype=np.int64)
    res = get_resolution(arr)
    assert res == Resolution.RESO_NS


def test_get_resolution_non_nano_data():
    # [FUNCTION DOC]
    # With the microsecond unit code, resolution is microsecond both without a timezone and with the UTC timezone.
    arr = np.array([1], dtype=np.int64)
    res = get_resolution(arr, None, NpyDatetimeUnit.NPY_FR_us.value)
    assert res == Resolution.RESO_US

    res = get_resolution(arr, datetime.UTC, NpyDatetimeUnit.NPY_FR_us.value)
    assert res == Resolution.RESO_US


@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("Y", "year"),
        ("Q", "quarter"),
        ("M", "month"),
        ("D", "day"),
        ("h", "hour"),
        ("min", "minute"),
        ("s", "second"),
        ("ms", "millisecond"),
        ("us", "microsecond"),
        ("ns", "nanosecond"),
    ],
)
def test_get_attrname_from_abbrev(freqstr, expected):
    # [FUNCTION DOC]
    # Parametrized over frequency abbreviations (year, quarter, month, day, hour, minute, second, milli, micro, nano) and the attribute names. The resolution found from the frequency string has an abbreviation equal to the input and an attribute name equal to the expected name.
    reso = Resolution.get_reso_from_freqstr(freqstr)
    assert reso.attr_abbrev == freqstr
    assert reso.attrname == expected


@pytest.mark.parametrize("freq", ["H", "S"])
def test_unit_H_S_raises(freq):
    # [FUNCTION DOC]
    # Regression test (GH#59143): the uppercase hour and second aliases raise a value error with the invalid frequency message when looking up a resolution.
    # GH#59143
    msg = f"Invalid frequency: {freq}"

    with pytest.raises(ValueError, match=msg):
        Resolution.get_reso_from_freqstr(freq)
