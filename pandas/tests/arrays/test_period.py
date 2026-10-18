# [MODULE DOC]
# Tests specific to PeriodArray: dtype registration, integer representation,
# take / fillna / setitem error handling (incompatible frequency, length, type),
# subtraction errors, where with a different frequency, and repr. Complements the
# shared datetime-like tests. No import-time side effects.

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray

# ----------------------------------------------------------------------------
# Dtype


def test_registered():
    # [FUNCTION DOC]
    # Verifies the period dtype class is registered in the extension dtype registry and
    # that looking up the string alias Period[D] yields the daily period dtype.
    assert PeriodDtype in registry.dtypes
    result = registry.find("Period[D]")
    expected = PeriodDtype("D")
    assert result == expected


# ----------------------------------------------------------------------------
# period_array


def test_asi8():
    # [FUNCTION DOC]
    # Verifies the asi8 integer representation of daily periods in the years 2000
    # and 2001 is the day ordinal (10957, 11323) with the integer NaT marker for missing.
    result = PeriodArray._from_sequence(["2000", "2001", None], dtype="period[D]").asi8
    expected = np.array([10957, 11323, iNaT])
    tm.assert_numpy_array_equal(result, expected)


def test_take_raises():
    # [FUNCTION DOC]
    # Verifies take with a fill period of another frequency raises an incompatible
    # frequency error, and a string fill value raises a type error saying a Period
    # or NaT is expected.
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr.take([0, -1], allow_fill=True, fill_value=pd.Period("2000", freq="W"))

    msg = "value should be a 'Period' or 'NaT'. Got 'str' instead"
    with pytest.raises(TypeError, match=msg):
        arr.take([0, -1], allow_fill=True, fill_value="foo")


def test_fillna_raises():
    # [FUNCTION DOC]
    # Verifies fillna with an array of the wrong length raises a value error mentioning
    # Length.
    arr = PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    with pytest.raises(ValueError, match="Length"):
        arr.fillna(arr[:2])


def test_fillna_copies():
    # [FUNCTION DOC]
    # Verifies fillna with a scalar period returns a new array (not the same object)
    # even if nothing needed filling.
    arr = PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    result = arr.fillna(pd.Period("2000", "D"))
    assert result is not arr


# ----------------------------------------------------------------------------
# setitem


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ([0], pd.Period("2000", "D"), [10957, 1, 2]),
        ([0], None, [iNaT, 1, 2]),
        ([0], np.nan, [iNaT, 1, 2]),
        ([0, 1, 2], pd.Period("2000", "D"), [10957] * 3),
        (
            [0, 1, 2],
            [pd.Period("2000", "D"), pd.Period("2001", "D"), pd.Period("2002", "D")],
            [10957, 11323, 11688],
        ),
    ],
)
def test_setitem(key, value, expected):
    # [FUNCTION DOC]
    # Parametrized over (key list, value, expected integer ordinals): scalar period,
    # None, NaN (both become missing), the whole array with a scalar, and elementwise
    # list values. Verifies setitem on a daily period array equals the expected array.
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    expected = PeriodArray(expected, dtype="period[D]")
    arr[key] = value
    tm.assert_period_array_equal(arr, expected)


def test_setitem_raises_incompatible_freq():
    # [FUNCTION DOC]
    # Verifies assigning a period of another frequency, or an array with another
    # frequency, raises an incompatible frequency error.
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr[0] = pd.Period("2000", freq="Y")

    other = PeriodArray._from_sequence(["2000", "2001"], dtype="period[Y]")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr[[0, 1]] = other


def test_setitem_raises_length():
    # [FUNCTION DOC]
    # Verifies assigning a list value of shorter length than the key raises a value
    # error mentioning length.
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    with pytest.raises(ValueError, match="length"):
        arr[[0, 1]] = [pd.Period("2000", freq="D")]


def test_setitem_raises_type():
    # [FUNCTION DOC]
    # Verifies assigning an integer raises a type error mentioning int.
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    with pytest.raises(TypeError, match="int"):
        arr[0] = 1


# ----------------------------------------------------------------------------
# Ops


def test_sub_period():
    # [FUNCTION DOC]
    # Verifies subtracting a period of a different frequency raises an incompatible
    # frequency error.
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    other = pd.Period("2000", freq="M")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr - other


def test_sub_period_overflow():
    # [FUNCTION DOC]
    # Regression test (GH#47538). Verifies subtraction of a nanosecond period array and
    # a very large ordinal period overflowing 64-bit arithmetic raises an overflow
    # error ("Overflow in int64 addition"), in both operand orders.
    # GH#47538
    dti = pd.date_range("1677-09-22", periods=2, freq="D")
    pi = dti.to_period("ns")

    per = pd.Period._from_ordinal(10**14, pi.freq)

    with pytest.raises(OverflowError, match="Overflow in int64 addition"):
        pi - per

    with pytest.raises(OverflowError, match="Overflow in int64 addition"):
        per - pi


# ----------------------------------------------------------------------------
# Methods


@pytest.mark.parametrize(
    "other",
    [
        pd.Period("2000", freq="h"),
        PeriodArray._from_sequence(["2000", "2001", "2000"], dtype="period[h]"),
    ],
)
def test_where_different_freq_raises(other):
    # [FUNCTION DOC]
    # Regression test (GH#45768). Parametrized over a period scalar and a period
    # array of hourly frequency. Verifies the private where method on a daily array
    # raises an incompatible frequency error while Series.where coerces to object and
    # equals the object-dtype result.
    # GH#45768 The PeriodArray method raises, the Series method coerces
    ser = pd.Series(
        PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    )
    cond = np.array([True, False, True])

    with pytest.raises(IncompatibleFrequency, match="freq"):
        ser.array._where(cond, other)

    res = ser.where(cond, other)
    expected = ser.astype(object).where(cond, other)
    tm.assert_series_equal(res, expected)


# ----------------------------------------------------------------------------
# Printing


def test_repr_small():
    # [FUNCTION DOC]
    # Verifies the string form of a short period array: class header, list of ISO dates,
    # length and dtype lines.
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    result = str(arr)
    expected = (
        "<PeriodArray>\n['2000-01-01', '2001-01-01']\nLength: 2, dtype: period[D]"
    )
    assert result == expected


def test_repr_large():
    # [FUNCTION DOC]
    # Verifies the string form of a long (1000 elements) period array is truncated
    # with an ellipsis after the first ten and before the last ten elements, wrapped
    # across lines, and ends with length and dtype.
    arr = PeriodArray._from_sequence(["2000", "2001"] * 500, dtype="period[D]")
    result = str(arr)
    expected = (
        "<PeriodArray>\n"
        "['2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01',\n"
        " ...\n"
        " '2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01']\n"
        "Length: 1000, dtype: period[D]"
    )
    assert result == expected
