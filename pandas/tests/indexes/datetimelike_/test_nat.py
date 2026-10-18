# [MODULE DOC]
# Test of NaT-related attributes (the NA sentinel, NA-holding capability, NaN mask, hasnans) on
# TimedeltaIndex, PeriodIndex and DatetimeIndex (naive and UTC).
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "index_without_na",
    [
        TimedeltaIndex(["1 days", "2 days"]),
        PeriodIndex(["2011-01-01", "2011-01-02"], freq="D"),
        DatetimeIndex(["2011-01-01", "2011-01-02"]),
        DatetimeIndex(["2011-01-01", "2011-01-02"], tz="UTC"),
    ],
)
def test_nat(index_without_na):
    # [FUNCTION DOC]
    # Parametrized over four two-element indexes without missing values. Checks the NA value of
    # empty, missing-containing and plain indexes is NaT; the index can hold NA; the NaN mask and
    # hasnans reflect no missing values in the original, and a deep copy with NaT assigned into the
    # second element reports a mask with True second and hasnans True.
    empty_index = index_without_na[:0]

    index_with_na = index_without_na.copy(deep=True)
    index_with_na._data[1] = NaT

    assert empty_index._na_value is NaT
    assert index_with_na._na_value is NaT
    assert index_without_na._na_value is NaT

    idx = index_without_na
    assert idx._can_hold_na

    tm.assert_numpy_array_equal(idx._isnan, np.array([False, False]))
    assert idx.hasnans is False

    idx = index_with_na
    assert idx._can_hold_na

    tm.assert_numpy_array_equal(idx._isnan, np.array([False, True]))
    assert idx.hasnans is True
