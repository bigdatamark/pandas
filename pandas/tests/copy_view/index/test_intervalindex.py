# [MODULE DOC]
# Copy-on-write tests for IntervalIndex construction: inputs are copied by
# default and a Series from a temporary IntervalIndex can be modified safely even
# when the source data is read-only. No import-time side effects.

import numpy as np

from pandas import (
    Interval,
    IntervalIndex,
    Series,
    array,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_constructor_copy_input_interval_ea_default():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Verifies constructing an IntervalIndex from an
    # interval extension array copies by default (no shared memory).
    # GH 63388
    arr = array([Interval(0, 1), Interval(1, 2)])
    idx = IntervalIndex(arr)
    assert not tm.shares_memory(arr, idx.array)


def test_series_from_temporary_intervalindex_readonly_data():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Makes the left and right endpoint arrays of the
    # source read-only, builds a Series from a temporary IntervalIndex, verifies it does
    # not share memory with the source, and that setting an element works.
    # GH 63388
    arr = array([Interval(0, 1), Interval(1, 2)])
    arr._left.flags.writeable = False
    arr._right.flags.writeable = False
    ser = Series(IntervalIndex(arr))
    assert not np.shares_memory(arr._left, get_array(ser)._left)
    ser.iloc[0] = Interval(5, 6)
    expected = Series([Interval(5, 6), Interval(1, 2)], dtype="interval[int64, right]")
    tm.assert_series_equal(ser, expected)
