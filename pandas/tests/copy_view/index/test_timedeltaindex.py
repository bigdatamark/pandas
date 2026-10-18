# [MODULE DOC]
# Copy-on-write tests for TimedeltaIndex: an index created from a Series must
# not change when the Series is modified, and construction copies its input.
# A module-level marker ignores future warnings about setting a value on a view.
# No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Series,
    Timedelta,
    TimedeltaIndex,
    array,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)


@pytest.mark.parametrize(
    "cons",
    [
        lambda x: TimedeltaIndex(x),
        lambda x: TimedeltaIndex(TimedeltaIndex(x)),
    ],
)
def test_timedeltaindex(cons):
    # [FUNCTION DOC]
    # Parametrized over constructor (single or doubly wrapped TimedeltaIndex). Builds
    # the index from a timedelta Series, modifies the Series, and verifies the index is
    # unchanged.
    dt = timedelta_range("1 day", periods=3)
    ser = Series(dt)
    idx = cons(ser)
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timedelta("5 days")
    tm.assert_index_equal(idx, expected)


def test_constructor_copy_input_timedelta_ndarray_default():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Verifies constructing a TimedeltaIndex from a numpy
    # timedelta array copies by default.
    # GH 63388
    arr = np.array([1, 2], dtype="timedelta64[ns]")
    idx = TimedeltaIndex(arr)
    assert not np.shares_memory(arr, get_array(idx))


def test_constructor_copy_input_timedelta_ea_default():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Verifies constructing a TimedeltaIndex from a timedelta
    # extension array copies by default.
    # GH 63388
    arr = array([1, 2], dtype="timedelta64[ns]")
    idx = TimedeltaIndex(arr)
    assert not tm.shares_memory(arr, idx.array)


def test_series_from_temporary_timedeltaindex_readonly_data():
    # [FUNCTION DOC]
    # Regression test (GH 63388). With a read-only numpy source, verifies a Series
    # built from a temporary TimedeltaIndex does not share memory and can be modified,
    # yielding the expected values.
    # GH 63388
    arr = np.array([1, 2], dtype="timedelta64[ns]")
    arr.flags.writeable = False
    ser = Series(TimedeltaIndex(arr))
    assert not np.shares_memory(arr, get_array(ser))
    ser.iloc[0] = Timedelta(days=1)
    expected = Series(
        [Timedelta(days=1), Timedelta(nanoseconds=2)], dtype="timedelta64[ns]"
    )
    tm.assert_series_equal(ser, expected)
