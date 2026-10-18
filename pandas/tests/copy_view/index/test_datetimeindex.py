# [MODULE DOC]
# Copy-on-write tests for DatetimeIndex: an index created from a Series must
# not change when the Series is later modified, and constructors copy their
# input. Uses the get_array helper of the copy_view test utilities. A module-level
# marker ignores future warnings about setting a value on a view. No import-time
# side effects.

import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Series,
    Timestamp,
    array,
    date_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)


@pytest.mark.parametrize("box", [lambda x: x, DatetimeIndex])
def test_datetimeindex(box):
    # [FUNCTION DOC]
    # Parametrized over wrapper (identity, or an extra DatetimeIndex construction). Builds
    # a DatetimeIndex from a datetime Series, modifies the Series, and verifies the
    # index is unchanged (it does not share data with the Series).
    dt = date_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    idx = box(DatetimeIndex(ser))
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31")
    tm.assert_index_equal(idx, expected)


def test_datetimeindex_tz_convert():
    # [FUNCTION DOC]
    # Same protection check for the index produced by tz_convert on an index built from a
    # tz-aware Series.
    dt = date_range("2019-12-31", periods=3, freq="D", tz="Europe/Berlin")
    ser = Series(dt)
    idx = DatetimeIndex(ser).tz_convert("US/Eastern")
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31", tz="Europe/Berlin")
    tm.assert_index_equal(idx, expected)


def test_datetimeindex_tz_localize():
    # [FUNCTION DOC]
    # Same protection check for the index produced by tz_localize on an index built
    # from a Series.
    dt = date_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    idx = DatetimeIndex(ser).tz_localize("Europe/Berlin")
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31")
    tm.assert_index_equal(idx, expected)


def test_datetimeindex_isocalendar():
    # [FUNCTION DOC]
    # Same protection check for the index of the frame returned by isocalendar on an
    # index built from a Series.
    dt = date_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    df = DatetimeIndex(ser).isocalendar()
    expected = df.index.copy(deep=True)
    ser.iloc[0] = Timestamp("2020-12-31")
    tm.assert_index_equal(df.index, expected)


def test_index_values():
    # [FUNCTION DOC]
    # Verifies the values attribute of a DatetimeIndex is a read-only numpy array.
    idx = date_range("2019-12-31", periods=3, freq="D")
    result = idx.values
    assert result.flags.writeable is False


def test_constructor_copy_input_datetime_ndarray_default():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Verifies constructing a DatetimeIndex from a numpy
    # datetime array copies it by default (no shared memory).
    # GH 63388
    arr = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    idx = DatetimeIndex(arr)
    assert not np.shares_memory(arr, get_array(idx))


def test_constructor_copy_input_datetime_ea_default():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Verifies constructing a DatetimeIndex from a
    # datetime extension array copies it by default.
    # GH 63388
    arr = array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    idx = DatetimeIndex(arr)
    assert not tm.shares_memory(arr, idx.array)


def test_series_from_temporary_datetimeindex_readonly_data():
    # [FUNCTION DOC]
    # Regression test (GH 63388). With a read-only numpy source, verifies a Series built
    # from a temporary DatetimeIndex does not share memory with the source and can
    # be modified, yielding the expected values.
    # GH 63388
    arr = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    arr.flags.writeable = False
    ser = Series(DatetimeIndex(arr))
    assert not np.shares_memory(arr, get_array(ser))
    ser.iloc[0] = Timestamp("2020-01-01")
    expected = Series(
        [Timestamp("2020-01-01"), Timestamp("2020-01-02")], dtype="datetime64[ns]"
    )
    tm.assert_series_equal(ser, expected)
