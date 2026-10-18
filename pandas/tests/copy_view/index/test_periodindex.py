# [MODULE DOC]
# Copy-on-write tests for PeriodIndex: an index created from a Series must not
# change when the Series is modified, and construction copies its input.
# A module-level marker ignores future warnings about setting a value on a view.
# No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Period,
    PeriodIndex,
    Series,
    array,
    period_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)


@pytest.mark.parametrize("box", [lambda x: x, PeriodIndex])
def test_periodindex(box):
    # [FUNCTION DOC]
    # Parametrized over wrapper (identity, or an extra PeriodIndex construction).
    # Builds a PeriodIndex from a period Series, modifies the Series, and verifies the
    # index is unchanged.
    dt = period_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    idx = box(PeriodIndex(ser))
    expected = idx.copy(deep=True)
    ser.iloc[0] = Period("2020-12-31")
    tm.assert_index_equal(idx, expected)


def test_constructor_copy_input_period_ea_default():
    # [FUNCTION DOC]
    # Regression test (GH 63388). Verifies constructing a PeriodIndex from a period
    # extension array copies by default.
    # GH 63388
    arr = array(["2020-01-01", "2020-01-02"], dtype="period[D]")
    idx = PeriodIndex(arr)
    assert not tm.shares_memory(arr, idx.array)


def test_series_from_temporary_periodindex_readonly_data():
    # [FUNCTION DOC]
    # Regression test (GH 63388). With a read-only backing array, verifies a Series
    # built from a temporary PeriodIndex does not share memory and can be modified.
    # GH 63388
    arr = array(["2020-01-01", "2020-01-02"], dtype="period[D]")
    arr._ndarray.flags.writeable = False
    ser = Series(PeriodIndex(arr))
    assert not np.shares_memory(arr._ndarray, get_array(ser))
    ser.iloc[0] = Period("2022-01-01", freq="D")
    expected = Series(
        [Period("2022-01-01", freq="D"), Period("2020-01-02", freq="D")],
        dtype="period[D]",
    )
    tm.assert_series_equal(ser, expected)
