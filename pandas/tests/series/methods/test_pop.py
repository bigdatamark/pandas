# [MODULE DOC]
# Test for Series.pop. No import-time side effects.

from pandas import Series
import pandas._testing as tm


def test_pop():
    # [FUNCTION DOC]
    # GH#6600: popping a label returns its value and removes the entry in place,
    # keeping the Series name.
    # GH#6600
    ser = Series([0, 4, 0], index=["A", "B", "C"], name=4)

    result = ser.pop("B")
    assert result == 4

    expected = Series([0, 0], index=["A", "C"], name=4)
    tm.assert_series_equal(ser, expected)
