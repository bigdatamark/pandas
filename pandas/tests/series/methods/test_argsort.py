# [MODULE DOC]
# Tests for Series.argsort, including interaction with NumPy's argsort,
# datetime data with missing values, sort stability and name preservation.
# Uses the datetime_series and unit fixtures. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm


class TestSeriesArgsort:
    # [CLASS DOC]
    # Groups the tests of Series.argsort. Stateless; fixtures are supplied per test.
    def test_argsort_axis(self):
        # [FUNCTION DOC]
        # Regression test (GH#54257): calling argsort with an axis value that does not
        # exist for a Series (2) raises a value error saying no such axis exists.
        # GH#54257
        ser = Series(range(3))

        msg = "No axis named 2 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser.argsort(axis=2)

    def test_argsort_numpy(self, datetime_series):
        # [FUNCTION DOC]
        # Applying NumPy's argsort function to the series must give the same integer
        # positions as applying it to the underlying plain array.
        ser = datetime_series
        res = np.argsort(ser).values
        expected = np.argsort(np.array(ser))
        tm.assert_numpy_array_equal(res, expected)

    def test_argsort_numpy_missing(self):
        # [FUNCTION DOC]
        # NumPy's argsort applied to a float series containing NaN must match the
        # result for the equivalent plain array (NaN values sorted to the end).
        data = [0.1, np.nan, 0.2, np.nan, 0.3]
        ser = Series(data)
        result = np.argsort(ser)
        expected = np.argsort(np.array(data))

        tm.assert_numpy_array_equal(result.values, expected)

    def test_argsort(self, datetime_series):
        # [FUNCTION DOC]
        # The result of argsort on the datetime_series fixture has an integer dtype.
        argsorted = datetime_series.argsort()
        assert issubclass(argsorted.dtype.type, np.integer)

    def test_argsort_dt64(self, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime resolution unit. Builds five consecutive dates with
        # the given unit, checks that a shift by minus one keeps the dtype and makes the
        # last element missing, and that argsort of the original gives positions 0 to 4
        # and argsort of the shifted series also gives 0 to 4 (the missing entry is
        # placed last). Expected results have the platform pointer integer dtype.
        # GH#2967 (introduced bug in 0.11-dev I think)
        ser = Series(
            [Timestamp(f"201301{i:02d}") for i in range(1, 6)], dtype=f"M8[{unit}]"
        )
        assert ser.dtype == f"datetime64[{unit}]"
        shifted = ser.shift(-1)
        assert shifted.dtype == f"datetime64[{unit}]"
        assert isna(shifted[4])

        result = ser.argsort()
        expected = Series(range(5), dtype=np.intp)
        tm.assert_series_equal(result, expected)

        result = shifted.argsort()
        expected = Series([*list(range(4)), 4], dtype=np.intp)
        tm.assert_series_equal(result, expected)

    def test_argsort_stable(self):
        # [FUNCTION DOC]
        # Sorts 10000 random integers with the mergesort kind and with the default kind
        # and compares each against NumPy's argsort with the same kind, expecting equal
        # positions. Also verifies that comparing a Series directly with the numpy array
        # comparison helper raises an assertion error about the wrong type.
        ser = Series(np.random.default_rng(2).integers(0, 100, size=10000))
        mindexer = ser.argsort(kind="mergesort")
        qindexer = ser.argsort()

        mexpected = np.argsort(ser.values, kind="mergesort")
        qexpected = np.argsort(ser.values, kind="quicksort")

        tm.assert_series_equal(mindexer.astype(np.intp), Series(mexpected))
        tm.assert_series_equal(qindexer.astype(np.intp), Series(qexpected))
        msg = (
            r"ndarray Expected type <class 'numpy\.ndarray'>, "
            r"found <class 'pandas\.Series'> instead"
        )
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(qindexer, mindexer)

    def test_argsort_preserve_name(self, datetime_series):
        # [FUNCTION DOC]
        # The Series returned by argsort keeps the name of the original series.
        result = datetime_series.argsort()
        assert result.name == datetime_series.name
