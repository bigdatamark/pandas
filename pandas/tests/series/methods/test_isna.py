"""
We also test Series.notna in this file.
"""

# [MODULE DOC]
# Tests for Series.isna and Series.notna (in addition to isna as the docstring
# says). No import-time side effects.

import numpy as np

from pandas import (
    Period,
    Series,
)
import pandas._testing as tm


class TestIsna:
    # [CLASS DOC]
    # Groups the tests of Series.isna and notna. Stateless.
    def test_isna_period_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH#13737): in a monthly period Series with a NaT period, isna
        # flags the NaT and notna gives the complement.
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        expected = Series([False, True])

        result = ser.isna()
        tm.assert_series_equal(result, expected)

        result = ser.notna()
        tm.assert_series_equal(result, ~expected)

    def test_isna(self):
        # [FUNCTION DOC]
        # isna and notna on float and string Series: only NaN is missing, not the empty
        # string or zero.
        ser = Series([0, 5.4, 3, np.nan, -0.001])
        expected = Series([False, False, False, True, False])
        tm.assert_series_equal(ser.isna(), expected)
        tm.assert_series_equal(ser.notna(), ~expected)

        ser = Series(["hi", "", np.nan])
        expected = Series([False, False, True])
        tm.assert_series_equal(ser.isna(), expected)
        tm.assert_series_equal(ser.notna(), ~expected)
