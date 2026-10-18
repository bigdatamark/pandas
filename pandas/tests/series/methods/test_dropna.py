# [MODULE DOC]
# Tests for Series.dropna: empty and complete Series, name preservation,
# in-place operation, interval, period, datetime and tz-aware data, and
# ignore_index. Uses the datetime_series and unit fixtures. No import-time side
# effects.

import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    IntervalIndex,
    NaT,
    Period,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestDropna:
    # [CLASS DOC]
    # Groups the tests of Series.dropna. Stateless.
    def test_dropna_empty(self):
        # [FUNCTION DOC]
        # Dropping from an empty object Series returns an empty Series; the in-place form
        # returns None and leaves it empty; asking for axis 1 raises a value error that
        # no such axis exists for a Series.
        ser = Series([], dtype=object)

        assert len(ser.dropna()) == 0
        return_value = ser.dropna(inplace=True)
        assert return_value is None
        assert len(ser) == 0

        # invalid axis
        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser.dropna(axis=1)

    def test_dropna_preserve_name(self, datetime_series):
        # [FUNCTION DOC]
        # After setting the first five values to NaN, dropna (copying or in place, the
        # latter returning None) keeps the Series name.
        datetime_series[:5] = np.nan
        result = datetime_series.dropna()
        assert result.name == datetime_series.name
        name = datetime_series.name
        ts = datetime_series.copy()
        return_value = ts.dropna(inplace=True)
        assert return_value is None
        assert ts.name == name

    def test_dropna_no_nan(self):
        # [FUNCTION DOC]
        # For integer and boolean Series without missing values, dropna returns an equal
        # but distinct object; the in-place form returns None and leaves the Series equal.
        for ser in [
            Series([1, 2, 3], name="x"),
            Series([False, True, False], name="x"),
        ]:
            result = ser.dropna()
            tm.assert_series_equal(result, ser)
            assert result is not ser

            s2 = ser.copy()
            return_value = s2.dropna(inplace=True)
            assert return_value is None
            tm.assert_series_equal(s2, ser)

    def test_dropna_intervals(self):
        # [FUNCTION DOC]
        # Series with an interval index containing a missing interval: dropna removes the
        # entry whose value is NaN, keeping the other three.
        ser = Series(
            [np.nan, 1, 2, 3],
            IntervalIndex.from_arrays([np.nan, 0, 1, 2], [np.nan, 1, 2, 3]),
        )

        result = ser.dropna()
        expected = ser.iloc[1:]
        tm.assert_series_equal(result, expected)

    def test_dropna_period_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH#13737): a period Series with a NaT period drops the
        # missing one.
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])
        result = ser.dropna()
        expected = Series([Period("2011-01", freq="M")])

        tm.assert_series_equal(result, expected)

    def test_datetime64_tz_dropna(self, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime unit. Naive datetime Series with NaT drops the NaT
        # entries and keeps the labels 0 and 2 and the dtype; the same for a tz-aware
        # (Asia/Tokyo) Series, whose dtype keeps the tz.
        # DatetimeLikeBlock
        ser = Series(
            [
                Timestamp("2011-01-01 10:00"),
                NaT,
                Timestamp("2011-01-03 10:00"),
                NaT,
            ],
            dtype=f"M8[{unit}]",
        )
        result = ser.dropna()
        expected = Series(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-03 10:00")],
            index=[0, 2],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

        # DatetimeTZBlock
        idx = DatetimeIndex(
            ["2011-01-01 10:00", NaT, "2011-01-03 10:00", NaT], tz="Asia/Tokyo"
        ).as_unit(unit)
        ser = Series(idx)
        assert ser.dtype == f"datetime64[{unit}, Asia/Tokyo]"
        result = ser.dropna()
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-03 10:00", tz="Asia/Tokyo"),
            ],
            index=[0, 2],
            dtype=f"datetime64[{unit}, Asia/Tokyo]",
        )
        assert result.dtype == f"datetime64[{unit}, Asia/Tokyo]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [1, 1.5])
    def test_dropna_ignore_index(self, val):
        # [FUNCTION DOC]
        # Regression test (GH#31725), parametrized over an integer or float last value:
        # dropna with the ignore_index option renumbers the result, both when returning a
        # new object and in place.
        # GH#31725
        ser = Series([1, 2, val], index=[3, 2, 1])
        result = ser.dropna(ignore_index=True)
        expected = Series([1, 2, val])
        tm.assert_series_equal(result, expected)

        ser.dropna(ignore_index=True, inplace=True)
        tm.assert_series_equal(ser, expected)
