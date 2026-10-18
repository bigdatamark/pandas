# [MODULE DOC]
# Tests for Series.clip: thresholds given as scalars, Series, lists and arrays,
# NaN thresholds, extension dtypes with NA, in-place operation, and datetime
# data including out-of-bounds timestamps. Uses the datetime_series and
# any_numeric_ea_dtype fixtures. No import-time side effects.

from datetime import datetime

import numpy as np
import pytest

from pandas.errors import OutOfBoundsDatetime

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    isna,
    notna,
)
import pandas._testing as tm


class TestSeriesClip:
    # [CLASS DOC]
    # Groups the tests of Series.clip. Stateless.
    def test_clip(self, datetime_series):
        # [FUNCTION DOC]
        # Clipping the datetime_series fixture at its median from below gives a minimum
        # equal to the median, from above a maximum equal to the median, and clipping to
        # the range minus 0.5 to 0.5 equals the NumPy clip function applied to the Series
        # (and still returns a Series).
        val = datetime_series.median()

        assert datetime_series.clip(lower=val).min() == val
        assert datetime_series.clip(upper=val).max() == val

        result = datetime_series.clip(-0.5, 0.5)
        expected = np.clip(datetime_series, -0.5, 0.5)
        tm.assert_series_equal(result, expected)
        assert isinstance(expected, Series)

    def test_clip_types_and_nulls(self):
        # [FUNCTION DOC]
        # For float, object (with None and strings) and datetime Series with a leading
        # missing value: clipping from below and from above at the third element gives
        # non-missing minimum or maximum equal to that element, and leaves the missing
        # value positions unchanged.
        sers = [
            Series([np.nan, 1.0, 2.0, 3.0]),
            Series([None, "a", "b", "c"]),
            Series(pd.to_datetime([np.nan, 1, 2, 3], unit="D")),
        ]

        for s in sers:
            thresh = s[2]
            lower = s.clip(lower=thresh)
            upper = s.clip(upper=thresh)
            assert lower[notna(lower)].min() == thresh
            assert upper[notna(upper)].max() == thresh
            assert list(isna(s)) == list(isna(lower))
            assert list(isna(s)) == list(isna(upper))

    def test_series_clipping_with_na_values(self, any_numeric_ea_dtype):
        # [FUNCTION DOC]
        # Parametrized over nullable numeric dtypes (GH#40581): clipping a Series with a
        # leading NA from above and from below works and keeps NA in place, preserving
        # the dtype.
        # Ensure that clipping method can handle NA values with out failing
        # GH#40581

        ser = Series([pd.NA, 1.0, 3.0], dtype=any_numeric_ea_dtype)
        s_clipped_upper = ser.clip(upper=2.0)
        s_clipped_lower = ser.clip(lower=2.0)

        expected_upper = Series([pd.NA, 1.0, 2.0], dtype=any_numeric_ea_dtype)
        expected_lower = Series([pd.NA, 2.0, 3.0], dtype=any_numeric_ea_dtype)

        tm.assert_series_equal(s_clipped_upper, expected_upper)
        tm.assert_series_equal(s_clipped_lower, expected_lower)

    def test_clip_with_na_args(self):
        # [FUNCTION DOC]
        # NaN thresholds are treated as no threshold: scalar and keyword NaN leave the
        # Series unchanged (GH#17276); a list-like threshold with NaN entries only clips
        # the other positions and promotes to float (GH#19992); a list of all NaN as upper
        # bound with a scalar lower leaves the Series equal (GH#40420).
        """Should process np.nan argument as None"""
        # GH#17276
        s = Series([1, 2, 3])

        tm.assert_series_equal(s.clip(np.nan), Series([1, 2, 3]))
        tm.assert_series_equal(s.clip(upper=np.nan, lower=np.nan), Series([1, 2, 3]))

        # GH#19992

        res = s.clip(lower=[0, 4, np.nan])
        tm.assert_series_equal(res, Series([1, 4, 3.0]))
        res = s.clip(upper=[1, np.nan, 1])
        tm.assert_series_equal(res, Series([1, 2, 1.0]))

        # GH#40420
        s = Series([1, 2, 3])
        result = s.clip(0, [np.nan, np.nan, np.nan])
        tm.assert_series_equal(s, result)

    def test_clip_against_series(self):
        # [FUNCTION DOC]
        # Regression test (GH#6966): clipping to bounds given as Series works elementwise,
        # including a scalar lower bound with a Series upper bound.
        # GH#6966

        s = Series([1.0, 1.0, 4.0])

        lower = Series([1.0, 2.0, 3.0])
        upper = Series([1.5, 2.5, 3.5])

        tm.assert_series_equal(s.clip(lower, upper), Series([1.0, 2.0, 3.5]))
        tm.assert_series_equal(s.clip(1.5, upper), Series([1.5, 1.5, 3.5]))

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("upper", [[1, 2, 3], np.asarray([1, 2, 3])])
    def test_clip_against_list_like(self, inplace, upper):
        # [FUNCTION DOC]
        # Parametrized over in-place or not, and over list or NumPy array upper bound
        # (GH#15390). The upper bound is applied elementwise on an integer Series;
        # in the in-place case the returned object is the original Series object. The
        # result must equal the bounds exactly.
        # GH#15390
        original = Series([5, 6, 7])
        result = original.clip(upper=upper, inplace=inplace)
        expected = Series([1, 2, 3])

        if inplace:
            assert result is original
        tm.assert_series_equal(result, expected, check_exact=True)

    def test_clip_with_datetimes(self):
        # [FUNCTION DOC]
        # Regression test (GH#11838): clipping from above with a Timestamp works for
        # naive datetimes and for tz-aware (US/Eastern) datetimes, replacing later values
        # with the threshold.
        # GH#11838
        # naive and tz-aware datetimes

        t = Timestamp("2015-12-01 09:30:30")
        s = Series([Timestamp("2015-12-01 09:30:00"), Timestamp("2015-12-01 09:31:00")])
        result = s.clip(upper=t)
        expected = Series(
            [Timestamp("2015-12-01 09:30:00"), Timestamp("2015-12-01 09:30:30")]
        )
        tm.assert_series_equal(result, expected)

        t = Timestamp("2015-12-01 09:30:30", tz="US/Eastern")
        s = Series(
            [
                Timestamp("2015-12-01 09:30:00", tz="US/Eastern"),
                Timestamp("2015-12-01 09:31:00", tz="US/Eastern"),
            ]
        )
        result = s.clip(upper=t)
        expected = Series(
            [
                Timestamp("2015-12-01 09:30:00", tz="US/Eastern"),
                Timestamp("2015-12-01 09:30:30", tz="US/Eastern"),
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_clip_with_timestamps_and_oob_datetimes_object(self):
        # [FUNCTION DOC]
        # Regression test (GH-42794): clipping an object Series of datetimes outside the
        # nanosecond range with the minimum and maximum Timestamp gives an object Series
        # of those two timestamps.
        # GH-42794
        ser = Series([datetime(1, 1, 1), datetime(9999, 9, 9)], dtype=object)

        result = ser.clip(lower=Timestamp.min, upper=Timestamp.max)
        expected = Series([Timestamp.min, Timestamp.max], dtype=object)

        tm.assert_series_equal(result, expected)

    def test_clip_with_timestamps_and_oob_datetimes_non_nano(self):
        # [FUNCTION DOC]
        # Regression test (GH#56410): for a microsecond-resolution Series, clipping with
        # nanosecond-resolution Timestamp bounds raises an out-of-bounds datetime error
        # telling the user to cast explicitly; clipping with the bounds converted to
        # microseconds returns a Series of the bounds.
        # GH#56410
        dtype = "M8[us]"
        ser = Series([datetime(1, 1, 1), datetime(9999, 9, 9)], dtype=dtype)

        msg = (
            r"Incompatible \(high-resolution\) value for dtype='datetime64\[us\]'. "
            "Explicitly cast before operating"
        )
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ser.clip(lower=Timestamp.min, upper=Timestamp.max)

        lower = Timestamp.min.as_unit("us")
        upper = Timestamp.max.as_unit("us")
        result = ser.clip(lower=lower, upper=upper)
        expected = Series([lower, upper], dtype=dtype)

        tm.assert_series_equal(result, expected)
