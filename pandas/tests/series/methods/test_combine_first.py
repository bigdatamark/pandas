# [MODULE DOC]
# Tests for Series.combine_first, which fills missing values from another Series
# and takes the union of the indexes. Covers period and datetime indexes, name
# propagation, mixed dtypes, datetime resolutions and time zones, an empty
# argument, dtype preservation and None values. Uses the datetime_series, unit,
# tz_naive_fixture and using_infer_string fixtures. No import-time side effects.

from datetime import datetime

import numpy as np

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    Period,
    Series,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


class TestCombineFirst:
    # [CLASS DOC]
    # Groups the tests of Series.combine_first. Stateless.
    def test_combine_first_period_datetime(self):
        # [FUNCTION DOC]
        # Regression test (GH#3367): for both a datetime and a period monthly index,
        # missing entries of the first Series are filled from the second, producing a
        # float Series on the same index.
        # GH#3367
        didx = date_range(start="1950-01-31", end="1950-07-31", freq="ME")
        pidx = period_range(start=Period("1950-1"), end=Period("1950-7"), freq="M")
        # check to be consistent with DatetimeIndex
        for idx in [didx, pidx]:
            a = Series([1, np.nan, np.nan, 4, 5, np.nan, 7], index=idx)
            b = Series([9, 9, 9, 9, 9, 9, 9], index=idx)

            result = a.combine_first(b)
            expected = Series([1, 9, 9, 4, 5, 9, 7], index=idx, dtype=np.float64)
            tm.assert_series_equal(result, expected)

    def test_combine_first_name(self, datetime_series):
        # [FUNCTION DOC]
        # The result keeps the name of the calling Series.
        result = datetime_series.combine_first(datetime_series[:5])
        assert result.name == datetime_series.name

    def test_combine_first(self, using_infer_string):
        # [FUNCTION DOC]
        # Float Series: combining with an alternative that has NaN at every other
        # position changes nothing; the reverse fills holes from the other values. With
        # string and float Series on a string index, the result keeps strings where
        # present and the float values (as objects) elsewhere. Combining with an empty
        # Series returns the original as object dtype, with the index also cast to object
        # unless the inferred-string option is on.
        values = np.arange(20, dtype=np.float64)
        series = Series(values, index=np.arange(20, dtype=np.int64))

        series_copy = series * 2
        series_copy[::2] = np.nan

        # nothing used from the input
        combined = series.combine_first(series_copy)

        tm.assert_series_equal(combined, series)

        # Holes filled from input
        combined = series_copy.combine_first(series)
        assert np.isfinite(combined).all()

        tm.assert_series_equal(combined[::2], series[::2])
        tm.assert_series_equal(combined[1::2], series_copy[1::2])

        # mixed types
        index = pd.Index([str(i) for i in range(20)])
        floats = Series(np.random.default_rng(2).standard_normal(20), index=index)
        strings = Series([str(i) for i in range(10)], index=index[::2], dtype=object)

        combined = strings.combine_first(floats)

        tm.assert_series_equal(strings, combined.loc[index[::2]])
        tm.assert_series_equal(floats[1::2].astype(object), combined.loc[index[1::2]])

        # corner case
        ser = Series([1.0, 2, 3], index=[0, 1, 2])
        empty = Series([], index=[], dtype=object)
        result = ser.combine_first(empty)
        if not using_infer_string:
            ser.index = ser.index.astype("O")
        tm.assert_series_equal(result, ser.astype(object))

    def test_combine_first_dt64(self, unit):
        # [FUNCTION DOC]
        # Parametrized over datetime resolution: datetime Series with complementary
        # missing values combine into the full Series, keeping the unit.
        s0 = to_datetime(Series(["2010", np.nan])).dt.as_unit(unit)
        s1 = to_datetime(Series([np.nan, "2011"])).dt.as_unit(unit)
        rs = s0.combine_first(s1)
        xp = to_datetime(Series(["2010", "2011"])).dt.as_unit(unit)
        tm.assert_series_equal(rs, xp)

    def test_combine_first_dt64_casting_deprecation(self, unit):
        # [FUNCTION DOC]
        # GH#62931: combining a datetime Series with a non-datetime Series emits a
        # Pandas4Warning ("Silently casting non-datetime 'other' to datetime") and gives a
        # datetime Series of the same unit; for second and millisecond units the
        # expected result is converted to microseconds.
        # KNOWN ISSUE: a TODO in the test notes it is unclear whether the cast should
        # preserve the second/millisecond unit; the expected result encodes the
        # current promotion to microseconds.
        # GH#62931
        s0 = to_datetime(Series(["2010", np.nan])).dt.as_unit(unit)
        s1 = Series([np.nan, "2011"])

        msg = "Silently casting non-datetime 'other' to datetime"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            rs = s0.combine_first(s1)

        xp = Series([datetime(2010, 1, 1), "2011"], dtype=f"datetime64[{unit}]")
        if unit in ["s", "ms"]:
            # TODO: should _cast_pointwise_result attempt to preserve unit?
            xp = xp.dt.as_unit("us")
        tm.assert_series_equal(rs, xp)

    def test_combine_first_dt_tz_values(self, tz_naive_fixture):
        # [FUNCTION DOC]
        # Tz-aware (parametrized over the timezone fixture) datetime Series with
        # different indexes: the result is the union, using the first Series' values
        # and the second's for new labels, with the first Series' name.
        ser1 = Series(
            pd.DatetimeIndex(["20150101", "20150102", "20150103"], tz=tz_naive_fixture),
            name="ser1",
        )
        ser2 = Series(
            pd.DatetimeIndex(["20160514", "20160515", "20160516"], tz=tz_naive_fixture),
            index=[2, 3, 4],
            name="ser2",
        )
        result = ser1.combine_first(ser2)
        exp_vals = pd.DatetimeIndex(
            ["20150101", "20150102", "20150103", "20160515", "20160516"],
            tz=tz_naive_fixture,
        )
        exp = Series(exp_vals, name="ser1")
        tm.assert_series_equal(exp, result)

    def test_combine_first_timezone_series_with_empty_series(self):
        # [FUNCTION DOC]
        # Regression test (GH 41800): combining a tz-aware Series with an all-NaN empty
        # Series on the same index gives the first Series cast to float.
        # GH 41800
        time_index = date_range(
            datetime(2021, 1, 1, 1),
            datetime(2021, 1, 1, 10),
            freq="h",
            tz="Europe/Rome",
        )
        s1 = Series(range(10), index=time_index)
        s2 = Series(index=time_index)
        result = s1.combine_first(s2)
        tm.assert_series_equal(result, s1.astype(np.float64))

    def test_combine_first_preserves_dtype(self):
        # [FUNCTION DOC]
        # Regression test (GH51764): large integers beyond float precision keep integer
        # dtype and exact values when combined with a longer integer Series.
        # GH51764
        s1 = Series([1666880195890293744, 1666880195890293837])
        s2 = Series([1, 2, 3])
        result = s1.combine_first(s2)
        expected = Series([1666880195890293744, 1666880195890293837, 3])
        tm.assert_series_equal(result, expected)

    def test_combine_mixed_timezone(self):
        # [FUNCTION DOC]
        # Regression test (GH 26283): combining Series with UTC and mixed-timezone
        # timestamp indexes gives a Series with an object-dtype index of UTC timestamps
        # and values taken from the first where labels overlap.
        # GH 26283
        uniform_tz = Series({pd.Timestamp("2019-05-01", tz="UTC"): 1.0})
        multi_tz = Series(
            {
                pd.Timestamp("2019-05-01 01:00:00+0100", tz="Europe/London"): 2.0,
                pd.Timestamp("2019-05-02", tz="UTC"): 3.0,
            }
        )

        result = uniform_tz.combine_first(multi_tz)
        expected = Series(
            [1.0, 3.0],
            index=pd.Index(
                [
                    pd.Timestamp("2019-05-01 00:00:00+00:00", tz="UTC"),
                    pd.Timestamp("2019-05-02 00:00:00+00:00", tz="UTC"),
                ],
                dtype="object",
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_combine_first_none_not_nan(self):
        # [FUNCTION DOC]
        # Regression test (GH#58977): all-None Series with partially overlapping
        # indexes combine into an object Series of four None values (None stays None, not
        # converted to NaN).
        # GH#58977
        s1 = Series([None, None, None], index=["a", "b", "c"])
        s2 = Series([None, None, None], index=["b", "c", "d"])

        result = s1.combine_first(s2)
        expected = Series([None] * 4, index=["a", "b", "c", "d"])
        tm.assert_series_equal(result, expected)
