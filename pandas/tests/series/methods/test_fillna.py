# [MODULE DOC]
# Tests for Series.fillna and the forward/backward fill methods (ffill, bfill):
# scalar, dict, Series and list-like fill values, limits and limit_area,
# datetime/timedelta/period/categorical dtypes with matching or mismatched
# resolution and timezone, in-place operation, float casting, and error cases.
# Uses the frame_or_series, unit and any_real_numpy_dtype fixtures. No
# import-time side effects.

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import period_array


class TestSeriesFillNA:
    # [CLASS DOC]
    # Main group of fillna tests for Series (and sometimes the equivalent frame
    # operation). Stateless; fixtures supply parameters.
    def test_fillna_nat(self):
        # [FUNCTION DOC]
        # Datetime64 Series with a NaT in the last or first position: forward fill and
        # fill with the preceding value agree (and so do backward fill and fill with the
        # following value), both for a Series and a one-column frame.
        series = Series([0, 1, 2, NaT._value], dtype="M8[ns]")

        filled = series.ffill()
        filled2 = series.fillna(value=series.values[2])

        expected = series.copy()
        expected.iloc[3] = expected.iloc[2]

        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)

        df = DataFrame({"A": series})
        filled = df.ffill()
        filled2 = df.fillna(value=series.values[2])
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

        series = Series([NaT._value, 0, 1, 2], dtype="M8[ns]")

        filled = series.bfill()
        filled2 = series.fillna(value=series[1])

        expected = series.copy()
        expected[0] = expected[1]

        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)

        df = DataFrame({"A": series})
        filled = df.bfill()
        filled2 = df.fillna(value=series[1])
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

    def test_fillna(self):
        # [FUNCTION DOC]
        # Float Series with a date index: with no missing values forward fill is a no-op;
        # after setting a middle value to NaN, forward fill uses the previous value,
        # backward fill the next value, and a scalar fill 5.
        ts = Series(
            [0.0, 1.0, 2.0, 3.0, 4.0], index=date_range("2020-01-01", periods=5)
        )

        tm.assert_series_equal(ts, ts.ffill())

        ts.iloc[2] = np.nan

        exp = Series([0.0, 1.0, 1.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.ffill(), exp)

        exp = Series([0.0, 1.0, 3.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.bfill(), exp)

        exp = Series([0.0, 1.0, 5.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(value=5), exp)

    def test_fillna_nonscalar(self):
        # [FUNCTION DOC]
        # Regression test (GH#5703): fillna with non-scalar values. A Series fills
        # aligned on index; an empty dict or empty Series changes nothing; a dict or
        # Series with a matching label fills that position; labels not in the Series are
        # ignored.
        # GH#5703
        s1 = Series([np.nan])
        s2 = Series([1])
        result = s1.fillna(s2)
        expected = Series([1.0])
        tm.assert_series_equal(result, expected)
        result = s1.fillna({})
        tm.assert_series_equal(result, s1)
        result = s1.fillna(Series((), dtype=object))
        tm.assert_series_equal(result, s1)
        result = s2.fillna(s1)
        tm.assert_series_equal(result, s2)
        result = s1.fillna({0: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna({1: 1})
        tm.assert_series_equal(result, Series([np.nan]))
        result = s1.fillna({0: 1, 1: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}))
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
        tm.assert_series_equal(result, s1)

    def test_fillna_aligns(self):
        # [FUNCTION DOC]
        # Filling with a Series aligns on the index labels rather than position.
        s1 = Series([0, 1, 2], list("abc"))
        s2 = Series([0, np.nan, 2], list("bac"))
        result = s2.fillna(s1)
        expected = Series([0, 0, 2.0], list("bac"))
        tm.assert_series_equal(result, expected)

    def test_fillna_limit(self):
        # [FUNCTION DOC]
        # For an all-NaN Series, a limit of 1 fills only the first element and a limit
        # of 2 fills the first two.
        ser = Series(np.nan, index=[0, 1, 2])
        result = ser.fillna(999, limit=1)
        expected = Series([999, np.nan, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        result = ser.fillna(999, limit=2)
        expected = Series([999, 999, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

    def test_fillna_dont_cast_strings(self):
        # [FUNCTION DOC]
        # Regression test (GH#9043): filling a float Series with a numeric-looking
        # string stores the string unconverted, giving object dtype.
        # GH#9043
        # make sure a string representation of int/float values can be filled
        # correctly without raising errors or being converted
        vals = ["0", "1.5", "-0.3"]
        for val in vals:
            ser = Series([0, 1, np.nan, np.nan, 4], dtype="float64")
            result = ser.fillna(val)
            expected = Series([0, 1, val, val, 4], dtype="object")
            tm.assert_series_equal(result, expected)

    def test_fillna_consistency(self):
        # [FUNCTION DOC]
        # Regression test (GH#16402): filling a naive datetime Series with a tz-aware
        # timestamp, or masking in one, gives object dtype holding both kinds; filling
        # with a non-datetime string keeps the datetime and adds the string; assigning
        # the string with setitem raises a type error about an invalid value.
        # KNOWN ISSUE: the call that masks with the tz-aware timestamp and its
        # assertion appear twice in a row; the second copy is redundant.
        # GH#16402
        # fillna with a tz aware to a tz-naive, should result in object

        ser = Series([Timestamp("20130101"), NaT])

        result = ser.fillna(Timestamp("20130101", tz="US/Eastern"))
        expected = Series(
            [Timestamp("20130101"), Timestamp("2013-01-01", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(result, expected)

        result = ser.where([True, False], Timestamp("20130101", tz="US/Eastern"))
        tm.assert_series_equal(result, expected)

        result = ser.where([True, False], Timestamp("20130101", tz="US/Eastern"))
        tm.assert_series_equal(result, expected)

        # with a non-datetime
        result = ser.fillna("foo")
        expected = Series([Timestamp("20130101"), "foo"])
        tm.assert_series_equal(result, expected)

        # assignment
        ser2 = ser.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            ser2[1] = "foo"

    def test_timedelta_fillna(self, frame_or_series, unit):
        # [FUNCTION DOC]
        # Regression test (GH#3371), parametrized over frame_or_series and datetime unit.
        # Builds timedeltas via diff of dates (first value NaT). Fill with a zero or
        # non-zero Timedelta, a standard timedelta, a NumPy timedelta and NaT gives
        # timedelta results in the same unit; an integer fill casts to object
        # (consistent with other dtypes; GH#45746); forward fill and backward fill of an
        # inserted missing value give the neighbouring value.
        # GH#3371
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ],
            dtype=f"M8[{unit}]",
        )
        td = ser.diff()
        obj = frame_or_series(td).copy()

        # reg fillna
        result = obj.fillna(Timedelta(seconds=0))
        expected = Series(
            [
                timedelta(0),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        # GH#45746 pre-1.? ints were interpreted as seconds.  then that was
        #  deprecated and changed to raise. In 2.0 it casts to common dtype,
        #  consistent with every other dtype's behavior
        res = obj.fillna(1)
        expected = obj.astype(object).fillna(1)
        tm.assert_equal(res, expected)

        result = obj.fillna(Timedelta(seconds=1))
        expected = Series(
            [
                timedelta(seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(timedelta(days=1, seconds=1))
        expected = Series(
            [
                timedelta(days=1, seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(np.timedelta64(10**9, "ns"))
        expected = Series(
            [
                timedelta(seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(NaT)
        expected = Series(
            [
                NaT,
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype=f"m8[{unit}]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        # ffill
        td[2] = np.nan
        obj = frame_or_series(td).copy()
        result = obj.ffill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[0] = np.nan
        expected = frame_or_series(expected)

        tm.assert_equal(result, expected)

        # bfill
        td[2] = np.nan
        obj = frame_or_series(td)
        result = obj.bfill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[2] = timedelta(days=1, seconds=9 * 3600 + 60 + 1)
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

    def test_datetime64_fillna(self):
        # [FUNCTION DOC]
        # Datetime Series with a missing value in the third position: forward fill
        # repeats the previous timestamp, backward fill uses the next.
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan

        # ffill
        result = ser.ffill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

        # bfill
        result = ser.bfill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "scalar",
        [
            False,
            True,
        ],
    )
    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_datetime64_fillna_mismatched_reso_no_rounding(self, tz, scalar):
        # [FUNCTION DOC]
        # Regression test (GH#56410), parametrized over timezone and a boolean selecting
        # whether the scalar or Series path is checked. A second-resolution datetime
        # Series with a missing first value, filled with a millisecond-resolution
        # timestamp (scalar or Series), must keep the milliseconds (no rounding), giving
        # a millisecond-resolution result.
        # GH#56410
        dti = date_range("2016-01-01", periods=3, unit="s", tz=tz)
        item = Timestamp("2016-02-03 04:05:06.789", tz=tz).as_unit("ms")
        vec = date_range(item, periods=3, unit="ms")

        exp_dtype = "M8[ms]" if tz is None else "M8[ms, UTC]"
        expected = Series([item, dti[1], dti[2]], dtype=exp_dtype)

        ser = Series(dti)
        ser[0] = NaT
        ser2 = ser.copy()

        res = ser.fillna(item)
        res2 = ser2.fillna(Series(vec))

        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    @pytest.mark.parametrize(
        "scalar",
        [
            False,
            True,
        ],
    )
    def test_timedelta64_fillna_mismatched_reso_no_rounding(self, scalar):
        # [FUNCTION DOC]
        # Regression test (GH#56410), parametrized over scalar or Series fill path: same
        # as the datetime version but for second-resolution timedeltas filled with a
        # millisecond-resolution timedelta; the result has millisecond unit.
        # GH#56410
        ts = Timestamp("1970-01-01").as_unit("s")
        tdi = date_range("2016-01-01", periods=3, unit="s") - ts
        item = Timestamp("2016-02-03 04:05:06.789").as_unit("ms") - ts
        vec = timedelta_range(item, periods=3, unit="ms")

        expected = Series([item, tdi[1], tdi[2]], dtype="m8[ms]")

        ser = Series(tdi)
        ser[0] = NaT
        ser2 = ser.copy()

        res = ser.fillna(item)
        res2 = ser2.fillna(Series(vec))

        if scalar:
            tm.assert_series_equal(res, expected)
        else:
            tm.assert_series_equal(res2, expected)

    def test_datetime64_fillna_backfill(self):
        # [FUNCTION DOC]
        # Regression test (GH#6587): backward fill of leading NaT values uses the
        # following timestamp at full precision (nanosecond), treating values as integers.
        # GH#6587
        # make sure that we are treating as integer when filling
        ser = Series([NaT, NaT, "2013-08-05 15:30:00.000001"], dtype="M8[ns]")

        expected = Series(
            [
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
            ],
            dtype="M8[ns]",
        )
        result = ser.bfill()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_datetime64_tz_fillna(self, tz, unit):
        # [FUNCTION DOC]
        # Parametrized over timezone (US/Eastern, Asia/Tokyo) and datetime unit. For a
        # naive datetime Series with missing values and, then, a tz-aware one: fills with
        # naive or tz-aware timestamp of different resolution, an incompatible string
        # (result object), dicts of timestamps per label, and pre-2.0 mixed-timezone
        # filling (the dtype is now retained by converting to the Series zone). After every
        # fill the original Series still has its original missing positions.
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
        null_loc = Series([False, True, False, True])

        result = ser.fillna(Timestamp("2011-01-02 10:00").as_unit("s"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-02 10:00"),
            ],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(expected, result)
        # check s is not changed
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00").as_unit("s"),
                Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-03 10:00").as_unit("s"),
                Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
            ],
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna("AAA")
        expected = Series(
            [
                Timestamp("2011-01-01 10:00").as_unit("s"),
                "AAA",
                Timestamp("2011-01-03 10:00").as_unit("s"),
                "AAA",
            ],
            dtype=object,
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
                3: Timestamp("2011-01-04 10:00").as_unit("s"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00").as_unit("s"),
                Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-03 10:00").as_unit("s"),
                Timestamp("2011-01-04 10:00").as_unit("s"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00").as_unit("s"),
                3: Timestamp("2011-01-04 10:00").as_unit("s"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-04 10:00"),
            ],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # DatetimeTZBlock
        idx = DatetimeIndex(
            ["2011-01-01 10:00", NaT, "2011-01-03 10:00", NaT], tz=tz
        ).as_unit(unit)
        ser = Series(idx)
        assert ser.dtype == f"datetime64[{unit}, {tz}]"
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00").as_unit("s"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-02 10:00").as_unit("s"),
                Timestamp("2011-01-03 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-02 10:00").as_unit("s"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"))
        idx = DatetimeIndex(
            [
                "2011-01-01 10:00",
                "2011-01-02 10:00",
                "2011-01-03 10:00",
                "2011-01-02 10:00",
            ],
            tz=tz,
        ).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz).to_pydatetime())
        idx = DatetimeIndex(
            [
                "2011-01-01 10:00",
                "2011-01-02 10:00",
                "2011-01-03 10:00",
                "2011-01-02 10:00",
            ],
            tz=tz,
        ).as_unit(unit)
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna("AAA")
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz).as_unit("s"),
                "AAA",
                Timestamp("2011-01-03 10:00", tz=tz).as_unit("s"),
                "AAA",
            ],
            dtype=object,
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
                3: Timestamp("2011-01-04 10:00").as_unit("s"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-03 10:00", tz=tz).as_unit("s"),
                Timestamp("2011-01-04 10:00").as_unit("s"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz).as_unit("s"),
                3: Timestamp("2011-01-04 10:00", tz=tz).as_unit("s"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-04 10:00", tz=tz),
            ]
        ).dt.as_unit(unit)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # filling with a naive/other zone, coerce to object
        result = ser.fillna(Timestamp("20130101").as_unit("s"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz).as_unit("s"),
                Timestamp("2013-01-01").as_unit("s"),
                Timestamp("2011-01-03 10:00", tz=tz).as_unit("s"),
                Timestamp("2013-01-01").as_unit("s"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # pre-2.0 fillna with mixed tzs would cast to object, in 2.0
        #  it retains dtype.
        result = ser.fillna(Timestamp("20130101", tz="US/Pacific").as_unit("s"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz).as_unit("s"),
                Timestamp("2013-01-01", tz="US/Pacific").as_unit("s").tz_convert(tz),
                Timestamp("2011-01-03 10:00", tz=tz).as_unit("s"),
                Timestamp("2013-01-01", tz="US/Pacific").as_unit("s").tz_convert(tz),
            ]
        ).dt.as_unit(unit)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

    def test_fillna_dt64tz_with_method(self):
        # [FUNCTION DOC]
        # Regression test (GH#15855): forward/backward fill on tz-aware timestamps
        # (with offset) fills NaT from the neighbouring value.
        # with timezone
        # GH#15855
        ser = Series([Timestamp("2012-11-11 00:00:00+01:00").as_unit("s"), NaT])
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00").as_unit("s"),
                Timestamp("2012-11-11 00:00:00+01:00").as_unit("s"),
            ]
        )
        tm.assert_series_equal(ser.ffill(), exp)

        ser = Series([NaT, Timestamp("2012-11-11 00:00:00+01:00").as_unit("s")])
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00").as_unit("s"),
                Timestamp("2012-11-11 00:00:00+01:00").as_unit("s"),
            ]
        )
        tm.assert_series_equal(ser.bfill(), exp)

    def test_fillna_pytimedelta(self):
        # [FUNCTION DOC]
        # Regression test (GH#8209): filling a timedelta Series with a standard library
        # timedelta gives microsecond-resolution timedelta values.
        # GH#8209
        ser = Series([np.nan, Timedelta("1 days")], index=["A", "B"])

        result = ser.fillna(timedelta(1))
        expected = Series(Timedelta("1 days"), index=["A", "B"], dtype="m8[us]")
        tm.assert_series_equal(result, expected)

    def test_fillna_period(self):
        # [FUNCTION DOC]
        # Regression test (GH#13737): filling a monthly period Series with a monthly
        # Period replaces NaT and keeps the monthly period dtype.
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        res = ser.fillna(Period("2012-01", freq="M"))
        exp = Series([Period("2011-01", freq="M"), Period("2012-01", freq="M")])
        tm.assert_series_equal(res, exp)
        assert res.dtype == "Period[M]"

    def test_fillna_dt64_timestamp(self, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over Series or frame: filling a datetime Series with a Timestamp
        # replaces the missing entry; filling with NaT leaves the object unchanged.
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan
        obj = frame_or_series(ser)

        # reg fillna
        result = obj.fillna(Timestamp("20130104"))
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130104"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(NaT)
        expected = obj
        tm.assert_equal(result, expected)

    def test_fillna_dt64_non_nao(self):
        # [FUNCTION DOC]
        # Regression test (GH#27419): filling NaT with a millisecond-resolution NumPy
        # datetime64 produces the corresponding timestamp.
        # KNOWN ISSUE: the test name contains a typo ("nao" instead of "nano").
        # GH#27419
        ser = Series([Timestamp("2010-01-01"), NaT, Timestamp("2000-01-01")])
        val = np.datetime64("1975-04-05", "ms")

        result = ser.fillna(val)
        expected = Series(
            [Timestamp("2010-01-01"), Timestamp("1975-04-05"), Timestamp("2000-01-01")]
        )
        tm.assert_series_equal(result, expected)

    def test_fillna_numeric_inplace(self):
        # [FUNCTION DOC]
        # In-place fill with a scalar returns the Series itself and the result equals
        # filling a copy not in place.
        x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        y = x.copy()

        result = y.fillna(value=0, inplace=True)
        assert result is y

        expected = x.fillna(value=0)
        tm.assert_series_equal(y, expected)

    # ---------------------------------------------------------------
    # CategoricalDtype

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            ("a", ["a", "a", "b", "a", "a"]),
            ({1: "a", 3: "b", 4: "b"}, ["a", "a", "b", "b", "b"]),
            ({1: "a"}, ["a", "a", "b", np.nan, np.nan]),
            ({1: "a", 3: "b"}, ["a", "a", "b", "b", np.nan]),
            (Series("a"), ["a", np.nan, "b", np.nan, np.nan]),
            (Series("a", index=[1]), ["a", "a", "b", np.nan, np.nan]),
            (Series({1: "a", 3: "b"}), ["a", "a", "b", "b", np.nan]),
            (Series(["a", "b"], index=[3, 4]), ["a", np.nan, "b", "a", "b"]),
        ],
    )
    def test_fillna_categorical(self, fill_value, expected_output):
        # [FUNCTION DOC]
        # Regression test (GH#17033), parametrized over scalar, dict and Series fill
        # values (various index alignments). Fills a categorical Series with missing
        # values and compares against the expected categorical (same categories).
        # GH#17033
        # Test fillna for a Categorical series
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b"]))
        exp = Series(Categorical(expected_output, categories=["a", "b"]))
        result = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            (["a", "b", "c", "d", "e"], ["a", "b", "b", "d", "e"]),
            (["b", "d", "a", "d", "a"], ["a", "d", "b", "d", "a"]),
            (
                Categorical(
                    ["b", "d", "a", "d", "a"], categories=["b", "c", "d", "e", "a"]
                ),
                ["a", "d", "b", "d", "a"],
            ),
        ],
    )
    def test_fillna_categorical_with_new_categories(self, fill_value, expected_output):
        # [FUNCTION DOC]
        # Regression test (GH#26215), parametrized over fill values given as a list or
        # categorical (converted to a Series): filling a categorical with five categories
        # fills missing entries from the aligned positions of the fill Series.
        # GH#26215
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b", "c", "d", "e"]))
        exp = Series(Categorical(expected_output, categories=["a", "b", "c", "d", "e"]))
        fill_value = Series(fill_value)
        result = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    def test_fillna_categorical_raises(self):
        # [FUNCTION DOC]
        # Error cases for categorical Series: filling with a value that is not a
        # category raises a type error; filling the underlying array with a Series of the
        # wrong length raises a value error; dicts containing a non-category raise a
        # type error; a list, tuple or frame raise type errors stating the value must be
        # a scalar, dict or Series.
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b"]))
        cat = ser._values

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            ser.fillna("d")

        msg2 = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg2):
            cat.fillna(Series("d"))

        with pytest.raises(TypeError, match=msg):
            ser.fillna({1: "d", 3: "a"})

        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(["a", "b"])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(("a", "b"))

        msg = (
            '"value" parameter must be a scalar, dict '
            'or Series, but you passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            ser.fillna(DataFrame({1: ["a"], 3: ["b"]}))

    @pytest.mark.parametrize("dtype", [float, "float32", "float64"])
    @pytest.mark.parametrize("scalar", [True, False])
    def test_fillna_float_casting(self, dtype, any_real_numpy_dtype, scalar):
        # [FUNCTION DOC]
        # Regression test (GH-43424), parametrized over float dtype, real numpy dtype of
        # the fill and whether the fill is scalar or a Series. Float Series with NaN
        # filled with 2 (via fillna, setitem with a mask, in-place mask and where) keeps
        # its float dtype and gives 2.0 and 1.2.
        # GH-43424
        ser = Series([np.nan, 1.2], dtype=dtype)
        fill_values = Series([2, 2], dtype=any_real_numpy_dtype)
        if scalar:
            fill_values = fill_values.dtype.type(2)

        result = ser.fillna(fill_values)
        expected = Series([2.0, 1.2], dtype=dtype)
        tm.assert_series_equal(result, expected)

        ser = Series([np.nan, 1.2], dtype=dtype)
        mask = ser.isna().to_numpy()
        ser[mask] = fill_values
        tm.assert_series_equal(ser, expected)

        ser = Series([np.nan, 1.2], dtype=dtype)
        ser.mask(mask, fill_values, inplace=True)
        tm.assert_series_equal(ser, expected)

        ser = Series([np.nan, 1.2], dtype=dtype)
        res = ser.where(~mask, fill_values)
        tm.assert_series_equal(res, expected)

    def test_fillna_f32_upcast_with_dict(self):
        # [FUNCTION DOC]
        # Regression test (GH-43424): filling float32 with a dict keyed by label keeps
        # the float32 dtype.
        # GH-43424
        ser = Series([np.nan, 1.2], dtype=np.float32)
        result = ser.fillna({0: 1})
        expected = Series([1.0, 1.2], dtype=np.float32)
        tm.assert_series_equal(result, expected)

    # ---------------------------------------------------------------
    # Invalid Usages

    def test_fillna_listlike_invalid(self):
        # [FUNCTION DOC]
        # Passing a list or tuple as the fill value raises a type error saying the
        # value must be a scalar or dict.
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna([1, 2])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna((1, 2))

    def test_fillna_method_and_limit_invalid(self):
        # [FUNCTION DOC]
        # A limit that is not a positive integer (negative, zero or a float) raises a
        # value error, with a message about the limit (or about specifying both value
        # and method).
        # KNOWN ISSUE: the name mentions method but no method argument is passed, and
        # the float limits 1.0 and 2.0 are expected to be rejected as non-integers.
        # related GH#9217, make sure limit is an int and greater than 0
        ser = Series([1, 2, 3, None])
        msg = "|".join(
            [
                r"Cannot specify both 'value' and 'method'\.",
                "Limit must be greater than 0",
                "Limit must be an integer",
            ]
        )
        for limit in [-1, 0, 1.0, 2.0]:
            with pytest.raises(ValueError, match=msg):
                ser.fillna(1, limit=limit)

    def test_fillna_datetime64_with_timezone_tzinfo(self):
        # [FUNCTION DOC]
        # Regression test (GitHub issue 38851): different tzinfo objects representing UTC
        # are treated as equal when filling; filling a dateutil-US/Pacific Series with a
        # pytz-style US/Pacific timestamp keeps the Series dtype (pre-2.0 this cast to
        # object).
        # https://github.com/pandas-dev/pandas/issues/38851
        # different tzinfos representing UTC treated as equal
        ser = Series(date_range("2020", periods=3, tz="UTC"))
        expected = ser.copy()
        ser[1] = NaT
        result = ser.fillna(datetime(2020, 1, 2, tzinfo=timezone.utc))
        tm.assert_series_equal(result, expected)

        # pre-2.0 we cast to object with mixed tzs, in 2.0 we retain dtype
        ts = Timestamp("2000-01-01", tz="US/Pacific")
        ser2 = Series(ser._values.tz_convert("dateutil/US/Pacific"))
        assert ser2.dtype.kind == "M"
        result = ser2.fillna(ts)
        expected = Series(
            [ser2[0], ts.tz_convert(ser2.dtype.tz), ser2[2]],
            dtype=ser2.dtype,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input, input_fillna, expected_data, expected_categories",
        [
            (["A", "B", None, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
            (["A", "B", np.nan, "A"], "B", ["A", "B", "B", "A"], ["A", "B"]),
        ],
    )
    def test_fillna_categorical_accept_same_type(
        self, input, input_fillna, expected_data, expected_categories
    ):
        # [FUNCTION DOC]
        # Regression test (GH32414), parametrized over None or NaN as the missing marker:
        # filling a categorical with a categorical-valued Series built from the same
        # data works and keeps the categories.
        # GH32414
        cat = Categorical(input)
        ser = Series(cat).fillna(input_fillna)
        filled = cat.fillna(ser)
        result = cat.fillna(filled)
        expected = Categorical(expected_data, categories=expected_categories)
        tm.assert_categorical_equal(result, expected)


class TestFillnaPad:
    # [CLASS DOC]
    # Tests of ffill and bfill (formerly "pad"/"backfill") behavior, limits, in-place
    # use, and edge dtypes. Stateless.
    def test_fillna_bug(self):
        # [FUNCTION DOC]
        # Forward fill propagates the last valid value (leading NaN stays); backward fill
        # propagates the next valid value (trailing NaN stays).
        ser = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        filled = ser.ffill()
        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ser.index)
        tm.assert_series_equal(filled, expected)

        filled = ser.bfill()
        expected = Series([1.0, 1.0, 3.0, 3.0, np.nan], ser.index)
        tm.assert_series_equal(filled, expected)

    def test_ffill_mixed_dtypes_without_missing_data(self):
        # [FUNCTION DOC]
        # Regression test (GH#14956): forward fill of an object Series with mixed types
        # and no missing data returns an equal Series.
        # GH#14956
        series = Series([datetime(2015, 1, 1, tzinfo=timezone.utc), 1])
        result = series.ffill()
        tm.assert_series_equal(series, result)

    def test_pad_nan(self):
        # [FUNCTION DOC]
        # In-place forward fill returns the Series itself; all but the first value
        # equal the expected values and the first stays NaN.
        # KNOWN ISSUE: the final assertion uses a comma, so its second expression is
        # only the failure message and the expected value is never compared.
        x = Series(
            [np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"], dtype=float
        )

        result = x.ffill(inplace=True)
        assert result is x

        expected = Series(
            [np.nan, 1.0, 1.0, 3.0, 3.0], ["z", "a", "b", "c", "d"], dtype=float
        )
        tm.assert_series_equal(x[1:], expected[1:])
        assert np.isnan(x.iloc[0]), np.isnan(expected.iloc[0])

    def test_series_fillna_limit(self):
        # [FUNCTION DOC]
        # With ten elements of which only the first two (or last two) are non-missing,
        # forward fill (or backward fill) with limit 5 fills only five positions, leaving
        # the remaining as NaN.
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)

        result = s[:2].reindex(index)
        result = result.ffill(limit=5)

        expected = s[:2].reindex(index).ffill()
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        result = s[-2:].reindex(index)
        result = result.bfill(limit=5)

        expected = s[-2:].reindex(index).bfill()
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_series_pad_backfill_limit(self):
        # [FUNCTION DOC]
        # Same as the limit test but through reindex with the pad or backfill method
        # and a limit.
        index = np.arange(10)
        s = Series(np.random.default_rng(2).standard_normal(10), index=index)

        result = s[:2].reindex(index, method="pad", limit=5)

        expected = s[:2].reindex(index).ffill()
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        result = s[-2:].reindex(index, method="backfill", limit=5)

        expected = s[-2:].reindex(index).bfill()
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_fillna_int(self):
        # [FUNCTION DOC]
        # Forward fill on an integer Series in place returns the Series; the non-in-place
        # fill returns an equal Series.
        ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
        result = ser.ffill(inplace=True)
        assert result is ser
        tm.assert_series_equal(ser.ffill(inplace=False), ser)

    def test_datetime64tz_fillna_round_issue(self):
        # [FUNCTION DOC]
        # Regression test (GH#14872): backward fill of NaT preceding a tz-aware datetime
        # with microsecond precision keeps the exact value (no rounding).
        # GH#14872

        data = Series(
            [NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc)]
        )

        filled = data.bfill()

        expected = Series(
            [
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=timezone.utc),
            ]
        )

        tm.assert_series_equal(filled, expected)

    def test_fillna_parr(self):
        # [FUNCTION DOC]
        # Regression test (GH-24537): forward fill of a nanosecond period Series near the
        # maximum timestamp fills the missing period with the preceding one.
        # GH-24537
        dti = date_range(
            Timestamp.max - Timedelta(nanoseconds=10), periods=5, freq="ns"
        )
        ser = Series(dti.to_period("ns"))
        ser[2] = NaT
        arr = period_array(
            [
                Timestamp("2262-04-11 23:47:16.854775797"),
                Timestamp("2262-04-11 23:47:16.854775798"),
                Timestamp("2262-04-11 23:47:16.854775798"),
                Timestamp("2262-04-11 23:47:16.854775800"),
                Timestamp("2262-04-11 23:47:16.854775801"),
            ],
            freq="ns",
        )
        expected = Series(arr)

        filled = ser.ffill()

        tm.assert_series_equal(filled, expected)


@pytest.mark.parametrize(
    "data, expected_data, method, kwargs",
    (
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
            "ffill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            range(5),
            range(5),
            "ffill",
            {"limit_area": "outside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside", "limit": 1},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside"},
        ),
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside", "limit": 1},
        ),
    ),
)
def test_ffill_bfill_limit_area(data, expected_data, method, kwargs):
    # [FUNCTION DOC]
    # Regression test (GH#56492), parametrized over data, expected values, the method
    # name (ffill or bfill) and keyword arguments with limit_area (inside or
    # outside) and optional limit. Calls the named method on the Series with the
    # keywords and expects the result: "inside" fills only gaps between valid values,
    # "outside" fills only leading (backward) or trailing (forward) gaps; the limit
    # caps fill length per gap; all-NaN and fully valid data stay unchanged.
    # GH#56492
    s = Series(data)
    expected = Series(expected_data)
    result = getattr(s, method)(**kwargs)
    tm.assert_series_equal(result, expected)
