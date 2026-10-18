# [MODULE DOC]
# Tests for concatenating datetime-like data: naive and tz-aware datetimes, NaT
# handling, period data, timedeltas and mixed datetime-like/other dtypes, for
# series and frames, including frequency, multi-level index and resolution (unit)
# handling. Three test classes group datetime, timezone and period cases, followed
# by module-level tests for timedelta blocks, NaT in multi-level indexes and
# float/datetime mixes. Uses the unit and unit2 fixtures (datetime resolutions)
# from conftest.
import datetime as dt
from datetime import datetime

import dateutil
import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    to_timedelta,
)
import pandas._testing as tm


class TestDatetimeConcat:
    # [CLASS DOC]
    # Test collection for naive datetime and NaT behavior in concatenation.
    # Stateless.
    def test_concat_datetime64_block(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a frame holding a datetime column with itself keeps
        # the values in both halves.
        rng = date_range("1/1/2000", periods=10)

        df = DataFrame({"time": rng})

        result = concat([df, df])
        assert (result.iloc[:10]["time"] == rng).all()
        assert (result.iloc[10:]["time"] == rng).all()

    def test_concat_datetime_datetime64_frame(self):
        # [FUNCTION DOC]
        # Smoke test: concatenating a frame with a datetime column with one built from
        # mixed object records must not raise (GH 2624).
        # GH#2624
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])

        df2_obj = DataFrame.from_records(rows, columns=["date", "test"])

        ind = date_range(start="2000/1/1", freq="D", periods=10)
        df1 = DataFrame({"date": ind, "test": range(10)})

        # it works!
        concat([df1, df2_obj])

    def test_concat_datetime_timezone(self):
        # [FUNCTION DOC]
        # Verifies concatenation along columns of frames with tz-aware indexes: same zone
        # (the naive-built second index inherits the zone) aligns and keeps frequency;
        # different zones (Paris, Tokyo) union to a UTC index and emit a Pandas4Warning
        # about default sorting; and after resampling, row-wise concatenation with sorting
        # gives the combined index (GH 18523, GH 13783).
        # GH 18523
        idx1 = date_range(
            "2011-01-01", periods=3, freq="h", tz="Europe/Paris", unit="ns"
        )
        idx2 = date_range(start=idx1[0], end=idx1[-1], freq="h", unit="ns")
        df1 = DataFrame({"a": [1, 2, 3]}, index=idx1)
        df2 = DataFrame({"b": [1, 2, 3]}, index=idx2)
        result = concat([df1, df2], axis=1)

        exp_idx = DatetimeIndex(
            [
                "2011-01-01 00:00:00+01:00",
                "2011-01-01 01:00:00+01:00",
                "2011-01-01 02:00:00+01:00",
            ],
            dtype="M8[ns, Europe/Paris]",
            freq="h",
        )
        expected = DataFrame(
            [[1, 1], [2, 2], [3, 3]], index=exp_idx, columns=["a", "b"]
        )

        tm.assert_frame_equal(result, expected)

        idx3 = date_range("2011-01-01", periods=3, freq="h", tz="Asia/Tokyo", unit="ns")
        df3 = DataFrame({"b": [1, 2, 3]}, index=idx3)
        msg = "Sorting by default when concatenating all DatetimeIndex"
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = concat([df1, df3], axis=1)

        exp_idx = DatetimeIndex(
            [
                "2010-12-31 15:00:00+00:00",
                "2010-12-31 16:00:00+00:00",
                "2010-12-31 17:00:00+00:00",
                "2010-12-31 23:00:00+00:00",
                "2011-01-01 00:00:00+00:00",
                "2011-01-01 01:00:00+00:00",
            ]
        ).as_unit("ns")

        expected = DataFrame(
            [
                [np.nan, 1],
                [np.nan, 2],
                [np.nan, 3],
                [1, np.nan],
                [2, np.nan],
                [3, np.nan],
            ],
            index=exp_idx,
            columns=["a", "b"],
        )

        tm.assert_frame_equal(result, expected)

        # GH 13783: Concat after resample
        result = concat([df1.resample("h").mean(), df2.resample("h").mean()], sort=True)
        expected = DataFrame(
            {"a": [1, 2, 3] + [np.nan] * 3, "b": [np.nan] * 3 + [1, 2, 3]},
            index=idx1.append(idx1),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_datetimeindex_freq(self):
        # [FUNCTION DOC]
        # Verifies that concatenating consecutive slices keeps a frequency-bearing
        # tz-aware index equal to the original, while concatenating the slices in reversed
        # order gives a non-monotonic index without frequency (GH 3232).
        # GH 3232
        # Monotonic index result
        dr = date_range("01-Jan-2013", periods=100, freq="50ms", tz="UTC")
        data = list(range(100))
        expected = DataFrame(data, index=dr)
        result = concat([expected[:50], expected[50:]])
        tm.assert_frame_equal(result, expected)

        # Non-monotonic index result
        result = concat([expected[50:], expected[:50]])
        expected = DataFrame(data[50:] + data[:50], index=dr[50:].append(dr[:50]))
        expected.index._data.freq = None
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_datetime_object_index(self):
        # [FUNCTION DOC]
        # Verifies column-wise concatenation of series with multi-level indexes whose
        # second level is an object index of date objects keeps an object level and
        # aligns the rows (GH 11058).
        # https://github.com/pandas-dev/pandas/issues/11058
        idx = Index(
            [dt.date(2013, 1, 1), dt.date(2014, 1, 1), dt.date(2015, 1, 1)],
            dtype="object",
        )

        s = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [
                    [1, 2],
                    idx[:-1],
                ],
                names=["first", "second"],
            ),
        )
        s2 = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [[1, 2], idx[::2]],
                names=["first", "second"],
            ),
        )
        mi = MultiIndex.from_arrays(
            [[1, 2, 2], idx],
            names=["first", "second"],
        )
        assert mi.levels[1].dtype == object

        expected = DataFrame(
            [["a", "a"], ["b", np.nan], [np.nan, "b"]],
            index=mi,
        )
        result = concat([s, s2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_NaT_series(self):
        # [FUNCTION DOC]
        # Verifies concatenating a tz-aware datetime series with an all-NaT series of the
        # same dtype, and two all-NaT series, keeps the tz-aware datetime dtype (GH 11693).
        # GH 11693
        # test for merging NaT series with datetime series.
        x = Series(
            date_range(
                "20151124 08:00",
                "20151124 09:00",
                freq="1h",
                tz="US/Eastern",
                unit="ns",
            )
        )
        y = Series(pd.NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])

        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # all NaT with tz
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns, US/Eastern]")
        result = concat([y, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_NaT_series2(self):
        # [FUNCTION DOC]
        # Verifies the same for tz-naive data: datetime with an all-NaT series, then two
        # all-NaT series giving a naive datetime dtype.
        # without tz
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", unit="ns"))
        y = Series(date_range("20151124 10:00", "20151124 11:00", freq="1h", unit="ns"))
        y[:] = pd.NaT
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # all NaT without tz
        x[:] = pd.NaT
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_concat_NaT_dataframes(self, tz):
        # [FUNCTION DOC]
        # Parametrized on timezone (none or UTC). Verifies concatenating a frame of NaT
        # with a frame of timestamps along rows keeps the datetime dtype (GH 12396).
        # GH 12396

        dti = DatetimeIndex([pd.NaT, pd.NaT], tz=tz)
        first = DataFrame({0: dti})
        second = DataFrame(
            [[Timestamp("2015/01/01", tz=tz)], [Timestamp("2016/01/01", tz=tz)]],
            index=[2, 3],
        )
        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz),
                Timestamp("2016/01/01", tz=tz),
            ]
        )

        result = concat([first, second], axis=0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    @pytest.mark.parametrize("item", [pd.NaT, Timestamp("20150101").as_unit("ns")])
    def test_concat_NaT_dataframes_all_NaT_axis_0(self, tz1, tz2, item):
        # [FUNCTION DOC]
        # Parametrized over timezones of both frames and the second item (NaT or a
        # timestamp). Verifies row concatenation of an all-NaT frame with another frame:
        # result takes the second frame's timezone, or object dtype when the timezones
        # differ (GH 12396).
        # GH 12396

        # tz-naive
        first = DataFrame([[pd.NaT], [pd.NaT]]).apply(lambda x: x.dt.tz_localize(tz1))
        second = DataFrame([item]).apply(lambda x: x.dt.tz_localize(tz2))

        result = concat([first, second], axis=0)
        expected = DataFrame(Series([pd.NaT, pd.NaT, item], index=[0, 1, 0]))
        expected = expected.apply(lambda x: x.dt.tz_localize(tz2))
        if tz1 != tz2:
            expected = expected.astype(object)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_dataframes_all_NaT_axis_1(self, tz1, tz2):
        # [FUNCTION DOC]
        # Parametrized over two timezones. Verifies column-wise concatenation of all-NaT
        # frames keeps each column's own timezone-aware or naive dtype (GH 12396).
        # GH 12396

        first = DataFrame(Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1))
        second = DataFrame(Series([pd.NaT]).dt.tz_localize(tz2), columns=[1])
        expected = DataFrame(
            {
                0: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1),
                1: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz2),
            }
        )
        result = concat([first, second], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_series_dataframe_all_NaT(self, tz1, tz2):
        # [FUNCTION DOC]
        # Parametrized over two timezones. Verifies concatenating an all-NaT series with a
        # timestamp frame gives datetimes if timezones match, otherwise object (GH 12396).
        # GH 12396

        # tz-naive
        first = Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1)
        second = DataFrame(
            [
                [Timestamp("2015/01/01", tz=tz2)],
                [Timestamp("2016/01/01", tz=tz2)],
            ],
            index=[2, 3],
        )

        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz2),
                Timestamp("2016/01/01", tz=tz2),
            ]
        )
        if tz1 != tz2:
            expected = expected.astype(object)

        result = concat([first, second])
        tm.assert_frame_equal(result, expected)

    def test_concat_compat_on_non_ns_datetime_EA(self):
        # [FUNCTION DOC]
        # Verifies concatenating a day-resolution datetime series with a categorical string
        # series gives an object series with Timestamp and strings (GH 33331).
        # GH#33331
        first = Series(np.array([datetime(2010, 1, 1)], dtype="datetime64[D]"))
        second = Series(pd.array(["a", "b"], dtype="category"))

        expected = Series([Timestamp("2010-01-01 00:00:00"), "a", "b"])

        result = concat([first, second], ignore_index=True)
        tm.assert_series_equal(result, expected)


class TestTimezoneConcat:
    # [CLASS DOC]
    # Test collection for concatenation of timezone-aware data. Stateless.
    def test_concat_tz_series(self):
        # [FUNCTION DOC]
        # Verifies tz-aware plus naive datetime series concatenation yields object dtype
        # (GH 11755).
        # gh-11755: tz and no tz
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        y = Series(date_range("2012-01-01", "2012-01-02"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series2(self):
        # [FUNCTION DOC]
        # Verifies tz-aware datetime plus object series yields object dtype (GH 11887).
        # gh-11887: concat tz and object
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        y = Series(["a", "b"])
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series3(self, unit, unit2):
        # [FUNCTION DOC]
        # Takes the unit and unit2 fixtures. Verifies concatenating UTC datetime columns of
        # two resolutions gives a UTC dtype in the finer unit (GH 12217, GH 12306).
        # see gh-12217 and gh-12306
        # Concatenating two UTC times
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        first[0] = first[0].dt.tz_localize("UTC")

        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f"M8[{unit2}]")
        second[0] = second[0].dt.tz_localize("UTC")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, UTC]"

    def test_concat_tz_series4(self, unit, unit2):
        # [FUNCTION DOC]
        # Same as above but for the Europe/London zone with two single-row frames.
        # Concatenating two London times
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f"M8[{unit2}]")
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series5(self, unit, unit2):
        # [FUNCTION DOC]
        # Same as above with a two-row first frame and one-row second frame (London).
        # Concatenating 2+1 London times
        first = DataFrame(
            [[datetime(2016, 1, 1)], [datetime(2016, 1, 2)]], dtype=f"M8[{unit}]"
        )
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 3)]], dtype=f"M8[{unit2}]")
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series6(self, unit, unit2):
        # [FUNCTION DOC]
        # Same as above with a one-row first frame and two-row second frame (London).
        # Concatenating 1+2 London times
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame(
            [[datetime(2016, 1, 2)], [datetime(2016, 1, 3)]], dtype=f"M8[{unit2}]"
        )
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series_tzlocal(self):
        # [FUNCTION DOC]
        # Verifies concatenating series of timestamps with the dateutil local timezone
        # keeps equal values and the local-timezone microsecond dtype (GH 13583).
        # see gh-13583
        x = [
            Timestamp("2011-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2011-02-01", tz=dateutil.tz.tzlocal()),
        ]
        y = [
            Timestamp("2012-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2012-02-01", tz=dateutil.tz.tzlocal()),
        ]

        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y))
        assert result.dtype == "datetime64[us, tzlocal()]"

    def test_concat_tz_series_with_datetimelike(self):
        # [FUNCTION DOC]
        # Verifies tz-aware timestamps mixed with timedeltas, or with periods, concatenate
        # to object dtype (GH 12620).
        # see gh-12620: tz and timedelta
        x = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-02-01", tz="US/Eastern"),
        ]
        y = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

        # tz and period
        y = [pd.Period("2011-03", freq="M"), pd.Period("2011-04", freq="M")]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

    def test_concat_tz_frame(self):
        # [FUNCTION DOC]
        # Verifies column-wise concatenation of two single-column frames with different
        # timezones reconstructs the original two-timezone frame.
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        )

        # concat
        df3 = concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        tm.assert_frame_equal(df2, df3)

    def test_concat_multiple_tzs(self):
        # [FUNCTION DOC]
        # Verifies concatenating frames of a naive timestamp, a UTC one and an EST one:
        # naive with either gives object dtype; UTC with EST gives the pair (GH 12467).
        # GH#12467
        # combining datetime tz-aware and naive DataFrames
        ts1 = Timestamp("2015-01-01", tz=None)
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="EST")

        df1 = DataFrame({"time": [ts1]})
        df2 = DataFrame({"time": [ts2]})
        df3 = DataFrame({"time": [ts3]})

        results = concat([df1, df2]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts2]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df1, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts3]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df2, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts2, ts3]})
        tm.assert_frame_equal(results, expected)

    def test_concat_multiindex_with_tz(self):
        # [FUNCTION DOC]
        # Verifies that concatenating a frame twice, indexed by a tz-aware datetime level
        # and a string level, preserves the multi-level index and zone (GH 6606).
        # GH 6606
        df = DataFrame(
            {
                "dt": DatetimeIndex(
                    [
                        datetime(2014, 1, 1),
                        datetime(2014, 1, 2),
                        datetime(2014, 1, 3),
                    ],
                    dtype="M8[ns, US/Pacific]",
                ),
                "b": ["A", "B", "C"],
                "c": [1, 2, 3],
                "d": [4, 5, 6],
            }
        )
        df = df.set_index(["dt", "b"])

        exp_idx1 = DatetimeIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03"] * 2,
            dtype="M8[ns, US/Pacific]",
            name="dt",
        )
        exp_idx2 = Index(["A", "B", "C"] * 2, name="b")
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame(
            {"c": [1, 2, 3] * 2, "d": [4, 5, 6] * 2}, index=exp_idx, columns=["c", "d"]
        )

        result = concat([df, df])
        tm.assert_frame_equal(result, expected)

    def test_concat_tz_not_aligned(self):
        # [FUNCTION DOC]
        # Verifies that with sorting and ignored index, concatenating frames with one
        # shared and one extra tz-aware column gives NaT in the missing part (GH 22796).
        # GH#22796
        ts = pd.to_datetime([1, 2]).tz_localize("UTC")
        a = DataFrame({"A": ts})
        b = DataFrame({"A": ts, "B": ts})
        result = concat([a, b], sort=True, ignore_index=True)
        expected = DataFrame(
            {"A": list(ts) + list(ts), "B": [pd.NaT, pd.NaT, *list(ts)]}
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "t1",
        [
            "2015-01-01",
            pytest.param(
                pd.NaT,
                marks=pytest.mark.xfail(
                    reason="GH23037 incorrect dtype when concatenating"
                ),
            ),
        ],
    )
    def test_concat_tz_NaT(self, t1):
        # [FUNCTION DOC]
        # Parametrized on the first timestamp (a date string, or NaT marked as expected
        # failure for GH 23037 incorrect dtype). Verifies concatenating tz-aware
        # multi-column frames of different widths keeps the dtype with NaT fill (GH 22796).
        # GH#22796
        # Concatenating tz-aware multicolumn DataFrames
        ts1 = Timestamp(t1, tz="UTC")
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="UTC")

        df1 = DataFrame([[ts1, ts2]])
        df2 = DataFrame([[ts3]])

        result = concat([df1, df2])
        expected = DataFrame([[ts1, ts2], [ts3, pd.NaT]], index=[0, 0])

        tm.assert_frame_equal(result, expected)

    def test_concat_tz_with_empty(self):
        # [FUNCTION DOC]
        # Verifies concatenating a tz-aware frame with an empty frame returns the tz-aware
        # frame unchanged (GH 9188).
        # GH 9188
        result = concat(
            [DataFrame(date_range("2000", periods=1, tz="UTC")), DataFrame()]
        )
        expected = DataFrame(date_range("2000", periods=1, tz="UTC"))
        tm.assert_frame_equal(result, expected)


class TestPeriodConcat:
    # [CLASS DOC]
    # Test collection for concatenation of period data. Stateless.
    def test_concat_period_series(self):
        # [FUNCTION DOC]
        # Verifies concatenating period series of the same frequency keeps the period
        # dtype.
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="D"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="Period[D]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_period_multiple_freq_series(self):
        # [FUNCTION DOC]
        # Verifies series of periods with different frequencies (day and month) concatenate
        # to object dtype.
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="M"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series(self):
        # [FUNCTION DOC]
        # Verifies day-frequency and month-frequency periods with the same dates
        # concatenate to object dtype.
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="M"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series2(self):
        # [FUNCTION DOC]
        # Verifies periods with datetimes concatenate to object dtype.
        # non-period
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(DatetimeIndex(["2015-11-01", "2015-12-01"]))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series3(self):
        # [FUNCTION DOC]
        # Verifies periods with strings concatenate to object dtype.
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(["A", "B"])
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"


def test_concat_timedelta64_block():
    # [FUNCTION DOC]
    # Verifies concatenating a frame with a timedelta column with itself keeps the
    # values in both halves (index type not checked).
    rng = to_timedelta(np.arange(10), unit="s")

    df = DataFrame({"time": rng})

    result = concat([df, df])
    tm.assert_frame_equal(result.iloc[:10], df, check_index_type=False)
    tm.assert_frame_equal(result.iloc[10:], df, check_index_type=False)


def test_concat_multiindex_datetime_nat():
    # [FUNCTION DOC]
    # Verifies column-wise concatenation of frames whose multi-level indexes contain
    # NaT in the second level aligns rows correctly with missing fill (GH 44900).
    # GH#44900
    left = DataFrame({"a": 1}, index=MultiIndex.from_tuples([(1, pd.NaT)]))
    right = DataFrame(
        {"b": 2}, index=MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)])
    )
    result = concat([left, right], axis="columns")
    expected = DataFrame(
        {"a": [1.0, np.nan], "b": 2}, MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)])
    )
    tm.assert_frame_equal(result, expected)


def test_concat_float_datetime64():
    # [FUNCTION DOC]
    # Verifies concatenation of a datetime frame with a float frame yields an object
    # column holding both values; with empty slices of either (or both) the result
    # is an object column, with the non-empty values retained as object (GH 32934).
    # GH#32934
    df_time = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")})
    df_float = DataFrame({"A": pd.array([1.0], dtype="float64")})

    expected = DataFrame(
        {
            "A": [
                pd.array(["2000"], dtype="datetime64[ns]")[0],
                pd.array([1.0], dtype="float64")[0],
            ]
        },
        index=[0, 0],
    )
    result = concat([df_time, df_float])
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"A": pd.array([], dtype="object")})
    result = concat([df_time.iloc[:0], df_float.iloc[:0]])
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"A": pd.array([1.0], dtype="object")})
    result = concat([df_time.iloc[:0], df_float])
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")}).astype(
        object
    )

    result = concat([df_time, df_float.iloc[:0]])
    tm.assert_frame_equal(result, expected)
