# [MODULE DOC]
# Tests for indexing with datetime-like indexes and timezones: key errors for malformed
# string keys, row access on timezone-aware columns, scalar access consistency, list-like
# indexing on tz-aware indexes, nanosecond precision, millisecond-resolution string slicing, and
# pyarrow-backed timestamp indexes.
import re

import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndex:
    # [CLASS DOC]
    # Test collection for indexing with datetime indexes and timezone-aware data.
    def test_get_loc_naive_dti_aware_str_deprecated(self):
        # GH#46903
        # [FUNCTION DOC]
        # GH 46903. A string key with a UTC offset on a timezone-naive nanosecond index raises a key error
        # (message is the repr of the key) both for series getitem and for get_loc on the index.
        # KNOWN ISSUE: the name suggests a deprecation, but the test asserts that a key error is raised.
        ts = Timestamp("20130101")._value
        dti = pd.DatetimeIndex([ts + 50 + i for i in range(100)])
        ser = Series(range(100), index=dti)

        key = "2013-01-01 00:00:00.000000050+0000"
        msg = re.escape(repr(key))
        with pytest.raises(KeyError, match=msg):
            ser[key]

        with pytest.raises(KeyError, match=msg):
            dti.get_loc(key)

    def test_indexing_with_datetime_tz(self):
        # GH#8260
        # support datetime64 with tz

        # [FUNCTION DOC]
        # GH 8260. A frame with timezone-aware columns where some entries are set to NaT: selecting a
        # row by position or label gives an object series with a tz-aware timestamp and NaT values.
        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        dr = date_range("20130110", periods=3)
        df = DataFrame({"A": idx, "B": dr})
        df["C"] = idx
        df.iloc[1, 1] = pd.NaT
        df.iloc[1, 2] = pd.NaT

        expected = Series(
            [Timestamp("2013-01-02 00:00:00-0500", tz="US/Eastern"), pd.NaT, pd.NaT],
            index=list("ABC"),
            dtype="object",
            name=1,
        )

        # indexing
        result = df.iloc[1]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_indexing_fast_xs(self):
        # indexing - fast_xs
        # [FUNCTION DOC]
        # Row selection on a frame with a single UTC datetime column by position or label returns a
        # series with the tz-aware dtype and the row number as name; a boolean mask on the column gives
        # the later rows.
        df = DataFrame({"a": date_range("2014-01-01", periods=10, tz="UTC", unit="ns")})
        result = df.iloc[5]
        expected = Series(
            [Timestamp("2014-01-06 00:00:00+0000", tz="UTC")],
            index=["a"],
            name=5,
            dtype="M8[ns, UTC]",
        )
        tm.assert_series_equal(result, expected)

        result = df.loc[5]
        tm.assert_series_equal(result, expected)

        # indexing - boolean
        result = df[df.a > df.a[3]]
        expected = df.iloc[4:]
        tm.assert_frame_equal(result, expected)

    def test_consistency_with_tz_aware_scalar(self):
        # xef gh-12938
        # various ways of indexing the same tz-aware scalar
        # [FUNCTION DOC]
        # GH 12938. All of getitem, iloc, loc, iat, at on a frame (and the column series) return the same
        # timezone-aware Brussels timestamp scalar.
        df = Series([Timestamp("2016-03-30 14:35:25", tz="Europe/Brussels")]).to_frame()

        df = pd.concat([df, df]).reset_index(drop=True)
        expected = Timestamp("2016-03-30 14:35:25+0200", tz="Europe/Brussels")

        result = df[0][0]
        assert result == expected

        result = df.iloc[0, 0]
        assert result == expected

        result = df.loc[0, 0]
        assert result == expected

        result = df.iat[0, 0]
        assert result == expected

        result = df.at[0, 0]
        assert result == expected

        result = df[0].loc[0]
        assert result == expected

        result = df[0].at[0]
        assert result == expected

    def test_indexing_with_datetimeindex_tz(self, indexer_sl):
        # GH 12050
        # indexing on a series with a datetimeindex with tz
        # [FUNCTION DOC]
        # GH 12050. Uses the fixture that provides both loc and getitem-style accessors. A tz-aware
        # index or list of its entries as selector returns the whole series (index frequency dropped for
        # the list); setting through them assigns scalar everywhere; single-timestamp get and set work.
        index = date_range("2015-01-01", periods=2, tz="utc")

        ser = Series(range(2), index=index, dtype="int64")

        # list-like indexing

        for sel in (index, list(index)):
            # getitem
            result = indexer_sl(ser)[sel]
            expected = ser.copy()
            if sel is not index:
                expected.index = expected.index._with_freq(None)
            tm.assert_series_equal(result, expected)

            # setitem
            result = ser.copy()
            indexer_sl(result)[sel] = 1
            expected = Series(1, index=index)
            tm.assert_series_equal(result, expected)

        # single element indexing

        # getitem
        assert indexer_sl(ser)[index[1]] == 1

        # setitem
        result = ser.copy()
        indexer_sl(result)[index[1]] = 5
        expected = Series([0, 5], index=index)
        tm.assert_series_equal(result, expected)

    def test_nanosecond_getitem_setitem_with_tz(self):
        # GH 11679
        # [FUNCTION DOC]
        # GH 11679. For a frame indexed by a tz-aware nanosecond timestamp, label lookup returns the row
        # as a series named with that timestamp, and setting a value by label writes it.
        data = ["2016-06-28 08:30:00.123456789"]
        index = pd.DatetimeIndex(data, dtype="datetime64[ns, America/Chicago]")
        df = DataFrame({"a": [10]}, index=index)
        result = df.loc[df.index[0]]
        expected = Series(10, index=["a"], name=df.index[0])
        tm.assert_series_equal(result, expected)

        result = df.copy()
        result.loc[df.index[0], "a"] = -1
        expected = DataFrame(-1, index=index, columns=["a"])
        tm.assert_frame_equal(result, expected)

    def test_getitem_str_slice_millisecond_resolution(self, frame_or_series):
        # GH#33589

        # [FUNCTION DOC]
        # GH 33589. Parametrized over frame and series. A string slice whose bounds have millisecond
        # resolution includes both bounds and returns the two middle rows.
        keys = [
            "2017-10-25T16:25:04.151",
            "2017-10-25T16:25:04.252",
            "2017-10-25T16:50:05.237",
            "2017-10-25T16:50:05.238",
        ]
        obj = frame_or_series(
            [1, 2, 3, 4],
            index=[Timestamp(x) for x in keys],
        )
        result = obj[keys[1] : keys[2]]
        expected = frame_or_series(
            [2, 3],
            index=[
                Timestamp(keys[1]),
                Timestamp(keys[2]),
            ],
        )
        tm.assert_equal(result, expected)

    def test_getitem_pyarrow_index(self, frame_or_series):
        # GH 53644
        # [FUNCTION DOC]
        # GH 53644. Parametrized over frame and series. Skipped if pyarrow is missing. Label selection with
        # the first two entries of a pyarrow-timestamp index returns the first two rows.
        pytest.importorskip("pyarrow")
        obj = frame_or_series(
            range(5),
            index=date_range("2020", freq="D", periods=5).astype(
                "timestamp[us][pyarrow]"
            ),
        )
        result = obj.loc[obj.index[:-3]]
        expected = frame_or_series(
            range(2),
            index=date_range("2020", freq="D", periods=2).astype(
                "timestamp[us][pyarrow]"
            ),
        )
        tm.assert_equal(result, expected)
