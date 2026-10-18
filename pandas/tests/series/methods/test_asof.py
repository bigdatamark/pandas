# [MODULE DOC]
# Tests for Series.asof, which returns the last non-missing value at or before
# each requested label of a sorted index. Covers nanosecond-resolution datetime
# indexes, array/list/scalar/string lookups, NaN gaps, period indexes, error
# cases and all-NaN series. No import-time side effects.

import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    DatetimeIndex,
    PeriodIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    offsets,
    period_range,
)
import pandas._testing as tm


class TestSeriesAsof:
    # [CLASS DOC]
    # Groups the tests of Series.asof. Stateless.
    def test_asof_nanosecond_index_access(self):
        # [FUNCTION DOC]
        # Regression test (GH#46903): with a nanosecond-resolution datetime index,
        # asof of the first label equals the value looked up by a nanosecond-precision
        # string and by a Timestamp, and the index resolution is reported as nanosecond.
        ts = Timestamp("20130101").as_unit("ns")._value
        dti = DatetimeIndex([ts + 50 + i for i in range(100)])
        ser = Series(np.random.default_rng(2).standard_normal(100), index=dti)

        first_value = ser.asof(ser.index[0])

        # GH#46903 previously incorrectly was "day"
        assert dti.resolution == "nanosecond"

        # this used to not work bc parsing was done by dateutil that didn't
        #  handle nanoseconds
        assert first_value == ser["2013-01-01 00:00:00.000000050"]

        expected_ts = np.datetime64("2013-01-01 00:00:00.000000050", "ns")
        assert first_value == ser[Timestamp(expected_ts)]

    def test_basic(self):
        # [FUNCTION DOC]
        # Builds a random series with a block of NaN values in the middle and asks asof
        # for a denser date range given as an index and as a list. Expects no missing
        # results, that dates inside the NaN gap take the last valid value before it,
        # and that the first date at or after the next valid label takes that label's value.
        # KNOWN ISSUE: the lower and upper bound labels are computed twice and the
        # first computation is never used.
        # array or list or dates
        N = 50
        rng = date_range("1/1/1990", periods=N, freq="53s")
        ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        ts.iloc[15:30] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        result = ts.asof(dates)
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        result = ts.asof(list(dates))
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        mask = (result.index >= lb) & (result.index < ub)
        rs = result[mask]
        assert (rs == ts[lb]).all()

        val = result[result.index[result.index >= ub][0]]
        assert ts[ub] == val

    def test_scalar(self):
        # [FUNCTION DOC]
        # Scalar lookups on a float series with two NaN gaps: a label inside a gap
        # returns the last valid value before the gap; strings are accepted as labels;
        # a label present with a valid value returns it; a label before the first
        # index entry returns NaN.
        N = 30
        rng = date_range("1/1/1990", periods=N, freq="53s")
        # Explicit cast to float avoid implicit cast when setting nan
        ts = Series(np.arange(N), index=rng, dtype="float")
        ts.iloc[5:10] = np.nan
        ts.iloc[15:20] = np.nan

        val1 = ts.asof(ts.index[7])
        val2 = ts.asof(ts.index[19])

        assert val1 == ts.iloc[4]
        assert val2 == ts.iloc[14]

        # accepts strings
        val1 = ts.asof(str(ts.index[7]))
        assert val1 == ts.iloc[4]

        # in there
        result = ts.asof(ts.index[3])
        assert result == ts.iloc[3]

        # no as of value
        d = ts.index[0] - offsets.BDay()
        assert np.isnan(ts.asof(d))

    def test_with_nan(self):
        # [FUNCTION DOC]
        # Resamples a series to a finer frequency (creating NaN entries) and checks asof
        # over its own index: first with the initial NaNs forward-filled from the last
        # valid value, then after setting additional NaN values in the middle and at the
        # end, expecting values carried forward from the last valid entry.
        # basic asof test
        rng = date_range("1/1/2000", "1/2/2000", freq="4h")
        s = Series(np.arange(len(rng)), index=rng)
        r = s.resample("2h").mean()

        result = r.asof(r.index)
        expected = Series(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        tm.assert_series_equal(result, expected)

        r.iloc[3:5] = np.nan
        result = r.asof(r.index)
        expected = Series(
            [0, 0, 1, 1, 1, 1, 3, 3, 4, 4, 5, 5, 6.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        tm.assert_series_equal(result, expected)

        r.iloc[-3:] = np.nan
        result = r.asof(r.index)
        expected = Series(
            [0, 0, 1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        tm.assert_series_equal(result, expected)

    def test_periodindex(self):
        # [FUNCTION DOC]
        # Same scenarios as the datetime test but with a period-indexed series: asof by
        # a datetime index and list returns no missing values and the right filled
        # values; scalar, string and exact lookups, a label before the start gives a
        # missing value; and passing an index with a different period frequency raises
        # the incompatible frequency error with message "Input has different freq".
        # KNOWN ISSUE: the lower and upper bound labels are computed twice and the
        # first computation is never used.
        # array or list or dates
        N = 50
        rng = period_range("1/1/1990", periods=N, freq="h")
        ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        ts.iloc[15:30] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="37min")

        result = ts.asof(dates)
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        result = ts.asof(list(dates))
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        pix = PeriodIndex(result.index.values, freq="h")
        mask = (pix >= lb) & (pix < ub)
        rs = result[mask]
        assert (rs == ts[lb]).all()

        ts.iloc[5:10] = np.nan
        ts.iloc[15:20] = np.nan

        val1 = ts.asof(ts.index[7])
        val2 = ts.asof(ts.index[19])

        assert val1 == ts.iloc[4]
        assert val2 == ts.iloc[14]

        # accepts strings
        val1 = ts.asof(str(ts.index[7]))
        assert val1 == ts.iloc[4]

        # in there
        assert ts.asof(ts.index[3]) == ts.iloc[3]

        # no as of value
        d = ts.index[0].to_timestamp() - offsets.BDay()
        assert isna(ts.asof(d))

        # Mismatched freq
        msg = "Input has different freq"
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts.asof(rng.asfreq("D"))

    def test_errors(self):
        # [FUNCTION DOC]
        # Error cases: asof on a series with a non-monotonic index raises a value error
        # requiring a sorted index; passing a subset argument for a Series raises a
        # value error saying subset is not valid for Series.
        s = Series(
            [1, 2, 3],
            index=[Timestamp("20130101"), Timestamp("20130103"), Timestamp("20130102")],
        )

        # non-monotonic
        assert not s.index.is_monotonic_increasing
        with pytest.raises(ValueError, match="requires a sorted index"):
            s.asof(s.index[0])

        # subset with Series
        N = 10
        rng = date_range("1/1/1990", periods=N, freq="53s")
        s = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        with pytest.raises(ValueError, match="not valid for Series"):
            s.asof(s.index[0], subset="foo")

    def test_all_nans(self):
        # [FUNCTION DOC]
        # Regression test (GH 15713): when every value is NaN, asof returns an all-NaN
        # series for a date index, NaN for a scalar, and keeps the series name.
        # GH 15713
        # series is all nans

        # testing non-default indexes
        N = 50
        rng = date_range("1/1/1990", periods=N, freq="53s")

        dates = date_range("1/1/1990", periods=N * 3, freq="25s")
        result = Series(np.nan, index=rng).asof(dates)
        expected = Series(np.nan, index=dates)
        tm.assert_series_equal(result, expected)

        # testing scalar input
        date = date_range("1/1/1990", periods=N * 3, freq="25s")[0]
        result = Series(np.nan, index=rng).asof(date)
        assert isna(result)

        # test name is propagated
        result = Series(np.nan, index=[1, 2, 3, 4], name="test").asof([4, 5])
        expected = Series(np.nan, index=[4, 5], name="test")
        tm.assert_series_equal(result, expected)
