# [MODULE DOC]
# Tests for the frequency-conversion method of frames and series (asfreq): upsampling
# and downsampling a datetime or period indexed object to a given frequency, with fill
# method, fill value, normalization, end/start conversion for periods, index name and
# freq preservation, empty and unsorted input, and invalid or deprecated frequency
# aliases. Uses the frame-or-series, datetime frame and time unit fixtures plus the
# testing helpers module. No import-time side effects.

from datetime import datetime

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import MonthEnd

from pandas import (
    DataFrame,
    DatetimeIndex,
    PeriodIndex,
    Series,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestAsFreq:
    # [CLASS DOC]
    # Stateless pytest class grouping the asfreq tests.
    def test_asfreq2(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Converts a three-point business-month-end
        # indexed object to business-day frequency and back (by alias, with pad method,
        # and with offset objects); round trip must equal the original. An empty slice
        # returns a new empty object. For series only, conversion to daily with a fill
        # value is verified by counting values.
        ts = frame_or_series(
            [0.0, 1.0, 2.0],
            index=DatetimeIndex(
                [
                    datetime(2009, 10, 30),
                    datetime(2009, 11, 30),
                    datetime(2009, 12, 31),
                ],
                dtype="M8[ns]",
                freq="BME",
            ),
        )

        daily_ts = ts.asfreq("B")
        monthly_ts = daily_ts.asfreq("BME")
        tm.assert_equal(monthly_ts, ts)

        daily_ts = ts.asfreq("B", method="pad")
        monthly_ts = daily_ts.asfreq("BME")
        tm.assert_equal(monthly_ts, ts)

        daily_ts = ts.asfreq(offsets.BDay())
        monthly_ts = daily_ts.asfreq(offsets.BMonthEnd())
        tm.assert_equal(monthly_ts, ts)

        result = ts[:0].asfreq("ME")
        assert len(result) == 0
        assert result is not ts

        if frame_or_series is Series:
            daily_ts = ts.asfreq("D", fill_value=-1)
            result = daily_ts.value_counts().sort_index()
            expected = Series(
                [60, 1, 1, 1], index=[-1.0, 2.0, 1.0, 0.0], name="count"
            ).sort_index()
            tm.assert_series_equal(result, expected)

    def test_asfreq_datetimeindex_empty(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. Converting a one-element datetime-indexed
        # object to hourly frequency gives the same index as converting an empty object
        # of that index.
        # GH#14320
        index = DatetimeIndex(["2016-09-29 11:00"])
        expected = frame_or_series(index=index, dtype=object).asfreq("h")
        result = frame_or_series([3], index=index.copy()).asfreq("h")
        tm.assert_index_equal(expected.index, result.index)

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_tz_aware_asfreq_smoke(self, tz, frame_or_series):
        # [FUNCTION DOC]
        # Parametrized over a pytz and a dateutil US/Eastern zone, with the
        # frame-or-series fixture. Smoke test: converting a long tz-aware daily random
        # object to minute frequency must not fail; no assertion on the result.
        dr = date_range("2011-12-01", "2012-07-20", freq="D", tz=tz)

        obj = frame_or_series(
            np.random.default_rng(2).standard_normal(len(dr)), index=dr
        )

        # it works!
        obj.asfreq("min")

    def test_asfreq_normalize(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. With normalize true, an index with a 9:30 time
        # of day is converted to the midnight index with daily frequency and the values
        # stay unchanged.
        rng = date_range("1/1/2000 09:30", periods=20)
        norm = date_range("1/1/2000", periods=20)

        vals = np.random.default_rng(2).standard_normal((20, 3))

        obj = DataFrame(vals, index=rng)
        expected = DataFrame(vals, index=norm)
        if frame_or_series is Series:
            obj = obj[0]
            expected = expected[0]

        result = obj.asfreq("D", normalize=True)
        tm.assert_equal(result, expected)

    def test_asfreq_keep_index_name(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. The index name is preserved by the conversion
        # to a ten-day frequency.
        # GH#9854
        index_name = "bar"
        index = date_range("20130101", periods=20, name=index_name)
        obj = DataFrame(list(range(20)), columns=["foo"], index=index)
        obj = tm.get_obj(obj, frame_or_series)

        assert index_name == obj.index.name
        assert index_name == obj.asfreq("10D").index.name

    def test_asfreq_ts(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. For an annual period index converted to daily
        # frequency with how end and how start, the length is unchanged and the index
        # equals the period index converted with the same how.
        index = period_range(freq="Y", start="1/1/2001", end="12/31/2010")
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 3)), index=index
        )
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.asfreq("D", how="end")
        exp_index = index.asfreq("D", how="end")
        assert len(result) == len(obj)
        tm.assert_index_equal(result.index, exp_index)

        result = obj.asfreq("D", how="start")
        exp_index = index.asfreq("D", how="start")
        assert len(result) == len(obj)
        tm.assert_index_equal(result.index, exp_index)

    def test_asfreq_resample_set_correct_freq(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. An index without freq but with inferred daily
        # frequency gets freq daily after asfreq and after resample followed by asfreq.
        # GH#5613
        # we test if .asfreq() and .resample() set the correct value for .freq
        dti = to_datetime(["2012-01-01", "2012-01-02", "2012-01-03"])
        obj = DataFrame({"col": [1, 2, 3]}, index=dti)
        obj = tm.get_obj(obj, frame_or_series)

        # testing the settings before calling .asfreq() and .resample()
        assert obj.index.freq is None
        assert obj.index.inferred_freq == "D"

        # does .asfreq() set .freq correctly?
        assert obj.asfreq("D").index.freq == "D"

        # does .resample() set .freq correctly?
        assert obj.resample("D").asfreq().index.freq == "D"

    def test_asfreq_empty(self, datetime_frame):
        # [FUNCTION DOC]
        # Converting a frame with zero rows to business-month-end must not fail and
        # returns a new object.
        # test does not blow up on length-0 DataFrame
        zero_length = datetime_frame.reindex([])
        result = zero_length.asfreq("BME")
        assert result is not zero_length

    def test_asfreq(self, datetime_frame):
        # [FUNCTION DOC]
        # Offset object and alias string give equal results on the datetime frame; then
        # calls a further conversion with pad method.
        # KNOWN ISSUE: the test has a comment promising to check the pad-conversion
        # result, but the results are discarded (and the same call is made twice), so
        # nothing is asserted about that part.
        offset_monthly = datetime_frame.asfreq(offsets.BMonthEnd())
        rule_monthly = datetime_frame.asfreq("BME")

        tm.assert_frame_equal(offset_monthly, rule_monthly)

        rule_monthly.asfreq("B", method="pad")
        # TODO: actually check that this worked.

        # don't forget!
        rule_monthly.asfreq("B", method="pad")

    def test_asfreq_datetimeindex(self):
        # [FUNCTION DOC]
        # Conversion to business frequency of a frame and of its column both produce a
        # datetime index.
        df = DataFrame(
            {"A": [1, 2, 3]},
            index=[datetime(2011, 11, 1), datetime(2011, 11, 2), datetime(2011, 11, 3)],
        )
        df = df.asfreq("B")
        assert isinstance(df.index, DatetimeIndex)

        ts = df["A"].asfreq("B")
        assert isinstance(ts.index, DatetimeIndex)

    def test_asfreq_fillvalue(self):
        # [FUNCTION DOC]
        # Upsampling a float frame and series (with a pre-existing missing value) with a
        # fill value fills only the newly introduced rows, not the pre-existing missing,
        # matching a fill of the unfilled result.
        # test for fill value during upsampling, related to issue 3715

        # setup
        rng = date_range("1/1/2016", periods=10, freq="2s")
        # Explicit cast to 'float' to avoid implicit cast when setting None
        ts = Series(np.arange(len(rng)), index=rng, dtype="float")
        df = DataFrame({"one": ts})

        # insert pre-existing missing value
        df.loc["2016-01-01 00:00:08", "one"] = None

        actual_df = df.asfreq(freq="1s", fill_value=9.0)
        expected_df = df.asfreq(freq="1s").fillna(9.0)
        expected_df.loc["2016-01-01 00:00:08", "one"] = None
        tm.assert_frame_equal(expected_df, actual_df)

        expected_series = ts.asfreq(freq="1s").fillna(9.0)
        actual_series = ts.asfreq(freq="1s", fill_value=9.0)
        tm.assert_series_equal(expected_series, actual_series)

    def test_asfreq_with_date_object_index(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. An index of python date objects is accepted;
        # conversion with forward fill equals conversion of the equivalent
        # datetime-indexed object.
        rng = date_range("1/1/2000", periods=20, unit="ns")
        ts = frame_or_series(np.random.default_rng(2).standard_normal(20), index=rng)

        ts2 = ts.copy()
        ts2.index = [x.date() for x in ts2.index]

        result = ts2.asfreq("4h", method="ffill")
        expected = ts.asfreq("4h", method="ffill")
        tm.assert_equal(result, expected)

    def test_asfreq_with_unsorted_index(self, frame_or_series):
        # [FUNCTION DOC]
        # Takes frame-or-series fixture. With an unsorted datetime index, conversion to
        # daily frequency keeps all rows and returns them sorted with inferred daily
        # freq.
        # GH#39805
        # Test that rows are not dropped when the datetime index is out of order
        index = to_datetime(["2021-01-04", "2021-01-02", "2021-01-03", "2021-01-01"])
        result = frame_or_series(range(4), index=index)

        expected = result.reindex(sorted(index))
        expected.index = expected.index._with_freq("infer")

        result = result.asfreq("D")
        tm.assert_equal(result, expected)

    def test_asfreq_after_normalize(self, unit):
        # [FUNCTION DOC]
        # Takes time unit fixture. Checks that a datetime index created from a
        # normalized date range with an explicit daily freq keeps the unit and equals
        # the expected index.
        # KNOWN ISSUE: despite living in the asfreq test class, this test never calls
        # the asfreq method; it only exercises normalization and index construction.
        # https://github.com/pandas-dev/pandas/issues/50727
        result = DatetimeIndex(
            date_range("2000", periods=2).as_unit(unit).normalize(), freq="D"
        )
        expected = DatetimeIndex(["2000-01-01", "2000-01-02"], freq="D").as_unit(unit)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, freq_half",
        [
            ("2ME", "ME"),
            (MonthEnd(2), MonthEnd(1)),
        ],
    )
    def test_asfreq_2ME(self, freq, freq_half):
        # [FUNCTION DOC]
        # Parametrized over alias and offset spellings of a two-month-end frequency.
        # Converting a six-month series to two-month-end frequency keeps every other
        # value.
        index = date_range("1/1/2000", periods=6, freq=freq_half)
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=index)})
        expected = df.asfreq(freq=freq)

        index = date_range("1/1/2000", periods=3, freq=freq)
        result = DataFrame({"s": Series([0.0, 2.0, 4.0], index=index)})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, freq_depr",
        [
            ("2ME", "2M"),
            ("2ME", "2m"),
            ("2QE", "2Q"),
            ("2QE-SEP", "2Q-SEP"),
            ("1BQE", "1BQ"),
            ("2BQE-SEP", "2BQ-SEP"),
            ("2BQE-SEP", "2bq-sep"),
            ("1YE", "1y"),
            ("2YE-MAR", "2Y-MAR"),
        ],
    )
    def test_asfreq_frequency_M_Q_Y_raises(self, freq, freq_depr):
        # [FUNCTION DOC]
        # Parametrized over (valid alias, deprecated alias). Using a removed alias (e.g.
        # two M, quarter, business quarter, year forms, lower-case variants) in the
        # conversion raises a value error saying the frequency is invalid.
        msg = f"Invalid frequency: {freq_depr}"

        index = date_range("1/1/2000", periods=4, freq=f"{freq[1:]}")
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0], index=index)})
        with pytest.raises(ValueError, match=msg):
            df.asfreq(freq=freq_depr)

    @pytest.mark.parametrize(
        "freq, error_msg",
        [
            (
                "2MS",
                "Invalid frequency: 2MS",
            ),
            (
                offsets.MonthBegin(),
                r"\<MonthBegin\> is not supported as period frequency",
            ),
            (
                offsets.DateOffset(months=2),
                r"\<DateOffset: months=2\> is not supported as period frequency",
            ),
        ],
    )
    def test_asfreq_unsupported_freq(self, freq, error_msg):
        # [FUNCTION DOC]
        # Parametrized over month-start alias, month-begin offset and a multi-month date
        # offset. On a monthly period index the conversion raises a value error with the
        # message matching the case (invalid frequency, or not supported as a period
        # frequency).
        # https://github.com/pandas-dev/pandas/issues/56718
        index = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")
        df = DataFrame({"a": Series([0, 1], index=index)})

        with pytest.raises(ValueError, match=error_msg):
            df.asfreq(freq=freq)

    @pytest.mark.parametrize(
        "freq, freq_depr",
        [
            ("2YE", "2A"),
            ("2BYE-MAR", "2BA-MAR"),
        ],
    )
    def test_asfreq_frequency_A_BA_raises(self, freq, freq_depr):
        # [FUNCTION DOC]
        # Parametrized over (valid, deprecated) annual alias pairs. The deprecated alias
        # raises a value error about invalid frequency.
        msg = f"Invalid frequency: {freq_depr}"

        index = date_range("1/1/2000", periods=4, freq=freq)
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0], index=index)})

        with pytest.raises(ValueError, match=msg):
            df.asfreq(freq=freq_depr)
