# [MODULE DOC]
# Tests for resampling objects indexed by a timedelta index (frames and series).
# Covers asfreq, missing values in the index, sub-period and edge-case binning,
# offsets, categorical data, quantiles, closed/label handling and pyarrow
# duration indexes. Uses shared fixtures (the unit fixture comes from conftest).
from datetime import timedelta

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range


def test_asfreq_bug():
    # [FUNCTION DOC]
    # Verifies that resampling a two-row frame indexed by zero and three minutes
    # with asfreq on one-minute bins inserts missing rows for the gaps. Expected is a
    # four-row float frame with a microsecond timedelta range index.
    df = DataFrame(data=[1, 3], index=[timedelta(), timedelta(minutes=3)])
    result = df.resample("1min").asfreq()
    expected = DataFrame(
        data=[1, np.nan, np.nan, 3],
        index=timedelta_range("0 day", periods=4, freq="1min", unit="us"),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_with_nat():
    # [FUNCTION DOC]
    # Verifies that a missing value in the timedelta index is dropped and the mean
    # over one-second bins leaves a missing result for the empty middle bin.
    # GH 13223
    index = pd.to_timedelta(["0s", pd.NaT, "2s"])
    result = DataFrame({"value": [2, 3, 5]}, index).resample("1s").mean()
    expected = DataFrame(
        {"value": [2.5, np.nan, 5.0]},
        index=timedelta_range("0 day", periods=3, freq="1s"),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_as_freq_with_subperiod():
    # [FUNCTION DOC]
    # Verifies that upsampling a five-minute timedelta index to two-minute bins with
    # asfreq produces missing values at the new points and keeps original values at
    # matching timestamps.
    # GH 13022
    index = timedelta_range("00:00:00", "00:10:00", freq="5min")
    df = DataFrame(data={"value": [1, 5, 10]}, index=index)
    result = df.resample("2min").asfreq()
    expected_data = {"value": [1, np.nan, np.nan, np.nan, np.nan, 10]}
    expected = DataFrame(
        data=expected_data, index=timedelta_range("00:00:00", "00:10:00", freq="2min")
    )
    tm.assert_frame_equal(result, expected)


def test_resample_with_timedeltas():
    # [FUNCTION DOC]
    # Verifies that summing over thirty-minute bins of a minute-level timedelta
    # index equals a group-by-sum on integer bucket positions, for both a frame and
    # a single series column.
    expected = DataFrame({"A": np.arange(1480)})
    expected = expected.groupby(expected.index // 30).sum()
    expected.index = timedelta_range("0 days", freq="30min", periods=50)

    df = DataFrame(
        {"A": np.arange(1480)},
        index=pd.to_timedelta(np.arange(1480), unit="min").as_unit("us"),
    )
    result = df.resample("30min").sum()

    tm.assert_frame_equal(result, expected)

    s = df["A"]
    result = s.resample("30min").sum()
    tm.assert_series_equal(result, expected["A"])


def test_resample_single_period_timedelta():
    # [FUNCTION DOC]
    # Verifies two-second sum resampling on a series with a one-second timedelta
    # index yields three bins with the expected sums.
    s = Series(list(range(5)), index=timedelta_range("1 day", freq="s", periods=5))
    result = s.resample("2s").sum()
    expected = Series([1, 5, 4], index=timedelta_range("1 day", freq="2s", periods=3))
    tm.assert_series_equal(result, expected)


def test_resample_timedelta_idempotency():
    # [FUNCTION DOC]
    # Verifies that resampling at the same frequency as the index (ten milliseconds)
    # with mean returns the data unchanged (as floats).
    # GH 12072
    index = timedelta_range("0", periods=9, freq="10ms")
    series = Series(range(9), index=index)
    result = series.resample("10ms").mean()
    expected = series.astype(float)
    tm.assert_series_equal(result, expected)


def test_resample_offset_with_timedeltaindex():
    # [FUNCTION DOC]
    # Verifies that the offset argument shifts the bin edges of a timedelta-indexed
    # series, comparing resulting indexes with and without the offset against
    # explicit timedelta ranges.
    # GH 10530 & 31809
    rng = timedelta_range(start="0s", periods=25, freq="s")
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    with_base = ts.resample("2s", offset="5s").mean()
    without_base = ts.resample("2s").mean()

    exp_without_base = timedelta_range(start="0s", end="25s", freq="2s")
    exp_with_base = timedelta_range(start="5s", end="29s", freq="2s")

    tm.assert_index_equal(without_base.index, exp_without_base)
    tm.assert_index_equal(with_base.index, exp_with_base)


def test_resample_categorical_data_with_timedeltaindex():
    # [FUNCTION DOC]
    # Verifies that aggregating with a function returning the most frequent value
    # works for object and categorical columns on a timedelta index; the categorical
    # column result is expected to have categorical dtype.
    # GH #12169
    df = DataFrame({"Group_obj": "A"}, index=pd.to_timedelta(list(range(20)), unit="s"))
    df["Group"] = df["Group_obj"].astype("category")
    result = df.resample("10s").agg(lambda x: x.value_counts().index[0])
    exp_tdi = pd.TimedeltaIndex(np.array([0, 10], dtype="m8[s]"), freq="10s")
    expected = DataFrame(
        {"Group_obj": ["A", "A"], "Group": ["A", "A"]},
        index=exp_tdi,
    )
    expected = expected.reindex(["Group_obj", "Group"], axis=1)
    expected["Group"] = expected["Group_obj"].astype("category")
    tm.assert_frame_equal(result, expected)


def test_resample_timedelta_values():
    # [FUNCTION DOC]
    # Verifies that the timedelta dtype of a column is preserved when resampling
    # with first introduces a missing value for an empty bin, for both frame and
    # series resampling.
    # GH 13119
    # check that timedelta dtype is preserved when NaT values are
    # introduced by the resampling

    times = timedelta_range("1 day", "6 day", freq="4D")
    df = DataFrame({"time": times}, index=times)

    times2 = timedelta_range("1 day", "6 day", freq="2D")
    exp = Series(times2, index=times2, name="time")
    exp.iloc[1] = pd.NaT

    res = df.resample("2D").first()["time"]
    tm.assert_series_equal(res, exp)
    res = df["time"].resample("2D").first()
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize(
    "start, end, freq, resample_freq",
    [
        ("8h", "21h59min50s", "10s", "3h"),  # GH 30353 example
        ("3h", "22h", "1h", "5h"),
        ("527D", "5006D", "3D", "10D"),
        ("1D", "10D", "1D", "2D"),  # GH 13022 example
        # tests that worked before GH 33498:
        ("8h", "21h59min50s", "10s", "2h"),
        ("0h", "21h59min50s", "10s", "3h"),
        ("10D", "85D", "D", "2D"),
    ],
)
def test_resample_timedelta_edge_case(start, end, freq, resample_freq):
    # [FUNCTION DOC]
    # Parametrized over start, end, source frequency and resample frequency. Verifies
    # that the resulting bins do not contain an extra trailing bin: the result index
    # equals the expected timedelta range including its frequency, and the last
    # value is not missing.
    # GH 33498
    # check that the timedelta bins does not contains an extra bin
    idx = timedelta_range(start=start, end=end, freq=freq)
    s = Series(np.arange(len(idx)), index=idx)
    result = s.resample(resample_freq).min()
    expected_index = timedelta_range(freq=resample_freq, start=start, end=end)
    tm.assert_index_equal(result.index, expected_index)
    assert result.index.freq == expected_index.freq
    assert not np.isnan(result.iloc[-1])


@pytest.mark.parametrize("duplicates", [True, False])
def test_resample_with_timedelta_yields_no_empty_groups(duplicates):
    # [FUNCTION DOC]
    # Parametrized on whether the columns contain duplicates. Verifies that
    # applying a length function over three-second bins of a large nanosecond-spaced
    # timedelta frame gives the expected counts per bin (twelve full bins and a
    # shorter final bin) with no empty groups.
    # GH 10603
    df = DataFrame(
        np.random.default_rng(2).normal(size=(10000, 4)),
        index=timedelta_range(start="0s", periods=10000, freq="3906250ns"),
    )
    if duplicates:
        # case with non-unique columns
        df.columns = ["A", "B", "A", "C"]

    result = df.loc["1s":, :].resample("3s").apply(lambda x: len(x))

    expected = DataFrame(
        [[768] * 4] * 12 + [[528] * 4],
        index=timedelta_range(start="1s", periods=13, freq="3s", unit="ns"),
    )
    expected.columns = df.columns
    tm.assert_frame_equal(result, expected)


def test_resample_quantile_timedelta(unit):
    # [FUNCTION DOC]
    # Takes the unit fixture (timedelta resolution). Verifies that a quantile
    # resample over a tz-aware date index with two-day bins returns timedelta values
    # preserving the requested resolution.
    # GH: 29485
    dtype = np.dtype(f"m8[{unit}]")
    df = DataFrame(
        {"value": pd.to_timedelta(np.arange(4), unit="s").astype(dtype)},
        index=pd.date_range("20200101", periods=4, tz="UTC"),
    )
    result = df.resample("2D").quantile(0.99)
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),
                pd.Timedelta("0 days 00:00:02.990000"),
            ]
        },
        index=pd.date_range("20200101", periods=2, tz="UTC", freq="2D"),
    ).astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_resample_closed_right():
    # [FUNCTION DOC]
    # Verifies that closed and label both set to right with minute bins on a
    # timedelta index of thirty-second spacing produces the expected sums and a
    # timedelta index with minute frequency.
    # GH#45414
    idx = pd.Index([pd.Timedelta(seconds=120 + i * 30) for i in range(10)])
    ser = Series(range(10), index=idx)
    result = ser.resample("min", closed="right", label="right").sum()
    expected = Series(
        [0, 3, 7, 11, 15, 9],
        index=pd.TimedeltaIndex(
            [pd.Timedelta(seconds=120 + i * 60) for i in range(6)], freq="min"
        ),
    )
    tm.assert_series_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_arrow_duration_resample():
    # [FUNCTION DOC]
    # Skipped without pyarrow. Verifies that resampling a series with a pyarrow
    # duration index by one day with mean returns the same values as floats with
    # the same index (GH 56371).
    # GH 56371
    idx = pd.Index(timedelta_range("1 day", periods=5), dtype="duration[ns][pyarrow]")
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    result = expected.resample("1D").mean()
    tm.assert_series_equal(result, expected)
