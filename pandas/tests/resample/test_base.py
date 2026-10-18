# [MODULE DOC]
# Tests of resampling behavior shared across datetime, timedelta and period indexes:
# frequency conversion without aggregation, filling and interpolation after resampling,
# behavior on empty or all-missing-time inputs for each resample method, iteration over a
# resampler, quantile, and first/last with skipna. Uses the shared resample-method fixture
# from the package conftest. Imports a private compatibility helper from the resample
# implementation to compute the expected index. All tests are module-level functions; one
# module fixture lists interpolation methods that need no extra arguments.

from datetime import datetime

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas.core.dtypes.common import is_extension_array_dtype

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat


@pytest.fixture(
    params=[
        "linear",
        "time",
        "index",
        "values",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "barycentric",
        "krogh",
        "from_derivatives",
        "piecewise_polynomial",
        "pchip",
        "akima",
    ],
)
def all_1d_no_arg_interpolation_methods(request):
    # [FUNCTION DOC]
    # Fixture parametrized over fifteen interpolation method names that take no extra
    # arguments (linear, time, index, values, nearest, zero, slinear, quadratic, cubic,
    # barycentric, krogh, from derivatives, piecewise polynomial, pchip, akima). Returns the
    # current method name.
    return request.param


@pytest.mark.parametrize("freq", ["2D", "1h"])
@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),
    ],
)
def test_asfreq(frame_or_series, index, freq):
    # [FUNCTION DOC]
    # Parametrized by frame-or-series fixture, index (timedelta or daily datetime of ten
    # points) and frequency (two days or one hour). Verifies the resampler's frequency-conversion
    # method equals reindexing the object on a regular range from the first to the last
    # label at that frequency.
    obj = frame_or_series(range(len(index)), index=index)
    idx_range = date_range if isinstance(index, DatetimeIndex) else timedelta_range

    result = obj.resample(freq).asfreq()
    new_index = idx_range(obj.index[0], obj.index[-1], freq=freq)
    expected = obj.reindex(new_index)
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),
    ],
)
def test_asfreq_fill_value(index):
    # [FUNCTION DOC]
    # Regression test (issue 3715), parametrized by timedelta or datetime index. A resampled
    # series converted to hourly frequency equals the reindex on the hourly range; then a frame
    # with a missing row (float cast to allow missing) converted with a fill value of 4.0 equals
    # the reindex with that fill value.
    # test for fill value during resampling, issue 3715

    ser = Series(range(len(index)), index=index, name="a")
    idx_range = date_range if isinstance(index, DatetimeIndex) else timedelta_range

    result = ser.resample("1h").asfreq()
    new_index = idx_range(ser.index[0], ser.index[-1], freq="1h")
    expected = ser.reindex(new_index)
    tm.assert_series_equal(result, expected)

    # Explicit cast to float to avoid implicit cast when setting None
    frame = ser.astype("float").to_frame("value")
    frame.iloc[1] = None
    result = frame.resample("1h").asfreq(fill_value=4.0)
    new_index = idx_range(frame.index[0], frame.index[-1], freq="1h")
    expected = frame.reindex(new_index, fill_value=4.0)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "3 day", freq="D"),
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 3), freq="D"),
        period_range(datetime(2005, 1, 1), datetime(2005, 1, 3), freq="D"),
    ],
)
def test_resample_interpolate(index):
    # [FUNCTION DOC]
    # Regression test (GH#12925), parametrized by timedelta, datetime and period three-point
    # indexes. Converting to one-minute frequency then interpolating equals the resampler's
    # interpolate directly.
    # GH#12925
    df = DataFrame(range(len(index)), index=index)
    result = df.resample("1min").asfreq().interpolate()
    expected = df.resample("1min").interpolate()
    tm.assert_frame_equal(result, expected)


def test_resample_interpolate_inplace_deprecated():
    # [FUNCTION DOC]
    # Regression test (GH#58690): the in-place keyword of the resampler's interpolate emits a
    # deprecation warning mentioning the keyword; setting it to true additionally raises a value
    # error saying a resampled object cannot be interpolated in place.
    # GH#58690
    dti = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")

    df = DataFrame(range(len(dti)), index=dti)
    rs = df.resample("1min")
    msg = "The 'inplace' keyword in DatetimeIndexResampler.interpolate"
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        rs.interpolate(inplace=False)

    msg2 = "Cannot interpolate inplace on a resampled object"
    with pytest.raises(ValueError, match=msg2):
        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            rs.interpolate(inplace=True)


def test_resample_interpolate_regular_sampling_off_grid(
    all_1d_no_arg_interpolation_methods,
):
    # [FUNCTION DOC]
    # Regression test (GH#21351), parametrized by interpolation method (fixture); skipped
    # without scipy. Resamples a series with two-hour spacing offset by one minute to hourly and
    # interpolates; the first value is missing and the rest depend on method: linear gives thirds
    # steps, nearest and zero repeat values, and the other methods give values offset
    # by a fixed constant plus half steps.
    pytest.importorskip("scipy")
    # GH#21351
    index = date_range("2000-01-01 00:01:00", periods=5, freq="2h")
    ser = Series(np.arange(5.0), index)

    method = all_1d_no_arg_interpolation_methods
    result = ser.resample("1h").interpolate(method)

    if method == "linear":
        values = np.repeat(np.arange(0.0, 4.0), 2) + np.tile([1 / 3, 2 / 3], 4)
    elif method == "nearest":
        values = np.repeat(np.arange(0.0, 5.0), 2)[1:-1]
    elif method == "zero":
        values = np.repeat(np.arange(0.0, 4.0), 2)
    else:
        values = 0.491667 + np.arange(0.0, 4.0, 0.5)
    values = np.insert(values, 0, np.nan)
    index = date_range("2000-01-01 00:00:00", periods=9, freq="1h")
    expected = Series(values, index=index)
    tm.assert_series_equal(result, expected)


def test_resample_interpolate_irregular_sampling(all_1d_no_arg_interpolation_methods):
    # [FUNCTION DOC]
    # Regression test (GH#21351), parametrized by interpolation method (fixture); skipped
    # without scipy. A series with irregular timestamps resampled to five seconds and interpolated
    # has a missing first value (three seconds before the first data point) and no other
    # missing values.
    pytest.importorskip("scipy")
    # GH#21351
    ser = Series(
        np.linspace(0.0, 1.0, 5),
        index=DatetimeIndex(
            [
                "2000-01-01 00:00:03",
                "2000-01-01 00:00:22",
                "2000-01-01 00:00:24",
                "2000-01-01 00:00:31",
                "2000-01-01 00:00:39",
            ]
        ),
    )

    # Resample to 5 second sampling and interpolate with the given method
    ser_resampled = ser.resample("5s").interpolate(all_1d_no_arg_interpolation_methods)

    # Check that none of the resampled values are NaN, except the first one
    # which lies 3 seconds before the first actual data point
    assert np.isnan(ser_resampled.iloc[0])
    assert not ser_resampled.iloc[1:].isna().any()


def test_raises_on_non_datetimelike_index():
    # [FUNCTION DOC]
    # Verifies that resampling an empty frame with a plain range index raises a type error saying
    # resampling is only valid with datetime, timedelta or period indexes and naming the range
    # index.
    # this is a non datetimelike index
    xp = DataFrame()
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'RangeIndex'"
    )
    with pytest.raises(TypeError, match=msg):
        xp.resample("YE")


@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="D", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_empty_series(freq, index, resample_method):
    # [FUNCTION DOC]
    # Regression test (GH12771, GH12868), parametrized by empty period/datetime/timedelta index,
    # frequency (month end, day, hour) and the resample method fixture. For a timedelta index
    # with month frequency, asserts a value error about requiring a fixed-duration frequency
    # and returns. For period indexes month end is converted to the period month frequency.
    # The reduction on the empty float series equals the series with its index converted to the
    # new frequency (open-high-low-close gives an empty frame with four named columns). Index
    # equality and frequency equality are also checked.
    # GH12771 & GH12868

    ser = Series(index=index, dtype=float)
    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"
    rs = ser.resample(freq)
    result = getattr(rs, resample_method)()

    if resample_method == "ohlc":
        expected = DataFrame(
            [], index=ser.index[:0], columns=["open", "high", "low", "close"]
        )
        expected.index = _asfreq_compat(ser.index, freq)
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = ser.copy()
        expected.index = _asfreq_compat(ser.index, freq)
        tm.assert_series_equal(result, expected, check_dtype=False)

    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq


@pytest.mark.parametrize("min_count", [0, 1])
def test_resample_empty_sum_string(string_dtype_no_object, min_count):
    # [FUNCTION DOC]
    # Regression test (issue 60229), parametrized by minimum count (zero, one) and the string
    # dtype fixture excluding object. A series of missing strings resampled to twenty seconds and
    # summed gives an empty string for each bin when the minimum count is zero and the missing
    # marker otherwise; the index has microsecond resolution.
    # https://github.com/pandas-dev/pandas/issues/60229
    dtype = string_dtype_no_object
    ser = Series(
        pd.NA,
        index=DatetimeIndex(
            [
                "2000-01-01 00:00:00",
                "2000-01-01 00:00:10",
                "2000-01-01 00:00:20",
                "2000-01-01 00:00:30",
            ]
        ),
        dtype=dtype,
    )
    rs = ser.resample("20s")
    result = rs.sum(min_count=min_count)

    value = "" if min_count == 0 else pd.NA
    index = date_range(start="2000-01-01", freq="20s", periods=2, unit="us")
    expected = Series(value, index=index, dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "freq",
    [
        pytest.param("ME", marks=pytest.mark.xfail(reason="Don't know why this fails")),
        "D",
        "h",
    ],
)
def test_resample_nat_index_series(freq, resample_method):
    # [FUNCTION DOC]
    # Regression test (GH39227), parametrized by frequency (month end expected to fail for an
    # unknown reason, day, hour) and the resample method fixture. A series with a period index of
    # all not-a-time values resampled and reduced gives an empty result with the right index
    # and frequency (open-high-low-close gives an empty frame).
    # GH39227

    ser = Series(range(5), index=PeriodIndex([NaT] * 5, freq=freq))

    rs = ser.resample(freq)
    result = getattr(rs, resample_method)()

    if resample_method == "ohlc":
        expected = DataFrame(
            [], index=ser.index[:0], columns=["open", "high", "low", "close"]
        )
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = ser[:0].copy()
        tm.assert_series_equal(result, expected, check_dtype=False)
    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq


@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="D", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
@pytest.mark.parametrize("resample_method", ["count", "size"])
def test_resample_count_empty_series(freq, index, resample_method):
    # [FUNCTION DOC]
    # Regression test (GH28427), parametrized by empty index types, frequency and count or
    # size. The result is an empty integer series on the index converted to the frequency; the
    # timedelta month-end combination raises a value error requiring a fixed-duration frequency,
    # and period month end is converted to the period month frequency.
    # GH28427
    ser = Series(index=index)
    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"
    rs = ser.resample(freq)

    result = getattr(rs, resample_method)()

    index = _asfreq_compat(ser.index, freq)

    expected = Series([], dtype="int64", index=index, name=ser.name)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index", [DatetimeIndex([]), TimedeltaIndex([]), PeriodIndex([], freq="D")]
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_empty_dataframe(index, freq, resample_method):
    # [FUNCTION DOC]
    # Regression test (GH13212), parametrized by empty index type, frequency and resample method
    # fixture. Empty frame resampled with group keys disabled and reduced: the expected value is
    # the frame itself (index converted), a zero-row frame with a two-level column index for
    # open-high-low-close, or an empty integer series for size (GH14962). Timedelta with month end
    # raises a value error; period month end is converted.
    # KNOWN ISSUE: the trailing comment about testing size for GH13212 (stays as a frame) is stale and contradicts the code, which expects an empty series for size.
    # GH13212
    df = DataFrame(index=index)
    # count retains dimensions too
    if freq == "ME" and isinstance(df.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            df.resample(freq, group_keys=False)
        return
    elif freq == "ME" and isinstance(df.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"
    rs = df.resample(freq, group_keys=False)
    result = getattr(rs, resample_method)()
    if resample_method == "ohlc":
        # TODO: no tests with len(df.columns) > 0
        mi = MultiIndex.from_product([df.columns, ["open", "high", "low", "close"]])
        expected = DataFrame([], index=df.index[:0], columns=mi, dtype=np.float64)
        expected.index = _asfreq_compat(df.index, freq)

    elif resample_method != "size":
        expected = df.copy()
    else:
        # GH14962
        expected = Series([], dtype=np.int64)

    expected.index = _asfreq_compat(df.index, freq)

    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq
    tm.assert_almost_equal(result, expected)

    # test size for GH13212 (currently stays as df)


@pytest.mark.parametrize(
    "index", [DatetimeIndex([]), TimedeltaIndex([]), PeriodIndex([], freq="D")]
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_count_empty_dataframe(freq, index):
    # [FUNCTION DOC]
    # Regression test (GH28427), parametrized by empty index type and frequency. Counting an
    # empty one-column frame gives an empty integer frame with the same column on the converted
    # index; timedelta with month end raises a value error and period month end is converted.
    # GH28427
    empty_frame_dti = DataFrame(index=index, columns=Index(["a"], dtype=object))

    if freq == "ME" and isinstance(empty_frame_dti.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return
    elif freq == "ME" and isinstance(empty_frame_dti.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"
    result = empty_frame_dti.resample(freq).count()

    index = _asfreq_compat(empty_frame_dti.index, freq)

    expected = DataFrame(dtype="int64", index=index, columns=Index(["a"], dtype=object))

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index", [DatetimeIndex([]), TimedeltaIndex([]), PeriodIndex([], freq="D")]
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_resample_size_empty_dataframe(freq, index):
    # [FUNCTION DOC]
    # Regression test (GH28427), parametrized by empty index type and frequency. The size of an
    # empty one-column frame resampled is an empty integer series on the converted index; the
    # same month-end special cases as in the count test apply.
    # GH28427

    empty_frame_dti = DataFrame(index=index, columns=Index(["a"], dtype=object))

    if freq == "ME" and isinstance(empty_frame_dti.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return
    elif freq == "ME" and isinstance(empty_frame_dti.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"
    result = empty_frame_dti.resample(freq).size()

    index = _asfreq_compat(empty_frame_dti.index, freq)

    expected = Series([], dtype="int64", index=index)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("index", [DatetimeIndex([]), TimedeltaIndex([])])
@pytest.mark.parametrize("freq", ["D", "h"])
@pytest.mark.parametrize(
    "method", ["ffill", "bfill", "nearest", "asfreq", "interpolate", "mean"]
)
def test_resample_apply_empty_dataframe(index, freq, method):
    # [FUNCTION DOC]
    # Regression test (GH#55572), parametrized by empty datetime or timedelta index, frequency
    # (day, hour) and method name (forward fill, backward fill, nearest, frequency conversion,
    # interpolate, mean). Applying the resampler's own method through the apply interface on an
    # empty frame gives an empty frame on the converted index.
    # GH#55572
    empty_frame_dti = DataFrame(index=index)

    rs = empty_frame_dti.resample(freq)
    result = rs.apply(getattr(rs, method))

    expected_index = _asfreq_compat(empty_frame_dti.index, freq)
    expected = DataFrame([], index=expected_index)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="M", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("dtype", [float, int, object, "datetime64[ns]"])
def test_resample_empty_dtypes(index, dtype, resample_method):
    # [FUNCTION DOC]
    # Regression test (GH #10228), parametrized by empty index types and dtypes (float, integer,
    # object, datetime). Resamples an empty series to daily and runs every resample method,
    # ignoring the data error raised for invalid dtype and method combinations; the test only
    # asserts that nothing crashes or raises an index error.
    # Empty series were sometimes causing a segfault (for the functions
    # with Cython bounds-checking disabled) or an IndexError.  We just run
    # them to ensure they no longer do.  (GH #10228)
    empty_series_dti = Series([], index, dtype)
    rs = empty_series_dti.resample("D", group_keys=False)
    try:
        getattr(rs, resample_method)()
    except DataError:
        # Ignore these since some combinations are invalid
        # (ex: doing mean with dtype of np.object_)
        pass


@pytest.mark.parametrize(
    "index",
    [
        PeriodIndex([], freq="D", name="a"),
        DatetimeIndex([], name="a"),
        TimedeltaIndex([], name="a"),
    ],
)
@pytest.mark.parametrize("freq", ["ME", "D", "h"])
def test_apply_to_empty_series(index, freq):
    # [FUNCTION DOC]
    # Regression test (GH 14313), parametrized by empty index types and frequency. Applying a
    # function that always returns one on an empty resampled series equals the sum result,
    # ignoring dtype. Timedelta with month end raises a value error; period month end is
    # converted.
    # GH 14313
    ser = Series(index=index)

    if freq == "ME" and isinstance(ser.index, TimedeltaIndex):
        msg = (
            "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
            "e.g. '24h' or '3D', not <MonthEnd>"
        )
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == "ME" and isinstance(ser.index, PeriodIndex):
        # index is PeriodIndex, so convert to corresponding Period freq
        freq = "M"
    result = ser.resample(freq, group_keys=False).apply(lambda x: 1)
    expected = ser.resample(freq).apply("sum")

    tm.assert_series_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),
        period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),
    ],
)
def test_resampler_is_iterable(index):
    # [FUNCTION DOC]
    # Regression test (GH 15314), parametrized by timedelta, datetime and period indexes.
    # Iterating a resampler to hourly frequency yields the same keys and groups as grouping the
    # series with a grouper of the same frequency.
    # GH 15314
    series = Series(range(len(index)), index=index)
    freq = "h"
    tg = Grouper(freq=freq, convention="start")
    grouped = series.groupby(tg)
    resampled = series.resample(freq)
    for (rk, rv), (gk, gv) in zip(resampled, grouped):
        assert rk == gk
        tm.assert_series_equal(rv, gv)


@pytest.mark.parametrize(
    "index",
    [
        timedelta_range("1 day", "10 day", freq="D"),
        date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),
        period_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D"),
    ],
)
def test_resample_quantile(index):
    # [FUNCTION DOC]
    # Regression test (GH 15023), parametrized by index type. The resampler's quantile at
    # 0.75 on hourly frequency equals aggregating each bin with a quantile function (renamed to
    # the series name).
    # GH 15023
    ser = Series(range(len(index)), index=index)
    q = 0.75
    freq = "h"

    result = ser.resample(freq).quantile(q)
    expected = ser.resample(freq).agg(lambda x: x.quantile(q)).rename(ser.name)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["first", "last"])
def test_first_last_skipna(any_real_nullable_dtype, skipna, how):
    # [FUNCTION DOC]
    # Regression test (GH#57019), parametrized by first or last, nullable numeric dtype fixture
    # and skipna fixture. Resamples a three-column frame to month end and compares first/last
    # with the grouped equivalent where all rows are placed in the month-end timestamp; the
    # expected index frequency is set to month end.
    # GH#57019
    if is_extension_array_dtype(any_real_nullable_dtype):
        na_value = Series(dtype=any_real_nullable_dtype).dtype.na_value
    else:
        na_value = np.nan
    df = DataFrame(
        {
            "a": [2, 1, 1, 2],
            "b": [na_value, 3.0, na_value, 4.0],
            "c": [na_value, 3.0, na_value, 4.0],
        },
        index=date_range("2020-01-01", periods=4, freq="D", unit="ns"),
        dtype=any_real_nullable_dtype,
    )
    rs = df.resample("ME")
    method = getattr(rs, how)
    result = method(skipna=skipna)

    ts = pd.to_datetime("2020-01-31").as_unit("ns")
    gb = df.groupby(df.shape[0] * [ts])
    expected = getattr(gb, how)(skipna=skipna)
    expected.index.freq = "ME"
    tm.assert_frame_equal(result, expected)
