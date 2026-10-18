# [MODULE DOC]
# Tests for the quantile-based binning function qcut: bin edges, label handling, duplicates handling, missing values,
# datetime, timedelta and time-zone-aware inputs, boolean coercion, nullable extension dtypes and containment of values in their buckets.
# Some tests use the unit fixture for datetime resolution and the datapath fixture for a bundled data file.

import os

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DatetimeIndex,
    Interval,
    IntervalIndex,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    cut,
    date_range,
    isna,
    qcut,
    timedelta_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype

from pandas.tseries.offsets import Day


def test_qcut():
    # [FUNCTION DOC]
    # For one thousand random normal values and four quantile bins, the left and right edges of the resulting categories
    # must approximately match the quantiles of the data (absolute tolerance of one hundredth), and the result must equal binning
    # the data by cut with those quantiles as edges including the lowest value.
    arr = np.random.default_rng(2).standard_normal(1000)

    # We store the bins as Index that have been
    # rounded to comparisons are a bit tricky.
    labels, _ = qcut(arr, 4, retbins=True)
    ex_bins = np.quantile(arr, [0, 0.25, 0.5, 0.75, 1.0])

    result = labels.categories.left.values
    assert np.allclose(result, ex_bins[:-1], atol=1e-2)

    result = labels.categories.right.values
    assert np.allclose(result, ex_bins[1:], atol=1e-2)

    ex_levels = cut(arr, ex_bins, include_lowest=True)
    tm.assert_categorical_equal(labels, ex_levels)


def test_qcut_bounds():
    # [FUNCTION DOC]
    # Requesting ten bins with labels turned off over a thousand random values must produce all ten distinct integer codes.
    arr = np.random.default_rng(2).standard_normal(1000)

    factor = qcut(arr, 10, labels=False)
    assert len(np.unique(factor)) == 10


def test_qcut_specify_quantiles():
    # [FUNCTION DOC]
    # Passing explicit quantile fractions (zero, quarter, half, three quarters, one) must be equivalent to asking for four bins.
    arr = np.random.default_rng(2).standard_normal(100)
    factor = qcut(arr, [0, 0.25, 0.5, 0.75, 1.0])

    expected = qcut(arr, 4)
    tm.assert_categorical_equal(factor, expected)


def test_qcut_all_bins_same():
    # [FUNCTION DOC]
    # Ten identical values cannot be split into three bins; must raise a value error saying bin edges must be unique.
    with pytest.raises(ValueError, match="edges.*unique"):
        qcut([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3)


def test_qcut_include_lowest():
    # [FUNCTION DOC]
    # Ten consecutive integers split into four bins give categories of right-closed intervals whose first left edge is slightly
    # below the minimum so that the lowest value is included.
    values = np.arange(10)
    ii = qcut(values, 4)

    ex_levels = IntervalIndex(
        [
            Interval(-0.001, 2.25),
            Interval(2.25, 4.5),
            Interval(4.5, 6.75),
            Interval(6.75, 9),
        ]
    )
    tm.assert_index_equal(ii.categories, ex_levels)


def test_qcut_nas():
    # [FUNCTION DOC]
    # With the first twenty of one hundred values set to missing, every one of the first twenty results must be missing.
    arr = np.random.default_rng(2).standard_normal(100)
    arr[:20] = np.nan

    result = qcut(arr, 4)
    assert isna(result[:20]).all()


def test_qcut_index():
    # [FUNCTION DOC]
    # A two-element list split in two bins returns an ordered categorical of two intervals, the first starting just below the minimum.
    result = qcut([0, 2], 2)
    intervals = [Interval(-0.001, 1), Interval(1, 2)]

    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)


def test_qcut_binning_issues(datapath):
    # [FUNCTION DOC]
    # gh-1978, gh-1979. Reads values from a bundled csv data file (located with the datapath fixture) and bins them into twenty
    # quantiles. For every bin the left edge must be below the right edge, and each bin must start no later than the previous bin ends.
    # see gh-1978, gh-1979
    cut_file = datapath(os.path.join("reshape", "data", "cut_data.csv"))
    arr = np.loadtxt(cut_file)
    result = qcut(arr, 20)

    starts = result.categories.left
    ends = result.categories.right
    assert (starts < ends).all()
    assert (starts[1:] <= ends[:-1]).all()


def test_qcut_return_intervals():
    # [FUNCTION DOC]
    # Binning a series of nine integers by quantile fractions at one third and two thirds returns a series (not a categorical)
    # of ordered categorical dtype whose values are three intervals repeated three times each.
    ser = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    res = qcut(ser, [0, 0.333, 0.666, 1])

    exp_levels = np.array(
        [Interval(-0.001, 2.664), Interval(2.664, 5.328), Interval(5.328, 8)]
    )
    exp = Series(exp_levels.take([0, 0, 0, 1, 1, 1, 2, 2, 2])).astype(
        CategoricalDtype(ordered=True)
    )
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize("labels", ["foo", 1, True])
def test_qcut_incorrect_labels(labels):
    # [FUNCTION DOC]
    # GH 13318. Labels given as a string, a number or a boolean true value are invalid: must raise a value error saying labels must be
    # false, none, or a list-like argument.
    # GH 13318
    values = range(5)
    msg = "Bin labels must either be False, None or passed in as a list-like argument"
    with pytest.raises(ValueError, match=msg):
        qcut(values, 4, labels=labels)


@pytest.mark.parametrize("labels", [["a", "b", "c"], list(range(3))])
def test_qcut_wrong_length_labels(labels):
    # [FUNCTION DOC]
    # GH 13318. Four bins are requested with only three labels (strings or integers);
    # this must raise a value error saying labels must be one fewer than the number of bin edges.
    # GH 13318
    values = range(10)
    msg = "Bin labels must be one fewer than the number of bin edges"
    with pytest.raises(ValueError, match=msg):
        qcut(values, 4, labels=labels)


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (list(range(3)), [0, 1, 2]),
    ],
)
def test_qcut_list_like_labels(labels, expected):
    # [FUNCTION DOC]
    # GH 13318. Three values in three bins with a list of three labels (strings or integers) must return an ordered categorical
    # whose values are the labels.
    # GH 13318
    values = range(3)
    result = qcut(values, 3, labels=labels)
    expected = Categorical(expected, ordered=True)
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"duplicates": "drop"}, None),
        ({}, "Bin edges must be unique"),
        ({"duplicates": "raise"}, "Bin edges must be unique"),
        ({"duplicates": "foo"}, "invalid value for 'duplicates' parameter"),
    ],
)
def test_qcut_duplicates_bin(kwargs, msg):
    # [FUNCTION DOC]
    # see gh-7751. Values with many repeated zeros need duplicate edges. Parametrized over keyword arguments and an expected message:
    # dropping duplicates gives two intervals (starting just below zero, up to one, then to three); the default or raise option must
    # raise a value error saying bin edges must be unique; an unknown option must raise a value error about an invalid duplicates parameter.
    # see gh-7751
    values = [0, 0, 0, 0, 1, 2, 3]

    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            qcut(values, 3, **kwargs)
    else:
        result = qcut(values, 3, **kwargs)
        expected = IntervalIndex([Interval(-0.001, 1), Interval(1, 3)])
        tm.assert_index_equal(result.categories, expected)


@pytest.mark.parametrize(
    "data,start,end", [(9.0, 8.999, 9.0), (0.0, -0.001, 0.0), (-9.0, -9.001, -9.0)]
)
@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("labels", [None, False])
def test_single_quantile(data, start, end, length, labels):
    # [FUNCTION DOC]
    # see gh-15431. Parametrized over a data value with its expected interval endpoints, a series length of one or two, and
    # labels unset or turned off. Binning a series of repeated identical values into a single quantile returns either an ordered categorical
    # of the identical right-closed interval, or an array of integer zeros of the platform pointer-size integer dtype.
    # see gh-15431
    ser = Series([data] * length)
    result = qcut(ser, 1, labels=labels)

    if labels is None:
        intervals = IntervalIndex([Interval(start, end)] * length, closed="right")
        expected = Series(intervals).astype(CategoricalDtype(ordered=True))
    else:
        expected = Series([0] * length, dtype=np.intp)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser",
    [
        DatetimeIndex(["20180101", NaT, "20180103"]),
        TimedeltaIndex(["0 days", NaT, "2 days"]),
    ],
    ids=lambda x: str(x.dtype),
)
def test_qcut_nat(ser, unit):
    # [FUNCTION DOC]
    # see gh-19768. Parametrized over a datetime index and a timedelta index, each with one missing value, and the unit fixture. Converts the
    # series to the unit and bins in two quantiles; the result must be an ordered categorical series of intervals with missing in the middle, whose
    # edges are the first value minus one unit, the last value minus one day, and the last value, in the same dtype as the input.
    # see gh-19768
    ser = Series(ser)
    ser = ser.dt.as_unit(unit)
    td = Timedelta(1, unit=unit).as_unit(unit)

    left = Series([ser[0] - td, np.nan, ser[2] - Day()], dtype=ser.dtype)
    right = Series([ser[2] - Day(), np.nan, ser[2]], dtype=ser.dtype)
    intervals = IntervalIndex.from_arrays(left, right)
    expected = Series(Categorical(intervals, ordered=True))

    result = qcut(ser, 2)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("bins", [3, np.linspace(0, 1, 4)])
def test_datetime_tz_qcut(bins):
    # [FUNCTION DOC]
    # see gh-19872. A series of three daily time-zone-aware timestamps (US Eastern) is binned with either three bins or the
    # fractions zero, one third, two thirds, one. The result must equal an ordered categorical series of three timestamp intervals, with the
    # first left edge one nanosecond before the minimum.
    # see gh-19872
    tz = "US/Eastern"
    ser = Series(date_range("20130101", periods=3, tz=tz, unit="ns"))

    result = qcut(ser, bins)
    expected = Series(
        IntervalIndex(
            [
                Interval(
                    Timestamp("2012-12-31 23:59:59.999999999", tz=tz),
                    Timestamp("2013-01-01 16:00:00", tz=tz),
                ),
                Interval(
                    Timestamp("2013-01-01 16:00:00", tz=tz),
                    Timestamp("2013-01-02 08:00:00", tz=tz),
                ),
                Interval(
                    Timestamp("2013-01-02 08:00:00", tz=tz),
                    Timestamp("2013-01-03 00:00:00", tz=tz),
                ),
            ]
        )
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arg,expected_bins",
    [
        [
            timedelta_range("1day", periods=3),
            TimedeltaIndex(["1 days", "2 days", "3 days"]),
        ],
        [
            date_range("20180101", periods=3),
            DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-03"]),
        ],
    ],
)
def test_date_like_qcut_bins(arg, expected_bins, unit):
    # [FUNCTION DOC]
    # see gh-19891. Parametrized over a timedelta range and a date range with expected bin edges and the unit fixture. Both inputs
    # and expected edges are converted to the unit. Binning the series in two quantiles with the bins returned must give bin edges equal to
    # the expected index (the quantiles of the three evenly spaced values).
    # see gh-19891
    arg = arg.as_unit(unit)
    expected_bins = expected_bins.as_unit(unit)
    ser = Series(arg)
    result, result_bins = qcut(ser, 2, retbins=True)
    tm.assert_index_equal(result_bins, expected_bins)


@pytest.mark.parametrize("bins", [6, 7])
@pytest.mark.parametrize(
    "box, compare",
    [
        (Series, tm.assert_series_equal),
        (np.array, tm.assert_categorical_equal),
        (list, tm.assert_equal),
    ],
)
def test_qcut_bool_coercion_to_int(bins, box, compare):
    # [FUNCTION DOC]
    # issue 20303. Parametrized over six or seven bins and three container kinds (series, array, list) each with its matching comparison
    # helper. Qcut applied to a boolean sequence with duplicate edges dropped must give the same result as qcut on the equivalent zero/one integers.
    # issue 20303
    data_expected = box([0, 1, 1, 0, 1] * 10)
    data_result = box([False, True, True, False, True] * 10)
    expected = qcut(data_expected, bins, duplicates="drop")
    result = qcut(data_result, bins, duplicates="drop")
    compare(result, expected)


@pytest.mark.parametrize("q", [2, 5, 10])
def test_qcut_nullable_integer(q, any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Parametrized over q in two, five and ten and the any numeric extension dtype fixture. A hundred-element nullable array with every other element
    # missing is binned; the result must equal the one obtained from the float conversion of the array.
    arr = pd.array(np.arange(100), dtype=any_numeric_ea_dtype)
    arr[::2] = pd.NA

    result = qcut(arr, q)
    expected = qcut(arr.astype(float), q)

    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize("scale", [1.0, 1 / 3, 17.0])
@pytest.mark.parametrize("q", [3, 7, 9])
@pytest.mark.parametrize("precision", [1, 3, 16])
def test_qcut_contains(scale, q, precision):
    # [FUNCTION DOC]
    # GH-59355. Parametrized over a scale, a number of bins q and a rounding precision. Builds q plus one evenly spaced values scaled
    # and rounded to the precision, bins them in q quantiles at that label precision, and asserts that every value lies inside the bucket it was assigned to.
    # GH-59355
    arr = (scale * np.arange(q + 1)).round(precision)
    result = qcut(arr, q, precision=precision)

    for value, bucket in zip(arr, result):
        assert value in bucket
