# [MODULE DOC]
# Tests for exponentially weighted windows (the ewm accessor): constructor
# argument validation (center of mass, span, half-life and alpha being mutually
# exclusive and range-checked), the times argument and time-based half-life,
# unsupported combinations (raising not implemented errors), mean, sum,
# variance, standard deviation, covariance and correlation values, missing
# value handling with adjust and ignore_na, minimum periods, and numeric-only
# handling. Uses frame_or_series, adjust, ignore_na, halflife_with_times,
# series, frame and arithmetic_win_operators fixtures.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm


def test_doc_string():
    # [FUNCTION DOC]
    # Smoke test of the documentation example: builds a one-column frame with a
    # missing value and computes an ewm mean with center of mass 0.5. Asserts
    # nothing; passes if no exception is raised.
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.ewm(com=0.5).mean()


def test_constructor(frame_or_series):
    # [FUNCTION DOC]
    # Parametrized over series and frame. Verifies ewm argument validation: each of
    # the four decay parameters alone is accepted (also with the others explicitly
    # none); giving two at once raises a value error saying they are mutually
    # exclusive; a negative center of mass, span below 1, half-life of zero, and
    # alpha at or below 0 or above 1 each raise a value error with the matching
    # "must satisfy" message.
    c = frame_or_series(range(5)).ewm

    # valid
    c(com=0.5)
    c(span=1.5)
    c(alpha=0.5)
    c(halflife=0.75)
    c(com=0.5, span=None)
    c(alpha=0.5, com=None)
    c(halflife=0.75, alpha=None)

    # not valid: mutually exclusive
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        c(com=0.5, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        c(span=1.5, halflife=0.75)
    with pytest.raises(ValueError, match=msg):
        c(alpha=0.5, span=1.5)

    # not valid: com < 0
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):
        c(com=-0.5)

    # not valid: span < 1
    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):
        c(span=0.5)

    # not valid: halflife <= 0
    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):
        c(halflife=0)

    # not valid: alpha <= 0 or alpha > 1
    msg = "alpha must satisfy: 0 < alpha <= 1"
    for alpha in (-0.5, 1.5):
        with pytest.raises(ValueError, match=msg):
            c(alpha=alpha)


def test_ewma_times_not_datetime_type():
    # [FUNCTION DOC]
    # Verifies passing a times array that is not datetime typed raises a value
    # error saying times must be datetime64 dtype.
    msg = r"times must be datetime64 dtype."
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(times=np.arange(5))


def test_ewma_times_not_same_length():
    # [FUNCTION DOC]
    # Verifies passing a datetime times array shorter than the object raises a value
    # error saying times must be the same length as the object.
    msg = "times must be the same length as the object."
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(times=np.arange(4).astype("datetime64[ns]"))


def test_ewma_halflife_not_correct_type():
    # [FUNCTION DOC]
    # Verifies passing times together with a numeric half-life raises a value error
    # saying the half-life must be convertible to a time duration.
    msg = "halflife must be a timedelta convertible object"
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(halflife=1, times=np.arange(5).astype("datetime64[ns]"))


def test_ewma_halflife_without_times(halflife_with_times):
    # [FUNCTION DOC]
    # Verifies passing a duration-like half-life (each variant of the fixture)
    # without times raises a value error saying a duration half-life requires times.
    msg = "halflife can only be a timedelta convertible argument if times is not None."
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(halflife=halflife_with_times)


@pytest.mark.parametrize(
    "times",
    [
        np.arange(10).astype("datetime64[D]").astype("datetime64[ns]"),
        date_range("2000", freq="D", periods=10),
        date_range("2000", freq="D", periods=10).tz_localize("UTC"),
    ],
)
@pytest.mark.parametrize("min_periods", [0, 2])
def test_ewma_with_times_equal_spacing(halflife_with_times, times, min_periods):
    # [FUNCTION DOC]
    # Parametrized over three equivalent daily time arrays (numpy, naive datetime
    # index, tz-aware datetime index), minimum periods 0 and 2, and half-life
    # duration variants of one day. With every other value missing, verifies the
    # mean with times and a one-day half-life equals the mean without times and a
    # numeric half-life of 1.0, because equal daily spacing is equivalent to unit
    # steps.
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    df = DataFrame({"A": data})
    result = df.ewm(halflife=halflife, min_periods=min_periods, times=times).mean()
    expected = df.ewm(halflife=1.0, min_periods=min_periods).mean()
    tm.assert_frame_equal(result, expected)


def test_ewma_with_times_variable_spacing(tz_aware_fixture, unit, adjust):
    # [FUNCTION DOC]
    # Regression test for GH 54328. Uses tz-aware fixtures, a time unit fixture and
    # the adjust fixture. Three irregular timestamps with a 23 days half-life: the
    # mean of 0, 1, 2 equals precomputed values, which differ for adjust true and
    # false.
    # GH 54328
    tz = tz_aware_fixture
    halflife = "23 days"
    times = (
        DatetimeIndex(["2020-01-01", "2020-01-10T00:04:05", "2020-02-23T05:00:23"])
        .tz_localize(tz)
        .as_unit(unit)
    )
    data = np.arange(3)
    df = DataFrame(data)
    result = df.ewm(halflife=halflife, times=times, adjust=adjust).mean()
    if adjust:
        expected = DataFrame([0.0, 0.5674161888241773, 1.545239952073459])
    else:
        expected = DataFrame([0.0, 0.23762518642226227, 1.534926369128742])
    tm.assert_frame_equal(result, expected)


def test_ewm_with_nat_raises(halflife_with_times):
    # [FUNCTION DOC]
    # Regression test for GH#38535: times containing a missing timestamp makes the
    # constructor raise a value error saying NaT values cannot be converted to
    # integer.
    # GH#38535
    ser = Series(range(1))
    times = DatetimeIndex(["NaT"])
    with pytest.raises(ValueError, match="Cannot convert NaT values to integer"):
        ser.ewm(com=0.1, halflife=halflife_with_times, times=times)


def test_ewm_with_times_getitem(halflife_with_times):
    # [FUNCTION DOC]
    # Regression test for GH 40164. Verifies selecting a column after creating a
    # time-based ewm keeps the times and half-life: the mean of column A with a
    # daily half-life of one day equals the mean without times and half-life 1.0.
    # GH 40164
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    times = date_range("2000", freq="D", periods=10)
    df = DataFrame({"A": data, "B": data})
    result = df.ewm(halflife=halflife, times=times)["A"].mean()
    expected = df.ewm(halflife=1.0)["A"].mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("arg", ["com", "halflife", "span", "alpha"])
def test_ewm_getitem_attributes_retained(arg, adjust, ignore_na):
    # [FUNCTION DOC]
    # Regression test for GH 40164. Parametrized over which decay argument is set
    # (center of mass, half-life, span, alpha). Intends to verify that selecting a
    # column keeps all of the ewm object's declared attributes.
    # KNOWN ISSUE: the result mapping is built by reading attributes from the original ewm object rather than from the sliced one, so the final assertion compares the original object to itself and cannot detect lost attributes.
    # GH 40164
    kwargs = {arg: 1, "adjust": adjust, "ignore_na": ignore_na}
    ewm = DataFrame({"A": range(1), "B": range(1)}).ewm(**kwargs)
    expected = {attr: getattr(ewm, attr) for attr in ewm._attributes}
    ewm_slice = ewm["A"]
    result = {attr: getattr(ewm, attr) for attr in ewm_slice._attributes}
    assert result == expected


def test_ewma_times_adjust_false_with_disallowed_com():
    # [FUNCTION DOC]
    # Regression test for GH 54328: times with adjust false and an explicit center
    # of mass raises a not implemented error saying none of center of mass, span or
    # alpha may be given in that case.
    # GH 54328
    with pytest.raises(
        NotImplementedError,
        match=(
            "None of com, span, or alpha can be specified "
            "if times is provided and adjust=False"
        ),
    ):
        Series(range(1)).ewm(
            0.1,
            adjust=False,
            times=date_range("2000", freq="D", periods=1),
            halflife="1D",
        )


def test_ewma_times_adjust_false_with_disallowed_alpha():
    # [FUNCTION DOC]
    # Regression test for GH 54328: same not implemented error when alpha is given
    # with times, adjust false and a half-life.
    # GH 54328
    with pytest.raises(
        NotImplementedError,
        match=(
            "None of com, span, or alpha can be specified "
            "if times is provided and adjust=False"
        ),
    ):
        Series(range(1)).ewm(
            0.1,
            adjust=False,
            times=date_range("2000", freq="D", periods=1),
            alpha=0.5,
            halflife="1D",
        )


def test_ewma_times_adjust_false_with_disallowed_span():
    # [FUNCTION DOC]
    # Regression test for GH 54328: same not implemented error when span is given
    # with times, adjust false and a half-life.
    # GH 54328
    with pytest.raises(
        NotImplementedError,
        match=(
            "None of com, span, or alpha can be specified "
            "if times is provided and adjust=False"
        ),
    ):
        Series(range(1)).ewm(
            0.1,
            adjust=False,
            times=date_range("2000", freq="D", periods=1),
            span=10,
            halflife="1D",
        )


def test_times_string_col_raises():
    # [FUNCTION DOC]
    # Regression test for GH 43265: passing a column name as times raises a value
    # error saying times must be datetime64 type.
    # GH 43265
    df = DataFrame(
        {"A": np.arange(10.0), "time_col": date_range("2000", freq="D", periods=10)}
    )
    with pytest.raises(ValueError, match="times must be datetime64"):
        df.ewm(halflife="1 day", min_periods=0, times="time_col")


def test_ewm_sum_adjust_false_notimplemented():
    # [FUNCTION DOC]
    # Verifies the ewm sum with adjust false raises a not implemented error saying
    # sum is not supported.
    data = Series(range(1)).ewm(com=1, adjust=False)
    with pytest.raises(NotImplementedError, match="sum is not"):
        data.sum()


@pytest.mark.parametrize("method", ["sum", "std", "var", "cov", "corr"])
def test_times_only_mean_implemented(frame_or_series, method):
    # [FUNCTION DOC]
    # Regression test for GH 51695. Parametrized over series/frame and the methods
    # sum, std, var, cov, corr. Verifies each raises a not implemented error saying
    # the method is not implemented with times (only mean is).
    # GH 51695
    halflife = "1 day"
    times = date_range("2000", freq="D", periods=10)
    ewm = frame_or_series(range(10)).ewm(halflife=halflife, times=times)
    with pytest.raises(
        NotImplementedError, match=f"{method} is not implemented with times"
    ):
        getattr(ewm, method)()


@pytest.mark.parametrize(
    "expected_data, ignore",
    [[[10.0, 5.0, 2.5, 11.25], False], [[10.0, 5.0, 5.0, 12.5], True]],
)
def test_ewm_sum(expected_data, ignore):
    # [FUNCTION DOC]
    # Parametrized over ignore_na false and true with expected values. Verifies ewm
    # sum with alpha 0.5 on data with a missing value matches numbagg reference
    # values (expected to differ at the missing position depending on ignore_na).
    # xref from Numbagg tests
    # https://github.com/numbagg/numbagg/blob/v0.2.1/numbagg/test/test_moving.py#L50
    data = Series([10, 0, np.nan, 10])
    result = data.ewm(alpha=0.5, ignore_na=ignore).sum()
    expected = Series(expected_data)
    tm.assert_series_equal(result, expected)


def test_ewma_adjust():
    # [FUNCTION DOC]
    # Verifies that with span 100 and adjust false, the total of the ewm mean of an
    # impulse series (a single 1 in 1000 zeros) is close to 1 (tolerance 0.01).
    vals = Series(np.zeros(1000))
    vals[5] = 1
    result = vals.ewm(span=100, adjust=False).mean().sum()
    assert np.abs(result - 1) < 1e-2


def test_ewma_cases(adjust, ignore_na):
    # [FUNCTION DOC]
    # Verifies ewm mean with center of mass 2 on four values equals the expected
    # numbers for adjust true and false; ignore_na has no effect without missing
    # values, so both values give the same result.
    # try adjust/ignore_na args matrix

    s = Series([1.0, 2.0, 4.0, 8.0])

    if adjust:
        expected = Series([1.0, 1.6, 2.736842, 4.923077])
    else:
        expected = Series([1.0, 1.333333, 2.222222, 4.148148])

    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()
    tm.assert_series_equal(result, expected)


def test_ewma_nan_handling():
    # [FUNCTION DOC]
    # Verifies ewm mean with center of mass 5 carries the previous value over
    # missing entries and stays missing before the first valid value.
    s = Series([1.0] + [np.nan] * 5 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([1.0] * len(s)))

    s = Series([np.nan] * 2 + [1.0] + [np.nan] * 2 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([np.nan] * 2 + [1.0] * 4))


@pytest.mark.parametrize(
    "s, adjust, ignore_na, w",
    [
        (
            [np.nan, 1.0, 101.0],
            True,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), 1.0],
        ),
        (
            [np.nan, 1.0, 101.0],
            True,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), 1.0],
        ),
        (
            [np.nan, 1.0, 101.0],
            False,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), (1.0 / (1.0 + 2.0))],
        ),
        (
            [np.nan, 1.0, 101.0],
            False,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), (1.0 / (1.0 + 2.0))],
        ),
        (
            [1.0, np.nan, 101.0],
            True,
            False,
            [(1.0 - (1.0 / (1.0 + 2.0))) ** 2, np.nan, 1.0],
        ),
        (
            [1.0, np.nan, 101.0],
            True,
            True,
            [(1.0 - (1.0 / (1.0 + 2.0))), np.nan, 1.0],
        ),
        (
            [1.0, np.nan, 101.0],
            False,
            False,
            [(1.0 - (1.0 / (1.0 + 2.0))) ** 2, np.nan, (1.0 / (1.0 + 2.0))],
        ),
        (
            [1.0, np.nan, 101.0],
            False,
            True,
            [(1.0 - (1.0 / (1.0 + 2.0))), np.nan, (1.0 / (1.0 + 2.0))],
        ),
        (
            [np.nan, 1.0, np.nan, np.nan, 101.0, np.nan],
            True,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))) ** 3, np.nan, np.nan, 1.0, np.nan],
        ),
        (
            [np.nan, 1.0, np.nan, np.nan, 101.0, np.nan],
            True,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), np.nan, np.nan, 1.0, np.nan],
        ),
        (
            [np.nan, 1.0, np.nan, np.nan, 101.0, np.nan],
            False,
            False,
            [
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                np.nan,
                (1.0 / (1.0 + 2.0)),
                np.nan,
            ],
        ),
        (
            [np.nan, 1.0, np.nan, np.nan, 101.0, np.nan],
            False,
            True,
            [
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                np.nan,
                np.nan,
                (1.0 / (1.0 + 2.0)),
                np.nan,
            ],
        ),
        (
            [1.0, np.nan, 101.0, 50.0],
            True,
            False,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                1.0,
            ],
        ),
        (
            [1.0, np.nan, 101.0, 50.0],
            True,
            True,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 2,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                1.0,
            ],
        ),
        (
            [1.0, np.nan, 101.0, 50.0],
            False,
            False,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) * (1.0 / (1.0 + 2.0)),
                (1.0 / (1.0 + 2.0))
                * ((1.0 - (1.0 / (1.0 + 2.0))) ** 2 + (1.0 / (1.0 + 2.0))),
            ],
        ),
        (
            [1.0, np.nan, 101.0, 50.0],
            False,
            True,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 2,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) * (1.0 / (1.0 + 2.0)),
                (1.0 / (1.0 + 2.0)),
            ],
        ),
    ],
)
def test_ewma_nan_handling_cases(s, adjust, ignore_na, w):
    # [FUNCTION DOC]
    # Regression test for GH 7603. Parametrized over data containing missing values
    # with adjust, ignore_na and the corresponding relative weights w (derived from
    # the decay factor for center of mass 2). Verifies the ewm mean equals the
    # cumulative weighted sum divided by the cumulative weight, forward filled. When
    # ignore_na is false, also verifies that omitting ignore_na gives the same
    # result since false is the default.
    # GH 7603
    s = Series(s)
    expected = (s.multiply(w).cumsum() / Series(w).cumsum()).ffill()
    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()

    tm.assert_series_equal(result, expected)
    if ignore_na is False:
        # check that ignore_na defaults to False
        result = s.ewm(com=2.0, adjust=adjust).mean()
        tm.assert_series_equal(result, expected)


def test_ewm_alpha():
    # [FUNCTION DOC]
    # Regression test for GH 10789. Verifies four equivalent parameterizations
    # (alpha, center of mass, span, half-life with matching values) give the same
    # mean on data with a block of missing values.
    # GH 10789
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan

    s = Series(arr)
    a = s.ewm(alpha=0.61722699889169674).mean()
    b = s.ewm(com=0.62014947789973052).mean()
    c = s.ewm(span=2.240298955799461).mean()
    d = s.ewm(halflife=0.721792864318).mean()
    tm.assert_series_equal(a, b)
    tm.assert_series_equal(a, c)
    tm.assert_series_equal(a, d)


def test_ewm_domain_checks():
    # [FUNCTION DOC]
    # Regression test for GH 12492. Verifies the accepted ranges: center of mass at
    # or above 0, span at or above 1, half-life above 0, alpha above 0 and at most 1;
    # values outside raise value errors with the matching messages, boundary values
    # are accepted.
    # GH 12492
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan

    s = Series(arr)
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=-0.1)
    s.ewm(com=0.0)
    s.ewm(com=0.1)

    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.9)
    s.ewm(span=1.0)
    s.ewm(span=1.1)

    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=0.0)
    s.ewm(halflife=0.1)

    msg = "alpha must satisfy: 0 < alpha <= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=0.0)
    s.ewm(alpha=0.1)
    s.ewm(alpha=1.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=1.1)


@pytest.mark.parametrize("method", ["mean", "std", "var"])
def test_ew_empty_series(method):
    # [FUNCTION DOC]
    # Parametrized over mean, std and var. Verifies the ewm result on an empty float
    # series is the empty series.
    vals = Series([], dtype=np.float64)

    ewm = vals.ewm(3)
    result = getattr(ewm, method)()
    tm.assert_almost_equal(result, vals)


@pytest.mark.parametrize("min_periods", [0, 1])
@pytest.mark.parametrize("name", ["mean", "var", "std"])
def test_ew_min_periods(min_periods, name):
    # [FUNCTION DOC]
    # Parametrized over minimum periods 0 and 1 and the method mean, var, std. On
    # 50 random values with ten missing at each end, verifies: with minimum periods
    # 2 the first 11 results are missing and the remaining are not; with the
    # parametrized value, mean is missing for the first 10 and non-missing after,
    # while var and std need at least two values (first 11 missing). Also checks
    # an empty series returns an empty float series, a length one series gives 1.0
    # for mean and missing for var and std, and integer input yields float64.
    # excluding NaNs correctly
    arr = np.random.default_rng(2).standard_normal(50)
    arr[:10] = np.nan
    arr[-10:] = np.nan
    s = Series(arr)

    # check min_periods
    # GH 7898
    result = getattr(s.ewm(com=50, min_periods=2), name)()
    assert result[:11].isna().all()
    assert not result[11:].isna().any()

    result = getattr(s.ewm(com=50, min_periods=min_periods), name)()
    if name == "mean":
        assert result[:10].isna().all()
        assert not result[10:].isna().any()
    else:
        # ewm.std, ewm.var (with bias=False) require at least
        # two values
        assert result[:11].isna().all()
        assert not result[11:].isna().any()

    # check series of length 0
    result = getattr(Series(dtype=object).ewm(com=50, min_periods=min_periods), name)()
    tm.assert_series_equal(result, Series(dtype="float64"))

    # check series of length 1
    result = getattr(Series([1.0]).ewm(50, min_periods=min_periods), name)()
    if name == "mean":
        tm.assert_series_equal(result, Series([1.0]))
    else:
        # ewm.std, ewm.var with bias=False require at least
        # two values
        tm.assert_series_equal(result, Series([np.nan]))

    # pass in ints
    result2 = getattr(Series(np.arange(50)).ewm(span=10), name)()
    assert result2.dtype == np.float64


@pytest.mark.parametrize("name", ["cov", "corr"])
def test_ewm_corr_cov(name):
    # [FUNCTION DOC]
    # Parametrized over cov and corr. With two partially overlapping random series
    # (the second is the first without its first two items plus noise, with missing
    # values at the start of the first and end of the second), verifies results are
    # missing for the first 14 entries and non-missing afterwards with minimum
    # periods 5.
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)

    A[:10] = np.nan
    B.iloc[-10:] = np.nan

    result = getattr(A.ewm(com=20, min_periods=5), name)(B)
    assert np.isnan(result.values[:14]).all()
    assert not np.isnan(result.values[14:]).any()


@pytest.mark.parametrize("min_periods", [0, 1, 2])
@pytest.mark.parametrize("name", ["cov", "corr"])
def test_ewm_corr_cov_min_periods(name, min_periods):
    # [FUNCTION DOC]
    # Regression test for GH 7898. Parametrized over cov and corr and minimum
    # periods 0, 1, 2. Verifies the first 11 results are missing and the rest are
    # not (binary functions need at least two overlapping values), that empty
    # inputs give an empty result, and that single-element inputs give missing.
    # GH 7898
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)

    A[:10] = np.nan
    B.iloc[-10:] = np.nan

    result = getattr(A.ewm(com=20, min_periods=min_periods), name)(B)
    # binary functions (ewmcov, ewmcorr) with bias=False require at
    # least two values
    assert np.isnan(result.values[:11]).all()
    assert not np.isnan(result.values[11:]).any()

    # check series of length 0
    empty = Series([], dtype=np.float64)
    result = getattr(empty.ewm(com=50, min_periods=min_periods), name)(empty)
    tm.assert_series_equal(result, empty)

    # check series of length 1
    result = getattr(Series([1.0]).ewm(com=50, min_periods=min_periods), name)(
        Series([1.0])
    )
    tm.assert_series_equal(result, Series([np.nan]))


@pytest.mark.parametrize("name", ["cov", "corr"])
def test_different_input_array_raise_exception(name):
    # [FUNCTION DOC]
    # Parametrized over cov and corr. Verifies passing a plain numpy array as the
    # other operand raises a value error saying other must be a frame or series.
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    A[:10] = np.nan

    msg = "other must be a DataFrame or Series"
    # exception raised is Exception
    with pytest.raises(ValueError, match=msg):
        getattr(A.ewm(com=20, min_periods=5), name)(
            np.random.default_rng(2).standard_normal(50)
        )


@pytest.mark.parametrize("name", ["var", "std", "mean"])
def test_ewma_series(series, name):
    # [FUNCTION DOC]
    # Parametrized over var, std, mean. Verifies ewm with center of mass 10 on the
    # series fixture returns a series.
    series_result = getattr(series.ewm(com=10), name)()
    assert isinstance(series_result, Series)


@pytest.mark.parametrize("name", ["var", "std", "mean"])
def test_ewma_frame(frame, name):
    # [FUNCTION DOC]
    # Parametrized over var, std, mean. Verifies ewm with center of mass 10 on the
    # frame fixture returns a frame.
    frame_result = getattr(frame.ewm(com=10), name)()
    assert isinstance(frame_result, DataFrame)


def test_ewma_span_com_args(series):
    # [FUNCTION DOC]
    # Verifies center of mass 9.5 and span 20 give almost equal means, giving both
    # raises a value error that they are mutually exclusive, and giving none raises
    # a value error that one must be passed.
    A = series.ewm(com=9.5).mean()
    B = series.ewm(span=20).mean()
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20)

    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        series.ewm().mean()


def test_ewma_halflife_arg(series):
    # [FUNCTION DOC]
    # Verifies a half-life of 10 equals the equivalent center of mass, that
    # combining half-life with span or center of mass raises the mutual exclusion
    # value error, and that none raises the must-pass-one error.
    A = series.ewm(com=13.932726172912965).mean()
    B = series.ewm(halflife=10.0).mean()
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        series.ewm(span=20, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20, halflife=50)
    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        series.ewm()


def test_ewm_alpha_arg(series):
    # [FUNCTION DOC]
    # Regression test for GH 10789. Verifies no decay argument raises the
    # must-pass-one value error, and alpha combined with center of mass, span or
    # half-life raises the mutual exclusion value error.
    # GH 10789
    s = series
    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        s.ewm()

    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=10.0, alpha=0.5)


@pytest.mark.parametrize("func", ["cov", "corr"])
def test_ewm_pairwise_cov_corr(func, frame):
    # [FUNCTION DOC]
    # Parametrized over cov and corr. Verifies the pairwise ewm result of the frame
    # (span 10, minimum periods 5), restricted to rows of column label 1 and column
    # 5, equals the ewm of column 1 against column 5 computed directly.
    result = getattr(frame.ewm(span=10, min_periods=5), func)()
    result = result.loc[(slice(None), 1), 5]
    result.index = result.index.droplevel(1)
    expected = getattr(frame[1].ewm(span=10, min_periods=5), func)(frame[5])
    tm.assert_series_equal(result, expected, check_names=False)


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Frame with two numeric columns and one object
    # column. For each window operation present on the ewm object, verifies the
    # result for the numeric_only fixture equals the aggregation of the
    # corresponding columns (a and b, plus c when numeric_only is false), reset to
    # a plain index and cast to float.
    # GH#46560
    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    ewm = df.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel, None)
    if op is not None:
        result = op(numeric_only=numeric_only)

        columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        assert list(expected.columns) == columns

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over corr/cov, with or without the
    # other frame argument, and numeric_only. Verifies the result on a frame with an
    # object column equals the result on a float-cast frame of the selected columns
    # (c is dropped when numeric_only is true).
    # GH#46560
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    arg = (df,) if use_arg else ()
    ewm = df.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel)
    result = op(*arg, numeric_only=numeric_only)

    # Compare result to op using float dtypes, dropping c when numeric_only is True
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    ewm2 = df2.ewm(span=2, min_periods=1)
    op2 = getattr(ewm2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over integer and object dtype and
    # numeric_only. Skips if the operation does not exist. For object dtype with
    # numeric_only true, verifies a not implemented error saying the operation does
    # not implement numeric_only; otherwise verifies the result equals the series
    # aggregate cast to float.
    # GH#46560
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    ewm = ser.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel, None)
    if op is None:
        # Nothing to test
        pytest.skip("No op to test")
    if numeric_only and dtype is object:
        msg = f"ExponentialMovingWindow.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result = op(numeric_only=numeric_only)
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # [FUNCTION DOC]
    # Regression test for GH#46560. Parametrized over corr/cov, with or without an
    # other argument, dtype integer or object, and numeric_only. For object dtype
    # with numeric_only true verifies a not implemented error; otherwise verifies
    # the result equals the same operation on a float-cast copy.
    # GH#46560
    ser = Series([1, 2, 3], dtype=dtype)
    arg = (ser,) if use_arg else ()
    ewm = ser.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel)
    if numeric_only and dtype is object:
        msg = f"ExponentialMovingWindow.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)

        ser2 = ser.astype(float)
        arg2 = (ser2,) if use_arg else ()
        ewm2 = ser2.ewm(span=2, min_periods=1)
        op2 = getattr(ewm2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)
