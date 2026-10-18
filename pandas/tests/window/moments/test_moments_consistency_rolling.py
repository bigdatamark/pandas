# [MODULE DOC]
# Consistency tests for fixed-window rolling moments: checks that sum, mean,
# variance, standard deviation, covariance and correlation computed by the
# rolling window agree with each other and with the apply path, for two
# (window, min_periods) cases and both center settings. Uses data fixtures
# from the moments conftest and center from the window conftest.
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


def no_nans(x):
    # [FUNCTION DOC]
    # HELPER: returns true when a series or frame contains no missing values
    # (reduces over all axes).
    return x.notna().all(axis=None)


def all_na(x):
    # [FUNCTION DOC]
    # HELPER: returns true when every element of a series or frame is missing
    # (reduces over all axes; true for empty objects).
    return x.isnull().all(axis=None)


@pytest.fixture(params=[(1, 0), (5, 1)])
def rolling_consistency_cases(request):
    # [FUNCTION DOC]
    # Fixture parametrized over two (window size, minimum periods) pairs:
    # (1, 0) and (5, 1). Yields the pair as a tuple.
    """window, min_periods"""
    return request.param


@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_rolling_apply_consistency_sum(
    request, all_data, rolling_consistency_cases, center, f
):
    # [FUNCTION DOC]
    # Verifies the rolling sum equals rolling apply (raw mode) with three
    # equivalent functions: a series-sum lambda, nan-skipping sum and plain
    # numpy sum. For plain numpy sum the test is marked expected-to-fail
    # unless the data has no missing values or is entirely missing, non-empty
    # and minimum periods is positive, since plain sum propagates missing
    # values.
    window, min_periods = rolling_consistency_cases

    if f is np.sum:
        if not no_nans(all_data) and not (
            all_na(all_data) and not all_data.empty and min_periods > 0
        ):
            request.applymarker(
                pytest.mark.xfail(reason="np.sum has different behavior with NaNs")
            )
    rolling_f_result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).sum()
    rolling_apply_f_result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).apply(func=f, raw=True)
    tm.assert_equal(rolling_f_result, rolling_apply_f_result)


@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(all_data, rolling_consistency_cases, center, ddof):
    # [FUNCTION DOC]
    # Parametrized over ddof 0 and 1. Verifies rolling variance is never
    # negative and, with ddof zero, equals rolling mean of squares minus the
    # square of the rolling mean.
    window, min_periods = rolling_consistency_cases

    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    assert not (var_x < 0).any(axis=None)

    if ddof == 0:
        # check that biased var(x) == mean(x^2) - mean(x)^2
        mean_x = all_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_x2 = (
            (all_data * all_data)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(
    consistent_data, rolling_consistency_cases, center, ddof
):
    # [FUNCTION DOC]
    # Parametrized over ddof 0 and 1. For constant data verifies rolling
    # variance is never positive and equals zero where the rolling count
    # reaches the minimum (at least one), missing elsewhere, and with ddof one
    # also missing where fewer than two observations are in the window.
    window, min_periods = rolling_consistency_cases

    count_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).count()
    var_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)

    # check that variance of constant series is identically 0
    assert not (var_x > 0).any(axis=None)
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if ddof == 1:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_var_std_cov(
    all_data, rolling_consistency_cases, center, ddof
):
    # [FUNCTION DOC]
    # Parametrized over ddof 0 and 1. Verifies rolling variance, standard
    # deviation and self-covariance are non-negative, variance equals standard
    # deviation squared, and variance equals self-covariance.
    window, min_periods = rolling_consistency_cases

    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    assert not (var_x < 0).any(axis=None)

    std_x = all_data.rolling(window=window, min_periods=min_periods, center=center).std(
        ddof=ddof
    )
    assert not (std_x < 0).any(axis=None)

    # check that var(x) == std(x)^2
    tm.assert_equal(var_x, std_x * std_x)

    cov_x_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).cov(all_data, ddof=ddof)
    assert not (cov_x_x < 0).any(axis=None)

    # check that var(x) == cov(x, x)
    tm.assert_equal(var_x, cov_x_x)


@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_series_cov_corr(
    series_data, rolling_consistency_cases, center, ddof
):
    # [FUNCTION DOC]
    # Parametrized over ddof 0 and 1, series only. Verifies covariance equals
    # half of (variance of the sum minus the two variances), correlation
    # equals covariance over the product of standard deviations, and with ddof
    # zero covariance equals mean of product minus product of means.
    window, min_periods = rolling_consistency_cases

    var_x_plus_y = (
        (series_data + series_data)
        .rolling(window=window, min_periods=min_periods, center=center)
        .var(ddof=ddof)
    )
    var_x = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)
    var_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)
    cov_x_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).cov(series_data, ddof=ddof)
    # check that cov(x, y) == (var(x+y) - var(x) -
    # var(y)) / 2
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # check that corr(x, y) == cov(x, y) / (std(x) *
    # std(y))
    corr_x_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).corr(series_data)
    std_x = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).std(ddof=ddof)
    std_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).std(ddof=ddof)
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if ddof == 0:
        # check that biased cov(x, y) == mean(x*y) -
        # mean(x)*mean(y)
        mean_x = series_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_y = series_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_x_times_y = (
            (series_data * series_data)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))


def test_rolling_consistency_mean(all_data, rolling_consistency_cases, center):
    # [FUNCTION DOC]
    # Verifies the rolling mean equals rolling sum divided by rolling count
    # (same window, minimum periods and center), compared as float64.
    window, min_periods = rolling_consistency_cases

    result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()
    expected = (
        all_data.rolling(window=window, min_periods=min_periods, center=center)
        .sum()
        .divide(
            all_data.rolling(
                window=window, min_periods=min_periods, center=center
            ).count()
        )
    )
    tm.assert_equal(result, expected.astype("float64"))


def test_rolling_consistency_constant(
    consistent_data, rolling_consistency_cases, center
):
    # [FUNCTION DOC]
    # For constant data verifies the rolling mean equals the constant where
    # the rolling count reaches the minimum (at least one) and is missing
    # elsewhere, and rolling correlation of the data with itself is entirely
    # missing.
    window, min_periods = rolling_consistency_cases

    count_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).count()
    mean_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()
    # check that correlation of a series with itself is either 1 or NaN
    corr_x_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).corr(consistent_data)

    exp = (
        consistent_data.max()
        if isinstance(consistent_data, Series)
        else consistent_data.max().max()
    )

    # check mean of constant series
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    tm.assert_equal(mean_x, expected)

    # check correlation of constant series with itself is NaN
    expected[:] = np.nan
    tm.assert_equal(corr_x_x, expected)


def test_rolling_consistency_var_debiasing_factors(
    all_data, rolling_consistency_cases, center
):
    # [FUNCTION DOC]
    # Verifies unbiased rolling variance equals biased (ddof zero) variance
    # times rolling count divided by (count minus one), where a zero
    # denominator is treated as missing.
    window, min_periods = rolling_consistency_cases

    # check variance debiasing factors
    var_unbiased_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var()
    var_biased_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=0)
    var_debiasing_factors_x = (
        all_data.rolling(window=window, min_periods=min_periods, center=center)
        .count()
        .divide(
            (
                all_data.rolling(
                    window=window, min_periods=min_periods, center=center
                ).count()
                - 1.0
            ).replace(0.0, np.nan)
        )
    )
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)
