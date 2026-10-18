# [MODULE DOC]
# Shared pytest fixtures for the window (rolling, expanding, ewm) test package.
# Provides keyword-argument parameter fixtures (raw, center, min_periods,
# adjust, ignore_na, numeric_only, engine, engine_and_raw, halflife_with_times,
# step), the list of arithmetic window operator names, and random mocked
# series and frame objects. Numba-engine parameters are skipped when numba is
# missing and marked single-cpu. No import-time side effects besides
# registering fixtures.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    bdate_range,
)


@pytest.fixture(params=[True, False])
def raw(request):
    # [FUNCTION DOC]
    # Fixture parametrized over true and false; yields the value for the raw
    # keyword of the apply method of window objects.
    """raw keyword argument for rolling.apply"""
    return request.param


@pytest.fixture(
    params=[
        "sum",
        "mean",
        "median",
        "max",
        "min",
        "var",
        "std",
        "kurt",
        "skew",
        "count",
        "sem",
    ]
)
def arithmetic_win_operators(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the names of the eleven built-in window
    # aggregation methods (sum, mean, median, max, min, var, std, kurt, skew,
    # count, sem); yields one method name string per test.
    return request.param


@pytest.fixture(params=[True, False])
def center(request):
    # [FUNCTION DOC]
    # Fixture parametrized over true and false; yields the value for the
    # center option of rolling windows.
    return request.param


@pytest.fixture(params=[None, 1])
def min_periods(request):
    # [FUNCTION DOC]
    # Fixture parametrized over none and one; yields a minimum observations
    # value for window objects.
    return request.param


@pytest.fixture(params=[True, False])
def adjust(request):
    # [FUNCTION DOC]
    # Fixture parametrized over true and false; yields the adjust option for
    # exponentially weighted windows.
    """adjust keyword argument for ewm"""
    return request.param


@pytest.fixture(params=[True, False])
def ignore_na(request):
    # [FUNCTION DOC]
    # Fixture parametrized over true and false; yields the ignore-missing
    # option for exponentially weighted windows.
    """ignore_na keyword argument for ewm"""
    return request.param


@pytest.fixture(params=[True, False])
def numeric_only(request):
    # [FUNCTION DOC]
    # Fixture parametrized over true and false; yields the numeric-only
    # option for window aggregations.
    """numeric_only keyword argument"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("numba", marks=[td.skip_if_no("numba"), pytest.mark.single_cpu]),
        "cython",
    ]
)
def engine(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the execution engines: numba (skipped when
    # numba is not installed, marked single-cpu) and cython; yields the engine
    # name string.
    """engine keyword argument for rolling.apply"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            ("numba", True), marks=[td.skip_if_no("numba"), pytest.mark.single_cpu]
        ),
        ("cython", True),
        ("cython", False),
    ]
)
def engine_and_raw(request):
    # [FUNCTION DOC]
    # Fixture parametrized over valid (engine, raw) pairs: numba only with raw
    # true (skipped without numba, single-cpu), and cython with either raw
    # value. Yields the pair as a tuple.
    """engine and raw keyword arguments for rolling.apply"""
    return request.param


@pytest.fixture(params=["1 day", timedelta(days=1), np.timedelta64(1, "D")])
def halflife_with_times(request):
    # [FUNCTION DOC]
    # Fixture parametrized over three equivalent one-day durations (a string,
    # a standard timedelta, and a numpy timedelta); yields a half-life value
    # usable together with the times argument of exponentially weighted
    # windows.
    """Halflife argument for EWM when times is specified."""
    return request.param


@pytest.fixture
def series():
    # [FUNCTION DOC]
    # Fixture returning a float series of 100 standard normal draws (seeded
    # random generator, seed 2) with positions 20 to 39 set to missing, indexed
    # by 100 consecutive business days starting 2009-01-01.
    """Make mocked series as fixture."""
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan
    series = Series(arr, index=bdate_range(datetime(2009, 1, 1), periods=100))
    return series


@pytest.fixture
def frame():
    # [FUNCTION DOC]
    # Fixture returning a 100 by 10 frame of standard normal draws (seeded
    # generator, seed 2, no missing values) indexed by 100 consecutive
    # business days starting 2009-01-01.
    """Make mocked frame as fixture."""
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 10)),
        index=bdate_range(datetime(2009, 1, 1), periods=100),
    )


@pytest.fixture(params=[None, 1, 2, 5, 10])
def step(request):
    # [FUNCTION DOC]
    # Fixture parametrized over none, 1, 2, 5 and 10; yields the step option
    # for rolling window operations.
    """step keyword argument for rolling window operations."""
    return request.param
