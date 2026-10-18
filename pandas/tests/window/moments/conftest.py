# [MODULE DOC]
# Fixtures and data builders for the moments consistency window tests.
# Builds a standard collection of small series and frames (empty, all missing,
# constant, monotonic, with missing values) and exposes them as parametrized
# fixtures: consistent_data (only constant objects), series_data (series only),
# all_data (series and frames) and min_periods. The data collections are built
# at import time because fixture parameters are evaluated during collection.
import itertools

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    notna,
)


def create_series():
    # [FUNCTION DOC]
    # HELPER: builds the standard list of eight test series used for fixture
    # parametrization: an empty float series named "a", an all-missing series
    # of length five, a constant series of ones, a decreasing integer series
    # from five to one, an increasing integer series from zero to four, and
    # three length-five series mixing missing values with constants or
    # (increasing, decreasing) small values. Takes no inputs; returns a new
    # list each call.
    return [
        Series(dtype=np.float64, name="a"),
        Series([np.nan] * 5),
        Series([1.0] * 5),
        Series(range(5, 0, -1)),
        Series(range(5)),
        Series([np.nan, 1.0, np.nan, 1.0, 1.0]),
        Series([np.nan, 1.0, np.nan, 2.0, 3.0]),
        Series([np.nan, 1.0, np.nan, 3.0, 2.0]),
    ]


def create_dataframes():
    # [FUNCTION DOC]
    # HELPER: builds the standard list of test frames: an empty frame with two
    # duplicate columns both named "a", a five by three integer frame with
    # columns "a", "a" and 99, plus one single-column frame per series from
    # the series builder. Takes no inputs; returns a new list each call.
    return [
        DataFrame(columns=["a", "a"]),
        DataFrame(np.arange(15).reshape((5, 3)), columns=["a", "a", 99]),
    ] + [DataFrame(s) for s in create_series()]


def is_constant(x):
    # [FUNCTION DOC]
    # HELPER: used to filter fixture data. Takes a series or frame and returns
    # true when its non-missing values contain exactly one distinct value;
    # empty or all-missing objects give false (zero distinct values).
    values = x.values.ravel("K")
    return len(set(values[notna(values)])) == 1


@pytest.fixture(
    params=(
        obj
        for obj in itertools.chain(create_series(), create_dataframes())
        if is_constant(obj)
    ),
)
def consistent_data(request):
    # [FUNCTION DOC]
    # Fixture parametrized over only those series and frames from the standard
    # collections whose non-missing values are all identical. Yields one
    # object per test.
    return request.param


@pytest.fixture(params=create_series())
def series_data(request):
    # [FUNCTION DOC]
    # Fixture parametrized over every series in the standard collection (no
    # frames). Yields one series per test.
    return request.param


@pytest.fixture(params=itertools.chain(create_series(), create_dataframes()))
def all_data(request):
    # [FUNCTION DOC]
    # Fixture parametrized over all standard series followed by all standard
    # frames. Yields one object per test. The docstring below lists the kinds
    # of series covered.
    """
    Test:
        - Empty Series / DataFrame
        - All NaN
        - All consistent value
        - Monotonically decreasing
        - Monotonically increasing
        - Monotonically consistent with NaNs
        - Monotonically increasing with NaNs
        - Monotonically decreasing with NaNs
    """
    return request.param


@pytest.fixture(params=[0, 2])
def min_periods(request):
    # [FUNCTION DOC]
    # Fixture parametrized over zero and two; yields a minimum observations
    # value. Overrides the same-named fixture of the parent window conftest.
    return request.param
