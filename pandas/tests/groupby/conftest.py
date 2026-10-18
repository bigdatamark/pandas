# [MODULE DOC]
# Shared pytest fixtures for all groupby tests: sample frames and series (mixed
# string keys with random numeric columns, a business-day time series and frame, a
# three-key
# frame, a slice-test frame and its grouped object) and parametrized fixtures that yield
# names of
# groupby kernels. Imports the reduction and transformation kernel name sets from the
# core groupby base module. Random data uses a fixed seed. Fixtures are function scoped.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)
from pandas.core.groupby.base import (
    reduction_kernels,
    transformation_kernels,
)


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture. Returns an eight-row frame with string key columns A (foo/bar)
    # and B (one/two/three) and two random normal float columns C and D (seeded).
    return DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )


@pytest.fixture
def ts():
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a thirty-element random normal Series (seeded) indexed by
    # business days starting 2000-01-01.
    return Series(
        np.random.default_rng(2).standard_normal(30),
        index=date_range("2000-01-01", periods=30, freq="B"),
    )


@pytest.fixture
def tsframe():
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a thirty-by-four random normal frame (seeded) with
    # object-dtype column labels A to D and a business-day index starting 2000-01-01.
    return DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=30, freq="B"),
    )


@pytest.fixture
def three_group():
    # [FUNCTION DOC]
    # HELPER: fixture. Returns an eleven-row frame with string columns A (foo/bar), B
    # (one/two) and C (dull/shiny) as grouping keys and random normal float columns D, E
    # and F (seeded).
    return DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )


@pytest.fixture
def slice_test_df():
    # [FUNCTION DOC]
    # HELPER: fixture. Returns an eight-row frame indexed by an integer column named
    # Index with columns Group (a, b, c) and Value (labels like a0_at_0 encoding group
    # occurrence and position).
    data = [
        [0, "a", "a0_at_0"],
        [1, "b", "b0_at_1"],
        [2, "a", "a1_at_2"],
        [3, "b", "b1_at_3"],
        [4, "c", "c0_at_4"],
        [5, "a", "a2_at_5"],
        [6, "a", "a3_at_6"],
        [7, "a", "a4_at_7"],
    ]
    df = DataFrame(data, columns=["Index", "Group", "Value"])
    return df.set_index("Index")


@pytest.fixture
def slice_test_grouped(slice_test_df):
    # [FUNCTION DOC]
    # HELPER: fixture. Groups the slice test frame by Group without using the key as
    # index (as_index false) and returns the grouped object.
    return slice_test_df.groupby("Group", as_index=False)


@pytest.fixture(params=sorted(reduction_kernels))
def reduction_func(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the sorted names of all groupby reduction
    # kernels; yields each name string.
    """
    yields the string names of all groupby reduction functions, one at a time.
    """
    return request.param


@pytest.fixture(params=sorted(transformation_kernels))
def transformation_func(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the sorted names of all groupby transformation
    # kernels; yields each name string.
    """yields the string names of all groupby transformation functions."""
    return request.param


@pytest.fixture(params=sorted(reduction_kernels) + sorted(transformation_kernels))
def groupby_func(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the sorted reduction kernel names followed by
    # the sorted transformation kernel names; yields each name string.
    """yields both aggregation and transformation functions."""
    return request.param


@pytest.fixture(
    params=[
        ("mean", {}),
        ("var", {"ddof": 1}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("sum", {}),
        ("min", {}),
        ("max", {}),
        ("sum", {"min_count": 2}),
        ("min", {"min_count": 2}),
        ("max", {"min_count": 2}),
    ],
    ids=[
        "mean",
        "var_1",
        "var_0",
        "std_1",
        "std_0",
        "sum",
        "min",
        "max",
        "sum-min_count",
        "min-min_count",
        "max-min_count",
    ],
)
def numba_supported_reductions(request):
    # [FUNCTION DOC]
    # HELPER: fixture yielding pairs of reduction name and keyword dictionary supported
    # by the numba engine (mean, var and std with ddof 0 or 1, sum, min, max, and
    # sum/min/max with a min_count of 2). Test ids are mean, var_1, var_0, std_1, std_0,
    # sum, min, max and the min_count variants.
    """reductions supported with engine='numba'"""
    return request.param
