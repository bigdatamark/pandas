# [MODULE DOC]
# Shared pytest configuration for the resample test package. Defines module-level lists of
# resample method names (downsampling aggregations, upsampling methods that also work
# when counting, and the series-only unique-count method) plus their concatenation, and
# two parametrized fixtures built from them. No import-time side effects other than
# building the lists.

import pytest

# The various methods we support
downsample_methods = [
    "min",
    "max",
    "first",
    "last",
    "sum",
    "mean",
    "sem",
    "median",
    "prod",
    "var",
    "std",
    "ohlc",
    "quantile",
]
upsample_methods = ["count", "size"]
series_methods = ["nunique"]
resample_methods = downsample_methods + upsample_methods + series_methods


@pytest.fixture(params=downsample_methods)
def downsample_method(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the downsampling aggregation method names (min, max, first,
    # last, sum, mean, sem, median, prod, var, std, ohlc, quantile). Returns the current
    # method name as a string.
    """Fixture for parametrization of Grouper downsample methods."""
    return request.param


@pytest.fixture(params=resample_methods)
def resample_method(request):
    # [FUNCTION DOC]
    # Fixture parametrized over all resample method names: the downsample names plus count,
    # size and the series-only unique count. Returns the current method name as a string.
    """Fixture for parametrization of Grouper resample methods."""
    return request.param
