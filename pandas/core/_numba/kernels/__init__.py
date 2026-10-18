# [PACKAGE DOC]
# HELPER: internal package holding the numba-jitted 1D aggregation kernels used by
# the numba engine for rolling/expanding windows (sliding kernels, driven by
# start/end window bound arrays) and groupby (grouped kernels, driven by group
# labels). Submodules: the mean, min/max, sum and variance kernel modules plus a
# shared module with common helpers.
# Re-exports (also listed in the module export list): grouped and sliding
# variants of mean, min/max, sum and var. Importing this package imports the
# kernel modules, which require numba to be installed to be compiled when called.
from pandas.core._numba.kernels.mean_ import (
    grouped_mean,
    sliding_mean,
)
from pandas.core._numba.kernels.min_max_ import (
    grouped_min_max,
    sliding_min_max,
)
from pandas.core._numba.kernels.sum_ import (
    grouped_sum,
    sliding_sum,
)
from pandas.core._numba.kernels.var_ import (
    grouped_var,
    sliding_var,
)

__all__ = [
    "grouped_mean",
    "grouped_min_max",
    "grouped_sum",
    "grouped_var",
    "sliding_mean",
    "sliding_min_max",
    "sliding_sum",
    "sliding_var",
]
