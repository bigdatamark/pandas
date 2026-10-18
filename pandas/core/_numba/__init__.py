# [PACKAGE DOC]
# HELPER: internal package that supports the optional numba execution engine used
# by rolling, expanding, groupby and apply operations.
# Submodules: the executor module (builds numba-compiled loopers that apply 1D
# kernels over the columns of a 2D array), the extensions module (teaches numba
# about Index and Series objects) and the kernels subpackage (the actual
# aggregation kernels such as sum, mean, min/max and variance).
# This initializer is intentionally empty: it exports nothing and has no
# import-time side effects (numba itself is only imported by the submodules).
