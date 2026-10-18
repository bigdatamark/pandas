# [PACKAGE DOC]
# pandas._libs.window: compiled (Cython) kernels backing pandas rolling, expanding and
#   exponentially weighted window operations (pandas.core.window).
# Modules: indexers - computes per-row [start, end) window bounds for variable (time
#   based) windows; aggregations - given float64 values plus start/end bound arrays (one
#   pair per output row) computes rolling sum, mean, var, skew, kurt, median, min, max,
#   first, last, quantile, rank, nunique, apply, weighted (win_type) sum/mean/var and
#   ewm mean/sum/cov.
# Conventions: NaN marks missing values and is skipped; minp (min_periods) is the
#   minimum count of non-NaN observations required, otherwise the output is NaN. Kernels
#   mostly release the GIL and update running statistics incrementally when the bounds
#   are monotonically increasing.
