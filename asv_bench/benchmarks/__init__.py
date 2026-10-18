"""Pandas benchmarks."""

# [PACKAGE DOC]
# Root package of the pandas airspeed-velocity (asv) benchmark suite.
# It contains one benchmark module per area of the library (algorithms,
# arithmetic, groupby, indexing, joins, reshaping, rolling windows, and so on)
# plus the subpackages algos (benchmarks for single algorithm functions) and io
# (benchmarks for readers and writers). Benchmark discovery is done by asv, which
# collects classes whose methods are prefixed with time_, peakmem_, mem_ or track_.
# The module pandas_vb_common holds shared constants and the common setup hook.
# This package exports nothing and has no import-time side effects.
