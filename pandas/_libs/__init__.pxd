# [PACKAGE DOC]
# pandas._libs: the compiled (Cython + C) core of pandas. Performance-critical loops
# over numpy arrays live here; Python-level pandas code calls into these modules.
# Conventions shared across the package: data are numpy ndarrays (often 1-D, object
# dtype for mixed values); missing values are None, float NaN, NaT, the pandas NA
# singleton, and datetime64/timedelta64 values equal to INT64_MIN (NPY_NAT);
# datetime-like data are int64 counts since the epoch in a unit (s/ms/us/ns);
# indexers are intp arrays with -1 meaning "missing".
# Modules (by area):
# - lib: type predicates (is_scalar, is_list_like, ...), infer_dtype, object-array
#   conversion (maybe_convert_objects/numeric), misc array helpers, no_default.
# - missing: null detection and the NA singleton. tslib: arrays of date-likes to
#   datetime64 and datetime formatting. tslibs/ subpackage: Timestamp, Timedelta,
#   Period, NaT, offsets, time zones, parsing and datetime field logic.
# - algos, groupby, join, reshape, window/: sorting/ranking, grouped and rolling
#   aggregations, joins and reshaping kernels.
# - hashtable, khash, hashing, index, indexing, internals, interval/intervaltree:
#   hash tables (klib khash), Index engines, BlockManager internals, IntervalTree.
# - parsers, json, writers, sas, byteswap: I/O (CSV tokenizer, JSON, SAS).
# - arrays, sparse, ops, ops_dispatch, properties, testing, dtypes/util .pxd: support
#   utilities. *.pxi.in files are Tempita templates expanded per numeric dtype at
#   build time; src/ and include/ hold the C sources (parser, datetime, vendored).
# Importing this package first loads the pandas_parser and pandas_datetime C
# capsules (other modules depend on them), then re-exports Interval, NaT, NaTType,
# OutOfBoundsDatetime, Period, Timedelta, Timestamp and iNaT.
