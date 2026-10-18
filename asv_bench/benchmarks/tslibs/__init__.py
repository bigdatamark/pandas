"""
Benchmarks in this directory should depend only on tslibs, tseries.offsets,
and to_offset.

i.e. any code changes that do not touch those files should not need to
run these benchmarks.
"""

# [PACKAGE DOC]
# ASV benchmark subpackage for the low-level time series library. Submodules:
# fields, normalize, offsets, period, resolution, timedelta, timestamp, tslib
# and tz_convert; each defines benchmark classes that exercise only tslibs,
# the offsets module and the offset-from-string helper. The package itself
# defines and re-exports nothing and has no import-time side effects.
