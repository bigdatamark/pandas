# [PACKAGE DOC]
# pandas._libs.tslibs: the Cython core for scalar and vectorized datetime, timedelta,
#   period, frequency and timezone logic. Python-level classes (Timestamp, Timedelta,
#   Period, NaT, DateOffsets) and int64 kernels used by
#   DatetimeArray/TimedeltaArray/PeriodArray live here. This __init__ only re-exports
#   the public surface listed in __all__.
# Shared conventions: datetimes are int64 counts of a unit (s/ms/us/ns, NPY_DATETIMEUNIT
#   'reso'/'creso') since 1970-01-01 UTC; NaT is int64 min (iNaT/NPY_NAT) and
#   propagates; broken-down values use numpy's npy_datetimestruct; periods are (ordinal,
#   PeriodDtypeCode) pairs.
# Modules (roughly bottom-up dependency order): util.pxd (type checks, NaN/NaT helpers);
#   base (ABCTimestamp); ccalendar (Gregorian calendar math: days in month, weekday, ISO
#   week, day of year); np_datetime (datetime struct <-> int64 conversion, unit
#   handling, OutOfBounds errors, overflow-safe astype/add); dtypes (PeriodDtypeCode,
#   FreqGroup, Resolution, unit/abbreviation maps); nattype (NaT singleton, NaTType,
#   nat_strings); timezones (tzinfo classification, string keys, DST transition tables);
#   tzconversion (UTC <-> wall-time conversion, localization with ambiguous/nonexistent
#   handling); conversion (converting Python/numpy objects to int64 datetimes,
#   localize_pydatetime); parsing (datetime string parsing, format guessing, quarter
#   helpers); strptime (array_strptime for explicit formats); fields (vectorized field
#   extraction, rounding); timedeltas (Timedelta scalar, timedelta parsing and
#   conversion); timestamps (Timestamp scalar); offsets (DateOffset hierarchy, Tick,
#   Day, BDay, to_offset); period (Period scalar, period ordinal arithmetic,
#   IncompatibleFrequency); vectorized (array-wide tz-aware helpers: ints_to_pydatetime,
#   normalize, get_resolution, dt64arr_to_periodarr).
