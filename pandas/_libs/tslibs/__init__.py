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

__all__ = [
    "BaseOffset",
    "Day",
    "IncompatibleFrequency",
    "NaT",
    "NaTType",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "Period",
    "Resolution",
    "Tick",
    "Timedelta",
    "Timestamp",
    "add_overflowsafe",
    "astype_overflowsafe",
    "delta_to_nanoseconds",
    "dt64arr_to_periodarr",
    "dtypes",
    "get_resolution",
    "get_supported_dtype",
    "get_unit_from_dtype",
    "guess_datetime_format",
    "iNaT",
    "ints_to_pydatetime",
    "ints_to_pytimedelta",
    "is_date_array_normalized",
    "is_supported_dtype",
    "is_unitless",
    "localize_pydatetime",
    "nat_strings",
    "normalize_i8_timestamps",
    "periods_per_day",
    "periods_per_second",
    "to_offset",
    "tz_compare",
    "tz_convert_from_utc",
    "tz_convert_from_utc_single",
]

from pandas._libs.tslibs import dtypes
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.dtypes import (
    Resolution,
    periods_per_day,
    periods_per_second,
)
from pandas._libs.tslibs.nattype import (
    NaT,
    NaTType,
    iNaT,
    nat_strings,
)
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    add_overflowsafe,
    astype_overflowsafe,
    get_supported_dtype,
    is_supported_dtype,
    is_unitless,
    py_get_unit_from_dtype as get_unit_from_dtype,
)
from pandas._libs.tslibs.offsets import (
    BaseOffset,
    Day,
    Tick,
    to_offset,
)
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas._libs.tslibs.period import (
    IncompatibleFrequency,
    Period,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    delta_to_nanoseconds,
    ints_to_pytimedelta,
)
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.timezones import tz_compare
from pandas._libs.tslibs.tzconversion import tz_convert_from_utc_single
from pandas._libs.tslibs.vectorized import (
    dt64arr_to_periodarr,
    get_resolution,
    ints_to_pydatetime,
    is_date_array_normalized,
    normalize_i8_timestamps,
    tz_convert_from_utc,
)
