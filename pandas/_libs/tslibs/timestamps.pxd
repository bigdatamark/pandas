# [MODULE DOC]
# C-level declarations for timestamps.pyx so other tslibs modules (offsets,
# conversion, period, ...) can cimport the _Timestamp extension type and call its
# C methods directly. See timestamps.pyx for the data model (_value epoch count,
# _creso resolution, true _year, _nanosecond, fake datetime year outside 1..9999).
from cpython.datetime cimport (
    datetime,
    tzinfo,
)
from numpy cimport int64_t

from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    npy_datetimestruct,
)
from pandas._libs.tslibs.offsets cimport BaseOffset


# [FUNCTION DOC]
# Build a Timestamp from an int64 epoch value, matching local wall-time struct,
# tzinfo (or None), fold flag and resolution (default nanoseconds); no validation.
cdef _Timestamp create_timestamp_from_ts(int64_t value,
                                         npy_datetimestruct dts,
                                         tzinfo tz,
                                         bint fold,
                                         NPY_DATETIMEUNIT reso=*)


cdef class _Timestamp(ABCTimestamp):
    # [CLASS DOC]
    # Declaration of the Timestamp C base. Readonly C fields: _value (int64 epoch count
    # in _creso units, UTC if aware), _nanosecond (0-999), _year (true year), _creso
    # (resolution unit, seconds through nanoseconds).
    cdef readonly:
        int64_t _value, _nanosecond, _year
        NPY_DATETIMEUNIT _creso

    # [FUNCTION DOC]
    # Evaluate an is_*_start/end field relative to offset freq (fiscal month aware).
    cdef bint _get_start_end_field(self, str field, freq)
    # [FUNCTION DOC]
    # Locale-aware "day_name" / "month_name" of the local wall time.
    cdef _get_date_name_field(self, str field, object locale)
    # [FUNCTION DOC]
    # Epoch count of the local wall time (equals _value if naive or UTC).
    cdef int64_t _maybe_convert_value_to_local(self) except? -1
    # [FUNCTION DOC]
    # True if self and other are both tz-aware or both tz-naive.
    cdef bint _can_compare(self, datetime other)
    # [FUNCTION DOC]
    # numpy datetime64 with payload _value in the Timestamp's own unit.
    cpdef to_datetime64(self)
    # [FUNCTION DOC]
    # Convert to datetime.datetime, dropping nanoseconds (warning if warn is True).
    cpdef datetime to_pydatetime(_Timestamp self, bint warn=*)
    # [FUNCTION DOC]
    # Compare with a datetime not convertible to Timestamp (coarse, year-based).
    cdef bint _compare_outside_nanorange(_Timestamp self, datetime other,
                                         int op) except -1
    # [FUNCTION DOC]
    # Compare two Timestamps of different resolutions via broken-down structs.
    cdef bint _compare_mismatched_resos(_Timestamp self, _Timestamp other, int op)
    # [FUNCTION DOC]
    # Return self converted to resolution creso (rounding if round_ok, else raising).
    cdef _Timestamp _as_creso(_Timestamp self, NPY_DATETIMEUNIT creso, bint round_ok=*)
