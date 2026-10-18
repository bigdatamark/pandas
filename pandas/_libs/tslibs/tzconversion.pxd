# [MODULE DOC]
# C-level declarations for tzconversion.pyx: single-value UTC<->local conversions and
#   the Localizer helper class that caches a timezone's transition data for fast
#   repeated UTC-to-local conversion.
from cpython.datetime cimport tzinfo
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
)

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT


# [FUNCTION DOC]
# Declaration: convert one UTC epoch integer (in resolution creso, default ns) to wall
#   time in tz; returns -1 with exception set on error.
cpdef int64_t tz_convert_from_utc_single(
    int64_t utc_val, tzinfo tz, NPY_DATETIMEUNIT creso=*
) except? -1
# [FUNCTION DOC]
# Declaration: convert one wall-time epoch integer in tz to UTC, resolving
#   ambiguous/nonexistent times per the given policies; returns -1 with exception set on
#   error.
cdef int64_t tz_localize_to_utc_single(
    int64_t val,
    tzinfo tz,
    object ambiguous=*,
    object nonexistent=*,
    NPY_DATETIMEUNIT creso=*,
) except? -1


# [CLASS DOC]
# Precomputed per-timezone conversion state. Fields: tz; _creso (resolution); flags
#   classifying tz (use_utc, use_fixed, use_tzlocal, use_dst, use_pytz, use_zoneinfo);
#   has_tz_rule (tz has a recurring rule beyond its last listed transition); trans
#   (int64 array of UTC transition instants), ntrans (its length), tdata (raw pointer
#   into trans), deltas (UTC offsets per transition in creso units), delta (fixed
#   offset when use_fixed), last_trans (last transition minus 12 hours; beyond it
#   zoneinfo zones fall back to the tzinfo API).
cdef class Localizer:
    cdef:
        tzinfo tz
        NPY_DATETIMEUNIT _creso
        bint use_utc, use_fixed, use_tzlocal, use_dst, use_pytz, use_zoneinfo
        bint has_tz_rule
        ndarray trans
        Py_ssize_t ntrans
        const int64_t[::1] deltas
        int64_t delta
        int64_t* tdata
        int64_t last_trans

    # [FUNCTION DOC]
    # Declaration: map one UTC value to local value, optionally caching search position
    #   in pos and reporting fold.
    cdef int64_t utc_val_to_local_val(
        self,
        int64_t utc_val,
        Py_ssize_t* pos,
        bint* fold=?,
    ) except? -1
