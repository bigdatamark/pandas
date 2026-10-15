# [MODULE DOC]
# Cython declarations for strptime.pyx: parse_today_now and the DatetimeParseState
#   class (shared with tslib.array_to_datetime), whose attributes are declared here. See
#   strptime.pyx for behavior.
from cpython.datetime cimport (
    datetime,
    tzinfo,
)
from numpy cimport int64_t

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT


# [FUNCTION DOC]
# Declaration; see strptime.pyx (handle 'now'/'today' strings, writing the int64
#   value).
cdef bint parse_today_now(
    str val, int64_t* iresult, bint utc, NPY_DATETIMEUNIT creso, bint infer_reso=*
)


cdef class DatetimeParseState:
    # [CLASS DOC]
    # Per-parse state; see strptime.pyx. Attributes: found_tz, found_naive (datetime
    #   objects aware/naive), found_naive_str, found_aware_str (parsed strings),
    #   found_other (other naive inputs), creso_ever_changed, creso (current unit),
    #   out_tzoffset_vals (offsets in seconds plus "naive").
    cdef:
        # See comments describing these attributes in the __cinit__ method
        bint found_tz
        bint found_naive
        bint found_naive_str
        bint found_aware_str
        bint found_other
        bint creso_ever_changed
        NPY_DATETIMEUNIT creso
        set out_tzoffset_vals

    # [FUNCTION DOC]
    # Declaration; see strptime.pyx (record a datetime's tz awareness).
    cdef tzinfo process_datetime(self, datetime dt, tzinfo tz, bint utc_convert)
    # [FUNCTION DOC]
    # Declaration; see strptime.pyx (merge element resolution).
    cdef bint update_creso(self, NPY_DATETIMEUNIT item_reso) noexcept
    # [FUNCTION DOC]
    # Declaration; see strptime.pyx (final mixed-timezone check).
    cdef tzinfo check_for_mixed_inputs(self, tzinfo tz_out, bint utc)
