# [MODULE DOC]
# Cython declarations for timezones.pyx so other tslibs modules can cimport the tzinfo
#   classification helpers, get_utcoffset and get_dst_info at C speed. See timezones.pyx
#   for behavior.

from cpython.datetime cimport (
    datetime,
    timedelta,
    tzinfo,
)


# [FUNCTION DOC]
# Module-level reference to stdlib datetime.timezone.utc.
cdef tzinfo utc_stdlib

# [FUNCTION DOC]
# True iff tz is any library's UTC singleton (or a dateutil tzutc instance).
cpdef bint is_utc(tzinfo tz)
# [FUNCTION DOC]
# True iff tz is a dateutil tzlocal instance.
cdef bint is_tzlocal(tzinfo tz)
# [FUNCTION DOC]
# True iff tz is a zoneinfo.ZoneInfo instance.
cdef bint is_zoneinfo(tzinfo tz)

# [FUNCTION DOC]
# True iff tz duck-types as a pytz DST zone (has _utc_transition_times and
#   _transition_info).
cdef bint treat_tz_as_pytz(tzinfo tz)

# [FUNCTION DOC]
# Zone-level equality of two tzinfos (UTC flavors equal; None only equals None; else
#   compare get_timezone keys).
cpdef bint tz_compare(tzinfo start, tzinfo end)
# [FUNCTION DOC]
# Map tzinfo to a serializable zone key (str) or return the tz itself; raises TypeError
#   for None.
cpdef object get_timezone(tzinfo tz)
# [FUNCTION DOC]
# Build a tzinfo from a str/int spec; tzinfo/None pass through; TypeError for other
#   types.
cpdef tzinfo maybe_get_tz(object tz)

# [FUNCTION DOC]
# Return tz._utcoffset if present else tz.utcoffset(obj) (timedelta or None).
cdef timedelta get_utcoffset(tzinfo tz, datetime obj)
# [FUNCTION DOC]
# True if tz has no DST transitions (see pyx for the per-family rules).
cpdef bint is_fixed_offset(tzinfo tz)

# [FUNCTION DOC]
# Return cached (trans ns array, deltas ns array, typ str, has_tz_rule bool) describing
#   tz's UTC offset history.
cdef object get_dst_info(tzinfo tz)
