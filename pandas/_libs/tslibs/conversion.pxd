# [MODULE DOC]
# Cython declarations for conversion.pyx: the _TSObject class layout (readonly fields)
#   and conversion functions cimported by timestamps, tslib, strptime and others. See
#   conversion.pyx for behavior.
from cpython.datetime cimport (
    datetime,
    tzinfo,
)
from numpy cimport (
    int32_t,
    int64_t,
    ndarray,
)

from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    npy_datetimestruct,
)
from pandas._libs.tslibs.timestamps cimport _Timestamp
from pandas._libs.tslibs.timezones cimport tz_compare


cdef class _TSObject:
    # [CLASS DOC]
    # Holder for a converted timestamp: dts (wall-clock struct), value (UTC-based
    #   int64 in creso units or NaT), tzinfo, fold, creso. See conversion.pyx.
    cdef readonly:
        npy_datetimestruct dts      # npy_datetimestruct
        int64_t value               # numpy dt64
        tzinfo tzinfo
        bint fold
        NPY_DATETIMEUNIT creso

    # [FUNCTION DOC]
    # Declaration; see conversion.pyx (convert value to a resolution in place).
    cdef int64_t ensure_reso(
        self, NPY_DATETIMEUNIT creso, val=*, bint round_ok=*
    ) except? -1


# [FUNCTION DOC]
# Declaration; see conversion.pyx (any scalar to _TSObject).
cdef _TSObject convert_to_tsobject(object ts, tzinfo tz, str unit,
                                   bint dayfirst, bint yearfirst,
                                   int32_t nanos=*)

# [FUNCTION DOC]
# Declaration; see conversion.pyx (datetime to _TSObject).
cdef _TSObject convert_datetime_to_tsobject(datetime ts, tzinfo tz,
                                            int32_t nanos=*,
                                            NPY_DATETIMEUNIT reso=*)

# [FUNCTION DOC]
# Declaration; see conversion.pyx (string to _TSObject).
cdef _TSObject convert_str_to_tsobject(str ts, tzinfo tz,
                                       bint dayfirst=*,
                                       bint yearfirst=*)

# [FUNCTION DOC]
# Declaration; see conversion.pyx (datetime64 scalar value in a given unit).
cdef int64_t get_datetime64_nanos(object val, NPY_DATETIMEUNIT reso) except? -1

# [FUNCTION DOC]
# Declaration; see conversion.pyx (attach tz treating fields as wall time).
cpdef datetime localize_pydatetime(datetime dt, tzinfo tz)
# [FUNCTION DOC]
# Declaration; see conversion.pyx (numeric epoch offset in unit to int64).
cdef int64_t cast_from_unit(object ts, str unit, NPY_DATETIMEUNIT out_reso=*) except? -1

# [FUNCTION DOC]
# Declaration; see conversion.pyx (localize _TSObject and check bounds).
cdef maybe_localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso)


# [FUNCTION DOC]
# Declaration; see conversion.pyx (datetime to int64 in a unit).
cdef int64_t parse_pydatetime(
    datetime val,
    npy_datetimestruct *dts,
    NPY_DATETIMEUNIT creso,
) except? -1
