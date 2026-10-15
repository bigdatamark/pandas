# [MODULE DOC]
# Cython declarations for np_datetime.pyx: re-declares numpy's datetime struct and
#   unit enum, the pandas C datetime API (struct<->int64 conversion, timedelta struct,
#   format requirement enum, capsule import), and the cdef/cpdef function signatures
#   implemented in np_datetime.pyx so other tslibs modules can cimport them. See
#   np_datetime.pyx for behavior of each function.
cimport numpy as cnp
from cpython.datetime cimport (
    date,
    datetime,
)
from numpy cimport (
    int32_t,
    int64_t,
    npy_datetime,
    npy_timedelta,
)


# TODO(cython3): most of these can be cimported directly from numpy
cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct npy_datetimestruct:
        # [CLASS DOC]
        # numpy's broken-down datetime: int64 year, int32 month (1-12), day, hour,
        #   min, sec, us (microseconds), ps (picoseconds), as (attoseconds).
        int64_t year
        int32_t month, day, hour, min, sec, us, ps, as

    ctypedef enum NPY_DATETIMEUNIT:
        # [CLASS DOC]
        # numpy datetime unit codes ordered coarse to fine: Y, M, W, D, B (business
        #   day), h, m, s, ms, us, ns, ps, fs, as, then GENERIC (no unit).
        NPY_FR_Y
        NPY_FR_M
        NPY_FR_W
        NPY_FR_D
        NPY_FR_B
        NPY_FR_h
        NPY_FR_m
        NPY_FR_s
        NPY_FR_ms
        NPY_FR_us
        NPY_FR_ns
        NPY_FR_ps
        NPY_FR_fs
        NPY_FR_as
        NPY_FR_GENERIC

    int64_t NPY_DATETIME_NAT  # elsewhere we call this NPY_NAT


cdef extern from "pandas/datetime/pd_datetime.h":
    ctypedef struct pandas_timedeltastruct:
        # [CLASS DOC]
        # Broken-down timedelta: days plus hrs/min/sec/ms/us/ns components, and the
        #   datetime.timedelta-style normalised seconds/microseconds/nanoseconds.
        int64_t days
        int32_t hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds

    void pandas_datetime_to_datetimestruct(npy_datetime val,
                                           NPY_DATETIMEUNIT fr,
                                           npy_datetimestruct *result) nogil

    npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT fr,
                                                npy_datetimestruct *d) except? -1 nogil

    void pandas_timedelta_to_timedeltastruct(npy_timedelta val,
                                             NPY_DATETIMEUNIT fr,
                                             pandas_timedeltastruct *result
                                             ) nogil

    void PandasDateTime_IMPORT()

    ctypedef enum FormatRequirement:
        # [CLASS DOC]
        # How the ISO parser treats a supplied format: PARTIAL_MATCH (prefix match
        #   allowed), EXACT_MATCH (whole string must match), INFER_FORMAT (no format
        #   given).
        PARTIAL_MATCH
        EXACT_MATCH
        INFER_FORMAT

# You must call this before using the PandasDateTime CAPI functions
cdef inline void import_pandas_datetime() noexcept:
    # [FUNCTION DOC]
    # Import the pandas datetime C-API capsule; must be called in a module before
    #   using any of the C functions above. Inputs/outputs: none. Side effects: sets a
    #   module-level C-API pointer. Exceptions: none (noexcept).
    PandasDateTime_IMPORT

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (opcode compare of two int64).
cdef bint cmp_scalar(int64_t lhs, int64_t rhs, int op) except -1

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (struct to 'Y-MM-DD HH:MM:SS').
cdef str dts_to_iso_string(npy_datetimestruct *dts)

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (raise OutOfBoundsDatetime if out of unit's range).
cdef check_dts_bounds(npy_datetimestruct *dts, NPY_DATETIMEUNIT unit=?)

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (naive datetime to int64 in reso).
cdef int64_t pydatetime_to_dt64(
    datetime val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=?
) except? -1
# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (fill struct from datetime).
cdef void pydatetime_to_dtstruct(datetime dt, npy_datetimestruct *dts) noexcept
# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (date at midnight to int64 in reso).
cdef int64_t pydate_to_dt64(
    date val, npy_datetimestruct *dts, NPY_DATETIMEUNIT reso=?
) except? -1
# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (fill struct from date).
cdef void pydate_to_dtstruct(date val, npy_datetimestruct *dts) noexcept

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (unit of a numpy datetime64/timedelta64 scalar).
cdef NPY_DATETIMEUNIT get_datetime64_unit(object obj) noexcept nogil

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (ISO-8601 string to struct, 0/-1 status).
cdef int string_to_dts(
    str val,
    npy_datetimestruct* dts,
    NPY_DATETIMEUNIT* out_bestunit,
    int* out_local,
    int* out_tzoffset,
    bint want_exc,
    str format = *,
    bint exact = *
) except? -1

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (unit of an M8/m8 dtype).
cdef NPY_DATETIMEUNIT get_unit_from_dtype(cnp.dtype dtype)

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (overflow-checked unit cast of M8/m8 arrays).
cpdef cnp.ndarray astype_overflowsafe(
    cnp.ndarray values,  # ndarray[datetime64[anyunit]]
    cnp.dtype dtype,  # ndarray[datetime64[anyunit]]
    bint copy=*,
    bint round_ok=*,
    bint is_coerce=*,
)
# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (multiplier from coarser to finer unit).
cdef int64_t get_conversion_factor(
    NPY_DATETIMEUNIT from_unit,
    NPY_DATETIMEUNIT to_unit,
) except? -1

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (rich comparison of two structs).
cdef bint cmp_dtstructs(npy_datetimestruct* left, npy_datetimestruct* right, int op)
# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (min/max structs for a unit).
cdef get_implementation_bounds(
    NPY_DATETIMEUNIT reso, npy_datetimestruct *lower, npy_datetimestruct *upper
)

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (convert one int64 between units).
cdef int64_t convert_reso(
    int64_t value,
    NPY_DATETIMEUNIT from_reso,
    NPY_DATETIMEUNIT to_reso,
    bint round_ok,
) except? -1

# [FUNCTION DOC]
# Declaration; see np_datetime.pyx (overflow-checked NaT-aware array add).
cpdef cnp.ndarray add_overflowsafe(cnp.ndarray left, cnp.ndarray right)
