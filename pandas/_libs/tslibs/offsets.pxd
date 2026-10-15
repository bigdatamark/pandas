# [MODULE DOC]
# C-level declarations exported by offsets.pyx for other Cython modules:
# to_offset, the type-check helpers is_offset_object/is_tick_object, and the
# BaseOffset extension type layout (_n, _normalize, _cache fields).

from numpy cimport int64_t


# [FUNCTION DOC]
# Parse an offset/alias/timedelta into a DateOffset; is_period validates the
# result as a Period frequency. See to_offset in offsets.pyx.
cpdef to_offset(object obj, bint is_period=*)
# [FUNCTION DOC]
# Return True if obj is a BaseOffset instance.
cdef bint is_offset_object(object obj)
# [FUNCTION DOC]
# Return True if obj is a Tick (fixed sub-daily offset) instance.
cdef bint is_tick_object(object obj)

cdef class BaseOffset:
    # [CLASS DOC]
    # Base extension type for all offsets: _n (int64 multiple), _normalize (snap
    # to midnight), _cache (dict backing cache_readonly properties).
    cdef readonly:
        int64_t _n
        bint _normalize
        dict _cache
