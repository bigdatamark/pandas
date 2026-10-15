# [MODULE DOC]
# C-level declarations for timedeltas.pyx: conversion of timedelta-like scalars to
# integer counts at a given resolution, the rounding-unit helper, the timedelta
# scalar check, and the _Timedelta extension type (fields and C methods) so other
# Cython modules (tslib, timestamps, offsets, period) can use them directly.
from cpython.datetime cimport timedelta
from numpy cimport int64_t

from .np_datetime cimport NPY_DATETIMEUNIT


# [FUNCTION DOC]
# Converts a rounding frequency (string or offset; Day treated as 24 hours) into
# an integer count of creso units. Raises ValueError for non-fixed frequencies.
cpdef int64_t get_unit_for_round(freq, NPY_DATETIMEUNIT creso) except? -1
# Exposed for tslib, not intended for outside use.
# [FUNCTION DOC]
# Converts a Tick, Timedelta, timedelta64 or datetime.timedelta into an integer
# count of `reso` units (default nanoseconds); round_ok controls whether lossy
# conversion is allowed. Raises TypeError/ValueError/OutOfBoundsTimedelta.
cpdef int64_t delta_to_nanoseconds(
    delta, NPY_DATETIMEUNIT reso=*, bint round_ok=*
) except? -1
# [FUNCTION DOC]
# True if obj is a datetime.timedelta, numpy timedelta64, or Tick offset.
cdef bint is_any_td_scalar(object obj)


cdef class _Timedelta(timedelta):
    # [CLASS DOC]
    # C extension base of Timedelta. _value is the duration as an integer count of
    # _creso units (s, ms, us or ns); _d.._ns are lazily computed day/hour/minute/
    # second/milli/micro/nano components, valid only when _is_populated is true.
    cdef readonly:
        int64_t _value      # nanoseconds
        bint _is_populated  # are my components populated
        int64_t _d, _h, _m, _s, _ms, _us, _ns
        NPY_DATETIMEUNIT _creso

    # [FUNCTION DOC]
    # Converts to datetime.timedelta (microsecond precision, ns rounded half-up).
    cpdef timedelta to_pytimedelta(_Timedelta self)
    # [FUNCTION DOC]
    # True if the value has a non-zero sub-microsecond part (ns resolution only).
    cdef bint _has_ns(self)
    # [FUNCTION DOC]
    # True if the day component fits within datetime.timedelta's day range.
    cdef bint _is_in_pytimedelta_bounds(self)
    # [FUNCTION DOC]
    # Lazily fills the cached component fields; mutates self once.
    cdef _ensure_components(_Timedelta self)
    # [FUNCTION DOC]
    # Overflow-safe comparison of two Timedeltas with different resolutions.
    cdef bint _compare_mismatched_resos(self, _Timedelta other, op)
    # [FUNCTION DOC]
    # Returns a copy converted to resolution reso; raises OutOfBoundsTimedelta on
    # overflow and ValueError on lossy conversion when round_ok is false.
    cdef _Timedelta _as_creso(self, NPY_DATETIMEUNIT reso, bint round_ok=*)
    # [FUNCTION DOC]
    # Returns (self, other) both cast to the finer of the two resolutions.
    cpdef _maybe_cast_to_matching_resos(self, _Timedelta other)
