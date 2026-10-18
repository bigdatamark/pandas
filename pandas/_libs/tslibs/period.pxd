# [MODULE DOC]
# Cython declarations for period.pyx exported to other tslibs modules (period type check
#   and datetime-struct -> ordinal).

from numpy cimport int64_t

from .np_datetime cimport npy_datetimestruct


# [FUNCTION DOC]
# True iff obj is an instance of _Period.
cdef bint is_period_object(object obj)
# [FUNCTION DOC]
# Ordinal of the period at freq containing the datetime struct dts; nogil, no error
#   reporting.
cdef int64_t get_period_ordinal(npy_datetimestruct *dts, int freq) noexcept nogil
