# [MODULE DOC]
# C-level declarations for pandas/_libs/missing.pyx: scalar and array null checks
# (None, float/complex NaN, NaT, datetime64/timedelta64 NaT, pandas NA, Decimal NaN)
# and the C-visible pandas NA singleton and its base class.

from numpy cimport (
    ndarray,
    uint8_t,
)


# [FUNCTION DOC]
# Declaration: True if both scalars are missing-value markers of the same kind.
# Inputs: left, right - any scalars; nan_matches_none - optional, treat NaN and None
# as matching (default False).
# Outputs: boolean. Exceptions: none expected. Side effects: none.
cpdef bint is_matching_na(object left, object right, bint nan_matches_none=*)
# [FUNCTION DOC]
# Declaration: True if left and right are equal-length tuples where some position
# holds pandas NA on exactly one side.
# Inputs: left, right - any objects. Outputs: boolean. Side effects: none.
cpdef bint check_na_tuples_nonequal(object left, object right)

# [FUNCTION DOC]
# Declaration: scalar NA-like test (None, NaN, NaT, dt64/td64 NaT, NA, Decimal NaN).
# Inputs: val - any object. Outputs: boolean. Side effects: none.
cpdef bint checknull(object val)
# [FUNCTION DOC]
# Declaration: elementwise checknull over an array of any shape.
# Inputs: arr - ndarray (typically object dtype).
# Outputs: boolean array of the same shape. Side effects: allocates result.
cpdef ndarray[uint8_t] isnaobj(ndarray arr)

# [FUNCTION DOC]
# Declaration: True if v is None, float NaN, NaT, or datetime64 NaT (not timedelta64).
# Inputs: v - any object. Outputs: boolean. Side effects: none.
cdef bint is_null_datetime64(v)
# [FUNCTION DOC]
# Declaration: True if v is None, float NaN, NaT, or timedelta64 NaT (not datetime64).
# Inputs: v - any object. Outputs: boolean. Side effects: none.
cdef bint is_null_timedelta64(v)
# [FUNCTION DOC]
# Declaration: True if obj is None, float NaN, NaT, or the pandas NA singleton.
# Inputs: obj - any object. Outputs: boolean. Side effects: none.
cdef bint checknull_with_nat_and_na(object obj)

# [CLASS DOC]
# Declaration of the empty C extension base class of the pandas NA type; lets C code
# type-check and compare against the NA singleton by identity.
cdef class C_NAType:
    pass

# C-visible global holding the pandas NA singleton (assigned in missing.pyx).
cdef C_NAType C_NA
