# [MODULE DOC]
# C-level declarations for nattype.pyx so other Cython modules can cimport the NaT
# sentinel integer, the NaT string set, the _NaT base class, the C-visible NaT
# singleton and the fast null-checking helpers.
from cpython.datetime cimport datetime
from numpy cimport int64_t


# Shared NaT sentinel: the minimum 64-bit integer, the missing marker in
# datetime64/timedelta64 integer views.
cdef int64_t NPY_NAT

# Set of strings recognized as NaT when parsing ("NaT", "nat", "NAT", "nan", ...).
cdef set c_nat_strings

cdef class _NaT(datetime):
    # [CLASS DOC]
    # C extension base of NaTType (a datetime subclass); _value always holds NPY_NAT.
    # Full behavior is documented in nattype.pyx.
    cdef readonly:
        int64_t _value

# C-visible NaT singleton (the same object as pandas.NaT).
cdef _NaT c_NaT


# [FUNCTION DOC]
# Returns True if val is None, a NaN float/complex, or the NaT singleton.
# Exceptions/side effects: none.
cdef bint checknull_with_nat(object val)
# [FUNCTION DOC]
# Returns True if val is a numpy datetime64 scalar holding NaT. No side effects.
cdef bint is_dt64nat(object val)
# [FUNCTION DOC]
# Returns True if val is a numpy timedelta64 scalar holding NaT. No side effects.
cdef bint is_td64nat(object val)
