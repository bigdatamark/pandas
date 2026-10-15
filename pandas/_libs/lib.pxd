# [MODULE DOC]
# C-level declarations exported by pandas/_libs/lib.pyx so other Cython modules can
# cimport them: a fast list-like check and an NA-aware object-array equality helper.

from numpy cimport ndarray


# [FUNCTION DOC]
# Declaration of the C-level list-like test implemented in lib.pyx.
# Inputs: obj - any object; allow_sets - whether set/frozenset count as list-like.
# Outputs: 1 if list-like, 0 otherwise; -1 signals a propagated Python exception.
# Exceptions: propagates errors from attribute lookup / isinstance checks.
# Side effects: none.
cdef bint c_is_list_like(object, bint) except -1

# [FUNCTION DOC]
# Declaration of the NA-safe elementwise equality of an object array against a key,
# implemented in lib.pyx.
# Inputs: arr - 1-D object array; key - scalar to compare against.
# Outputs: 1-D boolean array, False wherever an element is the pandas NA singleton.
# Exceptions: propagates errors raised by element comparison.
# Side effects: allocates the result array.
cpdef ndarray eq_NA_compat(ndarray[object] arr, object key)
