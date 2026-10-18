# [MODULE DOC]
# Small inline type-check and helper routines for tslibs, inlined into every importing
#   module. Declares a few CPython/numpy C-API functions and type objects (with nogil
#   where safe), and provides: the NaT sentinel integer (minimum int64), Python/numpy
#   scalar type predicates (integer, float, complex, bool, real number, ndarray, NaN),
#   and locale-aware string encode/decode helpers.

from cpython.object cimport PyTypeObject


# [FUNCTION DOC]
# Declarations of CPython C-API type-check functions (bool/float/complex checks, generic
#   type check) and locale-aware unicode encode/decode, imported so they can be marked
#   nogil.
# KNOWN ISSUE: the comment says the locale encode/decode functions can raise and thus
#   cannot be declared nogil, but both are declared nogil anyway (callers still hold the
#   GIL in practice).
cdef extern from "Python.h":
    # Note: importing extern-style allows us to declare these as nogil
    # functions, whereas `from cpython cimport` does not.
    bint PyBool_Check(object obj) nogil
    bint PyFloat_Check(object obj) nogil
    bint PyComplex_Check(object obj) nogil
    bint PyObject_TypeCheck(object obj, PyTypeObject* type) nogil

    # Note that following functions can potentially raise an exception,
    # thus they cannot be declared 'nogil'.
    object PyUnicode_EncodeLocale(object obj, const char *errors) nogil
    object PyUnicode_DecodeLocale(const char *str, const char *errors) nogil


cimport numpy as cnp
from numpy cimport (
    PyArray_Check,
    float64_t,
    int64_t,
    is_timedelta64_object,
)


# [FUNCTION DOC]
# Declarations of numpy scalar type objects (floating, complex floating, bool), the
#   integer-scalar check, and the NPY_MIN_INT64 constant used as the NaT marker.
cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyFloatingArrType_Type

cdef extern from "numpy/ndarrayobject.h":
    PyTypeObject PyComplexFloatingArrType_Type
    PyTypeObject PyBoolArrType_Type

    bint PyArray_IsIntegerScalar(obj) nogil

cdef extern from "numpy/npy_common.h":
    int64_t NPY_MIN_INT64


cdef inline int64_t get_nat() noexcept:
    # [FUNCTION DOC]
    # Return the integer NaT sentinel, the minimum int64 value (-2**63). No inputs, no
    #   exceptions.
    return NPY_MIN_INT64


# --------------------------------------------------------------------
# Type Checking

cdef inline bint is_integer_object(object obj) noexcept:
    # [FUNCTION DOC]
    # True if obj is a Python int or numpy integer scalar, excluding bool (Python bool)
    #   and numpy timedelta64 scalars. Inputs: obj - any object. Outputs: bint.
    #   Exceptions: none.
    # KNOWN ISSUE: the docstring Notes say timedelta64 objects count as integers, but
    #   the code explicitly excludes them (and np.bool_ is not an integer subclass, so
    #   it is also excluded).
    """
    Cython equivalent of

    `isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.timedelta64))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_integer : bool

    Notes
    -----
    This counts np.timedelta64 objects as integers.
    """
    return (not PyBool_Check(obj) and isinstance(obj, (int, cnp.integer))
            and not is_timedelta64_object(obj))


cdef inline bint is_float_object(object obj) noexcept nogil:
    # [FUNCTION DOC]
    # True if obj is a Python float (or subclass) or a numpy floating scalar. Runs
    #   without the GIL. Exceptions: none.
    """
    Cython equivalent of `isinstance(val, (float, np.floating))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_float : bool
    """
    return (PyFloat_Check(obj) or
            (PyObject_TypeCheck(obj, &PyFloatingArrType_Type)))


cdef inline bint is_complex_object(object obj) noexcept nogil:
    # [FUNCTION DOC]
    # True if obj is a Python complex (or subclass) or a numpy complex-floating scalar.
    #   Runs without the GIL.
    """
    Cython equivalent of `isinstance(val, (complex, np.complexfloating))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_complex : bool
    """
    return (PyComplex_Check(obj) or
            PyObject_TypeCheck(obj, &PyComplexFloatingArrType_Type))


cdef inline bint is_bool_object(object obj) noexcept nogil:
    # [FUNCTION DOC]
    # True if obj is a Python bool or numpy bool scalar. Runs without the GIL.
    """
    Cython equivalent of `isinstance(val, (bool, np.bool_))`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_bool : bool
    """
    return (PyBool_Check(obj) or
            PyObject_TypeCheck(obj, &PyBoolArrType_Type))


cdef inline bint is_real_number_object(object obj) noexcept:
    # [FUNCTION DOC]
    # True if obj is a bool, integer (per is_integer_object) or float scalar; complex is
    #   excluded.
    return is_bool_object(obj) or is_integer_object(obj) or is_float_object(obj)


cdef inline bint is_array(object val) noexcept:
    # [FUNCTION DOC]
    # True if val is a numpy ndarray (or subclass).
    """
    Cython equivalent of `isinstance(val, np.ndarray)`

    Parameters
    ----------
    val : object

    Returns
    -------
    is_ndarray : bool
    """
    return PyArray_Check(val)


cdef inline bint is_nan(object val):
    # [FUNCTION DOC]
    # True if val is a float scalar that is NaN (tested via self-inequality after
    #   conversion to a C double), or a complex scalar not equal to itself (either part
    #   NaN). All other objects, including None and NaT, give False. Exceptions: may
    #   propagate errors from the complex comparison.
    """
    Check if val is a Not-A-Number float or complex, including
    float('NaN') and np.nan.

    Parameters
    ----------
    val : object

    Returns
    -------
    is_nan : bool
    """
    cdef float64_t fval
    if is_float_object(val):
        fval = val
        return fval != fval
    return is_complex_object(val) and val != val


cdef inline bytes string_encode_locale(str py_string):
    # [FUNCTION DOC]
    # Encode a str to bytes using the current system locale encoding with strict error
    #   handling. Exceptions: UnicodeEncodeError on unencodable characters.
    """As opposed to PyUnicode_Encode, use current system locale to encode."""
    return PyUnicode_EncodeLocale(py_string, NULL)


cdef inline object char_to_string_locale(const char* data):
    # [FUNCTION DOC]
    # Decode a NUL-terminated C char buffer to str using the current system locale,
    #   strict errors. Exceptions: UnicodeDecodeError on invalid bytes.
    """As opposed to PyUnicode_FromString, use current system locale to decode."""
    return PyUnicode_DecodeLocale(data, NULL)
