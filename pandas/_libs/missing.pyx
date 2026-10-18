# [MODULE DOC]
# Missing-value detection and the pandas NA singleton.
# Defines what counts as "null" in pandas object arrays: None, float/complex NaN,
# the NaT singleton, numpy datetime64/timedelta64 whose integer value is the NaT
# sentinel (INT64_MIN), the pandas NA singleton, and Decimal("NaN"). Provides scalar
# checks (checknull, is_matching_na, ...), vectorized masks over object arrays
# (isnaobj, is_numeric_na, is_pdna_or_none), and the NAType class implementing NA's
# propagating arithmetic/comparison semantics and Kleene three-valued logic.
# Module globals: INF/NEGINF float constants, NPY_NAT sentinel, is_32bit flag (used
# for NA's hash), cDecimal alias for fast isinstance checks. C_NA (C-visible) and NA
# (Python-visible) both reference the single NAType instance created at import.

from decimal import Decimal
import numbers
from sys import maxsize

cimport cython
from cpython.datetime cimport (
    date,
    time,
    timedelta,
)
from cython cimport Py_ssize_t

import numpy as np

cimport numpy as cnp
from numpy cimport (
    flatiter,
    float64_t,
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from pandas._libs cimport util
from pandas._libs.tslibs.nattype cimport (
    c_NaT as NaT,
    checknull_with_nat,
    is_dt64nat,
    is_td64nat,
)
from pandas._libs.tslibs.np_datetime cimport (
    get_datetime64_unit,
    import_pandas_datetime,
)

import_pandas_datetime()

from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

cdef:
    float64_t INF = <float64_t>np.inf
    float64_t NEGINF = -INF

    int64_t NPY_NAT = util.get_nat()

    bint is_32bit = maxsize <= 2 ** 32

    type cDecimal = Decimal  # for faster isinstance checks


cpdef bint check_na_tuples_nonequal(object left, object right):
    # [FUNCTION DOC]
    # Detect tuple pairs where an NA appears on one side but not the other, which
    # regular
    # tuple equality cannot handle (bool(NA) raises).
    # Inputs: left, right - any objects.
    # Outputs: False unless both are tuples of equal length; then True if at any
    # position
    # exactly one of the two elements is the NA singleton (identity check), else False.
    # Exceptions: none. Side effects: none.
    """
    When we have NA in one of the tuples but not the other we have to check here,
    because our regular checks fail before with ambiguous boolean value.

    Parameters
    ----------
    left: Any
    right: Any

    Returns
    -------
    True if we are dealing with tuples that have NA on one side and non NA on
    the other side.

    """
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return False

    if len(left) != len(right):
        return False

    for left_element, right_element in zip(left, right, strict=True):
        if left_element is C_NA and right_element is not C_NA:
            return True
        elif right_element is C_NA and left_element is not C_NA:
            return True

    return False


cpdef bint is_matching_na(object left, object right, bint nan_matches_none=False):
    # [FUNCTION DOC]
    # Check whether two scalars are both missing markers of the same kind.
    # Inputs: left, right - any scalars; nan_matches_none - if True, None and float NaN
    # are treated as matching each other (in either order).
    # Outputs: True when: both None; both NA; both NaT; both float NaN; both complex
    # NaN; both datetime64 NaT with the same unit; both timedelta64 NaT with the same
    # unit; or both Decimal NaN. Float NaN never matches complex NaN, and datetime64 NaT
    # never matches timedelta64 NaT or NaT. Otherwise False.
    # Exceptions: none. Side effects: none.
    """
    Check if two scalars are both NA of matching types.

    Parameters
    ----------
    left : Any
    right : Any
    nan_matches_none : bool, default False
        For backwards compatibility, consider NaN as matching None.

    Returns
    -------
    bool
    """
    if left is None:
        if nan_matches_none and util.is_nan(right):
            return True
        return right is None
    elif left is C_NA:
        return right is C_NA
    elif left is NaT:
        return right is NaT
    elif util.is_float_object(left):
        if nan_matches_none and right is None and util.is_nan(left):
            return True
        return (
            util.is_nan(left)
            and util.is_float_object(right)
            and util.is_nan(right)
        )
    elif util.is_complex_object(left):
        return (
            util.is_nan(left)
            and util.is_complex_object(right)
            and util.is_nan(right)
        )
    elif cnp.is_datetime64_object(left):
        return (
            cnp.get_datetime64_value(left) == NPY_NAT
            and cnp.is_datetime64_object(right)
            and cnp.get_datetime64_value(right) == NPY_NAT
            and get_datetime64_unit(left) == get_datetime64_unit(right)
        )
    elif cnp.is_timedelta64_object(left):
        return (
            cnp.get_timedelta64_value(left) == NPY_NAT
            and cnp.is_timedelta64_object(right)
            and cnp.get_timedelta64_value(right) == NPY_NAT
            and get_datetime64_unit(left) == get_datetime64_unit(right)
        )
    elif is_decimal_na(left):
        return is_decimal_na(right)
    return False


cpdef bint checknull(object val):
    # [FUNCTION DOC]
    # Scalar NA-like test used throughout pandas.
    # Inputs: val - any object.
    # Outputs: True if val is None, NaT, or NA (identity); a float or complex that is
    # not
    # equal to itself (NaN); a timedelta64 or datetime64 whose integer value is the NaT
    # sentinel; or a Decimal NaN. Otherwise False.
    # Exceptions: none. Side effects: none.
    """
    Return boolean describing of the input is NA-like, defined here as any
    of:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    val : object

    Returns
    -------
    bool
    """
    if val is None or val is NaT or val is C_NA:
        return True
    elif util.is_float_object(val) or util.is_complex_object(val):
        if val != val:
            return True
        return False
    elif cnp.is_timedelta64_object(val):
        return cnp.get_timedelta64_value(val) == NPY_NAT
    elif cnp.is_datetime64_object(val):
        return cnp.get_datetime64_value(val) == NPY_NAT
    else:
        return is_decimal_na(val)


cdef bint is_decimal_na(object val):
    # [FUNCTION DOC]
    # HELPER: True if val is a decimal.Decimal instance that is not equal to itself
    # (i.e. Decimal NaN/sNaN).
    # Inputs: val - any object. Outputs: boolean. Exceptions: none. Side effects: none.
    """
    Is this a decimal.Decimal object Decimal("NAN").
    """
    return isinstance(val, cDecimal) and val != val


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[uint8_t] isnaobj(ndarray arr):
    # [FUNCTION DOC]
    # Vectorized checknull over an array of any dimensionality.
    # Inputs: arr - ndarray (normally object dtype; elements are boxed via the array's
    # getitem so any dtype works).
    # Outputs: boolean array with the same shape as arr, True where checknull is True.
    # Iterates both arrays in flat (C-iterator) order writing one byte per element into
    # a uint8 buffer which is then viewed as bool.
    # Exceptions: none expected. Side effects: allocates the result; bounds/wraparound
    # checks disabled.
    """
    Return boolean mask denoting which elements of a 1-D array are na-like,
    according to the criteria defined in `checknull`:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    result : ndarray (dtype=np.bool_)
    """
    cdef:
        Py_ssize_t i, n = arr.size
        object val
        bint is_null
        ndarray result = np.empty((<object>arr).shape, dtype=np.uint8)
        flatiter it = cnp.PyArray_IterNew(arr)
        flatiter it2 = cnp.PyArray_IterNew(result)

    for i in range(n):
        # The PyArray_GETITEM and PyArray_ITER_NEXT are faster
        #  equivalents to `val = values[i]`
        val = cnp.PyArray_GETITEM(arr, cnp.PyArray_ITER_DATA(it))
        cnp.PyArray_ITER_NEXT(it)
        is_null = checknull(val)
        # Dereference pointer (set value)
        (<uint8_t *>(cnp.PyArray_ITER_DATA(it2)))[0] = <uint8_t>is_null
        cnp.PyArray_ITER_NEXT(it2)
    return result.view(np.bool_)


def isposinf_scalar(val: object) -> bool:
    # [FUNCTION DOC]
    # Inputs: val - any object.
    # Outputs: True only if val is a float (Python or numpy floating) equal to +inf.
    # Exceptions: none. Side effects: none.
    return util.is_float_object(val) and val == INF


def isneginf_scalar(val: object) -> bool:
    # [FUNCTION DOC]
    # Inputs: val - any object.
    # Outputs: True only if val is a float (Python or numpy floating) equal to -inf.
    # Exceptions: none. Side effects: none.
    return util.is_float_object(val) and val == NEGINF


cdef bint is_null_datetime64(v):
    # [FUNCTION DOC]
    # Null check for datetime-like contexts.
    # Inputs: v - any object.
    # Outputs: True if v is None, a float NaN, NaT, or a datetime64 NaT. A timedelta64
    # NaT returns False.
    # Exceptions: none. Side effects: none.
    # determine if we have a null for a datetime (or integer versions),
    # excluding np.timedelta64('nat')
    if checknull_with_nat(v) or is_dt64nat(v):
        return True
    return False


cdef bint is_null_timedelta64(v):
    # [FUNCTION DOC]
    # Null check for timedelta contexts.
    # Inputs: v - any object.
    # Outputs: True if v is None, a float NaN, NaT, or a timedelta64 NaT. A datetime64
    # NaT returns False.
    # Exceptions: none. Side effects: none.
    # determine if we have a null for a timedelta (or integer versions),
    # excluding np.datetime64('nat')
    if checknull_with_nat(v) or is_td64nat(v):
        return True
    return False


cdef bint checknull_with_nat_and_na(object obj):
    # [FUNCTION DOC]
    # Inputs: obj - any object.
    # Outputs: True if obj is None, a float NaN, NaT, or the NA singleton (GH#32214).
    # Exceptions: none. Side effects: none.
    # See GH#32214
    return checknull_with_nat(obj) or obj is C_NA


@cython.wraparound(False)
@cython.boundscheck(False)
def is_pdna_or_none(values: ndarray) -> ndarray:
    # [FUNCTION DOC]
    # Inputs: values - 1-D array (typically object dtype).
    # Outputs: 1-D bool array of the same length, True where the element is None or NA
    # (identity checks only; NaN and NaT are NOT flagged).
    # Exceptions: none. Side effects: allocates result.
    cdef:
        ndarray[uint8_t] result
        Py_ssize_t i, N
        object val

    N = len(values)
    result = np.zeros(N, dtype=np.uint8)

    for i in range(N):
        val = values[i]
        if val is None or val is C_NA:
            result[i] = True
    return result.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def is_numeric_na(values: ndarray) -> ndarray:
    # [FUNCTION DOC]
    # Mask of missing values valid for nullable numeric arrays (Integer/FloatingArray).
    # Inputs: values - 1-D array (typically object dtype).
    # Outputs: 1-D bool array, True where the element is None, NA, a float/complex NaN,
    # or Decimal NaN.
    # Exceptions: TypeError "'values' contains non-numeric NA ..." if an element passes
    # checknull but is none of the above (e.g. NaT or datetime64/timedelta64 NaT).
    # Side effects: allocates result.
    """
    Check for NA values consistent with IntegerArray/FloatingArray.

    Similar to a vectorized is_valid_na_for_dtype restricted to numeric dtypes.

    Returns
    -------
    ndarray[bool]
    """
    cdef:
        ndarray[uint8_t] result
        Py_ssize_t i, N
        object val

    N = len(values)
    result = np.zeros(N, dtype=np.uint8)

    for i in range(N):
        val = values[i]
        if checknull(val):
            if val is None or val is C_NA or util.is_nan(val) or is_decimal_na(val):
                result[i] = True
            else:
                raise TypeError(f"'values' contains non-numeric NA {val}")
    return result.view(bool)


# -----------------------------------------------------------------------------
# Implementation of NA singleton


def _create_binary_propagating_op(name, is_divmod=False):
    # [FUNCTION DOC]
    # HELPER: Factory building NAType's binary dunder methods, which propagate NA.
    # Inputs: name - dunder name (e.g. "__add__"), used to set the method's __name__ and
    # to decide special cases; is_divmod - if True results are 2-tuples.
    # Outputs: a function method(self, other) with the behavior below.
    # Exceptions: none. Side effects: none.
    is_cmp = name.strip("_") in ["eq", "ne", "le", "lt", "ge", "gt"]

    def method(self, other):
        # [FUNCTION DOC]
        # HELPER: Generated binary op for NA.
        # Inputs: self - NA; other - right operand.
        # Outputs: NA (or (NA, NA) for divmod) when other is NA, str, bytes, a
        # numbers.Number,
        # np.bool_, or a 0-d ndarray. For an n-d ndarray: a new object array of other's
        # shape
        # filled with NA (divmod returns it and a copy). For comparison ops: NA when
        # other is
        # a date, time or timedelta. For __sub__/__rsub__: NA when other is a
        # date/datetime.
        # For add/sub (and reflected): NA when other is a timedelta. Otherwise
        # NotImplemented.
        # Exceptions: none. Side effects: allocates arrays for array operands.
        if (other is C_NA or isinstance(other, (str, bytes))
                or isinstance(other, (numbers.Number, np.bool_))
                or util.is_array(other) and not other.shape):
            # Need the other.shape clause to handle NumPy scalars,
            # since we do a setitem on `out` below, which
            # won't work for NumPy scalars.
            if is_divmod:
                return NA, NA
            else:
                return NA

        elif util.is_array(other):
            out = np.empty(other.shape, dtype=object)
            out[:] = NA

            if is_divmod:
                return out, out.copy()
            else:
                return out

        elif is_cmp and isinstance(other, (date, time, timedelta)):
            return NA

        elif isinstance(other, date):
            if name in ["__sub__", "__rsub__"]:
                return NA

        elif isinstance(other, timedelta):
            if name in ["__sub__", "__rsub__", "__add__", "__radd__"]:
                return NA

        return NotImplemented

    method.__name__ = name
    return method


def _create_unary_propagating_op(name: str):
    # [FUNCTION DOC]
    # HELPER: Factory building NAType's unary dunder methods.
    # Inputs: name - dunder name assigned to the returned function's __name__.
    # Outputs: a function method(self) that always returns NA.
    # Exceptions: none. Side effects: none.
    def method(self):
        # [FUNCTION DOC]
        # HELPER: Generated unary op; ignores self and returns NA.
        return NA

    method.__name__ = name
    return method


cdef class C_NAType:
    # [CLASS DOC]
    # Empty C extension base class of NAType. Exists so C code can declare variables of
    # this type (C_NA) and do fast identity checks against the NA singleton.
    pass


class NAType(C_NAType):
    # [CLASS DOC]
    # Python class of the pandas NA missing-value singleton. __new__ caches one instance
    # in the class attribute _instance so NAType() always returns the same object.
    # Arithmetic and comparison operators propagate NA (see
    # _create_binary_propagating_op);
    # unary ops return NA; ** has identity special cases; &, |, ^ use Kleene logic with
    # bools. __array_priority__ = 1000 and __array_ufunc__ make NumPy defer to NA.
    # Hash is a fixed large value; pickling reduces to the module-level name "NA".
    """
    NA ("not available") missing value indicator.

    .. warning::

       Experimental: the behaviour of NA can still change without warning.

    The NA singleton is a missing value indicator defined by pandas. It is
    used in certain new extension dtypes (currently the "string" dtype).

    See Also
    --------
    numpy.nan : Floating point representation of Not a Number (NaN) for numerical data.
    isna : Detect missing values for an array-like object.
    notna : Detect non-missing values for an array-like object.
    DataFrame.fillna : Fill missing values in a DataFrame.
    Series.fillna : Fill missing values in a Series.

    Examples
    --------
    >>> pd.NA
    <NA>

    >>> True | pd.NA
    True

    >>> True & pd.NA
    <NA>

    >>> pd.NA != pd.NA
    <NA>

    >>> pd.NA == pd.NA
    <NA>

    >>> True | pd.NA
    True
    """
    __module__ = "pandas.api.typing"

    _instance = None

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Singleton constructor.
        # Inputs: cls; any args/kwargs are forwarded to the base allocator on first
        # call.
        # Outputs: the cached instance, created on first call and stored in
        # NAType._instance.
        # Side effects: sets the class attribute on first call.
        if NAType._instance is None:
            NAType._instance = C_NAType.__new__(cls, *args, **kwargs)
        return NAType._instance

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Outputs: the string "<NA>".
        return "<NA>"

    def __format__(self, format_spec) -> str:
        # [FUNCTION DOC]
        # Format NA using the repr string "<NA>" with the given format spec (so width
        # and
        # alignment work); if the spec is invalid for strings (ValueError), returns
        # "<NA>".
        # Inputs: format_spec - format mini-language string. Outputs: str.
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self):
        # [FUNCTION DOC]
        # Truthiness of NA is undefined.
        # Exceptions: always raises TypeError("boolean value of NA is ambiguous").
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self):
        # [FUNCTION DOC]
        # Outputs: 2**61 - 1 on 64-bit platforms or 2**31 - 1 on 32-bit, chosen to avoid
        # collisions with small integers (GH 30013).
        # GH 30013: Ensure hash is large enough to avoid hash collisions with integers
        exponent = 31 if is_32bit else 61
        return 2 ** exponent - 1

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support. Outputs: the string "NA", so unpickling resolves to the module
        # global NA singleton.
        return "NA"

    # Binary arithmetic and comparison ops -> propagate

    __add__ = _create_binary_propagating_op("__add__")
    __radd__ = _create_binary_propagating_op("__radd__")
    __sub__ = _create_binary_propagating_op("__sub__")
    __rsub__ = _create_binary_propagating_op("__rsub__")
    __mul__ = _create_binary_propagating_op("__mul__")
    __rmul__ = _create_binary_propagating_op("__rmul__")
    __matmul__ = _create_binary_propagating_op("__matmul__")
    __rmatmul__ = _create_binary_propagating_op("__rmatmul__")
    __truediv__ = _create_binary_propagating_op("__truediv__")
    __rtruediv__ = _create_binary_propagating_op("__rtruediv__")
    __floordiv__ = _create_binary_propagating_op("__floordiv__")
    __rfloordiv__ = _create_binary_propagating_op("__rfloordiv__")
    __mod__ = _create_binary_propagating_op("__mod__")
    __rmod__ = _create_binary_propagating_op("__rmod__")
    __divmod__ = _create_binary_propagating_op("__divmod__", is_divmod=True)
    __rdivmod__ = _create_binary_propagating_op("__rdivmod__", is_divmod=True)
    # __lshift__ and __rshift__ are not implemented

    __eq__ = _create_binary_propagating_op("__eq__")
    __ne__ = _create_binary_propagating_op("__ne__")
    __le__ = _create_binary_propagating_op("__le__")
    __lt__ = _create_binary_propagating_op("__lt__")
    __gt__ = _create_binary_propagating_op("__gt__")
    __ge__ = _create_binary_propagating_op("__ge__")

    # Unary ops

    __neg__ = _create_unary_propagating_op("__neg__")
    __pos__ = _create_unary_propagating_op("__pos__")
    __abs__ = _create_unary_propagating_op("__abs__")
    __invert__ = _create_unary_propagating_op("__invert__")

    # pow has special
    def __pow__(self, other):
        # [FUNCTION DOC]
        # NA ** other.
        # Inputs: other - right operand.
        # Outputs: NA if other is NA; for a numbers.Number or np.bool_: other's type(1)
        # if
        # other == 0 (anything to the 0 is 1), else NA; for an ndarray: an array whose
        # elements are 1 (of other's dtype) where other == 0 and NA elsewhere (object
        # dtype
        # when any NA present); else NotImplemented.
        # Exceptions: none. Side effects: allocates for array input.
        if other is C_NA:
            return NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 0:
                # returning positive is correct for +/- 0.
                return type(other)(1)
            else:
                return NA
        elif util.is_array(other):
            return np.where(other == 0, other.dtype.type(1), NA)

        return NotImplemented

    def __rpow__(self, other):
        # [FUNCTION DOC]
        # other ** NA.
        # Inputs: other - left operand.
        # Outputs: NA if other is NA; for a numbers.Number or np.bool_: other itself if
        # other == 1 (1 to any power is 1), else NA; for an ndarray: elementwise other
        # where
        # other == 1 else NA; else NotImplemented.
        # Exceptions: none. Side effects: allocates for array input.
        if other is C_NA:
            return NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 1:
                return other
            else:
                return NA
        elif util.is_array(other):
            return np.where(other == 1, other, NA)
        return NotImplemented

    # Logical ops using Kleene logic

    def __and__(self, other):
        # [FUNCTION DOC]
        # Kleene AND (also used as __rand__).
        # Inputs: other - operand.
        # Outputs: False if other is a false bool (Python or numpy); NA if other is a
        # true
        # bool or NA; NotImplemented for anything else.
        if other is False:
            return False
        elif other is True or other is C_NA:
            return NA
        elif util.is_bool_object(other):
            if not other:
                return False
            return NA
        return NotImplemented

    __rand__ = __and__

    def __or__(self, other):
        # [FUNCTION DOC]
        # Kleene OR (also used as __ror__).
        # Inputs: other - operand.
        # Outputs: True if other is a true bool (Python or numpy); NA if other is a
        # false
        # bool or NA; NotImplemented otherwise.
        if other is True:
            return True
        elif other is False or other is C_NA:
            return NA
        elif util.is_bool_object(other):
            if not other:
                return NA
            return True
        return NotImplemented

    __ror__ = __or__

    def __xor__(self, other):
        # [FUNCTION DOC]
        # Kleene XOR (also used as __rxor__).
        # Outputs: NA if other is a bool (Python or numpy) or NA; NotImplemented
        # otherwise.
        if util.is_bool_object(other) or other is C_NA:
            return NA
        return NotImplemented

    __rxor__ = __xor__

    __array_priority__ = 1000
    _HANDLED_TYPES = (np.ndarray, numbers.Number, str, np.bool_)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # [FUNCTION DOC]
        # NumPy ufunc protocol for NA.
        # Inputs: ufunc - the numpy ufunc; method - ufunc method name; inputs -
        # operands;
        # kwargs - ufunc keyword args.
        # Outputs: NotImplemented if any input is not ndarray, numbers.Number, str,
        # np.bool_
        # or NAType. Otherwise tries to dispatch to the matching dunder method (e.g.
        # np.add -> __add__). If no dunder applies, broadcasts all inputs and returns
        # the
        # broadcast view of the first NA input (unwrapped to a scalar if 0-d); if the
        # ufunc
        # has multiple outputs returns a tuple of NA of length nout.
        # Exceptions: ValueError if method is not "__call__" (reduce, accumulate, etc.).
        # Side effects: none.
        types = self._HANDLED_TYPES + (NAType,)
        for x in inputs:
            if not isinstance(x, types):
                return NotImplemented

        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")
        result = maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is NotImplemented:
            # For a NumPy ufunc that's not a binop, like np.logaddexp
            index = [i for i, x in enumerate(inputs) if x is NA][0]
            result = np.broadcast_arrays(*inputs)[index]
            if result.ndim == 0:
                result = result.item()
            if ufunc.nout > 1:
                result = (NA,) * ufunc.nout

        return result


C_NA = NAType()   # C-visible
NA = C_NA         # Python-visible
NA.__module__ = "pandas"
