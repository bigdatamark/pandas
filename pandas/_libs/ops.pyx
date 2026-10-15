# [MODULE DOC]
# Element-wise comparison and arithmetic kernels for object-dtype numpy arrays, used by
# pandas.core.ops (array_ops) when values cannot be handled by vectorized numpy ops.
# Null handling follows pandas semantics: comparisons with a missing value (None, NaN,
# NaT, NA per missing.checknull) are False, except != which is True. Also provides
# maybe_convert_bool, which turns arrays of bools / boolean strings into a bool array
# (used by parsers and by the binop kernels to downcast results).
import operator

cimport cython
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompareBool,
)
from cython cimport Py_ssize_t

import numpy as np

from numpy cimport (
    import_array,
    ndarray,
    uint8_t,
)

import_array()


from pandas._libs.missing cimport checknull
from pandas._libs.util cimport is_nan


@cython.wraparound(False)
@cython.boundscheck(False)
def scalar_compare(ndarray[object] values, object val, object op) -> ndarray:
    # [FUNCTION DOC]
    # Compare every element of an object array with a scalar. op must be one of
    # operator.lt/le/gt/ge/eq/ne (identity check), mapped to the C rich-comparison flag.
    # Result element i is: for !=, True if values[i] or val is null, else the Python
    # comparison result with TypeError mapped to True; for ==, False if either is null,
    # else the comparison with TypeError mapped to False; for ordering ops, False if
    # either is null, else the comparison (TypeError propagates).
    # Inputs: values - 1-D object ndarray; val - scalar; op - operator function.
    # Outputs: 1-D bool ndarray of len(values).
    # Exceptions: ValueError("Unrecognized operator") for other ops; TypeError from
    # ordering comparisons of incompatible types; any error from __eq__/__lt__ etc.
    # Side effects: none.
    """
    Compare each element of `values` array with the scalar `val`, with
    the comparison operation described by `op`.

    Parameters
    ----------
    values : ndarray[object]
    val : object
    op : {operator.eq, operator.ne,
          operator.le, operator.lt,
          operator.ge, operator.gt}

    Returns
    -------
    result : ndarray[bool]
    """
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[uint8_t, cast=True] result
        bint isnull_val
        int flag
        object x

    if op is operator.lt:
        flag = Py_LT
    elif op is operator.le:
        flag = Py_LE
    elif op is operator.gt:
        flag = Py_GT
    elif op is operator.ge:
        flag = Py_GE
    elif op is operator.eq:
        flag = Py_EQ
    elif op is operator.ne:
        flag = Py_NE
    else:
        raise ValueError("Unrecognized operator")

    result = np.empty(n, dtype=bool).view(np.uint8)
    isnull_val = checknull(val)

    if flag == Py_NE:
        for i in range(n):
            x = values[i]
            if checknull(x):
                result[i] = True
            elif isnull_val:
                result[i] = True
            else:
                try:
                    result[i] = PyObject_RichCompareBool(x, val, flag)
                except TypeError:
                    result[i] = True
    elif flag == Py_EQ:
        for i in range(n):
            x = values[i]
            if checknull(x):
                result[i] = False
            elif isnull_val:
                result[i] = False
            else:
                try:
                    result[i] = PyObject_RichCompareBool(x, val, flag)
                except TypeError:
                    result[i] = False

    else:
        for i in range(n):
            x = values[i]
            if checknull(x):
                result[i] = False
            elif isnull_val:
                result[i] = False
            else:
                result[i] = PyObject_RichCompareBool(x, val, flag)

    return result.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def vec_compare(ndarray[object] left, ndarray[object] right, object op) -> ndarray:
    # [FUNCTION DOC]
    # Pointwise comparison of two equal-length object arrays. op must be one of
    # operator.lt/le/gt/ge/eq/ne (identity check), mapped to the C rich-comparison flag.
    # Result i is True for != and False for all other ops when left[i] or right[i] is
    # null; otherwise the Python rich comparison result.
    # Inputs: left, right - 1-D object ndarrays; op - operator function.
    # Outputs: 1-D bool ndarray.
    # Exceptions: ValueError if lengths differ or op unrecognized; TypeError (not
    # caught, unlike scalar_compare) or other errors from element comparisons.
    # Side effects: none.
    """
    Compare the elements of `left` with the elements of `right` pointwise,
    with the comparison operation described by `op`.

    Parameters
    ----------
    left : ndarray[object]
    right : ndarray[object]
    op : {operator.eq, operator.ne,
          operator.le, operator.lt,
          operator.ge, operator.gt}

    Returns
    -------
    result : ndarray[bool]
    """
    cdef:
        Py_ssize_t i, n = len(left)
        ndarray[uint8_t, cast=True] result
        int flag

    if n != <Py_ssize_t>len(right):
        raise ValueError(f"Arrays were different lengths: {n} vs {len(right)}")

    if op is operator.lt:
        flag = Py_LT
    elif op is operator.le:
        flag = Py_LE
    elif op is operator.gt:
        flag = Py_GT
    elif op is operator.ge:
        flag = Py_GE
    elif op is operator.eq:
        flag = Py_EQ
    elif op is operator.ne:
        flag = Py_NE
    else:
        raise ValueError("Unrecognized operator")

    result = np.empty(n, dtype=bool).view(np.uint8)

    if flag == Py_NE:
        for i in range(n):
            x = left[i]
            y = right[i]

            if checknull(x) or checknull(y):
                result[i] = True
            else:
                result[i] = PyObject_RichCompareBool(x, y, flag)
    else:
        for i in range(n):
            x = left[i]
            y = right[i]

            if checknull(x) or checknull(y):
                result[i] = False
            else:
                result[i] = PyObject_RichCompareBool(x, y, flag)

    return result.view(bool)


@cython.wraparound(False)
@cython.boundscheck(False)
def scalar_binop(ndarray[object] values, object val, object op) -> ndarray:
    # [FUNCTION DOC]
    # Apply a binary function between each element of an object array and a scalar. If
    # val is None or a float NaN, every result element is val. Otherwise result[i] =
    # values[i] when it is None/NaN, else op(values[i], val). The object result is then
    # passed through maybe_convert_bool so an all-boolean result becomes a bool ndarray
    # (only the first tuple element is returned).
    # Inputs: values - 1-D object ndarray; val - scalar; op - binary callable.
    # Outputs: 1-D object ndarray, or bool ndarray if all results are bools.
    # Exceptions: propagates any exception from op. Side effects: none.
    # KNOWN ISSUE: only None and float NaN are treated as missing (is_nan), not pd.NA or
    # NaT, unlike the compare functions which use checknull.
    """
    Apply the given binary operator `op` between each element of the array
    `values` and the scalar `val`.

    Parameters
    ----------
    values : ndarray[object]
    val : object
    op : binary operator

    Returns
    -------
    result : ndarray[object]
    """
    cdef:
        Py_ssize_t i, n = len(values)
        object[::1] result
        object x

    result = np.empty(n, dtype=object)
    if val is None or is_nan(val):
        result[:] = val
        return result.base  # `.base` to access underlying np.ndarray

    for i in range(n):
        x = values[i]
        if x is None or is_nan(x):
            result[i] = x
        else:
            result[i] = op(x, val)

    return maybe_convert_bool(result.base)[0]


@cython.wraparound(False)
@cython.boundscheck(False)
def vec_binop(ndarray[object] left, ndarray[object] right, object op) -> ndarray:
    # [FUNCTION DOC]
    # Apply a binary function pointwise to two equal-length object arrays. result[i] =
    # op(left[i], right[i]); if that raises TypeError and left[i] is None/NaN the result
    # is left[i], else if right[i] is None/NaN it is right[i], otherwise the TypeError
    # is re-raised. Result is passed through maybe_convert_bool.
    # Inputs: left, right - 1-D object ndarrays; op - binary callable.
    # Outputs: 1-D object ndarray, or bool ndarray if all results are bools.
    # Exceptions: ValueError if lengths differ; TypeError / other errors from op.
    # Side effects: none.
    """
    Apply the given binary operator `op` pointwise to the elements of
    arrays `left` and `right`.

    Parameters
    ----------
    left : ndarray[object]
    right : ndarray[object]
    op : binary operator

    Returns
    -------
    result : ndarray[object]
    """
    cdef:
        Py_ssize_t i, n = len(left)
        object[::1] result

    if n != <Py_ssize_t>len(right):
        raise ValueError(f"Arrays were different lengths: {n} vs {len(right)}")

    result = np.empty(n, dtype=object)

    for i in range(n):
        x = left[i]
        y = right[i]
        try:
            result[i] = op(x, y)
        except TypeError:
            if x is None or is_nan(x):
                result[i] = x
            elif y is None or is_nan(y):
                result[i] = y
            else:
                raise

    return maybe_convert_bool(result.base)[0]  # `.base` to access np.ndarray


def maybe_convert_bool(ndarray[object] arr,
                       true_values=None,
                       false_values=None,
                       convert_to_masked_nullable=False
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    # [FUNCTION DOC]
    # Try to convert an object array to booleans. Each element must be an actual Python
    # bool, a member of the true set ({"True","TRUE","true"} plus true_values) or false
    # set ({"False","FALSE","false"} plus false_values), or missing (None or float NaN).
    # If any element fails, returns (arr, None) unchanged. If no missing values: returns
    # (bool ndarray, None). With missing values: if convert_to_masked_nullable, returns
    # (bool ndarray with False at missing slots, bool mask ndarray); else an object
    # ndarray of True/False with NaN at missing slots and None.
    # Inputs: arr - 1-D object ndarray; true_values / false_values - optional iterables
    # of extra values; convert_to_masked_nullable - bool flag.
    # Outputs: tuple (values ndarray, mask ndarray or None).
    # Exceptions: TypeError if an element is unhashable (set membership test) or if the
    # extra value iterables contain unhashables. Side effects: none (new arrays; the
    # input is never mutated).
    cdef:
        Py_ssize_t i, n
        ndarray[uint8_t] result
        ndarray[uint8_t] mask
        object val
        set true_vals, false_vals
        bint has_na = False

    n = len(arr)
    result = np.empty(n, dtype=np.uint8)
    mask = np.zeros(n, dtype=np.uint8)
    # the defaults
    true_vals = {"True", "TRUE", "true"}
    false_vals = {"False", "FALSE", "false"}

    if true_values is not None:
        true_vals = true_vals | set(true_values)

    if false_values is not None:
        false_vals = false_vals | set(false_values)

    for i in range(n):
        val = arr[i]

        if isinstance(val, bool):
            if val is True:
                result[i] = 1
            else:
                result[i] = 0
        elif val in true_vals:
            result[i] = 1
        elif val in false_vals:
            result[i] = 0
        elif is_nan(val) or val is None:
            mask[i] = 1
            result[i] = 0  # Value here doesn't matter, will be replaced w/ nan
            has_na = True
        else:
            return (arr, None)

    if has_na:
        if convert_to_masked_nullable:
            return (result.view(np.bool_), mask.view(np.bool_))
        else:
            arr = result.view(np.bool_).astype(object)
            np.putmask(arr, mask, np.nan)
            return (arr, None)
    else:
        return (result.view(np.bool_), None)
