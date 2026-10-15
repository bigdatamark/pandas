# [MODULE DOC]
# Cython implementation of the recursive approximate-equality assertion behind
# pandas._testing.assert_almost_equal, plus dict-equality. Handles sets, dicts, strings,
# numpy arrays, generic iterables (element-wise recursion), missing values
# (NaN/None/NaT/NA must match in kind), real numbers (math.isclose) and complex numbers
# (cmath.isclose). Raises AssertionError (often via pandas._testing helpers for detailed
# messages) on mismatch, returns True otherwise.
import cmath
import math

import numpy as np

from numpy cimport import_array

import_array()

from pandas._libs.missing cimport (
    checknull,
    is_matching_na,
)
from pandas._libs.util cimport (
    is_array,
    is_complex_object,
    is_real_number_object,
)


from pandas.core.dtypes.missing import array_equivalent


cdef bint isiterable(obj):
    # [FUNCTION DOC]
    # HELPER: True if obj has an __iter__ attribute.
    # Inputs: obj - any. Outputs: bint. Side effects: none.
    return hasattr(obj, "__iter__")


cdef bint has_length(obj):
    # [FUNCTION DOC]
    # HELPER: True if obj has a __len__ attribute.
    # Inputs: obj - any. Outputs: bint. Side effects: none.
    return hasattr(obj, "__len__")


cdef bint is_dictlike(obj):
    # [FUNCTION DOC]
    # HELPER: True if obj has both keys and __getitem__ attributes.
    # Inputs: obj - any. Outputs: bint. Side effects: none.
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


cpdef assert_dict_equal(a, b, bint compare_keys=True):
    # [FUNCTION DOC]
    # Assert two dict-like objects are approximately equal. If compare_keys, the
    # frozensets of keys must be equal; then for each key of a, a[k] and b[k] are
    # compared with assert_almost_equal (default tolerances).
    # Inputs: a, b - dict-likes; compare_keys - whether to require identical key sets.
    # Outputs: True. Exceptions: AssertionError if either is not dict-like, keys differ,
    # or values differ; KeyError if compare_keys is False and a has keys missing in b.
    # Side effects: none.
    assert is_dictlike(a) and is_dictlike(b), (
        "Cannot compare dict objects, one or both is not dict-like"
    )

    a_keys = frozenset(a.keys())
    b_keys = frozenset(b.keys())

    if compare_keys:
        assert a_keys == b_keys

    for k in a_keys:
        assert_almost_equal(a[k], b[k])

    return True


cpdef assert_almost_equal(a, b,
                          rtol=1.e-5, atol=1.e-8,
                          bint check_dtype=True,
                          obj=None, lobj=None, robj=None, index_values=None):
    # [FUNCTION DOC]
    # Recursively assert a and b are almost equal. Order of checks: sets and strs
    # compared with ==; dicts via assert_dict_equal; if a is iterable: b must be
    # iterable (else class-mismatch error) and both must have __len__; for two ndarrays
    # shapes must match, dtypes must match if check_dtype, and array_equivalent with
    # strict_nan short-circuits to True; lengths must match (symmetric difference shown
    # if < 10 apart); then each element pair is compared recursively, counting failures
    # and reporting the percentage different plus the first differing position. If only
    # b is iterable, class mismatch error. For scalars: nulls must match in kind
    # (is_matching_na, NaN does not match None), a null vs non-null fails; a == b
    # passes; two real numbers pass if math.isclose(rel_tol=rtol, abs_tol=atol); complex
    # likewise with cmath.isclose; else fail.
    # Inputs: a, b - objects; rtol, atol - float tolerances; check_dtype - for ndarrays;
    # obj, lobj, robj - names/objects used in messages; index_values - for messages.
    # Outputs: True. Exceptions: AssertionError on any mismatch.
    # Side effects: lazily imports pandas._testing helpers.
    # KNOWN ISSUE: the recursive element-wise call does not forward check_dtype (nested
    # arrays always check dtype).
    """
    Check that left and right objects are almost equal.

    Parameters
    ----------
    a : object
    b : object
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    check_dtype: bool, default True
        check dtype if both a and b are np.ndarray.
    obj : str, default None
        Specify object name being compared, internally used to show
        appropriate assertion message.
    lobj : str, default None
        Specify left object name being compared, internally used to show
        appropriate assertion message.
    robj : str, default None
        Specify right object name being compared, internally used to show
        appropriate assertion message.
    index_values : Index | ndarray, default None
        Specify shared index values of objects being compared, internally used
        to show appropriate assertion message.

    """
    cdef:
        double diff = 0.0
        Py_ssize_t i, na, nb
        double fa, fb
        bint is_unequal = False, a_is_ndarray, b_is_ndarray
        str first_diff = ""

    if lobj is None:
        lobj = a
    if robj is None:
        robj = b

    if isinstance(a, set) or isinstance(b, set):
        assert a == b, f"{a} != {b}"
        return True

    if isinstance(a, dict) or isinstance(b, dict):
        return assert_dict_equal(a, b)

    if isinstance(a, str) or isinstance(b, str):
        assert a == b, f"{a} != {b}"
        return True

    a_is_ndarray = is_array(a)
    b_is_ndarray = is_array(b)

    if obj is None:
        if a_is_ndarray or b_is_ndarray:
            obj = "numpy array"
        else:
            obj = "Iterable"

    if isiterable(a):

        if not isiterable(b):
            from pandas._testing import assert_class_equal

            # classes can't be the same, to raise error
            assert_class_equal(a, b, obj=obj)

        assert has_length(a) and has_length(b), (
            f"Can't compare objects without length, one or both is invalid: ({a}, {b})"
        )

        if a_is_ndarray and b_is_ndarray:
            na, nb = a.size, b.size
            if a.shape != b.shape:
                from pandas._testing import raise_assert_detail
                raise_assert_detail(
                    obj, f"{obj} shapes are different", a.shape, b.shape)

            if check_dtype and a.dtype != b.dtype:
                from pandas._testing import assert_attr_equal
                assert_attr_equal("dtype", a, b, obj=obj)

            if array_equivalent(a, b, strict_nan=True):
                return True

        else:
            na, nb = len(a), len(b)

        if na != nb:
            from pandas._testing import raise_assert_detail

            # if we have a small diff set, print it
            if abs(na - nb) < 10:
                r = list(set(a) ^ set(b))
            else:
                r = None

            raise_assert_detail(obj, f"{obj} length are different", na, nb, r)

        for i in range(len(a)):
            try:
                assert_almost_equal(a[i], b[i], rtol=rtol, atol=atol)
            except AssertionError:
                is_unequal = True
                diff += 1
                if not first_diff:
                    first_diff = (
                        f"At positional index {i}, first diff: {a[i]} != {b[i]}"
                    )

        if is_unequal:
            from pandas._testing import raise_assert_detail
            msg = (f"{obj} values are different "
                   f"({np.round(diff * 100.0 / na, 5)} %)")
            raise_assert_detail(
                obj, msg, lobj, robj, first_diff=first_diff, index_values=index_values
            )

        return True

    elif isiterable(b):
        from pandas._testing import assert_class_equal

        # classes can't be the same, to raise error
        assert_class_equal(a, b, obj=obj)

    if checknull(a):
        # nan / None comparison
        if is_matching_na(a, b, nan_matches_none=False):
            return True
        elif checknull(b):
            # GH#18463
            raise AssertionError(f"Mismatched null-like values {a} != {b}")
        raise AssertionError(f"{a} != {b}")
    elif checknull(b):
        raise AssertionError(f"{a} != {b}")

    if a == b:
        # object comparison
        return True

    if is_real_number_object(a) and is_real_number_object(b):
        fa, fb = a, b

        if not math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
            assert False, (f"expected {fb:.5f} but got {fa:.5f}, "
                           f"with rtol={rtol}, atol={atol}")
        return True

    if is_complex_object(a) and is_complex_object(b):
        if not cmath.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            assert False, (f"expected {b:.5f} but got {a:.5f}, "
                           f"with rtol={rtol}, atol={atol}")
        return True

    raise AssertionError(f"{a} != {b}")
