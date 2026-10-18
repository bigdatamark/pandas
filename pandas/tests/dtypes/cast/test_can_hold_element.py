# [MODULE DOC]
# Tests for the function that decides whether a numpy array can hold a given
# element without loss, covering range objects, float arrays with integer
# values, integer scalars of different widths, booleans and categoricals.

import numpy as np

from pandas.core.dtypes.cast import can_hold_element

from pandas import Categorical


def test_can_hold_element_range(any_int_numpy_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue; parametrized over integer numpy dtypes.
    # For an empty array of the dtype, ranges are accepted or rejected according to
    # whether their bounds fit: small positive ranges fit everywhere, negative starts
    # fit only signed types, a range up to 254 does not fit a signed byte, a wide range
    # (negative start up to 65536) fits only signed types of at least four bytes. A
    # very large empty range is held.
    # KNOWN ISSUE: a commented-out assertion for an empty range with huge negative bounds is left in the test, so that case is never actually checked.
    # GH#44261
    dtype = np.dtype(any_int_numpy_dtype)
    arr = np.array([], dtype=dtype)

    rng = range(2, 127)
    assert can_hold_element(arr, rng)

    # negatives -> can't be held by uint dtypes
    rng = range(-2, 127)
    if dtype.kind == "i":
        assert can_hold_element(arr, rng)
    else:
        assert not can_hold_element(arr, rng)

    rng = range(2, 255)
    if dtype == "int8":
        assert not can_hold_element(arr, rng)
    else:
        assert can_hold_element(arr, rng)

    rng = range(-255, 65537)
    if dtype.kind == "u":
        assert not can_hold_element(arr, rng)
    elif dtype.itemsize < 4:
        assert not can_hold_element(arr, rng)
    else:
        assert can_hold_element(arr, rng)

    # empty
    rng = range(-(10**10), -(10**10))
    assert len(rng) == 0
    # assert can_hold_element(arr, rng)

    rng = range(10**10, 10**10)
    assert len(rng) == 0
    assert can_hold_element(arr, rng)


def test_can_hold_element_int_values_float_ndarray():
    # [FUNCTION DOC]
    # An int64 array can hold a float array with integral values, cannot hold a float
    # array with fractional values, and cannot hold integral floats too big to cast
    # losslessly to int64.
    arr = np.array([], dtype=np.int64)

    element = np.array([1.0, 2.0])
    assert can_hold_element(arr, element)

    assert not can_hold_element(arr, element + 0.5)

    # integer but not losslessly castable to int64
    element = np.array([3, 2**65], dtype=np.float64)
    assert not can_hold_element(arr, element)


def test_can_hold_element_int8_int():
    # [FUNCTION DOC]
    # An int8 array can hold the value two as a python integer or as any numpy
    # signed/unsigned integer scalar width, but cannot hold a value too large for it
    # (512) in any type of 16 bits or more, or as a python integer.
    arr = np.array([], dtype=np.int8)

    element = 2
    assert can_hold_element(arr, element)
    assert can_hold_element(arr, np.int8(element))
    assert can_hold_element(arr, np.uint8(element))
    assert can_hold_element(arr, np.int16(element))
    assert can_hold_element(arr, np.uint16(element))
    assert can_hold_element(arr, np.int32(element))
    assert can_hold_element(arr, np.uint32(element))
    assert can_hold_element(arr, np.int64(element))
    assert can_hold_element(arr, np.uint64(element))

    element = 2**9
    assert not can_hold_element(arr, element)
    assert not can_hold_element(arr, np.int16(element))
    assert not can_hold_element(arr, np.uint16(element))
    assert not can_hold_element(arr, np.int32(element))
    assert not can_hold_element(arr, np.uint32(element))
    assert not can_hold_element(arr, np.int64(element))
    assert not can_hold_element(arr, np.uint64(element))


def test_can_hold_element_bool():
    # [FUNCTION DOC]
    # A boolean array can hold a boolean scalar, array and object array of it; it
    # cannot hold the integer one or NaN in those three forms.
    arr = np.array([], dtype=bool)

    element = True
    assert can_hold_element(arr, element)
    assert can_hold_element(arr, np.array([element]))
    assert can_hold_element(arr, np.array([element], dtype=object))

    element = 1
    assert not can_hold_element(arr, element)
    assert not can_hold_element(arr, np.array([element]))
    assert not can_hold_element(arr, np.array([element], dtype=object))

    element = np.nan
    assert not can_hold_element(arr, element)
    assert not can_hold_element(arr, np.array([element]))
    assert not can_hold_element(arr, np.array([element], dtype=object))


def test_can_hold_element_categorical():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: a float64 array can hold a categorical of
    # integers with a missing value.
    # GH#56376
    arr = np.array([], dtype=np.float64)
    cat = Categorical([1, 2, None])

    assert can_hold_element(arr, cat)
