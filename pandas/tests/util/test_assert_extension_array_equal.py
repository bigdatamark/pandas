# [MODULE DOC]
# Tests for the testing-utility function that asserts two extension arrays are equal, using sparse arrays and nullable integers: exact versus approximate comparison, tolerance, dtype mismatch, missing-value mask mismatch, non-extension inputs, ignoring dtype, and datetime unit/timezone differences.

import numpy as np
import pytest

from pandas import (
    Timestamp,
    array,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # Default is check_exact=False
        {"check_exact": False},
        {"check_exact": True},
    ],
)
def test_assert_extension_array_equal_not_exact(kwargs):
    # [FUNCTION DOC]
    # Regression test (gh-23709). Parametrized over keyword sets (no argument, exact false, exact true). Two sparse float arrays differing at the last decimal places: when exact comparison is requested the assertion raises an error with a message that 50 percent of values differ; otherwise (default is approximate) they compare equal.
    # see gh-23709
    arr1 = SparseArray([-0.17387645482451206, 0.3414148016424936])
    arr2 = SparseArray([-0.17387645482451206, 0.3414148016424937])

    if kwargs.get("check_exact", False):
        msg = """\
ExtensionArray are different

ExtensionArray values are different \\(50\\.0 %\\)
\\[left\\]:  \\[-0\\.17387645482.*, 0\\.341414801642.*\\]
\\[right\\]: \\[-0\\.17387645482.*, 0\\.341414801642.*\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)


@pytest.mark.parametrize("decimals", range(10))
def test_assert_extension_array_equal_less_precise(decimals):
    # [FUNCTION DOC]
    # Parametrized over decimals 0 to 9. With a relative tolerance of half of ten to the minus decimals, sparse arrays that differ in the sixth decimal place raise the assertion error (50 percent differ) when decimals is 5 or more, and pass otherwise.
    rtol = 0.5 * 10**-decimals
    arr1 = SparseArray([0.5, 0.123456])
    arr2 = SparseArray([0.5, 0.123457])

    if decimals >= 5:
        msg = """\
ExtensionArray are different

ExtensionArray values are different \\(50\\.0 %\\)
\\[left\\]:  \\[0\\.5, 0\\.123456\\]
\\[right\\]: \\[0\\.5, 0\\.123457\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)
    else:
        tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)


def test_assert_extension_array_equal_dtype_mismatch(check_dtype):
    # [FUNCTION DOC]
    # Uses the dtype check fixture. Sparse arrays of 64-bit versus 32-bit integers raise an assertion error saying the dtype attribute differs (the sparse dtype strings) when the dtype check is on, and pass when off.
    end = 5
    kwargs = {"check_dtype": check_dtype}

    arr1 = SparseArray(np.arange(end, dtype="int64"))
    arr2 = SparseArray(np.arange(end, dtype="int32"))

    if check_dtype:
        msg = """\
ExtensionArray are different

Attribute "dtype" are different
\\[left\\]:  Sparse\\[int64, 0\\]
\\[right\\]: Sparse\\[int32, 0\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)


def test_assert_extension_array_equal_missing_values():
    # [FUNCTION DOC]
    # Sparse arrays whose last element is missing in one and not in the other raise an assertion error stating the NA masks differ in 25 percent with the boolean mask listing.
    arr1 = SparseArray([np.nan, 1, 2, np.nan])
    arr2 = SparseArray([np.nan, 1, 2, 3])

    msg = """\
ExtensionArray NA mask are different

ExtensionArray NA mask values are different \\(25\\.0 %\\)
\\[left\\]:  \\[True, False, False, True\\]
\\[right\\]: \\[True, False, False, False\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(arr1, arr2)


@pytest.mark.parametrize("side", ["left", "right"])
def test_assert_extension_array_equal_non_extension_array(side):
    # [FUNCTION DOC]
    # Parametrized over which side is a plain numpy array. The assertion raises an error saying the side (left or right) is not an ExtensionArray.
    numpy_array = np.arange(5)
    extension_array = SparseArray(numpy_array)

    msg = f"{side} is not an ExtensionArray"
    args = (
        (numpy_array, extension_array)
        if side == "left"
        else (extension_array, numpy_array)
    )

    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(*args)


def test_assert_extension_array_equal_ignore_dtype_mismatch(any_int_dtype):
    # [FUNCTION DOC]
    # Regression test (issue 35715): a nullable Int64 array and an array of any integer dtype fixture with the same values compare equal when the dtype check is disabled.
    # https://github.com/pandas-dev/pandas/issues/35715
    left = array([1, 2, 3], dtype="Int64")
    right = array([1, 2, 3], dtype=any_int_dtype)
    tm.assert_extension_array_equal(left, right, check_dtype=False)


def test_assert_extension_array_equal_time_units():
    # [FUNCTION DOC]
    # Regression test (issue 55730): a naive nanosecond datetime array and a UTC-aware one with the same instant compare equal in both orders when the dtype check is disabled.
    # https://github.com/pandas-dev/pandas/issues/55730
    timestamp = Timestamp("2023-11-04T12")
    naive = array([timestamp], dtype="datetime64[ns]")
    utc = array([timestamp], dtype="datetime64[ns, UTC]")

    tm.assert_extension_array_equal(naive, utc, check_dtype=False)
    tm.assert_extension_array_equal(utc, naive, check_dtype=False)
