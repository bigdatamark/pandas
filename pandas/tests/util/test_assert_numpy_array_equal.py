# [MODULE DOC]
# Tests for the testing-utility function that asserts two numpy arrays are equal: shape and class mismatches, many value mismatch message formats (floats with missing values, integers, 2-D arrays, unicode, objects), custom object labels, the same-or-copy memory check, and handling of NA and other missing values in object arrays.

import copy

import numpy as np
import pytest

import pandas as pd
from pandas import Timestamp
import pandas._testing as tm


def test_assert_numpy_array_equal_shape_mismatch():
    # [FUNCTION DOC]
    # Arrays of length 2 and 3 raise an assertion error saying the numpy array shapes differ, showing both shapes.
    msg = """numpy array are different

numpy array shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]))


def test_assert_numpy_array_equal_bad_type():
    # [FUNCTION DOC]
    # Passing two plain integers instead of arrays raises an assertion error saying an array type was expected.
    expected = "Expected type"

    with pytest.raises(AssertionError, match=expected):
        tm.assert_numpy_array_equal(1, 2)


@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
def test_assert_numpy_array_equal_class_mismatch(a, b, klass1, klass2):
    # [FUNCTION DOC]
    # Parametrized over an array versus an integer in both orders. Raises an assertion error saying the classes differ, naming the array and integer classes on the correct sides.
    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


def test_assert_numpy_array_equal_value_mismatch1():
    # [FUNCTION DOC]
    # Float arrays with NaN in different positions raise an assertion error saying 66.67 percent of values differ, listing both arrays.
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))


def test_assert_numpy_array_equal_value_mismatch2():
    # [FUNCTION DOC]
    # Integer arrays differing in one of two values raise an assertion error with 50 percent differing and both listings.
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_numpy_array_equal_value_mismatch3():
    # [FUNCTION DOC]
    # Two-dimensional integer arrays differing in one of six values raise an assertion error with 16.67 percent differing and nested listings.
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_numpy_array_equal_value_mismatch4():
    # [FUNCTION DOC]
    # Float arrays differing in the sixth decimal place raise an assertion error (50 percent) because the comparison is exact.
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1\\.1, 2\\.000001\\]
\\[right\\]: \\[1\\.1, 2.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1.1, 2.000001]), np.array([1.1, 2.0]))


def test_assert_numpy_array_equal_value_mismatch5():
    # [FUNCTION DOC]
    # Same data and expected message as the third mismatch test.
    # KNOWN ISSUE: this test duplicates the third value-mismatch test exactly (same arrays, same message), so it adds no coverage.
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_numpy_array_equal_value_mismatch6():
    # [FUNCTION DOC]
    # Two-by-two integer arrays differing in one of four values raise an assertion error with 25 percent differing.
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]])
        )


def test_assert_numpy_array_equal_shape_mismatch_override():
    # [FUNCTION DOC]
    # With the object label passed as Index, a shape mismatch message begins with that label and says Index shapes are different.
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]), obj="Index")


def test_numpy_array_equal_unicode():
    # [FUNCTION DOC]
    # Regression test (gh-20503): string arrays with differing unicode characters raise an assertion error showing 33.33 percent differing and the unicode values.
    # see gh-20503
    #
    # Test ensures that `tm.assert_numpy_array_equals` raises the right
    # exception when comparing np.arrays containing differing unicode objects.
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array(["á", "à", "ä"]), np.array(["á", "à", "å"])
        )


def test_numpy_array_equal_object():
    # [FUNCTION DOC]
    # Object arrays of timestamps differing in one element raise an assertion error with 50 percent differing and the timestamp listings.
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


@pytest.mark.parametrize("other_type", ["same", "copy"])
@pytest.mark.parametrize("check_same", ["same", "copy"])
def test_numpy_array_equal_copy_flag(other_type, check_same):
    # [FUNCTION DOC]
    # Parametrized over how the other array is made (a view of the same data, or a copy) and which relation is required by the check-same option (same or copy). When the actual relation differs from the required one, an assertion error says the arrays are not the same object, or are the same object, respectively (the message shows the array representation); otherwise the assertion passes.
    a = np.array([1, 2, 3])
    msg = None

    if other_type == "same":
        other = a.view()
    else:
        other = a.copy()

    if check_same != other_type:
        msg = (
            r"array\(\[1, 2, 3\]\) is not array\(\[1, 2, 3\]\)"
            if check_same == "same"
            else r"array\(\[1, 2, 3\]\) is array\(\[1, 2, 3\]\)"
        )

    if msg is not None:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(a, other, check_same=check_same)
    else:
        tm.assert_numpy_array_equal(a, other, check_same=check_same)


def test_numpy_array_equal_contains_na():
    # [FUNCTION DOC]
    # Issue 31881. A boolean array versus an object array containing a missing-value marker raises an assertion error with 50 percent differing, listing the NA marker.
    # https://github.com/pandas-dev/pandas/issues/31881
    a = np.array([True, False])
    b = np.array([True, pd.NA], dtype=object)

    msg = """numpy array are different

numpy array values are different \\(50.0 %\\)
\\[left\\]:  \\[True, False\\]
\\[right\\]: \\[True, <NA>\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


def test_numpy_array_equal_identical_na(nulls_fixture):
    # [FUNCTION DOC]
    # Parametrized over missing-value scalars from a fixture. An object array holding the scalar equals itself; an array holding an equal but non-identical copy of the scalar (made through its own copy method if it has one, otherwise a standard copy) also compares equal.
    a = np.array([nulls_fixture], dtype=object)

    tm.assert_numpy_array_equal(a, a)

    # matching but not the identical object
    if hasattr(nulls_fixture, "copy"):
        other = nulls_fixture.copy()
    else:
        other = copy.copy(nulls_fixture)
    b = np.array([other], dtype=object)
    tm.assert_numpy_array_equal(a, b)


def test_numpy_array_equal_different_na():
    # [FUNCTION DOC]
    # An object array with NaN versus one with NA raises an assertion error with 100 percent differing.
    a = np.array([np.nan], dtype=object)
    b = np.array([pd.NA], dtype=object)

    msg = """numpy array are different

numpy array values are different \\(100.0 %\\)
\\[left\\]:  \\[nan\\]
\\[right\\]: \\[<NA>\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)
