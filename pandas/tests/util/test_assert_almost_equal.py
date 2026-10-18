# [MODULE DOC]
# Tests for the testing-utility function that asserts two objects are approximately equal: numbers (with absolute and relative tolerance, complex values, zeros, mixed types), strings, sets, dicts and dict-like objects, iterables, null-like values (including mismatched kinds of missing values), infinities, pandas objects, error message formats for value, class, shape and length mismatches, unicode, timestamps, and nested object arrays. Defines private helper assertion functions, a module-level list of missing-value kinds and a table of nested array cases.

import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    Index,
    NaT,
    Series,
    Timestamp,
)
import pandas._testing as tm


def _assert_almost_equal_both(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts approximate equality in both directions.
    # Parameters a and b are the objects to compare; extra keyword arguments are forwarded to the almost-equal assertion. Calls the assertion for (a, b) then (b, a). Raises an assertion error if either direction fails; returns nothing.
    """
    Check that two objects are approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    tm.assert_almost_equal(a, b, **kwargs)
    tm.assert_almost_equal(b, a, **kwargs)


def _assert_not_almost_equal(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts that two objects are not approximately equal.
    # Parameters a and b are the objects; extra keyword arguments are forwarded. Calls the almost-equal assertion; if it succeeds, fails the test with a message saying the objects were approximately equal; if it raises an assertion error, that is swallowed (the expected outcome). The test-failure exception from pytest is not an assertion error and so is not swallowed. Returns nothing.
    """
    Check that two objects are not approximately equal.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    try:
        tm.assert_almost_equal(a, b, **kwargs)
        msg = f"{a} and {b} were approximately equal when they shouldn't have been"
        pytest.fail(reason=msg)
    except AssertionError:
        pass


def _assert_not_almost_equal_both(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts the objects are not approximately equal in both directions, by calling the single-direction helper for (a, b) and (b, a) with the forwarded keyword arguments.
    """
    Check that two objects are not approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    _assert_not_almost_equal(a, b, **kwargs)
    _assert_not_almost_equal(b, a, **kwargs)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (np.int16(1), 1.000001),
        (np.float64(1.1), 1.1),
        (np.uint32(5), 5),
    ],
)
def test_assert_almost_equal_numbers(a, b):
    # [FUNCTION DOC]
    # Parametrized over number pairs (equal floats, close floats, numpy integer versus close float, numpy float, unsigned integer) that are approximately equal under default tolerances, in both directions.
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1),
        (1.1, True),
        (1, 2),
        (1.0001, np.int16(1)),
        # The following two examples are not "almost equal" due to tol.
        (0.1, 0.1001),
        (0.0011, 0.0012),
    ],
)
def test_assert_not_almost_equal_numbers(a, b):
    # [FUNCTION DOC]
    # Parametrized over number pairs that differ beyond default tolerance (including float versus boolean, and values like 0.1 versus 0.1001); not approximately equal in both directions.
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (1.1, 1.1001),
        (0.000001, 0.000005),
        (1000.0, 1000.0005),
        # Testing this example, as per #13357
        (0.000011, 0.000012),
    ],
)
def test_assert_almost_equal_numbers_atol(a, b):
    # [FUNCTION DOC]
    # Parametrized over pairs that are within a relative and absolute tolerance of 0.5e-3 (the equivalent of the removed less-precise option); approximately equal in both directions. Includes the case from issue 13357.
    # Equivalent to the deprecated check_less_precise=True, enforced in 2.0
    _assert_almost_equal_both(a, b, rtol=0.5e-3, atol=0.5e-3)


@pytest.mark.parametrize("a,b", [(1.1, 1.11), (0.1, 0.101), (0.000011, 0.001012)])
def test_assert_not_almost_equal_numbers_atol(a, b):
    # [FUNCTION DOC]
    # Parametrized over pairs that differ more than an absolute tolerance of 1e-3 (with default relative tolerance); not equal in both directions.
    _assert_not_almost_equal_both(a, b, atol=1e-3)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (1.1, 1.1001),
        (1000.0, 1000.0005),
        (1.1, 1.11),
        (0.1, 0.101),
    ],
)
def test_assert_almost_equal_numbers_rtol(a, b):
    # [FUNCTION DOC]
    # Parametrized over pairs equal within a relative tolerance of 0.05; equal in both directions.
    _assert_almost_equal_both(a, b, rtol=0.05)


@pytest.mark.parametrize("a,b", [(0.000011, 0.000012), (0.000001, 0.000005)])
def test_assert_not_almost_equal_numbers_rtol(a, b):
    # [FUNCTION DOC]
    # Parametrized over very small values that differ by more than 5 percent relatively; not equal with relative tolerance 0.05.
    _assert_not_almost_equal_both(a, b, rtol=0.05)


@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (1.00001, 1.00005, 0.001),
        (-0.908356 + 0.2j, -0.908358 + 0.2j, 1e-3),
        (0.1 + 1.009j, 0.1 + 1.006j, 0.1),
        (0.1001 + 2.0j, 0.1 + 2.001j, 0.01),
    ],
)
def test_assert_almost_equal_complex_numbers(a, b, rtol):
    # [FUNCTION DOC]
    # Parametrized over complex pairs with relative tolerances. Equal in both directions as Python complex numbers, as 64-bit complex and as 128-bit complex numpy scalars.
    _assert_almost_equal_both(a, b, rtol=rtol)
    _assert_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (0.58310768, 0.58330768, 1e-7),
        (-0.908 + 0.2j, -0.978 + 0.2j, 0.001),
        (0.1 + 1j, 0.1 + 2j, 0.01),
        (-0.132 + 1.001j, -0.132 + 1.005j, 1e-5),
        (0.58310768j, 0.58330768j, 1e-9),
    ],
)
def test_assert_not_almost_equal_complex_numbers(a, b, rtol):
    # [FUNCTION DOC]
    # Parametrized over complex pairs differing beyond the given tolerance; not equal in all three representations in both directions.
    _assert_not_almost_equal_both(a, b, rtol=rtol)
    _assert_not_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_not_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


@pytest.mark.parametrize("a,b", [(0, 0), (0, 0.0), (0, np.float64(0)), (0.00000001, 0)])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    # [FUNCTION DOC]
    # Parametrized over zero pairs of different types and a tiny value versus zero; approximately equal.
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [(0.001, 0), (1, 0)])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    # [FUNCTION DOC]
    # Parametrized over 0.001 versus zero and one versus zero; not approximately equal.
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [(1, "abc"), (1, [1]), (1, object())])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    # [FUNCTION DOC]
    # A number versus a string, a list, or a plain object is not approximately equal.
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "left_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
@pytest.mark.parametrize(
    "right_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    # [FUNCTION DOC]
    # Parametrized over all pairs of five dtypes (datetime, timedelta, float, integer, object). Two empty arrays of those dtypes compare approximately equal with the dtype check off.
    # Empty compare.
    _assert_almost_equal_both(
        np.array([], dtype=left_dtype),
        np.array([], dtype=right_dtype),
        check_dtype=False,
    )


def test_assert_almost_equal_sets():
    # [FUNCTION DOC]
    # GH#51727. Equal sets are approximately equal.
    # GH#51727
    _assert_almost_equal_both({1, 2, 3}, {1, 2, 3})


def test_assert_almost_not_equal_sets():
    # [FUNCTION DOC]
    # GH#51727. Sets differing in one element raise an assertion error whose message shows both sets.
    # GH#51727
    msg = r"{1, 2, 3} != {1, 2, 4}"
    with pytest.raises(AssertionError, match=msg):
        _assert_almost_equal_both({1, 2, 3}, {1, 2, 4})


def test_assert_almost_equal_dicts():
    # [FUNCTION DOC]
    # Equal dicts are approximately equal.
    _assert_almost_equal_both({"a": 1, "b": 2}, {"a": 1, "b": 2})


@pytest.mark.parametrize(
    "a,b",
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1}, 1),
        ({"a": 1}, "abc"),
        ({"a": 1}, [1]),
    ],
)
def test_assert_not_almost_equal_dicts(a, b):
    # [FUNCTION DOC]
    # Parametrized over dicts with different values, extra keys, or compared with a number, string or list; not approximately equal.
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("val", [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    # [FUNCTION DOC]
    # Parametrized over a value of 1 or 2. Compares a real dict holding that value against an object that exposes a keys method and item access returning 1 for the key a. Uses the equal helper when the value is 1 and the not-equal helper otherwise, with the dtype check off.
    dict_val = 1
    real_dict = {"a": val}

    class DictLikeObj:
        # [CLASS DOC]
        # HELPER: minimal dict-like class (keys and item access only) for the dict-like comparison test.
        def keys(self):
            # [FUNCTION DOC]
            # HELPER: returns a tuple holding the single key a.
            return ("a",)

        def __getitem__(self, item):
            # [FUNCTION DOC]
            # HELPER: returns the enclosing test's fixed value (1) for the key a; for any other key returns none.
            if item == "a":
                return dict_val

    func = (
        _assert_almost_equal_both if val == dict_val else _assert_not_almost_equal_both
    )
    func(real_dict, DictLikeObj(), check_dtype=False)


def test_assert_almost_equal_strings():
    # [FUNCTION DOC]
    # Equal strings are approximately equal.
    _assert_almost_equal_both("abc", "abc")


@pytest.mark.parametrize("b", ["abcd", "abd", 1, [1]])
def test_assert_not_almost_equal_strings(b):
    # [FUNCTION DOC]
    # Parametrized over a longer string, a different string, an integer and a list; none is approximately equal to the string abc.
    _assert_not_almost_equal_both("abc", b)


@pytest.mark.parametrize("box", [list, np.array])
def test_assert_almost_equal_iterables(box):
    # [FUNCTION DOC]
    # Parametrized over list and numpy array; equal contents are approximately equal.
    _assert_almost_equal_both(box([1, 2, 3]), box([1, 2, 3]))


@pytest.mark.parametrize(
    "a,b",
    [
        # Class is different.
        (np.array([1, 2, 3]), [1, 2, 3]),
        # Dtype is different.
        (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
        # Can't compare generators.
        (iter([1, 2, 3]), [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3], 1),
    ],
)
def test_assert_not_almost_equal_iterables(a, b):
    # [FUNCTION DOC]
    # Parametrized over pairs: array versus list (class differs), integer versus float arrays (dtype differs), generator versus list, differing values, differing lengths, and list versus number. Not equal (single-direction check).
    _assert_not_almost_equal(a, b)


def test_assert_almost_equal_null():
    # [FUNCTION DOC]
    # None equals None.
    _assert_almost_equal_both(None, None)


@pytest.mark.parametrize("a,b", [(None, np.nan), (None, 0), (np.nan, 0)])
def test_assert_not_almost_equal_null(a, b):
    # [FUNCTION DOC]
    # None versus NaN, None versus zero, NaN versus zero are not equal (single-direction check).
    _assert_not_almost_equal(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.inf, np.inf),
        (np.inf, float("inf")),
        (np.array([np.inf, np.nan, -np.inf]), np.array([np.inf, np.nan, -np.inf])),
    ],
)
def test_assert_almost_equal_inf(a, b):
    # [FUNCTION DOC]
    # Equal infinities (numpy and Python) and arrays containing infinities and NaN compare equal.
    _assert_almost_equal_both(a, b)


objs = [NA, np.nan, NaT, None, np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns")]


@pytest.mark.parametrize("left", objs)
@pytest.mark.parametrize("right", objs)
def test_mismatched_na_assert_almost_equal(left, right):
    # [FUNCTION DOC]
    # Parametrized over all pairs of six missing-value kinds (NA, NaN, NaT, None, numpy datetime NaT, numpy timedelta NaT). When both are the same object, almost-equal, numpy array, index, series and frame assertions of object-dtype wrappers all pass. When different kinds, the almost-equal assertion raises an assertion error about mismatched null-like values, and the series and frame equality assertions raise assertion errors; numpy array and index assertions are not checked (see the TODOs in the source about the stricter NaN handling needed).
    left_arr = np.array([left], dtype=object)
    right_arr = np.array([right], dtype=object)

    msg = "Mismatched null-like values"

    if left is right:
        _assert_almost_equal_both(left, right, check_dtype=False)
        tm.assert_numpy_array_equal(left_arr, right_arr)
        tm.assert_index_equal(
            Index(left_arr, dtype=object), Index(right_arr, dtype=object)
        )
        tm.assert_series_equal(
            Series(left_arr, dtype=object), Series(right_arr, dtype=object)
        )
        tm.assert_frame_equal(
            DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object)
        )

    else:
        with pytest.raises(AssertionError, match=msg):
            _assert_almost_equal_both(left, right, check_dtype=False)

        # TODO: to get the same deprecation in assert_numpy_array_equal we need
        #  to change/deprecate the default for strict_nan to become True
        # TODO: to get the same deprecation in assert_index_equal we need to
        #  change/deprecate array_equivalent_object to be stricter, as
        #  assert_index_equal uses Index.equal which uses array_equivalent.
        with pytest.raises(AssertionError, match="Series are different"):
            tm.assert_series_equal(
                Series(left_arr, dtype=object), Series(right_arr, dtype=object)
            )
        with pytest.raises(AssertionError, match="DataFrame.iloc.* are different"):
            tm.assert_frame_equal(
                DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object)
            )


def test_assert_not_almost_equal_inf():
    # [FUNCTION DOC]
    # Infinity versus zero is not approximately equal in both directions.
    _assert_not_almost_equal_both(np.inf, 0)


@pytest.mark.parametrize(
    "a,b",
    [
        (Index([1.0, 1.1]), Index([1.0, 1.100001])),
        (Series([1.0, 1.1]), Series([1.0, 1.100001])),
        (np.array([1.1, 2.000001]), np.array([1.1, 2.0])),
        (DataFrame({"a": [1.0, 1.1]}), DataFrame({"a": [1.0, 1.100001]})),
    ],
)
def test_assert_almost_equal_pandas(a, b):
    # [FUNCTION DOC]
    # Parametrized over an index, series, numpy array and data frame of floats differing by tiny amounts; approximately equal.
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_object():
    # [FUNCTION DOC]
    # Lists of identical timestamps are approximately equal.
    a = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    b = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_value_mismatch():
    # [FUNCTION DOC]
    # Comparing 1 and 2 raises an assertion error stating the expected and actual value with the default relative and absolute tolerances.
    msg = "expected 2\\.00000 but got 1\\.00000, with rtol=1e-05, atol=1e-08"

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(1, 2)


@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
def test_assert_almost_equal_class_mismatch(a, b, klass1, klass2):
    # [FUNCTION DOC]
    # Parametrized over an array versus an integer (both orders). Raises an assertion error saying the numpy array classes differ with the class names on each side.
    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_value_mismatch1():
    # [FUNCTION DOC]
    # Float arrays with NaN in different positions: assertion error with 66.67 percent differing and both listings.
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))


def test_assert_almost_equal_value_mismatch2():
    # [FUNCTION DOC]
    # Integer arrays differing in one of two values: 50 percent differing.
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_almost_equal_value_mismatch3():
    # [FUNCTION DOC]
    # Two-dimensional arrays (3 by 2) differing in one value: 16.67 percent differing, nested listings.
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_almost_equal_value_mismatch4():
    # [FUNCTION DOC]
    # Two-by-two arrays differing in one value: 25 percent differing.
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))


def test_assert_almost_equal_shape_mismatch_override():
    # [FUNCTION DOC]
    # With the object label Index, arrays of lengths 2 and 3 raise an assertion error whose message begins with Index and says shapes differ.
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([3, 4, 5]), obj="Index")


def test_assert_almost_equal_unicode():
    # [FUNCTION DOC]
    # Regression test (gh-20503): string arrays with different unicode characters raise an assertion error showing 33.33 percent differing.
    # see gh-20503
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array(["á", "à", "ä"]), np.array(["á", "à", "å"]))


def test_assert_almost_equal_timestamp():
    # [FUNCTION DOC]
    # Timestamp object arrays differing in one element raise an assertion error with 50 percent differing and the timestamp listings.
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_iterable_length_mismatch():
    # [FUNCTION DOC]
    # Lists of lengths 2 and 3 raise an assertion error saying the iterable lengths differ (2 versus 3).
    msg = """Iterable are different

Iterable length are different
\\[left\\]:  2
\\[right\\]: 3"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [3, 4, 5])


def test_assert_almost_equal_iterable_values_mismatch():
    # [FUNCTION DOC]
    # Lists differing in one value raise an assertion error with 50 percent of iterable values differing.
    msg = """Iterable are different

Iterable values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [1, 3])


subarr = np.empty(2, dtype=object)
subarr[:] = [np.array([None, "b"], dtype=object), np.array(["c", "d"], dtype=object)]

NESTED_CASES = [
    # nested array
    (
        np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object),
        np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object),
    ),
    # >1 level of nesting
    (
        np.array(
            [
                np.array([np.array([50, 70]), np.array([90])], dtype=object),
                np.array([np.array([20, 30])], dtype=object),
            ],
            dtype=object,
        ),
        np.array(
            [
                np.array([np.array([50, 70]), np.array([90])], dtype=object),
                np.array([np.array([20, 30])], dtype=object),
            ],
            dtype=object,
        ),
    ),
    # lists
    (
        np.array([[50, 70, 90], [20, 30]], dtype=object),
        np.array([[50, 70, 90], [20, 30]], dtype=object),
    ),
    # mixed array/list
    (
        np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
        np.array([[1, 2, 3], [4, 5]], dtype=object),
    ),
    (
        np.array([np.array([], dtype=object), None], dtype=object),
        np.array([[], None], dtype=object),
    ),
    (
        np.array(
            [
                np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
                np.array(
                    [np.array([6]), np.array([7, 8]), np.array([9])], dtype=object
                ),
            ],
            dtype=object,
        ),
        np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object),
    ),
    # same-length lists
    (
        np.array([subarr, None], dtype=object),
        np.array([[[None, "b"], ["c", "d"]], None], dtype=object),
    ),
    # dicts
    (
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
    ),
    (
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
        np.array([{"f1": 1, "f2": ["a", "b"]}], dtype=object),
    ),
    # array/list of dicts
    (
        np.array(
            [
                np.array(
                    [{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object
                ),
                np.array([], dtype=object),
            ],
            dtype=object,
        ),
        np.array([[{"f1": 1, "f2": ["a", "b"]}], []], dtype=object),
    ),
]


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.parametrize("a,b", NESTED_CASES)
def test_assert_almost_equal_array_nested(a, b):
    # [FUNCTION DOC]
    # Ignores a numpy elementwise comparison deprecation warning. Parametrized over a table of nested object arrays (nested arrays, deeper nesting, lists, mixed arrays and lists, empty subarrays with none, same-length sublists, dicts with array or list values, and arrays or lists of dicts); each pair compares approximately equal in both directions.
    _assert_almost_equal_both(a, b)
