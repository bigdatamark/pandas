# [MODULE DOC]
# Tests for the testing-utility function that asserts two series are equal: equal and unequal values and metadata, tolerance (relative), index checks and ordering options, error message formats for length, numeric, categorical and datetime mismatches, dtype handling for extension, interval, and mixed units, series subclasses, exact comparison for non-numeric data, the duplicate-labels flag, missing values, nested arrays, large integers, integer tolerance, and the default exactness of index comparison. Defines private helpers asserting equality or inequality (single or both directions).

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm


def _assert_series_equal_both(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts series equality in both directions.
    # Parameters a and b are two series; extra keyword arguments are forwarded to the series equality assertion. Calls it for (a, b) then (b, a); raises an assertion error if either fails; returns nothing.
    """
    Check that two Series equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    tm.assert_series_equal(a, b, **kwargs)
    tm.assert_series_equal(b, a, **kwargs)


def _assert_not_series_equal(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts two series are not equal.
    # Parameters a and b are series; extra keyword arguments are forwarded. Calls the series equality assertion; if it passes, fails the test with a message that the series were equal; if it raises an assertion error, that is swallowed. The test-failure exception from pytest is not an assertion error so is not swallowed. Returns nothing.
    """
    Check that two Series are not equal.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    try:
        tm.assert_series_equal(a, b, **kwargs)
        msg = "The two Series were equal when they shouldn't have been"

        pytest.fail(msg=msg)
    except AssertionError:
        pass


def _assert_not_series_equal_both(a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts two series are not equal in both directions, using the single-direction helper for (a, b) and (b, a) with forwarded keyword arguments.
    """
    Check that two Series are not equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    _assert_not_series_equal(a, b, **kwargs)
    _assert_not_series_equal(b, a, **kwargs)


@pytest.mark.parametrize("data", [range(3), list("abc"), list("áàä")])
def test_series_equal(data):
    # [FUNCTION DOC]
    # Parametrized over a range and lists of ASCII and accented letters. A series equals a series built from the same data in both directions.
    _assert_series_equal_both(Series(data), Series(data))


@pytest.mark.parametrize(
    "data1,data2",
    [
        (range(3), range(1, 4)),
        (list("abc"), list("xyz")),
        (list("áàä"), list("éèë")),
        (list("áàä"), list(b"aaa")),
        (range(3), range(4)),
    ],
)
def test_series_not_equal_value_mismatch(data1, data2):
    # [FUNCTION DOC]
    # Parametrized over pairs of data with different values or lengths (including accented letters versus bytes); series built from them are not equal in either direction.
    _assert_not_series_equal_both(Series(data1), Series(data2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "float64"},  # dtype mismatch
        {"index": [1, 2, 4]},  # index mismatch
        {"name": "foo"},  # name mismatch
    ],
)
def test_series_not_equal_metadata_mismatch(kwargs):
    # [FUNCTION DOC]
    # Parametrized over keyword sets that change the dtype (float), index, or name. A series built from the same data with that change is not equal to the plain one in either direction.
    data = range(3)
    s1 = Series(data)

    s2 = Series(data, **kwargs)
    _assert_not_series_equal_both(s1, s2)


@pytest.mark.parametrize("data1,data2", [(0.12345, 0.12346), (0.1235, 0.1236)])
@pytest.mark.parametrize("decimals", [0, 1, 2, 3, 5, 10])
def test_less_precise(data1, data2, any_float_dtype, decimals):
    # [FUNCTION DOC]
    # Parametrized over two near-equal value pairs, relative tolerance exponents (0 to 10 decimals) and the float dtype fixture. The relative tolerance is ten to the minus decimals. The assertion raises a values-differ error when decimals is 5 or 10, or when decimals is at least 3 and the values differ by at least 0.0005; otherwise the series are equal in both directions.
    rtol = 10**-decimals
    s1 = Series([data1], dtype=any_float_dtype)
    s2 = Series([data2], dtype=any_float_dtype)

    if decimals in (5, 10) or (decimals >= 3 and abs(data1 - data2) >= 0.0005):
        msg = "Series values are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, rtol=rtol)
    else:
        _assert_series_equal_both(s1, s2, rtol=rtol)


@pytest.mark.parametrize(
    "s1,s2,msg",
    [
        # Index
        (
            Series(["l1", "l2"], index=[1, 2]),
            Series(["l1", "l2"], index=[1.0, 2.0]),
            "Series\\.index are different",
        ),
        # MultiIndex
        (
            DataFrame.from_records(
                {"a": [1, 2], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ).c,
            DataFrame.from_records(
                {"a": [1.0, 2.0], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ).c,
            "Series\\.index level \\[0\\] are different",
        ),
    ],
)
def test_series_equal_index_dtype(s1, s2, msg, check_index_type):
    # [FUNCTION DOC]
    # Parametrized over a series with integer versus float index and a multi-level index series pair; plus the index-type check fixture. With the check on, an assertion error naming the index (or level 0) is raised; with it off they are equal.
    kwargs = {"check_index_type": check_index_type}

    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, **kwargs)
    else:
        tm.assert_series_equal(s1, s2, **kwargs)


@pytest.mark.parametrize("check_like", [True, False])
def test_series_equal_order_mismatch(check_like):
    # [FUNCTION DOC]
    # Parametrized over the like-ordering flag. Series with reversed order: raises an assertion error that the index differs when the flag is off; equal in both directions when on.
    s1 = Series([1, 2, 3], index=["a", "b", "c"])
    s2 = Series([3, 2, 1], index=["c", "b", "a"])

    if not check_like:  # Do not ignore index ordering.
        with pytest.raises(AssertionError, match="Series.index are different"):
            tm.assert_series_equal(s1, s2, check_like=check_like)
    else:
        _assert_series_equal_both(s1, s2, check_like=check_like)


@pytest.mark.parametrize("check_index", [True, False])
def test_series_equal_index_mismatch(check_index):
    # [FUNCTION DOC]
    # Parametrized over the index-check flag. Series with identical values but reversed index labels raise an assertion error about the index when index checking is on, and are equal when off.
    s1 = Series([1, 2, 3], index=["a", "b", "c"])
    s2 = Series([1, 2, 3], index=["c", "b", "a"])

    if check_index:  # Do not ignore index.
        with pytest.raises(AssertionError, match="Series.index are different"):
            tm.assert_series_equal(s1, s2, check_index=check_index)
    else:
        _assert_series_equal_both(s1, s2, check_index=check_index)


def test_series_invalid_param_combination():
    # [FUNCTION DOC]
    # Turning off the index check while turning on like-ordering raises a value error saying the like-ordering flag must be false if the index check is false.
    left = Series(dtype=object)
    right = Series(dtype=object)
    with pytest.raises(
        ValueError, match="check_like must be False if check_index is False"
    ):
        tm.assert_series_equal(left, right, check_index=False, check_like=True)


def test_series_equal_length_mismatch(rtol):
    # [FUNCTION DOC]
    # Uses the tolerance fixture. Series of lengths 3 and 4 raise an assertion error showing the lengths and the range indexes.
    msg = """Series are different

Series length are different
\\[left\\]:  3, RangeIndex\\(start=0, stop=3, step=1\\)
\\[right\\]: 4, RangeIndex\\(start=0, stop=4, step=1\\)"""

    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 3, 4])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_numeric_values_mismatch(rtol):
    # [FUNCTION DOC]
    # Uses the tolerance fixture. Integer series differing in one value raise an assertion error showing 33.33 percent differing, the index, and left and right values.
    msg = """Series are different

Series values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[1, 2, 3\\]
\\[right\\]: \\[1, 2, 4\\]"""

    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 4])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_categorical_values_mismatch(rtol, using_infer_string):
    # [FUNCTION DOC]
    # Uses the tolerance and infer-string fixtures. Categorical series with two swapped values raise an assertion error showing 66.67 percent differing and both categorical listings with their categories.
    dtype = "str" if using_infer_string else "object"
    msg = f"""Series are different

Series values are different \\(66\\.66667 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\['a', 'b', 'c'\\]
Categories \\(3, {dtype}\\): \\['a', 'b', 'c'\\]
\\[right\\]: \\['a', 'c', 'b'\\]
Categories \\(3, {dtype}\\): \\['a', 'b', 'c'\\]"""

    s1 = Series(Categorical(["a", "b", "c"]))
    s2 = Series(Categorical(["a", "c", "b"]))

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_datetime_values_mismatch(rtol):
    # [FUNCTION DOC]
    # Uses the tolerance fixture. Datetime series of different dates raise an assertion error showing all values (100 percent) differing as integer nanosecond values.
    msg = """Series are different

Series values are different \\(100.0 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[1514764800000000000, 1514851200000000000, 1514937600000000000\\]
\\[right\\]: \\[1549065600000000000, 1549152000000000000, 1549238400000000000\\]"""

    s1 = Series(pd.date_range("2018-01-01", periods=3, freq="D", unit="ns"))
    s2 = Series(pd.date_range("2019-02-02", periods=3, freq="D", unit="ns"))

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_categorical_mismatch(check_categorical, using_infer_string):
    # [FUNCTION DOC]
    # Uses the categorical-check and infer-string fixtures. Categorical series with different category sets raise an assertion error about the dtype attribute (showing both categorical dtypes) when categorical checking is on, and are equal when off.
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, \
categories_dtype={dtype}\\)
\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], \
ordered=False, categories_dtype={dtype}\\)"""

    s1 = Series(Categorical(["a", "b"]))
    s2 = Series(Categorical(["a", "b"], categories=list("abc")))

    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, check_categorical=check_categorical)
    else:
        _assert_series_equal_both(s1, s2, check_categorical=check_categorical)


def test_assert_series_equal_extension_dtype_mismatch():
    # [FUNCTION DOC]
    # Issue 32747. A nullable Int64 series versus its plain integer cast is equal with the dtype check off; with it on, raises an assertion error naming the dtype attribute.
    # KNOWN ISSUE: the expected message pattern writes the integer width choice as a bracketed set of characters with a vertical bar, which matches any one of those characters rather than the alternatives 32 or 64, so the pattern is looser than intended.
    # https://github.com/pandas-dev/pandas/issues/32747
    left = Series(pd.array([1, 2, 3], dtype="Int64"))
    right = left.astype(int)

    msg = """Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  Int64
\\[right\\]: int[32|64]"""

    tm.assert_series_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)


def test_assert_series_equal_interval_dtype_mismatch():
    # [FUNCTION DOC]
    # Issue 32747. An interval series versus its object cast is equal with the dtype check off and raises an assertion error naming the interval and object dtypes with it on.
    # https://github.com/pandas-dev/pandas/issues/32747
    left = Series([pd.Interval(0, 1)], dtype="interval")
    right = left.astype(object)

    msg = """Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  interval\\[int64, right\\]
\\[right\\]: object"""

    tm.assert_series_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)


def test_series_equal_series_type():
    # [FUNCTION DOC]
    # Builds a series, an identical series and an instance of a trivial series subclass. Equal series pass with the series type check on or off; the subclass instance is accepted in both orders when the check is off, and raises an assertion error saying the series classes differ in both orders when it is on.
    class MySeries(Series):
        # [CLASS DOC]
        # HELPER: trivial subclass of the series class used to test the series type check.
        pass

    s1 = Series([1, 2])
    s2 = Series([1, 2])
    s3 = MySeries([1, 2])

    tm.assert_series_equal(s1, s2, check_series_type=False)
    tm.assert_series_equal(s1, s2, check_series_type=True)

    tm.assert_series_equal(s1, s3, check_series_type=False)
    tm.assert_series_equal(s3, s1, check_series_type=False)

    with pytest.raises(AssertionError, match="Series classes are different"):
        tm.assert_series_equal(s1, s3, check_series_type=True)

    with pytest.raises(AssertionError, match="Series classes are different"):
        tm.assert_series_equal(s3, s1, check_series_type=True)


def test_series_equal_exact_for_nonnumeric():
    # [FUNCTION DOC]
    # Issue 35446. Equal string series pass exact comparison in both orders; series with swapped values raise an assertion error showing 100 percent differing and both value listings, in each direction.
    # https://github.com/pandas-dev/pandas/issues/35446
    s1 = Series(["a", "b"])
    s2 = Series(["a", "b"])
    s3 = Series(["b", "a"])

    tm.assert_series_equal(s1, s2, check_exact=True)
    tm.assert_series_equal(s2, s1, check_exact=True)

    msg = """Series are different

Series values are different \\(100\\.0 %\\)
\\[index\\]: \\[0, 1\\]
\\[left\\]:  \\[a, b\\]
\\[right\\]: \\[b, a\\]"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s3, check_exact=True)

    msg = """Series are different

Series values are different \\(100\\.0 %\\)
\\[index\\]: \\[0, 1\\]
\\[left\\]:  \\[b, a\\]
\\[right\\]: \\[a, b\\]"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s3, s1, check_exact=True)


def test_assert_series_equal_ignore_extension_dtype_mismatch():
    # [FUNCTION DOC]
    # Issue 35715. Nullable Int64 and Int32 series with the same values are equal with the dtype check off.
    # https://github.com/pandas-dev/pandas/issues/35715
    left = Series([1, 2, 3], dtype="Int64")
    right = Series([1, 2, 3], dtype="Int32")
    tm.assert_series_equal(left, right, check_dtype=False)


def test_assert_series_equal_ignore_extension_dtype_mismatch_cross_class():
    # [FUNCTION DOC]
    # Issue 35715. Nullable Int64 and numpy int64 series are equal with the dtype check off.
    # https://github.com/pandas-dev/pandas/issues/35715
    left = Series([1, 2, 3], dtype="Int64")
    right = Series([1, 2, 3], dtype="int64")
    tm.assert_series_equal(left, right, check_dtype=False)


def test_allows_duplicate_labels():
    # [FUNCTION DOC]
    # A series equals itself; a series and one with duplicate labels disallowed are equal when flags are not checked (both orders) and raise an assertion error mentioning the flags object when checked.
    # KNOWN ISSUE: the last two assertion blocks are identical (both compare left to right); the second probably was meant to swap the arguments.
    left = Series([1])
    right = Series([1]).set_flags(allows_duplicate_labels=False)
    tm.assert_series_equal(left, left)
    tm.assert_series_equal(right, right)
    tm.assert_series_equal(left, right, check_flags=False)
    tm.assert_series_equal(right, left, check_flags=False)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_series_equal(left, right)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_series_equal(left, right)


def test_assert_series_equal_identical_na(nulls_fixture):
    # [FUNCTION DOC]
    # Parametrized over missing-value scalars. A one-element series equals its copy, and an index made from it equals a deep copy of that index.
    ser = Series([nulls_fixture])

    tm.assert_series_equal(ser, ser.copy())

    # while we're here do Index too
    idx = pd.Index(ser)
    tm.assert_index_equal(idx, idx.copy(deep=True))


def test_identical_nested_series_is_equal():
    # [FUNCTION DOC]
    # GH#22400. Two series containing identical numbers and a numpy array element are equal with the default and exact comparison, for the same object and for separately built ones.
    # GH#22400
    x = Series(
        [
            0,
            0.0131142231938,
            1.77774652865e-05,
            np.array([0.4722720840328748, 0.4216929783681722]),
        ]
    )
    y = Series(
        [
            0,
            0.0131142231938,
            1.77774652865e-05,
            np.array([0.4722720840328748, 0.4216929783681722]),
        ]
    )
    # These two arrays should be equal, nesting could cause issue

    tm.assert_series_equal(x, x)
    tm.assert_series_equal(x, x, check_exact=True)
    tm.assert_series_equal(x, y)
    tm.assert_series_equal(x, y, check_exact=True)


@pytest.mark.parametrize("dtype", ["datetime64", "timedelta64"])
def test_check_dtype_false_different_reso(dtype):
    # [FUNCTION DOC]
    # GH 52449. Parametrized over datetime and timedelta. Second-unit versus millisecond-unit series with equal values: with the default dtype check an assertion error about attributes appears; with the check off they are equal. After subtracting one millisecond from the millisecond series, an assertion error that the series differ appears even with the dtype check off.
    # GH 52449
    ser_s = Series([1000213, 2131232, 21312331]).astype(f"{dtype}[s]")
    ser_ms = ser_s.astype(f"{dtype}[ms]")
    with pytest.raises(AssertionError, match="Attributes of Series are different"):
        tm.assert_series_equal(ser_s, ser_ms)
    tm.assert_series_equal(ser_ms, ser_s, check_dtype=False)

    ser_ms -= Series([1, 1, 1]).astype(f"{dtype}[ms]")

    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(ser_s, ser_ms)

    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(ser_s, ser_ms, check_dtype=False)


@pytest.mark.parametrize("dtype", ["Int64", "int64"])
def test_large_unequal_ints(dtype):
    # [FUNCTION DOC]
    # Issue 55882. Parametrized over nullable and plain 64-bit integers. Series with large integers differing in the last digits raise an assertion error.
    # https://github.com/pandas-dev/pandas/issues/55882
    left = Series([1577840521123000], dtype=dtype)
    right = Series([1577840521123543], dtype=dtype)
    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(left, right)


@pytest.mark.parametrize("dtype", [None, object])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("val", [3, 3.5])
def test_ea_and_numpy_no_dtype_check(val, check_exact, dtype):
    # [FUNCTION DOC]
    # GH#56651. Parametrized over the left dtype (inferred or object), exact flag, and value (3 or 3.5). A series and a series from a nullable array with the same values compare equal with the dtype check off.
    # GH#56651
    left = Series([1, 2, val], dtype=dtype)
    right = Series(pd.array([1, 2, val]))
    tm.assert_series_equal(left, right, check_dtype=False, check_exact=check_exact)


def test_assert_series_equal_int_tol():
    # [FUNCTION DOC]
    # GH#56646. Integer series that differ by large factors compare equal with relative tolerance 1.5 for series, frames and nullable integer arrays.
    # GH#56646
    left = Series([81, 18, 121, 38, 74, 72, 81, 81, 146, 81, 81, 170, 74, 74])
    right = Series([72, 9, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72])
    tm.assert_series_equal(left, right, rtol=1.5)

    tm.assert_frame_equal(left.to_frame(), right.to_frame(), rtol=1.5)
    tm.assert_extension_array_equal(
        left.astype("Int64").values, right.astype("Int64").values, rtol=1.5
    )


@pytest.mark.parametrize(
    "left_idx, right_idx",
    [
        (
            pd.Index([0, 0.2, 0.4, 0.6, 0.8, 1]),
            pd.Index(np.linspace(0, 1, 6)),
        ),
        (
            pd.MultiIndex.from_arrays([[0, 0, 0, 0, 1, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]]),
            pd.MultiIndex.from_arrays([[0, 0, 0, 0, 1, 1], np.linspace(0, 1, 6)]),
        ),
        (
            pd.MultiIndex.from_arrays(
                [["a", "a", "a", "b", "b", "b"], [1, 2, 3, 4, 5, 10000000000001]]
            ),
            pd.MultiIndex.from_arrays(
                [["a", "a", "a", "b", "b", "b"], [1, 2, 3, 4, 5, 10000000000002]]
            ),
        ),
        pytest.param(
            pd.Index([1, 2, 3, 4, 5, 10000000000001]),
            pd.Index([1, 2, 3, 4, 5, 10000000000002]),
            marks=pytest.mark.xfail(reason="check_exact_index defaults to True"),
        ),
        pytest.param(
            pd.MultiIndex.from_arrays(
                [[0, 0, 0, 0, 1, 1], [1, 2, 3, 4, 5, 10000000000001]]
            ),
            pd.MultiIndex.from_arrays(
                [[0, 0, 0, 0, 1, 1], [1, 2, 3, 4, 5, 10000000000002]]
            ),
            marks=pytest.mark.xfail(reason="check_exact_index defaults to True"),
        ),
    ],
)
def test_assert_series_equal_check_exact_index_default(left_idx, right_idx):
    # [FUNCTION DOC]
    # GH#57067. Parametrized over index pairs: float indexes with tiny differences (plain and multi-level, which pass because the index comparison tolerates floating error), integer indexes differing in a large last value (expected failures because exact index comparison is the default), and multi-index with such a level (expected failure). Series of zeros with those indexes, and their frames, are asserted equal.
    # GH#57067
    ser1 = Series(np.zeros(6, dtype=int), left_idx)
    ser2 = Series(np.zeros(6, dtype=int), right_idx)
    tm.assert_series_equal(ser1, ser2)
    tm.assert_frame_equal(ser1.to_frame(), ser2.to_frame())
