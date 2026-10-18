# [MODULE DOC]
# Tests for the testing-utility function that asserts two interval arrays are equal: identical arrays, differing closed side, differing lengths, differing left bounds, differing right bounds.

import pytest

from pandas import (
    Interval,
    interval_range,
)
import pandas._testing as tm
from pandas.arrays import IntervalArray


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 0, "periods": 4},
        {"start": 1, "periods": 5},
        {"start": 5, "end": 10, "closed": "left"},
    ],
)
def test_interval_array_equal(kwargs):
    # [FUNCTION DOC]
    # Parametrized over three keyword sets for building an interval range. An interval array compared with itself passes.
    arr = interval_range(**kwargs).values
    tm.assert_interval_array_equal(arr, arr)


def test_interval_array_equal_closed_mismatch():
    # [FUNCTION DOC]
    # Interval arrays identical except for the closed side (left versus right) raise an assertion error saying the closed attribute differs.
    kwargs = {"start": 0, "periods": 5}
    arr1 = interval_range(closed="left", **kwargs).values
    arr2 = interval_range(closed="right", **kwargs).values

    msg = """\
IntervalArray are different

Attribute "closed" are different
\\[left\\]:  left
\\[right\\]: right"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_periods_mismatch():
    # [FUNCTION DOC]
    # Interval arrays with five versus six periods raise an assertion error saying the left bounds have different shapes (5 versus 6).
    kwargs = {"start": 0}
    arr1 = interval_range(periods=5, **kwargs).values
    arr2 = interval_range(periods=6, **kwargs).values

    msg = """\
IntervalArray.left are different

IntervalArray.left shapes are different
\\[left\\]:  \\(5,\\)
\\[right\\]: \\(6,\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_end_mismatch():
    # [FUNCTION DOC]
    # Ranges with the same start and period count but different ends (10 versus 20) raise an assertion error saying 80 percent of the left bounds differ, listing them.
    kwargs = {"start": 0, "periods": 5}
    arr1 = interval_range(end=10, **kwargs).values
    arr2 = interval_range(end=20, **kwargs).values

    msg = """\
IntervalArray.left are different

IntervalArray.left values are different \\(80.0 %\\)
\\[left\\]:  \\[0, 2, 4, 6, 8\\]
\\[right\\]: \\[0, 4, 8, 12, 16\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_start_mismatch():
    # [FUNCTION DOC]
    # Ranges with different starts raise an assertion error saying all left bounds (100 percent) differ.
    kwargs = {"periods": 4}
    arr1 = interval_range(start=0, **kwargs).values
    arr2 = interval_range(start=1, **kwargs).values

    msg = """\
IntervalArray.left are different

IntervalArray.left values are different \\(100.0 %\\)
\\[left\\]:  \\[0, 1, 2, 3\\]
\\[right\\]: \\[1, 2, 3, 4\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


def test_interval_array_equal_end_mismatch_only():
    # [FUNCTION DOC]
    # Interval arrays with equal left bounds but differing right bounds raise an assertion error naming the right bounds with 50 percent differing.
    arr1 = IntervalArray([Interval(0, 1), Interval(0, 5)])
    arr2 = IntervalArray([Interval(0, 1), Interval(0, 6)])

    msg = """\
IntervalArray.right are different

IntervalArray.right values are different \\(50.0 %\\)
\\[left\\]:  \\[1, 5\\]
\\[right\\]: \\[1, 6\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)
