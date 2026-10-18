# [MODULE DOC]
# Tests for Series.equals, covering NaN positions, object data containing arrays,
# matching and mismatched NA kinds, None versus NaN, and a former false-equality
# bug between None and floats. Uses the nulls_fixture and nulls_fixture2
# fixtures. No import-time side effects.

import copy

import numpy as np
import pytest

from pandas._libs.missing import is_matching_na

from pandas.core.dtypes.common import is_float

from pandas import (
    Index,
    MultiIndex,
    Series,
)


@pytest.mark.parametrize(
    "arr, idx",
    [
        ([1, 2, 3, 4], [0, 2, 1, 3]),
        ([1, np.nan, 3, np.nan], [0, 2, 1, 3]),
        (
            [1, np.nan, 3, np.nan],
            MultiIndex.from_tuples([(0, "a"), (1, "b"), (2, "c"), (3, "c")]),
        ),
    ],
)
def test_equals(arr, idx):
    # [FUNCTION DOC]
    # Parametrized over integer data with permuted index, data with NaN, and data with
    # a MultiIndex. A Series equals its copy; after changing the second value they no
    # longer match.
    s1 = Series(arr, index=idx)
    s2 = s1.copy()
    assert s1.equals(s2)

    s1[1] = 9
    assert not s1.equals(s2)


@pytest.mark.parametrize(
    "val", [1, 1.1, 1 + 1j, True, "abc", [1, 2], (1, 2), {1, 2}, {"a": 1}, None]
)
def test_equals_list_array(val):
    # [FUNCTION DOC]
    # Regression test (GH20676), parametrized over scalar, list, tuple, set and dict
    # values: a Series whose elements are NumPy arrays equals its copy, and replacing
    # the second element with another kind of object makes them unequal.
    # GH20676 Verify equals operator for list of Numpy arrays
    arr = np.array([1, 2])
    s1 = Series([arr, arr])
    s2 = s1.copy()
    assert s1.equals(s2)

    s1[1] = val
    assert not s1.equals(s2)


def test_equals_false_negative():
    # [FUNCTION DOC]
    # Regression test (GH8437): object Series with a NaN, created through different
    # routes (copy, assigned element by element), compare as equal where expected;
    # tests no false negatives for object dtype.
    # GH8437 Verify false negative behavior of equals function for dtype object
    arr = [False, np.nan]
    s1 = Series(arr)
    s2 = s1.copy()
    s3 = Series(index=range(2), dtype=object)
    s4 = s3.copy()
    s5 = s3.copy()
    s6 = s3.copy()

    s3[:-1] = s4[:-1] = s5[0] = s6[0] = False
    assert s1.equals(s1)
    assert s1.equals(s2)
    assert s1.equals(s3)
    assert s1.equals(s4)
    assert s1.equals(s5)
    assert s5.equals(s6)


def test_equals_matching_nas():
    # [FUNCTION DOC]
    # Object-dtype Series holding NaT of datetime and timedelta kinds, and a float
    # NaN, that are matching but not identical objects: the Series, an Index built
    # from them, and their arrays are all equal.
    # matching but not identical NAs
    left = Series([np.datetime64("NaT", "ns")], dtype=object)
    right = Series([np.datetime64("NaT", "ns")], dtype=object)
    assert left.equals(right)
    assert Index(left).equals(Index(right))
    assert left.array.equals(right.array)

    left = Series([np.timedelta64("NaT", "ns")], dtype=object)
    right = Series([np.timedelta64("NaT", "ns")], dtype=object)
    assert left.equals(right)
    assert Index(left).equals(Index(right))
    assert left.array.equals(right.array)

    left = Series([np.float64("NaN")], dtype=object)
    right = Series([np.float64("NaN")], dtype=object)
    assert left.equals(right)
    assert Index(left, dtype=left.dtype).equals(Index(right, dtype=right.dtype))
    assert left.array.equals(right.array)


def test_equals_mismatched_nas(nulls_fixture, nulls_fixture2):
    # [FUNCTION DOC]
    # Regression test (GH#39650), parametrized over pairs of null scalars from the
    # fixtures. Object Series each holding one NA value are equal when the two NA
    # kinds match, or when one is None and the other a float NaN; they differ for all
    # other pairs.
    # GH#39650
    left = nulls_fixture
    right = nulls_fixture2
    if hasattr(right, "copy"):
        right = right.copy()
    else:
        right = copy.copy(right)

    ser = Series([left], dtype=object)
    ser2 = Series([right], dtype=object)

    if is_matching_na(left, right):
        assert ser.equals(ser2)
    elif (left is None and is_float(right)) or (right is None and is_float(left)):
        assert ser.equals(ser2)
    else:
        assert not ser.equals(ser2)


def test_equals_none_vs_nan():
    # [FUNCTION DOC]
    # Regression test (GH#39650): an object Series with None equals one with NaN in
    # the same position, and so do an Index and the arrays built from them.
    # GH#39650
    ser = Series([1, None], dtype=object)
    ser2 = Series([1, np.nan], dtype=object)

    assert ser.equals(ser2)
    assert Index(ser, dtype=ser.dtype).equals(Index(ser2, dtype=ser2.dtype))
    assert ser.array.equals(ser2.array)


def test_equals_None_vs_float():
    # [FUNCTION DOC]
    # Regression test (GH#44190): an object Series of floats and NaN is not equal to
    # a Series of all None (previously reported equal by a bug), in both operand
    # orders, for Series, frames and Index.
    # GH#44190
    left = Series([-np.inf, np.nan, -1.0, 0.0, 1.0, 10 / 3, np.inf], dtype=object)
    right = Series([None] * len(left))

    # these series were found to be equal due to a bug, check that they are correctly
    # found to not equal
    assert not left.equals(right)
    assert not right.equals(left)
    assert not left.to_frame().equals(right.to_frame())
    assert not right.to_frame().equals(left.to_frame())
    assert not Index(left, dtype="object").equals(Index(right, dtype="object"))
    assert not Index(right, dtype="object").equals(Index(left, dtype="object"))
