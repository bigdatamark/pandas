# [MODULE DOC]
# Tests for set operations on numeric indexes: unions with non-numeric and
# mixed numeric types, intersections (including unsigned values beyond the signed
# range), differences with string indexes, symmetric difference and sort behavior.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.indexes.api import (
    Index,
    RangeIndex,
)


@pytest.fixture
def index_large():
    # [FUNCTION DOC]
    # HELPER: fixture.
    # Returns an unsigned 64-bit index of five values just above the signed 64-bit
    # limit.
    # KNOWN ISSUE: this fixture is not used by any test in this module (the one test needing such values builds its own); the comment refers to a class that no longer exists.
    # large values used in TestUInt64Index where no compat needed with int64/float64
    large = [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25]
    return Index(large, dtype=np.uint64)


class TestSetOps:
    # [CLASS DOC]
    # Groups set operation tests for numeric indexes.
    @pytest.mark.parametrize("dtype", ["f8", "u8", "i8"])
    def test_union_non_numeric(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtypes. Union with an object index of datetimes yields an
        # object index that is the concatenation in operand order.
        # corner case, non-numeric
        index = Index(np.arange(5, dtype=dtype), dtype=dtype)
        assert index.dtype == dtype

        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        result = index.union(other)
        expected = Index(np.concatenate((index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    def test_intersection(self):
        # [FUNCTION DOC]
        # Intersection of a 64-bit integer index with an overlapping index, in both
        # orders, equals the sorted common values.
        index = Index(range(5), dtype=np.int64)

        other = Index([1, 2, 3, 4, 5])
        result = index.intersection(other)
        expected = Index(range(1, 5))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    def test_int_float_union_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over signed and unsigned integer. Union with floats gives a
        # sorted float index in either order (GitHub 26778).
        # https://github.com/pandas-dev/pandas/issues/26778
        # [u]int | float -> float
        index = Index([0, 2, 3], dtype=dtype)
        other = Index([0.5, 1.5], dtype=np.float64)
        expected = Index([0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float64)
        result = index.union(other)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_float_union_dtype(self):
        # [FUNCTION DOC]
        # Union of a range index and floats gives a float index, either order (GitHub
        # 26778).
        # https://github.com/pandas-dev/pandas/issues/26778
        index = RangeIndex(start=0, stop=3)
        other = Index([0.5, 1.5], dtype=np.float64)
        result = index.union(other)
        expected = Index([0.0, 0.5, 1, 1.5, 2.0], dtype=np.float64)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_uint64_union_dtype(self):
        # [FUNCTION DOC]
        # Union of a range index and unsigned integers gives an object index, either
        # order (GitHub 26778).
        # https://github.com/pandas-dev/pandas/issues/26778
        index = RangeIndex(start=0, stop=3)
        other = Index([0, 10], dtype=np.uint64)
        result = index.union(other)
        expected = Index([0, 1, 2, 10], dtype=object)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_float64_index_difference(self):
        # [FUNCTION DOC]
        # Difference between a float index and a string index returns the left operand
        # unchanged (GitHub 35217).
        # https://github.com/pandas-dev/pandas/issues/35217
        float_index = Index([1.0, 2, 3])
        string_index = Index(["1", "2", "3"])

        result = float_index.difference(string_index)
        tm.assert_index_equal(result, float_index)

        result = string_index.difference(float_index)
        tm.assert_index_equal(result, string_index)

    def test_intersection_uint64_outside_int64_range(self):
        # [FUNCTION DOC]
        # Intersection of unsigned values above the signed range with another index
        # equals the sorted common values in either order.
        index_large = Index(
            [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25],
            dtype=np.uint64,
        )
        other = Index([2**63, 2**63 + 5, 2**63 + 10, 2**63 + 15, 2**63 + 20])
        result = index_large.intersection(other)
        expected = Index(np.sort(np.intersect1d(index_large.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index_large)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index_large.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index2_name,keeps_name",
        [
            ("index", True),
            ("other", False),
        ],
    )
    def test_intersection_monotonic(self, index2_name, keeps_name, sort):
        # [FUNCTION DOC]
        # Parametrized over the second index name (same or different), sort fixture.
        # Intersection of unsorted integers keeps the name only if names match, and is
        # sorted when sort is none.
        index2 = Index([4, 7, 6, 5, 3], name=index2_name)
        index1 = Index([5, 3, 2, 4, 1], name="index")
        expected = Index([5, 3, 4])

        if keeps_name:
            expected.name = "index"

        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self, sort):
        # [FUNCTION DOC]
        # Symmetric difference with the sort fixture gives the two exclusive values,
        # name dropped because names differ; sorted when sort is none.
        # smoke
        index1 = Index([5, 2, 3, 4], name="index1")
        index2 = Index([2, 3, 4, 1])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = Index([5, 1])
        if sort is not None:
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result, expected.sort_values())
        assert result.name is None
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)


class TestSetOpsSort:
    # [CLASS DOC]
    # Groups sort-related union tests.
    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_other_special(self, slice_):
        # [FUNCTION DOC]
        # Parametrized over a full and an empty slice. Default and disabled sort give
        # the original unsorted order in both directions (GH 24959).
        # https://github.com/pandas-dev/pandas/issues/24959

        idx = Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]
        tm.assert_index_equal(idx.union(other), idx)
        tm.assert_index_equal(other.union(idx), idx)

        # sort=False
        tm.assert_index_equal(idx.union(other, sort=False), idx)

    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_special_true(self, slice_):
        # [FUNCTION DOC]
        # Parametrized over a full and an empty slice. Sort enabled returns the sorted
        # values.
        idx = Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]

        result = idx.union(other, sort=True)
        expected = Index([0, 1, 2])
        tm.assert_index_equal(result, expected)
