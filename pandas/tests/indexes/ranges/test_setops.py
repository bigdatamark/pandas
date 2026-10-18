# [MODULE DOC]
# Tests for set operations (intersection, union, difference, symmetric
# difference) involving the range-based integer index. Verifies both the values
# and whether the result stays a range index (exact type) or falls back to a
# general index, name propagation, and the sort keyword. Includes a
# property-based test (using the hypothesis library) comparing range difference
# against integer-index difference, and a private checker helper. The names,
# sort fixtures come from the shared test configuration.

from datetime import (
    datetime,
    timedelta,
)

from hypothesis import (
    assume,
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm


class TestRangeIndexSetOps:
    # [CLASS DOC]
    # Test class collecting set-operation tests for range indexes.
    @pytest.mark.parametrize("dtype", [None, "int64", "uint64"])
    def test_intersection_mismatched_dtype(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over dtype (default, int64, uint64). Intersecting an integer index
        # built from a range with the same values as float64 should cast to float rather
        # than object. Checks: equal indexes take a fast path; shifted (non-equal)
        # overlaps return the float values; an empty other returns an empty float index,
        # in both operand orders, with exact type checks.
        # check that we cast to float, not object
        index = RangeIndex(start=0, stop=20, step=2, name="foo")
        index = Index(index, dtype=dtype)

        flt = index.astype(np.float64)

        # bc index.equals(flt), we go through fastpath and get RangeIndex back
        result = index.intersection(flt)
        tm.assert_index_equal(result, index, exact=True)

        result = flt.intersection(index)
        tm.assert_index_equal(result, flt, exact=True)

        # neither empty, not-equals
        result = index.intersection(flt[1:])
        tm.assert_index_equal(result, flt[1:], exact=True)

        result = flt[1:].intersection(index)
        tm.assert_index_equal(result, flt[1:], exact=True)

        # empty other
        result = index.intersection(flt[:0])
        tm.assert_index_equal(result, flt[:0], exact=True)

        result = flt[:0].intersection(index)
        tm.assert_index_equal(result, flt[:0], exact=True)

    def test_intersection_empty(self, sort, names):
        # [FUNCTION DOC]
        # Uses the sort and names fixtures. The intersection of a named range with an
        # empty index (either operand being empty) returns an empty range slice with
        # the result name taken per the usual name-resolution rule (names[2]).
        # name retention on empty intersections
        index = RangeIndex(start=0, stop=20, step=2, name=names[0])

        # empty other
        result = index.intersection(index[:0].rename(names[1]), sort=sort)
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)

        # empty self
        result = index[:0].intersection(index.rename(names[1]), sort=sort)
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)

    def test_intersection(self, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Intersection of an even-valued range with: a plain
        # integer index, an increasing range, a decreasing range (and reversed operand
        # order), two decreasing ranges (expected a decreasing range), non-overlapping
        # ranges (empty range with preserved name when both names agree), and empty
        # ranges. Expected values come from numpy's sorted set intersection; type is
        # compared exactly or loosely depending on the case.
        # intersect with Index with dtype int64
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(1, 6))
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

        # intersect with increasing RangeIndex
        other = RangeIndex(1, 6)
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected, exact="equiv")

        # intersect with decreasing RangeIndex
        other = RangeIndex(5, 0, -1)
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected, exact="equiv")

        # reversed (GH 17296)
        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected, exact="equiv")

        # GH 17296: intersect two decreasing RangeIndexes
        first = RangeIndex(10, -2, -2)
        other = RangeIndex(5, -4, -1)
        expected = RangeIndex(start=4, stop=-2, step=-2)
        result = first.intersection(other, sort=sort)
        tm.assert_index_equal(result, expected)

        # reversed
        result = other.intersection(first, sort=sort)
        tm.assert_index_equal(result, expected)

        index = RangeIndex(5, name="foo")

        # intersect of non-overlapping indices
        other = RangeIndex(5, 10, 1, name="foo")
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1, name="foo")
        tm.assert_index_equal(result, expected)

        other = RangeIndex(-1, -5, -1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        # intersection of empty indices
        other = RangeIndex(0, 0, 1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection_non_overlapping_gcd(self, sort, names):
        # [FUNCTION DOC]
        # Two ranges whose steps share a gcd but whose start residues differ never
        # meet; the result is an empty range with the resolved name (names fixture).
        # intersection of non-overlapping values based on start value and gcd
        index = RangeIndex(1, 10, 2, name=names[0])
        other = RangeIndex(0, 10, 4, name=names[1])
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1, name=names[2])
        tm.assert_index_equal(result, expected)

    def test_union_noncomparable(self, sort):
        # [FUNCTION DOC]
        # Union of a range with an object index of current datetimes (not comparable with
        # integers) in both orders must not fail for either sort setting; the result is
        # the plain concatenation of the two operands as a general index.
        # corner case, Index with non-int64 dtype
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        result = index.union(other, sort=sort)
        expected = Index(np.concatenate((index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(index, sort=sort)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "idx1, idx2, expected_sorted, expected_notsorted",
        [
            (
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
            ),
            (
                RangeIndex(0, 10, 1),
                RangeIndex(5, 20, 1),
                RangeIndex(0, 20, 1),
                RangeIndex(0, 20, 1),
            ),
            (
                RangeIndex(0, 10, 1),
                RangeIndex(10, 20, 1),
                RangeIndex(0, 20, 1),
                RangeIndex(0, 20, 1),
            ),
            (
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
            ),
            (
                RangeIndex(0, -10, -1),
                RangeIndex(-10, -20, -1),
                RangeIndex(-19, 1, 1),
                RangeIndex(0, -20, -1),
            ),
            (
                RangeIndex(0, 10, 2),
                RangeIndex(1, 10, 2),
                RangeIndex(0, 10, 1),
                Index(list(range(0, 10, 2)) + list(range(1, 10, 2))),
            ),
            (
                RangeIndex(0, 11, 2),
                RangeIndex(1, 12, 2),
                RangeIndex(0, 12, 1),
                Index(list(range(0, 11, 2)) + list(range(1, 12, 2))),
            ),
            (
                RangeIndex(0, 21, 4),
                RangeIndex(-2, 24, 4),
                RangeIndex(-2, 24, 2),
                Index(list(range(0, 21, 4)) + list(range(-2, 24, 4))),
            ),
            (
                RangeIndex(0, -20, -2),
                RangeIndex(-1, -21, -2),
                RangeIndex(-19, 1, 1),
                Index(list(range(0, -20, -2)) + list(range(-1, -21, -2))),
            ),
            (
                RangeIndex(0, 100, 5),
                RangeIndex(0, 100, 20),
                RangeIndex(0, 100, 5),
                RangeIndex(0, 100, 5),
            ),
            (
                RangeIndex(0, -100, -5),
                RangeIndex(5, -100, -20),
                RangeIndex(-95, 10, 5),
                Index([*list(range(0, -100, -5)), 5]),
            ),
            (
                RangeIndex(0, -11, -1),
                RangeIndex(1, -12, -4),
                RangeIndex(-11, 2, 1),
                Index([*list(range(0, -11, -1)), 1, -11]),
            ),
            (RangeIndex(0), RangeIndex(0), RangeIndex(0), RangeIndex(0)),
            (
                RangeIndex(0, -10, -2),
                RangeIndex(0),
                RangeIndex(0, -10, -2),
                RangeIndex(0, -10, -2),
            ),
            (
                RangeIndex(0, 100, 2),
                RangeIndex(100, 150, 200),
                RangeIndex(0, 102, 2),
                RangeIndex(0, 102, 2),
            ),
            (
                RangeIndex(0, -100, -2),
                RangeIndex(-100, 50, 102),
                RangeIndex(-100, 4, 2),
                Index([*list(range(0, -100, -2)), -100, 2]),
            ),
            (
                RangeIndex(0, -100, -1),
                RangeIndex(0, -50, -3),
                RangeIndex(-99, 1, 1),
                RangeIndex(0, -100, -1),
            ),
            (
                RangeIndex(0, 1, 1),
                RangeIndex(5, 6, 10),
                RangeIndex(0, 6, 5),
                RangeIndex(0, 10, 5),
            ),
            (
                RangeIndex(0, 10, 5),
                RangeIndex(-5, -6, -20),
                RangeIndex(-5, 10, 5),
                Index([0, 5, -5]),
            ),
            (
                RangeIndex(0, 3, 1),
                RangeIndex(4, 5, 1),
                Index([0, 1, 2, 4]),
                Index([0, 1, 2, 4]),
            ),
            (
                RangeIndex(0, 10, 1),
                Index([], dtype=np.int64),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
            ),
            (
                RangeIndex(0),
                Index([1, 5, 6]),
                Index([1, 5, 6]),
                Index([1, 5, 6]),
            ),
            # GH 43885
            (
                RangeIndex(0, 10),
                RangeIndex(0, 5),
                RangeIndex(0, 10),
                RangeIndex(0, 10),
            ),
        ],
        ids=lambda x: repr(x) if isinstance(x, RangeIndex) else x,
    )
    def test_union_sorted(self, idx1, idx2, expected_sorted, expected_notsorted):
        # [FUNCTION DOC]
        # Parametrized over many pairs of range (or empty/integer) indexes with the
        # expected result when sort is unspecified (sorted, often a merged range) and
        # when sort is false (concatenation order). Checks the sorted union in both
        # operand orders with exact type, that the unsorted union has the given order,
        # and that a range converted to a plain integer index gives an equivalent result.
        res1 = idx1.union(idx2, sort=None)
        tm.assert_index_equal(res1, expected_sorted, exact=True)

        res1 = idx1.union(idx2, sort=False)
        tm.assert_index_equal(res1, expected_notsorted, exact=True)

        res2 = idx2.union(idx1, sort=None)
        res3 = Index(idx1._values, name=idx1.name).union(idx2, sort=None)
        tm.assert_index_equal(res2, expected_sorted, exact=True)
        tm.assert_index_equal(res3, expected_sorted, exact="equiv")

    def test_union_same_step_misaligned(self):
        # [FUNCTION DOC]
        # Two ranges with the same step but offset starts do not merge into a range; the union is a general integer index with the sorted values (exact type).
        # GH#44019
        left = RangeIndex(range(0, 20, 4))
        right = RangeIndex(range(1, 21, 4))

        result = left.union(right)
        expected = Index([0, 1, 4, 5, 8, 9, 12, 13, 16, 17])
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference(self):
        # [FUNCTION DOC]
        # Difference of range indexes where the result may remain a range: with itself (empty, keeping the name), with an empty differently named index (the name is dropped), removing leading or trailing slices, reversed self with default or false sort, and an interior hole which produces a general integer index.
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        result = obj.difference(obj)
        expected = RangeIndex.from_range(range(0), name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        result = obj.difference(expected.rename("bar"))
        tm.assert_index_equal(result, obj.rename(None), exact=True)

        result = obj.difference(obj[:3])
        tm.assert_index_equal(result, obj[3:], exact=True)

        result = obj.difference(obj[-3:])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        # Flipping the step of 'other' doesn't affect the result, but
        #  flipping the stepof 'self' does when sort=None
        result = obj[::-1].difference(obj[-3:])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        result = obj[::-1].difference(obj[-3:], sort=False)
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        result = obj[::-1].difference(obj[-3:][::-1])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        result = obj[::-1].difference(obj[-3:][::-1], sort=False)
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        result = obj.difference(obj[2:6])
        expected = Index([1, 2, 7, 8, 9], name="foo")
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_sort(self):
        # [FUNCTION DOC]
        # Checks the sort keyword on a descending integer index: default sorts the result ascending, false preserves the original (descending) order, and with no overlap and default sort the whole index is returned sorted ascending.
        # GH#44085 ensure we respect the sort keyword

        idx = Index(range(4))[::-1]
        other = Index(range(3, 4))

        result = idx.difference(other)
        expected = Index(range(3))
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.difference(other, sort=False)
        expected = expected[::-1]
        tm.assert_index_equal(result, expected, exact=True)

        # case where the intersection is empty
        other = range(10, 12)
        result = idx.difference(other, sort=None)
        expected = idx[::-1]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_mismatched_step(self):
        # [FUNCTION DOC]
        # Subtracting a stepped subset (every other element, with different step) from a range returns the remaining alternate elements as a range; with the reversed self and sort disabled the order is reversed.
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        result = obj.difference(obj[::2])
        expected = obj[1::2]
        tm.assert_index_equal(result, expected, exact=True)

        result = obj[::-1].difference(obj[::2], sort=False)
        tm.assert_index_equal(result, expected[::-1], exact=True)

        result = obj.difference(obj[1::2])
        expected = obj[::2]
        tm.assert_index_equal(result, expected, exact=True)

        result = obj[::-1].difference(obj[1::2], sort=False)
        tm.assert_index_equal(result, expected[::-1], exact=True)

    def test_difference_interior_overlap_endpoints_preserved(self):
        # [FUNCTION DOC]
        # Removing an interior section whose complement is evenly spaced (here the two endpoints) still returns a range index of step three.
        left = RangeIndex(range(4))
        right = RangeIndex(range(1, 3))

        result = left.difference(right)
        expected = RangeIndex(0, 4, 3)
        assert expected.tolist() == [0, 3]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_endpoints_overlap_interior_preserved(self):
        # [FUNCTION DOC]
        # Removing values that overlap only the endpoints of the left range leaves an evenly spaced interior and the result is a range index.
        left = RangeIndex(-8, 20, 7)
        right = RangeIndex(13, -9, -3)

        result = left.difference(right)
        expected = RangeIndex(-1, 13, 7)
        assert expected.tolist() == [-1, 6]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_interior_non_preserving(self):
        # [FUNCTION DOC]
        # Uses a general integer index of consecutive values as left operand (the "range index" is not preserved): removing one interior value, every third value, or alternate values from one side yields plain integer indexes with the expected remaining values (exact type).
        # case with intersection of length 1 but RangeIndex is not preserved
        idx = Index(range(10))

        other = idx[3:4]
        result = idx.difference(other)
        expected = Index([0, 1, 2, 4, 5, 6, 7, 8, 9])
        tm.assert_index_equal(result, expected, exact=True)

        # case with other.step / self.step > 2
        other = idx[::3]
        result = idx.difference(other)
        expected = Index([1, 2, 4, 5, 7, 8])
        tm.assert_index_equal(result, expected, exact=True)

        # cases with only reaching one end of left
        obj = Index(range(20))
        other = obj[:10:2]
        result = obj.difference(other)
        expected = Index([1, 3, 5, 7, 9, *list(range(10, 20))])
        tm.assert_index_equal(result, expected, exact=True)

        other = obj[1:11:2]
        result = obj.difference(other)
        expected = Index([0, 2, 4, 6, 8, 10, *list(range(11, 20))])
        tm.assert_index_equal(result, expected, exact=True)

    def test_symmetric_difference(self):
        # [FUNCTION DOC]
        # Symmetric difference of range indexes: with itself gives an empty named range; with an empty differently named range gives the left side with the name dropped; overlapping slices give a general index with the non-shared values and the common name; a disjoint adjacent range gives a merged range; a disjoint but not adjacent set gives a general integer index.
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        left = RangeIndex.from_range(range(1, 10), name="foo")

        result = left.symmetric_difference(left)
        expected = RangeIndex.from_range(range(0), name="foo")
        tm.assert_index_equal(result, expected)

        result = left.symmetric_difference(expected.rename("bar"))
        tm.assert_index_equal(result, left.rename(None))

        result = left[:-2].symmetric_difference(left[2:])
        expected = Index([1, 2, 8, 9], name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        right = RangeIndex.from_range(range(10, 15))

        result = left.symmetric_difference(right)
        expected = RangeIndex.from_range(range(1, 15))
        tm.assert_index_equal(result, expected)

        result = left.symmetric_difference(right[1:])
        expected = Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])
        tm.assert_index_equal(result, expected, exact=True)


def assert_range_or_not_is_rangelike(index):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the property-based test. Takes an index. If it is not a range index and is non-empty, asserts that its consecutive differences are not all equal, i.e. that it could not have been expressed as a range. Passes silently for range indexes and empty indexes.
    """
    Check that we either have a RangeIndex or that this index *cannot*
    be represented as a RangeIndex.
    """
    if not isinstance(index, RangeIndex) and len(index) > 0:
        diff = index[:-1] - index[1:]
        assert not (diff == diff[0]).all()


@pytest.mark.slow
@given(
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
)
def test_range_difference(start1, stop1, step1, start2, stop2, step2):
    # [FUNCTION DOC]
    # Property-based, marked slow, with six hypothesis integer inputs between minus twenty and twenty (start, stop, step for each of two ranges; zero steps are rejected through assumptions). Builds two range indexes and checks, for sort unspecified and for sort disabled, that the range difference (a) is a range whenever representable, using the helper, and (b) equals the difference computed on equivalent plain integer indexes (loose type comparison).
    # test that
    #  a) we match Index[int64].difference and
    #  b) we return RangeIndex whenever it is possible to do so.
    assume(step1 != 0)
    assume(step2 != 0)

    left = RangeIndex(start1, stop1, step1)
    right = RangeIndex(start2, stop2, step2)

    result = left.difference(right, sort=None)
    assert_range_or_not_is_rangelike(result)

    left_int64 = Index(left.to_numpy())
    right_int64 = Index(right.to_numpy())

    alt = left_int64.difference(right_int64, sort=None)
    tm.assert_index_equal(result, alt, exact="equiv")

    result = left.difference(right, sort=False)
    assert_range_or_not_is_rangelike(result)

    alt = left_int64.difference(right_int64, sort=False)
    tm.assert_index_equal(result, alt, exact="equiv")
