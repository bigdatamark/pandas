"""
The tests in this package are to ensure the proper resultant dtypes of
set operations.
"""

# [MODULE DOC]
# Tests that set operations (union, intersection, difference, symmetric difference) on indexes produce the proper result dtypes, names and contents. Covers combinations of dtypes, name retention for the many name combinations, behavior with array-like and series arguments, categorical operands, comparisons against the set-operation modules of other index types, and the sort keyword. Defines one equality helper, a dtype-list fixture, an alias fixture so two independent flat-index parameter axes can be combined, a family of module-level tests, a class of tests run against every index in the shared fixture, tests of duplicate handling in unions and intersections, a class of tests on unsorted/name behavior for string and numeric indexes, and tests that results do not share mutable state with the operands.

from datetime import datetime
import operator

import numpy as np
import pytest

from pandas._libs import lib
import pandas.util._test_decorators as td

from pandas.core.dtypes.cast import find_common_type

from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DatetimeTZDtype,
    Index,
    MultiIndex,
    PeriodDtype,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.api.types import (
    is_signed_integer_dtype,
    pandas_dtype,
)


def equal_contents(arr1, arr2) -> bool:
    # [FUNCTION DOC]
    # HELPER: takes two iterables and returns true when the sets of their unique elements are equal, ignoring order and multiplicity.
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)


@pytest.fixture(
    params=[
        *tm.ALL_REAL_NUMPY_DTYPES,
        "object",
        "category",
        "datetime64[ns]",
        "timedelta64[ns]",
    ]
)
def any_dtype_for_small_pos_integer_indexes(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture returning dtypes that can be given to an index constructed from small positive integers (all real numpy dtypes, object, category, datetime64 and timedelta64 in nanoseconds); each such dtype gives a valid index of the right subclass.
    """
    Dtypes that can be given to an Index with small positive integers.

    This means that for any dtype `x` in the params list, `Index([1, 2, 3], dtype=x)` is
    valid and gives the correct Index (sub-)class.
    """
    return request.param


@pytest.fixture
def index_flat2(index_flat):
    # [FUNCTION DOC]
    # HELPER: fixture that simply returns the flat-index fixture, so that a test can request two independently parametrized flat indexes (the cross product of fixture parameters).
    return index_flat


def test_union_same_types(index):
    # [FUNCTION DOC]
    # Uses the shared index fixture. The union of the sorted index with an equally sorted copy keeps the dtype.
    # Union with a non-unique, non-monotonic index raises error
    # Only needed for bool index factory
    idx1 = index.sort_values()
    idx2 = index.sort_values()
    assert idx1.union(idx2).dtype == idx1.dtype


def test_union_different_types(index_flat, index_flat2, request):
    # [FUNCTION DOC]
    # Takes two flat indexes (full cross product). Pairs of non-unique integer and boolean indexes are allowed to fail (True equals one issue). Computes the expected common dtype of the two; expects a runtime warning for complex versus non-numeric combinations ("'<' not supported between"), and a deprecation warning for period versus categorical (itself allowed to fail if the warning is not produced on all builds). Sorts both and takes the union in both orders; the result dtype is object when one operand is unsigned 64-bit and the other signed, otherwise the common dtype.
    # This test only considers combinations of indices
    # GH 23525
    idx1 = index_flat
    idx2 = index_flat2

    if (
        not idx1.is_unique
        and not idx2.is_unique
        and idx1.dtype.kind == "i"
        and idx2.dtype.kind == "b"
    ) or (
        not idx2.is_unique
        and not idx1.is_unique
        and idx2.dtype.kind == "i"
        and idx1.dtype.kind == "b"
    ):
        # Each condition had idx[1|2].is_monotonic_decreasing
        # but failed when e.g.
        # idx1 = Index(
        # [True, True, True, True, True, True, True, True, False, False], dtype='bool'
        # )
        # idx2 = Index([0, 0, 1, 1, 2, 2], dtype='int64')
        mark = pytest.mark.xfail(
            reason="GH#44000 True==1", raises=ValueError, strict=False
        )
        request.applymarker(mark)

    common_dtype = find_common_type([idx1.dtype, idx2.dtype])

    warn = None
    msg = "'<' not supported between"
    if not len(idx1) or not len(idx2):
        pass
    elif (idx1.dtype.kind == "c" and (not lib.is_np_dtype(idx2.dtype, "iufc"))) or (
        idx2.dtype.kind == "c" and (not lib.is_np_dtype(idx1.dtype, "iufc"))
    ):
        # complex objects non-sortable
        warn = RuntimeWarning
    elif (
        isinstance(idx1.dtype, PeriodDtype) and isinstance(idx2.dtype, CategoricalDtype)
    ) or (
        isinstance(idx2.dtype, PeriodDtype) and isinstance(idx1.dtype, CategoricalDtype)
    ):
        warn = FutureWarning
        msg = r"PeriodDtype\[B\] is deprecated"
        mark = pytest.mark.xfail(
            reason="Warning not produced on all builds",
            raises=AssertionError,
            strict=False,
        )
        request.applymarker(mark)

    any_uint64 = np.uint64 in (idx1.dtype, idx2.dtype)
    idx1_signed = is_signed_integer_dtype(idx1.dtype)
    idx2_signed = is_signed_integer_dtype(idx2.dtype)

    # Union with a non-unique, non-monotonic index raises error
    # This applies to the boolean index
    idx1 = idx1.sort_values()
    idx2 = idx2.sort_values()

    with tm.assert_produces_warning(warn, match=msg):
        res1 = idx1.union(idx2)
        res2 = idx2.union(idx1)

    if any_uint64 and (idx1_signed or idx2_signed):
        assert res1.dtype == np.dtype("O")
        assert res2.dtype == np.dtype("O")
    else:
        assert res1.dtype == common_dtype
        assert res2.dtype == common_dtype


@pytest.mark.parametrize(
    "idx1,idx2",
    [
        (Index(np.arange(5), dtype=np.int64), RangeIndex(5)),
        (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.int64)),
        (Index(np.arange(5), dtype=np.float64), RangeIndex(5)),
        (Index(np.arange(5), dtype=np.float64), Index(np.arange(5), dtype=np.uint64)),
    ],
)
def test_compatible_inconsistent_pairs(idx1, idx2):
    # [FUNCTION DOC]
    # Parametrized over pairs mixing integer, float, unsigned and range indexes. The union in either order has a dtype equal to the dtype of one of the operands.
    # GH 23525
    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)

    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("int64", "int64", "int64"),
        ("int64", "uint64", "object"),
        ("int64", "float64", "float64"),
        ("uint64", "float64", "float64"),
        ("uint64", "uint64", "uint64"),
        ("float64", "float64", "float64"),
        ("datetime64[ns]", "int64", "object"),
        ("datetime64[ns]", "uint64", "object"),
        ("datetime64[ns]", "float64", "object"),
        ("datetime64[ns, CET]", "int64", "object"),
        ("datetime64[ns, CET]", "uint64", "object"),
        ("datetime64[ns, CET]", "float64", "object"),
        ("Period[D]", "int64", "object"),
        ("Period[D]", "uint64", "object"),
        ("Period[D]", "float64", "object"),
    ],
)
@pytest.mark.parametrize("names", [("foo", "foo", "foo"), ("foo", "bar", None)])
def test_union_dtypes(left, right, expected, names):
    # [FUNCTION DOC]
    # Parametrized over (left dtype, right dtype, expected union dtype) combinations of numeric, datetime (naive and zoned) and period dtypes, and over name tuples (all the same, or differing). Unions of two empty indexes with those dtypes give the expected dtype and the expected name (kept only when both names agree); the intersection keeps the name under the same rule.
    left = pandas_dtype(left)
    right = pandas_dtype(right)
    a = Index([], dtype=left, name=names[0])
    b = Index([], dtype=right, name=names[1])
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]

    # Testing name retention
    # TODO: pin down desired dtype; do we want it to be commutative?
    result = a.intersection(b)
    assert result.name == names[2]


@pytest.mark.parametrize("values", [[1, 2, 2, 3], [3, 3]])
def test_intersection_duplicates(values):
    # [FUNCTION DOC]
    # Parametrized over two integer lists. Intersecting an index with duplicates against an index of two threes gives a single three.
    # GH#31326
    a = Index(values)
    b = Index([3, 3])
    result = a.intersection(b)
    expected = Index([3])
    tm.assert_index_equal(result, expected)


class TestSetOps:
    # [CLASS DOC]
    # Set operation tests shared by all indexes in the shared index fixture.
    # Set operation tests shared by all indexes in the `index` fixture
    @pytest.mark.parametrize("case", [0.5, "xxx"])
    @pytest.mark.parametrize(
        "method", ["intersection", "union", "difference", "symmetric_difference"]
    )
    def test_set_ops_error_cases(self, case, method, index):
        # [FUNCTION DOC]
        # Parametrized over a float and a string argument and each of the four set methods. A non-iterable argument raises a type error that the input must be an index or array-like.
        # non-iterable input
        msg = "Input must be Index or array-like"
        with pytest.raises(TypeError, match=msg):
            getattr(index, method)(case)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_intersection_base(self, index):
        # [FUNCTION DOC]
        # Skips categorical. Intersection of the first five unique elements with the first three equals the first three. Then (unless tz-aware) for argument forms numpy array, series, list: the result has the same unique contents. For multi-level indexes a list of non-tuples raises a type error ("other must be a MultiIndex or a list of tuples").
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"Not relevant for {type(index).__name__}")

        first = index[:5].unique()
        second = index[:3].unique()
        intersect = first.intersection(second)
        tm.assert_index_equal(intersect, second)

        if isinstance(index.dtype, DatetimeTZDtype):
            # The second.values below will drop tz, so the rest of this test
            #  is not applicable.
            return

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.intersection(case)
            assert equal_contents(result, second)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.intersection([1, 2, 3])

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_union_base(self, index):
        # [FUNCTION DOC]
        # Takes the unique values. Union of the tail and head overlapping slices equals the sorted whole; for arrays, series and lists the contents equal the whole; multi-level indexes raise a type error for a list of integers.
        index = index.unique()
        first = index[3:]
        second = index[:5]
        everything = index

        union = first.union(second)
        tm.assert_index_equal(union.sort_values(), everything.sort_values())

        if isinstance(index.dtype, DatetimeTZDtype):
            # The second.values below will drop tz, so the rest of this test
            #  is not applicable.
            return

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.union(case)
            assert equal_contents(result, everything)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.union([1, 2, 3])

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_difference_base(self, sort, index):
        # [FUNCTION DOC]
        # Uses the sort fixture. Differences of overlapping slices: expected answer is the tail for most indexes, a set difference for boolean indexes, and empty for categorical. Works for index, numpy array, series and list arguments; multi-level raises a type error for non-tuple lists.
        first = index[2:]
        second = index[:4]
        if index.inferred_type == "boolean":
            # i think (TODO: be sure) there assumptions baked in about
            #  the index fixture that don't hold here?
            answer = set(first).difference(set(second))
        elif isinstance(index, CategoricalIndex):
            answer = []
        else:
            answer = index[4:]
        result = first.difference(second, sort)
        assert equal_contents(result, answer)

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.difference(case, sort)
            assert equal_contents(result, answer)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.difference([1, 2, 3], sort)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_symmetric_difference(self, index, using_infer_string, request):
        # [FUNCTION DOC]
        # Expected failure for object string indexes with string inference on. Skips categorical, short indexes, and indexes where the first or last element repeats. The symmetric difference of all-but-first with all-but-last equals the first and last elements (compared after sorting); array, series and list arguments give the same contents; multi-level raises a type error for non-tuple lists.
        if (
            using_infer_string
            and index.dtype == "object"
            and index.inferred_type == "string"
        ):
            request.applymarker(pytest.mark.xfail(reason="TODO: infer_string"))
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"Not relevant for {type(index).__name__}")
        if len(index) < 2:
            pytest.skip("Too few values for test")
        if index[0] in index[1:] or index[-1] in index[:-1]:
            # index fixture has e.g. an index of bools that does not satisfy this,
            #  another with [0, 0, 1, 1, 2, 2]
            pytest.skip("Index values no not satisfy test condition.")

        first = index[1:]
        second = index[:-1]
        answer = index[[0, -1]]
        result = first.symmetric_difference(second)
        tm.assert_index_equal(result.sort_values(), answer.sort_values())

        # GH#10149
        cases = [second.to_numpy(), second.to_series(), second.to_list()]
        for case in cases:
            result = first.symmetric_difference(case)
            assert equal_contents(result, answer)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.symmetric_difference([1, 2, 3])

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_union(self, index_flat, fname, sname, expected_name):
        # [FUNCTION DOC]
        # Parametrized over (first name, second name, expected name). Unions of an index (made unique) with a copy, with an empty index, of an empty with the copy, and of two empties give the index (or empty) with the expected name (kept only when names agree).
        # GH#9943, GH#9862
        # Test unions with various name combinations
        # Do not test MultiIndex or repeats
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # Test copy.union(copy)
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test copy.union(empty)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(copy)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(empty)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_union_unequal(self, index_flat, fname, sname, expected_name):
        # [FUNCTION DOC]
        # Same name combinations. Union of the index with its tail yields the whole index (compared after sorting) with the expected name.
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # test copy.union(subset) - need sort for unicode and string
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        union = first.union(second).sort_values()
        expected = index.set_names(expected_name).sort_values()
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_intersect(self, index_flat, fname, sname, expected_name):
        # [FUNCTION DOC]
        # Same name combinations. Intersections of the index with a copy, with empty, of empty with the index, and of two empties give the index or empty index with the expected name.
        # GH#35847
        # Test intersections with various name combinations
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # Test copy.intersection(copy)
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test copy.intersection(empty)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(copy)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(empty)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_intersect_unequal(self, index_flat, fname, sname, expected_name):
        # [FUNCTION DOC]
        # Same name combinations. Intersection of the index with its tail yields the tail (compared after sorting) with the expected name.
        if not index_flat.is_unique:
            index = index_flat.unique()
        else:
            index = index_flat

        # test copy.intersection(subset) - need sort for unicode and string
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        intersect = first.intersection(second).sort_values()
        expected = index[1:].set_names(expected_name).sort_values()
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_intersection_name_retention_with_nameless(self, index):
        # [FUNCTION DOC]
        # Names the index (levels numbered for multi-level, "foo" otherwise) and intersects with its nameless array form, with an empty array, and an empty self with the array; the name is retained each time.
        if isinstance(index, MultiIndex):
            index = index.rename(list(range(index.nlevels)))
        else:
            index = index.rename("foo")

        other = np.asarray(index)

        result = index.intersection(other)
        assert result.name == index.name

        # empty other, same dtype
        result = index.intersection(other[:0])
        assert result.name == index.name

        # empty `self`
        result = index[:0].intersection(other)
        assert result.name == index.name

    def test_difference_preserves_type_empty(self, index, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Skips non-unique indexes. The difference of an index with itself is an empty index of the same type (exact).
        # GH#20040
        # If taking difference of a set and itself, it
        # needs to preserve the type of the index
        if not index.is_unique:
            pytest.skip("Not relevant since index is not unique")
        result = index.difference(index, sort=sort)
        expected = index[:0]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_name_retention_equals(self, index, names):
        # [FUNCTION DOC]
        # Uses the names fixture (multi-level indexes get per-level names). For two equal indexes with different names, the difference is an empty slice with the resolved name (names[2]).
        if isinstance(index, MultiIndex):
            names = [[x] * index.nlevels for x in names]
        index = index.rename(names[0])
        other = index.rename(names[1])

        assert index.equals(other)

        result = index.difference(other)
        expected = index[:0].rename(names[2])
        tm.assert_index_equal(result, expected)

    def test_intersection_difference_match_empty(self, index, sort):
        # [FUNCTION DOC]
        # Uses the sort fixture. Skips non-unique indexes. Intersection with an empty slice equals the difference of the index with itself (exact).
        # GH#20040
        # Test that the intersection of an index with an
        # empty index produces the same index as the difference
        # of an index with itself.  Test for all types
        if not index.is_unique:
            pytest.skip("Not relevant because index is not unique")
        inter = index.intersection(index[:0])
        diff = index.difference(index, sort=sort)
        tm.assert_index_equal(inter, diff, exact=True)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(index_flat, sort, method, using_infer_string):
    # [FUNCTION DOC]
    # Ignores cast and period warnings. Parametrized over the four methods; uses the sort fixture and infer-string flag. Applying the method between a flat index and its categorical cast (all or first five elements) should equal applying it between the index and itself (or its slice), with range indexes compared loosely. With string inference on, an empty index makes the union and symmetric difference expectations categorical.
    # MultiIndex tested separately in tests.indexes.multi.test_setops
    index = index_flat

    other = index.astype("category")
    exact = "equiv" if isinstance(index, RangeIndex) else True

    result = getattr(index, method)(other, sort=sort)
    expected = getattr(index, method)(index, sort=sort)
    if (
        using_infer_string
        and index.empty
        and method in ("union", "symmetric_difference")
    ):
        expected = expected.astype("category")
    tm.assert_index_equal(result, expected, exact=exact)

    result = getattr(index, method)(other[:5], sort=sort)
    expected = getattr(index, method)(index[:5], sort=sort)
    if (
        using_infer_string
        and index.empty
        and method in ("union", "symmetric_difference")
    ):
        expected = expected.astype("category")
    tm.assert_index_equal(result, expected, exact=exact)


def test_intersection_duplicates_all_indexes(index):
    # [FUNCTION DOC]
    # Uses the shared index fixture; skips empty. An index intersected with a version having a repeated first element gives a result that is the same in either operand order and is unique.
    # GH#38743
    if index.empty:
        # No duplicates in empty indexes
        pytest.skip("Not relevant for empty Index")

    idx = index
    idx_non_unique = idx[[0, 0, 1, 2]]

    assert idx.intersection(idx_non_unique).equals(idx_non_unique.intersection(idx))
    assert idx.intersection(idx_non_unique).is_unique


def test_union_duplicate_index_subsets_of_each_other(
    any_dtype_for_small_pos_integer_indexes,
):
    # [FUNCTION DOC]
    # Uses the small-integer dtype fixture. Union of indexes with duplicates where one's values are a subset of the other's gives the combined multiset (duplicates retained up to the larger count) with the same dtype (a plain index for categorical), for default and unsorted union.
    # GH#31326
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 2, 2, 3], dtype=dtype)
    b = Index([3, 3, 4], dtype=dtype)

    expected = Index([1, 2, 2, 3, 3, 4], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([1, 2, 2, 3, 3, 4])
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)


def test_union_with_duplicate_index_and_non_monotonic(
    any_dtype_for_small_pos_integer_indexes,
):
    # [FUNCTION DOC]
    # Union of a non-monotonic index with duplicates and a smaller index gives a sorted result keeping duplicates, in either order, for every small-integer dtype.
    # GH#36289
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 0, 0], dtype=dtype)
    b = Index([0, 1], dtype=dtype)
    expected = Index([0, 0, 1], dtype=dtype)

    result = a.union(b)
    tm.assert_index_equal(result, expected)

    result = b.union(a)
    tm.assert_index_equal(result, expected)


def test_union_duplicate_index_different_dtypes():
    # [FUNCTION DOC]
    # Unsorted union of an integer index with duplicates and a string index with duplicates is the concatenation as an object index.
    # GH#36289
    a = Index([1, 2, 2, 3])
    b = Index(["1", "0", "0"])
    expected = Index([1, 2, 2, 3, "1", "0", "0"])
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)


def test_union_same_value_duplicated_in_both():
    # [FUNCTION DOC]
    # When a value appears twice in both operands, the union keeps two copies.
    # GH#36289
    a = Index([0, 0, 1])
    b = Index([0, 0, 1, 2])
    result = a.union(b)
    expected = Index([0, 0, 1, 2])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dup", [1, np.nan])
def test_union_nan_in_both(dup):
    # [FUNCTION DOC]
    # Parametrized over the duplicate (one or NaN). Unsorted union of two float indexes both containing NaN keeps the expected multiplicity and order as floats.
    # GH#36289
    a = Index([np.nan, 1, 2, 2])
    b = Index([np.nan, dup, 1, 2])
    result = a.union(b, sort=False)
    expected = Index([np.nan, dup, 1.0, 2.0, 2.0])
    tm.assert_index_equal(result, expected)


def test_union_rangeindex_sort_true():
    # [FUNCTION DOC]
    # Union of two stepped range indexes with sort enabled gives the sorted merged general integer index of all values from both ranges.
    # GH 53490
    idx1 = RangeIndex(1, 100, 6)
    idx2 = RangeIndex(1, 50, 3)
    result = idx1.union(idx2, sort=True)
    expected = Index(
        [
            1,
            4,
            7,
            10,
            13,
            16,
            19,
            22,
            25,
            28,
            31,
            34,
            37,
            40,
            43,
            46,
            49,
            55,
            61,
            67,
            73,
            79,
            85,
            91,
            97,
        ]
    )
    tm.assert_index_equal(result, expected)


def test_union_with_duplicate_index_not_subset_and_non_monotonic(
    any_dtype_for_small_pos_integer_indexes,
):
    # [FUNCTION DOC]
    # Like the subset case but the operands overlap only partially; results are the sorted combination with duplicates kept (a plain index for categorical), in either order.
    # GH#36289
    dtype = any_dtype_for_small_pos_integer_indexes
    a = Index([1, 0, 2], dtype=dtype)
    b = Index([0, 0, 1], dtype=dtype)
    expected = Index([0, 0, 1, 2], dtype=dtype)
    if isinstance(a, CategoricalIndex):
        expected = Index([0, 0, 1, 2])

    result = a.union(b)
    tm.assert_index_equal(result, expected)

    result = b.union(a)
    tm.assert_index_equal(result, expected)


def test_union_int_categorical_with_nan():
    # [FUNCTION DOC]
    # A categorical index of integers with NaN (integer categories) unioned with a plain integer index (either order) gives a float64 index with NaN.
    ci = CategoricalIndex([1, 2, np.nan])
    assert ci.categories.dtype.kind == "i"

    idx = Index([1, 2])

    result = idx.union(ci)
    expected = Index([1, 2, np.nan], dtype=np.float64)
    tm.assert_index_equal(result, expected)

    result = ci.union(idx)
    tm.assert_index_equal(result, expected)


class TestSetOpsUnsorted:
    # [CLASS DOC]
    # Set-operation tests about sort and name behavior, mostly on the string index. Comment in source notes these may belong in dtype-specific files.
    # These may eventually belong in a dtype-specific test_setops, or
    #  parametrized over a more general fixture
    def test_intersect_str_dates(self):
        # [FUNCTION DOC]
        # Intersecting an object index of strings with an object index of datetimes gives an empty object index.
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        index1 = Index(dt_dates, dtype=object)
        index2 = Index(["aa"], dtype=object)
        result = index2.intersection(index1)

        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_intersection(self, index, sort):
        # [FUNCTION DOC]
        # Run for the string index with the sort fixture. Intersection of a longer and a shorter prefix equals the shorter (compared after sorting unless sorting was requested), and intersecting an index with itself returns a different object.
        first = index[:20]
        second = index[:10]
        intersect = first.intersection(second, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(intersect.sort_values(), second.sort_values())
        else:
            tm.assert_index_equal(intersect, second)

        # Corner cases
        inter = first.intersection(first, sort=sort)
        assert inter is not first

    @pytest.mark.parametrize(
        "index2_name,keeps_name",
        [
            ("index", True),  # preserve same name
            ("other", False),  # drop diff names
            (None, False),
        ],
    )
    def test_intersection_name_preservation(self, index2_name, keeps_name, sort):
        # [FUNCTION DOC]
        # Parametrized over the other index's name and whether the name is kept. The intersection of two overlapping integer indexes has the expected values and keeps the name only when both names are equal.
        index2 = Index([3, 4, 5, 6, 7], name=index2_name)
        index1 = Index([1, 2, 3, 4, 5], name="index")
        expected = Index([3, 4, 5])
        result = index1.intersection(index2, sort)

        if keeps_name:
            expected.name = "index"

        assert result.name == expected.name
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize(
        "first_name,second_name,expected_name",
        [("A", "A", "A"), ("A", "B", None), (None, "B", None)],
    )
    def test_intersection_name_preservation2(
        self, index, first_name, second_name, expected_name, sort
    ):
        # [FUNCTION DOC]
        # String index. Parametrized over name pairs; renames overlapping slices and checks the intersection has the expected name.
        first = index[5:20]
        second = index[:10]
        first.name = first_name
        second.name = second_name
        intersect = first.intersection(second, sort=sort)
        assert intersect.name == expected_name

    def test_chained_union(self, sort):
        # [FUNCTION DOC]
        # Union of an index with the union of two others equals union taken left to right, preserving names, for non-empty and for empty-operand cases.
        # Chained unions handles names correctly
        i1 = Index([1, 2], name="i1")
        i2 = Index([5, 6], name="i2")
        i3 = Index([3, 4], name="i3")
        union = i1.union(i2.union(i3, sort=sort), sort=sort)
        expected = i1.union(i2, sort=sort).union(i3, sort=sort)
        tm.assert_index_equal(union, expected)

        j1 = Index([1, 2], name="j1")
        j2 = Index([], name="j2")
        j3 = Index([], name="j3")
        union = j1.union(j2.union(j3, sort=sort), sort=sort)
        expected = j1.union(j2, sort=sort).union(j3, sort=sort)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union(self, index, sort):
        # [FUNCTION DOC]
        # String index. Union of two overlapping slices equals the covering slice (sorted when not requested sorted).
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        union = first.union(second, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(union.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_from_iterables(self, index, klass, sort):
        # [FUNCTION DOC]
        # String index, parametrized over numpy array, series, list as the other operand. Union equals the covering slice.
        # GH#10149
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        case = klass(second.values)
        result = first.union(case, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(result.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(result, everything)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_identity(self, index, sort):
        # [FUNCTION DOC]
        # String index. A union with itself, with an empty same-dtype index, and an empty index with it all return a new object, not the operand (identity not preserved to avoid shared mutable state). 
        # KNOWN ISSUE: the source comment says both objects would be recast to object dtype because an empty list is not consistent, but the code builds the empty index with the same dtype as the operand, so the comment is stale.
        first = index[5:20]

        union = first.union(first, sort=sort)
        # GH#63169 - identity is not preserved to prevent shared mutable state
        assert union is not first

        # This should no longer be the same object, since [] is not consistent,
        # both objects will be recast to dtype('O')
        union = first.union(Index([], dtype=first.dtype), sort=sort)
        assert union is not first

        union = Index([], dtype=first.dtype).union(first, sort=sort)
        assert union is not first

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("second_name,expected", [(None, None), ("name", "name")])
    def test_difference_name_preservation(self, index, second_name, expected, sort):
        # [FUNCTION DOC]
        # String index; parametrized over the second name. The difference of two overlapping slices equals the remaining slice (compared after sorting unless sort is true), and the result name is the first name only if the second name is the same.
        first = index[5:20]
        second = index[:10]
        answer = index[10:20]

        first.name = "name"
        second.name = second_name
        result = first.difference(second, sort=sort)

        if sort is True:
            tm.assert_index_equal(result, answer)
        else:
            answer.name = second_name
            tm.assert_index_equal(result.sort_values(), answer.sort_values())

        if expected is None:
            assert result.name is None
        else:
            assert result.name == expected

    def test_difference_empty_arg(self, index, sort):
        # [FUNCTION DOC]
        # Difference with an empty list returns the unique values of the original with its name.
        first = index.copy()
        first = first[5:20]
        first.name = "name"
        result = first.difference([], sort)
        expected = index[5:20].unique()
        expected.name = "name"
        tm.assert_index_equal(result, expected)

    def test_difference_should_not_compare(self):
        # [FUNCTION DOC]
        # Difference between an integer index with duplicates and a boolean index does not treat one as equal to True; result is a single one.
        # GH 55113
        left = Index([1, 1])
        right = Index([True])
        result = left.difference(right)
        expected = Index([1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_difference_identity(self, index, sort):
        # [FUNCTION DOC]
        # String index. Difference of a named slice with itself is empty and keeps the name.
        first = index[5:20]
        first.name = "name"
        result = first.difference(first, sort)

        assert len(result) == 0
        assert result.name == first.name

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_difference_sort(self, index, sort):
        # [FUNCTION DOC]
        # String index. Difference of overlapping slices equals the remainder, sorted if sort is none.
        first = index[5:20]
        second = index[:10]

        result = first.difference(second, sort)
        expected = index[10:20]

        if sort is None:
            expected = expected.sort_values()

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable(self, opname):
        # [FUNCTION DOC]
        # Parametrized over difference and symmetric difference. On an index mixing integers and timestamps (not comparable), the default sort emits a runtime warning ("not supported between") and the result is unsorted in original order; with sort disabled the same result without sorting.
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        op = operator.methodcaller(opname, b)

        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            # sort=None, the default
            result = op(a)
        expected = Index([3, Timestamp("2000"), 2, Timestamp("1999")])
        if opname == "difference":
            expected = expected[:2]
        tm.assert_index_equal(result, expected)

        # sort=False
        op = operator.methodcaller(opname, b, sort=False)
        result = op(a)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable_true(self, opname):
        # [FUNCTION DOC]
        # Parametrized likewise. With sort enabled on incomparable values, raises a type error that less-than is unsupported between timestamp and integer.
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        op = operator.methodcaller(opname, b, sort=True)

        msg = "'<' not supported between instances of 'Timestamp' and 'int'"
        with pytest.raises(TypeError, match=msg):
            op(a)

    def test_symmetric_difference_mi(self, sort):
        # [FUNCTION DOC]
        # Symmetric difference of two small multi-level indexes yields the three non-shared tuples (sorted if sort is none).
        index1 = MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3]))
        index2 = MultiIndex.from_tuples([("foo", 1), ("bar", 3)])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = MultiIndex.from_tuples([("bar", 2), ("baz", 3), ("bar", 3)])
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index2,expected",
        [
            ([0, 1, np.nan], [2.0, 3.0, 0.0]),
            ([0, 1], [np.nan, 2.0, 3.0, 0.0]),
        ],
    )
    def test_symmetric_difference_missing(self, index2, expected, sort):
        # [FUNCTION DOC]
        # Parametrized over second index (with or without NaN) and expectation. NaN in both operands cancels out (set semantics), NaN in one remains; results are floats, sorted if sort is none.
        index2 = Index(index2)
        expected = Index(expected)
        # GH#13514 change: {nan} - {nan} == {}
        # (GH#6444, sorting of nans, is no longer an issue)
        index1 = Index([1, np.nan, 2, 3])

        result = index1.symmetric_difference(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference_non_index(self, sort):
        # [FUNCTION DOC]
        # Symmetric difference with a numpy array operand keeps the first operand's name by default and takes the result name argument when supplied; ordering is checked after sorting when sort is false.
        index1 = Index([1, 2, 3, 4], name="index1")
        index2 = np.array([2, 3, 4, 5])
        expected = Index([1, 5], name="index1")
        result = index1.symmetric_difference(index2, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)
        assert result.name == "index1"

        result = index1.symmetric_difference(index2, result_name="new_name", sort=sort)
        expected.name = "new_name"
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)
        assert result.name == "new_name"

    def test_union_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        # [FUNCTION DOC]
        # Uses the nullable and Arrow numeric dtype fixture. Union of two overlapping three-element indexes in that dtype gives the five-element index with the same dtype.
        # GH#51365
        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        idx2 = Index([3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.union(idx2)
        expected = Index([1, 2, 3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        tm.assert_index_equal(result, expected)

    def test_union_string_array(self, any_string_dtype):
        # [FUNCTION DOC]
        # Uses the any string dtype fixture. Union of two single-element indexes gives the two-element index in that dtype.
        idx1 = Index(["a"], dtype=any_string_dtype)
        idx2 = Index(["b"], dtype=any_string_dtype)
        result = idx1.union(idx2)
        expected = Index(["a", "b"], dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)

    @td.skip_if_no("pyarrow")
    def test_union_pyarrow_timestamp(self):
        # [FUNCTION DOC]
        # Skipped without pyarrow. Union of two Arrow timestamp indexes with different dates keeps the Arrow dtype.
        # GH#58421
        left = Index(["2020-01-01"], dtype="timestamp[s][pyarrow]")
        right = Index(["2020-01-02"], dtype="timestamp[s][pyarrow]")

        res = left.union(right)
        expected = Index(["2020-01-01", "2020-01-02"], dtype=left.dtype)
        tm.assert_index_equal(res, expected)


def test_intersection_mutation_safety():
    # [FUNCTION DOC]
    # Intersection of two equal named indexes returns a new object distinct from both; later renaming the first operand does not change the result's name.
    # GH#63169
    index1 = Index([0, 1], name="original")
    index2 = Index([0, 1], name="original")

    result = index1.intersection(index2)

    assert result is not index1
    assert result is not index2

    tm.assert_index_equal(result, index1)
    assert result.name == "original"

    index1.name = "changed"

    assert result.name == "original"
    assert index1.name == "changed"


def test_union_mutation_safety():
    # [FUNCTION DOC]
    # Same as above for union: result is a new object and is independent of later rename of the first operand.
    # GH#63169
    index1 = Index([0, 1], name="original")
    index2 = Index([0, 1], name="original")

    result = index1.union(index2)

    assert result is not index1
    assert result is not index2

    tm.assert_index_equal(result, index1)
    assert result.name == "original"

    index1.name = "changed"

    assert result.name == "original"
    assert index1.name == "changed"


def test_union_mutation_safety_other():
    # [FUNCTION DOC]
    # Same for union but changing the name of the second operand afterward does not alter the result.
    # GH#63169
    index1 = Index([0, 1], name="original")
    index2 = Index([0, 1], name="original")

    result = index1.union(index2)

    assert result is not index2

    tm.assert_index_equal(result, index2)
    assert result.name == "original"

    index2.name = "changed"

    assert result.name == "original"
    assert index2.name == "changed"


def test_multiindex_intersection_mutation_safety():
    # [FUNCTION DOC]
    # Intersection of two equal multi-level indexes returns a new object; changing the first's names afterwards does not change the result's names.
    # GH#63169
    mi1 = MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])
    mi2 = MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])

    result = mi1.intersection(mi2)
    assert result is not mi1

    mi1.names = ["changed1", "changed2"]
    assert result.names == ["x", "y"]


def test_multiindex_union_mutation_safety():
    # [FUNCTION DOC]
    # Same for the union of multi-level indexes.
    # GH#63169
    mi1 = MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])
    mi2 = MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])

    result = mi1.union(mi2)
    assert result is not mi1

    mi1.names = ["changed1", "changed2"]
    assert result.names == ["x", "y"]
