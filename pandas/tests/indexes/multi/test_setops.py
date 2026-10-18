# [MODULE DOC]
# Tests of set operations (union, intersection, difference, symmetric difference)
# on MultiIndex: input validation, sort argument semantics, name handling, missing
# values, extension dtypes, duplicates, categorical levels and the private name
# matching helper. Uses the idx and sort fixtures from the shared conftests.
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.api.types import (
    is_float_dtype,
    is_unsigned_integer_dtype,
)


@pytest.mark.parametrize("case", [0.5, "xxx"])
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_set_ops_error_cases(idx, case, sort, method):
    # [FUNCTION DOC]
    # Parametrized over non-iterable inputs (a float and a string) and the four set
    # methods, combined with the sort fixture. Each must raise a type error stating the
    # input must be an Index or array-like.
    # non-iterable input
    msg = "Input must be Index or array-like"
    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)(case, sort=sort)


@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_intersection_base(idx, sort, klass):
    # [FUNCTION DOC]
    # Parametrized over how the second operand is passed (MultiIndex, array, Series,
    # list). Intersection with a reversed head of the fixture equals that head, sorted
    # when sort is none; a list of scalars raises a type error requiring a MultiIndex or
    # list of tuples.
    first = idx[2::-1]  # first 3 elements reversed
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    intersect = first.intersection(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(intersect, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.intersection([1, 2, 3], sort=sort)


@pytest.mark.arm_slow
@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_union_base(idx, sort, klass):
    # [FUNCTION DOC]
    # Same parametrization for union: union of a reversed index and its head equals the
    # reversed index (sorted if sort is none); list of scalars raises the same type
    # error.
    first = idx[::-1]
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    union = first.union(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(union, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.union([1, 2, 3], sort=sort)


def test_difference_base(idx, sort):
    # [FUNCTION DOC]
    # Difference of the fixture and its tail equals its head (sorted if sort is none),
    # also
    # for array, Series and list operands (GH 10149); list of scalars raises a type
    # error.
    second = idx[4:]
    answer = idx[:4]
    result = idx.difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    assert result.equals(answer)
    tm.assert_index_equal(result, answer)

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        idx.difference([1, 2, 3], sort=sort)


def test_symmetric_difference(idx, sort):
    # [FUNCTION DOC]
    # Symmetric difference of two overlapping slices gives the two end elements (sorted
    # if sort is none), also for other operand containers (GH 10149); bad operand raises
    # a type error.
    first = idx[1:]
    second = idx[:-1]
    answer = idx[[-1, 0]]
    result = first.symmetric_difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    tm.assert_index_equal(result, answer)

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.symmetric_difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.symmetric_difference([1, 2, 3], sort=sort)


def test_multiindex_symmetric_difference():
    # [FUNCTION DOC]
    # The result keeps names when both operands share them, and has none when names
    # differ (GH 13490).
    # GH 13490
    idx = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=["a", "b"])
    result = idx.symmetric_difference(idx)
    assert result.names == idx.names

    idx2 = idx.copy().rename(["A", "B"])
    result = idx.symmetric_difference(idx2)
    assert result.names == [None, None]


def test_empty(idx):
    # [FUNCTION DOC]
    # The empty property is false for the fixture and true for a zero-length slice (GH
    # 15270).
    # GH 15270
    assert not idx.empty
    assert idx[:0].empty


def test_difference(idx, sort):
    # [FUNCTION DOC]
    # Broad difference test: normal difference (names kept, sorted by sort option),
    # empty
    # results for self, superset and empty receiver, names dropped when operand names
    # differ, empty with a differently sorted equal index, non-MultiIndex operands,
    # empty
    # list operand and list of tuples operand, and invalid scalar list raising a type
    # error.
    # KNOWN ISSUE: the comment says an exception is raised for a non-MultiIndex operand, but the code asserts a normal empty result.
    # KNOWN ISSUE: in the final non-empty list case an expected index is built but never compared with the result; only the names are compared.
    first = idx
    result = first.difference(idx[-3:], sort=sort)
    vals = idx[:-3].values

    if sort is None:
        vals = sorted(vals)

    expected = MultiIndex.from_tuples(vals, sortorder=0, names=idx.names)

    assert isinstance(result, MultiIndex)
    assert result.equals(expected)
    assert result.names == idx.names
    tm.assert_index_equal(result, expected)

    # empty difference: reflexive
    result = idx.difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: superset
    result = idx[-3:].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: degenerate
    result = idx[:0].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # names not the same
    chunklet = idx[-3:]
    chunklet.names = ["foo", "baz"]
    result = first.difference(chunklet, sort=sort)
    assert result.names == (None, None)

    # empty, but non-equal
    result = idx.difference(idx.sortlevel(1)[0], sort=sort)
    assert len(result) == 0

    # raise Exception called with non-MultiIndex
    result = first.difference(first.values, sort=sort)
    assert result.equals(first[:0])

    # name from empty array
    result = first.difference([], sort=sort)
    assert first.equals(result)
    assert first.names == result.names

    # name from non-empty array
    result = first.difference([("foo", "one")], sort=sort)
    expected = MultiIndex.from_tuples(
        [("bar", "one"), ("baz", "two"), ("foo", "two"), ("qux", "one"), ("qux", "two")]
    )
    expected.names = first.names
    assert first.names == result.names

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.difference([1, 2, 3, 4, 5], sort=sort)


def test_difference_sort_special():
    # [FUNCTION DOC]
    # Default sort: difference with an empty list returns the index unchanged in its
    # original (unsorted) order (GH-24959).
    # GH-24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # sort=None, the default
    result = idx.difference([])
    tm.assert_index_equal(result, idx)


def test_difference_sort_special_true():
    # [FUNCTION DOC]
    # With sort enabled, difference with an empty list returns the index sorted.
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    result = idx.difference([], sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_difference_sort_incomparable():
    # [FUNCTION DOC]
    # Levels with incomparable objects: default sort emits a runtime warning that sort
    # order is undefined, and the result equals the original; sort disabled returns it
    # without warning (GH-24959).
    # GH-24959
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])

    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])
    # sort=None, the default
    msg = "sort order is undefined for incomparable objects"
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        result = idx.difference(other)
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.difference(other, sort=False)
    tm.assert_index_equal(result, idx)


def test_difference_sort_incomparable_true():
    # [FUNCTION DOC]
    # Forcing sort on incomparable levels raises a type error about unordered values.
    # KNOWN ISSUE: a TODO in the test notes the error arises from constructing a categorical during safe sort and may deserve a clearer message; the test pins the current message.
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])
    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])

    # TODO: this is raising in constructing a Categorical when calling
    #  algos.safe_sort. Should we catch and re-raise with a better message?
    msg = "'values' is not ordered, please explicitly specify the categories order "
    with pytest.raises(TypeError, match=msg):
        idx.difference(other, sort=True)


def test_union(idx, sort):
    # [FUNCTION DOC]
    # Union of overlapping pieces matches the full index (sorted unless sort is
    # enabled); unions with self, with empty, and with an array of tuples behave as
    # expected.
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    the_union = piece1.union(piece2, sort=sort)

    if sort in (None, False):
        tm.assert_index_equal(the_union.sort_values(), idx.sort_values())
    else:
        tm.assert_index_equal(the_union, idx)

    # corner case, pass self or empty thing:
    the_union = idx.union(idx, sort=sort)
    tm.assert_index_equal(the_union, idx)

    the_union = idx.union(idx[:0], sort=sort)
    tm.assert_index_equal(the_union, idx)

    tuples = idx.values
    result = idx[:4].union(tuples[4:], sort=sort)
    if sort is None:
        tm.assert_index_equal(result.sort_values(), idx.sort_values())
    else:
        assert result.equals(idx)


def test_union_with_regular_index(idx, using_infer_string):
    # [FUNCTION DOC]
    # A flat index unioned with a MultiIndex gives a flat result containing the tuples
    # and scalars; the reverse order either raises a not-implemented error (infer-string
    # mode) or warns about unorderable values and gives a non-equal result.
    other = Index(["A", "B", "C"])

    result = other.union(idx)
    assert ("foo", "one") in result
    assert "B" in result

    if using_infer_string:
        with pytest.raises(NotImplementedError, match="Can only union"):
            idx.union(other)
    else:
        msg = "The values in the array are unorderable"
        with tm.assert_produces_warning(RuntimeWarning, match=msg):
            result2 = idx.union(other)
        # This is more consistent now, if sorting fails then we don't sort at all
        # in the MultiIndex case.
        assert not result.equals(result2)


def test_intersection(idx, sort):
    # [FUNCTION DOC]
    # Intersection of overlapping pieces; sort none or true gives sorted overlap,
    # otherwise order-insensitive; self, disjoint (empty), and array-of-tuples cases.
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    the_int = piece1.intersection(piece2, sort=sort)

    if sort in (None, True):
        tm.assert_index_equal(the_int, idx[3:5])
    else:
        tm.assert_index_equal(the_int.sort_values(), idx[3:5])

    # corner case, pass self
    the_int = idx.intersection(idx, sort=sort)
    tm.assert_index_equal(the_int, idx)

    # empty intersection: disjoint
    empty = idx[:2].intersection(idx[2:], sort=sort)
    expected = idx[:0]
    assert empty.equals(expected)

    tuples = idx.values
    result = idx.intersection(tuples)
    assert result.equals(idx)


@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(idx, sort, method):
    # [FUNCTION DOC]
    # Parametrized over four set methods. A flat categorical of tuples as operand
    # behaves like the MultiIndex itself, with result names reset to none, also for a
    # shorter operand.
    other = idx.to_flat_index().astype("category")
    res_names = [None] * idx.nlevels

    result = getattr(idx, method)(other, sort=sort)
    expected = getattr(idx, method)(idx, sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)

    result = getattr(idx, method)(other[:5], sort=sort)
    expected = getattr(idx, method)(idx[:5], sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)


def test_intersection_non_object(idx, sort):
    # [FUNCTION DOC]
    # Intersection with a numeric flat index yields an empty MultiIndex without names;
    # with an empty array it keeps names; with a non-empty array it raises a type error.
    other = Index(range(3), name="foo")

    result = idx.intersection(other, sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
    tm.assert_index_equal(result, expected, exact=True)

    # if we pass a length-0 ndarray (i.e. no name, we retain our idx.name)
    result = idx.intersection(np.asarray(other)[:0], sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
    tm.assert_index_equal(result, expected, exact=True)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        # With non-zero length non-index, we try and fail to convert to tuples
        idx.intersection(np.asarray(other), sort=sort)


def test_intersect_equal_sort():
    # [FUNCTION DOC]
    # Intersection of an equal index with sort disabled or none returns the original
    # order (GH-24959).
    # GH-24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
    tm.assert_index_equal(idx.intersection(idx, sort=None), idx)


def test_intersect_equal_sort_true():
    # [FUNCTION DOC]
    # With sort enabled the result is sorted.
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    result = idx.intersection(idx, sort=True)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("slice_", [slice(None), slice(0)])
def test_union_sort_other_empty(slice_):
    # [FUNCTION DOC]
    # Parametrized over a full slice and an empty slice. Default and disabled sort
    # return the original unsorted index in both directions (GH 24959).
    # https://github.com/pandas-dev/pandas/issues/24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])

    # default, sort=None
    other = idx[slice_]
    tm.assert_index_equal(idx.union(other), idx)
    tm.assert_index_equal(other.union(idx), idx)

    # sort=False
    tm.assert_index_equal(idx.union(other, sort=False), idx)


def test_union_sort_other_empty_sort():
    # [FUNCTION DOC]
    # Sort enabled with an empty operand sorts the result.
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    other = idx[:0]
    result = idx.union(other, sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_union_sort_other_incomparable():
    # [FUNCTION DOC]
    # Incomparable level values: default sort warns about unorderable values and returns
    # the unsorted union; sort disabled does not warn (GH 24959).
    # https://github.com/pandas-dev/pandas/issues/24959
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])

    # default, sort=None
    with tm.assert_produces_warning(RuntimeWarning, match="are unorderable"):
        result = idx.union(idx[:1])
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.union(idx[:1], sort=False)
    tm.assert_index_equal(result, idx)


def test_union_sort_other_incomparable_sort():
    # [FUNCTION DOC]
    # Sort enabled with incomparable values raises a type error about comparing
    # Timestamp and int.
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])
    msg = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=msg):
        idx.union(idx[:1], sort=True)


def test_union_non_object_dtype_raises():
    # [FUNCTION DOC]
    # Union with a non-tuple flat index (an integer level) raises a not-implemented
    # error (GH#32646).
    # GH#32646 raise NotImplementedError instead of less-informative error
    mi = MultiIndex.from_product([["a", "b"], [1, 2]])

    idx = mi.levels[1]

    msg = "Can only union MultiIndex with MultiIndex or Index of tuples"
    with pytest.raises(NotImplementedError, match=msg):
        mi.union(idx)


def test_union_empty_self_different_names():
    # [FUNCTION DOC]
    # Union of an empty one-level index with a two-level index returns the second index
    # with no names (GH#38423).
    # GH#38423
    mi = MultiIndex.from_arrays([[]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    result = mi.union(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]])
    tm.assert_index_equal(result, expected)


def test_union_multiindex_empty_rangeindex():
    # [FUNCTION DOC]
    # Union with an empty range index in either order returns the MultiIndex (names not
    # compared) (GH#41234).
    # GH#41234
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    ri = pd.RangeIndex(0)

    result_left = mi.union(ri)
    tm.assert_index_equal(mi, result_left, check_names=False)

    result_right = ri.union(mi)
    tm.assert_index_equal(mi, result_right, check_names=False)


@pytest.mark.parametrize(
    "method", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_setops_sort_validation(method):
    # [FUNCTION DOC]
    # Parametrized over the four methods. An invalid sort value raises a value error
    # about allowed values; sort enabled runs without error.
    # KNOWN ISSUE: the trailing comment contains an unresolved placeholder issue reference.
    idx1 = MultiIndex.from_product([["a", "b"], [1, 2]])
    idx2 = MultiIndex.from_product([["b", "c"], [1, 2]])

    with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
        getattr(idx1, method)(idx2, sort=2)

    # sort=True is supported as of GH#?
    getattr(idx1, method)(idx2, sort=True)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_difference_keep_ea_dtypes(any_numeric_ea_dtype, val):
    # [FUNCTION DOC]
    # Parametrized over an NA or large value and numeric extension dtypes. Difference of
    # MultiIndexes keeps extension dtype, including for an empty result (names retained)
    # (GH#48606).
    # GH#48606
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result = midx.difference(midx2)
    expected = MultiIndex.from_arrays([Series([1], dtype=any_numeric_ea_dtype), [2]])
    tm.assert_index_equal(result, expected)

    result = midx.difference(midx.sort_values(ascending=False))
    expected = MultiIndex.from_arrays(
        [Series([], dtype=any_numeric_ea_dtype), Series([], dtype=np.int64)],
        names=["a", None],
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [pd.NA, 5])
def test_symmetric_difference_keeping_ea_dtype(any_numeric_ea_dtype, val):
    # [FUNCTION DOC]
    # Symmetric difference preserves extension dtypes of the first level (GH#48607).
    # GH#48607
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result = midx.symmetric_difference(midx2)
    expected = MultiIndex.from_arrays(
        [Series([1, 1, val], dtype=any_numeric_ea_dtype), [1, 2, 3]]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("tuples", "exp_tuples"),
    [
        ([("val1", "test1")], [("val1", "test1")]),
        ([("val1", "test1"), ("val1", "test1")], [("val1", "test1")]),
        (
            [("val2", "test2"), ("val1", "test1")],
            [("val2", "test2"), ("val1", "test1")],
        ),
    ],
)
def test_intersect_with_duplicates(tuples, exp_tuples):
    # [FUNCTION DOC]
    # Parametrized over left tuple lists. Intersection removes duplicates and keeps the
    # left order (GH#36915).
    # GH#36915
    left = MultiIndex.from_tuples(tuples, names=["first", "second"])
    right = MultiIndex.from_tuples(
        [("val1", "test1"), ("val1", "test1"), ("val2", "test2")],
        names=["first", "second"],
    )
    result = left.intersection(right)
    expected = MultiIndex.from_tuples(exp_tuples, names=["first", "second"])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "data, names, expected",
    [
        ((1,), None, [None, None]),
        ((1,), ["a"], [None, None]),
        ((1,), ["b"], [None, None]),
        ((1, 2), ["c", "d"], [None, None]),
        ((1, 2), ["b", "a"], [None, None]),
        ((1, 2, 3), ["a", "b", "c"], [None, None]),
        ((1, 2), ["a", "c"], ["a", None]),
        ((1, 2), ["c", "b"], [None, "b"]),
        ((1, 2), ["a", "b"], ["a", "b"]),
        ((1, 2), [None, "b"], [None, "b"]),
    ],
)
def test_maybe_match_names(data, names, expected):
    # [FUNCTION DOC]
    # Parametrized over tuple data, names and expected result. The private helper
    # returns, per level, the shared name when both indexes agree at that position, else
    # none; if level counts differ all entries are none (GH#38323).
    # GH#38323
    mi = MultiIndex.from_tuples([], names=["a", "b"])
    mi2 = MultiIndex.from_tuples([data], names=names)
    result = mi._maybe_match_names(mi2)
    assert result == expected


def test_intersection_equal_different_names():
    # [FUNCTION DOC]
    # Intersection of equal values with partly different names keeps only names that
    # match (GH#30302).
    # GH#30302
    mi1 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])

    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=[None, "b"])
    tm.assert_index_equal(result, expected)


def test_intersection_different_names():
    # [FUNCTION DOC]
    # Intersecting a named with an unnamed equal index returns the unnamed one
    # (GH#38323).
    # GH#38323
    mi = MultiIndex.from_arrays([[1], [3]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1], [3]])
    result = mi.intersection(mi2)
    tm.assert_index_equal(result, mi2)


def test_intersection_with_missing_values_on_both_sides(nulls_fixture):
    # [FUNCTION DOC]
    # Using the nulls fixture, intersection matches rows where both sides have missing
    # values in the same position (GH#38623).
    # GH#38623
    mi1 = MultiIndex.from_arrays([[3, nulls_fixture, 4, nulls_fixture], [1, 2, 4, 2]])
    mi2 = MultiIndex.from_arrays([[3, nulls_fixture, 3], [1, 2, 4]])
    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[3, nulls_fixture], [1, 2]])
    tm.assert_index_equal(result, expected)


def test_union_with_missing_values_on_both_sides(nulls_fixture):
    # [FUNCTION DOC]
    # Union with missing values on both sides keeps one missing entry and sorts it last
    # (GH#38623).
    # GH#38623
    mi1 = MultiIndex.from_arrays([[1, nulls_fixture]])
    mi2 = MultiIndex.from_arrays([[1, nulls_fixture, 3]])
    result = mi1.union(mi2)
    expected = MultiIndex.from_arrays([[1, 3, nulls_fixture]])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("sort", [None, False])
def test_union_nan_got_duplicated(dtype, sort):
    # [FUNCTION DOC]
    # Parametrized over float and nullable float dtypes and sort none or false. NaN must
    # not be duplicated by union; with sort none the result is sorted, otherwise it
    # equals the larger index (GH#38977, GH#49010).
    # GH#38977, GH#49010
    mi1 = MultiIndex.from_arrays([pd.array([1.0, np.nan], dtype=dtype), [2, 3]])
    mi2 = MultiIndex.from_arrays([pd.array([1.0, np.nan, 3.0], dtype=dtype), [2, 3, 4]])
    result = mi1.union(mi2, sort=sort)
    if sort is None:
        expected = MultiIndex.from_arrays(
            [pd.array([1.0, 3.0, np.nan], dtype=dtype), [2, 4, 3]]
        )
    else:
        expected = mi2
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [4, 1])
def test_union_keep_ea_dtype(any_numeric_ea_dtype, val):
    # [FUNCTION DOC]
    # Parametrized over a first-value choice; union keeps extension dtypes and merges
    # overlaps (GH#48505).
    # GH#48505

    arr1 = Series([val, 2], dtype=any_numeric_ea_dtype)
    arr2 = Series([2, 1], dtype=any_numeric_ea_dtype)
    midx = MultiIndex.from_arrays([arr1, [1, 2]], names=["a", None])
    midx2 = MultiIndex.from_arrays([arr2, [2, 1]])
    result = midx.union(midx2)
    if val == 4:
        expected = MultiIndex.from_arrays(
            [Series([1, 2, 4], dtype=any_numeric_ea_dtype), [1, 2, 1]]
        )
    else:
        expected = MultiIndex.from_arrays(
            [Series([1, 2], dtype=any_numeric_ea_dtype), [1, 2]]
        )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dupe_val", [3, pd.NA])
def test_union_with_duplicates_keep_ea_dtype(dupe_val, any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Union with duplicate entries (value or NA) keeps extension dtypes and duplicate
    # multiplicities (GH48900).
    # GH48900
    mi1 = MultiIndex.from_arrays(
        [
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
        ]
    )
    mi2 = MultiIndex.from_arrays(
        [
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    result = mi1.union(mi2)
    expected = MultiIndex.from_arrays(
        [
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_union_duplicates(index, request):
    # [FUNCTION DOC]
    # Uses the generic index fixture (skipped for empty, interval, categorical). Union
    # of an index with a duplicate-containing index equals the sorted duplicated one in
    # both directions, with level dtype adjustments for unsigned-integer and float cases
    # (GH#38977, GH#47294).
    # GH#38977
    if index.empty or isinstance(index, (IntervalIndex, CategoricalIndex)):
        pytest.skip(f"No duplicates in an empty {type(index).__name__}")

    values = index.unique().values.tolist()
    mi1 = MultiIndex.from_arrays([values, [1] * len(values)])
    mi2 = MultiIndex.from_arrays([[values[0], *values], [1] * (len(values) + 1)])
    result = mi2.union(mi1)
    expected = mi2.sort_values()
    tm.assert_index_equal(result, expected)

    if (
        is_unsigned_integer_dtype(mi2.levels[0])
        and (mi2.get_level_values(0) < 2**63).all()
    ):
        # GH#47294 - union uses lib.fast_zip, converting data to Python integers
        # and loses type information. Result is then unsigned only when values are
        # sufficiently large to require unsigned dtype. This happens only if other
        # has dups or one of both have missing values
        expected = expected.set_levels(
            [expected.levels[0].astype(np.int64), expected.levels[1]]
        )
    elif is_float_dtype(mi2.levels[0]):
        # mi2 has duplicates witch is a different path than above, Fix that path
        # to use correct float dtype?
        expected = expected.set_levels(
            [expected.levels[0].astype(float), expected.levels[1]]
        )

    result = mi1.union(mi2)
    tm.assert_index_equal(result, expected)


def test_union_keep_dtype_precision(any_real_numeric_dtype):
    # [FUNCTION DOC]
    # Parametrized over real numeric dtypes; union preserves the narrow dtype and sorts
    # (GH#48498).
    # GH#48498
    arr1 = Series([4, 1, 1], dtype=any_real_numeric_dtype)
    arr2 = Series([1, 4], dtype=any_real_numeric_dtype)
    midx = MultiIndex.from_arrays([arr1, [2, 1, 1]], names=["a", None])
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]], names=["a", None])

    result = midx.union(midx2)
    expected = MultiIndex.from_arrays(
        ([Series([1, 1, 4], dtype=any_real_numeric_dtype), [1, 1, 2]]),
        names=["a", None],
    )
    tm.assert_index_equal(result, expected)


def test_union_keep_ea_dtype_with_na(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Union of indexes both containing NA keeps extension dtype and duplicate NA entries
    # (GH#48498).
    # GH#48498
    arr1 = Series([4, pd.NA], dtype=any_numeric_ea_dtype)
    arr2 = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    midx = MultiIndex.from_arrays([arr1, [2, 1]], names=["a", None])
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]])
    result = midx.union(midx2)
    expected = MultiIndex.from_arrays(
        [Series([1, 4, pd.NA, pd.NA], dtype=any_numeric_ea_dtype), [1, 2, 1, 2]]
    )
    tm.assert_index_equal(result, expected)


def test_union_duplicates_different_names():
    # [FUNCTION DOC]
    # Union of an index with duplicates and differently named index gives all entries,
    # sorted, with no names (GH#62059).
    # GH#62059
    mi1 = MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["x", "y"])
    mi2 = MultiIndex.from_tuples([(2, "b"), (3, "c"), (2, "b")])

    result = mi1.union(mi2)
    expected = MultiIndex.from_tuples(
        [(1, "a"), (2, "b"), (2, "b"), (3, "c")], names=[None, None]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "levels1, levels2, codes1, codes2, names",
    [
        (
            [["a", "b", "c"], [0, ""]],
            [["c", "d", "b"], [""]],
            [[0, 1, 2], [1, 1, 1]],
            [[0, 1, 2], [0, 0, 0]],
            ["name1", "name2"],
        ),
    ],
)
def test_intersection_lexsort_depth(levels1, levels2, codes1, codes2, names):
    # [FUNCTION DOC]
    # Parametrized with one case of levels, codes and names. The intersection has full
    # lexsort depth of 2 (GH#25169).
    # GH#25169
    mi1 = MultiIndex(levels=levels1, codes=codes1, names=names)
    mi2 = MultiIndex(levels=levels2, codes=codes2, names=names)
    mi_int = mi1.intersection(mi2)
    assert mi_int._lexsort_depth == 2


@pytest.mark.parametrize(
    "a",
    [pd.Categorical(["a", "b"], categories=["a", "b"]), ["a", "b"]],
)
@pytest.mark.parametrize("b_ordered", [True, False])
def test_intersection_with_non_lex_sorted_categories(a, b_ordered):
    # [FUNCTION DOC]
    # Parametrized over categorical or plain first column and ordered flag. Intersection
    # of MultiIndexes from frames, sorted or unsorted, gives the same expected result in
    # all four combinations (GH#49974).
    # GH#49974
    other = ["1", "2"]
    b = pd.Categorical(["a", "b"], categories=["b", "a"], ordered=b_ordered)
    df1 = DataFrame({"x": a, "y": other})
    df2 = DataFrame({"x": b, "y": other})

    expected = MultiIndex.from_arrays([a, other], names=["x", "y"])

    res1 = MultiIndex.from_frame(df1).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )
    res2 = MultiIndex.from_frame(df1).intersection(MultiIndex.from_frame(df2))
    res3 = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2)
    )
    res4 = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )

    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_intersection_keep_ea_dtypes(val, any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Intersection preserves extension dtypes, with an NA or integer extra value in the
    # other operand (GH#48604).
    # GH#48604
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result = midx.intersection(midx2)
    expected = MultiIndex.from_arrays([Series([2], dtype=any_numeric_ea_dtype), [1]])
    tm.assert_index_equal(result, expected)


def test_union_with_na_when_constructing_dataframe():
    # [FUNCTION DOC]
    # Constructing a frame from two series with MultiIndexes containing missing keys
    # unions their indexes into columns correctly (GH43222).
    # GH43222
    series1 = Series(
        (1,),
        index=MultiIndex.from_arrays(
            [Series([None], dtype="str"), Series([None], dtype="str")]
        ),
    )
    series2 = Series((10, 20), index=MultiIndex.from_tuples(((None, None), ("a", "b"))))
    result = DataFrame([series1, series2])
    expected = DataFrame({(np.nan, np.nan): [1.0, 10.0], ("a", "b"): [np.nan, 20.0]})
    tm.assert_frame_equal(result, expected)
