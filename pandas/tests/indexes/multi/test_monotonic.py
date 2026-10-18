# [MODULE DOC]
# Tests for monotonicity properties (increasing, decreasing, strict variants) of
# MultiIndex, comparing against the equivalent flat index of tuples.
import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
)


def test_is_monotonic_increasing_lexsorted(lexsorted_two_level_string_multiindex):
    # [FUNCTION DOC]
    # Uses the lexsorted string fixture; verifies it is not monotonic increasing,
    # nor strictly, both as a MultiIndex and as a flat index of its values.
    # string ordering
    mi = lexsorted_two_level_string_multiindex
    assert mi.is_monotonic_increasing is False
    assert Index(mi.values).is_monotonic_increasing is False
    assert mi._is_strictly_monotonic_increasing is False
    assert Index(mi.values)._is_strictly_monotonic_increasing is False


def test_is_monotonic_increasing():
    # [FUNCTION DOC]
    # Several MultiIndexes (sorted product, descending in either level, with NaN, string
    # levels, mixed-type levels that trigger a comparison error, and empty). Asserts the
    # increasing and strictly-increasing flags agree with the flat tuple index.
    # KNOWN ISSUE: in the first and second blocks one assertion repeats the strict MultiIndex check where the neighboring lines check the flat index, so the strict flat-index check is omitted there.
    i = MultiIndex.from_product([np.arange(10), np.arange(10)], names=["one", "two"])
    assert i.is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True
    assert Index(i.values).is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True

    i = MultiIndex.from_product(
        [np.arange(10, 0, -1), np.arange(10)], names=["one", "two"]
    )
    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False
    assert Index(i.values).is_monotonic_increasing is False
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    i = MultiIndex.from_product(
        [np.arange(10), np.arange(10, 0, -1)], names=["one", "two"]
    )
    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False
    assert Index(i.values).is_monotonic_increasing is False
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    i = MultiIndex.from_product([[1.0, np.nan, 2.0], ["a", "b", "c"]])
    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False
    assert Index(i.values).is_monotonic_increasing is False
    assert Index(i.values)._is_strictly_monotonic_increasing is False

    i = MultiIndex(
        levels=[["bar", "baz", "foo", "qux"], ["mom", "next", "zenith"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    assert i.is_monotonic_increasing is True
    assert Index(i.values).is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True
    assert Index(i.values)._is_strictly_monotonic_increasing is True

    # mixed levels, hits the TypeError
    i = MultiIndex(
        levels=[
            [1, 2, 3, 4],
            [
                "gb00b03mlx29",
                "lu0197800237",
                "nl0000289783",
                "nl0000289965",
                "nl0000301109",
            ],
        ],
        codes=[[0, 1, 1, 2, 2, 2, 3], [4, 2, 0, 0, 1, 3, -1]],
        names=["household_id", "asset_id"],
    )

    assert i.is_monotonic_increasing is False
    assert i._is_strictly_monotonic_increasing is False

    # empty
    i = MultiIndex.from_arrays([[], []])
    assert i.is_monotonic_increasing is True
    assert Index(i.values).is_monotonic_increasing is True
    assert i._is_strictly_monotonic_increasing is True
    assert Index(i.values)._is_strictly_monotonic_increasing is True


def test_is_monotonic_decreasing():
    # [FUNCTION DOC]
    # Mirror of the increasing test for the decreasing and strictly-decreasing flags,
    # covering reversed products, NaN, string levels, mixed-type levels and empty.
    # KNOWN ISSUE: in the first block one assertion repeats the strict MultiIndex check instead of checking the flat index.
    i = MultiIndex.from_product(
        [np.arange(9, -1, -1), np.arange(9, -1, -1)], names=["one", "two"]
    )
    assert i.is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True
    assert Index(i.values).is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True

    i = MultiIndex.from_product(
        [np.arange(10), np.arange(10, 0, -1)], names=["one", "two"]
    )
    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    i = MultiIndex.from_product(
        [np.arange(10, 0, -1), np.arange(10)], names=["one", "two"]
    )
    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    i = MultiIndex.from_product([[2.0, np.nan, 1.0], ["c", "b", "a"]])
    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    # string ordering
    i = MultiIndex(
        levels=[["qux", "foo", "baz", "bar"], ["three", "two", "one"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    assert i.is_monotonic_decreasing is False
    assert Index(i.values).is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False
    assert Index(i.values)._is_strictly_monotonic_decreasing is False

    i = MultiIndex(
        levels=[["qux", "foo", "baz", "bar"], ["zenith", "next", "mom"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    assert i.is_monotonic_decreasing is True
    assert Index(i.values).is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True
    assert Index(i.values)._is_strictly_monotonic_decreasing is True

    # mixed levels, hits the TypeError
    i = MultiIndex(
        levels=[
            [4, 3, 2, 1],
            [
                "nl0000301109",
                "nl0000289965",
                "nl0000289783",
                "lu0197800237",
                "gb00b03mlx29",
            ],
        ],
        codes=[[0, 1, 1, 2, 2, 2, 3], [4, 2, 0, 0, 1, 3, -1]],
        names=["household_id", "asset_id"],
    )

    assert i.is_monotonic_decreasing is False
    assert i._is_strictly_monotonic_decreasing is False

    # empty
    i = MultiIndex.from_arrays([[], []])
    assert i.is_monotonic_decreasing is True
    assert Index(i.values).is_monotonic_decreasing is True
    assert i._is_strictly_monotonic_decreasing is True
    assert Index(i.values)._is_strictly_monotonic_decreasing is True


def test_is_strictly_monotonic_increasing():
    # [FUNCTION DOC]
    # An index with duplicate entries is monotonic increasing but not strictly.
    idx = MultiIndex(
        levels=[["bar", "baz"], ["mom", "next"]], codes=[[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    assert idx.is_monotonic_increasing is True
    assert idx._is_strictly_monotonic_increasing is False


def test_is_strictly_monotonic_decreasing():
    # [FUNCTION DOC]
    # An index with duplicate entries in decreasing order is monotonic decreasing but
    # not strictly.
    idx = MultiIndex(
        levels=[["baz", "bar"], ["next", "mom"]], codes=[[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    assert idx.is_monotonic_decreasing is True
    assert idx._is_strictly_monotonic_decreasing is False


@pytest.mark.parametrize("attr", ["is_monotonic_increasing", "is_monotonic_decreasing"])
@pytest.mark.parametrize(
    "values",
    [[(np.nan,), (1,), (2,)], [(1,), (np.nan,), (2,)], [(1,), (2,), (np.nan,)]],
)
def test_is_monotonic_with_nans(values, attr):
    # [FUNCTION DOC]
    # Parametrized over the attribute (increasing, decreasing) and tuple lists with a
    # NaN in each position. Both attributes must be false (GH: 37220).
    # GH: 37220
    idx = MultiIndex.from_tuples(values, names=["test"])
    assert getattr(idx, attr) is False
