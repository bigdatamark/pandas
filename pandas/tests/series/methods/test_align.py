# [MODULE DOC]
# Tests for Series.align.
# Covers all join types with and without a fill value, copy semantics of the
# aligned outputs, identity of the index when both inputs share it, MultiIndex
# alignment on a shared level, timezone-mismatched datetime indexes, period and
# string-dtype indexes, and alignment where one MultiIndex has fewer or
# differently-named levels. Uses the datetime_series, join_type and
# any_string_dtype fixtures. No import-time side effects.

from datetime import timezone

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "first_slice,second_slice",
    [
        [[2, None], [None, -5]],
        [[None, 0], [None, -5]],
        [[None, -5], [None, 0]],
        [[None, 0], [None, 0]],
    ],
)
@pytest.mark.parametrize("fill", [None, -1])
def test_align(datetime_series, first_slice, second_slice, join_type, fill):
    # [FUNCTION DOC]
    # Parametrized over two slice pairs (first_slice, second_slice), a fill value
    # (none or -1) and every join type. Slices two views of the datetime_series
    # fixture, aligns them with the given join and fill value, and expects each
    # output to equal the corresponding input reindexed onto the joined index
    # (with missing entries filled by fill when given). Also checks that any labels
    # introduced by the join hold the fill value and that every result keeps the
    # name "ts".
    a = datetime_series[slice(*first_slice)]
    b = datetime_series[slice(*second_slice)]

    aa, ab = a.align(b, join=join_type, fill_value=fill)

    join_index = a.index.join(b.index, how=join_type)
    if fill is not None:
        diff_a = aa.index.difference(join_index)
        diff_b = ab.index.difference(join_index)
        if len(diff_a) > 0:
            assert (aa.reindex(diff_a) == fill).all()
        if len(diff_b) > 0:
            assert (ab.reindex(diff_b) == fill).all()

    ea = a.reindex(join_index)
    eb = b.reindex(join_index)

    if fill is not None:
        ea = ea.fillna(fill)
        eb = eb.fillna(fill)

    tm.assert_series_equal(aa, ea)
    tm.assert_series_equal(ab, eb)
    assert aa.name == "ts"
    assert ea.name == "ts"
    assert ab.name == "ts"
    assert eb.name == "ts"


def test_align_nocopy(datetime_series):
    # [FUNCTION DOC]
    # Intended to verify when align returns copies versus views of the inputs, for
    # left and right joins: after assigning into the aligned result, the original
    # series must not show the assigned value.
    # KNOWN ISSUE: the "do copy" and "do not copy" sections of the test body are
    # identical in behavior (every section asserts the original is unchanged), so
    # the comments distinguishing them do not match what is actually checked.
    # Only the left-aligned result of the first input and the right-aligned result of
    # the second input are mutated and the originals are expected to stay untouched.
    b = datetime_series[:5].copy()

    # do copy
    a = datetime_series.copy()
    ra, _ = a.align(b, join="left")
    ra[:5] = 5
    assert not (a[:5] == 5).any()

    # do not copy
    a = datetime_series.copy()
    ra, _ = a.align(b, join="left")
    ra[:5] = 5
    assert not (a[:5] == 5).any()

    # do copy
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join="right")
    rb[:3] = 5
    assert not (b[:3] == 5).any()

    # do not copy
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join="right")
    rb[:2] = 5
    assert not (b[:2] == 5).any()


def test_align_same_index(datetime_series):
    # [FUNCTION DOC]
    # Aligning a series with itself must return two series whose index objects are
    # new objects (not the same Python object as the input index) but are
    # considered identical to it by the index identity check.
    a, b = datetime_series.align(datetime_series)
    assert a.index is not datetime_series.index
    assert b.index is not datetime_series.index
    assert a.index.is_(datetime_series.index)
    assert b.index.is_(datetime_series.index)


def test_align_multiindex():
    # [FUNCTION DOC]
    # Regression test (GH 10665): aligning a series with a three-level MultiIndex to
    # a series whose plain index is named like the middle level. Checks left join
    # and right join, and that swapping the operands with the opposite join gives
    # the same (flipped) results: the MultiIndex side is preserved or restricted to
    # matching middle-level values, and the other side is broadcast across the
    # levels, with missing values where the middle label is absent.
    # GH 10665

    midx = pd.MultiIndex.from_product(
        [range(2), range(3), range(2)], names=("a", "b", "c")
    )
    idx = pd.Index(range(2), name="b")
    s1 = Series(np.arange(12, dtype="int64"), index=midx)
    s2 = Series(np.arange(2, dtype="int64"), index=idx)

    # these must be the same results (but flipped)
    res1l, res1r = s1.align(s2, join="left")
    res2l, res2r = s2.align(s1, join="right")

    expl = s1
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)

    res1l, res1r = s1.align(s2, join="right")
    res2l, res2r = s2.align(s1, join="left")

    exp_idx = pd.MultiIndex.from_product(
        [range(2), range(2), range(2)], names=("a", "b", "c")
    )
    expl = Series([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1] * 2, index=exp_idx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)


def test_align_dt64tzindex_mismatched_tzs():
    # [FUNCTION DOC]
    # Aligns two series whose datetime indexes carry different timezones (the same
    # instants expressed in two zones) and expects both results to have an index
    # converted to the UTC timezone.
    idx1 = date_range("2001", periods=5, freq="h", tz="US/Eastern")
    ser = Series(np.random.default_rng(2).standard_normal(len(idx1)), index=idx1)
    ser_central = ser.tz_convert("US/Central")
    # different timezones convert to UTC

    new1, new2 = ser.align(ser_central)
    assert new1.index.tz is timezone.utc
    assert new2.index.tz is timezone.utc


def test_align_periodindex(join_type):
    # [FUNCTION DOC]
    # Smoke test parametrized over join_type: aligns a yearly period-indexed series
    # with every second element of itself.
    # KNOWN ISSUE: the test contains a TODO and makes no assertions; it only
    # checks that the call does not raise.
    rng = period_range("1/1/2000", "1/1/2010", freq="Y")
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # TODO: assert something?
    ts.align(ts[::2], join=join_type)


def test_align_stringindex(any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over string dtypes: aligns two series with partially overlapping
    # string indexes of that dtype using the default (outer) join. Expects the
    # union index sorted, with missing values (NaN) where a label was absent from
    # the input.
    left = Series(range(3), index=pd.Index(["a", "b", "d"], dtype=any_string_dtype))
    right = Series(range(3), index=pd.Index(["a", "b", "c"], dtype=any_string_dtype))
    result_left, result_right = left.align(right)

    expected_idx = pd.Index(["a", "b", "c", "d"], dtype=any_string_dtype)
    expected_left = Series([0, 1, np.nan, 2], index=expected_idx)
    expected_right = Series([0, 1, 2, np.nan], index=expected_idx)

    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)


def test_align_left_fewer_levels():
    # [FUNCTION DOC]
    # Regression test (GH#45224): the left series has a MultiIndex with fewer levels
    # than the right one (a subset of its level names). Align must add the missing
    # level to the left, placing it last, and reorder the right index levels to the
    # same order, so both results share the index with names a, c, b.
    # GH#45224
    left = Series([2], index=pd.MultiIndex.from_tuples([(1, 3)], names=["a", "c"]))
    right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"])
    )
    result_left, result_right = left.align(right)

    expected_right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=["a", "c", "b"])
    )
    expected_left = Series(
        [2], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=["a", "c", "b"])
    )
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)


def test_align_left_different_named_levels():
    # [FUNCTION DOC]
    # Regression test (GH#45224): left and right MultiIndexes have partially
    # different level names. The aligned results must use the left level order
    # followed by the extra levels from the right, giving both outputs the same
    # four-level index.
    # GH#45224
    left = Series(
        [2], index=pd.MultiIndex.from_tuples([(1, 4, 3)], names=["a", "d", "c"])
    )
    right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"])
    )
    result_left, result_right = left.align(right)

    expected_left = Series(
        [2], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=["a", "d", "c", "b"])
    )
    expected_right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=["a", "d", "c", "b"])
    )
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)
