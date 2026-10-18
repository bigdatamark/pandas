# [MODULE DOC]
# Tests for MultiIndex.isin: NaN handling, tuple membership, the level keyword,
# empty and generator inputs.
import numpy as np
import pytest

from pandas import MultiIndex
import pandas._testing as tm


def test_isin_nan():
    # [FUNCTION DOC]
    # Tuples containing NaN (numpy or python float NaN) are found by isin.
    idx = MultiIndex.from_arrays([["foo", "bar"], [1.0, np.nan]])
    tm.assert_numpy_array_equal(idx.isin([("bar", np.nan)]), np.array([False, True]))
    tm.assert_numpy_array_equal(
        idx.isin([("bar", float("nan"))]), np.array([False, True])
    )


def test_isin_missing(nulls_fixture):
    # [FUNCTION DOC]
    # Parametrized over the null-values fixture. Values with a null do not match
    # non-null tuples in isin of another MultiIndex (GH48905).
    # GH48905
    mi1 = MultiIndex.from_tuples([(1, nulls_fixture)])
    mi2 = MultiIndex.from_tuples([(1, 1), (1, 2)])
    result = mi2.isin(mi1)
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)


def test_isin():
    # [FUNCTION DOC]
    # isin with a list of tuples gives a boolean array of matches; an empty
    # MultiIndex gives an empty boolean result.
    values = [("foo", 2), ("bar", 3), ("quux", 4)]

    idx = MultiIndex.from_arrays([["qux", "baz", "foo", "bar"], np.arange(4)])
    result = idx.isin(values)
    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(result, expected)

    # empty, return dtype bool
    idx = MultiIndex.from_arrays([[], []])
    result = idx.isin(values)
    assert len(result) == 0
    assert result.dtype == np.bool_


def test_isin_level_kwarg():
    # [FUNCTION DOC]
    # isin restricted to a level (positive/negative number or name) matches level
    # values;
    # out-of-range integer levels raise an index error about too many levels;
    # unknown numbers/floats/names raise key errors about the level not being found;
    # names set later become usable.
    idx = MultiIndex.from_arrays([["qux", "baz", "foo", "bar"], np.arange(4)])

    vals_0 = ["foo", "bar", "quux"]
    vals_1 = [2, 3, 10]

    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=0))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=-2))

    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=1))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=-1))

    msg = "Too many levels: Index has only 2 levels, not 6"
    with pytest.raises(IndexError, match=msg):
        idx.isin(vals_0, level=5)
    msg = "Too many levels: Index has only 2 levels, -5 is not a valid level number"
    with pytest.raises(IndexError, match=msg):
        idx.isin(vals_0, level=-5)

    with pytest.raises(KeyError, match=r"'Level 1\.0 not found'"):
        idx.isin(vals_0, level=1.0)
    with pytest.raises(KeyError, match=r"'Level -1\.0 not found'"):
        idx.isin(vals_1, level=-1.0)
    with pytest.raises(KeyError, match="'Level A not found'"):
        idx.isin(vals_1, level="A")

    idx.names = ["A", "B"]
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level="A"))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level="B"))

    with pytest.raises(KeyError, match="'Level C not found'"):
        idx.isin(vals_1, level="C")


@pytest.mark.parametrize(
    "labels,expected,level",
    [
        ([("b", np.nan)], [False, False, True], None),
        ([np.nan, "a"], [True, True, False], 0),
        (["d", np.nan], [False, True, True], 1),
    ],
)
def test_isin_multi_index_with_missing_value(labels, expected, level):
    # [FUNCTION DOC]
    # Parametrized over labels, expected mask and level. isin with NaN labels matches
    # missing values, either as tuple or per-level (GH 19132).
    # GH 19132
    midx = MultiIndex.from_arrays([[np.nan, "a", "b"], ["c", "d", np.nan]])
    result = midx.isin(labels, level=level)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)


def test_isin_empty():
    # [FUNCTION DOC]
    # isin with an empty list returns all-false (GH#51599).
    # GH#51599
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]])
    result = midx.isin([])
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)


def test_isin_generator():
    # [FUNCTION DOC]
    # isin accepts a generator of tuples (GH#52568).
    # GH#52568
    midx = MultiIndex.from_tuples([(1, 2)])
    result = midx.isin(x for x in [(1, 2)])
    expected = np.array([True])
    tm.assert_numpy_array_equal(result, expected)
