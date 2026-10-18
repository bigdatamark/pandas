# [MODULE DOC]
# Tests for indexing on the base Index: slice-bound lookup, non-unique indexers, location
# lookup with NaN/None and large tuple indexes, and boolean extension-array indexing.
import numpy as np
import pytest

from pandas._libs import index as libindex

import pandas as pd
from pandas import (
    Index,
    NaT,
)
import pandas._testing as tm


class TestGetSliceBounds:
    # [CLASS DOC]
    # Groups tests of the get_slice_bound method.
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_within(self, side, expected):
        # [FUNCTION DOC]
        # Parametrized over side (left gives 4, right gives 5). Verifies the slice bound for a label
        # present in a sorted string index.
        index = Index(list("abcdef"))
        result = index.get_slice_bound("e", side=side)
        assert result == expected

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize(
        "data, bound, expected", [(list("abcdef"), "x", 6), (list("bcdefg"), "a", 0)]
    )
    def test_get_slice_bounds_outside(self, side, expected, data, bound):
        # [FUNCTION DOC]
        # Parametrized over side and two data/label pairs. Verifies a label beyond the end gives the
        # length and a label before the start gives zero, for either side.
        index = Index(data)
        result = index.get_slice_bound(bound, side=side)
        assert result == expected

    def test_get_slice_bounds_invalid_side(self):
        # [FUNCTION DOC]
        # Verifies that an invalid side argument raises a value error mentioning an invalid side value.
        with pytest.raises(ValueError, match="Invalid value for side kwarg"):
            Index([]).get_slice_bound("a", side="middle")


class TestGetIndexerNonUnique:
    # [CLASS DOC]
    # Groups tests of get_indexer_non_unique.
    def test_get_indexer_non_unique_dtype_mismatch(self):
        # [FUNCTION DOC]
        # Regression test: looking up integer targets in a string index returns an indexer of minus one
        # and a missing array listing the first position, both of platform integer dtype.
        # GH#25459
        indexes, missing = Index(["A", "B"]).get_indexer_non_unique(Index([0]))
        tm.assert_numpy_array_equal(np.array([-1], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), missing)

    @pytest.mark.parametrize(
        "idx_values,idx_non_unique",
        [
            ([np.nan, 100, 200, 100], [np.nan, 100]),
            ([np.nan, 100.0, 200.0, 100.0], [np.nan, 100.0]),
        ],
    )
    def test_get_indexer_non_unique_int_index(self, idx_values, idx_non_unique):
        # [FUNCTION DOC]
        # Parametrized over index values (int-with-NaN and float-with-NaN) and non-unique targets.
        # Verifies NaN lookup returns the position of NaN with no missing, and that a target containing
        # a duplicated value returns all matching positions with no missing.
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index([np.nan]))
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)

        indexes, missing = Index(idx_values).get_indexer_non_unique(
            Index(idx_non_unique)
        )
        tm.assert_numpy_array_equal(np.array([0, 1, 3], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)


class TestGetLoc:
    # [CLASS DOC]
    # Groups tests of get_loc on the base Index.
    @pytest.mark.slow  # to_flat_index takes a while
    def test_get_loc_tuple_monotonic_above_size_cutoff(self, monkeypatch):
        # [FUNCTION DOC]
        # Slow test. With the internal size cutoff of the index engine patched small (monkeypatch),
        # builds a large flat tuple index from a product of strings, integers and dates and verifies
        # get_loc of the middle tuple returns its position, exercising the binary-search path for
        # monotonic data above the cutoff.
        # Go through the libindex path for which using
        # _bin_search vs ndarray.searchsorted makes a difference

        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 100)
            lev = list("ABCD")
            dti = pd.date_range("2016-01-01", periods=10)

            mi = pd.MultiIndex.from_product([lev, range(5), dti])
            oidx = mi.to_flat_index()

            loc = len(oidx) // 2
            tup = oidx[loc]

            res = oidx.get_loc(tup)
        assert res == loc

    def test_get_loc_nan_object_dtype_nonmonotonic_nonunique(self):
        # [FUNCTION DOC]
        # Verifies on a non-monotonic non-unique object index with NaN and None: looking up NaN returns
        # the scalar position of the NaN, looking up None returns a boolean mask of the None positions,
        # and looking up NaT raises a key error.
        # case that goes through _maybe_get_bool_indexer
        idx = Index(["foo", np.nan, None, "foo", 1.0, None], dtype=object)

        # we dont raise KeyError on nan
        res = idx.get_loc(np.nan)
        assert res == 1

        # we only match on None, not on np.nan
        res = idx.get_loc(None)
        expected = np.array([False, False, True, False, False, True])
        tm.assert_numpy_array_equal(res, expected)

        # we don't match at all on mismatched NA
        with pytest.raises(KeyError, match="NaT"):
            idx.get_loc(NaT)


def test_getitem_boolean_ea_indexer():
    # [FUNCTION DOC]
    # Regression test: indexing an Index with a nullable boolean Series (containing NA) yields the
    # Index of positions with True, here just zero.
    # GH#45806
    ser = pd.Series([True, False, pd.NA], dtype="boolean")
    result = ser.index[ser]
    expected = Index([0])
    tm.assert_index_equal(result, expected)
