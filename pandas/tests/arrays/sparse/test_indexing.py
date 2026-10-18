# [MODULE DOC]
# Tests for indexing behavior of SparseArray: getitem (scalar, slice, boolean
# mask, sparse boolean mask), setitem rejection, take (with and without fill),
# and the where operation. Shares two fixtures (arr_data, arr) defined here.
# No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.fixture
def arr_data():
    # [FUNCTION DOC]
    # HELPER: fixture supplying shared test data for this module.
    # Returns a float numpy array of length 10 with missing values at positions
    # 0, 1, 5 and 8 and the values 1, 2, 3, 4, 5, 6 elsewhere.
    return np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])


@pytest.fixture
def arr(arr_data):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a SparseArray (default NaN fill value) built from
    # the arr_data fixture.
    return SparseArray(arr_data)


class TestGetitem:
    # [CLASS DOC]
    # Groups tests of SparseArray element and slice access (getitem).
    def test_getitem(self, arr):
        # [FUNCTION DOC]
        # Verifies that iterating the array and indexing it with positive and negative
        # integers agree with the equivalent positions of the densified array
        # (approximate equality, NaN-aware).
        dense = arr.to_dense()
        for i, value in enumerate(arr):
            tm.assert_almost_equal(value, dense[i])
            tm.assert_almost_equal(arr[-i], dense[-i])

    def test_getitem_arraylike_mask(self, arr):
        # [FUNCTION DOC]
        # Verifies that indexing a SparseArray with a plain list boolean mask returns a
        # SparseArray holding only the selected elements.
        arr = SparseArray([0, 1, 2])
        result = arr[[True, False, True]]
        expected = SparseArray([0, 2])
        tm.assert_sp_array_equal(result, expected)

    @pytest.mark.parametrize(
        "slc",
        [
            np.s_[:],
            np.s_[1:10],
            np.s_[1:100],
            np.s_[10:1],
            np.s_[:-3],
            np.s_[-5:-4],
            np.s_[:-12],
            np.s_[-12:],
            np.s_[2:],
            np.s_[2::3],
            np.s_[::2],
            np.s_[::-1],
            np.s_[::-2],
            np.s_[1:6:2],
            np.s_[:-6:-2],
        ],
    )
    @pytest.mark.parametrize(
        "as_dense", [[np.nan] * 10, [1] * 10, [np.nan] * 5 + [1] * 5, []]
    )
    def test_getslice(self, slc, as_dense):
        # [FUNCTION DOC]
        # Parametrized over many slice objects (forward, backward, strided, out of
        # range, empty result) and over dense inputs that are all NaN, all 1, half and
        # half, and empty. Verifies that slicing a SparseArray equals a SparseArray
        # built from slicing the dense numpy array with the same slice.
        as_dense = np.array(as_dense)
        arr = SparseArray(as_dense)

        result = arr[slc]
        expected = SparseArray(as_dense[slc])

        tm.assert_sp_array_equal(result, expected)

    def test_getslice_tuple(self):
        # [FUNCTION DOC]
        # Verifies that indexing with a one-element tuple containing a slice behaves
        # like a plain slice, for NaN and zero fill values, and that a tuple with more
        # indexers than dimensions raises an index error ("too many indices"), matching
        # numpy behavior on the dense array.
        dense = np.array([np.nan, 0, 3, 4, 0, 5, np.nan, np.nan, 0])

        sparse = SparseArray(dense)
        res = sparse[(slice(4, None),)]
        exp = SparseArray(dense[4:])
        tm.assert_sp_array_equal(res, exp)

        sparse = SparseArray(dense, fill_value=0)
        res = sparse[(slice(4, None),)]
        exp = SparseArray(dense[4:], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        msg = "too many indices for array"
        with pytest.raises(IndexError, match=msg):
            sparse[4:, :]

        with pytest.raises(IndexError, match=msg):
            # check numpy compat
            dense[4:, :]

    def test_boolean_slice_empty(self):
        # [FUNCTION DOC]
        # Verifies that an all-False boolean mask yields an empty result that keeps the
        # original sparse dtype.
        arr = SparseArray([0, 1, 2])
        res = arr[[False, False, False]]
        assert res.dtype == arr.dtype

    def test_getitem_bool_sparse_array(self, arr):
        # [FUNCTION DOC]
        # Regression test (GH 23122): indexing with a boolean SparseArray mask, for a
        # mask with fill value True, its inversion, and a mask with a NaN fill value
        # (NaN treated as not selected). The result must equal the SparseArray of the
        # selected elements.
        # GH 23122
        spar_bool = SparseArray([False, True] * 5, dtype=np.bool_, fill_value=True)
        exp = SparseArray([np.nan, 2, np.nan, 5, 6])
        tm.assert_sp_array_equal(arr[spar_bool], exp)

        spar_bool = ~spar_bool
        res = arr[spar_bool]
        exp = SparseArray([np.nan, 1, 3, 4, np.nan])
        tm.assert_sp_array_equal(res, exp)

        spar_bool = SparseArray(
            [False, True, np.nan] * 3, dtype=np.bool_, fill_value=np.nan
        )
        res = arr[spar_bool]
        exp = SparseArray([np.nan, 3, 5])
        tm.assert_sp_array_equal(res, exp)

    def test_getitem_bool_sparse_array_as_comparison(self):
        # [FUNCTION DOC]
        # Regression test (GH 45110): using a comparison result (a boolean SparseArray)
        # as a mask selects the expected float elements and keeps NaN fill value.
        # GH 45110
        arr = SparseArray([1, 2, 3, 4, np.nan, np.nan], fill_value=np.nan)
        res = arr[arr > 2]
        exp = SparseArray([3.0, 4.0], fill_value=np.nan)
        tm.assert_sp_array_equal(res, exp)

    def test_get_item(self, arr):
        # [FUNCTION DOC]
        # Verifies scalar getitem for NaN-fill and zero-fill arrays, that out of range
        # positive and negative positions raise an index error with a message giving the
        # valid integer range, and that the last element equals position length minus one.
        zarr = SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)

        assert np.isnan(arr[1])
        assert arr[2] == 1
        assert arr[7] == 5

        assert zarr[0] == 0
        assert zarr[2] == 1
        assert zarr[7] == 5

        errmsg = "must be an integer between -10 and 10"

        with pytest.raises(IndexError, match=errmsg):
            arr[11]

        with pytest.raises(IndexError, match=errmsg):
            arr[-11]

        assert arr[-1] == arr[len(arr) - 1]


class TestSetitem:
    # [CLASS DOC]
    # Groups tests of SparseArray item assignment.
    def test_set_item(self, arr_data):
        # [FUNCTION DOC]
        # Verifies that assigning to a single position or to a slice of a SparseArray
        # raises a type error mentioning that assignment via setitem is unsupported.
        arr = SparseArray(arr_data).copy()

        def setitem():
            # [FUNCTION DOC]
            # HELPER: closure that attempts single-position assignment of 3 at position 5
            # on the enclosing test's array; used only to trigger the expected error.
            arr[5] = 3

        def setslice():
            # [FUNCTION DOC]
            # HELPER: closure that attempts to assign 2 to positions 1 through 4 of the
            # enclosing test's array; used only to trigger the expected error.
            arr[1:5] = 2

        with pytest.raises(TypeError, match="assignment via setitem"):
            setitem()

        with pytest.raises(TypeError, match="assignment via setitem"):
            setslice()


class TestTake:
    # [CLASS DOC]
    # Groups tests of SparseArray take.
    def test_take_scalar_raises(self, arr):
        # [FUNCTION DOC]
        # Verifies that take with a scalar instead of an array raises a value error
        # saying indices must be an array.
        msg = "'indices' must be an array, not a scalar '2'."
        with pytest.raises(ValueError, match=msg):
            arr.take(2)

    def test_take(self, arr_data, arr):
        # [FUNCTION DOC]
        # Verifies take with positive indices equals a SparseArray built from taking
        # the same positions of the dense data.
        exp = SparseArray(np.take(arr_data, [2, 3]))
        tm.assert_sp_array_equal(arr.take([2, 3]), exp)

        exp = SparseArray(np.take(arr_data, [0, 1, 2]))
        tm.assert_sp_array_equal(arr.take([0, 1, 2]), exp)

    def test_take_all_empty(self):
        # [FUNCTION DOC]
        # Verifies take with fill allowed on an integer sparse array whose elements
        # equal its fill value, with all positions valid, returns an equal array.
        sparse = pd.array([0, 0], dtype=SparseDtype("int64"))
        result = sparse.take([0, 1], allow_fill=True, fill_value=np.nan)
        tm.assert_sp_array_equal(sparse, result)

    def test_take_different_fill_value(self):
        # [FUNCTION DOC]
        # Verifies that take with fill allowed and a NaN fill value on an array with a
        # zero fill value fills the out-of-range (-1) position with NaN while keeping the
        # dtype (and its zero fill value) unchanged.
        # Take with a different fill value shouldn't overwrite the original
        sparse = pd.array([0.0], dtype=SparseDtype("float64", fill_value=0.0))
        result = sparse.take([0, -1], allow_fill=True, fill_value=np.nan)
        expected = pd.array([0, np.nan], dtype=sparse.dtype)
        tm.assert_sp_array_equal(expected, result)

    def test_take_fill_value(self):
        # [FUNCTION DOC]
        # Verifies take on a zero-fill-value array returns the SparseArray of the dense
        # elements at the requested positions.
        data = np.array([1, np.nan, 0, 3, 0])
        sparse = SparseArray(data, fill_value=0)

        exp = SparseArray(np.take(data, [0]), fill_value=0)
        tm.assert_sp_array_equal(sparse.take([0]), exp)

        exp = SparseArray(np.take(data, [1, 3, 4]), fill_value=0)
        tm.assert_sp_array_equal(sparse.take([1, 3, 4]), exp)

    def test_take_negative(self, arr_data, arr):
        # [FUNCTION DOC]
        # Verifies take with negative indices counts from the end, matching numpy take.
        exp = SparseArray(np.take(arr_data, [-1]))
        tm.assert_sp_array_equal(arr.take([-1]), exp)

        exp = SparseArray(np.take(arr_data, [-4, -3, -2]))
        tm.assert_sp_array_equal(arr.take([-4, -3, -2]), exp)

    def test_bad_take(self, arr):
        # [FUNCTION DOC]
        # Verifies that an index beyond the length raises an index error mentioning bounds.
        with pytest.raises(IndexError, match="bounds"):
            arr.take([11])

    def test_take_filling(self):
        # [FUNCTION DOC]
        # Take with a NaN fill value: without allow_fill negative indices count from the
        # end; with allow_fill, -1 produces NaN; allow_fill false ignores the fill_value
        # argument; indices below -1 with allow_fill raise a value error ("Invalid value
        # in indices"); out-of-range indices raise an index error.
        # similar tests as GH 12631
        sparse = SparseArray([np.nan, np.nan, 1, np.nan, 4])
        result = sparse.take(np.array([1, 0, -1]))
        expected = SparseArray([np.nan, np.nan, 4])
        tm.assert_sp_array_equal(result, expected)

        # TODO: actionable?
        # XXX: test change: fill_value=True -> allow_fill=True
        result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
        expected = SparseArray([np.nan, np.nan, np.nan])
        tm.assert_sp_array_equal(result, expected)

        # allow_fill=False
        result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = SparseArray([np.nan, np.nan, 4])
        tm.assert_sp_array_equal(result, expected)

        msg = "Invalid value in 'indices'"
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -2]), allow_fill=True)

        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -5]), allow_fill=True)

        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), allow_fill=True)

    def test_take_filling_fill_value(self):
        # [FUNCTION DOC]
        # Same scenarios as the previous take-filling test but for an array with fill
        # value zero. With allow_fill, -1 produces NaN (the dtype's NA value), not the
        # array fill value; invalid negative indices raise a value error; out-of-bounds
        # indices raise an index error.
        # same tests as GH#12631
        sparse = SparseArray([np.nan, 0, 1, 0, 4], fill_value=0)
        result = sparse.take(np.array([1, 0, -1]))
        expected = SparseArray([0, np.nan, 4], fill_value=0)
        tm.assert_sp_array_equal(result, expected)

        # fill_value
        result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
        # TODO: actionable?
        # XXX: behavior change.
        # the old way of filling self.fill_value doesn't follow EA rules.
        # It's supposed to be self.dtype.na_value (nan in this case)
        expected = SparseArray([0, np.nan, np.nan], fill_value=0)
        tm.assert_sp_array_equal(result, expected)

        # allow_fill=False
        result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = SparseArray([0, np.nan, 4], fill_value=0)
        tm.assert_sp_array_equal(result, expected)

        msg = "Invalid value in 'indices'."
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -2]), allow_fill=True)
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -5]), allow_fill=True)

        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), fill_value=True)

    @pytest.mark.parametrize("kind", ["block", "integer"])
    def test_take_filling_all_nan(self, kind):
        # [FUNCTION DOC]
        # Parametrized over sparse index kind (block, integer). For an all-NaN array,
        # take returns an all-NaN array of the same kind, both by default and with a
        # fill_value argument; out-of-bounds indices raise an index error.
        sparse = SparseArray([np.nan, np.nan, np.nan, np.nan, np.nan], kind=kind)
        result = sparse.take(np.array([1, 0, -1]))
        expected = SparseArray([np.nan, np.nan, np.nan], kind=kind)
        tm.assert_sp_array_equal(result, expected)

        result = sparse.take(np.array([1, 0, -1]), fill_value=True)
        expected = SparseArray([np.nan, np.nan, np.nan], kind=kind)
        tm.assert_sp_array_equal(result, expected)

        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), fill_value=True)


class TestWhere:
    # [CLASS DOC]
    # Groups tests of the where operation on SparseArray.
    def test_where_retain_fill_value(self):
        # [FUNCTION DOC]
        # Regression test (GH#45691): the private where operation keeps the original
        # fill value, and Series.where on a Series wrapping the array gives a matching
        # Series.
        # GH#45691 don't lose fill_value on _where
        arr = SparseArray([np.nan, 1.0], fill_value=0)

        mask = np.array([True, False])

        res = arr._where(~mask, 1)
        exp = SparseArray([1, 1.0], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        ser = pd.Series(arr)
        res = ser.where(~mask, 1)
        tm.assert_series_equal(res, pd.Series(exp))
