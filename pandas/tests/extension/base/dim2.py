"""
Tests for 2D compatibility.
"""
# [MODULE DOC]
# Base tests for two-dimensional behavior of extension arrays. Public contents:
# Dim2CompatTests (for arrays that support 2D storage; not for pyarrow-backed arrays) and
# NDArrayBacked2DTests (extra tests specific to numpy-backed arrays, such as memory layout
# handling in copy). Uses data and data_missing fixtures; reshaping the 1D fixture produces the
# 2D arrays under test.

import numpy as np
import pytest

from pandas._libs.missing import is_matching_na

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
)

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE


class Dim2CompatTests:
    # [CLASS DOC]
    # Mixin of 2D tests. Includes an autouse fixture that skips the tests for dtypes that do
    # not support 2D. Most tests build a 2D array by repeating the 1D data and reshaping to a
    # fixed number of columns, then compare 2D operations with the 1D equivalents.
    # Note: these are ONLY for ExtensionArray subclasses that support 2D arrays.
    #  i.e. not for pyarrow-backed EAs.

    @pytest.fixture(autouse=True)
    def skip_if_doesnt_support_2d(self, dtype, request):
        # [FUNCTION DOC]
        # HELPER: autouse fixture skipping tests when the dtype does not support 2D.
        # Inputs: dtype and the request fixture. When mixed into the combined extension test class only
        # tests whose qualified name begins with this class's name are skipped; other inherited tests
        # are left alone.
        if not dtype._supports_2d:
            node = request.node
            # In cases where we are mixed in to ExtensionTests, we only want to
            #  skip tests that are defined in Dim2CompatTests
            test_func = node._obj
            if test_func.__qualname__.startswith("Dim2CompatTests"):
                # TODO: is there a less hacky way of checking this?
                pytest.skip(f"{dtype} does not support 2D.")

    def test_shift_2d(self, data):
        # [FUNCTION DOC]
        # Shifts a two-column array by 1 and -2, with no fill value and with the first element as fill
        # value; each column of the result must equal the 1D shift with the same arguments.
        arr2d = data.repeat(2).reshape(-1, 2)

        for n in [1, -2]:
            for fill_value in [None, data[0]]:
                result = arr2d.shift(n, fill_value=fill_value)
                expected_col = data.shift(n, fill_value=fill_value)
                tm.assert_extension_array_equal(result[:, 0], expected_col)
                tm.assert_extension_array_equal(result[:, 1], expected_col)

    def test_transpose(self, data):
        # [FUNCTION DOC]
        # Asserts the transpose of a two-column array has the reversed shape. Also guards that the
        # shape is not square so the check is meaningful.
        arr2d = data.repeat(2).reshape(-1, 2)
        shape = arr2d.shape
        assert shape[0] != shape[-1]  # otherwise the rest of the test is useless

        assert arr2d.T.shape == shape[::-1]

    def test_frame_from_2d_array(self, data):
        # [FUNCTION DOC]
        # Asserts a frame built from a 2D array equals a frame built from a dict of its two
        # columns.
        arr2d = data.repeat(2).reshape(-1, 2)

        df = pd.DataFrame(arr2d)
        expected = pd.DataFrame({0: arr2d[:, 0], 1: arr2d[:, 1]})
        tm.assert_frame_equal(df, expected)

    def test_swapaxes(self, data):
        # [FUNCTION DOC]
        # Asserts swapping axes 0 and 1 of a two-column array equals its transpose.
        arr2d = data.repeat(2).reshape(-1, 2)

        result = arr2d.swapaxes(0, 1)
        expected = arr2d.T
        tm.assert_extension_array_equal(result, expected)

    def test_delete_2d(self, data):
        # [FUNCTION DOC]
        # Three-column array: deleting position 1 along axis 0 equals the 1D delete repeated and
        # reshaped; deleting position 1 along axis 1 equals a two-column array.
        arr2d = data.repeat(3).reshape(-1, 3)

        # axis = 0
        result = arr2d.delete(1, axis=0)
        expected = data.delete(1).repeat(3).reshape(-1, 3)
        tm.assert_extension_array_equal(result, expected)

        # axis = 1
        result = arr2d.delete(1, axis=1)
        expected = data.repeat(2).reshape(-1, 2)
        tm.assert_extension_array_equal(result, expected)

    def test_take_2d(self, data):
        # [FUNCTION DOC]
        # For a single-column array, taking positions 0, 0, -1 along axis 0 equals the 1D take
        # reshaped into a column.
        arr2d = data.reshape(-1, 1)

        result = arr2d.take([0, 0, -1], axis=0)

        expected = data.take([0, 0, -1]).reshape(-1, 1)
        tm.assert_extension_array_equal(result, expected)

    def test_repr_2d(self, data):
        # [FUNCTION DOC]
        # Asserts the repr of a one-row and of a one-column reshaped array contains the class
        # marker exactly once (can fail if an element contains the class name).
        # this could fail in a corner case where an element contained the name
        res = repr(data.reshape(1, -1))
        assert res.count(f"<{type(data).__name__}") == 1

        res = repr(data.reshape(-1, 1))
        assert res.count(f"<{type(data).__name__}") == 1

    def test_reshape(self, data):
        # [FUNCTION DOC]
        # Reshaping to a column with inferred length gives shape of size by one and the same length,
        # whether the shape is passed as separate arguments or as a tuple. Reshaping to an incompatible
        # shape (size by two) must raise a value error originating from outside pandas, in both calling
        # forms.
        arr2d = data.reshape(-1, 1)
        assert arr2d.shape == (data.size, 1)
        assert len(arr2d) == len(data)

        arr2d = data.reshape((-1, 1))
        assert arr2d.shape == (data.size, 1)
        assert len(arr2d) == len(data)

        with tm.external_error_raised(ValueError):
            data.reshape((data.size, 2))
        with tm.external_error_raised(ValueError):
            data.reshape(data.size, 2)

    def test_getitem_2d(self, data):
        # [FUNCTION DOC]
        # One-row array: integer 0 returns the 1D data; integer 1 and -2 raise an index error; full
        # slices (one or two axes) return an equal 2D array; selecting all rows of column 0 equals a
        # one-element 1D selection; indexing a 1D array with a slice and a new-axis marker gives the
        # transpose of the 2D array.
        arr2d = data.reshape(1, -1)

        result = arr2d[0]
        tm.assert_extension_array_equal(result, data)

        with pytest.raises(IndexError):
            arr2d[1]

        with pytest.raises(IndexError):
            arr2d[-2]

        result = arr2d[:]
        tm.assert_extension_array_equal(result, arr2d)

        result = arr2d[:, :]
        tm.assert_extension_array_equal(result, arr2d)

        result = arr2d[:, 0]
        expected = data[[0]]
        tm.assert_extension_array_equal(result, expected)

        # dimension-expanding getitem on 1D
        result = data[:, np.newaxis]
        tm.assert_extension_array_equal(result, arr2d.T)

    def test_iter_2d(self, data):
        # [FUNCTION DOC]
        # Iterating a one-row 2D array yields as many items as rows, each an array of the same class
        # and dtype, one-dimensional, with length equal to the column count.
        arr2d = data.reshape(1, -1)

        objs = list(iter(arr2d))
        assert len(objs) == arr2d.shape[0]

        for obj in objs:
            assert isinstance(obj, type(data))
            assert obj.dtype == data.dtype
            assert obj.ndim == 1
            assert len(obj) == arr2d.shape[1]

    def test_tolist_2d(self, data):
        # [FUNCTION DOC]
        # The list conversion of a one-row 2D array is a list of lists equal to the 1D list wrapped
        # in an outer list.
        arr2d = data.reshape(1, -1)

        result = arr2d.tolist()
        expected = [data.tolist()]

        assert isinstance(result, list)
        assert all(isinstance(x, list) for x in result)

        assert result == expected

    def test_concat_2d(self, data):
        # [FUNCTION DOC]
        # Same-type concat of two copies of a two-column array: along axis 0 equals the 1D concat of
        # four copies reshaped; along axis 1 gives four columns whose halves equal the inputs; axis 2
        # raises a value error saying the axis is out of bounds for a two-dimensional array.
        left = type(data)._concat_same_type([data, data]).reshape(-1, 2)
        right = left.copy()

        # axis=0
        result = left._concat_same_type([left, right], axis=0)
        expected = data._concat_same_type([data] * 4).reshape(-1, 2)
        tm.assert_extension_array_equal(result, expected)

        # axis=1
        result = left._concat_same_type([left, right], axis=1)
        assert result.shape == (len(data), 4)
        tm.assert_extension_array_equal(result[:, :2], left)
        tm.assert_extension_array_equal(result[:, 2:], right)

        # axis > 1 -> invalid
        msg = "axis 2 is out of bounds for array of dimension 2"
        with pytest.raises(ValueError, match=msg):
            left._concat_same_type([left, right], axis=2)

    @pytest.mark.parametrize("method", ["backfill", "pad"])
    def test_fillna_2d_method(self, data_missing, method):
        # [FUNCTION DOC]
        # Parametrized over backfill and pad. Builds a two-by-two array whose first row is missing and
        # second is not, applies the internal pad-or-backfill (always along axis 0, no limit) and asserts
        # equality with the 1D result repeated and reshaped. Repeats with the rows reversed so the
        # backfill case is not a no-op.
        # pad_or_backfill is always along axis=0
        arr = data_missing.repeat(2).reshape(2, 2)
        assert arr[0].isna().all()
        assert not arr[1].isna().any()

        result = arr._pad_or_backfill(method=method, limit=None)

        expected = data_missing._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
        tm.assert_extension_array_equal(result, expected)

        # Reverse so that backfill is not a no-op.
        arr2 = arr[::-1]
        assert not arr2[0].isna().any()
        assert arr2[1].isna().all()

        result2 = arr2._pad_or_backfill(method=method, limit=None)

        expected2 = (
            data_missing[::-1]._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
        )
        tm.assert_extension_array_equal(result2, expected2)

    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    def test_reductions_2d_axis_none(self, data, method):
        # [FUNCTION DOC]
        # Parametrized over mean, median, var, std, sum, prod. Compares a reduction over all axes of
        # a one-row 2D array with the 1D reduction. If the 1D reduction raises, the 2D one must raise
        # the same exception type; otherwise results must be equal or both missing.
        arr2d = data.reshape(1, -1)

        err_expected = None
        err_result = None
        try:
            expected = getattr(data, method)()
        except Exception as err:
            # if the 1D reduction is invalid, the 2D reduction should be as well
            err_expected = err
            try:
                result = getattr(arr2d, method)(axis=None)
            except Exception as err2:
                err_result = err2

        else:
            result = getattr(arr2d, method)(axis=None)

        if err_result is not None or err_expected is not None:
            assert type(err_result) == type(err_expected)
            return

        assert is_matching_na(result, expected) or result == expected

    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    @pytest.mark.parametrize("min_count", [0, 1])
    def test_reductions_2d_axis0(self, data, method, min_count):
        # [FUNCTION DOC]
        # Parametrized over the reduction name and min_count 0 and 1 (min_count only for sum and
        # prod; otherwise skipped). Reduces a one-row 2D array along axis 0 (ddof 0 for std and var, so
        # results are zero not missing). If it raises, the 1D reduction must raise the same type, else
        # the test fails. Expected results: sum and prod preserve the (platform int-widened) dtype and
        # with min_count 0 fill missing with the identity element; median returns the data; mean, std
        # and var work on floats for integer and boolean data, the mean equals the data and std/var
        # equal zero.
        if min_count == 1 and method not in ["sum", "prod"]:
            pytest.skip(f"min_count not relevant for {method}")

        arr2d = data.reshape(1, -1)

        kwargs = {}
        if method in ["std", "var"]:
            # pass ddof=0 so we get all-zero std instead of all-NA std
            kwargs["ddof"] = 0
        elif method in ["prod", "sum"]:
            kwargs["min_count"] = min_count

        try:
            result = getattr(arr2d, method)(axis=0, **kwargs)
        except Exception as err:
            try:
                getattr(data, method)()
            except Exception as err2:
                assert type(err) == type(err2)
                return
            else:
                raise AssertionError("Both reductions should raise or neither")

        def get_reduction_result_dtype(dtype):
            # [FUNCTION DOC]
            # HELPER: maps an input numpy-like dtype to the dtype expected from a sum or product. Eight
            # byte dtypes are returned unchanged; signed integer and boolean kinds map to the platform
            # integer extension dtype; otherwise (unsigned) to the platform unsigned dtype. Needed because
            # some platforms use 32-bit defaults.
            # windows and 32bit builds will in some cases have int32/uint32
            #  where other builds will have int64/uint64.
            if dtype.itemsize == 8:
                return dtype
            elif dtype.kind in "ib":
                return NUMPY_INT_TO_DTYPE[np.dtype(int)]
            else:
                # i.e. dtype.kind == "u"
                return NUMPY_INT_TO_DTYPE[np.dtype("uint")]

        if method in ["sum", "prod"]:
            # std and var are not dtype-preserving
            expected = data
            if data.dtype.kind in "iub":
                dtype = get_reduction_result_dtype(data.dtype)
                expected = data.astype(dtype)
                assert dtype == expected.dtype

            if min_count == 0:
                fill_value = 1 if method == "prod" else 0
                expected = expected.fillna(fill_value)

            tm.assert_extension_array_equal(result, expected)
        elif method == "median":
            # std and var are not dtype-preserving
            expected = data
            tm.assert_extension_array_equal(result, expected)
        elif method in ["mean", "std", "var"]:
            if is_integer_dtype(data) or is_bool_dtype(data):
                data = data.astype("Float64")
            if method == "mean":
                tm.assert_extension_array_equal(result, data)
            else:
                tm.assert_extension_array_equal(result, data - data)

    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    def test_reductions_2d_axis1(self, data, method):
        # [FUNCTION DOC]
        # Parametrized over the reduction name. Reduces a one-row 2D array along axis 1. If it
        # raises, the 1D reduction must raise the same type. Otherwise the result has shape of one and
        # its element is either both-missing or equal to the 1D scalar reduction (weaker than dtype
        # preservation).
        arr2d = data.reshape(1, -1)

        try:
            result = getattr(arr2d, method)(axis=1)
        except Exception as err:
            try:
                getattr(data, method)()
            except Exception as err2:
                assert type(err) == type(err2)
                return
            else:
                raise AssertionError("Both reductions should raise or neither")

        # not necessarily type/dtype-preserving, so weaker assertions
        assert result.shape == (1,)
        expected_scalar = getattr(data, method)()
        res = result[0]
        assert is_matching_na(res, expected_scalar) or res == expected_scalar


class NDArrayBacked2DTests(Dim2CompatTests):
    # [CLASS DOC]
    # Extra 2D tests for numpy-backed extension arrays; inherits the generic 2D tests.
    # Requires access to the private underlying ndarray of the array.
    # More specific tests for NDArrayBackedExtensionArray subclasses

    def test_copy_order(self, data):
        # [FUNCTION DOC]
        # Verifies that the copy method follows numpy semantics for the order argument. A C-contiguous
        # 2D array stays C-contiguous when copied, including when copying a strided slice. Order F
        # gives a Fortran-contiguous non-C result; order K preserves layout (C stays C, the transpose
        # becomes F). An invalid order letter raises a value error listing the valid letters.
        # A non-contiguous row-strided slice is neither contiguous; copying it by default or with C or K
        # gives a C-contiguous result, with F gives an F-contiguous result.
        # We should be matching numpy semantics for the "order" keyword in 'copy'
        arr2d = data.repeat(2).reshape(-1, 2)
        assert arr2d._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d.copy()
        assert res._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d[::2, ::2].copy()
        assert res._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d.copy("F")
        assert not res._ndarray.flags["C_CONTIGUOUS"]
        assert res._ndarray.flags["F_CONTIGUOUS"]

        res = arr2d.copy("K")
        assert res._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d.T.copy("K")
        assert not res._ndarray.flags["C_CONTIGUOUS"]
        assert res._ndarray.flags["F_CONTIGUOUS"]

        # order not accepted by numpy
        msg = r"order must be one of 'C', 'F', 'A', or 'K' \(got 'Q'\)"
        with pytest.raises(ValueError, match=msg):
            arr2d.copy("Q")

        # neither contiguity
        arr_nc = arr2d[::2]
        assert not arr_nc._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc._ndarray.flags["F_CONTIGUOUS"]

        assert arr_nc.copy()._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy()._ndarray.flags["F_CONTIGUOUS"]

        assert arr_nc.copy("C")._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy("C")._ndarray.flags["F_CONTIGUOUS"]

        assert not arr_nc.copy("F")._ndarray.flags["C_CONTIGUOUS"]
        assert arr_nc.copy("F")._ndarray.flags["F_CONTIGUOUS"]

        assert arr_nc.copy("K")._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy("K")._ndarray.flags["F_CONTIGUOUS"]
