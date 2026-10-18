# [MODULE DOC]
# Tests for the take functions in the core algorithms module: the n-dimensional take with optional fill (dtype promotion rules, axes, dtypes) and the public extension take (bounds checks,
# fill values, input type validation).
from datetime import datetime

import numpy as np
import pytest

from pandas._libs import iNaT

from pandas import array
import pandas._testing as tm
import pandas.core.algorithms as algos


@pytest.fixture(
    params=[
        (np.int8, np.int16(127), np.int8),
        (np.int8, np.int16(128), np.int16),
        (np.int32, 1, np.int32),
        (np.int32, 2.0, np.float64),
        (np.int32, 3.0 + 4.0j, np.complex128),
        (np.int32, True, np.object_),
        (np.int32, "", np.object_),
        (np.float64, 1, np.float64),
        (np.float64, 2.0, np.float64),
        (np.float64, 3.0 + 4.0j, np.complex128),
        (np.float64, True, np.object_),
        (np.float64, "", np.object_),
        (np.complex128, 1, np.complex128),
        (np.complex128, 2.0, np.complex128),
        (np.complex128, 3.0 + 4.0j, np.complex128),
        (np.complex128, True, np.object_),
        (np.complex128, "", np.object_),
        (np.bool_, 1, np.object_),
        (np.bool_, 2.0, np.object_),
        (np.bool_, 3.0 + 4.0j, np.object_),
        (np.bool_, True, np.bool_),
        (np.bool_, "", np.object_),
    ]
)
def dtype_fill_out_dtype(request):
    # [FUNCTION DOC]
    # HELPER: parametrized fixture yielding triples (source dtype, fill value, expected output dtype) that capture dtype promotion rules when filling out-of-range positions: the dtype is kept if the fill value
    # fits, widened to a larger integer, float or complex when needed, and promoted to object when incompatible (e.g. boolean source with numeric fill).
    return request.param


class TestTake:
    # [CLASS DOC]
    # Tests for the n-dimensional take function.
    def test_1d_fill_nonna(self, dtype_fill_out_dtype):
        # [FUNCTION DOC]
        # Using the dtype/fill/output-dtype fixture. On a random 0-1 array cast to the source dtype, verifies an indexer with minus one gives the original values in the first three positions, the fill value in the last, and the
        # expected promoted dtype; with no negative index the dtype is unchanged and values match plain indexing.
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        data = np.random.default_rng(2).integers(0, 2, 4).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, fill_value=fill_value)
        assert (result[[0, 1, 2]] == data[[2, 1, 0]]).all()
        assert result[3] == fill_value
        assert result.dtype == out_dtype

        indexer = [2, 1, 0, 1]

        result = algos.take_nd(data, indexer, fill_value=fill_value)
        assert (result[[0, 1, 2, 3]] == data[indexer]).all()
        assert result.dtype == dtype

    def test_2d_fill_nonna(self, dtype_fill_out_dtype):
        # [FUNCTION DOC]
        # As the one-dimensional case but for a 5 by 3 array along both axes: values reordered, the filled row or column equals the fill value, and the dtype is promoted only when a fill is needed.
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        data = np.random.default_rng(2).integers(0, 2, (5, 3)).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2], :] == data[[2, 1, 0], :]).all()
        assert (result[3, :] == fill_value).all()
        assert result.dtype == out_dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2]] == data[:, [2, 1, 0]]).all()
        assert (result[:, 3] == fill_value).all()
        assert result.dtype == out_dtype

        indexer = [2, 1, 0, 1]
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2, 3], :] == data[indexer, :]).all()
        assert result.dtype == dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2, 3]] == data[:, indexer]).all()
        assert result.dtype == dtype

    def test_3d_fill_nonna(self, dtype_fill_out_dtype):
        # [FUNCTION DOC]
        # As the one-dimensional case but for a 5 by 4 by 3 array along all three axes.
        dtype, fill_value, out_dtype = dtype_fill_out_dtype

        data = np.random.default_rng(2).integers(0, 2, (5, 4, 3)).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2], :, :] == data[[2, 1, 0], :, :]).all()
        assert (result[3, :, :] == fill_value).all()
        assert result.dtype == out_dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2], :] == data[:, [2, 1, 0], :]).all()
        assert (result[:, 3, :] == fill_value).all()
        assert result.dtype == out_dtype

        result = algos.take_nd(data, indexer, axis=2, fill_value=fill_value)
        assert (result[:, :, [0, 1, 2]] == data[:, :, [2, 1, 0]]).all()
        assert (result[:, :, 3] == fill_value).all()
        assert result.dtype == out_dtype

        indexer = [2, 1, 0, 1]
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2, 3], :, :] == data[indexer, :, :]).all()
        assert result.dtype == dtype

        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2, 3], :] == data[:, indexer, :]).all()
        assert result.dtype == dtype

        result = algos.take_nd(data, indexer, axis=2, fill_value=fill_value)
        assert (result[:, :, [0, 1, 2, 3]] == data[:, :, indexer]).all()
        assert result.dtype == dtype

    def test_1d_other_dtypes(self):
        # [FUNCTION DOC]
        # Verifies take of a float32 vector with a negative position gives the same as plain take with NaN in the filled position.
        arr = np.random.default_rng(2).standard_normal(10).astype(np.float32)

        indexer = [1, 2, 3, -1]
        result = algos.take_nd(arr, indexer)
        expected = arr.take(indexer)
        expected[-1] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_2d_other_dtypes(self):
        # [FUNCTION DOC]
        # Verifies take of a float32 matrix on both axes with a negative position matches plain take with NaN in the filled row or column.
        arr = np.random.default_rng(2).standard_normal((10, 5)).astype(np.float32)

        indexer = [1, 2, 3, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected[-1] = np.nan
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected[:, -1] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_1d_bool(self):
        # [FUNCTION DOC]
        # Verifies boolean take without fill matches plain take (staying boolean), and that a negative index promotes the result to object dtype.
        arr = np.array([0, 1, 0], dtype=bool)

        result = algos.take_nd(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, -1])
        assert result.dtype == np.object_

    def test_2d_bool(self):
        # [FUNCTION DOC]
        # Verifies boolean matrix take along both axes matches plain take, and a negative index promotes to object dtype.
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)

        result = algos.take_nd(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1], axis=0)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, 2, 1], axis=1)
        expected = arr.take([0, 2, 2, 1], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, -1])
        assert result.dtype == np.object_

    def test_2d_float32(self):
        # [FUNCTION DOC]
        # Verifies float32 matrix take with several fill positions along both axes puts NaN in the filled rows or columns.
        arr = np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32)
        indexer = [0, 2, -1, 1, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)

        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = np.nan
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_2d_datetime64(self):
        # [FUNCTION DOC]
        # Builds a datetime64 nanosecond matrix of random values in 2005; verifies take with negative positions gives not-a-time at filled rows or columns (checked via the integer view), and an explicit datetime fill value is used instead, along both axes.
        # 2005/01/01 - 2006/01/01
        arr = (
            np.random.default_rng(2).integers(11_045_376, 11_360_736, (5, 3))
            * 100_000_000_000
        )
        arr = arr.view(dtype="datetime64[ns]")
        indexer = [0, 2, -1, 1, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected.view(np.int64)[[2, 4], :] = iNaT
        tm.assert_almost_equal(result, expected)

        result = algos.take_nd(arr, indexer, axis=0, fill_value=datetime(2007, 1, 1))
        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected.view(np.int64)[:, [2, 4]] = iNaT
        tm.assert_almost_equal(result, expected)

        result = algos.take_nd(arr, indexer, axis=1, fill_value=datetime(2007, 1, 1))
        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

    def test_take_axis_0(self):
        # [FUNCTION DOC]
        # Verifies the public take on a 4 by 3 matrix with indexes zero and minus one selects the first and last rows (negative as from the end), and with fill allowed the minus-one entry becomes the fill value (zero row).
        arr = np.arange(12).reshape(4, 3)
        result = algos.take(arr, [0, -1])
        expected = np.array([[0, 1, 2], [9, 10, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # allow_fill=True
        result = algos.take(arr, [0, -1], allow_fill=True, fill_value=0)
        expected = np.array([[0, 1, 2], [0, 0, 0]])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_axis_1(self):
        # [FUNCTION DOC]
        # Verifies take along columns with a negative index, with and without fill; regression check for a GitHub issue that an out-of-bounds positive index is validated against the correct axis and raises an index error.
        arr = np.arange(12).reshape(4, 3)
        result = algos.take(arr, [0, -1], axis=1)
        expected = np.array([[0, 2], [3, 5], [6, 8], [9, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # allow_fill=True
        result = algos.take(arr, [0, -1], axis=1, allow_fill=True, fill_value=0)
        expected = np.array([[0, 0], [3, 0], [6, 0], [9, 0]])
        tm.assert_numpy_array_equal(result, expected)

        # GH#26976 make sure we validate along the correct axis
        with pytest.raises(IndexError, match="indices are out-of-bounds"):
            algos.take(arr, [0, 3], axis=1, allow_fill=True, fill_value=0)

    def test_take_non_hashable_fill_value(self):
        # [FUNCTION DOC]
        # Verifies a list fill value raises a value error requiring a scalar for integer arrays when fill is allowed, but is permitted for object-dtype arrays and placed as the element.
        arr = np.array([1, 2, 3])
        indexer = np.array([1, -1])
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            algos.take(arr, indexer, allow_fill=True, fill_value=[1])

        # with object dtype it is allowed
        arr = np.array([1, 2, 3], dtype=object)
        result = algos.take(arr, indexer, allow_fill=True, fill_value=[1])
        expected = np.array([2, [1]], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


class TestExtensionTake:
    # [CLASS DOC]
    # Tests for the take function exposed through the public extensions API (comment in code says so).
    # The take method found in pd.api.extensions

    def test_bounds_check_large(self):
        # [FUNCTION DOC]
        # Verifies out-of-range positive positions raise an index error: with fill allowed, the message says indices are out-of-bounds; without fill, NumPy's out-of-bounds message.
        arr = np.array([1, 2])

        msg = "indices are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=True)

        msg = "index 2 is out of bounds for( axis 0 with)? size 2"
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=False)

    def test_bounds_check_small(self):
        # [FUNCTION DOC]
        # Verifies that with fill allowed, positions below minus one raise a value error saying the indices contain values less than allowed; without fill, negative positions count from the end.
        arr = np.array([1, 2, 3], dtype=np.int64)
        indexer = [0, -1, -2]

        msg = r"'indices' contains values less than allowed \(-2 < -1\)"
        with pytest.raises(ValueError, match=msg):
            algos.take(arr, indexer, allow_fill=True)

        result = algos.take(arr, indexer)
        expected = np.array([1, 3, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("allow_fill", [True, False])
    def test_take_empty(self, allow_fill):
        # [FUNCTION DOC]
        # Parametrized on the fill-allowed flag. Verifies taking nothing from an empty array returns an equal empty array, and taking a position from it raises an index error (either of two accepted messages).
        arr = np.array([], dtype=np.int64)
        # empty take is ok
        result = algos.take(arr, [], allow_fill=allow_fill)
        tm.assert_numpy_array_equal(arr, result)

        msg = "|".join(
            [
                "cannot do a non-empty take from an empty axes.",
                "indices are out-of-bounds",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [0], allow_fill=allow_fill)

    def test_take_na_empty(self):
        # [FUNCTION DOC]
        # Verifies taking only missing positions from an empty float array with fill allowed and a fill value gives an array of that fill value.
        result = algos.take(np.array([]), [-1, -1], allow_fill=True, fill_value=0.0)
        expected = np.array([0.0, 0.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_coerces_list(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue (coercion of lists was deprecated and is now disabled). Verifies a plain list input raises a type error stating the accepted input types.
        # GH#52981 coercing is deprecated, disabled in 3.0
        arr = [1, 2, 3]
        msg = (
            "pd.api.extensions.take requires a numpy.ndarray, ExtensionArray, "
            "Index, Series, or NumpyExtensionArray got list"
        )
        with pytest.raises(TypeError, match=msg):
            algos.take(arr, [0, 0])

    def test_take_NumpyExtensionArray(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Verifies take returns the scalar two for a one-position take on a NumPy-extension-array wrapping complex values and on an integer extension array.
        # GH#59177
        arr = array([1 + 1j, 2, 3])  # NumpyEADtype('complex128') (NumpyExtensionArray)
        assert algos.take(arr, [2]) == 2
        arr = array([1, 2, 3])  # Int64Dtype() (ExtensionArray)
        assert algos.take(arr, [2]) == 2
