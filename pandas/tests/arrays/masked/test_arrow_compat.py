# [MODULE DOC]
# Tests for converting masked arrays (integer, float, boolean) to and from pyarrow: array conversion, table round trips,
# types-mapper conversion, zero-chunk tables, unsigned integer mismatches, sliced tables, the buffer-to-numpy-and-mask helper and null/mismatched
# arrow input. The whole module is skipped if pyarrow is missing (import-time skip) and ignores a block-manager deprecation warning via a module marker.
# Builds a module-level list of masked arrays used to parametrize the data fixture.

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


pa = pytest.importorskip("pyarrow")

from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask

arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]
arrays += [pd.array([True, False, True, None], dtype="boolean")]


@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over arrays with a trailing missing value, one per integer extension dtype, float extension dtype, and boolean; ids are dtype names.
    """
    Fixture returning parametrized array from given dtype, including integer,
    float and boolean
    """
    return request.param


def test_arrow_array(data):
    # [FUNCTION DOC]
    # Converting a masked array to arrow equals an arrow array built from the object conversion with nulls and the matching numpy-derived type.
    arr = pa.array(data)
    expected = pa.array(
        data.to_numpy(object, na_value=None),
        type=pa.from_numpy_dtype(data.dtype.numpy_dtype),
    )
    assert arr.equals(expected)


def test_arrow_roundtrip(data):
    # [FUNCTION DOC]
    # A frame with the array column converted to an arrow table has the numpy-named arrow type for that column, and converting back restores the masked dtype and equal frame.
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    assert table.field("a").type == str(data.dtype.numpy_dtype)

    result = table.to_pandas()
    assert result["a"].dtype == data.dtype
    tm.assert_frame_equal(result, df)


def test_dataframe_from_arrow_types_mapper():
    # [FUNCTION DOC]
    # Converting a record batch to pandas with a types-mapper callback (boolean to nullable boolean, integer to Int64, else default) yields nullable boolean, Int64 and an 8-bit
    # column widened to Int64.
    def types_mapper(arrow_type):
        # [FUNCTION DOC]
        # HELPER: types-mapper callback for the test; takes an arrow type and returns the nullable boolean dtype for boolean, the Int64 dtype for any integer type, and nothing otherwise.
        if pa.types.is_boolean(arrow_type):
            return pd.BooleanDtype()
        elif pa.types.is_integer(arrow_type):
            return pd.Int64Dtype()

    bools_array = pa.array([True, None, False], type=pa.bool_())
    ints_array = pa.array([1, None, 2], type=pa.int64())
    small_ints_array = pa.array([-1, 0, 7], type=pa.int8())
    record_batch = pa.RecordBatch.from_arrays(
        [bools_array, ints_array, small_ints_array], ["bools", "ints", "small_ints"]
    )
    result = record_batch.to_pandas(types_mapper=types_mapper)
    bools = pd.Series([True, None, False], dtype="boolean")
    ints = pd.Series([1, None, 2], dtype="Int64")
    small_ints = pd.Series([-1, 0, 7], dtype="Int64")
    expected = pd.DataFrame({"bools": bools, "ints": ints, "small_ints": small_ints})
    tm.assert_frame_equal(result, expected)


def test_arrow_load_from_zero_chunks(data):
    # [FUNCTION DOC]
    # A zero-length array table rebuilt with zero chunks converts back to an empty frame with the masked dtype (regression).
    # GH-41040

    df = pd.DataFrame({"a": data[0:0]})
    table = pa.table(df)
    assert table.field("a").type == str(data.dtype.numpy_dtype)
    table = pa.table(
        [pa.chunked_array([], type=table.field("a").type)], schema=table.schema
    )
    result = table.to_pandas()
    assert result["a"].dtype == data.dtype
    tm.assert_frame_equal(result, df)


def test_arrow_from_arrow_uint():
    # [FUNCTION DOC]
    # Converting a 64-bit signed arrow array through the unsigned 32-bit dtype hook yields the matching unsigned nullable array (type mismatch tolerated).
    # https://github.com/pandas-dev/pandas/issues/31896
    # possible mismatch in types

    dtype = pd.UInt32Dtype()
    result = dtype.__from_arrow__(pa.array([1, 2, 3, 4, None], type="int64"))
    expected = pd.array([1, 2, 3, 4, None], dtype="UInt32")

    tm.assert_extension_array_equal(result, expected)


def test_arrow_sliced(data):
    # [FUNCTION DOC]
    # A table sliced from position two onward converts to the matching sliced frame, both with missing values and (after filling) without them.
    # https://github.com/pandas-dev/pandas/issues/38525

    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    result = table.slice(2, None).to_pandas()
    expected = df.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)

    # no missing values
    df2 = df.fillna(data[0])
    table = pa.table(df2)
    result = table.slice(2, None).to_pandas()
    expected = df2.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.fixture
def np_dtype_to_arrays(any_real_numpy_dtype):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over real numpy dtypes; returns the numpy dtype, an arrow array of the matching type holding zero, one, two and a null (which forces a validity bitmap),
    # the expected first three numpy values and the expected validity mask.
    """
    Fixture returning actual and expected dtype, pandas and numpy arrays and
    mask from a given numpy dtype
    """
    np_dtype = np.dtype(any_real_numpy_dtype)
    pa_type = pa.from_numpy_dtype(np_dtype)

    # None ensures the creation of a bitmask buffer.
    pa_array = pa.array([0, 1, 2, None], type=pa_type)
    # Since masked Arrow buffer slots are not required to contain a specific
    # value, assert only the first three values of the created np.array
    np_expected = np.array([0, 1, 2], dtype=np_dtype)
    mask_expected = np.array([True, True, True, False])
    return np_dtype, pa_array, np_expected, mask_expected


def test_pyarrow_array_to_numpy_and_mask(np_dtype_to_arrays):
    # [FUNCTION DOC]
    # Tests the helper that turns an arrow array into numpy data and a mask: the plain array, one with trailing padding in the data buffer, one with an array offset and a shifted validity bitmap,
    # and an empty array with non-empty buffers must all give the expected first three values and mask.
    """
    Test conversion from pyarrow array to numpy array.

    Modifies the pyarrow buffer to contain padding and offset, which are
    considered valid buffers by pyarrow.

    Also tests empty pyarrow arrays with non empty buffers.
    See https://github.com/pandas-dev/pandas/issues/40896
    """
    np_dtype, pa_array, np_expected, mask_expected = np_dtype_to_arrays
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    mask_buffer = pa_array.buffers()[0]
    data_buffer = pa_array.buffers()[1]
    data_buffer_bytes = pa_array.buffers()[1].to_pybytes()

    # Add trailing padding to the buffer.
    data_buffer_trail = pa.py_buffer(data_buffer_bytes + b"\x00")
    pa_array_trail = pa.Array.from_buffers(
        type=pa_array.type,
        length=len(pa_array),
        buffers=[mask_buffer, data_buffer_trail],
        offset=pa_array.offset,
    )
    pa_array_trail.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_trail, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    # Add offset to the buffer.
    offset = b"\x00" * (pa_array.type.bit_width // 8)
    data_buffer_offset = pa.py_buffer(offset + data_buffer_bytes)
    mask_buffer_offset = pa.py_buffer(b"\x0e")
    pa_array_offset = pa.Array.from_buffers(
        type=pa_array.type,
        length=len(pa_array),
        buffers=[mask_buffer_offset, data_buffer_offset],
        offset=pa_array.offset + 1,
    )
    pa_array_offset.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    # Empty array
    np_expected_empty = np.array([], dtype=np_dtype)
    mask_expected_empty = np.array([], dtype=np.bool_)

    pa_array_offset = pa.Array.from_buffers(
        type=pa_array.type,
        length=0,
        buffers=[mask_buffer, data_buffer],
        offset=pa_array.offset,
    )
    pa_array_offset.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected_empty)
    tm.assert_numpy_array_equal(mask, mask_expected_empty)


@pytest.mark.parametrize(
    "arr", [pa.nulls(10), pa.chunked_array([pa.nulls(4), pa.nulls(6)])]
)
def test_from_arrow_null(data, arr):
    # [FUNCTION DOC]
    # Parametrized over an all-null array and a chunked all-null array. The dtype's arrow hook gives an array of ten missing values.
    res = data.dtype.__from_arrow__(arr)
    assert res.isna().all()
    assert len(res) == 10


def test_from_arrow_type_error(data):
    # [FUNCTION DOC]
    # Passing a string-typed arrow array to the dtype's arrow hook raises a type error (message not checked).
    # ensure that __from_arrow__ returns a TypeError when getting a wrong
    # array type

    arr = pa.array(data).cast("string")
    with pytest.raises(TypeError, match=None):
        # we don't test the exact error message, only the fact that it raises
        # a TypeError is relevant
        data.dtype.__from_arrow__(arr)
