# [MODULE DOC]
# Base tests for constructing arrays, series and frames from extension data. Public
# contents: BaseConstructorsTests mixin. Uses the data, dtype and na_value fixtures.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock


class BaseConstructorsTests:
    # [CLASS DOC]
    # Mixin with constructor tests for an extension array and its dtype.
    def test_from_sequence_from_cls(self, data):
        # [FUNCTION DOC]
        # Asserts the class-level sequence constructor applied to an existing array with its dtype
        # returns an equal array, and does the same for an empty slice.
        result = type(data)._from_sequence(data, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

        data = data[:0]
        result = type(data)._from_sequence(data, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    def test_array_from_scalars(self, data):
        # [FUNCTION DOC]
        # Builds an array from the first three scalars via the sequence constructor with the array's
        # dtype and asserts the result is an instance of the array class.
        scalars = [data[0], data[1], data[2]]
        result = data._from_sequence(scalars, dtype=data.dtype)
        assert isinstance(result, type(data))

    def test_series_constructor(self, data):
        # [FUNCTION DOC]
        # Builds a series from the array without copying and asserts matching dtype and length,
        # that the (block-based) manager holds an extension-backed block, and that the manager's array
        # is the very same object. Then wraps the series in another series and asserts dtype and
        # block type are preserved.
        result = pd.Series(data, copy=False)
        assert result.dtype == data.dtype
        assert len(result) == len(data)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert result._mgr.array is data

        # Series[EA] is unboxed / boxed correctly
        result2 = pd.Series(result)
        assert result2.dtype == data.dtype
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result2._mgr.blocks[0], EABackedBlock)

    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        # [FUNCTION DOC]
        # Series built with only an index and the extension dtype must be filled with the missing
        # value (three-element index case); with an empty index it must equal an empty series with an
        # object index and that dtype (regression for an upstream issue on empty index).
        result = pd.Series(index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

        # GH 33559 - empty index
        result = pd.Series(index=[], dtype=dtype)
        expected = pd.Series([], index=pd.Index([], dtype="object"), dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        # [FUNCTION DOC]
        # A scalar missing value with a three-element index and the extension dtype produces a series
        # of that missing value repeated.
        result = pd.Series(na_value, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_series_constructor_scalar_with_index(self, data, dtype):
        # [FUNCTION DOC]
        # A scalar taken from the data, broadcast over a three-element index and over a one-element
        # string index with the extension dtype, equals a series built from a list of repeated scalars.
        scalar = data[0]
        result = pd.Series(scalar, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([scalar] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = pd.Series(scalar, index=["foo"], dtype=dtype)
        expected = pd.Series([scalar], index=["foo"], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("from_series", [True, False])
    def test_dataframe_constructor_from_dict(self, data, from_series):
        # [FUNCTION DOC]
        # Parametrized over whether the dict value is wrapped in a series first. Builds a one-column
        # frame and asserts the column dtype, shape of length by one, and that the block (when
        # block based) is extension-backed and its values are an extension array.
        if from_series:
            data = pd.Series(data)
        result = pd.DataFrame({"A": data})
        assert result.dtypes["A"] == data.dtype
        assert result.shape == (len(data), 1)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert isinstance(result._mgr.blocks[0].values, ExtensionArray)

    def test_dataframe_from_series(self, data):
        # [FUNCTION DOC]
        # Builds a frame from a series of the data and asserts the first column's dtype, shape
        # (length by one) and extension-backed block with extension array values.
        result = pd.DataFrame(pd.Series(data))
        assert result.dtypes[0] == data.dtype
        assert result.shape == (len(data), 1)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert isinstance(result._mgr.blocks[0].values, ExtensionArray)

    def test_series_given_mismatched_index_raises(self, data):
        # [FUNCTION DOC]
        # Building a series from three values with a five-element index must raise a value error whose
        # message says the length of values (3) does not match the length of index (5).
        msg = r"Length of values \(3\) does not match length of index \(5\)"
        with pytest.raises(ValueError, match=msg):
            pd.Series(data[:3], index=[0, 1, 2, 3, 4])

    def test_from_dtype(self, data):
        # [FUNCTION DOC]
        # Series and frames built from a list of the elements, with the dtype given either as an
        # object or as its string name, must equal the ones built from the array (series) or cast to the
        # dtype (frame; regression for an upstream issue).
        # construct from our dtype & string dtype
        dtype = data.dtype

        expected = pd.Series(data)
        result = pd.Series(list(data), dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = pd.Series(list(data), dtype=str(dtype))
        tm.assert_series_equal(result, expected)

        # gh-30280

        expected = pd.DataFrame(data).astype(dtype)
        result = pd.DataFrame(list(data), dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = pd.DataFrame(list(data), dtype=str(dtype))
        tm.assert_frame_equal(result, expected)

    def test_pandas_array(self, data):
        # [FUNCTION DOC]
        # The public array constructor is idempotent: applying it to an extension array returns an
        # equal array.
        # pd.array(extension_array) should be idempotent...
        result = pd.array(data)
        tm.assert_extension_array_equal(result, data)

    def test_pandas_array_dtype(self, data):
        # [FUNCTION DOC]
        # Specifying an object numpy dtype overrides idempotency: the result equals a numpy-wrapping
        # extension array of the data converted to object.
        # ... but specifying dtype will override idempotency
        result = pd.array(data, dtype=np.dtype(object))
        expected = pd.arrays.NumpyExtensionArray(np.asarray(data, dtype=object))
        tm.assert_equal(result, expected)

    def test_construct_empty_dataframe(self, dtype):
        # [FUNCTION DOC]
        # An empty frame with one named column and the extension dtype equals a frame whose column is
        # an empty array of that dtype with a zero-length range index (regression for an upstream
        # issue).
        # GH 33623
        result = pd.DataFrame(columns=["a"], dtype=dtype)
        expected = pd.DataFrame(
            {"a": pd.array([], dtype=dtype)}, index=pd.RangeIndex(0)
        )
        tm.assert_frame_equal(result, expected)

    def test_empty(self, dtype):
        # [FUNCTION DOC]
        # Asserts the array class's empty-allocation constructor given a shape of one element (four)
        # returns an instance of the class with the right dtype and shape. Also checks the dtype's
        # empty method with a tuple shape and with a plain integer shape give the same type, dtype and
        # shape.
        cls = dtype.construct_array_type()
        result = cls._empty((4,), dtype=dtype)
        assert isinstance(result, cls)
        assert result.dtype == dtype
        assert result.shape == (4,)

        # GH#19600 method on ExtensionDtype
        result2 = dtype.empty((4,))
        assert isinstance(result2, cls)
        assert result2.dtype == dtype
        assert result2.shape == (4,)

        result2 = dtype.empty(4)
        assert isinstance(result2, cls)
        assert result2.dtype == dtype
        assert result2.shape == (4,)
