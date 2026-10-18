"""
Tests for subclasses of NDArrayBackedExtensionArray
"""
# [MODULE DOC]
# Tests for the private empty-array constructor shared by subclasses of the
# numpy-backed extension array base (Categorical, DatetimeArray, TimedeltaArray
# and NumpyExtensionArray). No import-time side effects.

import numpy as np

from pandas import (
    CategoricalIndex,
    date_range,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    NumpyExtensionArray,
    TimedeltaArray,
)


class TestEmpty:
    # [CLASS DOC]
    # Groups tests of the private empty constructor for the numpy-backed array types.
    def test_empty_categorical(self):
        # [FUNCTION DOC]
        # Verifies Categorical empty creation: the result is a Categorical of the
        # requested shape whose integer codes use int8 for few categories (including a
        # large 4096-length case whose repr must not crash) and int16 for 512 categories.
        ci = CategoricalIndex(["a", "b", "c"], ordered=True)
        dtype = ci.dtype

        # case with int8 codes
        shape = (4,)
        result = Categorical._empty(shape, dtype=dtype)
        assert isinstance(result, Categorical)
        assert result.shape == shape
        assert result._ndarray.dtype == np.int8

        # case where repr would segfault if we didn't override base implementation
        result = Categorical._empty((4096,), dtype=dtype)
        assert isinstance(result, Categorical)
        assert result.shape == (4096,)
        assert result._ndarray.dtype == np.int8
        repr(result)

        # case with int16 codes
        ci = CategoricalIndex(list(range(512)) * 4, ordered=False)
        dtype = ci.dtype
        result = Categorical._empty(shape, dtype=dtype)
        assert isinstance(result, Categorical)
        assert result.shape == shape
        assert result._ndarray.dtype == np.int16

    def test_empty_dt64tz(self):
        # [FUNCTION DOC]
        # Verifies empty creation of a timezone-aware DatetimeArray of shape zero keeps
        # the dtype and class.
        dti = date_range("2016-01-01", periods=2, tz="Asia/Tokyo")
        dtype = dti.dtype

        shape = (0,)
        result = DatetimeArray._empty(shape, dtype=dtype)
        assert result.dtype == dtype
        assert isinstance(result, DatetimeArray)
        assert result.shape == shape

    def test_empty_dt64(self):
        # [FUNCTION DOC]
        # Verifies empty creation of a 2-d tz-naive nanosecond DatetimeArray has the
        # requested shape and class.
        shape = (3, 9)
        result = DatetimeArray._empty(shape, dtype="datetime64[ns]")
        assert isinstance(result, DatetimeArray)
        assert result.shape == shape

    def test_empty_td64(self):
        # [FUNCTION DOC]
        # Verifies empty creation of a 2-d timedelta nanosecond TimedeltaArray has the
        # requested shape and class.
        shape = (3, 9)
        result = TimedeltaArray._empty(shape, dtype="m8[ns]")
        assert isinstance(result, TimedeltaArray)
        assert result.shape == shape

    def test_empty_pandas_array(self):
        # [FUNCTION DOC]
        # Verifies empty creation of a 2-d NumpyExtensionArray keeps the dtype, class and shape.
        arr = NumpyExtensionArray(np.array([1, 2]))
        dtype = arr.dtype

        shape = (3, 9)
        result = NumpyExtensionArray._empty(shape, dtype=dtype)
        assert isinstance(result, NumpyExtensionArray)
        assert result.dtype == dtype
        assert result.shape == shape
