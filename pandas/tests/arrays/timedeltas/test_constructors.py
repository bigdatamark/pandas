# [MODULE DOC]
# Tests for the TimedeltaArray private sequence constructor: invalid input and
# dtype errors, and copy behavior. No import-time side effects.

import numpy as np
import pytest

from pandas.core.arrays import TimedeltaArray


class TestTimedeltaArrayConstructor:
    # [CLASS DOC]
    # Groups tests of the TimedeltaArray private sequence constructor.
    def test_other_type_raises(self):
        # [FUNCTION DOC]
        # Verifies constructing from a boolean numpy array raises a type error saying the
        # dtype cannot be converted to timedelta64 nanoseconds.
        msg = r"dtype bool cannot be converted to timedelta64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            TimedeltaArray._from_sequence(np.array([1, 2, 3], dtype="bool"))

    def test_incorrect_dtype_raises(self):
        # [FUNCTION DOC]
        # Verifies passing a dtype that is not a timedelta64 (category, int64, datetime64
        # naive or timezone-aware) raises a value error saying it should be a timedelta64
        # dtype, and an unsupported resolution (year) raises a value error listing the
        # supported resolutions.
        msg = "dtype 'category' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype="category"
            )

        msg = "dtype 'int64' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("int64")
            )

        msg = r"dtype 'datetime64\[ns\]' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("M8[ns]")
            )

        msg = (
            r"dtype 'datetime64\[us, UTC\]' is invalid, should be np.timedelta64 dtype"
        )
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype="M8[us, UTC]"
            )

        msg = "Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("m8[Y]")
            )

    def test_copy(self):
        # [FUNCTION DOC]
        # Verifies copy false keeps the very same numpy array while copy true creates a
        # new array that is not a view of the input.
        data = np.array([1, 2, 3], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(data, copy=False)
        assert arr._ndarray is data

        arr = TimedeltaArray._from_sequence(data, copy=True)
        assert arr._ndarray is not data
        assert arr._ndarray.base is not data

    def test_from_sequence_dtype(self):
        # [FUNCTION DOC]
        # Verifies passing object as dtype with an empty list raises a value error that it
        # should be a timedelta64 dtype.
        msg = "dtype 'object' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence([], dtype=object)
