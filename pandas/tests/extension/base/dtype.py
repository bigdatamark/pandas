# [MODULE DOC]
# Base tests for the extension dtype object itself: name, kind, equality, hashing, string
# construction and common-dtype resolution. Public contents: BaseDtypeTests mixin. Uses the
# dtype, data and data_missing fixtures.
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
    infer_dtype,
    is_object_dtype,
    is_string_dtype,
)


class BaseDtypeTests:
    # [CLASS DOC]
    # Mixin with tests that the dtype fixture satisfies the extension dtype contract.
    """Base class for ExtensionDtype classes"""

    def test_name(self, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype's name attribute is a string.
        assert isinstance(dtype.name, str)

    def test_kind(self, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype's kind character is one of the valid numpy kind codes (boolean, signed
        # and unsigned integer, float, complex, timedelta, datetime, object, bytes, text, void).
        valid = set("biufcmMOSUV")
        assert dtype.kind in valid

    def test_is_dtype_from_name(self, dtype):
        # [FUNCTION DOC]
        # Asserts the class-level dtype recognizer returns exactly true for the dtype's name string.
        result = type(dtype).is_dtype(dtype.name)
        assert result is True

    def test_is_dtype_unboxes_dtype(self, data, dtype):
        # [FUNCTION DOC]
        # Asserts the recognizer returns exactly true when given an array of the dtype (it
        # looks at the array's dtype).
        assert dtype.is_dtype(data) is True

    def test_is_dtype_from_self(self, dtype):
        # [FUNCTION DOC]
        # Asserts the class-level recognizer returns exactly true for a dtype instance.
        result = type(dtype).is_dtype(dtype)
        assert result is True

    def test_is_dtype_other_input(self, dtype):
        # [FUNCTION DOC]
        # Asserts the recognizer returns exactly false for an unrelated list of integers.
        assert dtype.is_dtype([1, 2, 3]) is False

    def test_is_not_string_type(self, dtype):
        # [FUNCTION DOC]
        # Asserts the generic string-dtype predicate is false for the dtype.
        assert not is_string_dtype(dtype)

    def test_is_not_object_type(self, dtype):
        # [FUNCTION DOC]
        # Asserts the generic object-dtype predicate is false for the dtype.
        assert not is_object_dtype(dtype)

    def test_eq_with_str(self, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype equals its own name and differs from the name with a suffix appended.
        assert dtype == dtype.name
        assert dtype != dtype.name + "-suffix"

    def test_eq_with_numpy_object(self, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype is not equal to the numpy object dtype.
        assert dtype != np.dtype("object")

    def test_eq_with_self(self, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype equals itself and differs from an arbitrary plain object.
        assert dtype == dtype
        assert dtype != object()

    def test_array_type(self, data, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype's array-class constructor returns exactly the class of the data fixture.
        assert dtype.construct_array_type() is type(data)

    def test_check_dtype(self, data):
        # [FUNCTION DOC]
        # Builds a frame with four columns: the array with explicit dtype, the bare array, an
        # object-dtype string column, and a scalar integer. Asserts that comparing the dtypes to the
        # string form of the extension dtype gives true, true, false, false, both directly and after
        # converting the dtypes to strings.
        # KNOWN ISSUE: the assertion comparing the numpy int64 dtype with the string Int64 is unrelated
        # to the extension dtype under test (leftover sanity check) and does not use any fixture.
        dtype = data.dtype

        # check equivalency for using .dtypes
        df = pd.DataFrame(
            {
                "A": pd.Series(data, dtype=dtype),
                "B": data,
                "C": pd.Series(["foo"] * len(data), dtype=object),
                "D": 1,
            }
        )
        result = df.dtypes == str(dtype)
        assert np.dtype("int64") != "Int64"

        expected = pd.Series([True, True, False, False], index=list("ABCD"))

        tm.assert_series_equal(result, expected)

        expected = pd.Series([True, True, False, False], index=list("ABCD"))
        result = df.dtypes.apply(str) == str(dtype)
        tm.assert_series_equal(result, expected)

    def test_hashable(self, dtype):
        # [FUNCTION DOC]
        # Asserts hashing the dtype does not raise.
        hash(dtype)  # no error

    def test_str(self, dtype):
        # [FUNCTION DOC]
        # Asserts the string form of the dtype equals its name.
        assert str(dtype) == dtype.name

    def test_eq(self, dtype):
        # [FUNCTION DOC]
        # Asserts the dtype equals its name and differs from an unrelated string.
        assert dtype == dtype.name
        assert dtype != "another_type"

    def test_construct_from_string_own_name(self, dtype):
        # [FUNCTION DOC]
        # Asserts constructing from the dtype's own name returns an instance of the same type, both
        # via the instance and via the class (so it must be callable as a class method).
        result = dtype.construct_from_string(dtype.name)
        assert type(result) is type(dtype)

        # check OK as classmethod
        result = type(dtype).construct_from_string(dtype.name)
        assert type(result) is type(dtype)

    def test_construct_from_string_another_type_raises(self, dtype):
        # [FUNCTION DOC]
        # Asserts constructing from an unrelated string raises a type error with a message saying it
        # cannot construct the dtype's class name from that string.
        msg = f"Cannot construct a '{type(dtype).__name__}' from 'another_type'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    def test_construct_from_string_wrong_type_raises(self, dtype):
        # [FUNCTION DOC]
        # Asserts constructing from a non-string (the integer zero) raises a type error stating that
        # the constructor expects a string and received the integer class.
        with pytest.raises(
            TypeError,
            match="'construct_from_string' expects a string, got <class 'int'>",
        ):
            type(dtype).construct_from_string(0)

    def test_get_common_dtype(self, dtype):
        # [FUNCTION DOC]
        # Asserts the common-dtype resolver, called with a single-element list holding the dtype,
        # returns that dtype. This is the only case testable in general.
        # in practice we will not typically call this with a 1-length list
        # (we shortcut to just use that dtype as the common dtype), but
        # still testing as good practice to have this working (and it is the
        # only case we can test in general)
        assert dtype._get_common_dtype([dtype]) == dtype

    @pytest.mark.parametrize("skipna", [True, False])
    def test_infer_dtype(self, data, data_missing, skipna):
        # [FUNCTION DOC]
        # Parametrized over skipna. Asserts the generic dtype inference function returns a string
        # (without raising) for both the data and the data_missing fixtures.
        # only testing that this works without raising an error
        res = infer_dtype(data, skipna=skipna)
        assert isinstance(res, str)
        res = infer_dtype(data_missing, skipna=skipna)
        assert isinstance(res, str)
