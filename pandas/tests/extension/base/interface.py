# [MODULE DOC]
# Base tests that an extension array satisfies the basic array interface: length, size,
# dimensionality, membership, memory usage, numpy conversion, copy and view semantics, and
# list conversion. Public contents: BaseInterfaceTests mixin. Uses data, data_missing and the
# using_nan_is_na fixture.
import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype

import pandas as pd
import pandas._testing as tm


class BaseInterfaceTests:
    # [CLASS DOC]
    # Mixin with interface-conformance tests. The data fixture is expected to be a one
    # dimensional array of length ten.
    """Tests that the basic interface is satisfied."""

    # ------------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------------

    def test_len(self, data):
        # [FUNCTION DOC]
        # Asserts the array length is ten.
        assert len(data) == 10

    def test_size(self, data):
        # [FUNCTION DOC]
        # Asserts the size attribute is ten.
        assert data.size == 10

    def test_ndim(self, data):
        # [FUNCTION DOC]
        # Asserts the array is one-dimensional.
        assert data.ndim == 1

    def test_can_hold_na_valid(self, data):
        # [FUNCTION DOC]
        # Asserts the private can-hold-missing flag is exactly true (regression for an upstream
        # issue).
        # GH-20761
        assert data._can_hold_na is True

    def test_contains(self, data, data_missing, using_nan_is_na):
        # [FUNCTION DOC]
        # Tests membership semantics. The rule: a missing-like value is a member only if it is the
        # dtype's missing value and the array has at least one missing element. Setup: first the
        # fixture data is filtered to its non-missing elements. Asserts: first elements are members of
        # their arrays; the dtype's missing value is in data_missing but not in the clean data; no other
        # missing-like object (from the shared list of null objects, skipping the dtype's own missing
        # value or objects of the same type) is in the clean data. For data_missing, other missing-like
        # objects are not members, except that when nan counts as missing and the object is a float
        # and the array is masked or arrow-backed, they are members (an override the code itself labels
        # as being in the wrong place).
        # GH-37867
        # Tests for membership checks. Membership checks for nan-likes is tricky and
        # the settled on rule is: `nan_like in arr` is True if nan_like is
        # arr.dtype.na_value and arr.isna().any() is True. Else the check returns False.

        na_value = data.dtype.na_value
        # ensure data without missing values
        data = data[~data.isna()]

        # first elements are non-missing
        assert data[0] in data
        assert data_missing[0] in data_missing

        # check the presence of na_value
        assert na_value in data_missing
        assert na_value not in data

        # the data can never contain other nan-likes than na_value
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value or type(na_value_obj) == type(na_value):
                # type check for e.g. two instances of Decimal("NAN")
                continue
            assert na_value_obj not in data
            if (
                using_nan_is_na
                and isinstance(na_value_obj, float)
                and isinstance(
                    data,
                    (
                        pd.core.arrays.BaseMaskedArray,
                        pd.core.arrays.ArrowExtensionArray,
                    ),
                )
            ):
                # TODO: wrong place for this override
                assert na_value_obj in data_missing
            else:
                assert na_value_obj not in data_missing

    def test_memory_usage(self, data):
        # [FUNCTION DOC]
        # Asserts the series memory usage excluding the index equals the series' byte size.
        s = pd.Series(data)
        result = s.memory_usage(index=False)
        assert result == s.nbytes

    def test_array_interface(self, data):
        # [FUNCTION DOC]
        # Converting to numpy gives an array whose first element equals the first array element.
        # Converting with the object dtype gives an object array equal to one built from the list of
        # elements; if the list build would be multidimensional (nested data) the expected array is
        # explicitly built one-dimensional.
        result = np.array(data)
        assert result[0] == data[0]

        result = np.array(data, dtype=object)
        expected = np.array(list(data), dtype=object)
        if expected.ndim > 1:
            # nested data, explicitly construct as 1D
            expected = construct_1d_object_array_from_listlike(list(data))
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface_copy(self, data):
        # [FUNCTION DOC]
        # Two numpy conversions with copy true must not share memory. On older numpy (before version
        # 2) nothing more is checked. On numpy 2 or newer, a conversion with copy false may raise a
        # value error (acceptable, test ends); if it does not raise, two such conversions must share
        # memory.
        result_copy1 = np.array(data, copy=True)
        result_copy2 = np.array(data, copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)

        if not np_version_gt2:
            # copy=False semantics are only supported in NumPy>=2.
            return

        try:
            result_nocopy1 = np.array(data, copy=False)
        except ValueError:
            # An error is always acceptable for `copy=False`
            return

        result_nocopy2 = np.array(data, copy=False)
        # If copy=False was given and did not raise, these must share the same data
        assert np.may_share_memory(result_nocopy1, result_nocopy2)

    def test_is_extension_array_dtype(self, data):
        # [FUNCTION DOC]
        # Asserts the generic extension-dtype predicate is true for the array, its dtype and a series
        # of it, and that the dtype is an instance of the extension dtype base class.
        assert is_extension_array_dtype(data)
        assert is_extension_array_dtype(data.dtype)
        assert is_extension_array_dtype(pd.Series(data))
        assert isinstance(data.dtype, ExtensionDtype)

    def test_no_values_attribute(self, data):
        # [FUNCTION DOC]
        # Asserts the array has neither a public nor a private values attribute (such attributes
        # confuse internal code; regression for an upstream issue).
        # GH-20735: EA's with .values attribute give problems with internal
        # code, disallowing this for now until solved
        assert not hasattr(data, "values")
        assert not hasattr(data, "_values")

    def test_is_numeric_honored(self, data):
        # [FUNCTION DOC]
        # For block-based managers, asserts the series' block reports numeric-ness equal to the
        # dtype's private numeric flag.
        result = pd.Series(data)
        if hasattr(result._mgr, "blocks"):
            assert result._mgr.blocks[0].is_numeric is data.dtype._is_numeric

    def test_isna_extension_array(self, data_missing):
        # [FUNCTION DOC]
        # If the missing-value mask is itself an extension array then it must support the reduction
        # protocol: any is true, all is false (both via the private reduce and the public methods) and
        # its dtype is boolean. Nothing is checked when the mask is a plain numpy array.
        # If your `isna` returns an ExtensionArray, you must also implement
        # _reduce. At the *very* least, you must implement any and all
        na = data_missing.isna()
        if is_extension_array_dtype(na):
            assert na._reduce("any")
            assert na.any()

            assert not na._reduce("all")
            assert not na.all()

            assert na.dtype._is_boolean

    def test_copy(self, data):
        # [FUNCTION DOC]
        # Copy must be independent. Requires the first two elements to differ. Takes a copy, skips
        # if the dtype is immutable, then mutates the original so element one equals element zero and
        # asserts the copy's two elements still differ (mutates the data fixture).
        # GH#27083 removing deep keyword from EA.copy
        assert data[0] != data[1]
        result = data.copy()

        if data.dtype._is_immutable:
            pytest.skip(f"test_copy assumes mutability and {data.dtype} is immutable")

        data[1] = data[0]
        assert result[1] != result[0]

    def test_view(self, data):
        # [FUNCTION DOC]
        # A view with no dtype returns a new object of the same class (a shallow copy). Skips if
        # immutable. Mutating the view changes the original (shared data). The dtype keyword argument
        # set to none must be accepted.
        # view with no dtype should return a shallow copy, *not* the same
        #  object
        assert data[1] != data[0]

        result = data.view()
        assert result is not data
        assert type(result) == type(data)

        if data.dtype._is_immutable:
            pytest.skip(f"test_view assumes mutability and {data.dtype} is immutable")

        result[1] = result[0]
        assert data[1] == data[0]

        # check specifically that the `dtype` kwarg is accepted
        data.view(dtype=None)

    def test_tolist(self, data):
        # [FUNCTION DOC]
        # Asserts the list conversion returns a list equal to the list of elements.
        result = data.tolist()
        expected = list(data)
        assert isinstance(result, list)
        assert result == expected
