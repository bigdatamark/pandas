# [MODULE DOC]
# Base tests for converting to and from extension dtypes: astype to object and string
# dtypes, list and numpy conversion, and astype to the same dtype. Public contents:
# BaseCastingTests mixin. Uses the all_data, data and dtype fixtures from the conftest.
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock


class BaseCastingTests:
    # [CLASS DOC]
    # Mixin with casting tests for an extension array.
    """Casting to and from ExtensionDtypes"""

    def test_astype_object_series(self, all_data):
        # [FUNCTION DOC]
        # Casts a named series built from every data fixture variant to object dtype. Asserts the
        # result dtype is object and, when the manager is block based, that the single block is a plain
        # numpy block flagged as object; asserts the manager's array is an object numpy array.
        ser = pd.Series(all_data, name="A")
        result = ser.astype(object)
        assert result.dtype == np.dtype(object)
        if hasattr(result._mgr, "blocks"):
            blk = result._mgr.blocks[0]
            assert isinstance(blk, NumpyBlock)
            assert blk.is_object
        assert isinstance(result._mgr.array, np.ndarray)
        assert result._mgr.array.dtype == np.dtype(object)

    def test_astype_object_frame(self, all_data):
        # [FUNCTION DOC]
        # Same as the series test but for a one-column frame cast to object. Asserts the block holds
        # an object numpy array, and that comparing the new dtypes with the old ones yields no
        # equal entries (the dtype changed).
        df = pd.DataFrame({"A": all_data})

        result = df.astype(object)
        if hasattr(result._mgr, "blocks"):
            blk = result._mgr.blocks[0]
            assert isinstance(blk, NumpyBlock), type(blk)
            assert blk.is_object
        arr = result._mgr.blocks[0].values
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.dtype(object)

        # check that we can compare the dtypes
        comp = result.dtypes == df.dtypes
        assert not comp.any()

    def test_tolist(self, data):
        # [FUNCTION DOC]
        # Asserts converting a series of the data to a list equals the list built directly from the
        # array.
        result = pd.Series(data).tolist()
        expected = list(data)
        assert result == expected

    def test_astype_str(self, data):
        # [FUNCTION DOC]
        # Casts a series of the first two elements to the str type and asserts it equals a series of
        # the elements' string forms with the string dtype.
        result = pd.Series(data[:2]).astype(str)
        expected = pd.Series([str(x) for x in data[:2]], dtype=str)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "nullable_string_dtype",
        [
            "string[python]",
            pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_astype_string(self, data, nullable_string_dtype):
        # [FUNCTION DOC]
        # Parametrized over the python-backed and (if pyarrow is installed) pyarrow-backed nullable
        # string dtypes. Casts a series of the first five elements to that dtype and asserts it equals
        # a series of the expected strings. Expected conversion: bytes are decoded to text, the dtype's
        # missing marker is kept as is, anything else uses its string form.
        # GH-33465, GH#45326 as of 2.0 we decode bytes instead of calling str(obj)
        def as_str(x):
            # [FUNCTION DOC]
            # HELPER: computes the expected string for one element in test_astype_string. Decodes bytes,
            # returns the missing marker unchanged when the value is the dtype's missing marker (identity
            # check), otherwise returns the string form.
            if isinstance(x, bytes):
                return x.decode()
            elif x is data.dtype.na_value:
                return x
            else:
                return str(x)

        result = pd.Series(data[:5]).astype(nullable_string_dtype)
        expected = pd.Series(
            [as_str(x) for x in data[:5]],
            dtype=nullable_string_dtype,
        )
        tm.assert_series_equal(result, expected)

    def test_to_numpy(self, data):
        # [FUNCTION DOC]
        # Asserts converting the array to numpy equals the plain numpy conversion, and that converting
        # a series of the data gives the same result.
        expected = np.asarray(data)

        result = data.to_numpy()
        tm.assert_equal(result, expected)

        result = pd.Series(data).to_numpy()
        tm.assert_equal(result, expected)

    def test_astype_empty_dataframe(self, dtype):
        # [FUNCTION DOC]
        # Regression for an upstream issue: casting an empty frame to the extension dtype succeeds and
        # returns a frame equal to the original empty frame.
        # https://github.com/pandas-dev/pandas/issues/33113
        df = pd.DataFrame()
        result = df.astype(dtype)
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("copy", [True, False])
    def test_astype_own_type(self, data, copy):
        # [FUNCTION DOC]
        # Parametrized over copy true and false. Casts the array to its own dtype with that copy
        # flag and asserts identity with the original array holds exactly when copy is false, and that
        # the result equals the original.
        # ensure that astype returns the original object for equal dtype and copy=False
        # https://github.com/pandas-dev/pandas/issues/28488
        result = data.astype(data.dtype, copy=copy)
        assert (result is data) is (not copy)
        tm.assert_extension_array_equal(result, data)
