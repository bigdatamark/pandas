# [MODULE DOC]
# Tests for the extension-array dtype utilities in the common dtype helpers: checking
# whether values count as extension array dtypes, plus astype behavior for a minimal
# custom array. Defines a dummy dtype, a dummy array and an array capturing the
# argument passed to item lookup. No import-time side effects.
import numpy as np
import pytest

from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class DummyDtype(dtypes.ExtensionDtype):
    # [CLASS DOC]
    # HELPER: minimal extension dtype used by the tests in this module. Subclasses the
    # extension dtype base class with no overrides; exists only to be recognized as an
    # extension dtype.
    pass


class DummyArray(ExtensionArray):
    # [CLASS DOC]
    # HELPER: minimal extension array wrapping an arbitrary array-like stored on the
    # data attribute. Provides conversion to a numpy array, a dtype property returning
    # the dummy dtype, and a restricted astype. Other abstract methods are not implemented.
    def __init__(self, data) -> None:
        # [FUNCTION DOC]
        # Stores the given data object on the data attribute without copying or validation.
        self.data = data

    def __array__(self, dtype=None, copy=None):
        # [FUNCTION DOC]
        # Converts to a numpy array by returning the stored data directly. The dtype and copy
        # arguments (both default none) are ignored.
        return self.data

    @property
    def dtype(self):
        # [FUNCTION DOC]
        # Property returning a new instance of the dummy dtype.
        return DummyDtype()

    def astype(self, dtype, copy=True):
        # [FUNCTION DOC]
        # Casts the array. Parameters: dtype is the target dtype; copy (default true).
        # If dtype is the dummy dtype: returns a new array of the same class built from the
        # data when copy is true, otherwise returns self. For other dtypes: when copy is
        # false returns a numpy array conversion with the dtype (no explicit copy), otherwise
        # returns a numpy array created with the dtype and the copy flag. Only the dummy dtype
        # is supported as an extension target.
        # we don't support anything but a single dtype
        if isinstance(dtype, DummyDtype):
            if copy:
                return type(self)(self.data)
            return self
        elif not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)


class TestExtensionArrayDtype:
    # [CLASS DOC]
    # Test class for the check that identifies extension array dtypes.
    @pytest.mark.parametrize(
        "values",
        [
            pd.Categorical([]),
            pd.Categorical([]).dtype,
            pd.Series(pd.Categorical([])),
            DummyDtype(),
            DummyArray(np.array([1, 2])),
        ],
    )
    def test_is_extension_array_dtype(self, values):
        # [FUNCTION DOC]
        # Parametrized over a categorical, its dtype, a series of categoricals, the dummy dtype
        # and a dummy array. Verifies each is reported as an extension array dtype.
        assert is_extension_array_dtype(values)

    @pytest.mark.parametrize("values", [np.array([]), pd.Series(np.array([]))])
    def test_is_not_extension_array_dtype(self, values):
        # [FUNCTION DOC]
        # Parametrized over an empty numpy array and an empty series of numpy values. Verifies
        # neither is reported as an extension array dtype.
        assert not is_extension_array_dtype(values)


def test_astype():
    # [FUNCTION DOC]
    # Verifies casting the dummy array to object, both as the type and as the string name,
    # returns a numpy object array equal to the original data.
    arr = DummyArray(np.array([1, 2, 3]))
    expected = np.array([1, 2, 3], dtype=object)

    result = arr.astype(object)
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("object")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_no_copy():
    # [FUNCTION DOC]
    # Verifies that casting the dummy array to its own dtype with copy disabled returns the
    # same object, and with the default copy returns a different object.
    arr = DummyArray(np.array([1, 2, 3], dtype=np.int64))
    result = arr.astype(arr.dtype, copy=False)

    assert arr is result

    result = arr.astype(arr.dtype)
    assert arr is not result


@pytest.mark.parametrize("dtype", [dtypes.CategoricalDtype(), dtypes.IntervalDtype()])
def test_is_extension_array_dtype(dtype):
    # [FUNCTION DOC]
    # Parametrized over the categorical and interval dtype instances. Verifies each is an
    # extension dtype instance and is reported as an extension array dtype.
    assert isinstance(dtype, dtypes.ExtensionDtype)
    assert is_extension_array_dtype(dtype)


class CapturingStringArray(pd.arrays.StringArray):
    # [CLASS DOC]
    # HELPER: string array subclass that records the argument most recently passed to
    # item lookup on an attribute called last_item_arg, then delegates to the parent
    # implementation. Used to inspect what indexer reaches the array.
    """Extend StringArray to capture arguments to __getitem__"""

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Records the item argument on the last_item_arg attribute (side effect), then returns
        # the parent class lookup result for the same item.
        self.last_item_arg = item
        return super().__getitem__(item)


def test_ellipsis_index():
    # [FUNCTION DOC]
    # Regression test for GH#42430: slicing rows of a frame holding an extension column
    # must pass a plain one-dimensional slice to the array, not a two-dimensional
    # (ellipsis plus slice) indexer. Compares the string form of the captured argument
    # because slices cannot be compared natively.
    # GH#42430 1D slices over extension types turn into N-dimensional slices
    #  over ExtensionArrays
    dtype = pd.StringDtype()
    df = pd.DataFrame(
        {
            "col1": CapturingStringArray(
                np.array(["hello", "world"], dtype=object), dtype=dtype
            )
        }
    )
    _ = df.iloc[:1]

    # String comparison because there's no native way to compare slices.
    # Before the fix for GH#42430, last_item_arg would get set to the 2D slice
    # (Ellipsis, slice(None, 1, None))
    out = df["col1"]._values.last_item_arg
    assert str(out) == "slice(None, 1, None)"
