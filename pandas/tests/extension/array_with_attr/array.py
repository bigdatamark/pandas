"""
Test extension array that has custom attribute information (not stored on the dtype).

"""
# [MODULE DOC]
# Test-only extension array whose float64 data carries an extra user attribute.
# Public contents: FloatAttrDtype and FloatAttrArray. The attribute is stored on the array
# instance (not on the dtype), so it is only preserved by operations that explicitly pass it
# along (slicing, take, copy, concatenation). Used by tests that check attribute propagation
# through merge and concat. Relies on the base extension dtype and array classes. No import-time
# side effects.

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype

import pandas as pd
from pandas.core.arrays import ExtensionArray

if TYPE_CHECKING:
    from pandas._typing import type_t


class FloatAttrDtype(ExtensionDtype):
    # [CLASS DOC]
    # Dtype for the float-with-attribute test array. Scalar type is the builtin float,
    # name is the string float_attr, and the missing-value marker is NaN. Stateless.
    type = float
    name = "float_attr"
    na_value = np.nan

    def construct_array_type(self) -> type_t[FloatAttrArray]:
        # [FUNCTION DOC]
        # Returns the array class associated with this dtype, which is FloatAttrArray.
        # Takes no inputs and has no side effects.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FloatAttrArray


class FloatAttrArray(ExtensionArray):
    # [CLASS DOC]
    # Extension array wrapping a float64 numpy array plus a free-form attribute.
    # State: data (the numpy array of float64 values) and attr (arbitrary object, default none).
    # Subclasses the base extension array; implements construction, item access, length, missing
    # detection, take, copy and same-type concatenation. The class-level dtype is a shared dtype
    # instance, and the array priority is raised so numpy defers to this class in binary operations.
    # Invariant: data is always a float64 numpy array.
    dtype = FloatAttrDtype()
    __array_priority__ = 1000

    def __init__(self, values, attr=None) -> None:
        # [FUNCTION DOC]
        # Builds the array from values and an optional attribute.
        # Inputs: values, which must be a numpy array of float64 dtype; attr, any object, default none.
        # Raises a type error if values is not a numpy array or if its dtype is not float64.
        # Stores values without copying and stores attr as given.
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array of float64 dtype as values")
        if not values.dtype == "float64":
            raise TypeError("Need to pass a numpy array of float64 dtype as values")
        self.data = values
        self.attr = attr

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # [FUNCTION DOC]
        # HELPER: alternate constructor used by the extension array machinery.
        # Inputs: scalars (sequence convertible to float64), dtype (ignored), copy (default false).
        # Converts to a float64 numpy array, copying only when copy is true, and returns a new array
        # with no attribute set (attr is lost). Never raises beyond numpy conversion errors.
        if not copy:
            data = np.asarray(scalars, dtype="float64")
        else:
            data = np.array(scalars, dtype="float64", copy=copy)
        return cls(data)

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Item access. An integer index returns the scalar float from the data. Any other
        # indexer (slice, list-like, boolean mask) is validated with the array indexer checker and
        # returns a new array holding the selected data with the attribute carried over (same object).
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            # slice, list-like, mask
            item = pd.api.indexers.check_array_indexer(self, item)
            return type(self)(self.data[item], self.attr)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of elements in the underlying data array.
        return len(self.data)

    def isna(self):
        # [FUNCTION DOC]
        # Returns a boolean numpy array, true where the value is NaN.
        return np.isnan(self.data)

    def take(self, indexer, allow_fill=False, fill_value=None):
        # [FUNCTION DOC]
        # Positional selection. Inputs: indexer (integer positions), allow_fill (default false;
        # when true, -1 means missing), fill_value (default none, which becomes the dtype's NaN when
        # filling). Delegates to the generic extension take helper (imported lazily inside the method)
        # and returns a new array carrying over the attribute. Out-of-bounds handling is that of the
        # helper.
        from pandas.api.extensions import take

        data = self.data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return type(self)(result, self.attr)

    def copy(self):
        # [FUNCTION DOC]
        # Returns a new array with a copied data buffer and the same attribute object.
        return type(self)(self.data.copy(), self.attr)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # [FUNCTION DOC]
        # HELPER: concatenates a sequence of arrays of this type into one.
        # Data is joined end to end; the attribute of the first array is kept (none if the sequence is
        # empty, although concatenating zero arrays would fail earlier in numpy). Other attributes are
        # discarded.
        data = np.concatenate([x.data for x in to_concat])
        attr = to_concat[0].attr if len(to_concat) else None
        return cls(data, attr)
