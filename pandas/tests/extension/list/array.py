"""
Test extension array for storing nested data in a pandas container.

The ListArray stores an ndarray of lists.
"""
# [MODULE DOC]
# Test-only extension array that stores one Python list per element in an object numpy array.
# Public contents: ListDtype, ListArray and make_data (random list generator). Missing values
# are NaN. Used by the list test module and as a nested-data fixture. No import-time side
# effects.

from __future__ import annotations

import numbers
import string
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.arrays import ExtensionArray

if TYPE_CHECKING:
    from pandas._typing import type_t


class ListDtype(ExtensionDtype):
    # [CLASS DOC]
    # Dtype for the list test array. Scalar type is list, name is the string list, and the
    # missing-value marker is NaN. Stateless.
    type = list
    name = "list"
    na_value = np.nan

    def construct_array_type(self) -> type_t[ListArray]:
        # [FUNCTION DOC]
        # Returns the array class associated with this dtype, which is ListArray.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return ListArray


class ListArray(ExtensionArray):
    # [CLASS DOC]
    # Extension array holding lists (or NaN for missing) in an object numpy array.
    # State: data, the object numpy array. Subclasses the base extension array and implements
    # construction, item access, length, missing detection, take, copy, astype and concatenation.
    # Invariant: every element is a list or a missing value.
    dtype = ListDtype()
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False) -> None:
        # [FUNCTION DOC]
        # Builds the array from values. Inputs: values (must be a numpy array), dtype (ignored),
        # copy (ignored). Raises a type error if values is not a numpy array, or if any element is
        # neither a list nor a missing value. Stores values without copying.
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array as values")
        for val in values:
            if not isinstance(val, self.dtype.type) and not pd.isna(val):
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # [FUNCTION DOC]
        # HELPER: alternate constructor used by the extension array machinery.
        # Creates an object numpy array of the sequence's length, assigns the scalars into it
        # elementwise, and wraps it. dtype and copy are ignored.
        data = np.empty(len(scalars), dtype=object)
        data[:] = scalars
        return cls(data)

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Item access. An integer index returns the stored list (or missing value). Any other
        # indexer is applied to the underlying data and wrapped in a new array.
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            # slice, list-like, mask
            return type(self)(self.data[item])

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of elements.
        return len(self.data)

    def isna(self):
        # [FUNCTION DOC]
        # Returns a boolean numpy array, true for elements that are not lists and are NaN.
        return np.array(
            [not isinstance(x, list) and np.isnan(x) for x in self.data], dtype=bool
        )

    def take(self, indexer, allow_fill=False, fill_value=None):
        # [FUNCTION DOC]
        # Positional selection implemented by hand because numpy has trouble storing sized objects
        # in object array slots. Inputs: indexer (integer positions), allow_fill (default false),
        # fill_value (default none, replaced by the dtype's NaN when filling).
        # With allow_fill, -1 selects the fill value and any position below -1 raises a value error;
        # an out of range position raises an index error with a descriptive message. Without
        # allow_fill, positions use normal negative-index semantics and out of range raises the same
        # index error. Returns a new array built from the selected elements.
        # re-implement here, since NumPy has trouble setting
        # sized objects like UserDicts into scalar slots of
        # an ndarary.
        indexer = np.asarray(indexer)
        msg = (
            "Index is out of bounds or cannot do a non-empty take from an empty array."
        )

        if allow_fill:
            if fill_value is None:
                fill_value = self.dtype.na_value
            # bounds check
            if (indexer < -1).any():
                raise ValueError
            try:
                output = [
                    self.data[loc] if loc != -1 else fill_value for loc in indexer
                ]
            except IndexError as err:
                raise IndexError(msg) from err
        else:
            try:
                output = [self.data[loc] for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err

        return self._from_sequence(output)

    def copy(self):
        # [FUNCTION DOC]
        # Returns a new array over a shallow copy of the data (the lists are shared, not copied).
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        # [FUNCTION DOC]
        # Conversion. Inputs: dtype, copy (default true). Same dtype: returns a copy if copy is
        # true, otherwise self. String (non-object) dtype: returns a numpy array of the string
        # form of each element, since numpy cannot convert nested elements itself. Otherwise returns a
        # numpy array of the data converted to dtype (copying only when copy is true).
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif is_string_dtype(dtype) and not is_object_dtype(dtype):
            # numpy has problems with astype(str) for nested elements
            return np.array([str(x) for x in self.data], dtype=dtype)
        elif not copy:
            return np.asarray(self.data, dtype=dtype)
        else:
            return np.array(self.data, dtype=dtype, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # [FUNCTION DOC]
        # HELPER: concatenates a sequence of ListArray objects by joining their data end to end.
        data = np.concatenate([x.data for x in to_concat])
        return cls(data)


def make_data(n: int):
    # [FUNCTION DOC]
    # HELPER: generates test data. Input n, the length. Returns an object numpy array of n lists,
    # each with a random length from zero to nine and containing random ASCII letters, drawn from a
    # generator with a fixed seed so output is deterministic.
    # KNOWN ISSUE: a leftover TODO comment about using a regular dict refers to unrelated indexer code
    # and does not describe this function.
    # TODO: Use a regular dict. See _NDFrameIndexer._setitem_with_indexer
    rng = np.random.default_rng(2)
    data = np.empty(n, dtype=object)
    data[:] = [
        [rng.choice(list(string.ascii_letters)) for _ in range(rng.integers(0, 10))]
        for _ in range(n)
    ]
    return data
