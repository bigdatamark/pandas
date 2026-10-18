"""
Test extension array for storing nested data in a pandas container.

The JSONArray stores lists of dictionaries. The storage mechanism is a list,
not an ndarray.

Note
----
We currently store lists of UserDicts. Pandas has a few places
internally that specifically check for dicts, and does non-scalar things
in that case. We *want* the dictionaries to be treated as scalars, so we
hack around pandas by using UserDicts.
"""
# [MODULE DOC]
# Test-only extension array whose elements are JSON-like mappings, held in a plain Python list
# (not a numpy array). Public contents: JSONDtype, JSONArray and make_data (random data).
# Elements are stored as dict-wrapping user mapping objects so pandas treats them as scalars
# instead of expanding real dicts. No import-time side effects.

from __future__ import annotations

from collections import (
    UserDict,
    abc,
)
import itertools
import numbers
import string
import sys
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_list_like,
    pandas_dtype,
)

import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
from pandas.core.indexers import (
    getitem_returns_view,
    unpack_tuple_and_ellipses,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pandas._typing import type_t


class JSONDtype(ExtensionDtype):
    # [CLASS DOC]
    # Dtype for the JSON test array. Scalar type is the abstract mapping type, name is the string
    # json, missing-value marker is an empty user mapping. Stateless.
    type = abc.Mapping
    name = "json"
    na_value: Mapping[str, Any] = UserDict()

    def construct_array_type(self) -> type_t[JSONArray]:
        # [FUNCTION DOC]
        # Returns the array class for this dtype, JSONArray.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return JSONArray


class JSONArray(ExtensionArray):
    # [CLASS DOC]
    # Extension array of mappings. State: data, a list of mappings, with aliases _items and _data
    # pointing to the same list. Class attributes: the shared dtype instance and a raised array
    # priority. Subclasses the base extension array; implements construction, factorization hooks,
    # item access/assignment, length, array conversion, byte-size estimate, missing detection, take,
    # copy, astype, unique and concatenation. Equality operators deliberately return the
    # not-implemented sentinel.
    dtype = JSONDtype()
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False) -> None:
        # [FUNCTION DOC]
        # Builds the array. Inputs: values (iterable of mappings; held directly, not copied), dtype and
        # copy (both ignored). Raises a type error if any element is not a mapping. Sets the data
        # attribute and its aliases.
        for val in values:
            if not isinstance(val, self.dtype.type):
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values

        # Some aliases for common attribute names to ensure pandas supports
        # these
        self._items = self._data = self.data
        # those aliases are currently not working due to assumptions
        # in internal code (GH-20735)
        # self._values = self.values = self.data

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # [FUNCTION DOC]
        # HELPER: alternate constructor used by the extension array machinery; builds the array from
        # scalars, ignoring dtype and copy.
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        # [FUNCTION DOC]
        # HELPER: rebuilds an array from unique values produced by factorization (sequences of
        # key-value pairs). Wraps each non-empty value in a user mapping; empty tuples (the missing
        # placeholder) are dropped. The original array is ignored.
        return cls([UserDict(x) for x in values if x != ()])

    def _cast_pointwise_result(self, values):
        # [FUNCTION DOC]
        # HELPER: converts the result of a pointwise operation to a suitable array. Input: values.
        # Tries to build an array of this type; if that raises a value or type error, converts to an
        # object numpy array and applies the library's object-to-specific-type inference with non-numeric
        # conversion enabled.
        try:
            return type(self)._from_sequence(values, dtype=self.dtype)
        except (ValueError, TypeError):
            # TODO replace with public function
            from pandas._libs import lib

            values = np.asarray(values, dtype=object)
            return lib.maybe_convert_objects(values, convert_non_numeric=True)

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Item access. Tuples are first reduced by removing ellipses. An integer returns the stored
        # mapping. The full slice returns a new array over the same list (a view) with the read-only flag
        # copied. Other slices return a new array over the sliced list, with the read-only flag propagated
        # when the slice is a view. Non-list-like keys (such as strings or floats) raise an index error
        # with numpy's message about valid indexers. Otherwise the key is validated by the array
        # indexer checker: boolean masks select elements where true (the result is built through the
        # sequence constructor), and integer arrays select by position.
        if isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if isinstance(item, numbers.Integral):
            return self.data[item]
        elif isinstance(item, slice) and item == slice(None):
            # Make sure we get a view
            result = type(self)(self.data)
            result._readonly = self._readonly
            return result
        elif isinstance(item, slice):
            # slice
            result = type(self)(self.data[item])
            if getitem_returns_view(self, item):
                result._readonly = self._readonly
            return result
        elif not is_list_like(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        else:
            item = pd.api.indexers.check_array_indexer(self, item)
            if is_bool_dtype(item.dtype):
                return type(self)._from_sequence(
                    [x for x, m in zip(self, item, strict=True) if m], dtype=self.dtype
                )
            # integer
            return type(self)([self.data[i] for i in item])

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Assignment. Raises a value error if read-only. An integer key assigns the value directly
        # without type checking. Otherwise a value that is neither an array of this type nor a sequence
        # is broadcast by cycling it. For a boolean numpy mask, walks the key and value in parallel and
        # assigns where the key entry is true; for any other key, pairs each key with a value. Each
        # assigned value is asserted to be a mapping (an assertion error otherwise). Mutates in place.
        # KNOWN ISSUE: with a boolean mask the value sequence is advanced for every mask position, not just
        # the true ones, so a sequence value is paired with mask positions rather than consumed one value
        # per selected element as standard assignment semantics would require; iteration also stops at
        # the shorter of the two without a length check.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        if isinstance(key, numbers.Integral):
            self.data[key] = value
        else:
            if not isinstance(value, (type(self), abc.Sequence)):
                # broadcast value
                value = itertools.cycle([value])

            if isinstance(key, np.ndarray) and key.dtype == "bool":
                # masking
                for i, (k, v) in enumerate(zip(key, value, strict=False)):
                    if k:
                        assert isinstance(v, self.dtype.type)
                        self.data[i] = v
            else:
                for k, v in zip(key, value, strict=False):
                    assert isinstance(v, self.dtype.type)
                    self.data[k] = v

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of elements.
        return len(self.data)

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Always returns the not-implemented sentinel, deferring to the other operand.
        return NotImplemented

    def __ne__(self, other):
        # [FUNCTION DOC]
        # Always returns the not-implemented sentinel, deferring to the other operand.
        return NotImplemented

    def __array__(self, dtype=None, copy=None):
        # [FUNCTION DOC]
        # Numpy array protocol. Inputs: dtype (default none, meaning object), copy (default none).
        # A copy value of false raises a value error since a copy cannot be avoided. For object dtype,
        # builds a one-dimensional object array from the elements. For other dtypes, converts the data
        # list with numpy, passing the copy flag only when it is not none.
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        if dtype is None:
            dtype = object
        if dtype == object:
            # on py38 builds it looks like numpy is inferring to a non-1D array
            return construct_1d_object_array_from_listlike(list(self))
        if copy is None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            return np.asarray(self.data, dtype=dtype)
        return np.asarray(self.data, dtype=dtype, copy=copy)

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: the interpreter's size of the data list (not including elements).
        return sys.getsizeof(self.data)

    def isna(self):
        # [FUNCTION DOC]
        # Returns a boolean numpy array, true where the element equals the dtype's missing value (an
        # empty mapping).
        return np.array([x == self.dtype.na_value for x in self.data], dtype=bool)

    def take(self, indexer, allow_fill=False, fill_value=None):
        # [FUNCTION DOC]
        # Positional selection implemented by hand because numpy has trouble storing sized objects
        # in object array slots. Inputs: indexer, allow_fill (default false), fill_value (default none,
        # replaced by the dtype's missing value). With allow_fill, -1 selects the fill value and any
        # position below -1 raises a value error; out of range positions raise an index error with a
        # descriptive message. Without allow_fill, normal negative indexing applies. Returns a new array.
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

        return type(self)._from_sequence(output, dtype=self.dtype)

    def copy(self):
        # [FUNCTION DOC]
        # Returns a new array over a shallow copy of the data list (the mappings are shared).
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        # [FUNCTION DOC]
        # Conversion. Inputs: dtype, copy (default true). Same dtype: a copy if copy is true else self.
        # A string dtype: builds that dtype's array from the elements without copying. Otherwise
        # converts to a numpy array from plain dict copies of each mapping (numpy mishandles equal-length
        # user mappings), without a copy when copy is false.
        # NumPy has issues when all the dicts are the same length.
        # np.array([UserDict(...), UserDict(...)]) fails,
        # but np.array([{...}, {...}]) works, so cast.
        from pandas.core.arrays.string_ import StringDtype

        dtype = pandas_dtype(dtype)
        # needed to add this check for the Series constructor
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, StringDtype):
            arr_cls = dtype.construct_array_type()
            return arr_cls._from_sequence(self, dtype=dtype, copy=False)
        elif not copy:
            return np.asarray([dict(x) for x in self], dtype=dtype)
        else:
            return np.array([dict(x) for x in self], dtype=dtype, copy=copy)

    def unique(self):
        # [FUNCTION DOC]
        # Returns a new array of the distinct elements, found by turning each mapping into a tuple of
        # key-value pairs and deduplicating with a set; the base implementation fails because numpy
        # infers a 2D object array.
        # KNOWN ISSUE: because a set is used, the order of the result is arbitrary and not the order of
        # first occurrence that the generic unique contract implies; also mappings with the same items in
        # different insertion order are treated as different.
        # Parent method doesn't work since np.array will try to infer
        # a 2-dim object.
        return type(self)([dict(x) for x in {tuple(d.items()) for d in self.data}])

    @classmethod
    def _concat_same_type(cls, to_concat):
        # [FUNCTION DOC]
        # HELPER: concatenates a sequence of JSON arrays by chaining their data lists into a new list.
        data = list(itertools.chain.from_iterable(x.data for x in to_concat))
        return cls(data)

    def _values_for_factorize(self):
        # [FUNCTION DOC]
        # HELPER: factorization hook. Returns a tuple of a one-dimensional object array of tuples of
        # items (via the argsort hook; an empty result is raveled to be one-dimensional) and an empty
        # tuple as the missing-value marker.
        frozen = self._values_for_argsort()
        if len(frozen) == 0:
            # factorize_array expects 1-d array, this is a len-0 2-d array.
            frozen = frozen.ravel()
        return frozen, ()

    def _values_for_argsort(self):
        # [FUNCTION DOC]
        # HELPER: sort-key hook. Returns a one-dimensional object array holding, for each element, a
        # tuple of its key-value pairs (built so numpy does not infer a 2D shape).
        # Bypass NumPy's shape inference to get a (N,) array of tuples.
        frozen = [tuple(x.items()) for x in self]
        return construct_1d_object_array_from_listlike(frozen)

    def _pad_or_backfill(self, *, method, limit=None, copy=True):
        # [FUNCTION DOC]
        # HELPER: fill-forward/backward hook. Inputs: method, limit (default none), copy (default true)
        # as keywords. Overrides the base method without the limit-area argument to test arrays that do
        # not support it (regression for an upstream issue); delegates to the base implementation.
        # GH#56616 - test EA method without limit_area argument
        return super()._pad_or_backfill(method=method, limit=limit, copy=copy)


def make_data(n: int):
    # [FUNCTION DOC]
    # HELPER: generates test data. Input n, the count. Returns a list of n user mappings, each with a
    # random number (zero to nine) of entries whose keys are random ASCII letters and whose values are
    # random integers from zero to ninety-nine, drawn from a fixed-seed generator so output is
    # deterministic. A TODO in the code suggests replacing the user mappings with plain dicts.
    # TODO: Use a regular dict. See _NDFrameIndexer._setitem_with_indexer
    rng = np.random.default_rng(2)
    return [
        UserDict(
            [
                (rng.choice(list(string.ascii_letters)), rng.integers(0, 100))
                for _ in range(rng.integers(0, 10))
            ]
        )
        for _ in range(n)
    ]
