# [MODULE DOC]
# Test-only extension array of decimal values (the standard library decimal type) stored in an
# object numpy array. Public contents: DecimalDtype (registered with pandas at import time),
# DecimalArray, to_decimal (build an array from a sequence of numbers) and make_data (random test
# data). DecimalArray mixes in the operator machinery and the scalar-ops mixin, so arithmetic is
# defined by elementwise scalar operators and comparison via a hand-written comparison method.
# Import-time side effects: registers the dtype and attaches the arithmetic operators to the array
# class at the end of the module.
from __future__ import annotations

import decimal
import numbers
import sys
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_float,
    is_integer,
    pandas_dtype,
)

import pandas as pd
from pandas.api.extensions import (
    no_default,
    register_extension_dtype,
)
from pandas.api.types import (
    is_list_like,
    is_scalar,
)
from pandas.core import arraylike
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
)
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
)

if TYPE_CHECKING:
    from pandas._typing import type_t


@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    # [CLASS DOC]
    # Dtype for the decimal test array. Scalar type is the decimal class, name is the string
    # decimal, and the missing-value marker is decimal not-a-number. State: context, the decimal
    # arithmetic context (the current global context if none is given); declared in the dtype
    # metadata so it takes part in equality and hashing. Numeric by declaration.
    type = decimal.Decimal
    name = "decimal"
    na_value = decimal.Decimal("NaN")
    _metadata = ("context",)

    def __init__(self, context=None) -> None:
        # [FUNCTION DOC]
        # Stores the given arithmetic context, or the current global decimal context when the argument
        # is none or otherwise false.
        self.context = context or decimal.getcontext()

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns text of the form DecimalDtype with the context shown.
        return f"DecimalDtype(context={self.context})"

    def construct_array_type(self) -> type_t[DecimalArray]:
        # [FUNCTION DOC]
        # Returns the array class for this dtype, DecimalArray.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray

    @property
    def _is_numeric(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property declaring the dtype numeric; always true.
        return True


class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    # [CLASS DOC]
    # Extension array of decimal values. State: _data (object numpy array of decimals) with the
    # aliases _items and data pointing at the same array, and _dtype (a DecimalDtype carrying the
    # context). Inherits the operator mixin (for comparison and arithmetic dispatch), the scalar-ops
    # mixin and the base extension array. Class attributes: array priority raised for numpy
    # dispatch, and _HANDLED_TYPES, the operand types accepted by the ufunc protocol (decimal, any
    # number, numpy array). The array overrides the fill-missing method to mimic a third-party array
    # that lacks a copy keyword.
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False, context=None) -> None:
        # [FUNCTION DOC]
        # Builds the array. Inputs: values (iterable of decimals, floats or integers), dtype (ignored),
        # copy (ignored), context (arithmetic context for the dtype, default none). Each float or integer is
        # converted to a decimal (NaN becomes the dtype's missing value); anything else that is not a
        # decimal raises a type error.
        # KNOWN ISSUE: conversion mutates the input container in place (assigning into the sequence
        # passed in), so callers' lists or arrays are modified as a side effect.
        # KNOWN ISSUE: the copy argument is ignored, and the final conversion to an object array does not
        # copy, so the new array may share memory with an input numpy array.
        for i, val in enumerate(values):
            if is_float(val) or is_integer(val):
                if np.isnan(val):
                    values[i] = DecimalDtype.na_value
                else:
                    # error: Argument 1 has incompatible type "float | int |
                    # integer[Any]"; expected "Decimal | float | str | tuple[int,
                    # Sequence[int], int]"
                    values[i] = DecimalDtype.type(val)  # type: ignore[arg-type]
            elif not isinstance(val, decimal.Decimal):
                raise TypeError("All values must be of type " + str(decimal.Decimal))
        values = np.asarray(values, dtype=object)

        self._data = values
        # Some aliases for common attribute names to ensure pandas supports
        # these
        self._items = self.data = self._data
        # those aliases are currently not working due to assumptions
        # in internal code (GH-20735)
        # self._values = self.values = self.data
        self._dtype = DecimalDtype(context)

    @property
    def dtype(self):
        # [FUNCTION DOC]
        # Property returning the stored DecimalDtype.
        return self._dtype

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # [FUNCTION DOC]
        # HELPER: alternate constructor used by the extension array machinery. Builds an array from
        # scalars; the dtype and copy arguments are ignored.
        return cls(scalars)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: ExtensionDtype, copy=False):
        # [FUNCTION DOC]
        # HELPER: constructor from text. Converts each string to a decimal and builds the array via the
        # sequence constructor, passing dtype and copy through.
        return cls._from_sequence(
            [decimal.Decimal(x) for x in strings], dtype=dtype, copy=copy
        )

    @classmethod
    def _from_factorized(cls, values, original):
        # [FUNCTION DOC]
        # HELPER: rebuilds an array from unique values produced by factorization; the original array is
        # ignored.
        return cls(values)

    # test to ensure that the base class _cast_pointwise_result works as expected
    # def _cast_pointwise_result(self, values):
    #     try:
    #         # If this were ever made a non-test EA, special-casing could
    #         #  be avoided by handling Decimal in maybe_convert_objects
    #         res = type(self)._from_sequence(values, dtype=self.dtype)
    #     except (ValueError, TypeError):
    #         return values
    #     return res

    _HANDLED_TYPES = (decimal.Decimal, numbers.Number, np.ndarray)

    def to_numpy(
        self,
        dtype=None,
        copy: bool = False,
        na_value: object = no_default,
        decimals=None,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Conversion to numpy. Inputs: dtype (default none), copy (default false), na_value (default
        # unspecified), decimals (default none; if given, every element is rounded to that many digits).
        # KNOWN ISSUE: the copy and na_value arguments are accepted but ignored, so missing values are
        # never replaced and no copy is guaranteed.
        result = np.asarray(self, dtype=dtype)
        if decimals is not None:
            result = np.asarray([round(x, decimals) for x in result])
        return result

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy ufunc protocol. Returns the not-implemented sentinel if any input is not a decimal, a
        # number, a numpy array or a decimal array. Otherwise first tries dispatching to the matching
        # dunder operator; if the call has an out argument, dispatches to the out-aware helper; otherwise
        # unwraps decimal arrays to their data, applies the ufunc method, and for reduce calls tries the
        # reduction dispatcher (returning its result if it handled it). Results (or each element of a tuple
        # result for multi-output ufuncs) are wrapped back into decimal arrays unless they are scalars.
        # KNOWN ISSUE: for reduce, the ufunc is evaluated on the raw data before the reduction dispatcher
        # is consulted, so the work is done twice when the dispatcher handles the call.
        if not all(isinstance(t, (*self._HANDLED_TYPES, DecimalArray)) for t in inputs):
            return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            # e.g. test_array_ufunc_series_scalar_other
            return result

        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        inputs = tuple(x._data if isinstance(x, DecimalArray) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        def reconstruct(x):
            # [FUNCTION DOC]
            # HELPER: converts one ufunc result. Scalars (decimals or numbers) are returned unchanged; any
            # other value is wrapped into a new decimal array of the same dtype.
            if isinstance(x, (decimal.Decimal, numbers.Number)):
                return x
            else:
                return type(self)._from_sequence(x, dtype=self.dtype)

        if ufunc.nout > 1:
            return tuple(reconstruct(x) for x in result)
        else:
            return reconstruct(result)

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Item access. An integer returns the stored decimal. Any other indexer is validated with the
        # array indexer checker and applied to the data, returning a new array with the same context;
        # if the indexer yields a view, the read-only flag is propagated to the result.
        if isinstance(item, numbers.Integral):
            return self._data[item]
        else:
            # array, slice.
            item = pd.api.indexers.check_array_indexer(self, item)
            result = type(self)(self._data[item], context=self.dtype.context)
            if getitem_returns_view(self, item):
                result._readonly = self._readonly
            return result

    def take(self, indexer, allow_fill=False, fill_value=None):
        # [FUNCTION DOC]
        # Positional selection. Inputs: indexer, allow_fill (default false; when true -1 means missing),
        # fill_value (default none, replaced by the dtype's missing value when filling). Uses the generic
        # take helper (imported lazily inside) and returns a new array of the same dtype.
        from pandas.api.extensions import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def copy(self):
        # [FUNCTION DOC]
        # Returns a new array over a copy of the data with the same dtype.
        # KNOWN ISSUE: the dtype is passed to the constructor, which ignores it, so a custom context is
        # dropped by the copy.
        return type(self)(self._data.copy(), dtype=self.dtype)

    def astype(self, dtype, copy=True):
        # [FUNCTION DOC]
        # Conversion. Inputs: dtype, copy (default true). If the target equals the array's dtype and copy
        # is false returns self. If the resolved dtype is a decimal dtype, returns a new array over the
        # same data using the target's context. Otherwise defers to the base implementation.
        # KNOWN ISSUE: the new array in the decimal case shares the same underlying data even when copy
        # is true, because the constructor does not copy.
        if is_dtype_equal(dtype, self._dtype):
            if not copy:
                return self
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, type(self.dtype)):
            return type(self)(self._data, copy=copy, context=dtype.context)

        return super().astype(dtype, copy=copy)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Assignment. Raises a value error if the array is read-only. If the value is list-like:
        # a scalar key raises a value error (setting an element with a sequence) and otherwise each
        # element is converted to a decimal; a non-list-like value is converted to a decimal. Then
        # validates the key with the array indexer checker and assigns into the data in place.
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        if is_list_like(value):
            if is_scalar(key):
                raise ValueError("setting an array element with a sequence.")
            value = [decimal.Decimal(v) for v in value]
        else:
            value = decimal.Decimal(value)

        key = check_array_indexer(self, key)
        self._data[key] = value

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of elements.
        return len(self._data)

    def __contains__(self, item) -> bool | np.bool_:
        # [FUNCTION DOC]
        # Membership. Non-decimal items are never members. A not-a-number decimal is a member exactly
        # when the array contains any missing value; otherwise defers to the base implementation.
        if not isinstance(item, decimal.Decimal):
            return False
        elif item.is_nan():
            return self.isna().any()
        else:
            return super().__contains__(item)

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: an estimate, the length times the in-memory size of the first element (zero when
        # empty).
        n = len(self)
        if n:
            return n * sys.getsizeof(self[0])
        return 0

    def isna(self):
        # [FUNCTION DOC]
        # Returns a boolean numpy array, true where the decimal is not-a-number.
        return np.array([x.is_nan() for x in self._data], dtype=bool)

    @property
    def _na_value(self):
        # [FUNCTION DOC]
        # HELPER: property returning decimal not-a-number as a fresh decimal.
        return decimal.Decimal("NaN")

    def _formatter(self, boxed=False):
        # [FUNCTION DOC]
        # HELPER: formatting hook. Input boxed (default false). Returns a formatter producing text like
        # the word Decimal, a colon and the value when boxed; otherwise the plain representation
        # function.
        if boxed:
            return "Decimal: {}".format
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat):
        # [FUNCTION DOC]
        # HELPER: concatenates a sequence of decimal arrays by joining their data end to end into a
        # new array (the default context is used).
        return cls(np.concatenate([x._data for x in to_concat]))

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # Reduction. Inputs: name (reduction name), skipna (keyword, default true), keepdims (keyword,
        # default false), extra keyword arguments. With skipna true and at least one missing value,
        # reduces the non-missing subset (extra arguments forwarded). Otherwise, a sum of an empty array
        # is decimal zero; any other case looks up the named method on the data array (raising a
        # not-implemented error naming the operation if absent) and applies it along axis zero.
        # Returns the scalar, or a one-element array of it when keepdims is true.
        if skipna and self.isna().any():
            # If we don't have any NAs, we can ignore skipna
            other = self[~self.isna()]
            result = other._reduce(name, **kwargs)
        elif name == "sum" and len(self) == 0:
            # GH#29630 avoid returning int 0 or np.bool_(False) on old numpy
            result = decimal.Decimal(0)
        else:
            try:
                op = getattr(self.data, name)
            except AttributeError as err:
                raise NotImplementedError(
                    f"decimal does not support the {name} operation"
                ) from err
            result = op(axis=0)

        if keepdims:
            return type(self)([result])
        else:
            return result

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: comparison hook for the operator mixin. Inputs: other (array, list-like or scalar), op
        # (binary comparison function). A scalar is broadcast to the array's length. Applies the operator
        # to each pair (raising a value error if lengths differ) and returns a boolean numpy array.
        # Operator failures (type errors) propagate.
        # For use with OpsMixin
        def convert_values(param):
            # [FUNCTION DOC]
            # HELPER: returns an operand as a sequence: extension arrays and list-likes unchanged, scalars as a
            # list repeated to the array's length.
            if isinstance(param, ExtensionArray) or is_list_like(param):
                ovalues = param
            else:
                # Assume it's an object
                ovalues = [param] * len(self)
            return ovalues

        lvalues = self
        rvalues = convert_values(other)

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues, strict=True)]

        return np.asarray(res, dtype=bool)

    # We override fillna here to simulate a 3rd party EA that has done so. This
    #  lets us test a 3rd-party EA that has not yet updated to include a "copy"
    #  keyword in its fillna method.
    def fillna(self, value=None, limit=None):
        # [FUNCTION DOC]
        # Fill-missing override that deliberately omits a copy keyword in its own signature to
        # simulate a third-party array that has not adopted it. Inputs: value, limit (both default
        # none). Delegates to the base implementation with copy forced to true.
        return super().fillna(value=value, limit=limit, copy=True)


def to_decimal(values, context=None):
    # [FUNCTION DOC]
    # Builds a decimal array from a sequence of values. Inputs: values (iterable convertible to
    # decimals), context (default none). Returns an array of the decimal conversions.
    return DecimalArray([decimal.Decimal(x) for x in values], context=context)


def make_data(n: int):
    # [FUNCTION DOC]
    # HELPER: generates test data. Input n, the count. Returns a list of n decimals converted from
    # uniform random floats drawn from a generator with a fixed seed, so output is deterministic.
    return [decimal.Decimal(val) for val in np.random.default_rng(2).random(n)]


DecimalArray._add_arithmetic_ops()
