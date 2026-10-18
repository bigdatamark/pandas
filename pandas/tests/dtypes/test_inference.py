"""
These the test the public routines exposed in types/common.py
related to inference and not otherwise tested in types/test_common.py

"""

# [MODULE DOC]
# Tests for the public inference helpers of the dtypes inference module and
# related type-checking helpers: list-like, sequence, array-like, nested
# list-like, dict-like, file-like, named tuple, hashable and regular expression
# predicates; the lib inference routines (inferring a dtype name from values,
# converting objects to numeric, datetime or timedelta, maybe-convert helpers);
# scalar type predicates (is number, integer, float, bool, complex, scalar and
# more); and missing-value detection. Defines a duck-typed numpy-like array
# class, shared fixtures and a table of list-like sample objects.
# Note: one tiny generic subclass in a list-like test is declared on a single
# line (the class body is on the same line as its header), so it cannot carry its
# own class documentation block and is described in the enclosing test instead.

import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
    Generic,
    TypeVar,
)

import numpy as np
import pytest

from pandas._libs import (
    lib,
    missing as libmissing,
    ops as libops,
)
from pandas.compat import PY312
from pandas.compat.numpy import np_version_gt2
from pandas.errors import Pandas4Warning

from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
    ensure_int32,
    is_bool,
    is_complex,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_float,
    is_integer,
    is_number,
    is_scalar,
    is_scipy_sparse,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
)

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    Interval,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    FloatingArray,
    IntegerArray,
)


@pytest.fixture(params=[True, False], ids=str)
def coerce(request):
    # [FUNCTION DOC]
    # Fixture parametrized over true and false (ids as strings); returns the value.
    # Used to toggle a coercion option in tests.
    return request.param


class MockNumpyLikeArray:
    # [CLASS DOC]
    # HELPER: test double for a numpy-like duck array (such as a unit-aware quantity)
    # that is not a real numpy array, so the array detection helper reports false.
    # Stores the wrapped values; defines iteration, length, conversion to a numpy
    # array, and read-only ndim, dtype, size and shape properties delegated to the
    # wrapped numpy array. A zero-dimensional instance must not be a scalar and not
    # list-like.
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy

    The key is that it is not actually a numpy array so
    ``util.is_array(mock_numpy_like_array_instance)`` returns ``False``. Other
    important properties are that the class defines a :meth:`__iter__` method
    (so that ``isinstance(abc.Iterable)`` returns ``True``) and has a
    :meth:`ndim` property, as pandas special-cases 0-dimensional arrays in some
    cases.

    We expect pandas to behave with respect to such duck arrays exactly as
    with real numpy arrays. In particular, a 0-dimensional duck array is *NOT*
    a scalar (`is_scalar(np.array(1)) == False`), but it is not list-like either.
    """

    def __init__(self, values) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the given values (a numpy array) on the instance. Returns none.
        self._values = values

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # HELPER: returns a generator-based iterator over the wrapped values (via a
        # nested generator function) so the object counts as iterable.
        iter_values = iter(self._values)

        def it_outer():
            # [FUNCTION DOC]
            # HELPER: nested generator function; yields each remaining item of the captured
            # iterator.
            yield from iter_values

        return it_outer()

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # HELPER: returns the length of the wrapped values.
        return len(self._values)

    def __array__(self, dtype=None, copy=None):
        # [FUNCTION DOC]
        # HELPER: numpy conversion hook. Parameters dtype (default none) and copy
        # (default none, ignored). Returns the wrapped values as a numpy array of the
        # requested dtype.
        return np.asarray(self._values, dtype=dtype)

    @property
    def ndim(self):
        # [FUNCTION DOC]
        # HELPER: property returning the number of dimensions of the wrapped values.
        return self._values.ndim

    @property
    def dtype(self):
        # [FUNCTION DOC]
        # HELPER: property returning the dtype of the wrapped values.
        return self._values.dtype

    @property
    def size(self):
        # [FUNCTION DOC]
        # HELPER: property returning the element count of the wrapped values.
        return self._values.size

    @property
    def shape(self):
        # [FUNCTION DOC]
        # HELPER: property returning the shape of the wrapped values.
        return self._values.shape


# collect all objects to be tested for list-like-ness; use tuples of objects,
# whether they are list-like or not (special casing for sets), and their ID
ll_params = [
    ([1], True, "list"),
    ([], True, "list-empty"),
    ((1,), True, "tuple"),
    ((), True, "tuple-empty"),
    ({"a": 1}, True, "dict"),
    ({}, True, "dict-empty"),
    ({"a", 1}, "set", "set"),
    (set(), "set", "set-empty"),
    (frozenset({"a", 1}), "set", "frozenset"),
    (frozenset(), "set", "frozenset-empty"),
    (iter([1, 2]), True, "iterator"),
    (iter([]), True, "iterator-empty"),
    ((x for x in [1, 2]), True, "generator"),
    ((_ for _ in []), True, "generator-empty"),
    (Series([1]), True, "Series"),
    (Series([], dtype=object), True, "Series-empty"),
    # Series.str will still raise a TypeError if iterated
    (Series(["a"]).str, True, "StringMethods"),
    (Series([], dtype="O").str, True, "StringMethods-empty"),
    (Index([1]), True, "Index"),
    (Index([]), True, "Index-empty"),
    (DataFrame([[1]]), True, "DataFrame"),
    (DataFrame(), True, "DataFrame-empty"),
    (np.ndarray((2,) * 1), True, "ndarray-1d"),
    (np.array([]), True, "ndarray-1d-empty"),
    (np.ndarray((2,) * 2), True, "ndarray-2d"),
    (np.array([[]]), True, "ndarray-2d-empty"),
    (np.ndarray((2,) * 3), True, "ndarray-3d"),
    (np.array([[[]]]), True, "ndarray-3d-empty"),
    (np.ndarray((2,) * 4), True, "ndarray-4d"),
    (np.array([[[[]]]]), True, "ndarray-4d-empty"),
    (np.array(2), False, "ndarray-0d"),
    (MockNumpyLikeArray(np.ndarray((2,) * 1)), True, "duck-ndarray-1d"),
    (MockNumpyLikeArray(np.array([])), True, "duck-ndarray-1d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 2)), True, "duck-ndarray-2d"),
    (MockNumpyLikeArray(np.array([[]])), True, "duck-ndarray-2d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 3)), True, "duck-ndarray-3d"),
    (MockNumpyLikeArray(np.array([[[]]])), True, "duck-ndarray-3d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 4)), True, "duck-ndarray-4d"),
    (MockNumpyLikeArray(np.array([[[[]]]])), True, "duck-ndarray-4d-empty"),
    (MockNumpyLikeArray(np.array(2)), False, "duck-ndarray-0d"),
    (1, False, "int"),
    (b"123", False, "bytes"),
    (b"", False, "bytes-empty"),
    ("123", False, "string"),
    ("", False, "string-empty"),
    (str, False, "string-type"),
    (object(), False, "object"),
    (np.nan, False, "NaN"),
    (None, False, "None"),
]
objs, expected, ids = zip(*ll_params, strict=True)


@pytest.fixture(params=zip(objs, expected, strict=True), ids=ids)
def maybe_list_like(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the module-level table of sample objects
    # paired with their expected list-like flag (true, false, or the word set for set
    # types, which are list-like only when sets are allowed). Test ids are the
    # descriptive names from the table. Returns the (object, expected) pair.
    return request.param


def test_is_list_like(maybe_list_like):
    # [FUNCTION DOC]
    # Using the list-like fixture. The predicate with default options returns the
    # expected flag, where the expectation for sets is treated as true.
    obj, expected = maybe_list_like
    expected = True if expected == "set" else expected
    assert inference.is_list_like(obj) == expected


def test_is_list_like_disallow_sets(maybe_list_like):
    # [FUNCTION DOC]
    # With sets disallowed, the predicate returns the expected flag, where the
    # expectation for sets is treated as false.
    obj, expected = maybe_list_like
    expected = False if expected == "set" else expected
    assert inference.is_list_like(obj, allow_sets=False) == expected


def test_is_list_like_recursion():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue where the interpreter crashed on infinite
    # recursion. A nested function recursing through the predicate is run with a
    # reduced recursion limit (to avoid a stack overflow on Windows); a recursion error
    # is expected from the external helper. The original limit is always restored.
    # GH 33721
    # interpreter would crash with SIGABRT
    def list_like():
        # [FUNCTION DOC]
        # HELPER: nested function; calls the list-like predicate on an empty list, then
        # recurses into itself without end, so a recursion error is eventually raised.
        inference.is_list_like([])
        list_like()

    rec_limit = sys.getrecursionlimit()
    try:
        # Limit to avoid stack overflow on Windows CI
        sys.setrecursionlimit(100)
        with tm.external_error_raised(RecursionError):
            list_like()
    finally:
        sys.setrecursionlimit(rec_limit)


def test_is_list_like_iter_is_none():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: an object with item access but whose
    # iteration attribute is set to none is not list-like.
    # GH 43373
    # is_list_like was yielding false positives with __iter__ == None
    class NotListLike:
        # [CLASS DOC]
        # HELPER: nested class used only by the preceding test; supports item access
        # (returning itself) and has iteration disabled.
        def __getitem__(self, item):
            # [FUNCTION DOC]
            # HELPER: returns the object itself for any key.
            return self

        __iter__ = None

    assert not inference.is_list_like(NotListLike())


def test_is_list_like_generic():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue about false positives for generic classes on
    # Python 3.11. A frame subclass made generic over a type variable (declared on a
    # single line, so it is not documented separately): the parametrized class object
    # is not list-like, but an instance created from data is a frame and list-like.
    # GH 49649
    # is_list_like was yielding false positives for Generic classes in python 3.11
    T = TypeVar("T")

    class MyDataFrame(DataFrame, Generic[T]): ...

    tstc = MyDataFrame[int]
    tst = MyDataFrame[int]({"x": [1, 2, 3]})

    assert not inference.is_list_like(tstc)
    assert isinstance(tst, DataFrame)
    assert inference.is_list_like(tst)


def test_is_list_like_native_container_types():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: parametrized native container type aliases
    # (list or tuple of int or str) are not list-like.
    # GH 61565
    # is_list_like was yielding false positives for native container types
    assert not inference.is_list_like(list[int])
    assert not inference.is_list_like(list[str])
    assert not inference.is_list_like(tuple[int])
    assert not inference.is_list_like(tuple[str])


def test_is_sequence():
    # [FUNCTION DOC]
    # The sequence predicate is true for tuples and lists; false for strings and
    # types; false for an object that only supports item access.
    is_seq = inference.is_sequence
    assert is_seq((1, 2))
    assert is_seq([1, 2])
    assert not is_seq("abcd")
    assert not is_seq(np.int64)

    class A:
        # [CLASS DOC]
        # HELPER: nested class used by the preceding test; supports item access always
        # returning one, but is not a sequence.
        def __getitem__(self, item):
            # [FUNCTION DOC]
            # HELPER: returns the integer one for any key.
            return 1

    assert not is_seq(A())


def test_is_array_like():
    # [FUNCTION DOC]
    # The array-like predicate is true for Series, numpy arrays, Index, the mock
    # numpy-like class and a list subclass carrying a dtype attribute; false for a
    # plain list, empty tuple, string and integer.
    assert inference.is_array_like(Series([], dtype=object))
    assert inference.is_array_like(Series([1, 2]))
    assert inference.is_array_like(np.array(["a", "b"]))
    assert inference.is_array_like(Index(["2016-01-01"]))
    assert inference.is_array_like(np.array([2, 3]))
    assert inference.is_array_like(MockNumpyLikeArray(np.array([2, 3])))

    class DtypeList(list):
        # [CLASS DOC]
        # HELPER: nested list subclass used by the preceding test; carries a dtype class
        # attribute set to a string so that it looks array-like.
        dtype = "special"

    assert inference.is_array_like(DtypeList())

    assert not inference.is_array_like([1, 2, 3])
    assert not inference.is_array_like(())
    assert not inference.is_array_like("foo")
    assert not inference.is_array_like(123)


@pytest.mark.parametrize(
    "inner",
    [
        [],
        [1],
        (1,),
        (1, 2),
        {"a": 1},
        {1, "a"},
        Series([1]),
        Series([], dtype=object),
        Series(["a"]).str,
        (x for x in range(5)),
    ],
)
@pytest.mark.parametrize("outer", [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner, outer):
    # [FUNCTION DOC]
    # Parametrized over inner containers (lists, tuples, dict, set, Series, string
    # accessor, generator) and outer constructors (list, Series, array, tuple). A
    # container of five such inner elements is list-like.
    result = outer([inner for _ in range(5)])
    assert inference.is_list_like(result)


@pytest.mark.parametrize(
    "obj",
    [
        "abc",
        [],
        [1],
        (1,),
        ["a"],
        "a",
        {"a"},
        [1, 2, 3],
        Series([1]),
        DataFrame({"A": [1]}),
        ([1, 2] for _ in range(5)),
    ],
)
def test_is_nested_list_like_fails(obj):
    # [FUNCTION DOC]
    # Parametrized over strings, flat lists, tuples, sets, a Series, a frame and a
    # generator of lists. The nested list-like predicate is false for all of them.
    assert not inference.is_nested_list_like(obj)


@pytest.mark.parametrize("ll", [{}, {"A": 1}, Series([1]), collections.defaultdict()])
def test_is_dict_like_passes(ll):
    # [FUNCTION DOC]
    # Parametrized over empty dict, dict, Series and default dict instance: all dict-like.
    assert inference.is_dict_like(ll)


@pytest.mark.parametrize(
    "ll",
    [
        "1",
        1,
        [1, 2],
        (1, 2),
        range(2),
        Index([1]),
        dict,
        collections.defaultdict,
        Series,
    ],
)
def test_is_dict_like_fails(ll):
    # [FUNCTION DOC]
    # Parametrized over strings, numbers, lists, tuples, range, Index and the classes
    # dict, default dict and Series themselves: none are dict-like.
    assert not inference.is_dict_like(ll)


@pytest.mark.parametrize("has_keys", [True, False])
@pytest.mark.parametrize("has_getitem", [True, False])
@pytest.mark.parametrize("has_contains", [True, False])
def test_is_dict_like_duck_type(has_keys, has_getitem, has_contains):
    # [FUNCTION DOC]
    # Parametrized over presence of keys, item access and containment methods. A
    # local class defined with exactly the chosen methods is dict-like only if it has
    # all three; the result must be an exact boolean.
    class DictLike:
        # [CLASS DOC]
        # HELPER: nested class defined per parameter combination; stores a dict and
        # conditionally defines keys, item access and containment by delegating to it.
        def __init__(self, d) -> None:
            # [FUNCTION DOC]
            # HELPER: stores the given dictionary on the instance.
            self.d = d

        if has_keys:

            def keys(self):
                # [FUNCTION DOC]
                # HELPER: conditionally defined; returns the keys of the stored dictionary.
                return self.d.keys()

        if has_getitem:

            def __getitem__(self, key):
                # [FUNCTION DOC]
                # HELPER: conditionally defined; returns the item for a key from the stored dictionary.
                return self.d.__getitem__(key)

        if has_contains:

            def __contains__(self, key) -> bool:
                # [FUNCTION DOC]
                # HELPER: conditionally defined; returns whether a key is in the stored dictionary.
                return self.d.__contains__(key)

    d = DictLike({1: 2})
    result = inference.is_dict_like(d)
    expected = has_keys and has_getitem and has_contains

    assert result is expected


def test_is_file_like():
    # [FUNCTION DOC]
    # The file-like predicate: true for an in-memory text stream; false for an
    # object lacking read, write and iteration; false with write only (not
    # iterable); true once it is iterable and has a write attribute (an iterator
    # method suffices, per a GitHub issue); true for read only plus iterable; false
    # for a list. Attributes are attached to and removed from the local class
    # dynamically during the test.
    class MockFile:
        # [CLASS DOC]
        # HELPER: nested empty class used by the preceding test; attributes such as write,
        # read and iteration are added to it dynamically.
        pass

    is_file = inference.is_file_like

    data = StringIO("data")
    assert is_file(data)

    # No read / write attributes
    # No iterator attributes
    m = MockFile()
    assert not is_file(m)

    MockFile.write = lambda self: 0

    # Write attribute but not an iterator
    m = MockFile()
    assert not is_file(m)

    # gh-16530: Valid iterator just means we have the
    # __iter__ attribute for our purposes.
    MockFile.__iter__ = lambda self: self

    # Valid write-only file
    m = MockFile()
    assert is_file(m)

    del MockFile.write
    MockFile.read = lambda self: 0

    # Valid read-only file
    m = MockFile()
    assert is_file(m)

    # Iterator but no read / write attributes
    data = [1, 2, 3]
    assert not is_file(data)


test_tuple = collections.namedtuple("test_tuple", ["a", "b", "c"])


@pytest.mark.parametrize("ll", [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll):
    # [FUNCTION DOC]
    # Parametrized over one named tuple instance: the named-tuple predicate is true.
    assert inference.is_named_tuple(ll)


@pytest.mark.parametrize("ll", [(1, 2, 3), "a", Series({"pi": 3.14})])
def test_is_names_tuple_fails(ll):
    # [FUNCTION DOC]
    # Parametrized over a plain tuple, a string and a Series: the named-tuple
    # predicate is false.
    assert not inference.is_named_tuple(ll)


def test_is_hashable():
    # [FUNCTION DOC]
    # Checks the hashability predicate with and without slice allowance. Hashable
    # sample values (numbers, string, tuples, plain object) are hashable in all
    # modes; lists and objects with unset or raising hash are not hashable in any mode
    # (including a tuple holding a list). A custom hashable slice wrapper is hashable.
    # Real slices and tuples containing slices are hashable (unless slices are
    # disallowed) on Python 3.12 and newer, and not hashable at all on older versions.
    # A numpy array is not hashable.
    # all new-style classes are hashable by default
    class HashableClass:
        # [CLASS DOC]
        # HELPER: nested class; a plain class that is hashable by default.
        pass

    class UnhashableClass1:
        # [CLASS DOC]
        # HELPER: nested class; sets its hash attribute to none so it is not hashable.
        __hash__ = None

    class UnhashableClass2:
        # [CLASS DOC]
        # HELPER: nested class; defines a hash method that raises a type error, so it
        # appears hashable to the abstract base class but is not really.
        def __hash__(self):
            # [FUNCTION DOC]
            # HELPER: always raises a type error saying it is not hashable.
            raise TypeError("Not hashable")

    # Temporary helper for Python 3.11 compatibility.
    # This can be removed once support for Python 3.11 is dropped.
    class HashableSlice:
        # [CLASS DOC]
        # HELPER: temporary wrapper around a slice for Python 3.11 compatibility (can be
        # removed once that version is unsupported); stores the slice and defines
        # equality, hash and repr so it is hashable.
        def __init__(self, start, stop, step=None):
            # [FUNCTION DOC]
            # HELPER: parameters start, stop and step (default none); stores a slice built from them.
            self.slice = slice(start, stop, step)

        def __eq__(self, other):
            # [FUNCTION DOC]
            # HELPER: equal to another instance of the same class with an equal stored slice;
            # false for other objects.
            return isinstance(other, HashableSlice) and self.slice == other.slice

        def __hash__(self):
            # [FUNCTION DOC]
            # HELPER: hash of the tuple of the slice's start, stop and step.
            return hash((self.slice.start, self.slice.stop, self.slice.step))

        def __repr__(self):
            # [FUNCTION DOC]
            # HELPER: returns a string showing the class name with start, stop and step.
            return (
                f"HashableSlice({self.slice.start}, {self.slice.stop}, "
                f"{self.slice.step})"
            )

    hashable = (1, 3.14, np.float64(3.14), "a", (), (1,), HashableClass())
    not_hashable = ([], UnhashableClass1())
    abc_hashable_not_really_hashable = (([],), UnhashableClass2())
    hashable_slice = HashableSlice(1, 2)
    tuple_with_slice = (slice(1, 2), 3)

    for i in hashable:
        assert inference.is_hashable(i)
        assert inference.is_hashable(i, allow_slice=True)
        assert inference.is_hashable(i, allow_slice=False)
    for i in not_hashable:
        assert not inference.is_hashable(i)
        assert not inference.is_hashable(i, allow_slice=True)
        assert not inference.is_hashable(i, allow_slice=False)
    for i in abc_hashable_not_really_hashable:
        assert not inference.is_hashable(i)
        assert not inference.is_hashable(i, allow_slice=True)
        assert not inference.is_hashable(i, allow_slice=False)

    assert inference.is_hashable(hashable_slice)
    assert inference.is_hashable(hashable_slice, allow_slice=True)
    assert inference.is_hashable(hashable_slice, allow_slice=False)

    if PY312:
        for obj in [slice(1, 2), tuple_with_slice]:
            assert inference.is_hashable(obj)
            assert inference.is_hashable(obj, allow_slice=True)
            assert not inference.is_hashable(obj, allow_slice=False)
    else:
        for obj in [slice(1, 2), tuple_with_slice]:
            assert not inference.is_hashable(obj)
            assert not inference.is_hashable(obj, allow_slice=True)
            assert not inference.is_hashable(obj, allow_slice=False)

    # numpy.array is no longer collections.abc.Hashable as of
    # https://github.com/numpy/numpy/pull/5326, just test
    # is_hashable()
    assert not inference.is_hashable(np.array([]))


@pytest.mark.parametrize("ll", [re.compile("ad")])
def test_is_re_passes(ll):
    # [FUNCTION DOC]
    # Parametrized over a compiled pattern: the regex predicate is true.
    assert inference.is_re(ll)


@pytest.mark.parametrize("ll", ["x", 2, 3, object()])
def test_is_re_fails(ll):
    # [FUNCTION DOC]
    # Parametrized over a string, integers and a plain object: the regex predicate is false.
    assert not inference.is_re(ll)


@pytest.mark.parametrize(
    "ll", [r"a", "x", r"asdf", re.compile("adsf"), r"\u2233\s*", re.compile(r"")]
)
def test_is_recompilable_passes(ll):
    # [FUNCTION DOC]
    # Parametrized over strings (including unicode escapes and the empty pattern) and
    # compiled patterns: the predicate for things compilable as a regex is true.
    assert inference.is_re_compilable(ll)


@pytest.mark.parametrize("ll", [1, [], object()])
def test_is_recompilable_fails(ll):
    # [FUNCTION DOC]
    # Parametrized over an integer, empty list and plain object: not regex-compilable.
    assert not inference.is_re_compilable(ll)


class TestInference:
    # [CLASS DOC]
    # Tests of inference routines in the lib module and related scalar predicates;
    # a large group with no shared state (continued below).
    @pytest.mark.parametrize(
        "arr",
        [
            np.array(list("abc"), dtype="S1"),
            np.array(list("abc"), dtype="S1").astype(object),
            [b"a", np.nan, b"c"],
        ],
    )
    def test_infer_dtype_bytes(self, arr):
        # [FUNCTION DOC]
        # Parametrized over bytes arrays (fixed width, object, list with NaN). Inferring
        # the dtype name with missing values skipped gives bytes.
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "bytes"

    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("inf"), True),
            (np.inf, True),
            (-np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isposinf_scalar(self, value, expected):
        # [FUNCTION DOC]
        # Parametrized over (value, expected): positive infinity values give true;
        # negative infinity, integer and string give false. Regression test for a GitHub
        # issue about the positive-infinity scalar check.
        # GH 11352
        result = libmissing.isposinf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("-inf"), True),
            (-np.inf, True),
            (np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isneginf_scalar(self, value, expected):
        # [FUNCTION DOC]
        # Parametrized over (value, expected): negative infinity values give true; positive
        # infinity, integer and string give false. The negative-infinity scalar check
        # returns an exact boolean.
        result = libmissing.isneginf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                BooleanArray(
                    np.array([True, False], dtype="bool"), np.array([False, True])
                ),
            ),
            (False, np.array([True, np.nan], dtype="object")),
        ],
    )
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable, exp):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over the masked-nullable
        # option and expected output. An object array holding a boolean and NaN is
        # converted by the boolean converter: with the option it returns data and mask
        # building an equal nullable boolean array; without it returns the unconverted
        # object array.
        # GH 40687
        arr = np.array([True, np.nan], dtype=object)
        result = libops.maybe_convert_bool(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    @pytest.mark.parametrize("coerce_numeric", [True, False])
    @pytest.mark.parametrize(
        "infinity", ["inf", "inF", "iNf", "Inf", "iNF", "InF", "INf", "INF"]
    )
    @pytest.mark.parametrize("prefix", ["", "-", "+"])
    def test_maybe_convert_numeric_infinities(
        self, coerce_numeric, infinity, prefix, convert_to_masked_nullable
    ):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Parametrized over masked-nullable option,
        # coerce option, infinity spellings in mixed case and sign prefix (none, minus,
        # plus). Converting the single string gives a float array holding positive
        # infinity (for none or plus) or negative infinity (for minus).
        # see gh-13274
        result, _ = lib.maybe_convert_numeric(
            np.array([prefix + infinity], dtype=object),
            na_values={"", "NULL", "nan"},
            coerce_numeric=coerce_numeric,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        expected = np.array([np.inf if prefix in ["", "+"] else -np.inf])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable):
        # [FUNCTION DOC]
        # Parametrized over the masked-nullable option. A string that merely contains an
        # infinity spelling and is not a valid number raises a value error when
        # unable to parse.
        msg = "Unable to parse string"
        with pytest.raises(ValueError, match=msg):
            lib.maybe_convert_numeric(
                np.array(["foo_inf"], dtype=object),
                na_values={"", "NULL", "nan"},
                coerce_numeric=False,
                convert_to_masked_nullable=convert_to_masked_nullable,
            )

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_maybe_convert_numeric_post_floatify_nan(
        self, coerce, convert_to_masked_nullable
    ):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; fixture coerce; parametrized over the masked
        # nullable option. Strings of floats where -999 is declared as a missing value
        # convert to floats with NaN there; with the option the result (data and mask)
        # builds an equal nullable float array.
        # see gh-13314
        data = np.array(["1.200", "-999.000", "4.500"], dtype=object)
        expected = np.array([1.2, np.nan, 4.5], dtype=np.float64)
        nan_values = {-999, -999.0}

        out = lib.maybe_convert_numeric(
            data,
            nan_values,
            coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            expected = FloatingArray(expected, np.isnan(expected))
            tm.assert_extension_array_equal(expected, FloatingArray(*out))
        else:
            out = out[0]
            tm.assert_numpy_array_equal(out, expected)

    def test_convert_infs(self):
        # [FUNCTION DOC]
        # Arrays of positive or negative infinity strings convert to float64.
        arr = np.array(["inf", "inf", "inf"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

        arr = np.array(["-inf", "-inf", "-inf"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

    def test_scientific_no_exponent(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub pull request: strings that look like scientific
        # notation without an exponent value become NaN (with coercion).
        # See PR 12215
        arr = np.array(["42E", "2E", "99e", "6e"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        assert np.all(np.isnan(result))

    def test_convert_non_hashable(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue: unhashable elements (a nested list) and
        # non-numeric strings convert to NaN with coercion while numbers are kept.
        # GH13324
        # make sure that we are handing non-hashables
        arr = np.array([[10.0, 2], 1.0, "apple"], dtype=object)
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))

    def test_convert_numeric_uint64(self):
        # [FUNCTION DOC]
        # A value too large for int64 given as an integer, a string or an unsigned numpy
        # scalar converts to an unsigned 64-bit integer array.
        arr = np.array([2**63], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        arr = np.array([str(2**63)], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        arr = np.array([np.uint64(2**63)], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([2**63, np.nan], dtype=object),
            np.array([str(2**63), np.nan], dtype=object),
            np.array([np.nan, 2**63], dtype=object),
            np.array([np.nan, str(2**63)], dtype=object),
        ],
    )
    def test_convert_numeric_uint64_nan(self, coerce, arr):
        # [FUNCTION DOC]
        # Fixture coerce; parametrized over arrays mixing a uint64-only value (integer
        # or string) with NaN in either order. The result approximately equals the float
        # version of the input if coercion is on, otherwise the input unchanged.
        expected = arr.astype(float) if coerce else arr.copy()
        result, _ = lib.maybe_convert_numeric(arr, set(), coerce_numeric=coerce)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_uint64_nan_values(
        self, coerce, convert_to_masked_nullable
    ):
        # [FUNCTION DOC]
        # Fixture coerce; parametrized over the masked-nullable option. With a large
        # value declared missing: coercion yields a float array with NaN; without it the
        # input is returned unchanged. With both coercion and the masked option the data
        # and mask build an equal nullable unsigned integer array.
        arr = np.array([2**63, 2**63 + 1], dtype=object)
        na_values = {2**63}

        expected = np.array([np.nan, 2**63 + 1], dtype=float) if coerce else arr.copy()
        result = lib.maybe_convert_numeric(
            arr,
            na_values,
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable and coerce:
            expected = IntegerArray(
                np.array([0, 2**63 + 1], dtype="u8"),
                np.array([True, False], dtype="bool"),
            )
            result = IntegerArray(*result)
        else:
            result = result[0]  # discard mask
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "case",
        [
            np.array([2**63, -1], dtype=object),
            np.array([str(2**63), -1], dtype=object),
            np.array([str(2**63), str(-1)], dtype=object),
            np.array([-1, 2**63], dtype=object),
            np.array([-1, str(2**63)], dtype=object),
            np.array([str(-1), str(2**63)], dtype=object),
        ],
    )
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_int64_uint64(
        self, case, coerce, convert_to_masked_nullable
    ):
        # [FUNCTION DOC]
        # Fixture coerce; parametrized over arrays mixing negative values and values above
        # the int64 maximum (ints and strings, both orders) and the masked-nullable
        # option. Result approximately equals the float conversion with coercion, or
        # the input unchanged without it.
        expected = case.astype(float) if coerce else case.copy()
        result, _ = lib.maybe_convert_numeric(
            case,
            set(),
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )

        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over the masked-nullable
        # option. A non-numeric string with coercion converts to NaN (as a nullable float
        # when the option is on).
        # GH32394
        result = lib.maybe_convert_numeric(
            np.array(["uint64"], dtype=object),
            set(),
            coerce_numeric=True,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            result = FloatingArray(*result)
        else:
            result = result[0]
        assert np.isnan(result)

    @pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
    def test_convert_int_overflow(self, value):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over values just outside the
        # 64-bit ranges. Maybe-convert-objects leaves the object array unchanged.
        # see gh-18584
        arr = np.array([value], dtype=object)
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(arr, result)

    @pytest.mark.parametrize(
        "value, expected_value",
        [
            (-(1 << 65), -(1 << 65)),
            (1 << 65, 1 << 65),
            (str(1 << 65), 1 << 65),
            (f"-{1 << 65}", -(1 << 65)),
        ],
    )
    @pytest.mark.parametrize("coerce_numeric", [False, True])
    def test_convert_numeric_overflow(self, value, expected_value, coerce_numeric):
        # [FUNCTION DOC]
        # Parametrized over very large integers as integer or string and the coerce
        # option. The numeric converter returns float values if coercing, otherwise
        # an object array holding the Python integer values.
        arr = np.array([value], dtype=object)
        expected = np.array([expected_value], dtype=float if coerce_numeric else object)
        result, _ = lib.maybe_convert_numeric(
            arr,
            set(),
            coerce_numeric=coerce_numeric,
        )
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("val", [None, np.nan, float("nan")])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_maybe_convert_objects_nat_inference(self, val, dtype):
        # [FUNCTION DOC]
        # Parametrized over a second element (none, NaN, float NaN) and datetime64 or
        # timedelta64 dtype. An object array of NaT and the missing value, in either
        # order, converts with non-numeric conversion on and the all-NaT dtype given to an
        # array of that dtype that is entirely NaT.
        dtype = np.dtype(dtype)
        vals = np.array([pd.NaT, val], dtype=object)
        result = lib.maybe_convert_objects(
            vals,
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        assert result.dtype == dtype
        assert np.isnat(result).all()

        result = lib.maybe_convert_objects(
            vals[::-1],
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        assert result.dtype == dtype
        assert np.isnat(result).all()

    @pytest.mark.parametrize(
        "value, expected_dtype",
        [
            # see gh-4471
            ([2**63], np.uint64),
            # NumPy bug: can't compare uint64 to int64, as that
            # results in both casting to float64, so we should
            # make sure that this function is robust against it
            ([np.uint64(2**63)], np.uint64),
            ([2, -1], np.int64),
            ([2**63, -1], object),
            # GH#47294
            ([np.uint8(1)], np.uint8),
            ([np.uint16(1)], np.uint16),
            ([np.uint32(1)], np.uint32),
            ([np.uint64(1)], np.uint64),
            ([np.uint8(2), np.uint16(1)], np.uint16),
            ([np.uint32(2), np.uint16(1)], np.uint32),
            ([np.uint32(2), -1], object),
            ([np.uint32(2), 1], np.uint64),
            ([np.uint32(2), np.int32(1)], object),
        ],
    )
    def test_maybe_convert_objects_uint(self, value, expected_dtype):
        # [FUNCTION DOC]
        # Parametrized over lists of Python and numpy integers and expected dtype (see
        # the GitHub issues cited in comments). Maybe-convert-objects on the object array
        # equals an array of that expected dtype: the smallest unsigned type for numpy
        # unsigned scalars, int64 for small mixed signs, unsigned 64-bit for a huge value,
        # and object when a huge value or unsigned scalar is mixed with negatives or
        # signed numpy integers.
        arr = np.array(value, dtype=object)
        exp = np.array(value, dtype=expected_dtype)
        tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)

    def test_maybe_convert_objects_datetime(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. An array mixing numpy datetime64 and
        # timedelta64 stays unchanged. NaT with a timedelta converts to a timedelta
        # array in seconds. A timedelta with NaN converts the same way reversed (NaN is a
        # valid missing value for timedelta).
        # GH27438
        arr = np.array(
            [np.datetime64("2000-01-01"), np.timedelta64(1, "s")], dtype=object
        )
        exp = arr.copy()
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

        arr = np.array([pd.NaT, np.timedelta64(1, "s")], dtype=object)
        exp = np.array(
            [np.timedelta64("NaT", "s"), np.timedelta64(1, "s")], dtype="m8[s]"
        )
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

        # with convert_non_numeric=True, the nan is a valid NA value for td64
        arr = np.array([np.timedelta64(1, "s"), np.nan], dtype=object)
        exp = exp[::-1]
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat(self):
        # [FUNCTION DOC]
        # An all-NaT object array is left as is without the all-NaT dtype; with a timedelta
        # dtype or datetime dtype given, it becomes an all-NaT array of that dtype.
        arr = np.array([pd.NaT, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no dtype_if_all_nat passed -> we dont guess
        tm.assert_numpy_array_equal(out, arr)

        out = lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("timedelta64[ns]"),
        )
        exp = np.array(["NaT", "NaT"], dtype="timedelta64[ns]")
        tm.assert_numpy_array_equal(out, exp)

        out = lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("datetime64[ns]"),
        )
        exp = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self):
        # [FUNCTION DOC]
        # Giving an unsupported integer dtype as the all-NaT dtype raises a value error
        # mentioning int64 (datetime64, timedelta64 and extension dtypes are accepted).
        # we accept datetime64[ns], timedelta64[ns], and EADtype
        arr = np.array([pd.NaT, pd.NaT], dtype=object)

        with pytest.raises(ValueError, match="int64"):
            lib.maybe_convert_objects(
                arr,
                convert_non_numeric=True,
                dtype_if_all_nat=np.dtype("int64"),
            )

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype):
        # [FUNCTION DOC]
        # Parametrized over datetime and timedelta nanosecond dtype names. A date far
        # beyond the nanosecond range (a fictional far-future date, or its offset from the
        # epoch) converts to a microsecond-resolution array without an out-of-bounds
        # error.
        stamp = datetime(2363, 10, 4)  # Enterprise-D launch date
        if dtype == "timedelta64[ns]":
            stamp = stamp - datetime(1970, 1, 1)
        arr = np.array([stamp], dtype=object)

        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no OutOfBoundsDatetime/OutOfBoundsTimedeltas
        if dtype == "datetime64[ns]":
            expected = np.array(["2363-10-04"], dtype="M8[us]")
        else:
            expected = arr.astype("m8[us]")
        tm.assert_numpy_array_equal(out, expected)

    def test_maybe_convert_objects_mixed_datetimes(self):
        # [FUNCTION DOC]
        # Every permutation of a Timestamp, python datetime, numpy datetime64, NaT, NaN
        # and none converts to the same datetime array as building a datetime index
        # from the same data.
        ts = Timestamp("now")
        vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]

        for data in itertools.permutations(vals):
            data = np.array(list(data), dtype=object)
            expected = DatetimeIndex(data)._data._ndarray
            result = lib.maybe_convert_objects(data, convert_non_numeric=True)
            tm.assert_numpy_array_equal(result, expected)

    def test_maybe_convert_objects_timedelta64_nat(self):
        # [FUNCTION DOC]
        # An object array holding a numpy timedelta NaT (identity checked) converts to a
        # nanosecond timedelta array holding NaT.
        obj = np.timedelta64("NaT", "ns")
        arr = np.array([obj], dtype=object)
        assert arr[0] is obj

        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)

        expected = np.array([obj], dtype="m8[ns]")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "exp",
        [
            IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True])),
            IntegerArray(np.array([2, 0], dtype="int64"), np.array([False, True])),
        ],
    )
    def test_maybe_convert_objects_nullable_integer(self, exp):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over two equivalent expected
        # nullable integer arrays (two spellings of int64). An integer with NaN converts
        # with the nullable option to a nullable integer array with the NaN masked.
        # GH27335
        arr = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)

        tm.assert_extension_array_equal(result, exp)

    @pytest.mark.parametrize(
        "dtype, val", [("int64", 1), ("uint64", np.iinfo(np.int64).max + 1)]
    )
    def test_maybe_convert_objects_nullable_none(self, dtype, val):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over int64 and a value only
        # representable as uint64. Integers with a none element convert with the
        # nullable option to a nullable integer array with that element masked.
        # GH#50043
        arr = np.array([val, None, 3], dtype="object")
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        expected = IntegerArray(
            np.array([val, 0, 3], dtype=dtype), np.array([False, True, False])
        )
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (True, IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True]))),
            (False, np.array([2, np.nan], dtype="float64")),
        ],
    )
    def test_maybe_convert_numeric_nullable_integer(
        self, convert_to_masked_nullable, exp
    ):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over the masked-nullable option.
        # An object array of an integer and NaN converts: with the option the
        # (data, mask) result builds an equal nullable integer array; without it a float
        # array with NaN.
        # GH 40687
        arr = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            result = IntegerArray(*result)
            tm.assert_extension_array_equal(result, exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                FloatingArray(
                    np.array([2.0, 0.0], dtype="float64"), np.array([False, True])
                ),
            ),
            (False, np.array([2.0, np.nan], dtype="float64")),
        ],
    )
    def test_maybe_convert_numeric_floating_array(
        self, convert_to_masked_nullable, exp
    ):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; same as above but with float input: with
        # the option the pair builds an equal nullable float array, else a float array with NaN.
        # GH 40687
        arr = np.array([2.0, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(FloatingArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    def test_maybe_convert_objects_bool_nan(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue: booleans with NaN in an object array stay
        # an object array when converting in safe mode.
        # GH32146
        ind = Index([True, False, np.nan], dtype=object)
        exp = np.array([True, False, np.nan], dtype=object)
        out = lib.maybe_convert_objects(ind.values, safe=1)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_nullable_boolean(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. A boolean object array converts with the
        # nullable option to a nullable boolean array; with a NaT mixed in it stays an
        # object array.
        # GH50047
        arr = np.array([True, False], dtype=object)
        exp = BooleanArray._from_sequence([True, False], dtype="boolean")
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(out, exp)

        arr = np.array([True, False, pd.NaT], dtype=object)
        exp = np.array([True, False, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_numpy_array_equal(out, exp)

    @pytest.mark.parametrize("val", [None, np.nan])
    def test_maybe_convert_objects_nullable_boolean_na(self, val):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over none and NaN. Booleans with
        # a missing element convert with the nullable option to a nullable boolean
        # array with that element masked.
        # GH50047
        arr = np.array([True, False, val], dtype=object)
        exp = BooleanArray(
            np.array([True, False, False]), np.array([False, False, True])
        )
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(out, exp)

    @pytest.mark.parametrize(
        "data0",
        [
            True,
            1,
            1.0,
            1.0 + 1.0j,
            np.int8(1),
            np.int16(1),
            np.int32(1),
            np.int64(1),
            np.float16(1),
            np.float32(1),
            np.float64(1),
            np.complex64(1),
            np.complex128(1),
        ],
    )
    @pytest.mark.parametrize(
        "data1",
        [
            True,
            1,
            1.0,
            1.0 + 1.0j,
            np.int8(1),
            np.int16(1),
            np.int32(1),
            np.int64(1),
            np.float16(1),
            np.float32(1),
            np.float64(1),
            np.complex64(1),
            np.complex128(1),
        ],
    )
    def test_maybe_convert_objects_itemsize(self, data0, data1):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over all pairs of Python and
        # numpy booleans, ints, floats and complex values. The converted array's dtype
        # follows numpy promotion: for two numpy scalars their common kind and larger
        # item size; with Python scalars, 8 bytes (16 for complex); a boolean mixed with a
        # non-boolean stays object; two booleans stay bool.
        # GH 40908
        data = [data0, data1]
        arr = np.array(data, dtype="object")

        common_kind = np.result_type(type(data0), type(data1)).kind
        kind0 = "python" if not hasattr(data0, "dtype") else data0.dtype.kind
        kind1 = "python" if not hasattr(data1, "dtype") else data1.dtype.kind
        if kind0 != "python" and kind1 != "python":
            kind = common_kind
            itemsize = max(data0.dtype.itemsize, data1.dtype.itemsize)
        elif is_bool(data0) or is_bool(data1):
            kind = "bool" if (is_bool(data0) and is_bool(data1)) else "object"
            itemsize = ""
        elif is_complex(data0) or is_complex(data1):
            kind = common_kind
            itemsize = 16
        else:
            kind = common_kind
            itemsize = 8

        expected = np.array(data, dtype=f"{kind}{itemsize}")
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_dtypes_remain_object_array(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue: an object array of a tz-aware datetime and an
        # integer is returned unchanged by the conversion.
        # GH14956
        arr = np.array([datetime(2015, 1, 1, tzinfo=timezone.utc), 1], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(result, arr)

    @pytest.mark.parametrize(
        "idx",
        [
            pd.IntervalIndex.from_breaks(range(5), closed="both"),
            pd.period_range("2016-01-01", periods=3, freq="D"),
        ],
    )
    def test_maybe_convert_objects_ea(self, idx):
        # [FUNCTION DOC]
        # Parametrized over an interval index and a daily period range. Converting the
        # object array of their elements with non-numeric conversion recovers the
        # original extension array.
        result = lib.maybe_convert_objects(
            np.array(idx, dtype=object),
            convert_non_numeric=True,
        )
        tm.assert_extension_array_equal(result, idx._data)


class TestTypeInference:
    # [CLASS DOC]
    # Tests for the dtype name inference routine (the lib function that returns
    # strings such as integer, floating, mixed, string, datetime and so on) across
    # many input kinds. Holds an empty nested helper class used as an arbitrary object.
    # Dummy class used for testing with Python objects
    class Dummy:
        # [CLASS DOC]
        # HELPER: nested empty class used as an arbitrary Python object in inference tests.
        pass

    def test_inferred_dtype_fixture(self, any_skipna_inferred_dtype):
        # [FUNCTION DOC]
        # Fixture any_skipna_inferred_dtype (defined in the main conftest) gives a pair of
        # expected dtype name and sample values. Inference with missing values skipped on
        # the values returns exactly that name.
        # see pandas/conftest.py
        inferred_dtype, values = any_skipna_inferred_dtype

        # make sure the inferred dtype of the fixture is as requested
        assert inferred_dtype == lib.infer_dtype(values, skipna=True)

    def test_length_zero(self, skipna):
        # [FUNCTION DOC]
        # Fixture skipna. An empty int32 array infers integer; an empty list infers
        # empty; an array of two empty arrays infers empty (regression test for a GitHub
        # issue).
        result = lib.infer_dtype(np.array([], dtype="i4"), skipna=skipna)
        assert result == "integer"

        result = lib.infer_dtype([], skipna=skipna)
        assert result == "empty"

        # GH 18004
        arr = np.array([np.array([], dtype=object), np.array([], dtype=object)])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "empty"

    def test_integers(self):
        # [FUNCTION DOC]
        # Python and numpy integers in an object array infer integer; adding a string
        # infers mixed-integer; an int32 array infers integer.
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "integer"

        arr = np.array([1, 2, 3, np.int64(4), np.int32(5), "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed-integer"

        arr = np.array([1, 2, 3, 4, 5], dtype="i4")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "integer"

    @pytest.mark.parametrize(
        "arr, skipna",
        [
            ([1, 2, np.nan, np.nan, 3], False),
            ([1, 2, np.nan, np.nan, 3], True),
            ([1, 2, 3, np.int64(4), np.int32(5), np.nan], False),
            ([1, 2, 3, np.int64(4), np.int32(5), np.nan], True),
        ],
    )
    def test_integer_na(self, arr, skipna):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over integer lists with NaN
        # and skipna. Result is integer when skipping missing values, else integer-na.
        # GH 27392
        result = lib.infer_dtype(np.array(arr, dtype="O"), skipna=skipna)
        expected = "integer" if skipna else "integer-na"
        assert result == expected

    def test_infer_dtype_skipna_default(self):
        # [FUNCTION DOC]
        # Calling without the skip option on integers with NaN infers integer (the
        # default became skipping after a deprecation).
        # infer_dtype `skipna` default deprecated in GH#24050,
        #  changed to True in GH#29876
        arr = np.array([1, 2, 3, np.nan], dtype=object)

        result = lib.infer_dtype(arr)
        assert result == "integer"

    def test_bools(self):
        # [FUNCTION DOC]
        # Booleans (Python or numpy, object or bool arrays) infer boolean; with a string
        # infers mixed; with NaN infers boolean when skipping and mixed otherwise.
        arr = np.array([True, False, True, True, True], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        arr = np.array([np.bool_(True), np.bool_(False)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        arr = np.array([True, False, True, "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        arr = np.array([True, False, True], dtype=bool)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        arr = np.array([True, np.nan, False], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        result = lib.infer_dtype(arr, skipna=False)
        assert result == "mixed"

    def test_floats(self):
        # [FUNCTION DOC]
        # Floats (Python and numpy) infer floating; mixing in a string infers
        # mixed-integer; float32 and float64 arrays infer floating.
        arr = np.array([1.0, 2.0, 3.0, np.float64(4), np.float32(5)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "floating"

        arr = np.array([1, 2, 3, np.float64(4), np.float32(5), "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed-integer"

        arr = np.array([1, 2, 3, 4, 5], dtype="f4")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "floating"

        arr = np.array([1, 2, 3, 4, 5], dtype="f8")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "floating"

    def test_decimals(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Decimals infer decimal; floats mixed with a
        # decimal infer mixed in either order; decimals with decimal NaN or float NaN
        # still infer decimal.
        # GH15690
        arr = np.array([Decimal(1), Decimal(2), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "decimal"

        arr = np.array([1.0, 2.0, Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        result = lib.infer_dtype(arr[::-1], skipna=True)
        assert result == "mixed"

        arr = np.array([Decimal(1), Decimal("NaN"), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "decimal"

        arr = np.array([Decimal(1), np.nan, Decimal(3)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "decimal"

    # complex is compatible with nan, so skipna has no effect
    def test_complex(self, skipna):
        # [FUNCTION DOC]
        # Fixture skipna (has no effect since complex is compatible with NaN). A numpy
        # array holding floats and a complex gets cast to complex and infers complex; the
        # same as an object array infers mixed in either order.
        # gets cast to complex on array construction
        arr = np.array([1.0, 2.0, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

        arr = np.array([1.0, 2.0, 1 + 1j], dtype="O")
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "mixed"

        result = lib.infer_dtype(arr[::-1], skipna=skipna)
        assert result == "mixed"

        # gets cast to complex on array construction
        arr = np.array([1, np.nan, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

        arr = np.array([1.0, np.nan, 1 + 1j], dtype="O")
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "mixed"

        # complex with nans stays complex
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype="O")
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

        # test smaller complex dtype; will pass through _try_infer_map fastpath
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype=np.complex64)
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

    def test_string(self):
        # [FUNCTION DOC]
        # KNOWN ISSUE: the body is only a pass statement, so this placeholder test verifies nothing.
        pass

    def test_unicode(self):
        # [FUNCTION DOC]
        # Strings with NaN infer mixed when not skipping missing values (noted as
        # possibly not optimal) and string when skipping; with the NA singleton and
        # skipping also string; with NaT it is mixed; plain strings infer string.
        arr = ["a", np.nan, "c"]
        result = lib.infer_dtype(arr, skipna=False)
        # This currently returns "mixed", but it's not clear that's optimal.
        # This could also return "string" or "mixed-string"
        assert result == "mixed"

        # even though we use skipna, we are only skipping those NAs that are
        #  considered matching by is_string_array
        arr = ["a", np.nan, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "string"

        arr = ["a", pd.NA, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "string"

        arr = ["a", pd.NaT, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        arr = ["a", "c"]
        result = lib.infer_dtype(arr, skipna=False)
        assert result == "string"

    @pytest.mark.parametrize(
        "dtype, missing, skipna, expected",
        [
            (float, np.nan, False, "floating"),
            (float, np.nan, True, "floating"),
            (object, np.nan, False, "floating"),
            (object, np.nan, True, "empty"),
            (object, None, False, "mixed"),
            (object, None, True, "empty"),
        ],
    )
    @pytest.mark.parametrize("box", [Series, np.array])
    def test_object_empty(self, box, missing, dtype, skipna, expected):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; parametrized over container (Series or
        # numpy array) and (dtype, missing value, skipna, expected) rows. Two missing
        # values: float dtype always infers floating; object dtype with NaN infers floating
        # if not skipping and empty if skipping; with none infers mixed or empty.
        # GH 23421
        arr = box([missing, missing], dtype=dtype)

        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == expected

    def test_datetime(self):
        # [FUNCTION DOC]
        # An Index of python datetimes has the inferred type datetime64.
        dates = [datetime(2012, 1, x) for x in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == "datetime64"

    def test_infer_dtype_datetime64(self):
        # [FUNCTION DOC]
        # An object array of numpy datetime64 values infers datetime64.
        arr = np.array(
            [np.datetime64("2011-01-01"), np.datetime64("2011-01-01")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    def test_infer_dtype_datetime64_with_na(self, na_value):
        # [FUNCTION DOC]
        # Parametrized over NaT and NaN. Datetime64 values with a leading and/or
        # trailing missing value infer datetime64 when skipping.
        # starts with nan
        arr = np.array([na_value, np.datetime64("2011-01-02")])
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

        arr = np.array([na_value, np.datetime64("2011-01-02"), na_value])
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array(
                [np.timedelta64("NaT", "ns"), np.datetime64("2011-01-02")], dtype=object
            ),
            np.array(
                [np.datetime64("2011-01-02"), np.timedelta64("NaT", "ns")], dtype=object
            ),
            np.array([np.datetime64("2011-01-01"), Timestamp("2011-01-02")]),
            np.array([Timestamp("2011-01-02"), np.datetime64("2011-01-01")]),
            np.array([np.nan, Timestamp("2011-01-02"), 1.1]),
            np.array([np.nan, "2011-01-01", Timestamp("2011-01-02")], dtype=object),
            np.array(
                [np.datetime64("nat", "ns"), np.timedelta64(1, "D")], dtype=object
            ),
            np.array(
                [np.timedelta64(1, "D"), np.datetime64("nat", "ns")], dtype=object
            ),
        ],
    )
    def test_infer_datetimelike_dtype_mixed(self, arr):
        # [FUNCTION DOC]
        # Parametrized over arrays mixing timedelta and datetime64, datetime64 and
        # Timestamp, Timestamp with a float or string, and timedelta with datetime NaT.
        # Without skipping missing values the result is mixed.
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    def test_infer_dtype_mixed_integer(self):
        # [FUNCTION DOC]
        # NaN, a Timestamp and an integer infer mixed-integer when skipping missing values.
        arr = np.array([np.nan, Timestamp("2011-01-02"), 1])
        assert lib.infer_dtype(arr, skipna=True) == "mixed-integer"

    @pytest.mark.parametrize(
        "arr",
        [
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")],
            [datetime(2011, 1, 1), datetime(2012, 2, 1)],
            [datetime(2011, 1, 1), Timestamp("2011-01-02")],
        ],
    )
    def test_infer_dtype_datetime(self, arr):
        # [FUNCTION DOC]
        # Parametrized over arrays of Timestamps, python datetimes, and both mixed. Inferred
        # type is datetime.
        assert lib.infer_dtype(np.array(arr), skipna=True) == "datetime"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    @pytest.mark.parametrize(
        "time_stamp", [Timestamp("2011-01-01"), datetime(2011, 1, 1)]
    )
    def test_infer_dtype_datetime_with_na(self, na_value, time_stamp):
        # [FUNCTION DOC]
        # Parametrized over NaT/NaN and Timestamp/datetime. Leading and/or trailing
        # missing values infer datetime when skipping.
        # starts with nan
        arr = np.array([na_value, time_stamp])
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

        arr = np.array([na_value, time_stamp, na_value])
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([Timedelta("1 days"), Timedelta("2 days")]),
            np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")], dtype=object),
            np.array([timedelta(1), timedelta(2)]),
        ],
    )
    def test_infer_dtype_timedelta(self, arr):
        # [FUNCTION DOC]
        # Parametrized over Timedelta, numpy timedelta64 (object) and python timedelta
        # arrays. Inferred type is timedelta.
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    @pytest.mark.parametrize(
        "delta", [Timedelta("1 days"), np.timedelta64(1, "D"), timedelta(1)]
    )
    def test_infer_dtype_timedelta_with_na(self, na_value, delta):
        # [FUNCTION DOC]
        # Parametrized over NaT/NaN and three timedelta kinds. Leading and/or trailing
        # missing values infer timedelta when skipping.
        # starts with nan
        arr = np.array([na_value, delta])
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

        arr = np.array([na_value, delta, na_value])
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

    def test_infer_dtype_period(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Periods with equal frequency infer period;
        # with differing frequencies infer mixed.
        # GH 13664
        arr = np.array([Period("2011-01", freq="D"), Period("2011-02", freq="D")])
        assert lib.infer_dtype(arr, skipna=True) == "period"

        # non-homogeneous freqs -> mixed
        arr = np.array([Period("2011-01", freq="D"), Period("2011-02", freq="M")])
        assert lib.infer_dtype(arr, skipna=True) == "mixed"

    def test_infer_dtype_period_array(self, index_or_series_or_array, skipna):
        # [FUNCTION DOC]
        # Fixtures index_or_series_or_array (container constructor) and skipna. Daily
        # periods with NaT infer period; periods with differing frequency infer mixed,
        # except when the container is the generic array constructor (it wraps into a
        # numpy extension array) which infers unknown-array. Regression test for a GitHub issue.
        klass = index_or_series_or_array
        # https://github.com/pandas-dev/pandas/issues/23553
        values = klass(
            [
                Period("2011-01-01", freq="D"),
                Period("2011-01-02", freq="D"),
                pd.NaT,
            ]
        )
        assert lib.infer_dtype(values, skipna=skipna) == "period"

        # periods but mixed freq
        values = klass(
            [
                Period("2011-01-01", freq="D"),
                Period("2011-01-02", freq="M"),
                pd.NaT,
            ]
        )
        # with pd.array this becomes NumpyExtensionArray which ends up
        #  as "unknown-array"
        exp = "unknown-array" if klass is pd.array else "mixed"
        assert lib.infer_dtype(values, skipna=skipna) == exp

    def test_infer_dtype_period_mixed(self):
        # [FUNCTION DOC]
        # Object arrays mixing a period and a datetime NaT in either order infer mixed
        # without skipping.
        arr = np.array(
            [Period("2011-01", freq="M"), np.datetime64("nat", "ns")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array(
            [np.datetime64("nat", "ns"), Period("2011-01", freq="M")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    def test_infer_dtype_period_with_na(self, na_value):
        # [FUNCTION DOC]
        # Parametrized over NaT/NaN. Leading and/or trailing missing values infer period
        # when skipping.
        # starts with nan
        arr = np.array([na_value, Period("2011-01", freq="D")])
        assert lib.infer_dtype(arr, skipna=True) == "period"

        arr = np.array([na_value, Period("2011-01", freq="D"), na_value])
        assert lib.infer_dtype(arr, skipna=True) == "period"

    @pytest.mark.parametrize("na_value", [pd.NA, np.nan])
    def test_infer_dtype_numeric_with_na(self, na_value):
        # [FUNCTION DOC]
        # Regression tests for GitHub issues; parametrized over NA and NaN. Object
        # Series of integers, floats and complex numbers with a missing value infer
        # integer, floating and complex respectively when skipping.
        # GH61621
        ser = Series([1, 2, na_value], dtype=object)
        assert lib.infer_dtype(ser, skipna=True) == "integer"

        ser = Series([1.0, 2.0, na_value], dtype=object)
        assert lib.infer_dtype(ser, skipna=True) == "floating"

        # GH#61976
        ser = Series([1 + 1j, na_value], dtype=object)
        assert lib.infer_dtype(ser, skipna=True) == "complex"

    def test_infer_dtype_all_nan_nat_like(self):
        # [FUNCTION DOC]
        # A battery of arrays of NaN, none and NaT-like values: all NaN infers floating;
        # NaN with none infers empty when skipping, mixed otherwise; various NaT
        # arrays infer datetime; datetime64 NaT with missing values infers datetime64;
        # timedelta64 NaT infers timedelta; datetime and timedelta NaTs together
        # infer mixed.
        arr = np.array([np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == "floating"

        # nan and None mix are result in mixed
        arr = np.array([np.nan, np.nan, None])
        assert lib.infer_dtype(arr, skipna=True) == "empty"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([None, np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == "empty"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # pd.NaT
        arr = np.array([pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([np.nan, pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([np.nan, pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([None, pd.NaT, None])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # np.datetime64(nat)
        arr = np.array([np.datetime64("nat", "ns")])
        assert lib.infer_dtype(arr, skipna=False) == "datetime64"

        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.datetime64("nat", "ns"), n])
            assert lib.infer_dtype(arr, skipna=False) == "datetime64"

            arr = np.array([pd.NaT, n, np.datetime64("nat", "ns"), n])
            assert lib.infer_dtype(arr, skipna=False) == "datetime64"

        arr = np.array([np.timedelta64("NaT", "ns")], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "timedelta"

        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.timedelta64("NaT", "ns"), n])
            assert lib.infer_dtype(arr, skipna=False) == "timedelta"

            arr = np.array([pd.NaT, n, np.timedelta64("NaT", "ns"), n])
            assert lib.infer_dtype(arr, skipna=False) == "timedelta"

        # datetime / timedelta mixed
        arr = np.array(
            [pd.NaT, np.datetime64("nat", "ns"), np.timedelta64("NaT", "ns"), np.nan]
        )
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array(
            [np.timedelta64("NaT", "ns"), np.datetime64("nat", "ns")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    def test_is_datetimelike_array_all_nan_nat_like(self):
        # [FUNCTION DOC]
        # Array predicates for datetime, datetime64 and timedelta on arrays of missing
        # values: NaN/NaT with datetime64 NaT is a datetime array but not timedelta;
        # with timedelta NaT the reverse; mixing both kinds matches neither; NaN and NaT
        # only match all three; NaN only matches none. The single-timezone datetime
        # predicate is true for same-zone Timestamps and false for different zones.
        arr = np.array([np.nan, pd.NaT, np.datetime64("nat", "ns")])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, pd.NaT, np.timedelta64("NaT", "ns")])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array(
            [np.nan, pd.NaT, np.datetime64("nat", "ns"), np.timedelta64("NaT", "ns")]
        )
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, pd.NaT])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, np.nan], dtype=object)
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        assert lib.is_datetime_with_singletz_array(
            np.array(
                [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130102", tz="US/Eastern"),
                ],
                dtype=object,
            )
        )
        assert not lib.is_datetime_with_singletz_array(
            np.array(
                [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130102", tz="CET"),
                ],
                dtype=object,
            )
        )

    @pytest.mark.parametrize(
        "func",
        [
            "is_datetime_array",
            "is_datetime64_array",
            "is_bool_array",
            "is_timedelta_or_timedelta64_array",
            "is_date_array",
            "is_time_array",
            "is_interval_array",
        ],
    )
    def test_other_dtypes_for_array(self, func):
        # [FUNCTION DOC]
        # Parametrized over predicate names (datetime, datetime64, bool, timedelta, date,
        # time, interval arrays). Each returns false for string and integer arrays, in
        # one-dimensional and column-shaped two-dimensional form.
        func = getattr(lib, func)
        arr = np.array(["foo", "bar"])
        assert not func(arr)
        assert not func(arr.reshape(2, 1))

        arr = np.array([1, 2])
        assert not func(arr)
        assert not func(arr.reshape(2, 1))

    def test_date(self):
        # [FUNCTION DOC]
        # An Index of dates has inferred type date. Dates plus NaN infer mixed without
        # skipping and date with skipping.
        dates = [date(2012, 1, day) for day in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == "date"

        dates = [date(2012, 1, day) for day in range(1, 20)] + [np.nan]
        result = lib.infer_dtype(dates, skipna=False)
        assert result == "mixed"

        result = lib.infer_dtype(dates, skipna=True)
        assert result == "date"

    @pytest.mark.parametrize(
        "values",
        [
            [date(2020, 1, 1), Timestamp("2020-01-01")],
            [Timestamp("2020-01-01"), date(2020, 1, 1)],
            [date(2020, 1, 1), pd.NaT],
            [pd.NaT, date(2020, 1, 1)],
        ],
    )
    def test_infer_dtype_date_order_invariant(self, values, skipna):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue; fixture skipna. Dates mixed with a
        # Timestamp or NaT infer date regardless of order.
        # https://github.com/pandas-dev/pandas/issues/33741
        result = lib.infer_dtype(values, skipna=skipna)
        assert result == "date"

    def test_is_numeric_array(self):
        # [FUNCTION DOC]
        # The float array predicate is true for floats with or without NaN and false
        # for integers; the integer array predicate is true for integers and false for floats.
        assert lib.is_float_array(np.array([1, 2.0]))
        assert lib.is_float_array(np.array([1, 2.0, np.nan]))
        assert not lib.is_float_array(np.array([1, 2]))

        assert lib.is_integer_array(np.array([1, 2]))
        assert not lib.is_integer_array(np.array([1, 2.0]))

    def test_is_string_array(self):
        # [FUNCTION DOC]
        # String array predicate: true for strings; with skipping, NA, none and NaN are
        # accepted as missing but NaT, datetime64 NaT and decimal NaN are not; without
        # skipping any missing value makes it false; integers are false.
        # We should only be accepting pd.NA, np.nan,
        # other floating point nans e.g. float('nan')]
        # when skipna is True.
        assert lib.is_string_array(np.array(["foo", "bar"]))
        assert not lib.is_string_array(
            np.array(["foo", "bar", pd.NA], dtype=object), skipna=False
        )
        assert lib.is_string_array(
            np.array(["foo", "bar", pd.NA], dtype=object), skipna=True
        )
        # we allow NaN/None in the StringArray constructor, so its allowed here
        assert lib.is_string_array(
            np.array(["foo", "bar", None], dtype=object), skipna=True
        )
        assert lib.is_string_array(
            np.array(["foo", "bar", np.nan], dtype=object), skipna=True
        )
        # But not e.g. datetimelike or Decimal NAs
        assert not lib.is_string_array(
            np.array(["foo", "bar", pd.NaT], dtype=object), skipna=True
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", np.datetime64("NaT", "ns")], dtype=object),
            skipna=True,
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", Decimal("NaN")], dtype=object), skipna=True
        )

        assert not lib.is_string_array(
            np.array(["foo", "bar", None], dtype=object), skipna=False
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", np.nan], dtype=object), skipna=False
        )
        assert not lib.is_string_array(np.array([1, 2]))

    def test_is_interval_array_subclass(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. An interval subclass built from timestamp
        # strings with an extra property is not recognized by the interval array
        # predicate or dtype inference as plain intervals, yet a Series keeps the
        # subclass type for its element.
        # GH#46945

        class TimestampsInterval(Interval):
            # [CLASS DOC]
            # HELPER: nested interval subclass used by the preceding test; its constructor
            # takes string endpoints and converts them to Timestamps; adds a seconds property
            # giving the length in seconds.
            def __init__(self, left: str, right: str, closed="both") -> None:
                # [FUNCTION DOC]
                # HELPER: parameters left and right (strings parsed to Timestamps) and closed
                # (default both); calls the parent constructor with the Timestamps.
                super().__init__(Timestamp(left), Timestamp(right), closed)

            @property
            def seconds(self) -> float:
                # [FUNCTION DOC]
                # HELPER: property returning the seconds component of the interval's length.
                return self.length.seconds

        item = TimestampsInterval("1970-01-01 00:00:00", "1970-01-01 00:00:01")
        arr = np.array([item], dtype=object)
        assert not lib.is_interval_array(arr)
        assert lib.infer_dtype(arr) != "interval"
        out = Series([item])[0]
        assert isinstance(out, TimestampsInterval)

    @pytest.mark.parametrize(
        "func",
        [
            "is_bool_array",
            "is_date_array",
            "is_datetime_array",
            "is_datetime64_array",
            "is_float_array",
            "is_integer_array",
            "is_interval_array",
            "is_string_array",
            "is_time_array",
            "is_timedelta_or_timedelta64_array",
        ],
    )
    def test_is_dtype_array_empty_obj(self, func):
        # [FUNCTION DOC]
        # Regression test for a GitHub pull request; parametrized over the array
        # predicate names. Empty two-dimensional object arrays (shape two by zero and zero
        # by two) give false.
        # https://github.com/pandas-dev/pandas/pull/60796
        func = getattr(lib, func)

        arr = np.empty((2, 0), dtype=object)
        assert not func(arr)

        arr = np.empty((0, 2), dtype=object)
        assert not func(arr)

    def test_to_object_array_tuples(self):
        # [FUNCTION DOC]
        # Converting a list holding a tuple, and one holding a named tuple, to an object
        # array does not raise.
        r = (5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

        # make sure record array works
        record = namedtuple("record", "x y")
        r = record(5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

    def test_object(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue: a single none in an object array infers
        # mixed without skipping, empty with skipping.
        # GH 7431
        # cannot infer more than this as only a single element
        arr = np.array([None], dtype="O")
        result = lib.infer_dtype(arr, skipna=False)
        assert result == "mixed"
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "empty"

    def test_to_object_array_width(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue. Converting rows to a 2-D object array gives
        # the same data; with a minimum width of one is the same; with minimum width five
        # pads each row with none.
        # see gh-13320
        rows = [[1, 2, 3], [4, 5, 6]]

        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows)
        tm.assert_numpy_array_equal(out, expected)

        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows, min_width=1)
        tm.assert_numpy_array_equal(out, expected)

        expected = np.array(
            [[1, 2, 3, None, None], [4, 5, 6, None, None]], dtype=object
        )
        out = lib.to_object_array(rows, min_width=5)
        tm.assert_numpy_array_equal(out, expected)

    def test_categorical(self):
        # [FUNCTION DOC]
        # Regression test for a GitHub issue: categoricals and Series of them (including all
        # missing values) infer categorical.
        # GH 8974
        arr = Categorical(list("abc"))
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "categorical"

        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == "categorical"

        arr = Categorical([None, None, None], categories=["cegfab"], ordered=True)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "categorical"

        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == "categorical"

    @pytest.mark.parametrize("asobject", [True, False])
    def test_interval(self, asobject):
        # [FUNCTION DOC]
        # Parametrized over whether the interval index is cast to object. Inference gives
        # interval for the index, its data array and a Series with the interval dtype.
        idx = pd.IntervalIndex.from_breaks(range(5), closed="both")
        if asobject:
            idx = idx.astype(object)

        inferred = lib.infer_dtype(idx, skipna=False)
        assert inferred == "interval"

        inferred = lib.infer_dtype(idx._data, skipna=False)
        assert inferred == "interval"

        inferred = lib.infer_dtype(Series(idx, dtype=idx.dtype), skipna=False)
        assert inferred == "interval"

    @pytest.mark.parametrize("value", [Timestamp(0), Timedelta(0), 0, 0.0])
    def test_interval_mismatched_closed(self, value):
        # [FUNCTION DOC]
        # Parametrized over Timestamp, Timedelta, integer and float endpoints. Intervals
        # with matching closure infer interval; with differing closure infer mixed.
        first = Interval(value, value, closed="left")
        second = Interval(value, value, closed="right")

        # if closed match, we should infer "interval"
        arr = np.array([first, first], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "interval"

        # if closed dont match, we should _not_ get "interval"
        arr2 = np.array([first, second], dtype=object)
        assert lib.infer_dtype(arr2, skipna=False) == "mixed"

    def test_interval_mismatched_subtype(self):
        # [FUNCTION DOC]
        # Intervals mixing integer, Timestamp and Timedelta subtypes infer mixed; integer
        # with float interval infers interval (compatible).
        first = Interval(0, 1, closed="left")
        second = Interval(Timestamp(0), Timestamp(1), closed="left")
        third = Interval(Timedelta(0), Timedelta(1), closed="left")

        arr = np.array([first, second])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([second, third])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([first, third])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # float vs int subdtype are compatible
        flt_interval = Interval(1.5, 2.5, closed="left")
        arr = np.array([first, flt_interval], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "interval"

    @pytest.mark.parametrize("data", [["a", "b", "c"], ["a", "b", pd.NA]])
    def test_string_dtype(
        self, data, skipna, index_or_series_or_array, nullable_string_dtype
    ):
        # [FUNCTION DOC]
        # Parametrized over strings with and without NA; fixtures skipna,
        # index_or_series_or_array and nullable_string_dtype. A container with a
        # nullable string dtype infers string.
        # StringArray
        val = index_or_series_or_array(data, dtype=nullable_string_dtype)
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == "string"

    @pytest.mark.parametrize("data", [[True, False, True], [True, False, pd.NA]])
    def test_boolean_dtype(self, data, skipna, index_or_series_or_array):
        # [FUNCTION DOC]
        # Parametrized over booleans with and without NA; fixtures skipna and
        # index_or_series_or_array. A container with the nullable boolean dtype infers
        # boolean.
        # BooleanArray
        val = index_or_series_or_array(data, dtype="boolean")
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == "boolean"


class TestNumberScalar:
    # [CLASS DOC]
    # Tests of scalar type predicates (number, bool, integer, float, datetime and
    # timedelta dtype predicates). No state.
    def test_is_number(self):
        # [FUNCTION DOC]
        # True for booleans, ints, floats, complex, numpy numeric scalars, NaN, and numpy
        # timedelta; false for none, string, datetimes, timedeltas of python/pandas
        # type and the numpy bool (noted as questionable).
        assert is_number(True)
        assert is_number(1)
        assert is_number(1.1)
        assert is_number(1 + 3j)
        assert is_number(np.int64(1))
        assert is_number(np.float64(1.1))
        assert is_number(np.complex128(1 + 3j))
        assert is_number(np.nan)

        assert not is_number(None)
        assert not is_number("x")
        assert not is_number(datetime(2011, 1, 1))
        assert not is_number(np.datetime64("2011-01-01"))
        assert not is_number(Timestamp("2011-01-01"))
        assert not is_number(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_number(timedelta(1000))
        assert not is_number(Timedelta("1 days"))

        # questionable
        assert not is_number(np.bool_(False))
        assert is_number(np.timedelta64(1, "D"))

    def test_is_bool(self):
        # [FUNCTION DOC]
        # True for Python and numpy booleans; false for numbers, NaN, none, string,
        # dates and times.
        assert is_bool(True)
        assert is_bool(False)
        assert is_bool(np.bool_(False))

        assert not is_bool(1)
        assert not is_bool(1.1)
        assert not is_bool(1 + 3j)
        assert not is_bool(np.int64(1))
        assert not is_bool(np.float64(1.1))
        assert not is_bool(np.complex128(1 + 3j))
        assert not is_bool(np.nan)
        assert not is_bool(None)
        assert not is_bool("x")
        assert not is_bool(datetime(2011, 1, 1))
        assert not is_bool(np.datetime64("2011-01-01"))
        assert not is_bool(Timestamp("2011-01-01"))
        assert not is_bool(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_bool(timedelta(1000))
        assert not is_bool(np.timedelta64(1, "D"))
        assert not is_bool(Timedelta("1 days"))

    def test_is_integer(self):
        # [FUNCTION DOC]
        # True for Python int and numpy int; false for booleans, floats, complex, NaN,
        # none, strings and date/time values.
        assert is_integer(1)
        assert is_integer(np.int64(1))

        assert not is_integer(True)
        assert not is_integer(1.1)
        assert not is_integer(1 + 3j)
        assert not is_integer(False)
        assert not is_integer(np.bool_(False))
        assert not is_integer(np.float64(1.1))
        assert not is_integer(np.complex128(1 + 3j))
        assert not is_integer(np.nan)
        assert not is_integer(None)
        assert not is_integer("x")
        assert not is_integer(datetime(2011, 1, 1))
        assert not is_integer(np.datetime64("2011-01-01"))
        assert not is_integer(Timestamp("2011-01-01"))
        assert not is_integer(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_integer(timedelta(1000))
        assert not is_integer(Timedelta("1 days"))
        assert not is_integer(np.timedelta64(1, "D"))

    def test_is_float(self):
        # [FUNCTION DOC]
        # True for floats and NaN; false for booleans, integers, complex, none, strings
        # and date/time values.
        assert is_float(1.1)
        assert is_float(np.float64(1.1))
        assert is_float(np.nan)

        assert not is_float(True)
        assert not is_float(1)
        assert not is_float(1 + 3j)
        assert not is_float(False)
        assert not is_float(np.bool_(False))
        assert not is_float(np.int64(1))
        assert not is_float(np.complex128(1 + 3j))
        assert not is_float(None)
        assert not is_float("x")
        assert not is_float(datetime(2011, 1, 1))
        assert not is_float(np.datetime64("2011-01-01"))
        assert not is_float(Timestamp("2011-01-01"))
        assert not is_float(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_float(timedelta(1000))
        assert not is_float(np.timedelta64(1, "D"))
        assert not is_float(Timedelta("1 days"))

    def test_is_datetime_dtypes(self):
        # [FUNCTION DOC]
        # Naive and tz-aware date ranges. The datetime64 predicate is true for the generic
        # and nanosecond names and naive range but false for tz-aware; the nanosecond
        # predicate is false for the generic name; the any-datetime predicate is true for all;
        # the deprecated tz predicate warns and is true only for the tz-aware range.
        ts = pd.date_range("20130101", periods=3, unit="ns")
        tsa = pd.date_range("20130101", periods=3, tz="US/Eastern", unit="ns")

        msg = "is_datetime64tz_dtype is deprecated"

        assert is_datetime64_dtype("datetime64")
        assert is_datetime64_dtype("datetime64[ns]")
        assert is_datetime64_dtype(ts)
        assert not is_datetime64_dtype(tsa)

        assert not is_datetime64_ns_dtype("datetime64")
        assert is_datetime64_ns_dtype("datetime64[ns]")
        assert is_datetime64_ns_dtype(ts)
        assert is_datetime64_ns_dtype(tsa)

        assert is_datetime64_any_dtype("datetime64")
        assert is_datetime64_any_dtype("datetime64[ns]")
        assert is_datetime64_any_dtype(ts)
        assert is_datetime64_any_dtype(tsa)

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            assert not is_datetime64tz_dtype("datetime64")
            assert not is_datetime64tz_dtype("datetime64[ns]")
            assert not is_datetime64tz_dtype(ts)
            assert is_datetime64tz_dtype(tsa)

    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC"])
    def test_is_datetime_dtypes_with_tz(self, tz):
        # [FUNCTION DOC]
        # Parametrized over zones. For a tz-aware dtype string: not datetime64; the
        # deprecated tz predicate warns and is true; nanosecond and any-datetime predicates are
        # true.
        dtype = f"datetime64[ns, {tz}]"
        assert not is_datetime64_dtype(dtype)

        msg = "is_datetime64tz_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_any_dtype(dtype)

    def test_is_timedelta(self):
        # [FUNCTION DOC]
        # The timedelta64 predicate is true for the generic and nanosecond names; the
        # nanosecond predicate is false for the generic name and true for the nanosecond
        # name; a nanosecond timedelta index passes both predicates (also after
        # casting); float and integer empty indexes are not nanosecond timedelta.
        assert is_timedelta64_dtype("timedelta64")
        assert is_timedelta64_dtype("timedelta64[ns]")
        assert not is_timedelta64_ns_dtype("timedelta64")
        assert is_timedelta64_ns_dtype("timedelta64[ns]")

        tdi = TimedeltaIndex([1e14, 2e14], dtype="timedelta64[ns]")
        assert is_timedelta64_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi.astype("timedelta64[ns]"))

        assert not is_timedelta64_ns_dtype(Index([], dtype=np.float64))
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.int64))


class TestIsScalar:
    # [CLASS DOC]
    # Tests of the scalar predicate on builtin, numpy and pandas scalars and
    # containers. No state.
    def test_is_scalar_builtin_scalars(self):
        # [FUNCTION DOC]
        # True for none, booleans, fraction, numbers, NaN, strings, bytes, date and
        # time types, NaT and NA.
        assert is_scalar(None)
        assert is_scalar(True)
        assert is_scalar(False)
        assert is_scalar(Fraction())
        assert is_scalar(0.0)
        assert is_scalar(1)
        assert is_scalar(complex(2))
        assert is_scalar(float("NaN"))
        assert is_scalar(np.nan)
        assert is_scalar("foobar")
        assert is_scalar(b"foobar")
        assert is_scalar(datetime(2014, 1, 1))
        assert is_scalar(date(2014, 1, 1))
        assert is_scalar(time(12, 0))
        assert is_scalar(timedelta(hours=1))
        assert is_scalar(pd.NaT)
        assert is_scalar(pd.NA)

    def test_is_scalar_builtin_nonscalars(self):
        # [FUNCTION DOC]
        # False for dict, list, tuple, slice and ellipsis.
        assert not is_scalar({})
        assert not is_scalar([])
        assert not is_scalar([1])
        assert not is_scalar(())
        assert not is_scalar((1,))
        assert not is_scalar(slice(None))
        assert not is_scalar(Ellipsis)

    def test_is_scalar_numpy_array_scalars(self):
        # [FUNCTION DOC]
        # True for numpy numeric, object, string and bytes scalars and numpy datetime64
        # and timedelta64 scalars.
        assert is_scalar(np.int64(1))
        assert is_scalar(np.float64(1.0))
        assert is_scalar(np.int32(1))
        assert is_scalar(np.complex64(2))
        assert is_scalar(np.object_("foobar"))
        assert is_scalar(np.str_("foobar"))
        assert is_scalar(np.bytes_(b"foobar"))
        assert is_scalar(np.datetime64("2014-01-01"))
        assert is_scalar(np.timedelta64(1, "h"))

    @pytest.mark.parametrize(
        "zerodim",
        [
            1,
            "foobar",
            np.datetime64("2014-01-01"),
            np.timedelta64(1, "h"),
            np.datetime64("NaT", "ns"),
        ],
    )
    def test_is_scalar_numpy_zerodim_arrays(self, zerodim):
        # [FUNCTION DOC]
        # Parametrized over an integer, string, datetime64, timedelta64 and datetime NaT.
        # A zero-dimensional array of the value is not a scalar, but the item extracted
        # from it by the lib helper is.
        zerodim = np.array(zerodim)
        assert not is_scalar(zerodim)
        assert is_scalar(lib.item_from_zerodim(zerodim))

    @pytest.mark.parametrize("arr", [np.array([]), np.array([[]])])
    def test_is_scalar_numpy_arrays(self, arr):
        # [FUNCTION DOC]
        # Parametrized over an empty 1-D and 2-D array. Neither the array nor the mock
        # numpy-like wrapper of it is a scalar.
        assert not is_scalar(arr)
        assert not is_scalar(MockNumpyLikeArray(arr))

    def test_is_scalar_pandas_scalars(self):
        # [FUNCTION DOC]
        # True for Timestamp, Timedelta, Period, Interval, date offset and a minute offset.
        assert is_scalar(Timestamp("2014-01-01"))
        assert is_scalar(Timedelta(hours=1))
        assert is_scalar(Period("2014-01-01"))
        assert is_scalar(Interval(left=0, right=1))
        assert is_scalar(DateOffset(days=1))
        assert is_scalar(pd.offsets.Minute(3))

    def test_is_scalar_pandas_containers(self):
        # [FUNCTION DOC]
        # False for Series, frames, Index, categorical, datetime/timedelta/period
        # arrays and a generic extension array.
        assert not is_scalar(Series(dtype=object))
        assert not is_scalar(Series([1]))
        assert not is_scalar(DataFrame())
        assert not is_scalar(DataFrame([[1]]))
        assert not is_scalar(Index([]))
        assert not is_scalar(Index([1]))
        assert not is_scalar(Categorical([]))
        assert not is_scalar(DatetimeIndex([])._data)
        assert not is_scalar(TimedeltaIndex([])._data)
        assert not is_scalar(DatetimeIndex([])._data.to_period("D"))
        assert not is_scalar(pd.array([1, 2, 3]))

    def test_is_scalar_number(self):
        # [FUNCTION DOC]
        # An instance of a concrete subclass of the abstract numeric base class is a
        # scalar (the abstract base itself is not recognized by the underlying numeric
        # check, per the comment).
        # Number() is not recognized by PyNumber_Check, so by extension
        #  is not recognized by is_scalar, but instances of non-abstract
        #  subclasses are.

        class Numeric(Number):
            # [CLASS DOC]
            # HELPER: nested concrete subclass of the abstract number base class used by the
            # preceding test; stores a value and can be converted to an integer.
            def __init__(self, value) -> None:
                # [FUNCTION DOC]
                # HELPER: stores the given value on the instance.
                self.value = value

            def __int__(self) -> int:
                # [FUNCTION DOC]
                # HELPER: returns the stored value as the integer conversion.
                return self.value

        num = Numeric(1)
        assert is_scalar(num)


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_datetimeindex_from_empty_datetime64_array(unit):
    # [FUNCTION DOC]
    # Parametrized over ms, us, ns units. A datetime index built from an empty
    # datetime64 array of that unit has length zero (no error).
    idx = DatetimeIndex(np.array([], dtype=f"datetime64[{unit}]"))
    assert len(idx) == 0


def test_nan_to_nat_conversions():
    # [FUNCTION DOC]
    # Setting NaN into a datetime column of a frame (via a block assignment) turns the
    # entries into NaT; reading a cell gives the NaT singleton; assigning NaN to a
    # Series slice also gives NaT.
    df = DataFrame(
        {"A": np.asarray(range(10), dtype="float64"), "B": Timestamp("20010101")}
    )
    df.iloc[3:6, :] = np.nan
    result = df.loc[4, "B"]
    assert result is pd.NaT

    s = df["B"].copy()
    s[8:9] = np.nan
    assert s[8] is pd.NaT


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize("spmatrix", ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
def test_is_scipy_sparse(spmatrix):
    # [FUNCTION DOC]
    # Parametrized over scipy sparse matrix formats; PendingDeprecationWarning is
    # ignored; skipped if scipy is missing. The scipy sparse predicate is true for a
    # matrix of the format and false for a numpy array.
    sparse = pytest.importorskip("scipy.sparse")

    klass = getattr(sparse, spmatrix + "_matrix")
    assert is_scipy_sparse(klass([[0, 1]]))
    assert not is_scipy_sparse(np.array([1]))


def test_ensure_int32():
    # [FUNCTION DOC]
    # The ensure-int32 helper returns an int32 array for int32 input and int64 input.
    values = np.arange(10, dtype=np.int32)
    result = ensure_int32(values)
    assert result.dtype == np.int32

    values = np.arange(10, dtype=np.int64)
    result = ensure_int32(values)
    assert result.dtype == np.int32


@pytest.mark.parametrize(
    "right,result",
    [
        (0, np.uint8),
        (-1, np.int16),
        (300, np.uint16),
        # For floats, we just upcast directly to float64 instead of trying to
        # find a smaller floating dtype
        (300.0, np.uint16),  # for integer floats, we convert them to ints
        (300.1, np.float64),
        (np.int16(300), np.int16 if np_version_gt2 else np.uint16),
    ],
)
def test_find_result_type_uint_int(right, result):
    # [FUNCTION DOC]
    # Parametrized over (right operand, expected type) with an unsigned 8-bit left
    # dtype: zero stays uint8, negative goes to int16, 300 to uint16, integral float
    # 300.0 is treated like the integer, a fractional float goes to float64, and a
    # numpy int16 scalar gives int16 on numpy 2 or later, else uint16.
    left_dtype = np.dtype("uint8")
    assert find_result_type(left_dtype, right) == result


@pytest.mark.parametrize(
    "right,result",
    [
        (0, np.int8),
        (-1, np.int8),
        (300, np.int16),
        # For floats, we just upcast directly to float64 instead of trying to
        # find a smaller floating dtype
        (300.0, np.int16),  # for integer floats, we convert them to ints
        (300.1, np.float64),
        (np.int16(300), np.int16),
    ],
)
def test_find_result_type_int_int(right, result):
    # [FUNCTION DOC]
    # Parametrized over (right operand, expected type) with a signed 8-bit left dtype:
    # zero and negative stay int8, 300 and 300.0 give int16, 300.1 gives float64,
    # numpy int16 scalar gives int16.
    left_dtype = np.dtype("int8")
    assert find_result_type(left_dtype, right) == result


@pytest.mark.parametrize(
    "right,result",
    [
        (300.0, np.float64),
        (np.float32(300), np.float32),
    ],
)
def test_find_result_type_floats(right, result):
    # [FUNCTION DOC]
    # Parametrized over (right, expected) with a float16 left dtype: the float
    # 300.0 gives float64 and a numpy float32 scalar gives float32.
    left_dtype = np.dtype("float16")
    assert find_result_type(left_dtype, right) == result
