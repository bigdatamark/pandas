"""basic inference routines"""

# [MODULE DOC]
# Basic predicates that classify arbitrary Python objects (number, file-like,
# regex, list-like, dict-like, hashable, sequence, data class, and so on).
# Public contents: several names re-bound directly to the compiled helper
# library implementations (bool, integer, float, complex, scalar, decimal,
# list-like and iterator checks), plus pure-Python predicates defined here.
# Many are re-exported in the public api.types namespace (the decorator that
# sets the public module name is applied to those). Used widely across core
# for argument validation. Import-time side effects: none beyond binding names.

from __future__ import annotations

from collections import abc
from numbers import Number
import re
from re import Pattern
from typing import (
    TYPE_CHECKING,
    TypeGuard,
)

import numpy as np

from pandas._libs import lib
from pandas.util._decorators import set_module

if TYPE_CHECKING:
    from collections.abc import Hashable

is_bool = lib.is_bool

is_integer = lib.is_integer

is_float = lib.is_float

is_complex = lib.is_complex

is_scalar = lib.is_scalar

is_decimal = lib.is_decimal

is_list_like = lib.is_list_like

is_iterator = lib.is_iterator


@set_module("pandas.api.types")
def is_number(obj: object) -> TypeGuard[Number | np.number]:
    # [FUNCTION DOC]
    # Returns true when obj is an instance of the standard numeric abstract base class
    # or a numpy numeric scalar type. Booleans count as numbers; strings do not. Never
    # raises.
    """
    Check if the object is a number.

    Returns True when the object is a number, and False if is not.

    Parameters
    ----------
    obj : any type
        The object to check if is a number.

    Returns
    -------
    bool
        Whether `obj` is a number or not.

    See Also
    --------
    api.types.is_integer: Checks a subgroup of numbers.

    Examples
    --------
    >>> from pandas.api.types import is_number
    >>> is_number(1)
    True
    >>> is_number(7.15)
    True

    Booleans are valid because they are int subclass.

    >>> is_number(False)
    True

    >>> is_number("foo")
    False
    >>> is_number("5")
    False
    """
    return isinstance(obj, (Number, np.number))


def iterable_not_string(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj is iterable (per the standard iterable abstract base class)
    # and is not a text string. Byte strings are still treated as iterable. Never
    # raises.
    """
    Check if the object is an iterable but not a string.

    Parameters
    ----------
    obj : The object to check.

    Returns
    -------
    is_iter_not_string : bool
        Whether `obj` is a non-string iterable.

    Examples
    --------
    >>> iterable_not_string([1, 2, 3])
    True
    >>> iterable_not_string("foo")
    False
    >>> iterable_not_string(1)
    False
    """
    return isinstance(obj, abc.Iterable) and not isinstance(obj, str)


@set_module("pandas.api.types")
def is_file_like(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj has a read or a write attribute and also has an iteration
    # attribute; otherwise false. Only attribute presence is tested, nothing is called.
    # # KNOWN ISSUE: the docstring says the object must be an iterator, but the code
    # only requires the iteration protocol method to exist, not the next-item method.
    """
    Check if the object is a file-like object.

    For objects to be considered file-like, they must
    be an iterator AND have either a `read` and/or `write`
    method as an attribute.

    Note: file-like objects must be iterable, but
    iterable objects need not be file-like.

    Parameters
    ----------
    obj : object
        The object to check for file-like properties.
        This can be any Python object, and the function will
        check if it has attributes typically associated with
        file-like objects (e.g., `read`, `write`, `__iter__`).

    Returns
    -------
    bool
        Whether `obj` has file-like properties.

    See Also
    --------
    api.types.is_dict_like : Check if the object is dict-like.
    api.types.is_hashable : Return True if hash(obj) will succeed, False otherwise.
    api.types.is_named_tuple : Check if the object is a named tuple.
    api.types.is_iterator : Check if the object is an iterator.

    Examples
    --------
    >>> import io
    >>> from pandas.api.types import is_file_like
    >>> buffer = io.StringIO("data")
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False

    return bool(hasattr(obj, "__iter__"))


@set_module("pandas.api.types")
def is_re(obj: object) -> TypeGuard[Pattern]:
    # [FUNCTION DOC]
    # Returns true when obj is an instance of the compiled regular-expression pattern
    # type; strings are not patterns. Never raises.
    """
    Check if the object is a regex pattern instance.

    Parameters
    ----------
    obj : object
        The object to check for being a regex pattern. Typically,
        this would be an object that you expect to be a compiled
        pattern from the `re` module.

    Returns
    -------
    bool
        Whether `obj` is a regex pattern.

    See Also
    --------
    api.types.is_float : Return True if given object is float.
    api.types.is_iterator : Check if the object is an iterator.
    api.types.is_integer : Return True if given object is integer.
    api.types.is_re_compilable : Check if the object can be compiled
                                into a regex pattern instance.

    Examples
    --------
    >>> from pandas.api.types import is_re
    >>> import re
    >>> is_re(re.compile(".*"))
    True
    >>> is_re("foo")
    False
    """
    return isinstance(obj, Pattern)


@set_module("pandas.api.types")
def is_re_compilable(obj: object) -> bool:
    # [FUNCTION DOC]
    # Tries to compile obj as a regular expression and returns true on success, false
    # when compilation raises a type error (for example a non-string, non-pattern
    # object). No state is kept.
    # # KNOWN ISSUE: only the type error is caught, so a string that is not a valid
    # pattern propagates the regex compile error instead of returning false, contrary to
    # the docstring.
    """
    Check if the object can be compiled into a regex pattern instance.

    Parameters
    ----------
    obj : The object to check
        The object to check if the object can be compiled into a regex pattern instance.

    Returns
    -------
    bool
        Whether `obj` can be compiled as a regex pattern.

    See Also
    --------
    api.types.is_re : Check if the object is a regex pattern instance.

    Examples
    --------
    >>> from pandas.api.types import is_re_compilable
    >>> is_re_compilable(".*")
    True
    >>> is_re_compilable(1)
    False
    """
    try:
        re.compile(obj)  # type: ignore[call-overload]
    except TypeError:
        return False
    else:
        return True


@set_module("pandas.api.types")
def is_array_like(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj is list-like (per the list-like helper) and also has a dtype
    # attribute. Plain lists and tuples are therefore not array-like.
    """
    Check if the object is array-like.

    For an object to be considered array-like, it must be list-like and
    have a `dtype` attribute.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_array_like : bool
        Whether `obj` has array-like properties.

    Examples
    --------
    >>> is_array_like(np.array([1, 2, 3]))
    True
    >>> is_array_like(pd.Series(["a", "b"]))
    True
    >>> is_array_like(pd.Index(["2016-01-01"]))
    True
    >>> is_array_like([1, 2, 3])
    False
    >>> is_array_like(("a", "b"))
    False
    """
    return is_list_like(obj) and hasattr(obj, "dtype")


def is_nested_list_like(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj is list-like, defines a length, has at least one element,
    # and every element is itself list-like. Objects without a length (such as
    # generators) return false so that they are never consumed; empty containers return
    # false.
    """
    Check if the object is list-like, and that all of its elements
    are also list-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_list_like : bool
        Whether `obj` has list-like properties.

    Examples
    --------
    >>> is_nested_list_like([[1, 2, 3]])
    True
    >>> is_nested_list_like([{1, 2, 3}, {1, 2, 3}])
    True
    >>> is_nested_list_like(["foo"])
    False
    >>> is_nested_list_like([])
    False
    >>> is_nested_list_like([[1, 2, 3], 1])
    False

    Notes
    -----
    This won't reliably detect whether a consumable iterator (e. g.
    a generator) is a nested-list-like without consuming the iterator.
    To avoid consuming it, we always return False if the outer container
    doesn't define `__len__`.

    See Also
    --------
    is_list_like
    """
    return (
        is_list_like(obj)
        and hasattr(obj, "__len__")
        # need PEP 724 to handle these typing errors
        and len(obj) > 0  # pyright: ignore[reportArgumentType]
        and all(is_list_like(item) for item in obj)  # type: ignore[attr-defined]
    )


@set_module("pandas.api.types")
def is_dict_like(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj has all of the item-access, keys and containment attributes
    # and is not itself a class. Only attribute presence is tested.
    """
    Check if the object is dict-like.

    Parameters
    ----------
    obj : object
        The object to check. This can be any Python object,
        and the function will determine whether it
        behaves like a dictionary.

    Returns
    -------
    bool
        Whether `obj` has dict-like properties.

    See Also
    --------
    api.types.is_list_like : Check if the object is list-like.
    api.types.is_file_like : Check if the object is a file-like.
    api.types.is_named_tuple : Check if the object is a named tuple.

    Examples
    --------
    >>> from pandas.api.types import is_dict_like
    >>> is_dict_like({1: 2})
    True
    >>> is_dict_like([1, 2, 3])
    False
    >>> is_dict_like(dict)
    False
    >>> is_dict_like(dict())
    True
    """
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    return (
        all(hasattr(obj, attr) for attr in dict_like_attrs)
        # [GH 25196] exclude classes
        and not isinstance(obj, type)
    )


@set_module("pandas.api.types")
def is_named_tuple(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj is a sequence instance that has a field-names attribute (the
    # marker of named tuples). Never raises.
    """
    Check if the object is a named tuple.

    Parameters
    ----------
    obj : object
        The object that will be checked to determine
        whether it is a named tuple.

    Returns
    -------
    bool
        Whether `obj` is a named tuple.

    See Also
    --------
    api.types.is_dict_like: Check if the object is dict-like.
    api.types.is_hashable: Return True if hash(obj)
                                  will succeed, False otherwise.
    api.types.is_categorical_dtype : Check if the dtype is categorical.

    Examples
    --------
    >>> from collections import namedtuple
    >>> from pandas.api.types import is_named_tuple
    >>> Point = namedtuple("Point", ["x", "y"])
    >>> p = Point(1, 2)
    >>>
    >>> is_named_tuple(p)
    True
    >>> is_named_tuple((1, 2))
    False
    """
    return isinstance(obj, abc.Sequence) and hasattr(obj, "_fields")


@set_module("pandas.api.types")
def is_hashable(obj: object, allow_slice: bool = True) -> TypeGuard[Hashable]:
    # [FUNCTION DOC]
    # Parameters: obj is any object; allow_slice (default true) controls whether slices
    # are acceptable. When allow_slice is false, a slice, or a tuple containing a slice
    # at the top level, returns false immediately. Otherwise obj is hashed and the
    # result is true on success and false if hashing raises a type error. Other
    # exceptions propagate. Uses a real hash attempt because the abstract base class
    # test is unreliable for numpy scalars.
    """
    Return True if hash(obj) will succeed, False otherwise.

    Some types will pass a test against collections.abc.Hashable but fail when
    they are actually hashed with hash().

    Distinguish between these and other types by trying the call to hash() and
    seeing if they raise TypeError.

    Parameters
    ----------
    obj : object
        The object to check for hashability. Any Python object can be passed here.
    allow_slice : bool
        If True, return True if the object is hashable (including slices).
        If False, return True if the object is hashable and not a slice.

    Returns
    -------
    bool
        True if object can be hashed (i.e., does not raise TypeError when
        passed to hash()) and passes the slice check according to 'allow_slice'.
        False otherwise (e.g., if object is mutable like a list or dictionary
        or if allow_slice is False and object is a slice or contains a slice).

    See Also
    --------
    api.types.is_float : Return True if given object is float.
    api.types.is_iterator : Check if the object is an iterator.
    api.types.is_list_like : Check if the object is list-like.
    api.types.is_dict_like : Check if the object is dict-like.

    Examples
    --------
    >>> import collections
    >>> from pandas.api.types import is_hashable
    >>> a = ([],)
    >>> isinstance(a, collections.abc.Hashable)
    True
    >>> is_hashable(a)
    False
    """
    # Unfortunately, we can't use isinstance(obj, collections.abc.Hashable),
    # which can be faster than calling hash. That is because numpy scalars
    # fail this test.

    # Reconsider this decision once this numpy bug is fixed:
    # https://github.com/numpy/numpy/issues/5562

    if allow_slice is False:
        if isinstance(obj, tuple) and any(isinstance(v, slice) for v in obj):
            return False
        elif isinstance(obj, slice):
            return False

    try:
        hash(obj)
    except TypeError:
        return False
    else:
        return True


def is_sequence(obj: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj can be iterated and has a length and is not a text or byte
    # string; false if iterating or taking the length raises a type error or attribute
    # error. Iterators and generators are not sequences because they have no length.
    # Note that iterating is only started, not consumed.
    """
    Check if the object is a sequence of objects.
    String types are not included as sequences here.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_sequence : bool
        Whether `obj` is a sequence of objects.

    Examples
    --------
    >>> l = [1, 2, 3]
    >>>
    >>> is_sequence(l)
    True
    >>> is_sequence(iter(l))
    False
    """
    try:
        # Can iterate over it.
        iter(obj)  # type: ignore[call-overload]
        # Has a length associated with it.
        len(obj)  # type: ignore[arg-type]
        return not isinstance(obj, (str, bytes))
    except (TypeError, AttributeError):
        return False


def is_dataclass(item: object) -> bool:
    # [FUNCTION DOC]
    # Returns true when obj is an instance of a data class and not the data class type
    # itself. Imports the data class support module lazily; returns false if that import
    # fails.
    # # KNOWN ISSUE: the import failure branch is effectively dead since the module is
    # part of the standard library.
    """
    Checks if the object is a data-class instance

    Parameters
    ----------
    item : object

    Returns
    --------
    is_dataclass : bool
        True if the item is an instance of a data-class,
        will return false if you pass the data class itself

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> is_dataclass(Point)
    False
    >>> is_dataclass(Point(0, 2))
    True

    """
    try:
        import dataclasses

        return dataclasses.is_dataclass(item) and not isinstance(item, type)
    except ImportError:
        return False
