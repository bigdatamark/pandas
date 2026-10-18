"""
frozen (immutable) data structures to support MultiIndexing

These are used for:

- .names (FrozenList)

"""

# [MODULE DOC]
# Immutable list container used for MultiIndex level and name collections.
# Public contents: one class, FrozenList (exposed publicly through the typing
# api namespace). It relies on the base pandas object class and the pretty
# printing helper for its textual form. No import-time side effects other than
# applying the module-relabelling decorator to the class.

from __future__ import annotations

from typing import (
    NoReturn,
    Self,
)

from pandas.util._decorators import set_module

from pandas.core.base import PandasObject

from pandas.io.formats.printing import pprint_thing


@set_module("pandas.api.typing")
class FrozenList(PandasObject, list):
    # [CLASS DOC]
    # A list subclass that rejects every mutating operation but is hashable and
    # compares equal to tuples and other frozen lists with the same elements.
    # Inherits from the pandas base object class and the builtin list (it must be a
    # real list subclass because the PyTables type checks require it). State is the
    # plain list contents only. Non-mutating operations (union, difference, slicing,
    # repeat, reverse add) return new instances of the same concrete class. Every
    # mutator (item and slice assignment or deletion, pop, append, extend, remove,
    # sort, insert) is aliased to a method that always raises a type error.
    """
    Container that doesn't allow setting item *but*
    because it's technically hashable, will be used
    for lookups, appropriately, etc.
    """

    # Side note: This has to be of type list. Otherwise,
    #            it messes up PyTables type checks.

    def union(self, other) -> FrozenList:
        # [FUNCTION DOC]
        # Return a new frozen list with the elements of other appended after those of
        # this list. The parameter other is any array-like or list; a tuple is first
        # converted to a list. Duplicates are kept (this is concatenation, not a set
        # union). Does not mutate this object.
        # KNOWN ISSUE: the docstring return description says it is the collection difference between self and other, which is a copy-paste error; the code returns the concatenation.
        """
        Returns a FrozenList with other concatenated to the end of self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are concatenating.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        if isinstance(other, tuple):
            other = list(other)
        return type(self)(super().__add__(other))

    def difference(self, other) -> FrozenList:
        # [FUNCTION DOC]
        # Return a new frozen list of the elements of this list that are not present in
        # other, preserving the original order and any duplicates among the kept
        # elements. The parameter other is any iterable of hashable items; it is
        # converted to a set for membership tests. Does not mutate this object.
        """
        Returns a FrozenList with elements from other removed from self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are removing self.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        other = set(other)
        temp = [x for x in self if x not in other]
        return type(self)(temp)

    # TODO: Consider deprecating these in favor of `union` (xref gh-15506)

    __add__ = __iadd__ = union  # pyright: ignore[reportAssignmentType]

    def __getitem__(self, n):
        # [FUNCTION DOC]
        # Item access. The parameter n is an integer position or a slice. A slice
        # yields a new instance of the same concrete class; an integer yields the
        # element itself. Index errors propagate from the list implementation.
        if isinstance(n, slice):
            return type(self)(super().__getitem__(n))
        return super().__getitem__(n)

    def __radd__(self, other) -> Self:
        # [FUNCTION DOC]
        # Reflected addition: return a new instance with the elements of other (a tuple
        # is converted to a list first) followed by the elements of this list. Used
        # when the left operand is a list or tuple.
        if isinstance(other, tuple):
            other = list(other)
        return type(self)(other + list(self))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality comparison against other. A tuple or another frozen list is first
        # converted to a plain list, then the builtin list comparison is used and its
        # result returned (which may be the not-implemented sentinel for unrelated
        # types).
        if isinstance(other, (tuple, FrozenList)):
            other = list(other)
        return super().__eq__(other)

    __req__ = __eq__

    def __mul__(self, other) -> Self:
        # [FUNCTION DOC]
        # Repeat the contents the given number of times (parameter other, an integer)
        # and return the result as a new instance of the same class. The in-place
        # variant is aliased to this method, so it never mutates and instead returns a
        # new object.
        return type(self)(super().__mul__(other))

    __imul__ = __mul__

    def __reduce__(self):
        # [FUNCTION DOC]
        # Support for pickling and copying: returns the concrete class together with an
        # argument tuple holding a plain list copy of the contents, so unpickling
        # rebuilds the object through the constructor.
        return type(self), (list(self),)

    # error: Signature of "__hash__" incompatible with supertype "list"
    def __hash__(self) -> int:  # type: ignore[override]
        # [FUNCTION DOC]
        # Return a hash computed from the contents converted to a tuple, which makes
        # frozen lists usable as dictionary keys or in lookups.
        return hash(tuple(self))

    def _disabled(self, *args, **kwargs) -> NoReturn:
        # [FUNCTION DOC]
        # HELPER: stand-in installed for every mutating list method of FrozenList.
        # Accepts and ignores any positional and keyword arguments and always raises a
        # type error stating that the class does not support mutable operations (the
        # message uses the concrete class name). Never returns.
        """
        This method will not function because object is immutable.
        """
        raise TypeError(f"'{type(self).__name__}' does not support mutable operations.")

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Return the pretty-printed text of the contents, with string elements quoted
        # and tab, carriage return, newline and single-quote characters escaped,
        # produced by the shared pretty printing helper.
        return pprint_thing(
            self, quote_strings=True, escape_chars=("\t", "\r", "\n", "'")
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return the class name followed by the string form of the contents in
        # parentheses.
        return f"{type(self).__name__}({self!s})"

    __setitem__ = __setslice__ = _disabled
    __delitem__ = __delslice__ = _disabled
    pop = append = extend = _disabled
    remove = sort = insert = _disabled  # pyright: ignore[reportAssignmentType]
