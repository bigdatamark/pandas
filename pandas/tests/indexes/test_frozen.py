# [MODULE DOC]
# Tests of the immutable list type used for index names and levels (imported from the internal frozen-index module): immutability, slicing and concatenation returning the same type, string rendering (including non-ASCII), set-like union and difference, and conversion to bytes. Defines three fixtures and one test class with two helper methods.

import re

import pytest

from pandas.core.indexes.frozen import FrozenList


@pytest.fixture
def lst():
    # [FUNCTION DOC]
    # HELPER: fixture returning a plain list of the integers one through five.
    return [1, 2, 3, 4, 5]


@pytest.fixture
def container(lst):
    # [FUNCTION DOC]
    # HELPER: fixture wrapping the plain list fixture in the immutable list type.
    return FrozenList(lst)


@pytest.fixture
def unicode_container():
    # [FUNCTION DOC]
    # HELPER: fixture returning an immutable list of two Hebrew letters and "c", used to test rendering of non-ASCII text.
    return FrozenList(["\u05d0", "\u05d1", "c"])


class TestFrozenList:
    # [CLASS DOC]
    # Tests for the immutable list class.
    def check_mutable_error(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: intended to assert that a callable raises a type error saying mutable operations are unsupported. Takes positional and keyword arguments to forward to the callable.
        # KNOWN ISSUE: the body never calls the function passed in; it compiles a regular expression and then calls the compiled pattern object itself with the arguments, and asserts the resulting "pattern object is not callable" type error. The check therefore always passes regardless of whether the immutable list actually forbids mutation.
        # Pass whatever function you normally would to pytest.raises
        # (after the Exception kind).
        mutable_regex = re.compile("does not support mutable operations")
        msg = "'(_s)?re.(SRE_)?Pattern' object is not callable"
        with pytest.raises(TypeError, match=msg):
            mutable_regex(*args, **kwargs)

    def test_no_mutable_funcs(self, container):
        # [FUNCTION DOC]
        # Defines nested functions that assign an item, assign a slice, delete an item and delete a slice on the container, and passes each, then each of the methods extend, pop, remove and insert, to the helper. 
        # KNOWN ISSUE: because the helper never invokes its argument, none of these mutations is really exercised, so the test does not verify immutability.
        def setitem():
            # [FUNCTION DOC]
            # HELPER: nested function that assigns a value to the first position of the container; intended to raise.
            container[0] = 5

        self.check_mutable_error(setitem)

        def setslice():
            # [FUNCTION DOC]
            # HELPER: nested function that assigns a value to a slice of the container; intended to raise.
            container[1:2] = 3

        self.check_mutable_error(setslice)

        def delitem():
            # [FUNCTION DOC]
            # HELPER: nested function that deletes the first item of the container; intended to raise.
            del container[0]

        self.check_mutable_error(delitem)

        def delslice():
            # [FUNCTION DOC]
            # HELPER: nested function that deletes a slice of the container; intended to raise.
            del container[0:3]

        self.check_mutable_error(delslice)

        mutable_methods = ("extend", "pop", "remove", "insert")

        for meth in mutable_methods:
            self.check_mutable_error(getattr(container, meth))

    def test_slicing_maintains_type(self, container, lst):
        # [FUNCTION DOC]
        # Slicing an immutable list returns an immutable list equal to the same slice of the plain list.
        result = container[1:2]
        expected = lst[1:2]
        self.check_result(result, expected)

    def check_result(self, result, expected):
        # [FUNCTION DOC]
        # HELPER: asserts that the result is an instance of the immutable list type and equal to the expected value.
        assert isinstance(result, FrozenList)
        assert result == expected

    def test_string_methods_dont_fail(self, container):
        # [FUNCTION DOC]
        # The text representation, string conversion and bytes conversion of the container all complete without raising.
        repr(container)
        str(container)
        bytes(container)

    def test_tricky_container(self, unicode_container):
        # [FUNCTION DOC]
        # The text and string forms of a container holding non-ASCII text complete without raising.
        repr(unicode_container)
        str(unicode_container)

    def test_add(self, container, lst):
        # [FUNCTION DOC]
        # Adding a tuple on the right or on the left returns an immutable list with the concatenated contents in the right order.
        result = container + (1, 2, 3)  # noqa: RUF005
        expected = FrozenList(lst + [1, 2, 3])  # noqa: RUF005
        self.check_result(result, expected)

        result = (1, 2, 3) + container  # noqa: RUF005
        expected = FrozenList([1, 2, 3] + lst)  # noqa: RUF005
        self.check_result(result, expected)

    def test_iadd(self, container, lst):
        # [FUNCTION DOC]
        # In-place add of a one-element list produces an immutable list with the extra item, while the other name bound to the original container is unchanged.
        q = r = container

        q += [5]
        self.check_result(q, lst + [5])  # noqa: RUF005

        # Other shouldn't be mutated.
        self.check_result(r, lst)

    def test_union(self, container, lst):
        # [FUNCTION DOC]
        # Union with a tuple returns the original items followed by the new ones (no de-duplication, keeps duplicates from the argument), as an immutable list.
        result = container.union((1, 2, 3))
        expected = FrozenList(lst + [1, 2, 3])  # noqa: RUF005
        self.check_result(result, expected)

    def test_difference(self, container):
        # [FUNCTION DOC]
        # Difference with a list removes matching items and returns an immutable list of the rest in order.
        result = container.difference([2])
        expected = FrozenList([1, 3, 4, 5])
        self.check_result(result, expected)

    def test_difference_dupe(self):
        # [FUNCTION DOC]
        # Difference removes all occurrences of a duplicated value.
        result = FrozenList([1, 2, 3, 2]).difference([2])
        expected = FrozenList([1, 3])
        self.check_result(result, expected)

    def test_tricky_container_to_bytes_raises(self, unicode_container):
        # [FUNCTION DOC]
        # Converting a container of strings to bytes raises a type error saying a string cannot be interpreted as an integer.
        # GH 26447
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(unicode_container)
