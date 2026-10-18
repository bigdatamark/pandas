"""
Tests for behavior if an author does *not* implement EA methods.
"""
# [MODULE DOC]
# Tests for the default behavior of the extension array base class when an author
# implements only the bare minimum (a constructor). Defines a minimal array subclass
# and a fixture; verifies that unimplemented arithmetic operators are not available.
# No import-time side effects.

import numpy as np
import pytest

from pandas.core.arrays import ExtensionArray


class MyEA(ExtensionArray):
    # [CLASS DOC]
    # Minimal ExtensionArray subclass implementing only a constructor that stores the
    # given values in an attribute. Inherits the abstract base class; no other required
    # methods are provided, so most behaviors are the base-class defaults.
    def __init__(self, values) -> None:
        # [FUNCTION DOC]
        # Stores the values argument (any array-like) unchanged on the private values
        # attribute. No validation or copying. Returns nothing.
        self._values = values


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: fixture for the tests in this module.
    # Returns a minimal extension array wrapping a numeric array of the integers zero
    # through nine.
    arr = np.arange(10)
    return MyEA(arr)


class TestExtensionArray:
    # [CLASS DOC]
    # Test class grouping checks on the base-class behavior of an extension array that
    # does not implement optional features.
    def test_errors(self, data, all_arithmetic_operators):
        # [FUNCTION DOC]
        # Verifies that looking up any arithmetic operator method (parametrized through the
        # all-arithmetic-operators fixture) on the minimal array raises an attribute error,
        # since the base class does not define arithmetic.
        # invalid ops
        op_name = all_arithmetic_operators
        with pytest.raises(AttributeError):
            getattr(data, op_name)
