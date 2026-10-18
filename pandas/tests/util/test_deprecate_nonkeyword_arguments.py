"""
Tests for the `deprecate_nonkeyword_arguments` decorator
"""

# [MODULE DOC]
# Tests for the decorator that deprecates passing arguments positionally (making them keyword-only): resulting signatures, no-warning and warning cases, warning message text with explicit allowed arguments, default allowed arguments, positional-only parameters, and methods. Defines decorated helpers at import time and a module constant naming the warning category used.

import inspect

from pandas.errors import _CurrentDeprecationWarning
from pandas.util._decorators import deprecate_nonkeyword_arguments

import pandas._testing as tm

WARNING_CATEGORY = _CurrentDeprecationWarning


@deprecate_nonkeyword_arguments(
    WARNING_CATEGORY, allowed_args=["a", "b"], name="f_add_inputs"
)
def f(a, b=0, c=0, d=0):
    # [FUNCTION DOC]
    # HELPER: decorated with allowed positional arguments a and b and the display name f_add_inputs. Parameters a, b (default 0), c (0), d (0); returns the sum of all four.
    return a + b + c + d


def test_f_signature():
    # [FUNCTION DOC]
    # The decorated function's signature shows a and b positional and c, d keyword-only.
    assert str(inspect.signature(f)) == "(a, b=0, *, c=0, d=0)"


def test_one_argument():
    # [FUNCTION DOC]
    # Calling with one positional argument emits no warning and returns it.
    with tm.assert_produces_warning(None):
        assert f(19) == 19


def test_one_and_one_arguments():
    # [FUNCTION DOC]
    # One positional and one keyword emits no warning; sums to 25.
    with tm.assert_produces_warning(None):
        assert f(19, d=6) == 25


def test_two_arguments():
    # [FUNCTION DOC]
    # Two positional arguments (both allowed) emit no warning; sum 6.
    with tm.assert_produces_warning(None):
        assert f(1, 5) == 6


def test_two_and_two_arguments():
    # [FUNCTION DOC]
    # Two positional plus two keywords emit no warning; sum 12.
    with tm.assert_produces_warning(None):
        assert f(1, 3, c=3, d=5) == 12


def test_three_arguments():
    # [FUNCTION DOC]
    # Three positional arguments emit the deprecation warning category and return the sum 12.
    with tm.assert_produces_warning(WARNING_CATEGORY):
        assert f(6, 3, 3) == 12


def test_four_arguments():
    # [FUNCTION DOC]
    # Four positional arguments emit the warning and return 10.
    with tm.assert_produces_warning(WARNING_CATEGORY):
        assert f(1, 2, 3, 4) == 10


def test_three_arguments_with_name_in_warning():
    # [FUNCTION DOC]
    # The warning message states the version, the display name f_add_inputs, and that all arguments except a and b will be keyword-only.
    msg = (
        f"Starting with pandas version {WARNING_CATEGORY.version()} all arguments of "
        "f_add_inputs except for the arguments 'a' and 'b' will be keyword-only."
    )
    with tm.assert_produces_warning(WARNING_CATEGORY, match=msg):
        assert f(6, 3, 3) == 12


@deprecate_nonkeyword_arguments(WARNING_CATEGORY)
def g(a, b=0, c=0, d=0):
    # [FUNCTION DOC]
    # HELPER: decorated with the default allowed arguments (only the first). Parameters a, b, c, d with defaults 0 for the last three. Inside, asserts that no warning is produced while computing (so the warning must be raised by the decorator before the call, not inside), and returns the sum.
    with tm.assert_produces_warning(None):
        return a + b + c + d


def test_g_signature():
    # [FUNCTION DOC]
    # Signature shows a positional and b, c, d keyword-only.
    assert str(inspect.signature(g)) == "(a, *, b=0, c=0, d=0)"


def test_one_and_three_arguments_default_allowed_args():
    # [FUNCTION DOC]
    # One positional and three keywords: no warning, sum 12.
    with tm.assert_produces_warning(None):
        assert g(1, b=3, c=3, d=5) == 12


def test_three_arguments_default_allowed_args():
    # [FUNCTION DOC]
    # Three positional arguments warn with the category; sum 12.
    with tm.assert_produces_warning(WARNING_CATEGORY):
        assert g(6, 3, 3) == 12


def test_three_positional_argument_with_warning_message_analysis():
    # [FUNCTION DOC]
    # The warning message says all arguments of g except the argument a will be keyword-only (with the version).
    msg = (
        f"Starting with pandas version {WARNING_CATEGORY.version()} all arguments of g "
        "except for the argument 'a' will be keyword-only."
    )
    with tm.assert_produces_warning(WARNING_CATEGORY, match=msg):
        assert g(6, 3, 3) == 12


@deprecate_nonkeyword_arguments(WARNING_CATEGORY)
def h(a=0, b=0, c=0, d=0):
    # [FUNCTION DOC]
    # HELPER: decorated with default allowed arguments, but every parameter has a default so none is positional. Returns the sum of a, b, c, d.
    return a + b + c + d


def test_h_signature():
    # [FUNCTION DOC]
    # Signature shows all parameters keyword-only.
    assert str(inspect.signature(h)) == "(*, a=0, b=0, c=0, d=0)"


def test_all_keyword_arguments():
    # [FUNCTION DOC]
    # All keyword arguments: no warning, result 3.
    with tm.assert_produces_warning(None):
        assert h(a=1, b=2) == 3


def test_one_positional_argument():
    # [FUNCTION DOC]
    # One positional argument warns; result 23.
    with tm.assert_produces_warning(WARNING_CATEGORY):
        assert h(23) == 23


def test_one_positional_argument_with_warning_message_analysis():
    # [FUNCTION DOC]
    # The warning message says all arguments of h will be keyword-only.
    msg = (
        f"Starting with pandas version {WARNING_CATEGORY.version()} all arguments "
        "of h will be keyword-only."
    )
    with tm.assert_produces_warning(WARNING_CATEGORY, match=msg):
        assert h(19) == 19


@deprecate_nonkeyword_arguments(WARNING_CATEGORY)
def i(a=0, /, b=0, *, c=0, d=0):
    # [FUNCTION DOC]
    # HELPER: decorated function with a positional-only parameter a, a normal parameter b and keyword-only c and d; returns the sum.
    return a + b + c + d


def test_i_signature():
    # [FUNCTION DOC]
    # Signature shows all parameters keyword-only (the decorator rewrites the positional-only marker away).
    assert str(inspect.signature(i)) == "(*, a=0, b=0, c=0, d=0)"


def test_i_warns_klass():
    # [FUNCTION DOC]
    # Two positional arguments warn with the category; result 3.
    with tm.assert_produces_warning(WARNING_CATEGORY):
        assert i(1, 2) == 3


class Foo:
    # [CLASS DOC]
    # HELPER: class with a method decorated to allow self and bar positionally; used to test the decorator on methods.
    @deprecate_nonkeyword_arguments(WARNING_CATEGORY, allowed_args=["self", "bar"])
    def baz(self, bar=None, foobar=None):
        # [FUNCTION DOC]
        # HELPER: decorated method with parameters bar and foobar defaulting to none; body is an ellipsis (does nothing, returns none).
        ...


def test_foo_signature():
    # [FUNCTION DOC]
    # The method signature shows self and bar positional and foobar keyword-only.
    assert str(inspect.signature(Foo.baz)) == "(self, bar=None, *, foobar=None)"


def test_class():
    # [FUNCTION DOC]
    # Calling the method with two positional arguments warns; the message names Foo.baz and says all arguments except the argument bar will be keyword-only.
    msg = (
        rf"Starting with pandas version {WARNING_CATEGORY.version()} all arguments "
        r"of Foo\.baz except for the argument \'bar\' will be keyword-only"
    )
    with tm.assert_produces_warning(WARNING_CATEGORY, match=msg):
        Foo().baz("qux", "quox")
