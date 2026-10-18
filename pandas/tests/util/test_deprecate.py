# [MODULE DOC]
# Tests for the function that builds a deprecated alias of another function: warning emission, docstring rewriting to insert a deprecation directive after the summary, handling of functions without docstrings, and rejection of badly formatted docstrings. Defines helper functions used as deprecation targets.

from textwrap import dedent

import pytest

from pandas.util._decorators import deprecate

import pandas._testing as tm


def new_func():
    # [FUNCTION DOC]
    # HELPER: target function for the deprecation tests. Takes no parameters and returns the string "new_func called". Its docstring has a summary line, a blank line, then an extended summary, which is the format the deprecation helper requires.
    """
    This is the summary. The deprecate directive goes next.

    This is the extended summary. The deprecate directive goes before this.
    """
    return "new_func called"


def new_func_no_docstring():
    # [FUNCTION DOC]
    # HELPER: target function with no docstring; returns the string "new_func_no_docstring called".
    return "new_func_no_docstring called"


def new_func_wrong_docstring():
    # [FUNCTION DOC]
    # HELPER: target function whose docstring has its summary on the first line instead of the next line (incorrect format); returns the string "new_func_wrong_docstring called".
    """Summary should be in the next line."""
    return "new_func_wrong_docstring called"


def new_func_with_deprecation():
    # [FUNCTION DOC]
    # HELPER: never called; exists only so that its docstring serves as the expected result of deprecating the well-formed target function (summary, a deprecation directive for version 1.0 saying to use the new function, then the extended summary).
    """
    This is the summary. The deprecate directive goes next.

    .. deprecated:: 1.0
        Use new_func instead.

    This is the extended summary. The deprecate directive goes before this.
    """


def test_deprecate_ok():
    # [FUNCTION DOC]
    # Builds a deprecated alias (future warning, version 1.0, custom message) of the well-formed function. Calling it emits a future warning and returns the target's result; the alias docstring equals the dedented docstring of the expected-docs helper.
    depr_func = deprecate(
        FutureWarning, "depr_func", new_func, "1.0", msg="Use new_func instead."
    )

    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()

    assert result == "new_func called"
    assert depr_func.__doc__ == dedent(new_func_with_deprecation.__doc__)


def test_deprecate_no_docstring():
    # [FUNCTION DOC]
    # The alias of a function without a docstring still works: calling it warns and returns the target's result.
    depr_func = deprecate(
        FutureWarning,
        "depr_func",
        new_func_no_docstring,
        "1.0",
        msg="Use new_func instead.",
    )
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()
    assert result == "new_func_no_docstring called"


def test_deprecate_wrong_docstring():
    # [FUNCTION DOC]
    # Building an alias of a function whose summary is not on the second line raises an assertion error saying the helper needs a correctly formatted docstring.
    msg = "deprecate needs a correctly formatted docstring"
    with pytest.raises(AssertionError, match=msg):
        deprecate(
            FutureWarning,
            "depr_func",
            new_func_wrong_docstring,
            "1.0",
            msg="Use new_func instead.",
        )
