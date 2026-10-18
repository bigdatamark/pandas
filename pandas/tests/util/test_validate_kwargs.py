# [MODULE DOC]
# Tests for the keyword-argument validator and the boolean keyword validator
# in the private validators module of the util package.
import pytest

from pandas.util._validators import (
    validate_bool_kwarg,
    validate_kwargs,
)


@pytest.fixture
def _fname():
    # [FUNCTION DOC]
    # HELPER: fixture supplying the fake function name used in error messages
    # by the tests in this module. Returns the string "func".
    return "func"


def test_bad_kwarg(_fname):
    # [FUNCTION DOC]
    # Verifies a type error reporting an unexpected keyword argument when a
    # supplied keyword ("fo") is not among the compatible names (which are
    # "f" and "foo").
    good_arg = "f"
    bad_arg = good_arg + "o"

    compat_args = {good_arg: "foo", bad_arg + "o": "bar"}
    kwargs = {good_arg: "foo", bad_arg: "bar"}

    msg = rf"{_fname}\(\) got an unexpected keyword argument '{bad_arg}'"

    with pytest.raises(TypeError, match=msg):
        validate_kwargs(_fname, kwargs, compat_args)


@pytest.mark.parametrize("i", range(1, 3))
def test_not_all_none(i, _fname):
    # [FUNCTION DOC]
    # Verifies a value error when a keyword's value differs from its default
    # in the compatibility mapping. Parametrized by how many leading keywords
    # (one or two) are passed; "foo" is given 2 versus default 1, so the
    # message names "foo" as unsupported in the pandas implementation.
    bad_arg = "foo"
    msg = (
        f"the '{bad_arg}' parameter is not supported "
        rf"in the pandas implementation of {_fname}\(\)"
    )

    compat_args = {"foo": 1, "bar": "s", "baz": None}

    kwarg_keys = ("foo", "bar", "baz")
    kwarg_vals = (2, "s", None)

    kwargs = dict(zip(kwarg_keys[:i], kwarg_vals[:i], strict=True))

    with pytest.raises(ValueError, match=msg):
        validate_kwargs(_fname, kwargs, compat_args)


def test_validation(_fname):
    # [FUNCTION DOC]
    # Verifies the success path: a subset of keywords equal to their defaults
    # passes without raising.
    # No exceptions should be raised.
    compat_args = {"f": None, "b": 1, "ba": "s"}

    kwargs = {"f": None, "b": 1}
    validate_kwargs(_fname, kwargs, compat_args)


@pytest.mark.parametrize("name", ["inplace", "copy"])
@pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
def test_validate_bool_kwarg_fail(name, value):
    # [FUNCTION DOC]
    # Verifies the boolean keyword validator raises a value error, whose
    # message gives the argument name and the received type name, for
    # non-boolean values (integer, string, list, float). Parametrized over the
    # argument names inplace and copy.
    msg = (
        f'For argument "{name}" expected type bool, '
        f"received type {type(value).__name__}"
    )

    with pytest.raises(ValueError, match=msg):
        validate_bool_kwarg(value, name)


@pytest.mark.parametrize("name", ["inplace", "copy"])
@pytest.mark.parametrize("value", [True, False, None])
def test_validate_bool_kwarg(name, value):
    # [FUNCTION DOC]
    # Verifies the boolean keyword validator returns the input unchanged for
    # true, false and none (none allowed by default). Parametrized over the
    # argument names inplace and copy.
    assert validate_bool_kwarg(value, name) == value
