# [MODULE DOC]
# Tests for the positional-argument validator in the private validators
# module of the util package. Covers invalid minimum argument counts, too many
# positional arguments (singular and plural messages), non-default values and
# the success path. All tests use a shared fixture supplying a function name.
import pytest

from pandas.util._validators import validate_args


@pytest.fixture
def _fname():
    # [FUNCTION DOC]
    # HELPER: fixture supplying the fake function name used in error messages
    # by the tests in this module. Returns the string "func".
    return "func"


def test_bad_min_fname_arg_count(_fname):
    # [FUNCTION DOC]
    # Verifies that a negative maximum count of function-name arguments makes
    # the validator raise a value error saying the count must be non-negative.
    msg = "'max_fname_arg_count' must be non-negative"

    with pytest.raises(ValueError, match=msg):
        validate_args(_fname, (None,), -1, "foo")


def test_bad_arg_length_max_value_single(_fname):
    # [FUNCTION DOC]
    # Verifies that passing two positional arguments when only one compatible
    # argument name (given as a tuple) is allowed raises a type error whose
    # message uses the singular word "argument" and reports the maximum and
    # actual lengths (both offset by the minimum count, here zero).
    args = (None, None)
    compat_args = ("foo",)

    min_fname_arg_count = 0
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(args) + min_fname_arg_count
    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"argument \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args(_fname, args, min_fname_arg_count, compat_args)


def test_bad_arg_length_max_value_multiple(_fname):
    # [FUNCTION DOC]
    # Verifies the too-many-arguments type error when the maximum is more than
    # one: compatible arguments given as a mapping with one entry and a
    # minimum count of two gives a maximum of three, with two supplied plus
    # the minimum giving four; the message must use the plural "arguments".
    args = (None, None)
    compat_args = {"foo": None}

    min_fname_arg_count = 2
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(args) + min_fname_arg_count
    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"arguments \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args(_fname, args, min_fname_arg_count, compat_args)


@pytest.mark.parametrize("i", range(1, 3))
def test_not_all_defaults(i, _fname):
    # [FUNCTION DOC]
    # Verifies a value error is raised when a supplied positional argument
    # differs from its default in the compatibility mapping. Parametrized by
    # how many leading values (one or two) are passed; in both cases the first
    # value differs from the default of the first parameter, so the message
    # names the parameter "foo" as unsupported in the pandas implementation.
    bad_arg = "foo"
    msg = (
        f"the '{bad_arg}' parameter is not supported "
        rf"in the pandas implementation of {_fname}\(\)"
    )

    compat_args = {"foo": 2, "bar": -1, "baz": 3}
    arg_vals = (1, -1, 3)

    with pytest.raises(ValueError, match=msg):
        validate_args(_fname, arg_vals[:i], 2, compat_args)


def test_validation(_fname):
    # [FUNCTION DOC]
    # Verifies the success path: arguments equal to their defaults (including
    # a partial list) pass without raising anything.
    # No exceptions should be raised.
    validate_args(_fname, (None,), 2, {"out": None})

    compat_args = {"axis": 1, "out": None}
    validate_args(_fname, (1, None), 2, compat_args)
