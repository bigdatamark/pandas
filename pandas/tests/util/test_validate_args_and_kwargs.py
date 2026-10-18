# [MODULE DOC]
# Tests for the combined positional and keyword argument validator in the
# private validators module of the util package: total length limits,
# missing or non-default values, duplicated arguments and the success path.
import pytest

from pandas.util._validators import validate_args_and_kwargs


@pytest.fixture
def _fname():
    # [FUNCTION DOC]
    # HELPER: fixture supplying the fake function name used in error messages
    # by the tests in this module. Returns the string "func".
    return "func"


def test_invalid_total_length_max_length_one(_fname):
    # [FUNCTION DOC]
    # Verifies a type error when positional plus keyword arguments exceed the
    # allowed total, with one compatible name: two positional and one keyword
    # value are supplied. The message uses the singular "argument" and
    # reports the maximum and the actual total (offset by the minimum count).
    compat_args = ("foo",)
    kwargs = {"foo": "FOO"}
    args = ("FoO", "BaZ")

    min_fname_arg_count = 0
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count

    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"argument \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


def test_invalid_total_length_max_length_multiple(_fname):
    # [FUNCTION DOC]
    # Same as the single-name case but with three compatible names and a
    # minimum count of two, so the maximum is more than one: the type error
    # message must use the plural "arguments" with the computed maximum and
    # actual totals.
    compat_args = ("foo", "bar", "baz")
    kwargs = {"foo": "FOO", "bar": "BAR"}
    args = ("FoO", "BaZ")

    min_fname_arg_count = 2
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count

    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"arguments \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


@pytest.mark.parametrize("args,kwargs", [((), {"foo": -5, "bar": 2}), ((-5, 2), {})])
def test_missing_args_or_kwargs(args, kwargs, _fname):
    # [FUNCTION DOC]
    # Verifies a value error naming the parameter "bar" as unsupported when its
    # value (2) differs from its default (1), whether the values arrive purely
    # as keywords or purely as positionals. Parametrized over those two
    # delivery styles.
    bad_arg = "bar"
    min_fname_arg_count = 2

    compat_args = {"foo": -5, bad_arg: 1}

    msg = (
        rf"the '{bad_arg}' parameter is not supported "
        rf"in the pandas implementation of {_fname}\(\)"
    )

    with pytest.raises(ValueError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


def test_duplicate_argument(_fname):
    # [FUNCTION DOC]
    # Verifies a type error saying the function got multiple values for the
    # keyword "foo" when "foo" is supplied both positionally (first slot) and
    # as a keyword.
    min_fname_arg_count = 2

    compat_args = {"foo": None, "bar": None, "baz": None}
    kwargs = {"foo": None, "bar": None}
    args = (None,)  # duplicate value for "foo"

    msg = rf"{_fname}\(\) got multiple values for keyword argument 'foo'"

    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


def test_validation(_fname):
    # [FUNCTION DOC]
    # Verifies the success path: positional and keyword values that all equal
    # their defaults, split across both forms without duplication, pass
    # without raising.
    # No exceptions should be raised.
    compat_args = {"foo": 1, "bar": None, "baz": -2}
    kwargs = {"baz": -2}

    args = (1, None)
    min_fname_arg_count = 2

    validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)
