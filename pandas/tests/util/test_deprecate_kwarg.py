# [MODULE DOC]
# Tests for the decorator that deprecates a keyword argument in favor of a new name: simple renaming, value mapping with a dictionary or a function, error handling, and deprecating a keyword without renaming. Defines decorated helper functions and mapping objects at import time (decoration happens at import).

import pytest

from pandas.util._decorators import deprecate_kwarg

import pandas._testing as tm


@deprecate_kwarg(FutureWarning, "old", new_arg_name="new")
def _f1(new=False):
    # [FUNCTION DOC]
    # HELPER: function decorated to accept the deprecated keyword old for the parameter new (future warning). Parameter new defaults to false; returns it.
    return new


_f2_mappings = {"yes": True, "no": False}


@deprecate_kwarg(FutureWarning, "old", new_arg_name="new", mapping=_f2_mappings)
def _f2(new=False):
    # [FUNCTION DOC]
    # HELPER: like the first helper but with a dictionary mapping old values (yes/no strings) to new values (true/false). Returns new.
    return new


def _f3_mapping(x):
    # [FUNCTION DOC]
    # HELPER: mapping function used by the third helper; takes a number and returns it plus one.
    return x + 1


@deprecate_kwarg(FutureWarning, "old", new_arg_name="new", mapping=_f3_mapping)
def _f3(new=0):
    # [FUNCTION DOC]
    # HELPER: like the first helper but with a callable mapping applied to the old value. Parameter new defaults to 0; returns it.
    return new


@pytest.mark.parametrize("key,klass", [("old", FutureWarning), ("new", None)])
def test_deprecate_kwarg(key, klass):
    # [FUNCTION DOC]
    # Parametrized over passing the old keyword (expects a future warning) or the new keyword (expects no warning). With value 78 the helper returns it unchanged.
    x = 78

    with tm.assert_produces_warning(klass):
        assert _f1(**{key: x}) == x


@pytest.mark.parametrize("key", list(_f2_mappings.keys()))
def test_dict_deprecate_kwarg(key):
    # [FUNCTION DOC]
    # Parametrized over the dictionary keys. Passing old with a key warns and the result equals the mapped value.
    with tm.assert_produces_warning(FutureWarning):
        assert _f2(old=key) == _f2_mappings[key]


@pytest.mark.parametrize("key", ["bogus", 12345, -1.23])
def test_missing_deprecate_kwarg(key):
    # [FUNCTION DOC]
    # Parametrized over values absent from the mapping (a string, an integer and a float). Passing the old keyword warns and the value is passed through unchanged.
    with tm.assert_produces_warning(FutureWarning):
        assert _f2(old=key) == key


@pytest.mark.parametrize("x", [1, -1.4, 0])
def test_callable_deprecate_kwarg(x):
    # [FUNCTION DOC]
    # Parametrized over numbers. Passing old warns and the result equals the mapping function applied to the number.
    with tm.assert_produces_warning(FutureWarning):
        assert _f3(old=x) == _f3_mapping(x)


def test_callable_deprecate_kwarg_fail():
    # [FUNCTION DOC]
    # Passing a string through the mapping function that adds one raises a type error (the message may be a concatenation, string, or conversion error depending on the Python version).
    msg = "((can only|cannot) concatenate)|(must be str)|(Can't convert)"

    with pytest.raises(TypeError, match=msg):
        _f3(old="hello")


def test_bad_deprecate_kwarg():
    # [FUNCTION DOC]
    # Decorating with a mapping that is neither a dictionary nor callable (the integer zero) raises a type error saying the mapping must be a dict or callable.
    msg = "mapping from old to new argument values must be dict or callable!"

    with pytest.raises(TypeError, match=msg):

        @deprecate_kwarg(FutureWarning, "old", "new", 0)
        def f4(new=None):
            # [FUNCTION DOC]
            # HELPER: function that the test tries to decorate with an invalid mapping; never reached because decoration fails. Returns new.
            return new


@deprecate_kwarg(FutureWarning, "old", new_arg_name=None)
def _f4(old=True, unchanged=True):
    # [FUNCTION DOC]
    # HELPER: function decorated to deprecate the keyword old without a replacement name. Parameters old and unchanged both default to true; returns the pair.
    return old, unchanged


@pytest.mark.parametrize("key", ["old", "unchanged"])
def test_deprecate_keyword(key):
    # [FUNCTION DOC]
    # Parametrized over passing old (future warning, result is the value and true) or unchanged (no warning, result is true and the value).
    x = 9

    if key == "old":
        klass = FutureWarning
        expected = (x, True)
    else:
        klass = None
        expected = (True, x)

    with tm.assert_produces_warning(klass):
        assert _f4(**{key: x}) == expected
