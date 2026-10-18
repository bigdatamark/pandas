# [MODULE DOC]
# Tests for the helpers in the apply machinery that give anonymous functions unique
# names when several are passed to an aggregation, and that make repeated
# (column, function-name) pairs unique. Imports the mangling helper and the
# unique-kwarg-list helper from the core apply module.
import numpy as np
import pytest

from pandas.core.apply import (
    _make_unique_kwarg_list,
    maybe_mangle_lambdas,
)


def test_maybe_mangle_lambdas_passthrough():
    # [FUNCTION DOC]
    # Verifies that a string function name, a single anonymous function, and a
    # one-element list holding one anonymous function are returned without renaming:
    # the anonymous functions keep their default name.
    assert maybe_mangle_lambdas("mean") == "mean"
    assert maybe_mangle_lambdas(lambda x: x).__name__ == "<lambda>"
    # don't mangle single lambda.
    assert maybe_mangle_lambdas([lambda x: x])[0].__name__ == "<lambda>"


def test_maybe_mangle_lambdas_listlike():
    # [FUNCTION DOC]
    # Verifies that a list of two anonymous functions is renamed with numeric suffixes
    # (zero then one) in order, and that the renamed functions still return the same
    # values as the originals.
    aggfuncs = [lambda x: 1, lambda x: 2]
    result = maybe_mangle_lambdas(aggfuncs)
    assert result[0].__name__ == "<lambda_0>"
    assert result[1].__name__ == "<lambda_1>"
    assert aggfuncs[0](None) == result[0](None)
    assert aggfuncs[1](None) == result[1](None)


def test_maybe_mangle_lambdas():
    # [FUNCTION DOC]
    # Verifies that, for a mapping from column name to a list of two anonymous
    # functions, each function in the list gets a numeric suffix in order.
    func = {"A": [lambda x: 0, lambda x: 1]}
    result = maybe_mangle_lambdas(func)
    assert result["A"][0].__name__ == "<lambda_0>"
    assert result["A"][1].__name__ == "<lambda_1>"


def test_maybe_mangle_lambdas_args():
    # [FUNCTION DOC]
    # Verifies that renaming works for anonymous functions that take extra positional
    # and keyword arguments with defaults, and that the original functions keep working
    # with those arguments after the call (the input mapping is not broken).
    func = {"A": [lambda x, a, b=1: (0, a, b), lambda x: 1]}
    result = maybe_mangle_lambdas(func)
    assert result["A"][0].__name__ == "<lambda_0>"
    assert result["A"][1].__name__ == "<lambda_1>"

    assert func["A"][0](0, 1) == (0, 1, 1)
    assert func["A"][0](0, 1, 2) == (0, 1, 2)
    assert func["A"][0](0, 2, b=3) == (0, 2, 3)


def test_maybe_mangle_lambdas_named():
    # [FUNCTION DOC]
    # Verifies that a mapping containing only named functions (including a nested
    # mapping from output name to function) is returned unchanged and equal to the input.
    func = {"C": np.mean, "D": {"foo": np.mean, "bar": np.mean}}
    result = maybe_mangle_lambdas(func)
    assert result == func


@pytest.mark.parametrize(
    "order, expected_reorder",
    [
        (
            [
                ("height", "<lambda>"),
                ("height", "max"),
                ("weight", "max"),
                ("height", "<lambda>"),
                ("weight", "<lambda>"),
            ],
            [
                ("height", "<lambda>_0"),
                ("height", "max"),
                ("weight", "max"),
                ("height", "<lambda>_1"),
                ("weight", "<lambda>"),
            ],
        ),
        (
            [
                ("col2", "min"),
                ("col1", "<lambda>"),
                ("col1", "<lambda>"),
                ("col1", "<lambda>"),
            ],
            [
                ("col2", "min"),
                ("col1", "<lambda>_0"),
                ("col1", "<lambda>_1"),
                ("col1", "<lambda>_2"),
            ],
        ),
        (
            [("col", "<lambda>"), ("col", "<lambda>"), ("col", "<lambda>")],
            [("col", "<lambda>_0"), ("col", "<lambda>_1"), ("col", "<lambda>_2")],
        ),
    ],
)
def test_make_unique(order, expected_reorder):
    # GH 27519, test if make_unique function reorders correctly
    # [FUNCTION DOC]
    # Parametrized over three (input pair list, expected list) cases; regression test
    # for a GitHub issue about ordering. Verifies that the unique-kwarg-list helper
    # appends a running index to repeated anonymous-function names for the same column
    # while leaving a single occurrence and non-anonymous names untouched, preserving
    # the original order of pairs.
    result = _make_unique_kwarg_list(order)

    assert result == expected_reorder
