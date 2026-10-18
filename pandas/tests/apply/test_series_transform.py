# [MODULE DOC]
# Tests for Series.transform: extra args and kwargs, list-like, and dict-like arguments.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args, kwargs, increment):
    # [FUNCTION DOC]
    # GH 43357: transform on a series with a function taking extra positional and keyword arguments.
    # Parametrized by args, kwargs and the expected increment added to every value.
    # Expected: the series plus the increment.
    # GH 43357
    def f(x, a=0, b=0, c=0):
        # [FUNCTION DOC]
        # HELPER: function adding its first argument plus a, ten times b and a hundred times c, all of which
        # default to zero; used to make the effect of the extra arguments visible.
        return x + a + 10 * b + 100 * c

    s = Series([1, 2])
    result = s.transform(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_transform_listlike(string_series, ops, names):
    # [FUNCTION DOC]
    # GH 35964: list-like transform with numpy functions given as lists or arrays, parametrized by ops and
    # expected column names. Expected: a frame with one column per function, concatenated column-wise and
    # renamed, computed with numpy floating point errors ignored.
    # GH 35964
    with np.errstate(all="ignore"):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.transform(ops)
        tm.assert_frame_equal(result, expected)


def test_transform_listlike_func_with_args():
    # [FUNCTION DOC]
    # GH 50624: list-like transform where functions take extra arguments. Passing an extra keyword argument
    # that one function does not accept must raise a type error naming that function and the argument.
    # Passing an accepted subset gives a frame with a column per function name holding value plus the extras.
    # GH 50624

    s = Series([1, 2, 3])

    def foo1(x, a=1, c=0):
        # [FUNCTION DOC]
        # HELPER: adds a and c to the input; used as one of the transformed functions.
        return x + a + c

    def foo2(x, b=2, c=0):
        # [FUNCTION DOC]
        # HELPER: adds b and c to the input; used as the second transformed function.
        return x + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        s.transform([foo1, foo2], 0, 3, b=3, c=4)

    result = s.transform([foo1, foo2], 0, 3, c=4)
    expected = DataFrame({"foo1": [8, 9, 10], "foo2": [8, 9, 10]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("box", [dict, Series])
def test_transform_dictlike(string_series, box):
    # [FUNCTION DOC]
    # GH 35964: dict-like transform with the mapping given as a plain dict or a Series (parametrized by box).
    # Expected: frame of the two transformed columns with the dict keys as column names.
    # GH 35964
    with np.errstate(all="ignore"):
        expected = concat([np.sqrt(string_series), np.abs(string_series)], axis=1)
    expected.columns = ["foo", "bar"]
    result = string_series.transform(box({"foo": np.sqrt, "bar": np.abs}))
    tm.assert_frame_equal(result, expected)


def test_transform_dictlike_mixed():
    # [FUNCTION DOC]
    # GH 40018: dict-like transform where values mix lists and plain strings. Result must be a frame
    # with MultiIndex columns where the plain-string entry is a single sub-column, compared to an expected
    # frame of square roots and absolute values.
    # GH 40018 - mix of lists and non-lists in values of a dictionary
    df = Series([1, 4])
    result = df.transform({"b": ["sqrt", "abs"], "c": "sqrt"})
    expected = DataFrame(
        [[1.0, 1, 1.0], [2.0, 4, 2.0]],
        columns=MultiIndex([("b", "c"), ("sqrt", "abs")], [(0, 0, 1), (0, 1, 0)]),
    )
    tm.assert_frame_equal(result, expected)
