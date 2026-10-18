# [MODULE DOC]
# Tests for the deferred column expression object created by the top-level column
# constructor: operators (arithmetic, comparison, unary, in-place, logical, matrix
# multiplication), usage as arguments to frame assignment, item access, indexing and
# setting, accessor namespaces, attribute and method forwarding, string representation,
# and the operations that must raise errors. Uses the accessor cleanup helper from the
# accessor registration tests.
from collections.abc import Callable
import copy
from datetime import datetime
import operator
import re

import numpy as np
import pytest

from pandas._libs.properties import cache_readonly

import pandas as pd
import pandas._testing as tm
from pandas.api.typing import Expression
from pandas.tests.test_register_accessor import ensure_removed


@pytest.mark.parametrize(
    ("expr", "expected_values", "expected_str"),
    [
        (pd.col("a"), [1, 2], "col('a')"),
        (pd.col("a") * 2, [2, 4], "col('a') * 2"),
        (pd.col("a").sum(), [3, 3], "col('a').sum()"),
        (pd.col("a") + 1, [2, 3], "col('a') + 1"),
        (1 + pd.col("a"), [2, 3], "1 + col('a')"),
        (pd.col("a") - 1, [0, 1], "col('a') - 1"),
        (1 - pd.col("a"), [0, -1], "1 - col('a')"),
        (pd.col("a") * 1, [1, 2], "col('a') * 1"),
        (1 * pd.col("a"), [1, 2], "1 * col('a')"),
        (2 ** pd.col("a"), [2, 4], "2 ** col('a')"),
        (pd.col("a") ** 2, [1, 4], "col('a') ** 2"),
        (pd.col("a") / 1, [1.0, 2.0], "col('a') / 1"),
        (1 / pd.col("a"), [1.0, 0.5], "1 / col('a')"),
        (pd.col("a") // 1, [1, 2], "col('a') // 1"),
        (1 // pd.col("a"), [1, 0], "1 // col('a')"),
        (pd.col("a") % 1, [0, 0], "col('a') % 1"),
        (1 % pd.col("a"), [0, 1], "1 % col('a')"),
        (pd.col("a") > 1, [False, True], "col('a') > 1"),
        (pd.col("a") >= 1, [True, True], "col('a') >= 1"),
        (pd.col("a") < 1, [False, False], "col('a') < 1"),
        (pd.col("a") <= 1, [True, False], "col('a') <= 1"),
        (pd.col("a") == 1, [True, False], "col('a') == 1"),
        (np.power(pd.col("a"), 2), [1, 4], "power(col('a'), 2)"),
        (np.divide(pd.col("a"), pd.col("a")), [1.0, 1.0], "divide(col('a'), col('a'))"),
        (
            (pd.col("a") + 1) * (pd.col("b") + 2),
            [10, 18],
            "(col('a') + 1) * (col('b') + 2)",
        ),
        (
            (pd.col("a") - 1).astype("bool"),
            [False, True],
            "(col('a') - 1).astype('bool')",
        ),
        # Unary operators
        (-pd.col("a"), [-1, -2], "-col('a')"),
        (+pd.col("a"), [1, 2], "+col('a')"),
        (-(pd.col("a") + 1), [-2, -3], "-(col('a') + 1)"),
        (-pd.col("a") * 2, [-2, -4], "(-col('a')) * 2"),
        (abs(pd.col("a")), [1, 2], "abs(col('a'))"),
        (abs(pd.col("a") - 2), [1, 0], "abs(col('a') - 2)"),
    ],
)
def test_col_simple(
    expr: Expression, expected_values: list[object], expected_str: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over (expression, expected values, expected string) for many binary and
    # reflected arithmetic operations, comparisons, ufunc calls, nested expressions, casting,
    # and unary operators on column "a" of a two-column frame. Verifies assigning the expression
    # as a new column gives the expected frame and that the expression's string form matches the
    # expected text.
    # https://github.com/pandas-dev/pandas/pull/64267
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.assign(c=expr)
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": expected_values})
    tm.assert_frame_equal(result, expected)
    assert str(expr) == expected_str


@pytest.mark.parametrize(
    ("op", "expected_values", "expected_str"),
    [
        (operator.iadd, [3, 4], "col('a') + 2"),
        (operator.iand, [0, 2], "col('a') & 2"),
        (operator.ifloordiv, [0, 1], "col('a') // 2"),
        (operator.imod, [1, 0], "col('a') % 2"),
        (operator.imul, [2, 4], "col('a') * 2"),
        (operator.ior, [3, 2], "col('a') | 2"),
        (operator.ipow, [1, 4], "col('a') ** 2"),
        (operator.isub, [-1, 0], "col('a') - 2"),
        (operator.itruediv, [0.5, 1.0], "col('a') / 2"),
        (operator.ixor, [3, 0], "col('a') ^ 2"),
    ],
)
def test_inplace_ops(
    op: Callable, expected_values: list[object], expected_str: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over in-place operator functions (add, and, floor-divide, modulo, multiply,
    # or, power, subtract, true-divide, xor) with the right operand 2. Verifies the result of the
    # in-place operator on an expression is a new expression (not mutated state) that assigns the
    # expected values and whose string form shows the corresponding non-in-place operator.
    # https://github.com/pandas-dev/pandas/pull/64267
    df = pd.DataFrame({"a": [1, 2]})
    expr = pd.col("a")
    expr = op(expr, 2)
    result = df.assign(c=expr)
    expected = pd.DataFrame({"a": [1, 2], "c": expected_values})
    tm.assert_frame_equal(result, expected)
    assert str(expr) == expected_str


def test_matmul():
    # [FUNCTION DOC]
    # Verifies matrix multiplication of the column with a list on either side evaluates to the
    # dot product repeated per row and has the expected string forms.
    # https://github.com/pandas-dev/pandas/pull/64267
    df = pd.DataFrame({"a": [1, 2]})

    expr = pd.col("a") @ [3, 4]
    result = df.assign(c=expr)
    expected = pd.DataFrame({"a": [1, 2], "c": [11, 11]})
    tm.assert_frame_equal(result, expected)
    assert str(expr) == "col('a') @ [3, 4]"

    expr = [3, 4] @ pd.col("a")
    result = df.assign(c=expr)
    expected = pd.DataFrame({"a": [1, 2], "c": [11, 11]})
    tm.assert_frame_equal(result, expected)
    assert str(expr) == "[3, 4] @ col('a')"


def test_frame_getitem() -> None:
    # [FUNCTION DOC]
    # Verifies indexing a frame with a boolean expression selects the matching rows.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expr = pd.col("a") == 2
    result = df[expr]
    expected = df.iloc[[1]]
    tm.assert_frame_equal(result, expected)


def test_frame_setitem() -> None:
    # [FUNCTION DOC]
    # Verifies setting through a boolean expression key assigns the value to all columns of the
    # matching rows only.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expr = pd.col("a") == 2

    result = df.copy()
    result[expr] = 100
    expected = pd.DataFrame({"a": [1, 100], "b": [3, 100]})
    tm.assert_frame_equal(result, expected)


def test_frame_loc() -> None:
    # [FUNCTION DOC]
    # Verifies label-based setting with a boolean expression row key and a column label assigns only
    # matching rows in that column.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expr = pd.col("a") == 2
    result = df.copy()
    result.loc[expr, "b"] = 100
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 100]})
    tm.assert_frame_equal(result, expected)


def test_frame_iloc() -> None:
    # [FUNCTION DOC]
    # Verifies position-based setting with a boolean expression row key and an integer column
    # position assigns only matching rows in that column.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expr = pd.col("a") == 2
    result = df.copy()
    result.iloc[expr, 1] = 100
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 100]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected_values", "expected_str"),
    [
        (pd.col("a").dt.year, [2020], "col('a').dt.year"),
        (pd.col("a").dt.strftime("%B"), ["January"], "col('a').dt.strftime('%B')"),
        (pd.col("b").str.upper(), ["FOO"], "col('b').str.upper()"),
    ],
)
def test_namespaces(
    expr: Expression, expected_values: list[object], expected_str: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over expressions using the datetime accessor (year, string formatting) and the
    # string accessor (uppercase). Verifies assignment gives the expected new column (dtype check
    # relaxed) and the string forms include the accessor path.
    df = pd.DataFrame({"a": [datetime(2020, 1, 1)], "b": ["foo"]})
    result = df.assign(c=expr)
    expected = pd.DataFrame(
        {"a": [datetime(2020, 1, 1)], "b": ["foo"], "c": expected_values}
    )
    tm.assert_frame_equal(result, expected, check_dtype=False)
    assert str(expr) == expected_str


def test_invalid() -> None:
    # [FUNCTION DOC]
    # Verifies that assigning an expression referencing a missing column raises a value error
    # suggesting existing columns; with eleven columns the suggestion lists the first ten followed
    # by an ellipsis marker.
    # KNOWN ISSUE: the test contains a stray empty string literal statement (a no-op) between building the message and the raises check; it has no effect.
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match=r"did you mean one of \['a', 'b'\] instead"):
        df.assign(c=pd.col("c").mean())
    df = pd.DataFrame({f"col_{i}": [0] for i in range(11)})
    msg = (
        "did you mean one of "
        r"\['col_0', 'col_1', 'col_2', 'col_3', "
        "'col_4', 'col_5', 'col_6', 'col_7', "
        r"'col_8', 'col_9',\.\.\.\] instead"
    )
    ""
    with pytest.raises(ValueError, match=msg):
        df.assign(c=pd.col("c").mean())


def test_custom_accessor() -> None:
    # [FUNCTION DOC]
    # Registers a custom Series accessor (removed afterward by the cleanup helper) and verifies it is
    # reachable from a column expression, calling its mean method through the deferred expression
    # and producing the expected broadcast result column.
    df = pd.DataFrame({"a": [1, 2, 3]})

    class XYZAccessor:
        # [CLASS DOC]
        # HELPER: custom accessor class defined locally; wraps a Series and offers a mean method.
        def __init__(self, pandas_obj):
            # [FUNCTION DOC]
            # HELPER: stores the wrapped object.
            self._obj = pandas_obj

        def mean(self):
            # [FUNCTION DOC]
            # HELPER: returns the mean of the wrapped object.
            return self._obj.mean()

    with ensure_removed(pd.Series, "xyz"):
        pd.api.extensions.register_series_accessor("xyz")(XYZAccessor)
        result = df.assign(b=pd.col("a").xyz.mean())
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected_values", "expected_str"),
    [
        (
            pd.col("a") & pd.col("b"),
            [False, False, True, False],
            "col('a') & col('b')",
        ),
        (
            pd.col("a") & True,
            [True, False, True, False],
            "col('a') & True",
        ),
        (
            pd.col("a") | pd.col("b"),
            [True, True, True, True],
            "col('a') | col('b')",
        ),
        (
            pd.col("a") | False,
            [True, False, True, False],
            "col('a') | False",
        ),
        (
            pd.col("a") ^ pd.col("b"),
            [True, True, False, True],
            "col('a') ^ col('b')",
        ),
        (
            pd.col("a") ^ True,
            [False, True, False, True],
            "col('a') ^ True",
        ),
        (
            ~pd.col("a"),
            [False, True, False, True],
            "~col('a')",
        ),
    ],
)
def test_col_logical_ops(
    expr: Expression, expected_values: list[bool], expected_str: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over (expression, expected values, expected string) for and, or, xor between
    # two boolean columns or a column and a scalar, and invert. Verifies assignment results and
    # string forms, then that using the expression as a row selector through label-based indexing
    # returns the rows matching the expected boolean values.
    # https://github.com/pandas-dev/pandas/issues/63322
    df = pd.DataFrame({"a": [True, False, True, False], "b": [False, True, True, True]})
    result = df.assign(c=expr)
    expected = pd.DataFrame(
        {
            "a": [True, False, True, False],
            "b": [False, True, True, True],
            "c": expected_values,
        }
    )
    tm.assert_frame_equal(result, expected)
    assert str(expr) == expected_str

    # Test that the expression works with .loc
    result = df.loc[expr]
    expected = df[expected_values]
    tm.assert_frame_equal(result, expected)


def test_expression_getitem() -> None:
    # [FUNCTION DOC]
    # Verifies item access on a column expression defers to the element at that position of the
    # column and broadcasts it; string form checked.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = pd.col("a")[1]
    expected_str = "col('a')[1]"

    assert str(expr) == expected_str

    result = df.assign(b=expr)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
    tm.assert_frame_equal(result, expected)


def test_property() -> None:
    # [FUNCTION DOC]
    # Verifies attribute access to a property (the index) on a column expression is deferred and
    # evaluates to the Series's index; string form checked.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = pd.col("a").index
    expected_str = "col('a').index"

    assert str(expr) == expected_str

    result = df.assign(b=expr)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})
    tm.assert_frame_equal(result, expected)


def test_cached_property() -> None:
    # [FUNCTION DOC]
    # Verifies accessing a cached property (the index's dtype, first asserted to be a cached
    # property on the Index class) through an expression is deferred; string form checked and the
    # assigned column compared with an expected frame.
    # https://github.com/pandas-dev/pandas/pull/63439
    # Ensure test is valid
    assert isinstance(pd.Index.dtype, cache_readonly)

    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = pd.col("a").index.dtype
    expected_str = "col('a').index.dtype"
    assert str(expr) == expected_str

    result = df.assign(b=expr)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": np.int64})
    tm.assert_frame_equal(result, expected)


def test_qcut() -> None:
    # [FUNCTION DOC]
    # Verifies the quantile-cut function accepts an expression, yields an expression whose string
    # form lists all its resolved arguments, and evaluates to the same result as applying it
    # directly to the column.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = pd.qcut(pd.col("a"), 3)
    expected_str = "qcut(x=col('a'), q=3, labels=None, retbins=False, precision=3)"
    assert str(expr) == expected_str, str(expr)

    result = df.assign(b=expr)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": pd.qcut(df["a"], 3)})
    tm.assert_frame_equal(result, expected)


def test_where() -> None:
    # [FUNCTION DOC]
    # Verifies the conditional-replacement method works with expression arguments, once with a
    # scalar replacement and once with an expression replacement, checking string forms and values.
    # https://github.com/pandas-dev/pandas/pull/63439
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = pd.col("a").where(pd.col("b") == 5, 100)
    expected_str = "col('a').where(col('b') == 5, 100)"
    assert str(expr) == expected_str, str(expr)

    result = df.assign(c=expr)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [100, 2, 100]})
    tm.assert_frame_equal(result, expected)

    expr = pd.col("a").where(pd.col("b") == 5, pd.col("a") + 1)
    expected_str = "col('a').where(col('b') == 5, col('a') + 1)"
    assert str(expr) == expected_str, str(expr)

    result = df.assign(c=expr)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [2, 2, 4]})
    tm.assert_frame_equal(result, expected)


# Unsupported ops
def test_bool():
    # [FUNCTION DOC]
    # Verifies converting an expression to a boolean raises a type error saying the truth value is ambiguous.
    with pytest.raises(TypeError, match="boolean value of an expression is ambiguous"):
        bool(pd.col("a"))


def test_iter():
    # [FUNCTION DOC]
    # Verifies iterating over an expression raises a type error saying expressions are not iterable.
    with pytest.raises(TypeError, match="Expression objects are not iterable"):
        iter(pd.col("a"))


def test_contains():
    # [FUNCTION DOC]
    # Verifies a membership test against an expression raises a type error whose message accepts
    # both older and newer interpreter wordings.
    # Python 3.14 changes the message from "is not iterable" to
    # "is not a container or iterable"
    with pytest.raises(
        TypeError, match="argument of type 'Expression' is not .*iterable"
    ):
        1 in pd.col("a")


def test_copy():
    # [FUNCTION DOC]
    # Verifies a shallow copy of an expression raises a type error saying expressions are not copiable.
    with pytest.raises(TypeError, match="Expression objects are not copiable"):
        copy.copy(pd.col("a"))


def test_deepcopy():
    # [FUNCTION DOC]
    # Verifies a deep copy of an expression raises a type error saying expressions are not copiable.
    with pytest.raises(TypeError, match="Expression objects are not copiable"):
        copy.deepcopy(pd.col("a"))


def test_divmod():
    # [FUNCTION DOC]
    # Verifies the combined quotient-and-remainder builtin raises a type error for an unsupported operand pair.
    msg = re.escape("unsupported operand type(s) for divmod(): 'Expression' and 'int'")
    with pytest.raises(TypeError, match=msg):
        divmod(pd.col("a"), 2)


def test_len():
    # [FUNCTION DOC]
    # Verifies taking the length of an expression raises a type error saying it has no length.
    with pytest.raises(TypeError, match="object of type 'Expression' has no len()"):
        len(pd.col("a"))


def test_round():
    # [FUNCTION DOC]
    # Verifies the round builtin raises a type error because the expression type defines no rounding method.
    msg = "type Expression doesn't define __round__ method"
    with pytest.raises(TypeError, match=msg):
        round(pd.col("a"), 2)
