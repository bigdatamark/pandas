# [MODULE DOC]
# Tests for Series.case_when, which replaces values by the first matching
# condition among a list of (condition, replacement) pairs. Covers argument
# validation, scalar/list/array/Series/extension replacements, non-range indexes,
# callables, and column expressions (col) inside DataFrame assign. Defines a
# shared DataFrame fixture. No import-time side effects.

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    array as pd_array,
    col,
    date_range,
)
import pandas._testing as tm


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a three-row DataFrame with integer columns a
    # (1, 2, 3) and b (4, 5, 6), shared by the case_when tests.
    """
    base dataframe for testing
    """
    return DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_case_when_caselist_is_not_a_list(df):
    # [FUNCTION DOC]
    # A caselist given as a tuple instead of a list raises a type error saying the
    # argument must be a list (GH39154).
    # KNOWN ISSUE: the docstring says a value error is raised, but the test
    # expects a type error.
    """
    Raise ValueError if caselist is not a list.
    """
    msg = "The caselist argument should be a list; "
    msg += "instead got.+"
    with pytest.raises(TypeError, match=msg):  # GH39154
        df["a"].case_when(caselist=())


def test_case_when_no_caselist(df):
    # [FUNCTION DOC]
    # An empty caselist raises a value error asking for at least one condition with a
    # corresponding replacement.
    """
    Raise ValueError if no caselist is provided.
    """
    msg = "provide at least one boolean condition, "
    msg += "with a corresponding replacement."
    with pytest.raises(ValueError, match=msg):  # GH39154
        df["a"].case_when([])


def test_case_when_odd_caselist(df):
    # [FUNCTION DOC]
    # A caselist element with three items instead of two raises a value error that
    # the element must have length 2.
    # KNOWN ISSUE: the docstring says the caselist length is odd, but the test
    # makes one element of the list have three items.
    """
    Raise ValueError if no of caselist is odd.
    """
    msg = "Argument 0 must have length 2; "
    msg += "a condition and replacement; instead got length 3."

    with pytest.raises(ValueError, match=msg):
        df["a"].case_when([(df["a"].eq(1), 1, df.a.gt(1))])


def test_case_when_raise_error_from_mask(df):
    # [FUNCTION DOC]
    # A replacement that cannot be applied by mask (a list of the wrong length) raises
    # a value error saying it failed to apply the first condition and replacement.
    """
    Raise Error from within Series.mask
    """
    msg = "Failed to apply condition0 and replacement0."
    with pytest.raises(ValueError, match=msg):
        df["a"].case_when([(df["a"].eq(1), [1, 2])])


def test_case_when_single_condition(df):
    # [FUNCTION DOC]
    # One condition on an all-NaN Series: only the matching position receives the
    # replacement; the rest stay NaN.
    """
    Test output on a single condition.
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([(df.a.eq(1), 1)])
    expected = Series([1, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions(df):
    # [FUNCTION DOC]
    # Two conditions, one a comparison of a column and one an explicit boolean
    # Series: positions get the replacement of the first matching condition.
    """
    Test output when booleans are derived from a computation
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [(df.a.eq(1), 1), (Series([False, True, False]), 2)]
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions_replacement_list(df):
    # [FUNCTION DOC]
    # Conditions given as list and computed boolean Series with scalar and list
    # replacements: the matching positions receive the corresponding replacement
    # values.
    """
    Test output when replacement is a list
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [([True, False, False], 1), (df["a"].gt(1) & df["b"].eq(5), [1, 2, 3])]
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions_replacement_extension_dtype(df):
    # [FUNCTION DOC]
    # Replacement that is an Int64 extension array together with a scalar: the result
    # has Float64 dtype with the replaced values and NaN elsewhere.
    """
    Test output when replacement has an extension dtype
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [
            ([True, False, False], 1),
            (df["a"].gt(1) & df["b"].eq(5), pd_array([1, 2, 3], dtype="Int64")),
        ],
    )
    expected = Series([1, 2, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions_replacement_series(df):
    # [FUNCTION DOC]
    # Replacement that is a Series (with a NumPy array condition): matching positions
    # take the aligned replacement values.
    """
    Test output when replacement is a Series
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [
            (np.array([True, False, False]), 1),
            (df["a"].gt(1) & df["b"].eq(5), Series([1, 2, 3])),
        ],
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_non_range_index():
    # [FUNCTION DOC]
    # A Series with a datetime index (not a range index) and a column as replacement:
    # equals masking where the condition holds and filling the rest with the default
    # value 5.
    """
    Test output if index is not RangeIndex
    """
    rng = np.random.default_rng(seed=123)
    dates = date_range("1/1/2000", periods=8)
    df = DataFrame(
        rng.standard_normal(size=(8, 4)), index=dates, columns=["A", "B", "C", "D"]
    )
    result = Series(5, index=df.index, name="A").case_when([(df.A.gt(0), df.B)])
    expected = df.A.mask(df.A.gt(0), df.B).where(df.A.gt(0), 5)
    tm.assert_series_equal(result, expected)


def test_case_when_callable():
    # [FUNCTION DOC]
    # Callable conditions and replacements are applied to the Series itself; the
    # result equals NumPy piecewise evaluation of the same rule (negate negatives,
    # keep non-negatives).
    """
    Test output on a callable
    """
    # https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html
    x = np.linspace(-2.5, 2.5, 6)
    ser = Series(x)
    result = ser.case_when(
        caselist=[
            (lambda df: df < 0, lambda df: -df),
            (lambda df: df >= 0, lambda df: df),
        ]
    )
    expected = np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    tm.assert_series_equal(result, Series(expected))


def test_case_when_expression_condition(df):
    # [FUNCTION DOC]
    # GH 64070: a column expression with a comparison condition and scalar
    # replacements used inside assign gives the expected new column.
    # GH 64070
    result = df.assign(out=col("a").case_when([(col("a") > 1, 10), (col("a") <= 1, 5)]))
    expected = df.assign(out=Series([5, 10, 10], index=df.index))
    tm.assert_frame_equal(result, expected)


def test_case_when_expression_replacement(df):
    # [FUNCTION DOC]
    # GH 64070: a column expression whose replacements are themselves column
    # expressions plus a number, with Series conditions, inside assign.
    # GH 64070
    result = df.assign(
        out=col("a").case_when(
            [(df["a"] > 1, col("a") + 100), (df["a"] <= 1, col("a") + 1)]
        )
    )
    expected = df.assign(out=Series([2, 102, 103], index=df.index))
    tm.assert_frame_equal(result, expected)


def test_case_when_expression_in_assign():
    # [FUNCTION DOC]
    # GH 64070: with a single condition using an expression, positions that do not
    # match keep their original values in the assigned column.
    # GH 64070
    df = DataFrame({"age": [65, 30], "name": ["Jason", "Anna"]})
    result = df.assign(elderly=col("age").case_when([(col("name").eq("Jason"), 1)]))
    expected = df.assign(elderly=Series([1, 30], index=df.index))
    tm.assert_frame_equal(result, expected)


def test_case_when_expression_mixed_args_in_assign():
    # [FUNCTION DOC]
    # GH 64070: mixes a callable condition and a column-expression replacement with
    # an expression condition; the first matching rule wins.
    # GH 64070
    df = DataFrame({"name": ["Jason", "Amy", "Bob"], "age": [42, 10, 5]})
    caselist = [
        (lambda s: s < 10, col("age") + 100),
        (col("name").eq("Jason"), 1),
    ]
    result = df.assign(elderly=col("age").case_when(caselist=caselist))
    expected = df.assign(elderly=Series([1, 10, 105], index=df.index))
    tm.assert_frame_equal(result, expected)


def test_case_when_expression_math_condition_in_assign():
    # [FUNCTION DOC]
    # GH 64070: the condition is a sum of two column expressions compared to zero;
    # matching rows are replaced by a constant.
    # GH 64070
    df = DataFrame({"a": [-1, 1, 2], "b": [0, -2, 1]})
    caselist = [(col("a") + col("b") > 0, 99)]
    result = df.assign(flag=col("a").case_when(caselist=caselist))
    expected = df.assign(flag=Series([-1, 1, 99], index=df.index))
    tm.assert_frame_equal(result, expected)


def test_case_when_expression_missing_values_in_assign():
    # [FUNCTION DOC]
    # GH 64070: conditions based on missing-value tests and numeric comparison map to
    # text labels, leaving unmatched values as the original number; the result has
    # object dtype.
    # GH 64070
    df = DataFrame({"age": [np.nan, 5.0, 20.0]})
    caselist = [
        (col("age").isna(), "missing"),
        (col("age") > 10, "adult"),
    ]
    result = df.assign(group=col("age").case_when(caselist=caselist))
    expected = df.assign(
        group=Series(["missing", 5.0, "adult"], index=df.index, dtype=object)
    )
    tm.assert_frame_equal(result, expected)
