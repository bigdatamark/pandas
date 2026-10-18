# Only tests that raise an error and have no better location should go here.
# Tests for specific groupby methods should go in their respective
# test file.
# [MODULE DOC]
# Tests that groupby reductions, transformations and aggregations raise the expected errors (or do not raise) for string, datetime, timedelta and categorical data, via direct method calls, aggregate by name, and transform by name; also user functions that raise propagate their error.
# Parametrized over many kinds of grouping keys (column names, lists, Grouper, callable, arrays, dict, series), frame vs selected column, and every groupby function from the shared fixtures. Per-function tables map a function name to the expected exception class (or none) and message pattern.

import datetime
import re

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Series,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


@pytest.fixture(
    params=[
        "a",
        ["a"],
        ["a", "b"],
        Grouper(key="a"),
        lambda x: x % 2,
        [0, 0, 0, 1, 2, 2, 2, 3, 3],
        np.array([0, 0, 0, 1, 2, 2, 2, 3, 3]),
        dict(zip(range(9), [0, 0, 0, 1, 2, 2, 2, 3, 3], strict=True)),
        Series([1, 1, 1, 1, 1, 2, 2, 2, 2]),
        [Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), Series([3, 3, 4, 4, 4, 4, 4, 3, 3])],
    ]
)
def by(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over ten different ways to specify the grouping (column name, list of names, Grouper, callable, list, array, mapping, series, list of series); returns the parameter.
    return request.param


@pytest.fixture(params=[True, False])
def groupby_series(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false, selecting whether the test groups the full frame or selects the single data column.
    return request.param


@pytest.fixture
def df_with_string_col():
    # [FUNCTION DOC]
    # HELPER: fixture returning a nine-row frame with integer key columns a and b, an integer column c and a string column d.
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": list("xyzwtyuio"),
        }
    )
    return df


@pytest.fixture
def df_with_datetime_col():
    # [FUNCTION DOC]
    # HELPER: fixture returning the same layout with a datetime column d (constant timestamp).
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": datetime.datetime(2005, 1, 1, 10, 30, 23, 540000),
        }
    )
    return df


@pytest.fixture
def df_with_cat_col():
    # [FUNCTION DOC]
    # HELPER: fixture returning the same layout with an ordered categorical column d with four categories (one unused).
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": Categorical(
                ["a", "a", "a", "a", "b", "b", "b", "b", "c"],
                categories=["a", "b", "c", "d"],
                ordered=True,
            ),
        }
    )
    return df


def _call_and_check(
    klass, msg, how, gb, groupby_func, args, warn_category=None, warn_msg=""
):
    # [FUNCTION DOC]
    # HELPER: runs the given operation (method call, aggregate by name or transform by name) on a groupby object inside a check for a warning of the given category and message (stacklevel not checked). If an exception class is given, the call must raise it with a message matching the pattern; otherwise it must succeed. Arguments: expected exception class or none, message pattern, how (method, agg or transform), groupby object, function name or callable, positional arguments, warning category and message.
    with tm.assert_produces_warning(
        warn_category, match=warn_msg, check_stacklevel=False
    ):
        if klass is None:
            if how == "method":
                getattr(gb, groupby_func)(*args)
            elif how == "agg":
                gb.agg(groupby_func, *args)
            else:
                gb.transform(groupby_func, *args)
        else:
            with pytest.raises(klass, match=msg):
                if how == "method":
                    getattr(gb, groupby_func)(*args)
                elif how == "agg":
                    gb.agg(groupby_func, *args)
                else:
                    gb.transform(groupby_func, *args)


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_string(
    how, by, groupby_series, groupby_func, df_with_string_col, using_infer_string
):
    # [FUNCTION DOC]
    # String data: for each groupby function and each call style, expects either success or a specific error (type, value or not-implemented) from a table; adjusts the table for string-inference mode (messages about the string dtype, type errors instead of value errors for some functions, pyarrow-specific messages for diff and percent change). For a selected column, the correlation-with-other function must be absent. Correlation-with-other expects a deprecation warning.
    df = df_with_string_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (TypeError, "Could not convert"),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(function|cummax) is not (implemented|supported) for (this|object) dtype",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(function|cummin) is not (implemented|supported) for (this|object) dtype",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(function|cumprod) is not (implemented|supported) for (this|object) dtype",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(function|cumsum) is not (implemented|supported) for (this|object) dtype",
        ),
        "diff": (TypeError, "unsupported operand type"),
        "ffill": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (
            TypeError,
            re.escape("agg function failed [how->mean,dtype->object]"),
        ),
        "median": (
            TypeError,
            re.escape("agg function failed [how->median,dtype->object]"),
        ),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "unsupported operand type"),
        "prod": (
            TypeError,
            re.escape("agg function failed [how->prod,dtype->object]"),
        ),
        "quantile": (TypeError, "dtype 'object' does not support operation 'quantile'"),
        "rank": (None, ""),
        "sem": (ValueError, "could not convert string to float"),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (ValueError, "could not convert string to float"),
        "kurt": (ValueError, "could not convert string to float"),
        "std": (ValueError, "could not convert string to float"),
        "sum": (None, ""),
        "var": (
            TypeError,
            re.escape("agg function failed [how->var,dtype->"),
        ),
    }[groupby_func]

    if using_infer_string:
        if groupby_func in [
            "prod",
            "mean",
            "median",
            "cumsum",
            "cumprod",
            "std",
            "sem",
            "var",
            "skew",
            "kurt",
            "quantile",
        ]:
            msg = f"dtype 'str' does not support operation '{groupby_func}'"
            if groupby_func in ["sem", "std", "skew", "kurt"]:
                # The object-dtype raises ValueError when trying to convert to numeric.
                klass = TypeError
        elif groupby_func == "pct_change" and df["d"].dtype.storage == "pyarrow":
            # This doesn't go through EA._groupby_op so the message isn't controlled
            #  there.
            msg = "operation 'truediv' not supported for dtype 'str' with dtype 'str'"
        elif groupby_func == "diff" and df["d"].dtype.storage == "pyarrow":
            # This doesn't go through EA._groupby_op so the message isn't controlled
            #  there.
            msg = "operation 'sub' not supported for dtype 'str' with dtype 'str'"

        elif groupby_func in ["cummin", "cummax"]:
            msg = msg.replace("object", "str")
        elif groupby_func == "corrwith":
            msg = "Cannot perform reduction 'mean' with string dtype"

    if groupby_func == "corrwith":
        warn_category = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_category = None
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_category, warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_string_udf(how, by, groupby_series, df_with_string_col):
    # [FUNCTION DOC]
    # User function raising a type error with a custom message propagates through aggregate and transform on string data, for all key types.
    df = df_with_string_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    def func(x):
        # [FUNCTION DOC]
        # HELPER: function that always raises a type error with a fixed message.
        raise TypeError("Test error message")

    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_string_np(
    how,
    by,
    groupby_series,
    groupby_func_np,
    df_with_string_col,
    using_infer_string,
):
    # [FUNCTION DOC]
    # Numpy sum and mean passed to aggregate and transform on string data (issue 50749): sum succeeds, mean raises a type error about converting a string or reducing a string dtype; messages differ under string inference.
    # GH#50749
    df = df_with_string_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    klass, msg = {
        np.sum: (None, ""),
        np.mean: (
            TypeError,
            "Could not convert string .* to numeric|"
            "Cannot perform reduction 'mean' with string dtype",
        ),
    }[groupby_func_np]

    if using_infer_string:
        if groupby_func_np is np.mean:
            klass = TypeError
        msg = f"Cannot perform reduction '{groupby_func_np.__name__}' with string dtype"

    _call_and_check(klass, msg, how, gb, groupby_func_np, ())


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_datetime(
    how, by, groupby_series, groupby_func, df_with_datetime_col
):
    # [FUNCTION DOC]
    # Datetime data: for each groupby function and call style, expects success or a type error with a specific message from a table (for example boolean reductions not supported on datetime, product and sum unsupported, percent change and correlation operand errors, skew and kurtosis dtype errors).
    df = df_with_datetime_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    klass, msg = {
        "all": (TypeError, "'all' with datetime64 dtypes is no longer supported"),
        "any": (TypeError, "'any' with datetime64 dtypes is no longer supported"),
        "bfill": (None, ""),
        "corrwith": (TypeError, "cannot perform __mul__ with this index type"),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (None, ""),
        "cummin": (None, ""),
        "cumprod": (TypeError, "datetime64 type does not support operation 'cumprod'"),
        "cumsum": (TypeError, "datetime64 type does not support operation 'cumsum'"),
        "diff": (None, ""),
        "ffill": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (None, ""),
        "median": (None, ""),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "cannot perform __truediv__ with this index type"),
        "prod": (TypeError, "datetime64 type does not support operation 'prod'"),
        "quantile": (None, ""),
        "rank": (None, ""),
        "sem": (None, ""),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    r"dtype datetime64\[ns\] does not support operation",
                    "datetime64 type does not support operation 'skew'",
                ]
            ),
        ),
        "kurt": (
            TypeError,
            "|".join(
                [
                    r"dtype datetime64\[ns\] does not support operation",
                    "datetime64 type does not support operation 'kurt'",
                ]
            ),
        ),
        "std": (None, ""),
        "sum": (TypeError, "datetime64 type does not support operation 'sum"),
        "var": (TypeError, "datetime64 type does not support operation 'var'"),
    }[groupby_func]

    if groupby_func == "corrwith":
        warn_category = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_category = None
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_category, warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_datetime_udf(how, by, groupby_series, df_with_datetime_col):
    # [FUNCTION DOC]
    # User function raising a type error with a custom message propagates through aggregate and transform on datetime data.
    df = df_with_datetime_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    def func(x):
        # [FUNCTION DOC]
        # HELPER: function that always raises a type error with a fixed message.
        raise TypeError("Test error message")

    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_datetime_np(
    how, by, groupby_series, groupby_func_np, df_with_datetime_col
):
    # [FUNCTION DOC]
    # Numpy sum and mean on datetime data (issue 50749): sum raises a type error about the microsecond datetime dtype, mean succeeds.
    # GH#50749
    df = df_with_datetime_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    klass, msg = {
        np.sum: (
            TypeError,
            re.escape("datetime64[us] does not support operation 'sum'"),
        ),
        np.mean: (None, ""),
    }[groupby_func_np]
    _call_and_check(klass, msg, how, gb, groupby_func_np, ())


@pytest.mark.parametrize("func", ["prod", "cumprod", "skew", "kurt", "var"])
def test_groupby_raises_timedelta(func):
    # [FUNCTION DOC]
    # Product, cumulative product, skew, kurtosis and variance on timedelta data raise a type error saying timedelta does not support those operations.
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": datetime.timedelta(days=1),
        }
    )
    gb = df.groupby(by="a")

    _call_and_check(
        TypeError,
        "timedelta64 type does not support .* operations",
        "method",
        gb,
        func,
        [],
    )


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category(
    how, by, groupby_series, groupby_func, df_with_cat_col
):
    # [FUNCTION DOC]
    # Categorical data (issue 50749): for each groupby function and call style, expects success (order-based functions on an ordered categorical) or a type or not-implemented error with one of several accepted messages from a table.
    # GH#50749
    df = df_with_cat_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (
            TypeError,
            r"unsupported operand type\(s\) for \*: 'Categorical' and 'int'",
        ),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(category type does not support cummax operations|"
            "category dtype not supported|"
            "cummax is not supported for category dtype)",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(category type does not support cummin operations|"
            "category dtype not supported|"
            "cummin is not supported for category dtype)",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(category type does not support cumprod operations|"
            "category dtype not supported|"
            "cumprod is not supported for category dtype)",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(category type does not support cumsum operations|"
            "category dtype not supported|"
            "cumsum is not supported for category dtype)",
        ),
        "diff": (
            TypeError,
            r"unsupported operand type\(s\) for -: 'Categorical' and 'Categorical'",
        ),
        "ffill": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'mean'",
                    "category dtype does not support aggregation 'mean'",
                ]
            ),
        ),
        "median": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'median'",
                    "category dtype does not support aggregation 'median'",
                ]
            ),
        ),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (
            TypeError,
            r"unsupported operand type\(s\) for /: 'Categorical' and 'Categorical'",
        ),
        "prod": (TypeError, "category type does not support prod operations"),
        "quantile": (TypeError, "No matching signature found"),
        "rank": (None, ""),
        "sem": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'sem'",
                    "category dtype does not support aggregation 'sem'",
                ]
            ),
        ),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    "dtype category does not support operation 'skew'",
                    "category type does not support skew operations",
                ]
            ),
        ),
        "kurt": (
            TypeError,
            "|".join(
                [
                    "dtype category does not support operation 'kurt'",
                    "category type does not support kurt operations",
                ]
            ),
        ),
        "std": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'std'",
                    "category dtype does not support aggregation 'std'",
                ]
            ),
        ),
        "sum": (TypeError, "category type does not support sum operations"),
        "var": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'var'",
                    "category dtype does not support aggregation 'var'",
                ]
            ),
        ),
    }[groupby_func]

    if groupby_func == "corrwith":
        warn_category = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_category = None
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_category, warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_category_udf(how, by, groupby_series, df_with_cat_col):
    # [FUNCTION DOC]
    # User function raising a type error with a custom message propagates through aggregate and transform on categorical data.
    # GH#50749
    df = df_with_cat_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    def func(x):
        # [FUNCTION DOC]
        # HELPER: function that always raises a type error with a fixed message.
        raise TypeError("Test error message")

    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_category_np(
    how, by, groupby_series, groupby_func_np, df_with_cat_col
):
    # [FUNCTION DOC]
    # Numpy sum and mean on categorical data (issue 50749): both raise a type error saying the category dtype does not support the operation.
    # GH#50749
    df = df_with_cat_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    klass, msg = {
        np.sum: (TypeError, "dtype category does not support operation 'sum'"),
        np.mean: (
            TypeError,
            "dtype category does not support operation 'mean'",
        ),
    }[groupby_func_np]
    _call_and_check(klass, msg, how, gb, groupby_func_np, ())


@pytest.mark.filterwarnings("ignore:In a future version, the keys")
@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category_on_category(
    how,
    by,
    groupby_series,
    groupby_func,
    observed,
    df_with_cat_col,
):
    # [FUNCTION DOC]
    # Categorical data where the grouping key column is also categorical (issue 50749), parametrized additionally over observed. When observed is False and groups are empty, the index-of-max and index-of-min functions raise a value error about unobserved categories (except for transform, where empty groups are ignored); other expectations are the same as in the categorical-data test with slightly different messages. Ignores a future-warning about keys.
    # GH#50749
    df = df_with_cat_col
    df["a"] = Categorical(
        ["a", "a", "a", "a", "b", "b", "b", "b", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by, observed=observed)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    empty_groups = not observed and any(group.empty for group in gb.groups.values())
    if how == "transform":
        # empty groups will be ignored
        empty_groups = False

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (
            TypeError,
            r"unsupported operand type\(s\) for \*: 'Categorical' and 'int'",
        ),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(cummax is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cummax operations)",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(cummin is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cummin operations)",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(cumprod is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cumprod operations)",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(cumsum is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cumsum operations)",
        ),
        "diff": (TypeError, "unsupported operand type"),
        "ffill": (None, ""),
        "first": (None, ""),
        "idxmax": (ValueError, "empty group due to unobserved categories")
        if empty_groups
        else (None, ""),
        "idxmin": (ValueError, "empty group due to unobserved categories")
        if empty_groups
        else (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (TypeError, "category dtype does not support aggregation 'mean'"),
        "median": (TypeError, "category dtype does not support aggregation 'median'"),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "unsupported operand type"),
        "prod": (TypeError, "category type does not support prod operations"),
        "quantile": (TypeError, "No matching signature found"),
        "rank": (None, ""),
        "sem": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'sem'",
                    "category dtype does not support aggregation 'sem'",
                ]
            ),
        ),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    "category type does not support skew operations",
                    "dtype category does not support operation 'skew'",
                ]
            ),
        ),
        "kurt": (
            TypeError,
            "|".join(
                [
                    "category type does not support kurt operations",
                    "dtype category does not support operation 'kurt'",
                ]
            ),
        ),
        "std": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'std'",
                    "category dtype does not support aggregation 'std'",
                ]
            ),
        ),
        "sum": (TypeError, "category type does not support sum operations"),
        "var": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support operation 'var'",
                    "category dtype does not support aggregation 'var'",
                ]
            ),
        ),
    }[groupby_func]

    if groupby_func == "corrwith":
        warn_category = Pandas4Warning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn_category = None
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_category, warn_msg)
