# [MODULE DOC]
# Tests for the numeric_only option of groupby reductions and transformations: which columns remain in the result by default, with numeric_only True, and with numeric_only False, and which errors are raised for non-numeric columns.
# Provides a class of per-family tests sharing a mixed-dtype frame fixture, a test over all kernels (with the shared method-argument helper) and a test of the series-level behavior.
import re

import pytest

from pandas._libs import lib
from pandas.errors import Pandas4Warning

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


class TestNumericOnly:
    # [CLASS DOC]
    # Groups tests that run one family of groupby methods against a frame with int, float, string, object, categorical, datetime, timezone-aware datetime and timedelta columns grouped by a group column; the shared checker verifies default and explicit numeric_only behavior. Holds the frame fixture, one test per method family and the checker helper.
    # make sure that we are passing thru kwargs to our agg functions

    @pytest.fixture
    def df(self):
        # [FUNCTION DOC]
        # Fixture building the mixed-dtype frame (issues 3668, 5724): a group column plus integer, float, string, object, categorical (string and integer), datetime, timezone-aware datetime and timedelta columns, three rows.
        # GH3668
        # GH5724
        df = DataFrame(
            {
                "group": [1, 1, 2],
                "int": [1, 2, 3],
                "float": [4.0, 5.0, 6.0],
                "string": Series(["a", "b", "c"], dtype="str"),
                "object": Series(["a", "b", "c"], dtype=object),
                "category_string": Series(list("abc")).astype("category"),
                "category_int": [7, 8, 9],
                "datetime": date_range("20130101", periods=3),
                "datetimetz": date_range("20130101", periods=3, tz="US/Eastern"),
                "timedelta": pd.timedelta_range("1 s", periods=3, freq="s"),
            },
            columns=[
                "group",
                "int",
                "float",
                "string",
                "object",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ],
        )
        return df

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_averages(self, df, method):
        # [FUNCTION DOC]
        # Parametrized over mean and median: with numeric_only True the numeric columns (int, float, integer categorical, as float means) are returned and match the expectation reindexed like the expected frame; then the common checker verifies default and explicit False behavior.
        # mean / median
        expected_columns_numeric = Index(["int", "float", "category_int"])

        gb = df.groupby("group")
        expected = DataFrame(
            {
                "category_int": [7.5, 9],
                "float": [4.5, 6.0],
                "timedelta": [pd.Timedelta("1.5s"), pd.Timedelta("3s")],
                "int": [1.5, 3],
                "datetime": [
                    Timestamp("2013-01-01 12:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                "datetimetz": [
                    Timestamp("2013-01-01 12:00:00", tz="US/Eastern"),
                    Timestamp("2013-01-03 00:00:00", tz="US/Eastern"),
                ],
            },
            index=Index([1, 2], name="group"),
            columns=[
                "int",
                "float",
                "category_int",
            ],
        )

        result = getattr(gb, method)(numeric_only=True)
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        expected_columns = expected.columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_extrema(self, df, method):
        # [FUNCTION DOC]
        # Parametrized over minimum and maximum: expects the columns that support ordering (not object or unordered categorical), then runs the checker. A TODO in the source notes ordered categoricals should eventually be supported.
        # TODO: min, max *should* handle
        # categorical (ordered) dtype

        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["first", "last"])
    def test_first_last(self, df, method):
        # [FUNCTION DOC]
        # Parametrized over first and last: all columns are valid, including object and categorical, for both the default and numeric_only False.
        expected_columns = Index(
            [
                "int",
                "float",
                "string",
                "object",
                "category_string",
                "category_int",
                "datetime",
                "datetimetz",
                "timedelta",
            ]
        )
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["sum", "cumsum"])
    def test_sum_cumsum(self, df, method):
        # [FUNCTION DOC]
        # Parametrized over sum and cumulative sum: numeric_only True returns int, float and integer categorical; the string column is kept for sum but lost for cumulative sum; timedelta is included.
        expected_columns_numeric = Index(["int", "float", "category_int"])
        expected_columns = Index(
            ["int", "float", "string", "category_int", "timedelta"]
        )
        if method == "cumsum":
            # cumsum loses string
            expected_columns = Index(["int", "float", "category_int", "timedelta"])

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["prod", "cumprod"])
    def test_prod_cumprod(self, df, method):
        # [FUNCTION DOC]
        # Parametrized over product and cumulative product: only int, float and integer categorical columns are valid in all modes.
        expected_columns = Index(["int", "float", "category_int"])
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, df, method):
        # [FUNCTION DOC]
        # Parametrized over cumulative minimum and maximum: includes datetimes and timedeltas but not strings; the default is numeric_only False like minimum and maximum (issue 15561).
        # like min, max, but don't include strings
        expected_columns = Index(
            ["int", "float", "category_int", "datetime", "datetimetz", "timedelta"]
        )

        # GH#15561: numeric_only=False set by default like min/max
        expected_columns_numeric = expected_columns

        self._check(df, method, expected_columns, expected_columns_numeric)

    def _check(self, df, method, expected_columns, expected_columns_numeric):
        # [FUNCTION DOC]
        # HELPER: shared checker for the family tests. Groups by the group column and, depending on the method, verifies: the default call (and, for non-first/last methods, the numeric_only False call) either raises a type error (or not-implemented error for cumulative methods) with one of several accepted messages when nuisance columns cannot be handled, or returns a frame whose columns equal the expected numeric column index (for default) or the expected full column index (for numeric_only False, first/last only). Parameters: the frame, the method name, the expected column index for numeric_only False, and the expected index for the default call.
        gb = df.groupby("group")

        # object dtypes for transformations are not implemented in Cython and
        # have no Python fallback
        exception = (
            (NotImplementedError, TypeError) if method.startswith("cum") else TypeError
        )

        if method in ("min", "max", "cummin", "cummax", "cumsum", "cumprod"):
            # The methods default to numeric_only=False and raise TypeError
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    # cumsum/cummin/cummax/cumprod
                    "function is not implemented for this dtype",
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        elif method in ("sum", "mean", "median", "prod"):
            msg = "|".join(
                [
                    "category type does not support sum operations",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    re.escape(f"agg function failed [how->{method},dtype->string]"),
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        else:
            result = getattr(gb, method)()
            tm.assert_index_equal(result.columns, expected_columns_numeric)

        if method not in ("first", "last"):
            msg = "|".join(
                [
                    "Categorical is not ordered",
                    "category type does not support",
                    "function is not implemented for this dtype",
                    f"Cannot perform {method} with non-ordered Categorical",
                    re.escape(f"agg function failed [how->{method},dtype->object]"),
                    re.escape(f"agg function failed [how->{method},dtype->string]"),
                    f"dtype 'str' does not support operation '{method}'",
                ]
            )
            with pytest.raises(exception, match=msg):
                getattr(gb, method)(numeric_only=False)
        else:
            result = getattr(gb, method)(numeric_only=False)
            tm.assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize(
    "kernel, has_arg",
    [
        ("all", False),
        ("any", False),
        ("bfill", False),
        ("corr", True),
        ("corrwith", True),
        ("cov", True),
        ("cummax", True),
        ("cummin", True),
        ("cumprod", True),
        ("cumsum", True),
        ("diff", False),
        ("ffill", False),
        ("first", True),
        ("idxmax", True),
        ("idxmin", True),
        ("last", True),
        ("max", True),
        ("mean", True),
        ("median", True),
        ("min", True),
        ("nth", False),
        ("nunique", False),
        ("pct_change", False),
        ("prod", True),
        ("quantile", True),
        ("sem", True),
        ("skew", True),
        ("kurt", True),
        ("std", True),
        ("sum", True),
        ("var", True),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False, lib.no_default])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_numeric_only(kernel, has_arg, numeric_only, keys):
    # [FUNCTION DOC]
    # Parametrized over many kernels, whether they accept a numeric_only argument, numeric_only values (True, False or unspecified) and one or two keys (issue 46072): on a frame with one object column, checks that the object column is absent with numeric_only True; present for first/last and for dtype-agnostic kernels without the argument; errors with kernel-specific messages when numeric_only is False and the kernel needs numeric data; an unexpected-keyword type error for kernels without the argument when it is given; and for diff and percent change a type error from unsupported operands. Expects a deprecation warning for the correlation-with-other kernel.
    # GH#46072
    # drops_nuisance: Whether the op drops nuisance columns even when numeric_only=False
    # has_arg: Whether the op has a numeric_only arg
    df = DataFrame({"a1": [1, 1], "a2": [2, 2], "a3": [5, 6], "b": 2 * [object]})

    args = get_groupby_method_args(kernel, df)
    kwargs = {} if numeric_only is lib.no_default else {"numeric_only": numeric_only}

    gb = df.groupby(keys)
    method = getattr(gb, kernel)
    if has_arg and numeric_only is True:
        # Cases where b does not appear in the result
        if kernel == "corrwith":
            warn = Pandas4Warning
            msg = "DataFrameGroupBy.corrwith is deprecated"
        else:
            warn = None
            msg = ""
        with tm.assert_produces_warning(warn, match=msg):
            result = method(*args, **kwargs)
        assert "b" not in result.columns
    elif (
        # kernels that work on any dtype and have numeric_only arg
        kernel in ("first", "last")
        or (
            # kernels that work on any dtype and don't have numeric_only arg
            kernel in ("any", "all", "bfill", "ffill", "nth", "nunique")
            and numeric_only is lib.no_default
        )
    ):
        result = method(*args, **kwargs)
        assert "b" in result.columns
    elif has_arg:
        assert numeric_only is not True
        # kernels that are successful on any dtype were above; this will fail

        # object dtypes for transformations are not implemented in Cython and
        # have no Python fallback
        exception = NotImplementedError if kernel.startswith("cum") else TypeError

        msg = "|".join(
            [
                "not allowed for this dtype",
                "cannot be performed against 'object' dtypes",
                "must be a string or a real number",
                "unsupported operand type",
                "function is not implemented for this dtype",
                re.escape(f"agg function failed [how->{kernel},dtype->object]"),
            ]
        )
        if kernel == "quantile":
            msg = "dtype 'object' does not support operation 'quantile'"
        elif kernel == "idxmin":
            msg = "'<' not supported between instances of 'type' and 'type'"
        elif kernel == "idxmax":
            msg = "'>' not supported between instances of 'type' and 'type'"
        with pytest.raises(exception, match=msg):
            if kernel == "corrwith":
                warn = Pandas4Warning
                msg = "DataFrameGroupBy.corrwith is deprecated"
            else:
                warn = None
                msg = ""
            with tm.assert_produces_warning(warn, match=msg):
                method(*args, **kwargs)
    elif not has_arg and numeric_only is not lib.no_default:
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'numeric_only'"
        ):
            method(*args, **kwargs)
    else:
        assert kernel in ("diff", "pct_change")
        assert numeric_only is lib.no_default
        # Doesn't have numeric_only argument and fails on nuisance columns
        with pytest.raises(TypeError, match=r"unsupported operand type"):
            method(*args, **kwargs)


@pytest.mark.parametrize("dtype", [bool, int, float, object])
def test_deprecate_numeric_only_series(dtype, groupby_func, request):
    # [FUNCTION DOC]
    # Issue 46560: for series groupby over bool, int, float and object dtype and each groupby function, checks default behavior on object dtype (type error for kernels not supported on object, otherwise result equal to the numeric result cast to object where appropriate) and the numeric_only keyword (unexpected-keyword error when unsupported; type error on object dtype; quantile on bool raises; otherwise numeric_only True and False agree). The correlation-with-other kernel is asserted to be absent from series groupby and the test returns early.
    # GH#46560
    grouper = [0, 0, 1]

    ser = Series([1, 0, 0], dtype=dtype)
    gb = ser.groupby(grouper)

    if groupby_func == "corrwith":
        # corrwith is not implemented on SeriesGroupBy
        assert not hasattr(gb, groupby_func)
        return

    method = getattr(gb, groupby_func)

    expected_ser = Series([1, 0, 0])
    expected_gb = expected_ser.groupby(grouper)
    expected_method = getattr(expected_gb, groupby_func)

    args = get_groupby_method_args(groupby_func, ser)

    fails_on_numeric_object = (
        "corr",
        "cov",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "quantile",
    )
    # ops that give an object result on object input
    obj_result = (
        "first",
        "last",
        "nth",
        "bfill",
        "ffill",
        "shift",
        "sum",
        "diff",
        "pct_change",
        "var",
        "mean",
        "median",
        "min",
        "max",
        "prod",
        "skew",
        "kurt",
    )

    # Test default behavior; kernels that fail may be enabled in the future but kernels
    # that succeed should not be allowed to fail (without deprecation, at least)
    if groupby_func in fails_on_numeric_object and dtype is object:
        if groupby_func == "quantile":
            msg = "dtype 'object' does not support operation 'quantile'"
        else:
            msg = "is not supported for object dtype"
        with pytest.raises(TypeError, match=msg):
            method(*args)
    elif dtype is object:
        result = method(*args)
        expected = expected_method(*args)
        if groupby_func in obj_result:
            expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    has_numeric_only = (
        "first",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "sem",
        "skew",
        "kurt",
        "std",
        "sum",
        "var",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
    )
    if groupby_func not in has_numeric_only:
        msg = "got an unexpected keyword argument 'numeric_only'"
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype is object:
        msg = "|".join(
            [
                "SeriesGroupBy.sem called with numeric_only=True and dtype object",
                "Series.skew does not allow numeric_only=True with non-numeric",
                "cum(sum|prod|min|max) is not supported for object dtype",
                r"Cannot use numeric_only=True with SeriesGroupBy\..* and non-numeric",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, numeric_only=True)
    elif dtype == bool and groupby_func == "quantile":
        msg = "Cannot use quantile with bool dtype"
        with pytest.raises(TypeError, match=msg):
            # GH#51424
            method(*args, numeric_only=False)
    else:
        result = method(*args, numeric_only=True)
        expected = method(*args, numeric_only=False)
        tm.assert_series_equal(result, expected)
