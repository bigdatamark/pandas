# [MODULE DOC]
# Tests for statistical and boolean reductions on the tabular container
# (count, sum, mean, product, median, min, max, nunique, variance, standard
# deviation, standard error, skew, kurtosis, mode, quantile-like and
# any/all, idxmin/idxmax, cumulative-free), covering axis handling, missing-value
# handling, the numeric-only flag, mixed and extension dtypes, datetime-like
# data, empty frames, and result dtypes. Defines module-level platform flags
# (Windows with older NumPy or 32-bit; Windows or 32-bit) used to select
# expected dtypes, helper functions that compare a reduction to a reference
# implementation, and fixtures of frames with missing values. Uses fixtures from
# the test-suite conftest; no import-time side effects beyond evaluating the
# platform flags.
from datetime import timedelta
from decimal import Decimal
import re

from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeIndex,
    Index,
    PeriodIndex,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core import (
    algorithms,
    nanops,
)

is_windows_np2_or_is32 = (is_platform_windows() and not np_version_gt2) or not IS64
is_windows_or_is32 = is_platform_windows() or not IS64


def make_skipna_wrapper(alternative, skipna_alternative=None):
    # [FUNCTION DOC]
    # HELPER: for the reduction tests. Builds a wrapper to apply to a Series for the
    # reference implementation of a reduction. Parameters: alternative is a callable
    # used on the values with missing entries removed; skipna_alternative is an
    # optional callable that is instead applied directly to the raw values (so it must
    # handle missing values itself). Returns the wrapper: with skipna_alternative it
    # applies that to the raw values; otherwise it drops missing values and returns a
    # missing marker when nothing is left, else the alternative on the remainder.
    """
    Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    function
    """
    if skipna_alternative:

        def skipna_wrapper(x):
            # [FUNCTION DOC]
            # HELPER: for the wrapper builder. Applies the skipna-aware alternative to the
            # raw values of the given Series.
            return skipna_alternative(x.values)

    else:

        def skipna_wrapper(x):
            # [FUNCTION DOC]
            # HELPER: for the wrapper builder. Drops missing values of the given Series;
            # returns NaN if empty, otherwise the alternative applied to the cleaned Series.
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def assert_stat_op_calc(
    opname,
    alternative,
    frame,
    has_skipna=True,
    check_dtype=True,
    check_dates=False,
    rtol=1e-5,
    atol=1e-8,
    skipna_alternative=None,
):
    # [FUNCTION DOC]
    # HELPER: shared checker for reduction tests. Verifies that a named frame
    # reduction agrees with a reference function. Parameters: opname is the method
    # name; alternative is the reference callable; frame is the data; has_skipna says
    # whether the method accepts the skip-missing flag (when true, also the non-skipping
    # results are compared against column and row application of the reference on
    # raw values); check_dtype whether result dtypes are compared; check_dates also
    # runs the method on a datetime-only frame and a datetime-plus-integer frame and
    # requires a Series result without warnings; rtol and atol are numeric tolerances;
    # skipna_alternative is the missing-safe reference. Checks axis zero equals
    # column-wise application and, for sum and product, axis one equals row-wise
    # application with relaxed dtype. When dtype checking, results have the frame's
    # common dtype. An invalid axis raises a value error naming the axis. For
    # methods with skip flag on an all-missing frame, sum and product give the
    # identity (zero or one) for each label.
    # KNOWN ISSUE: the docstring calls the keyword a skip_na form while the actual keyword name is skipna.
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """
    f = getattr(frame, opname)

    if check_dates:
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)

        df["a"] = range(len(df))
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        assert len(result)

    if has_skipna:

        def wrapper(x):
            # [FUNCTION DOC]
            # HELPER: for the checker. Applies the reference callable to the raw values of
            # the given Series (no missing handling).
            return alternative(x.values)

        skipna_wrapper = make_skipna_wrapper(alternative, skipna_alternative)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(
            result0, frame.apply(wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol
        )
        tm.assert_series_equal(
            result1,
            frame.apply(wrapper, axis=1),
            rtol=rtol,
            atol=atol,
        )
    else:
        skipna_wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(
        result0,
        frame.apply(skipna_wrapper),
        check_dtype=check_dtype,
        rtol=rtol,
        atol=atol,
    )

    if opname in ["sum", "prod"]:
        expected = frame.apply(skipna_wrapper, axis=1)
        tm.assert_series_equal(
            result1, expected, check_dtype=False, rtol=rtol, atol=atol
        )

    # check dtypes
    if check_dtype:
        lcd_dtype = frame.values.dtype
        assert lcd_dtype == result0.dtype
        assert lcd_dtype == result1.dtype

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.nan
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname in ["sum", "prod"]:
            unit = 1 if opname == "prod" else 0  # result for empty sum/prod
            expected = Series(unit, index=r0.index, dtype=r0.dtype)
            tm.assert_series_equal(r0, expected)
            expected = Series(unit, index=r1.index, dtype=r1.dtype)
            tm.assert_series_equal(r1, expected)


@pytest.fixture
def bool_frame_with_na():
    # [FUNCTION DOC]
    # HELPER: fixture. A 30x4 object-dtype frame with columns A to D, string index
    # foo_0 to foo_29, whose first 15 rows are true and rest false, with rows five
    # to nine entirely missing and rows fifteen to nineteen missing in the last two
    # columns.
    """
    Fixture for DataFrame of booleans with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.concatenate(
            [np.ones((15, 4), dtype=bool), np.zeros((15, 4), dtype=bool)], axis=0
        ),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
        dtype=object,
    )
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


@pytest.fixture
def float_frame_with_na():
    # [FUNCTION DOC]
    # HELPER: fixture. A 30x4 random normal float frame with columns A to D and
    # string index foo_0 to foo_29, with rows five to nine entirely missing and rows
    # fifteen to nineteen missing in the last two columns.
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
    )
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


class TestDataFrameAnalytics:
    # [CLASS DOC]
    # Test group for frame reductions and related analytics. Stateless.
    # ---------------------------------------------------------------------
    # Reductions
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "nunique",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_string_frame(self, float_string_frame, axis, opname):
        # [FUNCTION DOC]
        # Parametrized over axis and over reduction names (some need scipy). On the
        # mixed float/string fixture: sum, min, max on axis zero, and count and nunique,
        # must work; other combinations must raise a type error whose message depends
        # on the operation (string-to-float conversion failures, sequence multiplication,
        # unsupported operand for addition, comparison failures, conversion failures for
        # median) or generic does-not-support text. Then, except for nunique, the same
        # call with the numeric-only flag must succeed.
        if (opname in ("sum", "min", "max") and axis == 0) or opname in (
            "count",
            "nunique",
        ):
            getattr(float_string_frame, opname)(axis=axis)
        else:
            if opname in ["var", "std", "sem", "skew", "kurt"]:
                msg = "could not convert string to float: 'bar'"
            elif opname == "product":
                if axis == 1:
                    msg = "can't multiply sequence by non-int of type 'float'"
                else:
                    msg = "can't multiply sequence by non-int of type 'str'"
            elif opname == "sum":
                msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname == "mean":
                if axis == 0:
                    # different message on different builds
                    msg = "|".join(
                        [
                            r"Could not convert \['.*'\] to numeric",
                            "Could not convert string '(bar){30}' to numeric",
                        ]
                    )
                else:
                    msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname in ["min", "max"]:
                msg = "'[><]=' not supported between instances of 'float' and 'str'"
            elif opname == "median":
                msg = re.compile(
                    r"Cannot convert \[.*\] to numeric|does not support|Cannot perform",
                    flags=re.S,
                )
            if not isinstance(msg, re.Pattern):
                msg = msg + "|does not support|Cannot perform reduction"
            with pytest.raises(TypeError, match=msg):
                getattr(float_string_frame, opname)(axis=axis)
        if opname != "nunique":
            getattr(float_string_frame, opname)(axis=axis, numeric_only=True)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_frame(self, float_frame, axis, opname):
        # [FUNCTION DOC]
        # Parametrized over axis and reduction names. Smoke test: each reduction runs on
        # the all-float fixture with the numeric-only flag explicitly false.
        getattr(float_frame, opname)(axis=axis, numeric_only=False)

    def test_stat_op_calc(self, float_frame_with_na, mixed_float_frame):
        # [FUNCTION DOC]
        # Checks sum (mixed float frame cast to 32-bit with loose tolerance GH#32571),
        # sum and product with NaN-aware references, mean, variance, standard deviation,
        # standard error, count and nunique against reference implementations through
        # the shared checker, with date checks for nunique, mean and count.
        def count(s):
            # [FUNCTION DOC]
            # HELPER: for the stat-op test. Reference count: number of non-missing entries.
            return notna(s).sum()

        def nunique(s):
            # [FUNCTION DOC]
            # HELPER: for the stat-op test. Reference unique count: the number of distinct
            # values after dropping missing entries.
            return len(algorithms.unique1d(s.dropna()))

        def var(x):
            # [FUNCTION DOC]
            # HELPER: for the stat-op test. Reference sample variance (delta degrees of
            # freedom one).
            return np.var(x, ddof=1)

        def std(x):
            # [FUNCTION DOC]
            # HELPER: for the stat-op test. Reference sample standard deviation (delta
            # degrees of freedom one).
            return np.std(x, ddof=1)

        def sem(x):
            # [FUNCTION DOC]
            # HELPER: for the stat-op test. Reference standard error of the mean: sample
            # standard deviation over the square root of the length.
            return np.std(x, ddof=1) / np.sqrt(len(x))

        assert_stat_op_calc(
            "nunique",
            nunique,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

        # GH#32571: rol needed for flaky CI builds
        # mixed types (with upcasting happening)
        assert_stat_op_calc(
            "sum",
            np.sum,
            mixed_float_frame.astype("float32"),
            check_dtype=False,
            rtol=1e-3,
        )

        assert_stat_op_calc(
            "sum", np.sum, float_frame_with_na, skipna_alternative=np.nansum
        )
        assert_stat_op_calc("mean", np.mean, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "product", np.prod, float_frame_with_na, skipna_alternative=np.nanprod
        )

        assert_stat_op_calc("var", var, float_frame_with_na)
        assert_stat_op_calc("std", std, float_frame_with_na)
        assert_stat_op_calc("sem", sem, float_frame_with_na)

        assert_stat_op_calc(
            "count",
            count,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

    def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na):
        # [FUNCTION DOC]
        # Skipped without scipy. Compares skew and kurtosis to scipy's unbiased
        # estimators through the shared checker; those estimators are replaced by NaN for
        # fewer than three (skew) or four (kurtosis) values.
        sp_stats = pytest.importorskip("scipy.stats")

        def skewness(x):
            # [FUNCTION DOC]
            # HELPER: for the skew/kurtosis test. Unbiased skewness of the values, NaN for
            # fewer than three values.
            if len(x) < 3:
                return np.nan
            return sp_stats.skew(x, bias=False)

        def kurt(x):
            # [FUNCTION DOC]
            # HELPER: for the skew/kurtosis test. Unbiased excess kurtosis of the values,
            # NaN for fewer than four values.
            if len(x) < 4:
                return np.nan
            return sp_stats.kurtosis(x, bias=False)

        assert_stat_op_calc("skew", skewness, float_frame_with_na)
        assert_stat_op_calc("kurt", kurt, float_frame_with_na)

    def test_median(self, float_frame_with_na, int_frame):
        # [FUNCTION DOC]
        # Checks median against a reference (NaN when any value is missing, else the
        # NumPy median) on the float frame with missing values and on the integer
        # frame (dtype not checked), both with date checks.
        def wrapper(x):
            # [FUNCTION DOC]
            # HELPER: for the median test. Reference median that returns NaN if any missing
            # value is present.
            if isna(x).any():
                return np.nan
            return np.median(x)

        assert_stat_op_calc("median", wrapper, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "median", wrapper, int_frame, check_dtype=False, check_dates=True
        )

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "prod", "var", "std", "skew", "min", "max"]
    )
    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(
                {
                    "a": [
                        -0.00049987540199591344,
                        -0.0016467257772919831,
                        0.00067695870775883013,
                    ],
                    "b": [-0, -0, 0.0],
                    "c": [
                        0.00031111847529610595,
                        0.0014902627951905339,
                        -0.00094099200035979691,
                    ],
                },
                index=["foo", "bar", "baz"],
                dtype="O",
            ),
            DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Mismatched null-like values:FutureWarning")
    def test_stat_operators_attempt_obj_array(self, method, df, axis):
        # [FUNCTION DOC]
        # GH#676: parametrized over method names, two object-dtype frames (one containing
        # negative zeros and small floats, one with NaN columns), and axis. Reducing an
        # object-dtype frame equals reducing the float-cast frame and casting the result
        # back to object (for min and max over rows, missing results become None).
        # Filters a warning about mismatched null-like values.
        # GH#676
        assert df.values.dtype == np.object_
        result = getattr(df, method)(axis=axis)
        expected = getattr(df.astype("f8"), method)(axis=axis).astype(object)
        if axis in [1, "columns"] and method in ["min", "max"]:
            expected[expected.isna()] = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", ["mean", "std", "var", "skew", "kurt", "sem"])
    def test_mixed_ops(self, op):
        # [FUNCTION DOC]
        # GH#16116: parametrized over mean, std, var, skew, kurt, sem. A frame with
        # integer, float and string columns raises a type error with one of several
        # messages, both with the bottleneck option on and off.
        # GH#16116
        df = DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.0, 2.0, 3.0, 4.0],
                "str": ["a", "b", "c", "d"],
            }
        )
        msg = "|".join(
            [
                "Could not convert",
                "could not convert",
                "can't multiply sequence by non-int",
                "does not support",
                "Cannot perform",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)()

        with pd.option_context("use_bottleneck", False):
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)()

    def test_reduce_mixed_frame(self):
        # [FUNCTION DOC]
        # GH 6806: summing a frame with boolean, integer and string columns gives an
        # object array of the boolean count, integer sum, and concatenated string, and
        # equals the sum along axis one of the transposed frame.
        # KNOWN ISSUE: the reindex call result is discarded, so that statement has no effect on the frame under test.
        # GH 6806
        df = DataFrame(
            {
                "bool_data": [True, True, False, False, False],
                "int_data": [10, 20, 30, 40, 50],
                "string_data": ["a", "b", "c", "d", "e"],
            }
        )
        df.reindex(columns=["bool_data", "int_data", "string_data"])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(
            test.values, np.array([2, 150, "abcde"], dtype=object)
        )
        alt = df.T.sum(axis=1)
        tm.assert_series_equal(test, alt)

    def test_nunique(self):
        # [FUNCTION DOC]
        # Number of distinct values per column and per row, with and without counting
        # missing values as a value, for a small frame with a missing entry.
        df = DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({"A": 1, "B": 3, "C": 2}))
        tm.assert_series_equal(
            df.nunique(dropna=False), Series({"A": 1, "B": 3, "C": 3})
        )
        tm.assert_series_equal(df.nunique(axis=1), Series([1, 2, 2]))
        tm.assert_series_equal(df.nunique(axis=1, dropna=False), Series([1, 3, 2]))

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_mixed_datetime_numeric(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (none, UTC). GH 24752: the mean of a frame with an
        # integer column and a datetime column returns a Series with the float mean and
        # the datetime.
        # https://github.com/pandas-dev/pandas/issues/24752
        df = DataFrame({"A": [1, 1], "B": [Timestamp("2000", tz=tz)] * 2})
        result = df.mean()
        expected = Series([1.0, Timestamp("2000", tz=tz)], index=["A", "B"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_includes_datetimes(self, tz):
        # [FUNCTION DOC]
        # Parametrized over timezone (none, UTC). GH 24752: as of 2.0 the mean of a
        # single datetime column is the timestamp (datetime columns are not dropped).
        # https://github.com/pandas-dev/pandas/issues/24752
        # Behavior in 0.24.0rc1 was buggy.
        # As of 2.0 with numeric_only=None we do *not* drop datetime columns
        df = DataFrame({"A": [Timestamp("2000", tz=tz)] * 2})
        result = df.mean()

        expected = Series([Timestamp("2000", tz=tz)], index=["A"])
        tm.assert_series_equal(result, expected)

    def test_mean_mixed_string_decimal(self):
        # [FUNCTION DOC]
        # GH 11670: a mean over a frame with a string column among integer and
        # decimal columns raises a type error; restricting to integer and decimal columns
        # gives object-dtype means 2.7 and 681.6.
        # GH 11670
        # possible bug when calculating mean of DataFrame?

        d = [
            {"A": 2, "B": None, "C": Decimal("628.00")},
            {"A": 1, "B": None, "C": Decimal("383.00")},
            {"A": 3, "B": None, "C": Decimal("651.00")},
            {"A": 2, "B": None, "C": Decimal("575.00")},
            {"A": 4, "B": None, "C": Decimal("1114.00")},
            {"A": 1, "B": "TEST", "C": Decimal("241.00")},
            {"A": 2, "B": None, "C": Decimal("572.00")},
            {"A": 4, "B": None, "C": Decimal("609.00")},
            {"A": 3, "B": None, "C": Decimal("820.00")},
            {"A": 5, "B": None, "C": Decimal("1223.00")},
        ]

        df = DataFrame(d)

        with pytest.raises(
            TypeError, match="unsupported operand type|does not support|Cannot perform"
        ):
            df.mean()
        result = df[["A", "C"]].mean()
        expected = Series([2.7, 681.6], index=["A", "C"], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_var_std(self, datetime_frame):
        # [FUNCTION DOC]
        # Sample standard deviation and variance with delta degrees of freedom four equal
        # the column-wise application on the datetime-indexed fixture; the low-level
        # variance of a 1000x1000 matrix of identical rows is never negative, with and
        # without bottleneck.
        result = datetime_frame.std(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4))
        tm.assert_almost_equal(result, expected)

        result = datetime_frame.var(ddof=4)
        expected = datetime_frame.apply(lambda x: x.var(ddof=4))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context("use_bottleneck", False):
            result = nanops.nanvar(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize("meth", ["sem", "var", "std"])
    def test_numeric_only_flag(self, meth):
        # [FUNCTION DOC]
        # GH 9201: parametrized over standard error, variance and standard deviation. On
        # two frames where one object column holds a numeric string and a non-numeric
        # string, row-wise reduction with the numeric-only flag equals reducing the two
        # numeric columns; without the flag, they raise type errors (unsupported operand for
        # float and string, and conversion failure).
        # GH 9201
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            columns=["foo", "bar", "baz"],
        )
        # Cast to object to avoid implicit cast when setting entry to "100" below
        df1 = df1.astype({"foo": object})
        # set one entry to a number in str format
        df1.loc[0, "foo"] = "100"

        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            columns=["foo", "bar", "baz"],
        )
        # Cast to object to avoid implicit cast when setting entry to "a" below
        df2 = df2.astype({"foo": object})
        # set one entry to a non-number str
        df2.loc[0, "foo"] = "a"

        result = getattr(df1, meth)(axis=1, numeric_only=True)
        expected = getattr(df1[["bar", "baz"]], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        result = getattr(df2, meth)(axis=1, numeric_only=True)
        expected = getattr(df2[["bar", "baz"]], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        # df1 has all numbers, df2 has a letter inside
        msg = r"unsupported operand type\(s\) for -: 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            getattr(df1, meth)(axis=1, numeric_only=False)
        msg = "could not convert string to float: 'a'"
        with pytest.raises(TypeError, match=msg):
            getattr(df2, meth)(axis=1, numeric_only=False)

    def test_sem(self, datetime_frame):
        # [FUNCTION DOC]
        # Standard error with delta degrees of freedom four equals the column-wise
        # standard deviation over root length; the low-level standard error of identical
        # rows is never negative, with and without bottleneck.
        result = datetime_frame.sem(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4) / np.sqrt(len(x)))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nansem(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context("use_bottleneck", False):
            result = nanops.nansem(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize(
        "dropna, expected",
        [
            (
                True,
                {
                    "A": [12],
                    "B": [10.0],
                    "C": [1.0],
                    "D": ["a"],
                    "E": Categorical(["a"], categories=["a"]),
                    "F": DatetimeIndex(["2000-01-02"], dtype="M8[ns]"),
                    "G": to_timedelta(["1 days"]),
                },
            ),
            (
                False,
                {
                    "A": [12],
                    "B": [10.0],
                    "C": [np.nan],
                    "D": Series([np.nan], dtype="str"),
                    "E": Categorical([np.nan], categories=["a"]),
                    "F": DatetimeIndex([pd.NaT], dtype="M8[ns]"),
                    "G": to_timedelta([pd.NaT]).as_unit("us"),
                },
            ),
            (
                True,
                {
                    "H": [8, 9, np.nan, np.nan],
                    "I": [8, 9, np.nan, np.nan],
                    "J": [1, np.nan, np.nan, np.nan],
                    "K": Categorical(["a", np.nan, np.nan, np.nan], categories=["a"]),
                    "L": DatetimeIndex(
                        ["2000-01-02", "NaT", "NaT", "NaT"], dtype="M8[ns]"
                    ),
                    "M": to_timedelta(["1 days", "nan", "nan", "nan"]),
                    "N": [0, 1, 2, 3],
                },
            ),
            (
                False,
                {
                    "H": [8, 9, np.nan, np.nan],
                    "I": [8, 9, np.nan, np.nan],
                    "J": [1, np.nan, np.nan, np.nan],
                    "K": Categorical([np.nan, "a", np.nan, np.nan], categories=["a"]),
                    "L": DatetimeIndex(
                        ["NaT", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"
                    ),
                    "M": to_timedelta(["nan", "1 days", "nan", "nan"]),
                    "N": [0, 1, 2, 3],
                },
            ),
        ],
    )
    def test_mode_dropna(self, dropna, expected):
        # [FUNCTION DOC]
        # Parametrized over the drop-missing flag and the expected mode frame. Builds a
        # frame of integer, float, string, categorical, datetime and timedelta columns
        # (with ties and missing values), selects the columns listed in the expectation,
        # and checks the mode result frame: with drop-missing the most frequent non-missing
        # values (padded with missing where columns have fewer modes), without it missing
        # can itself be the mode.
        df = DataFrame(
            {
                "A": [12, 12, 19, 11],
                "B": [10, 10, np.nan, 3],
                "C": [1, np.nan, np.nan, np.nan],
                "D": Series([np.nan, np.nan, "a", np.nan], dtype="str"),
                "E": Categorical([np.nan, np.nan, "a", np.nan]),
                "F": DatetimeIndex(["NaT", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"),
                "G": to_timedelta(["1 days", "nan", "nan", "nan"]),
                "H": [8, 8, 9, 9],
                "I": [9, 9, 8, 8],
                "J": [1, 1, np.nan, np.nan],
                "K": Categorical(["a", np.nan, "a", np.nan]),
                "L": DatetimeIndex(
                    ["2000-01-02", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"
                ),
                "M": to_timedelta(["1 days", "nan", "1 days", "nan"]),
                "N": np.arange(4, dtype="int64"),
            }
        )

        result = df[sorted(expected.keys())].mode(dropna=dropna)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_mode_sort_with_na(self, using_infer_string):
        # [FUNCTION DOC]
        # Mode without dropping missing of a string-ish column with two missing values
        # and two equal strings sorts the modes so the string comes first then missing.
        # KNOWN ISSUE: the inferred-string fixture parameter is accepted but never used.
        df = DataFrame({"A": [np.nan, np.nan, "a", "a"]})
        expected = DataFrame({"A": ["a", np.nan]})
        result = df.mode(dropna=False)
        tm.assert_frame_equal(result, expected)

    def test_mode_empty_df(self):
        # [FUNCTION DOC]
        # Mode of a frame with columns but no rows returns an equal empty frame.
        df = DataFrame([], columns=["a", "b"])
        expected = df.copy()
        result = df.mode()
        tm.assert_frame_equal(result, expected)

    def test_operators_timedelta64(self):
        # [FUNCTION DOC]
        # Timedelta reductions: differences of datetime columns; min and max by column and
        # by row; absolute value (function and builtin); a mixed frame with string,
        # integer, float and timestamp columns where min gives an object Series of those
        # minima; numeric-only row-wise min excludes non-numeric columns; selecting only
        # the timedelta columns gives timedelta-typed results; GH 3106: timedelta columns
        # created from datetime differences keep timedelta dtype, also after in-place
        # consolidation.
        df = DataFrame(
            {
                "A": date_range("2012-1-1", periods=3, freq="D", unit="ns"),
                "B": date_range("2012-1-2", periods=3, freq="D", unit="ns"),
                "C": Timestamp("20120101") - timedelta(minutes=5, seconds=5),
            }
        )

        diffs = DataFrame({"A": df["A"] - df["C"], "B": df["A"] - df["B"]})

        # min
        result = diffs.min()
        assert result.iloc[0] == diffs.loc[0, "A"]
        assert result.iloc[1] == diffs.loc[0, "B"]

        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, "B"]).all()

        # max
        result = diffs.max()
        assert result.iloc[0] == diffs.loc[2, "A"]
        assert result.iloc[1] == diffs.loc[2, "B"]

        result = diffs.max(axis=1)
        assert (result == diffs["A"]).all()

        # abs
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame({"A": df["A"] - df["C"], "B": df["B"] - df["A"]})
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # mixed frame
        mixed = diffs.copy()
        mixed["C"] = "foo"
        mixed["D"] = 1
        mixed["E"] = 1.0
        mixed["F"] = Timestamp("20130101")

        # results in an object array
        result = mixed.min()
        expected = Series(
            [
                pd.Timedelta(timedelta(seconds=5 * 60 + 5)),
                pd.Timedelta(timedelta(days=-1)),
                "foo",
                1,
                1.0,
                Timestamp("20130101"),
            ],
            index=mixed.columns,
        )
        tm.assert_series_equal(result, expected)

        # excludes non-numeric
        result = mixed.min(axis=1, numeric_only=True)
        expected = Series([1, 1, 1.0])
        tm.assert_series_equal(result, expected)

        # works when only those columns are selected
        result = mixed[["A", "B"]].min(axis=1)
        expected = Series([timedelta(days=-1)] * 3, dtype="m8[ns]")
        tm.assert_series_equal(result, expected)

        result = mixed[["A", "B"]].min()
        expected = Series(
            [timedelta(seconds=5 * 60 + 5), timedelta(days=-1)],
            index=["A", "B"],
            dtype="m8[ns]",
        )
        tm.assert_series_equal(result, expected)

        # GH 3106
        df = DataFrame(
            {
                "time": date_range("20130102", periods=5, unit="ns"),
                "time2": date_range("20130105", periods=5, unit="ns"),
            }
        )
        df["off1"] = df["time2"] - df["time"]
        assert df["off1"].dtype == "timedelta64[ns]"

        df["off2"] = df["time"] - df["time2"]
        df._consolidate_inplace()
        assert df["off1"].dtype == "timedelta64[ns]"
        assert df["off2"].dtype == "timedelta64[ns]"

    def test_std_timedelta64_skipna_false(self):
        # [FUNCTION DOC]
        # GH#37392: standard deviation of timedelta columns with skipna false gives a
        # timedelta (microsecond unit) Series where the column with a missing value is NaT,
        # and along rows gives zero timedeltas except NaT on the row with the missing
        # value.
        # GH#37392
        tdi = pd.timedelta_range("1 Day", periods=10)
        df = DataFrame({"A": tdi, "B": tdi}, copy=True)
        df.iloc[-2, -1] = pd.NaT

        result = df.std(skipna=False)
        expected = Series(
            [df["A"].std(), pd.NaT], index=["A", "B"], dtype="timedelta64[us]"
        )
        tm.assert_series_equal(result, expected)

        result = df.std(axis=1, skipna=False)
        expected = Series(
            [pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)], dtype="m8[us]"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values", [["2022-01-01", "2022-01-02", pd.NaT, "2022-01-03"], 4 * [pd.NaT]]
    )
    def test_std_datetime64_with_nat(self, values, skipna, request, unit):
        # [FUNCTION DOC]
        # GH#51335: parametrized over datetime values (some NaT, or all NaT), the skipna
        # fixture and the time-unit fixture. Standard deviation of a datetime column is NaT
        # if not skipping or all values are NaT, else one day as a timedelta in the same
        # unit.
        # KNOWN ISSUE: the request fixture parameter is accepted but never used.
        # GH#51335
        dti = to_datetime(values).as_unit(unit)
        df = DataFrame({"a": dti})
        result = df.std(skipna=skipna)
        if not skipna or all(value is pd.NaT for value in values):
            expected = Series({"a": pd.NaT}, dtype=f"timedelta64[{unit}]")
        else:
            expected = Series({"a": "1 days"}, dtype=f"timedelta64[{unit}]")
        tm.assert_series_equal(result, expected)

    def test_sum_corner(self):
        # [FUNCTION DOC]
        # Summing an empty frame along either axis returns an empty Series.
        empty_frame = DataFrame()

        axis0 = empty_frame.sum(axis=0)
        axis1 = empty_frame.sum(axis=1)
        assert isinstance(axis0, Series)
        assert isinstance(axis1, Series)
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize(
        "index",
        [
            RangeIndex(0),
            DatetimeIndex([]),
            Index([], dtype=np.int64),
            Index([], dtype=np.float64),
            DatetimeIndex([], freq="ME"),
            PeriodIndex([], freq="D"),
        ],
    )
    def test_axis_1_empty(self, all_reductions, index):
        # [FUNCTION DOC]
        # Parametrized over many empty index types and every reduction fixture. A reduction
        # along axis one on a frame with one column and no rows returns an empty Series
        # with the empty index; dtype is boolean for any/all, 64-bit integer for count,
        # and object otherwise.
        df = DataFrame(columns=["a"], index=index)
        result = getattr(df, all_reductions)(axis=1)
        if all_reductions in ("any", "all"):
            expected_dtype = "bool"
        elif all_reductions == "count":
            expected_dtype = "int64"
        else:
            expected_dtype = "object"
        expected = Series([], index=index, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("min_count", [0, 1])
    def test_axis_1_sum_na(self, string_dtype_no_object, skipna, min_count):
        # [FUNCTION DOC]
        # Parametrized over string-without-object dtype fixture, skipna fixture and
        # min_count (zero, one). Issue 60229: summing across columns of a single
        # missing string value gives an empty string only when skipping missing with
        # min_count zero; otherwise missing; result keeps the string dtype.
        # https://github.com/pandas-dev/pandas/issues/60229
        dtype = string_dtype_no_object
        df = DataFrame({"a": [pd.NA]}, dtype=dtype)
        result = df.sum(axis=1, skipna=skipna, min_count=min_count)
        value = "" if skipna and min_count == 0 else pd.NA
        expected = Series([value], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method, unit", [("sum", 0), ("prod", 1)])
    @pytest.mark.parametrize("numeric_only", [None, True, False])
    def test_sum_prod_nanops(self, method, unit, numeric_only):
        # [FUNCTION DOC]
        # Parametrized over sum/product with their identity (zero/one) and numeric-only
        # none/true/false. A frame with identity values and missing: default result is
        # the identity for every column, as floats; min_count one gives missing for the
        # all-missing column; min_count zero gives the identity; slicing off a row with
        # min_count one gives missing when no values remain; larger min_count values.
        # KNOWN ISSUE: in the last two checks the expected Series is built from the result itself, so those assertions compare the result with itself and cannot fail.
        idx = ["a", "b", "c"]
        df = DataFrame({"a": [unit, unit], "b": [unit, np.nan], "c": [np.nan, np.nan]})
        # The default
        result = getattr(df, method)(numeric_only=numeric_only)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count > 1
        df = DataFrame({"A": [unit] * 10, "B": [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

        result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_timedelta(self):
        # [FUNCTION DOC]
        # Sum of timedelta columns (product is undefined for timedelta): defaults to a
        # zero timedelta for all-missing columns, min_count zero equals the default, and
        # min_count one gives NaT (as missing) for the all-missing column.
        # prod isn't defined on timedeltas
        idx = ["a", "b", "c"]
        df = DataFrame({"a": [0, 0], "b": [0, np.nan], "c": [np.nan, np.nan]})

        df2 = df.apply(to_timedelta)

        # 0 by default
        result = df2.sum()
        expected = Series([0, 0, 0], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = df2.sum(min_count=1)
        expected = Series([0, 0, np.nan], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_min_count(self):
        # [FUNCTION DOC]
        # Issue 39738: sum with a minimum count greater than the number of rows gives
        # missing for each column.
        # https://github.com/pandas-dev/pandas/issues/39738
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = df.sum(min_count=10)
        expected = Series([np.nan, np.nan], index=["x", "y"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("float_type", ["float16", "float32", "float64"])
    @pytest.mark.parametrize(
        "kwargs, expected_result",
        [
            ({"axis": 1, "min_count": 2}, [3.2, 5.3, np.nan]),
            ({"axis": 1, "min_count": 3}, [np.nan, np.nan, np.nan]),
            ({"axis": 1, "skipna": False}, [3.2, 5.3, np.nan]),
        ],
    )
    def test_sum_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        # [FUNCTION DOC]
        # GH#46947: parametrized over float16/32/64 and keyword sets (row-wise min_count
        # two or three, skipna false). Row-wise sum of two columns with one missing gives
        # the expected sums or missing, with the result cast to the float type.
        # GH#46947
        df = DataFrame({"a": [1.0, 2.3, 4.4], "b": [2.2, 3, np.nan]}, dtype=float_type)
        result = df.sum(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("float_type", ["float16", "float32", "float64"])
    @pytest.mark.parametrize(
        "kwargs, expected_result",
        [
            ({"axis": 1, "min_count": 2}, [2.0, 4.0, np.nan]),
            ({"axis": 1, "min_count": 3}, [np.nan, np.nan, np.nan]),
            ({"axis": 1, "skipna": False}, [2.0, 4.0, np.nan]),
        ],
    )
    def test_prod_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        # [FUNCTION DOC]
        # GH#46947: same as the sum case but for product.
        # GH#46947
        df = DataFrame(
            {"a": [1.0, 2.0, 4.4], "b": [2.0, 2.0, np.nan]}, dtype=float_type
        )
        result = df.prod(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame):
        # [FUNCTION DOC]
        # Smoke test: summing a frame of timedelta (object) values computed by multiplying
        # an integer frame by a day timedelta does not raise; nothing is asserted.
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index, columns=float_frame.columns)
        deltas = frame * timedelta(1)
        deltas.sum()

    def test_sum_bool(self, float_frame):
        # [FUNCTION DOC]
        # Smoke test, from a bug report: summing a boolean frame (the missing-value mask
        # of the float fixture) along both axes does not raise.
        # ensure this works, bug report
        bools = np.isnan(float_frame)
        bools.sum(axis=1)
        bools.sum(axis=0)

    def test_sum_mixed_datetime(self):
        # [FUNCTION DOC]
        # GH#30886: summing a frame with a datetime column and an integer column raises a
        # type error saying the datetime type does not support sum.
        # GH#30886
        df = DataFrame({"A": date_range("2000", periods=4), "B": [1, 2, 3, 4]}).reindex(
            [2, 3, 4]
        )
        with pytest.raises(TypeError, match="does not support operation 'sum'"):
            df.sum()

    def test_mean_corner(self, float_frame, float_string_frame):
        # [FUNCTION DOC]
        # Mean on the float/string fixture along axis zero raises a type error about
        # conversion; along axis one raises an unsupported-operand type error; the
        # mean of a boolean column added to the float fixture equals the mean of its
        # values.
        # unit test when have object data
        msg = "Could not convert|does not support|Cannot perform"
        with pytest.raises(TypeError, match=msg):
            float_string_frame.mean(axis=0)

        # xs sum mixed type, just want to know it works...
        with pytest.raises(TypeError, match="unsupported operand type"):
            float_string_frame.mean(axis=1)

        # take mean of boolean column
        float_frame["bool"] = float_frame["A"] > 0
        means = float_frame.mean(axis=0)
        assert means["bool"] == float_frame["bool"].values.mean()

    def test_mean_datetimelike(self):
        # [FUNCTION DOC]
        # GH#24757: mean with numeric-only true on a frame with integer, datetime,
        # timedelta and period columns returns only the integer column's mean; the
        # default mean raises a type error saying mean is not implemented for the period
        # array. As of 2.0 datetime-like columns are not excluded by default.
        # GH#24757 check that datetimelike are excluded by default, handled
        #  correctly with numeric_only=True
        #  As of 2.0, datetimelike are *not* excluded with numeric_only=None

        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
                "D": pd.period_range("2016", periods=3, freq="Y"),
            }
        )
        result = df.mean(numeric_only=True)
        expected = Series({"A": 1.0})
        tm.assert_series_equal(result, expected)

        with pytest.raises(TypeError, match="mean is not implemented for PeriodArray"):
            df.mean()

    def test_mean_datetimelike_numeric_only_false(self):
        # [FUNCTION DOC]
        # Mean with numeric-only false on integer, datetime and timedelta columns gives
        # the integer mean and the middle values of the datetime and timedelta; adding a
        # period column makes it raise a type error saying mean is not implemented for
        # periods.
        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
            }
        )

        # datetime(tz) and timedelta work
        result = df.mean(numeric_only=False)
        expected = Series({"A": 1, "B": df.loc[1, "B"], "C": df.loc[1, "C"]})
        tm.assert_series_equal(result, expected)

        # mean of period is not allowed
        df["D"] = pd.period_range("2016", periods=3, freq="Y")

        with pytest.raises(TypeError, match="mean is not implemented for Period"):
            df.mean(numeric_only=False)

    def test_mean_extensionarray_numeric_only_true(self):
        # [FUNCTION DOC]
        # Issue 33256: mean with numeric-only true on a nullable-integer frame equals the
        # mean of the same data as NumPy integers, cast to the nullable float dtype.
        # https://github.com/pandas-dev/pandas/issues/33256
        arr = np.random.default_rng(2).integers(1000, size=(10, 5))
        df = DataFrame(arr, dtype="Int64")
        result = df.mean(numeric_only=True)
        expected = DataFrame(arr).mean().astype("Float64")
        tm.assert_series_equal(result, expected)

    def test_stats_mixed_type(self, float_string_frame):
        # [FUNCTION DOC]
        # Row-wise standard deviation, variance and skew on the float/string fixture raise
        # a type error about conversion, and row-wise mean raises an unsupported-operand
        # type error.
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.std(axis=1)
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.var(axis=1)
        with pytest.raises(TypeError, match="unsupported operand type"):
            float_string_frame.mean(axis=1)
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.skew(axis=1)

    def test_sum_bools(self):
        # [FUNCTION DOC]
        # Summing the missing-value mask of a one-row frame of ten missing entries along
        # axis one gives ten.
        df = DataFrame(index=range(1), columns=range(10))
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    @pytest.mark.parametrize(
        "input_data, expected_data",
        [
            ({"a": ["483", "3"], "b": ["94", "759"]}, ["48394", "3759"]),
            (
                {"a": ["483.948", "3.0"], "b": ["94.2", "759.93"]},
                ["483.94894.2", "3.0759.93"],
            ),
            ({"a": ["483", "3.0"], "b": ["94.2", "79"]}, ["48394.2", "3.079"]),
        ],
    )
    def test_sum_string_dtype_coercion(self, input_data, expected_data):
        # [FUNCTION DOC]
        # GH#22642: parametrized over numeric-looking string pairs. Row-wise sum of string
        # columns concatenates the strings, not converting to numbers.
        # GH#22642
        # Check that summing numeric strings results in concatenation
        # and not conversion to dtype int64 or float64
        df = DataFrame(input_data)
        expected = Series(expected_data)
        result = df.sum(axis=1)
        tm.assert_series_equal(result, expected)

    # ----------------------------------------------------------------------
    # Index of max / min

    @pytest.mark.parametrize("axis", [0, 1])
    def test_idxmin(self, float_frame, int_frame, skipna, axis):
        # [FUNCTION DOC]
        # Parametrized over axis and the skipna fixture. Index of the minimum for the
        # float fixture with some missing rows and the integer fixture: when not skipping
        # missing or reducing across columns on the float frame, it raises a value error
        # (an NA value was encountered, or all values NA when skipping); otherwise it equals
        # applying the Series version, cast to the index dtype. The raising check is
        # repeated twice.
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            if (not skipna or axis == 1) and df is not int_frame:
                if skipna:
                    msg = "Encountered all NA values"
                else:
                    msg = "Encountered an NA value"
                with pytest.raises(ValueError, match=msg):
                    df.idxmin(axis=axis, skipna=skipna)
                with pytest.raises(ValueError, match=msg):
                    df.idxmin(axis=axis, skipna=skipna)
            else:
                result = df.idxmin(axis=axis, skipna=skipna)
                expected = df.apply(Series.idxmin, axis=axis, skipna=skipna)
                expected = expected.astype(df.index.dtype)
                tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_idxmin_empty(self, index, skipna, axis):
        # [FUNCTION DOC]
        # GH53265: parametrized over axis, skipna and an index fixture (filtering a
        # period-business-day deprecation warning). Index of the minimum on an empty
        # frame (empty rows or empty columns) gives an empty Series with the index dtype.
        # GH53265
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)

        result = frame.idxmin(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_idxmin_numeric_only(self, numeric_only):
        # [FUNCTION DOC]
        # Parametrized over numeric-only. Index of the minimum for a frame with two
        # integer columns and a string column: with numeric-only only the integer columns
        # are returned; otherwise all three (positions two, one, zero).
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1], "c": list("xyx")})
        result = df.idxmin(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([2, 1], index=["a", "b"])
        else:
            expected = Series([2, 1, 0], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

    def test_idxmin_axis_2(self, float_frame):
        # [FUNCTION DOC]
        # Index of minimum with axis two raises a value error saying there is no such axis
        # for the frame type.
        frame = float_frame
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.idxmin(axis=2)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_idxmax(self, float_frame, int_frame, skipna, axis):
        # [FUNCTION DOC]
        # Same as the minimum test but for the index of the maximum. In the float frame
        # raising cases the check ends the test early by returning, so the integer frame
        # is not examined for those parameter combinations.
        # KNOWN ISSUE: the early return ends the whole test (not just the loop iteration), skipping the integer frame whenever the float frame raises; the condition also differs from the minimum test.
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            if (skipna is False or axis == 1) and df is frame:
                if skipna:
                    msg = "Encountered all NA values"
                else:
                    msg = "Encountered an NA value"
                with pytest.raises(ValueError, match=msg):
                    df.idxmax(axis=axis, skipna=skipna)
                return

            result = df.idxmax(axis=axis, skipna=skipna)
            expected = df.apply(Series.idxmax, axis=axis, skipna=skipna)
            expected = expected.astype(df.index.dtype)
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_idxmax_empty(self, index, skipna, axis):
        # [FUNCTION DOC]
        # GH53265: as the empty minimum test but for the index of the maximum.
        # GH53265
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)

        result = frame.idxmax(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_idxmax_numeric_only(self, numeric_only):
        # [FUNCTION DOC]
        # Parametrized over numeric-only. Index of the maximum of the small frame with a
        # string column: numeric-only gives positions one and zero for the integer
        # columns; otherwise adds the string column's position (one).
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1], "c": list("xyx")})
        result = df.idxmax(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([1, 0], index=["a", "b"])
        else:
            expected = Series([1, 0, 1], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

    def test_idxmax_arrow_types(self):
        # [FUNCTION DOC]
        # GH#55368: skipped without pyarrow. Index of maximum and minimum for Arrow
        # integer columns give the expected positions; for an Arrow string column with
        # numeric-only false the maximum position is one and the minimum is two.
        # GH#55368
        pytest.importorskip("pyarrow")

        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1]}, dtype="int64[pyarrow]")
        result = df.idxmax()
        expected = Series([1, 0], index=["a", "b"])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([2, 1], index=["a", "b"])
        tm.assert_series_equal(result, expected)

        df = DataFrame({"a": ["b", "c", "a"]}, dtype="string[pyarrow]")
        result = df.idxmax(numeric_only=False)
        expected = Series([1], index=["a"])
        tm.assert_series_equal(result, expected)

        result = df.idxmin(numeric_only=False)
        expected = Series([2], index=["a"])
        tm.assert_series_equal(result, expected)

    def test_idxmax_axis_2(self, float_frame):
        # [FUNCTION DOC]
        # Index of maximum with axis two raises a value error saying there is no such axis
        # for the frame type.
        frame = float_frame
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.idxmax(axis=2)

    def test_idxmax_mixed_dtype(self):
        # [FUNCTION DOC]
        # Index of maximum and minimum for a frame with integer and datetime columns do
        # not cast to object; also with NaT in one datetime column, and with a second
        # datetime column forming a multi-column datetime block after consolidation.
        # don't cast to object, which would raise in nanops
        dti = date_range("2016-01-01", periods=3)
        df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti})

        result = df.idxmax()
        expected = Series([1, 0, 2], index=range(1, 4))
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 0], index=range(1, 4))
        tm.assert_series_equal(result, expected)

        # with NaTs
        df.loc[0, 3] = pd.NaT
        result = df.idxmax()
        expected = Series([1, 0, 2], index=range(1, 4))
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 1], index=range(1, 4))
        tm.assert_series_equal(result, expected)

        # with multi-column dt64 block
        df[4] = dti[::-1]
        df._consolidate_inplace()

        result = df.idxmax()
        expected = Series([1, 0, 2, 0], index=range(1, 5))
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 1, 2], index=range(1, 5))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected_value",
        [("idxmax", [0, 4]), ("idxmin", [0, 5])],
    )
    def test_idxmax_idxmin_convert_dtypes(self, op, expected_value):
        # [FUNCTION DOC]
        # GH 40346: parametrized over index-of-maximum and index-of-minimum with expected
        # values. Grouped nullable-integer data returns, per group, the index label of
        # the extreme, as a frame with a nullable-integer group index.
        # GH 40346
        df = DataFrame(
            {
                "ID": [100, 100, 100, 200, 200, 200],
                "value": [0, 0, 0, 1, 2, 0],
            },
            dtype="Int64",
        )
        df = df.groupby("ID")

        result = getattr(df, op)()
        expected = DataFrame(
            {"value": expected_value},
            index=Index([100, 200], name="ID", dtype="Int64"),
        )
        tm.assert_frame_equal(result, expected)

    def test_idxmax_dt64_multicolumn_axis1(self):
        # [FUNCTION DOC]
        # Datetime columns with a missing value in a multi-column datetime block (after
        # in-place consolidation): index of the maximum and minimum along rows give the
        # column labels of the extreme values, ignoring missing, as expected Series.
        dti = date_range("2016-01-01", periods=3)
        df = DataFrame({3: dti, 4: dti[::-1]}, copy=True)
        df.iloc[0, 0] = pd.NaT

        df._consolidate_inplace()

        result = df.idxmax(axis=1)
        expected = Series([4, 3, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin(axis=1)
        expected = Series([4, 3, 4])
        tm.assert_series_equal(result, expected)

    # ----------------------------------------------------------------------
    # Logical reductions

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("bool_only", [False, True])
    def test_any_all_mixed_float(
        self, all_boolean_reductions, axis, bool_only, float_string_frame
    ):
        # [FUNCTION DOC]
        # Parametrized over any/all fixture, axis and bool-only. Smoke test: any and all
        # run on the mixed float/string fixture with an added random boolean column;
        # nothing is asserted.
        # make sure op works on mixed-type frame
        mixed = float_string_frame
        mixed["_bool_"] = np.random.default_rng(2).standard_normal(len(mixed)) > 0.5

        getattr(mixed, all_boolean_reductions)(axis=axis, bool_only=bool_only)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_any_all_bool_with_na(
        self, all_boolean_reductions, axis, bool_frame_with_na
    ):
        # [FUNCTION DOC]
        # Parametrized over axis. Smoke test: any/all with bool-only false run on the
        # object-dtype boolean frame with missing values.
        getattr(bool_frame_with_na, all_boolean_reductions)(axis=axis, bool_only=False)

    def test_any_all_bool_frame(self, all_boolean_reductions, bool_frame_with_na):
        # [FUNCTION DOC]
        # GH#12863: after filling missing with true, any/all along both axes, with and
        # without skipping, equal the NumPy function applied to the values (dropping
        # missing in the skipping case; row-wise dtype not checked); an invalid axis
        # raises a value error; on an all-missing frame any is false and all is true.
        # GH#12863: numpy gives back non-boolean data for object type
        # so fill NaNs to compare with pandas behavior
        frame = bool_frame_with_na.fillna(True)
        alternative = getattr(np, all_boolean_reductions)
        f = getattr(frame, all_boolean_reductions)

        def skipna_wrapper(x):
            # [FUNCTION DOC]
            # HELPER: for the bool frame test. Applies the NumPy reference to the Series
            # values after dropping missing entries.
            nona = x.dropna().values
            return alternative(nona)

        def wrapper(x):
            # [FUNCTION DOC]
            # HELPER: for the bool frame test. Applies the NumPy reference to the Series
            # values as is.
            return alternative(x.values)

        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)

        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1))

        result0 = f(axis=0)
        result1 = f(axis=1)

        tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
        tm.assert_series_equal(
            result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False
        )

        # bad axis
        with pytest.raises(ValueError, match="No axis named 2"):
            f(axis=2)

        # all NA case
        all_na = frame * np.nan
        r0 = getattr(all_na, all_boolean_reductions)(axis=0)
        r1 = getattr(all_na, all_boolean_reductions)(axis=1)
        if all_boolean_reductions == "any":
            assert not r0.any()
            assert not r1.any()
        else:
            assert r0.all()
            assert r1.all()

    def test_any_all_extra(self, using_python_scalars):
        # [FUNCTION DOC]
        # Row-wise any/all on subsets of a small boolean frame (with and without
        # bool-only) give the expected Series; with axis none the whole-frame any/all
        # return a scalar (a built-in boolean when the python-scalars option is on, else
        # a NumPy boolean converted by the test), identical to false, true and true.
        df = DataFrame(
            {
                "A": [True, False, False],
                "B": [True, True, False],
                "C": [True, True, True],
            },
            index=["a", "b", "c"],
        )
        result = df[["A", "B"]].any(axis=1)
        expected = Series([True, True, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        result = df[["A", "B"]].any(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)

        result = df.all(axis=1)
        expected = Series([True, False, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        result = df.all(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)

        # Axis is None
        result = df.all(axis=None)
        if not using_python_scalars:
            result = result.item()
        assert result is False

        result = df.any(axis=None)
        if not using_python_scalars:
            result = result.item()
        assert result is True

        result = df[["C"]].all(axis=None)
        if not using_python_scalars:
            result = result.item()
        assert result is True

    @pytest.mark.parametrize("axis", [0, 1])
    def test_any_all_object_dtype(self, axis, all_boolean_reductions, skipna):
        # [FUNCTION DOC]
        # GH#35450: parametrized over axis, any/all fixture and skipna. For an
        # object-dtype frame mixing numbers, missing values, a string and booleans,
        # every row/column is reported true.
        # GH#35450
        df = DataFrame(
            data=[
                [1, np.nan, np.nan, True],
                [np.nan, 2, np.nan, True],
                [np.nan, np.nan, np.nan, True],
                [np.nan, np.nan, "5", np.nan],
            ]
        )
        result = getattr(df, all_boolean_reductions)(axis=axis, skipna=skipna)
        expected = Series([True, True, True, True])
        tm.assert_series_equal(result, expected)

    def test_any_datetime(self):
        # [FUNCTION DOC]
        # GH 23070: row-wise any on a frame with a float and a datetime column raises a
        # type error saying the datetime type does not support any.
        # GH 23070
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [
            Timestamp("1960-02-15"),
            Timestamp("1960-02-16"),
            pd.NaT,
            pd.NaT,
        ]
        df = DataFrame({"A": float_data, "B": datetime_data})

        msg = "datetime64 type does not support operation 'any'"
        with pytest.raises(TypeError, match=msg):
            df.any(axis=1)

    def test_any_all_bool_only(self):
        # [FUNCTION DOC]
        # GH 25101: all with bool-only on a frame without boolean columns gives an empty
        # boolean Series; with a boolean column among numeric and all-None columns the
        # result contains only that column.
        # GH 25101
        df = DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [None, None, None]},
            columns=Index(["col1", "col2", "col3"], dtype=object),
        )

        result = df.all(bool_only=True)
        expected = Series(dtype=np.bool_, index=[])
        tm.assert_series_equal(result, expected)

        df = DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [None, None, None],
                "col4": [False, False, True],
            }
        )

        result = df.all(bool_only=True)
        expected = Series({"col4": False})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, data, expected",
        [
            (np.any, {}, False),
            (np.all, {}, True),
            (np.any, {"A": []}, False),
            (np.all, {"A": []}, True),
            (np.any, {"A": [False, False]}, False),
            (np.all, {"A": [False, False]}, False),
            (np.any, {"A": [True, False]}, True),
            (np.all, {"A": [True, False]}, False),
            (np.any, {"A": [True, True]}, True),
            (np.all, {"A": [True, True]}, True),
            (np.any, {"A": [False], "B": [False]}, False),
            (np.all, {"A": [False], "B": [False]}, False),
            (np.any, {"A": [False, False], "B": [False, True]}, True),
            (np.all, {"A": [False, False], "B": [False, True]}, False),
            # other types
            (np.all, {"A": Series([0.0, 1.0], dtype="float")}, False),
            (np.any, {"A": Series([0.0, 1.0], dtype="float")}, True),
            (np.all, {"A": Series([0, 1], dtype=int)}, False),
            (np.any, {"A": Series([0, 1], dtype=int)}, True),
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns]")}, False),
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, False),
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns]")}, True),
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.all, {"A": Series([0, 1], dtype="m8[ns]")}, False),
            pytest.param(np.any, {"A": Series([0, 1], dtype="m8[ns]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            # np.all on Categorical raises, so the reduction drops the
            #  column, so all is being done on an empty Series, so is True
            (np.all, {"A": Series([0, 1], dtype="category")}, True),
            (np.any, {"A": Series([0, 1], dtype="category")}, False),
            (np.all, {"A": Series([1, 2], dtype="category")}, True),
            (np.any, {"A": Series([1, 2], dtype="category")}, False),
            # Mix GH#21484
            pytest.param(
                np.all,
                {
                    "A": Series([10, 20], dtype="M8[ns]"),
                    "B": Series([10, 20], dtype="m8[ns]"),
                },
                True,
            ),
        ],
    )
    def test_any_all_np_func(self, func, data, expected, using_python_scalars):
        # [FUNCTION DOC]
        # GH 19976: parametrized over NumPy any/all, data (empty, booleans, floats,
        # integers, datetimes, timedeltas, categoricals, a mix) and the expected
        # boolean. For categorical data both the NumPy function and the method with axis
        # none raise a type error about the category dtype; for datetime data they raise
        # a type error that datetime does not support any/all (GH#34479); otherwise
        # results are the expected boolean in the python-scalar or NumPy-boolean form,
        # for both the function and the method.
        # GH 19976
        data = DataFrame(data)

        if any(isinstance(x, CategoricalDtype) for x in data.dtypes):
            with pytest.raises(
                TypeError, match=".* dtype category does not support operation"
            ):
                func(data)

            # method version
            with pytest.raises(
                TypeError, match=".* dtype category does not support operation"
            ):
                getattr(DataFrame(data), func.__name__)(axis=None)
        if data.dtypes.apply(lambda x: x.kind == "M").any():
            # GH#34479
            msg = "datetime64 type does not support operation '(any|all)'"
            with pytest.raises(TypeError, match=msg):
                func(data)

            # method version
            with pytest.raises(TypeError, match=msg):
                getattr(DataFrame(data), func.__name__)(axis=None)

        elif data.dtypes.apply(lambda x: x != "category").any():
            result = func(data)
            if using_python_scalars:
                assert result is expected
            else:
                assert isinstance(result, np.bool_)
                assert result.item() is expected

            # method version
            result = getattr(DataFrame(data), func.__name__)(axis=None)
            if using_python_scalars:
                assert result is expected
            else:
                assert isinstance(result, np.bool_)
                assert result.item() is expected

    def test_any_all_object(self, using_python_scalars):
        # [FUNCTION DOC]
        # GH 19976: NumPy all of an empty-rowed frame with columns is true and any is
        # false (converted from NumPy booleans when the python-scalars option is off).
        # GH 19976
        result = np.all(DataFrame(columns=["a", "b"]))
        if not using_python_scalars:
            result = result.item()
        assert result is True

        result = np.any(DataFrame(columns=["a", "b"]))
        if not using_python_scalars:
            result = result.item()
        assert result is False

    def test_any_all_object_bool_only(self):
        # [FUNCTION DOC]
        # A frame with an object column, a boolean object column, a boolean column and a
        # categorical-of-bool column: the private boolean-data selector returns only the
        # true boolean column, all/any with bool-only return a Series for that column,
        # subsets do not produce a larger Series, and axis-none results are true.
        df = DataFrame({"A": ["foo", 2], "B": [True, False]}).astype(object)
        df._consolidate_inplace()
        df["C"] = Series([True, True])

        # Categorical of bools is _not_ considered booly
        df["D"] = df["C"].astype("category")

        # The underlying bug is in DataFrame._get_bool_data, so we check
        #  that while we're here
        res = df._get_bool_data()
        expected = df[["C"]]
        tm.assert_frame_equal(res, expected)

        res = df.all(bool_only=True, axis=0)
        expected = Series([True], index=["C"])
        tm.assert_series_equal(res, expected)

        # operating on a subset of columns should not produce a _larger_ Series
        res = df[["B", "C"]].all(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)

        assert df.all(bool_only=True, axis=None)

        res = df.any(bool_only=True, axis=0)
        expected = Series([True], index=["C"])
        tm.assert_series_equal(res, expected)

        # operating on a subset of columns should not produce a _larger_ Series
        res = df[["C"]].any(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)

        assert df.any(bool_only=True, axis=None)

    # ---------------------------------------------------------------------
    # Unsorted

    def test_series_broadcasting(self):
        # [FUNCTION DOC]
        # Smoke test for NumPy warnings (GH 16378, GH 16306): clipping with a Series
        # lower bound and comparison methods against a Series with missing values along
        # axis zero emit no warnings.
        # smoke test for numpy warnings
        # GH 16378, GH 16306
        df = DataFrame([1.0, 1.0, 1.0])
        df_nan = DataFrame({"A": [np.nan, 2.0, np.nan]})
        s = Series([1, 1, 1])
        s_nan = Series([np.nan, np.nan, 1])

        with tm.assert_produces_warning(None):
            df_nan.clip(lower=s, axis=0)
            for op in ["lt", "le", "gt", "ge", "eq", "ne"]:
                getattr(df, op)(s_nan, axis=0)


class TestDataFrameReductions:
    # [CLASS DOC]
    # Test group for datetime-like and misc reductions: min/max with NaT, timezone
    # preservation, any with timedelta, skipna validation, and smallest-unit
    # results.
    def test_min_max_dt64_with_NaT(self):
        # [FUNCTION DOC]
        # Min and max of a datetime column with NaT and one timestamp give that
        # timestamp; of all-NaT columns give NaT (GH12941).
        # Both NaT and Timestamp are in DataFrame.
        df = DataFrame({"foo": [pd.NaT, pd.NaT, Timestamp("2012-05-01")]})

        res = df.min()
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)

        res = df.max()
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)

        # GH12941, only NaTs are in DataFrame.
        df = DataFrame({"foo": [pd.NaT, pd.NaT]})

        res = df.min()
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

        res = df.max()
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_precision(self):
        # [FUNCTION DOC]
        # GH#60646: row-wise min and max of a nanosecond datetime column with NaT and
        # one timestamp keep full precision (no float cast), equal to the column
        # without name.
        # GH#60646 Make sure the reduction doesn't cast input timestamps to
        # float and lose precision.
        df = DataFrame(
            {"foo": [pd.NaT, pd.NaT, Timestamp("2012-05-01 09:20:00.123456789")]},
            dtype="datetime64[ns]",
        )

        res = df.min(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)

        res = df.max(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)

    def test_min_max_td64_with_NaT_precision(self):
        # [FUNCTION DOC]
        # GH#60646: same precision check as for datetimes but for a nanosecond timedelta
        # column.
        # GH#60646 Make sure the reduction doesn't cast input timedeltas to
        # float and lose precision.
        df = DataFrame(
            {
                "foo": [
                    pd.NaT,
                    pd.NaT,
                    to_timedelta("10000 days 06:05:01.123456789"),
                ],
            },
            dtype="timedelta64[ns]",
        )

        res = df.min(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)

        res = df.max(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_skipna_false(self, request, tz_naive_fixture):
        # [FUNCTION DOC]
        # GH#36907: parametrized over timezone fixture (skipped for local timezones on
        # Windows). Row-wise min and max with skipna false of two datetime columns, one
        # with NaT and a pre-1970 date, give NaT for the row with missing and the extreme
        # value otherwise, keeping dtype.
        # KNOWN ISSUE: the request fixture parameter is accepted but never used.
        # GH#36907
        tz = tz_naive_fixture
        if isinstance(tz, tzlocal) and is_platform_windows():
            pytest.skip(
                "GH#37659 OSError raised within tzlocal bc Windows "
                "chokes in times before 1970-01-01"
            )

        df = DataFrame(
            {
                "a": [
                    Timestamp("2020-01-01 08:00:00", tz=tz),
                    Timestamp("1920-02-01 09:00:00", tz=tz),
                ],
                "b": [Timestamp("2020-02-01 08:00:00", tz=tz), pd.NaT],
            }
        )
        res = df.min(axis=1, skipna=False)
        expected = Series([df.loc[0, "a"], pd.NaT])
        assert expected.dtype == df["a"].dtype

        tm.assert_series_equal(res, expected)

        res = df.max(axis=1, skipna=False)
        expected = Series([df.loc[0, "b"], pd.NaT])
        assert expected.dtype == df["a"].dtype

        tm.assert_series_equal(res, expected)

    def test_min_max_dt64_api_consistency_with_NaT(self):
        # [FUNCTION DOC]
        # GH:33704: min/max of an empty datetime column behave consistently between
        # frame and Series: the column-wise results are NaT iff the Series result is, and
        # the row-wise result equals an empty datetime Series.
        # Calling the following sum functions returned an error for dataframes but
        # returned NaT for series. These tests check that the API is consistent in
        # min/max calls on empty Series/DataFrames. See GH:33704 for more
        # information
        df = DataFrame({"x": to_datetime([])})
        expected_dt_series = Series(to_datetime([]))
        # check axis 0
        assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
        assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)

        # check axis 1
        tm.assert_series_equal(df.min(axis=1), expected_dt_series)
        tm.assert_series_equal(df.max(axis=1), expected_dt_series)

    def test_min_max_dt64_api_consistency_empty_df(self):
        # [FUNCTION DOC]
        # Min/max of a frame with an empty float column are consistent with the empty
        # Series: column-wise results are NaN, and row-wise results equal an empty float
        # Series.
        # KNOWN ISSUE: the last row-wise check calls the minimum twice; the maximum is never checked row-wise.
        # check DataFrame/Series api consistency when calling min/max on an empty
        # DataFrame/Series.
        df = DataFrame({"x": []})
        expected_float_series = Series([], dtype=float)
        # check axis 0
        assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
        assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
        # check axis 1
        tm.assert_series_equal(df.min(axis=1), expected_float_series)
        tm.assert_series_equal(df.min(axis=1), expected_float_series)

    @pytest.mark.parametrize(
        "initial",
        ["2018-10-08 13:36:45+00:00", "2018-10-08 13:36:45+03:00"],  # Non-UTC timezone
    )
    @pytest.mark.parametrize("method", ["min", "max"])
    def test_preserve_timezone(self, initial: str, method):
        # [FUNCTION DOC]
        # GH 28552: parametrized over min/max and two timestamp strings (UTC and
        # non-UTC offsets). Row-wise min/max of a frame holding one timezone-aware value
        # preserves the timezone.
        # GH 28552
        initial_dt = to_datetime(initial)
        expected = Series([initial_dt])
        df = DataFrame([expected])
        result = getattr(df, method)(axis=1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_minmax_tzaware_skipna_axis_1(self, method, skipna):
        # [FUNCTION DOC]
        # GH#51242: parametrized over min/max and skipna. Row-wise reductions of two UTC
        # columns with NaT: skipping missing gives NaT only for the all-missing row;
        # not skipping gives NaT for any row with a missing value.
        # GH#51242
        val = to_datetime("1900-01-01", utc=True)
        df = DataFrame(
            {"a": Series([pd.NaT, pd.NaT, val]), "b": Series([pd.NaT, val, val])}
        )
        op = getattr(df, method)
        result = op(axis=1, skipna=skipna)
        if skipna:
            expected = Series([pd.NaT, val, val])
        else:
            expected = Series([pd.NaT, pd.NaT, val])
        tm.assert_series_equal(result, expected)

    def test_frame_any_with_timedelta(self):
        # [FUNCTION DOC]
        # GH#17667: any on a frame of integer zeros and a timedelta column treats zero
        # timedelta as false and non-zero as true, both column-wise and row-wise.
        # GH#17667
        df = DataFrame(
            {
                "a": Series([0, 0]),
                "t": Series([to_timedelta(0, "s"), to_timedelta(1, "ms")]),
            }
        )

        result = df.any(axis=0)
        expected = Series(data=[False, True], index=["a", "t"])
        tm.assert_series_equal(result, expected)

        result = df.any(axis=1)
        expected = Series(data=[False, True])
        tm.assert_series_equal(result, expected)

    def test_reductions_skipna_none_raises(
        self, request, frame_or_series, all_reductions
    ):
        # [FUNCTION DOC]
        # Parametrized over frame/Series and reduction fixture. Passing a missing
        # value for skipna raises a value error saying the argument expected a boolean;
        # count is marked expected-fail since it does not take skipna.
        if all_reductions == "count":
            request.applymarker(
                pytest.mark.xfail(reason="Count does not accept skipna")
            )
        obj = frame_or_series([1, 2, 3])
        msg = 'For argument "skipna" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            getattr(obj, all_reductions)(skipna=None)

    def test_reduction_timestamp_smallest_unit(self):
        # [FUNCTION DOC]
        # GH#52524: max of a second-resolution and a millisecond-resolution timestamp
        # column gives timestamps in the finest resolution (milliseconds).
        # GH#52524
        df = DataFrame(
            {
                "a": Series([Timestamp("2019-12-31")], dtype="datetime64[s]"),
                "b": Series(
                    [Timestamp("2019-12-31 00:00:00.123")], dtype="datetime64[ms]"
                ),
            }
        )
        result = df.max()
        expected = Series(
            [Timestamp("2019-12-31"), Timestamp("2019-12-31 00:00:00.123")],
            dtype="datetime64[ms]",
            index=["a", "b"],
        )
        tm.assert_series_equal(result, expected)

    def test_reduction_timedelta_smallest_unit(self):
        # [FUNCTION DOC]
        # GH#52524: same as the timestamp case but for timedelta columns, giving
        # millisecond resolution.
        # GH#52524
        df = DataFrame(
            {
                "a": Series([pd.Timedelta("1 days")], dtype="timedelta64[s]"),
                "b": Series([pd.Timedelta("1 days")], dtype="timedelta64[ms]"),
            }
        )
        result = df.max()
        expected = Series(
            [pd.Timedelta("1 days"), pd.Timedelta("1 days")],
            dtype="timedelta64[ms]",
            index=["a", "b"],
        )
        tm.assert_series_equal(result, expected)


class TestNuisanceColumns:
    # [CLASS DOC]
    # Test group for how reductions treat columns that cannot be reduced
    # ("nuisance" columns), such as categorical columns with any/all.
    def test_any_all_categorical_dtype_nuisance_column(self, all_boolean_reductions):
        # [FUNCTION DOC]
        # GH#36076: for a categorical column, Series any/all, the NumPy function on the
        # Series, and frame any/all with bool-only false or none, all raise a type error
        # saying the operation is not supported (the frame matches the Series).
        # GH#36076 DataFrame should match Series behavior
        ser = Series([0, 1], dtype="category", name="A")
        df = ser.to_frame()

        # Double-check the Series behavior is to raise
        with pytest.raises(TypeError, match="does not support operation"):
            getattr(ser, all_boolean_reductions)()

        with pytest.raises(TypeError, match="does not support operation"):
            getattr(np, all_boolean_reductions)(ser)

        with pytest.raises(TypeError, match="does not support operation"):
            getattr(df, all_boolean_reductions)(bool_only=False)

        with pytest.raises(TypeError, match="does not support operation"):
            getattr(df, all_boolean_reductions)(bool_only=None)

        with pytest.raises(TypeError, match="does not support operation"):
            getattr(np, all_boolean_reductions)(df, axis=0)

    def test_median_categorical_dtype_nuisance_column(self):
        # [FUNCTION DOC]
        # GH#21020: frame median on a categorical column (with numeric-only false or the
        # default) raises a type error saying the operation is not supported, as the
        # Series does; also when an integer column is added alongside. A comment notes a
        # known difference for the NumPy median function.
        # GH#21020 DataFrame.median should match Series.median
        df = DataFrame({"A": Categorical([1, 2, 2, 2, 3])})
        ser = df["A"]

        # Double-check the Series behavior is to raise
        with pytest.raises(TypeError, match="does not support operation"):
            ser.median()

        with pytest.raises(TypeError, match="does not support operation"):
            df.median(numeric_only=False)

        with pytest.raises(TypeError, match="does not support operation"):
            df.median()

        # same thing, but with an additional non-categorical column
        df["B"] = df["A"].astype(int)

        with pytest.raises(TypeError, match="does not support operation"):
            df.median(numeric_only=False)

        with pytest.raises(TypeError, match="does not support operation"):
            df.median()

        # TODO: np.median(df, axis=0) gives np.array([2.0, 2.0]) instead
        #  of expected.values

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_min_max_categorical_dtype_non_ordered_nuisance_column(self, method):
        # [FUNCTION DOC]
        # GH#28949: parametrized over min/max. For an unordered categorical, Series, the
        # NumPy function on a Series, frame methods (numeric-only false and default) and
        # the NumPy function on the frame all raise a type error saying the categorical is
        # not ordered for the operation; the same with an added object column.
        # GH#28949 DataFrame.min should behave like Series.min
        cat = Categorical(["a", "b", "c", "b"], ordered=False)
        ser = Series(cat)
        df = ser.to_frame("A")

        # Double-check the Series behavior
        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(ser, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(ser)

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)(numeric_only=False)

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(df, axis=0)

        # same thing, but with an additional non-categorical column
        df["B"] = df["A"].astype(object)
        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(df, axis=0)


class TestEmptyDataFrameReductions:
    # [CLASS DOC]
    # Test group for sum and product (and covariance) on frames with no rows.
    @pytest.mark.parametrize(
        "opname, dtype, exp_value, exp_dtype",
        [
            ("sum", np.int8, 0, np.int64),
            ("prod", np.int8, 1, np.int_),
            ("sum", np.int64, 0, np.int64),
            ("prod", np.int64, 1, np.int64),
            ("sum", np.uint8, 0, np.uint64),
            ("prod", np.uint8, 1, np.uint),
            ("sum", np.uint64, 0, np.uint64),
            ("prod", np.uint64, 1, np.uint64),
            ("sum", np.float32, 0, np.float32),
            ("prod", np.float32, 1, np.float32),
            ("sum", np.float64, 0, np.float64),
        ],
    )
    def test_df_empty_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized over sum/product, integer/unsigned/float dtype, expected value and
        # expected result dtype. With minimum count zero, an empty two-column frame gives
        # the identity (zero for sum, one for product) per column, with widened integer
        # result dtype (small integers widen to the platform default or 64-bit).
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)

        expected = Series([exp_value, exp_value], dtype=exp_dtype, index=range(2))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "opname, dtype, exp_dtype",
        [
            ("sum", np.int8, np.float64),
            ("prod", np.int8, np.float64),
            ("sum", np.int64, np.float64),
            ("prod", np.int64, np.float64),
            ("sum", np.uint8, np.float64),
            ("prod", np.uint8, np.float64),
            ("sum", np.uint64, np.float64),
            ("prod", np.uint64, np.float64),
            ("sum", np.float32, np.float32),
            ("prod", np.float32, np.float32),
            ("sum", np.float64, np.float64),
        ],
    )
    def test_df_empty_min_count_1(self, opname, dtype, exp_dtype):
        # [FUNCTION DOC]
        # Parametrized as above. With minimum count one, an empty frame gives missing
        # (NaN) for each column; integer dtypes become float64, float dtypes keep their
        # width.
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)

        expected = Series([np.nan, np.nan], dtype=exp_dtype, index=Index([0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "opname, dtype, exp_value, exp_dtype",
        [
            ("sum", "Int8", 0, ("Int32" if is_windows_np2_or_is32 else "Int64")),
            ("prod", "Int8", 1, ("Int32" if is_windows_np2_or_is32 else "Int64")),
            ("sum", "Int64", 0, "Int64"),
            ("prod", "Int64", 1, "Int64"),
            ("sum", "UInt8", 0, ("UInt32" if is_windows_np2_or_is32 else "UInt64")),
            ("prod", "UInt8", 1, ("UInt32" if is_windows_np2_or_is32 else "UInt64")),
            ("sum", "UInt64", 0, "UInt64"),
            ("prod", "UInt64", 1, "UInt64"),
            ("sum", "Float32", 0, "Float32"),
            ("prod", "Float32", 1, "Float32"),
            ("sum", "Float64", 0, "Float64"),
        ],
    )
    def test_df_empty_nullable_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        # [FUNCTION DOC]
        # Like the zero minimum-count test, but for nullable dtypes. Small nullable
        # integers widen to 32-bit on Windows with older NumPy or on 32-bit platforms, else
        # to 64-bit; others keep their dtype. Result is the identity per column.
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)

        expected = Series([exp_value, exp_value], dtype=exp_dtype, index=Index([0, 1]))
        tm.assert_series_equal(result, expected)

    # TODO: why does min_count=1 impact the resulting Windows dtype
    # differently than min_count=0?
    @pytest.mark.parametrize(
        "opname, dtype, exp_dtype",
        [
            ("sum", "Int8", ("Int32" if is_windows_or_is32 else "Int64")),
            ("prod", "Int8", ("Int32" if is_windows_or_is32 else "Int64")),
            ("sum", "Int64", "Int64"),
            ("prod", "Int64", "Int64"),
            ("sum", "UInt8", ("UInt32" if is_windows_or_is32 else "UInt64")),
            ("prod", "UInt8", ("UInt32" if is_windows_or_is32 else "UInt64")),
            ("sum", "UInt64", "UInt64"),
            ("prod", "UInt64", "UInt64"),
            ("sum", "Float32", "Float32"),
            ("prod", "Float32", "Float32"),
            ("sum", "Float64", "Float64"),
        ],
    )
    def test_df_empty_nullable_min_count_1(self, opname, dtype, exp_dtype):
        # [FUNCTION DOC]
        # Like the minimum-count one test for nullable dtypes: result is missing (NA) per
        # column in the (possibly widened) nullable dtype, where widening to 32-bit
        # depends on Windows or 32-bit platform.
        # KNOWN ISSUE: a comment above this test is an open TODO asking why the minimum count changes the Windows dtype choice differently from the zero case (the platform flag used here differs from the one in the zero case).
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)

        expected = Series([pd.NA, pd.NA], dtype=exp_dtype, index=Index([0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            {"a": [0, 1, 2], "b": [pd.NaT, pd.NaT, pd.NaT]},
            {"a": [0, 1, 2], "b": [Timestamp("1990-01-01"), pd.NaT, pd.NaT]},
            {
                "a": [0, 1, 2],
                "b": [
                    Timestamp("1990-01-01"),
                    Timestamp("1991-01-01"),
                    Timestamp("1992-01-01"),
                ],
            },
            {
                "a": [0, 1, 2],
                "b": [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.NaT],
            },
            {
                "a": [0, 1, 2],
                "b": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            },
        ],
    )
    def test_df_cov_pd_nat(self, data):
        # [FUNCTION DOC]
        # GH #53115: parametrized over frames with an integer column and datetime or
        # timedelta columns containing NaT. Covariance raises a type error saying the type
        # is not supported for cov.
        # GH #53115
        df = DataFrame(data)
        with pytest.raises(TypeError, match="not supported for cov"):
            df.cov()


def test_sum_timedelta64_skipna_false():
    # [FUNCTION DOC]
    # GH#17235: sum with skipna false on a 4x2 second-resolution timedelta frame
    # containing one NaT gives the sum for the first column and NaT for the second,
    # both by default and with axis zero, and row sums with NaT only in the last
    # row.
    # GH#17235
    arr = np.arange(8).astype(np.int64).view("m8[s]").reshape(4, 2)
    arr[-1, -1] = "Nat"

    df = DataFrame(arr)
    assert (df.dtypes == arr.dtype).all()

    result = df.sum(skipna=False)
    expected = Series([pd.Timedelta(seconds=12), pd.NaT], dtype="m8[s]")
    tm.assert_series_equal(result, expected)

    result = df.sum(axis=0, skipna=False)
    tm.assert_series_equal(result, expected)

    result = df.sum(axis=1, skipna=False)
    expected = Series(
        [
            pd.Timedelta(seconds=1),
            pd.Timedelta(seconds=5),
            pd.Timedelta(seconds=9),
            pd.NaT,
        ],
        dtype="m8[s]",
    )
    tm.assert_series_equal(result, expected)


def test_mixed_frame_with_integer_sum():
    # [FUNCTION DOC]
    # Issue 34520: summing a frame of a string column and a nullable-integer column
    # returns an object Series of the string and integer one.
    # https://github.com/pandas-dev/pandas/issues/34520
    df = DataFrame([["a", 1]], columns=list("ab"))
    df = df.astype({"b": "Int64"})
    result = df.sum()
    expected = Series(["a", 1], index=["a", "b"])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False, None])
@pytest.mark.parametrize("method", ["min", "max"])
def test_minmax_extensionarray(method, numeric_only):
    # [FUNCTION DOC]
    # Issue 32651: module-level test parametrized over min/max and numeric-only
    # (true, false, none). A nullable 64-bit integer column holding the largest and
    # smallest 64-bit integers plus a missing value yields the extreme integer as a
    # nullable-integer Series.
    # https://github.com/pandas-dev/pandas/issues/32651
    int64_info = np.iinfo("int64")
    ser = Series([int64_info.max, None, int64_info.min], dtype=pd.Int64Dtype())
    df = DataFrame({"Int64": ser})
    result = getattr(df, method)(numeric_only=numeric_only)
    expected = Series(
        [getattr(int64_info, method)],
        dtype="Int64",
        index=Index(["Int64"]),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ts_value", [Timestamp("2000-01-01"), pd.NaT])
def test_frame_mixed_numeric_object_with_timestamp(ts_value):
    # [FUNCTION DOC]
    # GH 13912: parametrized over a timestamp and NaT. Summing a frame with integer,
    # float, string and datetime columns raises a type error saying the operation is
    # not supported (or cannot be performed).
    # GH 13912
    df = DataFrame({"a": [1], "b": [1.1], "c": ["foo"], "d": [ts_value]})
    with pytest.raises(TypeError, match="does not support operation|Cannot perform"):
        df.sum()


def test_prod_sum_min_count_mixed_object():
    # [FUNCTION DOC]
    # Issue 41074: product with minimum count one and numeric-only false on an
    # object column of integer, string and boolean gives the string as an object
    # Series; sum on it raises a type error about adding an integer and a string.
    # https://github.com/pandas-dev/pandas/issues/41074
    df = DataFrame([1, "a", True])

    result = df.prod(axis=0, min_count=1, numeric_only=False)
    expected = Series(["a"], dtype=object)
    tm.assert_series_equal(result, expected)

    msg = re.escape("unsupported operand type(s) for +: 'int' and 'str'")
    with pytest.raises(TypeError, match=msg):
        df.sum(axis=0, min_count=1, numeric_only=False)


@pytest.mark.parametrize("method", ["min", "max", "mean", "median", "skew", "kurt"])
@pytest.mark.parametrize("numeric_only", [True, False])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_reduction_axis_none_returns_scalar(method, numeric_only, dtype):
    # [FUNCTION DOC]
    # GH#21597: parametrized over method (min, max, mean, median, skew, kurtosis),
    # numeric-only and float dtype (plain or nullable). Reducing with axis none
    # reduces over all values and returns a scalar equal to the NumPy (or scipy,
    # unbiased, skipped without scipy) result on the underlying array.
    # GH#21597 As of 2.0, axis=None reduces over all axes.

    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), dtype=dtype)

    result = getattr(df, method)(axis=None, numeric_only=numeric_only)
    np_arr = df.to_numpy(dtype=np.float64)
    if method in {"skew", "kurt"}:
        comp_mod = pytest.importorskip("scipy.stats")
        if method == "kurt":
            method = "kurtosis"
        expected = getattr(comp_mod, method)(np_arr, bias=False, axis=None)
        tm.assert_almost_equal(result, expected)
    else:
        expected = getattr(np, method)(np_arr, axis=None)
        assert result == expected


@pytest.mark.parametrize(
    "kernel",
    [
        "corr",
        "corrwith",
        "cov",
        "idxmax",
        "idxmin",
        "kurt",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
    ],
)
def test_fails_on_non_numeric(kernel):
    # [FUNCTION DOC]
    # GH#46852: parametrized over many reduction/statistics names. On a frame with a
    # numeric column and a column filled with the object type itself, each raises a
    # type error with one of several messages (the median message names the
    # conversion failure for the object values; correlation-with-another is called with
    # the frame as argument).
    # GH#46852
    df = DataFrame({"a": [1, 2, 3], "b": object})
    args = (df,) if kernel == "corrwith" else ()
    msg = "|".join(
        [
            "not allowed for this dtype",
            "argument must be a string or a number",
            "not supported between instances of",
            "unsupported operand type",
            "argument must be a string or a real number",
        ]
    )
    if kernel == "median":
        # slightly different message on different builds
        msg1 = (
            r"Cannot convert \[\[<class 'object'> <class 'object'> "
            r"<class 'object'>\]\] to numeric"
        )
        msg2 = (
            r"Cannot convert \[<class 'object'> <class 'object'> "
            r"<class 'object'>\] to numeric"
        )
        msg = "|".join([msg1, msg2])
    with pytest.raises(TypeError, match=msg):
        getattr(df, kernel)(*args)


@pytest.mark.parametrize(
    "method",
    [
        "all",
        "any",
        "count",
        "idxmax",
        "idxmin",
        "kurt",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "nunique",
        "prod",
        "product",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
    ],
)
@pytest.mark.parametrize("min_count", [0, 2])
def test_numeric_ea_axis_1(
    method, skipna, min_count, any_numeric_ea_dtype, using_nan_is_na
):
    # [FUNCTION DOC]
    # GH 54341: parametrized over many reduction names, skipna fixture, minimum count
    # (zero, two), numeric extension dtype fixture and the nan-is-NA option. Compare
    # row-wise reduction of a frame of two nullable columns (one with a missing value)
    # to the same reduction of an equivalent float frame with NaN, converted to the
    # expected dtype (count/nunique 64-bit integer, any/all nullable boolean, moments
    # as nullable float64 unless already float, otherwise the dtype); with the option
    # off, missing positions are restored to NA. For index-of-extreme with skipna
    # false, both raise a value error about an NA value and the test ends.
    # GH 54341
    df = DataFrame(
        {
            "a": Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype),
            "b": Series([0, 1, pd.NA, 3], dtype=any_numeric_ea_dtype),
        },
    )
    expected_df = DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [0.0, 1.0, np.nan, 3.0],
        },
    )
    if method in ("count", "nunique"):
        expected_dtype = "int64"
    elif method in ("all", "any"):
        expected_dtype = "boolean"
    elif method in (
        "kurt",
        "kurtosis",
        "mean",
        "median",
        "sem",
        "skew",
        "std",
        "var",
    ) and not any_numeric_ea_dtype.startswith("Float"):
        expected_dtype = "Float64"
    else:
        expected_dtype = any_numeric_ea_dtype

    kwargs = {}
    if method not in ("count", "nunique", "quantile"):
        kwargs["skipna"] = skipna
    if method in ("prod", "product", "sum"):
        kwargs["min_count"] = min_count

    if not skipna and method in ("idxmax", "idxmin"):
        with pytest.raises(ValueError, match="encountered an NA value"):
            getattr(df, method)(axis=1, **kwargs)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(expected_df, method)(axis=1, **kwargs)
        return
    result = getattr(df, method)(axis=1, **kwargs)
    expected = getattr(expected_df, method)(axis=1, **kwargs)
    if method not in ("idxmax", "idxmin"):
        if using_nan_is_na:
            expected = expected.astype(expected_dtype)
        else:
            mask = np.isnan(expected)
            expected[mask] = 0
            expected = expected.astype(expected_dtype)
            expected[mask] = pd.NA
    tm.assert_series_equal(result, expected)


def test_mean_nullable_int_axis_1():
    # [FUNCTION DOC]
    # GH##36585: row-wise mean of an integer and a nullable-integer column gives a
    # nullable Float64 Series; with skipna true the missing value is skipped; with
    # skipna false the row with the missing value is NA.
    # GH##36585
    df = DataFrame(
        {"a": [1, 2, 3, 4], "b": Series([1, 2, 4, None], dtype=pd.Int64Dtype())}
    )

    result = df.mean(axis=1, skipna=True)
    expected = Series([1.0, 2.0, 3.5, 4.0], dtype="Float64")
    tm.assert_series_equal(result, expected)

    result = df.mean(axis=1, skipna=False)
    expected = Series([1.0, 2.0, 3.5, pd.NA], dtype="Float64")
    tm.assert_series_equal(result, expected)
