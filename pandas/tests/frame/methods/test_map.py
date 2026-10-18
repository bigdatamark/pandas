# [MODULE DOC]
# Tests for DataFrame map (elementwise function application), covering dtype preservation, missing-value handling, kwargs, empty frames and boxing of datetime-like values.

from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


def test_map(float_frame):
    # [FUNCTION DOC]
    # Using float_frame: verifies map with a doubling function equals multiplication by two; mapping the type constructor works; mapping to tuples gives tuple elements (GH 465).
    result = float_frame.map(lambda x: x * 2)
    tm.assert_frame_equal(result, float_frame * 2)
    float_frame.map(type)

    # GH 465: function returning tuples
    result = float_frame.map(lambda x: (x, x))["A"].iloc[0]
    assert isinstance(result, tuple)


@pytest.mark.parametrize("val", [1, 1.0])
def test_map_float_object_conversion(val):
    # [FUNCTION DOC]
    # GH 2909. Parametrized over an int and a float value, with a string, in one column. Verifies an identity map keeps object dtype instead of converting to a numeric dtype.
    # GH 2909: object conversion to float in constructor?
    df = DataFrame(data=[val, "a"])
    result = df.map(lambda x: x).dtypes[0]
    assert result == object


def test_map_keeps_dtype(na_action):
    # [FUNCTION DOC]
    # GH52219. Parametrized over the na_action fixture. Frame with an object column and a sparse object column holding strings and a missing value; the helper uppercases non-missing values. Verifies the dense column result is object dtype, the sparse column keeps its sparse dtype, and an empty slice also preserves the dtypes.
    # GH52219
    arr = Series(["a", np.nan, "b"])
    sparse_arr = arr.astype(pd.SparseDtype(object))
    df = DataFrame(data={"a": arr, "b": sparse_arr})

    def func(x):
        # [FUNCTION DOC]
        # HELPER: function applied by the dtype-preservation test; returns the uppercased string for non-missing input and returns the missing value unchanged.
        return str.upper(x) if not pd.isna(x) else x

    result = df.map(func, na_action=na_action)

    expected_sparse = pd.array(["A", np.nan, "B"], dtype=pd.SparseDtype(object))
    expected_arr = expected_sparse.astype(object)
    expected = DataFrame({"a": expected_arr, "b": expected_sparse})

    tm.assert_frame_equal(result, expected)

    result_empty = df.iloc[:0, :].map(func, na_action=na_action)
    expected_empty = expected.iloc[:0, :]
    tm.assert_frame_equal(result_empty, expected_empty)


def test_map_str():
    # [FUNCTION DOC]
    # GH 2786. Frame with duplicate column labels. Verifies map with the string conversion keeps those labels and equals the result computed on a uniquely-labeled copy.
    # GH 2786
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    df2 = df.copy()
    cols = ["a", "a", "a", "a"]
    df.columns = cols

    expected = df2.map(str)
    expected.columns = cols
    result = df.map(str)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "col, val",
    [["datetime", Timestamp("20130101")], ["timedelta", pd.Timedelta("1 min")]],
)
def test_map_datetimelike(col, val):
    # [FUNCTION DOC]
    # Parametrized over a datetime or timedelta value assigned to a new column. Verifies map with the string conversion on that column gives the string of the original value.
    # datetime/timedelta
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    df[col] = val
    result = df.map(str)
    assert result.loc[0, col] == str(df.loc[0, col])


@pytest.mark.parametrize(
    "expected",
    [
        DataFrame(),
        DataFrame(columns=list("ABC")),
        DataFrame(index=list("ABC")),
        DataFrame({"A": [], "B": [], "C": []}),
    ],
)
@pytest.mark.parametrize("func", [round, lambda x: x])
def test_map_empty(expected, func):
    # [FUNCTION DOC]
    # GH 8222. Parametrized over several empty frames (no rows and/or no columns) and over two functions (rounding and identity). Verifies map on an empty frame returns an equal empty frame.
    # GH 8222
    result = expected.map(func)
    tm.assert_frame_equal(result, expected)


def test_map_kwargs():
    # [FUNCTION DOC]
    # GH 40652. Verifies extra keyword arguments passed to map are forwarded to the function.
    # GH 40652
    result = DataFrame([[1, 2], [3, 4]]).map(lambda x, y: x + y, y=2)
    expected = DataFrame([[3, 4], [5, 6]])
    tm.assert_frame_equal(result, expected)


def test_map_na_ignore(float_frame):
    # [FUNCTION DOC]
    # GH 23803. Compares mapping a string-length function with na_action set to ignore on a frame with random missing entries against the full result with the same positions set to missing. Verifies missing values are not passed to the function and stay missing.
    # GH 23803
    strlen_frame = float_frame.map(lambda x: len(str(x)))
    float_frame_with_na = float_frame.copy()
    mask = np.random.default_rng(2).integers(0, 2, size=float_frame.shape, dtype=bool)
    float_frame_with_na[mask] = pd.NA
    strlen_frame_na_ignore = float_frame_with_na.map(
        lambda x: len(str(x)), na_action="ignore"
    )
    # Set float64 type to avoid upcast when setting NA below
    strlen_frame_with_na = strlen_frame.copy().astype("float64")
    strlen_frame_with_na[mask] = pd.NA
    tm.assert_frame_equal(strlen_frame_na_ignore, strlen_frame_with_na)


def test_map_box_timestamps():
    # [FUNCTION DOC]
    # GH 2689, GH 2627. Smoke test: mapping a function that reads hour, day and month attributes over a datetime column. Verifies the function receives timestamp objects and no error is raised.
    # GH 2689, GH 2627
    ser = Series(date_range("1/1/2000", periods=10))

    def func(x):
        # [FUNCTION DOC]
        # HELPER: function returning the hour, day and month of a timestamp as a tuple; used by the timestamp boxing test.
        return (x.hour, x.day, x.month)

    # it works!
    DataFrame(ser).map(func)


def test_map_box():
    # [FUNCTION DOC]
    # Verifies the function is applied to boxed scalar objects: timestamps (naive and timezone-aware), timedeltas and periods are seen as their pandas scalar types, not raw arrays or integers.
    # KNOWN ISSUE: the inline comment says these are the same test cases as the test named test_map_box, which is this very test; the intended reference was probably the equivalent apply or Series test.
    # ufunc will not be boxed. Same test cases as the test_map_box
    df = DataFrame(
        {
            "a": [Timestamp("2011-01-01"), Timestamp("2011-01-02")],
            "b": [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
            ],
            "c": [pd.Timedelta("1 days"), pd.Timedelta("2 days")],
            "d": [
                pd.Period("2011-01-01", freq="M"),
                pd.Period("2011-01-02", freq="M"),
            ],
        }
    )

    result = df.map(lambda x: type(x).__name__)
    expected = DataFrame(
        {
            "a": ["Timestamp", "Timestamp"],
            "b": ["Timestamp", "Timestamp"],
            "c": ["Timedelta", "Timedelta"],
            "d": ["Period", "Period"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_frame_map_dont_convert_datetime64(unit):
    # [FUNCTION DOC]
    # Parametrized over datetime unit. Frame with one datetime column of that unit; maps a function adding a business day twice. Verifies the column dtype remains the same datetime unit.
    dtype = f"M8[{unit}]"
    df = DataFrame({"x1": [datetime(1996, 1, 1)]}, dtype=dtype)

    df = df.map(lambda x: x + BDay())
    df = df.map(lambda x: x + BDay())

    result = df.x1.dtype
    assert result == dtype


def test_map_function_runs_once():
    # [FUNCTION DOC]
    # Verifies map calls the function exactly once per element, in order, whether it returns nothing or returns its input. A list collects the seen values and must equal the column values.
    df = DataFrame({"a": [1, 2, 3]})
    values = []  # Save values function is applied to

    def reducing_function(val):
        # [FUNCTION DOC]
        # HELPER: records each value it receives in the enclosing list and returns nothing.
        values.append(val)

    def non_reducing_function(val):
        # [FUNCTION DOC]
        # HELPER: records each value it receives in the enclosing list and returns the value unchanged.
        values.append(val)
        return val

    for func in [reducing_function, non_reducing_function]:
        del values[:]

        df.map(func)
        assert values == df.a.to_list()


def test_map_type():
    # [FUNCTION DOC]
    # GH 46719. Verifies map with the type function on object columns containing mixed Python objects returns the type of each element (int, str, type, float, datetime).
    # GH 46719
    df = DataFrame(
        {"col1": [3, "string", float], "col2": [0.25, datetime(2020, 1, 1), np.nan]},
        index=["a", "b", "c"],
    )

    result = df.map(type)
    expected = DataFrame(
        {"col1": [int, str, type], "col2": [float, datetime, float]},
        index=["a", "b", "c"],
    )
    tm.assert_frame_equal(result, expected)


def test_map_invalid_na_action(float_frame):
    # [FUNCTION DOC]
    # GH 23803. Verifies map raises a value error saying na_action must be one of the allowed values and naming the bad value when given an invalid na_action.
    # GH 23803
    with pytest.raises(ValueError, match="na_action must be .*Got 'abc'"):
        float_frame.map(lambda x: len(str(x)), na_action="abc")
