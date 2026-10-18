# [MODULE DOC]
# Tests for the cumulative groupby operations cumprod, cumsum, cummin and cummax on DataFrameGroupBy and SeriesGroupBy: agreement with applying the per-group Series method, overflow behavior, NaN and skipna handling, nullable and datetime dtypes, dtype boundary values, numpy-compat argument rejection and as_index independence.
import numpy as np
import pytest

from pandas.errors import UnsupportedFunctionCall
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


@pytest.fixture(
    params=[np.int32, np.int64, np.float32, np.float64, "Int64", "Float64"],
    ids=["np.int32", "np.int64", "np.float32", "np.float64", "Int64", "Float64"],
)
def dtypes_for_minmax(request):
    # [FUNCTION DOC]
    # HELPER: pytest fixture parametrized over int32, int64, float32, float64, Int64 and Float64 (with readable ids). Returns a tuple of the dtype parameter, the minimum representable value and the maximum representable value of the equivalent numpy type (integer limits for integer kinds, finite float limits for floats; nullable names map to the 64-bit numpy types). Used by the cummin/cummax boundary tests.
    """
    Fixture of dtypes with min and max values used for testing
    cummin and cummax
    """
    dtype = request.param

    np_type = dtype
    if dtype == "Int64":
        np_type = np.int64
    elif dtype == "Float64":
        np_type = np.float64

    min_val = (
        np.iinfo(np_type).min
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).min
    )
    max_val = (
        np.iinfo(np_type).max
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).max
    )

    return (dtype, min_val, max_val)


def test_groupby_cumprod():
    # [FUNCTION DOC]
    # GH 4095: cumprod of a constant integer column of ten rows grouped by a single key, and of a float column with 100 rows, must equal applying the Series cumprod per group, with the result named value.
    # GH 4095
    df = DataFrame({"key": ["b"] * 10, "value": 2})

    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    tm.assert_series_equal(actual, expected)

    df = DataFrame({"key": ["b"] * 100, "value": 2})
    df["value"] = df["value"].astype(float)
    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    tm.assert_series_equal(actual, expected)


def test_groupby_cumprod_overflow():
    # [FUNCTION DOC]
    # GH#37493: cumprod of 100000 repeated four times overflows int64; the result must equal a fixed list ending in an overflowed (wrapped) number consistent with numpy, and also equal the per-group apply result.
    # GH#37493 if we overflow we return garbage consistent with numpy
    df = DataFrame({"key": ["b"] * 4, "value": 100_000})
    actual = df.groupby("key")["value"].cumprod()
    expected = Series(
        [100_000, 10_000_000_000, 1_000_000_000_000_000, 7766279631452241920],
        name="value",
    )
    tm.assert_series_equal(actual, expected)

    numpy_result = df.groupby("key", group_keys=False)["value"].apply(
        lambda x: x.cumprod()
    )
    numpy_result.name = "value"
    tm.assert_series_equal(actual, numpy_result)


def test_groupby_cumprod_nan_influences_other_columns():
    # [FUNCTION DOC]
    # GH#48064: cumprod with numeric_only true and skipna false on a frame grouped by a constant: a NaN in column b propagates to later rows of b only; column c is unaffected (1, 2, 6).
    # GH#48064
    df = DataFrame(
        {
            "a": 1,
            "b": [1, np.nan, 2],
            "c": [1, 2, 3.0],
        }
    )
    result = df.groupby("a").cumprod(numeric_only=True, skipna=False)
    expected = DataFrame({"b": [1, np.nan, np.nan], "c": [1, 2, 6.0]})
    tm.assert_frame_equal(result, expected)


def test_cummin(dtypes_for_minmax):
    # [FUNCTION DOC]
    # GH 15048: cummin for each dtype in the dtypes_for_minmax fixture on two groups gives running minima 3,3,3,2 and 2,2,2,1 in the dtype of the input, equal to the per-group apply of Series cummin.
    dtype = dtypes_for_minmax[0]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]

    df = base_df.astype(dtype)
    expected = DataFrame({"B": expected_mins}).astype(dtype)
    result = df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    tm.assert_frame_equal(result, expected)


def test_cummin_min_value_for_dtype(dtypes_for_minmax):
    # [FUNCTION DOC]
    # GH 15048: cummin when values equal to the dtype's minimum (and minimum plus one) are present must not lose precision: positions after a minimum remain at the minimum, and the value min plus one is not rounded to min. Results compared exactly against expectations and the per-group apply.
    dtype = dtypes_for_minmax[0]
    min_val = dtypes_for_minmax[1]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]
    expected = DataFrame({"B": expected_mins}).astype(dtype)
    df = base_df.astype(dtype)
    df.loc[[2, 6], "B"] = min_val
    df.loc[[1, 5], "B"] = min_val + 1
    expected.loc[[2, 3, 6, 7], "B"] = min_val
    expected.loc[[1, 5], "B"] = min_val + 1  # should not be rounded to min_val
    result = df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected, check_exact=True)
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_cummin_nan_in_some_values(dtypes_for_minmax):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtypes_for_minmax fixture is requested but never used in the body, so the test just runs the same float case once per fixture parameter.
    # cummin with NaN at alternate positions in a float column: NaN rows stay NaN and the running minimum skips NaN within the group (4, 2 in the first group; 3, 1 in the second). Compared to the per-group apply.
    # Explicit cast to float to avoid implicit cast when setting nan
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    base_df = base_df.astype({"B": "float"})
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 2, np.nan, 3, np.nan, 1]})
    result = base_df.groupby("A").cummin()
    tm.assert_frame_equal(result, expected)
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    tm.assert_frame_equal(result, expected)


def test_cummin_datetime():
    # [FUNCTION DOC]
    # GH 15561: cummin of a datetime column of a single row returns that datetime as a Series named b with the original index.
    # GH 15561
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    result = df.groupby("a")["b"].cummin()
    tm.assert_series_equal(expected, result)


def test_cummin_getattr_series():
    # [FUNCTION DOC]
    # GH 15635: cummin through attribute access of column b on a groupby (1, 2, 1 across the rows) returns a Series named b.
    # GH 15635
    df = DataFrame({"a": [1, 2, 1], "b": [1, 2, 2]})
    result = df.groupby("a").b.cummin()
    expected = Series([1, 2, 1], name="b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["UInt64", "Int64", "Float64", "float", "boolean"])
def test_cummin_max_all_nan_column(method, dtype):
    # [FUNCTION DOC]
    # Parametrized over cummin/cummax and nullable or float dtypes (UInt64, Int64, Float64, float, boolean). A column that is entirely missing (NaN for float, NA otherwise) must remain entirely missing with dtype preserved, for both frame groupby and column-selection groupby.
    item = np.nan if dtype == "float" else pd.NA
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [item] * 8})
    base_df["B"] = base_df["B"].astype(dtype)
    grouped = base_df.groupby("A")

    expected = DataFrame({"B": [item] * 8}, dtype=dtype)
    result = getattr(grouped, method)()
    tm.assert_frame_equal(expected, result)

    result = getattr(grouped["B"], method)().to_frame()
    tm.assert_frame_equal(expected, result)


def test_cummax(dtypes_for_minmax):
    # [FUNCTION DOC]
    # GH 15048: cummax for each dtype of the dtypes_for_minmax fixture on two groups gives running maxima 3,4,4,4 and 2,3,3,3 in the original dtype, equal to the per-group apply of cummax.
    dtype = dtypes_for_minmax[0]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_maxs = [3, 4, 4, 4, 2, 3, 3, 3]

    df = base_df.astype(dtype)

    expected = DataFrame({"B": expected_maxs}).astype(dtype)
    result = df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    tm.assert_frame_equal(result, expected)


def test_cummax_min_value_for_dtype(dtypes_for_minmax):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says min value but the test checks cummax with the maximum value of the dtype.
    # GH 15048: cummax with values equal to the dtype maximum placed in rows 2 and 6: the running maximum remains the dtype maximum from there to the end of each group. Compared with the per-group apply.
    dtype = dtypes_for_minmax[0]
    max_val = dtypes_for_minmax[2]

    # GH 15048
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    expected_maxs = [3, 4, 4, 4, 2, 3, 3, 3]

    df = base_df.astype(dtype)
    df.loc[[2, 6], "B"] = max_val
    expected = DataFrame({"B": expected_maxs}).astype(dtype)
    expected.loc[[2, 3, 6, 7], "B"] = max_val
    result = df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    tm.assert_frame_equal(result, expected)


def test_cummax_nan_in_some_values(dtypes_for_minmax):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtypes_for_minmax fixture is requested but never used in the body.
    # cummax with NaN at alternate positions in a float column: NaN rows remain NaN and the running maximum skips NaN (4, 4 in group one; 3, 3 in group two). Compared with the per-group apply.
    # Test nan in some values
    # Explicit cast to float to avoid implicit cast when setting nan
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    base_df = base_df.astype({"B": "float"})
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 4, np.nan, 3, np.nan, 3]})
    result = base_df.groupby("A").cummax()
    tm.assert_frame_equal(result, expected)
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    tm.assert_frame_equal(result, expected)


def test_cummax_datetime():
    # [FUNCTION DOC]
    # GH 15561: cummax of a datetime column of a single row returns that datetime as a Series named b with the original index.
    # GH 15561
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    result = df.groupby("a")["b"].cummax()
    tm.assert_series_equal(expected, result)


def test_cummax_getattr_series():
    # [FUNCTION DOC]
    # GH 15635: cummax through attribute access on column b gives 2, 1, 2 with name b.
    # GH 15635
    df = DataFrame({"a": [1, 2, 1], "b": [2, 1, 1]})
    result = df.groupby("a").b.cummax()
    expected = Series([2, 1, 2], name="b")
    tm.assert_series_equal(result, expected)


def test_cummax_i8_at_implementation_bound():
    # [FUNCTION DOC]
    # GH#46382: cummax on int64 and datetime columns whose values start at the minimum representable internal timestamp value (the NaT sentinel) plus n; the minimum value used to be treated as the sentinel plus one. Results must equal the input unchanged (the values are increasing).
    # the minimum value used to be treated as NPY_NAT+1 instead of NPY_NAT
    #  for int64 dtype GH#46382
    ser = Series([pd.NaT._value + n for n in range(5)])
    df = DataFrame({"A": 1, "B": ser, "C": ser._values.view("M8[ns]")})
    gb = df.groupby("A")

    res = gb.cummax()
    exp = df[["B", "C"]]
    tm.assert_frame_equal(res, exp)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["float", "Int64", "Float64"])
@pytest.mark.parametrize(
    "groups,expected_data",
    [
        ([1, 1, 1], [1, None, None]),
        ([1, 2, 3], [1, None, 2]),
        ([1, 3, 3], [1, None, None]),
    ],
)
def test_cummin_max_skipna(method, dtype, groups, expected_data):
    # [FUNCTION DOC]
    # GH-34047: parametrized over cummin/cummax, dtype (float, Int64, Float64), group assignments and expected values. With skipna false, once a missing value is seen in a group, the rest of that group's running result is missing. Also asserts the source frame was not mutated.
    # GH-34047
    df = DataFrame({"a": Series([1, None, 2], dtype=dtype)})
    orig = df.copy()
    gb = df.groupby(groups)["a"]

    result = getattr(gb, method)(skipna=False)
    expected = Series(expected_data, dtype=dtype, name="a")

    # check we didn't accidentally alter df
    tm.assert_frame_equal(df, orig)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
def test_cummin_max_skipna_multiple_cols(method):
    # [FUNCTION DOC]
    # Parametrized over cummin/cummax. With skipna false on a two-column selection, NaN in column a makes all later rows of a NaN but must not affect column b (stays 2.0).
    # Ensure missing value in "a" doesn't cause "b" to be nan-filled
    df = DataFrame({"a": [np.nan, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})
    gb = df.groupby([1, 1, 1])[["a", "b"]]

    result = getattr(gb, method)(skipna=False)
    expected = DataFrame({"a": [np.nan, np.nan, np.nan], "b": [2.0, 2.0, 2.0]})

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["cumprod", "cumsum"])
def test_numpy_compat(func):
    # [FUNCTION DOC]
    # See gh-12811: cumprod and cumsum called with extra positional arguments or an unexpected keyword (numpy compatibility signature) must raise an unsupported function call error whose message states numpy operations are not valid with groupby.
    # see gh-12811
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    g = df.groupby("A")

    msg = "numpy operations are not valid with groupby"

    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(foo=1)


@td.skip_if_32bit
@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize(
    "dtype,val", [("UInt64", np.iinfo("uint64").max), ("Int64", 2**53 + 1)]
)
def test_nullable_int_not_cast_as_float(method, dtype, val):
    # [FUNCTION DOC]
    # Skipped on 32-bit platforms. Parametrized over cummin/cummax and nullable UInt64 (max value) or Int64 (2 to the 53 plus 1) values mixed with NA. Calls the method via the transform path; values must not be cast through float and lose precision, and dtype is preserved.
    data = [val, pd.NA]
    df = DataFrame({"grp": [1, 1], "b": data}, dtype=dtype)
    grouped = df.groupby("grp")

    result = grouped.transform(method)
    expected = DataFrame({"b": data}, dtype=dtype)

    tm.assert_frame_equal(result, expected)


def test_cython_api2(as_index):
    # [FUNCTION DOC]
    # GH5614 and GH 5755: cumsum is a transformer and ignores as_index; with either value of the as_index fixture, the grouped cumsum returns a frame of the B and C columns with NaN left in place (not skipped into the sum), using the original row index.
    # this takes the fast apply path

    # cumsum (GH5614)
    # GH 5755 - cumsum is a transformer and should ignore as_index
    df = DataFrame([[1, 2, np.nan], [1, np.nan, 9], [3, 4, 9]], columns=["A", "B", "C"])
    expected = DataFrame([[2, np.nan], [np.nan, 9], [4, 9]], columns=["B", "C"])
    result = df.groupby("A", as_index=as_index).cumsum()
    tm.assert_frame_equal(result, expected)
