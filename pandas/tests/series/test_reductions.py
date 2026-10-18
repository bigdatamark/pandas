# [MODULE DOC]
# Module-level tests for Series reductions: string min/max, the mode for extension and nullable dtypes, timedelta reductions with missing values or overflow, product, validation of unsupported NumPy-compat keywords, and errors when taking mean or median of strings.
# No import-time side effects.

import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


@pytest.mark.parametrize("operation, expected", [("min", "a"), ("max", "b")])
def test_reductions_series_strings(operation, expected):
    # [FUNCTION DOC]
    # Verifies (issue 31746) that the minimum and maximum of a string-dtype Series with values a and b are a and b. Parametrized over the method name and expected value.
    # GH#31746
    ser = Series(["a", "b"], dtype="string")
    res_operation_serie = getattr(ser, operation)()
    assert res_operation_serie == expected


@pytest.mark.parametrize("as_period", [True, False])
def test_mode_extension_dtype(as_period):
    # [FUNCTION DOC]
    # Verifies (issue 41927) that the mode of four distinct daily timestamps keeps the extension dtype, whether the series is converted to period (daily) or localized to the US/Central zone: the result equals the series (all values are modes) and has the same dtype.
    # GH#41927 preserve dt64tz dtype
    ser = Series([pd.Timestamp(1979, 4, n) for n in range(1, 5)])

    if as_period:
        ser = ser.dt.to_period("D")
    else:
        ser = ser.dt.tz_localize("US/Central")

    res = ser.mode()
    assert res.dtype == ser.dtype
    tm.assert_series_equal(res, ser)


def test_mode_nullable_dtype(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 55340) the mode of a nullable numeric Series with values one, three, two, missing, three, two, missing, over all nullable numeric dtypes from a fixture.
    # With missing values kept, the modes are two, three and missing; with them dropped, two and three.
    # The test then assigns missing to label minus one; modes are re-checked: dropping missing gives two and three, keeping missing gives only the missing value.
    # KNOWN ISSUE: the assignment by the label minus one targets a label, not the last position; the integer index has no such label, so a new missing entry is appended instead of overwriting the last element (which was already missing). This is what makes missing the unique mode.
    ser = Series([1, 3, 2, pd.NA, 3, 2, pd.NA], dtype=any_numeric_ea_dtype)
    result = ser.mode(dropna=False)
    expected = Series([2, 3, pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    ser[-1] = pd.NA

    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.mode(dropna=False)
    expected = Series([pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)


def test_mode_nullable_dtype_edge_case(any_numeric_ea_dtype):
    # [FUNCTION DOC]
    # Verifies mode edge cases for nullable numeric dtypes with missing values kept (issue 58926): four series, namely values with one repeated and no missing (mode is the repeated value), a repeated value with a single missing (the repeated value only), a single value with a repeated missing (missing only), and a tie between a repeated value and repeated missing (both, value first).
    # GH##58926
    ser = Series([1, 2, 3, 1], dtype=any_numeric_ea_dtype)
    result = ser.mode(dropna=False)
    expected = Series([1], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    ser2 = Series([1, 1, 2, 3, pd.NA], dtype=any_numeric_ea_dtype)
    result = ser2.mode(dropna=False)
    expected = Series([1], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    ser3 = Series([1, pd.NA, pd.NA], dtype=any_numeric_ea_dtype)
    result = ser3.mode(dropna=False)
    expected = Series([pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    ser4 = Series([1, 1, pd.NA, pd.NA], dtype=any_numeric_ea_dtype)
    result = ser4.mode(dropna=False)
    expected = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)


def test_mode_infer_string():
    # [FUNCTION DOC]
    # Verifies (issue 56183) that under the future infer-string option the mode of an object Series of strings is computed and still has object dtype with values a and b. Skipped if pyarrow is missing.
    # GH#56183
    pytest.importorskip("pyarrow")
    ser = Series(["a", "b"], dtype=object)
    with pd.option_context("future.infer_string", True):
        result = ser.mode()
    expected = Series(["a", "b"], dtype=object)
    tm.assert_series_equal(result, expected)


def test_reductions_td64_with_nat():
    # [FUNCTION DOC]
    # Verifies (issue 8617) that the median, minimum and maximum of a timedelta Series holding zero and NaT all equal the first element (the zero timedelta), skipping NaT.
    # GH#8617
    ser = Series([0, pd.NaT], dtype="m8[ns]")
    exp = ser[0]
    assert ser.median() == exp
    assert ser.min() == exp
    assert ser.max() == exp


def test_td64_sum_empty(skipna):
    # [FUNCTION DOC]
    # Verifies (issue 37151) that summing an empty timedelta Series gives a Timedelta of zero for both settings of skipna (parametrized by a fixture).
    # GH#37151
    ser = Series([], dtype="timedelta64[ns]")

    result = ser.sum(skipna=skipna)
    assert isinstance(result, pd.Timedelta)
    assert result == pd.Timedelta(0)


def test_td64_summation_overflow():
    # [FUNCTION DOC]
    # Verifies (issue 9442) timedelta summation near overflow on a hundred-thousand-element hourly series with the first value nudged by a second and a millisecond. The mean of offsets from the minimum matches a manually computed mean to float precision (compared after scaling down). The sum of all offsets raises a value error about overflow in timedelta operation, as does the sum of the first ten thousand; the sum of the first thousand succeeds.
    # GH#9442
    ser = Series(pd.date_range("20130101", periods=100000, freq="h", unit="ns"))
    ser[0] += pd.Timedelta("1s 1ms")

    # mean
    result = (ser - ser.min()).mean()
    expected = pd.Timedelta((pd.TimedeltaIndex(ser - ser.min()).asi8 / len(ser)).sum())

    # the computation is converted to float so
    # might be some loss of precision
    assert np.allclose(result._value / 1000, expected._value / 1000)

    # sum
    msg = "overflow in timedelta operation"
    with pytest.raises(ValueError, match=msg):
        (ser - ser.min()).sum()

    s1 = ser[0:10000]
    with pytest.raises(ValueError, match=msg):
        (s1 - s1.min()).sum()
    s2 = ser[0:1000]
    (s2 - s2.min()).sum()


def test_prod_numpy16_bug():
    # [FUNCTION DOC]
    # Verifies that the product of a three-element float Series is a scalar and not a Series (a regression for a NumPy-version bug).
    ser = Series([1.0, 1.0, 1.0], index=range(3))
    result = ser.prod()

    assert not isinstance(result, Series)


@pytest.mark.parametrize("func", [np.any, np.all])
@pytest.mark.parametrize("kwargs", [{"keepdims": True}, {"out": object()}])
def test_validate_any_all_out_keepdims_raises(kwargs, func):
    # [FUNCTION DOC]
    # Verifies that the NumPy any and all functions applied to a Series raise a value error stating that the keepdims or out parameter is not supported by the library implementation. Parametrized over the function and the offending keyword argument (keepdims true, or an arbitrary object as out); the expected message embeds the parameter and function names.
    ser = Series([1, 2])
    param = next(iter(kwargs))
    name = func.__name__

    msg = (
        f"the '{param}' parameter is not "
        "supported in the pandas "
        rf"implementation of {name}\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        func(ser, **kwargs)


def test_validate_sum_initial():
    # [FUNCTION DOC]
    # Verifies that the NumPy sum applied to a Series with the initial keyword raises a value error that the initial parameter is unsupported in the implementation of sum.
    ser = Series([1, 2])
    msg = (
        r"the 'initial' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, initial=10)


def test_validate_median_initial():
    # [FUNCTION DOC]
    # Verifies that the median method with the overwrite-input keyword raises a value error that the parameter is unsupported in the implementation of median (the method is used because the NumPy median does not dispatch to it).
    # KNOWN ISSUE: the test name says initial but the keyword being tested is overwrite-input.
    ser = Series([1, 2])
    msg = (
        r"the 'overwrite_input' parameter is not "
        r"supported in the pandas "
        r"implementation of median\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        # It seems like np.median doesn't dispatch, so we use the
        # method instead of the ufunc.
        ser.median(overwrite_input=True)


def test_validate_stat_keepdims():
    # [FUNCTION DOC]
    # Verifies that the NumPy sum applied to a Series with the keepdims keyword raises a value error that keepdims is unsupported in the implementation of sum.
    ser = Series([1, 2])
    msg = (
        r"the 'keepdims' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, keepdims=True)


def test_mean_with_convertible_string_raises():
    # [FUNCTION DOC]
    # Verifies (issue 44008) that the sum of a Series of numeric-looking strings concatenates them, while the mean of the Series and of the equivalent one-column frame raise a type error (message variants about failing to convert, not supporting the operation, or inability to perform).
    # GH#44008
    ser = Series(["1", "2"])
    assert ser.sum() == "12"

    msg = "Could not convert string '12' to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        ser.mean()

    df = ser.to_frame()
    msg = r"Could not convert \['12'\] to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        df.mean()


def test_mean_dont_convert_j_to_complex():
    # [FUNCTION DOC]
    # Verifies (issue 36703) that a string "J" is not interpreted as a complex number when averaging: mean and aggregate-mean on a frame, mean on the string column, and the NumPy mean of its string-extension array all raise type errors with variant messages.
    # GH#36703
    df = pd.DataFrame([{"db": "J", "numeric": 123}])
    msg = r"Could not convert \['J'\] to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        df.mean()

    with pytest.raises(TypeError, match=msg):
        df.agg("mean")

    msg = "Could not convert string 'J' to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        df["db"].mean()
    msg = "Could not convert string 'J' to numeric|ufunc 'divide'|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        np.mean(df["db"].astype("string").array)


def test_median_with_convertible_string_raises():
    # [FUNCTION DOC]
    # Verifies (issue 34671) that the median of a Series of numeric-looking strings, and of its one-column frame, raises a type error (conversion failure, unsupported, or cannot perform) and does not return a number.
    # GH#34671 this _could_ return a string "2", but definitely not float 2.0
    msg = r"Cannot convert \['1' '2' '3'\] to numeric|does not support|Cannot perform"
    ser = Series(["1", "2", "3"])
    with pytest.raises(TypeError, match=msg):
        ser.median()

    msg = (
        r"Cannot convert \[\['1' '2' '3'\]\] to numeric|does not support|Cannot perform"
    )
    df = ser.to_frame()
    with pytest.raises(TypeError, match=msg):
        df.median()
