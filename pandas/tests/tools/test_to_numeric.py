# [MODULE DOC]
# Tests for the public numeric conversion function (to_numeric): Series, list, array, scalar and DataFrame-column inputs; error handling modes; very large integers; datetime-like, timedelta and period inputs; downcasting to the smallest integer, unsigned or float types including limits and nullable dtypes; nullable string input; and the dtype backend option (numpy-nullable and pyarrow).
# Defines parametrized fixtures (errors mode, signedness, input transform, large integers, whether extra elements are present, and transform/assertion pairs for index, series and array containers). Pyarrow-dependent tests skip when it is missing. No import-time side effects.
import decimal

import numpy as np
from numpy import iinfo
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    Series,
    option_context,
    to_numeric,
)
import pandas._testing as tm


@pytest.fixture(params=[None, "raise", "coerce"])
def errors(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the errors argument values: not given (None), "raise" and "coerce". Returns the current value.
    return request.param


@pytest.fixture(params=[True, False])
def signed(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false, telling tests whether to negate the test value. Returns the current boolean.
    return request.param


@pytest.fixture(params=[lambda x: x, str], ids=["identity", "str"])
def transform(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over a function that returns its input unchanged and the string conversion function (ids identity and str). Returns the current function.
    return request.param


@pytest.fixture(params=[47393996303418497800, 100000000000000000000])
def large_val(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over two integers beyond the 64-bit unsigned range (47393996303418497800 and one hundred quintillion). Returns the current value.
    return request.param


@pytest.fixture(params=[True, False])
def multiple_elts(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false, telling tests whether to add a second, unparseable element to an array. Returns the current boolean.
    return request.param


@pytest.fixture(
    params=[
        (lambda x: Index(x, name="idx"), tm.assert_index_equal),
        (lambda x: Series(x, name="ser"), tm.assert_series_equal),
        (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal),
    ]
)
def transform_assert_equal(request):
    # [FUNCTION DOC]
    # HELPER: fixture yielding (constructor, matching assertion function) pairs for three container kinds: a named Index, a named Series, and a plain NumPy array of the values. Returns the current pair.
    return request.param


@pytest.mark.parametrize(
    "input_kwargs,result_kwargs",
    [
        ({}, {"dtype": np.int64}),
        ({"errors": "coerce", "downcast": "integer"}, {"dtype": np.int8}),
    ],
)
def test_empty(input_kwargs, result_kwargs):
    # [FUNCTION DOC]
    # see gh-16302. Parametrized over (keyword arguments, result dtype). An empty object Series converts to an empty Series with 64-bit integer dtype by default, and to an 8-bit integer dtype when coerce errors and integer downcast are requested.
    # see gh-16302
    ser = Series([], dtype=object)
    result = to_numeric(ser, **input_kwargs)

    expected = Series([], **result_kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("last_val", ["7", 7])
def test_series(last_val, infer_string):
    # [FUNCTION DOC]
    # Parametrized over the final value ("7" or 7) and whether the future string-inference option is on (the on case requires pyarrow). A Series of "1", "-3.14" and the last value converts to float values 1, -3.14 and 7. The option is set only while the Series is built and converted.
    with option_context("future.infer_string", infer_string):
        ser = Series(["1", "-3.14", last_val])
        result = to_numeric(ser)

    expected = Series([1, -3.14, 7])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [1, 3, 4, 5],
        [1.0, 3.0, 4.0, 5.0],
        # Bool is regarded as numeric.
        [True, False, True, True],
    ],
)
def test_series_numeric(data):
    # [FUNCTION DOC]
    # Parametrized over integer, float and boolean data. A Series that is already numeric (booleans count as numeric) with a custom index and name is returned equal to the input.
    ser = Series(data, index=list("ABCD"), name="EFG")

    result = to_numeric(ser)
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "data,msg",
    [
        ([1, -3.14, "apple"], 'Unable to parse string "apple" at position 2'),
        (
            ["orange", 1, -3.14, "apple"],
            'Unable to parse string "orange" at position 0',
        ),
    ],
)
def test_error(data, msg):
    # [FUNCTION DOC]
    # Parametrized over (data, message). With errors set to raise, an unparseable string raises a value error that names the string and the position of the first failure (position 2 for the apple, position 0 for the orange).
    ser = Series(data)

    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, errors="raise")


def test_ignore_error():
    # [FUNCTION DOC]
    # A Series with an unparseable string converted in coerce mode gives floats with a missing value in place of the string.
    # KNOWN ISSUE: the name says ignore error, but the test uses coerce mode; the ignore option is no longer exercised.
    ser = Series([1, -3.14, "apple"])
    result = to_numeric(ser, errors="coerce")

    expected = Series([1, -3.14, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "errors,exp",
    [
        ("raise", 'Unable to parse string "apple" at position 2'),
        # Coerces to float.
        ("coerce", [1.0, 0.0, np.nan]),
    ],
)
def test_bool_handling(errors, exp):
    # [FUNCTION DOC]
    # Parametrized over (errors mode, expected). A Series of two booleans and an unparseable string: raise mode raises a value error naming the string at position 2; coerce mode gives floats 1.0, 0.0 and a missing value.
    ser = Series([True, False, "apple"])

    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        expected = Series(exp)

        tm.assert_series_equal(result, expected)


def test_list():
    # [FUNCTION DOC]
    # A list of numeric strings converts to a NumPy float array with the three values.
    ser = ["1", "-3.14", "7"]
    res = to_numeric(ser)

    expected = np.array([1, -3.14, 7])
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "data,arr_kwargs",
    [
        ([1, 3, 4, 5], {"dtype": np.int64}),
        ([1.0, 3.0, 4.0, 5.0], {}),
        # Boolean is regarded as numeric.
        ([True, False, True, True], {}),
    ],
)
def test_list_numeric(data, arr_kwargs):
    # [FUNCTION DOC]
    # Parametrized over integer, float and boolean lists with the expected array keyword arguments. A list that is already numeric converts to the equivalent NumPy array (64-bit integers for integers; booleans kept as booleans).
    result = to_numeric(data)
    expected = np.array(data, **arr_kwargs)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"dtype": "O"}, {}])
def test_numeric(kwargs):
    # [FUNCTION DOC]
    # Parametrized over an object dtype and the default dtype. A Series of 1, -3.14, 7 converts to a Series equal to the plain float Series.
    data = [1, -3.14, 7]

    ser = Series(data, **kwargs)
    result = to_numeric(ser)

    expected = Series(data)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "columns",
    [
        # One column.
        "a",
        # Multiple columns.
        ["a", "b"],
    ],
)
def test_numeric_df_columns(columns):
    # [FUNCTION DOC]
    # see gh-14827. Parametrized over a single column name or a list of both. Applies the conversion to column a, which mixes float, Decimal values (including infinity) and a numeric string, and optionally column b; the frame becomes all-float with values 1.2, 3.14, infinity, 0.1.
    # see gh-14827
    df = DataFrame(
        {
            "a": [1.2, decimal.Decimal("3.14"), decimal.Decimal("infinity"), "0.1"],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )

    expected = DataFrame({"a": [1.2, 3.14, np.inf, 0.1], "b": [1.0, 2.0, 3.0, 4.0]})
    df[columns] = df[columns].apply(to_numeric)

    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            [[decimal.Decimal("3.14"), 1.0], decimal.Decimal("1.6"), 0.1],
            [[3.14, 1.0], 1.6, 0.1],
        ),
        ([np.array([decimal.Decimal("3.14"), 1.0]), 0.1], [[3.14, 1.0], 0.1]),
    ],
)
def test_numeric_embedded_arr_likes(data, exp_data):
    # [FUNCTION DOC]
    # Parametrized over data containing lists and arrays of Decimal values mixed with scalars. Applying the conversion element-wise to a column converts embedded Decimals to floats while keeping the nested structure.
    # Test to_numeric with embedded lists and arrays
    df = DataFrame({"a": data})
    df["a"] = df["a"].apply(to_numeric)

    expected = DataFrame({"a": exp_data})
    tm.assert_frame_equal(df, expected)


def test_all_nan():
    # [FUNCTION DOC]
    # In coerce mode, a Series of three non-numeric strings becomes all missing floats.
    ser = Series(["a", "b", "c"])
    result = to_numeric(ser, errors="coerce")

    expected = Series([np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_type_check(errors):
    # [FUNCTION DOC]
    # see gh-11776. Uses the errors fixture. Passing a DataFrame raises a type error mentioning a 1-d array requirement, regardless of the errors argument (omitted when None).
    # see gh-11776
    df = DataFrame({"a": [1, -3.14, 7], "b": ["4", "5", "6"]})
    kwargs = {"errors": errors} if errors is not None else {}
    with pytest.raises(TypeError, match="1-d array"):
        to_numeric(df, **kwargs)


@pytest.mark.parametrize("val", [1, 1.1, 20001])
def test_scalar(val, signed, transform):
    # [FUNCTION DOC]
    # Parametrized over 1, 1.1, 20001, the signed and transform fixtures. Converting the (possibly negated, possibly stringified) scalar equals the float value.
    val = -val if signed else val
    assert to_numeric(transform(val)) == float(val)


def test_really_large_scalar(large_val, signed, transform, errors):
    # [FUNCTION DOC]
    # see gh-24910. Uses the large value, sign, transform and errors fixtures. A scalar larger than the 64-bit unsigned range converts to an integer in raise or default mode and to a float in coerce mode (compared approximately).
    # see gh-24910
    kwargs = {"errors": errors} if errors is not None else {}
    val = -large_val if signed else large_val

    val = transform(val)

    expected = float(val) if errors == "coerce" else int(val)
    tm.assert_almost_equal(to_numeric(val, **kwargs), expected)


def test_really_large_in_arr(large_val, signed, transform, multiple_elts, errors):
    # [FUNCTION DOC]
    # see gh-24910. Uses large value, sign, transform, multiple-elements and errors fixtures. Builds an array holding the large value and optionally an extra non-numeric string. In raise or default mode with the extra element, a value error naming the string at position 1 is expected. Otherwise the result is an array: floats (with a missing value for the extra string) in coerce mode; an object array of Python integers (plus the unparsed string) in the non-coerce case; for the single element case the dtype is float when the expected value is float, otherwise object.
    # see gh-24910
    kwargs = {"errors": errors} if errors is not None else {}
    val = -large_val if signed else large_val
    val = transform(val)

    extra_elt = "string"
    arr = [val] + multiple_elts * [extra_elt]

    coercing = errors == "coerce"

    if errors in (None, "raise") and multiple_elts:
        msg = 'Unable to parse string "string" at position 1'

        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result = to_numeric(arr, **kwargs)

        exp_val = float(val) if (coercing) else int(val)
        expected = [exp_val]

        if multiple_elts:
            if coercing:
                expected.append(np.nan)
                exp_dtype = float
            else:
                expected.append(extra_elt)
                exp_dtype = object
        else:
            exp_dtype = float if isinstance(exp_val, float) else object

        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


def test_really_large_in_arr_consistent(large_val, signed, multiple_elts, errors):
    # [FUNCTION DOC]
    # see gh-24910. Even after having to hold a float, later elements that fail to be integer must not be leniently accepted. Builds a list with the large value as a string, optionally preceded by the large value as an integer. Result is a float array in coerce mode and an object array of integers otherwise, compared approximately.
    # see gh-24910
    #
    # Even if we discover that we have to hold float, does not mean
    # we should be lenient on subsequent elements that fail to be integer.
    kwargs = {"errors": errors} if errors is not None else {}
    arr = [str(-large_val if signed else large_val)]

    if multiple_elts:
        arr.insert(0, large_val)

    result = to_numeric(arr, **kwargs)
    expected = [float(i) if errors == "coerce" else int(i) for i in arr]
    exp_dtype = float if errors == "coerce" else object

    tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


@pytest.mark.parametrize(
    "errors,checker",
    [
        ("raise", 'Unable to parse string "fail" at position 0'),
        ("coerce", lambda x: np.isnan(x)),
    ],
)
def test_scalar_fail(errors, checker):
    # [FUNCTION DOC]
    # Parametrized over (errors mode, checker). Scalar "fail" raises a value error naming the string at position 0 in raise mode, and returns a missing value in coerce mode.
    scalar = "fail"

    if isinstance(checker, str):
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        assert checker(to_numeric(scalar, errors=errors))


@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(data, transform_assert_equal):
    # [FUNCTION DOC]
    # Parametrized over an integer list and a float list with missing values, and the container fixture. Already-numeric Index, Series or array input is returned equal to the input.
    transform, assert_equal = transform_assert_equal
    data = transform(data)

    result = to_numeric(data)
    assert_equal(result, data)


@pytest.mark.parametrize(
    "data,exp",
    [
        (["1", "2", "3"], np.array([1, 2, 3], dtype="int64")),
        (["1.5", "2.7", "3.4"], np.array([1.5, 2.7, 3.4])),
    ],
)
def test_str(data, exp, transform_assert_equal):
    # [FUNCTION DOC]
    # Parametrized over (strings, expected array), integer strings give 64-bit integers, decimal strings give floats, for each container type. The result equals the expected values wrapped in the same container.
    transform, assert_equal = transform_assert_equal
    result = to_numeric(transform(data))

    expected = transform(exp)
    assert_equal(result, expected)


def test_datetime_like(tz_naive_fixture, transform_assert_equal):
    # [FUNCTION DOC]
    # Uses the time zone and container fixtures. Converting a three-element datetime range (in any container, naive or aware) returns the underlying integer values (nanosecond-based integer representation) in the same container type.
    transform, assert_equal = transform_assert_equal
    idx = pd.date_range("20130101", periods=3, tz=tz_naive_fixture)

    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)


def test_timedelta(transform_assert_equal):
    # [FUNCTION DOC]
    # Uses the container fixture. Converting a three-element daily timedelta range returns its underlying integers in the same container type.
    transform, assert_equal = transform_assert_equal
    idx = pd.timedelta_range("1 days", periods=3, freq="D")

    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "scalar",
    [
        pd.Timedelta(1, "D"),
        pd.Timestamp("2017-01-01T12"),
        pd.Timestamp("2017-01-01T12", tz="US/Pacific"),
    ],
)
def test_timedelta_timestamp_scalar(scalar):
    # [FUNCTION DOC]
    # GH#59944. Parametrized over a Timedelta, a naive Timestamp and a Pacific-time Timestamp. Converting the scalar gives the same value as converting a one-element Series of it and taking the first element.
    # GH#59944
    result = to_numeric(scalar)
    expected = to_numeric(Series(scalar))[0]
    assert result == expected


def test_period(request, transform_assert_equal):
    # [FUNCTION DOC]
    # Uses the container fixture. An index of monthly periods converts to its underlying integers. For non-Index containers (Series, array) the test marks an expected failure because period dtype is not supported there.
    transform, assert_equal = transform_assert_equal

    idx = pd.period_range("2011-01", periods=3, freq="M", name="")
    inp = transform(idx)

    if not isinstance(inp, Index):
        request.applymarker(
            pytest.mark.xfail(reason="Missing PeriodDtype support in to_numeric")
        )
    result = to_numeric(inp)
    expected = transform(idx.asi8)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "errors,expected",
    [
        ("raise", "Invalid object type at position 0"),
        ("coerce", Series([np.nan, 1.0, np.nan])),
    ],
)
def test_non_hashable(errors, expected):
    # [FUNCTION DOC]
    # see gh-13324. Parametrized over (errors mode, expected). A Series containing a list, a float and a string: raise mode raises a type error "Invalid object type at position 0"; coerce mode gives a missing value for the list and the string and keeps 1.0.
    # see gh-13324
    ser = Series([[10.0, 2], 1.0, "apple"])

    if isinstance(expected, str):
        with pytest.raises(TypeError, match=expected):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, expected)


def test_downcast_invalid_cast():
    # [FUNCTION DOC]
    # see gh-13352. An unknown downcast option raises a value error "invalid downcasting method provided".
    # see gh-13352
    data = ["1", 2, 3]
    invalid_downcast = "unsigned-integer"
    msg = "invalid downcasting method provided"

    with pytest.raises(ValueError, match=msg):
        to_numeric(data, downcast=invalid_downcast)


def test_errors_invalid_value():
    # [FUNCTION DOC]
    # see gh-26466. An unknown errors option raises a value error "invalid error value specified".
    # see gh-26466
    data = ["1", 2, 3]
    invalid_error_value = "invalid"
    msg = "invalid error value specified"

    with pytest.raises(ValueError, match=msg):
        to_numeric(data, errors=invalid_error_value)


@pytest.mark.parametrize(
    "data",
    [
        ["1", 2, 3],
        [1, 2, 3],
        np.array(["1970-01-02", "1970-01-03", "1970-01-04"], dtype="datetime64[D]"),
    ],
)
@pytest.mark.parametrize(
    "kwargs,exp_dtype",
    [
        # Basic function tests.
        ({}, np.int64),
        ({"downcast": None}, np.int64),
        # Support below np.float32 is rare and far between.
        ({"downcast": "float"}, np.dtype(np.float32).char),
        # Basic dtype support.
        ({"downcast": "unsigned"}, np.dtype(np.typecodes["UnsignedInteger"][0])),
    ],
)
def test_downcast_basic(data, kwargs, exp_dtype):
    # [FUNCTION DOC]
    # see gh-13352. Parametrized over inputs (strings and numbers, integers, an array of daily datetimes) and (keyword, expected dtype): default and downcast none give 64-bit integers; downcast float gives the 32-bit float; downcast unsigned gives the smallest unsigned integer type. The result equals an array of 1, 2, 3 of that dtype.
    # see gh-13352
    result = to_numeric(data, **kwargs)
    expected = np.array([1, 2, 3], dtype=exp_dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("signed_downcast", ["integer", "signed"])
@pytest.mark.parametrize(
    "data",
    [
        ["1", 2, 3],
        [1, 2, 3],
        np.array(["1970-01-02", "1970-01-03", "1970-01-04"], dtype="datetime64[D]"),
    ],
)
def test_signed_downcast(data, signed_downcast):
    # [FUNCTION DOC]
    # see gh-13352. Parametrized over the integer and signed options and three inputs. The result equals an array of 1, 2, 3 in the smallest signed integer dtype.
    # see gh-13352
    smallest_int_dtype = np.dtype(np.typecodes["Integer"][0])
    expected = np.array([1, 2, 3], dtype=smallest_int_dtype)

    res = to_numeric(data, downcast=signed_downcast)
    tm.assert_numpy_array_equal(res, expected)


def test_ignore_downcast_neg_to_unsigned():
    # [FUNCTION DOC]
    # Cannot cast to unsigned because a value is negative: an unsigned downcast request on data with -1 returns 64-bit integers.
    # Cannot cast to an unsigned integer
    # because we have a negative number.
    data = ["-1", 2, 3]
    expected = np.array([-1, 2, 3], dtype=np.int64)

    res = to_numeric(data, downcast="unsigned")
    tm.assert_numpy_array_equal(res, expected)


# Warning in 32 bit platforms
@pytest.mark.parametrize("downcast", ["integer", "signed", "unsigned"])
@pytest.mark.parametrize(
    "data,expected",
    [
        (["1.1", 2, 3], np.array([1.1, 2, 3], dtype=np.float64)),
        (
            [10000.0, 20000, 3000, 40000.36, 50000, 50000.00],
            np.array(
                [10000.0, 20000, 3000, 40000.36, 50000, 50000.00], dtype=np.float64
            ),
        ),
    ],
)
def test_ignore_downcast_cannot_convert_float(data, expected, downcast):
    # [FUNCTION DOC]
    # Parametrized over integer, signed and unsigned downcast options and two inputs with non-integer values. The result stays as a 64-bit float array (the downcast is ignored).
    # Cannot cast to an integer (signed or unsigned)
    # because we have a float number.
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "downcast,expected_dtype",
    [("integer", np.int16), ("signed", np.int16), ("unsigned", np.uint16)],
)
def test_downcast_not8bit(downcast, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over (option, dtype). Values 256, 257, 258 do not fit in 8 bits: integer and signed give 16-bit signed, unsigned gives 16-bit unsigned.
    # the smallest integer dtype need not be np.(u)int8
    data = ["256", 257, 258]

    expected = np.array([256, 257, 258], dtype=expected_dtype)
    res = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize(
    "dtype,downcast,min_max",
    [
        ("int8", "integer", [iinfo(np.int8).min, iinfo(np.int8).max]),
        ("int16", "integer", [iinfo(np.int16).min, iinfo(np.int16).max]),
        ("int32", "integer", [iinfo(np.int32).min, iinfo(np.int32).max]),
        ("int64", "integer", [iinfo(np.int64).min, iinfo(np.int64).max]),
        ("uint8", "unsigned", [iinfo(np.uint8).min, iinfo(np.uint8).max]),
        ("uint16", "unsigned", [iinfo(np.uint16).min, iinfo(np.uint16).max]),
        ("uint32", "unsigned", [iinfo(np.uint32).min, iinfo(np.uint32).max]),
        ("uint64", "unsigned", [iinfo(np.uint64).min, iinfo(np.uint64).max]),
        ("int16", "integer", [iinfo(np.int8).min, iinfo(np.int8).max + 1]),
        ("int32", "integer", [iinfo(np.int16).min, iinfo(np.int16).max + 1]),
        ("int64", "integer", [iinfo(np.int32).min, iinfo(np.int32).max + 1]),
        ("int16", "integer", [iinfo(np.int8).min - 1, iinfo(np.int16).max]),
        ("int32", "integer", [iinfo(np.int16).min - 1, iinfo(np.int32).max]),
        ("int64", "integer", [iinfo(np.int32).min - 1, iinfo(np.int64).max]),
        ("uint16", "unsigned", [iinfo(np.uint8).min, iinfo(np.uint8).max + 1]),
        ("uint32", "unsigned", [iinfo(np.uint16).min, iinfo(np.uint16).max + 1]),
        ("uint64", "unsigned", [iinfo(np.uint32).min, iinfo(np.uint32).max + 1]),
    ],
)
def test_downcast_limits(dtype, downcast, min_max):
    # [FUNCTION DOC]
    # see gh-14404. Parametrized over (target dtype, downcast option, [min, max] values). A Series of the two limit values downcasts to exactly the expected dtype, testing each integer width's bounds and values just beyond the narrower type's bounds.
    # see gh-14404: test the limits of each downcast.
    series = to_numeric(Series(min_max), downcast=downcast)
    assert series.dtype == dtype


def test_downcast_float64_to_float32():
    # [FUNCTION DOC]
    # GH-43693. A float64 Series with a value above the 32-bit exact-integer limit (16777217), the largest float64 and a missing value, downcast to float, keeps its float64 dtype.
    # GH-43693: Check float64 preservation when >= 16,777,217
    series = Series([16777217.0, np.finfo(np.float64).max, np.nan], dtype=np.float64)
    result = to_numeric(series, downcast="float")

    assert series.dtype == result.dtype


def test_downcast_uint64():
    # [FUNCTION DOC]
    # see gh-14422. A Series with 0 and 9223372036854775808 downcast to unsigned gives unsigned 64-bit dtype.
    # see gh-14422:
    # BUG: to_numeric doesn't work uint64 numbers
    ser = Series([0, 9223372036854775808])
    result = to_numeric(ser, downcast="unsigned")
    expected = Series([0, 9223372036854775808], dtype=np.uint64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            [200, 300, "", "NaN", 30000000000000000000],
            [200, 300, np.nan, np.nan, 30000000000000000000],
        ),
        (
            ["12345678901234567890", "1234567890", "ITEM"],
            [12345678901234567890, 1234567890, np.nan],
        ),
    ],
)
def test_coerce_uint64_conflict(data, exp_data):
    # [FUNCTION DOC]
    # see gh-17007 and gh-17125. Parametrized over two datasets mixing very large integers, empty and invalid strings. In coerce mode the result is float despite the conflict between unsigned 64-bit values and missing values (which would normally force object dtype).
    # see gh-17007 and gh-17125
    #
    # Still returns float despite the uint64-nan conflict,
    # which would normally force the casting to object.
    result = to_numeric(Series(data), errors="coerce")
    expected = Series(exp_data, dtype=float)
    tm.assert_series_equal(result, expected)


def test_non_coerce_uint64_conflict():
    # [FUNCTION DOC]
    # see gh-17007 and gh-17125. For completeness: the same kind of data without coerce raises a value error "Unable to parse string".
    # see gh-17007 and gh-17125
    #
    # For completeness.
    ser = Series(["12345678901234567890", "1234567890", "ITEM"])

    with pytest.raises(ValueError, match="Unable to parse string"):
        to_numeric(ser, errors="raise")


@pytest.mark.parametrize("dc1", ["integer", "float", "unsigned"])
@pytest.mark.parametrize("dc2", ["integer", "float", "unsigned"])
def test_downcast_empty(dc1, dc2):
    # [FUNCTION DOC]
    # GH32493. Parametrized over pairs of downcast options (integer, float, unsigned) applied to an empty list; results are equal arrays regardless of option (dtype not checked).
    # GH32493

    tm.assert_numpy_array_equal(
        to_numeric([], downcast=dc1),
        to_numeric([], downcast=dc2),
        check_dtype=False,
    )


def test_failure_to_convert_uint64_string_to_NaN():
    # [FUNCTION DOC]
    # GH 32394. The string "uint64" in coerce mode gives a missing value as a scalar and within a Series (result is the float Series 32, 64, missing).
    # GH 32394
    result = to_numeric("uint64", errors="coerce")
    assert np.isnan(result)

    ser = Series([32, 64, np.nan])
    result = to_numeric(Series(["32", "64", "uint64"]), errors="coerce")
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "strrep",
    [
        "243.164",
        "245.968",
        "249.585",
        "259.745",
        "265.742",
        "272.567",
        "279.196",
        "280.366",
        "275.034",
        "271.351",
        "272.889",
        "270.627",
        "280.828",
        "290.383",
        "308.153",
        "319.945",
        "336.0",
        "344.09",
        "351.385",
        "356.178",
        "359.82",
        "361.03",
        "367.701",
        "380.812",
        "387.98",
        "391.749",
        "391.171",
        "385.97",
        "385.345",
        "386.121",
        "390.996",
        "399.734",
        "413.073",
        "421.532",
        "430.221",
        "437.092",
        "439.746",
        "446.01",
        "451.191",
        "460.463",
        "469.779",
        "472.025",
        "479.49",
        "474.864",
        "467.54",
        "471.978",
    ],
)
def test_precision_float_conversion(strrep):
    # [FUNCTION DOC]
    # GH 31364. Parametrized over many decimal strings. Converting the string gives exactly the value of the standard float conversion (no precision difference).
    # GH 31364
    result = to_numeric(strrep)

    assert result == float(strrep)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["1", "2", None], Series([1, 2, pd.NA], dtype="Int64")),
        (["1", "2", "3"], Series([1, 2, 3], dtype="Int64")),
        (["1", "2", 3], Series([1, 2, 3], dtype="Int64")),
        (["1", "2", 3.5], Series([1, 2, 3.5], dtype="Float64")),
        (["1", None, 3.5], Series([1, pd.NA, 3.5], dtype="Float64")),
        (["1", "2", "3.5"], Series([1, 2, 3.5], dtype="Float64")),
    ],
)
def test_to_numeric_from_nullable_string(values, nullable_string_dtype, expected):
    # [FUNCTION DOC]
    # Issue 37262. Parametrized over (values, expected Series) and the nullable string dtype fixture. A nullable-string Series converts to a nullable Int64 Series when all values are integers (with missing allowed), and to nullable Float64 when any value is fractional.
    # https://github.com/pandas-dev/pandas/issues/37262
    s = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(s)
    tm.assert_series_equal(result, expected)


def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype):
    # [FUNCTION DOC]
    # GH#52146. Uses the nullable string dtype fixture. A Series with "a" and "1" in coerce mode gives a nullable Int64 Series with a missing value and 1.
    # GH#52146
    values = ["a", "1"]
    ser = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(ser, errors="coerce")
    expected = Series([pd.NA, 1], dtype="Int64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, input_dtype, downcast, expected_dtype",
    (
        ([1, 1], "Int64", "integer", "Int8"),
        ([1.0, pd.NA], "Float64", "integer", "Int8"),
        ([1.0, 1.1], "Float64", "integer", "Float64"),
        ([1, pd.NA], "Int64", "integer", "Int8"),
        ([450, 300], "Int64", "integer", "Int16"),
        ([1, 1], "Float64", "integer", "Int8"),
        ([np.iinfo(np.int64).max - 1, 1], "Int64", "integer", "Int64"),
        ([1, 1], "Int64", "signed", "Int8"),
        ([1.0, 1.0], "Float32", "signed", "Int8"),
        ([1.0, 1.1], "Float64", "signed", "Float64"),
        ([1, pd.NA], "Int64", "signed", "Int8"),
        ([450, -300], "Int64", "signed", "Int16"),
        ([np.iinfo(np.uint64).max - 1, 1], "UInt64", "signed", "UInt64"),
        ([1, 1], "Int64", "unsigned", "UInt8"),
        ([1.0, 1.0], "Float32", "unsigned", "UInt8"),
        ([1.0, 1.1], "Float64", "unsigned", "Float64"),
        ([1, pd.NA], "Int64", "unsigned", "UInt8"),
        ([450, -300], "Int64", "unsigned", "Int64"),
        ([-1, -1], "Int32", "unsigned", "Int32"),
        ([1, 1], "Float64", "float", "Float32"),
        ([1, 1.1], "Float64", "float", "Float32"),
        ([1, 1], "Float32", "float", "Float32"),
        ([1, 1.1], "Float32", "float", "Float32"),
    ),
)
def test_downcast_nullable_numeric(data, input_dtype, downcast, expected_dtype):
    # [FUNCTION DOC]
    # Parametrized over (data, input nullable dtype, downcast option, expected nullable dtype). Builds an extension array and downcasts: integer/signed go to the smallest nullable signed type that fits (or stay if floats have fractional values or values exceed range); unsigned goes to the smallest unsigned or stays if a negative exists; float goes to 32-bit nullable float. Compares as extension arrays.
    arr = pd.array(data, dtype=input_dtype)
    result = to_numeric(arr, downcast=downcast)
    expected = pd.array(data, dtype=expected_dtype)
    tm.assert_extension_array_equal(result, expected)


def test_downcast_nullable_mask_is_copied():
    # [FUNCTION DOC]
    # GH38974. Downcasting a nullable integer array with a missing value gives an Int8 array; setting an element of the input array to missing afterwards must not change the previously returned result (the mask is copied).
    # GH38974

    arr = pd.array([1, 2, pd.NA], dtype="Int64")

    result = to_numeric(arr, downcast="integer")
    expected = pd.array([1, 2, pd.NA], dtype="Int8")
    tm.assert_extension_array_equal(result, expected)

    arr[1] = pd.NA  # should not modify result
    tm.assert_extension_array_equal(result, expected)


def test_to_numeric_scientific_notation():
    # [FUNCTION DOC]
    # GH 15898. The string "1.7e+308" converts to the equal float64.
    # GH 15898
    result = to_numeric("1.7e+308")
    expected = np.float64(1.7e308)
    assert result == expected


@pytest.mark.parametrize("val", [9876543210.0, 2.0**128])
def test_to_numeric_large_float_not_downcast_to_float_32(val):
    # [FUNCTION DOC]
    # GH 19729. Parametrized over 9876543210.0 and 2 to the 128th power. A float Series downcast to float keeps its original float64 dtype and values because float32 cannot represent them exactly.
    # GH 19729
    expected = Series([val])
    result = to_numeric(expected, downcast="float")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "val, dtype", [(1, "Int64"), (1.5, "Float64"), (True, "boolean")]
)
def test_to_numeric_dtype_backend(val, dtype):
    # [FUNCTION DOC]
    # GH#50505. Parametrized over (value, expected nullable dtype): 1 gives Int64, 1.5 gives Float64, True gives boolean. An object Series converted with the numpy-nullable backend gives a Series of that dtype.
    # GH#50505
    ser = Series([val], dtype=object)
    result = to_numeric(ser, dtype_backend="numpy_nullable")
    expected = Series([val], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "val, dtype",
    [
        (1, "Int64"),
        (1.5, "Float64"),
        (True, "boolean"),
        (1, "int64[pyarrow]"),
        (1.5, "float64[pyarrow]"),
        (True, "bool[pyarrow]"),
    ],
)
def test_to_numeric_dtype_backend_na(val, dtype):
    # [FUNCTION DOC]
    # GH#50505. Parametrized over (value, dtype) for numpy-nullable and pyarrow types (pyarrow cases skip if missing). An object Series with the value and None converts with the matching backend to the expected dtype with a missing value.
    # GH#50505
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
        dtype_backend = "pyarrow"
    else:
        dtype_backend = "numpy_nullable"
    ser = Series([val, None], dtype=object)
    result = to_numeric(ser, dtype_backend=dtype_backend)
    expected = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "val, dtype, downcast",
    [
        (1, "Int8", "integer"),
        (1.5, "Float32", "float"),
        (1, "Int8", "signed"),
        (1, "int8[pyarrow]", "integer"),
        (1.5, "float[pyarrow]", "float"),
        (1, "int8[pyarrow]", "signed"),
    ],
)
def test_to_numeric_dtype_backend_downcasting(val, dtype, downcast):
    # [FUNCTION DOC]
    # GH#50505. Parametrized over (value, dtype, downcast option) for nullable and pyarrow backends. An object Series with the value and None, converted with the backend and the downcast, has the smallest suitable dtype (8-bit integer or 32-bit float) with a missing value.
    # GH#50505
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
        dtype_backend = "pyarrow"
    else:
        dtype_backend = "numpy_nullable"
    ser = Series([val, None], dtype=object)
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast=downcast)
    expected = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "smaller, dtype_backend",
    [["UInt8", "numpy_nullable"], ["uint8[pyarrow]", "pyarrow"]],
)
def test_to_numeric_dtype_backend_downcasting_uint(smaller, dtype_backend):
    # [FUNCTION DOC]
    # GH#50505. Parametrized over (smaller dtype, backend). A nullable UInt64 Series with a missing value downcast to unsigned using the backend gives an 8-bit unsigned dtype of the corresponding family (pyarrow case skips if missing).
    # GH#50505
    if dtype_backend == "pyarrow":
        pytest.importorskip("pyarrow")
    ser = Series([1, pd.NA], dtype="UInt64")
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast="unsigned")
    expected = Series([1, pd.NA], dtype=smaller)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        "UInt64",
        "Float64",
        "boolean",
        "int64[pyarrow]",
        "uint64[pyarrow]",
        "float64[pyarrow]",
        "bool[pyarrow]",
    ],
)
def test_to_numeric_dtype_backend_already_nullable(dtype):
    # [FUNCTION DOC]
    # GH#50505. Parametrized over nullable and pyarrow-backed dtypes (pyarrow ones skip if missing). A Series already of such a dtype converted with the numpy-nullable backend is returned with the same dtype and values.
    # GH#50505
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    ser = Series([1, pd.NA], dtype=dtype)
    result = to_numeric(ser, dtype_backend="numpy_nullable")
    expected = Series([1, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_to_numeric_dtype_backend_error(dtype_backend):
    # [FUNCTION DOC]
    # GH#50505. Uses the dtype backend fixture. A Series of non-numeric strings and an empty string raises a value error "Unable to parse string" in default mode, and in coerce mode gives all-missing values with the double-precision pyarrow type or nullable Float64 depending on the backend.
    # KNOWN ISSUE: an expected copy of the input is created at the top and then overwritten before use, so the first assignment is dead code.
    # GH#50505
    ser = Series(["a", "b", ""])
    expected = ser.copy()
    with pytest.raises(ValueError, match="Unable to parse string"):
        to_numeric(ser, dtype_backend=dtype_backend)

    result = to_numeric(ser, dtype_backend=dtype_backend, errors="coerce")
    if dtype_backend == "pyarrow":
        dtype = "double[pyarrow]"
    else:
        dtype = "Float64"
    expected = Series([pd.NA, pd.NA, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_invalid_dtype_backend():
    # [FUNCTION DOC]
    # An invalid backend name raises a value error "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed".
    ser = Series([1, 2, 3])
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, dtype_backend="numpy")


def test_coerce_pyarrow_backend():
    # [FUNCTION DOC]
    # GH 52588. Skipped without pyarrow. A pyarrow-string Series "1", "2", "x" in coerce mode with the pyarrow backend gives a pyarrow 64-bit integer Series with a missing value for the bad entry.
    # GH 52588
    pa = pytest.importorskip("pyarrow")
    ser = Series(list("12x"), dtype=ArrowDtype(pa.string()))
    result = to_numeric(ser, errors="coerce", dtype_backend="pyarrow")
    expected = Series([1, 2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)
