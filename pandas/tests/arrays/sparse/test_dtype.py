# [MODULE DOC]
# Tests for SparseDtype: fill value inference, construction from another sparse dtype, equality and hashing (including NaN), string
# parsing and construction, numeric flag, dtype updating, error cases and repr. Import-time side effects: a module-level block, run with a deprecation warning
# filter (about arbitrary scalar fill values) active, builds the list of dtype pairs used to parametrize the inequality test.

import re
import warnings

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype


@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        ("int", 0),
        ("float", np.nan),
        ("bool", False),
        ("object", np.nan),
        ("datetime64[ns]", np.datetime64("NaT", "ns")),
        ("timedelta64[ns]", np.timedelta64("NaT", "ns")),
    ],
)
def test_inferred_dtype(dtype, fill_value):
    # [FUNCTION DOC]
    # Parametrized over dtype and its expected inferred fill value (zero, NaN, false, NaN, NaT for datetime, NaT for timedelta). Checks value and, for missing values, the exact missing-value type.
    sparse_dtype = SparseDtype(dtype)
    result = sparse_dtype.fill_value
    if pd.isna(fill_value):
        assert pd.isna(result) and type(result) == type(fill_value)
    else:
        assert result == fill_value


def test_from_sparse_dtype():
    # [FUNCTION DOC]
    # Constructing a sparse dtype from another keeps its fill value.
    dtype = SparseDtype("float", 0)
    result = SparseDtype(dtype)
    assert result.fill_value == 0


def test_from_sparse_dtype_fill_value():
    # [FUNCTION DOC]
    # An explicit fill value overrides the one in the source sparse dtype.
    dtype = SparseDtype("int", 1)
    result = SparseDtype(dtype, fill_value=2)
    expected = SparseDtype("int", 2)
    assert result == expected


@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        ("int", None),
        ("float", None),
        ("bool", None),
        ("object", None),
        ("datetime64[ns]", None),
        ("timedelta64[ns]", None),
        ("int", np.nan),
        ("float", 0),
    ],
)
def test_equal(dtype, fill_value):
    # [FUNCTION DOC]
    # Parametrized over dtypes and fill values (including none and mismatched NaN or zero). Two dtypes built identically compare equal in both directions.
    a = SparseDtype(dtype, fill_value)
    b = SparseDtype(dtype, fill_value)
    assert a == b
    assert b == a


def test_nans_equal():
    # [FUNCTION DOC]
    # Two NaN fill values (python and numpy) are equal.
    a = SparseDtype(float, float("nan"))
    b = SparseDtype(float, np.nan)
    assert a == b
    assert b == a


def test_nans_not_equal():
    # [FUNCTION DOC]
    # A zero fill value and NA fill value are not equal.
    # GH 54770
    a = SparseDtype(float, 0)
    b = SparseDtype(float, pd.NA)
    assert a != b
    assert b != a


with warnings.catch_warnings():
    msg = "Allowing arbitrary scalar fill_value in SparseDtype is deprecated"
    warnings.filterwarnings("ignore", msg, category=FutureWarning)

    tups = [
        (SparseDtype("float64"), SparseDtype("float32")),
        (SparseDtype("float64"), SparseDtype("float64", 0)),
        (SparseDtype("float64"), SparseDtype("datetime64[ns]", np.nan)),
        (SparseDtype("float64"), np.dtype("float64")),
    ]


@pytest.mark.parametrize(
    "a, b",
    tups,
)
def test_not_equal(a, b):
    # [FUNCTION DOC]
    # Parametrized over module-level dtype pairs (different subtype, fill value, datetime vs float, and a plain numpy dtype). The dtypes compare unequal.
    assert a != b


def test_construct_from_string_raises():
    # [FUNCTION DOC]
    # An unknown string raises a type error that it cannot construct a SparseDtype from it.
    with pytest.raises(
        TypeError, match="Cannot construct a 'SparseDtype' from 'not a dtype'"
    ):
        SparseDtype.construct_from_string("not a dtype")


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (int, True),
        (float, True),
        (bool, True),
        (object, False),
        (str, False),
    ],
)
def test_is_numeric(dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over integer, float, bool (true) and object, string (false). The private numeric flag reflects the subtype.
    assert SparseDtype(dtype)._is_numeric is expected


def test_str_uses_object():
    # [FUNCTION DOC]
    # A string subtype is stored as object dtype.
    result = SparseDtype(str).subtype
    assert result == np.dtype("object")


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[float64]", SparseDtype(np.dtype("float64"))),
        ("Sparse[float32]", SparseDtype(np.dtype("float32"))),
        ("Sparse[int]", SparseDtype(np.dtype("int"))),
        ("Sparse[str]", SparseDtype(np.dtype("str"))),
        ("Sparse[datetime64[ns]]", SparseDtype(np.dtype("datetime64[ns]"))),
        ("Sparse", SparseDtype(np.dtype("float"), np.nan)),
    ],
)
def test_construct_from_string(string, expected):
    # [FUNCTION DOC]
    # Parametrized over strings with expected dtypes: float, int, str, datetime subtypes, and the bare name giving float with NaN fill.
    result = SparseDtype.construct_from_string(string)
    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (SparseDtype(float, 0.0), SparseDtype(np.dtype("float"), 0.0), True),
        (SparseDtype(int, 0), SparseDtype(int, 0), True),
        (SparseDtype(float, float("nan")), SparseDtype(float, np.nan), True),
        (SparseDtype(float, 0), SparseDtype(float, np.nan), False),
        (SparseDtype(int, 0.0), SparseDtype(float, 0.0), False),
    ],
)
def test_hash_equal(a, b, expected):
    # [FUNCTION DOC]
    # Parametrized over pairs and expected result. Equality and hash equality agree: equal for same subtype and fill (including NaN), unequal for different fill or subtype.
    result = a == b
    assert result is expected

    result = hash(a) == hash(b)
    assert result is expected


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[int]", "int"),
        ("Sparse[int, 0]", "int"),
        ("Sparse[int64]", "int64"),
        ("Sparse[int64, 0]", "int64"),
        ("Sparse[datetime64[ns], 0]", "datetime64[ns]"),
    ],
)
def test_parse_subtype(string, expected):
    # [FUNCTION DOC]
    # Parametrized over strings with and without fill value. The private subtype parser returns the subtype part of the string.
    subtype, _ = SparseDtype._parse_subtype(string)
    assert subtype == expected


@pytest.mark.parametrize(
    "string", ["Sparse[int, 1]", "Sparse[float, 0.0]", "Sparse[bool, True]"]
)
def test_construct_from_string_fill_value_raises(string):
    # [FUNCTION DOC]
    # A string that includes a fill value (for int, float or bool) raises a type error that the fill value in the string is not supported.
    with pytest.raises(TypeError, match="fill_value in the string is not"):
        SparseDtype.construct_from_string(string)


@pytest.mark.parametrize(
    "original, dtype, expected",
    [
        (SparseDtype(int, 0), float, SparseDtype(float, 0.0)),
        (SparseDtype(int, 1), float, SparseDtype(float, 1.0)),
        (SparseDtype(int, 1), np.str_, SparseDtype(object, "1")),
        (SparseDtype(float, 1.5), int, SparseDtype(int, 1)),
    ],
)
def test_update_dtype(original, dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over (original, new subtype, expected). Updating the subtype converts the fill value to the new type (including to string and truncation to integer).
    result = original.update_dtype(dtype)
    assert result == expected


@pytest.mark.parametrize(
    "original, dtype, expected_error_msg",
    [
        (
            SparseDtype(float, np.nan),
            int,
            re.escape("Cannot convert non-finite values (NA or inf) to integer"),
        ),
        (
            SparseDtype(str, "abc"),
            int,
            r"invalid literal for int\(\) with base 10: ('abc'|np\.str_\('abc'\))",
        ),
    ],
)
def test_update_dtype_raises(original, dtype, expected_error_msg):
    # [FUNCTION DOC]
    # Updating NaN fill to integer, or a non-numeric string fill to integer, raises a value error with the corresponding message.
    with pytest.raises(ValueError, match=expected_error_msg):
        original.update_dtype(dtype)


def test_repr():
    # [FUNCTION DOC]
    # String form includes the subtype and the fill value, quoted when a string.
    # GH-34352
    result = str(SparseDtype("int64", fill_value=0))
    expected = "Sparse[int64, 0]"
    assert result == expected

    result = str(SparseDtype(object, fill_value="0"))
    expected = "Sparse[object, '0']"
    assert result == expected


def test_sparse_dtype_subtype_must_be_numpy_dtype():
    # [FUNCTION DOC]
    # A categorical subtype raises a type error that the subtype must be a numpy dtype.
    # GH#53160
    msg = "SparseDtype subtype must be a numpy dtype"
    with pytest.raises(TypeError, match=msg):
        SparseDtype("category", fill_value="c")
