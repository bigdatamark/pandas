# [MODULE DOC]
# Tests for the missing-value singleton NA (pandas' scalar for missing data in nullable dtypes): singleton identity, text formatting,
# truthiness, hashing, arithmetic, comparison, power special cases, unary and logical operators, interaction with numpy arrays and universal functions,
# scalar and missing-value checks, and pickling. The singleton is imported from the internal missing-values module; fixtures supply arithmetic functions, comparison operators and a temporary file.

from datetime import (
    date,
    time,
    timedelta,
)
import pickle

import numpy as np
import pytest

from pandas._libs.missing import NA

from pandas.core.dtypes.common import is_scalar

import pandas as pd
import pandas._testing as tm


def test_singleton():
    # [FUNCTION DOC]
    # The singleton type always yields the same object: the constant is itself, and calling its type returns the same instance.
    assert NA is NA
    new_NA = type(NA)()
    assert new_NA is NA


def test_repr():
    # [FUNCTION DOC]
    # Both the detailed representation and the string form of NA are the text with angle brackets around NA.
    assert repr(NA) == "<NA>"
    assert str(NA) == "<NA>"


def test_format():
    # [FUNCTION DOC]
    # GH-34740. Formatting NA with no spec, a width spec, or an arbitrary meaningless spec (all accepted) gives the angle-bracket text, padded when a width/alignment spec is given; same through formatted string literals.
    # GH-34740
    assert format(NA) == "<NA>"
    assert format(NA, ">10") == "      <NA>"
    assert format(NA, "xxx") == "<NA>"  # NA is flexible, accept any format spec

    assert f"{NA}" == "<NA>"
    assert f"{NA:>10}" == "      <NA>"
    assert f"{NA:xxx}" == "<NA>"


def test_truthiness():
    # [FUNCTION DOC]
    # Converting NA to a boolean or negating it with logical not must raise a type error saying the boolean value of NA is ambiguous.
    msg = "boolean value of NA is ambiguous"

    with pytest.raises(TypeError, match=msg):
        bool(NA)

    with pytest.raises(TypeError, match=msg):
        not NA


def test_hashable():
    # [FUNCTION DOC]
    # NA is hashable consistently and usable as a dictionary key.
    assert hash(NA) == hash(NA)
    d = {NA: "test"}
    assert d[NA] == "test"


@pytest.mark.parametrize(
    "other", [NA, 1, 1.0, "a", b"a", np.int64(1), np.nan], ids=repr
)
def test_arithmetic_ops(all_arithmetic_functions, other):
    # [FUNCTION DOC]
    # Parametrized over the fixture of all arithmetic functions (forward and reflected) and other operands (NA, ints, floats, strings, bytes, numpy int, nan). Pairs that are not defined are skipped (power or reflected modulo with text operands).
    # Division-with-remainder functions return a pair of NA values; all others return NA, with one added to the operand for reflected power to avoid the special case where the base is one.
    op = all_arithmetic_functions

    if op.__name__ in ("pow", "rpow", "rmod") and isinstance(other, (str, bytes)):
        pytest.skip(reason=f"{op.__name__} with NA and {other} not defined.")
    if op.__name__ in ("divmod", "rdivmod"):
        assert op(NA, other) is (NA, NA)
    else:
        if op.__name__ == "rpow":
            # avoid special case
            other += 1
        assert op(NA, other) is NA


@pytest.mark.parametrize(
    "other",
    [
        NA,
        1,
        1.0,
        "a",
        b"a",
        np.int64(1),
        np.nan,
        np.bool_(True),
        time(0),
        date(1, 2, 3),
        timedelta(1),
        pd.NaT,
    ],
)
def test_comparison_ops(comparison_op, other):
    # [FUNCTION DOC]
    # Parametrized over the comparison operator fixture and operands of many kinds (NA, numbers, text, bytes, numpy bool, time, date, timedelta, not-a-time). Comparing NA with the operand in either order returns NA.
    assert comparison_op(NA, other) is NA
    assert comparison_op(other, NA) is NA


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        -0,
        -0.0,
        False,
        np.bool_(False),
        np.int_(0),
        np.float64(0),
        np.int_(-0),
        np.float64(-0),
    ],
)
@pytest.mark.parametrize("asarray", [True, False])
def test_pow_special(value, asarray):
    # [FUNCTION DOC]
    # Parametrized over zero-valued numbers of many types (including negative zero and numpy scalars), as scalars or one-element arrays. Raising NA to the power zero gives one; for scalars the result has the same type as the exponent.
    if asarray:
        value = np.array([value])
    result = NA**value

    if asarray:
        result = result[0]
    else:
        # this assertion isn't possible for ndarray.
        assert isinstance(result, type(value))
    assert result == 1


@pytest.mark.parametrize(
    "value", [1, 1.0, True, np.bool_(True), np.int_(1), np.float64(1)]
)
@pytest.mark.parametrize("asarray", [True, False])
def test_rpow_special(value, asarray):
    # [FUNCTION DOC]
    # Parametrized over one-valued numbers of many types, as scalars or one-element arrays. Raising one to the power NA gives one (equal to the base); for pure Python scalars the result keeps the base's type.
    if asarray:
        value = np.array([value])
    result = value**NA

    if asarray:
        result = result[0]
    elif not isinstance(value, (np.float64, np.bool_, np.int_)):
        # this assertion isn't possible with asarray=True
        assert isinstance(result, type(value))

    assert result == value


@pytest.mark.parametrize("value", [-1, -1.0, np.int_(-1), np.float64(-1)])
@pytest.mark.parametrize("asarray", [True, False])
def test_rpow_minus_one(value, asarray):
    # [FUNCTION DOC]
    # Parametrized over minus-one numbers (scalar or one-element array). Raising minus one to the power NA gives a missing value.
    if asarray:
        value = np.array([value])
    result = value**NA

    if asarray:
        result = result[0]

    assert pd.isna(result)


def test_unary_ops():
    # [FUNCTION DOC]
    # Unary plus, minus, absolute value and invert applied to NA all return NA.
    assert +NA is NA
    assert -NA is NA
    assert abs(NA) is NA
    assert ~NA is NA


def test_logical_and():
    # [FUNCTION DOC]
    # Logical and with NA: with true or NA the result is NA, with false the result is false (either operand order), including numpy booleans (GH#58427); combining with an integer must raise a type error for unsupported operand.
    assert NA & True is NA
    assert True & NA is NA
    assert NA & False is False
    assert False & NA is False
    assert NA & NA is NA

    # GH#58427
    assert NA & np.bool_(True) is NA
    assert np.bool_(True) & NA is NA
    assert NA & np.bool_(False) is False
    assert np.bool_(False) & NA is False

    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA & 5


def test_logical_or():
    # [FUNCTION DOC]
    # Logical or with NA: with true the result is true, with false or NA the result is NA (either order), including numpy booleans (GH#58427); combining with an integer must raise a type error for unsupported operand.
    assert NA | True is True
    assert True | NA is True
    assert NA | False is NA
    assert False | NA is NA
    assert NA | NA is NA

    # GH#58427
    assert NA | np.bool_(True) is True
    assert np.bool_(True) | NA is True
    assert NA | np.bool_(False) is NA
    assert np.bool_(False) | NA is NA

    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA | 5


def test_logical_xor():
    # [FUNCTION DOC]
    # Logical exclusive-or with NA gives NA for true, false, NA and numpy booleans in either order (GH#58427); combining with an integer must raise a type error for unsupported operand.
    assert NA ^ True is NA
    assert True ^ NA is NA
    assert NA ^ False is NA
    assert False ^ NA is NA
    assert NA ^ NA is NA

    # GH#58427
    assert NA ^ np.bool_(True) is NA
    assert np.bool_(True) ^ NA is NA
    assert NA ^ np.bool_(False) is NA
    assert np.bool_(False) ^ NA is NA

    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA ^ 5


def test_logical_not():
    # [FUNCTION DOC]
    # Bitwise inversion of NA returns NA.
    # KNOWN ISSUE: this duplicates the invert assertion already made in the unary operators test.
    assert ~NA is NA


@pytest.mark.parametrize("shape", [(3,), (3, 3), (1, 2, 3)])
def test_arithmetic_ndarray(shape, all_arithmetic_functions):
    # [FUNCTION DOC]
    # Parametrized over array shapes (one, two and three dimensions) and all arithmetic functions. Applying the function to NA and an array of zeros (fives for power) returns an object array of the same shape filled with NA.
    op = all_arithmetic_functions
    a = np.zeros(shape)
    if op.__name__ == "pow":
        a += 5
    result = op(NA, a)
    expected = np.full(a.shape, NA, dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_is_scalar():
    # [FUNCTION DOC]
    # The scalar-check helper from the dtype utilities reports NA as a scalar.
    assert is_scalar(NA) is True


def test_isna():
    # [FUNCTION DOC]
    # The missing-value check is true and the not-missing check false for NA.
    assert pd.isna(NA) is True
    assert pd.notna(NA) is False


def test_series_isna():
    # [FUNCTION DOC]
    # An object-dtype series containing NA flags it as missing and other entries as not missing in the isna result.
    s = pd.Series([1, NA], dtype=object)
    expected = pd.Series([False, True])
    tm.assert_series_equal(s.isna(), expected)


def test_ufunc():
    # [FUNCTION DOC]
    # Numpy universal functions applied to NA return NA: a unary one (logarithm), a binary one with a number (addition), and ones with two outputs (divmod and frexp), whose outputs are both NA.
    assert np.log(NA) is NA
    assert np.add(NA, 1) is NA
    result = np.divmod(NA, 1)
    assert result[0] is NA and result[1] is NA

    result = np.frexp(NA)
    assert result[0] is NA and result[1] is NA


def test_ufunc_raises():
    # [FUNCTION DOC]
    # Calling the in-place at method of a universal function with NA as input must raise a value error mentioning that the ufunc method at is not supported.
    msg = "ufunc method 'at'"
    with pytest.raises(ValueError, match=msg):
        np.log.at(NA, 0)


def test_binary_input_not_dunder():
    # [FUNCTION DOC]
    # Binary universal functions that have no matching operator method (log-add-exp) combine NA with an integer array in either order to give an object array of NA values; with two NA inputs the result is NA;
    # a two-output function (modf) given two NA inputs returns a pair of two NA values.
    a = np.array([1, 2, 3])
    expected = np.array([NA, NA, NA], dtype=object)
    result = np.logaddexp(a, NA)
    tm.assert_numpy_array_equal(result, expected)

    result = np.logaddexp(NA, a)
    tm.assert_numpy_array_equal(result, expected)

    # all NA, multiple inputs
    assert np.logaddexp(NA, NA) is NA

    result = np.modf(NA, NA)
    assert len(result) == 2
    assert all(x is NA for x in result)


def test_divmod_ufunc():
    # [FUNCTION DOC]
    # Divmod ufunc with an array and NA (either order) returns a tuple of two object arrays filled with NA.
    # KNOWN ISSUE: the array comparison inside both loops is repeated twice with identical arguments, so one of the two assertions is redundant.
    # binary in, binary out.
    a = np.array([1, 2, 3])
    expected = np.array([NA, NA, NA], dtype=object)

    result = np.divmod(a, NA)
    assert isinstance(result, tuple)
    for arr in result:
        tm.assert_numpy_array_equal(arr, expected)
        tm.assert_numpy_array_equal(arr, expected)

    result = np.divmod(NA, a)
    for arr in result:
        tm.assert_numpy_array_equal(arr, expected)
        tm.assert_numpy_array_equal(arr, expected)


def test_integer_hash_collision_dict():
    # [FUNCTION DOC]
    # GH 30013. A dictionary can hold both NA and the integer equal to its hash value as separate keys.
    # GH 30013
    result = {NA: "foo", hash(NA): "bar"}

    assert result[NA] == "foo"
    assert result[hash(NA)] == "bar"


def test_integer_hash_collision_set():
    # [FUNCTION DOC]
    # GH 30013. A set can hold both NA and the integer equal to its hash value as separate elements.
    # GH 30013
    result = {NA, hash(NA)}

    assert len(result) == 2
    assert NA in result
    assert hash(NA) in result


def test_pickle_roundtrip():
    # [FUNCTION DOC]
    # Standard-library pickle round trip returns the same singleton object (GH 31847).
    # https://github.com/pandas-dev/pandas/issues/31847
    result = pickle.loads(pickle.dumps(NA))
    assert result is NA


def test_pickle_roundtrip_pandas(temp_file):
    # [FUNCTION DOC]
    # Round trip through pandas' pickle test helper using the temporary file fixture returns the same singleton.
    result = tm.round_trip_pickle(NA, temp_file)
    assert result is NA


@pytest.mark.parametrize(
    "values, dtype", [([1, 2, NA], "Int64"), (["A", "B", NA], "string")]
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_pickle_roundtrip_containers(as_frame, values, dtype, temp_file):
    # [FUNCTION DOC]
    # Parametrized over nullable integer and string values with NA, and as series or single-column frame. Round trip through the pickle helper (temporary file fixture) returns an equal object.
    s = pd.Series(pd.array(values, dtype=dtype))
    if as_frame:
        s = s.to_frame(name="A")
    result = tm.round_trip_pickle(s, temp_file)
    tm.assert_equal(result, s)
