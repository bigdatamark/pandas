# [MODULE DOC]
# Arithmetic and comparison tests for string-dtype data (python and pyarrow backed string dtypes, with NaN or NA as the missing marker, plus object dtype) across arrays, series, indexes, and frames.
# Covers equality with all-NA input, reflected logical operators between booleans and strings, division with path objects, custom-object addition, mixed object comparisons, invalid multiplication by booleans, addition and multiplication, and the dunder comparison methods against scalars, arrays, and lists. Many tests use shared fixtures that enumerate string dtypes and the comparison operators, and attach expected-failure marks for known unsupported combinations.
import operator
from pathlib import Path

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
from pandas.errors import Pandas4Warning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    NA,
    ArrowDtype,
    Series,
    StringDtype,
)
import pandas._testing as tm
from pandas.core.construction import extract_array


def string_dtype_highest_priority(dtype1, dtype2):
    # [FUNCTION DOC]
    # HELPER: for comparison tests. Given two string dtypes, returns whichever has the higher priority in a fixed hierarchy that decides the result type of mixed operations. The hierarchy, from lowest: python storage with NaN, pyarrow storage with NaN, python storage with NA, pyarrow storage with NA; if the pyarrow library is unavailable only the two python entries exist. Raises a value error (from list lookup) if a dtype is not in the hierarchy.
    if HAS_PYARROW:
        DTYPE_HIERARCHY = [
            StringDtype("python", na_value=np.nan),
            StringDtype("pyarrow", na_value=np.nan),
            StringDtype("python", na_value=NA),
            StringDtype("pyarrow", na_value=NA),
        ]
    else:
        DTYPE_HIERARCHY = [
            StringDtype("python", na_value=np.nan),
            StringDtype("python", na_value=NA),
        ]

    h1 = DTYPE_HIERARCHY.index(dtype1)
    h2 = DTYPE_HIERARCHY.index(dtype2)
    return DTYPE_HIERARCHY[max(h1, h2)]


def test_eq_all_na():
    # [FUNCTION DOC]
    # Requires pyarrow (skipped otherwise). Verifies equality of an all-NA pyarrow-backed string array with itself returns an all-NA pyarrow boolean array.
    pytest.importorskip("pyarrow")
    a = pd.array([NA, NA], dtype=StringDtype("pyarrow"))
    result = a == a
    expected = pd.array([NA, NA], dtype="boolean[pyarrow]")
    tm.assert_extension_array_equal(result, expected)


def test_reversed_logical_ops(any_string_dtype):
    # [FUNCTION DOC]
    # Uses the any-string-dtype fixture. Verifies that or, and, and xor of a boolean series (left) with a string series (right) give the same result as with the right side cast to bool, and that for non-object string dtypes each operation emits a Pandas4Warning whose message mentions operations between boolean dtype and another dtype (no warning for object dtype).
    # GH#60234
    dtype = any_string_dtype
    warn = None if dtype == object else Pandas4Warning
    left = Series([True, False, False, True])
    right = Series(["", "", "b", "c"], dtype=dtype)

    msg = "operations between boolean dtype and"
    with tm.assert_produces_warning(warn, match=msg):
        result = left | right
    expected = left | right.astype(bool)
    tm.assert_series_equal(result, expected)

    with tm.assert_produces_warning(warn, match=msg):
        result = left & right
    expected = left & right.astype(bool)
    tm.assert_series_equal(result, expected)

    with tm.assert_produces_warning(warn, match=msg):
        result = left ^ right
    expected = left ^ right.astype(bool)
    tm.assert_series_equal(result, expected)


def test_pathlib_path_division(any_string_dtype, request):
    # [FUNCTION DOC]
    # Linked to an issue about path division. Uses the string-dtype fixture and request object. A path object divided by a string series (and a series divided by a path, with the path as right operand) gives an object series of joined paths with the dtype's missing value in the NA position. Object dtype is marked expected-failure because the masked-operation fallback raises a type error for paths.
    # GH#61940
    if any_string_dtype == object:
        mark = pytest.mark.xfail(
            reason="with NA present we go through _masked_arith_op which "
            "raises TypeError bc Path is not recognized by lib.is_scalar."
        )
        request.applymarker(mark)

    item = Path("/Users/Irv/")
    ser = Series(["A", "B", NA], dtype=any_string_dtype)

    result = item / ser
    expected = Series([item / "A", item / "B", ser.dtype.na_value], dtype=object)
    tm.assert_series_equal(result, expected)

    result = ser / item
    expected = Series(["A" / item, "B" / item, ser.dtype.na_value], dtype=object)
    tm.assert_series_equal(result, expected)


def test_arithmetic_custom_object(any_string_dtype):
    # [FUNCTION DOC]
    # Linked to an issue about custom objects. Defines a local class whose instances support addition and reflected addition (returning not-implemented when the other operand has a dtype attribute, so the string series' implementation is tried), equality by value, and a readable repr. Verifies adding a string series to a scalar custom object, to an array of them, or to an object series of them, in both operand orders, gives an object series of custom objects with the concatenated values.
    # GH#64107
    class CustomObject:
        # [CLASS DOC]
        # HELPER: local value-holder class used by the test above. Stores one value; addition operators defer when the other operand has a dtype, otherwise return a new instance combining the value with the other operand.
        def __init__(self, val):
            # [FUNCTION DOC]
            # HELPER: constructor; stores the given value as an attribute.
            self.val = val

        def __add__(self, other):
            # [FUNCTION DOC]
            # HELPER: addition; returns the not-implemented sentinel if the other operand has a dtype attribute, otherwise a new instance holding the sum of its value and the other operand.
            if hasattr(other, "dtype"):
                return NotImplemented
            return CustomObject(self.val + other)

        def __radd__(self, other):
            # [FUNCTION DOC]
            # HELPER: reflected addition; returns the not-implemented sentinel if the other operand has a dtype attribute, otherwise a new instance holding the other operand plus its value.
            if hasattr(other, "dtype"):
                return NotImplemented
            return CustomObject(other + self.val)

        def __eq__(self, other):
            # [FUNCTION DOC]
            # HELPER: equality; true only if the other is an instance of this class with an equal value.
            return isinstance(other, CustomObject) and self.val == other.val

        def __repr__(self):
            # [FUNCTION DOC]
            # HELPER: returns a readable string showing the class name and the stored value.
            return f"<CustomObject({self.val})>"

    arr = np.array([CustomObject("a"), CustomObject("b")], dtype=object)
    ser = Series(["1", "2"], dtype=any_string_dtype)
    obj_ser = Series(arr)

    cases = [
        (ser + arr[0], [CustomObject("1a"), CustomObject("2a")]),
        (ser + arr, [CustomObject("1a"), CustomObject("2b")]),
        (ser + obj_ser, [CustomObject("1a"), CustomObject("2b")]),
        (arr[0] + ser, [CustomObject("a1"), CustomObject("a2")]),
        (arr + ser, [CustomObject("a1"), CustomObject("b2")]),
        (obj_ser + ser, [CustomObject("a1"), CustomObject("b2")]),
    ]
    for result, expected_values in cases:
        expected = Series(expected_values, dtype=object)
        tm.assert_series_equal(result, expected)


def test_mixed_object_comparison(any_string_dtype):
    # [FUNCTION DOC]
    # Linked to an issue. Verifies equality between a string series and an object series with a mixed integer/string entry gives false for the mismatch and true for the match. The expected dtype is plain bool for object and NaN-based dtypes, nullable boolean for python NA-based dtypes, and pyarrow boolean for pyarrow NA-based dtypes.
    # GH#60228
    dtype = any_string_dtype
    ser = Series(["a", "b"], dtype=dtype)

    mixed = Series([1, "b"], dtype=object)

    result = ser == mixed
    expected = Series([False, True], dtype=bool)
    if dtype == object:
        pass
    elif dtype.storage == "python" and dtype.na_value is NA:
        expected = expected.astype("boolean")
    elif dtype.storage == "pyarrow" and dtype.na_value is NA:
        expected = expected.astype("bool[pyarrow]")

    tm.assert_series_equal(result, expected)


def test_pyarrow_numpy_string_invalid():
    # [FUNCTION DOC]
    # Requires pyarrow. Linked to issues. Verifies comparing a boolean series with a NaN-based string series: equality is all false, inequality is all true, and ordering comparisons raise a type error "Invalid comparison"; the same holds for a pyarrow-string-typed and an Arrow-extension-string-typed version, where the equality results are pyarrow booleans.
    # GH#56008
    pa = pytest.importorskip("pyarrow")
    ser = Series([False, True])
    ser2 = Series(["a", "b"], dtype=StringDtype(na_value=np.nan))
    result = ser == ser2
    expected_eq = Series(False, index=ser.index)
    tm.assert_series_equal(result, expected_eq)

    result = ser != ser2
    expected_ne = Series(True, index=ser.index)
    tm.assert_series_equal(result, expected_ne)

    with pytest.raises(TypeError, match="Invalid comparison"):
        ser > ser2

    # GH#59505
    ser3 = ser2.astype("string[pyarrow]")
    result3_eq = ser3 == ser
    tm.assert_series_equal(result3_eq, expected_eq.astype("bool[pyarrow]"))
    result3_ne = ser3 != ser
    tm.assert_series_equal(result3_ne, expected_ne.astype("bool[pyarrow]"))

    with pytest.raises(TypeError, match="Invalid comparison"):
        ser > ser3

    ser4 = ser2.astype(ArrowDtype(pa.string()))
    result4_eq = ser4 == ser
    tm.assert_series_equal(result4_eq, expected_eq.astype("bool[pyarrow]"))
    result4_ne = ser4 != ser
    tm.assert_series_equal(result4_ne, expected_ne.astype("bool[pyarrow]"))

    with pytest.raises(TypeError, match="Invalid comparison"):
        ser > ser4


def test_mul_bool_invalid(any_string_dtype):
    # [FUNCTION DOC]
    # Linked to an issue. Verifies that multiplying a string series by a bool scalar or bool array (either order) raises a type error: for python storage a message about not multiplying by bools and to cast to integers; for pyarrow storage a message that only integers are allowed. Skipped for object dtype.
    # GH#62595
    dtype = any_string_dtype
    ser = Series(["a", "b", "c"], dtype=dtype)

    if dtype == object:
        pytest.skip("This is not expect to raise")
    elif dtype.storage == "python":
        msg = "Cannot multiply StringArray by bools. Explicitly cast to integers"
    else:
        msg = "Can only string multiply by an integer"

    with pytest.raises(TypeError, match=msg):
        False * ser
    with pytest.raises(TypeError, match=msg):
        ser * True
    with pytest.raises(TypeError, match=msg):
        ser * np.array([True, False, True], dtype=bool)
    with pytest.raises(TypeError, match=msg):
        np.array([True, False, True], dtype=bool) * ser


def test_add(any_string_dtype, request):
    # [FUNCTION DOC]
    # Verifies series addition of two string series with missing values: missing on either side gives missing, the add and reflected add methods agree with the operators in order, and add with a fill value treats missing as that value on one side only (both missing stays missing). Object dtype is marked expected-failure because the expected values differ for it.
    dtype = any_string_dtype
    if dtype == object:
        mark = pytest.mark.xfail(
            reason="Need to update expected for numpy object dtype"
        )
        request.applymarker(mark)

    a = Series(["a", "b", "c", None, None], dtype=dtype)
    b = Series(["x", "y", None, "z", None], dtype=dtype)

    result = a + b
    expected = Series(["ax", "by", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b)
    tm.assert_series_equal(result, expected)

    result = a.radd(b)
    expected = Series(["xa", "yb", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b, fill_value="-")
    expected = Series(["ax", "by", "c-", "-z", None], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_add_2d(any_string_dtype, request):
    # [FUNCTION DOC]
    # Verifies adding a string array (or series) to a two-dimensional object NumPy array of different length raises a value error with a message about the lengths (three versus one). For object dtype this is marked expected-failure because no error is raised.
    dtype = any_string_dtype

    if dtype == object:
        reason = "Failed: DID NOT RAISE <class 'ValueError'>"
        mark = pytest.mark.xfail(raises=None, reason=reason)
        request.applymarker(mark)

    a = pd.array(["a", "b", "c"], dtype=dtype)
    b = np.array([["a", "b", "c"]], dtype=object)
    with pytest.raises(ValueError, match="3 != 1"):
        a + b

    s = Series(a)
    with pytest.raises(ValueError, match="3 != 1"):
        s + b


def test_add_sequence(any_string_dtype, request, using_infer_string):
    # [FUNCTION DOC]
    # Uses the infer-string option fixture. Verifies adding a list of strings and missing values to a string array, and a list plus the array, gives elementwise concatenation with missing where either side is missing, as a string array. Marked expected-failure for the python NaN-based dtype when pyarrow is present with infer-string (list gets wrapped into a higher-priority array, yielding not-implemented) and for object dtype with infer-string (cannot broadcast a list).
    dtype = any_string_dtype
    if (
        dtype != object
        and dtype.storage == "python"
        and dtype.na_value is np.nan
        and HAS_PYARROW
        and using_infer_string
    ):
        mark = pytest.mark.xfail(
            reason="As of GH#62522, the list gets wrapped with sanitize_array, "
            "which casts to a higher-priority StringArray, so we get "
            "NotImplemented."
        )
        request.applymarker(mark)
    if dtype == np.dtype(object) and using_infer_string:
        mark = pytest.mark.xfail(reason="Cannot broadcast list")
        request.applymarker(mark)

    a = pd.array(["a", "b", None, None], dtype=dtype)
    other = ["x", None, "y", None]

    result = a + other
    expected = pd.array(["ax", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = other + a
    expected = pd.array(["xa", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_string_add_missing_values(string_dtype_no_object):
    # [FUNCTION DOC]
    # Linked to an issue. Uses a fixture of string dtypes without object. Verifies that adding a missing value (None, NaN, or NA) to a one-element string array on either side yields an array with NA, of the same dtype.
    # GH#64968 Arrow-backed str arrays should return NA when added to missing
    arr = pd.array(["y"], dtype=string_dtype_no_object)
    expected = pd.array([NA], dtype=string_dtype_no_object)

    for na_val in [None, np.nan, NA]:
        # left side
        result = arr + na_val
        tm.assert_extension_array_equal(result, expected)

        # right side
        result = na_val + arr
        tm.assert_extension_array_equal(result, expected)


def test_mul(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies multiplying a string array by the integer two on either side repeats each string and keeps missing as missing.
    dtype = any_string_dtype
    a = pd.array(["a", "b", None], dtype=dtype)
    result = a * 2
    expected = pd.array(["aa", "bb", None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = 2 * a
    tm.assert_extension_array_equal(result, expected)


def test_add_strings(any_string_dtype, request):
    # [FUNCTION DOC]
    # Marked expected-failure for non-object dtypes (a linked issue). Verifies that the array's add method returns the not-implemented sentinel for a frame, and that adding a frame of object strings to the array, and the frame plus the array, concatenate elementwise into a frame cast to the dtype.
    dtype = any_string_dtype
    if dtype != np.dtype(object):
        mark = pytest.mark.xfail(reason="GH-28527")
        request.applymarker(mark)
    arr = pd.array(["a", "b", "c", "d"], dtype=dtype)
    df = pd.DataFrame([["t", "y", "v", "w"]], dtype=object)
    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["at", "by", "cv", "dw"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["ta", "yb", "vc", "wd"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_frame(dtype):
    # [FUNCTION DOC]
    # Always marked expected-failure (a linked issue). Uses the dtype fixture. Verifies adding an array to an object frame gives not-implemented from the array's method, and that addition in either order concatenates with NaN where either side is missing.
    arr = pd.array(["a", "b", np.nan, np.nan], dtype=dtype)
    df = pd.DataFrame([["x", np.nan, "y", np.nan]])

    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["ax", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["xa", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_comparison_methods_scalar(comparison_op, any_string_dtype):
    # [FUNCTION DOC]
    # Uses the comparison-operator fixture and the string-dtype fixture. Calls the dunder comparison method for the operator with a string scalar on an array with a missing value. For object or NaN-based dtypes, expects a NumPy bool array (after extracting from the result) equal to the per-item comparison, with the missing position true for not-equal and false otherwise. For NA-based dtypes, expects a nullable boolean (or pyarrow boolean for pyarrow storage) array with missing where the input is missing.
    dtype = any_string_dtype
    op_name = f"__{comparison_op.__name__}__"
    a = pd.array(["a", None, "c"], dtype=dtype)
    other = "a"
    result = getattr(a, op_name)(other)
    if dtype == object or dtype.na_value is np.nan:
        expected = np.array([getattr(item, op_name)(other) for item in a])
        if comparison_op == operator.ne:
            expected[1] = True
        else:
            expected[1] = False
        result = extract_array(result, extract_numpy=True)
        tm.assert_numpy_array_equal(result, expected.astype(np.bool_))
    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = np.array([getattr(item, op_name)(other) for item in a], dtype=object)
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_pd_na(comparison_op, any_string_dtype):
    # [FUNCTION DOC]
    # Same setup as the scalar test but compares with the NA singleton. For object or NaN-based dtypes the result is all true for not-equal and all false otherwise; for NA-based dtypes the result is an all-missing nullable boolean (or pyarrow boolean) array.
    # KNOWN ISSUE: the final assertion in the NA-based branch is duplicated; the second one is redundant.
    dtype = any_string_dtype
    op_name = f"__{comparison_op.__name__}__"
    a = pd.array(["a", None, "c"], dtype=dtype)
    result = getattr(a, op_name)(NA)

    if dtype == np.dtype(object) or dtype.na_value is np.nan:
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        result = extract_array(result, extract_numpy=True)
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_not_string(comparison_op, any_string_dtype):
    # [FUNCTION DOC]
    # Compares a string array with the integer forty-two via the dunder method. Ordering operators must raise a type error ("Invalid comparison" or not supported between). For equality and inequality: object or NaN-based dtypes give all false or all true NumPy bool arrays; NA-based dtypes give false/missing/false or true/missing/true in a nullable boolean (or pyarrow boolean) array.
    op_name = f"__{comparison_op.__name__}__"
    dtype = any_string_dtype

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = 42

    if op_name not in ["__eq__", "__ne__"]:
        with pytest.raises(TypeError, match="Invalid comparison|not supported between"):
            getattr(a, op_name)(other)

        return

    result = getattr(a, op_name)(other)
    result = extract_array(result, extract_numpy=True)

    if dtype == np.dtype(object) or dtype.na_value is np.nan:
        expected_data = {
            "__eq__": [False, False, False],
            "__ne__": [True, True, True],
        }[op_name]
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_data = {"__eq__": [False, None, False], "__ne__": [True, None, True]}[
            op_name
        ]
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array(expected_data, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_array(comparison_op, any_string_dtype, any_string_dtype2):
    # [FUNCTION DOC]
    # Uses the comparison-operator fixture and two string-dtype fixtures. Compares a string array with missing values against another string array (other dtype) and checks commutativity of the operator. When both dtypes are object or NaN-based, expects a NumPy bool result (true everywhere for not-equal except the last position, false for others with the last position computed from the strings). Otherwise the expected result is a nullable boolean (python storage) or pyarrow boolean array determined by the higher-priority dtype (object defers to the other), missing except at the last position.
    op_name = f"__{comparison_op.__name__}__"
    dtype = any_string_dtype
    dtype2 = any_string_dtype2

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = pd.array([None, None, "c"], dtype=dtype2)
    result = comparison_op(a, other)
    result = extract_array(result, extract_numpy=True)

    # ensure operation is commutative
    result2 = comparison_op(other, a)
    result2 = extract_array(result2, extract_numpy=True)
    tm.assert_equal(result, result2)

    if (dtype == object or dtype.na_value is np.nan) and (
        dtype2 == object or dtype2.na_value is np.nan
    ):
        if operator.ne == comparison_op:
            expected = np.array([True, True, False])
        else:
            expected = np.array([False, False, False])
            expected[-1] = getattr(other[-1], op_name)(a[-1])
        result = extract_array(result, extract_numpy=True)
        tm.assert_numpy_array_equal(result, expected)

    else:
        if dtype == object:
            max_dtype = dtype2
        elif dtype2 == object:
            max_dtype = dtype
        else:
            max_dtype = string_dtype_highest_priority(dtype, dtype2)
        if max_dtype.storage == "python":
            expected_dtype = "boolean"
        else:
            expected_dtype = "bool[pyarrow]"

        expected = np.full(len(a), fill_value=None, dtype="object")
        expected[-1] = getattr(other[-1], op_name)(a[-1])
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_comparison_methods_array_arrow_extension(comparison_op, any_string_dtype):
    # [FUNCTION DOC]
    # Requires pyarrow. Compares an Arrow-extension string array with a string array of the fixture dtype: operator is commutative and the result is a pyarrow boolean array missing where either side is missing and computed from the strings at the last position.
    # Test pd.ArrowDtype(pa.string()) against other string arrays
    import pyarrow as pa

    dtype2 = any_string_dtype

    op_name = f"__{comparison_op.__name__}__"
    dtype = ArrowDtype(pa.string())
    a = pd.array(["a", None, "c"], dtype=dtype)
    other = pd.array([None, None, "c"], dtype=dtype2)
    result = comparison_op(a, other)

    # ensure operation is commutative
    result2 = comparison_op(other, a)
    tm.assert_equal(result, result2)

    expected = pd.array([None, None, True], dtype="bool[pyarrow]")
    expected[-1] = getattr(other[-1], op_name)(a[-1])
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("box", [pd.array, pd.Index, Series])
def test_comparison_methods_list(comparison_op, any_string_dtype, box, request):
    # [FUNCTION DOC]
    # Parametrized over the container type (array, index, series); uses the comparison-operator and string-dtype fixtures. Compares a container of strings with a plain list of values (with missing entries) on either side, checks commutativity, and checks the result: for object or NaN-based dtypes a boolean NumPy-backed result (boxed unless the container is an index, per a linked issue), for NA-based dtypes a nullable or pyarrow boolean result with missing except at the last position. Marked expected-failure for an array of NaN-based dtype since the wrapped list returns not-implemented.
    dtype = any_string_dtype

    if box is pd.array and dtype != object and dtype.na_value is np.nan:
        mark = pytest.mark.xfail(
            reason="After wrapping list, op returns NotImplemented, see GH#62522"
        )
        request.applymarker(mark)

    op_name = f"__{comparison_op.__name__}__"

    a = box(pd.array(["a", None, "c"], dtype=dtype))
    item = "c"
    other = [None, None, "c"]
    result = comparison_op(a, other)

    # ensure operation is commutative
    result2 = comparison_op(other, a)
    tm.assert_equal(result, result2)

    if dtype == np.dtype(object) or dtype.na_value is np.nan:
        if operator.ne == comparison_op:
            expected = np.array([True, True, False])
        else:
            expected = np.array([False, False, False])
            expected[-1] = getattr(item, op_name)(item)
        if box is not pd.Index:
            # if GH#62766 is addressed this check can be removed
            expected = box(expected, dtype=expected.dtype)
        tm.assert_equal(result, expected)

    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = np.full(len(a), fill_value=None, dtype="object")
        expected[-1] = getattr(item, op_name)(item)
        expected = pd.array(expected, dtype=expected_dtype)
        expected = extract_array(expected, extract_numpy=True)
        if box is not pd.Index:
            # if GH#62766 is addressed this check can be removed
            expected = tm.box_expected(expected, box)
        tm.assert_equal(result, expected)
