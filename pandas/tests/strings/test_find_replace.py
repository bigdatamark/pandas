# [MODULE DOC]
# Tests for the string accessor's pattern-matching and replacement methods: contains, starts-with, ends-with, replace, match, full match, find-all, find/right-find, translate, and the flags argument. Sections are marked by header comments. Uses the string dtype fixtures (object, inferred string, nullable string with python or pyarrow storage), the helper identifying object or NaN-based dtypes (which decide whether boolean results are numpy bool/object with NaN or nullable boolean with NA), and the helper converting expected missing values. Many tests repeat the same lookaround, end-of-string and compiled-regex scenarios for each method.
from datetime import datetime
import re

import numpy as np
import pytest

from pandas._libs import lib
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Series,
    StringDtype,
    _testing as tm,
)
from pandas.tests.strings import (
    _convert_na_value,
    is_object_or_nan_string_dtype,
)

# --------------------------------------------------------------------------------------
# str.contains
# --------------------------------------------------------------------------------------


def test_contains(any_string_dtype):
    # [FUNCTION DOC]
    # Uses the string dtype fixture. Verifies contains with a regex pattern and with regex disabled, case-insensitive matching with and without regex, unicode text, the na argument, and results on all-valid data. Missing values give NaN (object result) for object-like dtypes, NA (boolean dtype) for nullable dtypes, and false for the inferred string dtype; with na false the result is plain boolean.
    values = np.array(
        ["foo", np.nan, "fooommm__foo", "mmm_", "foommm[_]+bar"], dtype=np.object_
    )
    values = Series(values, dtype=any_string_dtype)
    pat = "mmm[_]+"

    result = values.str.contains(pat)
    if any_string_dtype == "str":
        # NaN propagates as False
        expected = Series([False, False, True, True, False], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series(
            np.array([False, np.nan, True, True, False], dtype=np.object_),
            dtype=expected_dtype,
        )

    tm.assert_series_equal(result, expected)

    result = values.str.contains(pat, regex=False)
    if any_string_dtype == "str":
        expected = Series([False, False, False, False, True], dtype=bool)
    else:
        expected = Series(
            np.array([False, np.nan, False, False, True], dtype=np.object_),
            dtype=expected_dtype,
        )
    tm.assert_series_equal(result, expected)

    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=object),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # case insensitive using regex
    values = Series(
        np.array(["Foo", "xYz", "fOOomMm__fOo", "MMM_"], dtype=object),
        dtype=any_string_dtype,
    )

    result = values.str.contains("FOO|mmm", case=False)
    expected = Series(np.array([True, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # case insensitive without regex
    result = values.str.contains("foo", regex=False, case=False)
    expected = Series(np.array([True, False, True, False]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # unicode
    values = Series(
        np.array(["foo", np.nan, "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    pat = "mmm[_]+"

    result = values.str.contains(pat)
    if any_string_dtype == "str":
        expected = Series([False, False, True, True], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series(
            np.array([False, np.nan, True, True], dtype=np.object_),
            dtype=expected_dtype,
        )
    tm.assert_series_equal(result, expected)

    result = values.str.contains(pat, na=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_contains_object_mixed():
    # [FUNCTION DOC]
    # Verifies contains on an object Series with non-strings gives booleans for strings, NaN for non-string and missing elements, and None for None.
    mixed = Series(
        np.array(
            ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
            dtype=object,
        )
    )
    result = mixed.str.contains("o")
    expected = Series(
        np.array(
            [False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan],
            dtype=np.object_,
        )
    )
    tm.assert_series_equal(result, expected)


def test_contains_na_kwarg_for_object_category():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies that the na argument fills missing values on categorical and object data for both true and false.
    # gh 22158

    # na for category
    values = Series(["a", "b", "c", "a", np.nan], dtype="category")
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)

    # na for objects
    values = Series(["a", "b", "c", "a", np.nan])
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "na, expected",
    [
        (None, pd.NA),
        (True, True),
        (False, False),
        (0, False),
        (3, True),
        (np.nan, pd.NA),
    ],
)
@pytest.mark.parametrize("regex", [True, False])
def test_contains_na_kwarg_for_nullable_string_dtype(
    nullable_string_dtype, na, expected, regex
):
    # [FUNCTION DOC]
    # Parametrized over na values (none, true, false, zero, three, NaN), the regex flag and the nullable string dtype fixture. For zero and three verifies a value error saying na must be none, NA, NaN, true or false; otherwise verifies a boolean-dtype result with the na value in the missing position (NA for none or NaN).
    # KNOWN ISSUE: the expected values paired with the integer na arguments (false for zero, true for three) are never used because those cases only check for the error.
    # https://github.com/pandas-dev/pandas/pull/41025#issuecomment-824062416

    values = Series(["a", "b", "c", "a", np.nan], dtype=nullable_string_dtype)

    if na in [0, 3] and na is not False:
        msg = f"na must be None, pd.NA, np.nan, True, or False; got {na}"
        with pytest.raises(ValueError, match=msg):
            values.str.contains("a", na=na, regex=regex)
    else:
        result = values.str.contains("a", na=na, regex=regex)
        expected = Series([True, False, False, True, expected], dtype="boolean")
        tm.assert_series_equal(result, expected)


def test_contains_moar(any_string_dtype):
    # [FUNCTION DOC]
    # From an old pull request. Verifies contains on a ten-element Series including empty string, NaN and mixed-case strings, for case-sensitive and case-insensitive patterns; the missing element is NaN (object), NA/boolean (nullable) or false (inferred string).
    # PR #1179
    s = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    result = s.str.contains("a")
    if any_string_dtype == "str":
        # NaN propagates as False
        expected_dtype = bool
        na_value = False
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        na_value = np.nan
    expected = Series(
        [False, False, False, True, True, False, na_value, False, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("a", case=False)
    expected = Series(
        [True, False, False, True, True, False, na_value, True, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("Aa")
    expected = Series(
        [False, False, False, True, False, False, na_value, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba")
    expected = Series(
        [False, False, False, True, False, False, na_value, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba", case=False)
    expected = Series(
        [False, False, False, True, True, False, na_value, True, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_contains_nan(any_string_dtype):
    # [FUNCTION DOC]
    # From a pull request. Verifies contains on an all-NaN Series: na false gives all false, na true all true, no na gives all-missing (or false for the inferred string dtype), and a string na value raises a value error.
    # PR #14171
    s = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)

    result = s.str.contains("foo", na=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series([False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = s.str.contains("foo", na=True)
    expected = Series([True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    msg = "na must be None, pd.NA, np.nan, True, or False; got foo"
    with pytest.raises(ValueError, match=msg):
        s.str.contains("foo", na="foo")

    result = s.str.contains("foo")
    if any_string_dtype == "str":
        # NaN propagates as False
        expected = Series([False, False, False], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series([np.nan, np.nan, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_contains_compiled_regex(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a compiled pattern is accepted; a case argument with a compiled pattern raises a value error saying flags cannot be processed with a compiled pattern, except for pyarrow-backed strings where it works; a compiled case-insensitive pattern works; passing the flags argument with a compiled pattern always raises.
    # KNOWN ISSUE: the pyarrow-backed dtype accepts the case argument with a compiled pattern while the other dtypes raise; the test encodes the inconsistency and carries to-do comments.
    # GH#61942
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    ser = Series(["foo", "bar", "Baz"], dtype=any_string_dtype)

    pat = re.compile("ba.")
    result = ser.str.contains(pat)
    expected = Series([False, True, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # TODO this currently works for pyarrow-backed dtypes but raises for python
    if any_string_dtype == "string" and any_string_dtype.storage == "pyarrow":
        result = ser.str.contains(pat, case=False)
        expected = Series([False, True, True], dtype=expected_dtype)
        tm.assert_series_equal(result, expected)
    else:
        with pytest.raises(
            ValueError, match="cannot process flags argument with a compiled pattern"
        ):
            ser.str.contains(pat, case=False)

    pat = re.compile("ba.", flags=re.IGNORECASE)
    result = ser.str.contains(pat)
    expected = Series([False, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # TODO should this be supported?
    with pytest.raises(
        ValueError, match="cannot process flags argument with a compiled pattern"
    ):
        ser.str.contains(pat, flags=re.IGNORECASE)


def test_contains_compiled_regex_flags(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that flags other than ignore-case (multiline) set on a compiled pattern are respected: the line-start anchor only matches after the newline when multiline is set, and case-insensitivity combines.
    # ensure other (than ignorecase) flags are respected
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    ser = Series(["foobar", "foo\nbar", "Baz"], dtype=any_string_dtype)

    pat = re.compile("^ba")
    result = ser.str.contains(pat)
    expected = Series([False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    pat = re.compile("^ba", flags=re.MULTILINE)
    result = ser.str.contains(pat)
    expected = Series([False, True, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    pat = re.compile("^ba", flags=re.MULTILINE | re.IGNORECASE)
    result = ser.str.contains(pat)
    expected = Series([False, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"a(?=b)", [False, True, False, False]),
        (r"(?<=a)b", [False, True, False, False]),
        (r"a(?!b)", [True, False, True, False]),
        (r"(?<!b)a", [True, True, False, False]),
        ("ab", [False, True, False, False]),
    ],
)
@pytest.mark.parametrize("na", [lib.no_default, True, False, np.nan, None, pd.NA])
def test_contains_lookarounds(any_string_dtype, pat, expected_data, na):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookahead/lookbehind patterns and a plain pattern with expected booleans, and over the na argument (default, true, false, NaN, None, NA). Verifies the result with regex enabled, with the expected value for the missing element depending on dtype and na argument (none/NaN/NA map to None for object dtype, false for the inferred string dtype, NA for nullable), and expected dtype object for object data with a non-boolean na. A comment notes the behavior for NA in object dtype looks wrong.
    # https://github.com/pandas-dev/pandas/issues/60833
    if any_string_dtype == "object" and not isinstance(na, bool):
        expected_dtype = "object"
    else:
        expected_dtype = (
            np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
    if any_string_dtype == "object":
        # The behavior here for `na=pd.NA` looks wrong.
        if (na is lib.no_default or pd.isna(na)) and na is not pd.NA:
            na_result = None
        else:
            na_result = na
    elif na is lib.no_default or pd.isna(na):
        if any_string_dtype == "str":
            na_result = False
        elif any_string_dtype == "string":
            na_result = pd.NA
        else:
            raise ValueError(f"Unrecognized string dtype {any_string_dtype}")
    else:
        na_result = na
    expected_data = expected_data.copy()
    expected_data.append(na_result)
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.contains(pat, regex=True, na=na)
    expected = Series(expected_data, dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_contains_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies end-of-string anchors with contains: dollar matches before a trailing newline for Python-based dtypes but not for pyarrow-backed strings; the end-of-text escape matches only at the very end (translated for pyarrow); and literal backslash-Z sequences of various escaping levels can still be found.
    # https://github.com/pandas-dev/pandas/pull/63613
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.contains("bar$")
    if any_string_dtype == "string" and any_string_dtype.storage == "pyarrow":
        # pyarrow (RE2) only matches $ at the very end of the line
        expected = Series([False, True, False, False], dtype=expected_dtype)
    else:
        # python matches $ before or after an ending newline
        expected = Series([False, True, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.contains(r"bar\Z")
    expected = Series([False, True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series(
        ["bar", r"bar{}".format("\\"), r"bar\Z", r"bar\\Z", "bars", "bar\n"],
        dtype=any_string_dtype,
    )

    result = ser.str.contains(r"bar\\Z")
    expected = Series([False, False, True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.contains(r"bar\\\Z")
    expected = Series([False, True, False, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.contains(r"bar\\\\Z")
    expected = Series([False, False, False, True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.startswith
# --------------------------------------------------------------------------------------


def test_startswith_endswith_validate_na(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies that a string value for the na argument of starts-with and ends-with raises a value error saying na must be none, NA, NaN, true or false.
    # GH#59615
    ser = Series(
        ["om", np.nan, "foo_nom", "nom", "bar_foo", np.nan, "foo"],
        dtype=any_string_dtype,
    )
    msg = "na must be None, pd.NA, np.nan, True, or False; got baz"
    with pytest.raises(ValueError, match=msg):
        ser.str.startswith("kapow", na="baz")
    with pytest.raises(ValueError, match=msg):
        ser.str.endswith("bar", na="baz")


@pytest.mark.parametrize("pat", ["foo", ("foo", "baz")])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_startswith(pat, dtype, null_value, na, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized over a string or tuple of prefixes, object or categorical dtype, the type of null (None, NaN, NA) and na true or false, with the infer-string flag fixture. Verifies the default result puts the dtype-appropriate missing value in missing positions (None, NA or NaN for object; false for categorical under inferred strings), the na argument fills missing positions, and a mixed-type object array gives booleans for strings and missing values elsewhere.
    # add category dtype parametrizations for GH-36241
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    result = values.str.startswith(pat)
    exp = Series([False, np.nan, True, False, False, np.nan, True])
    if dtype == "object" and null_value is pd.NA:
        # GH#18463
        exp = exp.fillna(null_value)
    elif dtype == "object" and null_value is None:
        exp[exp.isna()] = None
    elif using_infer_string and dtype == "category":
        exp = exp.fillna(False).astype(bool)
    tm.assert_series_equal(result, exp)

    result = values.str.startswith(pat, na=na)
    exp = Series([False, na, True, False, False, na, True])
    tm.assert_series_equal(result, exp)

    # mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=np.object_,
    )
    rs = Series(mixed).str.startswith("f")
    xp = Series([False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)


@pytest.mark.parametrize("na", [None, True, False])
def test_startswith_string_dtype(any_string_dtype, na):
    # [FUNCTION DOC]
    # Parametrized over na (none, true, false). Verifies starts-with on string dtypes with missing values: expected dtype object (na none) or bool for object-like dtypes, boolean for nullable, bool for the inferred string dtype where an unset na becomes false; and that a dot in the prefix is literal (matches only the literal text).
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=any_string_dtype,
    )
    result = values.str.startswith("foo", na=na)

    expected_dtype = (
        (object if na is None else bool)
        if is_object_or_nan_string_dtype(any_string_dtype)
        else "boolean"
    )
    if any_string_dtype == "str":
        # NaN propagates as False
        expected_dtype = bool
        if na is None:
            na = False
    exp = Series(
        [False, na, True, False, False, na, True, False, False], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)

    result = values.str.startswith("rege.", na=na)
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)


# --------------------------------------------------------------------------------------
# str.endswith
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pat", ["foo", ("foo", "baz")])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_endswith(pat, dtype, null_value, na, using_infer_string):
    # [FUNCTION DOC]
    # Parametrized as the starts-with test. Verifies ends-with for a string or tuple of suffixes on object or categorical data with different null kinds and na values, plus mixed-type object data.
    # add category dtype parametrizations for GH-36241
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    result = values.str.endswith(pat)
    exp = Series([False, np.nan, False, False, True, np.nan, True])
    if dtype == "object" and null_value is pd.NA:
        # GH#18463
        exp = exp.fillna(null_value)
    elif dtype == "object" and null_value is None:
        exp[exp.isna()] = None
    elif using_infer_string and dtype == "category":
        exp = exp.fillna(False).astype(bool)
    tm.assert_series_equal(result, exp)

    result = values.str.endswith(pat, na=na)
    exp = Series([False, na, False, False, True, na, True])
    tm.assert_series_equal(result, exp)

    # mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    rs = Series(mixed).str.endswith("f")
    xp = Series([False, np.nan, False, np.nan, np.nan, False, None, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)


@pytest.mark.parametrize("na", [None, True, False])
def test_endswith_string_dtype(any_string_dtype, na):
    # [FUNCTION DOC]
    # Parametrized over na (none, true, false). Verifies ends-with on string dtypes like the starts-with counterpart, including a literal dot in the suffix.
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=any_string_dtype,
    )
    result = values.str.endswith("foo", na=na)
    expected_dtype = (
        (object if na is None else bool)
        if is_object_or_nan_string_dtype(any_string_dtype)
        else "boolean"
    )
    if any_string_dtype == "str":
        # NaN propagates as False
        expected_dtype = bool
        if na is None:
            na = False
    exp = Series(
        [False, na, False, False, True, na, True, False, False], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)

    result = values.str.endswith("rege.", na=na)
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)


# --------------------------------------------------------------------------------------
# str.replace
# --------------------------------------------------------------------------------------
def test_replace_dict_invalid(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies passing both a dictionary pattern and a replacement raises a value error saying the replacement cannot be used when the pattern is a dictionary.
    # KNOWN ISSUE: the test requests the string dtype fixture but builds its Series without a dtype, so the fixture parametrization only repeats the same check.
    # GH 51914
    series = Series(data=["A", "B_junk", "C_gunk"], name="my_messy_col")
    msg = "repl cannot be used when pat is a dictionary"

    with pytest.raises(ValueError, match=msg):
        series.str.replace(pat={"A": "a", "B": "b"}, repl="A")


def test_replace_dict(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a dictionary pattern maps each matching key to its value, leaving other strings alone, and keeps the Series name.
    # KNOWN ISSUE: the test requests the string dtype fixture but builds its Series without a dtype, so the fixture parametrization only repeats the same check.
    # GH 51914
    series = Series(data=["A", "B", "C"], name="my_messy_col")
    new_series = series.str.replace(pat={"A": "a", "B": "b"})
    expected = Series(data=["a", "b", "C"], name="my_messy_col")
    tm.assert_series_equal(new_series, expected)


def test_replace(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies regex replacement deletes the matched text, keeping the missing value.
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    result = ser.str.replace("BAD[_]*", "", regex=True)
    expected = Series(["foobar", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_max_replacements(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the n argument limits replacements, for a regex pattern and for a literal pattern.
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    expected = Series(["foobarBAD", np.nan], dtype=any_string_dtype)
    result = ser.str.replace("BAD[_]*", "", n=1, regex=True)
    tm.assert_series_equal(result, expected)

    expected = Series(["foo__barBAD", np.nan], dtype=any_string_dtype)
    result = ser.str.replace("BAD", "", n=1, regex=False)
    tm.assert_series_equal(result, expected)


def test_replace_mixed_object():
    # [FUNCTION DOC]
    # Verifies replace on an object Series with non-strings gives replaced strings for strings and missing for non-strings (None kept).
    ser = Series(
        ["aBAD", np.nan, "bBAD", True, datetime.today(), "fooBAD", None, 1, 2.0]
    )
    result = Series(ser).str.replace("BAD[_]*", "", regex=True)
    expected = Series(
        ["a", np.nan, "b", np.nan, np.nan, "foo", None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)


def test_replace_unicode(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a regex with lookbehind and lookahead and the unicode flag replaces a comma between word characters including accented letters.
    ser = Series([b"abcd,\xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    expected = Series([b"abcd, \xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    result = ser.str.replace(r"(?<=\w),(?=\w)", ", ", flags=re.UNICODE, regex=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("repl", [None, 3, {"a": "b"}])
@pytest.mark.parametrize("data", [["a", "b", None], ["a", "b", "c", "ad"]])
def test_replace_wrong_repl_type_raises(any_string_dtype, index_or_series, repl, data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over a replacement of none, an integer, or a dictionary, over data with and without missing values, and over Series or Index. Verifies a type error saying the replacement must be a string or callable.
    # https://github.com/pandas-dev/pandas/issues/13438
    msg = "repl must be a string or callable"
    obj = index_or_series(data, dtype=any_string_dtype)
    with pytest.raises(TypeError, match=msg):
        obj.str.replace("a", repl)


def test_replace_callable(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a callable replacement receives each match and its return value is substituted, with the n argument limiting replacements.
    # GH 15055
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # test with callable
    repl = lambda m: m.group(0).swapcase()
    result = ser.str.replace("[a-z][A-Z]{2}", repl, n=2, regex=True)
    expected = Series(["foObaD__baRbaD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "repl", [lambda: None, lambda m, x: None, lambda m, x, y=None: None]
)
def test_replace_callable_raises(any_string_dtype, repl):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over callables taking zero, two required, or two with a default parameters. Verifies a type error about the wrong number of positional arguments when used as the replacement.
    # GH 15055
    values = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # test with wrong number of arguments, raising an error
    msg = (
        r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
        r"(?(3)required )positional arguments?"
    )
    with pytest.raises(TypeError, match=msg):
        values.str.replace("a", repl, regex=True)


@pytest.mark.parametrize(
    "repl, expected_list",
    [
        (
            r"\g<three> \g<two> \g<one>",
            ["Three Two One", "Baz Bar Foo"],
        ),
        (
            r"\3 \2 \1",
            ["Three Two One", "Baz Bar Foo"],
        ),
        (
            r"\g<3> \g<2> \g<1>",
            ["Three Two One", "Baz Bar Foo"],
        ),
        (
            r"\g<2>0",
            ["Two0", "Bar0"],
        ),
        (
            r"\g<2>0 \1",
            ["Two0 One", "Bar0 Foo"],
        ),
    ],
    ids=[
        "named_groups_full_swap",
        "numbered_groups_no_g_full_swap",
        "numbered_groups_full_swap",
        "single_group_with_literal",
        "mixed_group_reference_with_literal",
    ],
)
@pytest.mark.parametrize("use_compile", [True, False])
def test_replace_named_groups_regex_swap(
    any_string_dtype, use_compile, repl, expected_list
):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over five replacement templates (named group references, numbered references with and without the group syntax, a group followed by a literal zero, mixed) with ids, and over compiled or uncompiled patterns. Verifies group references in the replacement swap the words of three-word strings.
    # GH#57636
    ser = Series(["One Two Three", "Foo Bar Baz"], dtype=any_string_dtype)
    pattern = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
    if use_compile:
        pattern = re.compile(pattern)
    result = ser.str.replace(pattern, repl, regex=True)
    expected = Series(expected_list, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "repl",
    [
        r"\g<20>",
        r"\20",
        r"\40",
        r"\4",
    ],
)
@pytest.mark.parametrize("use_compile", [True, False])
def test_replace_named_groups_regex_swap_expected_fail(
    any_string_dtype, repl, use_compile, request
):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over references to nonexistent groups (twenty with and without group syntax, forty, four) and compiled/uncompiled patterns, using the request fixture. For pyarrow-backed strings with an uncompiled pattern and plain numbered references (the library call goes to pyarrow directly) expects pyarrow's invalid error about too few parenthesized subexpressions, and marks the two-digit-twenty case as an expected failure because pyarrow reads it as group two plus a literal zero; otherwise expects the regex module's error about an invalid group reference.
    # GH#57636
    if (
        not use_compile
        and r"\g" not in repl
        and isinstance(any_string_dtype, StringDtype)
        and any_string_dtype.storage == "pyarrow"
    ):
        # calls pyarrow method directly
        if repl == r"\20":
            mark = pytest.mark.xfail(reason="PyArrow interprets as group + literal")
            request.applymarker(mark)

        pa = pytest.importorskip("pyarrow")
        error_type = pa.ArrowInvalid
        error_msg = r"only has \d parenthesized subexpressions"
    else:
        error_type = re.error
        error_msg = "invalid group reference"

    pattern = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
    if use_compile:
        pattern = re.compile(pattern)
    ser = Series(["One Two Three", "Foo Bar Baz"], dtype=any_string_dtype)

    with pytest.raises(error_type, match=error_msg):
        ser.str.replace(pattern, repl, regex=True)


@pytest.mark.parametrize(
    "pattern, repl",
    [
        (r"(\w+) (\w+) (\w+)", r"\20"),
        (r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)", r"\20"),
    ],
)
def test_pyarrow_ambiguous_group_references(pyarrow_string_dtype, pattern, repl):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over numbered and named patterns with the replacement referring to group two followed by zero (written as twenty). Uses the pyarrow string dtype fixture. Verifies pyarrow interprets this as group two followed by a literal zero.
    # GH#62653
    ser = Series(["One Two Three", "Foo Bar Baz"], dtype=pyarrow_string_dtype)

    result = ser.str.replace(pattern, repl, regex=True)
    expected = Series(["Two0", "Bar0"], dtype=pyarrow_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pattern, repl, expected_list",
    [
        (
            r"\[(?P<one>\d+)\]",
            r"(\1)",
            ["var.one(0)", "var.two(1)", "var.three(2)"],
        ),
        (
            r"\[(\d+)\]",
            r"(\1)",
            ["var.one(0)", "var.two(1)", "var.three(2)"],
        ),
    ],
)
@td.skip_if_no("pyarrow")
def test_pyarrow_backend_group_replacement(pattern, repl, expected_list):
    # [FUNCTION DOC]
    # Skipped without pyarrow. Parametrized over a named-group and a numbered-group pattern with a parenthesized group-one replacement. Verifies replacing bracketed numbers on Series converted to the pyarrow dtype backend.
    ser = Series(["var.one[0]", "var.two[1]", "var.three[2]"]).convert_dtypes(
        dtype_backend="pyarrow"
    )
    result = ser.str.replace(pattern, repl, regex=True)
    expected = Series(expected_list).convert_dtypes(dtype_backend="pyarrow")
    tm.assert_series_equal(result, expected)


def test_replace_callable_named_groups(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a callable replacement can read a named group from the match and returns its swapped-case middle word, with the missing value kept.
    # test regex named groups
    ser = Series(["Foo Bar Baz", np.nan], dtype=any_string_dtype)
    pat = r"(?P<first>\w+) (?P<middle>\w+) (?P<last>\w+)"
    repl = lambda m: m.group("middle").swapcase()
    result = ser.str.replace(pat, repl, regex=True)
    expected = Series(["bAR", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a compiled pattern works in replace, including the n argument.
    # GH 15446
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # test with compiled regex
    pat = re.compile(r"BAD_*")
    result = ser.str.replace(pat, "", regex=True)
    expected = Series(["foobar", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace(pat, "", n=1, regex=True)
    expected = Series(["foobarBAD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex_mixed_object():
    # [FUNCTION DOC]
    # Verifies a compiled pattern on an object Series with non-strings replaces in strings and gives missing for non-strings.
    pat = re.compile(r"BAD_*")
    ser = Series(
        ["aBAD", np.nan, "bBAD", True, datetime.today(), "fooBAD", None, 1, 2.0]
    )
    result = Series(ser).str.replace(pat, "", regex=True)
    expected = Series(
        ["a", np.nan, "b", np.nan, np.nan, "foo", None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex_unicode(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a compiled pattern with the unicode flag replaces a comma between word characters.
    ser = Series([b"abcd,\xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    expected = Series([b"abcd, \xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    pat = re.compile(r"(?<=\w),(?=\w)", flags=re.UNICODE)
    result = ser.str.replace(pat, ", ", regex=True)
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that passing flags, or either value of the case argument, together with a compiled pattern raises a value error saying case and flags cannot be set when the pattern is a compiled regex.
    # KNOWN ISSUE: the comment says the arguments have no effect and produce warnings, but the test asserts that errors are raised.
    # case and flags provided to str.replace will have no effect
    # and will produce warnings
    ser = Series(["fooBAD__barBAD__bad", np.nan], dtype=any_string_dtype)
    pat = re.compile(r"BAD_*")

    msg = "case and flags cannot be set when pat is a compiled regex"

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", flags=re.IGNORECASE, regex=True)

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", case=False, regex=True)

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", case=True, regex=True)


def test_replace_compiled_regex_callable(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a compiled pattern combined with a callable replacement and an n limit.
    # test with callable
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)
    repl = lambda m: m.group(0).swapcase()
    pat = re.compile("[a-z][A-Z]{2}")
    result = ser.str.replace(pat, repl, n=2, regex=True)
    expected = Series(["foObaD__baRbaD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("regex,expected_val", [(True, "bao"), (False, "foo")])
def test_replace_literal(regex, expected_val, any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over the regex flag. Verifies a dot is a wildcard when regex is true but literal when false.
    # GH16808 literal replace (regex=False vs regex=True)
    ser = Series(["f.o", "foo", np.nan], dtype=any_string_dtype)
    expected = Series(["bao", expected_val, np.nan], dtype=any_string_dtype)
    result = ser.str.replace("f.", "ba", regex=regex)
    tm.assert_series_equal(result, expected)


def test_replace_literal_callable_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a callable replacement with regex disabled raises a value error saying a callable cannot be used when regex is false.
    ser = Series([], dtype=any_string_dtype)
    repl = lambda m: m.group(0).swapcase()

    msg = "Cannot use a callable replacement when regex=False"
    with pytest.raises(ValueError, match=msg):
        ser.str.replace("abc", repl, regex=False)


def test_replace_literal_compiled_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a compiled pattern with regex disabled raises a value error saying a compiled regex cannot be used.
    ser = Series([], dtype=any_string_dtype)
    pat = re.compile("[a-z][A-Z]{2}")

    msg = "Cannot use a compiled regex as replacement pattern with regex=False"
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", regex=False)


def test_replace_moar(any_string_dtype):
    # [FUNCTION DOC]
    # From an old pull request. Verifies case-sensitive and case-insensitive literal replacement and a regex with alternation and anchors on a ten-element Series including empty and missing values.
    # PR #1179
    ser = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    result = ser.str.replace("A", "YYY")
    expected = Series(
        ["YYY", "B", "C", "YYYaba", "Baca", "", np.nan, "CYYYBYYY", "dog", "cat"],
        dtype=any_string_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("A", "YYY", case=False)
    expected = Series(
        [
            "YYY",
            "B",
            "C",
            "YYYYYYbYYY",
            "BYYYcYYY",
            "",
            np.nan,
            "CYYYBYYY",
            "dog",
            "cYYYt",
        ],
        dtype=any_string_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("^.a|dog", "XX-XX ", case=False, regex=True)
    expected = Series(
        [
            "A",
            "B",
            "C",
            "XX-XX ba",
            "XX-XX ca",
            "",
            np.nan,
            "XX-XX BA",
            "XX-XX ",
            "XX-XX t",
        ],
        dtype=any_string_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_replace_not_case_sensitive_not_regex(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies case-insensitive literal (non-regex) replacement treats the dot literally.
    # https://github.com/pandas-dev/pandas/issues/41602
    ser = Series(["A.", "a.", "Ab", "ab", np.nan], dtype=any_string_dtype)

    result = ser.str.replace("a", "c", case=False, regex=False)
    expected = Series(["c.", "c.", "cb", "cb", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("a.", "c.", case=False, regex=False)
    expected = Series(["c.", "c.", "Ab", "ab", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_regex(any_string_dtype):
    # [FUNCTION DOC]
    # From a pull request. Verifies a regex that matches exactly one character is replaced and an empty string is not matched.
    # https://github.com/pandas-dev/pandas/pull/24809
    s = Series(["a", "b", "ac", np.nan, ""], dtype=any_string_dtype)
    result = s.str.replace("^.$", "a", regex=True)
    expected = Series(["a", "a", "ac", np.nan, ""], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("regex", [True, False])
def test_replace_regex_single_character(regex, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over the regex flag. Regression test for GitHub issues. Verifies a single dot is a wildcard replacement when regex is true and literal when false.
    # https://github.com/pandas-dev/pandas/pull/24809, enforced in 2.0
    # GH 24804
    s = Series(["a.b", ".", "b", np.nan, ""], dtype=any_string_dtype)

    result = s.str.replace(".", "a", regex=regex)
    if regex:
        expected = Series(["aaa", "a", "a", np.nan, ""], dtype=any_string_dtype)
    else:
        expected = Series(["aab", "a", "b", np.nan, ""], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"a(?=b)", ["aa", "xb", "ba", "bb"]),
        (r"(?<=a)b", ["aa", "ax", "ba", "bb"]),
        (r"a(?!b)", ["xx", "ab", "bx", "bb"]),
        (r"(?<!b)a", ["xx", "xb", "ba", "bb"]),
        ("ab", ["aa", "x", "ba", "bb"]),
    ],
)
def test_replace_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookaround patterns and a plain pattern with expected replaced strings. Verifies the replacement of the pattern by a letter, with the missing value as None, NaN or NA by dtype.
    # https://github.com/pandas-dev/pandas/issues/60833
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.replace(pat, "x", regex=True)
    if any_string_dtype == "object":
        null_result = None
    elif any_string_dtype == "str":
        null_result = np.nan
    elif any_string_dtype == "string":
        null_result = pd.NA
    else:
        raise ValueError(f"Unrecognized dtype: {any_string_dtype}")
    expected = Series([*expected_data, null_result], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies end-of-string anchor behavior in replace: dollar matches before a trailing newline for Python-based dtypes only; the end-of-text escape matches only at the very end; a literal backslash-Z sequence can still be replaced.
    # https://github.com/pandas-dev/pandas/pull/63613
    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.replace("bar$", "x", regex=True)
    if any_string_dtype == "string" and any_string_dtype.storage == "pyarrow":
        # pyarrow (RE2) only matches $ at the very end of the line
        expected = Series(["baz", "x", "bars", "bar\n"], dtype=any_string_dtype)
    else:
        # python matches $ before or after an ending newline
        expected = Series(["baz", "x", "bars", "x\n"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.replace(r"bar\Z", "x", regex=True)
    expected = Series(["baz", "x", "bars", "bar\n"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"bar\Z", "bar", "bars", "bar\n"], dtype=any_string_dtype)
    result = ser.str.replace(r"bar\\Z", "x", regex=True)
    expected = Series(["x", "bar", "bars", "bar\n"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_empty_pattern(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies replacing an empty pattern with an empty replacement leaves the text, with a letter inserts it between every character and at both ends, and works on an empty Series.
    # https://github.com/pandas-dev/pandas/issues/64941
    ser = Series(["abcd"], dtype=any_string_dtype)

    result = ser.str.replace("", "")
    expected = Series(["abcd"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("", "X")
    expected = Series(["XaXbXcXdX"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    ser = Series([], dtype=any_string_dtype)
    result = ser.str.replace("", "X")
    expected = Series([], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.match
# --------------------------------------------------------------------------------------


def test_match(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies matching anchors at the start only and handles groups, with missing values as NaN (object), NA (nullable) or false (inferred string), on several data sets including a caret in the text, which must be escaped to match literally.
    if any_string_dtype == "str":
        # NaN propagates as False
        expected_dtype = bool
        na_value = False
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        na_value = np.nan

    values = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    result = values.str.match(".*(BAD[_]+).*(BAD)")
    expected = Series([True, na_value, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = values.str.match(".*BAD[_]+.*BAD")
    expected = Series([True, True, na_value, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = values.str.match("BAD[_]+.*BAD")
    expected = Series([False, True, na_value, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        ["fooBAD__barBAD", "^BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = values.str.match("^BAD[_]+.*BAD")
    expected = Series([False, False, na_value, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = values.str.match("\\^BAD[_]+.*BAD")
    expected = Series([False, True, na_value, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat",
    [r"^foo|bar", r"^bar|foo", r"^foo|bar$", r"^(foo)|bar"],
)
def test_match_anchored_alternation(any_string_dtype, pat):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over four patterns combining anchors and alternation. Verifies that a start anchor applies only to the alternative it belongs to, so strings starting with the prefix or exactly containing the other alternative at the start match.
    # GH#66069
    ser = Series(["xbar", "bar", "foo", "xfoo"], dtype=any_string_dtype)
    result = ser.str.match(pat)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series([False, True, True, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_match_mixed_object():
    # [FUNCTION DOC]
    # Verifies match on an object Series with non-strings gives booleans for strings, NaN for others and None for None.
    mixed = Series(
        [
            "aBAD_BAD",
            np.nan,
            "BAD_b_BAD",
            True,
            datetime.today(),
            "foo",
            None,
            1,
            2.0,
        ]
    )
    result = Series(mixed).str.match(".*(BAD[_]+).*(BAD)")
    expected = Series([True, np.nan, True, np.nan, np.nan, False, None, np.nan, np.nan])
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)


def test_match_na_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies the na argument false gives plain booleans with false for missing, and the default gives the dtype-appropriate missing value.
    # GH #6609
    s = Series(["a", "b", np.nan], dtype=any_string_dtype)

    result = s.str.match("a", na=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series([True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = s.str.match("a")
    if any_string_dtype == "str":
        # NaN propagates as False
        expected_dtype = bool
        na_value = False
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        na_value = np.nan

    expected = Series([True, False, na_value], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_match_case_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies case-insensitive matching.
    values = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)
    result = values.str.match("ab", case=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series([True, True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_match_compiled_regex(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a compiled pattern is accepted; a conflicting case argument or flags argument with a compiled pattern raises a value error naming the conflict; consistent flags are accepted.
    # GH#61952
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    values = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)

    result = values.str.match(re.compile("ab"))
    expected = Series([True, False, True, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    msg = (
        "Cannot both specify 'case' and pass a compiled "
        "regexp object with conflicting case-sensitivity"
    )
    with pytest.raises(ValueError, match=msg):
        values.str.match(re.compile("ab"), case=False)

    result = values.str.match(re.compile("ab", flags=re.IGNORECASE))
    expected = Series([True, True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    msg = (
        "Cannot both specify 'flags' and pass a compiled "
        "regexp object with conflicting flags"
    )
    with pytest.raises(ValueError, match=msg):
        values.str.match(re.compile("ab"), flags=re.IGNORECASE)

    # But if the flags match you're OK
    values.str.match(re.compile("ab", flags=re.IGNORECASE), flags=re.IGNORECASE)


@pytest.mark.parametrize(
    "pat, case, exp",
    [
        ["ab", False, [True, False]],
        ["Ab", True, [False, False]],
        ["bc", True, [False, False]],
        ["a[a-z]{1}", False, [True, False]],
        ["A[a-z]{1}", True, [False, False]],
        # https://github.com/pandas-dev/pandas/issues/61072
        ["(bc)|(ab)", True, [True, False]],
        ["((bc)|(ab))", True, [True, False]],
    ],
)
def test_str_match_extra_cases(any_string_dtype, pat, case, exp):
    # [FUNCTION DOC]
    # Parametrized over patterns, case flags and expected booleans on two strings, including patterns with alternation and groups (regression for a GitHub issue). Verifies case handling and that matching is anchored at the start.
    ser = Series(["abc", "Xab"], dtype=any_string_dtype)
    result = ser.str.match(pat, case=case)

    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(exp, dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"a(?=b)", [False, True, False, False]),
        (r"(?<=a)b", [False, False, False, False]),
        (r"a(?!b)", [True, False, False, False]),
        (r"(?<!b)a", [True, True, False, False]),
        ("ab", [False, True, False, False]),
    ],
)
def test_match_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookaround patterns with expected booleans. Verifies match results and that the missing value is None (object), false (inferred string) or NA (nullable); raises a value error for any other dtype.
    # https://github.com/pandas-dev/pandas/issues/60833
    if any_string_dtype == "object":
        expected_dtype, null_result = "object", None
    elif any_string_dtype == "str":
        expected_dtype, null_result = "bool", False
    elif any_string_dtype == "string":
        expected_dtype, null_result = "boolean", pd.NA
    else:
        raise ValueError(f"Unrecognized dtype: {any_string_dtype}")
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.match(pat)
    expected = Series([*expected_data, null_result], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_match_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies end-of-string anchor behavior with match as in the contains version: dollar before trailing newline differs for pyarrow-backed strings; the end-of-text escape matches only at the very end; literal backslash-Z still matches.
    # https://github.com/pandas-dev/pandas/pull/63613
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.match("bar$")
    if any_string_dtype == "string" and any_string_dtype.storage == "pyarrow":
        # pyarrow (RE2) only matches $ at the very end of the line
        expected = Series([False, True, False, False], dtype=expected_dtype)
    else:
        # python matches $ before or after an ending newline
        expected = Series([False, True, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.match(r"bar\Z")
    expected = Series([False, True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"bar\Z", "bar", "bars", "bar\n"], dtype=any_string_dtype)
    result = ser.str.match(r"bar\\Z")
    expected = Series([True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.fullmatch
# --------------------------------------------------------------------------------------


def test_fullmatch(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a full match requires the whole string to match; missing values give NaN, NA or false by dtype.
    # GH 32806
    ser = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = ser.str.fullmatch(".*BAD[_]+.*BAD")
    if any_string_dtype == "str":
        # NaN propagates as False
        expected = Series([True, False, False, False], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series([True, False, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_fullmatch_dollar_literal(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies an escaped dollar in the pattern matches a literal dollar at the end of the whole string only.
    # GH 56652
    ser = Series(["foo", "foo$foo", np.nan, "foo$"], dtype=any_string_dtype)
    result = ser.str.fullmatch("foo\\$")
    if any_string_dtype == "str":
        # NaN propagates as False
        expected = Series([False, False, False, True], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series([False, False, np.nan, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_fullmatch_na_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the na argument false gives plain booleans with false for missing.
    ser = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = ser.str.fullmatch(".*BAD[_]+.*BAD", na=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series([True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_fullmatch_case_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies case-sensitive and case-insensitive full match, and that the ignore-case flag gives the same result as the case argument.
    ser = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    expected = Series([True, False, False, False], dtype=expected_dtype)

    result = ser.str.fullmatch("ab", case=True)
    tm.assert_series_equal(result, expected)

    expected = Series([True, True, False, False], dtype=expected_dtype)

    result = ser.str.fullmatch("ab", case=False)
    tm.assert_series_equal(result, expected)

    result = ser.str.fullmatch("ab", flags=re.IGNORECASE)
    tm.assert_series_equal(result, expected)


def test_fullmatch_compiled_regex(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Same scenarios as for contains with a compiled pattern: accepted; the case argument works on pyarrow-backed strings but raises a value error about flags elsewhere; a case-insensitive compiled pattern works; the flags argument with a compiled pattern raises.
    # KNOWN ISSUE: the pyarrow-backed dtype accepts the case argument with a compiled pattern while the other dtypes raise; the test encodes the inconsistency and carries a to-do comment.
    # GH#61952
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    values = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)

    result = values.str.fullmatch(re.compile("ab"))
    expected = Series([True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # TODO this currently works for pyarrow-backed dtypes but raises for python
    if any_string_dtype == "string" and any_string_dtype.storage == "pyarrow":
        result = values.str.fullmatch(re.compile("ab"), case=False)
        expected = Series([True, True, False, False], dtype=expected_dtype)
        tm.assert_series_equal(result, expected)
    else:
        with pytest.raises(
            ValueError, match="cannot process flags argument with a compiled pattern"
        ):
            values.str.fullmatch(re.compile("ab"), case=False)

    result = values.str.fullmatch(re.compile("ab", flags=re.IGNORECASE))
    expected = Series([True, True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    with pytest.raises(
        ValueError, match="cannot process flags argument with a compiled pattern"
    ):
        values.str.fullmatch(re.compile("ab"), flags=re.IGNORECASE)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    # Note: keep cases in sync with
    # pandas/tests/extension/test_arrow.py::test_str_fullmatch
    [
        ["abc", False, None, [True, False, False, None]],
        ["Abc", True, None, [False, False, False, None]],
        ["bc", True, None, [False, False, False, None]],
        ["ab", False, None, [False, False, False, None]],
        ["a[a-z]{2}", False, None, [True, False, False, None]],
        ["A[a-z]{1}", True, None, [False, False, False, None]],
        # GH Issue: #56652
        ["abc$", False, None, [True, False, False, None]],
        ["abc\\$", False, None, [False, True, False, None]],
        ["Abc$", True, None, [False, False, False, None]],
        ["Abc\\$", True, None, [False, False, False, None]],
        # https://github.com/pandas-dev/pandas/issues/61072
        ["(abc)|(abx)", True, None, [True, False, False, None]],
        ["((abc)|(abx))", True, None, [True, False, False, None]],
    ],
)
def test_str_fullmatch_extra_cases(any_string_dtype, pat, case, na, exp):
    # [FUNCTION DOC]
    # Parametrized over patterns, case flag, na and expected values (kept in sync with a pyarrow extension test), including patterns with dollar, escaped dollar and alternation (regression for GitHub issues). Verifies the full match results for strings with and without dollar signs and a missing value; for the inferred string dtype the missing entry is changed to false in the expected list.
    ser = Series(["abc", "abc$", "$abc", None], dtype=any_string_dtype)
    result = ser.str.fullmatch(pat, case=case, na=na)

    if any_string_dtype == "str":
        # NaN propagates as False
        exp[-1] = False
        expected_dtype = bool
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
    expected = Series(exp, dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat",
    [(r"a(?=b)"), (r"(?<=a)b"), (r"a(?!b)"), (r"(?<!b)a"), ("ab")],
)
def test_fullmatch_lookarounds(any_string_dtype, pat):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookaround patterns and a plain pattern. Verifies that any pattern containing a lookaround never fully matches (only the plain two-letter pattern matches the matching string), with the missing value None, false or NA by dtype.
    # https://github.com/pandas-dev/pandas/issues/60833
    # Note: By definition, any match with a lookaround is not a full match.
    if any_string_dtype == "object":
        expected_dtype, null_result = "object", None
    elif any_string_dtype == "str":
        expected_dtype, null_result = "bool", False
    elif any_string_dtype == "string":
        expected_dtype, null_result = "boolean", pd.NA
    else:
        raise ValueError(f"Unrecognized dtype: {any_string_dtype}")
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.fullmatch(pat)
    expected = Series(
        [False, True if pat == "ab" else False, False, False, null_result],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_fullmatch_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies end-of-string anchors with full match behave the same for Python and pyarrow strings: dollar and the end-of-text escape match only strings that end right after the text; a literal backslash-Z still matches.
    # https://github.com/pandas-dev/pandas/pull/63613
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )

    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign (for fullmatch, no difference between python and pyarrow)
    result = ser.str.fullmatch("bar$")
    expected = Series([False, True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.fullmatch(r"bar\Z")
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"bar\Z", "bar", "bars", "bar\n"], dtype=any_string_dtype)
    result = ser.str.fullmatch(r"bar\\Z")
    expected = Series([True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.findall
# --------------------------------------------------------------------------------------


def test_findall(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies find-all returns a list of matches per element, empty list for no match, and the dtype-appropriate missing value for missing input.
    ser = Series(["fooBAD__barBAD", np.nan, "foo", "BAD"], dtype=any_string_dtype)
    result = ser.str.findall("BAD[_]*")
    expected = Series([["BAD__", "BAD"], np.nan, [], ["BAD"]])
    expected = _convert_na_value(ser, expected)
    tm.assert_series_equal(result, expected)


def test_findall_mixed_object():
    # [FUNCTION DOC]
    # Verifies find-all on an object Series with non-strings gives lists for strings, NaN for non-strings and None for None.
    ser = Series(
        [
            "fooBAD__barBAD",
            np.nan,
            "foo",
            True,
            datetime.today(),
            "BAD",
            None,
            1,
            2.0,
        ]
    )

    result = ser.str.findall("BAD[_]*")
    expected = Series(
        [
            ["BAD__", "BAD"],
            np.nan,
            [],
            np.nan,
            np.nan,
            ["BAD"],
            None,
            np.nan,
            np.nan,
        ]
    )

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"a(?=b)", [[], ["a"], [], []]),
        (r"(?<=a)b", [[], ["b"], [], []]),
        (r"a(?!b)", [["a", "a"], [], ["a"], []]),
        (r"(?<!b)a", [["a", "a"], ["a"], [], []]),
        ("ab", [[], ["ab"], [], []]),
    ],
)
def test_findall_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookaround patterns with expected lists. Verifies matches, with the missing value being None, NaN or NA by dtype.
    # https://github.com/pandas-dev/pandas/issues/60833
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.findall(pat)
    if any_string_dtype == "object":
        null_result = None
    elif any_string_dtype == "str":
        null_result = np.nan
    elif any_string_dtype == "string":
        null_result = pd.NA
    else:
        raise ValueError(f"Unrecognized dtype: {any_string_dtype}")
    expected = Series([*expected_data, null_result])
    tm.assert_series_equal(result, expected)


def test_findall_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies find-all with the dollar anchor matches before a trailing newline, the end-of-text escape matches only at the end, and a literal backslash-Z sequence is found.
    # https://github.com/pandas-dev/pandas/pull/63613
    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.findall("bar$")
    expected = Series([[], ["bar"], [], ["bar"]], dtype=object)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.findall(r"bar\Z")
    expected = Series([[], ["bar"], [], []], dtype=object)
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"bar\Z", "bar", "bars", "bar\n"], dtype=any_string_dtype)
    result = ser.str.findall(r"bar\\Z")
    expected = Series([["bar\\Z"], [], [], []], dtype=object)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.find
# --------------------------------------------------------------------------------------


def test_find(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies find and right find with no bounds, a start, and start and end, on five strings (one without a match gives minus one); compares results with the Python standard library per element.
    ser = Series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF", "XXXX"], dtype=any_string_dtype
    )
    expected_dtype = (
        np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )

    result = ser.str.find("EF")
    expected = Series([4, 3, 1, 0, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.find("EF") for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.rfind("EF")
    expected = Series([4, 5, 7, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rfind("EF") for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.find("EF", 3)
    expected = Series([4, 3, 7, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.find("EF", 3) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.rfind("EF", 3)
    expected = Series([4, 5, 7, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rfind("EF", 3) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.find("EF", 3, 6)
    expected = Series([4, 3, -1, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.find("EF", 3, 6) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.rfind("EF", 3, 6)
    expected = Series([4, 3, -1, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rfind("EF", 3, 6) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)


def test_find_bad_arg_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a non-string argument raises a type error saying a string object was expected, not int, for find and right find.
    ser = Series([], dtype=any_string_dtype)
    with pytest.raises(TypeError, match="expected a string object, not int"):
        ser.str.find(0)

    with pytest.raises(TypeError, match="expected a string object, not int"):
        ser.str.rfind(0)


def test_find_nan(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies find and right find with a missing value, giving NaN (float result) or NA (nullable integer) for the missing element and integers elsewhere.
    ser = Series(
        ["ABCDEFG", np.nan, "DEFGHIJEF", np.nan, "XXXX"], dtype=any_string_dtype
    )
    if is_object_or_nan_string_dtype(any_string_dtype):
        expected_dtype = np.float64
        item = np.nan
    else:
        expected_dtype = "Int64"
        item = pd.NA

    result = ser.str.find("EF")
    expected = Series([4, item, 1, item, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rfind("EF")
    expected = Series([4, item, 7, item, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.find("EF", 3)
    expected = Series([4, item, 7, item, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rfind("EF", 3)
    expected = Series([4, item, 7, item, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.find("EF", 3, 6)
    expected = Series([4, item, -1, item, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rfind("EF", 3, 6)
    expected = Series([4, item, -1, item, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_find_multibyte_chars(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies find returns character offsets, not byte offsets, for strings starting with multibyte characters (accented letter, CJK, emoji), also with a start argument.
    # GH#64123 - str.find should return character offsets, not byte offsets
    ser = Series(["ga", "Áa", "永a", "🐍a"], dtype=any_string_dtype)
    expected_dtype = (
        np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )

    result = ser.str.find("a")
    expected = Series([1, 1, 1, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.find("a", start=1)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.translate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
def test_translate(index_or_series, any_string_dtype, infer_string):
    # [FUNCTION DOC]
    # Parametrized over the infer-string flag (the true case requires pyarrow), with the Series-or-Index fixture and string dtype fixture. Verifies translating characters through a translation table gives the expected strings of the same container type and dtype.
    # KNOWN ISSUE: the infer-string parameter is accepted but never used in the body, so both parametrizations run identical checks.
    obj = index_or_series(
        ["abcdefg", "abcc", "cdddfg", "cdefggg"], dtype=any_string_dtype
    )
    table = str.maketrans("abc", "cde")
    result = obj.str.translate(table)
    expected = index_or_series(
        ["cdedefg", "cdee", "edddfg", "edefggg"], dtype=any_string_dtype
    )
    tm.assert_equal(result, expected)


def test_translate_mixed_object():
    # [FUNCTION DOC]
    # Verifies translate on an object Series with a non-string element gives translated strings and NaN for the non-string.
    # Series with non-string values
    s = Series(["a", "b", "c", 1.2])
    table = str.maketrans("abc", "cde")
    expected = Series(["c", "d", "e", np.nan], dtype=object)
    result = s.str.translate(table)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------


def test_flags_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the regular-expression flags argument (ignore case) is honored by extract, match, full match, find-all and count on an email-address pattern, and that contains with a pattern with capture groups emits a user warning that the pattern has match groups.
    data = {
        "Dave": "dave@google.com",
        "Steve": "steve@gmail.com",
        "Rob": "rob@gmail.com",
        "Wes": np.nan,
    }
    data = Series(data, dtype=any_string_dtype)

    pat = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"

    result = data.str.extract(pat, flags=re.IGNORECASE, expand=True)
    assert result.iloc[0].tolist() == ["dave", "google", "com"]

    result = data.str.match(pat, flags=re.IGNORECASE)
    assert result.iloc[0]

    result = data.str.fullmatch(pat, flags=re.IGNORECASE)
    assert result.iloc[0]

    result = data.str.findall(pat, flags=re.IGNORECASE)
    assert result.iloc[0][0] == ("dave", "google", "com")

    result = data.str.count(pat, flags=re.IGNORECASE)
    assert result.iloc[0] == 1

    msg = "has match groups"
    with tm.assert_produces_warning(UserWarning, match=msg):
        result = data.str.contains(pat, flags=re.IGNORECASE)
    assert result.iloc[0]
