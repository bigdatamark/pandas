# [MODULE DOC]
# Tests for the general string accessor methods on Series and Index (count, repeat, character-class predicates, length, index lookup, slicing, stripping, prefix/suffix removal, encode/decode, normalize, zero-fill, accessor validation, per-storage setitem) across the parametrized string dtype fixtures. Uses the helper that identifies object or NaN-based string dtypes to decide whether results are numpy-based (NaN for missing) or nullable (NA for missing). Also tests a regex-feature detector used by the Arrow string array mixin.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import pa_version_under21p0

from pandas import (
    NA,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    StringDtype,
    option_context,
)
import pandas._testing as tm
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import is_object_or_nan_string_dtype


@pytest.mark.parametrize("pattern", [0, True, Series(["foo", "bar"])])
def test_startswith_endswith_non_str_patterns(pattern):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over an integer, a boolean and a Series as the pattern. Verifies both starts-with and ends-with raise a type error saying a string or tuple was expected, naming the given type.
    # GH3485
    ser = Series(["foo", "bar"])
    msg = f"expected a string or tuple, not {type(pattern).__name__}"
    with pytest.raises(TypeError, match=msg):
        ser.str.startswith(pattern)
    with pytest.raises(TypeError, match=msg):
        ser.str.endswith(pattern)


def test_iter_raises():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies iterating over the string accessor object raises a type error saying it is not iterable.
    # GH 54173
    ser = Series(["foo", "bar"])
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)


# test integer/float dtypes (inferred by constructor) and mixed


def test_count(any_string_dtype):
    # [FUNCTION DOC]
    # Uses the string dtype fixture. Verifies counting a regex pattern matches per element gives the expected counts, with a float result and NaN for missing on object-like dtypes and nullable Int64 with NA otherwise.
    ser = Series(["foo", "foofoo", np.nan, "foooofooofommmfoo"], dtype=any_string_dtype)
    result = ser.str.count("f[o]+")
    if is_object_or_nan_string_dtype(any_string_dtype):
        expected_dtype = np.float64
        item = np.nan
    else:
        expected_dtype = "Int64"
        item = NA

    expected = Series([1, 2, item, 4], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_count_mixed_object():
    # [FUNCTION DOC]
    # Verifies counting on an object Series containing non-strings (boolean, datetime, number, None, NaN) gives counts for strings and NaN for all non-strings.
    ser = Series(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    result = ser.str.count("a")
    expected = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"a(?=b)", [0, 1, 0, 0, None]),
        (r"(?<=a)b", [0, 1, 0, 0, None]),
        (r"a(?!b)", [2, 0, 1, 0, None]),
        (r"(?<!b)a", [2, 1, 0, 0, None]),
        ("ab", [0, 1, 0, 0, None]),
    ],
)
def test_count_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookahead and lookbehind patterns (positive and negative) and a plain pattern with expected counts. Verifies counts on a Series with a missing value, with float result for object-like dtypes and Int64 otherwise.
    # https://github.com/pandas-dev/pandas/issues/60833
    expected_dtype = (
        "float64" if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.count(pat)
    expected = Series(expected_data, dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_count_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies end-of-string handling in patterns: a dollar anchor matches before a trailing newline for Python-based dtypes but only at the very end for pyarrow-backed strings; the end-of-text escape matches only at the very end for all dtypes (it must be translated for pyarrow); and a literal backslash followed by the escape letter is still searchable.
    # https://github.com/pandas-dev/pandas/pull/63613
    expected_dtype = (
        "int64" if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )

    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.count("bar$")
    if any_string_dtype == "string" and any_string_dtype.storage == "pyarrow":
        # pyarrow (RE2) only matches $ at the very end of the line
        expected = Series([0, 1, 0, 0], dtype=expected_dtype)
    else:
        # python matches $ before or after an ending newline
        expected = Series([0, 1, 0, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.count(r"bar\Z")
    expected = Series([0, 1, 0, 0], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"bar\Z", "bar", "bars", "bar\n"], dtype=any_string_dtype)
    result = ser.str.count(r"bar\\Z")
    expected = Series([1, 0, 0, 0], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_repeat(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies repeating with a scalar count and with a per-element list of counts, preserving the missing values and the dtype.
    ser = Series(["a", "b", np.nan, "c", np.nan, "d"], dtype=any_string_dtype)

    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", "bbb", np.nan, "ccc", np.nan, "ddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.repeat([1, 2, 3, 4, 5, 6])
    expected = Series(
        ["a", "bb", np.nan, "cccc", np.nan, "dddddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_repeat_mixed_object():
    # [FUNCTION DOC]
    # Verifies repeating on an object Series with non-strings gives repeated strings for strings and NaN (or None for the None element) for the rest.
    ser = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])
    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", np.nan, "bbb", np.nan, np.nan, "foofoofoo", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("arg, repeat", [[None, 4], ["b", None]])
def test_repeat_with_null(any_string_dtype, arg, repeat):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over a missing element value and a missing repeat count. Verifies repeating with a list of counts yields a missing value in the position where either the element or the count is missing.
    # GH: 31632
    ser = Series(["a", arg], dtype=any_string_dtype)
    result = ser.str.repeat([3, repeat])
    expected = Series(["aaa", None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_empty_str_methods(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue about extract. Verifies that for an empty Series of the given dtype every string method returns the right empty result type: empty string Series, integer Series (numpy or Int64 per dtype family), boolean Series, object Series, an empty frame for partition and dummies, frames with the right number of columns for extract, inferred-string for decode, object for encode, and that concatenation of nothing returns an empty string.
    empty_str = empty = Series(dtype=any_string_dtype)
    empty_inferred_str = Series(dtype="str")
    if is_object_or_nan_string_dtype(any_string_dtype):
        empty_int = Series(dtype="int64")
        empty_bool = Series(dtype=bool)
    else:
        empty_int = Series(dtype="Int64")
        empty_bool = Series(dtype="boolean")
    empty_object = Series(dtype=object)
    empty_bytes = Series(dtype=object)
    empty_df = DataFrame()

    # GH7241
    # (extract) on empty series

    tm.assert_series_equal(empty_str, empty.str.cat(empty))
    assert "" == empty.str.cat()
    tm.assert_series_equal(empty_str, empty.str.title())
    tm.assert_series_equal(empty_int, empty.str.count("a"))
    tm.assert_series_equal(empty_bool, empty.str.contains("a"))
    tm.assert_series_equal(empty_bool, empty.str.startswith("a"))
    tm.assert_series_equal(empty_bool, empty.str.endswith("a"))
    tm.assert_series_equal(empty_str, empty.str.lower())
    tm.assert_series_equal(empty_str, empty.str.upper())
    tm.assert_series_equal(empty_str, empty.str.replace("a", "b"))
    tm.assert_series_equal(empty_str, empty.str.repeat(3))
    tm.assert_series_equal(empty_bool, empty.str.match("^a"))
    tm.assert_frame_equal(
        DataFrame(columns=range(1), dtype=any_string_dtype),
        empty.str.extract("()", expand=True),
    )
    tm.assert_frame_equal(
        DataFrame(columns=range(2), dtype=any_string_dtype),
        empty.str.extract("()()", expand=True),
    )
    tm.assert_series_equal(empty_str, empty.str.extract("()", expand=False))
    tm.assert_frame_equal(
        DataFrame(columns=range(2), dtype=any_string_dtype),
        empty.str.extract("()()", expand=False),
    )
    tm.assert_frame_equal(empty_df.set_axis([], axis=1), empty.str.get_dummies())
    tm.assert_series_equal(empty_str, empty_str.str.join(""))
    tm.assert_series_equal(empty_int, empty.str.len())
    tm.assert_series_equal(empty_object, empty_str.str.findall("a"))
    tm.assert_series_equal(empty_int, empty.str.find("a"))
    tm.assert_series_equal(empty_int, empty.str.rfind("a"))
    tm.assert_series_equal(empty_str, empty.str.pad(42))
    tm.assert_series_equal(empty_str, empty.str.center(42))
    tm.assert_series_equal(empty_object, empty.str.split("a"))
    tm.assert_series_equal(empty_object, empty.str.rsplit("a"))
    tm.assert_series_equal(empty_object, empty.str.partition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.partition("a"))
    tm.assert_series_equal(empty_object, empty.str.rpartition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.rpartition("a"))
    tm.assert_series_equal(empty_str, empty.str.slice(stop=1))
    tm.assert_series_equal(empty_str, empty.str.slice(step=1))
    tm.assert_series_equal(empty_str, empty.str.strip())
    tm.assert_series_equal(empty_str, empty.str.lstrip())
    tm.assert_series_equal(empty_str, empty.str.rstrip())
    tm.assert_series_equal(empty_str, empty.str.wrap(42))
    tm.assert_series_equal(empty_str, empty.str.get(0))
    tm.assert_series_equal(empty_inferred_str, empty_bytes.str.decode("ascii"))
    tm.assert_series_equal(empty_bytes, empty.str.encode("ascii"))
    # ismethods should always return boolean (GH 29624)
    tm.assert_series_equal(empty_bool, empty.str.isalnum())
    tm.assert_series_equal(empty_bool, empty.str.isalpha())
    tm.assert_series_equal(empty_bool, empty.str.isascii())
    tm.assert_series_equal(empty_bool, empty.str.isdigit())
    tm.assert_series_equal(empty_bool, empty.str.isspace())
    tm.assert_series_equal(empty_bool, empty.str.islower())
    tm.assert_series_equal(empty_bool, empty.str.isupper())
    tm.assert_series_equal(empty_bool, empty.str.istitle())
    tm.assert_series_equal(empty_bool, empty.str.isnumeric())
    tm.assert_series_equal(empty_bool, empty.str.isdecimal())
    tm.assert_series_equal(empty_str, empty.str.capitalize())
    tm.assert_series_equal(empty_str, empty.str.swapcase())
    tm.assert_series_equal(empty_str, empty.str.normalize("NFC"))

    table = str.maketrans("a", "b")
    tm.assert_series_equal(empty_str, empty.str.translate(table))


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isascii", [True, True, True, True, True, True, True, True, True, True]),
        ("isalnum", [True, True, True, True, True, False, True, True, False, False]),
        ("isalpha", [True, True, True, False, False, False, True, False, False, False]),
        (
            "isdigit",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isnumeric",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isspace",
            [False, False, False, False, False, False, False, False, False, True],
        ),
        (
            "islower",
            [False, True, False, False, False, False, False, False, False, False],
        ),
        (
            "isupper",
            [True, False, False, False, True, False, True, False, False, False],
        ),
        (
            "istitle",
            [True, False, True, False, True, False, False, False, False, False],
        ),
    ],
)
def test_ismethods(method, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over the nine character-class predicate method names (ascii, alphanumeric, alpha, digit, numeric, space, lower, upper, title) with expected lists for a ten-element sample. Verifies the result values and dtype (plain bool for object-like dtypes, boolean for nullable), that the results equal the Python standard library results per element, then sets four elements to NaN and recomputes the result.
    # KNOWN ISSUE: the final section builds an expected result for the missing-value case in three branches by dtype but never compares it to the result, so the missing-value behavior is not actually asserted.
    ser = Series(
        ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "], dtype=any_string_dtype
    )
    expected_dtype = (
        "bool" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # compare with standard library
    expected_stdlib = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib

    # with missing value
    ser.iloc[[1, 2, 3, 4]] = np.nan
    result = getattr(ser.str, method)()
    if ser.dtype == "object":
        expected = expected.astype(object)
        expected.iloc[[1, 2, 3, 4]] = np.nan
    elif ser.dtype == "str":
        # NaN propagates as False
        expected.iloc[[1, 2, 3, 4]] = False
    else:
        # nullable dtypes propagate NaN
        expected.iloc[[1, 2, 3, 4]] = np.nan


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, True, True, True, False, True, True, False]),
        ("isdecimal", [False, True, False, False, False, False, True, False]),
        ("isdigit", [False, True, True, False, False, False, True, False]),
    ],
)
def test_isnumeric_unicode(method, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over numeric, decimal and digit predicates with expected lists for a sample of unicode characters (superscript three, vulgar fraction, star, Ethiopic number, fullwidth digit, text). Verifies values and dtype; for the digit predicate on pyarrow-backed strings with a recent pyarrow, two positions are expected true because pyarrow differs from Python; for object and Python-storage dtypes also compares with the Python standard library results.
    # 0x00bc: ¼ VULGAR FRACTION ONE QUARTER
    # 0x2605: ★ not number
    # 0x1378: ፸ ETHIOPIC NUMBER SEVENTY
    # 0xFF13: ３ Em 3  # noqa: RUF003
    ser = Series(
        ["A", "3", "³", "¼", "★", "፸", "３", "four"],  # noqa: RUF001
        dtype=any_string_dtype,
    )
    expected_dtype = (
        "bool" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(expected, dtype=expected_dtype)
    if (
        method == "isdigit"
        and isinstance(ser.dtype, StringDtype)
        and ser.dtype.storage == "pyarrow"
        and not pa_version_under21p0
    ):
        # known difference in behavior between python and pyarrow unicode handling
        # pyarrow 21+ considers ¼ and ፸ as a digit, while python does not
        expected.iloc[3] = True
        expected.iloc[5] = True

    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # compare with standard library
    # (only for non-pyarrow storage given the above differences)
    if any_string_dtype == "object" or (
        isinstance(any_string_dtype, StringDtype)
        and any_string_dtype.storage == "python"
    ):
        expected = [getattr(item, method)() for item in ser]
        assert list(result) == expected


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, np.nan, True, False, np.nan, True, False]),
        ("isdecimal", [False, np.nan, False, False, np.nan, True, False]),
    ],
)
def test_isnumeric_unicode_missing(method, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over numeric and decimal predicates. With a sample including NaN, verifies results: for the inferred string dtype missing counts as false in a boolean result; for object-like dtypes an object Series with NaN; otherwise boolean with missing.
    values = ["A", np.nan, "¼", "★", np.nan, "３", "four"]  # noqa: RUF001
    ser = Series(values, dtype=any_string_dtype)
    if any_string_dtype == "str":
        # NaN propagates as False
        expected = Series(expected, dtype=object).fillna(False).astype(bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)


def test_split_join_roundtrip(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies splitting by underscore then joining with underscore returns the original strings, as object dtype with the missing value kept.
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = ser.str.split("_").str.join("_")
    expected = ser.astype(object)
    tm.assert_series_equal(result, expected)


def test_split_join_roundtrip_mixed_object():
    # [FUNCTION DOC]
    # Verifies the split-then-join roundtrip on an object Series with non-string values returns strings for strings and missing values for non-strings.
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.split("_").str.join("_")
    expected = Series(
        ["a_b", np.nan, "asdf_cas_asdf", np.nan, np.nan, "foo", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_len(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the string length (counting characters, including a multi-byte character and trailing newline) with float and NaN for object-like dtypes and Int64 with NA otherwise.
    ser = Series(
        ["foo", "fooo", "fooooo", np.nan, "fooooooo", "foo\n", "あ"],
        dtype=any_string_dtype,
    )
    result = ser.str.len()
    if is_object_or_nan_string_dtype(any_string_dtype):
        expected_dtype = "float64"
        item = np.nan
    else:
        expected_dtype = "Int64"
        item = NA
    expected = Series([3, 4, 6, item, 8, 4, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_len_mixed():
    # [FUNCTION DOC]
    # Verifies length on an object Series with non-strings gives lengths for strings and NaN for non-strings.
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.len()
    expected = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,sub,start,end,expected",
    [
        ("index", "EF", None, None, [4, 3, 1, 0]),
        ("rindex", "EF", None, None, [4, 5, 7, 4]),
        ("index", "EF", 3, None, [4, 3, 7, 4]),
        ("rindex", "EF", 3, None, [4, 5, 7, 4]),
        ("index", "E", 4, 8, [4, 5, 7, 4]),
        ("rindex", "E", 0, 5, [4, 3, 1, 4]),
    ],
)
def test_index(method, sub, start, end, index_or_series, any_string_dtype, expected):
    # [FUNCTION DOC]
    # Parametrized over first or last index lookup, substring, start, end and expected positions, with the Series-or-Index fixture and string dtype fixture. Verifies the result values and container type (integer, or Int64 for nullable), and equality with the Python standard library results per element.
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    expected_dtype = (
        np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected = index_or_series(expected, dtype=expected_dtype)

    result = getattr(obj.str, method)(sub, start, end)

    if index_or_series is Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)

    # compare with standard library
    expected = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected


def test_index_not_found_raises(index_or_series, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies looking up a substring that is absent from some elements raises a value error saying the substring was not found.
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    with pytest.raises(ValueError, match="substring not found"):
        obj.str.index("DE")


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_wrong_type_raises(index_or_series, any_string_dtype, method):
    # [FUNCTION DOC]
    # Parametrized over first and last index lookup. Verifies a non-string substring raises a type error saying a string object was expected, not int, even on empty data.
    obj = index_or_series([], dtype=any_string_dtype)
    msg = "expected a string object, not int"

    with pytest.raises(TypeError, match=msg):
        getattr(obj.str, method)(0)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["index", [1, 1, 0]],
        ["rindex", [3, 1, 2]],
    ],
)
def test_index_missing(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Parametrized over first and last lookup with expected positions. Verifies a Series with a missing value gives positions plus NaN (float) for object-like dtypes or NA (Int64) otherwise.
    ser = Series(["abcb", "ab", "bcbe", np.nan], dtype=any_string_dtype)
    if is_object_or_nan_string_dtype(any_string_dtype):
        expected_dtype = np.float64
        item = np.nan
    else:
        expected_dtype = "Int64"
        item = NA

    result = getattr(ser.str, method)("b")
    expected = Series([*exp, item], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_pipe_failures(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies a pipe character is treated literally by split and by a non-regex replace.
    # #2119
    ser = Series(["A|B|C"], dtype=any_string_dtype)

    result = ser.str.split("|")
    expected = Series([["A", "B", "C"]], dtype=object)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("|", " ", regex=False)
    expected = Series(["A B C"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", "bar", np.nan, "baz"]),
        (0, 3, -1, ["", "", np.nan, ""]),
        (None, None, -1, ["owtoofaa", "owtrabaa", np.nan, "xuqzabaa"]),
        (None, 2, -1, ["owtoo", "owtra", np.nan, "xuqza"]),
        (3, 10, 2, ["oto", "ato", np.nan, "aqx"]),
        (3, 0, -1, ["ofa", "aba", np.nan, "aba"]),
    ],
)
def test_slice(start, stop, step, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over (start, stop, step, expected) including negative steps and out-of-range values. Verifies slicing each string with missing preserved and dtype kept.
    ser = Series(["aafootwo", "aabartwo", np.nan, "aabazqux"], dtype=any_string_dtype)
    result = ser.str.slice(start, stop, step)
    expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", np.nan, "bar", np.nan, np.nan, None, np.nan, np.nan]),
        (4, 1, -1, ["oof", np.nan, "rab", np.nan, np.nan, None, np.nan, np.nan]),
    ],
)
def test_slice_mixed_object(start, stop, step, expected):
    # [FUNCTION DOC]
    # Parametrized over two slice specifications. Verifies slicing on an object Series with non-strings gives sliced strings, missing for non-strings (None stays None).
    ser = Series(["aafootwo", np.nan, "aabartwo", True, datetime.today(), None, 1, 2.0])
    result = ser.str.slice(start, stop, step)
    expected = Series(expected, dtype=object)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start,stop,repl,expected",
    [
        (2, 3, None, ["shrt", "a it longer", "evnlongerthanthat", "", np.nan]),
        (2, 3, "z", ["shzrt", "a zit longer", "evznlongerthanthat", "z", np.nan]),
        (2, 2, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (2, 1, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (-1, None, "z", ["shorz", "a bit longez", "evenlongerthanthaz", "z", np.nan]),
        (None, -2, "z", ["zrt", "zer", "zat", "z", np.nan]),
        (6, 8, "z", ["shortz", "a bit znger", "evenlozerthanthat", "z", np.nan]),
        (-10, 3, "z", ["zrt", "a zit longer", "evenlongzerthanthat", "z", np.nan]),
    ],
)
def test_slice_replace(start, stop, repl, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over (start, stop, replacement, expected) including a missing replacement, empty ranges, negative positions and replacement beyond the string. Verifies the replaced strings and preserved missing value and dtype.
    ser = Series(
        ["short", "a bit longer", "evenlongerthanthat", "", np.nan],
        dtype=any_string_dtype,
    )
    expected = Series(expected, dtype=any_string_dtype)
    result = ser.str.slice_replace(start, stop, repl)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", "bb", np.nan, "cc"]],
        ["lstrip", ["aa   ", "bb \n", np.nan, "cc  "]],
        ["rstrip", ["  aa", " bb", np.nan, "cc"]],
    ],
)
def test_strip_lstrip_rstrip(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Parametrized over strip, left strip and right strip. Verifies whitespace (including newline) removal from the respective ends, with missing preserved.
    ser = Series(["  aa   ", " bb \n", np.nan, "cc  "], dtype=any_string_dtype)

    result = getattr(ser.str, method)()
    expected = Series(exp, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", np.nan, "bb"]],
        ["lstrip", ["aa  ", np.nan, "bb \t\n"]],
        ["rstrip", ["  aa", np.nan, " bb"]],
    ],
)
def test_strip_lstrip_rstrip_mixed_object(method, exp):
    # [FUNCTION DOC]
    # Parametrized over the three strip methods. Verifies on object data with non-strings that strings are stripped and non-strings become missing (None stays None).
    ser = Series(["  aa  ", np.nan, " bb \t\n", True, datetime.today(), None, 1, 2.0])

    result = getattr(ser.str, method)()
    expected = Series([*exp, np.nan, np.nan, None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["ABC", " BNSD", "LDFJH "]],
        ["lstrip", ["ABCxx", " BNSD", "LDFJH xx"]],
        ["rstrip", ["xxABC", "xx BNSD", "LDFJH "]],
    ],
)
def test_strip_lstrip_rstrip_args(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Parametrized over the three strip methods. Verifies stripping a given character from the respective ends.
    ser = Series(["xxABCxx", "xx BNSD", "LDFJH xx"], dtype=any_string_dtype)

    result = getattr(ser.str, method)("x")
    expected = Series(exp, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("a", ["b", " b c", "bc"]),
        ("ab", ["", "a b c", "bc"]),
        ("", ["ab", "a b c", "bc"]),
    ],
)
def test_removeprefix(any_string_dtype, prefix, expected):
    # [FUNCTION DOC]
    # Parametrized over prefixes (one letter, two letters, empty). Verifies the prefix is removed only from the start when present.
    ser = Series(["ab", "a b c", "bc"], dtype=any_string_dtype)
    result = ser.str.removeprefix(prefix)
    ser_expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)


@pytest.mark.parametrize(
    "suffix, expected",
    [
        ("c", ["ab", "a b ", "b"]),
        ("bc", ["ab", "a b c", ""]),
        ("", ["ab", "a b c", "bc"]),
    ],
)
def test_removesuffix(any_string_dtype, suffix, expected):
    # [FUNCTION DOC]
    # Parametrized over suffixes (one letter, two letters, empty). Verifies the suffix is removed only from the end when present.
    ser = Series(["ab", "a b c", "bc"], dtype=any_string_dtype)
    result = ser.str.removesuffix(suffix)
    ser_expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)


def test_string_slice_get_syntax(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies subscript syntax on the string accessor matches the element-get method for an integer index and the slice method for a stop-only slice and for a negative-step slice with a start.
    ser = Series(
        ["YYY", "B", "C", "YYYYYYbYYY", "BYYYcYYY", np.nan, "CYYYBYYY", "dog", "cYYYt"],
        dtype=any_string_dtype,
    )

    result = ser.str[0]
    expected = ser.str.get(0)
    tm.assert_series_equal(result, expected)

    result = ser.str[:3]
    expected = ser.str.slice(stop=3)
    tm.assert_series_equal(result, expected)

    result = ser.str[2::-1]
    expected = ser.str.slice(start=2, step=-1)
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds_nested():
    # [FUNCTION DOC]
    # Verifies subscripting the accessor on a Series of tuples with an index beyond the shortest tuple gives the element where available and NaN otherwise.
    ser = Series([(1, 2), (1,), (3, 4, 5)])
    result = ser.str[1]
    expected = Series([2, np.nan, 4])
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies subscripting with an index beyond a string's length gives a missing value for that element and the character otherwise.
    ser = Series(["foo", "b", "ba"], dtype=any_string_dtype)
    result = ser.str[1]
    expected = Series(["o", np.nan, "a"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_encode_decode(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies encoding to UTF-8 bytes then decoding returns the original strings with the inferred string dtype.
    ser = Series(["a", "b", "a\xe4"], dtype=any_string_dtype).str.encode("utf-8")
    result = ser.str.decode("utf-8")
    expected = Series(["a", "b", "a\xe4"], dtype="str")
    tm.assert_series_equal(result, expected)


def test_encode_errors_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies encoding with a code page that lacks a character raises a Unicode encode error with the codec's message, and that passing the errors mode ignore as second argument drops the character, matching mapping the standard encode.
    ser = Series(["a", "b", "a\x9d"], dtype=any_string_dtype)

    msg = (
        r"'charmap' codec can't encode character '\\x9d' in position 1: "
        "character maps to <undefined>"
    )
    with pytest.raises(UnicodeEncodeError, match=msg):
        ser.str.encode("cp1252")

    result = ser.str.encode("cp1252", "ignore")
    expected = ser.map(lambda x: x.encode("cp1252", "ignore"))
    tm.assert_series_equal(result, expected)


def test_decode_errors_kwarg():
    # [FUNCTION DOC]
    # Verifies decoding invalid bytes with a code page raises a Unicode decode error with the codec's message, and that the errors mode ignore gives the expected decoded strings as string dtype.
    ser = Series([b"a", b"b", b"a\x9d"])

    msg = (
        "'charmap' codec can't decode byte 0x9d in position 1: "
        "character maps to <undefined>"
    )
    with pytest.raises(UnicodeDecodeError, match=msg):
        ser.str.decode("cp1252")

    result = ser.str.decode("cp1252", "ignore")
    expected = ser.map(lambda x: x.decode("cp1252", "ignore")).astype("str")
    tm.assert_series_equal(result, expected)


def test_decode_string_dtype(string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Uses the string dtype fixture. Verifies decoding with an explicit dtype argument returns that dtype.
    # https://github.com/pandas-dev/pandas/pull/60940
    ser = Series([b"a", b"b"])
    result = ser.str.decode("utf-8", dtype=string_dtype)
    expected = Series(["a", "b"], dtype=string_dtype)
    tm.assert_series_equal(result, expected)


def test_decode_object_dtype(object_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Uses the object dtype fixture. Verifies decoding bytes containing a literal backslash escape sequence with object dtype returns object dtype with the same text.
    # https://github.com/pandas-dev/pandas/pull/60940
    ser = Series([b"a", rb"\ud800"])
    result = ser.str.decode("utf-8", dtype=object_dtype)
    expected = Series(["a", r"\ud800"], dtype=object_dtype)
    tm.assert_series_equal(result, expected)


def test_decode_bad_dtype():
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies passing an integer dtype raises a value error saying the dtype must be string or object.
    # https://github.com/pandas-dev/pandas/pull/60940
    ser = Series([b"a", b"b"])
    msg = "dtype must be string or object, got dtype='int64'"
    with pytest.raises(ValueError, match=msg):
        ser.str.decode("utf-8", dtype="int64")


@pytest.mark.parametrize(
    "form, expected",
    [
        ("NFKC", ["ABC", "ABC", "123", np.nan, "アイエ"]),
        ("NFC", ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]),  # noqa: RUF001
    ],
)
def test_normalize(form, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over unicode normalization forms compatibility-composed and canonical-composed with expected results for full-width and half-width characters, with a missing value and custom index. Verifies the normalized Series.
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],  # noqa: RUF001
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    expected = Series(expected, index=["a", "b", "c", "d", "e"], dtype=any_string_dtype)
    result = ser.str.normalize(form)
    tm.assert_series_equal(result, expected)


def test_normalize_bad_arg_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies an unknown normalization form raises a value error saying invalid normalization form.
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],  # noqa: RUF001
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    with pytest.raises(ValueError, match="invalid normalization form"):
        ser.str.normalize("xxx")


def test_normalize_index():
    # [FUNCTION DOC]
    # Verifies normalization on an Index of full-width characters with the compatibility form returns the expected Index.
    idx = Index(["ＡＢＣ", "１２３", "ｱｲｴ"])  # noqa: RUF001
    expected = Index(["ABC", "123", "アイエ"])
    result = idx.str.normalize("NFKC")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "values,inferred_type",
    [
        (["a", "b"], "string"),
        (["a", "b", 1], "mixed-integer"),
        (["a", "b", 1.3], "mixed"),
        (["a", "b", 1.3, 1], "mixed-integer"),
        (["aa", datetime(2011, 1, 1)], "mixed"),
    ],
)
def test_index_str_accessor_visibility(values, inferred_type, index_or_series):
    # [FUNCTION DOC]
    # Parametrized over value lists (strings, strings with integer, float, mixed) and their inferred types. Verifies that for an Index the inferred type is as expected and the string accessor is available returning the accessor class.
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    assert isinstance(obj.str, StringMethods)


@pytest.mark.parametrize(
    "values,inferred_type",
    [
        ([1, np.nan], "floating"),
        ([datetime(2011, 1, 1)], "datetime64"),
        ([timedelta(1)], "timedelta64"),
    ],
)
def test_index_str_accessor_non_string_values_raises(
    values, inferred_type, index_or_series
):
    # [FUNCTION DOC]
    # Parametrized over non-string value lists (float with NaN, datetimes, timedeltas) with expected inferred types. Verifies the accessor raises an attribute error saying it can only be used with string values.
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    msg = "Can only use .str accessor with string values"
    with pytest.raises(AttributeError, match=msg):
        obj.str


def test_index_str_accessor_multiindex_raises():
    # [FUNCTION DOC]
    # Verifies a MultiIndex (inferred type mixed) raises an attribute error saying the accessor works on an Index, not a MultiIndex.
    # MultiIndex has mixed dtype, but not allow to use accessor
    idx = MultiIndex.from_tuples([("a", "b"), ("a", "b")])
    assert idx.inferred_type == "mixed"

    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        idx.str


def test_str_accessor_no_new_attributes(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies assigning a new attribute on the accessor object raises an attribute error saying new attributes cannot be added.
    # https://github.com/pandas-dev/pandas/issues/10673
    ser = Series(list("aabbcde"), dtype=any_string_dtype)
    with pytest.raises(AttributeError, match="You cannot add any new attribute"):
        ser.str.xlabel = "a"


def test_cat_on_bytes_raises():
    # [FUNCTION DOC]
    # Verifies concatenating two Series of byte strings raises a type error saying the inferred dtype bytes is not allowed.
    lhs = Series(np.array(list("abc"), "S1").astype(object))
    rhs = Series(np.array(list("def"), "S1").astype(object))
    msg = "Cannot use .str.cat with values of inferred dtype 'bytes'"
    with pytest.raises(TypeError, match=msg):
        lhs.str.cat(rhs)


def test_str_accessor_in_apply_func():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies using the string accessor on each row inside a row-wise apply works, producing joined uppercase strings.
    # https://github.com/pandas-dev/pandas/issues/38979
    df = DataFrame(zip("abc", "def", strict=True))
    expected = Series(["A/D", "B/E", "C/F"])
    result = df.apply(lambda f: "/".join(f.str.upper()), axis=1)
    tm.assert_series_equal(result, expected)


def test_zfill():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies zero-fill pads after a sign to the requested width, leaves longer strings, and gives NaN for non-strings (a number) and missing values.
    # https://github.com/pandas-dev/pandas/issues/20868
    value = Series(["-1", "1", "1000", 10, np.nan])
    expected = Series(["-01", "001", "1000", np.nan, np.nan], dtype=object)
    tm.assert_series_equal(value.str.zfill(3), expected)

    value = Series(["-2", "+5"])
    expected = Series(["-0002", "+0005"])
    tm.assert_series_equal(value.str.zfill(5), expected)


def test_zfill_with_non_integer_argument():
    # [FUNCTION DOC]
    # Verifies a non-integer width raises a type error saying the width must be of integer type, naming the given type.
    value = Series(["-2", "+5"])
    wid = "a"
    msg = f"width must be of integer type, not {type(wid).__name__}"
    with pytest.raises(TypeError, match=msg):
        value.str.zfill(wid)


def test_zfill_with_leading_sign():
    # [FUNCTION DOC]
    # Verifies zero-fill puts zeros after a leading plus or minus sign, including for text after the sign.
    value = Series(["-cat", "-1", "+dog"])
    expected = Series(["-0cat", "-0001", "+0dog"])
    tm.assert_series_equal(value.str.zfill(5), expected)


def test_get_with_dict_label():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies the element-get method with a key works on a Series of dictionaries, giving None where the key is missing.
    # GH47911
    s = Series(
        [
            {"name": "Hello", "value": "World"},
            {"name": "Goodbye", "value": "Planet"},
            {"value": "Sea"},
        ]
    )
    result = s.str.get("name")
    expected = Series(["Hello", "Goodbye", None], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.get("value")
    expected = Series(["World", "Planet", "Sea"], dtype=object)
    tm.assert_series_equal(result, expected)


def test_series_str_decode():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies decoding bytes with explicit encoding and strict errors gives strings of the inferred string dtype.
    # GH 22613
    result = Series([b"x", b"y"]).str.decode(encoding="UTF-8", errors="strict")
    expected = Series(["x", "y"], dtype="str")
    tm.assert_series_equal(result, expected)


def test_decode_with_dtype_none():
    # [FUNCTION DOC]
    # With the option enabling inferred strings, verifies decoding with a None dtype gives the inferred string dtype.
    with option_context("future.infer_string", True):
        ser = Series([b"a", b"b", b"c"])
        result = ser.str.decode("utf-8", dtype=None)
        expected = Series(["a", "b", "c"], dtype="str")
        tm.assert_series_equal(result, expected)


def test_setitem_with_different_string_storage():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Skips without pyarrow. Verifies assigning a slice of a Python-storage string Series from a pyarrow-storage Series and vice versa keeps the target's storage and writes the converted values, including a slice assignment containing a missing value.
    # GH#52987
    # Test setitem with values from different string storage type
    pytest.importorskip("pyarrow")

    # Test Series[string[python]].__setitem__(Series[string[pyarrow]])
    ser_python = Series(range(5), dtype="string[python]")
    ser_pyarrow = ser_python.astype("string[pyarrow]")

    ser_python[:2] = ser_pyarrow[:2]
    expected = Series(["0", "1", "2", "3", "4"], dtype="string[python]")
    tm.assert_series_equal(ser_python, expected)

    # Test Series[string[pyarrow]].__setitem__(Series[string[python]])
    ser_pyarrow = Series(range(5), dtype="string[pyarrow]")
    ser_python = ser_pyarrow.astype("string[python]")

    ser_pyarrow[:2] = ser_python[:2]
    expected = Series(["0", "1", "2", "3", "4"], dtype="string[pyarrow]")
    tm.assert_series_equal(ser_pyarrow, expected)

    # Test with slice and missing values
    ser_python = Series(["a", "b", None, "d", "e"], dtype="string[python]")
    ser_pyarrow = Series(["X", "Y", None], dtype="string[pyarrow]")

    ser_python[1:4] = ser_pyarrow
    expected = Series(["a", "X", "Y", NA, "e"], dtype="string[python]")
    tm.assert_series_equal(ser_python, expected)


@pytest.mark.parametrize(
    "pat, expected",
    [
        # lookaround assertions
        (r"(?=abc)", True),
        (r"(?<=123)", True),
        (r"(?!xyz)", True),
        (r"(?<!\d)", True),
        (r"(?=a|b)(?<=c)", True),
        (r"abc", False),
        (r"\d+", False),
        (r"(abc)", False),
        (r"a|b", False),
        (r"a*", False),
        (r"", False),
        (r"\\(?=abc)", True),
        (r"(?=.*[A-Z])", True),
        (r"a(?=)", True),
        (r"(?![0-9])", True),
        (r"(?=(?!nested))", True),
        (r"test\(\?\=ing\)", False),
        (r"[(?=)]", False),
        (r"(?#(?=comment)", False),
        (r"(test # (?=comment))", True),
        (r"(?=test)+", False),
        (r"(?=test)*", False),
        (r"(?=test)?", False),
        (r"abc|(?=test)", True),
        (r"^(?=test)$", True),
        # backreferences
        (r"(abc)\1", True),
        (r"\b(\w+)\s+\1\b", True),
        (r"\b(?P<word>\w+)\s+(?P=word)\b", True),
    ],
)
def test_has_regex_unsupported_code(pat, expected):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over about thirty regex patterns with an expected flag. Verifies the private helper of the Arrow string mixin reports true for patterns using lookaround assertions or backreferences (including named backreferences, nested lookaheads and lookaheads combined with alternation) and false for plain patterns, escaped parentheses, character classes containing lookahead syntax, inline comment groups, and lookaheads followed by a quantifier.
    # https://github.com/pandas-dev/pandas/issues/60833
    assert ArrowStringArrayMixin._has_unsupported_regex(pat) == expected
