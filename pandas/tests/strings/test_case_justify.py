# [MODULE DOC]
# Tests for the string accessor's case-conversion methods (title, lower, upper, capitalize, swapcase, casefold) and justification/padding methods (pad, center, ljust, rjust, zfill, wrap), across string dtypes (fixture) and object data mixing strings with booleans, datetimes, None, NaN and numbers (non-string entries become NaN, None stays None).
# No import-time side effects.

from datetime import datetime
import operator

import numpy as np
import pytest

from pandas import (
    Series,
    _testing as tm,
)


def test_title(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that the title method capitalizes each word and keeps NaN, for any string dtype (fixture).
    s = Series(["FOO", "BAR", np.nan, "Blah", "blurg"], dtype=any_string_dtype)
    result = s.str.title()
    expected = Series(["Foo", "Bar", np.nan, "Blah", "Blurg"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_title_mixed_object():
    # [FUNCTION DOC]
    # Verifies title on object data mixing strings with NaN, boolean, datetime, None and numbers: strings are title-cased, None remains None, every other non-string becomes NaN; compared with approximate equality.
    s = Series(["FOO", np.nan, "bar", True, datetime.today(), "blah", None, 1, 2.0])
    result = s.str.title()
    expected = Series(
        ["Foo", np.nan, "Bar", np.nan, np.nan, "Blah", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_almost_equal(result, expected)


def test_lower_upper(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the upper method on strings with a missing value (any string dtype), and that applying lower to the result restores the original Series.
    s = Series(["om", np.nan, "nom", "nom"], dtype=any_string_dtype)

    result = s.str.upper()
    expected = Series(["OM", np.nan, "NOM", "NOM"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = result.str.lower()
    tm.assert_series_equal(result, s)


def test_lower_upper_mixed_object():
    # [FUNCTION DOC]
    # Verifies upper and lower on mixed object data: strings are converted, None stays None, other non-strings become NaN.
    s = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])

    result = s.str.upper()
    expected = Series(
        ["A", np.nan, "B", np.nan, np.nan, "FOO", None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)

    result = s.str.lower()
    expected = Series(
        ["a", np.nan, "b", np.nan, np.nan, "foo", None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            ["FOO", "BAR", np.nan, "Blah", "blurg"],
            ["Foo", "Bar", np.nan, "Blah", "Blurg"],
        ),
        (["a", "b", "c"], ["A", "B", "C"]),
        (["a b", "a bc. de"], ["A b", "A bc. de"]),
    ],
)
def test_capitalize(data, expected, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the capitalize method for any string dtype over three parametrized inputs (mixed case with NaN, single letters, words with punctuation): only the first character is uppercased and the rest lowercased; NaN is preserved.
    s = Series(data, dtype=any_string_dtype)
    result = s.str.capitalize()
    expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_capitalize_mixed_object():
    # [FUNCTION DOC]
    # Verifies capitalize on mixed object data: strings converted, None kept, other non-strings become NaN.
    s = Series(["FOO", np.nan, "bar", True, datetime.today(), "blah", None, 1, 2.0])
    result = s.str.capitalize()
    expected = Series(
        ["Foo", np.nan, "Bar", np.nan, np.nan, "Blah", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_swapcase(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies swapcase on strings with a missing value for any string dtype.
    s = Series(["FOO", "BAR", np.nan, "Blah", "blurg"], dtype=any_string_dtype)
    result = s.str.swapcase()
    expected = Series(["foo", "bar", np.nan, "bLAH", "BLURG"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_swapcase_mixed_object():
    # [FUNCTION DOC]
    # Verifies swapcase on mixed object data: strings swapped, None kept, other non-strings become NaN.
    s = Series(["FOO", np.nan, "bar", True, datetime.today(), "Blah", None, 1, 2.0])
    result = s.str.swapcase()
    expected = Series(
        ["foo", np.nan, "BAR", np.nan, np.nan, "bLAH", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_casefold():
    # [FUNCTION DOC]
    # Verifies (issue 25405) casefold on default-dtype strings: the German sharp s becomes two letters s, NaN is preserved, other text is lowercased.
    # GH25405
    expected = Series(["ss", np.nan, "case", "ssd"])
    s = Series(["ß", np.nan, "case", "ßd"])
    result = s.str.casefold()

    tm.assert_series_equal(result, expected)


def test_casemethods(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that lower, upper, title, capitalize and swapcase, for any string dtype, give the same list of results as the corresponding built-in string methods applied to each of five sample strings.
    values = ["aaa", "bbb", "CCC", "Dddd", "eEEE"]
    s = Series(values, dtype=any_string_dtype)
    assert s.str.lower().tolist() == [v.lower() for v in values]
    assert s.str.upper().tolist() == [v.upper() for v in values]
    assert s.str.title().tolist() == [v.title() for v in values]
    assert s.str.capitalize().tolist() == [v.capitalize() for v in values]
    assert s.str.swapcase().tolist() == [v.swapcase() for v in values]


def test_pad(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies pad to width five with side left, right and both (any string dtype): shorter strings are padded with spaces (for both, extra space goes to the right when odd is not an issue here), strings at or over the width are unchanged, and NaN is preserved.
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    result = s.str.pad(5, side="left")
    expected = Series(
        ["    a", "    b", np.nan, "    c", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = s.str.pad(5, side="right")
    expected = Series(
        ["a    ", "b    ", np.nan, "c    ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = s.str.pad(5, side="both")
    expected = Series(
        ["  a  ", "  b  ", np.nan, "  c  ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_pad_mixed_object():
    # [FUNCTION DOC]
    # Verifies pad to width five on all three sides for mixed object data: strings padded (for both sides, an odd remainder puts the extra space on the right), None kept, other non-strings become NaN.
    s = Series(["a", np.nan, "b", True, datetime.today(), "ee", None, 1, 2.0])

    result = s.str.pad(5, side="left")
    expected = Series(
        ["    a", np.nan, "    b", np.nan, np.nan, "   ee", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.pad(5, side="right")
    expected = Series(
        ["a    ", np.nan, "b    ", np.nan, np.nan, "ee   ", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.pad(5, side="both")
    expected = Series(
        ["  a  ", np.nan, "  b  ", np.nan, np.nan, "  ee ", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_pad_fillchar(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies pad with an explicit fill character on all three sides for any string dtype, using the character X in place of space.
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    result = s.str.pad(5, side="left", fillchar="X")
    expected = Series(
        ["XXXXa", "XXXXb", np.nan, "XXXXc", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = s.str.pad(5, side="right", fillchar="X")
    expected = Series(
        ["aXXXX", "bXXXX", np.nan, "cXXXX", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = s.str.pad(5, side="both", fillchar="X")
    expected = Series(
        ["XXaXX", "XXbXX", np.nan, "XXcXX", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_pad_fillchar_bad_arg_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that pad raises a type error when the fill character is a multi-character string (message names type str) or an integer (message names type int).
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    msg = "fillchar must be a character, not str"
    with pytest.raises(TypeError, match=msg):
        s.str.pad(5, fillchar="XY")

    msg = "fillchar must be a character, not int"
    with pytest.raises(TypeError, match=msg):
        s.str.pad(5, fillchar=5)


@pytest.mark.parametrize("method_name", ["center", "ljust", "rjust", "zfill", "pad"])
def test_pad_width_bad_arg_raises(method_name, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 13598) that center, left-justify, right-justify, zero-fill and pad raise a type error saying the width must be of integer type, not str, when given a string width. Parametrized over the method name; the call is built with a method-caller helper.
    # see gh-13598
    s = Series(["1", "22", "a", "bb"], dtype=any_string_dtype)
    op = operator.methodcaller(method_name, "f")

    msg = "width must be of integer type, not str"
    with pytest.raises(TypeError, match=msg):
        op(s.str)


def test_center_ljust_rjust(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies center, left-justify and right-justify to width five with the default space fill on strings with NaN for any string dtype; strings of length six or more are unchanged.
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    result = s.str.center(5)
    expected = Series(
        ["  a  ", "  b  ", np.nan, "  c  ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = s.str.ljust(5)
    expected = Series(
        ["a    ", "b    ", np.nan, "c    ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = s.str.rjust(5)
    expected = Series(
        ["    a", "    b", np.nan, "    c", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_center_ljust_rjust_mixed_object():
    # [FUNCTION DOC]
    # Verifies center, left-justify and right-justify to width five on mixed object data, converting strings, keeping None, and giving NaN for other non-strings.
    s = Series(["a", np.nan, "b", True, datetime.today(), "c", "eee", None, 1, 2.0])

    result = s.str.center(5)
    expected = Series(
        [
            "  a  ",
            np.nan,
            "  b  ",
            np.nan,
            np.nan,
            "  c  ",
            " eee ",
            None,
            np.nan,
            np.nan,
        ],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.ljust(5)
    expected = Series(
        [
            "a    ",
            np.nan,
            "b    ",
            np.nan,
            np.nan,
            "c    ",
            "eee  ",
            None,
            np.nan,
            np.nan,
        ],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.rjust(5)
    expected = Series(
        [
            "    a",
            np.nan,
            "    b",
            np.nan,
            np.nan,
            "    c",
            "  eee",
            None,
            np.nan,
            np.nan,
        ],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_center_ljust_rjust_fillchar(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issues 54533, 54792) center, left-justify and right-justify with fill character X for any string dtype. Each result is compared with an expected Series and also with the built-in string method applied elementwise to the raw values, as an object NumPy array.
    # GH#54533, GH#54792
    s = Series(["a", "bb", "cccc", "ddddd", "eeeeee"], dtype=any_string_dtype)

    result = s.str.center(5, fillchar="X")
    expected = Series(
        ["XXaXX", "XXbbX", "Xcccc", "ddddd", "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)
    expected = np.array([v.center(5, "X") for v in np.array(s)], dtype=np.object_)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)

    result = s.str.ljust(5, fillchar="X")
    expected = Series(
        ["aXXXX", "bbXXX", "ccccX", "ddddd", "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)
    expected = np.array([v.ljust(5, "X") for v in np.array(s)], dtype=np.object_)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)

    result = s.str.rjust(5, fillchar="X")
    expected = Series(
        ["XXXXa", "XXXbb", "Xcccc", "ddddd", "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rjust(5, "X") for v in np.array(s)], dtype=np.object_)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)


def test_center_ljust_rjust_fillchar_bad_arg_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that center, left-justify and right-justify each raise a type error when the fill character is a multi-character string (message names str) or an integer (message names int).
    s = Series(["a", "bb", "cccc", "ddddd", "eeeeee"], dtype=any_string_dtype)

    # If fillchar is not a character, normal str raises TypeError
    # 'aaa'.ljust(5, 'XY')
    # TypeError: must be char, not str
    template = "fillchar must be a character, not {dtype}"

    with pytest.raises(TypeError, match=template.format(dtype="str")):
        s.str.center(5, fillchar="XY")

    with pytest.raises(TypeError, match=template.format(dtype="str")):
        s.str.ljust(5, fillchar="XY")

    with pytest.raises(TypeError, match=template.format(dtype="str")):
        s.str.rjust(5, fillchar="XY")

    with pytest.raises(TypeError, match=template.format(dtype="int")):
        s.str.center(5, fillchar=1)

    with pytest.raises(TypeError, match=template.format(dtype="int")):
        s.str.ljust(5, fillchar=1)

    with pytest.raises(TypeError, match=template.format(dtype="int")):
        s.str.rjust(5, fillchar=1)


def test_zfill(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies zero-fill to widths five and three for any string dtype (shorter strings left-padded with zeros, longer ones unchanged), compared with the built-in zero-fill applied elementwise, and with NaN entries preserved.
    s = Series(["1", "22", "aaa", "333", "45678"], dtype=any_string_dtype)

    result = s.str.zfill(5)
    expected = Series(
        ["00001", "00022", "00aaa", "00333", "45678"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)
    expected = np.array([v.zfill(5) for v in np.array(s)], dtype=np.object_)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)

    result = s.str.zfill(3)
    expected = Series(["001", "022", "aaa", "333", "45678"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.zfill(3) for v in np.array(s)], dtype=np.object_)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)

    s = Series(["1", np.nan, "aaa", np.nan, "45678"], dtype=any_string_dtype)
    result = s.str.zfill(5)
    expected = Series(
        ["00001", np.nan, "00aaa", np.nan, "45678"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_wrap(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies wrapping at width twelve with long words allowed to break, for any string dtype, over nine inputs covering: two words shorter than, equal to and longer than the width; a single word shorter, equal and longer; trailing whitespace; and a tab-only string (which becomes empty). Breaks are inserted as newlines and trailing whitespace is dropped.
    # test values are: two words less than width, two words equal to width,
    # two words greater than width, one word less than width, one word
    # equal to width, one word greater than width, multiple tokens with
    # trailing whitespace equal to width
    s = Series(
        [
            "hello world",
            "hello world!",
            "hello world!!",
            "abcdefabcde",
            "abcdefabcdef",
            "abcdefabcdefa",
            "ab ab ab ab ",
            "ab ab ab ab a",
            "\t",
        ],
        dtype=any_string_dtype,
    )

    # expected values
    expected = Series(
        [
            "hello world",
            "hello world!",
            "hello\nworld!!",
            "abcdefabcde",
            "abcdefabcdef",
            "abcdefabcdef\na",
            "ab ab ab ab",
            "ab ab ab ab\na",
            "",
        ],
        dtype=any_string_dtype,
    )

    result = s.str.wrap(12, break_long_words=True)
    tm.assert_series_equal(result, expected)


def test_wrap_unicode(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies wrapping at width six with leading and trailing whitespace, NaN, and non-ASCII characters: the whitespace-only prefix is retained but trailing space removed, and the long unicode string wraps at the width.
    # test with pre and post whitespace (non-unicode), NaN, and non-ascii Unicode
    s = Series(
        ["  pre  ", np.nan, "\xac\u20ac\U00008000 abadcafe"], dtype=any_string_dtype
    )
    expected = Series(
        ["  pre", np.nan, "\xac\u20ac\U00008000 ab\nadcafe"], dtype=any_string_dtype
    )
    result = s.str.wrap(6)
    tm.assert_series_equal(result, expected)
