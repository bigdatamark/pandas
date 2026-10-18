# [MODULE DOC]
# Tests for the string accessor's splitting and partitioning methods (split, rsplit, partition, rpartition) and the element-get method on split results, for Series and Index, with the string dtype fixtures. Uses the helper that converts the expected missing value to the one appropriate for the dtype and the helper that identifies object or NaN-based string dtypes.
from datetime import datetime
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)
from pandas.tests.strings import (
    _convert_na_value,
    is_object_or_nan_string_dtype,
)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split(any_string_dtype, method):
    # [FUNCTION DOC]
    # Parametrized over split and right split. Verifies splitting on an underscore gives lists of parts with the missing value kept (converted to the dtype's missing value).
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    result = getattr(values.str, method)("_")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_more_than_one_char(any_string_dtype, method):
    # [FUNCTION DOC]
    # Parametrized over split and right split. Verifies a two-character separator splits correctly, and that the explicit non-expanding argument gives the same result.
    # more than one char
    values = Series(["a__b__c", "c__d__e", np.nan, "f__g__h"], dtype=any_string_dtype)
    result = getattr(values.str, method)("__")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)

    result = getattr(values.str, method)("__", expand=False)
    tm.assert_series_equal(result, exp)


def test_split_more_regex_split(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies a multi-character pattern in brackets (comma or underscore) is treated as a regular expression by split and splits on either character.
    # regex split
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    result = values.str.split("[,_]")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


def test_split_regex(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies split with a regex pattern for a literal dot and explicit regex flag on a one-element Series gives the pieces including a trailing empty string.
    # GH 43563
    # explicit regex = True split
    values = Series("xxxjpgzzz.jpg", dtype=any_string_dtype)
    result = values.str.split(r"\.jpg", regex=True)
    exp = Series([["xxxjpgzzz", ""]])
    tm.assert_series_equal(result, exp)


def test_split_regex_explicit(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies: a compiled pattern splits as regex (the dot matches any character); regex false treats the pattern literally so nothing splits; a single-character pattern is literal by default; a longer pattern is a regex by default; and a compiled pattern with regex false raises a value error saying a compiled regex cannot be used with regex false.
    # explicit regex = True split with compiled regex
    regex_pat = re.compile(r".jpg")
    values = Series("xxxjpgzzz.jpg", dtype=any_string_dtype)
    result = values.str.split(regex_pat)
    exp = Series([["xx", "zzz", ""]])
    tm.assert_series_equal(result, exp)

    # explicit regex = False split
    result = values.str.split(r"\.jpg", regex=False)
    exp = Series([["xxxjpgzzz.jpg"]])
    tm.assert_series_equal(result, exp)

    # non explicit regex split, pattern length == 1
    result = values.str.split(r".")
    exp = Series([["xxxjpgzzz", "jpg"]])
    tm.assert_series_equal(result, exp)

    # non explicit regex split, pattern length != 1
    result = values.str.split(r".jpg")
    exp = Series([["xx", "zzz", ""]])
    tm.assert_series_equal(result, exp)

    # regex=False with pattern compiled regex raises error
    with pytest.raises(
        ValueError,
        match="Cannot use a compiled regex as replacement pattern with regex=False",
    ):
        values.str.split(regex_pat, regex=False)


@pytest.mark.parametrize("expand", [None, False])
@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_object_mixed(expand, method):
    # [FUNCTION DOC]
    # Parametrized over the expand argument (none or false) and split or right split. Verifies on an object Series with non-strings a Series is returned with lists for strings and missing values for others (None kept).
    mixed = Series(["a_b_c", np.nan, "d_e_f", True, datetime.today(), None, 1, 2.0])
    result = getattr(mixed.str, method)("_", expand=expand)
    exp = Series(
        [
            ["a", "b", "c"],
            np.nan,
            ["d", "e", "f"],
            np.nan,
            np.nan,
            None,
            np.nan,
            np.nan,
        ]
    )
    assert isinstance(result, Series)
    tm.assert_almost_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
@pytest.mark.parametrize("n", [None, 0])
def test_split_n(any_string_dtype, method, n):
    # [FUNCTION DOC]
    # Parametrized over split and right split and the maximum split count (none or zero). Verifies both mean no limit, with a missing value kept.
    s = Series(["a b", pd.NA, "b c"], dtype=any_string_dtype)
    expected = Series([["a", "b"], pd.NA, ["b", "c"]])
    result = getattr(s.str, method)(" ", n=n)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


def test_rsplit(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies right split does not treat the pattern as a regex: the bracket pattern is not found so strings are returned whole in lists.
    # regex split is not supported by rsplit
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    result = values.str.rsplit("[,_]")
    exp = Series([["a,b_c"], ["c_d,e"], np.nan, ["f,g,h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


def test_rsplit_max_number(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the maximum split count for right split counts from the right.
    # setting max number of splits, make sure it's from reverse
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = values.str.rsplit("_", n=1)
    exp = Series([["a_b", "c"], ["c_d", "e"], np.nan, ["f_g", "h"]])
    exp = _convert_na_value(values, exp)
    tm.assert_series_equal(result, exp)


def test_split_blank_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies expanded split of a Series with an empty string gives a one-row frame with no columns (explicitly not an empty frame).
    # expand blank split GH 20067
    values = Series([""], name="test", dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame([[]], dtype=any_string_dtype)  # NOTE: this is NOT an empty df
    tm.assert_frame_equal(result, exp)


def test_split_blank_string_with_non_empty(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expanded split without separator on whitespace-only and empty strings pads with missing values to the maximum number of parts.
    values = Series(["a b c", "a b", "", " "], name="test", dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame(
        [
            ["a", "b", "c"],
            ["a", "b", None],
            [None, None, None],
            [None, None, None],
        ],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_noargs(any_string_dtype, method):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over split and right split. Verifies that splitting with no arguments splits on runs of whitespace.
    # #1859
    s = Series(["Wes McKinney", "Travis  Oliphant"], dtype=any_string_dtype)
    result = getattr(s.str, method)()
    expected = ["Travis", "Oliphant"]
    assert result[1] == expected


@pytest.mark.parametrize(
    "data, pat",
    [
        (["bd asdf jfg", "kjasdflqw asdfnfk"], None),
        (["bd asdf jfg", "kjasdflqw asdfnfk"], "asdf"),
        (["bd_asdf_jfg", "kjasdflqw_asdfnfk"], "_"),
    ],
)
@pytest.mark.parametrize("n", [-1, 0])
def test_split_maxsplit(data, pat, any_string_dtype, n):
    # [FUNCTION DOC]
    # Parametrized over data and separators (none, a word, underscore) and over a split count of minus one or zero. Verifies the result equals the default unlimited split.
    # re.split 0, str.split -1
    s = Series(data, dtype=any_string_dtype)

    result = s.str.split(pat=pat, n=n)
    xp = s.str.split(pat=pat)
    tm.assert_series_equal(result, xp)


@pytest.mark.parametrize(
    "data, pat, expected_val",
    [
        (
            ["split once", "split once too!"],
            None,
            "once too!",
        ),
        (
            ["split_once", "split_once_too!"],
            "_",
            "once_too!",
        ),
    ],
)
def test_split_no_pat_with_nonzero_n(data, pat, expected_val, any_string_dtype):
    # [FUNCTION DOC]
    # Parametrized over data and separator (none or underscore) with expected last part. Verifies a maximum split count of one gives two parts (whitespace runs handled when no separator is given), compared with an index-type check relaxed.
    s = Series(data, dtype=any_string_dtype)
    result = s.str.split(pat=pat, n=1)
    expected = Series({0: ["split", "once"], 1: ["split", expected_val]})
    tm.assert_series_equal(expected, result, check_index_type=False)


def test_split_to_dataframe_no_splits(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expand with no separator present gives a single column frame equal to the originals.
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    result = s.str.split("_", expand=True)
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)})
    tm.assert_frame_equal(result, exp)


def test_split_to_dataframe(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expand with equal part counts gives a frame with one column per part.
    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_split_to_dataframe_unequal_splits(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expand with unequal part counts pads shorter rows with missing values.
    s = Series(
        ["some_unequal_splits", "one_of_these_things_is_not"], dtype=any_string_dtype
    )
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {
            0: ["some", "one"],
            1: ["unequal", "of"],
            2: ["splits", "these"],
            3: [None, "things"],
            4: [None, "is"],
            5: [None, "not"],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_split_to_dataframe_with_index(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expand keeps the original index, and that a non-boolean expand argument raises a value error.
    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    with pytest.raises(ValueError, match="expand must be"):
        s.str.split("_", expand="not_a_boolean")


def test_split_to_multiindex_expand_no_splits():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies expanded split on an Index with no splits (and a NaN) returns an equal Index with one level.
    # https://github.com/pandas-dev/pandas/issues/23677

    idx = Index(["nosplit", "alsonosplit", np.nan])
    result = idx.str.split("_", expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1


def test_split_to_multiindex_expand():
    # [FUNCTION DOC]
    # Verifies expanded split on an Index gives a three-level MultiIndex with missing entries expanded for NaN and None.
    idx = Index(["some_equal_splits", "with_no_nans", np.nan, None])
    result = idx.str.split("_", expand=True)
    exp = MultiIndex.from_tuples(
        [
            ("some", "equal", "splits"),
            ("with", "no", "nans"),
            [np.nan, np.nan, np.nan],
            [None, None, None],
        ]
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3


def test_split_to_multiindex_expand_unequal_splits():
    # [FUNCTION DOC]
    # Verifies unequal part counts on an Index give a six-level MultiIndex padded with NaN or None as appropriate, and a non-boolean expand argument raises a value error.
    idx = Index(["some_unequal_splits", "one_of_these_things_is_not", np.nan, None])
    result = idx.str.split("_", expand=True)
    exp = MultiIndex.from_tuples(
        [
            ("some", "unequal", "splits", np.nan, np.nan, np.nan),
            ("one", "of", "these", "things", "is", "not"),
            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
            (None, None, None, None, None, None),
        ]
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 6

    with pytest.raises(ValueError, match="expand must be"):
        idx.str.split("_", expand="not_a_boolean")


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"a(?=b)", [["aa"], ["", "b"], ["ba"], ["bb"]]),
        (r"(?<=a)b", [["aa"], ["a", ""], ["ba"], ["bb"]]),
        (r"a(?!b)", [["", "", ""], ["ab"], ["b", ""], ["bb"]]),
        (r"(?<!b)a", [["", "", ""], ["", "b"], ["ba"], ["bb"]]),
        ("ab", [["aa"], ["", ""], ["ba"], ["bb"]]),
    ],
)
def test_split_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over lookahead and lookbehind patterns with expected parts. Verifies regex split on four strings and a missing value; the missing result is None for object, NaN for the inferred string dtype, and NA for the nullable dtype.
    # https://github.com/pandas-dev/pandas/issues/60833
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.split(pat, regex=True)
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


def test_split_regex_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub pull request. Verifies end-of-string anchors in regex split: a dollar anchor also matches before a trailing newline, giving the newline as the last part; the end-of-text escape only matches at the very end (and is translated for pyarrow); a literal escaped backslash-Z still splits.
    # https://github.com/pandas-dev/pandas/pull/63613
    ser = Series(["baz", "bar", "bars", "bar\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.split("r$", regex=True)
    expected = Series([["baz"], ["ba", ""], ["bars"], ["ba", "\n"]], dtype=object)
    tm.assert_series_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.split(r"r\Z", regex=True)
    expected = Series([["baz"], ["ba", ""], ["bars"], ["bar\n"]], dtype=object)
    tm.assert_series_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"bar\Z", "bar", r"bar\Zs", "bar\n"], dtype=any_string_dtype)
    result = ser.str.split(r"r\\Z", regex=True)
    expected = Series([["ba", ""], ["bar"], ["ba", "s"], ["bar\n"]], dtype=object)
    tm.assert_series_equal(result, expected)


def test_rsplit_to_dataframe_expand_no_splits(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expanded right split with no separator found gives a one-column frame.
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"])}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_dataframe_expand(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expanded right split with default, maximum two, and maximum one splits gives the expected columns, counting splits from the right.
    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    result = s.str.rsplit("_", expand=True, n=2)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    result = s.str.rsplit("_", expand=True, n=1)
    exp = DataFrame(
        {0: ["some_equal", "with_no"], 1: ["splits", "nans"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_dataframe_expand_with_index(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies expanded right split keeps the original index.
    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_multiindex_expand_no_split():
    # [FUNCTION DOC]
    # Verifies expanded right split with no separators on an Index returns an equal Index with one level.
    idx = Index(["nosplit", "alsonosplit"])
    result = idx.str.rsplit("_", expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1


def test_rsplit_to_multiindex_expand():
    # [FUNCTION DOC]
    # Verifies expanded right split on an Index gives a three-level MultiIndex.
    idx = Index(["some_equal_splits", "with_no_nans"])
    result = idx.str.rsplit("_", expand=True)
    exp = MultiIndex.from_tuples([("some", "equal", "splits"), ("with", "no", "nans")])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3


def test_rsplit_to_multiindex_expand_n():
    # [FUNCTION DOC]
    # Verifies expanded right split with a maximum of one split on an Index gives a two-level MultiIndex.
    idx = Index(["some_equal_splits", "with_no_nans"])
    result = idx.str.rsplit("_", expand=True, n=1)
    exp = MultiIndex.from_tuples([("some_equal", "splits"), ("with_no", "nans")])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 2


def test_split_nan_expand(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies expanded split with a missing value gives a row of missing values in the right dtype, and that the entries really are NaN (object-like dtypes) or NA (nullable) rather than None.
    # gh-18450
    s = Series(["foo,bar,baz", np.nan], dtype=any_string_dtype)
    result = s.str.split(",", expand=True)
    exp = DataFrame(
        [["foo", "bar", "baz"], [np.nan, np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)

    # check that these are actually np.nan/pd.NA and not None
    # TODO see GH 18463
    # tm.assert_frame_equal does not differentiate
    if is_object_or_nan_string_dtype(any_string_dtype):
        assert all(np.isnan(x) for x in result.iloc[1])
    else:
        assert all(x is pd.NA for x in result.iloc[1])


def test_split_with_name_series(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies the Series name is kept for non-expanded split and that the expanded frame has the expected values.
    # GH 12617

    # should preserve name
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    res = s.str.split(",")
    exp = Series([["a", "b"], ["c", "d"]], name="xxx")
    tm.assert_series_equal(res, exp)

    res = s.str.split(",", expand=True)
    exp = DataFrame([["a", "b"], ["c", "d"]], dtype=any_string_dtype)
    tm.assert_frame_equal(res, exp)


def test_split_with_name_index():
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies non-expanded split on a named Index returns a one-level Index of lists keeping the name, and expanded split gives a two-level MultiIndex.
    # GH 12617
    idx = Index(["a,b", "c,d"], name="xxx")
    res = idx.str.split(",")
    exp = Index([["a", "b"], ["c", "d"]], name="xxx")
    assert res.nlevels == 1
    tm.assert_index_equal(res, exp)

    res = idx.str.split(",", expand=True)
    exp = MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    assert res.nlevels == 2
    tm.assert_index_equal(res, exp)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [
                ("a", "__", "b__c"),
                ("c", "__", "d__e"),
                np.nan,
                ("f", "__", "g__h"),
                None,
            ],
        ],
        [
            "rpartition",
            [
                ("a__b", "__", "c"),
                ("c__d", "__", "e"),
                np.nan,
                ("f__g", "__", "h"),
                None,
            ],
        ],
    ],
)
def test_partition_series_more_than_one_char(method, exp, any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies a two-character separator gives a Series of 3-tuples with missing values for NaN and None inputs.
    # https://github.com/pandas-dev/pandas/issues/23558
    # more than one char
    s = Series(["a__b__c", "c__d__e", np.nan, "f__g__h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("__", expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("a", " ", "b c"), ("c", " ", "d e"), np.nan, ("f", " ", "g h"), None],
        ],
        [
            "rpartition",
            [("a b", " ", "c"), ("c d", " ", "e"), np.nan, ("f g", " ", "h"), None],
        ],
    ],
)
def test_partition_series_none(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies default whitespace separator partition gives tuples, keeping missing values.
    # https://github.com/pandas-dev/pandas/issues/23558
    # None
    s = Series(["a b c", "c d e", np.nan, "f g h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)(expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("abc", "", ""), ("cde", "", ""), np.nan, ("fgh", "", ""), None],
        ],
        [
            "rpartition",
            [("", "", "abc"), ("", "", "cde"), np.nan, ("", "", "fgh"), None],
        ],
    ],
)
def test_partition_series_not_split(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies strings without the separator give tuples with the whole string in the first slot (partition) or last slot (right partition) and empty strings elsewhere.
    # https://github.com/pandas-dev/pandas/issues/23558
    # Not split
    s = Series(["abc", "cde", np.nan, "fgh", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("_", expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("a", "_", "b_c"), ("c", "_", "d_e"), np.nan, ("f", "_", "g_h")],
        ],
        [
            "rpartition",
            [("a_b", "_", "c"), ("c_d", "_", "e"), np.nan, ("f_g", "_", "h")],
        ],
    ],
)
def test_partition_series_unicode(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies partition with an underscore separator gives the expected tuples.
    # https://github.com/pandas-dev/pandas/issues/23558
    # unicode
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    result = getattr(s.str, method)("_", expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["partition", "rpartition"])
def test_partition_series_stdlib(any_string_dtype, method):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies the non-expanded result list equals the Python standard library partition of each element.
    # https://github.com/pandas-dev/pandas/issues/23558
    # compare to standard lib
    s = Series(["A_B_C", "B_C_D", "E_F_G", "EFGHEF"], dtype=any_string_dtype)
    result = getattr(s.str, method)("_", expand=False).tolist()
    assert result == [getattr(v, method)("_") for v in s]


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            [("a", "_", "b_c"), ("c", "_", "d_e"), ("f", "_", "g_h"), np.nan, None],
        ],
        [
            "rpartition",
            [("a_b", "_", "c"), ("c_d", "_", "e"), ("f_g", "_", "h"), np.nan, None],
        ],
    ],
)
def test_partition_index(method, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies non-expanded partition on an Index gives an object Index of tuples with one level and missing entries kept.
    # https://github.com/pandas-dev/pandas/issues/23558

    values = Index(["a_b_c", "c_d_e", "f_g_h", np.nan, None])

    result = getattr(values.str, method)("_", expand=False)
    exp = Index(np.array(exp, dtype=object), dtype=object)
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            {
                0: ["a", "c", np.nan, "f", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["b_c", "d_e", np.nan, "g_h", None],
            },
        ],
        [
            "rpartition",
            {
                0: ["a_b", "c_d", np.nan, "f_g", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["c", "e", np.nan, "h", None],
            },
        ],
    ],
)
def test_partition_to_dataframe(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over partition and right partition. Verifies the default (expanding) partition gives a three-column frame of the dtype.
    # https://github.com/pandas-dev/pandas/issues/23558

    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("_")
    expected = DataFrame(
        exp,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        [
            "partition",
            {
                0: ["a", "c", np.nan, "f", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["b_c", "d_e", np.nan, "g_h", None],
            },
        ],
        [
            "rpartition",
            {
                0: ["a_b", "c_d", np.nan, "f_g", None],
                1: ["_", "_", np.nan, "_", None],
                2: ["c", "e", np.nan, "h", None],
            },
        ],
    ],
)
def test_partition_to_dataframe_from_series(any_string_dtype, method, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Same as the previous test but passing expand true explicitly.
    # https://github.com/pandas-dev/pandas/issues/23558
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    result = getattr(s.str, method)("_", expand=True)
    expected = DataFrame(
        exp,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_partition_with_name(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies partition on a named Series gives a three-column frame with the expected values.
    # GH 12617

    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    result = s.str.partition(",")
    expected = DataFrame(
        {0: ["a", "c"], 1: [",", ","], 2: ["b", "d"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_partition_with_name_expand(any_string_dtype):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Verifies non-expanded partition on a named Series returns a Series of tuples keeping the name.
    # GH 12617
    # should preserve name
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    result = s.str.partition(",", expand=False)
    expected = Series([("a", ",", "b"), ("c", ",", "d")], name="xxx")
    tm.assert_series_equal(result, expected)


def test_partition_index_with_name():
    # [FUNCTION DOC]
    # Verifies partition on a named Index gives a three-level MultiIndex.
    idx = Index(["a,b", "c,d"], name="xxx")
    result = idx.str.partition(",")
    expected = MultiIndex.from_tuples([("a", ",", "b"), ("c", ",", "d")])
    assert result.nlevels == 3
    tm.assert_index_equal(result, expected)


def test_partition_index_with_name_expand_false():
    # [FUNCTION DOC]
    # Verifies non-expanded partition on a named Index keeps the name and has one level.
    idx = Index(["a,b", "c,d"], name="xxx")
    # should preserve name
    result = idx.str.partition(",", expand=False)
    expected = Index(np.array([("a", ",", "b"), ("c", ",", "d")]), name="xxx")
    assert result.nlevels == 1
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("method", ["partition", "rpartition"])
def test_partition_sep_kwarg(any_string_dtype, method):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue (the old keyword was replaced by the separator keyword). Parametrized over partition and right partition. Verifies passing the separator by keyword and by position give equal frames.
    # GH 22676; depr kwarg "pat" in favor of "sep"
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    expected = getattr(s.str, method)(sep="_")
    result = getattr(s.str, method)("_")
    tm.assert_frame_equal(result, expected)


def test_get():
    # [FUNCTION DOC]
    # Verifies the element-get method after splitting returns the second part, with NaN for the missing value.
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"])
    result = ser.str.split("_").str.get(1)
    expected = Series(["b", "d", np.nan, "g"], dtype=object)
    tm.assert_series_equal(result, expected)


def test_get_mixed_object():
    # [FUNCTION DOC]
    # Verifies element-get after splitting an object Series with non-strings gives parts for strings and missing values for others.
    ser = Series(["a_b_c", np.nan, "c_d_e", True, datetime.today(), None, 1, 2.0])
    result = ser.str.split("_").str.get(1)
    expected = Series(
        ["b", np.nan, "d", np.nan, np.nan, None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("idx", [2, -3])
def test_get_bounds(idx):
    # [FUNCTION DOC]
    # Parametrized over a positive and a negative index. Verifies element-get on split lists returns the element at that position and NaN when the list is too short.
    ser = Series(["1_2_3_4_5", "6_7_8_9_10", "11_12"])
    result = ser.str.split("_").str.get(idx)
    expected = Series(["3", "8", np.nan], dtype=object)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "idx, exp", [[2, [3, 3, np.nan, "b"]], [-1, [3, 3, np.nan, np.nan]]]
)
def test_get_complex(idx, exp):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue (getting a value not in a dictionary raised a key error). Parametrized over index 2 and minus 1. Verifies element-get on tuples, lists, sets and dicts: sequences give elements, sets and absent dictionary keys give NaN, a present dictionary key gives its value.
    # GH 20671, getting value not in dict raising `KeyError`
    ser = Series([(1, 2, 3), [1, 2, 3], {1, 2, 3}, {1: "a", 2: "b", 3: "c"}])

    result = ser.str.get(idx)
    expected = Series(exp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("to_type", [tuple, list, np.array])
def test_get_complex_nested(to_type):
    # [FUNCTION DOC]
    # Parametrized over tuple, list and array constructors. Verifies element-get on nested sequences returns the inner sequence at index zero and NaN beyond the end.
    ser = Series([to_type([to_type([1, 2])])])

    result = ser.str.get(0)
    expected = Series([to_type([1, 2])])
    tm.assert_series_equal(result, expected)

    result = ser.str.get(1)
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_get_strings(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies element-get with index two on strings returns the character where long enough and missing otherwise.
    ser = Series(["a", "ab", np.nan, "abc"], dtype=any_string_dtype)
    result = ser.str.get(2)
    expected = Series([np.nan, np.nan, np.nan, "c"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["split", "rsplit", "partition", "rpartition"])
def test_split_category_returns_list_not_string(any_string_dtype, method):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over split, right split, partition, right partition. On a categorical Series of strings with a missing value, verifies non-expanded results are object dtype Series of lists (split) or tuples (partition) and missing for the missing value.
    # GH#66341
    ser = Series(
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            np.nan,
        ],
        dtype=any_string_dtype,
    ).astype("category")

    if method in ("split", "rsplit"):
        result = getattr(ser.str, method)()
        expected = Series(
            [
                ["this", "is", "a", "regular", "sentence"],
                ["https://docs.python.org/3/tutorial/index.html"],
                np.nan,
            ]
        )
    elif method == "partition":
        result = ser.str.partition(expand=False)
        expected = Series(
            [
                ("this", " ", "is a regular sentence"),
                ("https://docs.python.org/3/tutorial/index.html", "", ""),
                np.nan,
            ]
        )
    else:
        result = ser.str.rpartition(expand=False)
        expected = Series(
            [
                ("this is a regular", " ", "sentence"),
                ("", "", "https://docs.python.org/3/tutorial/index.html"),
                np.nan,
            ]
        )

    assert result.dtype == object
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["split", "rsplit", "partition", "rpartition"])
def test_split_category_expand(any_string_dtype, method):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. Parametrized over the four methods. Verifies expanded results on a categorical Series are frames whose column dtypes equal the categories' dtype.
    # GH#66341 - expand=True on categorical should return string columns
    ser = Series(
        ["a b", "c d"],
        dtype=any_string_dtype,
    ).astype("category")

    result = getattr(ser.str, method)(expand=True)

    expected_dtype = ser.dtype.categories.dtype
    assert isinstance(result, DataFrame)
    for col_dtype in result.dtypes:
        assert col_dtype == expected_dtype
