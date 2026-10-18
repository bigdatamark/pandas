# [MODULE DOC]
# Tests for the string accessor's regular-expression group extraction methods: extract (first match, returning a Series/Index for one group with expand off, or a frame) and extractall (all matches, returning a frame with a multi-level index whose last level, named match, numbers the match within each subject).
# Covers group naming, optional groups, non-capturing groups, errors, mixed object data, index inputs, dtype preservation, lookaround patterns and end-of-string anchors. Uses the any-string-dtype and box fixtures from the shared conftest files. No import-time side effects.

from datetime import datetime
import re

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import ArrowDtype

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


def test_extract_expand_kwarg_wrong_type_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that extract with a non-boolean expand value (none) raises a value error saying expand must be true or false, for any string dtype.
    # KNOWN ISSUE: a leading TODO questions whether this should be a type error instead of a value error; the test asserts the value error behavior.
    # TODO: should this raise TypeError
    values = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    with pytest.raises(ValueError, match="expand must be True or False"):
        values.str.extract(".*(BAD[_]+).*(BAD)", expand=None)


def test_extract_expand_kwarg(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies the expand argument: the default and an explicit true both return a one-column frame for a single group; expand off with two groups returns a two-column frame. NaN input and non-matching rows give NaN. Result dtype matches the input string dtype.
    s = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    expected = DataFrame(["BAD__", np.nan, np.nan], dtype=any_string_dtype)

    result = s.str.extract(".*(BAD[_]+).*")
    tm.assert_frame_equal(result, expected)

    result = s.str.extract(".*(BAD[_]+).*", expand=True)
    tm.assert_frame_equal(result, expected)

    expected = DataFrame(
        [["BAD__", "BAD"], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    result = s.str.extract(".*(BAD[_]+).*(BAD)", expand=False)
    tm.assert_frame_equal(result, expected)


def test_extract_expand_False_mixed_object():
    # [FUNCTION DOC]
    # Verifies extract with expand off on object data mixing strings, NaN, boolean, datetime, None and numbers: two groups give a frame with an all-NaN row for non-matching and non-string entries; a single group gives an object Series with NaN, except None input stays None.
    ser = Series(
        ["aBAD_BAD", np.nan, "BAD_b_BAD", True, datetime.today(), "foo", None, 1, 2.0]
    )

    # two groups
    result = ser.str.extract(".*(BAD[_]+).*(BAD)", expand=False)
    er = [np.nan, np.nan]  # empty row
    expected = DataFrame(
        [["BAD_", "BAD"], er, ["BAD_", "BAD"], er, er, er, er, er, er], dtype=object
    )
    tm.assert_frame_equal(result, expected)

    # single group
    result = ser.str.extract(".*(BAD[_]+).*BAD", expand=False)
    expected = Series(
        ["BAD_", np.nan, "BAD_", np.nan, np.nan, np.nan, None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_extract_expand_index_raises():
    # [FUNCTION DOC]
    # Verifies (issue 9980) that extract on an Index with expand off and two groups raises a value error that only one regex group is supported with an Index.
    # GH9980
    # Index only works with one regex group since
    # multi-group would expand to a frame
    idx = Index(["A1", "A2", "A3", "A4", "B5"])
    msg = "only one regex group is supported with Index"
    with pytest.raises(ValueError, match=msg):
        idx.str.extract("([AB])([123])", expand=False)


def test_extract_expand_no_capture_groups_raises(index_or_series, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that extract with expand off raises a value error (no capture groups in the pattern) for patterns with no groups or only non-capturing groups, for Series and Index and any string dtype.
    s_or_idx = index_or_series(["A1", "B2", "C3"], dtype=any_string_dtype)
    msg = "pattern contains no capture groups"

    # no groups
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("[ABC][123]", expand=False)

    # only non-capturing groups
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("(?:[AB]).*", expand=False)


def test_extract_expand_single_capture_group(index_or_series, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that extract with expand off and one named group returns a Series or Index (matching the container kind) of the matched text, renamed to the group name.
    # single group renames series/index properly
    s_or_idx = index_or_series(["A1", "A2"], dtype=any_string_dtype)
    result = s_or_idx.str.extract(r"(?P<uno>A)\d", expand=False)

    expected = index_or_series(["A", "A"], name="uno", dtype=any_string_dtype)
    if index_or_series == Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)


def test_extract_expand_capture_groups(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extract with expand off on a three-element Series across group configurations: one or two groups with no matches (NaN Series or frame), some matches, one or two named groups (Series named by the group or frame with group names as columns), a mix of named and unnamed groups (unnamed column labeled by its group position), non-capturing groups ignored, an optional group before or after a normal group (missing optional group gives NaN).
    s = Series(["A1", "B2", "C3"], dtype=any_string_dtype)
    # one group, no matches
    result = s.str.extract("(_)", expand=False)
    expected = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two groups, no matches
    result = s.str.extract("(_)(_)", expand=False)
    expected = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one group, some matches
    result = s.str.extract("([AB])[123]", expand=False)
    expected = Series(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two groups, some matches
    result = s.str.extract("([AB])([123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named group
    result = s.str.extract("(?P<letter>[AB])", expand=False)
    expected = Series(["A", "B", np.nan], name="letter", dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two named groups
    result = s.str.extract("(?P<letter>[AB])(?P<number>[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # mix named and unnamed groups
    result = s.str.extract("([AB])(?P<number>[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=[0, "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group, one non-capturing group
    result = s.str.extract("([AB])(?:[123])", expand=False)
    expected = Series(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two normal groups, one non-capturing group
    s = Series(["A11", "B22", "C33"], dtype=any_string_dtype)
    result = s.str.extract("([AB])([123])(?:[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one optional group followed by one normal group
    s = Series(["A1", "B2", "3"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group followed by one optional group
    s = Series(["A1", "B2", "C"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[ABC])(?P<number>[123])?", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_expand_capture_groups_index(index, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 6348) that extract with expand off keeps the original index of the Series: the fixture index is repeated as needed and truncated to three entries (skipped if empty), then a one-group pattern returns a Series and a two-named-group pattern with an optional group returns a frame, both carrying the same index.
    # https://github.com/pandas-dev/pandas/issues/6348
    # not passing index to the extractor
    data = ["A1", "B2", "C"]

    if len(index) == 0:
        pytest.skip("Test requires len(index) > 0")
    while len(index) < len(data):
        index = index.repeat(2)

    index = index[: len(data)]
    ser = Series(data, index=index, dtype=any_string_dtype)

    result = ser.str.extract(r"(\d)", expand=False)
    expected = Series(["1", "2", np.nan], index=index, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.extract(r"(?P<letter>\D)(?P<number>\d)?", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        index=index,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_single_series_name_is_preserved(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that for a single named group with expand off the result Series takes the group name, replacing the input Series' name.
    s = Series(["a3", "b3", "c2"], name="bob", dtype=any_string_dtype)
    result = s.str.extract(r"(?P<sue>[a-z])", expand=False)
    expected = Series(["a", "b", "c"], name="sue", dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_extract_expand_True(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extract with expand on and two groups returns a two-column frame (NaN for non-matching and missing rows) with the input dtype; the comment notes it overlaps with match tests.
    # Contains tests like those in test_match and some others.
    s = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)

    result = s.str.extract(".*(BAD[_]+).*(BAD)", expand=True)
    expected = DataFrame(
        [["BAD__", "BAD"], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_extract_expand_True_mixed_object():
    # [FUNCTION DOC]
    # Verifies extract with expand on and two groups on mixed object data (strings, NaN, boolean, datetime, None, numbers): a frame of object dtype with a full-NaN row for non-matching and non-string entries.
    er = [np.nan, np.nan]  # empty row
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

    result = mixed.str.extract(".*(BAD[_]+).*(BAD)", expand=True)
    expected = DataFrame(
        [["BAD_", "BAD"], er, ["BAD_", "BAD"], er, er, er, er, er, er], dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_extract_expand_True_single_capture_group_raises(
    index_or_series, any_string_dtype
):
    # [FUNCTION DOC]
    # Verifies that extract with expand on raises a value error (no capture groups) for patterns with no groups or only non-capturing groups, for Series and Index and any string dtype.
    # these should work for both Series and Index
    # no groups
    s_or_idx = index_or_series(["A1", "B2", "C3"], dtype=any_string_dtype)
    msg = "pattern contains no capture groups"
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("[ABC][123]", expand=True)

    # only non-capturing groups
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("(?:[AB]).*", expand=True)


def test_extract_expand_True_single_capture_group(index_or_series, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that extract with expand on and one named group returns a one-column frame named by the group, for Series and Index and any string dtype.
    # single group renames series/index properly
    s_or_idx = index_or_series(["A1", "A2"], dtype=any_string_dtype)
    result = s_or_idx.str.extract(r"(?P<uno>A)\d", expand=True)
    expected = DataFrame({"uno": ["A", "A"]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("name", [None, "series_name"])
def test_extract_series(name, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extract with expand on over the same group configurations as the expand-off test (no matches, some matches, named, mixed named/unnamed, non-capturing) always returns a frame, and that the result does not depend on whether the Series has a name (parametrized: no name or a name).
    # extract should give the same result whether or not the series has a name.
    s = Series(["A1", "B2", "C3"], name=name, dtype=any_string_dtype)

    # one group, no matches
    result = s.str.extract("(_)", expand=True)
    expected = DataFrame([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two groups, no matches
    result = s.str.extract("(_)(_)", expand=True)
    expected = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one group, some matches
    result = s.str.extract("([AB])[123]", expand=True)
    expected = DataFrame(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two groups, some matches
    result = s.str.extract("([AB])([123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named group
    result = s.str.extract("(?P<letter>[AB])", expand=True)
    expected = DataFrame({"letter": ["A", "B", np.nan]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two named groups
    result = s.str.extract("(?P<letter>[AB])(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # mix named and unnamed groups
    result = s.str.extract("([AB])(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=[0, "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group, one non-capturing group
    result = s.str.extract("([AB])(?:[123])", expand=True)
    expected = DataFrame(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


def test_extract_optional_groups(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extract with expand on for patterns with a non-capturing group, and with an optional group before or after a normal group: absent optional groups give NaN while present ones fill their column; group names become column labels.
    # two normal groups, one non-capturing group
    s = Series(["A11", "B22", "C33"], dtype=any_string_dtype)
    result = s.str.extract("([AB])([123])(?:[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one optional group followed by one normal group
    s = Series(["A1", "B2", "3"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group followed by one optional group
    s = Series(["A1", "B2", "C"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[ABC])(?P<number>[123])?", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_dataframe_capture_groups_index(index, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 6348) that extract with expand on preserves the index of the Series: the fixture index is truncated to three entries (skipped if shorter than three) for a one-group pattern and a two-named-group pattern with an optional group.
    # GH6348
    # not passing index to the extractor

    data = ["A1", "B2", "C"]

    if len(index) < len(data):
        pytest.skip(f"Index needs more than {len(data)} values")

    index = index[: len(data)]
    s = Series(data, index=index, dtype=any_string_dtype)

    result = s.str.extract(r"(\d)", expand=True)
    expected = DataFrame(["1", "2", np.nan], index=index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    result = s.str.extract(r"(?P<letter>\D)(?P<number>\d)?", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        index=index,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_single_group_returns_frame(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 11386) that extract with expand on always returns a frame, even for a single group; the column is named by the group. Before version 0.18.0 a Series was returned.
    # GH11386 extract should always return DataFrame, even when
    # there is only one group. Prior to v0.18.0, extract returned
    # Series when there was only one group in the regex.
    s = Series(["a3", "b3", "c2"], name="series_name", dtype=any_string_dtype)
    result = s.str.extract(r"(?P<letter>[a-z])", expand=True)
    expected = DataFrame({"letter": ["a", "b", "c"]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


def test_extractall(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extractall for email-like strings with a verbose multi-line pattern with named user, domain and top-level-domain groups. Seven subjects (including NaN and empty) yield eight matches: the result has one row per match with a two-level index (subject position and match number named match). The test repeats with a multi-index on the Series (output index extends it with the match level) and with a named multi-index (names preserved).
    data = [
        "dave@google.com",
        "tdhock5@gmail.com",
        "maudelaperriere@gmail.com",
        "rob@gmail.com some text steve@gmail.com",
        "a@b.com some text c@d.com and e@f.com",
        np.nan,
        "",
    ]
    expected_tuples = [
        ("dave", "google", "com"),
        ("tdhock5", "gmail", "com"),
        ("maudelaperriere", "gmail", "com"),
        ("rob", "gmail", "com"),
        ("steve", "gmail", "com"),
        ("a", "b", "com"),
        ("c", "d", "com"),
        ("e", "f", "com"),
    ]
    pat = r"""
    (?P<user>[a-z0-9]+)
    @
    (?P<domain>[a-z]+)
    \.
    (?P<tld>[a-z]{2,4})
    """
    expected_columns = ["user", "domain", "tld"]
    s = Series(data, dtype=any_string_dtype)
    # extractall should return a DataFrame with one row for each match, indexed by the
    # subject from which the match came.
    expected_index = MultiIndex.from_tuples(
        [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)],
        names=(None, "match"),
    )
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)

    # The index of the input Series should be used to construct the index of the output
    # DataFrame:
    mi = MultiIndex.from_tuples(
        [
            ("single", "Dave"),
            ("single", "Toby"),
            ("single", "Maude"),
            ("multiple", "robAndSteve"),
            ("multiple", "abcdef"),
            ("none", "missing"),
            ("none", "empty"),
        ]
    )
    s = Series(data, index=mi, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples(
        [
            ("single", "Dave", 0),
            ("single", "Toby", 0),
            ("single", "Maude", 0),
            ("multiple", "robAndSteve", 0),
            ("multiple", "robAndSteve", 1),
            ("multiple", "abcdef", 0),
            ("multiple", "abcdef", 1),
            ("multiple", "abcdef", 2),
        ],
        names=(None, None, "match"),
    )
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)

    # MultiIndexed subject with names.
    s = Series(data, index=mi, dtype=any_string_dtype)
    s.index.names = ("matches", "description")
    expected_index.names = ("matches", "description", "match")
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "pat,expected_names",
    [
        # optional groups.
        ("(?P<letter>[AB])?(?P<number>[123])", ["letter", "number"]),
        # only one of two groups has a name.
        ("([AB])?(?P<number>[123])", [0, "number"]),
    ],
)
def test_extractall_column_names(pat, expected_names, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extractall column labels: groups named by the pattern use those names; an unnamed group is labeled by its position. Parametrized over a pattern with two named groups and a pattern with one unnamed and one named group. Rows are only produced for matching subjects; optional unmatched groups are NaN.
    s = Series(["", "A1", "32"], dtype=any_string_dtype)

    result = s.str.extractall(pat)
    expected = DataFrame(
        [("A", "1"), (np.nan, "3"), (np.nan, "2")],
        index=MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)], names=(None, "match")),
        columns=expected_names,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_single_group(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extractall with a single group yields a one-column frame with a multi-index (subject, match): a named group gives a column with that name; an unnamed group gives a default-labeled column.
    s = Series(["a3", "b3", "d4c2"], name="series_name", dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples(
        [(0, 0), (1, 0), (2, 0), (2, 1)], names=(None, "match")
    )

    # extractall(one named group) returns DataFrame with one named column.
    result = s.str.extractall(r"(?P<letter>[a-z])")
    expected = DataFrame(
        {"letter": ["a", "b", "d", "c"]}, index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # extractall(one un-named group) returns DataFrame with one un-named column.
    result = s.str.extractall(r"([a-z])")
    expected = DataFrame(
        ["a", "b", "d", "c"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_single_group_with_quantifier(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 13382) extractall with one unnamed group containing a quantifier returns a one-column frame with consecutive matching runs, indexed by subject and match number.
    # GH#13382
    # extractall(one un-named group with quantifier) returns DataFrame with one un-named
    # column.
    s = Series(["ab3", "abc3", "d4cd2"], name="series_name", dtype=any_string_dtype)
    result = s.str.extractall(r"([a-z]+)")
    expected = DataFrame(
        ["ab", "abc", "d", "cd"],
        index=MultiIndex.from_tuples(
            [(0, 0), (1, 0), (2, 0), (2, 1)], names=(None, "match")
        ),
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, names",
    [
        ([], (None,)),
        ([], ("i1",)),
        ([], (None, "i2")),
        ([], ("i1", "i2")),
        (["a3", "b3", "d4c2"], (None,)),
        (["a3", "b3", "d4c2"], ("i1", "i2")),
        (["a3", "b3", "d4c2"], (None, "i2")),
    ],
)
def test_extractall_no_matches(data, names, any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (issue 19075) that extractall with no matches returns an empty frame with a valid multi-index whose names are the original index names followed by match, for empty and non-empty Series with single or multi-level indexes (parametrized by data and index names). Column labels follow the group configuration: one or two unnamed groups give default range columns; named groups give those names; a mix gives zero and the name. Dtype of the empty result equals the input dtype.
    # GH19075 extractall with no matches should return a valid MultiIndex
    n = len(data)
    if len(names) == 1:
        index = Index(range(n), name=names[0])
    else:
        tuples = (tuple([i] * (n - 1)) for i in range(n))
        index = MultiIndex.from_tuples(tuples, names=names)
    s = Series(data, name="series_name", index=index, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples([], names=((*names, "match")))

    # one un-named group.
    result = s.str.extractall("(z)")
    expected = DataFrame(columns=range(1), index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected, check_column_type=True)

    # two un-named groups.
    result = s.str.extractall("(z)(z)")
    expected = DataFrame(columns=range(2), index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected, check_column_type=True)

    # one named group.
    result = s.str.extractall("(?P<first>z)")
    expected = DataFrame(
        columns=["first"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # two named groups.
    result = s.str.extractall("(?P<first>z)(?P<second>z)")
    expected = DataFrame(
        columns=["first", "second"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named, one un-named.
    result = s.str.extractall("(z)(?P<second>z)")
    expected = DataFrame(
        columns=[0, "second"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_stringindex(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies extractall on a Series with a default index and with a string index with a name: the result index is the original index (keeping its name) plus match. For object dtype it additionally checks that an Index of strings, with or without a name, gives the same result as the default-indexed Series.
    s = Series(["a1a2", "b1", "c1"], name="xxx", dtype=any_string_dtype)
    result = s.str.extractall(r"[ab](?P<digit>\d)")
    expected = DataFrame(
        {"digit": ["1", "2", "1"]},
        index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=[None, "match"]),
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # index should return the same result as the default index without name thus
    # index.name doesn't affect to the result
    if any_string_dtype == "object":
        for idx in [
            Index(["a1a2", "b1", "c1"], dtype=object),
            Index(["a1a2", "b1", "c1"], name="xxx", dtype=object),
        ]:
            result = idx.str.extractall(r"[ab](?P<digit>\d)")
            tm.assert_frame_equal(result, expected)

    s = Series(
        ["a1a2", "b1", "c1"],
        name="s_name",
        index=Index(["XX", "yy", "zz"], name="idx_name"),
        dtype=any_string_dtype,
    )
    result = s.str.extractall(r"[ab](?P<digit>\d)")
    expected = DataFrame(
        {"digit": ["1", "2", "1"]},
        index=MultiIndex.from_tuples(
            [("XX", 0), ("XX", 1), ("yy", 0)], names=["idx_name", "match"]
        ),
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_no_capture_groups_raises(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that extractall with a pattern without capture groups raises a value error mentioning no capture groups.
    # Does not make sense to use extractall with a regex that has no capture groups.
    # (it returns DataFrame with one column for each capture group)
    s = Series(["a3", "b3", "d4c2"], name="series_name", dtype=any_string_dtype)
    with pytest.raises(ValueError, match="no capture groups"):
        s.str.extractall(r"[a-z]")


def test_extract_index_one_two_groups():
    # [FUNCTION DOC]
    # Verifies that extract with expand on applied to an Index of strings returns a frame, for one group (a single unnamed column) and two named groups (columns by group name); before version 0.18.0 one group returned an Index and more groups raised (issue 9980).
    s = Series(["a3", "b3", "d4c2"], index=["A3", "B3", "D4"], name="series_name")
    r = s.index.str.extract(r"([A-Z])", expand=True)
    e = DataFrame(["A", "B", "D"])
    tm.assert_frame_equal(r, e)

    # Prior to v0.18.0, index.str.extract(regex with one group)
    # returned Index. With more than one group, extract raised an
    # error (GH9980). Now extract always returns DataFrame.
    r = s.index.str.extract(r"(?P<letter>[A-Z])(?P<digit>[0-9])", expand=True)
    e_list = [("A", "3"), ("B", "3"), ("D", "4")]
    e = DataFrame(e_list, columns=["letter", "digit"])
    tm.assert_frame_equal(r, e)


def test_extractall_same_as_extract(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies that extractall results, with the match level of the index dropped for match number zero, equal the extract results for the same Series, for patterns with two unnamed groups, two named groups, one named group and one unnamed group.
    s = Series(["a3", "b3", "c2"], name="series_name", dtype=any_string_dtype)

    pattern_two_noname = r"([a-z])([0-9])"
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    has_multi_index = s.str.extractall(pattern_two_noname)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_noname, no_multi_index)

    pattern_two_named = r"(?P<letter>[a-z])(?P<digit>[0-9])"
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    has_multi_index = s.str.extractall(pattern_two_named)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_named, no_multi_index)

    pattern_one_named = r"(?P<group_name>[a-z])"
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    has_multi_index = s.str.extractall(pattern_one_named)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_named, no_multi_index)

    pattern_one_noname = r"([a-z])"
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    has_multi_index = s.str.extractall(pattern_one_noname)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_noname, no_multi_index)


def test_extractall_same_as_extract_subject_index(any_string_dtype):
    # [FUNCTION DOC]
    # Same comparison as the previous extract-versus-extractall test, but for a Series with a named two-level index: dropping the match level from extractall output equals the extract result for the four group configurations.
    # same as above tests, but s has a MultiIndex.
    mi = MultiIndex.from_tuples(
        [("A", "first"), ("B", "second"), ("C", "third")],
        names=("capital", "ordinal"),
    )
    s = Series(["a3", "b3", "c2"], index=mi, name="series_name", dtype=any_string_dtype)

    pattern_two_noname = r"([a-z])([0-9])"
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    has_match_index = s.str.extractall(pattern_two_noname)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_noname, no_match_index)

    pattern_two_named = r"(?P<letter>[a-z])(?P<digit>[0-9])"
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    has_match_index = s.str.extractall(pattern_two_named)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_named, no_match_index)

    pattern_one_named = r"(?P<group_name>[a-z])"
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    has_match_index = s.str.extractall(pattern_one_named)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_named, no_match_index)

    pattern_one_noname = r"([a-z])"
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    has_match_index = s.str.extractall(pattern_one_noname)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_noname, no_match_index)


def test_extractall_preserves_dtype():
    # [FUNCTION DOC]
    # Verifies that extractall on a pyarrow-backed string Series preserves the pyarrow string dtype in the resulting column. Skipped if pyarrow is missing.
    # Ensure that when extractall is called on a series with specific dtypes set, that
    # the dtype is preserved in the resulting DataFrame's column.
    pa = pytest.importorskip("pyarrow")

    result = Series(["abc", "ab"], dtype=ArrowDtype(pa.string())).str.extractall("(ab)")
    assert result.dtypes[0] == "string[pyarrow]"


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"(a(?=b))", [None, "a", None, None]),
        (r"((?<=a)b)", [None, "b", None, None]),
        (r"(a(?!b))", ["a", None, "a", None]),
        (r"((?<!b)a)", ["a", "a", None, None]),
        ("(ab)", [None, "ab", None, None]),
    ],
)
def test_extract_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Verifies (issue 60833) extract with expand off on four strings plus a missing value using lookahead and lookbehind patterns (positive and negative) and a plain pattern. Parametrized over pattern and expected values. For object dtype, entries without a match are NaN instead of None; the final missing input stays missing.
    # https://github.com/pandas-dev/pandas/issues/60833
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.extract(pat, expand=False)
    if any_string_dtype == "object":
        # object input will preserve None but any result with no matches gets NaN
        expected_data = [np.nan if e is None else e for e in expected_data]
    expected = Series([*expected_data, None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, expected_data",
    [
        (r"(a(?=b))", {(1, 0): "a"}),
        (r"((?<=a)b)", {(1, 0): "b"}),
        (r"(a(?!b))", {(0, 0): "a", (0, 1): "a", (2, 0): "a"}),
        (r"((?<!b)a)", {(0, 0): "a", (0, 1): "a", (1, 0): "a"}),
        ("(ab)", {(1, 0): "ab"}),
    ],
)
def test_extractall_lookarounds(any_string_dtype, pat, expected_data):
    # [FUNCTION DOC]
    # Verifies (issue 60833) extractall with lookahead and lookbehind patterns: the result is a one-column frame holding matches keyed by (subject position, match number), index levels named none and match. Parametrized over pattern and expected mapping.
    # https://github.com/pandas-dev/pandas/issues/60833
    ser = Series(["aa", "ab", "ba", "bb", None], dtype=any_string_dtype)
    result = ser.str.extractall(pat)
    expected = Series(expected_data, dtype=any_string_dtype).to_frame()
    expected.index.names = [None, "match"]
    tm.assert_frame_equal(result, expected)


def test_extract_end_of_string(any_string_dtype):
    # [FUNCTION DOC]
    # Verifies (pull request 63613) end-of-string anchors in extract: the dollar anchor matches before a trailing newline, while the absolute end-of-string anchor does not (and must be translated for pyarrow); a pattern with an escaped backslash followed by the letter Z still finds the literal text.
    # https://github.com/pandas-dev/pandas/pull/63613
    ser = Series(["aa", "abc", "bb\n"], dtype=any_string_dtype)

    # with dollar sign
    result = ser.str.extract("([ab]+)$")
    expected = Series(["aa", np.nan, "bb"], dtype=any_string_dtype).to_frame()
    tm.assert_frame_equal(result, expected)

    # with \Z (ensure this is translated to \z for pyarrow)
    result = ser.str.extract(r"([ab]+)\Z")
    expected = Series(["aa", np.nan, np.nan], dtype=any_string_dtype).to_frame()
    tm.assert_frame_equal(result, expected)

    # ensure finding a literal \Z still works
    ser = Series([r"aa\Z", "abc", "bb\\Z\n"], dtype=any_string_dtype)
    result = ser.str.extract(r"([ab]+)\\Z")
    expected = Series(["aa", np.nan, "bb"], dtype=any_string_dtype).to_frame()
    tm.assert_frame_equal(result, expected)
