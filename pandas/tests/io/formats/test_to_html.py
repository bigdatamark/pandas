# [MODULE DOC]
# Tests for HTML output of frames (to_html) and the notebook representation
# (the private repr hook used by notebooks). Most tests compare against
# reference HTML files stored under the formats data directory, read through the
# module-level helper expected_html and the datapath fixture. Covers column
# spacing, escaping, indexes and multi-indexes (sparsify, index off, names),
# truncation, formatters, borders, justify, CSS classes, links, display options
# (min rows, max rows, mathjax, dimensions), and a few exact-string cases.
# Module-level data: a long placeholder-text constant used by the column width
# test. Imports the formatting module under the short name fmt.

from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    get_option,
    option_context,
)

import pandas.io.formats.format as fmt

lorem_ipsum = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex "
    "ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur "
    "sint occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum."
)


def expected_html(datapath, name):
    # [FUNCTION DOC]
    # HELPER: loads a reference HTML file for the tests. Parameters: datapath, the
    # pytest fixture that builds paths inside the test data folder; name, the file
    # name without its extension. Opens the file named with the html extension under
    # the io, formats, data, html data subdirectory as utf-8 and returns the content
    # with trailing whitespace removed. Raises the usual file-not-found error if the
    # file is missing.
    """
    Read HTML file from formats data directory.

    Parameters
    ----------
    datapath : pytest fixture
        The datapath fixture injected into a test by pytest.
    name : str
        The name of the HTML file without the suffix.

    Returns
    -------
    str : contents of HTML file.
    """
    filename = ".".join([name, "html"])
    filepath = datapath("io", "formats", "data", "html", filename)
    with open(filepath, encoding="utf-8") as f:
        html = f.read()
    return html.rstrip()


@pytest.fixture(params=["mixed", "empty"])
def biggie_df_fixture(request):
    # [FUNCTION DOC]
    # Fixture parametrized over the names mixed and empty. Mixed returns a 200-row
    # frame with a random float column and a string column, whose first 21 rows are
    # set to missing in both columns; empty returns a frame with a 200-row index and
    # no columns.
    """Fixture for a big mixed Dataframe and an empty Dataframe"""
    if request.param == "mixed":
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": Index([f"{i}?!" for i in range(200)]),
            },
            index=np.arange(200),
        )
        df.loc[:20, "A"] = np.nan
        df.loc[:20, "B"] = np.nan
        return df
    elif request.param == "empty":
        df = DataFrame(index=np.arange(200))
        return df


@pytest.fixture(params=fmt.VALID_JUSTIFY_PARAMETERS)
def justify(request):
    # [FUNCTION DOC]
    # Fixture parametrized over all justify values accepted by the formatting module
    # (its tuple of valid justify parameters); returns the current value.
    return request.param


@pytest.mark.parametrize("col_space", [30, 50])
def test_to_html_with_col_space(col_space):
    # [FUNCTION DOC]
    # Parametrized on col_space of 30 or 50. Renders a one-row frame and tries to find
    # header cells lines that carry a minimum width equal to the spacing.
    # KNOWN ISSUE: the output is split on a literal backslash-n (raw string) rather than a real newline, so there is only one chunk containing the whole document and the per-line header checks are weaker than intended.
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    # check that col_space affects HTML generation
    # and be very brittle about it.
    result = df.to_html(col_space=col_space)
    hdrs = [x for x in result.split(r"\n") if re.search(r"<th[>\s]", x)]
    assert len(hdrs) > 0
    for h in hdrs:
        assert "min-width" in h
        assert str(col_space) in h


def test_to_html_with_column_specific_col_space_raises():
    # [FUNCTION DOC]
    # A col_space sequence whose length differs from the column count raises a value
    # error about the length mismatch; a mapping with an unknown column name raises a
    # value error saying unknown column.
    df = DataFrame(
        np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
    )

    msg = (
        "Col_space length\\(\\d+\\) should match DataFrame number of columns\\(\\d+\\)"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40])

    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40, 50, 60])

    msg = "unknown column"
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space={"a": "foo", "b": 23, "d": 34})


def test_to_html_with_column_specific_col_space():
    # [FUNCTION DOC]
    # Per-column spacing: a mapping gives the em string and pixel number only for the
    # named columns (others unstyled); a sequence gives each column its own minimum
    # width, with numbers converted to pixels.
    df = DataFrame(
        np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
    )

    result = df.to_html(col_space={"a": "2em", "b": 23})
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    assert 'min-width: 2em;">a</th>' in hdrs[1]
    assert 'min-width: 23px;">b</th>' in hdrs[2]
    assert "<th>c</th>" in hdrs[3]

    result = df.to_html(col_space=["1em", 2, 3])
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    assert 'min-width: 1em;">a</th>' in hdrs[1]
    assert 'min-width: 2px;">b</th>' in hdrs[2]
    assert 'min-width: 3px;">c</th>' in hdrs[3]


def test_to_html_with_empty_string_label():
    # [FUNCTION DOC]
    # GH 3547. In a two-level index, an empty-string label must not be treated as a
    # repeated label, so no row-span attribute appears in the output.
    # GH 3547, to_html regards empty string labels as repeated labels
    data = {"c1": ["a", "b"], "c2": ["a", ""], "data": [1, 2]}
    df = DataFrame(data).set_index(["c1", "c2"])
    result = df.to_html()
    assert "rowspan" not in result


@pytest.mark.parametrize(
    "df_data,expected",
    [
        ({"\u03c3": np.arange(10.0)}, "unicode_1"),
        ({"A": ["\u03c3"]}, "unicode_2"),
    ],
)
def test_to_html_unicode(df_data, expected, datapath):
    # [FUNCTION DOC]
    # Parametrized on frame data (a non-ASCII column label and a non-ASCII cell)
    # and the matching reference file name. The output must equal the reference.
    df = DataFrame(df_data)
    expected = expected_html(datapath, expected)
    result = df.to_html()
    assert result == expected


def test_to_html_encoding(float_frame, temp_file):
    # [FUNCTION DOC]
    # GH 28663. Writing to a path with the gbk encoding and reading the file back
    # with that encoding must equal the string output.
    # GH 28663
    float_frame.to_html(temp_file, encoding="gbk")
    expected = temp_file.read_text(encoding="gbk")
    assert float_frame.to_html() == expected


def test_to_html_decimal(datapath):
    # [FUNCTION DOC]
    # GH 12031. The decimal separator argument is applied; output equals the
    # reference file.
    # GH 12031
    df = DataFrame({"A": [6.0, 3.1, 2.2]})
    result = df.to_html(decimal=",")
    expected = expected_html(datapath, "gh12031_expected_output")
    assert result == expected


@pytest.mark.parametrize(
    "kwargs,string,expected",
    [
        ({}, "<type 'str'>", "escaped"),
        ({"escape": False}, "<b>bold</b>", "escape_disabled"),
    ],
)
def test_to_html_escaped(kwargs, string, expected, datapath):
    # [FUNCTION DOC]
    # Parametrized on keyword arguments, a cell text and the reference name. Column
    # and index labels and cell text containing angle brackets and ampersands are
    # escaped by default and left raw when escape is false.
    a = "str<ing1 &amp;"
    b = "stri>ng2 &amp;"

    test_dict = {"co<l1": {a: string, b: string}, "co>l2": {a: string, b: string}}
    result = DataFrame(test_dict).to_html(**kwargs)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize("index_is_named", [True, False])
def test_to_html_multiindex_index_false(index_is_named, datapath):
    # [FUNCTION DOC]
    # GH 8452. Parametrized on whether the index is named. A frame with two-level
    # columns written with index off equals the reference file in both cases.
    # GH 8452
    df = DataFrame(
        {"a": range(2), "b": range(3, 5), "c": range(5, 7), "d": range(3, 5)}
    )
    df.columns = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    if index_is_named:
        df.index = Index(df.index.values, name="idx")
    result = df.to_html(index=False)
    expected = expected_html(datapath, "gh8452_expected_output")
    assert result == expected


@pytest.mark.parametrize(
    "multi_sparse,expected",
    [
        (False, "multiindex_sparsify_false_multi_sparse_1"),
        (False, "multiindex_sparsify_false_multi_sparse_2"),
        (True, "multiindex_sparsify_1"),
        (True, "multiindex_sparsify_2"),
    ],
)
def test_to_html_multiindex_sparsify(multi_sparse, expected, datapath):
    # [FUNCTION DOC]
    # Parametrized on the multi-sparse option and the reference name. A frame with a
    # two-level named row index, and (for names ending in 2) a multi-index as
    # columns, is rendered under the option and compared to the reference.
    index = MultiIndex.from_arrays([[0, 0, 1, 1], [0, 1, 0, 1]], names=["foo", None])
    df = DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], index=index)
    if expected.endswith("2"):
        df.columns = index[::2]
    with option_context("display.multi_sparse", multi_sparse):
        result = df.to_html()
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "max_rows,expected",
    [
        (60, "gh14882_expected_output_1"),
        # Test that ... appears in a middle level
        (56, "gh14882_expected_output_2"),
    ],
)
def test_to_html_multiindex_odd_even_truncate(max_rows, expected, datapath):
    # [FUNCTION DOC]
    # GH 14882. Parametrized on max_rows (60 and 56) and the reference. A frame
    # indexed by a three-level product of 63 rows is truncated, including at a
    # middle level for the smaller limit.
    # GH 14882 - Issue on truncation with odd length DataFrame
    index = MultiIndex.from_product(
        [[100, 200, 300], [10, 20, 30], [1, 2, 3, 4, 5, 6, 7]], names=["a", "b", "c"]
    )
    df = DataFrame({"n": range(len(index))}, index=index)
    result = df.to_html(max_rows=max_rows)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "df,formatters,expected",
    [
        (
            DataFrame(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                columns=Index(["foo", None], dtype=object),
                index=np.arange(4),
            ),
            {"__index__": lambda x: "abcd"[x]},
            "index_formatter",
        ),
        (
            DataFrame({"months": [datetime(2016, 1, 1), datetime(2016, 2, 2)]}),
            {"months": lambda x: x.strftime("%Y-%m")},
            "datetime64_monthformatter",
        ),
        (
            DataFrame(
                {
                    "hod": pd.to_datetime(
                        ["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f"
                    )
                }
            ),
            {"hod": lambda x: x.strftime("%H:%M")},
            "datetime64_hourformatter",
        ),
        (
            DataFrame(
                {
                    "i": pd.Series([1, 2], dtype="int64"),
                    "f": pd.Series([1, 2], dtype="float64"),
                    "I": pd.Series([1, 2], dtype="Int64"),
                    "s": pd.Series([1, 2], dtype="string"),
                    "b": pd.Series([True, False], dtype="boolean"),
                    "c": pd.Series(["a", "b"], dtype=pd.CategoricalDtype(["a", "b"])),
                    "o": pd.Series([1, "2"], dtype=object),
                }
            ),
            [lambda x: "formatted"] * 7,
            "various_dtypes_formatted",
        ),
    ],
)
def test_to_html_formatters(df, formatters, expected, datapath):
    # [FUNCTION DOC]
    # Parametrized on a frame, a formatters argument and a reference name: an index
    # formatter mapping, datetime formatters in month and hour forms, and one
    # callable per column for a frame covering many dtypes. The output must equal the
    # reference.
    expected = expected_html(datapath, expected)
    result = df.to_html(formatters=formatters)
    assert result == expected


def test_to_html_regression_GH6098():
    # [FUNCTION DOC]
    # Smoke test: a pivot table over a frame with non-ASCII labels can produce its
    # notebook HTML without error.
    df = DataFrame(
        {
            "clé1": ["a", "a", "b", "b", "a"],
            "clé2": ["1er", "2ème", "1er", "2ème", "1er"],
            "données1": np.random.default_rng(2).standard_normal(5),
            "données2": np.random.default_rng(2).standard_normal(5),
        }
    )

    # it works
    df.pivot_table(index=["clé1"], columns=["clé2"])._repr_html_()


def test_to_html_truncate(datapath):
    # [FUNCTION DOC]
    # A 20 by 20 frame with a date index rendered with eight rows and four columns
    # maximum equals the reference file.
    index = pd.date_range(start="20010101", freq="D", periods=20)
    df = DataFrame(index=index, columns=range(20))
    result = df.to_html(max_rows=8, max_cols=4)
    expected = expected_html(datapath, "truncate")
    assert result == expected


@pytest.mark.parametrize("size", [1, 5])
def test_html_invalid_formatters_arg_raises(size):
    # [FUNCTION DOC]
    # Issue 28469. Parametrized on length one or five. A formatters sequence whose
    # length differs from the three columns raises a value error whose message gives
    # the supplied length and the column count.
    # issue-28469
    df = DataFrame(columns=["a", "b", "c"])
    msg = "Formatters length({}) should match DataFrame number of columns(3)"
    with pytest.raises(ValueError, match=re.escape(msg.format(size))):
        df.to_html(formatters=["{}".format] * size)


def test_to_html_truncate_formatter(datapath):
    # [FUNCTION DOC]
    # Issue 25955. Formatters given as a sequence (two callables and two none) with
    # the maximum column count of three; the output must equal the reference,
    # showing that formatting aligns with the truncated columns.
    # KNOWN ISSUE: the local variable named fmt shadows the module-level import of the formatting module within this test; harmless, but misleading.
    # issue-25955
    data = [
        {"A": 1, "B": 2, "C": 3, "D": 4},
        {"A": 5, "B": 6, "C": 7, "D": 8},
        {"A": 9, "B": 10, "C": 11, "D": 12},
        {"A": 13, "B": 14, "C": 15, "D": 16},
    ]

    df = DataFrame(data)
    fmt = lambda x: str(x) + "_mod"
    formatters = [fmt, fmt, None, None]
    result = df.to_html(formatters=formatters, max_cols=3)
    expected = expected_html(datapath, "truncate_formatter")
    assert result == expected


@pytest.mark.parametrize(
    "sparsify,expected",
    [(True, "truncate_multi_index"), (False, "truncate_multi_index_sparse_off")],
)
def test_to_html_truncate_multi_index(sparsify, expected, datapath):
    # [FUNCTION DOC]
    # Parametrized on sparsify. A frame with a two-level index on both axes,
    # rendered with at most seven rows and columns, equals the matching reference.
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]
    df = DataFrame(index=arrays, columns=arrays)
    result = df.to_html(max_rows=7, max_cols=7, sparsify=sparsify)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "option,result,expected",
    [
        (None, lambda df: df.to_html(), "1"),
        (None, lambda df: df.to_html(border=2), "2"),
        (2, lambda df: df.to_html(), "2"),
        (2, lambda df: df._repr_html_(), "2"),
    ],
)
def test_to_html_border(option, result, expected):
    # [FUNCTION DOC]
    # Parametrized on an option value, a callable producing output, and the expected
    # border value. Checks the border attribute in the output: default 1; the border
    # argument 2; and the display option set to 2 through both to_html and the
    # notebook repr.
    df = DataFrame({"A": [1, 2]})
    if option is None:
        result = result(df)
    else:
        with option_context("display.html.border", option):
            result = result(df)
    expected = f'border="{expected}"'
    assert expected in result


@pytest.mark.parametrize("biggie_df_fixture", ["mixed"], indirect=True)
def test_to_html(biggie_df_fixture):
    # [FUNCTION DOC]
    # Uses the mixed big frame. Writing into a buffer returns none and equals the
    # string result, which is a string; various calls with column selection, spacing,
    # formatters and a string float format must not fail. A comment notes the test
    # should be split.
    # TODO: split this test
    df = biggie_df_fixture
    s = df.to_html()

    buf = StringIO()
    retval = df.to_html(buf=buf)
    assert retval is None
    assert buf.getvalue() == s

    assert isinstance(s, str)

    df.to_html(columns=["B", "A"], col_space=17)
    df.to_html(columns=["B", "A"], formatters={"A": lambda x: f"{x:.1f}"})

    df.to_html(columns=["B", "A"], float_format=str)
    df.to_html(columns=["B", "A"], col_space=12, float_format=str)


@pytest.mark.parametrize("biggie_df_fixture", ["empty"], indirect=True)
def test_to_html_empty_dataframe(biggie_df_fixture):
    # [FUNCTION DOC]
    # Uses the empty frame; rendering must not fail.
    df = biggie_df_fixture
    df.to_html()


def test_to_html_filename(biggie_df_fixture, tmpdir):
    # [FUNCTION DOC]
    # Rendering to a path (the legacy temp directory object) writes the same text as
    # the string output.
    df = biggie_df_fixture
    expected = df.to_html()
    path = tmpdir.join("test.html")
    df.to_html(path)
    result = path.read()
    assert result == expected


def test_to_html_with_no_bold():
    # [FUNCTION DOC]
    # Intended to check that bold_rows false produces no strong tags in the body.
    # KNOWN ISSUE: the code indexes the string with the position found for the head-closing tag, yielding a single character instead of a slice, so the assertion that no strong tag appears is trivially true and checks nothing.
    df = DataFrame({"x": np.random.default_rng(2).standard_normal(5)})
    html = df.to_html(bold_rows=False)
    result = html[html.find("</thead>")]
    assert "<strong" not in result


def test_to_html_columns_arg(float_frame):
    # [FUNCTION DOC]
    # Selecting only column A means the header cell for column B is absent.
    result = float_frame.to_html(columns=["A"])
    assert "<th>B</th>" not in result


@pytest.mark.parametrize(
    "columns,justify,expected",
    [
        (
            MultiIndex.from_arrays(
                [np.arange(2).repeat(2), np.mod(range(4), 2)],
                names=["CL0", "CL1"],
            ),
            "left",
            "multiindex_1",
        ),
        (
            MultiIndex.from_arrays([np.arange(4), np.mod(range(4), 2)]),
            "right",
            "multiindex_2",
        ),
    ],
)
def test_to_html_multiindex(columns, justify, expected, datapath):
    # [FUNCTION DOC]
    # Parametrized on a column multi-index, justify and reference name. A two-row
    # text frame with multi-level columns is rendered with that justify and compared
    # to the reference.
    df = DataFrame([list("abcd"), list("efgh")], columns=columns)
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, expected)
    assert result == expected


def test_to_html_justify(justify, datapath):
    # [FUNCTION DOC]
    # Parametrized by the justify fixture. A numeric frame is rendered with each
    # valid justify value and compared with the reference text with the justify
    # value substituted into its placeholder.
    df = DataFrame(
        {"A": [6, 30000, 2], "B": [1, 2, 70000], "C": [223442, 0, 1]},
        columns=["A", "B", "C"],
    )
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, "justify").format(justify=justify)
    assert result == expected


@pytest.mark.parametrize(
    "justify", ["super-right", "small-left", "noinherit", "tiny", "pandas"]
)
def test_to_html_invalid_justify(justify):
    # [FUNCTION DOC]
    # GH 17527. Parametrized on invalid justify names; raises a value error saying
    # invalid value for justify parameter.
    # GH 17527
    df = DataFrame()
    msg = "Invalid value for justify parameter"

    with pytest.raises(ValueError, match=msg):
        df.to_html(justify=justify)


class TestHTMLIndex:
    @pytest.fixture
    # [CLASS DOC]
    # Tests of index display options for the HTML writer, sharing a small three-row
    # frame fixture and a reference for the no-index output.
    def df(self):
        # [FUNCTION DOC]
        # Fixture: a three-row frame with integer, float and string-with-missing columns
        # and string index foo, bar, baz.
        index = ["foo", "bar", "baz"]
        df = DataFrame(
            {"A": [1, 2, 3], "B": [1.2, 3.4, 5.6], "C": ["one", "two", np.nan]},
            columns=["A", "B", "C"],
            index=index,
        )
        return df

    @pytest.fixture
    def expected_without_index(self, datapath):
        # [FUNCTION DOC]
        # Fixture: the reference HTML for output with the index suppressed.
        return expected_html(datapath, "index_2")

    def test_to_html_flat_index_without_name(
        self, datapath, df, expected_without_index
    ):
        # [FUNCTION DOC]
        # Default output equals the indexed reference. With index off no index label
        # appears in the output and the result equals the no-index reference.
        expected_with_index = expected_html(datapath, "index_1")
        assert df.to_html() == expected_with_index

        result = df.to_html(index=False)
        for i in df.index:
            assert i not in result
        assert result == expected_without_index

    def test_to_html_flat_index_with_name(self, datapath, df, expected_without_index):
        # [FUNCTION DOC]
        # With a named flat index the default output equals the named reference, and index
        # off still gives the shared no-index reference.
        df.index = Index(["foo", "bar", "baz"], name="idx")
        expected_with_index = expected_html(datapath, "index_3")
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index

    def test_to_html_multiindex_without_names(
        self, datapath, df, expected_without_index
    ):
        # [FUNCTION DOC]
        # With an unnamed two-level index the default equals its reference; with index off
        # none of the level labels appear and the result equals the no-index reference.
        tuples = [("foo", "car"), ("foo", "bike"), ("bar", "car")]
        df.index = MultiIndex.from_tuples(tuples)

        expected_with_index = expected_html(datapath, "index_4")
        assert df.to_html() == expected_with_index

        result = df.to_html(index=False)
        for i in ["foo", "bar", "car", "bike"]:
            assert i not in result
        # must be the same result as normal index
        assert result == expected_without_index

    def test_to_html_multiindex_with_names(self, datapath, df, expected_without_index):
        # [FUNCTION DOC]
        # With a named two-level index the default equals its reference; index off gives
        # the shared no-index reference.
        tuples = [("foo", "car"), ("foo", "bike"), ("bar", "car")]
        df.index = MultiIndex.from_tuples(tuples, names=["idx1", "idx2"])
        expected_with_index = expected_html(datapath, "index_5")
        assert df.to_html() == expected_with_index
        assert df.to_html(index=False) == expected_without_index


@pytest.mark.parametrize("classes", ["sortable draggable", ["sortable", "draggable"]])
def test_to_html_with_classes(classes, datapath):
    # [FUNCTION DOC]
    # Parametrized on a string and a list of CSS classes. An empty frame rendered with
    # either form equals the reference with the classes in the table tag.
    df = DataFrame()
    expected = expected_html(datapath, "with_classes")
    result = df.to_html(classes=classes)
    assert result == expected


def test_to_html_no_index_max_rows(datapath):
    # [FUNCTION DOC]
    # GH 14998. With index off and a maximum of one row the output equals the
    # reference.
    # GH 14998
    df = DataFrame({"A": [1, 2, 3, 4]})
    result = df.to_html(index=False, max_rows=1)
    expected = expected_html(datapath, "gh14998_expected_output")
    assert result == expected


def test_to_html_multiindex_max_cols(datapath):
    # [FUNCTION DOC]
    # GH 6131. A frame with row and column multi-indexes (the latter with a missing
    # level name), limited to two columns, equals the reference.
    # GH 6131
    index = MultiIndex(
        levels=[["ba", "bb", "bc"], ["ca", "cb", "cc"]],
        codes=[[0, 1, 2], [0, 1, 2]],
        names=["b", "c"],
    )
    columns = MultiIndex(
        levels=[["d"], ["aa", "ab", "ac"]],
        codes=[[0, 0, 0], [0, 1, 2]],
        names=[None, "a"],
    )
    data = np.array(
        [[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, 3.0]]
    )
    df = DataFrame(data, index, columns)
    result = df.to_html(max_cols=2)
    expected = expected_html(datapath, "gh6131_expected_output")
    assert result == expected


def test_to_html_multi_indexes_index_false(datapath):
    # [FUNCTION DOC]
    # GH 22579. Multi-index rows and columns with index off equals the reference.
    # GH 22579
    df = DataFrame(
        {"a": range(10), "b": range(10, 20), "c": range(10, 20), "d": range(10, 20)}
    )
    df.columns = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    df.index = MultiIndex.from_product([["a", "b"], ["c", "d", "e", "f", "g"]])
    result = df.to_html(index=False)
    expected = expected_html(datapath, "gh22579_expected_output")
    assert result == expected


@pytest.mark.parametrize("index_names", [True, False])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "column_index, column_type",
    [
        (Index([0, 1]), "unnamed_standard"),
        (Index([0, 1], name="columns.name"), "named_standard"),
        (MultiIndex.from_product([["a"], ["b", "c"]]), "unnamed_multi"),
        (
            MultiIndex.from_product(
                [["a"], ["b", "c"]], names=["columns.name.0", "columns.name.1"]
            ),
            "named_multi",
        ),
    ],
)
@pytest.mark.parametrize(
    "row_index, row_type",
    [
        (Index([0, 1]), "unnamed_standard"),
        (Index([0, 1], name="index.name"), "named_standard"),
        (MultiIndex.from_product([["a"], ["b", "c"]]), "unnamed_multi"),
        (
            MultiIndex.from_product(
                [["a"], ["b", "c"]], names=["index.name.0", "index.name.1"]
            ),
            "named_multi",
        ),
    ],
)
def test_to_html_basic_alignment(
    datapath, row_index, row_type, column_index, column_type, index, header, index_names
):
    # [FUNCTION DOC]
    # GH 22747, GH 22579. Heavily parametrized on row index type, column index type
    # (unnamed or named, flat or two-level), index flag, header flag and index
    # names flag. Renders a two by two zero frame, derives a reference file name from
    # the combination (index off gives none; header off gives none; names off turns
    # named types into unnamed) and compares.
    # GH 22747, GH 22579
    df = DataFrame(np.zeros((2, 2), dtype=int), index=row_index, columns=column_index)
    result = df.to_html(index=index, header=header, index_names=index_names)

    if not index:
        row_type = "none"
    elif not index_names and row_type.startswith("named"):
        row_type = "un" + row_type

    if not header:
        column_type = "none"
    elif not index_names and column_type.startswith("named"):
        column_type = "un" + column_type

    filename = "index_" + row_type + "_columns_" + column_type
    expected = expected_html(datapath, filename)
    assert result == expected


@pytest.mark.parametrize("index_names", [True, False])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "column_index, column_type",
    [
        (Index(np.arange(8)), "unnamed_standard"),
        (Index(np.arange(8), name="columns.name"), "named_standard"),
        (
            MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),
            "unnamed_multi",
        ),
        (
            MultiIndex.from_product(
                [["a", "b"], ["c", "d"], ["e", "f"]], names=["foo", None, "baz"]
            ),
            "named_multi",
        ),
    ],
)
@pytest.mark.parametrize(
    "row_index, row_type",
    [
        (Index(np.arange(8)), "unnamed_standard"),
        (Index(np.arange(8), name="index.name"), "named_standard"),
        (
            MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),
            "unnamed_multi",
        ),
        (
            MultiIndex.from_product(
                [["a", "b"], ["c", "d"], ["e", "f"]], names=["foo", None, "baz"]
            ),
            "named_multi",
        ),
    ],
)
def test_to_html_alignment_with_truncation(
    datapath, row_index, row_type, column_index, column_type, index, header, index_names
):
    # [FUNCTION DOC]
    # GH 22747, GH 22579. Same parametrization as the previous test but with three-level
    # multi-indexes and an eight by eight frame truncated to four rows and four
    # columns; reference file names have a truncation prefix.
    # GH 22747, GH 22579
    df = DataFrame(np.arange(64).reshape(8, 8), index=row_index, columns=column_index)
    result = df.to_html(
        max_rows=4, max_cols=4, index=index, header=header, index_names=index_names
    )

    if not index:
        row_type = "none"
    elif not index_names and row_type.startswith("named"):
        row_type = "un" + row_type

    if not header:
        column_type = "none"
    elif not index_names and column_type.startswith("named"):
        column_type = "un" + column_type

    filename = "trunc_df_index_" + row_type + "_columns_" + column_type
    expected = expected_html(datapath, filename)
    assert result == expected


@pytest.mark.parametrize("index", [False, 0])
def test_to_html_truncation_index_false_max_rows(datapath, index):
    # [FUNCTION DOC]
    # GH 15019. Parametrized on index false or zero. A five-row frame with maximum
    # four rows and no index equals the reference.
    # GH 15019
    data = [
        [1.764052, 0.400157],
        [0.978738, 2.240893],
        [1.867558, -0.977278],
        [0.950088, -0.151357],
        [-0.103219, 0.410599],
    ]
    df = DataFrame(data)
    result = df.to_html(max_rows=4, index=index)
    expected = expected_html(datapath, "gh15019_expected_output")
    assert result == expected


@pytest.mark.parametrize("index", [False, 0])
@pytest.mark.parametrize(
    "col_index_named, expected_output",
    [(False, "gh22783_expected_output"), (True, "gh22783_named_columns_index")],
)
def test_to_html_truncation_index_false_max_cols(
    datapath, index, col_index_named, expected_output
):
    # [FUNCTION DOC]
    # GH 22783. Parametrized on index false or zero and on whether the column index is
    # named. A two by five frame with maximum four columns and no index equals the
    # reference for each case.
    # GH 22783
    data = [
        [1.764052, 0.400157, 0.978738, 2.240893, 1.867558],
        [-0.977278, 0.950088, -0.151357, -0.103219, 0.410599],
    ]
    df = DataFrame(data)
    if col_index_named:
        df.columns.rename("columns.name", inplace=True)
    result = df.to_html(max_cols=4, index=index)
    expected = expected_html(datapath, expected_output)
    assert result == expected


@pytest.mark.parametrize("notebook", [True, False])
def test_to_html_notebook_has_style(notebook):
    # [FUNCTION DOC]
    # Parametrized on the notebook flag. When true, the output contains the
    # embedded style rules (the row header only-of-type selector, middle vertical
    # alignment, head header rule); when false none are present.
    df = DataFrame({"A": [1, 2, 3]})
    result = df.to_html(notebook=notebook)

    if notebook:
        assert "tbody tr th:only-of-type" in result
        assert "vertical-align: middle;" in result
        assert "thead th" in result
    else:
        assert "tbody tr th:only-of-type" not in result
        assert "vertical-align: middle;" not in result
        assert "thead th" not in result


def test_to_html_with_index_names_false():
    # [FUNCTION DOC]
    # GH 16493. With index names disabled, the index name is not in the output.
    # GH 16493
    df = DataFrame({"A": [1, 2]}, index=Index(["a", "b"], name="myindexname"))
    result = df.to_html(index_names=False)
    assert "myindexname" not in result


def test_to_html_with_id():
    # [FUNCTION DOC]
    # GH 8496. The table_id argument adds an id attribute to the table.
    # GH 8496
    df = DataFrame({"A": [1, 2]}, index=Index(["a", "b"], name="myindexname"))
    result = df.to_html(index_names=False, table_id="TEST_ID")
    assert ' id="TEST_ID"' in result


@pytest.mark.parametrize(
    "value,float_format,expected",
    [
        (0.19999, "%.3f", "gh21625_expected_output"),
        (100.0, "%.0f", "gh22270_expected_output"),
    ],
)
def test_to_html_float_format_no_fixed_width(value, float_format, expected, datapath):
    # [FUNCTION DOC]
    # GH 21625, GH 22270. Parametrized on value, float format and reference. A
    # single-cell frame rendered with a percent format equals the reference.
    # GH 21625, GH 22270
    df = DataFrame({"x": [value]})
    expected = expected_html(datapath, expected)
    result = df.to_html(float_format=float_format)
    assert result == expected


@pytest.mark.parametrize(
    "render_links,expected",
    [(True, "render_links_true"), (False, "render_links_false")],
)
def test_to_html_render_links(render_links, expected, datapath):
    # [FUNCTION DOC]
    # GH 2679. Parametrized on render_links. Cell text that is a URL becomes an anchor
    # when true; the output equals the matching reference.
    # GH 2679
    data = [
        [0, "https://pandas.pydata.org/?q1=a&q2=b", "pydata.org"],
        [0, "www.pydata.org", "pydata.org"],
    ]
    df = DataFrame(data, columns=Index(["foo", "bar", None], dtype=object))

    result = df.to_html(render_links=render_links)
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize(
    "method,expected",
    [
        ("to_html", lambda x: lorem_ipsum),
        ("_repr_html_", lambda x: lorem_ipsum[: x - 4] + "..."),  # regression case
    ],
)
@pytest.mark.parametrize("max_colwidth", [10, 20, 50, 100])
def test_ignore_display_max_colwidth(method, expected, max_colwidth):
    # [FUNCTION DOC]
    # See gh-17004. Parametrized on the method (the HTML writer or the notebook repr)
    # and on max_colwidth values. A single long text cell is shown in full by the
    # writer regardless of the display option, while the notebook repr truncates it
    # to the width minus four characters plus an ellipsis; the expected text must be
    # contained in the output.
    # see gh-17004
    df = DataFrame([lorem_ipsum])
    with option_context("display.max_colwidth", max_colwidth):
        result = getattr(df, method)()
    expected = expected(max_colwidth)
    assert expected in result


@pytest.mark.parametrize("classes", [True, 0])
def test_to_html_invalid_classes_type(classes):
    # [FUNCTION DOC]
    # GH 25608. A classes argument that is a boolean or integer raises a type error
    # saying classes must be a string, list or tuple.
    # GH 25608
    df = DataFrame()
    msg = "classes must be a string, list, or tuple"

    with pytest.raises(TypeError, match=msg):
        df.to_html(classes=classes)


def test_to_html_round_column_headers():
    # [FUNCTION DOC]
    # GH 17280. With display precision three, column header floats appear unrounded in
    # plain output but rounded in notebook output.
    # GH 17280
    df = DataFrame([1], columns=[0.55555])
    with option_context("display.precision", 3):
        html = df.to_html(notebook=False)
        notebook = df.to_html(notebook=True)
    assert "0.55555" in html
    assert "0.556" in notebook


@pytest.mark.parametrize("unit", ["100px", "10%", "5em", 150])
def test_to_html_with_col_space_units(unit):
    # [FUNCTION DOC]
    # GH 25941. Parametrized on col_space values with percent, pixel, em units and a
    # bare integer (meaning pixels). Every header cell before the body carries the
    # minimum width style with that unit.
    # GH 25941
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    result = df.to_html(col_space=unit)
    result = result.split("tbody")[0]
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    if isinstance(unit, int):
        unit = str(unit) + "px"
    for h in hdrs:
        expected = f'<th style="min-width: {unit};">'
        assert expected in h


class TestReprHTML:
    # [CLASS DOC]
    # Tests for the notebook HTML representation hook and the display options that
    # control it (min rows, max rows and columns, large frame behaviour, dimensions,
    # mathjax).
    def test_html_repr_min_rows_default(self, datapath):
        # gh-27991

        # [FUNCTION DOC]
        # gh-27991. By default a 20-row frame is not truncated, and a 61-row frame is
        # truncated since the default maximum is 60; both compared to references.
        # default setting no truncation even if above min_rows
        df = DataFrame({"a": range(20)})
        result = df._repr_html_()
        expected = expected_html(datapath, "html_repr_min_rows_default_no_truncation")
        assert result == expected

        # default of max_rows 60 triggers truncation if above
        df = DataFrame({"a": range(61)})
        result = df._repr_html_()
        expected = expected_html(datapath, "html_repr_min_rows_default_truncated")
        assert result == expected

    @pytest.mark.parametrize(
        "max_rows,min_rows,expected",
        [
            # truncated after first two rows
            (10, 4, "html_repr_max_rows_10_min_rows_4"),
            # when set to None, follow value of max_rows
            (12, None, "html_repr_max_rows_12_min_rows_None"),
            # when set value higher as max_rows, use the minimum
            (10, 12, "html_repr_max_rows_10_min_rows_12"),
            # max_rows of None -> never truncate
            (None, 12, "html_repr_max_rows_None_min_rows_12"),
        ],
    )
    def test_html_repr_min_rows(self, datapath, max_rows, min_rows, expected):
        # gh-27991

        # [FUNCTION DOC]
        # gh-27991. Parametrized on max_rows and min_rows combinations including none
        # values. A 61-row frame under those options equals the matching reference
        # (truncated to min rows, to max rows when min is none, min capped by max, and
        # untruncated when max is none).
        df = DataFrame({"a": range(61)})
        expected = expected_html(datapath, expected)
        with option_context("display.max_rows", max_rows, "display.min_rows", min_rows):
            result = df._repr_html_()
        assert result == expected

    def test_repr_html_ipython_config(self, ip):
        # [FUNCTION DOC]
        # Uses the interactive shell fixture. Runs a cell that builds a frame, calls the
        # HTML repr, reads the kernel app config and calls the repr again; the cell must
        # finish without an execution error.
        code = textwrap.dedent(
            """\
        from pandas import DataFrame
        df = DataFrame({"A": [1, 2]})
        df._repr_html_()

        cfg = get_ipython().config
        cfg['IPKernelApp']['parent_appname']
        df._repr_html_()
        """
        )
        result = ip.run_cell(code, silent=True)
        assert not result.error_in_exec

    def test_info_repr_html(self):
        # [FUNCTION DOC]
        # With the large-repr option set to info, a frame that is too long (61 rows) or
        # too wide (21 columns) gives an escaped class listing in the HTML, while with
        # the default option it does not.
        max_rows = 60
        max_cols = 20
        # Long
        h, w = max_rows + 1, max_cols - 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert r"&lt;class" not in df._repr_html_()
        with option_context("display.large_repr", "info"):
            assert r"&lt;class" in df._repr_html_()

        # Wide
        h, w = max_rows - 1, max_cols + 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert "<class" not in df._repr_html_()
        with option_context(
            "display.large_repr", "info", "display.max_columns", max_cols
        ):
            assert "&lt;class" in df._repr_html_()

    def test_fake_qtconsole_repr_html(self, float_frame):
        # [FUNCTION DOC]
        # Renders a float fixture frame as HTML and again with row and column limits
        # that force the fallback; the second output must contain the word class (the
        # summary information fallback).
        df = float_frame

        def get_ipython():
            # [FUNCTION DOC]
            # HELPER: intended fake of the interactive-shell lookup that returns a nested
            # configuration naming a qtconsole parent application. Takes no inputs.
            # KNOWN ISSUE: this fake is defined but never installed or called, so the test does not actually simulate a Qt console.
            return {"config": {"KernelApp": {"parent_appname": "ipython-qtconsole"}}}

        repstr = df._repr_html_()
        assert repstr is not None

        with option_context("display.max_rows", 5, "display.max_columns", 2):
            repstr = df._repr_html_()

        assert "class" in repstr  # info fallback

    def test_repr_html(self, float_frame):
        # [FUNCTION DOC]
        # Smoke checks: the repr works for a float frame under small limits and with the
        # notebook repr option off; for a two by two frame it includes a row count text
        # when show_dimensions is true and omits it when false.
        df = float_frame
        df._repr_html_()

        with option_context("display.max_rows", 1, "display.max_columns", 1):
            df._repr_html_()

        with option_context("display.notebook_repr_html", False):
            df._repr_html_()

        df = DataFrame([[1, 2], [3, 4]])
        with option_context("display.show_dimensions", True):
            assert "2 rows" in df._repr_html_()
        with option_context("display.show_dimensions", False):
            assert "2 rows" not in df._repr_html_()

    def test_repr_html_mathjax(self):
        # [FUNCTION DOC]
        # By default the output has no tex2jax or mathjax ignore classes; with the
        # use_mathjax option set to false both are present.
        df = DataFrame([[1, 2], [3, 4]])
        assert "tex2jax_ignore" not in df._repr_html_()
        assert "mathjax_ignore" not in df._repr_html_()

        with option_context("display.html.use_mathjax", False):
            assert "tex2jax_ignore" in df._repr_html_()
            assert "mathjax_ignore" in df._repr_html_()

    def test_repr_html_wide(self):
        # [FUNCTION DOC]
        # A frame with 19 columns is not truncated (no ellipsis) under a 20-column limit;
        # a 21-column frame is truncated and shows an ellipsis.
        max_cols = 20
        df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." not in df._repr_html_()

        wide_df = DataFrame([["a" * 25] * (max_cols + 1)] * 10)
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." in wide_df._repr_html_()

    def test_repr_html_wide_multiindex_cols(self):
        # [FUNCTION DOC]
        # A frame with 20 column labels from a two-level multi-index is not truncated by
        # default; with 22 labels and a 20 column limit it is truncated.
        max_cols = 20

        mcols = MultiIndex.from_product(
            [np.arange(max_cols // 2), ["foo", "bar"]], names=["first", "second"]
        )
        df = DataFrame([["a" * 25] * len(mcols)] * 10, columns=mcols)
        reg_repr = df._repr_html_()
        assert "..." not in reg_repr

        mcols = MultiIndex.from_product(
            (np.arange(1 + (max_cols // 2)), ["foo", "bar"]), names=["first", "second"]
        )
        df = DataFrame([["a" * 25] * len(mcols)] * 10, columns=mcols)
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." in df._repr_html_()

    def test_repr_html_long(self):
        # [FUNCTION DOC]
        # Under a 60-row limit, a 59-row frame is not truncated and shows the middle
        # value; a 61-row frame is truncated, omits the middle value and reports row and
        # column counts in a summary.
        with option_context("display.max_rows", 60):
            max_rows = get_option("display.max_rows")
            h = max_rows - 1
            df = DataFrame({"A": np.arange(1, 1 + h), "B": np.arange(41, 41 + h)})
            reg_repr = df._repr_html_()
            assert ".." not in reg_repr
            assert str(41 + max_rows // 2) in reg_repr

            h = max_rows + 1
            df = DataFrame({"A": np.arange(1, 1 + h), "B": np.arange(41, 41 + h)})
            long_repr = df._repr_html_()
            assert ".." in long_repr
            assert str(41 + max_rows // 2) not in long_repr
            assert f"{h} rows " in long_repr
            assert "2 columns" in long_repr

    def test_repr_html_float(self):
        # [FUNCTION DOC]
        # Same long-frame behaviour for a frame with a float index: not truncated at 59
        # rows (contains the last value as a cell), truncated at 61 rows (summary shows
        # rows and columns, middle value absent).
        with option_context("display.max_rows", 60):
            max_rows = get_option("display.max_rows")
            h = max_rows - 1
            df = DataFrame(
                {
                    "idx": np.linspace(-10, 10, h),
                    "A": np.arange(1, 1 + h),
                    "B": np.arange(41, 41 + h),
                }
            ).set_index("idx")
            reg_repr = df._repr_html_()
            assert ".." not in reg_repr
            assert f"<td>{40 + h}</td>" in reg_repr

            h = max_rows + 1
            df = DataFrame(
                {
                    "idx": np.linspace(-10, 10, h),
                    "A": np.arange(1, 1 + h),
                    "B": np.arange(41, 41 + h),
                }
            ).set_index("idx")
            long_repr = df._repr_html_()
            assert ".." in long_repr
            assert "<td>31</td>" not in long_repr
            assert f"{h} rows " in long_repr
            assert "2 columns" in long_repr

    def test_repr_html_long_multiindex(self):
        # [FUNCTION DOC]
        # A frame with 60 rows from a two-level index is not truncated under 60 rows and
        # 20 columns; with 62 rows (default options at that moment) the output contains
        # an ellipsis.
        max_rows = 60
        max_L1 = max_rows // 2

        tuples = list(itertools.product(np.arange(max_L1), ["foo", "bar"]))
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((max_L1 * 2, 2)),
            index=idx,
            columns=["A", "B"],
        )
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            reg_repr = df._repr_html_()
        assert "..." not in reg_repr

        tuples = list(itertools.product(np.arange(max_L1 + 1), ["foo", "bar"]))
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal(((max_L1 + 1) * 2, 2)),
            index=idx,
            columns=["A", "B"],
        )
        long_repr = df._repr_html_()
        assert "..." in long_repr

    def test_repr_html_long_and_wide(self):
        # [FUNCTION DOC]
        # A 59 by 19 frame is not truncated under 60 rows and 20 columns; a 61 by 21
        # frame is truncated with an ellipsis.
        max_cols = 20
        max_rows = 60

        h, w = max_rows - 1, max_cols - 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." not in df._repr_html_()

        h, w = max_rows + 1, max_cols + 1
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            assert "..." in df._repr_html_()


def test_to_html_multilevel(multiindex_year_month_day_dataframe_random_data):
    # [FUNCTION DOC]
    # Uses a year-month-day multi-index fixture frame: after naming its columns,
    # rendering it and its transpose must not fail.
    ymd = multiindex_year_month_day_dataframe_random_data

    ymd.columns.name = "foo"
    ymd.to_html()
    ymd.T.to_html()


@pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
def test_to_html_na_rep_and_float_format(na_rep, datapath):
    # [FUNCTION DOC]
    # GH 13828. Parametrized on the missing representation. Combining na_rep and a
    # two-decimal string float format equals the reference with the representation
    # substituted.
    # https://github.com/pandas-dev/pandas/issues/13828
    df = DataFrame(
        [
            ["A", 1.2225],
            ["A", None],
        ],
        columns=["Group", "Data"],
    )
    result = df.to_html(na_rep=na_rep, float_format="{:.2f}".format)
    expected = expected_html(datapath, "gh13828_expected_output")
    expected = expected.format(na_rep=na_rep)
    assert result == expected


def test_to_html_na_rep_non_scalar_data(datapath):
    # [FUNCTION DOC]
    # GH47103. A cell containing a list is displayed as the list text and not
    # treated as missing; compared with the reference.
    # GH47103
    df = DataFrame([{"a": 1, "b": [1, 2, 3]}])
    result = df.to_html(na_rep="-")
    expected = expected_html(datapath, "gh47103_expected_output")
    assert result == expected


def test_to_html_float_format_object_col(datapath):
    # [FUNCTION DOC]
    # GH#40024. A float format callable is applied to float entries of an object
    # column mixing a float with a string; compared with the reference.
    # GH#40024
    df = DataFrame(data={"x": [1000.0, "test"]})
    result = df.to_html(float_format=lambda x: f"{x:,.0f}")
    expected = expected_html(datapath, "gh40024_expected_output")
    assert result == expected


def test_to_html_multiindex_col_with_colspace():
    # [FUNCTION DOC]
    # GH#53885. Two-level integer columns with col_space of 100: every header cell
    # including blank ones has the minimum width style; exact expected string.
    # GH#53885
    df = DataFrame([[1, 2]])
    df.columns = MultiIndex.from_tuples([(1, 1), (2, 1)])
    result = df.to_html(col_space=100)
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        "    <tr>\n"
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">2</th>\n'
        "    </tr>\n"
        "    <tr>\n"
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>"
    )
    assert result == expected


def test_to_html_tuple_col_with_colspace():
    # [FUNCTION DOC]
    # GH#53885. A frame with a tuple label column and a plain label column with
    # col_space of 100: tuple label rendered as text; exact expected string.
    # GH#53885
    df = DataFrame({("a", "b"): [1], "b": [2]})
    result = df.to_html(col_space=100)
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">(a, b)</th>\n'
        '      <th style="min-width: 100px;">b</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>"
    )
    assert result == expected


def test_to_html_empty_complex_array():
    # [FUNCTION DOC]
    # GH#54167. A frame with an empty complex column and col_space of 100 renders
    # header cells with the width style and an empty body; exact expected string.
    # GH#54167
    df = DataFrame({"x": np.array([], dtype="complex")})
    result = df.to_html(col_space=100)
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">x</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "  </tbody>\n"
        "</table>"
    )
    assert result == expected
