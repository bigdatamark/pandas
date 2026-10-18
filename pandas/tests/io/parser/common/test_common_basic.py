"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests (run against every engine through the parser fixture) of basic reading
# behavior: local file URLs, thousands separators, unnamed columns, row limits, trailing
# delimiters and whitespace handling, escape characters, blank and whitespace-only
# lines, regex whitespace separators, special characters in file names, equivalence of
# the csv and table readers, byte-order marks, invalid argument combinations, and seek
# positions.
# The module defines two marker aliases: one marks a test as expected to fail for the
# pyarrow engine and one skips it for that engine. Many tests branch on the pyarrow
# engine to check that unsupported options raise a value error with the engine-specific
# message.

from datetime import datetime
from inspect import signature
from io import StringIO
import os
import sys

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import HAS_PYARROW
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas import (
    DataFrame,
    Index,
    compat,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_read_csv_local(all_parsers, csv1):
    # [FUNCTION DOC]
    # Reads a sample CSV through a file URL (with the platform-specific prefix), using
    # the first column as index and date parsing, and verifies the seven-row frame of
    # floats with a date index named index. The expected index has microsecond datetime
    # unit, converted to second unit for pyarrow.
    prefix = "file:///" if compat.is_platform_windows() else "file://"
    parser = all_parsers

    fname = prefix + str(os.path.abspath(csv1))
    result = parser.read_csv(fname, index_col=0, parse_dates=True)

    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            dtype="M8[us]",
            name="index",
        ),
    )
    if parser.engine == "pyarrow":
        expected.index = expected.index.astype("M8[s]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "number_csv, expected_number",
    [
        ("2,334", 2334),
        ("-2,334", -2334),
        ("-2,334,", -2334),
        # Multiple consecutive thousand separators are allowed in C engine,
        # but it's not necessarily intended behavior and may change in the future.
        ("2,,,,,,,,,,,,,,,5", 25),
        ("2,,3,4,,,,,,,,,,,,5", 2345),
    ],
)
def test_1000_sep(all_parsers, number_csv, expected_number, request):
    # [FUNCTION DOC]
    # Verifies the thousands separator option on pipe-separated data with values such as
    # two thousand three hundred thirty-four (with a negative sign or trailing
    # separator) and with runs of consecutive separators, which the C engine accepts and
    # joins into one number; the expected integer is parametrized.
    # The python engine is marked as expected to fail for the consecutive-separator
    # cases, and pyarrow is expected to raise a value error saying the thousands option
    # is unsupported.
    parser = all_parsers
    data = f"""A|B|C
1|{number_csv}|5
10|13|10.
"""
    expected = DataFrame({"A": [1, 10], "B": [expected_number, 13], "C": [5, 10.0]})

    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="|", thousands=",")
        return
    elif parser.engine == "python" and ",," in number_csv:
        mark = pytest.mark.xfail(
            reason="Python engine doesn't allow consecutive thousands separators"
        )
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), sep="|", thousands=",")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("value", ["1 ,", ", 1", ",1"])
def test_1000_sep_not_stripped_after_whitespace(all_parsers, value):
    # [FUNCTION DOC]
    # Verifies that a value consisting of a thousands separator next to whitespace or at
    # the start (such as a space before a comma) is not converted to a number and
    # remains a string. For pyarrow, verifies the unsupported thousands error.
    parser = all_parsers
    data = f"a\n{value}\n"
    expected = DataFrame({"a": [value]})

    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=";", thousands=",")
        return

    result = parser.read_csv(StringIO(data), sep=";", thousands=",")
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
def test_unnamed_columns(all_parsers):
    # [FUNCTION DOC]
    # Marked as expected to fail for pyarrow (non-unique column index error). Verifies
    # that empty header fields get the placeholder names Unnamed with their column
    # position, and values stay int64.
    data = """A,B,C,,
1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    parser = all_parsers
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=np.int64,
        columns=["A", "B", "C", "Unnamed: 3", "Unnamed: 4"],
    )
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_csv_mixed_type(all_parsers):
    # [FUNCTION DOC]
    # Verifies a string column and two integer columns are read with the right inferred
    # types.
    data = """A,B,C
a,1,2
b,3,4
c,4,5
"""
    parser = all_parsers
    expected = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]})
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_read_csv_low_memory_no_rows_with_index(all_parsers):
    # [FUNCTION DOC]
    # Skipped unless the parser fixture has low memory mode on. Verifies that reading
    # with the first column as index and a row limit of zero returns an empty frame with
    # the three header columns (regression for an issue with low memory reading), even
    # though data rows are wider than the header. For pyarrow, verifies the unsupported
    # row-limit error.
    # see gh-21141
    parser = all_parsers

    if not parser.low_memory:
        pytest.skip("This is a low-memory specific test")

    data = """A,B,C
1,1,1,2
2,2,3,4
3,3,4,5
"""

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
        return

    result = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    expected = DataFrame(columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


def test_read_csv_dataframe(all_parsers, csv1):
    # [FUNCTION DOC]
    # Reads the sample CSV file directly (index in first column, dates parsed) and
    # verifies the same expected seven-row frame as the file URL test, including the
    # pyarrow second-unit index adjustment.
    parser = all_parsers
    result = parser.read_csv(csv1, index_col=0, parse_dates=True)
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            dtype="M8[us]",
            name="index",
        ),
    )
    if parser.engine == "pyarrow":
        expected.index = expected.index.astype("M8[s]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [3, 3.0])
def test_read_nrows(all_parsers, nrows):
    # [FUNCTION DOC]
    # Verifies a row limit given as an integer or a whole-number float returns only the
    # first three data rows. For pyarrow, verifies the unsupported row-limit error.
    # see gh-10476
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    expected = DataFrame(
        [["foo", 2, 3, 4, 5], ["bar", 7, 8, 9, 10], ["baz", 12, 13, 14, 15]],
        columns=["index", "A", "B", "C", "D"],
    )
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), nrows=nrows)
        return

    result = parser.read_csv(StringIO(data), nrows=nrows)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1.2, "foo", -1])
def test_read_nrows_bad(all_parsers, nrows):
    # [FUNCTION DOC]
    # Verifies invalid row limits (non-whole float, string, negative) raise a value
    # error saying nrows must be an integer of at least zero; pyarrow expects its
    # unsupported-option message.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    msg = r"'nrows' must be an integer >=0"
    parser = all_parsers
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), nrows=nrows)


def test_nrows_skipfooter_errors(all_parsers):
    # [FUNCTION DOC]
    # Verifies passing both a footer-skip count and a row limit raises a value error
    # saying skipfooter is not supported with nrows.
    msg = "'skipfooter' not supported with 'nrows'"
    data = "a\n1\n2\n3\n4\n5\n6"
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)


@skip_pyarrow
def test_missing_trailing_delimiters(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies rows with fewer fields than the header (missing
    # trailing delimiters) are padded with missing values.
    parser = all_parsers
    data = """A,B,C,D
1,2,3,4
1,3,3,
1,4,5"""

    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]],
        columns=["A", "B", "C", "D"],
    )
    tm.assert_frame_equal(result, expected)


def test_skip_initial_space(all_parsers):
    # [FUNCTION DOC]
    # Verifies a single long comma-separated row with spaces after delimiters and quotes
    # is read with no header, 33 column names, the skip-initial-space option and a
    # custom missing-value marker, producing numeric columns, string columns and NaN for
    # the marker values; numbers with leading zeros become integers. For pyarrow,
    # verifies the unsupported skip-initial-space error.
    data = (
        '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, '
        "1.00361,  1.12551, 330.65659, 0355626618.16711,  73.48821, "
        "314.11625,  1917.09447,   179.71425,  80.000, 240.000, -350,  "
        "70.06056, 344.98370, 1,   1, -0.689265, -0.692787,  "
        "0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   "
        "-9999.0,   -9999.0,   -9999.0,  -9999.0, 000, 012, 128"
    )
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=list(range(33)),
                header=None,
                na_values=["-9999.0"],
                skipinitialspace=True,
            )
        return

    result = parser.read_csv(
        StringIO(data),
        names=list(range(33)),
        header=None,
        na_values=["-9999.0"],
        skipinitialspace=True,
    )
    expected = DataFrame(
        [
            [
                "09-Apr-2012",
                "01:10:18.300",
                2456026.548822908,
                12849,
                1.00361,
                1.12551,
                330.65659,
                355626618.16711,
                73.48821,
                314.11625,
                1917.09447,
                179.71425,
                80.0,
                240.0,
                -350,
                70.06056,
                344.9837,
                1,
                1,
                -0.689265,
                -0.692787,
                0.212036,
                14.7674,
                41.605,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                12,
                128,
            ]
        ]
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_trailing_delimiters(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that data rows with one extra trailing delimiter are
    # read correctly when the index column option is turned off, giving the three named
    # columns.
    # see gh-2442
    data = """A,B,C
1,2,3,
4,5,6,
7,8,9,"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=False)

    expected = DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


def test_escapechar(all_parsers):
    # [FUNCTION DOC]
    # Verifies that a backslash escape character with a double-quote quote character
    # lets a quoted field contain escaped quotes and commas, and that the column names
    # are preserved. Uses long URLs and a non-ASCII word, read with UTF-8 encoding.
    # https://stackoverflow.com/questions/13824840/feature-request-for-
    # pandas-read-csv
    data = '''SEARCH_TERM,ACTUAL_URL
"bra tv board","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"tv p\xc3\xa5 hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'''

    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), escapechar="\\", quotechar='"', encoding="utf-8"
    )

    assert result["SEARCH_TERM"][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series'

    tm.assert_index_equal(result.columns, Index(["SEARCH_TERM", "ACTUAL_URL"]))


def test_ignore_leading_whitespace(all_parsers):
    # [FUNCTION DOC]
    # Verifies that a regex whitespace separator ignores leading whitespace on lines,
    # giving three integer columns. For pyarrow, verifies the value error that regex
    # separators are unsupported.
    # see gh-3374, gh-6607
    parser = all_parsers
    data = " a b c\n 1 2 3\n 4 5 6\n 7 8 9"

    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return
    result = parser.read_csv(StringIO(data), sep=r"\s+")

    expected = DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("usecols", [None, [0, 1], ["a", "b"]])
def test_uneven_lines_with_usecols(all_parsers, usecols):
    # [FUNCTION DOC]
    # Skipped for pyarrow. With a ragged row that has too many fields, verifies that
    # without a column selection a parser error about the expected field count is
    # raised, while selecting the first two columns (by position or name) works and
    # returns the selected columns for all rows.
    # see gh-12203
    parser = all_parsers
    data = r"""a,b,c
0,1,2
3,4,5,6,7
8,9,10"""

    if usecols is None:
        # Make sure that an error is still raised
        # when the "usecols" parameter is not provided.
        msg = r"Expected \d+ fields in line \d+, saw \d+"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        expected = DataFrame({"a": [0, 3, 8], "b": [1, 4, 9]})

        result = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # First, check to see that the response of parser when faced with no
        # provided columns raises the correct error, with or without usecols.
        ("", {}, None),
        ("", {"usecols": ["X"]}, None),
        (
            ",,",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"], index=[0], dtype=np.float64),
        ),
        (
            "",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"]),
        ),
    ],
)
def test_read_empty_with_usecols(all_parsers, data, kwargs, expected):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies empty input raises an empty-data error ("No columns
    # to parse from file"), with or without column selection, and that input with only
    # delimiters, given names and a selection, gives a one-row float frame, while empty
    # input with names and a selection gives an empty frame with the selected column.
    # see gh-12493
    parser = all_parsers

    if expected is None:
        msg = "No columns to parse from file"
        with pytest.raises(EmptyDataError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,expected_data",
    [
        # gh-8661, gh-8679: this should ignore six lines, including
        # lines with trailing whitespace and blank lines.
        (
            {
                "header": None,
                "sep": r"\s+",
                "skiprows": [0, 1, 2, 3, 5, 6],
                "skip_blank_lines": True,
            },
            [[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]],
        ),
        # gh-8983: test skipping set of rows after a row with trailing spaces.
        (
            {
                "sep": r"\s+",
                "skiprows": [1, 2, 3, 5, 6],
                "skip_blank_lines": True,
            },
            {"A": [1.0, 5.1], "B": [2.0, np.nan], "C": [4.0, 10]},
        ),
    ],
)
def test_trailing_spaces(all_parsers, kwargs, expected_data):
    # [FUNCTION DOC]
    # Verifies that, with whitespace as separator, unwanted lines (including ones with
    # trailing spaces or tabs and blank lines) can be skipped by row positions, with and
    # without header, with blank-line skipping on. Input commas are replaced by double
    # spaces before parsing. For pyarrow, verifies an unsupported-feature value error.
    data = "A B C  \nrandom line with trailing spaces    \nskip\n1,2,3\n1,2.,4.\nrandom line with trailing tabs\t\t\t\n   \n5.1,NaN,10.0\n"  # noqa: E501
    parser = all_parsers

    if parser.engine == "pyarrow":
        with pytest.raises(ValueError, match="the 'pyarrow' engine does not support"):
            parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
        return
    expected = DataFrame(expected_data)
    result = parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
    tm.assert_frame_equal(result, expected)


def test_read_filepath_or_buffer(all_parsers):
    # [FUNCTION DOC]
    # Verifies that passing a bytes object as the path-or-buffer keyword raises a type
    # error saying a file path or file-like object was expected.
    # see gh-43366
    parser = all_parsers

    with pytest.raises(TypeError, match="Expected file path name or file-like"):
        parser.read_csv(filepath_or_buffer=b"input")


def test_single_char_leading_whitespace(all_parsers):
    # [FUNCTION DOC]
    # Verifies single-column data with the skip-initial-space option and whitespace
    # separator is read as one string column of the four values. For pyarrow, verifies
    # the unsupported skip-initial-space error.
    # see gh-9710
    parser = all_parsers
    data = """\
MyColumn
a
b
a
b\n"""

    if parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                skipinitialspace=True,
            )
        return
    expected = DataFrame({"MyColumn": list("abab")})
    result = parser.read_csv(StringIO(data), skipinitialspace=True, sep=r"\s+")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sep,skip_blank_lines,exp_data",
    [
        (",", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (r"\s+", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (
            ",",
            False,
            [
                [1.0, 2.0, 4.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [5.0, np.nan, 10.0],
                [np.nan, np.nan, np.nan],
                [-70.0, 0.4, 1.0],
            ],
        ),
    ],
)
def test_empty_lines(all_parsers, sep, skip_blank_lines, exp_data, request):
    # [FUNCTION DOC]
    # Verifies empty-line handling with comma or whitespace separators and with
    # blank-line skipping on or off: when on, blank lines are dropped; when off, each
    # blank line yields a row of NaN. For pyarrow with whitespace separators, verifies
    # the regex separator error.
    parser = all_parsers
    data = """\
A,B,C
1,2.,4.


5.,NaN,10.0

-70,.4,1
"""

    if sep == r"\s+":
        data = data.replace(",", "  ")

        if parser.engine == "pyarrow":
            msg = "the 'pyarrow' engine does not support regex separators"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(
                    StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines
                )
            return

    result = parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
    expected = DataFrame(exp_data, columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_whitespace_lines(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that leading lines consisting only of whitespace are
    # ignored and the header is the first real line, and that a data row with leading
    # whitespace is parsed correctly.
    parser = all_parsers
    data = """

\t  \t\t
\t
A,B,C
\t    1,2.,4.
5.,NaN,10.0
"""
    expected = DataFrame([[1, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            """   A   B   C   D
a   1   2   3   4
b   1   2   3   4
c   1   2   3   4
""",
            DataFrame(
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                columns=["A", "B", "C", "D"],
                index=["a", "b", "c"],
            ),
        ),
        (
            "    a b c\n1 2 3 \n4 5  6\n 7 8 9",
            DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]),
        ),
    ],
)
def test_whitespace_regex_separator(all_parsers, data, expected):
    # [FUNCTION DOC]
    # Verifies that data with irregular runs of spaces, parsed with a whitespace regex
    # separator, gives the expected frames: one where the first column becomes the index
    # because the header has one less field, and one with leading spaces on every line.
    # For pyarrow, verifies the regex separator error.
    # see gh-6607
    parser = all_parsers
    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return

    result = parser.read_csv(StringIO(data), sep=r"\s+")
    tm.assert_frame_equal(result, expected)


def test_sub_character(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Verifies a file containing a substitute control character in a column name is read
    # with that character retained in the name.
    # see gh-16893
    filename = os.path.join(csv_dir_path, "sub_char.csv")
    expected = DataFrame([[1, 2, 3]], columns=["a", "\x1ab", "c"])

    parser = all_parsers
    result = parser.read_csv(filename)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("filename", ["sé-es-vé.csv", "ru-sй.csv", "中文文件名.csv"])
def test_filename_with_special_chars(all_parsers, filename, tmp_path):
    # [FUNCTION DOC]
    # Verifies a frame written to a path with accented, Cyrillic or Chinese characters
    # in its name can be read back equal.
    # see gh-15086.
    parser = all_parsers
    df = DataFrame({"a": [1, 2, 3]})

    path = tmp_path / filename
    df.to_csv(path, index=False)

    result = parser.read_csv(path)
    tm.assert_frame_equal(result, df)


def test_read_table_same_signature_as_read_csv(all_parsers):
    # [FUNCTION DOC]
    # Verifies the signatures of the table reader and the csv reader are identical: same
    # parameter names and return annotation, and every parameter equal except the
    # separator, which defaults to a tab for the table reader and a comma for csv, with
    # identical annotation and kind.
    # GH-34976
    parser = all_parsers

    table_sign = signature(parser.read_table)
    csv_sign = signature(parser.read_csv)

    assert table_sign.parameters.keys() == csv_sign.parameters.keys()
    assert table_sign.return_annotation == csv_sign.return_annotation

    for key, csv_param in csv_sign.parameters.items():
        table_param = table_sign.parameters[key]
        if key == "sep":
            assert csv_param.default == ","
            assert table_param.default == "\t"
            assert table_param.annotation == csv_param.annotation
            assert table_param.kind == csv_param.kind
            continue

        assert table_param == csv_param


def test_read_table_equivalency_to_read_csv(all_parsers):
    # [FUNCTION DOC]
    # Verifies the table reader on tab-separated text gives the same frame as the csv
    # reader with an explicit tab separator.
    # see gh-21948
    # As of 0.25.0, read_table is undeprecated
    parser = all_parsers
    data = "a\tb\n1\t2\n3\t4"
    expected = parser.read_csv(StringIO(data), sep="\t")
    result = parser.read_table(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("read_func", ["read_csv", "read_table"])
def test_read_csv_and_table_sys_setprofile(all_parsers, read_func):
    # [FUNCTION DOC]
    # Verifies that the csv and table readers (parametrized by reader name) work while a
    # profiling function is installed in the interpreter, returning a single column
    # named a b with the unsplit value (regression for a profiler crash).
    # Side effect: installs a no-op profile function and then clears it afterward (not
    # in a protected cleanup, so a failure in between leaves it installed).
    # GH#41069
    parser = all_parsers
    data = "a b\n0 1"

    sys.setprofile(lambda *a, **k: None)
    result = getattr(parser, read_func)(StringIO(data))
    sys.setprofile(None)

    expected = DataFrame({"a b": ["0 1"]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_first_row_bom(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that a byte-order mark before a quoted header in the
    # first row is stripped, giving three empty columns named Head1 to Head3 when
    # reading tab-separated text.
    # see gh-26545
    parser = all_parsers
    data = '''\ufeff"Head1"\t"Head2"\t"Head3"'''

    result = parser.read_csv(StringIO(data), delimiter="\t")
    expected = DataFrame(columns=["Head1", "Head2", "Head3"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_first_row_bom_unquoted(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Same as the previous test but with an unquoted header.
    # see gh-36343
    parser = all_parsers
    data = """\ufeffHead1\tHead2\tHead3"""

    result = parser.read_csv(StringIO(data), delimiter="\t")
    expected = DataFrame(columns=["Head1", "Head2", "Head3"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers, nrows):
    # [FUNCTION DOC]
    # Verifies that with header at row three, blank-line skipping off and a row limit
    # from one to five, the result is the matching leading rows of a reference frame
    # that includes NaN rows for the blank lines. For pyarrow, verifies the unsupported
    # row-limit error.
    # GH 28071
    ref = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]],
        columns=list("ab"),
    )
    csv = "\nheader\n\na,b\n\n\n1,2\n\n3,4"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False
            )
        return

    df = parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
    tm.assert_frame_equal(df, ref[:nrows])


@skip_pyarrow
def test_no_header_two_extra_columns(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that with no header, three names, the index column
    # off, and a row with five fields, a parser warning about the header or names length
    # not matching the data is emitted and the extra fields are dropped.
    # GH 26218
    column_names = ["one", "two", "three"]
    ref = DataFrame([["foo", "bar", "baz"]], columns=column_names)
    stream = StringIO("foo,bar,baz,bam,blah")
    parser = all_parsers
    df = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header or names does not match length of data. "
        "This leads to a loss of data with index_col=False.",
        stream,
        header=None,
        names=column_names,
        index_col=False,
    )
    tm.assert_frame_equal(df, ref)


def test_read_csv_names_not_accepting_sets(all_parsers):
    # [FUNCTION DOC]
    # Verifies that passing a set as column names raises a value error saying names
    # should be an ordered collection.
    # GH 34946
    data = """\
    1,2,3
    4,5,6\n"""
    parser = all_parsers
    with pytest.raises(ValueError, match="Names should be an ordered collection."):
        parser.read_csv(StringIO(data), names=set("QAZ"))


def test_read_csv_delimiter_and_sep_no_default(all_parsers):
    # [FUNCTION DOC]
    # Verifies specifying both separator and delimiter with different values raises a
    # value error that only one may be specified.
    # GH#39823
    f = StringIO("a,b\n1,2")
    parser = all_parsers
    msg = "Specified a sep and a delimiter; you can only specify one."
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, sep=" ", delimiter=".")


@pytest.mark.parametrize("kwargs", [{"delimiter": "\n"}, {"sep": "\n"}])
def test_read_csv_line_break_as_separator(kwargs, all_parsers):
    # [FUNCTION DOC]
    # Verifies using a newline as the delimiter or as the separator raises a value error
    # explaining this forces the python engine which does not accept a line terminator.
    # GH#43528
    parser = all_parsers
    data = """a,b,c
1,2,3
    """
    msg = (
        r"Specified \\n as separator or delimiter. This forces the python engine "
        r"which does not accept a line terminator. Hence it is not allowed to use "
        r"the line terminator as separator."
    )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


@skip_pyarrow
def test_dict_keys_as_names(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies a dictionary keys view is accepted as names,
    # creating columns a and b.
    # GH: 36928
    data = "1,2"

    keys = {"a": int, "b": int}.keys()
    parser = all_parsers

    result = parser.read_csv(StringIO(data), names=keys)
    expected = DataFrame({"a": [1], "b": [2]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(using_string_dtype() and HAS_PYARROW, reason="TODO(infer_string)")
@xfail_pyarrow  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 0
def test_encoding_surrogatepass(all_parsers, tmp_path):
    # [FUNCTION DOC]
    # Marked expected to fail when string dtype inference with pyarrow is enabled, and
    # expected to fail for the pyarrow engine (unicode decode error). Writes bytes that
    # are an encoded surrogate to a temporary file and verifies reading with the
    # surrogate-pass error handler yields a frame whose index, column and value are the
    # decoded surrogate string, while reading without the handler raises a unicode
    # decode error.
    # GH39017
    parser = all_parsers
    content = b"\xed\xbd\xbf"
    decoded = content.decode("utf-8", errors="surrogatepass")
    expected = DataFrame({decoded: [decoded]}, index=[decoded * 2])
    expected.index.name = decoded * 2

    path = tmp_path / "test_encoding.csv"
    path.write_bytes(
        content * 2 + b"," + content + b"\n" + content * 2 + b"," + content
    )
    df = parser.read_csv(path, encoding_errors="surrogatepass", index_col=0)
    tm.assert_frame_equal(df, expected)
    with pytest.raises(UnicodeDecodeError, match="'utf-8' codec can't decode byte"):
        parser.read_csv(path)


def test_malformed_second_line(all_parsers):
    # [FUNCTION DOC]
    # Verifies that with blank-line skipping off and header at row one, input with a
    # blank first line reads the third line as the single data value under column a.
    # see GH14782
    parser = all_parsers
    data = "\na\nb\n"
    result = parser.read_csv(StringIO(data), skip_blank_lines=False, header=1)
    expected = DataFrame({"a": ["b"]})
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_short_single_line(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies a single short line read with no header and three
    # names pads the third column with NaN.
    # GH 47566
    parser = all_parsers
    columns = ["a", "b", "c"]
    data = "1,2"
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    expected = DataFrame({"a": [1], "b": [2], "c": [np.nan]})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Length mismatch: Expected axis has 2 elements
def test_short_multi_line(all_parsers):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (length mismatch). Verifies two short lines
    # read with no header and three names pad the third column with NaN.
    # GH 47566
    parser = all_parsers
    columns = ["a", "b", "c"]
    data = "1,2\n1,2"
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    expected = DataFrame({"a": [1, 1], "b": [2, 2], "c": [np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_read_seek(all_parsers, tmp_path):
    # [FUNCTION DOC]
    # Verifies that reading from an already-opened file after consuming its first line
    # starts from the current position (not the beginning), giving the same frame as
    # reading the content string without the prefix line.
    # GH48646
    parser = all_parsers
    prefix = "### DATA\n"
    content = "nkey,value\ntables,rectangular\n"
    path = tmp_path / "test_seek.csv"
    path.write_text(prefix + content, encoding="utf-8")
    with open(path, encoding="utf-8") as file:
        file.readline()
        actual = parser.read_csv(file)
    expected = parser.read_csv(StringIO(content))
    tm.assert_frame_equal(actual, expected)
