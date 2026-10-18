"""
Tests that apply specifically to the Python parser. Unless specifically
stated as a Python-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the C parser can accept further
arguments when parsing.
"""
# [MODULE DOC]
# Tests specific to the python parser engine, run through the python_parser_only fixture. Upstream intent: move tests out of here as the C engine gains support for the same arguments.
# Topics: separator sniffing, footer skipping validation, multi-character and regex separators with encodings and compression, on_bad_lines callables (return values, side effects, chunked iteration, error propagation, index_col handling), thousands/decimal handling with dtypes, and quote handling combined with skiprows.

from __future__ import annotations

import csv
from io import (
    BytesIO,
    StringIO,
    TextIOWrapper,
)
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas.errors import (
    ParserError,
    ParserWarning,
)

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_default_separator(python_parser_only):
    # [FUNCTION DOC]
    # See gh-17333. With no separator given, the sniffing logic treats the letter o as the separator in the sample data; asserts the two-column frame parsed accordingly.
    # see gh-17333
    #
    # csv.Sniffer in Python treats "o" as separator.
    data = "aob\n1o2\n3o4"
    parser = python_parser_only
    expected = DataFrame({"a": [1, 3], "b": [2, 4]})

    result = parser.read_csv(StringIO(data), sep=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipfooter", ["foo", 1.5, True])
def test_invalid_skipfooter_non_int(python_parser_only, skipfooter):
    # [FUNCTION DOC]
    # See gh-15925. Parametrized over a string, a float and a boolean for skipfooter; asserts a value error saying skipfooter must be an integer.
    # see gh-15925 (comment)
    data = "a\n1\n2"
    parser = python_parser_only
    msg = "skipfooter must be an integer"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=skipfooter)


def test_invalid_skipfooter_negative(python_parser_only):
    # [FUNCTION DOC]
    # See gh-15925. A negative skipfooter raises a value error that it cannot be negative.
    # see gh-15925 (comment)
    data = "a\n1\n2"
    parser = python_parser_only
    msg = "skipfooter cannot be negative"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=-1)


@pytest.mark.parametrize("kwargs", [{"sep": None}, {"delimiter": "|"}])
def test_sniff_delimiter(python_parser_only, kwargs):
    # [FUNCTION DOC]
    # Parametrized over sniffing (no separator) and an explicit pipe delimiter; asserts a three-column frame with the first column as a named index.
    data = """index|A|B|C
foo|1|2|3
bar|4|5|6
baz|7|8|9
"""
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


def test_sniff_delimiter_comment(python_parser_only):
    # [FUNCTION DOC]
    # Sniffing the separator in data containing full-line and trailing comments (with a pipe inside a trailing comment) and a comment marker; asserts the comments are ignored and the frame parsed with the first column as index.
    data = """# comment line
index|A|B|C
# comment line
foo|1|2|3 # ignore | this
bar|4|5|6
baz|7|8|9
"""
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, sep=None, comment="#")
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_sniff_delimiter_encoding(python_parser_only, encoding):
    # [FUNCTION DOC]
    # Parametrized over no encoding (text stream) and utf-8 (encoded bytes wrapped in a text wrapper). Sniffs the separator after skipping two junk lines; asserts the same three-column indexed frame.
    parser = python_parser_only
    data = """ignore this
ignore this too
index|A|B|C
foo|1|2|3
bar|4|5|6
baz|7|8|9
"""

    if encoding is not None:
        data = data.encode(encoding)
        data = BytesIO(data)
        data = TextIOWrapper(data, encoding=encoding)
    else:
        data = StringIO(data)

    result = parser.read_csv(data, index_col=0, sep=None, skiprows=2, encoding=encoding)
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


def test_single_line(python_parser_only):
    # [FUNCTION DOC]
    # See gh-6607. A single-line input with sniffed separator, no header and given names parses to a one-row frame.
    # see gh-6607: sniff separator
    parser = python_parser_only
    result = parser.read_csv(StringIO("1,2"), names=["a", "b"], header=None, sep=None)

    expected = DataFrame({"a": [1], "b": [2]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"skipfooter": 2}, {"nrows": 3}])
def test_skipfooter(python_parser_only, kwargs):
    # [FUNCTION DOC]
    # See gh-6607. Parametrized over skipfooter of two and nrows of three: asserts both drop the trailing text lines and give the same three-row frame.
    # see gh-6607
    data = """A,B,C
1,2,3
4,5,6
7,8,9
want to skip this
also also skip this
"""
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), **kwargs)

    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "compression,klass", [("gzip", "GzipFile"), ("bz2", "BZ2File")]
)
def test_decompression_regex_sep(
    temp_file, python_parser_only, csv1, compression, klass
):
    # [FUNCTION DOC]
    # See gh-6607. Parametrized over gzip and bzip2 (module skipped if unavailable). Writes the bundled csv with commas replaced by double colons into a compressed temp file, reads it with a double-colon separator and the compression option, and asserts equality with reading the original file.
    # see gh-6607
    parser = python_parser_only

    with open(csv1, "rb") as f:
        data = f.read()

    data = data.replace(b",", b"::")
    expected = parser.read_csv(csv1)

    module = pytest.importorskip(compression)
    klass = getattr(module, klass)

    with klass(temp_file, mode="wb") as tmp:
        tmp.write(data)

    result = parser.read_csv(temp_file, sep="::", compression=compression)
    tm.assert_frame_equal(result, expected)


def test_read_csv_buglet_4x_multi_index(python_parser_only):
    # [FUNCTION DOC]
    # See gh-6607. Whitespace-separated table whose header has fewer fields than data rows: the first four columns become a multi-index with the given names; asserts the expected five float columns and index values.
    # see gh-6607
    data = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
    parser = python_parser_only

    expected = DataFrame(
        [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.3640],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
            [-0.6662, -0.5243, -0.3580, 0.89145, 2.5838],
        ],
        columns=["A", "B", "C", "D", "E"],
        index=MultiIndex.from_tuples(
            [("a", "b", 10.0032, 5), ("a", "q", 20, 4), ("x", "q", 30, 3)],
            names=["one", "two", "three", "four"],
        ),
    )
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    tm.assert_frame_equal(result, expected)


def test_read_csv_buglet_4x_multi_index2(python_parser_only):
    # [FUNCTION DOC]
    # See gh-6893. Whitespace-separated data with a six-field data row and a three-field header pair; asserts a frame built from records with columns a, b, c, A, B, C and index labels a, b, c.
    # see gh-6893
    data = "      A B C\na b c\n1 3 7 0 3 6\n3 1 4 1 5 9"
    parser = python_parser_only

    expected = DataFrame.from_records(
        [(1, 3, 7, 0, 3, 6), (3, 1, 4, 1, 5, 9)],
        columns=list("abcABC"),
        index=list("abc"),
    )
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("add_footer", [True, False])
def test_skipfooter_with_decimal(python_parser_only, add_footer):
    # [FUNCTION DOC]
    # See gh-6971. Parametrized over adding a stray text footer: with the footer skipped via skipfooter and decimal mark set to a hash sign, the first two lines are still converted to floats. Asserts the frame of 1.2 and 3.4.
    # see gh-6971
    data = "1#2\n3#4"
    parser = python_parser_only
    expected = DataFrame({"a": [1.2, 3.4]})

    if add_footer:
        # The stray footer line should not mess with the
        # casting of the first two lines if we skip it.
        kwargs = {"skipfooter": 1}
        data += "\nFooter"
    else:
        kwargs = {}

    result = parser.read_csv(StringIO(data), names=["a"], decimal="#", **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sep", ["::", "#####", "!!!", "123", "#1!c5", "%!c!d", "@@#4:2", "_!pd#_"]
)
@pytest.mark.parametrize(
    "encoding", ["utf-16", "utf-16-be", "utf-16-le", "utf-32", "cp037"]
)
def test_encoding_non_utf8_multichar_sep(python_parser_only, sep, encoding):
    # [FUNCTION DOC]
    # See gh-3404. Parametrized over many multi-character separators and several non-utf8 encodings (utf-16 variants, utf-32, cp037): encodes the two-field line, reads it from bytes with names, and asserts a one-row frame.
    # see gh-3404
    expected = DataFrame({"a": [1], "b": [2]})
    parser = python_parser_only

    data = "1" + sep + "2"
    encoded_data = data.encode(encoding)

    result = parser.read_csv(
        BytesIO(encoded_data), sep=sep, names=["a", "b"], encoding=encoding
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
def test_multi_char_sep_quotes(python_parser_only, quoting):
    # [FUNCTION DOC]
    # See gh-13374. Parametrized over minimal and no quoting with a two-character separator on data containing a quoted field containing the separator. With no quoting, asserts a parser error expecting two fields in line three but seeing three. With minimal quoting, asserts a parser error saying quoting is ignored with a multi-character delimiter.
    # see gh-13374
    kwargs = {"sep": ",,"}
    parser = python_parser_only

    data = 'a,,b\n1,,a\n2,,"2,,b"'

    if quoting == csv.QUOTE_NONE:
        msg = "Expected 2 fields in line 3, saw 3"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)
    else:
        msg = "ignored when a multi-char delimiter is used"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)


def test_none_delimiter(python_parser_only):
    # [FUNCTION DOC]
    # See gh-13374 and gh-17465. Sniffing the separator with the warn option for bad lines: the malformed third line is skipped with a parser warning (stacklevel not checked) and the remaining rows parse into a three-column frame.
    # see gh-13374 and gh-17465
    parser = python_parser_only
    data = "a,b,c\n0,1,2\n3,4,5,6\n7,8,9"
    expected = DataFrame({"a": [0, 7], "b": [1, 8], "c": [2, 9]})

    # We expect the third line in the data to be
    # skipped because it is malformed, but we do
    # not expect any errors to occur.
    with tm.assert_produces_warning(
        ParserWarning, match="Skipping line 3", check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), header=0, sep=None, on_bad_lines="warn"
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data", ['a\n1\n"b"a', 'a,b,c\ncat,foo,bar\ndog,foo,"baz'])
@pytest.mark.parametrize("skipfooter", [0, 1])
def test_skipfooter_bad_row(python_parser_only, data, skipfooter):
    # [FUNCTION DOC]
    # See gh-13879 and gh-15910. Parametrized over two inputs ending in an unterminated quote and skipfooter of zero or one. With a footer skipped, asserts a parser error about parsing errors in the skipped footer rows; with none, a parser error about unexpected end of data or something expected after the quote.
    # see gh-13879 and gh-15910
    parser = python_parser_only
    if skipfooter:
        msg = "parsing errors in the skipped footer rows"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)
    else:
        msg = "unexpected end of data|expected after"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)


def test_malformed_skipfooter(python_parser_only):
    # [FUNCTION DOC]
    # Data with a leading junk line, header on row one, trailing comment, a too-long row and a footer: asserts a parser error that line four expected three fields but saw five.
    parser = python_parser_only
    data = """ignore
A,B,C
1,2,3 # comment
1,2,3,4,5
2,3,4
footer
"""
    msg = "Expected 3 fields in line 4, saw 5"
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment="#", skipfooter=1)


def test_python_engine_file_no_next(python_parser_only):
    # [FUNCTION DOC]
    # Reads from a file-like object that supports iteration, read and readline but not a next method, using the python parser; passes if no exception is raised (no assertions).
    parser = python_parser_only

    class NoNextBuffer:
        # [CLASS DOC]
        # HELPER: minimal fake file object for test_python_engine_file_no_next. Holds the csv string as data attribute; iteration delegates to the string's iterator, and read and readline both return the whole string.
        def __init__(self, csv_data) -> None:
            # [FUNCTION DOC]
            # HELPER: stores the given csv data in the data attribute.
            self.data = csv_data

        def __iter__(self) -> Iterator:
            # [FUNCTION DOC]
            # HELPER: returns an iterator over the stored data (by delegating to its own iteration).
            return self.data.__iter__()

        def read(self):
            # [FUNCTION DOC]
            # HELPER: returns the whole stored data string (no size argument).
            return self.data

        def readline(self):
            # [FUNCTION DOC]
            # HELPER: returns the whole stored data string as if it were one line.
            return self.data

    parser.read_csv(NoNextBuffer("a\n1"))


@pytest.mark.parametrize("bad_line_func", [lambda x: ["2", "3"], lambda x: x[:2]])
def test_on_bad_lines_callable(python_parser_only, bad_line_func):
    # [FUNCTION DOC]
    # GH 5686. Parametrized over a callable returning a fixed two-item list and one truncating the bad line to its first two items. Rows with too many fields are replaced by the callable's output; asserts the three-row frame with columns a and b.
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_write_to_external_list(python_parser_only):
    # [FUNCTION DOC]
    # GH 5686. The callable records each bad line in an external list and returns a fixed two-item replacement; asserts the same frame and that the list holds exactly the one over-long row as a list of strings.
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)
    lst = []

    def bad_line_func(bad_line: list[str]) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: callback passed as on_bad_lines. Appends the bad line (a list of strings) to the enclosing list as a side effect and returns the replacement row of two strings.
        lst.append(bad_line)
        return ["2", "3"]

    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    tm.assert_frame_equal(result, expected)
    assert lst == [["2", "3", "4", "5", "6"]]


@pytest.mark.parametrize("bad_line_func", [lambda x: ["foo", "bar"], lambda x: x[:2]])
@pytest.mark.parametrize("sep", [",", "111"])
def test_on_bad_lines_callable_iterator_true(python_parser_only, bad_line_func, sep):
    # [FUNCTION DOC]
    # GH 5686. Parametrized over callables (fixed replacement or truncation) and separators (comma and a multi-character one). Uses chunked reading with the iterator option, which has a separate code path from non-iterator reading; asserts each single-row chunk equals the expected row, including the replaced bad line, with a continued row index.
    # GH 5686
    # iterator=True has a separate code path than iterator=False
    parser = python_parser_only
    data = f"""
0{sep}1
hi{sep}there
foo{sep}bar{sep}baz
good{sep}bye
"""
    bad_sio = StringIO(data)
    result_iter = parser.read_csv(
        bad_sio, on_bad_lines=bad_line_func, chunksize=1, iterator=True, sep=sep
    )
    expecteds = [
        {"0": "hi", "1": "there"},
        {"0": "foo", "1": "bar"},
        {"0": "good", "1": "bye"},
    ]
    for i, (result, expected) in enumerate(zip(result_iter, expecteds)):
        expected = DataFrame(expected, index=range(i, i + 1))
        tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_dont_swallow_errors(python_parser_only):
    # [FUNCTION DOC]
    # GH 5686. A callable that raises a value error: asserts the error propagates out of read_csv with its message unchanged.
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)
    msg = "This function is buggy."

    def bad_line_func(bad_line):
        # [FUNCTION DOC]
        # HELPER: callback that always raises a value error with the enclosing test's message.
        raise ValueError(msg)

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(bad_sio, on_bad_lines=bad_line_func)


def test_on_bad_lines_callable_not_expected_length(python_parser_only):
    # [FUNCTION DOC]
    # GH 5686. A callable returning the line unchanged (too many fields): asserts a parser warning that the line returned from the bad-lines callable was truncated/handled, and that the frame holds the first two fields of each row.
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)

    result = parser.read_csv_check_warnings(
        ParserWarning, "from bad_lines callable", bad_sio, on_bad_lines=lambda x: x
    )
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_returns_none(python_parser_only):
    # [FUNCTION DOC]
    # GH 5686. A callable returning None causes the bad line to be dropped; asserts a two-row frame.
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)

    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: None)
    expected = DataFrame({"a": [1, 3], "b": [2, 4]})
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_index_col_inferred(python_parser_only):
    # [FUNCTION DOC]
    # GH 5686. Rows with one more field than the header produce an inferred index from the first field; the callable's replacement is not used because the rows are not too long relative to the implicit index; asserts the frame with columns a and b indexed by the first field.
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2,3
4,5,6
"""
    bad_sio = StringIO(data)

    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: ["99", "99"])
    expected = DataFrame({"a": [2, 5], "b": [3, 6]}, index=[1, 4])
    tm.assert_frame_equal(result, expected)


def test_index_col_false_and_header_none(python_parser_only):
    # [FUNCTION DOC]
    # GH#46955. Header disabled and index_col False with a row longer than the first: asserts a parser warning about header length and a two-column frame with extra fields truncated.
    # GH#46955
    parser = python_parser_only
    data = """
0.5,0.03
0.1,0.2,0.3,2
"""
    result = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header",
        StringIO(data),
        sep=",",
        header=None,
        index_col=False,
    )
    expected = DataFrame({0: [0.5, 0.1], 1: [0.03, 0.2]})
    tm.assert_frame_equal(result, expected)


def test_header_int_do_not_infer_multiindex_names_on_different_line(python_parser_only):
    # [FUNCTION DOC]
    # GH#46569. A header as integer with rows that have more fields than the header and index_col False: asserts a parser warning about header length and a single column of the first values.
    # GH#46569
    parser = python_parser_only
    data = StringIO("a\na,b\nc,d,e\nf,g,h")
    result = parser.read_csv_check_warnings(
        ParserWarning, "Length of header", data, engine="python", index_col=False
    )
    expected = DataFrame({"a": ["a", "c", "f"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", [{"a": object}, {"a": str, "b": np.int64, "c": np.int64}]
)
def test_no_thousand_convert_with_dot_for_non_numeric_cols(python_parser_only, dtype):
    # [FUNCTION DOC]
    # GH#50270. Parametrized over dtype mappings (object for column a, or string for a plus integer columns for b and c). With a dot thousands separator and semicolon separator, asserts that column a keeps its dotted text unchanged while numeric columns b and c are converted using the thousands separator.
    # GH#50270
    parser = python_parser_only
    data = """\
a;b;c
0000.7995;16.000;0
3.03.001.00514;0;4.000
4923.600.041;23.000;131"""
    result = parser.read_csv(
        StringIO(data),
        sep=";",
        dtype=dtype,
        thousands=".",
    )
    expected = DataFrame(
        {
            "a": ["0000.7995", "3.03.001.00514", "4923.600.041"],
            "b": [16000, 0, 23000],
            "c": [0, 4000, 131],
        }
    )
    if dtype["a"] == object:
        expected["a"] = expected["a"].astype(object)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (
            {"a": str, "b": np.float64, "c": np.int64},
            {
                "b": [16000.1, 0, 23000],
                "c": [0, 4001, 131],
            },
        ),
        (
            str,
            {
                "b": ["16,000.1", "0", "23,000"],
                "c": ["0", "4,001", "131"],
            },
        ),
    ],
)
def test_no_thousand_convert_for_non_numeric_cols(python_parser_only, dtype, expected):
    # [FUNCTION DOC]
    # GH#50270. Parametrized over a dtype mapping (string, float, integer) and over a global string dtype with comma thousands separator. Asserts column a stays text; with the mapping, b and c are converted as float and integer; with a global string dtype, all values remain text with separators intact.
    # GH#50270
    parser = python_parser_only
    data = """a;b;c
0000,7995;16,000.1;0
3,03,001,00514;0;4,001
4923,600,041;23,000;131
"""
    result = parser.read_csv(
        StringIO(data),
        sep=";",
        dtype=dtype,
        thousands=",",
    )
    expected = DataFrame(expected)
    expected.insert(0, "a", ["0000,7995", "3,03,001,00514", "4923,600,041"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_col", [None, 0])
def test_on_bad_lines_callable_warns_and_truncates_with_index_col(
    python_parser_only, index_col
):
    # [FUNCTION DOC]
    # GH#61837. Parametrized over no index column and index column zero. The callable returns the bad line extended with two extra items; asserts a parser warning that the line returned from the callable is truncated to the expected number of columns, and the frame has the three rows with fields (and the id index when index_col is zero).
    # GH#61837
    parser = python_parser_only
    data = "id,field_1,field_2\n101,A,B\n102,C,D,E\n103,F,G\n"

    def fixer(bad_line):
        # [FUNCTION DOC]
        # HELPER: callback passed as on_bad_lines; returns a new list made of the bad line's items followed by two extra marker strings (so it is always longer than expected).
        return [*list(bad_line), "EXTRA1", "EXTRA2"]

    result = parser.read_csv_check_warnings(
        ParserWarning,
        "from bad_lines callable",
        StringIO(data),
        on_bad_lines=fixer,
        index_col=index_col,
    )

    if index_col is None:
        expected = DataFrame(
            {
                "id": [101, 102, 103],
                "field_1": ["A", "C", "F"],
                "field_2": ["B", "D", "G"],
            }
        )
    else:
        expected = DataFrame(
            {"field_1": ["A", "C", "F"], "field_2": ["B", "D", "G"]},
            index=Index([101, 102, 103], name="id"),
        )

    tm.assert_frame_equal(result, expected)


def test_read_csv_leading_quote_skip(python_parser_only):
    # [FUNCTION DOC]
    # GH 62739. A leading line containing only a lone quote character is skipped with skiprows one; asserts the remaining space-separated data parse into a one-row frame.
    # GH 62739
    tbl = """\
    "
a b
1 3
"""
    parser = python_parser_only
    result = parser.read_csv(
        StringIO(tbl),
        delimiter=" ",
        skiprows=1,
    )
    expected = DataFrame({"a": [1], "b": [3]})
    tm.assert_frame_equal(result, expected)


def test_read_csv_unclosed_double_quote_in_data_still_errors(python_parser_only):
    # [FUNCTION DOC]
    # GH 62739. A lone quote in the data (not in the skipped part) still raises a parser error about unexpected end of data.
    # GH 62739
    tbl = """\
a b
"
1 3
"""
    parser = python_parser_only
    with pytest.raises(ParserError, match="unexpected end of data"):
        parser.read_csv(StringIO(tbl), delimiter=" ", skiprows=1)


def test_read_csv_skiprows_zero(python_parser_only):
    # [FUNCTION DOC]
    # GH 62739. With skiprows zero (nothing skipped) the leading lone quote is not skipped and a parser error about unexpected end of data is raised.
    # GH 62739
    tbl = """\
"
a b
1 3
"""
    parser = python_parser_only
    # don't skip anything
    with pytest.raises(ParserError, match="unexpected end of data"):
        parser.read_csv(StringIO(tbl), delimiter=" ", skiprows=0, engine="python")
