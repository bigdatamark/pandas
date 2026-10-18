"""
Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of comment handling: trailing
# comments, comment-only lines, interaction with row skipping and header position,
# custom comment characters, comments at the start of data, and a comment character
# inside missing-value markers.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# The pyarrow engine does not support the comment option, so most tests expect a value
# error naming the option for that engine and return early.

from io import StringIO

import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


@pytest.mark.parametrize("na_values", [None, ["NaN"]])
def test_comment(all_parsers, na_values):
    # [FUNCTION DOC]
    # Verifies that text after the comment character at the end of a data line is
    # ignored, with and without an extra missing-value list, giving a two-row float
    # frame. For pyarrow, verifies the unsupported comment option error.
    parser = all_parsers
    data = """A,B,C
1,2.,4.#hello world
5.,NaN,10.0
"""
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", na_values=na_values)
        return
    result = parser.read_csv(StringIO(data), comment="#", na_values=na_values)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("read_kwargs", [{}, {"lineterminator": "*"}, {"sep": r"\s+"}])
def test_line_comment(all_parsers, read_kwargs):
    # [FUNCTION DOC]
    # Verifies lines starting with the comment character (including the first line) are
    # ignored, for three read option sets: defaults, a custom asterisk line terminator
    # (newlines in the data replaced by it), and a whitespace regex separator (commas
    # replaced by spaces). The python engine raises a value error that custom line
    # terminators are unsupported; pyarrow raises the unsupported lineterminator or
    # comment option error.
    # Side effect: adds the comment option to the parametrized options dictionary,
    # mutating the shared parameter object.
    parser = all_parsers
    data = """# empty
A,B,C
1,2.,4.#hello world
#ignore this line
5.,NaN,10.0
"""
    if read_kwargs.get("sep"):
        data = data.replace(",", " ")
    elif read_kwargs.get("lineterminator"):
        data = data.replace("\n", read_kwargs.get("lineterminator"))

    read_kwargs["comment"] = "#"
    if parser.engine == "pyarrow":
        if "lineterminator" in read_kwargs:
            msg = (
                "The 'lineterminator' option is not supported with the 'pyarrow' engine"
            )
        else:
            msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **read_kwargs)
        return
    elif parser.engine == "python" and read_kwargs.get("lineterminator"):
        msg = r"Custom line terminators not supported in python parser \(yet\)"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **read_kwargs)
        return

    result = parser.read_csv(StringIO(data), **read_kwargs)
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows(all_parsers):
    # [FUNCTION DOC]
    # Verifies skipping the first four lines (which include comment lines) leaves the
    # header at the A,B,C line. For pyarrow, verifies the unsupported comment option
    # error.
    parser = all_parsers
    data = """# empty
random line
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # This should ignore the first four lines (including comments).
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", skiprows=4)
        return

    result = parser.read_csv(StringIO(data), comment="#", skiprows=4)
    tm.assert_frame_equal(result, expected)


def test_comment_header(all_parsers):
    # [FUNCTION DOC]
    # Verifies that with comment lines at the top and header row index one, the header
    # is the second non-comment line. For pyarrow, verifies the unsupported comment
    # option error.
    parser = all_parsers
    data = """# empty
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Header should begin at the second non-comment line.
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", header=1)
        return
    result = parser.read_csv(StringIO(data), comment="#", header=1)
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows_header(all_parsers):
    # [FUNCTION DOC]
    # Verifies combining four skipped rows with header row index one (counted after
    # skipping, ignoring comment lines) picks the A,B,C line as header. For pyarrow,
    # verifies the unsupported comment option error.
    parser = all_parsers
    data = """# empty
# second empty line
# third empty line
X,Y,Z
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Skiprows should skip the first 4 lines (including comments),
    # while header should start from the second non-commented line,
    # starting with line 5.
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
        return

    result = parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("comment_char", ["#", "~", "&", "^", "*", "@"])
def test_custom_comment_char(all_parsers, comment_char):
    # [FUNCTION DOC]
    # Verifies that each of several comment characters (hash, tilde, ampersand, caret,
    # asterisk, at sign) strips trailing comments from data lines. For pyarrow, verifies
    # the unsupported comment option error.
    parser = all_parsers
    data = "a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo"

    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data.replace("#", comment_char)), comment=comment_char
            )
        return
    result = parser.read_csv(
        StringIO(data.replace("#", comment_char)), comment=comment_char
    )

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", ["infer", None])
def test_comment_first_line(all_parsers, header):
    # [FUNCTION DOC]
    # Verifies that comment lines before and after the header are ignored: with header
    # inferred the first non-comment line is the header; with no header, the header line
    # becomes the first data row as strings. For pyarrow, verifies the unsupported
    # comment option error.
    # see gh-4623
    parser = all_parsers
    data = "# notes\na,b,c\n# more notes\n1,2,3"

    if header is None:
        expected = DataFrame({0: ["a", "1"], 1: ["b", "2"], 2: ["c", "3"]})
    else:
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", header=header)
        return
    result = parser.read_csv(StringIO(data), comment="#", header=header)
    tm.assert_frame_equal(result, expected)


def test_comment_char_in_default_value(all_parsers, request):
    # [FUNCTION DOC]
    # Verifies that a missing-value marker (a hash-prefixed token) that contains the
    # comment character is still treated as a missing value when it stands alone in a
    # field, while content after the comment character in other fields is dropped
    # (making the affected rows partial). The C engine is marked as expected to fail
    # with an assertion error because there it is interpreted as a comment (known gap).
    # For pyarrow, verifies the unsupported comment option error.
    # GH#34002
    if all_parsers.engine == "c":
        reason = "see gh-34002: works on the python engine but not the c engine"
        # NA value containing comment char is interpreted as comment
        request.applymarker(pytest.mark.xfail(reason=reason, raises=AssertionError))
    parser = all_parsers

    data = (
        "# this is a comment\n"
        "col1,col2,col3,col4\n"
        "1,2,3,4#inline comment\n"
        "4,5#,6,10\n"
        "7,8,#N/A,11\n"
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", na_values="#N/A")
        return
    result = parser.read_csv(StringIO(data), comment="#", na_values="#N/A")
    expected = DataFrame(
        {
            "col1": [1, 4, 7],
            "col2": [2, 5, 8],
            "col3": [3.0, np.nan, np.nan],
            "col4": [4.0, np.nan, 11.0],
        }
    )
    tm.assert_frame_equal(result, expected)
