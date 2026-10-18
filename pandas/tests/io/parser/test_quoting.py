"""
Tests that quoting specifications are properly handled
during parsing for all of the parsers defined in parsers.py
"""
# [MODULE DOC]
# Parser tests for the quoting-related read_csv options (quotechar, quoting, doublequote) run against every parser engine through the all_parsers fixture.
# Module-level state: a pytestmark that ignores the BlockManager-passing deprecation warning, xfail/skip marker shortcuts for the pyarrow engine, and expected error-message patterns that depend on whether the interpreter is Python 3.14 or later (a permissive pattern is used there).
# KNOWN ISSUE: in the non-3.14 branch the second message pattern is assigned twice with the identical value, so the repeated assignment is redundant.

import csv
from io import StringIO

import pytest

from pandas.compat import PY314
from pandas.errors import ParserError

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


if PY314:
    # TODO: write a regex that works with all new possitibilities here
    MSG1 = ""
    MSG2 = r"[\s\S]*"
else:
    MSG1 = "a(n)? 1-character string"
    MSG2 = "string( or None)?"


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"quotechar": "foo"}, f'"quotechar" must be {MSG1}'),
        (
            {"quotechar": None, "quoting": csv.QUOTE_MINIMAL},
            "quotechar must be set if quoting enabled",
        ),
        ({"quotechar": 2}, f'"quotechar" must be {MSG2}, not int'),
    ],
)
@skip_pyarrow  # ParserError: CSV parse error: Empty CSV file or block
def test_bad_quote_char(all_parsers, kwargs, msg):
    # [FUNCTION DOC]
    # Parametrized over (kwargs, message): a multi-character quotechar, quotechar None with minimal quoting, and an integer quotechar. Asserts read_csv raises a type error matching the message. Skipped for the pyarrow engine.
    data = "1,2,3"
    parser = all_parsers

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


@pytest.mark.parametrize(
    "quoting,msg",
    [
        ("foo", '"quoting" must be an integer|Argument'),
        (10, 'bad "quoting" value'),  # quoting must be in the range [0, 3]
    ],
)
@xfail_pyarrow  # ValueError: The 'quoting' option is not supported
def test_bad_quoting(all_parsers, quoting, msg):
    # [FUNCTION DOC]
    # Parametrized over invalid quoting values: a non-integer string and the out-of-range integer ten. Asserts read_csv raises a type error whose message matches. Expected to fail on pyarrow, which rejects the quoting option.
    data = "1,2,3"
    parser = all_parsers

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), quoting=quoting)


def test_quote_char_basic(all_parsers):
    # [FUNCTION DOC]
    # Reads simple text with a double-quoted field using an explicit double-quote quotechar and asserts the quotes are stripped in the resulting frame.
    parser = all_parsers
    data = 'a,b,c\n1,2,"cat"'
    expected = DataFrame([[1, 2, "cat"]], columns=["a", "b", "c"])

    result = parser.read_csv(StringIO(data), quotechar='"')
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quote_char", ["~", "*", "%", "$", "@", "P"])
def test_quote_char_various(all_parsers, quote_char):
    # [FUNCTION DOC]
    # Parametrized over several single-character quote characters: replaces the double quotes in the data with the character, passes it as quotechar and asserts the same frame as the default case.
    parser = all_parsers
    expected = DataFrame([[1, 2, "cat"]], columns=["a", "b", "c"])

    data = 'a,b,c\n1,2,"cat"'
    new_data = data.replace('"', quote_char)

    result = parser.read_csv(StringIO(new_data), quotechar=quote_char)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: The 'quoting' option is not supported
@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
@pytest.mark.parametrize("quote_char", ["", None])
def test_null_quote_char(all_parsers, quoting, quote_char):
    # [FUNCTION DOC]
    # Parametrized over quoting (minimal, none) and quote_char (empty string, None). With minimal quoting, asserts a type error is raised: the message says quotechar must be a one-character string (or a unicode character or None on newer Python) for the python engine with an empty string, else that quotechar must be set when quoting is enabled. With no quoting and a non-python engine, asserts the data parses normally. The python engine with no quoting is not checked. Expected to fail on pyarrow.
    kwargs = {"quotechar": quote_char, "quoting": quoting}
    data = "a,b,c\n1,2,3"
    parser = all_parsers

    if quoting != csv.QUOTE_NONE:
        # Sanity checking.
        if not PY314:
            msg = "1-character string"
        else:
            msg = "unicode character or None"
        msg = (
            f'"quotechar" must be a {msg}'
            if all_parsers.engine == "python" and quote_char == ""
            else "quotechar must be set if quoting enabled"
        )

        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    elif all_parsers.engine != "python":
        # Python doesn't support null/blank quote chars in their csv parsers
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,exp_data",
    [
        ({}, [[1, 2, "foo"]]),  # Test default.
        # QUOTE_MINIMAL only applies to CSV writing, so no effect on reading.
        ({"quotechar": '"', "quoting": csv.QUOTE_MINIMAL}, [[1, 2, "foo"]]),
        # QUOTE_MINIMAL only applies to CSV writing, so no effect on reading.
        ({"quotechar": '"', "quoting": csv.QUOTE_ALL}, [[1, 2, "foo"]]),
        # QUOTE_NONE tells the reader to do no special handling
        # of quote characters and leave them alone.
        ({"quotechar": '"', "quoting": csv.QUOTE_NONE}, [[1, 2, '"foo"']]),
        # QUOTE_NONNUMERIC tells the reader to cast
        # all non-quoted fields to float
        ({"quotechar": '"', "quoting": csv.QUOTE_NONNUMERIC}, [[1.0, 2.0, "foo"]]),
    ],
)
@xfail_pyarrow  # ValueError: The 'quoting' option is not supported
def test_quoting_various(all_parsers, kwargs, exp_data):
    # [FUNCTION DOC]
    # Parametrized over quoting settings and expected data: default, minimal and all (quotes stripped), none (quotes retained in the string), and nonnumeric (unquoted numeric fields become floats). Reads a single quoted-field row with given column names and compares with the expected frame. Expected to fail on pyarrow.
    data = '1,2,"foo"'
    parser = all_parsers
    columns = ["a", "b", "c"]

    result = parser.read_csv(StringIO(data), names=columns, **kwargs)
    expected = DataFrame(exp_data, columns=columns)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "doublequote,exp_data", [(True, [[3, '4 " 5']]), (False, [[3, '4 " 5"']])]
)
def test_double_quote(all_parsers, doublequote, exp_data, request):
    # [FUNCTION DOC]
    # Parametrized over doublequote True/False: reads a field containing an escaped doubled quote and asserts the doubled quote collapses to one quote when enabled, and is kept otherwise (with a trailing quote). For the pyarrow engine with doublequote False the result is marked as an expected failure (mismatched result).
    parser = all_parsers
    data = 'a,b\n3,"4 "" 5"'

    if parser.engine == "pyarrow" and not doublequote:
        mark = pytest.mark.xfail(reason="Mismatched result")
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), quotechar='"', doublequote=doublequote)
    expected = DataFrame(exp_data, columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quotechar", ['"', "\u0001"])
def test_quotechar_unicode(all_parsers, quotechar):
    # [FUNCTION DOC]
    # Parametrized over a regular and a control-character (code point one) quotechar: asserts a one-column file parses to the expected frame (gh-14477).
    # see gh-14477
    data = "a\n1"
    parser = all_parsers
    expected = DataFrame({"a": [1]})

    result = parser.read_csv(StringIO(data), quotechar=quotechar)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("balanced", [True, False])
def test_unbalanced_quoting(all_parsers, balanced, request):
    # [FUNCTION DOC]
    # Parametrized over balanced True/False (gh-22789). Balanced: the missing closing quote is appended and the frame parses. Unbalanced: asserts a parser error with an end-of-file-inside-string message for the C engine and an unexpected-end-of-data message otherwise. The pyarrow engine with the unbalanced case is marked as an expected failure.
    # see gh-22789.
    parser = all_parsers
    data = 'a,b,c\n1,2,"3'

    if parser.engine == "pyarrow" and not balanced:
        mark = pytest.mark.xfail(reason="Mismatched result")
        request.applymarker(mark)

    if balanced:
        # Re-balance the quoting and read in without errors.
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        result = parser.read_csv(StringIO(data + '"'))
        tm.assert_frame_equal(result, expected)
    else:
        msg = (
            "EOF inside string starting at row 1"
            if parser.engine == "c"
            else "unexpected end of data"
        )

        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
