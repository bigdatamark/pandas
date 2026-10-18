"""
Tests that dialects are properly handled during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of csv dialect handling: dialect
# objects and registered dialect names, invalid dialects, and the warnings produced when
# explicit options conflict with a dialect's settings (for delimiter and for other
# options).
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# The pyarrow engine does not support the dialect option, so tests expect a value error
# naming the option for that engine.

import csv
from io import StringIO

import pytest

from pandas.errors import ParserWarning

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def custom_dialect():
    # [FUNCTION DOC]
    # HELPER: fixture returning a pair of a dialect name (weird) and a dictionary of its
    # settings: double-quote off, tilde escape character, colon delimiter,
    # skip-initial-space off, backtick quote character, and quoting mode three (no
    # quoting).
    dialect_name = "weird"
    dialect_kwargs = {
        "doublequote": False,
        "escapechar": "~",
        "delimiter": ":",
        "skipinitialspace": False,
        "quotechar": "`",
        "quoting": 3,
    }
    return dialect_name, dialect_kwargs


def test_dialect(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading with a dialect object (excel-style with quoting disabled) treats
    # the stray quote character as ordinary data, so a field beginning with a quote
    # keeps it and rows are split on commas; the result equals reading the same text
    # without the quote and then re-adding the quote to the affected value. For pyarrow,
    # verifies the unsupported dialect option error.
    parser = all_parsers
    data = """\
label1,label2,label3
index1,"a,c,e
index2,b,d,f
"""

    dia = csv.excel()
    dia.quoting = csv.QUOTE_NONE

    if parser.engine == "pyarrow":
        msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dialect=dia)
        return

    df = parser.read_csv(StringIO(data), dialect=dia)

    data = """\
label1,label2,label3
index1,a,c,e
index2,b,d,f
"""
    exp = parser.read_csv(StringIO(data))
    exp.replace("a", '"a', inplace=True)
    tm.assert_frame_equal(df, exp)


def test_dialect_str(all_parsers):
    # [FUNCTION DOC]
    # Registers a temporary dialect with a colon delimiter, and verifies a dialect given
    # by name is used to split colon-separated data into the expected two-column frame.
    # For pyarrow, verifies the unsupported dialect option error.
    dialect_name = "mydialect"
    parser = all_parsers
    data = """\
fruit:vegetable
apple:broccoli
pear:tomato
"""
    exp = DataFrame({"fruit": ["apple", "pear"], "vegetable": ["broccoli", "tomato"]})

    with tm.with_csv_dialect(dialect_name, delimiter=":"):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), dialect=dialect_name)
            return

        df = parser.read_csv(StringIO(data), dialect=dialect_name)
        tm.assert_frame_equal(df, exp)


def test_invalid_dialect(all_parsers):
    # [FUNCTION DOC]
    # Verifies that passing a class lacking the required dialect attributes raises a
    # value error saying the dialect is invalid.
    class InvalidDialect:
        # [CLASS DOC]
        # HELPER: empty class used as an invalid dialect argument.
        pass

    data = "a\n1"
    parser = all_parsers
    msg = "Invalid dialect"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dialect=InvalidDialect)


@pytest.mark.parametrize(
    "arg",
    [None, "doublequote", "escapechar", "skipinitialspace", "quotechar", "quoting"],
)
@pytest.mark.parametrize("value", ["dialect", "default", "other"])
def test_dialect_conflict_except_delimiter(all_parsers, custom_dialect, arg, value):
    # [FUNCTION DOC]
    # Parametrized over each dialect option (or none) and three kinds of value: the
    # dialect's own value or the reader's default (no warning), or a conflicting value
    # (parser warning containing "Conflicting values for"). Verifies, with a registered
    # custom dialect and colon-separated data, that the frame is read as expected and
    # the appropriate warning is or is not produced. For pyarrow, verifies the
    # unsupported dialect option error (no warning is expected since the call raises).
    # see gh-23761.
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers

    expected = DataFrame({"a": [1], "b": [2]})
    data = "a:b\n1:2"

    warning_klass = None
    kwds = {}

    # arg=None tests when we pass in the dialect without any other arguments.
    if arg is not None:
        if value == "dialect":  # No conflict --> no warning.
            kwds[arg] = dialect_kwargs[arg]
        elif value == "default":  # Default --> no warning.
            from pandas.io.parsers.base_parser import parser_defaults

            kwds[arg] = parser_defaults[arg]
        else:  # Non-default + conflict with dialect --> warning.
            warning_klass = ParserWarning
            kwds[arg] = "blah"

    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(
                    # No warning bc we raise
                    None,
                    "Conflicting values for",
                    StringIO(data),
                    dialect=dialect_name,
                    **kwds,
                )
            return
        result = parser.read_csv_check_warnings(
            warning_klass,
            "Conflicting values for",
            StringIO(data),
            dialect=dialect_name,
            **kwds,
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,warning_klass",
    [
        ({"sep": ","}, None),  # sep is default --> sep_override=True
        ({"sep": "."}, ParserWarning),  # sep isn't default --> sep_override=False
        ({"delimiter": ":"}, None),  # No conflict
        ({"delimiter": None}, None),  # Default arguments --> sep_override=True
        ({"delimiter": ","}, ParserWarning),  # Conflict
        ({"delimiter": "."}, ParserWarning),  # Conflict
    ],
    ids=[
        "sep-override-true",
        "sep-override-false",
        "delimiter-no-conflict",
        "delimiter-default-arg",
        "delimiter-conflict",
        "delimiter-conflict2",
    ],
)
def test_dialect_conflict_delimiter(all_parsers, custom_dialect, kwargs, warning_klass):
    # [FUNCTION DOC]
    # Parametrized over separator and delimiter arguments that conflict or not with the
    # dialect's colon delimiter (a default separator, a non-default separator, the same
    # delimiter, a missing delimiter, and two conflicting delimiters). Verifies the
    # frame is read correctly and a parser warning about conflicting values for the
    # delimiter is emitted only for the conflicting cases. For pyarrow, verifies the
    # unsupported dialect option error.
    # see gh-23761.
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers

    expected = DataFrame({"a": [1], "b": [2]})
    data = "a:b\n1:2"

    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(
                    # no warning bc we raise
                    None,
                    "Conflicting values for 'delimiter'",
                    StringIO(data),
                    dialect=dialect_name,
                    **kwargs,
                )
            return
        result = parser.read_csv_check_warnings(
            warning_klass,
            "Conflicting values for 'delimiter'",
            StringIO(data),
            dialect=dialect_name,
            **kwargs,
        )
        tm.assert_frame_equal(result, expected)
