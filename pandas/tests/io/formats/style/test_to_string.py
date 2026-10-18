# [MODULE DOC]
# Tests for the plain text output of the table styler (including the delimiter
# option and concatenating stylers). The jinja2 skip call runs at import time
# before the styler import. Defines a frame fixture and a styler fixture with
# precision two.
from textwrap import dedent

import pytest

from pandas import (
    DataFrame,
    Series,
)

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-row frame with an integer column A, a float
    # column B (minus 0.61 and minus 1.22) and an object column C of strings.
    return DataFrame(
        {"A": [0, 1], "B": [-0.61, -1.22], "C": Series(["ab", "cd"], dtype=object)}
    )


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture with no unique id suffix
    # and precision of two decimals.
    return Styler(df, uuid_len=0, precision=2)


def test_basic_string(styler):
    # [FUNCTION DOC]
    # Verifies the default text output: a header line with the column labels, then one
    # line per row, fields separated by single spaces, floats shown with the styler
    # precision, and a trailing newline.
    result = styler.to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    """
    )
    assert result == expected


def test_string_delimiter(styler):
    # [FUNCTION DOC]
    # Verifies that the delimiter argument replaces the field separator (a semicolon)
    # in the header and the rows.
    result = styler.to_string(delimiter=";")
    expected = dedent(
        """\
    ;A;B;C
    0;0;-0.61;ab
    1;1;-1.22;cd
    """
    )
    assert result == expected


def test_concat(styler):
    # [FUNCTION DOC]
    # Verifies that concatenating a styler holding the column sums (a one-row
    # aggregated frame with the label sum) appends a sum row to the text output; the
    # appended styler uses its own default precision (so the float shows six decimals)
    # and string values are concatenated.
    result = styler.concat(styler.data.agg(["sum"]).style).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830000 abcd
    """
    )
    assert result == expected


def test_concat_recursion(styler):
    # [FUNCTION DOC]
    # Verifies nested concatenation (a styler concatenated with another that itself
    # concatenates a third): the sum rows appear in nesting order, each formatted
    # with its own precision (three, then four decimals).
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2.concat(styler3)).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830 abcd
    sum 1 -1.8300 abcd
    """
    )
    assert result == expected


def test_concat_chain(styler):
    # [FUNCTION DOC]
    # Verifies chained concatenation (two successive concatenations) yields the same
    # text as the nested case: sum rows with three and four decimals in order.
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2).concat(styler3).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830 abcd
    sum 1 -1.8300 abcd
    """
    )
    assert result == expected
