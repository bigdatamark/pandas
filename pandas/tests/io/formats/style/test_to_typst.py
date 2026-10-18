# [MODULE DOC]
# Tests for typst table output of the table styler, including concatenated
# stylers. The jinja2 skip call runs at import time before the styler import.
# Defines a frame fixture and a styler fixture with precision two. Expected text
# is a table call with a column count, a header row (empty corner cell then
# column labels), a blank line and one bracketed row per data row.
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


def test_basic_table(styler):
    # [FUNCTION DOC]
    # Verifies the typst output for the fixture: four columns, the header row, and the
    # two data rows with floats at two decimals; no trailing newline.
    result = styler.to_typst()
    expected = dedent(
        """\
    #table(
      columns: 4,
      [], [A], [B], [C],

      [0], [0], [-0.61], [ab],
      [1], [1], [-1.22], [cd],
    )"""
    )
    assert result == expected


def test_concat(styler):
    # [FUNCTION DOC]
    # Verifies that a concatenated sum styler adds a sum row; the appended styler has
    # default precision, so its float shows six decimals.
    result = styler.concat(styler.data.agg(["sum"]).style).to_typst()
    expected = dedent(
        """\
    #table(
      columns: 4,
      [], [A], [B], [C],

      [0], [0], [-0.61], [ab],
      [1], [1], [-1.22], [cd],
      [sum], [1], [-1.830000], [abcd],
    )"""
    )
    assert result == expected


def test_concat_recursion(styler):
    # [FUNCTION DOC]
    # Verifies nested concatenation outputs two extra sum rows in nesting order with
    # three and four decimals respectively.
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2.concat(styler3)).to_typst()
    expected = dedent(
        """\
    #table(
      columns: 4,
      [], [A], [B], [C],

      [0], [0], [-0.61], [ab],
      [1], [1], [-1.22], [cd],
      [sum], [1], [-1.830], [abcd],
      [sum], [1], [-1.8300], [abcd],
    )"""
    )
    assert result == expected


def test_concat_chain(styler):
    # [FUNCTION DOC]
    # Verifies chained concatenation gives the same output as the nested case.
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2).concat(styler3).to_typst()
    expected = dedent(
        """\
    #table(
      columns: 4,
      [], [A], [B], [C],

      [0], [0], [-0.61], [ab],
      [1], [1], [-1.22], [cd],
      [sum], [1], [-1.830], [abcd],
      [sum], [1], [-1.8300], [abcd],
    )"""
    )
    assert result == expected
