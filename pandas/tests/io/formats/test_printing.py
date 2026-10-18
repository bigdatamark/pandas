# Note! This file is aimed specifically at pandas.io.formats.printing utility
# functions, not the general printing of pandas objects.
# [MODULE DOC]
# Tests for the low-level text layout helpers of the formatting package:
# escaping of index names in their string form, column-wise joining of string
# lists (adjoin), the generic pretty printer for scalars, sets, dicts and
# mappings with sequence-length limiting, and the East Asian aware width and
# justification adjuster (including the ambiguous-width display option).
# Defines one small custom mapping class and two test classes.
from collections.abc import Mapping
import string

import pytest

import pandas._config.config as cf

import pandas as pd

from pandas.io.formats import printing


@pytest.mark.parametrize(
    "input_names, expected_names",
    [
        (["'a b"], "['\\'a b']"),  # Escape leading quote
        (["test's b"], "['test\\'s b']"),  # Escape apostrophe
        (["'test' b"], "['\\'test\\' b']"),  # Escape surrounding quotes
        (["test b'"], "['test b\\'']"),  # Escape single quote
        (["test\n' b"], "['test\\n\\' b']"),  # Escape quotes, preserve newline
    ],
)
def test_formatted_index_names(input_names, expected_names):
    # [FUNCTION DOC]
    # Regression test (GH#60190): the string form of the index names list of a
    # frame indexed by a single column escapes quotes and apostrophes (and keeps
    # newlines as escapes). Parametrized over input names with leading,
    # embedded, surrounding and trailing quotes and with a newline; the
    # result must equal the expected escaped list text.
    # GH#60190
    df = pd.DataFrame({name: [1, 2, 3] for name in input_names}).set_index(input_names)
    formatted_names = str(df.index.names)

    assert formatted_names == expected_names


def test_adjoin():
    # [FUNCTION DOC]
    # Joining three equal-length string lists as columns with a spacing of two
    # yields lines whose columns are left aligned to the widest entry plus the
    # spacing, separated by newlines.
    # KNOWN ISSUE: duplicates the adjoin test of the format-base test class
    # below (same data and expected result).
    data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
    expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

    adjoined = printing.adjoin(2, *data)

    assert adjoined == expected


class MyMapping(Mapping):
    # [CLASS DOC]
    # HELPER: minimal read-only mapping used to test pretty printing of
    # mapping types that are not dicts. Always has the two keys a and b (in that
    # order) and every lookup returns the integer four. Subclasses the abstract
    # mapping base class.
    def __getitem__(self, key):
        # [FUNCTION DOC]
        # HELPER: returns the integer four for any key (the key is ignored).
        return 4

    def __iter__(self):
        # [FUNCTION DOC]
        # HELPER: returns an iterator over the two fixed keys a then b.
        return iter(["a", "b"])

    def __len__(self):
        # [FUNCTION DOC]
        # HELPER: returns the fixed length two.
        return 2


class TestPPrintThing:
    # [CLASS DOC]
    # Groups tests of the generic pretty-printing function for bytes-derived
    # text, sequence length limits, sets, dicts, mappings and frozen sets.
    def test_repr_binary_type(self):
        # [FUNCTION DOC]
        # Builds the ascii letters as bytes using the configured display
        # encoding (falling back to a plain conversion on a type error), decodes
        # them to text, and checks that pretty printing with quoting returns the
        # quoted representation while without quoting it returns the text
        # unchanged.
        letters = string.ascii_letters
        try:
            raw = bytes(letters, encoding=cf.get_option("display.encoding"))
        except TypeError:
            raw = bytes(letters)
        b = str(raw.decode("utf-8"))
        res = printing.pprint_thing(b, quote_strings=True)
        assert res == repr(b)
        res = printing.pprint_thing(b, quote_strings=False)
        assert res == b

    def test_repr_obeys_max_seq_limit(self):
        # [FUNCTION DOC]
        # Pretty printing a list of one thousand integers respects the maximum
        # sequence items display option: with a limit of 2000 the output is
        # longer than 1000 characters, with a limit of 5 shorter than 100, and
        # with a limit of 1 shorter than 9.
        with cf.option_context("display.max_seq_items", 2000):
            assert len(printing.pprint_thing(list(range(1000)))) > 1000

        with cf.option_context("display.max_seq_items", 5):
            assert len(printing.pprint_thing(list(range(1000)))) < 100

        with cf.option_context("display.max_seq_items", 1):
            assert len(printing.pprint_thing(list(range(1000)))) < 9

    def test_repr_set(self):
        # [FUNCTION DOC]
        # A one-element set is printed in brace notation.
        assert printing.pprint_thing({1}) == "{1}"

    def test_repr_dict(self):
        # [FUNCTION DOC]
        # A two-entry dict is printed in brace notation with quoted string keys.
        assert printing.pprint_thing({"a": 4, "b": 4}) == "{'a': 4, 'b': 4}"

    def test_repr_mapping(self):
        # [FUNCTION DOC]
        # A non-dict mapping (the helper class) is printed the same way as a
        # dict, with its keys and values.
        assert printing.pprint_thing(MyMapping()) == "{'a': 4, 'b': 4}"

    def test_repr_frozenset(self):
        # [FUNCTION DOC]
        # A frozen set is printed with the type name wrapping the braces.
        assert printing.pprint_thing(frozenset([1, 2])) == "frozenset({1, 2})"


class TestFormatBase:
    # [CLASS DOC]
    # Groups tests of column joining, justification and display width
    # calculation, including the East Asian aware adjuster.
    def test_adjoin(self):
        # [FUNCTION DOC]
        # Same check as the module-level adjoin test: three lists joined with a
        # spacing of two produce left-aligned columns.
        data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
        expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

        adjoined = printing.adjoin(2, *data)

        assert adjoined == expected

    def test_adjoin_unicode(self):
        # [FUNCTION DOC]
        # With full-width Japanese characters, the default joiner pads by
        # character count, while the East Asian adjuster pads by display width
        # (full-width characters count as two). Checks the joined text for
        # spacings of two and seven and the adjuster's measured line widths
        # (13, 13, 16 and 23, 23, 26 respectively).
        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "hhh", "いいい"]]
        expected = "あ  dd  ggg\nb  ええ  hhh\nc  ff  いいい"
        adjoined = printing.adjoin(2, *data)
        assert adjoined == expected

        adj = printing._EastAsianTextAdjustment()

        expected = """あ  dd    ggg
b   ええ  hhh
c   ff    いいい"""

        adjoined = adj.adjoin(2, *data)
        assert adjoined == expected
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 13
        assert adj.len(cols[1]) == 13
        assert adj.len(cols[2]) == 16

        expected = """あ       dd         ggg
b        ええ       hhh
c        ff         いいい"""

        adjoined = adj.adjoin(7, *data)
        assert adjoined == expected
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 23
        assert adj.len(cols[1]) == 23
        assert adj.len(cols[2]) == 26

    def test_justify(self):
        # [FUNCTION DOC]
        # The East Asian adjuster pads single strings to a width for left,
        # center and right modes: ascii text is padded by character count and
        # full-width text by display width (no padding when already at or
        # beyond the width).
        # KNOWN ISSUE: the first three ascii assertions are repeated verbatim,
        # so the duplicate block adds no coverage.
        # The inner helper wraps a single string in a list for the adjuster.
        adj = printing._EastAsianTextAdjustment()

        def just(x, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: justifies one string by passing it as a one-item list to
            # the adjuster's justify method with the remaining arguments, and
            # returns the single justified string.
            # wrapper to test single str
            return adj.justify([x], *args, **kwargs)[0]

        assert just("abc", 5, mode="left") == "abc  "
        assert just("abc", 5, mode="center") == " abc "
        assert just("abc", 5, mode="right") == "  abc"
        assert just("abc", 5, mode="left") == "abc  "
        assert just("abc", 5, mode="center") == " abc "
        assert just("abc", 5, mode="right") == "  abc"

        assert just("パンダ", 5, mode="left") == "パンダ"
        assert just("パンダ", 5, mode="center") == "パンダ"
        assert just("パンダ", 5, mode="right") == "パンダ"

        assert just("パンダ", 10, mode="left") == "パンダ    "
        assert just("パンダ", 10, mode="center") == "  パンダ  "
        assert just("パンダ", 10, mode="right") == "    パンダ"

    def test_east_asian_len(self):
        # [FUNCTION DOC]
        # The adjuster's width measure counts ascii as one, full-width
        # katakana as two and half-width katakana as one per character,
        # including in mixed strings.
        # KNOWN ISSUE: the first two ascii assertions are identical.
        adj = printing._EastAsianTextAdjustment()

        assert adj.len("abc") == 3
        assert adj.len("abc") == 3

        assert adj.len("パンダ") == 6
        assert adj.len("ﾊﾟﾝﾀﾞ") == 5
        assert adj.len("パンダpanda") == 11
        assert adj.len("ﾊﾟﾝﾀﾞpanda") == 10

    def test_ambiguous_width(self):
        # [FUNCTION DOC]
        # Ambiguous-width characters (inverted exclamation marks) count as one
        # column by default and as two when the ambiguous-as-wide display option
        # is on (checked by building a new adjuster inside the option context).
        # Then, with that adjuster still in use after leaving the context, joins
        # mixed data and checks the padded output uses the wide measure.
        adj = printing._EastAsianTextAdjustment()
        assert adj.len("¡¡ab") == 4

        with cf.option_context("display.unicode.ambiguous_as_wide", True):
            adj = printing._EastAsianTextAdjustment()
            assert adj.len("¡¡ab") == 6

        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "¡¡ab", "いいい"]]
        expected = "あ  dd    ggg \nb   ええ  ¡¡ab\nc   ff    いいい"
        adjoined = adj.adjoin(2, *data)
        assert adjoined == expected
