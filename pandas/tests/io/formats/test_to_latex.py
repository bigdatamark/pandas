# [MODULE DOC]
# Tests for LaTeX output of frames and series (to_latex), which is implemented
# on top of the styler and therefore requires jinja2; the module is skipped
# when that package is missing. Each test builds a small object, converts it with
# a given option set, and compares the entire output text to an expected LaTeX
# snippet. Tests are grouped in classes: basic tabular, longtable, header
# options, bold rows, caption and label, escaping, position, formatters and
# multi-index (multirow and multicolumn). A private helper normalizes the
# expected strings and a short two-column frame fixture is shared.

from datetime import datetime
from textwrap import dedent

import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)

pytest.importorskip("jinja2")


def _dedent(string):
    # [FUNCTION DOC]
    # HELPER: normalizes triple-quoted expected strings for the tests. Parameter
    # string: multi-line text. Returns the text with common leading indentation
    # removed (as the standard dedent function does) and, unlike that function, with
    # all leading whitespace and newlines at the start dropped, so a string that
    # begins on the line after the opening quotes does not start with a newline.
    """Dedent without new line in the beginning.

    Built-in textwrap.dedent would keep new line character in the beginning
    of multi-line string starting from the new line.
    This version drops the leading new line character.
    """
    return dedent(string).lstrip()


@pytest.fixture
def df_short():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-row frame with an integer column a (1, 2) and
    # a string column b (b1, b2), used by caption, label and longtable tests.
    """Short dataframe for testing table/tabular/longtable LaTeX env."""
    return DataFrame({"a": [1, 2], "b": ["b1", "b2"]})


class TestToLatex:
    # [CLASS DOC]
    # Basic tabular output tests: writing to files, index handling, column format,
    # and trivial inputs.
    def test_to_latex_to_file(self, float_frame, temp_file):
        # [FUNCTION DOC]
        # Writing the float fixture frame to a path produces a file with the same text as
        # returning a string.
        float_frame.to_latex(temp_file)
        with open(temp_file, encoding="utf-8") as f:
            assert float_frame.to_latex() == f.read()

    def test_to_latex_to_file_utf8_with_encoding(self, temp_file):
        # [FUNCTION DOC]
        # GH 7061. A frame holding a non-ASCII character written to a path with utf-8
        # encoding matches the string output when read back as utf-8.
        # test with utf-8 and encoding option (GH 7061)
        df = DataFrame([["au\xdfgangen"]])
        df.to_latex(temp_file, encoding="utf-8")
        with open(temp_file, encoding="utf-8") as f:
            assert df.to_latex() == f.read()

    def test_to_latex_to_file_utf8_without_encoding(self, temp_file):
        # [FUNCTION DOC]
        # Same as the previous test without an encoding argument; the default must also
        # be utf-8.
        # test with utf-8 without encoding option
        df = DataFrame([["au\xdfgangen"]])
        df.to_latex(temp_file)
        with open(temp_file, encoding="utf-8") as f:
            assert df.to_latex() == f.read()

    def test_to_latex_tabular_with_index(self):
        # [FUNCTION DOC]
        # Default output of the short frame: tabular with left index column, right
        # aligned integers and left aligned strings, with the toprule, midrule and
        # bottomrule lines and an empty top-left header cell.
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_tabular_without_index(self):
        # [FUNCTION DOC]
        # With index off, the index column and its alignment letter are dropped.
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(index=False)
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            a & b \\
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "bad_column_format",
        [5, 1.2, ["l", "r"], ("r", "c"), {"r", "c", "l"}, {"a": "r", "b": "l"}],
    )
    def test_to_latex_bad_column_format(self, bad_column_format):
        # [FUNCTION DOC]
        # Parametrized on invalid column_format values (number, float, list, tuple, set,
        # mapping). Each must raise a value error saying column_format must be a string.
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        msg = r"`column_format` must be str or unicode"
        with pytest.raises(ValueError, match=msg):
            df.to_latex(column_format=bad_column_format)

    def test_to_latex_column_format_just_works(self, float_frame):
        # [FUNCTION DOC]
        # GH Bug #9402. Passing a three-letter column format for the float fixture frame
        # must not raise. No output is checked.
        # GH Bug #9402
        float_frame.to_latex(column_format="lcr")

    def test_to_latex_column_format(self):
        # [FUNCTION DOC]
        # A custom column format of three letters replaces the automatic alignment
        # specification in the tabular header.
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(column_format="lcr")
        expected = _dedent(
            r"""
            \begin{tabular}{lcr}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_float_format_object_col(self):
        # [FUNCTION DOC]
        # GH#40024. A series mixing a float and a string with a thousands-separator float
        # format: the float is formatted, the string untouched.
        # GH#40024
        ser = Series([1000.0, "test"])
        result = ser.to_latex(float_format="{:,.0f}".format)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & 1,000 \\
            1 & test \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_empty_tabular(self):
        # [FUNCTION DOC]
        # An empty frame gives a tabular with a single left column and only the three
        # rules.
        df = DataFrame()
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{l}
            \toprule
            \midrule
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_series(self):
        # [FUNCTION DOC]
        # A series is written as a two-column table (index and values) with an empty
        # header cell and the column heading zero.
        s = Series(["a", "b", "c"])
        result = s.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & a \\
            1 & b \\
            2 & c \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_midrule_location(self):
        # [FUNCTION DOC]
        # GH 18326. A named index with index_names off: no extra index name row is
        # printed and the midrule follows the header.
        # GH 18326
        df = DataFrame({"a": [1, 2]})
        df.index.name = "foo"
        result = df.to_latex(index_names=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & a \\
            \midrule
            0 & 1 \\
            1 & 2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexLongtable:
    # [CLASS DOC]
    # Tests for the longtable environment.
    def test_to_latex_empty_longtable(self):
        # [FUNCTION DOC]
        # An empty frame in longtable mode gives the repeated header, footer and
        # last-foot blocks with a column span of zero in the continuation line.
        df = DataFrame()
        result = df.to_latex(longtable=True)
        expected = _dedent(
            r"""
            \begin{longtable}{l}
            \toprule
            \midrule
            \endfirsthead
            \toprule
            \midrule
            \endhead
            \midrule
            \multicolumn{0}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_with_index(self):
        # [FUNCTION DOC]
        # The short frame in longtable mode gives first-head, head, foot ("Continued on
        # next page" spanning three columns) and last-foot sections followed by the rows.
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(longtable=True)
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_without_index(self):
        # [FUNCTION DOC]
        # Same with index off; the continuation spans two columns.
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(index=False, longtable=True)
        expected = _dedent(
            r"""
            \begin{longtable}{rl}
            \toprule
            a & b \\
            \midrule
            \endfirsthead
            \toprule
            a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{2}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            1 & b1 \\
            2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "df_data, expected_number",
        [
            ({"a": [1, 2]}, 1),
            ({"a": [1, 2], "b": [3, 4]}, 2),
            ({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, 3),
        ],
    )
    def test_to_latex_longtable_continued_on_next_page(self, df_data, expected_number):
        # [FUNCTION DOC]
        # Parametrized on frames with one, two or three columns and the span they must
        # produce. With index off, the continuation row's multicolumn span equals the
        # number of data columns.
        df = DataFrame(df_data)
        result = df.to_latex(index=False, longtable=True)
        assert rf"\multicolumn{{{expected_number}}}" in result


class TestToLatexHeader:
    # [CLASS DOC]
    # Tests for the header option and decimal handling.
    def test_to_latex_no_header_with_index(self):
        # [FUNCTION DOC]
        # GH 7124. With header off the header row is omitted but the rules and index
        # remain.
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_no_header_without_index(self):
        # [FUNCTION DOC]
        # GH 7124. Header and index both off leaves only the rules and rows.
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(index=False, header=False)
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_specified_header_with_index(self):
        # [FUNCTION DOC]
        # GH 7124. A list of aliases replaces the column names in the header row.
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=["AA", "BB"])
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & AA & BB \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_specified_header_without_index(self):
        # [FUNCTION DOC]
        # GH 7124. Aliases replace column names with the index off.
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=["AA", "BB"], index=False)
        expected = _dedent(
            r"""
            \begin{tabular}{rl}
            \toprule
            AA & BB \\
            \midrule
            1 & b1 \\
            2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "header, num_aliases",
        [
            (["A"], 1),
            (("B",), 1),
            (("Col1", "Col2", "Col3"), 3),
            (("Col1", "Col2", "Col3", "Col4"), 4),
        ],
    )
    def test_to_latex_number_of_items_in_header_missmatch_raises(
        self,
        header,
        num_aliases,
    ):
        # [FUNCTION DOC]
        # GH 7124. Parametrized on header sequences of lengths one, one, three and four
        # (list and tuples). Any length other than two raises a value error stating that
        # two columns are written but the given number of aliases was received.
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        msg = f"Writing 2 cols but got {num_aliases} aliases"
        with pytest.raises(ValueError, match=msg):
            df.to_latex(header=header)

    def test_to_latex_decimal(self):
        # [FUNCTION DOC]
        # GH 12031. The decimal argument replaces the point in floats, which are shown
        # with six decimals by default.
        # GH 12031
        df = DataFrame({"a": [1.0, 2.1], "b": ["b1", "b2"]})
        result = df.to_latex(decimal=",")
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1,000000 & b1 \\
            1 & 2,100000 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexBold:
    # [CLASS DOC]
    # Tests for bold_rows.
    def test_to_latex_bold_rows(self):
        # [FUNCTION DOC]
        # GH 16707. With bold_rows on, index labels are wrapped in a bold text macro.
        # GH 16707
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(bold_rows=True)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            \textbf{0} & 1 & b1 \\
            \textbf{1} & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_no_bold_rows(self):
        # [FUNCTION DOC]
        # GH 16707. With bold_rows off index labels are plain text.
        # GH 16707
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(bold_rows=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexCaptionLabel:
    @pytest.fixture
    # [CLASS DOC]
    # Tests for caption and label options, in table/tabular and longtable
    # environments, with shared string fixtures.
    def caption_table(self):
        # [FUNCTION DOC]
        # Fixture returning a caption text containing a typewriter-font macro, for the
        # table environment.
        """Caption for table/tabular LaTeX environment."""
        return "a table in a \\texttt{table/tabular} environment"

    @pytest.fixture
    def short_caption(self):
        # [FUNCTION DOC]
        # Fixture returning the short caption text (a table).
        """Short caption for testing \\caption[short_caption]{full_caption}."""
        return "a table"

    @pytest.fixture
    def label_table(self):
        # [FUNCTION DOC]
        # Fixture returning the label used for the table environment.
        """Label for table/tabular LaTeX environment."""
        return "tab:table_tabular"

    @pytest.fixture
    def caption_longtable(self):
        # [FUNCTION DOC]
        # Fixture returning a caption text for the longtable environment.
        """Caption for longtable LaTeX environment."""
        return "a table in a \\texttt{longtable} environment"

    @pytest.fixture
    def label_longtable(self):
        # [FUNCTION DOC]
        # Fixture returning the label used for the longtable environment.
        """Label for longtable LaTeX environment."""
        return "tab:longtable"

    def test_to_latex_caption_only(self, df_short, caption_table):
        # [FUNCTION DOC]
        # GH 25436. A caption alone wraps the tabular in a table environment with a
        # caption line.
        # GH 25436
        result = df_short.to_latex(caption=caption_table)
        expected = _dedent(
            r"""
            \begin{table}
            \caption{a table in a \texttt{table/tabular} environment}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_label_only(self, df_short, label_table):
        # [FUNCTION DOC]
        # GH 25436. A label alone wraps the tabular in a table environment with a label
        # line.
        # GH 25436
        result = df_short.to_latex(label=label_table)
        expected = _dedent(
            r"""
            \begin{table}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_caption_and_label(self, df_short, caption_table, label_table):
        # [FUNCTION DOC]
        # GH 25436. Caption and label lines both appear, caption first.
        # GH 25436
        result = df_short.to_latex(caption=caption_table, label=label_table)
        expected = _dedent(
            r"""
            \begin{table}
            \caption{a table in a \texttt{table/tabular} environment}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_caption_and_shortcaption(
        self,
        df_short,
        caption_table,
        short_caption,
    ):
        # [FUNCTION DOC]
        # A two-item caption (full and short) produces a caption line with the short text
        # in brackets before the full text.
        result = df_short.to_latex(caption=(caption_table, short_caption))
        expected = _dedent(
            r"""
            \begin{table}
            \caption[a table]{a table in a \texttt{table/tabular} environment}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short):
        # [FUNCTION DOC]
        # A tuple and a list of the same two strings give identical output.
        caption = ("Long-long-caption", "Short")
        result_tuple = df_short.to_latex(caption=caption)
        result_list = df_short.to_latex(caption=list(caption))
        assert result_tuple == result_list

    def test_to_latex_caption_shortcaption_and_label(
        self,
        df_short,
        caption_table,
        short_caption,
        label_table,
    ):
        # [FUNCTION DOC]
        # Short caption, full caption and label together give the bracketed caption line
        # followed by the label line.
        # test when the short_caption is provided alongside caption and label
        result = df_short.to_latex(
            caption=(caption_table, short_caption),
            label=label_table,
        )
        expected = _dedent(
            r"""
            \begin{table}
            \caption[a table]{a table in a \texttt{table/tabular} environment}
            \label{tab:table_tabular}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    @pytest.mark.parametrize(
        "bad_caption",
        [
            ("full_caption", "short_caption", "extra_string"),
            ("full_caption", "short_caption", 1),
            ("full_caption", "short_caption", None),
            ("full_caption",),
            (None,),
        ],
    )
    def test_to_latex_bad_caption_raises(self, bad_caption):
        # [FUNCTION DOC]
        # Parametrized on captions of the wrong shape (three items with different third
        # items, one-item tuple, tuple of none). Each must raise a value error saying the
        # caption must be a string or a two-tuple of strings.
        # test that wrong number of params is raised
        df = DataFrame({"a": [1]})
        msg = "`caption` must be either a string or 2-tuple of strings"
        with pytest.raises(ValueError, match=msg):
            df.to_latex(caption=bad_caption)

    def test_to_latex_two_chars_caption(self, df_short):
        # [FUNCTION DOC]
        # A two-character string caption is used as a single caption, not unpacked into
        # a long and short caption.
        # test that two chars caption is handled correctly
        # it must not be unpacked into long_caption, short_caption.
        result = df_short.to_latex(caption="xy")
        expected = _dedent(
            r"""
            \begin{table}
            \caption{xy}
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_longtable_caption_only(self, df_short, caption_longtable):
        # [FUNCTION DOC]
        # GH 25436. In longtable mode, the caption is put in the first head with a line
        # break, and repeated with empty brackets in the continuation head.
        # KNOWN ISSUE: the comment says the no-caption, no-label case is covered by a test named for longtable output, but no test with that name exists in this module (the closest are the longtable with and without index tests).
        # GH 25436
        # test when no caption and no label is provided
        # is performed by test_to_latex_longtable()
        result = df_short.to_latex(longtable=True, caption=caption_longtable)
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \caption{a table in a \texttt{longtable} environment} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \caption[]{a table in a \texttt{longtable} environment} \\
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_label_only(self, df_short, label_longtable):
        # [FUNCTION DOC]
        # GH 25436. In longtable mode a label alone is placed on its own line in the first
        # head followed by a line break.
        # GH 25436
        result = df_short.to_latex(longtable=True, label=label_longtable)
        expected = _dedent(
            r"""
            \begin{longtable}{lrl}
            \label{tab:longtable} \\
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected

    def test_to_latex_longtable_caption_and_label(
        self,
        df_short,
        caption_longtable,
        label_longtable,
    ):
        # [FUNCTION DOC]
        # GH 25436. Longtable with caption and label: the first line holds both, the
        # repeated head holds the caption with empty brackets.
        # GH 25436
        result = df_short.to_latex(
            longtable=True,
            caption=caption_longtable,
            label=label_longtable,
        )
        expected = _dedent(
            r"""
        \begin{longtable}{lrl}
        \caption{a table in a \texttt{longtable} environment} \label{tab:longtable} \\
        \toprule
         & a & b \\
        \midrule
        \endfirsthead
        \caption[]{a table in a \texttt{longtable} environment} \\
        \toprule
         & a & b \\
        \midrule
        \endhead
        \midrule
        \multicolumn{3}{r}{Continued on next page} \\
        \midrule
        \endfoot
        \bottomrule
        \endlastfoot
        0 & 1 & b1 \\
        1 & 2 & b2 \\
        \end{longtable}
        """
        )
        assert result == expected

    def test_to_latex_longtable_caption_shortcaption_and_label(
        self,
        df_short,
        caption_longtable,
        short_caption,
        label_longtable,
    ):
        # [FUNCTION DOC]
        # Longtable with short caption, caption and label: the first line shows the short
        # caption in brackets, the full caption and the label.
        # test when the caption, the short_caption and the label are provided
        result = df_short.to_latex(
            longtable=True,
            caption=(caption_longtable, short_caption),
            label=label_longtable,
        )
        expected = _dedent(
            r"""
\begin{longtable}{lrl}
\caption[a table]{a table in a \texttt{longtable} environment} \label{tab:longtable} \\
\toprule
 & a & b \\
\midrule
\endfirsthead
\caption[]{a table in a \texttt{longtable} environment} \\
\toprule
 & a & b \\
\midrule
\endhead
\midrule
\multicolumn{3}{r}{Continued on next page} \\
\midrule
\endfoot
\bottomrule
\endlastfoot
0 & 1 & b1 \\
1 & 2 & b2 \\
\end{longtable}
"""
        )
        assert result == expected


class TestToLatexEscape:
    @pytest.fixture
    # [CLASS DOC]
    # Tests for escaping special LaTeX characters.
    def df_with_symbols(self):
        # [FUNCTION DOC]
        # Fixture: a two-row frame whose column labels contain dollar and caret
        # characters.
        """Dataframe with special characters for testing chars escaping."""
        a = "a"
        b = "b"
        return DataFrame({"co$e^x$": {a: "a", b: "b"}, "co^l1": {a: "a", b: "b"}})

    def test_to_latex_escape_false(self, df_with_symbols):
        # [FUNCTION DOC]
        # With escape off, labels with dollar and caret symbols are written unchanged.
        result = df_with_symbols.to_latex(escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             & co$e^x$ & co^l1 \\
            \midrule
            a & a & a \\
            b & b & b \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_default(self, df_with_symbols):
        # [FUNCTION DOC]
        # gh50871. Since version 2.0 escaping is off by default, so the default output
        # differs from the output with escape enabled.
        # gh50871: in v2.0 escape is False by default (styler.format.escape=None)
        default = df_with_symbols.to_latex()
        specified_true = df_with_symbols.to_latex(escape=True)
        assert default != specified_true

    def test_to_latex_special_escape(self):
        # [FUNCTION DOC]
        # Backslash, caret and tilde in values are written using the text macros for
        # backslash, ASCII circumflex and ASCII tilde when escaping is on.
        df = DataFrame([r"a\b\c", r"^a^b^c", r"~a~b~c"])
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & a\textbackslash b\textbackslash c \\
            1 & \textasciicircum a\textasciicircum b\textasciicircum c \\
            2 & \textasciitilde a\textasciitilde b\textasciitilde c \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_special_chars(self):
        # [FUNCTION DOC]
        # All ten special characters (ampersand, percent, dollar, hash, underscore, braces,
        # tilde, caret, backslash) are escaped as expected with escape on.
        special_characters = ["&", "%", "$", "#", "_", "{", "}", "~", "^", "\\"]
        df = DataFrame(data=special_characters)
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & 0 \\
            \midrule
            0 & \& \\
            1 & \% \\
            2 & \$ \\
            3 & \# \\
            4 & \_ \\
            5 & \{ \\
            6 & \} \\
            7 & \textasciitilde  \\
            8 & \textasciicircum  \\
            9 & \textbackslash  \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_special_chars_in_index_names(self):
        # [FUNCTION DOC]
        # See issues 61309 and 57362. An index whose name consists of all the special
        # characters is escaped in the header line.
        # https://github.com/pandas-dev/pandas/issues/61309
        # https://github.com/pandas-dev/pandas/issues/57362
        index = "&%$#_{}}~^\\"
        df = DataFrame({index: [1, 2, 3]}).set_index(index)
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{l}
            \toprule
            \&\%\$\#\_\{\}\}\textasciitilde \textasciicircum \textbackslash  \\
            \midrule
            1 \\
            2 \\
            3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_escape_special_chars_in_column_name(self):
        # [FUNCTION DOC]
        # The columns axis name with underscore, caret and tilde is escaped in the
        # top-left header cell.
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        df.columns.name = "_^~"
        result = df.to_latex(escape=True)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
            \_\textasciicircum \textasciitilde  & A & B \\
            \midrule
            0 & 1 & a \\
            1 & 2 & b \\
            2 & 3 & c \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_specified_header_special_chars_without_escape(self):
        # [FUNCTION DOC]
        # GH 7124. Header aliases containing dollar signs are written unchanged with
        # escape off.
        # GH 7124
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(header=["$A$", "$B$"], escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrl}
            \toprule
             & $A$ & $B$ \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected


class TestToLatexPosition:
    # [CLASS DOC]
    # Tests for the position option.
    def test_to_latex_position(self):
        # [FUNCTION DOC]
        # A position specifier is added in brackets after the table environment's begin.
        the_position = "h"
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(position=the_position)
        expected = _dedent(
            r"""
            \begin{table}[h]
            \begin{tabular}{lrl}
            \toprule
             & a & b \\
            \midrule
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \bottomrule
            \end{tabular}
            \end{table}
            """
        )
        assert result == expected

    def test_to_latex_longtable_position(self):
        # [FUNCTION DOC]
        # In longtable mode the position specifier is added after the begin of the
        # longtable environment before the column format.
        the_position = "t"
        df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
        result = df.to_latex(longtable=True, position=the_position)
        expected = _dedent(
            r"""
            \begin{longtable}[t]{lrl}
            \toprule
             & a & b \\
            \midrule
            \endfirsthead
            \toprule
             & a & b \\
            \midrule
            \endhead
            \midrule
            \multicolumn{3}{r}{Continued on next page} \\
            \midrule
            \endfoot
            \bottomrule
            \endlastfoot
            0 & 1 & b1 \\
            1 & 2 & b2 \\
            \end{longtable}
            """
        )
        assert result == expected


class TestToLatexFormatters:
    # [CLASS DOC]
    # Tests for formatters, float format and missing value handling.
    def test_to_latex_with_formatters(self):
        # [FUNCTION DOC]
        # Formatters given per column (datetime, float, integer, object) and for the
        # index. Output cells are the formatter results; alignment letters follow dtypes.
        df = DataFrame(
            {
                "datetime64": [
                    datetime(2016, 1, 1),
                    datetime(2016, 2, 5),
                    datetime(2016, 3, 3),
                ],
                "float": [1.0, 2.0, 3.0],
                "int": [1, 2, 3],
                "object": [(1, 2), True, False],
            }
        )

        formatters = {
            "datetime64": lambda x: x.strftime("%Y-%m"),
            "float": lambda x: f"[{x: 4.1f}]",
            "int": lambda x: f"0x{x:x}",
            "object": lambda x: f"-{x!s}-",
            "__index__": lambda x: f"index: {x}",
        }
        result = df.to_latex(formatters=dict(formatters))

        expected = _dedent(
            r"""
            \begin{tabular}{llrrl}
            \toprule
             & datetime64 & float & int & object \\
            \midrule
            index: 0 & 2016-01 & [ 1.0] & 0x1 & -(1, 2)- \\
            index: 1 & 2016-02 & [ 2.0] & 0x2 & -True- \\
            index: 2 & 2016-03 & [ 3.0] & 0x3 & -False- \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_3decimals(self):
        # [FUNCTION DOC]
        # GH 21625. A three-decimal percent float format is applied to a float column.
        # GH 21625
        df = DataFrame({"x": [0.19999]})
        result = df.to_latex(float_format="%.3f")
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & x \\
            \midrule
            0 & 0.200 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_integer(self):
        # [FUNCTION DOC]
        # GH 22270. A zero-decimal format is applied to a float column.
        # GH 22270
        df = DataFrame({"x": [100.0]})
        result = df.to_latex(float_format="%.0f")
        expected = _dedent(
            r"""
            \begin{tabular}{lr}
            \toprule
             & x \\
            \midrule
            0 & 100 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
    def test_to_latex_na_rep_and_float_format(self, na_rep):
        # [FUNCTION DOC]
        # Parametrized on the missing value representation. Floats are formatted with two
        # decimals and missing values use the representation.
        df = DataFrame(
            [
                ["A", 1.2225],
                ["A", None],
            ],
            columns=["Group", "Data"],
        )
        result = df.to_latex(na_rep=na_rep, float_format="{:.2f}".format)
        expected = _dedent(
            rf"""
            \begin{{tabular}}{{llr}}
            \toprule
             & Group & Data \\
            \midrule
            0 & A & 1.22 \\
            1 & A & {na_rep} \\
            \bottomrule
            \end{{tabular}}
            """
        )
        assert result == expected


class TestToLatexMultiindex:
    @pytest.fixture
    # [CLASS DOC]
    # Tests for multi-level row and column indexes, including multirow, multicolumn,
    # names, and styler-based formatting. Two fixtures build a frame with a
    # multi-level row index and one with multi-level columns.
    def multiindex_frame(self):
        # [FUNCTION DOC]
        # Fixture: a five-row frame with a two-level row index (c1, c2 and c3 with
        # sublevels) and four integer columns; built by transposing a frame of series.
        """Multiindex dataframe for testing multirow LaTeX macros."""
        return DataFrame.from_dict(
            {
                ("c1", 0): Series({x: x for x in range(4)}),
                ("c1", 1): Series({x: x + 4 for x in range(4)}),
                ("c2", 0): Series({x: x for x in range(4)}),
                ("c2", 1): Series({x: x + 4 for x in range(4)}),
                ("c3", 0): Series({x: x for x in range(4)}),
            }
        ).T

    @pytest.fixture
    def multicolumn_frame(self):
        # [FUNCTION DOC]
        # Fixture: a five-row frame with two-level columns (c1, c2, c3 with sublevels)
        # and integer data.
        """Multicolumn dataframe for testing multicolumn LaTeX macros."""
        return DataFrame(
            {
                ("c1", 0): {x: x for x in range(5)},
                ("c1", 1): {x: x + 5 for x in range(5)},
                ("c2", 0): {x: x for x in range(5)},
                ("c2", 1): {x: x + 5 for x in range(5)},
                ("c3", 0): {x: x for x in range(5)},
            }
        )

    def test_to_latex_multindex_header(self):
        # [FUNCTION DOC]
        # GH 16718. A two-level index with two header aliases and multirow off: the
        # aliases appear on the first header row and the index names on the second.
        # GH 16718
        df = DataFrame({"a": [0], "b": [1], "c": [2], "d": [3]})
        df = df.set_index(["a", "b"])
        observed = df.to_latex(header=["r1", "r2"], multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llrr}
            \toprule
             &  & r1 & r2 \\
            a & b &  &  \\
            \midrule
            0 & 1 & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert observed == expected

    def test_to_latex_multiindex_empty_name(self):
        # [FUNCTION DOC]
        # GH 18669. A one-level multi-index whose name is empty gives an extra empty row
        # under the header.
        # GH 18669
        mi = pd.MultiIndex.from_product([[1, 2]], names=[""])
        df = DataFrame(-1, index=mi, columns=range(4))
        observed = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrr}
            \toprule
             & 0 & 1 & 2 & 3 \\
             &  &  &  &  \\
            \midrule
            1 & -1 & -1 & -1 & -1 \\
            2 & -1 & -1 & -1 & -1 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert observed == expected

    def test_to_latex_multiindex_column_tabular(self):
        # [FUNCTION DOC]
        # Two-level columns of a single column are written as two header rows.
        df = DataFrame({("x", "y"): ["a"]})
        result = df.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{ll}
            \toprule
             & x \\
             & y \\
            \midrule
            0 & a \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_small_tabular(self):
        # [FUNCTION DOC]
        # A transposed frame with a two-level row index and multirow off writes both
        # index values in the first two columns.
        df = DataFrame({("x", "y"): ["a"]}).T
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             &  & 0 \\
            \midrule
            x & y & a \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_tabular(self, multiindex_frame):
        # [FUNCTION DOC]
        # With multirow off, repeated outer index values are blanked in subsequent rows.
        result = multiindex_frame.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrr}
            \toprule
             &  & 0 & 1 & 2 & 3 \\
            \midrule
            c1 & 0 & 0 & 1 & 2 & 3 \\
             & 1 & 4 & 5 & 6 & 7 \\
            c2 & 0 & 0 & 1 & 2 & 3 \\
             & 1 & 4 & 5 & 6 & 7 \\
            c3 & 0 & 0 & 1 & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumn_tabular(self, multiindex_frame):
        # [FUNCTION DOC]
        # GH 14184. Multi-level columns with named levels: groups are written with the
        # multicolumn macro spanning their sublevels, and level names appear in the first
        # column of each header row.
        # GH 14184
        df = multiindex_frame.T
        df.columns.names = ["a", "b"]
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
            a & \multicolumn{2}{r}{c1} & \multicolumn{2}{r}{c2} & c3 \\
            b & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 4 & 0 & 4 & 0 \\
            1 & 1 & 5 & 1 & 5 & 1 \\
            2 & 2 & 6 & 2 & 6 & 2 \\
            3 & 3 & 7 & 3 & 7 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_index_has_name_tabular(self):
        # [FUNCTION DOC]
        # GH 10660. A named two-level index: a header row after the column row shows the
        # index names; outer repeated values blanked.
        # GH 10660
        df = DataFrame({"a": [0, 0, 1, 1], "b": list("abab"), "c": [1, 2, 3, 4]})
        result = df.set_index(["a", "b"]).to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & c \\
            a & b &  \\
            \midrule
            0 & a & 1 \\
             & b & 2 \\
            1 & a & 3 \\
             & b & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_groupby_tabular(self):
        # [FUNCTION DOC]
        # GH 10660. A grouped describe result (multi-level columns) is written with a
        # float format and escaping: the percent signs in labels are escaped and the
        # summary columns appear under a spanning column label.
        # GH 10660
        df = DataFrame({"a": [0, 0, 1, 1], "b": list("abab"), "c": [1, 2, 3, 4]})
        result = (
            df.groupby("a")
            .describe()
            .to_latex(float_format="{:.1f}".format, escape=True)
        )
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrrrrr}
            \toprule
             & \multicolumn{8}{r}{c} \\
             & count & mean & std & min & 25\% & 50\% & 75\% & max \\
            a &  &  &  &  &  &  &  &  \\
            \midrule
            0 & 2.0 & 1.5 & 0.7 & 1.0 & 1.2 & 1.5 & 1.8 & 2.0 \\
            1 & 2.0 & 3.5 & 0.7 & 3.0 & 3.2 & 3.5 & 3.8 & 4.0 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_dupe_level(self):
        # [FUNCTION DOC]
        # See gh-14484. A repeated inner value is blanked only when all outer levels are
        # also equal; here the inner value must be printed twice because outer values
        # differ.
        # see gh-14484
        #
        # If an index is repeated in subsequent rows, it should be
        # replaced with a blank in the created table. This should
        # ONLY happen if all higher order indices (to the left) are
        # equal too. In this test, 'c' has to be printed both times
        # because the higher order index 'A' != 'B'.
        df = DataFrame(
            index=pd.MultiIndex.from_tuples([("A", "c"), ("B", "c")]), columns=["col"]
        )
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
             &  & col \\
            \midrule
            A & c & NaN \\
            B & c & NaN \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumn_default(self, multicolumn_frame):
        # [FUNCTION DOC]
        # By default multi-level columns use the multicolumn macro with right alignment.
        result = multicolumn_frame.to_latex()
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
             & \multicolumn{2}{r}{c1} & \multicolumn{2}{r}{c2} & c3 \\
             & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 5 & 0 & 5 & 0 \\
            1 & 1 & 6 & 1 & 6 & 1 \\
            2 & 2 & 7 & 2 & 7 & 2 \\
            3 & 3 & 8 & 3 & 8 & 3 \\
            4 & 4 & 9 & 4 & 9 & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumn_false(self, multicolumn_frame):
        # [FUNCTION DOC]
        # With multicolumn off, outer labels are written once in the first cell of the
        # group with empty cells following.
        result = multicolumn_frame.to_latex(multicolumn=False, multicolumn_format="l")
        expected = _dedent(
            r"""
            \begin{tabular}{lrrrrr}
            \toprule
             & c1 & & c2 & & c3 \\
             & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            0 & 0 & 5 & 0 & 5 & 0 \\
            1 & 1 & 6 & 1 & 6 & 1 \\
            2 & 2 & 7 & 2 & 7 & 2 \\
            3 & 3 & 8 & 3 & 8 & 3 \\
            4 & 4 & 9 & 4 & 9 & 4 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multirow_true(self, multicolumn_frame):
        # [FUNCTION DOC]
        # With multirow on, outer index values use the multirow macro spanning their rows,
        # and a partial horizontal line separates groups.
        result = multicolumn_frame.T.to_latex(multirow=True)
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrrr}
            \toprule
             &  & 0 & 1 & 2 & 3 & 4 \\
            \midrule
            \multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            \multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\
            \cline{1-7}
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multicolumnrow_with_multicol_format(self, multicolumn_frame):
        # [FUNCTION DOC]
        # Multirow and multicolumn both on, with the multicolumn alignment set to centered
        # for the group headers; row groups are merged as in the multirow test.
        multicolumn_frame.index = multicolumn_frame.T.index
        result = multicolumn_frame.T.to_latex(
            multirow=True,
            multicolumn=True,
            multicolumn_format="c",
        )
        expected = _dedent(
            r"""
            \begin{tabular}{llrrrrr}
            \toprule
             &  & \multicolumn{2}{c}{c1} & \multicolumn{2}{c}{c2} & c3 \\
             &  & 0 & 1 & 0 & 1 & 0 \\
            \midrule
            \multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            \multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\
             & 1 & 5 & 6 & 7 & 8 & 9 \\
            \cline{1-7}
            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\
            \cline{1-7}
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    @pytest.mark.parametrize("name0", [None, "named0"])
    @pytest.mark.parametrize("name1", [None, "named1"])
    @pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
    def test_to_latex_multiindex_names(self, name0, name1, axes):
        # [FUNCTION DOC]
        # GH 18667. Parametrized on two level names (each none or set) and on which axes
        # (rows, columns, or both) receive names. With a four by four frame of constant
        # values, the expected text is built by placing the column level names in the
        # first cells of the two header rows and, when row names exist, adding an extra
        # row of index names before the midrule.
        # GH 18667
        names = [name0, name1]
        mi = pd.MultiIndex.from_product([[1, 2], [3, 4]])
        df = DataFrame(-1, index=mi.copy(), columns=mi.copy())
        for idx in axes:
            df.axes[idx].names = names

        idx_names = tuple(n or "" for n in names)
        idx_names_row = (
            f"{idx_names[0]} & {idx_names[1]} &  &  &  &  \\\\\n"
            if (0 in axes and any(names))
            else ""
        )
        col_names = [n if (bool(n) and 1 in axes) else "" for n in names]
        observed = df.to_latex(multirow=False)
        expected = r"""\begin{{tabular}}{{llrrrr}}
\toprule
 & {} & \multicolumn{{2}}{{r}}{{1}} & \multicolumn{{2}}{{r}}{{2}} \\
 & {} & 3 & 4 & 3 & 4 \\
{}\midrule
1 & 3 & -1 & -1 & -1 & -1 \\
 & 4 & -1 & -1 & -1 & -1 \\
2 & 3 & -1 & -1 & -1 & -1 \\
 & 4 & -1 & -1 & -1 & -1 \\
\bottomrule
\end{{tabular}}
""".format(*col_names, idx_names_row)
        assert observed == expected

    @pytest.mark.parametrize("one_row", [True, False])
    def test_to_latex_multiindex_nans(self, one_row):
        # [FUNCTION DOC]
        # GH 14249. Parametrized on single-row or two-row frames. A missing index value
        # is written as NaN and a float level shown with six decimals; expected text is
        # assembled in pieces.
        # GH 14249
        df = DataFrame({"a": [None, 1], "b": [2, 3], "c": [4, 5]})
        if one_row:
            df = df.iloc[[0]]
        observed = df.set_index(["a", "b"]).to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & c \\
            a & b &  \\
            \midrule
            NaN & 2 & 4 \\
            """
        )
        if not one_row:
            expected += r"""1.000000 & 3 & 5 \\
"""
        expected += r"""\bottomrule
\end{tabular}
"""
        assert observed == expected

    def test_to_latex_non_string_index(self):
        # [FUNCTION DOC]
        # GH 19981. Integer index levels are written without trouble, with blanking of
        # repeated outer value.
        # GH 19981
        df = DataFrame([[1, 2, 3]] * 2).set_index([0, 1])
        result = df.to_latex(multirow=False)
        expected = _dedent(
            r"""
            \begin{tabular}{llr}
            \toprule
             &  & 2 \\
            0 & 1 &  \\
            \midrule
            1 & 2 & 3 \\
             & 2 & 3 \\
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_multirow(self):
        # [FUNCTION DOC]
        # GH 16719. A three-level index with no data columns, multirow on, escape off:
        # nested multirow macros with partial lines between inner groups and a combined
        # line between outer groups.
        # GH 16719
        mi = pd.MultiIndex.from_product(
            [[0.0, 1.0], [3.0, 2.0, 1.0], ["0", "1"]], names=["i", "val0", "val1"]
        )
        df = DataFrame(index=mi)
        result = df.to_latex(multirow=True, escape=False)
        expected = _dedent(
            r"""
            \begin{tabular}{lll}
            \toprule
            i & val0 & val1 \\
            \midrule
            \multirow[t]{6}{*}{0.000000} & \multirow[t]{2}{*}{3.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{2.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{1.000000} & 0 \\
             &  & 1 \\
            \cline{1-3} \cline{2-3}
            \multirow[t]{6}{*}{1.000000} & \multirow[t]{2}{*}{3.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{2.000000} & 0 \\
             &  & 1 \\
            \cline{2-3}
             & \multirow[t]{2}{*}{1.000000} & 0 \\
             &  & 1 \\
            \cline{1-3} \cline{2-3}
            \bottomrule
            \end{tabular}
            """
        )
        assert result == expected

    def test_to_latex_multiindex_format_single_index_hidden(self):
        # [FUNCTION DOC]
        # GH 52218. Through the styler: hide the index and apply a column style that wraps
        # column headers in bold; output uses the bold macro and omits the index column
        # and rules.
        # GH 52218
        df = DataFrame(
            {
                "A": [1, 2],
                "B": [4, 5],
            }
        )
        result = (
            df.style.hide(axis="index")
            .map_index(lambda v: "textbf:--rwrap;", axis="columns")
            .to_latex()
        )
        expected = _dedent(r"""
            \begin{tabular}{rr}
            \textbf{A} & \textbf{B} \\
            1 & 4 \\
            2 & 5 \\
            \end{tabular}
            """)
        assert result == expected

    def test_to_latex_multiindex_format_triple_index_two_hidden(self):
        # [FUNCTION DOC]
        # GH 52218. Styler with a three-level index of which two levels are hidden: only
        # the last index level and its name row remain, column headers in bold.
        # GH 52218
        arrays = [
            ["A", "A", "B", "B"],
            ["one", "two", "one", "two"],
            ["x", "x", "y", "y"],
        ]
        index = pd.MultiIndex.from_arrays(
            arrays, names=["Level 0", "Level 1", "Level 2"]
        )
        df = DataFrame(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            index=index,
            columns=["C1", "C2", "C3"],
        )
        result = (
            df.style.hide(axis="index", level=[0, 1])
            .map_index(lambda v: "textbf:--rwrap;", axis="columns")
            .to_latex()
        )
        expected = _dedent(r"""
            \begin{tabular}{lrrr}
             & \textbf{C1} & \textbf{C2} & \textbf{C3} \\
            Level 2 &  &  &  \\
            x & 0 & 0 & 0 \\
            x & 0 & 0 & 0 \\
            y & 0 & 0 & 0 \\
            y & 0 & 0 & 0 \\
            \end{tabular}
            """)
        assert result == expected

    def test_to_latex_multiindex_format_triple_index_all_hidden(self):
        # [FUNCTION DOC]
        # GH 52218. Styler with all three index levels hidden: the output has only the
        # bold column headers and data rows.
        # GH 52218
        arrays = [
            ["A", "A", "B", "B"],
            ["one", "two", "one", "two"],
            ["x", "x", "y", "y"],
        ]
        index = pd.MultiIndex.from_arrays(
            arrays, names=["Level 0", "Level 1", "Level 2"]
        )
        df = DataFrame(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            index=index,
            columns=["C1", "C2", "C3"],
        )
        result = (
            df.style.hide(axis="index", level=[0, 1, 2])
            .map_index(lambda v: "textbf:--rwrap;", axis="columns")
            .to_latex()
        )
        expected = _dedent(r"""
            \begin{tabular}{rrr}
            \textbf{C1} & \textbf{C2} & \textbf{C3} \\
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            \end{tabular}
            """)
        assert result == expected
