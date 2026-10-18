# [MODULE DOC]
# Tests for the markdown output of frames and series (to_markdown), which
# delegates to the optional tabulate package; the whole module is skipped when
# tabulate is not installed. Covers keyword deprecation, default and custom
# table formats, headers, index handling, buffer versus string output, and
# rejection of a conflicting keyword.

from io import StringIO

import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
import pandas._testing as tm

pytest.importorskip("tabulate")


def test_keyword_deprecation():
    # [FUNCTION DOC]
    # GH 57280. Calling the series markdown writer with positional arguments beyond
    # the buffer must emit the version-4 deprecation warning stating that all
    # arguments except the buffer will become keyword-only. The test uses an empty
    # series and passes no buffer and a mode string positionally.
    # GH 57280
    msg = (
        "Starting with pandas version 4.0 all arguments of to_markdown "
        "except for the argument 'buf' will be keyword-only."
    )
    s = pd.Series()
    with tm.assert_produces_warning(Pandas4Warning, match=msg):
        s.to_markdown(None, "wt")


def test_simple():
    # [FUNCTION DOC]
    # Writes a one-column integer frame to a text buffer in the default format and
    # expects a pipe table with right-aligned numeric columns, an empty-named index
    # column and three rows.
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


def test_empty_frame():
    # [FUNCTION DOC]
    # Writes an empty frame with three columns, the first set as index, to a buffer.
    # The output must be only the header row and the separator row, with left
    # alignment (no data rows).
    buf = StringIO()
    df = pd.DataFrame({"id": [], "first_name": [], "last_name": []}).set_index("id")
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == (
        "| id   | first_name   | last_name   |\n|------|--------------|-------------|"
    )


def test_other_tablefmt():
    # [FUNCTION DOC]
    # Passes an alternative table format name (jira) through to the underlying
    # library; the output must use the doubled header pipe style of that format.
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, tablefmt="jira")
    result = buf.getvalue()
    assert result == "||    ||   0 ||\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"


def test_other_headers():
    # [FUNCTION DOC]
    # Passes custom column headers through to the underlying library; the header row
    # must use the supplied names, with the index column shown without a heading
    # shift as in the expected string.
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, headers=["foo", "bar"])
    result = buf.getvalue()
    assert result == (
        "|   foo |   bar |\n|------:|------:|\n|     0 "
        "|     1 |\n|     1 |     2 |\n|     2 |     3 |"
    )


def test_series():
    # [FUNCTION DOC]
    # Writes a named series to a buffer; the header uses the series name for the
    # value column and the rows list the three values with their index.
    buf = StringIO()
    s = pd.Series([1, 2, 3], name="foo")
    s.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == (
        "|    |   foo |\n|---:|------:|\n|  0 |     1 |\n|  1 |     2 |\n|  2 |     3 |"
    )


def test_no_buf():
    # [FUNCTION DOC]
    # With no buffer, the method returns the markdown table as a string; it must
    # match the default table for the simple frame.
    df = pd.DataFrame([1, 2, 3])
    result = df.to_markdown()
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


@pytest.mark.parametrize("index", [True, False])
def test_index(index):
    # GH 32667

    # [FUNCTION DOC]
    # GH 32667. Parametrized on the index flag. With true the output includes the
    # index column; with false the index column is omitted and only the values are
    # shown.
    df = pd.DataFrame([1, 2, 3])

    result = df.to_markdown(index=index)

    if index:
        expected = (
            "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
        )
    else:
        expected = "|   0 |\n|----:|\n|   1 |\n|   2 |\n|   3 |"
    assert result == expected


def test_showindex_disallowed_in_kwargs():
    # [FUNCTION DOC]
    # GH 32667. Passing the underlying library's own showindex keyword together with
    # the index parameter must raise a value error telling the user to pass index
    # instead (enforced since version 2.0).
    # GH 32667; disallowing showindex in kwargs enforced in 2.0
    df = pd.DataFrame([1, 2, 3])
    with pytest.raises(ValueError, match="Pass 'index' instead of 'showindex"):
        df.to_markdown(index=True, showindex=True)
