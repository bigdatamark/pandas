# [MODULE DOC]
# Tests for value and label formatting in the table styler: the value formatter
# (format string, callable, dict per column, missing-value text, subset, thousands
# and decimal separators, precision, escaping for html and latex), the label
# formatter for index and column labels (including levels of a multi-level index),
# the formatter for index names, relabelling of index labels, the global styler
# format options and the internal string escape helper. Most tests inspect the
# translated context (lists of header and body cells with display values) obtained
# from the private translate method with both header and index shown. The jinja2
# skip call runs at import time before the styler imports.
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    IndexSlice,
    MultiIndex,
    NaT,
    Timestamp,
    option_context,
)

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a two-row frame with columns A (zero, one) and B
    # (minus 0.609, minus 1.228) and index x, y.
    return DataFrame(
        data=[[0, -0.609], [1, -1.228]],
        columns=["A", "B"],
        index=["x", "y"],
    )


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture with no unique id suffix.
    return Styler(df, uuid_len=0)


@pytest.fixture
def df_multi():
    # [FUNCTION DOC]
    # HELPER: fixture returning a four-by-four integer frame (zero to fifteen) with a
    # two-level row index (X, Y by x, y) and a two-level column index (A, B by a, b),
    # the row index levels named 0_0 and 0_1 and the column levels 1_0 and 1_1.
    return (
        DataFrame(
            data=np.arange(16).reshape(4, 4),
            columns=MultiIndex.from_product([["A", "B"], ["a", "b"]]),
            index=MultiIndex.from_product([["X", "Y"], ["x", "y"]]),
        )
        .rename_axis(["0_0", "0_1"], axis=0)
        .rename_axis(["1_0", "1_1"], axis=1)
    )


@pytest.fixture
def styler_multi(df_multi):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df_multi fixture with no unique id
    # suffix.
    return Styler(df_multi, uuid_len=0)


def test_display_format(styler):
    # [FUNCTION DOC]
    # Verifies that a one-decimal format string produces display values for every body
    # cell, each (apart from the index cell) at most three characters long.
    ctx = styler.format("{:0.1f}")._translate(True, True)
    assert all(["display_value" in c for c in row] for row in ctx["body"])
    assert all([len(c["display_value"]) <= 3 for c in row[1:]] for row in ctx["body"])
    assert len(ctx["body"][0][1]["display_value"].lstrip("-")) <= 3


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns", [True, False])
def test_display_format_index(styler, index, columns):
    # [FUNCTION DOC]
    # Verifies index label formatting independently for rows (a callable that upper
    # cases) and columns (a string template with asterisks). Parametrized over whether
    # each axis is formatted; checks the display values of the row labels in the body
    # and of the column labels in the header against the expected unformatted or
    # formatted labels.
    exp_index = ["x", "y"]
    if index:
        styler.format_index(lambda v: v.upper(), axis=0)  # test callable
        exp_index = ["X", "Y"]

    exp_columns = ["A", "B"]
    if columns:
        styler.format_index("*{}*", axis=1)  # test string
        exp_columns = ["*A*", "*B*"]

    ctx = styler._translate(True, True)

    for r, row in enumerate(ctx["body"]):
        assert row[0]["display_value"] == exp_index[r]

    for c, col in enumerate(ctx["head"][1:]):
        assert col["display_value"] == exp_columns[c]


def test_format_dict(styler):
    # [FUNCTION DOC]
    # Verifies a dict of per-column formats (one decimal for A, two-decimal percent
    # for B) gives the expected display strings in the first row.
    ctx = styler.format({"A": "{:0.1f}", "B": "{0:.2%}"})._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "0.0"
    assert ctx["body"][0][2]["display_value"] == "-60.90%"


def test_format_index_dict(styler):
    # [FUNCTION DOC]
    # Verifies that a dict keyed by level number given to the index formatter applies
    # the callable to the row labels (upper cased).
    ctx = styler.format_index({0: lambda v: v.upper()})._translate(True, True)
    for i, val in enumerate(["X", "Y"]):
        assert ctx["body"][i][0]["display_value"] == val


def test_format_string(styler):
    # [FUNCTION DOC]
    # Verifies a two-decimal format string applies to all body cells, with rounding.
    ctx = styler.format("{:.2f}")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "0.00"
    assert ctx["body"][0][2]["display_value"] == "-0.61"
    assert ctx["body"][1][1]["display_value"] == "1.00"
    assert ctx["body"][1][2]["display_value"] == "-1.23"


def test_format_callable(styler):
    # [FUNCTION DOC]
    # Verifies a callable formatter returning a sign word depending on the value is
    # applied to all body cells.
    ctx = styler.format(lambda v: "neg" if v < 0 else "pos")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "pos"
    assert ctx["body"][0][2]["display_value"] == "neg"
    assert ctx["body"][1][1]["display_value"] == "pos"
    assert ctx["body"][1][2]["display_value"] == "neg"


def test_format_with_na_rep():
    # [FUNCTION DOC]
    # Regression test (GH 21527 and 28358): the missing-value text replaces missing
    # cells without a formatter, with a percent formatter (other cells formatted) and
    # when restricted to a column subset (only that column is affected by the
    # formatter, the missing text is still applied there).
    # GH 21527 28358
    df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

    ctx = df.style.format(None, na_rep="-")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "-"
    assert ctx["body"][0][2]["display_value"] == "-"

    ctx = df.style.format("{:.2%}", na_rep="-")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "-"
    assert ctx["body"][0][2]["display_value"] == "-"
    assert ctx["body"][1][1]["display_value"] == "110.00%"
    assert ctx["body"][1][2]["display_value"] == "120.00%"

    ctx = df.style.format("{:.2%}", na_rep="-", subset=["B"])._translate(True, True)
    assert ctx["body"][0][2]["display_value"] == "-"
    assert ctx["body"][1][2]["display_value"] == "120.00%"


def test_format_index_with_na_rep():
    # [FUNCTION DOC]
    # Verifies that column labels that are none, nan, not-a-time or the missing
    # marker are displayed as the supplied missing text while a normal label is kept.
    df = DataFrame([[1, 2, 3, 4, 5]], columns=["A", None, np.nan, NaT, NA])
    ctx = df.style.format_index(None, na_rep="--", axis=1)._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "A"
    for i in [2, 3, 4, 5]:
        assert ctx["head"][0][i]["display_value"] == "--"


def test_format_non_numeric_na():
    # [FUNCTION DOC]
    # Regression test (GH 21527 and 28358): missing values in object and datetime
    # columns (none, nan, not-a-time) are all displayed with the supplied missing text.
    # GH 21527 28358
    df = DataFrame(
        {
            "object": [None, np.nan, "foo"],
            "datetime": [None, NaT, Timestamp("20120101")],
        }
    )
    ctx = df.style.format(None, na_rep="-")._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "-"
    assert ctx["body"][0][2]["display_value"] == "-"
    assert ctx["body"][1][1]["display_value"] == "-"
    assert ctx["body"][1][2]["display_value"] == "-"


@pytest.mark.parametrize(
    "func, attr, kwargs",
    [
        ("format", "_display_funcs", {}),
        ("format_index", "_display_funcs_index", {"axis": 0}),
        ("format_index", "_display_funcs_columns", {"axis": 1}),
    ],
)
def test_format_clear(styler, func, attr, kwargs):
    # [FUNCTION DOC]
    # Verifies that calling the value or index (rows or columns) formatter with a
    # format string registers a formatter for the first cell in the corresponding
    # private mapping, and calling it again without a formatter removes it. Parametrized
    # over method name, attribute name and axis keyword.
    assert (0, 0) not in getattr(styler, attr)  # using default
    getattr(styler, func)("{:.2f}", **kwargs)
    assert (0, 0) in getattr(styler, attr)  # formatter is specified
    getattr(styler, func)(**kwargs)
    assert (0, 0) not in getattr(styler, attr)  # formatter cleared to default


@pytest.mark.parametrize(
    "escape, exp",
    [
        ("html", "&lt;&gt;&amp;&#34;%$#_{}~^\\~ ^ \\ "),
        (
            "latex",
            '<>\\&"\\%\\$\\#\\_\\{\\}\\textasciitilde \\textasciicircum '
            "\\textbackslash \\textasciitilde \\space \\textasciicircum \\space "
            "\\textbackslash \\space ",
        ),
    ],
)
def test_format_escape_html(escape, exp):
    # [FUNCTION DOC]
    # Verifies escaping in value and label formatting. With escaping off, the special
    # characters appear raw in the formatter output; with html or latex escaping, only
    # the value is escaped and not the template text around it (the template adds
    # ampersands). Checked on rendered html for values and on the translated header
    # for index labels. Parametrized over the escape mode and its expected escaped
    # text.
    # KNOWN ISSUE: the function name mentions html only but it is also parametrized
    # over and tests latex escaping.
    chars = '<>&"%$#_{}~^\\~ ^ \\ '
    df = DataFrame([[chars]])

    s = Styler(df, uuid_len=0).format("&{0}&", escape=None)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{chars}&</td>'
    assert expected in s.to_html()

    # only the value should be escaped before passing to the formatter
    s = Styler(df, uuid_len=0).format("&{0}&", escape=escape)
    expected = f'<td id="T__row0_col0" class="data row0 col0" >&{exp}&</td>'
    assert expected in s.to_html()

    # also test format_index()
    styler = Styler(DataFrame(columns=[chars]), uuid_len=0)
    styler.format_index("&{0}&", escape=None, axis=1)
    assert styler._translate(True, True)["head"][0][1]["display_value"] == f"&{chars}&"
    styler.format_index("&{0}&", escape=escape, axis=1)
    assert styler._translate(True, True)["head"][0][1]["display_value"] == f"&{exp}&"


@pytest.mark.parametrize(
    "chars, expected",
    [
        (
            r"$ \$&%#_{}~^\ $ &%#_{}~^\ $",
            "".join(
                [
                    r"$ \$&%#_{}~^\ $ ",
                    r"\&\%\#\_\{\}\textasciitilde \textasciicircum ",
                    r"\textbackslash \space \$",
                ]
            ),
        ),
        (
            r"\( &%#_{}~^\ \) &%#_{}~^\ \(",
            "".join(
                [
                    r"\( &%#_{}~^\ \) ",
                    r"\&\%\#\_\{\}\textasciitilde \textasciicircum ",
                    r"\textbackslash \space \textbackslash (",
                ]
            ),
        ),
        (
            r"$\&%#_{}^\$",
            r"\$\textbackslash \&\%\#\_\{\}\textasciicircum \textbackslash \$",
        ),
        (
            r"$ \frac{1}{2} $ \( \frac{1}{2} \)",
            "".join(
                [
                    r"$ \frac{1}{2} $",
                    r" \textbackslash ( \textbackslash frac\{1\}\{2\} \textbackslash )",
                ]
            ),
        ),
    ],
)
def test_format_escape_latex_math(chars, expected):
    # [FUNCTION DOC]
    # Regression test (GH 51903): the latex-math escape mode escapes each cell
    # separately: text inside dollar signs or inside backslash-parenthesis pairs is
    # left raw, text outside is escaped (specials prefixed, tilde and caret
    # spelled out, backslashes spelled out). Parametrized over four strings with
    # combinations of dollar signs, parentheses and braces and their expected display
    # text.
    # GH 51903
    # latex-math escape works for each DataFrame cell separately. If we have
    # a combination of dollar signs and brackets, the dollar sign would apply.
    df = DataFrame([[chars]])
    s = df.style.format("{0}", escape="latex-math")
    assert s._translate(True, True)["body"][0][1]["display_value"] == expected


def test_format_escape_na_rep():
    # [FUNCTION DOC]
    # Verifies the missing-value text is not escaped: with html escaping and a
    # template, a missing cell shows the raw ampersand text while an ordinary cell is
    # escaped; the same holds for column labels with the label formatter.
    # tests the na_rep is not escaped
    df = DataFrame([['<>&"', None]])
    s = Styler(df, uuid_len=0).format("X&{0}>X", escape="html", na_rep="&")
    ex = '<td id="T__row0_col0" class="data row0 col0" >X&&lt;&gt;&amp;&#34;>X</td>'
    expected2 = '<td id="T__row0_col1" class="data row0 col1" >&</td>'
    assert ex in s.to_html()
    assert expected2 in s.to_html()

    # also test for format_index()
    df = DataFrame(columns=['<>&"', None])
    styler = Styler(df, uuid_len=0)
    styler.format_index("X&{0}>X", escape="html", na_rep="&", axis=1)
    ctx = styler._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "X&&lt;&gt;&amp;&#34;>X"
    assert ctx["head"][0][2]["display_value"] == "&"


def test_format_escape_floats(styler):
    # [FUNCTION DOC]
    # Verifies that escaping does not change numeric formatting: a one-decimal format
    # string and a precision of one both render the expected values in html with
    # html escaping on.
    # test given formatter for number format is not impacted by escape
    s = styler.format("{:.1f}", escape="html")
    for expected in [">0.0<", ">1.0<", ">-1.2<", ">-0.6<"]:
        assert expected in s.to_html()
    # tests precision of floats is not impacted by escape
    s = styler.format(precision=1, escape="html")
    for expected in [">0<", ">1<", ">-1.2<", ">-0.6<"]:
        assert expected in s.to_html()


@pytest.mark.parametrize("formatter", [5, True, [2.0]])
@pytest.mark.parametrize("func", ["format", "format_index"])
def test_format_raises(styler, formatter, func):
    # [FUNCTION DOC]
    # Verifies that a formatter that is not a string or callable (an integer, a
    # boolean or a list) raises a type error saying a string or callable is expected.
    # Parametrized over formatter value and over the value and label formatter methods.
    with pytest.raises(TypeError, match="expected str or callable"):
        getattr(styler, func)(formatter)


@pytest.mark.parametrize(
    "precision, expected",
    [
        (1, ["1.0", "2.0", "3.2", "4.6"]),
        (2, ["1.00", "2.01", "3.21", "4.57"]),
        (3, ["1.000", "2.009", "3.212", "4.566"]),
    ],
)
def test_format_with_precision(precision, expected):
    # [FUNCTION DOC]
    # Regression test (issue 13257): the precision argument rounds both the values and
    # the column labels of a one-row float frame whose labels equal its values.
    # Parametrized over precision one, two, three with expected strings. Uses a
    # styler without a fixed unique id.
    # Issue #13257
    df = DataFrame([[1.0, 2.0090, 3.2121, 4.566]], columns=[1.0, 2.0090, 3.2121, 4.566])
    styler = Styler(df)
    styler.format(precision=precision)
    styler.format_index(precision=precision, axis=1)

    ctx = styler._translate(True, True)
    for col, exp in enumerate(expected):
        assert ctx["body"][0][col + 1]["display_value"] == exp  # format test
        assert ctx["head"][0][col + 1]["display_value"] == exp  # format_index test


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "level, expected",
    [
        (0, ["X", "X", "_", "_"]),  # level int
        ("zero", ["X", "X", "_", "_"]),  # level name
        (1, ["_", "_", "X", "X"]),  # other level int
        ("one", ["_", "_", "X", "X"]),  # other level name
        ([0, 1], ["X", "X", "X", "X"]),  # both levels
        ([0, "zero"], ["X", "X", "_", "_"]),  # level int and name simultaneous
        ([0, "one"], ["X", "X", "X", "X"]),  # both levels as int and name
        (["one", "zero"], ["X", "X", "X", "X"]),  # both level names, reversed
    ],
)
def test_format_index_level(axis, level, expected):
    # [FUNCTION DOC]
    # Verifies the label formatter's level argument on a two-level index with all
    # labels equal to an underscore: a callable replacing the label with X is applied
    # to the selected levels only. Parametrized over axis and level selection (by
    # number, by name, a list of both, mixed, reversed); the expected result lists
    # display values of the first level followed by the second level for the chosen
    # axis.
    midx = MultiIndex.from_arrays([["_", "_"], ["_", "_"]], names=["zero", "one"])
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx

    styler = df.style.format_index(lambda v: "X", level=level, axis=axis)
    ctx = styler._translate(True, True)

    if axis == 0:  # compare index
        result = [ctx["body"][s][0]["display_value"] for s in range(2)]
        result += [ctx["body"][s][1]["display_value"] for s in range(2)]
    else:  # compare columns
        result = [ctx["head"][0][s + 1]["display_value"] for s in range(2)]
        result += [ctx["head"][1][s + 1]["display_value"] for s in range(2)]

    assert expected == result


def test_format_subset():
    # [FUNCTION DOC]
    # Verifies that the subset argument restricts formatting: a dict of formats
    # restricted to the first row, a plain format restricted to the first row, to a
    # column given by a single label, to a single cell, and to a list of rows and
    # columns; cells
    # outside the subset keep default six-decimal display.
    df = DataFrame([[0.1234, 0.1234], [1.1234, 1.1234]], columns=["a", "b"])
    ctx = df.style.format(
        {"a": "{:0.1f}", "b": "{0:.2%}"}, subset=IndexSlice[0, :]
    )._translate(True, True)
    expected = "0.1"
    raw_11 = "1.123400"
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11
    assert ctx["body"][0][2]["display_value"] == "12.34%"

    ctx = df.style.format("{:0.1f}", subset=IndexSlice[0, :])._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11

    ctx = df.style.format("{:0.1f}", subset=IndexSlice["a"])._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][0][2]["display_value"] == "0.123400"

    ctx = df.style.format("{:0.1f}", subset=IndexSlice[0, "a"])._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == raw_11

    ctx = df.style.format("{:0.1f}", subset=IndexSlice[[0, 1], ["a"]])._translate(
        True, True
    )
    assert ctx["body"][0][1]["display_value"] == expected
    assert ctx["body"][1][1]["display_value"] == "1.1"
    assert ctx["body"][0][2]["display_value"] == "0.123400"
    assert ctx["body"][1][2]["display_value"] == raw_11


@pytest.mark.parametrize("formatter", [None, "{:,.1f}"])
@pytest.mark.parametrize("decimal", [".", "*"])
@pytest.mark.parametrize("precision", [None, 2])
@pytest.mark.parametrize("func, col", [("format", 1), ("format_index", 0)])
def test_format_thousands(formatter, decimal, precision, func, col):
    # [FUNCTION DOC]
    # Verifies the thousands separator (an underscore) appears in the display value of
    # a large float, integer and complex value, with or without a formatter string,
    # with different decimal characters and precision settings. Parametrized over
    # formatter, decimal separator, precision, and over whether the value formatter
    # (testing the first column) or the label formatter (testing the index) is used.
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(  # testing float
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    assert "1_000_000" in result["body"][0][col]["display_value"]

    styler = DataFrame([[1000000]], index=[1000000]).style
    result = getattr(styler, func)(  # testing int
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    assert "1_000_000" in result["body"][0][col]["display_value"]

    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(  # testing complex
        thousands="_", formatter=formatter, decimal=decimal, precision=precision
    )._translate(True, True)
    assert "1_000_000" in result["body"][0][col]["display_value"]


@pytest.mark.parametrize("formatter", [None, "{:,.4f}"])
@pytest.mark.parametrize("thousands", [None, ",", "*"])
@pytest.mark.parametrize("precision", [None, 4])
@pytest.mark.parametrize("func, col", [("format", 1), ("format_index", 0)])
def test_format_decimal(formatter, thousands, precision, func, col):
    # [FUNCTION DOC]
    # Verifies the decimal separator (an underscore) appears in the display value of a
    # large float and a complex value (the digits after the separator are checked),
    # regardless of the thousands setting, formatter and precision. Parametrized over
    # formatter, thousands separator, precision and over the value formatter versus
    # the label formatter.
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(  # testing float
        decimal="_", formatter=formatter, thousands=thousands, precision=precision
    )._translate(True, True)
    assert "000_123" in result["body"][0][col]["display_value"]

    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(  # testing complex
        decimal="_", formatter=formatter, thousands=thousands, precision=precision
    )._translate(True, True)
    assert "000_123" in result["body"][0][col]["display_value"]


def test_str_escape_error():
    # [FUNCTION DOC]
    # Verifies the private escape helper raises a value error with the message
    # listing allowed modes when given an unknown mode string or a non-string mode
    # for text input, but does not raise for a float input with an invalid mode (the
    # mode is ignored for non-text).
    msg = "`escape` only permitted in {'html', 'latex', 'latex-math'}, got "
    with pytest.raises(ValueError, match=msg):
        _str_escape("text", "bad_escape")

    with pytest.raises(ValueError, match=msg):
        _str_escape("text", [])

    _str_escape(2.00, "bad_escape")  # OK since dtype is float


def test_long_int_formatting():
    # [FUNCTION DOC]
    # Verifies a large integer (nineteen digits) is displayed in full by default and
    # with the thousands separator when requested, i.e. without float precision loss.
    df = DataFrame(data=[[1234567890123456789]], columns=["test"])
    styler = df.style
    ctx = styler._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "1234567890123456789"

    styler = df.style.format(thousands="_")
    ctx = styler._translate(True, True)
    assert ctx["body"][0][1]["display_value"] == "1_234_567_890_123_456_789"


def test_format_options():
    # [FUNCTION DOC]
    # Verifies the global styler format options (missing-value text, decimal,
    # precision, thousands, escape in html, latex and latex-math modes, and formatter
    # as a dict) each change the display values of a mixed frame while inside an
    # option context, with defaults checked outside. The frame has integer, float with
    # a missing value and string columns.
    df = DataFrame({"int": [2000, 1], "float": [1.009, None], "str": ["&<", "&~"]})
    ctx = df.style._translate(True, True)

    # test option: na_rep
    assert ctx["body"][1][2]["display_value"] == "nan"
    with option_context("styler.format.na_rep", "MISSING"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][1][2]["display_value"] == "MISSING"

    # test option: decimal and precision
    assert ctx["body"][0][2]["display_value"] == "1.009000"
    with option_context("styler.format.decimal", "_"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][2]["display_value"] == "1_009000"
    with option_context("styler.format.precision", 2):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][2]["display_value"] == "1.01"

    # test option: thousands
    assert ctx["body"][0][1]["display_value"] == "2000"
    with option_context("styler.format.thousands", "_"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][1]["display_value"] == "2_000"

    # test option: escape
    assert ctx["body"][0][3]["display_value"] == "&<"
    assert ctx["body"][1][3]["display_value"] == "&~"
    with option_context("styler.format.escape", "html"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][3]["display_value"] == "&amp;&lt;"
    with option_context("styler.format.escape", "latex"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][1][3]["display_value"] == "\\&\\textasciitilde "
    with option_context("styler.format.escape", "latex-math"):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][1][3]["display_value"] == "\\&\\textasciitilde "

    # test option: formatter
    with option_context("styler.format.formatter", {"int": "{:,.2f}"}):
        ctx_with_op = df.style._translate(True, True)
        assert ctx_with_op["body"][0][1]["display_value"] == "2,000.00"


def test_precision_zero(df):
    # [FUNCTION DOC]
    # Verifies that a styler precision of zero displays the floats in the second
    # column as rounded integers (minus one).
    styler = Styler(df, precision=0)
    ctx = styler._translate(True, True)
    assert ctx["body"][0][2]["display_value"] == "-1"
    assert ctx["body"][1][2]["display_value"] == "-1"


@pytest.mark.parametrize(
    "formatter, exp",
    [
        (lambda x: f"{x:.3f}", "9.000"),
        ("{:.2f}", "9.00"),
        ({0: "{:.1f}"}, "9.0"),
        (None, "9"),
    ],
)
def test_formatter_options_validator(formatter, exp):
    # [FUNCTION DOC]
    # Verifies the formatter option accepts a callable, a format string, a dict keyed
    # by column, or none, and that the latex output of a one-cell frame contains the
    # expected formatted value padded with spaces. Parametrized over formatter and
    # expected text.
    df = DataFrame([[9]])
    with option_context("styler.format.formatter", formatter):
        assert f" {exp} " in df.style.to_latex()


def test_formatter_options_raises():
    # [FUNCTION DOC]
    # Verifies that setting the formatter option to a list raises a value error
    # saying the value must be an instance of the allowed types.
    msg = "Value must be an instance of"
    with pytest.raises(ValueError, match=msg):
        with option_context("styler.format.formatter", ["bad", "type"]):
            DataFrame().style.to_latex()


def test_1level_multiindex():
    # [FUNCTION DOC]
    # Regression test (GH 43383): a row index with a single level that has an empty
    # name is displayed with its labels visible in the translated context.
    # GH 43383
    midx = MultiIndex.from_product([[1, 2]], names=[""])
    df = DataFrame(-1, index=midx, columns=[0, 1])
    ctx = df.style._translate(True, True)
    assert ctx["body"][0][0]["display_value"] == "1"
    assert ctx["body"][0][0]["is_visible"] is True
    assert ctx["body"][1][0]["display_value"] == "2"
    assert ctx["body"][1][0]["is_visible"] is True


def test_boolean_format():
    # [FUNCTION DOC]
    # Regression test (gh 46384): boolean values keep their boolean display value
    # (not converted to integers) in the translated context.
    # gh 46384: booleans do not collapse to integer representation on display
    df = DataFrame([[True, False]])
    ctx = df.style._translate(True, True)
    assert ctx["body"][0][1]["display_value"] is True
    assert ctx["body"][0][2]["display_value"] is False


@pytest.mark.parametrize(
    "hide, labels",
    [
        (False, [1, 2]),
        (True, [1, 2, 3, 4]),
    ],
)
def test_relabel_raise_length(styler_multi, hide, labels):
    # [FUNCTION DOC]
    # Verifies that relabelling the row index with a list of labels whose length does
    # not match the number of displayed rows raises a value error about the labels
    # length. Parametrized over whether rows are first hidden (hidden: expects the
    # visible count, so a list of four fails) or not (a list of two fails with four
    # rows).
    if hide:
        styler_multi.hide(axis=0, subset=[("X", "x"), ("Y", "y")])
    with pytest.raises(ValueError, match="``labels`` must be of length equal"):
        styler_multi.relabel_index(labels=labels)


def test_relabel_index(styler_multi):
    # [FUNCTION DOC]
    # Verifies relabelling a two-level row index with a list of tuples after hiding
    # two rows: the visible rows show the new level labels as display values while
    # values stay the original labels.
    labels = [(1, 2), (3, 4)]
    styler_multi.hide(axis=0, subset=[("X", "x"), ("Y", "y")])
    styler_multi.relabel_index(labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {"value": "X", "display_value": 1}.items() <= ctx["body"][0][0].items()
    assert {"value": "y", "display_value": 2}.items() <= ctx["body"][0][1].items()
    assert {"value": "Y", "display_value": 3}.items() <= ctx["body"][1][0].items()
    assert {"value": "x", "display_value": 4}.items() <= ctx["body"][1][1].items()


def test_relabel_columns(styler_multi):
    # [FUNCTION DOC]
    # Verifies relabelling a two-level column index with tuples after hiding two
    # columns: the visible header cells show the new labels as display values while
    # values stay original.
    labels = [(1, 2), (3, 4)]
    styler_multi.hide(axis=1, subset=[("A", "a"), ("B", "b")])
    styler_multi.relabel_index(axis=1, labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {"value": "A", "display_value": 1}.items() <= ctx["head"][0][3].items()
    assert {"value": "B", "display_value": 3}.items() <= ctx["head"][0][4].items()
    assert {"value": "b", "display_value": 2}.items() <= ctx["head"][1][3].items()
    assert {"value": "a", "display_value": 4}.items() <= ctx["head"][1][4].items()


def test_relabel_roundtrip(styler):
    # [FUNCTION DOC]
    # Verifies relabelling the row index with identity templates leaves values and
    # display values equal to the original labels.
    styler.relabel_index(["{}", "{}"])
    ctx = styler._translate(True, True)
    assert {"value": "x", "display_value": "x"}.items() <= ctx["body"][0][0].items()
    assert {"value": "y", "display_value": "y"}.items() <= ctx["body"][1][0].items()


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "level, expected",
    [
        (0, ["X", "one"]),  # level int
        ("zero", ["X", "one"]),  # level name
        (1, ["zero", "X"]),  # other level int
        ("one", ["zero", "X"]),  # other level name
        ([0, 1], ["X", "X"]),  # both levels
        ([0, "zero"], ["X", "one"]),  # level int and name simultaneous
        ([0, "one"], ["X", "X"]),  # both levels as int and name
        (["one", "zero"], ["X", "X"]),  # both level names, reversed
    ],
)
def test_format_index_names_level(axis, level, expected):
    # [FUNCTION DOC]
    # Verifies the index-names formatter's level argument on a two-level index or
    # column index with names zero and one: a callable replacing the name with X is
    # applied to the selected levels only. Parametrized over axis and level
    # selections (as in the label level test); the expected result is the displayed
    # names of the two levels.
    midx = MultiIndex.from_arrays([["_", "_"], ["_", "_"]], names=["zero", "one"])
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx

    styler = df.style.format_index_names(lambda v: "X", level=level, axis=axis)
    ctx = styler._translate(True, True)

    if axis == 0:  # compare index
        result = [ctx["head"][1][s]["display_value"] for s in range(2)]
    else:  # compare columns
        result = [ctx["head"][s][0]["display_value"] for s in range(2)]
    assert expected == result


@pytest.mark.parametrize(
    "attr, kwargs",
    [
        ("_display_funcs_index_names", {"axis": 0}),
        ("_display_funcs_column_names", {"axis": 1}),
    ],
)
def test_format_index_names_clear(styler, attr, kwargs):
    # [FUNCTION DOC]
    # Verifies that the index-names formatter registers a formatter for the first
    # level in the corresponding private mapping (index names or column names
    # depending on axis) and that calling it with no formatter removes it.
    assert 0 not in getattr(styler, attr)  # using default
    styler.format_index_names("{:.2f}", **kwargs)
    assert 0 in getattr(styler, attr)  # formatter is specified
    styler.format_index_names(**kwargs)
    assert 0 not in getattr(styler, attr)  # formatter cleared to default


@pytest.mark.parametrize("axis", [0, 1])
def test_format_index_names_callable(styler_multi, axis):
    # [FUNCTION DOC]
    # Verifies that a callable index-names formatter replaces underscores with the
    # letter A in the names of the row or column levels depending on the axis,
    # leaving the other axis names unchanged. Parametrized over axis.
    ctx = styler_multi.format_index_names(
        lambda v: v.replace("_", "A"), axis=axis
    )._translate(True, True)
    result = [
        ctx["head"][2][0]["display_value"],
        ctx["head"][2][1]["display_value"],
        ctx["head"][0][1]["display_value"],
        ctx["head"][1][1]["display_value"],
    ]
    if axis == 0:
        expected = ["0A0", "0A1", "1_0", "1_1"]
    else:
        expected = ["0_0", "0_1", "1A0", "1A1"]
    assert result == expected


def test_format_index_names_dict(styler_multi):
    # [FUNCTION DOC]
    # Verifies dict-based index-names formatting: a dict keyed by a row level name
    # applies a left-aligned fill format and a dict keyed by a column level name with
    # the column axis applies a right-aligned fill format, each only to its named
    # level.
    ctx = (
        styler_multi.format_index_names({"0_0": "{:<<5}"})
        .format_index_names({"1_1": "{:>>4}"}, axis=1)
        ._translate(True, True)
    )
    assert ctx["head"][2][0]["display_value"] == "0_0<<"
    assert ctx["head"][1][1]["display_value"] == ">1_1"


def test_format_index_names_with_hidden_levels(styler_multi):
    # [FUNCTION DOC]
    # Verifies that hiding the second row level and second column level reduces the
    # header height and width by one each (from three by six), and that name
    # formatting (right-aligned fill on column names, left-aligned fill on row names)
    # still shows the formatted names in the remaining header cells.
    ctx = styler_multi._translate(True, True)
    full_head_height = len(ctx["head"])
    full_head_width = len(ctx["head"][0])
    assert full_head_height == 3
    assert full_head_width == 6

    ctx = (
        styler_multi.hide(axis=0, level=1)
        .hide(axis=1, level=1)
        .format_index_names("{:>>4}", axis=1)
        .format_index_names("{:!<5}")
        ._translate(True, True)
    )
    assert len(ctx["head"]) == full_head_height - 1
    assert len(ctx["head"][0]) == full_head_width - 1
    assert ctx["head"][0][0]["display_value"] == ">1_0"
    assert ctx["head"][1][0]["display_value"] == "0_0!!"
