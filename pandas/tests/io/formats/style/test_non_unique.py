# [MODULE DOC]
# Tests for table styler behaviour when the frame has duplicate index or column
# labels: formatting by label, errors for label-based apply/map, per-label table
# styles, tooltip and cell-class frames that must have unique labels, hidden
# duplicate columns in the translated context, and latex output. The jinja2 skip
# call runs at import time before the styler import.
from textwrap import dedent

import pytest

from pandas import (
    DataFrame,
    IndexSlice,
)

pytest.importorskip("jinja2")

from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a three-by-three float frame with values one to nine,
    # index labels i, j, j and column labels c, d, d (duplicates on both axes).
    return DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["i", "j", "j"],
        columns=["c", "d", "d"],
        dtype=float,
    )


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture without a unique id
    # suffix.
    return Styler(df, uuid_len=0)


def test_format_non_unique(df):
    # [FUNCTION DOC]
    # Regression test (GH 41269): formatting with a dict keyed by a duplicated column
    # label applies to both duplicate columns (one decimal), leaving column c at
    # default precision; and formatting with a precision and a subset selecting row
    # label j and column label d applies only to the four cells in the duplicated
    # rows and columns. Checks substrings of the rendered html.
    # GH 41269

    # test dict
    html = df.style.format({"d": "{:.1f}"}).to_html()
    for val in ["1.000000<", "4.000000<", "7.000000<"]:
        assert val in html
    for val in ["2.0<", "3.0<", "5.0<", "6.0<", "8.0<", "9.0<"]:
        assert val in html

    # test subset
    html = df.style.format(precision=1, subset=IndexSlice["j", "d"]).to_html()
    for val in ["1.000000<", "4.000000<", "7.000000<", "2.000000<", "3.000000<"]:
        assert val in html
    for val in ["5.0<", "6.0<", "8.0<", "9.0<"]:
        assert val in html


@pytest.mark.parametrize("func", ["apply", "map"])
def test_apply_map_non_unique_raises(df, func):
    # [FUNCTION DOC]
    # Regression test (GH 41269): applying a styling function to a frame with
    # duplicate labels raises a key error saying the apply and map methods are not
    # supported for non-unique labels, when the style is computed. Parametrized over
    # the method name apply (function returning a style per element of each column)
    # and map (function returning one style for a value).
    # GH 41269
    if func == "apply":
        op = lambda s: ["color: red;"] * len(s)
    else:
        op = lambda v: "color: red;"

    with pytest.raises(KeyError, match="`Styler.apply` and `.map` are not"):
        getattr(df.style, func)(op)._compute()


def test_table_styles_dict_non_unique_index(styler):
    # [FUNCTION DOC]
    # Verifies a dict of table styles keyed by the duplicated row label j with column
    # axis argument expands to one style per matching row, using positional row
    # selectors (rows one and two) with the properties parsed into pairs.
    styles = styler.set_table_styles(
        {"j": [{"selector": "td", "props": "a: v;"}]}, axis=1
    ).table_styles
    assert styles == [
        {"selector": "td.row1", "props": [("a", "v")]},
        {"selector": "td.row2", "props": [("a", "v")]},
    ]


def test_table_styles_dict_non_unique_columns(styler):
    # [FUNCTION DOC]
    # Verifies a dict of table styles keyed by the duplicated column label d with row
    # axis argument expands to one style per matching column, using positional column
    # selectors (columns one and two).
    styles = styler.set_table_styles(
        {"d": [{"selector": "td", "props": "a: v;"}]}, axis=0
    ).table_styles
    assert styles == [
        {"selector": "td.col1", "props": [("a", "v")]},
        {"selector": "td.col2", "props": [("a", "v")]},
    ]


def test_tooltips_non_unique_raises(styler):
    # [FUNCTION DOC]
    # Verifies that tooltip frames with unique labels are accepted, and that a
    # tooltip frame with duplicated columns or duplicated index labels raises a key
    # error saying tooltips render only if the frame has unique labels.
    # ttips has unique keys
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "b"])
    styler.set_tooltips(ttips=ttips)  # OK

    # ttips has non-unique columns
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "c"], index=["a", "b"])
    with pytest.raises(KeyError, match="Tooltips render only if `ttips` has unique"):
        styler.set_tooltips(ttips=ttips)

    # ttips has non-unique index
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "a"])
    with pytest.raises(KeyError, match="Tooltips render only if `ttips` has unique"):
        styler.set_tooltips(ttips=ttips)


def test_set_td_classes_non_unique_raises(styler):
    # [FUNCTION DOC]
    # Verifies that a cell-class frame with unique labels is accepted, and that one
    # with duplicated columns or duplicated index labels raises a key error saying
    # classes render only if the frame has unique labels.
    # classes has unique keys
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "b"])
    styler.set_td_classes(classes=classes)  # OK

    # classes has non-unique columns
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "c"], index=["a", "b"])
    with pytest.raises(KeyError, match="Classes render only if `classes` has unique"):
        styler.set_td_classes(classes=classes)

    # classes has non-unique index
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "a"])
    with pytest.raises(KeyError, match="Classes render only if `classes` has unique"):
        styler.set_td_classes(classes=classes)


def test_hide_columns_non_unique(styler):
    # [FUNCTION DOC]
    # Verifies that hiding columns by the duplicated label d in the translated
    # context marks both duplicate header cells and the body cells of those columns
    # as not visible, while column c stays visible; header display values are kept.
    ctx = styler.hide(["d"], axis="columns")._translate(True, True)

    assert ctx["head"][0][1]["display_value"] == "c"
    assert ctx["head"][0][1]["is_visible"] is True

    assert ctx["head"][0][2]["display_value"] == "d"
    assert ctx["head"][0][2]["is_visible"] is False

    assert ctx["head"][0][3]["display_value"] == "d"
    assert ctx["head"][0][3]["is_visible"] is False

    assert ctx["body"][0][1]["is_visible"] is True
    assert ctx["body"][0][2]["is_visible"] is False
    assert ctx["body"][0][3]["is_visible"] is False


def test_latex_non_unique(styler):
    # [FUNCTION DOC]
    # Verifies latex output for the duplicate-label frame equals the expected
    # tabular text with all labels shown and six-decimal floats.
    result = styler.to_latex()
    assert result == dedent(
        """\
        \\begin{tabular}{lrrr}
         & c & d & d \\\\
        i & 1.000000 & 2.000000 & 3.000000 \\\\
        j & 4.000000 & 5.000000 & 6.000000 \\\\
        j & 7.000000 & 8.000000 & 9.000000 \\\\
        \\end{tabular}
    """
    )
