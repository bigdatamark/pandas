# [MODULE DOC]
# Tests for the tooltip feature of the table styler: rendering of hover tooltips
# via a hidden span plus generated css, ignoring of blank or unknown entries, the
# css class and properties options, and the title attribute rendering mode
# (including a frame with a hidden index level). Assertions check substrings of
# the html output. The jinja2 skip call runs at import time before the styler
# import.
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    # [FUNCTION DOC]
    # HELPER: fixture returning a three-by-three integer frame (values zero to eight)
    # with columns A, B, C and index x, y, z.
    return DataFrame(
        data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        columns=["A", "B", "C"],
        index=["x", "y", "z"],
    )


@pytest.fixture
def styler(df):
    # [FUNCTION DOC]
    # HELPER: fixture returning a styler over the df fixture with no unique id suffix.
    return Styler(df, uuid_len=0)


@pytest.mark.parametrize(
    "data, columns, index",
    [
        # Test basic reindex and ignoring blank
        ([["Min", "Max"], [np.nan, ""]], ["A", "C"], ["x", "y"]),
        # Test non-referenced columns, reversed col names, short index
        ([["Max", "Min", "Bad-Col"]], ["C", "A", "D"], ["x"]),
    ],
)
def test_tooltip_render(data, columns, index, styler):
    # [FUNCTION DOC]
    # Regression test (GH 21266): a tooltip frame is aligned to the data (reindexing,
    # extra columns dropped, reversed column order handled). Parametrized over a case
    # with a missing value and empty string, and a case with a reversed column order,
    # extra column and short index. Verifies the generated css for the hidden class,
    # the hover rule and the content rule for the Min and Max entries, the span in the
    # cell markup, and that missing, empty and unreferenced entries produce no rule
    # or text.
    ttips = DataFrame(data=data, columns=columns, index=index)

    # GH 21266
    result = styler.set_tooltips(ttips).to_html()

    # test tooltip table level class
    assert "#T_ .pd-t {\n  visibility: hidden;\n" in result

    # test 'Min' tooltip added
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' in result
    assert 'class="data row0 col0" >0<span class="pd-t"></span></td>' in result

    # test 'Max' tooltip added
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' in result
    assert 'class="data row0 col2" >2<span class="pd-t"></span></td>' in result

    # test Nan, empty string and bad column ignored
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "Bad-Col" not in result


def test_tooltip_ignored(styler):
    # [FUNCTION DOC]
    # Regression test (GH 21266): without setting tooltips the html has an empty
    # style element, no tooltip span, and no title attribute.
    # GH 21266
    result = styler.to_html()  # no set_tooltips() creates no <span>
    assert '<style type="text/css">\n</style>' in result
    assert '<span class="pd-t"></span>' not in result
    assert 'title="' not in result


def test_tooltip_css_class(styler):
    # [FUNCTION DOC]
    # Regression test (GH 21266 and GH 39563): a custom css class and properties
    # (given as a list of pairs) appear in the generated css, and that setting tooltips
    # again replaces the previous ones; properties given as a string with repeated
    # declarations are all kept.
    # GH 21266
    result = styler.set_tooltips(
        DataFrame([["tooltip"]], index=["x"], columns=["A"]),
        css_class="other-class",
        props=[("color", "green")],
    ).to_html()
    assert "#T_ .other-class {\n  color: green;\n" in result
    assert '#T_ #T__row0_col0 .other-class::after {\n  content: "tooltip";\n' in result

    # GH 39563
    result = styler.set_tooltips(  # set_tooltips overwrites previous
        DataFrame([["tooltip"]], index=["x"], columns=["A"]),
        css_class="another-class",
        props="color:green;color:red;",
    ).to_html()
    assert "#T_ .another-class {\n  color: green;\n  color: red;\n}" in result


@pytest.mark.parametrize(
    "data, columns, index",
    [
        # Test basic reindex and ignoring blank
        ([["Min", "Max"], [np.nan, ""]], ["A", "C"], ["x", "y"]),
        # Test non-referenced columns, reversed col names, short index
        ([["Max", "Min", "Bad-Col"]], ["C", "A", "D"], ["x"]),
    ],
)
def test_tooltip_render_as_title(data, columns, index, styler):
    # [FUNCTION DOC]
    # Regression test (GH 56605): with the title attribute option, tooltips are
    # rendered as title attributes on the cells and no tooltip css or span is added.
    # Same parametrization as the render test. Cells without a tooltip are rendered
    # without a title.
    ttips = DataFrame(data=data, columns=columns, index=index)
    # GH 56605
    result = styler.set_tooltips(ttips, as_title_attribute=True).to_html()

    # test css not added
    assert "#T_ .pd-t {\n  visibility: hidden;\n" not in result

    # test 'Min' tooltip added as title attribute and css does not exist
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' not in result
    assert 'class="data row0 col0"  title="Min">0</td>' in result

    # test 'Max' tooltip added as title attribute and css does not exist
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' not in result
    assert 'class="data row0 col2"  title="Max">2</td>' in result

    # test Nan, empty string and bad column ignored
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "Bad-Col" not in result
    assert 'class="data row0 col1" >1</td>' in result
    assert 'class="data row1 col0" >3</td>' in result
    assert 'class="data row1 col1" >4</td>' in result
    assert 'class="data row1 col2" >5</td>' in result
    assert 'class="data row2 col0" >6</td>' in result
    assert 'class="data row2 col1" >7</td>' in result
    assert 'class="data row2 col2" >8</td>' in result


def test_tooltip_render_as_title_with_hidden_index_level():
    # [FUNCTION DOC]
    # Regression test (GH 56605): same as the title attribute test but the frame has a
    # three-level named index, the last level hidden (with names) and the tooltip
    # frame has a matching index with a subset of rows; checks title attributes on
    # the right cells and that no tooltip css is produced.
    df = DataFrame(
        data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        columns=["A", "B", "C"],
        index=MultiIndex.from_arrays(
            [["x", "y", "z"], [1, 2, 3], ["aa", "bb", "cc"]],
            names=["alpha", "num", "char"],
        ),
    )
    ttips = DataFrame(
        # Test basic reindex and ignoring blank, and hide level 2 (num) from index
        data=[["Min", "Max"], [np.nan, ""]],
        columns=["A", "C"],
        index=MultiIndex.from_arrays(
            [["x", "y"], [1, 2], ["aa", "bb"]], names=["alpha", "num", "char"]
        ),
    )
    styler = Styler(df, uuid_len=0)
    styler = styler.hide(axis=0, level=-1, names=True)
    # GH 56605
    result = styler.set_tooltips(ttips, as_title_attribute=True).to_html()

    # test css not added
    assert "#T_ .pd-t {\n  visibility: hidden;\n" not in result

    # test 'Min' tooltip added as title attribute and css does not exist
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' not in result
    assert 'class="data row0 col0"  title="Min">0</td>' in result

    # test 'Max' tooltip added as title attribute and css does not exist
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' not in result
    assert 'class="data row0 col2"  title="Max">2</td>' in result

    # test Nan, empty string and bad column ignored
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "Bad-Col" not in result
    assert 'class="data row0 col1" >1</td>' in result
    assert 'class="data row1 col0" >3</td>' in result
    assert 'class="data row1 col1" >4</td>' in result
    assert 'class="data row1 col2" >5</td>' in result
    assert 'class="data row2 col0" >6</td>' in result
    assert 'class="data row2 col1" >7</td>' in result
    assert 'class="data row2 col2" >8</td>' in result
