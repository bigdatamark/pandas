"""Tests formatting as writer-agnostic ExcelCells

ExcelFormatter is tested implicitly in pandas/tests/io/excel
"""

# [MODULE DOC]
# Tests for the CSS to Excel style conversion used by the Excel formatter:
# CSSToExcelConverter (maps CSS declarations to a nested dictionary of font,
# fill, border, alignment and number format settings) and CssExcelCell (applies
# cached conversion for a cell). The ExcelFormatter itself is exercised
# indirectly in the excel tests elsewhere. Also checks the named color table and
# its caching behaviour.

import string

import pytest

from pandas.errors import CSSWarning

import pandas._testing as tm

from pandas.io.formats.excel import (
    CssExcelCell,
    CSSToExcelConverter,
)


@pytest.mark.parametrize(
    "css,expected",
    [
        # FONT
        # - name
        ("font-family: foo,bar", {"font": {"name": "foo"}}),
        ('font-family: "foo bar",baz', {"font": {"name": "foo bar"}}),
        ("font-family: foo,\nbar", {"font": {"name": "foo"}}),
        ("font-family: foo, bar,    baz", {"font": {"name": "foo"}}),
        ("font-family: bar, foo", {"font": {"name": "bar"}}),
        ("font-family: 'foo bar', baz", {"font": {"name": "foo bar"}}),
        ("font-family: 'foo \\'bar', baz", {"font": {"name": "foo 'bar"}}),
        ('font-family: "foo \\"bar", baz', {"font": {"name": 'foo "bar'}}),
        ('font-family: "foo ,bar", baz', {"font": {"name": "foo ,bar"}}),
        # - family
        ("font-family: serif", {"font": {"name": "serif", "family": 1}}),
        ("font-family: Serif", {"font": {"name": "serif", "family": 1}}),
        ("font-family: roman, serif", {"font": {"name": "roman", "family": 1}}),
        ("font-family: roman, sans-serif", {"font": {"name": "roman", "family": 2}}),
        ("font-family: roman, sans serif", {"font": {"name": "roman"}}),
        ("font-family: roman, sansserif", {"font": {"name": "roman"}}),
        ("font-family: roman, cursive", {"font": {"name": "roman", "family": 4}}),
        ("font-family: roman, fantasy", {"font": {"name": "roman", "family": 5}}),
        # - size
        ("font-size: 1em", {"font": {"size": 12}}),
        ("font-size: xx-small", {"font": {"size": 6}}),
        ("font-size: x-small", {"font": {"size": 7.5}}),
        ("font-size: small", {"font": {"size": 9.6}}),
        ("font-size: medium", {"font": {"size": 12}}),
        ("font-size: large", {"font": {"size": 13.5}}),
        ("font-size: x-large", {"font": {"size": 18}}),
        ("font-size: xx-large", {"font": {"size": 24}}),
        ("font-size: 50%", {"font": {"size": 6}}),
        # - bold
        ("font-weight: 100", {"font": {"bold": False}}),
        ("font-weight: 200", {"font": {"bold": False}}),
        ("font-weight: 300", {"font": {"bold": False}}),
        ("font-weight: 400", {"font": {"bold": False}}),
        ("font-weight: normal", {"font": {"bold": False}}),
        ("font-weight: lighter", {"font": {"bold": False}}),
        ("font-weight: bold", {"font": {"bold": True}}),
        ("font-weight: bolder", {"font": {"bold": True}}),
        ("font-weight: 700", {"font": {"bold": True}}),
        ("font-weight: 800", {"font": {"bold": True}}),
        ("font-weight: 900", {"font": {"bold": True}}),
        # - italic
        ("font-style: italic", {"font": {"italic": True}}),
        ("font-style: oblique", {"font": {"italic": True}}),
        # - underline
        ("text-decoration: underline", {"font": {"underline": "single"}}),
        ("text-decoration: overline", {}),
        ("text-decoration: none", {}),
        # - strike
        ("text-decoration: line-through", {"font": {"strike": True}}),
        (
            "text-decoration: underline line-through",
            {"font": {"strike": True, "underline": "single"}},
        ),
        (
            "text-decoration: underline; text-decoration: line-through",
            {"font": {"strike": True}},
        ),
        # - color
        ("color: red", {"font": {"color": "FF0000"}}),
        ("color: #ff0000", {"font": {"color": "FF0000"}}),
        ("color: #f0a", {"font": {"color": "FF00AA"}}),
        # - shadow
        ("text-shadow: none", {"font": {"shadow": False}}),
        ("text-shadow: 0px -0em 0px #CCC", {"font": {"shadow": False}}),
        ("text-shadow: 0px -0em 0px #999", {"font": {"shadow": False}}),
        ("text-shadow: 0px -0em 0px", {"font": {"shadow": False}}),
        ("text-shadow: 2px -0em 0px #CCC", {"font": {"shadow": True}}),
        ("text-shadow: 0px -2em 0px #CCC", {"font": {"shadow": True}}),
        ("text-shadow: 0px -0em 2px #CCC", {"font": {"shadow": True}}),
        ("text-shadow: 0px -0em 2px", {"font": {"shadow": True}}),
        ("text-shadow: 0px -2em", {"font": {"shadow": True}}),
        # FILL
        # - color, fillType
        (
            "background-color: red",
            {"fill": {"fgColor": "FF0000", "patternType": "solid"}},
        ),
        (
            "background-color: #ff0000",
            {"fill": {"fgColor": "FF0000", "patternType": "solid"}},
        ),
        (
            "background-color: #f0a",
            {"fill": {"fgColor": "FF00AA", "patternType": "solid"}},
        ),
        # BORDER
        # - style
        (
            "border-style: solid",
            {
                "border": {
                    "top": {"style": "medium"},
                    "bottom": {"style": "medium"},
                    "left": {"style": "medium"},
                    "right": {"style": "medium"},
                }
            },
        ),
        (
            "border-style: solid; border-width: thin",
            {
                "border": {
                    "top": {"style": "thin"},
                    "bottom": {"style": "thin"},
                    "left": {"style": "thin"},
                    "right": {"style": "thin"},
                }
            },
        ),
        (
            "border-top-style: solid; border-top-width: thin",
            {"border": {"top": {"style": "thin"}}},
        ),
        (
            "border-top-style: solid; border-top-width: 1pt",
            {"border": {"top": {"style": "thin"}}},
        ),
        ("border-top-style: solid", {"border": {"top": {"style": "medium"}}}),
        (
            "border-top-style: solid; border-top-width: medium",
            {"border": {"top": {"style": "medium"}}},
        ),
        (
            "border-top-style: solid; border-top-width: 2pt",
            {"border": {"top": {"style": "medium"}}},
        ),
        (
            "border-top-style: solid; border-top-width: thick",
            {"border": {"top": {"style": "thick"}}},
        ),
        (
            "border-top-style: solid; border-top-width: 4pt",
            {"border": {"top": {"style": "thick"}}},
        ),
        (
            "border-top-style: solid; border-top-width: none",
            {"border": {"top": {"style": "none"}}},
        ),
        (
            "border-top-style: solid; border-top-width: 0.000001pt",
            {"border": {"top": {"style": "none"}}},
        ),
        (
            "border-top-style: dotted",
            {"border": {"top": {"style": "mediumDashDotDot"}}},
        ),
        (
            "border-top-style: dotted; border-top-width: thin",
            {"border": {"top": {"style": "dotted"}}},
        ),
        ("border-top-style: dashed", {"border": {"top": {"style": "mediumDashed"}}}),
        (
            "border-top-style: dashed; border-top-width: thin",
            {"border": {"top": {"style": "dashed"}}},
        ),
        ("border-top-style: double", {"border": {"top": {"style": "double"}}}),
        # - color
        (
            "border-style: solid; border-color: #0000ff",
            {
                "border": {
                    "top": {"style": "medium", "color": "0000FF"},
                    "right": {"style": "medium", "color": "0000FF"},
                    "bottom": {"style": "medium", "color": "0000FF"},
                    "left": {"style": "medium", "color": "0000FF"},
                }
            },
        ),
        (
            "border-top-style: double; border-top-color: blue",
            {"border": {"top": {"style": "double", "color": "0000FF"}}},
        ),
        (
            "border-top-style: solid; border-top-color: #06c",
            {"border": {"top": {"style": "medium", "color": "0066CC"}}},
        ),
        (
            "border-top-color: blue",
            {"border": {"top": {"color": "0000FF", "style": "none"}}},
        ),
        (
            "border-top-style: slantDashDot; border-top-color: blue",
            {"border": {"top": {"style": "slantDashDot", "color": "0000FF"}}},
        ),
        # ALIGNMENT
        # - horizontal
        ("text-align: center", {"alignment": {"horizontal": "center"}}),
        ("text-align: left", {"alignment": {"horizontal": "left"}}),
        ("text-align: right", {"alignment": {"horizontal": "right"}}),
        ("text-align: justify", {"alignment": {"horizontal": "justify"}}),
        # - vertical
        ("vertical-align: top", {"alignment": {"vertical": "top"}}),
        ("vertical-align: text-top", {"alignment": {"vertical": "top"}}),
        ("vertical-align: middle", {"alignment": {"vertical": "center"}}),
        ("vertical-align: bottom", {"alignment": {"vertical": "bottom"}}),
        ("vertical-align: text-bottom", {"alignment": {"vertical": "bottom"}}),
        # - wrap_text
        ("white-space: nowrap", {"alignment": {"wrap_text": False}}),
        ("white-space: pre", {"alignment": {"wrap_text": False}}),
        ("white-space: pre-line", {"alignment": {"wrap_text": False}}),
        ("white-space: normal", {"alignment": {"wrap_text": True}}),
        # NUMBER FORMAT
        ("number-format: 0%", {"number_format": {"format_code": "0%"}}),
        (
            "number-format: 0§[Red](0)§-§@;",
            {"number_format": {"format_code": "0;[red](0);-;@"}},  # GH 46152
        ),
    ],
)
def test_css_to_excel(css, expected):
    # [FUNCTION DOC]
    # Large table-driven test, parametrized on a CSS declaration string and the
    # expected style dictionary. Groups: font (family name parsing with quotes,
    # escapes, commas and generic family codes; sizes from keywords, em and
    # percent; bold from numeric and keyword weights; italic; underline versus
    # unsupported overline; strike through; color names and short or long hex;
    # shadow presence), fill (background color gives a solid fill), border (style
    # and width combinations mapping to medium, thin, thick, none, dotted and dashed
    # variants, per-side versus all-sides, color handling, passthrough of
    # spreadsheet-native style names), alignment (horizontal, vertical, wrap from
    # white-space) and number format (including a replacement of a special
    # separator character, GH 46152). Each case converts the CSS and compares the
    # whole result with the expectation.
    convert = CSSToExcelConverter()
    assert expected == convert(css)


def test_css_to_excel_multiple():
    # [FUNCTION DOC]
    # Converts a multi-line CSS block with bold, underline, red color, thin border
    # width, centered and top alignment, and one unknown property. Expects a font
    # entry (bold, single underline, red), borders on all four sides in thin style,
    # alignment for both axes, and nothing for the unknown property.
    convert = CSSToExcelConverter()
    actual = convert(
        """
        font-weight: bold;
        text-decoration: underline;
        color: red;
        border-width: thin;
        text-align: center;
        vertical-align: top;
        unused: something;
    """
    )
    assert {
        "font": {"bold": True, "underline": "single", "color": "FF0000"},
        "border": {
            "top": {"style": "thin"},
            "right": {"style": "thin"},
            "bottom": {"style": "thin"},
            "left": {"style": "thin"},
        },
        "alignment": {"horizontal": "center", "vertical": "top"},
    } == actual


@pytest.mark.parametrize(
    "css",
    [
        "border-top-style: unhandled-border-style",
        "border-style: another-unhandled-style",
    ],
)
def test_css_to_excel_unhandled_border_style_warns(css):
    # [FUNCTION DOC]
    # Parametrized on CSS with an unrecognized border style, either per side or for
    # all sides. Conversion must emit the CSS warning saying the border style format
    # is unhandled.
    """Test that unhandled border styles raise a CSSWarning."""
    convert = CSSToExcelConverter()
    with tm.assert_produces_warning(CSSWarning, match="Unhandled border style format"):
        convert(css)


@pytest.mark.parametrize(
    "css,inherited,expected",
    [
        ("font-weight: bold", "", {"font": {"bold": True}}),
        ("", "font-weight: bold", {"font": {"bold": True}}),
        (
            "font-weight: bold",
            "font-style: italic",
            {"font": {"bold": True, "italic": True}},
        ),
        ("font-style: normal", "font-style: italic", {"font": {"italic": False}}),
        ("font-style: inherit", "", {}),
        (
            "font-style: normal; font-style: inherit",
            "font-style: italic",
            {"font": {"italic": True}},
        ),
    ],
)
def test_css_to_excel_inherited(css, inherited, expected):
    # [FUNCTION DOC]
    # Parametrized on CSS, an inherited CSS string given to the converter
    # constructor, and the expected result. Checks that inherited declarations are
    # used when absent, merged with new ones, overridden by explicit values, and
    # that the inherit keyword resolves to the inherited value (or to nothing if
    # none is inherited).
    convert = CSSToExcelConverter(inherited)
    assert expected == convert(css)


@pytest.mark.parametrize(
    "input_color,output_color",
    (
        list(CSSToExcelConverter.NAMED_COLORS.items())
        + [("#" + rgb, rgb) for rgb in CSSToExcelConverter.NAMED_COLORS.values()]
        + [("#F0F", "FF00FF"), ("#ABC", "AABBCC")]
    ),
)
def test_css_to_excel_good_colors(input_color, output_color):
    # [FUNCTION DOC]
    # See gh-18392. Parametrized on every named color with its hex value, every hex
    # value written with a leading hash, and two short hex forms. The same color is
    # applied to all four border colors, the background and the text color; expects
    # solid fill with that foreground, font color, and per-side border color with
    # style none, and no warning is emitted.
    # see gh-18392
    css = (
        f"border-top-color: {input_color}; "
        f"border-right-color: {input_color}; "
        f"border-bottom-color: {input_color}; "
        f"border-left-color: {input_color}; "
        f"background-color: {input_color}; "
        f"color: {input_color}"
    )

    expected = {}

    expected["fill"] = {"patternType": "solid", "fgColor": output_color}

    expected["font"] = {"color": output_color}

    expected["border"] = {
        k: {"color": output_color, "style": "none"}
        for k in ("top", "right", "bottom", "left")
    }

    with tm.assert_produces_warning(None):
        convert = CSSToExcelConverter()
        assert expected == convert(css)


@pytest.mark.parametrize("input_color", [None, "not-a-color"])
def test_css_to_excel_bad_colors(input_color):
    # [FUNCTION DOC]
    # See gh-18392. Parametrized on a missing value or a non-color word used as color
    # everywhere. Conversion must emit the warning for unhandled color format; the
    # result contains a solid pattern fill without a color only when the value was
    # not missing, and nothing else.
    # see gh-18392
    css = (
        f"border-top-color: {input_color}; "
        f"border-right-color: {input_color}; "
        f"border-bottom-color: {input_color}; "
        f"border-left-color: {input_color}; "
        f"background-color: {input_color}; "
        f"color: {input_color}"
    )

    expected = {}

    if input_color is not None:
        expected["fill"] = {"patternType": "solid"}

    with tm.assert_produces_warning(CSSWarning, match="Unhandled color format"):
        convert = CSSToExcelConverter()
        assert expected == convert(css)


@pytest.mark.parametrize("input_color", ["#", "#1234567"])
def test_css_to_excel_invalid_color_raises(input_color):
    # [FUNCTION DOC]
    # Parametrized on malformed hex colors (a bare hash and a seven-digit value).
    # Conversion must raise a value error saying the color is unexpected.
    """Test that invalid colors raise a ValueError."""
    css = (
        f"border-top-color: {input_color}; "
        f"border-right-color: {input_color}; "
        f"border-bottom-color: {input_color}; "
        f"border-left-color: {input_color}; "
        f"background-color: {input_color}; "
        f"color: {input_color}"
    )

    convert = CSSToExcelConverter()
    with pytest.raises(ValueError, match=f"Unexpected color {input_color}"):
        convert(css)


def tests_css_named_colors_valid():
    # [FUNCTION DOC]
    # Checks every value in the converter's named-color table is a six-character
    # string consisting only of hexadecimal digits in upper case (digits count as
    # valid).
    upper_hexs = set(map(str.upper, string.hexdigits))
    for color in CSSToExcelConverter.NAMED_COLORS.values():
        assert len(color) == 6 and all(c in upper_hexs for c in color)


def test_css_named_colors_from_mpl_present():
    # [FUNCTION DOC]
    # Skipped when matplotlib is missing. Every CSS4 color name in matplotlib's
    # table must be present in the converter's table with the same hex value once
    # the leading hash is removed.
    mpl_colors = pytest.importorskip("matplotlib.colors")

    pd_colors = CSSToExcelConverter.NAMED_COLORS
    for name, color in mpl_colors.CSS4_COLORS.items():
        assert name in pd_colors and pd_colors[name] == color[1:]


@pytest.mark.parametrize(
    "styles,expected",
    [
        ([("color", "green"), ("color", "red")], "color: red;"),
        ([("font-weight", "bold"), ("font-weight", "normal")], "font-weight: normal;"),
        ([("text-align", "center"), ("TEXT-ALIGN", "right")], "text-align: right;"),
    ],
)
def test_css_excel_cell_precedence(styles, expected):
    # [FUNCTION DOC]
    # See GH 47371. Parametrized on lists of style pairs where a property repeats,
    # and the CSS text expected to win. Builds a cell with a converter whose style
    # cache was cleared before and after construction, with those styles at its
    # position, and expects the cell's style to equal the direct conversion of the
    # expected text; later declarations (and property names compared without regard
    # to case) override earlier ones.
    """It applies favors latter declarations over former declarations"""
    # See GH 47371
    converter = CSSToExcelConverter()
    converter._call_cached.cache_clear()
    css_styles = {(0, 0): styles}
    cell = CssExcelCell(
        row=0,
        col=0,
        val="",
        style=None,
        css_styles=css_styles,
        css_row=0,
        css_col=0,
        css_converter=converter,
    )
    converter._call_cached.cache_clear()

    assert cell.style == converter(expected)


@pytest.mark.parametrize(
    "styles,cache_hits,cache_misses",
    [
        ([[("color", "green"), ("color", "red"), ("color", "green")]], 0, 1),
        (
            [
                [("font-weight", "bold")],
                [("font-weight", "normal"), ("font-weight", "bold")],
            ],
            1,
            1,
        ),
        ([[("text-align", "center")], [("TEXT-ALIGN", "center")]], 1, 1),
        (
            [
                [("font-weight", "bold"), ("text-align", "center")],
                [("font-weight", "bold"), ("text-align", "left")],
            ],
            0,
            2,
        ),
        (
            [
                [("font-weight", "bold"), ("text-align", "center")],
                [("font-weight", "bold"), ("text-align", "left")],
                [("font-weight", "bold"), ("text-align", "center")],
            ],
            1,
            2,
        ),
    ],
)
def test_css_excel_cell_cache(styles, cache_hits, cache_misses):
    # [FUNCTION DOC]
    # See GH 47371. Parametrized on a list of style lists and the expected cache
    # hit and miss counts. Clears the converter's cache, creates one cell for each
    # style list position, reads the cache statistics, clears the cache again, and
    # asserts exact hit and miss counts. Identical effective styles (after
    # normalizing case and overridden duplicates) should hit the cache, different
    # ones miss.
    """It caches unique cell styles"""
    # See GH 47371
    converter = CSSToExcelConverter()
    converter._call_cached.cache_clear()

    css_styles = {(0, i): _style for i, _style in enumerate(styles)}
    for css_row, css_col in css_styles:
        CssExcelCell(
            row=0,
            col=0,
            val="",
            style=None,
            css_styles=css_styles,
            css_row=css_row,
            css_col=css_col,
            css_converter=converter,
        )
    cache_info = converter._call_cached.cache_info()
    converter._call_cached.cache_clear()

    assert cache_info.hits == cache_hits
    assert cache_info.misses == cache_misses
