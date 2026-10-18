# [MODULE DOC]
# Tests for the CSS declaration resolver used when converting style strings
# (for spreadsheet export). Covers normalisation of whitespace, case and empty
# declarations, warnings on malformed input, expansion of side shorthands
# (margin, padding, border width, colour and style), the border shorthand
# parsing in any token order, inheritance and precedence rules, and conversion
# of absolute and relative font sizes to points (default base size twelve
# points). Defines two assertion helpers; the rest are parametrized tests.
import pytest

from pandas.errors import CSSWarning

import pandas._testing as tm

from pandas.io.formats.css import CSSResolver


def assert_resolves(css, props, inherited=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the resolver tests.
    # Resolves the CSS text with a fresh resolver, optionally given a mapping of
    # inherited properties (default none), and asserts that the resulting property
    # mapping equals the expected props mapping.
    resolve = CSSResolver()
    actual = resolve(css, inherited=inherited)
    assert props == actual


def assert_same_resolution(css1, css2, inherited=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the resolver tests.
    # Resolves two CSS strings with a fresh resolver and the same optional inherited
    # mapping (default none) and asserts that both resolve to equal property mappings.
    resolve = CSSResolver()
    resolved1 = resolve(css1, inherited=inherited)
    resolved2 = resolve(css2, inherited=inherited)
    assert resolved1 == resolved2


@pytest.mark.parametrize(
    "name,norm,abnorm",
    [
        (
            "whitespace",
            "hello: world; foo: bar",
            " \t hello \t :\n  world \n  ;  \n foo: \tbar\n\n",
        ),
        ("case", "hello: world; foo: bar", "Hello: WORLD; foO: bar"),
        ("empty-decl", "hello: world; foo: bar", "; hello: world;; foo: bar;\n; ;"),
        ("empty-list", "", ";"),
    ],
)
def test_css_parse_normalisation(name, norm, abnorm):
    # [FUNCTION DOC]
    # Verifies that a normalised declaration list and a messy variant resolve
    # identically. Parametrized over cases named whitespace (extra blanks, tabs and
    # newlines), case (mixed case names and values), empty declarations (stray
    # semicolons) and an empty list (single semicolon); the name argument is only a
    # label.
    assert_same_resolution(norm, abnorm)


@pytest.mark.parametrize(
    "invalid_css,remainder,msg",
    [
        # No colon
        ("hello-world", "", "expected a colon"),
        ("border-style: solid; hello-world", "border-style: solid", "expected a colon"),
        (
            "border-style: solid; hello-world; font-weight: bold",
            "border-style: solid; font-weight: bold",
            "expected a colon",
        ),
        # Unclosed string fail
        # Invalid size
        ("font-size: blah", "font-size: 1em", "Unhandled size"),
        ("font-size: 1a2b", "font-size: 1em", "Unhandled size"),
        ("font-size: 1e5pt", "font-size: 1em", "Unhandled size"),
        ("font-size: 1+6pt", "font-size: 1em", "Unhandled size"),
        ("font-size: 1unknownunit", "font-size: 1em", "Unhandled size"),
        ("font-size: 10", "font-size: 1em", "Unhandled size"),
        ("font-size: 10 pt", "font-size: 1em", "Unhandled size"),
        # Too many args
        ("border-top: 1pt solid red green", "border-top: 1pt solid green", "Too many"),
    ],
)
def test_css_parse_invalid(invalid_css, remainder, msg):
    # [FUNCTION DOC]
    # Verifies that invalid declarations emit a CSS warning with a matching message
    # and are dropped or replaced by a default, so the result equals that of the
    # remaining valid text. Parametrized over missing colons (message about an
    # expected colon), invalid font sizes (message about an unhandled size, falling
    # back to one em) and a border shorthand with too many tokens (message about too
    # many, last colour wins).
    with tm.assert_produces_warning(CSSWarning, match=msg):
        assert_same_resolution(invalid_css, remainder)


@pytest.mark.parametrize(
    "shorthand,expansions",
    [
        ("margin", ["margin-top", "margin-right", "margin-bottom", "margin-left"]),
        ("padding", ["padding-top", "padding-right", "padding-bottom", "padding-left"]),
        (
            "border-width",
            [
                "border-top-width",
                "border-right-width",
                "border-bottom-width",
                "border-left-width",
            ],
        ),
        (
            "border-color",
            [
                "border-top-color",
                "border-right-color",
                "border-bottom-color",
                "border-left-color",
            ],
        ),
        (
            "border-style",
            [
                "border-top-style",
                "border-right-style",
                "border-bottom-style",
                "border-left-style",
            ],
        ),
    ],
)
def test_css_side_shorthands(shorthand, expansions):
    # [FUNCTION DOC]
    # Verifies expansion of a four-sided shorthand (margin, padding, border width,
    # border colour, border style) into top, right, bottom and left properties.
    # Checks one, two, three and four value forms follow the usual clockwise
    # repetition rules, and that five values emit a CSS warning about being unable
    # to expand and resolve to an empty mapping.
    top, right, bottom, left = expansions

    assert_resolves(
        f"{shorthand}: 1pt", {top: "1pt", right: "1pt", bottom: "1pt", left: "1pt"}
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt", {top: "1pt", right: "4pt", bottom: "1pt", left: "4pt"}
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt 2pt",
        {top: "1pt", right: "4pt", bottom: "2pt", left: "4pt"},
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt 2pt 0pt",
        {top: "1pt", right: "4pt", bottom: "2pt", left: "0pt"},
    )

    with tm.assert_produces_warning(CSSWarning, match="Could not expand"):
        assert_resolves(f"{shorthand}: 1pt 1pt 1pt 1pt 1pt", {})


@pytest.mark.parametrize(
    "shorthand,sides",
    [
        ("border-top", ["top"]),
        ("border-right", ["right"]),
        ("border-bottom", ["bottom"]),
        ("border-left", ["left"]),
        ("border", ["top", "right", "bottom", "left"]),
    ],
)
def test_css_border_shorthand_sides(shorthand, sides):
    # [FUNCTION DOC]
    # Verifies that a border shorthand with colour, style and width resolves into
    # colour, style and width properties for the sides it names: one side for the
    # per-side shorthands, all four for the plain border shorthand. The inner
    # helper builds the expected mapping.
    def create_border_dict(sides, color=None, style=None, width=None):
        # [FUNCTION DOC]
        # HELPER: builds the expected property mapping for the border test: for each
        # given side, adds colour, style and width entries named after the side, only for
        # the arguments that are given (truthy). Returns the mapping.
        resolved = {}
        for side in sides:
            if color:
                resolved[f"border-{side}-color"] = color
            if style:
                resolved[f"border-{side}-style"] = style
            if width:
                resolved[f"border-{side}-width"] = width
        return resolved

    assert_resolves(
        f"{shorthand}: 1pt red solid", create_border_dict(sides, "red", "solid", "1pt")
    )


@pytest.mark.parametrize(
    "prop, expected",
    [
        ("1pt red solid", ("red", "solid", "1pt")),
        ("red 1pt solid", ("red", "solid", "1pt")),
        ("red solid 1pt", ("red", "solid", "1pt")),
        ("solid 1pt red", ("red", "solid", "1pt")),
        ("red solid", ("red", "solid", "1.500000pt")),
        # Note: color=black is not CSS conforming
        # (See https://drafts.csswg.org/css-backgrounds/#border-shorthands)
        ("1pt solid", ("black", "solid", "1pt")),
        ("1pt red", ("red", "none", "1pt")),
        ("red", ("red", "none", "1.500000pt")),
        ("1pt", ("black", "none", "1pt")),
        ("solid", ("black", "solid", "1.500000pt")),
        # Sizes
        ("1em", ("black", "none", "12pt")),
    ],
)
def test_css_border_shorthands(prop, expected):
    # [FUNCTION DOC]
    # Verifies the border shorthand on the left side with tokens in any order and
    # with missing parts. Missing colour defaults to black, missing style to none and
    # missing width to a medium width of one and a half points; a width of one em
    # resolves to twelve points. Parametrized over the input text and the expected
    # (colour, style, width) tuple.
    color, style, width = expected

    assert_resolves(
        f"border-left: {prop}",
        {
            "border-left-color": color,
            "border-left-style": style,
            "border-left-width": width,
        },
    )


@pytest.mark.parametrize(
    "style,inherited,equiv",
    [
        ("margin: 1px; margin: 2px", "", "margin: 2px"),
        ("margin: 1px", "margin: 2px", "margin: 1px"),
        ("margin: 1px; margin: inherit", "margin: 2px", "margin: 2px"),
        (
            "margin: 1px; margin-top: 2px",
            "",
            "margin-left: 1px; margin-right: 1px; margin-bottom: 1px; margin-top: 2px",
        ),
        ("margin-top: 2px", "margin: 1px", "margin: 1px; margin-top: 2px"),
        ("margin: 1px", "margin-top: 2px", "margin: 1px"),
        (
            "margin: 1px; margin-top: inherit",
            "margin: 2px",
            "margin: 1px; margin-top: 2px",
        ),
    ],
)
def test_css_precedence(style, inherited, equiv):
    # [FUNCTION DOC]
    # Verifies cascading rules: later declarations win, own declarations override
    # inherited ones, the inherit keyword takes the inherited value, and longhand
    # declarations override parts of an earlier shorthand regardless of inherited
    # shorthands. Parametrized over the style text, the inherited text and an
    # equivalent flat text; the style resolved against the resolved inherited
    # properties must equal the equivalent resolved alone.
    resolve = CSSResolver()
    inherited_props = resolve(inherited)
    style_props = resolve(style, inherited=inherited_props)
    equiv_props = resolve(equiv)
    assert style_props == equiv_props


@pytest.mark.parametrize(
    "style,equiv",
    [
        (
            "margin: 1px; margin-top: inherit",
            "margin-bottom: 1px; margin-right: 1px; margin-left: 1px",
        ),
        ("margin-top: inherit", ""),
        ("margin-top: initial", ""),
    ],
)
def test_css_none_absent(style, equiv):
    # [FUNCTION DOC]
    # Verifies that inherit and initial values with nothing to inherit leave the
    # property absent: the result equals that of the same text without it.
    assert_same_resolution(style, equiv)


@pytest.mark.parametrize(
    "size,resolved",
    [
        ("xx-small", "6pt"),
        ("x-small", f"{7.5:f}pt"),
        ("small", f"{9.6:f}pt"),
        ("medium", "12pt"),
        ("large", f"{13.5:f}pt"),
        ("x-large", "18pt"),
        ("xx-large", "24pt"),
        ("8px", "6pt"),
        ("1.25pc", "15pt"),
        (".25in", "18pt"),
        ("02.54cm", "72pt"),
        ("25.4mm", "72pt"),
        ("101.6q", "72pt"),
    ],
)
@pytest.mark.parametrize("relative_to", [None, "16pt"])  # invariant to inherited size
def test_css_absolute_font_size(size, relative_to, resolved):
    # [FUNCTION DOC]
    # Verifies conversion of absolute font sizes (keyword sizes and px, pc, in, cm,
    # mm, q units) to points, independent of the inherited font size. Parametrized
    # over size text and expected points, and over no inherited size versus an
    # inherited size of sixteen points.
    if relative_to is None:
        inherited = None
    else:
        inherited = {"font-size": relative_to}
    assert_resolves(f"font-size: {size}", {"font-size": resolved}, inherited=inherited)


@pytest.mark.parametrize(
    "size,relative_to,resolved",
    [
        ("1em", None, "12pt"),
        ("1.0em", None, "12pt"),
        ("1.25em", None, "15pt"),
        ("1em", "16pt", "16pt"),
        ("1.0em", "16pt", "16pt"),
        ("1.25em", "16pt", "20pt"),
        ("1rem", "16pt", "12pt"),
        ("1.0rem", "16pt", "12pt"),
        ("1.25rem", "16pt", "15pt"),
        ("100%", None, "12pt"),
        ("125%", None, "15pt"),
        ("100%", "16pt", "16pt"),
        ("125%", "16pt", "20pt"),
        ("2ex", None, "12pt"),
        ("2.0ex", None, "12pt"),
        ("2.50ex", None, "15pt"),
        ("inherit", "16pt", "16pt"),
        ("smaller", None, "10pt"),
        ("smaller", "18pt", "15pt"),
        ("larger", None, f"{14.4:f}pt"),
        ("larger", "15pt", "18pt"),
    ],
)
def test_css_relative_font_size(size, relative_to, resolved):
    # [FUNCTION DOC]
    # Verifies conversion of relative font sizes (em, rem, percent, ex, inherit,
    # smaller, larger) against the inherited size, defaulting to twelve points when
    # none is inherited; rem is always relative to the default. Parametrized over
    # size text, optional inherited size and the expected point value.
    if relative_to is None:
        inherited = None
    else:
        inherited = {"font-size": relative_to}
    assert_resolves(f"font-size: {size}", {"font-size": resolved}, inherited=inherited)
