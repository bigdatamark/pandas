"""
Utilities for interpreting CSS from Stylers for formatting non-HTML outputs.
"""

# [MODULE DOC]
# Utilities for interpreting CSS declarations from Styler objects so that non-HTML
#   output (notably Excel and LaTeX conversion) can use atomic CSS 2.2 properties.
# Public contents: CSSResolver (callable class that parses a declaration string or
#   pairs, expands shorthand properties, resolves inheritance and converts sizes to
#   points), plus two private factory functions that build the shorthand expansion
#   callbacks stored in the resolver's expansion table. Problems are reported via the
#   CSS warning category from the errors module.
# Import-time side effects: none (the class body builds lookup tables when the module is
#   imported).

from __future__ import annotations

import re
from typing import TYPE_CHECKING
import warnings

from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
    )


def _side_expander(prop_fmt: str) -> Callable:
    # [FUNCTION DOC]
    # HELPER: factory for CSSResolver shorthand expansion callbacks (used to fill the
    #   expansion table). Input: prop_fmt, a format string with one placeholder for the
    #   side name (for example for margin or per-side border attributes). Returns a
    #   function taking the resolver instance, the property name and the value string,
    #   which is a generator of (property, value) pairs.
    # KNOWN ISSUE: the docstring documents a parameter called side, but the actual
    #   parameter is the format string prop_fmt; the Returns description is also loosely
    #   formatted.
    """
    Wrapper to expand shorthand property into top, right, bottom, left properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """

    def expand(self: CSSResolver, prop: str, value: str) -> Generator[tuple[str, str]]:
        # [FUNCTION DOC]
        # HELPER: nested generator returned by the side expander factory. Inputs: self
        #   (the resolver, used for the side shorthand table and the side names), prop
        #   (the property name, used only in the warning text) and value (the value
        #   string). Splits the value on whitespace; the number of tokens (one to four)
        #   selects, through the shorthand table, which token applies to the sides in
        #   the order top, right, bottom, left, following CSS semantics. Yields one pair
        #   per side with the side name inserted into the format string. If the token
        #   count is not one to four, emits a CSS warning that the property could not be
        #   expanded and yields nothing.
        """
        Expand shorthand property into side-specific property (top, right, bottom, left)

        Parameters
        ----------
            prop (str): CSS property name
            value (str): String token for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        try:
            mapping = self.SIDE_SHORTHANDS[len(tokens)]
        except KeyError:
            warnings.warn(
                f'Could not expand "{prop}: {value}"',
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return
        for key, idx in zip(self.SIDES, mapping, strict=True):
            yield prop_fmt.format(key), tokens[idx]

    return expand


def _border_expander(side: str = "") -> Callable:
    # [FUNCTION DOC]
    # HELPER: factory for border shorthand expanders. Input: side, one of the empty
    #   string (all sides) or a side name; when not empty it is prefixed with a dash.
    #   Returns a generator function for border or per-side border declarations.
    """
    Wrapper to expand 'border' property into border color, style, and width properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """
    if side != "":
        side = f"-{side}"

    def expand(self: CSSResolver, prop: str, value: str) -> Generator[tuple[str, str]]:
        # [FUNCTION DOC]
        # HELPER: nested generator returned by the border expander factory. Inputs: self
        #   (the resolver), prop (property name for the warning text) and value (the
        #   declaration value). Splits into tokens and warns with the CSS warning when
        #   there are zero or more than three tokens. Starts from defaults (color black,
        #   style none, width medium), then classifies each token: a known border style
        #   name sets the style; a token whose lower cased form contains any key of the
        #   border width table sets the width; anything else sets the color. Yields the
        #   atomized pairs of the resulting declarations, so a border shorthand resets
        #   earlier longhand declarations.
        # KNOWN ISSUE: the warning text says too many tokens even when there are no
        #   tokens at all. The width test is a substring test against all unit and
        #   keyword names, so color names containing a unit abbreviation (for example
        #   pink contains the abbreviation for inches) are misclassified as widths.
        """
        Expand border into color, style, and width tuples

        Parameters
        ----------
            prop : str
                CSS property name passed to styler
            value : str
                Value passed to styler for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        if len(tokens) == 0 or len(tokens) > 3:
            warnings.warn(
                f'Too many tokens provided to "{prop}" (expected 1-3)',
                CSSWarning,
                stacklevel=find_stack_level(),
            )

        # TODO: Can we use current color as initial value to comply with CSS standards?
        border_declarations = {
            f"border{side}-color": "black",
            f"border{side}-style": "none",
            f"border{side}-width": "medium",
        }
        for token in tokens:
            if token.lower() in self.BORDER_STYLES:
                border_declarations[f"border{side}-style"] = token
            elif any(ratio in token.lower() for ratio in self.BORDER_WIDTH_RATIOS):
                border_declarations[f"border{side}-width"] = token
            else:
                border_declarations[f"border{side}-color"] = token
            # TODO: Warn user if item entered more than once (e.g. "border: red green")

        # Per CSS, "border" will reset previous "border-*" definitions
        yield from self.atomize(border_declarations.items())

    return expand


class CSSResolver:
    # [CLASS DOC]
    # Callable that resolves CSS declarations to a dictionary of atomic CSS 2.2
    #   properties. Has no instance state; all behavior is driven by class level tables:
    #   unit ratios and the derived tables for font size, margins and border widths
    #   (each mapping a unit to the unit it converts into and a multiplier), the list of
    #   border styles, the shorthand token to side mapping, the side names (top, right,
    #   bottom, left) and the expansion table that maps shorthand properties (border and
    #   its per side and per attribute forms, margin and padding) to expander callbacks.
    #   Key methods: the call operator (entry point), parse (declaration string to
    #   pairs), atomize (expansion), size_to_pt (unit conversion) and the private
    #   resolution steps.
    """
    A callable for parsing and resolving CSS to atomic properties.
    """

    UNIT_RATIOS = {
        "pt": ("pt", 1),
        "em": ("em", 1),
        "rem": ("pt", 12),
        "ex": ("em", 0.5),
        # 'ch':
        "px": ("pt", 0.75),
        "pc": ("pt", 12),
        "in": ("pt", 72),
        "cm": ("in", 1 / 2.54),
        "mm": ("in", 1 / 25.4),
        "q": ("mm", 0.25),
        "!!default": ("em", 0),
    }

    FONT_SIZE_RATIOS = UNIT_RATIOS.copy()
    FONT_SIZE_RATIOS.update(
        {
            "%": ("em", 0.01),
            "xx-small": ("rem", 0.5),
            "x-small": ("rem", 0.625),
            "small": ("rem", 0.8),
            "medium": ("rem", 1),
            "large": ("rem", 1.125),
            "x-large": ("rem", 1.5),
            "xx-large": ("rem", 2),
            "smaller": ("em", 1 / 1.2),
            "larger": ("em", 1.2),
            "!!default": ("em", 1),
        }
    )

    MARGIN_RATIOS = UNIT_RATIOS.copy()
    MARGIN_RATIOS.update({"none": ("pt", 0)})

    BORDER_WIDTH_RATIOS = UNIT_RATIOS.copy()
    BORDER_WIDTH_RATIOS.update(
        {
            "none": ("pt", 0),
            "thick": ("px", 4),
            "medium": ("px", 2),
            "thin": ("px", 1),
            # Default: medium only if solid
        }
    )

    BORDER_STYLES = [
        "none",
        "hidden",
        "dotted",
        "dashed",
        "solid",
        "double",
        "groove",
        "ridge",
        "inset",
        "outset",
        "mediumdashdot",
        "dashdotdot",
        "hair",
        "mediumdashdotdot",
        "dashdot",
        "slantdashdot",
        "mediumdashed",
    ]

    SIDE_SHORTHANDS = {
        1: [0, 0, 0, 0],
        2: [0, 1, 0, 1],
        3: [0, 1, 2, 1],
        4: [0, 1, 2, 3],
    }

    SIDES = ("top", "right", "bottom", "left")

    CSS_EXPANSIONS = {
        **{
            (f"border-{prop}" if prop else "border"): _border_expander(prop)
            for prop in ["", "top", "right", "bottom", "left"]
        },
        **{
            f"border-{prop}": _side_expander(f"border-{{:s}}-{prop}")
            for prop in ["color", "style", "width"]
        },
        "margin": _side_expander("margin-{:s}"),
        "padding": _side_expander("padding-{:s}"),
    }

    def __call__(
        self,
        declarations: str | Iterable[tuple[str, str]],
        inherited: dict[str, str] | None = None,
    ) -> dict[str, str]:
        # [FUNCTION DOC]
        # Resolve declarations into atomic properties. Inputs: declarations (a CSS
        #   string such as semicolon separated name colon value pairs, or an iterable of
        #   property and value pairs) and inherited (optional dictionary of already
        #   resolved properties from the parent context; default none means empty).
        #   Returns a dictionary of atomic properties.
        # Behavior: parses a string into pairs; expands shorthand properties through the
        #   atomize method (later duplicate properties override earlier ones); then
        #   applies, in order, inheritance and initial value resolution, relative font
        #   size resolution against the inherited font size, and conversion of border
        #   widths, margins and paddings to points.
        # KNOWN ISSUE: the docstring names the first parameter declarations_str and its
        #   summary sentence is incomplete, but the actual parameter name is
        #   declarations.
        """
        The given declarations to atomic properties.

        Parameters
        ----------
        declarations_str : str | Iterable[tuple[str, str]]
            A CSS string or set of CSS declaration tuples
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}
        inherited : dict, optional
            Atomic properties indicating the inherited style context in which
            declarations_str is to be resolved. ``inherited`` should already
            be resolved, i.e. valid output of this method.

        Returns
        -------
        dict
            Atomic CSS 2.2 properties.

        Examples
        --------
        >>> resolve = CSSResolver()
        >>> inherited = {"font-family": "serif", "font-weight": "bold"}
        >>> out = resolve(
        ...     '''
        ...               border-color: BLUE RED;
        ...               font-size: 1em;
        ...               font-size: 2em;
        ...               font-weight: normal;
        ...               font-weight: inherit;
        ...               ''',
        ...     inherited,
        ... )
        >>> sorted(out.items())  # doctest: +NORMALIZE_WHITESPACE
        [('border-bottom-color', 'blue'),
         ('border-left-color', 'red'),
         ('border-right-color', 'red'),
         ('border-top-color', 'blue'),
         ('font-family', 'serif'),
         ('font-size', '24pt'),
         ('font-weight', 'bold')]
        """
        if isinstance(declarations, str):
            declarations = self.parse(declarations)
        props = dict(self.atomize(declarations))
        if inherited is None:
            inherited = {}

        props = self._update_initial(props, inherited)
        props = self._update_font_size(props, inherited)
        return self._update_other_units(props)

    def _update_initial(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: step one of resolution. Copies any inherited property that is not
        #   already set into props (mutating it), then builds a new dictionary in which
        #   the value inherit is replaced by the inherited value (or initial when none),
        #   and properties whose value is initial or none are removed since no complete
        #   initial stylesheet exists. Returns the new dictionary.
        # 1. resolve inherited, initial
        for prop, val in inherited.items():
            if prop not in props:
                props[prop] = val

        new_props = props.copy()
        for prop, val in props.items():
            if val == "inherit":
                val = inherited.get(prop, "initial")

            if val in ("initial", None):
                # we do not define a complete initial stylesheet
                del new_props[prop]
            else:
                new_props[prop] = val
        return new_props

    def _update_font_size(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: step two of resolution. If props has a non empty font size, replaces
        #   it (in place) by its size in points, using the inherited font size as the
        #   base for relative units and the font size conversion table. Returns the same
        #   dictionary.
        # 2. resolve relative font size
        if props.get("font-size"):
            props["font-size"] = self.size_to_pt(
                props["font-size"],
                self._get_font_size(inherited),
                conversions=self.FONT_SIZE_RATIOS,
            )
        return props

    def _get_font_size(self, props: dict[str, str]) -> float | None:
        # [FUNCTION DOC]
        # HELPER: returns the font size in points as a float from a resolved property
        #   dictionary, or none when it has no non empty font size entry.
        if props.get("font-size"):
            font_size_string = props["font-size"]
            return self._get_float_font_size_from_pt(font_size_string)
        return None

    def _get_float_font_size_from_pt(self, font_size_string: str) -> float:
        # [FUNCTION DOC]
        # HELPER: converts a font size string that must end in the points unit into a
        #   float. Fails an assertion if the string does not end in the unit. Returns
        #   the number.
        assert font_size_string.endswith("pt")
        return float(font_size_string.rstrip("pt"))

    def _update_other_units(self, props: dict[str, str]) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: step three of resolution. Using the font size of props as the base for
        #   relative units, converts border widths, margins and paddings of each of the
        #   four sides to points, using the border width table for borders and the
        #   margin table for margins and paddings, in place. Returns the same
        #   dictionary.
        font_size = self._get_font_size(props)
        # 3. TODO: resolve other font-relative units
        for side in self.SIDES:
            prop = f"border-{side}-width"
            if prop in props:
                props[prop] = self.size_to_pt(
                    props[prop],
                    em_pt=font_size,
                    conversions=self.BORDER_WIDTH_RATIOS,
                )

            for prop in [f"margin-{side}", f"padding-{side}"]:
                if prop in props:
                    # TODO: support %
                    props[prop] = self.size_to_pt(
                        props[prop],
                        em_pt=font_size,
                        conversions=self.MARGIN_RATIOS,
                    )
        return props

    def size_to_pt(
        self, in_val: str, em_pt: float | None = None, conversions: dict = UNIT_RATIOS
    ) -> str:
        # [FUNCTION DOC]
        # Convert a CSS size string to points. Inputs: in_val (a string such as a number
        #   with a unit or a keyword such as large), em_pt (the size in points of one
        #   em; default none) and conversions (table mapping unit to the next unit and
        #   multiplier; default the general unit table). Returns a string made of the
        #   rounded (five decimals) number and the points unit, with no decimal part
        #   when whole.
        # Behavior: splits the value into a numeric prefix and a unit suffix (a keyword
        #   has an empty prefix, treated as one); then repeatedly converts the unit
        #   through the table until points is reached. An em unit multiplies by em_pt,
        #   or, when em_pt is unknown, switches to the root em unit. An unparsable value
        #   or an unknown unit emits a CSS warning about an unhandled size and returns
        #   the conversion of the default entry of the table.
        def _error() -> str:
            # [FUNCTION DOC]
            # HELPER: nested error handler of the size conversion. Emits a CSS warning
            #   that the size could not be handled and returns the result of converting
            #   the placeholder default size using the same conversion table.
            warnings.warn(
                f"Unhandled size: {in_val!r}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return self.size_to_pt("1!!default", conversions=conversions)

        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
        if match is None:
            return _error()

        val, unit = match.groups()
        if val == "":
            # hack for 'large' etc.
            val = 1
        else:
            try:
                val = float(val)
            except ValueError:
                return _error()

        while unit != "pt":
            if unit == "em":
                if em_pt is None:
                    unit = "rem"
                else:
                    val *= em_pt
                    unit = "pt"
                continue

            try:
                unit, mul = conversions[unit]
            except KeyError:
                return _error()
            val *= mul

        val = round(val, 5)
        if int(val) == val:
            size_fmt = f"{int(val):d}pt"
        else:
            size_fmt = f"{val:f}pt"
        return size_fmt

    def atomize(self, declarations: Iterable) -> Generator[tuple[str, str]]:
        # [FUNCTION DOC]
        # Generator that expands shorthand declarations. Input: iterable of (property,
        #   value) pairs. Lower cases both elements; a property that is in the expansion
        #   table is expanded by its callback, which yields several atomic pairs; any
        #   other pair is yielded unchanged.
        for prop, value in declarations:
            prop = prop.lower()
            value = value.lower()
            if prop in self.CSS_EXPANSIONS:
                expand = self.CSS_EXPANSIONS[prop]
                yield from expand(self, prop, value)
            else:
                yield prop, value

    def parse(self, declarations_str: str) -> Iterator[tuple[str, str]]:
        # [FUNCTION DOC]
        # Generator that parses a declaration string. Input: declarations_str, semicolon
        #   separated declarations. Skips blank declarations; splits each at the first
        #   colon; yields (property, value) pairs with surrounding whitespace removed
        #   and lower cased. A declaration without a colon emits a CSS warning about an
        #   ill-formatted attribute and is skipped.
        # KNOWN ISSUE: values are lower cased unconditionally, so case sensitive parts
        #   of values (such as quoted strings) are altered; the code carries a note
        #   about this.
        """
        Generates (prop, value) pairs from declarations.

        In a future version may generate parsed tokens from tinycss/tinycss2

        Parameters
        ----------
        declarations_str : str
        """
        for decl in declarations_str.split(";"):
            if not decl.strip():
                continue
            prop, sep, val = decl.partition(":")
            prop = prop.strip().lower()
            # TODO: don't lowercase case sensitive parts of values (strings)
            val = val.strip().lower()
            if sep:
                yield prop, val
            else:
                warnings.warn(
                    f"Ill-formatted attribute: expected a colon in {decl!r}",
                    CSSWarning,
                    stacklevel=find_stack_level(),
                )
