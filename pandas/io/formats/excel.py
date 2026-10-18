"""
Utilities for conversion to writer-agnostic Excel representation.
"""

# [MODULE DOC]
# Conversion of frames (and Styler objects) into a writer independent list of Excel
#   cells, used by the frame and Styler Excel export methods.
# Public contents: ExcelCell (plain cell record), CssExcelCell (cell whose style is
#   computed from Styler CSS), CSSToExcelConverter (translates CSS declarations into the
#   nested style dictionary understood by the Excel writer engines) and ExcelFormatter
#   (walks a frame producing header, index and body cells and hands them to an
#   ExcelWriter).
# Relationships: uses the CSS resolver and warning category from the css module, the
#   named color table from the color data module, and the level length helper from the
#   format module; the writer engines in the excel subpackage consume the cells and
#   style dictionaries (their style translators expect the keys produced here:
#   alignment, border, fill, font and number format).
# Import-time side effects: none beyond importing the top level pandas objects.

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import functools
import itertools
import re
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import numpy as np

from pandas._libs.lib import is_list_like
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
    is_float,
    is_scalar,
)

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    PeriodIndex,
)
import pandas.core.common as com

from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
    CSSResolver,
    CSSWarning,
)
from pandas.io.formats.format import get_level_lengths

if TYPE_CHECKING:
    from pandas._typing import (
        ExcelWriterMergeCells,
        FilePath,
        IndexLabel,
        StorageOptions,
        WriteExcelBuffer,
    )

    from pandas import ExcelWriter


class ExcelCell:
    # [CLASS DOC]
    # Plain record for one cell to write. Uses fixed slots with the fields row, col,
    #   val, style, mergestart and mergeend (the field names are also exposed as a
    #   tuple). Positions are zero based and relative to the start row and column given
    #   when writing. When mergestart and mergeend are both set the cell is merged over
    #   a rectangle from its position to row mergestart and column mergeend (both
    #   inclusive). style is a nested dictionary or none.
    __fields__ = ("row", "col", "val", "style", "mergestart", "mergeend")
    __slots__ = __fields__

    def __init__(
        self,
        row: int,
        col: int,
        val,
        style=None,
        mergestart: int | None = None,
        mergeend: int | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the given values in the slots. Inputs: row, col, val (the cell value,
        #   any object), style (default none), mergestart and mergeend (default none,
        #   meaning not merged).
        self.row = row
        self.col = col
        self.val = val
        self.style = style
        self.mergestart = mergestart
        self.mergeend = mergeend


class CssExcelCell(ExcelCell):
    # [CLASS DOC]
    # ExcelCell subclass that optionally derives its style from computed Styler CSS. It
    #   adds no state; the style given to the base class is replaced by the converted
    #   CSS when css information is available.
    def __init__(
        self,
        row: int,
        col: int,
        val,
        style: dict | None,
        css_styles: dict[tuple[int, int], list[tuple[str, Any]]] | None,
        css_row: int,
        css_col: int,
        css_converter: Callable | None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Create a cell. Inputs: row, col, val and style as for the base class;
        #   css_styles (mapping from a pair of row and column positions in the styler
        #   context to a list of property and value pairs, or none), css_row and css_col
        #   (which entry of that mapping applies), css_converter (callable converting a
        #   set of declarations to a style, or none) and extra keyword arguments
        #   (mergestart, mergeend) passed to the base class. When both css_styles and
        #   css_converter are given, the declarations of the entry are reduced to one
        #   per lower cased property (the last wins), turned into a frozen set so that
        #   the converter can cache on it, and the converter result becomes the cell
        #   style, replacing the style argument; a missing mapping key raises a key
        #   error. Otherwise the style argument is kept.
        if css_styles and css_converter:
            # Use dict to get only one (case-insensitive) declaration per property
            declaration_dict = {
                prop.lower(): val for prop, val in css_styles[css_row, css_col]
            }
            # Convert to frozenset for order-invariant caching
            unique_declarations = frozenset(declaration_dict.items())
            style = css_converter(unique_declarations)

        super().__init__(row=row, col=col, val=val, style=style, **kwargs)


class CSSToExcelConverter:
    # [CLASS DOC]
    # Callable that converts CSS declarations to the nested style dictionary understood
    #   by the writer engines (keys alignment, border, fill, font and number_format).
    #   Supports parts of CSS 2.2 and a little of CSS 3 (text shadow), concentrating on
    #   fonts, background color, borders and alignment. Works in two phases: first
    #   resolving the CSS to atomic properties with the CSS resolver (class attribute
    #   compute_css), then mapping the properties to Excel style properties with the
    #   build methods.
    # State: inherited (resolved properties of an enclosing scope, or none) and a cached
    #   wrapper of the uncached conversion (kept per instance to avoid keeping instances
    #   alive). Class level tables: named colors, vertical alignment map, bold weight
    #   map, italic map, font family map (to numeric Excel family codes) and border
    #   style map (lower case name to Excel name). Most methods could be class methods
    #   but are left as instance methods so users can experiment with extensions.
    """
    A callable for converting CSS declarations to ExcelWriter styles

    Supports parts of CSS 2.2, with minimal CSS 3.0 support (e.g. text-shadow),
    focusing on font styling, backgrounds, borders and alignment.

    Operates by first computing CSS styles in a fairly generic
    way (see :meth:`compute_css`) then determining Excel style
    properties from CSS properties (see :meth:`build_xlstyle`).

    Parameters
    ----------
    inherited : str, optional
        CSS declarations understood to be the containing scope for the
        CSS processed by :meth:`__call__`.
    """

    NAMED_COLORS = CSS4_COLORS

    VERTICAL_MAP = {
        "top": "top",
        "text-top": "top",
        "middle": "center",
        "baseline": "bottom",
        "bottom": "bottom",
        "text-bottom": "bottom",
        # OpenXML also has 'justify', 'distributed'
    }

    BOLD_MAP = {
        "bold": True,
        "bolder": True,
        "600": True,
        "700": True,
        "800": True,
        "900": True,
        "normal": False,
        "lighter": False,
        "100": False,
        "200": False,
        "300": False,
        "400": False,
        "500": False,
    }

    ITALIC_MAP = {
        "normal": False,
        "italic": True,
        "oblique": True,
    }

    FAMILY_MAP = {
        "serif": 1,  # roman
        "sans-serif": 2,  # swiss
        "cursive": 4,  # script
        "fantasy": 5,  # decorative
    }

    BORDER_STYLE_MAP = {
        style.lower(): style
        for style in [
            "dashed",
            "mediumDashDot",
            "dashDotDot",
            "hair",
            "dotted",
            "mediumDashDotDot",
            "double",
            "dashDot",
            "slantDashDot",
            "mediumDashed",
        ]
    }

    # NB: Most of the methods here could be classmethods, as only __init__
    #     and __call__ make use of instance attributes.  We leave them as
    #     instancemethods so that users can easily experiment with extensions
    #     without monkey-patching.
    inherited: dict[str, str] | None

    def __init__(self, inherited: str | None = None) -> None:
        # [FUNCTION DOC]
        # Create the converter. Input: inherited, an optional CSS string for the
        #   enclosing scope, resolved immediately into properties (none if not given).
        #   Also creates the cached call wrapper on the instance.
        if inherited is not None:
            self.inherited = self.compute_css(inherited)
        else:
            self.inherited = None
        # We should avoid cache on the __call__ method.
        # Otherwise once the method __call__ has been called
        # garbage collection no longer deletes the instance.
        self._call_cached = functools.cache(self._call_uncached)

    compute_css = CSSResolver()

    def __call__(
        self, declarations: str | frozenset[tuple[str, str]]
    ) -> dict[str, dict[str, str]]:
        # [FUNCTION DOC]
        # Convert CSS to an Excel style. Input: declarations, a CSS string or a
        #   frozenset of property and value pairs (the latter is hashable, so the result
        #   is cached per distinct input). Returns the nested style dictionary.
        #   Delegates to the cached call wrapper.
        """
        Convert CSS declarations to ExcelWriter style.

        Parameters
        ----------
        declarations : str | frozenset[tuple[str, str]]
            CSS string or set of CSS declaration tuples.
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}

        Returns
        -------
        xlstyle : dict
            A style as interpreted by ExcelWriter when found in
            ExcelCell.style.
        """
        return self._call_cached(declarations)

    def _call_uncached(
        self, declarations: str | frozenset[tuple[str, str]]
    ) -> dict[str, dict[str, str]]:
        # [FUNCTION DOC]
        # HELPER: the uncached conversion. Resolves the declarations against the
        #   inherited properties with the CSS resolver and builds the Excel style from
        #   the resulting properties. Returns the style dictionary.
        properties = self.compute_css(declarations, self.inherited)
        return self.build_xlstyle(properties)

    def build_xlstyle(self, props: Mapping[str, str]) -> dict[str, dict[str, str]]:
        # [FUNCTION DOC]
        # Build the full style dictionary from resolved CSS properties. Input: props
        #   (mapping of atomic CSS property to value). Returns a dictionary with
        #   alignment, border, fill, font and number format entries, from which entries
        #   whose value is none and dictionaries that become empty have been removed
        #   recursively.
        out = {
            "alignment": self.build_alignment(props),
            "border": self.build_border(props),
            "fill": self.build_fill(props),
            "font": self.build_font(props),
            "number_format": self.build_number_format(props),
        }

        # TODO: handle cell width and height: needs support in pandas.io.excel

        def remove_none(d: dict[str, str | None]) -> None:
            # [FUNCTION DOC]
            # HELPER: nested function that removes, in place, every key whose value is
            #   none and every nested dictionary that is or becomes empty, recursing
            #   through nested dictionaries.
            """Remove key where value is None, through nested dicts"""
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, dict):
                    remove_none(v)
                    if not v:
                        del d[k]

        remove_none(out)
        return out

    def build_alignment(self, props: Mapping[str, str]) -> dict[str, bool | str | None]:
        # [FUNCTION DOC]
        # Build the alignment entry from properties. Returns a dictionary with
        #   horizontal (from the text alignment property, unchanged), vertical (mapped
        #   through the vertical alignment helper) and wrap text (from the white space
        #   helper); values may be none when not specified.
        # TODO: text-indent, padding-left -> alignment.indent
        return {
            "horizontal": props.get("text-align"),
            "vertical": self._get_vertical_alignment(props),
            "wrap_text": self._get_is_wrap_text(props),
        }

    def _get_vertical_alignment(self, props: Mapping[str, str]) -> str | None:
        # [FUNCTION DOC]
        # HELPER: maps the vertical alignment property through the vertical alignment
        #   table (top, center or bottom); none when the property is absent or has an
        #   unmapped value.
        vertical_align = props.get("vertical-align")
        if vertical_align:
            return self.VERTICAL_MAP.get(vertical_align)
        return None

    def _get_is_wrap_text(self, props: Mapping[str, str]) -> bool | None:
        # [FUNCTION DOC]
        # HELPER: determines text wrapping from the white space property. None when
        #   absent; false for no wrap, pre and pre-line values; true for others.
        if props.get("white-space") is None:
            return None
        return bool(props["white-space"] not in ("nowrap", "pre", "pre-line"))

    def build_border(
        self, props: Mapping[str, str]
    ) -> dict[str, dict[str, str | None]]:
        # [FUNCTION DOC]
        # Build the border entry. Returns a dictionary keyed by the four sides (top,
        #   right, bottom, left); each value has a style (derived from the side's style,
        #   width and color via the border style helper) and a color (converted to an
        #   Excel color string).
        # KNOWN ISSUE: the color of each side is converted twice (once for the style
        #   computation and once for the result), so an unrecognized color name triggers
        #   the unhandled color warning twice.
        return {
            side: {
                "style": self._border_style(
                    props.get(f"border-{side}-style"),
                    props.get(f"border-{side}-width"),
                    self.color_to_excel(props.get(f"border-{side}-color")),
                ),
                "color": self.color_to_excel(props.get(f"border-{side}-color")),
            }
            for side in ["top", "right", "bottom", "left"]
        }

    def _border_style(
        self, style: str | None, width: str | None, color: str | None
    ) -> str | None:
        # [FUNCTION DOC]
        # HELPER: converts a CSS border style, width and color into an Excel border
        #   style name. If neither style nor width is given: none when there is no color
        #   (so the border is dropped) or the name for no border when a color is present
        #   (so the border entry is kept). The styles none and hidden give no border. A
        #   width that rounds to zero gives no border. Solid, groove, ridge, inset,
        #   outset and unspecified style give the width name (thin, medium or thick; the
        #   unsupported styles are not distinguished). Double gives double. Dotted gives
        #   dotted for hair and thin widths, else medium dash dot dot. Dashed gives
        #   dashed for hair and thin widths, else medium dashed. Excel specific style
        #   names pass through the map. Anything else emits a CSS warning about an
        #   unhandled border style and gives no border.
        # convert styles and widths to openxml, one of:
        #       'dashDot'
        #       'dashDotDot'
        #       'dashed'
        #       'dotted'
        #       'double'
        #       'hair'
        #       'medium'
        #       'mediumDashDot'
        #       'mediumDashDotDot'
        #       'mediumDashed'
        #       'slantDashDot'
        #       'thick'
        #       'thin'
        if width is None and style is None:
            if color is None:
                # Return None will remove "border" from style dictionary
                return None
            else:
                # Return "none" will keep "border" in style dictionary
                return "none"

        if style in ("none", "hidden"):
            return "none"

        width_name = self._get_width_name(width)
        if width_name is None:
            return "none"

        if style in (None, "groove", "ridge", "inset", "outset", "solid"):
            # not handled
            return width_name

        if style == "double":
            return "double"
        if style == "dotted":
            if width_name in ("hair", "thin"):
                return "dotted"
            return "mediumDashDotDot"
        if style == "dashed":
            if width_name in ("hair", "thin"):
                return "dashed"
            return "mediumDashed"
        elif style in self.BORDER_STYLE_MAP:
            # Excel-specific styles
            return self.BORDER_STYLE_MAP[style]
        else:
            warnings.warn(
                f"Unhandled border style format: {style!r}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return "none"

    def _get_width_name(self, width_input: str | None) -> str | None:
        # [FUNCTION DOC]
        # HELPER: converts a CSS width to a name: none when it is effectively zero
        #   (below one hundred thousandth of a point), thin below 1.3 points, medium
        #   below 2.8 points, otherwise thick.
        width = self._width_to_float(width_input)
        if width < 1e-5:
            return None
        elif width < 1.3:
            return "thin"
        elif width < 2.8:
            return "medium"
        return "thick"

    def _width_to_float(self, width: str | None) -> float:
        # [FUNCTION DOC]
        # HELPER: converts a width string in points to a number, using two points when
        #   the width is absent.
        if width is None:
            width = "2pt"
        return self._pt_to_float(width)

    def _pt_to_float(self, pt_string: str) -> float:
        # [FUNCTION DOC]
        # HELPER: converts a string ending in the points unit to a float. Fails an
        #   assertion if the unit is missing.
        assert pt_string.endswith("pt")
        return float(pt_string.rstrip("pt"))

    def build_fill(self, props: Mapping[str, str]):
        # [FUNCTION DOC]
        # Build the fill entry. Returns a dictionary with the foreground color
        #   (converted from the background color property) and a solid pattern type, or
        #   nothing (none) when the background color is absent, transparent or none.
        #   Other fill patterns are not supported.
        # TODO: perhaps allow for special properties
        #       -excel-pattern-bgcolor and -excel-pattern-type
        fill_color = props.get("background-color")
        if fill_color not in (None, "transparent", "none"):
            return {"fgColor": self.color_to_excel(fill_color), "patternType": "solid"}

    def build_number_format(self, props: Mapping[str, str]) -> dict[str, str | None]:
        # [FUNCTION DOC]
        # Build the number format entry from the custom number format property (not
        #   standard CSS). Returns a dictionary with the format code, in which the
        #   section sign character is replaced by a semicolon (since semicolons separate
        #   CSS declarations); the code is none when absent.
        fc = props.get("number-format")
        fc = fc.replace("§", ";") if isinstance(fc, str) else fc
        return {"format_code": fc}

    def build_font(
        self, props: Mapping[str, str]
    ) -> dict[str, bool | float | str | None]:
        # [FUNCTION DOC]
        # Build the font entry. Returns a dictionary with the first listed font name,
        #   the numeric family selected from the font family list, the size in points,
        #   bold, italic, underline (single when the decoration contains underline),
        #   strike (true when it contains line-through, else none), color (converted)
        #   and shadow (from the text shadow property). Entries not specified are none.
        font_names = self._get_font_names(props)
        decoration = self._get_decoration(props)
        return {
            "name": font_names[0] if font_names else None,
            "family": self._select_font_family(font_names),
            "size": self._get_font_size(props),
            "bold": self._get_is_bold(props),
            "italic": self._get_is_italic(props),
            "underline": ("single" if "underline" in decoration else None),
            "strike": ("line-through" in decoration) or None,
            "color": self.color_to_excel(props.get("color")),
            # shadow if nonzero digit before shadow color
            "shadow": self._get_shadow(props),
        }

    def _get_is_bold(self, props: Mapping[str, str]) -> bool | None:
        # [FUNCTION DOC]
        # HELPER: maps the font weight property through the bold table (true for bold
        #   weights, false for normal ones); none if absent or unmapped.
        weight = props.get("font-weight")
        if weight:
            return self.BOLD_MAP.get(weight)
        return None

    def _get_is_italic(self, props: Mapping[str, str]) -> bool | None:
        # [FUNCTION DOC]
        # HELPER: maps the font style property through the italic table (true for italic
        #   or oblique, false for normal); none if absent or unmapped.
        font_style = props.get("font-style")
        if font_style:
            return self.ITALIC_MAP.get(font_style)
        return None

    def _get_decoration(self, props: Mapping[str, str]) -> Sequence[str]:
        # [FUNCTION DOC]
        # HELPER: returns the whitespace separated words of the text decoration
        #   property, or an empty tuple when absent.
        decoration = props.get("text-decoration")
        if decoration is not None:
            return decoration.split()
        else:
            return ()

    def _get_shadow(self, props: Mapping[str, str]) -> bool | None:
        # [FUNCTION DOC]
        # HELPER: determines shadow from the text shadow property: true when a nonzero
        #   digit appears before any hash sign or opening parenthesis (that is, before
        #   the color), none when the property is absent.
        if "text-shadow" in props:
            return bool(re.search("^[^#(]*[1-9]", props["text-shadow"]))
        return None

    def _get_font_names(self, props: Mapping[str, str]) -> Sequence[str]:
        # [FUNCTION DOC]
        # HELPER: parses the font family property into a list of family names, splitting
        #   on commas outside quotes and stripping matching single or double quotes
        #   (unescaping escaped quotes). Blank names are dropped. Returns an empty list
        #   when the property is absent.
        font_names_tmp = re.findall(
            r"""(?x)
            (
            "(?:[^"]|\\")+"
            |
            '(?:[^']|\\')+'
            |
            [^'",]+
            )(?=,|\s*$)
        """,
            props.get("font-family", ""),
        )

        font_names = []
        for name in font_names_tmp:
            if name[:1] == '"':
                name = name[1:-1].replace('\\"', '"')
            elif name[:1] == "'":
                name = name[1:-1].replace("\\'", "'")
            else:
                name = name.strip()
            if name:
                font_names.append(name)
        return font_names

    def _get_font_size(self, props: Mapping[str, str]) -> float | None:
        # [FUNCTION DOC]
        # HELPER: returns the font size property converted to a float number of points,
        #   or none when absent.
        size = props.get("font-size")
        if size is None:
            return size
        return self._pt_to_float(size)

    def _select_font_family(self, font_names: Sequence[str]) -> int | None:
        # [FUNCTION DOC]
        # HELPER: returns the numeric Excel family code of the first listed name that is
        #   a generic family (serif, sans serif, cursive or fantasy), or none if there
        #   is none.
        family = None
        for name in font_names:
            family = self.FAMILY_MAP.get(name)
            if family:
                break

        return family

    def color_to_excel(self, val: str | None) -> str | None:
        # [FUNCTION DOC]
        # Converts a CSS color to an Excel color string. Input: val, a color (hex with a
        #   leading hash, or a named color) or none. None gives none. Hex colors are
        #   converted (three digit shorthand expanded, upper cased). Named colors are
        #   looked up in the table (case sensitive, expected already lower case from the
        #   resolver). An unknown name emits a CSS warning about an unhandled color
        #   format and gives none.
        if val is None:
            return None

        if self._is_hex_color(val):
            return self._convert_hex_to_excel(val)

        try:
            return self.NAMED_COLORS[val]
        except KeyError:
            warnings.warn(
                f"Unhandled color format: {val!r}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
        return None

    def _is_hex_color(self, color_string: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: true when the color text starts with a hash sign.
        return bool(color_string.startswith("#"))

    def _convert_hex_to_excel(self, color_string: str) -> str:
        # [FUNCTION DOC]
        # HELPER: converts a hex color string to an upper case six digit string without
        #   the hash sign, expanding the three digit shorthand by doubling each digit.
        #   Raises a value error (from the shorthand check) for any other length.
        code = color_string.lstrip("#")
        if self._is_shorthand_color(color_string):
            return (code[0] * 2 + code[1] * 2 + code[2] * 2).upper()
        else:
            return code.upper()

    def _is_shorthand_color(self, color_string: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true for a three digit color code, false for a six digit one,
        #   and raises a value error for any other length.
        """Check if color code is shorthand.

        #FFF is a shorthand as opposed to full #FFFFFF.
        """
        code = color_string.lstrip("#")
        if len(code) == 3:
            return True
        elif len(code) == 6:
            return False
        else:
            raise ValueError(f"Unexpected color {color_string}")


class ExcelFormatter:
    # [CLASS DOC]
    # Builds the list of cells to write for a frame or Styler. Constructed from the data
    #   and the output options; the cell generation is lazy and driven by
    #   get_formatted_cells, and the write method sends the cells to an Excel writer.
    # State: rowcounter (the next row to write, advanced while header and index rows are
    #   generated; the lazy generator ordering matters because the body generators read
    #   the counter after the header generator has been consumed), na_rep, float_format,
    #   index, index_label, header (a boolean or aliases), merge_cells (boolean or the
    #   string columns), inf_rep, autofilter, df (the data, reindexed to the requested
    #   columns), columns, styler (the Styler or none) and style_converter (the CSS
    #   converter, only when a Styler was given). Class constants max_rows (about one
    #   million) and max_cols (sixteen thousand) are Excel sheet limits.
    # KNOWN ISSUE: the docstring says the style converter is only called for body cells,
    #   but the converter is also passed to the header and index cell generators, so it
    #   is applied to column header, row index and body cells whenever the Styler has
    #   style information for them.
    """
    Class for formatting a DataFrame to a list of ExcelCells,

    Parameters
    ----------
    df : DataFrame or Styler
    na_rep: na representation
    float_format : str, default None
        Format string for floating point numbers
    cols : sequence, optional
        Columns to write
    header : bool or sequence of str, default True
        Write out column names. If a list of string is given it is
        assumed to be aliases for the column names
    index : bool, default True
        output row names (index)
    index_label : str or sequence, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.
    merge_cells : bool or 'columns', default False
        Format MultiIndex column headers and Hierarchical Rows as merged cells
        if True. Merge MultiIndex column headers only if 'columns'.
        .. versionchanged:: 3.0.0
            Added the 'columns' option.
    inf_rep : str, default `'inf'`
        representation for np.inf values (which aren't representable in Excel)
        A `'-'` sign will be added in front of -inf.
    style_converter : callable, optional
        This translates Styler styles (CSS) into ExcelWriter styles.
        Defaults to ``CSSToExcelConverter()``.
        It should have signature css_declarations string -> excel style.
        This is only called for body cells.
    autofilter : bool, default False
        If True, add automatic filters to all columns.
    """

    max_rows = 2**20
    max_cols = 2**14

    def __init__(
        self,
        df,
        na_rep: str = "",
        float_format: str | None = None,
        cols: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        merge_cells: ExcelWriterMergeCells = False,
        inf_rep: str = "inf",
        style_converter: Callable | None = None,
        autofilter: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Initialize the formatter. Inputs: df (a frame or a Styler), na_rep (text for
        #   missing values; default empty), float_format (percent style format for
        #   floats; default none), cols (optional columns to write), header (boolean, or
        #   sequence of aliases; default true), index (default true), index_label
        #   (default none), merge_cells (false, true or the string columns; default
        #   false), inf_rep (text for infinity; default inf), style_converter (default
        #   none) and autofilter (default false).
        # Behavior: for a Styler, computes its applied styles, takes the data, and uses
        #   the given converter or a new CSS converter; for a frame there is no styler
        #   and no converter. When cols is given, raises a key error if none of them is
        #   among the columns, or if not all distinct ones are, then reindexes to those
        #   columns. Raises a value error when merge_cells is neither a boolean nor the
        #   string columns.
        # Side effects: calls the Styler's compute step.
        self.rowcounter = 0
        self.na_rep = na_rep
        if not isinstance(df, DataFrame):
            self.styler = df
            self.styler._compute()  # calculate applied styles
            df = df.data
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            self.style_converter: Callable | None = style_converter
        else:
            self.styler = None
            self.style_converter = None
        self.df = df
        if cols is not None:
            # all missing, raise
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError("passes columns are not ALL present dataframe")

            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                # Deprecated in GH#17295, enforced in 1.0.0
                raise KeyError("Not all names specified in 'columns' are found")

            self.df = df.reindex(columns=cols)

        self.columns = self.df.columns
        self.float_format = float_format
        self.index = index
        self.index_label = index_label
        self.header = header

        if not isinstance(merge_cells, bool) and merge_cells != "columns":
            raise ValueError(f"Unexpected value for {merge_cells=}.")
        self.merge_cells = merge_cells
        self.inf_rep = inf_rep
        self.autofilter = autofilter

    def _format_value(self, val):
        # [FUNCTION DOC]
        # HELPER: formats one cell value. Missing scalars become the na representation.
        #   Floats: positive infinity becomes the infinity text, negative infinity the
        #   same text prefixed with a minus sign, and when a float format is set other
        #   floats are formatted with it and converted back to float. A value with
        #   timezone information raises a value error because Excel cannot store
        #   timezones. Returns the value.
        if is_scalar(val) and missing.isna(val):
            val = self.na_rep
        elif is_float(val):
            if missing.isposinf_scalar(val):
                val = self.inf_rep
            elif missing.isneginf_scalar(val):
                val = f"-{self.inf_rep}"
            elif self.float_format is not None:
                val = float(self.float_format % val)
        if getattr(val, "tzinfo", None) is not None:
            raise ValueError(
                "Excel does not support datetimes with "
                "timezones. Please ensure that datetimes "
                "are timezone unaware before writing to Excel."
            )
        return val

    def _format_header_mi(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: generator of header cells for multi level columns. Raises a not
        #   implemented error when there are several column levels but the index is not
        #   written. Produces nothing when headers are disabled. Otherwise yields one
        #   cell per column level name in the column just before the data, then, per
        #   level, one cell per label span at the right offset (taking the index width
        #   into account), with merge information when merging of columns is enabled and
        #   the span covers several columns; styles come from the styler column context.
        #   Finally sets the row counter to the number of the last level row.
        if self.columns.nlevels > 1:
            if not self.index:
                raise NotImplementedError(
                    "Writing to Excel with MultiIndex columns and no "
                    "index ('index'=False) is not yet implemented."
                )

        if not (self._has_aliases or self.header):
            return

        columns = self.columns
        merge_columns = self.merge_cells in {True, "columns"}
        level_strs = columns._format_multi(sparsify=merge_columns, include_names=False)
        level_lengths = get_level_lengths(level_strs)
        coloffset = 0
        lnum = 0

        if self.index and isinstance(self.df.index, MultiIndex):
            coloffset = self.df.index.nlevels - 1

        for lnum, name in enumerate(columns.names):
            yield ExcelCell(
                row=lnum,
                col=coloffset,
                val=name,
                style=None,
            )

        for lnum, (spans, levels, level_codes) in enumerate(
            zip(level_lengths, columns.levels, columns.codes, strict=True)
        ):
            values = levels.take(level_codes)
            for i, span_val in spans.items():
                mergestart, mergeend = None, None
                if merge_columns and span_val > 1:
                    mergestart, mergeend = lnum, coloffset + i + span_val
                yield CssExcelCell(
                    row=lnum,
                    col=coloffset + i + 1,
                    val=values[i],
                    style=None,
                    css_styles=getattr(self.styler, "ctx_columns", None),
                    css_row=lnum,
                    css_col=i,
                    css_converter=self.style_converter,
                    mergestart=mergestart,
                    mergeend=mergeend,
                )
        self.rowcounter = lnum

    def _format_header_regular(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: generator of header cells for a flat column index. Produces nothing
        #   when headers are disabled. The column offset is one when the index is
        #   written (the number of index levels for a multi level index). Uses aliases
        #   instead of the column labels if given, raising a value error if their count
        #   differs from the number of columns. Yields one cell per column in the
        #   current row.
        if self._has_aliases or self.header:
            coloffset = 0

            if self.index:
                coloffset = 1
                if isinstance(self.df.index, MultiIndex):
                    coloffset = len(self.df.index.names)

            colnames = self.columns
            if self._has_aliases:
                self.header = cast(Sequence, self.header)
                if len(self.header) != len(self.columns):
                    raise ValueError(
                        f"Writing {len(self.columns)} cols "
                        f"but got {len(self.header)} aliases"
                    )
                colnames = self.header

            for colindex, colname in enumerate(colnames):
                yield CssExcelCell(
                    row=self.rowcounter,
                    col=colindex + coloffset,
                    val=colname,
                    style=None,
                    css_styles=getattr(self.styler, "ctx_columns", None),
                    css_row=0,
                    css_col=colindex,
                    css_converter=self.style_converter,
                )

    def _format_header(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: builds the header cells: those for multi level or flat columns,
        #   chained with a row of index names when applicable. Returns an iterator.
        # KNOWN ISSUE: the index name row is only produced if every element of the row
        #   is non empty, but the row always includes one empty string for each column;
        #   as written the extra row therefore appears only for frames without columns
        #   (when all index names are set), which looks unintended.
        gen: Iterable[ExcelCell]

        if isinstance(self.columns, MultiIndex):
            gen = self._format_header_mi()
        else:
            gen = self._format_header_regular()

        gen2: Iterable[ExcelCell] = ()

        if self.df.index.names:
            row = [x if x is not None else "" for x in self.df.index.names] + [
                ""
            ] * len(self.columns)
            if functools.reduce(lambda x, y: x and y, (x != "" for x in row)):
                gen2 = (
                    ExcelCell(self.rowcounter, colindex, val, None)
                    for colindex, val in enumerate(row)
                )
                self.rowcounter += 1
        return itertools.chain(gen, gen2)

    def _format_body(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: dispatches to the hierarchical rows generator for a multi level row
        #   index and to the regular generator otherwise. Returns an iterator of cells.
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()
        else:
            return self._format_regular_rows()

    def _format_regular_rows(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: generator of index and body cells for a flat index. Advances the row
        #   counter past the header when headers are written. When the index is written:
        #   picks the index label (first element of a given list, the given string, or
        #   the index name), advances the row counter once more for multi level columns,
        #   writes the label in the row above the data unless headers are disabled, and
        #   writes the index values (period indexes converted to timestamps) in the
        #   first column with styler index context. Then yields the body cells with the
        #   right column offset.
        if self._has_aliases or self.header:
            self.rowcounter += 1

        # output index and index_label?
        if self.index:
            # check aliases
            # if list only take first as this is not a MultiIndex
            if self.index_label and isinstance(
                self.index_label, (list, tuple, np.ndarray, Index)
            ):
                index_label = self.index_label[0]
            # if string good to go
            elif self.index_label and isinstance(self.index_label, str):
                index_label = self.index_label
            else:
                index_label = self.df.index.names[0]

            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1

            if index_label and self.header is not False:
                yield ExcelCell(self.rowcounter - 1, 0, index_label, None)

            # write index_values
            index_values = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values = self.df.index.to_timestamp()

            for idx, idxval in enumerate(index_values):
                yield CssExcelCell(
                    row=self.rowcounter + idx,
                    col=0,
                    val=idxval,
                    style=None,
                    css_styles=getattr(self.styler, "ctx_index", None),
                    css_row=idx,
                    css_col=0,
                    css_converter=self.style_converter,
                )
            coloffset = 1
        else:
            coloffset = 0

        yield from self._generate_body(coloffset)

    def _format_hierarchical_rows(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: generator of index and body cells for a multi level index. Advances
        #   the row counter past the header, picks index labels (a given sequence or the
        #   index names), advances once more for multi level columns, and writes the
        #   labels in the row above the data if any is set and headers are enabled. With
        #   merged cells enabled (and not columns only), writes each level's labels once
        #   per span with merge information covering the span; otherwise writes every
        #   label of every level for every row. Period values are converted to
        #   timestamps. Finally yields the body cells after the index columns.
        if self._has_aliases or self.header:
            self.rowcounter += 1

        gcolidx = 0

        if self.index:
            index_labels = self.df.index.names
            # check for aliases
            if self.index_label and isinstance(
                self.index_label, (list, tuple, np.ndarray, Index)
            ):
                index_labels = self.index_label

            # MultiIndex columns require an extra row
            # with index names (blank if None) for
            # unambiguous round-trip, Issue #11328
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1

            # if index labels are not empty go ahead and dump
            if com.any_not_none(*index_labels) and self.header is not False:
                for cidx, name in enumerate(index_labels):
                    yield ExcelCell(self.rowcounter - 1, cidx, name, None)

            if self.merge_cells and self.merge_cells != "columns":
                # Format hierarchical rows as merged cells.
                level_strs = self.df.index._format_multi(
                    sparsify=True, include_names=False
                )
                level_lengths = get_level_lengths(level_strs)

                for spans, levels, level_codes in zip(
                    level_lengths,
                    self.df.index.levels,
                    self.df.index.codes,
                    strict=False,
                ):
                    values = levels.take(
                        level_codes,
                        allow_fill=levels._can_hold_na,
                        fill_value=levels._na_value,
                    )
                    # GH#60099
                    if isinstance(values[0], Period):
                        values = values.to_timestamp()

                    for i, span_val in spans.items():
                        mergestart, mergeend = None, None
                        if span_val > 1:
                            mergestart = self.rowcounter + i + span_val - 1
                            mergeend = gcolidx
                        yield CssExcelCell(
                            row=self.rowcounter + i,
                            col=gcolidx,
                            val=values[i],
                            style=None,
                            css_styles=getattr(self.styler, "ctx_index", None),
                            css_row=i,
                            css_col=gcolidx,
                            css_converter=self.style_converter,
                            mergestart=mergestart,
                            mergeend=mergeend,
                        )
                    gcolidx += 1

            else:
                # Format hierarchical rows with non-merged values.
                for indexcolvals in zip(*self.df.index, strict=True):
                    for idx, indexcolval in enumerate(indexcolvals):
                        # GH#60099
                        if isinstance(indexcolval, Period):
                            indexcolval = indexcolval.to_timestamp()

                        yield CssExcelCell(
                            row=self.rowcounter + idx,
                            col=gcolidx,
                            val=indexcolval,
                            style=None,
                            css_styles=getattr(self.styler, "ctx_index", None),
                            css_row=idx,
                            css_col=gcolidx,
                            css_converter=self.style_converter,
                        )
                    gcolidx += 1

        yield from self._generate_body(gcolidx)

    @property
    def _has_aliases(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property that is true if the header option is a list like set of
        #   column aliases.
        """Whether the aliases for column names are present."""
        return is_list_like(self.header)

    def _generate_body(self, coloffset: int) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # HELPER: generator of body cells. Input: coloffset, the number of columns used
        #   by the index. Yields cells series by series (column major), each at the
        #   current row counter plus the row position, with the styler body context.
        # Write the body of the frame data series by series.
        for colidx in range(len(self.columns)):
            series = self.df.iloc[:, colidx]
            for i, val in enumerate(series):
                yield CssExcelCell(
                    row=self.rowcounter + i,
                    col=colidx + coloffset,
                    val=val,
                    style=None,
                    css_styles=getattr(self.styler, "ctx", None),
                    css_row=i,
                    css_col=colidx,
                    css_converter=self.style_converter,
                )

    def get_formatted_cells(self) -> Iterable[ExcelCell]:
        # [FUNCTION DOC]
        # Generator of all cells ready for writing: chains the header and body cells and
        #   replaces each cell value by its formatted value. Yields cells.
        for cell in itertools.chain(self._format_header(), self._format_body()):
            cell.val = self._format_value(cell.val)
            yield cell

    def _num2excel(self, index: int) -> str:
        # [FUNCTION DOC]
        # HELPER: converts a zero based column index to an Excel column name using
        #   letters. Raises a value error if the index is negative.
        # KNOWN ISSUE: the conversion divides by twenty six with remainder rather than
        #   using the bijective numbering Excel uses, so for indexes of twenty six and
        #   above it gives wrong names (for example index twenty six gives BA rather
        #   than AA). It only affects autofilter ranges that extend beyond column Z.
        """
        Convert 0-based column index to Excel column name.

        Parameters
        ----------
        index : int
            The numeric column index to convert to an Excel column name.

        Returns
        -------
        column_name : str
            The column name corresponding to the index.

        Raises
        ------
        ValueError
            Index is negative
        """
        if index < 0:
            raise ValueError(f"Index cannot be negative: {index}")
        column_name = ""
        # while loop in case column name needs to be longer than 1 character
        while index > 0 or not column_name:
            index, remainder = divmod(index, 26)
            column_name = chr(65 + remainder) + column_name
        return column_name

    def write(
        self,
        writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = "Sheet1",
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
        engine: str | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Write the formatted cells to an Excel destination. Inputs: writer (path,
        #   buffer or an ExcelWriter instance), sheet_name (default Sheet1), startrow
        #   and startcol (zero based offsets of the upper left cell, default zero),
        #   freeze_panes (optional pair), engine (engine name if writer is a path),
        #   storage_options and engine_kwargs (default none, treated as empty). Returns
        #   nothing.
        # Behavior: raises a value error if the frame exceeds the sheet limits. If
        #   autofilter is on, computes the filter range string: it starts one row below
        #   the start row and spans the data columns (index levels count; with flat
        #   index and columns one extra column), with extra rows for multi level column
        #   headers; raises a value error when merged cells are enabled together with a
        #   multi level index or multi level columns since Excel would filter only the
        #   first row of merged cells. Creates an ExcelWriter if writer is not already
        #   one (and then closes it at the end), and sends the cells, the offsets, the
        #   freeze pane pair and the filter range to the writer's cell writing hook.
        # Side effects: writes the file or buffer; closes only a writer it created.
        # KNOWN ISSUE: the docstring does not document the autofilter behavior and
        #   describes freeze_panes as one based bottom most row and right most column,
        #   whereas the writer engines treat the pair as the number of rows and columns
        #   to freeze from the top left.
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``,
            or ``io.excel.xlsm.writer``.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        engine_kwargs: dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        from pandas.io.excel import ExcelWriter

        num_rows, num_cols = self.df.shape
        if num_rows > self.max_rows or num_cols > self.max_cols:
            raise ValueError(
                f"This sheet is too large! Your sheet size is: {num_rows}, {num_cols} "
                f"Max sheet size is: {self.max_rows}, {self.max_cols}"
            )

        if self.autofilter:
            # default offset for header row
            startrowsoffset = 1
            endrowsoffset = 1

            if num_cols == 0:
                indexoffset = 0
            elif self.index:
                indexoffset = 0
                if isinstance(self.df.index, MultiIndex):
                    if self.merge_cells:
                        raise ValueError(
                            "Excel filters merged cells by showing only the first row. "
                            "'autofilter' and 'merge_cells' cannot "
                            "be used simultaneously."
                        )
                    else:
                        indexoffset = self.df.index.nlevels - 1

                if isinstance(self.columns, MultiIndex):
                    if self.merge_cells:
                        raise ValueError(
                            "Excel filters merged cells by showing only the first row. "
                            "'autofilter' and 'merge_cells' cannot "
                            "be used simultaneously."
                        )
                    else:
                        startrowsoffset = self.columns.nlevels
                        # multindex columns add a blank row between header and data
                        endrowsoffset = self.columns.nlevels + 1
            else:
                # no index column
                indexoffset = -1
            start = f"{self._num2excel(startcol)}{startrow + startrowsoffset}"
            autofilter_end_column = self._num2excel(startcol + num_cols + indexoffset)
            end = f"{autofilter_end_column}{startrow + num_rows + endrowsoffset}"
            autofilter_range = f"{start}:{end}"
        else:
            autofilter_range = None

        if engine_kwargs is None:
            engine_kwargs = {}

        formatted_cells = self.get_formatted_cells()
        if isinstance(writer, ExcelWriter):
            need_save = False
        else:
            writer = ExcelWriter(
                writer,
                engine=engine,
                storage_options=storage_options,
                engine_kwargs=engine_kwargs,
            )
            need_save = True

        try:
            writer._write_cells(
                formatted_cells,
                sheet_name,
                startrow=startrow,
                startcol=startcol,
                freeze_panes=freeze_panes,
                autofilter_range=autofilter_range,
            )
        finally:
            # make sure to close opened file handles
            if need_save:
                writer.close()
