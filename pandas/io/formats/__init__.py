# ruff: noqa: TC004

# [PACKAGE DOC]
# Package of output formatting machinery for pandas I/O: converting frames and series to
#   text, HTML, LaTeX, XML, CSV and Excel cell representations, plus the Styler and its
#   rendering support.
# Submodules: _color_data (named CSS color table for Excel export), console (terminal
#   size and environment detection), css (CSS declaration resolver), csvs (CSV writer
#   formatter), excel (frame to Excel cell conversion), format (core frame and series
#   formatters and display helpers), html (HTML table renderer), info (summary
#   information for frames and series), printing (string and nested structure pretty
#   printing helpers), string (plain text table renderer), style and style_render (the
#   Styler class and its rendering engine), xml (XML export), and a templates directory
#   of jinja templates.
# Exports: nothing at run time. Under type checking only, the style submodule is
#   imported and listed as the sole public module name in the export list, so the export
#   list does not exist when the package is actually imported.
# Import-time side effects: none (the package initializer only imports the typing flag).

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions
    from pandas.io.formats import style

    # and mark only those modules as public
    __all__ = ["style"]
