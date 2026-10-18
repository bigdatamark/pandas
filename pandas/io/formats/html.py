"""
Module for formatting output data in HTML.
"""

# [MODULE DOC]
# HTML rendering of frames for the frame HTML conversion method and the notebook rich
#   display hook.
# Public contents: HTMLFormatter (generic table renderer that accumulates lines of
#   markup) and NotebookFormatter (subclass that adds a scoped style block and a
#   wrapping division for notebooks). Both are driven by an already configured frame
#   formatter from the format module, which supplies the truncated frame, formatting of
#   columns and index, and display options; the level length helper from that module is
#   used for sparsified multi level headers.
# Import-time side effects: none. Reads display options for border and math rendering
#   when instances are created or rendered.

from __future__ import annotations

from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    cast,
)

from pandas._config import get_option

from pandas._libs import lib

from pandas import (
    MultiIndex,
    option_context,
)

from pandas.io.common import is_url
from pandas.io.formats.format import (
    DataFrameFormatter,
    get_level_lengths,
)
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
    )


class HTMLFormatter:
    # [CLASS DOC]
    # Produces HTML table markup for a frame. Generates only markup; any shared logic
    #   with other output formats lives in the frame formatter classes.
    # State: fmt (frame formatter), classes (extra css classes: string, list or tuple,
    #   normalized to a list at render time), frame (the full frame), columns (columns
    #   of the truncated frame), elements (list of output lines accumulated by the write
    #   methods), bold_rows, escape, show_dimensions, border (integer border attribute
    #   or none when disabled), table_id, render_links (turn URL cells into links) and
    #   col_space (mapping from column label to a minimum width string, with integer
    #   values given pixel units). The class constant indent_delta (two) is the extra
    #   indentation per nesting level.
    # Key methods: to_string and render (public), the write helpers for cells and rows,
    #   and the private writers for table, header, body and rows.
    """
    Internal class for formatting output data in html.
    This class is intended for shared functionality between
    DataFrame.to_html() and DataFrame._repr_html_().
    Any logic in common with other output formatting methods
    should ideally be inherited from classes in format.py
    and this class responsible for only producing html markup.
    """

    indent_delta: Final = 2

    def __init__(
        self,
        formatter: DataFrameFormatter,
        classes: str | list[str] | tuple[str, ...] | None = None,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Initialize the renderer. Inputs: formatter (frame formatter), classes (extra
        #   css classes, default none), border (integer border width; none or true means
        #   use the display option for the border; any other falsy value disables the
        #   attribute), table_id (optional id attribute), render_links (default false).
        # Copies options from the formatter and builds the column space mapping: integer
        #   widths get a pixel suffix, other values are kept as they are; for multi
        #   level columns, tuple keys are also registered under each of their components
        #   as text so that header cells of single levels find their width.
        self.fmt = formatter
        self.classes = classes

        self.frame = self.fmt.frame
        self.columns = self.fmt.tr_frame.columns
        self.elements: list[str] = []
        self.bold_rows = self.fmt.bold_rows
        self.escape = self.fmt.escape
        self.show_dimensions = self.fmt.show_dimensions
        if border is None or border is True:
            border = cast(int, get_option("display.html.border"))
        elif not border:
            border = None

        self.border = border
        self.table_id = table_id
        self.render_links = render_links

        self.col_space = {}
        is_multi_index = isinstance(self.columns, MultiIndex)
        for column, value in self.fmt.col_space.items():
            col_space_value = f"{value}px" if isinstance(value, int) else value
            self.col_space[column] = col_space_value
            # GH 53885: Handling case where column is index
            # Flatten the data in the multi index and add in the map
            if is_multi_index and isinstance(column, tuple):
                for column_index in column:
                    self.col_space[str(column_index)] = col_space_value

    def to_string(self) -> str:
        # [FUNCTION DOC]
        # Return the whole table markup as one string with lines joined by newlines.
        #   Renders the table; if any line is text, converts all lines to text first.
        lines = self.render()
        if any(isinstance(x, str) for x in lines):
            lines = [str(x) for x in lines]
        return "\n".join(lines)

    def render(self) -> list[str]:
        # [FUNCTION DOC]
        # Build the markup and return the list of lines. Writes the table; if the
        #   dimensions should be shown, then appends a paragraph stating the number of
        #   rows and columns of the full (not truncated) frame, separated by a
        #   multiplication sign. Returns the accumulated list of lines (the same list
        #   object stored in the state).
        # Side effects: appends to the elements list on every call, so calling it twice
        #   duplicates the output.
        self._write_table()

        if self.should_show_dimensions:
            by = chr(215)  # ×  # noqa: RUF003
            self.write(
                f"<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>"
            )

        return self.elements

    @property
    def should_show_dimensions(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: whether to show the dimensions line, taken from the
        #   formatter.
        return self.fmt.should_show_dimensions

    @property
    def show_row_idx_names(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: whether to show row index names in the header, taken from
        #   the formatter.
        return self.fmt.show_row_idx_names

    @property
    def show_col_idx_names(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: whether to show column index names, taken from the
        #   formatter.
        return self.fmt.show_col_idx_names

    @property
    def row_levels(self) -> int:
        # [FUNCTION DOC]
        # Read only property: the number of leading columns used by the row labels. The
        #   number of index levels when the index is shown; otherwise one if column
        #   index names are shown (a blank column is then needed to keep alignment),
        #   else zero.
        if self.fmt.index:
            # showing (row) index
            return self.frame.index.nlevels
        elif self.show_col_idx_names:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # If the row index is not displayed a column of
            # blank cells need to be included before the DataFrame values.
            return 1
        # not showing (row) index
        return 0

    def _get_columns_formatted_values(self) -> Iterable:
        # [FUNCTION DOC]
        # HELPER: returns the column labels to place in a header row for a flat column
        #   index. Here it returns the columns as they are; the notebook subclass
        #   overrides it to return formatted labels.
        return self.columns

    @property
    def is_truncated(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: whether the frame is truncated, taken from the formatter.
        return self.fmt.is_truncated

    @property
    def ncols(self) -> int:
        # [FUNCTION DOC]
        # Read only property: the number of columns in the truncated frame.
        return len(self.fmt.tr_frame.columns)

    def write(self, s: Any, indent: int = 0) -> None:
        # [FUNCTION DOC]
        # Append one line to the output. Inputs: s (any object, converted to text by the
        #   pretty printing helper) and indent (number of leading spaces, default zero).
        #   Returns nothing.
        rs = pprint_thing(s)
        self.elements.append(" " * indent + rs)

    def write_th(
        self, s: Any, header: bool = False, indent: int = 0, tags: str | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Write a header cell. Inputs: s (cell content), header (true when the cell is
        #   inside the table head, default false), indent (default zero) and tags (extra
        #   attribute text, default none). When the cell is a header cell and the column
        #   space mapping has an entry for the content, a minimum width style attribute
        #   is appended to the tags. Delegates to the cell writer with the header cell
        #   kind. Returns nothing.
        # KNOWN ISSUE: the docstring says the method returns a written cell, but it
        #   returns nothing (it appends a line). When tags is already non empty, the
        #   style attribute is appended with no separating space, which can produce
        #   malformed markup.
        """
        Method for writing a formatted <th> cell.

        If col_space is set on the formatter then that is used for
        the value of min-width.

        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : bool, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : str, default None
            Tags to include in the cell.

        Returns
        -------
        A written <th> cell.
        """
        col_space = self.col_space.get(s, None)

        if header and col_space is not None:
            tags = tags or ""
            tags += f'style="min-width: {col_space};"'

        self._write_cell(s, kind="th", indent=indent, tags=tags)

    def write_td(self, s: Any, indent: int = 0, tags: str | None = None) -> None:
        # [FUNCTION DOC]
        # Write a data cell with the given content, indent and attribute text by
        #   delegating to the cell writer with the data cell kind. Returns nothing.
        self._write_cell(s, kind="td", indent=indent, tags=tags)

    def _write_cell(
        self, s: Any, kind: str = "td", indent: int = 0, tags: str | None = None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: writes a single cell element. Inputs: s (content), kind (th or td,
        #   default td), indent and tags (attribute text). Builds the start tag with the
        #   attributes; converts the content to text with the pretty printer, escaping
        #   ampersand, less than and greater than characters when escaping is on, strips
        #   surrounding whitespace and replaces runs of two spaces with non-breaking
        #   space entities. When link rendering is on and the content looks like a URL,
        #   wraps it in an anchor opening in a new window, using the unescaped text as
        #   the target. Appends the finished line. Returns nothing.
        if tags is not None:
            start_tag = f"<{kind} {tags}>"
        else:
            start_tag = f"<{kind}>"

        if self.escape:
            # escape & first to prevent double escaping of &
            esc = {"&": r"&amp;", "<": r"&lt;", ">": r"&gt;"}
        else:
            esc = {}

        rs = pprint_thing(s, escape_chars=esc).strip()
        # replace spaces betweens strings with non-breaking spaces
        rs = rs.replace("  ", "&nbsp;&nbsp;")

        if self.render_links and is_url(rs):
            rs_unescaped = pprint_thing(s, escape_chars={}).strip()
            start_tag += f'<a href="{rs_unescaped}" target="_blank">'
            end_a = "</a>"
        else:
            end_a = ""

        self.write(f"{start_tag}{rs}{end_a}</{kind}>", indent)

    def write_tr(
        self,
        line: Iterable,
        indent: int = 0,
        indent_delta: int = 0,
        header: bool = False,
        align: str | None = None,
        tags: dict[int, str] | None = None,
        nindex_levels: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Write a table row. Inputs: line (iterable of cell contents), indent,
        #   indent_delta (extra indentation for cells), header (true to emit header
        #   cells), align (optional text alignment style for the row), tags (optional
        #   mapping from cell position to attribute text) and nindex_levels (number of
        #   leading index cells; with bold rows enabled these are written as header
        #   cells). Writes the opening row tag, each cell as a header cell when header
        #   is set or the cell is an index cell under bold rows (else a data cell), and
        #   the closing tag. Returns nothing.
        if tags is None:
            tags = {}

        if align is None:
            self.write("<tr>", indent)
        else:
            self.write(f'<tr style="text-align: {align};">', indent)
        indent += indent_delta

        for i, s in enumerate(line):
            val_tag = tags.get(i, None)
            if header or (self.bold_rows and i < nindex_levels):
                self.write_th(s, indent=indent, header=header, tags=val_tag)
            else:
                self.write_td(s, indent, tags=val_tag)

        indent -= indent_delta
        self.write("</tr>", indent)

    def _write_table(self, indent: int = 0) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the table element. Builds the class list starting with the
        #   default frame class; adds two classes that disable math rendering when the
        #   display option for math rendering is off; adds user classes (a string is
        #   split on whitespace); raises a type error if the classes value is not a
        #   string, list or tuple. Adds the id and border attributes when set. Writes
        #   the opening tag, then the head (when headers or row index names are shown),
        #   then the body, then the closing tag, with nested indentation.
        # Side effects: replaces the stored classes with the split list when a string
        #   was given.
        _classes = ["dataframe"]  # Default class.
        use_mathjax = get_option("display.html.use_mathjax")
        if not use_mathjax:
            _classes.append("tex2jax_ignore")
            _classes.append("mathjax_ignore")
        if self.classes is not None:
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if not isinstance(self.classes, (list, tuple)):
                raise TypeError(
                    "classes must be a string, list, "
                    f"or tuple, not {type(self.classes)}"
                )
            _classes.extend(self.classes)

        if self.table_id is None:
            id_section = ""
        else:
            id_section = f' id="{self.table_id}"'

        if self.border is None:
            border_attr = ""
        else:
            border_attr = f' border="{self.border}"'

        self.write(
            f'<table{border_attr} class="{" ".join(_classes)}"{id_section}>',
            indent,
        )

        if self.fmt.header or self.show_row_idx_names:
            self._write_header(indent + self.indent_delta)

        self._write_body(indent + self.indent_delta)

        self.write("</table>", indent)

    def _write_col_header(self, indent: int) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the column header rows inside the table head. For multi level
        #   columns writes one row per level: applies horizontal truncation by inserting
        #   an ellipsis column into each level (adjusting span records, for sparsified
        #   headers, so that spans crossing the insertion point grow and later ones
        #   shift), starts each row with blank cells and the level name when index names
        #   are shown, then emits values, adding a colspan and left alignment attribute
        #   for entries spanning more than one column. For a flat column index writes
        #   one header row: blank cells and the column index name, then the column
        #   labels, with an ellipsis inserted at the truncation position, aligned
        #   according to the formatter's justification.
        row: list[Hashable]
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        if isinstance(self.columns, MultiIndex):
            template = 'colspan="{span:d}" halign="left"'

            sentinel: lib.NoDefault | bool
            if self.fmt.sparsify:
                # GH3547
                sentinel = lib.no_default
            else:
                sentinel = False
            levels = self.columns._format_multi(sparsify=sentinel, include_names=False)
            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            for lnum, (records, values) in enumerate(
                zip(level_lengths, levels, strict=True)
            ):
                if is_truncated_horizontally:
                    # modify the header lines
                    ins_col = self.fmt.tr_col_num
                    if self.fmt.sparsify:
                        recs_new = {}
                        # Increment tags after ... col.
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            elif tag + span > ins_col:
                                recs_new[tag] = span + 1
                                if lnum == inner_lvl:
                                    values = (
                                        *values[:ins_col],
                                        "...",
                                        *values[ins_col:],
                                    )
                                else:
                                    # sparse col headers do not receive a ...
                                    values = (
                                        *values[:ins_col],
                                        values[ins_col - 1],
                                        *values[ins_col:],
                                    )
                            else:
                                recs_new[tag] = span
                            # if ins_col lies between tags, all col headers
                            # get ...
                            if tag + span == ins_col:
                                recs_new[ins_col] = 1
                                values = (*values[:ins_col], "...", *values[ins_col:])
                        records = recs_new
                        inner_lvl = len(level_lengths) - 1
                        if lnum == inner_lvl:
                            records[ins_col] = 1
                    else:
                        recs_new = {}
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            else:
                                recs_new[tag] = span
                        recs_new[ins_col] = 1
                        records = recs_new
                        values = [*values[:ins_col], "...", *values[ins_col:]]

                # see gh-22579
                # Column Offset Bug with to_html(index=False) with
                # MultiIndex Columns and Index.
                # Initially fill row with blank cells before column names.
                # TODO: Refactor to remove code duplication with code
                # block below for standard columns index.
                row = [""] * (self.row_levels - 1)
                if self.fmt.index or self.show_col_idx_names:
                    # see gh-22747
                    # If to_html(index_names=False) do not show columns
                    # index names.
                    # TODO: Refactor to use _get_column_name_list from
                    # DataFrameFormatter class and create a
                    # _get_formatted_column_labels function for code
                    # parity with DataFrameFormatter class.
                    if self.fmt.show_index_names:
                        name = self.columns.names[lnum]
                        row.append(pprint_thing(name or ""))
                    else:
                        row.append("")

                tags = {}
                j = len(row)
                for i, v in enumerate(values):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        continue
                    j += 1
                    row.append(v)
                self.write_tr(row, indent, self.indent_delta, tags=tags, header=True)
        else:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Initially fill row with blank cells before column names.
            # TODO: Refactor to remove code duplication with code block
            # above for columns MultiIndex.
            row = [""] * (self.row_levels - 1)
            if self.fmt.index or self.show_col_idx_names:
                # see gh-22747
                # If to_html(index_names=False) do not show columns
                # index names.
                # TODO: Refactor to use _get_column_name_list from
                # DataFrameFormatter class.
                if self.fmt.show_index_names:
                    row.append(self.columns.name or "")
                else:
                    row.append("")
            row.extend(self._get_columns_formatted_values())
            align = self.fmt.justify

            if is_truncated_horizontally:
                ins_col = self.row_levels + self.fmt.tr_col_num
                row.insert(ins_col, "...")

            self.write_tr(row, indent, self.indent_delta, header=True, align=align)

    def _write_row_header(self, indent: int) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the extra header row that holds the index level names: the
        #   names (empty string for unnamed) followed by blank cells for every column,
        #   plus one for the ellipsis column when truncated horizontally.
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        row = [x if x is not None else "" for x in self.frame.index.names] + [""] * (
            self.ncols + (1 if is_truncated_horizontally else 0)
        )
        self.write_tr(row, indent, self.indent_delta, header=True)

    def _write_header(self, indent: int) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the head element: the column header rows if headers are shown,
        #   and the row for index names if shown.
        self.write("<thead>", indent)

        if self.fmt.header:
            self._write_col_header(indent + self.indent_delta)

        if self.show_row_idx_names:
            self._write_row_header(indent + self.indent_delta)

        self.write("</thead>", indent)

    def _get_formatted_values(self) -> dict[int, list[str]]:
        # [FUNCTION DOC]
        # HELPER: formats every column of the truncated frame to text with the
        #   formatter, with the column width display option temporarily unlimited so
        #   that long values are not cut. Returns a mapping from column position to the
        #   list of formatted strings.
        with option_context("display.max_colwidth", None):
            fmt_values = {i: self.fmt.format_col(i) for i in range(self.ncols)}
        return fmt_values

    def _write_body(self, indent: int) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the body element. Formats the columns, then writes hierarchical
        #   rows when the index is shown and is multi level, otherwise regular rows.
        self.write("<tbody>", indent)
        fmt_values = self._get_formatted_values()

        # write values
        if self.fmt.index and isinstance(self.frame.index, MultiIndex):
            self._write_hierarchical_rows(fmt_values, indent + self.indent_delta)
        else:
            self._write_regular_rows(fmt_values, indent + self.indent_delta)

        self.write("</tbody>", indent)

    def _write_regular_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: writes body rows for a flat (or hidden) index. Input: fmt_values
        #   (mapping from column position to formatted strings) and indent. Computes the
        #   formatted index labels (through a custom index formatter if one is
        #   registered, otherwise the flat formatting without name). For each row, first
        #   writes an ellipsis row when the row number equals the vertical truncation
        #   position, then builds the row from the index label (or a blank cell when
        #   only column names are shown), the formatted values and an ellipsis cell at
        #   the horizontal truncation position.
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically

        nrows = len(self.fmt.tr_frame)

        if self.fmt.index:
            fmt = self.fmt._get_formatter("__index__")
            if fmt is not None:
                index_values = self.fmt.tr_frame.index.map(fmt)
            else:
                # only reached with non-Multi index
                index_values = self.fmt.tr_frame.index._format_flat(include_name=False)

        row: list[str] = []
        for i in range(nrows):
            if is_truncated_vertically and i == (self.fmt.tr_row_num):
                str_sep_row = ["..."] * len(row)
                self.write_tr(
                    str_sep_row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=self.row_levels,
                )

            row = []
            if self.fmt.index:
                row.append(index_values[i])
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Add blank cell before data cells.
            elif self.show_col_idx_names:
                row.append("")
            row.extend(fmt_values[j][i] for j in range(self.ncols))

            if is_truncated_horizontally:
                dot_col_ix = self.fmt.tr_col_num + self.row_levels
                row.insert(dot_col_ix, "...")
            self.write_tr(
                row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels
            )

    def _write_hierarchical_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: writes body rows for a multi level index. With sparsification,
        #   computes the level lengths of the index; when truncated vertically, inserts
        #   an ellipsis row into the index values, level spans and each formatted column
        #   (adjusting spans that cross the insertion point, and placing ellipses in the
        #   right levels); then writes each row, emitting label cells only at positions
        #   where a span starts (with a rowspan attribute when longer than one) and
        #   tracking how many label cells were skipped to keep the truncation column
        #   aligned. Without sparsification writes every row with the full list of
        #   labels, inserting an ellipsis row at the truncation position.
        template = 'rowspan="{span}" valign="top"'

        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically
        frame = self.fmt.tr_frame
        nrows = len(frame)

        assert isinstance(frame.index, MultiIndex)
        idx_values = frame.index._format_multi(sparsify=False, include_names=False)
        idx_values = list(zip(*idx_values, strict=True))

        if self.fmt.sparsify:
            # GH3547
            sentinel = lib.no_default
            levels = frame.index._format_multi(sparsify=sentinel, include_names=False)

            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            if is_truncated_vertically:
                # Insert ... row and adjust idx_values and
                # level_lengths to take this into account.
                ins_row = self.fmt.tr_row_num
                inserted = False
                for lnum, records in enumerate(level_lengths):
                    rec_new = {}
                    for tag, span in list(records.items()):
                        if tag >= ins_row:
                            rec_new[tag + 1] = span
                        elif tag + span > ins_row:
                            rec_new[tag] = span + 1

                            # GH 14882 - Make sure insertion done once
                            if not inserted:
                                dot_row = list(idx_values[ins_row - 1])
                                dot_row[-1] = "..."
                                idx_values.insert(ins_row, tuple(dot_row))
                                inserted = True
                            else:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = "..."
                                idx_values[ins_row] = tuple(dot_row)
                        else:
                            rec_new[tag] = span
                        # If ins_row lies between tags, all cols idx cols
                        # receive ...
                        if tag + span == ins_row:
                            rec_new[ins_row] = 1
                            if lnum == 0:
                                idx_values.insert(
                                    ins_row, tuple(["..."] * len(level_lengths))
                                )

                            # GH 14882 - Place ... in correct level
                            elif inserted:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = "..."
                                idx_values[ins_row] = tuple(dot_row)
                    level_lengths[lnum] = rec_new

                level_lengths[inner_lvl][ins_row] = 1
                for ix_col in fmt_values:
                    fmt_values[ix_col].insert(ins_row, "...")
                nrows += 1

            for i in range(nrows):
                row = []
                tags = {}

                sparse_offset = 0
                j = 0
                for records, v in zip(level_lengths, idx_values[i], strict=True):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        sparse_offset += 1
                        continue

                    j += 1
                    row.append(v)

                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if is_truncated_horizontally:
                    row.insert(
                        self.row_levels - sparse_offset + self.fmt.tr_col_num, "..."
                    )
                self.write_tr(
                    row,
                    indent,
                    self.indent_delta,
                    tags=tags,
                    nindex_levels=len(levels) - sparse_offset,
                )
        else:
            row = []
            for i in range(len(frame)):
                if is_truncated_vertically and i == (self.fmt.tr_row_num):
                    str_sep_row = ["..."] * len(row)
                    self.write_tr(
                        str_sep_row,
                        indent,
                        self.indent_delta,
                        tags=None,
                        nindex_levels=self.row_levels,
                    )

                idx_values = list(
                    zip(
                        *frame.index._format_multi(sparsify=False, include_names=False),
                        strict=True,
                    )
                )
                row = []
                row.extend(idx_values[i])
                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if is_truncated_horizontally:
                    row.insert(self.row_levels + self.fmt.tr_col_num, "...")
                self.write_tr(
                    row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=frame.index.nlevels,
                )


class NotebookFormatter(HTMLFormatter):
    # [CLASS DOC]
    # Subclass of the HTML table renderer used for notebook rich display. Adds a scoped
    #   style element and wraps the table in a division, formats columns without
    #   lifting the column width limit, and uses formatted flat labels for the column
    #   header row. No additional state.
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    def _get_formatted_values(self) -> dict[int, list[str]]:
        # [FUNCTION DOC]
        # Override: formats every column without temporarily lifting the column width
        #   limit, returning a mapping from column position to the list of formatted
        #   strings.
        return {i: self.fmt.format_col(i) for i in range(self.ncols)}

    def _get_columns_formatted_values(self) -> list[str]:
        # [FUNCTION DOC]
        # Override: returns the formatted flat column labels (without the name). Only
        #   valid for flat column indexes.
        # only reached with non-Multi Index
        return self.columns._format_flat(include_name=False)

    def write_style(self) -> None:
        # [FUNCTION DOC]
        # Writes the style element that scopes notebook specific css to the table:
        #   vertical alignment of row labels, text alignment of header cells (left for
        #   multi level columns, with right for the last head row when row index names
        #   are shown; right for flat columns). Appends the lines to the output. Returns
        #   nothing.
        # We use the "scoped" attribute here so that the desired
        # style properties for the data frame are not then applied
        # throughout the entire notebook.
        template_first = """\
            <style scoped>"""
        template_last = """\
            </style>"""
        template_select = """\
                .dataframe %s {
                    %s: %s;
                }"""
        element_props = [
            ("tbody tr th:only-of-type", "vertical-align", "middle"),
            ("tbody tr th", "vertical-align", "top"),
        ]
        if isinstance(self.columns, MultiIndex):
            element_props.append(("thead tr th", "text-align", "left"))
            if self.show_row_idx_names:
                element_props.append(
                    ("thead tr:last-of-type th", "text-align", "right")
                )
        else:
            element_props.append(("thead th", "text-align", "right"))
        template_mid = "\n\n".join(template_select % t for t in element_props)
        template = dedent(f"{template_first}\n{template_mid}\n{template_last}")
        self.write(template)

    def render(self) -> list[str]:
        # [FUNCTION DOC]
        # Override: builds the notebook markup. Writes an opening division, the style
        #   block, the normal table markup from the base class, and the closing
        #   division. Returns the accumulated list of lines.
        self.write("<div>")
        self.write_style()
        super().render()
        self.write("</div>")
        return self.elements
