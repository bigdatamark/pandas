"""
Module for formatting output data in console (to string).
"""

# [MODULE DOC]
# Plain text rendering of frames for console display and the frame string conversion
#   method.
# Public contents: StringFormatter (renders a frame, using an already configured frame
#   formatter, to a text table, with optional wrapping to a line width) and the private
#   helper _binify used for wrapping. Depends on the frame formatter class from the
#   format module (for truncation state, formatted columns and the text adjustment
#   object) and on the printing helpers.
# Import-time side effects: none.

from __future__ import annotations

from shutil import get_terminal_size
from typing import TYPE_CHECKING

import numpy as np

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas.io.formats.format import DataFrameFormatter


class StringFormatter:
    # [CLASS DOC]
    # Renders the string representation of a frame. State: fmt (the frame formatter),
    #   adj (the formatter's text adjustment object used for measuring, justifying and
    #   joining columns), frame (the formatter's frame) and line_width (maximum width of
    #   a line, or none meaning no wrapping). Key methods: to_string (public) and the
    #   private steps for truncation dots, wrapping into multiple blocks and fitting to
    #   the terminal width.
    """Formatter for string representation of a dataframe."""

    def __init__(self, fmt: DataFrameFormatter, line_width: int | None = None) -> None:
        # [FUNCTION DOC]
        # Store the formatter, its text adjuster, its frame and the line width (default
        #   none).
        self.fmt = fmt
        self.adj = fmt.adj
        self.frame = fmt.frame
        self.line_width = line_width

    def to_string(self) -> str:
        # [FUNCTION DOC]
        # Return the complete text for the frame. Builds the table text and, if the
        #   formatter says the dimensions should be shown, appends the dimensions
        #   information text. Returns a string.
        text = self._get_string_representation()
        if self.fmt.should_show_dimensions:
            text = f"{text}{self.fmt.dimensions_info}"
        return text

    def _get_strcols(self) -> list[list[str]]:
        # [FUNCTION DOC]
        # HELPER: returns the list of formatted columns (each a list of strings,
        #   including the index column when shown). If the formatter reports truncation,
        #   the placeholder separators are inserted.
        strcols = self.fmt.get_strcols()
        if self.fmt.is_truncated:
            strcols = self._insert_dot_separators(strcols)
        return strcols

    def _get_string_representation(self) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the table text. An empty frame gives the empty frame summary
        #   text. Without a line width returns all columns joined side by side. With a
        #   line width: if wrapping is allowed, splits the columns into several blocks
        #   with the multi line joiner; otherwise fits the columns to the terminal width
        #   by truncating.
        if self.fmt.frame.empty:
            return self._empty_info_line

        strcols = self._get_strcols()

        if self.line_width is None:
            # no need to wrap around just print the whole frame
            return self.adj.adjoin(1, *strcols)

        if self._need_to_wrap_around:
            return self._join_multiline(strcols)

        return self._fit_strcols_to_terminal_width(strcols)

    @property
    def _empty_info_line(self) -> str:
        # [FUNCTION DOC]
        # HELPER: property returning the text for an empty frame: a line naming the
        #   frame type, a line listing the columns and a line listing the index.
        return (
            f"Empty {type(self.frame).__name__}\n"
            f"Columns: {pprint_thing(self.frame.columns)}\n"
            f"Index: {pprint_thing(self.frame.index)}"
        )

    @property
    def _need_to_wrap_around(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property that is true when the maximum columns option is unset or
        #   positive, meaning wide frames wrap into several blocks instead of being
        #   truncated to the terminal width.
        return bool(self.fmt.max_cols is None or self.fmt.max_cols > 0)

    def _insert_dot_separators(self, strcols: list[list[str]]) -> list[list[str]]:
        # [FUNCTION DOC]
        # HELPER: inserts truncation marker rows and columns into the formatted columns.
        #   Computes the number of rows of the formatted index, then adds the horizontal
        #   separator column if the frame is truncated horizontally and the vertical
        #   separator row if truncated vertically. Returns the list of columns.
        str_index = self.fmt._get_formatted_index(self.fmt.tr_frame)
        index_length = len(str_index)

        if self.fmt.is_truncated_horizontally:
            strcols = self._insert_dot_separator_horizontal(strcols, index_length)

        if self.fmt.is_truncated_vertically:
            strcols = self._insert_dot_separator_vertical(strcols, index_length)

        return strcols

    @property
    def _adjusted_tr_col_num(self) -> int:
        # [FUNCTION DOC]
        # HELPER: property giving the position of the truncation column among the
        #   formatted columns, shifted by one when an index column is present.
        return self.fmt.tr_col_num + 1 if self.fmt.index else self.fmt.tr_col_num

    def _insert_dot_separator_horizontal(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        # [FUNCTION DOC]
        # HELPER: inserts a column of ellipsis markers, as many as index_length, at the
        #   adjusted truncation column position. Mutates and returns the list of
        #   columns.
        strcols.insert(self._adjusted_tr_col_num, [" ..."] * index_length)
        return strcols

    def _insert_dot_separator_vertical(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        # [FUNCTION DOC]
        # HELPER: inserts a row of ellipsis markers into every formatted column at the
        #   truncation row position (after the header rows, whose count is index_length
        #   minus the frame length). The marker is three dots when the column is wider
        #   than three characters or is the horizontal truncation column, otherwise two
        #   dots; the index column is left justified and other columns right justified
        #   (the horizontal truncation column gets width four). Mutates and returns the
        #   list of columns.
        n_header_rows = index_length - len(self.fmt.tr_frame)
        row_num = self.fmt.tr_row_num
        for ix, col in enumerate(strcols):
            cwidth = self.adj.len(col[row_num])

            if self.fmt.is_truncated_horizontally:
                is_dot_col = ix == self._adjusted_tr_col_num
            else:
                is_dot_col = False

            if cwidth > 3 or is_dot_col:
                dots = "..."
            else:
                dots = ".."

            if ix == 0 and self.fmt.index:
                dot_mode = "left"
            elif is_dot_col:
                cwidth = 4
                dot_mode = "right"
            else:
                dot_mode = "right"

            dot_str = self.adj.justify([dots], cwidth, mode=dot_mode)[0]
            col.insert(row_num + n_header_rows, dot_str)
        return strcols

    def _join_multiline(self, strcols_input: Iterable[list[str]]) -> str:
        # [FUNCTION DOC]
        # HELPER: splits the formatted columns into blocks that fit the line width and
        #   joins the blocks vertically. Input: strcols_input, the formatted columns.
        #   The index column, if shown, is repeated in each block and its width (plus
        #   the separator) is subtracted from the available width. Column widths are
        #   measured and grouped into bins with the binning helper. When there is more
        #   than one block, a continuation column (a backslash on the first line) is
        #   appended to every block except the last, and a blank column to the last.
        #   Blocks are joined side by side and separated by a blank line. Returns the
        #   text.
        lwidth = self.line_width
        adjoin_width = 1
        strcols = list(strcols_input)

        if self.fmt.index:
            idx = strcols.pop(0)
            lwidth -= np.array([self.adj.len(x) for x in idx]).max() + adjoin_width

        col_widths = [
            np.array([self.adj.len(x) for x in col]).max() if len(col) > 0 else 0
            for col in strcols
        ]

        assert lwidth is not None
        col_bins = _binify(col_widths, lwidth)
        nbins = len(col_bins)

        str_lst = []
        start = 0
        for i, end in enumerate(col_bins):
            row = strcols[start:end]
            if self.fmt.index:
                row.insert(0, idx)
            if nbins > 1:
                nrows = len(row[-1])
                if end <= len(strcols) and i < nbins - 1:
                    row.append([" \\"] + ["  "] * (nrows - 1))
                else:
                    row.append([" "] * nrows)
            str_lst.append(self.adj.adjoin(adjoin_width, *row))
            start = end
        return "\n\n".join(str_lst)

    def _fit_strcols_to_terminal_width(self, strcols: list[list[str]]) -> str:
        # [FUNCTION DOC]
        # HELPER: used when no wrapping is wanted but the frame is wider than the
        #   terminal. Joins the columns, measures the longest line against the terminal
        #   width, and repeatedly discards the column nearest the middle (by width,
        #   accounting for separators) until the estimated excess width is gone. Sets
        #   the formatter's fitted column count (at least two, excluding the index
        #   column), asks the formatter to truncate the frame again, rebuilds the
        #   columns and returns the joined text. Side effects: modifies the formatter
        #   state.
        from pandas import Series

        lines = self.adj.adjoin(1, *strcols).split("\n")
        max_len = Series(lines).str.len().max()
        # plus truncate dot col
        width, _ = get_terminal_size()
        dif = max_len - width
        # '+ 1' to avoid too wide repr (GH PR #17023)
        adj_dif = dif + 1
        col_lens = Series([Series(ele).str.len().max() for ele in strcols])
        n_cols = len(col_lens)
        counter = 0
        while adj_dif > 0 and n_cols > 1:
            counter += 1
            mid = round(n_cols / 2)
            mid_ix = col_lens.index[mid]
            col_len = col_lens[mid_ix]
            # adjoin adds one
            adj_dif -= col_len + 1
            col_lens = col_lens.drop(mid_ix)
            n_cols = len(col_lens)

        # subtract index column
        max_cols_fitted = n_cols - self.fmt.index
        # GH-21180. Ensure that we print at least two.
        max_cols_fitted = max(max_cols_fitted, 2)
        self.fmt.max_cols_fitted = max_cols_fitted

        # Call again _truncate to cut frame appropriately
        # and then generate string representation
        self.fmt.truncate()
        strcols = self._get_strcols()
        return self.adj.adjoin(1, *strcols)


def _binify(cols: list[int], line_width: int) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: groups column widths into blocks for wrapping. Input: cols, a list of
    #   column widths, and line_width, the available width. Accumulates widths (plus a
    #   separator each) and starts a new block when the accumulated width, with a margin
    #   of two for non final columns and one for the final column, would exceed the line
    #   width and the column is not the first. Returns the list of exclusive end
    #   positions of each block; the last entry is the number of columns.
    adjoin_width = 1
    bins = []
    curr_width = 0
    i_last_column = len(cols) - 1
    for i, w in enumerate(cols):
        w_adjoined = w + adjoin_width
        curr_width += w_adjoined
        if i_last_column == i:
            wrap = curr_width + 1 > line_width and i > 0
        else:
            wrap = curr_width + 2 > line_width and i > 0
        if wrap:
            bins.append(i)
            curr_width = w_adjoined

    bins.append(len(cols))
    return bins
