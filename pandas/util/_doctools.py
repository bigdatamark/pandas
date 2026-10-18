# [MODULE DOC]
# Documentation helper for drawing table-style figures that illustrate data
# frame operations (for example merge and concatenation pictures used in the user
# guide). Public content: the TablePlotter class and a demonstration entry
# function. Depends on numpy, the top-level pandas package, and (lazily, at
# plotting time) matplotlib. Importing has no side effects; running as a script
# executes the demonstration, which opens interactive figure windows.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.figure import Figure


class TablePlotter:
    # [CLASS DOC]
    # Lays out several input data frames and one result data frame as tables in a
    # single matplotlib figure, vertically stacked or side by side. State: cell_width,
    # cell_height and font_size, the sizing parameters set at construction and used
    # to derive figure size and text size. Key methods: plot (public, builds the
    # figure), plus private helpers for shape computation, size computation, data
    # conversion, index insertion and per-axis table drawing.
    """
    Layout some DataFrames in vertical/horizontal layout for explanation.
    Used in merging.rst
    """

    def __init__(
        self,
        cell_width: float = 0.37,
        cell_height: float = 0.25,
        font_size: float = 7.5,
    ) -> None:
        # [FUNCTION DOC]
        # Stores sizing parameters. Inputs: cell_width, default 0.37 (inches per table
        # column); cell_height, default 0.25 (inches per table row); font_size, default
        # 7.5 (points). No validation.
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.font_size = font_size

    def _shape(self, df: pd.DataFrame) -> tuple[int, int]:
        # [FUNCTION DOC]
        # HELPER: computes the displayed grid size of a data frame. Input: df. Returns a
        # pair (rows, columns) equal to the frame shape plus the number of column index
        # levels added to the rows and the number of row index levels added to the columns.
        """
        Calculate table shape considering index levels.
        """
        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

    def _get_cells(self, left, right, vertical) -> tuple[int, int]:
        # [FUNCTION DOC]
        # HELPER: computes the figure size in table cells. Inputs: left, a list of frames;
        # right, the result frame; vertical, whether the left frames are stacked. Returns
        # a pair (horizontal cells, vertical cells). Vertical layout: vertical cells are
        # the larger of the summed heights of left frames and the right frame height,
        # horizontal cells are the widest left frame plus the right frame width.
        # Horizontal layout: vertical cells are the maximum height and horizontal cells
        # the sum of all widths.
        """
        Calculate appropriate figure size based on left and right data.
        """
        if vertical:
            # calculate required number of cells
            vcells = max(sum(self._shape(df)[0] for df in left), self._shape(right)[0])
            hcells = max(self._shape(df)[1] for df in left) + self._shape(right)[1]
        else:
            vcells = max([self._shape(df)[0] for df in left] + [self._shape(right)[0]])
            hcells = sum([self._shape(df)[1] for df in left] + [self._shape(right)[1]])
        return hcells, vcells

    def plot(
        self, left, right, labels: Iterable[str] = (), vertical: bool = True
    ) -> Figure:
        # [FUNCTION DOC]
        # Draws the left frames and the result frame into a new matplotlib figure and
        # returns it. Inputs: left, one frame or a list of frames shown before the
        # operation; right, the operation result (frame or named series); labels, titles
        # for the left frames, default empty; vertical, default true, stacks the left
        # frames in a column with the result on the right, otherwise places everything
        # in one row. Inputs are converted by the conversion helper first (series become
        # frames, missing values are shown as text). The result table is titled "Result".
        # Raises a value error (strict pairing) when the number of labels differs from
        # the number of left frames, which includes the default empty labels. Requires
        # matplotlib; creates a figure as a side effect.
        # KNOWN ISSUE: both branches of the figure-size computation are identical, so the
        # "include margin for titles" comment for the horizontal case is not reflected.
        """
        Plot left / right DataFrames in specified layout.

        Parameters
        ----------
        left : list of DataFrames before operation is applied
        right : DataFrame of operation result
        labels : list of str to be drawn as titles of left DataFrames
        vertical : bool, default True
            If True, use vertical layout. If False, use horizontal layout.
        """
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        if not isinstance(left, list):
            left = [left]
        left = [self._conv(df) for df in left]
        right = self._conv(right)

        hcells, vcells = self._get_cells(left, right, vertical)

        if vertical:
            figsize = self.cell_width * hcells, self.cell_height * vcells
        else:
            # include margin for titles
            figsize = self.cell_width * hcells, self.cell_height * vcells
        fig = plt.figure(figsize=figsize)

        if vertical:
            gs = gridspec.GridSpec(len(left), hcells)
            # left
            max_left_cols = max(self._shape(df)[1] for df in left)
            max_left_rows = max(self._shape(df)[0] for df in left)
            for i, (_left, _label) in enumerate(zip(left, labels, strict=True)):
                ax = fig.add_subplot(gs[i, 0:max_left_cols])
                self._make_table(ax, _left, title=_label, height=1.0 / max_left_rows)
            # right
            ax = plt.subplot(gs[:, max_left_cols:])
            self._make_table(ax, right, title="Result", height=1.05 / vcells)
            fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
        else:
            max_rows = max(self._shape(df)[0] for df in [*left, right])
            height = 1.0 / np.max(max_rows)
            gs = gridspec.GridSpec(1, hcells)
            # left
            i = 0
            for df, _label in zip(left, labels, strict=True):
                sp = self._shape(df)
                ax = fig.add_subplot(gs[0, i : i + sp[1]])
                self._make_table(ax, df, title=_label, height=height)
                i += sp[1]
            # right
            ax = plt.subplot(gs[0, i:])
            self._make_table(ax, right, title="Result", height=height)
            fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

        return fig

    def _conv(self, data):
        # [FUNCTION DOC]
        # HELPER: normalizes an input for table drawing. A series is turned into a
        # one-column frame (empty column name if the series is unnamed); then missing
        # values are replaced by the text "NaN". Returns the converted object (a new one;
        # input not modified).
        """
        Convert each input to appropriate for table outplot.
        """
        if isinstance(data, pd.Series):
            if data.name is None:
                data = data.to_frame(name="")
            else:
                data = data.to_frame()
        data = data.fillna("NaN")
        return data

    def _insert_index(self, data):
        # [FUNCTION DOC]
        # HELPER: prepares a frame for rendering with its index and column levels shown as
        # ordinary cells. Works on a copy. The index is inserted as leading column(s)
        # named "Index" (single level) or "Index" plus the level number (several levels).
        # For multi-level columns, the upper column levels other than the first are
        # turned into extra leading rows via concatenation and the first level becomes
        # the column labels. Returns the new frame.
        # insert is destructive
        data = data.copy()
        idx_nlevels = data.index.nlevels
        if idx_nlevels == 1:
            data.insert(0, "Index", data.index)
        else:
            for i in range(idx_nlevels):
                data.insert(i, f"Index{i}", data.index._get_level_values(i))

        col_nlevels = data.columns.nlevels
        if col_nlevels > 1:
            col = data.columns._get_level_values(0)
            values = [
                data.columns._get_level_values(i)._values for i in range(1, col_nlevels)
            ]
            col_df = pd.DataFrame(values)
            data.columns = col_df.columns
            data = pd.concat([col_df, data])
            data.columns = col
        return data

    def _make_table(self, ax, df, title: str, height: float | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: draws one frame as a table on a given axis. Inputs: ax, the matplotlib
        # axis; df, the frame, or none (the axis is then hidden and nothing is drawn);
        # title, text for the axis title; height, row height as a fraction of the axis,
        # default none meaning one over the number of rows plus one. Inserts the index
        # via the index helper, creates the table at the top centre position, sets font
        # size, hides cells in the index-corner area and the dummy column, shades header
        # and index cells grey, sets all cell heights, titles the axis, and turns the
        # axis lines off. Returns nothing.
        if df is None:
            ax.set_visible(False)
            return

        from pandas import plotting

        idx_nlevels = df.index.nlevels
        col_nlevels = df.columns.nlevels
        # must be convert here to get index levels for colorization
        df = self._insert_index(df)
        tb = plotting.table(ax, df, loc=9)
        tb.set_fontsize(self.font_size)

        if height is None:
            height = 1.0 / (len(df) + 1)

        props = tb.properties()
        for (r, c), cell in props["celld"].items():
            if c == -1:
                cell.set_visible(False)
            elif r < col_nlevels and c < idx_nlevels:
                cell.set_visible(False)
            elif r < col_nlevels or c < idx_nlevels:
                cell.set_facecolor("#AAAAAA")
            cell.set_height(height)

        ax.set_title(title, size=self.font_size)
        ax.axis("off")


def main() -> None:
    # [FUNCTION DOC]
    # Demonstration entry point (run when the module is executed as a script).
    # Builds sample frames, including a multi-level index/columns example, and shows
    # three figures with a plotter using default sizing, vertically and horizontally.
    # Blocks on interactive figure windows; requires matplotlib.
    import matplotlib.pyplot as plt

    p = TablePlotter()

    df1 = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22], "C": [30, 31, 32]})
    df2 = pd.DataFrame({"A": [10, 12], "C": [30, 32]})

    p.plot([df1, df2], pd.concat([df1, df2]), labels=["df1", "df2"], vertical=True)
    plt.show()

    df3 = pd.DataFrame({"X": [10, 12], "Z": [30, 32]})

    p.plot(
        [df1, df3], pd.concat([df1, df3], axis=1), labels=["df1", "df2"], vertical=False
    )
    plt.show()

    idx = pd.MultiIndex.from_tuples(
        [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")]
    )
    column = pd.MultiIndex.from_tuples([(1, "A"), (1, "B")])
    df3 = pd.DataFrame({"v1": [1, 2, 3, 4, 5, 6], "v2": [5, 6, 7, 8, 9, 10]}, index=idx)
    df3.columns = column
    p.plot(df3, df3, labels=["df3"])
    plt.show()


if __name__ == "__main__":
    main()
