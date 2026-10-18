# [MODULE DOC]
# Benchmarks for the table styling (styler) HTML rendering path, for time and peak
# memory, exercising cell-wise styling functions, CSS classes, tooltips, number
# formatting and hiding of rows and columns. Public contents: the benchmark class
# Render. Uses the private render method of the styler. Does not re-export the
# common asv setup hook. No import-time side effects.
import numpy as np

from pandas import (
    DataFrame,
    IndexSlice,
)


class Render:
    # [CLASS DOC]
    # Benchmark of building a styled view of a float frame in several ways and then
    # rendering it to HTML. Parametrized by column count (12, 24, 36) and row count
    # (12, 120). State: df (set in setup) and st (the styler built by the private
    # style helper methods during each timing run).
    params = [[12, 24, 36], [12, 120]]
    param_names = ["cols", "rows"]

    def setup(self, cols, rows):
        # [FUNCTION DOC]
        # Builds a random normal frame of the requested shape with column labels
        # float_1 and up and row labels row_1 and up, stored as df.
        self.df = DataFrame(
            np.random.randn(rows, cols),
            columns=[f"float_{i + 1}" for i in range(cols)],
            index=[f"row_{i + 1}" for i in range(rows)],
        )

    def time_apply_render(self, cols, rows):
        # [FUNCTION DOC]
        # Times applying a row-wise style function and rendering to HTML (with both
        # flags of the private render call enabled). Parameters are unused here.
        self._style_apply()
        self.st._render_html(True, True)

    def peakmem_apply_render(self, cols, rows):
        # [FUNCTION DOC]
        # Peak-memory benchmark of the row-wise style application and rendering.
        self._style_apply()
        self.st._render_html(True, True)

    def time_classes_render(self, cols, rows):
        # [FUNCTION DOC]
        # Times assigning CSS classes per cell and rendering to HTML.
        self._style_classes()
        self.st._render_html(True, True)

    def peakmem_classes_render(self, cols, rows):
        # [FUNCTION DOC]
        # Peak-memory benchmark of per-cell class assignment and rendering.
        self._style_classes()
        self.st._render_html(True, True)

    def time_tooltips_render(self, cols, rows):
        # [FUNCTION DOC]
        # Times attaching tooltips and rendering to HTML.
        self._style_tooltips()
        self.st._render_html(True, True)

    def peakmem_tooltips_render(self, cols, rows):
        # [FUNCTION DOC]
        # Peak-memory benchmark of attaching tooltips and rendering.
        self._style_tooltips()
        self.st._render_html(True, True)

    def time_format_render(self, cols, rows):
        # [FUNCTION DOC]
        # Times number formatting of a sub-block of the table and rendering to HTML.
        self._style_format()
        self.st._render_html(True, True)

    def peakmem_format_render(self, cols, rows):
        # [FUNCTION DOC]
        # Peak-memory benchmark of sub-block number formatting and rendering.
        self._style_format()
        self.st._render_html(True, True)

    def time_apply_format_hide_render(self, cols, rows):
        # [FUNCTION DOC]
        # Times cell styling, number formatting, hiding of all but the first row and
        # column, and rendering to HTML.
        self._style_apply_format_hide()
        self.st._render_html(True, True)

    def peakmem_apply_format_hide_render(self, cols, rows):
        # [FUNCTION DOC]
        # Peak-memory benchmark of the style, format, hide and render sequence.
        self._style_apply_format_hide()
        self.st._render_html(True, True)

    def _style_apply(self):
        # [FUNCTION DOC]
        # HELPER: builds the styler for the apply benchmarks. Applies, row by row,
        # a function that returns a light cyan background for every cell of the row
        # labelled row_1 and an empty style for other rows, and stores the styler
        # as st.
        def _apply_func(s):
            # [FUNCTION DOC]
            # HELPER: row-wise style function for the apply helper method. Input s
            # is one row as a series. Returns a list with one style string per
            # value: a light cyan background if the row name is row_1, otherwise an
            # empty string.
            return [
                "background-color: lightcyan" if s.name == "row_1" else "" for v in s
            ]

        self.st = self.df.style.apply(_apply_func, axis=1)

    def _style_classes(self):
        # [FUNCTION DOC]
        # HELPER: builds the styler for the classes benchmarks. Computes a frame of
        # class names, cls-1 for positive cells and empty otherwise, restores its
        # index and columns to the original labels, assigns it as the cell classes
        # of a styler, and stores the styler as st.
        classes = self.df.map(lambda v: "cls-1" if v > 0 else "")
        classes.index, classes.columns = self.df.index, self.df.columns
        self.st = self.df.style.set_td_classes(classes)

    def _style_format(self):
        # [FUNCTION DOC]
        # HELPER: builds the styler for the format benchmarks. Computes three
        # quarters of the column count and row count (as integers) and applies a
        # thousands-separated three-decimal number format to the top-left block of
        # rows row_1 to that row and columns float_1 to that column. The subset
        # selection is flexible but prevents vectorized formatting. Stores the
        # styler as st.
        ic = int(len(self.df.columns) / 4 * 3)
        ir = int(len(self.df.index) / 4 * 3)
        # apply a formatting function
        # subset is flexible but hinders vectorised solutions
        self.st = self.df.style.format(
            "{:,.3f}",
            subset=IndexSlice["row_1" : f"row_{ir}", "float_1" : f"float_{ic}"],
        )

    def _style_apply_format_hide(self):
        # [FUNCTION DOC]
        # HELPER: builds the styler for the hide benchmarks. Applies a constant red
        # text style to every cell, formats numbers to three decimals, then hides
        # all rows except the first and all columns except the first. Stores the
        # styler as st.
        self.st = self.df.style.map(lambda v: "color: red;")
        self.st.format("{:.3f}")
        self.st.hide(self.st.index[1:], axis=0)
        self.st.hide(self.st.columns[1:], axis=1)

    def _style_tooltips(self):
        # [FUNCTION DOC]
        # HELPER: builds the styler for the tooltips benchmarks. Creates a frame of
        # the text abc over every second row and every second column, attaches it as
        # tooltips, then hides all rows and columns from position 12 onward. Stores
        # the styler as st.
        ttips = DataFrame("abc", index=self.df.index[::2], columns=self.df.columns[::2])
        self.st = self.df.style.set_tooltips(ttips)
        self.st.hide(self.st.index[12:], axis=0)
        self.st.hide(self.st.columns[12:], axis=1)
