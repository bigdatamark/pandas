# [MODULE DOC]
# Excel engine for xlsx and xlsm workbooks built on the optional openpyxl library,
#   providing both a writer and a reader.
# Public contents: OpenpyxlWriter (subclass of the ExcelWriter base class; engine name
#   openpyxl; supports append mode) and OpenpyxlReader (subclass of the generic reader
#   base class; registered in the ExcelFile engine table under the name openpyxl). The
#   writer also holds class methods that translate pandas style dictionaries (as
#   produced by the Excel formatter) into native openpyxl style objects. openpyxl is
#   imported lazily inside methods.
# Import-time side effects: none (imports the shared docstring templates used by the doc
#   decorator).

from __future__ import annotations

import mmap
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import (
    BaseExcelReader,
    ExcelWriter,
)
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

if TYPE_CHECKING:
    from openpyxl import Workbook
    from openpyxl.descriptors.serialisable import Serialisable
    from openpyxl.styles import Fill

    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        ReadBuffer,
        Scalar,
        StorageOptions,
        WriteExcelBuffer,
    )


class OpenpyxlWriter(ExcelWriter):
    # [CLASS DOC]
    # Writer for xlsx and xlsm files using openpyxl. Class attributes: engine name
    #   openpyxl, supported extensions xlsx and xlsm. State: _book (the openpyxl
    #   workbook; either newly created or loaded from the existing file in append mode).
    #   Supports append mode, with the existing content handled according to the sheet
    #   exists policy. Style translation is done by a family of class methods named
    #   after the style group they convert.
    _engine = "openpyxl"
    _supported_extensions = (".xlsx", ".xlsm")

    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Create the writer. Inputs: same as the base class plus extra keyword arguments
        #   merged into engine_kwargs. Merges the two keyword sources with the util
        #   helper, runs the base constructor (which validates the path extension, maps
        #   append to read/update mode and opens the output), then creates the workbook.
        #   In read/update mode loads the existing workbook from the open handle with
        #   engine_kwargs, closing the handle and re raising if the library rejects the
        #   keywords with a type error, and rewinds the handle. Otherwise creates a new
        #   workbook with engine_kwargs (closing the handle and re raising on a type
        #   error) and removes the default first worksheet if any exist.
        # Side effects: holds an open file handle until close.
        # Use the openpyxl module as the Excel writer.
        from openpyxl.workbook import Workbook

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

        super().__init__(
            path,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        # ExcelWriter replaced "a" by "r+" to allow us to first read the excel file from
        # the file and later write to it
        if "r+" in self._mode:  # Load from existing workbook
            from openpyxl import load_workbook

            try:
                self._book = load_workbook(self._handles.handle, **engine_kwargs)
            except TypeError:
                self._handles.handle.close()
                raise
            self._handles.handle.seek(0)
        else:
            # Create workbook object with default optimized_write=True.
            try:
                self._book = Workbook(**engine_kwargs)
            except TypeError:
                self._handles.handle.close()
                raise

            if self.book.worksheets:
                self.book.remove(self.book.worksheets[0])

    @property
    def book(self) -> Workbook:
        # [FUNCTION DOC]
        # Read only property: the underlying openpyxl workbook, for engine specific
        #   access.
        """
        Book instance of class openpyxl.workbook.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    @property
    def sheets(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Read only property: builds and returns a new mapping from sheet name to
        #   worksheet object for all sheets in the workbook.
        """Mapping of sheet names to sheet objects."""
        result = {name: self.book[name] for name in self.book.sheetnames}
        return result

    def _save(self) -> None:
        # [FUNCTION DOC]
        # HELPER: implements the save hook. Saves the workbook to the output handle; in
        #   append (read/update) mode, unless the handle is a memory map, truncates the
        #   file at the current position so that leftover bytes of a longer previous
        #   file are removed. Returns nothing.
        """
        Save workbook to disk.
        """
        self.book.save(self._handles.handle)
        if "r+" in self._mode and not isinstance(self._handles.handle, mmap.mmap):
            # truncate file to the written content
            self._handles.handle.truncate()

    @classmethod
    def _convert_to_style_kwargs(
        cls, style_dict: dict[str, Serialisable]
    ) -> dict[str, Serialisable]:
        # [FUNCTION DOC]
        # HELPER: converts a style dictionary into keyword arguments for openpyxl cell
        #   style attributes. Input: dictionary whose keys are style groups (font, fill,
        #   border or its synonym borders, alignment, number_format, protection) and
        #   whose values are nested descriptions. Returns a new dictionary with
        #   normalized keys and native openpyxl style objects as values. For each group
        #   looks up the matching class method for that group; groups without a
        #   converter, or whose converted value is empty or false, are dropped.
        """
        Convert a style_dict to a set of kwargs suitable for initializing
        or updating-on-copy an openpyxl v2 style object.

        Parameters
        ----------
        style_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'font'
                'fill'
                'border' ('borders')
                'alignment'
                'number_format'
                'protection'

        Returns
        -------
        style_kwargs : dict
            A dict with the same, normalized keys as ``style_dict`` but each
            value has been replaced with a native openpyxl style object of the
            appropriate class.
        """
        _style_key_map = {"borders": "border"}

        style_kwargs: dict[str, Serialisable] = {}
        for k, v in style_dict.items():
            k = _style_key_map.get(k, k)
            _conv_to_x = getattr(cls, f"_convert_to_{k}", lambda x: None)
            new_v = _conv_to_x(v)
            if new_v:
                style_kwargs[k] = new_v

        return style_kwargs

    @classmethod
    def _convert_to_color(cls, color_spec):
        # [FUNCTION DOC]
        # HELPER: converts a color specification to an openpyxl color object. Input:
        #   either a string (an aarrggbb hex value) or a dictionary of color attributes
        #   (rgb, indexed, auto, theme, tint, index, type) passed as keyword arguments.
        #   Returns the color object.
        """
        Convert ``color_spec`` to an openpyxl v2 Color object.

        Parameters
        ----------
        color_spec : str, dict
            A 32-bit ARGB hex string, or a dict with zero or more of the
            following keys.
                'rgb'
                'indexed'
                'auto'
                'theme'
                'tint'
                'index'
                'type'

        Returns
        -------
        color : openpyxl.styles.Color
        """
        from openpyxl.styles import Color

        if isinstance(color_spec, str):
            return Color(color_spec)
        else:
            return Color(**color_spec)

    @classmethod
    def _convert_to_font(cls, font_dict):
        # [FUNCTION DOC]
        # HELPER: converts a font description dictionary to an openpyxl font object.
        #   Short key synonyms (sz, b, i, u, strike, vertalign) are renamed to the full
        #   names; the color entry is converted by the color helper; all other keys are
        #   passed through unchanged. Returns the font object. The library raises a type
        #   error for unknown keys.
        """
        Convert ``font_dict`` to an openpyxl v2 Font object.

        Parameters
        ----------
        font_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'name'
                'size' ('sz')
                'bold' ('b')
                'italic' ('i')
                'underline' ('u')
                'strikethrough' ('strike')
                'color'
                'vertAlign' ('vertalign')
                'charset'
                'scheme'
                'family'
                'outline'
                'shadow'
                'condense'

        Returns
        -------
        font : openpyxl.styles.Font
        """
        from openpyxl.styles import Font

        _font_key_map = {
            "sz": "size",
            "b": "bold",
            "i": "italic",
            "u": "underline",
            "strike": "strikethrough",
            "vertalign": "vertAlign",
        }

        font_kwargs = {}
        for k, v in font_dict.items():
            k = _font_key_map.get(k, k)
            if k == "color":
                v = cls._convert_to_color(v)
            font_kwargs[k] = v

        return Font(**font_kwargs)

    @classmethod
    def _convert_to_stop(cls, stop_seq):
        # [FUNCTION DOC]
        # HELPER: converts an iterable of color specifications into the stop colors for
        #   a gradient fill by applying the color helper to each item.
        # KNOWN ISSUE: the docstring says a list is returned, but the function returns a
        #   lazy mapping iterator that can be consumed only once.
        """
        Convert ``stop_seq`` to a list of openpyxl v2 Color objects,
        suitable for initializing the ``GradientFill`` ``stop`` parameter.

        Parameters
        ----------
        stop_seq : iterable
            An iterable that yields objects suitable for consumption by
            ``_convert_to_color``.

        Returns
        -------
        stop : list of openpyxl.styles.Color
        """
        return map(cls._convert_to_color, stop_seq)

    @classmethod
    def _convert_to_fill(cls, fill_dict: dict[str, Any]) -> Fill:
        # [FUNCTION DOC]
        # HELPER: converts a fill description dictionary to an openpyxl fill object.
        #   Keys are split into pattern fill keywords (with synonyms such as pattern
        #   type, foreground color and background color renamed to fill type, start
        #   color and end color, and colors converted) and gradient fill keywords (fill
        #   type renamed to type, stop converted to colors); keys that are neither are
        #   given to both. First tries to build a pattern fill; if the library rejects
        #   the keywords with a type error, builds a gradient fill from the gradient
        #   keywords instead. Returns the fill object.
        """
        Convert ``fill_dict`` to an openpyxl v2 Fill object.

        Parameters
        ----------
        fill_dict : dict
            A dict with one or more of the following keys (or their synonyms),
                'fill_type' ('patternType', 'patterntype')
                'start_color' ('fgColor', 'fgcolor')
                'end_color' ('bgColor', 'bgcolor')
            or one or more of the following keys (or their synonyms).
                'type' ('fill_type')
                'degree'
                'left'
                'right'
                'top'
                'bottom'
                'stop'

        Returns
        -------
        fill : openpyxl.styles.Fill
        """
        from openpyxl.styles import (
            GradientFill,
            PatternFill,
        )

        _pattern_fill_key_map = {
            "patternType": "fill_type",
            "patterntype": "fill_type",
            "fgColor": "start_color",
            "fgcolor": "start_color",
            "bgColor": "end_color",
            "bgcolor": "end_color",
        }

        _gradient_fill_key_map = {"fill_type": "type"}

        pfill_kwargs = {}
        gfill_kwargs = {}
        for k, v in fill_dict.items():
            pk = _pattern_fill_key_map.get(k)
            gk = _gradient_fill_key_map.get(k)
            if pk in ["start_color", "end_color"]:
                v = cls._convert_to_color(v)
            if gk == "stop":
                v = cls._convert_to_stop(v)
            if pk:
                pfill_kwargs[pk] = v
            elif gk:
                gfill_kwargs[gk] = v
            else:
                pfill_kwargs[k] = v
                gfill_kwargs[k] = v

        try:
            return PatternFill(**pfill_kwargs)
        except TypeError:
            return GradientFill(**gfill_kwargs)

    @classmethod
    def _convert_to_side(cls, side_spec):
        # [FUNCTION DOC]
        # HELPER: converts a border side specification to an openpyxl side object. A
        #   string is taken as the border style. A dictionary has the key border_style
        #   renamed to style and the color entry converted; the rest are passed through.
        #   Returns the side object.
        """
        Convert ``side_spec`` to an openpyxl v2 Side object.

        Parameters
        ----------
        side_spec : str, dict
            A string specifying the border style, or a dict with zero or more
            of the following keys (or their synonyms).
                'style' ('border_style')
                'color'

        Returns
        -------
        side : openpyxl.styles.Side
        """
        from openpyxl.styles import Side

        _side_key_map = {"border_style": "style"}

        if isinstance(side_spec, str):
            return Side(style=side_spec)

        side_kwargs = {}
        for k, v in side_spec.items():
            k = _side_key_map.get(k, k)
            if k == "color":
                v = cls._convert_to_color(v)
            side_kwargs[k] = v

        return Side(**side_kwargs)

    @classmethod
    def _convert_to_border(cls, border_dict):
        # [FUNCTION DOC]
        # HELPER: converts a border description dictionary to an openpyxl border object.
        #   Synonym keys for diagonal up and diagonal down are renamed; the left, right,
        #   top, bottom and diagonal entries are converted with the side helper; a color
        #   entry, if present, is converted with the color helper; other keys pass
        #   through. Returns the border object.
        """
        Convert ``border_dict`` to an openpyxl v2 Border object.

        Parameters
        ----------
        border_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'left'
                'right'
                'top'
                'bottom'
                'diagonal'
                'diagonal_direction'
                'vertical'
                'horizontal'
                'diagonalUp' ('diagonalup')
                'diagonalDown' ('diagonaldown')
                'outline'

        Returns
        -------
        border : openpyxl.styles.Border
        """
        from openpyxl.styles import Border

        _border_key_map = {"diagonalup": "diagonalUp", "diagonaldown": "diagonalDown"}

        border_kwargs = {}
        for k, v in border_dict.items():
            k = _border_key_map.get(k, k)
            if k == "color":
                v = cls._convert_to_color(v)
            if k in ["left", "right", "top", "bottom", "diagonal"]:
                v = cls._convert_to_side(v)
            border_kwargs[k] = v

        return Border(**border_kwargs)

    @classmethod
    def _convert_to_alignment(cls, alignment_dict):
        # [FUNCTION DOC]
        # HELPER: converts an alignment dictionary (horizontal, vertical, text rotation,
        #   wrap text, shrink to fit, indent) to an openpyxl alignment object by passing
        #   all entries as keyword arguments. Returns the alignment object.
        """
        Convert ``alignment_dict`` to an openpyxl v2 Alignment object.

        Parameters
        ----------
        alignment_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'horizontal'
                'vertical'
                'text_rotation'
                'wrap_text'
                'shrink_to_fit'
                'indent'
        Returns
        -------
        alignment : openpyxl.styles.Alignment
        """
        from openpyxl.styles import Alignment

        return Alignment(**alignment_dict)

    @classmethod
    def _convert_to_number_format(cls, number_format_dict):
        # [FUNCTION DOC]
        # HELPER: extracts the number format code string from the number format
        #   dictionary and returns it.
        # KNOWN ISSUE: the docstring says the dictionary may have zero or more keys, but
        #   the format code entry is required; a dictionary without it raises a key
        #   error.
        """
        Convert ``number_format_dict`` to an openpyxl v2.1.0 number format
        initializer.

        Parameters
        ----------
        number_format_dict : dict
            A dict with zero or more of the following keys.
                'format_code' : str

        Returns
        -------
        number_format : str
        """
        return number_format_dict["format_code"]

    @classmethod
    def _convert_to_protection(cls, protection_dict):
        # [FUNCTION DOC]
        # HELPER: converts a protection dictionary (locked, hidden) to an openpyxl
        #   protection object by passing entries as keyword arguments. Returns the
        #   protection object.
        # KNOWN ISSUE: the Returns section of the docstring is empty.
        """
        Convert ``protection_dict`` to an openpyxl v2 Protection object.

        Parameters
        ----------
        protection_dict : dict
            A dict with zero or more of the following keys.
                'locked'
                'hidden'

        Returns
        -------
        """
        from openpyxl.styles import Protection

        return Protection(**protection_dict)

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
        autofilter_range: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: implements the cell writing hook. Inputs: cells (iterable of cell
        #   objects with row, column, value, style and merge fields), sheet_name
        #   (default none means current sheet), startrow and startcol (zero based
        #   offsets), freeze_panes (optional pair of row and column counts) and
        #   autofilter_range (optional range string).
        # Behavior: resolves the target sheet. If the sheet already exists and the
        #   policy is not new: in append mode, replace deletes the old sheet and creates
        #   a fresh one at the same position, error raises a value error saying the
        #   sheet exists, overlay reuses the existing sheet; in write mode reuses the
        #   sheet. Otherwise creates a new sheet with the given title (openpyxl makes
        #   titles unique, which is how new works). If the freeze pane pair is valid,
        #   freezes at the cell one row and one column past the given counts. Then
        #   writes each cell at its offset position (one based): converted value from
        #   the base class helper, number format if any, and style attributes (converted
        #   once per distinct style and cached by the style's text form). Merged cells
        #   are merged over the stated range and, because only the top left cell is
        #   preserved, the style is applied to every other cell in the merged range.
        #   Finally sets the autofilter range if given.
        # KNOWN ISSUE: the final branch that raises a value error for an invalid policy
        #   inside the append mode handling cannot be reached, because the base class
        #   constructor already rejects invalid policies and the new policy is excluded
        #   by the enclosing condition.
        # Write the frame cells using openpyxl.
        sheet_name = self._get_sheet_name(sheet_name)

        _style_cache: dict[str, dict[str, Serialisable]] = {}

        if sheet_name in self.sheets and self._if_sheet_exists != "new":
            if "r+" in self._mode:
                if self._if_sheet_exists == "replace":
                    old_wks = self.sheets[sheet_name]
                    target_index = self.book.index(old_wks)
                    del self.book[sheet_name]
                    wks = self.book.create_sheet(sheet_name, target_index)
                elif self._if_sheet_exists == "error":
                    raise ValueError(
                        f"Sheet '{sheet_name}' already exists and "
                        f"if_sheet_exists is set to 'error'."
                    )
                elif self._if_sheet_exists == "overlay":
                    wks = self.sheets[sheet_name]
                else:
                    raise ValueError(
                        f"'{self._if_sheet_exists}' is not valid for if_sheet_exists. "
                        "Valid options are 'error', 'new', 'replace' and 'overlay'."
                    )
            else:
                wks = self.sheets[sheet_name]
        else:
            wks = self.book.create_sheet()
            wks.title = sheet_name

        if validate_freeze_panes(freeze_panes):
            freeze_panes = cast(tuple[int, int], freeze_panes)
            wks.freeze_panes = wks.cell(
                row=freeze_panes[0] + 1, column=freeze_panes[1] + 1
            )

        for cell in cells:
            xcell = wks.cell(
                row=startrow + cell.row + 1, column=startcol + cell.col + 1
            )
            xcell.value, fmt = self._value_with_fmt(cell.val)
            if fmt:
                xcell.number_format = fmt

            style_kwargs: dict[str, Serialisable] | None = {}
            if cell.style:
                key = str(cell.style)
                style_kwargs = _style_cache.get(key)
                if style_kwargs is None:
                    style_kwargs = self._convert_to_style_kwargs(cell.style)
                    _style_cache[key] = style_kwargs

            if style_kwargs:
                for k, v in style_kwargs.items():
                    setattr(xcell, k, v)

            if cell.mergestart is not None and cell.mergeend is not None:
                wks.merge_cells(
                    start_row=startrow + cell.row + 1,
                    start_column=startcol + cell.col + 1,
                    end_column=startcol + cell.mergeend + 1,
                    end_row=startrow + cell.mergestart + 1,
                )

                # When cells are merged only the top-left cell is preserved
                # The behaviour of the other cells in a merged range is
                # undefined
                if style_kwargs:
                    first_row = startrow + cell.row + 1
                    last_row = startrow + cell.mergestart + 1
                    first_col = startcol + cell.col + 1
                    last_col = startcol + cell.mergeend + 1

                    for row in range(first_row, last_row + 1):
                        for col in range(first_col, last_col + 1):
                            if row == first_row and col == first_col:
                                # Ignore first cell. It is already handled.
                                continue
                            xcell = wks.cell(column=col, row=row)
                            for k, v in style_kwargs.items():
                                setattr(xcell, k, v)

        if autofilter_range:
            wks.auto_filter.ref = autofilter_range


class OpenpyxlReader(BaseExcelReader["Workbook"]):
    # [CLASS DOC]
    # Reader for xlsx and xlsm workbooks using openpyxl. Loads workbooks in read only
    #   mode with cached formula values (formula results rather than formulas) and
    #   without external links, unless overridden by engine_kwargs. Overrides
    #   construction, workbook class, loader, sheet listing, lookups and the sheet to
    #   rows hook.
    @doc(storage_options=_shared_docs["storage_options"])
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Create the reader. Inputs: filepath_or_buffer, storage_options, engine_kwargs.
        #   Verifies that openpyxl is installed (the helper raises an import error
        #   otherwise) and runs the base constructor. The docstring is completed with
        #   the shared storage options text by the doc decorator.
        """
        Reader using openpyxl engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("openpyxl")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[Workbook]:
        # [FUNCTION DOC]
        # Returns the openpyxl workbook class (imported lazily); used by the base class
        #   to recognize already loaded workbooks.
        from openpyxl import Workbook

        return Workbook

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> Workbook:
        # [FUNCTION DOC]
        # Loads a workbook with openpyxl. Default options are read only true, data only
        #   true (cached values rather than formulas) and keep links false;
        #   engine_kwargs override these defaults by key. Returns the workbook.
        from openpyxl import load_workbook

        default_kwargs = {"read_only": True, "data_only": True, "keep_links": False}

        return load_workbook(
            filepath_or_buffer,
            **(default_kwargs | engine_kwargs),
        )

    @property
    def sheet_names(self) -> list[str]:
        # [FUNCTION DOC]
        # Read only property: the titles of all worksheets (chart sheets are not
        #   included, so positions count only worksheets).
        return [sheet.title for sheet in self.book.worksheets]

    def get_sheet_by_name(self, name: str):
        # [FUNCTION DOC]
        # Returns the sheet with the given name after the base class name check (raising
        #   a value error for unknown names).
        self.raise_if_bad_sheet_by_name(name)
        return self.book[name]

    def get_sheet_by_index(self, index: int):
        # [FUNCTION DOC]
        # Returns the worksheet at the given zero based position after the base class
        #   position check (raising a value error when out of range).
        self.raise_if_bad_sheet_by_index(index)
        return self.book.worksheets[index]

    def _convert_cell(self, cell) -> Scalar:
        # [FUNCTION DOC]
        # HELPER: converts an openpyxl cell to a Python value. A cell without a value
        #   becomes the empty string (for compatibility with xlrd). Error cells become
        #   not a number. Numeric cells that are whole become integers, other numbers
        #   become floats. Everything else is returned as is.
        from openpyxl.cell.cell import (
            TYPE_ERROR,
            TYPE_NUMERIC,
        )

        if cell.value is None:
            return ""  # compat with xlrd
        elif cell.data_type == TYPE_ERROR:
            return np.nan
        elif cell.data_type == TYPE_NUMERIC:
            val = int(cell.value)
            if val == cell.value:
                return val
            return float(cell.value)

        return cell.value

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # Converts a worksheet to a list of equal length rows. Input: sheet and
        #   file_rows_needed (default none; reading stops once at least that many rows,
        #   including empty rows, have been collected). Returns the list of rows.
        # Behavior: in read only mode first resets the sheet's stored dimensions (the
        #   stored dimensions can be wrong, which would truncate the rows). Each row is
        #   converted cell by cell with trailing empty strings trimmed; all rows are
        #   collected but trailing completely empty rows are dropped; remaining rows are
        #   padded with empty strings to the widest row.
        if self.book.read_only:
            sheet.reset_dimensions()

        data: list[list[Scalar]] = []
        last_row_with_data = -1
        for row_number, row in enumerate(sheet.rows):
            converted_row = [self._convert_cell(cell) for cell in row]
            while converted_row and converted_row[-1] == "":
                # trim trailing empty elements
                converted_row.pop()
            if converted_row:
                last_row_with_data = row_number
            data.append(converted_row)
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break

        # Trim trailing empty rows
        data = data[: last_row_with_data + 1]

        if len(data) > 0:
            # extend rows to max width
            max_width = max(len(data_row) for data_row in data)
            if min(len(data_row) for data_row in data) < max_width:
                empty_cell: list[Scalar] = [""]
                data = [
                    data_row + (max_width - len(data_row)) * empty_cell
                    for data_row in data
                ]

        return data
