# [MODULE DOC]
# Excel reader engine for OpenDocument spreadsheets (ods and odf files, also odt) built
#   on the optional odfpy library.
# Public contents: ODFReader, a subclass of the generic reader base class from the base
#   module, registered in the ExcelFile engine table under the name odf. All odfpy
#   imports are done lazily inside methods.
# Import-time side effects: imports the top level pandas package and the shared
#   docstring templates (used to fill the storage_options text through the doc
#   decorator).

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._typing import (
    FilePath,
    ReadBuffer,
    Scalar,
    StorageOptions,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

import pandas as pd
from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from odf.opendocument import OpenDocument

    from pandas._libs.tslibs.nattype import NaTType


@doc(storage_options=_shared_docs["storage_options"])
class ODFReader(BaseExcelReader["OpenDocument"]):
    # [CLASS DOC]
    # Concrete reader for OpenDocument files. The workbook type is the odfpy document
    #   object. Implements the construction, workbook class, loader, sheet listing,
    #   sheet lookup and sheet-to-rows hooks, and private helpers that decode repeated
    #   rows and columns, typed cell values and run-length encoded spaces. The class
    #   docstring placeholder for storage options is filled in by the doc decorator.
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Create the reader. Inputs: filepath_or_buffer (path or readable binary
        #   stream), storage_options, engine_kwargs (passed to the document loader).
        #   Verifies the odfpy dependency is installed (the helper raises an import
        #   error otherwise), then runs the base constructor which opens and loads the
        #   document.
        """
        Read tables out of OpenDocument formatted files.

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("odf")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[OpenDocument]:
        # [FUNCTION DOC]
        # Returns the odfpy document class (imported lazily); used by the base class to
        #   recognize already loaded documents.
        from odf.opendocument import OpenDocument

        return OpenDocument

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> OpenDocument:
        # [FUNCTION DOC]
        # Loads an OpenDocument file from a path or binary stream using the odfpy
        #   loader, passing engine_kwargs as keyword arguments. Returns the document.
        from odf.opendocument import load

        return load(filepath_or_buffer, **engine_kwargs)

    @property
    def empty_value(self) -> str:
        # [FUNCTION DOC]
        # Read only property kept for compatibility with other readers: the value that
        #   represents an empty cell, which is the empty string.
        """Property for compat with other readers."""
        return ""

    @property
    def sheet_names(self) -> list[str]:
        # [FUNCTION DOC]
        # Read only property: the names of all tables (sheets) in the document, in
        #   document order, taken from each table's name attribute.
        """Return a list of sheet names present in the document"""
        from odf.table import Table

        tables = self.book.getElementsByType(Table)
        return [t.getAttribute("name") for t in tables]

    def get_sheet_by_index(self, index: int):
        # [FUNCTION DOC]
        # Returns the table at the given zero based position. Raises a value error (via
        #   the base class check) if the position is not smaller than the number of
        #   tables.
        from odf.table import Table

        self.raise_if_bad_sheet_by_index(index)
        tables = self.book.getElementsByType(Table)
        return tables[index]

    def get_sheet_by_name(self, name: str):
        # [FUNCTION DOC]
        # Returns the table with the given name. First applies the base class name
        #   check, which raises a value error if the name is absent.
        # KNOWN ISSUE: the code after the base class check closes the reader and raises
        #   a second value error when no table matches; since the check already
        #   guarantees the name is present, that tail is effectively unreachable.
        from odf.table import Table

        self.raise_if_bad_sheet_by_name(name)
        tables = self.book.getElementsByType(Table)

        for table in tables:
            if table.getAttribute("name") == name:
                return table

        self.close()
        raise ValueError(f"sheet {name} not found")

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar | NaTType]]:
        # [FUNCTION DOC]
        # Converts one table into a list of equal length rows of Python values. Input:
        #   sheet is an odfpy table; file_rows_needed (default none) stops reading once
        #   at least that many rows have been produced. Returns the list of rows.
        # Behavior: iterates over table rows and, within each row, over cells and
        #   covered cells (covered cells, produced by merges, yield the empty value).
        #   Each cell is repeated according to its column repeat count. Runs of empty
        #   cells are held back and only written when a later non-empty cell follows, so
        #   trailing empties are dropped. Rows are repeated according to their row
        #   repeat count; rows that are completely empty are queued and only emitted (as
        #   one-cell blank rows) if a later non-empty row appears. The longest row
        #   length seen is tracked and all rows are finally padded with the empty value
        #   to that width.
        """
        Parse an ODF Table into a list of lists
        """
        from odf.table import (
            CoveredTableCell,
            TableCell,
            TableRow,
        )

        covered_cell_name = CoveredTableCell().qname
        table_cell_name = TableCell().qname
        cell_names = {covered_cell_name, table_cell_name}

        sheet_rows = sheet.getElementsByType(TableRow)
        empty_rows = 0
        max_row_len = 0

        table: list[list[Scalar | NaTType]] = []

        for sheet_row in sheet_rows:
            empty_cells = 0
            table_row: list[Scalar | NaTType] = []

            for sheet_cell in sheet_row.childNodes:
                if hasattr(sheet_cell, "qname") and sheet_cell.qname in cell_names:
                    if sheet_cell.qname == table_cell_name:
                        value = self._get_cell_value(sheet_cell)
                    else:
                        value = self.empty_value

                    column_repeat = self._get_column_repeat(sheet_cell)

                    # Queue up empty values, writing only if content succeeds them
                    if value == self.empty_value:
                        empty_cells += column_repeat
                    else:
                        table_row.extend([self.empty_value] * empty_cells)
                        empty_cells = 0
                        table_row.extend([value] * column_repeat)

            if max_row_len < len(table_row):
                max_row_len = len(table_row)

            row_repeat = self._get_row_repeat(sheet_row)
            if len(table_row) == 0:
                empty_rows += row_repeat
            else:
                # add blank rows to our table
                table.extend([[self.empty_value]] * empty_rows)
                empty_rows = 0
                table.extend(table_row for _ in range(row_repeat))
            if file_rows_needed is not None and len(table) >= file_rows_needed:
                break

        # Make our table square
        for row in table:
            if len(row) < max_row_len:
                row.extend([self.empty_value] * (max_row_len - len(row)))

        return table

    def _get_row_repeat(self, row) -> int:
        # [FUNCTION DOC]
        # HELPER: reads how many times a row is repeated from the row's repeat
        #   attribute, defaulting to one. Returns an integer. Repeating empty rows is
        #   how the format stores sparse sheets.
        """
        Return number of times this row was repeated
        Repeating an empty row appeared to be a common way
        of representing sparse rows in the table.
        """
        from odf.namespaces import TABLENS

        return int(row.attributes.get((TABLENS, "number-rows-repeated"), 1))

    def _get_column_repeat(self, cell) -> int:
        # [FUNCTION DOC]
        # HELPER: reads how many times a cell is repeated horizontally from the cell's
        #   column repeat attribute, defaulting to one. Returns an integer.
        from odf.namespaces import TABLENS

        return int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))

    def _get_cell_value(self, cell) -> Scalar | NaTType:
        # [FUNCTION DOC]
        # HELPER: converts one cell into a Python value based on its declared value
        #   type. A cell whose text is the not available marker gives a missing value.
        #   Boolean cells give true only when the text is the upper case word for true.
        #   Cells with no type give the empty value. Floats become integers when whole,
        #   otherwise floats. Percentage and currency cells give floats. String cells
        #   are decoded by the string helper. Date cells give a timestamp from the date
        #   value attribute. Time cells give a time of day parsed from the cell text.
        #   Any other type closes the reader and raises a value error naming the
        #   unrecognized type.
        from odf.namespaces import OFFICENS

        if str(cell) == "#N/A":
            return np.nan

        cell_type = cell.attributes.get((OFFICENS, "value-type"))
        if cell_type == "boolean":
            if str(cell) == "TRUE":
                return True
            return False
        if cell_type is None:
            return self.empty_value
        elif cell_type == "float":
            # GH5394
            cell_value = float(cell.attributes.get((OFFICENS, "value")))
            val = int(cell_value)
            if val == cell_value:
                return val
            return cell_value
        elif cell_type == "percentage":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        elif cell_type == "string":
            return self._get_cell_string_value(cell)
        elif cell_type == "currency":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        elif cell_type == "date":
            cell_value = cell.attributes.get((OFFICENS, "date-value"))
            return pd.Timestamp(cell_value)
        elif cell_type == "time":
            stamp = pd.Timestamp(str(cell))
            # cast needed here because Scalar doesn't include datetime.time
            return cast(Scalar, stamp.time())
        else:
            self.close()
            raise ValueError(f"Unrecognized type {cell_type}")

    def _get_cell_string_value(self, cell) -> str:
        # [FUNCTION DOC]
        # HELPER: extracts the text of a cell or nested element. Walks the child nodes:
        #   elements that encode a run of spaces are expanded to that many spaces
        #   (default one), annotation elements are skipped, other elements are processed
        #   recursively (needed for nested fragments), and plain text nodes have newline
        #   characters stripped from both ends. Returns the concatenation of the pieces.
        """
        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        """
        from odf.element import Element
        from odf.namespaces import TEXTNS
        from odf.office import Annotation
        from odf.text import S

        office_annotation = Annotation().qname
        text_s = S().qname

        value = []

        for fragment in cell.childNodes:
            if isinstance(fragment, Element):
                if fragment.qname == text_s:
                    spaces = int(fragment.attributes.get((TEXTNS, "c"), 1))
                    value.append(" " * spaces)
                elif fragment.qname == office_annotation:
                    continue
                else:
                    # recursive impl needed in case of nested fragments
                    # with multiple spaces
                    # https://github.com/pandas-dev/pandas/pull/36175#discussion_r484639704
                    value.append(self._get_cell_string_value(fragment))
            else:
                value.append(str(fragment).strip("\n"))
        return "".join(value)
