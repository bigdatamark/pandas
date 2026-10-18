# [MODULE DOC]
# Excel reader engine for old style xls workbooks built on the optional xlrd library
#   (version 2.0.1 or newer).
# Public contents: XlrdReader, a subclass of the generic reader base class from the base
#   module, registered in the ExcelFile engine table under the name xlrd.
# Import-time side effects: none; library imports happen lazily inside methods.

from __future__ import annotations

from datetime import time
import math
from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from xlrd import Book

    from pandas._typing import (
        Scalar,
        StorageOptions,
    )


class XlrdReader(BaseExcelReader["Book"]):
    # [CLASS DOC]
    # Concrete reader for xls files. The workbook type is the xlrd book object.
    #   Overrides construction, workbook class, loader, sheet names and lookups, and the
    #   sheet-to-rows hook that converts typed cells to Python values.
    def __init__(
        self,
        filepath_or_buffer,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Create the reader. Inputs: filepath_or_buffer (path, stream, or xlrd book),
        #   storage_options, engine_kwargs (passed to the library's open function).
        #   Verifies that xlrd is installed, with a custom message asking for version
        #   2.0.1 or newer for xls support (the helper raises an import error), then
        #   runs the base constructor.
        """
        Reader using xlrd engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection,
            e.g. host, port, username, password, etc. For HTTP(S) URLs the
            key-value pairs are forwarded to ``urllib.request.Request`` as
            header options. For other URLs (e.g. starting with "s3://", and
            "gcs://") the key-value pairs are forwarded to ``fsspec.open``.
            Please see ``fsspec`` and ``urllib`` for more details, and for more
            examples on storage options refer `here <https://pandas.pydata.org/
            pandas-docs/stable/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`__.
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        err_msg = "Install xlrd >= 2.0.1 for xls Excel support"
        import_optional_dependency("xlrd", extra=err_msg)
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[Book]:
        # [FUNCTION DOC]
        # Returns the xlrd book class (imported lazily); used by the base class to
        #   recognize already loaded books.
        from xlrd import Book

        return Book

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> Book:
        # [FUNCTION DOC]
        # Loads a workbook. If the input has a read method its entire content is read
        #   and passed to the library as file contents; otherwise the input is treated
        #   as a path. engine_kwargs are passed through as keyword arguments. Returns
        #   the book.
        from xlrd import open_workbook

        if hasattr(filepath_or_buffer, "read"):
            data = filepath_or_buffer.read()
            return open_workbook(file_contents=data, **engine_kwargs)
        else:
            return open_workbook(filepath_or_buffer, **engine_kwargs)

    @property
    def sheet_names(self):
        # [FUNCTION DOC]
        # Read only property: the list of sheet names, from the book.
        return self.book.sheet_names()

    def get_sheet_by_name(self, name):
        # [FUNCTION DOC]
        # Returns the sheet with the given name after the base class name check (which
        #   raises a value error for unknown names).
        self.raise_if_bad_sheet_by_name(name)
        return self.book.sheet_by_name(name)

    def get_sheet_by_index(self, index):
        # [FUNCTION DOC]
        # Returns the sheet at the given zero based position after the base class
        #   position check (which raises a value error for out of range positions).
        self.raise_if_bad_sheet_by_index(index)
        return self.book.sheet_by_index(index)

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # Converts a sheet to a list of rows of Python values. Input: sheet and
        #   file_rows_needed (default none; limits the number of rows read to the
        #   smaller of the sheet row count and this limit). Returns one list per row,
        #   built by converting each value according to its xlrd cell type; the value
        #   and type sequences must be of equal length or a value error is raised by the
        #   strict pairing. Uses the book's date mode to know whether the 1900 or the
        #   1904 epoch applies.
        from xlrd import (
            XL_CELL_BOOLEAN,
            XL_CELL_DATE,
            XL_CELL_ERROR,
            XL_CELL_NUMBER,
            xldate,
        )

        epoch1904 = self.book.datemode

        def _parse_cell(cell_contents, cell_typ):
            # [FUNCTION DOC]
            # HELPER: nested cell converter. Inputs: the raw cell contents and the xlrd
            #   cell type code. Date cells are converted to datetimes (on an overflow
            #   error the raw number is returned); datetimes that fall on the epoch day
            #   itself (the last day of 1899 for the 1900 system or the first day of
            #   1904 for the 1904 system) are turned into a time of day because Excel
            #   does not distinguish times from dates. Error cells become not a number.
            #   Boolean cells become booleans. Number cells that are finite and whole
            #   become integers (non-finite numbers are left as they are). Other types
            #   are returned unchanged.
            """
            converts the contents of the cell into a pandas appropriate object
            """
            if cell_typ == XL_CELL_DATE:
                # Use the newer xlrd datetime handling.
                try:
                    cell_contents = xldate.xldate_as_datetime(cell_contents, epoch1904)
                except OverflowError:
                    return cell_contents

                # Excel doesn't distinguish between dates and time,
                # so we treat dates on the epoch as times only.
                # Also, Excel supports 1900 and 1904 epochs.
                year = (cell_contents.timetuple())[0:3]
                if (not epoch1904 and year == (1899, 12, 31)) or (
                    epoch1904 and year == (1904, 1, 1)
                ):
                    cell_contents = time(
                        cell_contents.hour,
                        cell_contents.minute,
                        cell_contents.second,
                        cell_contents.microsecond,
                    )

            elif cell_typ == XL_CELL_ERROR:
                cell_contents = np.nan
            elif cell_typ == XL_CELL_BOOLEAN:
                cell_contents = bool(cell_contents)
            elif cell_typ == XL_CELL_NUMBER:
                # GH5394 - Excel 'numbers' are always floats
                # it's a minimal perf hit and less surprising
                if math.isfinite(cell_contents):
                    # GH54564 - don't attempt to convert NaN/Inf
                    val = int(cell_contents)
                    if val == cell_contents:
                        cell_contents = val
            return cell_contents

        nrows = sheet.nrows
        if file_rows_needed is not None:
            nrows = min(nrows, file_rows_needed)
        return [
            [
                _parse_cell(value, typ)
                for value, typ in zip(
                    sheet.row_values(i), sheet.row_types(i), strict=True
                )
            ]
            for i in range(nrows)
        ]
