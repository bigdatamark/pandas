# [MODULE DOC]
# Excel reader engine backed by the optional third party calamine library; supports
#   xlsx, xls, xlsb and ods workbooks.
# Public contents: CalamineReader (subclass of the generic reader base class from the
#   base module, registered in the ExcelFile engine table under the name calamine). A
#   private type alias lists the cell value types the library can return (numbers, text,
#   booleans, times, dates, datetimes and time deltas).
# Import-time side effects: none; the library is imported lazily inside methods (type
#   checking imports only).

from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)

from pandas.compat._optional import import_optional_dependency

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from python_calamine import (
        CalamineSheet,
        CalamineWorkbook,
    )

    from pandas._typing import (
        FilePath,
        NaTType,
        ReadBuffer,
        Scalar,
        StorageOptions,
    )

_CellValue: TypeAlias = int | float | str | bool | time | date | datetime | timedelta


class CalamineReader(BaseExcelReader["CalamineWorkbook"]):
    # [CLASS DOC]
    # Concrete reader for the calamine engine. The workbook type is the library's
    #   workbook object. Overrides the construction, workbook class, loader, sheet name
    #   listing, sheet lookup and sheet-to-rows hooks of the base reader; all parsing
    #   beyond turning a sheet into rows is inherited.
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Create the reader. Inputs: filepath_or_buffer (path or readable binary
        #   stream), storage_options (for remote paths), engine_kwargs (extra options
        #   handed to the library's workbook loader).
        # Behavior: first verifies that the calamine library is installed (the optional
        #   dependency helper raises an import error if it is missing), then runs the
        #   base constructor which opens the input and loads the workbook.
        """
        Reader using calamine engine (xlsx/xls/xlsb/ods).

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("python_calamine")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[CalamineWorkbook]:
        # [FUNCTION DOC]
        # Returns the library's workbook class (imported lazily), used by the base class
        #   to recognize already loaded workbooks.
        from python_calamine import CalamineWorkbook

        return CalamineWorkbook

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs: Any
    ) -> CalamineWorkbook:
        # [FUNCTION DOC]
        # Loads a workbook from a path or binary stream using the library's loader,
        #   passing the engine_kwargs mapping through as keyword arguments. Returns the
        #   workbook object.
        from python_calamine import load_workbook

        return load_workbook(
            filepath_or_buffer,
            **engine_kwargs,
        )

    @property
    def sheet_names(self) -> list[str]:
        # [FUNCTION DOC]
        # Read only property: names of the worksheets in the workbook, in document
        #   order. Only entries whose metadata type is a regular worksheet are included,
        #   so chart sheets and similar are skipped (which is why positions count only
        #   worksheets).
        from python_calamine import SheetTypeEnum

        return [
            sheet.name
            for sheet in self.book.sheets_metadata
            if sheet.typ == SheetTypeEnum.WorkSheet
        ]

    def get_sheet_by_name(self, name: str) -> CalamineSheet:
        # [FUNCTION DOC]
        # Looks up a worksheet by name. Raises a value error (via the base class check)
        #   if the name is unknown; otherwise returns the library's sheet object.
        self.raise_if_bad_sheet_by_name(name)
        return self.book.get_sheet_by_name(name)

    def get_sheet_by_index(self, index: int) -> CalamineSheet:
        # [FUNCTION DOC]
        # Looks up a worksheet by zero based position. Raises a value error (via the
        #   base class check) if the position is not smaller than the number of
        #   worksheets; otherwise returns the library's sheet object.
        self.raise_if_bad_sheet_by_index(index)
        return self.book.get_sheet_by_index(index)

    def get_sheet_data(
        self, sheet: CalamineSheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar | NaTType | time]]:
        # [FUNCTION DOC]
        # Converts a sheet to a list of rows of Python values. Input: sheet is the
        #   library's sheet object; file_rows_needed (default none) limits how many rows
        #   are requested from the library. Returns a list of rows, each a list of
        #   converted cell values.
        # Behavior: asks the library for Python values without skipping empty leading
        #   areas, so row and column offsets are preserved, then converts each cell with
        #   the nested converter and returns the new nested list.
        def _convert_cell(value: _CellValue) -> Scalar | NaTType | time:
            # [FUNCTION DOC]
            # HELPER: nested cell converter for the sheet-to-rows method. Input: one
            #   cell value. Floats whose value is a whole number become integers, other
            #   floats are unchanged. Datetimes and time deltas are returned unchanged;
            #   plain dates are converted to midnight datetimes (to match what the
            #   openpyxl engine returns); times and all remaining values are returned
            #   unchanged.
            # KNOWN ISSUE: the explicit branch for time values returns the value
            #   unchanged, exactly like the final fallthrough return, so that branch is
            #   redundant. Also, converting a non-finite float to an integer would raise
            #   an error, since there is no finiteness check as the xlrd engine has.
            if isinstance(value, float):
                val = int(value)
                if val == value:
                    return val
                else:
                    return value
            elif isinstance(value, (datetime, timedelta)):
                # Return as-is to match openpyxl behavior (GH#59186)
                return value
            elif isinstance(value, date):
                # Convert date to datetime to match openpyxl behavior (GH#59186)
                return datetime(value.year, value.month, value.day)
            elif isinstance(value, time):
                return value

            return value

        rows: list[list[_CellValue]] = sheet.to_python(
            skip_empty_area=False, nrows=file_rows_needed
        )
        data = [[_convert_cell(cell) for cell in row] for row in rows]

        return data
