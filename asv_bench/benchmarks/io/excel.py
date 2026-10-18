# [MODULE DOC]
# Benchmarks for spreadsheet writing and reading: writing plain and styled frames
# to an in-memory workbook with two writer engines, and reading xlsx and ods files
# (full and limited rows). Public contents are the helper function
# _generate_dataframe and the benchmark classes WriteExcel, WriteExcelStyled,
# ReadExcel and ReadExcelNRows. Requires the optional OpenDocument library at
# import time (the module imports its spreadsheet, table and text classes
# unconditionally, so import fails if it is not installed). Final line
# re-exports the common asv setup hook.
from io import BytesIO

import numpy as np
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import (
    Table,
    TableCell,
    TableRow,
)
from odf.text import P

from pandas import (
    DataFrame,
    ExcelWriter,
    Index,
    date_range,
    read_excel,
)


def _generate_dataframe():
    # [FUNCTION DOC]
    # HELPER: data factory shared by the spreadsheet benchmark classes in this
    # module. Takes no input. Returns a frame of 2000 rows indexed by hourly
    # timestamps from 2000-01-01, with five random normal float columns named
    # float0 to float4 and an object column named object holding distinct strings.
    N = 2000
    C = 5
    df = DataFrame(
        np.random.randn(N, C),
        columns=[f"float{i}" for i in range(C)],
        index=date_range("20000101", periods=N, freq="h"),
    )
    df["object"] = Index([f"i-{i}" for i in range(N)], dtype=object)
    return df


class WriteExcel:
    # [CLASS DOC]
    # Benchmark of writing a frame to an in-memory workbook with each writer
    # engine. Parametrized by engine (openpyxl, xlsxwriter). State: df.
    params = ["openpyxl", "xlsxwriter"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Creates the shared sample frame as df. The engine parameter is unused
        # here.
        self.df = _generate_dataframe()

    def time_write_excel(self, engine):
        # [FUNCTION DOC]
        # Times writing the frame to a sheet named Sheet1 of an in-memory workbook
        # using the writer engine inside a context manager. The initial seek on the
        # fresh buffer has no effect.
        bio = BytesIO()
        bio.seek(0)
        with ExcelWriter(bio, engine=engine) as writer:
            self.df.to_excel(writer, sheet_name="Sheet1")


class WriteExcelStyled:
    # [CLASS DOC]
    # Benchmark of writing a styled frame to an in-memory workbook with each
    # writer engine. Parametrized by engine. State: df.
    params = ["openpyxl", "xlsxwriter"]
    param_names = ["engine"]

    def setup(self, engine):
        # [FUNCTION DOC]
        # Creates the shared sample frame as df. The engine parameter is unused
        # here.
        self.df = _generate_dataframe()

    def time_write_excel_style(self, engine):
        # [FUNCTION DOC]
        # Times building a styler for the frame, applying three element-wise style
        # rules (a red solid border everywhere, blue text everywhere, and a
        # green/black border color on the float1 column only), and writing it to a
        # sheet named Sheet1 of an in-memory workbook with the engine. Style
        # construction is part of the timing.
        bio = BytesIO()
        bio.seek(0)
        with ExcelWriter(bio, engine=engine) as writer:
            df_style = self.df.style
            df_style.map(lambda x: "border: red 1px solid;")
            df_style.map(lambda x: "color: blue")
            df_style.map(lambda x: "border-color: green black", subset=["float1"])
            df_style.to_excel(writer, sheet_name="Sheet1")


class ReadExcel:
    # [CLASS DOC]
    # Benchmark of reading a spreadsheet file written once by the cache hook, with
    # the xlsx reader engine or the OpenDocument reader engine. Parametrized by
    # engine (openpyxl, odf). Class attributes fname_excel and fname_odf give the
    # file names (written to the working directory and never removed; this class
    # does not use the file-cleanup base class).
    params = ["openpyxl", "odf"]
    param_names = ["engine"]
    fname_excel = "spreadsheet.xlsx"
    fname_odf = "spreadsheet.ods"

    def _create_odf(self):
        # [FUNCTION DOC]
        # HELPER: used by the cache hook to build the OpenDocument file. Reads the
        # frame stored on the instance, creates a spreadsheet with one table named
        # Table1, writes each row of the frame values as a table row with one text
        # cell per value (all cells typed as string, without header or index), and
        # saves it to the ods file name (side effect: file I/O).
        doc = OpenDocumentSpreadsheet()
        table = Table(name="Table1")
        for row in self.df.values:
            tr = TableRow()
            for val in row:
                tc = TableCell(valuetype="string")
                tc.addElement(P(text=val))
                tr.addElement(tc)
            table.addElement(tr)

        doc.spreadsheet.addElement(table)
        doc.save(self.fname_odf)

    def setup_cache(self):
        # [FUNCTION DOC]
        # asv cache hook run once. Creates the shared sample frame as df, writes it
        # to the xlsx file in a sheet named Sheet1, and then writes the ods file
        # with the helper. Returns nothing. Side effect: creates both files in the
        # working directory.
        self.df = _generate_dataframe()

        self.df.to_excel(self.fname_excel, sheet_name="Sheet1")
        self._create_odf()

    def time_read_excel(self, engine):
        # [FUNCTION DOC]
        # Times reading the whole file: the ods file for the odf engine, otherwise
        # the xlsx file, with the engine.
        if engine == "odf":
            fname = self.fname_odf
        else:
            fname = self.fname_excel
        read_excel(fname, engine=engine)


class ReadExcelNRows(ReadExcel):
    # [CLASS DOC]
    # Variant of the spreadsheet read benchmark that reads only the first ten
    # rows. Inherits parameters, file names and the cache hook from the parent
    # class and overrides the timing method.
    def time_read_excel(self, engine):
        # [FUNCTION DOC]
        # Times reading only the first ten data rows of the file chosen by engine
        # (ods for the odf engine, otherwise xlsx).
        if engine == "odf":
            fname = self.fname_odf
        else:
            fname = self.fname_excel
        read_excel(fname, engine=engine, nrows=10)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
