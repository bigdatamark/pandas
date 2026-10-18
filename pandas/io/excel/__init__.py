# [PACKAGE DOC]
# Excel reading and writing package. Submodules: the base module (reader and
# writer base classes and the public read function), engine specific reader and
# writer modules (openpyxl, xlsxwriter, odf spreadsheet, xlrd, pyxlsb, calamine)
# and a utilities module with the writer registry.
# Exports (public all): ExcelFile, ExcelWriter and read_excel.
# Import-time side effects: registers the openpyxl, xlsxwriter and ODS writer
# classes in the writer registry so they can be selected by engine name or file
# extension. Readers are registered lazily elsewhere.
from pandas.io.excel._base import (
    ExcelFile,
    ExcelWriter,
    read_excel,
)
from pandas.io.excel._odswriter import ODSWriter as _ODSWriter
from pandas.io.excel._openpyxl import OpenpyxlWriter as _OpenpyxlWriter
from pandas.io.excel._util import register_writer
from pandas.io.excel._xlsxwriter import XlsxWriter as _XlsxWriter

__all__ = ["ExcelFile", "ExcelWriter", "read_excel"]


register_writer(_OpenpyxlWriter)

register_writer(_XlsxWriter)


register_writer(_ODSWriter)
