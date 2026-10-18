# [MODULE DOC]
# Tests for the xlrd based Excel reader (legacy xls format only).
# The whole module is skipped at import time when xlrd is not installed.
# Covers reading from an already opened xlrd workbook, rejection of xlsx files
# by xlrd, missing cells in xls files, and sniffing of old xls file headers.
# Relies on the datapath fixture and the excel reader entry points of the io
# package.
import io

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format

xlrd = pytest.importorskip("xlrd")


@pytest.fixture
def read_ext_xlrd():
    # [FUNCTION DOC]
    # Fixture returning the single file extension readable by the xlrd engine
    # in this module: the legacy xls extension (string with a leading dot).
    # Takes no inputs and has no side effects.
    """
    Valid extensions for reading Excel files with xlrd.

    Similar to read_ext, but excludes .ods, .xlsb, and for xlrd>2 .xlsx, .xlsm
    """
    return ".xls"


def test_read_xlrd_book(read_ext_xlrd, datapath):
    # [FUNCTION DOC]
    # Verifies that reading a sheet from an ExcelFile wrapping an opened xlrd
    # book gives the same frame as passing the book directly to the reader
    # with the xlrd engine, using the first column as index.
    # Setup: opens the legacy test workbook located via datapath; the book is
    # closed by the context manager. Expected: the two frames are equal.
    engine = "xlrd"
    sheet_name = "Sheet1"
    pth = datapath("io", "data", "excel", "test1.xls")
    with xlrd.open_workbook(pth) as book:
        with ExcelFile(book, engine=engine) as xl:
            result = pd.read_excel(xl, sheet_name=sheet_name, index_col=0)

        expected = pd.read_excel(
            book, sheet_name=sheet_name, engine=engine, index_col=0
        )
    tm.assert_frame_equal(result, expected)


def test_read_xlsx_fails(datapath):
    # [FUNCTION DOC]
    # Regression test (GH 29375): forcing the xlrd engine on an xlsx file must
    # raise xlrd's format error whose message says that xlsx files are not
    # supported. Input is the xlsx test workbook located via datapath.
    # GH 29375
    from xlrd.biffh import XLRDError

    path = datapath("io", "data", "excel", "test1.xlsx")
    with pytest.raises(XLRDError, match="Excel xlsx file; not supported"):
        pd.read_excel(path, engine="xlrd")


def test_nan_in_xls(datapath):
    # [FUNCTION DOC]
    # Regression test (GH 54564): an xls file with an empty cell in the second
    # column is read without header; the expected frame has an int64 first
    # column (0, 2) and a float second column (1, missing). Compared with the
    # frame-equality helper.
    # GH 54564
    path = datapath("io", "data", "excel", "test6.xls")

    expected = pd.DataFrame({0: np.r_[0, 2].astype("int64"), 1: np.r_[1, np.nan]})

    result = pd.read_excel(path, header=None)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_header",
    [
        b"\x09\x00\x04\x00\x07\x00\x10\x00",
        b"\x09\x02\x06\x00\x00\x00\x10\x00",
        b"\x09\x04\x06\x00\x00\x00\x10\x00",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    ],
)
def test_read_old_xls_files(file_header):
    # [FUNCTION DOC]
    # Regression test (GH 41226): format sniffing recognises several old BIFF
    # style and OLE compound document byte headers as the xls format.
    # Parametrized over four raw header byte strings; each is wrapped in an
    # in-memory binary buffer and the detected format must equal the xls tag.
    # GH 41226
    f = io.BytesIO(file_header)
    assert inspect_excel_format(f) == "xls"
