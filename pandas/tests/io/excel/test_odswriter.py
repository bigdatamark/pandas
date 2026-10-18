# [MODULE DOC]
# Tests for the odf-based writer of OpenDocument spreadsheets: append-mode rejection, engine keyword handling, sheet tracking and the value type attributes written to cells.
# Skipped at import time when the odf library is missing. Defines local fixtures for the file extension and a temporary file path.

from datetime import (
    date,
    datetime,
)
import re
import uuid

import pytest

import pandas as pd

from pandas.io.excel import ExcelWriter

odf = pytest.importorskip("odf")


@pytest.fixture
def ext():
    # [FUNCTION DOC]
    # Fixture returning the file extension used by these tests: ".ods".
    return ".ods"


@pytest.fixture
def tmp_excel(ext, tmp_path):
    # [FUNCTION DOC]
    # Fixture creating an empty file with a random unique name and the extension from the extension fixture inside the per-test temporary directory; returns its path as text.
    tmp = tmp_path / f"{uuid.uuid4()}{ext}"
    tmp.touch()
    return str(tmp)


def test_write_append_mode_raises(tmp_excel):
    # [FUNCTION DOC]
    # Verifies that opening the writer in append mode with the odf engine raises a value error saying append mode is not supported.
    msg = "Append mode is not supported with odf!"

    with pytest.raises(ValueError, match=msg):
        ExcelWriter(tmp_excel, engine="odf", mode="a")


@pytest.mark.parametrize("engine_kwargs", [None, {"kwarg": 1}])
def test_engine_kwargs(tmp_excel, engine_kwargs):
    # [FUNCTION DOC]
    # Verifies how engine keyword arguments are passed to the document constructor (issues 42286 and 43445).
    # Parametrized over none and a mapping with an unknown key. With the unknown key, creating the writer raises a type error saying the document class got an unexpected keyword argument. With none, the writer opens and closes without error.
    # GH 42286
    # GH 43445
    # test for error: OpenDocumentSpreadsheet does not accept any arguments
    if engine_kwargs is not None:
        error = re.escape(
            "OpenDocumentSpreadsheet() got an unexpected keyword argument 'kwarg'"
        )
        with pytest.raises(
            TypeError,
            match=error,
        ):
            ExcelWriter(tmp_excel, engine="odf", engine_kwargs=engine_kwargs)
    else:
        with ExcelWriter(tmp_excel, engine="odf", engine_kwargs=engine_kwargs) as _:
            pass


def test_book_and_sheets_consistent(tmp_excel):
    # [FUNCTION DOC]
    # Verifies the sheets mapping reflects tables added directly to the underlying document (issue 45687): initially empty, and after adding a table named test_name through the book's spreadsheet it maps that name to the table.
    # GH#45687 - Ensure sheets is updated if user modifies book
    with ExcelWriter(tmp_excel) as writer:
        assert writer.sheets == {}
        table = odf.table.Table(name="test_name")
        writer.book.spreadsheet.addElement(table)
        assert writer.sheets == {"test_name": table}


@pytest.mark.parametrize(
    ["value", "cell_value_type", "cell_value_attribute", "cell_value"],
    argvalues=[
        (True, "boolean", "boolean-value", "true"),
        ("test string", "string", "string-value", "test string"),
        (1, "float", "value", "1"),
        (1.5, "float", "value", "1.5"),
        (
            datetime(2010, 10, 10, 10, 10, 10),
            "date",
            "date-value",
            "2010-10-10T10:10:10",
        ),
        (date(2010, 10, 10), "date", "date-value", "2010-10-10"),
    ],
)
def test_cell_value_type(
    tmp_excel, value, cell_value_type, cell_value_attribute, cell_value
):
    # [FUNCTION DOC]
    # Verifies the office value-type attribute and the specific value attribute written for each cell type (issue 54994).
    # Parametrized over (python value, value type, value attribute name, attribute value): boolean true, string, integer, float, datetime (written with time) and date (written without time). Writes a one-cell frame without header or index, reopens the file with the excel file class, finds the first table cell in the first row and compares the two attributes to the expected strings.
    # GH#54994 ODS: cell attributes should follow specification
    # http://docs.oasis-open.org/office/v1.2/os/OpenDocument-v1.2-os-part1.html#refTable13
    from odf.namespaces import OFFICENS
    from odf.table import (
        TableCell,
        TableRow,
    )

    table_cell_name = TableCell().qname

    pd.DataFrame([[value]]).to_excel(tmp_excel, header=False, index=False)

    with pd.ExcelFile(tmp_excel) as wb:
        sheet = wb._reader.get_sheet_by_index(0)
        sheet_rows = sheet.getElementsByType(TableRow)
        sheet_cells = [
            x
            for x in sheet_rows[0].childNodes
            if hasattr(x, "qname") and x.qname == table_cell_name
        ]

        cell = sheet_cells[0]
        assert cell.attributes.get((OFFICENS, "value-type")) == cell_value_type
        assert cell.attributes.get((OFFICENS, cell_value_attribute)) == cell_value
