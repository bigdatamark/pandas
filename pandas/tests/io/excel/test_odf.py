# [MODULE DOC]
# Tests for reading OpenDocument spreadsheet and text files through the odf engine.
# At import time the whole module is skipped when the odf library is not installed. An automatic fixture makes the generic reader default to the odf engine and sets the working directory to the excel test data directory, so tests refer to data files by bare name.

import functools

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pytest.importorskip("odf")


@pytest.fixture(autouse=True)
def cd_and_set_engine(monkeypatch, datapath):
    # [FUNCTION DOC]
    # Autouse fixture (inputs: monkeypatch and data path fixtures). Replaces the top-level excel reading function with a partial application that fixes the engine to odf, and changes the working directory to the excel data directory; both are undone after each test.
    func = functools.partial(pd.read_excel, engine="odf")
    monkeypatch.setattr(pd, "read_excel", func)
    monkeypatch.chdir(datapath("io", "data", "excel"))


def test_read_invalid_types_raises():
    # [FUNCTION DOC]
    # Verifies that reading a file with an unknown cell value type raises a value error saying the type is unrecognized (the data file was manually edited to contain a made-up type).
    # the invalid_value_type.ods required manually editing
    # of the included content.xml file
    with pytest.raises(ValueError, match="Unrecognized type awesome_new_type"):
        pd.read_excel("invalid_value_type.ods")


def test_read_writer_table():
    # [FUNCTION DOC]
    # Verifies reading the table named Table1 from an OpenDocument text file with the first column as index.
    # Expected: a three-row frame indexed by row labels with index name Header; the empty middle column is named with the automatic unnamed label and holds missing values.
    # Also test reading tables from a text OpenDocument file
    # (.odt)
    index = pd.Index(["Row 1", "Row 2", "Row 3"], name="Header")
    expected = pd.DataFrame(
        [[1, np.nan, 7], [2, np.nan, 8], [3, np.nan, 9]],
        index=index,
        columns=["Column 1", "Unnamed: 2", "Column 3"],
    )

    result = pd.read_excel("writertable.odt", sheet_name="Table1", index_col=0)

    tm.assert_frame_equal(result, expected)


def test_read_newlines_between_xml_elements_table():
    # [FUNCTION DOC]
    # Verifies a file with newline text nodes between xml elements is read correctly (issue 45598): three rows of three columns where some cells are missing; the third column is integer-valued while the others are float.
    # GH#45598
    expected = pd.DataFrame(
        [[1.0, 4.0, 7], [np.nan, np.nan, 8], [3.0, 6.0, 9]],
        columns=["Column 1", "Column 2", "Column 3"],
    )

    result = pd.read_excel("test_newlines.ods")

    tm.assert_frame_equal(result, expected)


def test_read_unempty_cells():
    # [FUNCTION DOC]
    # Verifies empty cells that are not truly empty in the file (whitespace-only content) are read as missing values: a single column with values one, three and five and missing between.
    expected = pd.DataFrame(
        [1, np.nan, 3, np.nan, 5],
        columns=["Column 1"],
    )

    result = pd.read_excel("test_unempty_cells.ods")

    tm.assert_frame_equal(result, expected)


def test_read_cell_annotation():
    # [FUNCTION DOC]
    # Verifies that cells carrying annotations are read by their text value, not the annotation: a column with the strings test and test 3 and a missing value between.
    expected = pd.DataFrame(
        ["test", np.nan, "test 3"],
        columns=["Column 1"],
    )

    result = pd.read_excel("test_cell_annotation.ods")

    tm.assert_frame_equal(result, expected)
