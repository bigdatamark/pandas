# [MODULE DOC]
# Tests for the openpyxl engine: style conversion, merged and styled cell writing, engine keyword arguments, append modes and the sheet-exists policies, reading from already loaded workbooks (with and without the read-only option), and several reader edge cases (missing dimension metadata, trailing empty rows, multi-level headers).
# Skipped at import time when openpyxl is missing. Defines local fixtures for the extension and a temporary file path. Reader data files live in the excel test data directory and are located through the data path fixture.

import contextlib
from pathlib import Path
import re
import uuid

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

from pandas.io.excel import (
    ExcelWriter,
    _OpenpyxlWriter,
)
from pandas.io.excel._openpyxl import OpenpyxlReader

openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def ext():
    # [FUNCTION DOC]
    # Fixture returning the extension used by these tests: ".xlsx".
    return ".xlsx"


@pytest.fixture
def tmp_excel(ext, tmp_path):
    # [FUNCTION DOC]
    # Fixture creating an empty file with a unique random name and the extension fixture inside the temporary directory fixture; returns its path as text.
    tmp = tmp_path / f"{uuid.uuid4()}{ext}"
    tmp.touch()
    return str(tmp)


def test_to_excel_styleconverter():
    # [FUNCTION DOC]
    # Verifies the writer's style conversion maps a nested style description to equal openpyxl style objects.
    # Input style has font (color and bold), borders on four sides, alignment, a solid fill with a tinted foreground color, a number format code and protection. Expected: the converted keyword mapping has font, border, alignment, fill, number format and protection equal to separately constructed openpyxl objects (thin side borders, center and top alignment, and so on).
    from openpyxl import styles

    hstyle = {
        "font": {"color": "00FF0000", "bold": True},
        "borders": {"top": "thin", "right": "thin", "bottom": "thin", "left": "thin"},
        "alignment": {"horizontal": "center", "vertical": "top"},
        "fill": {"patternType": "solid", "fgColor": {"rgb": "006666FF", "tint": 0.3}},
        "number_format": {"format_code": "0.00"},
        "protection": {"locked": True, "hidden": False},
    }

    font_color = styles.Color("00FF0000")
    font = styles.Font(bold=True, color=font_color)
    side = styles.Side(style=styles.borders.BORDER_THIN)
    border = styles.Border(top=side, right=side, bottom=side, left=side)
    alignment = styles.Alignment(horizontal="center", vertical="top")
    fill_color = styles.Color(rgb="006666FF", tint=0.3)
    fill = styles.PatternFill(patternType="solid", fgColor=fill_color)

    number_format = "0.00"

    protection = styles.Protection(locked=True, hidden=False)

    kw = _OpenpyxlWriter._convert_to_style_kwargs(hstyle)
    assert kw["font"] == font
    assert kw["border"] == border
    assert kw["alignment"] == alignment
    assert kw["fill"] == fill
    assert kw["number_format"] == number_format
    assert kw["protection"] == protection


def test_write_cells_merge_styled(tmp_excel):
    # [FUNCTION DOC]
    # Verifies writing a merged, styled cell range over earlier styled cells.
    # Writes two single cells in B1 and A2 with different font colors, then a merged region starting at A1 spanning to row and column one with a bold blue font. Expected: the cells B1 and A2 in the worksheet both carry the merged cell's converted font.
    from pandas.io.formats.excel import ExcelCell

    sheet_name = "merge_styled"

    sty_b1 = {"font": {"color": "00FF0000"}}
    sty_a2 = {"font": {"color": "0000FF00"}}

    initial_cells = [
        ExcelCell(col=1, row=0, val=42, style=sty_b1),
        ExcelCell(col=0, row=1, val=99, style=sty_a2),
    ]

    sty_merged = {"font": {"color": "000000FF", "bold": True}}
    sty_kwargs = _OpenpyxlWriter._convert_to_style_kwargs(sty_merged)
    openpyxl_sty_merged = sty_kwargs["font"]
    merge_cells = [
        ExcelCell(
            col=0, row=0, val="pandas", mergestart=1, mergeend=1, style=sty_merged
        )
    ]

    with _OpenpyxlWriter(tmp_excel) as writer:
        writer._write_cells(initial_cells, sheet_name=sheet_name)
        writer._write_cells(merge_cells, sheet_name=sheet_name)

        wks = writer.sheets[sheet_name]
    xcell_b1 = wks["B1"]
    xcell_a2 = wks["A2"]
    assert xcell_b1.font == openpyxl_sty_merged
    assert xcell_a2.font == openpyxl_sty_merged


@pytest.mark.parametrize("iso_dates", [True, False])
def test_engine_kwargs_write(tmp_excel, iso_dates):
    # [FUNCTION DOC]
    # Verifies that engine keyword arguments reach the workbook for writing.
    # Parametrized over the iso dates flag true and false (issues 42286 and 43445). The writer's workbook reports the flag as given; an empty frame is written because the writer requires something to be written to close cleanly.
    # GH 42286 GH 43445
    engine_kwargs = {"iso_dates": iso_dates}
    with ExcelWriter(
        tmp_excel, engine="openpyxl", engine_kwargs=engine_kwargs
    ) as writer:
        assert writer.book.iso_dates == iso_dates
        # ExcelWriter won't allow us to close without writing something
        DataFrame().to_excel(writer)


def test_engine_kwargs_append_invalid(tmp_excel):
    # [FUNCTION DOC]
    # Verifies that an invalid engine keyword argument in append mode is forwarded to the workbook loader and raises a type error naming the unexpected argument (issue 43445). A first file is written normally; the append attempt raises when the writer opens.
    # GH 43445
    # test whether an invalid engine kwargs actually raises
    DataFrame(["hello", "world"]).to_excel(tmp_excel)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "load_workbook() got an unexpected keyword argument 'apple_banana'"
        ),
    ):
        with ExcelWriter(
            tmp_excel,
            engine="openpyxl",
            mode="a",
            engine_kwargs={"apple_banana": "fruit"},
        ) as writer:
            # ExcelWriter needs us to write something to close properly
            DataFrame(["good"]).to_excel(writer, sheet_name="Sheet2")


@pytest.mark.parametrize("data_only, expected", [(True, 0), (False, "=1+1")])
def test_engine_kwargs_append_data_only(tmp_excel, data_only, expected):
    # [FUNCTION DOC]
    # Verifies the data-only engine keyword in append mode and in reading (issue 43445).
    # Parametrized over (flag, expected cell value): when true the cell with a formula text evaluates (no cached value) and is expected to read as zero; when false the formula text is preserved. The same expectation must hold when the file is read through the generic reader with the same engine keyword.
    # GH 43445
    # tests whether the data_only engine_kwarg actually works well for
    # openpyxl's load_workbook
    DataFrame(["=1+1"]).to_excel(tmp_excel)
    with ExcelWriter(
        tmp_excel, engine="openpyxl", mode="a", engine_kwargs={"data_only": data_only}
    ) as writer:
        assert writer.sheets["Sheet1"]["B2"].value == expected
        # ExcelWriter needs us to writer something to close properly?
        DataFrame().to_excel(writer, sheet_name="Sheet2")

    # ensure that data_only also works for reading
    #  and that formulas/values roundtrip
    assert (
        pd.read_excel(
            tmp_excel,
            sheet_name="Sheet1",
            engine="openpyxl",
            engine_kwargs={"data_only": data_only},
        ).iloc[0, 1]
        == expected
    )


@pytest.mark.parametrize("kwarg_name", ["read_only", "data_only"])
@pytest.mark.parametrize("kwarg_value", [True, False])
def test_engine_kwargs_append_reader(datapath, ext, kwarg_name, kwarg_value):
    # [FUNCTION DOC]
    # Verifies the read-only and data-only options are passed through the reader's engine keyword mapping to the workbook (issue 55027): the workbook attribute with that name has the passed value. Parametrized over the option name and value; uses a bundled data file, and the reader is closed afterwards.
    # GH 55027
    # test that `read_only` and `data_only` can be passed to
    #  `openpyxl.reader.excel.load_workbook` via `engine_kwargs`
    filename = datapath("io", "data", "excel", "test1" + ext)
    with contextlib.closing(
        OpenpyxlReader(filename, engine_kwargs={kwarg_name: kwarg_value})
    ) as reader:
        assert getattr(reader.book, kwarg_name) == kwarg_value


@pytest.mark.parametrize(
    "mode,expected", [("w", ["baz"]), ("a", ["foo", "bar", "baz"])]
)
def test_write_append_mode(tmp_excel, mode, expected):
    # [FUNCTION DOC]
    # Verifies write versus append mode.
    # Parametrized over (mode, expected sheet titles): write mode replaces the file so only the new sheet remains; append mode keeps the existing two sheets and adds the new sheet at the end. Also checks the first-cell value of each sheet is the sheet title (existing sheets keep their data).
    df = DataFrame([1], columns=["baz"])

    wb = openpyxl.Workbook()
    wb.worksheets[0].title = "foo"
    wb.worksheets[0]["A1"].value = "foo"
    wb.create_sheet("bar")
    wb.worksheets[1]["A1"].value = "bar"
    wb.save(tmp_excel)

    with ExcelWriter(tmp_excel, engine="openpyxl", mode=mode) as writer:
        df.to_excel(writer, sheet_name="baz", index=False)

    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb2:
        result = [sheet.title for sheet in wb2.worksheets]
        assert result == expected

        for index, cell_value in enumerate(expected):
            assert wb2.worksheets[index]["A1"].value == cell_value


@pytest.mark.parametrize(
    "if_sheet_exists,num_sheets,expected",
    [
        ("new", 2, ["apple", "banana"]),
        ("replace", 1, ["pear"]),
        ("overlay", 1, ["pear", "banana"]),
    ],
)
def test_if_sheet_exists_append_modes(tmp_excel, if_sheet_exists, num_sheets, expected):
    # [FUNCTION DOC]
    # Verifies the policy for appending to a sheet that already exists.
    # Parametrized over (policy, expected sheet count, expected values of the first sheet): new keeps the old sheet and creates a second one (the new sheet's content equals the new data), replace leaves one sheet with the new data, overlay writes over the old cells so remaining old values persist. Issue 40230.
    # GH 40230
    df1 = DataFrame({"fruit": ["apple", "banana"]})
    df2 = DataFrame({"fruit": ["pear"]})

    df1.to_excel(tmp_excel, engine="openpyxl", sheet_name="foo", index=False)
    with ExcelWriter(
        tmp_excel, engine="openpyxl", mode="a", if_sheet_exists=if_sheet_exists
    ) as writer:
        df2.to_excel(writer, sheet_name="foo", index=False)

    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as wb:
        assert len(wb.sheetnames) == num_sheets
        assert wb.sheetnames[0] == "foo"
        result = pd.read_excel(wb, "foo", engine="openpyxl")
        assert list(result["fruit"]) == expected
        if len(wb.sheetnames) == 2:
            result = pd.read_excel(wb, wb.sheetnames[1], engine="openpyxl")
            tm.assert_frame_equal(result, df2)


@pytest.mark.parametrize(
    "startrow, startcol, greeting, goodbye",
    [
        (0, 0, ["poop", "world"], ["goodbye", "people"]),
        (0, 1, ["hello", "world"], ["poop", "people"]),
        (1, 0, ["hello", "poop"], ["goodbye", "people"]),
        (1, 1, ["hello", "world"], ["goodbye", "poop"]),
    ],
)
def test_append_overlay_startrow_startcol(
    tmp_excel, startrow, startcol, greeting, goodbye
):
    # [FUNCTION DOC]
    # Verifies overlay-mode writing with a start row and start column.
    # Parametrized over (start row, start column, expected greeting column, expected goodbye column). A single value is written without header or index, shifted by one row for the header, over an existing two-by-two table, and the expected frames show which cell was replaced.
    df1 = DataFrame({"greeting": ["hello", "world"], "goodbye": ["goodbye", "people"]})
    df2 = DataFrame(["poop"])

    df1.to_excel(tmp_excel, engine="openpyxl", sheet_name="poo", index=False)
    with ExcelWriter(
        tmp_excel, engine="openpyxl", mode="a", if_sheet_exists="overlay"
    ) as writer:
        # use startrow+1 because we don't have a header
        df2.to_excel(
            writer,
            index=False,
            header=False,
            startrow=startrow + 1,
            startcol=startcol,
            sheet_name="poo",
        )

    result = pd.read_excel(tmp_excel, sheet_name="poo", engine="openpyxl")
    expected = DataFrame({"greeting": greeting, "goodbye": goodbye})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "if_sheet_exists,msg",
    [
        (
            "invalid",
            "'invalid' is not valid for if_sheet_exists. Valid options "
            "are 'error', 'new', 'replace' and 'overlay'.",
        ),
        (
            "error",
            "Sheet 'foo' already exists and if_sheet_exists is set to 'error'.",
        ),
        (
            None,
            "Sheet 'foo' already exists and if_sheet_exists is set to 'error'.",
        ),
    ],
)
def test_if_sheet_exists_raises(tmp_excel, if_sheet_exists, msg):
    # [FUNCTION DOC]
    # Verifies errors from the sheet-exists policy.
    # Parametrized over (policy, message): an invalid policy value raises a value error listing the valid options; the error policy and the missing policy (none) raise a value error saying the sheet already exists and the policy is error. Issue 40230.
    # GH 40230
    df = DataFrame({"fruit": ["pear"]})
    df.to_excel(tmp_excel, sheet_name="foo", engine="openpyxl")
    with pytest.raises(ValueError, match=re.escape(msg)):
        with ExcelWriter(
            tmp_excel, engine="openpyxl", mode="a", if_sheet_exists=if_sheet_exists
        ) as writer:
            df.to_excel(writer, sheet_name="foo")


def test_to_excel_with_openpyxl_engine(tmp_excel):
    # [FUNCTION DOC]
    # Verifies that a styled frame (map of font colors and highlight of the maximum) can be written with the openpyxl engine (issue 29854). There is no assertion beyond completing without error.
    # GH 29854
    df1 = DataFrame({"A": np.linspace(1, 10, 10)})
    df2 = DataFrame({"B": np.linspace(1, 20, 10)})
    df = pd.concat([df1, df2], axis=1)
    styled = df.style.map(
        lambda val: f"color: {'red' if val < 0 else 'black'}"
    ).highlight_max()

    styled.to_excel(tmp_excel, engine="openpyxl")


@pytest.mark.parametrize("read_only", [True, False])
def test_read_workbook(datapath, ext, read_only):
    # [FUNCTION DOC]
    # Verifies reading from an already loaded openpyxl workbook object, parametrized over read-only true and false (issue 39528): the result equals reading the same file by path.
    # GH 39528
    filename = datapath("io", "data", "excel", "test1" + ext)
    with contextlib.closing(
        openpyxl.load_workbook(filename, read_only=read_only)
    ) as wb:
        result = pd.read_excel(wb, engine="openpyxl")
    expected = pd.read_excel(filename)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "header, expected_data",
    [
        (
            0,
            {
                "Title": [np.nan, "A", 1, 2, 3],
                "Unnamed: 1": [np.nan, "B", 4, 5, 6],
                "Unnamed: 2": [np.nan, "C", 7, 8, 9],
            },
        ),
        (2, {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
    ],
)
@pytest.mark.parametrize(
    "filename", ["dimension_missing", "dimension_small", "dimension_large"]
)
# When read_only is None, use read_excel instead of a workbook
@pytest.mark.parametrize("read_only", [True, False, None])
def test_read_with_bad_dimension(
    datapath, ext, header, expected_data, filename, read_only
):
    # [FUNCTION DOC]
    # Verifies reading sheets whose dimension metadata is missing, too small or too large (issues 38956 and 39001).
    # Parametrized over header row (0 or 2) with matching expected data, three file names, and the loading mode (read-only workbook, normal workbook, or reading directly by path when none). The expected frame is the same for all files in each header case.
    # GH 38956, 39001 - no/incorrect dimension information
    path = datapath("io", "data", "excel", f"{filename}{ext}")
    if read_only is None:
        result = pd.read_excel(path, header=header)
    else:
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            result = pd.read_excel(wb, engine="openpyxl", header=header)
    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)


def test_append_mode_file(tmp_excel):
    # [FUNCTION DOC]
    # Verifies appending to an existing workbook does not concatenate zip archives (issue 39576): after writing an empty frame twice (the second as a new sheet in append mode) the raw bytes contain the document properties entry name exactly twice.
    # GH 39576
    df = DataFrame()

    df.to_excel(tmp_excel, engine="openpyxl")

    with ExcelWriter(
        tmp_excel, mode="a", engine="openpyxl", if_sheet_exists="new"
    ) as writer:
        df.to_excel(writer)

    # make sure that zip files are not concatenated by making sure that
    # "docProps/app.xml" only occurs twice in the file
    data = Path(tmp_excel).read_bytes()
    first = data.find(b"docProps/app.xml")
    second = data.find(b"docProps/app.xml", first + 1)
    third = data.find(b"docProps/app.xml", second + 1)
    assert second != -1 and third == -1


# When read_only is None, use read_excel instead of a workbook
@pytest.mark.parametrize("read_only", [True, False, None])
def test_read_with_empty_trailing_rows(datapath, ext, read_only):
    # [FUNCTION DOC]
    # Verifies a sheet with empty trailing rows is read without the trailing rows (issue 39181), for read-only workbook, normal workbook, and path; the result is a five-row frame with the title row and a letters row.
    # GH 39181
    path = datapath("io", "data", "excel", f"empty_trailing_rows{ext}")
    if read_only is None:
        result = pd.read_excel(path)
    else:
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            result = pd.read_excel(wb, engine="openpyxl")
    expected = DataFrame(
        {
            "Title": [np.nan, "A", 1, 2, 3],
            "Unnamed: 1": [np.nan, "B", 4, 5, 6],
            "Unnamed: 2": [np.nan, "C", 7, 8, 9],
        }
    )
    tm.assert_frame_equal(result, expected)


# When read_only is None, use read_excel instead of a workbook
@pytest.mark.parametrize("read_only", [True, False, None])
def test_read_empty_with_blank_row(datapath, ext, read_only):
    # [FUNCTION DOC]
    # Verifies a sheet that is empty except for a blank row is read as an empty frame (issue 39547), for the three loading modes.
    # GH 39547 - empty excel file with a row that has no data
    path = datapath("io", "data", "excel", f"empty_with_blank_row{ext}")
    if read_only is None:
        result = pd.read_excel(path)
    else:
        with contextlib.closing(
            openpyxl.load_workbook(path, read_only=read_only)
        ) as wb:
            result = pd.read_excel(wb, engine="openpyxl")
    expected = DataFrame()
    tm.assert_frame_equal(result, expected)


def test_book_and_sheets_consistent(tmp_excel):
    # [FUNCTION DOC]
    # Verifies the sheets mapping reflects a sheet created directly through the workbook (issue 45687): initially empty, then it maps the new sheet name to the sheet object.
    # GH#45687 - Ensure sheets is updated if user modifies book
    with ExcelWriter(tmp_excel, engine="openpyxl") as writer:
        assert writer.sheets == {}
        sheet = writer.book.create_sheet("test_name", 0)
        assert writer.sheets == {"test_name": sheet}


def test_ints_spelled_with_decimals(datapath, ext):
    # [FUNCTION DOC]
    # Verifies a sheet whose integer values are stored with a decimal part is read back as integers (issue 46988), as a one-column frame with the column label one and values two to eleven.
    # GH 46988 - openpyxl returns this sheet with floats
    path = datapath("io", "data", "excel", f"ints_spelled_with_decimals{ext}")
    result = pd.read_excel(path)
    expected = DataFrame(range(2, 12), columns=[1])
    tm.assert_frame_equal(result, expected)


def test_read_multiindex_header_no_index_names(datapath, ext):
    # [FUNCTION DOC]
    # Verifies reading a three-row header and three index columns with no index names (issue 47487), producing a frame with multi-level columns and a multi-level index, with missing values where cells are empty.
    # GH#47487
    path = datapath("io", "data", "excel", f"multiindex_no_index_names{ext}")
    result = pd.read_excel(path, index_col=[0, 1, 2], header=[0, 1, 2])
    expected = DataFrame(
        [[np.nan, "x", "x", "x"], ["x", np.nan, np.nan, np.nan]],
        columns=pd.MultiIndex.from_tuples(
            [("X", "Y", "A1"), ("X", "Y", "A2"), ("XX", "YY", "B1"), ("XX", "YY", "B2")]
        ),
        index=pd.MultiIndex.from_tuples([("A", "AA", "AAA"), ("A", "BB", "BBB")]),
    )
    tm.assert_frame_equal(result, expected)
