# [MODULE DOC]
# Tests for the generic spreadsheet reader across all supported engines and file extensions (old and new Excel formats, binary Excel, OpenDocument).
# Builds at import time a list of engines (xlrd, openpyxl, automatic selection, pyxlsb, odf, calamine), each marked to skip when its library is missing, and a list of file extensions, then a parametrized fixture of all valid (engine, extension) pairs: invalid pairs are filtered out instead of skipped to avoid hundreds of skips. Provides fixtures for the engine, extension, temporary file and a reference frame read from a csv file, helpers for expected index units and expected frame adjustments, and an expectation-failure marker helper for the binary engine, which cannot read datetimes.
# Test classes cover the reader (reading options, dtypes, multi-level indexes, sheets, file sources, error cases), the file wrapper class (reuse, context handling, engine detection) and miscellaneous tests. Data files are located in the excel test data directory; an autouse fixture in the reader class changes into that directory so the tests refer to bare names, and sets the default engine of the top-level reader function.

from __future__ import annotations

from datetime import (
    datetime,
    time,
)
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
import uuid
from zipfile import BadZipFile

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    read_csv,
)
import pandas._testing as tm

read_ext_params = [".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"]
engine_params = [
    # Add any engines to test here
    # When defusedxml is installed it triggers deprecation warnings for
    # xlrd and openpyxl, so catch those here
    pytest.param(
        "xlrd",
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param(
        "openpyxl",
        marks=[
            td.skip_if_no("openpyxl"),
        ],
    ),
    pytest.param(
        None,
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param("pyxlsb", marks=td.skip_if_no("pyxlsb")),
    pytest.param("odf", marks=td.skip_if_no("odf")),
    pytest.param("calamine", marks=td.skip_if_no("python_calamine")),
]


def _is_valid_engine_ext_pair(engine, read_ext: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: decides whether an (engine, extension) combination should be generated for the engine and extension fixture.
    # Inputs: engine, a parametrized-value object whose first value is the engine name (the engine may be none for automatic selection), and read_ext, the extension text with the leading dot. Returns true when the pair is valid, false otherwise.
    # Rules: openpyxl cannot read the old format; the OpenDocument engine reads only the open document extension; that extension is read only by the odf and calamine engines; pyxlsb reads only the binary format; the binary extension is read only by pyxlsb and calamine; xlrd reads only the old format. Otherwise valid. Note that the engine name is taken from the first value of the parameter object.
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine = engine.values[0]
    if engine == "openpyxl" and read_ext == ".xls":
        return False
    if engine == "odf" and read_ext != ".ods":
        return False
    if read_ext == ".ods" and engine not in {"odf", "calamine"}:
        return False
    if engine == "pyxlsb" and read_ext != ".xlsb":
        return False
    if read_ext == ".xlsb" and engine not in {"pyxlsb", "calamine"}:
        return False
    if engine == "xlrd" and read_ext != ".xls":
        return False
    return True


def _transfer_marks(engine, read_ext):
    # [FUNCTION DOC]
    # HELPER: builds a new parameter for the engine-extension fixture.
    # Inputs: engine, a parameter object carrying values and skip marks; read_ext, the extension. Returns a new parameter whose value is a tuple of the engine's values followed by the extension, and which keeps the engine's marks so the skip conditions carry over.
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values = (*engine.values, read_ext)
    new_param = pytest.param(values, marks=engine.marks)
    return new_param


@pytest.fixture(
    params=[
        _transfer_marks(eng, ext)
        for eng in engine_params
        for ext in read_ext_params
        if _is_valid_engine_ext_pair(eng, ext)
    ],
    ids=str,
)
def engine_and_read_ext(request):
    # [FUNCTION DOC]
    # Fixture parametrized over all valid (engine, extension) pairs built from the module-level lists (ids are the string form). Returns the selected pair as a tuple.
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param


@pytest.fixture
def engine(engine_and_read_ext):
    # [FUNCTION DOC]
    # Fixture returning the engine element of the engine-extension pair fixture (may be none for automatic selection).
    engine, read_ext = engine_and_read_ext
    return engine


@pytest.fixture
def read_ext(engine_and_read_ext):
    # [FUNCTION DOC]
    # Fixture returning the extension element of the engine-extension pair fixture.
    engine, read_ext = engine_and_read_ext
    return read_ext


@pytest.fixture
def tmp_excel(read_ext, tmp_path):
    # [FUNCTION DOC]
    # Fixture creating an empty file named with a unique random name and the current extension in the temporary directory; returns the path as text.
    tmp = tmp_path / f"{uuid.uuid4()}{read_ext}"
    tmp.touch()
    return str(tmp)


@pytest.fixture
def df_ref(datapath):
    # [FUNCTION DOC]
    # Fixture returning the reference frame read from the test1 csv file (first column as index, dates parsed) using the pure Python parser engine; used as the expected content of the matching spreadsheet files.
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath = datapath("io", "data", "csv", "test1.csv")
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine="python")
    return df_ref


def get_exp_unit(read_ext: str, engine: str | None) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the expected resolution of parsed datetimes for a given extension and engine. Both arguments are currently ignored and the function always returns the microsecond unit.
    return "us"


def adjust_expected(expected: DataFrame, read_ext: str, engine: str | None) -> None:
    # [FUNCTION DOC]
    # HELPER: mutates an expected frame in place so it matches what the reader produces: clears the index name, and converts the index to the resolution given by the expected unit helper. Returns nothing. A type-check suppression is used because the index type is not known to have the unit conversion.
    expected.index.name = None
    unit = get_exp_unit(read_ext, engine)
    # error: "Index" has no attribute "as_unit"
    expected.index = expected.index.as_unit(unit)  # type: ignore[attr-defined]


def xfail_datetimes_with_pyxlsb(engine, request):
    # [FUNCTION DOC]
    # HELPER: marks the current test as an expected failure when the engine is the binary-format engine, which cannot read sheets with datetimes. Inputs: engine name; request, the pytest request object used to apply the marker. Does nothing for other engines.
    if engine == "pyxlsb":
        request.applymarker(
            pytest.mark.xfail(
                reason="Sheets containing datetimes not supported by pyxlsb"
            )
        )


class TestReaders:
    # [CLASS DOC]
    # Tests of the top-level reader function over all engines and extensions: column selection, index columns, dtypes and converters, multiple sheets, sources (buffers, paths, urls, cloud storage), skip rows, row limits, error handling and special values.
    # An autouse fixture sets the working directory and the default engine; most tests refer to data files by bare name.
    @pytest.mark.parametrize("col", [[True, None, False], [True], [True, False]])
    def test_read_excel_type_check(self, col, tmp_excel, read_ext):
        # [FUNCTION DOC]
        # Verifies a nullable boolean column (with and without missing values) round trips through writing and reading with an explicit dtype mapping (issue 58159). Parametrized over three column contents. Skipped for the old and binary formats because there is no engine to write them.
        # GH 58159
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({"bool_column": col}, dtype="boolean")
        df.to_excel(tmp_excel, index=False)
        df2 = pd.read_excel(tmp_excel, dtype={"bool_column": "boolean"})
        tm.assert_frame_equal(df, df2)

    def test_pass_none_type(self, datapath):
        # [FUNCTION DOC]
        # Verifies reading a bundled file of booleans with empty/none-like cells with the nullable boolean dtype and extra missing-value markers (issue 58159) using the openpyxl engine: expected a column of true, missing, false and so on with boolean dtype.
        # GH 58159
        f_path = datapath("io", "data", "excel", "test_none_type.xlsx")

        with pd.ExcelFile(f_path) as excel:
            parsed = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["nan", "None", "abcd"],
                dtype="boolean",
                engine="openpyxl",
            )
        expected = DataFrame(
            {"Test": [True, None, False, None, False, None, True]},
            dtype="boolean",
        )

        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        # [FUNCTION DOC]
        # Autouse fixture (inputs: engine, data path fixtures and monkeypatch). Changes into the excel data directory and replaces the top-level reader function with a partial application fixing the engine; both changes are undone after each test.
        """
        Change directory and set engine for read_excel calls.
        """
        func = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "read_excel", func)

    def test_engine_used(self, read_ext, engine, monkeypatch):
        # [FUNCTION DOC]
        # Verifies which engine the reader uses (issue 38884).
        # Patches the file wrapper's parse method to return the engine name instead of parsing, opens a bundled file in binary mode and reads it; when an engine was given the result is that engine, otherwise the default engine for the extension (openpyxl for the XML formats, the binary engine for xlsb, xlrd for xls and odf for ods).
        # GH 38884
        def parser(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: stand-in for the wrapper's parse method used by the engine test. Accepts and ignores all arguments and returns the engine attribute of the wrapper instead of parsing.
            return self.engine

        monkeypatch.setattr(pd.ExcelFile, "parse", parser)

        expected_defaults = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        with open("test1" + read_ext, "rb") as f:
            result = pd.read_excel(f)

        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_engine_kwargs(self, read_ext, engine):
        # [FUNCTION DOC]
        # Verifies that engine keyword arguments are passed to the underlying loader (issue 52214).
        # Passes an unknown keyword for every extension and expects a type error naming the loader's unexpected argument: the open-workbook function for xlrd and pyxlsb, the load function for odf, and the load-workbook function for the others. When no engine is given the test performs no check.
        # GH#52214
        expected_defaults = {
            "xlsx": {"foo": "abcd"},
            "xlsm": {"foo": 123},
            "xlsb": {"foo": "True"},
            "xls": {"foo": True},
            "ods": {"foo": "abcd"},
        }

        if engine in {"xlrd", "pyxlsb"}:
            msg = re.escape(r"open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == "odf":
            msg = re.escape(r"load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape(r"load_workbook() got an unexpected keyword argument 'foo'")

        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel(
                    "test1" + read_ext,
                    sheet_name="Sheet1",
                    index_col=0,
                    engine_kwargs=expected_defaults[read_ext[1:]],
                )

    def test_usecols_int(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that passing an integer as the column selection raises a value error (messages saying an integer for the selection is not allowed), for two sheets, the second with a skipped row.
        # usecols as int
        msg = "Passing an integer for `usecols`"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=3
            )

        # usecols as int
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet2",
                skiprows=[1],
                index_col=0,
                usecols=3,
            )

    def test_usecols_list(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies column selection by a list of positions on two sheets (the second skipping one row): the result equals the reference columns B and C, with the index adjusted. Expected failure for the binary engine.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)

        df1 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=[0, 2, 3]
        )
        df2 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols=[0, 2, 3],
        )

        # TODO add index to xls file)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_usecols_str(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies column selection by spreadsheet column letters on two sheets: a contiguous range, a comma-separated list, and a mixed list with range. Results are compared with the matching reference columns (A, B, C; then B, C) after adjusting the expected index.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["A", "B", "C"]]
        adjust_expected(expected, read_ext, engine)

        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A:D",
        )

        # TODO add index to xls, read xls ignores index name ?
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)

        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C,D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C,D",
        )
        # TODO add index to xls file
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C:D",
        )
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize(
        "usecols", [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]]
    )
    def test_usecols_diff_positional_int_columns_order(
        self, request, engine, read_ext, usecols, df_ref
    ):
        # [FUNCTION DOC]
        # Verifies that the order of integer column positions does not change the output column order. Parametrized over all six permutations of positions zero, one and three. Expected: the reference columns A and C in original order.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["A", "C"]]
        adjust_expected(expected, read_ext, engine)

        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=usecols
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("usecols", [["B", "D"], ["D", "B"]])
    def test_usecols_diff_positional_str_columns_order(self, read_ext, usecols, df_ref):
        # [FUNCTION DOC]
        # Verifies that the order of selected column names does not change the output order: parametrized over two orders of B and D; expected is the reference with a default range index.
        expected = df_ref[["B", "D"]]
        expected.index = range(len(expected))

        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_without_slicing(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies reading a whole sheet with the first column as index equals the reference frame after adjustments. Expected failure for the binary engine.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies a column selection mixing a single letter and a range (A and D to E) yields the reference columns C and D.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref[["C", "D"]]
        adjust_expected(expected, read_ext, engine)

        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,D:E"
        )
        tm.assert_frame_equal(result, expected)

    def test_usecols_excel_range_str_invalid(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a column range whose end is not a valid column name raises a value error with an invalid column name message.
        msg = "Invalid column name: E1"

        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols="D:E1")

    def test_index_col_label_error(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that selecting an index column by label together with a list of column names raises a type error about list indices needing to be integers.
        msg = "list indices must be integers.*, not str"

        with pytest.raises(TypeError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet1",
                index_col=["A"],
                usecols=["A", "C"],
            )

    def test_index_col_str(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading an empty sheet with an index column given by name (issue 52716) yields an empty frame with that index name and the remaining columns.
        # see gh-52716
        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet3", index_col="A")
        expected = DataFrame(
            columns=["B", "C", "D", "E", "F"], index=Index([], name="A")
        )
        tm.assert_frame_equal(result, expected)

    def test_index_col_empty(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading an empty sheet with a list of three index column positions (issue 9208) yields an empty frame with a three-level empty multi-level index with the given names and the other columns.
        # see gh-9208
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet3", index_col=["A", "B", "C"]
        )
        expected = DataFrame(
            columns=["D", "E", "F"],
            index=MultiIndex(levels=[[]] * 3, codes=[[]] * 3, names=["A", "B", "C"]),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("index_col", [None, 2])
    def test_index_col_with_unnamed(self, read_ext, index_col):
        # [FUNCTION DOC]
        # Verifies handling of an unnamed first column (issue 18792).
        # Parametrized over index column none and two. Expected: a two-row frame with columns named automatically for the unnamed one; when the index column is set, the frame is indexed by that column.
        # see gh-18792
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet4", index_col=index_col
        )
        expected = DataFrame(
            [["i1", "a", "x"], ["i2", "b", "y"]], columns=["Unnamed: 0", "col1", "col2"]
        )
        if index_col:
            expected = expected.set_index(expected.columns[index_col])

        tm.assert_frame_equal(result, expected)

    def test_usecols_pass_non_existent_column(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that selecting a column that does not exist raises a value error listing the columns expected but not found.
        msg = (
            "Usecols do not match columns, "
            "columns expected but not found: "
            r"\['E'\]"
        )

        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E"])

    def test_usecols_wrong_type(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that a column selection mixing strings and an integer raises a value error stating it must be list-like of all strings, all unicode, all integers or a callable.
        msg = (
            "'usecols' must either be list-like of "
            "all strings, all unicode, all integers or a callable."
        )

        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E1", 0])

    def test_excel_stop_iterator(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading a small sheet with a single row works and gives the expected two-column frame.
        parsed = pd.read_excel("test2" + read_ext, sheet_name="Sheet1")
        expected = DataFrame([["aaaa", "bbbbb"]], columns=["Test", "Test1"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_cell_error_na(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies a cell holding an error value is read as a missing value. Expected failure for the binary engine, and for calamine with the open document format (the error cannot be extracted, see the referenced calamine issue 355).
        xfail_datetimes_with_pyxlsb(engine, request)

        # https://github.com/tafia/calamine/issues/355
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(reason="Calamine can't extract error from ods files")
            )

        parsed = pd.read_excel("test3" + read_ext, sheet_name="Sheet1")
        expected = DataFrame([[np.nan]], columns=["Test"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies reading the table sheets match the reference: a sheet read with the first column as index, a second sheet skipping one row, and a read dropping the last row via the footer-skip option equals the first result without its last row.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        df1 = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet2", skiprows=[1], index_col=0
        )
        # TODO add index to file
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        df3 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, skipfooter=1
        )
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_reader_special_dtypes(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies type inference for integers, floats, bools, strings, mixed strings and numbers, and dates (issue related to special dtypes). Datetime unit follows the expected-unit helper; expected failure for the binary engine.
        # Steps: reads the sheet and compares with the expected frame; reads again and compares with a variant where the integer 3 in the mixed column is a float; reads with each column in turn as the index; and reads with a converter turning the numeric string column into text, expecting that column as text.
        # KNOWN ISSUE: the second read uses exactly the same arguments as the first, although its comment says the numbers are not coerced; only the expected frame differs (a float instead of an integer in one cell), so the second comparison does not test any additional reader option.
        xfail_datetimes_with_pyxlsb(engine, request)

        unit = get_exp_unit(read_ext, engine)
        expected = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, 4, 0],
                "FloatCol": [1.25, 2.25, 1.83, 1.92, 0.0000000005],
                "BoolCol": [True, False, True, True, False],
                "StrCol": [1, 2, 3, 4, 5],
                "Str2Col": ["a", 3, "c", "d", "e"],
                "DateCol": Index(
                    [
                        datetime(2013, 10, 30),
                        datetime(2013, 10, 31),
                        datetime(1905, 1, 1),
                        datetime(2013, 12, 14),
                        datetime(2015, 3, 14),
                    ],
                    dtype=f"M8[{unit}]",
                ),
            },
        )
        basename = "test_types"

        # should read in correctly and infer types
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

        # if not coercing number, then int comes in as float
        float_expected = expected.copy()
        float_expected.loc[float_expected.index[1], "Str2Col"] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, float_expected)

        # check setting Index (assuming xls and xlsx are the same here)
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(
                basename + read_ext, sheet_name="Sheet1", index_col=icol
            )
            exp = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)

        expected["StrCol"] = expected["StrCol"].apply(str)
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters={"StrCol": str}
        )
        tm.assert_frame_equal(actual, expected)

    # GH8212 - support for converters and missing values
    def test_reader_converters(self, read_ext):
        # [FUNCTION DOC]
        # Verifies per-column converters (issue 8212), keyed by name and by position, applied to single cell values: an integer converter with a default for empty cells, a float converter scaling by ten with missing for empty, a position-keyed converter producing found / not found text, and a position-keyed text converter. Compares with an expected frame including missing values.
        basename = "test_converters"

        expected = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, -1000, 0],
                "FloatCol": [12.5, np.nan, 18.3, 19.2, 0.000000005],
                "BoolCol": ["Found", "Found", "Found", "Not found", "Found"],
                "StrCol": ["1", np.nan, "3", "4", "5"],
            }
        )

        converters = {
            "IntCol": lambda x: int(x) if x != "" else -1000,
            "FloatCol": lambda x: 10 * x if x else np.nan,
            2: lambda x: "Found" if x != "" else "Not found",
            3: lambda x: str(x) if x else "",
        }

        # should read in correctly and set types of single cells (not array
        # dtypes)
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters=converters
        )
        tm.assert_frame_equal(actual, expected)

    def test_reader_dtype(self, read_ext):
        # [FUNCTION DOC]
        # Verifies dtype handling (issue 8212): default inference; a mapping of dtypes (64-bit float, 32-bit float, text) producing matching dtypes with zero-padded text values; and requesting an integer type for a column with a missing value raises a value error saying the column cannot be converted to that type.
        # GH 8212
        basename = "testdtype"
        actual = pd.read_excel(basename + read_ext)

        expected = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2.5, 3.5, 4.5, 5.5],
                "c": [1, 2, 3, 4],
                "d": [1.0, 2.0, np.nan, 4.0],
            }
        )

        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel(
            basename + read_ext, dtype={"a": "float64", "b": "float32", "c": str}
        )

        expected["a"] = expected["a"].astype("float64")
        expected["b"] = expected["b"].astype("float32")
        expected["c"] = Series(["001", "002", "003", "004"], dtype="str")
        tm.assert_frame_equal(actual, expected)

        msg = "Unable to convert column d to type int64"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={"d": "int64"})

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (
                None,
                {
                    "a": [1, 2, 3, 4],
                    "b": [2.5, 3.5, 4.5, 5.5],
                    "c": [1, 2, 3, 4],
                    "d": [1.0, 2.0, np.nan, 4.0],
                },
            ),
            (
                {"a": "float64", "b": "float32", "c": str, "d": str},
                {
                    "a": Series([1, 2, 3, 4], dtype="float64"),
                    "b": Series([2.5, 3.5, 4.5, 5.5], dtype="float32"),
                    "c": Series(["001", "002", "003", "004"], dtype="str"),
                    "d": Series(["1", "2", np.nan, "4"], dtype="str"),
                },
            ),
        ],
    )
    def test_reader_dtype_str(self, read_ext, dtype, expected):
        # [FUNCTION DOC]
        # Verifies text dtype conversion (issue 20377). Parametrized over no dtype and a mapping converting four columns to float64, float32, text and text; the text conversion keeps missing values as missing.
        # see gh-20377
        basename = "testdtype"

        actual = pd.read_excel(basename + read_ext, dtype=dtype)
        expected = DataFrame(expected)
        tm.assert_frame_equal(actual, expected)

    def test_dtype_backend(self, read_ext, dtype_backend, engine, tmp_excel):
        # [FUNCTION DOC]
        # Verifies the dtype backend option with numpy-nullable and pyarrow backends (issue 36712).
        # Skipped for old and binary formats. Writes a frame with nullable integer, float, boolean, string, columns with missing values, a timestamp column and an all-missing integer column, then reads it with the backend fixture. For the pyarrow backend the expected frame is built from arrow arrays, with the string columns cast to plain string (the default is large string), the timestamp cast to microsecond resolution and the all-missing column using the arrow null type. For the other backend the expected is the original with the timestamp column cast to the expected unit.
        # GH#36712
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        df = DataFrame(
            {
                "a": Series([1, 3], dtype="Int64"),
                "b": Series([2.5, 4.5], dtype="Float64"),
                "c": Series([True, False], dtype="boolean"),
                "d": Series(["a", "b"], dtype="string"),
                "e": Series([pd.NA, 6], dtype="Int64"),
                "f": Series([pd.NA, 7.5], dtype="Float64"),
                "g": Series([pd.NA, True], dtype="boolean"),
                "h": Series([pd.NA, "a"], dtype="string"),
                "i": Series([pd.Timestamp("2019-12-31")] * 2),
                "j": Series([pd.NA, pd.NA], dtype="Int64"),
            }
        )
        df.to_excel(tmp_excel, sheet_name="test", index=False)
        result = pd.read_excel(
            tmp_excel, sheet_name="test", dtype_backend=dtype_backend
        )
        if dtype_backend == "pyarrow":
            import pyarrow as pa

            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )

            # pandas uses large_string by default, but pyarrow infers string
            expected["d"] = expected["d"].astype(pd.ArrowDtype(pa.string()))
            expected["h"] = expected["h"].astype(pd.ArrowDtype(pa.string()))
            # pyarrow by default infers timestamp resolution as us, not ns
            expected["i"] = ArrowExtensionArray(
                expected["i"].array._pa_array.cast(pa.timestamp(unit="us"))
            )
            # pyarrow supports a null type, so don't have to default to Int64
            expected["j"] = ArrowExtensionArray(pa.array([None, None]))
        else:
            expected = df
            unit = get_exp_unit(read_ext, engine)
            expected["i"] = expected["i"].astype(f"M8[{unit}]")

        tm.assert_frame_equal(result, expected)

    def test_dtype_backend_and_dtype(self, read_ext, tmp_excel):
        # [FUNCTION DOC]
        # Verifies that the numpy-nullable backend together with an explicit float dtype (issue 36712) returns plain float64 columns equal to the original floats with missing values. Skipped for old and binary formats.
        # GH#36712
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        df = DataFrame({"a": [np.nan, 1.0], "b": [2.5, np.nan]})
        df.to_excel(tmp_excel, sheet_name="test", index=False)
        result = pd.read_excel(
            tmp_excel,
            sheet_name="test",
            dtype_backend="numpy_nullable",
            dtype="float64",
        )
        tm.assert_frame_equal(result, df)

    def test_dtype_backend_string(self, read_ext, string_storage, tmp_excel):
        # [FUNCTION DOC]
        # Verifies the string storage option affects the string dtype produced with the numpy-nullable backend (issue 36712). Parametrized by the string-storage fixture. The expected columns have the string dtype with that storage; column index type is not compared. Skipped for old and binary formats.
        # GH#36712
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        df = DataFrame(
            {
                "a": np.array(["a", "b"], dtype=np.object_),
                "b": np.array(["x", pd.NA], dtype=np.object_),
            }
        )
        df.to_excel(tmp_excel, sheet_name="test", index=False)

        with pd.option_context("mode.string_storage", string_storage):
            result = pd.read_excel(
                tmp_excel, sheet_name="test", dtype_backend="numpy_nullable"
            )

        expected = DataFrame(
            {
                "a": Series(["a", "b"], dtype=pd.StringDtype(string_storage)),
                "b": Series(["x", None], dtype=pd.StringDtype(string_storage)),
            }
        )
        # the storage of the str columns' Index is also affected by the
        # string_storage setting -> ignore that for checking the result
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize("dtypes, exp_value", [({}, 1), ({"a.1": "int64"}, 1)])
    def test_dtype_mangle_dup_cols(self, read_ext, dtypes, exp_value):
        # [FUNCTION DOC]
        # Verifies that dtype mappings apply to automatically renamed duplicate column names (issues 35211 and 42462).
        # Parametrized over (extra dtype mapping, expected value): no extra mapping, and a mapping keying the renamed duplicate to 64-bit integer. The base mapping sets the first duplicate to object. The passed mapping must not be modified by the call. Expected frame has two object-typed columns (the second defaults to inferred type when an explicit dtype was passed).
        # GH#35211
        basename = "df_mangle_dup_col_dtypes"
        dtype_dict = {"a": object, **dtypes}
        dtype_dict_copy = dtype_dict.copy()
        # GH#42462
        result = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        expected = DataFrame(
            {
                "a": Series([1], dtype=object),
                "a.1": Series([exp_value], dtype=object if not dtypes else None),
            }
        )
        assert dtype_dict == dtype_dict_copy, "dtype dict changed"
        tm.assert_frame_equal(result, expected)

    def test_reader_spaces(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that leading, trailing and repeated inner spaces in strings are preserved (issue 32207).
        # see gh-32207
        basename = "test_spaces"

        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame(
            {
                "testcol": [
                    "this is great",
                    "4    spaces",
                    "1 trailing ",
                    " 1 leading",
                    "2  spaces  multiple  times",
                ]
            }
        )
        tm.assert_frame_equal(actual, expected)

    # gh-36122, gh-35802
    @pytest.mark.parametrize(
        "basename,expected",
        [
            ("gh-35802", DataFrame({"COLUMN": ["Test (1)"]})),
            ("gh-36122", DataFrame(columns=["got 2nd sa"])),
        ],
    )
    def test_read_excel_ods_nested_xml(self, engine, read_ext, basename, expected):
        # [FUNCTION DOC]
        # Verifies reading open document files with nested xml in cells (issues 35802 and 36122). Parametrized over two file base names with expected frames (a single text value; an empty frame with one column). Skipped unless the engine is odf.
        # see gh-35802
        if engine != "odf":
            pytest.skip(f"Skipped for engine: {engine}")

        actual = pd.read_excel(basename + read_ext)
        tm.assert_frame_equal(actual, expected)

    def test_reading_all_sheets(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading every sheet (sheet name none) returns an ordered mapping from name to frame with the sheet order preserved (PR 9450 and issue 9930): keys Charlie, Alpha, Beta in that, non-alphabetical, order.
        # Test reading all sheet names by setting sheet_name to None,
        # Ensure a dict is returned.
        # See PR #9450
        basename = "test_multisheet"
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        # ensure this is not alphabetical to test order preservation
        expected_keys = ["Charlie", "Alpha", "Beta"]
        tm.assert_contains_all(expected_keys, dfs.keys())
        # Issue 9930
        # Ensure sheet order is preserved
        assert expected_keys == list(dfs.keys())

    def test_reading_multiple_specific_sheets(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading a mixed list of a position and names with duplicates (PR 9450) returns a mapping with the duplicates removed: the keys are the distinct requested entries.
        # Test reading specific sheet names by specifying a mixed list
        # of integers and strings, and confirm that duplicated sheet
        # references (positions/names) are removed properly.
        # Ensure a dict is returned
        # See PR #9450
        basename = "test_multisheet"
        # Explicitly request duplicates. Only the set should be returned.
        expected_keys = [2, "Charlie", "Charlie"]
        dfs = pd.read_excel(basename + read_ext, sheet_name=expected_keys)
        expected_keys = list(set(expected_keys))
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert len(expected_keys) == len(dfs.keys())

    def test_reading_all_sheets_with_blank(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading all sheets from a workbook where some sheets are blank (issue 11711) still returns all three sheet names.
        # Test reading all sheet names by setting sheet_name to None,
        # In the case where some sheets are blank.
        # Issue #11711
        basename = "blank_with_header"
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ["Sheet1", "Sheet2", "Sheet3"]
        tm.assert_contains_all(expected_keys, dfs.keys())

    # GH6403
    def test_read_excel_blank(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a blank sheet is read as an empty frame (issue 6403).
        actual = pd.read_excel("blank" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, DataFrame())

    def test_read_excel_blank_with_header(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a sheet containing only a header row is read as an empty frame with the two header column names.
        expected = DataFrame(columns=["col_1", "col_2"])
        actual = pd.read_excel("blank_with_header" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    def test_exception_message_includes_sheet_name(self, read_ext):
        # [FUNCTION DOC]
        # Verifies errors raised while parsing a sheet have the sheet name appended to the message (issue 48706): both a value error from an invalid header list and a zero-division error raised inside a user column-selection callable end with the sheet suffix.
        # GH 48706
        with pytest.raises(ValueError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("blank_with_header" + read_ext, header=[1], sheet_name=None)
        with pytest.raises(ZeroDivisionError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("test1" + read_ext, usecols=lambda x: 1 / 0, sheet_name=None)

    @pytest.mark.filterwarnings("ignore:Cell A4 is marked:UserWarning:openpyxl")
    def test_date_conversion_overflow(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies a date column containing a number too large to be a date keeps the numeric value as float while other cells are timestamps (issue 10001). Ignores an openpyxl warning about a cell being marked. Expected failure for the binary engine, for openpyxl and for automatic selection of it on the XML extensions (issue 35029).
        # GH 10001 : pandas.ExcelFile ignore parse_dates=False
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = DataFrame(
            [
                [pd.Timestamp("2016-03-12"), "Marc Johnson"],
                [pd.Timestamp("2016-03-16"), "Jack Black"],
                [1e20, "Timothy Brown"],
            ],
            columns=["DateColWithBigInt", "StringCol"],
        )

        if engine == "openpyxl":
            request.applymarker(
                pytest.mark.xfail(reason="Maybe not supported by openpyxl")
            )

        if engine is None and read_ext in (".xlsx", ".xlsm"):
            # GH 35029
            request.applymarker(
                pytest.mark.xfail(reason="Defaults to openpyxl, maybe not supported")
            )

        result = pd.read_excel("testdateoverflow" + read_ext)
        tm.assert_frame_equal(result, expected)

    def test_sheet_name(self, request, read_ext, engine, df_ref):
        # [FUNCTION DOC]
        # Verifies the sheet name argument works as keyword in either position: both results equal the reference. Expected failure for the binary engine.
        xfail_datetimes_with_pyxlsb(engine, request)

        filename = "test1"
        sheet_name = "Sheet1"

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        df1 = pd.read_excel(
            filename + read_ext, sheet_name=sheet_name, index_col=0
        )  # doc
        df2 = pd.read_excel(filename + read_ext, index_col=0, sheet_name=sheet_name)

        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_excel_read_buffer(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading from an open binary file gives the same frame as reading by path.
        pth = "test1" + read_ext
        expected = pd.read_excel(pth, sheet_name="Sheet1", index_col=0)
        with open(pth, "rb") as f:
            actual = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            tm.assert_frame_equal(expected, actual)

    def test_bad_engine_raises(self):
        # [FUNCTION DOC]
        # Verifies an unknown engine name raises a value error saying the engine is unknown.
        bad_engine = "foo"
        with pytest.raises(ValueError, match="Unknown engine: foo"):
            pd.read_excel("", engine=bad_engine)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"]],
    )
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        # [FUNCTION DOC]
        # Verifies selecting a missing sheet by position or name, or in a list, raises a value error (issue 39250) with either an invalid worksheet index or a worksheet not found message. Parametrized over six sheet selections.
        # GH 39250
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("blank" + read_ext, sheet_name=sheet_name)

    def test_missing_file_raises(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading a missing file raises a file-not-found error whose message matches the operating system message in English, Chinese or Italian.
        bad_file = f"foo{read_ext}"
        # CI tests with other languages, translates to "No such file or directory"
        match = "|".join(
            [
                "(No such file or directory",
                "没有那个文件或目录",
                "File o directory non esistente)",
            ]
        )
        with pytest.raises(FileNotFoundError, match=match):
            pd.read_excel(bad_file)

    def test_corrupt_bytes_raises(self, engine):
        # [FUNCTION DOC]
        # Verifies reading invalid bytes raises an engine-specific error: for automatic selection a value error saying the format cannot be determined; for xlrd its own error about an unsupported format or corrupt file; for calamine its own error that the format cannot be detected; for the others a bad zip file error saying the file is not a zip file.
        bad_stream = b"foo"
        if engine is None:
            error = ValueError
            msg = (
                "Excel file format cannot be determined, you must "
                "specify an engine manually."
            )
        elif engine == "xlrd":
            from xlrd import XLRDError

            error = XLRDError
            msg = (
                "Unsupported format, or corrupt file: Expected BOF record; found b'foo'"
            )
        elif engine == "calamine":
            from python_calamine import CalamineError

            error = CalamineError
            msg = "Cannot detect file format"
        else:
            error = BadZipFile
            msg = "File is not a zip file"
        with pytest.raises(error, match=msg):
            pd.read_excel(BytesIO(bad_stream))

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_read_from_http_url(self, httpserver, read_ext):
        # [FUNCTION DOC]
        # Verifies reading a spreadsheet served over http gives the same frame as reading the local file. Marked as needing network and single cpu; uses an http server fixture that serves the file content.
        with open("test1" + read_ext, "rb") as f:
            httpserver.serve_content(content=f.read())
        url_table = pd.read_excel(httpserver.url)
        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @td.skip_if_not_us_locale
    @pytest.mark.single_cpu
    def test_read_from_s3_url(self, read_ext, s3_bucket_public, s3so):
        # [FUNCTION DOC]
        # Verifies reading from an s3 url with storage options against the mock server gives the same frame as the local file. Requires a US locale and single cpu.
        with open("test1" + read_ext, "rb") as f:
            s3_bucket_public.put_object(Key="test1" + read_ext, Body=f)

        url = f"s3://{s3_bucket_public.name}/test1" + read_ext

        url_table = pd.read_excel(url, storage_options=s3so)
        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.single_cpu
    def test_read_from_s3_object(self, read_ext, s3_bucket_public, s3so):
        # [FUNCTION DOC]
        # Verifies reading from an open file object from the s3 filesystem library (issue 38788) gives the same frame as the local file. Single cpu.
        # GH 38788
        with open("test1" + read_ext, "rb") as f:
            s3_bucket_public.put_object(Key="test1" + read_ext, Body=f)

        import s3fs

        s3 = s3fs.S3FileSystem(**s3so)

        with s3.open(f"s3://{s3_bucket_public.name}/test1" + read_ext) as f:
            url_table = pd.read_excel(f)

        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.slow
    def test_read_from_file_url(self, read_ext, datapath):
        # [FUNCTION DOC]
        # Verifies reading via a file url equals reading the path directly. Marked slow. A URL error causes a skip with the platform description (fails on some systems).
        # FILE
        localtable = os.path.join(datapath("io", "data", "excel"), "test1" + read_ext)
        local_table = pd.read_excel(localtable)

        try:
            url_table = pd.read_excel("file://localhost/" + localtable)
        except URLError:
            # fails on some systems
            platform_info = " ".join(platform.uname()).strip()
            pytest.skip(f"failing on {platform_info}")

        tm.assert_frame_equal(url_table, local_table)

    def test_read_from_pathlib_path(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a path object works the same as a string path (issue 12655).
        # GH12655
        str_path = "test1" + read_ext
        expected = pd.read_excel(str_path, sheet_name="Sheet1", index_col=0)

        path_obj = Path("test1" + read_ext)
        actual = pd.read_excel(path_obj, sheet_name="Sheet1", index_col=0)

        tm.assert_frame_equal(expected, actual)

    def test_close_from_py_localpath(self, read_ext):
        # [FUNCTION DOC]
        # Verifies the reader does not close a caller-provided file object (issue 31467): after reading, the file can still be read.
        # GH31467
        str_path = os.path.join("test1" + read_ext)
        with open(str_path, "rb") as f:
            x = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            del x
            # should not throw an exception because the passed file was closed
            f.read()

    def test_reader_seconds(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies reading times with and without fractional seconds (issues 5945 and 55045), from files using the 1900 and 1904 date systems, into time objects. Expected failure for the binary engine and for calamine with the open document format (invalid seconds stored as text).
        xfail_datetimes_with_pyxlsb(engine, request)

        # GH 55045
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(
                    reason="ODS file contains bad datetime (seconds as text)"
                )
            )

        # Test reading times with and without milliseconds. GH5945.
        expected = DataFrame.from_dict(
            {
                "Time": [
                    time(1, 2, 3),
                    time(2, 45, 56, 100000),
                    time(4, 29, 49, 200000),
                    time(6, 13, 42, 300000),
                    time(7, 57, 35, 400000),
                    time(9, 41, 28, 500000),
                    time(11, 25, 21, 600000),
                    time(13, 9, 14, 700000),
                    time(14, 53, 7, 800000),
                    time(16, 37, 0, 900000),
                    time(18, 20, 54),
                ]
            }
        )

        actual = pd.read_excel("times_1900" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel("times_1904" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_multiindex(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies reading multi-level headers and indexes from many sheets of one workbook (issues 4679 and 11317).
        # Sheets covered: multi-level columns; multi-level index; both; index names; column names; an integer level in the column names; both with names; names with skipped rows. Each result is compared to an expected frame built by progressively changing the index and columns of one expected frame. The date column unit follows the expected-unit helper. Expected failure for the binary engine.
        # see gh-4679
        xfail_datetimes_with_pyxlsb(engine, request)

        unit = get_exp_unit(read_ext, engine)

        mi = MultiIndex.from_product([["foo", "bar"], ["a", "b"]])
        mi_file = "testmultiindex" + read_ext

        # "mi_column" sheet
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")

        actual = pd.read_excel(
            mi_file, sheet_name="mi_column", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)

        # "mi_index" sheet
        expected.index = mi
        expected.columns = ["a", "b", "c", "d"]

        actual = pd.read_excel(mi_file, sheet_name="mi_index", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # "both" sheet
        expected.columns = mi

        actual = pd.read_excel(
            mi_file, sheet_name="both", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # "mi_index_name" sheet
        expected.columns = ["a", "b", "c", "d"]
        expected.index = mi.set_names(["ilvl1", "ilvl2"])

        actual = pd.read_excel(mi_file, sheet_name="mi_index_name", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # "mi_column_name" sheet
        expected.index = range(4)
        expected.columns = mi.set_names(["c1", "c2"])
        actual = pd.read_excel(
            mi_file, sheet_name="mi_column_name", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)

        # see gh-11317
        # "name_with_int" sheet
        expected.columns = mi.set_levels([1, 2], level=1).set_names(["c1", "c2"])

        actual = pd.read_excel(
            mi_file, sheet_name="name_with_int", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # "both_name" sheet
        expected.columns = mi.set_names(["c1", "c2"])
        expected.index = mi.set_names(["ilvl1", "ilvl2"])

        actual = pd.read_excel(
            mi_file, sheet_name="both_name", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # "both_skiprows" sheet
        actual = pd.read_excel(
            mi_file,
            sheet_name="both_name_skiprows",
            index_col=[0, 1],
            header=[0, 1],
            skiprows=2,
        )
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "sheet_name,idx_lvl2",
        [
            ("both_name_blank_after_mi_name", [np.nan, "b", "a", "b"]),
            ("both_name_multiple_blanks", [np.nan] * 4),
        ],
    )
    def test_read_excel_multiindex_blank_after_name(
        self, request, engine, read_ext, sheet_name, idx_lvl2
    ):
        # [FUNCTION DOC]
        # Verifies multi-level index reading when blank cells follow the index names (issue 34673).
        # Parametrized over (sheet, second index level values): one sheet gives missing then letters, the other gives all missing. Expected frame uses named column and index levels.
        # GH34673
        xfail_datetimes_with_pyxlsb(engine, request)

        mi_file = "testmultiindex" + read_ext
        mi = MultiIndex.from_product([["foo", "bar"], ["a", "b"]], names=["c1", "c2"])

        unit = get_exp_unit(read_ext, engine)
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
            index=MultiIndex.from_arrays(
                (["foo", "foo", "bar", "bar"], idx_lvl2),
                names=["ilvl1", "ilvl2"],
            ),
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")
        result = pd.read_excel(
            mi_file,
            sheet_name=sheet_name,
            index_col=[0, 1],
            header=[0, 1],
        )
        tm.assert_frame_equal(result, expected)

    def test_read_excel_multiindex_header_only(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a multi-level header with no index column names does not try to parse a header name (issue 11733); expected two identical rows with two-level columns.
        # see gh-11733.
        #
        # Don't try to parse a header name if there isn't one.
        mi_file = "testmultiindex" + read_ext
        result = pd.read_excel(mi_file, sheet_name="index_col_none", header=[0, 1])

        exp_columns = MultiIndex.from_product([("A", "B"), ("key", "val")])
        expected = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
        tm.assert_frame_equal(result, expected)

    def test_excel_old_index_format(self, read_ext):
        # [FUNCTION DOC]
        # Verifies reading files in the older format for index names (issue 4679), where the header is detected to decide whether index names exist.
        # Four sheets: single and multi-level indexes with names (the name row is read as a data row with missing values) and without names. Each expected frame has columns of the same pattern and an index of the given levels (no names).
        # see gh-4679
        filename = "test_index_name_pre17" + read_ext

        # We detect headers to determine if index names exist, so
        # that "index" name in the "names" version of the data will
        # now be interpreted as rows that include null data.
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ],
            dtype=object,
        )
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        mi = MultiIndex(
            levels=[
                ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R1", "R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            names=[None, None],
        )
        si = Index(
            ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None
        )

        expected = DataFrame(data, index=si, columns=columns)

        actual = pd.read_excel(filename, sheet_name="single_names", index_col=0)
        tm.assert_frame_equal(actual, expected)

        expected.index = mi

        actual = pd.read_excel(filename, sheet_name="multi_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # The analogous versions of the "names" version data
        # where there are explicitly no names for the indices.
        data = np.array(
            [
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ]
        )
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        mi = MultiIndex(
            levels=[
                ["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            names=[None, None],
        )
        si = Index(["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None)

        expected = DataFrame(data, index=si, columns=columns)

        actual = pd.read_excel(filename, sheet_name="single_no_names", index_col=0)
        tm.assert_frame_equal(actual, expected)

        expected.index = mi

        actual = pd.read_excel(filename, sheet_name="multi_no_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_bool_header_arg(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that a boolean header argument (true or false) raises a type error saying a bool to header is invalid (issue 6114).
        # GH 6114
        msg = "Passing a bool to header is invalid"
        for arg in [True, False]:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel("test1" + read_ext, header=arg)

    def test_read_excel_skiprows(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies row skipping (issues 4903 and 36435) with a list, an array, a callable, and a count together with explicit names.
        # The first three produce the same frame (header and one middle row skipped); an integer count of three with column names skips the header and the first data rows, leaving three rows. The date column unit follows the helper. Expected failure for the binary engine.
        # GH 4903
        xfail_datetimes_with_pyxlsb(engine, request)

        unit = get_exp_unit(read_ext, engine)

        actual = pd.read_excel(
            "testskiprows" + read_ext, sheet_name="skiprows_list", skiprows=[0, 2]
        )
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=np.array([0, 2]),
        )
        tm.assert_frame_equal(actual, expected)

        # GH36435
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x in [0, 2],
        )
        tm.assert_frame_equal(actual, expected)

        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=3,
            names=["a", "b", "c", "d"],
        )
        expected = DataFrame(
            [
                # [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_skiprows_callable_not_in(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies row skipping with a callable that skips every row not in a given set (issue 4903): the two data rows kept form the expected frame.
        # GH 4903
        xfail_datetimes_with_pyxlsb(engine, request)
        unit = get_exp_unit(read_ext, engine)

        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x not in [1, 3, 5],
        )
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                # [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                # [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows(self, read_ext):
        # [FUNCTION DOC]
        # Verifies the row limit option (issue 16645): reading five rows equals the first five rows of a full read.
        # GH 16645
        num_rows_to_pull = 5
        actual = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        expected = pd.read_excel("test1" + read_ext)
        expected = expected[:num_rows_to_pull]
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_greater_than_nrows_in_file(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a row limit larger than the number of rows in the file returns the full data (issue 16645).
        # GH 16645
        expected = pd.read_excel("test1" + read_ext)
        num_records_in_file = len(expected)
        num_rows_to_pull = num_records_in_file + 10
        actual = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        tm.assert_frame_equal(actual, expected)

    def test_read_excel_nrows_non_integer_parameter(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a text row limit raises a value error saying the limit must be an integer greater or equal to zero (issue 16645).
        # GH 16645
        msg = "'nrows' must be an integer >=0"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, nrows="5")

    @pytest.mark.parametrize(
        "filename,sheet_name,header,index_col,skiprows",
        [
            ("testmultiindex", "mi_column", [0, 1], 0, None),
            ("testmultiindex", "mi_index", None, [0, 1], None),
            ("testmultiindex", "both", [0, 1], [0, 1], None),
            ("testmultiindex", "mi_column_name", [0, 1], 0, None),
            ("testskiprows", "skiprows_list", None, None, [0, 2]),
            ("testskiprows", "skiprows_list", None, None, lambda x: x in (0, 2)),
        ],
    )
    def test_read_excel_nrows_params(
        self, read_ext, filename, sheet_name, header, index_col, skiprows
    ):
        # [FUNCTION DOC]
        # Verifies that limiting rows to three at load time equals loading everything and taking the first three rows (issue 46894). Parametrized over (file, sheet, header, index column, skip rows) including multi-level headers and indexes and list and callable row skipping.
        """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
        # GH 46894
        expected = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
        ).iloc[:3]
        actual = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
            nrows=3,
        )
        tm.assert_frame_equal(actual, expected)

    def test_deprecated_kwargs(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that passing the sheet name and index column positionally raises a type error saying only the first positional argument is allowed (message mentions three positional arguments given).
        with pytest.raises(TypeError, match="but 3 positional arguments"):
            pd.read_excel("test1" + read_ext, "Sheet1", 0)

    def test_no_header_with_list_index_col(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a sheet read with no header and a list of two index columns (issue 31783) yields a frame with a two-level index whose names are the integer positions, and integer column labels two and three.
        # GH 31783
        file_name = "testmultiindex" + read_ext
        data = [("B", "B"), ("key", "val"), (3, 4), (3, 4)]
        idx = MultiIndex.from_tuples(
            [("A", "A"), ("key", "val"), (1, 2), (1, 2)], names=(0, 1)
        )
        expected = DataFrame(data, index=idx, columns=(2, 3))
        result = pd.read_excel(
            file_name, sheet_name="index_col_none", index_col=[0, 1], header=None
        )
        tm.assert_frame_equal(expected, result)

    def test_one_col_noskip_blank_line(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a one-column sheet with a blank cell in the middle keeps the blank as a missing value instead of skipping the row (issue 39808).
        # GH 39808
        file_name = "one_col_blank_line" + read_ext
        data = [0.5, np.nan, 1, 2]
        expected = DataFrame(data, columns=["numbers"])
        result = pd.read_excel(file_name)
        tm.assert_frame_equal(result, expected)

    def test_multiheader_two_blank_lines(self, read_ext):
        # [FUNCTION DOC]
        # Verifies a two-row header followed by blank rows keeps the blank rows as rows of missing values (issue 40442).
        # GH 40442
        file_name = "testmultiindex" + read_ext
        columns = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
        data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
        expected = DataFrame(data, columns=columns)
        result = pd.read_excel(
            file_name, sheet_name="mi_column_empty_rows", header=[0, 1]
        )
        tm.assert_frame_equal(result, expected)

    def test_trailing_blanks(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that trailing blank cells with no data are not included: the result has three rows and three columns.
        """
        Sheets can contain blank cells with no data. Some of our readers
        were including those cells, creating many empty rows and columns
        """
        file_name = "trailing_blanks" + read_ext
        result = pd.read_excel(file_name)
        assert result.shape == (3, 3)

    def test_ignore_chartsheets_by_str(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies that chart sheets are not treated as worksheets when selecting by name (issue 41448): asking for the chart sheet's name raises a value error that the worksheet was not found. Skipped for the open document format; expected failure for pyxlsb, which cannot tell chart sheets from worksheets.
        # GH 41448
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pytest.raises(ValueError, match="Worksheet named 'Chart1' not found"):
            pd.read_excel("chartsheet" + read_ext, sheet_name="Chart1")

    def test_ignore_chartsheets_by_int(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Same as the by-name test but selecting by position one raises a value error saying the index is invalid with one worksheet found (issue 41448).
        # GH 41448
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pytest.raises(
            ValueError, match="Worksheet index 1 is invalid, 1 worksheets found"
        ):
            pd.read_excel("chartsheet" + read_ext, sheet_name=1)

    def test_euro_decimal_format(self, read_ext):
        # [FUNCTION DOC]
        # Verifies the decimal separator option for text cells containing comma decimals, with one skipped row: numbers parse as floats and other columns are text.
        # copied from read_csv
        result = pd.read_excel("test_decimal" + read_ext, decimal=",", skiprows=1)
        expected = DataFrame(
            [
                [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
                [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
                [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
            ],
            columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
        )
        tm.assert_frame_equal(result, expected)


class TestExcelFileRead:
    # [CLASS DOC]
    # Tests of the file wrapper class and of the reader function applied to it: engine detection, missing-value options, sheet selection by position or name, buffers, closing behavior and error handling.
    # An autouse fixture sets the working directory and substitutes the wrapper class constructor with one fixing the engine.
    def test_raises_bytes_input(self, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies that raw bytes passed to the reader raise a type error saying a file path name or file-like object was expected (issue 53830).
        # GH 53830
        msg = "Expected file path name or file-like object"
        with pytest.raises(TypeError, match=msg):
            with open("test1" + read_ext, "rb") as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        # [FUNCTION DOC]
        # Autouse fixture. Changes into the excel data directory and replaces the file wrapper class in the top-level namespace with a partial application that fixes the engine; both reverted after each test.
        """
        Change directory and set engine for ExcelFile objects.
        """
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "ExcelFile", func)

    def test_engine_used(self, read_ext, engine):
        # [FUNCTION DOC]
        # Verifies the wrapper's engine attribute: the engine given, or the default for the extension (openpyxl for xlsx and xlsm, pyxlsb for xlsb, xlrd for xls, odf for ods).
        expected_defaults = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        with pd.ExcelFile("test1" + read_ext) as excel:
            result = excel.engine

        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_excel_passes_na(self, read_ext):
        # [FUNCTION DOC]
        # Verifies the missing-value options when parsing through the wrapper.
        # Two files: one with text such as the default missing markers and an extra marker, one with special text (issue 13967). With the default markers off, only the extra marker is missing; with them on, both defaults and the extra marker are missing; text values with the number or special markers are preserved as text when defaults are off.
        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        expected = DataFrame(
            [["NA"], [1], ["NA"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        # 13967
        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        expected = DataFrame(
            [["1.#QNAN"], [1], ["nan"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize("na_filter", [None, True, False])
    def test_excel_passes_na_filter(self, read_ext, na_filter):
        # [FUNCTION DOC]
        # Verifies the missing-value filter option (issue 25453). Parametrized over none (option not passed), true and false. With the filter off all markers stay as text (including the extra marker); otherwise defaults and the extra marker become missing.
        # gh-25453
        kwargs = {}

        if na_filter is not None:
            kwargs["na_filter"] = na_filter

        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["apple"],
                **kwargs,
            )

        if na_filter is False:
            expected = [["1.#QNAN"], [1], ["nan"], ["apple"], ["rabbit"]]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]]

        expected = DataFrame(expected, columns=["Test"])
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table_sheet_by_index(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies selecting sheets by position through the reader function and through the wrapper's parse method, with the footer-skip option, equals the reference frame (and its first rows). Expected failure for the binary engine.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)

        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_sheet_name(self, request, engine, read_ext, df_ref):
        # [FUNCTION DOC]
        # Verifies the sheet name keyword in either argument order of the wrapper's parse method equals the reference frame. Expected failure for the binary engine.
        xfail_datetimes_with_pyxlsb(engine, request)

        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        filename = "test1"
        sheet_name = "Sheet1"

        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse = excel.parse(sheet_name=sheet_name, index_col=0)  # doc

        with pd.ExcelFile(filename + read_ext) as excel:
            df2_parse = excel.parse(index_col=0, sheet_name=sheet_name)

        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"]],
    )
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        # [FUNCTION DOC]
        # Verifies a missing sheet selected through the wrapper's parse method raises a value error (issue 39250) with the same two message variants as the function test. Parametrized over six selections.
        # GH 39250
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile("blank" + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)

    def test_excel_read_buffer(self, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies a wrapper built on an open binary file, read through the reader function, equals reading the path directly with the same engine.
        pth = "test1" + read_ext
        expected = pd.read_excel(pth, sheet_name="Sheet1", index_col=0, engine=engine)

        with open(pth, "rb") as f:
            with pd.ExcelFile(f) as xls:
                actual = pd.read_excel(xls, sheet_name="Sheet1", index_col=0)

        tm.assert_frame_equal(expected, actual)

    def test_reader_closes_file(self, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies that closing the wrapper context closes the underlying file passed in by the caller.
        with open("test1" + read_ext, "rb") as f:
            with pd.ExcelFile(f) as xlsx:
                # parses okay
                pd.read_excel(xlsx, sheet_name="Sheet1", index_col=0, engine=engine)

        assert f.closed

    def test_conflicting_excel_engines(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that passing an engine to the reader function together with a wrapper object raises a value error saying the engine should not be specified in that case (issue 26566).
        # GH 26566
        msg = "Engine should not be specified when passing an ExcelFile"

        with pd.ExcelFile("test1" + read_ext) as xl:
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine="foo")

    def test_excel_read_binary(self, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies reading from an in-memory bytes buffer equals reading the path (issue 15914).
        # GH 15914
        expected = pd.read_excel("test1" + read_ext, engine=engine)

        with open("test1" + read_ext, "rb") as f:
            data = f.read()

        actual = pd.read_excel(BytesIO(data), engine=engine)
        tm.assert_frame_equal(expected, actual)

    def test_excel_read_binary_via_read_excel(self, read_ext, engine):
        # [FUNCTION DOC]
        # Verifies reading from an open binary file through the reader function with an engine equals reading the path (issue 38424).
        # GH 38424
        with open("test1" + read_ext, "rb") as f:
            result = pd.read_excel(f, engine=engine)
        expected = pd.read_excel("test1" + read_ext, engine=engine)
        tm.assert_frame_equal(result, expected)

    def test_read_excel_header_index_out_of_range(self, engine):
        # [FUNCTION DOC]
        # Verifies a header list that exceeds the data raises a value error saying it exceeds the maximum (issue 43143). Takes the engine fixture but does not pass it on.
        # GH#43143
        with open("df_header_oob.xlsx", "rb") as f:
            with pytest.raises(ValueError, match="exceeds maximum"):
                pd.read_excel(f, header=[0, 1])

    @pytest.mark.parametrize("filename", ["df_empty.xlsx", "df_equals.xlsx"])
    def test_header_with_index_col(self, filename):
        # [FUNCTION DOC]
        # Verifies a two-row header with the first column as index on two files gives a multi-level column index with named levels and duplicate names made unique (issue 33476). Parametrized over two file names.
        # GH 33476
        idx = Index(["Z"], name="I2")
        cols = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
        expected = DataFrame([[1, 3]], index=idx, columns=cols, dtype="int64")
        result = pd.read_excel(
            filename, sheet_name="Sheet1", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(expected, result)

    def test_read_datetime_multiindex(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies a multi-level column header consisting of dates is read as datetime index levels with the expected unit and python datetime level names (issue 34748). The data part is empty. Expected failure for the binary engine.
        # GH 34748
        xfail_datetimes_with_pyxlsb(engine, request)

        f = "test_datetime_mi" + read_ext
        with pd.ExcelFile(f) as excel:
            actual = pd.read_excel(excel, header=[0, 1], index_col=0, engine=engine)

        unit = get_exp_unit(read_ext, engine)

        dti = pd.DatetimeIndex(["2020-02-29", "2020-03-01"], dtype=f"M8[{unit}]")
        expected_column_index = MultiIndex.from_arrays(
            [dti[:1], dti[1:]],
            names=[
                dti[0].to_pydatetime(),
                dti[1].to_pydatetime(),
            ],
        )
        expected = DataFrame([], index=[], columns=expected_column_index)

        tm.assert_frame_equal(expected, actual)

    def test_engine_invalid_option(self, read_ext):
        # [FUNCTION DOC]
        # Verifies that setting the reader option for an extension to an invalid engine name raises a value error saying the value must be one of the allowed values.
        # read_ext includes the '.' hence the weird formatting
        with pytest.raises(ValueError, match="Value must be one of *"):
            with pd.option_context(f"io.excel{read_ext}.reader", "abc"):
                pass

    def test_ignore_chartsheets(self, request, engine, read_ext):
        # [FUNCTION DOC]
        # Verifies the wrapper lists only worksheet names, not chart sheets (issue 41448): the sheet names are the single worksheet. Skipped for the open document format; expected failure for pyxlsb.
        # GH 41448
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        with pd.ExcelFile("chartsheet" + read_ext) as excel:
            assert excel.sheet_names == ["Sheet1"]

    def test_corrupt_files_closed(self, engine, tmp_excel):
        # [FUNCTION DOC]
        # Verifies that constructing the wrapper on a corrupt file closes the file without emitting any warning (issue 41778).
        # Skipped for automatic engine selection. Writes the text corrupt to the temporary file; construction is expected to raise an engine-specific error (a bad zip error, xlrd's own error, or calamine's error) which is swallowed; any warning (such as an unclosed resource) fails the test.
        # GH41778
        errors = (BadZipFile,)
        if engine is None:
            pytest.skip(f"Invalid test for engine={engine}")
        elif engine == "xlrd":
            import xlrd

            errors = (BadZipFile, xlrd.biffh.XLRDError)
        elif engine == "calamine":
            from python_calamine import CalamineError

            errors = (CalamineError,)

        Path(tmp_excel).write_text("corrupt", encoding="utf-8")
        with tm.assert_produces_warning(False):
            try:
                pd.ExcelFile(tmp_excel, engine=engine)
            except errors:
                pass
