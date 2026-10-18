# [MODULE DOC]
# Tests for reading and writing line-delimited JSON (one record per line) with the JSON
# reader and the DataFrame JSON writer, including chunked reading, row limits, append
# mode, and the pyarrow engine.
# Relies on the testing helpers for frame/series comparison and on the internal JSON
# reader class for stream-closing checks. The module-level mark ignores the
# deprecation warning about passing a block manager to a DataFrame.
# Defines fixtures: a line-delimited JSON string and an engine selector.

from collections.abc import Iterator
from io import (
    BytesIO,
    StringIO,
)
from pathlib import Path

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    read_json,
)
import pandas._testing as tm

from pandas.io.json._json import JsonReader

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def lines_json_df():
    # [FUNCTION DOC]
    # HELPER: fixture for the tests in this module.
    # Returns a line-delimited JSON string (records orientation) of a three-row frame
    # with integer columns A and B.
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    return df.to_json(lines=True, orient="records")


@pytest.fixture(params=["ujson", "pyarrow"])
def engine(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over the two JSON engines (the default one and
    # pyarrow).
    # Skips the test when the pyarrow JSON module is unavailable and returns the engine
    # name.
    if request.param == "pyarrow":
        pytest.importorskip("pyarrow.json")
    return request.param


def test_read_jsonl():
    # [FUNCTION DOC]
    # Reads a line-delimited JSON string with irregular whitespace between keys and
    # values and verifies it yields a two-row frame with columns a and b.
    # GH9180
    result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
    expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_engine_pyarrow(datapath, engine):
    # [FUNCTION DOC]
    # Reads the line-delimited JSON data file from the test data directory with each
    # engine and verifies the three-row frame of columns a and b.
    result = read_json(
        datapath("io", "json", "data", "line_delimited.json"),
        lines=True,
        engine=engine,
    )
    expected = DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    tm.assert_frame_equal(result, expected)


def test_read_datetime(request, engine):
    # [FUNCTION DOC]
    # Reads a line-delimited record holding a list column and date-like strings and
    # verifies the list is expanded into rows while date strings stay strings.
    # The pyarrow engine is expected to fail with a value error because it only accepts
    # file paths.
    # GH33787
    if engine == "pyarrow":
        # GH 48893
        reason = "Pyarrow only supports a file path as an input and line delimited json"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    df = DataFrame(
        [([1, 2], ["2020-03-05", "2020-04-08T09:58:49+00:00"], "hector")],
        columns=["accounts", "date", "name"],
    )
    json_line = df.to_json(lines=True, orient="records")

    if engine == "pyarrow":
        result = read_json(StringIO(json_line), engine=engine)
    else:
        result = read_json(StringIO(json_line), engine=engine)
    expected = DataFrame(
        [[1, "2020-03-05", "hector"], [2, "2020-04-08T09:58:49+00:00", "hector"]],
        columns=["accounts", "date", "name"],
    )
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_unicode_chars():
    # [FUNCTION DOC]
    # Reads line-delimited JSON containing a non-ASCII quotation character from a text
    # stream and verifies the character is preserved, using two equivalent stream
    # constructions.
    # GH15132: non-ascii unicode characters
    # \u201d == RIGHT DOUBLE QUOTATION MARK

    # simulate file handle
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    json = StringIO(json)
    result = read_json(json, lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # simulate string
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    result = read_json(StringIO(json), lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_to_jsonl():
    # [FUNCTION DOC]
    # Verifies that writing in records orientation with lines enabled produces one
    # compact JSON object per line with a trailing newline.
    # Checks quote and brace escaping, and escaped backslashes in both column names and
    # data, and round-trips the escaped output through the reader.
    # GH9180
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
    assert result == expected

    df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    # GH15096: escaped characters in columns and data
    df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)


def test_to_jsonl_count_new_lines():
    # [FUNCTION DOC]
    # Verifies the line-delimited JSON writer emits exactly one newline per record (two
    # for a two-row frame).
    # GH36888
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    actual_new_lines_count = df.to_json(orient="records", lines=True).count("\n")
    expected_new_lines_count = 2
    assert actual_new_lines_count == expected_new_lines_count


@pytest.mark.parametrize("chunksize", [1, 1.0])
@pytest.mark.parametrize("buffer", [BytesIO, StringIO])
def test_readjson_chunks(request, lines_json_df, chunksize, buffer, engine):
    # [FUNCTION DOC]
    # Verifies reading in chunks (chunk size given as integer or float one) over text
    # and bytes buffers and concatenating the chunks gives the same frame as an
    # unchunked read.
    # The pyarrow engine is expected to fail with a value error since it does not
    # support a chunk size.
    # Basic test that read_json(chunks=True) gives the same result as
    # read_json(chunks=False)
    # GH17048: memory usage when lines=True
    # GH#28906: read binary json lines in chunks

    if buffer == BytesIO:
        lines_json_df = lines_json_df.encode()

    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    unchunked = read_json(buffer(lines_json_df), lines=True)
    with (
        buffer(lines_json_df) as buf,
        read_json(buf, lines=True, chunksize=chunksize, engine=engine) as reader,
    ):
        chunked = pd.concat(reader)

    tm.assert_frame_equal(chunked, unchunked)


def test_readjson_chunksize_requires_lines(lines_json_df, engine):
    # [FUNCTION DOC]
    # Verifies that passing a chunk size without enabling lines raises a value error
    # saying chunksize can only be passed if lines is true.
    msg = "chunksize can only be passed if lines=True"
    with pytest.raises(ValueError, match=msg):
        with read_json(
            StringIO(lines_json_df), lines=False, chunksize=2, engine=engine
        ) as _:
            pass


def test_readjson_chunks_series(request, engine):
    # [FUNCTION DOC]
    # Verifies chunked line-delimited reading with series output reproduces the
    # unchunked series after concatenation.
    # The pyarrow engine is expected to fail (no chunk size support).
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    # Test reading line-format JSON to Series with chunksize param
    s = pd.Series({"A": 1, "B": 2})

    strio = StringIO(s.to_json(lines=True, orient="records"))
    unchunked = read_json(strio, lines=True, typ="series", engine=engine)

    strio = StringIO(s.to_json(lines=True, orient="records"))
    with read_json(
        strio, lines=True, typ="series", chunksize=1, engine=engine
    ) as reader:
        chunked = pd.concat(reader)

    tm.assert_series_equal(chunked, unchunked)


def test_readjson_each_chunk(request, lines_json_df, engine):
    # [FUNCTION DOC]
    # Verifies the intermediate chunks: with a chunk size of two on three records, the
    # first chunk has shape two by two and the second one by two.
    # The pyarrow engine is expected to fail with a value error.
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # Other tests check that the final result of read_json(chunksize=True)
    # is correct. This checks the intermediate chunks.
    with read_json(
        StringIO(lines_json_df), lines=True, chunksize=2, engine=engine
    ) as reader:
        chunks = list(reader)
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (1, 2)


def test_readjson_chunks_from_file(request, engine, temp_file):
    # [FUNCTION DOC]
    # Writes a line-delimited JSON file to a temporary path, reads it in one-row chunks,
    # and verifies the concatenation equals the unchunked read.
    # The pyarrow engine is expected to fail with a value error.
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.to_json(temp_file, lines=True, orient="records")
    with read_json(temp_file, lines=True, chunksize=1, engine=engine) as reader:
        chunked = pd.concat(reader)
    unchunked = read_json(temp_file, lines=True, engine=engine)
    tm.assert_frame_equal(unchunked, chunked)


@pytest.mark.parametrize("chunksize", [None, 1])
def test_readjson_chunks_closes(chunksize, temp_file):
    # [FUNCTION DOC]
    # Verifies the underlying file handle of the JSON reader is closed after reading
    # inside a context manager, both without a chunk size and with a chunk size of one.
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.to_json(temp_file, lines=True, orient="records")
    reader = JsonReader(
        temp_file,
        orient=None,
        typ="frame",
        dtype=True,
        convert_axes=True,
        convert_dates=True,
        keep_default_dates=True,
        precise_float=False,
        date_unit=None,
        encoding=None,
        lines=True,
        chunksize=chunksize,
        compression=None,
        nrows=None,
    )
    with reader:
        reader.read()
    assert reader.handles.handle.closed, (
        f"didn't close stream with chunksize = {chunksize}"
    )


@pytest.mark.parametrize("chunksize", [0, -1, 2.2, "foo"])
def test_readjson_invalid_chunksize(lines_json_df, chunksize, engine):
    # [FUNCTION DOC]
    # Verifies invalid chunk sizes (zero, negative, non-integer float, string) raise a
    # value error stating chunksize must be an integer of at least one.
    msg = r"'chunksize' must be an integer >=1"

    with pytest.raises(ValueError, match=msg):
        with read_json(
            StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine
        ) as _:
            pass


@pytest.mark.parametrize("chunksize", [None, 1, 2])
def test_readjson_chunks_multiple_empty_lines(chunksize):
    # [FUNCTION DOC]
    # Verifies that blank lines and indentation between line-delimited records are
    # ignored, with no chunking and with chunk sizes one and two, yielding the three-row
    # expected frame.
    j = """

    {"A":1,"B":4}



    {"A":2,"B":5}







    {"A":3,"B":6}
    """
    orig = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test = read_json(StringIO(j), lines=True, chunksize=chunksize)
    if chunksize is not None:
        with test:
            test = pd.concat(test)
    tm.assert_frame_equal(orig, test, obj=f"chunksize: {chunksize}")


def test_readjson_unicode(request, monkeypatch, engine, temp_file):
    # [FUNCTION DOC]
    # Verifies reading a UTF-8 file containing non-ASCII column names and values works
    # even when the preferred locale encoding is patched to a different one (cp949) via
    # the monkeypatch fixture.
    # The pyarrow engine is expected to fail with a value error.
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    monkeypatch.setattr("locale.getpreferredencoding", lambda do_setlocale: "cp949")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write('{"£©µÀÆÖÞßéöÿ":["АБВГДабвгд가"]}')

    result = read_json(temp_file, engine=engine)
    expected = DataFrame({"£©µÀÆÖÞßéöÿ": ["АБВГДабвгд가"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1, 2])
def test_readjson_nrows(nrows, engine):
    # [FUNCTION DOC]
    # Verifies the row limit option for line-delimited reading returns only the first
    # requested rows (one or two) of a four-record input.
    # GH 33916
    # Test reading line-format JSON to Series with nrows param
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    result = read_json(StringIO(jsonl), lines=True, nrows=nrows)
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows,chunksize", [(2, 2), (4, 2)])
def test_readjson_nrows_chunks(request, nrows, chunksize, engine):
    # [FUNCTION DOC]
    # Verifies combining a row limit with a chunk size (two with two, four with two)
    # gives the concatenated result equal to the first requested rows.
    # The pyarrow engine is expected to fail with a value error; note that for that
    # engine the raw string is passed in place of a stream.
    # GH 33916
    # Test reading line-format JSON to Series with nrows and chunksize param
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""

    if engine != "pyarrow":
        with read_json(
            StringIO(jsonl), lines=True, nrows=nrows, chunksize=chunksize, engine=engine
        ) as reader:
            chunked = pd.concat(reader)
    else:
        with read_json(
            jsonl, lines=True, nrows=nrows, chunksize=chunksize, engine=engine
        ) as reader:
            chunked = pd.concat(reader)
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(chunked, expected)


def test_readjson_nrows_requires_lines(engine):
    # [FUNCTION DOC]
    # Verifies that giving a row limit without enabling lines raises a value error
    # saying nrows can only be passed if lines is true.
    # GH 33916
    # Test ValueError raised if nrows is set without setting lines in read_json
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    msg = "nrows can only be passed if lines=True"
    with pytest.raises(ValueError, match=msg):
        read_json(jsonl, lines=False, nrows=2, engine=engine)


def test_readjson_lines_chunks_fileurl(request, datapath, engine):
    # [FUNCTION DOC]
    # Reads a line-delimited JSON file through a file URL in one-row chunks and verifies
    # each chunk equals the matching single-row expected frame (with the right index
    # label).
    # The pyarrow engine is expected to fail with a value error.
    # GH 27135
    # Test reading line-format JSON from file url
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    df_list_expected = [
        DataFrame([[1, 2]], columns=["a", "b"], index=[0]),
        DataFrame([[3, 4]], columns=["a", "b"], index=[1]),
        DataFrame([[5, 6]], columns=["a", "b"], index=[2]),
    ]
    os_path = datapath("io", "json", "data", "line_delimited.json")
    file_url = Path(os_path).as_uri()
    with read_json(file_url, lines=True, chunksize=1, engine=engine) as url_reader:
        for index, chuck in enumerate(url_reader):
            tm.assert_frame_equal(chuck, df_list_expected[index])


def test_chunksize_is_incremental():
    # [FUNCTION DOC]
    # Verifies chunked reading is incremental: reads a large repeated line-delimited
    # input through a counting reader object with a chunk size of 100, and verifies more
    # than one chunk is produced and the underlying read method is called more than ten
    # times.
    # See https://github.com/pandas-dev/pandas/issues/34548
    jsonl = (
        """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}\n"""
        * 1000
    )

    class MyReader:
        # [CLASS DOC]
        # HELPER: minimal file-like reader used by the incremental-chunk test.
        # Wraps a text stream and counts how many times reading or iteration is
        # requested via the read_count attribute.
        def __init__(self, contents) -> None:
            # [FUNCTION DOC]
            # HELPER: constructor. Takes the contents string, initializes the read
            # counter to zero and wraps the contents in a text stream.
            self.read_count = 0
            self.stringio = StringIO(contents)

        def read(self, *args):
            # [FUNCTION DOC]
            # HELPER: increments the read counter and delegates to the wrapped stream's
            # read with the same arguments, returning the data read.
            self.read_count += 1
            return self.stringio.read(*args)

        def __iter__(self) -> Iterator:
            # [FUNCTION DOC]
            # HELPER: increments the read counter and returns an iterator over the
            # wrapped stream's lines.
            self.read_count += 1
            return iter(self.stringio)

    reader = MyReader(jsonl)
    assert len(list(read_json(reader, lines=True, chunksize=100))) > 1
    assert reader.read_count > 10


@pytest.mark.parametrize("orient_", ["split", "index", "table"])
def test_to_json_append_orient(orient_):
    # [FUNCTION DOC]
    # Verifies that appending to a file with the writer in append mode combined with the
    # split, index, or table orientation raises a value error explaining append is only
    # supported with lines enabled and records orientation.
    # GH 35849
    # Test ValueError when orient is not 'records'
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    msg = (
        r"mode='a' \(append\) is only supported when "
        "lines is True and orient is 'records'"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode="a", orient=orient_)


def test_to_json_append_lines():
    # [FUNCTION DOC]
    # Verifies that append mode with lines disabled (records orientation) raises the
    # same value error about append requiring lines and records orientation.
    # GH 35849
    # Test ValueError when lines is not True
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    msg = (
        r"mode='a' \(append\) is only supported when "
        "lines is True and orient is 'records'"
    )
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode="a", lines=False, orient="records")


@pytest.mark.parametrize("mode_", ["r", "x"])
def test_to_json_append_mode(mode_):
    # [FUNCTION DOC]
    # Verifies that modes other than write and append (read, exclusive create) raise a
    # value error stating the mode is not a valid option.
    # GH 35849
    # Test ValueError when mode is not supported option
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    msg = (
        f"mode={mode_} is not a valid option.Only 'w' and 'a' are currently supported."
    )
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode=mode_, lines=False, orient="records")


def test_to_json_append_output_consistent_columns(temp_file):
    # [FUNCTION DOC]
    # Appends two frames with identical columns to one temporary file in line-delimited
    # form and verifies the file reads back as the four-row concatenation.
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing same columns, new rows
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})

    expected = DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]})
    # Save dataframes to the same file
    df1.to_json(temp_file, lines=True, orient="records")
    df2.to_json(temp_file, mode="a", lines=True, orient="records")

    # Read path file
    result = read_json(temp_file, lines=True)
    tm.assert_frame_equal(result, expected)


def test_to_json_append_output_inconsistent_columns(temp_file):
    # [FUNCTION DOC]
    # Appends two frames with partially overlapping columns to a file and verifies the
    # read-back frame has missing values where a column was absent from a record.
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing one new column, one old column, new rows
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})

    expected = DataFrame(
        {
            "col1": [1, 2, None, None],
            "col2": ["a", "b", "e", "f"],
            "col3": [np.nan, np.nan, "!", "#"],
        }
    )
    # Save dataframes to the same file
    df1.to_json(temp_file, mode="a", lines=True, orient="records")
    df3.to_json(temp_file, mode="a", lines=True, orient="records")

    # Read path file
    result = read_json(temp_file, lines=True)
    tm.assert_frame_equal(result, expected)


def test_to_json_append_output_different_columns(temp_file):
    # [FUNCTION DOC]
    # Appends four frames with same, differing and new columns and verifies the
    # read-back frame is the union of columns with missing values, with the boolean
    # column converted to float because of the missing entries.
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing same, differing and new columns
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})
    df4 = DataFrame({"col4": [True, False]})

    expected = DataFrame(
        {
            "col1": [1, 2, 3, 4, None, None, None, None],
            "col2": ["a", "b", "c", "d", "e", "f", np.nan, np.nan],
            "col3": [np.nan, np.nan, np.nan, np.nan, "!", "#", np.nan, np.nan],
            "col4": [None, None, None, None, None, None, True, False],
        }
    ).astype({"col4": "float"})
    # Save dataframes to the same file
    df1.to_json(temp_file, mode="a", lines=True, orient="records")
    df2.to_json(temp_file, mode="a", lines=True, orient="records")
    df3.to_json(temp_file, mode="a", lines=True, orient="records")
    df4.to_json(temp_file, mode="a", lines=True, orient="records")

    # Read path file
    result = read_json(temp_file, lines=True)
    tm.assert_frame_equal(result, expected)


def test_to_json_append_output_different_columns_reordered(temp_file):
    # [FUNCTION DOC]
    # Appends four frames in reverse order and verifies the read-back frame's column
    # order follows first appearance in the file (the last-defined frame's columns
    # first), with missing values elsewhere.
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing specific result column order.
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})
    df4 = DataFrame({"col4": [True, False]})

    # df4, df3, df2, df1 (in that order)
    expected = DataFrame(
        {
            "col4": [True, False, None, None, None, None, None, None],
            "col2": [np.nan, np.nan, "e", "f", "c", "d", "a", "b"],
            "col3": [np.nan, np.nan, "!", "#", np.nan, np.nan, np.nan, np.nan],
            "col1": [None, None, None, None, 3, 4, 1, 2],
        }
    ).astype({"col4": "float"})
    # Save dataframes to the same file
    df4.to_json(temp_file, mode="a", lines=True, orient="records")
    df3.to_json(temp_file, mode="a", lines=True, orient="records")
    df2.to_json(temp_file, mode="a", lines=True, orient="records")
    df1.to_json(temp_file, mode="a", lines=True, orient="records")

    # Read path file
    result = read_json(temp_file, lines=True)
    tm.assert_frame_equal(result, expected)
