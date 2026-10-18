"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests for reading from in-memory lists of rows, via the text-parser class that
# accepts a list of lists of strings, and for a simple one-column list-like input.

import csv
from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.parsers import TextParser

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow
def test_read_data_list(all_parsers):
    # [FUNCTION DOC]
    # Marked as expected to fail for the pyarrow engine. Verifies that the text parser
    # fed a list of string rows (header plus two data rows, first column as index, chunk
    # size two) produces, on a full read, the same frame as reading the equivalent CSV
    # text.
    parser = all_parsers
    kwargs = {"index_col": 0}
    data = "A,B,C\nfoo,1,2,3\nbar,4,5,6"

    data_list = [["A", "B", "C"], ["foo", "1", "2", "3"], ["bar", "4", "5", "6"]]
    expected = parser.read_csv(StringIO(data), **kwargs)

    with TextParser(data_list, chunksize=2, **kwargs) as parser:
        result = parser.read()

    tm.assert_frame_equal(result, expected)


def test_reader_list(all_parsers):
    # [FUNCTION DOC]
    # Parses a CSV into a list of rows with the standard csv reader, feeds it to the
    # text parser with an index column and chunk size two, and verifies the three chunks
    # equal the corresponding slices of the frame read directly from the text.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0}

    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, **kwargs) as reader:
        chunks = list(reader)

    expected = parser.read_csv(StringIO(data), **kwargs)

    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


def test_reader_list_skiprows(all_parsers):
    # [FUNCTION DOC]
    # Same as the reader-list test but skipping the row at position one; verifies the
    # first chunk equals the rows at positions one to three of the directly read frame.
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0}

    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, skiprows=[1], **kwargs) as reader:
        chunks = list(reader)

    expected = parser.read_csv(StringIO(data), **kwargs)

    tm.assert_frame_equal(chunks[0], expected[1:3])


def test_read_csv_parse_simple_list(all_parsers):
    # [FUNCTION DOC]
    # Verifies that one-column text containing values with spaces, read with no header,
    # yields a single-column frame of those strings.
    parser = all_parsers
    data = """foo
bar baz
qux foo
foo
bar"""

    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame(["foo", "bar baz", "qux foo", "foo", "bar"])
    tm.assert_frame_equal(result, expected)
