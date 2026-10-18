"""
Tests encoding functionality during parsing
for all of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture, except where noted) of encoding
# handling: bytes input with a non-UTF-8 encoding, UTF-16 files with skipped rows and
# byte-order marks, UTF-8 byte-order marks, encoding name aliases, binary-mode file
# objects, temporary files, named temporary files, special characters, memory mapping
# with multibyte characters, and unreadable (write-only) handles.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.
# Defines a marker alias that skips tests for the pyarrow engine.

from io import (
    BytesIO,
    TextIOWrapper,
)
import os
import tempfile

import numpy as np
import pytest

from pandas import (
    DataFrame,
    read_csv,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_bytes_io_input(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading a binary stream containing Hebrew text encoded as cp1255 with the
    # matching encoding and a colon separator gives a one-row frame with Hebrew column
    # names.
    encoding = "cp1255"
    parser = all_parsers

    data = BytesIO("שלום:1234\n562:123".encode(encoding))
    result = parser.read_csv(data, sep=":", encoding=encoding)

    expected = DataFrame([[562, 123]], columns=["שלום", "1234"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
def test_read_csv_unicode(all_parsers):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies UTF-8 bytes containing a non-ASCII letter read with
    # a semicolon separator and no header give a row with the full string and an
    # integer.
    parser = all_parsers
    data = BytesIO("\u0141aski, Jan;1".encode())

    result = parser.read_csv(data, sep=";", encoding="utf-8", header=None)
    expected = DataFrame([["\u0141aski, Jan", 1]])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize("sep", [",", "\t"])
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16le", "utf-16be"])
def test_utf16_bom_skiprows(all_parsers, sep, encoding, temp_file):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Writes a UTF-16 (with byte-order mark, little endian, big
    # endian) encoded file with two junk lines and comma or tab delimiters, and verifies
    # reading with the given encoding and skiprows equals reading the same text from a
    # UTF-8 text stream (regression for BOM handling with skiprows).
    # Side effect: writes bytes to the temporary file.
    # see gh-2298
    parser = all_parsers
    data = """skip this
skip this too
A,B,C
1,2,3
4,5,6""".replace(",", sep)
    kwargs = {"sep": sep, "skiprows": 2}
    utf8 = "utf-8"

    bytes_data = data.encode(encoding)

    with open(temp_file, "wb") as f:
        f.write(bytes_data)

    with TextIOWrapper(BytesIO(data.encode(utf8)), encoding=utf8) as bytes_buffer:
        result = parser.read_csv(temp_file, encoding=encoding, **kwargs)
        expected = parser.read_csv(bytes_buffer, encoding=utf8, **kwargs)
    tm.assert_frame_equal(result, expected)


def test_utf16_example(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Verifies a sample UTF-16 tab-separated file is read into 50 rows.
    path = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser = all_parsers
    result = parser.read_csv(path, encoding="utf-16", sep="\t")
    assert len(result) == 50


def test_unicode_encoding(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Verifies a latin-1 encoded file read with no header and its first column as index:
    # the value in row 1632, second column, equals a specific Icelandic movie title
    # string.
    path = os.path.join(csv_dir_path, "unicode_series.csv")
    parser = all_parsers

    result = parser.read_csv(path, header=None, encoding="latin-1")
    result = result.set_index(0)
    got = result[1][1632]

    expected = "\xc1 k\xf6ldum klaka (Cold Fever) (1994)"
    assert got == expected


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # Basic test
        ("a\n1", {}, [1]),
        # "Regular" quoting
        ('"a"\n1', {"quotechar": '"'}, [1]),
        # Test in a data row instead of header
        ("b\n1", {"names": ["a"]}, ["b", "1"]),
        # Test in empty data row with skipping
        ("\n1", {"names": ["a"], "skip_blank_lines": True}, [1]),
        # Test in empty data row without skipping
        (
            "\n1",
            {"names": ["a"], "skip_blank_lines": False},
            [np.nan, 1],
        ),
    ],
)
def test_utf8_bom(all_parsers, data, kwargs, expected):
    # [FUNCTION DOC]
    # Parametrized over inputs with a UTF-8 byte-order mark prepended: simple data,
    # quoted header, a name given for a data-like header, and blank lines with skipping
    # on or off. Verifies the BOM is removed and the single column a has the expected
    # values (including NaN for a skipped-off blank line). The pyarrow case with a
    # leading blank line and skipping on is skipped (upstream parse error).
    # see gh-4793
    parser = all_parsers
    bom = "\ufeff"
    utf8 = "utf-8"

    def _encode_data_with_bom(_data):
        # [FUNCTION DOC]
        # HELPER: takes a text, prepends a byte-order mark character, encodes it as
        # UTF-8 and returns it wrapped in an in-memory binary stream.
        bom_data = (bom + _data).encode(utf8)
        return BytesIO(bom_data)

    if (
        parser.engine == "pyarrow"
        and data == "\n1"
        and kwargs.get("skip_blank_lines", True)
    ):
        # CSV parse error: Empty CSV file or block: cannot infer number of columns
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    result = parser.read_csv(_encode_data_with_bom(data), encoding=utf8, **kwargs)
    expected = DataFrame({"a": expected})
    tm.assert_frame_equal(result, expected)


def test_read_csv_utf_aliases(all_parsers, utf_value, encoding_fmt):
    # [FUNCTION DOC]
    # Verifies that every spelling of UTF-8, UTF-16 and UTF-32 encoding names (case and
    # separator variants) can decode bytes encoded in that encoding and read a small
    # two-column frame.
    # see gh-13549
    expected = DataFrame({"mb_num": [4.8], "multibyte": ["test"]})
    parser = all_parsers

    encoding = encoding_fmt.format(utf_value)
    data = "mb_num,multibyte\n4.8,test".encode(encoding)

    result = parser.read_csv(BytesIO(data), encoding=encoding)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_path,encoding",
    [
        (("io", "data", "csv", "test1.csv"), "utf-8"),
        (("io", "parser", "data", "unicode_series.csv"), "latin-1"),
        (("io", "parser", "data", "sauron.SHIFT_JIS.csv"), "shiftjis"),
    ],
)
def test_binary_mode_file_buffers(all_parsers, file_path, encoding, datapath):
    # [FUNCTION DOC]
    # Parametrized over three data files and their encodings. Verifies reading the file
    # from a text-mode handle opened with the encoding, from a buffered binary handle
    # (with encoding given) and from an unbuffered binary handle gives the same frame as
    # reading by path, and that none of the handles is closed by the reader (regression
    # for the python engine failing on binary handles).
    # gh-23779: Python csv engine shouldn't error on files opened in binary.
    # gh-31575: Python csv engine shouldn't error on files opened in raw binary.
    parser = all_parsers

    fpath = datapath(*file_path)
    expected = parser.read_csv(fpath, encoding=encoding)

    with open(fpath, encoding=encoding) as fa:
        result = parser.read_csv(fa)
        assert not fa.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb") as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb", buffering=0) as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("pass_encoding", [True, False])
def test_encoding_temp_file(
    all_parsers, utf_value, encoding_fmt, pass_encoding, temp_file
):
    # [FUNCTION DOC]
    # Writes a small file in a UTF-8, UTF-16 or UTF-32 variant encoding through a
    # temporary file opened for reading and writing, rewinds, and verifies reading from
    # the handle works whether or not the encoding is passed. Pyarrow with an explicit
    # 16 or 32-bit encoding is skipped because those cases freeze (marked as a fixme in
    # the code).
    # see gh-24130
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)

    if parser.engine == "pyarrow" and pass_encoding is True and utf_value in [16, 32]:
        # FIXME: this is bad!
        pytest.skip("These cases freeze")

    expected = DataFrame({"foo": ["bar"]})

    with temp_file.open(mode="w+", encoding=encoding) as f:
        f.write("foo\nbar")
        f.seek(0)

        result = parser.read_csv(f, encoding=encoding if pass_encoding else None)
        tm.assert_frame_equal(result, expected)


def test_encoding_named_temp_file(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading from a named temporary file holding Shift-JIS encoded Japanese
    # title and data with the matching encoding gives a one-row frame with the Japanese
    # column name, and that the handle remains open.
    # see gh-31819
    parser = all_parsers
    encoding = "shift-jis"

    title = "てすと"
    data = "こむ"

    expected = DataFrame({title: [data]})

    with tempfile.NamedTemporaryFile() as f:
        f.write(f"{title}\n{data}".encode(encoding))

        f.seek(0)

        result = parser.read_csv(f, encoding=encoding)
        tm.assert_frame_equal(result, expected)
        assert not f.closed


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-16", "utf-16-be", "utf-16-le", "utf-32"]
)
def test_parse_encoded_special_characters(encoding):
    # [FUNCTION DOC]
    # Verifies tab-separated data containing a fullwidth colon character, encoded in
    # several UTF encodings, is parsed correctly.
    # Note: this test calls the top-level reader directly and does not use the parser
    # fixture, so it runs with the default engine only.
    # GH16218 Verify parsing of data with encoded special characters
    # Data contains a Unicode 'FULLWIDTH COLON' (U+FF1A) at position (0,"a")
    data = "a\tb\n：foo\t0\nbar\t1\nbaz\t2"  # noqa: RUF001
    encoded_data = BytesIO(data.encode(encoding))
    result = read_csv(encoded_data, delimiter="\t", encoding=encoding)

    expected = DataFrame(
        data=[["：foo", 0], ["bar", 1], ["baz", 2]],  # noqa: RUF001
        columns=["a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", ["utf-8", None, "utf-16", "cp1255", "latin-1"])
def test_encoding_memory_map(all_parsers, encoding, temp_file):
    # [FUNCTION DOC]
    # Writes a small frame in various encodings (including none and utf-16) and verifies
    # reading it with memory mapping gives the same frame. For pyarrow, verifies the
    # unsupported memory-map error.
    # GH40986
    parser = all_parsers
    expected = DataFrame(
        {
            "name": ["Raphael", "Donatello", "Miguel Angel", "Leonardo"],
            "mask": ["red", "purple", "orange", "blue"],
            "weapon": ["sai", "bo staff", "nunchunk", "katana"],
        }
    )
    expected.to_csv(temp_file, index=False, encoding=encoding)

    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(temp_file, encoding=encoding, memory_map=True)
        return

    df = parser.read_csv(temp_file, encoding=encoding, memory_map=True)
    tm.assert_frame_equal(df, expected)


def test_chunk_splits_multibyte_char(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Regression for a multibyte character split across the 262144-byte internal chunk
    # boundary with memory mapping: writes 2048 rows of 127 characters whose last row
    # ends in a two-byte UTF-8 character placed at the chunk end, and verifies the data
    # is read back equal. For pyarrow, verifies the unsupported memory-map error.
    """
    Chunk splits a multibyte character with memory_map=True

    GH 43540
    """
    parser = all_parsers
    # DEFAULT_CHUNKSIZE = 262144, defined in parsers.pyx
    df = DataFrame(data=["a" * 127] * 2048)

    # Put two-bytes utf-8 encoded character "ą" at the end of chunk
    # utf-8 encoding of "ą" is b'\xc4\x85'
    df.iloc[2047] = "a" * 127 + "ą"
    df.to_csv(temp_file, index=False, header=False, encoding="utf-8")

    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(temp_file, header=None, memory_map=True)
        return

    dfr = parser.read_csv(temp_file, header=None, memory_map=True)
    tm.assert_frame_equal(dfr, df)


def test_readcsv_memmap_utf8(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Regression for UTF-8 handling with memory mapping: builds 128-character lines of
    # consecutive unicode characters from space up to a supplementary plane character,
    # skipping lines that cannot be encoded, writes them as UTF-8, and verifies reading
    # back with memory mapping and explicit UTF-8 encoding gives an equal frame. For
    # pyarrow, verifies the unsupported memory-map error.
    """
    GH 43787

    Test correct handling of UTF-8 chars when memory_map=True and encoding is UTF-8
    """
    lines = []
    line_length = 128
    start_char = " "
    end_char = "\U00010080"
    # This for loop creates a list of 128-char strings
    # consisting of consecutive Unicode chars
    for lnum in range(ord(start_char), ord(end_char), line_length):
        line = "".join([chr(c) for c in range(lnum, lnum + 0x80)]) + "\n"
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            continue
        lines.append(line)
    parser = all_parsers
    df = DataFrame(lines)
    df.to_csv(temp_file, index=False, header=False, encoding="utf-8")

    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(temp_file, header=None, memory_map=True, encoding="utf-8")
        return

    dfr = parser.read_csv(temp_file, header=None, memory_map=True, encoding="utf-8")
    tm.assert_frame_equal(df, dfr)


@pytest.mark.usefixtures("pyarrow_xfail")
@pytest.mark.parametrize("mode", ["w+b", "w+t"])
def test_not_readable(all_parsers, mode):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow. With a spooled temporary file in binary or
    # text mode holding the text abcd, verifies reading it yields an empty frame with a
    # single column named abcd (the content is treated as a header).
    # GH43439
    parser = all_parsers
    content = b"abcd"
    if "t" in mode:
        content = "abcd"
    with tempfile.SpooledTemporaryFile(mode=mode, encoding="utf-8") as handle:
        handle.write(content)
        handle.seek(0)
        df = parser.read_csv(handle)
    expected = DataFrame([], columns=["abcd"])
    tm.assert_frame_equal(df, expected)
