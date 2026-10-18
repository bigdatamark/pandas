"""
Tests compressed data parsing functionality for all
of the parsers defined in parsers.py
"""

# [MODULE DOC]
# Parser tests (all engines via the parser fixture) of reading compressed data: zip
# archives (single, multiple and no members, invalid files), compression inference by
# extension, explicit compression types written with a test helper, file-like inputs,
# UTF encodings inside zip files, invalid compression names, tar archives, and ignoring
# the file extension when compression is disabled.
# KNOWN ISSUE: the module docstring says the parsers are defined in a file named
# parsers.py, but no such module exists here; the parser fixtures come from the
# package's shared fixtures module.

import os
from pathlib import Path
import tarfile
import zipfile

import pytest

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture(params=[True, False])
def buffer(request):
    # [FUNCTION DOC]
    # HELPER: fixture parametrized over true and false, telling a test whether to pass
    # an open file object (buffer) instead of a path. Returns the current boolean.
    return request.param


@pytest.fixture
def parser_and_data(all_parsers, csv1):
    # [FUNCTION DOC]
    # HELPER: fixture that prepares shared inputs. Reads the raw bytes of the sample CSV
    # file and the parsed frame produced by the current parser, and returns a triple of
    # the parser wrapper, the raw bytes and the expected frame.
    parser = all_parsers

    with open(csv1, "rb") as f:
        data = f.read()
    expected = parser.read_csv(csv1)

    return parser, data, expected


@pytest.mark.parametrize("compression", ["zip", "infer", "zip2"])
def test_zip(tmp_path, parser_and_data, compression):
    # [FUNCTION DOC]
    # Writes the sample data into a single-member zip file and verifies reading it with
    # explicit zip compression, inferred compression, or from an open binary file object
    # with explicit zip compression equals the expected frame.
    parser, data, expected = parser_and_data

    path = tmp_path / "test_file.zip"
    with zipfile.ZipFile(path, mode="w") as tmp:
        tmp.writestr("test_file", data)

    if compression == "zip2":
        with open(path, "rb") as f:
            result = parser.read_csv(f, compression="zip")
    else:
        result = parser.read_csv(path, compression=compression)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("compression", ["zip", "infer"])
def test_zip_error_multiple_files(tmp_path, parser_and_data, compression):
    # [FUNCTION DOC]
    # Verifies a zip archive containing two members raises a value error mentioning
    # multiple files, with explicit or inferred compression.
    parser, data, expected = parser_and_data

    path = tmp_path / "combined_zip.zip"
    inner_file_names = ["test_file", "second_file"]

    with zipfile.ZipFile(path, mode="w") as tmp:
        for file_name in inner_file_names:
            tmp.writestr(file_name, data)

    with pytest.raises(ValueError, match="Multiple files"):
        parser.read_csv(path, compression=compression)


def test_zip_error_no_files(tmp_path, parser_and_data):
    # [FUNCTION DOC]
    # Verifies an empty zip archive raises a value error mentioning zero files.
    parser, _, _ = parser_and_data

    path = tmp_path / "test_file.zip"
    with zipfile.ZipFile(path, mode="w"):
        pass

    with pytest.raises(ValueError, match="Zero files"):
        parser.read_csv(path, compression="zip")


def test_zip_error_invalid_zip(tmp_path, parser_and_data):
    # [FUNCTION DOC]
    # Verifies an empty file read as zip raises a bad-zip-file error saying the file is
    # not a zip file.
    parser, _, _ = parser_and_data

    path = tmp_path / "invalid_file.zip"
    path.touch()
    with open(path, "rb") as f:
        with pytest.raises(zipfile.BadZipFile, match="File is not a zip file"):
            parser.read_csv(f, compression="zip")


@pytest.mark.parametrize("filename", [None, "test.{ext}"])
def test_compression(
    tmp_path,
    request,
    parser_and_data,
    compression_only,
    buffer,
    filename,
    compression_to_extension,
):
    # [FUNCTION DOC]
    # Writes the sample data compressed with each supported compression type (with or
    # without a recognizable file name extension) and verifies reading it (from a path
    # or an open binary file) with inferred or explicit compression equals the expected
    # frame. The case of a named file passed as a buffer is expected to fail because
    # compression cannot be deduced from a buffer.
    # Side effect: writes compressed files in the temporary directory.
    parser, data, expected = parser_and_data
    compress_type = compression_only

    ext = compression_to_extension[compress_type]
    filename = filename if filename is None else filename.format(ext=ext)

    if filename and buffer:
        request.applymarker(
            pytest.mark.xfail(
                reason="Cannot deduce compression from buffer of compressed data."
            )
        )

    path = tmp_path / filename if filename else tmp_path / "test_file"
    tm.write_to_compressed(compress_type, path, data)
    compression = "infer" if filename else compress_type

    if buffer:
        with open(path, "rb") as f:
            result = parser.read_csv(f, compression=compression)
    else:
        result = parser.read_csv(path, compression=compression)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ext", [None, "gz", "bz2"])
def test_infer_compression(all_parsers, csv1, buffer, ext):
    # [FUNCTION DOC]
    # Verifies that reading the sample file with compression inferred works for the
    # plain file and for gzip or bzip2 variants of it (by extension), and for an open
    # text buffer, with the first column as index and date parsing.
    # see gh-9770
    parser = all_parsers
    kwargs = {"index_col": 0, "parse_dates": True}

    expected = parser.read_csv(csv1, **kwargs)
    kwargs["compression"] = "infer"

    if buffer:
        with open(csv1, encoding="utf-8") as f:
            result = parser.read_csv(f, **kwargs)
    else:
        ext = "." + ext if ext else ""
        result = parser.read_csv(csv1 + ext, **kwargs)

    tm.assert_frame_equal(result, expected)


def test_compression_utf_encoding(all_parsers, csv_dir_path, utf_value, encoding_fmt):
    # [FUNCTION DOC]
    # Verifies reading a zip archive of a small tab-separated file encoded in UTF-8,
    # UTF-16 or UTF-32 (with every case and separator spelling of the encoding name)
    # yields the expected two-row frame with accented characters.
    # see gh-18071, gh-24130
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)
    path = os.path.join(csv_dir_path, f"utf{utf_value}_ex_small.zip")

    result = parser.read_csv(path, encoding=encoding, compression="zip", sep="\t")
    expected = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("invalid_compression", ["sfark", "bz3", "zipper"])
def test_invalid_compression(all_parsers, invalid_compression):
    # [FUNCTION DOC]
    # Verifies unknown compression names raise a value error saying the compression type
    # is unrecognized.
    parser = all_parsers
    compress_kwargs = {"compression": invalid_compression}

    msg = f"Unrecognized compression type: {invalid_compression}"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test_file.zip", **compress_kwargs)


def test_compression_tar_archive(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Verifies a gzipped tar archive is read (compression inferred) and yields a frame
    # with a single column named a.
    parser = all_parsers
    path = os.path.join(csv_dir_path, "tar_csv.tar.gz")
    df = parser.read_csv(path)
    assert list(df.columns) == ["a"]


def test_ignore_compression_extension(tmp_path, all_parsers):
    # [FUNCTION DOC]
    # Verifies that a plain-text file with a zip extension is read correctly when
    # compression is explicitly disabled.
    parser = all_parsers
    df = DataFrame({"a": [0, 1]})

    path_csv = tmp_path / "test.csv"
    path_zip = tmp_path / "test.csv.zip"
    # make sure to create un-compressed file with zip extension
    df.to_csv(path_csv, index=False)
    Path(path_zip).write_text(
        Path(path_csv).read_text(encoding="utf-8"), encoding="utf-8"
    )

    tm.assert_frame_equal(parser.read_csv(path_zip, compression=None), df)


def test_writes_tar_gz(tmp_path, all_parsers):
    # [FUNCTION DOC]
    # Verifies writing a frame to a path ending in tar.gz and reading it back infers
    # gzip and returns the same frame, and that the file really is a gzipped tar archive
    # by extracting its first member and reading it as a file object.
    parser = all_parsers
    data = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )
    tar_path = tmp_path / "test.tar.gz"
    data.to_csv(tar_path, index=False)

    # test that read_csv infers .tar.gz to gzip:
    tm.assert_frame_equal(parser.read_csv(tar_path), data)

    # test that file is indeed gzipped:
    with tarfile.open(tar_path, "r:gz") as tar:
        result = parser.read_csv(
            tar.extractfile(tar.getnames()[0]), compression="infer"
        )
        tm.assert_frame_equal(result, data)
