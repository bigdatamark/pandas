# [MODULE DOC]
# Tests for compression support in read_json and to_json: round trips for each
# compression method, zip reading of a stored data file, line-delimited and
# chunked reading with compression, S3 reading, inference from the file
# extension, and errors for unknown compression names. The compression fixtures
# come from the shared io conftest.

from io import (
    BytesIO,
    StringIO,
)
import uuid

import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


def test_compression_roundtrip(compression, temp_file):
    # [FUNCTION DOC]
    # Writes a small float frame with row and column labels to a path using the
    # parametrized compression, reads it back with the same compression and checks
    # equality. Then decompresses the file by hand, decodes it as text and reads it
    # as plain JSON, which must again equal the original, proving the file really
    # was compressed.
    df = pd.DataFrame(
        [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        index=["A", "B"],
        columns=["X", "Y", "Z"],
    )

    df.to_json(temp_file, compression=compression)
    tm.assert_frame_equal(df, pd.read_json(temp_file, compression=compression))

    # explicitly ensure file was compressed.
    with tm.decompress_file(temp_file, compression) as fh:
        result = fh.read().decode("utf8")
        data = StringIO(result)
    tm.assert_frame_equal(df, pd.read_json(data))


def test_read_zipped_json(datapath):
    # [FUNCTION DOC]
    # Reads a stored JSON data file uncompressed and its zipped counterpart with
    # zip compression; both frames must be equal.
    uncompressed_path = datapath("io", "json", "data", "tsframe_v012.json")
    uncompressed_df = pd.read_json(uncompressed_path)

    compressed_path = datapath("io", "json", "data", "tsframe_v012.json.zip")
    compressed_df = pd.read_json(compressed_path, compression="zip")

    tm.assert_frame_equal(uncompressed_df, compressed_df)


@td.skip_if_not_us_locale
@pytest.mark.single_cpu
@pytest.mark.network
def test_with_s3_url(compression, s3_bucket_public, s3so, temp_file):
    # [FUNCTION DOC]
    # Network, single-CPU test, skipped outside a US locale. Writes a small frame
    # to a local compressed file, uploads it to the public mock S3 bucket under a
    # random key, reads it back via an S3 URL with the parametrized compression and
    # storage options, and checks the result equals the original.
    # Bucket created in tests/io/conftest.py
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))

    key = f"{uuid.uuid4()}.json"
    df.to_json(temp_file, compression=compression)
    with open(temp_file, "rb") as f:
        s3_bucket_public.put_object(Key=key, Body=f)

    roundtripped_df = pd.read_json(
        f"s3://{s3_bucket_public.name}/{key}",
        compression=compression,
        storage_options=s3so,
    )
    tm.assert_frame_equal(df, roundtripped_df)


def test_lines_with_compression(compression, temp_file):
    # [FUNCTION DOC]
    # Writes a frame as line-delimited records with the parametrized compression
    # and reads it back with lines enabled and the same compression; the result must
    # equal the original.
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
    df.to_json(temp_file, orient="records", lines=True, compression=compression)
    roundtripped_df = pd.read_json(temp_file, lines=True, compression=compression)
    tm.assert_frame_equal(df, roundtripped_df)


def test_chunksize_with_compression(compression, temp_file):
    # [FUNCTION DOC]
    # Writes a frame (with a string column) as compressed line-delimited records,
    # reads it in chunks of one row using the reader as a context manager,
    # concatenates the chunks and checks equality with the original.
    df = pd.read_json(StringIO('{"a": ["foo", "bar", "baz"], "b": [4, 5, 6]}'))
    df.to_json(temp_file, orient="records", lines=True, compression=compression)

    with pd.read_json(
        temp_file, lines=True, chunksize=1, compression=compression
    ) as res:
        roundtripped_df = pd.concat(res)
    tm.assert_frame_equal(df, roundtripped_df)


def test_write_unsupported_compression_type(temp_file):
    # [FUNCTION DOC]
    # Writing with an unrecognized compression name must raise a value error whose
    # message says the compression type is unrecognized.
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
    msg = "Unrecognized compression type: unsupported"
    with pytest.raises(ValueError, match=msg):
        df.to_json(temp_file, compression="unsupported")


def test_read_unsupported_compression_type(temp_file):
    # [FUNCTION DOC]
    # Reading with an unrecognized compression name must raise a value error whose
    # message says the compression type is unrecognized.
    msg = "Unrecognized compression type: unsupported"
    with pytest.raises(ValueError, match=msg):
        pd.read_json(temp_file, compression="unsupported")


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("to_infer", [True, False])
@pytest.mark.parametrize("read_infer", [True, False])
def test_to_json_compression(
    compression_only,
    read_infer,
    to_infer,
    compression_to_extension,
    infer_string,
    tmp_path,
):
    # [FUNCTION DOC]
    # Parametrized over string inference on/off, inferring versus explicit
    # compression on write, and on read (gh-15008). Using a compression-only
    # fixture, builds a file name with the matching extension, writes a one-column
    # frame and reads it back; the frames must be equal. The whole body runs under
    # the chosen future string-inference option.
    with pd.option_context("future.infer_string", infer_string):
        # see gh-15008
        compression = compression_only

        # We'll complete file extension subsequently.
        filename = tmp_path / f"test.{compression_to_extension[compression]}"

        df = pd.DataFrame({"A": [1]})

        to_compression = "infer" if to_infer else compression
        read_compression = "infer" if read_infer else compression

        df.to_json(filename, compression=to_compression)
        result = pd.read_json(filename, compression=read_compression)
        tm.assert_frame_equal(result, df)


def test_to_json_compression_mode(compression):
    # [FUNCTION DOC]
    # Regression for GH 39985: writing a frame to an in-memory binary buffer with
    # each compression must be accepted. Reading back is intentionally not done
    # because read_json does not support user-provided binary files; the commented
    # out lines record that.
    # GH 39985 (read_json does not support user-provided binary files)
    expected = pd.DataFrame({"A": [1]})

    with BytesIO() as buffer:
        expected.to_json(buffer, compression=compression)
        # df = pd.read_json(buffer, compression=compression)
        # tm.assert_frame_equal(expected, df)
