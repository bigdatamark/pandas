# [MODULE DOC]
# Tests for reading from S3 related sources with CSV reader: streaming body objects from
# botocore and public bucket reads with and without credentials (the latter using mock
# S3 fixtures).

from io import BytesIO

import pytest

from pandas import read_csv


@pytest.mark.parametrize("data", [b"foo,bar,baz\n1,2,3\n4,5,6\n", b"just,the,header\n"])
def test_streaming_s3_objects(data):
    # [FUNCTION DOC]
    # GH 17135. Parametrized over two small CSV byte strings (with rows, and header
    # only). Requires botocore 1.10.47 or newer (which added iteration support). Wraps
    # the bytes in a botocore streaming body and passes it to the CSV reader; passes if
    # no error occurs (no assertion on content).
    # GH 17135
    # botocore gained iteration support in 1.10.47, can now be used in read_*
    pytest.importorskip("botocore", minversion="1.10.47")
    from botocore.response import StreamingBody

    body = StreamingBody(BytesIO(data), content_length=len(data))
    read_csv(body)


@pytest.mark.single_cpu
@pytest.mark.parametrize("header", ["infer", None])
def test_read_with_and_without_creds_from_pub_bucket(
    s3_bucket_public_with_data, s3so, header
):
    # [FUNCTION DOC]
    # GH 34626. Single-cpu, parametrized by header inferred or none. Requires s3fs.
    # Reads the first 5 rows of tips.csv from the mock public bucket using the s3
    # storage options and asserts exactly 5 rows were read.
    # GH 34626
    pytest.importorskip("s3fs")
    nrows = 5
    df = read_csv(
        f"s3://{s3_bucket_public_with_data.name}/tips.csv",
        nrows=nrows,
        header=header,
        storage_options=s3so,
    )
    assert len(df) == nrows
