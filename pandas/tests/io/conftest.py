# [MODULE DOC]
# Shared fixtures for all input/output tests: lookups of bundled data files, mocked cloud storage (a local object-store server, buckets with and without data and the storage options pointing to it), and compression parameter fixtures.
# Fixtures are available to every test module under the io test package. Import-time side effect: builds a module-level list of (file extension, compression name) pairs, with two entries for zstandard marked to be skipped when the zstandard library is missing.

import uuid

import pytest

from pandas.compat import (
    is_ci_environment,
    is_platform_arm,
    is_platform_mac,
    is_platform_windows,
)
import pandas.util._test_decorators as td

import pandas.io.common as icom
from pandas.io.parsers import read_csv


@pytest.fixture
def compression_to_extension():
    # [FUNCTION DOC]
    # Fixture returning the reverse of the library's mapping from file extension to compression name: a mapping from compression name to extension.
    return {value: key for key, value in icom.extension_to_compression.items()}


@pytest.fixture
def tips_file(datapath):
    # [FUNCTION DOC]
    # Fixture returning the path of the restaurant tips csv file in the test data directory (built through the data path fixture).
    """Path to the tips dataset"""
    return datapath("io", "data", "csv", "tips.csv")


@pytest.fixture
def jsonl_file(datapath):
    # [FUNCTION DOC]
    # Fixture returning the path of the line-delimited json items file in the parser test data directory.
    """Path to a JSONL dataset"""
    return datapath("io", "parser", "data", "items.jsonl")


@pytest.fixture
def salaries_table(datapath):
    # [FUNCTION DOC]
    # Fixture returning a frame read from the tab-separated salaries csv file in the parser test data directory.
    """DataFrame with the salaries dataset"""
    return read_csv(datapath("io", "parser", "data", "salaries.csv"), sep="\t")


@pytest.fixture
def feather_file(datapath):
    # [FUNCTION DOC]
    # Fixture returning the path of the simple feather dataset in the test data directory.
    return datapath("io", "data", "feather", "simple_dataset.feather")


@pytest.fixture
def xml_file(datapath):
    # [FUNCTION DOC]
    # Fixture returning the path of the books xml file in the test data directory.
    return datapath("io", "data", "xml", "books.xml")


@pytest.fixture(scope="session")
def aws_credentials(monkeysession):
    # [FUNCTION DOC]
    # Session-scoped fixture that sets mocked cloud credentials and a default region in the environment using the session-scoped monkeypatch fixture. Needed for the mock object-store server; values are the fixed text "testing" and the region us-east-1. Environment variables are restored when the session ends.
    # KNOWN ISSUE: one of the variables set is named with a doubled AWS_SESSION prefix, which looks like a typo for the standard session token variable, so the standard session token variable is never set (the security token variable is set instead).
    """Mocked AWS Credentials for moto."""
    monkeysession.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeysession.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeysession.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeysession.setenv("AWS_SESSION_AWS_SESSION_TOKEN", "testing")
    monkeysession.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(scope="session")
def moto_server(aws_credentials):
    # [FUNCTION DOC]
    # Session-scoped fixture that provides the URL of a mock S3 server and depends on the credentials fixture.
    # On Linux CI that is not mac, arm or Windows it yields a fixed local address that is expected to be served by a service container. Otherwise it skips the test when the mock server library is missing, starts a threaded mock server on an automatically chosen port, yields its address as a url built from the host and port, and stops the server on teardown.
    # use service container for Linux on GitHub Actions
    if is_ci_environment() and not (
        is_platform_mac() or is_platform_arm() or is_platform_windows()
    ):
        yield "http://localhost:5000"
    else:
        moto_server = pytest.importorskip("moto.server")
        server = moto_server.ThreadedMotoServer(port=0)
        server.start()
        host, port = server.get_host_and_port()
        yield f"http://{host}:{port}"
        server.stop()


@pytest.fixture
def moto_s3_resource(moto_server):
    # [FUNCTION DOC]
    # Fixture returning an S3 resource object of the boto3 library connected to the mock server URL; skips when boto3 is not installed.
    boto3 = pytest.importorskip("boto3")
    s3 = boto3.resource("s3", endpoint_url=moto_server)
    return s3


@pytest.fixture(scope="session")
def s3so(moto_server):
    # [FUNCTION DOC]
    # Session-scoped fixture returning storage options (a mapping with client keyword arguments holding the endpoint url of the mock server) for passing to filesystem-aware readers and writers.
    return {
        "client_kwargs": {
            "endpoint_url": moto_server,
        }
    }


@pytest.fixture
def s3_bucket_public(moto_s3_resource):
    # [FUNCTION DOC]
    # Fixture that creates a uniquely named bucket (prefixed with a fixed name and a random identifier) with public read access on the mock server, yields it, and on teardown deletes all objects in it and then the bucket.
    """
    Create a public S3 bucket using moto.
    """
    bucket_name = f"pandas-test-{uuid.uuid4()}"
    bucket = moto_s3_resource.Bucket(bucket_name)
    bucket.create(ACL="public-read")
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_bucket_private(moto_s3_resource):
    # [FUNCTION DOC]
    # Same as the public bucket fixture but the bucket has a private access control list and a different name prefix.
    """
    Create a private S3 bucket using moto.
    """
    bucket_name = f"cant_get_it-{uuid.uuid4()}"
    bucket = moto_s3_resource.Bucket(bucket_name)
    bucket.create(ACL="private")
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_bucket_public_with_data(
    s3_bucket_public, tips_file, jsonl_file, feather_file, xml_file
):
    # [FUNCTION DOC]
    # Fixture that fills the public bucket with seven test files and returns the bucket: the tips csv under two keys (one containing a hash character), its gzip and bzip2 compressed siblings (found by appending the extension to the path), the json lines file, the feather file and the books xml file. Files are uploaded as binary.
    # KNOWN ISSUE: the docstring lists only four datasets (and calls the compressed ones csv-named), but the fixture uploads seven objects including the feather and xml files and a second tips key.
    """
    The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl
    """
    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
        ("books.xml", xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, "rb") as f:
            s3_bucket_public.put_object(Key=s3_key, Body=f)
    return s3_bucket_public


@pytest.fixture
def s3_bucket_private_with_data(
    s3_bucket_private, tips_file, jsonl_file, feather_file, xml_file
):
    # [FUNCTION DOC]
    # Same as the public-data fixture but the files are uploaded into the private bucket and the private bucket is returned.
    # KNOWN ISSUE: the docstring lists only four datasets but seven objects are uploaded.
    """
    The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl
    """
    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
        ("books.xml", xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, "rb") as f:
            s3_bucket_private.put_object(Key=s3_key, Body=f)
    return s3_bucket_private


_compression_formats_params = [
    (".no_compress", None),
    ("", None),
    (".gz", "gzip"),
    (".GZ", "gzip"),
    (".bz2", "bz2"),
    (".BZ2", "bz2"),
    (".zip", "zip"),
    (".ZIP", "zip"),
    (".xz", "xz"),
    (".XZ", "xz"),
    pytest.param((".zst", "zstd"), marks=td.skip_if_no("zstandard")),
    pytest.param((".ZST", "zstd"), marks=td.skip_if_no("zstandard")),
]


@pytest.fixture(params=_compression_formats_params[1:])
def compression_format(request):
    # [FUNCTION DOC]
    # Fixture parametrized over every entry of the module-level compression list except the first (the one with the made-up extension), returning the (extension, compression name) pair, so the no-compression entry with empty extension is included.
    return request.param


@pytest.fixture(params=_compression_formats_params)
def compression_ext(request):
    # [FUNCTION DOC]
    # Fixture parametrized over every entry of the module-level compression list; returns only the extension part.
    return request.param[0]
