# [MODULE DOC]
# Tests of fsspec-backed remote/in-memory file system support in the I/O functions:
# reading and writing via URLs for memory and a custom test protocol, passing storage
# options through, and error handling for missing protocols. S3 tests use mock-server
# fixtures. A module-wide mark ignores a block-manager deprecation warning.

import io

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import HAS_PYARROW
from pandas.compat.pyarrow import pa_version_under14p0

from pandas import (
    DataFrame,
    date_range,
    read_csv,
    read_excel,
    read_feather,
    read_json,
    read_parquet,
    read_pickle,
    read_stata,
    read_table,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td
from pandas.util.version import Version

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def fsspectest():
    # [FUNCTION DOC]
    # HELPER: fixture registering a custom in-memory file system.
    # Skips if fsspec is missing. Defines a memory file system subclass with protocol
    # name testmem that records the test storage option into a class-level one-element
    # list when constructed, registers it (replacing any existing registration), and
    # yields an instance. On teardown removes the registry entry, resets the recorded
    # option and clears the shared store.
    pytest.importorskip("fsspec")
    from fsspec import register_implementation
    from fsspec.implementations.memory import MemoryFileSystem
    from fsspec.registry import _registry as registry

    class TestMemoryFS(MemoryFileSystem):
        # [CLASS DOC]
        # HELPER: in-memory file system subclass used by the fsspectest fixture. Class
        # attributes: protocol testmem and a one-element list recording the last test
        # storage option seen.
        protocol = "testmem"
        test = [None]

        def __init__(self, **kwargs) -> None:
            # [FUNCTION DOC]
            # HELPER: constructor of the test file system: pops the test keyword
            # argument (default none) into the class-level record, then delegates the
            # rest to the parent constructor.
            self.test[0] = kwargs.pop("test", None)
            super().__init__(**kwargs)

    register_implementation("testmem", TestMemoryFS, clobber=True)
    yield TestMemoryFS()
    registry.pop("testmem", None)
    TestMemoryFS.test[0] = None
    TestMemoryFS.store.clear()


@pytest.fixture
def df1():
    # [FUNCTION DOC]
    # HELPER: fixture returning a small frame with integer, float (with NaN), string and
    # two-day datetime columns.
    return DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )


@pytest.fixture
def cleared_fs():
    # [FUNCTION DOC]
    # HELPER: fixture that skips without fsspec, yields the shared memory file system
    # and clears its store afterwards.
    fsspec = pytest.importorskip("fsspec")

    memfs = fsspec.filesystem("memory")
    yield memfs
    memfs.store.clear()


def test_read_csv(cleared_fs, df1):
    # [FUNCTION DOC]
    # Writes CSV bytes of the sample frame to the memory file system and reads it by URL
    # with the datetime column parsed; expected equals the frame with the datetime
    # column in microsecond resolution.
    text = str(df1.to_csv(index=False)).encode()
    with cleared_fs.open("test/test.csv", "wb") as w:
        w.write(text)
    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"])

    expected = df1.copy()
    expected["dt"] = expected["dt"].astype("M8[us]")
    tm.assert_frame_equal(df2, expected)


def test_reasonable_error(monkeypatch, cleared_fs):
    # [FUNCTION DOC]
    # Reading from an unknown protocol must raise a value error naming the protocol. For
    # a protocol known to fsspec but whose implementation cannot be imported (registered
    # via monkeypatch with a custom error text), reading must raise an import error
    # carrying that text.
    from fsspec.registry import known_implementations

    with pytest.raises(ValueError, match="nosuchprotocol"):
        read_csv("nosuchprotocol://test/test.csv")
    err_msg = "test error message"
    monkeypatch.setitem(
        known_implementations,
        "couldexist",
        {"class": "unimportable.CouldExist", "err": err_msg},
    )
    with pytest.raises(ImportError, match=err_msg):
        read_csv("couldexist://test/test.csv")


def test_to_csv(cleared_fs, df1):
    # [FUNCTION DOC]
    # Writes the frame as CSV to a memory URL with index, reads back with parsed dates
    # and first column as index; equals the frame with microsecond datetimes.
    df1.to_csv("memory://test/test.csv", index=True)

    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"], index_col=0)

    expected = df1.copy()
    expected["dt"] = expected["dt"].astype("M8[us]")
    tm.assert_frame_equal(df2, expected)


def test_to_excel(cleared_fs, df1):
    # [FUNCTION DOC]
    # Requires openpyxl. Writes an xlsx to a memory URL and reads back with parsed dates
    # and index; equals the frame with microsecond datetimes.
    pytest.importorskip("openpyxl")
    ext = "xlsx"
    path = f"memory://test/test.{ext}"
    df1.to_excel(path, index=True)

    df2 = read_excel(path, parse_dates=["dt"], index_col=0)

    expected = df1.copy()
    expected["dt"] = expected["dt"].astype("M8[us]")
    tm.assert_frame_equal(df2, expected)


@pytest.mark.parametrize("binary_mode", [False, True])
def test_to_csv_fsspec_object(cleared_fs, binary_mode, df1):
    # [FUNCTION DOC]
    # Parametrized over binary and text mode. Writes CSV into an already-open fsspec
    # file object (must remain open), then reads from an fsspec object opened for
    # reading (must remain open); the result equals the frame with microsecond
    # datetimes.
    fsspec = pytest.importorskip("fsspec")

    path = "memory://test/test.csv"
    mode = "wb" if binary_mode else "w"
    with fsspec.open(path, mode=mode).open() as fsspec_object:
        df1.to_csv(fsspec_object, index=True)
        assert not fsspec_object.closed

    mode = mode.replace("w", "r")
    with fsspec.open(path, mode=mode) as fsspec_object:
        df2 = read_csv(
            fsspec_object,
            parse_dates=["dt"],
            index_col=0,
        )
        assert not fsspec_object.closed

    expected = df1.copy()
    expected["dt"] = expected["dt"].astype("M8[us]")
    tm.assert_frame_equal(df2, expected)


def test_csv_options(fsspectest):
    # [FUNCTION DOC]
    # Storage options are passed to the file system: after writing a CSV to the custom
    # protocol, the recorded option equals the write value; after reading, it equals the
    # read value.
    df = DataFrame({"a": [0]})
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    assert fsspectest.test[0] == "csv_write"
    read_csv("testmem://test/test.csv", storage_options={"test": "csv_read"})
    assert fsspectest.test[0] == "csv_read"


def test_read_table_options(fsspectest):
    # [FUNCTION DOC]
    # GH 39167. Same as the CSV options test but for read_table.
    # GH #39167
    df = DataFrame({"a": [0]})
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    assert fsspectest.test[0] == "csv_write"
    read_table("testmem://test/test.csv", storage_options={"test": "csv_read"})
    assert fsspectest.test[0] == "csv_read"


def test_excel_options(fsspectest):
    # [FUNCTION DOC]
    # Requires openpyxl. Storage options are forwarded when writing and reading an Excel
    # file on the custom protocol.
    pytest.importorskip("openpyxl")
    extension = "xlsx"

    df = DataFrame({"a": [0]})

    path = f"testmem://test/test.{extension}"

    df.to_excel(path, storage_options={"test": "write"}, index=False)
    assert fsspectest.test[0] == "write"
    read_excel(path, storage_options={"test": "read"})
    assert fsspectest.test[0] == "read"


def test_to_parquet_new_file(cleared_fs, df1, request):
    # [FUNCTION DOC]
    # Requires fastparquet. Expected to fail when the infer-string mode, pyarrow of at
    # least version 14 and a fastparquet version up to 2025.12.0 coincide. Writes a
    # frame to a memory URL with the fastparquet engine and no compression; only checks
    # that no error occurs.
    # KNOWN ISSUE: the docstring mentions writing to a not-yet-existent GCS Parquet
    # file, but the test writes to the in-memory file system with a csv-style name.
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    fp = pytest.importorskip("fastparquet")

    request.applymarker(
        pytest.mark.xfail(
            using_string_dtype()
            and HAS_PYARROW
            and not pa_version_under14p0
            and Version(fp.__version__) <= Version("2025.12.0"),
            reason="TODO(infer_string) fastparquet",
        )
    )

    df1.to_parquet(
        "memory://test/test.csv", index=True, engine="fastparquet", compression=None
    )


def test_arrowparquet_options(fsspectest):
    # [FUNCTION DOC]
    # Requires pyarrow. Storage options are forwarded on parquet write and read with the
    # pyarrow engine on the custom protocol.
    # KNOWN ISSUE: the docstring mentions GCS regression, but the test uses the custom
    # in-memory test protocol and tests options passing.
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [0]})
    df.to_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    assert fsspectest.test[0] == "parquet_write"
    read_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        storage_options={"test": "parquet_read"},
    )
    assert fsspectest.test[0] == "parquet_read"


def test_fastparquet_options(fsspectest):
    # [FUNCTION DOC]
    # Requires fastparquet. Same as the pyarrow variant but with the fastparquet engine.
    # KNOWN ISSUE: the docstring mentions a GCS regression, but the test uses the custom
    # in-memory test protocol and tests options passing.
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("fastparquet")

    df = DataFrame({"a": [0]})
    df.to_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    assert fsspectest.test[0] == "parquet_write"
    read_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        storage_options={"test": "parquet_read"},
    )
    assert fsspectest.test[0] == "parquet_read"


@pytest.mark.single_cpu
@pytest.mark.parametrize("compression_suffix", ["", ".gz", ".bz2"])
def test_from_s3_csv(s3_bucket_public_with_data, s3so, tips_file, compression_suffix):
    # [FUNCTION DOC]
    # Single-cpu. Parametrized over no, gz, and bz2 suffixes. Requires s3fs. Reads
    # tips.csv (possibly compressed) from the mocked public S3 bucket with the s3
    # storage options; the result equals reading the local file.
    pytest.importorskip("s3fs")
    df_from_s3 = read_csv(
        f"s3://{s3_bucket_public_with_data.name}/tips.csv{compression_suffix}",
        storage_options=s3so,
    )
    df_from_local = read_csv(tips_file)
    tm.assert_equal(df_from_s3, df_from_local)


@pytest.mark.single_cpu
@pytest.mark.parametrize("protocol", ["s3", "s3a", "s3n"])
def test_s3_protocols(s3_bucket_public_with_data, s3so, tips_file, protocol):
    # [FUNCTION DOC]
    # Single-cpu. Parametrized over s3, s3a and s3n URL schemes. Requires s3fs. Reading
    # tips.csv from the mock bucket under each scheme equals the local read.
    pytest.importorskip("s3fs")
    df_from_s3 = read_csv(
        f"{protocol}://{s3_bucket_public_with_data.name}/tips.csv",
        storage_options=s3so,
    )
    df_from_local = read_csv(tips_file)
    tm.assert_equal(df_from_s3, df_from_local)


@pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string) fastparquet")
@pytest.mark.single_cpu
def test_s3_parquet(s3_bucket_public, s3so, df1):
    # [FUNCTION DOC]
    # Single-cpu. Expected to fail when the infer-string mode is on (fastparquet).
    # Requires fastparquet and s3fs. Writes the frame to the mock S3 bucket as parquet
    # without index or compression and reads it back; must be equal.
    pytest.importorskip("fastparquet")
    pytest.importorskip("s3fs")

    fn = f"s3://{s3_bucket_public.name}/test.parquet"
    df1.to_parquet(
        fn, index=False, engine="fastparquet", compression=None, storage_options=s3so
    )
    df2 = read_parquet(fn, engine="fastparquet", storage_options=s3so)
    tm.assert_equal(df1, df2)


@td.skip_if_installed("fsspec")
def test_not_present_exception():
    # [FUNCTION DOC]
    # Runs only when fsspec is not installed. Reading a memory URL must raise an import
    # error whose message says that importing fsspec failed and to install the package.
    msg = "`Import fsspec` failed.  Use pip or conda to install the fsspec package."
    with pytest.raises(ImportError, match=msg):
        read_csv("memory://test/test.csv")


def test_feather_options(fsspectest):
    # [FUNCTION DOC]
    # Requires pyarrow. Storage options are forwarded when writing and reading Feather
    # on the custom protocol; the round trip is equal.
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [0]})
    df.to_feather("testmem://mockfile", storage_options={"test": "feather_write"})
    assert fsspectest.test[0] == "feather_write"
    out = read_feather("testmem://mockfile", storage_options={"test": "feather_read"})
    assert fsspectest.test[0] == "feather_read"
    tm.assert_frame_equal(df, out)


def test_pickle_options(fsspectest):
    # [FUNCTION DOC]
    # Storage options are forwarded when writing and reading a pickle on the custom
    # protocol; round trip is equal.
    df = DataFrame({"a": [0]})
    df.to_pickle("testmem://mockfile", storage_options={"test": "pickle_write"})
    assert fsspectest.test[0] == "pickle_write"
    out = read_pickle("testmem://mockfile", storage_options={"test": "pickle_read"})
    assert fsspectest.test[0] == "pickle_read"
    tm.assert_frame_equal(df, out)


def test_json_options(fsspectest, compression):
    # [FUNCTION DOC]
    # Parametrized by compression fixture. Storage options are forwarded when writing
    # and reading JSON on the custom protocol; round trip is equal.
    df = DataFrame({"a": [0]})
    df.to_json(
        "testmem://mockfile",
        compression=compression,
        storage_options={"test": "json_write"},
    )
    assert fsspectest.test[0] == "json_write"
    out = read_json(
        "testmem://mockfile",
        compression=compression,
        storage_options={"test": "json_read"},
    )
    assert fsspectest.test[0] == "json_read"
    tm.assert_frame_equal(df, out)


def test_stata_options(fsspectest):
    # [FUNCTION DOC]
    # Storage options are forwarded when writing (without index) and reading Stata on
    # the custom protocol; the result cast to int64 equals the frame.
    df = DataFrame({"a": [0]})
    df.to_stata(
        "testmem://mockfile", storage_options={"test": "stata_write"}, write_index=False
    )
    assert fsspectest.test[0] == "stata_write"
    out = read_stata("testmem://mockfile", storage_options={"test": "stata_read"})
    assert fsspectest.test[0] == "stata_read"
    tm.assert_frame_equal(df, out.astype("int64"))


def test_markdown_options(fsspectest):
    # [FUNCTION DOC]
    # Requires tabulate. Writing markdown to the custom protocol forwards the storage
    # options and the file content is non-empty.
    pytest.importorskip("tabulate")
    df = DataFrame({"a": [0]})
    df.to_markdown("testmem://mockfile", storage_options={"test": "md_write"})
    assert fsspectest.test[0] == "md_write"
    assert fsspectest.cat("testmem://mockfile")


def test_non_fsspec_options():
    # [FUNCTION DOC]
    # Requires pyarrow. Passing storage options with a local path or an in-memory bytes
    # buffer to the CSV reader, a local path to the parquet reader, or a local path to
    # the parquet writer, must raise a value error mentioning storage_options.
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="storage_options"):
        read_csv("localfile", storage_options={"a": True})
    with pytest.raises(ValueError, match="storage_options"):
        # separate test for parquet, which has a different code path
        read_parquet("localfile", storage_options={"a": True})
    by = io.BytesIO()

    with pytest.raises(ValueError, match="storage_options"):
        read_csv(by, storage_options={"a": True})

    df = DataFrame({"a": [0]})
    with pytest.raises(ValueError, match="storage_options"):
        df.to_parquet("nonfsspecpath", storage_options={"a": True})
