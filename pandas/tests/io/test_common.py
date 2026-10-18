"""
Tests for the pandas.io.common functionalities
"""

# [MODULE DOC]
# Tests for the shared I/O utilities module (path handling, handle creation, compression
# inference, encoding handling, memory mapping, fsspec URL detection) plus cross-reader
# checks that readers and writers accept path-like objects and give consistent errors. A
# module-wide mark ignores a deprecation warning about passing a block manager.

import codecs
import errno
from functools import partial
from io import (
    BytesIO,
    StringIO,
    UnsupportedOperation,
)
import mmap
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pytest

from pandas.compat import (
    WASM,
    is_platform_windows,
)
from pandas.compat.pyarrow import pa_version_under19p0
import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


class CustomFSPath:
    # [CLASS DOC]
    # HELPER: test double for an object that is path-like but not a built-in path type.
    # Stores a path value and exposes it through the file-system-path protocol method;
    # used to check that readers and writers accept arbitrary path-like objects.
    """For testing fspath on unknown objects"""

    def __init__(self, path) -> None:
        # [FUNCTION DOC]
        # HELPER: constructor of the CustomFSPath test double. Takes the path value and
        # stores it on the instance. Returns nothing.
        self.path = path

    def __fspath__(self):
        # [FUNCTION DOC]
        # HELPER: implements the file-system-path protocol for the test double; returns
        # the stored path unchanged.
        return self.path


HERE = os.path.abspath(os.path.dirname(__file__))


# https://github.com/cython/cython/issues/1720
class TestCommonIOCapabilities:
    # [CLASS DOC]
    # Tests of core I/O helpers: path expansion and stringification, compression
    # inference, handle creation, iterator reading, and consistent errors/path-like
    # support across all readers and writers. Holds a class-level sample CSV text with
    # an index column and four numeric columns.
    data1 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    def test_expand_user(self):
        # [FUNCTION DOC]
        # A home-relative path must be expanded by the private expand-user helper into
        # an absolute path different from the input and equal to the standard library
        # expansion.
        filename = "~/sometest"
        expanded_name = icom._expand_user(filename)

        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self):
        # [FUNCTION DOC]
        # A path without a home marker is returned unchanged by the expand-user helper.
        filename = "/somefolder/sometest"
        expanded_name = icom._expand_user(filename)

        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self):
        # [FUNCTION DOC]
        # Stringifying a path object turns the current-directory path into a dot, and
        # normalizes a path with a doubled separator to the platform-joined form.
        rel_path = icom.stringify_path(Path("."))
        assert rel_path == "."
        redundant_path = icom.stringify_path(Path("foo//bar"))
        assert redundant_path == os.path.join("foo", "bar")

    def test_stringify_path_fspath(self):
        # [FUNCTION DOC]
        # A custom path-like object is converted to its path string.
        p = CustomFSPath("foo/bar.csv")
        result = icom.stringify_path(p)
        assert result == "foo/bar.csv"

    def test_stringify_file_and_path_like(self, temp_file):
        # [FUNCTION DOC]
        # GH 38125. Needs fsspec. An opened fsspec file object (which is both file-like
        # and path-like) must be returned as is by the path stringifier, not converted
        # to a string.
        # GH 38125: do not stringify file objects that are also path-like
        fsspec = pytest.importorskip("fsspec")
        with fsspec.open(f"file://{temp_file}", mode="wb") as fsspec_obj:
            assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_infer_compression_from_path(self, compression_format, path_type):
        # [FUNCTION DOC]
        # Parametrized over path-like types (str, custom path-like, Path) and the
        # compression_format fixture (extension, expected compression). Inferring
        # compression with the infer setting from a path built with that extension must
        # give the expected compression.
        extension, expected = compression_format
        path = path_type("foo/bar.csv" + extension)
        compression = icom.infer_compression(path, compression="infer")
        assert compression == expected

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type):
        # [FUNCTION DOC]
        # Parametrized over the three path types. In a temporary directory inside the
        # home directory, opens a home-relative path for writing; the handle name must
        # be absolute and equal to the expanded input.
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type("~/" + Path(tmp).name + "/sometest")
            with icom.get_handle(filename, "w") as handles:
                assert Path(handles.handle.name).is_absolute()
                assert os.path.expanduser(filename) == handles.handle.name

    def test_get_handle_with_buffer(self):
        # [FUNCTION DOC]
        # Passing an in-memory text buffer to get_handle must return that same buffer as
        # the handle; the buffer stays open after the handle context exits only until
        # the buffer's own context ends, after which it is closed.
        with StringIO() as input_buffer:
            with icom.get_handle(input_buffer, "r") as handles:
                assert handles.handle == input_buffer
            assert not input_buffer.closed
        assert input_buffer.closed

    # Test that BytesIOWrapper(get_handle) returns correct amount of bytes every time
    def test_bytesiowrapper_returns_correct_bytes(self):
        # [FUNCTION DOC]
        # Opens a text buffer in binary mode with text flag off, so it is wrapped to
        # present bytes. Reads in chunks of 5 bytes, asserting no chunk exceeds the
        # size, a short chunk happens only at end of data, and the concatenation equals
        # the utf-8 encoding of the data (including 2-, 3- and 4-byte characters).
        # Test latin1, ucs-2, and ucs-4 chars
        data = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
        with icom.get_handle(StringIO(data), "rb", is_text=False) as handles:
            result = b""
            chunksize = 5
            while True:
                chunk = handles.handle.read(chunksize)
                # Make sure each chunk is correct amount of bytes
                assert len(chunk) <= chunksize
                if len(chunk) < chunksize:
                    # Can be less amount of bytes, but only at EOF
                    # which happens when read returns empty
                    assert len(handles.handle.read()) == 0
                    result += chunk
                    break
                result += chunk
            assert result == data.encode("utf-8")

    # Test that pyarrow can handle a file opened with get_handle
    def test_get_handle_pyarrow_compat(sel, using_infer_string):
        # [FUNCTION DOC]
        # Needs pyarrow.
        # KNOWN ISSUE: the first parameter is named sel rather than self; it still
        # receives the instance, so the test works, but the name is a typo.
        # Reads a text buffer wrapped by get_handle in binary mode through the pyarrow
        # CSV reader and converts to a frame; expected values are cast to object dtype
        # for old pyarrow versions, or the NaN-variant string dtype when the infer-
        # string option is off. Asserts frame equality and that the source buffer was
        # not closed.
        pa_csv = pytest.importorskip("pyarrow.csv")

        # Test latin1, ucs-2, and ucs-4 chars
        data = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
        expected = pd.DataFrame(
            {"a": ["1", "©", "Look"], "b": ["2", "®", "a snake"], "c": ["3", "®", "🐍"]}
        )
        s = StringIO(data)
        with icom.get_handle(s, "rb", is_text=False) as handles:
            df = pa_csv.read_csv(handles.handle).to_pandas()
            if pa_version_under19p0:
                expected = expected.astype("object")
            elif not using_infer_string:
                expected = expected.astype(pd.StringDtype(na_value=np.nan))
            tm.assert_frame_equal(df, expected)
            assert not s.closed

    def test_iterator(self):
        # [FUNCTION DOC]
        # Reading the sample CSV in chunks of one row and concatenating with a fresh
        # index equals reading it whole. GH 12153: after taking the first chunk, it
        # equals the first row and concatenating the rest of the iterator equals the
        # remaining rows.
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result = pd.concat(reader, ignore_index=True)
        expected = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)

        # GH12153
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_non_existent(self, reader, module, error_class, fn_ext):
        # [FUNCTION DOC]
        # Skipped on WASM. Parametrized over nine readers (csv, fwf, excel, feather,
        # hdf, stata, sas, json, pickle) with the module that must be importable, the
        # expected exception class and a file extension. Reading a nonexistent path must
        # raise that exception with a message matching one of several accepted variants
        # (various library or locale phrasings of file not found).
        pytest.importorskip(module)

        path = os.path.join(HERE, "data", "does_not_exist." + fn_ext)
        msg1 = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Expected object or value"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "method, module, error_class, fn_ext",
        [
            (pd.DataFrame.to_csv, "os", OSError, "csv"),
            (pd.DataFrame.to_html, "os", OSError, "html"),
            (pd.DataFrame.to_excel, "xlrd", OSError, "xlsx"),
            (pd.DataFrame.to_feather, "pyarrow", OSError, "feather"),
            (pd.DataFrame.to_parquet, "pyarrow", OSError, "parquet"),
            (pd.DataFrame.to_stata, "os", OSError, "dta"),
            (pd.DataFrame.to_json, "os", OSError, "json"),
            (pd.DataFrame.to_pickle, "os", OSError, "pickle"),
        ],
    )
    # NOTE: Missing parent directory for pd.DataFrame.to_hdf is handled by PyTables
    def test_write_missing_parent_directory(self, method, module, error_class, fn_ext):
        # [FUNCTION DOC]
        # Parametrized over eight writers (csv, html, excel, feather, parquet, stata,
        # json, pickle). Writing a small frame to a path whose parent directory is
        # missing must raise an OS error saying the file cannot be saved into a non-
        # existent directory. HDF writing is excluded because PyTables handles it.
        pytest.importorskip(module)

        dummy_frame = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})

        path = os.path.join(HERE, "data", "missing_folder", "does_not_exist." + fn_ext)

        with pytest.raises(
            error_class,
            match=r"Cannot save file into a non-existent directory: .*missing_folder",
        ):
            method(dummy_frame, path)

    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_table, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_expands_user_home_dir(
        self, reader, module, error_class, fn_ext, monkeypatch
    ):
        # [FUNCTION DOC]
        # Skipped on WASM. Parametrized over ten readers. Patches the expand-user helper
        # to prefix a folder, then reads a home-relative nonexistent path and expects
        # the reader to raise the given exception with one of several accepted messages,
        # showing the home expansion hook is called for each reader.
        pytest.importorskip(module)

        path = os.path.join("~", "does_not_exist." + fn_ext)
        monkeypatch.setattr(icom, "_expand_user", lambda x: os.path.join("foo", x))

        msg1 = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "reader, module, path",
        [
            (pd.read_csv, "os", ("io", "data", "csv", "iris.csv")),
            (pd.read_table, "os", ("io", "data", "csv", "iris.csv")),
            (
                pd.read_fwf,
                "os",
                ("io", "data", "fixed_width", "fixed_width_format.txt"),
            ),
            (pd.read_excel, "xlrd", ("io", "data", "excel", "test1.xlsx")),
            (
                pd.read_feather,
                "pyarrow",
                ("io", "data", "feather", "simple_dataset.feather"),
            ),
            (
                pd.read_hdf,
                "tables",
                ("io", "data", "legacy_hdf", "pytables_native2.h5"),
            ),
            (pd.read_stata, "os", ("io", "data", "stata", "stata10_115.dta")),
            (pd.read_sas, "os", ("io", "sas", "data", "test1.sas7bdat")),
            (pd.read_json, "os", ("io", "json", "data", "tsframe_v012.json")),
            (
                pd.read_pickle,
                "os",
                ("io", "data", "pickle", "categorical.0.25.0.pickle"),
            ),
        ],
    )
    def test_read_fspath_all(self, reader, module, path, datapath):
        # [FUNCTION DOC]
        # Parametrized over ten readers with a sample data file each. Reads via a custom
        # path-like object and via a plain path; results must be equal (categorical
        # comparison for the pickle case, frame comparison otherwise).
        pytest.importorskip(module)
        path = datapath(*path)

        mypath = CustomFSPath(path)
        result = reader(mypath)
        expected = reader(path)

        if path.endswith(".pickle"):
            # categorical
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "writer_name, writer_kwargs, module",
        [
            ("to_csv", {}, "os"),
            ("to_excel", {"engine": "openpyxl"}, "openpyxl"),
            ("to_feather", {}, "pyarrow"),
            ("to_html", {}, "os"),
            ("to_json", {}, "os"),
            ("to_latex", {}, "os"),
            ("to_pickle", {}, "os"),
            ("to_stata", {"time_stamp": pd.to_datetime("2019-01-01 00:00")}, "os"),
        ],
    )
    def test_write_fspath_all(self, writer_name, writer_kwargs, module, tmp_path):
        # [FUNCTION DOC]
        # Parametrized over eight writer methods with their keyword arguments (excel
        # uses the openpyxl engine, stata uses a fixed timestamp). Writes a small frame
        # once to a string path and once to a custom path-like in a temporary directory,
        # then compares the files byte for byte; for excel it reads both back and
        # compares frames instead because the binary contains creation time. The latex
        # writer requires jinja2.
        if writer_name in ["to_latex"]:  # uses Styler implementation
            pytest.importorskip("jinja2")
        string = str(tmp_path / "string")
        fspath = str(tmp_path / "fspath")
        df = pd.DataFrame({"A": [1, 2]})

        pytest.importorskip(module)
        mypath = CustomFSPath(fspath)
        writer = getattr(df, writer_name)

        writer(string, **writer_kwargs)
        writer(mypath, **writer_kwargs)
        with open(string, "rb") as f_str, open(fspath, "rb") as f_path:
            if writer_name == "to_excel":
                # binary representation of excel contains time creation
                # data that causes flaky CI failures
                result = pd.read_excel(f_str, **writer_kwargs)
                expected = pd.read_excel(f_path, **writer_kwargs)
                tm.assert_frame_equal(result, expected)
            else:
                result = f_str.read()
                expected = f_path.read()
                assert result == expected

    def test_write_fspath_hdf5(self, tmp_path):
        # [FUNCTION DOC]
        # Needs pytables. Writes the same frame through a custom path-like and a string
        # path, then reads both back and compares frames (HDF5 files are not byte-
        # identical).
        # Same test as write_fspath_all, except HDF5 files aren't
        # necessarily byte-for-byte identical for a given dataframe, so we'll
        # have to read and compare equality
        pytest.importorskip("tables")

        df = pd.DataFrame({"A": [1, 2]})
        string = str(tmp_path / "string")
        fspath = str(tmp_path / "fspath")

        mypath = CustomFSPath(fspath)
        df.to_hdf(mypath, key="bar")
        df.to_hdf(string, key="bar")

        result = pd.read_hdf(fspath, key="bar")
        expected = pd.read_hdf(string, key="bar")

        tm.assert_frame_equal(result, expected)


@pytest.fixture
def mmap_file(datapath):
    # [FUNCTION DOC]
    # HELPER: fixture returning the path of the sample CSV used for memory-map tests.
    return datapath("io", "data", "csv", "test_mmap.csv")


class TestMMapWrapper:
    # [CLASS DOC]
    # Tests for memory-mapped file handling and some encoding/engine option validation
    # using the CSV reader and writer.
    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    def test_constructor_bad_file(self, mmap_file):
        # [FUNCTION DOC]
        # Skipped on WASM. The private memory-map helper given a text buffer whose file
        # descriptor is invalid must raise the memory-map error (Errno 22) or on Windows
        # an OS error with a parameter-incorrect message. Given an already-closed real
        # file it must raise a value error about I/O on a closed file.
        non_file = StringIO("I am not a file")
        non_file.fileno = lambda: -1

        # the error raised is different on Windows
        if is_platform_windows():
            msg = "The parameter is incorrect"
            err = OSError
        else:
            msg = "[Errno 22]"
            err = mmap.error

        with pytest.raises(err, match=msg):
            icom._maybe_memory_map(non_file, True)

        with open(mmap_file, encoding="utf-8") as target:
            pass

        msg = "I/O operation on closed file"
        with pytest.raises(ValueError, match=msg):
            icom._maybe_memory_map(target, True)

    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    def test_next(self, mmap_file):
        # [FUNCTION DOC]
        # Skipped on WASM. Opens the sample file in text mode with memory mapping; the
        # wrapper's underlying buffer must be a memory map. Each call to get the next
        # item must return a line equal (ignoring surrounding whitespace) to the
        # corresponding line, and an extra call after the end must raise a stop-
        # iteration with an empty message.
        with open(mmap_file, encoding="utf-8") as target:
            lines = target.readlines()

            with icom.get_handle(
                target, "r", is_text=True, memory_map=True
            ) as wrappers:
                wrapper = wrappers.handle
                assert isinstance(wrapper.buffer.buffer, mmap.mmap)

                for line in lines:
                    next_line = next(wrapper)
                    assert next_line.strip() == line.strip()

                with pytest.raises(StopIteration, match=r"^$"):
                    next(wrapper)

    def test_unknown_engine(self, temp_file):
        # [FUNCTION DOC]
        # Reading a CSV with an unrecognized engine name must raise a value error that
        # says the engine is unknown.
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        )
        df.to_csv(temp_file)
        with pytest.raises(ValueError, match="Unknown engine"):
            pd.read_csv(temp_file, engine="pyt")

    def test_binary_mode(self, temp_file):
        # [FUNCTION DOC]
        # GH 35058. Writing a frame with a binary write mode must not pass encoding to
        # open; reading it back with the first column as index equals the original.
        """
        'encoding' shouldn't be passed to 'open' in binary mode.

        GH 35058
        """
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        )
        df.to_csv(temp_file, mode="w+b")
        tm.assert_frame_equal(df, pd.read_csv(temp_file, index_col=0))

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
    @pytest.mark.parametrize("compression_", ["bz2", "xz"])
    def test_warning_missing_utf_bom(self, encoding, compression_, temp_file):
        # [FUNCTION DOC]
        # GH 35681. Parametrized over utf-16/utf-32 and bz2/xz compression. Writing a
        # CSV must produce a unicode warning about a missing byte order mark, and
        # reading it back must raise a unicode error (BOM or decode failure).
        """
        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.

        https://stackoverflow.com/questions/55171439

        GH 35681
        """
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        )
        with tm.assert_produces_warning(UnicodeWarning, match="byte order mark"):
            df.to_csv(temp_file, compression=compression_, encoding=encoding)

        # reading should fail (otherwise we wouldn't need the warning)
        msg = (
            r"UTF-\d+ stream does not start with BOM|"
            r"'utf-\d+' codec can't decode byte"
        )
        with pytest.raises(UnicodeError, match=msg):
            pd.read_csv(temp_file, compression=compression_, encoding=encoding)


def test_is_fsspec_url():
    # [FUNCTION DOC]
    # Checks the fsspec URL detector: gcs and gs schemes are accepted, plain http,
    # unknown prefixes and local paths are not, URLs embedded in other text are not, and
    # any scheme conforming to RFC 3986 is accepted.
    assert icom.is_fsspec_url("gcs://pandas/somethingelse.com")
    assert icom.is_fsspec_url("gs://pandas/somethingelse.com")
    # the following is the only remote URL that is handled without fsspec
    assert not icom.is_fsspec_url("http://pandas/somethingelse.com")
    assert not icom.is_fsspec_url("random:pandas/somethingelse.com")
    assert not icom.is_fsspec_url("/local/path")
    assert not icom.is_fsspec_url("relative/local/path")
    # fsspec URL in string should not be recognized
    assert not icom.is_fsspec_url("this is not fsspec://url")
    assert not icom.is_fsspec_url("{'url': 'gs://pandas/somethingelse.com'}")
    # accept everything that conforms to RFC 3986 schema
    assert icom.is_fsspec_url("RFC-3986+compliant.spec://something")


def test_is_fsspec_url_chained():
    # [FUNCTION DOC]
    # GH 48978. Chained URLs joined with double colons (such as cache plus zip plus
    # cloud) are accepted; single-colon, triple-colon and empty-scheme chains are
    # rejected.
    # GH#48978 Support chained fsspec URLs
    # See https://filesystem-spec.readthedocs.io/en/latest/features.html#url-chaining.
    assert icom.is_fsspec_url("filecache::s3://pandas/test.csv")
    assert icom.is_fsspec_url("zip://test.csv::filecache::gcs://bucket/file.zip")
    assert icom.is_fsspec_url("filecache::zip://test.csv::gcs://bucket/file.zip")
    assert icom.is_fsspec_url("filecache::dask::s3://pandas/test.csv")
    assert not icom.is_fsspec_url("filecache:s3://pandas/test.csv")
    assert not icom.is_fsspec_url("filecache:::s3://pandas/test.csv")
    assert not icom.is_fsspec_url("filecache::://pandas/test.csv")


@pytest.mark.parametrize("format", ["csv", "json"])
def test_codecs_encoding(format, temp_file):
    # [FUNCTION DOC]
    # GH 39247. Parametrized over csv and json. Writes a frame to a file handle opened
    # in text mode with utf-8, reads it back through another handle (csv with first
    # column index) and asserts equality.
    # GH39247
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    with open(temp_file, mode="w", encoding="utf-8") as handle:
        getattr(expected, f"to_{format}")(handle)
    with open(temp_file, encoding="utf-8") as handle:
        if format == "csv":
            df = pd.read_csv(handle, index_col=0)
        else:
            df = pd.read_json(handle)
    tm.assert_frame_equal(expected, df)


def test_codecs_get_writer_reader(temp_file):
    # [FUNCTION DOC]
    # GH 39247. Writes a CSV via a codecs utf-8 stream writer wrapping a binary file,
    # reads via a codecs stream reader; frames must be equal.
    # GH39247
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    with open(temp_file, "wb") as handle:
        with codecs.getwriter("utf-8")(handle) as encoded:
            expected.to_csv(encoded)
    with open(temp_file, "rb") as handle:
        with codecs.getreader("utf-8")(handle) as encoded:
            df = pd.read_csv(encoded, index_col=0)
    tm.assert_frame_equal(expected, df)


@pytest.mark.parametrize(
    "io_class,mode,msg",
    [
        (BytesIO, "t", "a bytes-like object is required, not 'str'"),
        (StringIO, "b", "string argument expected, got 'bytes'"),
    ],
)
def test_explicit_encoding(io_class, mode, msg):
    # [FUNCTION DOC]
    # GH 39247. Parametrized over (binary buffer, text mode flag, message) and (text
    # buffer, binary mode flag, message). If the caller passes an explicit text or
    # binary mode it is honored; writing a CSV to the mismatched buffer type must raise
    # a type error with the given message.
    # GH39247; this test makes sure that if a user provides mode="*t" or "*b",
    # it is used. In the case of this test it leads to an error as intentionally the
    # wrong mode is requested
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    with io_class() as buffer:
        with pytest.raises(TypeError, match=msg):
            expected.to_csv(buffer, mode=f"w{mode}")


@pytest.mark.parametrize("encoding_errors", ["strict", "replace"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_encoding_errors(encoding_errors, format, temp_file):
    # [FUNCTION DOC]
    # GH 39450. Parametrized over encoding_errors strict and replace and over csv and
    # json. Writes invalid utf-8 bytes to a temporary file. With strict it expects a
    # unicode decode error; with replace it reads successfully with the replacement
    # character in index, column and value, matching the expected frame.
    # GH39450
    msg = "'utf-8' codec can't decode byte"
    bad_encoding = b"\xe4"

    if format == "csv":
        content = b"," + bad_encoding + b"\n" + bad_encoding * 2 + b"," + bad_encoding
        reader = partial(pd.read_csv, index_col=0)
    else:
        content = (
            b'{"'
            + bad_encoding * 2
            + b'": {"'
            + bad_encoding
            + b'":"'
            + bad_encoding
            + b'"}}'
        )
        reader = partial(pd.read_json, orient="index")
    file = temp_file
    file.write_bytes(content)

    if encoding_errors != "replace":
        with pytest.raises(UnicodeDecodeError, match=msg):
            reader(temp_file, encoding_errors=encoding_errors)
    else:
        df = reader(temp_file, encoding_errors=encoding_errors)
        decoded = bad_encoding.decode(errors=encoding_errors)
        expected = pd.DataFrame({decoded: [decoded]}, index=[decoded * 2])
        tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("encoding_errors", [0, None])
def test_encoding_errors_badtype(encoding_errors):
    # [FUNCTION DOC]
    # GH 59075. A non-string encoding_errors (zero or none) passed to read_csv must
    # raise a value error whose message states that it must be a string and names the
    # type received.
    # GH 59075
    content = StringIO("A,B\n1,2\n3,4\n")
    reader = partial(pd.read_csv, encoding_errors=encoding_errors)
    expected_error = "encoding_errors must be a string, got "
    expected_error += f"{type(encoding_errors).__name__}"
    with pytest.raises(ValueError, match=expected_error):
        reader(content)


def test_bad_encdoing_errors(temp_file):
    # [FUNCTION DOC]
    # GH 39777. Asking get_handle for write mode with an unknown error-handler name must
    # raise a lookup error about the unknown handler name.
    # GH 39777
    with pytest.raises(LookupError, match="unknown error handler name"):
        icom.get_handle(temp_file, "w", errors="bad")


@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_errno_attribute():
    # [FUNCTION DOC]
    # GH 13872. Skipped on WASM. Reading a nonexistent file must raise a file-not-found
    # error with Errno 2 in its message.
    # KNOWN ISSUE: the assertion on the error number is placed inside the raises block
    # after the call that raises, so it never runs; and it examines the exception-info
    # wrapper rather than the exception.
    # GH 13872
    with pytest.raises(FileNotFoundError, match="\\[Errno 2\\]") as err:
        pd.read_csv("doesnt_exist")
        assert err.errno == errno.ENOENT


def test_fail_mmap():
    # [FUNCTION DOC]
    # Requesting memory mapping on a buffer without a file descriptor must raise an
    # unsupported-operation error mentioning fileno.
    with pytest.raises(UnsupportedOperation, match="fileno"):
        with BytesIO() as buffer:
            icom.get_handle(buffer, "rb", memory_map=True)


def test_close_on_error():
    # [FUNCTION DOC]
    # GH 47136. Appends an object whose close method raises an OS error to the handles'
    # created_handles list inside get_handle; leaving the context must propagate that OS
    # error.
    # GH 47136
    class TestError:
        # [CLASS DOC]
        # HELPER: test double with a close method that always raises an OS error with
        # message test.
        def close(self):
            # [FUNCTION DOC]
            # HELPER: close method of the TestError double; raises an OS error with the
            # message test.
            raise OSError("test")

    with pytest.raises(OSError, match="test"):
        with BytesIO() as buffer:
            with icom.get_handle(buffer, "rb") as handles:
                handles.created_handles.append(TestError())


@td.skip_if_no("fsspec")
@pytest.mark.parametrize("compression", [None, "infer"])
def test_read_csv_chained_url_no_error(datapath, compression):
    # [FUNCTION DOC]
    # GH 60100. Needs fsspec. Reads a CSV member of a tar archive through a chained URL
    # (tar over local file) with compression none or infer and semicolon separator; the
    # result equals a one-row frame with columns named 1 and 2 holding 3 and 4.
    # GH 60100
    tar_file_path = datapath("io", "data", "tar", "test-csv.tar")
    chained_file_url = f"tar://test.csv::file://{tar_file_path}"

    result = pd.read_csv(chained_file_url, compression=compression, sep=";")
    expected = pd.DataFrame({"1": {0: 3}, "2": {0: 4}})

    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "reader",
    [
        pd.read_csv,
        pd.read_fwf,
        pd.read_excel,
        pd.read_feather,
        pd.read_hdf,
        pd.read_stata,
        pd.read_sas,
        pd.read_json,
        pd.read_pickle,
    ],
)
def test_pickle_reader(reader):
    # [FUNCTION DOC]
    # GH 22265. Parametrized over nine reader functions; each must be picklable (dumping
    # to an in-memory buffer must not raise).
    # GH 22265
    with BytesIO() as buffer:
        pickle.dump(reader, buffer)


@td.skip_if_no("pyarrow")
def test_pyarrow_read_csv_datetime_dtype():
    # [FUNCTION DOC]
    # GH 59904. Needs pyarrow. Reads CSV text with a quoted date column containing a
    # blank, parsing dates day-first with the pyarrow dtype backend; the result must
    # equal a frame of the same dates parsed with day-first, with the blank as missing.
    # GH 59904
    data = '"date"\n"20/12/2025"\n""\n"31/12/2020"'
    result = pd.read_csv(
        StringIO(data), parse_dates=["date"], dayfirst=True, dtype_backend="pyarrow"
    )

    expect_data = pd.to_datetime(["20/12/2025", pd.NaT, "31/12/2020"], dayfirst=True)
    expect = pd.DataFrame({"date": expect_data})

    tm.assert_frame_equal(expect, result)
