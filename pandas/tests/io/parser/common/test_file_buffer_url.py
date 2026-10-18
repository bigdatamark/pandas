"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# [MODULE DOC]
# Parser tests (run against every engine through the parser fixture) of the different
# kinds of inputs and of end-of-input states: URLs, local file URLs, path objects,
# nonexistent and unreadable paths, odd end-of-file situations in the tokenizer,
# temporary files, embedded end-of-file bytes, ownership and closing of user-provided
# file handles, invalid buffer objects, memory mapping (also with compression), and
# reader context managers.
# Defines two marker aliases: one marks a test as expected to fail for the pyarrow
# engine and one skips it for that engine.

from io import (
    BytesIO,
    StringIO,
)
import os
import platform
from urllib.error import URLError
import uuid

import numpy as np
import pytest

from pandas.compat import WASM
from pandas.errors import (
    EmptyDataError,
    ParserError,
)
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@pytest.mark.network
@pytest.mark.single_cpu
def test_url(all_parsers, csv_dir_path, httpserver):
    # [FUNCTION DOC]
    # Marked as a network test, single cpu. Serves a local salaries data file from the
    # test HTTP server fixture and verifies that reading from the served URL with a tab
    # separator gives the same frame as reading the local file.
    parser = all_parsers
    kwargs = {"sep": "\t"}

    local_path = os.path.join(csv_dir_path, "salaries.csv")
    with open(local_path, encoding="utf-8") as f:
        httpserver.serve_content(content=f.read())

    url_result = parser.read_csv(httpserver.url, **kwargs)

    local_result = parser.read_csv(local_path, **kwargs)
    tm.assert_frame_equal(url_result, local_result)


@pytest.mark.slow
def test_local_file(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Marked slow. Reads a salaries file with a tab separator both by path and through a
    # localhost file URL and verifies equal frames. If the URL read raises a URL error
    # (fails on some systems) the test is skipped with the platform description.
    parser = all_parsers
    kwargs = {"sep": "\t"}

    local_path = os.path.join(csv_dir_path, "salaries.csv")
    local_result = parser.read_csv(local_path, **kwargs)
    url = "file://localhost/" + local_path

    try:
        url_result = parser.read_csv(url, **kwargs)
        tm.assert_frame_equal(url_result, local_result)
    except URLError:
        # Fails on some systems.
        pytest.skip("Failing on: " + " ".join(platform.uname()))


@xfail_pyarrow  # AssertionError: DataFrame.index are different
def test_path_path_lib(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Marked expected to fail for pyarrow (index differs). Round-trips a thirty-row
    # float frame with string index through writing and reading using a path object in a
    # temporary location, reading with the first column as index, and verifies the frame
    # is unchanged.
    parser = all_parsers
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    result = tm.round_trip_pathlib(
        df.to_csv, lambda p: parser.read_csv(p, index_col=0), temp_file
    )
    tm.assert_frame_equal(df, result)


@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_nonexistent_path(all_parsers):
    # [FUNCTION DOC]
    # Skipped on WASM. Verifies that reading a randomly named path that does not exist
    # raises a file-not-found error with the operating system error number two in the
    # message, and that the error's filename attribute equals the path given.
    # gh-2428: pls no segfault
    # gh-14086: raise more helpful FileNotFoundError
    # GH#29233 "File foo" instead of "File b'foo'"
    parser = all_parsers
    path = f"{uuid.uuid4()}.csv"

    msg = r"\[Errno 2\]"
    with pytest.raises(FileNotFoundError, match=msg) as e:
        parser.read_csv(path)
    assert path == e.value.filename


@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
@td.skip_if_windows  # os.chmod does not work in windows
def test_no_permission(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Skipped on WASM and on Windows. Removes all permissions from a temporary file,
    # skips the test if the process can nonetheless open it (running with elevated
    # rights), and otherwise verifies that reading raises a permission error with error
    # number thirteen and the resolved path as the error's filename.
    # Side effect: changes permissions of the temporary file.
    # GH 23784
    parser = all_parsers

    msg = r"\[Errno 13\]"
    path = temp_file
    os.chmod(path, 0)  # make file unreadable

    # verify that this process cannot open the file (not running as sudo)
    try:
        with open(path, encoding="utf-8"):
            pass
        pytest.skip("Running as sudo.")
    except PermissionError:
        pass

    with pytest.raises(PermissionError, match=msg) as e:
        parser.read_csv(path)
    assert str(path.resolve()) == e.value.filename


@pytest.mark.parametrize(
    "data,kwargs,expected,msg",
    [
        # gh-10728: WHITESPACE_LINE
        (
            "a,b,c\n4,5,6\n ",
            {},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # gh-10548: EAT_LINE_COMMENT
        (
            "a,b,c\n4,5,6\n#comment",
            {"comment": "#"},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_CRNL_NOP
        (
            "a,b,c\n4,5,6\n\r",
            {},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_COMMENT
        (
            "a,b,c\n4,5,6#comment",
            {"comment": "#"},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # SKIP_LINE
        (
            "a,b,c\n4,5,6\nskipme",
            {"skiprows": [2]},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_LINE_COMMENT
        (
            "a,b,c\n4,5,6\n#comment",
            {"comment": "#", "skip_blank_lines": False},
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # IN_FIELD
        (
            "a,b,c\n4,5,6\n ",
            {"skip_blank_lines": False},
            DataFrame([["4", 5, 6], [" ", None, None]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_CRNL
        (
            "a,b,c\n4,5,6\n\r",
            {"skip_blank_lines": False},
            DataFrame([[4, 5, 6], [None, None, None]], columns=["a", "b", "c"]),
            None,
        ),
        # ESCAPED_CHAR
        (
            "a,b,c\n4,5,6\n\\",
            {"escapechar": "\\"},
            None,
            "(EOF following escape character)|(unexpected end of data)",
        ),
        # ESCAPE_IN_QUOTED_FIELD
        (
            'a,b,c\n4,5,6\n"\\',
            {"escapechar": "\\"},
            None,
            "(EOF inside string starting at row 2)|(unexpected end of data)",
        ),
        # IN_QUOTED_FIELD
        (
            'a,b,c\n4,5,6\n"',
            {"escapechar": "\\"},
            None,
            "(EOF inside string starting at row 2)|(unexpected end of data)",
        ),
    ],
    ids=[
        "whitespace-line",
        "eat-line-comment",
        "eat-crnl-nop",
        "eat-comment",
        "skip-line",
        "eat-line-comment",
        "in-field",
        "eat-crnl",
        "escaped-char",
        "escape-in-quoted-field",
        "in-quoted-field",
    ],
)
def test_eof_states(all_parsers, data, kwargs, expected, msg, request):
    # [FUNCTION DOC]
    # Parametrized over inputs that end in each internal tokenizer state (whitespace
    # line, line comment, carriage return, comment, skipped line, in a field, escape
    # character, escape inside quoted field, in quoted field). Verifies that either the
    # expected frame with one data row is returned, or a parser error about end of file
    # (inside a string or following an escape character) is raised, depending on the
    # case.
    # For pyarrow, cases with a comment option expect a value error that the option is
    # unsupported, and cases without a carriage return are skipped (known upstream
    # behavior difference).
    # KNOWN ISSUE: two cases share the id eat-line-comment (the second one also turns
    # off blank-line skipping), so they cannot be told apart by id.
    # see gh-10728, gh-10548
    parser = all_parsers

    if parser.engine == "pyarrow" and "comment" in kwargs:
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    if parser.engine == "pyarrow" and "\r" not in data:
        # pandas.errors.ParserError: CSV parse error: Expected 3 columns, got 1:
        # ValueError: skiprows argument must be an integer when using engine='pyarrow'
        # AssertionError: Regex pattern did not match.
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    if expected is None:
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


def test_temporary_file(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Writes a line of two zeros to a temporary file opened for reading and writing,
    # rewinds it, and verifies reading from the open handle with whitespace separator
    # and no header gives a one-row frame of two zeros. For pyarrow, verifies the regex
    # separator error.
    # see gh-13398
    parser = all_parsers
    data = "0 0"

    with temp_file.open(mode="w+", encoding="utf-8") as new_file:
        new_file.write(data)
        new_file.flush()
        new_file.seek(0)

        if parser.engine == "pyarrow":
            msg = "the 'pyarrow' engine does not support regex separators"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(new_file, sep=r"\s+", header=None)
            return

        result = parser.read_csv(new_file, sep=r"\s+", header=None)

        expected = DataFrame([[0, 0]])
        tm.assert_frame_equal(result, expected)


def test_internal_eof_byte(all_parsers):
    # [FUNCTION DOC]
    # Verifies a control-Z (substitute) character inside a field of text input is
    # retained as part of the string value.
    # see gh-5500
    parser = all_parsers
    data = "a,b\n1\x1a,2"

    expected = DataFrame([["1\x1a", 2]], columns=["a", "b"])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_internal_eof_byte_to_file(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Writes bytes containing a control-Z character inside a quoted field to a randomly
    # named file next to the temporary file and verifies reading it returns the string
    # with the character retained, and the unquoted second field with its leading space.
    # Side effect: leaves the created file in the temporary file's directory.
    # see gh-16559
    parser = all_parsers
    data = b'c1,c2\r\n"test \x1a    test", test\r\n'
    expected = DataFrame([["test \x1a    test", " test"]], columns=["c1", "c2"])
    path = f"__{uuid.uuid4()}__.csv"

    path2 = temp_file.parent / path
    with open(path2, "wb") as f:
        f.write(data)

    result = parser.read_csv(path2)
    tm.assert_frame_equal(result, expected)


def test_file_handle_string_io(all_parsers):
    # [FUNCTION DOC]
    # Verifies reading from an in-memory text stream does not close the stream.
    # gh-14418
    #
    # Don't close user provided file handles.
    parser = all_parsers
    data = "a,b\n1,2"

    fh = StringIO(data)
    parser.read_csv(fh)
    assert not fh.closed


def test_file_handles_with_open(all_parsers, csv1):
    # [FUNCTION DOC]
    # Verifies that reading from an opened file handle, in text mode and in binary mode,
    # does not close the handle.
    # gh-14418
    #
    # Don't close user provided file handles.
    parser = all_parsers

    for mode in ["r", "rb"]:
        with open(csv1, mode, encoding="utf-8" if mode == "r" else None) as f:
            parser.read_csv(f)
            assert not f.closed


def test_invalid_file_buffer_class(all_parsers):
    # [FUNCTION DOC]
    # Verifies passing an instance of a plain class (no file interface) raises a value
    # error about an invalid file path or buffer object type.
    # see gh-15337
    class InvalidBuffer:
        # [CLASS DOC]
        # HELPER: empty class used as an invalid buffer argument.
        pass

    parser = all_parsers
    msg = "Invalid file path or buffer object type"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(InvalidBuffer())


def test_invalid_file_buffer_mock(all_parsers):
    # [FUNCTION DOC]
    # Verifies, as the previous test, that an instance of another plain class raises the
    # invalid file path or buffer object type value error.
    # KNOWN ISSUE: despite the name no mock is used; the test is effectively a
    # duplicate of the invalid buffer class test.
    # see gh-15337
    parser = all_parsers
    msg = "Invalid file path or buffer object type"

    class Foo:
        # [CLASS DOC]
        # HELPER: empty class used as an invalid buffer argument.
        pass

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(Foo())


def test_valid_file_buffer_seems_invalid(all_parsers):
    # [FUNCTION DOC]
    # Verifies that a text stream whose position query and reposition methods raise an
    # attribute error can still be read, proving the reader does not rely on those
    # methods.
    # gh-16135: we want to ensure that "tell" and "seek"
    # aren't actually being used when we call `read_csv`
    #
    # Thus, while the object may look "invalid" (these
    # methods are attributes of the `StringIO` class),
    # it is still a valid file-object for our purposes.
    class NoSeekTellBuffer(StringIO):
        # [CLASS DOC]
        # HELPER: text stream subclass whose position-query and seek methods both raise
        # an attribute error.
        def tell(self):
            # [FUNCTION DOC]
            # HELPER: always raises an attribute error saying there is no tell method.
            raise AttributeError("No tell method")

        def seek(self, pos, whence=0):
            # [FUNCTION DOC]
            # HELPER: takes a position and an optional mode argument and always raises
            # an attribute error saying there is no seek method.
            raise AttributeError("No seek method")

    data = "a\n1"
    parser = all_parsers
    expected = DataFrame({"a": [1]})

    result = parser.read_csv(NoSeekTellBuffer(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("io_class", [StringIO, BytesIO])
@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_read_csv_file_handle(all_parsers, io_class, encoding):
    # [FUNCTION DOC]
    # Verifies reading from text or bytes in-memory streams, with or without an explicit
    # encoding, gives the expected frame and leaves the user-provided stream open.
    """
    Test whether read_csv does not close user-provided file handles.

    GH 36980
    """
    parser = all_parsers
    expected = DataFrame({"a": [1], "b": [2]})

    content = "a,b\n1,2"
    handle = io_class(content.encode("utf-8") if io_class == BytesIO else content)

    tm.assert_frame_equal(parser.read_csv(handle, encoding=encoding), expected)
    assert not handle.closed


def test_memory_map_compression(all_parsers, compression, temp_file):
    # [FUNCTION DOC]
    # Writes a small frame to a temporary file with the compression fixture value and
    # verifies reading it with memory mapping enabled and the same compression gives
    # back the frame. For pyarrow, verifies the unsupported memory-map error.
    """
    Support memory map for compressed files.

    GH 37621
    """
    parser = all_parsers
    expected = DataFrame({"a": [1], "b": [2]})

    path = temp_file
    expected.to_csv(path, index=False, compression=compression)

    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(path, memory_map=True, compression=compression)
        return

    result = parser.read_csv(path, memory_map=True, compression=compression)

    tm.assert_frame_equal(
        result,
        expected,
    )


def test_context_manager(all_parsers, datapath):
    # [FUNCTION DOC]
    # Verifies the file handle opened by a chunked reader on a path is open before use
    # and closed after leaving the reader's context manager (an assertion error is
    # raised deliberately inside to leave the block early). For pyarrow, verifies the
    # unsupported chunk size error.
    # make sure that opened files are closed
    parser = all_parsers

    path = datapath("io", "data", "csv", "iris.csv")

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(path, chunksize=1)
        return

    reader = parser.read_csv(path, chunksize=1)
    assert not reader.handles.handle.closed
    try:
        with reader:
            next(reader)
            raise AssertionError
    except AssertionError:
        assert reader.handles.handle.closed


def test_context_manageri_user_provided(all_parsers, datapath):
    # [FUNCTION DOC]
    # Same as the previous test but with a user-provided open file: verifies the handle
    # stays open after the reader's context is exited. For pyarrow, verifies the
    # unsupported chunk size error.
    # KNOWN ISSUE: the function name has a stray trailing letter (manageri) instead of
    # context_manager_user_provided.
    # make sure that user-provided handles are not closed
    parser = all_parsers

    with open(datapath("io", "data", "csv", "iris.csv"), encoding="utf-8") as path:
        if parser.engine == "pyarrow":
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, chunksize=1)
            return

        reader = parser.read_csv(path, chunksize=1)
        assert not reader.handles.handle.closed
        try:
            with reader:
                next(reader)
                raise AssertionError
        except AssertionError:
            assert not reader.handles.handle.closed


@skip_pyarrow  # ParserError: Empty CSV file
def test_file_descriptor_leak(all_parsers, temp_file):
    # [FUNCTION DOC]
    # Skipped for pyarrow. Verifies that reading an empty file raises an empty-data
    # error ("No columns to parse from file"; regression for a descriptor leak on this
    # error path, though the test itself does not inspect descriptors).
    # GH 31488
    parser = all_parsers
    path = temp_file
    with pytest.raises(EmptyDataError, match="No columns to parse from file"):
        parser.read_csv(path)


def test_memory_map(all_parsers, csv_dir_path):
    # [FUNCTION DOC]
    # Reads a small three-column file with memory mapping enabled and verifies the
    # expected frame. For pyarrow, verifies the unsupported memory-map error.
    mmap_file = os.path.join(csv_dir_path, "test_mmap.csv")
    parser = all_parsers

    expected = DataFrame(
        {"a": [1, 2, 3], "b": ["one", "two", "three"], "c": ["I", "II", "III"]}
    )

    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(mmap_file, memory_map=True)
        return

    result = parser.read_csv(mmap_file, memory_map=True)
    tm.assert_frame_equal(result, expected)
