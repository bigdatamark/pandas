"""Common I/O API utilities"""

# [MODULE DOC]
# Shared plumbing for all pandas readers and writers: turning user supplied
# paths, URLs, remote filesystem URLs and buffers into usable open handles,
# with transparent compression and text/binary adaptation.
# Public contents: the two small result dataclasses (resolved arguments and
# managed handles, the latter a context manager that closes everything it
# created); URL and path helpers (url check, remote-filesystem url check,
# path stringification and user expansion, file existence, parent directory
# check, path to file-URL conversion); compression helpers (extension map,
# method extraction from a string or mapping, inference from a file name); the
# main handle-opening function (with typed overloads) supporting gzip, bz2,
# zip, tar, xz and zstd; private writer/reader wrapper classes for archives and
# for adapting text buffers to binary and partial buffer interfaces; memory
# mapping and binary-mode detection helpers; header argument validation; and
# helpers to detect potential multi-index columns and to de-duplicate column
# names.
# Relationships: used by essentially every io module (csv/parsers, excel,
# json, pickle, parquet, xml, html, stata, formats) and by frame writers.
# Import-time side effects: builds the set of valid URL schemes and the
# compiled remote-url regular expression; no I/O.

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import codecs
from collections import defaultdict
from collections.abc import (
    Hashable,
    Mapping,
    Sequence,
)
import dataclasses
import functools
import gzip
from io import (
    BufferedIOBase,
    BytesIO,
    RawIOBase,
    StringIO,
    TextIOBase,
    TextIOWrapper,
)
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    DefaultDict,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)
from urllib.parse import (
    urljoin,
    urlparse as parse_url,
    uses_netloc,
    uses_params,
    uses_relative,
)
import warnings
import zipfile

from pandas._typing import (
    BaseBuffer,
    ReadCsvBuffer,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool,
    is_file_like,
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import ABCMultiIndex

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")
_FSSPEC_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+\-+.]*(::[A-Za-z0-9+\-+.]+)*://")

BaseBufferT = TypeVar("BaseBufferT", bound=BaseBuffer)


if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        CompressionDict,
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import MultiIndex


@dataclasses.dataclass
class IOArgs:
    # [CLASS DOC]
    # Simple data container (dataclass) returned by the path/buffer resolution
    # helper. Fields: filepath_or_buffer (a string path or a buffer, possibly
    # an in-memory copy of a downloaded URL or an opened remote file), encoding
    # (text encoding to use), mode (file mode, adjusted for remote files),
    # compression (mapping with at least a method entry plus extra options),
    # should_close (default false: whether the caller must close the buffer
    # because it was created by the helper) and close_handles (default empty
    # list of extra objects to close after the buffer).
    """
    Return value of io/common.py:_get_filepath_or_buffer.
    """

    filepath_or_buffer: str | BaseBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = False
    close_handles: list[Any] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    # [CLASS DOC]
    # Dataclass, generic over text or bytes content, returned by the handle
    # opening function and usable as a context manager. Fields: handle (the
    # object to read/write through), compression (mapping of the resolved
    # method and options), created_handles (default empty list: every object
    # the opener created, in the order they should be closed) and is_wrapped
    # (default false: true when a text wrapper was put around a caller-owned
    # buffer and must be detached instead of closed).
    # Methods: close releases created handles; enter returns self; exit closes.
    # Invariant: after close, the created list is empty and is_wrapped is
    # false.
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    # handle might not implement the IO-interface
    handle: IO[AnyStr]
    compression: CompressionDict
    created_handles: list[IO[bytes] | IO[str]] = dataclasses.field(default_factory=list)
    is_wrapped: bool = False

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes every created handle. If a text wrapper was inserted around a
        # caller-owned buffer, it is flushed and detached (so the underlying
        # buffer stays open) and removed from the created list first. Then each
        # remaining created handle is closed, the list is cleared and the
        # wrapped flag reset. Returns nothing. Raises an assertion error if the
        # wrapped flag is set but the handle is not a text wrapper.
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        for handle in self.created_handles:
            handle.close()
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self) -> IOHandles[AnyStr]:
        # [FUNCTION DOC]
        # Context manager entry: returns this same object.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC]
        # Context manager exit: ignores exception details, closes all created
        # handles, and does not suppress exceptions.
        self.close()


def is_url(url: object) -> bool:
    # [FUNCTION DOC]
    # Returns true only if the input is a string whose parsed scheme is one of
    # the recognised URL schemes (relative, network-location or parameter
    # capable schemes known to the standard library, excluding the empty
    # scheme). Non-strings give false. No exceptions for non-string input.
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS


@overload
def _expand_user(filepath_or_buffer: str) -> str: ...
# [FUNCTION DOC]
# Overload stub (string input): see the implementation of the user expansion
# helper below.


@overload
def _expand_user(filepath_or_buffer: BaseBufferT) -> BaseBufferT: ...
# [FUNCTION DOC]
# Overload stub (buffer input): see the implementation of the user expansion
# helper below.


def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT:
    # [FUNCTION DOC]
    # HELPER: expands a leading tilde in a string path to the user's home
    # directory (via the operating system path routine); any non-string input
    # is returned unchanged. The two preceding overload stubs only describe
    # the string-to-string and buffer-to-buffer typing.
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer


def validate_header_arg(header: object) -> None:
    # [FUNCTION DOC]
    # Validates the header argument of the readers; returns nothing.
    # Accepted: none; a non-negative integer; a list-like (not a set) of
    # non-negative integers. Raises a value error for a negative integer
    # (suggesting none for no header), for a list containing non-integers or
    # negative integers, and for any other non-boolean type; raises a type
    # error for a boolean.
    if header is None:
        return
    if is_integer(header):
        header = cast(int, header)
        if header < 0:
            # GH 27779
            raise ValueError(
                "Passing negative integer to header is invalid. "
                "For no header, use header=None instead"
            )
        return
    if is_list_like(header, allow_sets=False):
        header = cast(Sequence, header)
        if not all(map(is_integer, header)):
            raise ValueError("header must be integer or list of integers")
        if any(i < 0 for i in header):
            raise ValueError("cannot specify multi-index header with negative integers")
        return
    if is_bool(header):
        raise TypeError(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify "
            "the row(s) making up the column names"
        )
    # GH 16338
    raise ValueError("header must be integer or list of integers")


@overload
def stringify_path(
    filepath_or_buffer: FilePath, convert_file_like: bool = ...
) -> str: ...
# [FUNCTION DOC]
# Overload stub (path input returns a string): see the implementation of the
# path stringification function below.


@overload
def stringify_path(
    filepath_or_buffer: BaseBufferT, convert_file_like: bool = ...
) -> BaseBufferT: ...
# [FUNCTION DOC]
# Overload stub (buffer input returns a buffer): see the implementation of the
# path stringification function below.


def stringify_path(
    filepath_or_buffer: FilePath | BaseBufferT,
    convert_file_like: bool = False,
) -> str | BaseBufferT:
    # [FUNCTION DOC]
    # Converts a path-like object to a string with the user home expanded.
    # Inputs: filepath_or_buffer (anything); convert_file_like (default false)
    # controls whether file-like objects that also implement the path
    # protocol are converted (otherwise they are returned untouched, so an
    # already open fsspec file is not opened twice; compression inference
    # passes true). Objects implementing the path protocol are replaced by
    # their file-system path; the result then goes through user expansion, so
    # everything else (bytes, buffers, non-path objects) passes through
    # unchanged. No exceptions of its own.
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if not convert_file_like and is_file_like(filepath_or_buffer):
        # GH 38125: some fsspec objects implement os.PathLike but have already opened a
        # file. This prevents opening the file a second time. infer_compression calls
        # this function with convert_file_like=True to infer the compression.
        return cast(BaseBufferT, filepath_or_buffer)

    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)


def urlopen(*args: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC]
    # Thin wrapper that imports the standard url-request module on first use
    # (to avoid its import cost at pandas import) and forwards all positional
    # and keyword arguments to its open function, returning the response
    # object (a context manager). Network errors propagate.
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request

    return urllib.request.urlopen(*args, **kwargs)  # noqa: TID251


def is_fsspec_url(url: FilePath | BaseBuffer) -> bool:
    # [FUNCTION DOC]
    # Returns true if the input is a string that starts with a protocol
    # (letters, digits and some punctuation, optionally chained with a double
    # colon) followed by a double slash, and is not an http or https url
    # (those are handled directly). Non-strings give false.
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    return (
        isinstance(url, str)
        and bool(_FSSPEC_URL_PATTERN.match(url))
        and not url.startswith(("http://", "https://"))
    )


def _get_filepath_or_buffer(
    filepath_or_buffer: FilePath | BaseBuffer,
    encoding: str = "utf-8",
    compression: CompressionOptions | None = None,
    mode: str = "r",
    storage_options: StorageOptions | None = None,
) -> IOArgs:
    # [FUNCTION DOC]
    # HELPER: resolves a user supplied path, url or buffer into an argument
    # container for the handle opener.
    # Inputs: filepath_or_buffer (path, path-like, url, remote url or buffer);
    # encoding (default utf-8); compression (string, mapping with a method key,
    # or none meaning no compression; the string for inference is also
    # accepted); mode (default read); storage_options (default none: headers
    # for http urls, or keyword arguments for the remote filesystem opener).
    # Behavior: stringifies the path; splits the compression argument into
    # method and options and infers the method from the name when requested;
    # warns (runtime warning) and drops compression when a non-binary writable
    # buffer is given; warns (unicode warning) when writing utf-16/utf-32 with
    # bz2 or xz since no byte order mark is written; warns (runtime warning)
    # for zip or tar in append mode. The fsspec mode adds binary unless text or
    # binary was requested. For http/other recognised urls: downloads the whole
    # body into memory, switching compression to gzip if the response says its
    # content is gzip encoded, and marks it to be closed. For remote filesystem
    # urls: rewrites s3a/s3n schemes to s3, opens through the optional fsspec
    # library, and if botocore is installed retries once anonymously (copying
    # storage options, never mutating the caller's) on credential or
    # permission errors; marks the opened file for closing. Plain strings,
    # bytes and memory maps are returned with user expansion applied.
    # Raises a value error if storage options are given for something that is
    # not a url, and a value error if the object has neither read nor write.
    # Returns the container with should_close true only for downloaded or
    # remote-opened files.
    # KNOWN ISSUE: the byte order mark warning interpolates the whole
    # compression mapping into its message instead of the method name.
    # KNOWN ISSUE: the docstring below claims compression defaults to the
    # inference string, but the signature default is none.
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str or pathlib.Path),
                         or buffer

    compression : str or dict, default 'infer'
        For on-the-fly compression of the output data. If 'infer' and
        'filepath_or_buffer' is path-like, then detect compression from the
        following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
        and other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to
        create a reproducible gzip archive:
        ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

    encoding : the encoding to use to decode bytes, default is 'utf-8'
    mode : str, optional

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.


    Returns the dataclass IOArgs.
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)

    # handle compression dict
    compression_method, compression = get_compression_method(compression)
    compression_method = infer_compression(filepath_or_buffer, compression_method)

    # GH21227 internal compression is not used for non-binary handles.
    if compression_method and hasattr(filepath_or_buffer, "write") and "b" not in mode:
        warnings.warn(
            "compression has no effect when passing a non-binary object as input.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )
        compression_method = None

    compression = dict(compression, method=compression_method)

    # bz2 and xz do not write the byte order mark for utf-16 and utf-32
    # print a warning when writing such files
    if (
        "w" in mode
        and compression_method in ["bz2", "xz"]
        and encoding in ["utf-16", "utf-32"]
    ):
        warnings.warn(
            f"{compression} will not write the byte order mark for {encoding}",
            UnicodeWarning,
            stacklevel=find_stack_level(),
        )

    if "a" in mode and compression_method in ["zip", "tar"]:
        # GH56778
        warnings.warn(
            "zip and tar do not support mode 'a' properly. "
            "This combination will result in multiple files with same name "
            "being added to the archive.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )

    # Use binary mode when converting path-like objects to file-like objects (fsspec)
    # except when text mode is explicitly requested. The original mode is returned if
    # fsspec is not used.
    fsspec_mode = mode
    if "t" not in fsspec_mode and "b" not in fsspec_mode:
        fsspec_mode += "b"

    if isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer):
        # TODO: fsspec can also handle HTTP via requests, but leaving this
        # unchanged. using fsspec appears to break the ability to infer if the
        # server responded with gzipped data
        storage_options = storage_options or {}

        # waiting until now for importing to match intended lazy logic of
        # urlopen function defined elsewhere in this module
        import urllib.request

        # assuming storage_options is to be interpreted as headers
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        with urlopen(req_info) as req:
            content_encoding = req.headers.get("Content-Encoding", None)
            if content_encoding == "gzip":
                # Override compression based on Content-Encoding header
                compression = {"method": "gzip"}
            reader = BytesIO(req.read())
        return IOArgs(
            filepath_or_buffer=reader,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )

    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(
            filepath_or_buffer, str
        )  # just to appease mypy for this branch
        # two special-case s3-like protocols; these have special meaning in Hadoop,
        # but are equivalent to just "s3" from fsspec's point of view
        # cc #11071
        if filepath_or_buffer.startswith("s3a://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3a://", "s3://")
        if filepath_or_buffer.startswith("s3n://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3n://", "s3://")
        fsspec = import_optional_dependency("fsspec")

        # If botocore is installed we fallback to reading with anon=True
        # to allow reads from public buckets
        err_types_to_retry_with_anon: list[Any] = []
        try:
            import_optional_dependency("botocore")
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
            )

            err_types_to_retry_with_anon = [
                ClientError,
                NoCredentialsError,
                PermissionError,
            ]
        except ImportError:
            pass

        try:
            open_file = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            )
            file_obj = open_file.open()
        # GH 34626 Reads from Public Buckets without Credentials needs anon=True
        except tuple(err_types_to_retry_with_anon):
            if storage_options is None:
                storage_options = {"anon": True}
            else:
                # don't mutate user input.
                storage_options = dict(storage_options)
                storage_options["anon"] = True
            open_file = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            )
            file_obj = open_file.open()

        return IOArgs(
            filepath_or_buffer=file_obj,
            encoding=encoding,
            compression=compression,
            close_handles=[open_file],
            should_close=True,
            mode=fsspec_mode,
        )
    elif storage_options:
        raise ValueError(
            "storage_options passed with file object or non-fsspec file path"
        )

    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(
            filepath_or_buffer=_expand_user(filepath_or_buffer),
            encoding=encoding,
            compression=compression,
            should_close=False,
            mode=mode,
        )

    # is_file_like requires (read | write) & __iter__ but __iter__ is only
    # needed for read_csv(engine=python)
    if not (
        hasattr(filepath_or_buffer, "read") or hasattr(filepath_or_buffer, "write")
    ):
        msg = f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
        raise ValueError(msg)

    return IOArgs(
        filepath_or_buffer=filepath_or_buffer,
        encoding=encoding,
        compression=compression,
        should_close=False,
        mode=mode,
    )


def file_path_to_url(path: str) -> str:
    # [FUNCTION DOC]
    # Converts an absolute native file system path to a file URL. Input: path
    # string in the platform's format. Returns the URL string built by
    # percent-encoding the path and joining it to the file scheme. The
    # url-request import is deferred because it is slow.
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    # lazify expensive import (~30ms)
    from urllib.request import pathname2url

    return urljoin("file:", pathname2url(path))


extension_to_compression = {
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tar.bz2": "tar",
    ".tar.xz": "tar",
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd",
}
_supported_compressions = set(extension_to_compression.values())


def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]:
    # [FUNCTION DOC]
    # Splits a compression argument into (method, options). Input: a string
    # or none (taken as the method, with empty options) or a mapping that must
    # contain a method key (that entry is removed from a copy and becomes the
    # method; remaining entries are the options). Returns the pair. Raises a
    # value error if a mapping lacks the method key. The input mapping is not
    # mutated.
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
    compression_method: str | None
    if isinstance(compression, Mapping):
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop("method")
        except KeyError as err:
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        compression_args = {}
        compression_method = compression
    return compression_method, compression_args


def infer_compression(
    filepath_or_buffer: FilePath | BaseBuffer, compression: str | None
) -> str | None:
    # [FUNCTION DOC]
    # Determines the compression method to use. Inputs: filepath_or_buffer
    # (path, url or buffer) and compression (a method name, the inference
    # marker, or none). None gives none. For the inference marker: chained
    # url suffixes after a double colon are cut off, the input is stringified
    # (also converting file-like path objects); non-string inputs (buffers)
    # give none; otherwise the lower-cased name is matched against the known
    # extensions in table order and the first matching method is returned, or
    # none if no match. An explicit method must be in the supported set and is
    # returned unchanged. Raises a value error listing valid types otherwise.
    # Note the extension table lists the multi-part tar suffixes first so they
    # match before the single compression suffixes.
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.

    compression : str or dict, default 'infer'
        For on-the-fly compression of the output data. If 'infer' and
        'filepath_or_buffer' is path-like, then detect compression from the
        following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
        and other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to
        create a reproducible gzip archive:
        ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

    Returns
    -------
    string or None

    Raises
    ------
    ValueError on invalid compression specified.
    """
    if compression is None:
        return None

    # Infer compression
    if compression == "infer":
        # Convert all path types (e.g. pathlib.Path) to strings
        if isinstance(filepath_or_buffer, str) and "::" in filepath_or_buffer:
            # chained URLs contain ::
            filepath_or_buffer = filepath_or_buffer.split("::")[0]
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        if not isinstance(filepath_or_buffer, str):
            # Cannot infer compression of a buffer, assume no compression
            return None

        # Infer compression from the filename/URL extension
        for extension, compression in extension_to_compression.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _supported_compressions:
        return compression

    valid = ["infer", None, *sorted(_supported_compressions)]
    msg = (
        f"Unrecognized compression type: {compression}\n"
        f"Valid compression types are {valid}"
    )
    raise ValueError(msg)


def check_parent_directory(path: Path | str) -> None:
    # [FUNCTION DOC]
    # Verifies the parent directory of a file path exists. Input: path or
    # string. Returns nothing; raises an operating-system error naming the
    # missing directory if the parent is not an existing directory.
    """
    Check if parent directory of a file exists, raise OSError if it does not

    Parameters
    ----------
    path: Path or str
        Path to check parent directory of
    """
    parent = Path(path).parent
    if not parent.is_dir():
        raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[False],
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[bytes]: ...
# [FUNCTION DOC]
# Overload stub (binary content, handles of bytes): see the handle opening
# implementation below.


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[True] = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str]: ...
# [FUNCTION DOC]
# Overload stub (text content, handles of strings): see the handle opening
# implementation below.


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str] | IOHandles[bytes]: ...
# [FUNCTION DOC]
# Overload stub (content kind not known statically): see the handle opening
# implementation below.


def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = None,
    compression: CompressionOptions | None = None,
    memory_map: bool = False,
    is_text: bool = True,
    errors: str | None = None,
    storage_options: StorageOptions | None = None,
) -> IOHandles[str] | IOHandles[bytes]:
    # [FUNCTION DOC]
    # Opens a path, url or buffer for reading or writing and returns a managed
    # handle container.
    # Inputs: path_or_buf (path, url or buffer); mode (open mode string);
    # keyword-only: encoding (default utf-8 when none), compression (string,
    # mapping with a method key, inference marker, or none), memory_map
    # (default false; only honoured for readers of delimited text and only if
    # the object has a descriptor or is a path), is_text (default true: the
    # consumer wants strings rather than bytes), errors (default strict;
    # encoding error policy) and storage_options (default none; passed to the
    # path resolver).
    # Behavior: appends binary to the mode when the target is binary and the
    # mode lacks it; validates the encoding and error policy names (lookup
    # errors propagate); resolves the argument; optionally memory maps first;
    # for write modes with a plain path checks the parent directory exists.
    # With compression, removes text mode flags (except for zstd, which is
    # forced to binary) and opens through the matching library: gzip (path or
    # file object), bz2, zip (read mode requires exactly one member, whose
    # stream is used), tar (same one-member rule when reading), xz and zstd
    # (optional library; reader or writer context according to mode).
    # Raises a value error for zero or several archive members or an unknown
    # compression. Without compression, a string path is opened as a file,
    # with the encoding, error policy and an empty newline setting unless in
    # binary mode. Then, for binary consumers of a text stream in read mode, a
    # bytes-producing wrapper is installed; for text consumers of binary
    # sources (compressed, memory mapped or binary-mode), buffers lacking
    # readable/writable/seekable methods are first wrapped, and a text wrapper
    # is installed (flagged as wrapped only when the caller supplied the
    # buffer so it will be detached rather than closed). In read mode raises a
    # type error if the final handle cannot be read.
    # The created handle list is reversed so the newest closes first, and for
    # urls or remote files the resolved buffer and extra handles are appended.
    # Side effects: opens files and archives, may download data.
    # KNOWN ISSUE: the docstring below states a compression default of the
    # inference string, but the actual default is none.
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : str or file handle
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str or None
        Encoding to use.
    compression : str or dict, default 'infer'
        For on-the-fly compression of the output data. If 'infer' and 'path_or_buf'
        is path-like, then detect compression from the following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        Set to ``None`` for no compression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
        and other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression and to
        create a reproducible gzip archive:
        ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

           May be a dict with key 'method' as compression mode
           and other keys as compression options if compression
           mode is 'zip'.

           Passing compression options as keys in dict is
           supported for compression modes 'gzip', 'bz2', 'zstd' and 'zip'.

    memory_map : bool, default False
        See parsers._parser_params for more information. Only used by read_csv.
    is_text : bool, default True
        Whether the type of the content passed to the file/buffer is string or
        bytes. This is not the same as `"b" not in mode`. If a string content is
        passed to a binary file/buffer, a wrapper is inserted.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    storage_options: StorageOptions = None
        Passed to _get_filepath_or_buffer

    Returns the dataclass IOHandles
    """
    # Windows does not default to utf-8. Set to utf-8 for a consistent behavior
    encoding = encoding or "utf-8"

    errors = errors or "strict"

    # read_csv does not know whether the buffer is opened in binary/text mode
    if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
        mode += "b"

    # validate encoding and errors
    codecs.lookup(encoding)
    if isinstance(errors, str):
        codecs.lookup_error(errors)

    # open URLs
    ioargs = _get_filepath_or_buffer(
        path_or_buf,
        encoding=encoding,
        compression=compression,
        mode=mode,
        storage_options=storage_options,
    )

    handle = ioargs.filepath_or_buffer
    handles: list[BaseBuffer]

    # memory mapping needs to be the first step
    # only used for read_csv
    handle, memory_map, handles = _maybe_memory_map(handle, memory_map)

    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop("method")

    # Only for write methods
    if "r" not in mode and is_path:
        check_parent_directory(str(handle))

    if compression:
        if compression != "zstd":
            # compression libraries do not like an explicit text-mode
            ioargs.mode = ioargs.mode.replace("t", "")
        elif compression == "zstd" and "b" not in ioargs.mode:
            # python-zstandard defaults to text mode, but we always expect
            # compression libraries to use binary mode.
            ioargs.mode += "b"

        # GZ Compression
        if compression == "gzip":
            if isinstance(handle, str):
                # error: Incompatible types in assignment (expression has type
                # "GzipFile", variable has type "Union[str, BaseBuffer]")
                handle = gzip.GzipFile(  # type: ignore[assignment]
                    filename=handle,
                    mode=ioargs.mode,
                    **compression_args,
                )
            else:
                handle = gzip.GzipFile(
                    # No overload variant of "GzipFile" matches argument types
                    # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                    fileobj=handle,  # type: ignore[call-overload]
                    mode=ioargs.mode,
                    **compression_args,
                )

        # BZ Compression
        elif compression == "bz2":
            import bz2

            # Overload of "BZ2File" to handle pickle protocol 5
            # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
            handle = bz2.BZ2File(  # type: ignore[call-overload]
                handle,
                mode=ioargs.mode,
                **compression_args,
            )

        # ZIP Compression
        elif compression == "zip":
            # error: Argument 1 to "_BytesZipFile" has incompatible type
            # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
            # ReadBuffer[bytes], WriteBuffer[bytes]]"
            handle = _BytesZipFile(
                handle,  # type: ignore[arg-type]
                ioargs.mode,
                **compression_args,
            )
            if handle.buffer.mode == "r":
                handles.append(handle)
                zip_names = handle.buffer.namelist()
                if len(zip_names) == 1:
                    handle = handle.buffer.open(zip_names.pop())
                elif not zip_names:
                    raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in ZIP file. "
                        f"Only one file per ZIP: {zip_names}"
                    )

        # TAR Encoding
        elif compression == "tar":
            compression_args.setdefault("mode", ioargs.mode)
            if isinstance(handle, str):
                handle = _BytesTarFile(name=handle, **compression_args)
            else:
                # error: Argument "fileobj" to "_BytesTarFile" has incompatible
                # type "BaseBuffer"; expected "Union[ReadBuffer[bytes],
                # WriteBuffer[bytes], None]"
                handle = _BytesTarFile(
                    fileobj=handle,  # type: ignore[arg-type]
                    **compression_args,
                )
            assert isinstance(handle, _BytesTarFile)
            if "r" in handle.buffer.mode:
                handles.append(handle)
                files = handle.buffer.getnames()
                if len(files) == 1:
                    file = handle.buffer.extractfile(files[0])
                    assert file is not None
                    handle = file
                elif not files:
                    raise ValueError(f"Zero files found in TAR archive {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in TAR archive. "
                        f"Only one file per TAR archive: {files}"
                    )

        # XZ Compression
        elif compression == "xz":
            # error: Argument 1 to "LZMAFile" has incompatible type "Union[str,
            # BaseBuffer]"; expected "Optional[Union[Union[str, bytes, PathLike[str],
            # PathLike[bytes]], IO[bytes]], None]"
            import lzma

            handle = lzma.LZMAFile(
                handle,  # type: ignore[arg-type]
                ioargs.mode,
                **compression_args,
            )

        # Zstd Compression
        elif compression == "zstd":
            zstd = import_optional_dependency("zstandard")
            if "r" in ioargs.mode:
                open_args = {"dctx": zstd.ZstdDecompressor(**compression_args)}
            else:
                open_args = {"cctx": zstd.ZstdCompressor(**compression_args)}
            handle = zstd.open(
                handle,
                mode=ioargs.mode,
                **open_args,
            )

        # Unrecognized Compression
        else:
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        assert not isinstance(handle, str)
        handles.append(handle)

    elif isinstance(handle, str):
        # Check whether the filename is to be opened in binary mode.
        # Binary mode does not support 'encoding' and 'newline'.
        if ioargs.encoding and "b" not in ioargs.mode:
            # Encoding
            handle = open(
                handle,
                ioargs.mode,
                encoding=ioargs.encoding,
                errors=errors,
                newline="",
            )
        else:
            # Binary mode
            handle = open(handle, ioargs.mode)
        handles.append(handle)

    # Convert BytesIO or file objects passed with an encoding
    is_wrapped = False
    if not is_text and ioargs.mode == "rb" and isinstance(handle, TextIOBase):
        # not added to handles as it does not open/buffer resources
        handle = _BytesIOWrapper(
            handle,
            encoding=ioargs.encoding,
        )
    elif is_text and (
        compression or memory_map or _is_binary_mode(handle, ioargs.mode)
    ):
        if (
            not hasattr(handle, "readable")
            or not hasattr(handle, "writable")
            or not hasattr(handle, "seekable")
        ):
            handle = _IOWrapper(handle)
        # error: Value of type variable "_BufferT_co" of "TextIOWrapper" cannot
        # be "_IOWrapper | BaseBuffer" [type-var]
        handle = TextIOWrapper(
            handle,  # type: ignore[type-var]
            encoding=ioargs.encoding,
            errors=errors,
            newline="",
        )
        handles.append(handle)
        # only marked as wrapped when the caller provided a handle
        is_wrapped = not (
            isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close
        )

    if "r" in ioargs.mode and not hasattr(handle, "read"):
        raise TypeError(
            "Expected file path name or file-like object, "
            f"got {type(ioargs.filepath_or_buffer)} type"
        )

    handles.reverse()  # close the most recently added buffer first
    if ioargs.should_close:
        assert not isinstance(ioargs.filepath_or_buffer, str)
        handles.append(ioargs.filepath_or_buffer)
        handles.extend(ioargs.close_handles)

    return IOHandles(
        # error: Argument "handle" to "IOHandles" has incompatible type
        # "Union[TextIOWrapper, GzipFile, BaseBuffer, typing.IO[bytes],
        # typing.IO[Any]]"; expected "pandas._typing.IO[Any]"
        handle=handle,  # type: ignore[arg-type]
        # error: Argument "created_handles" to "IOHandles" has incompatible type
        # "List[BaseBuffer]"; expected "List[Union[IO[bytes], IO[str]]]"
        created_handles=handles,  # type: ignore[arg-type]
        is_wrapped=is_wrapped,
        compression=ioargs.compression,
    )


class _BufferedWriter(BytesIO, ABC):
    # [CLASS DOC]
    # HELPER: abstract in-memory byte buffer for archive formats that cannot
    # accept several write calls. Everything written is held in memory and
    # flushed into the real archive object (the buffer attribute, whose class
    # attribute default is an empty in-memory buffer replaced by subclasses)
    # in one step on close. Subclasses implement write_to_buffer. Inherits the
    # in-memory byte stream and the abstract base class machinery.
    # Invariant: after close, the archive is closed, whether or not anything
    # was written.
    """
    Some objects do not support multiple .write() calls (TarFile and ZipFile).
    This wrapper writes to the underlying buffer on close.
    """

    buffer = BytesIO()

    @abstractmethod
    def write_to_buffer(self) -> None: ...
    # [FUNCTION DOC]
    # Abstract hook: subclasses add the accumulated bytes as a single member
    # of the underlying archive. Takes no arguments and returns nothing.

    def close(self) -> None:
        # [FUNCTION DOC]
        # HELPER: closes the buffer exactly once. If already closed, does
        # nothing. If any bytes were written, rewinds and, inside a context
        # on the archive object, calls the subclass write hook (so the archive
        # is closed afterwards). If nothing was written, simply closes the
        # archive. Finally closes the in-memory stream itself.
        if self.closed:
            # already closed
            return
        if self.getbuffer().nbytes:
            # write to buffer
            self.seek(0)
            with self.buffer:
                self.write_to_buffer()
        else:
            self.buffer.close()
        super().close()


class _BytesTarFile(_BufferedWriter):
    # [CLASS DOC]
    # HELPER: tar archive adapter used by the handle opener. Wraps a tar file
    # object (attribute buffer) opened from a name or a file object; written
    # bytes accumulate in memory and are added on close as one member.
    # Attributes: archive_name (member name given by the caller, may be none)
    # and name (path of the archive, may be none).
    def __init__(
        self,
        name: str | None = None,
        mode: Literal["r", "a", "w", "x"] = "r",
        fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None = None,
        archive_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: opens the tar archive. name is the archive path (optional);
        # mode (default read) is one of read, append, write, exclusive-create
        # and is normalised by the mode extender; fileobj is an alternative
        # underlying file object; archive_name optionally names the member
        # written; extra keyword arguments are passed to the tar open routine.
        # Side effect: opens the archive immediately.
        super().__init__()
        self.archive_name = archive_name
        self.name = name
        #  error: No overload variant of "open" of "TarFile" matches argument
        # types "str | None", "str", "ReadBuffer[bytes] | WriteBuffer[bytes] | None",
        # "dict[str, Any]"
        # error: Incompatible types in assignment (expression has type "TarFile",
        #  base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: tarfile.TarFile = tarfile.TarFile.open(  # type: ignore[call-overload, assignment]
            name=name,
            mode=self.extend_mode(mode),
            fileobj=fileobj,
            **kwargs,
        )

    def extend_mode(self, mode: str) -> str:
        # [FUNCTION DOC]
        # HELPER: normalises a mode for the tar open routine. Removes binary
        # flags; read/append/other modes are returned as is; for write mode,
        # if the archive name ends in a gzip, xz or bz2 suffix, the matching
        # compressed-write mode (mode, colon, suffix without dot) is returned,
        # otherwise plain write.
        mode = mode.replace("b", "")
        if mode != "w":
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in (".gz", ".xz", ".bz2"):
                mode = f"{mode}:{suffix[1:]}"
        return mode

    def infer_filename(self) -> str | None:
        # [FUNCTION DOC]
        # HELPER: chooses a member name from the archive name when none was
        # given. Returns none if the archive has no name; strips a trailing tar
        # extension; otherwise returns the final path component.
        # KNOWN ISSUE: the branch intended to strip double extensions (tar
        # plus gzip, bz2 or xz) can never match because the path suffix is
        # only the last extension, so such names keep their inner tar
        # extension; the docstring also says zip instead of tar.
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
        if self.name is None:
            return None

        filename = Path(self.name)
        if filename.suffix == ".tar":
            return filename.with_suffix("").name
        elif filename.suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            return filename.with_suffix("").with_suffix("").name
        return filename.name

    def write_to_buffer(self) -> None:
        # [FUNCTION DOC]
        # HELPER: adds the accumulated bytes as one member of the tar archive,
        # named by the explicit archive name, else the inferred name, else the
        # literal tar (the archive needs a non-empty name). Member size is the
        # number of accumulated bytes.
        # TarFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "tar"
        tarinfo = tarfile.TarInfo(name=archive_name)
        tarinfo.size = len(self.getvalue())
        self.buffer.addfile(tarinfo, self)


class _BytesZipFile(_BufferedWriter):
    # [CLASS DOC]
    # HELPER: zip archive adapter used by the handle opener. Wraps a zip file
    # object (attribute buffer); written bytes accumulate in memory and are
    # stored on close as a single member. Attribute archive_name is the
    # caller-chosen member name (may be none).
    def __init__(
        self,
        file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
        mode: str,
        archive_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: opens the zip archive. file is a path or buffer; mode is the
        # open mode with binary flags removed; archive_name optionally names
        # the member written; extra keyword arguments go to the zip file
        # constructor, with deflate compression as the default method.
        # Side effect: opens the archive immediately.
        super().__init__()
        mode = mode.replace("b", "")
        self.archive_name = archive_name

        kwargs.setdefault("compression", zipfile.ZIP_DEFLATED)
        # error: No overload variant of "ZipFile" matches argument types
        # "str | PathLike[str] | ReadBuffer[bytes] | WriteBuffer[bytes]",
        # "str", "dict[str, Any]"
        # error: Incompatible types in assignment (expression has type "ZipFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[call-overload, assignment]
            file, mode, **kwargs
        )

    def infer_filename(self) -> str | None:
        # [FUNCTION DOC]
        # HELPER: chooses a member name from the archive's file name when none
        # was given: strips a trailing zip extension, otherwise returns the
        # final path component; returns none when the archive has no path-like
        # or string file name (for example a buffer).
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.zip, because that causes confusion (GH39465).
        """
        if isinstance(self.buffer.filename, (os.PathLike, str)):
            filename = Path(self.buffer.filename)
            if filename.suffix == ".zip":
                return filename.with_suffix("").name
            return filename.name
        return None

    def write_to_buffer(self) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the accumulated bytes as one member of the zip
        # archive, named by the explicit archive name, else the inferred name,
        # else the literal zip (the archive needs a non-empty name).
        # ZipFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "zip"
        self.buffer.writestr(archive_name, self.getvalue())


class _IOWrapper:
    # [CLASS DOC]
    # HELPER: adapter that gives a buffer the readable, writable and seekable
    # query methods required by the standard text wrapper. Attribute buffer is
    # the wrapped object; any other attribute access is forwarded to it. Each
    # query method defers to the buffer's own method if present and otherwise
    # answers true (assume capable). Not a subclass of any io base class.
    # TextIOWrapper is overly strict: it request that the buffer has seekable, readable,
    # and writable. If we have a read-only buffer, we shouldn't need writable and vice
    # versa. Some buffers, are seek/read/writ-able but they do not have the "-able"
    # methods, e.g., tempfile.SpooledTemporaryFile.
    # If a buffer does not have the above "-able" methods, we simple assume they are
    # seek/read/writ-able.
    def __init__(self, buffer: BaseBuffer) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the buffer to wrap; no validation.
        self.buffer = buffer

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC]
        # HELPER: forwards lookups of attributes not found on the wrapper to
        # the wrapped buffer; raises an attribute error as the buffer does.
        return getattr(self.buffer, name)

    def readable(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns the buffer's own readable answer if it has one,
        # otherwise true.
        if hasattr(self.buffer, "readable"):
            return self.buffer.readable()
        return True

    def seekable(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns the buffer's own seekable answer if it has one,
        # otherwise true.
        if hasattr(self.buffer, "seekable"):
            return self.buffer.seekable()
        return True

    def writable(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns the buffer's own writable answer if it has one,
        # otherwise true.
        if hasattr(self.buffer, "writable"):
            return self.buffer.writable()
        return True


class _BytesIOWrapper:
    # [CLASS DOC]
    # HELPER: read-only adapter presenting a text stream as a byte stream
    # (needed by the arrow csv reader). Attributes: buffer (the wrapped text
    # stream), encoding (default utf-8, used to encode what is read) and
    # overflow (bytes produced by an earlier read that exceeded the requested
    # size, returned first on the next read). Unknown attributes are forwarded
    # to the wrapped stream. Not a subclass of any io base class.
    # Wrapper that wraps a StringIO buffer and reads bytes from it
    # Created for compat with pyarrow read_csv
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = "utf-8") -> None:
        # [FUNCTION DOC]
        # HELPER: stores the text stream and encoding and starts with an empty
        # overflow.
        self.buffer = buffer
        self.encoding = encoding
        # Because a character can be represented by more than 1 byte,
        # it is possible that reading will produce more bytes than n
        # We store the extra bytes in this overflow variable, and append the
        # overflow to the front of the bytestring the next time reading is performed
        self.overflow = b""

    def __getattr__(self, attr: str) -> Any:
        # [FUNCTION DOC]
        # HELPER: forwards lookups of attributes not defined on the wrapper to
        # the wrapped text stream.
        return getattr(self.buffer, attr)

    def read(self, n: int | None = -1) -> bytes:
        # [FUNCTION DOC]
        # HELPER: reads up to n characters from the text stream (n of none or
        # negative reads everything), encodes them, prepends the saved
        # overflow, and returns at most n bytes (everything if n is none,
        # negative, or not smaller than the available bytes). Excess bytes
        # (because characters can encode to several bytes) are stored as the
        # new overflow; otherwise overflow is cleared.
        # KNOWN ISSUE: n counts characters when reading the underlying text
        # stream but bytes when slicing the result, so the returned length
        # semantic mixes the two units.
        assert self.buffer is not None
        bytestring = self.buffer.read(n).encode(self.encoding)
        # When n=-1/n greater than remaining bytes: Read entire file/rest of file
        combined_bytestring = self.overflow + bytestring
        if n is None or n < 0 or n >= len(combined_bytestring):
            self.overflow = b""
            return combined_bytestring
        else:
            to_return = combined_bytestring[:n]
            self.overflow = combined_bytestring[n:]
            return to_return


def _maybe_memory_map(
    handle: str | BaseBuffer, memory_map: bool
) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
    # [FUNCTION DOC]
    # HELPER: optionally memory maps a path or buffer for reading. Inputs:
    # handle (path string or buffer) and memory_map (requested flag, which is
    # turned off if the handle is neither a path nor has a file descriptor).
    # Returns a triple of (handle to use, effective memory-map flag, list of
    # handles the caller must close). When not mapping, returns the inputs with
    # an empty list. When mapping: opens a path in binary read mode, maps the
    # whole file read-only, wraps the map with the readable/writable/seekable
    # adapter, and always closes the temporary file object it opened (the
    # mapping survives); the returned close list holds only the wrapper.
    """Try to memory map file/buffer."""
    handles: list[BaseBuffer] = []
    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
    if not memory_map:
        return handle, memory_map, handles

    # mmap used by only read_csv
    handle = cast(ReadCsvBuffer, handle)

    # need to open the file first
    if isinstance(handle, str):
        handle = open(handle, "rb")
        handles.append(handle)

    try:
        # open mmap and adds *-able
        # error: Argument 1 to "_IOWrapper" has incompatible type "mmap";
        # expected "BaseBuffer"
        wrapped = _IOWrapper(
            mmap.mmap(
                handle.fileno(),
                0,
                access=mmap.ACCESS_READ,  # type: ignore[arg-type]
            )
        )
    finally:
        for handle in reversed(handles):
            # error: "BaseBuffer" has no attribute "close"
            handle.close()  # type: ignore[attr-defined]

    return wrapped, memory_map, [wrapped]


def file_exists(filepath_or_buffer: FilePath | BaseBuffer) -> bool:
    # [FUNCTION DOC]
    # Returns whether the input names an existing path. The input is
    # stringified first; non-string inputs (buffers) give false. Type or value
    # errors from the file system check (for example over-long paths) are
    # swallowed and give false.
    """Test whether file exists."""
    exists = False
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if not isinstance(filepath_or_buffer, str):
        return exists
    try:
        exists = os.path.exists(filepath_or_buffer)
        # gh-5874: if the filepath is too long will raise here
    except (TypeError, ValueError):
        pass
    return exists


def _is_binary_mode(handle: FilePath | BaseBuffer, mode: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: decides whether a handle expects bytes. Inputs: handle and the
    # requested mode string. If the mode explicitly contains text or binary
    # flag, the answer is whether it contains the binary flag. Codec stream
    # reader/writer classes are text even though their mode may say binary
    # (answer false). Otherwise true if the handle is an instance of a known
    # binary io class or its own mode attribute (falling back to the given
    # mode) contains the binary flag.
    """Whether the handle is opened in binary mode"""
    # specified by user
    if "t" in mode or "b" in mode:
        return "b" in mode

    # exceptions
    text_classes = (
        # classes that expect string but have 'b' in mode
        codecs.StreamWriter,
        codecs.StreamReader,
        codecs.StreamReaderWriter,
    )
    if issubclass(type(handle), text_classes):
        return False

    return isinstance(handle, _get_binary_io_classes()) or "b" in getattr(
        handle, "mode", mode
    )


@functools.lru_cache
def _get_binary_io_classes() -> tuple[type, ...]:
    # [FUNCTION DOC]
    # HELPER: returns (and caches for the process) the tuple of classes whose
    # instances expect bytes: the buffered and raw io base classes, plus, if
    # the optional zstandard package is installed, the class of its stream
    # reader (obtained by creating a decompressor reader over empty input,
    # because the class is not exported). Missing package is ignored.
    """IO classes that that expect bytes"""
    binary_classes: tuple[type, ...] = (BufferedIOBase, RawIOBase)

    # python-zstandard doesn't use any of the builtin base classes; instead we
    # have to use the `zstd.ZstdDecompressionReader` class for isinstance checks.
    # Unfortunately `zstd.ZstdDecompressionReader` isn't exposed by python-zstandard
    # so we have to get it from a `zstd.ZstdDecompressor` instance.
    # See also https://github.com/indygreg/python-zstandard/pull/165.
    zstd = import_optional_dependency("zstandard", errors="ignore")
    if zstd is not None:
        with zstd.ZstdDecompressor().stream_reader(b"") as reader:
            binary_classes += (type(reader),)

    return binary_classes


def is_potential_multi_index(
    columns: Sequence[Hashable] | MultiIndex,
    index_col: bool | Sequence[int] | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Decides whether a list of column labels could form a multi-index.
    # Inputs: columns (sequence of labels or an existing multi-index) and
    # index_col (none, a boolean, or a sequence of positions/labels naming
    # index columns, default none; booleans and none mean no index columns).
    # Returns true only if the columns are non-empty, not already a
    # multi-index, and every label not listed in index_col is a tuple.
    # KNOWN ISSUE: the index_col entries are compared against the labels
    # themselves, although the parameter is documented as column positions, so
    # positional integers only work when labels are equal to those integers.
    """
    Check whether or not the `columns` parameter
    could be converted into a MultiIndex.

    Parameters
    ----------
    columns : array-like
        Object which may or may not be convertible into a MultiIndex
    index_col : None, bool or list, optional
        Column or columns to use as the (possibly hierarchical) index

    Returns
    -------
    bool : Whether or not columns could become a MultiIndex
    """
    if index_col is None or isinstance(index_col, bool):
        index_columns = set()
    else:
        index_columns = set(index_col)

    return bool(
        len(columns)
        and not isinstance(columns, ABCMultiIndex)
        and all(isinstance(c, tuple) for c in columns if c not in index_columns)
    )


def dedup_names(
    names: Sequence[Hashable], is_potential_multiindex: bool
) -> Sequence[Hashable]:
    # [FUNCTION DOC]
    # Makes column names unique. Inputs: names (sequence of hashable labels)
    # and is_potential_multiindex (true when labels are tuples, so the suffix
    # is appended to the last tuple element instead). Returns a new list in
    # which the second and later occurrences of a name get a period and a
    # running counter appended (counter starts at one); the suffixed name is
    # itself checked for collisions with existing names and bumped until
    # unique. The input is not modified. For tuple mode, labels must be tuples
    # (assertion error otherwise).
    """
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    but a custom pattern may be supported in the future.

    Examples
    --------
    >>> dedup_names(["x", "y", "x", "x"], is_potential_multiindex=False)
    ['x', 'y', 'x.1', 'x.2']
    """
    names = list(names)  # so we can index
    counts: DefaultDict[Hashable, int] = defaultdict(int)

    for i, col in enumerate(names):
        cur_count = counts[col]

        while cur_count > 0:
            counts[col] = cur_count + 1

            if is_potential_multiindex:
                # for mypy
                assert isinstance(col, tuple)
                col = (*col[:-1], f"{col[-1]}.{cur_count}")
            else:
                col = f"{col}.{cur_count}"
            cur_count = counts[col]

        names[i] = col
        counts[col] = cur_count + 1

    return names
