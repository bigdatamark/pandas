"""
Read SAS sas7bdat or xport files.
"""

# [MODULE DOC]
# Public entry point for reading SAS files. Defines the abstract reader base
# class shared by the two concrete readers and the read_sas function, which
# picks a reader by explicit format or by file name.
# Relationships: the concrete readers live in the xport and sas7bdat modules and
# are imported lazily inside read_sas; path normalization comes from the io
# common module.
# Import-time side effects: none.
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Iterator
from typing import (
    TYPE_CHECKING,
    Self,
    overload,
)

from pandas.util._decorators import set_module

from pandas.io.common import stringify_path

if TYPE_CHECKING:
    from collections.abc import Hashable
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
    )

    from pandas import DataFrame


@set_module("pandas.api.typing")
class SASReader(Iterator["DataFrame"], ABC):
    # [CLASS DOC]
    # Abstract base for the concrete SAS readers (the xport reader and the
    # sas7bdat reader). It is an iterator of data frames and an abstract base
    # class, and also a context manager that closes the reader on exit.
    # Subclasses must implement read (returns a data frame, optionally limited
    # to a number of rows) and close (releases the underlying file handles).
    # Holds no state of its own.
    """
    Abstract class for XportReader and SAS7BDATReader.
    """

    @abstractmethod
    def read(self, nrows: int | None = None) -> DataFrame:
        # [FUNCTION DOC]
        # Abstract method: subclasses return up to nrows rows (all remaining
        # rows when none) as a data frame.
        ...

    @abstractmethod
    def close(self) -> None:
        # [FUNCTION DOC]
        # Abstract method: subclasses release the underlying file handles.
        ...

    def __enter__(self) -> Self:
        # [FUNCTION DOC]
        # Context manager entry; returns the reader itself.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC]
        # Context manager exit; closes the reader. The exception information
        # arguments are ignored, and exceptions are not suppressed.
        self.close()


@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
) -> SASReader:
    # [FUNCTION DOC]
    # Typing overload for a chunk size given: returns a reader. See the
    # implementation below.
    ...


@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: None = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
) -> DataFrame | SASReader:
    # [FUNCTION DOC]
    # Typing overload for no chunk size: returns a data frame or a reader. See
    # the implementation below.
    ...


@set_module("pandas")
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> DataFrame | SASReader:
    # [FUNCTION DOC]
    # Reads a SAS XPORT or sas7bdat file.
    # Parameters: filepath_or_buffer is a path, URL or binary file-like object;
    # format is "xport" or "sas7bdat" (case-insensitive) or none to infer it
    # from the file name; index is the column to use as index (default none);
    # encoding decodes text (none leaves strings as raw bytes); chunksize is the
    # number of rows per chunk (default none); iterator (default false) asks
    # for a reader instead of a frame; compression defaults to inference from
    # the file name.
    # Behavior: when no format is given the argument is stringified and must be
    # a string; names containing the transport extension select xport, names
    # containing the sas7bdat extension select sas7bdat (checked in that
    # order, case-insensitively). The matching reader is constructed with the
    # index, encoding, chunksize and compression options.
    # Returns: the reader itself if iterator is true or chunksize is truthy;
    # otherwise the whole file is read inside a context manager (closing the
    # reader) and the resulting data frame is returned.
    # Raises: a value error if the format is none and the input is not a path
    # string, if the format cannot be inferred from the name, or if the format
    # name is not recognized.
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be
        a URL. Valid URL schemes include http, ftp, s3, and file. For file
        URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.sas7bdat``.
    format : str {'xport', 'sas7bdat'} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.
    compression : str or dict, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and
        'filepath_or_buffer' is path-like, then detect compression from the
        following extensions: '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar',
        '.tar.gz', '.tar.xz' or '.tar.bz2' (otherwise no compression).
        Set to ``None`` for no decompression.
        Can also be a dict with key ``'method'`` set to one of {``'zip'``,
        ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and other
        key-value pairs are forwarded to ``zipfile.ZipFile``,
        ``gzip.GzipFile``, ``bz2.BZ2File``, ``zstandard.ZstdCompressor``,
        ``lzma.LZMAFile`` or ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for faster compression
        and to create a reproducible gzip archive:
        ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

    Returns
    -------
    DataFrame, SAS7BDATReader, or XportReader
        DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
        or XportReader, file format is inferred from file extension.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into a DataFrame.
    read_excel : Read an Excel file into a pandas DataFrame.
    read_spss : Read an SPSS file into a pandas DataFrame.
    read_orc : Load an ORC object into a pandas DataFrame.
    read_feather : Load a feather-format object into a pandas DataFrame.

    Examples
    --------
    >>> df = pd.read_sas("sas_data.sas7bdat")  # doctest: +SKIP
    """
    if format is None:
        buffer_error_msg = (
            "If this is a buffer object rather "
            "than a string name, you must specify a format string"
        )
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if not isinstance(filepath_or_buffer, str):
            raise ValueError(buffer_error_msg)
        fname = filepath_or_buffer.lower()
        if ".xpt" in fname:
            format = "xport"
        elif ".sas7bdat" in fname:
            format = "sas7bdat"
        else:
            raise ValueError(
                f"unable to infer format of SAS file from filename: {fname!r}"
            )

    reader: SASReader
    if format.lower() == "xport":
        from pandas.io.sas.sas_xport import XportReader

        reader = XportReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    elif format.lower() == "sas7bdat":
        from pandas.io.sas.sas7bdat import SAS7BDATReader

        reader = SAS7BDATReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    else:
        raise ValueError("unknown SAS format")

    if iterator or chunksize:
        return reader

    with reader:
        return reader.read()
