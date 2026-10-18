# [MODULE DOC]
# HELPER: test support module with file-input/output helpers. Public contents:
# round_trip_pickle, round_trip_pathlib and write_to_compressed (re-exported by
# the testing package). No import-time side effects beyond imports.
from __future__ import annotations

import gzip
import io
import tarfile
from typing import (
    TYPE_CHECKING,
    Any,
)
import zipfile

from pandas.compat._optional import import_optional_dependency

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pandas import (
        DataFrame,
        Series,
    )

# ------------------------------------------------------------------
# File-IO


def round_trip_pickle(obj: Any, tmp_path: Path) -> DataFrame | Series:
    # [FUNCTION DOC]
    # HELPER: test helper. Writes the object to the given path with the
    # library's pickle writer and reads it back with the pickle reader.
    # Parameters: obj (object to write) and tmp_path (file path to use,
    # typically a temporary path from the test framework). Returns the object
    # read back. Side effect: creates or overwrites the file.
    # KNOWN ISSUE: the existing docstring documents a parameter named path
    # with a default of none, but the actual parameter is named tmp_path and is
    # required.
    """
    Pickle an object and then read it again.

    Parameters
    ----------
    obj : any object
        The object to pickle and then re-read.
    path : str, path object or file-like object, default None
        The path where the pickled object is written and then read.

    Returns
    -------
    pandas object
        The original object that was pickled and then re-read.
    """
    pd.to_pickle(obj, tmp_path)
    return pd.read_pickle(tmp_path)


def round_trip_pathlib(writer, reader, tmp_path: Path):
    # [FUNCTION DOC]
    # HELPER: test helper. Calls the writer callable with the path, then calls
    # the reader callable with the same path and returns what the reader
    # returns. Parameters: writer (callable taking the path, usually a bound
    # writing method), reader (callable taking the path) and tmp_path (file
    # path object). Side effect: whatever files the writer creates.
    # KNOWN ISSUE: the existing docstring documents a parameter named path of
    # string type with a default of none, but the actual parameter is tmp_path
    # and is required.
    """
    Write an object to file specified by a pathlib.Path and read it back

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
    writer(tmp_path)
    obj = reader(tmp_path)
    return obj


def write_to_compressed(compression, path: str, data, dest: str = "test") -> None:
    # [FUNCTION DOC]
    # HELPER: test helper that writes data into a compressed file.
    # Parameters: compression (one of gzip, bz2, zip, xz, zstd or tar), path
    # (output file path), data (bytes for most formats; text or bytes for zip;
    # bytes for tar) and dest (name of the single member inside a zip or tar
    # archive, default test; unused for the stream formats).
    # Behavior: picks the writer class for the format (the zstd one comes from
    # the optional zstandard dependency), opens the path in binary write mode
    # (plain write mode for zip and tar), and for zip stores the data under the
    # member name, for tar adds a member of the data length read from an
    # in-memory buffer, and for the others writes the data directly.
    # Raises a value error for an unrecognized compression name; raises the
    # optional-dependency import error if the zstandard package is missing.
    # KNOWN ISSUE: the existing docstring omits the tar option that the code
    # supports, and says data is text although most formats require bytes.
    # KNOWN ISSUE: the local variable holding the in-memory buffer in the tar
    # branch is named like the builtin bytes type, shadowing it.
    """
    Write data to a compressed file.

    Parameters
    ----------
    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd'}
        The compression type to use.
    path : str
        The file path to write the data.
    data : str
        The data to write.
    dest : str, default "test"
        The destination file (for ZIP only)

    Raises
    ------
    ValueError : An invalid compression value was passed in.
    """
    args: tuple[Any, ...] = (data,)
    mode = "wb"
    method = "write"
    compress_method: Callable

    if compression == "zip":
        compress_method = zipfile.ZipFile
        mode = "w"
        args = (dest, data)
        method = "writestr"
    elif compression == "tar":
        compress_method = tarfile.TarFile
        mode = "w"
        file = tarfile.TarInfo(name=dest)
        bytes = io.BytesIO(data)
        file.size = len(data)
        args = (file, bytes)
        method = "addfile"
    elif compression == "gzip":
        compress_method = gzip.GzipFile
    elif compression == "bz2":
        import bz2

        compress_method = bz2.BZ2File
    elif compression == "zstd":
        compress_method = import_optional_dependency("zstandard").open
    elif compression == "xz":
        import lzma

        compress_method = lzma.LZMAFile
    else:
        raise ValueError(f"Unrecognized compression type: {compression}")

    # error: No overload variant of "ZipFile" matches argument types "str", "str"
    # error: No overload variant of "BZ2File" matches argument types "str", "str"
    # error: Argument "mode" to "TarFile" has incompatible type "str";
    #  expected "Literal['r', 'a', 'w', 'x']
    with compress_method(path, mode=mode) as f:  # type: ignore[call-overload, arg-type]
        getattr(f, method)(*args)
