"""parquet compat"""

# [MODULE DOC]
# Parquet reading and writing with two interchangeable engines (pyarrow and
# fastparquet), both optional dependencies imported lazily.
# Public contents: to_parquet (used by the DataFrame writer method) and
# read_parquet (exported as a top-level pandas function). Internal: an engine
# selector, a path/handle preparation helper, an abstract base implementation
# class, and the pyarrow and fastparquet implementation classes.
# Relationships: engine "auto" consults the io.parquet.engine option and tries
# pyarrow then fastparquet; file opening goes through the shared io common
# helpers; arrow results are converted to frames via the shared arrow helper.
# No import-time side effects beyond imports.
from __future__ import annotations

import io
import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)
from warnings import (
    catch_warnings,
    filterwarnings,
)

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    Pandas4Warning,
)
from pandas.util._decorators import set_module
from pandas.util._validators import check_dtype_backend

from pandas import (
    DataFrame,
    get_option,
)

from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import (
    IOHandles,
    get_handle,
    is_fsspec_url,
    is_url,
    stringify_path,
)

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ParquetCompressionOptions,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )


def get_engine(engine: str) -> BaseImpl:
    # [FUNCTION DOC]
    # Selects and instantiates a parquet engine implementation.
    # Input engine is auto, pyarrow or fastparquet. For auto, the io.parquet.engine
    # option is read first; if that is still auto, tries the pyarrow
    # implementation then the fastparquet one and returns the first whose
    # dependency imports, otherwise raises an import error that lists each
    # failure message. Explicit names build that implementation directly (an
    # import error propagates if its dependency is missing). Any other value
    # raises a value error. Returns the new implementation instance.
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.parquet.engine")

    if engine == "auto":
        # try engines in this order
        engine_classes = [PyArrowImpl, FastParquetImpl]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'pyarrow', 'fastparquet'.\n"
            "A suitable version of "
            "pyarrow or fastparquet is required for parquet "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "pyarrow":
        return PyArrowImpl()
    elif engine == "fastparquet":
        return FastParquetImpl()

    raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")


def _get_path_or_handle(
    path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
    fs: Any,
    storage_options: StorageOptions | None = None,
    mode: str = "rb",
    is_dir: bool = False,
) -> tuple[
    FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any
]:
    # [FUNCTION DOC]
    # HELPER: normalizes a path/buffer and filesystem for the pyarrow engine.
    # Inputs: path is a path, URL or binary buffer; fs is an optional pyarrow
    # or fsspec filesystem; storage_options are remote connection options; mode
    # is the open mode (read-binary default); is_dir says the target is known to
    # be a directory (e.g. partitioned writes).
    # Returns a triple: the path or handle to hand to pyarrow, an I/O handles
    # object that the caller must close (or None), and the filesystem to use (or
    # None).
    # Behavior: converts path-like to string. A given filesystem must be a
    # pyarrow or fsspec one (else value error); a pyarrow filesystem combined
    # with storage options is a not-implemented error. For fsspec-style URLs
    # without a filesystem: if no storage options, tries a pyarrow filesystem
    # from the URI (type/arrow-invalid errors ignored); if still none, resolves
    # an fsspec filesystem and stripped path using the storage options.
    # Otherwise, storage options with a non-URL or non-read use raise a value
    # error. Finally, when there is no filesystem, the target is not known to be
    # a directory, and it is a string that is not an existing directory, it is
    # opened through the shared handle opener (binary) so compression/URL
    # handling applies, and the opened stream replaces the path.
    # KNOWN ISSUE: the URI-resolution step passes the original path argument
    # rather than the stringified path, so path-like inputs reaching that
    # step are not stringified there.
    """File handling for PyArrow."""
    path_or_handle = stringify_path(path)
    if fs is not None:
        pa_fs = import_optional_dependency("pyarrow.fs", errors="ignore")
        fsspec = import_optional_dependency("fsspec", errors="ignore")
        if pa_fs is not None and isinstance(fs, pa_fs.FileSystem):
            if storage_options:
                raise NotImplementedError(
                    "storage_options not supported with a pyarrow FileSystem."
                )
        elif fsspec is not None and isinstance(fs, fsspec.spec.AbstractFileSystem):
            pass
        else:
            raise ValueError(
                f"filesystem must be a pyarrow or fsspec FileSystem, "
                f"not a {type(fs).__name__}"
            )
    if is_fsspec_url(path_or_handle) and fs is None:
        if storage_options is None:
            pa = import_optional_dependency("pyarrow")
            pa_fs = import_optional_dependency("pyarrow.fs")

            try:
                fs, path_or_handle = pa_fs.FileSystem.from_uri(path)
            except (TypeError, pa.ArrowInvalid):
                pass
        if fs is None:
            fsspec = import_optional_dependency("fsspec")
            fs, path_or_handle = fsspec.core.url_to_fs(
                path_or_handle, **(storage_options or {})
            )
    elif storage_options and (not is_url(path_or_handle) or mode != "rb"):
        # can't write to a remote url
        # without making use of fsspec at the moment
        raise ValueError("storage_options passed with buffer, or non-supported URL")

    handles = None
    if (
        not fs
        and not is_dir
        and isinstance(path_or_handle, str)
        and not os.path.isdir(path_or_handle)
    ):
        # use get_handle only when we are very certain that it is not a directory
        # fsspec resources can also point to directories
        # this branch is used for example when reading from non-fsspec URLs
        handles = get_handle(
            path_or_handle, mode, is_text=False, storage_options=storage_options
        )
        fs = None
        path_or_handle = handles.handle
    return path_or_handle, handles, fs


class BaseImpl:
    # [CLASS DOC]
    # Abstract base for parquet engine implementations. Holds no state. Provides
    # a shared dataframe validator and abstract write and read methods that
    # subclasses (the pyarrow and fastparquet engines) must override.
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None:
        # [FUNCTION DOC]
        # Static check that the argument is a frame; raises a value error
        # otherwise. Returns nothing.
        if not isinstance(df, DataFrame):
            raise ValueError("to_parquet only supports IO with DataFrames")

    def write(self, df: DataFrame, path, compression, **kwargs) -> None:
        # [FUNCTION DOC]
        # Abstract write; always raises the abstract-method error. Subclasses
        # accept the frame, destination, compression and engine options.
        raise AbstractMethodError(self)

    def read(self, path, columns=None, **kwargs) -> DataFrame:
        # [FUNCTION DOC]
        # Abstract read; always raises the abstract-method error. Subclasses
        # return a frame from the source, optionally limited to columns.
        raise AbstractMethodError(self)


class PyArrowImpl(BaseImpl):
    # [CLASS DOC]
    # Parquet engine backed by pyarrow. Subclass of the abstract base.
    # State: api, the imported pyarrow package (with its parquet submodule
    # loaded). Construction requires pyarrow and registers pandas' arrow
    # extension types as an import side effect. Methods: write (frame to
    # file/dataset, preserving frame attributes in schema metadata) and read
    # (file/dataset to frame, restoring attributes).
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Imports pyarrow (raising an import error with a hint if missing),
        # imports its parquet module, imports the pandas arrow extension type
        # module purely for its registration side effect, and stores the pyarrow
        # package as the api attribute.
        import_optional_dependency(
            "pyarrow", extra="pyarrow is required for parquet support."
        )
        import pyarrow.parquet

        # import utils to register the pyarrow extension types
        import pandas.core.arrays.arrow.extension_types  # pyright: ignore[reportUnusedImport] # noqa: F401

        self.api = pyarrow

    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: ParquetCompressionOptions = "snappy",
        index: bool | None = None,
        storage_options: StorageOptions | None = None,
        partition_cols: list[str] | None = None,
        filesystem=None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Writes a frame with pyarrow.
        # Inputs: df the frame (validated); path destination path/buffer;
        # compression codec (default snappy, or none); index whether to store
        # the index (None lets pyarrow decide, storing a range index as
        # metadata); storage_options remote options; partition_cols column names
        # for a partitioned dataset write (None for a single file); filesystem
        # optional filesystem; extra keyword arguments go to the pyarrow
        # writer, except a schema entry which is removed and used for the arrow
        # table conversion.
        # Behavior: converts to an arrow table; if the frame has attributes,
        # serializes them as JSON into the table schema metadata (merged with
        # existing metadata). Prepares the path/handle via the helper; a
        # buffered file writer with a string or bytes name is replaced by that
        # name decoded to text. Writes a dataset when partition columns are
        # given, otherwise a single table. Any handle opened by the helper is
        # closed afterwards even on error. Returns nothing.
        # Errors: value error for non-frame input or bad filesystem or storage
        # options; others propagate.
        self.validate_dataframe(df)

        from_pandas_kwargs: dict[str, Any] = {"schema": kwargs.pop("schema", None)}
        if index is not None:
            from_pandas_kwargs["preserve_index"] = index

        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)

        if df.attrs:
            df_metadata = {"PANDAS_ATTRS": json.dumps(df.attrs)}
            existing_metadata = table.schema.metadata
            merged_metadata = {**existing_metadata, **df_metadata}
            table = table.replace_schema_metadata(merged_metadata)

        path_or_handle, handles, filesystem = _get_path_or_handle(
            path,
            filesystem,
            storage_options=storage_options,
            mode="wb",
            is_dir=partition_cols is not None,
        )
        if (
            isinstance(path_or_handle, io.BufferedWriter)
            and hasattr(path_or_handle, "name")
            and isinstance(path_or_handle.name, (str, bytes))
        ):
            if isinstance(path_or_handle.name, bytes):
                path_or_handle = path_or_handle.name.decode()
            else:
                path_or_handle = path_or_handle.name

        try:
            if partition_cols is not None:
                # writes to multiple files under the given path
                self.api.parquet.write_to_dataset(
                    table,
                    path_or_handle,
                    compression=compression,
                    partition_cols=partition_cols,
                    filesystem=filesystem,
                    **kwargs,
                )
            else:
                # write to single output file
                self.api.parquet.write_table(
                    table,
                    path_or_handle,
                    compression=compression,
                    filesystem=filesystem,
                    **kwargs,
                )
        finally:
            if handles is not None:
                handles.close()

    def read(
        self,
        path,
        columns=None,
        filters=None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        to_pandas_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> DataFrame:
        # [FUNCTION DOC]
        # Reads a parquet file or dataset into a frame with pyarrow.
        # Inputs: path source; columns optional column subset; filters optional
        # row filters in pyarrow format; dtype_backend unset, nullable numpy or
        # pyarrow; storage_options remote options; filesystem optional
        # filesystem; to_pandas_kwargs extra options for the arrow-to-pandas
        # conversion; extra keyword arguments are passed to the arrow reader
        # (the use-pandas-metadata flag is always forced on, overriding any
        # caller value, which mutates the local keyword mapping).
        # Behavior: prepares path/handle, reads the arrow table, converts it
        # while suppressing a block-construction deprecation warning, then, if
        # the schema metadata has a pandas attributes entry, decodes it from
        # JSON and assigns it to the frame attributes. Closes any opened handle
        # in all cases. Returns the frame.
        kwargs["use_pandas_metadata"] = True

        path_or_handle, handles, filesystem = _get_path_or_handle(
            path,
            filesystem,
            storage_options=storage_options,
            mode="rb",
        )
        try:
            pa_table = self.api.parquet.read_table(
                path_or_handle,
                columns=columns,
                filesystem=filesystem,
                filters=filters,
                **kwargs,
            )
            with catch_warnings():
                filterwarnings(
                    "ignore",
                    "make_block is deprecated",
                    Pandas4Warning,
                )
                result = arrow_table_to_pandas(
                    pa_table,
                    dtype_backend=dtype_backend,
                    to_pandas_kwargs=to_pandas_kwargs,
                )

            if pa_table.schema.metadata:
                if b"PANDAS_ATTRS" in pa_table.schema.metadata:
                    df_metadata = pa_table.schema.metadata[b"PANDAS_ATTRS"]
                    result.attrs = json.loads(df_metadata)
            return result
        finally:
            if handles is not None:
                handles.close()


class FastParquetImpl(BaseImpl):
    # [CLASS DOC]
    # Parquet engine backed by fastparquet. Subclass of the abstract base.
    # State: api, the imported fastparquet module. Supports fewer features than
    # the pyarrow engine: no filesystem argument, no dtype backend, no
    # to_pandas options, and no file-like destinations for writing. Methods:
    # write and read.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Imports fastparquet lazily (raising an import error with a hint if
        # missing; lazy because fastparquet itself depends on pandas) and stores
        # it as the api attribute.
        # since pandas is a dependency of fastparquet
        # we need to import on first use
        fastparquet = import_optional_dependency(
            "fastparquet", extra="fastparquet is required for parquet support."
        )
        self.api = fastparquet

    def write(
        self,
        df: DataFrame,
        path,
        compression: Literal["snappy", "gzip", "brotli"] | None = "snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Writes a frame with fastparquet.
        # Inputs: df frame (validated); path destination (converted to string);
        # compression snappy (default), gzip, brotli or none; index whether to
        # write the index (passed as the engine's write-index option);
        # partition_cols partition columns; storage_options remote options;
        # filesystem must be None; extra keyword arguments go to the engine.
        # Behavior: a partition_on keyword together with partition_cols is a
        # value error; a lone partition_on keyword is moved into the partition
        # columns. With partition columns the hive file scheme is requested.
        # A filesystem raises a not-implemented error. For fsspec URLs, an
        # opener callable (a lambda ignoring its second argument) is supplied
        # that opens the path for binary writing with the storage options;
        # storage options with any other path raise a value error. The engine
        # write runs with warnings recorded (so they are swallowed). Returns
        # nothing.
        self.validate_dataframe(df)

        if "partition_on" in kwargs and partition_cols is not None:
            raise ValueError(
                "Cannot use both partition_on and "
                "partition_cols. Use partition_cols for partitioning data"
            )
        if "partition_on" in kwargs:
            partition_cols = kwargs.pop("partition_on")

        if partition_cols is not None:
            kwargs["file_scheme"] = "hive"

        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )

        # cannot use get_handle as write() does not accept file buffers
        path = stringify_path(path)
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")

            # if filesystem is provided by fsspec, file must be opened in 'wb' mode.
            kwargs["open_with"] = lambda path, _: fsspec.open(
                path, "wb", **(storage_options or {})
            ).open()
        elif storage_options:
            raise ValueError(
                "storage_options passed with file object or non-fsspec file path"
            )

        with catch_warnings(record=True):
            self.api.write(
                path,
                df,
                compression=compression,
                write_index=index,
                partition_on=partition_cols,
                **kwargs,
            )

    def read(
        self,
        path,
        columns=None,
        filters=None,
        storage_options: StorageOptions | None = None,
        filesystem=None,
        to_pandas_kwargs: dict | None = None,
        **kwargs,
    ) -> DataFrame:
        # [FUNCTION DOC]
        # Reads a parquet file with fastparquet.
        # Inputs: path source; columns optional subset; filters optional
        # partition/row-group filters; storage_options remote options;
        # filesystem and to_pandas_kwargs must be None (not-implemented error
        # otherwise); extra keyword arguments pass to the engine's conversion,
        # except dtype_backend which is removed and must be unset (value error
        # otherwise). Nullable dtype use in the engine is always disabled.
        # Behavior: for fsspec URLs, passes the filesystem from an opened
        # fsspec file; for other string paths that are not directories, opens
        # through the shared handle opener and reads from the stream. Reads the
        # file, converts to pandas while suppressing a block-construction
        # deprecation warning, and always closes any opened handle. Returns the
        # frame.
        parquet_kwargs: dict[str, Any] = {}
        dtype_backend = kwargs.pop("dtype_backend", lib.no_default)
        # We are disabling nullable dtypes for fastparquet pending discussion
        parquet_kwargs["pandas_nulls"] = False
        if dtype_backend is not lib.no_default:
            raise ValueError(
                "The 'dtype_backend' argument is not supported for the "
                "fastparquet engine"
            )
        if filesystem is not None:
            raise NotImplementedError(
                "filesystem is not implemented for the fastparquet engine."
            )
        if to_pandas_kwargs is not None:
            raise NotImplementedError(
                "to_pandas_kwargs is not implemented for the fastparquet engine."
            )
        path = stringify_path(path)
        handles = None
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")

            parquet_kwargs["fs"] = fsspec.open(path, "rb", **(storage_options or {})).fs
        elif isinstance(path, str) and not os.path.isdir(path):
            # use get_handle only when we are very certain that it is not a directory
            # fsspec resources can also point to directories
            # this branch is used for example when reading from non-fsspec URLs
            handles = get_handle(
                path, "rb", is_text=False, storage_options=storage_options
            )
            path = handles.handle

        try:
            parquet_file = self.api.ParquetFile(path, **parquet_kwargs)
            with catch_warnings():
                filterwarnings(
                    "ignore",
                    "make_block is deprecated",
                    Pandas4Warning,
                )
                return parquet_file.to_pandas(
                    columns=columns, filters=filters, **kwargs
                )
        finally:
            if handles is not None:
                handles.close()


def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = None,
    engine: str = "auto",
    compression: ParquetCompressionOptions = "snappy",
    index: bool | None = None,
    storage_options: StorageOptions | None = None,
    partition_cols: list[str] | None = None,
    filesystem: Any = None,
    **kwargs,
) -> bytes | None:
    # [FUNCTION DOC]
    # Writes a frame to parquet through the chosen engine.
    # Inputs: df frame; path destination or None to return bytes; engine auto,
    # pyarrow or fastparquet; compression codec (default snappy); index whether
    # to store the index; storage_options remote options; partition_cols a
    # column name or list (a single string is wrapped into a list); filesystem
    # optional filesystem; extra keyword arguments go to the engine.
    # Behavior: selects the engine, writes into an in-memory buffer when path is
    # None, and returns its bytes; otherwise returns None. Errors from engine
    # selection or writing propagate.
    # KNOWN ISSUE: the docstring says the partition columns must be None if the
    # path is not a string, but this is not validated here.
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, file-like object, or None, default None
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function. If None, the result
        is returned as bytes. If a string, it will be used as Root Directory
        path when writing a partitioned dataset. The engine fastparquet does
        not accept file-like objects.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    compression : {'snappy', 'gzip', 'brotli', 'lz4', 'zstd', None},
        default 'snappy'. Name of the compression to use. Use ``None``
        for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value
        pairs are forwarded to ``urllib.request.Request`` as header options.
        For other URLs (e.g. starting with "s3://", and "gcs://") the
        key-value pairs are forwarded to ``fsspec.open``. Please see ``fsspec``
        and ``urllib`` for more details, and for more examples on storage
        options refer `here <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.
    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    **kwargs
        Additional keyword arguments passed to the engine:

        * For ``engine="pyarrow"``: passed to :func:`pyarrow.parquet.write_table`
          or :func:`pyarrow.parquet.write_to_dataset` (when using partition_cols)
        * For ``engine="fastparquet"``: passed to :func:`fastparquet.write`

    Returns
    -------
    bytes if no path argument is provided else None
    """
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]
    impl = get_engine(engine)

    path_or_buf: FilePath | WriteBuffer[bytes] = io.BytesIO() if path is None else path

    impl.write(
        df,
        path_or_buf,
        compression=compression,
        index=index,
        partition_cols=partition_cols,
        storage_options=storage_options,
        filesystem=filesystem,
        **kwargs,
    )

    if path is None:
        assert isinstance(path_or_buf, io.BytesIO)
        return path_or_buf.getvalue()
    else:
        return None


@set_module("pandas")
def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: str = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    filesystem: Any = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    to_pandas_kwargs: dict | None = None,
    **kwargs,
) -> DataFrame:
    # [FUNCTION DOC]
    # Public reader (exported as a top-level pandas function) that loads a
    # parquet file or dataset into a frame.
    # Inputs: path source; engine auto, pyarrow or fastparquet; columns optional
    # subset; storage_options remote options; dtype_backend unset, nullable
    # numpy or pyarrow (validated; the fastparquet engine rejects it);
    # filesystem optional filesystem (pyarrow only); filters optional row/
    # partition filters; to_pandas_kwargs options for the arrow-to-pandas step
    # (pyarrow only); extra keyword arguments go to the engine.
    # Behavior: selects the engine, validates the backend, and delegates to the
    # engine reader, returning its frame.
    """
    Load a parquet object from the file path, returning a DataFrame.

    The function automatically handles reading the data from a parquet file
    and creates a DataFrame with the appropriate structure.

    Parameters
    ----------
    path : str, path object or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function.
        The string could be a URL. Valid URL schemes include http, ftp, s3,
        gs, and file. For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value
        pairs are forwarded to ``urllib.request.Request`` as header options.
        For other URLs (e.g. starting with "s3://", and "gcs://") the
        key-value pairs are forwarded to ``fsspec.open``. Please see ``fsspec``
        and ``urllib`` for more details, and for more examples on storage
        options refer `here <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    filters : List[Tuple] or List[List[Tuple]], default None
        To filter out data.
        Filter syntax: [[(column, op, val), ...],...]
        where op is [==, =, >, >=, <, <=, !=, in, not in]
        The innermost tuples are transposed into a set of filters applied
        through an `AND` operation.
        The outer list combines these sets of filters through an `OR`
        operation.
        A single list of tuples can also be used, meaning that no `OR`
        operation between set of filters is to be conducted.

        Using this argument will NOT result in row-wise filtering of the final
        partitions unless ``engine="pyarrow"`` is also specified.  For
        other engines, filtering is only performed at the partition level, that is,
        to prevent the loading of some row-groups and/or files.

        .. versionadded:: 2.1.0

    to_pandas_kwargs : dict | None, default None
        Keyword arguments to pass through to :func:`pyarrow.Table.to_pandas`
        when ``engine="pyarrow"``.

        .. versionadded:: 3.0.0

    **kwargs
        Additional keyword arguments passed to the engine:

        * For ``engine="pyarrow"``: passed to :func:`pyarrow.parquet.read_table`
        * For ``engine="fastparquet"``: passed to
          :meth:`fastparquet.ParquetFile.to_pandas`

    Returns
    -------
    DataFrame
        DataFrame based on parquet file.

    See Also
    --------
    DataFrame.to_parquet : Create a parquet object that serializes a DataFrame.

    Examples
    --------
    >>> original_df = pd.DataFrame({"foo": range(5), "bar": range(5, 10)})
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> df_parquet_bytes = original_df.to_parquet()
    >>> from io import BytesIO
    >>> restored_df = pd.read_parquet(BytesIO(df_parquet_bytes))
    >>> restored_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> restored_df.equals(original_df)
    True
    >>> restored_bar = pd.read_parquet(BytesIO(df_parquet_bytes), columns=["bar"])
    >>> restored_bar
        bar
    0    5
    1    6
    2    7
    3    8
    4    9
    >>> restored_bar.equals(original_df[["bar"]])
    True

    The function uses `kwargs` that are passed directly to the engine.
    In the following example, we use the `filters` argument of the pyarrow
    engine to filter the rows of the DataFrame.

    Since `pyarrow` is the default engine, we can omit the `engine` argument.
    Note that the `filters` argument is implemented by the `pyarrow` engine,
    which can benefit from multithreading and also potentially be more
    economical in terms of memory.

    >>> sel = [("foo", ">", 2)]
    >>> restored_part = pd.read_parquet(BytesIO(df_parquet_bytes), filters=sel)
    >>> restored_part
        foo  bar
    0    3    8
    1    4    9
    """

    impl = get_engine(engine)
    check_dtype_backend(dtype_backend)

    return impl.read(
        path,
        columns=columns,
        filters=filters,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
        filesystem=filesystem,
        to_pandas_kwargs=to_pandas_kwargs,
        **kwargs,
    )
