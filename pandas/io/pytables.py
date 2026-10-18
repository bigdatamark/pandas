"""
High level interface to PyTables for reading and writing pandas data structures
to disk
"""
# [MODULE DOC]
# High level interface to PyTables (HDF5) for persisting pandas objects.
# Public contents: the to_hdf and read_hdf functions, the HDFStore class (a
# dict-like store over an HDF5 file), and the Term alias for query expressions.
# Internally it defines a hierarchy of storer classes (fixed format: generic,
# series, frame, block-manager, sparse-free variants; table format: appendable
# series, frame, multi-frame, worm-table), an index/column description layer
# (IndexCol, DataCol, DataIndexableCol, GenericDataIndexableCol), selection and
# iteration helpers, and many conversion helpers between pandas/numpy values and
# PyTables-storable values.
# Relationships: relies on the optional pytables dependency (imported lazily to
# reduce import time through a small accessor), the expression machinery in the
# core computation pytables module for where clauses, the concat function and
# core array types, and the io common path helper.
# Import-time side effects: registers two configuration options in the hdf
# section (a boolean controlling dropping all-missing rows on table append,
# default false, and the default write format, one of fixed, table or none).
# Module-level state: lazily-populated cache of the tables module and a flag for
# the PyTables file-open policy.

from __future__ import annotations

from contextlib import suppress
import copy
from datetime import (
    date,
    tzinfo,
)
import itertools
import os
import re
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Self,
    TypeAlias,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import (
    config,
    get_option,
    using_string_dtype,
)

from pandas._libs import (
    lib,
    writers as libwriters,
)
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat import HAS_PYARROW
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
    AttributeConflictWarning,
    ClosedFileError,
    IncompatibilityWarning,
    PerformanceWarning,
    PossibleDataLossError,
)
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_complex_dtype,
    is_list_like,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import array_equivalent

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    concat,
    isna,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    PeriodArray,
)
from pandas.core.arrays.datetimes import tz_to_dtype
from pandas.core.arrays.string_ import BaseStringArray
import pandas.core.common as com
from pandas.core.computation.pytables import (
    PyTablesExpr,
    maybe_expression,
)
from pandas.core.construction import (
    array as pd_array,
    extract_array,
)
from pandas.core.indexes.api import ensure_index

from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
    adjoin,
    pprint_thing,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
        Sequence,
    )
    from types import (
        ModuleType,
        TracebackType,
    )

    from tables import (
        Col,
        File,
        Node,
    )

    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        AxisInt,
        DtypeArg,
        FilePath,
        TimeUnit,
        npt,
    )

    from pandas.core.internals import Block

# versioning attribute
_version = "0.15.2"

# encoding
_default_encoding = "UTF-8"


def _ensure_encoding(encoding: str | None) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the given encoding name, or the module default (UTF-8) when the
    # argument is none.
    # set the encoding if we need
    if encoding is None:
        encoding = _default_encoding

    return encoding


def _ensure_str(name):
    # [FUNCTION DOC]
    # HELPER: normalizes an index or column name that may be a numpy string subtype to
    # a plain text string; any non-text value is returned unchanged.
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
    if isinstance(name, str):
        name = str(name)
    return name


Term: TypeAlias = PyTablesExpr


def _ensure_term(where, scope_level: int):
    # [FUNCTION DOC]
    # HELPER: normalizes a where argument of a read or query into query term objects.
    # Input where is none, a single expression text, a list or tuple of items (text
    # expressions are converted to terms, other items such as coordinate arrays are
    # kept, none items dropped), or an array of coordinates (returned unchanged).
    # Input scope_level is the number of stack frames between the user's call and
    # this function; it is used so term evaluation can look up local variables of the
    # caller's frame (one extra frame is added, and another for list items).
    # Returns the converted value, or none if the result is empty or was none.
    # # KNOWN ISSUE: the docstring mentions a frame level argument fixed to 2, but the
    # function takes a scope level parameter and computes the level from it.
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    # only consider list/tuple here as an ndarray is automatically a coordinate
    # list
    level = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [
            Term(term, scope_level=level + 1) if maybe_expression(term) else term
            for term in where
            if term is not None
        ]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return where if where is None or len(where) else None


incompatibility_doc: Final = """
where criteria is being ignored as this version [%s] is too old (or
not-defined), read the file in and write it out to a new file to upgrade (with
the copy_to method)
"""

attribute_conflict_doc: Final = """
the [%s] attribute of the existing index is [%s] which conflicts with the new
[%s], resetting the attribute to None
"""

performance_doc: Final = """
your performance may suffer as PyTables will pickle object types that it cannot
map directly to c-types [inferred_type->%s,key->%s] [items->%s]
"""

# formats
_FORMAT_MAP = {"f": "fixed", "fixed": "fixed", "t": "table", "table": "table"}

# axes map
_AXES_MAP = {DataFrame: [0]}

# register our configuration options
dropna_doc: Final = """
: boolean
    drop ALL nan rows when appending to a table
"""
format_doc: Final = """
: format
    default format writing format, if None, then
    put will default to 'fixed' and append will default to 'table'
"""

with config.config_prefix("io.hdf"):
    config.register_option("dropna_table", False, dropna_doc, validator=config.is_bool)
    config.register_option(
        "default_format",
        None,
        format_doc,
        validator=config.is_one_of_factory(["fixed", "table", None]),
    )

# oh the troubles to reduce import time
_table_mod: ModuleType | None = None
_table_file_open_policy_is_strict = False


def _tables():
    # [FUNCTION DOC]
    # HELPER: lazily imports and returns the PyTables module, caching it in module
    # state. On the first call also records in module state whether the PyTables file
    # open policy is strict (silently skipped if the attribute is missing in the
    # installed version). Raises an import error if PyTables is not installed.
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables

        _table_mod = tables

        # set the file open policy
        # return the file open policy; this changes as of pytables 3.1
        # depending on the HDF5 version
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (
                tables.file._FILE_OPEN_POLICY == "strict"
            )

    return _table_mod


# interface to/from ###


def to_hdf(
    path_or_buf: FilePath | HDFStore,
    key: str,
    value: DataFrame | Series,
    mode: str = "a",
    complevel: int | None = None,
    complib: str | None = None,
    append: bool = False,
    format: str | None = None,
    index: bool = True,
    min_itemsize: int | dict[str, int] | None = None,
    nan_rep=None,
    dropna: bool | None = None,
    data_columns: Literal[True] | list[str] | None = None,
    errors: str = "strict",
    encoding: str = "UTF-8",
) -> None:
    # [FUNCTION DOC]
    # Writes a pandas object into an HDF5 store, opening and closing the file if a
    # path was given.
    # Parameters: path_or_buf is a path or an already-open store; key is the group
    # name; value is a frame or series; mode is the file open mode used only when a
    # path is given (default append); complevel and complib configure compression
    # for a newly opened store; append selects appending to a table instead of
    # replacing; format, index, min_itemsize, nan_rep, data_columns, errors and
    # encoding are forwarded to the store write method (default format none lets the
    # store decide, index true creates table indexes, errors is strict, encoding is
    # UTF-8); dropna is forwarded (to append directly; to put it is also forwarded
    # as a keyword in the code, although a comment claims it is not).
    # Behavior: builds a local function (a lambda assigned to a name) that calls
    # either the append method or the put method of a store with the options, then
    # applies it to the given store, or to a freshly opened store (as a context
    # manager, so it is closed afterwards) built from the stringified path. A store
    # passed in by the caller is not closed.
    # Returns nothing. Errors raised by the store propagate.
    # # KNOWN ISSUE: the inline comment says dropna is not passed to the put method,
    # but the code does pass it.
    """store this object, close it if we opened it"""
    if append:
        f = lambda store: store.append(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )
    else:
        # NB: dropna is not passed to `put`
        f = lambda store: store.put(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
            dropna=dropna,
        )

    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        path_or_buf = stringify_path(path_or_buf)
        with HDFStore(
            path_or_buf, mode=mode, complevel=complevel, complib=complib
        ) as store:
            f(store)


@set_module("pandas")
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key=None,
    mode: str = "r",
    errors: str = "strict",
    where: str | list | None = None,
    start: int | None = None,
    stop: int | None = None,
    columns: list[str] | None = None,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwargs,
):
    # [FUNCTION DOC]
    # Reads a pandas object from an HDF5 file or an open store, optionally filtered.
    # Parameters: path_or_buf is a path or an open HDFStore; key is the group name
    # (none means the sole pandas object in the file); mode is the open mode, one of
    # read, read-write or append (default read), ignored for an open store; errors is
    # the text encoding error handling (default strict); where is a query expression
    # or list of them (converted using the caller's variable scope); start and stop
    # bound the selected rows; columns restricts columns; iterator and chunksize
    # request chunked iteration; extra keyword arguments are forwarded to the store
    # constructor.
    # Behavior: validates the mode; converts where; if given an open store, requires
    # it to be open and will not close it; otherwise requires a string path that
    # exists and opens a new store, which is closed automatically when the read
    # completes (delegated to the iterator when iterating). When key is none it
    # inspects the groups: there must be at least one, and every group after the
    # first must be a metadata group of the first, otherwise a key is required. Then
    # calls the store select method.
    # If a value, type or lookup error occurs and the store was opened here, the store
    # is closed (ignoring attribute errors) before re-raising.
    # Raises: a value error for a disallowed mode, no datasets, or multiple datasets
    # without key; an os error if a passed store is closed; a not-implemented error
    # for non-path buffers; a file-not-found error if the path does not exist.
    # Returns the selected object (a frame, series, or an iterator when requested).
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path_or_buf : str, path object, pandas.HDFStore
        Any valid string path is acceptable. Only supports the local file system,
        remote URLs and file-like objects are not supported.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.

    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {'r', 'r+', 'a'}, default 'r'
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is 'r'.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    where : list, optional
        A list of Term (or convertible) objects.
    start : int, optional
        Row number to start selection.
    stop : int, optional
        Row number to stop selection.
    columns : list, optional
        A list of columns names to return.
    iterator : bool, optional
        Return an iterator object.
    chunksize : int, optional
        Number of rows to include in an iteration when using an iterator.
    **kwargs
        Additional keyword arguments passed to HDFStore.

    Returns
    -------
    object
        The selected object. Return type depends on the object stored.

    See Also
    --------
    DataFrame.to_hdf : Write an HDF file from a DataFrame.
    HDFStore : Low-level access to HDF files.

    Notes
    -----
    When ``errors="surrogatepass"``, ``pd.options.future.infer_string`` is true,
    and PyArrow is installed, if a UTF-16 surrogate is encountered when decoding
    to UTF-8, the resulting dtype will be
    ``pd.StringDtype(storage="python", na_value=np.nan)``.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 1.0, "a"]], columns=["x", "y", "z"])  # doctest: +SKIP
    >>> df.to_hdf("./store.h5", "data")  # doctest: +SKIP
    >>> reread = pd.read_hdf("./store.h5")  # doctest: +SKIP
    """
    if mode not in ["r", "r+", "a"]:
        raise ValueError(
            f"mode {mode} is not allowed while performing a read. "
            f"Allowed modes are r, r+ and a."
        )
    # grab the scope
    if where is not None:
        where = _ensure_term(where, scope_level=1)

    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError("The HDFStore must be open for reading.")

        store = path_or_buf
        auto_close = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError(
                "Support for generic buffers has not been implemented."
            )
        try:
            exists = os.path.exists(path_or_buf)

        # if filepath is too long
        except (TypeError, ValueError):
            exists = False

        if not exists:
            raise FileNotFoundError(f"File {path_or_buf} does not exist")

        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        # can't auto open/close if we are using an iterator
        # so delegate to the iterator
        auto_close = True

    try:
        if key is None:
            groups = store.groups()
            if len(groups) == 0:
                raise ValueError(
                    "Dataset(s) incompatible with Pandas data types, "
                    "not table, or no datasets found in HDF5 file."
                )
            candidate_only_group = groups[0]

            # For the HDF file to have only one dataset, all other groups
            # should then be metadata groups for that candidate group. (This
            # assumes that the groups() method enumerates parent groups
            # before their children.)
            for group_to_check in groups[1:]:
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError(
                        "key must be provided when HDF5 "
                        "file contains multiple datasets."
                    )
            key = candidate_only_group._v_pathname
        return store.select(
            key,
            where=where,
            start=start,
            stop=stop,
            columns=columns,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            # if there is an error, close the store if we opened it.
            with suppress(AttributeError):
                store.close()

        raise


def _is_metadata_of(group: Node, parent_group: Node) -> bool:
    # [FUNCTION DOC]
    # HELPER: for read_hdf with no key. Returns true if group is a metadata group of
    # parent_group: it must be deeper than the parent, and walking up its ancestors
    # there must be a node named meta whose parent is the parent group. Returns false
    # otherwise.
    """Check if a given group is a metadata group for a given parent_group."""
    if group._v_depth <= parent_group._v_depth:
        return False

    current = group
    while current._v_depth > 1:
        parent = current._v_parent
        if parent == parent_group and current._v_name == "meta":
            return True
        current = current._v_parent
    return False


@set_module("pandas")
class HDFStore:
    # [CLASS DOC]
    # Dict-like interface to an HDF5 file holding pandas objects, in either the fixed
    # or the table format; the public store class of this module, usable as a context
    # manager.
    # State: the file path, open mode, the PyTables file handle (none when closed),
    # compression level, compression library, fletcher32 flag, and the PyTables
    # filters object built from them (created at open).
    # Key methods: open and close, is_open property, keys, groups, walk, get, select,
    # select_as_coordinates, select_column, select_as_multiple, put, append,
    # append_to_multiple, remove, create_table_index, get_node, get_storer, copy,
    # info; plus mapping-style access (get item, set item, delete item, contains,
    # length) and attribute access to stored keys.
    # Invariants: operations other than open require an open handle and raise a
    # closed-file error otherwise. Mode r forbids writes.
    """
    Dict-like IO interface for storing pandas objects in PyTables.

    Either Fixed or Table format.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path : str
        File path to HDF5 file.
    mode : {'a', 'w', 'r', 'r+'}, default 'a'

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        These additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
         'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.
    fletcher32 : bool, default False
        If applying compression use the fletcher32 checksum.
    **kwargs
        These parameters will be passed to the PyTables open_file method.

    Examples
    --------
    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore("test.h5")
    >>> store["foo"] = bar  # write to HDF5
    >>> bar = store["foo"]  # retrieve
    >>> store.close()

    **Create or load HDF5 file in-memory**

    When passing the `driver` option to the PyTables open_file method through
    **kwargs, the HDF5 file is loaded or created in-memory and will only be
    written when closed:

    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore("test.h5", driver="H5FD_CORE")
    >>> store["foo"] = bar
    >>> store.close()  # only now, data is written to disk
    """

    _handle: File | None
    _mode: str

    def __init__(
        self,
        path,
        mode: str = "a",
        complevel: int | None = None,
        complib=None,
        fletcher32: bool = False,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Creates the store and opens the file.
        # Parameters: path is a path-like (stringified); mode defaults to append (a none
        # value is treated as append); complevel is a compression level (zero or none
        # disables); complib is a compression library name or none; fletcher32 adds the
        # checksum filter; extra keyword arguments are passed to the PyTables open call.
        # Raises a value error if a format keyword is passed, or if complib is not one of
        # the libraries PyTables supports. Raises import error if PyTables is missing. If
        # a compression level is given without a library, the PyTables default library is
        # used. Opening may raise further errors from the open method (file not found for
        # read modes, etc.).
        if "format" in kwargs:
            raise ValueError("format is not a defined argument for HDFStore")

        tables = import_optional_dependency("tables")

        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(
                f"complib only supports {tables.filters.all_complibs} compression."
            )

        if complib is None and complevel is not None:
            complib = tables.filters.default_complib

        self._path = stringify_path(path)
        if mode is None:
            mode = "a"
        self._mode = mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        # [FUNCTION DOC]
        # Returns the file path text so that the store is usable as a path-like object.
        return self._path

    @property
    def root(self):
        # [FUNCTION DOC]
        # Property returning the root group node of the open file. Raises a closed-file
        # error if the store is closed.
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None  # for mypy
        return self._handle.root

    @property
    def filename(self) -> str:
        # [FUNCTION DOC]
        # Property returning the file path text given at construction.
        return self._path

    def __getitem__(self, key: str):
        # [FUNCTION DOC]
        # Mapping read access: returns the object stored under key, same as the get
        # method. Raises a key error if there is no such key.
        return self.get(key)

    def __setitem__(self, key: str, value) -> None:
        # [FUNCTION DOC]
        # Mapping write access: stores value under key with default options, same as the
        # put method (replacing existing contents).
        self.put(key, value)

    def __delitem__(self, key: str) -> int | None:
        # [FUNCTION DOC]
        # Mapping delete: removes the key (or table rows if applicable) via the remove
        # method and returns its result.
        return self.remove(key)

    def __getattr__(self, name: str):
        # [FUNCTION DOC]
        # Attribute access fallback: if no regular attribute exists, tries to return the
        # stored object whose key equals the attribute name. Raises an attribute error
        # when the key is missing or the file is closed.
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __contains__(self, key: str) -> bool:
        # [FUNCTION DOC]
        # Membership test by key. Looks up the node for the key and returns true only if
        # the key equals the node's absolute path or that path without its leading
        # slash; otherwise false.
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        node = self.get_node(key)
        if node is not None:
            name = node._v_pathname
            if key in (name, name[1:]):
                return True
        return False

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of top-level pandas groups (the count of groups reported by
        # the groups method).
        return len(self.groups())

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a text representation: the class, then the file path on a following
        # line.
        pstr = pprint_thing(self._path)
        return f"{type(self)}\nFile path: {pstr}\n"

    def __enter__(self) -> Self:
        # [FUNCTION DOC]
        # Context manager entry: returns the store itself.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC]
        # Context manager exit: closes the store; exception arguments are ignored and
        # exceptions are not suppressed.
        self.close()

    def keys(self, include: str = "pandas") -> list[str]:
        # [FUNCTION DOC]
        # Lists the keys of objects in the store.
        # Parameter include is either "pandas" (default) to list the absolute path names
        # of the top-level pandas groups, or "native" to list the paths of all native
        # PyTables table nodes found by walking the file from the root.
        # Returns a list of path text values, each with a leading slash.
        # Raises a value error for any other include value. Raises a closed-file error if
        # the store is closed (via the groups method for pandas keys).
        # # KNOWN ISSUE: the docstring describes the parameter as a kind and its
        # description refers to a kind argument, but the parameter is named include.
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------

        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').

        Raises
        ------
        raises ValueError if kind has an illegal value

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.close()  # doctest: +SKIP
        """
        if include == "pandas":
            return [n._v_pathname for n in self.groups()]

        elif include == "native":
            assert self._handle is not None  # mypy
            return [
                n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")
            ]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # Iterates over the keys returned by the keys method (pandas groups).
        return iter(self.keys())

    def items(self) -> Iterator[tuple[str, list]]:
        # [FUNCTION DOC]
        # Generator yielding pairs of the absolute path text and the PyTables group node
        # for every top-level pandas group, in the order of the groups method.
        # # KNOWN ISSUE: the return annotation says the second member is a list, but the
        # yielded value is a group node object.
        """
        iterate on key->group
        """
        for g in self.groups():
            yield g._v_pathname, g

    def open(self, mode: str = "a", **kwargs) -> None:
        # [FUNCTION DOC]
        # Opens (or re-opens) the HDF5 file.
        # Parameters: mode is the open mode (default append); extra keyword arguments go
        # to the PyTables open call.
        # Behavior: if the mode differs from the stored one, switching from a writing
        # mode (append or write) to a read mode is allowed; requesting write mode while
        # the file is open raises a possible-data-loss error (it would truncate the file);
        # the stored mode is updated. An already open handle is closed first. If a
        # compression level above zero is set, builds the PyTables filters object from
        # the level, library and checksum flag. Finally opens the file and stores the
        # handle.
        # Raises: possible-data-loss error as above; a value error if the PyTables open
        # policy is strict and the file is still open; errors from PyTables on open.
        # # KNOWN ISSUE: the strict-policy check runs after the handle was closed, so the
        # store is never open at that point and the value error can never be raised.
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        tables = _tables()

        if self._mode != mode:
            # if we are changing a write mode to read, ok
            if self._mode in ["a", "w"] and mode in ["r", "r+"]:
                pass
            elif mode in ["w"]:
                # this would truncate, raise here
                if self.is_open:
                    raise PossibleDataLossError(
                        f"Re-opening the file [{self._path}] with mode [{self._mode}] "
                        "will delete the current file!"
                    )

            self._mode = mode

        # close and reopen the handle
        if self.is_open:
            self.close()

        if self._complevel and self._complevel > 0:
            self._filters = _tables().Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )

        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                "Cannot open HDF5 file, which is already opened, "
                "even in read-only mode."
            )
            raise ValueError(msg)

        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the underlying PyTables file handle if there is one and clears it. Safe
        # to call repeatedly.
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if there is a handle and PyTables reports it as open, otherwise
        # false.
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync: bool = False) -> None:
        # [FUNCTION DOC]
        # Flushes buffered writes of the open file to disk.
        # Parameter fsync (default false): also forces the operating system to sync the
        # file descriptor, ignoring operating system errors. Does nothing if the store is
        # closed. Returns nothing.
        """
        Force all buffered modifications to be written to disk.

        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.

        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key: str):
        # [FUNCTION DOC]
        # Retrieves the whole object stored under key.
        # Looks up the node and reads it with a pickle-compatibility patch applied
        # (so legacy offset objects can be unpickled). Raises a key error if no object
        # exists, or a closed-file error if the store is closed. Returns the reconstructed
        # frame or series.
        """
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str
            Object to retrieve from file. Raises KeyError if not found.

        Returns
        -------
        object
            Same type as object stored in file.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        with patch_pickle():
            # GH#31167 Without this patch, pickle doesn't know how to unpickle
            #  old DateOffset objects now that they are cdef classes.
            group = self.get_node(key)
            if group is None:
                raise KeyError(f"No object named {key} in the file")
            return self._read_group(group)

    def select(
        self,
        key: str,
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ):
        # [FUNCTION DOC]
        # Reads the object under key, optionally filtered by a where expression.
        # Parameters: key is the group name; where is a query expression or list of
        # them (table format only); start and stop bound rows; columns restricts columns;
        # iterator and chunksize return a lazy iterator instead of the object;
        # auto_close closes the store when the read (or the iteration) finishes.
        # Behavior: raises a key error if the key is missing; converts where using the
        # caller's variable scope; builds the storer for the group and infers its axes;
        # defines a local reader that reads a row range and where with the column
        # restriction; wraps it in a table iterator over the number of rows and returns
        # that iterator's result, which is the object itself when not iterating.
        # Returns the object, or an iterator of chunks.
        """
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.

        See Also
        --------
        HDFStore.select_as_coordinates : Returns the selection as an index.
        HDFStore.select_column : Returns a single column from the table.
        HDFStore.select_as_multiple : Retrieves pandas objects from multiple tables.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.select("/data1")  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        >>> store.select("/data1", where="columns == A")  # doctest: +SKIP
           A
        0  1
        1  3
        >>> store.close()  # doctest: +SKIP
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        # create the storer and axes
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        # function to call on iteration
        def func(_start, _stop, _where):
            # [FUNCTION DOC]
            # HELPER: closure for the select method. Reads rows between the given start and
            # stop positions that satisfy the given where selection from the storer, limited
            # to the requested columns, and returns the resulting object.
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result()

    def select_as_coordinates(
        self,
        key: str,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        # [FUNCTION DOC]
        # Returns the row coordinates (positions) of rows matching a where expression, as
        # an index.
        # Parameters: key is the table name; where is a query expression (converted using
        # the caller's variable scope); start and stop bound the rows searched.
        # Raises a type error if the stored object is not in table format.
        """
        return the selection as an Index

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.


        Parameters
        ----------
        key : str
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        """
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_coordinates with a table")
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(
        self,
        key: str,
        column: str,
        start: int | None = None,
        stop: int | None = None,
    ):
        # [FUNCTION DOC]
        # Returns a single column of a table-format object, generally an indexable or
        # data column.
        # Parameters: key is the table name; column the column name; start and stop bound
        # the rows.
        # Raises a type error if the object is not a table. The table read can raise a
        # key error if the column is not found, or a value error if the column is stored
        # inside a combined block and cannot be extracted individually.
        """
        return a single column from the table. This is generally only useful to
        select an indexable

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
        column : str
            The column of interest.
        start : int or None, default None
        stop : int or None, default None

        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)

        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_column with a table")
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(
        self,
        keys,
        where=None,
        selector=None,
        columns=None,
        start=None,
        stop=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ):
        # [FUNCTION DOC]
        # Reads several tables that share the same rows and combines them into one frame,
        # using one table (the selector) to evaluate the where condition.
        # Parameters: keys is a table name or a list or tuple of names (a one-element
        # list is reduced to a single name); where is a query expression; selector is the
        # table used to evaluate where (default the first key); columns restricts the
        # columns; start and stop bound rows; iterator, chunksize and auto_close behave
        # as for select.
        # Behavior: a single key delegates to select. Otherwise requires a non-empty list
        # or tuple, fetches all storers, checks that each is a table with the same
        # number of rows, finds the concatenation axis from the tables' non-index axis,
        # and reads the selector rows as coordinates and the objects from every table at
        # those coordinates, concatenated along that axis and consolidated.
        # Raises: type error if keys is not a list or tuple, or any object is not a
        # table; value error if keys is empty or row counts differ; key error if a
        # storer is missing.
        """
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : bool, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        """
        # default to single select
        where = _ensure_term(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(
                key=keys,
                where=where,
                columns=columns,
                start=start,
                stop=stop,
                iterator=iterator,
                chunksize=chunksize,
                auto_close=auto_close,
            )

        if not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a list/tuple")

        if not len(keys):
            raise ValueError("keys must have a non-zero length")

        if selector is None:
            selector = keys[0]

        # collect the tables
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)

        # validate rows
        nrows = None
        for t, k in itertools.chain([(s, selector)], zip(tbls, keys, strict=True)):
            if t is None:
                raise KeyError(f"Invalid table [{k}]")
            if not t.is_table:
                raise TypeError(
                    f"object [{t.pathname}] is not a table, and cannot be used in all "
                    "select as multiple"
                )

            if nrows is None:
                nrows = t.nrows
            elif t.nrows != nrows:
                raise ValueError("all tables must have exactly the same nrows!")

        # The isinstance checks here are redundant with the check above,
        #  but necessary for mypy; see GH#29757
        _tbls = [x for x in tbls if isinstance(x, Table)]

        # axis is the concentration axes
        axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func(_start, _stop, _where):
            # [FUNCTION DOC]
            # HELPER: closure for select-as-multiple. Reads the given row range with the
            # given coordinates from every table, restricted to the columns, and concatenates
            # the pieces along the chosen axis without checking index integrity, then
            # consolidates blocks. The where argument is always coordinates here.
            # retrieve the objs, _where is always passed as a set of
            # coordinates here
            objs = [
                t.read(where=_where, columns=columns, start=_start, stop=_stop)
                for t in tbls
            ]

            # concat and return
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result(coordinates=True)

    def put(
        self,
        key: str,
        value: DataFrame | Series,
        format=None,
        index: bool = True,
        append: bool = False,
        complib=None,
        complevel: int | None = None,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        data_columns: Literal[True] | list[str] | None = None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
        dropna: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores a frame or series under a key, replacing any existing object at that key
        # unless appending.
        # Parameters: key is the node name; value is the frame or series; format is fixed,
        # table or their short forms (default taken from the default-format option, else
        # fixed); index (default true) creates table indexes after writing (or a list of
        # column names); append (default false) adds to an existing table; complib and
        # complevel configure compression for tables; min_itemsize sets minimum string
        # widths; nan_rep is the text for missing strings; data_columns lists columns to
        # store as queryable columns (true for all); encoding (default none, UTF-8 at the
        # storer) and errors control string encoding; track_times (default true) is passed
        # to PyTables table creation; dropna (default false) drops all-missing rows when
        # writing tables.
        # Behavior: resolves and validates the format (type error if invalid) and delegates
        # to the shared group writer. See that writer for the checks: an empty object in
        # table format is silently not written, appending requires a table, and a
        # compression library with fixed format raises a value error.
        # # KNOWN ISSUE: the docstring says appending forces the table format and that the
        # compression library is not accepted. In the code, appending with the default fixed
        # format raises a value error (only tables can be appended), and the compression
        # library is accepted and used for table format (only rejected for fixed format).
        """
        Store object in HDFStore.

        This method writes a pandas DataFrame or Series into an HDF5 file using
        either the fixed or table format. The `table` format allows additional
        operations like incremental appends and queries but may have performance
        trade-offs. The `fixed` format provides faster read/write operations but
        does not support appends or queries.

        Parameters
        ----------
        key : str
            Key of object to store in file.
        value : {Series, DataFrame}
            Value of object to store in file.
        format : 'fixed(f)|table(t)', default is 'fixed'
            Format to use when storing object in HDFStore. Value can be one of:

            ``'fixed'``
                Fixed format.  Fast writing/reading. Not-appendable, nor searchable.
            ``'table'``
                Table format.  Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        index : bool, default True
            Write DataFrame index as a column.
        append : bool, default False
            This will force Table format, append the input data to the existing.
        complib : default None
            This parameter is currently not accepted.
        complevel : int, 0-9, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        min_itemsize : int, dict, or None
            Dict of columns that specify minimum str sizes.
        nan_rep : str
            Str to use as str nan representation.
        data_columns : list of columns or True, default None
            List of columns to create as data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str, default None
            Provide an encoding for strings.
        errors : str, default 'strict'
            The error handling scheme to use for encoding errors.
            The default is 'strict' meaning that encoding errors raise a
            UnicodeEncodeError.  Other possible values are 'ignore', 'replace' and
            'xmlcharrefreplace' as well as any other name registered with
            codecs.register_error that can handle UnicodeEncodeErrors.
        track_times : bool, default True
            Parameter is propagated to 'create_table' method of 'PyTables'.
            If set to False it enables to have the same h5 files (same hashes)
            independent on creation time.
        dropna : bool, default False, optional
            Remove missing values.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        """
        if format is None:
            format = get_option("io.hdf.default_format") or "fixed"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
            dropna=dropna,
        )

    def remove(self, key: str, where=None, start=None, stop=None) -> int | None:
        # [FUNCTION DOC]
        # Removes a whole node or rows of a table.
        # Parameters: key is the node; where is a query expression; start and stop are
        # row bounds. Returns the number of rows removed for a partial removal, or none
        # when the entire node was removed.
        # Behavior: converts where; gets the storer. A key error from that propagates.
        # For other failures to build a storer (for example a non-pandas group): if a
        # where clause was given, raises a value error; else removes the node and its
        # children and returns none. If no where, start or stop is given, removes the
        # whole group recursively. Otherwise requires table format (value error if not)
        # and deletes the matching rows via the table delete method.
        # # KNOWN ISSUE: if building the storer fails and the node lookup then returns no
        # node, execution continues with the storer variable unset, which would raise an
        # unbound variable error; this path is effectively unreachable because the
        # storer lookup would have raised a key error first.
        """
        Remove pandas object partially by specifying the where condition

        Parameters
        ----------
        key : str
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection

        Returns
        -------
        number of rows removed (or None if not a Table)

        Raises
        ------
        raises KeyError if key is not a valid store

        """
        where = _ensure_term(where, scope_level=1)
        try:
            s = self.get_storer(key)
        except KeyError:
            # the key is not a valid store, re-raising KeyError
            raise
        except AssertionError:
            # surface any assertion errors for e.g. debugging
            raise
        except Exception as err:
            # In tests we get here with ClosedFileError, TypeError, and
            #  _table_mod.NoSuchNodeError.  TODO: Catch only these?

            if where is not None:
                raise ValueError(
                    "trying to remove a node with a non-None where clause!"
                ) from err

            # we are actually trying to remove a node (with children)
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None

        # remove the node
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)
            return None

        # delete from the table
        if not s.is_table:
            raise ValueError("can only remove with where on objects written as tables")
        return s.delete(where=where, start=start, stop=stop)

    def append(
        self,
        key: str,
        value: DataFrame | Series,
        format=None,
        axes=None,
        index: bool | list[str] = True,
        append: bool = True,
        complib=None,
        complevel: int | None = None,
        columns=None,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool | None = None,
        data_columns: Literal[True] | list[str] | None = None,
        encoding=None,
        errors: str = "strict",
    ) -> None:
        # [FUNCTION DOC]
        # Appends a frame or series to a table, creating it if needed.
        # Parameters: key is the node name; value is the data; format is fixed, table or
        # short forms (default from the default-format option, else table); axes selects
        # the index axes (default the first axis); index (default true) creates indexes after
        # writing; append (default true) must stay true for appending semantics; complib and
        # complevel configure compression of a new table; columns is not supported and any
        # value raises a type error; min_itemsize sets minimum string widths; nan_rep is the
        # text for missing strings; chunksize is the rows per write chunk; expectedrows is a
        # size hint for creation; dropna (default from the drop-all-missing-rows option)
        # drops rows with all values missing; data_columns lists columns stored as
        # queryable columns (true for all); encoding and errors control string encoding.
        # Behavior: validates the format and delegates to the shared group writer. Does not
        # check whether appended rows overlap existing ones. A frame whose columns or dtypes
        # do not match the existing table is rejected by the writer's validation.
        # # KNOWN ISSUE: the docstring says the node must already exist, but the writer
        # creates the node when it is missing. It also says the axes and compression library
        # parameters are not accepted, while the code passes both through and uses them.
        """
        Append to Table in file.

        Node must already exist and be Table format.

        Parameters
        ----------
        key : str
            Key of object to append.
        value : {Series, DataFrame}
            Value of object to append.
        format : 'table' is the default
            Format to use when storing object in HDFStore.  Value can be one of:

            ``'table'``
                Table format. Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        axes : default None
            This parameter is currently not accepted.
        index : bool, default True
            Write DataFrame index as a column.
        append : bool, default True
            Append the input data to the existing.
        complib : default None
            This parameter is currently not accepted.
        complevel : int, 0-9, default None
            Specifies a compression level for data.
            A value of 0 or None disables compression.
        columns : default None
            This parameter is currently not accepted, try data_columns.
        min_itemsize : int, dict, or None
            Dict of columns that specify minimum str sizes.
        nan_rep : str
            Str to use as str nan representation.
        chunksize : int or None
            Size to chunk the writing.
        expectedrows : int
            Expected TOTAL row size of this table.
        dropna : bool, default False, optional
            Do not write an ALL nan row to the store settable
            by the option 'io.hdf.dropna_table'.
        data_columns : list of columns, or True, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : default None
            Provide an encoding for str.
        errors : str, default 'strict'
            The error handling scheme to use for encoding errors.
            The default is 'strict' meaning that encoding errors raise a
            UnicodeEncodeError.  Other possible values are 'ignore', 'replace' and
            'xmlcharrefreplace' as well as any other name registered with
            codecs.register_error that can handle UnicodeEncodeErrors.

        See Also
        --------
        HDFStore.append_to_multiple : Append to multiple tables.

        Notes
        -----
        Does *not* check if data being appended overlaps with existing
        data in the table, so be careful

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8
        """
        if columns is not None:
            raise TypeError(
                "columns is not a supported keyword in append, try data_columns"
            )

        if dropna is None:
            dropna = get_option("io.hdf.dropna_table")
        if format is None:
            format = get_option("io.hdf.default_format") or "table"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            axes=axes,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
        )

    def append_to_multiple(
        self,
        d: dict,
        value,
        selector,
        data_columns=None,
        axes=None,
        dropna: bool = False,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Splits a frame by columns into several tables and appends each, using the
        # selector table as the indexable one.
        # Parameters: d is a dictionary from table name to list of columns (at most one
        # value may be none meaning all remaining columns; this entry is replaced
        # in the passed dictionary by the computed remaining columns, so the caller's
        # dictionary is mutated); value is the frame; selector is a key of d naming the
        # table whose columns become data columns (unless data_columns is given); axes
        # must be none; dropna (default false) drops, from all tables, rows where any
        # table has all values missing; extra keyword arguments go to the append method
        # (a min_itemsize mapping is filtered per table).
        # Raises a type error if axes is given; value errors if d is not a dictionary,
        # the selector is not in d, or more than one entry is none.
        # Side effects: appends to each table.
        """
        Append to multiple tables

        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : list of columns to create as data columns, or True to
            use all columns
        dropna : if evaluates to True, drop rows from all tables if any single
                 row in each table has all NaN. Default False.

        Notes
        -----
        axes parameter is currently not accepted

        """
        if axes is not None:
            raise TypeError(
                "axes is currently not accepted as a parameter to append_to_multiple; "
                "you can create the tables independently instead"
            )

        if not isinstance(d, dict):
            raise ValueError(
                "append_to_multiple must have a dictionary specified as the "
                "way to split the value"
            )

        if selector not in d:
            raise ValueError(
                "append_to_multiple requires a selector that is in passed dict"
            )

        # figure out the splitting axis (the non_index_axis)
        axis = next(iter(set(range(value.ndim)) - set(_AXES_MAP[type(value)])))

        # figure out how to split the value
        remain_key = None
        remain_values: list = []
        for k, v in d.items():
            if v is None:
                if remain_key is not None:
                    raise ValueError(
                        "append_to_multiple can only have one value in d that is None"
                    )
                remain_key = k
            else:
                remain_values.extend(v)
        if remain_key is not None:
            ordered = value.axes[axis]
            ordd = ordered.difference(Index(remain_values))
            ordd = sorted(ordered.get_indexer(ordd))
            d[remain_key] = ordered.take(ordd)

        # data_columns
        if data_columns is None:
            data_columns = d[selector]

        # ensure rows are synchronized across the tables
        if dropna:
            idxs = (value[cols].dropna(how="all").index for cols in d.values())
            valid_index = next(idxs)
            for index in idxs:
                valid_index = valid_index.intersection(index)
            value = value.loc[valid_index]

        min_itemsize = kwargs.pop("min_itemsize", None)

        # append
        for k, v in d.items():
            dc = data_columns if k == selector else None

            # compute the val
            val = value.reindex(v, axis=axis)

            filtered = (
                {key: value for (key, value) in min_itemsize.items() if key in v}
                if min_itemsize is not None
                else None
            )
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def create_table_index(
        self,
        key: str,
        columns=None,
        optlevel: int | None = None,
        kind: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Creates PyTables indexes on a table-format object.
        # Parameters: key is the table name; columns is false (no indexes), true or none
        # (all indexable columns), or a list of column names; optlevel is the PyTables
        # optimization level (none uses the library default); kind is the index kind
        # (none uses the library default).
        # Raises a type error if the object is in fixed format. Returns nothing; returns
        # silently if no storer was found.
        """
        Create a pytables index on the table.

        Parameters
        ----------
        key : str
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError: raises if the node is not a table
        """
        # version requirements
        _tables()
        s = self.get_storer(key)
        if s is None:
            return

        if not isinstance(s, Table):
            raise TypeError("cannot create table index on a Fixed format store")
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self) -> list:
        # [FUNCTION DOC]
        # Returns the list of top-level nodes that hold pandas objects: groups that
        # carry a pandas type attribute, or have a table child, or are tables not named
        # table, and that are not links. Raises a closed-file error if closed.
        # Returns the list of PyTables nodes (not pandas objects).
        """
        Return a list of all the top-level nodes.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> print(store.groups())  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        [/data (Group) ''
          children := ['axis0' (Array), 'axis1' (Array), 'block0_values' (Array),
          'block0_items' (Array)]]
        """
        _tables()
        self._check_if_open()
        assert self._handle is not None  # for mypy
        assert _table_mod is not None  # for mypy
        return [
            g
            for g in self._handle.walk_groups()
            if (
                not isinstance(g, _table_mod.link.Link)
                and (
                    getattr(g._v_attrs, "pandas_type", None)
                    or getattr(g, "table", None)
                    or (isinstance(g, _table_mod.table.Table) and g._v_name != "table")
                )
            )
        ]

    def walk(self, where: str = "/") -> Iterator[tuple[str, list[str], list[str]]]:
        # [FUNCTION DOC]
        # Generator that walks the group hierarchy for pandas objects, similar to a file
        # system walk.
        # Parameter where is the starting group path (default the root).
        # For every PyTables group under where (pre-order, children alphabetically as the
        # underlying walker provides) that is not itself a pandas object (no pandas type
        # attribute), yields a triple: the group's absolute path without a trailing
        # slash, the list of names of child groups that are not pandas objects, and the
        # list of names of children that are pandas objects. Non-group, non-pandas
        # children are ignored.
        # Raises a closed-file error if the store is closed.
        """
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing '/').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        _tables()
        self._check_if_open()
        assert self._handle is not None  # for mypy
        assert _table_mod is not None  # for mypy

        for g in self._handle.walk_groups(where):
            if getattr(g._v_attrs, "pandas_type", None) is not None:
                continue

            groups = []
            leaves = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, "pandas_type", None)
                if pandas_type is None:
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)

            yield (g._v_pathname.rstrip("/"), groups, leaves)

    def get_node(self, key: str) -> Node | None:
        # [FUNCTION DOC]
        # Returns the PyTables node at the given key (a leading slash is added if
        # missing), or none if no such node exists. Raises a closed-file error if the
        # store is closed.
        """return the node with the key or None if it does not exist"""
        self._check_if_open()
        if not key.startswith("/"):
            key = "/" + key

        assert self._handle is not None
        assert _table_mod is not None  # for mypy
        try:
            node = self._handle.get_node(self.root, key)
        except _table_mod.exceptions.NoSuchNodeError:
            return None

        assert isinstance(node, _table_mod.Node), type(node)
        return node

    def get_storer(self, key: str) -> GenericFixed | Table:
        # [FUNCTION DOC]
        # Returns the storer object (the fixed or table wrapper that knows how to read and
        # write the data) for a key, with its axes inferred from the stored metadata.
        # Raises a key error if the key does not exist.
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        s = self._create_storer(group)
        s.infer_axes()
        return s

    def copy(
        self,
        file,
        mode: str = "w",
        propindexes: bool = True,
        keys=None,
        complib=None,
        complevel: int | None = None,
        fletcher32: bool = False,
        overwrite: bool = True,
    ) -> HDFStore:
        # [FUNCTION DOC]
        # Copies the selected objects of this store into another file and returns the
        # new (still open) store.
        # Parameters: file is the destination path; mode (default write), complib,
        # complevel and fletcher32 configure the new store as for the constructor; keys is
        # a key, or list or tuple of keys, to copy (default all pandas keys);
        # propindexes (default true) re-creates, for table objects, indexes on those axes
        # that were indexed in the source; overwrite (default true) removes existing keys
        # in the destination before copying.
        # Behavior: for each key reads the whole object. Table-format objects are
        # appended to the destination (preserving data columns and encoding, with index
        # creation set to the indexed axes or disabled); fixed-format objects are put
        # with their encoding. When overwrite is false and a key already exists in the
        # destination, tables get the rows appended and fixed objects replace.
        # Raises a key error for a missing key; the source must be open.
        # # KNOWN ISSUE: the storer-is-none check is redundant because the storer lookup
        # raises instead of returning none.
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list, optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        open file handle of the new store
        """
        new_store = HDFStore(
            file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32
        )
        if keys is None:
            keys = list(self.keys())
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        for k in keys:
            s = self.get_storer(k)
            if s is not None:
                if k in new_store:
                    if overwrite:
                        new_store.remove(k)

                data = self.select(k)
                if isinstance(s, Table):
                    index: bool | list[str] = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(
                        k,
                        data,
                        index=index,
                        data_columns=getattr(s, "data_columns", None),
                        encoding=s.encoding,
                    )
                else:
                    new_store.put(k, data, encoding=s.encoding)

        return new_store

    def info(self) -> str:
        # [FUNCTION DOC]
        # Builds and returns (does not print) a text summary of the
        # store: the class name and file path, then one line per sorted key giving the
        # storer's description, aligned in columns. If the store has no keys the text is
        # the word for empty; if the file is closed, a closed notice. A key whose storer
        # cannot be created is listed with an invalid-node message containing the error
        # text (assertion errors propagate).
        """
        Print detailed information on the store.

        Returns
        -------
        str
            A String containing the python pandas class name, filepath to the HDF5
            file and all the object keys along with their respective dataframe shapes.

        See Also
        --------
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["C", "D"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data1", df1)  # doctest: +SKIP
        >>> store.put("data2", df2)  # doctest: +SKIP
        >>> print(store.info())  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        <class 'pandas.io.pytables.HDFStore'>
        File path: store.h5
        /data1            frame        (shape->[2,2])
        /data2            frame        (shape->[2,2])
        """
        path = pprint_thing(self._path)
        output = f"{type(self)}\nFile path: {path}\n"

        if self.is_open:
            lkeys = sorted(self.keys())
            if lkeys:
                keys = []
                values = []

                for k in lkeys:
                    try:
                        s = self.get_storer(k)
                        if s is not None:
                            keys.append(pprint_thing(s.pathname or k))
                            values.append(pprint_thing(s or "invalid_HDFStore node"))
                    except AssertionError:
                        # surface any assertion errors for e.g. debugging
                        raise
                    except Exception as detail:
                        keys.append(k)
                        dstr = pprint_thing(detail)
                        values.append(f"[invalid_HDFStore node: {dstr}]")

                output += adjoin(12, keys, values)
            else:
                output += "Empty"
        else:
            output += "File is CLOSED"

        return output

    # ------------------------------------------------------------------------
    # private methods

    def _check_if_open(self) -> None:
        # [FUNCTION DOC]
        # HELPER: raises a closed-file error naming the path if the store is not open.
        if not self.is_open:
            raise ClosedFileError(f"{self._path} file is not open!")

    def _validate_format(self, format: str) -> str:
        # [FUNCTION DOC]
        # HELPER: normalizes a format name. Accepts, case-insensitively, the short or long
        # forms for fixed and table, and returns the long form. Raises a type error for
        # anything else (including values without a lower-case method).
        """validate / deprecate formats"""
        # validate
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f"invalid HDFStore format specified [{format}]") from err

        return format

    def _create_storer(
        self,
        group,
        format=None,
        value: DataFrame | Series | None = None,
        encoding: str = "UTF-8",
        errors: str = "strict",
    ) -> GenericFixed | Table:
        # [FUNCTION DOC]
        # HELPER: chooses and instantiates the storer class for a group.
        # Parameters: group is the PyTables node; format is the requested format (only
        # consulted when the node does not yet carry type information); value is an
        # optional frame or series being written (must be one of those or none, else type
        # error); encoding and errors are passed to the storer.
        # Behavior: reads the pandas type and table type attributes of the group. If no
        # pandas type: with no value, a node that is or has a table is treated as a
        # generic frame table, otherwise raises a type error; with a value, the type is
        # series or frame, with a table suffix when the format is table. For non-table
        # types the class comes from a map of series and frame to the fixed storers (type
        # error if unknown). For table types, if the table type attribute is missing and a
        # value is given, it is chosen from the value being a series or frame and
        # whether the index is single-level (appendable) or multi-level (appendable
        # multi). The class comes from a map of table kinds (generic, appendable
        # series, multi series, frame, multi frame, worm) and a missing entry raises a
        # type error. Returns the new storer instance.
        """return a suitable class to operate"""
        cls: type[GenericFixed | Table]

        if value is not None and not isinstance(value, (Series, DataFrame)):
            raise TypeError("value must be None, Series, or DataFrame")

        pt = getattr(group._v_attrs, "pandas_type", None)
        tt = getattr(group._v_attrs, "table_type", None)

        # infer the pt from the passed value
        if pt is None:
            if value is None:
                _tables()
                assert _table_mod is not None  # for mypy
                if getattr(group, "table", None) or isinstance(
                    group, _table_mod.table.Table
                ):
                    pt = "frame_table"
                    tt = "generic_table"
                else:
                    raise TypeError(
                        "cannot create a storer if the object is not existing "
                        "nor a value are passed"
                    )
            else:
                if isinstance(value, Series):
                    pt = "series"
                else:
                    pt = "frame"

                # we are actually a table
                if format == "table":
                    pt += "_table"

        # a storer node
        if "table" not in pt:
            _STORER_MAP = {"series": SeriesFixed, "frame": FrameFixed}
            try:
                cls = _STORER_MAP[pt]
            except KeyError as err:
                raise TypeError(
                    f"cannot properly create the storer for: [_STORER_MAP] [group->"
                    f"{group},value->{type(value)},format->{format}"
                ) from err
            return cls(self, group, encoding=encoding, errors=errors)

        # existing node (and must be a table)
        if tt is None:
            # if we are a writer, determine the tt
            if value is not None:
                if pt == "series_table":
                    index = getattr(value, "index", None)
                    if index is not None:
                        if index.nlevels == 1:
                            tt = "appendable_series"
                        elif index.nlevels > 1:
                            tt = "appendable_multiseries"
                elif pt == "frame_table":
                    index = getattr(value, "index", None)
                    if index is not None:
                        if index.nlevels == 1:
                            tt = "appendable_frame"
                        elif index.nlevels > 1:
                            tt = "appendable_multiframe"

        _TABLE_MAP = {
            "generic_table": GenericTable,
            "appendable_series": AppendableSeriesTable,
            "appendable_multiseries": AppendableMultiSeriesTable,
            "appendable_frame": AppendableFrameTable,
            "appendable_multiframe": AppendableMultiFrameTable,
            "worm": WORMTable,
        }
        try:
            cls = _TABLE_MAP[tt]  # type: ignore[index]
        except KeyError as err:
            raise TypeError(
                f"cannot properly create the storer for: [_TABLE_MAP] [group->"
                f"{group},value->{type(value)},format->{format}"
            ) from err

        return cls(self, group, encoding=encoding, errors=errors)

    def _write_to_group(
        self,
        key: str,
        value: DataFrame | Series,
        format,
        axes=None,
        index: bool | list[str] = True,
        append: bool = False,
        complib=None,
        complevel: int | None = None,
        fletcher32=None,
        min_itemsize: int | dict[str, int] | None = None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool = False,
        nan_rep=None,
        data_columns=None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: shared implementation of put and append: writes value under key.
        # Parameters: key and value; format already validated (fixed or table); axes,
        # complib, complevel, fletcher32, min_itemsize, chunksize, expectedrows, dropna,
        # nan_rep, data_columns, track_times are passed to the storer write method;
        # append says whether to add to an existing table; index is true, false or a list
        # of column names for index creation after writing; encoding and errors are for
        # string handling.
        # Behavior: if the value reports empty and the format is table or append is set,
        # returns without writing anything. Otherwise finds or creates the group (an
        # existing group is deleted first unless appending), creates the storer for the
        # value and format, and when appending requires a table storer (and disallows a
        # fixed-format request on an existing node) else raises a value error; writes the
        # object info for a new storer; raises a value error if compression library was
        # specified for a non-table storer; calls the storer write; and for table storers
        # with truthy index, creates the indexes for the given columns.
        # Raises value errors as above. Side effects: modifies the file.
        # we don't want to store a table node at all if our object is 0-len
        # as there are not dtypes
        if getattr(value, "empty", None) and (format == "table" or append):
            return

        group = self._identify_group(key, append)

        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
        if append:
            # raise if we are trying to append to a Fixed format,
            #       or a table that exists (and we are putting)
            if not s.is_table or (s.is_table and format == "fixed" and s.is_exists):
                raise ValueError("Can only append to Tables")
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()

        if not s.is_table and complib:
            raise ValueError("Compression not supported on Fixed format stores")

        # write the object
        s.write(
            obj=value,
            axes=axes,
            append=append,
            complib=complib,
            complevel=complevel,
            fletcher32=fletcher32,
            min_itemsize=min_itemsize,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            nan_rep=nan_rep,
            data_columns=data_columns,
            track_times=track_times,
        )

        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def _read_group(self, group: Node):
        # [FUNCTION DOC]
        # HELPER: reads an entire stored object from a group by creating its storer,
        # inferring axes and reading everything. Returns the object.
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def _identify_group(self, key: str, append: bool) -> Node:
        # [FUNCTION DOC]
        # HELPER: returns the group for a key. If the node exists and append is false the
        # node is removed recursively and re-created; if it does not exist it is created
        # along with missing parent groups. Returns the (possibly new) group node.
        # Side effect: removes existing data when not appending.
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)

        # we make this assertion for mypy; the get_node call will already
        # have raised if this is incorrect
        assert self._handle is not None

        # remove the node if we are not appending
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None

        if group is None:
            group = self._create_nodes_and_group(key)

        return group

    def _create_nodes_and_group(self, key: str) -> Node:
        # [FUNCTION DOC]
        # HELPER: creates, one level at a time, any missing groups along a slash-separated
        # key and returns the final group node. Empty path parts are skipped; existing
        # groups are reused. Returns the last visited node (a group), not created again
        # if it already existed.
        """Create nodes from key and return group name."""
        # assertion for mypy
        assert self._handle is not None

        paths = key.split("/")
        # recursively create the groups
        path = "/"
        for p in paths:
            if not len(p):
                continue
            new_path = path
            if not path.endswith("/"):
                new_path += "/"
            new_path += p
            group = self.get_node(new_path)
            if group is None:
                group = self._handle.create_group(path, p)
            path = new_path
        return group


class TableIterator:
    # [CLASS DOC]
    # Lazy-result wrapper used by select methods. Depending on the arguments, getting
    # the result either performs a single read immediately or returns this object as
    # an iterator of chunks.
    # State: the owning store and storer, the read function (taking start, stop and
    # where or coordinates), the where selection, total rows, start and stop bounds
    # (defaulted for tables), the chunk size (none when not iterating; when
    # iteration is requested without a size the default is one hundred thousand rows),
    # the coordinates list (set when iterating), and the auto-close flag.
    # Key methods: get_result, iteration, close.
    """
    Define the iteration interface on a table

    Parameters
    ----------
    store : HDFStore
    s     : the referred storer
    func  : the function to execute the query
    where : the where of the query
    nrows : the rows to iterate on
    start : the passed start value (default is None)
    stop  : the passed stop value (default is None)
    iterator : bool, default False
        Whether to use the default iterator.
    chunksize : the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    chunksize: int | None
    store: HDFStore
    s: GenericFixed | Table

    def __init__(
        self,
        store: HDFStore,
        s: GenericFixed | Table,
        func,
        where,
        nrows,
        start=None,
        stop=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the arguments. For table storers, normalizes the bounds: a missing row
        # count becomes zero, start defaults to zero, stop defaults to the row count and is
        # capped by it. Sets the chunk size to an integer: the given size, or the default
        # of one hundred thousand rows when iterator is true without a size; none when
        # neither iterator nor a chunk size is requested. The coordinates start out unset.
        self.store = store
        self.s = s
        self.func = func
        self.where = where

        # set start/stop if they are not set if we are a table
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)

        self.nrows = nrows
        self.start = start
        self.stop = stop

        self.coordinates = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize = int(chunksize)
        else:
            self.chunksize = None

        self.auto_close = auto_close

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Generator over chunks. Requires that get_result has been called (value error
        # otherwise because the coordinates are unset). Walks the stored coordinates from
        # start to stop in steps of the chunk size, calling the read function with the
        # coordinate slice for each chunk and yielding non-empty results (empty or none
        # chunks are skipped). When exhausted, closes the store if auto-close was set.
        # Note that closing happens only after complete consumption, so a second
        # iteration after an auto-closing run fails.
        # iterate
        current = self.start
        if self.coordinates is None:
            raise ValueError("Cannot iterate until get_result is called.")
        while current < self.stop:
            stop = min(current + self.chunksize, self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue

            yield value

        self.close()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the store if auto-close is enabled; otherwise does nothing.
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates: bool = False):
        # [FUNCTION DOC]
        # Produces the result of the selection.
        # Parameter coordinates (default false): if true, evaluates where into row
        # coordinates first (needed for multi-table reads).
        # Behavior: if a chunk size is set, requires a table storer (type error
        # otherwise), computes the coordinates of rows matching where over the whole table,
        # stores them, and returns this iterator object without closing. Otherwise
        # optionally converts where to coordinates over the start/stop range (type error
        # if not a table), calls the read function once for the start and stop bounds,
        # closes the store if auto-close, and returns the single result object.
        #  return the actual iterator
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError("can only use an iterator or chunksize on a table")

            self.coordinates = self.s.read_coordinates(where=self.where)

            return self

        # if specified read via coordinates (necessary for multiple selections
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError("can only read_coordinates on a table")
            where = self.s.read_coordinates(
                where=self.where, start=self.start, stop=self.stop
            )
        else:
            where = self.where

        # directly return the result
        results = self.func(self.start, self.stop, where)
        self.close()
        return results


class IndexCol:
    # [CLASS DOC]
    # Description of one indexable axis (index) column of a table-format object:
    # where its values, type and metadata are kept, and how to convert it to and from
    # the stored form.
    # State: converted values array, the kind text (such as string, integer,
    # datetime64, period, and so on), the PyTables column type, the name and the
    # column name in the table (defaults to the name), the axis number, the position in
    # the table, optional frequency, time zone, index name, categorical ordering flag,
    # the owning table, and the metadata and meta kind. Class attributes mark it as
    # indexable and as a data-indexable column; the info fields persisted in table
    # attributes are frequency, time zone and index name.
    # Subclasses: GenericIndexCol and DataCol (itself the base of the indexable data
    # column classes).
    # Equality compares name, column name, axis and position only.
    """
    an index column description class

    Parameters
    ----------
    axis   : axis which I reference
    values : the ndarray like converted values
    kind   : a string description of this type
    typ    : the pytables type
    pos    : the position in the pytables

    """

    is_an_indexable: bool = True
    is_data_indexable: bool = True
    _info_fields = ["freq", "tz", "index_name"]

    def __init__(
        self,
        name: str,
        values=None,
        kind=None,
        typ=None,
        cname: str | None = None,
        axis=None,
        pos=None,
        freq=None,
        tz=None,
        index_name=None,
        ordered=None,
        table=None,
        meta=None,
        metadata=None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the arguments as attributes. The name is required to be text (value error
        # otherwise); the column name defaults to the name. If a position is given, sets
        # the position on the column type too. Parameters beyond the name are all optional
        # and default to none.
        if not isinstance(name, str):
            raise ValueError("`name` must be a str.")

        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata

        if pos is not None:
            self.set_pos(pos)

        # These are ensured as long as the passed arguments match the
        #  constructor annotations.
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self) -> int:
        # [FUNCTION DOC]
        # Property: the byte size of one element of the column type (assumes the type was
        # set).
        # Assumes self.typ has already been initialized
        return self.typ.itemsize

    @property
    def kind_attr(self) -> str:
        # [FUNCTION DOC]
        # Property: name of the table attribute holding the kind of this column: the name
        # followed by the kind suffix.
        return f"{self.name}_kind"

    def set_pos(self, pos: int) -> None:
        # [FUNCTION DOC]
        # Sets the column position and, if a column type exists, also sets the position on
        # that type.
        """set the position of this column in the Table"""
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Text representation: comma-separated name, column name, axis, position and kind,
        # each formatted as a label arrow value.
        temp = tuple(
            map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind))
        )
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(
                    ["name", "cname", "axis", "pos", "kind"], temp, strict=True
                )
            ]
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality: true if the other object has the same name, column name, axis and
        # position (missing attributes compare as none). Because equality is defined
        # without a hash, instances are unhashable.
        """compare 2 col items"""
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "axis", "pos"]
        )

    def __ne__(self, other) -> bool:
        # [FUNCTION DOC]
        # Inequality: the negation of equality.
        return not self.__eq__(other)

    @property
    def is_indexed(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if the table column is indexed by PyTables. Returns false if the
        # table is not yet set or has no columns attribute.
        """return whether I am an indexed column"""
        if not hasattr(self.table, "cols"):
            # e.g. if infer hasn't been called yet, self.table will be None.
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ) -> tuple[np.ndarray, np.ndarray] | tuple[Index, Index]:
        # [FUNCTION DOC]
        # Converts the raw stored values of this index column into a pandas Index.
        # Parameters: values is a numpy array (asserted), possibly a record array from
        # which this column's field is copied; nan_rep and encoding/errors drive string
        # decoding via the shared conversion helper using the column kind. Note nan_rep
        # is accepted but not used here.
        # Behavior: builds keyword arguments with the index name and the frequency if set.
        # Chooses a datetime index when values are datetime64 or timezone-aware, a period
        # index built from integer ordinals when the values are 64-bit integers and a
        # frequency is known, else the generic index. If constructing raises a unicode
        # encode error caused by surrogates when errors is surrogatepass, the string dtype
        # option is on and PyArrow is installed, retries with a python-storage string
        # dtype with NaN missing value; other encode errors propagate. If construction
        # raises a value error, retries with the frequency cleared. Finally, for a
        # datetime index with a stored time zone, localizes the values as UTC and converts
        # to that zone.
        # Returns a pair holding the same Index object twice (once as index, once as
        # values).
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)

        # values is a recarray
        if values.dtype.fields is not None:
            # Copy, otherwise values will be a view
            # preventing the original recarry from being free'ed
            values = values[self.cname].copy()

        val_kind = self.kind
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs = {}
        kwargs["name"] = self.index_name

        if self.freq is not None:
            kwargs["freq"] = self.freq

        factory: type[Index | DatetimeIndex] = Index
        if lib.is_np_dtype(values.dtype, "M") or isinstance(
            values.dtype, DatetimeTZDtype
        ):
            factory = DatetimeIndex
        elif values.dtype == "i8" and "freq" in kwargs:
            # PeriodIndex data is stored as i8
            # error: Incompatible types in assignment (expression has type
            # "Callable[[Any, KwArg(Any)], PeriodIndex]", variable has type
            # "Union[Type[Index], Type[DatetimeIndex]]")
            factory = lambda x, **kwds: PeriodIndex.from_ordinals(  # type: ignore[assignment]
                x, freq=kwds.get("freq", None)
            )._rename(kwds["name"])

        # making an Index instance could throw a number of different errors
        try:
            new_pd_index = factory(values, **kwargs)
        except UnicodeEncodeError as err:
            if (
                errors == "surrogatepass"
                and using_string_dtype()
                and str(err).endswith("surrogates not allowed")
                and HAS_PYARROW
            ):
                new_pd_index = factory(
                    values,
                    dtype=StringDtype(storage="python", na_value=np.nan),
                    **kwargs,
                )
            else:
                raise
        except ValueError:
            # if the output freq is different that what we recorded,
            # it should be None (see also 'doc example part 2')
            if "freq" in kwargs:
                kwargs["freq"] = None
            new_pd_index = factory(values, **kwargs)

        final_pd_index: Index
        if self.tz is not None and isinstance(new_pd_index, DatetimeIndex):
            final_pd_index = new_pd_index.tz_localize("UTC").tz_convert(self.tz)
        else:
            final_pd_index = new_pd_index
        return final_pd_index, final_pd_index

    def take_data(self):
        # [FUNCTION DOC]
        # Returns the stored values array of this column.
        """return the values"""
        return self.values

    @property
    def attrs(self):
        # [FUNCTION DOC]
        # Property: the attribute set of the owning table.
        return self.table._v_attrs

    @property
    def description(self):
        # [FUNCTION DOC]
        # Property: the column description structure of the owning table.
        return self.table.description

    @property
    def col(self):
        # [FUNCTION DOC]
        # Property: this column's description entry from the table, or none if absent.
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self):
        # [FUNCTION DOC]
        # Property: the values to hand to the compiled layer; for this class simply the
        # values array.
        """return my cython values"""
        return self.values

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Iterates over the values array.
        return iter(self.values)

    def maybe_set_size(self, min_itemsize=None) -> None:
        # [FUNCTION DOC]
        # Possibly enlarges the column type for string columns. Only acts when the kind is
        # string. Parameter min_itemsize is an integer, or a dictionary keyed by column
        # name (the entry for this column is used), or none. If the value is not none and
        # bigger than the current size, replaces the type with a string column type of
        # that size at the same position.
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        if self.kind == "string":
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)

            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    def validate_names(self) -> None:
        # [FUNCTION DOC]
        # Placeholder for subclasses that need to validate names; does nothing here.
        pass

    def validate_and_set(self, handler: AppendableTable, append: bool) -> None:
        # [FUNCTION DOC]
        # Binds this column to the handler's table and runs, in order, the column
        # validation (string size), attribute validation (kind compatibility on append),
        # metadata validation and writing, and then writes the column attributes.
        # Parameters: handler is the appendable table being written; append says whether
        # appending to existing data.
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize=None):
        # [FUNCTION DOC]
        # HELPER: validates string columns against the existing table column.
        # Parameter itemsize is the width needed (defaults to this column's width). If the
        # kind is string and the table already has this column and the existing column
        # is narrower than needed, raises a value error advising the use of the minimum
        # size option. Otherwise returns the existing column width for string columns,
        # and none if not applicable.
        """validate this column: return the compared against itemsize"""
        # validate this column for string truncation (or reset to the max size)
        if self.kind == "string":
            c = self.col
            if c is not None:
                if itemsize is None:
                    itemsize = self.itemsize
                if c.itemsize < itemsize:
                    raise ValueError(
                        f"Trying to store a string with len [{itemsize}] in "
                        f"[{self.cname}] column but\nthis column has a limit of "
                        f"[{c.itemsize}]!\nConsider using min_itemsize to "
                        "preset the sizes on these columns"
                    )
                return c.itemsize

        return None

    def validate_attr(self, append: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: when appending, checks that a previously stored kind attribute for this
        # column equals the current kind; raises a type error on mismatch. Does nothing
        # when not appending or no kind was stored.
        # check for backwards incompatibility
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(
                    f"incompatible kind in col [{existing_kind} - {self.kind}]"
                )

    def update_info(self, info) -> None:
        # [FUNCTION DOC]
        # Records this column's metadata (the info fields listed on the class: for index
        # columns the frequency, time zone and index name) into the shared info mapping
        # that is stored on the table, detecting conflicts with what is already stored.
        # Parameter info is a dictionary keyed by column name whose values are
        # dictionaries of field to value; an entry for this column is created if absent.
        # For each field: if the stored value exists, differs from the current one, and
        # the current one is not none, then for frequency or index name emits an
        # attribute-conflict warning and resets both the stored value and this column's
        # attribute to none; for other fields raises a value error. Otherwise, if either
        # side is not none, stores the current value.
        # Side effects: mutates info, may mutate this column, may warn.
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})

            existing_value = idx.get(key)
            if key in idx and value is not None and existing_value != value:
                # frequency/name just warn
                if key in ["freq", "index_name"]:
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    warnings.warn(
                        ws, AttributeConflictWarning, stacklevel=find_stack_level()
                    )

                    # reset
                    idx[key] = None
                    setattr(self, key, None)

                else:
                    raise ValueError(
                        f"invalid info for [{self.name}] for [{key}], "
                        f"existing_value [{existing_value}] conflicts with "
                        f"new value [{value}]"
                    )
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info) -> None:
        # [FUNCTION DOC]
        # Restores this column's state from the info mapping: if there is an entry for
        # this column's name, copies all its fields onto the instance attributes. Does
        # nothing if there is no entry.
        """set my state from the passed info"""
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self) -> None:
        # [FUNCTION DOC]
        # Writes the kind of this column into the table attributes under the kind
        # attribute name.
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler: AppendableTable) -> None:
        # [FUNCTION DOC]
        # For categorical columns (meta kind category), checks that the categories to
        # append equal those already stored on the handler's table (compared with missing
        # values treated strictly and dtypes assumed equal). Raises a value error if both
        # exist and differ. Does nothing for other kinds.
        """validate that kind=category does not change the categories"""
        if self.meta == "category":
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if (
                new_metadata is not None
                and cur_metadata is not None
                and not array_equivalent(
                    new_metadata, cur_metadata, strict_nan=True, dtype_equal=True
                )
            ):
                raise ValueError(
                    "cannot append a categorical with "
                    "different categories to the existing"
                )

    def write_metadata(self, handler: AppendableTable) -> None:
        # [FUNCTION DOC]
        # Writes this column's metadata (such as categorical categories) through the
        # handler, under the column name, if any metadata exists.
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)


class GenericIndexCol(IndexCol):
    # [CLASS DOC]
    # Subclass of the index column for an index that is not stored in the table data;
    # it is regenerated on read as a default integer range index of the same length.
    # It is never indexed and writes no attributes.
    """an index which is not represented in the data of the table"""

    @property
    def is_indexed(self) -> bool:
        # [FUNCTION DOC]
        # Property: always false.
        return False

    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ) -> tuple[Index, Index]:
        # [FUNCTION DOC]
        # Builds a default range index with the same length as the values array and
        # returns it twice (as index and as values). The other arguments (missing
        # representation, encoding, errors) are unused. The values argument must be a
        # numpy array (asserted).
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep : str
        encoding : str
        errors : str
        """
        assert isinstance(values, np.ndarray), type(values)

        index = RangeIndex(len(values))
        return index, index

    def set_attr(self) -> None:
        # [FUNCTION DOC]
        # Overrides the attribute writer to do nothing, since nothing about a generic
        # index is stored.
        pass


class DataCol(IndexCol):
    # [CLASS DOC]
    # Subclass of the index column describing a data-holding column (or block of
    # columns) of a table; by definition not an indexable axis. The info fields it
    # persists are time zone and the ordering flag of categoricals.
    # State beyond the index column: the dtype name (text), and the data array
    # (the values to be stored, or after reading the converted result). Inherited
    # fields are reused with a different meaning: the values field holds the list of
    # column labels in the block; the kind attribute name stores those labels on the
    # table.
    # Key behavior: setting data computes the stored dtype and kind; conversion
    # reconstructs pandas arrays from the stored form; class methods create the
    # PyTables column type (atom) for a given kind. Equality compares name, column
    # name, dtype and position.
    """
    a data holding column, by definition this is not indexable

    Parameters
    ----------
    data   : the actual data
    cname  : the column name in the table to hold the data (typically
                values)
    meta   : a string description of the metadata
    metadata : the actual metadata
    """

    is_an_indexable = False
    is_data_indexable = False
    _info_fields = ["tz", "ordered"]

    def __init__(
        self,
        name: str,
        values=None,
        kind=None,
        typ=None,
        cname: str | None = None,
        pos=None,
        tz=None,
        ordered=None,
        table=None,
        meta=None,
        metadata=None,
        dtype: DtypeArg | None = None,
        data=None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the arguments by delegating the shared ones to the parent constructor
        # (name, values, kind, type, column name, position, time zone, ordered flag, table,
        # meta kind and metadata) and storing the dtype name and the data array.
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            pos=pos,
            cname=cname,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )
        self.dtype = dtype
        self.data = data

    @property
    def dtype_attr(self) -> str:
        # [FUNCTION DOC]
        # Property: name of the table attribute that stores this column's dtype name (the
        # column name followed by the dtype suffix).
        return f"{self.name}_dtype"

    @property
    def meta_attr(self) -> str:
        # [FUNCTION DOC]
        # Property: name of the table attribute that stores the meta kind (the column
        # name followed by the meta suffix).
        return f"{self.name}_meta"

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Text representation: comma-separated name, column name, dtype, kind and data
        # shape as label arrow value pairs.
        temp = tuple(
            map(
                pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)
            )
        )
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(
                    ["name", "cname", "dtype", "kind", "shape"], temp, strict=True
                )
            ]
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality: true if the other object has the same name, column name, dtype and
        # position (missing attributes compare as none). Defined without a hash, so
        # instances are unhashable.
        """compare 2 col items"""
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "dtype", "pos"]
        )

    def set_data(self, data: ArrayLike) -> None:
        # [FUNCTION DOC]
        # Sets the data to be stored. The argument must not be none, and the column must
        # not already have a dtype (both asserted). Converts the array to its storable
        # form together with a dtype name via the shared helper, then sets data, dtype
        # name and the kind derived from the dtype name.
        assert data is not None
        assert self.dtype is None

        data, dtype_name = _get_data_and_dtype_name(data)

        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def take_data(self):
        # [FUNCTION DOC]
        # Returns the data array (rather than the values) of this column.
        """return the data"""
        return self.data

    @classmethod
    def _get_atom(cls, values: ArrayLike) -> Col:
        # [FUNCTION DOC]
        # HELPER: builds the PyTables column type description for an array of values.
        # Class method; input values is a numpy or extension array (one-dimensional arrays
        # are treated as a block of shape one by size). Chooses by the array type:
        # categoricals use an integer type matching the codes dtype; datetimes and
        # timedeltas use the 64-bit integer type; complex uses a complex column of the
        # item size; string dtypes use a string column of the item size; everything else
        # uses the numeric type named after the dtype. The shape of each column type is
        # the first dimension of the block shape (number of items).
        # Returns the column type object.
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        dtype = values.dtype
        # error: Item "ExtensionDtype" of "Union[ExtensionDtype, dtype[Any]]" has no
        # attribute "itemsize"
        itemsize = dtype.itemsize  # type: ignore[union-attr]

        shape = values.shape
        if values.ndim == 1:
            # EA, use block shape pretending it is 2D
            # TODO(EA2D): not necessary with 2D EAs
            shape = (1, values.size)

        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            atom = cls.get_atom_datetime64(shape)
        elif lib.is_np_dtype(dtype, "m"):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)

        return atom

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        # [FUNCTION DOC]
        # HELPER: class method creating a string column type of the given item size with
        # shape given by the first dimension of the block shape. Overridden in the
        # indexable subclass to omit the shape.
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]:
        # [FUNCTION DOC]
        # HELPER: class method mapping a dtype kind text to a PyTables column class from
        # the lazily imported tables module. Unsigned integer kinds map to the matching
        # unsigned column of that bit width; period kinds map to the 64-bit integer
        # column; every other kind maps to the column class named by capitalizing the kind
        # text. A kind with no such class raises an attribute error.
        """return the PyTables column class for this column"""
        if kind.startswith("uint"):
            k4 = kind[4:]
            col_name = f"UInt{k4}Col"
        elif kind.startswith("period"):
            # we store as integer
            col_name = "Int64Col"
        else:
            kcap = kind.capitalize()
            col_name = f"{kcap}Col"

        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        # [FUNCTION DOC]
        # HELPER: class method creating a numeric (non-string) column type for the kind,
        # with shape given by the first dimension of the block shape. Overridden in the
        # indexable subclass to omit the shape.
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape):
        # [FUNCTION DOC]
        # HELPER: class method returning a 64-bit integer column type with the first
        # dimension of the shape, used to store datetime values as integers.
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape):
        # [FUNCTION DOC]
        # HELPER: class method returning a 64-bit integer column type with the first
        # dimension of the shape, used to store timedelta values as integers.
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self):
        # [FUNCTION DOC]
        # Property: the shape of the data array, or none if there is no data or it has no
        # shape.
        return getattr(self.data, "shape", None)

    @property
    def cvalues(self):
        # [FUNCTION DOC]
        # Property: the data array (instead of the values) handed to the compiled layer.
        """return my cython values"""
        return self.data

    def validate_attr(self, append) -> None:
        # [FUNCTION DOC]
        # When appending, checks the new block against the stored one. Raises a value
        # error if the stored column labels (kept under the kind attribute) differ from
        # this block's labels in value or order, or if the stored dtype differs from this
        # dtype. Does nothing when not appending or when nothing was stored.
        """validate that we have the same order as the existing & same dtype"""
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError("appended items do not match existing items in table!")

            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError(
                    "appended items dtype do not match existing items dtype in table!"
                )

    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str):
        # [FUNCTION DOC]
        # Converts the raw stored array of this data block back to pandas form.
        # Parameters: values is a numpy array (asserted), possibly a record array from
        # which this column's field is taken; nan_rep is the string used for missing text
        # values; encoding and errors control decoding of string data.
        # Behavior: if no dtype is known, infers dtype and kind from the values; else uses
        # the stored dtype and kind. Then reverses the storage conversion by dtype:
        # datetime types are re-interpreted (with an old unitless name meaning nanoseconds)
        # and localized to the stored time zone; timedelta types are viewed with their unit
        # (unitless old storage means nanoseconds); a date dtype is rebuilt as an object
        # array of dates, first assuming ordinals and falling back to timestamps on a
        # value error; categorical blocks are rebuilt from the integer codes and the stored
        # categories (a missing-value category is stripped and codes shifted, and a
        # categorical with no categories gets an empty float category list), with the
        # stored ordering flag and without validation; all other types are cast to the
        # stored dtype, falling back to object on a type error. Finally, for string kind,
        # the array is decoded and the missing representation converted back to missing
        # values.
        # Returns a pair: the column labels (the values field) and the converted array.
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep :
        encoding : str
        errors : str

        Returns
        -------
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
        assert isinstance(values, np.ndarray), type(values)

        # values is a recarray
        if values.dtype.fields is not None:
            values = values[self.cname]

        assert self.typ is not None
        if self.dtype is None:
            # Note: in tests we never have timedelta64 or datetime64,
            #  so the _get_data_and_dtype_name may be unnecessary
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype
            kind = self.kind

        assert isinstance(converted, np.ndarray)  # for mypy

        # use the meta if needed
        meta = self.meta
        metadata = self.metadata
        ordered = self.ordered
        tz = self.tz

        assert dtype_name is not None
        # convert to the correct dtype
        dtype = dtype_name

        # reverse converts
        if dtype.startswith("datetime64"):
            # recreate with tz if indicated
            if dtype == "datetime64":
                dtype = "datetime64[ns]"
            converted = _set_tz(converted, tz, dtype)

        elif dtype.startswith("timedelta64"):
            if dtype == "timedelta64":
                # from before we started storing timedelta64 unit
                converted = np.asarray(converted, dtype="m8[ns]")
            else:
                converted = np.asarray(converted, dtype=dtype)
        elif dtype == "date":
            try:
                converted = np.asarray(
                    [date.fromordinal(v) for v in converted], dtype=object
                )
            except ValueError:
                converted = np.asarray(
                    [date.fromtimestamp(v) for v in converted], dtype=object
                )

        elif meta == "category":
            # we have a categorical
            categories = metadata
            codes = converted.ravel()

            # if we have stored a NaN in the categories
            # then strip it; in theory we could have BOTH
            # -1s in the codes and nulls :<
            if categories is None:
                # Handle case of NaN-only categorical columns in which case
                # the categories are an empty array; when this is stored,
                # pytables cannot write a zero-len array, so on readback
                # the categories would be None and `read_hdf()` would fail.
                categories = Index([], dtype=np.float64)
            else:
                mask = isna(categories)
                if mask.any():
                    categories = categories[~mask]
                    codes[codes != -1] -= mask.astype(int).cumsum()._values

            converted = Categorical.from_codes(
                codes, categories=categories, ordered=ordered, validate=False
            )

        else:
            try:
                converted = converted.astype(dtype, copy=False)
            except TypeError:
                converted = converted.astype("O", copy=False)

        # convert nans / decode
        if kind == "string":
            converted = _unconvert_string_array(
                converted, nan_rep=nan_rep, encoding=encoding, errors=errors
            )

        return self.values, converted

    def set_attr(self) -> None:
        # [FUNCTION DOC]
        # Writes the column labels (under the kind attribute name), the meta kind and the
        # dtype name into the table attributes. The dtype must have been set (asserted).
        """set the data for this column"""
        setattr(self.attrs, self.kind_attr, self.values)
        setattr(self.attrs, self.meta_attr, self.meta)
        assert self.dtype is not None
        setattr(self.attrs, self.dtype_attr, self.dtype)


class DataIndexableCol(DataCol):
    # [CLASS DOC]
    # Subclass of the data column for a data column that can itself be indexed and
    # queried (a user-selected data column, stored as its own table column). Differs
    # from the plain data column in that the column types are created without the
    # block shape (one value per row) and labels must be strings.
    """represent a data column that can be indexed"""

    is_data_indexable = True

    def validate_names(self) -> None:
        # [FUNCTION DOC]
        # Checks that the column labels are strings; raises a value error if they are not
        # (for example integers).
        if not is_string_dtype(Index(self.values).dtype):
            # TODO: should the message here be more specifically non-str?
            raise ValueError("cannot have non-object label DataIndexableCol")

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        # [FUNCTION DOC]
        # HELPER: class method returning a string column type of the given item size
        # without a shape.
        return _tables().StringCol(itemsize=itemsize)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        # [FUNCTION DOC]
        # HELPER: class method returning a column type for the kind without a shape.
        return cls.get_atom_coltype(kind=kind)()

    @classmethod
    def get_atom_datetime64(cls, shape):
        # [FUNCTION DOC]
        # HELPER: class method returning a 64-bit integer column type without a shape.
        return _tables().Int64Col()

    @classmethod
    def get_atom_timedelta64(cls, shape):
        # [FUNCTION DOC]
        # HELPER: class method returning a 64-bit integer column type without a shape.
        return _tables().Int64Col()


class GenericDataIndexableCol(DataIndexableCol):
    # [CLASS DOC]
    # Subclass of the indexable data column used for data columns when reading a
    # generic PyTables table that was not written by pandas. No extra state or behavior.
    """represent a generic pytables data column"""


class Fixed:
    # [CLASS DOC]
    # Abstract base class for a storer: the object that knows how to read and write
    # one pandas object (series or frame) located at a group node of the file.
    # State: the owning store, the group node, the string encoding (default UTF-8) and
    # the error handling mode. Class attributes: the pandas kind text (set by
    # subclasses), the format type (fixed; needed by external library dask), the
    # object type, the dimension count, and a flag saying whether it is a table.
    # Derived properties expose the file handle, filters, compression level and
    # checksum flag of the owning store, the group attributes and path, row count,
    # shape and version tuple.
    # Concrete methods here are mostly stubs: read and write raise a not-implemented
    # error; delete only supports deleting the whole node.
    """
    represent an object in my store
    facilitate read/write of various types of objects
    this is an abstract base class

    Parameters
    ----------
    parent : HDFStore
    group : Node
        The group node where the table resides.
    """

    pandas_kind: str
    format_type: str = "fixed"  # GH#30962 needed by dask
    obj_type: type[DataFrame | Series]
    ndim: int
    parent: HDFStore
    is_table: bool = False

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: str | None = "UTF-8",
        errors: str = "strict",
    ) -> None:
        # [FUNCTION DOC]
        # Stores the arguments. The parent must be a store, the group must be a PyTables
        # node, and the tables module must already be imported (all asserted). The
        # encoding is normalized to the module default when none. The errors argument is
        # the text error-handling mode.
        assert isinstance(parent, HDFStore), type(parent)
        assert _table_mod is not None  # needed for mypy
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent = parent
        self.group = group
        self.encoding = _ensure_encoding(encoding)
        self.errors = errors

    @property
    def is_old_version(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if the stored pandas version tuple is below 0.10.1 (that is, an
        # old file format), false otherwise.
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[2] < 1

    @property
    def version(self) -> tuple[int, int, int]:
        # [FUNCTION DOC]
        # Property: the pandas version recorded in the group attributes as a three-integer
        # tuple. A two-part version gets a trailing zero. A missing or non-text version
        # yields all zeros.
        # # KNOWN ISSUE: the docstring says the version is computed and set, but the
        # property only computes and returns it, without storing it.
        """compute and set our version"""
        version = getattr(self.group._v_attrs, "pandas_version", None)
        if isinstance(version, str):
            version_tup = tuple(int(x) for x in version.split("."))
            if len(version_tup) == 2:
                version_tup = (*version_tup, 0)
            assert len(version_tup) == 3  # needed for mypy
            return version_tup
        else:
            return (0, 0, 0)

    @property
    def pandas_type(self):
        # [FUNCTION DOC]
        # Property: the pandas type text recorded in the group attributes, or none.
        return getattr(self.group._v_attrs, "pandas_type", None)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Pretty text of the storer: infers the axes first, then formats the pandas type
        # padded and truncated to twelve characters followed by the shape (a list or tuple
        # shape is shown as comma-separated items in brackets). If no shape, returns just
        # the type.
        # Side effect: calls the axes inference (reads attributes).
        """return a pretty representation of myself"""
        self.infer_axes()
        s = self.shape
        if s is not None:
            if isinstance(s, (list, tuple)):
                jshape = ",".join([pprint_thing(x) for x in s])
                s = f"[{jshape}]"
            return f"{self.pandas_type:12.12} (shape->{s})"
        return self.pandas_type

    def set_object_info(self) -> None:
        # [FUNCTION DOC]
        # Writes the pandas kind and module version text into the group attributes.
        """set my pandas type & version"""
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    def copy(self) -> Fixed:
        # [FUNCTION DOC]
        # Returns a shallow copy of this storer.
        new_self = copy.copy(self)
        return new_self

    @property
    def shape(self):
        # [FUNCTION DOC]
        # Property: the shape of the stored object; in the base class just the row count.
        return self.nrows

    @property
    def pathname(self):
        # [FUNCTION DOC]
        # Property: the absolute path name of the group.
        return self.group._v_pathname

    @property
    def _handle(self):
        # [FUNCTION DOC]
        # Property: the PyTables file handle of the owning store.
        return self.parent._handle

    @property
    def _filters(self):
        # [FUNCTION DOC]
        # Property: the PyTables filters object of the owning store.
        return self.parent._filters

    @property
    def _complevel(self) -> int:
        # [FUNCTION DOC]
        # Property: the compression level of the owning store.
        return self.parent._complevel

    @property
    def _fletcher32(self) -> bool:
        # [FUNCTION DOC]
        # Property: the checksum flag of the owning store.
        return self.parent._fletcher32

    @property
    def attrs(self):
        # [FUNCTION DOC]
        # Property: the attribute set of the group node.
        return self.group._v_attrs

    def set_attrs(self) -> None:
        # [FUNCTION DOC]
        # Hook for subclasses to write their own attributes; does nothing here.
        """set our object attributes"""

    def get_attrs(self) -> None:
        # [FUNCTION DOC]
        # Hook for subclasses to read their own attributes; does nothing here.
        """get our object attributes"""

    @property
    def storable(self):
        # [FUNCTION DOC]
        # Property: the node holding the data; here the group itself.
        """return my storable"""
        return self.group

    @property
    def is_exists(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether the storer refers to existing data; false in the base class.
        return False

    @property
    def nrows(self):
        # [FUNCTION DOC]
        # Property: the number of rows of the storable node, or none if it has no such
        # attribute.
        return getattr(self.storable, "nrows", None)

    def validate(self, other) -> Literal[True] | None:
        # [FUNCTION DOC]
        # Validates against an existing storable: returns none if the argument is none,
        # otherwise true. No other checks in the base class.
        """validate against an existing storable"""
        if other is None:
            return None
        return True

    def validate_version(self, where=None) -> None:
        # [FUNCTION DOC]
        # Hook to reject operating on an old file version; does nothing in the base class.
        # Parameter where is accepted for subclass use.
        """are we trying to operate on an old version?"""

    def infer_axes(self) -> bool:
        # [FUNCTION DOC]
        # Infers the axes of the storer from stored attributes. Returns false if there is
        # no storable node, otherwise reads the attributes (via the attribute hook) and
        # returns true.
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
        s = self.storable
        if s is None:
            return False
        self.get_attrs()
        return True

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series | DataFrame:
        # [FUNCTION DOC]
        # Abstract read: always raises a not-implemented error. Subclasses implement it
        # with the same parameters (where, columns, start, stop) and return a series or
        # frame.
        raise NotImplementedError(
            "cannot read on an abstract storer: subclasses should implement"
        )

    def write(self, obj, **kwargs) -> None:
        # [FUNCTION DOC]
        # Abstract write: always raises a not-implemented error. Subclasses implement it.
        raise NotImplementedError(
            "cannot write on an abstract storer: subclasses should implement"
        )

    def delete(
        self, where=None, start: int | None = None, stop: int | None = None
    ) -> int | None:
        # [FUNCTION DOC]
        # Deletes the stored node in its entirety. Only allowed when where, start and stop
        # are all none: removes the group recursively and returns none. Otherwise raises a
        # type error because partial deletes are not supported on an abstract storer.
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None

        raise TypeError("cannot delete on an abstract storer")


class GenericFixed(Fixed):
    # [CLASS DOC]
    # Base class for the fixed-format storers. Stores whole objects as plain arrays
    # under the group (not queryable, not appendable). Subclass of the abstract storer.
    # Class state: a map between index classes and short alias texts (datetime and
    # period) and its reverse, and the list of attribute names that subclasses
    # persist and restore.
    # Provides helpers to read and write arrays and indexes (regular and multi level)
    # in this layout, a factory builder for index classes, and the validation that
    # partial reads are not allowed.
    """a generified fixed version"""

    _index_type_map = {DatetimeIndex: "datetime", PeriodIndex: "period"}
    _reverse_index_map = {v: k for k, v in _index_type_map.items()}
    attributes: list[str] = []

    # indexer helpers
    def _class_to_alias(self, cls) -> str:
        # [FUNCTION DOC]
        # HELPER: maps an index class to its short alias (datetime or period) or an empty
        # text if it has none.
        return self._index_type_map.get(cls, "")

    def _alias_to_class(self, alias):
        # [FUNCTION DOC]
        # HELPER: maps a stored alias back to an index class. If the alias is already a
        # class (old compatibility behavior) it is returned unchanged; unknown aliases
        # return the generic index class.
        if isinstance(alias, type):  # pragma: no cover
            # compat: for a short period of time master stored types
            return alias
        return self._reverse_index_map.get(alias, Index)

    def _get_index_factory(self, attrs):
        # [FUNCTION DOC]
        # HELPER: builds the way to create an index from stored data.
        # Input attrs is the attribute set of the index node. Reads the stored index class
        # alias, and returns a pair of a callable factory and a dictionary of keyword
        # arguments. For datetime the factory wraps the stored UTC integer values into a
        # datetime index and, when a time zone is given, localizes to UTC then converts to
        # that zone. For period the factory builds a period index from the ordinals and the
        # frequency. Otherwise the factory is the index class itself, with copying
        # disabled. If a frequency attribute exists it is passed on, and when the class is
        # the generic index the factory becomes a timedelta index. If a time zone attribute
        # exists it is passed on (asserted to go with a datetime index).
        index_class = self._alias_to_class(getattr(attrs, "index_class", ""))

        factory: Callable

        kwargs = {}
        if index_class == DatetimeIndex:

            def f(values, freq=None, tz=None):
                # [FUNCTION DOC]
                # HELPER: nested factory for datetime indexes. Takes the stored integer values
                # object, an optional frequency and an optional time zone. Builds a datetime array
                # without validation from the underlying values, using the values dtype and
                # frequency, wraps it in an unnamed datetime index, and if a zone is given
                # localizes to UTC and converts to the zone. Returns the index.
                # data are already in UTC, localize and convert if tz present
                dta = DatetimeArray._simple_new(
                    values.values, dtype=values.dtype, freq=freq
                )
                result = DatetimeIndex._simple_new(dta, name=None)
                if tz is not None:
                    result = result.tz_localize("UTC").tz_convert(tz)
                return result

            factory = f
        elif index_class == PeriodIndex:

            def f(values, freq=None, tz=None):
                # [FUNCTION DOC]
                # HELPER: nested factory for period indexes. Takes the stored ordinal values, a
                # frequency, and an unused time zone argument. Builds a period dtype from the
                # frequency, an unnamed period array from the values and wraps it in a period
                # index without validation. Returns the index.
                dtype = PeriodDtype(freq)
                parr = PeriodArray._simple_new(values, dtype=dtype)
                return PeriodIndex._simple_new(parr, name=None)

            factory = f
        else:
            factory = index_class
            kwargs["copy"] = False

        if "freq" in attrs:
            kwargs["freq"] = attrs["freq"]
            if index_class is Index:
                # DTI/PI would be gotten by _alias_to_class
                factory = TimedeltaIndex

        if "tz" in attrs:
            kwargs["tz"] = attrs["tz"]
            assert index_class is DatetimeIndex  # just checking

        return factory, kwargs

    def validate_read(self, columns, where) -> None:
        # [FUNCTION DOC]
        # Raises a type error if a columns specification or a where selection is given,
        # because fixed-format objects can only be read in their entirety. Both checks are
        # for values that are not none.
        """
        raise if any keywords are passed which are not-None
        """
        if columns is not None:
            raise TypeError(
                "cannot pass a column specification when reading "
                "a Fixed format store. this store must be selected in its entirety"
            )
        if where is not None:
            raise TypeError(
                "cannot pass a where specification when reading "
                "from a Fixed format store. this store must be selected in its entirety"
            )

    @property
    def is_exists(self) -> bool:
        # [FUNCTION DOC]
        # Property: always true for fixed-format storers.
        return True

    def set_attrs(self) -> None:
        # [FUNCTION DOC]
        # Writes the encoding and error-handling mode into the group attributes.
        """set our object attributes"""
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def get_attrs(self) -> None:
        # [FUNCTION DOC]
        # Reads the encoding (default UTF-8) and error mode (default strict) from the
        # group attributes, and loads every attribute named in the class attribute list
        # onto this storer, using none for any that is absent.
        """retrieve our attributes"""
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        self.errors = getattr(self.attrs, "errors", "strict")
        for n in self.attributes:
            setattr(self, n, getattr(self.attrs, n, None))

    def write(self, obj, **kwargs) -> None:
        # [FUNCTION DOC]
        # Writes the object. At this level it only saves encoding and error attributes;
        # the obj and keyword arguments are ignored. Subclasses extend it.
        self.set_attrs()

    def read_array(self, key: str, start: int | None = None, stop: int | None = None):
        # [FUNCTION DOC]
        # Reads the array stored under a child node name.
        # Parameters: key is the child name; start and stop slice the rows.
        # Behavior: imports the tables module. For a variable-length array node it takes
        # the first element sliced, and if a value type attribute exists, converts to a
        # pandas array with that dtype. For other nodes: if a shape attribute exists (the
        # stored array had a zero-length axis) it creates an empty array of that shape
        # and value type, otherwise slices the node; then for datetime value types
        # (unitless old name meaning nanoseconds) rebuilds with the stored time zone, and
        # for timedelta types converts to that unit (unitless meaning nanoseconds).
        # If the node carries a transposed flag, returns the transpose.
        # Returns a numpy or pandas array.
        """read an array for the specified node (off of group"""
        import tables

        node = getattr(self.group, key)
        attrs = node._v_attrs

        transposed = getattr(attrs, "transposed", False)

        if isinstance(node, tables.VLArray):
            ret = node[0][start:stop]
            dtype = getattr(attrs, "value_type", None)
            if dtype is not None:
                ret = pd_array(ret, dtype=dtype)
        else:
            dtype = getattr(attrs, "value_type", None)
            shape = getattr(attrs, "shape", None)

            if shape is not None:
                # length 0 axis
                ret = np.empty(shape, dtype=dtype)
            else:
                ret = node[start:stop]

            if dtype and dtype.startswith("datetime64"):
                # reconstruct a timezone if indicated
                if dtype == "datetime64":
                    dtype = "datetime64[ns]"
                tz = getattr(attrs, "tz", None)
                ret = _set_tz(ret, tz, dtype)

            elif dtype and dtype.startswith("timedelta64"):
                if dtype == "timedelta64":
                    # This was written back before we started writing
                    # timedelta64 units
                    ret = np.asarray(ret, dtype="m8[ns]")
                else:
                    ret = np.asarray(ret, dtype=dtype)

        if transposed:
            return ret.T
        else:
            return ret

    def read_index(
        self, key: str, start: int | None = None, stop: int | None = None
    ) -> Index:
        # [FUNCTION DOC]
        # Reads an index stored under a child key. Parameters: key; start and stop slice.
        # Looks up the variety attribute stored with the key: a multi variety dispatches
        # to the multi level reader; a regular variety reads the node and builds the
        # index. Any other variety raises a type error. Returns the Index.
        variety = getattr(self.attrs, f"{key}_variety")

        if variety == "multi":
            return self.read_multi_index(key, start=start, stop=stop)
        elif variety == "regular":
            node = getattr(self.group, key)
            index = self.read_index_node(node, start=start, stop=stop)
            return index
        else:  # pragma: no cover
            raise TypeError(f"unrecognized index variety: {variety}")

    def write_index(self, key: str, index: Index) -> None:
        # [FUNCTION DOC]
        # Writes an index under a child key.
        # Parameters: key and index (any index).
        # For a multi-level index, records variety as multi and writes the levels. For
        # other indexes records variety as regular, converts the index with the shared
        # helper, writes the resulting values as an array, and on that node records the
        # kind and index name; datetime and period indexes record their alias; datetime,
        # period and timedelta indexes record their frequency; timezone-aware datetime
        # indexes record the time zone name.
        if isinstance(index, MultiIndex):
            setattr(self.attrs, f"{key}_variety", "multi")
            self.write_multi_index(key, index)
        else:
            setattr(self.attrs, f"{key}_variety", "regular")
            converted = _convert_index("index", index, self.encoding, self.errors)

            self.write_array(key, converted.values)

            node = getattr(self.group, key)
            node._v_attrs.kind = converted.kind
            node._v_attrs.name = index.name

            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                node._v_attrs.index_class = self._class_to_alias(type(index))

            if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
                node._v_attrs.freq = index.freq

            if isinstance(index, DatetimeIndex) and index.tz is not None:
                node._v_attrs.tz = _get_tz(index.tz)

    def write_multi_index(self, key: str, index: MultiIndex) -> None:
        # [FUNCTION DOC]
        # Writes a multi-level index under a key.
        # Records the number of levels. For each level (levels, codes, names in parallel)
        # raises a not-implemented error if the level has an extension dtype other than
        # a string dtype. Converts the level values, writes them under a key built from
        # the key and level number, records the kind and name on that node (and also an
        # extra name attribute combining key and name), and writes the integer codes under
        # a separate label key.
        setattr(self.attrs, f"{key}_nlevels", index.nlevels)

        for i, (lev, level_codes, name) in enumerate(
            zip(index.levels, index.codes, index.names, strict=True)
        ):
            # write the level
            if isinstance(lev.dtype, ExtensionDtype) and not isinstance(
                lev.dtype, StringDtype
            ):
                raise NotImplementedError(
                    "Saving a MultiIndex with an extension dtype is not supported."
                )
            level_key = f"{key}_level{i}"
            conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
            self.write_array(level_key, conv_level.values)
            node = getattr(self.group, level_key)
            node._v_attrs.kind = conv_level.kind
            node._v_attrs.name = name

            # write the name
            setattr(node._v_attrs, f"{key}_name{name}", name)

            # write the labels
            label_key = f"{key}_label{i}"
            self.write_array(label_key, level_codes)

    def read_multi_index(
        self, key: str, start: int | None = None, stop: int | None = None
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Reads a multi-level index written by the multi-level writer.
        # Parameters: key; start and stop slice every stored array.
        # Reads the level count attribute, then for each level reads the level index node
        # (taking its name from the node) and the integer codes array. Builds a MultiIndex
        # with integrity verification and returns it.
        nlevels = getattr(self.attrs, f"{key}_nlevels")

        levels = []
        codes = []
        names: list[Hashable] = []
        for i in range(nlevels):
            level_key = f"{key}_level{i}"
            node = getattr(self.group, level_key)
            lev = self.read_index_node(node, start=start, stop=stop)
            levels.append(lev)
            names.append(lev.name)

            label_key = f"{key}_label{i}"
            level_codes = self.read_array(label_key, start=start, stop=stop)
            codes.append(level_codes)

        return MultiIndex(
            levels=levels, codes=codes, names=names, verify_integrity=True
        )

    def read_index_node(
        self, node: Node, start: int | None = None, stop: int | None = None
    ) -> Index:
        # [FUNCTION DOC]
        # Builds an Index from a stored index node.
        # Parameters: node is the PyTables array node; start and stop slice the rows.
        # Behavior: reads the slice; if the node carries a shape attribute describing a
        # zero-size array (written by the empty-array hack) the data is replaced by an
        # empty array of that shape and stored value type. Reads the kind attribute and
        # the optional name (normalized to a plain string). Gets the index factory and
        # keyword arguments from the node attributes. The raw data is converted by the
        # shared helper according to kind, then handed to the factory (for date and object
        # kinds with an object dtype forced). For other kinds, a unicode encode error
        # caused by surrogates under surrogatepass error mode, with the string dtype
        # option on and PyArrow installed, is retried with a python-storage string dtype
        # (NaN as missing); other encode errors propagate. Sets the name on the result and
        # returns it.
        data = node[start:stop]
        # If the index was an empty array write_array_empty() will
        # have written a sentinel. Here we replace it with the original.
        if "shape" in node._v_attrs and np.prod(node._v_attrs.shape) == 0:
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        kind = node._v_attrs.kind
        name = None

        if "name" in node._v_attrs:
            name = _ensure_str(node._v_attrs.name)

        attrs = node._v_attrs
        factory, kwargs = self._get_index_factory(attrs)

        if kind in ("date", "object"):
            index = factory(
                _unconvert_index(
                    data, kind, encoding=self.encoding, errors=self.errors
                ),
                dtype=object,
                **kwargs,
            )
        else:
            try:
                index = factory(
                    _unconvert_index(
                        data, kind, encoding=self.encoding, errors=self.errors
                    ),
                    **kwargs,
                )
            except UnicodeEncodeError as err:
                if (
                    self.errors == "surrogatepass"
                    and using_string_dtype()
                    and str(err).endswith("surrogates not allowed")
                    and HAS_PYARROW
                ):
                    index = factory(
                        _unconvert_index(
                            data, kind, encoding=self.encoding, errors=self.errors
                        ),
                        dtype=StringDtype(storage="python", na_value=np.nan),
                        **kwargs,
                    )
                else:
                    raise

        index.name = name

        return index

    def write_array_empty(self, key: str, value: ArrayLike) -> None:
        # [FUNCTION DOC]
        # Writes a placeholder for an array with zero size. Parameters: key (child name)
        # and value (the empty array). Creates a one-element array with as many unit
        # dimensions as the value, then records the real dtype (as text) and shape on
        # the node attributes so the reader can recreate the empty array.
        """write a 0-len array"""
        # ugly hack for length 0 axes
        arr = np.empty((1,) * value.ndim)
        self._handle.create_array(self.group, key, arr)
        node = getattr(self.group, key)
        node._v_attrs.value_type = str(value.dtype)
        node._v_attrs.shape = value.shape

    def write_array(
        self, key: str, obj: AnyArrayLike, items: Index | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Writes an array (or the values of an index or series) under a child name.
        # Parameters: key is the child node name (an existing node of that name is
        # removed first); obj is the array-like to store (unwrapped to its array); items
        # is an optional index only used in the performance warning message.
        # Behavior: categorical dtype raises a not-implemented error (fixed format cannot
        # store it). Non-empty arrays are transposed when they support it (flag recorded).
        # Then by type: string extension arrays go to a variable-length array of objects
        # (with filters) with the dtype name recorded; otherwise, if compression filters
        # are configured and an atom type exists for the dtype, a chunked compressed array
        # is created and filled (empty arrays use the placeholder writer); object dtype
        # arrays are written as variable-length object arrays, with a performance warning
        # when the inferred type is not string, the array is not empty and the
        # performance-warnings option is on; naive datetimes and timedeltas are stored as
        # 64-bit integers with the dtype recorded; time-zone-aware datetimes are stored as
        # 64-bit UTC integers with zone name and unit-bearing dtype recorded; empty arrays
        # use the placeholder; anything else is stored directly. Finally records the
        # transposed flag on the node.
        # Side effects: creates and removes nodes, may warn.
        value = extract_array(obj, extract_numpy=True)

        if key in self.group:
            self._handle.remove_node(self.group, key)

        # Transform needed to interface with pytables row/col notation
        empty_array = value.size == 0
        transposed = False

        if isinstance(value.dtype, CategoricalDtype):
            raise NotImplementedError(
                "Cannot store a category dtype in an HDF5 dataset that uses format="
                '"fixed". Use format="table".'
            )
        if not empty_array:
            if hasattr(value, "T"):
                # ExtensionArrays (1d) may not have transpose.
                value = value.T
                transposed = True

        if isinstance(value, BaseStringArray):
            # GH#64180: BaseStringArray must use the VLArray path.
            # Atom.from_dtype does not handle ExtensionDtype.
            vlarr = self._handle.create_vlarray(
                self.group, key, _tables().ObjectAtom(), filters=self._filters
            )
            vlarr.append(value.to_numpy())
            node = getattr(self.group, key)
            node._v_attrs.value_type = str(value.dtype)

        else:
            atom = None
            if self._filters is not None:
                with suppress(ValueError):
                    # get the atom for this datatype
                    atom = _tables().Atom.from_dtype(value.dtype)

            if atom is not None:
                # We only get here if self._filters is non-None and
                #  the Atom.from_dtype call succeeded

                # create an empty chunked array and fill it from value
                if not empty_array:
                    ca = self._handle.create_carray(
                        self.group, key, atom, value.shape, filters=self._filters
                    )
                    ca[:] = value

                else:
                    self.write_array_empty(key, value)

            elif value.dtype.type == np.object_:
                # infer the type, warn if we have a non-string type here
                # (for performance)
                inferred_type = lib.infer_dtype(value, skipna=False)
                if empty_array:
                    pass
                elif inferred_type == "string":
                    pass
                elif get_option("performance_warnings"):
                    ws = performance_doc % (inferred_type, key, items)
                    warnings.warn(ws, PerformanceWarning, stacklevel=find_stack_level())

                vlarr = self._handle.create_vlarray(
                    self.group, key, _tables().ObjectAtom()
                )
                vlarr.append(value)

            elif lib.is_np_dtype(value.dtype, "M"):
                self._handle.create_array(self.group, key, value.view("i8"))
                getattr(self.group, key)._v_attrs.value_type = str(value.dtype)
            elif isinstance(value.dtype, DatetimeTZDtype):
                # store as UTC
                # with a zone

                # error: "ExtensionArray" has no attribute "asi8"
                self._handle.create_array(
                    self.group,
                    key,
                    value.asi8,  # type: ignore[attr-defined]
                )

                node = getattr(self.group, key)
                # error: "ExtensionArray" has no attribute "tz"
                node._v_attrs.tz = _get_tz(value.tz)  # type: ignore[attr-defined]
                node._v_attrs.value_type = f"datetime64[{value.dtype.unit}]"
            elif lib.is_np_dtype(value.dtype, "m"):
                self._handle.create_array(self.group, key, value.view("i8"))
                getattr(self.group, key)._v_attrs.value_type = str(value.dtype)
            elif empty_array:
                self.write_array_empty(key, value)
            else:
                self._handle.create_array(self.group, key, value)

        getattr(self.group, key)._v_attrs.transposed = transposed


class SeriesFixed(GenericFixed):
    # [CLASS DOC]
    # Fixed-format storer for a Series. Stores the index under the name index and the
    # values under the name values, and the series name as a group attribute.
    # Class state: pandas kind is series; persisted attribute list holds the name.
    pandas_kind = "series"
    attributes = ["name"]

    name: Hashable

    @property
    def shape(self) -> tuple[int] | None:
        # [FUNCTION DOC]
        # Property: one-element tuple with the number of stored values, or none if the
        # values node is missing or has no length.
        try:
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            return None

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        # [FUNCTION DOC]
        # Reads the series. Where and columns must be none (validation raises a type
        # error otherwise); start and stop slice rows. Reads the index and values arrays,
        # builds a Series with the stored name without copying; a unicode encode error
        # caused by surrogates under surrogatepass mode with the string dtype option and
        # PyArrow installed is retried with a python-storage string dtype; other errors
        # propagate. Returns the series.
        self.validate_read(columns, where)
        index = self.read_index("index", start=start, stop=stop)
        values = self.read_array("values", start=start, stop=stop)
        try:
            result = Series(values, index=index, name=self.name, copy=False)
        except UnicodeEncodeError as err:
            if (
                self.errors == "surrogatepass"
                and using_string_dtype()
                and str(err).endswith("surrogates not allowed")
                and HAS_PYARROW
            ):
                result = Series(
                    values,
                    index=index,
                    name=self.name,
                    copy=False,
                    dtype=StringDtype(storage="python", na_value=np.nan),
                )
            else:
                raise
        return result

    def write(self, obj, **kwargs) -> None:
        # [FUNCTION DOC]
        # Writes the series: base attributes, the index, the values array, and the series
        # name as a group attribute. Extra keyword arguments are passed to the base class
        # and otherwise ignored.
        super().write(obj, **kwargs)
        self.write_index("index", obj.index)
        self.write_array("values", obj)
        self.attrs.name = obj.name


class BlockManagerFixed(GenericFixed):
    # [CLASS DOC]
    # Fixed-format storer for objects backed by a block manager (the base of the frame
    # storer). Stores each axis as an index named by position (axis zero, axis one),
    # and each block as a pair: block values and block items (the column labels of the
    # block), plus the dimension count and number of blocks as attributes.
    # Class state: persisted attributes are the dimension count and block count.
    attributes = ["ndim", "nblocks"]

    nblocks: int

    @property
    def shape(self) -> list[int] | None:
        # [FUNCTION DOC]
        # Property: the shape as a list: the leading data dimensions taken from the first
        # block's values node followed by the total item count summed over all blocks'
        # item nodes. Returns none if attributes are missing.
        try:
            ndim = self.ndim

            # items
            items = 0
            for i in range(self.nblocks):
                node = getattr(self.group, f"block{i}_items")
                shape = getattr(node, "shape", None)
                if shape is not None:
                    items += shape[0]

            # data shape
            node = self.group.block0_values
            shape = getattr(node, "shape", None)
            if shape is not None:
                shape = list(shape[0 : (ndim - 1)])
            else:
                shape = []

            shape.append(items)

            return shape
        except AttributeError:
            return None

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> DataFrame:
        # [FUNCTION DOC]
        # Reads the whole frame (where and columns must be none, else a type error; start
        # and stop select rows only).
        # Behavior: finds the block manager axis that holds the rows, reads each axis
        # index (slicing only the row axis), then for each block reads its items and its
        # values (sliced by the row bounds), builds a frame from the transposed values
        # with the block's columns (positions mapped through the full columns index), and
        # for string-typed object arrays under the string dtype option casts to the string
        # dtype. Concatenates the pieces column-wise, copies, and reorders to the full
        # columns. With no blocks returns an empty frame with the stored axes.
        # # KNOWN ISSUE: the row bounds used when reading block values are the loop
        # variables left over from the preceding axis loop, which only works because the
        # row axis is the last axis visited.
        # start, stop applied to rows, so 0th axis only
        self.validate_read(columns, where)
        select_axis = self.obj_type()._get_block_manager_axis(0)

        axes = []
        for i in range(self.ndim):
            _start, _stop = (start, stop) if i == select_axis else (None, None)
            ax = self.read_index(f"axis{i}", start=_start, stop=_stop)
            axes.append(ax)

        items = axes[0]
        dfs = []

        for i in range(self.nblocks):
            blk_items = self.read_index(f"block{i}_items")
            values = self.read_array(f"block{i}_values", start=_start, stop=_stop)

            columns = items[items.get_indexer(blk_items)]
            df = DataFrame(values.T, columns=columns, index=axes[1], copy=False)
            if (
                using_string_dtype()
                and isinstance(values, np.ndarray)
                and is_string_array(values, skipna=True)
            ):
                df = df.astype(StringDtype(na_value=np.nan))
            dfs.append(df)

        if len(dfs) > 0:
            out = concat(dfs, axis=1).copy()
            return out.reindex(columns=items)

        return DataFrame(columns=axes[0], index=axes[1])

    def write(self, obj, **kwargs) -> None:
        # [FUNCTION DOC]
        # Writes the frame: base attributes, then consolidates the block manager if
        # needed, records the dimension count, writes each axis index (raising a value
        # error if the columns axis is not unique), records the number of blocks, and for
        # each block writes the values array and then the items index. The values are
        # written before the items on purpose, according to an inline comment about an old
        # bug fix.
        super().write(obj, **kwargs)

        data = obj._mgr
        if not data.is_consolidated():
            data = data.consolidate()

        self.attrs.ndim = data.ndim
        for i, ax in enumerate(data.axes):
            if i == 0 and (not ax.is_unique):
                raise ValueError("Columns index has to be unique for fixed format")
            self.write_index(f"axis{i}", ax)

        # Supporting mixed-type DataFrame objects...nontrivial
        self.attrs.nblocks = len(data.blocks)
        for i, blk in enumerate(data.blocks):
            # I have no idea why, but writing values before items fixed #2299
            blk_items = data.items.take(blk.mgr_locs)
            self.write_array(f"block{i}_values", blk.values, items=blk_items)
            self.write_index(f"block{i}_items", blk_items)


class FrameFixed(BlockManagerFixed):
    # [CLASS DOC]
    # Fixed-format storer for a DataFrame: the block manager storer specialized with
    # pandas kind frame and the frame object type. No other behavior.
    pandas_kind = "frame"
    obj_type = DataFrame


class Table(Fixed):
    # [CLASS DOC]
    # Base class for table-format storers: row-oriented, appendable and queryable
    # objects stored as a PyTables table.
    # Table layout state (also persisted as group attributes): index axes (index
    # column descriptions), non-index axes (pairs of axis number and labels), values
    # axes (data column descriptions), data columns (names of columns stored
    # separately so they can be queried), the string used for missing text, level
    # names (a count of one, or a list for a multi-level index), metadata column
    # names, an info mapping of per-axis metadata, plus the encoding and error mode
    # inherited from the base storer.
    # Class attributes: default pandas kind wide table, format type table, a table
    # type text set by subclasses, and the table flag set to true.
    # Methods cover validation against an existing table, queryable column
    # discovery, metadata read and write, attribute persistence, index creation, and
    # (in the next part of the class) reading axes, preparing and writing data. The
    # class is a base of the appendable and worm tables.
    """
    represent a table:
        facilitate read/write of various types of tables

    Attrs in Table Node
    -------------------
    These are attributes that are store in the main table node, they are
    necessary to recreate these tables when read back in.

    index_axes    : a list of tuples of the (original indexing axis and
        index column)
    non_index_axes: a list of tuples of the (original index axis and
        columns on a non-indexing axis)
    values_axes   : a list of the columns which comprise the data of this
        table
    data_columns  : a list of the columns that we are allowing indexing
        (these become single columns in values_axes)
    nan_rep       : the string to use for nan representations for string
        objects
    levels        : the names of levels
    metadata      : the names of the metadata columns
    """

    pandas_kind = "wide_table"
    format_type: str = "table"  # GH#30962 needed by dask
    table_type: str
    levels: int | list[Hashable] = 1
    is_table = True

    metadata: list

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: str | None = None,
        errors: str = "strict",
        index_axes: list[IndexCol] | None = None,
        non_index_axes: list[tuple[AxisInt, Any]] | None = None,
        values_axes: list[DataCol] | None = None,
        data_columns: list | None = None,
        info: dict | None = None,
        nan_rep=None,
    ) -> None:
        # [FUNCTION DOC]
        # Initializes the base storer with parent, group, encoding and errors, then stores
        # the index axes, non-index axes, values axes and data columns (each defaulting to
        # an empty list), the info mapping (default empty) and the missing-text
        # representation (default none).
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes = index_axes or []
        self.non_index_axes = non_index_axes or []
        self.values_axes = values_axes or []
        self.data_columns = data_columns or []
        self.info = info or {}
        self.nan_rep = nan_rep

    @property
    def table_type_short(self) -> str:
        # [FUNCTION DOC]
        # Property: the first underscore-separated part of the table type text.
        return self.table_type.split("_")[0]

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Pretty text: first infers axes, then returns the padded pandas type, the version
        # in brackets only for old versions, the short table type, the row count, the
        # column count, the indexer names and the data column names.
        """return a pretty representation of myself"""
        self.infer_axes()
        jdc = ",".join(self.data_columns) if len(self.data_columns) else ""
        dc = f",dc->[{jdc}]"

        ver = ""
        if self.is_old_version:
            jver = ".".join([str(x) for x in self.version])
            ver = f"[{jver}]"

        jindex_axes = ",".join([a.name for a in self.index_axes])
        return (
            f"{self.pandas_type:12.12}{ver} "
            f"(typ->{self.table_type_short},nrows->{self.nrows},"
            f"ncols->{self.ncols},indexers->[{jindex_axes}]{dc})"
        )

    def __getitem__(self, c: str):
        # [FUNCTION DOC]
        # Looks up an axis description (index or values) by name among all axes; returns
        # none if absent.
        """return the axis for c"""
        for a in self.axes:
            if c == a.name:
                return a
        return None

    def validate(self, other) -> None:
        # [FUNCTION DOC]
        # Validates that data to be appended is compatible with an existing table.
        # Parameter other is the existing table storer (none means nothing to check).
        # Raises a type error if the table types differ. Then compares the index axes,
        # non-index axes and values axes lists; on a difference it finds the first
        # differing axis and raises a value error: for values axes whose kinds differ the
        # message says the column cannot be serialized because its contents are not of the
        # new kind; otherwise a message about an invalid combination. If lists differ but
        # no differing axis is found, raises a generic exception (considered unreachable).
        """validate against an existing table"""
        if other is None:
            return

        if other.table_type != self.table_type:
            raise TypeError(
                "incompatible table_type with existing "
                f"[{other.table_type} - {self.table_type}]"
            )

        for c in ["index_axes", "non_index_axes", "values_axes"]:
            sv = getattr(self, c, None)
            ov = getattr(other, c, None)
            if sv != ov:
                # show the error for the specific axes
                # Argument 1 to "enumerate" has incompatible type
                # "Optional[Any]"; expected "Iterable[Any]"  [arg-type]
                for i, sax in enumerate(sv):  # type: ignore[arg-type]
                    # Value of type "Optional[Any]" is not indexable  [index]
                    oax = ov[i]  # type: ignore[index]
                    if sax != oax:
                        if c == "values_axes" and sax.kind != oax.kind:
                            raise ValueError(
                                f"Cannot serialize the column [{oax.values[0]}] "
                                f"because its data contents are not [{sax.kind}] "
                                f"but [{oax.kind}] object dtype"
                            )
                        raise ValueError(
                            f"invalid combination of [{c}] on appending data "
                            f"[{sax}] vs current table [{oax}]"
                        )

                # should never get here
                raise Exception(
                    f"invalid combination of [{c}] on appending data [{sv}] vs "
                    f"current table [{ov}]"
                )

    @property
    def is_multi_index(self) -> bool:
        # [FUNCTION DOC]
        # Property: true when the levels attribute is a list (a multi-level index was
        # stored); the attribute is a plain number one otherwise.
        """the levels attribute is 1 or a list in the case of a multi-index"""
        return isinstance(self.levels, list)

    def validate_multiindex(
        self, obj: DataFrame | Series
    ) -> tuple[DataFrame, list[Hashable]]:
        # [FUNCTION DOC]
        # Prepares an object with a multi-level index for table storage.
        # Returns a pair: the object with its index reset into ordinary columns (a frame),
        # and the list of level names with missing names filled with generated defaults.
        # Raises a value error with a duplicate names message if resetting fails.
        """
        validate that we can store the multi-index; reset and return the
        new object
        """
        levels = com.fill_missing_names(obj.index.names)
        try:
            reset_obj = obj.reset_index()
        except ValueError as err:
            raise ValueError(
                "duplicate names/columns in the multi-index when storing as a table"
            ) from err
        assert isinstance(reset_obj, DataFrame)  # for mypy
        return reset_obj, levels

    @property
    def nrows_expected(self) -> int:
        # [FUNCTION DOC]
        # Property: the expected number of rows, computed as the product of the lengths of
        # the values of all index axes.
        """based on our axes, compute the expected nrows"""
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])

    @property
    def is_exists(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if the group contains a child named table (the table has been
        # created).
        """has this table been created"""
        return "table" in self.group

    @property
    def storable(self):
        # [FUNCTION DOC]
        # Property: the PyTables table node under the group, or none.
        return getattr(self.group, "table", None)

    @property
    def table(self):
        # [FUNCTION DOC]
        # Property: the PyTables table (same as the storable).
        """return the table group (this is my storable)"""
        return self.storable

    @property
    def dtype(self):
        # [FUNCTION DOC]
        # Property: the numpy structured dtype of the PyTables table.
        return self.table.dtype

    @property
    def description(self):
        # [FUNCTION DOC]
        # Property: the column description object of the PyTables table.
        return self.table.description

    @property
    def axes(self) -> itertools.chain[IndexCol]:
        # [FUNCTION DOC]
        # Property: an iterator chaining the index axes followed by the values axes. It is
        # a one-shot iterator created on each access.
        return itertools.chain(self.index_axes, self.values_axes)

    @property
    def ncols(self) -> int:
        # [FUNCTION DOC]
        # Property: the total number of columns, the sum of the lengths of the values of
        # all values axes.
        """the number of total columns in the values axes"""
        return sum(len(a.values) for a in self.values_axes)

    @property
    def is_transposed(self) -> bool:
        # [FUNCTION DOC]
        # Property: false in this class; overridden by subclasses that store transposed
        # data.
        return False

    @property
    def data_orientation(self) -> tuple[int, ...]:
        # [FUNCTION DOC]
        # Property: tuple of axis numbers: first the axes of the non-index axes, then the
        # axes of the index axes, each as integers.
        """return a tuple of my permutated axes, non_indexable at the front"""
        return tuple(
            itertools.chain(
                [int(a[0]) for a in self.non_index_axes],
                [int(a.axis) for a in self.index_axes],
            )
        )

    def queryables(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the dictionary of names allowed in query expressions mapped to their
        # column descriptions. Contains each index axis by column name, each non-index axis
        # by its standard name (index for axis zero, columns for axis one) mapped to none,
        # and each values axis whose name is in the data columns, by column name.
        """return a dict of the kinds allowable columns for this object"""
        # mypy doesn't recognize DataFrame._AXIS_NAMES, so we re-write it here
        axis_names = {0: "index", 1: "columns"}

        # compute the values_axes queryables
        d1 = [(a.cname, a) for a in self.index_axes]
        d2 = [(axis_names[axis], None) for axis, values in self.non_index_axes]
        d3 = [
            (v.cname, v) for v in self.values_axes if v.name in set(self.data_columns)
        ]

        return dict(d1 + d2 + d3)

    def index_cols(self) -> list[tuple[Any, Any]]:
        # [FUNCTION DOC]
        # Returns a list of pairs (axis number, column name) for the index axes.
        """return a list of my index cols"""
        # Note: each `i.cname` below is assured to be a str.
        return [(i.axis, i.cname) for i in self.index_axes]

    def values_cols(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the list of column names of the values axes.
        """return a list of my values cols"""
        return [i.cname for i in self.values_axes]

    def _get_metadata_path(self, key: str) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the path of the metadata table for a key: the group path, then
        # the meta folder, the key, and a meta node.
        """return the metadata pathname for this key"""
        group = self.group._v_pathname
        return f"{group}/meta/{key}/meta"

    def write_metadata(self, key: str, values: np.ndarray) -> None:
        # [FUNCTION DOC]
        # Writes a metadata array (such as categories) for a key as its own object in the
        # store, via the parent store put method.
        # Parameters: key and the values array (wrapped in a series without copying).
        # The object is written at the metadata path using table format, the storer's
        # encoding, error mode and missing-text representation.
        # # KNOWN ISSUE: the docstring says the array is written as a fixed-format series,
        # but the code requests table format.
        """
        Write out a metadata array to the key as a fixed-format Series.

        Parameters
        ----------
        key : str
        values : ndarray
        """
        self.parent.put(
            self._get_metadata_path(key),
            Series(values, copy=False),
            format="table",
            encoding=self.encoding,
            errors=self.errors,
            nan_rep=self.nan_rep,
        )

    def read_metadata(self, key: str):
        # [FUNCTION DOC]
        # Reads the metadata array for a key: if the group has a meta folder with an entry
        # for the key, selects the object at the metadata path from the parent store and
        # returns it; otherwise returns none.
        """return the meta data array for this key"""
        if getattr(getattr(self.group, "meta", None), key, None) is not None:
            return self.parent.select(self._get_metadata_path(key))
        return None

    def set_attrs(self) -> None:
        # [FUNCTION DOC]
        # Persists the table description to the group attributes: table type text, index
        # column list, values column list, non-index axes, data columns, missing-text
        # representation, encoding, error mode, levels and the info mapping.
        """set our table type & indexables"""
        self.attrs.table_type = str(self.table_type)
        self.attrs.index_cols = self.index_cols()
        self.attrs.values_cols = self.values_cols()
        self.attrs.non_index_axes = self.non_index_axes
        self.attrs.data_columns = self.data_columns
        self.attrs.nan_rep = self.nan_rep
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors
        self.attrs.levels = self.levels
        self.attrs.info = self.info

    def get_attrs(self) -> None:
        # [FUNCTION DOC]
        # Restores the table description from the group attributes, with defaults: empty
        # non-index axes, data columns and info; none missing-text representation; UTF-8
        # encoding; strict error mode; and an empty list for levels. Then derives the index
        # axes (the indexables flagged as indexable axes) and values axes (the rest) from
        # the cached list of indexables.
        """retrieve our attributes"""
        self.non_index_axes = getattr(self.attrs, "non_index_axes", None) or []
        self.data_columns = getattr(self.attrs, "data_columns", None) or []
        self.info = getattr(self.attrs, "info", None) or {}
        self.nan_rep = getattr(self.attrs, "nan_rep", None)
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        self.errors = getattr(self.attrs, "errors", "strict")
        self.levels: list[Hashable] = getattr(self.attrs, "levels", None) or []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]

    def validate_version(self, where=None) -> None:
        # [FUNCTION DOC]
        # Warns about an old file format. Only when a where clause is given and the stored
        # version is old, issues an incompatibility warning (attributed to the caller)
        # saying the where criteria are ignored. Does not raise.
        """are we trying to operate on an old version?"""
        if where is not None:
            if self.is_old_version:
                ws = incompatibility_doc % ".".join([str(x) for x in self.version])
                warnings.warn(
                    ws,
                    IncompatibilityWarning,
                    stacklevel=find_stack_level(),
                )

    def validate_min_itemsize(self, min_itemsize) -> None:
        # [FUNCTION DOC]
        # Validates a minimum size mapping for string columns. Returns without checking
        # unless the argument is a dictionary. For every key other than the generic
        # values key, the key must be among the queryable names (index axes, data
        # columns); otherwise raises a value error.
        """
        validate the min_itemsize doesn't contain items that are not in the
        axes this needs data_columns to be defined
        """
        if min_itemsize is None:
            return
        if not isinstance(min_itemsize, dict):
            return

        q = self.queryables()
        for k in min_itemsize:
            # ok, apply generally
            if k == "values":
                continue
            if k not in q:
                raise ValueError(
                    f"min_itemsize has the key [{k}] which is not an axis or "
                    "data_column"
                )

    @cache_readonly
    def indexables(self):
        # [FUNCTION DOC]
        # Cached property creating the list of column descriptions of the existing table
        # from its stored attributes and the PyTables description.
        # Index columns: for each stored (axis, name) pair, builds an index column with
        # its position, the kind from the table attributes, the column type, the table and
        # categorical metadata (when a metadata object exists the meta kind is category).
        # Value columns: for each stored values column name, builds a data column
        # (indexable data column when the name is in the data columns), using the version
        # adjusted name, the labels stored under the kind attribute, the stored dtype and
        # the kind derived from it, the metadata object, the stored meta kind attribute, and
        # positions continuing after the index columns.
        # Returns the combined list. Result is computed once per storer instance.
        """create/cache the indexables if they don't exist"""
        _indexables = []

        desc = self.description
        table_attrs = self.table.attrs

        # Note: each of the `name` kwargs below are str, ensured
        #  by the definition in index_cols.
        # index columns
        for i, (axis, name) in enumerate(self.attrs.index_cols):
            atom = getattr(desc, name)
            md = self.read_metadata(name)
            meta = "category" if md is not None else None

            kind_attr = f"{name}_kind"
            kind = getattr(table_attrs, kind_attr, None)

            index_col = IndexCol(
                name=name,
                axis=axis,
                pos=i,
                kind=kind,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            _indexables.append(index_col)

        # values columns
        dc = set(self.data_columns)
        base_pos = len(_indexables)

        def f(i, c: str) -> DataCol:
            # [FUNCTION DOC]
            # HELPER: closure inside the indexables property. Builds a data column object for
            # one stored values column. Parameters: i is its zero-based position among values
            # columns; c is the column name (asserted to be a string). Chooses the indexable
            # data column class if c is a data column, else the plain data column; fetches the
            # PyTables column type; computes the version-adjusted name; reads the stored
            # labels (kept under the kind attribute name), dtype and meta kind attributes of
            # that name; derives the kind from the dtype; reads metadata; and constructs the
            # column with position equal to the base position plus i. Returns it.
            assert isinstance(c, str)
            klass = DataCol
            if c in dc:
                klass = DataIndexableCol

            atom = getattr(desc, c)
            adj_name = _maybe_adjust_name(c, self.version)

            # TODO: why kind_attr here?
            values = getattr(table_attrs, f"{adj_name}_kind", None)
            dtype = getattr(table_attrs, f"{adj_name}_dtype", None)
            # Argument 1 to "_dtype_to_kind" has incompatible type
            # "Optional[Any]"; expected "str"  [arg-type]
            kind = _dtype_to_kind(dtype)  # type: ignore[arg-type]

            md = self.read_metadata(c)
            # TODO: figure out why these two versions of `meta` dont always match.
            #  meta = "category" if md is not None else None
            meta = getattr(table_attrs, f"{adj_name}_meta", None)

            obj = klass(
                name=adj_name,
                cname=c,
                values=values,
                kind=kind,
                pos=base_pos + i,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
                dtype=dtype,
            )
            return obj

        # Note: the definition of `values_cols` ensures that each
        #  `c` below is a str.
        _indexables.extend([f(i, c) for i, c in enumerate(self.attrs.values_cols)])

        return _indexables

    def create_index(
        self, columns=None, optlevel=None, kind: str | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Creates PyTables indexes on columns of the table.
        # Parameters: columns is false (no indexing), true or none (all indexable axes and
        # data columns), or a column name or list or tuple of names; optlevel is the
        # optimization level and kind the index kind (none leaves the library default).
        # Behavior: returns silently if the axes cannot be inferred or columns is false.
        # For each requested column found on the table: if already indexed, an index with
        # a different explicit kind or level is removed so it can be rebuilt, otherwise the
        # existing kind and level are reused; if not indexed after that, creates the
        # index, raising a type error for complex-valued columns. A requested column that
        # is not on the table but is one of the non-index column labels raises an
        # attribute error saying the column is not a data column and must be reloaded with
        # data columns. Unknown names are silently ignored.
        # Side effects: modifies the index state of the file.
        # # KNOWN ISSUE: the options dictionary is shared across columns in the loop, so
        # the kind and level of an already-indexed column leak into the options used for
        # later columns.
        """
        Create a pytables index on the specified columns.

        Parameters
        ----------
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError if trying to create an index on a complex-type column.

        Notes
        -----
        Cannot index Time64Col or ComplexCol.
        Pytables must be >= 3.0.
        """
        if not self.infer_axes():
            return
        if columns is False:
            return

        # index all indexables and data_columns
        if columns is None or columns is True:
            columns = [a.cname for a in self.axes if a.is_data_indexable]
        if not isinstance(columns, (tuple, list)):
            columns = [columns]

        kw = {}
        if optlevel is not None:
            kw["optlevel"] = optlevel
        if kind is not None:
            kw["kind"] = kind

        table = self.table
        for c in columns:
            v = getattr(table.cols, c, None)
            if v is not None:
                # remove the index if the kind/optlevel have changed
                if v.is_indexed:
                    index = v.index
                    cur_optlevel = index.optlevel
                    cur_kind = index.kind

                    if kind is not None and cur_kind != kind:
                        v.remove_index()
                    else:
                        kw["kind"] = cur_kind

                    if optlevel is not None and cur_optlevel != optlevel:
                        v.remove_index()
                    else:
                        kw["optlevel"] = cur_optlevel

                # create the index
                if not v.is_indexed:
                    if v.type.startswith("complex"):
                        raise TypeError(
                            "Columns containing complex values can be stored but "
                            "cannot be indexed when using table format. Either use "
                            "fixed format, set index=False, or do not include "
                            "the columns containing complex values to "
                            "data_columns when initializing the table."
                        )
                    v.create_index(**kw)
            elif c in self.non_index_axes[0][1]:
                # GH 28156
                raise AttributeError(
                    f"column {c} is not a data_column.\n"
                    f"In order to read column {c} you must reload the dataframe \n"
                    f"into HDFStore and include {c} with the data_columns argument."
                )

    def _read_axes(
        self, where, start: int | None = None, stop: int | None = None
    ) -> list[tuple[np.ndarray, np.ndarray] | tuple[Index, Index]]:
        # [FUNCTION DOC]
        # HELPER: reads and converts the axes of the table for a selection.
        # Parameters: where is a selection (expression, term list or coordinates); start
        # and stop are optional row bounds.
        # Behavior: builds a selection object over this table, runs it to fetch the raw
        # structured rows, then for every axis (index axes first, then values axes) loads
        # the axis info from the stored info mapping and converts the raw rows with the
        # axis's convert method using the table's missing-text representation, encoding and
        # error mode.
        # Returns a list with one pair per axis: for index axes the converted index (twice),
        # for data blocks the column labels and the converted array.
        """
        Create the axes sniffed from the table.

        Parameters
        ----------
        where : ???
        start : int or None, default None
        stop : int or None, default None

        Returns
        -------
        List[Tuple[index_values, column_values]]
        """
        # create the selection
        selection = Selection(self, where=where, start=start, stop=stop)
        values = selection.select()

        results = []
        # convert the data
        for a in self.axes:
            a.set_info(self.info)
            res = a.convert(
                values,
                nan_rep=self.nan_rep,
                encoding=self.encoding,
                errors=self.errors,
            )
            results.append(res)

        return results

    @classmethod
    def get_object(cls, obj, transposed: bool):
        # [FUNCTION DOC]
        # Class method hook returning the object to be stored for an orientation: here the
        # object unchanged (the transposed flag is ignored). Subclasses may transpose.
        """return the data for this obj"""
        return obj

    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes) -> list:
        # [FUNCTION DOC]
        # HELPER: computes the list of data column names (columns stored as separate,
        # queryable table columns).
        # Parameters: data_columns is true (all labels of the non-index axis), none (no
        # columns) or a list of labels; min_itemsize is a size, or a mapping whose keys
        # other than the generic values key are also added as data columns; non_index_axes
        # is the list of pairs of axis number and labels (if empty, returns an empty list).
        # Raises a value error if the non-index axis is a multi-level index and data
        # columns were requested. Returns the requested data columns (plus any min size
        # keys) that actually are labels of the non-index axis, keeping the order of the
        # requested list. Does not mutate the argument list.
        # # KNOWN ISSUE: an inline comment says the result follows the order of the axis,
        # but the result keeps the order of the requested list.
        """
        take the input data_columns and min_itemize and create a data
        columns spec
        """
        if not len(non_index_axes):
            return []

        axis, axis_labels = non_index_axes[0]
        info = self.info.get(axis, {})
        if info.get("type") == "MultiIndex" and data_columns:
            raise ValueError(
                f"cannot use a multi-index on axis [{axis}] with "
                f"data_columns {data_columns}"
            )

        # evaluate the passed data_columns, True == use all columns
        # take only valid axis labels
        if data_columns is True:
            data_columns = list(axis_labels)
        elif data_columns is None:
            data_columns = []

        # if min_itemsize is a dict, add the keys (exclude 'values')
        if isinstance(min_itemsize, dict):
            existing_data_columns = set(data_columns)
            data_columns = list(data_columns)  # ensure we do not modify
            data_columns.extend(
                [
                    k
                    for k in min_itemsize.keys()
                    if k != "values" and k not in existing_data_columns
                ]
            )

        # return valid columns in the order of our axis
        return [c for c in data_columns if c in axis_labels]

    def _create_axes(
        self,
        axes,
        obj: DataFrame,
        validate: bool = True,
        nan_rep=None,
        data_columns=None,
        min_itemsize=None,
    ):
        # [FUNCTION DOC]
        # HELPER: builds a new table description (index axes, non-index axes and values
        # axes) for the object to be written, validated against the existing table if any.
        # Parameters: axes is a list of axis names or numbers to use as index axes (default
        # the first axis); obj is the frame (a type error is raised otherwise); validate
        # says whether to validate against an existing table (default true); nan_rep is the
        # missing-text representation (default the text nan); data_columns is true, a list
        # of labels, or none; min_itemsize is the minimum string size spec.
        # Behavior: converts axes to numbers; if the table already exists, its index axes,
        # data columns and missing-text representation override the arguments. Only one
        # index axis is supported (otherwise value error). Builds the non-index axis from
        # the other axis labels, recording its names and type in the info mapping; when
        # appending, if the new labels are a permutation of the existing ones they are
        # reordered to the existing order. Builds the index column description (position
        # zero, converted via the shared helper) and updates info and string sizes.
        # Reindexes the object to the non-index labels, determines data columns, gets the
        # consolidated (possibly transposed) frame, splits it into blocks and block items
        # (matching the order of an existing table), and for each block builds a data
        # column description: data columns of a single item become indexable data columns
        # (name must be a string, else value error) and others get generated block names;
        # when validating, the existing column at the same position must exist (value error
        # on incompatible structure); the block values are prepared for string storage; the
        # PyTables type, kind, time zone, categorical ordering and categories (or string
        # dtype name as meta) are determined and the data stored in the description.
        # Finally creates a new table storer of the same class with these descriptions,
        # copies the levels attribute, validates the minimum size mapping, and when
        # validating against an existing table checks compatibility (which may raise).
        # Returns the new table storer (not yet written). Side effects: updates the info
        # mapping of this table.
        """
        Create and return the axes.

        Parameters
        ----------
        axes: list or None
            The names or numbers of the axes to create.
        obj : DataFrame
            The object to create axes on.
        validate: bool, default True
            Whether to validate the obj against an existing object already written.
        nan_rep :
            A value to use for string column nan_rep.
        data_columns : List[str], True, or None, default None
            Specify the columns that we want to create to allow indexing on.

            * True : Use all available columns.
            * None : Use no columns.
            * List[str] : Use the specified columns.

        min_itemsize: Dict[str, int] or None, default None
            The min itemsize for a column in bytes.
        """
        if not isinstance(obj, DataFrame):
            group = self.group._v_name
            raise TypeError(
                f"cannot properly create the storer for: [group->{group},"
                f"value->{type(obj)}]"
            )

        # set the default axes if needed
        if axes is None:
            axes = [0]

        # map axes to numbers
        axes = [obj._get_axis_number(a) for a in axes]

        # do we have an existing table (if so, use its axes & data_columns)
        if self.infer_axes():
            table_exists = True
            axes = [a.axis for a in self.index_axes]
            data_columns = list(self.data_columns)
            nan_rep = self.nan_rep
            # TODO: do we always have validate=True here?
        else:
            table_exists = False

        new_info = self.info

        assert self.ndim == 2  # with next check, we must have len(axes) == 1
        # currently support on ndim-1 axes
        if len(axes) != self.ndim - 1:
            raise ValueError(
                "currently only support ndim-1 indexers in an AppendableTable"
            )

        # create according to the new data
        new_non_index_axes: list = []

        # nan_representation
        if nan_rep is None:
            nan_rep = "nan"

        # We construct the non-index-axis first, since that alters new_info
        idx = next(x for x in [0, 1] if x not in axes)

        a = obj.axes[idx]
        # we might be able to change the axes on the appending data if necessary
        append_axis = list(a)
        if table_exists:
            indexer = len(new_non_index_axes)  # i.e. 0
            exist_axis = self.non_index_axes[indexer][1]
            if not array_equivalent(
                np.array(append_axis),
                np.array(exist_axis),
                strict_nan=True,
                dtype_equal=True,
            ):
                # ahah! -> reindex
                if array_equivalent(
                    np.array(sorted(append_axis)),
                    np.array(sorted(exist_axis)),
                    strict_nan=True,
                    dtype_equal=True,
                ):
                    append_axis = exist_axis

        # the non_index_axes info
        info = new_info.setdefault(idx, {})
        info["names"] = list(a.names)
        info["type"] = type(a).__name__

        new_non_index_axes.append((idx, append_axis))

        # Now we can construct our new index axis
        idx = axes[0]
        a = obj.axes[idx]
        axis_name = obj._get_axis_name(idx)
        new_index = _convert_index(axis_name, a, self.encoding, self.errors)
        new_index.axis = idx

        # Because we are always 2D, there is only one new_index, so
        #  we know it will have pos=0
        new_index.set_pos(0)
        new_index.update_info(new_info)
        new_index.maybe_set_size(min_itemsize)  # check for column conflicts

        new_index_axes = [new_index]
        j = len(new_index_axes)  # i.e. 1
        assert j == 1

        # reindex by our non_index_axes & compute data_columns
        assert len(new_non_index_axes) == 1
        for a in new_non_index_axes:
            obj = _reindex_axis(obj, a[0], a[1])

        transposed = new_index.axis == 1

        # figure out data_columns and get out blocks
        data_columns = self.validate_data_columns(
            data_columns, min_itemsize, new_non_index_axes
        )

        frame = self.get_object(obj, transposed)._consolidate()

        blocks, blk_items = self._get_blocks_and_items(
            frame, table_exists, new_non_index_axes, self.values_axes, data_columns
        )

        # add my values
        vaxes = []
        for i, (blk, b_items) in enumerate(zip(blocks, blk_items, strict=True)):
            # shape of the data column are the indexable axes
            klass = DataCol
            name = None

            # we have a data_column
            if data_columns and len(b_items) == 1 and b_items[0] in data_columns:
                klass = DataIndexableCol
                name = b_items[0]
                if not (name is None or isinstance(name, str)):
                    # TODO: should the message here be more specifically non-str?
                    raise ValueError("cannot have non-object label DataIndexableCol")

            # make sure that we match up the existing columns
            # if we have an existing table
            existing_col: DataCol | None

            if table_exists and validate:
                try:
                    existing_col = self.values_axes[i]
                except (IndexError, KeyError) as err:
                    raise ValueError(
                        f"Incompatible appended table [{blocks}]"
                        f"with existing table [{self.values_axes}]"
                    ) from err
            else:
                existing_col = None

            new_name = name or f"values_block_{i}"
            data_converted = _maybe_convert_for_string_atom(
                new_name,
                blk.values,
                existing_col=existing_col,
                min_itemsize=min_itemsize,
                nan_rep=nan_rep,
                encoding=self.encoding,
                errors=self.errors,
                columns=b_items,
            )
            adj_name = _maybe_adjust_name(new_name, self.version)

            typ = klass._get_atom(data_converted)
            kind = _dtype_to_kind(data_converted.dtype.name)
            tz = None
            if getattr(data_converted, "tz", None) is not None:
                tz = _get_tz(data_converted.tz)

            meta = metadata = ordered = None
            if isinstance(data_converted.dtype, CategoricalDtype):
                ordered = data_converted.ordered
                meta = "category"
                metadata = np.asarray(data_converted.categories).ravel()
            elif isinstance(blk.dtype, StringDtype):
                meta = str(blk.dtype)

            data, dtype_name = _get_data_and_dtype_name(data_converted)

            col = klass(
                name=adj_name,
                cname=new_name,
                values=list(b_items),
                typ=typ,
                pos=j,
                kind=kind,
                tz=tz,
                ordered=ordered,
                meta=meta,
                metadata=metadata,
                dtype=dtype_name,
                data=data,
            )
            col.update_info(new_info)

            vaxes.append(col)

            j += 1

        dcs = [col.name for col in vaxes if col.is_data_indexable]

        new_table = type(self)(
            parent=self.parent,
            group=self.group,
            encoding=self.encoding,
            errors=self.errors,
            index_axes=new_index_axes,
            non_index_axes=new_non_index_axes,
            values_axes=vaxes,
            data_columns=dcs,
            info=new_info,
            nan_rep=nan_rep,
        )
        if hasattr(self, "levels"):
            # TODO: get this into constructor, only for appropriate subclass
            new_table.levels = self.levels

        new_table.validate_min_itemsize(min_itemsize)

        if validate and table_exists:
            new_table.validate(self)

        return new_table

    @staticmethod
    def _get_blocks_and_items(
        frame: DataFrame,
        table_exists: bool,
        new_non_index_axes,
        values_axes,
        data_columns,
    ):
        # [FUNCTION DOC]
        # HELPER: static method splitting a frame into its storage blocks.
        # Parameters: frame is the consolidated frame; table_exists says whether to align
        # with an existing table; new_non_index_axes is the pair list for the non-index
        # axis; values_axes are the existing value columns; data_columns are labels to be
        # stored separately.
        # Behavior: takes the frame's blocks with their item labels. If data columns are
        # given, the non-data columns are reindexed into blocks and each data column is
        # reindexed separately and appended as its own block. If the table exists, blocks
        # are reordered to the existing table's block structure, matching on item labels
        # (value error if some existing block cannot be matched).
        # Returns a pair of lists: blocks and their item indexes.
        # Helper to clarify non-state-altering parts of _create_axes
        def get_blk_items(mgr):
            # [FUNCTION DOC]
            # HELPER: nested function returning, for a block manager, the list of item labels
            # of each of its blocks.
            return [mgr.items.take(blk.mgr_locs) for blk in mgr.blocks]

        mgr = frame._mgr
        blocks: list[Block] = list(mgr.blocks)
        blk_items: list[Index] = get_blk_items(mgr)

        if len(data_columns):
            # TODO: prove that we only get here with axis == 1?
            #  It is the case in all extant tests, but NOT the case
            #  outside this `if len(data_columns)` check.

            axis, axis_labels = new_non_index_axes[0]
            new_labels = Index(axis_labels).difference(Index(data_columns))
            mgr = frame.reindex(new_labels, axis=axis)._mgr

            blocks = list(mgr.blocks)
            blk_items = get_blk_items(mgr)
            for c in data_columns:
                # This reindex would raise ValueError if we had a duplicate
                #  index, so we can infer that (as long as axis==1) we
                #  get a single column back, so a single block.
                mgr = frame.reindex([c], axis=axis)._mgr
                blocks.extend(mgr.blocks)
                blk_items.extend(get_blk_items(mgr))

        # reorder the blocks in the same order as the existing table if we can
        if table_exists:
            by_items = {
                tuple(b_items.tolist()): (b, b_items)
                for b, b_items in zip(blocks, blk_items, strict=True)
            }
            new_blocks: list[Block] = []
            new_blk_items = []
            for ea in values_axes:
                items = tuple(ea.values)
                try:
                    b, b_items = by_items.pop(items)
                    new_blocks.append(b)
                    new_blk_items.append(b_items)
                except (IndexError, KeyError) as err:
                    jitems = ",".join([pprint_thing(item) for item in items])
                    raise ValueError(
                        f"cannot match existing table structure for [{jitems}] "
                        "on appending data"
                    ) from err
            blocks = new_blocks
            blk_items = new_blk_items

        return blocks, blk_items

    def process_axes(self, obj, selection: Selection, columns=None) -> DataFrame:
        # [FUNCTION DOC]
        # Applies post-read axis processing and filters to a frame.
        # Parameters: obj is the frame read; selection is the selection object (its filter
        # may hold filtering conditions); columns is an optional list of columns to keep
        # (copied; for multi-level indexes the level names are added to the front).
        # Behavior: reindexes by the stored non-index axes limited to the requested
        # columns, then for each filter in the selection applies it via a local function,
        # and returns the resulting frame.
        # # KNOWN ISSUE: the local filter function is defined inside the loop over the
        # non-index axes but used afterwards, so a table without non-index axes and with a
        # filter would hit an unbound name. It also reads the current frame through a
        # closure.
        """process axes filters"""
        # make a copy to avoid side effects
        if columns is not None:
            columns = list(columns)

        # make sure to include levels if we have them
        if columns is not None and self.is_multi_index:
            assert isinstance(self.levels, list)  # assured by is_multi_index
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)

        # reorder by any non_index_axes & limit to the select columns
        for axis, labels in self.non_index_axes:
            obj = _reindex_axis(obj, axis, labels, columns)

            def process_filter(field, filt, op):
                # [FUNCTION DOC]
                # HELPER: local filter function. Given a field name, a filter (values to match) and
                # a comparison operator function, finds which axis the field refers to. If the field
                # is the name of an axis, filters that axis's labels (adding level names for
                # multi-level indexes). If it is a label on an axis, filters on that label's values
                # (the axis number is swapped for frames). Returns the filtered object. Raises a
                # value error if the field is not found.
                for axis_name in obj._AXIS_ORDERS:
                    axis_number = obj._get_axis_number(axis_name)
                    axis_values = obj._get_axis(axis_name)
                    assert axis_number is not None

                    # see if the field is the name of an axis
                    if field == axis_name:
                        # if we have a multi-index, then need to include
                        # the levels
                        if self.is_multi_index:
                            filt = filt.union(Index(self.levels))

                        takers = op(axis_values, filt)
                        return obj.loc(axis=axis_number)[takers]

                    # this might be the name of a file IN an axis
                    elif field in axis_values:
                        # we need to filter on this dimension
                        values = ensure_index(getattr(obj, field).values)
                        filt = ensure_index(filt)

                        # hack until we support reversed dim flags
                        if isinstance(obj, DataFrame):
                            axis_number = 1 - axis_number

                        takers = op(values, filt)
                        return obj.loc(axis=axis_number)[takers]

                raise ValueError(f"cannot find the field [{field}] for filtering!")

        # apply the selection filters (but keep in the same order)
        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                obj = process_filter(field, filt, op)

        return obj

    def create_description(
        self,
        complib,
        complevel: int | None,
        fletcher32: bool,
        expectedrows: int | None,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: builds the keyword options for creating the PyTables table.
        # Parameters: complib and complevel (default the store's level or nine) and
        # fletcher32 (or the store's flag) configure compression when complib is given;
        # expectedrows defaults to the larger of the expected row count and ten thousand.
        # Returns a dictionary with the table name (table), expected rows, the column
        # description mapping each axis's column name to its type, and filters when a
        # library was passed (or, otherwise, the store-wide filters if set).
        """create the description of the table from the axes & values"""
        # provided expected rows if its passed
        if expectedrows is None:
            expectedrows = max(self.nrows_expected, 10000)

        d = {"name": "table", "expectedrows": expectedrows}

        # description from the axes & values
        d["description"] = {a.cname: a.typ for a in self.axes}

        if complib:
            if complevel is None:
                complevel = self._complevel or 9
            filters = _tables().Filters(
                complevel=complevel,
                complib=complib,
                fletcher32=fletcher32 or self._fletcher32,
            )
            d["filters"] = filters
        elif self._filters is not None:
            d["filters"] = self._filters

        return d

    def read_coordinates(
        self, where=None, start: int | None = None, stop: int | None = None
    ):
        # [FUNCTION DOC]
        # Returns the row positions matching a selection.
        # Parameters: where is a selection; start and stop bound the rows.
        # Behavior: warns if the file version is old and a where clause is present;
        # returns false if the axes cannot be inferred; builds a selection and selects
        # coordinates; when the selection has additional filters, evaluates each by reading
        # the filtered column over the coordinate range and keeps matching coordinates.
        # Returns an index of integer row positions (or false as above).
        """
        select coordinates (row numbers) from a table; return the
        coordinates object
        """
        # validate the version
        self.validate_version(where)

        # infer the data kind
        if not self.infer_axes():
            return False

        # create the selection
        selection = Selection(self, where=where, start=start, stop=stop)
        coords = selection.select_coords()
        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                data = self.read_column(
                    field, start=coords.min(), stop=coords.max() + 1
                )
                coords = coords[op(data.iloc[coords - coords.min()], filt).values]

        return Index(coords, copy=False)

    def read_column(
        self,
        column: str,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        # [FUNCTION DOC]
        # Returns one column of the table as a series.
        # Parameters: column is the name; where must be none (type error if given); start
        # and stop bound the rows.
        # Behavior: returns false if axes cannot be inferred. Finds the axis of that name;
        # it must be data indexable (value error otherwise). Reads the column slice from the
        # table, converts it with the axis's convert method, and returns it as a series
        # named after the column, using the stored meta text as its dtype.
        # Raises a key error if no axis has that name.
        """
        return a single column from the table, generally only indexables
        are interesting
        """
        # validate the version
        self.validate_version()

        # infer the data kind
        if not self.infer_axes():
            return False

        if where is not None:
            raise TypeError("read_column does not currently accept a where clause")

        # find the axes
        for a in self.axes:
            if column == a.name:
                if not a.is_data_indexable:
                    raise ValueError(
                        f"column [{column}] can not be extracted individually; "
                        "it is not data indexable"
                    )

                # column must be an indexable or a data column
                c = getattr(self.table.cols, column)
                a.set_info(self.info)
                col_values = a.convert(
                    c[start:stop],
                    nan_rep=self.nan_rep,
                    encoding=self.encoding,
                    errors=self.errors,
                )
                cvs = col_values[1]
                dtype = getattr(self.table.attrs, f"{column}_meta", None)
                return Series(cvs, name=column, copy=False, dtype=dtype)

        raise KeyError(f"column [{column}] not found in the table")


class WORMTable(Table):
    # [CLASS DOC]
    # Placeholder table kind for write-once read-many tables (searchable but not
    # appendable). Table type text is worm. Read and write are not implemented.
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """

    table_type = "worm"

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        # [FUNCTION DOC]
        # Not implemented: always raises a not-implemented error.
        """
        read the indices and the indexing array, calculate offset rows and return
        """
        raise NotImplementedError("WORMTable needs to implement read")

    def write(self, obj, **kwargs) -> None:
        # [FUNCTION DOC]
        # Not implemented: always raises a not-implemented error.
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        raise NotImplementedError("WORMTable needs to implement write")


class AppendableTable(Table):
    # [CLASS DOC]
    # Base class of the appendable table storers; table type text is appendable.
    # Adds the write path (creating the PyTables table, validating axes, and writing
    # rows in chunks with optional dropping of all-missing rows), chunk writing, and
    # partial deletion of rows.
    """support the new appendable table formats"""

    table_type = "appendable"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(  # type: ignore[override]
        self,
        obj,
        axes=None,
        append: bool = False,
        complib=None,
        complevel=None,
        fletcher32=None,
        min_itemsize=None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool = False,
        nan_rep=None,
        data_columns=None,
        track_times: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Writes (or appends) an object to the table.
        # Parameters: obj is the frame; axes is the list of index axes (default the first);
        # append says whether to add to an existing table (otherwise an existing table node
        # is removed first); complib, complevel and fletcher32 configure compression of a
        # newly created table; min_itemsize sets string widths; chunksize is rows per write
        # chunk (default one hundred thousand); expectedrows hints the final size;
        # dropna drops rows that are missing in all value columns; nan_rep is the
        # missing-text representation; data_columns lists labels to store as separate
        # columns; track_times is passed to the PyTables create call (false makes files
        # reproducible).
        # Behavior: builds the new table description (validating against the existing table
        # when appending), validates axis names, creates the PyTables table with its
        # description and attributes if it does not exist, stores the info mapping,
        # validates each axis against stored attributes and writes the kind attributes, and
        # finally writes the data in chunks.
        # Raises value or type errors from the validation steps.
        if not append and self.is_exists:
            self._handle.remove_node(self.group, "table")

        # create the axes
        table = self._create_axes(
            axes=axes,
            obj=obj,
            validate=append,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
        )

        for a in table.axes:
            a.validate_names()

        if not table.is_exists:
            # create the table
            options = table.create_description(
                complib=complib,
                complevel=complevel,
                fletcher32=fletcher32,
                expectedrows=expectedrows,
            )

            # set the table attributes
            table.set_attrs()

            options["track_times"] = track_times

            # create the table
            table._handle.create_table(table.group, **options)

        # update my info
        table.attrs.info = table.info

        # validate the axes and set the kinds
        for a in table.axes:
            a.validate_and_set(table, append)

        # add the rows
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize: int | None, dropna: bool = False) -> None:
        # [FUNCTION DOC]
        # Writes all rows of this table's data to the file in chunks.
        # Parameters: chunksize is the rows per chunk (default one hundred thousand);
        # dropna (default false) skips rows where every value column is missing.
        # Behavior: when dropping, computes for each value block a per-row flag saying
        # whether all entries of that row are missing (only for blocks where this can be
        # computed) and combines the flags across blocks with a logical and, so a flagged
        # row is dropped by the chunk writer. Requires a
        # single index axis (asserted). Moves the first dimension of each value block to
        # the end, reshapes each to rows by the column shape of the table dtype, then
        # for each chunk of rows calls the chunk writer with slices of the index values, the
        # mask and the value blocks. The loop stops at the first empty chunk.
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        names = self.dtype.names
        nrows = self.nrows_expected

        # if dropna==True, then drop ALL nan rows
        masks = []
        if dropna:
            for a in self.values_axes:
                # figure the mask: only do if we can successfully process this
                # column, otherwise ignore the mask
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype("u1", copy=False))

        # consolidate masks
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None

        # broadcast the indexes if needed
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes  # ensures we dont need to broadcast

        # transpose the values so first dimension is last
        # reshape the values if needed
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
        bvalues = []
        for i, v in enumerate(values):
            new_shape = (nrows, *self.dtype[names[nindexes + i]].shape)
            bvalues.append(v.reshape(new_shape))

        # write the chunks
        if chunksize is None:
            chunksize = 100000

        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break

            self.write_data_chunk(
                rows,
                indexes=[a[start_i:end_i] for a in indexes],
                mask=mask[start_i:end_i] if mask is not None else None,
                values=[v[start_i:end_i] for v in bvalues],
            )

    def write_data_chunk(
        self,
        rows: np.ndarray,
        indexes: list[np.ndarray],
        mask: npt.NDArray[np.bool_] | None,
        values: list[np.ndarray],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: writes one chunk of rows to the PyTables table.
        # Parameters: rows is a preallocated structured buffer, reused if the chunk has the
        # same length (otherwise a new one is made); indexes is the list of index value
        # arrays for this chunk; mask is an optional flags array where true marks rows to
        # drop; values is the list of value block arrays for this chunk.
        # Behavior: returns without writing if any value block has no elements. Fills the
        # structured buffer's fields in order (index fields first, then value fields),
        # removes masked rows, and if any rows remain appends them to the table and
        # flushes.
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
        # 0 len
        for v in values:
            if not np.prod(v.shape):
                return

        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)

        # indexes
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx

        # values
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v

        # mask
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]

        if len(rows):
            self.table.append(rows)
            self.table.flush()

    def delete(
        self, where=None, start: int | None = None, stop: int | None = None
    ) -> int | None:
        # [FUNCTION DOC]
        # Deletes rows from the table.
        # Parameters: where is a selection; start and stop bound the rows.
        # Behavior: with no where (none or empty): with neither start nor stop removes the
        # entire group and returns the previous number of rows; otherwise removes the row
        # range (stop defaults to the end) and flushes, returning the count removed.
        # With a where: returns none if axes cannot be inferred; selects the matching row
        # coordinates, sorts them, splits them into runs of consecutive coordinates, and
        # removes each run from the last to the first so earlier positions stay valid;
        # flushes. Returns the number of matching rows.
        # delete all rows (and return the nrows)
        if where is None or not len(where):
            if start is None and stop is None:
                nrows = self.nrows
                self._handle.remove_node(self.group, recursive=True)
            else:
                # pytables<3.0 would remove a single row with stop=None
                if stop is None:
                    stop = self.nrows
                nrows = self.table.remove_rows(start=start, stop=stop)
                self.table.flush()
            return nrows

        # infer the data kind
        if not self.infer_axes():
            return None

        # create the selection
        table = self.table
        selection = Selection(self, where, start=start, stop=stop)
        values = selection.select_coords()

        # delete the rows in reverse order
        sorted_series = Series(values, copy=False).sort_values()
        ln = len(sorted_series)

        if ln:
            # construct groups of consecutive rows
            diff = sorted_series.diff()
            groups = list(diff[diff > 1].index)

            # 1 group
            if not groups:
                groups = [0]

            # final element
            if groups[-1] != ln:
                groups.append(ln)

            # initial element
            if groups[0] != 0:
                groups.insert(0, 0)

            # we must remove in reverse order!
            pg = groups.pop()
            for g in reversed(groups):
                rows = sorted_series.take(range(g, pg))
                table.remove_rows(
                    start=rows[rows.index[0]], stop=rows[rows.index[-1]] + 1
                )
                pg = g

            self.table.flush()

        # return the number of rows removed
        return ln


class AppendableFrameTable(AppendableTable):
    # [CLASS DOC]
    # Appendable table storer for DataFrames (table type appendable frame, pandas kind
    # frame table, two dimensions, object type frame). Frames whose index axis is the
    # second axis are stored transposed. The Series subclasses adapt this class.
    """support the new appendable table formats"""

    pandas_kind = "frame_table"
    table_type = "appendable_frame"
    ndim = 2
    obj_type: type[DataFrame | Series] = DataFrame

    @property
    def is_transposed(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if the first index axis is axis one (data stored transposed).
        return self.index_axes[0].axis == 1

    @classmethod
    def get_object(cls, obj, transposed: bool):
        # [FUNCTION DOC]
        # Class method returning the object to store: transposed if requested, otherwise
        # unchanged.
        """these are written transposed"""
        if transposed:
            obj = obj.T
        return obj

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        # [FUNCTION DOC]
        # Reads the table into a frame.
        # Parameters: where is a selection; columns restricts the returned columns; start
        # and stop bound the rows.
        # Behavior: warns if the version is old and a where is present; returns none if the
        # axes cannot be inferred. Reads and converts the axes for the selection. For each
        # data block builds the column labels (a multi-level index of tuples when the stored
        # info says so; stored names are restored), then, depending on whether the table is
        # stored transposed, builds a frame from the block values with the row index from
        # the index axis; one-dimensional data-column arrays are reshaped to a single row;
        # extension and datetime arrays and plain arrays are handled; categoricals are
        # built from arrays. A unicode encode error caused by surrogates under surrogatepass
        # mode with the string dtype option and PyArrow installed is retried with a
        # python-storage string dtype. Asserts that the resulting dtypes equal the block
        # dtype (skipped for object data under the string dtype option). Columns whose
        # stored meta text is str or string are cast to that dtype. Frames of all blocks
        # are concatenated column-wise (a single block is used as is). Finally applies the
        # selection filters, axis ordering and column limit through the axis processing
        # method and returns the frame.
        # validate the version
        self.validate_version(where)

        # infer the data kind
        if not self.infer_axes():
            return None

        result = self._read_axes(where=where, start=start, stop=stop)

        info = (
            self.info.get(self.non_index_axes[0][0], {})
            if len(self.non_index_axes)
            else {}
        )

        inds = [i for i, ax in enumerate(self.axes) if ax is self.index_axes[0]]
        assert len(inds) == 1
        ind = inds[0]

        index = result[ind][0]

        frames = []
        for i, a in enumerate(self.axes):
            if a not in self.values_axes:
                continue
            index_vals, cvalues = result[i]

            # we could have a multi-index constructor here
            # ensure_index doesn't recognized our list-of-tuples here
            if info.get("type") != "MultiIndex":
                cols = Index(index_vals)
            else:
                cols = MultiIndex.from_tuples(index_vals)

            names = info.get("names")
            if names is not None:
                cols.set_names(names, inplace=True)

            if self.is_transposed:
                values = cvalues
                index_ = cols
                cols_ = Index(index, name=getattr(index, "name", None))
            else:
                values = cvalues.T
                index_ = Index(index, name=getattr(index, "name", None))
                cols_ = cols

            # if we have a DataIndexableCol, its shape will only be 1 dim
            if values.ndim == 1 and isinstance(values, np.ndarray):
                values = values.reshape((1, values.shape[0]))

            if isinstance(values, (np.ndarray, DatetimeArray)):
                try:
                    df = DataFrame(values.T, columns=cols_, index=index_, copy=False)
                except UnicodeEncodeError as err:
                    if (
                        self.errors == "surrogatepass"
                        and using_string_dtype()
                        and str(err).endswith("surrogates not allowed")
                        and HAS_PYARROW
                    ):
                        df = DataFrame(
                            values.T,
                            columns=cols_,
                            index=index_,
                            copy=False,
                            dtype=StringDtype(storage="python", na_value=np.nan),
                        )
                    else:
                        raise
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                # Categorical
                df = DataFrame._from_arrays([values], columns=cols_, index=index_)
            if not (using_string_dtype() and values.dtype.kind == "O"):
                assert (df.dtypes == values.dtype).all(), (df.dtypes, values.dtype)

            # If str / string dtype is stored in meta, use that.
            for column in cols_:
                dtype = getattr(self.table.attrs, f"{column}_meta", None)
                if dtype in ["str", "string"]:
                    df[column] = df[column].astype(dtype)
            frames.append(df)

        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)

        selection = Selection(self, where=where, start=start, stop=stop)
        # apply the selection filters & axis orderings
        df = self.process_axes(df, selection=selection, columns=columns)
        return df


class AppendableSeriesTable(AppendableFrameTable):
    # [CLASS DOC]
    # Appendable table storer for a Series, implemented on top of the frame table
    # (table type appendable series, pandas kind series table, object type series). A
    # series is stored as a frame with one column, which is named values when the
    # series has no name; data is never transposed.
    """support the new appendable table formats"""

    pandas_kind = "series_table"
    table_type = "appendable_series"
    ndim = 2
    obj_type = Series

    @property
    def is_transposed(self) -> bool:
        # [FUNCTION DOC]
        # Property: always false for series.
        return False

    @classmethod
    def get_object(cls, obj, transposed: bool):
        # [FUNCTION DOC]
        # Class method returning the object to store unchanged; the transposed flag is
        # ignored.
        return obj

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Writes a series (or a frame) as a one-column frame table.
        # Parameters: obj is the series (a non-frame input is converted to a frame whose
        # single column is named after the series, or the generic name values); data_columns
        # is accepted but ignored; extra keyword arguments go to the frame table write.
        # All columns of the converted object are made data columns.
        # # KNOWN ISSUE: the data_columns parameter is overridden with the list of all
        # columns, so a caller supplied value has no effect.
        """we are going to write this as a frame table"""
        if not isinstance(obj, DataFrame):
            name = obj.name or "values"
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        # [FUNCTION DOC]
        # Reads the series. Parameters: where, columns, start, stop as for the frame table.
        # For a multi-level index, the level names are added to the front of the columns
        # list (mutating the caller's list), the frame is read, and the levels are set as
        # the index. Returns the first column as a series, with the generic name values
        # replaced by no name.
        is_multi_index = self.is_multi_index
        if columns is not None and is_multi_index:
            assert isinstance(self.levels, list)  # needed for mypy
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s.set_index(self.levels, inplace=True)

        s = s.iloc[:, 0]

        # remove the default name
        if s.name == "values":
            s.name = None
        return s


class AppendableMultiSeriesTable(AppendableSeriesTable):
    # [CLASS DOC]
    # Appendable table storer for a Series with a multi-level index (table type
    # appendable multiseries). Same pandas kind as the series storer.
    """support the new appendable table formats"""

    pandas_kind = "series_table"
    table_type = "appendable_multiseries"

    #  error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, **kwargs) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Writes a series with a multi-level index. Resets the index into ordinary columns
        # (via the multi-index validation, which also stores the level names on the storer
        # and may raise a value error for duplicate names), names the value column after
        # the series (or values), and writes the result through the series table writer.
        """we are going to write this as a frame table"""
        name = obj.name or "values"
        newobj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        cols = list(self.levels)
        cols.append(name)
        newobj.columns = Index(cols)
        super().write(obj=newobj, **kwargs)


class GenericTable(AppendableFrameTable):
    # [CLASS DOC]
    # Read-only table storer for a generic PyTables table not written by pandas (table
    # type generic table, pandas kind frame table). The columns are described from the
    # PyTables table description rather than from stored pandas attributes. Writing is
    # not supported.
    """a table that read/writes the generic pytables table format"""

    pandas_kind = "frame_table"
    table_type = "generic_table"
    ndim = 2
    obj_type = DataFrame
    levels: list[Hashable]

    @property
    def pandas_type(self) -> str:
        # [FUNCTION DOC]
        # Property: the pandas type is the class pandas kind rather than a stored
        # attribute.
        return self.pandas_kind

    @property
    def storable(self):
        # [FUNCTION DOC]
        # Property: the PyTables table node if present, otherwise the group.
        return getattr(self.group, "table", None) or self.group

    def get_attrs(self) -> None:
        # [FUNCTION DOC]
        # Sets up the table description without stored attributes: no non-index axes, no
        # missing-text representation, no levels; index axes and values axes are derived from
        # the indexables; all value columns are data columns.
        """retrieve our attributes"""
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []

        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]
        self.data_columns = [a.name for a in self.values_axes]

    @cache_readonly
    def indexables(self):
        # [FUNCTION DOC]
        # Cached property creating the column descriptions from the PyTables description:
        # one generic index column named index on axis zero (flagged categorical when a
        # metadata object exists), followed by one generic data indexable column per name
        # in the description, each holding its own name as the only label, with its column
        # type, position and optional category metadata.
        """create the indexables from the table description"""
        d = self.description

        # TODO: can we get a typ for this?  AFAICT it is the only place
        #  where we aren't passing one
        # the index columns is just a simple index
        md = self.read_metadata("index")
        meta = "category" if md is not None else None
        index_col = GenericIndexCol(
            name="index", axis=0, table=self.table, meta=meta, metadata=md
        )

        _indexables: list[GenericIndexCol | GenericDataIndexableCol] = [index_col]

        for i, n in enumerate(d._v_names):
            assert isinstance(n, str)

            atom = getattr(d, n)
            md = self.read_metadata(n)
            meta = "category" if md is not None else None
            dc = GenericDataIndexableCol(
                name=n,
                pos=i,
                values=[n],
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            _indexables.append(dc)

        return _indexables

    # error: Signature of "write" incompatible with supertype "AppendableTable"
    def write(self, **kwargs) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Not supported: always raises a not-implemented error.
        raise NotImplementedError("cannot write on a generic table")


class AppendableMultiFrameTable(AppendableFrameTable):
    # [CLASS DOC]
    # Appendable table storer for a DataFrame with a multi-level index (table type
    # appendable multiframe, short type appendable_multi). The index levels are stored
    # as ordinary data columns named by their level names, and a class level pattern
    # matches default names of the form level followed by digits.
    """a frame with a multi-index"""

    table_type = "appendable_multiframe"
    obj_type = DataFrame
    ndim = 2
    _re_levels = re.compile(r"^level_\d+$")

    @property
    def table_type_short(self) -> str:
        # [FUNCTION DOC]
        # Property: the short table type name, the fixed text appendable_multi.
        return "appendable_multi"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Writes a frame with a multi-level index.
        # Parameters: obj is the frame; data_columns is none, true (all columns) or a list
        # of labels; extra keyword arguments go to the frame table write.
        # Resets the index into columns (storing the level names on the storer), adds the
        # level names to the front of the data columns (mutating a caller-supplied list)
        # and writes through the frame table writer. May raise a value error for duplicate
        # names.
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> DataFrame:
        # [FUNCTION DOC]
        # Reads the frame and restores the multi-level index: reads through the frame
        # table reader, sets the stored level names as the index and clears level names that
        # match the default level pattern. Parameters as for the frame table read.
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)

        # remove names for 'level_%d'
        df.index = df.index.set_names(
            [None if self._re_levels.search(name) else name for name in df.index.names]
        )

        return df


def _reindex_axis(
    obj: DataFrame, axis: AxisInt, labels: Index, other=None
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: reindexes a frame along one axis to given labels.
    # Parameters: obj is the frame; axis is the axis number; labels are the wanted
    # labels; other is an optional second label set to restrict to.
    # Behavior: returns the frame unchanged when the current axis already equals the
    # labels (and the other set, if given). Otherwise de-duplicates the labels, when
    # other is given keeps those of other that are among the labels (in the order of
    # other), and selects them by label if they differ from the current axis. Returns
    # the original or a new frame.
    ax = obj._get_axis(axis)
    labels = ensure_index(labels)

    # try not to reindex even if other is provided
    # if it equals our current index
    if other is not None:
        other = ensure_index(other)
    if (other is None or labels.equals(other)) and labels.equals(ax):
        return obj

    labels = ensure_index(labels.unique())
    if other is not None:
        labels = ensure_index(other.unique()).intersection(labels, sort=False)
    if not labels.equals(ax):
        slicer: list[slice | Index] = [slice(None, None)] * obj.ndim
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj


# tz to/from coercion


def _get_tz(tz: tzinfo) -> str | tzinfo:
    # [FUNCTION DOC]
    # HELPER: for a time-zone-aware type, returns the zone encoded as a name text (or
    # the time zone object when no name is available), via the timezone utility.
    """for a tz-aware type, return an encoded zone"""
    zone = timezones.get_timezone(tz)
    return zone


def _set_tz(
    values: npt.NDArray[np.int64], tz: str | tzinfo | None, datetime64_dtype: str
) -> DatetimeArray:
    # [FUNCTION DOC]
    # HELPER: converts stored 64-bit integers to a datetime array with a time zone.
    # Parameters: values is an int64 array (asserted); tz is a zone name, zone object
    # or none (naive); datetime64_dtype is the dtype name whose unit gives the
    # resolution (any multiplier count in the dtype is ignored).
    # Builds a datetime dtype from the zone and unit and constructs the array from the
    # integers. Returns the datetime array.
    """
    Coerce the values to a DatetimeArray with appropriate tz.

    Parameters
    ----------
    values : ndarray[int64]
    tz : str, tzinfo, or None
    datetime64_dtype : str, e.g. "datetime64[ns]", "datetime64[25s]"
    """
    assert values.dtype == "i8", values.dtype
    # Argument "tz" to "tz_to_dtype" has incompatible type "str | tzinfo | None";
    # expected "tzinfo"
    unit, _ = np.datetime_data(datetime64_dtype)  # parsing dtype: unit, count
    unit = cast("TimeUnit", unit)
    # error: Argument "tz" to "tz_to_dtype" has incompatible type
    #  "str | tzinfo | None"; expected "tzinfo"
    dtype = tz_to_dtype(tz=tz, unit=unit)  # type: ignore[arg-type]
    dta = DatetimeArray._from_sequence(values, dtype=dtype)
    return dta


def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol:
    # [FUNCTION DOC]
    # HELPER: converts a pandas index into an index column description for storage.
    # Parameters: name is the column name (text, asserted); index is the pandas index;
    # encoding and errors apply to string conversion.
    # Behavior: converts the index to a storable array plus dtype name and derives kind
    # and column type. Integer, unsigned, bool and datetime-like (including period and
    # timedelta) indexes return an index column carrying the frequency and time zone.
    # A multi-level index raises a type error. Otherwise by inferred type: dates are
    # stored as 32-bit ordinals; strings as fixed-width encoded strings sized to the
    # longest value; integer or floating object indexes with the converted array;
    # anything else must be object dtype with kind object and is stored with the object
    # atom.
    # Returns the index column description (carrying the index name).
    assert isinstance(name, str)

    index_name = index.name
    # error: Argument 1 to "_get_data_and_dtype_name" has incompatible type "Index";
    # expected "Union[ExtensionArray, ndarray]"
    converted, dtype_name = _get_data_and_dtype_name(index)  # type: ignore[arg-type]
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)

    if (
        lib.is_np_dtype(index.dtype, "iu")
        or needs_i8_conversion(index.dtype)
        or is_bool_dtype(index.dtype)
    ):
        # Includes Index, RangeIndex, DatetimeIndex, TimedeltaIndex, PeriodIndex,
        #  in which case "kind" is "integer", "integer", "datetime64",
        #  "timedelta64", and "integer", respectively.
        return IndexCol(
            name,
            values=converted,
            kind=kind,
            typ=atom,
            freq=getattr(index, "freq", None),
            tz=getattr(index, "tz", None),
            index_name=index_name,
        )

    if isinstance(index, MultiIndex):
        raise TypeError("MultiIndex not supported here!")

    inferred_type = lib.infer_dtype(index, skipna=False)
    # we won't get inferred_type of "datetime64" or "timedelta64" as these
    #  would go through the DatetimeIndex/TimedeltaIndex paths above

    values = np.asarray(index)

    if inferred_type == "date":
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(
            name, converted, "date", _tables().Time32Col(), index_name=index_name
        )
    elif inferred_type == "string":
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(
            name,
            converted,
            "string",
            _tables().StringCol(itemsize),
            index_name=index_name,
        )

    elif inferred_type in ["integer", "floating"]:
        return IndexCol(
            name, values=converted, kind=kind, typ=atom, index_name=index_name
        )
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == "object", kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)


def _unconvert_index(data, kind: str, encoding: str, errors: str) -> np.ndarray | Index:
    # [FUNCTION DOC]
    # HELPER: inverse of the index conversion for reading.
    # Parameters: data is the stored array; kind is the stored kind text; encoding and
    # errors for decoding strings.
    # Behavior by kind: datetime and timedelta kinds produce datetime or timedelta
    # indexes (a unitless old kind means nanoseconds; otherwise the data are
    # reinterpreted with the stored unit); date builds an object array of dates, from
    # ordinals with a fallback to timestamps; integer, float and bool become a plain
    # array; string is decoded to an object array of text; object takes the first stored
    # element (the stored variable-length object array). Any other kind raises a value
    # error.
    # Returns an array or index.
    # # KNOWN ISSUE: the string branch tests membership in a parenthesized plain string
    # rather than a tuple, so it does substring matching (for example an empty kind or a
    # fragment of the word string would also match).
    index: Index | np.ndarray

    if kind.startswith("datetime64"):
        if kind == "datetime64":
            # created before we stored resolution information
            index = DatetimeIndex(data, copy=False)
        else:
            index = DatetimeIndex(data.view(kind), copy=False)
    elif kind.startswith("timedelta64"):
        if kind == "timedelta64":
            # created before we stored resolution information
            index = TimedeltaIndex(data, copy=False)
        else:
            index = TimedeltaIndex(data.view(kind), copy=False)
    elif kind == "date":
        try:
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ("integer", "float", "bool"):
        index = np.asarray(data)
    elif kind in ("string"):
        index = _unconvert_string_array(
            data, nan_rep=None, encoding=encoding, errors=errors
        )
    elif kind == "object":
        index = np.asarray(data[0])
    else:  # pragma: no cover
        raise ValueError(f"unrecognized index type {kind}")
    return index


def _maybe_convert_for_string_atom(
    name: str,
    bvalues: ArrayLike,
    existing_col,
    min_itemsize,
    nan_rep,
    encoding,
    errors,
    columns: list[str],
):
    # [FUNCTION DOC]
    # HELPER: prepares a block of values for storage as fixed-width strings.
    # Parameters: name is the column/block name; bvalues is the block array; existing_col
    # is the already-stored column description (or none); min_itemsize is a size, a
    # mapping by column name or the generic values key, or none; nan_rep is the text used
    # for missing entries; encoding and errors for encoding; columns are the labels of
    # the block (used in error messages).
    # Behavior: string extension dtypes are converted to numpy objects; non-object arrays
    # are returned unchanged. For object arrays: raises a type error for dates or for
    # mixed time zones (datetimes). Missing entries in a copy are replaced by the
    # missing text, and a value error is raised if that text is longer than the existing
    # column width. If the data are still not all strings, finds the first row that is
    # not and raises a type error naming that column. Otherwise encodes to fixed-width
    # byte strings, sizes the width as the largest of the encoded width, the requested
    # minimum and the existing column width (validating against the existing column) and
    # returns the array cast to that width.
    # # KNOWN ISSUE: the check that returns the original values when the inferred type is
    # not string and the dtype name is not object can never trigger, because only object
    # arrays reach it (so the dtype name is always object).
    if isinstance(bvalues.dtype, StringDtype):
        bvalues = bvalues.to_numpy()
    if bvalues.dtype != object:
        return bvalues

    bvalues = cast(np.ndarray, bvalues)

    dtype_name = bvalues.dtype.name
    inferred_type = lib.infer_dtype(bvalues, skipna=False)

    if inferred_type == "date":
        raise TypeError("[date] is not implemented as a table column")
    if inferred_type == "datetime":
        # after GH#8260
        # this only would be hit for a multi-timezone dtype which is an error
        raise TypeError(
            "too many timezones in this block, create separate data columns"
        )

    if not (inferred_type == "string" or dtype_name == "object"):
        return bvalues

    mask = isna(bvalues)
    data = bvalues.copy()
    data[mask] = nan_rep

    if existing_col and mask.any() and len(nan_rep) > existing_col.itemsize:
        raise ValueError("NaN representation is too large for existing column size")

    # see if we have a valid string type
    inferred_type = lib.infer_dtype(data, skipna=False)
    if inferred_type != "string":
        # we cannot serialize this data, so report an exception on a column
        # by column basis

        # expected behaviour:
        # search block for a non-string object column by column
        for i in range(data.shape[0]):
            col = data[i]
            inferred_type = lib.infer_dtype(col, skipna=False)
            if inferred_type != "string":
                error_column_label = columns[i] if len(columns) > i else f"No.{i}"
                raise TypeError(
                    f"Cannot serialize the column [{error_column_label}]\n"
                    f"because its data contents are not [string] but "
                    f"[{inferred_type}] object dtype"
                )

    # itemsize is the maximum length of a string (along any dimension)

    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    itemsize = data_converted.itemsize

    # specified min_itemsize?
    if isinstance(min_itemsize, dict):
        min_itemsize = int(min_itemsize.get(name) or min_itemsize.get("values") or 0)
    itemsize = max(min_itemsize or 0, itemsize)

    # check for column in the values conflicts
    if existing_col is not None:
        eci = existing_col.validate_col(itemsize)
        if eci is not None and eci > itemsize:
            itemsize = eci

    data_converted = data_converted.astype(f"|S{itemsize}", copy=False)
    return data_converted


def _convert_string_array(data: np.ndarray, encoding: str, errors: str) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: converts an object array of strings to a fixed-width byte string array.
    # Parameters: data is the object array; encoding and errors define the text
    # encoding. Non-empty data are encoded elementwise (shape preserved); the width is
    # the longest encoded value (at least one). Returns the fixed-width array.
    """
    Take a string-like that is object dtype and coerce to a fixed size string type.

    Parameters
    ----------
    data : np.ndarray[object]
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[fixed-length-string]
    """
    # encode if needed
    if len(data):
        data = (
            Series(data.ravel(), copy=False, dtype="object")
            .str.encode(encoding, errors)
            ._values.reshape(data.shape)
        )

    # create the sized dtype
    ensured = ensure_object(data.ravel())
    itemsize = max(1, libwriters.max_len_string_array(ensured))

    data = np.asarray(data, dtype=f"S{itemsize}")
    return data


def _unconvert_string_array(
    data: np.ndarray, nan_rep, encoding: str, errors: str
) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: inverse of the string conversion. Parameters: data is a fixed-width string
    # array; nan_rep is the stored missing text (default nan); encoding and errors for
    # decoding. Flattens to objects, decodes byte strings with the encoding (or widens
    # text), then replaces entries equal to the missing text with a missing value.
    # Returns the object array in the original shape.
    # Side effect: the replacement happens in the decoded array only.
    """
    Inverse of _convert_string_array.

    Parameters
    ----------
    data : np.ndarray[fixed-length-string]
    nan_rep : the storage repr of NaN
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[object]
        Decoded data.
    """
    shape = data.shape
    data = np.asarray(data.ravel(), dtype=object)

    if len(data):
        itemsize = libwriters.max_len_string_array(ensure_object(data))
        dtype = f"U{itemsize}"

        if isinstance(data[0], bytes):
            ser = Series(data, copy=False).str.decode(
                encoding, errors=errors, dtype="object"
            )
            data = ser.to_numpy()
            data.flags.writeable = True
        else:
            data = data.astype(dtype, copy=False).astype(object, copy=False)

    if nan_rep is None:
        nan_rep = "nan"

    libwriters.string_array_replace_from_nan_rep(data, nan_rep)
    return data.reshape(shape)


def _maybe_convert(values: np.ndarray, val_kind: str, encoding: str, errors: str):
    # [FUNCTION DOC]
    # HELPER: converts raw stored values if the kind requires it (datetime kinds and
    # string); otherwise returns them unchanged. Asserts the kind is text.
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values


def _get_converter(kind: str, encoding: str, errors: str):
    # [FUNCTION DOC]
    # HELPER: returns a conversion function for a kind: unitless datetime64 kind gives a
    # function casting to nanosecond datetimes; other datetime64 kinds cast to that
    # dtype; string gives a function decoding fixed-width strings (no missing text).
    # Any other kind raises a value error.
    if kind == "datetime64":
        return lambda x: np.asarray(x, dtype="M8[ns]")
    elif "datetime64" in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == "string":
        return lambda x: _unconvert_string_array(
            x, nan_rep=None, encoding=encoding, errors=errors
        )
    else:  # pragma: no cover
        raise ValueError(f"invalid kind {kind}")


def _need_convert(kind: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true for the kinds needing conversion on read: string and any
    # datetime64 kind; false otherwise.
    if kind in ("datetime64", "string") or "datetime64" in kind:
        return True
    return False


def _maybe_adjust_name(name: str, version: Sequence[int]) -> str:
    # [FUNCTION DOC]
    # HELPER: adjusts a block name for files written before version 0.10.1. The version
    # must be a sequence of at least three integers (a string or shorter sequence
    # raises a value error). For versions 0.x.0 with x at most 10, a name of the form
    # values_block_N becomes values_N; otherwise the name is returned unchanged.
    """
    Prior to 0.10.1, we named values blocks like: values_block_0 and the
    name values_0, adjust the given name if necessary.

    Parameters
    ----------
    name : str
    version : Tuple[int, int, int]

    Returns
    -------
    str
    """
    if isinstance(version, str) or len(version) < 3:
        raise ValueError("Version is incorrect, expected sequence of 3 integers.")

    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r"values_block_(\d+)", name)
        if m:
            grp = m.groups()[0]
            name = f"values_{grp}"
    return name


def _dtype_to_kind(dtype_str: str) -> str:
    # [FUNCTION DOC]
    # HELPER: maps a dtype name text to a storage kind text: string/bytes to string,
    # float, complex, int and uint to integer, datetime64 and timedelta kinds keep the
    # full dtype text, bool, category, period to integer (the frequency is stored
    # separately), object, and str. Raises a value error for any other name.
    """
    Find the "kind" string describing the given dtype name.
    """
    if dtype_str.startswith(("string", "bytes")):
        kind = "string"
    elif dtype_str.startswith("float"):
        kind = "float"
    elif dtype_str.startswith("complex"):
        kind = "complex"
    elif dtype_str.startswith(("int", "uint")):
        kind = "integer"
    elif dtype_str.startswith("datetime64"):
        kind = dtype_str
    elif dtype_str.startswith("timedelta"):
        kind = dtype_str
    elif dtype_str.startswith("bool"):
        kind = "bool"
    elif dtype_str.startswith("category"):
        kind = "category"
    elif dtype_str.startswith("period"):
        # We store the `freq` attr so we can restore from integers
        kind = "integer"
    elif dtype_str == "object":
        kind = "object"
    elif dtype_str == "str":
        kind = "str"
    else:
        raise ValueError(f"cannot interpret dtype of [{dtype_str}]")

    return kind


def _get_data_and_dtype_name(data: ArrayLike):
    # [FUNCTION DOC]
    # HELPER: converts an array into a storable numpy array plus a dtype name text.
    # Categoricals are replaced by their integer codes. The dtype name of
    # time-zone-aware datetimes is a naive datetime name with the unit; others use the
    # array dtype name. Datetime and timedelta kinds are viewed as 64-bit integers and
    # period indexes use their integer ordinals. Returns the pair (array, dtype name).
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(data, Categorical):
        data = data.codes

    if isinstance(data.dtype, DatetimeTZDtype):
        # For datetime64tz we need to drop the TZ in tests TODO: why?
        dtype_name = f"datetime64[{data.dtype.unit}]"
    else:
        dtype_name = data.dtype.name

    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
        # TODO: we used to reshape for the dt64tz case, but no longer
        #  doing that doesn't seem to break anything.  why?

    elif isinstance(data, PeriodIndex):
        data = data.asi8

    data = np.asarray(data)
    return data, dtype_name


class Selection:
    # [CLASS DOC]
    # Evaluates a selection on a table: converts a where argument into either a list of
    # row coordinates or a query condition plus post-read filter.
    # State: the table, the where argument, start and stop bounds, and the computed
    # condition, filter, parsed terms and coordinates (each none until set).
    # Key methods: generate (parse where into terms), select (read rows), select_coords
    # (return row positions).
    """
    Carries out a selection operation on a tables.Table object.

    Parameters
    ----------
    table : a Table object
    where : list of Terms (or convertible to)
    start, stop: indices to start and/or stop selection

    """

    def __init__(
        self,
        table: Table,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the selection. If where is list-like, tries to interpret it as coordinates:
        # a boolean array selects positions within the start/stop range, an integer array is
        # used as coordinates after checking that they lie within start and stop. Otherwise
        # (or when no coordinates), parses the where into terms and evaluates them into a
        # numexpr condition and a filter.
        # # KNOWN ISSUE: the value error for coordinates outside start and stop is raised
        # inside a block that suppresses value errors, so it is swallowed and the code falls
        # back to parsing the array as an expression.
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None

        if is_list_like(where):
            # see if we have a passed coordinate like
            with suppress(ValueError):
                inferred = lib.infer_dtype(where, skipna=False)
                if inferred in ("integer", "boolean"):
                    where = np.asarray(where)
                    if where.dtype == np.bool_:
                        start, stop = self.start, self.stop
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = self.table.nrows
                        self.coordinates = np.arange(start, stop)[where]
                    elif issubclass(where.dtype.type, np.integer):
                        if (self.start is not None and (where < self.start).any()) or (
                            self.stop is not None and (where >= self.stop).any()
                        ):
                            raise ValueError(
                                "where must have index locations >= start and < stop"
                            )
                        self.coordinates = where

        if self.coordinates is None:
            self.terms = self.generate(where)

            # create the numexpr & the filter
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    @overload
    def generate(self, where: dict | list | tuple | str) -> PyTablesExpr:
        # [FUNCTION DOC]
        # Typing overload for a non-empty where: returns a parsed expression. See the
        # implementation below.
        ...

    @overload
    def generate(self, where: None) -> None:
        # [FUNCTION DOC]
        # Typing overload for a none where: returns none. See the implementation below.
        ...

    def generate(self, where: dict | list | tuple | str | None) -> PyTablesExpr | None:
        # [FUNCTION DOC]
        # Parses a where into a query expression object.
        # Parameter where is a dictionary, list, tuple, text or none. Returns none for none.
        # Otherwise builds the expression using the table's queryable names and encoding. If
        # the expression refers to unknown names, raises a value error saying the references
        # must be axes or data columns and listing the defined ones.
        """where can be a : dict,list,tuple,string"""
        if where is None:
            return None

        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            # raise a nice message, suggesting that the user should use
            # data_columns
            qkeys = ",".join(q.keys())
            msg = dedent(
                f"""\
                The passed where expression: {where}
                            contains an invalid variable reference
                            all of the variable references must be a reference to
                            an axis (e.g. 'index' or 'columns'), or a data_column
                            The currently defined references are: {qkeys}
                """
            )
            raise ValueError(msg) from err

    def select(self):
        # [FUNCTION DOC]
        # Runs the selection and returns the raw rows: rows satisfying the condition within
        # start and stop, else the given coordinates, else all rows within the bounds.
        """
        generate the selection
        """
        if self.condition is not None:
            return self.table.table.read_where(
                self.condition.format(), start=self.start, stop=self.stop
            )
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self):
        # [FUNCTION DOC]
        # Returns the row positions: negative start or stop are shifted by the row count;
        # with a condition returns the sorted matching positions within the bounds; with
        # coordinates returns them unchanged (start and stop not applied); otherwise returns
        # the full range of positions between start and stop.
        """
        generate the selection
        """
        start, stop = self.start, self.stop
        nrows = self.table.nrows
        if start is None:
            start = 0
        elif start < 0:
            start += nrows
        if stop is None:
            stop = nrows
        elif stop < 0:
            stop += nrows

        if self.condition is not None:
            return self.table.table.get_where_list(
                self.condition.format(), start=start, stop=stop, sort=True
            )
        elif self.coordinates is not None:
            return self.coordinates

        return np.arange(start, stop)
