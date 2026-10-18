"""
Collection of query wrappers / abstractions to both facilitate data
retrieval and to reduce dependency on DB-specific API.
"""
# [MODULE DOC]
# SQL input and output for data frames.
# Public contents: the read_sql_table, read_sql_query and read_sql readers (exposed
# at the top level of the library), the to_sql writer (called by the frame and series
# write methods), has_table with the alias table_exists, get_schema (table creation
# statement text), and pandasSQL_builder (chooses the database helper for a
# connection).
# Internal structure: a set of database helper classes sharing one abstract
# interface: SQLDatabase (SQLAlchemy based, full featured), ADBCDatabase (Arrow
# database connectivity, partial features) and SQLiteDatabase (plain sqlite3 fallback
# without SQLAlchemy). SQLTable and its sqlite variant map a frame to a table
# description, create it and move data in both directions. A small engine layer
# (base engine, SQLAlchemy engine, get_engine) performs the row insertion. Helper
# functions handle date parsing, conversion of row tuples to frames, and identifier
# quoting; a dictionary maps inferred kinds to sqlite type names.
# Relationships: uses the optional dependencies sqlalchemy and the ADBC driver manager
# (plus pyarrow) imported lazily; uses date conversion, frame construction and
# dtype backend validation from the core; reads the io.sql.engine option.
# Import-time side effects: none (no global registration); sqlite adapters are
# registered later, when a sqlite table object is created.

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from contextlib import (
    ExitStack,
    contextmanager,
)
from datetime import (
    date,
    datetime,
    time,
)
from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat._optional import (
    VERSIONS,
    import_optional_dependency,
)
from pandas.errors import (
    AbstractMethodError,
    DatabaseError,
)
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas import get_option
from pandas.core.api import (
    DataFrame,
    Series,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime

from pandas.io._util import arrow_table_to_pandas

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterator,
        Mapping,
    )

    from sqlalchemy import Table
    from sqlalchemy.sql.expression import (
        Delete,
        Select,
        TextClause,
    )

    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        IndexLabel,
    )

    from pandas import Index

# -----------------------------------------------------------------------------
# -- Helper functions


def _process_parse_dates_argument(parse_dates):
    # [FUNCTION DOC]
    # HELPER: normalizes the date-parsing argument of the read functions into a
    # container. A true, false or none value becomes an empty list; a value that is not
    # iterable becomes a one-element list; any other value (list, dictionary, or even a
    # string) is returned unchanged.
    # # KNOWN ISSUE: a plain string is iterable so it is returned as is rather than
    # wrapped in a list; later membership tests then do substring matching and the
    # format lookup fails, so a single column name given as text is not handled
    # gracefully despite the comment about non-list entries.
    """Process parse_dates argument for read_sql functions"""
    # handle non-list entries for parse_dates gracefully
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []

    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    return parse_dates


def _handle_date_column(
    col, utc: bool = False, format: str | dict[str, Any] | None = None
):
    # [FUNCTION DOC]
    # HELPER: converts one column to datetimes.
    # Parameters: col is a series; utc (default false) asks for UTC results; format is
    # none, a format text, a unit code, or a dictionary of extra arguments for the
    # general datetime conversion.
    # Behavior: a dictionary format is passed directly to the datetime conversion as
    # keyword arguments (the utc flag is then ignored). Otherwise, with no format, float
    # and integer columns default to the seconds unit. A unit code (day, hour, minute,
    # second, millisecond, microsecond, nanosecond) converts numbers since the epoch
    # with invalid values coerced to missing. A time-zone-aware column (when no unit)
    # is converted to UTC. Anything else is parsed with the format text, coercing
    # invalid values.
    # Returns the converted series.
    if isinstance(format, dict):
        # GH35185 Allow custom error values in parse_dates argument of
        # read_sql like functions.
        # Format can take on custom to_datetime argument values such as
        # {"errors": "coerce"} or {"dayfirst": True}
        return to_datetime(col, **format)
    else:
        # Allow passing of formatting string for integers
        # GH17855
        if format is None and (
            issubclass(col.dtype.type, np.floating)
            or issubclass(col.dtype.type, np.integer)
        ):
            format = "s"
        if format in ["D", "d", "h", "m", "s", "ms", "us", "ns"]:
            return to_datetime(col, errors="coerce", unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            # coerce to UTC timezone
            # GH11216
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors="coerce", format=format, utc=utc)


def _parse_date_columns(data_frame: DataFrame, parse_dates) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: forces selected columns of a result frame to datetimes, in place.
    # Parameters: data_frame is the result; parse_dates is a list of column names or a
    # dictionary mapping names to a format, unit code or argument dictionary (normalized
    # first).
    # For each column that is time-zone-aware (always converted, to UTC) or named in the
    # date list, the format is looked up in the dictionary (none when the argument is a
    # list), the column is converted and the converted column replaces the original by
    # position. Returns the same frame object.
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    parse_dates = _process_parse_dates_argument(parse_dates)

    # we want to coerce datetime64_tz dtypes for now to UTC
    # we could in theory do a 'nice' conversion from a FixedOffset tz
    # GH11216
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))

    return data_frame


def _convert_arrays_to_dataframe(
    data,
    columns,
    coerce_float: bool = True,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: builds a frame from a sequence of database rows.
    # Parameters: data is a sequence of row tuples; columns are the column names;
    # coerce_float (default true) converts decimal-like objects to floating point;
    # dtype_backend is numpy (default) or a nullable/pyarrow backend name.
    # Behavior: converts the rows to an object array, splits it into per-column arrays and
    # infers column types. For the pyarrow backend each column is turned into an Arrow
    # array (string-typed columns are cast to the regular, not large, string type) and
    # wrapped in an Arrow-backed extension array. Builds a frame with a default integer
    # index of the row count; with no columns at all returns an empty frame with the given
    # column names.
    content = lib.to_object_array_tuples(data)
    idx_len = content.shape[0]
    arrays = convert_object_array(
        list(content.T),
        dtype=None,
        coerce_float=coerce_float,
        dtype_backend=dtype_backend,
    )
    if dtype_backend == "pyarrow":
        pa = import_optional_dependency("pyarrow")

        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                # TODO: Arrow still infers strings arrays as regular strings instead
                # of large_string, which is what we preserver everywhere else for
                # dtype_backend="pyarrow". We may want to reconsider this
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays  # type: ignore[assignment]
    if arrays:
        return DataFrame._from_arrays(
            arrays, columns=columns, index=range(idx_len), verify_integrity=False
        )
    else:
        return DataFrame(columns=columns)


def _wrap_result(
    data,
    columns,
    index_col=None,
    coerce_float: bool = True,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: turns the rows of a query result into a finished frame.
    # Parameters: data is the list of rows; columns are names; index_col is the optional
    # column(s) to set as index; coerce_float, dtype_backend as for the row converter;
    # parse_dates is the date-parsing argument; dtype is an optional dtype or mapping to
    # cast the frame to.
    # Steps in order: build the frame, cast to the dtype if given, parse date columns,
    # set the index column if given. Returns the frame.
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)

    if dtype:
        frame = frame.astype(dtype)

    frame = _parse_date_columns(frame, parse_dates)

    if index_col is not None:
        frame = frame.set_index(index_col)

    return frame


def _wrap_result_adbc(
    df: DataFrame,
    *,
    index_col=None,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: post-processes a frame already produced by an ADBC cursor: casts to the
    # dtype if given, parses date columns, and sets the index column if given
    # (keyword-only arguments index_col, parse_dates, dtype; dtype_backend is accepted
    # but unused). Returns the frame.
    # # KNOWN ISSUE: the docstring is a copy of the SQLAlchemy wrapper's and talks about
    # a SQLAlchemy result set; the function actually receives an ADBC result frame, and
    # the dtype backend argument is ignored.
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)

    df = _parse_date_columns(df, parse_dates)

    if index_col is not None:
        df = df.set_index(index_col)

    return df


# -----------------------------------------------------------------------------
# -- Read and write to DataFrames


@overload
def read_sql_table(  # pyright: ignore[reportOverlappingOverload]
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] | None = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    # [FUNCTION DOC]
    # Typing overload for reading a table without chunking: returns one frame. See the
    # implementation below.
    ...


@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] | None = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]:
    # [FUNCTION DOC]
    # Typing overload for reading a table in chunks: returns an iterator of frames. See
    # the implementation below.
    ...


@set_module("pandas")
def read_sql_table(
    table_name: str,
    con,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    # [FUNCTION DOC]
    # Reads a whole database table into a frame (or an iterator of frames).
    # Parameters: table_name is the table; con is a SQLAlchemy connectable or URI text
    # (plain DBAPI connections such as sqlite are not supported); schema is the schema
    # name (default the connection's default); index_col is a column or list of columns
    # to use as index; coerce_float (default true) converts decimal-like values to
    # floats; parse_dates is a list of columns, or a mapping of column to format, unit or
    # argument dictionary; columns restricts the columns selected; chunksize (default
    # none) returns an iterator of frames of that many rows; dtype_backend selects
    # nullable or pyarrow dtypes (default numpy).
    # Behavior: validates the backend, opens a database helper within a transaction,
    # raises a value error if the table does not exist, and delegates to the helper's
    # table reader. Returns the result, or raises a value error (including the
    # connection) if nothing came back.
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table("table_name", "postgres:///db_name")  # doctest:+SKIP
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f"Table {table_name} not found")

        table = pandas_sql.read_table(
            table_name,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    if table is not None:
        return table
    else:
        raise ValueError(f"Table {table_name} not found", con)


@overload
def read_sql_query(  # pyright: ignore[reportOverlappingOverload]
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    # [FUNCTION DOC]
    # Typing overload for a query without chunking: returns one frame. See the
    # implementation below.
    ...


@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    chunksize: int = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]:
    # [FUNCTION DOC]
    # Typing overload for a query in chunks: returns an iterator of frames. See the
    # implementation below.
    ...


@set_module("pandas")
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list[Any] | Mapping[str, Any] | None = None,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    # [FUNCTION DOC]
    # Runs a SQL query and returns its result as a frame (or an iterator of frames).
    # Parameters: sql is a query text or a SQLAlchemy selectable; con is a SQLAlchemy
    # connectable, URI text or sqlite3 connection (other DBAPI connections trigger a
    # warning and are treated like sqlite); index_col, coerce_float, parse_dates,
    # chunksize, dtype_backend as in the table reader; params are query parameters in the
    # driver's parameter style (list, tuple or mapping); dtype is a dtype or mapping to
    # cast the result to.
    # Behavior: validates the backend (default numpy), builds the database helper with
    # the connection, and delegates to the helper's query reader. Returns the frame or
    # iterator.
    """
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or mapping, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        Returns a DataFrame object that contains the result set of the
        executed SQL query, in relation to the specified database connection.

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.

    Examples
    --------
    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("sqlite:///database.db")  # doctest: +SKIP
    >>> sql_query = "SELECT int_column FROM test_data"  # doctest: +SKIP
    >>> with engine.connect() as conn, conn.begin():  # doctest: +SKIP
    ...     data = pd.read_sql_query(sql_query, conn)  # doctest: +SKIP
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        return pandas_sql.read_query(
            sql,
            index_col=index_col,
            params=params,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            chunksize=chunksize,
            dtype=dtype,
            dtype_backend=dtype_backend,
        )


@overload
def read_sql(  # pyright: ignore[reportOverlappingOverload]
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> DataFrame:
    # [FUNCTION DOC]
    # Typing overload for the combined reader without chunking: returns one frame. See
    # the implementation below.
    ...


@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> Iterator[DataFrame]:
    # [FUNCTION DOC]
    # Typing overload for the combined reader in chunks: returns an iterator of frames.
    # See the implementation below.
    ...


@set_module("pandas")
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params=None,
    parse_dates=None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    dtype: DtypeArg | None = None,
) -> DataFrame | Iterator[DataFrame]:
    # [FUNCTION DOC]
    # Reads either a query or a whole table, choosing automatically.
    # Parameters: sql is a query or a table name; con may be a SQLAlchemy connectable, URI
    # text, sqlite3 connection or ADBC connection; index_col, coerce_float, params,
    # parse_dates, chunksize, dtype_backend and dtype as in the query reader; columns
    # restricts the columns when a table is read; dtype is ignored for tables.
    # Behavior: validates the backend (default numpy) and builds the database helper. For
    # a sqlite helper always runs the text as a query (table names are not accepted).
    # Otherwise asks the helper whether the text is a table name, treating any error from
    # the driver as no; if it is a table name reads the table, else runs it as a query.
    # Returns the frame or iterator.
    """
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the ADBC connection and
        SQLAlchemy connectable; str connections are closed automatically. See
        `here <https://docs.sqlalchemy.org/en/20/core/connections.html>`_.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.
        The argument is ignored if a table is passed instead of a query.

        .. versionadded:: 2.0.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        Returns a DataFrame object that contains the result set of the
        executed SQL query or an SQL Table based on the provided input,
        in relation to the specified database connection.

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Notes
    -----
    ``pandas`` does not attempt to sanitize SQL statements;
    instead it simply forwards the statement you are executing
    to the underlying driver, which may or may not sanitize from there.
    Please refer to the underlying driver documentation for any details.
    Generally, be wary when accepting statements from arbitrary sources.

    Examples
    --------
    Read data from SQL via either a SQL query or a SQL tablename.
    When using a SQLite database only SQL queries are accepted,
    providing only the SQL tablename will result in an error.

    >>> from sqlite3 import connect
    >>> conn = connect(":memory:")
    >>> df = pd.DataFrame(
    ...     data=[[0, "10/11/12"], [1, "12/11/10"]],
    ...     columns=["int_column", "date_column"],
    ... )
    >>> df.to_sql(name="test_data", con=conn)
    2

    >>> pd.read_sql("SELECT int_column, date_column FROM test_data", conn)
       int_column date_column
    0           0    10/11/12
    1           1    12/11/10

    >>> pd.read_sql("test_data", "postgres:///db_name")  # doctest:+SKIP

    For parameterized query, using ``params`` is recommended over string interpolation.

    >>> from sqlalchemy import text
    >>> sql = text(
    ...     "SELECT int_column, date_column FROM test_data WHERE int_column=:int_val"
    ... )
    >>> pd.read_sql(sql, conn, params={"int_val": 1})  # doctest:+SKIP
       int_column date_column
    0           1    12/11/10

    Apply date parsing to columns through the ``parse_dates`` argument
    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.
    Custom argument values for applying ``pd.to_datetime`` on a column are specified
    via a dictionary format:

    >>> pd.read_sql(
    ...     "SELECT int_column, date_column FROM test_data",
    ...     conn,
    ...     parse_dates={"date_column": {"format": "%d/%m/%y"}},
    ... )
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12

    .. versionadded:: 2.2.0

       pandas now supports reading via ADBC drivers

    >>> from adbc_driver_postgresql import dbapi  # doctest:+SKIP
    >>> with dbapi.connect("postgres:///db_name") as conn:  # doctest:+SKIP
    ...     pd.read_sql("SELECT int_column FROM test_data", conn)
       int_column
    0           0
    1           1
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        if isinstance(pandas_sql, SQLiteDatabase):
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )

        try:
            _is_table_name = pandas_sql.has_table(sql)
        except Exception:
            # using generic exception to catch errors from sql drivers (GH24988)
            _is_table_name = False

        if _is_table_name:
            return pandas_sql.read_table(
                sql,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
            )
        else:
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )


def to_sql(
    frame,
    name: str,
    con,
    schema: str | None = None,
    if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
    index: bool = True,
    index_label: IndexLabel | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    method: Literal["multi"] | Callable | None = None,
    engine: str = "auto",
    **engine_kwargs,
) -> int | None:
    # [FUNCTION DOC]
    # Writes a frame (or series) to a database table and returns the number of rows
    # affected.
    # Parameters: frame is a frame or series (a series is converted to a one-column
    # frame; anything else raises a not-implemented error); name is the table; con is a
    # connection or URI as for the readers; schema is the schema; if_exists is fail,
    # replace, append or delete_rows (a value error for anything else); index (default
    # true) writes the index as columns; index_label names the index column(s);
    # chunksize is rows per batch; dtype maps column names to database types (or a single
    # type for all); method is none (one insert per row), the multi-value keyword, or a
    # callable taking the table, connection, column names and a row iterator; engine is
    # auto or sqlalchemy; extra keyword arguments go to the engine.
    # Behavior: validates arguments, builds the database helper within a transaction and
    # delegates to its writer. Returns the count, or none when a custom method returns no
    # count.
    """
    Write records stored in a DataFrame to a SQL database.

    .. warning::
        The pandas library does not attempt to sanitize inputs provided via a to_sql call.
        Please refer to the documentation for the underlying database driver to see if it
        will properly prevent injection, or alternatively be advised of a security risk when
        executing arbitrary commands in a to_sql call.

    Parameters
    ----------
    frame : DataFrame, Series
    name : str
        Name of SQL table.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        or sqlite3 DBAPI2 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : str, optional
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    if_exists : {'fail', 'replace', 'append', 'delete_rows'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
        - delete_rows: If a table exists, delete all records and insert data.
    index : bool, default True
        Write DataFrame index as a column.
    index_label : str or sequence, optional
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.
    chunksize : int, optional
        Specify the number of rows in each batch to be written at a time.
        By default, all rows will be written at once.
    dtype : dict or scalar, optional
        Specifying the datatype for columns. If a dictionary is used, the
        keys should be the column names and the values should be the
        SQLAlchemy types or strings for the sqlite3 fallback mode. If a
        scalar is provided, it will be applied to all columns.
    method : {None, 'multi', callable}, optional
        Controls the SQL insertion clause used:

        - None : Uses standard SQL ``INSERT`` clause (one per row).
        - ``'multi'``: Pass multiple values in a single ``INSERT`` clause.
        - callable with signature ``(pd_table, conn, keys, data_iter) -> int | None``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.
    engine : {'auto', 'sqlalchemy'}, default 'auto'
        SQL engine library to use. If 'auto', then the option
        ``io.sql.engine`` is used. The default ``io.sql.engine``
        behavior is 'sqlalchemy'

    **engine_kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    None or int
        Number of rows affected by to_sql. None is returned if the callable
        passed into ``method`` does not return an integer number of rows.

    Notes
    -----
    The returned rows affected is the sum of the ``rowcount`` attribute of ``sqlite3.Cursor``
    or SQLAlchemy connectable. If using ADBC the returned rows are the result
    of ``Cursor.adbc_ingest``. The returned value may not reflect the exact number of written
    rows as stipulated in the
    `sqlite3 <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.rowcount>`__ or
    `SQLAlchemy <https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.BaseCursorResult.rowcount>`__
    """  # noqa: E501
    if if_exists not in ("fail", "replace", "append", "delete_rows"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError(
            "'frame' argument should be either a Series or a DataFrame"
        )

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        return pandas_sql.to_sql(
            frame,
            name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            engine=engine,
            **engine_kwargs,
        )


def has_table(table_name: str, con, schema: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Returns true if the database has a table of the given name. Parameters: table_name;
    # con as elsewhere; schema (default none) optionally restricts the lookup. Builds a
    # database helper and asks it. Note: a module level alias with a different name
    # refers to this function.
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        return pandas_sql.has_table(table_name)


table_exists = has_table


def pandasSQL_builder(
    con,
    schema: str | None = None,
    need_transaction: bool = False,
) -> PandasSQL:
    # [FUNCTION DOC]
    # Chooses and creates the database helper object for a connection.
    # Parameters: con is the connection; schema is the default schema; need_transaction
    # says whether a transaction should be opened for the lifetime of the helper.
    # Behavior in order: a sqlite3 connection (or none) gives the sqlite helper. If
    # sqlalchemy is not installed and con is a URI text, raises an import error. If
    # sqlalchemy is installed and con is URI text or a SQLAlchemy connectable, returns the
    # SQLAlchemy helper. If the ADBC driver manager is installed and con is an ADBC
    # connection, returns the ADBC helper. Otherwise warns that only SQLAlchemy, URI text
    # and sqlite3 are supported and returns the sqlite helper for the object.
    # Side effects: may open connections and transactions (via the helper), may warn.
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.
    """
    import sqlite3

    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)

    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")

    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError(
            f"Using URI string without version '{VERSIONS['sqlalchemy']}' or newer "
            "of 'sqlalchemy' installed."
        )

    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)

    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if adbc and isinstance(con, adbc.Connection):
        return ADBCDatabase(con)

    warnings.warn(
        "pandas only supports SQLAlchemy connectable (engine/connection) or "
        "database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 "
        "objects are not tested. Please consider using SQLAlchemy.",
        UserWarning,
        stacklevel=find_stack_level(),
    )
    return SQLiteDatabase(con)


class SQLTable(PandasObject):
    # [CLASS DOC]
    # Maps one pandas frame (or an existing database table) to a SQLAlchemy table.
    # State: table name, owning database helper (called the pandas SQL engine), the frame
    # being written or last read, the index column names (computed), the schema, the
    # if-exists policy, optional primary key columns, an optional dtype mapping or
    # single dtype, a prefix, and the SQLAlchemy table object. With a frame the table
    # description is built from the frame (write mode); without a frame it is reflected
    # from the database (read mode).
    # Key methods: create (honoring if-exists), insert (batched, with pluggable insertion
    # method), read (select into a frame, possibly chunked), conversion helpers between
    # pandas and SQL types, and the column type harmonizer.
    # # KNOWN ISSUE: a comment above the initializer says support for multi-level
    # indexes is a to-do, but multi-level index labels are handled by the index
    # naming code.
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """

    # TODO: support for multiIndex

    def __init__(
        self,
        name: str,
        pandas_sql_engine,
        frame=None,
        index: bool | str | list[str] | None = True,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        prefix: str = "pandas",
        index_label=None,
        schema=None,
        keys=None,
        dtype: DtypeArg | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the mapping. Parameters: name is the table name; pandas_sql_engine is the
        # database helper; frame is the frame to write (none for read mode); index is true
        # (write the frame index), a column name or list (read mode: the columns to use as
        # index), or false or none; if_exists is fail, replace, append or delete_rows;
        # prefix is stored but unused here; index_label overrides index column names when
        # writing; schema is the schema; keys are primary key columns; dtype is a mapping or
        # single SQL type.
        # Behavior: stores the arguments, computes the index column names, and builds the
        # table description from the frame or reflects it from the database. Raises a value
        # error if no table could be obtained or the table name is empty (the name check
        # happens after the table was set up).
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype

        if frame is not None:
            # We want to initialize based on a dataframe
            self.table = self._create_table_setup()
        else:
            # no data provided, read-only mode
            self.table = self.pd_sql.get_table(self.name, self.schema)

        if self.table is None:
            raise ValueError(f"Could not init table '{name}'")

        if not len(self.name):
            raise ValueError("Empty table name specified")

    def exists(self):
        # [FUNCTION DOC]
        # Returns whether the table already exists in the database (asks the helper with the
        # name and schema).
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self) -> str:
        # [FUNCTION DOC]
        # Returns the CREATE TABLE statement text for the table, compiled for the helper's
        # connection.
        from sqlalchemy.schema import CreateTable

        return str(CreateTable(self.table).compile(self.pd_sql.con))

    def _execute_create(self) -> None:
        # [FUNCTION DOC]
        # HELPER: creates the table in the database. Re-attaches the table description to the
        # helper's metadata and issues the create inside a transaction.
        # Inserting table into database, add to MetaData object
        self.table = self.table.to_metadata(self.pd_sql.meta)
        with self.pd_sql.run_transaction():
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        # [FUNCTION DOC]
        # Prepares the database table according to the if-exists policy. If the table exists:
        # fail raises a value error; replace drops it and creates it again; append does
        # nothing; delete_rows removes all rows keeping the table; any other policy raises a
        # value error. If the table does not exist, creates it.
        if self.exists():
            if self.if_exists == "fail":
                raise ValueError(f"Table '{self.name}' already exists.")
            elif self.if_exists == "replace":
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == "append":
                pass
            elif self.if_exists == "delete_rows":
                self.pd_sql.delete_rows(self.name, self.schema)
            else:
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            self._execute_create()

    def _execute_insert(self, conn, keys: list[str], data_iter) -> int:
        # [FUNCTION DOC]
        # HELPER: default insertion method. Parameters: conn is unused (the helper executes on
        # its own connection); keys are column names; data_iter yields row sequences. Builds a
        # list of mappings from column names to values and executes one insert statement with
        # all of them. Returns the reported number of rows affected.
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
           Column names
        data_iter : generator of list
           Each item contains a list of values to be inserted
        """
        data = [dict(zip(keys, row, strict=True)) for row in data_iter]
        result = self.pd_sql.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn, keys: list[str], data_iter) -> int:
        # [FUNCTION DOC]
        # HELPER: multi-value insertion method. Same parameters as the default method. Builds
        # a single insert statement carrying all rows as values and executes it. Intended for
        # databases where this is faster, with columns that are few. Returns the reported row
        # count.
        """
        Alternative to _execute_insert for DBs support multi-value INSERT.

        Note: multi-value insert is usually faster for analytics DBs
        and tables containing a few columns
        but performance degrades quickly with increase of columns.

        """

        from sqlalchemy import insert

        data = [dict(zip(keys, row, strict=True)) for row in data_iter]
        stmt = insert(self.table).values(data)
        result = self.pd_sql.execute(stmt)
        return result.rowcount

    def insert_data(self) -> tuple[list[str], list[np.ndarray]]:
        # [FUNCTION DOC]
        # HELPER: prepares the frame content for insertion.
        # Behavior: if an index is to be written, makes a shallow copy of the frame, assigns
        # the index column names and resets the index into columns (raising a value error that
        # mentions duplicate names if that fails); otherwise uses the frame as is. Column
        # names are converted to text. Each column becomes an object array: datetime columns
        # become arrays of Python datetime objects (Arrow date columns become object arrays of
        # dates), timedelta columns become integers (nanoseconds, as objects), and everything
        # else is converted to objects. Columns that can hold missing values get missing
        # entries replaced by none.
        # Returns a pair: the list of column names and the list of per-column object arrays.
        if self.index is not None:
            temp = self.frame.copy(deep=False)
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f"duplicate name in index/columns: {err}") from err
        else:
            temp = self.frame

        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        # this just pre-allocates the list: None's will be replaced with ndarrays
        # error: List item 0 has incompatible type "None"; expected "ndarray"
        data_list: list[np.ndarray] = [None] * ncols  # type: ignore[list-item]

        for i, (_, ser) in enumerate(temp.items()):
            if ser.dtype.kind == "M":
                if isinstance(ser._values, ArrowExtensionArray):
                    import pyarrow as pa

                    if pa.types.is_date(ser.dtype.pyarrow_dtype):
                        # GH#53854 to_pydatetime not supported for pyarrow date dtypes
                        d = ser._values.to_numpy(dtype=object)
                    else:
                        d = ser.dt.to_pydatetime()._values
                else:
                    d = ser._values.to_pydatetime()
            elif ser.dtype.kind == "m":
                vals = ser._values
                if isinstance(vals, ArrowExtensionArray):
                    vals = vals.to_numpy(dtype=np.dtype("m8[ns]"))
                # store as integers, see GH#6921, GH#7076
                d = vals.view("i8").astype(object)
            else:
                d = ser._values.astype(object)

            assert isinstance(d, np.ndarray), type(d)

            if ser._can_hold_na:
                # Note: this will miss timedeltas since they are converted to int
                mask = isna(d)
                d[mask] = None

            data_list[i] = d

        return column_names, data_list

    def insert(
        self,
        chunksize: int | None = None,
        method: Literal["multi"] | Callable | None = None,
    ) -> int | None:
        # [FUNCTION DOC]
        # Inserts the frame's rows into the table.
        # Parameters: chunksize is the number of rows per batch (default all rows; zero raises
        # a value error); method is none (one statement with many parameter sets), the
        # multi-value keyword, or a callable which receives the table, connection, column names
        # and row iterator; any other value raises a value error.
        # Behavior: selects the insert function, prepares the data, returns zero if the frame is
        # empty, then inside a transaction iterates over chunks, passing each to the insert
        # function and summing the returned row counts.
        # Returns the total rows reported, or none if the insert function never reports a count.
        # set insert method
        if method is None:
            exec_insert = self._execute_insert
        elif method == "multi":
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f"Invalid parameter `method`: {method}")

        keys, data_list = self.insert_data()

        nrows = len(self.frame)

        if nrows == 0:
            return 0

        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError("chunksize argument should be non-zero")

        chunks = (nrows // chunksize) + 1
        total_inserted = None
        with self.pd_sql.run_transaction() as conn:
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break

                chunk_iter = zip(
                    *(arr[start_i:end_i] for arr in data_list), strict=True
                )
                num_inserted = exec_insert(conn, keys, chunk_iter)
                # GH 46891
                if num_inserted is not None:
                    if total_inserted is None:
                        total_inserted = num_inserted
                    else:
                        total_inserted += num_inserted
        return total_inserted

    def _query_iterator(
        self,
        result,
        exit_stack: ExitStack,
        chunksize: int | None,
        columns,
        coerce_float: bool = True,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> Generator[DataFrame]:
        # [FUNCTION DOC]
        # HELPER: generator that yields a table read in chunks.
        # Parameters: result is the open result cursor; exit_stack is the stack of resources to
        # release when finished; chunksize is the rows per chunk; columns are the column names;
        # coerce_float, parse_dates, dtype_backend as in the reader.
        # Behavior: inside the exit stack, repeatedly fetches a chunk; if the very first fetch is
        # empty yields a single empty frame with the columns; for each chunk converts to a
        # frame, stores it in the frame attribute, harmonizes column types with the table
        # definition, sets the index if any, and yields it. Closes resources at the end.
        """Return generator through chunked result set."""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield DataFrame.from_records(
                            [], columns=columns, coerce_float=coerce_float
                        )
                    break

                has_read_data = True
                self.frame = _convert_arrays_to_dataframe(
                    data, columns, coerce_float, dtype_backend
                )

                self._harmonize_columns(
                    parse_dates=parse_dates, dtype_backend=dtype_backend
                )

                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)

                yield self.frame

    def read(
        self,
        exit_stack: ExitStack,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Reads the table into a frame.
        # Parameters: exit_stack holds resources closed after chunked iteration; coerce_float,
        # parse_dates, dtype_backend as in the module readers; columns restricts the selection
        # (index columns are prepended in order); chunksize (none means all rows).
        # Behavior: builds a select of the chosen columns or the whole table, executes it, and
        # takes the column names from the result. With a chunk size returns the chunk generator;
        # otherwise fetches all rows, builds the frame, harmonizes types with the table
        # definition, sets the index if any, and returns it (also stored in the frame attribute).
        from sqlalchemy import select

        if columns is not None and len(columns) > 0:
            cols = [self.table.c[n] for n in columns]
            if self.index is not None:
                for idx in self.index[::-1]:
                    cols.insert(0, self.table.c[idx])
            sql_select = select(*cols)
        else:
            sql_select = select(self.table)
        result = self.pd_sql.execute(sql_select)
        column_names = result.keys()

        if chunksize is not None:
            return self._query_iterator(
                result,
                exit_stack,
                chunksize,
                column_names,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            self.frame = _convert_arrays_to_dataframe(
                data, column_names, coerce_float, dtype_backend
            )

            self._harmonize_columns(
                parse_dates=parse_dates, dtype_backend=dtype_backend
            )

            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)

            return self.frame

    def _index_name(self, index, index_label):
        # [FUNCTION DOC]
        # HELPER: determines the list of index column names.
        # When index is true (writing): with an index label, wraps a non-list label into a list
        # and requires its length to equal the number of index levels (value error otherwise);
        # without a label uses the single name index when there is one unnamed level and no
        # column already named index, else the index level names with missing names filled with
        # generated defaults. When index is a text it becomes a one-element list; a list is
        # returned unchanged; anything else (including false and none) yields none.
        # for writing: index=True to include index in sql table
        if index is True:
            nlevels = self.frame.index.nlevels
            # if index_label is specified, set this as index name(s)
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError(
                        "Length of 'index_label' should match number of "
                        f"levels, which is {nlevels}"
                    )
                return index_label
            # return the used column labels for the index columns
            if (
                nlevels == 1
                and "index" not in self.frame.columns
                and self.frame.index.name is None
            ):
                return ["index"]
            else:
                return com.fill_missing_names(self.frame.index.names)

        # for reading: index=(list of) string to specify column to set as index
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(self, dtype_mapper):
        # [FUNCTION DOC]
        # HELPER: lists, for the table columns, triples of name text, mapped type and whether
        # it is an index column. Parameter dtype_mapper maps an index level or a column to a
        # database type. Index levels come first (named by the index labels), followed by the
        # frame columns.
        column_names_and_types = []
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
                column_names_and_types.append((str(idx_label), idx_type, True))

        column_names_and_types += [
            (str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False)
            for i in range(len(self.frame.columns))
        ]

        return column_names_and_types

    def _create_table_setup(self):
        # [FUNCTION DOC]
        # HELPER: builds the SQLAlchemy table description from the frame. Maps each index
        # level and column to a column type; index columns are flagged as indexed. If primary
        # keys are given (a single name or list) adds a primary key constraint named after the
        # table with a pk suffix. The schema is the table's schema or the helper's default.
        # The table is attached to a fresh metadata object and only moved to the helper's
        # metadata when it is actually created. Returns the table.
        from sqlalchemy import (
            Column,
            PrimaryKeyConstraint,
            Table,
        )
        from sqlalchemy.schema import MetaData

        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)

        columns: list[Any] = [
            Column(name, typ, index=is_index)
            for name, typ, is_index in column_names_and_types
        ]

        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + "_pk")
            columns.append(pkc)

        schema = self.schema or self.pd_sql.meta.schema

        # At this point, attach to new metadata, only attach to self.meta
        # once table is created.
        meta = MetaData()
        return Table(self.name, meta, *columns, schema=schema)

    def _harmonize_columns(
        self,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: aligns the types of the read frame columns with the SQL column types.
        # Parameters: parse_dates is the date-parsing argument; dtype_backend is numpy or other.
        # For each SQL column present in the frame: columns listed in the date argument are
        # parsed (with their given format) and skipped otherwise; otherwise the SQL type is
        # mapped to a pandas type, and datetime and date columns are parsed (time-zone-aware
        # columns to UTC); with the numpy backend float columns are cast to float; under the
        # string dtype option object columns of string SQL type are cast to the string dtype;
        # with the numpy backend and no missing values, integer and boolean columns are cast. A
        # column absent from the frame is skipped silently. Modifies the frame in place.
        """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
        parse_dates = _process_parse_dates_argument(parse_dates)

        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]

                # Handle date parsing upfront; don't try to convert columns
                # twice
                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue

                # the type the dataframe column should have
                col_type = self._get_dtype(sql_col.type)

                if (
                    col_type is datetime
                    or col_type is date
                    or col_type is DatetimeTZDtype
                ):
                    # Convert tz-aware Datetime SQL columns to UTC
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif dtype_backend == "numpy" and col_type is float:
                    # floats support NA, can always convert!
                    self.frame[col_name] = df_col.astype(col_type)
                elif (
                    using_string_dtype()
                    and is_string_dtype(col_type)
                    and is_object_dtype(self.frame[col_name])
                ):
                    self.frame[col_name] = df_col.astype(col_type)
                elif dtype_backend == "numpy" and len(df_col) == df_col.count():
                    # No NA values, can convert ints and bools
                    if col_type is np.dtype("int64") or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type)
            except KeyError:
                pass  # this column not in results

    def _sqlalchemy_type(self, col: Index | Series):
        # [FUNCTION DOC]
        # HELPER: chooses the SQL type for a pandas column or index level.
        # Behavior: if the dtype argument given at construction is mapping-like and has an entry
        # for the column name, that entry is returned. Otherwise infers the kind of data while
        # ignoring missing values: datetimes give a timestamp-with-time-zone type when
        # time-zone-aware, else the plain datetime type; timedeltas warn that they are not
        # supported and are stored as big integers; floats give a float of precision 23 (32 bit)
        # or 53; integers give small integer for 8-bit and signed 16-bit, integer for 32-bit and
        # unsigned 16-bit, big integer otherwise, and unsigned 64-bit raises a value error;
        # booleans, dates and times give the matching types; complex raises a value error;
        # anything else is text.
        # Note: a non-mapping dtype argument is not consulted here.
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        from sqlalchemy.types import (
            TIMESTAMP,
            BigInteger,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            SmallInteger,
            Text,
            Time,
        )

        if col_type in ("datetime64", "datetime"):
            # GH 9086: TIMESTAMP is the suggested type if the column contains
            # timezone information
            try:
                # error: Item "Index" of "Union[Index, Series]" has no attribute "dt"
                if col.dt.tz is not None:  # type: ignore[union-attr]
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                # The column is actually a DatetimeIndex
                # GH 26761 or an Index with date-like data e.g. 9999-01-01
                if getattr(col, "tz", None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime
        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return BigInteger
        elif col_type == "floating":
            if col.dtype == "float32":
                return Float(precision=23)
            else:
                return Float(precision=53)
        elif col_type == "integer":
            # GH35076 Map pandas integer to optimal SQLAlchemy integer type
            if col.dtype.name.lower() in ("int8", "uint8", "int16"):
                return SmallInteger
            elif col.dtype.name.lower() in ("uint16", "int32"):
                return Integer
            elif col.dtype.name.lower() == "uint64":
                raise ValueError("Unsigned 64 bit integer datatype is not supported")
            else:
                return BigInteger
        elif col_type == "boolean":
            return Boolean
        elif col_type == "date":
            return Date
        elif col_type == "time":
            return Time
        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        return Text

    def _get_dtype(self, sqltype):
        # [FUNCTION DOC]
        # HELPER: maps a SQLAlchemy column type to the pandas-side type used when reading.
        # Float gives the float type; integer gives 64-bit integer (size refinement is a to-do);
        # timestamp gives datetime, or the time-zone-aware dtype class when the type has a time
        # zone; plain datetime gives datetime; date gives date; boolean gives bool; string gives
        # the NaN-based string dtype when the string dtype option is on; everything else gives
        # the object type. Returns the type object.
        from sqlalchemy.types import (
            TIMESTAMP,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            String,
        )

        if isinstance(sqltype, Float):
            return float
        elif isinstance(sqltype, Integer):
            # TODO: Refine integer size.
            return np.dtype("int64")
        elif isinstance(sqltype, TIMESTAMP):
            # we have a timezone capable type
            if not sqltype.timezone:
                return datetime
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            # Caution: np.datetime64 is also a subclass of np.number.
            return datetime
        elif isinstance(sqltype, Date):
            return date
        elif isinstance(sqltype, Boolean):
            return bool
        elif isinstance(sqltype, String):
            if using_string_dtype():
                return StringDtype(na_value=np.nan)

        return object


class PandasSQL(PandasObject, ABC):
    # [CLASS DOC]
    # Abstract base class for the database helpers (SQLAlchemy, ADBC, sqlite). Defines the
    # interface: read a table (not abstract; default raises not-implemented), run a query,
    # write a frame, execute a statement, test table existence and create schema text. Is a
    # context manager whose exit does nothing by default.
    """
    Subclasses Should define read_query and to_sql.
    """

    def __enter__(self) -> Self:
        # [FUNCTION DOC]
        # Context manager entry: returns the helper itself.
        return self

    def __exit__(self, *args) -> None:
        # [FUNCTION DOC]
        # Context manager exit: does nothing in the base class; arguments are ignored.
        pass

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Reads a table into a frame or iterator (parameters as for the module level table reader
        # plus a schema). The base implementation always raises a not-implemented error;
        # subclasses that support tables override it.
        raise NotImplementedError

    @abstractmethod
    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Abstract: runs a query and returns a frame or an iterator of frames (parameters as for
        # the module level query reader).
        pass

    @abstractmethod
    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        # [FUNCTION DOC]
        # Abstract: writes a frame to a table and returns the number of rows affected or none
        # (parameters as for the module level writer).
        pass

    @abstractmethod
    def execute(self, sql: str | Select | TextClause, params=None):
        # [FUNCTION DOC]
        # Abstract: executes a statement (text or SQLAlchemy construct) with optional
        # parameters and returns the driver's result.
        pass

    @abstractmethod
    def has_table(self, name: str, schema: str | None = None) -> bool:
        # [FUNCTION DOC]
        # Abstract: returns whether a table of the given name exists, optionally within a schema.
        pass

    @abstractmethod
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Abstract: returns the text of the table creation statement for a frame, with optional
        # key columns, dtype mapping and schema.
        pass


class BaseEngine:
    # [CLASS DOC]
    # Base class for write engines, objects that insert prepared rows into a table. Defines
    # a single method that subclasses override.
    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        # [FUNCTION DOC]
        # Abstract: inserts records into an already prepared table. Parameters: table is the table
        # mapping; con, frame, name, index, schema are for engine use; chunksize and method
        # control the insertion. Always raises an abstract-method error in the base class.
        """
        Inserts data into already-prepared table
        """
        raise AbstractMethodError(self)


class SQLAlchemyEngine(BaseEngine):
    # [CLASS DOC]
    # Write engine based on SQLAlchemy; the only engine implemented. Stateless.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Checks that SQLAlchemy is importable, raising an import error with a message saying it
        # is required for SQL support.
        import_optional_dependency(
            "sqlalchemy", extra="sqlalchemy is required for SQL support."
        )

    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        # [FUNCTION DOC]
        # Inserts the rows through the table mapping's insert method (parameters chunksize and
        # method are forwarded; the connection, frame, name, index, schema and extra engine
        # arguments are not used). If SQLAlchemy raises a statement error whose original text
        # shows a MySQL infinity problem, re-raises it as a value error saying infinity cannot
        # be used with MySQL; other statement errors propagate. Returns the number of rows
        # inserted or none.
        from sqlalchemy import exc

        try:
            return table.insert(chunksize=chunksize, method=method)
        except exc.StatementError as err:
            # GH34431
            # https://stackoverflow.com/a/67358288/6067848
            msg = r"""(\(1054, "Unknown column 'inf(e0)?' in 'field list'"\))(?#
            )|inf can not be used with MySQL"""
            err_text = str(err.orig)
            if re.search(msg, err_text):
                raise ValueError("inf cannot be used with MySQL") from err
            raise err


def get_engine(engine: str) -> BaseEngine:
    # [FUNCTION DOC]
    # HELPER: returns a write engine object. The argument auto reads the sql engine option
    # and, if that is also auto, tries the known engines in order (only SQLAlchemy), raising
    # an import error listing the failures if none works; sqlalchemy returns the SQLAlchemy
    # engine. Anything else raises a value error.
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.sql.engine")

    if engine == "auto":
        # try engines in this order
        engine_classes = [SQLAlchemyEngine]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'sqlalchemy'.\n"
            "A suitable version of "
            "sqlalchemy is required for sql I/O "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "sqlalchemy":
        return SQLAlchemyEngine()

    raise ValueError("engine must be one of 'auto', 'sqlalchemy'")


class SQLDatabase(PandasSQL):
    # [CLASS DOC]
    # Database helper using SQLAlchemy (subclass of the abstract helper). Holds the
    # connection, a metadata object carrying the default schema, an exit stack that releases
    # any engine or connection and commits any transaction this helper created, and a flag
    # saying whether a read returned a lazy generator (in which case the resources must
    # outlive the context manager and are closed by the generator).
    # Key methods: run_transaction, execute, read_table, read_query, to_sql, has_table,
    # get_table, drop_table, delete_rows, schema creation text, and a case sensitivity check.
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.

    """

    def __init__(
        self, con, schema: str | None = None, need_transaction: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: con is a URI text, an engine or a connection; schema is the default schema;
        # need_transaction (default false) opens a transaction held until the helper is closed.
        # Behavior: a URI text creates an engine (disposed on close); an engine is replaced by a
        # connection opened through the exit stack; if a transaction is needed and none is
        # active on the connection, begins one through the exit stack. Stores the connection, a
        # metadata object with the schema, and the generator flag set to false.
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.schema import MetaData

        # self.exit_stack cleans up the Engine and Connection and commits the
        # transaction if any of those objects was created below.
        # Cleanup happens either in self.__exit__ or at the end of the iterator
        # returned by read_sql when chunksize is not None.
        self.exit_stack = ExitStack()
        if isinstance(con, str):
            con = create_engine(con)
            self.exit_stack.callback(con.dispose)
        if isinstance(con, Engine):
            con = self.exit_stack.enter_context(con.connect())
        if need_transaction and not con.in_transaction():
            self.exit_stack.enter_context(con.begin())
        self.con = con
        self.meta = MetaData(schema=schema)
        self.returns_generator = False

    def __exit__(self, *args) -> None:
        # [FUNCTION DOC]
        # Context manager exit: closes the exit stack (committing and releasing resources the
        # helper created) unless a lazy generator took over responsibility for cleanup; arguments
        # are ignored.
        if not self.returns_generator:
            self.exit_stack.close()

    @contextmanager
    def run_transaction(self):
        # [FUNCTION DOC]
        # Context manager yielding the connection. If no transaction is active starts one that
        # commits at the end of the block (or rolls back on error); if one is already active just
        # yields the connection and leaves it alone.
        if not self.con.in_transaction():
            with self.con.begin():
                yield self.con
        else:
            yield self.con

    def execute(self, sql: str | Select | TextClause | Delete, params=None):
        # [FUNCTION DOC]
        # Executes a statement and returns the result. A text statement is executed as raw driver
        # SQL, a SQLAlchemy construct through the connection. Parameters, if given, are passed
        # along. Any SQLAlchemy error is re-raised as the library's database error with a message
        # that includes the statement.
        """Simple passthrough to SQLAlchemy connectable"""
        from sqlalchemy.exc import SQLAlchemyError

        args = [] if params is None else [params]
        if isinstance(sql, str):
            execute_function = self.con.exec_driver_sql
        else:
            execute_function = self.con.execute

        try:
            return execute_function(sql, *args)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Execution failed on sql '{sql}': {exc}") from exc

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Reads a table through SQLAlchemy.
        # Parameters: table_name; index_col (column or list); coerce_float; parse_dates;
        # columns to select; schema (stored in the table mapping); chunksize (none means all
        # rows); dtype_backend (numpy or other).
        # Behavior: reflects the named table or view into the helper's metadata, builds a
        # read-mode table mapping, flags that a lazy generator will be returned when
        # chunksize is given (so the connection is kept open until iteration finishes), and
        # delegates to the mapping's read method with the helper's exit stack.
        # Returns a frame, or a generator of frames when chunked.
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        index_col : string, optional, default: None
            Column to set as index.
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype_backend : {'numpy_nullable', 'pyarrow'}
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). If not specified, the default behavior
            is to not use nullable data types. If specified, the behavior
            is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
            * ``"pyarrow"``: returns pyarrow-backed nullable
              :class:`ArrowDtype` :class:`DataFrame`

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        self.meta.reflect(bind=self.con, only=[table_name], views=True)
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        if chunksize is not None:
            self.returns_generator = True
        return table.read(
            self.exit_stack,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    @staticmethod
    def _query_iterator(
        result,
        exit_stack: ExitStack,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> Generator[DataFrame]:
        # [FUNCTION DOC]
        # HELPER: static generator yielding chunks of a query result.
        # Parameters: result is the open result; exit_stack holds resources released at the end;
        # chunksize is the rows per chunk; columns are column names; index_col, coerce_float,
        # parse_dates, dtype, dtype_backend as in the module readers.
        # Behavior: inside the exit stack, repeatedly fetches chunks; if the very first fetch is
        # empty yields one empty finished frame; otherwise wraps each chunk into a frame (with
        # dtype cast, date parsing and index). Resources are released when the generator ends.
        """Return generator through chunked result set"""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield _wrap_result(
                            [],
                            columns,
                            index_col=index_col,
                            coerce_float=coerce_float,
                            parse_dates=parse_dates,
                            dtype=dtype,
                            dtype_backend=dtype_backend,
                        )
                    break

                has_read_data = True
                yield _wrap_result(
                    data,
                    columns,
                    index_col=index_col,
                    coerce_float=coerce_float,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    dtype_backend=dtype_backend,
                )

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Runs a query through SQLAlchemy and returns frames.
        # Parameters: sql is query text or a SQLAlchemy statement; index_col, coerce_float,
        # parse_dates, params, chunksize, dtype, dtype_backend as in the module query reader.
        # Behavior: executes the statement with the parameters, takes the column names from the
        # result. With a chunk size sets the lazy generator flag and returns the chunk generator
        # using the helper's exit stack. Otherwise fetches all rows and returns one finished
        # frame. The class attribute alias read_sql refers to this method.
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        result = self.execute(sql, params)
        columns = result.keys()

        if chunksize is not None:
            self.returns_generator = True
            return self._query_iterator(
                result,
                self.exit_stack,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    read_sql = read_query

    def prep_table(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: bool | str | list[str] | None = True,
        index_label=None,
        schema=None,
        dtype: DtypeArg | None = None,
    ) -> SQLTable:
        # [FUNCTION DOC]
        # Prepares the database table for an upcoming insert and returns its mapping.
        # Parameters: frame; name; if_exists (fail, replace, append, delete_rows); index (true or
        # column names); index_label; schema; dtype (a single type, or a mapping).
        # Behavior: if a dtype is given, a single (non-mapping) type is expanded into a mapping
        # for all frame columns (index columns are not included); each mapped type must be a
        # SQLAlchemy type class or instance, otherwise a value error names the column. Builds the
        # table mapping and creates the table according to the if-exists policy. Returns the
        # mapping.
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = dict.fromkeys(frame, dtype)  # type: ignore[arg-type]
            else:
                dtype = cast(dict, dtype)

            from sqlalchemy.types import TypeEngine

            for col, my_type in dtype.items():
                if isinstance(my_type, type) and issubclass(my_type, TypeEngine):
                    pass
                elif isinstance(my_type, TypeEngine):
                    pass
                else:
                    raise ValueError(f"The type of {col} is not a SQLAlchemy type")

        table = SQLTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )
        table.create()
        return table

    def check_case_sensitive(
        self,
        name: str,
        schema: str | None,
    ) -> None:
        # [FUNCTION DOC]
        # Warns if the table name might have been affected by case sensitivity. Parameters:
        # name; schema. Runs only for names that are neither all digits nor all lower case: after
        # writing, looks up the actual table names in the database (default schema if none) and
        # issues a user warning if the exact name is not among them. Returns nothing.
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
        if not name.isdigit() and not name.islower():
            # check for potentially case sensitivity issues (GH7815)
            # Only check when name is not a number and name is not lower case
            from sqlalchemy import inspect as sqlalchemy_inspect

            insp = sqlalchemy_inspect(self.con)
            table_names = insp.get_table_names(schema=schema or self.meta.schema)
            if name not in table_names:
                msg = (
                    f"The provided table name '{name}' is not found exactly as "
                    "such in the database after writing the table, possibly "
                    "due to case sensitivity issues. Consider using lower "
                    "case table names."
                )
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        # [FUNCTION DOC]
        # Writes a frame to a table.
        # Parameters: frame; name; if_exists; index; index_label; schema; chunksize; dtype; method;
        # engine (auto or sqlalchemy); extra keyword arguments passed to the engine.
        # Behavior: gets the write engine, prepares the table (creating or clearing it per the
        # policy), inserts the records through the engine (with the connection and chunking
        # options), then checks case sensitivity of the name. Returns the number of rows inserted,
        # or none if the insertion method does not report a count.
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append', 'delete_rows'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
            - delete_rows: If a table exists, delete all records and insert data.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.
        method : {None', 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            SQL engine library to use. If 'auto', then the option
            ``io.sql.engine`` is used. The default ``io.sql.engine``
            behavior is 'sqlalchemy'

        **engine_kwargs
            Any additional kwargs are passed to the engine.
        """
        sql_engine = get_engine(engine)

        table = self.prep_table(
            frame=frame,
            name=name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )

        total_inserted = sql_engine.insert_records(
            table=table,
            con=self.con,
            frame=frame,
            name=name,
            index=index,
            schema=schema,
            chunksize=chunksize,
            method=method,
            **engine_kwargs,
        )

        self.check_case_sensitive(name=name, schema=schema)
        return total_inserted

    @property
    def tables(self):
        # [FUNCTION DOC]
        # Property: the mapping of table names to table objects held in the helper's metadata.
        return self.meta.tables

    def has_table(self, name: str, schema: str | None = None) -> bool:
        # [FUNCTION DOC]
        # Returns whether the table (or view) exists, using a SQLAlchemy inspector on the
        # connection, in the given schema or the helper's default schema.
        from sqlalchemy import inspect as sqlalchemy_inspect

        insp = sqlalchemy_inspect(self.con)
        return insp.has_table(name, schema or self.meta.schema)

    def get_table(self, table_name: str, schema: str | None = None) -> Table:
        # [FUNCTION DOC]
        # Reflects and returns the table object from the database in the given or default schema
        # (registered in the helper's metadata). Numeric columns are set to return floating point
        # instead of decimals.
        from sqlalchemy import (
            Numeric,
            Table,
        )

        schema = schema or self.meta.schema
        tbl = Table(table_name, self.meta, autoload_with=self.con, schema=schema)
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    def drop_table(self, table_name: str, schema: str | None = None) -> None:
        # [FUNCTION DOC]
        # Drops the table if it exists (given or default schema): reflects it, drops it inside a
        # transaction and clears the metadata. Does nothing if it does not exist.
        schema = schema or self.meta.schema
        if self.has_table(table_name, schema):
            self.meta.reflect(
                bind=self.con, only=[table_name], schema=schema, views=True
            )
            with self.run_transaction():
                self.get_table(table_name, schema).drop(bind=self.con)
            self.meta.clear()

    def delete_rows(self, table_name: str, schema: str | None = None) -> None:
        # [FUNCTION DOC]
        # Deletes all rows of the table if it exists (given or default schema): reflects it, runs
        # a delete statement and closes the result, then clears the metadata. Does nothing if
        # the table does not exist.
        schema = schema or self.meta.schema
        if self.has_table(table_name, schema):
            self.meta.reflect(
                bind=self.con, only=[table_name], schema=schema, views=True
            )
            table = self.get_table(table_name, schema)
            self.execute(table.delete()).close()
            self.meta.clear()

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Returns the table creation statement text for a frame. Parameters: frame; table_name;
        # keys are primary key columns; dtype; schema. Builds a table mapping without index
        # columns and returns its schema text.
        table = SQLTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


# ---- SQL without SQLAlchemy ---


class ADBCDatabase(PandasSQL):
    # [CLASS DOC]
    # Database helper for ADBC connections (subclass of the abstract helper). State: the ADBC
    # connection. Supports raw SQL execution, whole-table and query reads through Arrow
    # tables, writes through the ADBC bulk ingest call, table existence checks and row
    # deletion. Many options (chunking, float coercion, query parameters, index labels, dtypes,
    # insert methods, engines) are rejected with not-implemented errors.
    """
    This class enables conversion between DataFrame and SQL databases
    using ADBC to handle DataBase abstraction.

    Parameters
    ----------
    con : adbc_driver_manager.dbapi.Connection
    """

    def __init__(self, con) -> None:
        # [FUNCTION DOC]
        # Stores the ADBC connection.
        self.con = con

    @contextmanager
    def run_transaction(self):
        # [FUNCTION DOC]
        # Context manager yielding a new cursor; rolls back and re-raises on any error in the
        # block, and commits the connection if the block finishes normally. The cursor is
        # closed on exit.
        with self.con.cursor() as cur:
            try:
                yield cur
            except Exception:
                self.con.rollback()
                raise
            self.con.commit()

    def execute(self, sql: str | Select | TextClause, params=None):
        # [FUNCTION DOC]
        # Executes a text statement and returns the open cursor (the caller must close it).
        # Parameters: sql must be text (type error otherwise); params optional, passed to the
        # cursor. On an ADBC error rolls back the connection and raises the library's database
        # error naming the statement (with a different message if the rollback also fails).
        from adbc_driver_manager import Error

        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Error as exc:
            try:
                self.con.rollback()
            except Error as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Reads a whole table through ADBC.
        # Parameters: table_name; index_col; coerce_float (must be true, else not-implemented
        # error); parse_dates; columns to select (index columns are prepended); schema
        # (optional); chunksize (non-zero raises not-implemented); dtype_backend.
        # Behavior: builds a select statement with quoted identifiers, executes it, fetches the
        # result as an Arrow table, converts it to a frame with the dtype backend, then
        # finishes it (date parsing, index). Returns the frame.
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        coerce_float : bool, default True
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype_backend : {'numpy_nullable', 'pyarrow'}
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). If not specified, the default behavior
            is to not use nullable data types. If specified, the behavior
            is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
            * ``"pyarrow"``: returns pyarrow-backed nullable
              :class:`ArrowDtype` :class:`DataFrame`

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")

        if columns:
            if index_col:
                index_select = maybe_make_list(index_col)
            else:
                index_select = []
            to_select = index_select + columns
            select_list = ", ".join(_quote_identifier(x) for x in to_select)
        else:
            select_list = "*"
        quoted_table_name = _quote_identifier(table_name)
        if schema:
            quoted_schema = _quote_identifier(schema)
            stmt = f"SELECT {select_list} FROM {quoted_schema}.{quoted_table_name}"
        else:
            stmt = f"SELECT {select_list} FROM {quoted_table_name}"

        with self.execute(stmt) as cur:
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)

        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
        )

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Runs a query through ADBC and returns a frame.
        # Parameters: sql (text); index_col; coerce_float (must be true); parse_dates; params
        # (any truthy value raises not-implemented); chunksize (non-zero raises not-implemented);
        # dtype to cast to; dtype_backend.
        # Behavior: executes the query, fetches an Arrow table, converts to a frame with the
        # backend, applies dtype, date parsing and index. The class attribute alias read_sql
        # refers to this method.
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Raises NotImplementedError
        params : list, tuple or dict, optional, default: None
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if params:
            raise NotImplementedError("'params' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")

        with self.execute(sql) as cur:
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)

        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
            dtype=dtype,
        )

    read_sql = read_query

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        # [FUNCTION DOC]
        # Writes a frame to a table through the ADBC bulk ingest call.
        # Parameters: frame; name; if_exists (fail, replace, append, delete_rows); index
        # (whether to preserve the index); schema. The index label, chunksize, dtype and method
        # parameters must be empty, and engine must be auto, else not-implemented errors.
        # Behavior: requires pyarrow. If the table exists: fail raises a value error; replace
        # drops it (ingest then creates); append ingests in append mode; delete_rows empties it
        # and ingests in append mode. If it does not exist, ingests in create mode. Converts the
        # frame to an Arrow table (value error if a data type is unsupported), ingests it, wraps
        # ADBC errors in the library's database error, commits and returns the reported number of
        # rows.
        # # KNOWN ISSUE: the docstring lists the if_exists choices without the delete_rows
        # choice in its heading, although the description below it and the code both support it.
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
            - delete_rows: If a table exists, delete all records and insert data.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Raises NotImplementedError
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : single type or dict of column name to SQL type, default None
            Raises NotImplementedError
        method : {None', 'multi', callable}, default None
            Raises NotImplementedError
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            Raises NotImplementedError if not set to 'auto'
        """
        pa = import_optional_dependency("pyarrow")
        from adbc_driver_manager import Error

        if index_label:
            raise NotImplementedError(
                "'index_label' is not implemented for ADBC drivers"
            )
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if dtype:
            raise NotImplementedError("'dtype' is not implemented for ADBC drivers")
        if method:
            raise NotImplementedError("'method' is not implemented for ADBC drivers")
        if engine != "auto":
            raise NotImplementedError(
                "engine != 'auto' not implemented for ADBC drivers"
            )

        quoted_name = _quote_identifier(name)
        if schema:
            quoted_schema = _quote_identifier(schema)
            quoted_table_name = f"{quoted_schema}.{quoted_name}"
        else:
            quoted_table_name = quoted_name

        # pandas if_exists="append" will still create the
        # table if it does not exist; ADBC is more explicit with append/create
        # as applicable modes, so the semantics get blurred across
        # the libraries
        mode = "create"
        if self.has_table(name, schema):
            if if_exists == "fail":
                raise ValueError(f"Table '{name}' already exists.")
            elif if_exists == "replace":
                sql_statement = f"DROP TABLE {quoted_table_name}"
                self.execute(sql_statement).close()
            elif if_exists == "append":
                mode = "append"
            elif if_exists == "delete_rows":
                mode = "append"
                self.delete_rows(name, schema)

        try:
            tbl = pa.Table.from_pandas(frame, preserve_index=index)
        except pa.ArrowNotImplementedError as exc:
            raise ValueError("datatypes not supported") from exc

        with self.con.cursor() as cur:
            try:
                total_inserted = cur.adbc_ingest(
                    table_name=name, data=tbl, mode=mode, db_schema_name=schema
                )
            except Error as exc:
                raise DatabaseError(
                    f"Failed to insert records on table={name} with {mode=}"
                ) from exc

        self.con.commit()
        return total_inserted

    def has_table(self, name: str, schema: str | None = None) -> bool:
        # [FUNCTION DOC]
        # Returns whether the table exists, by asking the ADBC driver for the objects matching the
        # schema and table name filters and scanning the nested catalog, schema and table records
        # for the exact name.
        meta = self.con.adbc_get_objects(
            db_schema_filter=schema, table_name_filter=name
        ).read_all()

        for catalog_schema in meta["catalog_db_schemas"].to_pylist():
            if not catalog_schema:
                continue
            for schema_record in catalog_schema:
                if not schema_record:
                    continue

                for table_record in schema_record["db_schema_tables"]:
                    if table_record["table_name"] == name:
                        return True

        return False

    def delete_rows(self, name: str, schema: str | None = None) -> None:
        # [FUNCTION DOC]
        # Deletes all rows of the table if it exists, with quoted names and optional schema, and
        # closes the cursor.
        quoted_table_name = _quote_identifier(name)
        if schema:
            quoted_schema = _quote_identifier(schema)
            quoted_table_name = f"{quoted_schema}.{quoted_table_name}"

        if self.has_table(name, schema):
            self.execute(f"DELETE FROM {quoted_table_name}").close()

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Not supported for ADBC: always raises a not-implemented error.
        raise NotImplementedError("not implemented for adbc")


# sqlite-specific sql strings and handler class
# dictionary used for readability purposes
_SQL_TYPES = {
    "string": "TEXT",
    "floating": "REAL",
    "integer": "INTEGER",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "INTEGER",
}


def _get_unicode_name(name: object) -> str:
    # [FUNCTION DOC]
    # HELPER: converts any object to text and verifies it round-trips as UTF-8; raises a value
    # error if it cannot be encoded.
    try:
        uname = str(name).encode("utf-8", "strict").decode("utf-8")
    except UnicodeError as err:
        raise ValueError(f"Cannot convert identifier to UTF-8: '{name}'") from err
    return uname


def _get_valid_sqlite_name(name: object) -> str:
    # [FUNCTION DOC]
    # HELPER: makes a name safe for use as a quoted identifier in sqlite statements: converts
    # to valid UTF-8 text, rejects empty names and names containing a NUL character (value
    # errors), doubles embedded double quotes and wraps the result in double quotes.
    # See https://stackoverflow.com/questions/6514274/how-do-you-escape-strings\
    # -for-sqlite-table-column-names-in-python
    # Ensure the string can be encoded as UTF-8.
    # Ensure the string does not include any NUL characters.
    # Replace all " with "".
    # Wrap the entire thing in double quotes.

    uname = _get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        raise ValueError("SQLite identifier cannot contain NULs")
    return '"' + uname.replace('"', '""') + '"'


def _quote_identifier(name: object) -> str:
    # [FUNCTION DOC]
    # HELPER: same quoting for statements built for ADBC drivers (see the docstring for the
    # standard escaping rules): converts to UTF-8 text, rejects empty names and names with a NUL
    # character with value errors, doubles embedded double quotes and wraps in double quotes.
    # Behavior is identical to the sqlite quoting function apart from one error message.
    """
    Escape a SQL identifier (table name or schema name) for safe use in
    ADBC-generated SQL statements.

    Uses ANSI SQL double-quote escaping, which is supported by PostgreSQL,
    DuckDB, SQLite, and other databases targeted by ADBC drivers.

    Parameters
    ----------
    name : object
        Identifier to escape.

    Returns
    -------
    str
        The identifier wrapped in double quotes with any internal double
        quotes doubled, e.g. ``my"table`` becomes ``"my""table"``.

    Raises
    ------
    ValueError
        If *name* is empty or contains a NUL character.
    """
    uname = _get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        raise ValueError("SQL identifier cannot contain NUL characters")

    return '"' + uname.replace('"', '""') + '"'


class SQLiteTable(SQLTable):
    # [CLASS DOC]
    # Table mapping for the sqlite fallback (subclass of the SQLAlchemy based table mapping).
    # Instead of a SQLAlchemy table object, its table attribute is the list of SQL statements
    # (the create table statement followed by create index statements). Overrides schema text,
    # table creation, the insert statement builders and the type mapping to produce raw sqlite
    # SQL. Creating an instance registers global sqlite adapters.
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Passes all arguments to the parent initializer and then registers the sqlite date
        # adapters and converters.
        super().__init__(*args, **kwargs)

        self._register_date_adapters()

    def _register_date_adapters(self) -> None:
        # [FUNCTION DOC]
        # HELPER: registers, with the sqlite library, adapters that convert Python time, date and
        # datetime values to text (time as hour, minute, second and microseconds; date in ISO form;
        # datetime in ISO form with a space separator) and converters that parse the declared types
        # date and timestamp from ISO text back to date and datetime objects. Needed because newer
        # Python versions no longer register these automatically.
        # Side effects: modifies global sqlite adapter and converter registries each time a table
        # object is created; takes no parameters and returns nothing. The adapters and converters
        # are small lambdas assigned to local names.
        # GH 8341
        # register an adapter callable for datetime.time object
        import sqlite3

        # this will transform time(12,34,56,789) into '12:34:56.000789'
        # (this is what sqlalchemy does)
        def _adapt_time(t) -> str:
            # [FUNCTION DOC]
            # HELPER: nested adapter converting a time value to text with two-digit hour, minute and
            # second and six-digit microseconds, separated by colons and a dot.
            # This is faster than strftime
            return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"

        # Also register adapters for date/datetime and co
        # xref https://docs.python.org/3.12/library/sqlite3.html#adapter-and-converter-recipes
        # Python 3.12+ doesn't auto-register adapters for us anymore

        adapt_date_iso = lambda val: val.isoformat()
        adapt_datetime_iso = lambda val: val.isoformat(" ")

        sqlite3.register_adapter(time, _adapt_time)

        sqlite3.register_adapter(date, adapt_date_iso)
        sqlite3.register_adapter(datetime, adapt_datetime_iso)

        convert_date = lambda val: date.fromisoformat(val.decode())
        convert_timestamp = lambda val: datetime.fromisoformat(val.decode())

        sqlite3.register_converter("date", convert_date)
        sqlite3.register_converter("timestamp", convert_timestamp)

    def sql_schema(self) -> str:
        # [FUNCTION DOC]
        # Returns the creation statements joined by a semicolon and newline.
        return str(";\n".join(self.table))

    def _execute_create(self) -> None:
        # [FUNCTION DOC]
        # HELPER: executes each statement in the table's list through the helper's transaction
        # cursor, creating the table and its indexes.
        with self.pd_sql.run_transaction() as cur:
            for stmt in self.table:
                cur.execute(stmt)

    def insert_statement(self, *, num_rows: int) -> str:
        # [FUNCTION DOC]
        # HELPER: builds the insert statement text. Keyword parameter num_rows is the number of rows
        # to insert. Column names are the frame columns (as text) preceded by the index column names,
        # each quoted safely. Produces an insert into the quoted table name with one parenthesized
        # group of question mark placeholders per row.
        names = list(map(str, self.frame.columns))
        wld = "?"  # wildcard char
        escape = _get_valid_sqlite_name

        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)

        bracketed_names = [escape(column) for column in names]
        col_names = ",".join(bracketed_names)

        row_wildcards = ",".join([wld] * len(names))
        wildcards = ",".join([f"({row_wildcards})" for _ in range(num_rows)])
        insert_statement = (
            f"INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}"
        )
        return insert_statement

    def _execute_insert(self, conn, keys, data_iter) -> int:
        # [FUNCTION DOC]
        # HELPER: default insertion method. Collects the rows, runs the single-row statement for
        # all rows using the connection's bulk execute, wraps sqlite errors in the library's
        # database error, and returns the connection's row count. The keys argument is unused.
        from sqlite3 import Error

        data_list = list(data_iter)
        try:
            conn.executemany(self.insert_statement(num_rows=1), data_list)
        except Error as exc:
            raise DatabaseError("Execution failed") from exc
        return conn.rowcount

    def _execute_insert_multi(self, conn, keys, data_iter) -> int:
        # [FUNCTION DOC]
        # HELPER: multi-value insertion method. Collects the rows, flattens all values into one
        # sequence, and runs one statement with a group of placeholders per row. Returns the
        # connection's row count. The keys argument is unused; sqlite errors are not wrapped.
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)
        return conn.rowcount

    def _create_table_setup(self):
        # [FUNCTION DOC]
        # HELPER: builds the list of statements that create the table.
        # First statement: create table with the (optional unquoted) schema prefix, then each
        # column as a quoted name with its sqlite type, and, if keys are given, a primary key
        # constraint named from the table name. If any index columns exist, a second statement
        # creates one index over them, named from the table and column names.
        # Returns the list of statement texts.
        """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = _get_valid_sqlite_name

        create_tbl_stmts = [
            escape(cname) + " " + ctype for cname, ctype, _ in column_names_and_types
        ]

        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ", ".join([escape(c) for c in keys])
            create_tbl_stmts.append(
                f"CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})"
            )
        if self.schema:
            schema_name = self.schema + "."
        else:
            schema_name = ""
        create_stmts = [
            "CREATE TABLE "
            + schema_name
            + escape(self.name)
            + " (\n"
            + ",\n  ".join(create_tbl_stmts)
            + "\n)"
        ]

        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if ix_cols:
            cnames = "_".join(ix_cols)
            cnames_br = ",".join([escape(c) for c in ix_cols])
            create_stmts.append(
                "CREATE INDEX "
                + escape("ix_" + self.name + "_" + cnames)
                + "ON "
                + escape(self.name)
                + " ("
                + cnames_br
                + ")"
            )

        return create_stmts

    def _sql_type_name(self, col):
        # [FUNCTION DOC]
        # HELPER: maps a column to a sqlite type name. A mapping dtype argument with an entry for the
        # column name wins. Otherwise infers the data kind ignoring missing values: timedeltas warn
        # and become integer, datetimes become the datetime kind, empty columns become string,
        # complex raises a value error; kinds not in the module type table fall back to string.
        # Returns the type name text (text, real, integer, timestamp, date, time).
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            col_type = "integer"

        elif col_type == "datetime64":
            col_type = "datetime"

        elif col_type == "empty":
            col_type = "string"

        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        if col_type not in _SQL_TYPES:
            col_type = "string"

        return _SQL_TYPES[col_type]


class SQLiteDatabase(PandasSQL):
    # [CLASS DOC]
    # Database helper for the sqlite fallback (subclass of the abstract helper), used when
    # SQLAlchemy is not installed or a sqlite3 connection is given. State: the connection
    # object. Supports queries, writing through the sqlite table mapping, existence checks, drop
    # and delete rows; table reads are not supported (inherited stub raises not-implemented).
    # Intended for internal use.
    """
    Version of SQLDatabase to support SQLite connections (fallback without
    SQLAlchemy). This should only be used internally.

    Parameters
    ----------
    con : sqlite connection object

    """

    def __init__(self, con) -> None:
        # [FUNCTION DOC]
        # Stores the connection.
        self.con = con

    @contextmanager
    def run_transaction(self):
        # [FUNCTION DOC]
        # Context manager yielding a new cursor; commits when the block ends normally, rolls back
        # and re-raises on error, and always closes the cursor.
        cur = self.con.cursor()
        try:
            yield cur
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str | Select | TextClause, params=None):
        # [FUNCTION DOC]
        # Executes a text statement and returns the open cursor (the caller must close it).
        # Parameters: sql must be text (type error otherwise); params optional. On a sqlite error rolls
        # back and raises the library's database error naming the statement (different message if
        # rollback fails too).
        from sqlite3 import Error

        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Error as exc:
            try:
                self.con.rollback()
            except Error as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    @staticmethod
    def _query_iterator(
        cursor,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> Generator[DataFrame]:
        # [FUNCTION DOC]
        # HELPER: static generator yielding chunks of a cursor result.
        # Parameters: cursor; chunksize; columns; index_col, coerce_float, parse_dates, dtype,
        # dtype_backend as for the module readers.
        # Behavior: repeatedly fetches chunks (a tuple result is converted to a list). When no more
        # rows, closes the cursor and, if no rows were ever read, yields one empty frame with the
        # columns (cast to the dtype if given). Each non-empty chunk is turned into a finished frame.
        """Return generator through chunked result set"""
        has_read_data = False
        while True:
            data = cursor.fetchmany(chunksize)
            if type(data) == tuple:
                data = list(data)
            if not data:
                cursor.close()
                if not has_read_data:
                    result = DataFrame.from_records(
                        [], columns=columns, coerce_float=coerce_float
                    )
                    if dtype:
                        result = result.astype(dtype)
                    yield result
                break

            has_read_data = True
            yield _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

    def read_query(
        self,
        sql,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        # [FUNCTION DOC]
        # Runs a query on the sqlite connection.
        # Parameters: sql; index_col; coerce_float; parse_dates; params; chunksize; dtype; dtype_backend.
        # Behavior: executes the statement, takes the column names from the cursor description. With a
        # chunk size returns the chunk generator (which closes the cursor at the end). Otherwise
        # fetches all rows as a list, closes the cursor and returns one finished frame.
        cursor = self.execute(sql, params)
        columns = [col_desc[0] for col_desc in cursor.description]

        if chunksize is not None:
            return self._query_iterator(
                cursor,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = self._fetchall_as_list(cursor)
            cursor.close()

            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    def _fetchall_as_list(self, cur):
        # [FUNCTION DOC]
        # HELPER: fetches all rows from a cursor and ensures they are returned as a list.
        result = cur.fetchall()
        if not isinstance(result, list):
            result = list(result)
        return result

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: str = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        # [FUNCTION DOC]
        # Writes a frame to a sqlite table.
        # Parameters: frame; name; if_exists; index; index_label; chunksize; dtype (a type name text or a
        # mapping to text; a single value is expanded to all frame columns; a non-text type raises a
        # value error); method. The schema, engine and extra engine arguments are accepted but ignored.
        # Behavior: builds the sqlite table mapping, creates the table according to the if-exists
        # policy, and inserts the rows. Returns the number of rows inserted or none.
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame: DataFrame
        name: string
            Name of SQL table.
        if_exists: {'fail', 'replace', 'append', 'delete_rows'}, default 'fail'
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if it does not exist.
            delete_rows: If a table exists, delete all records and insert data.
        index : bool, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Ignored parameter included for compatibility with SQLAlchemy
            version of ``to_sql``.
        chunksize : int, default None
            If not None, then rows will be written in batches of this
            size at a time. If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a string. If all columns are of the same type, one single value
            can be used.
        method : {None, 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = dict.fromkeys(frame, dtype)  # type: ignore[arg-type]
            else:
                dtype = cast(dict, dtype)

            for col, my_type in dtype.items():
                if not isinstance(my_type, str):
                    raise ValueError(f"{col} ({my_type}) not a string")

        table = SQLiteTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            dtype=dtype,
        )
        table.create()
        return table.insert(chunksize, method)

    def has_table(self, name: str, schema: str | None = None) -> bool:
        # [FUNCTION DOC]
        # Returns whether a table or view with the name exists, by querying sqlite's master table
        # with a bound parameter. The schema argument is ignored.
        wld = "?"
        query = f"""
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type IN ('table', 'view')
            AND name={wld};
        """

        return len(self.execute(query, [name]).fetchall()) > 0

    def get_table(self, table_name: str, schema: str | None = None) -> None:
        # [FUNCTION DOC]
        # Returns none; table reflection is not supported in the fallback.
        return None  # not supported in fallback mode

    def drop_table(self, name: str, schema: str | None = None) -> None:
        # [FUNCTION DOC]
        # Drops the named table (quoted name) and closes the cursor. The schema argument is ignored.
        drop_sql = f"DROP TABLE {_get_valid_sqlite_name(name)}"
        self.execute(drop_sql).close()

    def delete_rows(self, name: str, schema: str | None = None) -> None:
        # [FUNCTION DOC]
        # Deletes all rows of the named table if it exists (quoted name) and closes the cursor.
        delete_sql = f"DELETE FROM {_get_valid_sqlite_name(name)}"
        if self.has_table(name, schema):
            self.execute(delete_sql).close()

    def _create_sql_schema(
        self,
        frame,
        table_name: str,
        keys=None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Returns the table creation statement text for a frame using the sqlite table mapping without
        # index columns. Parameters: frame; table_name; keys; dtype; schema.
        table = SQLiteTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


def get_schema(
    frame,
    name: str,
    keys=None,
    con=None,
    dtype: DtypeArg | None = None,
    schema: str | None = None,
) -> str:
    # [FUNCTION DOC]
    # Returns the SQL statements text that would create a table suited to the frame.
    # Parameters: frame; name (table name); keys (primary key column(s)); con (connection, default
    # none which selects the sqlite fallback); dtype (mapping of column to SQL type; type objects
    # for SQLAlchemy, text for sqlite); schema.
    # Behavior: builds the helper for the connection and asks it for the schema text. ADBC
    # connections raise a not-implemented error. Returns the text.
    """
    Get the SQL db table schema for the given frame.

    Parameters
    ----------
    frame : DataFrame
    name : str
        name of SQL table
    keys : string or sequence, default: None
        columns to use a primary key
    con: ADBC Connection, SQLAlchemy connectable, sqlite3 connection, default: None
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library
        If a DBAPI2 object, only sqlite3 is supported.
    dtype : dict of column name to SQL type, default None
        Optional specifying the datatype for columns. The SQL type should
        be a SQLAlchemy type, or a string for sqlite3 fallback connection.
    schema: str, default: None
        Optional specifying the schema to be used in creating the table.
    """
    with pandasSQL_builder(con=con) as pandas_sql:
        return pandas_sql._create_sql_schema(
            frame, name, keys=keys, dtype=dtype, schema=schema
        )
