# [MODULE DOC]
# Implementation of JSON reading and writing for pandas (the frame/series JSON
# writer and the top-level JSON reader).
# Public contents: to_json (writer entry point used by the frame and series
# methods), read_json (public reader, exposed at top level), JsonReader (iterator
# and context manager returned for chunked reads, exposed in the public typing
# namespace), plus re-exports of the C-level ujson encoder/decoder functions.
# Internal contents: writer classes (an abstract Writer with series, frame and
# Table Schema variants), parser classes (an abstract Parser with series and
# frame variants that decode and then coerce types) and a date-column predicate.
# Relationships: the C extension does the actual encoding/decoding; line-delimited
# conversion and Table Schema logic come from sibling modules; file access comes
# from the io common module. Optional dependency pyarrow is imported lazily.
# Import-time side effects: none other than imports and defining a type variable.
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections import abc
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import option_context

from pandas._libs import lib
from pandas._libs.json import (
    ujson_dumps,
    ujson_loads,
)
from pandas._libs.tslibs import iNaT
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    OutOfBoundsDatetime,
)
from pandas.util._decorators import set_module
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    ensure_str,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
    to_datetime,
)
from pandas.core.reshape.concat import concat

from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import (
    IOHandles,
    dedup_names,
    get_handle,
    is_potential_multi_index,
    stringify_path,
)
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import (
    build_table_schema,
    parse_table_schema,
    set_default_names,
)
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Mapping,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        JSONEngine,
        JSONSerializable,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas.core.generic import NDFrame

FrameSeriesStrT = TypeVar("FrameSeriesStrT", bound=Literal["frame", "series"])


# interface to/from
@overload
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> None: ...
    # [FUNCTION DOC]
    # Typing overload only: signature for writing to a path or buffer, returning
    # nothing. See the implementation below for behavior.


@overload
def to_json(
    path_or_buf: None,
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> str: ...
    # [FUNCTION DOC]
    # Typing overload only: signature for path_or_buf being none, returning the JSON
    # text. See the implementation below for behavior.


def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] | None,
    obj: NDFrame,
    orient: str | None = None,
    date_format: str = "epoch",
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: str = "ms",
    default_handler: Callable[[Any], JSONSerializable] | None = None,
    lines: bool = False,
    compression: CompressionOptions = "infer",
    index: bool | None = None,
    indent: int = 0,
    storage_options: StorageOptions | None = None,
    mode: Literal["a", "w"] = "w",
) -> str | None:
    # [FUNCTION DOC]
    # Serializes a frame or series to JSON, writing to a target or returning text.
    # Parameters: path_or_buf is a path, writable text/binary buffer, or none to
    # return a string; obj is the series or frame; orient is the layout (none means
    # the writer's default: index for series, columns for frame, records for table);
    # date_format is epoch or iso (default epoch); double_precision is the number of
    # decimals for floats (default 10); force_ascii escapes non-ASCII output (default
    # true); date_unit is s, ms, us or ns (default ms); default_handler is a fallback
    # for unserializable objects (default none); lines writes one record per line
    # (default false); compression and storage_options are used when opening a
    # target (defaults infer and none); index says whether to include the index
    # (none means true, ignored for records and values orients); indent is the
    # pretty-print indent width (default 0); mode is w or a (default w).
    # Raises a value error for: index true with records or values orient, index
    # false with index or columns orient, lines without records orient, a mode other
    # than w or a, append mode unless lines is set with records orient, and an invalid
    # date_unit when epoch dates are used. Raises a not-implemented error if obj is
    # not a series or frame.
    # Behavior: a series with the table orient becomes a one-column frame (named by
    # the series name or "values"). With epoch dates, datetime-like data, index and
    # columns are first converted (on shallow copies, never mutating the caller's
    # object) to the requested unit. The writer class is chosen (table, series, or
    # frame), the text is produced, optionally converted to line-delimited form, and
    # then written to the target through a handle with the compression options (returns
    # none) or returned as a string when no target is given.
    # KNOWN ISSUE: the error message for an invalid mode runs two sentences together without a space between them.
    if orient in ["records", "values"] and index is True:
        raise ValueError(
            "'index=True' is only valid when 'orient' is 'split', 'table', "
            "'index', or 'columns'."
        )
    elif orient in ["index", "columns"] and index is False:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split', 'table', "
            "'records', or 'values'."
        )
    elif index is None:
        # will be ignored for orient='records' and 'values'
        index = True

    if lines and orient != "records":
        raise ValueError("'lines' keyword only valid when 'orient' is records")

    if mode not in ["a", "w"]:
        msg = (
            f"mode={mode} is not a valid option."
            "Only 'w' and 'a' are currently supported."
        )
        raise ValueError(msg)

    if mode == "a" and (not lines or orient != "records"):
        msg = (
            "mode='a' (append) is only supported when "
            "lines is True and orient is 'records'"
        )
        raise ValueError(msg)

    if orient == "table" and isinstance(obj, Series):
        obj = obj.to_frame(name=obj.name or "values")

    if date_format == "epoch":
        # for epoch (numeric) format, convert datetime-likes to the desired
        # unit up front, such that the C ObjToJSON code can simply write out
        # the integer values without worrying about conversion
        if date_unit not in ["s", "ms", "us", "ns"]:
            raise ValueError(f"Invalid value '{date_unit}' for option 'date_unit'")
        if isinstance(obj, DataFrame):
            copied = False
            cols = np.nonzero(obj.dtypes.map(lambda dt: dt.kind in ["M", "m"]))[0]
            if len(cols):
                obj = obj.copy(deep=False)
                copied = True
                for col in cols:
                    obj.isetitem(col, obj.iloc[:, col].dt.as_unit(date_unit))
            if obj.index.dtype.kind in "Mm":
                if not copied:
                    obj = obj.copy(deep=False)
                    copied = True
                obj.index = Series(obj.index).dt.as_unit(date_unit)
            if obj.columns.dtype.kind in "Mm":
                if not copied:
                    obj = obj.copy(deep=False)
                    copied = True
                obj.columns = Series(obj.columns).dt.as_unit(date_unit)
        elif isinstance(obj, Series):
            if obj.dtype.kind in "Mm":
                obj = obj.copy(deep=False)
                obj = obj.dt.as_unit(date_unit)
            if obj.index.dtype.kind in "Mm":
                obj = obj.copy(deep=False)
                obj.index = Series(obj.index).dt.as_unit(date_unit)

    writer: type[Writer]
    if orient == "table" and isinstance(obj, DataFrame):
        writer = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")

    s = writer(
        obj,
        orient=orient,
        date_format=date_format,
        double_precision=double_precision,
        ensure_ascii=force_ascii,
        date_unit=date_unit,
        default_handler=default_handler,
        index=index,
        indent=indent,
    ).write()

    if lines:
        s = convert_to_line_delimits(s)

    if path_or_buf is not None:
        # apply compression and byte/text conversion
        with get_handle(
            path_or_buf, mode, compression=compression, storage_options=storage_options
        ) as handles:
            handles.handle.write(s)
    else:
        return s
    return None


class Writer(ABC):
    # [CLASS DOC]
    # Abstract base for JSON writers. Subclasses set the default orient name as a
    # class attribute and implement the axes validation and the property giving the
    # object to serialize. State: obj, orient (default filled in when none), and the
    # formatting options date_format, double_precision, ensure_ascii, date_unit,
    # default_handler, index, indent. Key method: write.
    _default_orient: str

    def __init__(
        self,
        obj: NDFrame,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Stores all arguments as attributes (a none orient is replaced by the class
        # default orient), then calls the axes validation hook, which may raise.
        # Parameters: obj is the object to write; orient the layout; date_format epoch or
        # iso; double_precision float decimals; ensure_ascii escape non-ASCII;
        # date_unit time unit; index whether to include the index; default_handler
        # fallback for unsupported objects (default none); indent (default 0).
        self.obj = obj

        if orient is None:
            orient = self._default_orient

        self.orient = orient
        self.date_format = date_format
        self.double_precision = double_precision
        self.ensure_ascii = ensure_ascii
        self.date_unit = date_unit
        self.default_handler = default_handler
        self.index = index
        self.indent = indent
        self._format_axes()

    def _format_axes(self) -> None:
        # [FUNCTION DOC]
        # HELPER: abstract validation hook run from the constructor; subclasses raise a
        # value error for configurations that cannot be written. Base raises an
        # abstract-method error.
        raise AbstractMethodError(self)

    def write(self) -> str:
        # [FUNCTION DOC]
        # Encodes the object given by the obj_to_write property with the C encoder,
        # passing orient, float precision, ASCII escaping, date unit, an ISO-date flag
        # (true when date_format is iso), the default handler and the indent. Returns the
        # JSON text.
        iso_dates = self.date_format == "iso"
        return ujson_dumps(
            self.obj_to_write,
            orient=self.orient,
            double_precision=self.double_precision,
            ensure_ascii=self.ensure_ascii,
            date_unit=self.date_unit,
            iso_dates=iso_dates,
            default_handler=self.default_handler,
            indent=self.indent,
        )

    @property
    @abstractmethod
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # [FUNCTION DOC]
        # Abstract property; subclasses return the object (or mapping) that is handed to
        # the encoder.
        """Object to write in JSON format."""


class SeriesWriter(Writer):
    # [CLASS DOC]
    # Writer for series. Default orient is index. Writes the series itself, except
    # for the split orient without index, which writes a mapping of name and values.
    _default_orient = "index"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # [FUNCTION DOC]
        # Returns a mapping with the series name and its values when the index is
        # excluded and the orient is split; otherwise returns the series unchanged.
        if not self.index and self.orient == "split":
            return {"name": self.obj.name, "data": self.obj.values}
        else:
            return self.obj

    def _format_axes(self) -> None:
        # [FUNCTION DOC]
        # HELPER: validation hook. Raises a value error if the orient is index and the
        # series index has duplicate labels. No return value.
        if not self.obj.index.is_unique and self.orient == "index":
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")


class FrameWriter(Writer):
    # [CLASS DOC]
    # Writer for frames. Default orient is columns. Writes the frame itself except
    # for split orient without index, which writes the split-form mapping without its
    # index entry.
    _default_orient = "columns"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # [FUNCTION DOC]
        # Returns, when the index is excluded and the orient is split, the split-form
        # dictionary of the frame with the index entry removed; otherwise returns the
        # frame.
        if not self.index and self.orient == "split":
            obj_to_write = self.obj.to_dict(orient="split")
            del obj_to_write["index"]
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self) -> None:
        # [FUNCTION DOC]
        # HELPER: validation hook. Raises a value error when the index has duplicates and
        # the orient is index or columns, or when the columns have duplicates and the
        # orient is index, columns or records.
        # KNOWN ISSUE: the docstring says it tries to format axes that are datelike, but the code only checks uniqueness.
        """
        Try to format axes if they are datelike.
        """
        if not self.obj.index.is_unique and self.orient in ("index", "columns"):
            raise ValueError(
                f"DataFrame index must be unique for orient='{self.orient}'."
            )
        if not self.obj.columns.is_unique and self.orient in (
            "index",
            "columns",
            "records",
        ):
            raise ValueError(
                f"DataFrame columns must be unique for orient='{self.orient}'."
            )


class JSONTableWriter(FrameWriter):
    # [CLASS DOC]
    # Writer for the Table Schema layout. Extends the frame writer; default orient
    # is records. State beyond the base: schema (Table Schema description built from
    # the original object) and obj (the frame actually written, with index moved to
    # columns, or dropped when index is false, and timedelta columns turned into
    # ISO strings). The effective orient is forced to records and date_format to iso.
    # The text written is a mapping with the schema and the data.
    _default_orient = "records"

    def __init__(
        self,
        obj,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Builds a table-schema writer. Parameters match the base writer. Calls the base
        # constructor, then requires date_format to be iso (raises a value error
        # otherwise), builds the schema from the frame with the given index setting, gives
        # default names to unnamed index levels when the index is included, and rejects
        # column MultiIndex frames (not-implemented error) and overlapping names between
        # index and columns (value error). Timedelta columns are converted to ISO duration
        # strings on a copy. When the index is excluded the stored object is the frame
        # with its index dropped; otherwise a period index is converted to timestamps
        # (on a shallow copy if not already copied) and the index is reset into columns.
        # Finally forces date_format to iso, orient to records and stores index.
        # KNOWN ISSUE: the constructor docstring is an incomplete sentence (unbalanced parenthesis) and the base constructor has already stored the orient and date format before they are validated or overridden.
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        super().__init__(
            obj,
            orient,
            date_format,
            double_precision,
            ensure_ascii,
            date_unit,
            index,
            default_handler=default_handler,
            indent=indent,
        )

        if date_format != "iso":
            msg = (
                "Trying to write with `orient='table'` and "
                f"`date_format='{date_format}'`. Table Schema requires dates "
                "to be formatted with `date_format='iso'`"
            )
            raise ValueError(msg)

        self.schema = build_table_schema(obj, index=self.index)
        if self.index:
            obj = set_default_names(obj)

        # NotImplemented on a column MultiIndex
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError(
                "orient='table' is not supported for MultiIndex columns"
            )

        # TODO: Do this timedelta properly in objToJSON.c See GH #15137
        if ((obj.ndim == 1) and (obj.name in set(obj.index.names))) or len(
            obj.columns.intersection(obj.index.names)
        ):
            msg = "Overlapping names between the index and columns"
            raise ValueError(msg)

        timedeltas = obj.select_dtypes(include=["timedelta"]).columns
        copied = False
        if len(timedeltas):
            obj = obj.copy()
            copied = True
            obj[timedeltas] = obj[timedeltas].map(lambda x: x.isoformat())

        # exclude index from obj if index=False
        if not self.index:
            self.obj = obj.reset_index(drop=True)
        else:
            # Convert PeriodIndex to datetimes before serializing
            if isinstance(obj.index.dtype, PeriodDtype):
                if not copied:
                    obj = obj.copy(deep=False)
                obj.index = obj.index.to_timestamp()
            self.obj = obj.reset_index(drop=False)
        self.date_format = "iso"
        self.orient = "records"
        self.index = index

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        # [FUNCTION DOC]
        # Returns a mapping with the entry for the schema and an entry for the data frame.
        return {"schema": self.schema, "data": self.obj}


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> JsonReader[Literal["frame"]]: ...
    # [FUNCTION DOC]
    # Typing overload only: with chunksize given and frame type, returns a chunked
    # reader. See the implementation.


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> JsonReader[Literal["series"]]: ...
    # [FUNCTION DOC]
    # Typing overload only: with chunksize given and series type, returns a chunked
    # reader. See the implementation.


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> Series: ...
    # [FUNCTION DOC]
    # Typing overload only: series type without chunksize returns a series. See the
    # implementation.


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> DataFrame: ...
    # [FUNCTION DOC]
    # Typing overload only: frame type without chunksize returns a frame. See the
    # implementation.


@set_module("pandas")
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = None,
    typ: Literal["frame", "series"] = "frame",
    dtype: DtypeArg | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    lines: bool = False,
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine: JSONEngine = "ujson",
) -> DataFrame | Series | JsonReader:
    # [FUNCTION DOC]
    # Public JSON reader. All parameters except path_or_buf are keyword-only:
    # path_or_buf is a path, URL or readable buffer; orient is the input layout
    # (none uses the type's default); typ is frame (default) or series; dtype is true
    # to infer, false for none, or a per-column mapping (none becomes true unless the
    # orient is table); convert_axes tries to coerce index and columns (none becomes
    # true unless table); convert_dates is true, false or a list of columns (default
    # true); keep_default_dates enables conversion of conventionally named date
    # columns (default true); precise_float uses the slower precise float decoder
    # (default false); date_unit forces a time unit (default none, auto-detect);
    # encoding and encoding_errors (default strict) control text decoding; lines reads
    # one record per line (default false); chunksize returns an iterator of that many
    # lines at a time (default none); compression, nrows (line limit), storage_options,
    # dtype_backend and engine (ujson default, or pyarrow) as described in the public
    # docstring.
    # Raises a value error when the table orient is combined with a dtype or a
    # truthy convert_axes; validates the dtype_backend argument.
    # Builds a JsonReader; returns the reader itself when chunksize is truthy,
    # otherwise reads everything and returns a frame or series.
    # KNOWN ISSUE: the docstring says the encoding default is utf-8, but the code default is none (decided by the file-opening layer); it also lists the default datelike labels without the label datetime, which the code treats as datelike.
    """
    Convert a JSON string to pandas object.

    This method reads JSON files or JSON-like data and converts them into pandas
    objects. It supports a variety of input formats, including line-delimited JSON,
    compressed files, and various data representations (table, records, index-based,
    etc.). When `chunksize` is specified, an iterator is returned instead of loading
    the entire data into memory.

    Parameters
    ----------
    path_or_buf : a str path, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.json``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.

    orient : str, optional
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{index -> [index], columns -> [columns], data -> [values]}``
        - ``'records'`` : list like
          ``[{column -> value}, ... , {column -> value}]``
        - ``'index'`` : dict like ``{index -> {column -> value}}``
        - ``'columns'`` : dict like ``{column -> {index -> value}}``
        - ``'values'`` : just the values array
        - ``'table'`` : dict like ``{'schema': {schema}, 'data': {data}}``

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'``,

          - allowed orients are ``{'split','records','index'}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{'split','records','index',
            'columns','values', 'table'}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    typ : {'frame', 'series'}, default 'frame'
        The type of object to recover.

    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.

        For all ``orient`` values except ``'table'``, default is True.

    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.

        For all ``orient`` values except ``'table'``, default is True.

    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on
        keep_default_dates).
        If False, no dates will be converted.
        If a list of column names, then those columns will be converted and
        default datelike columns may also be converted (depending on
        keep_default_dates).

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the
        default datelike columns.
        A column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``.

    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.

    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    encoding_errors : str, optional, default "strict"
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

    lines : bool, default False
        Read the file as a json object per line.

    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `line-delimited json docs
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.
        If this is None, the file will be read into memory all at once.

    compression : str or dict, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and 'path_or_buf' is
        path-like, then detect compression from the following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        If using 'zip' or 'tar', the ZIP file must contain only one data file to be
        read in.
        Set to ``None`` for no decompression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
        and other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for Zstandard decompression using a
        custom compression dictionary:
        ``compression={'method': 'zstd', 'dict_data': my_compression_dict}``.

    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read.
        This can only be passed if `lines=True`.
        If this is None, all the rows will be returned.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
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

    engine : {"ujson", "pyarrow"}, default "ujson"
        Parser engine to use. The ``"pyarrow"`` engine is only available when
        ``lines=True``.

        .. versionadded:: 2.0

    Returns
    -------
    Series, DataFrame, or pandas.api.typing.JsonReader
        A JsonReader is returned when ``chunksize`` is not ``0`` or ``None``.
        Otherwise, the type returned depends on the value of ``typ``.

    See Also
    --------
    DataFrame.to_json : Convert a DataFrame to a JSON string.
    Series.to_json : Convert a Series to a JSON string.
    json_normalize : Normalize semi-structured JSON data into a flat table.

    Notes
    -----
    Specific to ``orient='table'``, if a :class:`DataFrame` with a literal
    :class:`Index` name of `index` gets written with :func:`to_json`, the
    subsequent read operation will incorrectly set the :class:`Index` name to
    ``None``. This is because `index` is also used by :func:`DataFrame.to_json`
    to denote a missing :class:`Index` name, and the subsequent
    :func:`read_json` operation cannot distinguish between the two. The same
    limitation is encountered with a :class:`MultiIndex` and any names
    beginning with ``'level_'``.

    Examples
    --------
    >>> from io import StringIO
    >>> df = pd.DataFrame(
    ...     [["a", "b"], ["c", "d"]],
    ...     index=["row 1", "row 2"],
    ...     columns=["col 1", "col 2"],
    ... )

    Encoding/decoding a Dataframe using ``'split'`` formatted JSON:

    >>> df.to_json(orient="split")
    '{"columns":["col 1","col 2"],"index":["row 1","row 2"],"data":[["a","b"],["c","d"]]}'

    >>> pd.read_json(StringIO(_), orient="split")  # noqa: F821
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'index'`` formatted JSON:

    >>> df.to_json(orient="index")
    '{"row 1":{"col 1":"a","col 2":"b"},"row 2":{"col 1":"c","col 2":"d"}}'

    >>> pd.read_json(StringIO(_), orient="index")  # noqa: F821
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'records'`` formatted JSON.
    Note that index labels are not preserved with this encoding.

    >>> df.to_json(orient="records")
    '[{"col 1":"a","col 2":"b"},{"col 1":"c","col 2":"d"}]'

    >>> pd.read_json(StringIO(_), orient="records")  # noqa: F821
      col 1 col 2
    0     a     b
    1     c     d

    Encoding with Table Schema

    >>> df.to_json(orient="table")
    '{"schema":{"fields":[{"name":"index","type":"string","extDtype":"str"},{"name":"col 1","type":"string","extDtype":"str"},{"name":"col 2","type":"string","extDtype":"str"}],"primaryKey":["index"],"pandas_version":"1.4.0"},"data":[{"index":"row 1","col 1":"a","col 2":"b"},{"index":"row 2","col 1":"c","col 2":"d"}]}'

    The following example uses ``dtype_backend="numpy_nullable"``

    >>> data = '''{"index": {"0": 0, "1": 1},
    ...        "a": {"0": 1, "1": null},
    ...        "b": {"0": 2.5, "1": 4.5},
    ...        "c": {"0": true, "1": false},
    ...        "d": {"0": "a", "1": "b"},
    ...        "e": {"0": 1577.2, "1": 1577.1}}'''
    >>> pd.read_json(StringIO(data), dtype_backend="numpy_nullable")
       index     a    b      c  d       e
    0      0     1  2.5   True  a  1577.2
    1      1  <NA>  4.5  False  b  1577.1
    """  # noqa: E501
    if orient == "table" and dtype:
        raise ValueError("cannot pass both dtype and orient='table'")
    if orient == "table" and convert_axes:
        raise ValueError("cannot pass both convert_axes and orient='table'")

    check_dtype_backend(dtype_backend)

    if dtype is None and orient != "table":
        # error: Incompatible types in assignment (expression has type "bool", variable
        # has type "Union[ExtensionDtype, str, dtype[Any], Type[str], Type[float],
        # Type[int], Type[complex], Type[bool], Type[object], Dict[Hashable,
        # Union[ExtensionDtype, Union[str, dtype[Any]], Type[str], Type[float],
        # Type[int], Type[complex], Type[bool], Type[object]]], None]")
        dtype = True  # type: ignore[assignment]
    if convert_axes is None and orient != "table":
        convert_axes = True

    json_reader = JsonReader(
        path_or_buf,
        orient=orient,
        typ=typ,
        dtype=dtype,
        convert_axes=convert_axes,
        convert_dates=convert_dates,
        keep_default_dates=keep_default_dates,
        precise_float=precise_float,
        date_unit=date_unit,
        encoding=encoding,
        lines=lines,
        chunksize=chunksize,
        compression=compression,
        nrows=nrows,
        storage_options=storage_options,
        encoding_errors=encoding_errors,
        dtype_backend=dtype_backend,
        engine=engine,
    )

    if chunksize:
        return json_reader
    else:
        return json_reader.read()


@set_module("pandas.api.typing")
class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    # [CLASS DOC]
    # Iterator and context manager for reading JSON input; the type returned by the
    # reader when chunksize is given (exposed in the public typing namespace).
    # Generic in whether it produces frames or series.
    # State: the options passed to the constructor (orient, typ, dtype, convert_axes,
    # convert_dates, keep_default_dates, precise_float, date_unit, encoding, engine,
    # compression, storage_options, lines, chunksize, nrows, encoding_errors,
    # dtype_backend); nrows_seen (rows yielded so far in chunk mode); handles (the
    # opened file handles, or none when the caller supplied an open stream); data
    # (for the ujson engine either the full text or the open stream when
    # chunking or limiting rows; for pyarrow the raw input).
    # Methods: read for whole-input reads; iteration yields chunks of the lines
    # format; close releases handles opened here; enter/exit make it a context manager.
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """

    def __init__(
        self,
        filepath_or_buffer,
        orient,
        typ: FrameSeriesStrT,
        dtype,
        convert_axes: bool | None,
        convert_dates,
        keep_default_dates: bool,
        precise_float: bool,
        date_unit,
        encoding,
        lines: bool,
        chunksize: int | None,
        compression: CompressionOptions,
        nrows: int | None,
        storage_options: StorageOptions | None = None,
        encoding_errors: str | None = "strict",
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        engine: JSONEngine = "ujson",
    ) -> None:
        # [FUNCTION DOC]
        # Stores options and prepares the data. Raises a value error for an engine other
        # than ujson or pyarrow; for a chunksize that is not a positive integer (checked
        # by an integer validator), or given without lines, or given with the pyarrow
        # engine; for nrows that is negative or non-integer, or given without lines;
        # and for the pyarrow engine without lines.
        # The pyarrow engine stores the input untouched. The ujson engine opens the
        # input via the file helper; when neither chunksize nor nrows is set the whole
        # content is read immediately (and the handle closed), otherwise the open stream is
        # kept for lazy line reading. Raises a file-not-found error if the path cannot be opened.
        self.orient = orient
        self.typ = typ
        self.dtype = dtype
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.keep_default_dates = keep_default_dates
        self.precise_float = precise_float
        self.date_unit = date_unit
        self.encoding = encoding
        self.engine = engine
        self.compression = compression
        self.storage_options = storage_options
        self.lines = lines
        self.chunksize = chunksize
        self.nrows_seen = 0
        self.nrows = nrows
        self.encoding_errors = encoding_errors
        self.handles: IOHandles[str] | None = None
        self.dtype_backend = dtype_backend

        if self.engine not in {"pyarrow", "ujson"}:
            raise ValueError(
                f"The engine type {self.engine} is currently not supported."
            )
        if self.chunksize is not None:
            self.chunksize = validate_integer("chunksize", self.chunksize, 1)
            if not self.lines:
                raise ValueError("chunksize can only be passed if lines=True")
            if self.engine == "pyarrow":
                raise ValueError(
                    "currently pyarrow engine doesn't support chunksize parameter"
                )
        if self.nrows is not None:
            self.nrows = validate_integer("nrows", self.nrows, 0)
            if not self.lines:
                raise ValueError("nrows can only be passed if lines=True")
        if self.engine == "pyarrow":
            if not self.lines:
                raise ValueError(
                    "currently pyarrow engine only supports "
                    "the line-delimited JSON format"
                )
            self.data = filepath_or_buffer
        elif self.engine == "ujson":
            data = self._get_data_from_filepath(filepath_or_buffer)
            # If self.chunksize, we prepare the data for the `__next__` method.
            # Otherwise, we read it into memory for the `read` method.
            if not (self.chunksize or self.nrows):
                with self:
                    self.data = data.read()
            else:
                self.data = data

    def _get_data_from_filepath(self, filepath_or_buffer):
        # [FUNCTION DOC]
        # HELPER: opens the input. Converts path-like input to a string, opens a text
        # handle with the configured encoding, compression, storage options and error
        # handling, stores the handle set in the handles attribute and returns the
        # underlying readable stream. An OS error becomes a file-not-found error.
        # KNOWN ISSUE: the docstring claims three input types but lists two.
        """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
        """
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        try:
            self.handles = get_handle(
                filepath_or_buffer,
                "r",
                encoding=self.encoding,
                compression=self.compression,
                storage_options=self.storage_options,
                errors=self.encoding_errors,
            )
        except OSError as err:
            raise FileNotFoundError(
                f"File {filepath_or_buffer} does not exist"
            ) from err
        filepath_or_buffer = self.handles.handle
        return filepath_or_buffer

    def _combine_lines(self, lines) -> str:
        # [FUNCTION DOC]
        # HELPER: joins lines of line-delimited JSON into one JSON array text. Each line
        # is stripped and blank lines are dropped; the rest are joined with commas inside
        # square brackets. Returns the string.
        """
        Combines a list of JSON objects into one JSON object.
        """
        return (
            f"[{','.join([line for line in (line.strip() for line in lines) if line])}]"
        )

    @overload
    def read(self: JsonReader[Literal["frame"]]) -> DataFrame: ...
        # [FUNCTION DOC]
        # Typing overload only: reader of frames returns a frame. See the implementation.

    @overload
    def read(self: JsonReader[Literal["series"]]) -> Series: ...
        # [FUNCTION DOC]
        # Typing overload only: reader of series returns a series. See the implementation.

    @overload
    def read(self: JsonReader[Literal["frame", "series"]]) -> DataFrame | Series: ...
        # [FUNCTION DOC]
        # Typing overload only: reader of either returns a frame or series. See the
        # implementation.

    def read(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Reads the entire input into one frame or series. Runs inside the reader's
        # context so handles are closed afterwards. Dispatches on the engine to the
        # pyarrow or ujson implementation. Returns the object.
        """
        Read the whole JSON input into a pandas object.
        """
        obj: DataFrame | Series
        with self:
            if self.engine == "pyarrow":
                obj = self._read_pyarrow()
            elif self.engine == "ujson":
                obj = self._read_ujson()

        return obj

    def _read_pyarrow(self) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: reads with the pyarrow JSON reader. If dtype is a mapping, builds an
        # explicit schema from the entries that map to arrow-backed dtypes (other
        # entries are ignored) and lets unexpected fields be inferred. Converts the
        # arrow table to a frame honoring dtype_backend. Raises an import error when
        # pyarrow is missing. Returns a frame.
        """
        Read JSON using the pyarrow engine.
        """
        pyarrow_json = import_optional_dependency("pyarrow.json")
        options = None

        if isinstance(self.dtype, dict):
            pa = import_optional_dependency("pyarrow")
            fields = []
            for field, dtype in self.dtype.items():
                pd_dtype = pandas_dtype(dtype)
                if isinstance(pd_dtype, ArrowDtype):
                    fields.append((field, pd_dtype.pyarrow_dtype))

            schema = pa.schema(fields)
            options = pyarrow_json.ParseOptions(
                explicit_schema=schema, unexpected_field_behavior="infer"
            )

        pa_table = pyarrow_json.read_json(self.data, parse_options=options)
        df = arrow_table_to_pandas(pa_table, dtype_backend=self.dtype_backend)

        return df

    def _read_ujson(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: reads with the ujson engine. For line-delimited input: with chunksize,
        # concatenates all chunks from iterating the reader; with nrows, takes that many
        # lines from the stream, combines them and parses; otherwise splits the full
        # text on newlines, combines and parses. For regular JSON parses the whole text.
        # If a dtype_backend is set, converts the result to nullable dtypes of that
        # backend without distinguishing NaN from NA (temporary option change).
        # Returns a frame or series.
        """
        Read JSON using the ujson engine.
        """
        obj: DataFrame | Series
        if self.lines:
            if self.chunksize:
                obj = concat(self)
            elif self.nrows:
                lines = list(islice(self.data, self.nrows))
                lines_json = self._combine_lines(lines)
                obj = self._get_object_parser(lines_json)
            else:
                data = ensure_str(self.data)
                data_lines = data.split("\n")
                obj = self._get_object_parser(self._combine_lines(data_lines))
        else:
            obj = self._get_object_parser(self.data)
        if self.dtype_backend is not lib.no_default:
            with option_context("future.distinguish_nan_and_na", False):
                return obj.convert_dtypes(
                    infer_objects=False, dtype_backend=self.dtype_backend
                )
        else:
            return obj

    def _get_object_parser(self, json: str) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: decodes one JSON text into a frame or series via the frame or series
        # parser according to typ, passing the stored options (orient, dtype,
        # convert_axes, convert_dates, keep_default_dates, precise_float, date_unit,
        # dtype_backend). Raises a value error for any other typ.
        # KNOWN ISSUE: in the series branch, the dtype is reassigned to the same value when it is not a boolean, which has no effect.
        """
        Parses a json document into a pandas object.
        """
        typ = self.typ
        dtype = self.dtype
        kwargs = {
            "orient": self.orient,
            "dtype": self.dtype,
            "convert_axes": self.convert_axes,
            "convert_dates": self.convert_dates,
            "keep_default_dates": self.keep_default_dates,
            "precise_float": self.precise_float,
            "date_unit": self.date_unit,
            "dtype_backend": self.dtype_backend,
        }
        if typ == "frame":
            return FrameParser(json, **kwargs).parse()
        elif typ == "series":
            if not isinstance(dtype, bool):
                kwargs["dtype"] = dtype
            return SeriesParser(json, **kwargs).parse()
        else:
            raise ValueError(f"{typ=} must be 'frame' or 'series'.")

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the file handles if this reader opened them (handles is not none);
        # streams passed in by the caller are left open. Safe to call repeatedly.
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
        if self.handles is not None:
            self.handles.close()

    def __iter__(self) -> Self:
        # [FUNCTION DOC]
        # Returns the reader itself so that it acts as its own iterator.
        return self

    @overload
    def __next__(self: JsonReader[Literal["frame"]]) -> DataFrame: ...
        # [FUNCTION DOC]
        # Typing overload only: frame reader yields a frame per chunk.

    @overload
    def __next__(self: JsonReader[Literal["series"]]) -> Series: ...
        # [FUNCTION DOC]
        # Typing overload only: series reader yields a series per chunk.

    @overload
    def __next__(
        self: JsonReader[Literal["frame", "series"]],
    ) -> DataFrame | Series: ...
        # [FUNCTION DOC]
        # Typing overload only: yields a frame or series per chunk.

    def __next__(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Returns the next chunk of line-delimited input. Stops iteration (closing
        # handles) when nrows is set and already reached, or when no more lines remain.
        # Reads up to chunksize lines (all remaining when chunksize is none), combines and
        # parses them, then replaces the index by consecutive integers continuing from
        # the number of rows already returned and updates that counter. Any exception
        # during parsing closes the handles and is re-raised. If a dtype_backend is set the
        # chunk is converted to nullable dtypes.
        # KNOWN ISSUE: when nrows is smaller than a chunk, the chunk is not truncated, so more than nrows rows can be returned in total.
        if self.nrows and self.nrows_seen >= self.nrows:
            self.close()
            raise StopIteration

        lines = list(islice(self.data, self.chunksize))
        if not lines:
            self.close()
            raise StopIteration

        try:
            lines_json = self._combine_lines(lines)
            obj = self._get_object_parser(lines_json)

            # Make sure that the returned objects have the right index.
            obj.index = range(self.nrows_seen, self.nrows_seen + len(obj))
            self.nrows_seen += len(obj)
        except Exception as ex:
            self.close()
            raise ex

        if self.dtype_backend is not lib.no_default:
            with option_context("future.distinguish_nan_and_na", False):
                return obj.convert_dtypes(
                    infer_objects=False, dtype_backend=self.dtype_backend
                )
        else:
            return obj

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
        # Context manager exit; closes the handles regardless of any exception and returns
        # nothing (exceptions propagate).
        self.close()


class Parser:
    # [CLASS DOC]
    # Abstract base for decoding JSON text into frames or series and coercing types.
    # Class constants: allowed time units, and a table of minimum plausible epoch
    # values per unit, used to decide whether numbers could be timestamps.
    # State: json (text), orient (default orient of the subclass when none),
    # dtype, convert_axes, convert_dates, keep_default_dates, precise_float,
    # date_unit (lower-cased), min_stamp (minimum epoch value for the chosen unit,
    # seconds if no unit), dtype_backend. Subclasses supply the decode step, the
    # split-key set, the default orient and the type conversion step.
    _split_keys: tuple[str, ...]
    _default_orient: str

    _STAMP_UNITS = ("s", "ms", "us", "ns")
    _MIN_STAMPS = {
        "s": 31536000,
        "ms": 31536000000,
        "us": 31536000000000,
        "ns": 31536000000000000,
    }
    json: str

    def __init__(
        self,
        json: str,
        orient,
        dtype: DtypeArg | None = None,
        convert_axes: bool = True,
        convert_dates: bool | list[str] = True,
        keep_default_dates: bool = False,
        precise_float: bool = False,
        date_unit=None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the options. A none orient becomes the class default. A date_unit,
        # if provided, is lower-cased and must be one of s, ms, us, ns, otherwise a value
        # error is raised; it sets the minimum epoch threshold (seconds threshold when no
        # unit given). Defaults: dtype none, convert_axes true, convert_dates true,
        # keep_default_dates false, precise_float false, date_unit none,
        # dtype_backend unset.
        self.json = json

        if orient is None:
            orient = self._default_orient

        self.orient = orient

        self.dtype = dtype

        if date_unit is not None:
            date_unit = date_unit.lower()
            if date_unit not in self._STAMP_UNITS:
                raise ValueError(f"date_unit must be one of {self._STAMP_UNITS}")
            self.min_stamp = self._MIN_STAMPS[date_unit]
        else:
            self.min_stamp = self._MIN_STAMPS["s"]

        self.precise_float = precise_float
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.date_unit = date_unit
        self.keep_default_dates = keep_default_dates
        self.dtype_backend = dtype_backend

    @final
    def check_keys_split(self, decoded: dict) -> None:
        # [FUNCTION DOC]
        # Validates the keys of a split-orient mapping: raises a value error listing the
        # keys that are not in the class's allowed split keys. Returns nothing.
        """
        Checks that dict has only the appropriate keys for orient='split'.
        """
        bad_keys = set(decoded.keys()).difference(set(self._split_keys))
        if bad_keys:
            bad_keys_joined = ", ".join(bad_keys)
            raise ValueError(f"JSON data had unexpected key(s): {bad_keys_joined}")

    @final
    def parse(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # Main entry point. Decodes the text via the subclass decode step, converts axes
        # when convert_axes is set, then converts value types via the subclass step.
        # Returns the resulting frame or series.
        obj = self._parse()

        if self.convert_axes:
            obj = self._convert_axes(obj)
        obj = self._try_convert_types(obj)
        return obj

    def _parse(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: abstract decode step implemented by subclasses; base raises an
        # abstract-method error.
        raise AbstractMethodError(self)

    @final
    def _convert_axes(self, obj: DataFrame | Series) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: attempts to convert each axis of the object (in place) into better
        # typed labels. Each axis is wrapped as a series and run through the data
        # conversion helper without honoring dtype settings but allowing date
        # conversion; if the helper reports a change, the axis is replaced with the
        # converted labels. Returns the same object.
        """
        Try to convert axes.
        """
        for axis_name in obj._AXIS_ORDERS:
            ax = obj._get_axis(axis_name)
            ser = Series(ax, dtype=ax.dtype, copy=False)
            new_ser, result = self._try_convert_data(
                name=axis_name,
                data=ser,
                use_dtypes=False,
                convert_dates=True,
                is_axis=True,
            )
            if result:
                new_axis = Index(new_ser, dtype=new_ser.dtype, copy=False)
                setattr(obj, axis_name, new_axis)
        return obj

    def _try_convert_types(self, obj):
        # [FUNCTION DOC]
        # HELPER: abstract hook to coerce value types of the decoded object; subclasses
        # return the converted object. Base raises an abstract-method error.
        raise AbstractMethodError(self)

    @final
    def _try_convert_data(
        self,
        name: Hashable,
        data: Series,
        use_dtypes: bool = True,
        convert_dates: bool | list[str] = True,
        is_axis: bool = False,
    ) -> tuple[Series, bool]:
        # [FUNCTION DOC]
        # HELPER: tries to coerce one series to a better dtype. Parameters: name is the
        # column or axis name; data the series; use_dtypes says whether the dtype option
        # applies; convert_dates says whether (or for which names) date parsing is
        # allowed; is_axis says it is an axis being converted. Returns a pair of the
        # resulting series and a flag telling whether it was changed.
        # Behavior: when dtype use is on, a falsy dtype leaves fully non-missing data
        # untouched and otherwise returns data with missing values normalized to NaN;
        # a mapping or single dtype is applied through a cast (when the column is not
        # a date column) and failure returns the original unchanged; a true dtype goes to
        # inference. Then date parsing is attempted. With a dtype backend set and not an
        # axis, data is returned changed so later conversion handles it. Otherwise string
        # data is tried as float64, other floats are widened to float64, float or object
        # data is tried as int64 when lossless, and smaller ints are widened to int64.
        # For the index of a split orient with data, the result is reported as unchanged.
        """
        Try to parse a Series into a column by inferring dtype.
        """
        org_data = data
        # don't try to coerce, unless a force conversion
        if use_dtypes:
            if not self.dtype:
                if all(notna(data)):
                    return data, False

                filled = data.fillna(np.nan)

                return filled, True

            elif self.dtype is True:
                pass
            elif not _should_convert_dates(
                convert_dates, self.keep_default_dates, name
            ):
                # convert_dates takes precedence over columns listed in dtypes
                dtype = (
                    self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype
                )
                if dtype is not None:
                    try:
                        return data.astype(dtype), True
                    except (TypeError, ValueError):
                        return data, False

        if convert_dates:
            new_data = self._try_convert_to_date(data)
            if new_data is not data:
                return new_data, True

        converted = False
        if self.dtype_backend is not lib.no_default and not is_axis:
            # Fall through for conversion later on
            return data, True
        elif is_string_dtype(data.dtype):
            # try float
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        if data.dtype.kind == "f" and data.dtype != "float64":
            # coerce floats to 64
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        # don't coerce 0-len data
        if len(data) and data.dtype in ("float", "object"):
            # coerce ints if we can
            try:
                new_data = org_data.astype("int64")
                if (new_data == data).all():
                    data = new_data
                    converted = True
            except (TypeError, ValueError, OverflowError):
                pass

        if data.dtype == "int" and data.dtype != "int64":
            # coerce ints to 64
            try:
                data = data.astype("int64")
                converted = True
            except (TypeError, ValueError):
                pass

        # if we have an index, we want to preserve dtypes
        if name == "index" and len(data):
            if self.orient == "split":
                return data, False

        return data, converted

    @final
    def _try_convert_to_date(self, data: Series) -> Series:
        # [FUNCTION DOC]
        # HELPER: attempts to parse a series into datetimes. Returns it unchanged if empty.
        # Object or string data is first tried as int64 (overflow returns unchanged data).
        # Numbers below the minimum timestamp threshold (ignoring missing values and the
        # missing marker) leave the data unchanged. String data is tried with
        # datetime parsing using default, ISO 8601 and mixed formats, ignoring format
        # inference warnings, returning the first success. Other data is tried with each
        # allowed unit (only date_unit when set), requiring the result to fit into
        # nanosecond bounds; out-of-bounds moves to the next unit, other errors are
        # ignored. Returns converted data or the (possibly integer-converted) original.
        """
        Try to parse an ndarray like into a date column.

        Try to coerce object in epoch/iso formats and integer/float in epoch
        formats.
        """
        # no conversion on empty
        if not len(data):
            return data

        new_data = data

        if new_data.dtype == "object" or new_data.dtype == "string":  # noqa: PLR1714
            try:
                new_data = data.astype("int64")
            except OverflowError:
                return data
            except (TypeError, ValueError):
                pass

        # ignore numbers that are out of range
        if issubclass(new_data.dtype.type, np.number):
            in_range = (
                isna(new_data._values)
                | (new_data > self.min_stamp)
                | (new_data._values == iNaT)
            )
            if not in_range.all():
                return data

        if new_data.dtype == "string":
            with warnings.catch_warnings():
                # ignore "Could not infer format" warnings from to_datetime
                # which is incorrectly raised for non-date strings
                warnings.simplefilter("ignore", UserWarning)
                for format in (None, "iso8601", "mixed"):
                    try:
                        return to_datetime(new_data, errors="raise", format=format)
                    except Exception:
                        pass
        else:
            # numeric or mixed objects
            date_units = (self.date_unit,) if self.date_unit else self._STAMP_UNITS
            for date_unit in date_units:
                try:
                    # In case of multiple possible units, infer the likely unit
                    # based on the first unit for which the parsed dates fit
                    # within the nanoseconds bounds
                    # -> do as_unit cast to ensure OutOfBounds error
                    data = to_datetime(new_data, errors="raise", unit=date_unit)
                    _ = data.dt.as_unit("ns")
                    break
                except OutOfBoundsDatetime:
                    continue
                except (ValueError, OverflowError, TypeError):
                    pass
        return data


class SeriesParser(Parser):
    # [CLASS DOC]
    # Parser producing a series. Default orient is index; allowed split keys are
    # name, index and data.
    _default_orient = "index"
    _split_keys = ("name", "index", "data")

    def _parse(self) -> Series:
        # [FUNCTION DOC]
        # HELPER: decodes the JSON text with the C decoder (honoring precise_float). For
        # split orient, checks the keys and builds a series from name, index and data
        # entries; otherwise builds a series from the decoded mapping or list. Returns
        # a series. Raises a value error for unexpected split keys.
        data = ujson_loads(self.json, precise_float=self.precise_float)

        if self.orient == "split":
            decoded = {str(k): v for k, v in data.items()}
            self.check_keys_split(decoded)
            return Series(**decoded)
        else:
            return Series(data)

    def _try_convert_types(self, obj: Series) -> Series:
        # [FUNCTION DOC]
        # HELPER: applies the data conversion helper to the whole series (labeled data,
        # with the configured date conversion) and returns the converted series.
        obj, _ = self._try_convert_data("data", obj, convert_dates=self.convert_dates)
        return obj


class FrameParser(Parser):
    # [CLASS DOC]
    # Parser producing a frame. Default orient is columns; allowed split keys are
    # columns, index and data.
    _default_orient = "columns"
    _split_keys = ("columns", "index", "data")

    def _parse(self) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: decodes the text into a frame according to orient. Split: decodes,
        # checks keys, turns list-valued column labels into tuples and de-duplicates
        # column names (as a multi-level label set when applicable), then builds a
        # frame without dtype inference. Index orient: builds from a mapping of rows.
        # Table orient: delegates to the Table Schema parser. Any other orient (including
        # columns) builds from a mapping of columns. Raises a value error for unexpected
        # split keys.
        json = self.json
        orient = self.orient

        if orient == "split":
            decoded = {
                str(k): v
                for k, v in ujson_loads(json, precise_float=self.precise_float).items()
            }
            self.check_keys_split(decoded)
            orig_names = [
                (tuple(col) if isinstance(col, list) else col)
                for col in decoded["columns"]
            ]
            decoded["columns"] = dedup_names(
                orig_names,
                is_potential_multi_index(orig_names, None),
            )
            return DataFrame(dtype=None, **decoded)
        elif orient == "index":
            return DataFrame.from_dict(
                ujson_loads(json, precise_float=self.precise_float),
                dtype=None,
                orient="index",
            )
        elif orient == "table":
            return parse_table_schema(json, precise_float=self.precise_float)
        else:
            # includes orient == "columns"
            return DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )

    def _try_convert_types(self, obj: DataFrame) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: converts the type of every column. For each column applies the data
        # conversion helper with date conversion decided per column by the date-column
        # predicate, collects the resulting arrays and rebuilds a frame with the same
        # columns and index without integrity verification. Returns the new frame.
        arrays = []
        for col_label, series in obj.items():
            result, _ = self._try_convert_data(
                col_label,
                series,
                convert_dates=_should_convert_dates(
                    self.convert_dates,
                    keep_default_dates=self.keep_default_dates,
                    col=col_label,
                ),
            )
            arrays.append(result.array)
        return DataFrame._from_arrays(
            arrays, obj.columns, obj.index, verify_integrity=False
        )


def _should_convert_dates(
    convert_dates: bool | list[str],
    keep_default_dates: bool,
    col: Hashable,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: predicate for the parsers. Decides whether a column should be parsed
    # as dates. convert_dates false: never. A list containing the column: yes.
    # Otherwise requires keep_default_dates and a string label whose lower-cased
    # form ends with an at or time suffix, equals modified, date or datetime, or starts
    # with timestamp. Returns a boolean.
    # KNOWN ISSUE: the comment in the first branch says that true follows the default-dates setting, which is misplaced since that branch handles the false case.
    """
    Return bool whether a DataFrame column should be cast to datetime.
    """
    if convert_dates is False:
        # convert_dates=True means follow keep_default_dates
        return False
    elif not isinstance(convert_dates, bool) and col in set(convert_dates):
        return True
    elif not keep_default_dates:
        return False
    elif not isinstance(col, str):
        return False
    col_lower = col.lower()
    if (
        col_lower.endswith(("_at", "_time"))
        or col_lower in {"modified", "date", "datetime"}
        or col_lower.startswith("timestamp")
    ):
        return True
    return False
