# [MODULE DOC]
# HELPER: central module of type aliases, type variables and structural
# protocols used for annotations throughout pandas. It is for static type
# checking; nothing here changes runtime behavior.
# Contents (grouped as in the file): parameter-spec and type variables (P,
# HashableT, MutableMappingT, ArrayLikeT, IntStrT, NDFrameT, IndexT, FreqIndexT,
# NumpyIndexT, T, F, TypeT, SequenceT and the covariant/contravariant string
# variables); aliases for array-likes, list-likes, scalars, timestamp and
# timedelta convertible inputs, timezones, axes, labels, shapes, dtypes and
# dtype arguments, converters and date-parsing arguments, renamers, function
# types and aggregation function types, file path and storage and compression
# options, formatter types, fill and interpolate options, indexers, merge and
# join options, interval closed types, sorting and quantile options, engines
# and parsers for readers, and many other string-literal option types; and
# structural protocols for sequences that exclude strings (SequenceNotStr),
# file-like buffers (BaseBuffer, ReadBuffer, WriteBuffer, ReadPickleBuffer,
# WriteExcelBuffer, ReadCsvBuffer) and objects exporting data through the Arrow
# PyCapsule interface (ArrowArrayExportable, ArrowStreamExportable).
# The module's __all__ contains only the alias for the builtin type function
# (type_t), which exists so annotations can name the builtin type even where a
# local name shadows it.
# Import cycles are avoided: pandas internal types are imported only under the
# type-checking flag and referenced by string forward references (a union
# containing a pandas type must use the Union form for this reason). A few
# protocol classes set their reported module to the public typing aliases
# module so documentation shows a public path.
# Import-time side effects: none beyond importing numpy typing helpers.
# KNOWN ISSUE: the timestamp-nonexistent alias and the time-nonexistent alias
# are defined identically, duplicating the same literal choices.
from __future__ import annotations

from builtins import type as type_t  # pyright: ignore[reportUnusedImport]
from collections.abc import (
    Callable,
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import (
    date,
    datetime,
    timedelta,
    tzinfo,
)
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles

# Note that Union is needed when a Union includes a pandas type

if TYPE_CHECKING:
    from pandas._libs import (
        NaTType,
        Period,
        Timedelta,
        Timestamp,
    )
    from pandas._libs.tslibs import BaseOffset

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import (
        DatetimeIndex,
        Interval,
        PeriodIndex,
        TimedeltaIndex,
    )
    from pandas.arrays import (
        DatetimeArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.base import ExtensionArray
    from pandas.core.frame import DataFrame
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.generic import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )
    from pandas.core.indexes.base import Index
    from pandas.core.internals import (
        BlockManager,
        SingleBlockManager,
    )
    from pandas.core.resample import Resampler
    from pandas.core.series import Series
    from pandas.core.window.rolling import BaseWindow

    from pandas.io.formats.format import EngFormatter
    from pandas.tseries.holiday import AbstractHolidayCalendar

    ScalarLike_co: TypeAlias = int | float | complex | str | bytes | np.generic

    # numpy compatible types
    NumpyValueArrayLike: TypeAlias = ScalarLike_co | npt.ArrayLike
    NumpySorter: TypeAlias = npt.NDArray[np.integer] | None


P = ParamSpec("P")

HashableT = TypeVar("HashableT", bound=Hashable)
HashableT2 = TypeVar("HashableT2", bound=Hashable)
MutableMappingT = TypeVar("MutableMappingT", bound=MutableMapping)

# array-like

ArrayLike: TypeAlias = Union["ExtensionArray", np.ndarray]
ArrayLikeT = TypeVar("ArrayLikeT", "ExtensionArray", np.ndarray)
AnyArrayLike: TypeAlias = Union[ArrayLike, "Index", "Series"]
TimeArrayLike: TypeAlias = Union["DatetimeArray", "TimedeltaArray"]

# list-like

# from https://github.com/hauntsaninja/useful_types
# includes Sequence-like objects but excludes str and bytes
_T_co = TypeVar("_T_co", covariant=True)


class SequenceNotStr(Protocol[_T_co]):
    # [CLASS DOC]
    # Structural protocol (generic over a covariant element type) describing a
    # read-only sequence that is deliberately not satisfied by text or bytes
    # (idea from the useful_types project). Required members: indexing by
    # integer-like or by slice, membership test, length, iteration, index lookup
    # with optional start and stop, count, and reversed iteration. Because str
    # has a differing signature for index, it is excluded by checkers. Reported
    # module is the public typing aliases module. Only for static checking.
    __module__: str = "pandas.api.typing.aliases"

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co:
        # [FUNCTION DOC]
        # Protocol stub (overload for an integer-like index): returns one element.
        # No implementation exists; the combined signature is declared by both
        # overloads.
        ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]:
        # [FUNCTION DOC]
        # Protocol stub (overload for a slice): returns a sequence of elements.
        # No implementation exists in the protocol.
        ...

    def __contains__(self, value: object, /) -> bool:
        # [FUNCTION DOC]
        # Protocol stub: membership test for any object; returns a boolean.
        ...

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Protocol stub: number of elements, as an integer.
        ...

    def __iter__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC]
        # Protocol stub: returns an iterator over the elements in order.
        ...

    def index(self, value: Any, start: int = ..., stop: int = ..., /) -> int:
        # [FUNCTION DOC]
        # Protocol stub: position of the first occurrence of value, optionally
        # restricted to the range from start to stop (both positional-only,
        # optional). Returns an integer.
        ...

    def count(self, value: Any, /) -> int:
        # [FUNCTION DOC]
        # Protocol stub: number of occurrences of value, as an integer.
        ...

    def __reversed__(self) -> Iterator[_T_co]:
        # [FUNCTION DOC]
        # Protocol stub: returns an iterator over the elements in reverse order.
        ...


ListLike: TypeAlias = AnyArrayLike | SequenceNotStr | range

# scalars

PythonScalar: TypeAlias = str | float | bool
DatetimeLikeScalar: TypeAlias = Union["Period", "Timestamp", "Timedelta"]

# aligned with pandas-stubs - typical scalars found in Series.  Explicitly leaves
# out object
_IndexIterScalar: TypeAlias = Union[
    str,
    bytes,
    date,
    datetime,
    timedelta,
    np.datetime64,
    np.timedelta64,
    bool,
    int,
    float,
    "Timestamp",
    "Timedelta",
]
Scalar: TypeAlias = Union[
    _IndexIterScalar, "Interval", complex, np.integer, np.floating, np.complexfloating
]

IntStrT = TypeVar("IntStrT", bound=int | str)

# timestamp and timedelta convertible types

TimestampConvertibleTypes: TypeAlias = Union[
    "Timestamp", date, np.datetime64, np.int64, float, str
]
TimestampNonexistent: TypeAlias = (
    Literal["shift_forward", "shift_backward", "NaT", "raise"] | timedelta
)

TimedeltaConvertibleTypes: TypeAlias = Union[
    "Timedelta", timedelta, np.timedelta64, np.int64, float, str
]
Timezone: TypeAlias = str | tzinfo

ToTimestampHow: TypeAlias = Literal["s", "e", "start", "end"]

# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound="NDFrame")

IndexT = TypeVar("IndexT", bound="Index")
FreqIndexT = TypeVar("FreqIndexT", "DatetimeIndex", "PeriodIndex", "TimedeltaIndex")
NumpyIndexT = TypeVar("NumpyIndexT", np.ndarray, "Index")

AxisInt: TypeAlias = int
Axis: TypeAlias = AxisInt | Literal["index", "columns", "rows"]
IndexLabel: TypeAlias = Hashable | Sequence[Hashable]
Level: TypeAlias = Hashable
Shape: TypeAlias = tuple[int, ...]
Suffixes: TypeAlias = Sequence[str | None]
Ordered: TypeAlias = bool | None
JSONSerializable: TypeAlias = PythonScalar | list | dict | None
Frequency: TypeAlias = Union[str, "BaseOffset"]
Axes: TypeAlias = ListLike

RandomState: TypeAlias = (
    int
    | np.ndarray
    | np.random.Generator
    | np.random.BitGenerator
    | np.random.RandomState
)


# dtypes
NpDtype: TypeAlias = str | np.dtype | type[str | complex | bool | object]
Dtype: TypeAlias = Union["ExtensionDtype", NpDtype]
AstypeArg: TypeAlias = Union["ExtensionDtype", npt.DTypeLike]
# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg: TypeAlias = Dtype | Mapping[Hashable, Dtype]
DtypeObj: TypeAlias = Union[np.dtype, "ExtensionDtype"]

# converters
ConvertersArg: TypeAlias = dict[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg: TypeAlias = (
    bool | list[Hashable] | list[list[Hashable]] | dict[Hashable, list[Hashable]]
)

# For functions like rename that convert one label to another
Renamer: TypeAlias = Mapping[Any, Hashable] | Callable[[Any], Hashable]

# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType: TypeAlias = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
TypeT = TypeVar("TypeT", bound=type)

# types of vectorized key functions for DataFrame::sort_values and
# DataFrame::sort_index, among others
ValueKeyFunc: TypeAlias = Callable[["Series"], Union["Series", AnyArrayLike]] | None
IndexKeyFunc: TypeAlias = Callable[["Index"], Union["Index", AnyArrayLike]] | None

# types of `func` kwarg for DataFrame.aggregate and Series.aggregate
AggFuncTypeBase: TypeAlias = Callable | str
AggFuncTypeDict: TypeAlias = MutableMapping[
    Hashable, AggFuncTypeBase | list[AggFuncTypeBase]
]
AggFuncType: TypeAlias = AggFuncTypeBase | list[AggFuncTypeBase] | AggFuncTypeDict
AggObjType: TypeAlias = Union[
    "Series",
    "DataFrame",
    "GroupBy",
    "SeriesGroupBy",
    "DataFrameGroupBy",
    "BaseWindow",
    "Resampler",
]

PythonFuncType: TypeAlias = Callable[[Any], Any]

# filenames and file-like-objects
AnyStr_co = TypeVar("AnyStr_co", str, bytes, covariant=True)
AnyStr_contra = TypeVar("AnyStr_contra", str, bytes, contravariant=True)


class BaseBuffer(Protocol):
    # [CLASS DOC]
    # Structural protocol for the minimal file-like object accepted by the
    # reading and writing machinery: a mode property and the methods seek,
    # seekable and tell. The comments in the source note which standard file
    # classes motivated each member. Base of the read and write buffer
    # protocols. Only for static checking.
    @property
    def mode(self) -> str:
        # [FUNCTION DOC]
        # Protocol stub (property): the mode string of the open buffer. Needed by
        # the internal file-path-or-buffer resolution.
        # for _get_filepath_or_buffer
        ...

    def seek(self, offset: int, whence: int = ..., /) -> int:
        # [FUNCTION DOC]
        # Protocol stub: moves to a position. Parameters: offset (integer) and an
        # optional whence reference point (positional only). Returns the new
        # absolute position. The source comment notes some file classes accept
        # one argument and others two.
        # with one argument: gzip.GzipFile, bz2.BZ2File
        # with two arguments: zip.ZipFile, read_sas
        ...

    def seekable(self) -> bool:
        # [FUNCTION DOC]
        # Protocol stub: returns whether the buffer supports seeking (required
        # for compressed file classes).
        # for bz2.BZ2File
        ...

    def tell(self) -> int:
        # [FUNCTION DOC]
        # Protocol stub: returns the current absolute position as an integer.
        # for zip.ZipFile, read_stata, to_stata
        ...


class ReadBuffer(BaseBuffer, Protocol[AnyStr_co]):
    # [CLASS DOC]
    # Protocol for a readable buffer, generic over text or bytes (covariant).
    # Extends the base buffer protocol with a read method. Reported module is the
    # public typing aliases module.
    __module__: str = "pandas.api.typing.aliases"

    def read(self, n: int = ..., /) -> AnyStr_co:
        # [FUNCTION DOC]
        # Protocol stub: reads up to n units (optional, positional only) and
        # returns text or bytes according to the type parameter.
        # for BytesIOWrapper, gzip.GzipFile, bz2.BZ2File
        ...


class WriteBuffer(BaseBuffer, Protocol[AnyStr_contra]):
    # [CLASS DOC]
    # Protocol for a writable buffer, generic over text or bytes
    # (contravariant). Extends the base buffer protocol with write and flush.
    # Reported module is the public typing aliases module.
    __module__: str = "pandas.api.typing.aliases"

    def write(self, b: AnyStr_contra, /) -> Any:
        # [FUNCTION DOC]
        # Protocol stub: writes the given text or bytes (positional only); the
        # return value is unconstrained.
        # for gzip.GzipFile, bz2.BZ2File
        ...

    def flush(self) -> Any:
        # [FUNCTION DOC]
        # Protocol stub: flushes buffered output; the return value is
        # unconstrained.
        # for gzip.GzipFile, bz2.BZ2File
        ...


class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    # [CLASS DOC]
    # Protocol for a binary readable buffer that also supports reading a line,
    # as required by the unpickler. Reported module is the public typing aliases
    # module.
    __module__: str = "pandas.api.typing.aliases"

    def readline(self) -> bytes:
        # [FUNCTION DOC]
        # Protocol stub: reads and returns one line of bytes.
        ...


class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    # [CLASS DOC]
    # Protocol for a binary writable buffer that also supports truncation, as
    # required when writing spreadsheet files. Reported module is the public
    # typing aliases module.
    __module__: str = "pandas.api.typing.aliases"

    def truncate(self, size: int | None = ..., /) -> int:
        # [FUNCTION DOC]
        # Protocol stub: resizes the buffer to the given size (optional,
        # positional only; default the current position) and returns the new
        # size.
        ...


class ReadCsvBuffer(ReadBuffer[AnyStr_co], Protocol):
    # [CLASS DOC]
    # Protocol for a readable buffer sufficient for the delimited-text reader:
    # iteration by line, a file descriptor, line reading and a closed property.
    # Reported module is the public typing aliases module.
    __module__: str = "pandas.api.typing.aliases"

    def __iter__(self) -> Iterator[AnyStr_co]:
        # [FUNCTION DOC]
        # Protocol stub: iterator over lines; used by the pure-python parser
        # engine.
        # for engine=python
        ...

    def fileno(self) -> int:
        # [FUNCTION DOC]
        # Protocol stub: returns the integer file descriptor; used by the
        # memory-mapped wrapper.
        # for _MMapWrapper
        ...

    def readline(self) -> AnyStr_co:
        # [FUNCTION DOC]
        # Protocol stub: reads and returns one line; used by the pure-python
        # parser engine.
        # for engine=python
        ...

    @property
    def closed(self) -> bool:
        # [FUNCTION DOC]
        # Protocol stub (property): whether the buffer has been closed; used by
        # the arrow parser engine.
        # for engine=pyarrow
        ...


FilePath: TypeAlias = str | PathLike[str]

# for arbitrary kwargs passed during reading/writing files
StorageOptions: TypeAlias = dict[str, Any] | None

# compression keywords and compression
CompressionDict: TypeAlias = dict[str, Any]
CompressionOptions: TypeAlias = (
    Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"] | CompressionDict | None
)
ParquetCompressionOptions: TypeAlias = (
    Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
)

# types in DataFrameFormatter
FormattersType: TypeAlias = (
    list[Callable] | tuple[Callable, ...] | Mapping[str | int, Callable]
)
ColspaceType: TypeAlias = Mapping[Hashable, str | int]
FloatFormatType: TypeAlias = Union[str, Callable, "EngFormatter"]
ColspaceArgType: TypeAlias = (
    str | int | Sequence[str | int] | Mapping[Hashable, str | int]
)

# Arguments for fillna()
FillnaOptions: TypeAlias = Literal["backfill", "bfill", "ffill", "pad"]
InterpolateOptions: TypeAlias = Literal[
    "linear",
    "time",
    "index",
    "values",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]

# internals
Manager: TypeAlias = Union["BlockManager", "SingleBlockManager"]

# indexing
# PositionalIndexer -> valid 1D positional indexer, e.g. can pass
# to ndarray.__getitem__
# ScalarIndexer is for a single value as the index
# SequenceIndexer is for list like or slices (but not tuples)
# PositionalIndexerTuple is extends the PositionalIndexer for 2D arrays
# These are used in various __getitem__ overloads
# TODO(typing#684): add Ellipsis, see
# https://github.com/python/typing/issues/684#issuecomment-548203158
# https://bugs.python.org/issue41810
# Using List[int] here rather than Sequence[int] to disallow tuples.
ScalarIndexer: TypeAlias = int | np.integer
SequenceIndexer: TypeAlias = slice | list[int] | np.ndarray
PositionalIndexer: TypeAlias = ScalarIndexer | SequenceIndexer
PositionalIndexerTuple: TypeAlias = tuple[PositionalIndexer, PositionalIndexer]
PositionalIndexer2D: TypeAlias = PositionalIndexer | PositionalIndexerTuple
TakeIndexer: TypeAlias = Sequence[int] | Sequence[np.integer] | npt.NDArray[np.integer]

# Shared by functions such as drop and astype
IgnoreRaise: TypeAlias = Literal["ignore", "raise"]

# Windowing rank methods
WindowingRankType: TypeAlias = Literal["average", "min", "max"]

# read_csv engines
CSVEngine: TypeAlias = Literal["c", "python", "pyarrow", "python-fwf"]

# read_json engines
JSONEngine: TypeAlias = Literal["ujson", "pyarrow"]

# read_xml parsers
XMLParsers: TypeAlias = Literal["lxml", "etree"]

# read_html flavors
HTMLFlavors: TypeAlias = Literal["lxml", "html5lib", "bs4"]

# Interval closed type
IntervalLeftRight: TypeAlias = Literal["left", "right"]
IntervalClosedType: TypeAlias = IntervalLeftRight | Literal["both", "neither"]

# datetime and NaTType
DatetimeNaTType: TypeAlias = Union[datetime, "NaTType"]
DateTimeErrorChoices: TypeAlias = Literal["raise", "coerce"]

# sort_index
SortKind: TypeAlias = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition: TypeAlias = Literal["first", "last"]

# Arguments for nsmallest and nlargest
NsmallestNlargestKeep: TypeAlias = Literal["first", "last", "all"]

# quantile interpolation
QuantileInterpolation: TypeAlias = Literal[
    "linear", "lower", "higher", "midpoint", "nearest"
]

# plotting
PlottingOrientation: TypeAlias = Literal["horizontal", "vertical"]

# dropna
AnyAll: TypeAlias = Literal["any", "all"]

# merge
MergeHow: TypeAlias = Literal[
    "left", "right", "inner", "outer", "cross", "left_anti", "right_anti"
]
MergeValidate: TypeAlias = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]

# join
JoinHow: TypeAlias = Literal["left", "right", "inner", "outer"]
JoinValidate: TypeAlias = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]

# reindex
ReindexMethod: TypeAlias = FillnaOptions | Literal["nearest"]

MatplotlibColor: TypeAlias = str | Sequence[float]
TimeGrouperOrigin: TypeAlias = Union[
    "Timestamp", Literal["epoch", "start", "start_day", "end", "end_day"]
]
TimeAmbiguous: TypeAlias = (
    Literal["infer", "NaT", "raise"] | bool | npt.NDArray[np.bool_]
)
TimeNonexistent: TypeAlias = (
    Literal["shift_forward", "shift_backward", "NaT", "raise"] | timedelta
)

DropKeep: TypeAlias = Literal["first", "last", False]
CorrelationMethod: TypeAlias = (
    Literal["pearson", "kendall", "spearman"]
    | Callable[[np.ndarray, np.ndarray], float]
)

AlignJoin: TypeAlias = Literal["outer", "inner", "left", "right"]
DtypeBackend: TypeAlias = Literal["pyarrow", "numpy_nullable"]

TimeUnit: TypeAlias = Literal["s", "ms", "us", "ns"]
OpenFileErrors: TypeAlias = Literal[
    "strict",
    "ignore",
    "replace",
    "surrogateescape",
    "xmlcharrefreplace",
    "backslashreplace",
    "namereplace",
]

# update
UpdateJoin: TypeAlias = Literal["left"]

# applymap
NaAction: TypeAlias = Literal["ignore"]

# from_dict
FromDictOrient: TypeAlias = Literal["columns", "index", "tight"]

# to_stata
ToStataByteorder: TypeAlias = Literal[">", "<", "little", "big"]

# ExcelWriter
ExcelWriterIfSheetExists: TypeAlias = Literal["error", "new", "replace", "overlay"]
ExcelWriterMergeCells: TypeAlias = bool | Literal["columns"]

# Offsets
OffsetCalendar: TypeAlias = Union[np.busdaycalendar, "AbstractHolidayCalendar"]

# read_csv: usecols
UsecolsArgType: TypeAlias = (
    SequenceNotStr[Hashable] | range | AnyArrayLike | Callable[[HashableT], bool] | None
)

# maintain the sub-type of any hashable sequence
SequenceT = TypeVar("SequenceT", bound=Sequence[Hashable])

SliceType: TypeAlias = Hashable | None


# Arrow PyCapsule Interface
# from https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html#protocol-typehints


class ArrowArrayExportable(Protocol):
    # [CLASS DOC]
    # Protocol for objects that can export their data as an Arrow array through
    # the PyCapsule interface, i.e. that have the array export special method.
    # Taken from the Arrow documentation's type hints. Only for static checking.
    """
    An object with an ``__arrow_c_array__`` method.

    This method indicates the object is an Arrow-compatible object implementing
    the `Arrow PyCapsule Protocol`_ (exposing the `Arrow C Data Interface`_ in
    Python), enabling zero-copy Arrow data interchange across libraries.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
    .. _Arrow C Data Interface: https://arrow.apache.org/docs/format/CDataInterface.html

    """

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]:
        # [FUNCTION DOC]
        # Protocol stub: exports the data. Parameter requested_schema is an
        # optional schema capsule suggesting the desired type (default none).
        # Returns a pair of capsule objects: the schema and the array.
        ...


class ArrowStreamExportable(Protocol):
    # [CLASS DOC]
    # Protocol for objects that can export their data as an Arrow stream through
    # the PyCapsule interface, i.e. that have the stream export special method.
    # Only for static checking.
    # KNOWN ISSUE: the existing docstring refers to a link target named the Arrow
    # C Data Interface, but the only target defined in that docstring is named
    # the Arrow C Stream Interface, so the reference is unresolved in the docs.
    """
    An object with an ``__arrow_c_stream__`` method.

    This method indicates the object is an Arrow-compatible object implementing
    the `Arrow PyCapsule Protocol`_ (exposing the `Arrow C Data Interface`_
    for streams in Python), enabling zero-copy Arrow data interchange across
    libraries.

    .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
    .. _Arrow C Stream Interface: https://arrow.apache.org/docs/format/CStreamInterface.html

    """

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        # [FUNCTION DOC]
        # Protocol stub: exports the data as a stream. Parameter requested_schema
        # is an optional schema capsule suggesting the desired type (default
        # none). Returns a stream capsule object.
        ...


__all__ = ["type_t"]
