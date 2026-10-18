"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""
# [MODULE DOC]
# Vendored definition of the dataframe interchange protocol: enums, typed
# dictionaries and abstract base classes that describe what an exporting
# dataframe library must provide.
# Public contents: an enum of DLPack device codes; an enum of dtype kinds
# (ints, unsigned, floats, then bool, string, datetime, categorical with the
# latter codes starting at twenty); an enum of null representations; two typed
# mappings (the buffers of a column, and the description of a categorical); and
# three abstract classes: Buffer, Column and DataFrame. The pandas
# implementations of these are in the buffer, column and dataframe modules,
# and the consumer side is in from_dataframe. Only the frame class is tagged
# as public (published under the public interchange API namespace).
# The only import-time effect is applying the module-renaming decorator to the
# frame class.
# KNOWN ISSUE: the module docstring calls this a verbatim copy of the upstream specification, but the frame class carries a pandas-specific decorator and the chunk docstrings were adapted, so it is not strictly verbatim.

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
)

from pandas.util._decorators import set_module

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )


class DlpackDeviceType(enum.IntEnum):
    # [CLASS DOC]
    # Integer enum of device type codes matching the DLPack standard: CPU, CUDA,
    # pinned CPU, OpenCL, Vulkan, Metal, VPI, ROCm. Used as the first element of
    # the pair returned by the buffer's device query.
    """Integer enum for device type codes matching DLPack."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


class DtypeKind(enum.IntEnum):
    # [CLASS DOC]
    # Integer enum of protocol data type kinds: signed int (0), unsigned int (1),
    # float (2), bool (20), utf-8 string (21), datetime (22), categorical (23).
    # The values align with DLPack for the first three and leave a gap before the
    # rest for future extensions.
    """
    Integer enum for data types.

    Attributes
    ----------
    INT : int
        Matches to signed integer data type.
    UINT : int
        Matches to unsigned integer data type.
    FLOAT : int
        Matches to floating point data type.
    BOOL : int
        Matches to boolean data type.
    STRING : int
        Matches to string data type (UTF-8 encoded).
    DATETIME : int
        Matches to datetime data type.
    CATEGORICAL : int
        Matches to categorical data type.
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


class ColumnNullType(enum.IntEnum):
    # [CLASS DOC]
    # Integer enum of null representations: non-nullable (0), explicit float
    # not-a-number (1), a sentinel value other than not-a-number (2), a bit mask
    # (3) and a byte mask (4). Returned by the column's null description together
    # with an associated value.
    """
    Integer enum for null type representation.

    Attributes
    ----------
    NON_NULLABLE : int
        Non-nullable column.
    USE_NAN : int
        Use explicit float NaN value.
    USE_SENTINEL : int
        Sentinel value besides NaN/NaT.
    USE_BITMASK : int
        The bit is set/unset representing a null on a certain position.
    USE_BYTEMASK : int
        The byte is set/unset representing a null on a certain position.
    """

    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4


class ColumnBuffers(TypedDict):
    # [CLASS DOC]
    # Typed mapping describing the buffers of one column: a data entry holding a
    # pair of (buffer, dtype tuple); a validity entry that is such a pair or None
    # when the null representation is not a bit or byte mask; an offsets entry that
    # is such a pair or None when the data has no offsets buffer (used for
    # variable-length data like strings). Has no behavior.
    # first element is a buffer containing the column data;
    # second element is the data buffer's associated dtype
    data: tuple[Buffer, Any]

    # first element is a buffer containing mask values indicating missing data;
    # second element is the mask value buffer's associated dtype.
    # None if the null representation is not a bit or byte mask
    validity: tuple[Buffer, Any] | None

    # first element is a buffer containing the offset values for
    # variable-size binary data (e.g., variable-length strings);
    # second element is the offsets buffer's associated dtype.
    # None if the data buffer does not have an associated offsets buffer
    offsets: tuple[Buffer, Any] | None


class CategoricalDescription(TypedDict):
    # [CLASS DOC]
    # Typed mapping describing a categorical column: a flag telling whether the
    # ordering of the codes is meaningful, a flag telling whether a dictionary-style
    # mapping from codes to values exists, and the column of categories (None if
    # not dictionary-style). Has no behavior.
    # whether the ordering of dictionary indices is semantically meaningful
    is_ordered: bool
    # whether a dictionary-style mapping of categorical values to other objects exists
    is_dictionary: bool
    # Python-level only (e.g. ``{int: str}``).
    # None if not a dictionary-style categorical.
    categories: Column | None


class Buffer(ABC):
    # [CLASS DOC]
    # Abstract base for a contiguous block of memory exchanged between libraries.
    # Subclasses must provide size in bytes, start address, a DLPack export and the
    # device. A buffer has no dtype; dtype information comes from the owning
    # column (or from the DLPack export where supported).
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.
    """

    @property
    @abstractmethod
    def bufsize(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: buffer size in bytes (integer).
        """
        Buffer size in bytes.
        """

    @property
    @abstractmethod
    def ptr(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: address of the first byte of the buffer as an integer.
        """
        Pointer to start of the buffer as an integer.
        """

    @abstractmethod
    def __dlpack__(self):
        # [FUNCTION DOC]
        # Abstract method: produce a DLPack capsule for the buffer. Subclasses may
        # raise a type error for unsupported dtypes or a not-implemented error when
        # DLPack is unsupported. The base body raises a not-implemented error.
        """
        Produce DLPack capsule (see array API standard).

        Raises:

            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented

        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        raise NotImplementedError("__dlpack__")

    @abstractmethod
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        # [FUNCTION DOC]
        # Abstract method: return a pair of (device type code, optional device id)
        # describing where the data resides. Must be implemented even when the DLPack
        # export is not.
        """
        Device type and device ID for where the data in the buffer resides.
        Uses device type codes matching DLPack.
        Note: must be implemented even if ``__dlpack__`` is not.
        """


class Column(ABC):
    # [CLASS DOC]
    # Abstract base for one column of an interchange frame. A column may consist of
    # one or more chunks and each chunk can have up to three buffers: data, a mask
    # (depending on the null representation) and offsets (for variable-size
    # binary data). Subclasses supply size, offset, dtype, categorical and null
    # descriptions, null count, metadata, chunk handling and the buffers. The
    # column has no versioning of its own because it is only obtained from a
    # frame export.
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.

         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.

         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.

    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.

         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

    @abstractmethod
    def size(self) -> int:
        # [FUNCTION DOC]
        # Abstract method: number of elements of the column (or of this chunk if the
        # column is split).
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.
        """

    @property
    @abstractmethod
    def offset(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: offset of the first element; may be positive when chunks
        # are used (chunk number times the common chunk length).
        """
        Offset of first element.

        May be > 0 if using chunks; for example for a column with N chunks of
        equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.
        """

    @property
    @abstractmethod
    def dtype(self) -> tuple[DtypeKind, int, str, str]:
        # [FUNCTION DOC]
        # Abstract property: dtype description as a tuple of (kind, bit width, Arrow C
        # format string, endianness character). Masks must be boolean with width one for
        # bit masks or eight for byte masks; only native endianness is currently
        # supported; complex, Arrow null, binary, decimal and nested types are excluded.
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``.

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes:
            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for bit
              masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the future
              we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary, decimal,
              and nested (list, struct, map, union) dtypes.
        """

    @property
    @abstractmethod
    def describe_categorical(self) -> CategoricalDescription:
        # [FUNCTION DOC]
        # Abstract property: for categorical columns return a categorical description
        # (ordering flag, dictionary flag, categories column); raises a type error if
        # the column is not categorical.
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding for categorical values.

        Raises TypeError if the dtype is not categorical

        Returns the dictionary with description on how to interpret the data buffer:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of indices to
                             category values (e.g. an array of cat1, cat2, ...).
                             None if not a dictionary-style categorical.

        TBD: are there any other in-memory representations that are needed?
        """

    @property
    @abstractmethod
    def describe_null(self) -> tuple[ColumnNullType, Any]:
        # [FUNCTION DOC]
        # Abstract property: return a pair (null kind, value). The value is the
        # sentinel when the kind is sentinel-based, the mask value (zero or one) that
        # marks a missing entry when the kind is a bit or byte mask, and None
        # otherwise.
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Value : if kind is "sentinel value", the actual value. If kind is a bit
        mask or a byte mask, the value (0 or 1) indicating a missing value. None
        otherwise.
        """

    @property
    @abstractmethod
    def null_count(self) -> int | None:
        # [FUNCTION DOC]
        # Abstract property: number of missing elements, or None when unknown.
        """
        Number of null elements, if known.

        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract property: mapping of library-specific metadata for the column; see
        # the frame's metadata for key naming rules.
        """
        The metadata for the column. See `DataFrame.metadata` for more details.
        """

    @abstractmethod
    def num_chunks(self) -> int:
        # [FUNCTION DOC]
        # Abstract method: return the number of chunks of the column.
        """
        Return the number of chunks the column consists of.
        """

    @abstractmethod
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[Column]:
        # [FUNCTION DOC]
        # Abstract method: return an iterable of column chunks. Parameter n_chunks:
        # optional integer; semantics are those of the frame's chunk method.
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """

    @abstractmethod
    def get_buffers(self) -> ColumnBuffers:
        # [FUNCTION DOC]
        # Abstract method: return the buffers mapping (data, validity, offsets) of the
        # column, each entry a pair of buffer and its dtype tuple, or None where
        # there is no such buffer.
        """
        Return a dictionary containing the underlying buffers.

        The returned dictionary has the following contents:

            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """


#    def get_children(self) -> Iterable[Column]:
#        """
#        Children columns underneath the column, each object in this iterator
#        must adhere to the column specification.
#        """
#        pass


@set_module("pandas.api.interchange")
class DataFrame(ABC):
    # [CLASS DOC]
    # Abstract base for an interchange frame: an ordered collection of columns with
    # unique string names, accessible by name or by position. Class attribute
    # version (integer, zero) gives the protocol version. Subclasses supply
    # metadata, counts, column access and selection, and chunking. Published in
    # the public interchange namespace.
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string.
    Columns may be accessed by name or by position.

    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    version = 0  # version of the protocol

    @abstractmethod
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        # [FUNCTION DOC]
        # Abstract method: construct a new interchange object, possibly with changed
        # parameters. Parameters: nan_as_null (default false) and allow_copy (default
        # true).
        """Construct a new interchange object, potentially changing the parameters."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract property: mapping with string keys of library-specific metadata for
        # roundtripping; keys should be prefixed with the library name and a period.
        """
        The metadata for the data frame, as a dictionary with string keys. The
        contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.
        """

    @abstractmethod
    def num_columns(self) -> int:
        # [FUNCTION DOC]
        # Abstract method: number of columns.
        """
        Return the number of columns in the DataFrame.
        """

    @abstractmethod
    def num_rows(self) -> int | None:
        # [FUNCTION DOC]
        # Abstract method: number of rows, or None if not available (it may be
        # expensive to compute).
        # TODO: not happy with Optional, but need to flag it may be expensive
        #       why include it if it may be None - what do we expect consumers
        #       to do here?
        """
        Return the number of rows in the DataFrame, if available.
        """

    @abstractmethod
    def num_chunks(self) -> int:
        # [FUNCTION DOC]
        # Abstract method: number of chunks of the frame.
        """
        Return the number of chunks the DataFrame consists of.
        """

    @abstractmethod
    def column_names(self) -> Iterable[str]:
        # [FUNCTION DOC]
        # Abstract method: iterable of the column names.
        """
        Return an iterator yielding the column names.
        """

    @abstractmethod
    def get_column(self, i: int) -> Column:
        # [FUNCTION DOC]
        # Abstract method: return the column at integer position i.
        """
        Return the column at the indicated position.
        """

    @abstractmethod
    def get_column_by_name(self, name: str) -> Column:
        # [FUNCTION DOC]
        # Abstract method: return the column with the given name.
        """
        Return the column whose name is the indicated name.
        """

    @abstractmethod
    def get_columns(self) -> Iterable[Column]:
        # [FUNCTION DOC]
        # Abstract method: iterable of all columns in order.
        """
        Return an iterator yielding the columns.
        """

    @abstractmethod
    def select_columns(self, indices: Sequence[int]) -> DataFrame:
        # [FUNCTION DOC]
        # Abstract method: return a new frame holding the columns at the given
        # sequence of integer positions.
        """
        Create a new DataFrame by selecting a subset of columns by index.
        """

    @abstractmethod
    def select_columns_by_name(self, names: Sequence[str]) -> DataFrame:
        # [FUNCTION DOC]
        # Abstract method: return a new frame holding the columns with the given
        # sequence of names.
        """
        Create a new DataFrame by selecting a subset of columns by name.
        """

    @abstractmethod
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[DataFrame]:
        # [FUNCTION DOC]
        # Abstract method: iterable of frame chunks. Parameter n_chunks (optional
        # integer): None yields the producer's own chunks; if given it must be a
        # multiple of the number of chunks and the producer subdivides each chunk.
        """
        Return an iterator yielding the chunks.

        By default (None), yields the chunks that the data is stored as by the
        producer. If given, ``n_chunks`` must be a multiple of
        ``self.num_chunks()``, meaning the producer must subdivide each chunk
        before yielding it.
        """
