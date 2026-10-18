# [MODULE DOC]
# Pandas implementation of the interchange protocol's column object.
# Public content: the column wrapper class (subclass of the protocol column).
# Private module constants: a mapping from numpy dtype kind letters to protocol
# dtype kinds (ints, unsigned, floats, bools, unicode strings, and both
# datetime and timedelta kinds map to the datetime kind); a mapping from
# protocol dtype kind to a pair of (null representation, sentinel/mask value)
# for non-extension data (floats use not-a-number, datetimes use the integer
# minimum sentinel, ints/uints/bools are non-nullable, categoricals use a
# sentinel of minus one, strings use a byte mask whose valid value is zero);
# and a mapping from null representation to the message explaining why there
# is no validity buffer. Depends on the buffer, protocol and utils modules.
# No import-time side effects.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._config import using_python_scalars

from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.dtypes import BaseMaskedDtype

import pandas as pd
from pandas import (
    ArrowDtype,
    DatetimeTZDtype,
)
from pandas.api.types import is_string_dtype
from pandas.core.interchange.buffer import (
    PandasBuffer,
    PandasBufferPyarrow,
)
from pandas.core.interchange.dataframe_protocol import (
    Column,
    ColumnBuffers,
    ColumnNullType,
    DtypeKind,
)
from pandas.core.interchange.utils import (
    ArrowCTypes,
    Endianness,
    dtype_to_arrow_c_fmt,
)

if TYPE_CHECKING:
    from pandas.core.interchange.dataframe_protocol import Buffer

_NP_KINDS = {
    "i": DtypeKind.INT,
    "u": DtypeKind.UINT,
    "f": DtypeKind.FLOAT,
    "b": DtypeKind.BOOL,
    "U": DtypeKind.STRING,
    "M": DtypeKind.DATETIME,
    "m": DtypeKind.DATETIME,
}

_NULL_DESCRIPTION = {
    DtypeKind.FLOAT: (ColumnNullType.USE_NAN, None),
    DtypeKind.DATETIME: (ColumnNullType.USE_SENTINEL, iNaT),
    DtypeKind.INT: (ColumnNullType.NON_NULLABLE, None),
    DtypeKind.UINT: (ColumnNullType.NON_NULLABLE, None),
    DtypeKind.BOOL: (ColumnNullType.NON_NULLABLE, None),
    # Null values for categoricals are stored as `-1` sentinel values
    # in the category date (e.g., `col.values.codes` is int8 np.ndarray)
    DtypeKind.CATEGORICAL: (ColumnNullType.USE_SENTINEL, -1),
    # follow Arrow in using 1 as valid value and 0 for missing/null value
    DtypeKind.STRING: (ColumnNullType.USE_BYTEMASK, 0),
}

_NO_VALIDITY_BUFFER = {
    ColumnNullType.NON_NULLABLE: "This column is non-nullable",
    ColumnNullType.USE_NAN: "This column uses NaN as null",
    ColumnNullType.USE_SENTINEL: "This column uses a sentinel value",
}


class PandasColumn(Column):
    # [CLASS DOC]
    # Interchange-protocol view of one pandas series; subclass of the protocol
    # column. State: a private series and a private flag stating whether copies
    # are allowed. A column is always a single chunk unless split by get_chunks,
    # and can expose up to three buffers: data, validity mask and offsets (for
    # strings). Only creatable from the frame wrapper. Key methods: dtype
    # (protocol dtype tuple), describe_null, describe_categorical, null_count,
    # get_buffers and the private buffer builders for data, validity and offsets.
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.
    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).
    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(self, column: pd.Series, allow_copy: bool = True) -> None:
        # [FUNCTION DOC]
        # Wrap a series. Parameters: column (a Series); allow_copy (default true).
        # Raises a type error with an explanatory message if a frame is passed
        # (typically because duplicated column names after text conversion made
        # column selection return a frame), and a not-implemented error for any other
        # non-series type. Stores the series and the flag; returns nothing.
        # KNOWN ISSUE: the existing docstring says extension arrays are not handled, but the other methods of this class explicitly handle masked and Arrow-backed extension dtypes.
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        if isinstance(column, pd.DataFrame):
            raise TypeError(
                "Expected a Series, got a DataFrame. This likely happened "
                "because you called __dataframe__ on a DataFrame which, "
                "after converting column names to string, resulted in duplicated "
                f"names: {column.columns}. Please rename these columns before "
                "using the interchange protocol."
            )
        if not isinstance(column, pd.Series):
            raise NotImplementedError(f"Columns of type {type(column)} not handled yet")

        # Store the column as a private attribute
        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        # [FUNCTION DOC]
        # Return the number of elements in the column (the series size).
        """
        Size of the column, in elements.
        """
        return self._col.size

    @property
    def offset(self) -> int:
        # [FUNCTION DOC]
        # Property: offset of the first element, always zero.
        """
        Offset of first element. Always zero.
        """
        # TODO: chunks are implemented now, probably this should return something
        return 0

    @cache_readonly
    def dtype(self) -> tuple[DtypeKind, int, str, str]:
        # [FUNCTION DOC]
        # Cached property: the protocol dtype tuple of (kind, bit width, Arrow C format
        # string, byte order).
        # Categorical series report the categorical kind with the bit width, format
        # string of the integer codes dtype and native byte order. String-like dtypes
        # report the string kind with width 8, the format string for the dtype and
        # native byte order, but only if the inferred content type is string or empty;
        # otherwise a not-implemented error says non-string object dtypes are
        # unsupported. Every other dtype is delegated to the private dtype
        # converter.
        dtype = self._col.dtype

        if isinstance(dtype, pd.CategoricalDtype):
            codes = self._col.values.codes
            (
                _,
                bitwidth,
                c_arrow_dtype_f_str,
                _,
            ) = self._dtype_from_pandasdtype(codes.dtype)
            return (
                DtypeKind.CATEGORICAL,
                bitwidth,
                c_arrow_dtype_f_str,
                Endianness.NATIVE,
            )
        elif is_string_dtype(dtype):
            if infer_dtype(self._col) in ("string", "empty"):
                return (
                    DtypeKind.STRING,
                    8,
                    dtype_to_arrow_c_fmt(dtype),
                    Endianness.NATIVE,
                )
            raise NotImplementedError("Non-string object dtypes are not supported yet")
        else:
            return self._dtype_from_pandasdtype(dtype)

    def _dtype_from_pandasdtype(self, dtype) -> tuple[DtypeKind, int, str, str]:
        # [FUNCTION DOC]
        # HELPER: converts a numpy or extension dtype to the protocol dtype tuple used
        # by the dtype property. Parameter dtype: the dtype.
        # Looks up the protocol kind from the numpy kind letter (a missing kind raises
        # a value error saying the type is unsupported; complex, bytes and void kinds
        # are therefore unsupported). The byte order comes from the numpy dtype
        # (for Arrow, masked and tz-aware dtypes from the underlying numpy or base
        # dtype). The pyarrow boolean dtype returns early with its unscaled item size
        # (because it is a bitmask) and the boolean format. All other dtypes return
        # the kind, the item size times eight bits, the Arrow C format string from the
        # utils function and the byte order.
        # KNOWN ISSUE: a comment in the unsupported-kind branch says to check whether it is categorical, but categoricals are handled by the caller; the comment is stale.
        """
        See `self.dtype` for details.
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void) not handled
        #       datetime and timedelta both map to datetime (is timedelta handled?)

        kind = _NP_KINDS.get(dtype.kind, None)
        if kind is None:
            # Not a NumPy dtype. Check if it's a categorical maybe
            raise ValueError(f"Data type {dtype} not supported by interchange protocol")
        if isinstance(dtype, ArrowDtype):
            byteorder = dtype.numpy_dtype.byteorder
        elif isinstance(dtype, DatetimeTZDtype):
            byteorder = dtype.base.byteorder  # type: ignore[union-attr]
        elif isinstance(dtype, BaseMaskedDtype):
            byteorder = dtype.numpy_dtype.byteorder
        else:
            byteorder = dtype.byteorder

        if dtype == "bool[pyarrow]":
            # return early to avoid the `* 8` below, as this is a bitmask
            # rather than a bytemask
            return (
                kind,
                dtype.itemsize,  # pyright: ignore[reportAttributeAccessIssue]
                ArrowCTypes.BOOL,
                byteorder,
            )

        return kind, dtype.itemsize * 8, dtype_to_arrow_c_fmt(dtype), byteorder

    @property
    def describe_categorical(self):
        # [FUNCTION DOC]
        # Property describing a categorical column. Raises a type error if the
        # column's protocol kind is not categorical. Returns a mapping with three
        # entries: whether the categories are ordered (from the series), a
        # dictionary flag that is always true, and a column wrapper over a new series
        # of the categories (created with the default copy flag).
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding for categorical values.

        Raises TypeError if the dtype is not categorical

        Content of returned dict:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a dictionary-style mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of indices to
                             category values (e.g. an array of cat1, cat2, ...).
                             None if not a dictionary-style categorical.
        """
        if not self.dtype[0] == DtypeKind.CATEGORICAL:
            raise TypeError(
                "describe_categorical only works on a column with categorical dtype!"
            )

        return {
            "is_ordered": self._col.cat.ordered,
            "is_dictionary": True,
            "categories": PandasColumn(pd.Series(self._col.cat.categories)),
        }

    @property
    def describe_null(self):
        # [FUNCTION DOC]
        # Property returning a pair (null representation, value) describing how
        # missing values are encoded.
        # Masked extension dtypes give a byte mask with null marker one. Arrow-backed
        # dtypes (assumed to be single-chunk) give non-nullable with no value when the
        # array has no validity buffer, otherwise a bit mask with marker zero. All
        # others look up the protocol kind in the private null mapping; an unknown kind
        # raises a not-implemented error that names the kind.
        if isinstance(self._col.dtype, BaseMaskedDtype):
            column_null_dtype = ColumnNullType.USE_BYTEMASK
            null_value = 1
            return column_null_dtype, null_value
        if isinstance(self._col.dtype, ArrowDtype):
            # We already rechunk (if necessary / allowed) upon initialization, so this
            # is already single-chunk by the time we get here.
            if self._col.array._pa_array.chunks[0].buffers()[0] is None:  # type: ignore[attr-defined]
                return ColumnNullType.NON_NULLABLE, None
            return ColumnNullType.USE_BITMASK, 0
        kind = self.dtype[0]
        try:
            null, value = _NULL_DESCRIPTION[kind]
        except KeyError as err:
            raise NotImplementedError(f"Data type {kind} not yet supported") from err

        return null, value

    @cache_readonly
    def null_count(self) -> int:
        # [FUNCTION DOC]
        # Cached property: number of missing elements, computed from the series' null
        # mask sum. The result is converted to a plain scalar unless the global option
        # for python scalars is on.
        """
        Number of null elements. Should always be known.
        """
        result = self._col.isna().sum()
        if not using_python_scalars():
            result = result.item()
        return result

    @property
    def metadata(self) -> dict[str, pd.Index]:
        # [FUNCTION DOC]
        # Property: library-specific metadata, a mapping with one entry keyed by the
        # pandas index key whose value is the series' row index.
        """
        Store specific metadata of the column.
        """
        return {"pandas.index": self._col.index}

    def num_chunks(self) -> int:
        # [FUNCTION DOC]
        # Return the number of chunks; always 1.
        """
        Return the number of chunks the column consists of.
        """
        return 1

    def get_chunks(self, n_chunks: int | None = None):
        # [FUNCTION DOC]
        # Generator yielding column wrappers for consecutive row slices.
        # Parameter n_chunks: optional integer. When it is greater than 1 the rows are
        # split into n_chunks pieces whose length is the ceiling of size divided by
        # n_chunks; each piece is a positional slice wrapped with this object's copy
        # flag. Otherwise yields this object once.
        # KNOWN ISSUE: for an empty column and n_chunks above 1 the chunk length is zero and the range used for iteration has a zero step, which raises a value error.
        """
        Return an iterator yielding the chunks.
        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        if n_chunks and n_chunks > 1:
            size = len(self._col)
            step = size // n_chunks
            if size % n_chunks != 0:
                step += 1
            for start in range(0, step * n_chunks, step):
                yield PandasColumn(
                    self._col.iloc[start : start + step], self._allow_copy
                )
        else:
            yield self

    def get_buffers(self) -> ColumnBuffers:
        # [FUNCTION DOC]
        # Return a mapping with keys for data, validity and offsets, each a pair of
        # (buffer, dtype tuple) or None. The data entry is always built (errors from
        # the data builder propagate). Validity and offsets are attempted and set to
        # None if the corresponding builder signals that no such buffer is present.
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
        buffers: ColumnBuffers = {
            "data": self._get_data_buffer(),
            "validity": None,
            "offsets": None,
        }

        try:
            buffers["validity"] = self._get_validity_buffer()
        except NoBufferPresent:
            pass

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except NoBufferPresent:
            pass

        return buffers

    def _get_data_buffer(
        self,
    ) -> tuple[Buffer, tuple[DtypeKind, int, str, str]]:
        # [FUNCTION DOC]
        # HELPER: builds the data buffer and its dtype tuple for get_buffers.
        # Datetime kind: converts to a numpy array (after dropping the timezone to
        # UTC when the format string is longer than four characters, i.e. includes a
        # timezone) and wraps it in a numpy buffer using the copy flag; the reported
        # dtype is a native-endian signed 64-bit integer.
        # Integer, unsigned, float and bool kinds: reports the column's own dtype; for
        # Arrow-backed series (assumed single chunk) wraps the first chunk's data
        # buffer in a pyarrow buffer and returns immediately; for masked dtypes uses
        # the raw data array; otherwise the backing numpy array; wrapped with the copy
        # flag.
        # Categorical: wraps the integer codes, dtype taken from the codes dtype.
        # String: concatenates the utf-8 encodings of all str values (missing or non-str
        # entries contribute nothing) into one unsigned 8-bit buffer (default copy
        # flag) with a native-endian dtype. Raises a not-implemented error for other
        # kinds.
        # KNOWN ISSUE: a comment says the dtype reported for the string data buffer will need correcting (see the upstream issue about it) and that the encoding loop is slow.
        """
        Return the buffer containing the data and the buffer's associated dtype.
        """
        buffer: Buffer
        if self.dtype[0] == DtypeKind.DATETIME:
            # self.dtype[2] is an ArrowCTypes.TIMESTAMP where the tz will make
            # it longer than 4 characters
            if len(self.dtype[2]) > 4:
                np_arr = self._col.dt.tz_convert(None).to_numpy()
            else:
                np_arr = self._col.to_numpy()
            buffer = PandasBuffer(np_arr, allow_copy=self._allow_copy)
            dtype = (
                DtypeKind.INT,
                64,
                ArrowCTypes.INT64,
                Endianness.NATIVE,
            )
        elif self.dtype[0] in (
            DtypeKind.INT,
            DtypeKind.UINT,
            DtypeKind.FLOAT,
            DtypeKind.BOOL,
        ):
            dtype = self.dtype
            arr = self._col.array
            if isinstance(self._col.dtype, ArrowDtype):
                # We already rechunk (if necessary / allowed) upon initialization, so
                # this is already single-chunk by the time we get here.
                arr = arr._pa_array.chunks[0]  # type: ignore[attr-defined]
                buffer = PandasBufferPyarrow(
                    arr.buffers()[1],
                    length=len(arr),
                )
                return buffer, dtype
            if isinstance(self._col.dtype, BaseMaskedDtype):
                np_arr = arr._data  # type: ignore[attr-defined]
            else:
                np_arr = arr._ndarray  # type: ignore[attr-defined]
            buffer = PandasBuffer(np_arr, allow_copy=self._allow_copy)
        elif self.dtype[0] == DtypeKind.CATEGORICAL:
            codes = self._col.values._codes
            buffer = PandasBuffer(codes, allow_copy=self._allow_copy)
            dtype = self._dtype_from_pandasdtype(codes.dtype)
        elif self.dtype[0] == DtypeKind.STRING:
            # Marshal the strings from a NumPy object array into a byte array
            buf = self._col.to_numpy()
            b = bytearray()

            # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
            for obj in buf:
                if isinstance(obj, str):
                    b.extend(obj.encode(encoding="utf-8"))

            # Convert the byte array to a Pandas "buffer" using
            # a NumPy array as the backing store
            buffer = PandasBuffer(np.frombuffer(b, dtype="uint8"))

            # Define the dtype for the returned buffer
            # TODO: this will need correcting
            # https://github.com/pandas-dev/pandas/issues/54781
            dtype = (
                DtypeKind.UINT,
                8,
                ArrowCTypes.UINT8,
                Endianness.NATIVE,
            )  # note: currently only support native endianness
        else:
            raise NotImplementedError(f"Data type {self._col.dtype} not handled yet")

        return buffer, dtype

    def _get_validity_buffer(self) -> tuple[Buffer, Any] | None:
        # [FUNCTION DOC]
        # HELPER: builds the validity buffer and dtype for get_buffers.
        # Returns a pair (buffer, dtype) or None. Raises a no-buffer-present error when
        # the null representation is non-nullable, nan-based or sentinel-based (the
        # message says why), and a not-implemented error for a null representation
        # that is neither of those nor handled below.
        # Arrow-backed series: dtype is a one-bit boolean; returns None when the first
        # chunk has no validity buffer, otherwise wraps that bitmap in a pyarrow
        # buffer. Masked dtypes: wraps the extension array's boolean mask (default copy
        # flag) with an 8-bit boolean dtype. Strings: builds a boolean array with one
        # entry per element that holds the valid encoding for str entries and the
        # invalid encoding otherwise, where the valid encoding is whether the null
        # value from describe_null equals zero.
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte mask.
        """
        null, invalid = self.describe_null
        buffer: Buffer
        if isinstance(self._col.dtype, ArrowDtype):
            # We already rechunk (if necessary / allowed) upon initialization, so this
            # is already single-chunk by the time we get here.
            arr = self._col.array._pa_array.chunks[0]  # type: ignore[attr-defined]
            dtype = (DtypeKind.BOOL, 1, ArrowCTypes.BOOL, Endianness.NATIVE)
            if arr.buffers()[0] is None:
                return None
            buffer = PandasBufferPyarrow(
                arr.buffers()[0],
                length=len(arr),
            )
            return buffer, dtype

        if isinstance(self._col.dtype, BaseMaskedDtype):
            mask = self._col.array._mask  # type: ignore[attr-defined]
            buffer = PandasBuffer(mask)
            dtype = (DtypeKind.BOOL, 8, ArrowCTypes.BOOL, Endianness.NATIVE)
            return buffer, dtype

        if self.dtype[0] == DtypeKind.STRING:
            # For now, use byte array as the mask.
            # TODO: maybe store as bit array to save space?..
            buf = self._col.to_numpy()

            # Determine the encoding for valid values
            valid = invalid == 0
            invalid = not valid

            mask = np.zeros(shape=(len(buf),), dtype=np.bool_)
            for i, obj in enumerate(buf):
                mask[i] = valid if isinstance(obj, str) else invalid

            # Convert the mask array to a Pandas "buffer" using
            # a NumPy array as the backing store
            buffer = PandasBuffer(mask)

            # Define the dtype of the returned buffer
            dtype = (DtypeKind.BOOL, 8, ArrowCTypes.BOOL, Endianness.NATIVE)

            return buffer, dtype

        try:
            msg = f"{_NO_VALIDITY_BUFFER[null]} so does not have a separate mask"
        except KeyError as err:
            # TODO: implement for other bit/byte masks?
            raise NotImplementedError("See self.describe_null") from err

        raise NoBufferPresent(msg)

    def _get_offsets_buffer(self) -> tuple[PandasBuffer, Any]:
        # [FUNCTION DOC]
        # HELPER: builds the offsets buffer for variable-size data.
        # For string kind returns (buffer, dtype) where the buffer is a 64-bit integer
        # array of length n plus one holding cumulative utf-8 byte lengths of the
        # values, missing and non-str entries adding zero length, and dtype is native
        # signed 64-bit integer. For any other kind raises the no-buffer-present error
        # saying fixed-length dtypes have no offsets.
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
        if self.dtype[0] == DtypeKind.STRING:
            # For each string, we need to manually determine the next offset
            values = self._col.to_numpy()
            ptr = 0
            offsets = np.zeros(shape=(len(values) + 1,), dtype=np.int64)
            for i, v in enumerate(values):
                # For missing values (in this case, `np.nan` values)
                # we don't increment the pointer
                if isinstance(v, str):
                    b = v.encode(encoding="utf-8")
                    ptr += len(b)

                offsets[i + 1] = ptr

            # Convert the offsets to a Pandas "buffer" using
            # the NumPy array as the backing store
            buffer = PandasBuffer(offsets)

            # Assemble the buffer dtype info
            dtype = (
                DtypeKind.INT,
                64,
                ArrowCTypes.INT64,
                Endianness.NATIVE,
            )  # note: currently only support native endianness
        else:
            raise NoBufferPresent(
                "This column has a fixed-length dtype so "
                "it does not have an offsets buffer"
            )

        return buffer, dtype
