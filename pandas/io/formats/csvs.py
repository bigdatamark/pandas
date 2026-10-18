"""
Module for formatting output data into CSV files.
"""

# [MODULE DOC]
# Formatting and writing of frames to CSV text for the frame and series CSV writers.
# Public contents: CSVFormatter, which wraps an already configured frame formatter
#   object and writes its data through the standard csv writer, plus the private
#   constant giving the target number of cells per written chunk (one hundred thousand).
#   The actual row serialization of each chunk is delegated to the compiled writers
#   extension.
# Relationships: consumes the frame formatter class from the format module (for the
#   frame and the number formatting options); opens files through the common handle
#   opener, which handles compression, encoding and storage options.
# Import-time side effects: none.

from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
import csv as csvlib
import os
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import notna

from pandas.core.indexes.api import Index

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        FloatFormatType,
        IndexLabel,
        StorageOptions,
        WriteBuffer,
        npt,
    )

    from pandas.io.formats.format import DataFrameFormatter


_DEFAULT_CHUNKSIZE_CELLS = 100_000


class CSVFormatter:
    # [CLASS DOC]
    # Writes a frame to a CSV target. Constructed from a frame formatter and the CSV
    #   options; construction normalizes options and does not write anything. Call save
    #   to open the target and write header and body.
    # State: fmt (the frame formatter), obj (the frame to write; replaced by a column
    #   subset when cols is given), filepath_or_buffer, encoding, compression, mode,
    #   storage_options, sep, index_label (normalized list, or false when index labels
    #   are suppressed), errors, quoting (falsy becomes minimal),
    #   doublequote, escapechar, quotechar (reset to none when quoting is none and no
    #   escape character is set, to avoid a crash in the csv module), lineterminator
    #   (defaults to the operating system line separator), date_format, cols (array of
    #   column labels already formatted as strings), chunksize (rows per chunk) and
    #   writer (the csv writer, created during save).
    # Key members: properties exposing formatter options (na_rep, float_format, decimal,
    #   header, index), save, and the private header and body writers.
    cols: npt.NDArray[np.object_]

    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] = "",
        sep: str = ",",
        cols: Sequence[Hashable] | None = None,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        encoding: str | None = None,
        errors: str = "strict",
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        lineterminator: str | None = "\n",
        chunksize: int | None = None,
        quotechar: str | None = '"',
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Initialize the CSV writer configuration. Inputs: formatter (frame formatter
        #   providing the frame and the number formatting options), path_or_buf (path or
        #   writable buffer; default empty string), sep (field delimiter, default
        #   comma), cols (optional sequence of column labels to write), index_label
        #   (optional label(s) for the index; none derives from the index names, false
        #   suppresses), mode (file mode, default write), encoding, errors (default
        #   strict), compression (default infer), quoting (csv quoting constant; a falsy
        #   value means minimal quoting), lineterminator (default newline; a falsy value
        #   means the operating system separator), chunksize (rows per chunk; none
        #   derives from the column count), quotechar (default double quote),
        #   date_format (format string for dates), doublequote (default true),
        #   escapechar, storage_options.
        # Raises a type error through the column initializer when cols is given and the
        #   frame has multi level columns. No output is produced.
        self.fmt = formatter

        self.obj = self.fmt.frame

        self.filepath_or_buffer = path_or_buf
        self.encoding = encoding
        self.compression: CompressionOptions = compression
        self.mode = mode
        self.storage_options = storage_options

        self.sep = sep
        self.index_label = self._initialize_index_label(index_label)
        self.errors = errors
        self.quoting = quoting or csvlib.QUOTE_MINIMAL
        self.doublequote = doublequote
        self.escapechar = escapechar
        self.quotechar = self._initialize_quotechar(quotechar)
        self.lineterminator = lineterminator or os.linesep
        self.date_format = date_format
        self.cols = self._initialize_columns(cols)
        self.chunksize = self._initialize_chunksize(chunksize)

    @property
    def na_rep(self) -> str:
        # [FUNCTION DOC]
        # Read only property: the string used for missing values, taken from the
        #   formatter.
        return self.fmt.na_rep

    @property
    def float_format(self) -> FloatFormatType | None:
        # [FUNCTION DOC]
        # Read only property: the float format option (or none), taken from the
        #   formatter.
        return self.fmt.float_format

    @property
    def decimal(self) -> str:
        # [FUNCTION DOC]
        # Read only property: the decimal separator string, taken from the formatter.
        return self.fmt.decimal

    @property
    def header(self) -> bool | SequenceNotStr[str]:
        # [FUNCTION DOC]
        # Read only property: the header option from the formatter; a boolean, or a
        #   sequence of column aliases.
        return self.fmt.header

    @property
    def index(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: whether to write the index, taken from the formatter.
        return self.fmt.index

    def _initialize_index_label(self, index_label: IndexLabel | None) -> IndexLabel:
        # [FUNCTION DOC]
        # HELPER: normalizes the index label option. If the option is false it is
        #   returned unchanged. If it is none, the labels are derived from the frame's
        #   index. If it is a single non sequence value, it is wrapped in a one element
        #   list. Sequences and index objects are returned as they are.
        if index_label is not False:
            if index_label is None:
                return self._get_index_label_from_obj()
            elif not isinstance(index_label, (list, tuple, np.ndarray, ABCIndex)):
                # given a string for a DF with Index
                return [index_label]
        return index_label

    def _get_index_label_from_obj(self) -> Sequence[Hashable]:
        # [FUNCTION DOC]
        # HELPER: derives index labels from the frame's index, using the multi level
        #   rule for multi indexes and the flat rule otherwise. Returns a sequence of
        #   labels.
        if isinstance(self.obj.index, ABCMultiIndex):
            return self._get_index_label_multiindex()
        else:
            return self._get_index_label_flat()

    def _get_index_label_multiindex(self) -> Sequence[Hashable]:
        # [FUNCTION DOC]
        # HELPER: returns the list of level names of a multi level index with unnamed
        #   (falsy) names replaced by the empty string.
        return [name or "" for name in self.obj.index.names]

    def _get_index_label_flat(self) -> Sequence[Hashable]:
        # [FUNCTION DOC]
        # HELPER: returns a one element list with the index name, or the empty string if
        #   the index has no name.
        index_label = self.obj.index.name
        return [""] if index_label is None else [index_label]

    def _initialize_quotechar(self, quotechar: str | None) -> str | None:
        # [FUNCTION DOC]
        # HELPER: decides the effective quote character. Returns the given character
        #   unless quoting is disabled and no escape character is set, in which case
        #   returns none (passing a quote character then would crash the csv module).
        if self.quoting != csvlib.QUOTE_NONE or self.escapechar is not None:
            # prevents crash in _csv
            return quotechar
        return None

    @property
    def has_mi_columns(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: true when the frame's columns are a multi level index.
        return bool(isinstance(self.obj.columns, ABCMultiIndex))

    def _initialize_columns(
        self, cols: Iterable[Hashable] | None
    ) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER: determines the array of column labels to write. Input: cols, an
        #   optional iterable of labels. Raises a type error if cols is given while the
        #   frame has multi level columns. If cols is given, converts it to a list (an
        #   index object is converted to formatted text values first) and replaces the
        #   working frame by the column subset in that order. Returns the frame's column
        #   labels formatted for CSV output (taking the number formatting options into
        #   account) as an object array, which keeps duplicate columns.
        # Side effects: may replace the stored frame by a column subset.
        # validate mi options
        if self.has_mi_columns:
            if cols is not None:
                msg = "cannot specify cols with a MultiIndex on the columns"
                raise TypeError(msg)

        if cols is not None:
            if isinstance(cols, ABCIndex):
                cols = cols._get_values_for_csv(**self._number_format)
            else:
                cols = list(cols)
            self.obj = self.obj.loc[:, cols]

        # update columns to include possible multiplicity of dupes
        # and make sure cols is just a list of labels
        new_cols = self.obj.columns
        return new_cols._get_values_for_csv(**self._number_format)

    def _initialize_chunksize(self, chunksize: int | None) -> int:
        # [FUNCTION DOC]
        # HELPER: computes the number of rows per chunk. If chunksize is none, the
        #   default target cell count is divided by the number of columns (treating zero
        #   columns as one), with a minimum of one row. Otherwise returns the given
        #   value converted to an integer.
        if chunksize is None:
            return (_DEFAULT_CHUNKSIZE_CELLS // (len(self.cols) or 1)) or 1
        return int(chunksize)

    @property
    def _number_format(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: property returning a dictionary of number formatting settings (na
        #   representation, float format, date format, quoting, decimal separator) that
        #   is passed to the value formatting routines of the index, columns and frame.
        """Dictionary used for storing number formatting settings."""
        return {
            "na_rep": self.na_rep,
            "float_format": self.float_format,
            "date_format": self.date_format,
            "quoting": self.quoting,
            "decimal": self.decimal,
        }

    @cache_readonly
    def data_index(self) -> Index:
        # [FUNCTION DOC]
        # Cached read only property: the index used for output rows. For a date time or
        #   period index when a date format is set, builds a new plain index of strings
        #   formatted with that format (missing entries become the empty string). For a
        #   multi level index drops unused levels. Otherwise the frame's index is
        #   returned unchanged. The result is computed once and cached.
        data_index = self.obj.index
        if (
            isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex))
            and self.date_format is not None
        ):
            data_index = Index(
                [x.strftime(self.date_format) if notna(x) else "" for x in data_index]
            )
        elif isinstance(data_index, ABCMultiIndex):
            data_index = data_index.remove_unused_levels()
        return data_index

    @property
    def nlevels(self) -> int:
        # [FUNCTION DOC]
        # Read only property: the number of index levels that will be written. Zero when
        #   the index is not written; otherwise the level count of the data index (one
        #   for a flat index).
        if self.index:
            return getattr(self.data_index, "nlevels", 1)
        else:
            return 0

    @property
    def _has_aliases(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: true when the header option is a sequence of column aliases (tuple,
        #   list, array or index) rather than a boolean.
        return isinstance(self.header, (tuple, list, np.ndarray, ABCIndex))

    @property
    def _need_to_save_header(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: true when a header line must be written, that is when aliases are
        #   given or the header option is truthy.
        return bool(self._has_aliases or self.header)

    @property
    def write_cols(self) -> SequenceNotStr[Hashable]:
        # [FUNCTION DOC]
        # Read only property: the labels to put in the header. Returns the aliases if
        #   given, after checking that their count equals the number of columns,
        #   otherwise raising a value error saying how many columns are written and how
        #   many aliases were given. Without aliases returns the formatted column label
        #   array.
        if self._has_aliases:
            assert not isinstance(self.header, bool)
            if len(self.header) != len(self.cols):
                raise ValueError(
                    f"Writing {len(self.cols)} cols but got {len(self.header)} aliases"
                )
            return self.header
        else:
            # self.cols is an ndarray derived from Index._get_values_for_csv,
            #  so its entries are strings, i.e. hashable
            return cast(SequenceNotStr[Hashable], self.cols)

    @property
    def encoded_labels(self) -> list[Hashable]:
        # [FUNCTION DOC]
        # Read only property: the labels for the single header line. Starts with the
        #   index labels when the index is written and labels are present, then appends
        #   the header labels unless the columns are multi level without aliases (in
        #   which case the column labels are written in separate rows). Returns a list.
        encoded_labels: list[Hashable] = []

        if self.index and self.index_label:
            assert isinstance(self.index_label, Sequence)
            encoded_labels = list(self.index_label)

        if not self.has_mi_columns or self._has_aliases:
            encoded_labels += list(self.write_cols)

        return encoded_labels

    def save(self) -> None:
        # [FUNCTION DOC]
        # Public entry point: write everything. Opens the target through the common
        #   handle opener with the configured mode, encoding, errors, compression and
        #   storage options, creates the csv writer on the handle with the configured
        #   line terminator, delimiter, quoting, double quote handling, escape character
        #   and quote character (storing it in the writer attribute), and writes the
        #   header and body. Closes the handle on exit. Side effects: creates or
        #   overwrites the target or appends to it depending on mode. Returns nothing.
        """
        Create the writer & save.
        """
        # apply compression and byte/text conversion
        with get_handle(
            self.filepath_or_buffer,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            compression=self.compression,
            storage_options=self.storage_options,
        ) as handles:
            # Note: self.encoding is irrelevant here
            # error: Argument "quoting" to "writer" has incompatible type "int";
            # expected "Literal[0, 1, 2, 3]"
            self.writer = csvlib.writer(
                handles.handle,
                lineterminator=self.lineterminator,
                delimiter=self.sep,
                quoting=self.quoting,  # type: ignore[arg-type]
                doublequote=self.doublequote,
                escapechar=self.escapechar,
                quotechar=self.quotechar,
            )

            self._save()

    def _save(self) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the header, if needed, and then the body. Returns nothing.
        if self._need_to_save_header:
            self._save_header()
        self._save_body()

    def _save_header(self) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the header. Without multi level columns (or with aliases)
        #   writes one row of encoded labels; otherwise writes one row for each row
        #   produced by the multi level header generator. Returns nothing.
        if not self.has_mi_columns or self._has_aliases:
            self.writer.writerow(self.encoded_labels)
        else:
            for row in self._generate_multiindex_header_rows():
                self.writer.writerow(row)

    def _generate_multiindex_header_rows(self) -> Iterator[list[Hashable]]:
        # [FUNCTION DOC]
        # HELPER: generator of header rows for multi level columns. For each column
        #   level yields a row consisting of, when the index is written, the level name
        #   followed by empty strings to fill the remaining index columns, and then the
        #   level values of all columns. After all levels, yields one more row holding
        #   the index labels padded with empty strings for the data columns, but only if
        #   there are index labels and they are not all empty (to avoid an extra blank
        #   line).
        columns = self.obj.columns
        for i in range(columns.nlevels):
            # we need at least 1 index column to write our col names
            col_line = []
            if self.index:
                # name is the first column
                col_line.append(columns.names[i])

                if isinstance(self.index_label, list) and len(self.index_label) > 1:
                    col_line.extend([""] * (len(self.index_label) - 1))

            col_line.extend(columns._get_level_values(i))
            yield col_line

        # Write out the index line if it's not empty.
        # Otherwise, we will print out an extraneous
        # blank line between the mi and the data rows.
        if self.encoded_labels and set(self.encoded_labels) != {""}:
            yield self.encoded_labels + [""] * len(columns)

    def _save_body(self) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the data rows in chunks. Splits the row range into consecutive
        #   chunks of the configured size and writes each non empty chunk, stopping at
        #   the first empty range. Returns nothing.
        nrows = len(self.data_index)
        chunks = (nrows // self.chunksize) + 1
        for i in range(chunks):
            start_i = i * self.chunksize
            end_i = min(start_i + self.chunksize, nrows)
            if start_i >= end_i:
                break
            self._save_chunk(start_i, end_i)

    def _save_chunk(self, start_i: int, end_i: int) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the rows between start_i (inclusive) and end_i (exclusive).
        #   Slices the frame by position, converts the slice to formatted text columns
        #   using the number formatting settings, converts the matching part of the data
        #   index when an index is written (otherwise an empty placeholder array), and
        #   hands the columns, the index values, the number of index levels, the column
        #   labels and the csv writer to the compiled row writing routine. Returns
        #   nothing.
        # create the data for a chunk
        slicer = slice(start_i, end_i)
        df = self.obj.iloc[slicer]

        res = df._get_values_for_csv(**self._number_format)
        data = list(res._iter_column_arrays())

        ix = (
            self.data_index[slicer]._get_values_for_csv(**self._number_format)
            if self.nlevels != 0
            else np.empty(end_i - start_i)
        )
        libwriters.write_csv_rows(
            data,
            ix,
            self.nlevels,
            self.cols,
            self.writer,
        )
