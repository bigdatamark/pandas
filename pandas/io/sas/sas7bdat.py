"""
Read SAS7BDAT files

Based on code written by Jared Hobbs:
  https://bitbucket.org/jaredhobbs/sas7bdat

See also:
  https://github.com/BioStatMatt/sas7bdat

Partial documentation of the file format:
  https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf

Reference for binary data compression:
  http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
"""

# [MODULE DOC]
# Reader for SAS sas7bdat binary data set files, producing data frames.
# Public contents: the SAS7BDATReader class (subclass of the abstract SAS
# reader). Private contents: a small column description class, a helper
# converting SAS date or datetime numbers to timestamps, and two origin
# timestamps (the Unix epoch and the SAS epoch of 1960-01-01).
# Relationships: layout constants come from the SAS constants module; the
# abstract reader and read_sas entry point live in the sasreader module (which
# imports this module lazily); the row-decoding hot loop is a compiled parser
# class from the internal sas library, plus compiled byte-swapping readers, and
# it calls back into methods of the reader defined here (page reading, row and
# page indexes, cached page and chunk buffers).
# Import-time side effects: creates the two module-level timestamps.
from __future__ import annotations

from datetime import datetime
import sys
from typing import TYPE_CHECKING

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs.byteswap import (
    read_double_with_byteswap,
    read_float_with_byteswap,
    read_uint16_with_byteswap,
    read_uint32_with_byteswap,
    read_uint64_with_byteswap,
)
from pandas._libs.sas import (
    Parser,
    get_subheader_index,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)

from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
    )


# HELPER: module constants. Timestamps for the Unix epoch and SAS epoch, used to
# shift SAS date numbers (days or seconds since 1960-01-01) to Unix-based values.
_unix_origin = Timestamp("1970-01-01")
_sas_origin = Timestamp("1960-01-01")


def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    """
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {'d', 's'}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    """
    # [FUNCTION DOC]
    # HELPER: converts SAS numeric dates or datetimes into datetime values.
    # Inputs: sas_datetimes is a series of floats (days or seconds since the SAS
    # epoch); unit is "s" for datetimes in seconds, anything else is treated as
    # dates in days.
    # For seconds, values are scaled to milliseconds and returned as a series
    # with millisecond resolution, shifted from the SAS epoch to the Unix epoch.
    # For days, values are cast to day-resolution dates, shifted likewise, and
    # returned with second resolution. The result keeps the input index and does
    # not copy the data.
    # KNOWN ISSUE: the docstring says the result may fall back to Python
    # datetime objects and describes unit as only d or s, but the code always
    # returns a datetime64 series with no fallback.
    td = (_sas_origin - _unix_origin).as_unit("s")
    if unit == "s":
        millis = cast_from_unit_vectorized(
            sas_datetimes._values, unit="s", out_unit="ms"
        )
        dt64ms = millis.view("M8[ms]") + td
        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
    else:
        vals = np.array(sas_datetimes, dtype="M8[D]") + td
        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)


class _Column:
    # [CLASS DOC]
    # HELPER: plain record describing one column of a sas7bdat file.
    # Attributes: column id (zero-based position), name, label, display format
    # (each text or raw bytes depending on the header text conversion option),
    # type code as a one-byte value (d for numeric, s for string), and data
    # length in bytes. No behavior beyond storing the values.
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int

    def __init__(
        self,
        col_id: int,
        # These can be bytes when convert_header_text is False
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the six given values (column id, name, label, format,
        # type code, length) as attributes of the same names. No validation.
        self.col_id = col_id
        self.name = name
        self.label = label
        self.format = format
        self.ctype = ctype
        self.length = length


# SAS7BDAT represents a SAS data file in SAS7BDAT format.
class SAS7BDATReader(SASReader):
    # [CLASS DOC]
    # Reader for one sas7bdat file. Subclass of the abstract SAS reader:
    # supports iteration (each step yields a chunk of rows) and use as a
    # context manager.
    # Construction opens the file in binary mode (compression option applies),
    # parses the file header (alignment, byte order, encoding, timestamps,
    # header and page length) and then walks metadata pages, running subheader
    # processors to learn row length and count, column names, labels, formats,
    # types, data offsets and lengths, and the compression scheme.
    # State: configuration options (index, date and text conversion, blank
    # handling, chunk size, encoding); header facts (64-bit flag, byte order,
    # whether byte swapping is needed, inferred encoding, creation and
    # modification times, header and page lengths, offsets used for pointer
    # sizes); metadata lists (raw and converted column names, formats, column
    # records, per-column lengths, offsets and type codes); read cursors (rows
    # read from the file, row index on the current page, rows in the current
    # chunk, current page type, block and subheader counts, data subheader
    # pointers for compressed pages); the cached current page bytes; and string
    # and byte chunk buffers filled by the compiled parser.
    # Key methods: read, the iterator step, close, the header and subheader
    # processors, page reader, and chunk to data frame conversion.
    # Invariant: the cached page always holds the most recently read page (or
    # the header while parsing the header).
    """
    Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : str, 'infer', defaults to None
        String encoding acc. to Python standard encodings,
        encoding='infer' tries to detect the encoding from the file header,
        encoding=None will leave the data in binary format.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """

    _int_length: int
    _cached_page: bytes | None

    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        index=None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: int | None = None,
        encoding: str | None = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: CompressionOptions = "infer",
    ) -> None:
        # [FUNCTION DOC]
        # Opens the file and reads all metadata.
        # Parameters: path_or_buf is a path or binary buffer; index is the
        # optional column to use as index; convert_dates (default true) turns
        # numeric columns with SAS date or datetime formats into timestamps;
        # blank_missing (default true) is stored for the row parser; chunksize
        # is the rows per iteration step (none means one); encoding is a codec
        # name, the word infer to use the codec named in the file header, or
        # none to leave text as bytes; convert_text (default true) decodes
        # string columns when an encoding is set; convert_header_text (default
        # true) decodes header text such as names and labels; compression is
        # the file compression option (default inferred).
        # The fallback decoding is latin-1 for header text when no encoding was
        # given. Initializes all metadata and cursor attributes, opens the file,
        # and builds the table of subheader processors in the order of the
        # subheader index constants (the last entry, for data subheaders, is
        # none).
        # If reading properties or metadata fails, the handles are closed and the
        # error propagates.
        # KNOWN ISSUE: the current row in file index attribute is initialized
        # twice, and the class docstring omits the compression parameter.
        self.index = index
        self.convert_dates = convert_dates
        self.blank_missing = blank_missing
        self.chunksize = chunksize
        self.encoding = encoding
        self.convert_text = convert_text
        self.convert_header_text = convert_header_text

        self.default_encoding = "latin-1"
        self.compression = b""
        self.column_names_raw: list[bytes] = []
        self.column_names: list[str | bytes] = []
        self.column_formats: list[str | bytes] = []
        self.columns: list[_Column] = []

        self._current_page_data_subheader_pointers: list[tuple[int, int]] = []
        self._cached_page = None
        self._column_data_lengths: list[int] = []
        self._column_data_offsets: list[int] = []
        self._column_types: list[bytes] = []

        self._current_row_in_file_index = 0
        self._current_row_on_page_index = 0
        self._current_row_in_file_index = 0

        self.handles = get_handle(
            path_or_buf, "rb", is_text=False, compression=compression
        )

        self._path_or_buf = self.handles.handle

        # Same order as const.SASIndex
        self._subheader_processors = [
            self._process_rowsize_subheader,
            self._process_columnsize_subheader,
            self._process_subheader_counts,
            self._process_columntext_subheader,
            self._process_columnname_subheader,
            self._process_columnattributes_subheader,
            self._process_format_subheader,
            self._process_columnlist_subheader,
            None,  # Data
        ]

        try:
            self._get_properties()
            self._parse_metadata()
        except Exception:
            self.close()
            raise

    def column_data_lengths(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns the per-column data lengths in bytes as a new 64-bit integer
        # array, in column order. Used by the compiled parser.
        """Return a numpy int64 array of the column data lengths"""
        return np.asarray(self._column_data_lengths, dtype=np.int64)

    def column_data_offsets(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns the per-column byte offsets within a row as a new 64-bit
        # integer array, in column order. Used by the compiled parser.
        """Return a numpy int64 array of the column offsets"""
        return np.asarray(self._column_data_offsets, dtype=np.int64)

    def column_types(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Returns the column type codes as a new array of one-byte strings,
        # in column order: d for numeric (double) and s for string. Used by the
        # compiled parser.
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        return np.asarray(self._column_types, dtype=np.dtype("S1"))

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes all file handles opened by this reader.
        self.handles.close()

    def _get_properties(self) -> None:
        # [FUNCTION DOC]
        # HELPER: parses the file header, setting reader attributes.
        # Rewinds the stream and caches the first 288 bytes, checking the magic
        # number. Then determines: 64-bit layout (selects integer width, page bit
        # offset and subheader pointer length; otherwise 32-bit values), an extra
        # alignment amount applied to later offsets, byte order (little endian
        # when the flag byte is one; sets whether swapping is needed relative to
        # the host), and the encoding code (mapped to a codec name stored as the
        # inferred encoding, or an unknown marker; when the requested encoding is
        # the word infer it is replaced by the inferred one, which stays the
        # word infer if the code is unknown).
        # Also reads creation and modification times (seconds since 1960) into
        # timestamps, the header length, appends the rest of the header to the
        # cache (the cache must then equal the header length) and reads the page
        # length.
        # Raises value errors for a magic mismatch or a truncated header.
        # KNOWN ISSUE: when the encoding is set to infer and the file reports an
        # unknown code, the encoding stays as the word infer, which later fails
        # as a codec name when decoding.
        # Check magic number
        self._path_or_buf.seek(0)
        self._cached_page = self._path_or_buf.read(288)
        if self._cached_page[0 : len(const.magic)] != const.magic:
            raise ValueError("magic number mismatch (not a SAS file?)")

        # Get alignment information
        buf = self._read_bytes(const.align_1_offset, const.align_1_length)
        if buf == const.u64_byte_checker_value:
            self.U64 = True
            self._int_length = 8
            self._page_bit_offset = const.page_bit_offset_x64
            self._subheader_pointer_length = const.subheader_pointer_length_x64
        else:
            self.U64 = False
            self._page_bit_offset = const.page_bit_offset_x86
            self._subheader_pointer_length = const.subheader_pointer_length_x86
            self._int_length = 4
        buf = self._read_bytes(const.align_2_offset, const.align_2_length)
        if buf == const.align_1_checker_value:
            align1 = const.align_2_value
        else:
            align1 = 0

        # Get endianness information
        buf = self._read_bytes(const.endianness_offset, const.endianness_length)
        if buf == b"\x01":
            self.byte_order = "<"
            self.need_byteswap = sys.byteorder == "big"
        else:
            self.byte_order = ">"
            self.need_byteswap = sys.byteorder == "little"

        # Get encoding information
        buf = self._read_bytes(const.encoding_offset, const.encoding_length)[0]
        if buf in const.encoding_names:
            self.inferred_encoding = const.encoding_names[buf]
            if self.encoding == "infer":
                self.encoding = self.inferred_encoding
        else:
            self.inferred_encoding = f"unknown (code={buf})"

        # Timestamp is epoch 01/01/1960
        epoch = datetime(1960, 1, 1)
        x = self._read_float(
            const.date_created_offset + align1, const.date_created_length
        )
        self.date_created = epoch + pd.to_timedelta(x, unit="s")
        x = self._read_float(
            const.date_modified_offset + align1, const.date_modified_length
        )
        self.date_modified = epoch + pd.to_timedelta(x, unit="s")

        self.header_length = self._read_uint(
            const.header_size_offset + align1, const.header_size_length
        )

        # Read the rest of the header into cached_page.
        buf = self._path_or_buf.read(self.header_length - 288)
        self._cached_page += buf
        if len(self._cached_page) != self.header_length:
            raise ValueError("The SAS7BDAT file appears to be truncated.")

        self._page_length = self._read_uint(
            const.page_size_offset + align1, const.page_size_length
        )

    def __next__(self) -> DataFrame:
        # [FUNCTION DOC]
        # Iterator step: reads the next chunk of rows (the chunk size, or one
        # row if unset). If the resulting frame is empty, closes the reader and
        # stops iteration; otherwise returns the frame.
        da = self.read(nrows=self.chunksize or 1)
        if da.empty:
            self.close()
            raise StopIteration
        return da

    # Read a single float of the given width (4 or 8).
    def _read_float(self, offset: int, width: int) -> float:
        # [FUNCTION DOC]
        # HELPER: reads one floating point number from the cached page.
        # Inputs: byte offset into the cached page and width (4 or 8 bytes),
        # honoring the file byte order via the byte swapping flag. Returns the
        # value. For any other width closes the reader and raises a value error.
        assert self._cached_page is not None
        if width == 4:
            return read_float_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 8:
            return read_double_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        else:
            self.close()
            raise ValueError("invalid float width")

    # Read a single unsigned integer of the given width (1, 2, 4 or 8).
    def _read_uint(self, offset: int, width: int) -> int:
        # [FUNCTION DOC]
        # HELPER: reads one unsigned integer from the cached page.
        # Inputs: byte offset and width (1, 2, 4 or 8 bytes); multi-byte values
        # honor the file byte order via the byte swapping flag. Returns the
        # value as an integer. For any other width closes the reader and raises a
        # value error.
        assert self._cached_page is not None
        if width == 1:
            return self._read_bytes(offset, 1)[0]
        elif width == 2:
            return read_uint16_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 4:
            return read_uint32_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 8:
            return read_uint64_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        else:
            self.close()
            raise ValueError("invalid int width")

    def _read_bytes(self, offset: int, length: int):
        # [FUNCTION DOC]
        # HELPER: returns a slice of the cached page of the given length starting
        # at the offset. If the slice would extend beyond the cached page, closes
        # the reader and raises a value error.
        assert self._cached_page is not None
        if offset + length > len(self._cached_page):
            self.close()
            raise ValueError("The cached page is too small.")
        return self._cached_page[offset : offset + length]

    def _parse_metadata(self) -> None:
        # [FUNCTION DOC]
        # HELPER: scans pages sequentially after the header until the first
        # page that holds data (or yields data subheaders), populating column
        # metadata along the way. Each page read becomes the cached page. Stops
        # at end of file. Raises a value error if a page read is incomplete.
        done = False
        while not done:
            self._cached_page = self._path_or_buf.read(self._page_length)
            if len(self._cached_page) <= 0:
                break
            if len(self._cached_page) != self._page_length:
                raise ValueError("Failed to read a meta data page from the SAS file.")
            done = self._process_page_meta()

    def _process_page_meta(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: handles one metadata-scanning page. Reads its header, and for
        # metadata, amendment and mixed page types processes the subheaders.
        # Returns true if the page is a data page, a mixed page, or has recorded
        # data subheader pointers, signalling the metadata scan to stop;
        # otherwise false.
        self._read_page_header()
        pt = [*const.page_meta_types, const.page_amd_type, const.page_mix_type]
        if self._current_page_type in pt:
            self._process_page_metadata()
        is_data_page = self._current_page_type == const.page_data_type
        is_mix_page = self._current_page_type == const.page_mix_type
        return bool(
            is_data_page
            or is_mix_page
            or self._current_page_data_subheader_pointers != []
        )

    def _read_page_header(self) -> None:
        # [FUNCTION DOC]
        # HELPER: decodes the header of the cached page, setting the current
        # page type (masked to keep type and compression bits), the block count
        # and the subheader count, using the page bit offset for the file
        # layout.
        bit_offset = self._page_bit_offset
        tx = const.page_type_offset + bit_offset
        self._current_page_type = (
            self._read_uint(tx, const.page_type_length) & const.page_type_mask2
        )
        tx = const.block_count_offset + bit_offset
        self._current_page_block_count = self._read_uint(tx, const.block_count_length)
        tx = const.subheader_count_offset + bit_offset
        self._current_page_subheaders_count = self._read_uint(
            tx, const.subheader_count_length
        )

    def _process_page_metadata(self) -> None:
        # [FUNCTION DOC]
        # HELPER: walks the subheader pointer table of the cached page. For
        # each pointer reads the subheader offset, length, compression flag and
        # type. Skips empty or truncated subheaders. Reads the signature at the
        # subheader offset, looks up its kind, and calls the matching processor
        # with the offset and length.
        # Data subheaders (no processor) are recorded as offset and length
        # pairs when the file is compressed and the flag and type match the
        # compressed values; otherwise the reader is closed and a value error
        # raised for an unknown signature.
        bit_offset = self._page_bit_offset

        for i in range(self._current_page_subheaders_count):
            offset = const.subheader_pointers_offset + bit_offset
            total_offset = offset + self._subheader_pointer_length * i

            subheader_offset = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            subheader_length = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            subheader_compression = self._read_uint(total_offset, 1)
            total_offset += 1

            subheader_type = self._read_uint(total_offset, 1)

            if (
                subheader_length == 0
                or subheader_compression == const.truncated_subheader_id
            ):
                continue

            subheader_signature = self._read_bytes(subheader_offset, self._int_length)
            subheader_index = get_subheader_index(subheader_signature)
            subheader_processor = self._subheader_processors[subheader_index]

            if subheader_processor is None:
                f1 = subheader_compression in (const.compressed_subheader_id, 0)
                f2 = subheader_type == const.compressed_subheader_type
                if self.compression and f1 and f2:
                    self._current_page_data_subheader_pointers.append(
                        (subheader_offset, subheader_length)
                    )
                else:
                    self.close()
                    raise ValueError(
                        f"Unknown subheader signature {subheader_signature}"
                    )
            else:
                subheader_processor(subheader_offset, subheader_length)

    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for the row size subheader. The length
        # argument is unused. Reads, at multiples of the integer width past the
        # offset: row length, row count, the two parts of the column count and
        # the row count on mixed pages. Reads the lengths of the creator
        # software and procedure strings as two byte values at fixed offsets that
        # differ between 64-bit and 32-bit files. Sets the corresponding
        # attributes.
        int_len = self._int_length
        lcs_offset = offset
        lcp_offset = offset
        if self.U64:
            lcs_offset += 682
            lcp_offset += 706
        else:
            lcs_offset += 354
            lcp_offset += 378

        self.row_length = self._read_uint(
            offset + const.row_length_offset_multiplier * int_len,
            int_len,
        )
        self.row_count = self._read_uint(
            offset + const.row_count_offset_multiplier * int_len,
            int_len,
        )
        self.col_count_p1 = self._read_uint(
            offset + const.col_count_p1_multiplier * int_len, int_len
        )
        self.col_count_p2 = self._read_uint(
            offset + const.col_count_p2_multiplier * int_len, int_len
        )
        mx = const.row_count_on_mix_page_offset_multiplier * int_len
        self._mix_page_row_count = self._read_uint(offset + mx, int_len)
        self._lcs = self._read_uint(lcs_offset, 2)
        self._lcp = self._read_uint(lcp_offset, 2)

    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for the column size subheader. The length
        # argument is unused. Reads the column count (an integer after the
        # first integer-width field) and stores it. If it differs from the sum of
        # the two parts read earlier, writes a warning line to standard output.
        # KNOWN ISSUE: the mismatch is reported with a plain print to standard
        # output rather than a warning, and processing continues.
        int_len = self._int_length
        offset += int_len
        self.column_count = self._read_uint(offset, int_len)
        if self.col_count_p1 + self.col_count_p2 != self.column_count:
            print(
                f"Warning: column count mismatch ({self.col_count_p1} + "
                f"{self.col_count_p2} != {self.column_count})\n"
            )

    # Unknown purpose
    def _process_subheader_counts(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for the subheader counts kind. Purpose
        # unknown, so does nothing.
        pass

    def _process_columntext_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for column text blocks. The length argument
        # is unused. Reads the text block size, extracts the block with trailing
        # null and space bytes removed, and appends it to the list of raw text
        # blocks (names, labels and formats are later cut out of these blocks).
        # For the first block only: detects the compression scheme by searching
        # the block for the known compression literals (the last match wins; none
        # if absent) and stores it, then locates the creator procedure name using
        # offsets that vary with 64-bit layout and with the compression scheme
        # and string lengths, stores it (converted with the header text rule)
        # when found.
        # KNOWN ISSUE: in the branch taken when the procedure length was
        # zeroed and the software length is positive, the length used to slice
        # the buffer is the zero just assigned, so the stored procedure name is
        # always empty.
        offset += self._int_length
        text_block_size = self._read_uint(offset, const.text_block_size_length)

        buf = self._read_bytes(offset, text_block_size)
        cname_raw = buf[0:text_block_size].rstrip(b"\x00 ")
        self.column_names_raw.append(cname_raw)

        if len(self.column_names_raw) == 1:
            compression_literal = b""
            for cl in const.compression_literals:
                if cl in cname_raw:
                    compression_literal = cl
            self.compression = compression_literal
            offset -= self._int_length

            offset1 = offset + 16
            if self.U64:
                offset1 += 4

            buf = self._read_bytes(offset1, self._lcp)
            compression_literal = buf.rstrip(b"\x00")
            if compression_literal == b"":
                self._lcs = 0
                offset1 = offset + 32
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcp)
                self.creator_proc = buf[0 : self._lcp]
            elif compression_literal == const.rle_compression:
                offset1 = offset + 40
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcp)
                self.creator_proc = buf[0 : self._lcp]
            elif self._lcs > 0:
                self._lcp = 0
                offset1 = offset + 16
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcs)
                self.creator_proc = buf[0 : self._lcp]
            if hasattr(self, "creator_proc"):
                self.creator_proc = self._convert_header_text(self.creator_proc)  # pyright: ignore[reportArgumentType]

    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for column names. Computes the number of
        # eight-byte name pointers from the subheader length. For each pointer
        # reads the index of the text block, the offset and the length of the
        # name inside that block, cuts the name out of the raw block, converts it
        # with the header text rule and appends it to the list of column names.
        int_len = self._int_length
        offset += int_len
        column_name_pointers_count = (length - 2 * int_len - 12) // 8
        for i in range(column_name_pointers_count):
            text_subheader = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_text_subheader_offset
            )
            col_name_offset = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_offset_offset
            )
            col_name_length = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_length_offset
            )

            idx = self._read_uint(
                text_subheader, const.column_name_text_subheader_length
            )
            col_offset = self._read_uint(
                col_name_offset, const.column_name_offset_length
            )
            col_len = self._read_uint(col_name_length, const.column_name_length_length)

            name_raw = self.column_names_raw[idx]
            cname = name_raw[col_offset : col_offset + col_len]
            self.column_names.append(self._convert_header_text(cname))

    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for column attributes. Computes the number
        # of attribute vectors from the subheader length and the integer width.
        # For each vector reads the column data offset, data length and type code
        # and appends them to the offset, length and type lists. A type code of
        # one means numeric (d); any other value means string (s).
        int_len = self._int_length
        column_attributes_vectors_count = (length - 2 * int_len - 12) // (int_len + 8)
        for i in range(column_attributes_vectors_count):
            col_data_offset = (
                offset + int_len + const.column_data_offset_offset + i * (int_len + 8)
            )
            col_data_len = (
                offset
                + 2 * int_len
                + const.column_data_length_offset
                + i * (int_len + 8)
            )
            col_types = (
                offset + 2 * int_len + const.column_type_offset + i * (int_len + 8)
            )

            x = self._read_uint(col_data_offset, int_len)
            self._column_data_offsets.append(x)

            x = self._read_uint(col_data_len, const.column_data_length_length)
            self._column_data_lengths.append(x)

            x = self._read_uint(col_types, const.column_type_length)
            self._column_types.append(b"d" if x == 1 else b"s")

    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for the column list kind. Purpose unknown,
        # so does nothing.
        # unknown purpose
        pass

    def _process_format_subheader(self, offset: int, length: int) -> None:
        # [FUNCTION DOC]
        # HELPER: subheader processor for format and label descriptors. The
        # length argument is unused. Reads the text block indexes (clamped to the
        # last available block), offsets and lengths for one column's format and
        # label, cuts them from the raw text blocks, and converts them with the
        # header text rule. Creates a column record for the next column in order
        # (taking its name, type and data length from the previously collected
        # lists, so those must already be available) and appends the format to
        # the format list and the record to the column list.
        int_len = self._int_length
        text_subheader_format = (
            offset + const.column_format_text_subheader_index_offset + 3 * int_len
        )
        col_format_offset = offset + const.column_format_offset_offset + 3 * int_len
        col_format_len = offset + const.column_format_length_offset + 3 * int_len
        text_subheader_label = (
            offset + const.column_label_text_subheader_index_offset + 3 * int_len
        )
        col_label_offset = offset + const.column_label_offset_offset + 3 * int_len
        col_label_len = offset + const.column_label_length_offset + 3 * int_len

        x = self._read_uint(
            text_subheader_format, const.column_format_text_subheader_index_length
        )
        format_idx = min(x, len(self.column_names_raw) - 1)

        format_start = self._read_uint(
            col_format_offset, const.column_format_offset_length
        )
        format_len = self._read_uint(col_format_len, const.column_format_length_length)

        label_idx = self._read_uint(
            text_subheader_label, const.column_label_text_subheader_index_length
        )
        label_idx = min(label_idx, len(self.column_names_raw) - 1)

        label_start = self._read_uint(
            col_label_offset, const.column_label_offset_length
        )
        label_len = self._read_uint(col_label_len, const.column_label_length_length)

        label_names = self.column_names_raw[label_idx]
        column_label = self._convert_header_text(
            label_names[label_start : label_start + label_len]
        )
        format_names = self.column_names_raw[format_idx]
        column_format = self._convert_header_text(
            format_names[format_start : format_start + format_len]
        )
        current_column_number = len(self.columns)

        col = _Column(
            current_column_number,
            self.column_names[current_column_number],
            column_label,
            column_format,
            self._column_types[current_column_number],
            self._column_data_lengths[current_column_number],
        )

        self.column_formats.append(column_format)
        self.columns.append(col)

    def read(self, nrows: int | None = None) -> DataFrame:
        # [FUNCTION DOC]
        # Reads rows and returns them as a data frame.
        # Parameter nrows is the number of rows; when none, the chunk size is
        # used if configured, else the whole file row count.
        # Raises an empty data error (after closing the reader) when the file has
        # no columns. Returns an empty frame if rows were requested but the
        # cursor is already at the end. Otherwise limits the count to the rows
        # remaining, allocates object-array storage for string columns and a
        # byte buffer for numeric columns, runs the compiled parser to fill them
        # and advance the cursors, builds the frame, and sets the configured
        # index column if any.
        # Side effects: resets the chunk row counter, replaces the chunk buffers,
        # advances file cursors and may read further pages.
        if (nrows is None) and (self.chunksize is not None):
            nrows = self.chunksize
        elif nrows is None:
            nrows = self.row_count

        if len(self._column_types) == 0:
            self.close()
            raise EmptyDataError("No columns to parse from file")

        if nrows > 0 and self._current_row_in_file_index >= self.row_count:
            return DataFrame()

        nrows = min(nrows, self.row_count - self._current_row_in_file_index)

        nd = self._column_types.count(b"d")
        ns = self._column_types.count(b"s")

        self._string_chunk = np.empty((ns, nrows), dtype=object)
        self._byte_chunk = np.zeros((nd, 8 * nrows), dtype=np.uint8)

        self._current_row_in_chunk_index = 0
        p = Parser(self)
        p.read(nrows)

        rslt = self._chunk_to_dataframe()
        if self.index is not None:
            rslt = rslt.set_index(self.index)

        return rslt

    def _read_next_page(self):
        # [FUNCTION DOC]
        # HELPER: called by the compiled parser to advance to the next page.
        # Clears the data subheader pointers and reads one page into the cache.
        # Returns true at end of file. If the read is short, closes the reader and
        # raises a value error reporting bytes read versus expected. Otherwise
        # reads the page header and, for metadata page types, processes their
        # subheaders. Pages whose type is not metadata, data or mixed are skipped
        # by recursing; returns false when a usable page was loaded.
        # KNOWN ISSUE: unlike the metadata scan, this path does not process
        # subheaders of amendment or mixed pages; only metadata page types are
        # processed.
        self._current_page_data_subheader_pointers = []
        self._cached_page = self._path_or_buf.read(self._page_length)
        if len(self._cached_page) <= 0:
            return True
        elif len(self._cached_page) != self._page_length:
            self.close()
            msg = (
                "failed to read complete page from file (read "
                f"{len(self._cached_page):d} of {self._page_length:d} bytes)"
            )
            raise ValueError(msg)

        self._read_page_header()
        if self._current_page_type in const.page_meta_types:
            self._process_page_metadata()

        if self._current_page_type not in [
            *const.page_meta_types,
            const.page_data_type,
            const.page_mix_type,
        ]:
            return self._read_next_page()

        return False

    def _chunk_to_dataframe(self) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: converts the filled chunk buffers into a data frame.
        # The row index is a range ending at the current file row index with
        # length equal to the rows in the chunk. For each column in file order:
        # numeric columns are viewed from the byte buffer using the file byte
        # order as float64 series, and when date conversion is on and the format
        # is among the known date formats they are converted as dates, or among
        # the datetime formats as datetimes; string columns come from the object
        # buffer and, when text conversion is on and an encoding is set, are
        # decoded and (when the string dtype option is enabled) cast to the
        # string dtype. Any other type code closes the reader and raises a value
        # error. Returns a frame with columns in the order of the column names
        # (duplicate names would collapse in the intermediate mapping).
        n = self._current_row_in_chunk_index
        m = self._current_row_in_file_index
        ix = range(m - n, m)
        rslt = {}

        js, jb = 0, 0
        infer_string = using_string_dtype()
        for j in range(self.column_count):
            name = self.column_names[j]

            if self._column_types[j] == b"d":
                col_arr = self._byte_chunk[jb, :].view(dtype=self.byte_order + "d")
                rslt[name] = pd.Series(col_arr, dtype=np.float64, index=ix, copy=False)
                if self.convert_dates:
                    if self.column_formats[j] in const.sas_date_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "d")
                    elif self.column_formats[j] in const.sas_datetime_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "s")
                jb += 1
            elif self._column_types[j] == b"s":
                rslt[name] = pd.Series(self._string_chunk[js, :], index=ix, copy=False)
                if self.convert_text and (self.encoding is not None):
                    rslt[name] = self._decode_string(rslt[name].str)
                    if infer_string:
                        rslt[name] = rslt[name].astype("str")

                js += 1
            else:
                self.close()
                raise ValueError(f"unknown column type {self._column_types[j]!r}")

        df = DataFrame(rslt, columns=self.column_names, index=ix, copy=False)
        return df

    def _decode_string(self, b):
        # [FUNCTION DOC]
        # HELPER: decodes bytes using the configured encoding, falling back to
        # latin-1 when none is set. Accepts a bytes value or a series string
        # accessor of bytes (anything with a decode method) and returns the
        # decoded result.
        return b.decode(self.encoding or self.default_encoding)

    def _convert_header_text(self, b: bytes) -> str | bytes:
        # [FUNCTION DOC]
        # HELPER: returns header text as decoded text when header text
        # conversion is enabled (using the string decoder), else the raw bytes.
        if self.convert_header_text:
            return self._decode_string(b)
        else:
            return b
