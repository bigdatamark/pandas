"""
Read a SAS XPort format file into a Pandas DataFrame.

Based on code from Jack Cushman (github.com/jcushman/xport).

The file format is defined here:

https://support.sas.com/content/dam/SAS/support/en/technical-papers/record-layout-of-a-sas-version-5-or-6-data-set-in-sas-transport-xport-format.pdf
"""

# [MODULE DOC]
# Reader for SAS version 5/6 transport (XPORT) files, producing data frames.
# Public contents: the XportReader class (a SASReader subclass).
# Also defines private module constants: the expected fixed-width header
# records, the list of field descriptor keys, and unused documentation
# template strings; plus private helpers to parse dates, split fixed-width
# lines, pad truncated floats and convert IBM floating point to IEEE.
# Relationships: subclasses the abstract reader in the sasreader module, which
# creates it lazily; opens files through the io common handle helper.
# Import-time side effects: none beyond importing pandas itself.
from __future__ import annotations

from datetime import datetime
import struct
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas.util._exceptions import find_stack_level

import pandas as pd

from pandas.io.common import get_handle
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        DatetimeNaTType,
        FilePath,
        ReadBuffer,
    )
# HELPER: module constants. The four expected header record strings (library
# header, member header prefix, descriptor header, observation header) and the
# ordered list of names for the 16 packed fields of a variable descriptor
# (the last, named with an underscore, is padding that is discarded).
_correct_line1 = (
    "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  "
)
_correct_header1 = (
    "HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000"
)
_correct_header2 = (
    "HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000  "
)
_correct_obs_header = (
    "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  "
)
_fieldkeys = [
    "ntype",
    "nhfun",
    "field_length",
    "nvar0",
    "name",
    "label",
    "nform",
    "nfl",
    "num_decimals",
    "nfj",
    "nfill",
    "niform",
    "nifl",
    "nifd",
    "npos",
    "_",
]


# HELPER: the following string templates only build documentation text; they
# are not referenced by any code in this module other than the class doc.
# KNOWN ISSUE: the read function template is never used (the read_sas
# docstring lives in the sasreader module), and it claims that only xport is
# supported.
_base_params_doc = """\
Parameters
----------
filepath_or_buffer : str or file-like object
    Path to SAS file or object implementing binary read method."""

_params2_doc = """\
index : identifier of index column
    Identifier of column that should be used as index of the DataFrame.
encoding : str
    Encoding for text data.
chunksize : int
    Read file `chunksize` lines at a time, returns iterator."""

_format_params_doc = """\
format : str
    File format, only `xport` is currently supported."""

_iterator_doc = """\
iterator : bool, default False
    Return XportReader object for reading file incrementally."""


_read_sas_doc = f"""Read a SAS file into a DataFrame.

{_base_params_doc}
{_format_params_doc}
{_params2_doc}
{_iterator_doc}

Returns
-------
DataFrame or XportReader

Examples
--------
Read a SAS Xport file:

>>> df = pd.read_sas('filename.XPT')

Read a Xport file in 10,000 line chunks:

>>> itr = pd.read_sas('filename.XPT', chunksize=10000)
>>> for chunk in itr:
>>>     do_something(chunk)

"""

_xport_reader_doc = f"""\
Class for reading SAS Xport files.

{_base_params_doc}
{_params2_doc}

Attributes
----------
member_info : list
    Contains information about the file
fields : list
    Contains information about the variables in the file
"""


def _parse_date(datestr: str) -> DatetimeNaTType:
    # [FUNCTION DOC]
    # HELPER: converts a date string from an xport header to a timestamp.
    # Input is a text value in day, abbreviated month, two-digit year, then
    # hour, minute, second form. Returns the parsed datetime, or the missing
    # timestamp marker when the text does not match that format. Never raises
    # for bad format.
    """Given a date in xport format, return Python date."""
    try:
        # e.g. "16FEB11:10:07:55"
        return datetime.strptime(datestr, "%d%b%y:%H:%M:%S")
    except ValueError:
        return pd.NaT


def _split_line(s: str, parts):
    # [FUNCTION DOC]
    # HELPER: slices a fixed-width string into consecutive named pieces.
    # Input s is the text and parts is a sequence of name and length pairs.
    # Each piece is stripped of surrounding whitespace and stored under its
    # name (a repeated name overwrites). The piece named with a single
    # underscore is removed from the result. Returns a dictionary. Raises a key
    # error if no part is named with an underscore.
    """
    Parameters
    ----------
    s: str
        Fixed-length string to split
    parts: list of (name, length) pairs
        Used to break up string, name '_' will be filtered from output.

    Returns
    -------
    Dict of name:contents of string at given location.
    """
    out = {}
    start = 0
    for name, length in parts:
        out[name] = s[start : start + length].strip()
        start += length
    del out["_"]
    return out


def _handle_truncated_float_vec(vec, nbytes):
    # [FUNCTION DOC]
    # HELPER: widens truncated numeric fields to full 8-byte values.
    # Input vec is an array of fixed-width byte strings of length nbytes
    # (between 2 and 8). When nbytes is not 8, returns a structured array view
    # of zero-filled 8-byte strings whose first member holds the data, which
    # effectively pads each value with zero bytes on the right; otherwise
    # returns the input unchanged.
    # This feature is not well documented, but some SAS XPORT files
    # have 2-7 byte "truncated" floats.  To read these truncated
    # floats, pad them with zeros on the right to make 8 byte floats.
    #
    # References:
    # https://github.com/jcushman/xport/pull/3
    # The R "foreign" library

    if nbytes != 8:
        vec1 = np.zeros(len(vec), np.dtype("S8"))
        dtype = np.dtype(f"S{nbytes},S{8 - nbytes}")
        vec2 = vec1.view(dtype=dtype)
        vec2["f0"] = vec
        return vec2

    return vec


def _parse_float_vec(vec):
    # [FUNCTION DOC]
    # HELPER: vectorized conversion of IBM mainframe floating point (hex
    # exponent excess 64, fraction in the low bits, big-endian) to native
    # double precision. Input vec is an array of 8-byte values.
    # Steps: view each value as two big-endian 32-bit words; derive from the
    # leading fraction bits how many places (0 to 3) to shift so the fraction
    # is normalized to the IEEE form; shift the fraction words, carrying the
    # low bits of the first word into the second; clear the implicit leading
    # bit; recompute the exponent from the IBM exponent (adjusted by 65 and
    # converted from base 16 to base 2) plus shift plus the IEEE bias, and keep
    # the sign bit; finally reassemble and return a native-endian float64
    # array. Precision may be lost relative to IBM format only in that IBM
    # carries up to 3 extra fraction bits. Input is not modified.
    """
    Parse a vector of float values representing IBM 8 byte floats into
    native 8 byte floats.
    """
    dtype = np.dtype(">u4,>u4")
    vec1 = vec.view(dtype=dtype)
    xport1 = vec1["f0"]
    xport2 = vec1["f1"]

    # Start by setting first half of ieee number to first half of IBM
    # number sans exponent
    ieee1 = xport1 & 0x00FFFFFF

    # The fraction bit to the left of the binary point in the ieee
    # format was set and the number was shifted 0, 1, 2, or 3
    # places. This will tell us how to adjust the ibm exponent to be a
    # power of 2 ieee exponent and how to shift the fraction bits to
    # restore the correct magnitude.
    shift = np.zeros(len(vec), dtype=np.uint8)
    shift[np.where(xport1 & 0x00200000)] = 1
    shift[np.where(xport1 & 0x00400000)] = 2
    shift[np.where(xport1 & 0x00800000)] = 3

    # shift the ieee number down the correct number of places then
    # set the second half of the ieee number to be the second half
    # of the ibm number shifted appropriately, ored with the bits
    # from the first half that would have been shifted in if we
    # could shift a double. All we are worried about are the low
    # order 3 bits of the first half since we're only shifting by
    # 1, 2, or 3.
    ieee1 >>= shift
    ieee2 = (xport2 >> shift) | ((xport1 & 0x00000007) << (29 + (3 - shift)))

    # clear the 1 bit to the left of the binary point
    ieee1 &= 0xFFEFFFFF

    # set the exponent of the ieee number to be the actual exponent
    # plus the shift count + 1023. Or this into the first half of the
    # ieee number. The ibm exponent is excess 64 but is adjusted by 65
    # since during conversion to ibm format the exponent is
    # incremented by 1 and the fraction bits left 4 positions to the
    # right of the radix point.  (had to add >> 24 because C treats &
    # 0x7f as 0x7f000000 and Python doesn't)
    ieee1 |= ((((((xport1 >> 24) & 0x7F) - 65) << 2) + shift + 1023) << 20) | (
        xport1 & 0x80000000
    )

    ieee = np.empty((len(ieee1),), dtype=">u4,>u4")
    ieee["f0"] = ieee1
    ieee["f1"] = ieee2
    ieee = ieee.view(dtype=">f8")
    ieee = ieee.astype("f8")

    return ieee


class XportReader(SASReader):
    # [CLASS DOC]
    # Reader for one SAS XPORT file (single member). Subclass of the abstract
    # SAS reader; supports iteration (each step yields a chunk) and use as a
    # context manager. The class documentation is assigned from a template.
    # State: the text encoding for character columns, the count of rows read so
    # far, the optional index column, the chunk size, the file handle bundle,
    # the underlying binary stream, and after header parsing the file info and
    # member info mappings, the list of field descriptors, the record length,
    # the stream position where records start, the total observation count,
    # the column names, and the structured dtype of one record.
    # Key methods: read (return rows), get_chunk, close, and the header parser.
    # Invariant: the stream is positioned at the next unread record between
    # reads.
    __doc__ = _xport_reader_doc

    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        index=None,
        encoding: str | None = "ISO-8859-1",
        chunksize: int | None = None,
        compression: CompressionOptions = "infer",
    ) -> None:
        # [FUNCTION DOC]
        # Opens the file in binary mode and parses its headers.
        # Parameters: filepath_or_buffer is a path, URL or binary buffer;
        # index is the optional column to set as index when reading; encoding
        # (default ISO-8859-1) decodes character columns, and none keeps them
        # as raw bytes; chunksize is the number of rows per iteration step
        # (none means one row per step); compression defaults to inference.
        # If header parsing raises, the handles are closed and the error
        # propagates. Raises value errors for malformed headers (see the header
        # parser).
        self._encoding = encoding
        self._lines_read = 0
        self._index = index
        self._chunksize = chunksize

        self.handles = get_handle(
            filepath_or_buffer,
            "rb",
            encoding=encoding,
            is_text=False,
            compression=compression,
        )
        self.filepath_or_buffer = self.handles.handle

        try:
            self._read_header()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes all file handles opened by this reader.
        self.handles.close()

    def _get_row(self):
        # [FUNCTION DOC]
        # HELPER: reads the next 80-byte card from the stream and returns it
        # decoded to text using the default decoding. Raises a decode error for
        # undecodable bytes; returns a shorter string at end of file.
        return self.filepath_or_buffer.read(80).decode()

    def _read_header(self) -> None:
        # [FUNCTION DOC]
        # HELPER: rewinds the stream and parses the file, member and variable
        # descriptor headers, leaving the stream at the first observation.
        # Reads, in order: the library header card; a card with prefix, version,
        # operating system and created time; a card with the modified time; the
        # member header and descriptor header cards (the first gives the
        # length of each variable descriptor, normally 140, sometimes 135);
        # two member info cards; the variable count card; the variable
        # descriptors padded to a multiple of 80 bytes; and the observation
        # header card.
        # Each descriptor is unpacked from a fixed binary layout into a
        # dictionary keyed by the field key list, its type code mapped to
        # numeric or char, and string members stripped. The record length is the
        # sum of the field lengths. Afterwards sets the file info, member info,
        # fields, record length, record start, observation count (via the
        # record counter), column names (decoded) and a structured dtype with
        # one fixed-length byte string per field.
        # Raises value errors when the first card is not the library header
        # (with a special message for compressed CPORT files), when the prefix
        # on the second card is wrong, when the member or observation headers
        # are missing; raises a type error when a numeric field width is not
        # between 2 and 8 (a value error would arguably fit better).
        # Raises a key error for an unknown variable type code.
        self.filepath_or_buffer.seek(0)

        # read file header
        line1 = self._get_row()
        if line1 != _correct_line1:
            if "**COMPRESSED**" in line1:
                # this was created with the PROC CPORT method and can't be read
                # https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/movefile/p1bm6aqp3fw4uin1hucwh718f6kp.htm
                raise ValueError(
                    "Header record indicates a CPORT file, which is not readable."
                )
            raise ValueError("Header record is not an XPORT file.")

        line2 = self._get_row()
        fif = [["prefix", 24], ["version", 8], ["OS", 8], ["_", 24], ["created", 16]]
        file_info = _split_line(line2, fif)
        if file_info["prefix"] != "SAS     SAS     SASLIB":
            raise ValueError("Header record has invalid prefix.")
        file_info["created"] = _parse_date(file_info["created"])
        self.file_info = file_info

        line3 = self._get_row()
        file_info["modified"] = _parse_date(line3[:16])

        # read member header
        header1 = self._get_row()
        header2 = self._get_row()
        headflag1 = header1.startswith(_correct_header1)
        headflag2 = header2 == _correct_header2
        if not (headflag1 and headflag2):
            raise ValueError("Member header not found")
        # usually 140, could be 135
        fieldnamelength = int(header1[-5:-2])

        # member info
        mem = [
            ["prefix", 8],
            ["set_name", 8],
            ["sasdata", 8],
            ["version", 8],
            ["OS", 8],
            ["_", 24],
            ["created", 16],
        ]
        member_info = _split_line(self._get_row(), mem)
        mem = [["modified", 16], ["_", 16], ["label", 40], ["type", 8]]
        member_info.update(_split_line(self._get_row(), mem))
        member_info["modified"] = _parse_date(member_info["modified"])
        member_info["created"] = _parse_date(member_info["created"])
        self.member_info = member_info

        # read field names
        types = {1: "numeric", 2: "char"}
        fieldcount = int(self._get_row()[54:58])
        datalength = fieldnamelength * fieldcount
        # round up to nearest 80
        if datalength % 80:
            datalength += 80 - datalength % 80
        fielddata = self.filepath_or_buffer.read(datalength)
        fields = []
        obs_length = 0
        while len(fielddata) >= fieldnamelength:
            # pull data for one field
            fieldbytes, fielddata = (
                fielddata[:fieldnamelength],
                fielddata[fieldnamelength:],
            )

            # rest at end gets ignored, so if field is short, pad out
            # to match struct pattern below
            fieldbytes = fieldbytes.ljust(140)

            fieldstruct = struct.unpack(">hhhh8s40s8shhh2s8shhl52s", fieldbytes)
            field = dict(zip(_fieldkeys, fieldstruct, strict=True))
            del field["_"]
            field["ntype"] = types[field["ntype"]]
            fl = field["field_length"]
            if field["ntype"] == "numeric" and ((fl < 2) or (fl > 8)):
                msg = f"Floating field width {fl} is not between 2 and 8."
                raise TypeError(msg)

            for k, v in field.items():
                try:
                    field[k] = v.strip()
                except AttributeError:
                    pass

            obs_length += field["field_length"]
            fields += [field]

        header = self._get_row()
        if not header == _correct_obs_header:
            raise ValueError("Observation header not found.")

        self.fields = fields
        self.record_length = obs_length
        self.record_start = self.filepath_or_buffer.tell()

        self.nobs = self._record_count()
        self.columns = [x["name"].decode() for x in self.fields]

        # Setup the dtype.
        dtypel = [
            ("s" + str(i), "S" + str(field["field_length"]))
            for i, field in enumerate(self.fields)
        ]
        dtype = np.dtype(dtypel)
        self._dtype = dtype

    def __next__(self) -> pd.DataFrame:
        # [FUNCTION DOC]
        # Iterator step: reads the next chunk, namely chunksize rows, or a
        # single row when no chunk size was set. Stops iteration (and closes the
        # reader) once all rows are consumed.
        return self.read(nrows=self._chunksize or 1)

    def _record_count(self) -> int:
        # [FUNCTION DOC]
        # HELPER: computes the number of observations from the file size.
        # Moves the stream to the end, takes the bytes after the record start
        # as the total record bytes, and warns (user-level warning attributed
        # to the caller) if that total is not a multiple of 80. If the record
        # length exceeds 80, returns total divided by record length (integer
        # division). Otherwise inspects the final 80-byte card as eight-byte
        # words, counts those that consist entirely of blanks, treats 8 bytes
        # per such word as trailing padding, and returns total minus padding
        # divided by the record length. Leaves the stream at the record start.
        """
        Get number of records in file.

        This is maybe suboptimal because we have to seek to the end of
        the file.

        Side effect: returns file position to record_start.
        """
        self.filepath_or_buffer.seek(0, 2)
        total_records_length = self.filepath_or_buffer.tell() - self.record_start

        if total_records_length % 80 != 0:
            warnings.warn(
                "xport file may be corrupted.",
                stacklevel=find_stack_level(),
            )

        if self.record_length > 80:
            self.filepath_or_buffer.seek(self.record_start)
            return total_records_length // self.record_length

        self.filepath_or_buffer.seek(-80, 2)
        last_card_bytes = self.filepath_or_buffer.read(80)
        last_card = np.frombuffer(last_card_bytes, dtype=np.uint64)

        # 8 byte blank
        ix = np.flatnonzero(last_card == 2314885530818453536)

        if len(ix) == 0:
            tail_pad = 0
        else:
            tail_pad = 8 * len(ix)

        self.filepath_or_buffer.seek(self.record_start)

        return (total_records_length - tail_pad) // self.record_length

    def get_chunk(self, size: int | None = None) -> pd.DataFrame:
        # [FUNCTION DOC]
        # Reads the next size rows as a data frame. When size is none the
        # configured chunk size is used, and if that is also none all remaining
        # rows are read. Delegates to the read method and so raises stop
        # iteration when no rows remain.
        # KNOWN ISSUE: the docstring says none reads the whole file, but the
        # code falls back to the configured chunk size first.
        """
        Reads lines from Xport file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    def _missing_double(self, vec):
        # [FUNCTION DOC]
        # HELPER: flags SAS missing-value encodings among 8-byte numeric
        # values. Input vec is an array of 8-byte values. A value is missing
        # when its first byte is an uppercase letter, an underscore or a period
        # and all remaining seven bytes are zero. Returns a boolean array of the
        # same length (true means missing).
        v = vec.view(dtype="u1,u1,u2,u4")
        miss = (v["f1"] == 0) & (v["f2"] == 0) & (v["f3"] == 0)
        miss1 = (
            ((v["f0"] >= 0x41) & (v["f0"] <= 0x5A))
            | (v["f0"] == 0x5F)
            | (v["f0"] == 0x2E)
        )
        miss &= miss1
        return miss

    def read(self, nrows: int | None = None) -> pd.DataFrame:
        # [FUNCTION DOC]
        # Reads the next observations and returns them as a data frame.
        # Parameter nrows is the maximum number of rows (none means all
        # observations in the file). The count actually read is capped by the
        # unread remainder.
        # Numeric columns: truncated widths are padded, SAS missing codes are
        # detected, IBM floats are converted to doubles and missing values set to
        # not-a-number. Character columns: trailing whitespace is stripped from
        # each value, and values are decoded with the configured encoding if one
        # is set (otherwise they remain bytes).
        # The index is a running integer range continuing from rows already read
        # when no index column was configured; otherwise the configured column is
        # made the index. Advances the rows-read counter.
        # Side effect and exception: if nothing remains to read, the reader is
        # closed and a stop iteration signal is raised.
        # KNOWN ISSUE: raising the iteration-stop signal from a plain read call
        # means reading an empty file or an exhausted reader fails with that
        # signal instead of returning an empty frame.
        """Read observations from SAS Xport file, returning as data frame.

        Parameters
        ----------
        nrows : int
            Number of rows to read from data file; if None, read whole
            file.

        Returns
        -------
        A DataFrame.
        """
        if nrows is None:
            nrows = self.nobs

        read_lines = min(nrows, self.nobs - self._lines_read)
        read_len = read_lines * self.record_length
        if read_len <= 0:
            self.close()
            raise StopIteration
        raw = self.filepath_or_buffer.read(read_len)
        data = np.frombuffer(raw, dtype=self._dtype, count=read_lines)

        df_data = {}
        for j, x in enumerate(self.columns):
            vec = data["s" + str(j)]
            ntype = self.fields[j]["ntype"]
            if ntype == "numeric":
                vec = _handle_truncated_float_vec(vec, self.fields[j]["field_length"])
                miss = self._missing_double(vec)
                v = _parse_float_vec(vec)
                v[miss] = np.nan
            elif self.fields[j]["ntype"] == "char":
                v = [y.rstrip() for y in vec]

                if self._encoding is not None:
                    v = [y.decode(self._encoding) for y in v]

            df_data.update({x: v})
        df = pd.DataFrame(df_data)

        if self._index is None:
            df.index = pd.Index(range(self._lines_read, self._lines_read + read_lines))
        else:
            df = df.set_index(self._index)

        self._lines_read += read_lines

        return df
