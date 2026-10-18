# [MODULE DOC]
# Cython core of the SAS7BDAT reader (pandas.io.sas.sas7bdat). Provides a bounds-checked
# byte Buffer struct, the two SAS decompression codecs (RLE and RDC), fast
# subheader-signature lookup, and the Parser extension type which walks pages of a
# SAS7BDAT file and scatters each row's column bytes into the Python-level reader's
# preallocated chunks (raw 8-byte slots for numeric columns, bytes objects or NaN for
# string columns). Module-level C globals cache constants from
# pandas.io.sas.sas_constants (page types, subheader pointer offset, signature tables)
# which are filled at import time by _init_subheader_signatures.
# cython: language_level=3, initializedcheck=False
# cython: warn.maybe_uninitialized=True, warn.unused=True
from cython cimport Py_ssize_t
from libc.stddef cimport size_t
from libc.stdint cimport (
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.stdlib cimport (
    calloc,
    free,
)

import numpy as np

import pandas.io.sas.sas_constants as const


cdef object np_nan = np.nan


cdef struct Buffer:
    # [CLASS DOC]
    # Plain C struct: pointer to uint8 bytes plus their length. Does not own memory
    # unless created via buf_new (then must be released with buf_free). Passed by value.
    # All accesses go through buf_get/buf_set/buf_as_bytes which assert bounds.
    # Convenience wrapper for uint8_t data to allow fast and safe reads and writes.
    # We use this as a replacement for np.array(..., dtype=np.uint8) because it's
    # much slower to create NumPy arrays and we create Buffer instances many times
    # when reading a SAS7BDAT file (roughly once per row that is being read).
    uint8_t *data
    size_t length


cdef uint8_t buf_get(Buffer buf, size_t offset) except? 255:
    # [FUNCTION DOC]
    # HELPER: Read the byte at offset from buf.
    # Inputs: buf - Buffer; offset - byte index. Outputs: uint8 value.
    # Exceptions: AssertionError("Out of bounds read") if offset >= buf.length (255 is
    # used as the ambiguous error return value). Side effects: none.
    assert offset < buf.length, "Out of bounds read"
    return buf.data[offset]


cdef bint buf_set(Buffer buf, size_t offset, uint8_t value) except 0:
    # [FUNCTION DOC]
    # HELPER: Write value to buf at offset.
    # Inputs: buf - Buffer; offset - byte index; value - uint8. Outputs: True.
    # Exceptions: AssertionError("Out of bounds write") if offset >= buf.length.
    # Side effects: mutates the memory buf points to.
    assert offset < buf.length, "Out of bounds write"
    buf.data[offset] = value
    return True


cdef bytes buf_as_bytes(Buffer buf, size_t offset, size_t length):
    # [FUNCTION DOC]
    # HELPER: Copy length bytes starting at offset into a new bytes object.
    # Inputs: buf - Buffer; offset, length - byte range. Outputs: bytes.
    # Exceptions: AssertionError if offset + length > buf.length.
    # Side effects: allocates a Python bytes object.
    assert offset + length <= buf.length, "Out of bounds read"
    return buf.data[offset:offset+length]


cdef Buffer buf_new(size_t length) except *:
    # [FUNCTION DOC]
    # HELPER: Allocate a zero-initialized Buffer of length bytes (calloc).
    # Inputs: length - size in bytes. Outputs: Buffer owning the memory.
    # Exceptions: MemoryError if allocation fails. Side effects: heap allocation that
    # the caller must free with buf_free.
    cdef uint8_t *data = <uint8_t *>calloc(length, sizeof(uint8_t))
    if data is NULL:
        raise MemoryError(f"Failed to allocate {length} bytes")
    return Buffer(data, length)


cdef buf_free(Buffer buf):
    # [FUNCTION DOC]
    # HELPER: Free the memory of a Buffer created by buf_new (no-op if data is NULL).
    # Inputs: buf - Buffer. Outputs: None. Side effects: frees heap memory; the Buffer
    # must not be used afterwards.
    if buf.data != NULL:
        free(buf.data)

# rle_decompress decompresses data using a Run Length Encoding
# algorithm.  It is partially documented here:
#
# https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf
# Licence at LICENSES/SAS7BDAT_LICENSE
cdef int rle_decompress(Buffer inbuff, Buffer outbuff) except? 0:

    # [FUNCTION DOC]
    # HELPER: Decompress one SAS RLE-compressed row from inbuff into outbuff. Repeatedly
    # reads a control byte: the high nibble selects the command, the low nibble (e) is a
    # length component. Commands: 0x00 copy (next byte + 64 + e*256) literal bytes; 0x40
    # repeat the next byte (next byte + 18 + e*256) times; 0x60 write (e*256 + next byte
    # + 17) spaces; 0x70 same count of zero bytes; 0x80/0x90/ 0xA0/0xB0 copy e+1 / e+17
    # / e+33 / e+49 literal bytes; 0xC0 repeat next byte e+3 times; 0xD0 write e+2 '@'
    # (0x40) bytes; 0xE0 e+2 spaces; 0xF0 e+2 zero bytes. Continues until inbuff is
    # exhausted.
    # Inputs: inbuff - compressed Buffer; outbuff - destination Buffer (row length).
    # Outputs: number of bytes written (0 is an ambiguous error value).
    # Exceptions: ValueError for an unknown control byte; AssertionError on any out of
    # bounds read/write. Side effects: writes into outbuff.
    cdef:
        uint8_t control_byte, x
        int rpos = 0
        int i, nbytes, end_of_first_byte
        size_t ipos = 0
        Py_ssize_t _

    while ipos < inbuff.length:
        control_byte = buf_get(inbuff, ipos) & 0xF0
        end_of_first_byte = <int>(buf_get(inbuff, ipos) & 0x0F)
        ipos += 1

        if control_byte == 0x00:
            nbytes = <int>(buf_get(inbuff, ipos)) + 64 + end_of_first_byte * 256
            ipos += 1
            for _ in range(nbytes):
                buf_set(outbuff, rpos, buf_get(inbuff, ipos))
                rpos += 1
                ipos += 1
        elif control_byte == 0x40:
            # not documented
            nbytes = <int>(buf_get(inbuff, ipos)) + 18 + end_of_first_byte * 256
            ipos += 1
            for _ in range(nbytes):
                buf_set(outbuff, rpos, buf_get(inbuff, ipos))
                rpos += 1
            ipos += 1
        elif control_byte == 0x60:
            nbytes = end_of_first_byte * 256 + <int>(buf_get(inbuff, ipos)) + 17
            ipos += 1
            for _ in range(nbytes):
                buf_set(outbuff, rpos, 0x20)
                rpos += 1
        elif control_byte == 0x70:
            nbytes = end_of_first_byte * 256 + <int>(buf_get(inbuff, ipos)) + 17
            ipos += 1
            for _ in range(nbytes):
                buf_set(outbuff, rpos, 0x00)
                rpos += 1
        elif control_byte == 0x80:
            nbytes = end_of_first_byte + 1
            for i in range(nbytes):
                buf_set(outbuff, rpos, buf_get(inbuff, ipos + i))
                rpos += 1
            ipos += nbytes
        elif control_byte == 0x90:
            nbytes = end_of_first_byte + 17
            for i in range(nbytes):
                buf_set(outbuff, rpos, buf_get(inbuff, ipos + i))
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xA0:
            nbytes = end_of_first_byte + 33
            for i in range(nbytes):
                buf_set(outbuff, rpos, buf_get(inbuff, ipos + i))
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xB0:
            nbytes = end_of_first_byte + 49
            for i in range(nbytes):
                buf_set(outbuff, rpos, buf_get(inbuff, ipos + i))
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xC0:
            nbytes = end_of_first_byte + 3
            x = buf_get(inbuff, ipos)
            ipos += 1
            for _ in range(nbytes):
                buf_set(outbuff, rpos, x)
                rpos += 1
        elif control_byte == 0xD0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                buf_set(outbuff, rpos, 0x40)
                rpos += 1
        elif control_byte == 0xE0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                buf_set(outbuff, rpos, 0x20)
                rpos += 1
        elif control_byte == 0xF0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                buf_set(outbuff, rpos, 0x00)
                rpos += 1
        else:
            raise ValueError(f"unknown control byte: {control_byte}")

    return rpos


# rdc_decompress decompresses data using the Ross Data Compression algorithm:
#
# http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
cdef int rdc_decompress(Buffer inbuff, Buffer outbuff) except? 0:

    # [FUNCTION DOC]
    # HELPER: Decompress one Ross Data Compression (RDC) row. Input is a stream of
    # 16-bit big-endian control words, each governing the next 16 items: a 0 bit means
    # copy one literal byte; a 1 bit means a command byte whose high nibble is cmd and
    # low nibble cnt: cmd 0 short RLE (repeat next byte cnt+3 times); cmd 1 long RLE
    # (repeat a byte cnt + (next byte << 4) + 19 times); cmd 2 long pattern (copy
    # (next-next byte + 16) bytes from earlier output at distance cnt + 3 + (next byte
    # << 4)); cmd >= 3 short pattern (copy cmd bytes from earlier output at that
    # distance).
    # Inputs: inbuff - compressed Buffer; outbuff - destination Buffer.
    # Outputs: number of bytes written.
    # Exceptions: AssertionError on out of bounds reads/writes (e.g. back-reference
    # before the start of output). Side effects: writes into outbuff.
    cdef:
        uint8_t cmd
        uint16_t ctrl_bits = 0, ctrl_mask = 0, ofs, cnt
        int rpos = 0, k, ii
        size_t ipos = 0

    ii = -1

    while ipos < inbuff.length:
        ii += 1
        ctrl_mask = ctrl_mask >> 1
        if ctrl_mask == 0:
            ctrl_bits = ((<uint16_t>buf_get(inbuff, ipos) << 8) +
                         <uint16_t>buf_get(inbuff, ipos + 1))
            ipos += 2
            ctrl_mask = 0x8000

        if ctrl_bits & ctrl_mask == 0:
            buf_set(outbuff, rpos, buf_get(inbuff, ipos))
            ipos += 1
            rpos += 1
            continue

        cmd = (buf_get(inbuff, ipos) >> 4) & 0x0F
        cnt = <uint16_t>(buf_get(inbuff, ipos) & 0x0F)
        ipos += 1

        # short RLE
        if cmd == 0:
            cnt += 3
            for k in range(cnt):
                buf_set(outbuff, rpos + k, buf_get(inbuff, ipos))
            rpos += cnt
            ipos += 1

        # long RLE
        elif cmd == 1:
            cnt += <uint16_t>buf_get(inbuff, ipos) << 4
            cnt += 19
            ipos += 1
            for k in range(cnt):
                buf_set(outbuff, rpos + k, buf_get(inbuff, ipos))
            rpos += cnt
            ipos += 1

        # long pattern
        elif cmd == 2:
            ofs = cnt + 3
            ofs += <uint16_t>buf_get(inbuff, ipos) << 4
            ipos += 1
            cnt = <uint16_t>buf_get(inbuff, ipos)
            ipos += 1
            cnt += 16
            for k in range(cnt):
                buf_set(outbuff, rpos + k, buf_get(outbuff, rpos - <int>ofs + k))
            rpos += cnt

        # short pattern
        else:
            ofs = cnt + 3
            ofs += <uint16_t>buf_get(inbuff, ipos) << 4
            ipos += 1
            for k in range(cmd):
                buf_set(outbuff, rpos + k, buf_get(outbuff, rpos - <int>ofs + k))
            rpos += cmd

    return rpos


cdef enum ColumnTypes:
    column_type_decimal = 1
    column_type_string = 2


# Const aliases
assert len(const.page_meta_types) == 2
cdef:
    int page_meta_types_0 = const.page_meta_types[0]
    int page_meta_types_1 = const.page_meta_types[1]
    int page_mix_type = const.page_mix_type
    int page_data_type = const.page_data_type
    int subheader_pointers_offset = const.subheader_pointers_offset

    # Copy of subheader_signature_to_index that allows for much faster lookups.
    # Lookups are done in get_subheader_index. The C structures are initialized
    # in _init_subheader_signatures().
    uint32_t subheader_signatures_32bit[13]
    int subheader_indices_32bit[13]
    uint64_t subheader_signatures_64bit[17]
    int subheader_indices_64bit[17]
    int data_subheader_index = const.SASIndex.data_subheader_index


def _init_subheader_signatures():
    # [FUNCTION DOC]
    # HELPER: Populate the module-level C arrays of 4-byte and 8-byte subheader
    # signatures (reinterpreted as native-endian uint32/uint64) and their SASIndex
    # values from const.subheader_signature_to_index. Called once at import.
    # Inputs: none. Outputs: None.
    # Exceptions: AssertionError if the constants do not contain exactly 13 4-byte and
    # 17 8-byte signatures. Side effects: mutates module-level C globals.
    subheaders_32bit = [
        (sig, idx)
        for sig, idx in const.subheader_signature_to_index.items()
        if len(sig) == 4
    ]
    subheaders_64bit = [
        (sig, idx)
        for sig, idx in const.subheader_signature_to_index.items()
        if len(sig) == 8
    ]
    assert len(subheaders_32bit) == 13
    assert len(subheaders_64bit) == 17
    assert len(const.subheader_signature_to_index) == 13 + 17
    for i, (signature, idx) in enumerate(subheaders_32bit):
        subheader_signatures_32bit[i] = (<uint32_t *><char *>signature)[0]
        subheader_indices_32bit[i] = idx
    for i, (signature, idx) in enumerate(subheaders_64bit):
        subheader_signatures_64bit[i] = (<uint64_t *><char *>signature)[0]
        subheader_indices_64bit[i] = idx


_init_subheader_signatures()


def get_subheader_index(bytes signature):
    # [FUNCTION DOC]
    # Map a 4- or 8-byte subheader signature to its SASIndex value by linear search over
    # the C signature tables (compares as native integers). Unknown signatures map to
    # data_subheader_index.
    # Inputs: signature - bytes of length 4 or 8. Outputs: int index.
    # Exceptions: AssertionError for other lengths. Side effects: none.
    """Fast version of 'subheader_signature_to_index.get(signature)'."""
    cdef:
        uint32_t sig32
        uint64_t sig64
        Py_ssize_t i
    assert len(signature) in (4, 8)
    if len(signature) == 4:
        sig32 = (<uint32_t *><char *>signature)[0]
        for i in range(len(subheader_signatures_32bit)):
            if subheader_signatures_32bit[i] == sig32:
                return subheader_indices_32bit[i]
    else:
        sig64 = (<uint64_t *><char *>signature)[0]
        for i in range(len(subheader_signatures_64bit)):
            if subheader_signatures_64bit[i] == sig64:
                return subheader_indices_64bit[i]

    return data_subheader_index


cdef class Parser:

    # [CLASS DOC]
    # Row reader bound to a Python SAS7BDATReader (self.parser). Caches the reader's
    # column metadata (lengths, offsets, types as decimal/string), output chunk arrays
    # (byte_chunk: uint8 2-D, 8 bytes per numeric value per row; string_chunk: object
    # 2-D), current page bytes and page metadata, and row counters. read() parses rows
    # into the chunks and writes counters back to the reader. Page fetching is delegated
    # to the Python reader's _read_next_page.
    cdef:
        int column_count
        int64_t[:] lengths
        int64_t[:] offsets
        int64_t[:] column_types
        uint8_t[:, :] byte_chunk
        object[:, :] string_chunk
        uint8_t *cached_page
        int cached_page_len
        int current_row_on_page_index
        int current_page_block_count
        int current_page_data_subheader_pointers_len
        int current_page_subheaders_count
        int current_row_in_chunk_index
        int current_row_in_file_index
        bint blank_missing
        int header_length
        int row_length
        int bit_offset
        int subheader_pointer_length
        int current_page_type
        bint is_little_endian
        int (*decompress)(Buffer, Buffer) except? 0
        object parser

    def __init__(self, object parser):
        # [FUNCTION DOC]
        # Initialize from a SAS7BDATReader: copy column metadata, chunk arrays, row
        # length, page bit offset, subheader pointer length and byte order; load the
        # current page state; map column type codes b"d"/b"s" to decimal/string; choose
        # the decompressor (RLE, RDC or none) from parser.compression; copy row
        # counters.
        # Inputs: parser - SAS7BDATReader with header already parsed.
        # Outputs: None. Exceptions: ValueError for an unknown column type.
        # Side effects: mutates self; keeps a reference to parser's cached page bytes.
        cdef:
            int j
            char[:] column_types

        self.parser = parser
        self.blank_missing = parser.blank_missing
        self.header_length = self.parser.header_length
        self.column_count = parser.column_count
        self.lengths = parser.column_data_lengths()
        self.offsets = parser.column_data_offsets()
        self.byte_chunk = parser._byte_chunk
        self.string_chunk = parser._string_chunk
        self.row_length = parser.row_length
        self.bit_offset = self.parser._page_bit_offset
        self.subheader_pointer_length = self.parser._subheader_pointer_length
        self.is_little_endian = parser.byte_order == "<"
        self.column_types = np.empty(self.column_count, dtype="int64")

        # page indicators
        self.update_next_page()

        column_types = parser.column_types()

        # map column types
        for j in range(self.column_count):
            if column_types[j] == b"d":
                self.column_types[j] = column_type_decimal
            elif column_types[j] == b"s":
                self.column_types[j] = column_type_string
            else:
                raise ValueError(f"unknown column type: {self.parser.columns[j].ctype}")

        # compression
        if parser.compression == const.rle_compression:
            self.decompress = rle_decompress
        elif parser.compression == const.rdc_compression:
            self.decompress = rdc_decompress
        else:
            self.decompress = NULL

        # update to current state of the parser
        self.current_row_in_chunk_index = parser._current_row_in_chunk_index
        self.current_row_in_file_index = parser._current_row_in_file_index
        self.current_row_on_page_index = parser._current_row_on_page_index

    def read(self, int nrows):
        # [FUNCTION DOC]
        # Read up to nrows rows (stopping at end of file) into the chunk arrays, then
        # copy the page/chunk/file row counters back onto the Python parser.
        # Inputs: nrows - number of rows. Outputs: None.
        # Exceptions: propagates parsing errors. Side effects: mutates parser state,
        # chunk arrays and file position.
        cdef:
            bint done
            Py_ssize_t _

        for _ in range(nrows):
            done = self.readline()
            if done:
                break

        # update the parser
        self.parser._current_row_on_page_index = self.current_row_on_page_index
        self.parser._current_row_in_chunk_index = self.current_row_in_chunk_index
        self.parser._current_row_in_file_index = self.current_row_in_file_index

    cdef bint read_next_page(self) except? True:
        # [FUNCTION DOC]
        # HELPER: Ask the Python parser to read the next page. If at EOF, clears
        # cached_page; otherwise refreshes the cached page state.
        # Outputs: True if no more pages. Exceptions: propagates reader errors.
        # Side effects: file I/O, mutates self.
        cdef bint done

        done = self.parser._read_next_page()
        if done:
            self.cached_page = NULL
        else:
            self.update_next_page()
        return done

    cdef update_next_page(self):
        # [FUNCTION DOC]
        # HELPER: Copy the Python parser's current page info into C fields: page bytes
        # pointer and length, page type, block count, number of data subheader pointers
        # and subheader count; reset row-on-page index to 0.
        # Outputs: None. Side effects: mutates self; the raw pointer borrows the
        # parser's bytes object (valid only while the parser keeps it).
        # update data for the current page

        self.cached_page = <uint8_t *>self.parser._cached_page
        self.cached_page_len = len(self.parser._cached_page)
        self.current_row_on_page_index = 0
        self.current_page_type = self.parser._current_page_type
        self.current_page_block_count = self.parser._current_page_block_count
        self.current_page_data_subheader_pointers_len = len(
            self.parser._current_page_data_subheader_pointers
        )
        self.current_page_subheaders_count = self.parser._current_page_subheaders_count

    cdef bint readline(self) except? True:

        # [FUNCTION DOC]
        # HELPER: Read a single data row. If no page is cached, seeks to the end of the
        # header and reads a page. Then by page type: meta pages - rows come from the
        # data subheader pointer list (offset, length); when exhausted read the next
        # page and retry; mix pages - rows follow the subheader pointers, aligned to 8
        # bytes, at current_row_on_page_index * row_length; reads the next page once the
        # row index reaches min(row_count, mix_page_row_count); data pages - rows follow
        # the pointer area at fixed stride; next page when the index reaches the block
        # count.
        # Outputs: True if end of file reached (no row read), else False.
        # Exceptions: ValueError for an unknown page type; errors from row processing.
        # Side effects: mutates self, chunk arrays, file position.
        cdef:
            int offset, length, bit_offset, align_correction
            int subheader_pointer_length, mn
            bint done, flag

        bit_offset = self.bit_offset
        subheader_pointer_length = self.subheader_pointer_length

        # If there is no page, go to the end of the header and read a page.
        if self.cached_page == NULL:
            self.parser._path_or_buf.seek(self.header_length)
            done = self.read_next_page()
            if done:
                return True

        # Loop until a data row is read
        while True:
            if self.current_page_type in (page_meta_types_0, page_meta_types_1):
                flag = self.current_row_on_page_index >=\
                    self.current_page_data_subheader_pointers_len
                if flag:
                    done = self.read_next_page()
                    if done:
                        return True
                    continue
                offset, length = self.parser._current_page_data_subheader_pointers[
                    self.current_row_on_page_index
                ]
                self.process_byte_array_with_data(offset, length)
                return False
            elif self.current_page_type == page_mix_type:
                align_correction = (
                    bit_offset
                    + subheader_pointers_offset
                    + self.current_page_subheaders_count * subheader_pointer_length
                )
                align_correction = align_correction % 8
                offset = bit_offset + align_correction
                offset += subheader_pointers_offset
                offset += self.current_page_subheaders_count * subheader_pointer_length
                offset += self.current_row_on_page_index * self.row_length
                self.process_byte_array_with_data(offset, self.row_length)
                mn = min(self.parser.row_count, self.parser._mix_page_row_count)
                if self.current_row_on_page_index == mn:
                    done = self.read_next_page()
                    if done:
                        return True
                return False
            elif self.current_page_type == page_data_type:
                self.process_byte_array_with_data(
                    bit_offset
                    + subheader_pointers_offset
                    + self.current_row_on_page_index * self.row_length,
                    self.row_length,
                )
                flag = self.current_row_on_page_index == self.current_page_block_count
                if flag:
                    done = self.read_next_page()
                    if done:
                        return True
                return False
            else:
                raise ValueError(f"unknown page type: {self.current_page_type}")

    cdef void process_byte_array_with_data(self, int offset, int length) except *:

        # [FUNCTION DOC]
        # HELPER: Decode one row located at offset/length in the cached page. If a
        # decompressor is set and length < row_length, the row is decompressed into a
        # temporary row_length buffer (must decompress to exactly row_length). For each
        # column (stopping at the first zero-length column): numeric columns copy their
        # lngt bytes into byte_chunk[numeric col index, 8*row ..] right-aligned for
        # little endian (left-aligned for big endian), leaving the rest of the 8-byte
        # slot as is; string columns strip trailing NUL/space bytes and store bytes, or
        # NaN when empty and blank_missing is set, into string_chunk[string col index,
        # row]. Increments the page, chunk and file row counters.
        # Inputs: offset, length - byte range within the cached page.
        # Outputs: None. Exceptions: AssertionError for out of bounds; ValueError if the
        # decompressed length is wrong. Side effects: mutates chunk arrays and counters;
        # allocates and frees a temporary buffer.
        # KNOWN ISSUE: the temporary decompression buffer is not freed if an exception
        # is raised after buf_new (length mismatch or out of bounds assertion), leaking
        # memory.
        # KNOWN ISSUE: a zero-length column causes a break, skipping all later columns
        # rather than just that column.
        cdef:
            Py_ssize_t j
            int s, k, m, jb, js, current_row, rpos
            int64_t lngt, start, ct
            Buffer source, decompressed_source
            int64_t[:] column_types
            int64_t[:] lengths
            int64_t[:] offsets
            uint8_t[:, :] byte_chunk
            object[:, :] string_chunk
            bint compressed

        assert offset + length <= self.cached_page_len, "Out of bounds read"
        source = Buffer(&self.cached_page[offset], length)

        compressed = self.decompress != NULL and length < self.row_length
        if compressed:
            decompressed_source = buf_new(self.row_length)
            rpos = self.decompress(source, decompressed_source)
            if rpos != self.row_length:
                raise ValueError(
                    f"Expected decompressed line of length {self.row_length} bytes "
                    f"but decompressed {rpos} bytes"
                )
            source = decompressed_source

        current_row = self.current_row_in_chunk_index
        column_types = self.column_types
        lengths = self.lengths
        offsets = self.offsets
        byte_chunk = self.byte_chunk
        string_chunk = self.string_chunk
        s = 8 * self.current_row_in_chunk_index
        js = 0
        jb = 0
        for j in range(self.column_count):
            lngt = lengths[j]
            if lngt == 0:
                break
            start = offsets[j]
            ct = column_types[j]
            if ct == column_type_decimal:
                # decimal
                if self.is_little_endian:
                    m = s + 8 - lngt
                else:
                    m = s
                for k in range(lngt):
                    byte_chunk[jb, m + k] = buf_get(source, start + k)
                jb += 1
            elif column_types[j] == column_type_string:
                # string
                # Skip trailing whitespace. This is equivalent to calling
                # .rstrip(b"\x00 ") but without Python call overhead.
                while lngt > 0 and buf_get(source, start + lngt - 1) in b"\x00 ":
                    lngt -= 1
                if lngt == 0 and self.blank_missing:
                    string_chunk[js, current_row] = np_nan
                else:
                    string_chunk[js, current_row] = buf_as_bytes(source, start, lngt)
                js += 1

        self.current_row_on_page_index += 1
        self.current_row_in_chunk_index += 1
        self.current_row_in_file_index += 1

        if compressed:
            buf_free(decompressed_source)
