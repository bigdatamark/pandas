# [MODULE DOC]
# Fast primitives to read a fixed-width number from a bytes object at a byte offset,
# optionally reversing its byte order, avoiding struct.unpack overhead. Used heavily by
# the SAS7BDAT reader (pandas/io/sas). Byte swapping uses compiler intrinsics
# (_byteswap_* on MSVC, __builtin_bswap* elsewhere) declared at the bottom. Values are
# read via memcpy into native-order integers, so the 'byteswap' flag means 'the data's
# endianness differs from the host's'.
"""
The following are faster versions of struct.unpack that avoid the overhead of Python
function calls.

In the SAS7BDAT parser, they may be called up to (n_rows * n_cols) times.
"""
from cython cimport Py_ssize_t
from libc.stdint cimport (
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.string cimport memcpy


def read_float_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    # [FUNCTION DOC]
    # Read a 4-byte IEEE-754 single-precision float from data at offset.
    # Inputs: data - bytes buffer; offset - byte offset of the value; byteswap - if
    # true, reverse the 4 bytes before reinterpreting them as a float.
    # Outputs: Python float (the C float widened to double).
    # Exceptions: AssertionError if offset + 4 is not strictly less than len(data).
    # Side effects: none.
    # KNOWN ISSUE: the bounds check uses strict less-than, so reading a value that ends
    # exactly at the end of data (offset + 4 == len(data)) is wrongly rejected; also no
    # check for negative offset. Same issue in all read_*_with_byteswap functions.
    cdef uint32_t value
    assert offset + <Py_ssize_t>sizeof(value) < len(data)
    cdef const void *ptr = <unsigned char*>(data) + offset
    memcpy(&value, ptr, sizeof(value))
    if byteswap:
        value = _byteswap4(value)

    cdef float res
    memcpy(&res, &value, sizeof(res))
    return res


def read_double_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    # [FUNCTION DOC]
    # Read an 8-byte IEEE-754 double from data at offset.
    # Inputs: data - bytes; offset - byte offset; byteswap - reverse the 8 bytes first.
    # Outputs: Python float.
    # Exceptions: AssertionError if offset + 8 >= len(data) (see KNOWN ISSUE on
    # read_float_with_byteswap). Side effects: none.
    cdef uint64_t value
    assert offset + <Py_ssize_t>sizeof(value) < len(data)
    cdef const void *ptr = <unsigned char*>(data) + offset
    memcpy(&value, ptr, sizeof(value))
    if byteswap:
        value = _byteswap8(value)

    cdef double res
    memcpy(&res, &value, sizeof(res))
    return res


def read_uint16_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    # [FUNCTION DOC]
    # Read an unsigned 16-bit integer from data at offset.
    # Inputs: data - bytes; offset - byte offset; byteswap - reverse the 2 bytes first.
    # Outputs: Python int in [0, 65535].
    # Exceptions: AssertionError if offset + 2 >= len(data) (off-by-one, see above).
    # Side effects: none.
    cdef uint16_t res
    assert offset + <Py_ssize_t>sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap2(res)
    return res


def read_uint32_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    # [FUNCTION DOC]
    # Read an unsigned 32-bit integer from data at offset.
    # Inputs: data - bytes; offset - byte offset; byteswap - reverse the 4 bytes first.
    # Outputs: Python int.
    # Exceptions: AssertionError if offset + 4 >= len(data) (off-by-one, see above).
    # Side effects: none.
    cdef uint32_t res
    assert offset + <Py_ssize_t>sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap4(res)
    return res


def read_uint64_with_byteswap(bytes data, Py_ssize_t offset, bint byteswap):
    # [FUNCTION DOC]
    # Read an unsigned 64-bit integer from data at offset.
    # Inputs: data - bytes; offset - byte offset; byteswap - reverse the 8 bytes first.
    # Outputs: Python int.
    # Exceptions: AssertionError if offset + 8 >= len(data) (off-by-one, see above).
    # Side effects: none.
    cdef uint64_t res
    assert offset + <Py_ssize_t>sizeof(res) < len(data)
    memcpy(&res, <const unsigned char*>(data) + offset, sizeof(res))
    if byteswap:
        res = _byteswap8(res)
    return res


# Byteswapping

cdef extern from *:
    """
    #ifdef _MSC_VER
        #define _byteswap2 _byteswap_ushort
        #define _byteswap4 _byteswap_ulong
        #define _byteswap8 _byteswap_uint64
    #else
        #define _byteswap2 __builtin_bswap16
        #define _byteswap4 __builtin_bswap32
        #define _byteswap8 __builtin_bswap64
    #endif
    """
    uint16_t _byteswap2(uint16_t)
    uint32_t _byteswap4(uint32_t)
    uint64_t _byteswap8(uint64_t)
