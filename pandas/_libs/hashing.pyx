# [MODULE DOC]
# pandas._libs.hashing: keyed hashing of object arrays with SipHash-2-4
# (2 compression rounds, 4 finalisation rounds, 64-bit output), used by
# pandas.util.hash_array / hash_pandas_object for object dtype. Each element is
# turned into a byte string and hashed with a caller-supplied 16-byte key.
# Translated from the reference implementation
# at https://github.com/veorq/SipHash

cimport cython
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

from numpy cimport (
    import_array,
    ndarray,
    uint8_t,
    uint64_t,
)

import_array()

from pandas._libs.util cimport is_nan


@cython.boundscheck(False)
def hash_object_array(
    ndarray[object, ndim=1] arr, str key, str encoding="utf8"
) -> np.ndarray[np.uint64]:
    # [FUNCTION DOC]
    # Hash each element of a 1-D object array to a uint64 with SipHash-2-4.
    # Element to bytes: bytes as is; str encoded with encoding; None or float/complex
    # NaN via str(val) encoded ("None"/"nan"); tuples via str(val) encoded after
    # checking hash(val) succeeds (so unhashable members raise). The byte strings are
    # kept alive in a list while raw pointers and lengths are collected into
    # malloc'ed arrays, then all hashes are computed without the GIL.
    # Inputs: arr - 1-D object ndarray; key - str whose encoding is exactly 16 bytes;
    # encoding - codec name for key and str elements (default utf8).
    # Outputs: 1-D uint64 ndarray of length len(arr).
    # Exceptions: ValueError if the encoded key is not 16 bytes; TypeError for any
    # other element type (e.g. int, pd.NA, NaT) or unhashable tuples;
    # UnicodeEncodeError from encoding; MemoryError if allocation fails.
    # Side effects: temporary malloc/free; releases the GIL for hashing.
    # KNOWN ISSUE: the docstring says TypeError is raised for mixed types, but any
    # single non-str/bytes/null/tuple element raises, and bytes/tuples are accepted
    # though undocumented.
    # KNOWN ISSUE: vecs and lens are freed only on the explicit TypeError path;
    # exceptions from encode() or hash(val), or failure of the second malloc, leak
    # the already allocated buffer(s).
    """
    Parameters
    ----------
    arr : 1-d object ndarray of objects
    key : hash key, must be 16 byte len encoded
    encoding : encoding for key & arr, default to 'utf8'

    Returns
    -------
    1-d uint64 ndarray of hashes.

    Raises
    ------
    TypeError
        If the array contains mixed types.

    Notes
    -----
    Allowed values must be strings, or nulls
    mixed array types will raise TypeError.
    """
    cdef:
        Py_ssize_t i, n
        uint64_t[::1] result
        bytes data, k
        uint8_t *kb
        uint64_t *lens
        char **vecs
        char *cdata
        object val
        list data_list = []

    k = <bytes>key.encode(encoding)
    kb = <uint8_t *>k
    if len(k) != 16:
        raise ValueError(
            f"key should be a 16-byte string encoded, got {k} (len {len(k)})"
        )

    n = len(arr)

    # create an array of bytes
    vecs = <char **>malloc(n * sizeof(char *))
    if vecs is NULL:
        raise MemoryError()
    lens = <uint64_t*>malloc(n * sizeof(uint64_t))
    if lens is NULL:
        raise MemoryError()

    for i in range(n):
        val = arr[i]
        if isinstance(val, bytes):
            data = <bytes>val
        elif isinstance(val, str):
            data = <bytes>val.encode(encoding)
        elif val is None or is_nan(val):
            # null, stringify and encode
            data = <bytes>str(val).encode(encoding)

        elif isinstance(val, tuple):
            # GH#28969 we could have a tuple, but need to ensure that
            #  the tuple entries are themselves hashable before converting
            #  to str
            hash(val)
            data = <bytes>str(val).encode(encoding)
        else:
            free(vecs)
            free(lens)
            raise TypeError(
                f"{val} of type {type(val)} is not a valid type for hashing, "
                "must be string or null"
            )

        lens[i] = len(data)
        cdata = data

        # keep the references alive through the end of the
        # function
        data_list.append(data)
        vecs[i] = cdata

    result = np.empty(n, dtype=np.uint64)
    with nogil:
        for i in range(n):
            result[i] = low_level_siphash(<uint8_t *>vecs[i], lens[i], kb)

    free(vecs)
    free(lens)
    return result.base  # .base to retrieve underlying np.ndarray


cdef uint64_t _rotl(uint64_t x, uint64_t b) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Rotate the 64-bit value x left by b bits (0 < b < 64; b == 0 would
    # shift right by 64, undefined in C, but callers use 13..32 only).
    # Outputs: rotated value. nogil; no side effects.
    return (x << b) | (x >> (64 - b))


cdef uint64_t u8to64_le(uint8_t* p) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: Read 8 bytes at p as a little-endian unsigned 64-bit integer
    # (independent of host endianness). nogil; no side effects.
    return (<uint64_t>p[0] |
            <uint64_t>p[1] << 8 |
            <uint64_t>p[2] << 16 |
            <uint64_t>p[3] << 24 |
            <uint64_t>p[4] << 32 |
            <uint64_t>p[5] << 40 |
            <uint64_t>p[6] << 48 |
            <uint64_t>p[7] << 56)


cdef void _sipround(uint64_t* v0, uint64_t* v1,
                    uint64_t* v2, uint64_t* v3) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: One SipRound applied in place to the four 64-bit state words (pointed
    # to by v0..v3): add/rotate/xor mixing with rotations 13, 32, 16, 21, 17, 32 as
    # in the SipHash reference. Outputs: none. Side effects: mutates *v0..*v3; nogil.
    v0[0] += v1[0]
    v1[0] = _rotl(v1[0], 13)
    v1[0] ^= v0[0]
    v0[0] = _rotl(v0[0], 32)
    v2[0] += v3[0]
    v3[0] = _rotl(v3[0], 16)
    v3[0] ^= v2[0]
    v0[0] += v3[0]
    v3[0] = _rotl(v3[0], 21)
    v3[0] ^= v0[0]
    v2[0] += v1[0]
    v1[0] = _rotl(v1[0], 17)
    v1[0] ^= v2[0]
    v2[0] = _rotl(v2[0], 32)


@cython.cdivision(True)
cdef uint64_t low_level_siphash(uint8_t* data, size_t datalen,
                                uint8_t* key) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: SipHash-2-4 of datalen bytes at data under a 16-byte key.
    # Initialise state with the standard constants xored with the key halves k0/k1
    # (little-endian), compress each full 8-byte block (2 SipRounds each), build the
    # final block from the remaining 0-7 bytes with datalen mod 256 in the top
    # byte, compress it, xor 0xff into v2, run 4 SipRounds and return
    # v0 ^ v1 ^ v2 ^ v3.
    # Inputs: data - byte pointer; datalen - byte count; key - pointer to 16 bytes.
    # Outputs: 64-bit hash. No exceptions; nogil; no side effects.
    cdef uint64_t v0 = 0x736f6d6570736575ULL
    cdef uint64_t v1 = 0x646f72616e646f6dULL
    cdef uint64_t v2 = 0x6c7967656e657261ULL
    cdef uint64_t v3 = 0x7465646279746573ULL
    cdef uint64_t b
    cdef uint64_t k0 = u8to64_le(key)
    cdef uint64_t k1 = u8to64_le(key + 8)
    cdef uint64_t m
    cdef int i
    cdef uint8_t* end = data + datalen - (datalen % sizeof(uint64_t))
    cdef int left = datalen & 7
    cdef int cROUNDS = 2
    cdef int dROUNDS = 4

    b = (<uint64_t>datalen) << 56
    v3 ^= k1
    v2 ^= k0
    v1 ^= k1
    v0 ^= k0

    while (data != end):
        m = u8to64_le(data)
        v3 ^= m
        for i in range(cROUNDS):
            _sipround(&v0, &v1, &v2, &v3)
        v0 ^= m

        data += sizeof(uint64_t)

    for i in range(left-1, -1, -1):
        b |= (<uint64_t>data[i]) << (i * 8)

    v3 ^= b

    for i in range(cROUNDS):
        _sipround(&v0, &v1, &v2, &v3)

    v0 ^= b
    v2 ^= 0xff

    for i in range(dROUNDS):
        _sipround(&v0, &v1, &v2, &v3)

    b = v0 ^ v1 ^ v2 ^ v3

    return b
