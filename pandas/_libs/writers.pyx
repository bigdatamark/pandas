# [MODULE DOC]
# Fast helpers for writers in pandas.io: chunked CSV row emission, JSON-records to
# JSON-lines conversion, maximum string length computation (Stata, PyTables) and nan_rep
# replacement for PyTables string columns. Defines the fused type pandas_string (str or
# bytes).
cimport cython
from cython cimport Py_ssize_t
import numpy as np

from cpython cimport (
    PyBytes_GET_SIZE,
    PyUnicode_GET_LENGTH,
)
from numpy cimport (
    ndarray,
    uint8_t,
)

ctypedef fused pandas_string:
    str
    bytes


@cython.boundscheck(False)
@cython.wraparound(False)
def write_csv_rows(
    list data,
    ndarray data_index,
    Py_ssize_t nlevels,
    ndarray cols,
    object writer
) -> None:
    # [FUNCTION DOC]
    # Write rows to a csv.writer in batches of N=100 rows. A pool of N row lists of
    # length nlevels + len(cols) is preallocated and reused cyclically. For each row j
    # in range(len(data_index)): if nlevels == 1 the first cell is data_index[j]; if
    # nlevels > 1 the first nlevels cells are list(data_index[j]) (a tuple of MultiIndex
    # values); if nlevels == 0 no index cells. Then cell for column i is data[i][j].
    # Every time the pool fills, writer.writerows(pool) is called; at the end the
    # remaining partial batch (rows[:(j+1) % N]) is written.
    # Inputs: data - list of per-column array-likes (each indexable by row); data_index
    # - ndarray of index values (or tuples); nlevels - number of index levels to write;
    # cols - ndarray of column labels (only its length is used); writer - object with a
    # writerows method.
    # Outputs: None. Exceptions: propagates writer errors; IndexError if data is shorter
    # than cols. Side effects: calls writer.writerows (I/O).
    # KNOWN ISSUE: when data_index is empty, j stays 0 so the final flush writes one
    # stale/blank row (rows[:1], all None); callers must avoid passing empty chunks.
    """
    Write the given data to the writer object, pre-allocating where possible
    for performance improvements.

    Parameters
    ----------
    data : list[ArrayLike]
    data_index : ndarray
    nlevels : int
    cols : ndarray
    writer : _csv.writer
    """
    # In crude testing, N>100 yields little marginal improvement
    cdef:
        Py_ssize_t i, j = 0, k = len(data_index), N = 100, ncols = len(cols)
        list rows

    # pre-allocate rows
    rows = [[None] * (nlevels + ncols) for _ in range(N)]

    if nlevels == 1:
        for j in range(k):
            row = rows[j % N]
            row[0] = data_index[j]
            for i in range(ncols):
                row[1 + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    elif nlevels > 1:
        for j in range(k):
            row = rows[j % N]
            row[:nlevels] = list(data_index[j])
            for i in range(ncols):
                row[nlevels + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    else:
        for j in range(k):
            row = rows[j % N]
            for i in range(ncols):
                row[i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)

    if j >= 0 and (j < N - 1 or (j % N) != N - 1):
        writer.writerows(rows[:((j + 1) % N)])


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_json_to_lines(arr: str) -> str:
    # [FUNCTION DOC]
    # Convert a JSON array of records (as produced by to_json(orient= "records")) into
    # JSON-lines by replacing top-level separator commas with newlines. Works on the
    # UTF-8 bytes: tracks whether inside a string (toggled by unescaped double quotes at
    # index > 0, with backslash escape tracking) and the curly-brace depth outside
    # strings; a comma at depth 0 outside quotes becomes a newline. A trailing newline
    # is appended. The caller strips the outer square brackets beforehand.
    # Inputs: arr - str. Outputs: str. Exceptions: UnicodeError is not expected since
    # only ASCII bytes are replaced. Side effects: none.
    # KNOWN ISSUE: square brackets are not tracked, so a top-level comma inside a nested
    # array that is not within curly braces would be split (not reachable for records
    # output, where every element is an object).
    """
    replace comma separated json with line feeds, paying special attention
    to quotes & brackets
    """
    cdef:
        Py_ssize_t i = 0, num_open_brackets_seen = 0, length
        bint in_quotes = False, is_escaping = False
        ndarray[uint8_t, ndim=1] narr
        unsigned char val, newline, comma, left_bracket, right_bracket, quote
        unsigned char backslash

    newline = ord("\n")
    comma = ord(",")
    left_bracket = ord("{")
    right_bracket = ord("}")
    quote = ord('"')
    backslash = ord("\\")

    narr = np.frombuffer(arr.encode("utf-8"), dtype="u1").copy()
    length = narr.shape[0]
    for i in range(length):
        val = narr[i]
        if val == quote and i > 0 and not is_escaping:
            in_quotes = ~in_quotes
        if val == backslash or is_escaping:
            is_escaping = ~is_escaping
        if val == comma:  # commas that should be \n
            if num_open_brackets_seen == 0 and not in_quotes:
                narr[i] = newline
        elif val == left_bracket:
            if not in_quotes:
                num_open_brackets_seen += 1
        elif val == right_bracket:
            if not in_quotes:
                num_open_brackets_seen -= 1

    return narr.tobytes().decode("utf-8") + "\n"  # GH:36888


# stata, pytables
@cython.boundscheck(False)
@cython.wraparound(False)
def max_len_string_array(pandas_string[:] arr) -> Py_ssize_t:
    # [FUNCTION DOC]
    # Return the maximum length among elements of a 1-D str or bytes array (characters
    # for str, bytes for bytes), using word_len. Returns 0 for an empty array.
    # Inputs: arr - 1-D memoryview of str or bytes objects (fused).
    # Outputs: Py_ssize_t maximum length. Exceptions: none beyond buffer acquisition.
    # Side effects: none.
    """
    Return the maximum size of elements in a 1-dim string array.
    """
    cdef:
        Py_ssize_t i, m = 0, wlen = 0, length = arr.shape[0]
        pandas_string val

    for i in range(length):
        val = arr[i]
        wlen = word_len(val)

        if wlen > m:
            m = wlen

    return m


cpdef inline Py_ssize_t word_len(object val):
    # [FUNCTION DOC]
    # Return the length of a single value: number of code points for str, number of
    # bytes for bytes, 0 for anything else.
    # Inputs: val - any object. Outputs: Py_ssize_t. Side effects: none.
    # KNOWN ISSUE: the docstring says "maximum length" but this is just the length of
    # one value.
    """
    Return the maximum length of a string or bytes value.
    """
    cdef:
        Py_ssize_t wlen = 0

    if isinstance(val, str):
        wlen = PyUnicode_GET_LENGTH(val)
    elif isinstance(val, bytes):
        wlen = PyBytes_GET_SIZE(val)

    return wlen

# ------------------------------------------------------------------
# PyTables Helpers


@cython.boundscheck(False)
@cython.wraparound(False)
def string_array_replace_from_nan_rep(
    ndarray[object, ndim=1] arr,
    object nan_rep,
) -> None:
    # [FUNCTION DOC]
    # Replace, in place, every element of an object array equal (==) to nan_rep with
    # np.nan. Used when reading PyTables string columns.
    # Inputs: arr - 1-D object ndarray; nan_rep - sentinel value.
    # Outputs: None. Exceptions: propagates errors from element __eq__ (e.g. an element
    # whose == returns an array would raise ValueError on truth testing).
    # Side effects: mutates arr.
    """
    Replace the values in the array with np.nan if they are nan_rep.
    """
    cdef:
        Py_ssize_t length = len(arr), i = 0

    for i in range(length):
        if arr[i] == nan_rep:
            arr[i] = np.nan
