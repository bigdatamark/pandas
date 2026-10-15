# [MODULE DOC]
# Low-level reshaping kernels used by pandas.core.reshape: unstack (long to wide scatter
# driven by a boolean mask) and explode (flatten list-like elements of an object array
# into a longer array with per-row counts).
cimport cython
from cython cimport Py_ssize_t
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

import numpy as np

cimport numpy as cnp
from libc.math cimport NAN

cnp.import_array()

from pandas._libs.dtypes cimport numeric_object_t
from pandas._libs.lib cimport c_is_list_like


@cython.wraparound(False)
@cython.boundscheck(False)
def unstack(const numeric_object_t[:, :] values, const uint8_t[:] mask,
            Py_ssize_t stride, Py_ssize_t length, Py_ssize_t width,
            numeric_object_t[:, :] new_values, uint8_t[:, :] new_mask) -> None:
    # [FUNCTION DOC]
    # Scatter rows of a compressed "long" 2-D array into a preallocated "wide" 2-D
    # array. Conceptually the target has length rows and stride*width columns; mask
    # (flattened length*width) marks which (row j, slot w) cells have data. For each
    # source column i in [0, stride): walk cells offset = j*width + w in row-major
    # order, counting missing cells (nulls); for present cells copy values[offset -
    # nulls, i] into new_values[j, i*width + w] and set new_mask[j, i*width + w] = 1.
    # Inputs: values - 2-D array (fused numeric or object) shaped (n_present, stride);
    # mask - 1-D uint8 of length length*width; stride - number of value columns; length
    # - number of output rows; width - number of unstacked levels; new_values - output
    # 2-D array (length, stride*width); new_mask - output uint8 2-D.
    # Outputs: None.
    # Exceptions: none checked (bounds checking disabled; bad shapes are undefined
    # behavior). Side effects: mutates new_values and new_mask; releases the GIL for
    # non-object dtypes.
    # KNOWN ISSUE: docstring labels new_values as np.ndarray[bool]; it actually has the
    # same dtype as values.
    """
    Transform long values to wide new_values.

    Parameters
    ----------
    values : typed ndarray
    mask : np.ndarray[bool]
    stride : int
    length : int
    width : int
    new_values : np.ndarray[bool]
        result array
    new_mask : np.ndarray[bool]
        result mask
    """
    cdef:
        Py_ssize_t i, j, w, nulls, s, offset

    with nogil(numeric_object_t is not object):
        for i in range(stride):

            nulls = 0
            for j in range(length):

                for w in range(width):

                    offset = j * width + w

                    if mask[offset]:
                        s = i * width + w
                        new_values[j, s] = values[offset - nulls, i]
                        new_mask[j, s] = 1
                    else:
                        nulls += 1


@cython.wraparound(False)
@cython.boundscheck(False)
def explode(ndarray[object, ndim=1] values):
    # [FUNCTION DOC]
    # Explode list-like entries of a 1-D object array. First pass counts output slots
    # per input element: len(v) for non-empty list-likes (strings are not list-like), 1
    # for empty list-likes and for scalars. Second pass fills an object array of total
    # size: elements of each list-like in order (converted via list()), NaN for an empty
    # list-like, and the scalar itself otherwise.
    # Inputs: values - 1-D object ndarray.
    # Outputs: tuple (result object ndarray, counts int64 ndarray of length
    # len(values)).
    # Exceptions: propagates errors from len()/list() on odd list-likes.
    # Side effects: none (allocates new arrays).
    """
    transform array list-likes to long form
    preserve non-list entries

    Parameters
    ----------
    values : ndarray[object]

    Returns
    -------
    ndarray[object]
        result
    ndarray[int64_t]
        counts
    """
    cdef:
        Py_ssize_t i, j, count, n
        object v
        ndarray[object] result
        ndarray[int64_t] counts

    # find the resulting len
    n = len(values)
    counts = np.zeros(n, dtype="int64")
    for i in range(n):
        v = values[i]

        if c_is_list_like(v, True):
            if len(v):
                counts[i] += len(v)
            else:
                # empty list-like, use a nan marker
                counts[i] += 1
        else:
            counts[i] += 1

    result = np.empty(counts.sum(), dtype="object")
    count = 0
    for i in range(n):
        v = values[i]

        if c_is_list_like(v, True):
            if len(v):
                v = list(v)
                for j in range(len(v)):
                    result[count] = v[j]
                    count += 1
            else:
                # empty list-like, use a nan marker
                result[count] = NAN
                count += 1
        else:
            # replace with the existing scalar
            result[count] = v
            count += 1
    return result, counts
