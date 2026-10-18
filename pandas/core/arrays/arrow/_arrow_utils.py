# [MODULE DOC]
# Internal helper module holding one function that converts a primitive pyarrow
# array into a numpy data array plus validity mask by reading the raw buffers.
# Used by the masked-array construction from arrow data. Imports pyarrow at
# import time, so it requires pyarrow to be installed. No other side effects.
from __future__ import annotations

import numpy as np
import pyarrow


def pyarrow_array_to_numpy_and_mask(
    arr, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC]
    # HELPER: zero-copy-style conversion of arrow data for masked arrays.
    # Parameters: arr is a primitive pyarrow array (boolean arrays are not
    # supported because their data buffer is bit-packed); dtype is the numpy
    # dtype used to interpret the data buffer (coerced to a numpy dtype).
    # Returns a pair (data, mask): data is a numpy array over the arrow data
    # buffer sliced by the array offset and length, and mask is a boolean array
    # where true means valid (not missing). For a null-typed array the data is
    # uninitialized and the mask is all false. When there is no validity buffer
    # the mask is all true. The validity bitmap is expanded through a pyarrow
    # boolean array using the array's offset.
    # The data array may share memory with the arrow buffer (no copy is made).
    """
    Convert a primitive pyarrow.Array to a numpy array and boolean mask based
    on the buffers of the Array.

    At the moment pyarrow.BooleanArray is not supported.

    Parameters
    ----------
    arr : pyarrow.Array
    dtype : numpy.dtype

    Returns
    -------
    (data, mask)
        Tuple of two numpy arrays with the raw data (with specified dtype) and
        a boolean mask (validity mask, so False means missing)
    """
    dtype = np.dtype(dtype)

    if pyarrow.types.is_null(arr.type):
        # No initialization of data is needed since everything is null
        data = np.empty(len(arr), dtype=dtype)
        mask = np.zeros(len(arr), dtype=bool)
        return data, mask
    buflist = arr.buffers()
    # Since Arrow buffers might contain padding and the data might be offset,
    # the buffer gets sliced here before handing it to numpy.
    # See also https://github.com/pandas-dev/pandas/issues/40896
    offset = arr.offset * dtype.itemsize
    length = len(arr) * dtype.itemsize
    data_buf = buflist[1][offset : offset + length]
    data = np.frombuffer(data_buf, dtype=dtype)
    bitmask = buflist[0]
    if bitmask is not None:
        mask = pyarrow.BooleanArray.from_buffers(
            pyarrow.bool_(), len(arr), [None, bitmask], offset=arr.offset
        )
        mask = np.asarray(mask)
    else:
        mask = np.ones(len(arr), dtype=bool)
    return data, mask
