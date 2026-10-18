# [MODULE DOC]
# Small helpers shared by the computation package: byte-string decoding and
# a dtype-promotion wrapper that works around numpy's argument-count limit.
# Public contents: ensure_decoded and result_type_many. Used by ops, expr and
# pytables. No import-time side effects.
from __future__ import annotations

from functools import reduce

import numpy as np

from pandas._config import get_option


def ensure_decoded(s) -> str:
    # [FUNCTION DOC]
    # Decodes a byte string to text using the configured display encoding.
    # Input s: any object. If it is a bytes object or numpy bytes scalar it is
    # decoded with the encoding option (display.encoding); any other object is
    # returned unchanged.
    # KNOWN ISSUE: the return annotation claims text, but non-bytes inputs are
    # passed through unchanged and may be of any type.
    # Raises a decoding error if the bytes are invalid in that encoding.
    """
    If we have bytes, decode them to unicode.
    """
    if isinstance(s, (np.bytes_, bytes)):
        s = s.decode(get_option("display.encoding"))
    return s


def result_type_many(*arrays_and_dtypes):
    # [FUNCTION DOC]
    # Computes the common result dtype of any number of arrays and dtypes.
    # Inputs: a variadic list of numpy arrays, numpy dtypes or extension
    # dtypes. Returns a numpy dtype, or an extension dtype when extension
    # dtypes participate.
    # Strategy: first try the numpy promotion routine directly. If numpy
    # raises a value error (too many operands for its fixed operand limit),
    # fold the operands pairwise with a reduction. If numpy raises a type error
    # (extension dtypes present), split the operands into extension and
    # non-extension groups, promote the non-extension group with numpy (falling
    # back to a pairwise reduction on a value error), then combine the numpy
    # result with the extension dtypes using the generic common-type finder;
    # with no non-extension operands only the extension dtypes are combined.
    # KNOWN ISSUE: in the nested value-error fallback the reduction runs over
    # all operands (including the extension ones) rather than only the
    # non-extension group, which would raise a type error again.
    """
    Wrapper around numpy.result_type which overcomes the NPY_MAXARGS (32)
    argument limit.
    """
    try:
        return np.result_type(*arrays_and_dtypes)
    except ValueError:
        # we have > NPY_MAXARGS terms in our expression
        return reduce(np.result_type, arrays_and_dtypes)
    except TypeError:
        from pandas.core.dtypes.cast import find_common_type
        from pandas.core.dtypes.common import is_extension_array_dtype

        arr_and_dtypes = list(arrays_and_dtypes)
        ea_dtypes, non_ea_dtypes = [], []
        for arr_or_dtype in arr_and_dtypes:
            if is_extension_array_dtype(arr_or_dtype):
                ea_dtypes.append(arr_or_dtype)
            else:
                non_ea_dtypes.append(arr_or_dtype)

        if non_ea_dtypes:
            try:
                np_dtype = np.result_type(*non_ea_dtypes)
            except ValueError:
                np_dtype = reduce(np.result_type, arrays_and_dtypes)
            return find_common_type([*ea_dtypes, np_dtype])

        return find_common_type(ea_dtypes)
