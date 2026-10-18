# [MODULE DOC]
# HELPER: test-support module for the copy_view tests. Provides get_array, which
# extracts the underlying array of a Series, Index or DataFrame column without
# going through ordinary getitem (so no reference tracking is triggered).

from pandas import (
    Categorical,
    Index,
    Series,
)
from pandas.core.arrays import BaseMaskedArray


def get_array(obj, col=None):
    # [FUNCTION DOC]
    # HELPER: used throughout the copy_view tests to compare memory sharing.
    # Takes obj (an Index, a Series, or a DataFrame) and col (default none; the
    # column label, required for a DataFrame, optional for a Series whose name must
    # equal it if given). Returns the raw array: for an Index or Series its stored
    # values, for a frame the column array obtained by position. Masked arrays
    # return their data buffer, categoricals are returned as is, otherwise the
    # numpy array of an extension array if present, else the array itself.
    # Raises an assertion error if col is missing for a frame or the column location is
    # not a single integer (duplicate or missing labels). The existing docstring
    # says it is equivalent to getting the values of a column, which matches.
    """
    Helper method to get array for a DataFrame column or a Series.

    Equivalent of df[col].values, but without going through normal getitem,
    which triggers tracking references / CoW (and we might be testing that
    this is done by some other operation).
    """
    if isinstance(obj, Index):
        arr = obj._values
    elif isinstance(obj, Series) and (col is None or obj.name == col):
        arr = obj._values
    else:
        assert col is not None
        icol = obj.columns.get_loc(col)
        assert isinstance(icol, int)
        arr = obj._get_column_array(icol)
    if isinstance(arr, BaseMaskedArray):
        return arr._data
    elif isinstance(arr, Categorical):
        return arr
    return getattr(arr, "_ndarray", arr)
