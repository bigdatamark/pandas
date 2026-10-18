"""
Helpers for sharing tests between DataFrame/Series
"""

# [MODULE DOC]
# HELPER: test support module with two tiny helpers that let tests be shared
# between frame and series parametrizations: get_dtype and get_obj (both
# re-exported by the testing package). No import-time side effects beyond
# imports.
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


def get_dtype(obj) -> DtypeObj:
    # [FUNCTION DOC]
    # HELPER: test helper. Parameter obj is a frame or a series (or other object
    # with a dtype attribute). Returns the dtype of a series-like object; for a
    # frame, returns the dtype of its first column (the code assumes the frame
    # has a single column and does not check; it fails on a frame with no
    # columns).
    if isinstance(obj, DataFrame):
        # Note: we are assuming only one column
        return obj.dtypes.iat[0]
    else:
        return obj.dtype


def get_obj(df: DataFrame, klass):
    # [FUNCTION DOC]
    # HELPER: test helper. Parameters: df (a frame) and klass (the class a test
    # is parametrized with). Returns df unchanged when klass is the frame class;
    # otherwise returns its first column (by position) as a series.
    """
    For sharing tests using frame_or_series, either return the DataFrame
    unchanged or return it's first column as a Series.
    """
    if klass is DataFrame:
        return df
    return df._ixs(0, axis=1)
