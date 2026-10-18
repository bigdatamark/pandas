# [MODULE DOC]
# HELPER: shared helper functions for the DataFrame tests. Provides assertion helpers for
# frames with mixed float or integer column dtypes, and a function to interleave frames.
# Imported by other frame test modules; no tests and no import-time side effects.
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import (
    DataFrame,
    concat,
)

if TYPE_CHECKING:
    from pandas._typing import AxisInt


def _check_mixed_float(df, dtype=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for frames having columns A to D with mixed float dtypes.
    # Parameters: df is the frame to check; dtype (default none) overrides expected dtypes:
    # a string applies to all four columns, a mapping overrides individual columns. Default
    # expectations are 32-bit float for A and B, 16-bit float for C and 64-bit float for D
    # (16-bit floats are the likeliest to get upcast to 32-bit). Each column whose expected
    # dtype is truthy must have exactly that dtype, else an assertion error is raised.
    # Returns nothing; does not mutate the frame.
    # float16 are most likely to be upcasted to float32
    dtypes = {"A": "float32", "B": "float32", "C": "float16", "D": "float64"}
    if isinstance(dtype, str):
        dtypes = {k: dtype for k, v in dtypes.items()}
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    if dtypes.get("A"):
        assert df.dtypes["A"] == dtypes["A"]
    if dtypes.get("B"):
        assert df.dtypes["B"] == dtypes["B"]
    if dtypes.get("C"):
        assert df.dtypes["C"] == dtypes["C"]
    if dtypes.get("D"):
        assert df.dtypes["D"] == dtypes["D"]


def _check_mixed_int(df, dtype=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for frames having columns A to D with mixed integer dtypes.
    # Parameters: df is the frame to check; dtype (default none) overrides expected dtypes:
    # a string applies to all four columns, a mapping overrides individual columns. Default
    # expectations are 32-bit signed int for A, 64-bit unsigned for B, 8-bit unsigned for C
    # and 64-bit signed int for D. Each column with a truthy expected dtype must match, else
    # an assertion error is raised. Returns nothing.
    dtypes = {"A": "int32", "B": "uint64", "C": "uint8", "D": "int64"}
    if isinstance(dtype, str):
        dtypes = {k: dtype for k, v in dtypes.items()}
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    if dtypes.get("A"):
        assert df.dtypes["A"] == dtypes["A"]
    if dtypes.get("B"):
        assert df.dtypes["B"] == dtypes["B"]
    if dtypes.get("C"):
        assert df.dtypes["C"] == dtypes["C"]
    if dtypes.get("D"):
        assert df.dtypes["D"] == dtypes["D"]


def zip_frames(frames: list[DataFrame], axis: AxisInt = 1) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: interleaves a list of frames. Parameters: frames is a list of DataFrames that
    # all share the first frame's index and columns; axis (default 1). With axis one, for each
    # column of the first frame the columns from every frame are collected in turn (column-major
    # order) and concatenated side by side, returning a frame with the repeated column labels.
    # For any other axis value, for each index label of the first frame the rows of every
    # frame are collected in turn and a new frame is built from those row Series.
    # Returns the new DataFrame; inputs are not mutated.
    """
    take a list of frames, zip them together under the
    assumption that these all have the first frames' index/columns.

    Returns
    -------
    new_frame : DataFrame
    """
    if axis == 1:
        columns = frames[0].columns
        zipped = [f.loc[:, c] for c in columns for f in frames]
        return concat(zipped, axis=1)
    else:
        index = frames[0].index
        zipped = [f.loc[i, :] for i in index for f in frames]
        return DataFrame(zipped)
