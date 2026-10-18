"""
transforms.py is for shape-preserving functions.
"""

# [MODULE DOC]
# Shape-preserving array transforms. Public contents: shift, which moves values
# along an axis and fills the vacated positions. Import-time side effects: none.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        Scalar,
    )


def shift(
    values: np.ndarray, periods: int, axis: AxisInt, fill_value: Scalar
) -> np.ndarray:
    # [FUNCTION DOC]
    # Shifts the values of a numpy array along an axis. Inputs: values (array),
    # periods (number of positions; positive shifts toward higher indices, negative
    # toward lower), axis (axis to shift along) and fill_value (assigned to the
    # vacated positions).
    # Behavior: zero periods or an empty array just returns a copy. Otherwise the
    # array is rolled circularly (a Fortran-ordered input is transposed first so the
    # roll sees C order, and transposed back after), then the wrapped-around
    # leading (or trailing) slice is overwritten with fill_value. Returns a new
    # array; the input is not modified. A shift larger than the axis length fills
    # the whole axis.
    new_values = values

    if periods == 0 or values.size == 0:
        return new_values.copy()

    # make sure array sent to np.roll is c_contiguous
    f_ordered = values.flags.f_contiguous
    if f_ordered:
        new_values = new_values.T
        axis = new_values.ndim - axis - 1

    if new_values.size:
        new_values = np.roll(
            new_values,
            np.intp(periods),
            axis=axis,
        )

    axis_indexer = [slice(None)] * values.ndim
    if periods > 0:
        axis_indexer[axis] = slice(None, periods)
    else:
        axis_indexer[axis] = slice(periods, None)
    new_values[tuple(axis_indexer)] = fill_value

    # restore original order
    if f_ordered:
        new_values = new_values.T

    return new_values
