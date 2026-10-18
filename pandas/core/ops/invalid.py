"""
Templates for invalid operations.
"""

# [MODULE DOC]
# Helpers for operations that are invalid for the operand types.
# Public contents: invalid_comparison (result for comparisons between mismatched types) and make_invalid_op (factory for methods that always raise a type error). No import-time side effects.

from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        ArrayLike,
        Scalar,
        npt,
    )


def invalid_comparison(
    left: ArrayLike,
    right: ArrayLike | list | range | Scalar,
    op: Callable[[Any, Any], bool],
) -> npt.NDArray[np.bool_]:
    # [FUNCTION DOC]
    # Produce the result of a comparison between mismatched types, following Python 3 conventions.
    # Parameters: left is an array (its shape and dtype are used); right is a scalar, list, range or array and is only used for its type name; op is the comparison operator function.
    # For equality returns an all-False boolean array shaped like left; for inequality returns an all-True array; for any other operator raises a type error naming the left dtype and the right type name.
    """
    If a comparison has mismatched types and is not necessarily meaningful,
    follow python3 conventions by:

        - returning all-False for equality
        - returning all-True for inequality
        - raising TypeError otherwise

    Parameters
    ----------
    left : array-like
    right : scalar, array-like
    op : operator.{eq, ne, lt, le, gt}

    Raises
    ------
    TypeError : on inequality comparisons
    """
    if op is operator.eq:
        res_values = np.zeros(left.shape, dtype=bool)
    elif op is operator.ne:
        res_values = np.ones(left.shape, dtype=bool)
    else:
        typ = type(right).__name__
        raise TypeError(f"Invalid comparison between dtype={left.dtype} and {typ}")
    return res_values


def make_invalid_op(name: str) -> Callable[..., NoReturn]:
    # [FUNCTION DOC]
    # Factory returning a binary method that always fails.
    # Parameter: name is the operator name. The returned function (named after the operator) takes self and an optional other (ignored) and raises a type error saying the operation cannot be performed with this index type, naming the class of self. Returns that function.
    """
    Return a binary method that always raises a TypeError.

    Parameters
    ----------
    name : str

    Returns
    -------
    invalid_op : function
    """

    def invalid_op(self: object, other: object = None) -> NoReturn:
        # [FUNCTION DOC]
        # HELPER: the function built by the factory. Ignores other and always raises a type error mentioning the operator name and the class name of self; never returns.
        typ = type(self).__name__
        raise TypeError(f"cannot perform {name} with this index type: {typ}")

    invalid_op.__name__ = name
    return invalid_op
