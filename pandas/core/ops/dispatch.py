"""
Functions for defining unary operations.
"""

# [MODULE DOC]
# Single predicate deciding whether a binary operation on arrays should be handled by an extension array implementation.
# KNOWN ISSUE: the module docstring says it defines unary operations, but its only content is a predicate for binary-operation dispatch.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.dtypes.generic import ABCExtensionArray

if TYPE_CHECKING:
    from pandas._typing import ArrayLike


def should_extension_dispatch(left: ArrayLike, right: Any) -> bool:
    # [FUNCTION DOC]
    # Returns True if either left (a numpy or extension array) or right (any object) is an extension array, meaning the operation should dispatch to the extension array implementation; otherwise False. Pure function.
    """
    Identify cases where Series operation should dispatch to ExtensionArray method.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object

    Returns
    -------
    bool
    """
    return isinstance(left, ABCExtensionArray) or isinstance(right, ABCExtensionArray)
