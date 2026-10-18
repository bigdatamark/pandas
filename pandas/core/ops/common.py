"""
Boilerplate functions used in defining binary operations.
"""

# [MODULE DOC]
# Boilerplate used when defining binary operator methods on pandas objects: a decorator that unpacks zero-dimensional inputs and defers to higher-priority classes, and helpers that pick the name attached to an operation result.
# Public contents: the decorator factory and the result-name function. Uses construction helpers to wrap list operands for extension arrays. No import-time side effects.

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na

from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)

from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    sanitize_array,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import F


def unpack_zerodim_and_defer(name: str) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # Decorator factory. Parameter name is the operator method name (for example an or-style or arithmetic name). Returns a decorator that wraps a binary method with the standard pandas pre-processing by delegating to the private implementation function.
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Parameters
    ----------
    name : str

    Returns
    -------
    decorator
    """

    def wrapper(method: F) -> F:
        # [FUNCTION DOC]
        # HELPER: the decorator returned by the factory. Takes a binary method and returns it wrapped for the stored operator name.
        return _unpack_zerodim_and_defer(method, name)

    return wrapper


def _unpack_zerodim_and_defer(method: F, name: str) -> F:
    # [FUNCTION DOC]
    # HELPER: wraps a binary operator method with pandas conventions.
    # Parameters: method is the binary method to wrap; name is the operator name, used to detect logical operators (and, or, xor and their reversed forms).
    # The wrapper preserves the method metadata. Returns the wrapped method.
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Ensure method returns NotImplemented when operating against "senior"
    classes.  Ensure zero-dimensional ndarrays are always unpacked.

    Parameters
    ----------
    method : binary method
    name : str

    Returns
    -------
    method
    """
    is_logical = name.strip("_") in ["or", "xor", "and", "ror", "rxor", "rand"]

    @wraps(method)
    def new_method(self, other):
        # [FUNCTION DOC]
        # HELPER: the wrapper body created by the private decorator.
        # Takes self and other. If other advertises a priority attribute that exceeds the priority of self, returns the not-implemented sentinel so the other operand handles the operation. Otherwise unpacks a zero-dimensional operand into a scalar. If self is an extension array, other is a plain list and the operation is not logical, other is converted to an array (via the sanitization routine) and datetime-like arrays are wrapped. Finally calls the original method and returns its result.
        prio = getattr(other, "__pandas_priority__", None)
        if prio is not None:
            if prio > self.__pandas_priority__:
                # e.g. other is DataFrame while self is Index/Series/EA
                return NotImplemented

        other = item_from_zerodim(other)
        if (
            isinstance(self, ABCExtensionArray)
            and isinstance(other, list)
            and not is_logical
        ):
            # See GH#62423
            other = sanitize_array(other, None)
            other = ensure_wrapped_if_datetimelike(other)

        return method(self, other)

    # error: Incompatible return value type (got "Callable[[Any, Any], Any]",
    # expected "F")
    return new_method  # type: ignore[return-value]


def get_op_result_name(left, right):
    # [FUNCTION DOC]
    # Choose the name to give the result of an operation between left and right.
    # Parameters: left is a Series or Index; right is any object. If right is a Series or Index, the name is determined by the matching helper (a shared name, or none); otherwise the name of left is used. Returns the name (usually a string, possibly none).
    """
    Find the appropriate name to pin to an operation result.  This result
    should always be either an Index or a Series.

    Parameters
    ----------
    left : {Series, Index}
    right : object

    Returns
    -------
    name : object
        Usually a string
    """
    if isinstance(right, (ABCSeries, ABCIndex)):
        name = _maybe_match_name(left, right)
    else:
        name = left.name
    return name


def _maybe_match_name(a, b):
    # [FUNCTION DOC]
    # HELPER: for the result-name function. Picks a name for the result of an operation between two objects.
    # If only one has a name attribute, its name is used. If both do: equal names win; matching missing-value names win; otherwise none. A type error during comparison (for example a missing-value marker) falls back to the missing-value match test; a value error (for example array-like names) yields none. Neither having a name yields none.
    """
    Try to find a name to attach to the result of an operation between
    a and b.  If only one of these has a `name` attribute, return that
    name.  Otherwise return a consensus name if they match or None if
    they have different names.

    Parameters
    ----------
    a : object
    b : object

    Returns
    -------
    name : str or None

    See Also
    --------
    pandas.core.common.consensus_name_attr
    """
    a_has = hasattr(a, "name")
    b_has = hasattr(b, "name")
    if a_has and b_has:
        try:
            if a.name == b.name:
                return a.name
            elif is_matching_na(a.name, b.name):
                # e.g. both are np.nan
                return a.name
            else:
                return None
        except TypeError:
            # pd.NA
            if is_matching_na(a.name, b.name):
                return a.name
            return None
        except ValueError:
            # e.g. np.int64(1) vs (np.int64(1), np.int64(2))
            return None
    elif a_has:
        return a.name
    elif b_has:
        return b.name
    return None
