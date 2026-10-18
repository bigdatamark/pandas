"""
Ops for masked arrays.
"""

# [MODULE DOC]
# Three-valued (Kleene) logical operations on boolean data with separate missing-value masks, used by masked boolean arrays.
# Public contents: kleene_or, kleene_xor, kleene_and, plus a small helper that rejects floating NaN operands. Operands may be arrays or scalars; at least one must be an array with a mask. No import-time side effects.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)

if TYPE_CHECKING:
    from pandas._typing import npt


def kleene_or(
    left: bool | np.ndarray | libmissing.NAType,
    right: bool | np.ndarray | libmissing.NAType,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Elementwise logical or under Kleene logic: the result is missing only where both are missing or one is missing and the other false; a missing value or'ed with true is true.
    # Parameters: left and right are boolean arrays, booleans, or the NA marker; left_mask and right_mask are boolean arrays flagging missing entries, or None when the corresponding operand is a scalar (at most one may be None).
    # If left_mask is None the operands are swapped (or is symmetric) so that left is always an array. Raises a type error if left is then not an array, and a value error if right is a floating NaN. Handles right being the NA marker, true, false, or an array with a mask.
    # Returns a tuple of the result values and the new mask. Inputs are not mutated.
    """
    Boolean ``or`` using Kleene logic.

    Values are NA where we have ``NA | NA`` or ``NA | False``.
    ``NA | True`` is considered True.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical or, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A | B == B | A
    if left_mask is None:
        return kleene_or(right, left, right_mask, left_mask)

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be an np.ndarray.")

    raise_for_nan(right, method="or")

    if right is libmissing.NA:
        result = left.copy()
    else:
        result = left | right

    if right_mask is not None:
        # output is unknown where (False & NA), (NA & False), (NA & NA)
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = (
            (left_false & right_mask)
            | (right_false & left_mask)
            | (left_mask & right_mask)
        )
    elif right is True:
        mask = np.zeros_like(left_mask)
    elif right is libmissing.NA:
        mask = (~left & ~left_mask) | left_mask
    else:
        # False
        mask = left_mask.copy()

    return result, mask


def kleene_xor(
    left: bool | np.ndarray | libmissing.NAType,
    right: bool | np.ndarray | libmissing.NAType,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Elementwise logical exclusive-or under Kleene logic: like the or version except that the result is missing whenever either side is missing (true with true gives false).
    # Parameters and exceptions as for the or version (operands swapped if left_mask is None; type error if left is not an array; value error for floating NaN right operand).
    # Returns a tuple of result values and mask. For an NA scalar, the values are all false and the mask is all true; otherwise the mask is the union of the masks (or a copy of the left mask for a scalar).
    """
    Boolean ``xor`` using Kleene logic.

    This is the same as ``or``, with the following adjustments

    * True, True -> False
    * True, NA   -> NA

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A ^ B == B ^ A
    if left_mask is None:
        return kleene_xor(right, left, right_mask, left_mask)

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be an np.ndarray.")

    raise_for_nan(right, method="xor")
    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        result = left ^ right

    if right_mask is None:
        if right is libmissing.NA:
            mask = np.ones_like(left_mask)
        else:
            mask = left_mask.copy()
    else:
        mask = left_mask | right_mask

    return result, mask


def kleene_and(
    left: bool | libmissing.NAType | np.ndarray,
    right: bool | libmissing.NAType | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    # [FUNCTION DOC]
    # Elementwise logical and under Kleene logic: the result is missing for missing and missing, or true and missing; a missing value and'ed with false is false.
    # Parameters and exceptions as for the or version (operands swapped if left_mask is None; type error if left is not an array; value error for floating NaN right operand).
    # Returns a tuple of result values and mask. A false scalar unmasks everything; an NA scalar yields masked entries wherever left is true or already masked; for array operands entries are unmasked where either side is definitely false.
    # KNOWN ISSUE: the docstring's Returns section says the result is of the logical xor, but this function computes and.
    """
    Boolean ``and`` using Kleene logic.

    Values are ``NA`` for ``NA & NA`` or ``True & NA``.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A & B == B & A
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be an np.ndarray.")
    raise_for_nan(right, method="and")

    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        result = left & right

    if right_mask is None:
        # Scalar `right`
        if right is libmissing.NA:
            mask = (left & ~left_mask) | left_mask

        else:
            mask = left_mask.copy()
            if right is False:
                # unmask everything
                mask[:] = False
    else:
        # unmask where either left or right is False
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = (left_mask & ~right_false) | (right_mask & ~left_false)

    return result, mask


def raise_for_nan(value: object, method: str) -> None:
    # [FUNCTION DOC]
    # HELPER: validation for the Kleene operations. Parameters: value is an operand; method is the logical operator name used in the message. Raises a value error if the value is a floating-point NaN; otherwise returns nothing.
    if lib.is_float(value) and np.isnan(value):
        raise ValueError(f"Cannot perform logical '{method}' with floating NaN")
