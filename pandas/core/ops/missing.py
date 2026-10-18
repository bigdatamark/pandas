"""
Missing data handling for arithmetic operations.

In particular, pandas conventions regarding division by zero differ
from numpy in the following ways:
    1) np.array([-1, 0, 1], dtype=dtype1) // np.array([0, 0, 0], dtype=dtype2)
       gives [nan, nan, nan] for most dtype combinations, and [0, 0, 0] for
       the remaining pairs
       (the remaining being dtype1==dtype2==intN and dtype==dtype2==uintN).

       pandas convention is to return [-inf, nan, inf] for all dtype
       combinations.


    2) np.array([-1, 0, 1], dtype=dtype1) % np.array([0, 0, 0], dtype=dtype2)
       gives precisely the same results as the // operation.

       pandas convention is to return [nan, nan, nan] for all dtype
       combinations.

    3) divmod behavior consistent with 1) and 2).
"""

# [MODULE DOC]
# Missing-data handling for arithmetic: enforces pandas conventions for division and modulo by zero on plain numpy results (documented above), which differ from numpy for integer dtypes.
# Contents: a private zero-fill function for modulo, a function that converts division by zero into signed infinity or NaN, and a dispatcher selecting the right treatment per operator, including divmod variants and reversed operators. No import-time side effects.

from __future__ import annotations

import operator

import numpy as np

from pandas.core import roperator


def _fill_zeros(result: np.ndarray, x, y) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: for the dispatcher. Replaces entries with NaN for integer-divisor zeros in a non-float result (used for modulo).
    # Parameters: result is the array computed by numpy; x and y are the operands (y is the divisor).
    # Float results are returned unchanged. A non-array divisor is turned into an array. If the divisor has an integer or unsigned dtype and contains zeros, the result is converted to float64 (no copy if already) and positions where the divisor is zero and the result is not already NaN are set to NaN in place. Returns the (possibly new) result.
    # KNOWN ISSUE: the docstring mentions flipping x and y for reversed ops and masking NaN from x, but neither happens here; x is unused and flipping is done by the caller.
    # KNOWN ISSUE: the early return for a divisor that has no dtype and is also not an array can never trigger, because every array has a dtype; the branch is dead.
    """
    If this is a reversed op, then flip x,y

    If we have an integer value (or array in y)
    and we have 0's, fill them with np.nan,
    return the result.

    Mask the nan's from x.
    """
    if result.dtype.kind == "f":
        return result

    is_variable_type = hasattr(y, "dtype")
    is_scalar_type = not isinstance(y, np.ndarray)

    if not is_variable_type and not is_scalar_type:
        # e.g. test_series_ops_name_retention with mod we get here with list/tuple
        return result

    if is_scalar_type:
        y = np.array(y)

    if y.dtype.kind in "iu":
        ymask = y == 0
        if ymask.any():
            # GH#7325, mask and nans must be broadcastable
            mask = ymask & ~np.isnan(result)

            # GH#9308 doing ravel on result and mask can improve putmask perf,
            #  but can also make unwanted copies.
            result = result.astype("float64", copy=False)

            np.putmask(result, mask, np.nan)

    return result


def mask_zero_div_zero(x, y, result: np.ndarray) -> np.ndarray:
    # [FUNCTION DOC]
    # Apply pandas division-by-zero conventions to the result of an integer-style floor division.
    # Parameters: x is the numerator and y the denominator (arrays or scalars; non-array inputs are converted); result is the raw numpy result.
    # If any denominator is zero: positions with zero numerator become NaN, positive numerators over positive zero (or negative over negative zero) become positive infinity, and the opposite signs become negative infinity; the sign of negative zero is respected. If any such position exists the result is converted to float64 and filled in place.
    # Returns the filled result (unchanged if no zeros).
    # KNOWN ISSUE: the docstring summary only mentions setting zero over zero to NaN, but the function also fills signed infinities.
    """
    Set results of  0 // 0 to np.nan, regardless of the dtypes
    of the numerator or the denominator.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    result : ndarray

    Returns
    -------
    ndarray
        The filled result.

    Examples
    --------
    >>> x = np.array([1, 0, -1], dtype=np.int64)
    >>> x
    array([ 1,  0, -1])
    >>> y = 0  # int 0; numpy behavior is different with float
    >>> result = x // y
    >>> result  # raw numpy result does not fill division by zero
    array([0, 0, 0])
    >>> mask_zero_div_zero(x, y, result)
    array([ inf,  nan, -inf])
    """

    if not hasattr(y, "dtype"):
        # e.g. scalar, tuple
        y = np.array(y)
    if not hasattr(x, "dtype"):
        # e.g scalar, tuple
        x = np.array(x)

    zmask = y == 0

    if zmask.any():
        # Flip sign if necessary for -0.0
        zneg_mask = zmask & np.signbit(y)
        zpos_mask = zmask & ~zneg_mask

        x_lt0 = x < 0
        x_gt0 = x > 0
        nan_mask = zmask & (x == 0)
        neginf_mask = (zpos_mask & x_lt0) | (zneg_mask & x_gt0)
        posinf_mask = (zpos_mask & x_gt0) | (zneg_mask & x_lt0)

        if nan_mask.any() or neginf_mask.any() or posinf_mask.any():
            # Fill negative/0 with -inf, positive/0 with +inf, 0/0 with NaN
            result = result.astype("float64", copy=False)

            result[nan_mask] = np.nan
            result[posinf_mask] = np.inf
            result[neginf_mask] = -np.inf

    return result


def dispatch_fill_zeros(op, left, right, result):
    # [FUNCTION DOC]
    # Select and apply the correct zero-division fix-up for an operator.
    # Parameters: op is the operator function; left and right are the operands (extension arrays already excluded; for reversed operators the roles are swapped); result is the numpy result (a pair of arrays for divmod operators).
    # Divmod and its reversed form fix the quotient with the zero-division masking and the remainder with the zero-fill, with operands swapped for the reversed form. Floor division and reversed floor division use the masking. Modulo and reversed modulo use the zero-fill. True division needs nothing because numpy already behaves as desired. Other operators are returned unchanged.
    # Returns the adjusted result (a pair for divmod operators).
    """
    Call _fill_zeros with the appropriate fill value depending on the operation,
    with special logic for divmod and rdivmod.

    Parameters
    ----------
    op : function (operator.add, operator.div, ...)
    left : object (np.ndarray for non-reversed ops)
        We have excluded ExtensionArrays here
    right : object (np.ndarray for reversed ops)
        We have excluded ExtensionArrays here
    result : ndarray

    Returns
    -------
    result : np.ndarray

    Notes
    -----
    For divmod and rdivmod, the `result` parameter and returned `result`
    is a 2-tuple of ndarray objects.
    """
    if op is divmod:
        result = (
            mask_zero_div_zero(left, right, result[0]),
            _fill_zeros(result[1], left, right),
        )
    elif op is roperator.rdivmod:
        result = (
            mask_zero_div_zero(right, left, result[0]),
            _fill_zeros(result[1], right, left),
        )
    elif op is operator.floordiv:
        # Note: no need to do this for truediv; numpy behaves the way
        #  we want.
        result = mask_zero_div_zero(left, right, result)
    elif op is roperator.rfloordiv:
        # Note: no need to do this for rtruediv; numpy behaves the wayS
        #  we want.
        result = mask_zero_div_zero(right, left, result)
    elif op is operator.mod:
        result = _fill_zeros(result, left, right)
    elif op is roperator.rmod:
        result = _fill_zeros(result, right, left)
    return result
