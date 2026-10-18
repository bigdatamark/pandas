"""
Methods used by Block.replace and related methods.
"""

# [MODULE DOC]
# Helpers for value replacement, used by the block level replace machinery.
# Public contents: should_use_regex (decide whether a pattern is treated as a
# regular expression), compare_or_regex_search (elementwise equality or regex
# match against a scalar or pattern) and replace_regex (in-place regex
# substitution on an object array). Import-time side effects: none.
from __future__ import annotations

import operator
import re
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.core.dtypes.common import (
    is_bool,
    is_re,
    is_re_compilable,
)
from pandas.core.dtypes.missing import isna

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Scalar,
        npt,
    )


def should_use_regex(regex: bool, to_replace: Any) -> bool:
    # [FUNCTION DOC]
    # Decides whether to_replace should be interpreted as a regular expression.
    # Inputs: regex (the user requested flag) and to_replace (any object).
    # Returns true if to_replace is already a compiled pattern, or regex is true;
    # in both cases to_replace must also be compilable as a pattern and compile to
    # a non-empty pattern, else false. May compile the pattern (and could raise a
    # regex error only for invalid patterns that the compilability check missed).
    """
    Decide whether to treat `to_replace` as a regular expression.
    """
    if is_re(to_replace):
        regex = True

    regex = regex and is_re_compilable(to_replace)

    # Don't use regex if the pattern is empty.
    regex = regex and re.compile(to_replace).pattern != ""
    return regex


def compare_or_regex_search(
    a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: npt.NDArray[np.bool_]
) -> ArrayLike:
    # [FUNCTION DOC]
    # Elementwise comparison of an array with a scalar or pattern.
    # Inputs: a (array-like or scalar), b (scalar or compiled pattern), regex (bool,
    # whether to attempt regex search) and mask (boolean array or none, true marks
    # positions that are valid to compare; positions outside it are never compared).
    # Behavior: if b is missing, returns the inverted mask (callers use this to find
    # missing positions). Otherwise a comparison is built: equality, unless regex is
    # requested and should_use_regex agrees, in which case a vectorized regex
    # search that is true only for string elements matching a string/pattern b. For
    # numpy arrays with a mask, only masked-in elements are compared and the result
    # is scattered into a full-shape boolean array (false elsewhere).
    # A type error is raised if the result is a plain boolean while a is a numpy
    # array (the comparison was not elementwise), naming the array dtype and type of
    # b. Returns the boolean result.
    """
    Compare two array-like inputs of the same shape or two scalar values

    Calls operator.eq or re.search, depending on regex argument. If regex is
    True, perform an element-wise regex matching.

    Parameters
    ----------
    a : array-like
    b : scalar or regex pattern
    regex : bool
    mask : np.ndarray[bool]

    Returns
    -------
    mask : array-like of bool
    """
    if isna(b):
        return ~mask

    def _check_comparison_types(
        result: ArrayLike | bool, a: ArrayLike, b: Scalar | Pattern
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: validation step of the comparison. Inputs: result (comparison
        # outcome), a and b (operands). Raises a type error when the result is a
        # single boolean although a is a numpy array (meaning numpy could not
        # compare elementwise). Returns nothing.
        # KNOWN ISSUE: the docstring claims it returns the comparison result, but
        # the function returns nothing.
        """
        Raises an error if the two arrays (a,b) cannot be compared.
        Otherwise, returns the comparison result as expected.
        """
        if is_bool(result) and isinstance(a, np.ndarray):
            type_names = [type(a).__name__, type(b).__name__]

            type_names[0] = f"ndarray(dtype={a.dtype})"

            raise TypeError(
                f"Cannot compare types {type_names[0]!r} and {type_names[1]!r}"
            )

    if not regex or not should_use_regex(regex, b):
        # TODO: should use missing.mask_missing?
        op = lambda x: operator.eq(x, b)
    else:
        op = np.vectorize(
            lambda x: (
                bool(re.search(b, x))
                if isinstance(x, str) and isinstance(b, (str, Pattern))
                else False
            ),
            otypes=[bool],
        )

    # GH#32621 use mask to avoid comparing to NAs
    if isinstance(a, np.ndarray) and mask is not None:
        a = a[mask]
        result = op(a)

        if isinstance(result, np.ndarray):
            # The shape of the mask can differ to that of the result
            # since we may compare only a subset of a's or b's elements
            tmp = np.zeros(mask.shape, dtype=np.bool_)
            np.place(tmp, mask, result)
            result = tmp
    else:
        result = op(a)

    _check_comparison_types(result, a, b)
    return result


def replace_regex(
    values: ArrayLike, rx: re.Pattern, value, mask: npt.NDArray[np.bool_] | None
) -> None:
    # [FUNCTION DOC]
    # Regex substitution on an object array, in place. Inputs: values (object
    # dtype array, mutated), rx (compiled pattern), value (replacement) and mask
    # (boolean array or none; when given only masked-in positions are processed,
    # broadcast to the shape of values if its dimensionality differs).
    # Behavior: only string elements are touched. If value is missing or not a
    # string, any element that the pattern matches is replaced wholesale by value;
    # otherwise matches inside the string are substituted using the normal regex
    # substitution rules. Non-string elements are kept. Returns nothing.
    """
    Parameters
    ----------
    values : ArrayLike
        Object dtype.
    rx : re.Pattern
    value : Any
    mask : np.ndarray[bool], optional

    Notes
    -----
    Alters values in-place.
    """

    # deal with replacing values with objects (strings) that match but
    # whose replacement is not a string (numeric, nan, object)
    if isna(value) or not isinstance(value, str):

        def re_replacer(s):
            # [FUNCTION DOC]
            # HELPER: element function for the whole-value replacement mode.
            # Returns the replacement value if s is a string the pattern matches,
            # otherwise returns s unchanged.
            if is_re(rx) and isinstance(s, str):
                return value if rx.search(s) is not None else s
            else:
                return s

    else:
        # value is guaranteed to be a string here, s can be either a string
        # or null if it's null it gets returned
        def re_replacer(s):
            # [FUNCTION DOC]
            # HELPER: element function for the substring substitution mode.
            # Returns the string with all pattern matches substituted when s is a
            # string, otherwise returns s unchanged (including nulls).
            if is_re(rx) and isinstance(s, str):
                return rx.sub(value, s)
            else:
                return s

    f = np.vectorize(re_replacer, otypes=[np.object_])

    if mask is None:
        values[:] = f(values)
    else:
        if values.ndim != mask.ndim:
            mask = np.broadcast_to(mask, values.shape)
        values[mask] = f(values[mask])
