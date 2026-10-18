"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""

# [MODULE DOC]
# Optional acceleration of element-wise binary operations and conditional
# selection via numexpr, used by the arithmetic layer. Public functions:
# evaluate (apply a binary operator to two operands), where (select between two
# operands by a boolean condition), set_use_numexpr and set_numexpr_threads
# (configuration), and set_test_mode / get_test_result (test instrumentation
# recording whether numexpr was really used).
# Module state: a flag saying whether numexpr is enabled, two function slots
# (the current evaluate and where implementations, chosen by that flag), a
# test-mode flag with a list of recorded results, a table of allowed dtypes
# per operation kind, a minimum element count (one million) below which
# numexpr is never used, and a table mapping operator functions to numexpr
# operator strings (entries with no string, such as floor division, modulo
# and divmod, always use the plain implementation).
# Import-time side effects: imports numexpr when installed, computes a version
# flag for it, and runs the enable routine with the compute.use_numexpr option
# value so the function slots are populated.

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._config import get_option

from pandas.compat.numpy import np_version_gt2_3
from pandas.util._exceptions import find_stack_level

from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
from pandas.util.version import Version

if NUMEXPR_INSTALLED:
    import numexpr as ne

    ne_gt_211 = Version(ne.__version__) >= Version("2.11.0")

if TYPE_CHECKING:
    from pandas._typing import FuncType

_TEST_MODE: bool | None = None
_TEST_RESULT: list[bool] = []
USE_NUMEXPR = NUMEXPR_INSTALLED
_evaluate: FuncType | None = None
_where: FuncType | None = None

# the set of dtypes that we will allow pass to numexpr
_ALLOWED_DTYPES = {
    "evaluate": {"int64", "int32", "float64", "float32", "bool"},
    "where": {"int64", "float64", "bool"},
}

# the minimum prod shape that we will use numexpr
_MIN_ELEMENTS = 1_000_000


def set_use_numexpr(v: bool = True) -> None:
    # [FUNCTION DOC]
    # Enables or disables numexpr use and selects the matching implementations.
    # Input v: desired state, default True. If numexpr is installed, the global
    # flag is set to v, except that it is forced off when the numpy version is
    # newer than 2.3 and numexpr is older than 2.11 (known wrong results).
    # If numexpr is not installed the flag is left unchanged. Then the global
    # evaluate and where function slots are bound to the numexpr versions when
    # the flag is on, else to the standard versions. Mutates module globals.
    # set/unset to use numexpr
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        if np_version_gt2_3 and not ne_gt_211:
            # incompatibility of numexpr 2.10 with newer pandas resulting in wrong data
            # https://github.com/pandas-dev/pandas/issues/63320
            USE_NUMEXPR = False
        else:
            USE_NUMEXPR = v

    # choose what we are going to do
    global _evaluate, _where

    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard


def set_numexpr_threads(n=None) -> None:
    # [FUNCTION DOC]
    # Sets the numexpr worker thread count. Input n: number of threads, or none
    # to use the detected core count. Does nothing unless numexpr is installed
    # and enabled. Side effect: changes numexpr's global thread setting.
    # if we are using numexpr, set the threads to n
    # otherwise reset
    if NUMEXPR_INSTALLED and USE_NUMEXPR:
        if n is None:
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)


def _evaluate_standard(op, op_str, left_op, right_op):
    # [FUNCTION DOC]
    # HELPER: plain (non-numexpr) evaluation used by the evaluate dispatcher.
    # Inputs: op, the operator function; op_str, the operator string (unused
    # here); left_op and right_op, the operands. Returns the operator applied to
    # the operands. In test mode records that numexpr was not used.
    """
    Standard evaluation.
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(left_op, right_op)


def _can_use_numexpr(op, op_str, left_op, right_op, dtype_check) -> bool:
    # [FUNCTION DOC]
    # HELPER: decides whether numexpr should be used for an operation.
    # Inputs: op (unused); op_str, the numexpr operator string or none;
    # left_op and right_op, operands (the left one must have a size); and
    # dtype_check, key selecting the allowed-dtype set ("evaluate" or "where").
    # Returns True only when op_str is not none, the left operand has more
    # elements than the minimum threshold, and the dtype names of the operands
    # that have a dtype are all within the allowed set (or no operand has a
    # dtype). Otherwise False.
    """return left_op boolean if we WILL be using numexpr"""
    if op_str is not None:
        # required min elements (otherwise we are adding overhead)
        if left_op.size > _MIN_ELEMENTS:
            # check for dtype compatibility
            dtypes: set[str] = set()
            for o in [left_op, right_op]:
                # ndarray and Series Case
                if hasattr(o, "dtype"):
                    dtypes |= {o.dtype.name}

            # allowed are a superset
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True

    return False


def _evaluate_numexpr(op, op_str, left_op, right_op):
    # [FUNCTION DOC]
    # HELPER: numexpr-backed implementation of the evaluate slot.
    # Inputs as for the standard version. If numexpr is usable for the
    # operands, evaluates the operation with safe casting; for a reversed
    # operator (name starts with r after stripping underscores) the operands are
    # swapped for the call, since the operator string is the unreversed one,
    # and swapped back afterwards for any fallback. A type error from numexpr
    # (e.g. integer array power) is swallowed, leaving no result; a
    # not-implemented error is swallowed only when the boolean arithmetic
    # fallback helper approves (it warns), otherwise re-raised. In test mode
    # records whether numexpr produced a result. If there is no result, falls
    # back to the standard evaluation. Returns the result.
    result = None

    if _can_use_numexpr(op, op_str, left_op, right_op, "evaluate"):
        is_reversed = op.__name__.strip("_").startswith("r")
        if is_reversed:
            # we were originally called by a reversed op method
            left_op, right_op = right_op, left_op

        left_value = left_op
        right_value = right_op

        try:
            result = ne.evaluate(
                f"left_value {op_str} right_value",
                local_dict={"left_value": left_value, "right_value": right_value},
                casting="safe",
            )
        except TypeError:
            # numexpr raises eg for array ** array with integers
            # (https://github.com/pydata/numexpr/issues/379)
            pass
        except NotImplementedError:
            if _bool_arith_fallback(op_str, left_op, right_op):
                pass
            else:
                raise

        if is_reversed:
            # reverse order to original for fallback
            left_op, right_op = right_op, left_op

    if _TEST_MODE:
        _store_test_result(result is not None)

    if result is None:
        result = _evaluate_standard(op, op_str, left_op, right_op)

    return result


_op_str_mapping = {
    operator.add: "+",
    roperator.radd: "+",
    operator.mul: "*",
    roperator.rmul: "*",
    operator.sub: "-",
    roperator.rsub: "-",
    operator.truediv: "/",
    roperator.rtruediv: "/",
    # floordiv not supported by numexpr 2.x
    operator.floordiv: None,
    roperator.rfloordiv: None,
    # we require Python semantics for mod of negative for backwards compatibility
    # see https://github.com/pydata/numexpr/issues/365
    # so sticking with unaccelerated for now GH#36552
    operator.mod: None,
    roperator.rmod: None,
    operator.pow: "**",
    roperator.rpow: "**",
    operator.eq: "==",
    operator.ne: "!=",
    operator.le: "<=",
    operator.lt: "<",
    operator.ge: ">=",
    operator.gt: ">",
    operator.and_: "&",
    roperator.rand_: "&",
    operator.or_: "|",
    roperator.ror_: "|",
    operator.xor: "^",
    roperator.rxor: "^",
    divmod: None,
    roperator.rdivmod: None,
}


def _where_standard(cond, left_op, right_op):
    # [FUNCTION DOC]
    # HELPER: plain selection: returns elements of left_op where cond is true,
    # else right_op, via the numpy conditional selection. Callers must pass
    # raw arrays where needed.
    # Caller is responsible for extracting ndarray if necessary
    return np.where(cond, left_op, right_op)


def _where_numexpr(cond, left_op, right_op):
    # [FUNCTION DOC]
    # HELPER: numexpr-backed selection. Uses numexpr (safe casting) when the
    # operands pass the size and where-dtype checks, else the standard
    # selection. Returns the selected array. Callers pass raw arrays.
    # Caller is responsible for extracting ndarray if necessary
    result = None

    if _can_use_numexpr(None, "where", left_op, right_op, "where"):
        result = ne.evaluate(
            "where(cond_value, a_value, b_value)",
            local_dict={"cond_value": cond, "a_value": left_op, "b_value": right_op},
            casting="safe",
        )

    if result is None:
        result = _where_standard(cond, left_op, right_op)

    return result


# turn myself on
set_use_numexpr(get_option("compute.use_numexpr"))


def _has_bool_dtype(x):
    # [FUNCTION DOC]
    # HELPER: True when x has a boolean dtype, or, for objects without a
    # dtype, when it is a Python or numpy boolean scalar; else False.
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))


_BOOL_OP_UNSUPPORTED = {"+": "|", "*": "&", "-": "^"}


def _bool_arith_fallback(op_str, left_op, right_op) -> bool:
    # [FUNCTION DOC]
    # HELPER: decides whether a numexpr not-implemented error on boolean
    # operands should fall back to Python evaluation. Inputs: op_str, the
    # operator string; left_op, right_op, the operands. If both are boolean and
    # the operator is one of the unsupported arithmetic ones (add, multiply,
    # subtract), emits a user warning suggesting the logical equivalent (or,
    # and, xor respectively) and returns True; otherwise returns False.
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
    if _has_bool_dtype(left_op) and _has_bool_dtype(right_op):
        if op_str in _BOOL_OP_UNSUPPORTED:
            warnings.warn(
                f"evaluating in Python space because the {op_str!r} "
                "operator is not supported by numexpr for the bool dtype, "
                f"use {_BOOL_OP_UNSUPPORTED[op_str]!r} instead.",
                stacklevel=find_stack_level(),
            )
            return True
    return False


def evaluate(op, left_op, right_op, use_numexpr: bool = True):
    # [FUNCTION DOC]
    # Public dispatcher for binary operations. Looks up the numexpr operator
    # string for op in the mapping (raises a key error for unmapped operators).
    # If a string exists and use_numexpr is true, delegates to the current
    # evaluate slot (numexpr or standard depending on configuration); otherwise
    # uses the standard implementation. Returns the operation result.
    """
    Evaluate and return the expression of the op on left_op and right_op.

    Parameters
    ----------
    op : the actual operand
    left_op : left operand
    right_op : right operand
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    op_str = _op_str_mapping[op]
    if op_str is not None:
        if use_numexpr:
            # error: "None" not callable
            return _evaluate(op, op_str, left_op, right_op)  # type: ignore[misc]
    return _evaluate_standard(op, op_str, left_op, right_op)


def where(cond, left_op, right_op, use_numexpr: bool = True):
    # [FUNCTION DOC]
    # Public selection dispatcher: returns left_op where the boolean array cond
    # is true and right_op elsewhere. Uses the current where slot when
    # use_numexpr is true, else the standard version. Asserts that the slot
    # has been initialised.
    """
    Evaluate the where condition cond on left_op and right_op.

    Parameters
    ----------
    cond : np.ndarray[bool]
    left_op : return if cond is True
    right_op : return if cond is False
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    assert _where is not None
    if use_numexpr:
        return _where(cond, left_op, right_op)
    else:
        return _where_standard(cond, left_op, right_op)


def set_test_mode(v: bool = True) -> None:
    # [FUNCTION DOC]
    # Turns test instrumentation on or off (input v, default True) and clears
    # the recorded results. Mutates module globals.
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
    global _TEST_MODE, _TEST_RESULT
    _TEST_MODE = v
    _TEST_RESULT = []


def _store_test_result(used_numexpr: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: test instrumentation. Appends a true entry to the recorded
    # results only when used_numexpr is true; false values are not recorded.
    if used_numexpr:
        _TEST_RESULT.append(used_numexpr)


def get_test_result() -> list[bool]:
    # [FUNCTION DOC]
    # Returns the list of recorded numexpr-used flags and resets the stored
    # list to empty.
    """
    Get test result and reset test_results.
    """
    global _TEST_RESULT
    res = _TEST_RESULT
    _TEST_RESULT = []
    return res
