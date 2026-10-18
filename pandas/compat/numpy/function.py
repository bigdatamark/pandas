"""
For compatibility with numpy libraries, pandas functions or methods have to
accept '*args' and '**kwargs' parameters to accommodate numpy arguments that
are not actually used or respected in the pandas implementation.

To ensure that users do not abuse these parameters, validation is performed in
'validators.py' to make sure that any extra parameters passed correspond ONLY
to those in the numpy signature. Part of that validation includes whether or
not the user attempted to pass in non-default values for these extraneous
parameters. As we want to discourage users from relying on these parameters
when calling the pandas implementation, we want them only to pass in the
default values for these parameters.

This module provides a set of commonly used default arguments for functions and
methods that are spread throughout the codebase. This module will make it
easier to adjust to future upstream changes in the analogous numpy signatures.
"""

# [MODULE DOC]
# HELPER: default-argument tables and validators for numpy-compatible extra
# arguments. Many pandas methods accept arbitrary positional and keyword
# arguments only so that numpy functions dispatching to them keep working; this
# module checks that such extra arguments are absent or equal the numpy
# defaults, otherwise an unsupported-call error is raised by the underlying
# validation helpers (in the util validators module).
# Public contents: the CompatValidator class; many defaults mappings (argmin and
# argmax, argsort in two variants, clip, cumulative functions, all and any,
# logical functions, min and max, repeat, round, statistics, sum, product and
# mean, median, ddof statistics, take, transpose); ready-made validator
# instances built from them (validate_argmin, validate_argmax, validate_argsort,
# validate_argsort_kind, validate_clip, validate_cum_func, validate_cumsum,
# validate_all, validate_any, validate_logical_func, validate_min, validate_max,
# validate_repeat, validate_round, validate_stat_func, validate_sum,
# validate_prod, validate_mean, validate_median, validate_stat_ddof_func,
# validate_take, validate_transpose); and wrapper functions that first
# re-arrange arguments when numpy passes an extra positional argument in the
# place of a pandas parameter (the with_skipna, with_ascending and with_axis
# wrappers), plus validate_groupby_func, validate_minmax_axis and validate_func
# (dispatches by function name to the right validator). Import-time side
# effects: builds the tables and validator objects.
# KNOWN ISSUE: the module docstring says validation is performed in a file named
# validators.py, but the validation helpers live in the util validators module
# (a private module of the utilities package).
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from numpy import ndarray

from pandas._libs.lib import (
    is_bool,
    is_integer,
)
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
    validate_args,
    validate_args_and_kwargs,
    validate_kwargs,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        AxisInt,
    )

    AxisNoneT = TypeVar("AxisNoneT", Axis, None)


class CompatValidator:
    # [CLASS DOC]
    # HELPER: callable validator object for numpy-compatibility arguments.
    # State: defaults (mapping of allowed extra argument names to their default
    # values), fname (default function name for error messages), method (which
    # validation to run: args, kwargs or both) and max_fname_arg_count (number
    # of leading positional parameters of the real function, used to compute
    # argument positions in messages). Calling the instance runs the chosen
    # validation; the per-call arguments can override the stored name, count and
    # method.
    def __init__(
        self,
        defaults,
        fname=None,
        method: str | None = None,
        max_fname_arg_count=None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the settings on the instance. Parameters: defaults (mapping of
        # allowed names to default values), fname (function name for messages,
        # default none), method (args, kwargs or both; default none, which would
        # make later calls raise unless a method is passed at call time) and
        # max_fname_arg_count (default none).
        self.fname = fname
        self.method = method
        self.defaults = defaults
        self.max_fname_arg_count = max_fname_arg_count

    def __call__(
        self,
        args,
        kwargs,
        fname=None,
        max_fname_arg_count=None,
        method: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Runs the validation. Parameters: args (tuple of extra positional
        # arguments), kwargs (mapping of extra keyword arguments), and optional
        # overrides fname, max_fname_arg_count and method (each falls back to the
        # stored value when none). Returns immediately if both args and kwargs are
        # empty. Otherwise for method args validates positional arguments against
        # the defaults, for kwargs validates keywords, for both validates both
        # together; any other method raises a value error naming it. The
        # underlying helpers raise errors when an extra argument is not allowed or
        # differs from the default.
        if not args and not kwargs:
            return None

        fname = self.fname if fname is None else fname
        max_fname_arg_count = (
            self.max_fname_arg_count
            if max_fname_arg_count is None
            else max_fname_arg_count
        )
        method = self.method if method is None else method

        if method == "args":
            validate_args(fname, args, max_fname_arg_count, self.defaults)
        elif method == "kwargs":
            validate_kwargs(fname, kwargs, self.defaults)
        elif method == "both":
            validate_args_and_kwargs(
                fname, args, kwargs, max_fname_arg_count, self.defaults
            )
        else:
            raise ValueError(f"invalid validation method '{method}'")


ARGMINMAX_DEFAULTS = {"out": None}
validate_argmin = CompatValidator(
    ARGMINMAX_DEFAULTS, fname="argmin", method="both", max_fname_arg_count=1
)
validate_argmax = CompatValidator(
    ARGMINMAX_DEFAULTS, fname="argmax", method="both", max_fname_arg_count=1
)


def process_skipna(skipna: bool | ndarray | None, args) -> tuple[bool, Any]:
    # [FUNCTION DOC]
    # HELPER: for the argmin and argmax wrappers. Parameters: skipna (a boolean,
    # or an array or none when numpy passed its output argument in that
    # position) and args (tuple of extra positional arguments). If skipna is an
    # array or none, it is moved to the front of the extra arguments and skipna
    # is reset to true. Returns the pair (skipna, args).
    if isinstance(skipna, ndarray) or skipna is None:
        args = (skipna, *args)
        skipna = True

    return skipna, args


def validate_argmin_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    # [FUNCTION DOC]
    # Normalizes and validates the extra arguments of an argmin call. Parameters:
    # skipna, args, kwargs as described for the helper. Moves a misplaced numpy
    # output argument out of skipna, validates the extra arguments against the
    # argmin defaults (raising an unsupported-call error on a violation) and
    # returns the effective boolean skipna.
    """
    If 'Series.argmin' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmin(args, kwargs)
    return skipna


def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -> bool:
    # [FUNCTION DOC]
    # Same as the argmin variant but validates against the argmax validator:
    # moves a misplaced numpy output argument out of skipna, validates the extra
    # arguments, and returns the effective boolean skipna.
    """
    If 'Series.argmax' is called via the 'numpy' library, the third parameter
    in its signature is 'out', which takes either an ndarray or 'None', so
    check if the 'skipna' parameter is either an instance of ndarray or is
    None, since 'skipna' itself should be a boolean
    """
    skipna, args = process_skipna(skipna, args)
    validate_argmax(args, kwargs)
    return skipna


ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"
ARGSORT_DEFAULTS["order"] = None
# KNOWN ISSUE: the kind default is assigned twice in this table (first to
# quicksort above, then overwritten with none here); the first assignment is
# dead and the effective default is none.
ARGSORT_DEFAULTS["kind"] = None
ARGSORT_DEFAULTS["stable"] = None


validate_argsort = CompatValidator(
    ARGSORT_DEFAULTS, fname="argsort", max_fname_arg_count=0, method="both"
)

# two different signatures of argsort, this second validation for when the
# `kind` param is supported
ARGSORT_DEFAULTS_KIND: dict[str, int | None] = {}
ARGSORT_DEFAULTS_KIND["axis"] = -1
ARGSORT_DEFAULTS_KIND["order"] = None
ARGSORT_DEFAULTS_KIND["stable"] = None
validate_argsort_kind = CompatValidator(
    ARGSORT_DEFAULTS_KIND, fname="argsort", max_fname_arg_count=0, method="both"
)


def validate_argsort_with_ascending(ascending: bool | int | None, args, kwargs) -> bool:
    # [FUNCTION DOC]
    # Normalizes and validates extra arguments of a categorical argsort call.
    # Parameters: ascending (a boolean, or an integer or none when numpy passed
    # its axis in that position), args and kwargs. If ascending is an integer or
    # none it is moved to the front of the extra arguments and ascending is set to
    # true. Validates against the argsort-with-kind validator with the maximum
    # leading argument count overridden to 3. Returns the effective ascending
    # flag.
    """
    If 'Categorical.argsort' is called via the 'numpy' library, the first
    parameter in its signature is 'axis', which takes either an integer or
    'None', so check if the 'ascending' parameter has either integer type or is
    None, since 'ascending' itself should be a boolean
    """
    if is_integer(ascending) or ascending is None:
        args = (ascending, *args)
        ascending = True

    validate_argsort_kind(args, kwargs, max_fname_arg_count=3)
    ascending = cast(bool, ascending)
    return ascending


CLIP_DEFAULTS: dict[str, Any] = {"out": None}
validate_clip = CompatValidator(
    CLIP_DEFAULTS, fname="clip", method="both", max_fname_arg_count=3
)


@overload
def validate_clip_with_axis(axis: ndarray, args, kwargs) -> None:
    # [FUNCTION DOC]
    # Typing overload stub: when axis is an array the result is none. See the
    # implementation below.
    ...


@overload
def validate_clip_with_axis(axis: AxisNoneT, args, kwargs) -> AxisNoneT:
    # [FUNCTION DOC]
    # Typing overload stub: when axis is an axis value or none the same type is
    # returned. See the implementation below.
    ...


def validate_clip_with_axis(
    axis: ndarray | AxisNoneT, args, kwargs
) -> AxisNoneT | None:
    # [FUNCTION DOC]
    # Normalizes and validates extra arguments of a clip call. Parameters: axis
    # (an axis value, none, or an array when numpy passed its output argument in
    # that position), args and kwargs. If axis is an array it is moved to the
    # front of the extra arguments and axis becomes none. Validates the extra
    # arguments against the clip defaults (raising an unsupported-call error on a
    # violation) and returns the effective axis.
    """
    If 'NDFrame.clip' is called via the numpy library, the third parameter in
    its signature is 'out', which can takes an ndarray, so check if the 'axis'
    parameter is an instance of ndarray, since 'axis' itself should either be
    an integer or None
    """
    if isinstance(axis, ndarray):
        args = (axis, *args)
        # error: Incompatible types in assignment (expression has type "None",
        # variable has type "Union[ndarray[Any, Any], str, int]")
        axis = None  # type: ignore[assignment]

    validate_clip(args, kwargs)
    # error: Incompatible return value type (got "Union[ndarray[Any, Any],
    # str, int]", expected "Union[str, int, None]")
    return axis  # type: ignore[return-value]


CUM_FUNC_DEFAULTS: dict[str, Any] = {}
CUM_FUNC_DEFAULTS["dtype"] = None
CUM_FUNC_DEFAULTS["out"] = None
validate_cum_func = CompatValidator(
    CUM_FUNC_DEFAULTS, method="both", max_fname_arg_count=1
)
validate_cumsum = CompatValidator(
    CUM_FUNC_DEFAULTS, fname="cumsum", method="both", max_fname_arg_count=1
)


def validate_cum_func_with_skipna(skipna: bool, args, kwargs, name) -> bool:
    # [FUNCTION DOC]
    # Normalizes and validates extra arguments of a cumulative function call.
    # Parameters: skipna (should be a boolean, but numpy may pass its dtype
    # argument here), args, kwargs and name (function name for messages). If
    # skipna is not a boolean it is moved to the front of the extra arguments and
    # reset to true; a numpy boolean is converted to a plain boolean. Validates
    # against the cumulative function defaults using the given name. Returns the
    # effective skipna.
    """
    If this function is called via the 'numpy' library, the third parameter in
    its signature is 'dtype', which takes either a 'numpy' dtype or 'None', so
    check if the 'skipna' parameter is a boolean or not
    """
    if not is_bool(skipna):
        args = (skipna, *args)
        skipna = True
    elif isinstance(skipna, np.bool_):
        skipna = bool(skipna)

    validate_cum_func(args, kwargs, fname=name)
    return skipna


ALLANY_DEFAULTS: dict[str, bool | None] = {}
ALLANY_DEFAULTS["dtype"] = None
ALLANY_DEFAULTS["out"] = None
ALLANY_DEFAULTS["keepdims"] = False
ALLANY_DEFAULTS["axis"] = None
validate_all = CompatValidator(
    ALLANY_DEFAULTS, fname="all", method="both", max_fname_arg_count=1
)
validate_any = CompatValidator(
    ALLANY_DEFAULTS, fname="any", method="both", max_fname_arg_count=1
)

LOGICAL_FUNC_DEFAULTS = {"out": None, "keepdims": False}
validate_logical_func = CompatValidator(LOGICAL_FUNC_DEFAULTS, method="kwargs")

MINMAX_DEFAULTS = {"axis": None, "dtype": None, "out": None, "keepdims": False}
validate_min = CompatValidator(
    MINMAX_DEFAULTS, fname="min", method="both", max_fname_arg_count=1
)
validate_max = CompatValidator(
    MINMAX_DEFAULTS, fname="max", method="both", max_fname_arg_count=1
)


REPEAT_DEFAULTS: dict[str, Any] = {"axis": None}
validate_repeat = CompatValidator(
    REPEAT_DEFAULTS, fname="repeat", method="both", max_fname_arg_count=1
)

ROUND_DEFAULTS: dict[str, Any] = {"out": None}
validate_round = CompatValidator(
    ROUND_DEFAULTS, fname="round", method="both", max_fname_arg_count=1
)

STAT_FUNC_DEFAULTS: dict[str, Any | None] = {}
STAT_FUNC_DEFAULTS["dtype"] = None
STAT_FUNC_DEFAULTS["out"] = None

SUM_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
SUM_DEFAULTS["axis"] = None
SUM_DEFAULTS["keepdims"] = False
SUM_DEFAULTS["initial"] = None

PROD_DEFAULTS = SUM_DEFAULTS.copy()

MEAN_DEFAULTS = SUM_DEFAULTS.copy()

MEDIAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
MEDIAN_DEFAULTS["overwrite_input"] = False
MEDIAN_DEFAULTS["keepdims"] = False

STAT_FUNC_DEFAULTS["keepdims"] = False

validate_stat_func = CompatValidator(STAT_FUNC_DEFAULTS, method="kwargs")
validate_sum = CompatValidator(
    SUM_DEFAULTS, fname="sum", method="both", max_fname_arg_count=1
)
validate_prod = CompatValidator(
    PROD_DEFAULTS, fname="prod", method="both", max_fname_arg_count=1
)
validate_mean = CompatValidator(
    MEAN_DEFAULTS, fname="mean", method="both", max_fname_arg_count=1
)
validate_median = CompatValidator(
    MEDIAN_DEFAULTS, fname="median", method="both", max_fname_arg_count=1
)

STAT_DDOF_FUNC_DEFAULTS: dict[str, bool | None] = {}
STAT_DDOF_FUNC_DEFAULTS["dtype"] = None
STAT_DDOF_FUNC_DEFAULTS["out"] = None
STAT_DDOF_FUNC_DEFAULTS["keepdims"] = False
validate_stat_ddof_func = CompatValidator(STAT_DDOF_FUNC_DEFAULTS, method="kwargs")

TAKE_DEFAULTS: dict[str, str | None] = {}
TAKE_DEFAULTS["out"] = None
TAKE_DEFAULTS["mode"] = "raise"
validate_take = CompatValidator(TAKE_DEFAULTS, fname="take", method="kwargs")


TRANSPOSE_DEFAULTS = {"axes": None}
validate_transpose = CompatValidator(
    TRANSPOSE_DEFAULTS, fname="transpose", method="both", max_fname_arg_count=0
)


def validate_groupby_func(name: str, args, kwargs, allowed=None) -> None:
    # [FUNCTION DOC]
    # Rejects numpy-style extra arguments to a group-by method. Parameters: name
    # (method name for the message), args (extra positional arguments), kwargs
    # (extra keyword arguments) and allowed (optional collection of keyword names
    # that are permitted; default none meaning empty). Raises the unsupported
    # function call error, advising to use the group-by method form, if there is
    # any positional argument or any keyword not in allowed. Returns nothing.
    """
    'args' and 'kwargs' should be empty, except for allowed kwargs because all
    of their necessary parameters are explicitly listed in the function
    signature
    """
    if allowed is None:
        allowed = []

    kwargs = set(kwargs) - set(allowed)

    if len(args) + len(kwargs) > 0:
        raise UnsupportedFunctionCall(
            "numpy operations are not valid with groupby. "
            f"Use .groupby(...).{name}() instead"
        )


def validate_minmax_axis(axis: AxisInt | None, ndim: int = 1) -> None:
    # [FUNCTION DOC]
    # Validates the axis for min, max, argmin and argmax. Parameters: axis
    # (integer or none) and ndim (number of dimensions, default 1). Returns
    # nothing when axis is none. Raises a value error when axis is at least ndim
    # or is negative with magnitude larger than ndim.
    # KNOWN ISSUE: the existing docstring says the axis must be zero or none, but
    # the code accepts any axis within the number of dimensions (including
    # negative ones), as the error message says.
    """
    Ensure that the axis argument passed to min, max, argmin, or argmax is zero
    or None, as otherwise it will be incorrectly ignored.

    Parameters
    ----------
    axis : int or None
    ndim : int, default 1

    Raises
    ------
    ValueError
    """
    if axis is None:
        return
    if axis >= ndim or (axis < 0 and ndim + axis < 0):
        raise ValueError(f"`axis` must be fewer than the number of dimensions ({ndim})")


_validation_funcs = {
    "median": validate_median,
    "mean": validate_mean,
    "min": validate_min,
    "max": validate_max,
    "sum": validate_sum,
    "prod": validate_prod,
}


def validate_func(fname, args, kwargs) -> None:
    # [FUNCTION DOC]
    # Validates extra arguments for a named statistical function. Parameters:
    # fname (function name), args and kwargs. If the name is not one of median,
    # mean, min, max, sum or prod, uses the generic keyword-only statistics
    # validator (called with the function name); otherwise uses that function's
    # own validator. Returns what the validator returns (nothing), and raises an
    # unsupported-call error on a violation.
    if fname not in _validation_funcs:
        return validate_stat_func(args, kwargs, fname=fname)

    validation_func = _validation_funcs[fname]
    return validation_func(args, kwargs)
