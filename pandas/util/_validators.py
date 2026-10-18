"""
Module that contains many useful utilities
for validating data or function arguments
"""

# [MODULE DOC]
# Argument and value validators used throughout pandas. Public contents: checks
# for compatibility-only positional and keyword arguments (the ones accepted for
# numpy API compatibility but required to have default values), a boolean
# keyword validator, a missing-value flag validator, a fill-method validator, a
# percentile validator, a sort-direction validator, interval-closure and
# inclusivity validators, an insertion-location validator and a dtype backend
# validator. Private helpers perform the argument-count, default-value and
# unexpected-key checks. Defines two type variables for the boolean-like
# validators. Depends on numpy, the compiled lib module, the missing-value
# singleton and the dtype predicates; one function lazily imports the
# fill-method cleaner. No import-time side effects.
from __future__ import annotations

from collections.abc import (
    Iterable,
    Sequence,
)
from typing import (
    TypeVar,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.missing import NA

from pandas.core.dtypes.common import (
    is_bool,
    is_integer,
)

BoolishT = TypeVar("BoolishT", bool, int)
BoolishNoneT = TypeVar("BoolishNoneT", bool, int, None)


def _check_arg_length(fname, args, max_fname_arg_count, compat_args) -> None:
    # [FUNCTION DOC]
    # HELPER: for the positional-argument validator. Inputs: fname, function name for
    # messages; args, the positional values passed; max_fname_arg_count, count of
    # other arguments the function accepts (non-negative); compat_args, mapping of
    # allowed compatibility names to defaults. Raises a value error if the count is
    # negative, and a type error that mimics the interpreter's too-many-arguments
    # message (singular or plural wording, counts include the other arguments) when
    # more values are passed than compat_args has entries.
    """
    Checks whether 'args' has length of at most 'compat_args'. Raises
    a TypeError if that is not the case, similar to in Python when a
    function is called with too many arguments.
    """
    if max_fname_arg_count < 0:
        raise ValueError("'max_fname_arg_count' must be non-negative")

    if len(args) > len(compat_args):
        max_arg_count = len(compat_args) + max_fname_arg_count
        actual_arg_count = len(args) + max_fname_arg_count
        argument = "argument" if max_arg_count == 1 else "arguments"

        raise TypeError(
            f"{fname}() takes at most {max_arg_count} {argument} "
            f"({actual_arg_count} given)"
        )


def _check_for_default_values(fname, arg_val_dict, compat_args) -> None:
    # [FUNCTION DOC]
    # HELPER: verifies that each given value equals its default. Inputs: fname, for
    # messages; arg_val_dict, mapping of names to passed values (all names must exist
    # in compat_args, which the caller guarantees); compat_args, mapping of names to
    # defaults. Equality is tried directly, except that a none-versus-not-none
    # mismatch is a failed match without comparing (to avoid array comparisons); if the
    # comparison result is not a boolean (for example an array), identity is used
    # instead. Raises a value error naming the parameter and function at the first
    # value that does not match.
    """
    Check that the keys in `arg_val_dict` are mapped to their
    default values as specified in `compat_args`.

    Note that this function is to be called only when it has been
    checked that arg_val_dict.keys() is a subset of compat_args
    """
    for key in arg_val_dict:
        # try checking equality directly with '=' operator,
        # as comparison may have been overridden for the left
        # hand object
        try:
            v1 = arg_val_dict[key]
            v2 = compat_args[key]

            # check for None-ness otherwise we could end up
            # comparing a numpy array vs None
            if (v1 is not None and v2 is None) or (v1 is None and v2 is not None):
                match = False
            else:
                match = v1 == v2

            if not is_bool(match):
                raise ValueError("'match' is not a boolean")

        # could not compare them directly, so try comparison
        # using the 'is' operator
        except ValueError:
            match = arg_val_dict[key] is compat_args[key]

        if not match:
            raise ValueError(
                f"the '{key}' parameter is not supported in "
                f"the pandas implementation of {fname}()"
            )


def validate_args(fname, args, max_fname_arg_count, compat_args) -> None:
    # [FUNCTION DOC]
    # Validates the positional extra arguments of a function that exists only for
    # numpy compatibility. Inputs: fname, function name used in messages;
    # args, the tuple of positional values; max_fname_arg_count, non-negative number
    # of other arguments, used only for the error message; compat_args, ordered
    # mapping of allowed names to default values (order matches positional order).
    # Raises a type error if there are more values than names, then a value error
    # if any value differs from its default. Returns nothing.
    """
    Checks whether the length of the `*args` argument passed into a function
    has at most `len(compat_args)` arguments and whether or not all of these
    elements in `args` are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `*args` parameter
    args : tuple
        The `*args` parameter passed into a function
    max_fname_arg_count : int
        The maximum number of arguments that the function `fname`
        can accept, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args : dict
        A dictionary of keys and their associated default values.
        In order to accommodate buggy behaviour in some versions of `numpy`,
        where a signature displayed keyword arguments but then passed those
        arguments **positionally** internally when calling downstream
        implementations, a dict ensures that the original
        order of the keyword arguments is enforced.

    Raises
    ------
    TypeError
        If `args` contains more values than there are `compat_args`
    ValueError
        If `args` contains values that do not correspond to those
        of the default values specified in `compat_args`
    """
    _check_arg_length(fname, args, max_fname_arg_count, compat_args)

    # We do this so that we can provide a more informative
    # error message about the parameters that we are not
    # supporting in the pandas implementation of 'fname'
    kwargs = dict(zip(compat_args, args, strict=False))
    _check_for_default_values(fname, kwargs, compat_args)


def _check_for_invalid_keys(fname, kwargs, compat_args) -> None:
    # [FUNCTION DOC]
    # HELPER: raises a type error naming one (arbitrary) key of the passed mapping
    # that is not among the allowed compatibility names; does nothing if all keys are
    # allowed. Inputs: fname, kwargs, compat_args.
    """
    Checks whether 'kwargs' contains any keys that are not
    in 'compat_args' and raises a TypeError if there is one.
    """
    # set(dict) --> set of the dictionary's keys
    diff = set(kwargs) - set(compat_args)

    if diff:
        bad_arg = next(iter(diff))
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")


def validate_kwargs(fname, kwargs, compat_args) -> None:
    # [FUNCTION DOC]
    # Validates keyword extras of a compatibility-only function. Inputs: fname, the
    # function name for messages; kwargs, mapping of passed keywords; compat_args,
    # mapping of allowed keywords to defaults. Raises a type error for any keyword
    # not allowed, and a value error for any allowed keyword whose value is not its
    # default. Returns nothing; does not modify the input.
    """
    Checks whether parameters passed to the **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `**kwargs` parameter
    kwargs : dict
        The `**kwargs` parameter passed into `fname`
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to have and their
        associated default values

    Raises
    ------
    TypeError if `kwargs` contains keys not in `compat_args`
    ValueError if `kwargs` contains keys in `compat_args` that do not
    map to the default values specified in `compat_args`
    """
    kwds = kwargs.copy()
    _check_for_invalid_keys(fname, kwargs, compat_args)
    _check_for_default_values(fname, kwds, compat_args)


def validate_args_and_kwargs(
    fname, args, kwargs, max_fname_arg_count, compat_args
) -> None:
    # [FUNCTION DOC]
    # Validates both positional and keyword extras of a compatibility-only function.
    # Inputs: fname; args, tuple of positional values; kwargs, mapping of keywords;
    # max_fname_arg_count, number of other arguments accepted (non-negative), used
    # for the error message; compat_args, ordered mapping of allowed names to defaults.
    # Checks the combined count against the allowed names (type error), rejects names
    # passed both positionally and by keyword (type error), then validates all values
    # as keywords (type error for unknown keywords, value error for non-default
    # values). Side effect: the positional values are added into the caller's kwargs
    # mapping.
    # KNOWN ISSUE: the docstring describes max_fname_arg_count as the minimum number
    # of arguments required, but the code (and the sibling positional validator)
    # treats it as a maximum accepted count for error messages. The docstring also
    # says positional values must equal none, whereas they are compared to the
    # defaults in compat_args.
    """
    Checks whether parameters passed to the *args and **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname: str
        The name of the function being passed the `**kwargs` parameter
    args: tuple
        The `*args` parameter passed into a function
    kwargs: dict
        The `**kwargs` parameter passed into `fname`
    max_fname_arg_count: int
        The minimum number of arguments that the function `fname`
        requires, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to
        have and their associated default values.

    Raises
    ------
    TypeError if `args` contains more values than there are
    `compat_args` OR `kwargs` contains keys not in `compat_args`
    ValueError if `args` contains values not at the default value (`None`)
    `kwargs` contains keys in `compat_args` that do not map to the default
    value as specified in `compat_args`

    See Also
    --------
    validate_args : Purely args validation.
    validate_kwargs : Purely kwargs validation.

    """
    # Check that the total number of arguments passed in (i.e.
    # args and kwargs) does not exceed the length of compat_args
    _check_arg_length(
        fname, args + tuple(kwargs.values()), max_fname_arg_count, compat_args
    )

    # Check there is no overlap with the positional and keyword
    # arguments, similar to what is done in actual Python functions
    args_dict = dict(zip(compat_args, args, strict=False))

    for key in args_dict:
        if key in kwargs:
            raise TypeError(
                f"{fname}() got multiple values for keyword argument '{key}'"
            )

    kwargs.update(args_dict)
    validate_kwargs(fname, kwargs, compat_args)


def validate_bool_kwarg(
    value: BoolishNoneT,
    arg_name: str,
    none_allowed: bool = True,
    int_allowed: bool = False,
) -> BoolishNoneT:
    # [FUNCTION DOC]
    # Checks that a value is a genuine boolean. Inputs: value; arg_name, used in the
    # message; none_allowed, default true, also accepts none; int_allowed, default
    # false, also accepts any integer (booleans count as integers). Returns the value
    # unchanged. Raises a value error ("expected type bool") naming the argument and
    # the received type when invalid.
    """
    Ensure that argument passed in arg_name can be interpreted as boolean.

    Parameters
    ----------
    value : bool
        Value to be validated.
    arg_name : str
        Name of the argument. To be reflected in the error message.
    none_allowed : bool, default True
        Whether to consider None to be a valid boolean.
    int_allowed : bool, default False
        Whether to consider integer value to be a valid boolean.

    Returns
    -------
    value
        The same value as input.

    Raises
    ------
    ValueError
        If the value is not a valid boolean.
    """
    good_value = is_bool(value)
    if none_allowed:
        good_value = good_value or (value is None)

    if int_allowed:
        good_value = good_value or isinstance(value, int)

    if not good_value:
        raise ValueError(
            f'For argument "{arg_name}" expected type bool, received '
            f"type {type(value).__name__}."
        )
    return value


def validate_na_arg(value, name: str):
    # [FUNCTION DOC]
    # Validates a missing-value-handling argument. Inputs: value; name, argument name
    # for the message. Accepts the library's no-default sentinel, a boolean, none, the
    # missing-value singleton, or a float that is not-a-number; returns nothing. Raises
    # a value error otherwise, naming the argument and received value.
    # KNOWN ISSUE: the docstring's section underline for the exceptions heading uses
    # underscores instead of dashes, which breaks the numpydoc format.
    """
    Validate na arguments.

    Parameters
    ----------
    value : object
        Value to validate.
    name : str
        Name of the argument, used to raise an informative error message.

    Raises
    ______
    ValueError
        When ``value`` is determined to be invalid.
    """
    if (
        value is lib.no_default
        or isinstance(value, bool)
        or value is None
        or value is NA
        or (lib.is_float(value) and np.isnan(value))
    ):
        return
    raise ValueError(f"{name} must be None, pd.NA, np.nan, True, or False; got {value}")


def validate_fillna_kwargs(value, method, validate_scalar_dict_value: bool = True):
    # [FUNCTION DOC]
    # Validates the value and method arguments of a fill-missing operation. Inputs:
    # value; method; validate_scalar_dict_value, default true, rejects lists and tuples
    # as value. Returns the pair (value, method), where a given method is normalized
    # by the fill-method cleaner (which may itself raise). Raises a value error if
    # neither or both are given, and a type error if value is a list or tuple while
    # that check is enabled. None is the "not given" marker for both.
    """
    Validate the keyword arguments to 'fillna'.

    This checks that exactly one of 'value' and 'method' is specified.
    If 'method' is specified, this validates that it's a valid method.

    Parameters
    ----------
    value, method : object
        The 'value' and 'method' keyword arguments for 'fillna'.
    validate_scalar_dict_value : bool, default True
        Whether to validate that 'value' is a scalar or dict. Specifically,
        validate that it is not a list or tuple.

    Returns
    -------
    value, method : object
    """
    from pandas.core.missing import clean_fill_method

    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    if value is None and method is not None:
        method = clean_fill_method(method)

    elif value is not None and method is None:
        if validate_scalar_dict_value and isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                f'you passed a "{type(value).__name__}"'
            )

    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    return value, method


def validate_percentile(q: float | Iterable[float]) -> np.ndarray:
    # [FUNCTION DOC]
    # Validates percentile input. Input: q, one number or an iterable of numbers.
    # Returns the input as a numpy array. Raises a value error if any element is
    # outside the closed unit interval (not-a-number also fails). The error message is
    # a plain string deliberately, to avoid formatting cost on the happy path.
    """
    Validate percentiles (used by describe and quantile).

    This function checks if the given float or iterable of floats is a valid percentile
    otherwise raises a ValueError.

    Parameters
    ----------
    q: float or iterable of floats
        A single percentile or an iterable of percentiles.

    Returns
    -------
    ndarray
        An ndarray of the percentiles if valid.

    Raises
    ------
    ValueError if percentiles are not in given interval([0, 1]).
    """
    q_arr = np.asarray(q)
    # Don't change this to an f-string. The string formatting
    # is too expensive for cases where we don't need it.
    msg = "percentiles should all be in the interval [0, 1]"
    if q_arr.ndim == 0:
        if not 0 <= q_arr <= 1:
            raise ValueError(msg)
    elif not all(0 <= qs <= 1 for qs in q_arr):
        raise ValueError(msg)
    return q_arr


@overload
def validate_ascending(ascending: BoolishT) -> BoolishT: ...
    # [FUNCTION DOC]
    # Type-checking overload: a single boolean-like value returns the same type.
    # The implementation is the third definition below.


@overload
def validate_ascending(ascending: Sequence[BoolishT]) -> list[BoolishT]: ...
    # [FUNCTION DOC]
    # Type-checking overload: a sequence of boolean-like values returns a list of the
    # same element type. The implementation is the next definition.


def validate_ascending(
    ascending: bool | int | Sequence[BoolishT],
) -> bool | int | list[BoolishT]:
    # [FUNCTION DOC]
    # Validates a sort-direction argument for index sorting. Input: ascending, a
    # boolean or integer, or a sequence of them. Each value must pass the boolean
    # validator with none disallowed and integers allowed (a string is a sequence and
    # so would be validated per character and fail). Returns the value itself for a
    # single input or a list of validated items for a sequence; raises a value error
    # via the boolean validator.
    """Validate ``ascending`` kwargs for ``sort_index`` method."""
    kwargs = {"none_allowed": False, "int_allowed": True}
    if not isinstance(ascending, Sequence):
        return validate_bool_kwarg(ascending, "ascending", **kwargs)

    return [validate_bool_kwarg(item, "ascending", **kwargs) for item in ascending]


def validate_endpoints(closed: str | None) -> tuple[bool, bool]:
    # [FUNCTION DOC]
    # Converts the closed argument into a pair of booleans (left closed, right
    # closed): none gives both true, "left" gives left only, "right" gives right only.
    # Raises a value error for any other value.
    """
    Check that the `closed` argument is among [None, "left", "right"]

    Parameters
    ----------
    closed : {None, "left", "right"}

    Returns
    -------
    left_closed : bool
    right_closed : bool

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")

    return left_closed, right_closed


def validate_inclusive(inclusive: str | None) -> tuple[bool, bool]:
    # [FUNCTION DOC]
    # Converts the inclusive argument into a pair of booleans (left included, right
    # included): "both" gives both true, "left" left only, "right" right only,
    # "neither" both false. Raises a value error for any other value, including non-text
    # input and none.
    """
    Check that the `inclusive` argument is among {"both", "neither", "left", "right"}.

    Parameters
    ----------
    inclusive : {"both", "neither", "left", "right"}

    Returns
    -------
    left_right_inclusive : tuple[bool, bool]

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_right_inclusive: tuple[bool, bool] | None = None

    if isinstance(inclusive, str):
        left_right_inclusive = {
            "both": (True, True),
            "left": (True, False),
            "right": (False, True),
            "neither": (False, False),
        }.get(inclusive)

    if left_right_inclusive is None:
        raise ValueError(
            "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        )

    return left_right_inclusive


def validate_insert_loc(loc: int, length: int) -> int:
    # [FUNCTION DOC]
    # Validates and normalizes an insertion location. Inputs: loc, expected integer;
    # length, the container length. A negative loc is shifted by length. Returns the
    # normalized location in the closed range from zero to length. Raises a type
    # error if loc is not an integer, and an index error if it is out of range
    # after normalization (messages mention the range negative length to length).
    """
    Check that we have an integer between -length and length, inclusive.

    Standardize negative loc to within [0, length].

    The exceptions we raise on failure match np.insert.
    """
    if not is_integer(loc):
        raise TypeError(f"loc must be an integer between -{length} and {length}")

    if loc < 0:
        loc += length
    if not 0 <= loc <= length:
        raise IndexError(f"loc must be an integer between -{length} and {length}")
    return loc  # pyright: ignore[reportReturnType]


def check_dtype_backend(dtype_backend) -> None:
    # [FUNCTION DOC]
    # Validates the dtype backend argument. Input: dtype_backend, either the no-default
    # sentinel (accepted, nothing checked) or one of the strings for the nullable numpy
    # backend or pyarrow. Raises a value error for any other value. Returns nothing.
    if dtype_backend is not lib.no_default:
        if dtype_backend not in ["numpy_nullable", "pyarrow"]:
            raise ValueError(
                f"dtype_backend {dtype_backend} is invalid, only 'numpy_nullable' and "
                f"'pyarrow' are allowed.",
            )
