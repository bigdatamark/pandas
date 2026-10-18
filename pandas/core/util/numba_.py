"""Common utilities for Numba operations"""
# [MODULE DOC]
# HELPER module: shared utilities for the numba engine used by window, groupby and
# apply code paths.
# Contents: a module-level global switch (default false) recording whether numba is
# enabled by default; functions to decide whether to use numba, to set the global
# switch, to normalize jit options, to mark a user function as jit-compilable, and
# to move keyword arguments into positional ones for jitted calls; and a private
# sentinel object.
# Relationships: depends on the optional-dependency importer (numba is imported
# lazily, only when needed) and on the pandas numba error class.
# Import-time side effects: none (numba is not imported at import time).

from __future__ import annotations

import inspect
import types
from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError

if TYPE_CHECKING:
    from collections.abc import Callable

GLOBAL_USE_NUMBA: bool = False


def maybe_use_numba(engine: str | None) -> bool:
    # [FUNCTION DOC]
    # Decide whether numba routines should be used.
    # Input: engine, a string or None. Returns true when engine is the string "numba",
    # or when engine is None and the global numba switch is enabled; otherwise false.
    # No side effects.
    """Signal whether to use numba routines."""
    return engine == "numba" or (engine is None and GLOBAL_USE_NUMBA)


def set_use_numba(enable: bool = False) -> None:
    # [FUNCTION DOC]
    # Set the global default for using numba (the module-level switch).
    # Input: enable (default false). When true, the optional numba dependency is
    # imported first, so an import error is raised from the optional-dependency
    # importer if numba is not installed, and the switch is left unchanged in that
    # case. Side effect: mutates the module-level switch. Returns nothing.
    global GLOBAL_USE_NUMBA
    if enable:
        import_optional_dependency("numba")
    GLOBAL_USE_NUMBA = enable


def get_jit_arguments(engine_kwargs: dict[str, bool] | None = None) -> dict[str, bool]:
    # [FUNCTION DOC]
    # Build the numba jit options from user-supplied engine keyword arguments.
    # Input: engine_kwargs, a mapping or None (default, treated as empty).
    # Returns a new mapping with exactly three boolean entries: nopython (default
    # true), nogil (default false) and parallel (default false), taking the user's
    # value for each when present. Other keys in the input are ignored. The input is
    # not mutated.
    # KNOWN ISSUE: the docstring says this function raises the numba utility error,
    # but nothing in the body raises it (unknown keys are silently ignored).
    """
    Return arguments to pass to numba.JIT, falling back on pandas default JIT settings.

    Parameters
    ----------
    engine_kwargs : dict, default None
        user passed keyword arguments for numba.JIT

    Returns
    -------
    dict[str, bool]
        nopython, nogil, parallel

    Raises
    ------
    NumbaUtilError
    """
    if engine_kwargs is None:
        engine_kwargs = {}

    nopython = engine_kwargs.get("nopython", True)
    nogil = engine_kwargs.get("nogil", False)
    parallel = engine_kwargs.get("parallel", False)
    return {"nopython": nopython, "nogil": nogil, "parallel": parallel}


def jit_user_function(func: Callable) -> Callable:
    # [FUNCTION DOC]
    # Make a user function usable from numba-compiled code.
    # Input: func, a callable. Imports numba lazily (import error if it is missing).
    # Returns func unchanged if it is already numba-jitted, or if it is the numpy
    # attribute of the same name as the function or a builtin function (registering
    # those would break registration); otherwise returns the result of registering
    # func as jit-able with numba. The input function is not mutated by pandas.
    """
    If user function is not jitted already, mark the user's function
    as jitable.

    Parameters
    ----------
    func : function
        user defined function

    Returns
    -------
    function
        Numba JITed function, or function marked as JITable by numba
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if numba.extending.is_jitted(func):
        # Don't jit a user passed jitted function
        numba_func = func
    elif getattr(np, func.__name__, False) is func or isinstance(
        func, types.BuiltinFunctionType
    ):
        # Not necessary to jit builtins or np functions
        # This will mess up register_jitable
        numba_func = func  # type: ignore[assignment]
    else:
        numba_func = numba.extending.register_jitable(func)  # type: ignore[arg-type]

    return numba_func


_sentinel = object()


def prepare_function_arguments(
    func: Callable, args: tuple, kwargs: dict, *, num_required_args: int
) -> tuple[tuple, dict]:
    # [FUNCTION DOC]
    # Move keyword arguments into positional ones, because numba functions cannot
    # take keyword arguments.
    # Inputs: func, the user function; args, the user's positional arguments; kwargs,
    # the user's keyword arguments; num_required_args (keyword-only), the count of
    # leading positional parameters of func that pandas itself supplies (for example
    # two for groupby kernels taking values and index).
    # Returns a pair (args, kwargs). If kwargs is empty both inputs are returned
    # untouched. Otherwise the function signature is bound using placeholder objects
    # for the required leading arguments plus the user's args and kwargs, defaults
    # are applied, and the resulting positional arguments (with the leading
    # placeholders dropped) and the leftover keyword arguments are returned.
    # Raises the numba utility error if keyword arguments remain after binding
    # (keyword-only parameters); signature binding itself may raise a type error if
    # the arguments do not fit the signature.
    """
    Prepare arguments for jitted function. As numba functions do not support kwargs,
    we try to move kwargs into args if possible.

    Parameters
    ----------
    func : function
        User defined function
    args : tuple
        User input positional arguments
    kwargs : dict
        User input keyword arguments
    num_required_args : int
        The number of leading positional arguments we will pass to udf.
        These are not supplied by the user.
        e.g. for groupby we require "values", "index" as the first two arguments:
        `numba_func(group, group_index, *args)`, in this case num_required_args=2.
        See :func:`pandas.core.groupby.numba_.generate_numba_agg_func`

    Returns
    -------
    tuple[tuple, dict]
        args, kwargs

    """
    if not kwargs:
        return args, kwargs

    # the udf should have this pattern: def udf(arg1, arg2, ..., *args, **kwargs):...
    signature = inspect.signature(func)
    arguments = signature.bind(*[_sentinel] * num_required_args, *args, **kwargs)
    arguments.apply_defaults()
    # Ref: https://peps.python.org/pep-0362/
    # Arguments which could be passed as part of either *args or **kwargs
    # will be included only in the BoundArguments.args attribute.
    args = arguments.args
    kwargs = arguments.kwargs

    if kwargs:
        # Note: in case numba supports keyword-only arguments in
        # a future version, we should remove this check. But this
        # seems unlikely to happen soon.

        raise NumbaUtilError(
            "numba does not support keyword-only arguments"
            "https://github.com/numba/numba/issues/2916, "
            "https://github.com/numba/numba/issues/6846"
        )

    args = args[num_required_args:]
    return args, kwargs
