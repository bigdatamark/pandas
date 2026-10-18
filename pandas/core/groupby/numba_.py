"""Common utilities for Numba operations with groupby ops"""

# [MODULE DOC]
# Helpers for running user defined functions through the optional Numba engine
# in groupby aggregation and transformation. Contents: a validator for the
# user function signature and two cached factories that build jitted loops
# applying the user function to every group (one for aggregation, one for
# transformation).
# Relationships: builds on the shared Numba utilities in core.util.numba_
# (user function jitting and the Numba error type) and the optional
# dependency importer; consumed by the groupby machinery in groupby.py and ops.py.
# Import-time side effects: none (Numba itself is imported lazily, only when a
# factory is first called for a given argument combination).
from __future__ import annotations

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.util.numba_ import (
    NumbaUtilError,
    jit_user_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import Scalar


def validate_udf(func: Callable) -> None:
    # [FUNCTION DOC]
    # Checks that the given object is callable and that its first two
    # parameters are named values and index, in that order. Returns nothing.
    # Raises a not-implemented error when the object is not callable, and the
    # Numba utility error when fewer than two parameters exist or the first
    # two names differ from the required ones.
    # KNOWN ISSUE: the docstring says the func parameter has a default of
    # False, but there is no default; it is a required argument.
    """
    Validate user defined function for ops when using Numba with groupby ops.

    The first signature arguments should include:

    def f(values, index, ...):
        ...

    Parameters
    ----------
    func : function, default False
        user defined function

    Returns
    -------
    None

    Raises
    ------
    NumbaUtilError
    """
    if not callable(func):
        raise NotImplementedError(
            "Numba engine can only be used with a single function."
        )
    udf_signature = list(inspect.signature(func).parameters.keys())
    expected_args = ["values", "index"]
    min_number_args = len(expected_args)
    if (
        len(udf_signature) < min_number_args
        or udf_signature[:min_number_args] != expected_args
    ):
        raise NumbaUtilError(
            f"The first {min_number_args} arguments to {func.__name__} must be "
            f"{expected_args}"
        )


@functools.cache
def generate_numba_agg_func(
    func: Callable[..., Scalar],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    # [FUNCTION DOC]
    # Cached factory (results memoized per distinct argument combination, so
    # all arguments must be hashable). Takes the user function that reduces a
    # group of values plus its index to a scalar, and the three Numba compile
    # flags (nopython, nogil, parallel). Jits the user function, imports the
    # optional Numba dependency (raising the usual missing-dependency error if
    # absent) and returns a jitted loop function compiled with the same flags.
    """
    Generate a numba jitted agg function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby agg function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each group and will be JITed
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_agg(
        values: np.ndarray,
        index: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        num_columns: int,
        *args: Any,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: jitted inner loop returned by the aggregation factory.
        # Inputs: values is a 2-D array (rows by columns) already sorted by
        # group, index the matching label array, begin and end the per-group
        # start (inclusive) and stop (exclusive) row positions, num_columns the
        # column count, and extra positional arguments forwarded to the user
        # function. Asserts begin and end have equal length. Returns a float
        # array with one row per group and one column per input column, where
        # each cell is the user function applied to that group's column slice
        # and index slice. Groups and columns are looped with parallel ranges.
        assert len(begin) == len(end)
        num_groups = len(begin)

        result = np.empty((num_groups, num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i] : end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i] : end[i], j]
                result[i, j] = numba_func(group, group_index, *args)
        return result

    return group_agg


@functools.cache
def generate_numba_transform_func(
    func: Callable[..., np.ndarray],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    # [FUNCTION DOC]
    # Cached factory (memoized per distinct, hashable argument combination),
    # the transformation counterpart of the aggregation factory. Takes the
    # user function that maps a group's values and index to an array of the
    # same length, plus the Numba flags nopython, nogil and parallel. Jits the
    # user function, lazily imports the optional Numba dependency (raising the
    # usual missing-dependency error if absent) and returns the jitted loop.
    """
    Generate a numba jitted transform function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby transform function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each window and will be JITed
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_transform(
        values: np.ndarray,
        index: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        num_columns: int,
        *args: Any,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: jitted inner loop returned by the transformation factory.
        # Inputs are as for the aggregation loop: group-sorted 2-D values,
        # label array, per-group begin and end row positions, column count and
        # extra arguments for the user function. Asserts begin and end have
        # equal length. Returns a float array with the same number of rows as
        # values and num_columns columns, where each group's rows in each
        # column are filled with the user function output for that group.
        assert len(begin) == len(end)
        num_groups = len(begin)

        result = np.empty((len(values), num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i] : end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i] : end[i], j]
                result[begin[i] : end[i], j] = numba_func(group, group_index, *args)
        return result

    return group_transform
