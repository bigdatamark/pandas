# [MODULE DOC]
# HELPER: internal module of the numba engine. It builds and caches numba-compiled
# loopers that apply a 1D aggregation kernel to every row of a 2D array (pandas
# passes the transposed block so rows are columns), and defines dtype mappings
# from input dtype to result dtype.
# Public contents: generate_apply_looper, make_looper, generate_shared_aggregator
# and the three dictionaries default_dtype_mapping, float_dtype_mapping and
# identity_dtype_mapping. Used by the window and groupby numba code paths and by
# the kernels subpackage. Numba is imported lazily (optional dependency) inside
# the functions, so importing this module has no side effects beyond
# creating the mappings and the caches.
from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import Scalar

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.util.numba_ import jit_user_function


@functools.cache
def generate_apply_looper(func, nopython=True, nogil=True, parallel=False):
    # [FUNCTION DOC]
    # Builds (and caches, keyed on all arguments) a numba-jitted looper that applies
    # a user function along one axis of a 2D array.
    # Parameters: func is the user function applied to each 1D slice; nopython,
    # nogil and parallel (defaults true, true, false) are forwarded to the jit
    # decorator. The user function is first wrapped to be numba compatible.
    # Returns the jitted looper taking the 2D values, an axis (0 means apply to each
    # column, otherwise to each row) and extra positional arguments. The looper
    # evaluates the function on the first slice to discover the output shape, then
    # allocates a float buffer with one entry (or one row/column of the output
    # length) per slice and fills the rest using parallel range loops.
    # Raises an import error if numba is not installed. No other side effects.
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")
    nb_compat_func = jit_user_function(func)

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def nb_looper(values, axis, *args):
        # [FUNCTION DOC]
        # HELPER: the jitted loop returned by generate_apply_looper.
        # Inputs: values is a 2D array, axis selects whether columns (zero) or rows
        # (non-zero) are the slices, args are extra arguments passed to the user
        # function. Returns a float array holding the stacked per-slice results
        # (transposed layout when applying along columns).
        # Operate on the first row/col in order to get
        # the output shape
        if axis == 0:
            first_elem = values[:, 0]
            dim0 = values.shape[1]
        else:
            first_elem = values[0]
            dim0 = values.shape[0]
        res0 = nb_compat_func(first_elem, *args)
        # Use np.asarray to get shape for
        # https://github.com/numba/numba/issues/4202#issuecomment-1185981507
        # Use tuple concatenation; numba doesn't support tuple unpacking syntax
        buf_shape = (dim0,) + np.atleast_1d(np.asarray(res0)).shape  # noqa: RUF005
        if axis == 0:
            buf_shape = buf_shape[::-1]
        buff = np.empty(buf_shape)

        if axis == 1:
            buff[0] = res0
            for i in numba.prange(1, values.shape[0]):
                buff[i] = nb_compat_func(values[i], *args)
        else:
            buff[:, 0] = res0
            for j in numba.prange(1, values.shape[1]):
                buff[:, j] = nb_compat_func(values[:, j], *args)
        return buff

    return nb_looper


@functools.cache
def make_looper(func, result_dtype, is_grouped_kernel, nopython, nogil, parallel):
    # [FUNCTION DOC]
    # Builds (and caches on all arguments) a numba-jitted column looper around a 1D
    # kernel.
    # Parameters: func is the kernel, called per row of a 2D array with the row,
    # the result dtype, either (labels, ngroups) or (start, end) bounds, the
    # minimum periods and extra arguments; it must return an output array and
    # an array of positions that should become missing. result_dtype is the
    # dtype of the allocated result; is_grouped_kernel chooses the groupby
    # signature (labels, number of groups, minimum periods) versus the window
    # signature (start, end, minimum periods); nopython, nogil, parallel are
    # forwarded to the jit decorator.
    # Returns the jitted looper, which returns a pair: the 2D result (one row per
    # input row; width is the number of groups or the number of windows) and a
    # dictionary mapping row index to the array of missing positions (only for
    # rows that have any). Raises an import error if numba is missing.
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if is_grouped_kernel:

        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def column_looper(
            values: np.ndarray,
            labels: np.ndarray,
            ngroups: int,
            min_periods: int,
            *args,
        ):
            # [FUNCTION DOC]
            # HELPER: jitted looper (grouped variant) returned by make_looper;
            # applies the kernel to every row of values using group labels and
            # returns the result array plus the dictionary of missing positions.
            result = np.empty((values.shape[0], ngroups), dtype=result_dtype)
            na_positions = {}
            for i in numba.prange(values.shape[0]):
                output, na_pos = func(
                    values[i], result_dtype, labels, ngroups, min_periods, *args
                )
                result[i] = output
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            return result, na_positions

    else:

        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        # error: Incompatible redefinition (redefinition with type
        # "Callable[[ndarray[Any, Any], ndarray[Any, Any], ndarray[Any, Any],
        # int, VarArg(Any)], Any]", original type "Callable[[ndarray[Any, Any],
        # ndarray[Any, Any], int, int, VarArg(Any)], Any]")
        def column_looper(  # type: ignore[misc]
            values: np.ndarray,
            start: np.ndarray,
            end: np.ndarray,
            min_periods: int,
            *args,
        ):
            # [FUNCTION DOC]
            # HELPER: jitted looper (window variant) returned by make_looper;
            # applies the kernel to every row of values using start/end window
            # bounds and returns the result array (one column per window) plus the
            # dictionary of missing positions.
            result = np.empty((values.shape[0], len(start)), dtype=result_dtype)
            na_positions = {}
            for i in numba.prange(values.shape[0]):
                output, na_pos = func(
                    values[i], result_dtype, start, end, min_periods, *args
                )
                result[i] = output
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            return result, na_positions

    return column_looper


default_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.int64,
    np.dtype("int16"): np.int64,
    np.dtype("int32"): np.int64,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint64,
    np.dtype("uint16"): np.uint64,
    np.dtype("uint32"): np.uint64,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex128,
    np.dtype("complex128"): np.complex128,
}


# TODO: Preserve complex dtypes

float_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.float64,
    np.dtype("int16"): np.float64,
    np.dtype("int32"): np.float64,
    np.dtype("int64"): np.float64,
    np.dtype("uint8"): np.float64,
    np.dtype("uint16"): np.float64,
    np.dtype("uint32"): np.float64,
    np.dtype("uint64"): np.float64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.float64,
    np.dtype("complex128"): np.float64,
}

identity_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.int8,
    np.dtype("int16"): np.int16,
    np.dtype("int32"): np.int32,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint8,
    np.dtype("uint16"): np.uint16,
    np.dtype("uint32"): np.uint32,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float32,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex64,
    np.dtype("complex128"): np.complex128,
}


def generate_shared_aggregator(
    func: Callable[..., Scalar],
    dtype_mapping: dict[np.dtype, np.dtype],
    is_grouped_kernel: bool,
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    # [FUNCTION DOC]
    # Returns a plain python wrapper (looper_wrapper) around the cached numba
    # looper for the given kernel. Parameters are as in make_looper, plus
    # dtype_mapping, a dictionary from input dtype to result dtype, looked up at
    # call time.
    # KNOWN ISSUE: the docstring says dtype_mapping may be None with a fallback to a
    # default mapping, but the code never handles None; a missing mapping would
    # fail on lookup. The docstring also lists a default for is_grouped_kernel that
    # the signature does not have.
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    dtype_mapping: dict or None
        If not None, maps a dtype to a result dtype.
        Otherwise, will fall back to default mapping.
    is_grouped_kernel: bool, default False
        Whether func operates using the group labels (True)
        or using starts/ends arrays

        If true, you also need to pass the number of groups to this function
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

    # A wrapper around the looper function,
    # to dispatch based on dtype since numba is unable to do that in nopython mode

    # It also post-processes the values by inserting nans where number of observations
    # is less than min_periods
    # Cannot do this in numba nopython mode
    # (you'll run into type-unification error when you cast int -> float)
    def looper_wrapper(
        values,
        start=None,
        end=None,
        labels=None,
        ngroups=None,
        min_periods: int = 0,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: the wrapper returned by generate_shared_aggregator.
        # Inputs: values is a 2D array (rows are series); start and end are window
        # bound arrays (used by non-grouped kernels); labels and ngroups are group
        # codes and group count (used by grouped kernels); min_periods (default 0)
        # is the minimum observation count; extra keyword arguments are passed
        # positionally (in order) to the kernel since numba only takes positional
        # arguments.
        # Behavior: picks the result dtype from the mapping by the dtype of values
        # (a missing dtype raises a key error), runs the cached looper, and then
        # post-processes missing positions: if the result is integer and any row
        # has missing positions, the whole result is converted to 64-bit float; the
        # missing positions are then set to NaN. Returns the 2D result array.
        result_dtype = dtype_mapping[values.dtype]
        column_looper = make_looper(
            func, result_dtype, is_grouped_kernel, nopython, nogil, parallel
        )
        # Need to unpack kwargs since numba only supports *args
        if is_grouped_kernel:
            result, na_positions = column_looper(
                values, labels, ngroups, min_periods, *kwargs.values()
            )
        else:
            result, na_positions = column_looper(
                values, start, end, min_periods, *kwargs.values()
            )
        if result.dtype.kind == "i":
            # Look if na_positions is not empty
            # If so, convert the whole block
            # This is OK since int dtype cannot hold nan,
            # so if min_periods not satisfied for 1 col, it is not satisfied for
            # all columns at that index
            for na_pos in na_positions.values():
                if len(na_pos) > 0:
                    result = result.astype("float64")
                    break
        # TODO: Optimize this
        for i, na_pos in na_positions.items():
            if len(na_pos) > 0:
                result[i, na_pos] = np.nan
        return result

    return looper_wrapper
