# [MODULE DOC]
# HELPER: numba code generation for window operations. Contains cached
# factory functions that, given a user function or exponentially-weighted
# parameters plus the numba jit flags (nopython, nogil, parallel), build and
# return jit-compiled kernels: a rolling apply kernel over each window, an
# exponentially weighted mean/sum kernel (per series and per table), a
# table-wise rolling apply kernel, and a helper wrapper that emulates a
# reduction along an axis for nan-aware numpy aggregations.
# Used by the rolling, expanding and ewm window classes when the numba engine
# is selected. All factories are memoised (functools cache), so their
# arguments must be hashable. Relies on the shared jit-user-function helper in
# the core util numba module. Import-time side effects: none; numba is
# imported lazily inside each factory.

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.util.numba_ import jit_user_function

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import Scalar


@functools.cache
def generate_numba_apply_func(
    func: Callable[..., Scalar],
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    # [FUNCTION DOC]
    # HELPER: cached factory for the rolling apply kernel.
    # Inputs: func is the user function mapping a one-dimensional window of
    # values (plus extra positional arguments) to a scalar and is compiled
    # first; nopython, nogil and parallel are the numba jit flags applied to
    # the kernel. Memoised, so func must be hashable.
    # Returns: a compiled kernel taking the values array, window start and end
    # bound arrays, a minimum periods count and extra arguments, and returning
    # a float array with one entry per window: the user function result when
    # the window has at least minimum periods non-missing values, otherwise
    # missing.
    # Raises an import error if numba is not installed.
    """
    Generate a numba jitted apply function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the rolling apply function.

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
    def roll_apply(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
        *args: Any,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: the compiled kernel returned by the factory; see the
        # factory's documentation for arguments and result. Iterates (in
        # parallel when enabled) over the windows given by the begin and end
        # bounds.
        result = np.empty(len(begin))
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window))
            if len(window) - count_nan >= minimum_periods:
                result[i] = numba_func(window, *args)
            else:
                result[i] = np.nan
        return result

    return roll_apply


@functools.cache
def generate_numba_ewm_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: tuple,
    normalize: bool,
):
    # [FUNCTION DOC]
    # HELPER: cached factory for the exponentially weighted mean (normalize
    # true) or sum (normalize false) kernel, computed per group window on a
    # one-dimensional array.
    # Inputs: nopython, nogil, parallel are numba jit flags; com is the
    # centre-of-mass decay parameter (alpha is one over one plus com); adjust
    # selects adjusted weighting; ignore_na controls whether missing values
    # decay prior weight; deltas is a hashable tuple of time gaps between
    # consecutive observations; normalize selects mean versus sum. All must be
    # hashable because of memoisation.
    # Returns: a compiled kernel taking values, window begin and end bound
    # arrays and a minimum periods count and returning a float array the size
    # of values, where each window is processed independently and entries with
    # fewer than minimum periods observations are missing.
    # KNOWN ISSUE: the new observation weight is a local variable that is
    # reassigned for irregular time series (non-adjusted with com equal to
    # one) and never reset between windows, so it leaks from one window to the
    # next.
    """
    Generate a numba jitted ewm mean or sum function specified by values
    from engine_kwargs.

    Parameters
    ----------
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit
    com : float
    adjust : bool
    ignore_na : bool
    deltas : tuple
    normalize : bool

    Returns
    -------
    Numba function
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: the compiled kernel returned by the factory; see the
        # factory's documentation. For each window, seeds the weighted value
        # with the first element, then updates sequentially: decays the prior
        # weight (mean) or the running value (sum), blends in each observed
        # value, and emits missing until minimum periods observations have
        # been seen.
        # KNOWN ISSUE: the inner check on the normalize flag is redundant
        # because it is already inside a branch that requires normalize.
        result = np.empty(len(values))
        alpha = 1.0 / (1.0 + com)
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0 if adjust else alpha

        for i in numba.prange(len(begin)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            sub_result = np.empty(len(window))

            weighted = window[0]
            nobs = int(not np.isnan(weighted))
            sub_result[0] = weighted if nobs >= minimum_periods else np.nan
            old_wt = 1.0

            for j in range(1, len(window)):
                cur = window[j]
                is_observation = not np.isnan(cur)
                nobs += is_observation
                if not np.isnan(weighted):
                    if is_observation or not ignore_na:
                        if normalize:
                            # note that len(deltas) = len(vals) - 1 and deltas[i]
                            # is to be used in conjunction with vals[i+1]
                            old_wt *= old_wt_factor ** deltas[start + j - 1]
                            if not adjust and com == 1:
                                # update in case of irregular-interval time series
                                new_wt = 1.0 - old_wt
                        else:
                            weighted = old_wt_factor * weighted
                        if is_observation:
                            if normalize:
                                # avoid numerical errors on constant series
                                if weighted != cur:
                                    weighted = old_wt * weighted + new_wt * cur
                                    if normalize:
                                        weighted = weighted / (old_wt + new_wt)
                                if adjust:
                                    old_wt += new_wt
                                else:
                                    old_wt = 1.0
                            else:
                                weighted += cur
                elif is_observation:
                    weighted = cur

                sub_result[j] = weighted if nobs >= minimum_periods else np.nan

            result[start:stop] = sub_result

        return result

    return ewm


@functools.cache
def generate_numba_table_func(
    func: Callable[..., np.ndarray],
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    # [FUNCTION DOC]
    # HELPER: cached factory for the table-wise rolling apply kernel.
    # Inputs: func is the user function taking a two-dimensional window (rows
    # by columns) plus extra arguments and returning one value per column; it
    # is compiled first; nopython, nogil, parallel are numba jit flags.
    # Returns: a compiled kernel taking the 2D values, window begin and end
    # bounds, minimum periods and extra arguments, returning a 2D float array
    # with one row per window; entries are missing for columns that have fewer
    # than minimum periods non-missing values in the window (the function is
    # only invoked when at least one column qualifies).
    # Raises an import error if numba is not installed.
    """
    Generate a numba jitted function to apply window calculations table-wise.

    Func will be passed an M window size x N number of columns array, and
    must return a 1 x N number of columns array.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

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
    def roll_table(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
        *args: Any,
    ):
        # [FUNCTION DOC]
        # HELPER: the compiled kernel returned by the factory; see the
        # factory's documentation. Builds a result row per window and masks
        # entries lacking enough non-missing values.
        result = np.empty((len(begin), values.shape[1]))
        min_periods_mask = np.empty(result.shape)
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window), axis=0)
            nan_mask = len(window) - count_nan >= minimum_periods
            if nan_mask.any():
                result[i, :] = numba_func(window, *args)
            min_periods_mask[i, :] = nan_mask
        result = np.where(min_periods_mask, result, np.nan)
        return result

    return roll_table


# This function will no longer be needed once numba supports
# axis for all np.nan* agg functions
# https://github.com/numba/numba/issues/1269
@functools.cache
def generate_manual_numpy_nan_agg_with_axis(nan_func):
    # [FUNCTION DOC]
    # HELPER: cached factory emulating column-wise reduction for nan-aware
    # numpy aggregations, because numba does not support the axis argument for
    # them. Input: nan_func, a numba-compatible function reducing a
    # one-dimensional array to a scalar. Returns a compiled function that
    # takes a two-dimensional array and returns a one-dimensional array
    # holding the reduction of each column. Always compiled in no-python,
    # no-gil, parallel mode. Requires numba (import error otherwise).
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=True, nogil=True, parallel=True)
    def nan_agg_with_axis(table):
        # [FUNCTION DOC]
        # HELPER: compiled function returned by the factory; applies the
        # reduction to each column of the table and returns the per-column
        # results as a float array.
        result = np.empty(table.shape[1])
        for i in numba.prange(table.shape[1]):
            partition = table[:, i]
            result[i] = nan_func(partition)
        return result

    return nan_agg_with_axis


@functools.cache
def generate_numba_ewm_table_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: tuple,
    normalize: bool,
):
    # [FUNCTION DOC]
    # HELPER: cached factory for the table-wise exponentially weighted mean
    # (normalize true) or sum (normalize false) kernel. Same parameters as the
    # per-series factory: numba jit flags, com, adjust, ignore_na, a hashable
    # tuple of deltas and normalize, all hashable for memoisation.
    # Returns: a compiled kernel taking a 2D values array, begin and end
    # arrays (accepted but unused: the whole table is one window) and a
    # minimum periods count, returning a float array with the shape of values
    # with missing entries until each column has minimum periods observations.
    """
    Generate a numba jitted ewm mean or sum function applied table wise specified
    by values from engine_kwargs.

    Parameters
    ----------
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit
    com : float
    adjust : bool
    ignore_na : bool
    deltas : tuple
    normalize: bool

    Returns
    -------
    Numba function
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm_table(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: the compiled kernel returned by the factory; processes rows
        # sequentially, updating a weighted value and weight per column, with
        # the same update rules as the per-series kernel. The first row seeds
        # the weighted values.
        # KNOWN ISSUE: the new observation weight is a single scalar that is
        # overwritten with the weight of whichever column was last processed
        # in the irregular time series case (non-adjusted, com equal to one),
        # so it is shared across columns; also the inner check on the
        # normalize flag is redundant.
        alpha = 1.0 / (1.0 + com)
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0 if adjust else alpha
        old_wt = np.ones(values.shape[1])

        result = np.empty(values.shape)
        weighted = values[0].copy()
        nobs = (~np.isnan(weighted)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted, np.nan)
        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted[j]):
                    if is_observations[j] or not ignore_na:
                        if normalize:
                            # note that len(deltas) = len(vals) - 1 and deltas[i]
                            # is to be used in conjunction with vals[i+1]
                            old_wt[j] *= old_wt_factor ** deltas[i - 1]
                            if not adjust and com == 1:
                                # update in case of irregular-interval time series
                                new_wt = 1.0 - old_wt[j]
                        else:
                            weighted[j] = old_wt_factor * weighted[j]
                        if is_observations[j]:
                            if normalize:
                                # avoid numerical errors on constant series
                                if weighted[j] != cur[j]:
                                    weighted[j] = (
                                        old_wt[j] * weighted[j] + new_wt * cur[j]
                                    )
                                    if normalize:
                                        weighted[j] = weighted[j] / (old_wt[j] + new_wt)
                                if adjust:
                                    old_wt[j] += new_wt
                                else:
                                    old_wt[j] = 1.0
                            else:
                                weighted[j] += cur[j]
                elif is_observations[j]:
                    weighted[j] = cur[j]

            result[i] = np.where(nobs >= minimum_periods, weighted, np.nan)

        return result

    return ewm_table
