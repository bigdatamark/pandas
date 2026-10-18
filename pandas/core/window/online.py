# [MODULE DOC]
# Numba support for online (streaming) exponentially weighted mean
# calculations used by the exponentially weighted window machinery.
# Public contents: a factory that builds a jit-compiled function computing an
# online exponentially weighted mean over a two dimensional array, and a small
# state holder class that keeps the running weights between successive chunks
# of data.
# Relationships: used by the online exponentially weighted window class in the
# ewm module, which passes the generated function into the state holder.
# Import-time side effects: none; numba is only imported (as an optional
# dependency) when the factory is called.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency


def generate_online_numba_ewma_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    # [FUNCTION DOC]
    # Factory returning a numba compiled online exponentially weighted mean
    # kernel.
    # Inputs: nopython, nogil and parallel are booleans forwarded unchanged to
    # the numba jit decorator (from the user's engine keyword arguments).
    # Returns: the compiled kernel function (see its own documentation).
    # Raises: an import error if the optional numba dependency is missing.
    # Side effects: imports numba lazily; compilation happens on first call of
    # the returned kernel.
    """
    Generate a numba jitted groupby ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
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
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def online_ewma(
        values: np.ndarray,
        deltas: np.ndarray,
        minimum_periods: int,
        old_wt_factor: float,
        new_wt: float,
        old_wt: np.ndarray,
        adjust: bool,
        ignore_na: bool,
    ):
        # [FUNCTION DOC]
        # HELPER: nested numba kernel built by the factory above.
        # Inputs: values is a 2D float array (rows are time steps, columns are
        # independent series); deltas is an array of time gaps between
        # observations; minimum_periods is the minimum number of non-missing
        # observations required before a result is emitted; old_wt_factor is
        # one minus alpha; new_wt is the weight of a new observation; old_wt is
        # a 1D array (one entry per column) of accumulated weights carried over
        # from a previous call and mutated in place; adjust selects the
        # adjusted weighting scheme (accumulate weights) versus recursive
        # (reset old weight to one); ignore_na controls whether missing values
        # still decay the old weight.
        # Returns: a tuple of the result array (same shape as values, with
        # missing where fewer than minimum_periods observations were seen) and
        # the updated old weights.
        # The first row seeds the weighted average; later rows update it per
        # column, and a column that is still missing adopts the first observed
        # value.
        # KNOWN ISSUE: the decay exponent is taken from the deltas entry at the
        # column position (column index minus one) rather than the row, which
        # contradicts the adjacent comment saying deltas are paired with rows
        # and looks like an indexing mistake (wraps to the last element for
        # column zero).
        """
        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        """
        result = np.empty(values.shape)
        weighted_avg = values[0].copy()
        nobs = (~np.isnan(weighted_avg)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)

        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted_avg[j]):
                    if is_observations[j] or not ignore_na:
                        # note that len(deltas) = len(vals) - 1 and deltas[i] is to be
                        # used in conjunction with vals[i+1]
                        old_wt[j] *= old_wt_factor ** deltas[j - 1]
                        if is_observations[j]:
                            # avoid numerical errors on constant series
                            if weighted_avg[j] != cur[j]:
                                weighted_avg[j] = (
                                    (old_wt[j] * weighted_avg[j]) + (new_wt * cur[j])
                                ) / (old_wt[j] + new_wt)
                            if adjust:
                                old_wt[j] += new_wt
                            else:
                                old_wt[j] = 1.0
                elif is_observations[j]:
                    weighted_avg[j] = cur[j]

            result[i] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)

        return result, old_wt

    return online_ewma


class EWMMeanState:
    # [CLASS DOC]
    # Holds the running state of an online exponentially weighted mean between
    # successive update calls.
    # Attributes: shape (shape of the data being tracked, last entry is the
    # number of columns), adjust and ignore_na flags, new_wt (weight of a new
    # observation: one when adjusting, otherwise alpha), old_wt_factor (one
    # minus alpha), old_wt (per-column array of accumulated old weights,
    # initially ones) and last_ewm (last emitted row of results, or none
    # before any data was processed).
    # Methods: run_ewm applies a kernel to a chunk and updates state; reset
    # restores the initial state.
    # Plain object with no base class.
    def __init__(self, com, adjust, ignore_na, shape) -> None:
        # [FUNCTION DOC]
        # Initialise the state. com is the centre-of-mass decay parameter
        # (alpha is derived as one over one plus com); adjust and ignore_na are
        # stored booleans; shape is the data shape whose last dimension sizes
        # the weight array. No validation is performed.
        alpha = 1.0 / (1.0 + com)
        self.shape = shape
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.new_wt = 1.0 if adjust else alpha
        self.old_wt_factor = 1.0 - alpha
        self.old_wt = np.ones(self.shape[-1])
        self.last_ewm = None

    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func):
        # [FUNCTION DOC]
        # Process one chunk of data with the compiled kernel.
        # Inputs: weighted_avg is the 2D chunk of values (despite its name it
        # holds raw values), deltas the time gaps, min_periods the minimum
        # observations, ewm_func the kernel produced by the factory function.
        # Returns the result array from the kernel. Side effects: replaces the
        # stored old weights with the kernel's updated weights and records the
        # last result row in last_ewm.
        result, old_wt = ewm_func(
            weighted_avg,
            deltas,
            min_periods,
            self.old_wt_factor,
            self.new_wt,
            self.old_wt,
            self.adjust,
            self.ignore_na,
        )
        self.old_wt = old_wt
        self.last_ewm = result[-1]
        return result

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Reset the state: old weights return to all ones (one per column) and
        # last_ewm is cleared. Returns nothing.
        self.old_wt = np.ones(self.shape[-1])
        self.last_ewm = None
