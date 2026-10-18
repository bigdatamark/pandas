# [MODULE DOC]
# HELPER: internal module with a small helper shared by the numba kernels. Public
# contents: is_monotonic_increasing, an eagerly compiled numba function (the
# signature is declared up front, so compilation happens at import time and
# requires numba). Used by the sum, mean and variance window kernels to decide
# whether window bounds allow incremental updates.
from __future__ import annotations

from typing import TYPE_CHECKING

import numba

if TYPE_CHECKING:
    import numpy as np


@numba.jit(
    # error: Any? not callable
    numba.boolean(numba.int64[:]),  # type: ignore[misc]
    nopython=True,
    nogil=True,
    parallel=False,
)
def is_monotonic_increasing(bounds: np.ndarray) -> bool:
    # [FUNCTION DOC]
    # HELPER: input is a one dimensional array of 64-bit integers. Returns true if
    # the values never decrease (equal neighbours are allowed; arrays with fewer
    # than two elements are considered increasing), false otherwise. No side
    # effects.
    """Check if int64 values are monotonically increasing."""
    n = len(bounds)
    if n < 2:
        return True
    prev = bounds[0]
    for i in range(1, n):
        cur = bounds[i]
        if cur < prev:
            return False
        prev = cur
    return True
