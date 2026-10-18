"""common utilities"""

# [MODULE DOC]
# Shared helper utilities for the indexing tests: a label generator and a smoke-test helper that attempts positional or label-based access on an object along chosen axes and tolerates specified failures. Imported by other indexing test modules; no tests live here and there are no import-time side effects.

from __future__ import annotations

from typing import (
    Any,
    Literal,
)


def _mklbl(prefix: str, n: int):
    # [FUNCTION DOC]
    # HELPER: builds a list of string labels. Takes a string prefix and a count n; returns a list of n strings made of the prefix followed by the integers zero up to n minus one.
    return [f"{prefix}{i}" for i in range(n)]


def check_indexing_smoketest_or_raises(
    obj,
    method: Literal["iloc", "loc"],
    key: Any,
    axes: Literal[0, 1] | None = None,
    fails=None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: smoke-test routine for indexing tests. Takes an object (series or frame), the accessor name (positional or label-based), a key, an optional axis (zero or one; both when none is given) and an optional exception class or tuple of classes called fails. For each selected axis that exists on the object it builds a full-slice tuple for all dimensions with the key placed on that axis and attempts to read through the accessor. If an index, type or key error is raised: when it is an instance of the fails argument the function returns immediately (without checking remaining axes), otherwise the error is re-raised. Returns nothing; asserts that the axis argument is zero or one when given.
    if axes is None:
        axes_list = [0, 1]
    else:
        assert axes in [0, 1]
        axes_list = [axes]

    for ax in axes_list:
        if ax < obj.ndim:
            # create a tuple accessor
            new_axes = [slice(None)] * obj.ndim
            new_axes[ax] = key
            axified = tuple(new_axes)
            try:
                getattr(obj, method).__getitem__(axified)
            except (IndexError, TypeError, KeyError) as detail:
                # if we are in fails, the ok, otherwise raise it
                if fails is not None:
                    if isinstance(detail, fails):
                        return
                raise
