# ruff: noqa: TC004
# [PACKAGE DOC]
# Package for time series related public utilities (frequency inference and
# date offsets). It holds the submodules frequencies, holiday, api and offsets.
# The package itself defines no runtime names: the import of the two public
# submodules and the declaration of the public name list happen only inside a
# type-checking-only conditional block, so they are visible to static type
# checkers but nothing is imported or exported at runtime and importing the
# package has no side effects.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions:
    from pandas.tseries import (
        frequencies,
        offsets,
    )

    # and mark only those modules as public
    __all__ = ["frequencies", "offsets"]
