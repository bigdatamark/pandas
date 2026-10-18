# [MODULE DOC]
# Public re-export surface for the reshape package. Collects the user-facing reshaping
# functions from the concat, encoding, melt, merge, pivot and tile submodules so that
# the top-level pandas namespace can import them from one place.
# Re-exported names: concat, crosstab, cut, from_dummies, get_dummies, lreshape, melt,
# merge, merge_asof, merge_ordered, pivot, pivot_table, qcut, wide_to_long.
# Defines no functions or classes of its own; the only import-time effect is importing
# all of those submodules. The explicit export list names exactly the re-exported items.
from pandas.core.reshape.concat import concat
from pandas.core.reshape.encoding import (
    from_dummies,
    get_dummies,
)
from pandas.core.reshape.melt import (
    lreshape,
    melt,
    wide_to_long,
)
from pandas.core.reshape.merge import (
    merge,
    merge_asof,
    merge_ordered,
)
from pandas.core.reshape.pivot import (
    crosstab,
    pivot,
    pivot_table,
)
from pandas.core.reshape.tile import (
    cut,
    qcut,
)

__all__ = [
    "concat",
    "crosstab",
    "cut",
    "from_dummies",
    "get_dummies",
    "lreshape",
    "melt",
    "merge",
    "merge_asof",
    "merge_ordered",
    "pivot",
    "pivot_table",
    "qcut",
    "wide_to_long",
]
