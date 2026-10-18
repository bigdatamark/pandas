"""
Public toolkit API.
"""

# [PACKAGE DOC]
# Public package of dtype inspection and dtype utility helpers. Re-exports: the
# whole public namespace of the core dtypes api module via a star import (the
# is_* type-checking predicates, dtype-lookup helpers and so on), the
# infer_dtype function (from the compiled lib module), the union_categoricals
# function, and the CategoricalDtype, DatetimeTZDtype, IntervalDtype and
# PeriodDtype classes.
# KNOWN ISSUE: __all__ lists only the six explicitly imported names, so the
# many names pulled in by the star import are public attributes of this package
# but are absent from __all__.
# No import-time side effects beyond the imports.
from pandas._libs.lib import infer_dtype

from pandas.core.dtypes.api import *  # noqa: F403
from pandas.core.dtypes.concat import union_categoricals
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

__all__ = [
    "CategoricalDtype",
    "DatetimeTZDtype",
    "IntervalDtype",
    "PeriodDtype",
    "infer_dtype",
    "union_categoricals",
]
