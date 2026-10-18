"""
Public API for Rolling Window Indexers.
"""

# [PACKAGE DOC]
# Public package for rolling-window indexer classes and indexer validation.
# Re-exports (all listed in __all__): BaseIndexer (base class for custom window
# indexers), FixedForwardWindowIndexer, VariableOffsetWindowIndexer, and the
# check_array_indexer validation function (despite the package docstring
# mentioning only rolling windows). Defines nothing itself; no import-time side
# effects beyond the imports.
from pandas.core.indexers import check_array_indexer
from pandas.core.indexers.objects import (
    BaseIndexer,
    FixedForwardWindowIndexer,
    VariableOffsetWindowIndexer,
)

__all__ = [
    "BaseIndexer",
    "FixedForwardWindowIndexer",
    "VariableOffsetWindowIndexer",
    "check_array_indexer",
]
