"""
Public API for function executor engines to be used with ``map`` and ``apply``.
"""

# [PACKAGE DOC]
# Public package for plugging in alternative execution engines for the apply
# and map methods. Re-exports (and lists in __all__) the single base class
# BaseExecutionEngine from the core apply module, which third-party engines
# subclass. No other contents and no import-time side effects beyond the
# import.
from pandas.core.apply import BaseExecutionEngine

__all__ = ["BaseExecutionEngine"]
