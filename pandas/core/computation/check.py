# [MODULE DOC]
# Detects whether the optional numexpr accelerator library is available.
# Public contents: a module-level handle to the numexpr module (or an empty
# value when it is missing or too old) and a boolean flag, NUMEXPR_INSTALLED,
# which is true exactly when that handle is present. Used by the engines,
# expressions and eval modules to choose between engines.
# Import-time side effect: attempts to import numexpr, emitting a warning
# (not an error) when it is installed but older than the minimum supported
# version; a plain missing install is silent.
from __future__ import annotations

from pandas.compat._optional import import_optional_dependency

ne = import_optional_dependency("numexpr", errors="warn")
NUMEXPR_INSTALLED = ne is not None

__all__ = ["NUMEXPR_INSTALLED"]
