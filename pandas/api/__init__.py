"""public toolkit API"""

# [PACKAGE DOC]
# Public namespace gathering the supported extension and utility APIs of pandas
# as subpackages. Subpackages (all imported here and listed in __all__):
# executors (execution engine base class for apply and map), extensions (tools
# for building extension arrays, dtypes and accessors), indexers (rolling
# window indexers), interchange (dataframe interchange protocol), types (dtype
# checking and related helpers) and typing (classes useful for type
# annotations). A sibling module, internals, is not imported here. This package
# defines no objects of its own. Import-time side effect: imports the six
# subpackages and thereby the core modules they re-export.

from pandas.api import (
    executors,
    extensions,
    indexers,
    interchange,
    types,
    typing,
)

__all__ = [
    "executors",
    "extensions",
    "indexers",
    "interchange",
    "types",
    "typing",
]
