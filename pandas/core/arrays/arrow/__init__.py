# [PACKAGE DOC]
# Package for the pyarrow-backed extension array and its accessors.
# Submodules: accessors (list and struct accessor classes), array (the main
# arrow-backed extension array class), extension_types (legacy pyarrow
# extension types for interval and period data) and an internal utility module
# for converting pyarrow arrays to numpy.
# Re-exports the arrow extension array class and the list and struct accessor
# classes, listed in the module's export list.
# Importing this package imports the accessors and array modules, which in turn
# require pyarrow to be installed.
from pandas.core.arrays.arrow.accessors import (
    ListAccessor,
    StructAccessor,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray

__all__ = ["ArrowExtensionArray", "ListAccessor", "StructAccessor"]
