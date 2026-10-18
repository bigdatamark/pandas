"""
Public API for extending pandas objects.
"""

# [PACKAGE DOC]
# Public package of the tools needed to extend pandas with custom objects.
# Re-exports (all listed in __all__): the no-default sentinel (from the compiled
# lib module), the ExtensionDtype base class and the decorator registering a
# dtype, the three accessor registration decorators (for frames, indexes and
# series), the take algorithm, the ExtensionArray base class and the mixin that
# adds scalar operators. Defines nothing itself; no import-time side effects
# beyond the imports.
from pandas._libs.lib import no_default

from pandas.core.dtypes.base import (
    ExtensionDtype,
    register_extension_dtype,
)

from pandas.core.accessor import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from pandas.core.algorithms import take
from pandas.core.arrays import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
)

__all__ = [
    "ExtensionArray",
    "ExtensionDtype",
    "ExtensionScalarOpsMixin",
    "no_default",
    "register_dataframe_accessor",
    "register_extension_dtype",
    "register_index_accessor",
    "register_series_accessor",
    "take",
]
