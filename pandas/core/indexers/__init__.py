# [PACKAGE DOC]
# Package of helpers for validating, normalizing and inspecting indexers (the
# keys used for getitem and setitem style operations), plus the window
# indexer classes used by rolling and expanding operations (in the objects
# submodule, which is not re-exported here).
# Submodules: utils (indexer validation and conversion helpers) and objects
# (window bound calculators).
# Re-exports the public helper functions from the utils submodule and lists
# them in the module-level export list. Importing this package only imports
# the utils submodule; there are no other import-time side effects.
from pandas.core.indexers.utils import (
    check_array_indexer,
    check_key_length,
    check_setitem_lengths,
    disallow_ndim_indexing,
    getitem_returns_view,
    is_empty_indexer,
    is_list_like_indexer,
    is_scalar_indexer,
    is_valid_positional_slice,
    length_of_indexer,
    maybe_convert_indices,
    unpack_1tuple,
    unpack_tuple_and_ellipses,
    validate_indices,
)

__all__ = [
    "check_array_indexer",
    "check_key_length",
    "check_setitem_lengths",
    "disallow_ndim_indexing",
    "getitem_returns_view",
    "is_empty_indexer",
    "is_list_like_indexer",
    "is_scalar_indexer",
    "is_valid_positional_slice",
    "length_of_indexer",
    "maybe_convert_indices",
    "unpack_1tuple",
    "unpack_tuple_and_ellipses",
    "validate_indices",
]
