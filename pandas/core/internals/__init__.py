# [PACKAGE DOC]
# Internal block-management layer of pandas: stores the data of a frame or
# series as one or more typed blocks coordinated by a manager.
# Submodules: api (pseudo-public helpers for downstream libraries), blocks
# (block classes), concat (joining managers), construction (building managers
# from arrays, dicts, records), managers (manager classes), ops (block-wise
# operations).
# Re-exports: the pseudo-public block factory, the function that concatenates
# managers, the two manager classes. The names for the base block class and
# the extension and tz-aware datetime block classes are listed as exported but
# are not bound here; they are provided lazily through a module attribute hook.
# No import-time side effects other than importing those submodules.
from pandas.core.internals.api import make_block  # 2023-09-18 pyarrow uses this
from pandas.core.internals.concat import concatenate_managers
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
)

__all__ = [
    "Block",  # pyright:ignore[reportUnsupportedDunderAll]
    "BlockManager",
    "DatetimeTZBlock",  # pyright:ignore[reportUnsupportedDunderAll]
    "ExtensionBlock",  # pyright:ignore[reportUnsupportedDunderAll]
    "SingleBlockManager",
    "concatenate_managers",
    "make_block",
]


def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module-level attribute hook (lazy, deprecated attribute access).
    # Parameter name: the missing attribute name. For the manager-from-blocks
    # creation function and for the three block class names, it emits a future
    # deprecation warning (the pandas 4 warning class) at stack level 2 telling
    # users to use public APIs, then lazily imports and returns the object (the
    # tz-aware datetime block comes from the api module, the other blocks from the
    # blocks module, the creator from the managers module). Any other name raises
    # an attribute error naming this module.
    # GH#55139
    import warnings

    from pandas.errors import Pandas4Warning

    if name == "create_block_manager_from_blocks":
        # GH#33892, GH#58715
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            Pandas4Warning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )
        from pandas.core.internals.managers import create_block_manager_from_blocks

        return create_block_manager_from_blocks

    if name in [
        "Block",
        "ExtensionBlock",
        "DatetimeTZBlock",
    ]:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            Pandas4Warning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )
        if name == "DatetimeTZBlock":
            from pandas.core.internals.api import _DatetimeTZBlock as DatetimeTZBlock

            return DatetimeTZBlock
        if name == "ExtensionBlock":
            from pandas.core.internals.blocks import ExtensionBlock

            return ExtensionBlock
        else:
            from pandas.core.internals.blocks import Block

            return Block

    raise AttributeError(f"module 'pandas.core.internals' has no attribute '{name}'")
