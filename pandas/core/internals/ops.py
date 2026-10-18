# [MODULE DOC]
# Internal helpers for applying an elementwise array operation block-by-block across two block managers (used by DataFrame arithmetic and comparison when both operands are frames).
# Contents: a named tuple describing one aligned pair of blocks, a generator that yields such pairs, the blockwise operation driver, a blockwise "all" reduction, and two small private helpers.
# Depends on the block manager and block classes (only imported for type checking) and on the dtype helper that identifies 1D-only extension dtypes. No import-time side effects.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NamedTuple,
)

from pandas.core.dtypes.common import is_1d_only_ea_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas._libs.internals import BlockPlacement
    from pandas._typing import ArrayLike

    from pandas.core.internals.blocks import Block
    from pandas.core.internals.managers import BlockManager


class BlockPairInfo(NamedTuple):
    # [CLASS DOC]
    # Named tuple holding one aligned pair of block values produced by the block-pair iterator.
    # Fields: lvals and rvals are the left and right arrays (already reshaped to be compatible), locs is the placement of the left block within the left manager, left_ea and right_ea are flags telling whether each side is a 1D-only extension array, and rblk is the right-hand block that the values came from.
    lvals: ArrayLike
    rvals: ArrayLike
    locs: BlockPlacement
    left_ea: bool
    right_ea: bool
    rblk: Block


def _iter_block_pairs(
    left: BlockManager, right: BlockManager
) -> Iterator[BlockPairInfo]:
    # [FUNCTION DOC]
    # HELPER: generator behind the blockwise operations in this module.
    # Parameters: left and right are block managers whose parent frames are assumed already aligned (not checked here).
    # For each block of the left manager, slices the right manager along the items axis to the matching locations (views only), then for each resulting right block yields a pair-info tuple with same-shape values. The left_ea flag is true when the left block values are one-dimensional.
    # No side effects; assertions that would validate shapes are intentionally disabled for performance.
    # At this point we have already checked the parent DataFrames for
    #  assert rframe._indexed_same(lframe)

    for blk in left.blocks:
        locs = blk.mgr_locs
        blk_vals = blk.values

        left_ea = blk_vals.ndim == 1

        rblks = right._slice_take_blocks_ax0(locs.indexer, only_slice=True)

        # Assertions are disabled for performance, but should hold:
        # if left_ea:
        #    assert len(locs) == 1, locs
        #    assert len(rblks) == 1, rblks
        #    assert rblks[0].shape[0] == 1, rblks[0].shape

        for rblk in rblks:
            right_ea = rblk.values.ndim == 1

            lvals, rvals = _get_same_shape_values(blk, rblk, left_ea, right_ea)
            info = BlockPairInfo(lvals, rvals, locs, left_ea, right_ea, rblk)
            yield info


def operate_blockwise(
    left: BlockManager, right: BlockManager, array_op
) -> BlockManager:
    # [FUNCTION DOC]
    # Apply a binary array operation to every aligned pair of blocks of two managers and assemble the results into a new manager.
    # Parameters: left and right are block managers (already aligned); array_op is a callable taking left values and right values and returning the result values.
    # Behavior: for each pair, applies array_op; if the left side was a 1D extension array, the right was not, and the result is a reshapeable array that is not a 1D-only extension dtype, the result is reshaped to a single row. The result is split into blocks using the right block's splitting logic, their placements are reset to the original frame positions, and all are collected.
    # Returns a new manager of the same type as right, with the right manager's axes, built without integrity verification.
    # KNOWN ISSUE: a TODO notes that the axes are reused rather than shallow-copied, so the result shares axis objects with the right manager.
    # At this point we have already checked the parent DataFrames for
    #  assert rframe._indexed_same(lframe)

    res_blks: list[Block] = []
    for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
        res_values = array_op(lvals, rvals)
        if (
            left_ea
            and not right_ea
            and hasattr(res_values, "reshape")
            and not is_1d_only_ea_dtype(res_values.dtype)
        ):
            res_values = res_values.reshape(1, -1)
        nbs = rblk._split_op_result(res_values)

        # Assertions are disabled for performance, but should hold:
        # if right_ea or left_ea:
        #    assert len(nbs) == 1
        # else:
        #    assert res_values.shape == lvals.shape, (res_values.shape, lvals.shape)

        _reset_block_mgr_locs(nbs, locs)

        res_blks.extend(nbs)

    # Assertions are disabled for performance, but should hold:
    #  slocs = {y for nb in res_blks for y in nb.mgr_locs.as_array}
    #  nlocs = sum(len(nb.mgr_locs.as_array) for nb in res_blks)
    #  assert nlocs == len(left.items), (nlocs, len(left.items))
    #  assert len(slocs) == nlocs, (len(slocs), nlocs)
    #  assert slocs == set(range(nlocs)), slocs

    # TODO shallow copy axes?
    new_mgr = type(right)(tuple(res_blks), axes=right.axes, verify_integrity=False)
    return new_mgr


def _reset_block_mgr_locs(nbs: list[Block], locs) -> None:
    # [FUNCTION DOC]
    # HELPER: for operate_blockwise. Rewrites the placement of each block in the given list, in place, so that it refers to positions in the original frame.
    # Parameters: nbs is a list of result blocks whose placements are relative to the sliced left block; locs is the placement of the original left block. Each block's placement becomes the original locations selected by its relative indexer. Returns nothing; mutates the blocks.
    """
    Reset mgr_locs to correspond to our original DataFrame.
    """
    for nb in nbs:
        nblocs = locs[nb.mgr_locs.indexer]
        nb.mgr_locs = nblocs
        # Assertions are disabled for performance, but should hold:
        #  assert len(nblocs) == nb.shape[0], (len(nblocs), nb.shape)
        #  assert all(x in locs.as_array for x in nb.mgr_locs.as_array)


def _get_same_shape_values(
    lblk: Block, rblk: Block, left_ea: bool, right_ea: bool
) -> tuple[ArrayLike, ArrayLike]:
    # [FUNCTION DOC]
    # HELPER: for the block-pair iterator. Returns the left and right block values with matching shapes.
    # Parameters: lblk and rblk are the left and right blocks; left_ea and right_ea flag whether each block holds a 1D extension array.
    # The left values are sliced to the rows covered by the right block's placement (requires that placement to be slice-like, asserted). Both 2D: left sliced, shapes asserted equal. Both 1D: shapes asserted equal. Only right 1D: left sliced then squeezed to its single row. Only left 1D: right squeezed to its single row.
    # Returns a tuple of (left values, right values). Raises an assertion error if shapes mismatch or the placement is not slice-like.
    """
    Slice lblk.values to align with rblk.  Squeeze if we have EAs.
    """
    lvals = lblk.values
    rvals = rblk.values

    # Require that the indexing into lvals be slice-like
    assert rblk.mgr_locs.is_slice_like, rblk.mgr_locs

    # TODO(EA2D): with 2D EAs only this first clause would be needed
    if not (left_ea or right_ea):
        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[Union[ndarray, slice], slice]"
        lvals = lvals[rblk.mgr_locs.indexer, :]  # type: ignore[call-overload]
        assert lvals.shape == rvals.shape, (lvals.shape, rvals.shape)
    elif left_ea and right_ea:
        assert lvals.shape == rvals.shape, (lvals.shape, rvals.shape)
    elif right_ea:
        # lvals are 2D, rvals are 1D

        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[Union[ndarray, slice], slice]"
        lvals = lvals[rblk.mgr_locs.indexer, :]  # type: ignore[call-overload]
        assert lvals.shape[0] == 1, lvals.shape
        lvals = lvals[0, :]
    else:
        # lvals are 1D, rvals are 2D
        assert rvals.shape[0] == 1, rvals.shape
        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[int, slice]"
        rvals = rvals[0, :]  # type: ignore[call-overload]

    return lvals, rvals


def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool:
    # [FUNCTION DOC]
    # Blockwise "all" reduction over two aligned managers.
    # Parameters: left and right are block managers; op is a callable taking left and right values and returning a truthy or falsy result.
    # Iterates over aligned block pairs and returns False as soon as op yields a falsy result for any pair, otherwise True. No mutation.
    """
    Blockwise `all` reduction.
    """
    for info in _iter_block_pairs(left, right):
        res = op(info.lvals, info.rvals)
        if not res:
            return False
    return True
