# [MODULE DOC]
# Concatenation of block managers, the engine behind row-wise and column-wise
# concatenation of frames and series.
# Public content: the function that concatenates managers (used by the
# top-level concatenation routine). Internal content: helpers to reindex
# columns using placeholder (void dtype) blocks, a fast path for managers
# that each hold a single float block, planning functions that align blocks
# across managers, and the join-unit class that wraps one block participating in
# a concatenation, plus helpers choosing a common dtype and the matching
# missing value. Depends on the blocks and managers modules and on dtype
# casting and concatenation utilities. No import-time side effects.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    NaT,
    algos as libalgos,
    internals as libinternals,
    lib,
)
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.cast import (
    ensure_dtype_can_hold_na,
    find_common_type,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.blocks import (
    ensure_block_shape,
    new_block_2d,
)
from pandas.core.internals.managers import (
    BlockManager,
    make_na_array,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        Shape,
    )

    from pandas import Index
    from pandas.core.internals.blocks import (
        Block,
        BlockPlacement,
    )


def concatenate_managers(
    mgrs_indexers, axes: list[Index], concat_axis: AxisInt, copy: bool
) -> BlockManager:
    # [FUNCTION DOC]
    # Concatenate block managers into one manager.
    # Parameters: mgrs_indexers (list of pairs of a manager and a mapping from axis
    # number to indexer array; the caller guarantees the concat axis is not in the
    # mappings); axes (list of the result's axis indexes); concat_axis (integer axis
    # along which to join, zero meaning columns); copy (boolean).
    # Copies are made only when copy is true and the axis is zero.
    # Axis zero: reindexes all managers to the result columns (void placeholders for
    # missing columns) and joins them horizontally with the first manager's
    # horizontal-concat method.
    # Other axes: if the first manager has blocks whose first dtype is a 64- or 32-bit
    # float, there is more than one manager and all managers are single-block,
    # contiguous and of that dtype, uses the float fast path and returns a manager
    # with a single block. Otherwise reindexes columns; a single manager yields a
    # shallow copy with the axes replaced. For several managers, plans per-placement
    # join units; blocks of uniform type are joined directly (numpy join for non-
    # extension blocks, the compat concat for extension ones, reshaped to 2D for
    # 1D-only extension dtypes) and keep the block class if the dtype is unchanged;
    # others go through the generic join-unit concatenation and make a new 2D block.
    # Returns a manager from the blocks and axes.
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int
    copy : bool

    Returns
    -------
    BlockManager
    """

    needs_copy = copy and concat_axis == 0

    # Assertions disabled for performance
    # for tup in mgrs_indexers:
    #    # caller is responsible for ensuring this
    #    indexers = tup[1]
    #    assert concat_axis not in indexers

    if concat_axis == 0:
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        return mgrs[0].concat_horizontal(mgrs, axes)

    if len(mgrs_indexers) > 0 and mgrs_indexers[0][0].nblocks > 0:
        first_dtype = mgrs_indexers[0][0].blocks[0].dtype
        if first_dtype in [np.float64, np.float32]:
            # TODO: support more dtypes here.  This will be simpler once
            #  JoinUnit.is_na behavior is deprecated.
            #  (update 2024-04-13 that deprecation has been enforced)
            if (
                all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in mgrs_indexers)
                and len(mgrs_indexers) > 1
            ):
                # Fastpath!
                # Length restriction is just to avoid having to worry about 'copy'
                shape = tuple(len(x) for x in axes)
                nb = _concat_homogeneous_fastpath(mgrs_indexers, shape, first_dtype)
                return BlockManager((nb,), axes)

    mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)

    if len(mgrs) == 1:
        mgr = mgrs[0]
        out = mgr.copy(deep=False)
        out.axes = axes
        return out

    blocks = []
    values: ArrayLike

    for placement, join_units in _get_combined_plan(mgrs):
        unit = join_units[0]
        blk = unit.block

        if _is_uniform_join_units(join_units):
            vals = [ju.block.values for ju in join_units]

            if not blk.is_extension:
                # _is_uniform_join_units ensures a single dtype, so
                #  we can use np.concatenate, which is more performant
                #  than concat_compat
                # error: Argument 1 to "concatenate" has incompatible type
                # "List[Union[ndarray[Any, Any], ExtensionArray]]";
                # expected "Union[_SupportsArray[dtype[Any]],
                # _NestedSequence[_SupportsArray[dtype[Any]]]]"
                values = np.concatenate(vals, axis=1)  # type: ignore[arg-type]
            elif is_1d_only_ea_dtype(blk.dtype):
                # TODO(EA2D): special-casing not needed with 2D EAs
                values = concat_compat(vals, axis=0, ea_compat_axis=True)
                values = ensure_block_shape(values, ndim=2)
            else:
                values = concat_compat(vals, axis=1)

            values = ensure_wrapped_if_datetimelike(values)

            fastpath = blk.values.dtype == values.dtype
        else:
            values = _concatenate_join_units(join_units, copy=copy)
            fastpath = False

        if fastpath:
            b = blk.make_block_same_class(values, placement=placement)
        else:
            b = new_block_2d(values, placement=placement)

        blocks.append(b)

    return BlockManager(tuple(blocks), axes)


def _maybe_reindex_columns_na_proxy(
    axes: list[Index],
    mgrs_indexers: list[tuple[BlockManager, dict[int, np.ndarray]]],
    needs_copy: bool,
) -> list[BlockManager]:
    # [FUNCTION DOC]
    # HELPER: reindexes every manager along the axes in its indexer mapping so that
    # all managers match the result axes.
    # Parameters: axes (result axes), mgrs_indexers (list of pairs of manager and
    # indexer mapping), needs_copy (whether managers without indexers should be
    # deep-copied). Reindexing uses placeholder columns of void dtype (ignored when
    # choosing the final dtype), slice-only and allows duplicates. Returns a new list
    # of managers in the same order.
    """
    Reindex along columns so that all of the BlockManagers being concatenated
    have matching columns.

    Columns added in this reindexing have dtype=np.void, indicating they
    should be ignored when choosing a column's final dtype.
    """
    new_mgrs = []

    for mgr, indexers in mgrs_indexers:
        # For axis=0 (i.e. columns) we use_na_proxy and only_slice, so this
        #  is a cheap reindexing.
        for i, indexer in indexers.items():
            mgr = mgr.reindex_indexer(
                axes[i],
                indexer,
                axis=i,
                only_slice=True,  # only relevant for i==0
                allow_dups=True,
                use_na_proxy=True,  # only relevant for i==0
            )
        if needs_copy and not indexers:
            mgr = mgr.copy(deep=True)

        new_mgrs.append(mgr)
    return new_mgrs


def _is_homogeneous_mgr(mgr: BlockManager, first_dtype: DtypeObj) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true if the manager can be treated as one contiguous array of
    # the given dtype: exactly one block, whose locations are a slice with step one,
    # and whose dtype equals first_dtype.
    """
    Check if this Manager can be treated as a single ndarray.
    """
    if mgr.nblocks != 1:
        return False
    blk = mgr.blocks[0]
    if not (blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1):
        return False

    return blk.dtype == first_dtype


def _concat_homogeneous_fastpath(
    mgrs_indexers, shape: Shape, first_dtype: np.dtype
) -> Block:
    # [FUNCTION DOC]
    # HELPER: float fast path for concatenating managers along the rows.
    # Parameters: mgrs_indexers (list of pairs; each manager is homogeneous per the
    # check helper), shape (tuple of the result lengths), first_dtype (numpy 64- or
    # 32-bit float dtype). If no manager needs reindexing, transposes the block
    # values, joins them and transposes back. Otherwise allocates an uninitialised
    # output array and fills a slice for each manager, either with the float take
    # routine for the matching width and the column indexer, or by direct copy.
    # Returns a single new block covering all columns.
    # KNOWN ISSUE: the existing docstring ends with an unfinished sentence and a placeholder in brackets and does not describe the behavior.
    """
    With single-Block managers with homogeneous dtypes (that can already hold nan),
    we avoid [...]
    """
    # assumes
    #  all(_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in in mgrs_indexers)

    if all(not indexers for _, indexers in mgrs_indexers):
        # https://github.com/pandas-dev/pandas/pull/52685#issuecomment-1523287739
        arrs = [mgr.blocks[0].values.T for mgr, _ in mgrs_indexers]
        arr = np.concatenate(arrs).T
        bp = libinternals.BlockPlacement(slice(shape[0]))
        nb = new_block_2d(arr, bp)
        return nb

    arr = np.empty(shape, dtype=first_dtype)

    if first_dtype == np.float64:
        take_func = libalgos.take_2d_axis0_float64_float64
    else:
        take_func = libalgos.take_2d_axis0_float32_float32

    start = 0
    for mgr, indexers in mgrs_indexers:
        mgr_len = mgr.shape[1]
        end = start + mgr_len

        if 0 in indexers:
            take_func(
                mgr.blocks[0].values,
                indexers[0],
                arr[:, start:end],
            )
        else:
            # No reindexing necessary, we can copy values directly
            arr[:, start:end] = mgr.blocks[0].values

        start += mgr_len

    bp = libinternals.BlockPlacement(slice(shape[0]))
    nb = new_block_2d(arr, bp)
    return nb


def _get_combined_plan(
    mgrs: list[BlockManager],
) -> Generator[tuple[BlockPlacement, list[JoinUnit]]]:
    # [FUNCTION DOC]
    # HELPER: generator planning the concatenation. Parameter mgrs: the managers
    # (already reindexed to the same columns). Gets from the compiled helper the
    # groups of consecutive column positions that map to the same block number in
    # each manager, and yields, for each group, a pair of the placement and a list
    # with one join unit per manager built from the corresponding block piece.
    max_len = mgrs[0].shape[0]

    blknos_list = [mgr.blknos for mgr in mgrs]
    pairs = libinternals.get_concat_blkno_indexers(blknos_list)
    for blknos, bp in pairs:
        # assert bp.is_slice_like
        # assert len(bp) > 0

        units_for_bp = []
        for k, mgr in enumerate(mgrs):
            blkno = blknos[k]

            nb = _get_block_for_concat_plan(mgr, bp, blkno, max_len=max_len)
            unit = JoinUnit(nb)
            units_for_bp.append(unit)

        yield bp, units_for_bp


def _get_block_for_concat_plan(
    mgr: BlockManager, bp: BlockPlacement, blkno: int, *, max_len: int
) -> Block:
    # [FUNCTION DOC]
    # HELPER: returns the piece of the manager's block selected by a placement.
    # Parameters: mgr, bp (block placement of the columns wanted), blkno (block
    # number in the manager), max_len (keyword-only number of columns of the result).
    # If the placement covers exactly all of the block's locations and these are a
    # contiguous slice, returns the block itself. Otherwise, converts the placement
    # to positions within the block and returns a column slice of the block when the
    # positions form a slice, else takes the columns.
    blk = mgr.blocks[blkno]
    # Assertions disabled for performance:
    #  assert bp.is_slice_like
    #  assert blkno != -1
    #  assert (mgr.blknos[bp] == blkno).all()

    if len(bp) == len(blk.mgr_locs) and (
        blk.mgr_locs.is_slice_like and blk.mgr_locs.as_slice.step == 1
    ):
        nb = blk
    else:
        ax0_blk_indexer = mgr.blklocs[bp.indexer]

        slc = lib.maybe_indices_to_slice(ax0_blk_indexer, max_len)
        # TODO: in all extant test cases 2023-04-08 we have a slice here.
        #  Will this always be the case?
        if isinstance(slc, slice):
            nb = blk.slice_block_columns(slc)
        else:
            nb = blk.take_block_columns(slc)

    # assert nb.shape == (len(bp), mgr.shape[1])
    return nb


class JoinUnit:
    # [CLASS DOC]
    # HELPER: wrapper around one block taking part in a concatenation. State: the
    # block. Provides the repr, a check for being all missing of a compatible
    # type, the cached missing flag and a method returning the values to be joined.
    def __init__(self, block: Block) -> None:
        # [FUNCTION DOC]
        # Store the block on the instance. Parameter block: the block.
        self.block = block

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return a string with the class name and the representation of the block.
        return f"{type(self).__name__}({self.block!r})"

    def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true if this unit is all missing with a missing type that is
        # compatible with the given dtype. Parameter dtype: target dtype.
        # False if the unit is not all missing. True for void-dtype blocks. For object
        # blocks, true if every element is a valid missing value for the dtype. For
        # datetime-like missing values the block's dtype must equal the target dtype
        # (datetime vs timedelta are not interchangeable). The extension missing value
        # is rejected when the target needs integer-based datetime handling. Otherwise
        # tests the block's fill value against the dtype.
        # KNOWN ISSUE: the code marks the nullable-missing-value rejection as a kludge that exists only to satisfy one test.
        """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
        if not self.is_na:
            return False

        blk = self.block
        if blk.dtype.kind == "V":
            return True

        if blk.dtype == object:
            values = blk.values
            return all(is_valid_na_for_dtype(x, dtype) for x in values.ravel(order="K"))

        na_value = blk.fill_value
        if na_value is NaT and blk.dtype != dtype:
            # e.g. we are dt64 and other is td64
            # fill_values match but we should not cast blk.values to dtype
            # TODO: this will need updating if we ever have non-nano dt64/td64
            return False

        if na_value is NA and needs_i8_conversion(dtype):
            # FIXME: kludge; test_append_empty_frame_with_timedelta64ns_nat
            #  e.g. blk.dtype == "Int64" and dtype is td64, we dont want
            #  to consider these as matching
            return False

        # TODO: better to use can_hold_element?
        return is_valid_na_for_dtype(na_value, dtype)

    @cache_readonly
    def is_na(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true if the block has void dtype (a placeholder for missing
        # columns); false otherwise.
        blk = self.block
        if blk.dtype.kind == "V":
            return True
        return False

    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike:
        # [FUNCTION DOC]
        # Return the values to use in the join. Parameters: empty_dtype (the common
        # result dtype), upcasted_na (missing value to fill with, or None if no upcasting
        # is necessary).
        # If upcasted_na is None and the block is not a void block, returns the block's
        # values unchanged. Otherwise, if the unit is all-missing and valid for empty_dtype,
        # returns a freshly built missing array of that dtype and the block's shape; for an
        # object block whose first element is None, None is used as fill instead of the
        # given value. Otherwise returns the block's values unchanged.
        values: ArrayLike

        if upcasted_na is None and self.block.dtype.kind != "V":
            # No upcasting is necessary
            return self.block.values
        else:
            fill_value = upcasted_na

            if self._is_valid_na_for(empty_dtype):
                # note: always holds when self.block.dtype.kind == "V"
                blk_dtype = self.block.dtype

                if blk_dtype == np.dtype("object"):
                    # we want to avoid filling with np.nan if we are
                    # using None; we already know that we are all
                    # nulls
                    values = cast(np.ndarray, self.block.values)
                    if values.size and values[0, 0] is None:
                        fill_value = None

                return make_na_array(empty_dtype, self.block.shape, fill_value)

            return self.block.values


def _concatenate_join_units(join_units: list[JoinUnit], copy: bool) -> ArrayLike:
    # [FUNCTION DOC]
    # HELPER: generic concatenation of the units' values along columns of the 2D
    # layout. Parameters: join_units (list), copy (accepted but unused).
    # Determines the common dtype and the missing value for it, obtains each unit's
    # values, and if any are 1D-only extension arrays reduces the others to 1D (first
    # row) and joins with the compat concat reshaped to 2D, otherwise joins along
    # axis one. Returns the joined array.
    # KNOWN ISSUE: the copy parameter is never used in the body.
    """
    Concatenate values from several join units along axis=1.
    """
    empty_dtype = _get_empty_dtype(join_units)

    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)
    upcasted_na = _dtype_to_na_value(empty_dtype, has_none_blocks)

    to_concat = [
        ju.get_reindexed_values(empty_dtype=empty_dtype, upcasted_na=upcasted_na)
        for ju in join_units
    ]

    if any(is_1d_only_ea_dtype(t.dtype) for t in to_concat):
        # TODO(EA2D): special case not needed if all EAs used HybridBlocks

        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[int, slice]"
        to_concat = [
            t if is_1d_only_ea_dtype(t.dtype) else t[0, :]  # type: ignore[call-overload]
            for t in to_concat
        ]
        concat_values = concat_compat(to_concat, axis=0, ea_compat_axis=True)
        concat_values = ensure_block_shape(concat_values, 2)

    else:
        concat_values = concat_compat(to_concat, axis=1)

    return concat_values


def _dtype_to_na_value(dtype: DtypeObj, has_none_blocks: bool):
    # [FUNCTION DOC]
    # HELPER: returns the missing value used to fill in for the dtype.
    # Parameters: dtype; has_none_blocks (true if any unit is a void block). Extension
    # dtypes use their own missing value; datetime/timedelta dtypes use a typed
    # NaT of the same unit; float and complex use typed NaN; bool returns None (no
    # upcast); integers return None unless void blocks exist, in which case plain NaN;
    # object returns NaN. Any other kind raises a not-implemented error.
    """
    Find the NA value to go with this dtype.
    """
    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        return dtype.type("NaT", np.datetime_data(dtype)[0])
    elif dtype.kind in "fc":
        return dtype.type("NaN")
    elif dtype.kind == "b":
        # different from missing.na_value_for_dtype
        return None
    elif dtype.kind in "iu":
        if not has_none_blocks:
            # different from missing.na_value_for_dtype
            return None
        return np.nan
    elif dtype.kind == "O":
        return np.nan
    raise NotImplementedError


def _get_empty_dtype(join_units: Sequence[JoinUnit]) -> DtypeObj:
    # [FUNCTION DOC]
    # HELPER: picks the dtype for the joined result. Parameter join_units (sequence).
    # If all unit dtypes are equal, that dtype is returned. Otherwise computes the
    # common type over the dtypes of units that are not all missing, and widens it
    # to hold missing values if there are void blocks.
    # KNOWN ISSUE: the existing docstring says it returns a dtype and missing values and mentions a missing value that may be None, but only the dtype is returned.
    """
    Return dtype and N/A values to use when concatenating specified units.

    Returned N/A value may be None which means there was no casting involved.

    Returns
    -------
    dtype
    """
    if lib.dtypes_all_equal([ju.block.dtype for ju in join_units]):
        empty_dtype = join_units[0].block.dtype
        return empty_dtype

    has_none_blocks = any(unit.block.dtype.kind == "V" for unit in join_units)

    dtypes = [unit.block.dtype for unit in join_units if not unit.is_na]

    dtype = find_common_type(dtypes)
    if has_none_blocks:
        dtype = ensure_dtype_can_hold_na(dtype)

    return dtype


def _is_uniform_join_units(join_units: list[JoinUnit]) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true if the join units can be joined by direct concatenation
    # of the block values rather than the generic path. Parameter join_units (list).
    # False if the first block is void. Otherwise true only if all blocks are of
    # exactly the same block class, all dtypes are equal or integer, unsigned or
    # boolean kinds, and no unit is all missing unless its block is an extension block.
    """
    Check if the join units consist of blocks of uniform type that can
    be concatenated using Block.concat_same_type instead of the generic
    _concatenate_join_units (which uses `concat_compat`).

    """
    first = join_units[0].block
    if first.dtype.kind == "V":
        return False
    return (
        # exclude cases where a) ju.block is None or b) we have e.g. Int64+int64
        all(type(ju.block) is type(first) for ju in join_units)
        and
        # e.g. DatetimeLikeBlock can be dt64 or td64, but these are not uniform
        all(
            ju.block.dtype == first.dtype
            # GH#42092 we only want the dtype_equal check for non-numeric blocks
            #  (for now, may change but that would need a deprecation)
            or ju.block.dtype.kind in "iub"
            for ju in join_units
        )
        and
        # no blocks that would get missing values (can lead to type upcasts)
        # unless we're an extension dtype.
        all(not ju.is_na or ju.block.is_extension for ju in join_units)
    )
