# [MODULE DOC]
# Block managers: the internal storage layer behind DataFrame and Series. A manager owns a sequence of axes (label indexes) and a tuple of blocks, where each block holds the values of one dtype for a set of positions along the items axis.
# Public contents: BaseBlockManager (shared logic), BlockManager (two-dimensional: items are columns, second axis is the row index), SingleBlockManager (one-dimensional, backs Series), constructor helpers that build managers from blocks or from per-column arrays, an error-raising helper for shape mismatches, a function that creates missing-value arrays, plus private helpers for consolidation and block formation.
# Key concepts: the placement of each block (positions of its items in the manager), the lookup arrays mapping each item to its block number and to its position within that block, consolidation (merging blocks of equal dtype), and copy-on-write reference tracking between blocks sharing memory. Most mutating methods change the manager in place and keep these lookup arrays consistent.
# Relationships: depends on the block classes and block-pair operations module (imported here), and on a compiled base class for the two-dimensional manager. Emits a performance warning when many non-extension blocks accumulate after insertion. No import-time side effects.

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Sequence,
)
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    Self,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._config.config import get_option

from pandas._libs import (
    algos as libalgos,
    internals as libinternals,
    lib,
)
from pandas._libs.internals import (
    BlockPlacement,
    BlockValuesRefs,
)
from pandas._libs.tslibs import Timestamp
from pandas.errors import (
    AbstractMethodError,
    PerformanceWarning,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    np_can_hold_element,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    array_equals,
    isna,
)

import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
    Index,
    default_index,
    ensure_index,
)
from pandas.core.internals.blocks import (
    Block,
    NumpyBlock,
    ensure_block_shape,
    extend_blocks,
    get_block_type,
    maybe_coerce_values,
    new_block,
    new_block_2d,
)
from pandas.core.internals.ops import (
    blockwise_all,
    operate_blockwise,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        QuantileInterpolation,
        Shape,
        npt,
    )

    from pandas.api.extensions import ExtensionArray


def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    # [FUNCTION DOC]
    # Find the common dtype for a list of block dtypes. Parameter: dtypes is a list of dtype objects. Returns none when the list is empty, otherwise the common dtype determined by the dtype promotion rules.
    # KNOWN ISSUE: the docstring names the parameter blocks, but the actual parameter is named dtypes.
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    if not len(dtypes):
        return None

    return find_common_type(dtypes)


def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    # [FUNCTION DOC]
    # Convert a dtype to a numpy dtype suitable for a plain array. Sparse dtypes map to their subtype; other extension dtypes map to object; the fixed-width string dtype maps to object; numpy dtypes pass through. Returns the numpy dtype.
    # TODO: https://github.com/pandas-dev/pandas/issues/22791
    # Give EAs some input on what happens here. Sparse needs this.
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype("object")
    elif dtype == np.dtype(str):
        dtype = np.dtype("object")
    return dtype


class BaseBlockManager(PandasObject):
    # [CLASS DOC]
    # Abstract base for block managers (not public API). Subclasses are the two-dimensional BlockManager and the one-dimensional SingleBlockManager. It declares the shared attributes: the tuple of blocks, the list of axes, and the lookup arrays of block numbers and in-block positions, plus consolidation flags. Slots are empty because subclasses hold the state.
    # Provides the shared operations: applying a block-level method to every block, element-wise helpers (fill, where, mask assignment, round, replace, interpolate, shift, astype and so on), copying, subsetting, reindexing, taking, consolidation hooks and equality.
    # Invariants: the union of block placements covers every item exactly once; block shapes beyond the items axis match the other axes; copy-on-write references are tracked on blocks.
    # KNOWN ISSUE: the docstring lists a method signature set (for example reindex_axis, delete, insert and set by label) that does not exactly match the methods defined here; delete and set by label do not exist on this class.
    """
    Core internal data structure to implement DataFrame, Series, etc.

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Attributes
    ----------
    shape
    ndim
    axes
    values
    items

    Methods
    -------
    set_axis(axis, new_labels)
    copy(deep=True)

    get_dtypes

    apply(func, axes, block_filter_fn)

    get_bool_data
    get_numeric_data

    get_slice(slice_like, axis)
    get(label)
    iget(loc)

    take(indexer, axis)
    reindex_axis(new_labels, axis)
    reindex_indexer(new_labels, indexer, axis)

    delete(label)
    insert(loc, label, value)
    set(label, value)

    Parameters
    ----------
    blocks: Sequence of Block
    axes: Sequence of Index
    verify_integrity: bool, default True

    Notes
    -----
    This is *not* a public API class
    """

    __slots__ = ()

    _blknos: npt.NDArray[np.intp]
    _blklocs: npt.NDArray[np.intp]
    blocks: tuple[Block, ...]
    axes: list[Index]

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: the number of dimensions. Subclasses supply it (two for BlockManager, one for SingleBlockManager); the base raises a not-implemented error.
        raise NotImplementedError

    _known_consolidated: bool
    _is_consolidated: bool

    def __init__(self, blocks, axes, verify_integrity: bool = True) -> None:
        # [FUNCTION DOC]
        # Abstract constructor: subclasses implement construction from blocks, axes and an integrity flag; the base raises a not-implemented error.
        raise NotImplementedError

    @final
    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Number of items: the length of the first axis.
        return len(self.items)

    @property
    def shape(self) -> Shape:
        # [FUNCTION DOC]
        # The shape as a tuple of the lengths of all axes.
        return tuple(len(ax) for ax in self.axes)

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        # [FUNCTION DOC]
        # Abstract class constructor: build a manager of this class from a list of blocks and a list of axes without verifying integrity. Subclasses implement it; the base raises a not-implemented error.
        raise NotImplementedError

    @property
    def blknos(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Lookup array giving, for each item, the number of the block that holds it. Built lazily: if not yet computed, triggers a rebuild of both lookup arrays. Values are positions in the block tuple; the in-block position is given by the companion property. The arrays may be altered by other manager methods.
        """
        Suppose we want to find the array corresponding to our i'th column.

        blknos[i] identifies the block from self.blocks that contains this column.

        blklocs[i] identifies the column of interest within
        self.blocks[self.blknos[i]]
        """
        if self._blknos is None:
            # Note: these can be altered by other BlockManager methods.
            self._rebuild_blknos_and_blklocs()

        return self._blknos

    @property
    def blklocs(self) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Lookup array giving, for each item, its position within the block identified by the block number array. Built lazily together with the block number array.
        """
        See blknos.__doc__
        """
        if self._blklocs is None:
            # Note: these can be altered by other BlockManager methods.
            self._rebuild_blknos_and_blklocs()

        return self._blklocs

    def make_empty(self, axes=None) -> Self:
        # [FUNCTION DOC]
        # Return an empty manager of the same class with the items axis of length zero.
        # Parameter: axes is an optional list of axes; by default a zero-length default index followed by the remaining existing axes. For one-dimensional managers, the single block is kept with its values truncated to zero length (preserving dtype); for two-dimensional managers, there are no blocks. Returns the new manager built from blocks and axes.
        """return an empty BlockManager with the items axis of len 0"""
        if axes is None:
            # TODO shallow copy remaining axis?
            axes = [default_index(0), *self.axes[1:]]

        # preserve dtype if possible
        if self.ndim == 1:
            assert isinstance(self, SingleBlockManager)  # for mypy
            blk = self.blocks[0]
            arr = blk.values[:0]
            bp = BlockPlacement(slice(0, 0))
            nb = blk.make_block_same_class(arr, placement=bp)
            blocks = [nb]
        else:
            blocks = []
        return type(self).from_blocks(blocks, axes)

    def __bool__(self) -> bool:
        # [FUNCTION DOC]
        # A manager is always truthy, regardless of length.
        return True

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # [FUNCTION DOC]
        # Replace the axis at the given position with new labels. Parameters: axis (position) and new_labels (must already be an index object; not coerced). Validates the length through the validation helper and then assigns in place; raises a value error on length mismatch (except when setting the second axis of a manager that has no items).
        # Caller is responsible for ensuring we have an Index object.
        self._validate_set_axis(axis, new_labels)
        self.axes[axis] = new_labels

    @final
    def _validate_set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # [FUNCTION DOC]
        # HELPER: for axis replacement. Compares the new label count with the current axis length and raises a value error describing the mismatch, except that changing the row axis is allowed when there are no items.
        # Caller is responsible for ensuring we have an Index object.
        old_len = len(self.axes[axis])
        new_len = len(new_labels)

        if axis == 1 and len(self.items) == 0:
            # If we are setting the index on a DataFrame with no columns,
            #  it is OK to change the length.
            pass

        elif new_len != old_len:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, new "
                f"values have {new_len} elements"
            )

    @property
    def is_single_block(self) -> bool:
        # [FUNCTION DOC]
        # True when there is exactly one block. Assumes a two-dimensional manager; the one-dimensional subclass overrides it with a constant.
        # Assumes we are 2D; overridden by SingleBlockManager
        return len(self.blocks) == 1

    @property
    def items(self) -> Index:
        # [FUNCTION DOC]
        # The first axis (the items axis: columns for a two-dimensional manager).
        return self.axes[0]

    def _has_no_reference(self, i: int) -> bool:
        # [FUNCTION DOC]
        # HELPER: for copy-on-write. Whether the item at position i has no references, meaning its block neither references another array nor is referenced by one. Looks up the block number and delegates to the block-level check. Returns a boolean.
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        blkno = self.blknos[i]
        return self._has_no_reference_block(blkno)

    def _has_no_reference_block(self, blkno: int) -> bool:
        # [FUNCTION DOC]
        # HELPER: whether the block at the given block number has no references (neither a view of nor viewed by another array). Returns a boolean.
        """
        Check for block `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the block has no references.
        """
        return not self.blocks[blkno].refs.has_reference()

    def add_references(self, mgr: BaseBlockManager) -> None:
        # [FUNCTION DOC]
        # Copy-on-write support: adopt the reference trackers of another manager. Parameter: mgr, a manager assumed to have the same block structure. If the number of blocks differs (meaning a copy was made) nothing happens; otherwise each block here takes the reference object of the block at the same position there and registers itself with it. Mutates blocks in place; returns nothing.
        """
        Adds the references from one manager to another. We assume that both
        managers have the same block structure.
        """
        if len(self.blocks) != len(mgr.blocks):
            # If block structure changes, then we made a copy
            return
        for i, blk in enumerate(self.blocks):
            blk.refs = mgr.blocks[i].refs
            blk.refs.add_reference(blk)

    def references_same_values(self, mgr: BaseBlockManager, blkno: int) -> bool:
        # [FUNCTION DOC]
        # Whether the block at the given block number in this manager shares underlying values with the block at the same number in another manager. Parameters: mgr (other manager) and blkno. Returns True if this block is among the blocks referenced by the other block's tracker.
        """
        Checks if two blocks from two different block managers reference the
        same underlying values.
        """
        blk = self.blocks[blkno]
        return any(blk is ref() for ref in mgr.blocks[blkno].refs.referenced_blocks)

    def get_unique_dtypes(self) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Return an array of the unique dtypes among the blocks (object array, order of first appearance).
        return algos.unique(np.array([blk.dtype for blk in self.blocks], dtype=object))

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Return an object array with one dtype per item, ordered by item position (the block dtypes expanded with the block-number lookup).
        dtypes = np.array([blk.dtype for blk in self.blocks], dtype=object)
        return dtypes.take(self.blknos)

    @property
    def arrays(self) -> list[ArrayLike]:
        # [FUNCTION DOC]
        # Property giving the list of the raw value arrays of the blocks, for testing and for external consumers. Warns in its documentation that it does not honor copy-on-write and should be used read-only. Returns a list with one array per block (not per item).
        """
        Quick access to the backing arrays of the Blocks.

        Only for compatibility with ArrayManager for testing convenience.
        Not to be used in actual code, and return value is not the same as the
        ArrayManager method (list of 1D arrays vs iterator of 2D ndarrays / 1D EAs).

        Warning! The returned arrays don't handle Copy-on-Write, so this should
        be used with caution (only in read-mode).
        """
        # TODO: Deprecate, usage in Dask
        # https://github.com/dask/dask/blob/484fc3f1136827308db133cd256ba74df7a38d8c/dask/base.py#L1312
        return [blk.values for blk in self.blocks]

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Text representation: the class name, one line for the items axis, one line for each further axis, then one line per block.
        output = type(self).__name__
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += f"\nItems: {ax}"
            else:
                output += f"\nAxis {i}: {ax}"

        for block in self.blocks:
            output += f"\n{block}"
        return output

    def _equal_values(self, other: Self) -> bool:
        # [FUNCTION DOC]
        # Abstract hook: subclasses compare the values of two managers whose shapes and axes were already checked and return a boolean. The base raises an abstract-method error.
        """
        To be implemented by the subclasses. Only check the column values
        assuming shape and indexes have already been checked.
        """
        raise AbstractMethodError(self)

    @final
    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality used by the frame equality method (final). Returns False when other is not an instance of this manager's class, when the number of axes differs, or when any pair of axes is not equal; otherwise defers to the subclass value comparison. Returns a boolean.
        """
        Implementation for DataFrame.equals
        """
        if not isinstance(other, type(self)):
            return False

        self_axes, other_axes = self.axes, other.axes
        if len(self_axes) != len(other_axes):
            return False
        if not all(
            ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes, strict=True)
        ):
            return False

        return self._equal_values(other)

    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Apply an operation to every block and build a new manager from the results.
        # Parameters: f is either the name of a block method or a callable (applied with the block-level apply helper); align_keys is an optional list of keyword names whose values (Series, DataFrame or ndarray) are sliced to each block's item positions before each call; extra keywords are forwarded to the block method. A keyword named filter is not allowed (assertion).
        # Results of each block may be one or several blocks and are flattened. The new manager has the same class and views of the axes. Returns the new manager.
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
        assert "filter" not in kwargs

        align_keys = align_keys or []
        result_blocks: list[Block] = []
        # fillna: Series/DataFrame is responsible for making sure value is aligned

        aligned_args = {k: kwargs[k] for k in align_keys}

        for b in self.blocks:
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[b.mgr_locs.indexer]._values
                        else:
                            kwargs[k] = obj.iloc[:, b.mgr_locs.indexer]._values
                    else:
                        # otherwise we have an ndarray
                        kwargs[k] = obj[b.mgr_locs.indexer]

            if callable(f):
                applied = b.apply(f, **kwargs)
            else:
                applied = getattr(b, f)(**kwargs)
            result_blocks = extend_blocks(applied, result_blocks)

        out = type(self).from_blocks(result_blocks, [ax.view() for ax in self.axes])
        return out

    @final
    def isna(self, func) -> Self:
        # [FUNCTION DOC]
        # Apply a function blockwise through the block-level apply method (the argument func is passed on). Despite its name it does not check for missing values itself; the callable decides. Returns a new manager.
        return self.apply("apply", func=func)

    @final
    def fillna(self, value, limit: int | None, inplace: bool) -> Self:
        # [FUNCTION DOC]
        # Fill missing values blockwise. Parameters: value (fill value), limit (maximum number of fills or none; validated up front even if no block needs filling), inplace flag. Returns the new manager from the block-level fill method.
        if limit is not None:
            # Do this validation even if we go through one of the no-op paths
            limit = libalgos.validate_limit(None, limit=limit)

        return self.apply(
            "fillna",
            value=value,
            limit=limit,
            inplace=inplace,
        )

    @final
    def where(self, other, cond, align: bool) -> Self:
        # [FUNCTION DOC]
        # Conditional replacement blockwise. Parameters: other (replacement values), cond (boolean condition), align (whether other should be sliced per block together with cond; if false, other is unwrapped to a raw array first and only cond is sliced). Returns a new manager.
        if align:
            align_keys = ["other", "cond"]
        else:
            align_keys = ["cond"]
            other = extract_array(other, extract_numpy=True)

        return self.apply(
            "where",
            align_keys=align_keys,
            other=other,
            cond=cond,
        )

    @final
    def putmask(self, mask, new, align: bool = True) -> Self:
        # [FUNCTION DOC]
        # Mask assignment blockwise. Parameters: mask (boolean positions to overwrite), new (new values), align (default True: slice both mask and new per block; if false, new is unwrapped to a raw array and only mask is sliced). Returns a new manager.
        if align:
            align_keys = ["new", "mask"]
        else:
            align_keys = ["mask"]
            new = extract_array(new, extract_numpy=True)

        return self.apply(
            "putmask",
            align_keys=align_keys,
            mask=mask,
            new=new,
        )

    @final
    def round(self, decimals: int) -> Self:
        # [FUNCTION DOC]
        # Round values blockwise to the given number of decimals. Returns a new manager.
        return self.apply("round", decimals=decimals)

    @final
    def replace(self, to_replace, value, inplace: bool) -> Self:
        # [FUNCTION DOC]
        # Replace a scalar with another scalar blockwise. Parameters: to_replace and value (both asserted non-list-like; the caller is responsible), inplace (validated as a boolean). Returns a new manager.
        inplace = validate_bool_kwarg(inplace, "inplace")
        # NDFrame.replace ensures the not-is_list_likes here
        assert not lib.is_list_like(to_replace)
        assert not lib.is_list_like(value)
        return self.apply(
            "replace",
            to_replace=to_replace,
            value=value,
            inplace=inplace,
        )

    @final
    def replace_regex(self, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Regex replacement blockwise: forwards all keywords to the block-level regex replacement and returns a new manager.
        return self.apply("_replace_regex", **kwargs)

    @final
    def replace_list(
        self,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Replace many values at once blockwise. Parameters: src_list (values to replace), dest_list (replacements, same length), inplace (default False, validated as boolean), regex (default False). After applying, consolidates the new manager in place because replacement can fragment blocks. Returns the new manager.
        """do a list replace"""
        inplace = validate_bool_kwarg(inplace, "inplace")

        bm = self.apply(
            "replace_list",
            src_list=src_list,
            dest_list=dest_list,
            inplace=inplace,
            regex=regex,
        )
        bm._consolidate_inplace()
        return bm

    def interpolate(self, inplace: bool, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Interpolate blockwise. Parameters: inplace plus keywords forwarded to the block interpolation. Returns a new manager.
        return self.apply("interpolate", inplace=inplace, **kwargs)

    def pad_or_backfill(self, inplace: bool, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Forward or backward fill blockwise: inplace flag plus keywords forwarded to the block method. Returns a new manager.
        return self.apply("pad_or_backfill", inplace=inplace, **kwargs)

    def shift(self, periods: int, fill_value) -> Self:
        # [FUNCTION DOC]
        # Shift values along the rows blockwise. Parameters: periods, fill_value (the no-default sentinel is converted to none). Returns a new manager.
        if fill_value is lib.no_default:
            fill_value = None

        return self.apply("shift", periods=periods, fill_value=fill_value)

    def setitem(self, indexer, value) -> Self:
        # [FUNCTION DOC]
        # Assign values at an indexer, returning the resulting manager (possibly a new one). For a Series this backs item assignment.
        # Parameters: indexer (positional indexer; an array with more dimensions than the manager raises a value error) and value. Copy-on-write: if the single block has references, then for a two-dimensional manager with a tuple indexer only the affected columns are split off into a new block (copy of just those columns), the indexer is rewritten against that block (handling duplicate or reordered columns and two-dimensional row indexers), and the value is set in place on the first block; an empty column selection returns a shallow copy and an empty row selection returns self unchanged. Otherwise the manager is deeply copied first. Without references, the block-level setitem is applied via apply. Returns the manager.
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")

        if not self._has_no_reference(0):
            # this method is only called if there is a single block -> hardcoded 0
            # Split blocks to only copy the columns we want to modify
            if self.ndim == 2 and isinstance(indexer, tuple):
                blk_loc = self.blklocs[indexer[1]]
                if is_list_like(blk_loc) and blk_loc.ndim == 2:
                    blk_loc = np.squeeze(blk_loc, axis=0)
                elif not is_list_like(blk_loc):
                    # Keep dimension and copy data later
                    blk_loc = [blk_loc]  # type: ignore[assignment]
                if len(blk_loc) == 0:
                    return self.copy(deep=False)

                values = self.blocks[0].values
                if values.ndim == 2:
                    # Block.delete in _iset_split_block requires sorted unique
                    # locs; inverse maps the requested column order onto the
                    # new block (GH#65446)
                    blk_loc, inverse = np.unique(blk_loc, return_inverse=True)
                    values = values[blk_loc]
                    # "T" has no attribute "_iset_split_block"
                    self._iset_split_block(  # type: ignore[attr-defined]
                        0, blk_loc, values
                    )

                    indexer = list(indexer)
                    # first block equals values we are setting to -> set to all columns
                    if lib.is_integer(indexer[1]):
                        col_indexer = 0
                    elif len(inverse) > 1 and lib.is_range_indexer(
                        inverse, len(blk_loc)
                    ):
                        col_indexer = slice(None)  # type: ignore[assignment]
                    else:
                        col_indexer = inverse  # type: ignore[assignment]
                    indexer[1] = col_indexer

                    row_indexer = indexer[0]
                    if isinstance(col_indexer, np.ndarray):
                        if (
                            isinstance(row_indexer, np.ndarray)
                            and row_indexer.ndim == 1
                        ):
                            # GH#65446: Make the row indexer 2d to take a cross product
                            row_indexer = row_indexer[:, None]
                    elif isinstance(row_indexer, np.ndarray) and row_indexer.ndim == 2:
                        # numpy cannot handle a 2d indexer in combo with a slice
                        row_indexer = np.squeeze(row_indexer, axis=1)
                    if isinstance(row_indexer, np.ndarray) and len(row_indexer) == 0:
                        # numpy does not like empty indexer combined with slice
                        # and we are setting nothing anyway
                        return self
                    indexer[0] = row_indexer
                    self.blocks[0].setitem(tuple(indexer), value)
                    return self
            # No need to split if we either set all columns or on a single block
            # manager
            self = self.copy(deep=True)

        return self.apply("setitem", indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        # [FUNCTION DOC]
        # Difference between rows n periods apart, blockwise. Only reached for two-dimensional data. Returns a new manager.
        # only reached with self.ndim == 2
        return self.apply("diff", n=n)

    def astype(self, dtype, errors: str = "raise") -> Self:
        # [FUNCTION DOC]
        # Cast blockwise. Parameters: dtype and errors (default raise; forwarded). Returns a new manager.
        return self.apply("astype", dtype=dtype, errors=errors)

    def convert(self) -> Self:
        # [FUNCTION DOC]
        # Run dtype inference conversion blockwise. Returns a new manager.
        return self.apply("convert")

    def convert_dtypes(self, **kwargs):
        # [FUNCTION DOC]
        # Convert to nullable and best-possible dtypes blockwise; keywords are forwarded to the block method. Returns a new manager.
        return self.apply("convert_dtypes", **kwargs)

    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Self:
        # [FUNCTION DOC]
        # Convert values to native objects for display or CSV output, blockwise (keyword only parameters).
        # Parameters: float_format, date_format, decimal (formatting options), na_rep (default the string nan), quoting (optional). Returns a new manager.
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        return self.apply(
            "get_values_for_csv",
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )

    @property
    def any_extension_types(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if any block is an extension block.
        """Whether any of the blocks in this manager are extension blocks"""
        return any(block.is_extension for block in self.blocks)

    @property
    def is_view(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when there is exactly one block and that block is a view; False otherwise (even if some blocks of a multi-block manager are views).
        """return a boolean if we are a single block and are a view"""
        if len(self.blocks) == 1:
            return self.blocks[0].is_view

        # It is technically possible to figure out which blocks are views
        # e.g. [ b.values.base is not None for b in self.blocks ]
        # but then we have the case of possibly some blocks being a view
        # and some blocks not. setting in theory is possible on the non-view
        # blocks. But this is a bit
        # complicated

        return False

    def _get_data_subset(self, predicate: Callable) -> Self:
        # [FUNCTION DOC]
        # HELPER: new manager containing only the blocks whose values satisfy the predicate. Parameter: predicate, a callable taking a block's values and returning a boolean. Returns the combined manager.
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        return self._combine(blocks)

    def _get_data_subset_indices(self, predicate: Callable) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: sorted array of the item positions of the blocks whose values satisfy the predicate. Parameter: predicate as above. Returns an integer array. Note it raises an error from concatenation if no block matches.
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        return indexer

    def get_bool_data(self) -> Self:
        # [FUNCTION DOC]
        # Select boolean data: keeps boolean-dtype blocks, and, for object-dtype blocks, splits them into columns and keeps those that are all boolean. Returns a manager built from the selected blocks.
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.
        """

        new_blocks = []

        for blk in self.blocks:
            if blk.dtype == bool:
                new_blocks.append(blk)

            elif blk.is_object:
                new_blocks.extend(nb for nb in blk._split() if nb.is_bool)

        return self._combine(new_blocks)

    def get_numeric_data(self) -> Self:
        # [FUNCTION DOC]
        # Select numeric blocks. If every block is numeric, returns self unchanged (avoiding a combine); otherwise returns a manager of the numeric blocks.
        numeric_blocks = [blk for blk in self.blocks if blk.is_numeric]
        if len(numeric_blocks) == len(self.blocks):
            # Avoid somewhat expensive _combine
            # TODO(CoW) need to return a shallow copy here?
            return self
        return self._combine(numeric_blocks)

    def _combine(self, blocks: list[Block], index: Index | None = None) -> Self:
        # [FUNCTION DOC]
        # HELPER: build a new manager from a subset of this manager's blocks.
        # Parameters: blocks (list of blocks of this manager); index (optional replacement for the last axis). With no blocks, returns an empty manager (keeping the dtype of the items axis; for two-dimensional data using the given index if provided). Otherwise the items of the selected blocks are sorted by position and renumbered contiguously; each block is shallow copied with its placement remapped; the items axis is taken at those positions. Returns the new manager.
        """return a new manager with the blocks"""
        if len(blocks) == 0:
            if self.ndim == 2:
                # retain our own Index dtype
                if index is not None:
                    axes = [self.items[:0], index]
                else:
                    axes = [self.items[:0], *self.axes[1:]]
                return self.make_empty(axes)
            return self.make_empty()

        # FIXME: optimization potential
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])

        new_blocks: list[Block] = []
        for b in blocks:
            nb = b.copy(deep=False)
            nb.mgr_locs = BlockPlacement(inv_indexer[nb.mgr_locs.indexer])
            new_blocks.append(nb)

        axes = list(self.axes)
        # TODO shallow copy of axes?
        if index is not None:
            axes[-1] = index
        axes[0] = self.items.take(indexer)

        return type(self).from_blocks(new_blocks, axes)

    @property
    def nblocks(self) -> int:
        # [FUNCTION DOC]
        # Number of blocks.
        return len(self.blocks)

    def copy(self, *, deep: bool) -> Self:
        # [FUNCTION DOC]
        # Copy the manager (keyword-only deep flag). Axes are replaced by views; block data is copied (deep) or shared (shallow) through block-level copy. For two-dimensional managers the lookup arrays are copied if already built. A deep copy also consolidates the result in place. Returns the new manager.
        # KNOWN ISSUE: the docstring says deep may be a string or none and has a default of True, but it is a required keyword-only boolean.
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : bool, string or None, default True
            If False, return a shallow copy (do not copy data)

        Returns
        -------
        BlockManager
        """
        # TODO: Should deep=True be respected for axes?
        new_axes = [ax.view() for ax in self.axes]

        res = self.apply("copy", deep=deep)
        res.axes = new_axes

        if self.ndim > 1:
            # Avoid needing to re-compute these
            blknos = self._blknos
            if blknos is not None:
                res._blknos = blknos.copy()
                res._blklocs = self._blklocs.copy()

        if deep:
            res._consolidate_inplace()
        return res

    def is_consolidated(self) -> bool:
        # [FUNCTION DOC]
        # Whether the manager is consolidated. The base implementation always returns True.
        return True

    def consolidate(self) -> Self:
        # [FUNCTION DOC]
        # Return a manager in which blocks of equal dtype are merged. If already consolidated returns self; otherwise builds a new manager of the same class over the same blocks and axes without verification, marks it not consolidated and consolidates it in place.
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        if self.is_consolidated():
            return self

        # TODO shallow copy is not needed here?
        bm = type(self)(self.blocks, self.axes, verify_integrity=False)
        bm._is_consolidated = False
        bm._consolidate_inplace()
        return bm

    def _consolidate_inplace(self) -> None:
        # [FUNCTION DOC]
        # Hook to consolidate in place. The base implementation does nothing; the two-dimensional manager overrides it.
        return

    @final
    def reindex_axis(
        self,
        new_index: Index,
        axis: AxisInt,
        fill_value=None,
        only_slice: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Conform the manager to a new label set along an axis (final). Parameters: new_index (labels to conform to), axis, fill_value (value for new positions), only_slice (take views, not copies, along the items axis). Computes the indexer by reindexing the current axis, then delegates to the indexer-based version. Returns a new manager.
        """
        Conform data manager to new index.
        """
        new_index, indexer = self.axes[axis].reindex(new_index)

        return self.reindex_indexer(
            new_index,
            indexer,
            axis=axis,
            fill_value=fill_value,
            only_slice=only_slice,
        )

    def reindex_indexer(
        self,
        new_axis: Index,
        indexer: npt.NDArray[np.intp] | None,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
    ) -> Self:
        # [FUNCTION DOC]
        # Reindex using a precomputed positional indexer; -1 entries mean missing.
        # Parameters: new_axis (labels), indexer (integer array or none), axis, fill_value (default none), allow_dups (default False; otherwise the existing axis checks that reindexing is allowed), only_slice (views not copies along columns), use_na_proxy (keyword only; use a void-typed placeholder array for new columns).
        # With no indexer: returns self if the axis is identical, else a shallow copy with the axis replaced. Otherwise raises an index error if the axis is out of range. For axis zero, the blocks are sliced or taken along the items axis (new items get missing blocks). For other axes, every block is reindexed along that axis, with a void-typed placeholder block reproduced as a void block of the new length. The other axes are replaced with views. When reindexing the second axis the lookup arrays are copied from this manager. Returns a new manager.
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        only_slice : bool, default False
            Whether to take views, not copies, along columns.
        use_na_proxy : bool, default False
            Whether to use an np.void ndarray for newly introduced columns.

        pandas-indexer with -1's only.
        """
        if indexer is None:
            if new_axis is self.axes[axis]:
                # TODO(CoW) need to handle CoW?
                return self

            result = self.copy(deep=False)
            result.axes = list(self.axes)
            result.axes[axis] = new_axis
            return result

        # Should be intp, but in some cases we get int64 on 32bit builds
        assert isinstance(indexer, np.ndarray)

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self.axes[axis]._validate_can_reindex(indexer)

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 0:
            new_blocks = list(
                self._slice_take_blocks_ax0(
                    indexer,
                    fill_value=fill_value,
                    only_slice=only_slice,
                    use_na_proxy=use_na_proxy,
                )
            )
        else:
            new_blocks = []
            for blk in self.blocks:
                if blk.dtype == np.void:
                    # GH#58316: np.void placeholders cast to b'' when
                    # reindexed; preserve np.void so _setitem_single_column
                    # can later infer the correct dtype
                    vals = np.empty((blk.values.shape[0], len(indexer)), dtype=np.void)
                    new_blocks.append(NumpyBlock(vals, blk.mgr_locs, ndim=2))
                else:
                    new_blocks.append(
                        blk.take_nd(
                            indexer,
                            axis=1,
                            fill_value=(
                                fill_value if fill_value is not None else blk.fill_value
                            ),
                        )
                    )

        new_axes = list(self.axes)
        new_axes[axis] = new_axis
        if self.ndim == 2:
            new_axes[1 - axis] = self.axes[1 - axis].view()

        new_mgr = type(self).from_blocks(new_blocks, new_axes)
        if axis == 1:
            # We can avoid the need to rebuild these
            new_mgr._blknos = self.blknos.copy()
            new_mgr._blklocs = self.blklocs.copy()
        return new_mgr

    def _slice_take_blocks_ax0(
        self,
        slice_or_indexer: slice | np.ndarray,
        fill_value=lib.no_default,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
        ref_inplace_op: bool = False,
    ) -> Generator[Block]:
        # [FUNCTION DOC]
        # Generator yielding the blocks resulting from slicing or taking along the items axis (overridden-like behavior for single-block managers is handled inline).
        # Parameters: slice_or_indexer (a slice, or an integer array in which -1 entries mean fill when filling is enabled); fill_value (default the no-default sentinel meaning no filling is allowed); only_slice (always yield views, never copies; used by blockwise operations); use_na_proxy (keyword only; use void placeholder for new columns); ref_inplace_op (keyword only; skip reference tracking because the operation is in place).
        # Single-block fast paths: a slice yields one sliced block (nothing when empty); a non-filling or one-dimensional take yields either per-item views (if only_slice) or one taken block. General path: looks up the block numbers and in-block positions for the requested items, groups them into runs, and for each run yields missing-value blocks (for -1, one per column when the dtype is one-dimensional-only extension and several items) or slices/takes of the existing blocks (a block that cannot be consolidated and has no extra dimension is copied per item).
        """
        Slice/take blocks along axis=0.

        Overloaded for SingleBlock

        Parameters
        ----------
        slice_or_indexer : slice or np.ndarray[int64]
        fill_value : scalar, default lib.no_default
        only_slice : bool, default False
            If True, we always return views on existing arrays, never copies.
            This is used when called from ops.blockwise.operate_blockwise.
        use_na_proxy : bool, default False
            Whether to use an np.void ndarray for newly introduced columns.
        ref_inplace_op: bool, default False
            Don't track refs if True because we operate inplace

        Yields
        ------
        Block : New Block
        """
        allow_fill = fill_value is not lib.no_default

        sl_type, slobj, sllen = _preprocess_slice_or_indexer(
            slice_or_indexer, self.shape[0], allow_fill=allow_fill
        )

        if self.is_single_block:
            blk = self.blocks[0]

            if sl_type == "slice":
                # GH#32959 EABlock would fail since we can't make 0-width
                # TODO(EA2D): special casing unnecessary with 2D EAs
                if sllen == 0:
                    return
                bp = BlockPlacement(slice(0, sllen))
                yield blk.getitem_block_columns(slobj, new_mgr_locs=bp)
                return
            elif not allow_fill or self.ndim == 1:
                if allow_fill and fill_value is None:
                    fill_value = blk.fill_value

                if not allow_fill and only_slice:
                    # GH#33597 slice instead of take, so we get
                    #  views instead of copies
                    for i, ml in enumerate(slobj):
                        yield blk.getitem_block_columns(
                            slice(ml, ml + 1),
                            new_mgr_locs=BlockPlacement(i),
                            ref_inplace_op=ref_inplace_op,
                        )
                else:
                    bp = BlockPlacement(slice(0, sllen))
                    yield blk.take_nd(
                        slobj,
                        axis=0,
                        new_mgr_locs=bp,
                        fill_value=fill_value,
                    )
                return

        if sl_type == "slice":
            blknos = self.blknos[slobj]
            blklocs = self.blklocs[slobj]
        else:
            blknos = algos.take_nd(
                self.blknos, slobj, fill_value=-1, allow_fill=allow_fill
            )
            blklocs = algos.take_nd(
                self.blklocs, slobj, fill_value=-1, allow_fill=allow_fill
            )

        # When filling blknos, make sure blknos is updated before appending to
        # blocks list, that way new blkno is exactly len(blocks).
        group = not only_slice
        for blkno, mgr_locs in libinternals.get_blkno_placements(blknos, group=group):
            if blkno == -1:
                # If we've got here, fill_value was not lib.no_default

                dtype, _ = infer_dtype_from_scalar(fill_value)
                if is_1d_only_ea_dtype(dtype) and len(mgr_locs) > 1:
                    # Handle 1D-only extension dtypes by creating separate blocks
                    # (GH#63993)
                    placements = [BlockPlacement(col_idx) for col_idx in mgr_locs]
                else:
                    placements = [mgr_locs]

                for placement in placements:
                    yield self._make_na_block(
                        placement=placement,
                        fill_value=fill_value,
                        use_na_proxy=use_na_proxy,
                    )
            else:
                blk = self.blocks[blkno]

                # Otherwise, slicing along items axis is necessary.
                if not blk._can_consolidate and not blk._validate_ndim:
                    # i.e. we dont go through here for DatetimeTZBlock
                    # A non-consolidatable block, it's easy, because there's
                    # only one item and each mgr loc is a copy of that single
                    # item.
                    deep = False
                    for mgr_loc in mgr_locs:
                        newblk = blk.copy(deep=deep)
                        newblk.mgr_locs = BlockPlacement(slice(mgr_loc, mgr_loc + 1))
                        yield newblk

                else:
                    # GH#32779 to avoid the performance penalty of copying,
                    #  we may try to only slice
                    taker = blklocs[mgr_locs.indexer]
                    max_len = max(len(mgr_locs), taker.max() + 1)
                    taker = lib.maybe_indices_to_slice(taker, max_len)

                    if isinstance(taker, slice):
                        nb = blk.getitem_block_columns(taker, new_mgr_locs=mgr_locs)
                        yield nb
                    elif only_slice:
                        # GH#33597 slice instead of take, so we get
                        #  views instead of copies
                        for i, ml in zip(taker, mgr_locs, strict=True):
                            slc = slice(i, i + 1)
                            bp = BlockPlacement(ml)
                            nb = blk.getitem_block_columns(slc, new_mgr_locs=bp)
                            # We have np.shares_memory(nb.values, blk.values)
                            yield nb
                    else:
                        nb = blk.take_nd(taker, axis=0, new_mgr_locs=mgr_locs)
                        yield nb

    def _make_na_block(
        self, placement: BlockPlacement, fill_value=None, use_na_proxy: bool = False
    ) -> Block:
        # [FUNCTION DOC]
        # HELPER: build a block of missing values for the given placement. Only used for two-dimensional managers. Parameters: placement; fill_value (default none); use_na_proxy (void placeholder; requires fill_value to be none, asserted). When the fill is none or NaN, NaN is used, typed like the common float dtype of the blocks if that is floating, to avoid needless upcasting. Infers the dtype from the fill scalar and creates the array through the missing-array function. Returns the new block with the given placement.
        # Note: we only get here with self.ndim == 2

        if use_na_proxy:
            assert fill_value is None
            shape = (len(placement), self.shape[1])
            vals = np.empty(shape, dtype=np.void)
            nb = NumpyBlock(vals, placement, ndim=2)
            return nb

        if fill_value is None or fill_value is np.nan:
            fill_value = np.nan
            # GH45857 avoid unnecessary upcasting
            dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
            if dtype is not None and np.issubdtype(dtype.type, np.floating):
                fill_value = dtype.type(fill_value)

        shape = (len(placement), self.shape[1])

        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        block_values = make_na_array(dtype, shape, fill_value)
        return new_block_2d(block_values, placement=placement)

    def take(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt = 1,
        verify: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Take items along any axis by position. Parameters: indexer (integer array), axis (default 1, the row axis), verify (default True; check that all entries lie within bounds, may be skipped if the caller already validated). Negative entries are converted by the index conversion helper. The labels are taken on that axis and the work is delegated to the reindex routine allowing duplicates. Returns a new manager.
        """
        Take items along any axis.

        indexer : np.ndarray[np.intp]
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
        # Caller is responsible for ensuring indexer annotation is accurate

        n = self.shape[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)

        new_labels = self.axes[axis].take(indexer)
        return self.reindex_indexer(
            new_axis=new_labels,
            indexer=indexer,
            axis=axis,
            allow_dups=True,
        )


class BlockManager(libinternals.BlockManager, BaseBlockManager):
    # [CLASS DOC]
    # Two-dimensional block manager. Inherits the shared logic from the base manager and the compiled base class that stores the blocks, axes and lookup arrays. Class attribute ndim is two. The first axis holds the columns (items), the second the rows. Blocks are two-dimensional with items along the first dimension.
    # Adds constructors, positional indexing (single row or column), setting and inserting columns, deleting columns, reductions, quantiles, unstacking, dictionary-of-dtype iteration, conversion to a single array, consolidation and horizontal concatenation.
    """
    BaseBlockManager that holds 2D blocks.
    """

    ndim = 2

    # ----------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        blocks: Sequence[Block],
        axes: Sequence[Index],
        verify_integrity: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Construct a manager from blocks and axes. Parameters: blocks and axes (sequences); verify_integrity (default True). Storage is handled by the compiled base class; this initializer only validates when asked. With verification: every block must have the same number of dimensions as the manager (assertion error otherwise) and the integrity check runs.
        if verify_integrity:
            # Assertion disabled for performance
            # assert all(isinstance(x, Index) for x in axes)

            for block in blocks:
                if self.ndim != block.ndim:
                    raise AssertionError(
                        f"Number of Block dimensions ({block.ndim}) must equal "
                        f"number of axes ({self.ndim})"
                    )
                # As of 2.0, the caller is responsible for ensuring that
                #  DatetimeTZBlock with block.ndim == 2 has block.values.ndim ==2;
                #  previously there was a special check for fastparquet compat.

            self._verify_integrity()

    def _verify_integrity(self) -> None:
        # [FUNCTION DOC]
        # HELPER: validate structure. Raises a construction error if any block's trailing shape differs from the manager's trailing shape, and an assertion error if the number of manager items differs from the total of block placements.
        mgr_shape = self.shape
        tot_items = sum(len(x.mgr_locs) for x in self.blocks)
        for block in self.blocks:
            if block.shape[1:] != mgr_shape[1:]:
                raise_construction_error(tot_items, block.shape[1:], self.axes)
        if len(self.items) != tot_items:
            raise AssertionError(
                "Number of manager items must equal union of "
                f"block items\n# manager items: {len(self.items)}, # "
                f"tot_items: {tot_items}"
            )

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        # [FUNCTION DOC]
        # Class constructor with the same signature as the one-dimensional manager's: builds a manager from blocks and axes without verification.
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        return cls(blocks, axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Indexing

    def fast_xs(self, loc: int) -> SingleBlockManager:
        # [FUNCTION DOC]
        # Return the row at the given position as a one-dimensional manager (cross-section).
        # Parameter: loc, the integer row position. With a single block, takes the row from that block and shares references with it (a view). Otherwise finds the common dtype of the blocks; for extension dtypes an object array is filled first and converted to the extension array at the end; for other dtypes a plain (datetime-wrapped) array is filled item by item, with a special case that upcasts to float or object when a categorical column yields NaN into an integer or boolean result. Returns a single-block manager over the items axis.
        # KNOWN ISSUE: the docstring says it returns an array, but the code returns a one-dimensional block manager.
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if len(self.blocks) == 1:
            # TODO: this could be wrong if blk.mgr_locs is not slice(None)-like;
            #  is this ruled out in the general case?
            result: np.ndarray | ExtensionArray = self.blocks[0].iget(
                (slice(None), loc)
            )
            # in the case of a single block, the new block is a view
            bp = BlockPlacement(slice(0, len(result)))
            block = new_block(
                result,
                placement=bp,
                ndim=1,
                refs=self.blocks[0].refs,
            )
            return SingleBlockManager(block, self.axes[0].view())

        dtype = interleaved_dtype([blk.dtype for blk in self.blocks])

        n = len(self)

        if isinstance(dtype, ExtensionDtype):
            # TODO: use object dtype as workaround for non-performant
            #  EA.__setitem__ methods. (primarily ArrowExtensionArray.__setitem__
            #  when iteratively setting individual values)
            #  https://github.com/pandas-dev/pandas/pull/54508#issuecomment-1675827918
            result = np.empty(n, dtype=object)
        else:
            result = np.empty(n, dtype=dtype)
            result = ensure_wrapped_if_datetimelike(result)

        for blk in self.blocks:
            # Such assignment may incorrectly coerce NaT to None
            # result[blk.mgr_locs] = blk._slice((slice(None), loc))
            for i, rl in enumerate(blk.mgr_locs):
                item = blk.iget((i, loc))
                if (
                    result.dtype.kind in "iub"
                    and lib.is_float(item)
                    and isna(item)
                    and isinstance(blk.dtype, CategoricalDtype)
                ):
                    # GH#58954 caused bc interleaved_dtype is wrong for Categorical
                    # TODO(GH#38240) this will be unnecessary
                    # Note that doing this in a try/except would work for the
                    #  integer case, but not for bool, which will cast the NaN
                    #  entry to True.
                    if result.dtype.kind == "b":
                        new_dtype = object
                    else:
                        new_dtype = np.float64
                    result = result.astype(new_dtype)
                result[rl] = item

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)

        bp = BlockPlacement(slice(0, len(result)))
        block = new_block(result, placement=bp, ndim=1)
        return SingleBlockManager(block, self.axes[0].view())

    def iget(self, i: int, track_ref: bool = True) -> SingleBlockManager:
        # [FUNCTION DOC]
        # Return the column at the given item position as a one-dimensional manager. Parameters: i (item position), track_ref (default True; when false the new block does not register as a view of the original, used for temporary managers that mutate in place). The returned manager uses the row axis as its index.
        """
        Return the data as a SingleBlockManager.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])

        # shortcut for select a single-dim from a 2-dim BM
        bp = BlockPlacement(slice(0, len(values)))
        nb = type(block)(
            values, placement=bp, ndim=1, refs=block.refs if track_ref else None
        )
        return SingleBlockManager(nb, self.axes[1].view())

    def iget_values(self, i: int) -> ArrayLike:
        # [FUNCTION DOC]
        # Return the raw values of the column at the given position (an array that is a view). Does not honor copy-on-write; use with caution.
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
        # TODO(CoW) making the arrays read-only might make this safer to use?
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return values

    @property
    def column_arrays(self) -> list[np.ndarray]:
        # [FUNCTION DOC]
        # Property for the JSON writer: list with one array per item (in item order), obtained from each block's array values converted for JSON. Does not honor copy-on-write.
        # KNOWN ISSUE: the docstring's second sentence is cut off mid-sentence ("by converting each"), so it does not finish the description.
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
        # This is an optimized equivalent to
        #  result = [self.iget_values(i) for i in range(len(self.items))]
        result: list[np.ndarray | None] = [None] * len(self.items)

        for blk in self.blocks:
            mgr_locs = blk._mgr_locs
            values = blk.array_values._values_for_json()
            if values.ndim == 1:
                # TODO(EA2D): special casing not needed with 2D EAs
                result[mgr_locs[0]] = values

            else:
                for i, loc in enumerate(mgr_locs):
                    result[loc] = values[i]

        # error: Incompatible return value type (got "List[None]",
        # expected "List[ndarray[Any, Any]]")
        return result  # type: ignore[return-value]

    def iset(
        self,
        loc: int | slice | np.ndarray,
        value: ArrayLike,
        inplace: bool = False,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Set a new value for one or more items, in place, without consolidating; adds a block if the items are not covered.
        # Parameters: loc (integer, slice or array of item positions), value (array; two-dimensional values are transposed and one-dimensional ones reshaped to a single row, unless the dtype is a one-dimensional-only extension type), inplace (default False: when True and the target block can store the value, overwrite its data), refs (reference tracker for the new values). Raises an assertion error if the trailing shape is incompatible.
        # For a single integer location in a single-item block, uses a fast path that swaps the block. Otherwise processes each affected block: storable in place values are written (splitting the block first if it has references, for copy-on-write); others mark their positions as needing a new block and the old block is split or scheduled for removal when entirely replaced. Removed blocks update the block-number lookup; new blocks are appended (one per item for extension values, a single block otherwise) with lookup arrays updated; consolidation status is marked unknown.
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """

        # FIXME: refactor, clearly separate broadcasting & zip-like assignment
        #        can prob also fix the various if tests for sparse/categorical
        if self._blklocs is None and self.ndim > 1:
            self._rebuild_blknos_and_blklocs()

        # Note: we exclude DTA/TDA here
        value_is_extension_type = is_1d_only_ea_dtype(value.dtype)
        if not value_is_extension_type:
            if value.ndim == 2:
                value = value.T
            else:
                value = ensure_block_shape(value, ndim=2)

            if value.shape[1:] != self.shape[1:]:
                raise AssertionError(
                    "Shape of new values must be compatible with manager shape"
                )

        if lib.is_integer(loc):
            # We have 6 tests where loc is _not_ an int.
            # In this case, get_blkno_placements will yield only one tuple,
            #  containing (self._blknos[loc], BlockPlacement(slice(0, 1, 1)))

            # Check if we can use _iset_single fastpath
            loc = cast(int, loc)
            blkno = self.blknos[loc]
            blk = self.blocks[blkno]
            if len(blk._mgr_locs) == 1:  # TODO: fastest way to check this?
                return self._iset_single(
                    loc,
                    value,
                    inplace=inplace,
                    blkno=blkno,
                    blk=blk,
                    refs=refs,
                )

            # error: Incompatible types in assignment (expression has type
            # "List[Union[int, slice, ndarray]]", variable has type "Union[int,
            # slice, ndarray]")
            loc = [loc]  # type: ignore[assignment]

        # categorical/sparse/datetimetz
        if value_is_extension_type:

            def value_getitem(placement):
                # [FUNCTION DOC]
                # HELPER: nested accessor used when the value is a one-dimensional-only extension array: ignores the placement and returns the whole value.
                return value

        else:

            def value_getitem(placement):
                # [FUNCTION DOC]
                # HELPER: nested accessor for array values: returns the rows of the value selected by the placement indexer.
                return value[placement.indexer]

        # Accessing public blknos ensures the public versions are initialized
        blknos = self.blknos[loc]
        blklocs = self.blklocs[loc].copy()

        unfit_mgr_locs = []
        unfit_val_locs = []
        removed_blknos = []
        for blkno_l, val_locs in libinternals.get_blkno_placements(blknos, group=True):
            blk = self.blocks[blkno_l]
            blk_locs = blklocs[val_locs.indexer]
            if inplace and blk.should_store(value):
                # Updating inplace -> check if we need to do Copy-on-Write
                if not self._has_no_reference_block(blkno_l):
                    self._iset_split_block(
                        blkno_l, blk_locs, value_getitem(val_locs), refs=refs
                    )
                else:
                    blk.set_inplace(blk_locs, value_getitem(val_locs))
                    continue
            else:
                unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
                unfit_val_locs.append(val_locs)

                # If all block items are unfit, schedule the block for removal.
                if len(val_locs) == len(blk.mgr_locs):
                    removed_blknos.append(blkno_l)
                    continue
                else:
                    # Defer setting the new values to enable consolidation
                    self._iset_split_block(blkno_l, blk_locs, refs=refs)

        if removed_blknos:
            # Remove blocks & update blknos accordingly
            is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
            is_deleted[removed_blknos] = True

            new_blknos = np.empty(self.nblocks, dtype=np.intp)
            new_blknos.fill(-1)
            new_blknos[~is_deleted] = np.arange(self.nblocks - len(removed_blknos))
            self._blknos = new_blknos[self._blknos]
            self.blocks = tuple(
                blk for i, blk in enumerate(self.blocks) if i not in set(removed_blknos)
            )

        if unfit_val_locs:
            unfit_idxr = np.concatenate(unfit_mgr_locs)
            unfit_count = len(unfit_idxr)

            new_blocks: list[Block] = []
            if value_is_extension_type:
                # This code (ab-)uses the fact that EA blocks contain only
                # one item.
                # TODO(EA2D): special casing unnecessary with 2D EAs
                new_blocks.extend(
                    new_block_2d(
                        values=value,
                        placement=BlockPlacement(slice(mgr_loc, mgr_loc + 1)),
                        refs=refs,
                    )
                    for mgr_loc in unfit_idxr
                )

                self._blknos[unfit_idxr] = np.arange(unfit_count) + len(self.blocks)
                self._blklocs[unfit_idxr] = 0

            else:
                # unfit_val_locs contains BlockPlacement objects
                unfit_val_items = unfit_val_locs[0].append(unfit_val_locs[1:])

                new_blocks.append(
                    new_block_2d(
                        values=value_getitem(unfit_val_items),
                        placement=BlockPlacement(unfit_idxr),
                        refs=refs,
                    )
                )

                self._blknos[unfit_idxr] = len(self.blocks)
                self._blklocs[unfit_idxr] = np.arange(unfit_count)

            self.blocks += tuple(new_blocks)

            # Newly created block's dtype may already be present.
            self._known_consolidated = False

    def _iset_split_block(
        self,
        blkno_l: int,
        blk_locs: np.ndarray | list[int],
        value: ArrayLike | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: remove columns from a block by splitting it, updating the block tuple and lookup arrays.
        # Parameters: blkno_l (block number), blk_locs (positions within the block to remove), value (optional replacement values to place at those items), refs (reference tracker for the new values). The block delete yields the remaining pieces. If a value is given, a new block with it replaces the removed positions at the original block number; otherwise the first remaining piece takes its place. Any further pieces are appended. Lookup arrays are rebuilt first if absent and then updated for the new pieces, unless no split happened. Mutates the manager; returns nothing.
        """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determining the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
        blk = self.blocks[blkno_l]

        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()

        nbs_tup = tuple(blk.delete(blk_locs))
        if value is not None:
            locs = blk.mgr_locs.as_array[blk_locs]
            first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
        else:
            first_nb = nbs_tup[0]
            nbs_tup = tuple(nbs_tup[1:])

        nr_blocks = len(self.blocks)
        blocks_tup = (
            *self.blocks[:blkno_l],
            first_nb,
            *self.blocks[blkno_l + 1 :],
            *nbs_tup,
        )
        self.blocks = blocks_tup

        if not nbs_tup and value is not None:
            # No need to update anything if split did not happen
            return

        self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))

        for i, nb in enumerate(nbs_tup):
            self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
            self._blknos[nb.mgr_locs.indexer] = i + nr_blocks

    def _iset_single(
        self,
        loc: int,
        value: ArrayLike,
        inplace: bool,
        blkno: int,
        blk: Block,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: fast path of the set-item routine for a single position held by a single-column block. Parameters: loc, value, inplace, blkno, blk, refs. If in place and the block can store the value, overwrites the data (copying first when the block has references); otherwise swaps the whole block for a new block with the same placement. Lookup arrays remain valid. The caller must have checked the value shape. Returns nothing.
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
        # Caller is responsible for verifying value.shape

        if inplace and blk.should_store(value):
            copy = not self._has_no_reference_block(blkno)
            iloc = self.blklocs[loc]
            blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
            return

        nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
        old_blocks = self.blocks
        new_blocks = (*old_blocks[:blkno], nb, *old_blocks[blkno + 1 :])
        self.blocks = new_blocks
        return

    def column_setitem(
        self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Set values at an index within a single column, without assigning the whole column.
        # Parameters: loc (item position), idx (row indexer: integer, slice or array), value, inplace_only (default False). If the column's block has references, first splits that column out with a copy of its data. Then gets a temporary single-column manager without reference tracking. With inplace_only the values are written in place (dtype never changes); otherwise the set-item logic runs (possibly changing dtype) and the result is stored back in place into the column. Returns nothing. Avoids creating an intermediate Series.
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        if not self._has_no_reference(loc):
            blkno = self.blknos[loc]
            # Split blocks to only copy the column we want to modify
            blk_loc = self.blklocs[loc]
            # Copy our values
            values = self.blocks[blkno].values
            if values.ndim == 1:
                values = values.copy()
            else:
                # Use [blk_loc] as indexer to keep ndim=2, this already results in a
                # copy
                values = values[[blk_loc]]
            self._iset_split_block(blkno, [blk_loc], values)

        # this manager is only created temporarily to mutate the values in place
        # so don't track references, otherwise the `setitem` would perform CoW again
        col_mgr = self.iget(loc, track_ref=False)
        if inplace_only:
            col_mgr.setitem_inplace(idx, value)
        else:
            new_mgr = col_mgr.setitem((idx,), value)
            self.iset(loc, new_mgr._block.values, inplace=True)

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        # [FUNCTION DOC]
        # Insert a new column at a position. Parameters: loc (position), item (new label), value (array; a two-dimensional value is transposed and must have a single row, else a value error; one-dimensional is reshaped), refs (reference tracker). Creates a one-column block, updates the placements of blocks above and the lookup arrays (fast path when there are no blocks), inserts the label into the items axis, appends the block and marks consolidation unknown. Emits a performance warning (when the option is on) if more than one hundred non-extension blocks exist. Mutates in place; returns nothing.
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        refs : The reference tracking object of the value to set.
        """
        new_axis = self.items.insert(loc, item)

        if value.ndim == 2:
            value = value.T
            if len(value) > 1:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.T.shape}"
                )
        else:
            value = ensure_block_shape(value, ndim=self.ndim)

        bp = BlockPlacement(slice(loc, loc + 1))
        block = new_block_2d(values=value, placement=bp, refs=refs)

        if not len(self.blocks):
            # Fastpath
            self._blklocs = np.array([0], dtype=np.intp)
            self._blknos = np.array([0], dtype=np.intp)
        else:
            self._insert_update_mgr_locs(loc)
            self._insert_update_blklocs_and_blknos(loc)

        self.axes[0] = new_axis
        self.blocks += (block,)

        self._known_consolidated = False

        if (
            get_option("performance_warnings")
            and sum(not block.is_extension for block in self.blocks) > 100
        ):
            warnings.warn(
                "DataFrame is highly fragmented.  This is usually the result "
                "of calling `frame.insert` many times, which has poor performance.  "
                "Consider joining all columns at once using pd.concat(axis=1) "
                "instead. To get a de-fragmented frame, use `newframe = frame.copy()`",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

    def _insert_update_mgr_locs(self, loc) -> None:
        # [FUNCTION DOC]
        # HELPER: when a column is inserted at a position, increments the placements of all blocks holding items at or above that position by one.
        """
        When inserting a new Block at location 'loc', we increment
        all of the mgr_locs of blocks above that by one.
        """
        # Faster version of set(arr) for sequences of small numbers
        blknos = np.bincount(self.blknos[loc:]).nonzero()[0]
        for blkno in blknos:
            # .620 this way, .326 of which is in increment_above
            blk = self.blocks[blkno]
            blk._mgr_locs = blk._mgr_locs.increment_above(loc)

    def _insert_update_blklocs_and_blknos(self, loc) -> None:
        # [FUNCTION DOC]
        # HELPER: update the lookup arrays after inserting an item at a position: the new item maps to the newest block and in-block position zero. Uses an append when inserting at the end, a prepend when at the start, and the compiled helper for the middle.
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """

        # Accessing public blklocs ensures the public versions are initialized
        if loc == self.blklocs.shape[0]:
            # np.append is a lot faster, let's use it if we can.
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        elif loc == 0:
            # As of numpy 1.26.4, np.concatenate faster than np.append
            self._blklocs = np.concatenate([[0], self._blklocs])
            self._blknos = np.concatenate([[len(self.blocks)], self._blknos])
        else:
            new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(
                self.blklocs, self.blknos, loc, len(self.blocks)
            )
            self._blklocs = new_blklocs
            self._blknos = new_blknos

    def idelete(self, indexer) -> BlockManager:
        # [FUNCTION DOC]
        # Delete items at the given positions and return a new manager (the original is not changed). Parameter: indexer (positions). Builds the complement of the deleted positions, takes those items as views without tracking references (in-place operation), takes the labels from the items axis and keeps the second axis. Built without verification.
        """
        Delete selected locations, returning a new BlockManager.
        """
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        taker = (~is_deleted).nonzero()[0]

        nbs = self._slice_take_blocks_ax0(taker, only_slice=True, ref_inplace_op=True)
        new_columns = self.items[~is_deleted]
        axes = [new_columns, self.axes[1]]
        return type(self)(tuple(nbs), axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Block-wise Operation

    def grouped_reduce(self, func: Callable) -> Self:
        # [FUNCTION DOC]
        # Apply a grouped reduction function to each block and return a new manager.
        # Parameter: func, a callable applied to each block (through the block-level apply). Object-dtype blocks are split into columns first because some columns may fail while others succeed. The row axis of the result is a default range index whose length is that of the first result block (zero if none). Returns a new manager with views of the items axis.
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
        result_blocks: list[Block] = []

        for blk in self.blocks:
            if blk.is_object:
                # split on object-dtype blocks bc some columns may raise
                #  while others do not.
                for sb in blk._split():
                    applied = sb.apply(func)
                    result_blocks = extend_blocks(applied, result_blocks)
            else:
                applied = blk.apply(func)
                result_blocks = extend_blocks(applied, result_blocks)

        if len(result_blocks) == 0:
            nrows = 0
        else:
            nrows = result_blocks[0].values.shape[-1]
        index = default_index(nrows)

        # TODO shallow copy columns?
        return type(self).from_blocks(result_blocks, [self.axes[0].view(), index])

    def reduce(self, func: Callable) -> Self:
        # [FUNCTION DOC]
        # Apply a reduction function to each block and return a manager with a single placeholder row (default index of length one). Parameter: func, the reduction callable. Asserts the manager is two-dimensional. The items axis is reused without copy. Returns a new manager.
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        BlockManager
        """
        # If 2D, we assume that we're operating column-wise
        assert self.ndim == 2

        res_blocks = [blk.reduce(func) for blk in self.blocks]
        index = default_index(1)  # placeholder
        # shallow copy self.items not needed because DataFrame._reduce does a getitem
        new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return new_mgr

    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        # [FUNCTION DOC]
        # Apply an elementwise array operation with another aligned manager, block by block, delegating to the module-level blockwise operation function. Parameters: other (manager) and array_op (callable). Returns a new manager.
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        return operate_blockwise(self, other, array_op)

    def _equal_values(self: BlockManager, other: BlockManager) -> bool:
        # [FUNCTION DOC]
        # Value comparison used by the equality method: compares block pairs with the array equality function through the blockwise-all helper, assuming shape and axes have been checked. Returns a boolean.
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        return blockwise_all(self, other, array_equals)

    def quantile(
        self,
        *,
        qs: Index,  # with dtype float 64
        interpolation: QuantileInterpolation = "linear",
    ) -> Self:
        # [FUNCTION DOC]
        # Compute quantiles of each block. Keyword-only parameters: qs (index of float64 quantile locations; asserted list-like and the manager asserted at least two-dimensional), interpolation (default linear). The result has the items axis (view) and a new axis of the quantiles. Each block computes its quantile (with type inference on the result) and a new manager is built with integrity verification. Returns the new manager.
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
        # Series dispatches to DataFrame for quantile, which allows us to
        #  simplify some of the code here and in the blocks
        assert self.ndim >= 2
        assert is_list_like(qs)  # caller is responsible for this

        new_axes = [self.axes[0].view(), Index(qs, dtype=np.float64)]

        blocks = [
            blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
        ]

        return type(self)(blocks, new_axes)

    # ----------------------------------------------------------------

    def unstack(self, unstacker, fill_value) -> BlockManager:
        # [FUNCTION DOC]
        # Return a manager with all blocks unstacked (used for reshape).
        # Parameters: unstacker (the reshape helper object providing new columns, new index, masks and shape), fill_value (value for newly introduced missing entries). Computes whether filling/masking is needed per output column, the replication factor of the columns (asserts the new column count is an integer multiple of the old count), unstacks each block with its tiled placement, concatenates the new blocks and column masks (asserts the mask count equals the number of produced items), filters the new columns by the mask. Returns a new two-dimensional manager built without verification.
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        new_columns = unstacker.get_new_columns(self.items)
        new_index = unstacker.new_index

        allow_fill = not unstacker.mask_all
        if allow_fill:
            # calculating the full mask once and passing it to Block._unstack is
            #  faster than letting calculating it in each repeated call
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        else:
            needs_masking = np.zeros(unstacker.full_shape[1], dtype=bool)

        new_blocks: list[Block] = []
        columns_mask: list[np.ndarray] = []

        if len(self.items) == 0:
            factor = 1
        else:
            fac = len(new_columns) / len(self.items)
            assert fac == int(fac)
            factor = int(fac)

        for blk in self.blocks:
            mgr_locs = blk.mgr_locs
            new_placement = mgr_locs.tile_for_unstack(factor)

            blocks, mask = blk._unstack(
                unstacker,
                fill_value,
                new_placement=new_placement,
                needs_masking=needs_masking,
            )

            new_blocks.extend(blocks)
            columns_mask.extend(mask)

            # Block._unstack should ensure this holds,
            assert mask.sum() == sum(len(nb._mgr_locs) for nb in blocks)
            # In turn this ensures that in the BlockManager call below
            #  we have len(new_columns) == sum(x.shape[0] for x in new_blocks)
            #  which suffices to allow us to pass verify_inegrity=False

        new_columns = new_columns[columns_mask]

        bm = BlockManager(new_blocks, [new_columns, new_index], verify_integrity=False)
        return bm

    def to_iter_dict(self) -> Generator[tuple[str, Self]]:
        # [FUNCTION DOC]
        # Generator yielding pairs of (string name of the dtype, manager) grouping the blocks by dtype name. Blocks are sorted by that name, grouped, and each group is combined into a manager.
        # KNOWN ISSUE: the docstring Returns section describes a single tuple, but this is a generator that yields many tuples.
        """
        Yield a tuple of (str(dtype), BlockManager)

        Returns
        -------
        values : a tuple of (str(dtype), BlockManager)
        """
        key = lambda block: str(block.dtype)
        for dtype, blocks in itertools.groupby(sorted(self.blocks, key=key), key=key):
            # TODO(EA2D): the combine will be unnecessary with 2D EAs
            yield dtype, self._combine(list(blocks))

    def as_array(
        self,
        dtype: np.dtype | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Convert the data to a numpy array with items as columns (transposed result).
        # Parameters: dtype (optional numpy dtype), copy (default False: no guarantee data is not copied, but a read-only view is returned when no copy is made), na_value (value to use for missing entries; the no-default sentinel leaves them as is). With no blocks returns an empty float array of the transposed shape. With a single block: forces a copy when a non-NaN missing value is given and the block is not already numpy float; extension blocks use their conversion method; datetime-like blocks have missing entries set when NaN was passed. Multi-block data are interleaved into one array (which copies). Finally missing entries are replaced with na_value unless that is the default or a NaN for a float result. Returns the transposed array.
        """
        Convert the blockmanager data into a numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        passed_nan = lib.is_float(na_value) and isna(na_value)

        if len(self.blocks) == 0:
            arr = np.empty(self.shape, dtype=float)
            return arr.transpose()

        if self.is_single_block:
            blk = self.blocks[0]

            if na_value is not lib.no_default:
                # We want to copy when na_value is provided to avoid
                # mutating the original object
                if lib.is_np_dtype(blk.dtype, "f") and passed_nan:
                    # We are already numpy-float and na_value=np.nan
                    pass
                else:
                    copy = True

            if blk.is_extension:
                # Avoid implicit conversion of extension blocks to object

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                    copy=copy,
                ).reshape(blk.shape)
            elif not copy:
                arr = np.asarray(blk.values, dtype=dtype)
            else:
                arr = np.array(blk.values, dtype=dtype, copy=copy)
            if passed_nan and blk.dtype.kind in "mM":
                arr[isna(blk.values)] = na_value

            if not copy:
                arr = arr.view()
                arr.flags.writeable = False
        else:
            arr = self._interleave(dtype=dtype, na_value=na_value)
            # The underlying data was copied within _interleave, so no need
            # to further copy if copy=True or setting na_value

        if na_value is lib.no_default:
            pass
        elif arr.dtype.kind == "f" and passed_nan:
            pass
        else:
            arr[isna(arr)] = na_value

        return arr.transpose()

    def _interleave(
        self,
        dtype: np.dtype | None = None,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: build one array from all blocks. Parameters: dtype (optional numpy dtype; defaults to the common dtype of the blocks, converted to a numpy dtype) and na_value. For object dtype without a na_value, blocks are copied in directly. Otherwise each block's values are converted (extension blocks via their conversion method with the dtype and na_value; others via their get-values with the dtype) and placed at the block positions, setting missing datetime-like entries if a na_value is given. Raises an assertion error if some items were not covered by blocks. Returns an array of the manager's shape.
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        if not dtype:
            # Incompatible types in assignment (expression has type
            # "Optional[Union[dtype[Any], ExtensionDtype]]", variable has
            # type "Optional[dtype[Any]]")
            dtype = interleaved_dtype(  # type: ignore[assignment]
                [blk.dtype for blk in self.blocks]
            )

        # error: Argument 1 to "ensure_np_dtype" has incompatible type
        # "Optional[dtype[Any]]"; expected "Union[dtype[Any], ExtensionDtype]"
        dtype = ensure_np_dtype(dtype)  # type: ignore[arg-type]
        result = np.empty(self.shape, dtype=dtype)

        itemmask = np.zeros(self.shape[0])

        if dtype == np.dtype("object") and na_value is lib.no_default:
            # much more performant than using to_numpy below
            for blk in self.blocks:
                rl = blk.mgr_locs
                arr = blk.get_values(dtype)
                result[rl.indexer] = arr
                itemmask[rl.indexer] = 1
            return result

        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                # Avoid implicit conversion of extension blocks to object

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                )
            else:
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            if na_value is not lib.no_default and blk.dtype.kind in "mM":
                result[rl.indexer][isna(arr)] = na_value
            itemmask[rl.indexer] = 1

        if not itemmask.all():
            raise AssertionError("Some items were not contained in blocks")

        return result

    # ----------------------------------------------------------------
    # Consolidation

    def is_consolidated(self) -> bool:
        # [FUNCTION DOC]
        # Whether the blocks are consolidated: using the cached known state, or computing it first. Returns the cached flag.
        # KNOWN ISSUE: the docstring says it returns True if there is more than one block with the same dtype, which is the opposite of the meaning; it returns True when no two consolidatable blocks share a dtype.
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self) -> None:
        # [FUNCTION DOC]
        # HELPER: compute and cache whether the manager is consolidated: a single block is consolidated; otherwise compares the number of consolidatable blocks to the number of distinct dtypes. Sets both the flag and the known flag.
        if len(self.blocks) == 1:
            # fastpath
            self._is_consolidated = True
            self._known_consolidated = True
            return
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True

    def _consolidate_inplace(self) -> None:
        # [FUNCTION DOC]
        # Consolidate in place: if not consolidated, merges blocks of the same dtype (leaving non-consolidatable blocks alone), sets the consolidated flags and rebuilds the lookup arrays.
        if not self.is_consolidated():
            self.blocks = _consolidate(self.blocks)
            self._is_consolidated = True
            self._known_consolidated = True
            self._rebuild_blknos_and_blklocs()

    # ----------------------------------------------------------------
    # Concatenation

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        # [FUNCTION DOC]
        # Class method: concatenate managers with identical row axes side by side. Parameters: mgrs (list of managers) and axes (the axes of the result). Each block is column-sliced (so original placements are not altered) and its placement shifted by the cumulative item count of the preceding managers. Returns a new manager of this class with verification.
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
        offset = 0
        blocks: list[Block] = []
        for mgr in mgrs:
            for blk in mgr.blocks:
                # We need to do getitem_block here otherwise we would be altering
                #  blk.mgr_locs in place, which would render it invalid. This is only
                #  relevant in the copy=False case.
                nb = blk.slice_block_columns(slice(None))
                nb._mgr_locs = nb._mgr_locs.add(offset)
                blocks.append(nb)

            offset += len(mgr.items)

        # TODO relevant axis already shallow-copied at caller?
        new_mgr = cls(tuple(blocks), axes)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        # [FUNCTION DOC]
        # Class method placeholder for vertical concatenation: always raises a not-implemented error, because that logic lives in the concatenation module of the internals package.
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
        raise NotImplementedError("This logic lives (for now) in internals.concat")


class SingleBlockManager(BaseBlockManager):
    # [CLASS DOC]
    # One-dimensional block manager holding exactly one block, backing Series. Inherits the shared base logic. Class attributes: the consolidation flags are fixed to true, the single-block flag is true, and slots are empty. State: the axes list has one index (the series index) and blocks is a one-tuple. The lookup arrays are always none.
    # KNOWN ISSUE: the class docstring is truncated ("manage a single block with") and does not complete the sentence.
    """manage a single block with"""

    @property
    def ndim(self) -> Literal[1]:
        # [FUNCTION DOC]
        # The number of dimensions, always one.
        return 1

    _is_consolidated = True
    _known_consolidated = True
    __slots__ = ()
    is_single_block = True

    def __init__(
        self,
        block: Block,
        axis: Index,
        verify_integrity: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Create a manager from a block and an index. Parameters: block, axis (the index) and verify_integrity (ignored; present for signature compatibility). Stores a list with the axis and a one-tuple with the block; no validation.
        # Assertions disabled for performance
        # assert isinstance(block, Block), type(block)
        # assert isinstance(axis, Index), type(axis)

        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(
        cls,
        blocks: list[Block],
        axes: list[Index],
    ) -> Self:
        # [FUNCTION DOC]
        # Class constructor with the same signature as the two-dimensional manager's. Asserts that exactly one block and one axis are given and builds the manager from them.
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        assert len(blocks) == 1
        assert len(axes) == 1
        return cls(blocks[0], axes[0], verify_integrity=False)

    @classmethod
    def from_array(
        cls, array: ArrayLike, index: Index, refs: BlockValuesRefs | None = None
    ) -> SingleBlockManager:
        # [FUNCTION DOC]
        # Create a manager from a bare array. Parameters: array (numpy or extension array; coerced by the value-coercion helper), index, refs (optional reference tracker). Wraps the array in a one-dimensional block spanning the index length. Returns a single-block manager.
        """
        Constructor for if we have an array that is not yet a Block.
        """
        array = maybe_coerce_values(array)
        bp = BlockPlacement(slice(0, len(index)))
        block = new_block(array, placement=bp, ndim=1, refs=refs)
        return cls(block, index)

    def to_2d_mgr(self, columns: Index) -> BlockManager:
        # [FUNCTION DOC]
        # Manager analogue of converting a Series to a frame. Parameter: columns (the new items axis). Reshapes the values to two dimensions, creates a block of the same type with placement at the first position sharing the reference tracker, and returns a two-dimensional manager with the columns and a view of the index, built without verification.
        """
        Manager analogue of Series.to_frame
        """
        blk = self.blocks[0]
        arr = ensure_block_shape(blk.values, ndim=2)
        bp = BlockPlacement(0)
        new_blk = type(blk)(arr, placement=bp, ndim=2, refs=blk.refs)
        axes = [columns, self.axes[0].view()]
        return BlockManager([new_blk], axes=axes, verify_integrity=False)

    def _has_no_reference(self, i: int = 0) -> bool:
        # [FUNCTION DOC]
        # Whether the single block has no references; the argument is ignored (default zero). Returns a boolean.
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        return not self.blocks[0].refs.has_reference()

    def __getstate__(self):
        # [FUNCTION DOC]
        # Pickle support. Returns a four-tuple: the axes list, the list of block values, the list of item labels per block (these first three keep compatibility with an old format), and a dictionary under the old-format version key holding the axes and, per block, its values and placement indexer.
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)

        extra_state = {
            "0.14.1": {
                "axes": axes_array,
                "blocks": [
                    {"values": b.values, "mgr_locs": b.mgr_locs.indexer}
                    for b in self.blocks
                ],
            }
        }

        # First three elements of the state are to maintain forward
        # compatibility with 0.13.1.
        return axes_array, block_values, block_items, extra_state

    def __setstate__(self, state) -> None:
        # [FUNCTION DOC]
        # Restore state when unpickling. Accepts only the tuple format of version 0.14.1 (a tuple of at least four with that key in the fourth element), rebuilding axes (coerced to indexes) and blocks; any other state raises a not-implemented error saying older pickles are unsupported. Afterwards calls the post-restore hook.
        def unpickle_block(values, mgr_locs, ndim: int) -> Block:
            # [FUNCTION DOC]
            # HELPER: nested in the state restoration. Builds a block from stored values, a placement and a number of dimensions: unwraps index-like values to raw arrays, converts the placement to a placement object if necessary, coerces the values and creates the block.
            # TODO(EA2D): ndim would be unnecessary with 2D EAs
            # older pickles may store e.g. DatetimeIndex instead of DatetimeArray
            values = extract_array(values, extract_numpy=True)
            if not isinstance(mgr_locs, BlockPlacement):
                mgr_locs = BlockPlacement(mgr_locs)

            values = maybe_coerce_values(values)
            return new_block(values, placement=mgr_locs, ndim=ndim)

        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            state = state[3]["0.14.1"]
            self.axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(self.axes)
            self.blocks = tuple(
                unpickle_block(b["values"], b["mgr_locs"], ndim=ndim)
                for b in state["blocks"]
            )
        else:
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        self._post_setstate()

    def _post_setstate(self) -> None:
        # [FUNCTION DOC]
        # Hook called at the end of state restoration; does nothing here (subclasses may extend).
        pass

    @cache_readonly
    def _block(self) -> Block:
        # [FUNCTION DOC]
        # Cached property: the only block.
        return self.blocks[0]

    @final
    @property
    def array(self) -> ArrayLike:
        # [FUNCTION DOC]
        # Property (final): the array backing the single block.
        """
        Quick access to the backing array of the Block.
        """
        return self.blocks[0].values

    # error: Cannot override writeable attribute with read-only property
    @property
    def _blknos(self) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Compatibility property: always none, since a one-dimensional manager has no lookup arrays.
        """compat with BlockManager"""
        return None

    # error: Cannot override writeable attribute with read-only property
    @property
    def _blklocs(self) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # Compatibility property: always none.
        """compat with BlockManager"""
        return None

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self:
        # [FUNCTION DOC]
        # Select rows by an indexer, not restricted to slices. Parameter: indexer (boolean array or other positional indexer). If it is a non-empty all-true array, returns a manager over a shallow copy of the block and the same index. Otherwise indexes the values; the new block shares the reference tracker unless the indexer is a boolean array (which always copies in numpy). The index is selected the same way. Returns a new manager.
        # similar to get_slice, but not restricted to slice indexer
        blk = self._block
        if len(indexer) > 0 and indexer.all():
            return type(self)(blk.copy(deep=False), self.index)
        array = blk.values[indexer]

        if isinstance(indexer, np.ndarray) and indexer.dtype.kind == "b":
            # boolean indexing always gives a copy with numpy
            refs = None
        else:
            # TODO(CoW) in theory only need to track reference if new_array is a view
            refs = blk.refs

        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(array, placement=bp, ndim=1, refs=refs)

        new_idx = self.index[indexer]
        return type(self)(block, new_idx)

    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> SingleBlockManager:
        # [FUNCTION DOC]
        # Slice rows. Parameters: slobj (a slice; not asserted) and axis (default zero; an axis at or above the number of dimensions raises an index error). The new block holds a view of the sliced values and shares the reference tracker. The index is sliced. Returns a new single-block manager.
        # Assertion disabled for performance
        # assert isinstance(slobj, slice), type(slobj)
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        blk = self._block
        array = blk.values[slobj]
        bp = BlockPlacement(slice(0, len(array)))
        # TODO this method is only used in groupby SeriesSplitter at the moment,
        # so passing refs is not yet covered by the tests
        block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
        new_index = self.index._getitem_slice(slobj)
        return type(self)(block, new_index)

    @property
    def index(self) -> Index:
        # [FUNCTION DOC]
        # The index: the only axis.
        return self.axes[0]

    @property
    def dtype(self) -> DtypeObj:
        # [FUNCTION DOC]
        # The dtype of the single block.
        return self._block.dtype

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # Return an object array containing the single dtype.
        return np.array([self._block.dtype], dtype=object)

    def external_values(self):
        # [FUNCTION DOC]
        # The array that the user-facing values attribute of a Series returns (from the block).
        """The array that Series.values returns"""
        return self._block.external_values()

    def internal_values(self):
        # [FUNCTION DOC]
        # The block values (the internal array that the underscore values attribute returns).
        """The array that Series._values returns"""
        return self._block.values

    def array_values(self) -> ExtensionArray:
        # [FUNCTION DOC]
        # The extension array wrapping the block values (what the array attribute of a Series returns), taken from the block. Defined as a method, not a property.
        """The array that Series.array returns"""
        return self._block.array_values

    def get_numeric_data(self) -> Self:
        # [FUNCTION DOC]
        # Return numeric data: a shallow copy if the block is numeric, otherwise an empty manager.
        if self._block.is_numeric:
            return self.copy(deep=False)
        return self.make_empty()

    @property
    def _can_hold_na(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether the block's dtype can hold missing values.
        return self._block._can_hold_na

    def setitem_inplace(self, indexer, value) -> None:
        # [FUNCTION DOC]
        # Assign values at an indexer in place, never changing the dtype and never returning a new manager (backs item assignment on a Series).
        # Parameters: indexer, value. If the block has references, first replaces the block by a deep copy and resets the cache. For numpy arrays, the value is checked to be holdable by the dtype (may raise); extension arrays validate on their own. A one-element one-dimensional array value is reduced to a scalar-like. Then writes the value at the indexer. Mutates the array.
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        if not self._has_no_reference(0):
            self.blocks = (self._block.copy(deep=True),)
            self._reset_cache()

        arr = self.array

        # EAs will do this validation in their own __setitem__ methods.
        if isinstance(arr, np.ndarray):
            # Note: checking for ndarray instead of np.dtype means we exclude
            #  dt64/td64, which do their own validation.
            value = np_can_hold_element(arr.dtype, value)

        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == 1:
            # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
            value = value[0, ...]

        arr[indexer] = value

    def idelete(self, indexer) -> SingleBlockManager:
        # [FUNCTION DOC]
        # Delete a position from the single block and from the index, in place. Parameter: indexer. Replaces the block with the first block of the block-level delete result (so the block never becomes empty list-wise), deletes from the index, resets the cached block property, and returns self.
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
        nb = self._block.delete(indexer)[0]
        self.blocks = (nb,)
        self.axes[0] = self.axes[0].delete(indexer)
        self._reset_cache()
        return self

    def fast_xs(self, loc):
        # [FUNCTION DOC]
        # Not supported for one-dimensional managers: always raises a not-implemented error directing callers to index the values directly.
        """
        fast path for getting a cross-section
        return a view of the data
        """
        raise NotImplementedError("Use series._values[loc] instead")

    def set_values(self, values: ArrayLike) -> None:
        # [FUNCTION DOC]
        # Replace the single block's values in place and reset its placement to span the new length.
        # Parameter: values (array). No validation of length or dtype and no reference tracking; for internal use where the caller handles copy-on-write.
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc),
        and this does not properly keep track of references.
        """
        # NOTE(CoW) Currently this is only used for FrameColumnApply.series_generator
        # which handles CoW by setting the refs manually if necessary
        self.blocks[0].values = values
        self.blocks[0]._mgr_locs = BlockPlacement(slice(len(values)))

    def _equal_values(self, other: Self) -> bool:
        # [FUNCTION DOC]
        # Value comparison used by the equality method. Returns False if the other manager is not one-dimensional; otherwise compares the two arrays with the array equality function. Shape and axes are assumed already checked.
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        # For SingleBlockManager (i.e.Series)
        if other.ndim != 1:
            return False
        left = self.blocks[0].values
        right = other.blocks[0].values
        return array_equals(left, right)

    def grouped_reduce(self, func):
        # [FUNCTION DOC]
        # Apply a reduction function to the array of the single block. Parameter: func (callable taking the array and returning an array-like of results). The result gets a default range index of its own length; returns a new single-block manager from the result array.
        arr = self.array
        res = func(arr)
        index = default_index(len(res))

        mgr = type(self).from_array(res, index)
        return mgr


# --------------------------------------------------------------------
# Constructor Helpers


def create_block_manager_from_blocks(
    blocks: list[Block],
    axes: list[Index],
    consolidate: bool = True,
    verify_integrity: bool = True,
) -> BlockManager:
    # [FUNCTION DOC]
    # Build a two-dimensional manager from blocks and axes. Parameters: blocks, axes, consolidate (default True: merge same-dtype blocks in place afterwards), verify_integrity (default True; if false the caller promises the block shapes and placements are consistent with the axes). A value error from construction is translated into a user-friendly shape error (using the first block's trailing shape and the total item count). Returns the manager.
    # If verify_integrity=False, then caller is responsible for checking
    #  all(x.shape[-1] == len(axes[1]) for x in blocks)
    #  sum(x.shape[0] for x in blocks) == len(axes[0])
    #  set(x for blk in blocks for x in blk.mgr_locs) == set(range(len(axes[0])))
    #  all(blk.ndim == 2 for blk in blocks)
    # This allows us to safely pass verify_integrity=False

    try:
        mgr = BlockManager(blocks, axes, verify_integrity=verify_integrity)

    except ValueError as err:
        arrays = [blk.values for blk in blocks]
        tot_items = sum(arr.shape[0] for arr in arrays)
        raise_construction_error(tot_items, arrays[0].shape[1:], axes, err)

    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def create_block_manager_from_column_arrays(
    arrays: list[ArrayLike],
    axes: list[Index],
    consolidate: bool,
    refs: list,
) -> BlockManager:
    # [FUNCTION DOC]
    # Build a two-dimensional manager from a list of one-dimensional column arrays. Parameters: arrays (one per column), axes, consolidate (whether to merge same-dtype columns into shared blocks), refs (one reference tracker per array). Caller guarantees validity; the manager is built without verification. A value error while forming blocks is converted into a shape error message. Consolidates in place if requested. Returns the manager.
    # Assertions disabled for performance (caller is responsible for verifying)
    # assert isinstance(axes, list)
    # assert all(isinstance(x, Index) for x in axes)
    # assert all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
    # assert all(type(x) is not NumpyExtensionArray for x in arrays)
    # assert all(x.ndim == 1 for x in arrays)
    # assert all(len(x) == len(axes[1]) for x in arrays)
    # assert len(arrays) == len(axes[0])
    # These last three are sufficient to allow us to safely pass
    #  verify_integrity=False below.

    try:
        blocks = _form_blocks(arrays, consolidate, refs)
        mgr = BlockManager(blocks, axes, verify_integrity=False)
    except ValueError as e:
        raise_construction_error(len(arrays), arrays[0].shape, axes, e)
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def raise_construction_error(
    tot_items: int,
    block_shape: Shape,
    axes: list[Index],
    e: ValueError | None = None,
) -> NoReturn:
    # [FUNCTION DOC]
    # Raise a helpful error about a shape mismatch during construction. Never returns.
    # Parameters: tot_items (total items given), block_shape (the shape of one block beyond the items axis), axes (the manager axes), e (optional original error). Builds the passed shape and the implied shape from the axes, reversing them when they have at most two dimensions to match user-facing order. If they are equal and an original error exists, that error is re-raised. If the first block dimension is zero, raises a value error saying empty data was passed with indices specified. Otherwise raises a value error stating both shapes.
    # KNOWN ISSUE: a comment says the exception object is returned instead of being raised, but the function raises it directly.
    """raise a helpful message about our construction"""
    passed = tuple(map(int, [tot_items, *block_shape]))
    # Correcting the user facing error message during dataframe construction
    if len(passed) <= 2:
        passed = passed[::-1]

    implied = tuple(len(ax) for ax in axes)
    # Correcting the user facing error message during dataframe construction
    if len(implied) <= 2:
        implied = implied[::-1]

    # We return the exception object instead of raising it so that we
    #  can raise it in the caller; mypy plays better with that
    if passed == implied and e is not None:
        raise e
    if block_shape[0] == 0:
        raise ValueError("Empty data passed with indices specified.")
    raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


# -----------------------------------------------------------------------


def _grouping_func(tup: tuple[int, ArrayLike]) -> tuple[int, DtypeObj]:
    # [FUNCTION DOC]
    # HELPER: grouping key for the block formation step. Parameter: tup is a pair of position and array. The key is a pair of a separator and the array dtype: for one-dimensional-only extension dtypes the separator is the object identity of the dtype, so equal dtypes are not merged (and comparisons avoided); otherwise zero. Returns the key.
    dtype = tup[1].dtype

    if is_1d_only_ea_dtype(dtype):
        # We know these won't be consolidated, so don't need to group these.
        # This avoids expensive comparisons of CategoricalDtype objects
        sep = id(dtype)
    else:
        sep = 0

    return sep, dtype


def _form_blocks(arrays: list[ArrayLike], consolidate: bool, refs: list) -> list[Block]:
    # [FUNCTION DOC]
    # HELPER: build blocks from a list of column arrays. Parameters: arrays, consolidate, refs. Without consolidation each array becomes its own block (reference trackers preserved). With consolidation (reference trackers ignored since stacking copies or the extension arrays were already copied): consecutive arrays are grouped by the grouping key; numpy dtypes are stacked into one block per group (string and bytes dtypes become object; datetime-like arrays are wrapped); one-dimensional-only extension dtypes become one block each; other extension dtypes become one block each with a reshaped array. Returns the list of blocks.
    tuples = enumerate(arrays)

    if not consolidate:
        return _tuples_to_blocks_no_consolidate(tuples, refs)

    # when consolidating, we can ignore refs (either stacking always copies,
    # or the EA is already copied in the calling dict_to_mgr)

    # group by dtype
    grouper = itertools.groupby(tuples, _grouping_func)

    nbs: list[Block] = []
    for (_, dtype), tup_block in grouper:
        block_type = get_block_type(dtype)

        if isinstance(dtype, np.dtype):
            is_dtlike = dtype.kind in "mM"

            if issubclass(dtype.type, (str, bytes)):
                dtype = np.dtype(object)

            values, placement = _stack_arrays(tup_block, dtype)
            if is_dtlike:
                values = ensure_wrapped_if_datetimelike(values)
            blk = block_type(values, placement=BlockPlacement(placement), ndim=2)
            nbs.append(blk)

        elif is_1d_only_ea_dtype(dtype):
            dtype_blocks = [
                block_type(x[1], placement=BlockPlacement(x[0]), ndim=2)
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)

        else:
            dtype_blocks = [
                block_type(
                    ensure_block_shape(x[1], 2), placement=BlockPlacement(x[0]), ndim=2
                )
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)
    return nbs


def _tuples_to_blocks_no_consolidate(tuples, refs) -> list[Block]:
    # [FUNCTION DOC]
    # HELPER: build one block per column array without consolidation. Parameters: tuples (iterable of position and array pairs, as produced by enumeration) and refs (one tracker per array, zipped strictly). Each array is reshaped to two dimensions and placed at its position. Returns the list of blocks.
    # tuples produced within _form_blocks are of the form (placement, array)
    return [
        new_block_2d(
            ensure_block_shape(arr, ndim=2), placement=BlockPlacement(i), refs=ref
        )
        for ((i, arr), ref) in zip(tuples, refs, strict=True)
    ]


def _stack_arrays(tuples, dtype: np.dtype):
    # [FUNCTION DOC]
    # HELPER: stack column arrays of one dtype into a single two-dimensional array. Parameters: tuples (iterable of position and array pairs) and dtype. Returns the pair of the stacked array (one row per input array) and the tuple of positions; arrays are copied into a fresh array of the given dtype.
    placement, arrays = zip(*tuples, strict=True)

    first = arrays[0]
    shape = (len(arrays), *first.shape)

    stacked = np.empty(shape, dtype=dtype)
    for i, arr in enumerate(arrays):
        stacked[i] = arr

    return stacked, placement


def _consolidate(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
    # [FUNCTION DOC]
    # HELPER: merge blocks of the same dtype. Parameter: blocks (tuple). Sorts and groups blocks by their consolidation key (whether they can consolidate, and dtype); each group is merged by the merge helper (non-consolidatable groups are left alone). Returns the new tuple of blocks.
    """
    Merge blocks having same dtype, exclude non-consolidating blocks
    """
    # sort by _can_consolidate, dtype
    gkey = lambda x: x._consolidate_key
    grouper = itertools.groupby(sorted(blocks, key=gkey), gkey)

    new_blocks: list[Block] = []
    for (_can_consolidate, dtype), group_blocks in grouper:
        merged_blocks, _ = _merge_blocks(
            list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
        )
        new_blocks = extend_blocks(merged_blocks, new_blocks)
    return tuple(new_blocks)


def _merge_blocks(
    blocks: list[Block], dtype: DtypeObj, can_consolidate: bool
) -> tuple[list[Block], bool]:
    # [FUNCTION DOC]
    # HELPER: merge a list of blocks of one dtype into a single block.
    # Parameters: blocks, dtype (unused here), can_consolidate. Returns a pair: the resulting list of blocks and a flag that is True only if merging occurred. A single block, or blocks that cannot consolidate, are returned unchanged with False. Otherwise the placements are concatenated, the values stacked (numpy stack or extension array concatenation along the first axis), both are reordered by sorted placement, and one new block is returned.
    if len(blocks) == 1:
        return blocks, False

    if can_consolidate:
        # TODO: optimization potential in case all mgrs contain slices and
        # combination of those slices is a slice, too.
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])

        new_values: ArrayLike

        if isinstance(blocks[0].dtype, np.dtype):
            # error: List comprehension has incompatible type List[Union[ndarray,
            # ExtensionArray]]; expected List[Union[complex, generic,
            # Sequence[Union[int, float, complex, str, bytes, generic]],
            # Sequence[Sequence[Any]], SupportsArray]]
            new_values = np.vstack([b.values for b in blocks])  # type: ignore[misc]
        else:
            bvals = [blk.values for blk in blocks]
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)

        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]

        bp = BlockPlacement(new_mgr_locs)
        return [new_block_2d(new_values, placement=bp)], True

    # can't consolidate --> no merge
    return blocks, False


def _preprocess_slice_or_indexer(
    slice_or_indexer: slice | np.ndarray, length: int, allow_fill: bool
):
    # [FUNCTION DOC]
    # HELPER: classify and normalize a slice or an indexer. Parameters: slice_or_indexer (slice or integer ndarray), length (size of the axis), allow_fill (whether -1 entries are meaningful). A slice returns the tag "slice", the slice and its resolved length. An array must be an integer ndarray (otherwise a type error is raised with the type and dtype); it is converted to the platform integer, and, if filling is not allowed, negative entries are converted (bounds checked); returns the tag "fancy", the indexer and its length.
    if isinstance(slice_or_indexer, slice):
        return (
            "slice",
            slice_or_indexer,
            libinternals.slice_len(slice_or_indexer, length),
        )
    else:
        if (
            not isinstance(slice_or_indexer, np.ndarray)
            or slice_or_indexer.dtype.kind != "i"
        ):
            dtype = getattr(slice_or_indexer, "dtype", None)
            raise TypeError(type(slice_or_indexer), dtype)

        indexer = ensure_platform_int(slice_or_indexer)
        if not allow_fill:
            indexer = maybe_convert_indices(indexer, length)
        return "fancy", indexer, len(indexer)


def make_na_array(dtype: DtypeObj, shape: Shape, fill_value) -> ArrayLike:
    # [FUNCTION DOC]
    # Create an array of missing values for a block. Parameters: dtype, shape (number of columns, number of rows) and fill_value. Timezone-aware datetime: array of the fill timestamp in the dtype's unit. One-dimensional-only extension dtypes: shape must have one column (asserted); builds an empty array and takes -1 positions with fill to produce missing entries. Other extension dtypes: an empty array of the shape filled with the value. Numpy dtypes: filled array, wrapped if datetime-like. Returns the array. Must not be called with integer or boolean numpy dtypes (the fill would produce nonsense).
    if isinstance(dtype, DatetimeTZDtype):
        # NB: exclude e.g. pyarrow[dt64tz] dtypes
        ts = Timestamp(fill_value).as_unit(dtype.unit)
        i8values = np.full(shape, ts._value)
        dt64values = i8values.view(f"M8[{dtype.unit}]")
        return DatetimeArray._simple_new(dt64values, dtype=dtype)

    elif is_1d_only_ea_dtype(dtype):
        dtype = cast(ExtensionDtype, dtype)
        cls = dtype.construct_array_type()

        missing_arr = cls._from_sequence([], dtype=dtype)
        ncols, nrows = shape
        assert ncols == 1, ncols
        empty_arr = -1 * np.ones((nrows,), dtype=np.intp)
        return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
    elif isinstance(dtype, ExtensionDtype):
        # TODO: no tests get here, a handful would if we disabled
        #  the dt64tz special-case above (which is faster)
        cls = dtype.construct_array_type()
        missing_arr = cls._empty(shape=shape, dtype=dtype)
        missing_arr[:] = fill_value
        return missing_arr
    else:
        # NB: we should never get here with dtype integer or bool;
        #  if we did, the missing_arr.fill would cast to gibberish
        missing_arr_np = np.empty(shape, dtype=dtype)
        missing_arr_np.fill(fill_value)

        if dtype.kind in "mM":
            missing_arr_np = ensure_wrapped_if_datetimelike(missing_arr_np)
        return missing_arr_np
