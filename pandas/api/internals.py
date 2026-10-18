# [MODULE DOC]
# Public module exposing one advanced, low-level function,
# create_dataframe_from_blocks, which builds a frame directly from block
# arrays. Relationships: uses the internal block factory and the block manager
# from the core internals package, and the frame's private constructor from a
# manager. Not exported through the api package namespace. No import-time side
# effects beyond imports.
import numpy as np

from pandas._typing import ArrayLike

from pandas import (
    DataFrame,
    Index,
)
from pandas.core.internals.api import _make_block
from pandas.core.internals.managers import BlockManager as _BlockManager


def create_dataframe_from_blocks(
    blocks: list[tuple[ArrayLike, np.ndarray]], index: Index, columns: Index
) -> DataFrame:
    # [FUNCTION DOC]
    # Builds a frame from raw blocks without any validation. Parameters:
    # blocks (list of pairs of a block array and a placement array; the block
    # array is a two-dimensional numpy array already transposed (columns by
    # rows) or an extension array, and the placement array is a one-dimensional
    # integer array of the column positions the block holds), index (Index for
    # the rows) and columns (Index for the columns).
    # Each pair is converted into an internal block object, a block manager is
    # built from the blocks and the axes (columns first, then index), and a frame
    # is created from that manager. Returns the frame. Nothing is type-inferred,
    # copied or checked: violating the documented assumptions (shape, complete
    # placement coverage) produces a corrupt frame that fails later.
    # KNOWN ISSUE: the existing docstring has a repeated word ("subsequent
    # subsequent") and says the block array may be an extension array in one
    # place and one-dimensional extension array in another; the code does not
    # check either.
    """
    Low-level function to create a DataFrame from arrays as they are
    representing the block structure of the resulting DataFrame.

    Attention: this is an advanced, low-level function that should only be
    used if you know that the below-mentioned assumptions are guaranteed.
    If passing data that do not follow those assumptions, subsequent
    subsequent operations on the resulting DataFrame might lead to strange
    errors.
    For almost all use cases, you should use the standard pd.DataFrame(..)
    constructor instead. If you are planning to use this function, let us
    know by opening an issue at https://github.com/pandas-dev/pandas/issues.

    Assumptions:

    - The block arrays are either a 2D numpy array or a pandas ExtensionArray
    - In case of a numpy array, it is assumed to already be in the expected
      shape for Blocks (2D, (cols, rows), i.e. transposed compared to the
      DataFrame columns).
    - All arrays are taken as is (no type inference) and expected to have the
      correct size.
    - The placement arrays have the correct length (equalling the number of
      columns that its equivalent block array represents), and all placement
      arrays together form a complete set of 0 to n_columns - 1.

    Parameters
    ----------
    blocks : list of tuples of (block_array, block_placement)
        This should be a list of tuples existing of (block_array, block_placement),
        where:

        - block_array is a 2D numpy array or a 1D ExtensionArray, following the
          requirements listed above.
        - block_placement is a 1D integer numpy array
    index : Index
        The Index object for the `index` of the resulting DataFrame.
    columns : Index
        The Index object for the `columns` of the resulting DataFrame.

    Returns
    -------
    DataFrame
    """
    block_objs = [_make_block(*block) for block in blocks]
    axes = [columns, index]
    mgr = _BlockManager(block_objs, axes)
    return DataFrame._from_mgr(mgr, mgr.axes)
