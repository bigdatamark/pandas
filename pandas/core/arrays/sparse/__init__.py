# [PACKAGE DOC]
# Sparse extension-array package. It groups the implementation of the sparse
# array type and its supporting pieces: the accessor module (the series level
# and frame level sparse accessors), the array module (the sparse array class,
# the two sparse index layouts and the index factory function) and the scipy
# interop module (conversion helpers between sparse series and scipy matrices,
# which is not re-exported here).
# Re-exports: the series accessor, the frame accessor, the block index, the
# integer index, the sparse array class and the sparse index factory function;
# exactly these names are listed in the public name list of the package.
# Import-time side effects: importing the submodules registers the sparse
# accessors on series and frames and registers the sparse dtype with the
# dtype registry; nothing else is executed.
from pandas.core.arrays.sparse.accessor import (
    SparseAccessor,
    SparseFrameAccessor,
)
from pandas.core.arrays.sparse.array import (
    BlockIndex,
    IntIndex,
    SparseArray,
    make_sparse_index,
)

__all__ = [
    "BlockIndex",
    "IntIndex",
    "SparseAccessor",
    "SparseArray",
    "SparseFrameAccessor",
    "make_sparse_index",
]
