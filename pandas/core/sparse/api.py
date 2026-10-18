# [MODULE DOC]
# Public re-export module for sparse data support.
# Re-exports SparseDtype (from the core dtypes dtypes module) and SparseArray (from
# the core arrays sparse package) and lists exactly those two in the public name
# list. It defines no logic of its own; importing it only triggers those imports.
from pandas.core.dtypes.dtypes import SparseDtype

from pandas.core.arrays.sparse import SparseArray

__all__ = ["SparseArray", "SparseDtype"]
