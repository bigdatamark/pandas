"""
Arithmetic operations for PandasObjects

This is not a public API.
"""

# [PACKAGE DOC]
# Private package implementing the arithmetic, comparison and logical machinery shared by Series, DataFrame, Index and arrays (not a public API).
# Submodules: array_ops (array-level arithmetic, comparison and logical operations), common (decorators and result-name helpers), dispatch (extension array dispatch test), docstrings (templates for generated flex-method docs), invalid (invalid comparison and invalid operator helpers), mask_ops (three-valued logic for masked arrays), missing (division by zero conventions).
# Re-exports: the public functions of those submodules, the reversed operator functions from the reversed-operator module, and defines a set of arithmetic binary operator names as a module constant. The all-list enumerates these re-exports. No import-time side effects beyond the imports.

from __future__ import annotations

from pandas.core.ops.array_ops import (
    arithmetic_op,
    comp_method_OBJECT_ARRAY,
    comparison_op,
    fill_binop,
    get_array_op,
    logical_op,
    maybe_prepare_scalar_for_op,
)
from pandas.core.ops.common import (
    get_op_result_name,
    unpack_zerodim_and_defer,
)
from pandas.core.ops.docstrings import make_flex_doc
from pandas.core.ops.invalid import invalid_comparison
from pandas.core.ops.mask_ops import (
    kleene_and,
    kleene_or,
    kleene_xor,
)
from pandas.core.roperator import (
    radd,
    rand_,
    rdiv,
    rdivmod,
    rfloordiv,
    rmod,
    rmul,
    ror_,
    rpow,
    rsub,
    rtruediv,
    rxor,
)

# -----------------------------------------------------------------------------
# constants
ARITHMETIC_BINOPS: set[str] = {
    "add",
    "sub",
    "mul",
    "pow",
    "mod",
    "floordiv",
    "truediv",
    "divmod",
    "radd",
    "rsub",
    "rmul",
    "rpow",
    "rmod",
    "rfloordiv",
    "rtruediv",
    "rdivmod",
}


__all__ = [
    "ARITHMETIC_BINOPS",
    "arithmetic_op",
    "comp_method_OBJECT_ARRAY",
    "comparison_op",
    "fill_binop",
    "get_array_op",
    "get_op_result_name",
    "invalid_comparison",
    "kleene_and",
    "kleene_or",
    "kleene_xor",
    "logical_op",
    "make_flex_doc",
    "maybe_prepare_scalar_for_op",
    "radd",
    "rand_",
    "rdiv",
    "rdivmod",
    "rfloordiv",
    "rmod",
    "rmul",
    "ror_",
    "rpow",
    "rsub",
    "rtruediv",
    "rxor",
    "unpack_zerodim_and_defer",
]
