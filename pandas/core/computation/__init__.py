# [PACKAGE DOC]
# Package holding the expression-evaluation machinery behind the top-level
# evaluation function, the query method and the expression-accelerated
# arithmetic helpers.
# Submodules: align (operand alignment), api (public re-export), check
# (optional numexpr detection), common (small shared helpers), engines
# (numexpr and pure-Python evaluation engines), eval (public entry point),
# expr (parser producing an expression tree), expressions (accelerated
# element-wise operations used by the arithmetic layer), ops (expression
# tree node classes), parsing (tokenizer helpers for backtick-quoted names),
# pytables (expression support for the HDF5 table store) and scope
# (variable resolution scopes).
# This package module itself exports nothing and has no import-time side
# effects; each submodule is imported on demand.
