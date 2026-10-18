"""
algos/ directory is intended for individual functions from core.algorithms

In many cases these algorithms are reachable in multiple ways:
   algos.foo(x, y)
   Series(x).foo(y)
   Index(x).foo(y)
   pd.array(x).foo(y)

In most cases we profile the Series variant directly, trusting the performance
of the others to be highly correlated.
"""

# [PACKAGE DOC]
# Subpackage of the benchmark suite holding benchmarks for individual algorithm
# functions that live in the core algorithms module of the library. Currently its
# only submodule is isin, which measures membership testing. The package re-exports
# nothing and has no import-time side effects.
