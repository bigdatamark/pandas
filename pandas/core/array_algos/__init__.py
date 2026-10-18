"""
core.array_algos is for algorithms that operate on ndarray and ExtensionArray.
These should:

- Assume that any Index, Series, or DataFrame objects have already been unwrapped.
- Assume that any list arguments have already been cast to ndarray/EA.
- Not depend on Index, Series, or DataFrame, nor import any of these.
- May dispatch to ExtensionArray methods, but should not import from core.arrays.
"""

# [PACKAGE DOC]
# Package of low-level algorithms that operate on raw numpy arrays and extension
# arrays after the callers have already unwrapped Index/Series/DataFrame objects
# and converted list inputs. Submodules: accumulations for datetime-like and
# masked arrays, masked reductions, in-place mask-based assignment, quantile
# computation, value replacement, take (positional gather) with fill handling and
# shift-like transforms. The package initializer defines and exports nothing and
# has no import-time side effects; the module layering rule is that these modules
# must not import the container classes or the arrays package.
