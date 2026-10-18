"""
Methods that can be shared by many array-like classes or subclasses:
    Series
    Index
    ExtensionArray
"""

# [MODULE DOC]
# Shared behavior for array-like pandas classes (Series, Index, extension arrays,
# and for ufunc handling also DataFrame).
# Public contents: REDUCTION_ALIASES (mapping from numpy ufunc names to the
# equivalent reduction method names), OpsMixin (defines all comparison, logical and
# arithmetic dunder methods by forwarding to three hook methods that subclasses
# implement), and the helpers behind the numpy ufunc protocol: array_ufunc (the
# main implementation of the protocol for pandas objects), dispatch_ufunc_with_out,
# default_array_ufunc, dispatch_reduction_ufunc and two private helpers.
# Relationships: uses the zero-dim unpacking decorator from the ops package, the
# reflected operator functions from the roperator module and construction helpers.
# Import-time side effects: none.
from __future__ import annotations

import operator
from typing import Any

import numpy as np

from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

from pandas.core.dtypes.cast import maybe_unbox_numpy_scalar
from pandas.core.dtypes.generic import ABCNDFrame

from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer

REDUCTION_ALIASES = {
    "maximum": "max",
    "minimum": "min",
    "add": "sum",
    "multiply": "prod",
}


class OpsMixin:
    # [CLASS DOC]
    # Mixin providing the comparison (equality, ordering), logical (and, or,
    # exclusive or) and arithmetic (add, subtract, multiply, true and floor
    # division, modulo, divmod, power, plus their reflected forms) dunder methods.
    # Every public dunder is decorated so that zero-dimensional arrays are unpacked
    # to scalars and the operation returns NotImplemented (deferring to the other
    # operand) when the other operand is a higher-priority pandas type; it then
    # forwards to one of three hook methods with the other operand and the matching
    # operator function. The hooks (comparison, logical, arithmetic) are meant to be
    # overridden by subclasses; the defaults return NotImplemented. Holds no state.
    # -------------------------------------------------------------
    # Comparisons

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: hook for comparison operators. Inputs: other operand and op, the
        # operator function. The default returns the not-implemented sentinel so
        # Python tries the reflected operation; subclasses override it.
        return NotImplemented

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality comparison; forwards to the comparison hook with the equality
        # operator and returns its result (typically an elementwise boolean object).
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        # [FUNCTION DOC]
        # Inequality comparison; forwards to the comparison hook with the
        # not-equal operator.
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        # [FUNCTION DOC]
        # Less-than comparison; forwards to the comparison hook.
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        # [FUNCTION DOC]
        # Less-than-or-equal comparison; forwards to the comparison hook.
        return self._cmp_method(other, operator.le)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        # [FUNCTION DOC]
        # Greater-than comparison; forwards to the comparison hook.
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        # [FUNCTION DOC]
        # Greater-than-or-equal comparison; forwards to the comparison hook.
        return self._cmp_method(other, operator.ge)

    # -------------------------------------------------------------
    # Logical Methods

    def _logical_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: hook for logical operators. Inputs: other operand and op, the
        # operator function. The default returns the not-implemented sentinel;
        # subclasses override it.
        return NotImplemented

    @unpack_zerodim_and_defer("__and__")
    def __and__(self, other):
        # [FUNCTION DOC]
        # Bitwise/logical and; forwards to the logical hook.
        return self._logical_method(other, operator.and_)

    @unpack_zerodim_and_defer("__rand__")
    def __rand__(self, other):
        # [FUNCTION DOC]
        # Reflected logical and (operands swapped); forwards to the logical hook
        # with the reflected operator function.
        return self._logical_method(other, roperator.rand_)

    @unpack_zerodim_and_defer("__or__")
    def __or__(self, other):
        # [FUNCTION DOC]
        # Bitwise/logical or; forwards to the logical hook.
        return self._logical_method(other, operator.or_)

    @unpack_zerodim_and_defer("__ror__")
    def __ror__(self, other):
        # [FUNCTION DOC]
        # Reflected logical or; forwards to the logical hook with the reflected
        # operator function.
        return self._logical_method(other, roperator.ror_)

    @unpack_zerodim_and_defer("__xor__")
    def __xor__(self, other):
        # [FUNCTION DOC]
        # Bitwise/logical exclusive or; forwards to the logical hook.
        return self._logical_method(other, operator.xor)

    @unpack_zerodim_and_defer("__rxor__")
    def __rxor__(self, other):
        # [FUNCTION DOC]
        # Reflected logical exclusive or; forwards to the logical hook with the
        # reflected operator function.
        return self._logical_method(other, roperator.rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: hook for arithmetic operators. Inputs: other operand and op, the
        # operator function. The default returns the not-implemented sentinel;
        # subclasses override it.
        return NotImplemented

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        # [FUNCTION DOC]
        # Addition; forwards to the arithmetic hook with the addition operator and
        # returns its result.
        # KNOWN ISSUE: the docstring below describes DataFrame-only behavior
        # (column-wise addition with DataFrame examples), but this method lives on
        # a mixin shared by Series, Index and extension arrays, and the code is
        # generic.
        """
        Get Addition of DataFrame and other, column-wise.

        Equivalent to ``DataFrame.add(other)``.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Object to be added to the DataFrame.

        Returns
        -------
        DataFrame
            The result of adding ``other`` to DataFrame.

        See Also
        --------
        DataFrame.add : Add a DataFrame and another object, with option for index-
            or column-oriented addition.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"height": [1.5, 2.6], "weight": [500, 800]}, index=["elk", "moose"]
        ... )
        >>> df
               height  weight
        elk       1.5     500
        moose     2.6     800

        Adding a scalar affects all rows and columns.

        >>> df[["height", "weight"]] + 1.5
               height  weight
        elk       3.0   501.5
        moose     4.1   801.5

        Each element of a list is added to a column of the DataFrame, in order.

        >>> df[["height", "weight"]] + [0.5, 1.5]
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        Keys of a dictionary are aligned to the DataFrame, based on column names;
        each value in the dictionary is added to the corresponding column.

        >>> df[["height", "weight"]] + {"height": 0.5, "weight": 1.5}
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        When `other` is a :class:`Series`, the index of `other` is aligned with the
        columns of the DataFrame.

        >>> s1 = pd.Series([0.5, 1.5], index=["weight", "height"])
        >>> df[["height", "weight"]] + s1
               height  weight
        elk       3.0   500.5
        moose     4.1   800.5

        Even when the index of `other` is the same as the index of the DataFrame,
        the :class:`Series` will not be reoriented. If index-wise alignment is desired,
        :meth:`DataFrame.add` should be used with `axis='index'`.

        >>> s2 = pd.Series([0.5, 1.5], index=["elk", "moose"])
        >>> df[["height", "weight"]] + s2
               elk  height  moose  weight
        elk    NaN     NaN    NaN     NaN
        moose  NaN     NaN    NaN     NaN

        >>> df[["height", "weight"]].add(s2, axis="index")
               height  weight
        elk       2.0   500.5
        moose     4.1   801.5

        When `other` is a :class:`DataFrame`, both columns names and the
        index are aligned.

        >>> other = pd.DataFrame(
        ...     {"height": [0.2, 0.4, 0.6]}, index=["elk", "moose", "deer"]
        ... )
        >>> df[["height", "weight"]] + other
               height  weight
        deer      NaN     NaN
        elk       1.7     NaN
        moose     3.0     NaN
        """
        return self._arith_method(other, operator.add)

    @unpack_zerodim_and_defer("__radd__")
    def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; forwards to the arithmetic hook with the reflected
        # operator function.
        return self._arith_method(other, roperator.radd)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        # [FUNCTION DOC]
        # Subtraction; forwards to the arithmetic hook.
        return self._arith_method(other, operator.sub)

    @unpack_zerodim_and_defer("__rsub__")
    def __rsub__(self, other):
        # [FUNCTION DOC]
        # Reflected subtraction; forwards to the arithmetic hook with the reflected
        # operator function.
        return self._arith_method(other, roperator.rsub)

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other):
        # [FUNCTION DOC]
        # Multiplication; forwards to the arithmetic hook.
        return self._arith_method(other, operator.mul)

    @unpack_zerodim_and_defer("__rmul__")
    def __rmul__(self, other):
        # [FUNCTION DOC]
        # Reflected multiplication; forwards to the arithmetic hook with the
        # reflected operator function.
        return self._arith_method(other, roperator.rmul)

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        # [FUNCTION DOC]
        # True (floating) division; forwards to the arithmetic hook.
        return self._arith_method(other, operator.truediv)

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        # [FUNCTION DOC]
        # Reflected true division; forwards to the arithmetic hook with the
        # reflected operator function.
        return self._arith_method(other, roperator.rtruediv)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        # [FUNCTION DOC]
        # Floor division; forwards to the arithmetic hook.
        return self._arith_method(other, operator.floordiv)

    @unpack_zerodim_and_defer("__rfloordiv")
    def __rfloordiv__(self, other):
        # [FUNCTION DOC]
        # Reflected floor division; forwards to the arithmetic hook with the
        # reflected operator function.
        # KNOWN ISSUE: the name passed to the decorator is missing its trailing
        # underscores (it reads as the name without them), unlike every other
        # method here; the decorator uses that name to look up the method to defer
        # to on higher-priority operands, so the lookup may use the wrong name.
        return self._arith_method(other, roperator.rfloordiv)

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        # [FUNCTION DOC]
        # Modulo (remainder); forwards to the arithmetic hook.
        return self._arith_method(other, operator.mod)

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        # [FUNCTION DOC]
        # Reflected modulo; forwards to the arithmetic hook with the reflected
        # operator function.
        return self._arith_method(other, roperator.rmod)

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        # [FUNCTION DOC]
        # Combined floor division and remainder; forwards to the arithmetic hook
        # with the builtin divmod function. The result is a pair.
        return self._arith_method(other, divmod)

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        # [FUNCTION DOC]
        # Reflected divmod; forwards to the arithmetic hook with the reflected
        # operator function.
        return self._arith_method(other, roperator.rdivmod)

    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other):
        # [FUNCTION DOC]
        # Exponentiation; forwards to the arithmetic hook.
        return self._arith_method(other, operator.pow)

    @unpack_zerodim_and_defer("__rpow__")
    def __rpow__(self, other):
        # [FUNCTION DOC]
        # Reflected exponentiation; forwards to the arithmetic hook with the
        # reflected operator function.
        return self._arith_method(other, roperator.rpow)


# -----------------------------------------------------------------------------
# Helpers to implement __array_ufunc__


def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    # [FUNCTION DOC]
    # Implements the numpy ufunc protocol for Series and DataFrame (and objects
    # with the same axis attributes). Inputs: self (the pandas object), ufunc (the
    # numpy universal function), method (ufunc method name: call, reduce, outer,
    # accumulate, and so on), positional inputs (operands, normally including self)
    # and keyword arguments passed to the ufunc.
    # Behavior, in order: (1) normalizes separate first/second output keywords into
    # a single output tuple; (2) tries to dispatch binary operators to the matching
    # dunder method and returns its result if it handled it; (3) returns the
    # not-implemented sentinel if any other input has a higher array priority or
    # defines its own ufunc handling that is not numpy's default (and is not one of
    # the types self handles); (4) if more than one input is a pandas object, aligns
    # them by taking the union of their axes and reindexing (raises a
    # not-implemented error for a mix of DataFrame and Series); (5) with an output
    # keyword, computes the ufunc and writes into the output, then reconstructs;
    # (6) for the reduce method tries to call the object's own reduction method;
    # (7) otherwise computes: for multi-dimensional input with several operands or
    # outputs, on plain arrays; for one-dimensional input, on the unwrapped
    # arrays; for a frame with a plain call and no keywords, block by block; else
    # through the default fallback.
    # The result is rebuilt into the pandas type with the aligned axes (names
    # propagated for one-dimensional objects, metadata copied when only one input
    # is a pandas object); scalars are returned as is; results with a different
    # dimensionality are returned unwrapped (outer raises not-implemented).
    # Multiple outputs are returned as a tuple.
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    from pandas.core.frame import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.internals import BlockManager

    cls = type(self)

    kwargs = _standardize_out_kwarg(**kwargs)

    # for binary ops, use our custom dunder methods
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result

    # Determine if we should defer.
    no_defer = (
        np.ndarray.__array_ufunc__,
        cls.__array_ufunc__,
    )

    for item in inputs:
        higher_priority = (
            hasattr(item, "__array_priority__")
            and item.__array_priority__ > self.__array_priority__
        )
        has_array_ufunc = (
            hasattr(item, "__array_ufunc__")
            and type(item).__array_ufunc__ not in no_defer
            and not isinstance(item, self._HANDLED_TYPES)
        )
        if higher_priority or has_array_ufunc:
            return NotImplemented

    # align all the inputs.
    types = tuple(type(x) for x in inputs)
    alignable = [
        x for x, t in zip(inputs, types, strict=True) if issubclass(t, NDFrame)
    ]

    if len(alignable) > 1:
        # This triggers alignment.
        # At the moment, there aren't any ufuncs with more than two inputs
        # so this ends up just being x1.index | x2.index, but we write
        # it to handle *args.
        set_types = set(types)
        if len(set_types) > 1 and {DataFrame, Series}.issubset(set_types):
            # We currently don't handle ufunc(DataFrame, Series)
            # well. Previously this raised an internal ValueError. We might
            # support it someday, so raise a NotImplementedError.
            raise NotImplementedError(
                f"Cannot apply ufunc {ufunc} to mixed DataFrame and Series inputs."
            )
        axes = self.axes
        for obj in alignable[1:]:
            # this relies on the fact that we aren't handling mixed
            # series / frame ufuncs.
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes, strict=True)):
                axes[i] = ax1.union(ax2)

        reconstruct_axes = dict(zip(self._AXIS_ORDERS, axes, strict=True))
        inputs = tuple(
            x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x
            for x, t in zip(inputs, types, strict=True)
        )
    else:
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes, strict=True))

    if self.ndim == 1:
        names = {x.name for x in inputs if hasattr(x, "name")}
        name = names.pop() if len(names) == 1 else None
        reconstruct_kwargs = {"name": name}
    else:
        reconstruct_kwargs = {}

    def reconstruct(result):
        # [FUNCTION DOC]
        # HELPER: converts the raw ufunc result back to pandas objects. For a
        # ufunc with several outputs, converts each element of the result and
        # returns a tuple; otherwise converts the single result.
        if ufunc.nout > 1:
            # np.modf, np.frexp, np.divmod
            return tuple(_reconstruct(x) for x in result)

        return _reconstruct(result)

    def _reconstruct(result):
        # [FUNCTION DOC]
        # HELPER: converts one raw result. Scalars are returned unchanged. A result
        # whose dimensionality differs from self is returned unwrapped (raising a
        # not-implemented error for the outer method). A block manager result is
        # wrapped using the constructor from manager with its own axes; any other
        # array is wrapped with the aligned axes and (for one-dimensional objects)
        # the common name, without copying. If only one input was a pandas object,
        # metadata is propagated from self. Returns the wrapped result.
        if lib.is_scalar(result):
            return result

        if result.ndim != self.ndim:
            if method == "outer":
                raise NotImplementedError
            return result
        if isinstance(result, BlockManager):
            # we went through BlockManager.apply e.g. np.sqrt
            result = self._constructor_from_mgr(result, axes=result.axes)
        else:
            # we converted an array, lost our axes
            result = self._constructor(
                result, **reconstruct_axes, **reconstruct_kwargs, copy=False
            )
        # TODO: When we support multiple values in __finalize__, this
        # should pass alignable to `__finalize__` instead of self.
        # Then `np.add(a, b)` would consider attrs from both a and b
        # when a and b are NDFrames.
        if len(alignable) == 1:
            result = result.__finalize__(self)
        return result

    if "out" in kwargs:
        # e.g. test_multiindex_get_loc
        result = dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        return reconstruct(result)

    if method == "reduce":
        # e.g. test.series.test_ufunc.test_reduce
        result = dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

    # We still get here with kwargs `axis` for e.g. np.maximum.accumulate
    #  and `dtype` and `keepdims` for np.ptp

    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        # Just give up on preserving types in the complex case.
        # In theory we could preserve them for them.
        # * nout>1 is doable if BlockManager.apply took nout and
        #   returned a Tuple[BlockManager].
        # * len(inputs) > 1 is doable when we know that we have
        #   aligned blocks / dtypes.

        # e.g. my_ufunc, modf, logaddexp, heaviside, subtract, add
        inputs = tuple(np.asarray(x) for x in inputs)
        # Note: we can't use default_array_ufunc here bc reindexing means
        #  that `self` may not be among `inputs`
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        # ufunc(series, ...)
        inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
    # ufunc(dataframe)
    elif method == "__call__" and not kwargs:
        # for np.<ufunc>(..) calls
        # kwargs cannot necessarily be handled block-by-block, so only
        # take this path if there are no kwargs
        mgr = inputs[0]._mgr  # pyright: ignore[reportGeneralTypeIssues]
        result = mgr.apply(getattr(ufunc, method))
    else:
        # otherwise specific ufunc methods (eg np.<ufunc>.accumulate(..))
        # Those can have an axis keyword and thus can't be called block-by-block
        result = default_array_ufunc(inputs[0], ufunc, method, *inputs, **kwargs)  # pyright: ignore[reportGeneralTypeIssues]
        # e.g. np.negative (only one reached), with "where" and "out" in kwargs

    result = reconstruct(result)
    return result


def _standardize_out_kwarg(**kwargs) -> dict:
    # [FUNCTION DOC]
    # HELPER: normalizes output keywords for the ufunc protocol. Input: keyword
    # arguments. If there is no combined output keyword but both a first and second
    # output keyword are present, removes them and adds a combined output keyword
    # holding the pair. Returns the (possibly modified) keyword dictionary.
    """
    If kwargs contain "out1" and "out2", replace that with a tuple "out"

    np.divmod, np.modf, np.frexp can have either `out=(out1, out2)` or
    `out1=out1, out2=out2)`
    """
    if "out" not in kwargs and "out1" in kwargs and "out2" in kwargs:
        out1 = kwargs.pop("out1")
        out2 = kwargs.pop("out2")
        out = (out1, out2)
        kwargs["out"] = out
    return kwargs


def dispatch_ufunc_with_out(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    # [FUNCTION DOC]
    # Runs a ufunc that was given an output target by computing it without the
    # target and then copying the result into the target. Inputs: self (unused
    # except for dispatch), ufunc, method, positional inputs and keywords which must
    # contain the output keyword (already normalized; a missing one raises a key
    # error) and may contain a where mask.
    # Behavior: the output (and where) keywords are removed from the keywords (the
    # dictionary passed in is a fresh copy so the caller's is unaffected); the ufunc
    # method is called; a not-implemented result is passed through. For a tuple
    # result the output must be a tuple of equal length (else not-implemented error)
    # and each piece is assigned. A single result with a one-element tuple as output
    # uses its element (longer tuples raise not-implemented). Side effect: mutates
    # the output arrays. Returns the output (tuple or single).
    """
    If we have an `out` keyword, then call the ufunc without `out` and then
    set the result into the given `out`.
    """

    # Note: we assume _standardize_out_kwarg has already been called.
    out = kwargs.pop("out")
    where = kwargs.pop("where", None)

    result = getattr(ufunc, method)(*inputs, **kwargs)

    if result is NotImplemented:
        return NotImplemented

    if isinstance(result, tuple):
        # i.e. np.divmod, np.modf, np.frexp
        if not isinstance(out, tuple) or len(out) != len(result):
            raise NotImplementedError

        for arr, res in zip(out, result, strict=True):
            _assign_where(arr, res, where)

        return out

    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            raise NotImplementedError

    _assign_where(out, result, where)
    return out


def _assign_where(out, result, where) -> None:
    # [FUNCTION DOC]
    # HELPER: writes a ufunc result into the output array in place. Inputs: out
    # (target array), result and where (boolean mask or none). With no mask the
    # whole output is overwritten; with a mask only masked-in positions are assigned
    # (numpy put-mask semantics). Returns nothing.
    """
    Set a ufunc result into 'out', masking with a 'where' argument if necessary.
    """
    if where is None:
        # no 'where' arg passed to ufunc
        out[:] = result
    else:
        np.putmask(out, where, result)


def default_array_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    # [FUNCTION DOC]
    # Fallback that runs the ufunc as numpy would without a custom protocol.
    # Inputs: self, ufunc, method, positional inputs and keywords. self must be
    # among the inputs (identity check) or a not-implemented error is raised. The
    # occurrences of self are converted to plain numpy arrays and the ufunc method
    # is called with the same keywords. Returns the raw numpy result.
    """
    Fallback to the behavior we would get if we did not define __array_ufunc__.

    Notes
    -----
    We are assuming that `self` is among `inputs`.
    """
    if not any(x is self for x in inputs):
        raise NotImplementedError

    new_inputs = [x if x is not self else np.asarray(x) for x in inputs]

    return getattr(ufunc, method)(*new_inputs, **kwargs)


def dispatch_reduction_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    # [FUNCTION DOC]
    # Maps a numpy ufunc reduction onto the pandas reduction method of the same
    # meaning. Inputs: self, ufunc, method (must be the reduce method; an assertion
    # error otherwise), positional inputs and keywords.
    # Returns the not-implemented sentinel when there is not exactly one input, the
    # input is not self, the ufunc has no alias in the alias table (maximum, minimum,
    # add, multiply), or self lacks the corresponding method. For
    # multi-dimensional objects, frames get numeric_only set to false, and if no axis
    # was given the axis is forced to zero to match numpy. The reduction method is
    # then called with skipna false (numpy does not skip missing values) and the
    # remaining keywords; the result has any numpy scalar unwrapped to a Python
    # scalar and is returned.
    """
    Dispatch ufunc reductions to self's reduction methods.
    """
    assert method == "reduce"

    if len(inputs) != 1 or inputs[0] is not self:
        return NotImplemented

    if ufunc.__name__ not in REDUCTION_ALIASES:
        return NotImplemented

    method_name = REDUCTION_ALIASES[ufunc.__name__]

    # NB: we are assuming that min/max represent minimum/maximum methods,
    #  which would not be accurate for e.g. Timestamp.min
    if not hasattr(self, method_name):
        return NotImplemented

    if self.ndim > 1:
        if isinstance(self, ABCNDFrame):
            # TODO: test cases where this doesn't hold, i.e. 2D DTA/TDA
            kwargs["numeric_only"] = False

        if "axis" not in kwargs:
            # For DataFrame reductions we don't want the default axis=0
            # Note: np.min is not a ufunc, but uses array_function_dispatch,
            #  so calls DataFrame.min (without ever getting here) with the np.min
            #  default of axis=None, which DataFrame.min catches and changes to axis=0.
            # np.minimum.reduce(df) gets here bc axis is not in kwargs,
            #  so we set axis=0 to match the behavior of np.minimum.reduce(df.values)
            kwargs["axis"] = 0

    # By default, numpy's reductions do not skip NaNs, so we have to
    #  pass skipna=False
    result = getattr(self, method_name)(skipna=False, **kwargs)
    result = maybe_unbox_numpy_scalar(result)
    return result
