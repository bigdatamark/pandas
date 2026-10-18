"""
Core eval alignment algorithms.
"""

# [MODULE DOC]
# Alignment step of expression evaluation: given the terms of a parsed
# expression, work out a common result container type and a common set of
# axes, reindex pandas operands onto those axes, and later rebuild the raw
# evaluation result into that container type.
# Public contents (used by the engines module): align_terms and
# reconstruct_object. The remaining functions are private helpers.
# Depends on the common module for dtype promotion and the pandas base class.
# Side effects: alignment mutates the terms in place (their stored values are
# replaced by reindexed raw arrays) and may emit a performance warning.

from __future__ import annotations

from functools import (
    partial,
    wraps,
)
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._config.config import get_option

from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation.common import result_type_many

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from pandas._typing import F

    from pandas.core.generic import NDFrame
    from pandas.core.indexes.api import Index


def _align_core_single_unary_op(
    term,
) -> tuple[partial | type[NDFrame], dict[str, Index] | None]:
    # [FUNCTION DOC]
    # HELPER: alignment of a lone operand, used by the special-case filter.
    # Input term: a term object whose stored value is either a raw array or a
    # pandas-like object. Returns a pair (container constructor, axes mapping).
    # For a raw array the constructor is a partial of the array-conversion
    # function bound to the array's dtype and the axes are empty. Otherwise the
    # constructor is the value's own type and, if the value has axes, the axes
    # mapping pairs each axis name of that type with the value's axis in order.
    typ: partial | type[NDFrame]
    axes: dict[str, Index] | None = None

    if isinstance(term.value, np.ndarray):
        typ = partial(np.asanyarray, dtype=term.value.dtype)
    else:
        typ = type(term.value)
        if hasattr(term.value, "axes"):
            axes = _zip_axes_from_type(typ, term.value.axes)

    return typ, axes


def _zip_axes_from_type(
    typ: type[NDFrame], new_axes: Sequence[Index]
) -> dict[str, Index]:
    # [FUNCTION DOC]
    # HELPER: builds a mapping from axis name to axis index object.
    # Inputs: typ, a container class exposing its ordered axis names; new_axes,
    # a sequence of index objects in the same order. Returns a new dict pairing
    # each axis name with the index at the same position.
    return {name: new_axes[i] for i, name in enumerate(typ._AXIS_ORDERS)}


def _any_pandas_objects(terms) -> bool:
    # [FUNCTION DOC]
    # HELPER: reports whether any term's value is a pandas object.
    # Input terms: iterable of term objects. Returns True when at least one
    # stored value is an instance of the pandas base class, else False.
    """
    Check a sequence of terms for instances of PandasObject.
    """
    return any(isinstance(term.value, PandasObject) for term in terms)


def _filter_special_cases(f) -> Callable[[F], F]:
    # [FUNCTION DOC]
    # HELPER: decorator for the core alignment function that short-circuits
    # trivial inputs. Input f: the full alignment function taking a list of
    # terms. Returns a wrapped function with the same name and metadata.
    # The wrapper handles two cases itself: a single term goes to the unary
    # helper; terms with no pandas objects return the promoted common dtype of
    # all values and no axes. Otherwise it delegates to f.
    @wraps(f)
    def wrapper(terms):
        # [FUNCTION DOC]
        # HELPER: the decorated entry point. Input terms: list of term
        # objects. Returns a pair (container type or dtype, axes mapping or
        # none) as described for the decorator.
        # single unary operand
        if len(terms) == 1:
            return _align_core_single_unary_op(terms[0])

        term_values = (term.value for term in terms)

        # we don't have any pandas objects
        if not _any_pandas_objects(terms):
            return result_type_many(*term_values), None

        return f(terms)

    return wrapper


@_filter_special_cases
def _align_core(terms):
    # [FUNCTION DOC]
    # HELPER: the main alignment routine, applied only when at least two
    # terms exist and at least one holds a pandas object (the special-case
    # decorator handles other inputs).
    # Input terms: list of term objects (mutated in place).
    # Steps: pick the operand with the most dimensions as the template for the
    # result container type and initial axes; widen each axis to the union of
    # all operands' matching axes (a one-dimensional series is matched to the
    # last axis when the template has several axes); then reindex every pandas
    # operand to the final axes and replace each term's value by the raw
    # underlying array. Emits a performance warning when the performance
    # warnings option is enabled, a reindex grows an axis by an order of
    # magnitude or more, and the new axis has at least 10000 entries.
    # Returns (container type, axes mapping by axis name).
    term_index = [i for i, term in enumerate(terms) if hasattr(term.value, "axes")]
    term_dims = [terms[i].value.ndim for i in term_index]

    from pandas import Series

    ndims = Series(dict(zip(term_index, term_dims, strict=True)))

    # initial axes are the axes of the largest-axis'd term
    biggest = terms[ndims.idxmax()].value
    typ = biggest._constructor
    axes = biggest.axes
    naxes = len(axes)
    gt_than_one_axis = naxes > 1

    for value in (terms[i].value for i in term_index):
        is_series = isinstance(value, ABCSeries)
        is_series_and_gt_one_axis = is_series and gt_than_one_axis

        for axis, items in enumerate(value.axes):
            if is_series_and_gt_one_axis:
                ax, itm = naxes - 1, value.index
            else:
                ax, itm = axis, items

            if not axes[ax].is_(itm):
                axes[ax] = axes[ax].union(itm)

    for i, ndim in ndims.items():
        for axis, items in zip(range(ndim), axes, strict=False):
            ti = terms[i].value

            if hasattr(ti, "reindex"):
                transpose = isinstance(ti, ABCSeries) and naxes > 1
                reindexer = axes[naxes - 1] if transpose else items

                term_axis_size = len(ti.axes[axis])
                reindexer_size = len(reindexer)

                ordm = np.log10(max(1, abs(reindexer_size - term_axis_size)))
                if (
                    get_option("performance_warnings")
                    and ordm >= 1
                    and reindexer_size >= 10000
                ):
                    w = (
                        f"Alignment difference on axis {axis} is larger "
                        f"than an order of magnitude on term {terms[i].name!r}, "
                        f"by more than {ordm:.4g}; performance may suffer."
                    )
                    warnings.warn(
                        w, category=PerformanceWarning, stacklevel=find_stack_level()
                    )

                obj = ti.reindex(reindexer, axis=axis)
                terms[i].update(obj)

        terms[i].update(terms[i].value.values)

    return typ, _zip_axes_from_type(typ, axes)


def align_terms(terms):
    # [FUNCTION DOC]
    # Aligns the operands of a parsed expression and determines the type of
    # the result.
    # Input terms: a nested list of term objects (the parse tree) or, when it
    # cannot be iterated, a single term (a lone constant or variable).
    # Returns a triple (result type, axes mapping, result name):
    # for a single non-iterable term holding a series or frame, its type, its
    # axes and the series name (none for frames); for any other lone term, the
    # promoted numpy type of the term's declared type with no axes or name.
    # When all flattened terms are scalars, the promoted scalar type with no
    # axes or name. Otherwise the name is the shared name when all series
    # operands agree on exactly one name (else none) and type and axes come
    # from the core alignment routine. Terms are mutated by that routine.
    """
    Align a set of terms.
    """
    try:
        # flatten the parse tree (a nested list, really)
        terms = list(com.flatten(terms))
    except TypeError:
        # can't iterate so it must just be a constant or single variable
        if isinstance(terms.value, (ABCSeries, ABCDataFrame)):
            typ = type(terms.value)
            name = terms.value.name if isinstance(terms.value, ABCSeries) else None
            return typ, _zip_axes_from_type(typ, terms.value.axes), name
        return np.result_type(terms.type), None, None

    # if all resolved variables are numeric scalars
    if all(term.is_scalar for term in terms):
        return result_type_many(*(term.value for term in terms)).type, None, None

    # if all input series have a common name, propagate it to the returned series
    names = {term.value.name for term in terms if isinstance(term.value, ABCSeries)}
    name = names.pop() if len(names) == 1 else None

    # perform the main alignment
    typ, axes = _align_core(terms)
    return typ, axes, name


def reconstruct_object(typ, obj, axes, dtype, name):
    # [FUNCTION DOC]
    # Wraps a raw evaluation result back into the container type decided
    # during alignment.
    # Inputs: typ, a pandas class, a partial constructor, or a numpy scalar
    # type (a wrapper exposing an inner type attribute is unwrapped first);
    # obj, the raw numpy result; axes, mapping of axis name to index or none;
    # dtype, the dtype the expression should be at least promoted to; name,
    # the series name or none.
    # The working dtype is the numpy promotion of the result's dtype and dtype.
    # Pandas classes are constructed from obj with that dtype, the axes and the
    # name when given. Otherwise the result is converted to a numpy scalar of
    # the working dtype (the boolean special case, e.g. logical negation of a
    # boolean constant, takes the same conversion). For a one-element
    # one-dimensional input that did not yield an array, the scalar is
    # rewrapped as a one-element array of the working dtype so the shape is
    # preserved. The boolean special case (typ is the numpy boolean type but
    # the working dtype is not) skips this one-element rewrapping.
    """
    Reconstruct an object given its type, raw value, and possibly empty
    (None) axes.

    Parameters
    ----------
    typ : object
        A type
    obj : object
        The value to use in the type constructor
    axes : dict
        The axes to use to construct the resulting pandas object

    Returns
    -------
    ret : typ
        An object of type ``typ`` with the value `obj` and possible axes
        `axes`.
    """
    try:
        typ = typ.type
    except AttributeError:
        pass

    res_t = np.result_type(obj.dtype, dtype)

    if not isinstance(typ, partial) and issubclass(typ, PandasObject):
        if name is None:
            return typ(obj, dtype=res_t, **axes)
        return typ(obj, dtype=res_t, name=name, **axes)

    # special case for pathological things like ~True/~False
    if hasattr(res_t, "type") and typ == np.bool_ and res_t != np.bool_:
        ret_value = res_t.type(obj)
    else:
        ret_value = res_t.type(obj)
        # The condition is to distinguish 0-dim array (returned in case of
        # scalar) and 1 element array
        # e.g. np.array(0) and np.array([0])
        if (
            len(obj.shape) == 1
            and len(obj) == 1
            and not isinstance(ret_value, np.ndarray)
        ):
            ret_value = np.array([ret_value]).astype(res_t)

    return ret_value
