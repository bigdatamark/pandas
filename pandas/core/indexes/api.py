# [MODULE DOC]
# Aggregation point for the index classes plus helpers that combine several
# indexes. Re-exports the base index, the concrete index classes (categorical,
# datetime, interval, multi, period, range, timedelta), the invalid-index
# error, the missing-timestamp sentinel, and base-module utilities for index
# construction and name handling. Defines: combining axes of objects (union or
# intersection), sorting an index safely, union of many indexes of mixed kinds,
# a same-contents test, and a default range index factory.
# Relationships: depends on the base index module and each concrete index
# module; used by frame, series, concat and alignment code. Import-time side
# effects: importing every index submodule listed above.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.cast import find_common_type

from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
    Index,
    _new_Index,
    ensure_index,
    ensure_index_from_sequences,
    get_unanimous_names,
    maybe_sequence_to_range,
)
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas._typing import Axis


__all__ = [
    "CategoricalIndex",
    "DatetimeIndex",
    "Index",
    "IntervalIndex",
    "InvalidIndexError",
    "MultiIndex",
    "NaT",
    "PeriodIndex",
    "RangeIndex",
    "TimedeltaIndex",
    "_new_Index",
    "all_indexes_same",
    "default_index",
    "ensure_index",
    "ensure_index_from_sequences",
    "get_objs_combined_axis",
    "get_unanimous_names",
    "maybe_sequence_to_range",
    "safe_sort_index",
    "union_indexes",
]


def get_objs_combined_axis(
    objs,
    intersect: bool = False,
    axis: Axis = 0,
    sort: bool | lib.NoDefault = True,
) -> Index:
    # [FUNCTION DOC]
    # Collects the label index of the given axis from every object and
    # combines them. objs is a list of series or frames; intersect (default
    # false) chooses intersection rather than union; axis is the axis number
    # or name (default zero); sort (default true, or the no-default sentinel
    # used for a deprecation) controls sorting of the result. Returns the
    # combined index via the private combiner. Raises whatever the axis lookup
    # raises for an invalid axis.
    # KNOWN ISSUE: the docstring says none is returned when objects lack
    # indexes and lists axis value "outer"; the code always returns an Index
    # (an empty default index for no objects) and has no such axis value.
    """
    Extract combined index: return intersection or union (depending on the
    value of "intersect") of indexes on given axis, or None if all objects
    lack indexes (e.g. they are numpy arrays).

    Parameters
    ----------
    objs : list
        Series or DataFrame objects, may be mix of the two.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    axis : {0 or 'index', 1 or 'outer'}, default 0
        The axis to extract indexes from.
    sort : bool, default True
        Whether the result index should come out sorted or not. NoDefault
        use for deprecation in GH#57335.

    Returns
    -------
    Index
    """
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort)


def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    # [FUNCTION DOC]
    # HELPER: for the index combiner. Returns a new list with duplicates
    # removed by object identity (not equality), keeping first-seen order.
    """
    Return a list with distinct elements of "objs" (different ids).
    Preserves order.
    """
    ids: set[int] = set()
    res = []
    for obj in objs:
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res


def _get_combined_index(
    indexes: list[Index],
    intersect: bool = False,
    sort: bool | lib.NoDefault = False,
) -> Index:
    # [FUNCTION DOC]
    # HELPER: core of the axis combiner. indexes is a list of indexes (lists
    # are only meaningful for union); intersect picks intersection over union;
    # sort may be true, false or the no-default sentinel. Drops
    # identical (same object) entries first. Empty input gives an empty default
    # range index; a single entry is returned as is; intersection is folded
    # pairwise left to right; union uses the union helper (never sorting
    # inside it unless the sentinel is passed through) and converts to an
    # index. Afterwards, if sort is truthy and not the sentinel, the result is
    # sorted with the safe sort helper. Returns an index.
    # KNOWN ISSUE: the union call passes the sort value through only when it is
    # the no-default sentinel and otherwise passes false, which looks inverted
    # relative to its apparent intent; sorting is then done separately below.
    # Index names are not handled (an unresolved TODO in the code).
    """
    Return the union or intersection of indexes.

    Parameters
    ----------
    indexes : list of Index or list objects
        When intersect=True, do not accept list of lists.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    sort : bool, default False
        Whether the result index should come out sorted or not. NoDefault
        used for deprecation of GH#57335

    Returns
    -------
    Index
    """
    # TODO: handle index names!
    indexes = _get_distinct_objs(indexes)
    if len(indexes) == 0:
        index: Index = default_index(0)
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        index = indexes[0]
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        index = union_indexes(indexes, sort=sort if sort is lib.no_default else False)
        index = ensure_index(index)

    if sort and sort is not lib.no_default:
        index = safe_sort_index(index)
    return index


def safe_sort_index(index: Index) -> Index:
    # [FUNCTION DOC]
    # Returns an index sorted ascending, keeping dtype and name(s). An index
    # that is already monotonic increasing is returned as is. Otherwise uses
    # the generic safe sort; if sorting raises a type error (unorderable
    # mixed values) the original index is returned unchanged. A result that
    # is an Index is returned directly; an array result is rebuilt as a
    # multi-level index (with the original level names) or as a plain index
    # with original name and dtype. No input mutation.
    """
    Returns the sorted index

    We keep the dtypes and the name attributes.

    Parameters
    ----------
    index : an Index

    Returns
    -------
    Index
    """
    if index.is_monotonic_increasing:
        return index

    try:
        array_sorted = safe_sort(index)
    except TypeError:
        pass
    else:
        if isinstance(array_sorted, Index):
            return array_sorted

        array_sorted = cast(np.ndarray, array_sorted)
        if isinstance(index, MultiIndex):
            index = MultiIndex.from_tuples(array_sorted, names=index.names)
        else:
            index = Index(array_sorted, name=index.name, dtype=index.dtype)

    return index


def union_indexes(indexes, sort: bool | lib.NoDefault = True) -> Index:
    # [FUNCTION DOC]
    # Union of a sequence of indexes and/or plain lists. sort (default true)
    # may be the no-default sentinel (treated as not sorting for lists, and
    # as sorting when all inputs are datetime indexes).
    # Raises an assertion error for empty input. A single entry is returned as
    # is, except a list, which becomes an index (sorted only if sort is truthy
    # and not the sentinel). Otherwise inputs are classified: all lists use a
    # fast ordered unique-merge of all values, with the common dtype of any
    # index inputs; plain indexes of identical type use common-dtype casting,
    # first unique values then appended new values, optionally sorted, and the
    # result is renamed to the shared name if one exists; any mix of special
    # types is folded pairwise with the index union method. In the special
    # case, mixing tz-aware and tz-naive datetime indexes raises a type error,
    # and mixed time zones are cast to object dtype with sorting turned off.
    # An unknown classification raises a value error (unreachable in practice).
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not. NoDefault
        used for deprecation of GH#57335.

    Returns
    -------
    Index
    """
    if len(indexes) == 0:
        raise AssertionError("Must have at least 1 Index to union")
    if len(indexes) == 1:
        result = indexes[0]
        if isinstance(result, list):
            if not sort or sort is lib.no_default:
                result = Index(result)
            else:
                result = Index(sorted(result))
        return result

    indexes, kind = _sanitize_and_check(indexes)

    if kind == "special":
        result = indexes[0]

        num_dtis = 0
        num_dti_tzs = 0
        for idx in indexes:
            if isinstance(idx, DatetimeIndex):
                num_dtis += 1
                if idx.tz is not None:
                    num_dti_tzs += 1
        if num_dti_tzs not in [0, num_dtis]:
            # TODO: this behavior is not tested (so may not be desired),
            #  but is kept in order to keep behavior the same when
            #  deprecating union_many
            # test_frame_from_dict_with_mixed_indexes
            raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        if num_dtis == len(indexes):
            if sort is lib.no_default:
                sort = True
            result = indexes[0]

        elif num_dtis > 1:
            # If we have mixed timezones, our casting behavior may depend on
            #  the order of indexes, which we don't want.
            sort = False

            # TODO: what about Categorical[dt64]?
            # test_frame_from_dict_with_mixed_indexes
            indexes = [x.astype(object, copy=False) for x in indexes]
            result = indexes[0]

        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result

    elif kind == "array":
        if not all_indexes_same(indexes):
            dtype = find_common_type([idx.dtype for idx in indexes])
            inds = [ind.astype(dtype, copy=False) for ind in indexes]
            index = inds[0].unique()
            other = inds[1].append(inds[2:])
            diff = other[index.get_indexer_for(other) == -1]
            if len(diff):
                index = index.append(diff.unique())
            if sort:
                index = index.sort_values()
        else:
            index = indexes[0]

        name = get_unanimous_names(*indexes)[0]
        if name != index.name:
            index = index.rename(name)
        return index
    elif kind == "list":
        dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
        if dtypes:
            dtype = find_common_type(dtypes)
        else:
            dtype = None
        all_lists = (idx.tolist() if isinstance(idx, Index) else idx for idx in indexes)
        return Index(
            lib.fast_unique_multiple_list_gen(all_lists, sort=bool(sort)),
            dtype=dtype,
        )
    else:
        raise ValueError(f"{kind=} must be 'special', 'array' or 'list'.")


def _sanitize_and_check(indexes):
    # [FUNCTION DOC]
    # HELPER: for the union helper. Takes a sequence of indexes and/or lists
    # and returns a pair of (possibly converted sequence, kind), where kind is
    # list (all inputs lists), special (more than one distinct type, or no
    # exact base index type present) or array (all exact base indexes). When
    # lists are mixed with indexes, each list is converted to an index and the
    # list type is ignored when classifying. No exceptions.
    # KNOWN ISSUE: the docstring says lists mixed with indexes are sorted
    # before conversion; the code converts them without sorting.
    """
    Verify the type of indexes and convert lists to Index.

    Cases:

    - [list, list, ...]: Return ([list, list, ...], 'list')
    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])
        Lists are sorted and converted to Index.
    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)
        TYPE = 'special' if at least one special type, 'array' otherwise.

    Parameters
    ----------
    indexes : list of Index or list objects

    Returns
    -------
    sanitized_indexes : list of Index or list objects
    type : {'list', 'array', 'special'}
    """
    kinds = {type(index) for index in indexes}

    if list in kinds:
        if len(kinds) > 1:
            indexes = [
                Index(list(x)) if not isinstance(x, Index) else x for x in indexes
            ]
            kinds -= {list}
        else:
            return indexes, "list"

    if len(kinds) > 1 or Index not in kinds:
        return indexes, "special"
    else:
        return indexes, "array"


def all_indexes_same(indexes) -> bool:
    # [FUNCTION DOC]
    # Returns true if every index equals the first one by the index equality
    # method. indexes is a non-empty iterable (an empty one raises a stop
    # iteration error). Consumes one-shot iterators.
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects

    Returns
    -------
    bool
        True if all indexes contain the same elements, False otherwise.
    """
    itr = iter(indexes)
    first = next(itr)
    return all(first.equals(index) for index in itr)


def default_index(n: int) -> RangeIndex:
    # [FUNCTION DOC]
    # Returns an unnamed range index covering zero up to n (exclusive), built
    # through the fast internal constructor with no validation of n.
    rng = range(n)
    return RangeIndex._simple_new(rng, name=None)
