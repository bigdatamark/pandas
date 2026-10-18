"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""

# [MODULE DOC]
# The "split" machinery of groupby. Public contents: the Grouper class, a
# user-facing specification of how to group (by column key, index level or, for
# a frequency, by time bins via a time-grouper subclass); an internal Grouping
# class describing the grouping of a single key (codes, unique values, group
# index, label to position mappings); and the get_grouper function, which
# interprets user keys/levels and builds the internal base grouper made of one
# or more Grouping objects, along with the set of column names to exclude from
# aggregation. Private helpers decide if a value is label-like and normalize a
# user grouper (dict, Series, list, etc.) into something usable.
# Relationships: consumed by groupby.py and by resample (which supplies the
# time-grouper subclass, imported lazily); builds on ops.py (base and bin
# groupers), categorical recoding helpers and the index API.
# No import-time side effects.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    final,
)

import numpy as np

from pandas._libs import (
    algos as libalgos,
)
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)

from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_list_like,
    is_scalar,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core import algorithms
from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
)
from pandas.core.series import Series

from pandas.io.formats.printing import (
    PrettyDict,
    pprint_thing,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas._typing import (
        ArrayLike,
        NDFrameT,
        npt,
    )

    from pandas.core.generic import NDFrame


@set_module("pandas")
class Grouper:
    # [CLASS DOC]
    # User-facing groupby specification. State: key (column label to group by),
    # level (index level), freq (frequency; when set, construction returns the
    # time-grouper subclass), sort and dropna flags, plus private slots filled
    # during grouping: a binner, the grouper axis and the sort indexer (and a
    # deprecated copy of it). The class attribute listing the attributes used
    # in the textual representation is in the attributes tuple. Key methods:
    # construction hook choosing the subclass, grouper setup helpers and a
    # representation. A key and a level are mutually exclusive.
    """
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level parameter is given, a level of the index of the target
    object.

    If ``level`` is passed as a keyword to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    *args
        Currently unused, reserved for future use.
    **kwargs
        Dictionary of the keyword arguments to pass to Grouper.

    Attributes
    ----------
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see :ref:`here<timeseries.offset_aliases>`.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {'left' or 'right'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {'left' or 'right'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {'start', 'end', 'e', 's'}
        If grouper is PeriodIndex and `freq` parameter is passed.

    origin : Timestamp or str, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If string, must be one of the following:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries

        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

    dropna : bool, default True
        If True, and if group keys contain NA values, NA values together with
        row/column will be dropped. If False, NA values will also be treated as
        the key in groups.

    Returns
    -------
    Grouper or pandas.api.typing.TimeGrouper
        A TimeGrouper is returned if ``freq`` is not ``None``. Otherwise, a Grouper
        is returned.

    See Also
    --------
    Series.groupby : Apply a function groupby to a Series.
    DataFrame.groupby : Apply a function groupby.

    Examples
    --------
    ``df.groupby(pd.Grouper(key="Animal"))`` is equivalent to ``df.groupby('Animal')``

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column 'Publish date'

    >>> df = pd.DataFrame(
    ...     {
    ...         "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16"),
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40],
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = "2000-10-01 23:30:00", "2000-10-02 00:30:00"
    >>> rng = pd.date_range(start, end, freq="7min")
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
    2000-10-02 00:05:00    15
    2000-10-02 00:12:00    18
    2000-10-02 00:19:00    21
    2000-10-02 00:26:00    24
    Freq: 7min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min")).sum()
    2000-10-01 23:14:00     0
    2000-10-01 23:31:00     9
    2000-10-01 23:48:00    21
    2000-10-02 00:05:00    54
    2000-10-02 00:22:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min", origin="epoch")).sum()
    2000-10-01 23:18:00     0
    2000-10-01 23:35:00    18
    2000-10-01 23:52:00    27
    2000-10-02 00:09:00    39
    2000-10-02 00:26:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min", origin="2000-01-01")).sum()
    2000-10-01 23:24:00     3
    2000-10-01 23:41:00    15
    2000-10-01 23:58:00    45
    2000-10-02 00:15:00    45
    Freq: 17min, dtype: int64

    If you want to adjust the start of the bins with an `offset` Timedelta, the two
    following lines are equivalent:

    >>> ts.groupby(pd.Grouper(freq="17min", origin="start")).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min", offset="23h30min")).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    To replace the use of the deprecated `base` argument, you can now use `offset`,
    in this example it is equivalent to have `base=2`:

    >>> ts.groupby(pd.Grouper(freq="17min", offset="2min")).sum()
    2000-10-01 23:16:00     0
    2000-10-01 23:33:00     9
    2000-10-01 23:50:00    36
    2000-10-02 00:07:00    39
    2000-10-02 00:24:00    24
    Freq: 17min, dtype: int64
    """

    sort: bool
    dropna: bool
    _grouper: Index | None

    _attributes: tuple[str, ...] = ("key", "level", "freq", "sort", "dropna")

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Construction hook. If the keyword arguments contain a frequency that
        # is not none, the class to instantiate is replaced by the time-grouper
        # subclass (imported lazily from the resample module); otherwise the
        # requested class is used. Positional arguments are ignored here.
        # Returns an uninitialized instance of the chosen class.
        if kwargs.get("freq") is not None:
            from pandas.core.resample import TimeGrouper

            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(
        self,
        key=None,
        level=None,
        freq=None,
        sort: bool = False,
        dropna: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Stores key (column label), level (index level name or number), freq
        # (frequency), sort (default false) and dropna (default true) as
        # attributes, and initializes the internal state (binner, grouper axis
        # and sort indexers) to none. No validation here; a key together with
        # a level is rejected later when the grouper is set up.
        self.key = key
        self.level = level
        self.freq = freq
        self.sort = sort
        self.dropna = dropna

        self._indexer_deprecated: npt.NDArray[np.intp] | None = None
        self.binner = None
        self._grouper = None
        self._indexer: npt.NDArray[np.intp] | None = None

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True, observed: bool = True
    ) -> tuple[ops.BaseGrouper, NDFrameT]:
        # [FUNCTION DOC]
        # HELPER: used by the grouping machinery to turn this specification into
        # an internal grouper. Calls the grouper setup step (which may sort the
        # object) and then the module-level grouper builder with this
        # specification's key, level, sort and dropna settings plus the given
        # validate and observed flags. Returns the base grouper and the
        # (possibly reordered) object. Errors propagate from those steps.
        """
        Parameters
        ----------
        obj : Series or DataFrame
            Object being grouped.
        validate : bool, default True
            If True, validate the grouper.
        observed : bool, default True
            Whether only observed groups should be in the result. Only
            has an impact when grouping on categorical data.

        Returns
        -------
        A tuple of grouper, obj (possibly sorted)
        """
        obj, _, _ = self._set_grouper(obj)
        grouper, _, obj = get_grouper(
            obj,
            [self.key],
            level=self.level,
            sort=self.sort,
            validate=validate,
            dropna=self.dropna,
            observed=observed,
        )

        return grouper, obj

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        # [FUNCTION DOC]
        # HELPER: resolves the grouping axis for an object and optionally
        # sorts. Inputs: obj (series or frame, must not be none), sort (extra
        # request to sort the axis, default false), gpr_index (a previously
        # computed grouper axis, keyword only, default none).
        # Raises a value error if both key and level are set, a key error if
        # the key is not found among the object's columns, and a value error
        # if a non-multi level is neither zero nor the index name.
        # Side effects: on the first call, stores gpr_index as the grouper axis
        # and copies the deprecated sort indexer into the current indexer; when
        # sorting, stores the new sort order in the deprecated indexer attribute.
        # The axis is chosen as: when a key is set and equals the name of the
        # supplied grouper axis and obj is a series, the stored grouper axis
        # (un-sorted first if a sort indexer exists) taken at obj's index;
        # otherwise the key column as an index; with no key, the object index or
        # the requested multi-index level values. If self or the sort argument
        # asks for sorting and the axis is not already monotonic increasing, a
        # stable mergesort order (missing first) is computed and both axis and
        # object are reordered along rows.
        # Returns (object, axis, sort indexer or none when not sorted).
        # KNOWN ISSUE: unresolved TODOs in the body admit that the assumptions
        # about repeated calls (the first-call-only state capture) and the
        # missing-first ordering for datetime-likes are not understood.
        """
        given an object and the specifications, setup the internal grouper
        for this particular specification

        Parameters
        ----------
        obj : Series or DataFrame
        sort : bool, default False
            whether the resulting grouper should be sorted
        gpr_index : Index or None, default None

        Returns
        -------
        NDFrame
        Index
        np.ndarray[np.intp] | None
        """
        assert obj is not None

        if self.key is not None and self.level is not None:
            raise ValueError("The Grouper cannot specify both a key and a level!")

        # Keep self._grouper value before overriding
        if self._grouper is None:
            # TODO: What are we assuming about subsequent calls?
            self._grouper = gpr_index
            self._indexer = self._indexer_deprecated

        # the key must be a valid info item
        if self.key is not None:
            key = self.key
            # The 'on' is already defined
            if getattr(gpr_index, "name", None) == key and isinstance(obj, Series):
                # Sometimes self._grouper will have been resorted while
                # obj has not. In this case there is a mismatch when we
                # call self._grouper.take(obj.index) so we need to undo the sorting
                # before we call _grouper.take.
                assert self._grouper is not None
                if self._indexer is not None:
                    reverse_indexer = self._indexer.argsort()
                    unsorted_ax = self._grouper.take(reverse_indexer)
                    ax = unsorted_ax.take(obj.index)
                else:
                    ax = self._grouper.take(obj.index)
            else:
                if key not in obj._info_axis:
                    raise KeyError(f"The grouper name {key} is not found")
                ax = Index(obj[key], name=key)

        else:
            ax = obj.index
            if self.level is not None:
                level = self.level

                # if a level is given it must be a mi level or
                # equivalent to the axis name
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax._get_level_values(level), name=ax.names[level])

                elif level not in (0, ax.name):
                    raise ValueError(f"The level {level} is not valid")

        # possibly sort
        indexer: npt.NDArray[np.intp] | None = None
        if (self.sort or sort) and not ax.is_monotonic_increasing:
            # use stable sort to support first, last, nth
            # TODO: why does putting na_position="first" fix datetimelike cases?
            indexer = self._indexer_deprecated = ax.array.argsort(
                kind="mergesort", na_position="first"
            )
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=0)

        return obj, ax, indexer

    @final
    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a string made of the class name followed by parentheses
        # containing comma separated name=repr pairs for each attribute listed in
        # the class attributes tuple (key, level, freq, sort, dropna) whose
        # value is not none. Note a false sort flag is still shown.
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)!r}"
            for attr_name in self._attributes
            if getattr(self, attr_name) is not None
        )
        attrs = ", ".join(attrs_list)
        cls_name = type(self).__name__
        return f"{cls_name}({attrs})"


@final
class Grouping:
    # [CLASS DOC]
    # Internal description of the grouping for one key. Not subclassable.
    # State: the axis index being grouped, the original grouper object as
    # passed, the processed grouping vector (array, extension array, index,
    # series, or a base/bin grouper), level, sort, observed, dropna, an
    # optional explicit uniques array, the object grouped, a flag saying if
    # the key is a column of the object, and the original categories when the
    # input was categorical. Lazily computed cached attributes: name, codes and
    # uniques (via one combined computation), indices, groups and a
    # copy of itself with observed forced on. Used as an element of the list
    # of groupings of a base grouper.
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
    grouper :
    obj : DataFrame or Series
    name : Label
    level :
    observed : bool, default False
        If we are a Categorical, use the observed values
    in_axis : if the Grouping is a column in self.obj and hence among
        Groupby.exclusions list
    dropna : bool, default True
        Whether to drop NA groups.
    uniques : Array-like, optional
        When specified, will be used for unique values. Enables including empty groups
        in the result for a BinGrouper. Must not contain duplicates.

    Attributes
    -------
    indices : dict
        Mapping of {group -> index_list}
    codes : ndarray
        Group codes
    group_index : Index or None
        unique groups
    groups : dict
        Mapping of {group -> label_list}
    """

    _codes: npt.NDArray[np.signedinteger] | None = None
    _orig_cats: Index | None
    _index: Index

    def __init__(
        self,
        index: Index,
        grouper=None,
        obj: NDFrame | None = None,
        level=None,
        sort: bool = True,
        observed: bool = False,
        in_axis: bool = False,
        dropna: bool = True,
        uniques: ArrayLike | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Parameters: index (axis being grouped), grouper (the key: mapping,
        # function, series, array-like, Grouper, base grouper, or none when a
        # level is used), obj (the grouped object), level (index level name or
        # number), sort (default true), observed (default false, only matters
        # for categorical data), in_axis (whether the key is a column of obj,
        # default false), dropna (default true), uniques (optional array of
        # unique values to use, enabling empty groups).
        # Series keys are shallow-copied. The grouper is normalized by the
        # module converter. If a level is given the grouping vector is the
        # index (or multi-index level) values, mapped through the grouper if
        # one was also given. Else if the vector is a Grouper object, it builds
        # that grouper against obj (replacing obj with the possibly reordered
        # result), keeping a bin grouper as is or, for a base grouper, taking
        # the first grouping's vector as a named index. Else if it is not an
        # array-like type, it must be one dimensional (else value error) and is
        # applied as a mapper to the index; the result must have the index's
        # length, otherwise an assertion error is raised. Datetime-like numpy
        # arrays are normalized via a series round trip; categorical vectors
        # have their categories remembered and are recoded for groupby using
        # sort and observed. Stores the result as the grouping vector.
        if isinstance(grouper, Series):
            grouper = grouper.copy(deep=False)
        self.level = level
        self._orig_grouper = grouper
        grouping_vector = _convert_grouper(index, grouper)
        self._orig_cats = None
        self._index = index
        self._sort = sort
        self.obj = obj
        self._observed = observed
        self.in_axis = in_axis
        self._dropna = dropna
        self._uniques = uniques

        # we have a single grouper which may be a myriad of things,
        # some of which are dependent on the passing in level

        ilevel = self._ilevel
        if ilevel is not None:
            # In extant tests, the new self.grouping_vector matches
            #  `index.get_level_values(ilevel)` whenever
            #  mapper is None and isinstance(index, MultiIndex)
            if isinstance(index, MultiIndex):
                index_level = index.get_level_values(ilevel)
            else:
                index_level = index

            if grouping_vector is None:
                grouping_vector = index_level
            else:
                mapper = grouping_vector
                grouping_vector = index_level.map(mapper)

        # a passed Grouper like, directly get the grouper in the same way
        # as single grouper groupby, use the group_info to get codes
        elif isinstance(grouping_vector, Grouper):
            # get the new grouper; we already have disambiguated
            # what key/level refer to exactly, don't need to
            # check again as we have by this point converted these
            # to an actual value (rather than a pd.Grouper)
            assert self.obj is not None  # for mypy
            newgrouper, newobj = grouping_vector._get_grouper(self.obj, validate=False)
            self.obj = newobj

            if isinstance(newgrouper, ops.BinGrouper):
                # TODO: can we unwrap this and get a tighter typing
                #  for self.grouping_vector?
                grouping_vector = newgrouper
            else:
                # ops.BaseGrouper
                # TODO: 2023-02-03 no test cases with len(newgrouper.groupings) > 1.
                #  If that were to occur, would we be throwing out information?
                # error: Cannot determine type of "grouping_vector"  [has-type]
                ng = newgrouper.groupings[0].grouping_vector  # type: ignore[has-type]
                # use Index instead of ndarray so we can recover the name
                grouping_vector = Index(
                    ng, name=newgrouper.result_index.name, copy=False
                )

        elif not isinstance(
            grouping_vector, (Series, Index, ExtensionArray, np.ndarray)
        ):
            # no level passed
            if getattr(grouping_vector, "ndim", 1) != 1:
                t = str(type(grouping_vector))
                raise ValueError(f"Grouper for '{t}' not 1-dimensional")

            grouping_vector = index.map(grouping_vector)

            if not (
                hasattr(grouping_vector, "__len__")
                and len(grouping_vector) == len(index)
            ):
                grper = pprint_thing(grouping_vector)
                errmsg = (
                    f"Grouper result violates len(labels) == len(data)\nresult: {grper}"
                )
                raise AssertionError(errmsg)

        if isinstance(grouping_vector, np.ndarray):
            if grouping_vector.dtype.kind in "mM":
                # if we have a date/time-like grouper, make sure that we have
                # Timestamps like
                # TODO 2022-10-08 we only have one test that gets here and
                #  values are already in nanoseconds in that case.
                grouping_vector = Series(grouping_vector).to_numpy()
        elif isinstance(getattr(grouping_vector, "dtype", None), CategoricalDtype):
            # a passed Categorical
            self._orig_cats = grouping_vector.categories
            grouping_vector = recode_for_groupby(grouping_vector, sort, observed)

        self.grouping_vector = grouping_vector

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a short string of the form Grouping(name) using the grouping name.
        return f"Grouping({self.name})"

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Iterates over the group labels (the keys of the indices mapping).
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached; true when the processed grouping vector has a
        # categorical dtype.
        dtype = getattr(self.grouping_vector, "dtype", None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Hashable:
        # [FUNCTION DOC]
        # Cached label for this grouping. Order of precedence: the index level
        # name when a level is used; the name of the original grouper if it
        # was an index or series; the result index name if the vector is a
        # base grouper; the name of the vector if it is an index; else none.
        ilevel = self._ilevel
        if ilevel is not None:
            return self._index.names[ilevel]

        if isinstance(self._orig_grouper, (Index, Series)):
            return self._orig_grouper.name

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.result_index.name

        elif isinstance(self.grouping_vector, Index):
            return self.grouping_vector.name

        # otherwise we have ndarray or ExtensionArray -> no name
        return None

    @cache_readonly
    def _ilevel(self) -> int | None:
        # [FUNCTION DOC]
        # HELPER: cached; returns none if no level was given, an integer level
        # as is, and otherwise the position of the level name among the index
        # names. Raises an assertion error if a non-integer level is not
        # among the names.
        """
        If necessary, converted index level name to index level position.
        """
        level = self.level
        if level is None:
            return None
        if not isinstance(level, int):
            index = self._index
            if level not in index.names:
                raise AssertionError(f"Level {level} not in index")
            return index.names.index(level)
        return level

    @property
    def ngroups(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of groups, i.e. the length of the unique values.
        return len(self.uniques)

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # Cached mapping from each group label to the integer positions of its
        # rows. Delegates to the wrapped grouper when the vector is a base
        # grouper; otherwise builds a categorical from the vector and uses its
        # reverse indexer.
        # we have a list of groupers
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.indices

        values = Categorical(self.grouping_vector)
        return values._reverse_indexer()

    @property
    def codes(self) -> npt.NDArray[np.signedinteger]:
        # [FUNCTION DOC]
        # Property: the integer group code of each row (first element of the
        # combined codes and uniques computation); missing groups are coded minus one
        # unless dropna is false.
        return self._codes_and_uniques[0]

    @property
    def uniques(self) -> ArrayLike:
        # [FUNCTION DOC]
        # Property: the unique group values in group-code order (second element
        # of the combined codes and uniques computation).
        return self._codes_and_uniques[1]

    @cache_readonly
    def _codes_and_uniques(self) -> tuple[npt.NDArray[np.signedinteger], ArrayLike]:
        # [FUNCTION DOC]
        # HELPER: cached core computation behind codes and uniques; returns a
        # pair (codes, uniques).
        # Categorical vector: uniques are a categorical built from codes that
        # are all categories (unobserved) or only the codes that appear (observed,
        # sorted when sort is on, minus the missing code). If dropna is false and
        # missing values exist, a missing group is inserted: at the end when
        # sorting (code equal to the category count), otherwise at the position
        # of first appearance (shifting higher codes up); missing rows get that
        # code. Base-grouper vector: its combined codes and result index values.
        # Explicit uniques given: codes come from a categorical of the vector
        # over those uniques (so absent uniques make empty groups). Otherwise
        # the vector is factorized, sorted if sort is on, and missing values
        # get their own group only when dropna is false.
        # KNOWN ISSUE: a comment says dropna=False is not handled for
        # categoricals (an old GH issue reference), but the code right below
        # does handle it, so the comment is stale.
        uniques: ArrayLike
        if self._passed_categorical:
            # we make a CategoricalIndex out of the cat grouper
            # preserving the categories / ordered attributes;
            # doesn't (yet - GH#46909) handle dropna=False
            cat = self.grouping_vector
            categories = cat.categories

            if self._observed:
                ucodes = algorithms.unique1d(cat.codes)
                ucodes = ucodes[ucodes != -1]
                if self._sort:
                    ucodes = np.sort(ucodes)
            else:
                ucodes = np.arange(len(categories))

            has_dropped_na = False
            if not self._dropna:
                na_mask = cat.isna()
                if np.any(na_mask):
                    has_dropped_na = True
                    if self._sort:
                        # NA goes at the end, gets `largest non-NA code + 1`
                        na_code = len(categories)
                    else:
                        # Insert NA in result based on first appearance, need
                        # the number of unique codes prior
                        na_idx = na_mask.argmax()
                        na_code = algorithms.nunique_ints(cat.codes[:na_idx])
                    ucodes = np.insert(ucodes, na_code, -1)

            uniques = Categorical.from_codes(
                codes=ucodes, categories=categories, ordered=cat.ordered, validate=False
            )
            codes = cat.codes

            if has_dropped_na:
                if not self._sort:
                    # NA code is based on first appearance, increment higher codes
                    codes = np.where(codes >= na_code, codes + 1, codes)
                codes = np.where(na_mask, na_code, codes)

            return codes, uniques

        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            # we have a list of groupers
            codes = self.grouping_vector.codes_info
            uniques = self.grouping_vector.result_index._values
        elif self._uniques is not None:
            # GH#50486 Code grouping_vector using _uniques; allows
            # including uniques that are not present in grouping_vector.
            cat = Categorical(self.grouping_vector, categories=self._uniques)
            codes = cat.codes
            uniques = self._uniques
        else:
            # GH35667, replace dropna=False with use_na_sentinel=False
            # error: Incompatible types in assignment (expression has type "Union[
            # ndarray[Any, Any], Index]", variable has type "Categorical")
            codes, uniques = algorithms.factorize(  # type: ignore[assignment]
                self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna
            )
        return codes, uniques

    @cache_readonly
    def groups(self) -> dict[Hashable, Index]:
        # [FUNCTION DOC]
        # Cached mapping from each group label to the index labels of its
        # rows, returned as a pretty-printing dict subclass. Sorts positions by
        # code with the group sort routine, splits them by cumulative counts and
        # takes those positions from the axis index; labels are the uniques
        # wrapped as an index with the grouping name.
        codes, uniques = self._codes_and_uniques
        uniques = Index._with_infer(uniques, name=self.name, copy=False)

        r, counts = libalgos.groupsort_indexer(ensure_platform_int(codes), len(uniques))
        counts = ensure_int64(counts).cumsum()
        _result = (r[start:end] for start, end in zip(counts, counts[1:], strict=False))
        # map to the label
        result = {k: self._index.take(v) for k, v in zip(uniques, _result, strict=True)}

        return PrettyDict(result)

    @property
    def observed_grouping(self) -> Grouping:
        # [FUNCTION DOC]
        # Property: this grouping itself if it is already observed-only,
        # otherwise the cached observed-only copy.
        if self._observed:
            return self

        return self._observed_grouping

    @cache_readonly
    def _observed_grouping(self) -> Grouping:
        # [FUNCTION DOC]
        # HELPER: cached; builds a new grouping from the same index, original
        # grouper, object, level, sort, in_axis, dropna and uniques but with
        # observed forced on.
        grouping = Grouping(
            self._index,
            self._orig_grouper,
            obj=self.obj,
            level=self.level,
            sort=self._sort,
            observed=True,
            in_axis=self.in_axis,
            dropna=self._dropna,
            uniques=self._uniques,
        )
        return grouping


def get_grouper(
    obj: NDFrameT,
    key=None,
    level=None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True,
) -> tuple[ops.BaseGrouper, frozenset[Hashable], NDFrameT]:
    # [FUNCTION DOC]
    # Builds the internal base grouper for an object. Parameters: obj (series
    # or frame; its index is the group axis), key (a label, array-like,
    # function, mapping, Grouper, Grouping, base grouper, or a list of those;
    # default none), level (index level or list of levels, default none), sort
    # (default true), observed (categoricals, default false), validate
    # (default true; check label/level ambiguity), dropna (default true).
    # Level handling: for a multi-index a length-one list is unwrapped and a
    # scalar level with no key is turned into that level's values as the key;
    # for a plain index a length-one list is unwrapped, an empty list raises a
    # value error (no group keys), several levels raise a value error, a
    # string level must equal the index name and an integer must be zero or
    # minus one (value errors otherwise), after which the whole index is the key.
    # A Grouper key is resolved via its own setup and returns immediately, with
    # its key (if any) as the sole exclusion; a base grouper key is returned as
    # is with no exclusions. For a list key whose length equals the axis length
    # and holds only plain labels none of which are columns or index names, the
    # list is treated as one array-like key. Each key then becomes a Grouping:
    # series that share values with an obj column count as in-axis (their name
    # is excluded); labels that are columns are looked up (value error if
    # the column is not one dimensional; ambiguity validated if requested) and
    # excluded; labels that are index level references become a level; other
    # labels raise a key error; Grouper objects with a key are excluded.
    # Existing Grouping objects are reused. If no groupings result and obj is
    # not empty, raises a value error (no group keys); for an empty obj a
    # placeholder empty grouping is used.
    # Returns (base grouper, frozenset of excluded column labels, obj).
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to level,sort, while
    the passed in level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    group_axis = obj.index

    # validate that the passed single level is compatible with the passed
    # index of the object
    if level is not None:
        # TODO: These if-block and else-block are almost same.
        # MultiIndex instance check is removable, but it seems that there are
        # some processes only for non-MultiIndex in else-block,
        # eg. `obj.index.name != level`. We have to consider carefully whether
        # these are applicable for MultiIndex. Even if these are applicable,
        # we need to check if it makes no side effect to subsequent processes
        # on the outside of this condition.
        # (GH 17621)
        if isinstance(group_axis, MultiIndex):
            if is_list_like(level) and len(level) == 1:
                level = level[0]

            if key is None and is_scalar(level):
                # Get the level values from group_axis
                key = group_axis.get_level_values(level)
                level = None

        else:
            # allow level to be a length-one list-like object
            # (e.g., level=[0])
            # GH 13901
            if is_list_like(level):
                nlevels = len(level)
                if nlevels == 1:
                    level = level[0]
                elif nlevels == 0:
                    raise ValueError("No group keys passed!")
                else:
                    raise ValueError("multiple levels only valid with MultiIndex")

            if isinstance(level, str):
                if obj.index.name != level:
                    raise ValueError(f"level name {level} is not the name of the index")
            elif level > 0 or level < -1:
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")

            # NOTE: `group_axis` and `group_axis.get_level_values(level)`
            # are same in this section.
            level = None
            key = group_axis

    # a passed-in Grouper, directly convert
    if isinstance(key, Grouper):
        grouper, obj = key._get_grouper(obj, validate=False, observed=observed)
        if key.key is None:
            return grouper, frozenset(), obj
        else:
            return grouper, frozenset({key.key}), obj

    # already have a BaseGrouper, just return it
    elif isinstance(key, ops.BaseGrouper):
        return key, frozenset(), obj

    if not isinstance(key, list):
        keys = [key]
        match_axis_length = False
    else:
        keys = key
        match_axis_length = len(keys) == len(group_axis)

    # what are we after, exactly?
    any_callable = any(callable(g) or isinstance(g, dict) for g in keys)
    any_groupers = any(isinstance(g, (Grouper, Grouping)) for g in keys)
    any_arraylike = any(
        isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys
    )

    # is this an index replacement?
    if (
        not any_callable
        and not any_arraylike
        and not any_groupers
        and match_axis_length
        and level is None
    ):
        if isinstance(obj, DataFrame):
            all_in_columns_index = all(
                g in obj.columns or g in obj.index.names for g in keys
            )
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all(g in obj.index.names for g in keys)

        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]

    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        levels = [level] * len(keys)

    groupings: list[Grouping] = []
    exclusions: set[Hashable] = set()

    # if the actual grouper should be obj[key]
    def is_in_axis(key) -> bool:
        # [FUNCTION DOC]
        # HELPER: nested in the grouper builder. Returns true if key is
        # label-like; otherwise false for a one-dimensional object, or, for a
        # frame, true only if the key can be located among the columns (lookup
        # failures such as missing key, wrong type or invalid index key give false).
        if not _is_label_like(key):
            if obj.ndim == 1:
                return False

            # items -> .columns for DataFrame, .index for Series
            items = obj.axes[-1]
            try:
                items.get_loc(key)
            except (KeyError, TypeError, InvalidIndexError):
                # TypeError shows up here if we pass e.g. an Index
                return False

        return True

    # if the grouper is obj[name]
    def is_in_obj(gpr) -> bool:
        # [FUNCTION DOC]
        # HELPER: nested in the grouper builder. Returns true when gpr is a
        # series that shares its underlying values (by reference) with the
        # object's column of the same name. False if gpr has no name, the
        # lookup of that name fails, or either is not a series.
        if not hasattr(gpr, "name"):
            return False
        # We check the references to determine if the
        # series is part of the object
        try:
            obj_gpr_column = obj[gpr.name]
        except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
            return False
        if isinstance(gpr, Series) and isinstance(obj_gpr_column, Series):
            return gpr._mgr.references_same_values(obj_gpr_column._mgr, 0)
        return False

    for gpr, level in zip(keys, levels, strict=True):
        if is_in_obj(gpr):  # df.groupby(df['name'])
            in_axis = True
            exclusions.add(gpr.name)

        elif is_in_axis(gpr):  # df.groupby('name')
            if obj.ndim != 1 and gpr in obj:
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=0)
                in_axis, name, gpr = True, gpr, obj[gpr]
                if gpr.ndim != 1:
                    # non-unique columns; raise here to get the name in the
                    # exception message
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=0):
                in_axis, level, gpr = False, gpr, None
            else:
                raise KeyError(gpr)
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            # Add key to exclusions
            exclusions.add(gpr.key)
            in_axis = True
        else:
            in_axis = False

        # create the Grouping
        # allow us to passing the actual Grouping as the gpr
        ping = (
            Grouping(
                group_axis,
                gpr,
                obj=obj,
                level=level,
                sort=sort,
                observed=observed,
                in_axis=in_axis,
                dropna=dropna,
            )
            if not isinstance(gpr, Grouping)
            else gpr
        )

        groupings.append(ping)

    if len(groupings) == 0 and len(obj):
        raise ValueError("No group keys passed!")
    if len(groupings) == 0:
        groupings.append(Grouping(default_index(0), np.array([], dtype=np.intp)))

    # create the internals grouper
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, dropna=dropna)
    return grouper, frozenset(exclusions), obj


def _is_label_like(val) -> bool:
    # [FUNCTION DOC]
    # HELPER: for the grouper builder. True for strings, tuples and any
    # non-none scalar; false otherwise.
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))


def _convert_grouper(axis: Index, grouper):
    # [FUNCTION DOC]
    # HELPER: for Grouping construction. Normalizes a user grouper against the
    # axis. A mapping becomes its lookup function (missing keys give none). A
    # series is returned as its raw values, first reindexed to the axis when
    # its index differs. A multi-index becomes its tuple values. A list, tuple,
    # index, categorical or numpy array must have the axis length (otherwise
    # raises a value error); lists and tuples become arrays that keep tuples
    # intact. Anything else is returned unchanged.
    if isinstance(grouper, dict):
        return grouper.get
    elif isinstance(grouper, Series):
        if grouper.index.equals(axis):
            return grouper._values
        else:
            return grouper.reindex(axis)._values
    elif isinstance(grouper, MultiIndex):
        return grouper._values
    elif isinstance(grouper, (list, tuple, Index, Categorical, np.ndarray)):
        if len(grouper) != len(axis):
            raise ValueError("Grouper and axis must be same length")

        if isinstance(grouper, (list, tuple)):
            grouper = com.asarray_tuplesafe(grouper)
        return grouper
    else:
        return grouper
