# [MODULE DOC]
# Time-based resampling of Series and DataFrame objects (the machinery behind the
# resample method) and the time grouper that defines the bins.
# Public contents: Resampler (base class: a groupby-like object over time bins, with
# aggregation and fill methods; exposed in the public typing namespace), the concrete
# resamplers for datetime, period and timedelta indexes, the grouping-aware variants
# used when resampling inside a groupby, and the time grouper class that stores the
# resample options (frequency, closed side, label side, origin, offset, convention).
# Also provides the factory function that builds the right resampler for an object, the
# as-frequency conversion function (backing the asfreq method), and helpers for
# computing bin edges, first/last timestamps and period conversions.
# Relationships: builds on groupby machinery (bin-based grouper and groupby base class),
# the aggregation-apply helper, datetime/period/timedelta index types and date range
# builders, and the offsets/frequency utilities from the tseries package.
# No import-time side effects other than defining the classes (and an empty dictionary
# of shared docstring kwargs).
from __future__ import annotations

import copy
from typing import (
    TYPE_CHECKING,
    Concatenate,
    Literal,
    Self,
    cast,
    final,
    no_type_check,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas._typing import NDFrameT
from pandas.errors import (
    AbstractMethodError,
    Pandas4Warning,
)
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.algorithms as algos
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
from pandas.core.generic import (
    NDFrame,
)
from pandas.core.groupby.groupby import (
    BaseGroupBy,
    GroupBy,
    get_groupby,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.period import (
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)
from pandas.core.reshape.concat import concat

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)
from pandas.tseries.offsets import (
    Day,
    Tick,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
    )

    from pandas._typing import (
        Any,
        AnyArrayLike,
        Axis,
        FreqIndexT,
        Frequency,
        IndexLabel,
        InterpolateOptions,
        P,
        T,
        TimedeltaConvertibleTypes,
        TimeGrouperOrigin,
        TimestampConvertibleTypes,
        TimeUnit,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame

_shared_docs_kwargs: dict[str, str] = {}


@set_module("pandas.api.typing")
class Resampler(BaseGroupBy, PandasObject):
    # [CLASS DOC]
    # Base class for resampling datetime-like data: a groupby-like object whose groups are
    # fixed time bins derived from a time grouper. Created by the resample method (through
    # the factory function), never directly by users.
    # Inherits from the groupby base class (selection and iteration behavior) and the
    # pandas object base.
    # State: the time grouper (stores the resample options), the object being resampled
    # (sorted by time) together with its time axis and the sort indexer used (if any), the
    # bin edges ("binner") and the bin-based grouper, the group_keys flag, an optional
    # column selection, and excluded columns (the "on" key column, if any). Resampler
    # option names (frequency, closed, label, convention, origin, offset) are readable
    # through attribute access delegated to the time grouper.
    # Key methods: aggregate (also named agg and apply), transform, pipe, fill methods
    # (ffill, bfill, nearest, interpolate, asfreq, fillna) and the reductions (sum, mean,
    # and so on). Subclasses must provide the bin computation, downsample and upsample.
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
    groupby : TimeGrouper

    Returns
    -------
    a Resampler of the appropriate type

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """

    _grouper: BinGrouper
    _timegrouper: TimeGrouper
    binner: DatetimeIndex | TimedeltaIndex | PeriodIndex  # depends on subclass
    exclusions: frozenset[Hashable] = frozenset()  # for SelectionMixin compat
    _internal_names_set = set({"obj", "ax", "_indexer"})

    # to the groupby descriptor
    _attributes = [
        "freq",
        "closed",
        "label",
        "convention",
        "origin",
        "offset",
    ]

    def __init__(
        self,
        obj: NDFrame,
        timegrouper: TimeGrouper,
        *,
        gpr_index: Index,
        group_keys: bool = False,
        selection=None,
        include_groups: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Builds a resampler.
        # Inputs: obj (the Series or DataFrame to resample), timegrouper (the time grouper
        # holding the resample options), gpr_index (the time axis used for grouping;
        # keyword-only), group_keys (keyword-only; default false), selection (an optional
        # column selection; default none), include_groups (keyword-only; must be false,
        # otherwise a value error is raised).
        # Effects: stores the grouper; sets keys to none, sorting on, index-as-keys on;
        # converts and sorts the object along the time axis through the grouper (storing the
        # object, its axis and the sorting indexer); computes the bin edges and bin grouper;
        # stores the selection; and records the grouping key column, if the grouper has one,
        # as an excluded column.
        if include_groups:
            raise ValueError("include_groups=True is no longer allowed.")
        self._timegrouper = timegrouper
        self.keys = None
        self.sort = True
        self.group_keys = group_keys
        self.as_index = True

        self.obj, self.ax, self._indexer = self._timegrouper._set_grouper(
            self._convert_obj(obj), sort=True, gpr_index=gpr_index
        )
        self.binner, self._grouper = self._get_binner()
        self._selection = selection
        if self._timegrouper.key is not None:
            self.exclusions = frozenset([self._timegrouper.key])
        else:
            self.exclusions = frozenset()

    @final
    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns a short text representation: the class name followed, in square brackets,
        # by the comma-separated name=value pairs for each resample option (frequency,
        # closed, label, convention, origin, offset) that is not none on the time grouper.
        """
        Provide a nice str repr of our rolling object.
        """
        attrs = (
            f"{k}={getattr(self._timegrouper, k)}"
            for k in self._attributes
            if getattr(self._timegrouper, k, None) is not None
        )
        return f"{type(self).__name__} [{', '.join(attrs)}]"

    @final
    def __getattr__(self, attr: str):
        # [FUNCTION DOC]
        # Attribute fallback, called only when normal lookup fails. Names in the internal
        # names set are looked up normally (which raises the usual attribute error if
        # absent); names of resample options are read from the time grouper; names that are
        # labels of the resampled object give the selection of that column (a new
        # resampler); anything else is looked up normally, which raises an attribute error.
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self._attributes:
            return getattr(self._timegrouper, attr)
        if attr in self.obj:
            return self[attr]

        return object.__getattribute__(self, attr)

    @final
    @property
    def _from_selection(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true when the grouping axis comes from a DataFrame column or
        # an index level (the time grouper has a key or a level) rather than the index
        # itself. Used to reject unsupported operations (upsampling, period resampling) in
        # that case.
        """
        Is the resampling from a DataFrame column or MultiIndex level.
        """
        # upsampling and PeriodIndex resampling do not work
        # with selection, this state used to catch and raise an error
        return self._timegrouper is not None and (
            self._timegrouper.key is not None or self._timegrouper.level is not None
        )

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        # [FUNCTION DOC]
        # HELPER: hook for subclasses to convert the object before grouping; the base
        # implementation returns the object after consolidating its internal blocks.
        """
        Provide any conversions for the object in order to correctly handle.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Series or DataFrame
        """
        return obj._consolidate()

    def _get_binner_for_time(self):
        # [FUNCTION DOC]
        # HELPER: abstract hook. Subclasses return the bin edges, the integer positions
        # that split the data into bins, and the bin labels. The base implementation raises
        # the abstract-method error.
        raise AbstractMethodError(self)

    @final
    def _get_binner(self):
        # [FUNCTION DOC]
        # HELPER: builds the bin edges and the bin-based grouper. Requires that the grouper
        # was already set on the object. Gets edges, bin positions and labels from the
        # subclass hook (asserting equal numbers of bins and labels), converts the labels to
        # the Arrow dtype of the time axis when the time grouper recorded one, and creates
        # the bin grouper using the sort indexer. Returns (edges, bin grouper).
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
        binner, bins, binlabels = self._get_binner_for_time()
        assert len(bins) == len(binlabels)
        if self._timegrouper._arrow_dtype is not None:
            binlabels = binlabels.astype(self._timegrouper._arrow_dtype)
        bin_grouper = BinGrouper(bins, binlabels, indexer=self._indexer)
        return binner, bin_grouper

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        # [FUNCTION DOC]
        # Typing overload only: pipe with a callable that takes the resampler as first
        # argument. See the final implementation of pipe.
        ...

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Typing overload only: pipe with a (callable, keyword name) pair, where the
        # resampler is passed under the given keyword. See the final implementation of pipe.
        ...

    @final
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Applies a function to this resampler and returns its result, enabling chaining.
        # Inputs: func (callable receiving the resampler first, or a pair of a callable and
        # the keyword name under which the resampler is passed), plus extra positional and
        # keyword arguments forwarded to it. Delegates to the groupby base implementation.
        # Returns whatever func returns.
        """
        Apply a ``func`` with arguments to this Resampler object and return its result.

        Use `.pipe` when you want to improve readability by chaining together
        functions that expect Series, DataFrames, GroupBy or Resampler objects.
        Instead of writing

        >>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
        >>> g = lambda x, arg1: x * 5 / arg1
        >>> f = lambda x: x**4
        >>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])
        >>> h(g(f(df.groupby("group")), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

        You can write

        >>> (
        ...     df.groupby("group").pipe(f).pipe(g, arg1=1).pipe(h, arg2=2, arg3=3)
        ... )  # doctest: +SKIP

        which is much more readable.

        Parameters
        ----------
        func : callable or tuple of (callable, str)
            Function to apply to this Resampler object or, alternatively,
            a `(callable, data_keyword)` tuple where `data_keyword` is a
            string indicating the keyword of `callable` that expects the
            Resampler object.
        *args : iterable, optional
            Positional arguments passed into `func`.
        **kwargs : dict, optional
                A dictionary of keyword arguments passed into `func`.

        Returns
        -------
        any
            The result of applying ``func`` to the Resampler object.

        See Also
        --------
        Series.pipe : Apply a function with arguments to a series.
        DataFrame.pipe: Apply a function with arguments to a dataframe.
        apply : Apply function to each group instead of to the
            full Resampler object.

        Notes
        -----
        See more `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": [1, 2, 3, 4]}, index=pd.date_range("2012-08-02", periods=4)
        ... )
        >>> df
                    A
        2012-08-02  1
        2012-08-03  2
        2012-08-04  3
        2012-08-05  4

        To get the difference between each 2-day period's maximum and minimum
        value in one pass, you can do

        >>> df.resample("2D").pipe(lambda x: x.max() - x.min())
                    A
        2012-08-02  1
        2012-08-04  1
        """
        return super().pipe(func, *args, **kwargs)

    @final
    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Aggregates each time bin with one or more operations.
        # Inputs: func (function, string name, list of these, or mapping from labels to
        # these; may be none when named aggregations are given via keyword arguments),
        # and extra positional and keyword arguments forwarded to func.
        # Tries the generic aggregation-application helper first (handles strings, lists,
        # mappings and named aggregation); if it yields nothing, falls back to the groupby
        # aggregation path with func as the operation (with fallback to apply for
        # non-reducing functions). Returns a scalar-like, Series or DataFrame indexed by the
        # bin labels. The names agg and apply are aliases of this method.
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        scalar, Series or DataFrame

            The return can be:

            * scalar : when Series.agg is called with single function
            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

        See Also
        --------
        DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
            or list of string/callables.
        DataFrame.resample.transform : Transforms the Series on each group
            based on the given function.
        DataFrame.aggregate: Aggregate using one or more
            operations over the specified axis.

        Notes
        -----
        The aggregation operations are always performed over an axis, either the
        index (default) or the column axis. This behavior is different from
        `numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
        `var`), where the default is to compute the aggregation of the flattened
        array, e.g., ``numpy.mean(arr_2d)`` as opposed to
        ``numpy.mean(arr_2d, axis=0)``.

        `agg` is an alias for `aggregate`. Use the alias.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        A passed user-defined-function will be passed a Series for evaluation.

        If ``func`` defines an index relabeling, ``axis`` must be ``0`` or ``index``.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 2, 3, 4, 5], index=pd.date_range("20130101", periods=5, freq="s")
        ... )
        >>> s
        2013-01-01 00:00:00    1
        2013-01-01 00:00:01    2
        2013-01-01 00:00:02    3
        2013-01-01 00:00:03    4
        2013-01-01 00:00:04    5
        Freq: s, dtype: int64

        >>> r = s.resample("2s")

        >>> r.agg("sum")
        2013-01-01 00:00:00    3
        2013-01-01 00:00:02    7
        2013-01-01 00:00:04    5
        Freq: 2s, dtype: int64

        >>> r.agg(["sum", "mean", "max"])
                            sum  mean  max
        2013-01-01 00:00:00    3   1.5    2
        2013-01-01 00:00:02    7   3.5    4
        2013-01-01 00:00:04    5   5.0    5

        >>> r.agg({"result": lambda x: x.mean() / x.std(), "total": "sum"})
                            result  total
        2013-01-01 00:00:00  2.121320      3
        2013-01-01 00:00:02  4.949747      7
        2013-01-01 00:00:04       NaN      5

        >>> r.agg(average="mean", total="sum")
                                average  total
        2013-01-01 00:00:00      1.5      3
        2013-01-01 00:00:02      3.5      7
        2013-01-01 00:00:04      5.0      5
        """
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            how = func
            result = self._groupby_and_aggregate(how, *args, **kwargs)

        return result

    agg = aggregate
    apply = aggregate

    @final
    def transform(self, arg, *args, **kwargs):
        # [FUNCTION DOC]
        # Applies a function to each time bin producing output indexed like the input.
        # Inputs: arg (function applied to each group, which must return like-indexed
        # output), plus extra arguments forwarded to it. Delegates to the transform of a
        # groupby built from the selected object and the time grouper. Returns the
        # transformed object (same index as the original).
        """
        Call function producing a like-indexed Series on each group.

        Return a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.
        *args, **kwargs
            Additional arguments and keywords.

        Returns
        -------
        Series
            A Series with the transformed values, maintaining the same index as
            the original object.

        See Also
        --------
        core.resample.Resampler.apply : Apply a function along each group.
        core.resample.Resampler.aggregate : Aggregate using one or more operations
            over the specified axis.

        Examples
        --------
        >>> s = pd.Series([1, 2], index=pd.date_range("20180101", periods=2, freq="1h"))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> resampled = s.resample("15min")
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        2018-01-01 00:00:00   NaN
        2018-01-01 01:00:00   NaN
        Freq: h, dtype: float64
        """
        return self._selected_obj.groupby(self._timegrouper).transform(
            arg, *args, **kwargs
        )

    def _downsample(self, how, **kwargs):
        # [FUNCTION DOC]
        # HELPER: abstract hook that subclasses implement to reduce each bin with the
        # named operation (how) and extra keyword arguments. The base implementation
        # raises the abstract-method error.
        raise AbstractMethodError(self)

    def _upsample(self, f, limit: int | None = None, fill_value=None):
        # [FUNCTION DOC]
        # HELPER: abstract hook that subclasses implement to convert to a higher frequency
        # with fill method f (such as forward fill, backward fill, nearest, or none), an
        # optional limit on filled values and an optional fill value. The base
        # implementation raises the abstract-method error.
        raise AbstractMethodError(self)

    def _gotitem(self, key, ndim: int, subset=None):
        # [FUNCTION DOC]
        # HELPER: returns a groupby object over a selection, used by the aggregation helpers.
        # Inputs: key (label or labels to select, or none), ndim (requested dimension, 1 or
        # 2), subset (object to act on; default is the resampler's object, narrowed by key
        # when key is not none; with no key the object must be one-dimensional).
        # Builds a groupby from the subset using this resampler's bin grouper and group_keys
        # setting; asserts that the subset is one-dimensional when ndim is one. Returns that
        # groupby object.
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        grouper = self._grouper
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                # reached via Apply.agg_dict_like with selection=None and ndim=1
                assert subset.ndim == 1
        if ndim == 1:
            assert subset.ndim == 1

        grouped = get_groupby(
            subset, by=None, grouper=grouper, group_keys=self.group_keys
        )
        return grouped

    def _groupby_and_aggregate(self, how, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: aggregates through an ordinary groupby over the bin grouper.
        # Inputs: how (callable or string/operation name) and extra arguments. Uses the
        # object without the excluded key column. A callable is wrapped in an anonymous
        # function that forwards the extra arguments (a TODO in the source notes that a test
        # fails when not using the wrapper). If aggregation fails with an attribute or key
        # error, or with a value error stating that an aggregated value must be produced,
        # the operation is evaluated with apply instead (any other value error is re-raised).
        # Returns the result after wrapping (naming, empty-result index handling and Arrow
        # dtype restoration).
        """
        Re-evaluate the obj with a groupby aggregation.
        """
        grouper = self._grouper

        # Excludes `on` column when provided
        obj = self._obj_with_exclusions

        grouped = get_groupby(obj, by=None, grouper=grouper, group_keys=self.group_keys)

        try:
            if callable(how):
                # TODO: test_resample_apply_with_additional_args fails if we go
                #  through the non-lambda path, not clear that it should.
                func = lambda x: how(x, *args, **kwargs)
                result = grouped.aggregate(func)
            else:
                result = grouped.aggregate(how, *args, **kwargs)
        except (AttributeError, KeyError):
            # we have a non-reducing function; try to evaluate
            # alternatively we want to evaluate only a column of the input

            # test_apply_to_one_column_of_df the function being applied references
            #  a DataFrame column, but aggregate_item_by_item operates column-wise
            #  on Series, raising AttributeError or KeyError
            #  (depending on whether the column lookup uses getattr/__getitem__)
            result = grouped.apply(how, *args, **kwargs)

        except ValueError as err:
            if "Must produce aggregated value" in str(err):
                # raised in _aggregate_named
                # see test_apply_without_aggregation, test_apply_with_mutated_index
                pass
            else:
                raise

            # we have a non-reducing function
            # try to evaluate
            result = grouped.apply(how, *args, **kwargs)

        return self._wrap_result(result)

    @final
    def _get_resampler_for_grouping(
        self,
        groupby: GroupBy,
        key,
    ):
        # [FUNCTION DOC]
        # HELPER: creates the resampler to use for groups in a groupby-then-resample
        # operation, by instantiating this class's grouping-aware resampler type with the
        # given groupby, key and this resampler as the parent. Returns that resampler.
        """
        Return the correct class for resampling with groupby.
        """
        return self._resampler_for_grouping(
            groupby=groupby,
            key=key,
            parent=self,
        )

    def _wrap_result(self, result):
        # [FUNCTION DOC]
        # HELPER: post-processes a result. If it is a Series and a column selection exists,
        # names the Series after the selection. If it is an empty Series (for example when
        # all index values are missing), gives it an empty index converted to this
        # resampler's frequency and the original object's name. If the time grouper recorded
        # an Arrow dtype, converts the result index to it and restores the original index
        # name. Modifies the result in place and returns it.
        """
        Potentially wrap any results.
        """
        if isinstance(result, ABCSeries) and self._selection is not None:
            result.name = self._selection

        if isinstance(result, ABCSeries) and result.empty:
            # When index is all NaT, result is empty but index is not
            obj = self.obj
            result.index = _asfreq_compat(obj.index[:0], freq=self.freq)
            result.name = getattr(obj, "name", None)

        if self._timegrouper._arrow_dtype is not None:
            result.index = result.index.astype(self._timegrouper._arrow_dtype)
            result.index.name = self.obj.index.name

        return result

    @final
    def ffill(self, limit: int | None = None):
        # [FUNCTION DOC]
        # Forward fills: when upsampling, new bins take the last valid observation; when
        # downsampling, takes the last value before each bin edge according to the
        # subclass's upsample behavior.
        # Input: limit (maximum number of consecutive new values to fill; default none).
        # Delegates to the subclass upsample hook with the forward-fill method. Returns the
        # resampled Series or DataFrame.
        """
        Forward fill the values.

        This method fills missing values by propagating the last valid
        observation forward, up to the next valid observation. It is commonly
        used in time series analysis when resampling data to a higher frequency
        (upsampling) and filling gaps in the resampled output.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series
            The resampled data with missing values filled forward.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.

        Examples
        --------
        Here we only create a ``Series``.

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        Example for ``ffill`` with downsampling (we have fewer dates after resampling):

        >>> ser.resample("MS").ffill()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64

        Example for ``ffill`` with upsampling (fill the new dates with
        the previous value):

        >>> ser.resample("W").ffill()
        2023-01-01    1
        2023-01-08    1
        2023-01-15    2
        2023-01-22    2
        2023-01-29    2
        2023-02-05    3
        2023-02-12    3
        2023-02-19    4
        Freq: W-SUN, dtype: int64

        With upsampling and limiting (only fill the first new date with the
        previous value):

        >>> ser.resample("W").ffill(limit=1)
        2023-01-01    1.0
        2023-01-08    1.0
        2023-01-15    2.0
        2023-01-22    2.0
        2023-01-29    NaN
        2023-02-05    3.0
        2023-02-12    NaN
        2023-02-19    4.0
        Freq: W-SUN, dtype: float64
        """
        return self._upsample("ffill", limit=limit)

    @final
    def nearest(self, limit: int | None = None):
        # [FUNCTION DOC]
        # Upsamples by filling newly created (missing) bins with the value of the nearest
        # original observation by index position. Existing missing values are not changed.
        # Input: limit (maximum number of values to fill in each direction around each
        # original value; default none). Delegates to the subclass upsample hook with the
        # nearest method. Returns the upsampled Series or DataFrame.
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        bfill : Backward fill the new missing values in the resampled data.
        ffill : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2], index=pd.date_range("20180101", periods=2, freq="1h"))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> s.resample("15min").nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15min, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample("15min").nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15min, dtype: float64
        """
        return self._upsample("nearest", limit=limit)

    @final
    def bfill(self, limit: int | None = None):
        # [FUNCTION DOC]
        # Upsamples by filling newly created (missing) bins with the next valid original
        # observation (backward fill). Existing missing values in the data are not modified.
        # Input: limit (maximum number of consecutive new values to fill; default none).
        # Delegates to the subclass upsample hook with the backward-fill method. Returns the
        # upsampled Series or DataFrame.
        """
        Backward fill the new missing values in the resampled data.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency). The backward fill will replace NaN values that appeared in
        the resampled data with the next value in the original sequence.
        Missing values that existed in the original data will not be modified.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series, DataFrame
            An upsampled Series or DataFrame with backward filled NaN values.

        See Also
        --------
        nearest : Fill NaN values with nearest neighbor starting from center.
        ffill : Forward fill NaN values.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'backfill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'backfill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_%28statistics%29

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series(
        ...     [1, 2, 3], index=pd.date_range("20180101", periods=3, freq="h")
        ... )
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: h, dtype: int64

        >>> s.resample("30min").bfill()
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample("15min").bfill(limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15min, dtype: float64

        Resampling a DataFrame that has missing values:

        >>> df = pd.DataFrame(
        ...     {"a": [2, np.nan, 6], "b": [1, 3, 5]},
        ...     index=pd.date_range("20180101", periods=3, freq="h"),
        ... )
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample("30min").bfill()
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5

        >>> df.resample("15min").bfill(limit=2)
                               a    b
        2018-01-01 00:00:00  2.0  1.0
        2018-01-01 00:15:00  NaN  NaN
        2018-01-01 00:30:00  NaN  3.0
        2018-01-01 00:45:00  NaN  3.0
        2018-01-01 01:00:00  NaN  3.0
        2018-01-01 01:15:00  NaN  NaN
        2018-01-01 01:30:00  6.0  5.0
        2018-01-01 01:45:00  6.0  5.0
        2018-01-01 02:00:00  6.0  5.0
        """
        return self._upsample("bfill", limit=limit)

    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area=None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Interpolates values at the resampled timestamps.
        # Inputs: method (interpolation technique, default linear; any method of the frame
        # interpolate function), axis (keyword-only, default 0), limit (maximum consecutive
        # missing values to fill), limit_direction (forward by default, or backward/both),
        # limit_area (none, inside or outside), and extra keyword arguments passed on to the
        # interpolation. A keyword named inplace is accepted only for backward compatibility:
        # it emits a Pandas4Warning deprecation warning, and a true value raises a value
        # error.
        # Algorithm: reindexes to the target bins (like asfreq); unless the index is a
        # period index, original timestamps that are not on the target grid are added back
        # (concatenated and sorted) so they act as interpolation anchors; interpolates;
        # then removes the extra anchor rows and restores the frequency-carrying target
        # index. For a period index the interpolated result is returned immediately. A
        # hierarchical result index raises a not-implemented error that suggests
        # interpolating per group instead.
        # Returns the interpolated Series or DataFrame.
        # KNOWN ISSUE: the docstring gives the default of axis as none, but the signature
        # default is 0.
        """
        Interpolate values between target timestamps according to different methods.

        The original index is first reindexed to target timestamps
        (see :meth:`core.resample.Resampler.asfreq`),
        then the interpolation of ``NaN`` values via :meth:`DataFrame.interpolate`
        happens.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. One of:

            * 'linear': Ignore the index and treat the values as equally
              spaced. This is the only method supported on MultiIndexes.
            * 'time': Works on daily and higher resolution data to interpolate
              given length of interval.
            * 'index', 'values': use the actual numerical values of the index.
            * 'pad': Fill in NaNs using existing values.
            * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
              'barycentric', 'polynomial': Passed to
              `scipy.interpolate.interp1d`, whereas 'spline' is passed to
              `scipy.interpolate.UnivariateSpline`. These methods use the numerical
              values of the index.  Both 'polynomial' and 'spline' require that
              you also specify an `order` (int), e.g.
              ``df.interpolate(method='polynomial', order=5)``. Note that,
              `slinear` method in Pandas refers to the Scipy first order `spline`
              instead of Pandas first order `spline`.
            * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
              'cubicspline': Wrappers around the SciPy interpolation methods of
              similar names. See `Notes`.
            * 'from_derivatives': Refers to
              `scipy.interpolate.BPoly.from_derivatives`.

        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to interpolate along. For `Series` this parameter is unused
            and defaults to 0.
        limit : int, optional
            Maximum number of consecutive NaNs to fill. Must be greater than
            0.
        limit_direction : {'forward', 'backward', 'both'}, Optional
            Consecutive NaNs will be filled in this direction.

        limit_area : {`None`, 'inside', 'outside'}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

        **kwargs : optional
            Keyword arguments to pass on to the interpolating function.

        Returns
        -------
        DataFrame or Series
            Interpolated values at the specified freq.

        See Also
        --------
        core.resample.Resampler.asfreq: Return the values at the new freq,
            essentially a reindex.
        DataFrame.interpolate: Fill NaN values using an interpolation method.
        DataFrame.bfill : Backward fill NaN values in the resampled data.
        DataFrame.ffill : Forward fill NaN values.

        Notes
        -----
        For high-frequent or non-equidistant time-series with timestamps
        the reindexing followed by interpolation may lead to information loss
        as shown in the last example.

        Examples
        --------

        >>> start = "2023-03-01T07:00:00"
        >>> timesteps = pd.date_range(start, periods=5, freq="s")
        >>> series = pd.Series(data=[1, -1, 2, 1, 3], index=timesteps)
        >>> series
        2023-03-01 07:00:00    1
        2023-03-01 07:00:01   -1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:03    1
        2023-03-01 07:00:04    3
        Freq: s, dtype: int64

        Downsample the dataframe to 0.5Hz by providing the period time of 2s.

        >>> series.resample("2s").interpolate("linear")
        2023-03-01 07:00:00    1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:04    3
        Freq: 2s, dtype: int64

        Upsample the dataframe to 2Hz by providing the period time of 500ms.

        >>> series.resample("500ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.500    0.0
        2023-03-01 07:00:01.000   -1.0
        2023-03-01 07:00:01.500    0.5
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.500    1.5
        2023-03-01 07:00:03.000    1.0
        2023-03-01 07:00:03.500    2.0
        2023-03-01 07:00:04.000    3.0
        Freq: 500ms, dtype: float64

        Internal reindexing with ``asfreq()`` prior to interpolation leads to
        an interpolated timeseries on the basis of the reindexed timestamps
        (anchors). It is assured that all available datapoints from original
        series become anchors, so it also works for resampling-cases that lead
        to non-aligned timestamps, as in the following example:

        >>> series.resample("400ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.000000
        2023-03-01 07:00:00.400    0.333333
        2023-03-01 07:00:00.800   -0.333333
        2023-03-01 07:00:01.200    0.000000
        2023-03-01 07:00:01.600    1.000000
        2023-03-01 07:00:02.000    2.000000
        2023-03-01 07:00:02.400    1.666667
        2023-03-01 07:00:02.800    1.333333
        2023-03-01 07:00:03.200    1.666667
        2023-03-01 07:00:03.600    2.333333
        2023-03-01 07:00:04.000    3.000000
        Freq: 400ms, dtype: float64

        Note that the series correctly decreases between two anchors
        ``07:00:00`` and ``07:00:02``.
        """
        if "inplace" in kwargs:
            # GH#58690
            warnings.warn(
                f"The 'inplace' keyword in {type(self).__name__}.interpolate "
                "is deprecated and will be removed in a future version. "
                "resample(...).interpolate is never inplace.",
                Pandas4Warning,
                stacklevel=find_stack_level(),
            )
            inplace = kwargs.pop("inplace")
            if inplace:
                raise ValueError("Cannot interpolate inplace on a resampled object.")

        result = self._upsample("asfreq")

        # If the original data has timestamps which are not aligned with the
        # target timestamps, we need to add those points back to the data frame
        # that is supposed to be interpolated. This does not work with
        # PeriodIndex, so we skip this case. GH#21351
        obj = self._selected_obj
        is_period_index = isinstance(obj.index, PeriodIndex)

        # Skip this step for PeriodIndex
        if not is_period_index:
            final_index = result.index
            if isinstance(final_index, MultiIndex):
                raise NotImplementedError(
                    "Direct interpolation of MultiIndex data frames is not "
                    "supported. If you tried to resample and interpolate on a "
                    "grouped data frame, please use:\n"
                    "`df.groupby(...).apply(lambda x: x.resample(...)."
                    "interpolate(...))`"
                    "\ninstead, as resampling and interpolation has to be "
                    "performed for each group independently."
                )

            missing_data_points_index = obj.index.difference(final_index)
            if len(missing_data_points_index) > 0:
                result = concat(
                    [result, obj.loc[missing_data_points_index]]
                ).sort_index()

        result_interpolated = result.interpolate(
            method=method,
            axis=axis,
            limit=limit,
            inplace=False,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )

        # No further steps if the original data has a PeriodIndex
        if is_period_index:
            return result_interpolated

        # Make sure that original data points which do not align with the
        # resampled index are removed
        result_interpolated = result_interpolated.loc[final_index]

        # Make sure frequency indexes are preserved
        result_interpolated.index = final_index
        return result_interpolated

    @final
    def asfreq(self, fill_value=None):
        # [FUNCTION DOC]
        # Returns the values at the new frequency, essentially a reindex onto the bin labels.
        # Input: fill_value (value for bins that are new when upsampling; does not alter
        # missing values already present; default none). Delegates to the subclass upsample
        # hook with the as-frequency method. Returns a Series or DataFrame.
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-31", "2023-02-01", "2023-02-28"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-31    2
        2023-02-01    3
        2023-02-28    4
        dtype: int64
        >>> ser.resample("MS").asfreq()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._upsample("asfreq", fill_value=fill_value)

    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        # [FUNCTION DOC]
        # Sums the values in each time bin. Inputs: numeric_only (default false; restrict
        # to numeric and boolean columns), min_count (default 0; minimum number of valid
        # values required per bin, otherwise the result is missing). Delegates to the
        # subclass downsample hook with the sum operation. Returns a Series or DataFrame
        # indexed by bin labels.
        """
        Compute sum of group values.

        This method provides a simple way to compute the sum of values within each
        resampled group, particularly useful for aggregating time-based data into
        daily, monthly, or yearly sums.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.

        See Also
        --------
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.count : Compute count of group, excluding missing
            values.
        DataFrame.resample : Resample time-series data.
        Series.sum : Return the sum of the values over the requested axis.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").sum()
        2023-01-01    3
        2023-02-01    7
        Freq: MS, dtype: int64
        """
        return self._downsample("sum", numeric_only=numeric_only, min_count=min_count)

    @final
    def prod(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        # [FUNCTION DOC]
        # Multiplies the values in each time bin. Inputs: numeric_only (default false),
        # min_count (default 0; minimum valid values per bin, otherwise missing). Delegates
        # to the subclass downsample hook with the product operation. Returns a Series or
        # DataFrame indexed by bin labels.
        """
        Compute prod of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed prod of values within each group.

        See Also
        --------
        core.resample.Resampler.sum : Compute sum of groups, excluding missing values.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").prod()
        2023-01-01    2
        2023-02-01   12
        Freq: MS, dtype: int64
        """
        return self._downsample("prod", numeric_only=numeric_only, min_count=min_count)

    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        # [FUNCTION DOC]
        # Takes the minimum value in each time bin. Inputs: numeric_only (default false),
        # min_count (default 0; minimum valid values per bin, otherwise missing). Delegates
        # to the subclass downsample hook with the minimum operation. Returns a Series or
        # DataFrame indexed by bin labels.
        """
        Compute min value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Compute the minimum value in the given Series or DataFrame.

        See Also
        --------
        core.resample.Resampler.max : Compute max value of group.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").min()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._downsample("min", numeric_only=numeric_only, min_count=min_count)

    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ):
        # [FUNCTION DOC]
        # Takes the maximum value in each time bin. Inputs: numeric_only (default false),
        # min_count (default 0; minimum valid values per bin, otherwise missing). Delegates
        # to the subclass downsample hook with the maximum operation. Returns a Series or
        # DataFrame indexed by bin labels.
        """
        Compute max value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computes the maximum value in the given Series or Dataframe.

        See Also
        --------
        core.resample.Resampler.min : Compute min value of group.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").max()
        2023-01-01    2
        2023-02-01    4
        Freq: MS, dtype: int64
        """
        return self._downsample("max", numeric_only=numeric_only, min_count=min_count)

    @final
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Takes the first entry of each column in each time bin (by default the first
        # non-missing one). Inputs: numeric_only (default false), min_count (default 0;
        # minimum valid values per bin), skipna (default true; skip missing values, so a
        # bin that is entirely missing gives missing). Delegates to the subclass downsample
        # hook with the first operation. Returns a Series or DataFrame.
        """
        Compute the first non-null entry of each column.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

        Returns
        -------
        Series or DataFrame
            First values within each group.

        See Also
        --------
        core.resample.Resampler.last : Compute the last non-null value in each group.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> s
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> s.resample("MS").first()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._downsample(
            "first", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ):
        # [FUNCTION DOC]
        # Takes the last entry of each column in each time bin (by default the last
        # non-missing one). Inputs: numeric_only (default false), min_count (default 0;
        # minimum valid values per bin), skipna (default true; skip missing values, so a
        # bin that is entirely missing gives missing). Delegates to the subclass downsample
        # hook with the last operation. Returns a Series or DataFrame.
        """
        Compute the last non-null entry of each column.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

        Returns
        -------
        Series or DataFrame
            Last of values within each group.

        See Also
        --------
        core.resample.Resampler.first : Compute the first non-null value in each group.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> s
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> s.resample("MS").last()
        2023-01-01    2
        2023-02-01    4
        Freq: MS, dtype: int64
        """
        return self._downsample(
            "last", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    def median(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Computes the median of each time bin, ignoring missing values. Input:
        # numeric_only (default false). Delegates to the subclass downsample hook with the
        # median operation. Returns a Series or DataFrame indexed by bin labels.
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to False.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.

        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 3, 4, 5],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").median()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64
        """
        return self._downsample("median", numeric_only=numeric_only)

    @final
    def mean(
        self,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Computes the mean of each time bin, ignoring missing values. Input: numeric_only
        # (default false). Delegates to the subclass downsample hook with the mean
        # operation. Returns a Series or DataFrame indexed by bin labels.
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Mean of values within each group.

        See Also
        --------
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.
        core.resample.Resampler.sum : Compute sum of groups, excluding missing values.
        core.resample.Resampler.std : Compute standard deviation of groups, excluding
            missing values.
        core.resample.Resampler.var : Compute variance of groups, excluding missing
            values.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").mean()
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        return self._downsample("mean", numeric_only=numeric_only)

    @final
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Computes the standard deviation of each time bin, ignoring missing values. Inputs:
        # ddof (delta degrees of freedom, default 1), numeric_only (default false).
        # Delegates to the subclass downsample hook with the standard-deviation operation
        # and the ddof. Returns a Series or DataFrame indexed by bin labels.
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Standard deviation of values within each group.

        See Also
        --------
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.
        core.resample.Resampler.var : Compute variance of groups, excluding missing
            values.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").std()
        2023-01-01    1.000000
        2023-02-01    2.645751
        Freq: MS, dtype: float64
        """
        return self._downsample("std", ddof=ddof, numeric_only=numeric_only)

    @final
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Computes the variance of each time bin, ignoring missing values. Inputs: ddof
        # (delta degrees of freedom, default 1), numeric_only (default false). Delegates to
        # the subclass downsample hook with the variance operation and the ddof. Returns a
        # Series or DataFrame indexed by bin labels.
        """
        Compute variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Variance of values within each group.

        See Also
        --------
        core.resample.Resampler.std : Compute standard deviation of groups, excluding
            missing values.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").var()
        2023-01-01    1.0
        2023-02-01    7.0
        Freq: MS, dtype: float64

        >>> ser.resample("MS").var(ddof=0)
        2023-01-01    0.666667
        2023-02-01    4.666667
        Freq: MS, dtype: float64
        """
        return self._downsample("var", ddof=ddof, numeric_only=numeric_only)

    @final
    def sem(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Computes the standard error of the mean of each time bin, ignoring missing values.
        # Inputs: ddof (delta degrees of freedom, default 1), numeric_only (default false).
        # Delegates to the subclass downsample hook with the standard-error operation and
        # the ddof. Returns a Series or DataFrame indexed by bin labels.
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.

        See Also
        --------
        DataFrame.sem : Return unbiased standard error of the mean over requested axis.
        Series.sem : Return unbiased standard error of the mean over requested axis.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").sem()
        2023-01-01    0.577350
        2023-02-01    1.527525
        Freq: MS, dtype: float64
        """
        return self._downsample("sem", ddof=ddof, numeric_only=numeric_only)

    @final
    def ohlc(self):
        # [FUNCTION DOC]
        # Computes the open (first), high (maximum), low (minimum) and close (last) values
        # of each time bin, ignoring missing values. No inputs.
        # For an empty time axis, returns a copy of the object (without excluded columns)
        # whose index is an empty index in the resampler frequency and whose columns are the
        # four names (for a Series: those four as columns; for a frame: a hierarchical
        # column index of each original column crossed with the four names), reindexed so
        # all values are missing. Otherwise delegates to the downsample hook with the
        # open-high-low-close operation. Returns a DataFrame.
        """
        Compute open, high, low and close values of a group, excluding missing values.

        Returns
        -------
        DataFrame
            Open, high, low and close values within each group.

        See Also
        --------
        DataFrame.agg : Aggregate using one or more operations over the specified axis.
        DataFrame.resample : Resample time-series data.
        DataFrame.groupby : Group DataFrame using a mapper or by a Series of columns.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 5],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").ohlc()
                    open  high  low  close
        2023-01-01     1     3    1      2
        2023-02-01     4     5    3      5
        """
        ax = self.ax
        obj = self._obj_with_exclusions
        if len(ax) == 0:
            # GH#42902
            obj = obj.copy()
            obj.index = _asfreq_compat(obj.index, self.freq)
            if obj.ndim == 1:
                obj = obj.to_frame()
                obj = obj.reindex(["open", "high", "low", "close"], axis=1)
            else:
                mi = MultiIndex.from_product(
                    [obj.columns, ["open", "high", "low", "close"]]
                )
                obj = obj.reindex(mi, axis=1)
            return obj

        return self._downsample("ohlc")

    @final
    def nunique(self):
        # [FUNCTION DOC]
        # Counts the distinct values in each time bin. No inputs. Delegates to the
        # downsample hook with the distinct-count operation. Returns a Series (or DataFrame
        # for frames) indexed by bin labels.
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.

        See Also
        --------
        core.groupby.SeriesGroupBy.nunique : Method nunique for SeriesGroupBy.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 3],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    3
        dtype: int64
        >>> ser.resample("MS").nunique()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        return self._downsample("nunique")

    @final
    def size(self):
        # [FUNCTION DOC]
        # Computes the number of rows in each time bin (including missing values). No
        # inputs. Delegates to the downsample hook with the size operation; if that yields a
        # non-empty DataFrame (frame input), the result is stacked into a Series. If the
        # time axis is empty, returns an empty 64-bit integer Series with the result index
        # and the name of the selected object when it is a Series (otherwise no name).
        # Returns a Series.
        """
        Compute group sizes.

        Returns
        -------
        Series
            Number of rows in each group.

        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby to each row
            or column of a DataFrame.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3],
        ...     index=pd.DatetimeIndex(["2023-01-01", "2023-01-15", "2023-02-01"]),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        dtype: int64
        >>> ser.resample("MS").size()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        result = self._downsample("size")

        # If the result is a non-empty DataFrame we stack to get a Series
        # GH 46826
        if isinstance(result, ABCDataFrame) and not result.empty:
            result = result.stack()

        if not len(self.ax):
            from pandas import Series

            if self._selected_obj.ndim == 1:
                name = self._selected_obj.name
            else:
                name = None
            result = Series([], index=result.index, dtype="int64", name=name)
        return result

    @final
    def count(self):
        # [FUNCTION DOC]
        # Counts the non-missing values in each time bin. No inputs. Delegates to the
        # downsample hook with the count operation. If the time axis is empty, the result is
        # replaced by an empty 64-bit integer object of the same kind (a Series carrying
        # the selected object's name, or a DataFrame with the result's columns) on the
        # result index. Returns a Series or DataFrame.
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series or DataFrame
            Count of values within each group.

        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby to each row
            or column of a DataFrame.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").count()
        2023-01-01    2
        2023-02-01    2
        Freq: MS, dtype: int64
        """
        result = self._downsample("count")
        if not len(self.ax):
            if self._selected_obj.ndim == 1:
                result = type(self._selected_obj)(
                    [], index=result.index, dtype="int64", name=self._selected_obj.name
                )
            else:
                from pandas import DataFrame

                result = DataFrame(
                    [], index=result.index, columns=result.columns, dtype="int64"
                )

        return result

    @final
    def quantile(self, q: float | list[float] | AnyArrayLike = 0.5, **kwargs):
        # [FUNCTION DOC]
        # Computes the quantile(s) of each time bin. Inputs: q (a number between 0 and 1,
        # or a list/array of such numbers; default 0.5), and extra keyword arguments passed
        # to the quantile computation. Delegates to the downsample hook with the quantile
        # operation. Returns a Series or DataFrame indexed by bin labels (with an extra
        # level for multiple quantiles).
        """
        Return value at the given quantile.

        Computes the quantile of values within each resampled group.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value between 0 <= q <= 1, the quantile(s) to compute.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the columns are groupby columns,
            and the values are its quantiles.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").quantile()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64

        >>> ser.resample("MS").quantile(0.25)
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        return self._downsample("quantile", q=q, **kwargs)


class _GroupByMixin(PandasObject, SelectionMixin):
    # [CLASS DOC]
    # HELPER: mixin for the resamplers returned by groupby-then-resample. It changes a
    # resampler so that every upsampling, downsampling or aggregation request is run once
    # per group of an underlying groupby.
    # Inherits from the pandas object base and the selection mixin; combined (with a
    # resampler class first in the inheritance order) by the three concrete groupby
    # resampler classes.
    # State: copies of the parent's resample option attributes, the selection, bin edges,
    # grouping key, the underlying groupby object, a shallow copy of the parent's time
    # grouper, and the parent's time axis and object.
    # The class-level names for upsample, downsample and groupby-and-aggregate are all
    # aliases of the single apply method defined here.
    """
    Provide the groupby facilities.
    """

    _attributes: list[str]  # in practice the same as Resampler._attributes
    _selection: IndexLabel | None = None
    _groupby: GroupBy
    _timegrouper: TimeGrouper

    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key=None,
        selection: IndexLabel | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds a grouping-aware resampler from a parent resampler.
        # Inputs (all keyword-only): parent (a resampler; asserted), groupby (the groupby
        # over the object; asserted to be a groupby), key (the grouping key, default none),
        # selection (column selection, default none).
        # Copies every resample option attribute from the parent, stores selection, bin
        # edges, key and groupby, takes a shallow copy of the parent's time grouper, and
        # shares the parent's time axis and object.
        # reached via ._gotitem and _get_resampler_for_grouping

        assert isinstance(groupby, GroupBy), type(groupby)

        # parent is always a Resampler, sometimes a _GroupByMixin
        assert isinstance(parent, Resampler), type(parent)

        # initialize our GroupByMixin object with
        # the resampler attributes
        for attr in self._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection

        self.binner = parent.binner
        self.key = key

        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)

        self.ax = parent.ax
        self.obj = parent.obj

    @no_type_check
    def _apply(self, f, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: runs a resampling operation separately for each group of the underlying
        # groupby and combines the results (this single method serves as upsample,
        # downsample and groupby-and-aggregate).
        # Inputs: f (operation name as a string, or a callable), extra positional arguments
        # (used only for callables) and keyword arguments (forwarded to named operations
        # and to callables). For each group, builds a resampler of the group-level class
        # on the group data with a copy of the time grouper and the shared time axis; a
        # string operation is invoked by name with the keyword arguments only (the
        # positional arguments are not forwarded), a callable is applied with all arguments.
        # An empty frame result without a period index gets the empty frequency-based index
        # appended as an extra index level. Returns the combined result after wrapping.
        # KNOWN ISSUE: for a string operation the extra positional arguments are silently
        # ignored; only keyword arguments reach the named operation.
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """

        def func(x):
            # [FUNCTION DOC]
            # HELPER: nested in the per-group apply; applies the requested operation to one
            # group x. Wraps x in the group-level resampler class (using the copied time
            # grouper and the shared time axis), then calls the named operation with the
            # keyword arguments, or applies the callable with all arguments. Returns the
            # per-group result.
            x = self._resampler_cls(x, timegrouper=self._timegrouper, gpr_index=self.ax)

            if isinstance(f, str):
                return getattr(x, f)(**kwargs)

            return x.apply(f, *args, **kwargs)

        result = self._groupby.apply(func)

        # GH 47705
        if (
            isinstance(result, ABCDataFrame)
            and len(result) == 0
            and not isinstance(result.index, PeriodIndex)
        ):
            result = result.set_index(
                _asfreq_compat(self.obj.index[:0], freq=self.freq), append=True
            )

        return self._wrap_result(result)

    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

    @final
    def _gotitem(self, key, ndim, subset=None):
        # [FUNCTION DOC]
        # HELPER: returns a new grouping-aware resampler restricted to a selection.
        # Inputs: key (label or list of labels; when a list, the grouping key is appended to
        # it if missing, mutating the passed list), ndim (requested dimension; unused),
        # subset (object to act on; default is the stored object narrowed by key; with no
        # key the object must be one-dimensional).
        # Selects the key from the underlying groupby (falling back to the whole groupby on
        # an index error), infers the selection, and builds a new instance of the same
        # class with that groupby, this object as parent, and the selection. Returns it.
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # create a new object to prevent aliasing
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                # reached via Apply.agg_dict_like with selection=None, ndim=1
                assert subset.ndim == 1

        # Try to select from a DataFrame, falling back to a Series
        try:
            if isinstance(key, list) and self.key not in key and self.key is not None:
                key.append(self.key)
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby

        selection = self._infer_selection(key, subset)

        new_rs = type(self)(
            groupby=groupby,
            parent=cast(Resampler, self),
            selection=selection,
        )
        return new_rs


class DatetimeIndexResampler(Resampler):
    # [CLASS DOC]
    # Resampler for objects whose time axis is a datetime index (the most common case);
    # also the base class of the period and time-delta resamplers.
    # Inherits the full resampler interface. Adds: the bin computation (delegated to the
    # time grouper's datetime bins), downsampling by an ordinary groupby aggregation over
    # the bin grouper, upsampling by reindexing the object onto the bin labels with an
    # optional fill method, a binner adjustment for upsampling that drops one boundary
    # edge, and result wrapping that converts the index back to periods when the original
    # axis was a period index.
    # State: the ax attribute is typed as the datetime index.
    ax: DatetimeIndex

    @property
    def _resampler_for_grouping(self) -> type[DatetimeIndexResamplerGroupby]:
        # [FUNCTION DOC]
        # HELPER: property giving the class used when resampling inside a groupby: the
        # datetime-index grouping-aware resampler class.
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self):
        # [FUNCTION DOC]
        # HELPER: computes the bins from the time grouper's datetime bin routine applied to
        # this resampler's time axis. Returns (bin edges, bin positions, bin labels).
        # this is how we are actually creating the bins
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(self, how, **kwargs):
        # [FUNCTION DOC]
        # HELPER: reduces each bin with the named operation.
        # Inputs: how (operation name or reducer), extra keyword arguments forwarded to it.
        # Uses the object without the excluded key column. If the time axis is empty,
        # returns a copy whose index is the empty index carrying the resampler's frequency
        # (asserting that the frequency was set). Otherwise groups by the bin grouper,
        # aggregates, and returns the wrapped result.
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        ax = self.ax

        # Excludes `on` column when provided
        obj = self._obj_with_exclusions

        if not len(ax):
            # reset to the new freq
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert obj.index.freq == self.freq, (obj.index.freq, self.freq)
            return obj

        # we are downsampling
        # we want to call the actual grouper method here
        result = obj.groupby(self._grouper).aggregate(how, **kwargs)
        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner):
        # [FUNCTION DOC]
        # HELPER: trims the bin edges to the labels used when upsampling so the new index
        # stays inside the specified range: drops the first edge when bins are closed on the
        # right, otherwise drops the last edge. Returns the trimmed edges.
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
        if self.closed == "right":
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner

    def _upsample(self, method, limit: int | None = None, fill_value=None):
        # [FUNCTION DOC]
        # HELPER: converts to a higher frequency by reindexing onto the new bin labels.
        # Inputs: method (fill method such as forward fill, backward fill, nearest, or the
        # as-frequency marker meaning no fill), limit (maximum consecutive fills), fill_value
        # (value for new rows).
        # Raises a value error when the grouping comes from a column or index level
        # selection (upsampling there is unsupported). The new index is the adjusted bin
        # edges. Fast path: if no limit is given and the original index already equals the
        # target grid (same inferred frequency, length and values), the object is just copied
        # with the new index. Otherwise reindexes the selected object onto the new index with
        # the method (none for the as-frequency marker), limit and fill value.
        # Returns the wrapped result.
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values
        """
        if self._from_selection:
            raise ValueError(
                "Upsampling from level= or on= selection "
                "is not supported, use .set_index(...) "
                "to explicitly set index to datetime-like"
            )

        ax = self.ax
        obj = self._selected_obj
        binner = self.binner
        res_index = self._adjust_binner_for_upsample(binner)

        # if index exactly matches target grid (same freq & alignment), use fast path
        if (
            limit is None
            and to_offset(ax.inferred_freq) == self.freq
            and len(obj) == len(res_index)
            and obj.index.equals(res_index)
        ):
            result = obj.copy()
            result.index = res_index
        else:
            if method == "asfreq":
                method = None
            result = obj.reindex(
                res_index, method=method, limit=limit, fill_value=fill_value
            )

        return self._wrap_result(result)

    def _wrap_result(self, result):
        # [FUNCTION DOC]
        # HELPER: extends the base result wrapping. After the base processing, if the
        # original time axis is a period index but the result index is not, converts the
        # result index to periods with the resampler frequency (for a hierarchical index,
        # converts its last level, if that level is not already periods). Returns the result.
        result = super()._wrap_result(result)

        # we may have a different kind that we were asked originally
        # convert if needed
        if isinstance(self.ax, PeriodIndex) and not isinstance(
            result.index, PeriodIndex
        ):
            if isinstance(result.index, MultiIndex):
                # GH 24103 - e.g. groupby resample
                if not isinstance(result.index.levels[-1], PeriodIndex):
                    new_level = result.index.levels[-1].to_period(self.freq)
                    result.index = result.index.set_levels(new_level, level=-1)
            else:
                result.index = result.index.to_period(self.freq)
        return result


@set_module("pandas.api.typing")
# error: Definition of "ax" in base class "_GroupByMixin" is incompatible
# with definition in base class "DatetimeIndexResampler"
class DatetimeIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, DatetimeIndexResampler
):
    # [CLASS DOC]
    # Resampler returned when resampling is done inside a groupby on a datetime index:
    # combines the grouping-aware mixin (which runs each operation per group) with the
    # datetime-index resampler. No extra state; the per-group resampler class is the plain
    # datetime-index resampler.
    """
    Provides a resample of a groupby implementation
    """

    @property
    def _resampler_cls(self):
        # [FUNCTION DOC]
        # HELPER: property naming the class used to resample each group: the plain
        # datetime-index resampler.
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    # [CLASS DOC]
    # Resampler for objects whose time axis is a period index. Subclass of the datetime
    # resampler (inherits wrapping and the interface) with period-specific behavior:
    # bins come from the period bin routine, the object is rejected if the grouping comes
    # from a column or level, downsampling checks that the frequencies are compatible
    # sub/super periods, and upsampling maps through the periods converted to the target
    # frequency using the convention (start or end of period).
    # State: the ax attribute is typed as the period index.
    # error: Incompatible types in assignment (expression has type "PeriodIndex", base
    # class "DatetimeIndexResampler" defined the type as "DatetimeIndex")
    ax: PeriodIndex  # type: ignore[assignment]

    @property
    def _resampler_for_grouping(self):
        # [FUNCTION DOC]
        # HELPER: property giving the class used when resampling inside a groupby: the
        # period-index grouping-aware resampler class.
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self):
        # [FUNCTION DOC]
        # HELPER: computes the bins from the time grouper's period bin routine applied to
        # this resampler's period axis. Returns (bin edges, bin positions, bin labels).
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        # [FUNCTION DOC]
        # HELPER: extends the base conversion. After the base conversion, raises a
        # not-implemented error if the grouping comes from a column or index level
        # selection (unsupported with a period index); otherwise returns the object.
        obj = super()._convert_obj(obj)

        if self._from_selection:
            # see GH 14008, GH 12871
            msg = (
                "Resampling from level= or on= selection "
                "with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )
            raise NotImplementedError(msg)

        return obj

    def _downsample(self, how, **kwargs):
        # [FUNCTION DOC]
        # HELPER: reduces periods to a coarser (or equal or finer) period frequency.
        # Inputs: how (operation name) and keyword arguments forwarded to it.
        # If the target frequency is a coarser frequency (the data frequency is a sub-period
        # of it), aggregates through the groupby path. If the target is finer (the data
        # frequency is a super-period), the result is the as-frequency conversion, except for
        # the open-high-low-close operation, which still goes through the groupby path (the
        # reduction operation is otherwise ignored in this case). If the frequencies are
        # equal, returns the as-frequency conversion. Otherwise raises an
        # incompatible-frequency error saying the frequencies are not sub or super periods.
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        ax = self.ax

        if is_subperiod(ax.freq, self.freq):
            # Downsampling
            return self._groupby_and_aggregate(how, **kwargs)
        elif is_superperiod(ax.freq, self.freq):
            if how == "ohlc":
                # GH #13083
                # upsampling to subperiods is handled as an asfreq, which works
                # for pure aggregating/reducing methods
                # OHLC reduces along the time dimension, but creates multiple
                # values for each period -> handle by _groupby_and_aggregate()
                return self._groupby_and_aggregate(how)
            return self.asfreq()
        elif ax.freq == self.freq:
            return self.asfreq()

        raise IncompatibleFrequency(
            f"Frequency {ax.freq} cannot be resampled to {self.freq}, "
            "as they are not sub or super periods"
        )

    def _upsample(self, method, limit: int | None = None, fill_value=None):
        # [FUNCTION DOC]
        # HELPER: converts periods to a finer frequency by mapping the new bin labels onto
        # the original periods.
        # Inputs: method (fill method; the as-frequency marker means exact matches only),
        # limit (maximum gap to fill), fill_value (accepted but unused here).
        # Converts the original period axis to the target frequency (start or end of
        # period per the convention), computes an indexer from the new bin labels into those
        # periods with the fill method and limit, and takes the object's rows through it onto
        # the new labels (using the full stored object, not the column selection).
        # Returns the wrapped result.
        # KNOWN ISSUE: the fill_value argument is accepted but never used, so a fill value
        # has no effect for period upsampling; the docstring also omits the as-frequency
        # method and the nearest method that the code supports.
        """
        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.
        """
        ax = self.ax
        obj = self.obj
        new_index = self.binner

        # Start vs. end of period
        memb = ax.asfreq(self.freq, how=self.convention)

        # Get the fill indexer
        if method == "asfreq":
            method = None
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        new_obj = _take_new_index(
            obj,
            indexer,
            new_index,
        )
        return self._wrap_result(new_obj)


@set_module("pandas.api.typing")
# error: Definition of "ax" in base class "_GroupByMixin" is incompatible with
# definition in base class "PeriodIndexResampler"
class PeriodIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, PeriodIndexResampler
):
    # [CLASS DOC]
    # Resampler returned when resampling is done inside a groupby on a period index:
    # combines the grouping-aware mixin (running each operation per group) with the
    # period-index resampler. No extra state; the per-group class is the plain period-index
    # resampler.
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _resampler_cls(self):
        # [FUNCTION DOC]
        # HELPER: property naming the class used to resample each group: the plain
        # period-index resampler.
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    # [CLASS DOC]
    # Resampler for objects whose time axis is a time-delta index. Subclass of the datetime
    # resampler; differs only in how bins are computed (time-delta bins, which require a
    # fixed-duration frequency) and in not trimming the binner when upsampling.
    # State: the ax attribute is typed as the time-delta index.
    # error: Incompatible types in assignment (expression has type "TimedeltaIndex",
    # base class "DatetimeIndexResampler" defined the type as "DatetimeIndex")
    ax: TimedeltaIndex  # type: ignore[assignment]

    @property
    def _resampler_for_grouping(self):
        # [FUNCTION DOC]
        # HELPER: property giving the class used when resampling inside a groupby: the
        # time-delta grouping-aware resampler class.
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self):
        # [FUNCTION DOC]
        # HELPER: computes the bins from the time grouper's time-delta bin routine applied
        # to this resampler's axis. Returns (bin edges, bin positions, bin labels).
        return self._timegrouper._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner):
        # [FUNCTION DOC]
        # HELPER: override that leaves the bin edges unchanged when upsampling (the range of
        # the new index may exceed the original range). Returns the edges as given.
        """
        Adjust our binner when upsampling.

        The range of a new index is allowed to be greater than original range
        so we don't need to change the length of a binner, GH 13022
        """
        return binner


@set_module("pandas.api.typing")
# error: Definition of "ax" in base class "_GroupByMixin" is incompatible with
# definition in base class "DatetimeIndexResampler"
class TimedeltaIndexResamplerGroupby(  # type: ignore[misc]
    _GroupByMixin, TimedeltaIndexResampler
):
    # [CLASS DOC]
    # Resampler returned when resampling is done inside a groupby on a time-delta index:
    # combines the grouping-aware mixin (running each operation per group) with the
    # time-delta resampler. No extra state; the per-group class is the plain time-delta
    # resampler.
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _resampler_cls(self):
        # [FUNCTION DOC]
        # HELPER: property naming the class used to resample each group: the plain
        # time-delta resampler.
        return TimedeltaIndexResampler


def get_resampler(obj: Series | DataFrame, **kwds) -> Resampler:
    # [FUNCTION DOC]
    # Factory behind the resample method. Inputs: obj (Series or DataFrame) and keyword
    # arguments for the time grouper (frequency, closed, label, and so on). Builds a time
    # grouper from them and asks it for the matching resampler for the object's time axis.
    # Raises a type error (from the grouper) if the axis is not datetime-like, period or
    # time-delta. Returns the resampler.
    # Module side effect: right after this definition, the function's documentation
    # string is replaced by the resampler class documentation.
    """
    Create a TimeGrouper and return our resampler.
    """
    tg = TimeGrouper(obj, **kwds)  # type: ignore[arg-type]
    return tg._get_resampler(obj)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby: GroupBy,
    rule,
    how=None,
    fill_method=None,
    limit: int | None = None,
    on=None,
    **kwargs,
) -> Resampler:
    # [FUNCTION DOC]
    # Factory for groupby-then-resample (used by the groupby resample method).
    # Inputs: groupby (the groupby object), rule (frequency), how and fill_method and limit
    # (accepted but unused), on (column to resample on, used as the grouper key), and extra
    # keyword arguments for the time grouper. Builds a time grouper with the rule and key,
    # gets the resampler for the groupby's object, and converts it to the grouping-aware
    # resampler for the groupby. Returns that resampler.
    # KNOWN ISSUE: the parameters how, fill_method and limit are never used.
    """
    Return our appropriate resampler when grouping as well.
    """
    # .resample uses 'on' similar to how .groupby uses 'key'
    tg = TimeGrouper(freq=rule, key=on, **kwargs)
    resampler = tg._get_resampler(groupby.obj)
    return resampler._get_resampler_for_grouping(groupby=groupby, key=tg.key)


@set_module("pandas.api.typing")
class TimeGrouper(Grouper):
    # [CLASS DOC]
    # Grouper for time-interval grouping: holds all the options of a resample call and
    # computes the bin edges, positions and labels for datetime, period and time-delta axes.
    # Inherits from the general groupby grouper; its option names list is extended with
    # closed, label, how, convention, origin and offset.
    # State: freq (offset), closed and label sides (defaulting to the right for end-type
    # frequencies such as month-end, year-end, quarter-end, business variants and weekly,
    # and when origin is the end or end of day; otherwise left), convention (start or end
    # of period; defaults to end), how, fill_method, limit, group_keys, origin (a keyword
    # or a timestamp), offset (a time-delta or none), and an optional Arrow dtype recorded
    # when the axis was Arrow-backed. Sorting is always on.
    # Key methods: constructor validation, resampler creation, bin computation per axis
    # kind, bin edge adjustment, and grouper setting with Arrow conversion.
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """

    _attributes = (
        *Grouper._attributes,
        "closed",
        "label",
        "how",
        "convention",
        "origin",
        "offset",
    )

    origin: TimeGrouperOrigin

    def __init__(
        self,
        obj: Grouper | None = None,
        freq: Frequency = "Min",
        key: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        how: str = "mean",
        fill_method=None,
        limit: int | None = None,
        convention: Literal["start", "end", "e", "s"] | None = None,
        origin: (
            Literal["epoch", "start", "start_day", "end", "end_day"]
            | TimestampConvertibleTypes
        ) = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Validates the resample options and stores them.
        # Inputs: obj (the object being resampled, used only to detect period-valued
        # axes or key columns; default none), freq (offset or alias; default one minute),
        # key (column to resample on), closed and label (left or right or none for the
        # default described in the class doc), how (default mean), fill_method and limit
        # (stored), convention (start, end, e or s, or none meaning end), origin (start_day
        # by default; epoch, start, start_day, end, end_day, or a timestamp-convertible
        # value), offset (time-delta convertible or none), group_keys (default false), and
        # extra keyword arguments for the base grouper (sorting is always forced on).
        # Errors (value errors): unsupported label, closed or convention values; origin that
        # is neither a keyword nor timestamp-convertible; offset that is not time-delta
        # convertible. Warnings (runtime warning category): offset given, or origin
        # different from start_day, when the frequency is not a fixed tick-like one (they
        # have no effect then).
        # The frequency is parsed as a period offset when the object's index (or key column)
        # is a period type.
        # Checks for correctness of the keyword arguments which would
        # otherwise silently use the default if misspelled
        # otherwise silently use the default if misspelled
        if label not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {label} for `label`")
        if closed not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {closed} for `closed`")
        if convention not in {None, "start", "end", "e", "s"}:
            raise ValueError(f"Unsupported value {convention} for `convention`")

        if (key is None and obj is not None and isinstance(obj.index, PeriodIndex)) or (  # type: ignore[attr-defined]
            key is not None
            and obj is not None
            and getattr(obj[key], "dtype", None) == "period"  # type: ignore[index]
        ):
            freq = to_offset(freq, is_period=True)
        else:
            freq = to_offset(freq)

        if not isinstance(freq, Tick):
            if offset is not None:
                warnings.warn(
                    "The 'offset' keyword does not take effect when resampling "
                    "with a 'freq' that is not Tick-like (h, m, s, ms, us, ns)",
                    RuntimeWarning,
                    stacklevel=find_stack_level(),
                )
            if origin != "start_day":
                warnings.warn(
                    "The 'origin' keyword does not take effect when resampling "
                    "with a 'freq' that is not Tick-like (h, m, s, ms, us, ns)",
                    RuntimeWarning,
                    stacklevel=find_stack_level(),
                )

        end_types = {"ME", "YE", "QE", "BME", "BYE", "BQE", "W"}
        rule = freq.rule_code
        if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        # The backward resample sets ``closed`` to ``'right'`` by default
        # since the last value should be considered as the edge point for
        # the last bin. When origin in "end" or "end_day", the value for a
        # specific ``Timestamp`` index stands for the resample result from
        # the current ``Timestamp`` minus ``freq`` to the current
        # ``Timestamp`` with a right close.
        elif origin in ["end", "end_day"]:
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        else:
            if closed is None:
                closed = "left"
            if label is None:
                label = "left"

        self.closed = closed
        self.label = label
        self.convention = convention if convention is not None else "e"
        self.how = how
        self.fill_method = fill_method
        self.limit = limit
        self.group_keys = group_keys
        self._arrow_dtype: ArrowDtype | None = None

        if origin in ("epoch", "start", "start_day", "end", "end_day"):
            # error: Incompatible types in assignment (expression has type "Union[Union[
            # Timestamp, datetime, datetime64, signedinteger[_64Bit], float, str],
            # Literal['epoch', 'start', 'start_day', 'end', 'end_day']]", variable has
            # type "Union[Timestamp, Literal['epoch', 'start', 'start_day', 'end',
            # 'end_day']]")
            self.origin = origin  # type: ignore[assignment]
        else:
            try:
                self.origin = Timestamp(origin)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "'origin' should be equal to 'epoch', 'start', 'start_day', "
                    "'end', 'end_day' or "
                    f"should be a Timestamp convertible type. Got '{origin}' instead."
                ) from err

        try:
            self.offset = Timedelta(offset) if offset is not None else None
        except (ValueError, TypeError) as err:
            raise ValueError(
                "'offset' should be a Timedelta convertible type. "
                f"Got '{offset}' instead."
            ) from err

        # always sort time groupers
        kwargs["sort"] = True

        super().__init__(freq=freq, key=key, **kwargs)

    def _get_resampler(self, obj: NDFrame) -> Resampler:
        # [FUNCTION DOC]
        # HELPER: creates the resampler matching the object's time axis. Sets the grouper
        # on the object to find the axis, then returns a datetime, period or time-delta
        # index resampler (passing this grouper, the group_keys option and the axis).
        # Raises a type error naming the axis type if it is none of those.
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
        _, ax, _ = self._set_grouper(obj, gpr_index=None)
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, PeriodIndex):
            return PeriodIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )

        raise TypeError(
            "Only valid with DatetimeIndex, "
            "TimedeltaIndex or PeriodIndex, "
            f"but got an instance of '{type(ax).__name__}'"
        )

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True, observed: bool = True
    ) -> tuple[BinGrouper, NDFrameT]:
        # [FUNCTION DOC]
        # HELPER: override used by groupby with a time grouper. Inputs: obj; validate and
        # observed (both unused, present for signature compatibility). Creates the
        # resampler for the object and returns its bin grouper together with its (possibly
        # sorted) object.
        """
        Parameters
        ----------
        obj : Series or DataFrame
            Object being grouped.
        validate : bool, default True
            Unused. Only for compatibility with ``Grouper._get_grouper``.
        observed : bool, default True
            Unused. Only for compatibility with ``Grouper._get_grouper``.

        Returns
        -------
        A tuple of grouper, obj (possibly sorted)
        """
        # create the resampler and return our binner
        r = self._get_resampler(obj)
        return r._grouper, cast(NDFrameT, r.obj)

    def _get_time_bins(self, ax: DatetimeIndex):
        # [FUNCTION DOC]
        # HELPER: computes the bins for a datetime axis.
        # Input: ax (must be a datetime index, otherwise a type error). For an empty axis
        # returns empty edges, an empty bin position list and empty labels, all with the
        # frequency.
        # Otherwise: finds the first and last bin edges from the axis minimum and maximum
        # (aligned to the frequency using closed, origin and offset), builds the full date
        # range between them (time-zone aware, shifting nonexistent times forward and
        # treating ambiguous times as the first), adjusts the edges for end-type
        # frequencies, and generates the bin positions from the integer values of the axis.
        # Labels: the edges, dropping the first edge when the label side is right; with
        # missing values in the axis, a missing-value bin is inserted at the front of both
        # edges and labels; if there are more labels than bins, the extra labels are cut.
        # Returns (edges, bin positions, labels).
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if len(ax) == 0:
            binner = labels = DatetimeIndex(
                data=[], freq=self.freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels

        first, last = _get_timestamp_range_edges(
            ax.min(),
            ax.max(),
            self.freq,
            unit=ax.unit,
            closed=self.closed,
            origin=self.origin,
            offset=self.offset,
        )
        # GH #12037
        # use first/last directly instead of call replace() on them
        # because replace() will swallow the nanosecond part
        # thus last bin maybe slightly before the end if the end contains
        # nanosecond part and lead to `Values falls after last bin` error
        # GH 25758: If DST lands at midnight (e.g. 'America/Havana'), user feedback
        # has noted that ambiguous=True provides the most sensible result
        binner = labels = date_range(
            freq=self.freq,
            start=first,
            end=last,
            tz=ax.tz,
            name=ax.name,
            ambiguous=True,
            nonexistent="shift_forward",
            unit=ax.unit,
        )

        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)

        # general version, knowing nothing about relative frequencies
        bins = lib.generate_bins_dt64(
            ax_values, bin_edges, self.closed, hasnans=ax.hasnans
        )

        if self.closed == "right":
            labels = binner
            if self.label == "right":
                labels = labels[1:]
        elif self.label == "right":
            labels = labels[1:]

        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)

        # if we end up with more labels than bins
        # adjust the labels
        # GH4076
        if len(bins) < len(labels):
            labels = labels[: len(bins)]

        return binner, bins, labels

    def _adjust_bin_edges(
        self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]
    ) -> tuple[DatetimeIndex, npt.NDArray[np.int64]]:
        # [FUNCTION DOC]
        # HELPER: computes the integer bin edges, with special handling for end-of-period
        # frequencies coarser than a day (month end, business month end, weekly, and
        # quarter and year end variants).
        # Inputs: binner (datetime edges) and ax_values (the axis as 64-bit integers).
        # For those frequencies when bins are closed on the right, each edge is moved to the
        # last instant of its day (computed on wall-clock time, then re-localized), so the
        # whole last day belongs to the bin; when closed left, the integer edges are used
        # unchanged. In both cases, if the second-to-last edge already exceeds the largest axis
        # value, the last edge (and binner entry) is dropped. Other frequencies use the
        # integer edges as they are.
        # Returns (possibly shortened binner, integer edges).
        # Some hacks for > daily data, see #1471, #1458, #1483

        if self.freq.name in ("BME", "ME", "W") or self.freq.name.split("-")[0] in (
            "BQE",
            "BYE",
            "QE",
            "YE",
            "W",
        ):
            # If the right end-point is on the last day of the month, roll forwards
            # until the last moment of that day. Note that we only do this for offsets
            # which correspond to the end of a super-daily period - "month start", for
            # example, is excluded.
            if self.closed == "right":
                # GH 21459, GH 9119: Adjust the bins relative to the wall time
                edges_dti = binner.tz_localize(None)
                edges_dti = (
                    edges_dti
                    + Timedelta(days=1).as_unit(edges_dti.unit)
                    - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                )
                bin_edges = edges_dti.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8

            # intraday values on last day
            if bin_edges[-2] > ax_values.max():
                bin_edges = bin_edges[:-1]
                binner = binner[:-1]
        else:
            bin_edges = binner.asi8
        return binner, bin_edges

    def _get_time_delta_bins(self, ax: TimedeltaIndex):
        # [FUNCTION DOC]
        # HELPER: computes the bins for a time-delta axis.
        # Input: ax (must be a time-delta index, otherwise a type error). The frequency must
        # be fixed-duration (a tick or a day), otherwise a value error. For an empty axis
        # returns empty edges, no bins and empty labels.
        # Otherwise builds the edges as a time-delta range from the axis minimum to its
        # maximum (extended by one frequency step when closed right), finds bin positions by
        # searching the axis for the end of each bin (end stamps are shifted by one step when
        # closed left), and adds the offset (if any) to the labels.
        # Returns (edges, bin positions, labels).
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(
                "axis must be a TimedeltaIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if not isinstance(self.freq, (Tick, Day)):
            # GH#51896
            raise ValueError(
                "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
                f"e.g. '24h' or '3D', not {self.freq}"
            )

        if not len(ax):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels

        start, end = ax.min(), ax.max()

        if self.closed == "right":
            end += self.freq

        labels = binner = timedelta_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        end_stamps = labels
        if self.closed == "left":
            end_stamps += self.freq

        bins = ax.searchsorted(end_stamps, side=self.closed)

        if self.offset:
            # GH 10530 & 31809
            labels += self.offset

        return binner, bins, labels

    def _get_time_period_bins(self, ax: DatetimeIndex):
        # [FUNCTION DOC]
        # HELPER: computes bins that are periods, for a datetime axis.
        # Input: ax (must be a datetime index, otherwise a type error). For an empty axis
        # returns empty period edges, no bins and empty labels. Otherwise builds a period range
        # from the first to the last axis value at the frequency, computes the timestamp
        # at the end of each period (localized to the axis time zone), and finds the bin
        # positions by searching the axis for those stamps (left side).
        # Returns (edges, bin positions, labels), where edges and labels are the same period
        # index.
        # KNOWN ISSUE: no code in the repository calls this method (dead code); the active
        # period bin routine is the one for period axes.
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        freq = self.freq

        if len(ax) == 0:
            binner = labels = PeriodIndex(
                data=[], freq=freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels

        labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)

        end_stamps = (labels + freq).asfreq(freq, "s").to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side="left")

        return binner, bins, labels

    def _get_period_bins(self, ax: PeriodIndex):
        # [FUNCTION DOC]
        # HELPER: computes the bins for a period axis.
        # Input: ax (must be a period index, otherwise a type error). Converts the axis to
        # the target frequency (by the convention); missing values are removed and counted.
        # If no valid values remain, returns empty edges and bins (with a single bin for the
        # missing values if the axis was not empty) and empty labels.
        # Otherwise: start is the first period at the target frequency (by convention) and end
        # is the last at the end; for tick-like frequencies the edges are aligned using origin
        # and offset, and a bin shift is derived from the difference between the original and
        # aligned start modulo the frequency multiple. Labels and edges are the period range
        # between start and end. Bin positions are found by searching the converted values
        # for the computed edge positions (enough bins are generated when upsampling to
        # sub-periods, and the shift is subtracted). If missing values existed, a leading bin
        # for them is inserted.
        # Returns (edges, bin positions, labels).
        if not isinstance(ax, PeriodIndex):
            raise TypeError(
                "axis must be a PeriodIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        memb = ax.asfreq(self.freq, how=self.convention)

        # NaT handling as in pandas._lib.lib.generate_bins_dt64()
        nat_count = 0
        if memb.hasnans:
            # error: Incompatible types in assignment (expression has type
            # "bool_", variable has type "int")  [assignment]
            nat_count = np.sum(memb._isnan)  # type: ignore[assignment]
            memb = memb[~memb._isnan]

        if not len(memb):
            # index contains no valid (non-NaT) values
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            if len(ax) > 0:
                # index is all NaT
                binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
            return binner, bins, labels

        freq_mult = self.freq.n

        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how="end")
        bin_shift = 0

        if isinstance(self.freq, Tick):
            # GH 23882 & 31809: get adjusted bin edge labels with 'origin'
            # and 'origin' support. This call only makes sense if the freq is a
            # Tick since offset and origin are only used in those cases.
            # Not doing this check could create an extra empty bin.
            p_start, end = _get_period_range_edges(
                start,
                end,
                self.freq,
                closed=self.closed,
                origin=self.origin,
                offset=self.offset,
            )

            # Get offset for bin edge (not label edge) adjustment
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            # error: Item "Period" of "Union[Period, Any]" has no attribute "n"
            bin_shift = start_offset.n % freq_mult  # type: ignore[union-attr]
            start = p_start

        labels = binner = period_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        i8 = memb.asi8

        # when upsampling to subperiods, we need to generate enough bins
        expected_bins_count = len(binner) * freq_mult
        i8_extend = expected_bins_count - (i8[-1] - i8[0])
        rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
        rng += freq_mult
        # adjust bin edge indexes to account for base
        rng -= bin_shift

        # Wrap in PeriodArray for PeriodArray.searchsorted
        prng = type(memb._data)(rng, dtype=memb.dtype)
        bins = memb.searchsorted(prng, side="left")

        if nat_count > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)

        return binner, bins, labels

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        # [FUNCTION DOC]
        # HELPER: extends the base grouper setup. Inputs: obj, sort (default false), gpr_index
        # (keyword-only; precomputed index to group on, or none). After the base setup, if
        # the axis dtype is an Arrow date or time-delta type, records that dtype on this
        # grouper and replaces the axis by the equivalent numpy-backed index so binning
        # works. Returns (object, axis, sort indexer or none).
        obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
        if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in "Mm":
            self._arrow_dtype = ax.dtype
            ax = Index(
                cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array()
            )
        return obj, ax, indexer


@overload
def _take_new_index(
    obj: DataFrame, indexer: npt.NDArray[np.intp], new_index: Index
) -> DataFrame:
    # [FUNCTION DOC]
    # Typing overload only: a DataFrame input gives a DataFrame. See the final
    # implementation of _take_new_index.
    ...


@overload
def _take_new_index(
    obj: Series, indexer: npt.NDArray[np.intp], new_index: Index
) -> Series:
    # [FUNCTION DOC]
    # Typing overload only: a Series input gives a Series. See the final implementation
    # of _take_new_index.
    ...


def _take_new_index(
    obj: DataFrame | Series,
    indexer: npt.NDArray[np.intp],
    new_index: Index,
) -> DataFrame | Series:
    # [FUNCTION DOC]
    # HELPER: for period upsampling; reindexes an object by a positional indexer and
    # attaches a new index. Inputs: obj (Series or DataFrame), indexer (integer positions,
    # minus one meaning missing), new_index (labels for the result). A Series is rebuilt
    # from its values taken at the indexer (keeping its name); a DataFrame is rebuilt via
    # its block manager's reindex with that indexer. Anything else raises a value error.
    # Returns the new object.
    if isinstance(obj, ABCSeries):
        new_values = algos.take_nd(obj._values, indexer)
        return obj._constructor(new_values, index=new_index, name=obj.name)
    elif isinstance(obj, ABCDataFrame):
        new_mgr = obj._mgr.reindex_indexer(new_axis=new_index, indexer=indexer, axis=1)
        return obj._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
    else:
        raise ValueError("'obj' should be either a Series or a DataFrame")


def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: TimeUnit,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    # [FUNCTION DOC]
    # HELPER: computes the outer bin edges (first and last timestamps) for a time range.
    # Inputs: first and last (timestamps of the data), freq (offset), unit (time unit for
    # the integer arithmetic), closed (left by default), origin (keyword or timestamp;
    # default start_day), offset (optional time-delta added to the origin).
    # For fixed-duration (tick) frequencies: a timestamp origin must have the same
    # time-zone awareness as the data (value error otherwise); the epoch keyword becomes
    # the 1970 timestamp in the data's zone; then the anchored-date helper rolls the
    # endpoints onto the grid. For other (calendar) frequencies: both endpoints are
    # normalized to midnight, the first is rolled back to an offset boundary (or moved back
    # by one frequency step when closed right), and the last is advanced by one step.
    # Returns the adjusted (first, last).
    # KNOWN ISSUE: the docstring lists the origin options as epoch, start and start_day
    # only, but the end and end_day keywords are also supported by the code (via the
    # anchored-date helper).
    """
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'} or Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    """
    if isinstance(freq, Tick):
        index_tz = first.tz
        if isinstance(origin, Timestamp) and (origin.tz is None) != (index_tz is None):
            raise ValueError("The origin must have the same timezone as the index.")
        if origin == "epoch":
            # set the epoch based on the timezone to have similar bins results when
            # resampling on the same kind of indexes on different timezones
            origin = Timestamp("1970-01-01", tz=index_tz)

        first, last = _adjust_dates_anchored(
            first,
            last,
            freq,
            closed=closed,
            origin=origin,
            offset=offset,
            unit=unit,
        )
    else:
        first = first.normalize()
        last = last.normalize()

        if closed == "left":
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp(first - freq)

        last = Timestamp(last + freq)

    return first, last


def _get_period_range_edges(
    first: Period,
    last: Period,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
) -> tuple[Period, Period]:
    # [FUNCTION DOC]
    # HELPER: computes outer bin edges as periods for period resampling.
    # Inputs: first and last (must both be periods, else a type error), freq, closed
    # (default left), origin (default start_day), offset (optional).
    # Converts both periods to timestamps, runs the timestamp edge routine (nanosecond
    # unit), then converts back to periods: the first is advanced by one frequency step if
    # the original first timestamp was not on the offset, and the last is moved back by
    # one step if the original last timestamp was on the offset.
    # Returns the adjusted (first, last) periods.
    # KNOWN ISSUE: as in the timestamp version, the docstring omits the end and end_day
    # origin keywords that the code supports.
    """
    Adjust the provided `first` and `last` Periods to the respective Period of
    the given offset that encompasses them.

    Parameters
    ----------
    first : pd.Period
        The beginning Period of the range to be adjusted.
    last : pd.Period
        The ending Period of the range to be adjusted.
    freq : pd.DateOffset
        The freq to which the Periods will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'}, Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.

        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Period objects.
    """
    if not all(isinstance(obj, Period) for obj in [first, last]):
        raise TypeError("'first' and 'last' must be instances of type Period")

    # GH 23882
    first_ts = first.to_timestamp()
    last_ts = last.to_timestamp()
    adjust_first = not freq.is_on_offset(first_ts)
    adjust_last = freq.is_on_offset(last_ts)

    first_ts, last_ts = _get_timestamp_range_edges(
        first_ts, last_ts, freq, unit="ns", closed=closed, origin=origin, offset=offset
    )

    first = (first_ts + int(adjust_first) * freq).to_period(freq)
    last = (last_ts - int(adjust_last) * freq).to_period(freq)
    return first, last


def _insert_nat_bin(
    binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int
) -> tuple[PeriodIndex, np.ndarray, PeriodIndex]:
    # [FUNCTION DOC]
    # HELPER: for period bin computation; adds a leading bin for missing values.
    # Inputs: binner and labels (period indexes), bins (integer bin end positions), nat_count
    # (number of missing values; asserted positive). Shifts all bin positions up by
    # nat_count (modifying the passed bins array in place), puts nat_count at the front as
    # the first bin end, and inserts a missing value at the front of both the binner and the
    # labels. Returns (binner, bins, labels).
    # NaT handling as in pandas._lib.lib.generate_bins_dt64()
    # shift bins by the number of NaT
    assert nat_count > 0
    bins += nat_count
    bins = np.insert(bins, 0, nat_count)

    # Incompatible types in assignment (expression has type "Index", variable
    # has type "PeriodIndex")
    binner = binner.insert(0, NaT)  # type: ignore[assignment]
    # Incompatible types in assignment (expression has type "Index", variable
    # has type "PeriodIndex")
    labels = labels.insert(0, NaT)  # type: ignore[assignment]
    return binner, bins, labels


def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",
    origin: TimeGrouperOrigin = "start_day",
    offset: Timedelta | None = None,
    unit: TimeUnit = "ns",
) -> tuple[Timestamp, Timestamp]:
    # [FUNCTION DOC]
    # HELPER: for fixed-duration frequencies; rolls the first and last timestamps onto the
    # bin grid anchored at an origin.
    # Inputs: first, last, freq (a tick offset), closed (right by default in this
    # function), origin (start_day default, start, epoch-equivalent zero, a timestamp, or
    # end / end_day), offset (optional time-delta added to the origin), unit (time unit for
    # integer arithmetic).
    # Origin: start_day is midnight of the first day; start is the first value; a timestamp
    # is used directly; for the end keywords the origin is the last value (end) or the end
    # of its day, and the first timestamp is moved to a whole number of steps back from it
    # (one more step when closed left); any other value leaves the origin at zero (epoch).
    # The offset is added to the origin. Computation is done in UTC for zone-aware
    # inputs to avoid ambiguous or nonexistent local times.
    # Rolling: when closed right the first moves back to the grid point at or before it
    # (one full step back if already on the grid) and the last moves forward to the grid
    # point at or after it; when closed left the first moves back to the grid point at or
    # before it and the last moves forward to the next grid point (one full step if already
    # on the grid). Results are converted back to the original time zone.
    # Returns (adjusted first, adjusted last).
    # First and last offsets should be calculated from the start day to fix an
    # error cause by resampling across multiple days when a one day period is
    # not a multiple of the frequency. See GH 8683
    # To handle frequencies that are not multiple or divisible by a day we let
    # the possibility to define a fixed origin timestamp. See GH 31809
    first = first.as_unit(unit)
    last = last.as_unit(unit)
    if offset is not None:
        offset = offset.as_unit(unit)

    freq_value = Timedelta(freq).as_unit(unit)._value

    origin_timestamp = 0  # origin == "epoch"
    if origin == "start_day":
        origin_timestamp = first.normalize()._value
    elif origin == "start":
        origin_timestamp = first._value
    elif isinstance(origin, Timestamp):
        origin_timestamp = origin.as_unit(unit)._value
    elif origin in ["end", "end_day"]:
        origin_last = last if origin == "end" else last.ceil("D")
        sub_freq_times = (origin_last._value - first._value) // freq_value
        if closed == "left":
            sub_freq_times += 1
        first = origin_last - sub_freq_times * freq
        origin_timestamp = first._value
    origin_timestamp += offset._value if offset else 0

    # GH 10117 & GH 19375. If first and last contain timezone information,
    # Perform the calculation in UTC in order to avoid localizing on an
    # Ambiguous or Nonexistent time.
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert("UTC")
    if last_tzinfo is not None:
        last = last.tz_convert("UTC")

    foffset = (first._value - origin_timestamp) % freq_value
    loffset = (last._value - origin_timestamp) % freq_value

    if closed == "right":
        if foffset > 0:
            # roll back
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value - freq_value

        if loffset > 0:
            # roll forward
            lresult_int = last._value + (freq_value - loffset)
        else:
            # already the end of the road
            lresult_int = last._value
    else:  # closed == 'left'
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            # start of the road
            fresult_int = first._value

        if loffset > 0:
            # roll forward
            lresult_int = last._value + (freq_value - loffset)
        else:
            lresult_int = last._value + freq_value
    fresult = Timestamp(fresult_int, unit=unit)
    lresult = Timestamp(lresult_int, unit=unit)
    if first_tzinfo is not None:
        fresult = fresult.tz_localize("UTC").tz_convert(first_tzinfo)
    if last_tzinfo is not None:
        lresult = lresult.tz_localize("UTC").tz_convert(last_tzinfo)
    return fresult, lresult


def asfreq(
    obj: NDFrameT,
    freq,
    method=None,
    how=None,
    normalize: bool = False,
    fill_value=None,
) -> NDFrameT:
    # [FUNCTION DOC]
    # Converts a Series or DataFrame to a new frequency (implementation behind the as-freq
    # methods of both).
    # Inputs: obj; freq (offset or alias); method (fill method for new rows; not supported
    # for a period index, which raises a not-implemented error); how (start or end for
    # period indexes; defaults to end; unused otherwise); normalize (for other indexes,
    # normalize the resulting index to midnight); fill_value (value for new rows).
    # Period index: copies the object and converts the index to the frequency (an offset
    # frequency is first translated to its period frequency string when it has a period
    # code). Empty index: copies and gives the empty index the frequency. Otherwise: builds a
    # date range from the minimum to the maximum index value at the frequency (in the
    # index's time unit if a datetime index, else nanoseconds), names it like the index,
    # reindexes the object onto it with method and fill value, and optionally normalizes.
    # Returns the new object; the input is not modified.
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
    if isinstance(obj.index, PeriodIndex):
        if method is not None:
            raise NotImplementedError("'method' argument is not supported")

        if how is None:
            how = "E"

        if isinstance(freq, BaseOffset):
            if hasattr(freq, "_period_dtype_code"):
                freq = PeriodDtype(freq)._freqstr

        new_obj = obj.copy()
        new_obj.index = obj.index.asfreq(freq, how=how)

    elif len(obj.index) == 0:
        new_obj = obj.copy()

        new_obj.index = _asfreq_compat(obj.index, freq)
    else:
        unit: TimeUnit = "ns"
        if isinstance(obj.index, DatetimeIndex):
            # TODO: should we disallow non-DatetimeIndex?
            unit = obj.index.unit
        dti = date_range(obj.index.min(), obj.index.max(), freq=freq, unit=unit)
        dti.name = obj.index.name
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        if normalize:
            new_obj.index = new_obj.index.normalize()

    return new_obj


def _asfreq_compat(index: FreqIndexT, freq) -> FreqIndexT:
    # [FUNCTION DOC]
    # HELPER: mimics the as-frequency conversion for an empty index. Inputs: index (empty
    # period, datetime or time-delta index) and freq. A non-empty index raises a value error
    # (the caller is supposed to check). A period index is converted to the frequency; an
    # empty datetime or time-delta index of the same dtype and name with the given
    # frequency is created; any other type raises a type error. Returns the new index.
    """
    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.

    Parameters
    ----------
    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex
    freq : DateOffset

    Returns
    -------
    same type as index
    """
    if len(index) != 0:
        # This should never be reached, always checked by the caller
        raise ValueError(
            "Can only set arbitrary freq for empty DatetimeIndex or TimedeltaIndex"
        )
    if isinstance(index, PeriodIndex):
        new_index = index.asfreq(freq=freq)
    elif isinstance(index, DatetimeIndex):
        new_index = DatetimeIndex([], dtype=index.dtype, freq=freq, name=index.name)
    elif isinstance(index, TimedeltaIndex):
        new_index = TimedeltaIndex([], dtype=index.dtype, freq=freq, name=index.name)
    else:  # pragma: no cover
        raise TypeError(type(index))
    return new_index
