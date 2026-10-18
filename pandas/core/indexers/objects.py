"""Indexer objects for computing start/end window bounds for rolling operations"""

# [MODULE DOC]
# Window-bound calculators used by rolling, expanding, exponentially weighted
# and groupby-rolling operations. Each indexer exposes a method that, given the
# number of values and options passed from the top-level window API, returns a
# pair of integer arrays (start inclusive, end exclusive) with one entry per
# output window. Contents: a base class (public, user-subclassable), fixed,
# variable (time-based, using a compiled routine), variable-offset (public),
# expanding, fixed-forward (public), groupby and exponentially-weighted indexers.
# Public ones are published under the public indexers api module name.
# Relationships: used by core.window modules; depends on the compiled window
# indexer routines, datetime index and the nanosecond offset.
# No import-time side effects.
from __future__ import annotations

from datetime import timedelta

import numpy as np

from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import set_module

from pandas.core.dtypes.common import ensure_platform_int

from pandas.core.indexes.datetimes import DatetimeIndex

from pandas.tseries.offsets import Nano


@set_module("pandas.api.indexers")
class BaseIndexer:
    # [CLASS DOC]
    # Abstract base for all window indexers and the extension point for user
    # defined windows. State: an optional index array of data point labels
    # and an integer window size, plus any extra keyword arguments stored as
    # attributes of the same name. Subclasses must override the window bounds
    # method, which here always raises not-implemented.
    """
    Base class for window bounds calculations.

    Parameters
    ----------
    index_array : np.ndarray, default None
        Array-like structure representing the indices for the data points.
        If None, the default indices are assumed. This can be useful for
        handling non-uniform indices in data, such as in time series
        with irregular timestamps.
    window_size : int, default 0
        Size of the moving window. This is the number of observations used
        for calculating the statistic. The default is to consider all
        observations within the window.
    **kwargs
        Additional keyword arguments passed to the subclass's methods.

    See Also
    --------
    DataFrame.rolling : Provides rolling window calculations on dataframe.
    Series.rolling : Provides rolling window calculations on series.

    Examples
    --------
    >>> from pandas.api.indexers import BaseIndexer
    >>> class CustomIndexer(BaseIndexer):
    ...     def get_window_bounds(self, num_values, min_periods, center, closed, step):
    ...         start = np.arange(num_values, dtype=np.int64)
    ...         end = np.arange(num_values, dtype=np.int64) + self.window_size
    ...         return start, end
    >>> df = pd.DataFrame({"values": range(5)})
    >>> indexer = CustomIndexer(window_size=2)
    >>> df.rolling(indexer).sum()
        values
    0	1.0
    1	3.0
    2	5.0
    3	7.0
    4	4.0
    """

    def __init__(
        self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Stores index_array (optional array of labels, default none) and
        # window_size (integer, default zero) as attributes, then sets every
        # extra keyword argument as an attribute of the same name on the
        # instance (overwriting any existing attribute, no validation).
        self.index_array = index_array
        self.window_size = window_size
        # Set user defined kwargs as attributes that can be used in get_window_bounds
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Abstract hook computing window bounds. Parameters (all optional):
        # num_values is the number of observations, min_periods, center,
        # closed and step are forwarded from the top-level window API.
        # Returns a pair of int64 arrays (starts inclusive, ends exclusive).
        # This base version always raises a not-implemented error.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature (window size is an
        # instance attribute).
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        raise NotImplementedError


class FixedWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # Indexer for windows of constant length equal to the window size
    # attribute, each ending at its own row (or shifted for centering).
    # Subclass of the base indexer; only overrides the window bounds method.
    """Creates window boundaries that are of fixed length."""

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Fixed-length windows. num_values is the observation count; center
        # shifts windows so they are centered (also applied when the window
        # size is zero); closed is one of right (default when none), left, both
        # or neither; step (none means one) subsamples the window ends;
        # min_periods is ignored. Ends run from one (plus the centering
        # offset, half the window size rounded down minus a half) up to the
        # count (plus offset); starts are ends minus the window size. Closed
        # left or both moves starts one earlier; left or neither moves ends
        # one earlier. Both arrays are then clipped into the range zero to
        # num_values. Returns (starts, ends) as int64 arrays.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        if center or self.window_size == 0:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0

        end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
        start = end - self.window_size
        if closed in ["left", "both"]:
            start -= 1
        if closed in ["left", "neither"]:
            end -= 1

        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        return start, end


class VariableWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # Indexer for windows whose row count varies, used for time-based rolling.
    # Relies on the index array attribute (the sorted labels, as integers) and
    # on the window size attribute (span in label units). Subclass of the base
    # indexer; delegates the computation to a compiled routine.
    """Creates window boundaries that are of variable length, namely for time series."""

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Computes label-span windows. Requires the index array attribute to be
        # set (an assertion fails otherwise). Raises a value error if the index
        # array is shorter than num_values. Passes num_values, window size,
        # min_periods, center, closed and the index array to the compiled
        # variable-window routine and returns its (starts, ends) int64 arrays.
        # step is accepted but ignored.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        assert self.index_array is not None
        if (index_length := len(self.index_array)) < num_values:
            raise ValueError(
                "Variable rolling window requires the index to be at least as long "
                f"as the 'other' index. Got {index_length} < {num_values}. "
                "Please align 'other' to the rolling object's index using "
                "reindex_like() or similar method."
            )
        # error: Argument 4 to "calculate_variable_window_bounds" has incompatible
        # type "Optional[bool]"; expected "bool"
        return calculate_variable_window_bounds(
            num_values,
            self.window_size,
            min_periods,
            center,  # type: ignore[arg-type]
            closed,
            self.index_array,
        )


@set_module("pandas.api.indexers")
class VariableOffsetWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # Public indexer for windows whose width is a calendar offset (such as a
    # business day) applied to a datetime index. State: the datetime index of
    # observation labels and the offset, in addition to the inherited attributes
    # (the inherited index array and window size are not used by the bounds
    # computation). Subclass of the base indexer.
    """
    Calculate window boundaries based on a non-fixed offset such as a BusinessDay.

    Parameters
    ----------
    index_array : np.ndarray, default 0
        Array-like structure specifying the indices for data points.
        This parameter is currently not used.

    window_size : int, optional, default 0
        Specifies the number of data points in each window.
        This parameter is currently not used.

    index : DatetimeIndex, optional
        ``DatetimeIndex`` of the labels of each observation.

    offset : BaseOffset, optional
        ``DateOffset`` representing the size of the window.

    **kwargs
        Additional keyword arguments passed to the parent class ``BaseIndexer``.

    See Also
    --------
    api.indexers.BaseIndexer : Base class for all indexers.
    DataFrame.rolling : Rolling window calculations on DataFrames.
    offsets : Module providing various time offset classes.

    Examples
    --------
    >>> from pandas.api.indexers import VariableOffsetWindowIndexer
    >>> df = pd.DataFrame(range(10), index=pd.date_range("2020", periods=10))
    >>> offset = pd.offsets.BDay(1)
    >>> indexer = VariableOffsetWindowIndexer(index=df.index, offset=offset)
    >>> df
                0
    2020-01-01  0
    2020-01-02  1
    2020-01-03  2
    2020-01-04  3
    2020-01-05  4
    2020-01-06  5
    2020-01-07  6
    2020-01-08  7
    2020-01-09  8
    2020-01-10  9
    >>> df.rolling(indexer).sum()
                   0
    2020-01-01   0.0
    2020-01-02   1.0
    2020-01-03   2.0
    2020-01-04   3.0
    2020-01-05   7.0
    2020-01-06  12.0
    2020-01-07   6.0
    2020-01-08   7.0
    2020-01-09   8.0
    2020-01-10   9.0
    """

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Initializes the base attributes from index_array, window_size and
        # extra keyword arguments, then validates and stores index (must be a
        # datetime index) and offset (must be an offset object). Raises a
        # value error if either is missing or of the wrong type (both default
        # to none, so both are effectively required).
        super().__init__(index_array, window_size, **kwargs)
        if not isinstance(index, DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex.")
        self.index = index
        if not isinstance(offset, BaseOffset):
            raise ValueError("offset must be a DateOffset-like object.")
        self.offset = offset

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Offset-based windows over the stored datetime index. Raises a
        # not-implemented error if step is given. With no values, returns two
        # empty int64 arrays. closed defaults to right; min_periods and
        # center are ignored. The offset sign is flipped when the index
        # decreases from first to last element. For each position the window
        # end is the position itself (inclusive when the right side is closed,
        # exclusive otherwise) and the window start is the first earlier
        # position whose label lies strictly after the label minus the
        # offset (minus one nanosecond when the left side is closed). The
        # scan is quadratic in the worst case. Returns (starts, ends) int64 arrays.
        # KNOWN ISSUE: the default for closed is chosen by checking whether
        # the stored index is present, which is always true here, so the
        # alternative default of both is unreachable.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        if step is not None:
            raise NotImplementedError("step not implemented for variable offset window")
        if num_values <= 0:
            return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

        # if windows is variable, default is 'right', otherwise default is 'both'
        if closed is None:
            closed = "right" if self.index is not None else "both"

        right_closed = closed in ["right", "both"]
        left_closed = closed in ["left", "both"]

        if self.index[num_values - 1] < self.index[0]:
            index_growth_sign = -1
        else:
            index_growth_sign = 1
        offset_diff = index_growth_sign * self.offset

        start = np.empty(num_values, dtype="int64")
        start.fill(-1)
        end = np.empty(num_values, dtype="int64")
        end.fill(-1)

        start[0] = 0

        # right endpoint is closed
        if right_closed:
            end[0] = 1
        # right endpoint is open
        else:
            end[0] = 0

        zero = timedelta(0)
        # start is start of slice interval (including)
        # end is end of slice interval (not including)
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = end_bound - offset_diff

            # left endpoint is closed
            if left_closed:
                start_bound -= Nano(1)

            # advance the start bound until we are
            # within the constraint
            start[i] = i
            for j in range(start[i - 1], i):
                start_diff = (self.index[j] - start_bound) * index_growth_sign
                if start_diff > zero:
                    start[i] = j
                    break

            # end bound is previous end
            # or current index
            end_diff = (self.index[end[i - 1]] - end_bound) * index_growth_sign
            if end_diff == zero and not right_closed:
                end[i] = end[i - 1] + 1
            elif end_diff <= zero:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # right endpoint is open
            if not right_closed:
                end[i] -= 1

        return start, end


class ExpandingIndexer(BaseIndexer):
    # [CLASS DOC]
    # Indexer for expanding windows, where every window starts at the first
    # row and ends at its own row. Subclass of the base indexer with no
    # additional state.
    """Calculate expanding window bounds, mimicking df.expanding()"""

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Returns (starts, ends) int64 arrays of length num_values: starts are
        # all zero, ends count from one to num_values inclusive. All other
        # arguments (min_periods, center, closed, step) are ignored.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        return (
            np.zeros(num_values, dtype=np.int64),
            np.arange(1, num_values + 1, dtype=np.int64),
        )


@set_module("pandas.api.indexers")
class FixedForwardWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # Public indexer for forward-looking fixed-length windows: each window
    # starts at its own row and extends forward by the window size attribute.
    # Subclass of the base indexer with no additional state.
    """
    Creates window boundaries for fixed-length windows that include the current row.

    Parameters
    ----------
    index_array : np.ndarray, default None
        Array-like structure representing the indices for the data points.
        If None, the default indices are assumed. This can be useful for
        handling non-uniform indices in data, such as in time series
        with irregular timestamps.
    window_size : int, default 0
        Size of the moving window. This is the number of observations used
        for calculating the statistic. The default is to consider all
        observations within the window.
    **kwargs
        Additional keyword arguments passed to the subclass's methods.

    See Also
    --------
    DataFrame.rolling : Provides rolling window calculations.
    api.indexers.VariableWindowIndexer : Calculate window bounds based on
        variable-sized windows.

    Examples
    --------
    >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0
    """

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Forward-looking windows. Raises a value error if center is truthy or
        # if closed is given. step (none means one) subsamples the window
        # starts; min_periods is ignored. Starts are positions from zero up to
        # num_values by step; ends are starts plus the window size, clipped
        # to the range zero to num_values only when the window size is
        # nonzero (with a zero window size, ends equal starts). Returns
        # (starts, ends) int64 arrays.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        if closed is not None:
            raise ValueError(
                "Forward-looking windows don't support setting the closed argument"
            )
        if step is None:
            step = 1

        start = np.arange(0, num_values, step, dtype="int64")
        end = start + self.window_size
        if self.window_size:
            end = np.clip(end, 0, num_values)

        return start, end


class GroupbyIndexer(BaseIndexer):
    # [CLASS DOC]
    # Indexer for rolling over groups: applies an inner window indexer class
    # separately to each group's rows and concatenates the bounds in group
    # order. State: a mapping from group label to positional row indices,
    # the inner indexer class, and extra keyword arguments for it, in
    # addition to the inherited attributes. Subclass of the base indexer.
    """Calculate bounds to compute groupby rolling, mimicking df.groupby().rolling()"""

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int | BaseIndexer = 0,
        groupby_indices: dict | None = None,
        window_indexer: type[BaseIndexer] = BaseIndexer,
        indexer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Stores groupby_indices (mapping of group label to row positions;
        # none becomes an empty mapping), window_indexer (inner indexer class,
        # default the base class) and a shallow copy of indexer_kwargs (none
        # becomes an empty mapping). A window_size entry in the copied kwargs
        # is removed and takes precedence over the window_size argument for
        # the base initializer, so the inner indexer receives window size
        # separately. Extra keyword arguments become attributes via the base.
        """
        Parameters
        ----------
        index_array : np.ndarray or None
            np.ndarray of the index of the original object that we are performing
            a chained groupby operation over. This index has been pre-sorted relative to
            the groups
        window_size : int or BaseIndexer
            window size during the windowing operation
        groupby_indices : dict or None
            dict of {group label: [positional index of rows belonging to the group]}
        window_indexer : BaseIndexer
            BaseIndexer class determining the start and end bounds of each group
        indexer_kwargs : dict or None
            Custom kwargs to be passed to window_indexer
        **kwargs :
            keyword arguments that will be available when get_window_bounds is called
        """
        self.groupby_indices = groupby_indices or {}
        self.window_indexer = window_indexer
        self.indexer_kwargs = indexer_kwargs.copy() if indexer_kwargs else {}
        super().__init__(
            index_array=index_array,
            window_size=self.indexer_kwargs.pop("window_size", window_size),
            **kwargs,
        )

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Per-group window bounds. For each group (in mapping order) it
        # builds an inner indexer from the stored class with the group's
        # slice of the index array (if any), the window size and stored
        # keyword arguments, and asks it for bounds over that group's row
        # count using the given min_periods, center, closed and step. The
        # group's bounds are converted to positions in the concatenated,
        # group-ordered result by offsetting by the cumulative count of earlier
        # rows (with one extra sentinel position so an end equal to the group
        # length maps correctly). Asserts that start and end lengths match.
        # Returns (starts, ends) int64 arrays concatenated across groups, or two
        # empty int64 arrays when there are no groups.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        # 1) For each group, get the indices that belong to the group
        # 2) Use the indices to calculate the start & end bounds of the window
        # 3) Append the window bounds in group order
        start_arrays = []
        end_arrays = []
        window_indices_start = 0
        for indices in self.groupby_indices.values():
            index_array: np.ndarray | None

            if self.index_array is not None:
                index_array = self.index_array.take(ensure_platform_int(indices))
            else:
                index_array = self.index_array
            indexer = self.window_indexer(
                index_array=index_array,
                window_size=self.window_size,
                **self.indexer_kwargs,
            )
            start, end = indexer.get_window_bounds(
                len(indices), min_periods, center, closed, step
            )
            start = start.astype(np.int64)
            end = end.astype(np.int64)
            assert len(start) == len(end), (
                "these should be equal in length from get_window_bounds"
            )
            # Cannot use groupby_indices as they might not be monotonic with the object
            # we're rolling over
            window_indices = np.arange(
                window_indices_start, window_indices_start + len(indices)
            )
            window_indices_start += len(indices)
            # Extend as we'll be slicing window like [start, end)
            window_indices = np.append(window_indices, [window_indices[-1] + 1]).astype(
                np.int64, copy=False
            )
            start_arrays.append(window_indices.take(ensure_platform_int(start)))
            end_arrays.append(window_indices.take(ensure_platform_int(end)))
        if len(start_arrays) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        start = np.concatenate(start_arrays)
        end = np.concatenate(end_arrays)
        return start, end


class ExponentialMovingWindowIndexer(BaseIndexer):
    # [CLASS DOC]
    # Indexer for exponentially weighted operations, which use the entire
    # series as one window. Subclass of the base indexer with no extra state.
    """Calculate ewm window bounds (the entire window)"""

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Returns a single window covering everything: a one-element start
        # array containing zero and a one-element end array containing
        # num_values, both int64. All other arguments are ignored.
        # KNOWN ISSUE: the docstring documents window_size and win_type
        # parameters that are not in the signature.
        """
        Computes the bounds of a window.

        Parameters
        ----------
        num_values : int, default 0
            number of values that will be aggregated over
        window_size : int, default 0
            the number of rows in a window
        min_periods : int, default None
            min_periods passed from the top level rolling API
        center : bool, default None
            center passed from the top level rolling API
        closed : str, default None
            closed passed from the top level rolling API
        step : int, default None
            step passed from the top level rolling API
        win_type : str, default None
            win_type passed from the top level rolling API

        Returns
        -------
        A tuple of ndarray[int64]s, indicating the boundaries of each
        window
        """
        return np.array([0], dtype=np.int64), np.array([num_values], dtype=np.int64)
